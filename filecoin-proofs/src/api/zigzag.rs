//! filecoin-proofs level API for ZigZag layered PoRep.
//!
//! This is the ZigZag analogue of the Stacked DRG seal API in [`crate::api::seal`]. ZigZag has a
//! much simpler replication model than Stacked (a single in-memory layered encoding, with no
//! on-disk labels / `p_aux` / `t_aux`), so the flow here is correspondingly smaller:
//!
//! * [`zigzag_pre_commit`] replicates the data in place and returns the commitments plus the
//!   in-memory prover state (per-layer replica trees) needed to build a SNARK.
//! * [`zigzag_prove`] turns that prover state into a Groth16 proof.
//! * [`zigzag_verify_seal`] verifies such a proof.
//! * [`zigzag_unseal`] recovers the original data (this is ZigZag's fast, parallel extraction).
//!
//! ZigZag uses its own Groth16 parameters (distinct cache id, see
//! [`storage_proofs_porep::zigzag::circuit::ZigZagCompound`]) and a binary Poseidon Merkle tree,
//! so it coexists with the Stacked path without sharing parameters.
//!
//! Challenge derivation follows the original ZigZag: challenges are derived from `replica_id` and
//! `comm_r_star` (no separately supplied seed), so verification does not need a chain-provided
//! challenge seed.

use anyhow::{ensure, Result};
use filecoin_hashers::Hasher;
use storage_proofs_core::{
    compound_proof::{self, CompoundProof},
    merkle::{create_base_merkle_tree, MerkleTreeTrait},
    multi_proof::MultiProof,
    sector::SectorId,
    util::NODE_SIZE,
};
use storage_proofs_porep::{
    stacked::generate_replica_id,
    zigzag::{
        circuit::ZigZagCompound, ChallengeRequirements, LayerTau, PrivateInputs, PublicInputs, Tau,
        ZigZagDrgPoRep,
    },
};

use crate::{
    api::{as_safe_commitment, commitment_from_fr},
    caches::{get_zigzag_params, get_zigzag_verifying_key},
    parameters::{zigzag_public_params, zigzag_setup_params},
    types::{Commitment, PoRepConfig, ProverId, SealCommitOutput, Ticket},
};

type TreeDomain<Tree> = <<Tree as MerkleTreeTrait>::Hasher as Hasher>::Domain;

/// The public commitments produced by ZigZag replication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZigZagPreCommitOutput {
    /// Commitment to the original (unsealed) data: the Poseidon root of layer 0.
    pub comm_d: Commitment,
    /// Commitment to the sealed replica: the Poseidon root of the final layer.
    pub comm_r: Commitment,
    /// Aggregate commitment folding `replica_id` and every per-layer replica root.
    pub comm_r_star: Commitment,
}

/// In-memory prover state carried from [`zigzag_pre_commit`] to [`zigzag_prove`].
///
/// Holds the per-layer replica trees and taus produced during replication. This is analogous to
/// Stacked's `p_aux`/`t_aux`, but is kept in memory rather than persisted to disk.
pub struct ZigZagProverState<Tree: MerkleTreeTrait> {
    replica_id: TreeDomain<Tree>,
    tau: Tau<TreeDomain<Tree>>,
    trees: Vec<Tree>,
}

/// Replicate `data` in place using ZigZag layered encoding and return the commitments plus the
/// in-memory prover state required to produce a SNARK proof.
///
/// `data` must be exactly the padded sector size and a multiple of the node size (32 bytes). On
/// return, `data` contains the sealed replica.
pub fn zigzag_pre_commit<Tree: 'static + MerkleTreeTrait>(
    porep_config: &PoRepConfig,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    data: &mut [u8],
) -> Result<(ZigZagPreCommitOutput, ZigZagProverState<Tree>)> {
    ensure!(
        data.len() % NODE_SIZE == 0,
        "data length ({}) must be a multiple of the node size ({})",
        data.len(),
        NODE_SIZE,
    );

    let pub_params = zigzag_public_params::<Tree>(porep_config)?;
    let leaves = data.len() / NODE_SIZE;

    // `replica_id` depends on `comm_d`, so we must commit to the original data before replicating.
    // Replication rebuilds this same layer-0 tree; the roots are identical.
    let comm_d_domain = create_base_merkle_tree::<Tree>(None, leaves, data)?.root();
    let comm_d = commitment_from_fr(comm_d_domain.into());

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d,
        &porep_config.porep_id,
    );

    let (tau, trees) = ZigZagDrgPoRep::<Tree>::transform_and_replicate_layers(
        &pub_params.graph,
        &pub_params.layer_challenges,
        &replica_id,
        data,
    )?;

    let simplified = tau.simplify();
    let out = ZigZagPreCommitOutput {
        comm_d: commitment_from_fr(simplified.comm_d.into()),
        comm_r: commitment_from_fr(simplified.comm_r.into()),
        comm_r_star: commitment_from_fr(tau.comm_r_star.into()),
    };

    let state = ZigZagProverState {
        replica_id,
        tau,
        trees,
    };

    Ok((out, state))
}

/// Produce a Groth16 proof for a previously replicated sector.
pub fn zigzag_prove<Tree: 'static + MerkleTreeTrait>(
    porep_config: &PoRepConfig,
    state: ZigZagProverState<Tree>,
) -> Result<SealCommitOutput> {
    let ZigZagProverState {
        replica_id,
        tau,
        trees,
    } = state;

    let compound_setup_params = compound_proof::SetupParams {
        vanilla_params: zigzag_setup_params(porep_config)?,
        partitions: Some(usize::from(porep_config.partitions)),
        priority: false,
    };
    let compound_public_params = ZigZagCompound::<Tree>::setup(&compound_setup_params)?;

    let public_inputs = PublicInputs::<TreeDomain<Tree>> {
        replica_id,
        seed: None,
        tau: Some(tau.simplify()),
        comm_r_star: tau.comm_r_star,
        k: None,
    };
    let private_inputs = PrivateInputs::<Tree> {
        aux: trees,
        tau: tau.layer_taus.clone(),
    };

    let groth_params = get_zigzag_params::<Tree>(porep_config)?;
    let proofs = ZigZagCompound::prove(
        &compound_public_params,
        &public_inputs,
        &private_inputs,
        &groth_params,
    )?;

    let verifying_key = get_zigzag_verifying_key::<Tree>(porep_config)?;
    let proof = MultiProof::new(proofs, &verifying_key);

    let mut buf = Vec::new();
    proof.write(&mut buf)?;

    Ok(SealCommitOutput { proof: buf })
}

/// Verify a ZigZag seal proof against the given commitments.
#[allow(clippy::too_many_arguments)]
pub fn zigzag_verify_seal<Tree: 'static + MerkleTreeTrait>(
    porep_config: &PoRepConfig,
    comm_r_in: Commitment,
    comm_d_in: Commitment,
    comm_r_star_in: Commitment,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    proof_vec: &[u8],
) -> Result<bool> {
    ensure!(comm_d_in != [0; 32], "Invalid all zero commitment (comm_d)");
    ensure!(comm_r_in != [0; 32], "Invalid all zero commitment (comm_r)");
    ensure!(
        comm_r_star_in != [0; 32],
        "Invalid all zero commitment (comm_r_star)"
    );
    ensure!(!proof_vec.is_empty(), "Invalid proof bytes (empty vector)");

    let comm_r: TreeDomain<Tree> = as_safe_commitment(&comm_r_in, "comm_r")?;
    let comm_d: TreeDomain<Tree> = as_safe_commitment(&comm_d_in, "comm_d")?;
    let comm_r_star: TreeDomain<Tree> = as_safe_commitment(&comm_r_star_in, "comm_r_star")?;

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d_in,
        &porep_config.porep_id,
    );

    let compound_setup_params = compound_proof::SetupParams {
        vanilla_params: zigzag_setup_params(porep_config)?,
        partitions: Some(usize::from(porep_config.partitions)),
        priority: false,
    };
    let compound_public_params = ZigZagCompound::<Tree>::setup(&compound_setup_params)?;

    let public_inputs = PublicInputs::<TreeDomain<Tree>> {
        replica_id,
        seed: None,
        tau: Some(LayerTau { comm_d, comm_r }),
        comm_r_star,
        k: None,
    };

    let verifying_key = get_zigzag_verifying_key::<Tree>(porep_config)?;
    let proof = MultiProof::new_from_reader(
        Some(usize::from(porep_config.partitions)),
        proof_vec,
        &verifying_key,
    )?;

    ZigZagCompound::verify(
        &compound_public_params,
        &public_inputs,
        &proof,
        &ChallengeRequirements {
            minimum_challenges: porep_config.minimum_challenges(),
        },
    )
}

/// Recover the original data from a sealed replica, in place.
///
/// This is ZigZag's fast extraction: decoding each layer is embarrassingly parallel (see
/// `storage_proofs_porep::zigzag::decode`), unlike the sequential encoding.
pub fn zigzag_unseal<Tree: 'static + MerkleTreeTrait>(
    porep_config: &PoRepConfig,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    comm_d_in: Commitment,
    data: &mut [u8],
) -> Result<()> {
    ensure!(
        data.len() % NODE_SIZE == 0,
        "data length ({}) must be a multiple of the node size ({})",
        data.len(),
        NODE_SIZE,
    );

    let pub_params = zigzag_public_params::<Tree>(porep_config)?;

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d_in,
        &porep_config.porep_id,
    );

    ZigZagDrgPoRep::<Tree>::extract_and_invert_transform_layers(
        &pub_params.graph,
        &pub_params.layer_challenges,
        &replica_id,
        data,
    )
}
