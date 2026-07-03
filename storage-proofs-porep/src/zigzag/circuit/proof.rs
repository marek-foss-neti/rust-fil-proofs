use bellperson::{
    gadgets::{boolean::Boolean, num::AllocatedNum},
    Circuit, ConstraintSystem, SynthesisError,
};
use blstrs::Scalar as Fr;
use filecoin_hashers::{HashFunction, Hasher};
use storage_proofs_core::{
    compound_proof::CircuitComponent,
    drgraph::Graph,
    gadgets::{constraint, encode as encode_gadget, por::AuthPath, por::PoRCircuit, variables::Root},
    merkle::MerkleTreeTrait,
    util::reverse_bit_numbering,
};

use crate::zigzag::{
    circuit::kdf::kdf,
    vanilla::{Proof, PublicParams},
};

/// The ZigZag layered PoRep circuit.
///
/// Public inputs (in synthesis order): `replica_id`, `comm_d`, `comm_r`, then for each layer and
/// each challenge the packed Merkle-path positions of (data node, replica node, each parent), and
/// finally `comm_r_star`.
pub struct ZigZagCircuit<Tree: MerkleTreeTrait> {
    pub public_params: PublicParams<Tree>,
    pub replica_id: Option<<Tree::Hasher as Hasher>::Domain>,
    pub comm_d: Option<<Tree::Hasher as Hasher>::Domain>,
    pub comm_r: Option<<Tree::Hasher as Hasher>::Domain>,
    pub comm_r_star: Option<<Tree::Hasher as Hasher>::Domain>,
    pub proof: Option<Proof<Tree>>,
}

impl<Tree: MerkleTreeTrait> Clone for ZigZagCircuit<Tree> {
    fn clone(&self) -> Self {
        ZigZagCircuit {
            public_params: self.public_params.clone(),
            replica_id: self.replica_id,
            comm_d: self.comm_d,
            comm_r: self.comm_r,
            comm_r_star: self.comm_r_star,
            proof: self.proof.clone(),
        }
    }
}

impl<Tree: MerkleTreeTrait> CircuitComponent for ZigZagCircuit<Tree> {
    type ComponentPrivateInputs = ();
}

/// Converts an allocated field element into the 256 big-endian-per-byte bits that the SHA256 KDF
/// consumes (matching the vanilla KDF's 32-byte little-endian serialization).
fn num_into_kdf_bits<CS: ConstraintSystem<Fr>>(
    cs: CS,
    num: &AllocatedNum<Fr>,
) -> Result<Vec<Boolean>, SynthesisError> {
    Ok(reverse_bit_numbering(num.to_bits_le(cs)?))
}

impl<Tree: 'static + MerkleTreeTrait> Circuit<Fr> for ZigZagCircuit<Tree> {
    fn synthesize<CS: ConstraintSystem<Fr>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let layers = self.public_params.layer_challenges.layers();
        let leaves = self.public_params.graph.size();
        let degree = self.public_params.graph.degree();

        // Public: replica_id.
        let replica_id_num = AllocatedNum::alloc(cs.namespace(|| "replica_id"), || {
            self.replica_id
                .map(Into::into)
                .ok_or(SynthesisError::AssignmentMissing)
        })?;
        replica_id_num.inputize(cs.namespace(|| "replica_id_input"))?;
        let replica_id_bits =
            num_into_kdf_bits(cs.namespace(|| "replica_id_bits"), &replica_id_num)?;

        // Public: comm_d (data commitment of the very first layer).
        let comm_d_num = AllocatedNum::alloc(cs.namespace(|| "comm_d"), || {
            self.comm_d
                .map(Into::into)
                .ok_or(SynthesisError::AssignmentMissing)
        })?;
        comm_d_num.inputize(cs.namespace(|| "comm_d_input"))?;

        // Public: comm_r (replica commitment of the last layer).
        let comm_r_num = AllocatedNum::alloc(cs.namespace(|| "comm_r"), || {
            self.comm_r
                .map(Into::into)
                .ok_or(SynthesisError::AssignmentMissing)
        })?;
        comm_r_num.inputize(cs.namespace(|| "comm_r_input"))?;

        // Per-layer replica roots, collected for the comm_r_star commitment.
        let mut layer_comm_rs: Vec<AllocatedNum<Fr>> = Vec::with_capacity(layers);

        for layer in 0..layers {
            let is_last_layer = layer == layers - 1;
            let challenge_count = self
                .public_params
                .layer_challenges
                .challenges_for_layer(layer);

            // comm_d for this layer: public comm_d on layer 0, else previous layer's comm_r.
            let comm_d_layer = if layer == 0 {
                comm_d_num.clone()
            } else {
                layer_comm_rs[layer - 1].clone()
            };

            // comm_r for this layer: public comm_r on the last layer, else witnessed from the proof.
            let comm_r_layer = if is_last_layer {
                comm_r_num.clone()
            } else {
                AllocatedNum::alloc(cs.namespace(|| format!("comm_r_layer_{}", layer)), || {
                    self.proof
                        .as_ref()
                        .map(|p| p.tau[layer].comm_r.into())
                        .ok_or(SynthesisError::AssignmentMissing)
                })?
            };
            layer_comm_rs.push(comm_r_layer.clone());

            for c in 0..challenge_count {
                let mut cs = cs.namespace(|| format!("layer_{}_challenge_{}", layer, c));

                let layer_proof = self.proof.as_ref().map(|p| &p.encoding_proofs[layer]);

                // Encoded (replica) node value.
                let replica_value = AllocatedNum::alloc(cs.namespace(|| "replica_value"), || {
                    layer_proof
                        .map(|lp| lp.replica_nodes[c].data.into())
                        .ok_or(SynthesisError::AssignmentMissing)
                })?;

                // Pre-encoding (data) node value.
                let data_value = AllocatedNum::alloc(cs.namespace(|| "data_value"), || {
                    layer_proof
                        .map(|lp| lp.nodes[c].data.into())
                        .ok_or(SynthesisError::AssignmentMissing)
                })?;

                // Parent values (encoded, from the replica tree).
                let mut parent_values = Vec::with_capacity(degree);
                let mut parent_kdf_bits = Vec::with_capacity(degree);
                for p in 0..degree {
                    let parent_num =
                        AllocatedNum::alloc(cs.namespace(|| format!("parent_{}", p)), || {
                            layer_proof
                                .map(|lp| lp.replica_parents[c][p].1.data.into())
                                .ok_or(SynthesisError::AssignmentMissing)
                        })?;
                    let bits = num_into_kdf_bits(
                        cs.namespace(|| format!("parent_{}_bits", p)),
                        &parent_num,
                    )?;
                    parent_kdf_bits.push(bits);
                    parent_values.push(parent_num);
                }

                // KDF: key = SHA256(replica_id | parents...).
                let key = kdf(cs.namespace(|| "kdf"), &replica_id_bits, parent_kdf_bits)?;

                // Encoding relation: decode(replica) == data.
                let decoded =
                    encode_gadget::decode(cs.namespace(|| "decode"), &key, &replica_value)?;
                constraint::equal(&mut cs, || "enforce encoding", &decoded, &data_value);

                // Merkle inclusion: data node in comm_d_layer.
                let data_auth_path = auth_path_for::<Tree>(layer_proof.map(|lp| &lp.nodes[c]), leaves);
                PoRCircuit::<Tree>::synthesize(
                    cs.namespace(|| "data_inclusion"),
                    Root::Var(data_value.clone()),
                    data_auth_path,
                    Root::Var(comm_d_layer.clone()),
                    true,
                )?;

                // Merkle inclusion: replica node in comm_r_layer.
                let replica_auth_path =
                    auth_path_for::<Tree>(layer_proof.map(|lp| &lp.replica_nodes[c]), leaves);
                PoRCircuit::<Tree>::synthesize(
                    cs.namespace(|| "replica_inclusion"),
                    Root::Var(replica_value.clone()),
                    replica_auth_path,
                    Root::Var(comm_r_layer.clone()),
                    true,
                )?;

                // Merkle inclusion: each parent in comm_r_layer.
                for (p, parent_value) in parent_values.into_iter().enumerate() {
                    let parent_auth_path = auth_path_for::<Tree>(
                        layer_proof.map(|lp| &lp.replica_parents[c][p].1),
                        leaves,
                    );
                    PoRCircuit::<Tree>::synthesize(
                        cs.namespace(|| format!("parent_inclusion_{}", p)),
                        Root::Var(parent_value),
                        parent_auth_path,
                        Root::Var(comm_r_layer.clone()),
                        true,
                    )?;
                }
            }
        }

        // comm_r_star = Poseidon-MD(replica_id, comm_r_0, ..., comm_r_{L-1}), matching the vanilla
        // `comm_r_star` which uses `HashFunction::hash_md`.
        let mut md_elements = Vec::with_capacity(layers + 1);
        md_elements.push(replica_id_num);
        md_elements.extend(layer_comm_rs.iter().cloned());
        let computed_comm_r_star = <Tree::Hasher as Hasher>::Function::hash_md_circuit(
            &mut cs.namespace(|| "comm_r_star_md"),
            &md_elements,
        )?;

        let comm_r_star_num = AllocatedNum::alloc(cs.namespace(|| "comm_r_star"), || {
            self.comm_r_star
                .map(Into::into)
                .ok_or(SynthesisError::AssignmentMissing)
        })?;
        constraint::equal(
            cs,
            || "enforce comm_r_star",
            &computed_comm_r_star,
            &comm_r_star_num,
        );
        comm_r_star_num.inputize(cs.namespace(|| "comm_r_star_input"))?;

        Ok(())
    }
}

/// Builds the circuit auth-path for a challenged node, using the vanilla proof when present and a
/// blank (properly-sized) path otherwise (for parameter generation).
fn auth_path_for<Tree: MerkleTreeTrait>(
    data_proof: Option<&storage_proofs_core::por::DataProof<Tree::Proof>>,
    leaves: usize,
) -> AuthPath<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity> {
    use storage_proofs_core::merkle::MerkleProofTrait;

    match data_proof {
        Some(dp) => dp.proof.as_options().into(),
        None => AuthPath::blank(leaves),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bellperson::util_cs::test_cs::TestConstraintSystem;
    use filecoin_hashers::poseidon::{PoseidonDomain, PoseidonHasher};
    use generic_array::typenum::{U0, U2};
    use storage_proofs_core::{
        api_version::ApiVersion,
        drgraph::BASE_DEGREE,
        merkle::{DiskStore, MerkleTreeWrapper},
        proof::ProofScheme,
        util::NODE_SIZE,
    };

    use crate::zigzag::vanilla::{
        LayerChallenges, PrivateInputs, PublicInputs, SetupParams, ZigZagDrgPoRep, EXP_DEGREE,
    };

    type ZZTree = MerkleTreeWrapper<PoseidonHasher, DiskStore<PoseidonDomain>, U2, U0, U0>;

    fn synthesize_satisfied(layers: usize, challenge_count: usize) {
        let nodes = 128;
        let porep_id = [21u8; 32];

        let sp = SetupParams {
            nodes,
            degree: BASE_DEGREE,
            expansion_degree: EXP_DEGREE,
            porep_id,
            api_version: ApiVersion::V1_2_0,
            layer_challenges: LayerChallenges::new_fixed(layers, challenge_count),
        };

        let pp = ZigZagDrgPoRep::<ZZTree>::setup(&sp).expect("setup failed");

        let replica_id = PoseidonDomain::from([19u8; 32]);
        let mut data = vec![0u8; nodes * NODE_SIZE];

        let (tau, trees) = ZigZagDrgPoRep::<ZZTree>::transform_and_replicate_layers(
            &pp.graph,
            &pp.layer_challenges,
            &replica_id,
            &mut data,
        )
        .expect("replication failed");

        let simplified = tau.simplify();

        let pub_inputs = PublicInputs::<PoseidonDomain> {
            replica_id,
            seed: None,
            tau: Some(simplified),
            comm_r_star: tau.comm_r_star,
            k: None,
        };

        let priv_inputs = PrivateInputs::<ZZTree> {
            aux: trees,
            tau: tau.layer_taus.clone(),
        };

        let vanilla_proof = ZigZagDrgPoRep::<ZZTree>::prove(&pp, &pub_inputs, &priv_inputs)
            .expect("vanilla prove failed");

        assert!(
            ZigZagDrgPoRep::<ZZTree>::verify(&pp, &pub_inputs, &vanilla_proof)
                .expect("vanilla verify errored"),
            "vanilla proof must verify before circuit synthesis"
        );

        let circuit = ZigZagCircuit::<ZZTree> {
            public_params: pp,
            replica_id: Some(replica_id),
            comm_d: Some(simplified.comm_d),
            comm_r: Some(simplified.comm_r),
            comm_r_star: Some(tau.comm_r_star),
            proof: Some(vanilla_proof),
        };

        let mut cs = TestConstraintSystem::<Fr>::new();
        circuit
            .synthesize(&mut cs)
            .expect("circuit synthesis failed");

        if !cs.is_satisfied() {
            panic!("unsatisfied: {:?}", cs.which_is_unsatisfied());
        }
        assert!(cs.is_satisfied(), "constraints not satisfied");
        println!(
            "zigzag circuit ({} layers, {} challenges/layer): {} constraints, {} inputs",
            layers,
            challenge_count,
            cs.num_constraints(),
            cs.num_inputs()
        );
    }

    #[test]
    fn zigzag_circuit_satisfied_single_layer() {
        synthesize_satisfied(1, 1);
    }

    #[test]
    fn zigzag_circuit_satisfied_multi_layer() {
        synthesize_satisfied(3, 2);
    }
}
