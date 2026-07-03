use filecoin_hashers::{Domain, HashFunction, Hasher};
use serde::{Deserialize, Serialize};
use storage_proofs_core::{
    api_version::ApiVersion, error::Result, merkle::MerkleTreeTrait,
    parameter_cache::ParameterSetMetadata, PoRepID,
};

use crate::zigzag::vanilla::{
    challenges::{derive_challenges, LayerChallenges},
    graph::ZigZagBucketGraph,
};

/// Parameters for setting up a ZigZag layered PoRep.
#[derive(Debug, Clone)]
pub struct SetupParams {
    pub nodes: usize,
    pub degree: usize,
    pub expansion_degree: usize,
    pub porep_id: PoRepID,
    pub api_version: ApiVersion,
    pub layer_challenges: LayerChallenges,
}

/// Public parameters for a ZigZag layered PoRep, over a Poseidon Merkle tree of type `Tree`.
pub struct PublicParams<Tree>
where
    Tree: MerkleTreeTrait,
{
    pub graph: ZigZagBucketGraph<Tree::Hasher>,
    pub layer_challenges: LayerChallenges,
}

impl<Tree: MerkleTreeTrait> std::fmt::Debug for PublicParams<Tree> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublicParams")
            .field("graph", &self.graph)
            .field("layer_challenges", &self.layer_challenges)
            .finish()
    }
}

impl<Tree: MerkleTreeTrait> Clone for PublicParams<Tree> {
    fn clone(&self) -> Self {
        PublicParams {
            graph: self.graph.clone(),
            layer_challenges: self.layer_challenges.clone(),
        }
    }
}

impl<Tree> PublicParams<Tree>
where
    Tree: MerkleTreeTrait,
{
    pub fn new(
        graph: ZigZagBucketGraph<Tree::Hasher>,
        layer_challenges: LayerChallenges,
    ) -> Self {
        PublicParams {
            graph,
            layer_challenges,
        }
    }
}

impl<Tree> ParameterSetMetadata for PublicParams<Tree>
where
    Tree: MerkleTreeTrait,
{
    fn identifier(&self) -> String {
        format!(
            "zigzag::PublicParams{{ graph: {}, challenges: {:?}, tree: {} }}",
            self.graph.identifier(),
            self.layer_challenges,
            Tree::display(),
        )
    }

    fn sector_size(&self) -> u64 {
        self.graph.sector_size()
    }
}

/// A single layer's data and replica commitments (`comm_d` = input tree root, `comm_r` = encoded
/// tree root). Equivalent to the 2019 `porep::Tau`, which no longer exists in the core crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerTau<D: Domain> {
    #[serde(bound = "")]
    pub comm_d: D,
    #[serde(bound = "")]
    pub comm_r: D,
}

impl<D: Domain> LayerTau<D> {
    pub fn new(comm_d: D, comm_r: D) -> Self {
        LayerTau { comm_d, comm_r }
    }
}

/// The per-layer commitments produced during replication, plus the aggregate `comm_r_star`.
#[derive(Debug, Clone)]
pub struct Tau<D: Domain> {
    pub layer_taus: Vec<LayerTau<D>>,
    pub comm_r_star: D,
}

impl<D: Domain> Tau<D> {
    /// Collapse the per-layer taus into a single `LayerTau` using the original data commitment
    /// (`comm_d` of the first layer) and the final replica commitment (`comm_r` of the last layer).
    pub fn simplify(&self) -> LayerTau<D> {
        LayerTau {
            comm_r: self.layer_taus[self.layer_taus.len() - 1].comm_r,
            comm_d: self.layer_taus[0].comm_d,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicInputs<D: Domain> {
    #[serde(bound = "")]
    pub replica_id: D,
    #[serde(bound = "")]
    pub seed: Option<D>,
    #[serde(bound = "")]
    pub tau: Option<LayerTau<D>>,
    #[serde(bound = "")]
    pub comm_r_star: D,
    pub k: Option<usize>,
}

impl<D: Domain> PublicInputs<D> {
    pub fn challenges(
        &self,
        layer_challenges: &LayerChallenges,
        leaves: usize,
        layer: u8,
        partition_k: Option<usize>,
    ) -> Vec<usize> {
        let commitment = self.seed.as_ref().unwrap_or(&self.comm_r_star);
        derive_challenges::<D>(
            layer_challenges,
            layer,
            leaves,
            &self.replica_id,
            commitment,
            partition_k.unwrap_or(0) as u8,
        )
    }
}

/// Private inputs for proving: the per-layer replica trees and their taus.
pub struct PrivateInputs<Tree>
where
    Tree: MerkleTreeTrait,
{
    pub aux: Vec<Tree>,
    pub tau: Vec<LayerTau<<Tree::Hasher as Hasher>::Domain>>,
}

/// The ZigZag challenge requirement: at least `minimum_challenges` across all partitions.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeRequirements {
    pub minimum_challenges: usize,
}

/// Computes the aggregate `comm_r_star` as a commitment over `replica_id` and every per-layer
/// replica root.
///
/// The 2019 ZigZag used Pedersen, which hashes arbitrary-length input. Poseidon only supports fixed
/// arities, so we instead fold `[replica_id, comm_r_0, comm_r_1, ...]` with a binary
/// Merkle-Damgard construction (`hash_md`, i.e. repeated `hash2`), which preserves the "vector
/// commitment over all layer roots" property.
pub fn comm_r_star<H: Hasher>(
    replica_id: &H::Domain,
    comm_rs: &[H::Domain],
) -> Result<H::Domain> {
    let input: Vec<H::Domain> = std::iter::once(*replica_id)
        .chain(comm_rs.iter().copied())
        .collect();

    Ok(H::Function::hash_md(&input))
}
