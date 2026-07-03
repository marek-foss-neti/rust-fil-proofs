use std::marker::PhantomData;

use filecoin_hashers::Hasher;
use serde::{Deserialize, Serialize};
use storage_proofs_core::{
    drgraph::Graph,
    error::Result,
    merkle::{create_base_merkle_tree, MerkleProofTrait, MerkleTreeTrait},
    por::DataProof,
    proof::ProofScheme,
    util::NODE_SIZE,
};

use crate::{
    encode,
    zigzag::vanilla::{
        challenges::LayerChallenges,
        graph::ZigZagBucketGraph,
        params::{
            comm_r_star, ChallengeRequirements, LayerTau, PrivateInputs, PublicInputs,
            PublicParams, SetupParams, Tau,
        },
        vde,
    },
};

/// ZigZag layered DRG PoRep.
///
/// Replicates data through `layers` layers, "zigzagging" (reversing) the graph between each layer.
/// Every layer's encoding is committed to with its own Merkle tree (of the caller-chosen `Tree`
/// type), and the per-layer replica roots are folded into a single `comm_r_star`.
#[derive(Debug)]
pub struct ZigZagDrgPoRep<Tree>
where
    Tree: MerkleTreeTrait,
{
    _tree: PhantomData<Tree>,
}

/// The DrgPoRep-style proof for a single ZigZag layer: for each challenged node, an inclusion proof
/// of the encoded node and its parents in the layer's replica tree, plus an inclusion proof of the
/// pre-encoding ("data") node in the previous layer's tree.
#[derive(Debug, Serialize, Deserialize)]
pub struct LayerProof<Tree: MerkleTreeTrait> {
    #[serde(bound = "")]
    pub replica_nodes: Vec<DataProof<Tree::Proof>>,
    #[serde(bound = "")]
    pub replica_parents: Vec<Vec<(u32, DataProof<Tree::Proof>)>>,
    #[serde(bound = "")]
    pub nodes: Vec<DataProof<Tree::Proof>>,
}

impl<Tree: MerkleTreeTrait> Clone for LayerProof<Tree> {
    fn clone(&self) -> Self {
        LayerProof {
            replica_nodes: self.replica_nodes.clone(),
            replica_parents: self.replica_parents.clone(),
            nodes: self.nodes.clone(),
        }
    }
}

/// A full ZigZag proof: one `LayerProof` per layer, plus the per-layer taus.
#[derive(Debug, Serialize, Deserialize)]
pub struct Proof<Tree: MerkleTreeTrait> {
    #[serde(bound = "")]
    pub encoding_proofs: Vec<LayerProof<Tree>>,
    #[serde(bound = "")]
    pub tau: Vec<LayerTau<<Tree::Hasher as Hasher>::Domain>>,
}

impl<Tree: MerkleTreeTrait> Clone for Proof<Tree> {
    fn clone(&self) -> Self {
        Proof {
            encoding_proofs: self.encoding_proofs.clone(),
            tau: self.tau.clone(),
        }
    }
}

impl<Tree> ZigZagDrgPoRep<Tree>
where
    Tree: 'static + MerkleTreeTrait,
{
    /// Transform a layer's graph into the next layer's graph. For ZigZag this simply toggles the
    /// graph's direction (the expensive work happens lazily when parents are computed).
    pub fn transform(graph: &ZigZagBucketGraph<Tree::Hasher>) -> ZigZagBucketGraph<Tree::Hasher> {
        graph.zigzag()
    }

    /// Transform a layer's graph into the previous layer's graph. Because `transform` is an
    /// involution for ZigZag, the inverse is the same toggle.
    pub fn invert_transform(
        graph: &ZigZagBucketGraph<Tree::Hasher>,
    ) -> ZigZagBucketGraph<Tree::Hasher> {
        graph.zigzag()
    }

    /// Encode `data` in place through all layers, returning the per-layer taus (with `comm_r_star`)
    /// and the per-layer replica Merkle trees.
    pub fn transform_and_replicate_layers(
        graph: &ZigZagBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        data: &mut [u8],
    ) -> Result<(Tau<<Tree::Hasher as Hasher>::Domain>, Vec<Tree>)> {
        let layers = layer_challenges.layers();
        assert!(layers > 0);
        assert_eq!(data.len() % NODE_SIZE, 0);

        let leaves = data.len() / NODE_SIZE;

        let mut trees: Vec<Tree> = Vec::with_capacity(layers + 1);
        let mut current_graph = graph.clone();

        // Layer 0: the Merkle tree over the original (unencoded) data.
        trees.push(create_base_merkle_tree::<Tree>(None, leaves, data)?);

        for _layer in 0..layers {
            // Encode this layer in place using the current (possibly reversed) graph.
            vde::encode(&current_graph, replica_id, data)?;

            // Commit to the freshly-encoded layer.
            trees.push(create_base_merkle_tree::<Tree>(None, leaves, data)?);

            // Prepare the next layer's graph.
            current_graph = Self::transform(&current_graph);
        }

        // Build the per-layer taus: layer `i`'s comm_d is tree `i`'s root, comm_r is tree `i+1`'s root.
        let mut layer_taus = Vec::with_capacity(layers);
        let mut comm_rs = Vec::with_capacity(layers);
        for i in 0..layers {
            let comm_d = trees[i].root();
            let comm_r = trees[i + 1].root();
            layer_taus.push(LayerTau::new(comm_d, comm_r));
            comm_rs.push(comm_r);
        }

        let comm_r_star = comm_r_star::<Tree::Hasher>(replica_id, &comm_rs)?;

        Ok((
            Tau {
                layer_taus,
                comm_r_star,
            },
            trees,
        ))
    }

    /// Invert the replication, recovering the original data in place.
    ///
    /// ZigZag decoding proceeds from the last layer back to the first. Because each layer's graph is
    /// the reverse of the previous, we start from the final layer's graph and toggle back on each
    /// step, decoding with the same graph direction that encoded that layer.
    pub fn extract_and_invert_transform_layers(
        graph: &ZigZagBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        data: &mut [u8],
    ) -> Result<()> {
        let layers = layer_challenges.layers();
        assert!(layers > 0);

        // Reconstruct the graph used by the final layer: `layers - 1` toggles from the base graph.
        let mut layer_graph = graph.clone();
        for _ in 0..(layers - 1) {
            layer_graph = Self::transform(&layer_graph);
        }

        // Decode from the last layer back to the first.
        for _ in 0..layers {
            let decoded = vde::decode(&layer_graph, replica_id, data)?;
            data.copy_from_slice(&decoded);
            layer_graph = Self::invert_transform(&layer_graph);
        }

        Ok(())
    }
}

/// Build the ZigZag public parameters for the given setup parameters.
pub fn setup<Tree: MerkleTreeTrait>(sp: &SetupParams) -> Result<PublicParams<Tree>> {
    let graph = ZigZagBucketGraph::<Tree::Hasher>::new_zigzag(
        None,
        sp.nodes,
        sp.degree,
        sp.expansion_degree,
        sp.porep_id,
        sp.api_version,
    )?;

    Ok(PublicParams::new(graph, sp.layer_challenges.clone()))
}

impl<'a, Tree> ProofScheme<'a> for ZigZagDrgPoRep<Tree>
where
    Tree: 'static + MerkleTreeTrait,
{
    type PublicParams = PublicParams<Tree>;
    type SetupParams = SetupParams;
    type PublicInputs = PublicInputs<<Tree::Hasher as Hasher>::Domain>;
    type PrivateInputs = PrivateInputs<Tree>;
    type Proof = Proof<Tree>;
    type Requirements = ChallengeRequirements;

    fn setup(sp: &Self::SetupParams) -> Result<Self::PublicParams> {
        setup::<Tree>(sp)
    }

    fn prove(
        pub_params: &Self::PublicParams,
        pub_inputs: &Self::PublicInputs,
        priv_inputs: &Self::PrivateInputs,
    ) -> Result<Self::Proof> {
        let proofs = Self::prove_all_partitions(pub_params, pub_inputs, priv_inputs, 1)?;
        Ok(proofs.into_iter().next().expect("missing partition proof"))
    }

    fn prove_all_partitions(
        pub_params: &Self::PublicParams,
        pub_inputs: &Self::PublicInputs,
        priv_inputs: &Self::PrivateInputs,
        partition_count: usize,
    ) -> Result<Vec<Self::Proof>> {
        assert!(partition_count > 0);

        let layers = pub_params.layer_challenges.layers();

        (0..partition_count)
            .map(|k| {
                let mut encoding_proofs = Vec::with_capacity(layers);
                let mut layer_graph = pub_params.graph.clone();

                for layer in 0..layers {
                    let tree_d = &priv_inputs.aux[layer];
                    let tree_r = &priv_inputs.aux[layer + 1];
                    let graph_size = layer_graph.size();
                    let degree = layer_graph.degree();

                    let challenges = pub_inputs.challenges(
                        &pub_params.layer_challenges,
                        graph_size,
                        layer as u8,
                        Some(k),
                    );

                    let mut replica_nodes = Vec::with_capacity(challenges.len());
                    let mut replica_parents = Vec::with_capacity(challenges.len());
                    let mut nodes = Vec::with_capacity(challenges.len());

                    let mut parents = vec![0u32; degree];
                    for challenge in challenges {
                        let challenge = challenge % graph_size;
                        assert_ne!(challenge, 0, "cannot prove the first node");

                        let replica_proof = tree_r.gen_proof(challenge)?;
                        let replica_data = replica_proof.leaf();
                        replica_nodes.push(DataProof {
                            data: replica_data,
                            proof: replica_proof,
                        });

                        layer_graph.parents(challenge, &mut parents)?;
                        let mut parent_proofs = Vec::with_capacity(parents.len());
                        for parent in &parents {
                            let proof = tree_r.gen_proof(*parent as usize)?;
                            let data = proof.leaf();
                            parent_proofs.push((*parent, DataProof { data, proof }));
                        }
                        replica_parents.push(parent_proofs);

                        let data_proof = tree_d.gen_proof(challenge)?;
                        let data = data_proof.leaf();
                        nodes.push(DataProof {
                            data,
                            proof: data_proof,
                        });
                    }

                    encoding_proofs.push(LayerProof {
                        replica_nodes,
                        replica_parents,
                        nodes,
                    });

                    layer_graph = Self::transform(&layer_graph);
                }

                Ok(Proof {
                    encoding_proofs,
                    tau: priv_inputs.tau.clone(),
                })
            })
            .collect()
    }

    fn verify(
        pub_params: &Self::PublicParams,
        pub_inputs: &Self::PublicInputs,
        proof: &Self::Proof,
    ) -> Result<bool> {
        let layers = pub_params.layer_challenges.layers();
        let k = pub_inputs.k.unwrap_or(0);

        if proof.encoding_proofs.len() != layers {
            return Ok(false);
        }
        if proof.tau.len() != layers {
            return Ok(false);
        }

        let mut layer_graph = pub_params.graph.clone();

        for layer in 0..layers {
            let layer_proof = &proof.encoding_proofs[layer];
            let tau = &proof.tau[layer];
            let graph_size = layer_graph.size();
            let degree = layer_graph.degree();

            let challenges = pub_inputs.challenges(
                &pub_params.layer_challenges,
                graph_size,
                layer as u8,
                Some(k),
            );

            if layer_proof.replica_nodes.len() != challenges.len()
                || layer_proof.replica_parents.len() != challenges.len()
                || layer_proof.nodes.len() != challenges.len()
            {
                return Ok(false);
            }

            let mut expected_parents = vec![0u32; degree];
            for (i, challenge) in challenges.into_iter().enumerate() {
                let challenge = challenge % graph_size;
                if challenge == 0 {
                    return Ok(false);
                }

                let replica_node = &layer_proof.replica_nodes[i];
                let parents = &layer_proof.replica_parents[i];
                let data_node = &layer_proof.nodes[i];

                // Merkle inclusion: encoded node and data node against their respective roots.
                if !replica_node.proof.validate(challenge)
                    || replica_node.proof.root() != tau.comm_r
                {
                    return Ok(false);
                }
                if !data_node.proof.validate(challenge) || data_node.proof.root() != tau.comm_d {
                    return Ok(false);
                }

                // Parents must match the graph and be included in the replica tree.
                layer_graph.parents(challenge, &mut expected_parents)?;
                if parents.len() != expected_parents.len() {
                    return Ok(false);
                }
                for ((parent, parent_proof), expected) in parents.iter().zip(&expected_parents) {
                    if parent != expected {
                        return Ok(false);
                    }
                    if !parent_proof.proof.validate(*parent as usize)
                        || parent_proof.proof.root() != tau.comm_r
                    {
                        return Ok(false);
                    }
                }

                // Encoding relation: decode(replica_node) with the KDF-derived key must equal the
                // data node.
                let parent_data: Vec<_> = parents.iter().map(|(_, p)| p.data).collect();
                let key = vde::create_key_from_domains::<Tree::Hasher>(
                    &pub_inputs.replica_id,
                    &parent_data,
                )?;
                let unsealed = encode::decode(key, replica_node.data);
                if unsealed != data_node.data {
                    return Ok(false);
                }
            }

            layer_graph = Self::transform(&layer_graph);
        }

        Ok(true)
    }

    fn with_partition(mut pub_in: Self::PublicInputs, k: Option<usize>) -> Self::PublicInputs {
        pub_in.k = k;
        pub_in
    }

    fn satisfies_requirements(
        pub_params: &Self::PublicParams,
        requirements: &Self::Requirements,
        partitions: usize,
    ) -> bool {
        let total = pub_params.layer_challenges.total_challenges() * partitions;
        total >= requirements.minimum_challenges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use filecoin_hashers::poseidon::{PoseidonDomain, PoseidonHasher};
    use generic_array::typenum::{U0, U2};
    use storage_proofs_core::{
        api_version::ApiVersion,
        drgraph::BASE_DEGREE,
        merkle::{DiskStore, MerkleTreeWrapper},
    };

    use crate::zigzag::vanilla::{
        graph::EXP_DEGREE,
        params::SetupParams,
    };

    type ZZTree = MerkleTreeWrapper<PoseidonHasher, DiskStore<PoseidonDomain>, U2, U0, U0>;

    fn replicate_extract_roundtrip(layers: usize) {
        let nodes = 128;
        let porep_id = [9u8; 32];

        let sp = SetupParams {
            nodes,
            degree: BASE_DEGREE,
            expansion_degree: EXP_DEGREE,
            porep_id,
            api_version: ApiVersion::V1_2_0,
            layer_challenges: LayerChallenges::new_fixed(layers, 1),
        };

        let pp = setup::<ZZTree>(&sp).expect("setup failed");

        let replica_id = PoseidonDomain::default();
        let original = vec![0u8; nodes * NODE_SIZE];
        let mut data = original.clone();

        let (tau, trees) =
            ZigZagDrgPoRep::<ZZTree>::transform_and_replicate_layers(
                &pp.graph,
                &pp.layer_challenges,
                &replica_id,
                &mut data,
            )
            .expect("replication failed");

        assert_ne!(data, original, "replication did not change data");
        assert_eq!(tau.layer_taus.len(), layers);
        assert_eq!(trees.len(), layers + 1);

        // comm_d of layer 0 must be the root of the tree over the original data.
        let original_tree = create_base_merkle_tree::<ZZTree>(None, nodes, &original)
            .expect("failed to build original tree");
        assert_eq!(tau.layer_taus[0].comm_d, original_tree.root());

        // The final replica commitment must match the last tree's root.
        assert_eq!(tau.simplify().comm_r, trees[layers].root());

        ZigZagDrgPoRep::<ZZTree>::extract_and_invert_transform_layers(
            &pp.graph,
            &pp.layer_challenges,
            &replica_id,
            &mut data,
        )
        .expect("extraction failed");

        assert_eq!(data, original, "extraction did not recover original data");
    }

    #[test]
    fn replicate_extract_single_layer() {
        replicate_extract_roundtrip(1);
    }

    #[test]
    fn replicate_extract_multi_layer() {
        replicate_extract_roundtrip(4);
    }

    fn prove_verify(layers: usize, challenge_count: usize) {
        use storage_proofs_core::proof::ProofScheme;

        use crate::zigzag::vanilla::params::{comm_r_star, PrivateInputs, PublicInputs};

        let nodes = 128;
        let porep_id = [11u8; 32];

        let sp = SetupParams {
            nodes,
            degree: BASE_DEGREE,
            expansion_degree: EXP_DEGREE,
            porep_id,
            api_version: ApiVersion::V1_2_0,
            layer_challenges: LayerChallenges::new_fixed(layers, challenge_count),
        };

        let pp = ZigZagDrgPoRep::<ZZTree>::setup(&sp).expect("setup failed");

        let replica_id = PoseidonDomain::default();
        let mut data = vec![0u8; nodes * NODE_SIZE];

        let (tau, trees) = ZigZagDrgPoRep::<ZZTree>::transform_and_replicate_layers(
            &pp.graph,
            &pp.layer_challenges,
            &replica_id,
            &mut data,
        )
        .expect("replication failed");

        let pub_inputs = PublicInputs::<PoseidonDomain> {
            replica_id,
            seed: None,
            tau: Some(tau.simplify()),
            comm_r_star: tau.comm_r_star,
            k: None,
        };

        let priv_inputs = PrivateInputs::<ZZTree> {
            aux: trees,
            tau: tau.layer_taus.clone(),
        };

        let proof = ZigZagDrgPoRep::<ZZTree>::prove(&pp, &pub_inputs, &priv_inputs)
            .expect("prove failed");

        assert!(
            ZigZagDrgPoRep::<ZZTree>::verify(&pp, &pub_inputs, &proof).expect("verify errored"),
            "valid proof did not verify"
        );

        // Tampering with the replica id must make verification fail.
        let mut bad_inputs = pub_inputs.clone();
        let mut wrong_comm_rs = Vec::new();
        for lt in &tau.layer_taus {
            wrong_comm_rs.push(lt.comm_r);
        }
        bad_inputs.replica_id = PoseidonDomain::from([3u8; 32]);
        bad_inputs.comm_r_star =
            comm_r_star::<PoseidonHasher>(&bad_inputs.replica_id, &wrong_comm_rs)
                .expect("comm_r_star failed");
        assert!(
            !ZigZagDrgPoRep::<ZZTree>::verify(&pp, &bad_inputs, &proof)
                .expect("verify errored"),
            "proof verified under the wrong replica id"
        );
    }

    #[test]
    fn prove_verify_single_layer() {
        prove_verify(1, 2);
    }

    #[test]
    fn prove_verify_multi_layer() {
        prove_verify(4, 2);
    }
}
