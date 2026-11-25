//! The SDR v2 graph: the stacked graph of SDR, read as a graph on label components.
//!
//! Nothing about the graph is new. The DRG parents are those of `StackedBucketGraph`, and the
//! component permutation is the Feistel permutation from which `StackedBucketGraph` derives its
//! expander parents: `word_source(8i + a) / 8` is expander parent `a` of node `i` in SDR.
//!
//! Both are computed here by code of this module ([`DrgSampler`], [`WordPermutation`]) because it
//! is several times cheaper than the general-purpose code of the stacked graph; the tests below
//! check that the results are the same.

use anyhow::ensure;
use filecoin_hashers::poseidon::PoseidonHasher;
use storage_proofs_core::{
    api_version::ApiVersion,
    drgraph::{Graph, BASE_DEGREE},
    error::Result,
    PoRepID,
};

use super::drg::DrgSampler;
use super::label::{LabelGraph, COMPONENTS, DRG_DEGREE};
use super::permutation::WordPermutation;
use crate::stacked::{StackedBucketGraph, EXP_DEGREE};

const _: () = assert!(DRG_DEGREE == BASE_DEGREE);
const _: () = assert!(COMPONENTS == EXP_DEGREE);

/// The graph of an SDR v2 sector.
pub struct SdrV2Graph {
    stacked: StackedBucketGraph<PoseidonHasher>,
    drg: DrgSampler,
    permutation: WordPermutation,
}

impl SdrV2Graph {
    /// `nodes` is the number of 32-byte nodes of the sector. `ApiVersion::V1_0_0` is rejected: its
    /// DRG puts the predecessor last and its expander truncates indices of large sectors.
    pub fn new(nodes: usize, porep_id: PoRepID, api_version: ApiVersion) -> Result<Self> {
        ensure!(
            !matches!(api_version, ApiVersion::V1_0_0),
            "SDR v2 needs the graph of API version 1.1.0 or later"
        );
        ensure!(nodes > 0, "SDR v2 graph has no nodes");
        let stacked = StackedBucketGraph::<PoseidonHasher>::new_stacked(
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            porep_id,
            api_version,
        )?;
        let drg = DrgSampler::new(&stacked.seed());
        let permutation =
            WordPermutation::new(&stacked.feistel_keys, (nodes as u64) * (EXP_DEGREE as u64));
        Ok(Self {
            stacked,
            drg,
            permutation,
        })
    }

    /// The SDR graph this graph is read from.
    pub fn stacked(&self) -> &StackedBucketGraph<PoseidonHasher> {
        &self.stacked
    }

    /// The component permutation between two layers.
    pub fn permutation(&self) -> &WordPermutation {
        &self.permutation
    }
}

// SAFETY: `word_source` is a Feistel permutation of `0..8 * nodes` (tested below against
// `feistel::permute`), and the DRG parents are those of `BucketGraph::parents` (also tested
// below), which have a lower index for every node but node 0.
unsafe impl LabelGraph for SdrV2Graph {
    fn nodes(&self) -> usize {
        self.stacked.size()
    }

    #[inline]
    fn drg_parents(&self, node: usize, parents: &mut [u32; DRG_DEGREE]) {
        self.drg.parents(node, std::slice::from_mut(parents));
    }

    #[inline]
    fn drg_parents_batch(&self, first: usize, parents: &mut [[u32; DRG_DEGREE]]) {
        self.drg.parents(first, parents);
    }

    #[inline]
    fn word_source(&self, word: u64) -> u64 {
        self.permutation.permute(word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use sha2::{Digest, Sha256};
    use storage_proofs_core::crypto::feistel;

    use crate::sdr_v2::{
        decode_layers, encode_layers, DrgParentTable, SdrV2Params, DEFAULT_DECODE_BATCH,
    };

    // Inputs and expected values shared with `FABLE-PLAN-SDRv2-scripts/sdrv2_model.py`.
    const MODEL_POREP_ID: PoRepID = [5u8; 32];
    const MODEL_NODES: usize = 64;

    fn model_replica_id() -> [u8; 32] {
        let mut id = [0u8; 32];
        for (index, byte) in id.iter_mut().enumerate() {
            *byte = index as u8 + 1;
        }
        id
    }

    fn pattern(nodes: usize) -> Vec<u8> {
        (0..nodes * 32)
            .map(|index| ((index * 31 + (index >> 3) + 17) & 0xff) as u8)
            .collect()
    }

    fn graph(nodes: usize, porep_id: PoRepID) -> SdrV2Graph {
        SdrV2Graph::new(nodes, porep_id, ApiVersion::V1_2_0).expect("graph")
    }

    #[test]
    fn rejects_the_first_api_version() {
        assert!(SdrV2Graph::new(64, MODEL_POREP_ID, ApiVersion::V1_0_0).is_err());
    }

    #[test]
    fn components_follow_the_expander_edges_of_sdr() {
        for nodes in [64usize, 1 << 10, 1 << 18] {
            let graph = graph(nodes, [32u8; 32]);
            let domain = (nodes * EXP_DEGREE) as u64;
            let keys = graph.stacked.feistel_keys;
            let precomputed = feistel::precompute(domain);
            let step = (nodes / 4096).max(1);
            let mut expander = [0u32; EXP_DEGREE];
            for node in (0..nodes).step_by(step) {
                graph.stacked.generate_expanded_parents(node, &mut expander);
                for (slot, parent) in expander.iter().enumerate() {
                    let word = (node * EXP_DEGREE + slot) as u64;
                    let source = graph.word_source(word);
                    assert_eq!(source, feistel::permute(domain, word, &keys, precomputed));
                    assert_eq!(source / EXP_DEGREE as u64, u64::from(*parent));
                    assert_eq!(graph.permutation.invert(source), word);
                    assert_eq!(
                        feistel::invert_permute(domain, source, &keys, precomputed),
                        word
                    );
                }
            }
        }
    }

    #[test]
    fn drg_parents_are_those_of_the_stacked_graph() {
        // `BucketGraph::parents` does not depend on the size of the graph, so one large graph
        // covers every node index up to the 32 GiB sector size.
        let nodes = 1usize << 30;
        let large = graph(nodes, [32u8; 32]);
        let mut expected = [0u32; DRG_DEGREE];
        let mut check = |first: usize, count: usize| {
            let mut batch = vec![[0u32; DRG_DEGREE]; count];
            large.drg_parents_batch(first, &mut batch);
            for (index, parents) in batch.iter().enumerate() {
                let node = first + index;
                large
                    .stacked
                    .base_parents(node, &mut expected[..])
                    .expect("base parents");
                assert_eq!(*parents, expected, "node {}", node);
                let mut single = [0u32; DRG_DEGREE];
                large.drg_parents(node, &mut single);
                assert_eq!(single, expected, "node {}", node);
            }
        };
        check(0, 100_000);
        // Around every change of the number of buckets, and at the end of the graph.
        for exponent in 4..=32u32 {
            let node = ((1u64 << exponent) / 5) as usize;
            if node + 40 <= nodes {
                check(node.saturating_sub(20), 40);
            }
        }
        check(nodes - 1000, 1000);
        for step in 1..200usize {
            check(step * 5_368_709, 37);
        }

        let graph = graph(MODEL_NODES, MODEL_POREP_ID);
        let mut parents = [0u32; DRG_DEGREE];
        for node in 1..MODEL_NODES {
            graph.drg_parents(node, &mut parents);
            assert!(parents.iter().all(|parent| (*parent as usize) < node));
            if node > 1 {
                assert_eq!(parents[0] as usize, node - 1);
            }
        }
        // The same values as the model.
        graph.drg_parents(7, &mut parents);
        assert_eq!(parents, [6, 6, 5, 6, 6, 2]);
        graph.drg_parents(63, &mut parents);
        assert_eq!(parents, [62, 62, 61, 23, 16, 22]);
        let sources: Vec<u64> = (0..8).map(|word| graph.word_source(word)).collect();
        assert_eq!(sources, [62, 173, 16, 227, 477, 345, 93, 442]);
    }

    #[test]
    fn replica_matches_the_model() {
        let graph = graph(MODEL_NODES, MODEL_POREP_ID);
        let expected = [
            (
                1,
                5,
                "43dd8d0589804a2597f73ea720713f6948d6c8a239e6d8dae636be00544f7b4e",
            ),
            (
                2,
                5,
                "972b38360975c632abf910aaa82317f49911b9b5e3a3525d8650281e68063e49",
            ),
            (
                11,
                5,
                "dbea49168d6cd941b2741130cd839102786e7146255fb62b7aa7ae1fe9b218fd",
            ),
            (
                11,
                16,
                "fe59a2cd79c72629dae9fda92408adcfca0aa5f8c7fe27537b2ed598579afa4b",
            ),
        ];
        let table = DrgParentTable::build(&graph).expect("parent table");
        for (index, (layers, key_blocks, digest)) in expected.iter().copied().enumerate() {
            // The table must not change a single byte: alternate between both sources.
            let (encode_table, decode_table) = if index % 2 == 0 {
                (None, Some(&table))
            } else {
                (Some(&table), None)
            };
            let params = SdrV2Params::new(layers, key_blocks).expect("parameters");
            let data = pattern(MODEL_NODES);
            let mut sealed = data.clone();
            let mut scratch = vec![0u8; sealed.len()];
            encode_layers(
                &graph,
                &params,
                &model_replica_id(),
                &mut sealed,
                &mut scratch,
                encode_table,
                &mut |_| {},
            )
            .expect("encode");
            let hex: String = Sha256::digest(&sealed)
                .iter()
                .map(|byte| format!("{:02x}", byte))
                .collect();
            assert_eq!(hex, digest, "layers {} key blocks {}", layers, key_blocks);

            let mut other = vec![0xa5u8; sealed.len()];
            let decoded = decode_layers(
                &graph,
                &params,
                &model_replica_id(),
                &mut sealed,
                &mut other,
                DEFAULT_DECODE_BATCH,
                decode_table,
                &mut |_| {},
            )
            .expect("decode");
            assert_eq!(&decoded[..], &data[..]);
        }
    }

    #[test]
    fn round_trip() {
        // Node counts that cross the chunk and window boundaries of the encoder and decoder; the
        // last one is an 8 MiB sector. Few layers on the large graphs: an unoptimized test build
        // spends about 10 seconds per layer of 2^18 nodes.
        let cases = [
            (20_000usize, 11usize, 5usize, 8usize),
            (20_000, 3, 16, 1),
            (70_001, 2, 18, 64),
            (1 << 18, 2, 5, 8),
        ];
        for (case, (nodes, layers, key_blocks, batch)) in cases.iter().copied().enumerate() {
            let graph = graph(nodes, [9u8; 32]);
            let table = DrgParentTable::build(&graph).expect("parent table");
            let (encode_table, decode_table) = if case % 2 == 0 {
                (Some(&table), None)
            } else {
                (None, Some(&table))
            };
            let params = SdrV2Params::new(layers, key_blocks).expect("parameters");
            let data = pattern(nodes);
            let mut sealed = data.clone();
            let mut scratch = vec![0u8; sealed.len()];
            encode_layers(
                &graph,
                &params,
                &model_replica_id(),
                &mut sealed,
                &mut scratch,
                encode_table,
                &mut |_| {},
            )
            .expect("encode");
            assert_ne!(sealed, data);
            let mut other = vec![0u8; sealed.len()];
            let decoded = decode_layers(
                &graph,
                &params,
                &model_replica_id(),
                &mut sealed,
                &mut other,
                batch,
                decode_table,
                &mut |_| {},
            )
            .expect("decode");
            assert!(
                decoded[..] == data[..],
                "nodes {} layers {} key blocks {}",
                nodes,
                layers,
                key_blocks
            );
        }
    }
}
