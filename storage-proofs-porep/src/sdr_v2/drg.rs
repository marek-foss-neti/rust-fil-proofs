//! The DRG parents of the stacked graph, computed in batches, and an optional table of them.
//!
//! The parents are those of `BucketGraph::parents` for API version 1.1.0 and later, bit for bit:
//! the tests of [`super::graph`] compare the two on the real graph. `BucketGraph` seeds a
//! `ChaCha8Rng` with `drg_seed (28 bytes) || node (little-endian u32)` and draws ten `u64`. Those
//! are the first 20 output words of ChaCha8, so two blocks are enough, and because the nodes of a
//! batch differ only in one key word, the blocks of a whole batch are computed side by side: one
//! lane per node, which the compiler turns into vector instructions.
//!
//! The parents of a node do not depend on the layer. [`DrgParentTable`] therefore computes them
//! once per sector, so that the passes over the layers read them back instead of recomputing them.

use std::convert::TryFrom;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{ensure, Result};
use rayon::prelude::*;

use super::label::{LabelGraph, DRG_DEGREE};

/// Nodes whose ChaCha8 blocks are computed together.
const LANES: usize = 16;
/// Parents kept per node in a [`DrgParentTable`]: all but the predecessor.
const STORED_PARENTS: usize = DRG_DEGREE - 1;
/// Nodes filled by one parallel task when a table is built.
const BUILD_CHUNK_NODES: usize = 1 << 12;

/// "expand 32-byte k".
const CHACHA_CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

// One lane per node, indexed so that each loop is a plain vector operation.
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn quarter_round<const L: usize>(
    state: &mut [[u32; L]; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) {
    for lane in 0..L {
        state[a][lane] = state[a][lane].wrapping_add(state[b][lane]);
        state[d][lane] = (state[d][lane] ^ state[a][lane]).rotate_left(16);
    }
    for lane in 0..L {
        state[c][lane] = state[c][lane].wrapping_add(state[d][lane]);
        state[b][lane] = (state[b][lane] ^ state[c][lane]).rotate_left(12);
    }
    for lane in 0..L {
        state[a][lane] = state[a][lane].wrapping_add(state[b][lane]);
        state[d][lane] = (state[d][lane] ^ state[a][lane]).rotate_left(8);
    }
    for lane in 0..L {
        state[c][lane] = state[c][lane].wrapping_add(state[d][lane]);
        state[b][lane] = (state[b][lane] ^ state[c][lane]).rotate_left(7);
    }
}

/// ChaCha8 block `counter` of `L` nodes: word `w` of node `nodes[lane]` is `result[w][lane]`.
// One lane per node, indexed so that each loop is a plain vector operation.
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn chacha8_block<const L: usize>(key: &[u32; 7], nodes: &[u32; L], counter: u32) -> [[u32; L]; 16] {
    let mut initial = [[0u32; L]; 16];
    for lane in 0..L {
        for (word, constant) in CHACHA_CONSTANTS.iter().enumerate() {
            initial[word][lane] = *constant;
        }
        for (word, key_word) in key.iter().enumerate() {
            initial[4 + word][lane] = *key_word;
        }
        // The last key word is the node; words 12 and 13 are the 64-bit block counter and words
        // 14 and 15 the stream, which `ChaCha8Rng::from_seed` leaves at zero.
        initial[11][lane] = nodes[lane];
        initial[12][lane] = counter;
    }

    let mut state = initial;
    for _ in 0..4 {
        quarter_round(&mut state, 0, 4, 8, 12);
        quarter_round(&mut state, 1, 5, 9, 13);
        quarter_round(&mut state, 2, 6, 10, 14);
        quarter_round(&mut state, 3, 7, 11, 15);
        quarter_round(&mut state, 0, 5, 10, 15);
        quarter_round(&mut state, 1, 6, 11, 12);
        quarter_round(&mut state, 2, 7, 8, 13);
        quarter_round(&mut state, 3, 4, 9, 14);
    }
    for word in 0..16 {
        for lane in 0..L {
            state[word][lane] = state[word][lane].wrapping_add(initial[word][lane]);
        }
    }
    state
}

/// `(x as f64).log2().ceil()` for `2 <= x < 2^53`, without floating point.
#[inline(always)]
fn ceil_log2(x: u64) -> u64 {
    64 - u64::from((x - 1).leading_zeros())
}

/// The parents of nodes `first..first + L`, all of which must be at least 2.
// One lane per node, indexed so that each loop is a plain vector operation.
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn sample<const L: usize>(key: &[u32; 7], first: u32, parents: &mut [[u32; DRG_DEGREE]; L]) {
    let mut nodes = [0u32; L];
    for (lane, node) in nodes.iter_mut().enumerate() {
        *node = first + lane as u32;
    }
    let block0 = chacha8_block(key, &nodes, 0);
    let block1 = chacha8_block(key, &nodes, 1);

    let m_prime = (DRG_DEGREE - 1) as u64;
    for lane in 0..L {
        let node = nodes[lane];
        debug_assert!(node >= 2);
        // `rng.gen::<u64>()` number `index`: two consecutive output words, low word first.
        let draw = |index: usize| -> u64 {
            let (low, high) = if 2 * index < 16 {
                (block0[2 * index][lane], block0[2 * index + 1][lane])
            } else {
                (block1[2 * index - 16][lane], block1[2 * index - 15][lane])
            };
            u64::from(low) | (u64::from(high) << 32)
        };

        let metagraph_node = u64::from(node) * m_prime;
        let n_buckets = ceil_log2(metagraph_node);
        parents[lane][0] = node - 1;
        for parent in 0..DRG_DEGREE - 1 {
            let bucket_index = draw(2 * parent) % n_buckets + 1;
            let largest_distance = metagraph_node.min(1 << bucket_index);
            let smallest_distance = (largest_distance >> 1).max(2);
            let distance = smallest_distance
                + draw(2 * parent + 1) % (largest_distance - smallest_distance + 1);
            let mapped = ((metagraph_node - distance) / m_prime) as u32;
            parents[lane][parent + 1] = if mapped == node { node - 1 } else { mapped };
        }
    }
}

/// Computes DRG parents from the seed of a `BucketGraph`.
#[derive(Clone, Copy, Debug)]
pub struct DrgSampler {
    key: [u32; 7],
}

impl DrgSampler {
    /// `seed` is `BucketGraph::seed()`.
    pub fn new(seed: &[u8; 28]) -> Self {
        let mut key = [0u32; 7];
        for (word, bytes) in key.iter_mut().zip(seed.chunks_exact(4)) {
            *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        Self { key }
    }

    /// The parents of nodes `first..first + parents.len()`, predecessor first. Nodes 0 and 1 get
    /// node 0 six times, as in `BucketGraph`. Node indices must fit in a `u32`.
    pub fn parents(&self, first: usize, parents: &mut [[u32; DRG_DEGREE]]) {
        debug_assert!(first + parents.len() <= u32::MAX as usize + 1);
        let without_sampling = parents.len().min(2usize.saturating_sub(first));
        let (low, sampled) = parents.split_at_mut(without_sampling);
        for parents in low.iter_mut() {
            *parents = [0; DRG_DEGREE];
        }

        let mut node = first + without_sampling;
        let mut groups = sampled.chunks_exact_mut(LANES);
        for group in &mut groups {
            let group = <&mut [[u32; DRG_DEGREE]; LANES]>::try_from(group).expect("a full group");
            sample::<LANES>(&self.key, node as u32, group);
            node += LANES;
        }
        for parents in groups.into_remainder().iter_mut() {
            sample::<1>(&self.key, node as u32, std::array::from_mut(parents));
            node += 1;
        }
    }
}

/// The DRG parents of every node of a sector, computed once and shared by all layers.
///
/// A node's predecessor is implied, so five `u32` are kept per node: 20 GiB for a 32 GiB sector.
pub struct DrgParentTable {
    entries: Vec<[u32; STORED_PARENTS]>,
}

impl DrgParentTable {
    /// Memory needed for the table of a graph with `nodes` nodes.
    pub fn bytes(nodes: usize) -> u64 {
        nodes as u64 * (STORED_PARENTS * std::mem::size_of::<u32>()) as u64
    }

    /// Computes the table in parallel. Fails for a graph whose first parent is not the
    /// predecessor, which this table does not store.
    pub fn build<G: LabelGraph>(graph: &G) -> Result<Self> {
        let nodes = graph.nodes();
        // Zeroed pages are only committed when the parallel tasks below write them.
        let mut entries = vec![[0u32; STORED_PARENTS]; nodes];
        let predecessor_first = AtomicBool::new(true);
        entries
            .par_chunks_mut(BUILD_CHUNK_NODES)
            .enumerate()
            .for_each_init(
                || vec![[0u32; DRG_DEGREE]; BUILD_CHUNK_NODES],
                |buffer, (chunk, stored)| {
                    let first = chunk * BUILD_CHUNK_NODES;
                    let parents = &mut buffer[..stored.len()];
                    graph.drg_parents_batch(first, parents);
                    for (index, (stored, parents)) in
                        stored.iter_mut().zip(parents.iter()).enumerate()
                    {
                        if parents[0] as usize != (first + index).saturating_sub(1) {
                            predecessor_first.store(false, Ordering::Relaxed);
                        }
                        stored.copy_from_slice(&parents[1..]);
                    }
                },
            );
        ensure!(
            predecessor_first.load(Ordering::Relaxed),
            "a DRG parent table needs a graph whose first parent is the predecessor"
        );
        Ok(Self { entries })
    }

    /// Nodes covered by the table.
    pub fn nodes(&self) -> usize {
        self.entries.len()
    }

    /// The parents of nodes `first..first + parents.len()`, predecessor first.
    #[inline]
    pub fn fill(&self, first: usize, parents: &mut [[u32; DRG_DEGREE]]) {
        let stored = &self.entries[first..first + parents.len()];
        for (index, (parents, stored)) in parents.iter_mut().zip(stored.iter()).enumerate() {
            parents[0] = (first + index).saturating_sub(1) as u32;
            parents[1..].copy_from_slice(stored);
        }
    }
}

/// The parents of nodes `first..first + parents.len()`, from the table if there is one.
#[inline]
pub(super) fn fill_parents<G: LabelGraph>(
    graph: &G,
    table: Option<&DrgParentTable>,
    first: usize,
    parents: &mut [[u32; DRG_DEGREE]],
) {
    match table {
        Some(table) => table.fill(first, parents),
        None => graph.drg_parents_batch(first, parents),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use sha2::{Digest, Sha256};

    fn sampler(porep_id: [u8; 32]) -> DrgSampler {
        // `derive_drg_seed`: the first 28 bytes of SHA-256("Filecoin_DRSample" || porep id).
        let digest = Sha256::new()
            .chain_update(b"Filecoin_DRSample")
            .chain_update(porep_id)
            .finalize();
        let mut seed = [0u8; 28];
        seed.copy_from_slice(&digest[..28]);
        DrgSampler::new(&seed)
    }

    #[test]
    fn integer_logarithm_matches_the_float_one() {
        for x in 2u64..10_000 {
            assert_eq!(ceil_log2(x), (x as f64).log2().ceil() as u64);
        }
        for exponent in 2..=40u32 {
            for x in [
                (1u64 << exponent) - 1,
                1u64 << exponent,
                (1u64 << exponent) + 1,
            ] {
                assert_eq!(ceil_log2(x), (x as f64).log2().ceil() as u64, "x {}", x);
            }
        }
    }

    // Values of `BucketGraph::parents` from the independent model
    // `FABLE-PLAN-SDRv2-scripts/sdrv2_model.py` (porep id `[5; 32]`).
    #[test]
    fn parents_match_the_model() {
        let sampler = sampler([5u8; 32]);
        let mut parents = [[0u32; DRG_DEGREE]; 64];
        sampler.parents(0, &mut parents);
        assert_eq!(parents[0], [0; 6]);
        assert_eq!(parents[1], [0; 6]);
        assert_eq!(parents[2], [1, 0, 1, 1, 1, 1]);
        assert_eq!(parents[3], [2, 1, 1, 1, 0, 2]);
        assert_eq!(parents[7], [6, 6, 5, 6, 6, 2]);
        assert_eq!(parents[10], [9, 9, 7, 1, 8, 5]);
        assert_eq!(parents[33], [32, 28, 32, 31, 8, 32]);
        assert_eq!(parents[63], [62, 62, 61, 23, 16, 22]);
    }

    #[test]
    fn batches_do_not_change_the_parents() {
        let sampler = sampler([9u8; 32]);
        let total = 5 * LANES + 3;
        for first in [
            0usize,
            1,
            2,
            3,
            1000,
            (1 << 30) - 7,
            u32::MAX as usize - total,
        ] {
            let mut together = vec![[0u32; DRG_DEGREE]; total];
            sampler.parents(first, &mut together);
            for (index, expected) in together.iter().enumerate() {
                let node = first + index;
                let mut alone = [[0u32; DRG_DEGREE]; 1];
                sampler.parents(node, &mut alone);
                assert_eq!(alone[0], *expected, "node {}", node);
                if node >= 2 {
                    assert_eq!(expected[0] as usize, node - 1);
                }
                assert!(expected
                    .iter()
                    .all(|parent| (*parent as usize) < node.max(1)));
            }
        }
    }

    struct TestGraph {
        nodes: usize,
        sampler: DrgSampler,
        predecessor_first: bool,
    }

    // SAFETY: `word_source` is the identity; the decoder is not run on this graph.
    unsafe impl LabelGraph for TestGraph {
        fn nodes(&self) -> usize {
            self.nodes
        }

        fn drg_parents(&self, node: usize, parents: &mut [u32; DRG_DEGREE]) {
            self.sampler.parents(node, std::slice::from_mut(parents));
            if !self.predecessor_first {
                parents.swap(0, 5);
            }
        }

        fn word_source(&self, word: u64) -> u64 {
            word
        }
    }

    #[test]
    fn table_returns_the_parents_of_the_graph() {
        let nodes = 3 * BUILD_CHUNK_NODES + 17;
        let graph = TestGraph {
            nodes,
            sampler: sampler([3u8; 32]),
            predecessor_first: true,
        };
        let table = DrgParentTable::build(&graph).expect("table");
        assert_eq!(table.nodes(), nodes);
        assert_eq!(DrgParentTable::bytes(nodes), nodes as u64 * 20);
        for first in [0usize, 1, 2, BUILD_CHUNK_NODES - 5, nodes - 40] {
            let mut from_table = [[0u32; DRG_DEGREE]; 40];
            let mut from_graph = [[0u32; DRG_DEGREE]; 40];
            fill_parents(&graph, Some(&table), first, &mut from_table);
            fill_parents(&graph, None, first, &mut from_graph);
            assert_eq!(from_table, from_graph, "first {}", first);
        }

        let unordered = TestGraph {
            nodes,
            sampler: sampler([3u8; 32]),
            predecessor_first: false,
        };
        assert!(DrgParentTable::build(&unordered).is_err());
    }
}
