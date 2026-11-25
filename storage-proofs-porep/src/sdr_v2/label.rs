//! The SDR v2 label: parameters, the graph it needs, and the key of a node.
//!
//! Working specification (the report leaves the byte level open; see `FABLE-PLAN-SDRv2.md`,
//! section 2). Everything that fixes the bytes of a replica is in this file and in
//! [`super::threefish`]:
//!
//! * a layer is `nodes` labels of 32 bytes; component `a` of a label is its bytes `4a..4a + 4`,
//!   so a layer is also `8 * nodes` four-byte words and node `i` owns words `8i..8i + 8`;
//! * the PRP input of node `i` in layer `j > 1` is, for each of its words `w`, word
//!   `word_source(w)` of layer `j - 1`; in layer 1 it is the data of node `i`;
//! * `key = SHA-256(replica_id || layer || node || parent labels)`, with `layer` (1-based) and
//!   `node` (0-based) as big-endian `u64` in the last 8 bytes of a zeroed 32-byte field, and the
//!   labels of the 6 DRG parents repeated in order, predecessor first, until the hash input needs
//!   exactly `key_blocks` SHA-256 compressions; node 0 hashes no label;
//! * `label = Threefish-256(key, input)` with a zero tweak.

use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};

/// Bytes of a label.
pub const NODE_BYTES: usize = 32;
/// Components of a label, one per expander edge.
pub const COMPONENTS: usize = 8;
/// Bytes of a component.
pub const COMPONENT_BYTES: usize = NODE_BYTES / COMPONENTS;
/// DRG parents of a node, all in its own layer.
pub const DRG_DEGREE: usize = 6;

/// The shortest key hash: every DRG parent is hashed once.
pub const MIN_KEY_BLOCKS: usize = 5;
/// The longest key hash accepted here (an SDR label needs 20 compressions).
pub const MAX_KEY_BLOCKS: usize = 21;

/// Bytes hashed before the parent labels: replica id, layer, node.
const HEADER_BYTES: usize = 3 * NODE_BYTES;
/// SHA-256 padding: the `0x80` byte and the 64-bit length.
const PADDING_BYTES: usize = 9;

/// The parameters that change the bytes of a replica.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdrV2Params {
    layers: usize,
    key_blocks: usize,
}

impl SdrV2Params {
    pub fn new(layers: usize, key_blocks: usize) -> Result<Self> {
        ensure!(
            (1..=64).contains(&layers),
            "SDR v2 layers must be between 1 and 64, got {layers}"
        );
        ensure!(
            (MIN_KEY_BLOCKS..=MAX_KEY_BLOCKS).contains(&key_blocks),
            "SDR v2 key blocks must be between {MIN_KEY_BLOCKS} and {MAX_KEY_BLOCKS}, got {key_blocks}"
        );
        Ok(Self { layers, key_blocks })
    }

    /// Number of layers; the last one is the replica.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// SHA-256 compressions of one key.
    pub fn key_blocks(&self) -> usize {
        self.key_blocks
    }

    /// Parent labels hashed into one key: as many as fit into `key_blocks` compressions.
    pub fn parent_labels(&self) -> usize {
        (64 * self.key_blocks - HEADER_BYTES - PADDING_BYTES) / NODE_BYTES
    }
}

/// The graph of one SDR v2 layer and of the edges to the layer before it.
///
/// # Safety
///
/// The decoder writes through raw pointers and relies on this contract:
///
/// * `word_source` is a bijection of `0..8 * nodes()`;
/// * `drg_parents` fills indices below `nodes()`, and for `node > 0` indices below `node`.
pub unsafe trait LabelGraph: Sync {
    /// Labels per layer.
    fn nodes(&self) -> usize;

    /// The DRG parents of `node`, predecessor first. Not used for node 0.
    fn drg_parents(&self, node: usize, parents: &mut [u32; DRG_DEGREE]);

    /// The DRG parents of nodes `first..first + parents.len()`. Implement it when a batch is
    /// cheaper than its nodes one by one.
    fn drg_parents_batch(&self, first: usize, parents: &mut [[u32; DRG_DEGREE]]) {
        for (index, parents) in parents.iter_mut().enumerate() {
            self.drg_parents(first + index, parents);
        }
    }

    /// The word of the previous layer that is input word `word` of this layer.
    fn word_source(&self, word: u64) -> u64;
}

/// A boundary of one pass over a layer, for measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerEvent {
    Start {
        operation: &'static str,
        /// 1-based layer.
        layer: usize,
    },
    End {
        operation: &'static str,
        layer: usize,
    },
}

/// Derives the keys of one layer.
#[derive(Clone)]
pub struct LayerKeyer {
    /// SHA-256 state after `replica_id || layer`, exactly one compression.
    prefix: Sha256,
    parent_labels: usize,
}

fn field(value: u64) -> [u8; NODE_BYTES] {
    let mut bytes = [0u8; NODE_BYTES];
    bytes[NODE_BYTES - 8..].copy_from_slice(&value.to_be_bytes());
    bytes
}

impl LayerKeyer {
    /// `layer` is 1-based.
    pub fn new(replica_id: &[u8; 32], layer: usize, params: &SdrV2Params) -> Self {
        let mut prefix = Sha256::new();
        prefix.update(replica_id);
        prefix.update(field(layer as u64));
        Self {
            prefix,
            parent_labels: params.parent_labels(),
        }
    }

    /// The key of `node`. `labels` holds the labels of the node's layer and must already contain
    /// those of its DRG parents.
    #[inline]
    pub fn key(&self, node: usize, parents: &[u32; DRG_DEGREE], labels: &[u8]) -> [u8; 32] {
        let mut hasher = self.prefix.clone();
        hasher.update(field(node as u64));
        if node > 0 {
            for index in 0..self.parent_labels {
                let start = parents[index % DRG_DEGREE] as usize * NODE_BYTES;
                hasher.update(&labels[start..start + NODE_BYTES]);
            }
        }
        hasher.finalize().into()
    }
}

/// Asks the CPU to load the cache line at `pointer`. A hint: it never faults.
#[inline(always)]
pub(super) fn prefetch(pointer: *const u8) {
    #[cfg(target_arch = "x86_64")]
    #[allow(unused_unsafe)]
    unsafe {
        use std::arch::x86_64 as x86;

        x86::_mm_prefetch(pointer as *const i8, x86::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = pointer;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_labels_fill_the_key_blocks() {
        let expected = [
            (5, 6),
            (8, 12),
            (9, 14),
            (12, 20),
            (14, 24),
            (15, 26),
            (16, 28),
        ];
        for (key_blocks, parent_labels) in expected {
            let params = SdrV2Params::new(11, key_blocks).expect("valid parameters");
            assert_eq!(params.parent_labels(), parent_labels);
            let hashed = HEADER_BYTES + parent_labels * NODE_BYTES + PADDING_BYTES;
            assert_eq!(hashed.div_ceil(64), key_blocks);
            // One more label would need one more compression.
            assert_eq!((hashed + NODE_BYTES).div_ceil(64), key_blocks + 1);
        }
        assert!(SdrV2Params::new(11, MIN_KEY_BLOCKS - 1).is_err());
        assert!(SdrV2Params::new(11, MAX_KEY_BLOCKS + 1).is_err());
        assert!(SdrV2Params::new(0, MIN_KEY_BLOCKS).is_err());
    }

    #[test]
    fn key_is_sha256_of_the_documented_input() {
        let replica_id = [7u8; 32];
        let params = SdrV2Params::new(11, 8).expect("valid parameters");
        let labels: Vec<u8> = (0..16 * NODE_BYTES)
            .map(|index| (index * 7 + 3) as u8)
            .collect();
        let parents = [9u32, 2, 4, 4, 0, 8];
        let layer = 3usize;
        let node = 10usize;

        let mut input = Vec::new();
        input.extend_from_slice(&replica_id);
        input.extend_from_slice(&[0u8; 24]);
        input.extend_from_slice(&(layer as u64).to_be_bytes());
        input.extend_from_slice(&[0u8; 24]);
        input.extend_from_slice(&(node as u64).to_be_bytes());
        for index in 0..params.parent_labels() {
            let start = parents[index % DRG_DEGREE] as usize * NODE_BYTES;
            input.extend_from_slice(&labels[start..start + NODE_BYTES]);
        }
        assert_eq!(input.len(), 96 + 12 * 32);
        let expected: [u8; 32] = Sha256::digest(&input).into();

        let keyer = LayerKeyer::new(&replica_id, layer, &params);
        assert_eq!(keyer.key(node, &parents, &labels), expected);

        // Node 0 has no DRG parent: only the header is hashed.
        let mut header = input[..64].to_vec();
        header.extend_from_slice(&[0u8; 32]);
        let node_zero: [u8; 32] = Sha256::digest(&header).into();
        assert_eq!(keyer.key(0, &parents, &labels), node_zero);
    }

    // Value from the independent model `FABLE-PLAN-SDRv2-scripts/sdrv2_model.py golden`.
    #[test]
    fn key_matches_the_model() {
        let mut replica_id = [0u8; 32];
        for (index, byte) in replica_id.iter_mut().enumerate() {
            *byte = index as u8 + 1;
        }
        let labels: Vec<u8> = (0..64 * NODE_BYTES)
            .map(|index| ((index * 31 + (index >> 3) + 17) & 0xff) as u8)
            .collect();
        let params = SdrV2Params::new(11, 5).expect("valid parameters");
        let keyer = LayerKeyer::new(&replica_id, 3, &params);
        // DRG parents of node 7 in the 64-node graph of porep id `[5; 32]`.
        let key = keyer.key(7, &MODEL_PARENTS_OF_NODE_7, &labels);
        let hex: String = key.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(hex, MODEL_KEY_OF_NODE_7);
    }

    const MODEL_PARENTS_OF_NODE_7: [u32; DRG_DEGREE] = [6, 6, 5, 6, 6, 2];
    const MODEL_KEY_OF_NODE_7: &str =
        "2c6844bcc97cc2a3ddfa6c8ffe0b32ea944c51600c1fe14b829b53e3fb550f58";
}
