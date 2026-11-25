//! SDR v2 unsealing of one sector: the parallel decoder.
//!
//! Layers are decoded from the replica down to layer 1. When layer `j` is known, the key of every
//! node of layer `j` can be computed at the same time, because its DRG parents are in layer `j`.
//! Decrypting a label gives the PRP input of its node: 8 components, each of which is one
//! component of a label of layer `j - 1` (in layer 1, the data of the node). So one pass over
//! layer `j` writes every word of layer `j - 1` exactly once.

use std::convert::TryInto;

use anyhow::{ensure, Result};
use log::info;
use rayon::prelude::*;

use super::drg::{fill_parents, DrgParentTable};
use super::label::{
    prefetch, LabelGraph, LayerEvent, LayerKeyer, SdrV2Params, COMPONENTS, COMPONENT_BYTES,
    DRG_DEGREE, NODE_BYTES,
};
use super::threefish;

/// Nodes decoded by one parallel task.
const CHUNK_NODES: usize = 1 << 12;
/// Largest number of nodes whose memory accesses are prefetched together.
pub const MAX_DECODE_BATCH: usize = 64;
/// Default for [`decode_layers`]'s `batch`.
pub const DEFAULT_DECODE_BATCH: usize = 8;

/// The output layer, shared by the decoding threads.
#[derive(Clone, Copy)]
struct SharedLayer {
    pointer: *mut u8,
    len: usize,
}

// SAFETY: threads write disjoint byte ranges (see `decode_layer`) and never read.
unsafe impl Send for SharedLayer {}
unsafe impl Sync for SharedLayer {}

impl SharedLayer {
    /// # Safety
    ///
    /// No other thread may write or read `offset..offset + bytes.len()` at the same time.
    #[inline(always)]
    unsafe fn write(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.len, "write outside the layer");
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.pointer.add(offset), bytes.len());
    }
}

/// Decodes a replica.
///
/// `replica` holds the replica and `scratch` is a second buffer of the same length; both are
/// overwritten. The returned slice is the one of the two that holds the fr32-padded sector data.
/// `batch` is the number of nodes whose parent labels and output words are prefetched together.
/// With a `parent_table`, the DRG parents are read from it instead of being computed in every
/// layer.
#[allow(clippy::too_many_arguments)]
pub fn decode_layers<'a, G: LabelGraph>(
    graph: &G,
    params: &SdrV2Params,
    replica_id: &[u8; 32],
    replica: &'a mut [u8],
    scratch: &'a mut [u8],
    batch: usize,
    parent_table: Option<&DrgParentTable>,
    observer: &mut dyn FnMut(LayerEvent),
) -> Result<&'a mut [u8]> {
    let nodes = graph.nodes();
    ensure!(nodes > 0, "SDR v2 graph has no nodes");
    ensure!(
        replica.len() == nodes * NODE_BYTES && scratch.len() == replica.len(),
        "SDR v2 buffers must hold {} labels: replica {} bytes, scratch {} bytes",
        nodes,
        replica.len(),
        scratch.len()
    );
    ensure!(
        (1..=MAX_DECODE_BATCH).contains(&batch),
        "SDR v2 decode batch must be between 1 and {MAX_DECODE_BATCH}, got {batch}"
    );
    if let Some(table) = parent_table {
        ensure!(
            table.nodes() == nodes,
            "DRG parent table covers {} nodes, the graph has {}",
            table.nodes(),
            nodes
        );
    }

    let mut labels = replica;
    let mut output = scratch;
    for layer in (1..=params.layers()).rev() {
        let keyer = LayerKeyer::new(replica_id, layer, params);
        let started = std::time::Instant::now();
        observer(LayerEvent::Start {
            operation: "sdrv2_decode_layer",
            layer,
        });
        decode_layer(
            graph,
            parent_table,
            &keyer,
            labels,
            output,
            layer > 1,
            batch,
        );
        observer(LayerEvent::End {
            operation: "sdrv2_decode_layer",
            layer,
        });
        info!(target: "sdrv2", "phase=decode_layer layer={} elapsed_ms={}", layer, started.elapsed().as_millis());
        std::mem::swap(&mut labels, &mut output);
    }
    Ok(labels)
}

/// One pass over a layer. With `scatter`, component `a` of the PRP input of node `i` goes to word
/// `word_source(8i + a)` of `output`; without it (layer 1), the input is the data of node `i`.
fn decode_layer<G: LabelGraph>(
    graph: &G,
    parent_table: Option<&DrgParentTable>,
    keyer: &LayerKeyer,
    labels: &[u8],
    output: &mut [u8],
    scatter: bool,
    batch: usize,
) {
    let nodes = graph.nodes();
    let shared = SharedLayer {
        pointer: output.as_mut_ptr(),
        len: output.len(),
    };

    (0..nodes.div_ceil(CHUNK_NODES))
        .into_par_iter()
        .for_each_init(
            || vec![[0u32; DRG_DEGREE]; CHUNK_NODES],
            |chunk_parents, chunk| {
                let chunk_first = chunk * CHUNK_NODES;
                let end = (chunk_first + CHUNK_NODES).min(nodes);
                // The parents of the whole chunk: they do not depend on labels.
                let chunk_parents = &mut chunk_parents[..end - chunk_first];
                fill_parents(graph, parent_table, chunk_first, chunk_parents);
                let mut targets = [[0usize; COMPONENTS]; MAX_DECODE_BATCH];

                let mut first = chunk_first;
                while first < end {
                    let count = batch.min(end - first);
                    let parents = &chunk_parents[first - chunk_first..first - chunk_first + count];

                    // Where this batch reads and writes. The cache lines are requested before the
                    // hashing starts.
                    for index in 0..count {
                        let node = first + index;
                        // The predecessor is next to the node; the others are anywhere before it.
                        for parent in parents[index].iter().skip(1) {
                            prefetch(labels.as_ptr().wrapping_add(*parent as usize * NODE_BYTES));
                        }
                        if scatter {
                            for (component, target) in targets[index].iter_mut().enumerate() {
                                let word = (node * COMPONENTS + component) as u64;
                                *target = graph.word_source(word) as usize * COMPONENT_BYTES;
                                prefetch(shared.pointer.wrapping_add(*target) as *const u8);
                            }
                        }
                    }

                    for index in 0..count {
                        let node = first + index;
                        let key = keyer.key(node, &parents[index], labels);
                        let offset = node * NODE_BYTES;
                        let mut input: [u8; NODE_BYTES] = labels[offset..offset + NODE_BYTES]
                            .try_into()
                            .expect("a label is 32 bytes");
                        threefish::decrypt_label(&key, &mut input);

                        if scatter {
                            for (component, target) in targets[index].iter().enumerate() {
                                let start = component * COMPONENT_BYTES;
                                // SAFETY: `word_source` is a bijection (contract of `LabelGraph`),
                                // so every word of `output` is written once, by this thread only.
                                unsafe {
                                    shared.write(*target, &input[start..start + COMPONENT_BYTES]);
                                }
                            }
                        } else {
                            // SAFETY: the label of node `i` is written by the task that owns it.
                            unsafe {
                                shared.write(offset, &input);
                            }
                        }
                    }
                    first += count;
                }
            },
        );
}
