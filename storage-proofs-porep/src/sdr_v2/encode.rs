//! SDR v2 sealing of one sector: the simple encoder.
//!
//! Each layer is one parallel pass that gathers the PRP inputs from the previous layer and one
//! sequential pass that turns inputs into labels. The sequential pass is a single thread: this
//! encoder exists to produce replicas for tests and unseal benchmarks, not to measure sealing.

use std::convert::TryInto;

use anyhow::{ensure, Result};
use log::info;
use rayon::prelude::*;

use super::drg::{fill_parents, DrgParentTable};
use super::label::{
    prefetch, LabelGraph, LayerEvent, LayerKeyer, SdrV2Params, COMPONENT_BYTES, DRG_DEGREE,
    NODE_BYTES,
};
use super::threefish;

/// Words gathered by one parallel task.
const GATHER_CHUNK_WORDS: usize = 1 << 14;
/// Words whose sources are computed and prefetched before they are copied.
const GATHER_BATCH: usize = 32;
/// Nodes whose DRG parents are computed in parallel ahead of the sequential pass.
const CHAIN_WINDOW: usize = 1 << 14;
/// Nodes whose DRG parents one parallel task computes.
const CHAIN_PARENT_BATCH: usize = 256;
/// How far ahead of the hashed node the labels of DRG parents are prefetched.
const CHAIN_LOOKAHEAD: usize = 16;

/// Seals `data` (fr32-padded sector data) in place: on return it holds the replica.
///
/// `scratch` is a second buffer of the same length; its contents are overwritten. With a
/// `parent_table`, the DRG parents are read from it instead of being computed in every layer.
pub fn encode_layers<G: LabelGraph>(
    graph: &G,
    params: &SdrV2Params,
    replica_id: &[u8; 32],
    data: &mut [u8],
    scratch: &mut [u8],
    parent_table: Option<&DrgParentTable>,
    observer: &mut dyn FnMut(LayerEvent),
) -> Result<()> {
    let nodes = graph.nodes();
    ensure!(nodes > 0, "SDR v2 graph has no nodes");
    ensure!(
        data.len() == nodes * NODE_BYTES && scratch.len() == data.len(),
        "SDR v2 buffers must hold {} labels: data {} bytes, scratch {} bytes",
        nodes,
        data.len(),
        scratch.len()
    );
    if let Some(table) = parent_table {
        ensure!(
            table.nodes() == nodes,
            "DRG parent table covers {} nodes, the graph has {}",
            table.nodes(),
            nodes
        );
    }

    // Layer 1: the PRP input of a node is its data.
    let mut labels_in_data = true;
    for layer in 1..=params.layers() {
        let keyer = LayerKeyer::new(replica_id, layer, params);
        let labels: &mut [u8] = if layer == 1 {
            &mut *data
        } else {
            let (previous, current): (&[u8], &mut [u8]) = if labels_in_data {
                (&*data, &mut *scratch)
            } else {
                (&*scratch, &mut *data)
            };
            labels_in_data = !labels_in_data;

            let started = std::time::Instant::now();
            observer(LayerEvent::Start {
                operation: "sdrv2_encode_gather",
                layer,
            });
            gather_inputs(graph, previous, current);
            observer(LayerEvent::End {
                operation: "sdrv2_encode_gather",
                layer,
            });
            info!(target: "sdrv2", "phase=encode_gather layer={} elapsed_ms={}", layer, started.elapsed().as_millis());
            current
        };

        let started = std::time::Instant::now();
        observer(LayerEvent::Start {
            operation: "sdrv2_encode_chain",
            layer,
        });
        label_chain(graph, parent_table, &keyer, labels);
        observer(LayerEvent::End {
            operation: "sdrv2_encode_chain",
            layer,
        });
        info!(target: "sdrv2", "phase=encode_chain layer={} elapsed_ms={}", layer, started.elapsed().as_millis());
    }

    if !labels_in_data {
        data.copy_from_slice(scratch);
    }
    Ok(())
}

/// `current[word] = previous[word_source(word)]` for every four-byte word.
fn gather_inputs<G: LabelGraph>(graph: &G, previous: &[u8], current: &mut [u8]) {
    current
        .par_chunks_mut(GATHER_CHUNK_WORDS * COMPONENT_BYTES)
        .enumerate()
        .for_each(|(chunk, output)| {
            let first_word = chunk * GATHER_CHUNK_WORDS;
            let mut sources = [0usize; GATHER_BATCH];
            for (batch, output) in output
                .chunks_mut(GATHER_BATCH * COMPONENT_BYTES)
                .enumerate()
            {
                let base = (first_word + batch * GATHER_BATCH) as u64;
                let count = output.len() / COMPONENT_BYTES;
                for (index, source) in sources.iter_mut().enumerate().take(count) {
                    *source = graph.word_source(base + index as u64) as usize * COMPONENT_BYTES;
                    prefetch(previous.as_ptr().wrapping_add(*source));
                }
                for (word, source) in output.chunks_exact_mut(COMPONENT_BYTES).zip(sources.iter()) {
                    word.copy_from_slice(&previous[*source..*source + COMPONENT_BYTES]);
                }
            }
        });
}

/// Replaces every PRP input in `labels` by its label, in node order.
fn label_chain<G: LabelGraph>(
    graph: &G,
    parent_table: Option<&DrgParentTable>,
    keyer: &LayerKeyer,
    labels: &mut [u8],
) {
    let nodes = graph.nodes();
    let mut parents = vec![[0u32; DRG_DEGREE]; CHAIN_WINDOW.min(nodes)];

    let mut start = 0;
    while start < nodes {
        let count = CHAIN_WINDOW.min(nodes - start);
        // The parents do not depend on labels, so they are prepared off the sequential path.
        match parent_table {
            Some(_) => fill_parents(graph, parent_table, start, &mut parents[..count]),
            None => parents[..count]
                .par_chunks_mut(CHAIN_PARENT_BATCH)
                .enumerate()
                .for_each(|(batch, parents)| {
                    graph.drg_parents_batch(start + batch * CHAIN_PARENT_BATCH, parents)
                }),
        }

        for index in 0..count {
            if index + CHAIN_LOOKAHEAD < count {
                for parent in parents[index + CHAIN_LOOKAHEAD].iter() {
                    prefetch(labels.as_ptr().wrapping_add(*parent as usize * NODE_BYTES));
                }
            }
            let node = start + index;
            let key = keyer.key(node, &parents[index], labels);
            let offset = node * NODE_BYTES;
            let label: &mut [u8; NODE_BYTES] = (&mut labels[offset..offset + NODE_BYTES])
                .try_into()
                .expect("a label is 32 bytes");
            threefish::encrypt_label(&key, label);
        }
        start += count;
    }
}
