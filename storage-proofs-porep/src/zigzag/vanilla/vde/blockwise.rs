//! Reverse-topological block decode. Only the output block is mutable while
//! workers read the sector; copy-back happens after the Rayon join/barrier.
use std::sync::atomic::AtomicBool;

use anyhow::{ensure, Context, Result};
use filecoin_hashers::{Domain, Hasher};
use rayon::prelude::*;
use storage_proofs_core::{
    drgraph::Graph, parameter_cache::ParameterSetMetadata, settings::SETTINGS, util::NODE_SIZE,
};

use super::super::{
    decode_parents::{DecodeParentCursor, DecodeParentReader, DecodeSession, ReadStats},
    graph::ZigZagGraph,
    parent_table::ZigZagParentTable,
};
use super::{decode_from_parents, ensure_graph_data_len, fill_parents, DecodeScratch};

mod schedule;
use schedule::{check_cancelled, validate_block, visit_blocks};

/// Decode one layer in a private state with a separate, bounded output block.
/// An error/cancellation/panic may leave earlier blocks decoded. Discard the
/// private state and retry from sealed input; never resume a partially decoded layer.
pub fn decode_in_place_with_block<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &mut [u8],
    block: &mut [u8],
    cancelled: Option<&AtomicBool>,
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    decode_with_session(
        graph,
        replica_id,
        data,
        block,
        cancelled,
        &mut DecodeSession::new()?,
    )
}

pub(in crate::zigzag::vanilla) fn decode_with_session<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &mut [u8],
    block: &mut [u8],
    cancelled: Option<&AtomicBool>,
    session: &mut DecodeSession,
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    ensure_graph_data_len(graph, data)?;
    validate_block(data.len(), block.len())?;
    check_cancelled(cancelled)?;
    let config = session.config;
    let use_table = SETTINGS.use_zigzag_parent_cache;
    let mut effective_config = config;
    if !use_table {
        effective_config.parent_reader = DecodeParentReader::Legacy;
    }
    let layout = effective_config.layout(
        block.len().min(data.len()) / NODE_SIZE,
        graph.degree(),
        rayon::current_num_threads(),
    )?;
    let source = if use_table && config.parent_reader != DecodeParentReader::Legacy {
        Some(session.source(graph)?)
    } else {
        None
    };
    let mut workers = (0..layout.readers)
        .map(|_| -> Result<_> {
            Ok(Worker {
                cursor: source.map(|s| s.reader(layout.buffer_nodes)).transpose()?,
                legacy: if use_table && source.is_none() {
                    Some(ZigZagParentTable::new(graph)?)
                } else {
                    None
                },
                scratch: DecodeScratch::new(graph.degree()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let stripe_bytes = layout
        .stripe_nodes
        .checked_mul(NODE_SIZE)
        .context("decode stripe overflow")?;
    let times = visit_blocks(
        data,
        block,
        graph.forward(),
        cancelled,
        |state, output, range| {
            output
                .par_chunks_mut(stripe_bytes)
                .zip(workers.par_iter_mut())
                .enumerate()
                .try_for_each(|(stripe, (chunk, worker))| -> Result<()> {
                    check_cancelled(cancelled)?;
                    let base = range.start / NODE_SIZE + stripe * layout.stripe_nodes;
                    for (index, out) in chunk.chunks_exact_mut(NODE_SIZE).enumerate() {
                        let node = base + index;
                        if let Some(reader) = worker.cursor.as_mut() {
                            reader.read_into(node, &mut worker.scratch.parents)?;
                        } else {
                            fill_parents(
                                graph,
                                &mut worker.legacy,
                                node,
                                &mut worker.scratch.parents,
                            )?;
                        }
                        // Enforce the ordering invariant before reading a partially overwritten
                        // state. Preserve duplicate parents and the special boundary-node KDF.
                        let parents = &worker.scratch.parents;
                        let boundary = if graph.forward() { 0 } else { graph.size() - 1 };
                        ensure!(
                            if node == boundary {
                                parents.iter().all(|p| *p as usize == node)
                            } else {
                                parents.iter().all(|p| {
                                    let p = *p as usize;
                                    p < graph.size()
                                        && if graph.forward() { p < node } else { p > node }
                                })
                            },
                            "ZigZag parent violates block decode order at node {node}"
                        );
                        decode_from_parents::<H>(replica_id, state, node, &mut worker.scratch)
                            .write_bytes(out)?;
                    }
                    Ok(())
                })
        },
    )?;
    let io = source.map(|_| {
        workers
            .iter()
            .filter_map(|w| w.cursor.as_ref())
            .fold(ReadStats::default(), |stats, reader| {
                stats.merge(reader.stats)
            })
    });
    log::info!(target: "zigzag_unseal", "zigzag_decode_parents {}", serde_json::json!({
        "event": "layer", "reader": if use_table { serde_json::to_value(config.parent_reader)? } else { "graph".into() },
        "requested_reader": config.parent_reader,
        "reversed": graph.reversed(), "nodes": graph.size(), "layout": layout, "io": io,
        "remaps": if source.is_some() { Some(0) } else { None::<usize> },
    }));
    log::info!(target: "zigzag_unseal", "zigzag_decode_state {}", serde_json::json!({
        "event": "layer", "mode": "blockwise", "reversed": graph.reversed(),
        "state_bytes": data.len(), "block_bytes": block.len(), "blocks": times.blocks,
        "decode_join_ns": times.decode_join_ns, "copy_back_ns": times.copy_back_ns,
    }));
    Ok(())
}

struct Worker<'a> {
    cursor: Option<DecodeParentCursor<'a>>,
    legacy: Option<ZigZagParentTable>,
    scratch: DecodeScratch,
}

#[cfg(test)]
mod tests;
