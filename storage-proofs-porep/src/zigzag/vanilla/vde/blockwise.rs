//! Reverse-topological block decode. Only the output block is mutable while
//! workers read the sector; copy-back happens after the Rayon join/barrier.
use std::{ops::Range, sync::atomic::AtomicBool, time::Instant};

use anyhow::{ensure, Context, Result};
use filecoin_hashers::{Domain, Hasher};
use rayon::prelude::*;
use storage_proofs_core::{
    drgraph::Graph, parameter_cache::ParameterSetMetadata, settings::SETTINGS, util::NODE_SIZE,
};

use super::super::{
    decode_parents::{
        DecodeParentCursor, DecodeParentReader, DecodeParentSource, DecodeSession, ReadStats,
        ReaderLayout,
    },
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
    let mut decoder = LayerDecoder::new(graph, block.len().min(data.len()) / NODE_SIZE, session)?;
    let times = visit_blocks(
        data,
        block,
        graph.forward(),
        cancelled,
        |state, output, range| {
            decoder.decode(
                graph,
                replica_id,
                state,
                output,
                range.start / NODE_SIZE,
                cancelled,
            )
        },
    )?;
    decoder.log_parents(graph)?;
    log::info!(target: "zigzag_unseal", "zigzag_decode_state {}", serde_json::json!({
        "event": "layer", "mode": "blockwise", "reversed": graph.reversed(),
        "state_bytes": data.len(), "block_bytes": block.len(), "blocks": times.blocks,
        "decode_join_ns": times.decode_join_ns, "copy_back_ns": times.copy_back_ns,
    }));
    Ok(())
}

/// The last layer reads immutable state, so either graph orientation can emit
/// increasing offsets. Only the requested padded nodes are decoded. All Rayon
/// readers join before the sink is called, and the sink applies backpressure.
#[allow(clippy::too_many_arguments)]
pub(in crate::zigzag::vanilla) fn stream_with_session<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &[u8],
    block: &mut [u8],
    range: Range<usize>,
    cancelled: Option<&AtomicBool>,
    session: &mut DecodeSession,
    mut emit: impl FnMut(usize, &[u8]) -> Result<()>,
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    ensure_graph_data_len(graph, data)?;
    validate_block(data.len(), block.len())?;
    ensure!(
        range.start <= range.end
            && range.end <= data.len()
            && range.start.is_multiple_of(NODE_SIZE)
            && range.end.is_multiple_of(NODE_SIZE),
        "invalid final-layer padded range"
    );
    check_cancelled(cancelled)?;
    if range.is_empty() {
        return Ok(());
    }
    let mut decoder = LayerDecoder::new(graph, block.len().min(range.len()) / NODE_SIZE, session)?;
    let mut blocks = 0;
    let mut decode_join_ns = 0;
    for start in (range.start..range.end).step_by(block.len()) {
        check_cancelled(cancelled)?;
        let count = block.len().min(range.end - start);
        let started = Instant::now();
        decoder.decode(
            graph,
            replica_id,
            data,
            &mut block[..count],
            start / NODE_SIZE,
            cancelled,
        )?;
        decode_join_ns += started.elapsed().as_nanos();
        check_cancelled(cancelled)?;
        emit(start, &block[..count])?;
        blocks += 1;
    }
    check_cancelled(cancelled)?;
    decoder.log_parents(graph)?;
    log::info!(target: "zigzag_unseal", "zigzag_decode_state {}", serde_json::json!({
        "event": "final_layer", "mode": "stream", "reversed": graph.reversed(),
        "state_bytes": data.len(), "block_bytes": block.len(), "blocks": blocks,
        "decoded_bytes": range.len(), "decode_join_ns": decode_join_ns, "copy_back_ns": 0,
    }));
    Ok(())
}

struct LayerDecoder<'a> {
    source: Option<&'a DecodeParentSource>,
    workers: Vec<Worker<'a>>,
    layout: ReaderLayout,
    reader: DecodeParentReader,
    use_table: bool,
}

impl<'a> LayerDecoder<'a> {
    fn new<H, G>(
        graph: &ZigZagGraph<H, G>,
        block_nodes: usize,
        session: &'a mut DecodeSession,
    ) -> Result<Self>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Sync + Send,
    {
        let config = session.config;
        let use_table = SETTINGS.use_zigzag_parent_cache;
        let mut effective_config = config;
        if !use_table {
            effective_config.parent_reader = DecodeParentReader::Legacy;
        }
        let layout =
            effective_config.layout(block_nodes, graph.degree(), rayon::current_num_threads())?;
        let source = if use_table && config.parent_reader != DecodeParentReader::Legacy {
            Some(session.source(graph)?)
        } else {
            None
        };
        let workers = (0..layout.readers)
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
        Ok(Self {
            source,
            workers,
            layout,
            reader: config.parent_reader,
            use_table,
        })
    }

    fn decode<H, G>(
        &mut self,
        graph: &ZigZagGraph<H, G>,
        replica_id: &H::Domain,
        state: &[u8],
        output: &mut [u8],
        base_node: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Sync + Send,
    {
        let stripe_nodes = self.layout.stripe_nodes;
        let stripe_bytes = stripe_nodes
            .checked_mul(NODE_SIZE)
            .context("decode stripe overflow")?;
        // zip must never silently leave an output suffix uninitialized.
        ensure!(
            output.len().div_ceil(stripe_bytes) <= self.workers.len(),
            "decode worker coverage mismatch"
        );
        output
            .par_chunks_mut(stripe_bytes)
            .zip(self.workers.par_iter_mut())
            .enumerate()
            .try_for_each(|(stripe, (chunk, worker))| -> Result<()> {
                check_cancelled(cancelled)?;
                let base = base_node + stripe * stripe_nodes;
                for (index, out) in chunk.chunks_exact_mut(NODE_SIZE).enumerate() {
                    if index % 1024 == 0 {
                        check_cancelled(cancelled)?;
                    }
                    let node = base + index;
                    if let Some(reader) = worker.cursor.as_mut() {
                        reader.read_into(node, &mut worker.scratch.parents)?;
                    } else {
                        fill_parents(graph, &mut worker.legacy, node, &mut worker.scratch.parents)?;
                    }
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
    }

    fn log_parents<H, G>(&self, graph: &ZigZagGraph<H, G>) -> Result<()>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Sync + Send,
    {
        let io = self.source.map(|_| {
            self.workers
                .iter()
                .filter_map(|w| w.cursor.as_ref())
                .fold(ReadStats::default(), |stats, reader| {
                    stats.merge(reader.stats)
                })
        });
        log::info!(target: "zigzag_unseal", "zigzag_decode_parents {}", serde_json::json!({
            "event": "layer", "reader": if self.use_table { serde_json::to_value(self.reader)? } else { "graph".into() },
            "requested_reader": self.reader, "reversed": graph.reversed(), "nodes": graph.size(),
            "layout": self.layout, "io": io,
            "remaps": if self.source.is_some() { Some(0) } else { None::<usize> },
        }));
        Ok(())
    }
}

struct Worker<'a> {
    cursor: Option<DecodeParentCursor<'a>>,
    legacy: Option<ZigZagParentTable>,
    scratch: DecodeScratch,
}

#[cfg(test)]
mod tests;
