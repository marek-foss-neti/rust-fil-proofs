use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use anyhow::{anyhow, ensure, Context};
use blstrs::Scalar as Fr;
use ff::PrimeField;
use filecoin_hashers::{Domain, Hasher};
use fr32::bytes_into_fr_repr_safe;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use sha2::{Digest, Sha256};
use storage_proofs_core::{
    drgraph::Graph,
    error::Result,
    parameter_cache::ParameterSetMetadata,
    settings::SETTINGS,
    util::{data_at_node_offset, NODE_SIZE},
};

use crate::encode;
use crate::zigzag::vanilla::graph::ZigZagGraph;
use crate::zigzag::vanilla::parent_table::ZigZagParentTable;

const DECODE_CHUNK_NODES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Default)]
struct ParentMask(u32);

impl ParentMask {
    #[inline]
    fn clear(&mut self) {
        self.0 = 0;
    }

    #[inline]
    fn set(&mut self, index: usize) {
        self.0 |= 1 << index;
    }

    #[inline]
    fn get(self, index: usize) -> bool {
        self.0 & (1 << index) != 0
    }
}

struct DecodeScratch {
    parents: Vec<u32>,
    key: KeyScratch,
}

impl DecodeScratch {
    fn new(degree: usize) -> Self {
        DecodeScratch {
            parents: vec![0u32; degree],
            key: KeyScratch::new(degree),
        }
    }
}

struct KeyScratch {
    input: Vec<u8>,
}

impl KeyScratch {
    fn new(degree: usize) -> Self {
        KeyScratch {
            input: vec![0u8; NODE_SIZE * (degree + 1)],
        }
    }
}

struct EncodePrefetchSlot {
    parents: Vec<u32>,
    key_input: Vec<u8>,
    input_len: usize,
    missing_parents: ParentMask,
}

impl EncodePrefetchSlot {
    fn new(degree: usize) -> Self {
        EncodePrefetchSlot {
            parents: vec![0u32; degree],
            key_input: vec![0u8; key_input_len(degree)],
            input_len: NODE_SIZE,
            missing_parents: ParentMask::default(),
        }
    }
}

struct EncodePrefetchRing {
    slots: Vec<UnsafeCell<EncodePrefetchSlot>>,
    lookahead: usize,
}

unsafe impl Sync for EncodePrefetchRing {}

impl EncodePrefetchRing {
    fn new(degree: usize, lookahead: usize) -> Self {
        let slots = (0..lookahead)
            .map(|_| UnsafeCell::new(EncodePrefetchSlot::new(degree)))
            .collect();

        EncodePrefetchRing { slots, lookahead }
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn slot_mut(&self, position: usize) -> &mut EncodePrefetchSlot {
        &mut *self.slots[position % self.lookahead].get()
    }
}

struct EncodeData {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Sync for EncodeData {}

impl EncodeData {
    fn new(data: &mut [u8]) -> Self {
        EncodeData {
            ptr: data.as_mut_ptr(),
            len: data.len(),
        }
    }

    unsafe fn node_bytes(&self, node: usize) -> &[u8] {
        let offset = data_at_node_offset(node);
        debug_assert!(offset + NODE_SIZE <= self.len);
        std::slice::from_raw_parts(self.ptr.add(offset), NODE_SIZE)
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn node_bytes_mut(&self, node: usize) -> &mut [u8] {
        let offset = data_at_node_offset(node);
        debug_assert!(offset + NODE_SIZE <= self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(offset), NODE_SIZE)
    }
}

/// Opens or generates the optional on-disk ZigZag parent table for `graph`.
///
/// This is primarily useful for benchmarks and process startup warmups, so the expensive table
/// generation cost can be kept outside encode/decode timing windows.
pub fn prepare_parent_table<H, G>(graph: &ZigZagGraph<H, G>) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    if SETTINGS.use_zigzag_parent_cache {
        let _parent_table = ZigZagParentTable::new(graph)?;
    }

    Ok(())
}

/// Encodes `data` in place using the ZigZag `graph`.
///
/// Because a node always follows all of its parents in the data, the nodes are already
/// topologically sorted, so a single in-order traversal can encode each node using its
/// already-encoded parents. The subtlety versus a plain DRG is that a reversed ZigZag layer must be
/// traversed from high to low node index; this is what `graph.forward()` selects.
///
/// **Performance:** encoding is inherently sequential (each node depends on already-encoded
/// parents). Multicore SDR-style strategies from Stacked do not apply directly; any future
/// speedup must exploit graph structure (e.g. independent sub-DAGs) rather than a flat parallel
/// map. Tree building after each layer can use GPU Poseidon builders when enabled.
pub fn encode<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &mut [u8],
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    ensure_graph_data_len(graph, data)?;

    let parent_table = if SETTINGS.use_zigzag_parent_cache {
        Some(ZigZagParentTable::new(graph)?)
    } else {
        None
    };

    if SETTINGS.zigzag_multicore_encode && graph.size() > 1 {
        encode_multicore(graph, parent_table.as_ref(), replica_id, data)
    } else {
        encode_sequential(graph, parent_table.as_ref(), replica_id, data)
    }
}

fn encode_sequential<H, G>(
    graph: &ZigZagGraph<H, G>,
    parent_table: Option<&ZigZagParentTable>,
    replica_id: &H::Domain,
    data: &mut [u8],
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    let mut parents = vec![0u32; graph.degree()];
    let mut key_scratch = KeyScratch::new(graph.degree());
    for n in 0..graph.size() {
        let node = traversal_node(graph, n);

        fill_parents(graph, parent_table, node, &mut parents)?;

        let key = create_key_with_scratch::<H>(replica_id, node, &parents, data, &mut key_scratch);

        let node_data = domain_at_node_unchecked::<H>(data, node);
        let encoded = encode::encode(key, node_data);

        let start = data_at_node_offset(node);
        let end = start + NODE_SIZE;
        encoded.write_bytes(&mut data[start..end])?;
    }

    Ok(())
}

fn encode_multicore<H, G>(
    graph: &ZigZagGraph<H, G>,
    parent_table: Option<&ZigZagParentTable>,
    replica_id: &H::Domain,
    data: &mut [u8],
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    let lookahead = SETTINGS
        .zigzag_multicore_encode_lookahead
        .max(1)
        .min(graph.size());
    let producers = SETTINGS.zigzag_multicore_encode_producers.max(1);
    let stride = SETTINGS
        .zigzag_multicore_encode_producer_stride
        .max(1)
        .min(lookahead);

    let ring = EncodePrefetchRing::new(graph.degree(), lookahead);
    let data = EncodeData::new(data);
    let next_work = AtomicUsize::new(0);
    let produced_count = AtomicUsize::new(0);
    let consumer_position = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let stop = AtomicBool::new(false);

    let scoped = crossbeam::thread::scope(|scope| -> Result<()> {
        let mut runners = Vec::with_capacity(producers);
        for _ in 0..producers {
            runners.push(scope.spawn(|_| {
                encode_prefetch_runner(
                    graph,
                    parent_table,
                    replica_id,
                    &data,
                    &ring,
                    &next_work,
                    &produced_count,
                    &consumer_position,
                    &failed,
                    &stop,
                    stride,
                )
            }));
        }

        let consumer_result = encode_prefetch_consumer::<H, G>(
            graph,
            &data,
            &ring,
            &produced_count,
            &consumer_position,
            &failed,
        );
        stop.store(true, Ordering::Release);

        for runner in runners {
            runner
                .join()
                .map_err(|_| anyhow!("zigzag multicore encode producer panicked"))??;
        }

        consumer_result
    })
    .map_err(|_| anyhow!("zigzag multicore encode scope panicked"))?;

    scoped
}

/// Decodes (extracts) all of `data`, returning the original pre-encoding bytes.
///
/// This is where ZigZag's fast-extraction asymmetry lives. Unlike [`encode`], which is inherently
/// sequential (a node cannot be encoded until its parents have been encoded), every node here is
/// decoded from the *fully-encoded, immutable* replica: [`decode_block`] reads a node's parents from
/// `data` (never from partially-decoded output), so the nodes are mutually independent and can be
/// decoded in any order or fully in parallel. We exploit that here with a parallel iterator.
pub fn decode<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &[u8],
) -> Result<Vec<u8>>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    ensure_graph_data_len(graph, data)?;

    let mut out = vec![0u8; data.len()];
    let parent_table = if SETTINGS.use_zigzag_parent_cache {
        Some(ZigZagParentTable::new(graph)?)
    } else {
        None
    };
    let parent_table = parent_table.as_ref();

    out.par_chunks_mut(DECODE_CHUNK_NODES * NODE_SIZE)
        .enumerate()
        .try_for_each(|(chunk_index, chunk)| -> Result<()> {
            let mut scratch = DecodeScratch::new(graph.degree());
            let base_node = chunk_index * DECODE_CHUNK_NODES;

            for (chunk_node, node_out) in chunk.chunks_mut(NODE_SIZE).enumerate() {
                let node = base_node + chunk_node;
                let decoded = decode_block_with_scratch(
                    graph,
                    parent_table,
                    replica_id,
                    data,
                    node,
                    &mut scratch,
                )?;
                decoded.write_bytes(node_out)?;
            }

            Ok(())
        })?;

    Ok(out)
}

/// Decodes a single node of `data`.
///
/// Depends only on the immutable, fully-encoded `data` (never on other decoded nodes), which is what
/// lets [`decode`] run every node independently/in parallel.
pub fn decode_block<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &[u8],
    v: usize,
) -> Result<H::Domain>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    ensure_graph_data_len(graph, data)?;
    ensure!(
        v < graph.size(),
        "ZigZag decode node {} outside graph size {}",
        v,
        graph.size()
    );

    let mut scratch = DecodeScratch::new(graph.degree());
    decode_block_with_scratch(graph, None, replica_id, data, v, &mut scratch)
}

fn decode_block_with_scratch<H, G>(
    graph: &ZigZagGraph<H, G>,
    parent_table: Option<&ZigZagParentTable>,
    replica_id: &H::Domain,
    data: &[u8],
    v: usize,
    scratch: &mut DecodeScratch,
) -> Result<H::Domain>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    fill_parents(graph, parent_table, v, &mut scratch.parents)?;
    let key = create_key_with_scratch::<H>(replica_id, v, &scratch.parents, data, &mut scratch.key);
    let node_data = domain_at_node_unchecked::<H>(data, v);

    Ok(encode::decode(key, node_data))
}

#[allow(clippy::too_many_arguments)]
fn encode_prefetch_runner<H, G>(
    graph: &ZigZagGraph<H, G>,
    parent_table: Option<&ZigZagParentTable>,
    replica_id: &H::Domain,
    data: &EncodeData,
    ring: &EncodePrefetchRing,
    next_work: &AtomicUsize,
    produced_count: &AtomicUsize,
    consumer_position: &AtomicUsize,
    failed: &AtomicBool,
    stop: &AtomicBool,
    stride: usize,
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    loop {
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }

        let work = next_work.fetch_add(stride, Ordering::AcqRel);
        if work >= graph.size() {
            return Ok(());
        }
        let count = stride.min(graph.size() - work);

        for position in work..work + count {
            while position >= consumer_position.load(Ordering::Acquire) + ring.lookahead {
                if stop.load(Ordering::Acquire) {
                    return Ok(());
                }
                thread::yield_now();
            }

            if let Err(err) = fill_encode_prefetch_slot(
                graph,
                parent_table,
                replica_id,
                data,
                position,
                consumer_position.load(Ordering::Acquire),
                // Safety: the ring protocol ensures a slot is written by at most one producer before
                // the consumer reads it, and is not reused until `consumer_position` advances.
                unsafe { ring.slot_mut(position) },
            ) {
                failed.store(true, Ordering::Release);
                return Err(err);
            }
        }

        while produced_count.load(Ordering::Acquire) != work {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            thread::yield_now();
        }
        produced_count.fetch_add(count, Ordering::AcqRel);
    }
}

fn fill_encode_prefetch_slot<H, G>(
    graph: &ZigZagGraph<H, G>,
    parent_table: Option<&ZigZagParentTable>,
    replica_id: &H::Domain,
    data: &EncodeData,
    position: usize,
    encoded_frontier: usize,
    slot: &mut EncodePrefetchSlot,
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    let node = traversal_node(graph, position);
    fill_parents(graph, parent_table, node, &mut slot.parents)?;

    slot.missing_parents.clear();
    slot.key_input[..NODE_SIZE].copy_from_slice(AsRef::<[u8]>::as_ref(replica_id));
    slot.input_len = NODE_SIZE;

    // The hash is about the parents, hence skip if a node doesn't have any parents.
    if node != slot.parents[0] as usize {
        for (index, parent) in slot.parents.iter().enumerate() {
            let input_offset = NODE_SIZE * (index + 1);
            let parent_position = traversal_position(graph, *parent as usize);

            if parent_position < encoded_frontier {
                // Safety: `encoded_frontier` is advanced only after the consumer has written the
                // encoded parent bytes, and encoded nodes are immutable for the rest of this layer.
                slot.key_input[input_offset..input_offset + NODE_SIZE]
                    .copy_from_slice(unsafe { data.node_bytes(*parent as usize) });
            } else {
                slot.missing_parents.set(index);
            }
        }
        slot.input_len = key_input_len(slot.parents.len());
    }

    Ok(())
}

fn encode_prefetch_consumer<H, G>(
    graph: &ZigZagGraph<H, G>,
    data: &EncodeData,
    ring: &EncodePrefetchRing,
    produced_count: &AtomicUsize,
    consumer_position: &AtomicUsize,
    failed: &AtomicBool,
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    for position in 0..graph.size() {
        while produced_count.load(Ordering::Acquire) <= position {
            if failed.load(Ordering::Acquire) {
                return Err(anyhow!("zigzag multicore encode producer failed"));
            }
            thread::yield_now();
        }

        let node = traversal_node(graph, position);
        // Safety: the slot is ready because `produced_count > position`, and it cannot be reused
        // until this loop advances `consumer_position` after writing the encoded node.
        let slot = unsafe { ring.slot_mut(position) };
        patch_missing_parents::<H>(data, slot);

        let key = create_key_from_staged_input::<H>(&slot.key_input[..slot.input_len]);
        let node_data = domain_from_node_bytes_unchecked::<H>(
            // Safety: this node is the current traversal frontier and has not been encoded yet.
            unsafe { data.node_bytes(node) },
        );
        let encoded = encode::encode(key, node_data);
        encoded.write_bytes(
            // Safety: only the consumer writes the current traversal node.
            unsafe { data.node_bytes_mut(node) },
        )?;
        consumer_position.store(position + 1, Ordering::Release);
    }

    Ok(())
}

fn patch_missing_parents<H: Hasher>(data: &EncodeData, slot: &mut EncodePrefetchSlot) {
    if slot.input_len == NODE_SIZE {
        return;
    }

    for (index, parent) in slot.parents.iter().enumerate() {
        if slot.missing_parents.get(index) {
            let input_offset = NODE_SIZE * (index + 1);
            // Safety: missing parents are patched only by the consumer, immediately before hashing
            // the current node, so all traversal-earlier parents are already encoded.
            slot.key_input[input_offset..input_offset + NODE_SIZE]
                .copy_from_slice(unsafe { data.node_bytes(*parent as usize) });
        }
    }
}

fn fill_parents<H, G>(
    graph: &ZigZagGraph<H, G>,
    parent_table: Option<&ZigZagParentTable>,
    node: usize,
    parents: &mut [u32],
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    if let Some(parent_table) = parent_table {
        parent_table.read_into(node, parents)
    } else {
        graph.parents(node, parents)
    }
}

/// Creates the encoding key: `SHA256(replica_id | parent_1 | parent_2 | ...)`, reduced to a field
/// element by clearing the top two bits (the standard LE-254 packing used throughout the codebase).
///
/// The KDF hash is SHA256 (rather than the 2019 original's Blake2s) so it can be reproduced
/// efficiently and reliably in-circuit with `bellperson`'s SHA256 gadget.
pub fn create_key<H: Hasher>(
    id: &H::Domain,
    node: usize,
    parents: &[u32],
    data: &[u8],
) -> Result<H::Domain> {
    let mut scratch = KeyScratch::new(parents.len());
    Ok(create_key_with_scratch::<H>(
        id,
        node,
        parents,
        data,
        &mut scratch,
    ))
}

fn create_key_with_scratch<H: Hasher>(
    id: &H::Domain,
    node: usize,
    parents: &[u32],
    data: &[u8],
    scratch: &mut KeyScratch,
) -> H::Domain {
    debug_assert!(scratch.input.len() >= key_input_len(parents.len()));

    scratch.input[..NODE_SIZE].copy_from_slice(AsRef::<[u8]>::as_ref(id));
    let mut input_len = NODE_SIZE;
    // The hash is about the parents, hence skip if a node doesn't have any parents.
    if node != parents[0] as usize {
        for (index, parent) in parents.iter().enumerate() {
            let offset = data_at_node_offset(*parent as usize);
            let input_offset = NODE_SIZE * (index + 1);
            scratch.input[input_offset..input_offset + NODE_SIZE]
                .copy_from_slice(&data[offset..offset + NODE_SIZE]);
        }
        input_len += NODE_SIZE * parents.len();
    }

    create_key_from_staged_input::<H>(&scratch.input[..input_len])
}

fn create_key_from_staged_input<H: Hasher>(input: &[u8]) -> H::Domain {
    let hash = Sha256::digest(input);
    bytes_into_fr_repr_safe(hash.as_ref()).into()
}

fn key_input_len(degree: usize) -> usize {
    NODE_SIZE * (degree + 1)
}

#[inline]
fn traversal_node<H, G>(graph: &ZigZagGraph<H, G>, position: usize) -> usize
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    if graph.forward() {
        position
    } else {
        graph.size() - 1 - position
    }
}

#[inline]
fn traversal_position<H, G>(graph: &ZigZagGraph<H, G>, node: usize) -> usize
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    if graph.forward() {
        node
    } else {
        graph.size() - 1 - node
    }
}

fn ensure_graph_data_len<H, G>(graph: &ZigZagGraph<H, G>, data: &[u8]) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    let expected_len = graph
        .size()
        .checked_mul(NODE_SIZE)
        .context("ZigZag graph data length overflow")?;
    ensure!(
        data.len() == expected_len,
        "invalid ZigZag data length: got {}, expected {}",
        data.len(),
        expected_len
    );

    Ok(())
}

fn domain_at_node_unchecked<H: Hasher>(data: &[u8], node: usize) -> H::Domain {
    let offset = data_at_node_offset(node);
    debug_assert!(offset + NODE_SIZE <= data.len());

    domain_from_node_bytes_unchecked::<H>(&data[offset..offset + NODE_SIZE])
}

fn domain_from_node_bytes_unchecked<H: Hasher>(node_data: &[u8]) -> H::Domain {
    debug_assert_eq!(node_data.len(), NODE_SIZE);

    let mut repr = <Fr as PrimeField>::Repr::default();
    repr.as_mut().copy_from_slice(node_data);
    H::Domain::from(repr)
}

/// Recreates the encoding key from already-materialized parent domain values (used during
/// verification, where parent bytes come from Merkle-proof leaves rather than the data buffer).
/// This must match `create_key`'s hashing exactly.
#[allow(clippy::unnecessary_wraps)]
pub fn create_key_from_domains<H: Hasher>(
    id: &H::Domain,
    parents_data: &[H::Domain],
) -> Result<H::Domain> {
    let mut hasher = Sha256::new();
    hasher.update(AsRef::<[u8]>::as_ref(id));

    for parent in parents_data.iter() {
        hasher.update(AsRef::<[u8]>::as_ref(parent));
    }

    let hash = hasher.finalize();
    Ok(bytes_into_fr_repr_safe(hash.as_ref()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    use filecoin_hashers::poseidon::{PoseidonDomain, PoseidonHasher};
    use storage_proofs_core::{api_version::ApiVersion, drgraph::BASE_DEGREE, util::NODE_SIZE};

    use crate::zigzag::vanilla::graph::{ZigZagBucketGraph, EXP_DEGREE};

    fn create_key_streaming_reference<H: Hasher>(
        id: &H::Domain,
        node: usize,
        parents: &[u32],
        data: &[u8],
    ) -> H::Domain {
        let mut hasher = Sha256::new();
        hasher.update(AsRef::<[u8]>::as_ref(id));

        // The hash is about the parents, hence skip if a node doesn't have any parents.
        if node != parents[0] as usize {
            for parent in parents.iter() {
                let offset = data_at_node_offset(*parent as usize);
                hasher.update(&data[offset..offset + NODE_SIZE]);
            }
        }

        let hash = hasher.finalize();
        bytes_into_fr_repr_safe(hash.as_ref()).into()
    }

    fn encode_decode_roundtrip(reversed: bool) {
        let nodes = 32;
        let porep_id = [7u8; 32];
        let mut graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            porep_id,
            ApiVersion::V1_2_0,
        )
        .expect("failed to create graph");
        if reversed {
            graph = graph.zigzag();
        }

        let replica_id = PoseidonDomain::default();

        // All-zero bytes is a valid field-element (Fr) representation for every node.
        let original = vec![0u8; nodes * NODE_SIZE];
        let mut data = original.clone();

        encode::<PoseidonHasher, _>(&graph, &replica_id, &mut data).expect("encode failed");
        assert_ne!(data, original, "encoding did not change data");

        let decoded =
            decode::<PoseidonHasher, _>(&graph, &replica_id, &data).expect("decode failed");
        assert_eq!(decoded, original, "decode did not recover original data");
    }

    #[test]
    fn encode_decode_forward() {
        encode_decode_roundtrip(false);
    }

    #[test]
    fn encode_decode_reversed() {
        encode_decode_roundtrip(true);
    }

    #[test]
    fn multicore_encode_matches_sequential_encode() {
        let nodes = 128;
        let porep_id = [19u8; 32];
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            porep_id,
            ApiVersion::V1_2_0,
        )
        .expect("failed to create graph");
        let reversed = graph.zigzag();
        let replica_id = PoseidonDomain::from([12u8; 32]);

        let mut original = vec![0u8; nodes * NODE_SIZE];
        for node in 0..nodes {
            original[node * NODE_SIZE] = (node as u8).wrapping_mul(7).wrapping_add(1);
            original[node * NODE_SIZE + 11] = (node as u8).wrapping_mul(29);
        }

        for graph in [&graph, &reversed] {
            let mut sequential = original.clone();
            encode_sequential::<PoseidonHasher, _>(graph, None, &replica_id, &mut sequential)
                .expect("sequential encode failed");

            let mut multicore = original.clone();
            encode_multicore::<PoseidonHasher, _>(graph, None, &replica_id, &mut multicore)
                .expect("multicore encode failed");

            assert_eq!(
                multicore,
                sequential,
                "multicore encode diverged for reversed={}",
                graph.reversed()
            );
        }
    }

    #[test]
    fn staged_key_input_matches_streaming_key_input() {
        let nodes = 64;
        let porep_id = [11u8; 32];
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            porep_id,
            ApiVersion::V1_2_0,
        )
        .expect("failed to create graph");
        let reversed = graph.zigzag();
        let replica_id = PoseidonDomain::from([5u8; 32]);

        let mut data = vec![0u8; nodes * NODE_SIZE];
        for node in 0..nodes {
            data[node * NODE_SIZE] = (node as u8).wrapping_mul(3).wrapping_add(1);
            data[node * NODE_SIZE + 7] = (node as u8).wrapping_mul(19);
        }

        for graph in [&graph, &reversed] {
            let mut parents = vec![0u32; graph.degree()];
            let mut key_scratch = KeyScratch::new(graph.degree());
            for node in [0, 1, 7, nodes - 2, nodes - 1] {
                graph.parents(node, &mut parents).expect("parents failed");
                let streaming = create_key_streaming_reference::<PoseidonHasher>(
                    &replica_id,
                    node,
                    &parents,
                    &data,
                );
                let staged = create_key_with_scratch::<PoseidonHasher>(
                    &replica_id,
                    node,
                    &parents,
                    &data,
                    &mut key_scratch,
                );
                assert_eq!(staged, streaming);

                let public = create_key::<PoseidonHasher>(&replica_id, node, &parents, &data)
                    .expect("public create_key failed");
                assert_eq!(public, streaming);
            }
        }
    }

    /// ZigZag's defining performance property: extraction is *asymmetric* to replication. Encoding is
    /// inherently sequential (a node depends on its already-encoded parents), but decoding each node
    /// depends only on the immutable, fully-encoded replica. This test locks in that independence:
    /// decoding in forward order, in reverse order, and in parallel must all yield identical results
    /// (and recover the original data). If a future change made `decode_block` read partially-decoded
    /// state, the order variants would diverge and this test would fail.
    #[test]
    fn decode_is_order_independent_and_parallelizable() {
        let nodes = 64;
        let porep_id = [13u8; 32];
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            porep_id,
            ApiVersion::V1_2_0,
        )
        .expect("failed to create graph");

        let replica_id = PoseidonDomain::from([2u8; 32]);

        // Non-trivial, field-valid data so encoding genuinely depends on parent values.
        let mut original = vec![0u8; nodes * NODE_SIZE];
        for i in 0..nodes {
            original[i * NODE_SIZE] = (i as u8).wrapping_mul(7).wrapping_add(3);
            original[i * NODE_SIZE + 1] = (i as u8).wrapping_mul(31);
        }

        let mut data = original.clone();
        encode::<PoseidonHasher, _>(&graph, &replica_id, &mut data).expect("encode failed");
        assert_ne!(data, original, "encoding did not change data");

        // Decode block-by-block in forward order.
        let mut forward = vec![0u8; data.len()];
        for v in 0..nodes {
            let d = decode_block::<PoseidonHasher, _>(&graph, &replica_id, &data, v)
                .expect("decode_block failed");
            d.write_bytes(&mut forward[v * NODE_SIZE..(v + 1) * NODE_SIZE])
                .expect("write_bytes failed");
        }

        // Decode block-by-block in reverse order, writing each result into its own position. Because
        // blocks are independent, this must produce exactly the same output as forward order.
        let mut reverse = vec![0u8; data.len()];
        for v in (0..nodes).rev() {
            let d = decode_block::<PoseidonHasher, _>(&graph, &replica_id, &data, v)
                .expect("decode_block failed");
            d.write_bytes(&mut reverse[v * NODE_SIZE..(v + 1) * NODE_SIZE])
                .expect("write_bytes failed");
        }

        // The production `decode` path, which decodes all nodes in parallel.
        let parallel =
            decode::<PoseidonHasher, _>(&graph, &replica_id, &data).expect("decode failed");

        assert_eq!(
            forward, original,
            "forward-order extraction did not recover original data"
        );
        assert_eq!(
            reverse, forward,
            "reverse-order extraction differs from forward order: decode blocks are not independent"
        );
        assert_eq!(
            parallel, forward,
            "parallel extraction differs from sequential: decode blocks are not independent"
        );
    }

    /// The other half of the asymmetry: encoding *is* order-dependent. On a forward graph every
    /// node's parents have a lower index, so a correct encoding must proceed low->high (each node
    /// consumes its already-encoded parents). Encoding high->low instead consumes not-yet-encoded
    /// parents and therefore yields different bytes. This guards the sequential nature of encoding,
    /// which is precisely what makes replication slow while extraction stays fast.
    #[test]
    fn encoding_is_order_dependent() {
        let nodes = 64;
        let porep_id = [17u8; 32];
        // Forward (non-reversed) graph: parents(v) are all < v.
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            porep_id,
            ApiVersion::V1_2_0,
        )
        .expect("failed to create graph");

        let replica_id = PoseidonDomain::from([9u8; 32]);

        let mut original = vec![0u8; nodes * NODE_SIZE];
        for i in 0..nodes {
            original[i * NODE_SIZE] = (i as u8).wrapping_mul(5).wrapping_add(1);
        }

        // Correct, sequential low->high encoding.
        let mut correct = original.clone();
        encode::<PoseidonHasher, _>(&graph, &replica_id, &mut correct).expect("encode failed");

        // Deliberately encode high->low; parents are not yet encoded, so keys differ.
        let mut wrong = original.clone();
        let mut parents = vec![0u32; graph.degree()];
        for v in (0..nodes).rev() {
            graph.parents(v, &mut parents).expect("parents failed");
            let key = create_key::<PoseidonHasher>(&replica_id, v, &parents, &wrong)
                .expect("create_key failed");
            let start = data_at_node_offset(v);
            let end = start + NODE_SIZE;
            let node_data =
                PoseidonDomain::try_from_bytes(&wrong[start..end]).expect("try_from_bytes failed");
            let encoded = encode::encode(key, node_data);
            encoded
                .write_bytes(&mut wrong[start..end])
                .expect("write_bytes failed");
        }

        assert_ne!(
            correct, wrong,
            "encoding was order-independent — the sequential dependency (and thus the \
             replication/extraction asymmetry) has been lost"
        );
    }
}
