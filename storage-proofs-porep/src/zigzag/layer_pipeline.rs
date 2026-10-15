//! Bounded overlap of immutable TreeR snapshots with the next ZigZag encode.
//! The active replica is never read by a background worker. Streaming transfers
//! owned chunks to the leaf section of the final store, including reverse layers.

use std::fs::File;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, ensure, Context, Result};
use filecoin_hashers::Hasher;
use merkletree::store::StoreConfig;
use serde::Serialize;
use storage_proofs_core::{drgraph::Graph, merkle::MerkleTreeTrait, settings::SETTINGS};

use super::cache_policy::{discard_file, CachePolicy};
use super::dirty_writeback::with_dirty_budget;
use super::measurements::{OperationGuard, WaitStats};
use super::merkle_builder::{supports_binary, LeafSnapshot, TreeRBuilder};
use super::vanilla::{vde, ZigZagBucketGraph};

const CHUNK_BYTES: usize = 1 << 20;
const DIRTY_BYTES: usize = 64 << 20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerPipelineMode {
    #[default]
    Sequential,
    Overlap,
    Stream,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct LayerPipelineConfig {
    pub mode: LayerPipelineMode,
    pub tree_threads: Option<usize>,
    pub snapshot_buffer_mib: usize,
}

impl LayerPipelineConfig {
    pub fn from_env() -> Result<Self> {
        let mode = match std::env::var("FIL_PROOFS_ZIGZAG_LAYER_PIPELINE") {
            Err(std::env::VarError::NotPresent) => LayerPipelineMode::Sequential,
            Ok(value) if value == "sequential" => LayerPipelineMode::Sequential,
            Ok(value) if value == "overlap" => LayerPipelineMode::Overlap,
            Ok(value) if value == "stream" => LayerPipelineMode::Stream,
            _ => bail!("FIL_PROOFS_ZIGZAG_LAYER_PIPELINE must be sequential, overlap or stream"),
        };
        let tree_threads = match std::env::var("FIL_PROOFS_ZIGZAG_TREE_R_THREADS") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) if value == "auto" => None,
            Ok(value) => Some(parse_bound(&value, "FIL_PROOFS_ZIGZAG_TREE_R_THREADS")?),
            _ => bail!("invalid FIL_PROOFS_ZIGZAG_TREE_R_THREADS"),
        };
        let snapshot_buffer_mib = match std::env::var("FIL_PROOFS_ZIGZAG_SNAPSHOT_BUFFER_MIB") {
            Err(std::env::VarError::NotPresent) => 64,
            Ok(value) => parse_bound(&value, "FIL_PROOFS_ZIGZAG_SNAPSHOT_BUFFER_MIB")?,
            _ => bail!("invalid FIL_PROOFS_ZIGZAG_SNAPSHOT_BUFFER_MIB"),
        };
        Ok(Self {
            mode,
            tree_threads,
            snapshot_buffer_mib,
        })
    }

    pub(crate) fn validate<Tree: 'static + MerkleTreeTrait>(
        self,
        builder: TreeRBuilder,
        cache: Option<&Path>,
    ) -> Result<()> {
        ensure!(
            (1..=64).contains(&self.snapshot_buffer_mib),
            "snapshot pool must be 1..64 MiB"
        );
        ensure!(
            self.tree_threads.is_none_or(|n| (1..=64).contains(&n)),
            "TreeR workers must be 1..64"
        );
        if self.mode != LayerPipelineMode::Sequential {
            ensure!(builder == TreeRBuilder::Binary && supports_binary::<Tree>() && cache.is_some(),
                "ZigZag layer pipeline requires the binary builder and a Linux binary Poseidon disk cache");
        }
        Ok(())
    }
}

fn parse_bound(value: &str, name: &str) -> Result<usize> {
    ensure!(
        !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
        "{name} must be 1..64"
    );
    let value: usize = value.parse().with_context(|| format!("invalid {name}"))?;
    ensure!((1..=64).contains(&value), "{name} must be 1..64");
    Ok(value)
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn checked_job<T>(cancelled: &AtomicBool, job: impl FnOnce() -> Result<T>) -> Result<T> {
    let result = catch_unwind(AssertUnwindSafe(job))
        .map_err(|_| anyhow!("ZigZag layer worker panicked"))
        .and_then(|result| result);
    if result.is_err() {
        cancelled.store(true, Ordering::Release);
    }
    result
}

struct PendingTree<Tree> {
    handle: Option<JoinHandle<Result<Tree>>>,
    cancelled: Arc<AtomicBool>,
}
impl<Tree> PendingTree<Tree> {
    fn abort(self, error: anyhow::Error) -> anyhow::Error {
        // Signal before join: a failed encoder must not wait for an entire
        // 32 GiB tree, and cancellation must not hide the initiating I/O error.
        self.cancelled.store(true, Ordering::Release);
        match self.join() {
            Ok(_) => error,
            Err(background) => {
                let details =
                    format!("layer pipeline failed: {error:#}; background TreeR: {background:#}");
                error.context(details)
            }
        }
    }

    fn join(mut self) -> Result<Tree> {
        let operation = OperationGuard::enter("tree_r_backpressure", None);
        let result = self
            .handle
            .take()
            .expect("pending TreeR")
            .join()
            .map_err(|_| anyhow!("TreeR worker panicked"))?;
        if result.is_ok() {
            operation.finish();
        }
        result
    }
}
impl<Tree> Drop for PendingTree<Tree> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.cancelled.store(true, Ordering::Release);
            let _ = handle.join();
        }
    }
}

/// At most one older tree is building. Stream mode may additionally capture the
/// current layer; it joins that writer and the older builder before spawning
/// another builder. The caller can publish AUX only after this returns all trees.
pub(crate) fn replicate<Tree: 'static + MerkleTreeTrait>(
    graph: &ZigZagBucketGraph<Tree::Hasher>,
    layers: usize,
    replica_id: &<Tree::Hasher as Hasher>::Domain,
    data: &mut [u8],
    cache: &Path,
    policy: CachePolicy,
    config: LayerPipelineConfig,
) -> Result<Vec<Tree>> {
    config.validate::<Tree>(TreeRBuilder::Binary, Some(cache))?;
    let total = rayon::current_num_threads();
    let encode_threads = if SETTINGS.zigzag_multicore_encode {
        SETTINGS
            .zigzag_multicore_encode_producers
            .max(1)
            .saturating_add(1)
    } else {
        1
    };
    let ceiling = total.saturating_sub(encode_threads).clamp(1, 64);
    let tree_threads = config.tree_threads.unwrap_or(ceiling).min(ceiling);
    // Build the pool before encoder affinity narrows the caller's CPU mask.
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(tree_threads)
            .thread_name(|n| format!("zigzag-tree-{n}"))
            .build()?,
    );
    log::info!(target: "zigzag_pipeline", "zigzag_pipeline {}", serde_json::json!({
        "mode": config.mode, "tree_threads": tree_threads, "requested_tree_threads": config.tree_threads,
        "encode_threads": encode_threads, "rayon_threads": total,
        "snapshot_buffer_bytes": if config.mode == LayerPipelineMode::Stream { (config.snapshot_buffer_mib << 20).min(data.len()) } else { 0 },
        "dirty_bytes_per_store": DIRTY_BYTES, "max_building_trees": 1, "max_capturing_trees": 1,
        "scope": "per-call worker/buffer bounds; dirty bound excludes active replica and clean page cache; CPU count is not exclusive CPU affinity"
    }));
    let cancelled = Arc::new(AtomicBool::new(false));
    let _cancel_on_exit = CancelOnDrop(cancelled.clone());
    let mut pending: Option<PendingTree<Tree>> = None;
    let mut trees = Vec::with_capacity(layers);
    let mut graph = graph.clone();
    for layer in 0..layers {
        if cancelled.load(Ordering::Acquire) {
            if let Some(tree) = pending.take() {
                tree.join()?;
            }
            bail!("ZigZag layer pipeline cancelled");
        }
        let store = StoreConfig::new(cache, format!("zigzag-tree-r-{layer}"), 0);
        let encode_operation = OperationGuard::enter("encode", Some(layer));
        let started = Instant::now();
        let (snapshot, tree_operation) = if config.mode == LayerPipelineMode::Stream {
            let tree_operation = OperationGuard::enter("tree_r", Some(layer));
            let mut writer = SnapshotWriter::start(
                store.clone(),
                graph.size(),
                layer,
                graph.forward(),
                CHUNK_BYTES.min(data.len()),
                config.snapshot_buffer_mib,
                cancelled.clone(),
            )?;
            let result = vde::encode_with_output(
                &graph,
                replica_id,
                data,
                &mut vde::EncodeOutput::new(&cancelled, Some(&mut writer), CHUNK_BYTES / 32)?,
            );
            if result.is_ok() {
                encode_operation.finish();
            }
            // Always join the writer, including when the encoder or older tree
            // failed. A concrete writer I/O error takes priority over cancellation.
            let snapshot = match writer.finish(result) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    return Err(match pending.take() {
                        Some(tree) => tree.abort(error),
                        None => error,
                    });
                }
            };
            (snapshot, tree_operation)
        } else {
            let result = vde::encode_with_output(
                &graph,
                replica_id,
                data,
                &mut vde::EncodeOutput::new(&cancelled, None, CHUNK_BYTES / 32)?,
            );
            if result.is_ok() {
                encode_operation.finish();
            }
            if let Err(error) = result {
                cancelled.store(true, Ordering::Release);
                return Err(match pending.take() {
                    Some(tree) => tree.abort(error),
                    None => error,
                });
            }
            if let Some(tree) = pending.take() {
                trees.push(tree.join()?);
            }
            let tree_operation = OperationGuard::enter("tree_r", Some(layer));
            let snapshot = LeafSnapshot::capture_with(
                store.clone(),
                graph.size(),
                layer,
                &cancelled,
                |file, expected| {
                    let (_, stats) = with_dirty_budget(file, DIRTY_BYTES, &cancelled, |dirty| {
                        for (index, bytes) in data.chunks(CHUNK_BYTES).enumerate() {
                            ensure!(!cancelled.load(Ordering::Acquire), "snapshot cancelled");
                            dirty.write(file, bytes, (index * CHUNK_BYTES) as u64)?;
                        }
                        Ok(())
                    })?;
                    log::info!(target: "zigzag_pipeline", "zigzag_pipeline {}", serde_json::json!({
                        "layer": layer, "mode": "overlap", "state": "written",
                        "written_bytes": expected, "dirty_budget": stats,
                        "scope": "snapshot writes and flushes completed; final store publication still pending"
                    }));
                    Ok(expected)
                },
            )?;
            (snapshot, tree_operation)
        };
        log::info!(target: "zigzag_pipeline", "zigzag_pipeline {}", serde_json::json!({
            "layer": layer, "mode": config.mode, "capture_ready_ms": started.elapsed().as_millis(),
            "scope": "encode and capture readiness, including backpressure; not additive to PC1"
        }));
        // Stream capture can overlap the older build, but two hash builders
        // never run concurrently, and no next layer starts with two pending trees.
        if let Some(tree) = pending.take() {
            trees.push(tree.join()?);
        }
        let pool = pool.clone();
        let cancel = cancelled.clone();
        let handle = thread::Builder::new()
            .name(format!("zigzag-layer-{layer}"))
            .spawn(move || {
                checked_job(&cancel, || {
                    let tree = pool.install(|| {
                        snapshot
                            .build_with_dirty_limit(layer, &cancel, Some(DIRTY_BYTES))?
                            .publish::<Tree>(layer, &cancel)
                    })?;
                    if policy.tree_r_dontneed {
                        // publish has already synced the completed store.
                        discard_file(&File::open(StoreConfig::data_path(&store.path, &store.id))?);
                    }
                    tree_operation.finish();
                    Ok(tree)
                })
            })?;
        pending = Some(PendingTree {
            handle: Some(handle),
            cancelled: cancelled.clone(),
        });
        graph = graph.zigzag();
    }
    if let Some(tree) = pending {
        trees.push(tree.join()?);
    }
    ensure!(trees.len() == layers, "incomplete ZigZag layer trees");
    Ok(trees)
}

struct Chunk {
    layer: usize,
    offset: usize,
    length: usize,
    bytes: Vec<u8>,
}

struct SnapshotWriter {
    layer: usize,
    sender: Option<mpsc::SyncSender<Chunk>>,
    available: mpsc::Receiver<Vec<u8>>,
    handle: Option<JoinHandle<Result<Option<LeafSnapshot>>>>,
    cancelled: Arc<AtomicBool>,
    captured: usize,
    expected: usize,
    buffer_wait: WaitStats,
}

impl SnapshotWriter {
    fn start(
        config: StoreConfig,
        leaves: usize,
        layer: usize,
        forward: bool,
        chunk_bytes: usize,
        pool_mib: usize,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Self> {
        Self::start_with_hook(
            config,
            leaves,
            layer,
            forward,
            chunk_bytes,
            ((pool_mib << 20) / chunk_bytes)
                .max(1)
                .min((leaves * 32).div_ceil(chunk_bytes)),
            cancelled,
            |_, _| Ok(()),
        )
    }

    fn start_with_hook(
        config: StoreConfig,
        leaves: usize,
        layer: usize,
        forward: bool,
        chunk_bytes: usize,
        buffers: usize,
        cancelled: Arc<AtomicBool>,
        before_write: impl Fn(usize, usize) -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        ensure!(
            chunk_bytes > 0 && chunk_bytes % 32 == 0 && chunk_bytes <= CHUNK_BYTES,
            "invalid snapshot chunk size"
        );
        ensure!(buffers > 0 && buffers <= 64, "invalid snapshot pool size");
        let expected = leaves.checked_mul(32).context("snapshot size overflow")?;
        let (sender, receiver) = mpsc::sync_channel::<Chunk>(buffers);
        let (free_sender, available) = mpsc::sync_channel(buffers);
        // Exactly these buffers circulate; no per-chunk allocation or borrowing
        // of mutable replica memory crosses the writer boundary.
        for _ in 0..buffers {
            free_sender.send(vec![0u8; chunk_bytes])?;
        }
        let (ready_sender, ready) = mpsc::sync_channel(1);
        let cancel = cancelled.clone();
        let handle = thread::Builder::new().name(format!("zigzag-snapshot-{layer}"))
            .spawn(move || checked_job(&cancel, || {
                let mut aborted = false;
                let snapshot = LeafSnapshot::capture_with(config, leaves, layer, &cancel, |file, expected| {
                    ready_sender.send(())?;
                    let mut written = 0usize;
                    let mut chunks = 0usize;
                    let mut input_wait = WaitStats::default();
                    let mut write_ns = 0u64;
                    let (_, dirty_stats) = with_dirty_budget(file, DIRTY_BYTES, &cancel, |dirty| {
                    loop {
                        let chunk = match receiver.try_recv() {
                            Ok(chunk) => chunk,
                            Err(mpsc::TryRecvError::Disconnected) => break,
                            Err(mpsc::TryRecvError::Empty) => {
                                let started = Instant::now();
                                let received = loop {
                                    ensure!(!cancel.load(Ordering::Acquire), "snapshot cancelled");
                                    match receiver.recv_timeout(Duration::from_millis(50)) {
                                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                                        result => break result,
                                    }
                                };
                                input_wait.record(started);
                                match received {
                                    Ok(chunk) => chunk,
                                    Err(_) => break,
                                }
                            }
                        };
                        if cancel.load(Ordering::Acquire) { aborted = true; return Ok(written); }
                        let length = chunk_bytes.min(expected - written);
                        let offset = if forward { written } else { expected - written - length };
                        ensure!(chunk.layer == layer && chunk.length == length && chunk.offset == offset
                            && chunk.bytes.len() >= length && length > 0, "out-of-order or invalid snapshot chunk");
                        before_write(chunks, offset)?;
                        let started = Instant::now();
                        dirty.write(file, &chunk.bytes[..length], offset as u64)?;
                        write_ns += started.elapsed().as_nanos() as u64;
                        written += length;
                        chunks += 1;
                        if free_sender.send(chunk.bytes).is_err() { aborted = true; return Ok(written); }
                    }
                    if cancel.load(Ordering::Acquire) { aborted = true; return Ok(written); }
                    ensure!(written == expected, "incomplete streamed snapshot");
                    Ok(written)
                    })?;
                    log::info!(target: "zigzag_pipeline", "zigzag_pipeline {}", serde_json::json!({
                        "layer": layer, "written_bytes": written, "chunks": chunks,
                        "pool_buffers": buffers, "pool_bytes": buffers * chunk_bytes, "forward": forward,
                        "writer_input_wait": input_wait, "writer_write_ns": write_ns, "dirty_budget": dirty_stats,
                        "state": "written", "scope": "leaf writes finished; final store durability and publication still pending"
                    }));
                    Ok(written)
                });
                match snapshot {
                    // Cancellation may leave an incomplete private snapshot. It
                    // is dropped, never promoted into a buildable/public tree.
                    Err(_) if aborted => Ok(None),
                    result => result.map(Some),
                }
            }))?;
        let writer = Self {
            layer,
            sender: Some(sender),
            available,
            handle: Some(handle),
            cancelled,
            captured: 0,
            expected,
            buffer_wait: WaitStats::default(),
        };
        if ready.recv().is_err() {
            let result = writer.finish(Err(anyhow!("snapshot allocation failed")));
            return Err(result
                .err()
                .unwrap_or_else(|| anyhow!("snapshot startup failed")));
        }
        Ok(writer)
    }

    fn finish(mut self, encode_result: Result<()>) -> Result<LeafSnapshot> {
        if encode_result.is_err() {
            self.cancelled.store(true, Ordering::Release);
        }
        self.sender.take();
        let started = Instant::now();
        let result = self
            .handle
            .take()
            .expect("snapshot writer")
            .join()
            .map_err(|_| anyhow!("snapshot writer panicked"))?;
        let finish_wait_ns = started.elapsed().as_nanos() as u64;
        let snapshot = result?;
        encode_result?;
        ensure!(
            self.captured == self.expected,
            "incomplete captured snapshot"
        );
        let snapshot = snapshot.context("snapshot writer cancelled")?;
        log::info!(target: "zigzag_pipeline", "zigzag_pipeline {}", serde_json::json!({
            "layer": self.layer, "mode": "stream", "state": "writer_joined",
            "snapshot_buffer_wait": self.buffer_wait, "writer_finish_wait_ns": finish_wait_ns,
            "scope": "encoder buffer starvation and final writer join; nested in encode/capture, not additive to PC1"
        }));
        Ok(snapshot)
    }
}

impl vde::EncodedSink for SnapshotWriter {
    fn capture(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        let mut waiting = None;
        let mut buffer = loop {
            ensure!(
                !self.cancelled.load(Ordering::Acquire),
                "snapshot writer cancelled"
            );
            match self.available.try_recv() {
                Ok(buffer) => break buffer,
                Err(mpsc::TryRecvError::Disconnected) => bail!("snapshot buffer pool closed"),
                Err(mpsc::TryRecvError::Empty) => {}
            }
            waiting.get_or_insert_with(Instant::now);
            match self.available.recv_timeout(Duration::from_millis(50)) {
                Ok(buffer) => break buffer,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(error) => return Err(error.into()),
            }
        };
        if let Some(started) = waiting {
            self.buffer_wait.record(started);
        }
        ensure!(bytes.len() <= buffer.len(), "encoded chunk exceeds buffer");
        buffer[..bytes.len()].copy_from_slice(bytes);
        self.sender
            .as_ref()
            .context("closed snapshot writer")?
            .send(Chunk {
                layer: self.layer,
                offset,
                length: bytes.len(),
                bytes: buffer,
            })?;
        self.captured += bytes.len();
        Ok(())
    }
}

impl Drop for SnapshotWriter {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.cancelled.store(true, Ordering::Release);
            self.sender.take();
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests;
