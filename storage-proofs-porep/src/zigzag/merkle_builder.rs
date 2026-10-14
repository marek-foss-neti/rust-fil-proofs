//! Bounded CPU construction of ZigZag's full binary Poseidon DiskStore.
//!
//! Adapts the level barriers and independent chunk work of merkletree's DiskStore
//! builder used by SDR. The shared builder, hashers and store format are unchanged.
//! A private, physically allocated snapshot owns the leaves before any parents are
//! built. Completed stores are published without replacing existing artifacts.

use std::any::TypeId;
use std::convert::TryFrom;
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use blstrs::Scalar;
use ff::{Field, PrimeField};
use filecoin_hashers::{
    poseidon::{PoseidonDomain, PoseidonHasher},
    Hasher, POSEIDON_CONSTANTS_2,
};
use generic_array::typenum::{U0, U2};
use merkletree::{
    merkle::MerkleTree,
    store::{DiskStore, Store, StoreConfig},
};
use neptune::poseidon::Poseidon;
use rayon::prelude::*;
use serde::Serialize;
use storage_proofs_core::merkle::{create_base_merkle_tree, MerkleTreeTrait, MerkleTreeWrapper};
use tempfile::NamedTempFile;

use super::measurements::OperationGuard;

const NODE_BYTES: usize = 32;
const CHUNK_PAIRS: usize = 4096;
const MAX_WORKERS: usize = 64;
const SNAPSHOT_CHUNK_BYTES: usize = 1 << 20;
type BinaryTree = MerkleTreeWrapper<PoseidonHasher, DiskStore<PoseidonDomain>, U2, U0, U0>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TreeRBuilder {
    Binary,
    Legacy,
}

impl TreeRBuilder {
    pub fn from_env() -> Result<Self> {
        match std::env::var("FIL_PROOFS_ZIGZAG_TREE_R_BUILDER") {
            // Keep the comparison baseline until a full sealing run validates
            // the specialized path's end-to-end performance, including C1 I/O.
            Err(std::env::VarError::NotPresent) => Ok(Self::Legacy),
            Ok(value) if value == "binary" => Ok(Self::Binary),
            Ok(value) if value == "legacy" => Ok(Self::Legacy),
            _ => bail!("FIL_PROOFS_ZIGZAG_TREE_R_BUILDER must be binary or legacy"),
        }
    }
}

/// Only the actual ZigZag disk-tree shape uses the specialized path. In-memory
/// and other generic callers retain the shared builder, without changing arity.
pub(crate) fn build_tree_r<Tree: 'static + MerkleTreeTrait>(
    builder: TreeRBuilder,
    config: Option<StoreConfig>,
    leaves: usize,
    data: &[u8],
    layer: usize,
) -> Result<Tree> {
    if builder == TreeRBuilder::Legacy
        || !cfg!(target_os = "linux")
        || config.is_none()
        || TypeId::of::<Tree>() != TypeId::of::<BinaryTree>()
    {
        log::info!(target: "zigzag_merkle", "zigzag_merkle {}", serde_json::json!({
            "builder": "legacy", "layer": layer,
            "reason": if builder == TreeRBuilder::Legacy { "explicit reference" } else { "unsupported platform, shape or non-disk request" },
        }));
        return create_base_merkle_tree::<Tree>(config, leaves, data);
    }
    let cancelled = AtomicBool::new(false);
    let snapshot = LeafSnapshot::capture(
        config.context("missing TreeR config")?,
        leaves,
        data,
        layer,
        &cancelled,
    )?;
    let completed = snapshot.build(layer, &cancelled)?;
    completed.publish::<Tree>(layer, &cancelled)
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    leaves: usize,
    elements: usize,
    leaf_bytes: usize,
    file_bytes: u64,
}

impl Geometry {
    fn new(leaves: usize) -> Result<Self> {
        ensure!(
            leaves >= 2 && leaves.is_power_of_two(),
            "binary TreeR requires a power-of-two leaf count >= 2"
        );
        let elements = leaves
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .context("TreeR element count overflow")?;
        let leaf_bytes = leaves
            .checked_mul(NODE_BYTES)
            .context("TreeR leaf size overflow")?;
        let file_bytes = elements
            .checked_mul(NODE_BYTES)
            .context("TreeR file size overflow")?;
        let file_bytes = u64::try_from(file_bytes)?;
        ensure!(
            file_bytes <= i64::MAX as u64,
            "TreeR exceeds file offset range"
        );
        Ok(Self {
            leaves,
            elements,
            leaf_bytes,
            file_bytes,
        })
    }
}

/// The private file contains a complete immutable leaf snapshot. No later encode
/// can affect it. Keeping this boundary explicit permits bounded overlap later.
pub(crate) struct LeafSnapshot {
    file: NamedTempFile,
    config: StoreConfig,
    geometry: Geometry,
}

impl LeafSnapshot {
    pub(crate) fn capture(
        config: StoreConfig,
        leaves: usize,
        data: &[u8],
        layer: usize,
        cancelled: &AtomicBool,
    ) -> Result<Self> {
        let geometry = Geometry::new(leaves)?;
        ensure!(
            data.len() == geometry.leaf_bytes,
            "TreeR leaf data length mismatch"
        );
        ensure!(
            config.rows_to_discard == 0,
            "binary TreeR must retain every row"
        );
        ensure!(
            config.size.is_none_or(|n| n == geometry.elements),
            "TreeR StoreConfig size mismatch"
        );
        check_cancelled(cancelled)?;
        let path = StoreConfig::data_path(&config.path, &config.id);
        match fs::symlink_metadata(&path) {
            Ok(_) => bail!(
                "refusing to replace an existing TreeR store: {}",
                path.display()
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let operation = OperationGuard::enter("tree_r_snapshot", Some(layer));
        let file = tempfile::Builder::new()
            .prefix(".zigzag-tree-r-")
            .tempfile_in(&config.path)?;
        reserve_file(file.as_file(), geometry.file_bytes).context("preallocate TreeR store")?;
        for (index, chunk) in data.chunks(SNAPSHOT_CHUNK_BYTES).enumerate() {
            check_cancelled(cancelled)?;
            write_all_at(file.as_file(), chunk, (index * SNAPSHOT_CHUNK_BYTES) as u64)?;
        }
        operation.finish();
        Ok(Self {
            file,
            config,
            geometry,
        })
    }

    pub(crate) fn build(self, layer: usize, cancelled: &AtomicBool) -> Result<CompletedTree> {
        let operation = OperationGuard::enter("tree_r_build", Some(layer));
        let (root, stats) = build_levels(
            self.file.as_file(),
            self.geometry,
            CHUNK_PAIRS,
            MAX_WORKERS,
            cancelled,
        )?;
        check_cancelled(cancelled)?;
        operation.finish();
        Ok(CompletedTree {
            snapshot: self,
            root,
            stats,
        })
    }
}

pub(crate) struct CompletedTree {
    snapshot: LeafSnapshot,
    root: [u8; NODE_BYTES],
    stats: BuildStats,
}

impl CompletedTree {
    pub(crate) fn publish<Tree: 'static + MerkleTreeTrait>(
        self,
        layer: usize,
        cancelled: &AtomicBool,
    ) -> Result<Tree> {
        check_cancelled(cancelled)?;
        ensure!(
            TypeId::of::<Tree>() == TypeId::of::<BinaryTree>(),
            "unsupported binary TreeR type"
        );
        let Self {
            snapshot,
            root,
            stats,
        } = self;
        let LeafSnapshot {
            file,
            config,
            geometry,
        } = snapshot;
        let operation = OperationGuard::enter("tree_r_sync", Some(layer));
        ensure!(
            file.as_file().metadata()?.len() == geometry.file_bytes,
            "TreeR store length changed"
        );
        file.as_file()
            .sync_all()
            .context("sync complete TreeR store")?;
        check_cancelled(cancelled)?;
        let path = StoreConfig::data_path(&config.path, &config.id);
        // No partial file is visible under the StoreConfig name. In particular,
        // an interrupted build can never be mistaken for a finished DiskStore.
        let persisted = file.persist_noclobber(&path).map_err(|error| error.error)?;
        let mut pending = PendingPublication { path, keep: false };
        File::open(&config.path)?
            .sync_all()
            .context("sync TreeR directory")?;
        let store = Tree::Store::new_from_disk(geometry.elements, 2, &config)?;
        let tree = Tree::from_merkle(MerkleTree::from_data_store(store, geometry.leaves)?);
        ensure!(
            <<Tree::Hasher as Hasher>::Domain as AsRef<[u8]>>::as_ref(&tree.root()) == root,
            "reopened TreeR root mismatch"
        );
        drop(persisted);
        pending.keep = true;
        operation.finish();
        log::info!(target: "zigzag_merkle", "zigzag_merkle {}", serde_json::json!({
            "builder": "binary", "layer": layer, "leaves": geometry.leaves,
            "file_bytes": geometry.file_bytes, "stats": stats,
            "timing_scope": "read/conversion/hash_serialize/write: summed hash-worker intervals; writeback_submit: one I/O thread; overlapping, not additive to TreeR wall time",
            "allocation_scope": "builder-owned buffers only; excludes shared Poseidon constants, Rayon and page cache",
        }));
        Ok(tree)
    }
}

struct PendingPublication {
    path: PathBuf,
    keep: bool,
}
impl Drop for PendingPublication {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn reserve_file(file: &File, bytes: u64) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let len = libc::off_t::try_from(bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "TreeR exceeds off_t"))?;
        loop {
            // SAFETY: an owned descriptor and a positive, checked file length.
            if unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, len) } == 0 {
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, bytes);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "binary TreeR requires Linux fallocate",
        ))
    }
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    ensure!(
        !cancelled.load(Ordering::Relaxed),
        "TreeR construction cancelled"
    );
    Ok(())
}

// Positioned bounded I/O avoids writable mmap entirely. Short writes, EINTR,
// ENOSPC and read failures remain ordinary Results, including after reservation.
trait PositionedIo: Sync {
    fn read_at(&self, buffer: &mut [u8], offset: u64) -> io::Result<usize>;
    fn write_at(&self, buffer: &[u8], offset: u64) -> io::Result<usize>;
    fn start_writeback(&self, offset: u64, bytes: u64) -> io::Result<()>;
}
impl PositionedIo for File {
    fn read_at(&self, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
        FileExt::read_at(self, buffer, offset)
    }
    fn write_at(&self, buffer: &[u8], offset: u64) -> io::Result<usize> {
        FileExt::write_at(self, buffer, offset)
    }
    fn start_writeback(&self, offset: u64, bytes: u64) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let offset = libc::off_t::try_from(offset)
                .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
            let bytes = libc::off_t::try_from(bytes)
                .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
            loop {
                // SAFETY: an owned regular-file descriptor and checked ranges.
                // This only submits writeback, and may block on a full I/O queue.
                // It is NOT a durability barrier: publish still syncs the entire
                // file and directory, including metadata and device caches.
                if unsafe {
                    libc::sync_file_range(
                        self.as_raw_fd(),
                        offset,
                        bytes,
                        libc::SYNC_FILE_RANGE_WRITE,
                    )
                } == 0
                {
                    return Ok(());
                }
                let error = io::Error::last_os_error();
                match error.raw_os_error() {
                    Some(libc::EINTR) => continue,
                    Some(libc::ENOSYS | libc::EOPNOTSUPP) => {
                        log::debug!(target: "zigzag_merkle", "writeback hint unavailable: {error}");
                        return Ok(());
                    }
                    _ => return Err(error),
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (offset, bytes);
            Ok(())
        }
    }
}

fn read_exact_at(io: &impl PositionedIo, mut bytes: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        match io.read_at(bytes, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short TreeR read",
                ))
            }
            Ok(count) => {
                offset += count as u64;
                bytes = &mut bytes[count..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn write_all_at(io: &impl PositionedIo, mut bytes: &[u8], mut offset: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        match io.write_at(bytes, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "short TreeR write",
                ))
            }
            Ok(count) => {
                offset += count as u64;
                bytes = &bytes[count..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[derive(Default, Debug, Serialize)]
struct BuildStats {
    workers: usize,
    chunk_pairs: usize,
    buffer_bytes: usize,
    buffer_allocations: usize,
    pairs: u64,
    read_bytes: u64,
    written_bytes: u64,
    read_ns: u64,
    conversion_ns: u64,
    hash_serialize_ns: u64,
    write_ns: u64,
    // One I/O thread, separate from the summed hash-worker intervals above.
    writeback_submit_ns: u64,
}

struct Worker {
    input: Vec<u8>,
    pairs: Vec<[Scalar; 2]>,
    output: Vec<u8>,
    stats: BuildStats,
}

impl Worker {
    fn new(pairs: usize) -> Self {
        Self {
            input: vec![0; pairs * 64],
            pairs: vec![[Scalar::ZERO; 2]; pairs],
            output: vec![0; pairs * 32],
            stats: BuildStats::default(),
        }
    }

    fn process(
        &mut self,
        io: &impl PositionedIo,
        count: usize,
        read: u64,
        write: u64,
    ) -> Result<()> {
        let started = Instant::now();
        read_exact_at(io, &mut self.input[..count * 64], read)?;
        self.stats.read_ns += started.elapsed().as_nanos() as u64;
        let started = Instant::now();
        for (pair, raw) in self.pairs[..count]
            .iter_mut()
            .zip(self.input.chunks_exact(64))
        {
            for (field, bytes) in pair.iter_mut().zip(raw.chunks_exact(32)) {
                let mut repr = [0; 32];
                repr.copy_from_slice(bytes);
                *field =
                    Scalar::from_repr_vartime(repr).context("noncanonical TreeR field element")?;
            }
        }
        self.stats.conversion_ns += started.elapsed().as_nanos() as u64;
        let started = Instant::now();
        // Keep the hot mutable hash state on this worker's stack, rather than
        // in adjacent heap-resident Worker records sharing cache lines.
        let mut poseidon = Poseidon::new(&POSEIDON_CONSTANTS_2);
        for (pair, output) in self.pairs[..count]
            .iter()
            .zip(self.output.chunks_exact_mut(32))
        {
            // Same MerkleTree U2 constants, strength and domain tag as
            // PoseidonFunction::node/multi_node; no new leaf hash or padding.
            poseidon.set_preimage(pair);
            output.copy_from_slice(&poseidon.hash().to_repr());
        }
        self.stats.hash_serialize_ns += started.elapsed().as_nanos() as u64;
        let started = Instant::now();
        write_all_at(io, &self.output[..count * 32], write)?;
        self.stats.write_ns += started.elapsed().as_nanos() as u64;
        self.stats.pairs += count as u64;
        self.stats.read_bytes += (count * 64) as u64;
        self.stats.written_bytes += (count * 32) as u64;
        Ok(())
    }
}

fn build_levels(
    io: &impl PositionedIo,
    geometry: Geometry,
    chunk_pairs: usize,
    max_workers: usize,
    cancelled: &AtomicBool,
) -> Result<([u8; 32], BuildStats)> {
    // Linux may block even when submitting writeback without WAIT flags. One
    // scoped thread overlaps that wait with hashing; a one-entry queue bounds
    // outstanding ranges. It never reads/writes caller-owned replica bytes.
    std::thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<(u64, u64)>(1);
        let writeback = std::thread::Builder::new()
            .name("zigzag-tree-writeback".into())
            .spawn_scoped(scope, || -> io::Result<u64> {
                let mut elapsed = 0;
                for (offset, bytes) in receiver {
                    if cancelled.load(Ordering::Relaxed) {
                        break;
                    }
                    let started = Instant::now();
                    if let Err(error) = io.start_writeback(offset, bytes) {
                        cancelled.store(true, Ordering::Relaxed);
                        return Err(error);
                    }
                    elapsed += started.elapsed().as_nanos() as u64;
                }
                Ok(elapsed)
            })?;
        let result = build_levels_inner(io, geometry, chunk_pairs, max_workers, cancelled, &sender);
        if result.is_err() {
            cancelled.store(true, Ordering::Relaxed);
        }
        drop(sender);
        // Always join before propagating an error and unlinking the private file.
        let elapsed = writeback
            .join()
            .map_err(|_| anyhow::anyhow!("TreeR writeback panicked"))??;
        let (root, mut stats) = result?;
        check_cancelled(cancelled)?;
        stats.writeback_submit_ns = elapsed;
        Ok((root, stats))
    })
}

fn build_levels_inner(
    io: &impl PositionedIo,
    geometry: Geometry,
    chunk_pairs: usize,
    max_workers: usize,
    cancelled: &AtomicBool,
    writeback: &std::sync::mpsc::SyncSender<(u64, u64)>,
) -> Result<([u8; 32], BuildStats)> {
    check_cancelled(cancelled)?;
    ensure!(
        chunk_pairs > 0
            && chunk_pairs <= CHUNK_PAIRS
            && max_workers > 0
            && max_workers <= MAX_WORKERS,
        "invalid TreeR buffer bounds"
    );
    let chunks = (geometry.leaves / 2).div_ceil(chunk_pairs);
    let workers = rayon::current_num_threads()
        .min(max_workers)
        .min(chunks)
        .max(1);
    let capacity = chunk_pairs.min(geometry.leaves / 2);
    let mut states: Vec<_> = (0..workers).map(|_| Worker::new(capacity)).collect();
    writeback.send((0, geometry.leaf_bytes as u64))?;
    let mut width = geometry.leaves;
    let mut read_start = 0;
    while width > 1 {
        check_cancelled(cancelled)?;
        let parents = width / 2;
        let write_start = read_start + width;
        let next = AtomicUsize::new(0);
        let error = Mutex::new(None);
        // At most `workers` jobs and three bounded buffers each. Worker buffers
        // survive level barriers; there is no Vec of all nodes or chunk tasks.
        states.par_iter_mut().for_each(|worker| {
            while !cancelled.load(Ordering::Relaxed) {
                let first = next.fetch_add(capacity, Ordering::Relaxed);
                if first >= parents {
                    break;
                }
                let count = capacity.min(parents - first);
                let read = ((read_start + first * 2) * NODE_BYTES) as u64;
                let write = ((write_start + first) * NODE_BYTES) as u64;
                if let Err(failure) = worker.process(io, count, read, write) {
                    let mut first_error = error
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if first_error.is_none() {
                        *first_error = Some(failure);
                    }
                    cancelled.store(true, Ordering::Relaxed);
                    break;
                }
            }
        });
        // Rayon has joined every worker, including on failure, before any file
        // cleanup or access to the next level can happen.
        if let Some(error) = error
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Err(error);
        }
        check_cancelled(cancelled)?;
        // Submit completed levels while the CPU hashes the next one, avoiding
        // deferring all buffered I/O to the final sync. No page is discarded.
        writeback.send((
            (write_start * NODE_BYTES) as u64,
            (parents * NODE_BYTES) as u64,
        ))?;
        read_start = write_start;
        width = parents;
    }
    let mut root = [0; 32];
    read_exact_at(io, &mut root, geometry.file_bytes - 32)?;
    let mut total = BuildStats {
        workers,
        chunk_pairs: capacity,
        buffer_allocations: 3 * workers,
        ..Default::default()
    };
    for worker in states {
        total.buffer_bytes += worker.input.capacity()
            + worker.output.capacity()
            + worker.pairs.capacity() * std::mem::size_of::<[Scalar; 2]>();
        total.pairs += worker.stats.pairs;
        total.read_bytes += worker.stats.read_bytes;
        total.written_bytes += worker.stats.written_bytes;
        total.read_ns += worker.stats.read_ns;
        total.conversion_ns += worker.stats.conversion_ns;
        total.hash_serialize_ns += worker.stats.hash_serialize_ns;
        total.write_ns += worker.stats.write_ns;
    }
    ensure!(
        total.pairs == (geometry.leaves - 1) as u64,
        "incomplete TreeR build"
    );
    Ok((root, total))
}

#[cfg(test)]
mod tests;
