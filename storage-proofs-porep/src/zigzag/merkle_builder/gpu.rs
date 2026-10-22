//! Optional bounded Poseidon U2 batches for the existing binary DiskStore.
//! One worker owns Neptune for a PC1, reusing its program/constants across layers.
//! No full level is materialized; the file layout and publication remain shared
//! with the CPU builder. GPU errors are fatal to this generation (no fallback).

use super::*;
use generic_array::GenericArray;
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEFAULT_BATCH: usize = 262_144;
const MAX_BATCH: usize = 1_048_576;
const DEFAULT_CPU_THRESHOLD: usize = 4096;
const DIRTY_BYTES: usize = 64 << 20;
// input bytes + Fr pairs + serialized output + Neptune's result and its clone.
const HOST_BYTES_PER_PAIR: usize = 64 + 64 + 32 + 32 + 32;
const DEVICE_BYTES_PER_PAIR: usize = 64 + 32;
const CONSTANTS_ALLOWANCE: usize = 16 << 20;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct TreeRGpuConfig {
    pub enabled: bool,
    pub batch_size: usize,
    pub cpu_threshold: usize,
}

impl TreeRGpuConfig {
    pub fn from_env() -> Result<Self> {
        let enabled = match std::env::var("FIL_PROOFS_ZIGZAG_TREE_R_GPU") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "0" => false,
            Ok(value) if value == "1" => true,
            _ => bail!("FIL_PROOFS_ZIGZAG_TREE_R_GPU must be 0 or 1"),
        };
        let config = Self {
            enabled,
            batch_size: setting("FIL_PROOFS_ZIGZAG_TREE_R_GPU_BATCH_SIZE", DEFAULT_BATCH)?,
            cpu_threshold: setting(
                "FIL_PROOFS_ZIGZAG_TREE_R_GPU_CPU_THRESHOLD",
                DEFAULT_CPU_THRESHOLD,
            )?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<()> {
        ensure!(
            (1..=MAX_BATCH).contains(&self.batch_size),
            "GPU TreeR batch must be 1..={MAX_BATCH}"
        );
        ensure!(
            self.cpu_threshold <= MAX_BATCH,
            "GPU TreeR CPU threshold exceeds {MAX_BATCH}"
        );
        Ok(())
    }

    pub(crate) fn start<Tree: 'static + MerkleTreeTrait>(
        builder: TreeRBuilder,
        cache: Option<&std::path::Path>,
    ) -> Result<Option<Arc<TreeGpuWorker>>> {
        let config = Self::from_env()?;
        if !config.enabled {
            return Ok(None);
        }
        ensure!(
            builder == TreeRBuilder::Binary && supports_binary::<Tree>() && cache.is_some(),
            "GPU TreeR requires the binary builder and a Linux binary Poseidon disk cache"
        );
        crate::zigzag::gpu::validate_tree_gpu_backend()?;
        Ok(Some(Arc::new(TreeGpuWorker::start(config)?)))
    }
}

fn setting(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(default),
        Ok(value) => {
            ensure!(
                !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
                "invalid {name}"
            );
            value.parse().with_context(|| format!("invalid {name}"))
        }
        Err(error) => Err(error.into()),
    }
}

#[derive(Default, Debug, Serialize)]
pub(super) struct GpuBuildStats {
    gpu_hashes: u64,
    cpu_hashes: u64,
    batches: u64,
    initialization_ns: u64,
    admission_ns: u64,
    // Neptune's public API combines allocation, H2D, kernel and D2H. The
    // external CUDA trace separates them; this interval is not kernel time.
    batch_call_ns: u64,
    serialize_ns: u64,
    h2d_payload_bytes: u64,
    d2h_payload_bytes: u64,
    host_buffer_peak_bound_bytes: usize,
    device_buffer_estimate_bytes: usize,
    cpu_threshold: usize,
}

struct Job {
    snapshot: LeafSnapshot,
    layer: usize,
    cancelled: Arc<AtomicBool>,
    reply: mpsc::SyncSender<Result<CompletedTree>>,
}

pub(crate) struct TreeGpuWorker {
    sender: Option<mpsc::SyncSender<Job>>,
    handle: Option<JoinHandle<()>>,
}

impl TreeGpuWorker {
    fn start(config: TreeRGpuConfig) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Job>(1);
        let (ready_tx, ready_rx) = mpsc::sync_channel(0);
        let handle = thread::Builder::new()
            .name("zigzag-poseidon".into())
            .spawn(move || {
                let admission = crate::zigzag::gpu::ProvingGuard::enter_for("tree_r");
                let _admission = match admission {
                    Ok(guard) => {
                        if ready_tx.send(Ok(())).is_err() {
                            return;
                        }
                        guard
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                let mut engine = Engine::new(config.batch_size);
                loop {
                    let job = match receiver.recv_timeout(Duration::from_millis(50)) {
                        Ok(job) => job,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            // Free persistent program/constants while idle when PoSt
                            // requests priority. Bulk buffers only live within hash().
                            if priority_taken().unwrap_or(true) {
                                engine.release();
                            }
                            continue;
                        }
                    };
                    let Job {
                        snapshot,
                        layer,
                        cancelled,
                        reply,
                    } = job;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let operation = OperationGuard::enter("tree_r_build", Some(layer));
                        let ((root, mut stats), dirty_stats) = with_dirty_budget(
                            snapshot.file.as_file(),
                            DIRTY_BYTES,
                            &cancelled,
                            |dirty| {
                                build_batched_levels(
                                    snapshot.file.as_file(),
                                    snapshot.geometry,
                                    config,
                                    &cancelled,
                                    Some(dirty),
                                    |pairs, stats| engine.hash(pairs, &cancelled, stats),
                                )
                            },
                        )?;
                        stats.dirty_budget = Some(dirty_stats);
                        check_cancelled(&cancelled)?;
                        operation.finish();
                        Ok(CompletedTree {
                            snapshot,
                            root,
                            stats,
                        })
                    }))
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("GPU TreeR worker panicked")));
                    let failed = result.is_err();
                    if failed {
                        cancelled.store(true, Ordering::Release);
                    }
                    if reply.send(result).is_err() || failed {
                        break;
                    }
                }
                // Drop Neptune before releasing the common ZigZag device admission.
                drop(engine);
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: Some(sender),
                handle: Some(handle),
            }),
            result => {
                drop(sender);
                let _ = handle.join();
                match result {
                    Ok(Err(error)) => Err(error),
                    _ => bail!("GPU TreeR worker failed during admission"),
                }
            }
        }
    }

    pub(crate) fn build(
        &self,
        snapshot: LeafSnapshot,
        layer: usize,
        cancelled: Arc<AtomicBool>,
    ) -> Result<CompletedTree> {
        let (reply, receiver) = mpsc::sync_channel(0);
        self.sender
            .as_ref()
            .context("GPU TreeR worker closed")?
            .send(Job {
                snapshot,
                layer,
                cancelled,
                reply,
            })
            .map_err(|_| anyhow::anyhow!("GPU TreeR worker stopped"))?;
        receiver
            .recv()
            .context("GPU TreeR worker stopped before returning its store")?
    }
}

impl Drop for TreeGpuWorker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                log::error!("GPU TreeR worker panicked during shutdown");
            }
        }
    }
}

struct Engine {
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    batcher: Option<neptune::batch_hasher::Batcher<Scalar, U2>>,
    batch_size: usize,
}

impl Engine {
    fn new(batch_size: usize) -> Self {
        Self {
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            batcher: None,
            batch_size,
        }
    }

    fn release(&mut self) {
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        {
            self.batcher.take();
        }
    }

    fn hash(
        &mut self,
        pairs: &[GenericArray<Scalar, U2>],
        cancelled: &AtomicBool,
        stats: &mut GpuBuildStats,
    ) -> Result<Vec<Scalar>> {
        ensure!(pairs.len() <= self.batch_size, "GPU TreeR batch overflow");
        let started = Instant::now();
        let _lease = batch_lease(cancelled, || self.release())?;
        stats.admission_ns += started.elapsed().as_nanos() as u64;
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        {
            use ec_gpu_gen::rust_gpu_tools::Device;
            use neptune::{batch_hasher::Batcher, BatchHasher};
            if self.batcher.is_none() {
                let started = Instant::now();
                let devices = Device::all();
                ensure!(devices.len() == 1, "GPU TreeR requires one visible device");
                let required = self.batch_size * DEVICE_BYTES_PER_PAIR + CONSTANTS_ALLOWANCE;
                ensure!(
                    devices[0].memory() >= required as u64,
                    "GPU too small for TreeR batch buffers"
                );
                // Same MerkleTree U2 domain and strength as POSEIDON_CONSTANTS_2.
                self.batcher = Some(Batcher::with_strength(
                    devices[0],
                    POSEIDON_CONSTANTS_2.strength,
                    self.batch_size,
                )?);
                stats.initialization_ns += started.elapsed().as_nanos() as u64;
            }
            let started = Instant::now();
            let result = self
                .batcher
                .as_mut()
                .expect("initialized GPU batcher")
                .hash(pairs)
                .context("GPU Poseidon U2 batch failed; generation aborted")?;
            stats.batch_call_ns += started.elapsed().as_nanos() as u64;
            // Release persistent context/constants before yielding the GPU lock
            // when a priority request arrived during this bounded batch.
            if priority_taken()? {
                self.release();
            }
            Ok(result)
        }
        #[cfg(not(any(feature = "cuda", feature = "opencl")))]
        bail!("GPU TreeR is unavailable in a CPU-only build")
    }
}

fn lock_file(directory: &std::path::Path, name: &str) -> Result<File> {
    Ok(std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(directory.join(name))?)
}

fn priority_taken() -> Result<bool> {
    priority_taken_in(&std::env::temp_dir())
}

fn priority_taken_in(directory: &std::path::Path) -> Result<bool> {
    use fs2::FileExt;
    let priority = lock_file(directory, "bellman.priority.lock")?;
    match FileExt::try_lock_shared(&priority) {
        Ok(()) => Ok(false),
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(true),
        Err(error) => Err(error.into()),
    }
}

/// Bellperson 0.27's priority flag is private. Follow its public on-disk
/// protocol, returning I/O errors rather than panicking. Hold the global GPU
/// lock only for one bounded batch; a PoSt request can run between batches.
fn batch_lease(cancelled: &AtomicBool, on_priority: impl FnMut()) -> Result<File> {
    batch_lease_in(&std::env::temp_dir(), cancelled, on_priority)
}

fn batch_lease_in(
    directory: &std::path::Path,
    cancelled: &AtomicBool,
    mut on_priority: impl FnMut(),
) -> Result<File> {
    use fs2::FileExt;
    let priority = lock_file(directory, "bellman.priority.lock")?;
    let gpu = lock_file(directory, "bellman.gpu.lock")?;
    loop {
        check_cancelled(cancelled)?;
        match FileExt::try_lock_shared(&priority) {
            Ok(()) => {
                let acquired = FileExt::try_lock_exclusive(&gpu);
                FileExt::unlock(&priority)?;
                match acquired {
                    Ok(()) => return Ok(gpu),
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => return Err(error.into()),
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => on_priority(),
            Err(error) => return Err(error.into()),
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn build_batched_levels(
    io: &impl PositionedIo,
    geometry: Geometry,
    config: TreeRGpuConfig,
    cancelled: &AtomicBool,
    dirty: Option<&DirtyBudget<'_>>,
    mut hash: impl FnMut(&[GenericArray<Scalar, U2>], &mut GpuBuildStats) -> Result<Vec<Scalar>>,
) -> Result<([u8; 32], BuildStats)> {
    config.validate()?;
    check_cancelled(cancelled)?;
    let capacity = config.batch_size.min(geometry.leaves / 2);
    let mut input = vec![0u8; capacity * 64];
    let mut pairs = vec![GenericArray::<Scalar, U2>::default(); capacity];
    let mut output = vec![0u8; capacity * 32];
    let mut stats = BuildStats {
        workers: 1,
        chunk_pairs: capacity,
        buffer_allocations: 3,
        buffer_bytes: capacity * 160,
        ..Default::default()
    };
    let mut gpu = GpuBuildStats {
        host_buffer_peak_bound_bytes: capacity * HOST_BYTES_PER_PAIR + CONSTANTS_ALLOWANCE,
        // This covers explicit buffers and a constants allowance, NOT driver
        // context/module overhead. Device-wide VRAM is measured by the runner.
        device_buffer_estimate_bytes: capacity * DEVICE_BYTES_PER_PAIR + CONSTANTS_ALLOWANCE,
        cpu_threshold: config.cpu_threshold,
        ..Default::default()
    };
    let mut poseidon = Poseidon::new(&POSEIDON_CONSTANTS_2);
    let mut width = geometry.leaves;
    let mut read_start = 0;
    while width > 1 {
        let parents = width / 2;
        let write_start = read_start + width;
        for first in (0..parents).step_by(capacity) {
            check_cancelled(cancelled)?;
            let count = capacity.min(parents - first);
            let started = Instant::now();
            read_exact_at(
                io,
                &mut input[..count * 64],
                ((read_start + first * 2) * NODE_BYTES) as u64,
            )?;
            stats.read_ns += started.elapsed().as_nanos() as u64;
            let started = Instant::now();
            for (pair, bytes) in pairs[..count].iter_mut().zip(input.chunks_exact(64)) {
                for (field, raw) in pair.iter_mut().zip(bytes.chunks_exact(32)) {
                    let mut repr = [0u8; 32];
                    repr.copy_from_slice(raw);
                    *field = Scalar::from_repr_vartime(repr)
                        .context("noncanonical TreeR field element")?;
                }
            }
            stats.conversion_ns += started.elapsed().as_nanos() as u64;
            let started = Instant::now();
            if parents > config.cpu_threshold {
                let result = hash(&pairs[..count], &mut gpu)?;
                ensure!(
                    result.len() == count,
                    "GPU TreeR returned the wrong number of hashes"
                );
                let serialization = Instant::now();
                for (field, bytes) in result.iter().zip(output.chunks_exact_mut(32)) {
                    bytes.copy_from_slice(&field.to_repr());
                }
                gpu.serialize_ns += serialization.elapsed().as_nanos() as u64;
                gpu.gpu_hashes += count as u64;
                gpu.batches += 1;
                gpu.h2d_payload_bytes += (count * 64) as u64;
                gpu.d2h_payload_bytes += (count * 32) as u64;
            } else {
                for (pair, bytes) in pairs[..count].iter().zip(output.chunks_exact_mut(32)) {
                    poseidon.set_preimage(pair.as_slice());
                    bytes.copy_from_slice(&poseidon.hash().to_repr());
                }
                gpu.cpu_hashes += count as u64;
            }
            stats.hash_serialize_ns += started.elapsed().as_nanos() as u64;
            check_cancelled(cancelled)?;
            let started = Instant::now();
            let offset = ((write_start + first) * NODE_BYTES) as u64;
            match dirty {
                Some(dirty) => dirty.write(io, &output[..count * 32], offset)?,
                None => write_all_at(io, &output[..count * 32], offset)?,
            }
            stats.write_ns += started.elapsed().as_nanos() as u64;
            stats.pairs += count as u64;
            stats.read_bytes += (count * 64) as u64;
            stats.written_bytes += (count * 32) as u64;
        }
        read_start = write_start;
        width = parents;
    }
    ensure!(
        stats.pairs == (geometry.leaves - 1) as u64,
        "incomplete GPU TreeR build"
    );
    let mut root = [0u8; 32];
    read_exact_at(io, &mut root, geometry.file_bytes - 32)?;
    stats.gpu = Some(gpu);
    Ok((root, stats))
}

#[cfg(test)]
mod tests;
