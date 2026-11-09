//! Optional ZigZag CPU decoder experiments. No change to the SDR KDF or field path.
use std::{
    convert::TryInto,
    mem::size_of,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[path = "decode_cpu/field.rs"]
mod field;
use field::{canonical, subtract};

const NODE: usize = 32;
const MAX_PARENTS: usize = 32;
const MAX_INPUT: usize = NODE * (MAX_PARENTS + 1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeField {
    Reference,
    Limbs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeSha {
    Serial,
    Interleaved,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DecodeCpuConfig {
    pub field: DecodeField,
    pub sha: DecodeSha,
    pub batch_nodes: usize,
    pub prefix_reuse: bool,
    pub prefetch: bool,
    /// Zero disables clocks in the node loop. Otherwise sample every Nth batch per worker.
    pub profile_every: usize,
    /// None means auto. Explicit values are also capped by the available CPU/Rayon budget.
    pub threads: Option<usize>,
}

impl Default for DecodeCpuConfig {
    fn default() -> Self {
        Self {
            field: DecodeField::Reference,
            sha: DecodeSha::Serial,
            batch_nodes: 1,
            prefix_reuse: false,
            prefetch: false,
            profile_every: 0,
            threads: None,
        }
    }
}

impl DecodeCpuConfig {
    pub fn from_env() -> Result<Self> {
        fn env(key: &str, default: &str) -> Result<String> {
            match std::env::var(format!("FIL_PROOFS_ZIGZAG_DECODE_{key}")) {
                Ok(value) => Ok(value),
                Err(std::env::VarError::NotPresent) => Ok(default.into()),
                Err(error) => Err(error).context("invalid ZigZag CPU decode environment"),
            }
        }
        Self::parse(
            &env("FIELD", "reference")?,
            &env("SHA", "serial")?,
            &env("BATCH_NODES", "1")?,
            &env("PREFIX_REUSE", "0")?,
            &env("PREFETCH", "0")?,
            &env("PROFILE_EVERY", "0")?,
            &env("THREADS", "auto")?,
        )
    }

    fn parse(
        field: &str,
        sha: &str,
        batch: &str,
        prefix: &str,
        prefetch: &str,
        profile: &str,
        threads: &str,
    ) -> Result<Self> {
        fn flag(value: &str) -> Result<bool> {
            match value {
                "0" => Ok(false),
                "1" => Ok(true),
                _ => anyhow::bail!("decode CPU flags must be 0 or 1"),
            }
        }
        let config = Self {
            field: match field {
                "reference" => DecodeField::Reference,
                "limbs" => DecodeField::Limbs,
                _ => anyhow::bail!("decode field must be reference or limbs"),
            },
            sha: match sha {
                "serial" => DecodeSha::Serial,
                "interleaved" => DecodeSha::Interleaved,
                _ => anyhow::bail!("decode SHA must be serial or interleaved"),
            },
            batch_nodes: batch.parse().context("invalid decode batch nodes")?,
            prefix_reuse: flag(prefix)?,
            prefetch: flag(prefetch)?,
            profile_every: profile.parse().context("invalid decode profile interval")?,
            threads: if threads == "auto" {
                None
            } else {
                Some(threads.parse().context("invalid decode threads")?)
            },
        };
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn validate(self) -> Result<()> {
        ensure!(
            (1..=32).contains(&self.batch_nodes),
            "decode batch must be 1..=32 nodes"
        );
        ensure!(
            self.profile_every <= 1_048_576,
            "decode profile interval must be 0..=1048576"
        );
        ensure!(
            self.threads.is_none_or(|n| (1..=128).contains(&n)),
            "decode threads must be auto or 1..=128"
        );
        Ok(())
    }

    pub(crate) fn enabled(self) -> bool {
        self.field != DecodeField::Reference
            || self.sha != DecodeSha::Serial
            || self.batch_nodes != 1
            || self.prefix_reuse
            || self.prefetch
            || self.profile_every != 0
            || self.threads.is_some()
    }

    pub(crate) fn worker_limit(self) -> usize {
        // available_parallelism accounts for the process's affinity and CPU quota.
        // Failure is conservative; never replace it with all logical host CPUs.
        self.worker_limit_for(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            rayon::current_num_threads(),
        )
    }

    fn worker_limit_for(self, available: usize, pool: usize) -> usize {
        available
            .max(1)
            .min(pool.max(1))
            .min(self.threads.unwrap_or(128))
            .min(128)
    }

    /// Conservative owned-workspace bound for FFI RAM admission, excluding existing parent buffers.
    pub fn workspace_bytes(self, degree: usize) -> Result<usize> {
        self.validate()?;
        if !self.enabled() || !(1..=MAX_PARENTS).contains(&degree) {
            return Ok(0);
        }
        // Admission precedes decode. Reserve for the configured maximum even
        // if affinity/quota or the Rayon pool changes between the two samples.
        Ok(self.threads.unwrap_or(128)
            * (self.batch_nodes * size_of::<Lane>() + size_of::<CpuWorker>())
            + (degree.div_ceil(2) + 1) * size_of::<Sha256>()
            + size_of::<CpuLayer>())
    }
}

/// Immutable for exactly one encoded layer. Copies the original boundary value
/// before any blockwise copy-back; never shared across layers or replicas.
pub(crate) struct CpuLayer {
    pub config: DecodeCpuConfig,
    nodes: usize,
    degree: usize,
    forward: bool,
    id: [u8; NODE],
    boundary: usize,
    canonical_domain: bool,
    prefixes: Vec<Sha256>,
    prefix_setup_ns: Option<u128>,
}

impl CpuLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: DecodeCpuConfig,
        nodes: usize,
        degree: usize,
        forward: bool,
        id: &[u8],
        state: &[u8],
        canonical_domain: bool,
    ) -> Result<Option<Self>> {
        config.validate()?;
        if !config.enabled() || !(1..=MAX_PARENTS).contains(&degree) || id.len() != NODE {
            return Ok(None);
        }
        ensure!(
            nodes > 0 && nodes.checked_mul(NODE) == Some(state.len()),
            "invalid CPU decode geometry"
        );
        let boundary = if forward { 0 } else { nodes - 1 };
        let id: [u8; NODE] = id.try_into()?;
        let mut prefixes = Vec::with_capacity(if config.prefix_reuse {
            degree.div_ceil(2) + 1
        } else {
            0
        });
        let started = (config.profile_every != 0).then(Instant::now);
        if config.prefix_reuse {
            let mut input = [0u8; MAX_INPUT];
            input[..NODE].copy_from_slice(&id);
            for bytes in input[NODE..(degree + 1) * NODE].chunks_exact_mut(NODE) {
                bytes.copy_from_slice(&state[boundary * NODE..(boundary + 1) * NODE]);
            }
            let mut hasher = Sha256::new();
            prefixes.push(hasher.clone());
            for block in input[..(degree + 1) * NODE].chunks_exact(64) {
                hasher.update(block);
                prefixes.push(hasher.clone());
            }
        }
        Ok(Some(Self {
            config,
            nodes,
            degree,
            forward,
            id,
            boundary,
            canonical_domain,
            prefixes,
            prefix_setup_ns: started.map(|t| t.elapsed().as_nanos()),
        }))
    }

    pub fn log(&self, stats: &CpuStats, workers: usize) {
        log::info!(target: "zigzag_unseal", "zigzag_decode_cpu {}", serde_json::json!({
            "event": "layer", "requested": self.config, "reversed": !self.forward,
            "field": if self.config.field == DecodeField::Limbs && self.canonical_domain { "limbs" } else { "reference" },
            "field_fallback": if self.config.field == DecodeField::Limbs && !self.canonical_domain { Some("unsupported_domain") } else { None },
            "workers": workers, "worker_limit": self.config.worker_limit(),
            "batch_workspace_bytes": workers * self.config.batch_nodes * size_of::<Lane>(),
            "prefix_setup_blocks": self.prefixes.len().saturating_sub(1), "prefix_setup_ns": self.prefix_setup_ns,
            "sha_backend": "sha2::Sha256 runtime dispatch; interleaved is block-round-robin, not a custom SIMD kernel",
            "prefetch_arch_supported": cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
            "stats": stats,
            "timing_scope": "sampled worker wall times; can overlap across workers and include descheduling; not additive to raw unseal",
        }));
    }
}

struct Lane {
    parents: [u32; MAX_PARENTS],
    input: [u8; MAX_INPUT],
    value: [u8; NODE],
    key: [u8; NODE],
    len: usize,
    processed: usize,
    hasher: Sha256,
}

impl Default for Lane {
    fn default() -> Self {
        Self {
            parents: [0; MAX_PARENTS],
            input: [0; MAX_INPUT],
            value: [0; NODE],
            key: [0; NODE],
            len: 0,
            processed: 0,
            hasher: Sha256::new(),
        }
    }
}

#[derive(Default, Clone, Serialize)]
pub(crate) struct CpuStats {
    pub nodes: u64,
    pub batches: u64,
    pub sampled_nodes: u64,
    pub sampled_batches: u64,
    pub gather_ns: u128,
    pub sha_ns: u128,
    pub field_ns: u128,
    pub logical_sha_blocks: u64,
    pub reused_sha_blocks: u64,
    pub prefix_blocks_histogram: [u64; 17],
}

impl CpuStats {
    pub fn merge(mut self, other: &Self) -> Self {
        self.nodes += other.nodes;
        self.batches += other.batches;
        self.sampled_nodes += other.sampled_nodes;
        self.sampled_batches += other.sampled_batches;
        self.gather_ns += other.gather_ns;
        self.sha_ns += other.sha_ns;
        self.field_ns += other.field_ns;
        self.logical_sha_blocks += other.logical_sha_blocks;
        self.reused_sha_blocks += other.reused_sha_blocks;
        for (total, n) in self
            .prefix_blocks_histogram
            .iter_mut()
            .zip(other.prefix_blocks_histogram)
        {
            *total += n;
        }
        self
    }
}

pub(crate) struct CpuWorker {
    lanes: Vec<Lane>,
    pub stats: CpuStats,
}

impl CpuWorker {
    pub fn new(config: DecodeCpuConfig) -> Self {
        Self {
            lanes: (0..config.batch_nodes).map(|_| Lane::default()).collect(),
            stats: CpuStats::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &mut self,
        layer: &CpuLayer,
        state: &[u8],
        output: &mut [u8],
        base: usize,
        cancelled: Option<&AtomicBool>,
        mut parents: impl FnMut(usize, &mut [u32]) -> Result<()>,
        mut reference_field: impl FnMut([u8; NODE], [u8; NODE], &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            output.len().is_multiple_of(NODE)
                && base
                    .checked_add(output.len() / NODE)
                    .is_some_and(|end| end <= layer.nodes),
            "invalid CPU decode output range"
        );
        ensure!(
            layer.nodes.checked_mul(NODE) == Some(state.len()),
            "invalid CPU decode state"
        );
        for (batch, out) in output.chunks_mut(self.lanes.len() * NODE).enumerate() {
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "ZigZag decode cancelled"
            );
            let first = base + batch * self.lanes.len();
            let lanes = &mut self.lanes[..out.len() / NODE];
            let sampled = layer.config.profile_every != 0
                && self
                    .stats
                    .batches
                    .is_multiple_of(layer.config.profile_every as u64);
            let started = sampled.then(Instant::now);
            // Validate each parent before issuing its optional prefetch hint. Pointers
            // only address live immutable input; no speculative out-of-bounds arithmetic.
            for (i, lane) in lanes.iter_mut().enumerate() {
                let node = first + i;
                parents(node, &mut lane.parents[..layer.degree])?;
                for &parent in &lane.parents[..layer.degree] {
                    let p = parent as usize;
                    ensure!(
                        if node == layer.boundary {
                            p == node
                        } else {
                            p < layer.nodes && if layer.forward { p < node } else { p > node }
                        },
                        "ZigZag parent violates decode order at node {node}"
                    );
                    if layer.config.prefetch {
                        prefetch_node(&state[p * NODE..(p + 1) * NODE]);
                    }
                }
            }
            for (i, lane) in lanes.iter_mut().enumerate() {
                let node = first + i;
                lane.value
                    .copy_from_slice(&state[node * NODE..(node + 1) * NODE]);
                lane.input[..NODE].copy_from_slice(&layer.id);
                lane.len = NODE;
                let mut prefix = 0;
                if node != layer.boundary {
                    for (i, &parent) in lane.parents[..layer.degree].iter().enumerate() {
                        lane.input[(i + 1) * NODE..(i + 2) * NODE].copy_from_slice(
                            &state[parent as usize * NODE..(parent as usize + 1) * NODE],
                        );
                    }
                    lane.len += layer.degree * NODE;
                    if layer.config.prefix_reuse {
                        let repeated = lane.parents[..layer.degree]
                            .iter()
                            .take_while(|&&p| p as usize == layer.boundary)
                            .count();
                        prefix = (NODE + repeated * NODE) / 64;
                    }
                }
                lane.hasher = if prefix > 0 {
                    layer.prefixes[prefix].clone()
                } else {
                    Sha256::new()
                };
                lane.processed = prefix * 64;
                self.stats.prefix_blocks_histogram[prefix] += 1;
                self.stats.reused_sha_blocks += prefix as u64;
                self.stats.logical_sha_blocks += (lane.len + 9).div_ceil(64) as u64;
            }
            if let Some(started) = started {
                self.stats.gather_ns += started.elapsed().as_nanos();
            }
            let started = sampled.then(Instant::now);
            hash_lanes(lanes, layer.config.sha);
            if let Some(started) = started {
                self.stats.sha_ns += started.elapsed().as_nanos();
            }
            let started = sampled.then(Instant::now);
            for (lane, destination) in lanes.iter_mut().zip(out.chunks_exact_mut(NODE)) {
                lane.key[31] &= 0x3f; // Exactly bytes_into_fr_repr_safe: no endian or KDF change.
                if layer.config.field == DecodeField::Limbs && layer.canonical_domain {
                    destination.copy_from_slice(&subtract(&lane.value, &lane.key)?);
                } else {
                    if layer.canonical_domain {
                        ensure!(
                            canonical(&lane.value),
                            "noncanonical ZigZag encoded field element"
                        );
                    }
                    reference_field(lane.key, lane.value, destination)?;
                }
            }
            if let Some(started) = started {
                self.stats.field_ns += started.elapsed().as_nanos();
            }
            self.stats.batches += 1;
            self.stats.nodes += lanes.len() as u64;
            if sampled {
                self.stats.sampled_batches += 1;
                self.stats.sampled_nodes += lanes.len() as u64;
            }
        }
        ensure!(
            !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
            "ZigZag decode cancelled"
        );
        Ok(())
    }
}

fn hash_lanes(lanes: &mut [Lane], mode: DecodeSha) {
    if mode == DecodeSha::Serial {
        for lane in lanes.iter_mut() {
            lane.hasher.update(&lane.input[lane.processed..lane.len]);
            lane.key = lane.hasher.finalize_reset().into();
        }
        return;
    }
    // Reuse sha2's hardware detection and software fallback on both x86 and
    // ARM. No custom padding, ISA requirement, global Cargo feature or pool.
    loop {
        let mut remaining = false;
        for lane in lanes.iter_mut() {
            if lane.processed < lane.len {
                let end = (lane.processed + 64).min(lane.len);
                lane.hasher.update(&lane.input[lane.processed..end]);
                lane.processed = end;
                remaining = true;
            }
        }
        if !remaining {
            break;
        }
    }
    for lane in lanes {
        lane.key = lane.hasher.finalize_reset().into();
    }
}

fn prefetch_node(bytes: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: x86_64 guarantees SSE; pointer is inside a validated live input slice.
    unsafe {
        std::arch::x86_64::_mm_prefetch(bytes.as_ptr().cast(), std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM is an architectural hint, not a load or an optional SHA ISA instruction.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{address}]", address = in(reg) bytes.as_ptr(), options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = bytes;
}

#[cfg(test)]
#[path = "decode_cpu/tests.rs"]
mod tests;
