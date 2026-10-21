//! ZigZag-only resource admission for file-backed parameter generation.
//! Worker counts change scheduling, never the circuit or parameter byte order.

use std::{fmt, mem::size_of};

use anyhow::{ensure, Context, Result};
use pairing::Engine;
use serde::Serialize;

use super::groth16_setup::SetupLimits;

mod resources;
pub(super) use resources::SetupResources;

const MAX_WORKERS: usize = 128;
const MIB: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SetupWorkers {
    Auto,
    Fixed(usize),
}

impl SetupWorkers {
    pub(crate) fn parse(raw: &str) -> Result<Self> {
        if raw == "auto" {
            return Ok(Self::Auto);
        }
        ensure!(
            !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()),
            "ZigZag setup workers must be auto or 1..=128"
        );
        let count: usize = raw.parse().context("invalid ZigZag setup workers")?;
        ensure!(
            (1..=MAX_WORKERS).contains(&count),
            "ZigZag setup workers must be auto or 1..=128"
        );
        Ok(Self::Fixed(count))
    }

    pub(crate) fn fixed(self) -> Option<usize> {
        match self {
            Self::Auto => None,
            Self::Fixed(count) => Some(count),
        }
    }
}

impl fmt::Display for SetupWorkers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Fixed(count) => write!(f, "{count}"),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SetupMemory {
    pub nonworker_peak_bytes: u64,
    pub h_fixed_bytes: u64,
    pub query_fixed_bytes: u64,
    pub h_worker_bytes: u64,
    pub query_worker_bytes: u64,
}

impl SetupMemory {
    pub fn new<E: Engine>(m: usize, n: usize, inputs: usize, batch: usize) -> Result<Self> {
        // Checked u128 intermediates also cover unreasonable explicit batch sizes.
        let f = size_of::<E::Fr>() as u128;
        let g1 = size_of::<E::G1>() as u128;
        let g2 = size_of::<E::G2>() as u128;
        let a1 = size_of::<E::G1Affine>() as u128;
        let a2 = size_of::<E::G2Affine>() as u128;
        let (m, n, inputs, batch) = (m as u128, n as u128, inputs as u128, batch as u128);
        let bytes = |n: u128| u64::try_from(n).context("ZigZag setup memory estimate overflow");
        Ok(Self {
            // IFFT workspace and the simultaneous Lagrange + A/B/C arrays.
            nonworker_peak_bytes: bytes((2 * m * f).max((m + 3 * n) * f))?,
            h_fixed_bytes: bytes(m * f)?,
            // IC vectors can have spare capacity and coexist with the final IC.
            query_fixed_bytes: bytes(3 * n * f + 3 * inputs * a1)?,
            // Allow normalization scratch as well as the explicit point vectors.
            // 8 MiB per worker covers segment I/O, thread stack and WNAF scratch.
            // Global allocator/page-cache/library headroom is reserved separately.
            h_worker_bytes: bytes(batch * (g1 + a1 + 2 * g1) + 8 * MIB as u128)?,
            query_worker_bytes: bytes(
                batch * (3 * (g1 + a1) + g2 + a2 + 2 * g2) + 8 * MIB as u128,
            )?,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct WorkerPlan {
    pub phase: &'static str,
    pub requested_workers: String,
    pub workers: usize,
    pub cpu_limit: usize,
    pub work_limit: usize,
    pub memory_limit: Option<usize>,
    pub configured_budget_bytes: u64,
    pub effective_budget_bytes: u64,
    pub reserve_bytes: u64,
    pub fixed_bytes: u64,
    pub worker_bytes: u64,
    pub estimated_peak_bytes: u64,
    pub batch_points: usize,
    pub limiting_factors: Vec<&'static str>,
    pub resources: SetupResources,
}

impl WorkerPlan {
    pub fn auto(
        phase: &'static str,
        limits: SetupLimits,
        work_items: usize,
        fixed_bytes: u64,
        worker_bytes: u64,
        initial_anon: u64,
        resources: SetupResources,
    ) -> Result<Self> {
        ensure!(
            limits.batch_points > 0 && worker_bytes > 0,
            "invalid setup allocation bounds"
        );
        let budget = resources.budget(limits.budget_bytes);
        let reserve = (budget / 10).max(512 * MIB);
        // Later phases sample RSS again. Count retained earlier allocations, but
        // do not add arrays twice when they are already resident in that sample.
        let fixed = initial_anon
            .checked_add(fixed_bytes)
            .context("setup fixed allocation overflow")?
            .max(resources.process_anon_bytes);
        let available = budget.saturating_sub(reserve).saturating_sub(fixed);
        let memory_limit = usize::try_from(available / worker_bytes).unwrap_or(usize::MAX);
        ensure!(memory_limit > 0,
            "ZigZag auto setup {phase} cannot fit one worker: effective budget {budget}, reserve {reserve}, fixed {fixed}, worker {worker_bytes}; lower SETUP_BATCH_POINTS or provide more memory");
        let cpu_limit = resources.cpu_limit.clamp(1, MAX_WORKERS);
        let work_limit = work_items.div_ceil(limits.batch_points).max(1);
        let workers = cpu_limit.min(work_limit).min(memory_limit);
        let mut limiting_factors = Vec::new();
        if workers == cpu_limit {
            limiting_factors.push("cpu_or_128_cap");
        }
        if workers == work_limit {
            limiting_factors.push("work");
        }
        if workers == memory_limit {
            limiting_factors.push("memory");
        }
        Ok(Self {
            phase,
            requested_workers: "auto".into(),
            workers,
            cpu_limit,
            work_limit,
            memory_limit: Some(memory_limit),
            configured_budget_bytes: limits.budget_bytes,
            effective_budget_bytes: budget,
            reserve_bytes: reserve,
            fixed_bytes: fixed,
            worker_bytes,
            estimated_peak_bytes: fixed + workers as u64 * worker_bytes + reserve,
            batch_points: limits.batch_points,
            limiting_factors,
            resources,
        })
    }

    pub fn log(&self) {
        log::info!(target: "zigzag_setup", "zigzag_setup_workers {}", serde_json::to_string(self).expect("setup plan JSON"));
    }
}

pub(super) fn check_nonworker_memory(
    memory: SetupMemory,
    limits: SetupLimits,
    initial_anon: u64,
    resources: SetupResources,
    resident_domain_bytes: u64,
) -> Result<()> {
    let budget = resources.budget(limits.budget_bytes);
    let reserve = (budget / 10).max(512 * MIB);
    // Recheck after H as well: worker arenas may retain freed buffers, and
    // other processes may have consumed memory since the initial admission.
    let pending = memory
        .nonworker_peak_bytes
        .saturating_sub(resident_domain_bytes);
    let peak = initial_anon
        .checked_add(memory.nonworker_peak_bytes)
        .zip(resources.process_anon_bytes.checked_add(pending))
        .map(|(initial, current)| initial.max(current));
    ensure!(
        peak.and_then(|n| n.checked_add(reserve)).is_some_and(|n| n <= budget),
        "ZigZag auto setup domain/QAP exceeds effective budget {budget}; fewer workers cannot fix this allocation"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::Bls12;
    const GIB: u64 = 1 << 30;

    fn resources(cpus: usize) -> SetupResources {
        SetupResources {
            cpu_limit: cpus,
            process_anon_bytes: GIB,
            host_available_bytes: Some(100 * GIB),
            cgroup_limit_bytes: Some(50 * GIB),
            cgroup_available_bytes: Some(49 * GIB),
        }
    }

    fn limits() -> SetupLimits {
        SetupLimits {
            batch_points: 65_536,
            workers: SetupWorkers::Auto,
            budget_bytes: 40 * GIB,
        }
    }

    #[test]
    fn worker_mode_accepts_auto_and_explicit_counts() {
        assert_eq!(SetupWorkers::parse("auto").unwrap(), SetupWorkers::Auto);
        for n in [1, 16, 96, 128] {
            assert_eq!(
                SetupWorkers::parse(&n.to_string()).unwrap(),
                SetupWorkers::Fixed(n)
            );
        }
        for raw in [
            "",
            "0",
            "129",
            "-1",
            "+2",
            " 2",
            "AUTO",
            "18446744073709551617",
        ] {
            assert!(SetupWorkers::parse(raw).is_err(), "{raw}");
        }
    }

    #[test]
    fn cpu_work_and_memory_each_limit_auto_workers() {
        for (cpus, expected) in [(1, 1), (24, 24), (96, 96), (128, 128), (256, 128)] {
            let plan =
                WorkerPlan::auto("h", limits(), 1 << 30, GIB, 16 * MIB, GIB, resources(cpus))
                    .unwrap();
            assert_eq!(plan.workers, expected);
            assert!(plan.estimated_peak_bytes <= plan.effective_budget_bytes);
        }
        let tiny = WorkerPlan::auto("h", limits(), 2, GIB, 16 * MIB, GIB, resources(96)).unwrap();
        assert_eq!(tiny.workers, 1);
        assert!(tiny.limiting_factors.contains(&"work"));
        let tight = SetupResources {
            cgroup_limit_bytes: Some(8 * GIB),
            cgroup_available_bytes: Some(7 * GIB),
            ..resources(96)
        };
        let plan =
            WorkerPlan::auto("queries", limits(), 1 << 30, 4 * GIB, GIB, GIB, tight).unwrap();
        assert_eq!(plan.workers, 2);
        assert!(plan.limiting_factors.contains(&"memory"));
    }

    #[test]
    fn phase_memory_can_select_different_counts_and_pressure_reduces_queries() {
        let memory = SetupMemory::new::<Bls12>(1 << 26, 1 << 26, 3, 65_536).unwrap();
        let tight = SetupResources {
            cgroup_limit_bytes: Some(12 * GIB),
            cgroup_available_bytes: Some(11 * GIB),
            ..resources(96)
        };
        let h = WorkerPlan::auto(
            "h",
            limits(),
            1 << 26,
            memory.h_fixed_bytes,
            memory.h_worker_bytes,
            GIB,
            tight,
        )
        .unwrap();
        let q = WorkerPlan::auto(
            "queries",
            limits(),
            1 << 26,
            memory.query_fixed_bytes,
            memory.query_worker_bytes,
            GIB,
            tight,
        )
        .unwrap();
        assert!(h.workers > q.workers);
        let pressure = SetupResources {
            process_anon_bytes: 7 * GIB,
            cgroup_available_bytes: Some(2 * GIB),
            ..tight
        };
        let q2 = WorkerPlan::auto(
            "queries",
            limits(),
            1 << 26,
            memory.query_fixed_bytes,
            memory.query_worker_bytes,
            GIB,
            pressure,
        )
        .unwrap();
        assert!(q2.workers < q.workers);
    }

    #[test]
    fn host_headroom_and_existing_allocations_are_counted_once() {
        let busy = SetupResources {
            process_anon_bytes: 5 * GIB,
            host_available_bytes: Some(3 * GIB),
            ..resources(96)
        };
        assert_eq!(busy.budget(40 * GIB), 8 * GIB);
        let plan =
            WorkerPlan::auto("queries", limits(), 1 << 30, 4 * GIB, 64 * MIB, GIB, busy).unwrap();
        assert_eq!(plan.fixed_bytes, 5 * GIB);
        assert!(plan.estimated_peak_bytes <= 8 * GIB);
    }

    #[test]
    fn impossible_worker_and_nonparallel_allocations_fail_before_work() {
        let busy = SetupResources {
            host_available_bytes: Some(128 * MIB),
            ..resources(96)
        };
        assert!(WorkerPlan::auto("queries", limits(), 100, GIB, GIB, GIB, busy).is_err());
        let memory = SetupMemory::new::<Bls12>(1 << 30, 1 << 30, 3, 65_536).unwrap();
        assert!(check_nonworker_memory(memory, limits(), GIB, resources(96), 0).is_err());
        assert!(SetupMemory::new::<Bls12>(usize::MAX, usize::MAX, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn nonworker_recheck_includes_retained_h_buffers_without_counting_domain_twice() {
        let memory = SetupMemory::new::<Bls12>(1 << 26, 1 << 26, 3, 65_536).unwrap();
        let small = SetupLimits {
            budget_bytes: 12 * GIB,
            ..limits()
        };
        assert!(check_nonworker_memory(memory, small, GIB, resources(96), 0).is_ok());
        let after_h = SetupResources {
            process_anon_bytes: GIB + memory.h_fixed_bytes,
            ..resources(96)
        };
        assert!(check_nonworker_memory(memory, small, GIB, after_h, memory.h_fixed_bytes).is_ok());
        let retained = SetupResources {
            process_anon_bytes: 6 * GIB,
            ..after_h
        };
        assert!(
            check_nonworker_memory(memory, small, GIB, retained, memory.h_fixed_bytes).is_err()
        );
    }
}
