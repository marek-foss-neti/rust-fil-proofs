//! Best-effort, ZigZag-local placement of an encode pipeline on one shared L3.
//!
//! Reserve before spawning producers, but bind each producer inside its own thread and bind the
//! consumer only after all producers have spawned. Otherwise the producers inherit the consumer's
//! single-CPU mask. Keep the reservation until the producers have joined, and drop the consumer's
//! binding before invoking other parallel work such as TreeR construction.

use std::marker::PhantomData;
use std::rc::Rc;

/// An exclusive reservation of distinct physical cores for one ZigZag encode pipeline.
///
/// Worker zero is the consumer; subsequent workers are producers. The reservation contains only
/// OS CPU numbers, so it can be shared with scoped threads. Unsupported builds/topologies and
/// exhausted core groups produce an empty reservation and preserve ordinary OS scheduling.
/// Reservations coordinate only ZigZag pipelines in this process, not SDR or other processes.
#[derive(Default)]
pub(super) struct EncodeAffinity {
    logical_cpus: Vec<usize>,
    #[cfg(all(target_os = "linux", feature = "hwloc"))]
    reserved_cpus: Vec<usize>,
}

impl EncodeAffinity {
    pub(super) fn new(workers: usize, enabled: bool) -> Self {
        if !enabled || workers == 0 {
            log::info!(target: "zigzag_cpu", "encode affinity fallback: disabled or no workers; using OS scheduling");
            return Self::default();
        }

        #[cfg(all(target_os = "linux", feature = "hwloc"))]
        match platform::reserve(workers) {
            Ok(selection) => {
                return Self {
                    logical_cpus: selection.logical_cpus,
                    reserved_cpus: selection.reserved_cpus,
                };
            }
            Err(reason) => {
                log::info!(target: "zigzag_cpu", "encode affinity fallback: {reason}; using OS scheduling")
            }
        }

        #[cfg(not(all(target_os = "linux", feature = "hwloc")))]
        log::info!(target: "zigzag_cpu", "encode affinity fallback: requires Linux build with hwloc; using OS scheduling");
        Self::default()
    }

    pub(super) fn logical_cpus(&self) -> &[usize] {
        &self.logical_cpus
    }

    /// Bind only the calling thread. Failures are diagnostic and never fail encoding.
    pub(super) fn bind_current(&self, worker: usize) -> AffinityGuard<'_> {
        #[cfg(not(all(target_os = "linux", feature = "hwloc")))]
        let _ = worker;

        AffinityGuard {
            #[cfg(all(target_os = "linux", feature = "hwloc"))]
            binding: self
                .logical_cpus
                .get(worker)
                .and_then(|&cpu| platform::bind_current(cpu)),
            _reservation: PhantomData,
            _same_thread: PhantomData,
        }
    }
}

impl Drop for EncodeAffinity {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", feature = "hwloc"))]
        if !self.reserved_cpus.is_empty() {
            platform::release(&self.reserved_cpus);
        }
    }
}

/// Restores the original affinity on normal exit, errors and unwinding.
///
/// The lifetime keeps the reservation alive; the `Rc` marker makes the guard neither Send nor Sync,
/// including on builds where affinity is a no-op. A binding must be restored by its own thread.
pub(super) struct AffinityGuard<'a> {
    #[cfg(all(target_os = "linux", feature = "hwloc"))]
    binding: Option<platform::Binding>,
    _reservation: PhantomData<&'a EncodeAffinity>,
    _same_thread: PhantomData<Rc<()>>,
}

impl Drop for AffinityGuard<'_> {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", feature = "hwloc"))]
        if let Some(binding) = self.binding.take() {
            platform::restore(binding);
        }
    }
}

#[cfg(any(test, all(target_os = "linux", feature = "hwloc")))]
mod selection {
    use std::collections::{BTreeMap, HashSet};

    /// Full, unfiltered OS-PU sets identify cores and their actual L3 ancestor. In particular, core
    /// identity cannot be a selected SMT sibling: separate callers may allow different siblings.
    #[derive(Clone, Debug)]
    pub(super) struct Core {
        pub(super) cpus: Vec<usize>,
        pub(super) l3_cpus: Vec<usize>,
    }

    pub(super) struct Selection {
        pub(super) logical_cpus: Vec<usize>,
        pub(super) reserved_cpus: Vec<usize>,
    }

    #[derive(Default)]
    pub(super) struct Reservations {
        busy: HashSet<usize>,
    }

    impl Reservations {
        pub(super) fn reserve(
            &mut self,
            cores: &[Core],
            allowed: &HashSet<usize>,
            workers: usize,
        ) -> Option<Selection> {
            if workers == 0 {
                return None;
            }
            let mut groups: BTreeMap<&[usize], Vec<&Core>> = BTreeMap::new();
            for core in cores {
                if !core.cpus.is_empty() && !core.l3_cpus.is_empty() {
                    groups.entry(&core.l3_cpus).or_default().push(core);
                }
            }

            let mut best: Option<(usize, Selection)> = None;
            for group in groups.values() {
                let occupied = group
                    .iter()
                    .filter(|core| core.cpus.iter().any(|cpu| self.busy.contains(cpu)))
                    .count();
                let mut selection = Selection {
                    logical_cpus: Vec::new(),
                    reserved_cpus: Vec::new(),
                };
                for core in group {
                    if core
                        .cpus
                        .iter()
                        .any(|cpu| self.busy.contains(cpu) || selection.reserved_cpus.contains(cpu))
                    {
                        continue;
                    }
                    if let Some(&cpu) = core.cpus.iter().find(|&&cpu| allowed.contains(&cpu)) {
                        selection.logical_cpus.push(cpu);
                        selection.reserved_cpus.extend_from_slice(&core.cpus);
                        if selection.logical_cpus.len() == workers {
                            break;
                        }
                    }
                }
                if selection.logical_cpus.len() == workers
                    && best.as_ref().map_or(true, |(count, _)| occupied < *count)
                {
                    best = Some((occupied, selection));
                }
            }

            let (_, selection) = best?;
            self.busy.extend(selection.reserved_cpus.iter().copied());
            Some(selection)
        }

        pub(super) fn release(&mut self, cpus: &[usize]) {
            for cpu in cpus {
                self.busy.remove(cpu);
            }
        }
    }
}

#[cfg(all(target_os = "linux", feature = "hwloc"))]
mod platform {
    use std::collections::HashSet;
    use std::convert::TryFrom;
    use std::sync::Mutex;

    use hwloc::{Bitmap, CpuBindFlags, ObjectType, Topology};
    use lazy_static::lazy_static;

    use super::selection::{Core, Reservations, Selection};

    struct Pool {
        topology: Topology,
        cores: Vec<Core>,
        reservations: Reservations,
    }

    lazy_static! {
        // This pool deliberately does not use SDR's static groups or its configured producer count.
        static ref POOL: Mutex<Option<Pool>> = Mutex::new(None);
    }

    impl Pool {
        fn new() -> Option<Self> {
            let topology = Topology::new()?;
            let mut cores = Vec::new();
            for core in topology.objects_with_type(&ObjectType::Core).ok()? {
                let mut ancestor = core.parent();
                while let Some(parent) = ancestor {
                    if parent.object_type() == ObjectType::L3Cache {
                        if let (Some(cpus), Some(l3_cpus)) = (core.cpuset(), parent.cpuset()) {
                            cores.push(Core {
                                cpus: cpu_numbers(cpus),
                                l3_cpus: cpu_numbers(l3_cpus),
                            });
                        }
                        break;
                    }
                    ancestor = parent.parent();
                }
            }
            Some(Self {
                topology,
                cores,
                reservations: Reservations::default(),
            })
        }
    }

    fn cpu_numbers(set: Bitmap) -> Vec<usize> {
        set.into_iter().map(|cpu| cpu as usize).collect()
    }

    pub(super) fn reserve(workers: usize) -> Result<Selection, &'static str> {
        let mut pool = match POOL.lock() {
            Ok(pool) => pool,
            Err(_) => {
                return Err("topology lock poisoned");
            }
        };
        if pool.is_none() {
            *pool = Pool::new();
        }
        let pool = pool.as_mut().ok_or("physical core topology unavailable")?;
        if pool.cores.is_empty() {
            return Err("no physical cores with known L3 ancestors");
        }
        // This is the calling thread's current mask, not a core index or the machine-wide CPU set.
        let allowed: HashSet<_> = cpu_numbers(
            pool.topology
                .get_cpubind(CpuBindFlags::CPUBIND_THREAD)
                .ok_or("calling thread's allowed CPU mask unavailable")?,
        )
        .into_iter()
        .collect();
        pool.reservations
            .reserve(&pool.cores, &allowed, workers)
            .ok_or("no free allowed physical core group large enough within one L3")
    }

    pub(super) fn release(cpus: &[usize]) {
        // Cleanup must not panic during another panic. Recovering here only releases our own IDs.
        let mut pool = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pool) = pool.as_mut() {
            pool.reservations.release(cpus);
        }
    }

    pub(super) struct Binding {
        thread: libc::pthread_t,
        prior: Bitmap,
    }

    pub(super) fn bind_current(cpu: usize) -> Option<Binding> {
        let mut pool = match POOL.lock() {
            Ok(pool) => pool,
            Err(_) => {
                log::warn!(target: "zigzag_cpu", "zigzag encode affinity: topology lock poisoned; skipping binding");
                return None;
            }
        };
        let topology = &mut pool.as_mut()?.topology;
        // Safety: pthread_self returns the current live thread; the !Send guard stays on that thread.
        let thread = unsafe { libc::pthread_self() };
        let prior = match topology.get_cpubind_for_thread(thread, CpuBindFlags::CPUBIND_THREAD) {
            Some(prior) => prior,
            None => {
                log::warn!(target: "zigzag_cpu", "zigzag encode affinity: cannot save current mask; skipping binding");
                return None;
            }
        };
        let cpu = u32::try_from(cpu).ok()?;
        if !prior.is_set(cpu) {
            log::warn!(target: "zigzag_cpu", "zigzag encode affinity: reserved CPU {cpu} is no longer allowed");
            return None;
        }
        if let Err(error) =
            topology.set_cpubind_for_thread(thread, Bitmap::from(cpu), CpuBindFlags::CPUBIND_THREAD)
        {
            log::warn!(target: "zigzag_cpu", "zigzag encode affinity: failed to bind CPU {cpu}: {error:?}");
            // A failed backend call need not be assumed atomic. Restore immediately and retain the
            // guard so it attempts restoration again on scope exit even if this attempt failed.
            if let Err(error) =
                topology.set_cpubind_for_thread(thread, prior.clone(), CpuBindFlags::CPUBIND_THREAD)
            {
                log::warn!(target: "zigzag_cpu", "zigzag encode affinity: failed to undo binding: {error:?}");
            }
        } else {
            log::info!(target: "zigzag_cpu", "encode affinity bound thread to CPU {cpu}");
        }
        Some(Binding { thread, prior })
    }

    pub(super) fn restore(binding: Binding) {
        let mut pool = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pool) = pool.as_mut() {
            if let Err(error) = pool.topology.set_cpubind_for_thread(
                binding.thread,
                binding.prior,
                CpuBindFlags::CPUBIND_THREAD,
            ) {
                log::warn!(target: "zigzag_cpu", "zigzag encode affinity: failed to restore thread mask: {error:?}");
            } else {
                log::info!(target: "zigzag_cpu", "encode affinity restored prior thread mask before subsequent work");
            }
        }
    }

    #[cfg(test)]
    pub(super) fn current_binding() -> Vec<usize> {
        let pool = POOL.lock().expect("test topology lock");
        cpu_numbers(
            pool.as_ref()
                .expect("test initialized topology")
                .topology
                .get_cpubind(CpuBindFlags::CPUBIND_THREAD)
                .expect("test current binding"),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::selection::{Core, Reservations};
    use super::EncodeAffinity;

    fn core(cpus: &[usize], l3_cpus: &[usize]) -> Core {
        Core {
            cpus: cpus.to_vec(),
            l3_cpus: l3_cpus.to_vec(),
        }
    }

    fn allowed(cpus: &[usize]) -> HashSet<usize> {
        cpus.iter().copied().collect()
    }

    #[test]
    fn selects_allowed_os_cpus_from_distinct_cores_in_one_l3() {
        let l3 = [2, 7, 11, 19, 40, 90];
        let cores = [
            core(&[2, 40], &l3),
            core(&[7, 90], &l3),
            core(&[11, 19], &l3),
        ];
        let selected = Reservations::default()
            .reserve(&cores, &allowed(&[40, 90, 19]), 3)
            .expect("three allowed physical cores");
        assert_eq!(selected.logical_cpus, [40, 90, 19]);
        assert_eq!(selected.reserved_cpus, [2, 40, 7, 90, 11, 19]);
    }

    #[test]
    fn never_combines_l3_groups_or_counts_smt_as_extra_cores() {
        let cores = [
            core(&[0, 8], &[0, 1, 8, 9]),
            core(&[1, 9], &[0, 1, 8, 9]),
            core(&[2, 10], &[2, 3, 10, 11]),
            core(&[3, 11], &[2, 3, 10, 11]),
        ];
        let mut pool = Reservations::default();
        let all = allowed(&[0, 1, 2, 3, 8, 9, 10, 11]);
        assert!(pool.reserve(&cores, &all, 3).is_none());
        assert!(pool.reserve(&cores, &all, 0).is_none());
        assert!(pool.reserve(&cores, &allowed(&[]), 1).is_none());
        assert!(pool.reserve(&cores, &all, 2).is_some());
    }

    #[test]
    fn reservations_exclude_other_smt_siblings_and_release_for_reuse() {
        let cores = [
            core(&[2, 22], &[2, 5, 22, 55]),
            core(&[5, 55], &[2, 5, 22, 55]),
        ];
        let mut pool = Reservations::default();
        let first = pool.reserve(&cores, &allowed(&[2, 5]), 2).unwrap();
        assert!(pool.reserve(&cores, &allowed(&[22, 55]), 2).is_none());
        pool.release(&first.reserved_cpus);
        let second = pool.reserve(&cores, &allowed(&[22, 55]), 2).unwrap();
        assert_eq!(second.logical_cpus, [22, 55]);
    }

    #[test]
    fn unknown_l3_topology_and_disallowed_cores_fall_back() {
        let cores = [core(&[3, 23], &[]), core(&[5, 25], &[5, 25])];
        let mut pool = Reservations::default();
        assert!(pool.reserve(&cores, &allowed(&[3, 23]), 1).is_none());
        assert!(pool.reserve(&[], &allowed(&[3, 23]), 1).is_none());
        assert_eq!(
            pool.reserve(&cores, &allowed(&[25]), 1)
                .unwrap()
                .logical_cpus,
            [25]
        );
    }

    #[test]
    fn dynamic_worker_counts_spread_across_free_l3_groups() {
        let cores: Vec<_> = (0..8)
            .map(|cpu| {
                if cpu < 4 {
                    core(&[cpu], &[0, 1, 2, 3])
                } else {
                    core(&[cpu], &[4, 5, 6, 7])
                }
            })
            .collect();
        let mut pool = Reservations::default();
        let all = allowed(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let first = pool.reserve(&cores, &all, 3).unwrap();
        assert_eq!(first.logical_cpus, [0, 1, 2]);
        let second = pool.reserve(&cores, &all, 4).unwrap();
        assert_eq!(second.logical_cpus, [4, 5, 6, 7]);
        assert!(pool.reserve(&cores, &all, 2).is_none());
        pool.release(&first.reserved_cpus);
        assert_eq!(
            pool.reserve(&cores, &all, 4).unwrap().logical_cpus,
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn disabled_affinity_is_a_noop_and_reservation_can_be_shared() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EncodeAffinity>();
        let group = EncodeAffinity::new(3, false);
        assert!(group.logical_cpus().is_empty());
        let _guard = group.bind_current(usize::MAX);
    }

    #[cfg(all(target_os = "linux", feature = "hwloc"))]
    #[test]
    #[ignore = "opt-in: needs at least two allowed physical cores sharing L3 and thread affinity"]
    fn live_binding_restores_masks_after_worker_exit_and_consumer_panic() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        use super::platform;

        let group = EncodeAffinity::new(2, true);
        assert_eq!(
            group.logical_cpus().len(),
            2,
            "test needs a suitable L3 group"
        );
        let prior = platform::current_binding();
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                let before = platform::current_binding();
                {
                    let _guard = group.bind_current(1);
                    assert_eq!(platform::current_binding(), [group.logical_cpus()[1]]);
                }
                assert_eq!(platform::current_binding(), before);
            });
            let mut observed = Vec::new();
            let result = catch_unwind(AssertUnwindSafe(|| {
                let _guard = group.bind_current(0);
                observed = platform::current_binding();
                panic!("exercise consumer affinity restoration during unwinding");
            }));
            assert!(result.is_err());
            assert_eq!(observed, [group.logical_cpus()[0]]);
            assert_eq!(platform::current_binding(), prior);
            worker.join().unwrap();
        });
        // A subsequent Rayon pool must inherit the restored broad mask.
        let rayon = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let masks = rayon.install(|| {
            use rayon::prelude::*;
            (0..8)
                .into_par_iter()
                .map(|_| platform::current_binding())
                .collect::<Vec<_>>()
        });
        assert!(masks.iter().all(|mask| mask == &prior));
    }
}
