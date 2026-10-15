//! A bounded window of positioned writes with one background flush worker.
//! Credits include reservations, in-flight writes and completed unflushed writes.
//! A flush retires only writes completed BEFORE that flush started. Concurrent
//! writes remain charged even if the OS happened to flush them too.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, ensure, Context, Result};
use serde::Serialize;

use super::measurements::WaitStats;
use super::merkle_builder::{write_all_at, PositionedIo};

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct DirtyStats {
    pub limit_bytes: usize,
    pub flush_threshold_bytes: usize,
    pub peak_outstanding_bytes: usize,
    pub budget_wait: WaitStats,
    pub flush_count: u64,
    pub flushed_bytes: u64,
    pub flush_ns: u64,
}

#[derive(Default)]
struct State {
    outstanding: usize,
    completed: usize,
    waiters: usize,
    finished: bool,
    stats: DirtyStats,
}

pub(crate) struct DirtyBudget<'a> {
    limit: usize,
    cancelled: &'a AtomicBool,
    state: Mutex<State>,
    changed: Condvar,
}

/// The job must join its writers before returning. Both job and flush panics
/// cancel the work, wake waiters and join the I/O worker before file cleanup.
pub(crate) fn with_dirty_budget<T>(
    io: &impl PositionedIo,
    limit: usize,
    cancelled: &AtomicBool,
    job: impl FnOnce(&DirtyBudget<'_>) -> Result<T>,
) -> Result<(T, DirtyStats)> {
    ensure!(limit > 0, "empty TreeR dirty budget");
    let budget = DirtyBudget {
        limit,
        cancelled,
        state: Mutex::new(State {
            stats: DirtyStats {
                limit_bytes: limit,
                flush_threshold_bytes: (limit / 2).max(1),
                ..Default::default()
            },
            ..Default::default()
        }),
        changed: Condvar::new(),
    };
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("zigzag-tree-writeback".into())
            .spawn_scoped(scope, || {
                let result = catch_unwind(AssertUnwindSafe(|| budget.flush_loop(io)))
                    .map_err(|_| anyhow!("TreeR writeback worker panicked"))
                    .and_then(|result| result);
                if result.is_err() {
                    cancelled.store(true, Ordering::Release);
                    budget.changed.notify_all();
                }
                result
            })?;
        let result = catch_unwind(AssertUnwindSafe(|| job(&budget)))
            .map_err(|_| anyhow!("TreeR write job panicked"))
            .and_then(|result| result);
        if result.is_err() {
            cancelled.store(true, Ordering::Release);
        }
        budget
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .finished = true;
        budget.changed.notify_all();
        // Prefer the actual flush error over a writer's consequent cancellation.
        worker
            .join()
            .map_err(|_| anyhow!("TreeR writeback join panicked"))??;
        let value = result?;
        ensure!(
            !cancelled.load(Ordering::Acquire),
            "TreeR writeback cancelled"
        );
        let state = budget
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ensure!(state.outstanding == 0, "incomplete TreeR writeback");
        Ok((value, state.stats.clone()))
    })
}

impl DirtyBudget<'_> {
    pub(crate) fn write(&self, io: &impl PositionedIo, data: &[u8], offset: u64) -> Result<()> {
        ensure!(data.len() <= self.limit, "TreeR write exceeds dirty budget");
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut waiting = None;
        while state.outstanding > self.limit - data.len() {
            if self.cancelled.load(Ordering::Acquire) {
                if let Some(started) = waiting {
                    state.waiters -= 1;
                    state.stats.budget_wait.record(started);
                }
                anyhow::bail!("TreeR dirty budget cancelled");
            }
            if waiting.is_none() {
                waiting = Some(Instant::now());
                state.waiters += 1;
                self.changed.notify_all();
            }
            // Other pipeline workers share cancellation but not this condvar.
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(50))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        if let Some(started) = waiting {
            state.waiters -= 1;
            state.stats.budget_wait.record(started);
        }
        ensure!(
            !self.cancelled.load(Ordering::Acquire),
            "TreeR dirty budget cancelled"
        );
        state.outstanding += data.len();
        state.stats.peak_outstanding_bytes =
            state.stats.peak_outstanding_bytes.max(state.outstanding);
        drop(state);

        // No bookkeeping lock covers I/O. Disjoint positioned writes may run
        // concurrently with each other and with the background flush.
        if let Err(error) = write_all_at(io, data, offset) {
            self.cancelled.store(true, Ordering::Release);
            self.changed.notify_all();
            return Err(error.into());
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.completed += data.len();
        self.changed.notify_all();
        Ok(())
    }

    fn flush_loop(&self, io: &impl PositionedIo) -> Result<()> {
        loop {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if self.cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                if state.completed > 0
                    && (state.finished
                        || state.waiters > 0
                        || state.completed >= state.stats.flush_threshold_bytes)
                {
                    break;
                }
                if state.finished {
                    return Ok(());
                }
                state = self
                    .changed
                    .wait_timeout(state, Duration::from_millis(50))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            let flushing = std::mem::take(&mut state.completed);
            drop(state);
            let started = Instant::now();
            io.sync_data().context("flush TreeR dirty window")?;
            let elapsed = started.elapsed().as_nanos() as u64;
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.outstanding -= flushing;
            state.stats.flush_count += 1;
            state.stats.flushed_bytes += flushing as u64;
            state.stats.flush_ns += elapsed;
            self.changed.notify_all();
        }
    }
}

#[cfg(test)]
mod tests;
