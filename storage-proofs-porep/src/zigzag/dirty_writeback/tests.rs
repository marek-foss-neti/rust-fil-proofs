use super::*;
use std::io;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;

struct CountingIo {
    written: AtomicUsize,
    flushed: AtomicUsize,
    peak: AtomicUsize,
    fail_flush: bool,
    panic_flush: bool,
    fail_write: bool,
}

impl CountingIo {
    fn new() -> Self {
        Self {
            written: AtomicUsize::new(0),
            flushed: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            fail_flush: false,
            panic_flush: false,
            fail_write: false,
        }
    }
}

impl PositionedIo for CountingIo {
    fn read_at(&self, _: &mut [u8], _: u64) -> io::Result<usize> {
        unreachable!()
    }
    fn start_writeback(&self, _: u64, _: u64) -> io::Result<()> {
        unreachable!()
    }
    fn write_at(&self, bytes: &[u8], _: u64) -> io::Result<usize> {
        if self.fail_write {
            return Err(io::Error::from_raw_os_error(libc::ENOSPC));
        }
        let written = self.written.fetch_add(bytes.len(), Ordering::SeqCst) + bytes.len();
        self.peak.fetch_max(
            written.saturating_sub(self.flushed.load(Ordering::SeqCst)),
            Ordering::SeqCst,
        );
        Ok(bytes.len())
    }
    fn sync_data(&self) -> io::Result<()> {
        assert!(!self.panic_flush, "injected flush panic");
        if self.fail_flush {
            return Err(io::Error::from_raw_os_error(libc::EIO));
        }
        let completed = self.written.load(Ordering::SeqCst);
        std::thread::yield_now();
        self.flushed.store(completed, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn concurrent_writes_stay_bounded_and_final_partial_window_is_flushed() {
    let io = CountingIo::new();
    let cancelled = AtomicBool::new(false);
    let (_, stats) = with_dirty_budget(&io, 96, &cancelled, |budget| {
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for index in 0..31 {
                        budget.write(&io, &[0; 32], index * 32).unwrap();
                    }
                });
            }
        });
        Ok(())
    })
    .unwrap();
    assert!(io.peak.load(Ordering::SeqCst) <= 96);
    assert!(stats.peak_outstanding_bytes <= 96);
    assert_eq!(stats.flushed_bytes, 4 * 31 * 32);
    assert_eq!(io.flushed.load(Ordering::SeqCst), 4 * 31 * 32);
    let (_, tail) =
        with_dirty_budget(&io, 96, &cancelled, |budget| budget.write(&io, &[0; 1], 0)).unwrap();
    assert_eq!(tail.flush_count, 1);
    assert_eq!(tail.flushed_bytes, 1);
}

struct GatedFlush {
    inner: CountingIo,
    entered: mpsc::SyncSender<()>,
    release: Mutex<mpsc::Receiver<()>>,
    calls: AtomicUsize,
}
impl PositionedIo for GatedFlush {
    fn read_at(&self, _: &mut [u8], _: u64) -> io::Result<usize> {
        unreachable!()
    }
    fn start_writeback(&self, _: u64, _: u64) -> io::Result<()> {
        unreachable!()
    }
    fn write_at(&self, bytes: &[u8], offset: u64) -> io::Result<usize> {
        self.inner.write_at(bytes, offset)
    }
    fn sync_data(&self) -> io::Result<()> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            self.entered.send(()).unwrap();
            self.release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
        }
        self.inner.sync_data()
    }
}

#[test]
fn writes_continue_during_flush_but_new_writes_need_their_own_flush() {
    let (entered, started) = mpsc::sync_channel(1);
    let (release, gate) = mpsc::sync_channel(1);
    let io = GatedFlush {
        inner: CountingIo::new(),
        entered,
        release: Mutex::new(gate),
        calls: AtomicUsize::new(0),
    };
    let cancelled = AtomicBool::new(false);
    let (_, stats) = with_dirty_budget(&io, 64, &cancelled, |budget| {
        budget.write(&io, &[0; 32], 0)?;
        started.recv_timeout(Duration::from_secs(2))?;
        std::thread::scope(|scope| {
            let (sent, received) = mpsc::sync_channel(1);
            let io_ref = &io;
            let writer =
                scope.spawn(move || sent.send(budget.write(io_ref, &[0; 32], 32)).unwrap());
            let result = received.recv_timeout(Duration::from_secs(2));
            release.send(()).unwrap(); // Release even if the assertion fails.
            writer.join().unwrap();
            result
                .expect("second half of window must progress during flush")
                .unwrap();
        });
        Ok(())
    })
    .unwrap();
    assert_eq!(stats.peak_outstanding_bytes, 64);
    assert_eq!(stats.flush_count, 2);
    assert_eq!(stats.flushed_bytes, 64);
}

#[test]
fn flush_failure_panic_write_failure_and_job_panic_cancel_and_join() {
    for failure in 0..4 {
        let mut io = CountingIo::new();
        io.fail_flush = failure == 0;
        io.panic_flush = failure == 1;
        io.fail_write = failure == 2;
        let cancelled = AtomicBool::new(false);
        let error = with_dirty_budget(&io, 32, &cancelled, |budget| {
            budget.write(&io, &[0; 32], 0)?;
            assert!(failure != 3, "injected writer panic");
            budget.write(&io, &[0; 32], 32)
        })
        .unwrap_err();
        assert!(cancelled.load(Ordering::Acquire));
        if failure == 0 || failure == 2 {
            assert_eq!(
                error.downcast_ref::<io::Error>().unwrap().raw_os_error(),
                Some(if failure == 0 {
                    libc::EIO
                } else {
                    libc::ENOSPC
                })
            );
        } else {
            assert!(error.to_string().contains("panicked"));
        }
    }
}

#[test]
fn external_cancellation_wakes_budget_waiters_without_releasing_flush_credits() {
    let (entered, started) = mpsc::sync_channel(1);
    let (release, gate) = mpsc::sync_channel(1);
    let io = GatedFlush {
        inner: CountingIo::new(),
        entered,
        release: Mutex::new(gate),
        calls: AtomicUsize::new(0),
    };
    let cancelled = AtomicBool::new(false);
    assert!(with_dirty_budget(&io, 32, &cancelled, |budget| {
        budget.write(&io, &[0; 32], 0)?;
        started.recv_timeout(Duration::from_secs(2))?;
        std::thread::scope(|scope| {
            let (sent, received) = mpsc::sync_channel(1);
            let io_ref = &io;
            let writer =
                scope.spawn(move || sent.send(budget.write(io_ref, &[0; 32], 32)).unwrap());
            let deadline = Instant::now() + Duration::from_secs(2);
            while budget.state.lock().unwrap().waiters == 0 && Instant::now() < deadline {
                std::thread::yield_now();
            }
            let blocked = budget.state.lock().unwrap().waiters;
            cancelled.store(true, Ordering::Release); // No condvar notification.
            let result = received.recv_timeout(Duration::from_secs(2));
            release.send(()).unwrap();
            writer.join().unwrap();
            assert_eq!(blocked, 1);
            assert!(result.unwrap().is_err());
            let state = budget.state.lock().unwrap();
            assert_eq!(state.stats.budget_wait.events, 1);
            assert!(state.stats.budget_wait.elapsed_ns > 0);
        });
        Ok(())
    })
    .is_err());
    assert_eq!(io.inner.written.load(Ordering::SeqCst), 32);
}

#[test]
fn invalid_limits_and_oversized_writes_fail_without_io() {
    let io = CountingIo::new();
    let cancelled = AtomicBool::new(false);
    assert!(with_dirty_budget(&io, 0, &cancelled, |_| Ok(())).is_err());
    assert!(
        with_dirty_budget(&io, 32, &cancelled, |budget| budget.write(&io, &[0; 33], 0)).is_err()
    );
    assert_eq!(io.written.load(Ordering::SeqCst), 0);
}

#[test]
fn a_flush_never_retires_a_reserved_write_still_in_flight() {
    struct GatedWrite {
        inner: CountingIo,
        entered: mpsc::SyncSender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl PositionedIo for GatedWrite {
        fn read_at(&self, _: &mut [u8], _: u64) -> io::Result<usize> {
            unreachable!()
        }
        fn start_writeback(&self, _: u64, _: u64) -> io::Result<()> {
            unreachable!()
        }
        fn sync_data(&self) -> io::Result<()> {
            self.inner.sync_data()
        }
        fn write_at(&self, bytes: &[u8], offset: u64) -> io::Result<usize> {
            if offset == 0 {
                self.entered.send(()).unwrap();
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();
            }
            self.inner.write_at(bytes, offset)
        }
    }
    let (entered, started) = mpsc::sync_channel(1);
    let (release, gate) = mpsc::sync_channel(1);
    let io = GatedWrite {
        inner: CountingIo::new(),
        entered,
        release: Mutex::new(gate),
    };
    let cancelled = AtomicBool::new(false);
    let (_, stats) = with_dirty_budget(&io, 64, &cancelled, |budget| {
        std::thread::scope(|scope| {
            let io_ref = &io;
            let writer = scope.spawn(move || budget.write(io_ref, &[0; 32], 0));
            started.recv_timeout(Duration::from_secs(2)).unwrap();
            budget.write(&io, &[0; 32], 32).unwrap();
            let deadline = Instant::now() + Duration::from_secs(2);
            while budget.state.lock().unwrap().stats.flush_count == 0 && Instant::now() < deadline {
                std::thread::yield_now();
            }
            let (outstanding, flushes) = {
                let state = budget.state.lock().unwrap();
                (state.outstanding, state.stats.flush_count)
            };
            release.send(()).unwrap();
            writer.join().unwrap().unwrap();
            assert_eq!(flushes, 1);
            assert_eq!(outstanding, 32, "the slow write must retain its credits");
        });
        Ok(())
    })
    .unwrap();
    assert_eq!(stats.flushed_bytes, 64);
    assert_eq!(stats.flush_count, 2);
}
