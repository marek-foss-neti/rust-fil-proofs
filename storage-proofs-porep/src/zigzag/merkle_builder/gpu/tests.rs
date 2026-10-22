use super::*;

fn data(leaves: usize) -> Vec<u8> {
    (0..leaves)
        .flat_map(|index| Scalar::from(index as u64 + 1).to_repr())
        .collect()
}

fn config(batch_size: usize, cpu_threshold: usize) -> TreeRGpuConfig {
    TreeRGpuConfig {
        enabled: true,
        batch_size,
        cpu_threshold,
    }
}

fn cpu_batch(pairs: &[GenericArray<Scalar, U2>], _: &mut GpuBuildStats) -> Result<Vec<Scalar>> {
    let mut poseidon = Poseidon::new(&POSEIDON_CONSTANTS_2);
    Ok(pairs
        .iter()
        .map(|pair| {
            poseidon.set_preimage(pair.as_slice());
            poseidon.hash()
        })
        .collect())
}

#[test]
fn configuration_enforces_bounded_batches_and_threshold() {
    assert!(config(0, 0).validate().is_err());
    assert!(config(MAX_BATCH + 1, 0).validate().is_err());
    assert!(config(1, MAX_BATCH + 1).validate().is_err());
    assert!(config(MAX_BATCH, 0).validate().is_ok());
}

// The injected hasher exercises the entire batching/store path without a GPU;
// the ignored test below covers the actual Neptune device implementation.
#[test]
#[cfg(target_os = "linux")]
fn batched_stores_match_cpu_bytes_and_reopened_proofs() {
    use storage_proofs_core::merkle::{create_disk_tree, MerkleProofTrait};
    for leaves in [2, 16, 256] {
        let directory = tempfile::tempdir().unwrap();
        let input = data(leaves);
        let reference_config = StoreConfig::new(directory.path(), "reference", 0);
        let reference =
            create_base_merkle_tree::<BinaryTree>(Some(reference_config.clone()), leaves, &input)
                .unwrap();
        let expected = fs::read(StoreConfig::data_path(
            &reference_config.path,
            &reference_config.id,
        ))
        .unwrap();
        for (index, (batch, threshold)) in [(1, 0), (3, 2), (5, 0), (7, 4), (5, MAX_BATCH)]
            .iter()
            .enumerate()
        {
            let config = config(*batch, *threshold);
            let store = StoreConfig::new(directory.path(), format!("batched-{index}"), 0);
            let cancelled = AtomicBool::new(false);
            let snapshot =
                LeafSnapshot::capture(store.clone(), leaves, &input, 0, &cancelled).unwrap();
            let ((root, mut stats), dirty) =
                with_dirty_budget(snapshot.file.as_file(), 256, &cancelled, |dirty| {
                    build_batched_levels(
                        snapshot.file.as_file(),
                        snapshot.geometry,
                        config,
                        &cancelled,
                        Some(dirty),
                        cpu_batch,
                    )
                })
                .unwrap();
            assert_eq!(dirty.flushed_bytes, ((leaves - 1) * 32) as u64);
            assert!(dirty.peak_outstanding_bytes <= 256);
            let gpu = stats.gpu.as_ref().unwrap();
            assert_eq!(gpu.gpu_hashes + gpu.cpu_hashes, (leaves - 1) as u64);
            assert_eq!(gpu.h2d_payload_bytes, gpu.gpu_hashes * 64);
            assert_eq!(gpu.d2h_payload_bytes, gpu.gpu_hashes * 32);
            assert_eq!(
                gpu.host_buffer_peak_bound_bytes,
                (*batch).min(leaves / 2) * HOST_BYTES_PER_PAIR + CONSTANTS_ALLOWANCE
            );
            if *threshold >= leaves / 2 {
                assert_eq!(gpu.batches, 0);
            }
            if *threshold == 0 {
                assert_eq!(gpu.cpu_hashes, 0);
            }
            stats.dirty_budget = Some(dirty);
            assert_eq!(fs::read(snapshot.file.path()).unwrap(), expected);
            let actual = CompletedTree {
                snapshot,
                root,
                stats,
            }
            .publish::<BinaryTree>(0, &cancelled)
            .unwrap();
            assert_eq!(actual.root(), reference.root());
            drop(actual);
            let reopened = create_disk_tree::<BinaryTree>(2 * leaves - 1, &[store]).unwrap();
            for leaf in [0, leaves / 2, leaves - 1] {
                assert!(reopened.gen_proof(leaf).unwrap().validate(leaf));
            }
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn failures_and_cancellation_leave_no_published_store_and_allow_retry() {
    let directory = tempfile::tempdir().unwrap();
    let store = StoreConfig::new(directory.path(), "retry", 0);
    let input = data(16);
    let cancelled = AtomicBool::new(false);
    for failure in 0..4 {
        let mut bytes = input.clone();
        if failure == 0 {
            bytes[..32].fill(255);
        }
        let snapshot = LeafSnapshot::capture(store.clone(), 16, &bytes, 0, &cancelled).unwrap();
        let mut calls = 0;
        let result = build_batched_levels(
            snapshot.file.as_file(),
            snapshot.geometry,
            config(3, 0),
            &cancelled,
            None,
            |pairs, stats| {
                calls += 1;
                match failure {
                    1 if calls == 2 => bail!("injected kernel failure after a successful batch"),
                    2 => Ok(Vec::new()),
                    3 => {
                        cancelled.store(true, Ordering::Release);
                        cpu_batch(pairs, stats)
                    }
                    _ => cpu_batch(pairs, stats),
                }
            },
        );
        assert!(result.is_err());
        if failure == 0 {
            assert_eq!(calls, 0);
        }
        drop(snapshot);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
        cancelled.store(false, Ordering::Release);
    }
    let snapshot = LeafSnapshot::capture(store.clone(), 16, &input, 0, &cancelled).unwrap();
    let (root, stats) = build_batched_levels(
        snapshot.file.as_file(),
        snapshot.geometry,
        config(3, 0),
        &cancelled,
        None,
        cpu_batch,
    )
    .unwrap();
    CompletedTree {
        snapshot,
        root,
        stats,
    }
    .publish::<BinaryTree>(0, &cancelled)
    .unwrap();
    assert!(StoreConfig::data_path(&store.path, &store.id).is_file());
}

#[test]
fn batch_lock_yields_to_priority_and_cancellation() {
    use fs2::FileExt;
    let directory = tempfile::tempdir().unwrap();
    let priority = lock_file(directory.path(), "bellman.priority.lock").unwrap();
    FileExt::lock_exclusive(&priority).unwrap();
    assert!(priority_taken_in(directory.path()).unwrap());
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancel = cancelled.clone();
    let path = directory.path().to_owned();
    let (tx, rx) = mpsc::channel();
    let (priority_tx, priority_rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        tx.send(batch_lease_in(&path, &cancel, || {
            let _ = priority_tx.send(());
        }))
        .unwrap()
    });
    priority_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(rx.recv_timeout(Duration::from_millis(50)).is_err());
    cancelled.store(true, Ordering::Release);
    assert!(rx.recv_timeout(Duration::from_secs(2)).unwrap().is_err());
    handle.join().unwrap();
    FileExt::unlock(&priority).unwrap();
    assert!(!priority_taken_in(directory.path()).unwrap());
    cancelled.store(false, Ordering::Release);
    let lease = batch_lease_in(directory.path(), &cancelled, || {}).unwrap();
    let competitor = lock_file(directory.path(), "bellman.gpu.lock").unwrap();
    assert_eq!(
        FileExt::try_lock_exclusive(&competitor).unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    drop(lease);
    FileExt::try_lock_exclusive(&competitor).unwrap();
}

#[test]
#[cfg(target_os = "linux")]
fn write_and_flush_failures_cancel_batched_build_and_join_writeback() {
    struct FailingIo<'a> {
        file: &'a File,
        fail_write: bool,
    }
    impl PositionedIo for FailingIo<'_> {
        fn read_at(&self, bytes: &mut [u8], offset: u64) -> io::Result<usize> {
            FileExt::read_at(self.file, bytes, offset)
        }
        fn write_at(&self, bytes: &[u8], offset: u64) -> io::Result<usize> {
            if self.fail_write {
                return Err(io::Error::from_raw_os_error(libc::ENOSPC));
            }
            FileExt::write_at(self.file, bytes, offset)
        }
        fn start_writeback(&self, _: u64, _: u64) -> io::Result<()> {
            Ok(())
        }
        fn sync_data(&self) -> io::Result<()> {
            Err(io::Error::from_raw_os_error(libc::EIO))
        }
    }
    for fail_write in [true, false] {
        let directory = tempfile::tempdir().unwrap();
        let store = StoreConfig::new(directory.path(), "io-failure", 0);
        let cancelled = AtomicBool::new(false);
        let snapshot = LeafSnapshot::capture(store, 16, &data(16), 0, &cancelled).unwrap();
        let io = FailingIo {
            file: snapshot.file.as_file(),
            fail_write,
        };
        let result = with_dirty_budget(&io, 256, &cancelled, |dirty| {
            build_batched_levels(
                &io,
                snapshot.geometry,
                config(3, 0),
                &cancelled,
                Some(dirty),
                cpu_batch,
            )
        });
        assert!(result.is_err());
        assert!(cancelled.load(Ordering::Acquire));
        drop(snapshot);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }
}

#[test]
#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "opencl")))]
#[ignore = "requires an explicitly configured, idle remote GPU and shared lock directory"]
fn neptune_gpu_stores_match_cpu_across_layers_and_partial_batches() {
    crate::zigzag::gpu::validate_tree_gpu_backend().unwrap();
    for (batch, threshold, leaves) in [
        (3, 0, 1024),
        (257, 4, 1024),
        // Exercise full production-sized batches as well as the CPU upper
        // levels; tiny fixtures alone never reach the default allocation size.
        (DEFAULT_BATCH, DEFAULT_CPU_THRESHOLD, 1 << 20),
    ] {
        let worker = TreeGpuWorker::start(config(batch, threshold)).unwrap();
        for layer in 0..2 {
            let directory = tempfile::tempdir().unwrap();
            let input = data(leaves);
            let reference_config = StoreConfig::new(directory.path(), "reference", 0);
            let expected = create_base_merkle_tree::<BinaryTree>(
                Some(reference_config.clone()),
                leaves,
                &input,
            )
            .unwrap();
            let store = StoreConfig::new(directory.path(), "gpu", 0);
            let cancelled = Arc::new(AtomicBool::new(false));
            let snapshot =
                LeafSnapshot::capture(store.clone(), leaves, &input, layer, &cancelled).unwrap();
            let completed = worker.build(snapshot, layer, cancelled.clone()).unwrap();
            assert!(completed.stats.gpu.as_ref().unwrap().gpu_hashes > 0);
            let actual = completed.publish::<BinaryTree>(layer, &cancelled).unwrap();
            assert_eq!(actual.root(), expected.root());
            assert_eq!(
                fs::read(StoreConfig::data_path(&store.path, &store.id)).unwrap(),
                fs::read(StoreConfig::data_path(
                    &reference_config.path,
                    &reference_config.id
                ))
                .unwrap()
            );
        }
    }
}
