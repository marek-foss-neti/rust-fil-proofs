use super::*;
use filecoin_hashers::poseidon::PoseidonFunction;
use merkletree::hash::Algorithm;
use storage_proofs_core::merkle::{create_disk_tree, MerkleProofTrait};

fn data(leaves: usize) -> Vec<u8> {
    (0..leaves)
        .flat_map(|index| Scalar::from(index as u64 + 1).to_repr())
        .collect()
}

#[test]
fn geometry_rejects_invalid_shapes_and_overflow() {
    for leaves in [0, 1, 3, 6, usize::MAX, 1usize << (usize::BITS - 1)] {
        assert!(Geometry::new(leaves).is_err(), "accepted {}", leaves);
    }
    let geometry = Geometry::new(1 << 30).unwrap();
    assert_eq!(geometry.leaf_bytes as u64, 32 << 30);
    assert_eq!(geometry.file_bytes, (64 << 30) - 32);
}

#[test]
fn reused_poseidon_matches_public_binary_and_multi_node_hashes() {
    let mut reused = Poseidon::new(&POSEIDON_CONSTANTS_2);
    let mut reference = PoseidonFunction::default();
    for index in 0..128 {
        let left = Scalar::from(index);
        let right = Scalar::from(index * index + 17);
        reused.set_preimage(&[left, right]);
        let actual = PoseidonDomain::from(reused.hash());
        assert_eq!(actual, reference.node(left.into(), right.into(), 0));
        assert_eq!(
            actual,
            reference.multi_node(&[left.into(), right.into()], 9)
        );
    }
}

#[test]
#[cfg(target_os = "linux")]
fn complete_store_matches_reference_across_chunk_boundaries_and_reopens() {
    for leaves in [2, 4, 16, 256, 16384] {
        let directory = tempfile::tempdir().unwrap();
        let reference_config = StoreConfig::new(directory.path(), "reference", 0);
        let config = StoreConfig::new(directory.path(), "binary", 0);
        let input = data(leaves);
        let expected =
            create_base_merkle_tree::<BinaryTree>(Some(reference_config.clone()), leaves, &input)
                .unwrap();
        let actual = build_tree_r::<BinaryTree>(
            TreeRBuilder::Binary,
            Some(config.clone()),
            leaves,
            &input,
            0,
        )
        .unwrap();
        assert_eq!(expected.root(), actual.root());
        assert_eq!(
            fs::read(StoreConfig::data_path(&config.path, &config.id)).unwrap(),
            fs::read(StoreConfig::data_path(
                &reference_config.path,
                &reference_config.id
            ))
            .unwrap()
        );
        drop(actual);
        let reopened = create_disk_tree::<BinaryTree>(2 * leaves - 1, &[config]).unwrap();
        for index in [0, leaves / 2, leaves - 1] {
            assert!(reopened.gen_proof(index).unwrap().validate(index));
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn snapshot_is_independent_and_buffers_are_bounded_with_partial_chunks() {
    let directory = tempfile::tempdir().unwrap();
    let leaves = 256;
    let mut input = data(leaves);
    let expected = create_base_merkle_tree::<BinaryTree>(None, leaves, &input).unwrap();
    let config = StoreConfig::new(directory.path(), "snapshot", 0);
    let cancelled = AtomicBool::new(false);
    let snapshot = LeafSnapshot::capture(config, leaves, &input, 0, &cancelled).unwrap();
    input.fill(0); // Models the next layer overwriting the replica after capture.
    let (root, stats) =
        build_levels(snapshot.file.as_file(), snapshot.geometry, 3, 4, &cancelled).unwrap();
    let expected_root: [u8; 32] = expected.root().into();
    assert_eq!(root, expected_root);
    assert_eq!(stats.pairs, (leaves - 1) as u64);
    assert_eq!(stats.read_bytes, (leaves as u64 - 1) * 64);
    assert_eq!(stats.written_bytes, (leaves as u64 - 1) * 32);
    assert!(stats.workers <= 4);
    assert!(stats.buffer_bytes <= 4 * 3 * 160);
    assert_eq!(stats.buffer_allocations, stats.workers * 3);
}

#[test]
#[cfg(target_os = "linux")]
fn failed_or_cancelled_build_never_publishes_and_allows_retry() {
    let directory = tempfile::tempdir().unwrap();
    let config = StoreConfig::new(directory.path(), "retry", 0);
    let final_path = StoreConfig::data_path(&config.path, &config.id);
    let input = data(16);
    let cancelled = AtomicBool::new(false);
    let mut invalid = input.clone();
    invalid[0..32].fill(255);
    let snapshot = LeafSnapshot::capture(config.clone(), 16, &invalid, 0, &cancelled).unwrap();
    assert!(snapshot.build(0, &cancelled).is_err());
    assert!(!final_path.exists());
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    cancelled.store(false, Ordering::Relaxed);
    let snapshot = LeafSnapshot::capture(config.clone(), 16, &input, 0, &cancelled).unwrap();
    cancelled.store(true, Ordering::Relaxed);
    assert!(snapshot.build(0, &cancelled).is_err());
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    build_tree_r::<BinaryTree>(TreeRBuilder::Binary, Some(config.clone()), 16, &input, 0).unwrap();
    let bytes = fs::read(&final_path).unwrap();
    assert!(build_tree_r::<BinaryTree>(TreeRBuilder::Binary, Some(config), 16, &input, 0).is_err());
    assert_eq!(fs::read(final_path).unwrap(), bytes);
}

struct ShortIo {
    file: File,
    calls: AtomicUsize,
    fail_after: Option<usize>,
    cancel_after: Option<usize>,
    cancelled: AtomicBool,
    zero_write: bool,
    fail_writeback: bool,
}

impl PositionedIo for ShortIo {
    fn sync_data(&self) -> io::Result<()> {
        self.file.sync_data()
    }
    fn start_writeback(&self, _offset: u64, _bytes: u64) -> io::Result<()> {
        if self.fail_writeback {
            Err(io::Error::from_raw_os_error(libc::EIO))
        } else {
            Ok(())
        }
    }
    fn read_at(&self, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
        let take = buffer.len().min(17);
        FileExt::read_at(&self.file, &mut buffer[..take], offset)
    }
    fn write_at(&self, buffer: &[u8], offset: u64) -> io::Result<usize> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fail_after.is_some_and(|n| call >= n) {
            return Err(io::Error::from_raw_os_error(libc::ENOSPC));
        }
        if self.cancel_after.is_some_and(|n| call >= n) {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        if self.zero_write {
            return Ok(0);
        }
        if call % 7 == 0 {
            return Err(io::ErrorKind::Interrupted.into());
        }
        FileExt::write_at(&self.file, &buffer[..buffer.len().min(13)], offset)
    }
}

#[test]
fn positioned_io_handles_short_transfers_interruptions_eof_and_write_zero() {
    let mut io = ShortIo {
        file: tempfile::tempfile().unwrap(),
        calls: AtomicUsize::new(0),
        fail_after: None,
        cancel_after: None,
        cancelled: AtomicBool::new(false),
        zero_write: false,
        fail_writeback: false,
    };
    let bytes = data(32);
    write_all_at(&io, &bytes, 19).unwrap();
    let mut actual = vec![0; bytes.len()];
    read_exact_at(&io, &mut actual, 19).unwrap();
    assert_eq!(actual, bytes);
    assert_eq!(
        read_exact_at(&io, &mut actual, 20).unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
    io.zero_write = true;
    assert_eq!(
        write_all_at(&io, &bytes, 0).unwrap_err().kind(),
        io::ErrorKind::WriteZero
    );
}

#[test]
#[cfg(target_os = "linux")]
fn worker_errors_and_cancellation_join_before_return() {
    for cancellation in [false, true] {
        let geometry = Geometry::new(256).unwrap();
        let file = tempfile::tempfile().unwrap();
        reserve_file(&file, geometry.file_bytes).unwrap();
        write_all_at(&file, &data(geometry.leaves), 0).unwrap();
        let io = ShortIo {
            file,
            calls: AtomicUsize::new(0),
            fail_after: (!cancellation).then_some(4),
            cancel_after: cancellation.then_some(4),
            cancelled: AtomicBool::new(false),
            zero_write: false,
            fail_writeback: false,
        };
        let result = build_levels(&io, geometry, 3, 4, &io.cancelled);
        let error = result.unwrap_err();
        if cancellation {
            assert!(error.to_string().contains("cancelled"));
        } else {
            assert_eq!(
                error.downcast_ref::<io::Error>().unwrap().raw_os_error(),
                Some(libc::ENOSPC)
            );
        }
        // All scoped worker activity finished; it is now safe to close/unlink.
        assert!(io.cancelled.load(Ordering::Relaxed));
        assert_eq!(io.file.metadata().unwrap().len(), geometry.file_bytes);
    }
}

#[test]
#[cfg(target_os = "linux")]
fn writeback_errors_are_returned_after_workers_join() {
    let geometry = Geometry::new(256).unwrap();
    let file = tempfile::tempfile().unwrap();
    reserve_file(&file, geometry.file_bytes).unwrap();
    write_all_at(&file, &data(geometry.leaves), 0).unwrap();
    let io = ShortIo {
        file,
        calls: AtomicUsize::new(0),
        fail_after: None,
        cancel_after: None,
        cancelled: AtomicBool::new(false),
        zero_write: false,
        fail_writeback: true,
    };
    let error = build_levels(&io, geometry, 3, 4, &io.cancelled).unwrap_err();
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().raw_os_error(),
        Some(libc::EIO)
    );
}

#[test]
fn dirty_budget_bounds_concurrent_writes_and_propagates_flush_failure() {
    struct CountIo {
        dirty: AtomicUsize,
        peak: AtomicUsize,
        flushes: AtomicUsize,
        fail: bool,
    }
    impl PositionedIo for CountIo {
        fn read_at(&self, _: &mut [u8], _: u64) -> io::Result<usize> {
            unreachable!()
        }
        fn start_writeback(&self, _: u64, _: u64) -> io::Result<()> {
            Ok(())
        }
        fn write_at(&self, bytes: &[u8], _: u64) -> io::Result<usize> {
            let dirty = self.dirty.fetch_add(bytes.len(), Ordering::SeqCst) + bytes.len();
            self.peak.fetch_max(dirty, Ordering::SeqCst);
            Ok(bytes.len())
        }
        fn sync_data(&self) -> io::Result<()> {
            if self.fail {
                return Err(io::Error::from_raw_os_error(libc::EIO));
            }
            self.flushes.fetch_add(1, Ordering::SeqCst);
            self.dirty.store(0, Ordering::SeqCst);
            Ok(())
        }
    }
    let mut io = CountIo {
        dirty: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
        flushes: AtomicUsize::new(0),
        fail: false,
    };
    let budget = DirtyBudget::new(96).unwrap();
    (0..32)
        .into_par_iter()
        .for_each(|i| budget.write(&io, &[0; 32], i * 32).unwrap());
    assert!(io.peak.load(Ordering::SeqCst) <= 96);
    assert_eq!(io.flushes.load(Ordering::SeqCst), 10);
    assert!(budget.write(&io, &[0; 128], 0).is_err());
    io.fail = true;
    let budget = DirtyBudget::new(32).unwrap();
    budget.write(&io, &[0; 32], 0).unwrap();
    assert_eq!(
        budget
            .write(&io, &[0; 32], 32)
            .unwrap_err()
            .downcast_ref::<io::Error>()
            .unwrap()
            .raw_os_error(),
        Some(libc::EIO)
    );
}

#[test]
#[ignore = "remote machine only; supply a small dedicated tmpfs in ZIGZAG_TREE_ENOSPC_DIR"]
#[cfg(target_os = "linux")]
fn physical_preallocation_failure_leaves_no_store() {
    let path = PathBuf::from(
        std::env::var_os("ZIGZAG_TREE_ENOSPC_DIR").expect("dedicated small filesystem required"),
    );
    let directory = tempfile::tempdir_in(path).unwrap();
    let config = StoreConfig::new(directory.path(), "no-space", 0);
    let input = data(65536); // Full store needs almost 4 MiB; fixture mount is 1 MiB.
    let error =
        build_tree_r::<BinaryTree>(TreeRBuilder::Binary, Some(config.clone()), 65536, &input, 0)
            .unwrap_err();
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().raw_os_error(),
        Some(libc::ENOSPC)
    );
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    // A smaller retry on the same config can succeed because no artifact escaped.
    build_tree_r::<BinaryTree>(TreeRBuilder::Binary, Some(config), 16, &data(16), 0).unwrap();
}

#[test]
#[ignore = "remote large-store compatibility fixture; requires ZIGZAG_TREE_TEST_BYTES and ZIGZAG_TREE_TEST_DIR"]
#[cfg(target_os = "linux")]
fn full_size_store_matches_reference() {
    use sha2::{Digest, Sha256};
    let bytes: usize = std::env::var("ZIGZAG_TREE_TEST_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    assert!(bytes == 512 << 20 || bytes == 32usize << 30);
    let output = PathBuf::from(std::env::var_os("ZIGZAG_TREE_TEST_DIR").unwrap());
    let directory = tempfile::tempdir_in(&output).unwrap();
    let leaves = bytes / 32;
    let input = tempfile::tempfile_in(directory.path()).unwrap();
    reserve_file(&input, bytes as u64).unwrap();
    let mut buffer = vec![0u8; 1 << 20];
    for first in (0..leaves).step_by(buffer.len() / 32) {
        for (index, node) in buffer.chunks_exact_mut(32).enumerate() {
            node.copy_from_slice(&Scalar::from((first + index + 1) as u64).to_repr());
        }
        write_all_at(&input, &buffer, (first * 32) as u64).unwrap();
    }
    // Private, fixed-length read-only input mapping; no writable sparse mapping.
    let data = unsafe { memmap2::MmapOptions::new().map(&input).unwrap() };
    let legacy_config = StoreConfig::new(directory.path(), "reference", 0);
    let config = StoreConfig::new(directory.path(), "binary", 0);
    let started = Instant::now();
    let legacy =
        create_base_merkle_tree::<BinaryTree>(Some(legacy_config.clone()), leaves, &data).unwrap();
    let reference_build_ms = started.elapsed().as_millis();
    File::open(StoreConfig::data_path(
        &legacy_config.path,
        &legacy_config.id,
    ))
    .unwrap()
    .sync_all()
    .unwrap();
    let legacy_ms = started.elapsed().as_millis();
    let started = Instant::now();
    let cancelled = AtomicBool::new(false);
    let snapshot = LeafSnapshot::capture(config.clone(), leaves, &data, 0, &cancelled).unwrap();
    let snapshot_ms = started.elapsed().as_millis();
    let build_started = Instant::now();
    let completed = snapshot.build(0, &cancelled).unwrap();
    let build_ms = build_started.elapsed().as_millis();
    let stats = serde_json::to_value(&completed.stats).unwrap();
    let sync_started = Instant::now();
    let actual = completed.publish::<BinaryTree>(0, &cancelled).unwrap();
    let sync_ms = sync_started.elapsed().as_millis();
    let binary_ms = started.elapsed().as_millis();
    assert_eq!(legacy.root(), actual.root());
    let left = File::open(StoreConfig::data_path(
        &legacy_config.path,
        &legacy_config.id,
    ))
    .unwrap();
    let right = File::open(StoreConfig::data_path(&config.path, &config.id)).unwrap();
    let mut other = vec![0; buffer.len()];
    let mut digest = Sha256::new();
    let length = left.metadata().unwrap().len();
    assert_eq!(right.metadata().unwrap().len(), length);
    let mut offset = 0;
    while offset < length {
        let take = (length - offset).min(buffer.len() as u64) as usize;
        read_exact_at(&left, &mut buffer[..take], offset).unwrap();
        read_exact_at(&right, &mut other[..take], offset).unwrap();
        assert_eq!(
            &buffer[..take],
            &other[..take],
            "store mismatch at {offset}"
        );
        digest.update(&buffer[..take]);
        offset += take as u64;
    }
    let reopened = create_disk_tree::<BinaryTree>(2 * leaves - 1, &[config]).unwrap();
    for index in [0, leaves / 2, leaves - 1] {
        assert!(reopened.gen_proof(index).unwrap().validate(index));
    }
    let report = serde_json::json!({ "sector_bytes": bytes, "store_bytes": length, "all_bytes_equal": true,
        "root": hex::encode(actual.root().as_ref() as &[u8]), "store_sha256": hex::encode(digest.finalize()),
        "reference_ms": legacy_ms, "binary_ms": binary_ms,
        "reference_build_ms": reference_build_ms,
        "snapshot_ms": snapshot_ms, "build_ms": build_ms, "sync_ms": sync_ms, "stats": stats,
        "scope": "single-tree compatibility fixture; not full sealing performance acceptance" });
    fs::write(
        output.join(format!("tree-r-{bytes}.json")),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{report}");
}
