use super::*;
use crate::zigzag::vanilla::vde::EncodedSink;
use blstrs::Scalar;
use ff::PrimeField;
use filecoin_hashers::poseidon::{PoseidonDomain, PoseidonHasher};
use generic_array::typenum::{U0, U2};
use merkletree::store::DiskStore;
use storage_proofs_core::merkle::{create_base_merkle_tree, MerkleTreeWrapper};

type Tree = MerkleTreeWrapper<PoseidonHasher, DiskStore<PoseidonDomain>, U2, U0, U0>;

fn input(leaves: usize) -> Vec<u8> {
    (0..leaves)
        .flat_map(|i| Scalar::from(i as u64 + 1).to_repr())
        .collect()
}

fn feed(writer: &mut SnapshotWriter, data: &[u8], forward: bool, chunk: usize) -> Result<()> {
    let mut captured = 0;
    while captured < data.len() {
        let length = chunk.min(data.len() - captured);
        let offset = if forward {
            captured
        } else {
            data.len() - captured - length
        };
        writer.capture(offset, &data[offset..offset + length])?;
        captured += length;
    }
    Ok(())
}

#[test]
fn pipeline_settings_reject_incompatible_builders_and_bounds() {
    for value in ["", "0", "65", "-1", "+2", " 2", "99999999999999999999999"] {
        assert!(parse_bound(value, "test").is_err());
    }
    assert_eq!(parse_bound("64", "test").unwrap(), 64);
    let config = LayerPipelineConfig {
        mode: LayerPipelineMode::Stream,
        tree_threads: Some(2),
        snapshot_buffer_mib: 1,
    };
    assert!(config
        .validate::<Tree>(TreeRBuilder::Legacy, Some(Path::new(".")))
        .is_err());
    assert!(config.validate::<Tree>(TreeRBuilder::Binary, None).is_err());
}

#[test]
#[cfg(target_os = "linux")]
fn streaming_preserves_reverse_offsets_with_one_buffer_and_slow_writer() -> Result<()> {
    for forward in [true, false] {
        let directory = tempfile::tempdir()?;
        let config = StoreConfig::new(directory.path(), "stream", 0);
        let bytes = input(256);
        let expected = create_base_merkle_tree::<Tree>(None, 256, &bytes)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut writer = SnapshotWriter::start_with_hook(
            config.clone(),
            256,
            7,
            forward,
            96,
            1,
            cancelled.clone(),
            |_, _| {
                thread::sleep(Duration::from_micros(100));
                Ok(())
            },
        )?;
        let result = feed(&mut writer, &bytes, forward, 96);
        let snapshot = writer.finish(result)?;
        let tree = snapshot
            .build_with_dirty_limit(7, &cancelled, Some(1024 * 1024))?
            .publish::<Tree>(7, &cancelled)?;
        assert_eq!(tree.root(), expected.root());
        let stored = std::fs::read(StoreConfig::data_path(&config.path, &config.id))?;
        assert_eq!(&stored[..bytes.len()], &bytes);
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
    }
    Ok(())
}

#[test]
#[cfg(target_os = "linux")]
fn writer_io_failure_and_panic_join_cleanup_and_allow_retry() -> Result<()> {
    for panic in [false, true] {
        let directory = tempfile::tempdir()?;
        let config = StoreConfig::new(directory.path(), "retry", 0);
        let bytes = input(256);
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut writer = SnapshotWriter::start_with_hook(
            config.clone(),
            256,
            0,
            true,
            96,
            1,
            cancelled.clone(),
            move |index, _| {
                if index == 2 {
                    if panic {
                        panic!("injected writer panic");
                    }
                    return Err(std::io::Error::from_raw_os_error(libc::ENOSPC).into());
                }
                Ok(())
            },
        )?;
        let result = feed(&mut writer, &bytes, true, 96);
        let error = writer.finish(result).err().expect("writer must fail");
        if panic {
            assert!(error.to_string().contains("panicked"));
        } else {
            assert_eq!(
                error
                    .downcast_ref::<std::io::Error>()
                    .unwrap()
                    .raw_os_error(),
                Some(libc::ENOSPC)
            );
        }
        assert!(cancelled.load(Ordering::Acquire));
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 0);
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut writer = SnapshotWriter::start_with_hook(
            config,
            256,
            0,
            false,
            96,
            1,
            cancelled.clone(),
            |_, _| Ok(()),
        )?;
        let result = feed(&mut writer, &bytes, false, 96);
        writer
            .finish(result)?
            .build(0, &cancelled)?
            .publish::<Tree>(0, &cancelled)?;
    }
    Ok(())
}

#[test]
#[cfg(target_os = "linux")]
fn invalid_order_and_encoder_failure_never_publish() -> Result<()> {
    for invalid_order in [true, false] {
        let directory = tempfile::tempdir()?;
        let config = StoreConfig::new(directory.path(), "reject", 0);
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut writer =
            SnapshotWriter::start_with_hook(config, 256, 0, true, 96, 1, cancelled, |_, _| Ok(()))?;
        writer.capture(if invalid_order { 32 } else { 0 }, &input(3))?;
        let result = if invalid_order {
            Ok(())
        } else {
            Err(anyhow!("encoder failure"))
        };
        assert!(writer.finish(result).is_err());
        assert_eq!(std::fs::read_dir(directory.path())?.count(), 0);
    }
    Ok(())
}

#[test]
fn pending_tree_drop_cancels_and_joins() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancelled.clone();
    let finished = Arc::new(AtomicBool::new(false));
    let worker_finished = finished.clone();
    let pending = PendingTree::<()> {
        cancelled,
        handle: Some(thread::spawn(move || {
            while !worker_cancel.load(Ordering::Acquire) {
                thread::yield_now();
            }
            worker_finished.store(true, Ordering::Release);
            Ok(())
        })),
    };
    drop(pending);
    assert!(finished.load(Ordering::Acquire));
}

#[test]
fn encoder_failure_cancels_before_join_and_keeps_both_errors() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancelled.clone();
    let pending = PendingTree::<()> {
        cancelled,
        handle: Some(thread::spawn(move || {
            while !worker_cancel.load(Ordering::Acquire) {
                thread::yield_now();
            }
            Err(anyhow!("background cancelled"))
        })),
    };
    let primary = std::io::Error::from_raw_os_error(libc::ENOSPC);
    let error = pending.abort(primary.into());
    assert_eq!(
        error
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .raw_os_error(),
        Some(libc::ENOSPC)
    );
    assert!(error.to_string().contains("background cancelled"));
    assert!(error
        .to_string()
        .contains(&std::io::Error::from_raw_os_error(libc::ENOSPC).to_string()));
}
