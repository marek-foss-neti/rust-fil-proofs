use super::*;
use std::{
    cell::Cell,
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
};

fn patterned(bytes: usize) -> Vec<u8> {
    let mut x = 0x123456789abcdef0u64;
    (0..bytes)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

#[test]
fn fixed_group_matches_reference_for_every_bit_and_padding() -> Result<()> {
    for bit in 0..1024 {
        let mut padded = [0; 128];
        padded[bit / 8] = 1 << (bit % 8);
        let mut expected = Vec::new();
        fr32::write_unpadded(&padded, &mut expected, 0, 127)?;
        let mut actual = [0; 127];
        unpad_group(&padded, &mut actual);
        assert_eq!(actual.as_slice(), expected, "bit={bit}");
    }
    for padded in [[0u8; 128].to_vec(), [255u8; 128].to_vec(), patterned(128)] {
        let mut expected = Vec::new();
        fr32::write_unpadded(&padded, &mut expected, 0, 127)?;
        let mut actual = [0; 127];
        unpad_group(&padded, &mut actual);
        assert_eq!(actual.as_slice(), expected);
    }
    Ok(())
}

#[test]
fn parallel_ranges_match_reference_across_groups_batches_and_workers() -> Result<()> {
    let padded = patterned(2 << 20);
    let capacity = padded.len() / 128 * 127;
    let mut expected = Vec::new();
    fr32::write_unpadded(&padded, &mut expected, 0, capacity)?;
    let config = ZigZagUnsealOutputConfig::parse("parallel", "1")?;
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?;
        pool.install(|| -> Result<()> {
            for offset in [0, 1, 126, 127, 128, 129, capacity - 1, capacity] {
                for length in [0, 1, 1024, 1 << 20, capacity - offset] {
                    if length > capacity - offset {
                        continue;
                    }
                    let mut actual = Vec::new();
                    let written = write_range(
                        &padded,
                        &mut actual,
                        offset as u64,
                        length as u64,
                        config,
                        None,
                    )?;
                    assert_eq!(written, length);
                    assert_eq!(
                        actual,
                        expected[offset..offset + length],
                        "workers={workers} range={offset}+{length}"
                    );
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

// Rc deliberately makes this writer neither Send nor Sync. Parallel conversion
// must never capture the caller's writer or require either bound.
struct PartialWriter<'a> {
    bytes: Vec<u8>,
    calls: Rc<Cell<usize>>,
    action: &'static str,
    cancel: Option<&'a AtomicBool>,
}
impl Write for PartialWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call == 0 {
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self.bytes.len() >= 7 {
            match self.action {
                "zero" => return Ok(0),
                "error" => return Err(io::ErrorKind::BrokenPipe.into()),
                "panic" => panic!("injected output panic"),
                _ => (),
            }
        }
        let n = 7.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..n]);
        if let Some(cancel) = self.cancel {
            cancel.store(true, Ordering::Relaxed);
        }
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn writers_handle_short_writes_interrupts_zero_error_cancel_and_panic() -> Result<()> {
    let padded = patterned(128 * 8);
    let config = ZigZagUnsealOutputConfig::parse("parallel", "1")?;
    let mut expected = Vec::new();
    fr32::write_unpadded(&padded, &mut expected, 129, 500)?;
    for action in ["ok", "zero", "error", "cancel", "panic"] {
        let cancel = AtomicBool::new(false);
        let mut writer = PartialWriter {
            bytes: Vec::new(),
            calls: Rc::new(Cell::new(0)),
            action,
            cancel: if action == "cancel" {
                Some(&cancel)
            } else {
                None
            },
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            write_range(&padded, &mut writer, 129, 500, config, Some(&cancel))
        }));
        if action == "ok" {
            assert_eq!(result.unwrap()?, 500);
            assert_eq!(writer.bytes, expected);
        } else {
            assert!(result.is_err() || result.unwrap().is_err());
            assert_eq!(writer.bytes, expected[..7]);
            assert!(writer.calls.get() <= 3);
        }
        // Retry uses a new writer and the unchanged padded source.
        let mut retry = Vec::new();
        write_range(&padded, &mut retry, 129, 500, config, None)?;
        assert_eq!(retry, expected);
    }
    Ok(())
}

#[test]
fn streaming_writer_preserves_order_and_bounds_its_reused_buffer() -> Result<()> {
    let padded = patterned(128 * 13);
    let range = OutputRange::new(padded.len(), 126, 127 * 11)?;
    let mut actual = Vec::new();
    {
        let mut writer = ParallelOutput::new(&mut actual, range.clone(), 128 * 3, None, true)?;
        let buffer_ptr = writer.buffer.as_ptr();
        let capacity = writer.buffer.capacity();
        assert!(writer
            .write_chunk(range.padded.start + 128, &padded[..128])
            .is_err());
        assert_eq!(writer.stats.written_bytes, 0);
        for start in (range.padded.start..range.padded.end).step_by(128 * 3) {
            let end = (start + 128 * 3).min(range.padded.end);
            writer.write_chunk(start, &padded[start..end])?;
            assert_eq!(writer.buffer.as_ptr(), buffer_ptr);
            assert_eq!(writer.buffer.capacity(), capacity);
        }
        assert_eq!(writer.finish()?, 127 * 11);
    }
    let mut expected = Vec::new();
    fr32::write_unpadded(&padded, &mut expected, 126, 127 * 11)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn configuration_ranges_and_cancellation_fail_before_output() -> Result<()> {
    for (mode, chunk) in [
        ("bad", "4"),
        ("parallel", "0"),
        ("stream", "65"),
        ("reference", ""),
    ] {
        assert!(ZigZagUnsealOutputConfig::parse(mode, chunk).is_err());
    }
    assert_eq!(
        ZigZagUnsealOutputConfig::parse("parallel", "4")?.buffer_bytes(512 << 20)?,
        4 * 1024 * 1024 / 128 * 127
    );
    for (offset, len) in [(2032, 1), (2033, 0), (u64::MAX, 1)] {
        assert!(OutputRange::new(2048, offset, len).is_err());
    }
    assert!(OutputRange::new(129, 0, 1).is_err());
    assert!(OutputRange::new(usize::MAX - 127, u64::MAX - 1, 2).is_err());
    assert!(OutputRange::new(2048, 2032, 0)?.padded.is_empty());
    let cancel = AtomicBool::new(true);
    let mut output = Vec::new();
    assert!(write_range(
        &[0; 128],
        &mut output,
        0,
        127,
        ZigZagUnsealOutputConfig::parse("parallel", "1")?,
        Some(&cancel)
    )
    .is_err());
    assert!(output.is_empty());
    Ok(())
}
