//! The copy-back barrier is independent of the parent reader and KDF.
use super::NODE_SIZE;
use anyhow::{ensure, Result};
use std::{
    ops::Range,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Default)]
pub(super) struct BlockTimes {
    pub blocks: usize,
    pub decode_join_ns: u128,
    pub copy_back_ns: u128,
}

pub(super) fn validate_block(sector_bytes: usize, block_bytes: usize) -> Result<()> {
    ensure!(
        sector_bytes > 0 && sector_bytes % NODE_SIZE == 0,
        "invalid decode sector length"
    );
    ensure!(
        block_bytes > 0 && block_bytes % NODE_SIZE == 0,
        "decode block must contain whole nodes"
    );
    Ok(())
}

pub(super) fn check_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    ensure!(
        !cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed)),
        "ZigZag block decode cancelled"
    );
    Ok(())
}

pub(super) fn visit_blocks(
    data: &mut [u8],
    block: &mut [u8],
    forward: bool,
    cancelled: Option<&AtomicBool>,
    mut decode: impl FnMut(&[u8], &mut [u8], Range<usize>) -> Result<()>,
) -> Result<BlockTimes> {
    validate_block(data.len(), block.len())?;
    let count = data.len().div_ceil(block.len());
    let mut times = BlockTimes::default();
    for ordinal in 0..count {
        check_cancelled(cancelled)?;
        let index = if forward {
            count - 1 - ordinal
        } else {
            ordinal
        };
        let start = index * block.len();
        let end = start + block.len().min(data.len() - start);
        let output = &mut block[..end - start];
        let started = Instant::now();
        // Immutable sector borrow ends here. A failed task or unwind cannot reach
        // copy-back; Rayon joins all readers before returning from the callback.
        decode(data, output, start..end)?;
        times.decode_join_ns += started.elapsed().as_nanos();
        check_cancelled(cancelled)?;
        let started = Instant::now();
        data[start..end].copy_from_slice(output);
        times.copy_back_ns += started.elapsed().as_nanos();
        times.blocks += 1;
    }
    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    #[test]
    fn block_barrier_preserves_current_input_on_error_cancel_and_panic() -> Result<()> {
        for forward in [false, true] {
            for failure in ["error", "cancel", "panic"] {
                let sealed = vec![1; 7 * NODE_SIZE];
                let mut state = sealed.clone();
                let mut block = vec![0; 3 * NODE_SIZE];
                let cancel = AtomicBool::new(false);
                let mut calls = 0;
                let mut failed_range = 0..0;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    visit_blocks(
                        &mut state,
                        &mut block,
                        forward,
                        Some(&cancel),
                        |input, output, range| {
                            assert!(input[range.clone()].iter().all(|b| *b == 1));
                            output.fill(9);
                            calls += 1;
                            if calls == 2 {
                                failed_range = range;
                                match failure {
                                    "error" => anyhow::bail!("injected read failure"),
                                    "cancel" => cancel.store(true, Ordering::Relaxed),
                                    _ => panic!("injected worker panic"),
                                }
                            }
                            Ok(())
                        },
                    )
                }));
                assert!(result.is_err() || result.unwrap().is_err());
                assert_eq!(state[failed_range.clone()], sealed[failed_range]);
                // A retry must start from the preserved sealed copy.
                state.copy_from_slice(&sealed);
                let mut ranges = Vec::new();
                visit_blocks(&mut state, &mut block, forward, None, |_, output, range| {
                    ranges.push(range);
                    output.fill(9);
                    Ok(())
                })?;
                assert!(state.iter().all(|b| *b == 9));
                assert_eq!(
                    ranges,
                    if forward {
                        vec![192..224, 96..192, 0..96]
                    } else {
                        vec![0..96, 96..192, 192..224]
                    }
                );
            }
        }
        for block in [0, 31, 33] {
            assert!(validate_block(64, block).is_err());
        }
        assert!(validate_block(63, 32).is_err());
        Ok(())
    }
}
