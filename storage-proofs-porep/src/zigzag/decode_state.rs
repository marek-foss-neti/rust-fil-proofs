//! ZigZag-only decode storage policy. Admission is a conservative observation, not
//! an OS memory reservation; unrelated processes can still consume the headroom.
use std::sync::{Mutex, MutexGuard, TryLockError};

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use super::groth16_setup_policy::SetupResources;

const MIB: usize = 1 << 20;
static RAM_DECODE: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeMode {
    PingPong,
    Blockwise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeStorage {
    Auto,
    File,
    Ram,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DecodeStateConfig {
    pub mode: DecodeMode,
    /// Experimental block size, bounded for the entire call, not per worker.
    pub block_mib: usize,
    /// Used by adapters which own the private sector copy. Ping-pong stays file-backed.
    pub storage: DecodeStorage,
}

impl DecodeStateConfig {
    pub fn from_env() -> Result<Self> {
        Self::parse(
            &std::env::var("FIL_PROOFS_ZIGZAG_DECODE_MODE").unwrap_or_else(|_| "ping_pong".into()),
            &std::env::var("FIL_PROOFS_ZIGZAG_DECODE_BLOCK_MIB").unwrap_or_else(|_| "64".into()),
            &std::env::var("FIL_PROOFS_ZIGZAG_DECODE_STORAGE").unwrap_or_else(|_| "auto".into()),
        )
    }

    fn parse(mode: &str, block: &str, storage: &str) -> Result<Self> {
        let mode = match mode {
            "ping_pong" => DecodeMode::PingPong,
            "blockwise" => DecodeMode::Blockwise,
            _ => anyhow::bail!("ZigZag decode mode must be ping_pong or blockwise"),
        };
        let storage = match storage {
            "auto" => DecodeStorage::Auto,
            "file" => DecodeStorage::File,
            "ram" => DecodeStorage::Ram,
            _ => anyhow::bail!("ZigZag decode storage must be auto, file or ram"),
        };
        ensure!(
            mode == DecodeMode::Blockwise || storage != DecodeStorage::Ram,
            "RAM decode storage requires blockwise mode"
        );
        let block_mib = block.parse().context("invalid ZigZag decode block MiB")?;
        ensure!(
            (1..=256).contains(&block_mib),
            "ZigZag decode block must be 1..=256 MiB"
        );
        Ok(Self {
            mode,
            block_mib,
            storage,
        })
    }

    pub fn block_bytes(self, sector_bytes: usize) -> Result<usize> {
        ensure!(
            (1..=256).contains(&self.block_mib),
            "invalid ZigZag decode block size"
        );
        ensure!(
            sector_bytes > 0 && sector_bytes % 32 == 0,
            "invalid ZigZag decode sector size"
        );
        Ok((self.block_mib * MIB).min(sector_bytes))
    }
}

/// Held until the private RAM state is dropped. Concurrent calls in this process
/// cannot both pass the same admission snapshot before their pages are resident.
/// Auto callers fall back to files; explicit RAM callers report admission failure.
pub struct DecodeRamAdmission {
    _guard: MutexGuard<'static, ()>,
}

impl DecodeRamAdmission {
    pub fn acquire(
        sector_bytes: usize,
        block_bytes: usize,
        parent_buffer_bytes: usize,
        parent_table_reserve_bytes: usize,
    ) -> Result<Self> {
        let guard = match RAM_DECODE.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => anyhow::bail!("another ZigZag RAM decode is active"),
        };
        let resources = SetupResources::detect().context("detect ZigZag decode memory headroom")?;
        let available = resources
            .host_available_bytes
            .context("host memory headroom unavailable")?
            .min(resources.cgroup_available_bytes.unwrap_or(u64::MAX));
        let required = required_ram(
            sector_bytes,
            block_bytes,
            parent_buffer_bytes,
            parent_table_reserve_bytes,
            available,
        )?;
        log::info!(target: "zigzag_unseal", "zigzag_decode_state {}", serde_json::json!({
            "event": "ram_admission", "available_bytes": available, "required_bytes": required,
            "accepted": required <= available, "resources": resources,
            "sector_bytes": sector_bytes, "block_and_output_bytes": block_bytes,
            "parent_buffer_bytes": parent_buffer_bytes,
            "parent_table_reserve_bytes": parent_table_reserve_bytes,
            "headroom_reserve_bytes": (available / 10).max(1 << 30),
        }));
        ensure!(required <= available, "ZigZag RAM decode needs {required} bytes including reserve; only {available} available");
        Ok(Self { _guard: guard })
    }
}

fn required_ram(
    sector: usize,
    block: usize,
    parents: usize,
    table: usize,
    available: u64,
) -> Result<u64> {
    // Include all owned buffers, the reader-specific table allowance and
    // allocator/runtime/I/O headroom. Buffered parent tables do not need to stay
    // resident: their clean file pages can be reclaimed. This is not a page-cache cap.
    let mut required = (available / 10).max(1 << 30);
    for bytes in [sector, block, parents, table] {
        required = required
            .checked_add(bytes as u64)
            .context("ZigZag decode memory estimate overflow")?;
    }
    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_and_budget_are_bounded() -> Result<()> {
        for mib in ["16", "64", "256"] {
            let config = DecodeStateConfig::parse("blockwise", mib, "auto")?;
            assert_eq!(config.block_bytes(2048)?, 2048);
        }
        for args in [
            ("bad", "64", "auto"),
            ("blockwise", "0", "file"),
            ("blockwise", "257", "file"),
            ("blockwise", "64", "bad"),
            ("ping_pong", "64", "ram"),
        ] {
            assert!(DecodeStateConfig::parse(args.0, args.1, args.2).is_err());
        }
        let gib = 1 << 30;
        assert_eq!(
            required_ram(32 * gib, 64 * MIB, 64 * MIB, 56 * gib, 120 * gib as u64)?,
            100 * gib as u64 + (128 * MIB) as u64
        );
        assert!(required_ram(usize::MAX, 1, 1, 1, u64::MAX).is_err());
        Ok(())
    }

    #[test]
    fn buffered_admission_fits_full_graph_with_bounded_headroom() -> Result<()> {
        use crate::zigzag::{DecodeConfig, DecodeParentReader};
        let gib = 1usize << 30;
        // Include block, streamed FR32 output and CPU scratch.
        let available = 96_163_864_576;
        let buffers = 128 * MIB + 2 * MIB;
        for parent_reader in [
            DecodeParentReader::Buffered,
            DecodeParentReader::Mmap,
            DecodeParentReader::Legacy,
        ] {
            let config = DecodeConfig {
                parent_reader,
                parent_buffer_mib: 64,
            };
            let required = required_ram(
                32 * gib,
                buffers,
                64 * MIB,
                config.parent_table_reserve_bytes(88 * gib),
                available,
            )?;
            assert_eq!(
                required <= available,
                parent_reader == DecodeParentReader::Buffered
            );
            assert!(
                required_ram(
                    32 * gib,
                    buffers,
                    64 * MIB,
                    config.parent_table_reserve_bytes(88 * gib),
                    32 * gib as u64
                )? > 32 * gib as u64
            );
        }
        Ok(())
    }
}
