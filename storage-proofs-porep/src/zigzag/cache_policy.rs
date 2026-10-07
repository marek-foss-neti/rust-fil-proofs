//! Optional ZigZag-only residency experiments. Advice never changes persisted bytes.
//! Keep these off until a full sealing run, including C1 rereads, demonstrates a benefit.

#[cfg(target_os = "linux")]
use std::convert::TryFrom;
use std::env;
use std::fs::File;
use std::io;

use anyhow::{ensure, Context, Result};
use serde::Serialize;

pub const MAX_PARENT_BUFFER_NODES: usize = 262_144;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CachePolicy {
    /// Owned u32 records per encode feeder; zero retains the existing mmap reader.
    pub parent_buffer_nodes: usize,
    /// Advice is issued only after all encode readers have joined and unmapped.
    pub parent_cache_dontneed: bool,
    /// Sync each completed historical TreeR before issuing advice; keep it for C1.
    pub tree_r_dontneed: bool,
}

impl CachePolicy {
    pub fn from_env() -> Result<Self> {
        let nodes =
            env::var("FIL_PROOFS_ZIGZAG_PARENT_BUFFER_NODES").unwrap_or_else(|_| "0".into());
        let parent_buffer_nodes = nodes
            .parse::<usize>()
            .context("invalid ZigZag parent buffer nodes")?;
        ensure!(
            parent_buffer_nodes <= MAX_PARENT_BUFFER_NODES,
            "ZigZag parent buffer exceeds {MAX_PARENT_BUFFER_NODES} nodes per reader"
        );
        Ok(Self {
            parent_buffer_nodes,
            parent_cache_dontneed: boolean("FIL_PROOFS_ZIGZAG_PARENT_CACHE_DONTNEED")?,
            tree_r_dontneed: boolean("FIL_PROOFS_ZIGZAG_TREE_R_DONTNEED")?,
        })
    }
}

fn boolean(name: &str) -> Result<bool> {
    match env::var(name)
        .unwrap_or_else(|_| "0".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => anyhow::bail!("invalid boolean {name}"),
    }
}

/// Only full pages inside the requested range are advised. Length zero must never
/// reach posix_fadvise: it means *to EOF*, not an empty range. This is a hint, not a RAM cap.
pub(crate) fn page_range(offset: u64, len: u64, page: u64) -> Option<(u64, u64)> {
    if page == 0 {
        return None;
    }
    let end = offset.checked_add(len)? / page * page;
    let start = offset.checked_add(page - 1)? / page * page;
    (end > start).then_some((start, end.saturating_sub(start)))
}

pub(crate) fn discard_file(file: &File) {
    let result = file
        .metadata()
        .and_then(|metadata| discard_range(file, 0, metadata.len()));
    match result {
        Ok(bytes) => {
            log::info!(target: "zigzag_cache", "DONTNEED advised_bytes={bytes}; file retained; residency is not guaranteed")
        }
        Err(error) => {
            log::warn!(target: "zigzag_cache", "DONTNEED unavailable: {error}; retaining pages")
        }
    }
}

#[cfg(target_os = "linux")]
fn discard_range(file: &File, offset: u64, len: u64) -> io::Result<u64> {
    use std::os::fd::AsRawFd;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return Err(io::Error::last_os_error());
    }
    let Some((offset, len)) = page_range(offset, len, page as u64) else {
        return Ok(0);
    };
    let offset =
        libc::off_t::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    let length =
        libc::off_t::try_from(len).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // POSIX returns an errno directly; it does not set errno on failure.
    let rc =
        unsafe { libc::posix_fadvise(file.as_raw_fd(), offset, length, libc::POSIX_FADV_DONTNEED) };
    if rc == 0 {
        Ok(len)
    } else {
        Err(io::Error::from_raw_os_error(rc))
    }
}

#[cfg(not(target_os = "linux"))]
fn discard_range(_file: &File, _offset: u64, _len: u64) -> io::Result<u64> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advice_never_extends_beyond_completed_pages_or_uses_zero_length() {
        assert_eq!(page_range(0, 0, 4096), None);
        assert_eq!(page_range(1, 4095, 4096), None);
        assert_eq!(page_range(1, 12287, 4096), Some((4096, 8192)));
        assert_eq!(page_range(0, 8193, 4096), Some((0, 8192)));
        assert_eq!(page_range(u64::MAX, 1, 4096), None);
        assert_eq!(page_range(0, 4096, 0), None);
    }

    #[test]
    fn advice_preserves_contents_and_reopen() -> Result<()> {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new()?;
        let original = vec![0x5a; 16385];
        file.write_all(&original)?;
        file.as_file().sync_all()?;
        discard_file(file.as_file());
        assert_eq!(std::fs::read(file.path())?, original);
        Ok(())
    }
}
