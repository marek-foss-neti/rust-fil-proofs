//! Optional ZigZag-only residency experiments. Advice never changes persisted bytes.
//! Keep these off until a full sealing run, including C1 rereads, demonstrates a benefit.

#[cfg(target_os = "linux")]
use std::convert::TryFrom;
use std::env;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{ensure, Context, Result};
use merkletree::store::StoreConfig;
use serde::Serialize;

use super::measurements::OperationGuard;

pub const MAX_PARENT_BUFFER_NODES: usize = 262_144;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CachePolicy {
    /// Owned u32 records per encode feeder; zero retains the existing mmap reader.
    pub parent_buffer_nodes: usize,
    /// Advice is issued only after all encode readers have joined and unmapped.
    pub parent_cache_dontneed: bool,
    /// Release TreeR pages only at C2 entry, after all C1 generation/serialization/validation.
    pub tree_r_dontneed: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct TreeRCacheRelease {
    pub advised_files: usize,
    pub advised_bytes: u64,
    pub skipped_files: usize,
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

    /// Optional local residency advice, never a dependency of C2 correctness.
    /// Call only after vanilla verification and any C1 artifact publication. Roots must
    /// come from the validated C1, not from the cache hint. The named stores are checked
    /// against those roots and the expected length before sync/advice. No TreeD, replica,
    /// parent cache or parameter files are touched; no directory scan or tree rebuild occurs.
    pub fn release_tree_r_cache(
        &self,
        cache_path: Option<&Path>,
        expected_roots: &[[u8; 32]],
        expected_file_bytes: u64,
    ) -> TreeRCacheRelease {
        let mut stats = TreeRCacheRelease::default();
        if !self.tree_r_dontneed {
            return stats;
        }
        let operation = OperationGuard::enter("groth16_tree_r_cache_release", None);
        if let Some(cache_path) = cache_path {
            for (layer, root) in expected_roots.iter().enumerate() {
                let path = StoreConfig::data_path(cache_path, &format!("zigzag-tree-r-{layer}"));
                let result = matching_tree_r(&path, root, expected_file_bytes).and_then(|file| {
                    // Legacy stores can still have dirty pages in the ordinary Rust API.
                    // Keep this work at C2 entry, never in PC1 or between C1 operations.
                    file.sync_data()?;
                    discard_range(&file, 0, expected_file_bytes)
                });
                match result {
                    Ok(bytes) => {
                        stats.advised_files += 1;
                        stats.advised_bytes += bytes;
                    }
                    Err(error) => {
                        stats.skipped_files += 1;
                        log::warn!(target: "zigzag_cache", "TreeR DONTNEED skipped at C2 layer={layer}: {error}");
                    }
                }
            }
        } else {
            stats.skipped_files = expected_roots.len();
        }
        log::info!(target: "zigzag_cache", "TreeR DONTNEED at C2 {}", serde_json::json!(stats));
        operation.finish();
        stats
    }
}

fn matching_tree_r(path: &Path, root: &[u8; 32], expected_file_bytes: u64) -> io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A portable C1 envelope may carry a stale path from another host. Never follow
        // a final-component symlink or block opening a FIFO/device instead of a store.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || expected_file_bytes < 32 || metadata.len() != expected_file_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "TreeR store type/size mismatch",
        ));
    }
    file.seek(SeekFrom::Start(expected_file_bytes - 32))?;
    let mut stored_root = [0; 32];
    file.read_exact(&mut stored_root)?;
    if stored_root != *root {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "TreeR root does not match validated C1",
        ));
    }
    Ok(file)
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

    fn store(cache: &Path, layer: usize, bytes: usize, root: [u8; 32]) -> io::Result<Vec<u8>> {
        let mut data = vec![0x5a; bytes];
        data[bytes - 32..].copy_from_slice(&root);
        std::fs::write(
            StoreConfig::data_path(cache, &format!("zigzag-tree-r-{layer}")),
            &data,
        )?;
        Ok(data)
    }

    #[test]
    fn tree_r_hint_rejects_wrong_root_size_and_missing_store() -> Result<()> {
        let cache = tempfile::tempdir()?;
        store(cache.path(), 0, 16384, [7; 32])?;
        let path = StoreConfig::data_path(cache.path(), "zigzag-tree-r-0");
        assert!(matching_tree_r(&path, &[7; 32], 16384).is_ok());
        assert!(matching_tree_r(&path, &[8; 32], 16384).is_err());
        assert!(matching_tree_r(&path, &[7; 32], 8192).is_err());
        assert!(matching_tree_r(&path, &[7; 32], 0).is_err());
        assert!(matching_tree_r(&cache.path().join("missing"), &[7; 32], 16384).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn tree_r_hint_rejects_symlink_directory_and_fifo_without_blocking() -> Result<()> {
        use std::os::unix::fs::symlink;
        let cache = tempfile::tempdir()?;
        store(cache.path(), 0, 16384, [7; 32])?;
        let path = StoreConfig::data_path(cache.path(), "zigzag-tree-r-0");
        let link = cache.path().join("link");
        symlink(&path, &link)?;
        assert!(matching_tree_r(&link, &[7; 32], 16384).is_err());
        assert!(matching_tree_r(cache.path(), &[7; 32], 16384).is_err());
        let fifo = cache.path().join("fifo");
        use std::os::unix::ffi::OsStrExt;
        let name = std::ffi::CString::new(fifo.as_os_str().as_bytes())?;
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        assert!(matching_tree_r(&fifo, &[7; 32], 16384).is_err());
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn tree_r_release_only_advises_matching_stores_and_preserves_every_file() -> Result<()> {
        let cache = tempfile::tempdir()?;
        let first = store(cache.path(), 0, 16384, [7; 32])?;
        let stale = store(cache.path(), 1, 16384, [8; 32])?;
        let unrelated = cache.path().join("replica.dat");
        std::fs::write(&unrelated, b"unrelated replica")?;
        let policy = CachePolicy {
            tree_r_dontneed: true,
            ..Default::default()
        };
        let stats = policy.release_tree_r_cache(Some(cache.path()), &[[7; 32]; 3], 16384);
        assert_eq!(stats.advised_files, 1);
        assert_eq!(stats.advised_bytes, 16384);
        assert_eq!(stats.skipped_files, 2);
        assert_eq!(
            std::fs::read(StoreConfig::data_path(cache.path(), "zigzag-tree-r-0"))?,
            first
        );
        assert_eq!(
            std::fs::read(StoreConfig::data_path(cache.path(), "zigzag-tree-r-1"))?,
            stale
        );
        assert_eq!(std::fs::read(&unrelated)?, b"unrelated replica");
        assert_eq!(
            policy
                .release_tree_r_cache(None, &[[7; 32]; 3], 16384)
                .skipped_files,
            3
        );
        let disabled =
            CachePolicy::default().release_tree_r_cache(Some(cache.path()), &[[7; 32]; 3], 16384);
        assert_eq!((disabled.advised_files, disabled.skipped_files), (0, 0));
        Ok(())
    }

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
