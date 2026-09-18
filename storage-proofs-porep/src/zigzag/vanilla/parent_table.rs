use std::collections::HashSet;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{BufReader, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

use anyhow::{ensure, Context};
use byteorder::{ByteOrder, LittleEndian};
use filecoin_hashers::Hasher;
use fs2::FileExt;
use lazy_static::lazy_static;
use log::{info, warn};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use storage_proofs_core::{
    drgraph::Graph,
    error::Result,
    parameter_cache::{LockedFile, ParameterSetMetadata, VERSION},
    settings::SETTINGS,
};

use crate::zigzag::vanilla::graph::ZigZagGraph;

/// u32 = 4 bytes.
const NODE_BYTES: usize = 4;
const PARENT_TABLE_MANIFEST_VERSION: u32 = 1;
const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

lazy_static! {
    static ref ZIGZAG_PARENT_TABLE_ACCESS_LOCK: Mutex<()> = Mutex::new(());
    static ref VERIFIED_PARENT_TABLES: Mutex<HashSet<VerifiedCache>> = Mutex::new(HashSet::new());
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ParentTableManifest {
    format_version: u32,
    nodes: u64,
    degree: u64,
    byte_len: u64,
    sha256: String,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct VerifiedCache {
    path: PathBuf,
    byte_len: u64,
    modified: SystemTime,
    device: u64,
    inode: u64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
    sha256: String,
}

#[cfg(unix)]
fn verified_cache_identity(
    path: &Path,
    metadata: &Metadata,
    sha256: &str,
) -> Option<VerifiedCache> {
    Some(VerifiedCache {
        path: path.to_path_buf(),
        byte_len: metadata.len(),
        modified: metadata.modified().ok()?,
        device: metadata.dev(),
        inode: metadata.ino(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
        sha256: sha256.to_owned(),
    })
}

// Without a reliable file identity and change timestamp, rehash on every open.
#[cfg(not(unix))]
fn verified_cache_identity(
    _path: &Path,
    _metadata: &Metadata,
    _sha256: &str,
) -> Option<VerifiedCache> {
    None
}

struct RemoveFileOnDrop(PathBuf);

impl Drop for RemoveFileOnDrop {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(Debug)]
pub(crate) struct ZigZagParentTable {
    degree: usize,
    nodes: usize,
    entry_bytes: usize,
    window_nodes: usize,
    cache: CacheData,
    path: PathBuf,
}

#[derive(Debug)]
struct CacheData {
    data: Mmap,
    offset: usize,
    len: usize,
    file: LockedFile,
}

impl CacheData {
    fn contains(&self, node: usize) -> bool {
        node >= self.offset && node < self.offset + self.len
    }

    fn shift(&mut self, offset: usize, len: usize, entry_bytes: usize, path: &Path) -> Result<()> {
        if self.offset == offset && self.len == len {
            return Ok(());
        }

        let byte_offset = offset
            .checked_mul(entry_bytes)
            .context("zigzag parent table mmap offset overflow")?;
        let byte_len = len
            .checked_mul(entry_bytes)
            .context("zigzag parent table mmap length overflow")?;

        self.data = unsafe {
            MmapOptions::new()
                .offset(byte_offset as u64)
                .len(byte_len)
                .map(self.file.as_ref())
                .with_context(|| {
                    format!("could not shift zigzag parent table={}", path.display())
                })?
        };
        self.offset = offset;
        self.len = len;

        Ok(())
    }

    fn open(
        offset: usize,
        len: usize,
        entry_bytes: usize,
        path: &Path,
        file: LockedFile,
    ) -> Result<Self> {
        let byte_offset = offset
            .checked_mul(entry_bytes)
            .context("zigzag parent table mmap offset overflow")?;
        let byte_len = len
            .checked_mul(entry_bytes)
            .context("zigzag parent table mmap length overflow")?;
        let data = unsafe {
            MmapOptions::new()
                .offset(byte_offset as u64)
                .len(byte_len)
                .map(file.as_ref())
                .with_context(|| format!("could not mmap zigzag parent table={}", path.display()))?
        };

        Ok(Self {
            data,
            offset,
            len,
            file,
        })
    }
}

impl ZigZagParentTable {
    pub(crate) fn new<H, G>(graph: &ZigZagGraph<H, G>) -> Result<Self>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Send + Sync,
    {
        let path = cache_path(graph);
        Self::new_at_path(graph, &path)
    }

    fn new_at_path<H, G>(graph: &ZigZagGraph<H, G>, path: &Path) -> Result<Self>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Send + Sync,
    {
        let _process_guard = ZIGZAG_PARENT_TABLE_ACCESS_LOCK
            .lock()
            .map_err(|_| anyhow::anyhow!("zigzag parent table process lock poisoned"))?;
        ensure_parent_directory(path)?;

        let lock_path = lock_path(path);
        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| {
                format!(
                    "could not open zigzag parent table lock={}",
                    lock_path.display()
                )
            })?;
        lock_file.lock_exclusive().with_context(|| {
            format!(
                "could not lock zigzag parent table lock={}",
                lock_path.display()
            )
        })?;

        let result = (|| {
            remove_stale_temp_files(path)?;
            let expected_size = expected_cache_size(graph)?;
            let validated =
                validate_cache(path, graph.size(), graph.degree(), expected_size as u64);
            let (_, file) = match validated {
                Ok(validated) => validated,
                Err(err) => {
                    warn!(
                        "zigzag parent table: cache missing or invalid, regenerating {}: {err:#}",
                        path.display()
                    );
                    forget_verified(path)?;
                    Self::generate(graph, path)?;
                    validate_cache(path, graph.size(), graph.degree(), expected_size as u64)?
                }
            };

            Self::open(graph, path, file)
        })();

        let unlock_result = FileExt::unlock(&lock_file).with_context(|| {
            format!(
                "could not unlock zigzag parent table lock={}",
                lock_path.display()
            )
        });

        match (result, unlock_result) {
            (Ok(table), Ok(())) => Ok(table),
            (Err(err), _) => Err(err),
            (Ok(_), Err(err)) => Err(err),
        }
    }

    pub(crate) fn read_into(&mut self, node: usize, parents: &mut [u32]) -> Result<()> {
        ensure!(node < self.nodes, "node {} outside parent table", node);
        ensure!(
            parents.len() >= self.degree,
            "parent buffer too small: {} < {}",
            parents.len(),
            self.degree
        );

        if !self.cache.contains(node) {
            let offset = (node / self.window_nodes) * self.window_nodes;
            let len = self.window_nodes.min(self.nodes - offset);
            self.cache
                .shift(offset, len, self.entry_bytes, &self.path)?;
        }

        let start = node
            .checked_sub(self.cache.offset)
            .context("zigzag parent table window offset underflow")?
            .checked_mul(self.entry_bytes)
            .context("zigzag parent table offset overflow")?;
        let end = start + self.entry_bytes;
        LittleEndian::read_u32_into(&self.cache.data[start..end], &mut parents[..self.degree]);

        Ok(())
    }

    fn open<H, G>(graph: &ZigZagGraph<H, G>, path: &Path, file: LockedFile) -> Result<Self>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Send + Sync,
    {
        let degree = graph.degree();
        let entry_bytes = entry_bytes(degree)?;
        let window_nodes = window_nodes(graph.size(), entry_bytes);
        let len = window_nodes.min(graph.size());

        info!(
            "zigzag parent table: opening {} with mmap window {} / {} nodes",
            path.display(),
            len,
            graph.size()
        );
        Ok(Self {
            degree,
            nodes: graph.size(),
            entry_bytes,
            window_nodes,
            cache: CacheData::open(0, len, entry_bytes, path, file)?,
            path: path.to_path_buf(),
        })
    }

    fn generate<H, G>(graph: &ZigZagGraph<H, G>, path: &Path) -> Result<()>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Send + Sync,
    {
        info!("zigzag parent table: generating {}", path.display());
        let cache_size = expected_cache_size(graph)?;
        let entry_bytes = entry_bytes(graph.degree())?;
        let window_nodes = window_nodes(graph.size(), entry_bytes);
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary_data_path = temporary_path(path, sequence, "data")?;
        let temporary_manifest_path = temporary_path(path, sequence, "manifest")?;
        let _data_cleanup = RemoveFileOnDrop(temporary_data_path.clone());
        let _manifest_cleanup = RemoveFileOnDrop(temporary_manifest_path.clone());

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temporary_data_path)
            .with_context(|| {
                format!(
                    "could not create temporary zigzag parent table={}",
                    temporary_data_path.display()
                )
            })?;
        file.set_len(cache_size as u64)
            .with_context(|| format!("failed to set parent table length: {cache_size}"))?;

        let mut hasher = Sha256::new();
        for offset in (0..graph.size()).step_by(window_nodes) {
            let len = window_nodes.min(graph.size() - offset);
            let byte_offset = offset
                .checked_mul(entry_bytes)
                .context("zigzag parent table generation offset overflow")?;
            let byte_len = len
                .checked_mul(entry_bytes)
                .context("zigzag parent table generation length overflow")?;
            let mut data = unsafe {
                MmapOptions::new()
                    .offset(byte_offset as u64)
                    .len(byte_len)
                    .map_mut(&file)
                    .with_context(|| {
                        format!(
                            "could not mmap temporary zigzag parent table={}",
                            temporary_data_path.display()
                        )
                    })?
            };

            data.par_chunks_mut(entry_bytes)
                .enumerate()
                .try_for_each_init(
                    || vec![0u32; graph.degree()],
                    |parents, (window_node, entry)| -> Result<()> {
                        let node = offset + window_node;
                        graph.parents(node, parents)?;
                        LittleEndian::write_u32_into(parents, entry);
                        Ok(())
                    },
                )?;

            data.flush()
                .context("failed to flush zigzag parent table window")?;
            hasher.update(&data[..]);
        }
        file.sync_all()
            .context("failed to sync temporary zigzag parent table")?;

        let manifest = ParentTableManifest {
            format_version: PARENT_TABLE_MANIFEST_VERSION,
            nodes: graph.size() as u64,
            degree: graph.degree() as u64,
            byte_len: cache_size as u64,
            sha256: hex::encode(hasher.finalize()),
        };
        write_manifest(&temporary_manifest_path, &manifest)?;

        fs::rename(&temporary_data_path, path).with_context(|| {
            format!(
                "could not publish zigzag parent table {} -> {}",
                temporary_data_path.display(),
                path.display()
            )
        })?;
        let final_manifest_path = manifest_path(path);
        fs::rename(&temporary_manifest_path, &final_manifest_path).with_context(|| {
            format!(
                "could not publish zigzag parent table manifest {} -> {}",
                temporary_manifest_path.display(),
                final_manifest_path.display()
            )
        })?;
        sync_parent_directory(path)?;
        remember_verified(path, &file, &manifest)?;
        info!("zigzag parent table: atomically published with verified manifest");

        Ok(())
    }
}

fn ensure_parent_directory(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .context("zigzag parent table path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "could not create zigzag parent table directory={}",
            parent.display()
        )
    })?;
    Ok(())
}

fn sibling_path(path: &Path, suffix: &str) -> Result<PathBuf> {
    let file_name = path
        .file_name()
        .context("zigzag parent table path has no file name")?;
    let mut sibling_name = file_name.to_os_string();
    sibling_name.push(suffix);
    Ok(path.with_file_name(sibling_name))
}

fn manifest_path(path: &Path) -> PathBuf {
    sibling_path(path, ".manifest.json").expect("parent table cache path must have a file name")
}

fn lock_path(path: &Path) -> PathBuf {
    sibling_path(path, ".lock").expect("parent table cache path must have a file name")
}

fn temporary_path(path: &Path, sequence: u64, kind: &str) -> Result<PathBuf> {
    let file_name = path
        .file_name()
        .context("zigzag parent table path has no file name")?
        .to_string_lossy();
    Ok(path.with_file_name(format!(
        ".{file_name}.tmp-{}-{sequence}.{kind}",
        std::process::id()
    )))
}

fn remove_stale_temp_files(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .context("zigzag parent table path has no parent directory")?;
    let file_name = path
        .file_name()
        .context("zigzag parent table path has no file name")?
        .to_string_lossy();
    let prefix = format!(".{file_name}.tmp-");

    for entry in fs::read_dir(parent).with_context(|| {
        format!(
            "could not scan zigzag parent table directory={}",
            parent.display()
        )
    })? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with(&prefix) {
            continue;
        }
        let file_type = entry.file_type()?;
        ensure!(
            file_type.is_file() || file_type.is_symlink(),
            "refusing to remove non-file stale parent table temporary path: {}",
            entry.path().display()
        );
        fs::remove_file(entry.path()).with_context(|| {
            format!(
                "could not remove stale zigzag parent table temporary file={}",
                entry.path().display()
            )
        })?;
    }

    Ok(())
}

fn write_manifest(path: &Path, manifest: &ParentTableManifest) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "could not create temporary zigzag parent table manifest={}",
                path.display()
            )
        })?;
    serde_json::to_writer(&mut file, manifest)
        .context("could not serialize zigzag parent table manifest")?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()
        .context("failed to sync temporary zigzag parent table manifest")?;
    Ok(())
}

fn validate_cache(
    path: &Path,
    expected_nodes: usize,
    expected_degree: usize,
    expected_size: u64,
) -> Result<(ParentTableManifest, LockedFile)> {
    let manifest_file = File::open(manifest_path(path)).with_context(|| {
        format!(
            "zigzag parent table completion manifest is missing for {}",
            path.display()
        )
    })?;
    let manifest: ParentTableManifest =
        serde_json::from_reader(manifest_file).with_context(|| {
            format!(
                "invalid zigzag parent table manifest for {}",
                path.display()
            )
        })?;

    ensure!(
        manifest.format_version == PARENT_TABLE_MANIFEST_VERSION,
        "unsupported zigzag parent table manifest version {} for {}",
        manifest.format_version,
        path.display()
    );
    ensure!(
        manifest.nodes == expected_nodes as u64,
        "zigzag parent table node count mismatch for {}: expected {}, got {}",
        path.display(),
        expected_nodes,
        manifest.nodes
    );
    ensure!(
        manifest.degree == expected_degree as u64,
        "zigzag parent table degree mismatch for {}: expected {}, got {}",
        path.display(),
        expected_degree,
        manifest.degree
    );
    ensure!(
        manifest.byte_len == expected_size,
        "zigzag parent table manifest length mismatch for {}: expected {}, got {}",
        path.display(),
        expected_size,
        manifest.byte_len
    );
    ensure!(
        manifest.sha256.len() == 64 && manifest.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid zigzag parent table digest in manifest for {}",
        path.display()
    );

    // Metadata, hashing and the eventual mmap must refer to the same open file, even if another
    // process replaces the path. The stable publication lock is held by the caller as well.
    let file = LockedFile::open_shared_read(path)
        .with_context(|| format!("could not open zigzag parent table={}", path.display()))?;
    let metadata = file.as_ref().metadata()?;
    ensure!(
        metadata.is_file(),
        "zigzag parent table is not a regular file: {}",
        path.display()
    );
    ensure!(
        metadata.len() == expected_size,
        "corrupted zigzag parent table: {}, expected {} bytes, got {} bytes",
        path.display(),
        expected_size,
        metadata.len()
    );

    let identity = verified_cache_identity(path, &metadata, &manifest.sha256);
    let already_verified = match &identity {
        Some(identity) => VERIFIED_PARENT_TABLES
            .lock()
            .map_err(|_| anyhow::anyhow!("zigzag parent table verification cache poisoned"))?
            .contains(identity),
        None => false,
    };
    if already_verified {
        return Ok((manifest, file));
    }

    let actual_digest = sha256_file(file.as_ref())
        .with_context(|| format!("could not hash zigzag parent table={}", path.display()))?;
    ensure!(
        actual_digest == manifest.sha256,
        "corrupted zigzag parent table digest for {}: expected {}, got {}",
        path.display(),
        manifest.sha256,
        actual_digest
    );
    ensure!(
        identity == verified_cache_identity(path, &file.as_ref().metadata()?, &manifest.sha256),
        "zigzag parent table changed during verification: {}",
        path.display()
    );
    if let Some(identity) = identity {
        VERIFIED_PARENT_TABLES
            .lock()
            .map_err(|_| anyhow::anyhow!("zigzag parent table verification cache poisoned"))?
            .insert(identity);
    }

    Ok((manifest, file))
}

fn sha256_file(file: &File) -> Result<String> {
    let mut reader = BufReader::with_capacity(HASH_BUFFER_BYTES, file);
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    let mut hasher = Sha256::new();

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

fn remember_verified(path: &Path, file: &File, manifest: &ParentTableManifest) -> Result<()> {
    if let Some(identity) = verified_cache_identity(path, &file.metadata()?, &manifest.sha256) {
        VERIFIED_PARENT_TABLES
            .lock()
            .map_err(|_| anyhow::anyhow!("zigzag parent table verification cache poisoned"))?
            .insert(identity);
    }
    Ok(())
}

fn forget_verified(path: &Path) -> Result<()> {
    VERIFIED_PARENT_TABLES
        .lock()
        .map_err(|_| anyhow::anyhow!("zigzag parent table verification cache poisoned"))?
        .retain(|entry| entry.path != path);
    Ok(())
}

fn sync_parent_directory(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .context("zigzag parent table path has no parent directory")?;
    File::open(parent)
        .with_context(|| format!("could not open parent cache directory={}", parent.display()))?
        .sync_all()
        .with_context(|| format!("could not sync parent cache directory={}", parent.display()))?;
    Ok(())
}

fn entry_bytes(degree: usize) -> Result<usize> {
    degree
        .checked_mul(NODE_BYTES)
        .context("zigzag parent table entry size overflow")
}

fn window_nodes(nodes: usize, entry_bytes: usize) -> usize {
    let requested = (SETTINGS.zigzag_parent_cache_size as usize)
        .max(1)
        .min(nodes);
    let alignment_nodes = mmap_alignment_nodes(entry_bytes);
    if requested < alignment_nodes || nodes < alignment_nodes {
        return nodes.min(alignment_nodes);
    }

    (requested / alignment_nodes).max(1) * alignment_nodes
}

fn mmap_alignment_nodes(entry_bytes: usize) -> usize {
    const MMAP_ALIGNMENT_BYTES: usize = 4096;
    MMAP_ALIGNMENT_BYTES / gcd(MMAP_ALIGNMENT_BYTES, entry_bytes)
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}

fn expected_cache_size<H, G>(graph: &ZigZagGraph<H, G>) -> Result<usize>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Send + Sync,
{
    graph
        .size()
        .checked_mul(entry_bytes(graph.degree())?)
        .context("zigzag parent table size overflow")
}

fn cache_path<H, G>(graph: &ZigZagGraph<H, G>) -> PathBuf
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Send + Sync,
{
    let mut hasher = Sha256::default();

    hasher.update(H::name());
    hasher.update(b"zigzag-parent-table-v1");
    hasher.update(graph.porep_id());
    hasher.update((graph.size() as u64).to_le_bytes());
    hasher.update((graph.base_degree() as u64).to_le_bytes());
    hasher.update((graph.expansion_degree() as u64).to_le_bytes());
    hasher.update([u8::from(graph.reversed())]);
    let api_version = graph.api_version().to_string();
    hasher.update(api_version.as_bytes());

    let digest = hasher.finalize();
    PathBuf::from(SETTINGS.parent_cache.clone()).join(format!(
        "v{}-zigzag-parent-{}.cache",
        VERSION,
        hex::encode(digest),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Seek, SeekFrom};

    use filecoin_hashers::poseidon::PoseidonHasher;
    use storage_proofs_core::{api_version::ApiVersion, drgraph::BASE_DEGREE};
    use tempfile::tempdir;

    use crate::zigzag::vanilla::graph::{ZigZagBucketGraph, EXP_DEGREE};

    #[test]
    fn parent_table_matches_graph_parents_both_orientations() {
        let nodes = 1 << 15;
        let porep_id = [42u8; 32];
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            porep_id,
            ApiVersion::V1_2_0,
        )
        .expect("failed to create zigzag graph");
        let cache_dir = tempdir().expect("failed to create parent table cache directory");

        for graph in [graph.clone(), graph.zigzag()] {
            let path = cache_dir
                .path()
                .join(format!("parents-{}.cache", graph.reversed()));
            let mut table = ZigZagParentTable::new_at_path(&graph, &path)
                .expect("failed to create parent table");
            let mut expected = vec![0u32; graph.degree()];
            let mut actual = vec![0u32; graph.degree()];

            for node in (0..nodes).chain((0..nodes).rev()) {
                graph
                    .parents(node, &mut expected)
                    .expect("failed to calculate parents");
                table
                    .read_into(node, &mut actual)
                    .expect("failed to read cached parents");

                assert_eq!(
                    actual,
                    expected,
                    "cached parents diverged at node={node} reversed={}",
                    graph.reversed()
                );
            }

            let parentless_node = if graph.reversed() {
                graph.size() - 1
            } else {
                0
            };
            table
                .read_into(parentless_node, &mut actual)
                .expect("failed to read first node");
            assert_eq!(actual[0], parentless_node as u32);
        }
    }

    #[test]
    fn exact_length_cache_without_completion_manifest_is_regenerated() {
        let nodes = 1 << 10;
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            [43u8; 32],
            ApiVersion::V1_2_0,
        )
        .expect("failed to create zigzag graph");
        let cache_dir = tempdir().expect("failed to create parent table cache directory");
        let path = cache_dir.path().join("interrupted.cache");
        let size = expected_cache_size(&graph).expect("failed to calculate cache size");

        File::create(&path)
            .and_then(|file| file.set_len(size as u64))
            .expect("failed to create incomplete parent table");
        let stale = cache_dir.path().join(".interrupted.cache.tmp-stale.data");
        File::create(&stale).expect("failed to create stale temporary file");

        let mut table = ZigZagParentTable::new_at_path(&graph, &path)
            .expect("incomplete cache was not regenerated");
        assert!(manifest_path(&path).is_file());
        assert!(!stale.exists(), "stale temporary file was not removed");

        let mut expected = vec![0u32; graph.degree()];
        let mut actual = vec![0u32; graph.degree()];
        graph
            .parents(nodes - 1, &mut expected)
            .expect("failed to calculate parents");
        table
            .read_into(nodes - 1, &mut actual)
            .expect("failed to read regenerated parent table");
        assert_eq!(actual, expected);
    }

    #[test]
    fn same_length_corruption_is_detected_and_repaired() {
        let nodes = 1 << 10;
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            [44u8; 32],
            ApiVersion::V1_2_0,
        )
        .expect("failed to create zigzag graph");
        let cache_dir = tempdir().expect("failed to create parent table cache directory");
        let path = cache_dir.path().join("corrupted.cache");
        let size = expected_cache_size(&graph).expect("failed to calculate cache size");

        drop(
            ZigZagParentTable::new_at_path(&graph, &path).expect("failed to generate parent table"),
        );
        let mut file = OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("failed to open parent table for corruption");
        file.seek(SeekFrom::Start(NODE_BYTES as u64))
            .expect("failed to seek parent table");
        file.write_all(&[0xff])
            .expect("failed to corrupt parent table");
        file.sync_all().expect("failed to sync corruption");
        let validation = validate_cache(&path, graph.size(), graph.degree(), size as u64)
            .expect_err("same-length corruption was accepted");
        assert!(format!("{validation:#}").contains("digest"));

        let mut repaired = ZigZagParentTable::new_at_path(&graph, &path)
            .expect("corrupted parent table was not repaired");
        let mut expected = vec![0u32; graph.degree()];
        let mut actual = vec![0u32; graph.degree()];
        graph
            .parents(nodes / 2, &mut expected)
            .expect("failed to calculate parents");
        repaired
            .read_into(nodes / 2, &mut actual)
            .expect("failed to read repaired parent table");
        assert_eq!(actual, expected);
    }

    fn preserved_mtime_corruption_is_detected(replace_file: bool) -> Result<()> {
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            1024,
            BASE_DEGREE,
            EXP_DEGREE,
            [45u8; 32],
            ApiVersion::V1_2_0,
        )
        .expect("failed to create zigzag graph");
        let cache_dir = tempdir().expect("failed to create cache directory");
        let path = cache_dir.path().join("preserved-mtime.cache");
        let size = expected_cache_size(&graph).expect("failed to calculate cache size");
        drop(ZigZagParentTable::new_at_path(&graph, &path).expect("failed to generate cache"));
        let modified = fs::metadata(&path)?.modified()?;

        let mut bytes = fs::read(&path)?;
        let node = 1;
        let offset = node * entry_bytes(graph.degree())?;
        bytes[offset..offset + NODE_BYTES].copy_from_slice(&u32::MAX.to_le_bytes());
        let corrupt_path = if replace_file {
            cache_dir.path().join("replacement.cache")
        } else {
            path.clone()
        };
        fs::write(&corrupt_path, bytes)?;
        let file = OpenOptions::new().write(true).open(&corrupt_path)?;
        file.set_modified(modified)?;
        file.sync_all()?;
        drop(file);
        if replace_file {
            fs::rename(&corrupt_path, &path)?;
        }
        assert_eq!(fs::metadata(&path)?.modified()?, modified);

        // Keep the real process memo populated: clearing it would hide the regression.
        let error = validate_cache(&path, graph.size(), graph.degree(), size as u64)
            .expect_err("corrupted cache with preserved mtime was accepted");
        assert!(format!("{error:#}").contains("digest"));
        let mut repaired = ZigZagParentTable::new_at_path(&graph, &path)
            .expect("corrupted cache was not repaired");
        let mut expected = vec![0; graph.degree()];
        let mut actual = vec![0; graph.degree()];
        graph.parents(node, &mut expected)?;
        repaired.read_into(node, &mut actual)?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn replaced_cache_with_preserved_mtime_is_revalidated() -> Result<()> {
        preserved_mtime_corruption_is_detected(true)
    }

    #[test]
    fn in_place_corruption_with_preserved_mtime_is_revalidated() -> Result<()> {
        preserved_mtime_corruption_is_detected(false)
    }

    #[test]
    fn mmap_uses_validated_file_after_path_is_replaced() -> Result<()> {
        let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            4096,
            BASE_DEGREE,
            EXP_DEGREE,
            [46u8; 32],
            ApiVersion::V1_2_0,
        )
        .expect("failed to create zigzag graph");
        let cache_dir = tempdir()?;
        let path = cache_dir.path().join("replaced-after-validation.cache");
        let size = expected_cache_size(&graph)?;
        drop(ZigZagParentTable::new_at_path(&graph, &path)?);
        let (_, verified_file) = validate_cache(&path, graph.size(), graph.degree(), size as u64)
            .expect("cache should validate before replacement");

        let replacement = cache_dir.path().join("replacement.cache");
        fs::write(&replacement, vec![0xff; size])?;
        fs::rename(&replacement, &path)?;
        let mut table = ZigZagParentTable::open(&graph, &path, verified_file)?;
        let mut expected = vec![0; graph.degree()];
        let mut actual = vec![0; graph.degree()];
        // Exercise the initial mmap and subsequent window shifts on the verified descriptor.
        for node in [1, graph.size() - 1, 1] {
            graph.parents(node, &mut expected)?;
            table.read_into(node, &mut actual)?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }
}
