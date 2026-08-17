use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, ensure, Context};
use byteorder::{ByteOrder, LittleEndian};
use filecoin_hashers::Hasher;
use lazy_static::lazy_static;
use log::info;
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use sha2::{Digest, Sha256};
use storage_proofs_core::{
    drgraph::Graph,
    error::Result,
    parameter_cache::{with_exclusive_lock, LockedFile, ParameterSetMetadata, VERSION},
    settings::SETTINGS,
};

use crate::zigzag::vanilla::graph::ZigZagGraph;

/// u32 = 4 bytes.
const NODE_BYTES: usize = 4;

lazy_static! {
    static ref ZIGZAG_PARENT_TABLE_ACCESS_LOCK: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
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

    fn open(offset: usize, len: usize, entry_bytes: usize, path: &Path) -> Result<Self> {
        let byte_offset = offset
            .checked_mul(entry_bytes)
            .context("zigzag parent table mmap offset overflow")?;
        let byte_len = len
            .checked_mul(entry_bytes)
            .context("zigzag parent table mmap length overflow")?;
        let file = LockedFile::open_shared_read(path)
            .with_context(|| format!("could not open zigzag parent table={}", path.display()))?;
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
        let generation_key = path.display().to_string();
        let mut generated = ZIGZAG_PARENT_TABLE_ACCESS_LOCK
            .lock()
            .expect("zigzag parent table generation lock failed");

        if path.exists() {
            if generated.get(&generation_key).is_none() {
                generated.insert(generation_key);
            }
            Self::open(graph, &path)
        } else {
            match Self::generate(graph, &path) {
                Ok(()) => {
                    generated.insert(generation_key);
                    Self::open(graph, &path)
                }
                Err(err) => match err.downcast::<io::Error>() {
                    Ok(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        Self::open(graph, &path)
                    }
                    Ok(error) => Err(error.into()),
                    Err(error) => Err(error),
                },
            }
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

    fn open<H, G>(graph: &ZigZagGraph<H, G>, path: &Path) -> Result<Self>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Send + Sync,
    {
        let expected_size = expected_cache_size(graph)?;
        let file = LockedFile::open_shared_read(path)
            .with_context(|| format!("could not open zigzag parent table={}", path.display()))?;

        let actual_len = file.as_ref().metadata()?.len();
        if actual_len != expected_size as u64 {
            bail!(
                "corrupted zigzag parent table: {}, expected {} bytes, got {} bytes",
                path.display(),
                expected_size,
                actual_len
            );
        }

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
            cache: CacheData::open(0, len, entry_bytes, path)?,
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

        with_exclusive_lock(path, |file| {
            file.as_ref()
                .set_len(cache_size as u64)
                .with_context(|| format!("failed to set length: {}", cache_size))?;

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
                        .map_mut(file.as_ref())
                        .with_context(|| {
                            format!("could not mmap zigzag parent table={}", path.display())
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
            }
            info!("zigzag parent table: written to disk");

            Ok(())
        })
    }
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

    use filecoin_hashers::poseidon::PoseidonHasher;
    use storage_proofs_core::{api_version::ApiVersion, drgraph::BASE_DEGREE};

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

        for graph in [graph.clone(), graph.zigzag()] {
            let mut table = ZigZagParentTable::new(&graph).expect("failed to create parent table");
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
}
