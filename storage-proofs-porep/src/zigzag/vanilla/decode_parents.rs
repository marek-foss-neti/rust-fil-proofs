//! Decode-only parent readers: one validated descriptor per orientation/session,
//! with either bounded positional reads or a stable read-only mapping.
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use filecoin_hashers::Hasher;
use memmap2::{Mmap, MmapOptions};
use serde::Serialize;
use storage_proofs_core::{
    drgraph::Graph,
    parameter_cache::{LockedFile, ParameterSetMetadata},
};

use super::{
    graph::ZigZagGraph,
    parent_table::{cache_path, ZigZagParentTable},
};

const MIB: usize = 1024 * 1024;
const MAX_BUFFER_NODES: usize = 32768;
const MIN_STRIPE_NODES: usize = 16384;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeParentReader {
    /// Reference implementation, retained as the default until full-sector A/B acceptance.
    Legacy,
    Buffered,
    Mmap,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DecodeConfig {
    pub parent_reader: DecodeParentReader,
    /// Total owned parent-record buffers per decode call, NOT per Rayon worker.
    /// File-backed page cache and sector buffers are additional cgroup-accounted memory.
    pub parent_buffer_mib: usize,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            parent_reader: DecodeParentReader::Legacy,
            parent_buffer_mib: 64,
        }
    }
}

impl DecodeConfig {
    /// Extra table residency reserved by the private-state RAM adapter. Buffered
    /// reads own only the bounded record buffers, counted separately by admission.
    /// Their clean page-cache pages are reclaimable, not a required resident table.
    /// Keep the conservative table allowance for the mapped/reference readers.
    pub fn parent_table_reserve_bytes(self, table_bytes: usize) -> usize {
        match self.parent_reader {
            DecodeParentReader::Buffered => 0,
            DecodeParentReader::Legacy | DecodeParentReader::Mmap => table_bytes,
        }
    }

    pub fn from_env() -> Result<Self> {
        Self::parse(
            &std::env::var("FIL_PROOFS_ZIGZAG_DECODE_PARENT_READER")
                .unwrap_or_else(|_| "legacy".into()),
            &std::env::var("FIL_PROOFS_ZIGZAG_DECODE_PARENT_BUFFER_MIB")
                .unwrap_or_else(|_| "64".into()),
        )
    }

    fn parse(reader: &str, mib: &str) -> Result<Self> {
        let parent_reader = match reader {
            "legacy" => DecodeParentReader::Legacy,
            "buffered" => DecodeParentReader::Buffered,
            "mmap" => DecodeParentReader::Mmap,
            _ => anyhow::bail!("ZigZag decode parent reader must be legacy, buffered or mmap"),
        };
        let parent_buffer_mib = mib
            .parse()
            .context("invalid ZigZag decode parent buffer MiB")?;
        ensure!(
            (1..=256).contains(&parent_buffer_mib),
            "ZigZag decode parent buffer must be 1..=256 MiB total"
        );
        Ok(Self {
            parent_reader,
            parent_buffer_mib,
        })
    }

    pub(super) fn layout(
        self,
        nodes: usize,
        degree: usize,
        threads: usize,
    ) -> Result<ReaderLayout> {
        ensure!(
            nodes > 0 && degree > 0 && threads > 0,
            "invalid ZigZag decode reader geometry"
        );
        ensure!(
            (1..=256).contains(&self.parent_buffer_mib),
            "invalid ZigZag decode parent buffer budget"
        );
        let entry_bytes = degree
            .checked_mul(4)
            .context("decode parent entry overflow")?;
        let budget = self.parent_buffer_mib * MIB;
        let chunks = 1 + (nodes - 1) / MIN_STRIPE_NODES;
        let mut readers = threads.min(chunks);
        if self.parent_reader == DecodeParentReader::Buffered {
            readers = readers.min(budget / entry_bytes);
        }
        ensure!(readers > 0, "decode parent budget cannot hold one record");
        let stripe_nodes = 1 + (nodes - 1) / readers;
        readers = 1 + (nodes - 1) / stripe_nodes;
        let buffer_nodes = if self.parent_reader == DecodeParentReader::Buffered {
            let bytes_per_row = readers
                .checked_mul(entry_bytes)
                .context("decode reader budget overflow")?;
            MAX_BUFFER_NODES
                .min(stripe_nodes)
                .min(budget / bytes_per_row)
        } else {
            0
        };
        Ok(ReaderLayout {
            readers,
            stripe_nodes,
            buffer_nodes,
            buffer_bytes: readers * buffer_nodes * entry_bytes,
        })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ReaderLayout {
    pub readers: usize,
    pub stripe_nodes: usize,
    pub buffer_nodes: usize,
    pub buffer_bytes: usize,
}

/// Scoped to one multilayer extraction. New extractions reopen/validate identities;
/// Rayon tasks never acquire the publication or verification locks.
pub(super) struct DecodeSession {
    pub config: DecodeConfig,
    pub cpu: crate::zigzag::decode_cpu::DecodeCpuConfig,
    sources: HashMap<PathBuf, DecodeParentSource>,
}

impl DecodeSession {
    pub fn new() -> Result<Self> {
        let mut session = Self::with_config(DecodeConfig::from_env()?);
        session.cpu = crate::zigzag::decode_cpu::DecodeCpuConfig::from_env()?;
        Ok(session)
    }

    pub fn with_config(config: DecodeConfig) -> Self {
        Self {
            config,
            cpu: Default::default(),
            sources: HashMap::new(),
        }
    }

    pub fn source<H, G>(&mut self, graph: &ZigZagGraph<H, G>) -> Result<&DecodeParentSource>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Send + Sync,
    {
        self.source_at_path(graph, cache_path(graph))
    }

    fn source_at_path<H, G>(
        &mut self,
        graph: &ZigZagGraph<H, G>,
        path: PathBuf,
    ) -> Result<&DecodeParentSource>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Send + Sync,
    {
        if !self.sources.contains_key(&path) {
            let source = DecodeParentSource::open(graph, &path, self.config.parent_reader)?;
            self.sources.insert(path.clone(), source);
        }
        Ok(&self.sources[&path])
    }
}

pub(super) struct DecodeParentSource {
    // Drop the mapping before releasing the descriptor's shared file lock.
    mapping: Option<Mmap>,
    file: LockedFile,
    nodes: usize,
    degree: usize,
    entry_bytes: usize,
}

impl DecodeParentSource {
    pub(super) fn open<H, G>(
        graph: &ZigZagGraph<H, G>,
        path: &Path,
        mode: DecodeParentReader,
    ) -> Result<Self>
    where
        H: Hasher,
        G: Graph<H> + ParameterSetMetadata + Send + Sync,
    {
        ensure!(
            mode != DecodeParentReader::Legacy,
            "legacy decode does not use shared sources"
        );
        let started = Instant::now();
        let (file, locks) = ZigZagParentTable::verified_file_at_path(graph, path)?;
        let entry_bytes = graph
            .degree()
            .checked_mul(4)
            .context("decode parent entry overflow")?;
        let bytes = graph
            .size()
            .checked_mul(entry_bytes)
            .context("decode parent table size overflow")?;
        let mapping = if mode == DecodeParentReader::Mmap {
            // The shared lock and descriptor stay alive with the mapping. Cache generation
            // publishes a new inode, never truncates this one. No pathname is reopened.
            Some(unsafe { MmapOptions::new().len(bytes).map(file.as_ref())? })
        } else {
            None
        };
        let page_cache_advice = if mode == DecodeParentReader::Buffered {
            match advise_streaming(file.as_ref()) {
                Ok(()) => "noreuse_requested",
                Err(error) => {
                    log::debug!(target: "zigzag_unseal", "decode parent NOREUSE unavailable: {error}");
                    "unavailable"
                }
            }
        } else {
            "unchanged"
        };
        log::info!(target: "zigzag_unseal", "zigzag_decode_parents {}", serde_json::json!({
            "event": "open", "reader": mode, "reversed": graph.reversed(),
            "nodes": graph.size(), "bytes": bytes, "source_opens": 1,
            "mappings": usize::from(mapping.is_some()), "remaps": 0,
            "page_cache_advice": page_cache_advice,
            "elapsed_ns": started.elapsed().as_nanos(), "locks": locks,
        }));
        Ok(Self {
            file,
            mapping,
            nodes: graph.size(),
            degree: graph.degree(),
            entry_bytes,
        })
    }

    pub fn reader(&self, buffer_nodes: usize) -> Result<DecodeParentCursor<'_>> {
        ensure!(
            self.mapping.is_some() || buffer_nodes > 0,
            "buffered decode needs a nonzero record buffer"
        );
        let bytes = if self.mapping.is_some() {
            0
        } else {
            buffer_nodes
                .checked_mul(self.entry_bytes)
                .context("decode parent buffer overflow")?
        };
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(bytes)
            .context("could not allocate decode parent buffer")?;
        buffer.resize(bytes, 0);
        Ok(DecodeParentCursor {
            source: self,
            buffer,
            offset: self.nodes,
            valid_nodes: 0,
            stats: ReadStats::default(),
        })
    }
}

// On Linux >= 6.3 NOREUSE avoids promoting this stream's pages during reclaim.
// Unlike DONTNEED it does not evict shared cache eagerly, so a warm table can
// still be reused by later layers/other readers when memory permits. Older
// kernels may ignore the hint; RAM admission does not depend on its success.
#[cfg(target_os = "linux")]
fn advise_streaming(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // Offset/length zero intentionally apply to the whole validated descriptor.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_NOREUSE) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(rc))
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_streaming(_file: &File) -> io::Result<()> {
    Err(io::ErrorKind::Unsupported.into())
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct ReadStats {
    pub read_calls: u64,
    pub read_bytes: u64,
    pub fills: u64,
}

impl ReadStats {
    pub fn merge(mut self, other: Self) -> Self {
        self.read_calls += other.read_calls;
        self.read_bytes += other.read_bytes;
        self.fills += other.fills;
        self
    }
}

pub(super) struct DecodeParentCursor<'a> {
    source: &'a DecodeParentSource,
    buffer: Vec<u8>,
    offset: usize,
    valid_nodes: usize,
    pub stats: ReadStats,
}

impl DecodeParentCursor<'_> {
    pub fn read_into(&mut self, node: usize, parents: &mut [u32]) -> Result<()> {
        let source = self.source;
        ensure!(node < source.nodes, "decode node outside parent table");
        ensure!(
            parents.len() >= source.degree,
            "decode parent output buffer too small"
        );
        let record = if let Some(mapping) = &source.mapping {
            let start = node
                .checked_mul(source.entry_bytes)
                .context("decode parent offset overflow")?;
            &mapping[start..start + source.entry_bytes]
        } else {
            if node < self.offset || node - self.offset >= self.valid_nodes {
                // Start at the requested node rather than a global window boundary:
                // neighbouring stripes need not read the unused prefix of a window.
                let count = (self.buffer.len() / source.entry_bytes).min(source.nodes - node);
                let bytes = count * source.entry_bytes;
                let offset = node
                    .checked_mul(source.entry_bytes)
                    .context("decode parent offset overflow")?;
                self.valid_nodes = 0; // A failed/short read must never expose stale or partial data.
                self.offset = source.nodes;
                read_exact_at(
                    source.file.as_ref(),
                    &mut self.buffer[..bytes],
                    offset as u64,
                    &mut self.stats,
                )
                .context("could not read ZigZag decode parent records")?;
                self.offset = node;
                self.valid_nodes = count;
                self.stats.fills += 1;
            }
            let start = (node - self.offset) * source.entry_bytes;
            &self.buffer[start..start + source.entry_bytes]
        };
        LittleEndian::read_u32_into(record, &mut parents[..source.degree]);
        Ok(())
    }
}

// Positional I/O leaves the validated descriptor's seek position untouched, so
// concurrent readers and the hashing code cannot race through a shared offset.
fn read_exact_at(
    file: &File,
    buffer: &mut [u8],
    offset: u64,
    stats: &mut ReadStats,
) -> io::Result<()> {
    read_exact_with(buffer, offset, stats, |buffer, offset| {
        #[cfg(unix)]
        {
            std::os::unix::fs::FileExt::read_at(file, buffer, offset)
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::FileExt::seek_read(file, buffer, offset)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (file, buffer, offset);
            Err(io::ErrorKind::Unsupported.into())
        }
    })
}

fn read_exact_with(
    mut buffer: &mut [u8],
    mut offset: u64,
    stats: &mut ReadStats,
    mut read: impl FnMut(&mut [u8], u64) -> io::Result<usize>,
) -> io::Result<()> {
    while !buffer.is_empty() {
        stats.read_calls += 1;
        match read(buffer, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                ensure_read_count(n, buffer.len())?;
                stats.read_bytes += n as u64;
                offset = offset
                    .checked_add(n as u64)
                    .ok_or(io::ErrorKind::InvalidInput)?;
                buffer = &mut buffer[n..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn ensure_read_count(count: usize, remaining: usize) -> io::Result<()> {
    if count > remaining {
        Err(io::ErrorKind::InvalidData.into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::graph::{ZigZagBucketGraph, EXP_DEGREE};
    use super::*;
    use filecoin_hashers::poseidon::PoseidonHasher;
    use std::fs;
    use std::io::{Seek, SeekFrom};
    use storage_proofs_core::{api_version::ApiVersion, drgraph::BASE_DEGREE};

    fn graph() -> Result<ZigZagBucketGraph<PoseidonHasher>> {
        ZigZagGraph::new_zigzag(
            None,
            4096,
            BASE_DEGREE,
            EXP_DEGREE,
            [61; 32],
            ApiVersion::V1_2_0,
        )
    }

    #[test]
    fn config_and_layout_bound_all_reader_buffers() -> Result<()> {
        for invalid in ["", "unknown", "BUFFERED"] {
            assert!(DecodeConfig::parse(invalid, "64").is_err());
        }
        for invalid in ["", "0", "257", "-1", "auto"] {
            assert!(DecodeConfig::parse("buffered", invalid).is_err());
        }
        assert_eq!(
            DecodeConfig::default().parent_reader,
            DecodeParentReader::Legacy
        );
        for mode in ["buffered", "mmap"] {
            for mib in [1, 64, 256] {
                let config = DecodeConfig::parse(mode, &mib.to_string())?;
                for nodes in [1, 17, 16384, 32769, 1 << 30] {
                    for workers in [1, 2, 4, 64, 96, 128, 4096] {
                        let layout = config.layout(nodes, 14, workers)?;
                        assert_eq!(layout.readers, 1 + (nodes - 1) / layout.stripe_nodes);
                        assert!(layout.readers <= workers);
                        assert!(layout.buffer_bytes <= mib * MIB);
                        assert!(layout.buffer_nodes <= MAX_BUFFER_NODES);
                        assert_eq!(layout.buffer_nodes > 0, mode == "buffered");
                    }
                }
            }
        }
        assert!(DecodeConfig::default().layout(0, 14, 1).is_err());
        assert!(DecodeConfig::default().layout(10, usize::MAX, 1).is_err());
        Ok(())
    }

    #[test]
    fn buffered_admission_does_not_reserve_the_whole_parent_table() -> Result<()> {
        for degree in [14, 22] {
            let table_bytes = (1usize << 30) * degree * 4;
            for reader in ["legacy", "mmap", "buffered"] {
                let config = DecodeConfig::parse(reader, "64")?;
                assert_eq!(config.parent_table_reserve_bytes(0), 0);
                assert_eq!(
                    config.parent_table_reserve_bytes(table_bytes),
                    if reader == "buffered" { 0 } else { table_bytes }
                );
                for workers in [1, 64, 128] {
                    let layout = config.layout(1 << 30, degree, workers)?;
                    assert!(layout.buffer_bytes <= 64 * MIB);
                }
            }
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn streaming_advice_preserves_file_contents_and_position() -> Result<()> {
        use std::io::{Read, Write};
        let mut file = tempfile::tempfile()?;
        file.write_all(b"parent table")?;
        file.seek(SeekFrom::Start(3))?;
        advise_streaming(&file)?;
        assert_eq!(file.stream_position()?, 3);
        let mut remaining = String::new();
        file.read_to_string(&mut remaining)?;
        assert_eq!(remaining, "ent table");
        Ok(())
    }

    #[test]
    fn positional_read_handles_interruptions_short_reads_and_eof() -> Result<()> {
        let input = b"parent records";
        let mut output = [0; 14];
        let mut stats = ReadStats::default();
        let mut interrupted = false;
        read_exact_with(&mut output, 7, &mut stats, |buffer, offset| {
            if !interrupted {
                interrupted = true;
                return Err(io::ErrorKind::Interrupted.into());
            }
            let start = offset as usize - 7;
            let n = buffer.len().min(3);
            buffer[..n].copy_from_slice(&input[start..start + n]);
            Ok(n)
        })?;
        assert_eq!(&output, input);
        assert_eq!((stats.read_calls, stats.read_bytes), (6, 14));
        let error = read_exact_with(&mut output, 0, &mut stats, |_, _| Ok(0)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        let error =
            read_exact_with(&mut output[..1], u64::MAX, &mut stats, |_, _| Ok(1)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }

    #[test]
    fn readers_match_both_orientations_tail_random_access_and_shared_descriptor() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let graph = graph()?;
        for graph in [graph.clone(), graph.zigzag()] {
            let path = directory
                .path()
                .join(format!("parents-{}", graph.reversed()));
            for mode in [DecodeParentReader::Buffered, DecodeParentReader::Mmap] {
                let source = DecodeParentSource::open(&graph, &path, mode)?;
                // Hashing leaves the seek cursor at EOF; readers must neither rely on nor change it.
                let mut file = source.file.as_ref();
                let cursor = file.seek(SeekFrom::Current(0))?;
                for buffer_nodes in [1, 17, 32768] {
                    let mut reader = source.reader(buffer_nodes)?;
                    let mut other = source.reader(buffer_nodes)?;
                    let mut actual = vec![0; graph.degree() + 1];
                    let mut expected = vec![0; graph.degree()];
                    for node in (0..graph.size()).chain([4095, 0, 17, 31, 4094, 1]) {
                        graph.parents(node, &mut expected)?;
                        actual[graph.degree()] = u32::MAX;
                        reader.read_into(node, &mut actual)?;
                        assert_eq!(&actual[..graph.degree()], &expected);
                        assert_eq!(actual[graph.degree()], u32::MAX);
                        other.read_into(node, &mut actual)?;
                        assert_eq!(&actual[..graph.degree()], &expected);
                    }
                    assert!(reader.read_into(graph.size(), &mut actual).is_err());
                    assert!(reader.read_into(0, &mut []).is_err());
                    assert!(reader.buffer.capacity() <= buffer_nodes * source.entry_bytes);
                }
                assert_eq!(file.seek(SeekFrom::Current(0))?, cursor);
            }
        }
        Ok(())
    }

    #[test]
    fn session_reuses_verified_inode_but_next_session_revalidates_replacement() -> Result<()> {
        let graph = graph()?;
        let directory = tempfile::tempdir()?;
        for mode in [DecodeParentReader::Buffered, DecodeParentReader::Mmap] {
            let path = directory.path().join(format!("parents-{mode:?}"));
            let config = DecodeConfig {
                parent_reader: mode,
                ..Default::default()
            };
            let mut session = DecodeSession::with_config(config);
            session.source_at_path(&graph, path.clone())?;
            let original = fs::read(&path)?;
            let replacement = directory.path().join("replacement");
            fs::write(&replacement, vec![0xff; original.len()])?;
            fs::rename(&replacement, &path)?;
            let mut expected = vec![0; graph.degree()];
            graph.parents(2049, &mut expected)?;
            let mut actual = vec![0; graph.degree()];
            session
                .source_at_path(&graph, path.clone())?
                .reader(17)?
                .read_into(2049, &mut actual)?;
            assert_eq!(
                actual, expected,
                "active session must keep the validated inode"
            );
            assert_eq!(session.sources.len(), 1);
            let mut next = DecodeSession::with_config(config);
            next.source_at_path(&graph, path.clone())?
                .reader(17)?
                .read_into(2049, &mut actual)?;
            assert_eq!(actual, expected);
            assert_eq!(
                fs::read(&path)?,
                original,
                "new session must repair corrupted replacement"
            );
        }
        Ok(())
    }

    #[test]
    fn failed_refill_cannot_expose_stale_or_partial_records_on_retry() -> Result<()> {
        let graph = graph()?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("parents");
        let source = DecodeParentSource::open(&graph, &path, DecodeParentReader::Buffered)?;
        let original = fs::read(&path)?;
        let mut reader = source.reader(17)?;
        let mut actual = vec![0; graph.degree()];
        reader.read_into(0, &mut actual)?;
        // Deliberately violate the publication protocol to exercise truncated positional I/O.
        fs::OpenOptions::new()
            .write(true)
            .open(&path)?
            .set_len(200 * source.entry_bytes as u64 + 1)?;
        assert!(reader.read_into(200, &mut actual).is_err());
        assert!(reader.read_into(200, &mut actual).is_err());
        assert_eq!(reader.valid_nodes, 0);
        fs::write(&path, original)?;
        reader.read_into(200, &mut actual)?;
        let mut expected = vec![0; graph.degree()];
        graph.parents(200, &mut expected)?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
