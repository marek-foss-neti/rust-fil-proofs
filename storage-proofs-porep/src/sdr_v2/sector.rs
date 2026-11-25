//! Sector-level entry points of SDR v2: settings, layer buffers, and encode/decode with
//! measurement boundaries.

use std::ops::{Deref, DerefMut};

use anyhow::{ensure, Context, Result};
use log::info;
use memmap2::{MmapMut, MmapOptions};
use serde::Serialize;

use super::decode::{decode_layers, DEFAULT_DECODE_BATCH, MAX_DECODE_BATCH};
use super::drg::DrgParentTable;
use super::encode::encode_layers;
use super::graph::SdrV2Graph;
use super::label::{LabelGraph, LayerEvent, SdrV2Params, NODE_BYTES};
use crate::zigzag::measurements::OperationGuard;

/// Layers of the report's parameters (Table 3).
pub const DEFAULT_LAYERS: usize = 11;
/// SHA-256 compressions of a key for which the report measures a label as costly as an SDR label
/// (Table 5).
pub const DEFAULT_KEY_BLOCKS: usize = 16;

/// Process settings of the SDR v2 benchmark paths, read from the environment.
///
/// * `FIL_PROOFS_SDRV2_LAYERS` (default 11) and `FIL_PROOFS_SDRV2_KEY_BLOCKS` (default 16) change
///   the replica;
/// * `FIL_PROOFS_SDRV2_DECODE_BATCH` (default 8) is the number of nodes prefetched together by the
///   decoder;
/// * `FIL_PROOFS_SDRV2_HUGE_PAGES` (default on) asks Linux for transparent huge pages on the layer
///   buffers;
/// * `FIL_PROOFS_SDRV2_DRG_PARENT_TABLE` (default on) keeps the DRG parents of the sector in
///   memory, computed once, instead of computing them in every layer. It needs 20 bytes per node
///   (20 GiB for a 32 GiB sector) and is skipped when the memory limit of the process has no room
///   for it next to the two sector buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SdrV2Settings {
    pub layers: usize,
    pub key_blocks: usize,
    pub decode_batch: usize,
    pub huge_pages: bool,
    pub drg_parent_table: bool,
}

fn env_value(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => Ok(Some(value.trim().to_string())),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("invalid {}", name)),
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env_value(name)? {
        None => Ok(default),
        Some(value) => value
            .parse::<usize>()
            .with_context(|| format!("{} must be a non-negative integer", name)),
    }
}

fn env_bool(name: &str, default: bool) -> Result<bool> {
    match env_value(name)? {
        None => Ok(default),
        Some(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => anyhow::bail!("{} must be a boolean", name),
        },
    }
}

impl SdrV2Settings {
    pub fn from_env() -> Result<Self> {
        let settings = Self {
            layers: env_usize("FIL_PROOFS_SDRV2_LAYERS", DEFAULT_LAYERS)?,
            key_blocks: env_usize("FIL_PROOFS_SDRV2_KEY_BLOCKS", DEFAULT_KEY_BLOCKS)?,
            decode_batch: env_usize("FIL_PROOFS_SDRV2_DECODE_BATCH", DEFAULT_DECODE_BATCH)?,
            huge_pages: env_bool("FIL_PROOFS_SDRV2_HUGE_PAGES", true)?,
            drg_parent_table: env_bool("FIL_PROOFS_SDRV2_DRG_PARENT_TABLE", true)?,
        };
        settings.params()?;
        ensure!(
            (1..=MAX_DECODE_BATCH).contains(&settings.decode_batch),
            "FIL_PROOFS_SDRV2_DECODE_BATCH must be between 1 and {}",
            MAX_DECODE_BATCH
        );
        Ok(settings)
    }

    /// The parameters that fix the bytes of a replica.
    pub fn params(&self) -> Result<SdrV2Params> {
        SdrV2Params::new(self.layers, self.key_blocks)
    }
}

/// One layer of a sector in anonymous memory.
pub struct LayerBuffer {
    map: MmapMut,
}

impl LayerBuffer {
    /// Allocates `bytes` zeroed bytes. With `huge_pages`, asks for transparent huge pages: a pass
    /// over a layer touches it at random, and with 4 KiB pages most accesses would also miss the
    /// TLB. The request is advisory; a refusal is logged and ignored.
    pub fn new(bytes: usize, huge_pages: bool) -> Result<Self> {
        ensure!(bytes > 0, "an SDR v2 layer buffer cannot be empty");
        #[allow(unused_mut)]
        let mut map = MmapOptions::new()
            .len(bytes)
            .map_anon()
            .with_context(|| format!("allocate an SDR v2 layer buffer of {} bytes", bytes))?;
        #[cfg(target_os = "linux")]
        if huge_pages {
            // SAFETY: the range is exactly the mapping created above.
            let status = unsafe {
                libc::madvise(
                    map.as_mut_ptr() as *mut libc::c_void,
                    bytes,
                    libc::MADV_HUGEPAGE,
                )
            };
            if status != 0 {
                log::warn!(target: "sdrv2", "huge pages not available for a layer buffer: {}", std::io::Error::last_os_error());
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = huge_pages;
        Ok(Self { map })
    }
}

impl Deref for LayerBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.map
    }
}

impl DerefMut for LayerBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.map
    }
}

/// Turns the layer boundaries of the encoder and decoder into the operation boundaries that the
/// benchmark telemetry records. Layers are reported zero-based, as for ZigZag.
#[derive(Default)]
struct OperationRecorder {
    open: Option<OperationGuard>,
}

impl OperationRecorder {
    fn record(&mut self, event: LayerEvent) {
        match event {
            LayerEvent::Start { operation, layer } => {
                self.open = Some(OperationGuard::enter(operation, Some(layer - 1)));
            }
            LayerEvent::End { .. } => {
                if let Some(guard) = self.open.take() {
                    guard.finish();
                }
            }
        }
    }
}

/// Memory left free next to the sector buffers and the parent table before the table is used.
const PARENT_TABLE_RESERVE_BYTES: u64 = 2 << 30;

fn read_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// The memory this process may use, if the system says so: the cgroup limit (v2, then v1) or the
/// physical memory, whichever is smaller. `None` where neither can be read.
fn memory_limit_bytes() -> Option<u64> {
    let cgroup = read_u64("/sys/fs/cgroup/memory.max")
        .or_else(|| read_u64("/sys/fs/cgroup/memory/memory.limit_in_bytes"));
    let physical = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let mut fields = line.strip_prefix("MemTotal:")?.split_whitespace();
                let kibibytes = fields.next()?.parse::<u64>().ok()?;
                Some(kibibytes.saturating_mul(1024))
            })
        });
    match (cgroup, physical) {
        (Some(cgroup), Some(physical)) => Some(cgroup.min(physical)),
        (limit, None) | (None, limit) => limit,
    }
}

/// Whether a table of `table_bytes` fits under `limit` next to two sector buffers.
fn parent_table_fits(table_bytes: u64, sector_bytes: u64, limit: Option<u64>) -> bool {
    match limit {
        None => true,
        Some(limit) => {
            table_bytes
                .saturating_add(sector_bytes.saturating_mul(2))
                .saturating_add(PARENT_TABLE_RESERVE_BYTES)
                <= limit
        }
    }
}

/// The DRG parent table of a sector, if it was requested and there is memory for it. A table that
/// cannot be used is not an error: the parents are then computed in every layer, as without it.
fn parent_table(graph: &SdrV2Graph, requested: bool) -> Option<DrgParentTable> {
    let nodes = graph.nodes();
    let table_bytes = DrgParentTable::bytes(nodes);
    if !requested {
        info!(target: "sdrv2", "phase=drg_parent_table used=false reason=disabled");
        return None;
    }
    let limit = memory_limit_bytes();
    if !parent_table_fits(table_bytes, (nodes * NODE_BYTES) as u64, limit) {
        info!(target: "sdrv2", "phase=drg_parent_table used=false reason=memory_limit table_bytes={} limit_bytes={:?}", table_bytes, limit);
        return None;
    }
    let started = std::time::Instant::now();
    let operation = OperationGuard::enter("sdrv2_drg_parent_table", None);
    match DrgParentTable::build(graph) {
        Ok(table) => {
            operation.finish();
            info!(target: "sdrv2", "phase=drg_parent_table used=true table_bytes={} elapsed_ms={}", table_bytes, started.elapsed().as_millis());
            Some(table)
        }
        Err(error) => {
            info!(target: "sdrv2", "phase=drg_parent_table used=false reason={:?}", error.to_string());
            None
        }
    }
}

fn ensure_sector(graph: &SdrV2Graph, bytes: usize) -> Result<()> {
    ensure!(
        bytes == graph.nodes() * NODE_BYTES,
        "sector of {} bytes does not match a graph of {} nodes",
        bytes,
        graph.nodes()
    );
    Ok(())
}

/// Seals fr32-padded sector data in place. `scratch` is a second buffer of the same length.
/// `drg_parent_table` asks for the in-memory table of DRG parents (see [`SdrV2Settings`]).
pub fn encode_sector(
    graph: &SdrV2Graph,
    params: &SdrV2Params,
    replica_id: &[u8; 32],
    data: &mut [u8],
    scratch: &mut [u8],
    drg_parent_table: bool,
) -> Result<()> {
    ensure_sector(graph, data.len())?;
    info!(target: "sdrv2", "phase=encode nodes={} layers={} key_blocks={}", graph.nodes(), params.layers(), params.key_blocks());
    let table = parent_table(graph, drg_parent_table);
    let mut operations = OperationRecorder::default();
    encode_layers(
        graph,
        params,
        replica_id,
        data,
        scratch,
        table.as_ref(),
        &mut |event| operations.record(event),
    )
}

/// Decodes a replica. Both buffers are overwritten; the returned slice is the one that holds the
/// fr32-padded sector data. `drg_parent_table` asks for the in-memory table of DRG parents (see
/// [`SdrV2Settings`]).
pub fn decode_sector<'a>(
    graph: &SdrV2Graph,
    params: &SdrV2Params,
    replica_id: &[u8; 32],
    replica: &'a mut [u8],
    scratch: &'a mut [u8],
    batch: usize,
    drg_parent_table: bool,
) -> Result<&'a mut [u8]> {
    ensure_sector(graph, replica.len())?;
    info!(target: "sdrv2", "phase=decode nodes={} layers={} key_blocks={} batch={}", graph.nodes(), params.layers(), params.key_blocks(), batch);
    let table = parent_table(graph, drg_parent_table);
    let mut operations = OperationRecorder::default();
    decode_layers(
        graph,
        params,
        replica_id,
        replica,
        scratch,
        batch,
        table.as_ref(),
        &mut |event| operations.record(event),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_table_needs_room_next_to_two_sector_buffers() {
        let gib = 1u64 << 30;
        // 32 GiB sector: 20 GiB table, 64 GiB of buffers, 2 GiB reserve.
        assert!(parent_table_fits(20 * gib, 32 * gib, Some(100 * gib)));
        assert!(parent_table_fits(20 * gib, 32 * gib, Some(86 * gib)));
        assert!(!parent_table_fits(20 * gib, 32 * gib, Some(86 * gib - 1)));
        assert!(!parent_table_fits(20 * gib, 32 * gib, Some(64 * gib)));
        assert!(parent_table_fits(20 * gib, 32 * gib, None));
        assert!(parent_table_fits(u64::MAX, u64::MAX, None));
        assert!(!parent_table_fits(u64::MAX, u64::MAX, Some(u64::MAX - 1)));
    }
}
