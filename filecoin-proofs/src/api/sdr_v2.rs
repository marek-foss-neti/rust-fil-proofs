//! Benchmark API for SDR v2 (see `storage_proofs_porep::sdr_v2`).
//!
//! Only what an unseal benchmark needs: sealing sector data in place to produce a replica, and
//! unsealing a replica to a writer. There are no commitments to the replica and no proofs yet, so
//! nothing here can be used to seal sectors for the chain.
//!
//! Unsealing mirrors the ZigZag raw-unseal path it is compared with: the sealed file is read into
//! memory, decoded, and the requested range is written with the same fr32 output writer.

use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use filecoin_hashers::poseidon::PoseidonHasher;
use log::info;
use storage_proofs_core::{api_version::ApiVersion, sector::SectorId, util::NODE_SIZE};
use storage_proofs_porep::{
    sdr_v2::{decode_sector, encode_sector, LayerBuffer, SdrV2Graph},
    stacked::generate_replica_id,
};

pub use storage_proofs_porep::sdr_v2::{SdrV2Params, SdrV2Settings};

use super::zigzag::{validate_zigzag_unseal_range, write_zigzag_unseal_range};
use crate::types::{Commitment, ProverId, Ticket, UnpaddedByteIndex, UnpaddedBytesAmount};

/// What identifies an SDR v2 replica format, plus the process settings of the decoder.
#[derive(Clone, Copy)]
pub struct SdrV2Config {
    /// Padded sector size in bytes.
    pub sector_size: u64,
    pub porep_id: [u8; 32],
    pub api_version: ApiVersion,
    pub params: SdrV2Params,
    /// Nodes prefetched together by the decoder.
    pub decode_batch: usize,
    /// Ask for transparent huge pages on the layer buffers.
    pub huge_pages: bool,
    /// Keep the DRG parents of the sector in memory instead of computing them in every layer
    /// (20 bytes per node; skipped when the memory limit has no room for it).
    pub drg_parent_table: bool,
}

impl SdrV2Config {
    pub fn new(
        sector_size: u64,
        porep_id: [u8; 32],
        api_version: ApiVersion,
        settings: &SdrV2Settings,
    ) -> Result<Self> {
        ensure!(
            sector_size >= NODE_SIZE as u64 && sector_size.is_multiple_of(NODE_SIZE as u64),
            "SDR v2 sector size {} is not a positive multiple of {} bytes",
            sector_size,
            NODE_SIZE
        );
        Ok(Self {
            sector_size,
            porep_id,
            api_version,
            params: settings.params()?,
            decode_batch: settings.decode_batch,
            huge_pages: settings.huge_pages,
            drg_parent_table: settings.drg_parent_table,
        })
    }

    /// Nodes of the sector.
    pub fn nodes(&self) -> usize {
        (self.sector_size / NODE_SIZE as u64) as usize
    }

    fn sector_bytes(&self) -> usize {
        self.nodes() * NODE_SIZE
    }

    fn graph(&self) -> Result<SdrV2Graph> {
        SdrV2Graph::new(self.nodes(), self.porep_id, self.api_version)
    }
}

/// The replica id of a sector, derived as for Stacked DRG.
pub fn sdr_v2_replica_id(
    config: &SdrV2Config,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    comm_d: Commitment,
) -> [u8; 32] {
    let domain = generate_replica_id::<PoseidonHasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d,
        &config.porep_id,
    );
    let mut replica_id = [0u8; 32];
    replica_id.copy_from_slice(AsRef::<[u8]>::as_ref(&domain));
    replica_id
}

/// Seals fr32-padded sector data in place: on return `data` is the replica.
///
/// Allocates one more sector-sized buffer, and the DRG parent table if the configuration asks
/// for it.
pub fn sdr_v2_seal_in_place(
    config: &SdrV2Config,
    replica_id: &[u8; 32],
    data: &mut [u8],
) -> Result<()> {
    ensure!(
        data.len() == config.sector_bytes(),
        "SDR v2 sector data has {} bytes, expected {}",
        data.len(),
        config.sector_bytes()
    );
    let graph = config.graph()?;
    let mut scratch = LayerBuffer::new(data.len(), config.huge_pages)?;
    encode_sector(
        &graph,
        &config.params,
        replica_id,
        data,
        &mut scratch,
        config.drg_parent_table,
    )
}

/// Seals a sector file in place: `sector` holds fr32-padded sector data and, on return, the
/// replica.
///
/// All layers are computed in anonymous memory and the file is written once at the end, so the
/// file is not rewritten layer by layer. Allocates two sector-sized buffers.
pub fn sdr_v2_seal_file(config: &SdrV2Config, replica_id: &[u8; 32], sector: &File) -> Result<()> {
    let sector_bytes = config.sector_bytes();
    let file_bytes = sector
        .metadata()
        .context("read SDR v2 sector metadata")?
        .len();
    ensure!(
        file_bytes == config.sector_size,
        "SDR v2 sector file has {} bytes, expected {}",
        file_bytes,
        config.sector_size
    );
    let mut data = LayerBuffer::new(sector_bytes, config.huge_pages)?;
    sector
        .read_exact_at(&mut data, 0)
        .context("read SDR v2 sector data into memory")?;
    sdr_v2_seal_in_place(config, replica_id, &mut data)?;
    let started = Instant::now();
    sector
        .write_all_at(&data, 0)
        .context("write SDR v2 replica")?;
    sector.sync_all().context("sync SDR v2 replica")?;
    info!(target: "sdrv2", "phase=seal_write elapsed_ms={}", started.elapsed().as_millis());
    Ok(())
}

/// Unseals a byte range of a sealed sector and writes the **unpadded** bytes to `output`.
///
/// The whole sector is decoded whatever the range: a label of one layer depends on labels spread
/// over the next one. `offset` and `num_bytes` are in unpadded bytes. The sealed file is only
/// read. Allocates two sector-sized buffers, and the DRG parent table if the configuration asks
/// for it.
pub fn sdr_v2_unseal_range<W: Write>(
    config: &SdrV2Config,
    sealed: &File,
    mut output: W,
    replica_id: &[u8; 32],
    offset: UnpaddedByteIndex,
    num_bytes: UnpaddedBytesAmount,
) -> Result<UnpaddedBytesAmount> {
    let sector_bytes = config.sector_bytes();
    let sealed_bytes = sealed
        .metadata()
        .context("read SDR v2 sealed sector metadata")?
        .len();
    ensure!(
        sealed_bytes == config.sector_size,
        "SDR v2 sealed sector has {} bytes, expected {}",
        sealed_bytes,
        config.sector_size
    );
    validate_zigzag_unseal_range(sector_bytes, offset, num_bytes)?;
    if num_bytes.0 == 0 {
        return Ok(num_bytes);
    }

    let started = Instant::now();
    let graph = config.graph()?;
    let mut replica = LayerBuffer::new(sector_bytes, config.huge_pages)?;
    let mut scratch = LayerBuffer::new(sector_bytes, config.huge_pages)?;
    info!(target: "sdrv2", "phase=unseal_setup elapsed_ms={}", started.elapsed().as_millis());

    let started = Instant::now();
    sealed
        .read_exact_at(&mut replica, 0)
        .context("read SDR v2 sealed sector into memory")?;
    info!(target: "sdrv2", "phase=unseal_load elapsed_ms={}", started.elapsed().as_millis());

    let started = Instant::now();
    let decoded = decode_sector(
        &graph,
        &config.params,
        replica_id,
        &mut replica,
        &mut scratch,
        config.decode_batch,
        config.drg_parent_table,
    )?;
    info!(target: "sdrv2", "phase=unseal_decode elapsed_ms={}", started.elapsed().as_millis());

    let started = Instant::now();
    let written = write_zigzag_unseal_range(decoded, &mut output, offset, num_bytes)?;
    info!(target: "sdrv2", "phase=unseal_output elapsed_ms={}", started.elapsed().as_millis());
    Ok(written)
}
