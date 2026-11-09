//! ZigZag-only FR32 output. One bounded batch is joined before the caller's
//! writer runs; there is no producer queue which can grow behind a slow writer.
use std::{
    convert::TryFrom,
    io::{self, Write},
    ops::Range,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use serde::Serialize;

const PADDED: usize = 128;
const RAW: usize = 127;
const GROUPS_PER_TASK: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZigZagUnsealOutputMode {
    Reference,
    Parallel,
    Stream,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ZigZagUnsealOutputConfig {
    pub mode: ZigZagUnsealOutputMode,
    /// Padded bytes per batch, in MiB. The owned unpadded buffer is 127/128 of this.
    pub chunk_mib: usize,
}

impl ZigZagUnsealOutputConfig {
    pub fn from_env() -> Result<Self> {
        Self::parse(
            &std::env::var("FIL_PROOFS_ZIGZAG_UNSEAL_OUTPUT")
                .unwrap_or_else(|_| "reference".into()),
            &std::env::var("FIL_PROOFS_ZIGZAG_UNSEAL_OUTPUT_CHUNK_MIB")
                // A 64 MiB final-layer chunk can use 128 decode stripes;
                // a small FR32 batch would also restrict decoder concurrency.
                .unwrap_or_else(|_| "64".into()),
        )
    }

    fn parse(mode: &str, chunk: &str) -> Result<Self> {
        let mode = match mode {
            "reference" => ZigZagUnsealOutputMode::Reference,
            "parallel" => ZigZagUnsealOutputMode::Parallel,
            "stream" => ZigZagUnsealOutputMode::Stream,
            _ => anyhow::bail!("ZigZag unseal output must be reference, parallel or stream"),
        };
        let chunk_mib = chunk
            .parse()
            .context("invalid ZigZag unseal output chunk MiB")?;
        let config = Self { mode, chunk_mib };
        config.chunk_bytes()?;
        Ok(config)
    }

    pub fn chunk_bytes(self) -> Result<usize> {
        ensure!(
            (1..=64).contains(&self.chunk_mib),
            "ZigZag unseal output chunk must be 1..=64 MiB"
        );
        Ok(self.chunk_mib * (1 << 20))
    }

    pub fn buffer_bytes(self, sector_bytes: usize) -> Result<usize> {
        let chunk = self.chunk_bytes()?.min(sector_bytes);
        Ok(if self.mode == ZigZagUnsealOutputMode::Reference {
            0
        } else {
            chunk / PADDED * RAW
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct OutputRange {
    pub padded: Range<usize>,
    skip: usize,
    bytes: usize,
}

impl OutputRange {
    pub fn new(padded_bytes: usize, offset: u64, count: u64) -> Result<Self> {
        ensure!(
            padded_bytes.is_multiple_of(PADDED),
            "unseal output needs complete FR32 groups"
        );
        let capacity = (padded_bytes / PADDED)
            .checked_mul(RAW)
            .context("unseal capacity overflow")?;
        let end = offset.checked_add(count).context("unseal range overflow")?;
        ensure!(end <= capacity as u64, "unseal range exceeds sector size");
        let offset = usize::try_from(offset).context("unseal offset exceeds address space")?;
        let end = usize::try_from(end).context("unseal end exceeds address space")?;
        let start = (offset / RAW)
            .checked_mul(PADDED)
            .context("unseal padded offset overflow")?;
        let padded_end = if count == 0 {
            start
        } else {
            end.div_ceil(RAW)
                .checked_mul(PADDED)
                .context("unseal padded end overflow")?
        };
        Ok(Self {
            padded: start..padded_end,
            skip: offset % RAW,
            bytes: end - offset,
        })
    }
}

fn check_cancelled(cancelled: Option<&AtomicBool>) -> io::Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        return Err(io::Error::other("ZigZag unseal output cancelled"));
    }
    Ok(())
}

/// Concatenate four little-endian 254-bit payloads, dropping their two padding
/// bits. The three boundary bytes splice adjacent fields; no field conversion
/// or heap allocation is needed. Padding bits are ignored, like fr32's reader.
fn unpad_group(input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), PADDED);
    debug_assert_eq!(output.len(), RAW);
    output[..31].copy_from_slice(&input[..31]);
    output[31] = (input[31] & 0x3f) | (input[32] << 6);
    for i in 32..63 {
        output[i] = (input[i] >> 2) | (input[i + 1] << 6);
    }
    output[63] = ((input[63] >> 2) & 0x0f) | (input[64] << 4);
    for i in 64..95 {
        output[i] = (input[i] >> 4) | (input[i + 1] << 4);
    }
    output[95] = ((input[95] >> 4) & 0x03) | (input[96] << 2);
    for i in 96..127 {
        output[i] = (input[i] >> 6) | (input[i + 1] << 2);
    }
}

#[derive(Default, Serialize)]
struct OutputStats {
    completed: bool,
    chunks: usize,
    written_bytes: usize,
    fr32_ns: u128,
    writer_ns: u128,
}

pub(super) struct ParallelOutput<'a, W: Write> {
    writer: &'a mut W,
    range: OutputRange,
    next_padded: usize,
    buffer: Vec<u8>,
    cancelled: Option<&'a AtomicBool>,
    stats: OutputStats,
    streamed: bool,
}

impl<'a, W: Write> ParallelOutput<'a, W> {
    pub fn new(
        writer: &'a mut W,
        range: OutputRange,
        chunk_bytes: usize,
        cancelled: Option<&'a AtomicBool>,
        streamed: bool,
    ) -> Result<Self> {
        ensure!(
            chunk_bytes > 0 && chunk_bytes.is_multiple_of(PADDED),
            "invalid unseal output batch size"
        );
        check_cancelled(cancelled)?;
        let bytes = chunk_bytes.min(range.padded.len()) / PADDED * RAW;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(bytes)
            .context("allocate bounded ZigZag FR32 output")?;
        buffer.resize(bytes, 0);
        Ok(Self {
            writer,
            next_padded: range.padded.start,
            range,
            buffer,
            cancelled,
            stats: OutputStats::default(),
            streamed,
        })
    }

    /// Chunks must arrive in ascending padded offset order and contain whole groups.
    pub fn write_chunk(&mut self, padded_offset: usize, input: &[u8]) -> Result<()> {
        ensure!(
            padded_offset == self.next_padded,
            "out-of-order ZigZag output chunk"
        );
        ensure!(
            !input.is_empty() && input.len().is_multiple_of(PADDED),
            "incomplete FR32 output group"
        );
        let end = padded_offset
            .checked_add(input.len())
            .context("output chunk overflow")?;
        ensure!(end <= self.range.padded.end, "output chunk exceeds range");
        let bytes = input.len() / PADDED * RAW;
        ensure!(
            bytes <= self.buffer.len(),
            "output chunk exceeds bounded buffer"
        );
        check_cancelled(self.cancelled)?;
        let started = Instant::now();
        let cancelled = self.cancelled;
        let result: io::Result<()> = self.buffer[..bytes]
            .par_chunks_mut(RAW * GROUPS_PER_TASK)
            .zip(input.par_chunks(PADDED * GROUPS_PER_TASK))
            .try_for_each(|(output, input)| {
                check_cancelled(cancelled)?;
                for (input, output) in input.chunks_exact(PADDED).zip(output.chunks_exact_mut(RAW))
                {
                    unpad_group(input, output);
                }
                Ok(())
            });
        self.stats.fr32_ns += started.elapsed().as_nanos();
        result?;
        let skip = if padded_offset == self.range.padded.start {
            self.range.skip
        } else {
            0
        };
        let take = (bytes - skip).min(self.range.bytes - self.stats.written_bytes);
        let started = Instant::now();
        let result = write_checked(
            self.writer,
            &self.buffer[skip..skip + take],
            self.cancelled,
            &mut self.stats.written_bytes,
        );
        self.stats.writer_ns += started.elapsed().as_nanos();
        result?;
        self.next_padded = end;
        self.stats.chunks += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<usize> {
        check_cancelled(self.cancelled)?;
        ensure!(
            self.next_padded == self.range.padded.end
                && self.stats.written_bytes == self.range.bytes,
            "incomplete ZigZag unseal output"
        );
        self.stats.completed = true;
        Ok(self.stats.written_bytes)
    }
}

impl<W: Write> Drop for ParallelOutput<'_, W> {
    fn drop(&mut self) {
        log::info!(target: "zigzag_unseal", "zigzag_unseal_output {}", serde_json::json!({
            "event": "output", "mode": if self.streamed { "stream" } else { "parallel" },
            "output_buffer_bytes": self.buffer.len(), "max_inflight_chunks": 1,
            "stats": self.stats,
        }));
    }
}

fn write_checked<W: Write>(
    writer: &mut W,
    mut data: &[u8],
    cancelled: Option<&AtomicBool>,
    written: &mut usize,
) -> io::Result<()> {
    while !data.is_empty() {
        check_cancelled(cancelled)?;
        match writer.write(data) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) if n <= data.len() => {
                *written += n;
                data = &data[n..];
            }
            Ok(_) => return Err(io::Error::other("writer returned an invalid byte count")),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub(super) fn write_range<W: Write>(
    decoded: &[u8],
    output: &mut W,
    offset: u64,
    count: u64,
    config: ZigZagUnsealOutputConfig,
    cancelled: Option<&AtomicBool>,
) -> Result<usize> {
    let range = OutputRange::new(decoded.len(), offset, count)?;
    if count == 0 {
        return Ok(0);
    }
    if config.mode == ZigZagUnsealOutputMode::Reference {
        let mut timed = TimedWriter {
            writer: output,
            cancelled,
            writer_ns: 0,
            written: 0,
        };
        let started = Instant::now();
        let result = fr32::write_unpadded(
            decoded,
            &mut timed,
            usize::try_from(offset)?,
            usize::try_from(count)?,
        );
        log::info!(target: "zigzag_unseal", "zigzag_unseal_output {}", serde_json::json!({
            "event": "output", "mode": "reference", "stats": {
                "completed": result.is_ok(), "written_bytes": timed.written,
                "fr32_ns": started.elapsed().as_nanos().saturating_sub(timed.writer_ns),
                "writer_ns": timed.writer_ns,
            },
        }));
        return result.context("write_unpadded failed");
    }
    let chunk = config.chunk_bytes()?;
    let mut writer = ParallelOutput::new(output, range.clone(), chunk, cancelled, false)?;
    for start in (range.padded.start..range.padded.end).step_by(chunk) {
        let end = start + chunk.min(range.padded.end - start);
        writer.write_chunk(start, &decoded[start..end])?;
    }
    writer.finish()
}

struct TimedWriter<'a, W> {
    writer: &'a mut W,
    cancelled: Option<&'a AtomicBool>,
    writer_ns: u128,
    written: usize,
}
impl<W: Write> Write for TimedWriter<'_, W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        check_cancelled(self.cancelled)?;
        let started = Instant::now();
        let result = self.writer.write(data);
        self.writer_ns += started.elapsed().as_nanos();
        if let Ok(n) = result {
            self.written += n;
        }
        result
    }
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

#[cfg(test)]
#[path = "unseal_output/tests.rs"]
mod tests;
