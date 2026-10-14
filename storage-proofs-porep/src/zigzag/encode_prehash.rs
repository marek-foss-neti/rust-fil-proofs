//! One-shot, bounded SHA preparation for ZigZag encode feeders only.
//! The public incremental hasher retains the original message length/padding.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub(crate) const SHA_BLOCK_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct EncodePrehash {
    /// Opt in until full encode/PC1 comparisons demonstrate a benefit.
    pub enabled: bool,
}

impl EncodePrehash {
    pub fn from_env() -> Result<Self> {
        match std::env::var("FIL_PROOFS_ZIGZAG_ENCODE_PREHASH") {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(error) => Err(error).context("invalid FIL_PROOFS_ZIGZAG_ENCODE_PREHASH"),
        }
    }

    fn parse(value: &str) -> Result<Self> {
        let enabled = match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => bail!("FIL_PROOFS_ZIGZAG_ENCODE_PREHASH must be a boolean"),
        };
        Ok(Self { enabled })
    }
}

pub(crate) struct PreparedHash {
    state: Sha256,
    pub(crate) processed_bytes: usize,
}

impl PreparedHash {
    /// Hash complete blocks of the contiguous stable prefix, never a ready
    /// parent after a hole. The remaining bytes stay in the existing slot.
    pub(crate) fn new(input: &[u8], stable_prefix_bytes: usize) -> Option<Self> {
        assert!(stable_prefix_bytes <= input.len());
        let processed_bytes = stable_prefix_bytes / SHA_BLOCK_BYTES * SHA_BLOCK_BYTES;
        if processed_bytes == 0 {
            return None;
        }
        let mut state = Sha256::new();
        state.update(&input[..processed_bytes]);
        Some(Self {
            state,
            processed_bytes,
        })
    }

    pub(crate) fn finish(mut self, input: &[u8]) -> [u8; 32] {
        self.state.update(&input[self.processed_bytes..]);
        self.state.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_is_explicit_and_strict() {
        assert!(!EncodePrehash::default().enabled);
        for value in ["1", "true", "YES", "on"] {
            assert!(EncodePrehash::parse(value).unwrap().enabled);
        }
        for value in ["0", "false", "NO", "off"] {
            assert!(!EncodePrehash::parse(value).unwrap().enabled);
        }
        for value in ["", "2", " true", "auto"] {
            assert!(EncodePrehash::parse(value).is_err());
        }
    }

    #[test]
    fn all_prefix_boundaries_preserve_length_padding_and_digest() {
        for len in [32, 64, 96, 448, 480, 512, 1056] {
            let input: Vec<u8> = (0..len).map(|i| (i * 37 + 19) as u8).collect();
            let reference: [u8; 32] = Sha256::digest(&input).into();
            for stable in 0..=len {
                let prepared = PreparedHash::new(&input, stable);
                if stable < SHA_BLOCK_BYTES {
                    assert!(prepared.is_none());
                } else {
                    let prepared = prepared.unwrap();
                    assert_eq!(prepared.processed_bytes, stable / 64 * 64);
                    assert_eq!(
                        prepared.finish(&input),
                        reference,
                        "len={len} stable={stable}"
                    );
                }
            }
        }
    }
}
