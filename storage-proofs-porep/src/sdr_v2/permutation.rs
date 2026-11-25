//! The component permutation between two SDR v2 layers.
//!
//! This is the Feistel permutation of `storage_proofs_core::crypto::feistel` on `8 * nodes`
//! indices, the one that `StackedGraph` uses to pick expander parents. `StackedGraph` keeps only
//! `image / 8` (the parent node); SDR v2 also needs `image % 8` (which component of that parent),
//! so it uses the whole image.
//!
//! The round function hashes one half of the index with the round key (Blake2b). Its value depends
//! only on that half and on the key, so it is tabulated: `3 * 2^half_bits` `u32` values, 1.5 MiB
//! for a 32 GiB sector. The tests compare the table with `feistel::permute`.

use blake2b_simd::blake2b;

/// Rounds of the Feistel network (`feistel::FEISTEL_ROUNDS`).
const FEISTEL_ROUNDS: usize = 3;

/// A permutation of `0..domain`, with its inverse.
pub struct WordPermutation {
    rounds: Box<[u32]>,
    domain: u64,
    right_mask: u64,
    half_bits: u32,
}

impl WordPermutation {
    /// `keys` are the Feistel keys of the stacked graph; `domain` is `8 * nodes`.
    pub fn new(keys: &[u64; 4], domain: u64) -> Self {
        // `feistel::precompute`: the smallest even number of bits that can hold `domain`.
        let mut next_pow4: u64 = 4;
        let mut half_bits: u32 = 1;
        while next_pow4 < domain {
            next_pow4 *= 4;
            half_bits += 1;
        }
        // A graph has at most 2^32 nodes with 8 slots each, so a half has at most 18 bits.
        assert!(half_bits <= 18, "Feistel table too large");
        let right_mask = (1u64 << half_bits) - 1;
        let halves = 1usize << half_bits;

        let mut rounds = vec![0u32; FEISTEL_ROUNDS * halves];
        for (round, key) in keys.iter().take(FEISTEL_ROUNDS).enumerate() {
            // The same input and output bytes as the round function in `feistel.rs`.
            let mut data = [0u8; 16];
            data[8..].copy_from_slice(&key.to_be_bytes());
            for half in 0..halves {
                data[..8].copy_from_slice(&(half as u64).to_be_bytes());
                let hash = blake2b(&data);
                let mut head = [0u8; 8];
                head.copy_from_slice(&hash.as_bytes()[..8]);
                rounds[round * halves + half] = (u64::from_be_bytes(head) & right_mask) as u32;
            }
        }

        Self {
            rounds: rounds.into_boxed_slice(),
            domain,
            right_mask,
            half_bits,
        }
    }

    /// Number of permuted indices.
    #[inline]
    pub fn domain(&self) -> u64 {
        self.domain
    }

    #[inline(always)]
    fn round(&self, round: usize, half: u64) -> u64 {
        u64::from(self.rounds[(round << self.half_bits) | half as usize])
    }

    #[inline(always)]
    fn encode(&self, index: u64) -> u64 {
        let mut left = index >> self.half_bits;
        let mut right = index & self.right_mask;
        for round in 0..FEISTEL_ROUNDS {
            let next = left ^ self.round(round, right);
            left = right;
            right = next;
        }
        (left << self.half_bits) | right
    }

    #[inline(always)]
    fn decode(&self, index: u64) -> u64 {
        let mut left = index >> self.half_bits;
        let mut right = index & self.right_mask;
        for round in (0..FEISTEL_ROUNDS).rev() {
            let previous = right ^ self.round(round, left);
            right = left;
            left = previous;
        }
        (left << self.half_bits) | right
    }

    /// `feistel::permute`: images outside the domain are permuted again.
    #[inline]
    pub fn permute(&self, index: u64) -> u64 {
        debug_assert!(index < self.domain);
        let mut image = self.encode(index);
        while image >= self.domain {
            image = self.encode(image);
        }
        image
    }

    /// `feistel::invert_permute`.
    #[inline]
    pub fn invert(&self, index: u64) -> u64 {
        debug_assert!(index < self.domain);
        let mut image = self.decode(index);
        while image >= self.domain {
            image = self.decode(image);
        }
        image
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEYS: [u64; 4] = [
        0x4bbb_a0f2_9444_cfa1,
        0xb8a6_06ce_d3a4_3675,
        0x99b0_2ec9_06f7_0a72,
        0x09cf_3687_b2a9_9106,
    ];

    #[test]
    fn is_a_permutation_with_inverse() {
        for domain in [8u64, 512, 1000, 8 << 10] {
            let permutation = WordPermutation::new(&KEYS, domain);
            let mut seen = vec![false; domain as usize];
            for index in 0..domain {
                let image = permutation.permute(index);
                assert!(image < domain);
                assert!(!seen[image as usize], "image {} repeats", image);
                seen[image as usize] = true;
                assert_eq!(permutation.invert(image), index);
            }
        }
    }

    // Values from the independent model `FABLE-PLAN-SDRv2-scripts/sdrv2_model.py golden`
    // (porep id `[5; 32]`, 64 nodes).
    #[test]
    fn matches_the_model() {
        let permutation = WordPermutation::new(&KEYS, 512);
        let images: Vec<u64> = (0..8).map(|index| permutation.permute(index)).collect();
        assert_eq!(images, [62, 173, 16, 227, 477, 345, 93, 442]);
    }
}
