//! Canonical, little-endian BLS12-381 scalar subtraction, without Montgomery conversion.
use anyhow::{ensure, Result};
use std::convert::TryInto;

const MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

fn limbs(bytes: &[u8; 32]) -> [u64; 4] {
    std::array::from_fn(|i| {
        u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().expect("eight bytes"))
    })
}

fn sub(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], bool) {
    let mut out = [0; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (value, b1) = a[i].overflowing_sub(b[i]);
        let (value, b2) = value.overflowing_sub(u64::from(borrow));
        out[i] = value;
        borrow = b1 || b2;
    }
    (out, borrow)
}

pub(super) fn canonical(bytes: &[u8; 32]) -> bool {
    sub(limbs(bytes), MODULUS).1
}

pub(super) fn subtract(value: &[u8; 32], key: &[u8; 32]) -> Result<[u8; 32]> {
    let a = limbs(value);
    let b = limbs(key);
    ensure!(
        sub(a, MODULUS).1 && sub(b, MODULUS).1,
        "noncanonical ZigZag field operand"
    );
    let (mut result, borrow) = sub(a, b);
    let mask = 0u64.wrapping_sub(u64::from(borrow));
    let mut carry = false;
    for i in 0..4 {
        let (value, c1) = result[i].overflowing_add(MODULUS[i] & mask);
        let (value, c2) = value.overflowing_add(u64::from(carry));
        result[i] = value;
        carry = c1 || c2;
    }
    let mut bytes = [0; 32];
    for (bytes, limb) in bytes.chunks_exact_mut(8).zip(result) {
        bytes.copy_from_slice(&limb.to_le_bytes());
    }
    Ok(bytes)
}
