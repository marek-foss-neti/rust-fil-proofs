//! Threefish-256, the keyed permutation of an SDR v2 label.
//!
//! Skein 1.3 (Ferguson et al., 2010): word permutation of Table 3, rotation constants of Table 4,
//! key schedule of Section 3.3.2. A label is one 256-bit block and its key is one SHA-256 output;
//! both are read as four little-endian 64-bit words. SDR v2 uses a zero tweak.

/// Rounds of Threefish-256.
pub const ROUNDS: usize = 72;

/// Key schedule parity constant `C240`.
const C240: u64 = 0x1BD1_1BDA_A9FC_1A22;

#[inline(always)]
fn key_schedule(key: &[u64; 4], tweak: &[u64; 2]) -> ([u64; 5], [u64; 3]) {
    (
        [
            key[0],
            key[1],
            key[2],
            key[3],
            C240 ^ key[0] ^ key[1] ^ key[2] ^ key[3],
        ],
        [tweak[0], tweak[1], tweak[0] ^ tweak[1]],
    )
}

/// Adds (`wrapping_add`) or removes (`wrapping_sub`) subkey `$s`. `$s` is a literal, so every
/// index below is a constant and no remainder is computed at run time.
macro_rules! inject {
    ($op:ident, $v0:ident, $v1:ident, $v2:ident, $v3:ident, $k:ident, $t:ident, $s:expr) => {
        $v0 = $v0.$op($k[$s % 5]);
        $v1 = $v1.$op($k[($s + 1) % 5].wrapping_add($t[$s % 3]));
        $v2 = $v2.$op($k[($s + 2) % 5].wrapping_add($t[($s + 1) % 3]));
        $v3 = $v3.$op($k[($s + 3) % 5].wrapping_add($s as u64));
    };
}

macro_rules! mix {
    ($a:ident, $b:ident, $rotation:expr) => {
        $a = $a.wrapping_add($b);
        $b = $b.rotate_left($rotation) ^ $a;
    };
}

macro_rules! unmix {
    ($a:ident, $b:ident, $rotation:expr) => {
        $b = ($b ^ $a).rotate_right($rotation);
        $a = $a.wrapping_sub($b);
    };
}

/// Rounds `8g..8g + 8` with the two subkeys injected before rounds `8g` and `8g + 4`.
///
/// The word permutation `(0, 3, 2, 1)` swaps words 1 and 3 after every round, so odd rounds mix
/// `(v0, v3)` and `(v2, v1)` and no word is moved.
macro_rules! encrypt_rounds {
    ($v0:ident, $v1:ident, $v2:ident, $v3:ident, $k:ident, $t:ident, $g:expr) => {
        inject!(wrapping_add, $v0, $v1, $v2, $v3, $k, $t, 2 * $g);
        mix!($v0, $v1, 14);
        mix!($v2, $v3, 16);
        mix!($v0, $v3, 52);
        mix!($v2, $v1, 57);
        mix!($v0, $v1, 23);
        mix!($v2, $v3, 40);
        mix!($v0, $v3, 5);
        mix!($v2, $v1, 37);
        inject!(wrapping_add, $v0, $v1, $v2, $v3, $k, $t, 2 * $g + 1);
        mix!($v0, $v1, 25);
        mix!($v2, $v3, 33);
        mix!($v0, $v3, 46);
        mix!($v2, $v1, 12);
        mix!($v0, $v1, 58);
        mix!($v2, $v3, 22);
        mix!($v0, $v3, 32);
        mix!($v2, $v1, 32);
    };
}

/// The inverse of `encrypt_rounds!`.
macro_rules! decrypt_rounds {
    ($v0:ident, $v1:ident, $v2:ident, $v3:ident, $k:ident, $t:ident, $g:expr) => {
        unmix!($v2, $v1, 32);
        unmix!($v0, $v3, 32);
        unmix!($v2, $v3, 22);
        unmix!($v0, $v1, 58);
        unmix!($v2, $v1, 12);
        unmix!($v0, $v3, 46);
        unmix!($v2, $v3, 33);
        unmix!($v0, $v1, 25);
        inject!(wrapping_sub, $v0, $v1, $v2, $v3, $k, $t, 2 * $g + 1);
        unmix!($v2, $v1, 37);
        unmix!($v0, $v3, 5);
        unmix!($v2, $v3, 40);
        unmix!($v0, $v1, 23);
        unmix!($v2, $v1, 57);
        unmix!($v0, $v3, 52);
        unmix!($v2, $v3, 16);
        unmix!($v0, $v1, 14);
        inject!(wrapping_sub, $v0, $v1, $v2, $v3, $k, $t, 2 * $g);
    };
}

/// Encrypts one block in place.
#[inline]
pub fn encrypt(key: &[u64; 4], tweak: &[u64; 2], block: &mut [u64; 4]) {
    let (k, t) = key_schedule(key, tweak);
    let [mut v0, mut v1, mut v2, mut v3] = *block;

    encrypt_rounds!(v0, v1, v2, v3, k, t, 0);
    encrypt_rounds!(v0, v1, v2, v3, k, t, 1);
    encrypt_rounds!(v0, v1, v2, v3, k, t, 2);
    encrypt_rounds!(v0, v1, v2, v3, k, t, 3);
    encrypt_rounds!(v0, v1, v2, v3, k, t, 4);
    encrypt_rounds!(v0, v1, v2, v3, k, t, 5);
    encrypt_rounds!(v0, v1, v2, v3, k, t, 6);
    encrypt_rounds!(v0, v1, v2, v3, k, t, 7);
    encrypt_rounds!(v0, v1, v2, v3, k, t, 8);
    inject!(wrapping_add, v0, v1, v2, v3, k, t, 18);

    *block = [v0, v1, v2, v3];
}

/// Decrypts one block in place; the inverse of [`encrypt`].
#[inline]
pub fn decrypt(key: &[u64; 4], tweak: &[u64; 2], block: &mut [u64; 4]) {
    let (k, t) = key_schedule(key, tweak);
    let [mut v0, mut v1, mut v2, mut v3] = *block;

    inject!(wrapping_sub, v0, v1, v2, v3, k, t, 18);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 8);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 7);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 6);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 5);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 4);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 3);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 2);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 1);
    decrypt_rounds!(v0, v1, v2, v3, k, t, 0);

    *block = [v0, v1, v2, v3];
}

#[inline(always)]
fn word(bytes: &[u8; 32], index: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[8 * index..8 * index + 8]);
    u64::from_le_bytes(raw)
}

#[inline(always)]
fn load(bytes: &[u8; 32]) -> [u64; 4] {
    [
        word(bytes, 0),
        word(bytes, 1),
        word(bytes, 2),
        word(bytes, 3),
    ]
}

#[inline(always)]
fn store(words: &[u64; 4], bytes: &mut [u8; 32]) {
    bytes[0..8].copy_from_slice(&words[0].to_le_bytes());
    bytes[8..16].copy_from_slice(&words[1].to_le_bytes());
    bytes[16..24].copy_from_slice(&words[2].to_le_bytes());
    bytes[24..32].copy_from_slice(&words[3].to_le_bytes());
}

/// `label = Threefish-256(key, label)` with a zero tweak.
#[inline]
pub fn encrypt_label(key: &[u8; 32], label: &mut [u8; 32]) {
    let mut block = load(label);
    encrypt(&load(key), &[0, 0], &mut block);
    store(&block, label);
}

/// The inverse of [`encrypt_label`].
#[inline]
pub fn decrypt_label(key: &[u8; 32], label: &mut [u8; 32]) {
    let mut block = load(label);
    decrypt(&load(key), &[0, 0], &mut block);
    store(&block, label);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes32(hex: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (index, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * index..2 * index + 2], 16).expect("hex digit");
        }
        out
    }

    // Skein 1.3 reference known answers for Threefish-256.
    #[test]
    fn known_answer_zero_key() {
        let mut block = [0u64; 4];
        encrypt(&[0; 4], &[0; 2], &mut block);
        let mut bytes = [0u8; 32];
        store(&block, &mut bytes);
        assert_eq!(
            bytes,
            bytes32("84da2a1f8beaee947066ae3e3103f1ad536db1f4a1192495116b9f3ce6133fd8")
        );
        decrypt(&[0; 4], &[0; 2], &mut block);
        assert_eq!(block, [0u64; 4]);
    }

    #[test]
    fn known_answer_with_tweak() {
        let mut key = [0u8; 32];
        let mut plain = [0u8; 32];
        for index in 0..32 {
            key[index] = 0x10 + index as u8;
            plain[index] = 0xff - index as u8;
        }
        let tweak = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
        let mut block = load(&plain);
        encrypt(&load(&key), &tweak, &mut block);
        let mut bytes = [0u8; 32];
        store(&block, &mut bytes);
        assert_eq!(
            bytes,
            bytes32("e0d091ff0eea8fdfc98192e62ed80ad59d865d08588df476657056b5955e97df")
        );
        decrypt(&load(&key), &tweak, &mut block);
        assert_eq!(block, load(&plain));
    }

    // Skein-256-256 of the one-byte message FF (Skein 1.3, Appendix C.1), the check used by the
    // report's `threefish-label-bench.c`.
    #[test]
    fn skein_256_256_of_ff() {
        let iv = [
            0xFC9D_A860_D048_B449,
            0x2FCA_6647_9FA7_D833,
            0xB33B_C389_6656_840F,
            0x6A54_E920_FDE8_DA69,
        ];
        let message = [0xff, 0, 0, 0];
        let mut state = message;
        encrypt(&iv, &[1, (48 << 56) | (1 << 62) | (1 << 63)], &mut state);
        for (word, input) in state.iter_mut().zip(message.iter()) {
            *word ^= input;
        }
        let mut output = [0u64; 4];
        encrypt(
            &state,
            &[8, (63 << 56) | (1 << 62) | (1 << 63)],
            &mut output,
        );
        let mut bytes = [0u8; 32];
        store(&output, &mut bytes);
        assert_eq!(
            bytes,
            bytes32("0b98dcd198ea0e50a7a244c444e25c23da30c10fc9a1f270a6637f1f34e67ed2")
        );
    }

    #[test]
    fn label_round_trip() {
        let mut key = [0u8; 32];
        let mut label = [0u8; 32];
        for round in 0..64u8 {
            for index in 0..32 {
                key[index] = key[index]
                    .wrapping_mul(31)
                    .wrapping_add(round ^ index as u8);
                label[index] = label[index]
                    .wrapping_mul(17)
                    .wrapping_add(round + index as u8);
            }
            let original = label;
            encrypt_label(&key, &mut label);
            assert_ne!(label, original);
            let mut decrypted = label;
            decrypt_label(&key, &mut decrypted);
            assert_eq!(decrypted, original);
        }
    }
}
