//! End-to-end tests for the ZigZag PoRep filecoin-proofs API.
//!
//! These exercise the full `filecoin-proofs` ZigZag path (parameters + caches + seal API), building
//! actual sectors of the requested size and confirming:
//!
//! * replication changes the data and produces non-trivial commitments,
//! * extraction (fast, parallel decode) recovers the original data bit-for-bit,
//! * (for the small size, behind `--ignored`) a full Groth16 proof verifies.
//!
//! The larger sizes are marked `#[ignore]` because they replicate a full sector (and, for the
//! Groth16 test, generate proving parameters), which is expensive.

use filecoin_proofs::constants::{ZigZagTree, SECTOR_SIZE_16_MIB, SECTOR_SIZE_2_KIB};
use filecoin_proofs::parameters::zigzag_public_params;
use filecoin_proofs::types::PoRepConfig;
use filecoin_proofs::{zigzag_pre_commit, zigzag_prove, zigzag_unseal, zigzag_verify_seal};
use rand::rngs::OsRng;
use storage_proofs_core::{
    api_version::ApiVersion,
    compound_proof::CompoundProof,
    parameter_cache::CacheableParameters,
    sector::SectorId,
    util::NODE_SIZE,
};
use storage_proofs_porep::zigzag::{circuit::ZigZagCompound, ZigZagDrgPoRep};

const PROVER_ID: [u8; 32] = [4u8; 32];
const TICKET: [u8; 32] = [7u8; 32];
const POREP_ID: [u8; 32] = [42u8; 32];

/// Generate `sector_size` bytes of random data whose every 32-byte node is a canonical BLS12-381
/// scalar (achieved by zeroing the most-significant little-endian byte of each node).
fn random_sector_data(sector_size: usize) -> Vec<u8> {
    let mut data = vec![0u8; sector_size];
    for node in data.chunks_mut(NODE_SIZE) {
        for b in node.iter_mut() {
            *b = rand::random::<u8>();
        }
        // Clear the top byte (little-endian) so the value is < the scalar field modulus.
        node[NODE_SIZE - 1] = 0;
    }
    data
}

/// Replicate a sector and extract it again, asserting a lossless roundtrip. Returns the padded
/// sector data after extraction (which must equal the original).
fn zigzag_extract_lifecycle(sector_size: u64) {
    let porep_config = PoRepConfig::new_groth16(sector_size, POREP_ID, ApiVersion::V1_2_0);
    let sector_id = SectorId::from(0);

    let original = random_sector_data(sector_size as usize);
    let mut data = original.clone();

    let (out, _state) = zigzag_pre_commit::<ZigZagTree>(
        &porep_config,
        PROVER_ID,
        sector_id,
        TICKET,
        &mut data,
    )
    .expect("zigzag_pre_commit failed");

    assert_ne!(data, original, "replication did not change the data");
    assert_ne!(out.comm_d, [0u8; 32], "comm_d is all zero");
    assert_ne!(out.comm_r, [0u8; 32], "comm_r is all zero");
    assert_ne!(out.comm_r_star, [0u8; 32], "comm_r_star is all zero");

    zigzag_unseal::<ZigZagTree>(
        &porep_config,
        PROVER_ID,
        sector_id,
        TICKET,
        out.comm_d,
        &mut data,
    )
    .expect("zigzag_unseal failed");

    assert_eq!(data, original, "extraction did not recover the original data");
}

/// Generate (and cache to disk) the ZigZag Groth16 parameters + verifying key for `porep_config`,
/// mirroring what `paramcache --only-zigzag` does. The seal API reads these from the cache.
fn generate_zigzag_params(porep_config: &PoRepConfig) {
    let public_params =
        zigzag_public_params::<ZigZagTree>(porep_config).expect("failed to get zigzag public params");

    let circuit = <ZigZagCompound<ZigZagTree> as CompoundProof<
        ZigZagDrgPoRep<ZigZagTree>,
        _,
    >>::blank_circuit(&public_params);

    ZigZagCompound::<ZigZagTree>::get_groth_params(Some(&mut OsRng), circuit.clone(), &public_params)
        .expect("failed to generate groth params");
    ZigZagCompound::<ZigZagTree>::get_verifying_key(Some(&mut OsRng), circuit, &public_params)
        .expect("failed to generate verifying key");
}

/// Full seal lifecycle including a Groth16 proof and verification.
fn zigzag_seal_lifecycle(sector_size: u64) {
    // Point the durable parameter cache at a throwaway directory and generate ZigZag params there,
    // so this test is self-contained (production would use pre-fetched params from `paramcache`).
    // Must be set before any code touches the lazily-initialized settings.
    let cache_dir = std::env::temp_dir().join(format!(
        "zigzag-params-{}-{}",
        sector_size,
        std::process::id()
    ));
    std::fs::create_dir_all(&cache_dir).expect("failed to create param cache dir");
    std::env::set_var("FIL_PROOFS_PARAMETER_CACHE", &cache_dir);

    let porep_config = PoRepConfig::new_groth16(sector_size, POREP_ID, ApiVersion::V1_2_0);
    let sector_id = SectorId::from(0);

    generate_zigzag_params(&porep_config);

    let original = random_sector_data(sector_size as usize);
    let mut data = original.clone();

    let (out, state) = zigzag_pre_commit::<ZigZagTree>(
        &porep_config,
        PROVER_ID,
        sector_id,
        TICKET,
        &mut data,
    )
    .expect("zigzag_pre_commit failed");

    let commit = zigzag_prove::<ZigZagTree>(&porep_config, state).expect("zigzag_prove failed");

    let verified = zigzag_verify_seal::<ZigZagTree>(
        &porep_config,
        out.comm_r,
        out.comm_d,
        out.comm_r_star,
        PROVER_ID,
        sector_id,
        TICKET,
        &commit.proof,
    )
    .expect("zigzag_verify_seal errored");
    assert!(verified, "zigzag proof failed to verify");

    // A proof over the wrong commitment must be rejected.
    let mut bad_comm_r = out.comm_r;
    bad_comm_r[0] ^= 0x01;
    let bad = zigzag_verify_seal::<ZigZagTree>(
        &porep_config,
        bad_comm_r,
        out.comm_d,
        out.comm_r_star,
        PROVER_ID,
        sector_id,
        TICKET,
        &commit.proof,
    )
    .expect("zigzag_verify_seal errored");
    assert!(!bad, "proof verified against a tampered comm_r");

    // Extraction still recovers the original data.
    zigzag_unseal::<ZigZagTree>(
        &porep_config,
        PROVER_ID,
        sector_id,
        TICKET,
        out.comm_d,
        &mut data,
    )
    .expect("zigzag_unseal failed");
    assert_eq!(data, original, "extraction did not recover the original data");
}

#[test]
fn test_zigzag_extract_roundtrip_2kib() {
    zigzag_extract_lifecycle(SECTOR_SIZE_2_KIB);
}

#[test]
#[ignore = "replicates a full 16MiB sector; slow and memory-heavy"]
fn test_zigzag_extract_roundtrip_16mib() {
    zigzag_extract_lifecycle(SECTOR_SIZE_16_MIB);
}

#[test]
#[ignore = "generates Groth16 parameters and a full proof; slow"]
fn test_zigzag_seal_lifecycle_2kib() {
    zigzag_seal_lifecycle(SECTOR_SIZE_2_KIB);
}
