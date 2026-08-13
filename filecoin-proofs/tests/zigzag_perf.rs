//! Ignored ZigZag VDE timing harness.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p filecoin-proofs --release --test zigzag_perf -- --ignored --nocapture
//! ```
//!
//! By default this measures a 512MiB sector. Set `ZIGZAG_PERF_SECTOR_SIZE=2kib|8mib|512mib`
//! for smaller smoke runs.

use std::time::{Duration, Instant};

use cpu_time::ProcessTime;
use filecoin_hashers::poseidon::{PoseidonDomain, PoseidonHasher};
use filecoin_proofs::constants::{SECTOR_SIZE_2_KIB, SECTOR_SIZE_512_MIB, SECTOR_SIZE_8_MIB};
use storage_proofs_core::{
    api_version::ApiVersion,
    drgraph::{Graph, BASE_DEGREE},
    util::NODE_SIZE,
};
use storage_proofs_porep::zigzag::{
    decode, encode, prepare_parent_table, ZigZagBucketGraph, ZigZagGraph, EXP_DEGREE,
};

const POREP_ID: [u8; 32] = [42u8; 32];

struct Measurement {
    wall: Duration,
    cpu: Duration,
}

fn measure<T>(action: impl FnOnce() -> T) -> (T, Measurement) {
    let cpu_start = ProcessTime::now();
    let wall_start = Instant::now();
    let result = action();
    (
        result,
        Measurement {
            wall: wall_start.elapsed(),
            cpu: cpu_start.elapsed(),
        },
    )
}

fn print_measurement(orientation: &str, phase: &str, nodes: usize, measurement: &Measurement) {
    let wall_s = measurement.wall.as_secs_f64();
    let cpu_s = measurement.cpu.as_secs_f64();
    let cpu_us_per_node = cpu_s * 1_000_000.0 / nodes as f64;
    let wall_us_per_node = wall_s * 1_000_000.0 / nodes as f64;
    println!(
        "zigzag_perf orientation={orientation} phase={phase} nodes={nodes} wall_s={wall_s:.6} \
         cpu_s={cpu_s:.6} wall_us_per_node={wall_us_per_node:.6} \
         cpu_us_per_node={cpu_us_per_node:.6}"
    );
}

fn sector_size_from_env() -> u64 {
    match std::env::var("ZIGZAG_PERF_SECTOR_SIZE") {
        Ok(value) => match value
            .to_ascii_lowercase()
            .replace(['_', '-', ' '], "")
            .replace("kb", "kib")
            .replace("mb", "mib")
            .as_str()
        {
            "2kib" => SECTOR_SIZE_2_KIB,
            "8mib" => SECTOR_SIZE_8_MIB,
            "512mib" => SECTOR_SIZE_512_MIB,
            other => panic!(
                "unsupported ZIGZAG_PERF_SECTOR_SIZE={}; expected 2kib, 8mib, or 512mib",
                other
            ),
        },
        Err(_) => SECTOR_SIZE_512_MIB,
    }
}

fn deterministic_sector(sector_size: u64) -> Vec<u8> {
    let nodes = sector_size as usize / NODE_SIZE;
    let mut data = vec![0u8; sector_size as usize];
    for node in 0..nodes {
        let start = node * NODE_SIZE;
        let value = (node as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(17);
        data[start..start + 8].copy_from_slice(&value.to_le_bytes());
        data[start + 8..start + 16].copy_from_slice(&value.rotate_left(17).to_le_bytes());
    }
    data
}

fn run_orientation(
    orientation: &str,
    graph: &ZigZagBucketGraph<PoseidonHasher>,
    replica_id: &PoseidonDomain,
    original: &[u8],
) {
    let nodes = graph.size();
    let mut data = original.to_vec();

    let (encoded_result, encode_measurement) =
        measure(|| encode::<PoseidonHasher, _>(graph, replica_id, &mut data));
    encoded_result.expect("zigzag encode failed");
    assert_ne!(data, original, "encoding did not change data");
    print_measurement(orientation, "encode", nodes, &encode_measurement);

    let (decoded_result, decode_measurement) =
        measure(|| decode::<PoseidonHasher, _>(graph, replica_id, &data));
    let decoded = decoded_result.expect("zigzag decode failed");
    assert_eq!(decoded, original, "decoding did not recover original data");
    print_measurement(orientation, "decode", nodes, &decode_measurement);
}

#[test]
#[ignore = "512MiB ZigZag VDE timing harness; run explicitly with --ignored --nocapture"]
fn zigzag_vde_encode_decode_perf() {
    let sector_size = sector_size_from_env();
    assert_eq!(sector_size as usize % NODE_SIZE, 0);
    let nodes = sector_size as usize / NODE_SIZE;
    let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
        None,
        nodes,
        BASE_DEGREE,
        EXP_DEGREE,
        POREP_ID,
        ApiVersion::V1_2_0,
    )
    .expect("failed to create ZigZag graph");
    let reversed = graph.zigzag();
    let replica_id = PoseidonDomain::from([9u8; 32]);
    let original = deterministic_sector(sector_size);

    println!(
        "zigzag_perf sector_size={} nodes={} degree={} chunk_nodes=16384",
        sector_size,
        nodes,
        graph.degree()
    );

    let (prewarm_result, prewarm_measurement) = measure(|| {
        prepare_parent_table::<PoseidonHasher, _>(&graph)?;
        prepare_parent_table::<PoseidonHasher, _>(&reversed)
    });
    prewarm_result.expect("zigzag parent table prewarm failed");
    print_measurement(
        "both",
        "parent_table_prewarm",
        nodes * 2,
        &prewarm_measurement,
    );

    run_orientation("forward", &graph, &replica_id, &original);
    run_orientation("reversed", &reversed, &replica_id, &original);
}
