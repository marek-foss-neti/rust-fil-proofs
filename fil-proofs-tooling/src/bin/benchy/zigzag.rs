use std::io::stdout;

use fil_proofs_tooling::shared::{PROVER_ID, TICKET_BYTES};
use fil_proofs_tooling::{measure, Metadata};
use filecoin_proofs::constants::ZigZagTree;
use filecoin_proofs::types::PoRepConfig;
use filecoin_proofs::{zigzag_pre_commit, zigzag_prove, zigzag_unseal, zigzag_verify_seal};
use log::info;
use serde::{Deserialize, Serialize};
use storage_proofs_core::{api_version::ApiVersion, sector::SectorId, util::NODE_SIZE};

const SECTOR_ID: u64 = 0;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Inputs {
    sector_size: u64,
    prove: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Outputs {
    /// Time to replicate (encode) the sector. This is the "slow" direction of the asymmetry.
    replicate_cpu_time_ms: u64,
    replicate_wall_time_ms: u64,
    /// Time to extract (decode) the original data. ZigZag's decode is embarrassingly parallel, so
    /// this is expected to be substantially faster than replication.
    extract_cpu_time_ms: u64,
    extract_wall_time_ms: u64,
    /// `replicate_wall_time_ms / extract_wall_time_ms`, the extraction speed-up. Values > 1 confirm
    /// the fast-extraction asymmetry holds.
    extract_speedup: f64,
    /// SNARK proving time (only populated when `--prove` is passed).
    seal_prove_cpu_time_ms: u64,
    seal_prove_wall_time_ms: u64,
    /// SNARK verification time (only populated when `--prove` is passed).
    verify_cpu_time_ms: u64,
    verify_wall_time_ms: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Report {
    inputs: Inputs,
    outputs: Outputs,
}

impl Report {
    fn print(&self) {
        let wrapped = Metadata::wrap(&self).expect("failed to retrieve metadata");
        serde_json::to_writer(stdout(), &wrapped).expect("cannot write report JSON to stdout");
    }
}

pub fn run(sector_size: usize, api_version: ApiVersion, prove: bool) -> anyhow::Result<()> {
    info!(
        "Benchy ZigZag PoRep: sector-size={}, api_version={}, prove={}",
        sector_size, api_version, prove
    );

    let sector_id = SectorId::from(SECTOR_ID);
    let porep_config = PoRepConfig::new_groth16(sector_size as u64, [0; 32], api_version);

    // Random sector data. ZigZag operates directly on 32-byte field-element nodes, so each node's
    // most-significant (little-endian) byte is cleared to keep it a canonical BLS12-381 scalar.
    let mut original = vec![0u8; sector_size];
    for node in original.chunks_mut(NODE_SIZE) {
        for b in node.iter_mut() {
            *b = rand::random::<u8>();
        }
        node[NODE_SIZE - 1] = 0;
    }
    let mut data = original.clone();

    // Replicate (encode) in place.
    let replicate = measure(|| {
        zigzag_pre_commit::<ZigZagTree>(
            &porep_config,
            PROVER_ID,
            sector_id,
            TICKET_BYTES,
            &mut data,
        )
    })
    .expect("failed in zigzag_pre_commit");
    let (pre_commit_out, prover_state) = replicate.return_value;

    // Optional SNARK proof + verification.
    let (seal_prove_cpu_time_ms, seal_prove_wall_time_ms, verify_cpu_time_ms, verify_wall_time_ms) =
        if prove {
            let prove_m =
                measure(|| zigzag_prove::<ZigZagTree>(&porep_config, prover_state))
                    .expect("failed in zigzag_prove");
            let proof = prove_m.return_value.proof;

            let verify_m = measure(|| {
                zigzag_verify_seal::<ZigZagTree>(
                    &porep_config,
                    pre_commit_out.comm_r,
                    pre_commit_out.comm_d,
                    pre_commit_out.comm_r_star,
                    PROVER_ID,
                    sector_id,
                    TICKET_BYTES,
                    &proof,
                )
            })
            .expect("failed in zigzag_verify_seal");
            assert!(verify_m.return_value, "zigzag proof failed to verify");

            (
                prove_m.cpu_time.as_millis() as u64,
                prove_m.wall_time.as_millis() as u64,
                verify_m.cpu_time.as_millis() as u64,
                verify_m.wall_time.as_millis() as u64,
            )
        } else {
            // Drop the prover state (per-layer trees) if we are not proving.
            drop(prover_state);
            (0, 0, 0, 0)
        };

    // Extract (decode) the original data back, in place.
    let extract = measure(|| {
        zigzag_unseal::<ZigZagTree>(
            &porep_config,
            PROVER_ID,
            sector_id,
            TICKET_BYTES,
            pre_commit_out.comm_d,
            &mut data,
        )
    })
    .expect("failed in zigzag_unseal");

    assert_eq!(data, original, "extracted data does not match original");

    let replicate_wall_time_ms = replicate.wall_time.as_millis() as u64;
    let extract_wall_time_ms = extract.wall_time.as_millis() as u64;
    let extract_speedup = if extract_wall_time_ms == 0 {
        f64::INFINITY
    } else {
        replicate_wall_time_ms as f64 / extract_wall_time_ms as f64
    };

    let report = Report {
        inputs: Inputs {
            sector_size: sector_size as u64,
            prove,
        },
        outputs: Outputs {
            replicate_cpu_time_ms: replicate.cpu_time.as_millis() as u64,
            replicate_wall_time_ms,
            extract_cpu_time_ms: extract.cpu_time.as_millis() as u64,
            extract_wall_time_ms,
            extract_speedup,
            seal_prove_cpu_time_ms,
            seal_prove_wall_time_ms,
            verify_cpu_time_ms,
            verify_wall_time_ms,
        },
    };

    report.print();
    Ok(())
}
