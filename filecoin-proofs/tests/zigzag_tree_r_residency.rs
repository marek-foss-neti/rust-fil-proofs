//! Small remote correctness fixture: no Groth16 parameter generation or proving.
use std::sync::Mutex;

use filecoin_proofs::{
    add_piece,
    constants::{DefaultPieceHasher, ZigZagTree},
    parameters::zigzag_public_params,
    types::PoRepConfig,
    zigzag_commit_phase1_from_cache, zigzag_commit_phase2, zigzag_commit_phase2_with_cache,
    zigzag_pre_commit_phase1, zigzag_prove, zigzag_validate_commit_phase1, UnpaddedBytesAmount,
};
use storage_proofs_core::{
    api_version::ApiVersion,
    parameter_cache::{self, CacheableParameters},
    sector::SectorId,
};
use storage_proofs_porep::zigzag::circuit::{ZigZagCircuit, ZigZagCompound};
use storage_proofs_porep::zigzag::measurements::{
    set_operation_observer, OperationBoundary, OperationEvent,
};

static EVENTS: Mutex<Vec<(&'static str, bool)>> = Mutex::new(Vec::new());

fn observe(event: OperationEvent) {
    EVENTS.lock().unwrap().push((
        event.name,
        matches!(event.boundary, OperationBoundary::Start),
    ));
}

fn assert_before_parameters(enabled: bool) {
    let events = std::mem::take(&mut *EVENTS.lock().unwrap());
    let load = events
        .iter()
        .position(|e| *e == ("groth16_parameter_load", true))
        .expect("parameter loader reached");
    let release = events
        .iter()
        .position(|e| *e == ("groth16_tree_r_cache_release", false));
    if enabled {
        assert!(release.expect("release completed") < load);
    } else {
        assert!(release.is_none());
    }
}

#[test]
fn tree_r_advice_is_deferred_until_validated_c2() -> anyhow::Result<()> {
    // Fresh processes isolate static settings and the operation observer.
    if std::env::var_os("ZIGZAG_RESIDENCY_TEST_CHILD").is_none() {
        for (builder, pipeline, advice) in [
            ("legacy", "sequential", "1"),
            ("binary", "sequential", "1"),
            ("binary", "overlap", "1"),
            ("binary", "stream", "1"),
            ("binary", "stream", "0"),
        ] {
            let cache = tempfile::tempdir()?;
            let status = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "tree_r_advice_is_deferred_until_validated_c2",
                    "--nocapture",
                ])
                .env("ZIGZAG_RESIDENCY_TEST_CHILD", "1")
                .env("FIL_PROOFS_PARAMETER_CACHE", cache.path())
                .env("FIL_PROOFS_PARENT_CACHE", cache.path().join("parents"))
                .env("FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS", "0")
                .env("FIL_PROOFS_ZIGZAG_PARAMETER_LOADER", "compact")
                .env("FIL_PROOFS_ZIGZAG_TREE_R_DONTNEED", advice)
                .env("FIL_PROOFS_ZIGZAG_PARENT_CACHE_DONTNEED", "0")
                .env("FIL_PROOFS_ZIGZAG_TREE_R_BUILDER", builder)
                .env("FIL_PROOFS_ZIGZAG_LAYER_PIPELINE", pipeline)
                .env("FIL_PROOFS_ZIGZAG_MULTICORE_ENCODE", "1")
                .env("FIL_PROOFS_ZIGZAG_MULTICORE_ENCODE_PRODUCER_STRIDE", "1")
                .env("FIL_PROOFS_ZIGZAG_MULTICORE_ENCODE_LOOKAHEAD", "4")
                .env("RAYON_NUM_THREADS", "4")
                .status()?;
            assert!(
                status.success(),
                "{}/{}/DONTNEED={}",
                builder,
                pipeline,
                advice
            );
        }
        return Ok(());
    }
    set_operation_observer(observe).expect("install observer");
    let enabled = std::env::var("FIL_PROOFS_ZIGZAG_TREE_R_DONTNEED")? == "1";
    let cache = tempfile::tempdir()?;
    let config = PoRepConfig::new_groth16(2048, [42; 32], ApiVersion::V1_2_0);
    // Existing but truncated files enter the measured loader and fail immediately.
    // An absent parameter file exits before that operation, so cannot test ordering.
    let params = zigzag_public_params::<ZigZagTree>(&config)?;
    let id = <ZigZagCompound<ZigZagTree, DefaultPieceHasher> as CacheableParameters<
        ZigZagCircuit<ZigZagTree, DefaultPieceHasher>,
        _,
    >>::cache_identifier(&params);
    std::fs::write(
        parameter_cache::parameter_cache_params_path(&id),
        b"truncated",
    )?;
    std::fs::write(
        parameter_cache::parameter_cache_verifying_key_path(&id),
        b"",
    )?;
    let sector = SectorId::from(0);
    let mut data = Vec::new();
    let (piece, _) = add_piece(
        std::io::Cursor::new(vec![9; 2032]),
        &mut data,
        UnpaddedBytesAmount(2032),
        &[],
    )?;
    let (out, state) = zigzag_pre_commit_phase1::<ZigZagTree>(
        &config,
        cache.path(),
        [4; 32],
        sector,
        [7; 32],
        &mut data,
        &[piece],
    )?;
    let c1 = zigzag_commit_phase1_from_cache::<ZigZagTree>(
        &config,
        cache.path(),
        out.comm_d,
        out.comm_r,
        out.comm_r_star,
        [4; 32],
        sector,
        [7; 32],
        Some([0xff; 32]),
    )?;
    let serialized = serde_json::to_vec(&c1)?;
    drop(c1);
    let c1: filecoin_proofs::ZigZagCommitPhase1Output<ZigZagTree> =
        serde_json::from_slice(&serialized)?;
    zigzag_validate_commit_phase1(&config, &c1, [4; 32], sector)?;
    assert!(!EVENTS
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.0 == "groth16_tree_r_cache_release"));
    EVENTS.lock().unwrap().clear();

    let mut invalid: filecoin_proofs::ZigZagCommitPhase1Output<ZigZagTree> =
        serde_json::from_slice(&serialized)?;
    invalid.layer_comm_rs[0][0] ^= 1;
    assert!(
        zigzag_commit_phase2_with_cache(&config, invalid, [4; 32], sector, Some(cache.path()))
            .is_err()
    );
    assert!(
        EVENTS.lock().unwrap().is_empty(),
        "invalid C1 must not release pages or load parameters"
    );

    // Parameters are deliberately truncated: stop at the release/loader boundary.
    assert!(
        zigzag_commit_phase2_with_cache(&config, c1, [4; 32], sector, Some(cache.path())).is_err()
    );
    assert_before_parameters(enabled);
    assert!(zigzag_prove(&config, state, Some([0xff; 32])).is_err());
    {
        let events = EVENTS.lock().unwrap();
        let validated = events
            .iter()
            .position(|e| *e == ("vanilla_verification", false))
            .unwrap();
        if enabled {
            let release = events
                .iter()
                .position(|e| *e == ("groth16_tree_r_cache_release", true))
                .unwrap();
            assert!(validated < release);
        }
    }
    assert_before_parameters(enabled);
    let c1 = serde_json::from_slice(&serialized)?;
    let missing = cache.path().join("missing-on-another-host");
    assert!(zigzag_commit_phase2_with_cache::<ZigZagTree>(
        &config,
        c1,
        [4; 32],
        sector,
        Some(&missing)
    )
    .is_err());
    assert_before_parameters(enabled);
    let c1 = serde_json::from_slice(&serialized)?;
    assert!(zigzag_commit_phase2::<ZigZagTree>(&config, c1, [4; 32], sector).is_err());
    assert_before_parameters(enabled);
    Ok(())
}
