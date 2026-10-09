use super::*;
use std::{fs, fs::File, path::PathBuf, process::Command};

use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use blstrs::Scalar;
use storage_proofs_core::api_version::ApiVersion;

use crate::constants::{ZigZagTree, SECTOR_SIZE_2_KIB};

#[derive(Clone)]
struct Multiply;

impl Circuit<Scalar> for Multiply {
    fn synthesize<CS: ConstraintSystem<Scalar>>(
        self,
        cs: &mut CS,
    ) -> std::result::Result<(), SynthesisError> {
        let x = cs.alloc(|| "x", || Ok(Scalar::from(3)))?;
        let y = cs.alloc(|| "y", || Ok(Scalar::from(5)))?;
        let z = cs.alloc_input(|| "z", || Ok(Scalar::from(15)))?;
        cs.enforce(|| "multiply", |lc| lc + x, |lc| lc + y, |lc| lc + z);
        Ok(())
    }
}

fn run_in_fresh_cache_process(test_name: &str) -> bool {
    if std::env::var("ZIGZAG_MEMORY_CACHE_TEST").ok().as_deref() != Some(test_name) {
        // Give each process its own SETTINGS/cache directory and selected loader.
        for loader in ["compact", "mapped"] {
            let directory = tempfile::tempdir().unwrap();
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    &format!("caches::tests::{}", test_name),
                    "--nocapture",
                ])
                .env("ZIGZAG_MEMORY_CACHE_TEST", test_name)
                .env("FIL_PROOFS_PARAMETER_CACHE", directory.path())
                .env("FIL_PROOFS_VERIFY_PRODUCTION_PARAMS", "false")
                .env("FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS", "false")
                .env("FIL_PROOFS_ZIGZAG_PARAMETER_LOADER", loader)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}: {}\n{}",
                loader,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return true;
    }
    false
}

fn tiny_parameter_cache() -> (PoRepConfig, PathBuf, PathBuf) {
    let config = PoRepConfig::new_groth16(SECTOR_SIZE_2_KIB, [42; 32], ApiVersion::V1_2_0);
    let public_params = zigzag_public_params::<ZigZagTree>(&config).unwrap();
    let id = <ZigZagCompound<ZigZagTree, DefaultPieceHasher> as CacheableParameters<
        ZigZagCircuit<ZigZagTree, DefaultPieceHasher>,
        _,
    >>::cache_identifier(&public_params);
    let params_path = parameter_cache::parameter_cache_params_path(&id);
    let vk_path = parameter_cache::parameter_cache_verifying_key_path(&id);
    fs::create_dir_all(params_path.parent().unwrap()).unwrap();

    // Only the cache/loader is exercised: this tiny fixture needs no PoRep setup.
    let fixture = groth16::generate_random_parameters::<Bls12, _, _>(Multiply, &mut OsRng).unwrap();
    fixture.write(File::create(&params_path).unwrap()).unwrap();
    fixture.vk.write(File::create(&vk_path).unwrap()).unwrap();
    (config, params_path, vk_path)
}

#[test]
fn zigzag_memory_cache_recovers_after_initialization_panic() {
    if run_in_fresh_cache_process("zigzag_memory_cache_recovers_after_initialization_panic") {
        return;
    }
    let (config, params_path, vk_path) = tiny_parameter_cache();
    let parameter_bytes = fs::read(&params_path).unwrap();
    let vk_bytes = fs::read(&vk_path).unwrap();

    // Simulate unwinding during initialization, before publishing a source.
    // The next lookup goes through the production recovery and loader paths.
    let failed = std::panic::catch_unwind(|| {
        let cache = ZIGZAG_PARAM_MEMORY_CACHE.lock().unwrap();
        assert!(cache.is_empty());
        let _unpublished = ZigZagParameters::open(&params_path, "fixture".into(), false).unwrap();
        panic!("injected parameter initialization failure");
    });
    assert!(failed.is_err());
    assert!(ZIGZAG_PARAM_MEMORY_CACHE.is_poisoned());
    let loaded =
        get_zigzag_params::<ZigZagTree>(&config).expect("retry after initialization panic");
    assert!(!ZIGZAG_PARAM_MEMORY_CACHE.is_poisoned());
    assert_eq!(ZIGZAG_PARAM_MEMORY_CACHE.lock().unwrap().len(), 1);

    // A later panic must also retain the already-published, immutable source.
    let failed = std::panic::catch_unwind(|| {
        let _cache = ZIGZAG_PARAM_MEMORY_CACHE.lock().unwrap();
        panic!("injected cache validation failure");
    });
    assert!(failed.is_err());
    let reused = get_zigzag_params::<ZigZagTree>(&config).expect("reuse after validation panic");
    assert!(!ZIGZAG_PARAM_MEMORY_CACHE.is_poisoned());
    assert!(Arc::ptr_eq(&loaded, &reused));
    assert_eq!(fs::read(&params_path).unwrap(), parameter_bytes);
    assert_eq!(fs::read(&vk_path).unwrap(), vk_bytes);
}

#[test]
fn zigzag_memory_cache_repairs_missing_vk_without_reloading_parameters() {
    if run_in_fresh_cache_process(
        "zigzag_memory_cache_repairs_missing_vk_without_reloading_parameters",
    ) {
        return;
    }
    let (config, params_path, vk_path) = tiny_parameter_cache();
    let lock_path = params_path.with_extension("params.lock");
    let parameter_bytes = fs::read(&params_path).unwrap();
    let vk_bytes = fs::read(&vk_path).unwrap();

    let loaded = get_zigzag_params::<ZigZagTree>(&config).unwrap();
    assert!(
        !lock_path.exists(),
        "a ready cache needs no publication lock"
    );
    fs::remove_file(&vk_path).unwrap();
    let repaired = get_zigzag_params::<ZigZagTree>(&config).expect("repair VK on memory cache hit");
    assert!(Arc::ptr_eq(&loaded, &repaired), "reuse the loaded source");
    assert_eq!(fs::read(&params_path).unwrap(), parameter_bytes);
    assert_eq!(fs::read(&vk_path).unwrap(), vk_bytes);
    assert!(lock_path.is_file(), "VK repair uses the publication lock");

    let mut wrong_vk = loaded.vk().clone();
    wrong_vk.ic.pop();
    wrong_vk.write(File::create(&vk_path).unwrap()).unwrap();
    let wrong_bytes = fs::read(&vk_path).unwrap();
    assert!(get_zigzag_params::<ZigZagTree>(&config).is_err());
    assert_eq!(
        fs::read(&vk_path).unwrap(),
        wrong_bytes,
        "reject an existing mismatch"
    );
    assert_eq!(fs::read(&params_path).unwrap(), parameter_bytes);
    fs::write(&vk_path, vk_bytes).unwrap();
    assert!(Arc::ptr_eq(
        &loaded,
        &get_zigzag_params::<ZigZagTree>(&config).unwrap()
    ));
}
