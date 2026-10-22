//! Shared ZigZag device admission for Bellperson proving and Neptune TreeR.
//!
//! This module does not implement kernels or claim execution from device discovery.
//! Benchmark acceptance uses an external CUDA trace and CPU proof verification.
use std::{env, fs::File, time::Instant};

#[cfg(any(feature = "cuda", feature = "opencl"))]
use anyhow::Context;
use anyhow::{ensure, Result};
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Auto,
    Cpu,
    Cuda,
    Opencl,
}

impl Backend {
    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "opencl" => Self::Opencl,
            _ => anyhow::bail!("ZigZag GPU backend must be auto, cpu, cuda or opencl"),
        })
    }

    // Validate without probing a device or mutating global environment. Keep CPU
    // controls and explicit GPU requests testable on a host without a driver.
    fn validate(
        self,
        disabled: bool,
        framework: Option<&str>,
        per_device_lock: bool,
    ) -> Result<()> {
        let compiled = cfg!(any(feature = "cuda", feature = "opencl"));
        if self == Backend::Cpu {
            ensure!(
                !compiled || disabled,
                "CPU control in a GPU-capable build requires BELLMAN_NO_GPU=1 before process startup"
            );
        }
        if matches!(self, Backend::Cuda | Backend::Opencl) {
            ensure!(
                !per_device_lock,
                "dedicated ZigZag GPU workers require Bellperson's default global lock"
            );
            ensure!(
                !disabled,
                "requested ZigZag GPU backend conflicts with BELLMAN_NO_GPU"
            );
            ensure!(
                !cfg!(feature = "cuda-supraseal"),
                "ZigZag GPU admission supports native Bellperson, not cuda-supraseal"
            );
            ensure!(
                (self == Backend::Cuda && cfg!(feature = "cuda"))
                    || (self == Backend::Opencl && cfg!(feature = "opencl")),
                "requested ZigZag GPU backend is absent from this build"
            );
            ensure!(
                (self == Backend::Cuda && framework == Some("cuda"))
                    || (self == Backend::Opencl && framework == Some("opencl")),
                "set EC_GPU_FRAMEWORK to the requested backend before starting the worker"
            );
        }
        Ok(())
    }
}

/// Holds device admission for one C2 or TreeR worker, across bounded batches.
/// Bellperson's priority-aware file protocol remains in force inside each batch.
pub(crate) struct ProvingGuard {
    _lock: Option<File>,
}

impl ProvingGuard {
    pub(crate) fn enter() -> Result<Self> {
        Self::enter_for("groth16_c2")
    }

    pub(crate) fn enter_for(operation: &str) -> Result<Self> {
        let started = Instant::now();
        let requested =
            env::var("FIL_PROOFS_ZIGZAG_GPU_BACKEND").unwrap_or_else(|_| "auto".to_owned());
        let backend = Backend::parse(&requested)?;
        let disabled = env::var("BELLMAN_NO_GPU").is_ok_and(|v| v != "0");
        let framework = env::var("EC_GPU_FRAMEWORK").ok();
        backend.validate(
            disabled,
            framework.as_deref(),
            env::var_os("BELLPERSON_GPUS_PER_LOCK").is_some(),
        )?;
        let mut details = json!({
            "operation": operation,
            "requested_backend": requested,
            "cuda_compiled": cfg!(feature = "cuda"),
            "opencl_compiled": cfg!(feature = "opencl"),
            "bellman_no_gpu": disabled,
            "ec_gpu_framework": framework,
            "cpu_utilization": env::var("BELLMAN_CPU_UTILIZATION").ok(),
            "scope": "device admission only; kernel execution requires external trace",
        });
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        let mut lock = None;
        #[cfg(not(any(feature = "cuda", feature = "opencl")))]
        let lock = None;
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        if matches!(backend, Backend::Cuda | Backend::Opencl) {
            use ec_gpu_gen::rust_gpu_tools::Device;
            use fs2::FileExt;
            use std::{fs::OpenOptions, path::PathBuf};

            let devices = Device::all();
            ensure!(
                devices.len() == 1,
                "ZigZag GPU workers require exactly one visible device; found {}",
                devices.len()
            );
            details["devices"] = json!(devices.iter().map(|device| json!({
                "name": device.name(), "id": device.unique_id().to_string(),
                "vram_bytes": device.memory(), "compute_capability": device.compute_capability(),
            })).collect::<Vec<_>>());
            // Explicit GPU workers must share this directory across containers.
            // Auto never probes devices here and preserves legacy admission semantics.
            {
                let directory = PathBuf::from(env::var("FIL_PROOFS_ZIGZAG_GPU_LOCK_DIR").context(
                    "explicit ZigZag GPU worker requires a shared FIL_PROOFS_ZIGZAG_GPU_LOCK_DIR",
                )?);
                ensure!(
                    directory.is_absolute() && directory.is_dir(),
                    "GPU lock directory must be an existing absolute directory"
                );
                let device = devices[0];
                let minimum: u64 = env::var("FIL_PROOFS_ZIGZAG_GPU_MIN_VRAM_BYTES")
                    .unwrap_or_else(|_| "0".to_owned())
                    .parse()
                    .context("invalid minimum ZigZag VRAM")?;
                ensure!(
                    device.memory() >= minimum,
                    "selected GPU has less VRAM than the worker reservation"
                );
                let path = directory.join(format!("zigzag-{}.lock", device.unique_id()));
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&path)
                    .with_context(|| format!("open GPU admission lock {}", path.display()))?;
                file.lock_exclusive()
                    .context("acquire ZigZag GPU admission lock")?;
                details["lock_path"] = json!(path);
                details["reserved_device_vram_bytes"] = json!(device.memory());
                lock = Some(file);
            }
        }
        // No process-global environment mutation: SDR/PoSt may be running in other threads.
        details["admission_ms"] = json!(started.elapsed().as_millis());
        log::info!(target: "zigzag_gpu", "zigzag_gpu {}", details);
        Ok(Self { _lock: lock })
    }
}

/// Neptune does not participate in Bellperson's locks. Explicit TreeR workers
/// use the same device admission and global/priority lock namespace as C2/PoSt.
pub(crate) fn validate_tree_gpu_backend() -> Result<()> {
    let backend = Backend::parse(&env::var("FIL_PROOFS_ZIGZAG_GPU_BACKEND").unwrap_or_default())?;
    ensure!(
        matches!(backend, Backend::Cuda | Backend::Opencl),
        "GPU TreeR requires an explicit cuda or opencl ZigZag worker"
    );
    backend.validate(
        env::var("BELLMAN_NO_GPU").is_ok_and(|v| v != "0"),
        env::var("EC_GPU_FRAMEWORK").ok().as_deref(),
        env::var_os("BELLPERSON_GPUS_PER_LOCK").is_some(),
    )?;
    let directory = env::var("FIL_PROOFS_ZIGZAG_GPU_LOCK_DIR")?;
    ensure!(
        std::path::Path::new(&directory).is_absolute(),
        "GPU lock directory must be absolute"
    );
    ensure!(
        std::fs::canonicalize(directory)? == std::fs::canonicalize(env::temp_dir())?,
        "GPU TreeR and Bellperson must share TMPDIR and FIL_PROOFS_ZIGZAG_GPU_LOCK_DIR"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Backend;

    #[test]
    fn backend_selection_rejects_typos_and_empty_overrides() {
        for value in ["", "CUDA", "gpu", "cuda-supraseal", " cuda"] {
            assert!(Backend::parse(value).is_err());
        }
        assert_eq!(Backend::parse("cuda").unwrap(), Backend::Cuda);
        assert_eq!(Backend::parse("cpu").unwrap(), Backend::Cpu);
        assert_eq!(Backend::parse("opencl").unwrap(), Backend::Opencl);
        assert_eq!(Backend::parse("auto").unwrap(), Backend::Auto);
    }

    #[test]
    fn cpu_controls_require_disable_only_in_gpu_builds() {
        assert!(Backend::Auto.validate(false, None, false).is_ok());
        assert!(Backend::Cpu.validate(true, None, false).is_ok());
        assert_eq!(
            Backend::Cpu.validate(false, None, false).is_ok(),
            !cfg!(any(feature = "cuda", feature = "opencl"))
        );
    }

    #[test]
    fn explicit_gpu_requires_feature_framework_and_global_lock() {
        for (backend, framework, compiled) in [
            (Backend::Cuda, "cuda", cfg!(feature = "cuda")),
            (Backend::Opencl, "opencl", cfg!(feature = "opencl")),
        ] {
            assert!(backend.validate(true, Some(framework), false).is_err());
            assert!(backend.validate(false, None, false).is_err());
            assert!(backend.validate(false, Some("invalid"), false).is_err());
            assert!(backend.validate(false, Some(framework), true).is_err());
            assert_eq!(
                backend.validate(false, Some(framework), false).is_ok(),
                compiled && !cfg!(feature = "cuda-supraseal")
            );
        }
    }
}
