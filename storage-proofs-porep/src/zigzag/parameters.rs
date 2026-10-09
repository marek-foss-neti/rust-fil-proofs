//! ZigZag's on-demand Groth16 source. The on-disk Bellperson format is unchanged.
//! Published parameter files must remain immutable while mapped (as for Bellperson).
#[cfg(not(feature = "cuda-supraseal"))]
use std::time::Instant;
use std::{
    fs::File,
    io::{self, Cursor, Seek, SeekFrom},
    ops::Range,
    path::Path,
    sync::{Arc, OnceLock},
};

use anyhow::{ensure, Context, Result};
use bellpepper_core::SynthesisError;
#[cfg(not(feature = "cuda-supraseal"))]
use bellperson::groth16::MappedParameters;
use bellperson::groth16::{ParameterSource, VerifyingKey};
use blstrs::{Bls12, G1Affine, G2Affine};
use fs2::FileExt;
use group::{prime::PrimeCurveAffine, UncompressedEncoding};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use storage_proofs_core::{error::Error, parameter_cache, settings::SETTINGS};

#[cfg(not(feature = "cuda-supraseal"))]
use super::measurements::{OperationDetails, OperationGuard};

#[cfg(all(test, not(feature = "cuda-supraseal")))]
mod tests;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterLoader {
    Compact,
    Mapped,
}

impl ParameterLoader {
    /// A process selects one representation. Switching environment variables must
    /// never retain both index representations in its cache.
    pub fn from_env() -> Result<Self> {
        static SELECTED: OnceLock<std::result::Result<ParameterLoader, String>> = OnceLock::new();
        SELECTED
            .get_or_init(
                || match std::env::var("FIL_PROOFS_ZIGZAG_PARAMETER_LOADER") {
                    Err(std::env::VarError::NotPresent) => Ok(Self::Compact),
                    Ok(value) if value == "compact" => Ok(Self::Compact),
                    Ok(value) if value == "mapped" => Ok(Self::Mapped),
                    _ => Err("ZigZag parameter loader must be compact or mapped".into()),
                },
            )
            .clone()
            .map_err(anyhow::Error::msg)
    }
}

/// One descriptor replaces a Vec<Range<usize>> regardless of family size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct QueryFamily {
    pub offset: usize,
    pub points: usize,
    pub encoded_point_bytes: usize,
}

impl QueryFamily {
    fn end(self) -> io::Result<usize> {
        self.points
            .checked_mul(self.encoded_point_bytes)
            .and_then(|bytes| self.offset.checked_add(bytes))
            .ok_or_else(|| invalid("parameter section offset overflow"))
    }

    fn range(self, index: usize) -> io::Result<Range<usize>> {
        if index >= self.points {
            return Err(invalid("parameter point index out of bounds"));
        }
        let start = index
            .checked_mul(self.encoded_point_bytes)
            .and_then(|bytes| self.offset.checked_add(bytes))
            .ok_or_else(|| invalid("parameter point offset overflow"))?;
        let end = start
            .checked_add(self.encoded_point_bytes)
            .ok_or_else(|| invalid("parameter point end overflow"))?;
        Ok(start..end)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ParameterLayout {
    pub file_bytes: usize,
    pub vk_bytes: usize,
    pub families: [QueryFamily; 5],
}

impl ParameterLayout {
    fn read_count(bytes: &[u8], offset: usize) -> io::Result<usize> {
        let end = offset
            .checked_add(4)
            .ok_or_else(|| invalid("count offset overflow"))?;
        let raw = bytes
            .get(offset..end)
            .ok_or_else(|| invalid("truncated parameter count"))?;
        Ok(u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize)
    }

    pub fn parse(bytes: &[u8]) -> io::Result<Self> {
        let g1 = G1Affine::identity().to_uncompressed().as_ref().len();
        let g2 = G2Affine::identity().to_uncompressed().as_ref().len();
        let fixed_vk = 3 * g1 + 3 * g2;
        let ic = Self::read_count(bytes, fixed_vk)?;
        let vk_bytes = QueryFamily {
            offset: fixed_vk + 4,
            points: ic,
            encoded_point_bytes: g1,
        }
        .end()?;
        if vk_bytes > bytes.len() {
            return Err(invalid("truncated verifying key inputs"));
        }
        let mut offset = vk_bytes;
        let mut families = [QueryFamily {
            offset: 0,
            points: 0,
            encoded_point_bytes: 0,
        }; 5];
        for (family, size) in families.iter_mut().zip([g1, g1, g1, g1, g2]) {
            let points = Self::read_count(bytes, offset)?;
            *family = QueryFamily {
                offset: offset
                    .checked_add(4)
                    .ok_or_else(|| invalid("section offset overflow"))?,
                points,
                encoded_point_bytes: size,
            };
            offset = family.end()?;
            if offset > bytes.len() {
                return Err(invalid("truncated parameter family"));
            }
        }
        if offset != bytes.len() {
            return Err(invalid("trailing parameter bytes"));
        }
        Ok(Self {
            file_bytes: bytes.len(),
            vk_bytes,
            families,
        })
    }

    pub fn legacy_index_bytes(&self) -> u64 {
        self.families
            .iter()
            .map(|family| family.points as u64)
            .sum::<u64>()
            * std::mem::size_of::<Range<usize>>() as u64
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct ParameterFileIdentity {
    pub bytes: u64,
    #[cfg(unix)]
    pub device: u64,
    #[cfg(unix)]
    pub inode: u64,
    #[cfg(unix)]
    pub modified: (i64, i64),
    #[cfg(unix)]
    pub changed: (i64, i64),
}

impl ParameterFileIdentity {
    fn from_file(file: &File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            bytes: metadata.len(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            #[cfg(unix)]
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }
}

/// No point-index vectors and no decoded query cache. File ownership pins the
/// verified inode; replacing its pathname cannot redirect subsequent reads.
pub struct CompactParameters {
    file: File,
    mmap: Mmap,
    pub vk: VerifyingKey<Bls12>,
    pub layout: ParameterLayout,
    pub identity: ParameterFileIdentity,
    pub parameter_id: String,
    pub vk_sha256: String,
    checked: bool,
}

impl CompactParameters {
    pub fn open(path: &Path, parameter_id: String, checked: bool) -> Result<Self> {
        let mut file = File::open(path)?;
        // Lock only while opening and validating: legacy readers acquire an
        // exclusive lock even for reads. Keep this descriptor after unlocking.
        FileExt::lock_shared(&file)?;
        let identity = ParameterFileIdentity::from_file(&file)?;
        if SETTINGS.verify_production_params {
            let key = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("invalid params filename")?;
            let data = parameter_cache::get_parameter_data_from_id(key)
                .ok_or_else(|| Error::InvalidParameters(path.display().to_string()))?;
            let mut hasher = blake2b_simd::Params::new().to_state();
            io::copy(&mut file, &mut hasher)?;
            ensure!(
                &hasher.finalize().to_hex()[..32] == data.digest,
                "invalid production ZigZag parameters"
            );
            file.seek(SeekFrom::Start(0))?;
        }
        // SAFETY: published parameter files remain immutable while mapped. The
        // held descriptor pins this inode even if its pathname is replaced.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let layout = ParameterLayout::parse(&mmap)?;
        // Validate the bounded prefix before VK decoding can allocate its IC.
        let vk = VerifyingKey::<Bls12>::read(Cursor::new(&mmap[..layout.vk_bytes]))?;
        let vk_sha256 = hex::encode(Sha256::digest(&mmap[..layout.vk_bytes]));
        ensure!(
            identity == ParameterFileIdentity::from_file(&file)?,
            "params changed during mapping"
        );
        FileExt::unlock(&file)?;
        Ok(Self {
            file,
            mmap,
            vk,
            layout,
            identity,
            parameter_id,
            vk_sha256,
            checked,
        })
    }

    pub fn matches_path(&self, path: &Path) -> bool {
        File::open(path)
            .and_then(|file| ParameterFileIdentity::from_file(&file))
            .ok()
            .as_ref()
            == Some(&self.identity)
            && ParameterFileIdentity::from_file(&self.file).ok().as_ref() == Some(&self.identity)
    }

    fn decode<C: UncompressedEncoding + PrimeCurveAffine + Send>(
        &self,
        family: usize,
    ) -> Result<Arc<Vec<C>>, SynthesisError> {
        let descriptor = self.layout.families[family];
        let points = (0..descriptor.points)
            .into_par_iter()
            .map(|index| {
                let range = descriptor.range(index)?;
                let bytes = self
                    .mmap
                    .get(range)
                    .ok_or_else(|| invalid("point outside mapping"))?;
                let mut repr = C::Uncompressed::default();
                if repr.as_ref().len() != bytes.len() {
                    return Err(invalid("wrong point encoding size"));
                }
                repr.as_mut().copy_from_slice(bytes);
                let value = if self.checked {
                    C::from_uncompressed(&repr)
                } else {
                    C::from_uncompressed_unchecked(&repr)
                };
                let point: C = Option::from(value).ok_or_else(|| invalid("not on curve"))?;
                if bool::from(point.is_identity()) {
                    return Err(invalid("point at infinity"));
                }
                Ok(point)
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Arc::new(points))
    }

    fn split<C: UncompressedEncoding + PrimeCurveAffine + Send>(
        &self,
        family: usize,
        inputs: usize,
    ) -> Result<((Arc<Vec<C>>, usize), (Arc<Vec<C>>, usize)), SynthesisError> {
        if inputs > self.layout.families[family].points {
            return Err(invalid("input query split out of bounds").into());
        }
        let points = self.decode(family)?;
        Ok(((points.clone(), 0), (points, inputs)))
    }
}

impl ParameterSource<Bls12> for &CompactParameters {
    type G1Builder = (Arc<Vec<G1Affine>>, usize);
    type G2Builder = (Arc<Vec<G2Affine>>, usize);
    fn get_vk(&self, inputs: usize) -> Result<&VerifyingKey<Bls12>, SynthesisError> {
        if inputs != self.vk.ic.len() {
            return Err(invalid("VK input count mismatch").into());
        }
        Ok(&self.vk)
    }
    fn get_h(&self, _: usize) -> Result<Self::G1Builder, SynthesisError> {
        Ok((self.decode(0)?, 0))
    }
    fn get_l(&self, _: usize) -> Result<Self::G1Builder, SynthesisError> {
        Ok((self.decode(1)?, 0))
    }
    fn get_a(
        &self,
        inputs: usize,
        _: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        self.split(2, inputs)
    }
    fn get_b_g1(
        &self,
        inputs: usize,
        _: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        self.split(3, inputs)
    }
    fn get_b_g2(
        &self,
        inputs: usize,
        _: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        self.split(4, inputs)
    }
}

/// Only the selected variant is constructed, before any legacy index allocation.
pub enum ZigZagParameters {
    #[cfg(not(feature = "cuda-supraseal"))]
    Compact(CompactParameters),
    #[cfg(not(feature = "cuda-supraseal"))]
    Mapped {
        params: MappedParameters<Bls12>,
        layout: ParameterLayout,
        identity: ParameterFileIdentity,
        parameter_id: String,
        vk_sha256: String,
    },
    #[cfg(feature = "cuda-supraseal")]
    Supraseal {
        params: bellperson::groth16::SuprasealParameters<Bls12>,
        vk: VerifyingKey<Bls12>,
        layout: ParameterLayout,
        identity: ParameterFileIdentity,
        parameter_id: String,
        vk_sha256: String,
        _file: File,
    },
}

impl ZigZagParameters {
    pub fn open(path: &Path, parameter_id: String, checked: bool) -> Result<Self> {
        #[cfg(feature = "cuda-supraseal")]
        {
            // The native backend owns its SRS and requires its concrete type.
            // Use the compact header validation without decoding any queries.
            let compact = CompactParameters::open(path, parameter_id, checked)?;
            let params = bellperson::groth16::SuprasealParameters::new(path.to_owned())?;
            ensure!(
                compact.matches_path(path),
                "Supraseal parameter file changed"
            );
            Ok(Self::Supraseal {
                params,
                vk: compact.vk,
                layout: compact.layout,
                identity: compact.identity,
                parameter_id: compact.parameter_id,
                vk_sha256: compact.vk_sha256,
                _file: compact.file,
            })
        }
        #[cfg(not(feature = "cuda-supraseal"))]
        {
            let loader = ParameterLoader::from_env()?;
            let compact = CompactParameters::open(path, parameter_id, checked)?;
            match loader {
                ParameterLoader::Compact => Ok(Self::Compact(compact)),
                ParameterLoader::Mapped => {
                    // The compact object is header-only: it never materializes queries.
                    let params = bellperson::groth16::Parameters::<Bls12>::build_mapped_parameters(
                        path.to_owned(),
                        checked,
                    )?;
                    ensure!(
                        params.vk == compact.vk
                            && ParameterFileIdentity::from_file(&params.param_file)?
                                == compact.identity,
                        "mapped parameter file changed"
                    );
                    Ok(Self::Mapped {
                        params,
                        layout: compact.layout,
                        identity: compact.identity,
                        parameter_id: compact.parameter_id,
                        vk_sha256: compact.vk_sha256,
                    })
                }
            }
        }
    }

    #[cfg(feature = "cuda-supraseal")]
    pub(crate) fn supraseal(&self) -> &bellperson::groth16::SuprasealParameters<Bls12> {
        match self {
            Self::Supraseal { params, .. } => params,
        }
    }

    pub fn vk(&self) -> &VerifyingKey<Bls12> {
        match self {
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Compact(params) => &params.vk,
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Mapped { params, .. } => &params.vk,
            #[cfg(feature = "cuda-supraseal")]
            Self::Supraseal { vk, .. } => vk,
        }
    }
    pub fn layout(&self) -> &ParameterLayout {
        match self {
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Compact(params) => &params.layout,
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Mapped { layout, .. } => layout,
            #[cfg(feature = "cuda-supraseal")]
            Self::Supraseal { layout, .. } => layout,
        }
    }
    pub fn matches_path(&self, path: &Path) -> bool {
        match self {
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Compact(params) => params.matches_path(path),
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Mapped {
                params, identity, ..
            } => {
                File::open(path)
                    .and_then(|file| ParameterFileIdentity::from_file(&file))
                    .ok()
                    .as_ref()
                    == Some(identity)
                    && ParameterFileIdentity::from_file(&params.param_file)
                        .ok()
                        .as_ref()
                        == Some(identity)
            }
            #[cfg(feature = "cuda-supraseal")]
            Self::Supraseal {
                _file, identity, ..
            } => {
                File::open(path)
                    .and_then(|file| ParameterFileIdentity::from_file(&file))
                    .ok()
                    .as_ref()
                    == Some(identity)
                    && ParameterFileIdentity::from_file(_file).ok().as_ref() == Some(identity)
            }
        }
    }
    pub fn diagnostics(&self) -> serde_json::Value {
        let layout = self.layout();
        let (identity, parameter_id, vk_sha256) = match self {
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Compact(params) => (&params.identity, &params.parameter_id, &params.vk_sha256),
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Mapped {
                identity,
                parameter_id,
                vk_sha256,
                ..
            } => (identity, parameter_id, vk_sha256),
            #[cfg(feature = "cuda-supraseal")]
            Self::Supraseal {
                identity,
                parameter_id,
                vk_sha256,
                ..
            } => (identity, parameter_id, vk_sha256),
        };
        let metadata_bytes = match self {
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Compact(_) => Some(std::mem::size_of::<[QueryFamily; 5]>() as u64),
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Mapped { params, .. } => Some(
                [
                    params.h.capacity(),
                    params.l.capacity(),
                    params.a.capacity(),
                    params.b_g1.capacity(),
                    params.b_g2.capacity(),
                ]
                .iter()
                .sum::<usize>() as u64
                    * std::mem::size_of::<Range<usize>>() as u64,
            ),
            // Native C++ query storage has no exposed Rust allocation size.
            #[cfg(feature = "cuda-supraseal")]
            Self::Supraseal { .. } => None::<u64>,
        };
        let loader = match self {
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Compact(_) => "compact",
            #[cfg(not(feature = "cuda-supraseal"))]
            Self::Mapped { .. } => "mapped",
            #[cfg(feature = "cuda-supraseal")]
            Self::Supraseal { .. } => "supraseal",
        };
        serde_json::json!({
            "kind": "layout", "loader": loader,
            "file_bytes": layout.file_bytes, "vk_bytes": layout.vk_bytes, "families": layout.families,
            "index_metadata_bytes": metadata_bytes, "legacy_index_element_bytes": layout.legacy_index_bytes(),
            "metadata_scope": if cfg!(feature = "cuda-supraseal") {
                "native Supraseal query storage; allocation size unavailable"
            } else {
                "query indices only; excludes VK, mmap residency and decoded queries"
            },
            "decoded_query_cache": if cfg!(feature = "cuda-supraseal") { None } else { Some(false) },
            "file_identity": identity, "parameter_id": parameter_id, "vk_sha256": vk_sha256,
        })
    }

    #[cfg(not(feature = "cuda-supraseal"))]
    fn query<T>(
        &self,
        family: usize,
        name: &'static str,
        read: impl FnOnce() -> Result<T, SynthesisError>,
    ) -> Result<T, SynthesisError> {
        let descriptor = self.layout().families[family];
        let decoded_point_bytes = if family == 4 {
            std::mem::size_of::<G2Affine>()
        } else {
            std::mem::size_of::<G1Affine>()
        };
        let details = OperationDetails {
            query_family: Some(["h", "l", "a", "b_g1", "b_g2"][family]),
            query_points: Some(descriptor.points),
            encoded_bytes: Some((descriptor.points as u64) * descriptor.encoded_point_bytes as u64),
            decoded_bytes: Some((descriptor.points as u64) * decoded_point_bytes as u64),
            ..Default::default()
        };
        let guard = OperationGuard::enter_with_details(name, None, details);
        let started = Instant::now();
        let result = read();
        log::info!(target: "zigzag_parameters", "zigzag_parameters {}", serde_json::json!({
            "kind": "query", "family": details.query_family, "points": descriptor.points,
            "encoded_bytes": details.encoded_bytes, "decoded_bytes": details.decoded_bytes,
            "wall_ms": started.elapsed().as_millis(), "completed": result.is_ok(),
        }));
        if result.is_ok() {
            guard.finish();
        }
        result
    }
}

#[cfg(not(feature = "cuda-supraseal"))]
impl ParameterSource<Bls12> for &ZigZagParameters {
    type G1Builder = (Arc<Vec<G1Affine>>, usize);
    type G2Builder = (Arc<Vec<G2Affine>>, usize);
    fn get_vk(&self, inputs: usize) -> Result<&VerifyingKey<Bls12>, SynthesisError> {
        if inputs != self.vk().ic.len() {
            return Err(invalid("VK input count mismatch").into());
        }
        Ok(self.vk())
    }
    fn get_h(&self, n: usize) -> Result<Self::G1Builder, SynthesisError> {
        self.query(0, "groth16_query_h", || match self {
            ZigZagParameters::Compact(p) => p.get_h(n),
            ZigZagParameters::Mapped { params, .. } => params.get_h(n),
        })
    }
    fn get_l(&self, n: usize) -> Result<Self::G1Builder, SynthesisError> {
        self.query(1, "groth16_query_l", || match self {
            ZigZagParameters::Compact(p) => p.get_l(n),
            ZigZagParameters::Mapped { params, .. } => params.get_l(n),
        })
    }
    fn get_a(
        &self,
        inputs: usize,
        aux: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        self.query(2, "groth16_query_a", || match self {
            ZigZagParameters::Compact(p) => p.get_a(inputs, aux),
            ZigZagParameters::Mapped { params, .. } => params.get_a(inputs, aux),
        })
    }
    fn get_b_g1(
        &self,
        inputs: usize,
        aux: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        self.query(3, "groth16_query_b_g1", || match self {
            ZigZagParameters::Compact(p) => p.get_b_g1(inputs, aux),
            ZigZagParameters::Mapped { params, .. } => params.get_b_g1(inputs, aux),
        })
    }
    fn get_b_g2(
        &self,
        inputs: usize,
        aux: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        self.query(4, "groth16_query_b_g2", || match self {
            ZigZagParameters::Compact(p) => p.get_b_g2(inputs, aux),
            ZigZagParameters::Mapped { params, .. } => params.get_b_g2(inputs, aux),
        })
    }
}
