// ZigZag-only, file-backed adaptation of bellperson 0.27.0
// src/groth16/generator.rs and src/groth16/params.rs. The crates.io archive
// has SHA-256 cd3291d3e15b3de935a9f58574dafadbe15686e56e1c7850596b14f275cdae73.
// Original license: MIT OR Apache-2.0. Random draw order, R1CS indexing,
// QAP arithmetic, query order and point encoding match that release.

use std::convert::{TryFrom, TryInto};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::mem::size_of;
use std::ops::{AddAssign, Mul, MulAssign};
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use bellperson::domain::EvaluationDomain;
use bellperson::gpu;
use bellperson::groth16::VerifyingKey;
use bellperson::{Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use ec_gpu_gen::threadpool::Worker;
use ff::{Field, PrimeField};
use fs2::FileExt;
use group::{prime::PrimeCurveAffine, Curve, Group, UncompressedEncoding, Wnaf, WnafGroup};
use pairing::{Engine, MultiMillerLoop};
use rand_core::RngCore;
use sha2::{Digest, Sha256};

use super::groth16_setup::SetupLimits;

const RECORD_INDEX_BYTES: usize = 8;
const RECORD_ROW_BYTES: usize = 8;

pub(crate) enum SetupWorkspaceKind {
    Scratch,
    Publication,
}

impl SetupWorkspaceKind {
    fn prefix(&self) -> &'static str {
        match self {
            Self::Scratch => "zigzag-setup-work-v1-",
            Self::Publication => ".zigzag-setup-publish-v1-",
        }
    }
}

/// A live workspace holds an OS lock, so recovery does not rely on PIDs
/// (which can repeat or refer to a different container's PID namespace).
pub(crate) struct SetupWorkspace {
    directory: Option<tempfile::TempDir>,
    root: PathBuf,
    _owner: File,
}

impl SetupWorkspace {
    pub(crate) fn new(root: &Path, kind: SetupWorkspaceKind) -> Result<Self> {
        fs::create_dir_all(root)?;
        ensure!(
            fs::symlink_metadata(root)?.is_dir(),
            "setup workspace root is not a directory"
        );
        // Serialize recovery and creation, including the interval before a
        // newly created directory gets its owner lock. Never unlink this lock.
        let recovery = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join(".zigzag-setup-recovery-v1.lock"))?;
        recovery.lock_exclusive()?;
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            if !entry
                .file_name()
                .to_string_lossy()
                .starts_with(kind.prefix())
                || !entry.file_type()?.is_dir()
            {
                continue;
            }
            let path = entry.path();
            let owner_path = path.join(".owner.lock");
            let metadata = match fs::symlink_metadata(&owner_path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    // A kill between mkdir and owner-file creation leaves an
                    // empty directory. Never remove unmarked nonempty data.
                    if fs::read_dir(&path)?.next().is_none() {
                        fs::remove_dir(&path)?;
                    }
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                continue;
            }
            let owner = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&owner_path)?;
            match owner.try_lock_exclusive() {
                Ok(()) => {
                    fs::remove_dir_all(&path).context("remove orphaned ZigZag setup workspace")?;
                    log::info!("zigzag_setup recovered_orphan={}", path.display());
                }
                Err(error)
                    if error.raw_os_error() == fs2::lock_contended_error().raw_os_error() => {}
                Err(error) => return Err(error.into()),
            }
        }
        let directory = tempfile::Builder::new()
            .prefix(kind.prefix())
            .tempdir_in(root)?;
        let owner = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(directory.path().join(".owner.lock"))?;
        owner.lock_exclusive()?;
        Ok(Self {
            directory: Some(directory),
            root: root.to_owned(),
            _owner: owner,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        self.directory
            .as_ref()
            .expect("live setup workspace")
            .path()
    }
}

impl Drop for SetupWorkspace {
    fn drop(&mut self) {
        let directory = self.directory.take().expect("live setup workspace");
        let result = (|| -> io::Result<()> {
            let recovery = OpenOptions::new()
                .read(true)
                .write(true)
                .open(self.root.join(".zigzag-setup-recovery-v1.lock"))?;
            recovery.lock_exclusive()?;
            directory.close()
        })();
        if let Err(error) = result {
            log::warn!("could not remove ZigZag setup workspace: {}", error);
        }
    }
}

pub(crate) fn mark_phase(name: &str) -> io::Result<()> {
    if let Some(path) = std::env::var_os("FIL_PROOFS_ZIGZAG_SETUP_PHASE_FILE") {
        fs::write(path, name)?;
    }
    Ok(())
}

pub(crate) fn release_file_cache(file: &File) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let result =
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = file;
    Ok(())
}

#[derive(Clone, Copy)]
enum VariableKind {
    Input = 0,
    Aux = 1,
}

struct RecordWriter<F: PrimeField> {
    writer: BufWriter<File>,
    path: PathBuf,
    count: u64,
    digest: Sha256,
    _field: std::marker::PhantomData<F>,
}

impl<F: PrimeField> RecordWriter<F> {
    fn new(path: PathBuf) -> io::Result<Self> {
        Ok(Self {
            writer: BufWriter::with_capacity(1 << 20, File::create(&path)?),
            path,
            count: 0,
            digest: Sha256::new(),
            _field: std::marker::PhantomData,
        })
    }

    fn write(&mut self, kind: VariableKind, index: usize, row: usize, coeff: &F) -> io::Result<()> {
        let index = u64::try_from(index)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "variable index overflow"))?;
        let row = u64::try_from(row).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "constraint index overflow")
        })?;
        let repr = coeff.to_repr();
        let header = [kind as u8];
        let index = index.to_le_bytes();
        let row = row.to_le_bytes();
        for bytes in [&header[..], &index[..], &row[..], repr.as_ref()] {
            self.writer.write_all(bytes)?;
            self.digest.update(bytes);
        }
        self.count += 1;
        Ok(())
    }

    fn finish(mut self) -> io::Result<RecordFile> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(RecordFile {
            path: self.path,
            count: self.count,
            digest: hex::encode(self.digest.finalize()),
        })
    }
}

struct RecordFile {
    path: PathBuf,
    count: u64,
    digest: String,
}

struct DiskAssembly<F: PrimeField> {
    num_inputs: usize,
    num_aux: usize,
    num_constraints: usize,
    a: RecordWriter<F>,
    b: RecordWriter<F>,
    c: RecordWriter<F>,
    write_error: Option<io::Error>,
}

impl<F: PrimeField> DiskAssembly<F> {
    fn at(dir: &Path) -> io::Result<Self> {
        Ok(Self {
            num_inputs: 0,
            num_aux: 0,
            num_constraints: 0,
            a: RecordWriter::new(dir.join("constraints-a.bin"))?,
            b: RecordWriter::new(dir.join("constraints-b.bin"))?,
            c: RecordWriter::new(dir.join("constraints-c.bin"))?,
            write_error: None,
        })
    }

    fn finish(self) -> Result<(usize, usize, usize, [RecordFile; 3])> {
        if let Some(error) = self.write_error {
            return Err(error.into());
        }
        let files = [self.a.finish()?, self.b.finish()?, self.c.finish()?];
        Ok((self.num_inputs, self.num_aux, self.num_constraints, files))
    }
}

impl<F: PrimeField> ConstraintSystem<F> for DiskAssembly<F> {
    type Root = Self;

    fn is_extensible() -> bool {
        false
    }

    fn alloc<WF, A, AR>(&mut self, _: A, _: WF) -> std::result::Result<Variable, SynthesisError>
    where
        WF: FnOnce() -> std::result::Result<F, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let index = self.num_aux;
        self.num_aux += 1;
        Ok(Variable(Index::Aux(index)))
    }

    fn alloc_input<WF, A, AR>(
        &mut self,
        _: A,
        _: WF,
    ) -> std::result::Result<Variable, SynthesisError>
    where
        WF: FnOnce() -> std::result::Result<F, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let index = self.num_inputs;
        self.num_inputs += 1;
        Ok(Variable(Index::Input(index)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
        LB: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
        LC: FnOnce(LinearCombination<F>) -> LinearCombination<F>,
    {
        fn emit<F: PrimeField>(
            writer: &mut RecordWriter<F>,
            lc: LinearCombination<F>,
            row: usize,
        ) -> io::Result<()> {
            for (variable, coeff) in lc.iter() {
                let (kind, index) = match variable {
                    Variable(Index::Input(index)) => (VariableKind::Input, index),
                    Variable(Index::Aux(index)) => (VariableKind::Aux, index),
                };
                writer.write(kind, index, row, coeff)?;
            }
            Ok(())
        }
        if self.write_error.is_none() {
            let result = emit(
                &mut self.a,
                a(LinearCombination::zero()),
                self.num_constraints,
            )
            .and_then(|_| {
                emit(
                    &mut self.b,
                    b(LinearCombination::zero()),
                    self.num_constraints,
                )
            })
            .and_then(|_| {
                emit(
                    &mut self.c,
                    c(LinearCombination::zero()),
                    self.num_constraints,
                )
            });
            self.write_error = result.err();
        }
        self.num_constraints += 1;
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self) {}

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

fn accumulate<F: PrimeField>(
    file: &RecordFile,
    num_inputs: usize,
    lagrange: &[F],
    values: &mut [F],
) -> Result<()> {
    let mut reader = BufReader::with_capacity(1 << 20, File::open(&file.path)?);
    let mut hash = Sha256::new();
    let mut record =
        vec![0u8; 1 + RECORD_INDEX_BYTES + RECORD_ROW_BYTES + F::Repr::default().as_ref().len()];
    for _ in 0..file.count {
        reader.read_exact(&mut record)?;
        hash.update(&record);
        let index = usize::try_from(u64::from_le_bytes(record[1..9].try_into()?))?;
        let row = usize::try_from(u64::from_le_bytes(record[9..17].try_into()?))?;
        let target = match record[0] {
            0 => index,
            1 => num_inputs
                .checked_add(index)
                .context("auxiliary index overflow")?,
            _ => bail!("invalid ZigZag constraint record kind"),
        };
        let mut repr = F::Repr::default();
        repr.as_mut().copy_from_slice(&record[17..]);
        let coeff: F =
            Option::from(F::from_repr(repr)).context("invalid constraint coefficient")?;
        let lagrange_coeff = *lagrange.get(row).context("constraint row exceeds domain")?;
        let value = values
            .get_mut(target)
            .context("constraint variable exceeds count")?;
        *value += coeff * lagrange_coeff;
    }
    let mut extra = [0u8; 1];
    ensure!(
        reader.read(&mut extra)? == 0,
        "extra bytes in ZigZag constraint file"
    );
    ensure!(
        hex::encode(hash.finalize()) == file.digest,
        "ZigZag constraint digest mismatch"
    );
    Ok(())
}

struct Segment {
    path: PathBuf,
    writer: BufWriter<File>,
    count: u32,
    point_bytes: usize,
    digest: Sha256,
}

impl Segment {
    fn new(path: PathBuf, point_bytes: usize) -> io::Result<Self> {
        Ok(Self {
            writer: BufWriter::with_capacity(1 << 20, File::create(&path)?),
            path,
            count: 0,
            point_bytes,
            digest: Sha256::new(),
        })
    }

    fn push<G: UncompressedEncoding>(&mut self, point: &G) -> Result<()> {
        let repr = point.to_uncompressed();
        ensure!(
            repr.as_ref().len() == self.point_bytes,
            "point encoding length changed"
        );
        self.writer.write_all(repr.as_ref())?;
        self.digest.update(repr.as_ref());
        self.count = self
            .count
            .checked_add(1)
            .context("Groth16 query exceeds u32 length")?;
        Ok(())
    }

    fn finish(mut self) -> io::Result<FinishedSegment> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(FinishedSegment {
            path: self.path,
            count: self.count,
            point_bytes: self.point_bytes,
            digest: hex::encode(self.digest.finalize()),
        })
    }
}

struct FinishedSegment {
    path: PathBuf,
    count: u32,
    point_bytes: usize,
    digest: String,
}

impl FinishedSegment {
    fn append_data_to(&self, out: &mut impl Write) -> Result<()> {
        let expected = u64::from(self.count)
            .checked_mul(u64::try_from(self.point_bytes)?)
            .context("Groth16 query byte count overflow")?;
        ensure!(
            fs::metadata(&self.path)?.len() == expected,
            "Groth16 query segment has wrong size"
        );
        let mut reader = BufReader::with_capacity(1 << 20, File::open(&self.path)?);
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 1 << 20];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
            out.write_all(&buffer[..count])?;
        }
        ensure!(
            hex::encode(hash.finalize()) == self.digest,
            "Groth16 query segment digest mismatch"
        );
        if let Err(error) = release_file_cache(reader.get_ref()) {
            log::warn!(
                "could not release ZigZag query segment page cache: {}",
                error
            );
        }
        Ok(())
    }
}

fn append_segments(segments: &[FinishedSegment], out: &mut impl Write) -> Result<()> {
    let count = segments.iter().try_fold(0u32, |total, segment| {
        total
            .checked_add(segment.count)
            .context("Groth16 query count overflow")
    })?;
    out.write_all(&count.to_be_bytes())?;
    for segment in segments {
        segment.append_data_to(out)?;
    }
    Ok(())
}

fn split_range(len: usize, parts: usize, index: usize) -> (usize, usize) {
    let base = len / parts;
    let extra = len % parts;
    let start = base * index + index.min(extra);
    let end = start + base + usize::from(index < extra);
    (start, end)
}

fn check_scratch_disk(path: &Path, required_bytes: u64) -> Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let path = CString::new(path.as_os_str().as_bytes())?;
    #[cfg(target_os = "linux")]
    {
        let mut fs_info = unsafe { std::mem::zeroed::<libc::statfs>() };
        if unsafe { libc::statfs(path.as_ptr(), &mut fs_info) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        ensure!(
            fs_info.f_type as i64 != 0x0102_1994,
            "ZigZag setup scratch directory is tmpfs"
        );
    }
    let mut fs_info = unsafe { std::mem::zeroed::<libc::statvfs>() };
    if unsafe { libc::statvfs(path.as_ptr(), &mut fs_info) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let available = (fs_info.f_bavail as u64).saturating_mul(fs_info.f_frsize as u64);
    ensure!(
        available >= required_bytes,
        "ZigZag setup scratch needs {} bytes, only {} bytes available",
        required_bytes,
        available
    );
    Ok(())
}

fn checked_memory_floor<E: Engine>(m: usize, n: usize, limits: SetupLimits) -> Result<u64> {
    let field = u64::try_from(size_of::<E::Fr>())?;
    let g1 = u64::try_from(size_of::<E::G1>())?;
    let g2 = u64::try_from(size_of::<E::G2>())?;
    let affine_g1 = u64::try_from(size_of::<E::G1Affine>())?;
    let affine_g2 = u64::try_from(size_of::<E::G2Affine>())?;
    let m = u64::try_from(m)?;
    let n = u64::try_from(n)?;
    let b = u64::try_from(limits.batch_points)?;
    let w = u64::try_from(limits.workers)?;
    // Two domains cover the IFFT workspace; one Lagrange domain and three
    // field arrays coexist during QAP accumulation. These are lower bounds,
    // with headroom still needed for libraries and file cache. Query scratch
    // and I/O buffers are bounded by W times B.
    let domain = 2u64
        .checked_mul(m)
        .and_then(|x| x.checked_mul(field))
        .context("domain byte estimate overflow")?;
    let qap = 3u64
        .checked_mul(n)
        .and_then(|x| x.checked_add(m))
        .and_then(|x| x.checked_mul(field))
        .context("QAP byte estimate overflow")?;
    let batch = w
        .checked_mul(b)
        .and_then(|x| x.checked_mul(3 * (g1 + affine_g1) + g2 + affine_g2))
        .context("query batch byte estimate overflow")?;
    let io_buffers = w
        .checked_mul(5 * 1024 * 1024)
        .context("I/O buffer estimate overflow")?;
    let peak_floor = domain
        .max(qap)
        .checked_add(batch)
        .and_then(|x| x.checked_add(io_buffers))
        .context("setup byte estimate overflow")?;
    ensure!(peak_floor < limits.budget_bytes,
        "ZigZag setup lower-bound allocation {} exceeds configured budget {} (domain {}, QAP {}, batch {})",
        peak_floor, limits.budget_bytes, domain, qap, batch);
    log::info!("zigzag_setup memory_floor_bytes={} domain_bytes={} qap_bytes={} batch_bytes={} io_buffer_bytes={} budget_bytes={}",
        peak_floor, domain, qap, batch, io_buffers, limits.budget_bytes);
    Ok(peak_floor)
}

pub(crate) fn generate_random_parameters_to_file<E, C, R>(
    circuit: C,
    rng: &mut R,
    limits: SetupLimits,
    scratch_dir: &Path,
    out: &mut impl Write,
) -> Result<VerifyingKey<E>>
where
    E: MultiMillerLoop,
    E::G1: WnafGroup,
    E::G2: WnafGroup,
    E::Fr: gpu::GpuName,
    C: Circuit<E::Fr>,
    R: RngCore,
{
    // Keep precisely Bellperson 0.27.0's random draw order.
    let g1 = E::G1::random(&mut *rng);
    let g2 = E::G2::random(&mut *rng);
    let alpha = E::Fr::random(&mut *rng);
    let beta = E::Fr::random(&mut *rng);
    let gamma = E::Fr::random(&mut *rng);
    let delta = E::Fr::random(&mut *rng);
    let tau = E::Fr::random(&mut *rng);
    generate_parameters_to_file::<E, C>(
        circuit,
        g1,
        g2,
        alpha,
        beta,
        gamma,
        delta,
        tau,
        limits,
        scratch_dir,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_parameters_to_file<E, C>(
    circuit: C,
    g1: E::G1,
    g2: E::G2,
    alpha: E::Fr,
    beta: E::Fr,
    gamma: E::Fr,
    delta: E::Fr,
    tau: E::Fr,
    limits: SetupLimits,
    scratch_dir: &Path,
    out: &mut impl Write,
) -> Result<VerifyingKey<E>>
where
    E: MultiMillerLoop,
    E::G1: WnafGroup,
    E::G2: WnafGroup,
    E::Fr: gpu::GpuName,
    C: Circuit<E::Fr>,
{
    ensure!(
        limits.batch_points > 0 && limits.workers > 0,
        "ZigZag setup requires nonzero B/W"
    );
    fs::create_dir_all(scratch_dir)?;
    let work = SetupWorkspace::new(scratch_dir, SetupWorkspaceKind::Scratch)?;
    // Recover abandoned data before checking the free-space reserve: stale
    // scratch may itself be the reason that the filesystem is nearly full.
    check_scratch_disk(scratch_dir, 0)?;
    mark_phase("setup_synthesis")?;
    let started = Instant::now();
    let mut assembly = DiskAssembly::<E::Fr>::at(work.path())?;
    assembly.alloc_input(|| "", || Ok(E::Fr::ONE))?;
    circuit.synthesize(&mut assembly)?;
    for i in 0..assembly.num_inputs {
        assembly.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
    }
    let (num_inputs, num_aux, num_constraints, records) = assembly.finish()?;
    let n = num_inputs
        .checked_add(num_aux)
        .context("ZigZag variable count overflow")?;
    let m = num_constraints
        .checked_next_power_of_two()
        .context("ZigZag domain size overflow")?;
    let g1_point_bytes = g1.to_affine().to_uncompressed().as_ref().len();
    let g2_point_bytes = g2.to_affine().to_uncompressed().as_ref().len();
    let maximum_query_bytes = (m - 1)
        .checked_add(num_aux)
        .and_then(|x| n.checked_mul(2).and_then(|twice_n| x.checked_add(twice_n)))
        .and_then(|x| x.checked_mul(g1_point_bytes))
        .and_then(|x| x.checked_add(n.checked_mul(g2_point_bytes)?))
        .context("ZigZag query size estimate overflow")?;
    check_scratch_disk(
        scratch_dir,
        u64::try_from(maximum_query_bytes)?.saturating_mul(2),
    )?;
    checked_memory_floor::<E>(m, n, limits)?;
    log::info!("zigzag_setup phase=synthesis elapsed_ms={} inputs={} aux={} constraints={} domain={} records_a={} records_b={} records_c={}",
        started.elapsed().as_millis(), num_inputs, num_aux, num_constraints, m,
        records[0].count, records[1].count, records[2].count);

    let mut powers_of_tau = EvaluationDomain::from_coeffs(vec![E::Fr::ZERO; num_constraints])?;
    let mut g1_table = Wnaf::new();
    let g1_window_uses = (m - 1)
        .checked_add(n.checked_mul(3).context("G1 window count overflow")?)
        .context("G1 window count overflow")?;
    let g1_wnaf = g1_table.base(g1, g1_window_uses);
    let mut g2_table = Wnaf::new();
    let g2_wnaf = g2_table.base(g2, n);
    let gamma_inverse: E::Fr =
        Option::from(gamma.invert()).ok_or(SynthesisError::UnexpectedIdentity)?;
    let delta_inverse: E::Fr =
        Option::from(delta.invert()).ok_or(SynthesisError::UnexpectedIdentity)?;
    let worker = Worker::new();

    {
        let powers = powers_of_tau.as_mut();
        worker.scope(powers.len(), |scope, chunk| {
            for (i, part) in powers.chunks_mut(chunk).enumerate() {
                scope.execute(move || {
                    let mut current = tau.pow_vartime([(i * chunk) as u64]);
                    for p in part {
                        *p = current;
                        current.mul_assign(&tau);
                    }
                });
            }
        });
    }
    let mut h_coeff = powers_of_tau.z(&tau);
    h_coeff.mul_assign(&delta_inverse);
    let h_started = Instant::now();
    mark_phase("setup_h")?;
    let h_len = m - 1;
    let h_workers = limits.workers.min(h_len.max(1));
    let h_segments = std::thread::scope(|scope| -> Result<Vec<FinishedSegment>> {
        let mut handles = Vec::with_capacity(h_workers);
        for index in 0..h_workers {
            let (start, end) = split_range(h_len, h_workers, index);
            let powers = &powers_of_tau.as_ref()[start..end];
            let mut wnaf = g1_wnaf.shared();
            let path = work.path().join(format!("h-{index}.bin"));
            let batch_points = limits.batch_points;
            handles.push(scope.spawn(move || -> Result<FinishedSegment> {
                let mut segment = Segment::new(path, g1_point_bytes)?;
                let mut projective = Vec::with_capacity(batch_points);
                let mut affine = Vec::with_capacity(batch_points);
                for part in powers.chunks(batch_points) {
                    projective.clear();
                    for p in part {
                        projective.push(wnaf.scalar(&(*p * h_coeff)));
                    }
                    affine.resize(part.len(), E::G1Affine::identity());
                    E::G1::batch_normalize(&projective, &mut affine);
                    for point in &affine {
                        segment.push(point)?;
                    }
                }
                Ok(segment.finish()?)
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("ZigZag H worker panicked"))
            .collect::<Result<Vec<_>>>()
    })?;
    log::info!(
        "zigzag_setup phase=h elapsed_ms={} points={}",
        h_started.elapsed().as_millis(),
        h_segments
            .iter()
            .map(|segment| u64::from(segment.count))
            .sum::<u64>()
    );

    let ifft_started = Instant::now();
    mark_phase("setup_ifft")?;
    powers_of_tau.ifft(&worker, &mut None)?;
    let lagrange = powers_of_tau.into_coeffs();
    log::info!(
        "zigzag_setup phase=ifft elapsed_ms={}",
        ifft_started.elapsed().as_millis()
    );

    let eval_started = Instant::now();
    mark_phase("setup_qap")?;
    let mut at = vec![E::Fr::ZERO; n];
    let mut bt = vec![E::Fr::ZERO; n];
    let mut ct = vec![E::Fr::ZERO; n];
    accumulate(&records[0], num_inputs, &lagrange, &mut at)?;
    accumulate(&records[1], num_inputs, &lagrange, &mut bt)?;
    accumulate(&records[2], num_inputs, &lagrange, &mut ct)?;
    drop(lagrange);
    for record in &records {
        let file = File::open(&record.path)?;
        if let Err(error) = release_file_cache(&file) {
            log::warn!("could not release ZigZag constraint page cache: {}", error);
        }
        fs::remove_file(&record.path)?;
    }
    log::info!(
        "zigzag_setup phase=qap_accumulate elapsed_ms={}",
        eval_started.elapsed().as_millis()
    );

    let query_started = Instant::now();
    mark_phase("setup_eval_input_aux")?;
    let query_workers = limits.workers.min(n.max(1));
    let query_parts = std::thread::scope(
        |scope| -> Result<Vec<(Vec<E::G1Affine>, [FinishedSegment; 4])>> {
            let mut handles = Vec::with_capacity(query_workers);
            for worker_index in 0..query_workers {
                let (range_start, range_end) = split_range(n, query_workers, worker_index);
                let mut g1_wnaf = g1_wnaf.shared();
                let mut g2_wnaf = g2_wnaf.shared();
                let work_path = work.path().to_path_buf();
                let at = &at;
                let bt = &bt;
                let ct = &ct;
                let batch_points = limits.batch_points;
                handles.push(scope.spawn(
                    move || -> Result<(Vec<E::G1Affine>, [FinishedSegment; 4])> {
                        let mut l_segment = Segment::new(
                            work_path.join(format!("l-{worker_index}.bin")),
                            g1_point_bytes,
                        )?;
                        let mut a_segment = Segment::new(
                            work_path.join(format!("a-{worker_index}.bin")),
                            g1_point_bytes,
                        )?;
                        let mut b_g1_segment = Segment::new(
                            work_path.join(format!("b-g1-{worker_index}.bin")),
                            g1_point_bytes,
                        )?;
                        let mut b_g2_segment = Segment::new(
                            work_path.join(format!("b-g2-{worker_index}.bin")),
                            g2_point_bytes,
                        )?;
                        let mut ic = Vec::new();
                        let mut a_projective = Vec::with_capacity(batch_points);
                        let mut b_g1_projective = Vec::with_capacity(batch_points);
                        let mut b_g2_projective = Vec::with_capacity(batch_points);
                        let mut ext_projective = Vec::with_capacity(batch_points);
                        let mut a_affine = Vec::with_capacity(batch_points);
                        let mut b_g1_affine = Vec::with_capacity(batch_points);
                        let mut b_g2_affine = Vec::with_capacity(batch_points);
                        let mut ext_affine = Vec::with_capacity(batch_points);
                        for start in (range_start..range_end).step_by(batch_points) {
                            let end = range_end.min(start.saturating_add(batch_points));
                            let count = end - start;
                            a_projective.clear();
                            b_g1_projective.clear();
                            b_g2_projective.clear();
                            ext_projective.clear();
                            for index in start..end {
                                let mut a = at[index];
                                let mut b = bt[index];
                                let c = ct[index];
                                a_projective.push(if bool::from(a.is_zero()) {
                                    E::G1::identity()
                                } else {
                                    g1_wnaf.scalar(&a)
                                });
                                if bool::from(b.is_zero()) {
                                    b_g1_projective.push(E::G1::identity());
                                    b_g2_projective.push(E::G2::identity());
                                } else {
                                    b_g1_projective.push(g1_wnaf.scalar(&b));
                                    b_g2_projective.push(g2_wnaf.scalar(&b));
                                }
                                a.mul_assign(&beta);
                                b.mul_assign(&alpha);
                                let mut e = a;
                                e.add_assign(&b);
                                e.add_assign(&c);
                                e.mul_assign(if index < num_inputs {
                                    &gamma_inverse
                                } else {
                                    &delta_inverse
                                });
                                ext_projective.push(g1_wnaf.scalar(&e));
                            }
                            a_affine.resize(count, E::G1Affine::identity());
                            b_g1_affine.resize(count, E::G1Affine::identity());
                            b_g2_affine.resize(count, E::G2Affine::identity());
                            ext_affine.resize(count, E::G1Affine::identity());
                            E::G1::batch_normalize(&a_projective, &mut a_affine);
                            E::G1::batch_normalize(&b_g1_projective, &mut b_g1_affine);
                            E::G2::batch_normalize(&b_g2_projective, &mut b_g2_affine);
                            E::G1::batch_normalize(&ext_projective, &mut ext_affine);
                            for offset in 0..count {
                                if !bool::from(a_affine[offset].is_identity()) {
                                    a_segment.push(&a_affine[offset])?;
                                }
                                if !bool::from(b_g1_affine[offset].is_identity()) {
                                    b_g1_segment.push(&b_g1_affine[offset])?;
                                }
                                if !bool::from(b_g2_affine[offset].is_identity()) {
                                    b_g2_segment.push(&b_g2_affine[offset])?;
                                }
                                if start + offset < num_inputs {
                                    ic.push(ext_affine[offset]);
                                } else {
                                    ensure!(
                                        !bool::from(ext_affine[offset].is_identity()),
                                        "unconstrained ZigZag auxiliary variable"
                                    );
                                    l_segment.push(&ext_affine[offset])?;
                                }
                            }
                        }
                        Ok((
                            ic,
                            [
                                l_segment.finish()?,
                                a_segment.finish()?,
                                b_g1_segment.finish()?,
                                b_g2_segment.finish()?,
                            ],
                        ))
                    },
                ));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("ZigZag query worker panicked"))
                .collect::<Result<Vec<_>>>()
        },
    )?;
    drop((at, bt, ct));
    let mut ic = Vec::with_capacity(num_inputs);
    let mut l_segments = Vec::with_capacity(query_workers);
    let mut a_segments = Vec::with_capacity(query_workers);
    let mut b_g1_segments = Vec::with_capacity(query_workers);
    let mut b_g2_segments = Vec::with_capacity(query_workers);
    for (worker_ic, [l, a, b_g1, b_g2]) in query_parts {
        ic.extend(worker_ic);
        l_segments.push(l);
        a_segments.push(a);
        b_g1_segments.push(b_g1);
        b_g2_segments.push(b_g2);
    }
    log::info!(
        "zigzag_setup phase=eval_input_aux elapsed_ms={} l={} a={} b_g1={} b_g2={}",
        query_started.elapsed().as_millis(),
        l_segments.iter().map(|s| u64::from(s.count)).sum::<u64>(),
        a_segments.iter().map(|s| u64::from(s.count)).sum::<u64>(),
        b_g1_segments
            .iter()
            .map(|s| u64::from(s.count))
            .sum::<u64>(),
        b_g2_segments
            .iter()
            .map(|s| u64::from(s.count))
            .sum::<u64>()
    );

    let g1 = g1.to_affine();
    let g2 = g2.to_affine();
    let vk = VerifyingKey::<E> {
        alpha_g1: g1.mul(alpha).to_affine(),
        beta_g1: g1.mul(beta).to_affine(),
        beta_g2: g2.mul(beta).to_affine(),
        gamma_g2: g2.mul(gamma).to_affine(),
        delta_g1: g1.mul(delta).to_affine(),
        delta_g2: g2.mul(delta).to_affine(),
        ic,
    };
    let write_started = Instant::now();
    mark_phase("setup_write")?;
    vk.write(&mut *out)?;
    append_segments(&h_segments, out)?;
    append_segments(&l_segments, out)?;
    append_segments(&a_segments, out)?;
    append_segments(&b_g1_segments, out)?;
    append_segments(&b_g2_segments, out)?;
    out.flush()?;
    log::info!(
        "zigzag_setup phase=write elapsed_ms={}",
        write_started.elapsed().as_millis()
    );
    Ok(vk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::{Bls12, G1Projective, G2Projective, Scalar};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[derive(Clone)]
    struct TinyCircuit;

    impl Circuit<Scalar> for TinyCircuit {
        fn synthesize<CS: ConstraintSystem<Scalar>>(
            self,
            cs: &mut CS,
        ) -> std::result::Result<(), SynthesisError> {
            let x = cs.alloc(|| "x", || Ok(Scalar::from(3)))?;
            let y = cs.alloc(|| "y", || Ok(Scalar::from(5)))?;
            let z = cs.alloc_input(|| "z", || Ok(Scalar::from(15)))?;
            cs.enforce(|| "multiply", |lc| lc + x, |lc| lc + y, |lc| lc + z);
            cs.enforce(|| "x", |lc| lc + x, |lc| lc + CS::one(), |lc| lc + x);
            cs.enforce(|| "y", |lc| lc + y, |lc| lc + CS::one(), |lc| lc + y);
            Ok(())
        }
    }

    fn limits(batch_points: usize, workers: usize) -> SetupLimits {
        SetupLimits {
            batch_points,
            workers,
            budget_bytes: 1_000_000_000,
        }
    }

    #[test]
    fn disk_setup_matches_bellperson_for_partial_batches() {
        let g1 = G1Projective::generator();
        let g2 = G2Projective::generator();
        let (alpha, beta, gamma, delta, tau) = (
            Scalar::from(2),
            Scalar::from(3),
            Scalar::from(5),
            Scalar::from(7),
            Scalar::from(11),
        );
        let reference = bellperson::groth16::generate_parameters::<Bls12, _>(
            TinyCircuit,
            g1,
            g2,
            alpha,
            beta,
            gamma,
            delta,
            tau,
        )
        .expect("Bellperson setup");
        let mut reference_bytes = Vec::new();
        reference
            .write(&mut reference_bytes)
            .expect("serialize reference");
        assert!(
            reference.b_g1.len() < 4,
            "test circuit must exercise neutral B points"
        );
        let bounded = crate::zigzag::groth16_setup::generate_parameters::<Bls12, _>(
            TinyCircuit,
            g1,
            g2,
            alpha,
            beta,
            gamma,
            delta,
            tau,
            limits(2, 2),
        )
        .expect("bounded in-memory setup");
        let mut bounded_bytes = Vec::new();
        bounded
            .write(&mut bounded_bytes)
            .expect("serialize bounded setup");
        assert_eq!(bounded_bytes, reference_bytes);
        for batch in [1, 2, 3, 7] {
            for workers in [1, 2] {
                let scratch = tempfile::tempdir().expect("scratch");
                let mut output = Vec::new();
                let vk = generate_parameters_to_file::<Bls12, _>(
                    TinyCircuit,
                    g1,
                    g2,
                    alpha,
                    beta,
                    gamma,
                    delta,
                    tau,
                    limits(batch, workers),
                    scratch.path(),
                    &mut output,
                )
                .expect("streaming setup");
                assert_eq!(output, reference_bytes, "batch={batch} workers={workers}");
                assert_eq!(vk, reference.vk);
                let parsed = bellperson::groth16::Parameters::<Bls12>::read(&output[..], true)
                    .expect("Bellperson parser");
                assert!(parsed == reference);
            }
        }
    }

    #[test]
    fn disk_setup_preserves_random_draw_order() {
        let seed = [27u8; 16];
        let mut reference_rng = XorShiftRng::from_seed(seed);
        let mut disk_rng = XorShiftRng::from_seed(seed);
        let reference = bellperson::groth16::generate_random_parameters::<Bls12, _, _>(
            TinyCircuit,
            &mut reference_rng,
        )
        .expect("Bellperson random setup");
        let scratch = tempfile::tempdir().expect("scratch");
        let mut output = Vec::new();
        generate_random_parameters_to_file::<Bls12, _, _>(
            TinyCircuit,
            &mut disk_rng,
            limits(2, 2),
            scratch.path(),
            &mut output,
        )
        .expect("streaming random setup");
        let mut expected = Vec::new();
        reference.write(&mut expected).expect("serialize reference");
        assert_eq!(output, expected);
    }

    #[test]
    fn interrupted_write_cleans_scratch_and_retry_succeeds() {
        struct BrokenWriter;
        impl Write for BrokenWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "simulated interruption",
                ))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let scratch = tempfile::tempdir().expect("scratch");
        let args = (
            G1Projective::generator(),
            G2Projective::generator(),
            Scalar::from(2),
            Scalar::from(3),
            Scalar::from(5),
            Scalar::from(7),
            Scalar::from(11),
        );
        assert!(generate_parameters_to_file::<Bls12, _>(
            TinyCircuit,
            args.0,
            args.1,
            args.2,
            args.3,
            args.4,
            args.5,
            args.6,
            limits(2, 2),
            scratch.path(),
            &mut BrokenWriter,
        )
        .is_err());
        assert_eq!(workspace_directories(scratch.path()).len(), 0);
        let mut result = Vec::new();
        generate_parameters_to_file::<Bls12, _>(
            TinyCircuit,
            args.0,
            args.1,
            args.2,
            args.3,
            args.4,
            args.5,
            args.6,
            limits(2, 2),
            scratch.path(),
            &mut result,
        )
        .expect("retry setup");
        assert!(!result.is_empty());
        assert_eq!(workspace_directories(scratch.path()).len(), 0);
    }

    fn workspace_directories(root: &Path) -> Vec<PathBuf> {
        fs::read_dir(root)
            .expect("read workspaces")
            .map(|entry| entry.expect("workspace entry"))
            .filter(|entry| {
                entry.file_type().expect("workspace type").is_dir()
                    && (entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with("zigzag-setup-work-v1-")
                        || entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with(".zigzag-setup-publish-v1-"))
            })
            .map(|entry| entry.path())
            .collect()
    }

    // Invoked in a separate process by the SIGKILL regression test below.
    #[test]
    fn killed_setup_workspace_child() {
        let Some(root) = std::env::var_os("FIL_PROOFS_TEST_KILL_WORKSPACE_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let publication =
            SetupWorkspace::new(&root.join("publication"), SetupWorkspaceKind::Publication)
                .expect("publication workspace");
        let mut pending =
            tempfile::NamedTempFile::new_in(publication.path()).expect("pending params");
        struct BlockingWriter<'a> {
            pending: &'a mut tempfile::NamedTempFile,
            ready: PathBuf,
        }
        impl Write for BlockingWriter<'_> {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.pending.write_all(bytes)?;
                self.pending.flush()?;
                fs::write(&self.ready, "ready")?;
                loop {
                    std::thread::park();
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                self.pending.flush()
            }
        }
        let mut rng = XorShiftRng::from_seed([27u8; 16]);
        generate_random_parameters_to_file::<Bls12, _, _>(
            TinyCircuit,
            &mut rng,
            limits(2, 2),
            &root.join("scratch"),
            &mut BlockingWriter {
                pending: &mut pending,
                ready: root.join("ready"),
            },
        )
        .expect("setup must remain blocked until killed");
    }

    #[cfg(unix)]
    #[test]
    fn sigkill_recovery_removes_orphans_and_preserves_active_workspaces() {
        let root = tempfile::tempdir().expect("kill test root");
        let mut child =
            std::process::Command::new(std::env::current_exe().expect("test executable"))
                .args([
                    "--exact",
                    "zigzag::groth16_setup_disk::tests::killed_setup_workspace_child",
                    "--nocapture",
                ])
                .env("FIL_PROOFS_TEST_KILL_WORKSPACE_ROOT", root.path())
                .stdout(std::process::Stdio::null())
                .spawn()
                .expect("spawn setup process");
        let deadline = Instant::now() + std::time::Duration::from_secs(30);
        while !root.path().join("ready").is_file() {
            if Instant::now() >= deadline || child.try_wait().expect("child status").is_some() {
                let _ = child.kill();
                let _ = child.wait();
                panic!("setup child did not reach partial parameter write");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let scratch = root.path().join("scratch");
        let publication = root.path().join("publication");
        let original_scratch = workspace_directories(&scratch)
            .pop()
            .expect("child scratch");
        let original_publication = workspace_directories(&publication)
            .pop()
            .expect("child publication");
        let active_scratch = SetupWorkspace::new(&scratch, SetupWorkspaceKind::Scratch)
            .expect("active scratch survives");
        let active_publication = SetupWorkspace::new(&publication, SetupWorkspaceKind::Publication)
            .expect("active pending params survive");
        assert!(original_scratch.is_dir() && original_publication.is_dir());
        drop((active_scratch, active_publication));
        child.kill().expect("SIGKILL setup process");
        assert!(!child.wait().expect("wait for SIGKILL").success());
        assert!(original_scratch.is_dir() && original_publication.is_dir());

        let unrelated = scratch.join("zigzag-setup-legacy-unmarked");
        fs::create_dir(&unrelated).expect("legacy directory");
        fs::write(unrelated.join("keep"), "unowned").expect("legacy file");
        let outside = root.path().join("outside");
        fs::create_dir(&outside).expect("outside directory");
        fs::write(outside.join("keep"), "unrelated").expect("outside file");
        std::os::unix::fs::symlink(&outside, scratch.join("zigzag-setup-work-v1-symlink"))
            .expect("unrelated symlink");
        let retry_publication = SetupWorkspace::new(&publication, SetupWorkspaceKind::Publication)
            .expect("recover pending params");
        let mut rng = XorShiftRng::from_seed([27u8; 16]);
        let mut output = Vec::new();
        generate_random_parameters_to_file::<Bls12, _, _>(
            TinyCircuit,
            &mut rng,
            limits(2, 2),
            &scratch,
            &mut output,
        )
        .expect("retry after SIGKILL");
        assert!(!original_scratch.exists() && !original_publication.exists());
        assert!(unrelated.join("keep").is_file() && outside.join("keep").is_file());
        assert!(bellperson::groth16::Parameters::<Bls12>::read(&output[..], true).is_ok());
        drop(retry_publication);
        assert!(workspace_directories(&scratch).is_empty());
        assert!(workspace_directories(&publication).is_empty());
    }

    #[test]
    fn small_zigzag_blank_circuit_matches_bellperson() {
        use crate::zigzag::{
            circuit::ZigZagCompound,
            vanilla::{LayerChallenges, SetupParams, EXP_DEGREE},
        };
        use filecoin_hashers::{
            poseidon::{PoseidonDomain, PoseidonHasher},
            sha256::Sha256Hasher,
        };
        use generic_array::typenum::{U0, U2};
        use storage_proofs_core::{
            api_version::ApiVersion,
            compound_proof::{self, CompoundProof},
            drgraph::BASE_DEGREE,
            merkle::{DiskStore, MerkleTreeWrapper},
        };
        type Tree = MerkleTreeWrapper<PoseidonHasher, DiskStore<PoseidonDomain>, U2, U0, U0>;
        type Compound = ZigZagCompound<Tree, Sha256Hasher>;
        let params = Compound::setup(&compound_proof::SetupParams {
            vanilla_params: SetupParams {
                nodes: 128,
                degree: BASE_DEGREE,
                expansion_degree: EXP_DEGREE,
                porep_id: [23u8; 32],
                api_version: ApiVersion::V1_2_0,
                layer_challenges: LayerChallenges::new_fixed(1, 1),
            },
            partitions: Some(1),
            priority: false,
        })
        .expect("ZigZag setup");
        let circuit = Compound::blank_circuit(&params.vanilla_params);
        let mut reference_rng = XorShiftRng::from_seed([37u8; 16]);
        let mut disk_rng = XorShiftRng::from_seed([37u8; 16]);
        let reference = bellperson::groth16::generate_random_parameters::<Bls12, _, _>(
            circuit.clone(),
            &mut reference_rng,
        )
        .expect("Bellperson ZigZag parameters");
        let scratch = tempfile::tempdir().expect("scratch");
        let mut output = Vec::new();
        generate_random_parameters_to_file::<Bls12, _, _>(
            circuit,
            &mut disk_rng,
            limits(17, 2),
            scratch.path(),
            &mut output,
        )
        .expect("streaming ZigZag parameters");
        let mut expected = Vec::new();
        reference.write(&mut expected).expect("serialize reference");
        assert_eq!(output, expected);
    }
}
