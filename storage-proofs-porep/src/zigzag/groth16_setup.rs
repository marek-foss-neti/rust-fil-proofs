// Adapted from bellperson 0.27.0 src/groth16/generator.rs (crates.io SHA-256
// cd3291d3e15b3de935a9f58574dafadbe15686e56e1c7850596b14f275cdae73).
// Bellperson is dual licensed under MIT OR Apache-2.0. The original arithmetic,
// random draw order, constraint indexing, and parameter layout are preserved.
// ZigZag bounds temporary projective buffers and keeps the SDR generator intact.
// This intermediate generator remains for Bellperson equivalence tests; the
// file-backed generator is used for cache misses in production.
#![allow(dead_code)]
use std::ops::{AddAssign, Mul, MulAssign};
use std::time::Instant;

use std::sync::Arc;

use ff::{Field, PrimeField};
use group::{
    prime::{PrimeCurve, PrimeCurveAffine},
    Curve, Group, Wnaf, WnafGroup,
};
use pairing::{Engine, MultiMillerLoop};
use rand_core::RngCore;

use bellperson::groth16::{Parameters, VerifyingKey};

use bellpepper_core::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use bellperson::domain::EvaluationDomain;
use bellperson::gpu;
use ec_gpu_gen::threadpool::Worker;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SetupLimits {
    pub batch_points: usize,
    pub workers: usize,
    pub budget_bytes: u64,
}

impl SetupLimits {
    pub(crate) fn from_env() -> anyhow::Result<Self> {
        fn read(name: &str, default: u64) -> anyhow::Result<u64> {
            match std::env::var(name) {
                Ok(value) => Ok(value.parse()?),
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(error) => Err(error.into()),
            }
        }
        let batch_points = read("FIL_PROOFS_ZIGZAG_SETUP_BATCH_POINTS", 65_536)?;
        let workers = read("FIL_PROOFS_ZIGZAG_SETUP_WORKERS", 2)?;
        let budget_bytes = read("FIL_PROOFS_ZIGZAG_SETUP_BUDGET_BYTES", 100_000_000_000)?;
        anyhow::ensure!(
            batch_points > 0 && batch_points <= usize::MAX as u64,
            "invalid ZigZag setup batch size"
        );
        anyhow::ensure!(
            workers > 0 && workers <= 32,
            "ZigZag setup workers must be 1..=32"
        );
        anyhow::ensure!(budget_bytes > 0, "invalid ZigZag setup memory budget");
        Ok(Self {
            batch_points: batch_points as usize,
            workers: workers as usize,
            budget_bytes,
        })
    }
}

/// Generates a random common reference string for
/// a circuit.
pub fn generate_random_parameters<E, C, R>(
    circuit: C,
    rng: &mut R,
    limits: SetupLimits,
) -> Result<Parameters<E>, SynthesisError>
where
    E: MultiMillerLoop,
    <E as Engine>::G1: WnafGroup,
    <E as Engine>::G2: WnafGroup,
    C: Circuit<E::Fr>,
    R: RngCore,
    E::Fr: gpu::GpuName,
{
    let g1 = E::G1::random(&mut *rng);
    let g2 = E::G2::random(&mut *rng);
    let alpha = E::Fr::random(&mut *rng);
    let beta = E::Fr::random(&mut *rng);
    let gamma = E::Fr::random(&mut *rng);
    let delta = E::Fr::random(&mut *rng);
    let tau = E::Fr::random(&mut *rng);

    generate_parameters::<E, C>(circuit, g1, g2, alpha, beta, gamma, delta, tau, limits)
}

/// This is our assembly structure that we'll use to synthesize the
/// circuit into a QAP.
struct KeypairAssembly<Scalar: PrimeField> {
    num_inputs: usize,
    num_aux: usize,
    num_constraints: usize,
    at_inputs: Vec<Vec<(Scalar, usize)>>,
    bt_inputs: Vec<Vec<(Scalar, usize)>>,
    ct_inputs: Vec<Vec<(Scalar, usize)>>,
    at_aux: Vec<Vec<(Scalar, usize)>>,
    bt_aux: Vec<Vec<(Scalar, usize)>>,
    ct_aux: Vec<Vec<(Scalar, usize)>>,
}

impl<Scalar: PrimeField> ConstraintSystem<Scalar> for KeypairAssembly<Scalar> {
    type Root = Self;

    fn new() -> Self {
        KeypairAssembly {
            num_inputs: 0,
            num_aux: 0,
            num_constraints: 0,
            at_inputs: vec![],
            bt_inputs: vec![],
            ct_inputs: vec![],
            at_aux: vec![],
            bt_aux: vec![],
            ct_aux: vec![],
        }
    }

    /// Explicitly declare this `ConstraintSystem` is not extensible as a reminder to future implementers.
    /// By forbidding use of `ConstraintSystem::extend` when generating Groth parameters, we enforce
    /// the requirement of a well-defined sequential circuit synthesis. This also means we know that any
    /// synthesized `ProvingAssignment` is well-formed if it leads to a verifiable proof using the resulting
    /// groth parameters and verifying key. This is true even if the `ProvingAssignment` was synthesized
    /// in parallel components which were then joined by `ConstraintSystem::extend`.
    fn is_extensible() -> bool {
        false
    }

    fn alloc<F, A, AR>(&mut self, _: A, _: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // There is no assignment, so we don't even invoke the
        // function for obtaining one.

        let index = self.num_aux;
        self.num_aux += 1;

        self.at_aux.push(vec![]);
        self.bt_aux.push(vec![]);
        self.ct_aux.push(vec![]);

        Ok(Variable(Index::Aux(index)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, _: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // There is no assignment, so we don't even invoke the
        // function for obtaining one.

        let index = self.num_inputs;
        self.num_inputs += 1;

        self.at_inputs.push(vec![]);
        self.bt_inputs.push(vec![]);
        self.ct_inputs.push(vec![]);

        Ok(Variable(Index::Input(index)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
        fn eval<Scalar: PrimeField>(
            l: LinearCombination<Scalar>,
            inputs: &mut [Vec<(Scalar, usize)>],
            aux: &mut [Vec<(Scalar, usize)>],
            this_constraint: usize,
        ) {
            for (index, coeff) in l.iter() {
                match index {
                    Variable(Index::Input(id)) => inputs[id].push((*coeff, this_constraint)),
                    Variable(Index::Aux(id)) => aux[id].push((*coeff, this_constraint)),
                }
            }
        }

        eval(
            a(LinearCombination::zero()),
            &mut self.at_inputs,
            &mut self.at_aux,
            self.num_constraints,
        );
        eval(
            b(LinearCombination::zero()),
            &mut self.bt_inputs,
            &mut self.bt_aux,
            self.num_constraints,
        );
        eval(
            c(LinearCombination::zero()),
            &mut self.ct_inputs,
            &mut self.ct_aux,
            self.num_constraints,
        );

        self.num_constraints += 1;
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

/// Create parameters for a circuit, given some toxic waste.
#[allow(clippy::too_many_arguments)]
pub fn generate_parameters<E, C>(
    circuit: C,
    g1: E::G1,
    g2: E::G2,
    alpha: E::Fr,
    beta: E::Fr,
    gamma: E::Fr,
    delta: E::Fr,
    tau: E::Fr,
    limits: SetupLimits,
) -> Result<Parameters<E>, SynthesisError>
where
    E: MultiMillerLoop,
    <E as Engine>::G1: WnafGroup,
    <E as Engine>::G2: WnafGroup,
    C: Circuit<E::Fr>,
    E::Fr: gpu::GpuName,
{
    let started = Instant::now();
    let mut assembly = KeypairAssembly::new();

    // Allocate the "one" input variable
    assembly.alloc_input(|| "", || Ok(<E::Fr as Field>::ONE))?;

    // Synthesize the circuit.
    circuit.synthesize(&mut assembly)?;

    // Input constraints to ensure full density of IC query
    // x * 0 = 0
    for i in 0..assembly.num_inputs {
        assembly.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
    }
    log::info!(
        "zigzag_setup phase=synthesis elapsed_ms={} inputs={} aux={} constraints={}",
        started.elapsed().as_millis(),
        assembly.num_inputs,
        assembly.num_aux,
        assembly.num_constraints
    );

    // Create bases for blind evaluation of polynomials at tau
    let powers_of_tau = vec![E::Fr::ZERO; assembly.num_constraints];
    let mut powers_of_tau = EvaluationDomain::from_coeffs(powers_of_tau)?;

    // Compute G1 window table
    let mut g1_wnaf = Wnaf::new();
    let g1_wnaf = g1_wnaf.base(g1, {
        // H query
        (powers_of_tau.as_ref().len() - 1)
        // IC/L queries
        + assembly.num_inputs + assembly.num_aux
        // A query
        + assembly.num_inputs + assembly.num_aux
        // B query
        + assembly.num_inputs + assembly.num_aux
    });

    // Compute G2 window table
    let mut g2_wnaf = Wnaf::new();
    let g2_wnaf = g2_wnaf.base(g2, {
        // B query
        assembly.num_inputs + assembly.num_aux
    });

    let gamma_inverse: E::Fr =
        Option::from(gamma.invert()).ok_or(SynthesisError::UnexpectedIdentity)?;
    let delta_inverse = Option::from(delta.invert()).ok_or(SynthesisError::UnexpectedIdentity)?;

    let worker = Worker::new();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(limits.workers)
        .build()
        .expect("ZigZag setup worker pool");

    let mut h_affine =
        vec![<E::G1 as PrimeCurve>::Affine::identity(); powers_of_tau.as_ref().len() - 1];
    {
        // Compute powers of tau
        {
            let powers_of_tau = powers_of_tau.as_mut();
            worker.scope(powers_of_tau.len(), |scope, chunk| {
                for (i, powers_of_tau) in powers_of_tau.chunks_mut(chunk).enumerate() {
                    scope.execute(move || {
                        let mut current_tau_power = tau.pow_vartime([(i * chunk) as u64]);

                        for p in powers_of_tau {
                            *p = current_tau_power;
                            current_tau_power.mul_assign(&tau);
                        }
                    });
                }
            });
        }

        // coeff = t(x) / delta
        let mut coeff = powers_of_tau.z(&tau);
        coeff.mul_assign(&delta_inverse);

        let h_started = Instant::now();
        pool.install(|| {
            h_affine
                .par_chunks_mut(limits.batch_points)
                .enumerate()
                .for_each(|(i, out)| {
                    let start = i * limits.batch_points;
                    let mut wnaf = g1_wnaf.shared();
                    let h: Vec<_> = powers_of_tau.as_ref()[start..start + out.len()]
                        .iter()
                        .map(|p| wnaf.scalar(&(*p * coeff)))
                        .collect();
                    E::G1::batch_normalize(&h, out);
                });
        });
        log::info!(
            "zigzag_setup phase=h elapsed_ms={} batch_points={} workers={}",
            h_started.elapsed().as_millis(),
            limits.batch_points,
            limits.workers
        );
    }

    // Use inverse FFT to convert powers of tau to Lagrange coefficients
    let ifft_started = Instant::now();
    powers_of_tau.ifft(&worker, &mut None)?;
    let powers_of_tau = powers_of_tau.into_coeffs();
    log::info!(
        "zigzag_setup phase=ifft elapsed_ms={}",
        ifft_started.elapsed().as_millis()
    );

    let mut a_affine =
        vec![<E::G1 as PrimeCurve>::Affine::identity(); assembly.num_inputs + assembly.num_aux];
    let mut b_g1_affine =
        vec![<E::G1 as PrimeCurve>::Affine::identity(); assembly.num_inputs + assembly.num_aux];
    let mut b_g2_affine =
        vec![<E::G2 as PrimeCurve>::Affine::identity(); assembly.num_inputs + assembly.num_aux];
    let mut ic_affine = vec![<E::G1 as PrimeCurve>::Affine::identity(); assembly.num_inputs];
    let mut l_affine = vec![<E::G1 as PrimeCurve>::Affine::identity(); assembly.num_aux];

    #[allow(clippy::too_many_arguments)]
    fn eval<E: Engine>(
        // wNAF window tables
        g1_wnaf: &Wnaf<usize, &[E::G1], &mut Vec<i64>>,
        g2_wnaf: &Wnaf<usize, &[E::G2], &mut Vec<i64>>,

        // Lagrange coefficients for tau
        powers_of_tau: &[E::Fr],

        // QAP polynomials
        at: &[Vec<(E::Fr, usize)>],
        bt: &[Vec<(E::Fr, usize)>],
        ct: &[Vec<(E::Fr, usize)>],

        // Resulting evaluated QAP polynomials
        a_affine: &mut [E::G1Affine],
        b_g1_affine: &mut [E::G1Affine],
        b_g2_affine: &mut [E::G2Affine],
        ext_affine: &mut [E::G1Affine],

        // Inverse coefficient for ext elements
        inv: &E::Fr,

        // Trapdoors
        alpha: &E::Fr,
        beta: &E::Fr,

        // Worker
        pool: &rayon::ThreadPool,
        batch_points: usize,
    ) {
        // Sanity check
        assert_eq!(a_affine.len(), at.len());
        assert_eq!(a_affine.len(), bt.len());
        assert_eq!(a_affine.len(), ct.len());
        assert_eq!(a_affine.len(), b_g1_affine.len());
        assert_eq!(a_affine.len(), b_g2_affine.len());
        assert_eq!(a_affine.len(), ext_affine.len());

        // Each pool worker allocates at most one batch of projective scratch.
        pool.install(|| {
            a_affine
                .par_chunks_mut(batch_points)
                .zip(b_g1_affine.par_chunks_mut(batch_points))
                .zip(b_g2_affine.par_chunks_mut(batch_points))
                .zip(ext_affine.par_chunks_mut(batch_points))
                .zip(at.par_chunks(batch_points))
                .zip(bt.par_chunks(batch_points))
                .zip(ct.par_chunks(batch_points))
                .for_each(
                    |((((((a_affine, b_g1_affine), b_g2_affine), ext_affine), at), bt), ct)| {
                        let mut g1_wnaf = g1_wnaf.shared();
                        let mut g2_wnaf = g2_wnaf.shared();

                        let mut a = vec![E::G1::identity(); a_affine.len()];
                        let mut b_g1 = vec![E::G1::identity(); a_affine.len()];
                        let mut b_g2 = vec![E::G2::identity(); a_affine.len()];
                        let mut ext = vec![E::G1::identity(); a_affine.len()];

                        for ((((((a, b_g1), b_g2), ext), at), bt), ct) in a
                            .iter_mut()
                            .zip(b_g1.iter_mut())
                            .zip(b_g2.iter_mut())
                            .zip(ext.iter_mut())
                            .zip(at.iter())
                            .zip(bt.iter())
                            .zip(ct.iter())
                        {
                            fn eval_at_tau<Scalar: PrimeField>(
                                powers_of_tau: &[Scalar],
                                p: &[(Scalar, usize)],
                            ) -> Scalar {
                                let mut acc = Scalar::ZERO;

                                for &(ref coeff, index) in p {
                                    let mut n = powers_of_tau[index];
                                    n.mul_assign(coeff);
                                    acc.add_assign(&n);
                                }

                                acc
                            }

                            // Evaluate QAP polynomials at tau
                            let mut at = eval_at_tau::<E::Fr>(powers_of_tau, at);
                            let mut bt = eval_at_tau::<E::Fr>(powers_of_tau, bt);
                            let ct = eval_at_tau::<E::Fr>(powers_of_tau, ct);

                            // Compute A query (in G1)
                            if !bool::from(at.is_zero()) {
                                *a = g1_wnaf.scalar(&at)
                            }

                            // Compute B query (in G1/G2)
                            if !bool::from(bt.is_zero()) {
                                *b_g1 = g1_wnaf.scalar(&bt);
                                *b_g2 = g2_wnaf.scalar(&bt);
                            }

                            at.mul_assign(beta);
                            bt.mul_assign(alpha);

                            let mut e = at;
                            e.add_assign(&bt);
                            e.add_assign(&ct);
                            e.mul_assign(inv);

                            *ext = g1_wnaf.scalar(&e);
                        }

                        // Batch normalize
                        E::G1::batch_normalize(&a, a_affine);
                        E::G1::batch_normalize(&b_g1, b_g1_affine);
                        E::G2::batch_normalize(&b_g2, b_g2_affine);
                        E::G1::batch_normalize(&ext, ext_affine);
                    },
                );
        });
    }

    let num_inputs = assembly.num_inputs;
    let at_inputs = std::mem::take(&mut assembly.at_inputs);
    let bt_inputs = std::mem::take(&mut assembly.bt_inputs);
    let ct_inputs = std::mem::take(&mut assembly.ct_inputs);
    let inputs_started = Instant::now();
    // Evaluate for inputs.
    eval::<E>(
        &g1_wnaf,
        &g2_wnaf,
        &powers_of_tau,
        &at_inputs,
        &bt_inputs,
        &ct_inputs,
        &mut a_affine[0..num_inputs],
        &mut b_g1_affine[0..num_inputs],
        &mut b_g2_affine[0..num_inputs],
        &mut ic_affine,
        &gamma_inverse,
        &alpha,
        &beta,
        &pool,
        limits.batch_points,
    );
    drop((at_inputs, bt_inputs, ct_inputs));
    log::info!(
        "zigzag_setup phase=eval_inputs elapsed_ms={}",
        inputs_started.elapsed().as_millis()
    );

    // Evaluate for auxiliary variables.
    let aux_started = Instant::now();
    eval::<E>(
        &g1_wnaf,
        &g2_wnaf,
        &powers_of_tau,
        &assembly.at_aux,
        &assembly.bt_aux,
        &assembly.ct_aux,
        &mut a_affine[assembly.num_inputs..],
        &mut b_g1_affine[assembly.num_inputs..],
        &mut b_g2_affine[assembly.num_inputs..],
        &mut l_affine,
        &delta_inverse,
        &alpha,
        &beta,
        &pool,
        limits.batch_points,
    );
    drop(assembly);
    drop(powers_of_tau);
    log::info!(
        "zigzag_setup phase=eval_aux elapsed_ms={}",
        aux_started.elapsed().as_millis()
    );

    // Don't allow any elements be unconstrained, so that
    // the L query is always fully dense.
    for e in l_affine.iter() {
        if e.is_identity().into() {
            return Err(SynthesisError::UnconstrainedVariable);
        }
    }

    let g1 = g1.to_affine();
    let g2 = g2.to_affine();

    let vk = VerifyingKey::<E> {
        alpha_g1: g1.mul(alpha).to_affine(),
        beta_g1: g1.mul(beta).to_affine(),
        beta_g2: g2.mul(beta).to_affine(),
        gamma_g2: g2.mul(gamma).to_affine(),
        delta_g1: g1.mul(delta).to_affine(),
        delta_g2: g2.mul(delta).to_affine(),
        ic: ic_affine,
    };

    // Filter in place, retaining the exact Bellperson order without a second
    // full-sized vector during collect.
    a_affine.retain(|e| !bool::from(e.is_identity()));
    b_g1_affine.retain(|e| !bool::from(e.is_identity()));
    b_g2_affine.retain(|e| !bool::from(e.is_identity()));
    Ok(Parameters {
        vk,
        h: Arc::new(h_affine),
        l: Arc::new(l_affine),

        // Filter points at infinity away from A/B queries
        a: Arc::new(a_affine),
        b_g1: Arc::new(b_g1_affine),
        b_g2: Arc::new(b_g2_affine),
    })
}
