use std::marker::PhantomData;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    time::Instant,
};

use anyhow::{bail, Context};
use bellperson::{groth16, Circuit};
use blstrs::{Bls12, Scalar as Fr};
use filecoin_hashers::Hasher;
use fs2::FileExt;
use group::prime::PrimeCurveAffine;
use rand_core::RngCore;
use storage_proofs_core::{
    compound_proof::{CircuitComponent, CompoundProof},
    drgraph::Graph,
    error::Result,
    merkle::MerkleTreeTrait,
    parameter_cache::{self, Bls12GrothParams, CacheableParameters, ParameterSetMetadata},
    proof::ProofScheme,
};

use crate::zigzag::{
    circuit::ZigZagCircuit,
    groth16_setup::SetupLimits,
    groth16_setup_disk,
    vanilla::{PublicInputs, PublicParams, ZigZagDrgPoRep},
};

/// Groth16 compound proof for ZigZag layered PoRep.
///
/// Uses a distinct `cache_prefix` (`"zigzag-proof-of-replication"`) and a distinct public-params
/// identifier so ZigZag gets its own Groth16 parameters, separate from Stacked DRG.
pub struct ZigZagCompound<Tree: MerkleTreeTrait, G: 'static + Hasher> {
    _tree: PhantomData<Tree>,
    _g: PhantomData<G>,
}

fn preflight_parameter_file(path: &std::path::Path) -> Result<()> {
    let mut file = File::open(path)?;
    let total = file.metadata()?.len();
    let g1 = blstrs::G1Affine::identity()
        .to_uncompressed()
        .as_ref()
        .len() as u64;
    let g2 = blstrs::G2Affine::identity()
        .to_uncompressed()
        .as_ref()
        .len() as u64;
    let fixed_vk = 3 * g1 + 3 * g2;
    anyhow::ensure!(
        total >= fixed_vk + 4,
        "truncated ZigZag verifying key in params"
    );
    file.seek(SeekFrom::Start(fixed_vk))?;
    let mut count = [0u8; 4];
    file.read_exact(&mut count)?;
    let ic_count = u32::from_be_bytes(count) as u64;
    let mut offset = fixed_vk + 4;
    offset = offset
        .checked_add(ic_count.checked_mul(g1).context("VK length overflow")?)
        .context("VK offset overflow")?;
    anyhow::ensure!(offset <= total, "truncated ZigZag public input query");
    let mut offset_entries = 0u64;
    for point_bytes in [g1, g1, g1, g1, g2] {
        anyhow::ensure!(
            offset.checked_add(4).is_some_and(|end| end <= total),
            "truncated ZigZag parameter query length"
        );
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut count)?;
        let entries = u32::from_be_bytes(count) as u64;
        offset_entries = offset_entries
            .checked_add(entries)
            .context("query count overflow")?;
        offset = offset
            .checked_add(4)
            .and_then(|start| {
                entries
                    .checked_mul(point_bytes)
                    .and_then(|bytes| start.checked_add(bytes))
            })
            .context("ZigZag parameter query offset overflow")?;
        anyhow::ensure!(offset <= total, "truncated ZigZag parameter query");
    }
    anyhow::ensure!(offset == total, "trailing bytes in ZigZag parameter file");
    let offset_bytes =
        offset_entries.saturating_mul(std::mem::size_of::<std::ops::Range<usize>>() as u64);
    log::info!(
        "zigzag_setup loader_offset_entries={} loader_offset_bytes={} param_file_bytes={}",
        offset_entries,
        offset_bytes,
        total
    );
    Ok(())
}

impl<Tree: MerkleTreeTrait, G: 'static + Hasher> ZigZagCompound<Tree, G> {
    /// Reads the VK prefix of a validated parameter file without depending on
    /// the in-memory representation selected by the Groth16 backend.
    pub fn read_parameter_verifying_key(
        path: &std::path::Path,
    ) -> Result<groth16::VerifyingKey<Bls12>> {
        Ok(groth16::VerifyingKey::<Bls12>::read(File::open(path)?)?)
    }

    fn read_cached_verifying_key(path: &std::path::Path) -> Result<groth16::VerifyingKey<Bls12>> {
        if storage_proofs_core::settings::SETTINGS.verify_production_params {
            let key = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("invalid ZigZag verifying key cache path")?;
            parameter_cache::verify_production_entry(
                path,
                key.to_owned(),
                parameter_cache::get_parameter_data_from_id,
            )?;
        }
        parameter_cache::with_exclusive_read_lock(path, |file| {
            groth16::VerifyingKey::<Bls12>::read(file)
        })
        .map_err(Into::into)
    }

    fn read_cached_parameters(
        path: &std::path::Path,
        vk_path: &std::path::Path,
    ) -> Result<(Bls12GrothParams, groth16::VerifyingKey<Bls12>)> {
        let read_started = Instant::now();
        groth16_setup_disk::mark_phase("setup_readback")?;
        preflight_parameter_file(path)?;
        let params = parameter_cache::read_cached_params(path)?;
        let param_vk = Self::read_parameter_verifying_key(path)?;
        if vk_path.exists() {
            let vk = Self::read_cached_verifying_key(vk_path)?;
            anyhow::ensure!(
                vk == param_vk,
                "ZigZag VK does not match params at {}",
                vk_path.display()
            );
        }
        log::info!(
            "zigzag_setup phase=readback elapsed_ms={}",
            read_started.elapsed().as_millis()
        );
        Ok((params, param_vk))
    }

    /// Returns whether published parameters already existed, either on the
    /// lock-free read path or after acquiring the per-cache writer lock.
    pub fn get_groth_params_with_cache_status<C, P, R>(
        rng: Option<&mut R>,
        circuit: C,
        pub_params: &P,
    ) -> Result<(Bls12GrothParams, bool)>
    where
        C: Circuit<Fr>,
        P: ParameterSetMetadata,
        R: RngCore,
    {
        Self::get_groth_params_locked(rng, circuit, pub_params)
    }

    fn get_groth_params_locked<C, P, R>(
        rng: Option<&mut R>,
        circuit: C,
        pub_params: &P,
    ) -> Result<(Bls12GrothParams, bool)>
    where
        C: Circuit<Fr>,
        P: ParameterSetMetadata,
        R: RngCore,
    {
        let id = <Self as CacheableParameters<C, P>>::cache_identifier(pub_params);
        let path = parameter_cache::parameter_cache_params_path(&id);
        let vk_path = parameter_cache::parameter_cache_verifying_key_path(&id);
        let parent = path
            .parent()
            .context("ZigZag parameter cache has no parent")?;
        groth16_setup_disk::mark_phase("setup_cache_lookup")?;

        // Published cache entries are immutable. Reading both files needs no
        // writable directory or writable per-cache lock.
        if path.is_file() && vk_path.is_file() {
            return Ok((Self::read_cached_parameters(&path, &vk_path)?.0, true));
        }

        fs::create_dir_all(parent)?;

        // Keep the lock across synthesis and publication. A ready cache is still
        // read by the existing integrity-checking loader.
        let lock_path = path.with_extension("params.lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(lock_path)?;
        lock.lock_exclusive()?;
        let publication = groth16_setup_disk::SetupWorkspace::new(
            parent,
            groth16_setup_disk::SetupWorkspaceKind::Publication,
        )?;
        let parameter_cache_hit = path.exists();
        if !parameter_cache_hit {
            let rng = match rng {
                Some(rng) => rng,
                None => bail!(
                    "No cached ZigZag parameters found for {} at {}",
                    id,
                    path.display()
                ),
            };
            let limits = SetupLimits::from_env()?;
            log::info!(
                "zigzag_setup budget_bytes={} batch_points={} workers={}",
                limits.budget_bytes,
                limits.batch_points,
                limits.workers
            );
            let started = Instant::now();
            let scratch = std::env::var_os("FIL_PROOFS_ZIGZAG_SETUP_SCRATCH_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| parent.join("zigzag-setup-scratch"));
            let mut pending = tempfile::NamedTempFile::new_in(publication.path())?;
            let vk = groth16_setup_disk::generate_random_parameters_to_file::<Bls12, _, _>(
                circuit,
                rng,
                limits,
                &scratch,
                &mut pending,
            )?;
            log::info!(
                "zigzag_setup phase=generation elapsed_ms={}",
                started.elapsed().as_millis()
            );
            let write_started = Instant::now();
            pending.flush()?;
            pending.as_file().sync_all()?;
            if let Err(error) = groth16_setup_disk::release_file_cache(pending.as_file()) {
                log::warn!("could not release ZigZag parameter write cache: {}", error);
            }
            let mut pending_vk = tempfile::NamedTempFile::new_in(publication.path())?;
            vk.write(&mut pending_vk)?;
            pending_vk.flush()?;
            pending_vk.as_file().sync_all()?;
            if vk_path.exists() {
                let old_vk = groth16::VerifyingKey::<Bls12>::read(File::open(&vk_path)?)?;
                anyhow::ensure!(
                    old_vk == vk,
                    "existing ZigZag VK does not match generated params"
                );
            }
            pending
                .persist_noclobber(&path)
                .context("publish ZigZag params")?;
            if !vk_path.exists() {
                pending_vk
                    .persist_noclobber(&vk_path)
                    .context("publish ZigZag VK")?;
            }
            File::open(parent)?.sync_all()?;
            log::info!(
                "zigzag_setup phase=write elapsed_ms={}",
                write_started.elapsed().as_millis()
            );
        }
        let (params, param_vk) = Self::read_cached_parameters(&path, &vk_path)?;
        if !vk_path.exists() {
            let mut pending_vk = tempfile::NamedTempFile::new_in(publication.path())?;
            param_vk.write(&mut pending_vk)?;
            pending_vk.flush()?;
            pending_vk.as_file().sync_all()?;
            pending_vk
                .persist_noclobber(&vk_path)
                .context("publish ZigZag VK")?;
        }
        Ok((params, parameter_cache_hit))
    }
}

impl<C: Circuit<Fr>, P: ParameterSetMetadata, Tree: MerkleTreeTrait, G: 'static + Hasher>
    CacheableParameters<C, P> for ZigZagCompound<Tree, G>
{
    fn cache_prefix() -> String {
        format!(
            "zigzag-proof-of-replication-{}-{}",
            Tree::display(),
            G::name()
        )
    }

    fn get_groth_params<R: RngCore>(
        rng: Option<&mut R>,
        circuit: C,
        pub_params: &P,
    ) -> Result<Bls12GrothParams> {
        Self::get_groth_params_with_cache_status(rng, circuit, pub_params).map(|(params, _)| params)
    }

    fn get_verifying_key<R: RngCore>(
        rng: Option<&mut R>,
        circuit: C,
        pub_params: &P,
    ) -> Result<groth16::VerifyingKey<Bls12>> {
        let id = <Self as CacheableParameters<C, P>>::cache_identifier(pub_params);
        let vk_path = parameter_cache::parameter_cache_verifying_key_path(&id);
        if vk_path.exists() {
            return Self::read_cached_verifying_key(&vk_path);
        }
        // The ZigZag parameter loader publishes a missing VK while holding
        // the per-cache lock. Do not attempt the trait default's second write.
        let _ = Self::get_groth_params_with_cache_status(rng, circuit, pub_params)?;
        Self::read_cached_verifying_key(&vk_path)
    }
}

impl<'a, Tree, G> CompoundProof<'a, ZigZagDrgPoRep<Tree, G>, ZigZagCircuit<Tree, G>>
    for ZigZagCompound<Tree, G>
where
    Tree: 'static + MerkleTreeTrait,
    G: 'static + Hasher,
{
    fn generate_public_inputs(
        pub_in: &PublicInputs<<Tree::Hasher as Hasher>::Domain, G::Domain>,
        pub_params: &PublicParams<Tree>,
        k: Option<usize>,
    ) -> Result<Vec<Fr>> {
        let tau = pub_in.tau.as_ref().expect("missing tau");

        let mut inputs = Vec::new();

        // Must exactly match the inputize order in `ZigZagCircuit::synthesize`.
        inputs.push(pub_in.replica_id.into());
        inputs.push(tau.comm_d.into());
        inputs.push(tau.comm_r.into());

        let layers = pub_params.layer_challenges.layers();
        let leaves = pub_params.graph.size();
        let degree = pub_params.graph.degree();

        let mut layer_graph = pub_params.graph.clone();
        for layer in 0..layers {
            let challenges =
                pub_in.challenges(&pub_params.layer_challenges, leaves, layer as u8, k);

            let mut parents = vec![0u32; degree];
            for challenge in challenges {
                let challenge = challenge % leaves;

                inputs.push(Fr::from(challenge as u64));
                inputs.push(Fr::from(challenge as u64));

                layer_graph.parents(challenge, &mut parents)?;
                for parent in &parents {
                    inputs.push(Fr::from(*parent as u64));
                }
            }

            layer_graph = ZigZagDrgPoRep::<Tree, G>::transform(&layer_graph);
        }

        inputs.push(pub_in.comm_r_star.into());

        Ok(inputs)
    }

    fn circuit(
        public_inputs: &PublicInputs<<Tree::Hasher as Hasher>::Domain, G::Domain>,
        _component_private_inputs: <ZigZagCircuit<Tree, G> as CircuitComponent>::ComponentPrivateInputs,
        vanilla_proof: &<ZigZagDrgPoRep<Tree, G> as ProofScheme<'a>>::Proof,
        public_params: &PublicParams<Tree>,
        _partition_k: Option<usize>,
    ) -> Result<ZigZagCircuit<Tree, G>> {
        let tau = public_inputs.tau.as_ref();

        Ok(ZigZagCircuit {
            public_params: public_params.clone(),
            replica_id: Some(public_inputs.replica_id),
            comm_d: tau.map(|t| t.comm_d),
            comm_r: tau.map(|t| t.comm_r),
            comm_r_star: Some(public_inputs.comm_r_star),
            proof: Some(vanilla_proof.clone()),
            _g: PhantomData,
        })
    }

    fn blank_circuit(public_params: &PublicParams<Tree>) -> ZigZagCircuit<Tree, G> {
        ZigZagCircuit {
            public_params: public_params.clone(),
            replica_id: None,
            comm_d: None,
            comm_r: None,
            comm_r_star: None,
            proof: None,
            _g: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellperson::{ConstraintSystem, SynthesisError};
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[derive(Clone)]
    struct CacheStatusCircuit;

    impl Circuit<Fr> for CacheStatusCircuit {
        fn synthesize<CS: ConstraintSystem<Fr>>(
            self,
            cs: &mut CS,
        ) -> std::result::Result<(), SynthesisError> {
            let x = cs.alloc(|| "x", || Ok(Fr::from(3)))?;
            let y = cs.alloc(|| "y", || Ok(Fr::from(5)))?;
            let z = cs.alloc_input(|| "z", || Ok(Fr::from(15)))?;
            cs.enforce(|| "multiply", |lc| lc + x, |lc| lc + y, |lc| lc + z);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct CacheStatusParams(String);

    impl ParameterSetMetadata for CacheStatusParams {
        fn identifier(&self) -> String {
            self.0.clone()
        }

        fn sector_size(&self) -> u64 {
            0
        }
    }

    #[test]
    fn rejects_partial_zigzag_parameter_file_before_mapping() {
        let mut file = tempfile::NamedTempFile::new().expect("temporary params");
        file.write_all(&[0u8; 32]).expect("write partial params");
        assert!(preflight_parameter_file(file.path()).is_err());
    }

    use filecoin_hashers::{
        poseidon::{PoseidonDomain, PoseidonHasher},
        sha256::Sha256Hasher,
    };
    use generic_array::typenum::{U0, U2};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use storage_proofs_core::{
        api_version::ApiVersion,
        compound_proof::{self, CompoundProof},
        drgraph::BASE_DEGREE,
        merkle::{DiskStore, MerkleTreeWrapper},
        util::NODE_SIZE,
        TEST_SEED,
    };

    use crate::zigzag::vanilla::{
        ChallengeRequirements, LayerChallenges, PrivateInputs, SetupParams, ZigZagDrgPoRep,
        EXP_DEGREE,
    };

    type ZZTree = MerkleTreeWrapper<PoseidonHasher, DiskStore<PoseidonDomain>, U2, U0, U0>;
    type Piece = Sha256Hasher;

    #[test]
    fn reports_cache_hit_after_waiting_for_parameter_lock() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let params = CacheStatusParams(format!(
            "zigzag-cache-status-{}-{unique}",
            std::process::id()
        ));
        let cache_id = <ZigZagCompound<ZZTree, Piece> as CacheableParameters<
            CacheStatusCircuit,
            CacheStatusParams,
        >>::cache_identifier(&params);
        let path = parameter_cache::parameter_cache_params_path(&cache_id);
        fs::create_dir_all(path.parent().expect("cache parent")).expect("create cache directory");
        let lock_path = path.with_extension("params.lock");
        let guard = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)
            .expect("open lock");
        guard.lock_exclusive().expect("hold lock before both calls");

        let barrier = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let params = params.clone();
                std::thread::spawn(move || {
                    let mut rng = XorShiftRng::from_seed(TEST_SEED);
                    barrier.wait();
                    let (_, hit) =
                        ZigZagCompound::<ZZTree, Piece>::get_groth_params_with_cache_status(
                            Some(&mut rng),
                            CacheStatusCircuit,
                            &params,
                        )
                        .expect("generate or read params");
                    hit
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        std::thread::sleep(Duration::from_millis(100));
        guard.unlock().expect("release lock");
        let mut hits = handles
            .into_iter()
            .map(|handle| handle.join().expect("cache worker panicked"))
            .collect::<Vec<_>>();
        hits.sort_unstable();
        assert_eq!(hits, [false, true]);
        fs::remove_file(&path).expect("remove params");
        fs::remove_file(parameter_cache::parameter_cache_verifying_key_path(
            &cache_id,
        ))
        .expect("remove verifying key");
        fs::remove_file(lock_path).expect("remove lock");
    }

    #[test]
    fn direct_verifying_key_lookup_repairs_missing_vk() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let metadata =
            CacheStatusParams(format!("zigzag-vk-repair-{}-{unique}", std::process::id()));
        let cache_id = <ZigZagCompound<ZZTree, Piece> as CacheableParameters<
            CacheStatusCircuit,
            CacheStatusParams,
        >>::cache_identifier(&metadata);
        let params_path = parameter_cache::parameter_cache_params_path(&cache_id);
        let vk_path = parameter_cache::parameter_cache_verifying_key_path(&cache_id);
        let lock_path = params_path.with_extension("params.lock");
        let mut rng = XorShiftRng::from_seed(TEST_SEED);
        ZigZagCompound::<ZZTree, Piece>::get_groth_params_with_cache_status(
            Some(&mut rng),
            CacheStatusCircuit,
            &metadata,
        )
        .expect("create cache");
        fs::remove_file(&vk_path).expect("remove VK");

        let vk = <ZigZagCompound<ZZTree, Piece> as CacheableParameters<
            CacheStatusCircuit,
            CacheStatusParams,
        >>::get_verifying_key(
            None::<&mut XorShiftRng>, CacheStatusCircuit, &metadata
        )
        .expect("repair VK without a second publish");
        assert_eq!(
            vk,
            ZigZagCompound::<ZZTree, Piece>::read_parameter_verifying_key(&params_path)
                .expect("read params VK")
        );
        assert!(vk_path.is_file());

        fs::remove_file(params_path).expect("remove params");
        fs::remove_file(vk_path).expect("remove VK");
        fs::remove_file(lock_path).expect("remove lock");
    }

    #[test]
    fn complete_cache_read_does_not_create_writer_lock() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let metadata =
            CacheStatusParams(format!("zigzag-read-only-{}-{unique}", std::process::id()));
        let cache_id = <ZigZagCompound<ZZTree, Piece> as CacheableParameters<
            CacheStatusCircuit,
            CacheStatusParams,
        >>::cache_identifier(&metadata);
        let params_path = parameter_cache::parameter_cache_params_path(&cache_id);
        let vk_path = parameter_cache::parameter_cache_verifying_key_path(&cache_id);
        let lock_path = params_path.with_extension("params.lock");
        let mut rng = XorShiftRng::from_seed(TEST_SEED);
        ZigZagCompound::<ZZTree, Piece>::get_groth_params_with_cache_status(
            Some(&mut rng),
            CacheStatusCircuit,
            &metadata,
        )
        .expect("create cache");
        fs::remove_file(&lock_path).expect("remove writer lock");

        let (_, hit) = ZigZagCompound::<ZZTree, Piece>::get_groth_params_with_cache_status(
            None::<&mut XorShiftRng>,
            CacheStatusCircuit,
            &metadata,
        )
        .expect("read complete cache");
        assert!(hit);
        assert!(
            !lock_path.exists(),
            "ready cache must not recreate writer lock"
        );

        fs::remove_file(params_path).expect("remove params");
        fs::remove_file(vk_path).expect("remove VK");
    }

    #[test]
    fn zigzag_compound_groth16_roundtrip() {
        let mut rng = XorShiftRng::from_seed(TEST_SEED);

        let nodes = 128;
        let layers = 2;
        let porep_id = [23u8; 32];

        let setup_params = compound_proof::SetupParams {
            vanilla_params: SetupParams {
                nodes,
                degree: BASE_DEGREE,
                expansion_degree: EXP_DEGREE,
                porep_id,
                api_version: ApiVersion::V1_2_0,
                layer_challenges: LayerChallenges::new_fixed(layers, 1),
            },
            partitions: Some(1),
            priority: false,
        };

        let public_params =
            ZigZagCompound::<ZZTree, Piece>::setup(&setup_params).expect("compound setup failed");

        let replica_id = PoseidonDomain::from([1u8; 32]);
        let mut data = vec![0u8; nodes * NODE_SIZE];

        let (tau, tree_d, replica_trees) =
            ZigZagDrgPoRep::<ZZTree, Piece>::transform_and_replicate_layers(
                &public_params.vanilla_params.graph,
                &public_params.vanilla_params.layer_challenges,
                &replica_id,
                &mut data,
                None,
            )
            .expect("replication failed");

        let public_inputs = PublicInputs {
            replica_id,
            seed: None,
            tau: Some(tau.simplify()),
            comm_r_star: tau.comm_r_star,
            k: None,
        };
        let private_inputs = PrivateInputs::<ZZTree, Piece> {
            tree_d,
            aux: replica_trees,
            layer_comm_rs: tau.layer_comm_rs.clone(),
            comm_d: tau.comm_d,
        };

        {
            use bellperson::util_cs::test_cs::TestConstraintSystem;
            use bellperson::Circuit;

            let (circuit, inputs) = ZigZagCompound::<ZZTree, Piece>::circuit_for_test(
                &public_params,
                &public_inputs,
                &private_inputs,
            )
            .expect("circuit_for_test failed");

            let mut cs = TestConstraintSystem::<Fr>::new();
            circuit.synthesize(&mut cs).expect("synthesis failed");
            assert!(cs.is_satisfied(), "constraints not satisfied");
            assert!(cs.verify(&inputs), "generated public inputs do not verify");
        }

        let groth_params = ZigZagCompound::<ZZTree, Piece>::groth_params(
            Some(&mut rng),
            &public_params.vanilla_params,
        )
        .expect("groth param generation failed");

        let proofs = ZigZagCompound::<ZZTree, Piece>::prove(
            &public_params,
            &public_inputs,
            &private_inputs,
            &groth_params,
        )
        .expect("groth prove failed");

        let verifying_key = ZigZagCompound::<ZZTree, Piece>::verifying_key::<XorShiftRng>(
            None,
            &public_params.vanilla_params,
        )
        .expect("failed to get verifying key");
        let prepared_verifying_key = bellperson::groth16::prepare_verifying_key(&verifying_key);
        let multi_proof =
            storage_proofs_core::multi_proof::MultiProof::new(proofs, &prepared_verifying_key);

        let verified = ZigZagCompound::<ZZTree, Piece>::verify(
            &public_params,
            &public_inputs,
            &multi_proof,
            &ChallengeRequirements {
                minimum_challenges: 1,
            },
        )
        .expect("groth verify errored");

        assert!(verified, "groth16 proof did not verify");
    }
}
