use super::*;
use blstrs::Scalar;
use group::{Curve, Group};

fn fixture() -> (
    tempfile::NamedTempFile,
    bellperson::groth16::Parameters<Bls12>,
) {
    let g1 = |n| (blstrs::G1Projective::generator() * Scalar::from(n)).to_affine();
    let g2 = |n| (blstrs::G2Projective::generator() * Scalar::from(n)).to_affine();
    let params = bellperson::groth16::Parameters::<Bls12> {
        vk: VerifyingKey {
            alpha_g1: g1(1),
            beta_g1: g1(2),
            beta_g2: g2(3),
            gamma_g2: g2(4),
            delta_g1: g1(5),
            delta_g2: g2(6),
            ic: vec![g1(7), g1(8)],
        },
        h: Arc::new(vec![g1(11), g1(12)]),
        l: Arc::new(vec![g1(13), g1(14), g1(15)]),
        a: Arc::new(vec![g1(16), g1(17), g1(18), g1(19)]),
        b_g1: Arc::new(vec![g1(20), g1(21), g1(22)]),
        b_g2: Arc::new(vec![g2(23), g2(24), g2(25)]),
    };
    let mut file = tempfile::NamedTempFile::new().unwrap();
    params.write(&mut file).unwrap();
    file.as_file().sync_all().unwrap();
    (file, params)
}

#[test]
fn compact_matches_legacy_all_families_splits_and_reopen() {
    let (file, _) = fixture();
    for checked in [false, true] {
        let compact = CompactParameters::open(file.path(), "fixture".into(), checked).unwrap();
        let legacy = bellperson::groth16::Parameters::<Bls12>::build_mapped_parameters(
            file.path().to_owned(),
            checked,
        )
        .unwrap();
        assert_eq!((&compact).get_vk(2).unwrap(), (&legacy).get_vk(2).unwrap());
        assert!((&compact).get_vk(3).is_err());
        assert_eq!((&compact).get_h(2).unwrap(), (&legacy).get_h(2).unwrap());
        assert_eq!((&compact).get_l(3).unwrap(), (&legacy).get_l(3).unwrap());
        for inputs in [0, 1, 2, 3] {
            let actual = (&compact).get_a(inputs, 1).unwrap();
            assert_eq!(actual, (&legacy).get_a(inputs, 1).unwrap());
            assert!(Arc::ptr_eq(&actual.0 .0, &actual.1 .0));
            assert_eq!(
                (&compact).get_b_g1(inputs, 0).unwrap(),
                (&legacy).get_b_g1(inputs, 0).unwrap()
            );
            assert_eq!(
                (&compact).get_b_g2(inputs, 0).unwrap(),
                (&legacy).get_b_g2(inputs, 0).unwrap()
            );
        }
        assert!((&compact).get_b_g2(4, 0).is_err());
        assert!(compact.layout.families[0].range(2).is_err());
        assert_eq!(
            compact.layout.legacy_index_bytes(),
            15 * std::mem::size_of::<Range<usize>>() as u64
        );
        assert_eq!(
            std::mem::size_of_val(&compact.layout.families),
            5 * std::mem::size_of::<QueryFamily>()
        );
        drop(compact);
        let reopened = CompactParameters::open(file.path(), "fixture".into(), checked).unwrap();
        assert_eq!((&reopened).get_h(2).unwrap(), (&legacy).get_h(2).unwrap());
    }
}

#[test]
fn legacy_reader_finishes_while_compact_source_remains_alive() {
    let (file, expected) = fixture();
    let compact = CompactParameters::open(file.path(), "fixture".into(), true).unwrap();
    let path = file.path().to_owned();
    let (send, receive) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let result = parameter_cache::read_cached_params(&path)
            .map(|params| ((&params).get_h(2).unwrap().0, params.vk));
        send.send(result).unwrap();
    });
    let result = receive.recv_timeout(std::time::Duration::from_secs(2));
    if let Err(error) = result {
        // Release a regressed lifetime lock before joining, so this test fails
        // promptly instead of leaving a blocked reader thread behind.
        drop(compact);
        reader.join().unwrap();
        panic!("legacy reader blocked by live compact source: {}", error);
    }
    let (h, vk) = result.unwrap().unwrap();
    reader.join().unwrap();
    assert_eq!(h, expected.h);
    assert_eq!(vk, compact.vk);
    assert_eq!((&compact).get_h(2).unwrap().0, h);
    assert!(compact.matches_path(file.path()));
}

#[test]
fn layout_rejects_every_truncation_trailing_bytes_counts_and_overflow() {
    let (file, _) = fixture();
    let bytes = std::fs::read(file.path()).unwrap();
    let layout = ParameterLayout::parse(&bytes).unwrap();
    for end in 0..bytes.len() {
        assert!(
            ParameterLayout::parse(&bytes[..end]).is_err(),
            "truncation at {}",
            end
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(ParameterLayout::parse(&trailing).is_err());
    for offset in std::iter::once(3 * 96 + 3 * 192)
        .chain(layout.families.iter().map(|family| family.offset - 4))
    {
        let mut corrupt = bytes.clone();
        corrupt[offset..offset + 4].fill(255);
        assert!(ParameterLayout::parse(&corrupt).is_err());
    }
    assert!(QueryFamily {
        offset: usize::MAX,
        points: 1,
        encoded_point_bytes: 96
    }
    .end()
    .is_err());
    assert!(QueryFamily {
        offset: 0,
        points: usize::MAX,
        encoded_point_bytes: 192
    }
    .end()
    .is_err());
    assert!(QueryFamily {
        offset: usize::MAX - 4,
        points: 2,
        encoded_point_bytes: 96
    }
    .range(1)
    .is_err());
}

#[test]
fn invalid_points_and_infinity_preserve_checked_semantics() {
    let (file, _) = fixture();
    let original = std::fs::read(file.path()).unwrap();
    let layout = ParameterLayout::parse(&original).unwrap();
    for family in [0, 4] {
        let range = layout.families[family].range(0).unwrap();
        let identity = if family == 4 {
            G2Affine::identity().to_uncompressed().as_ref().to_vec()
        } else {
            G1Affine::identity().to_uncompressed().as_ref().to_vec()
        };
        for encoding in [vec![0u8; range.len()], identity] {
            let mut bytes = original.clone();
            bytes[range.clone()].copy_from_slice(&encoding);
            let bad = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(bad.path(), &bytes).unwrap();
            for checked in [false, true] {
                let compact = CompactParameters::open(bad.path(), "bad".into(), checked).unwrap();
                let legacy = bellperson::groth16::Parameters::<Bls12>::build_mapped_parameters(
                    bad.path().to_owned(),
                    checked,
                )
                .unwrap();
                if family == 0 {
                    assert_eq!((&compact).get_h(2).is_err(), (&legacy).get_h(2).is_err());
                    assert!((&compact).get_h(2).is_err());
                } else {
                    assert_eq!(
                        (&compact).get_b_g2(1, 2).is_err(),
                        (&legacy).get_b_g2(1, 2).is_err()
                    );
                    assert!((&compact).get_b_g2(1, 2).is_err());
                }
            }
        }
    }
}

#[test]
fn concurrent_reads_use_owned_descriptor_after_path_replacement() {
    let (file, expected) = fixture();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("params");
    std::fs::copy(file.path(), &path).unwrap();
    let compact = Arc::new(CompactParameters::open(&path, "fixture".into(), true).unwrap());
    assert!(compact.matches_path(&path));
    std::fs::rename(&path, root.path().join("original")).unwrap();
    std::fs::write(&path, b"replacement").unwrap();
    assert!(!compact.matches_path(&path));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let compact = compact.clone();
            let h = expected.h.clone();
            std::thread::spawn(move || assert_eq!((&*compact).get_h(2).unwrap().0, h))
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert!(CompactParameters::open(&path, "fixture".into(), true).is_err());
}

#[test]
fn unchecked_subgroup_loading_matches_legacy_but_checked_rejects() {
    let (file, _) = fixture();
    let mut bytes = std::fs::read(file.path()).unwrap();
    let range = ParameterLayout::parse(&bytes).unwrap().families[0]
        .range(0)
        .unwrap();
    // Find an on-curve point outside the prime subgroup using BLST's decoder;
    // avoid special encodings which BLST may reject even in unchecked mode.
    let point = (1u8..=255)
        .find_map(|x| {
            let mut encoding = [0u8; 48];
            encoding[0] = 0x80;
            encoding[47] = x;
            let point: Option<G1Affine> =
                Option::from(G1Affine::from_compressed_unchecked(&encoding));
            point.filter(|point| !bool::from(point.is_torsion_free()))
        })
        .expect("non-subgroup fixture");
    bytes[range].copy_from_slice(&point.to_uncompressed());
    let bad = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(bad.path(), bytes).unwrap();
    for checked in [false, true] {
        let compact = CompactParameters::open(bad.path(), "subgroup".into(), checked).unwrap();
        let legacy = bellperson::groth16::Parameters::<Bls12>::build_mapped_parameters(
            bad.path().to_owned(),
            checked,
        )
        .unwrap();
        let actual = (&compact).get_h(2);
        let reference = (&legacy).get_h(2);
        assert_eq!(actual.is_err(), checked);
        assert_eq!(actual.is_err(), reference.is_err());
        if !checked {
            assert_eq!(actual.unwrap(), reference.unwrap());
        }
    }
}

#[test]
fn fresh_compact_source_proves_with_existing_bellperson_params() {
    use bellperson::{Circuit, ConstraintSystem};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;
    #[derive(Clone)]
    struct Multiply;
    impl Circuit<Scalar> for Multiply {
        fn synthesize<CS: ConstraintSystem<Scalar>>(
            self,
            cs: &mut CS,
        ) -> Result<(), SynthesisError> {
            let a = cs.alloc(|| "a", || Ok(Scalar::from(3)))?;
            let b = cs.alloc(|| "b", || Ok(Scalar::from(5)))?;
            let c = cs.alloc_input(|| "c", || Ok(Scalar::from(15)))?;
            cs.enforce(|| "multiply", |lc| lc + a, |lc| lc + b, |lc| lc + c);
            Ok(())
        }
    }
    let mut rng = XorShiftRng::from_seed(storage_proofs_core::TEST_SEED);
    // Run the same test in a fresh process with an existing file, ensuring the
    // production enum/loader selection does not depend on a setup object/cache.
    if let Ok(path) = std::env::var("ZIGZAG_PARAMETERS_TEST_FILE") {
        if let Ok(reason) = std::env::var("ZIGZAG_PARAMETERS_TEST_REJECTION") {
            let error = ZigZagParameters::open(Path::new(&path), "multiply".into(), true)
                .err()
                .expect("reject invalid configuration or production fixture");
            if reason == "production" {
                assert!(SETTINGS.verify_production_params);
                assert!(matches!(
                    error.downcast_ref::<Error>(),
                    Some(Error::InvalidParameters(_))
                ));
            } else {
                assert!(error
                    .to_string()
                    .contains("loader must be compact or mapped"));
            }
            return;
        }
        let parameters = ZigZagParameters::open(Path::new(&path), "multiply".into(), true).unwrap();
        let loader = std::env::var("FIL_PROOFS_ZIGZAG_PARAMETER_LOADER").unwrap();
        let diagnostics = parameters.diagnostics();
        assert_eq!(diagnostics["loader"], loader);
        if loader == "compact" {
            assert!(matches!(parameters, ZigZagParameters::Compact(_)));
            assert_eq!(
                diagnostics["index_metadata_bytes"],
                std::mem::size_of::<[QueryFamily; 5]>()
            );
        }
        let proofs = bellperson::groth16::create_random_proof_batch(
            vec![Multiply, Multiply],
            &parameters,
            &mut rng,
        )
        .unwrap();
        let pvk = bellperson::groth16::prepare_verifying_key(parameters.vk());
        let references: Vec<_> = proofs.iter().collect();
        assert!(bellperson::groth16::verify_proofs_batch(
            &pvk,
            &mut rng,
            &references,
            &[vec![Scalar::from(15)], vec![Scalar::from(15)]]
        )
        .unwrap());
        let mut bytes = Vec::new();
        for proof in proofs {
            proof.write(&mut bytes).unwrap();
        }
        std::fs::write(
            std::env::var("ZIGZAG_PARAMETERS_TEST_PROOF").unwrap(),
            bytes,
        )
        .unwrap();
        return;
    }
    let params =
        bellperson::groth16::generate_random_parameters::<Bls12, _, _>(Multiply, &mut rng).unwrap();
    let mut file = tempfile::NamedTempFile::new().unwrap();
    params.write(&mut file).unwrap();
    file.as_file().sync_all().unwrap();
    let vk = params.vk.clone();
    drop(params);
    for _ in 0..2 {
        let reopened = CompactParameters::open(file.path(), "multiply".into(), true).unwrap();
        let legacy = bellperson::groth16::Parameters::<Bls12>::build_mapped_parameters(
            file.path().to_owned(),
            true,
        )
        .unwrap();
        let mut reference_rng = rng.clone();
        let reference = bellperson::groth16::create_random_proof_batch(
            vec![Multiply],
            &legacy,
            &mut reference_rng,
        )
        .unwrap();
        let proofs =
            bellperson::groth16::create_random_proof_batch(vec![Multiply], &reopened, &mut rng)
                .unwrap();
        let mut expected_bytes = Vec::new();
        let mut actual_bytes = Vec::new();
        reference[0].write(&mut expected_bytes).unwrap();
        proofs[0].write(&mut actual_bytes).unwrap();
        assert_eq!(actual_bytes, expected_bytes);
        let pvk = bellperson::groth16::prepare_verifying_key(&vk);
        let references: Vec<_> = proofs.iter().collect();
        assert!(bellperson::groth16::verify_proofs_batch(
            &pvk,
            &mut rng,
            &references,
            &[vec![Scalar::from(15)]]
        )
        .unwrap());
    }
    let legacy = bellperson::groth16::Parameters::<Bls12>::build_mapped_parameters(
        file.path().to_owned(),
        true,
    )
    .unwrap();
    let mut reference_rng = XorShiftRng::from_seed(storage_proofs_core::TEST_SEED);
    let proofs = bellperson::groth16::create_random_proof_batch(
        vec![Multiply, Multiply],
        &legacy,
        &mut reference_rng,
    )
    .unwrap();
    let mut expected = Vec::new();
    for proof in proofs {
        proof.write(&mut expected).unwrap();
    }
    for loader in ["compact", "mapped"] {
        let output = tempfile::NamedTempFile::new().unwrap();
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "zigzag::parameters::tests::fresh_compact_source_proves_with_existing_bellperson_params", "--nocapture"])
            .env("ZIGZAG_PARAMETERS_TEST_FILE", file.path())
            .env("ZIGZAG_PARAMETERS_TEST_PROOF", output.path())
            .env("FIL_PROOFS_ZIGZAG_PARAMETER_LOADER", loader)
            .output().unwrap();
        assert!(
            child.status.success(),
            "fresh {} process: {} {}",
            loader,
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        assert_eq!(std::fs::read(output.path()).unwrap(), expected);
    }
    for reason in ["production", "loader"] {
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "zigzag::parameters::tests::fresh_compact_source_proves_with_existing_bellperson_params", "--nocapture"])
            .env("ZIGZAG_PARAMETERS_TEST_FILE", file.path())
            .env("ZIGZAG_PARAMETERS_TEST_REJECTION", reason)
            .env("FIL_PROOFS_VERIFY_PRODUCTION_PARAMS", if reason == "production" { "true" } else { "false" })
            .env("FIL_PROOFS_ZIGZAG_PARAMETER_LOADER", if reason == "loader" { "unknown" } else { "compact" })
            .output().unwrap();
        assert!(
            child.status.success(),
            "{} rejection: {} {}",
            reason,
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
    }
}
