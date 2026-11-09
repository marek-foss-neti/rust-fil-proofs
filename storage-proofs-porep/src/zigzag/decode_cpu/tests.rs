use super::*;
use num_bigint::BigUint;

fn modulus() -> BigUint {
    BigUint::parse_bytes(
        b"73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
        16,
    )
    .unwrap()
}

fn bytes(n: &BigUint) -> [u8; 32] {
    let mut result = [0; 32];
    let raw = n.to_bytes_le();
    result[..raw.len()].copy_from_slice(&raw);
    result
}

fn expected_subtract(value: [u8; 32], key: [u8; 32], out: &mut [u8]) -> Result<()> {
    let p = modulus();
    out.copy_from_slice(&bytes(
        &((BigUint::from_bytes_le(&value) + &p - BigUint::from_bytes_le(&key)) % p),
    ));
    Ok(())
}

fn reference_field(key: [u8; 32], value: [u8; 32], out: &mut [u8]) -> Result<()> {
    expected_subtract(value, key, out)
}

#[test]
fn canonical_limbs_match_big_integers_at_boundaries_and_random_operands() -> Result<()> {
    let p = modulus();
    let mut values = vec![BigUint::from(0u8), BigUint::from(1u8), &p - 1u8, &p - 2u8];
    for bit in [63usize, 64, 127, 128, 191, 192, 253, 254] {
        let n = BigUint::from(1u8) << bit;
        values.extend([&n - 1u8, n.clone(), n + 1u8]);
    }
    for a in &values {
        for b in &values {
            let actual = subtract(&bytes(a), &bytes(b))?;
            assert_eq!(actual, bytes(&((a + &p - b) % &p)), "{a} - {b}");
            assert!(canonical(&actual));
        }
    }
    let mut seed = 0xabc123deadbeefu64;
    for _ in 0..4096 {
        let mut next = || {
            let mut out = [0; 32];
            for chunk in out.chunks_exact_mut(8) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                chunk.copy_from_slice(&seed.to_le_bytes());
            }
            bytes(&(BigUint::from_bytes_le(&out) % &p))
        };
        let (a, b) = (next(), next());
        let mut expected = [0; 32];
        expected_subtract(a, b, &mut expected)?;
        assert_eq!(subtract(&a, &b)?, expected);
    }
    for invalid in [bytes(&p), bytes(&(&p + 1u8)), [255; 32]] {
        assert!(!canonical(&invalid));
        assert!(subtract(&invalid, &[0; 32]).is_err());
        assert!(subtract(&[0; 32], &invalid).is_err());
    }
    Ok(())
}

fn state(nodes: usize, salt: u64) -> Vec<u8> {
    let mut data = vec![0; nodes * NODE];
    for (i, out) in data.chunks_exact_mut(NODE).enumerate() {
        out[..8].copy_from_slice(&(salt + i as u64 * 0x102030405).to_le_bytes());
    }
    data
}

fn parents(node: usize, nodes: usize, forward: bool, out: &mut [u32]) -> Result<()> {
    let boundary = if forward { 0 } else { nodes - 1 };
    let repeated = node % (out.len() + 1);
    for (i, p) in out.iter_mut().enumerate() {
        *p = if node == boundary || i < repeated {
            boundary
        } else if forward {
            node - 1
        } else {
            node + 1
        } as u32;
    }
    Ok(())
}

fn reference(data: &[u8], id: [u8; 32], degree: usize, forward: bool) -> Result<Vec<u8>> {
    let nodes = data.len() / NODE;
    let boundary = if forward { 0 } else { nodes - 1 };
    let mut result = Vec::new();
    let mut list = vec![0; degree];
    for node in 0..nodes {
        parents(node, nodes, forward, &mut list)?;
        let mut hash = Sha256::new();
        hash.update(id);
        if node != boundary {
            for p in &list {
                hash.update(&data[*p as usize * NODE..(*p as usize + 1) * NODE]);
            }
        }
        let mut key: [u8; 32] = hash.finalize().into();
        key[31] &= 0x3f;
        let mut out = [0; NODE];
        expected_subtract(
            data[node * NODE..(node + 1) * NODE].try_into()?,
            key,
            &mut out,
        )?;
        result.extend_from_slice(&out);
    }
    Ok(result)
}

#[test]
fn batches_hash_modes_prefixes_and_prefetch_preserve_both_orientations() -> Result<()> {
    let nodes = 71;
    for degree in [1, 2, 13, 14, 15, 32] {
        for forward in [false, true] {
            let data = state(nodes, 91);
            let id = [0x3a; NODE];
            let expected = reference(&data, id, degree, forward)?;
            for batch_nodes in [1, 2, 4, 8, 32] {
                for sha in [DecodeSha::Serial, DecodeSha::Interleaved] {
                    for prefix_reuse in [false, true] {
                        for field in [DecodeField::Reference, DecodeField::Limbs] {
                            let config = DecodeCpuConfig {
                                batch_nodes,
                                sha,
                                prefix_reuse,
                                field,
                                prefetch: prefix_reuse,
                                profile_every: 2,
                                ..Default::default()
                            };
                            let layer =
                                CpuLayer::new(config, nodes, degree, forward, &id, &data, true)?
                                    .unwrap();
                            let mut worker = CpuWorker::new(config);
                            let ptr = worker.lanes.as_ptr();
                            let mut actual = vec![0; data.len()];
                            worker.decode(
                                &layer,
                                &data,
                                &mut actual,
                                0,
                                None,
                                |node, out| parents(node, nodes, forward, out),
                                reference_field,
                            )?;
                            assert_eq!(
                                actual, expected,
                                "degree={degree} forward={forward} config={config:?}"
                            );
                            assert_eq!(worker.lanes.as_ptr(), ptr);
                            assert_eq!(worker.stats.nodes, nodes as u64);
                            assert_eq!(worker.stats.batches, nodes.div_ceil(batch_nodes) as u64);
                            assert_eq!(
                                worker.stats.sampled_batches,
                                worker.stats.batches.div_ceil(2)
                            );
                            let expected_blocks =
                                1 + (nodes - 1) * ((degree + 1) * NODE + 9).div_ceil(64);
                            assert_eq!(worker.stats.logical_sha_blocks, expected_blocks as u64);
                            assert_eq!(
                                worker.stats.prefix_blocks_histogram.iter().sum::<u64>(),
                                nodes as u64
                            );
                            let saved = worker
                                .stats
                                .prefix_blocks_histogram
                                .iter()
                                .enumerate()
                                .map(|(blocks, count)| blocks as u64 * count)
                                .sum::<u64>();
                            assert_eq!(worker.stats.reused_sha_blocks, saved);
                            assert_eq!(saved == 0, !prefix_reuse);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn prefix_states_are_scoped_to_replica_layer_and_original_boundary() -> Result<()> {
    let nodes = 47;
    let config = DecodeCpuConfig {
        field: DecodeField::Limbs,
        prefix_reuse: true,
        batch_nodes: 8,
        ..Default::default()
    };
    for forward in [false, true] {
        for salt in [0, 777] {
            for id in [[0; NODE], [99; NODE]] {
                let mut data = state(nodes, salt);
                let expected = reference(&data, id, 14, forward)?;
                let layer = CpuLayer::new(config, nodes, 14, forward, &id, &data, true)?.unwrap();
                let mut worker = CpuWorker::new(config);
                // Simulate reverse-topological blocks. Prefix states remain tied to the
                // original boundary; no new prefix state is built from overwritten data.
                let mut blocks: Vec<_> = (0..nodes).step_by(11).collect();
                if forward {
                    blocks.reverse();
                }
                for start in blocks {
                    let end = (start + 11).min(nodes);
                    let mut block = vec![0; (end - start) * NODE];
                    worker.decode(
                        &layer,
                        &data,
                        &mut block,
                        start,
                        None,
                        |node, out| parents(node, nodes, forward, out),
                        reference_field,
                    )?;
                    data[start * NODE..end * NODE].copy_from_slice(&block);
                }
                assert_eq!(data, expected);
            }
        }
    }
    Ok(())
}

#[test]
fn fallback_ranges_invalid_parents_noncanonical_values_and_cancel() -> Result<()> {
    let nodes = 19;
    let config = DecodeCpuConfig {
        field: DecodeField::Limbs,
        batch_nodes: 4,
        prefetch: true,
        ..Default::default()
    };
    let data = state(nodes, 5);
    let id = [7; NODE];
    let layer = CpuLayer::new(config, nodes, 14, true, &id, &data, false)?.unwrap();
    let mut worker = CpuWorker::new(config);
    let mut out = vec![0; 13 * NODE];
    let mut calls = 0;
    worker.decode(
        &layer,
        &data,
        &mut out,
        3,
        None,
        |node, out| parents(node, nodes, true, out),
        |v, k, out| {
            calls += 1;
            expected_subtract(k, v, out)
        },
    )?;
    assert_eq!(
        calls, 13,
        "unsupported Domain must use the reference conversion"
    );
    assert_eq!(out, reference(&data, id, 14, true)?[3 * NODE..16 * NODE]);
    let layer = CpuLayer::new(config, nodes, 14, true, &id, &data, true)?.unwrap();
    for invalid in [u32::MAX, 1] {
        out.fill(0xa5);
        assert!(worker
            .decode(
                &layer,
                &data,
                &mut out,
                0,
                None,
                |_, out| {
                    out.fill(invalid);
                    Ok(())
                },
                reference_field
            )
            .is_err());
        assert!(out.iter().all(|&b| b == 0xa5));
    }
    assert!(worker
        .decode(
            &layer,
            &data,
            &mut out,
            usize::MAX,
            None,
            |_, _| Ok(()),
            reference_field
        )
        .is_err());
    let cancel = AtomicBool::new(true);
    assert!(worker
        .decode(
            &layer,
            &data,
            &mut out,
            0,
            Some(&cancel),
            |_, _| panic!("cancel must precede parents"),
            reference_field
        )
        .is_err());
    assert!(out.iter().all(|&b| b == 0xa5));
    let mut bad = data.clone();
    bad[..NODE].fill(255);
    assert!(worker
        .decode(
            &layer,
            &bad,
            &mut out,
            0,
            None,
            |node, out| parents(node, nodes, true, out),
            reference_field
        )
        .is_err());
    assert!(CpuLayer::new(config, nodes, 33, true, &id, &data, true)?.is_none());
    assert!(CpuLayer::new(config, nodes, 14, true, &[0; 16], &data, true)?.is_none());
    Ok(())
}

#[test]
fn configuration_worker_and_memory_limits_are_explicit() -> Result<()> {
    assert!(!DecodeCpuConfig::default().enabled());
    for args in [
        ("unknown", "serial", "1", "0", "0", "0", "auto"),
        ("limbs", "unknown", "1", "0", "0", "0", "auto"),
        ("limbs", "serial", "0", "0", "0", "0", "auto"),
        ("limbs", "serial", "33", "0", "0", "0", "auto"),
        ("limbs", "serial", "8", "2", "0", "0", "auto"),
        ("limbs", "serial", "8", "0", "2", "0", "auto"),
        ("limbs", "serial", "8", "0", "0", "1048577", "auto"),
        ("limbs", "serial", "8", "0", "0", "0", "129"),
        ("limbs", "serial", "8", "0", "0", "0", "0"),
    ] {
        assert!(
            DecodeCpuConfig::parse(args.0, args.1, args.2, args.3, args.4, args.5, args.6).is_err()
        );
    }
    let config = DecodeCpuConfig::parse("limbs", "interleaved", "32", "1", "1", "16", "auto")?;
    assert_eq!(config.worker_limit_for(96, 128), 96);
    assert_eq!(config.worker_limit_for(256, 256), 128);
    assert_eq!(config.worker_limit_for(16, 4), 4);
    assert_eq!(
        DecodeCpuConfig {
            threads: Some(8),
            ..config
        }
        .worker_limit_for(96, 128),
        8
    );
    assert_eq!(DecodeCpuConfig::default().workspace_bytes(14)?, 0);
    assert!(config.workspace_bytes(14)? < 8 * 1024 * 1024);
    assert_eq!(config.workspace_bytes(33)?, 0);
    Ok(())
}
