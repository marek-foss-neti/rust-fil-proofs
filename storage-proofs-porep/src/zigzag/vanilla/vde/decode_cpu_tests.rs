use super::*;
use crate::zigzag::vanilla::graph::{ZigZagBucketGraph, EXP_DEGREE};
use crate::zigzag::{
    decode_cpu::{DecodeCpuConfig, DecodeField, DecodeSha},
    DecodeConfig,
};
use filecoin_hashers::{blake2s::Blake2sHasher, poseidon::PoseidonHasher, sha256::Sha256Hasher};
use storage_proofs_core::{api_version::ApiVersion, drgraph::BASE_DEGREE};

fn compare<H: Hasher + 'static>() -> Result<()> {
    let graph: ZigZagBucketGraph<H> = ZigZagGraph::new_zigzag(
        None,
        128,
        BASE_DEGREE,
        EXP_DEGREE,
        [87; 32],
        ApiVersion::V1_2_0,
    )?;
    let id = H::Domain::default();
    let mut original = vec![0; graph.size() * NODE_SIZE];
    for (i, node) in original.chunks_exact_mut(NODE_SIZE).enumerate() {
        node[..8].copy_from_slice(&(i as u64 * 99).to_le_bytes());
    }
    for first in [graph.clone(), graph.zigzag()] {
        let mut states = vec![original.clone()];
        let mut graphs = Vec::new();
        let mut graph = first;
        for _ in 0..17 {
            let mut data = states.last().unwrap().clone();
            encode_sequential(&graph, None, &id, &mut data)?;
            states.push(data);
            graphs.push(graph.clone());
            graph = graph.zigzag();
        }
        for layers in [2, 11, 17] {
            for cpu in [
                DecodeCpuConfig {
                    field: DecodeField::Limbs,
                    ..Default::default()
                },
                DecodeCpuConfig {
                    batch_nodes: 8,
                    prefix_reuse: true,
                    ..Default::default()
                },
                DecodeCpuConfig {
                    field: DecodeField::Limbs,
                    batch_nodes: 8,
                    prefix_reuse: true,
                    prefetch: true,
                    sha: DecodeSha::Interleaved,
                    profile_every: 2,
                    threads: Some(4),
                },
            ] {
                for reader in [
                    DecodeParentReader::Legacy,
                    DecodeParentReader::Buffered,
                    DecodeParentReader::Mmap,
                ] {
                    let mut session = DecodeSession::with_config(DecodeConfig {
                        parent_reader: reader,
                        parent_buffer_mib: 1,
                    });
                    session.cpu = cpu;
                    let mut state = states[layers].clone();
                    let sealed = state.clone();
                    let mut block = vec![0; 17 * NODE_SIZE];
                    for layer in (0..layers).rev() {
                        let mut ping_pong = vec![0; state.len()];
                        decode_into_with_session(
                            &graphs[layer],
                            &id,
                            &state,
                            &mut ping_pong,
                            &mut session,
                        )?;
                        assert_eq!(
                            ping_pong, states[layer],
                            "ping-pong layer={layer} cpu={cpu:?}"
                        );
                        let range = NODE_SIZE..127 * NODE_SIZE;
                        let mut stream = Vec::new();
                        blockwise::stream_with_session(
                            &graphs[layer],
                            &id,
                            &state,
                            &mut block,
                            range.clone(),
                            None,
                            &mut session,
                            |_, bytes| {
                                stream.extend_from_slice(bytes);
                                Ok(())
                            },
                        )?;
                        assert_eq!(stream, states[layer][range]);
                        blockwise::decode_with_session(
                            &graphs[layer],
                            &id,
                            &mut state,
                            &mut block,
                            None,
                            &mut session,
                        )?;
                        assert_eq!(state, states[layer], "blockwise layer={layer} cpu={cpu:?}");
                    }
                    assert_eq!(state, original);
                    assert_eq!(sealed, states[layers]);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn cpu_kernels_match_reference_fr_for_supported_domains_and_all_decode_modes() -> Result<()> {
    // Encoding uses the original Fr arithmetic. Recovering every stored layer
    // checks the limb kernel against that implementation, not just a second formula.
    compare::<PoseidonHasher>()?;
    compare::<Sha256Hasher>()?;
    compare::<Blake2sHasher>()
}

#[test]
fn cpu_workers_cover_multiple_stripes_and_partial_blocks() -> Result<()> {
    // More than MIN_STRIPE_NODES, with a tail in both the stripe and node batch.
    let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
        None,
        32771,
        BASE_DEGREE,
        EXP_DEGREE,
        [88; 32],
        ApiVersion::V1_2_0,
    )?;
    let id = <PoseidonHasher as Hasher>::Domain::default();
    let mut original = vec![0; graph.size() * NODE_SIZE];
    for (i, node) in original.chunks_exact_mut(NODE_SIZE).enumerate() {
        node[..8].copy_from_slice(&(i as u64 * 97).to_le_bytes());
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    pool.install(|| -> Result<()> {
        for graph in [graph.clone(), graph.zigzag()] {
            let mut state = original.clone();
            encode_sequential(&graph, None, &id, &mut state)?;
            let mut session = DecodeSession::with_config(DecodeConfig {
                parent_reader: DecodeParentReader::Buffered,
                parent_buffer_mib: 1,
            });
            session.cpu = DecodeCpuConfig {
                field: DecodeField::Limbs,
                sha: DecodeSha::Interleaved,
                batch_nodes: 8,
                prefix_reuse: true,
                prefetch: true,
                threads: Some(2),
                ..Default::default()
            };
            let mut out = vec![0xa5; state.len()];
            decode_into_with_session(&graph, &id, &state, &mut out, &mut session)?;
            assert_eq!(out, original);
            let mut block = vec![0xa5; 32769 * NODE_SIZE];
            blockwise::decode_with_session(
                &graph,
                &id,
                &mut state,
                &mut block,
                None,
                &mut session,
            )?;
            assert_eq!(state, original);
        }
        Ok(())
    })
}
