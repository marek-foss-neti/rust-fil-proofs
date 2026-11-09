use filecoin_hashers::poseidon::{PoseidonDomain, PoseidonHasher};
use storage_proofs_core::{api_version::ApiVersion, drgraph::BASE_DEGREE};

use super::*;
use crate::zigzag::{DecodeConfig, ZigZagBucketGraph, EXP_DEGREE};

#[test]
fn every_layer_matches_reference_with_both_orientations_and_partial_blocks() -> Result<()> {
    let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
        None,
        128,
        BASE_DEGREE,
        EXP_DEGREE,
        [81; 32],
        ApiVersion::V1_2_0,
    )?;
    let id = PoseidonDomain::default();
    let mut original = vec![0; graph.size() * NODE_SIZE];
    for (i, node) in original.chunks_exact_mut(NODE_SIZE).enumerate() {
        node[..8].copy_from_slice(&(i as u64 * 37).to_le_bytes());
    }
    for first_graph in [graph.clone(), graph.zigzag()] {
        let mut states = vec![original.clone()];
        let mut graphs = Vec::new();
        let mut graph = first_graph;
        for _ in 0..17 {
            let mut encoded = states.last().unwrap().clone();
            super::super::encode_sequential(&graph, None, &id, &mut encoded)?;
            states.push(encoded);
            graphs.push(graph.clone());
            graph = graph.zigzag();
        }
        for layers in [2, 11, 17] {
            for reader in [
                DecodeParentReader::Legacy,
                DecodeParentReader::Buffered,
                DecodeParentReader::Mmap,
            ] {
                for threads in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()?;
                    for block_nodes in [1, 17, 129] {
                        pool.install(|| -> Result<()> {
                            // Exercise the multilayer entry point as well as every layer below.
                            if reader == DecodeParentReader::Legacy && threads == 1 && block_nodes == 17 {
                                use storage_proofs_core::merkle::BinaryMerkleTree;
                                use crate::zigzag::{LayerChallenges, ZigZagDrgPoRep};
                                let mut state = states[layers].clone();
                                let mut block = vec![0; block_nodes * NODE_SIZE];
                                ZigZagDrgPoRep::<BinaryMerkleTree<PoseidonHasher>, PoseidonHasher>::extract_and_invert_transform_layers_with_block(
                                    &graphs[0], &LayerChallenges::new_fixed(layers, 1), &id,
                                    &mut state, &mut block, None,
                                )?;
                                assert_eq!(state, original);
                            }
                            let mut session = DecodeSession::with_config(DecodeConfig { parent_reader: reader, parent_buffer_mib: 1 });
                            let sealed = states[layers].clone();
                            let mut state = sealed.clone();
                            let mut block = vec![0xa5; block_nodes * NODE_SIZE];
                            for layer in (0..layers).rev() {
                                decode_with_session(&graphs[layer], &id, &mut state, &mut block, None, &mut session)?;
                                assert_eq!(state, states[layer], "layer={layer} reader={reader:?} block={block_nodes} workers={threads}");
                            }
                            assert_eq!(sealed, states[layers]);
                            Ok(())
                        })?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn block_workers_cover_multiple_stripes_and_global_offsets() -> Result<()> {
    let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
        None,
        65536,
        BASE_DEGREE,
        EXP_DEGREE,
        [82; 32],
        ApiVersion::V1_2_0,
    )?;
    let id = PoseidonDomain::default();
    for graph in [graph.clone(), graph.zigzag()] {
        let original = vec![0; graph.size() * NODE_SIZE];
        let mut state = original.clone();
        super::super::encode_sequential(&graph, None, &id, &mut state)?;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        pool.install(|| -> Result<()> {
            let mut session = DecodeSession::with_config(DecodeConfig {
                parent_reader: DecodeParentReader::Buffered,
                parent_buffer_mib: 1,
            });
            let layout = session.config.layout(32769, graph.degree(), 4)?;
            assert!(layout.readers > 1);
            let mut block = vec![0; 32769 * NODE_SIZE];
            decode_with_session(&graph, &id, &mut state, &mut block, None, &mut session)
        })?;
        assert_eq!(state, original);
    }
    Ok(())
}
