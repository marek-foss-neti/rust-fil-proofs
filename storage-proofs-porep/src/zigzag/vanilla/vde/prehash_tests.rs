use super::*;
use filecoin_hashers::poseidon::{PoseidonDomain, PoseidonHasher};
use storage_proofs_core::{api_version::ApiVersion, drgraph::BASE_DEGREE};

use crate::zigzag::vanilla::graph::{ZigZagBucketGraph, EXP_DEGREE};

#[test]
fn holes_duplicates_and_slot_reuse_preserve_every_key() {
    let mut bytes = vec![0; 16 * NODE_SIZE];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = (index * 29 + index / NODE_SIZE) as u8;
    }
    let data = EncodeData::new(&mut bytes);
    let mut slot = EncodePrefetchSlot::new(14);
    slot.parents
        .copy_from_slice(&[1, 2, 1, 3, 5, 4, 4, 7, 9, 8, 2, 10, 12, 11]);
    let mut input = vec![31; key_input_len(14)];
    for (index, &parent) in slot.parents.iter().enumerate() {
        input[(index + 1) * NODE_SIZE..(index + 2) * NODE_SIZE]
            .copy_from_slice(unsafe { data.node_bytes(parent as usize) });
    }
    let reference = create_key_from_staged_input::<PoseidonHasher>(&input);
    // All masks include holes followed by ready parents and repeated indices.
    // Poison missing bytes to catch accidentally hashing through the first hole.
    for mask in 0..1u32 << 14 {
        slot.input_len = input.len();
        slot.key_input.copy_from_slice(&input);
        slot.missing_parents = ParentMask(mask);
        for index in 0..14 {
            if slot.missing_parents.get(index) {
                slot.key_input[(index + 1) * NODE_SIZE..(index + 2) * NODE_SIZE].fill(0xff);
            }
        }
        prepare_prefetched_hash(&mut slot, EncodePrehash { enabled: true });
        let ready = if mask == 0 {
            14
        } else {
            mask.trailing_zeros() as usize
        };
        assert_eq!(
            slot.prepared_hash
                .as_ref()
                .map_or(0, |hash| hash.processed_bytes),
            (ready + 1) * NODE_SIZE / 64 * 64
        );
        patch_missing_parents(&data, &mut slot);
        assert_eq!(
            finish_prefetched_key::<PoseidonHasher>(&mut slot),
            reference
        );
        assert!(slot.prepared_hash.is_none(), "consumer must take ownership");
    }
    // Reuse a previously fully prepared slot for the first node (ID only).
    slot.key_input.copy_from_slice(&input);
    slot.missing_parents.clear();
    prepare_prefetched_hash(&mut slot, EncodePrehash { enabled: true });
    assert!(slot.prepared_hash.is_some());
    slot.input_len = NODE_SIZE;
    prepare_prefetched_hash(&mut slot, EncodePrehash { enabled: true });
    assert!(slot.prepared_hash.is_none());
    assert_eq!(
        finish_prefetched_key::<PoseidonHasher>(&mut slot),
        create_key_from_staged_input::<PoseidonHasher>(&input[..NODE_SIZE])
    );
}

#[test]
fn actual_graph_frontiers_in_both_directions_match_reference() -> Result<()> {
    let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
        None,
        128,
        BASE_DEGREE,
        EXP_DEGREE,
        [67; 32],
        ApiVersion::V1_2_0,
    )?;
    let id = PoseidonDomain::from([13; 32]);
    let mut bytes = vec![0; graph.size() * NODE_SIZE];
    for node in 0..graph.size() {
        bytes[node * NODE_SIZE] = node as u8;
    }
    for graph in [graph.clone(), graph.zigzag()] {
        let data = EncodeData::new(&mut bytes);
        let mut slot = EncodePrefetchSlot::new(graph.degree());
        for position in 0..graph.size() {
            for frontier in 0..=position {
                slot.key_input.fill(0xee);
                fill_encode_prefetch_slot(
                    &graph,
                    &mut None,
                    &id,
                    &data,
                    position,
                    frontier,
                    &mut slot,
                    EncodePrehash { enabled: true },
                )?;
                patch_missing_parents(&data, &mut slot);
                let reference = create_key_from_staged_input::<PoseidonHasher>(
                    &slot.key_input[..slot.input_len],
                );
                assert_eq!(
                    finish_prefetched_key::<PoseidonHasher>(&mut slot),
                    reference
                );
            }
        }
    }
    Ok(())
}

#[test]
fn prehash_modes_match_across_ring_shapes_and_parent_sources() -> Result<()> {
    let graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
        None,
        128,
        BASE_DEGREE,
        EXP_DEGREE,
        [68; 32],
        ApiVersion::V1_2_0,
    )?;
    let id = PoseidonDomain::from([14; 32]);
    let mut original = vec![0; graph.size() * NODE_SIZE];
    for node in 0..graph.size() {
        original[node * NODE_SIZE] = (node * 11) as u8;
    }
    for graph in [graph.clone(), graph.zigzag()] {
        let mut reference = original.clone();
        encode_sequential(&graph, None, &id, &mut reference)?;
        for enabled in [false, true] {
            for parent_cache in [false, true] {
                for (lookahead, stride) in [(1, 1), (4, 2), (7, 3), (32, 17)] {
                    for producers in [1, 2, 3, 4] {
                        let mut encoded = original.clone();
                        encode_multicore_with_config(
                            &graph,
                            parent_cache,
                            &id,
                            &mut encoded,
                            EncodePipelineConfig {
                                lookahead,
                                stride,
                                producers,
                                affinity: false,
                                cache_policy: CachePolicy::default(),
                                prehash: EncodePrehash { enabled },
                            },
                            |_, _| Ok(()),
                            |_| Ok(()),
                        )?;
                        assert_eq!(encoded, reference, "enabled={enabled} cache={parent_cache} lookahead={lookahead} stride={stride} producers={producers}");
                    }
                }
            }
        }
    }
    Ok(())
}
