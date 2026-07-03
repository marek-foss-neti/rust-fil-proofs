use blake2s_simd::Params as Blake2s;
use filecoin_hashers::{Domain, Hasher};
use fr32::bytes_into_fr_repr_safe;
use storage_proofs_core::{
    drgraph::Graph,
    error::Result,
    parameter_cache::ParameterSetMetadata,
    util::{data_at_node, data_at_node_offset, NODE_SIZE},
};

use crate::encode;
use crate::zigzag::vanilla::graph::ZigZagGraph;

/// Encodes `data` in place using the ZigZag `graph`.
///
/// Because a node always follows all of its parents in the data, the nodes are already
/// topologically sorted, so a single in-order traversal can encode each node using its
/// already-encoded parents. The subtlety versus a plain DRG is that a reversed ZigZag layer must be
/// traversed from high to low node index; this is what `graph.forward()` selects.
pub fn encode<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &mut [u8],
) -> Result<()>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    let mut parents = vec![0u32; graph.degree()];
    for n in 0..graph.size() {
        let node = if graph.forward() {
            n
        } else {
            // If the graph is reversed, traverse in reverse order.
            (graph.size() - n) - 1
        };

        graph.parents(node, &mut parents)?;

        let key = create_key::<H>(replica_id, node, &parents, data)?;
        let start = data_at_node_offset(node);
        let end = start + NODE_SIZE;

        let node_data = H::Domain::try_from_bytes(&data[start..end])?;
        let encoded = encode::encode(key, node_data);

        encoded.write_bytes(&mut data[start..end])?;
    }

    Ok(())
}

/// Decodes (extracts) all of `data`, returning the original pre-encoding bytes.
pub fn decode<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &[u8],
) -> Result<Vec<u8>>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    (0..graph.size()).try_fold(Vec::with_capacity(data.len()), |mut acc, i| {
        acc.extend(decode_block(graph, replica_id, data, i)?.into_bytes());
        Ok(acc)
    })
}

/// Decodes a single node of `data`.
pub fn decode_block<H, G>(
    graph: &ZigZagGraph<H, G>,
    replica_id: &H::Domain,
    data: &[u8],
    v: usize,
) -> Result<H::Domain>
where
    H: Hasher,
    G: Graph<H> + ParameterSetMetadata + Sync + Send,
{
    let mut parents = vec![0u32; graph.degree()];
    graph.parents(v, &mut parents)?;
    let key = create_key::<H>(replica_id, v, &parents, data)?;
    let node_data = H::Domain::try_from_bytes(data_at_node(data, v)?)?;

    Ok(encode::decode(key, node_data))
}

/// Creates the encoding key: `Blake2s(replica_id | parent_1 | parent_2 | ...)`.
///
/// Faithful to the 2019 ZigZag KDF (Blake2s, 32-byte output), kept distinct from Stacked's SHA256
/// labeling.
pub fn create_key<H: Hasher>(
    id: &H::Domain,
    node: usize,
    parents: &[u32],
    data: &[u8],
) -> Result<H::Domain> {
    let mut hasher = Blake2s::new().hash_length(NODE_SIZE).to_state();
    hasher.update(id.into_bytes().as_ref());

    // The hash is about the parents, hence skip if a node doesn't have any parents.
    if node != parents[0] as usize {
        for parent in parents.iter() {
            let offset = data_at_node_offset(*parent as usize);
            hasher.update(&data[offset..offset + NODE_SIZE]);
        }
    }

    let hash = hasher.finalize();
    Ok(bytes_into_fr_repr_safe(hash.as_ref()).into())
}

/// Recreates the encoding key from already-materialized parent domain values (used during
/// verification, where parent bytes come from Merkle-proof leaves rather than the data buffer).
/// This must match `create_key`'s hashing exactly.
pub fn create_key_from_domains<H: Hasher>(
    id: &H::Domain,
    parents_data: &[H::Domain],
) -> Result<H::Domain> {
    let mut hasher = Blake2s::new().hash_length(NODE_SIZE).to_state();
    hasher.update(id.into_bytes().as_ref());

    for parent in parents_data.iter() {
        hasher.update(parent.into_bytes().as_ref());
    }

    let hash = hasher.finalize();
    Ok(bytes_into_fr_repr_safe(hash.as_ref()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    use filecoin_hashers::poseidon::{PoseidonDomain, PoseidonHasher};
    use storage_proofs_core::{api_version::ApiVersion, drgraph::BASE_DEGREE, util::NODE_SIZE};

    use crate::zigzag::vanilla::graph::{ZigZagBucketGraph, EXP_DEGREE};

    fn encode_decode_roundtrip(reversed: bool) {
        let nodes = 32;
        let porep_id = [7u8; 32];
        let mut graph: ZigZagBucketGraph<PoseidonHasher> = ZigZagGraph::new_zigzag(
            None,
            nodes,
            BASE_DEGREE,
            EXP_DEGREE,
            porep_id,
            ApiVersion::V1_2_0,
        )
        .expect("failed to create graph");
        if reversed {
            graph = graph.zigzag();
        }

        let replica_id = PoseidonDomain::default();

        // All-zero bytes is a valid field-element (Fr) representation for every node.
        let original = vec![0u8; nodes * NODE_SIZE];
        let mut data = original.clone();

        encode::<PoseidonHasher, _>(&graph, &replica_id, &mut data).expect("encode failed");
        assert_ne!(data, original, "encoding did not change data");

        let decoded =
            decode::<PoseidonHasher, _>(&graph, &replica_id, &data).expect("decode failed");
        assert_eq!(decoded, original, "decode did not recover original data");
    }

    #[test]
    fn encode_decode_forward() {
        encode_decode_roundtrip(false);
    }

    #[test]
    fn encode_decode_reversed() {
        encode_decode_roundtrip(true);
    }
}
