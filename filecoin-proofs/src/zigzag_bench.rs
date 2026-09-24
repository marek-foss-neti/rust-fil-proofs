//! Benchmark-only ZigZag profile. This is deliberately separate from registered sector rules.

use std::sync::OnceLock;

use storage_proofs_core::api_version::ApiVersion;

use crate::{
    constants::{SECTOR_SIZE_32_GIB, SECTOR_SIZE_512_MIB},
    types::{PoRepConfig, SectorSize},
};

pub const NAME: &str = "zigzag-512";
pub const LAYERS: usize = 11;
pub const PARTITIONS: u8 = 10;
pub const MINIMUM_CHALLENGES: usize = 176;
pub const CHALLENGES_PER_LAYER: usize = 18;
static POREP_ID: OnceLock<[u8; 32]> = OnceLock::new();

/// Carry the 32 GiB ZigZag proof budget onto 512 MiB data geometry.
/// `porep_id` and `api_version` must be those of the 32 GiB reference run.
pub fn config(porep_id: [u8; 32], api_version: ApiVersion) -> PoRepConfig {
    assert_eq!(*POREP_ID.get_or_init(|| porep_id), porep_id);
    let mut config = PoRepConfig::new_groth16(SECTOR_SIZE_32_GIB, porep_id, api_version);
    assert_eq!(config.partitions.0, PARTITIONS);
    config.sector_size = SectorSize(SECTOR_SIZE_512_MIB);
    config
}

pub(crate) fn is_config(config: &PoRepConfig) -> bool {
    POREP_ID.get() == Some(&config.porep_id)
        && u64::from(config.sector_size) == SECTOR_SIZE_512_MIB
        && config.partitions.0 == PARTITIONS
        && config.api_features.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::zigzag_setup_params;

    #[test]
    fn setup_has_512_mib_geometry_and_32_gib_proof_budget() {
        let config = config([0; 32], ApiVersion::V1_2_0);
        let setup = zigzag_setup_params(&config).unwrap();
        let reference = PoRepConfig::new_groth16(SECTOR_SIZE_32_GIB, [0; 32], ApiVersion::V1_2_0);
        let reference_setup = zigzag_setup_params(&reference).unwrap();
        assert_eq!(setup.nodes, 1 << 24);
        assert_eq!(reference_setup.nodes, 1 << 30);
        assert_eq!(setup.degree, reference_setup.degree);
        assert_eq!(setup.expansion_degree, reference_setup.expansion_degree);
        assert_eq!(setup.porep_id, reference_setup.porep_id);
        assert_eq!(setup.api_version, reference_setup.api_version);
        assert_eq!(setup.layer_challenges, reference_setup.layer_challenges);
        assert_eq!(config.partitions.0, reference.partitions.0);
        assert_eq!(config.minimum_challenges(), reference.minimum_challenges());
        assert_eq!(setup.layer_challenges.layers(), LAYERS);
        assert_eq!(
            setup.layer_challenges.challenges_for_layer(0),
            CHALLENGES_PER_LAYER
        );
        assert_eq!(config.minimum_challenges(), MINIMUM_CHALLENGES);
        assert_eq!(config.partitions.0, PARTITIONS);
        let regular = PoRepConfig::new_groth16(SECTOR_SIZE_512_MIB, [0; 32], ApiVersion::V1_2_0);
        let regular_setup = zigzag_setup_params(&regular).unwrap();
        assert_eq!(regular_setup.layer_challenges.layers(), 2);
        assert_eq!(regular.minimum_challenges(), 2);
        assert_eq!(regular.partitions.0, 1);
    }
}
