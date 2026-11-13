//! Benchmark-only ZigZag profiles, separate from registered sector rules.

use std::sync::OnceLock;

use storage_proofs_core::api_version::ApiVersion;

use crate::{
    constants::{SECTOR_SIZE_32_GIB, SECTOR_SIZE_512_MIB, SECTOR_SIZE_8_MIB},
    types::{PoRepConfig, SectorSize},
};

pub const NAME: &str = "zigzag-512";
pub const LAYERS: usize = 17;
pub const PARTITIONS: u8 = 63;
pub const MINIMUM_CHALLENGES: usize = 750;
// Round up per partition: 63 * 12 = 756 challenges per layer.
pub const CHALLENGES_PER_LAYER: usize = MINIMUM_CHALLENGES.div_ceil(PARTITIONS as usize);
pub const DEV_NAME: &str = "zigzag-8";
pub const DEV_PARTITIONS: u8 = PARTITIONS;
pub const DEV_CHALLENGES_PER_LAYER: usize = CHALLENGES_PER_LAYER;
pub const DEV_MINIMUM_CHALLENGES: usize = MINIMUM_CHALLENGES;
static POREP_ID: OnceLock<[u8; 32]> = OnceLock::new();

/// Benchmark-only 32 GiB challenge budget; layer count follows the process-local
/// sector settings so the runner can select 17 layers or an explicit override.
pub fn config_32gib(porep_id: [u8; 32], api_version: ApiVersion) -> PoRepConfig {
    assert_eq!(*POREP_ID.get_or_init(|| porep_id), porep_id);
    let mut config = PoRepConfig::new_groth16(SECTOR_SIZE_32_GIB, porep_id, api_version);
    config.partitions.0 = PARTITIONS;
    config
}

/// Carry the benchmark proof budget onto 512 MiB data geometry.
/// `porep_id` and `api_version` must be those of the 32 GiB reference run.
pub fn config(porep_id: [u8; 32], api_version: ApiVersion) -> PoRepConfig {
    let mut config = config_32gib(porep_id, api_version);
    config.sector_size = SectorSize(SECTOR_SIZE_512_MIB);
    config
}

/// Exercise the same proof budget on a smaller development sector.
pub fn config_8mib(porep_id: [u8; 32], api_version: ApiVersion) -> PoRepConfig {
    let mut config = config(porep_id, api_version);
    config.sector_size = SectorSize(SECTOR_SIZE_8_MIB);
    config.partitions.0 = DEV_PARTITIONS;
    config
}

pub(crate) fn is_config(config: &PoRepConfig) -> bool {
    POREP_ID.get() == Some(&config.porep_id)
        && matches!(
            (u64::from(config.sector_size), config.partitions.0),
            (
                SECTOR_SIZE_32_GIB | SECTOR_SIZE_512_MIB | SECTOR_SIZE_8_MIB,
                PARTITIONS
            )
        )
        && config.api_features.is_empty()
}

pub(crate) fn minimum_challenges(config: &PoRepConfig) -> Option<usize> {
    is_config(config).then_some(MINIMUM_CHALLENGES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::zigzag_setup_params;

    #[test]
    fn development_profile_uses_comparison_budget_without_changing_registered_proofs() {
        let config = config_8mib([0; 32], ApiVersion::V1_2_0);
        let setup = zigzag_setup_params(&config).unwrap();
        assert_eq!(setup.nodes, 1 << 18);
        assert_eq!(setup.layer_challenges.layers(), 17);
        assert_eq!(config.partitions.0, 63);
        assert_eq!(config.minimum_challenges(), 750);
        for layer in 0..17 {
            assert_eq!(setup.layer_challenges.challenges_for_layer(layer), 12);
        }
        let regular = PoRepConfig::new_groth16(SECTOR_SIZE_8_MIB, [0; 32], ApiVersion::V1_2_0);
        let regular_setup = zigzag_setup_params(&regular).unwrap();
        assert!(!is_config(&regular));
        assert_eq!(regular.partitions.0, 1);
        assert_eq!(regular.minimum_challenges(), 2);
        assert_eq!(regular_setup.layer_challenges.layers(), 2);
        assert_eq!(setup.degree, regular_setup.degree);
        assert_eq!(setup.expansion_degree, regular_setup.expansion_degree);
        let mut unrelated = config;
        unrelated.porep_id = [1; 32];
        assert!(!is_config(&unrelated));
    }

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
        // The benchmark has 17 layers; registered 32 GiB proofs retain 11.
        assert_eq!(reference_setup.layer_challenges.layers(), 11);
        assert_eq!(reference_setup.layer_challenges.challenges_for_layer(0), 18);
        assert_eq!(reference.partitions.0, 10);
        assert_eq!(reference.minimum_challenges(), 176);
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

    #[test]
    fn large_benchmark_budget_preserves_registered_rules_and_layer_overrides() {
        let config = config_32gib([0; 32], ApiVersion::V1_2_0);
        let setup = zigzag_setup_params(&config).unwrap();
        assert_eq!(config.partitions.0, 63);
        assert_eq!(config.minimum_challenges(), 750);
        assert_eq!(setup.layer_challenges.challenges_for_layer(0), 12);
        assert_eq!(usize::from(config.partitions.0) * CHALLENGES_PER_LAYER, 756);
        // Unlike the named small-sector profiles, the runner supplies the layer
        // override for 32 GiB. Without it, the registered layer count survives.
        assert_eq!(setup.layer_challenges.layers(), 11);
        let regular = PoRepConfig::new_groth16(SECTOR_SIZE_32_GIB, [0; 32], ApiVersion::V1_2_0);
        assert!(!is_config(&regular));
        assert_eq!(regular.partitions.0, 10);
        assert_eq!(regular.minimum_challenges(), 176);
    }
}
