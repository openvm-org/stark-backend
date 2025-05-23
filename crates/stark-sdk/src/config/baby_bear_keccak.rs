use p3_keccak::Keccak256Hash;

use super::baby_bear_bytehash::{
    self, config_from_byte_hash, BabyBearByteHashConfig, BabyBearByteHashEngine,
};
use crate::{
    assert_sc_compatible_with_serde,
    config::{
        baby_bear_bytehash::BabyBearByteHashEngineWithDefaultHash, fri_params::SecurityParameters,
    },
};

pub type BabyBearKeccakConfig = BabyBearByteHashConfig<Keccak256Hash>;
pub type BabyBearKeccakEngine = BabyBearByteHashEngine<Keccak256Hash>;

assert_sc_compatible_with_serde!(BabyBearKeccakConfig);

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine() -> BabyBearKeccakEngine {
    baby_bear_bytehash::default_engine(Keccak256Hash)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config() -> BabyBearKeccakConfig {
    config_from_byte_hash(Keccak256Hash, SecurityParameters::standard_fast())
}

impl BabyBearByteHashEngineWithDefaultHash<Keccak256Hash> for BabyBearKeccakEngine {
    fn default_hash() -> Keccak256Hash {
        Keccak256Hash
    }
}
