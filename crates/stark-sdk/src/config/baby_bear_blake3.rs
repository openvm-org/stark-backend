use p3_blake3::Blake3;

use super::baby_bear_bytehash::{
    self, config_from_byte_hash, BabyBearByteHashConfig, BabyBearByteHashEngine,
};
use crate::{
    assert_sc_compatible_with_serde,
    config::{
        baby_bear_bytehash::BabyBearByteHashEngineWithDefaultHash, fri_params::SecurityParameters,
    },
};

pub type BabyBearBlake3Config = BabyBearByteHashConfig<Blake3>;
pub type BabyBearBlake3Engine = BabyBearByteHashEngine<Blake3>;

assert_sc_compatible_with_serde!(BabyBearBlake3Config);

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine() -> BabyBearBlake3Engine {
    baby_bear_bytehash::default_engine(Blake3)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config() -> BabyBearBlake3Config {
    config_from_byte_hash(Blake3, SecurityParameters::standard_fast())
}

impl BabyBearByteHashEngineWithDefaultHash<Blake3> for BabyBearBlake3Engine {
    fn default_hash() -> Blake3 {
        Blake3
    }
}
