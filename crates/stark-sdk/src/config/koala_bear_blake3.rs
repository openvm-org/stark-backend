use p3_blake3::Blake3;

use super::koala_bear_bytehash::{
    self, config_from_byte_hash, KoalaBearByteHashConfig, KoalaBearByteHashEngine,
};
use crate::{
    assert_sc_compatible_with_serde,
    config::{
        fri_params::SecurityParameters, koala_bear_bytehash::KoalaBearByteHashEngineWithDefaultHash,
    },
};

pub type KoalaBearBlake3Config = KoalaBearByteHashConfig<Blake3>;
pub type KoalaBearBlake3Engine = KoalaBearByteHashEngine<Blake3>;

assert_sc_compatible_with_serde!(KoalaBearBlake3Config);

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine() -> KoalaBearBlake3Engine {
    koala_bear_bytehash::default_engine(Blake3)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config() -> KoalaBearBlake3Config {
    config_from_byte_hash(Blake3, SecurityParameters::standard_fast())
}

impl KoalaBearByteHashEngineWithDefaultHash<Blake3> for KoalaBearBlake3Engine {
    fn default_hash() -> Blake3 {
        Blake3
    }
}
