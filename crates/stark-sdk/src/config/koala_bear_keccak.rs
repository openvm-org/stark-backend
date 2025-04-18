use p3_keccak::Keccak256Hash;

use super::koala_bear_bytehash::{
    self, config_from_byte_hash, KoalaBearByteHashConfig, KoalaBearByteHashEngine,
};
use crate::{
    assert_sc_compatible_with_serde,
    config::{
        fri_params::SecurityParameters, koala_bear_bytehash::KoalaBearByteHashEngineWithDefaultHash,
    },
};

pub type KoalaBearKeccakConfig = KoalaBearByteHashConfig<Keccak256Hash>;
pub type KoalaBearKeccakEngine = KoalaBearByteHashEngine<Keccak256Hash>;

assert_sc_compatible_with_serde!(KoalaBearKeccakConfig);

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine() -> KoalaBearKeccakEngine {
    koala_bear_bytehash::default_engine(Keccak256Hash)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config() -> KoalaBearKeccakConfig {
    config_from_byte_hash(Keccak256Hash, SecurityParameters::standard_fast())
}

impl KoalaBearByteHashEngineWithDefaultHash<Keccak256Hash> for KoalaBearKeccakEngine {
    fn default_hash() -> Keccak256Hash {
        Keccak256Hash
    }
}
