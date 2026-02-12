use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{poseidon2::{sponge::Poseidon2Hasher, CHUNK}, StarkProtocolConfig};

/// ZST config type for BabyBear + Poseidon2.
pub struct BabyBearPoseidon2ConfigV2;

impl StarkProtocolConfig for BabyBearPoseidon2ConfigV2 {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type Digest = [BabyBear; CHUNK];
    type H = Poseidon2Hasher;
}

// Convenience type aliases (for internal use in Phase 1)
pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, 4>;
pub const D_EF: usize = 4;
pub const DIGEST_SIZE: usize = CHUNK;
pub type Digest = [F; DIGEST_SIZE];
