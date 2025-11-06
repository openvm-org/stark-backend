use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_util::log2_ceil_u64;

mod engine;
pub mod keygen;
pub mod poly_common;
pub mod poseidon2;
pub mod proof;
pub mod prover;
pub mod v1_shims;
pub mod verifier;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
#[cfg(test)]
mod tests;

pub use engine::*;

pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, D_EF>;
pub const D_EF: usize = 4;

pub const DIGEST_SIZE: usize = poseidon2::CHUNK;
pub type Digest = [F; DIGEST_SIZE];

/// Common utility function for computing `n_logup` parameter in terms of `total_interactions`,
/// which is the sum of interaction message counts across all traces, using the lifted trace
/// heights.
///
/// This calculation must be consistent between the prover and verifier and is enforced by
/// the verifier.
// NOTE: we could use a more strict calculation of `n_logup = log2_ceil(total_interactions >>
// l_skip)` but the `leading_zeros` calculation below is easier to check in the recursion
// circuit. The formula below is equivalent to `log2_ceil(total_interactions + 1) - l_skip`.
pub fn calculate_n_logup(l_skip: usize, total_interactions: u64) -> usize {
    if total_interactions != 0 {
        let n_logup = (u64::BITS - total_interactions.leading_zeros()) as usize - l_skip;
        debug_assert_eq!(
            n_logup + l_skip,
            log2_ceil_u64(total_interactions + 1) as usize
        );
        n_logup
    } else {
        0
    }
}
