// TODO[TEMP]: remove once we make traits generic in SC
pub use openvm_stark_sdk;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_util::log2_ceil_u64;

mod chip;
pub mod codec;
mod config;
pub mod debug;
pub mod dft;
mod engine;
pub mod keygen;
pub mod poly_common;
pub mod poseidon2;
pub mod proof;
pub mod prover;
pub mod utils;
pub mod v1_shims;
pub mod verifier;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
#[cfg(test)]
mod tests;

pub use chip::*;
pub use config::*;
pub use engine::*;

pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, D_EF>;
pub const D_EF: usize = 4;

pub const DIGEST_SIZE: usize = poseidon2::CHUNK;
pub type Digest = [F; DIGEST_SIZE];

/// GKR sumcheck block size (number of variables cleared per sumcheck round).
pub const GKR_BLOCK_SIZE: usize = 2;

/// Returns 4^k, the number of points in {0,1,2,3}^k.
pub(crate) fn gkr_block_len(k: usize) -> usize {
    debug_assert!(k > 0, "gkr_block_len expects k > 0");
    let shift = 2usize
        .checked_mul(k)
        .expect("gkr_block_len overflow computing 2*k");
    let shift_u32 = u32::try_from(shift).expect("gkr_block_len shift too large");
    1usize
        .checked_shl(shift_u32)
        .expect("gkr_block_len overflow computing 4^k")
}

/// Yields block sizes that sum to `num_vars`, with each block size <= GKR_BLOCK_SIZE.
pub(crate) fn block_sumcheck_sizes(num_vars: usize) -> impl Iterator<Item = usize> {
    let mut remaining = num_vars;
    std::iter::from_fn(move || {
        if remaining == 0 {
            None
        } else {
            let k = remaining.min(GKR_BLOCK_SIZE);
            remaining -= k;
            Some(k)
        }
    })
}

// TODO: remove after making SC generic in v2
pub type SC = BabyBearPoseidon2Config;

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
