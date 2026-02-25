//! Backend for proving and verifying mixed-matrix STARKs.
//! The backend is designed to be modular and compatible with different hardware implementations.
//! The backend provides prover and verifier implementations of the SWIRL proof system.
//!
//! The aim is to support different circuit representations and permutation/lookup arguments.

// Re-export all Plonky3 crates
pub use p3_air;
pub use p3_challenger;
pub use p3_field;
pub use p3_matrix;
pub use p3_maybe_rayon;
pub use p3_symmetric;
pub use p3_util;
use p3_util::log2_ceil_u64;

/// AIR builders for prover and verifier, including support for cross-matrix permutation arguments.
pub mod air_builders;
pub mod codec;
/// Copy of Plonky3's DFT module without parallelism (Rayon) for faster single-threaded execution.
/// Used by the prover **only**.
pub mod dft;
pub mod duplex_sponge;
/// Protocol hasher trait definition.
pub mod hasher;
/// Log-up permutation argument implementation as RAP.
pub mod interaction;
/// Proving and verifying key generation
pub mod keygen;
/// Common polynomial utilities shared by prover and verifier
pub mod poly_common;
/// Definition of the STARK proof struct.
pub mod proof;
pub mod prover;
pub mod soundness;
pub mod utils;
pub mod verifier;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
// #[cfg(test)]
// pub mod tests;

mod any_air;
/// STARK Protocol configuration trait
mod config;
/// Trait for STARK backend engine proving keygen, proviing, verifying API functions.
mod engine;
/// Fiat-Shamir transcript trait definition.
mod transcript;
pub use any_air::*;
pub use config::*;
pub use engine::*;
pub use transcript::*;

// Use jemalloc as global allocator for performance
#[cfg(all(feature = "jemalloc", unix, not(test)))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Use mimalloc as global allocator
#[cfg(all(feature = "mimalloc", not(test)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
