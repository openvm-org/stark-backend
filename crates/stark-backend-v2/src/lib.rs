use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

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
