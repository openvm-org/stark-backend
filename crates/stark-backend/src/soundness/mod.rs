//! Soundness analysis for the SWIRL proof system.

use p3_field::PrimeField64;

use crate::StarkProtocolConfig;

mod calculator;
mod vk;

pub use calculator::*;

/// Order `p` of the configured base field.
///
/// Not part of the public API: exposed only so concrete configs (e.g. in the SDK) can specialize
/// it without restating the formula.
#[doc(hidden)]
pub fn base_field_order<SC: StarkProtocolConfig>() -> f64 {
    SC::F::ORDER_U64 as f64
}

/// Number of bits in the challenge (extension) field: `D_EF * log2(|F|)`.
///
/// Derived from the configured extension degree (`StarkProtocolConfig::D_EF`), so it tracks
/// automatically if the extension field changes.
///
/// Not part of the public API: exposed only so concrete configs (e.g. in the SDK) can specialize
/// it without restating the formula.
#[doc(hidden)]
pub fn challenge_field_bits<SC: StarkProtocolConfig>() -> f64 {
    SC::D_EF as f64 * base_field_order::<SC>().log2()
}
