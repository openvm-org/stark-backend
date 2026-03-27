/// Permutation-based duplex sponge and recording/validating wrappers.
pub mod duplex_sponge;
/// Multi-field transcript: bit-packed observe, base-CF::ORDER sample expansion.
pub mod multi_field;

mod traits;
pub use traits::*;
