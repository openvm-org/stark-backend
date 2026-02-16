pub mod base;
pub mod data_transporter;
pub mod logup_zerocheck;
pub mod merkle_tree;
pub mod monomial;
pub mod ntt;
pub mod poly;
pub mod sponge;
pub mod stacked_pcs;
pub mod stacked_reduction;
pub mod utils;
pub mod whir;

/// Rust bindings for Metal kernels
#[allow(dead_code, unused_variables)]
mod metal;
mod device;
mod engine;
mod error;
mod metal_backend;
mod pkey;
mod sumcheck;
mod types;
pub use device::*;
pub use engine::*;
pub use error::*;
pub use metal_backend::*;
pub use pkey::*;

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::types::*;
}
