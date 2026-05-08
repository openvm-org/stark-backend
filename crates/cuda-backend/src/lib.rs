// cudaStream_t is an opaque CUDA handle (*mut c_void) passed through to FFI.
// Clippy's not_unsafe_ptr_arg_deref fires on functions that accept it, but the
// "pointer" is never dereferenced in Rust — it is just forwarded to CUDA runtime calls.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod base;
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub mod bn254_sponge;
pub mod data_transporter;
pub mod hash_scheme;
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

/// Rust bindings for CUDA kernels
mod cuda;
mod device;
mod engine;
mod error;
mod gpu_backend;
mod pkey;
mod sumcheck;
mod types;
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub use bn254_sponge::MultiFieldTranscriptGpu;
pub use device::*;
pub use engine::*;
pub use error::*;
pub use gpu_backend::*;
pub use hash_scheme::*;
pub use pkey::*;

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::types::*;
}
