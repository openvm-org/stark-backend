//! BN254 Poseidon2 permutations used by the BabyBear + BN254 STARK configuration.
//!
//! This namespace groups the two concrete BN254 Poseidon2 instantiations we use:
//! - width 3 for leaf hashing and the transcript sponge
//! - width 2 for Merkle compression

mod common;
mod width2;
mod width2_constants;
mod width3;

pub use common::Poseidon2Bn254Constants;
pub use width2::{
    default_bn254_poseidon2_width2, default_bn254_poseidon2_width2_constants, Poseidon2Bn254Width2,
    Poseidon2Bn254Width2Constants,
};
pub use width3::{
    default_bn254_poseidon2_width3, default_bn254_poseidon2_width3_constants, Poseidon2Bn254Width3,
    Poseidon2Bn254Width3Constants,
};
