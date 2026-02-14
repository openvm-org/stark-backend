use openvm_stark_backend::p3_challenger::DuplexChallenger;
use p3_baby_bear::Poseidon2BabyBear;

pub const WIDTH: usize = 16;
pub const RATE: usize = 8;

pub use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, Digest, CHUNK, DIGEST_SIZE, D_EF, EF, F,
};
pub type SC = BabyBearPoseidon2Config;
pub type Challenger = DuplexChallenger<F, Poseidon2BabyBear<WIDTH>, WIDTH, RATE>;
