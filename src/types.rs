use openvm_stark_backend::{p3_challenger::DuplexChallenger, prover::hal::ProverBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::extension::BinomialExtensionField;

pub const WIDTH: usize = 16;
pub const RATE: usize = 8;

pub type F = BabyBear;
pub type EF = BinomialExtensionField<F, 4>;
pub type SC = BabyBearPoseidon2Config;
pub type Challenger = DuplexChallenger<F, Poseidon2BabyBear<WIDTH>, WIDTH, RATE>;

pub mod prelude {
    pub use super::{Challenger, EF, F, RATE, SC, WIDTH};
}

pub struct DeviceProofInput<PB: ProverBackend> {
    /// (AIR id, AIR input)
    pub per_air: Vec<(usize, DeviceAirProofRawInput<PB>)>,
}

pub struct DeviceAirProofRawInput<PB: ProverBackend> {
    pub cached_mains: Vec<PB::Matrix>,
    pub common_main: Option<PB::Matrix>,
    pub public_values: Vec<PB::Val>,
}
