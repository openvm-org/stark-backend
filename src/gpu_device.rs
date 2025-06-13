use derivative::Derivative;
use openvm_stark_sdk::config::{baby_bear_poseidon2::horizen_round_consts_16, FriParameters};
use p3_baby_bear::BabyBear;
use p3_commit::TwoAdicMultiplicativeCoset;
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;

use crate::{
    cuda::{
        common::set_device,
        kernels::{ntt::sppark_init, poseidon2::init_poseidon2_constants},
    },
    fri_log_up::FriLogUpPhaseGpu,
};

#[derive(Derivative, derive_new::new, Clone, Copy, Debug)]
pub struct GpuConfig {
    pub fri: FriParameters,
    pub shift: BabyBear,
}

#[derive(Derivative, Clone, Debug)]
pub struct GpuDevice {
    pub config: GpuConfig,
    pub id: u32,
    rap_phase_seq: Option<FriLogUpPhaseGpu>,
}

#[warn(dead_code)]
impl GpuDevice {
    pub fn new(id: u32, config: GpuConfig, rap_phase_seq: Option<FriLogUpPhaseGpu>) -> Self {
        let (external, internal) = horizen_round_consts_16();
        unsafe {
            set_device(id as i32).unwrap();

            init_poseidon2_constants(
                external.get_initial_constants().as_ptr(),
                external.get_terminal_constants().as_ptr(),
                internal.as_ptr(),
            )
            .unwrap();

            sppark_init(id).unwrap();
        }
        Self {
            config,
            id,
            rap_phase_seq,
        }
    }

    pub fn rap_phase_seq(&self) -> &FriLogUpPhaseGpu {
        self.rap_phase_seq
            .as_ref()
            .expect("FriLogUpPhaseGpu is not initialized")
    }

    pub fn natural_domain_for_degree(&self, degree: usize) -> TwoAdicMultiplicativeCoset<BabyBear> {
        let log_n = log2_strict_usize(degree);
        TwoAdicMultiplicativeCoset {
            log_n,
            shift: BabyBear::ONE,
        }
    }
}
