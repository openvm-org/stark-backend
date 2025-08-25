use derivative::Derivative;
use openvm_cuda_common::common::set_device;
use openvm_stark_sdk::config::FriParameters;
use p3_baby_bear::BabyBear;
use p3_commit::TwoAdicMultiplicativeCoset;
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;

use crate::{cuda::kernels::ntt::sppark_init, fri_log_up::FriLogUpPhaseGpu};

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
    pub fn new(config: GpuConfig, rap_phase_seq: Option<FriLogUpPhaseGpu>) -> Self {
        let device_id = set_device().unwrap();
        unsafe {
            sppark_init().unwrap();
        }
        Self {
            config,
            id: device_id,
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
