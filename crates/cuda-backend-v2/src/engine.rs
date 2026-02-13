use openvm_cuda_backend::prelude::SC;
use openvm_stark_backend::{
    debug::debug_impl,
    prover::{CoordinatorV2, ProvingContextV2},
    AirRef, StarkEngineV2, StarkWhirEngine, SystemParams,
};

use crate::{sponge::DuplexSpongeGpu, GpuBackendV2, GpuDeviceV2};

pub struct BabyBearPoseidon2GpuEngineV2 {
    device: GpuDeviceV2,
}

impl BabyBearPoseidon2GpuEngineV2 {
    pub fn new(params: SystemParams) -> Self {
        Self {
            device: GpuDeviceV2::new(params),
        }
    }
}

impl StarkEngineV2 for BabyBearPoseidon2GpuEngineV2 {
    type SC = SC;
    type PB = GpuBackendV2;
    type PD = GpuDeviceV2;
    type TS = DuplexSpongeGpu;

    fn device(&self) -> &Self::PD {
        &self.device
    }
    fn prover_from_transcript(
        &self,
        transcript: DuplexSpongeGpu,
    ) -> CoordinatorV2<Self::PB, Self::PD, Self::TS> {
        CoordinatorV2::new(GpuBackendV2, self.device.clone(), transcript)
    }
    fn debug(&self, airs: &[AirRef<Self::SC>], ctx: &ProvingContextV2<Self::PB>) {
        debug_impl(self.config().clone(), self.device(), airs, ctx);
    }
}

impl StarkWhirEngine for BabyBearPoseidon2GpuEngineV2 {
    fn new(params: SystemParams) -> Self {
        Self::new(params)
    }
}

// #[cfg(feature = "touchemall")]
// pub fn check_trace_validity(proving_ctx: &AirProvingContext<GpuBackend>, name: &str) {
//     use openvm_cuda_common::copy::MemCopyD2H;
//     use openvm_stark_backend::prover::hal::MatrixDimensions;

//     use crate::types::F;

//     let trace = proving_ctx.common_main.as_ref().unwrap();
//     let height = trace.height();
//     let width = trace.width();
//     let trace = trace.to_host().unwrap();
//     for r in 0..height {
//         for c in 0..width {
//             let value = trace[c * height + r];
//             let value_u32 = unsafe { *(&value as *const F as *const u32) };
//             assert!(
//                 value_u32 != 0xffffffff,
//                 "potentially untouched value at ({r}, {c}) of a trace of size {height}x{width} for air {name}"
//             );
//         }
//     }
// }
