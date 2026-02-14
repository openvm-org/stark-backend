use getset::MutGetters;
use openvm_stark_backend::{prover::Coordinator, DefaultStarkEngine, StarkEngine, SystemParams};

use crate::{prelude::SC, sponge::DuplexSpongeGpu, GpuBackend, GpuDevice};

#[derive(MutGetters)]
pub struct BabyBearPoseidon2GpuEngine {
    #[getset(get_mut = "pub")]
    device: GpuDevice,
    #[getset(get_mut = "pub")]
    config: SC,
}

impl BabyBearPoseidon2GpuEngine {
    pub fn new(params: SystemParams) -> Self {
        let config = SC::default_from_params(params.clone());
        Self {
            device: GpuDevice::new(params),
            config,
        }
    }
}

impl StarkEngine for BabyBearPoseidon2GpuEngine {
    type SC = SC;
    type PB = GpuBackend;
    type PD = GpuDevice;
    type TS = DuplexSpongeGpu;

    fn config(&self) -> &Self::SC {
        &self.config
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn initial_transcript(&self) -> Self::TS {
        DuplexSpongeGpu::default()
    }

    fn prover_from_transcript(
        &self,
        transcript: DuplexSpongeGpu,
    ) -> Coordinator<SC, Self::PB, Self::PD, Self::TS> {
        Coordinator::new(GpuBackend, self.device.clone(), transcript)
    }
}

impl DefaultStarkEngine for BabyBearPoseidon2GpuEngine {
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
//                 "potentially untouched value at ({r}, {c}) of a trace of size {height}x{width}
// for air {name}"             );
//         }
//     }
// }
