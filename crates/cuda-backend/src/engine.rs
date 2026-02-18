use getset::MutGetters;
use openvm_stark_backend::{prover::Coordinator, StarkEngine, SystemParams};

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

    fn new(params: SystemParams) -> Self {
        Self::new(params)
    }

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
