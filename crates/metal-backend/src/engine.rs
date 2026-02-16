use getset::MutGetters;
use openvm_stark_backend::{prover::Coordinator, StarkEngine, SystemParams};

use crate::{prelude::SC, sponge::DuplexSpongeMetal, MetalBackend, MetalDevice};

#[derive(MutGetters)]
pub struct BabyBearPoseidon2MetalEngine {
    #[getset(get_mut = "pub")]
    device: MetalDevice,
    #[getset(get_mut = "pub")]
    config: SC,
}

impl BabyBearPoseidon2MetalEngine {
    pub fn new(params: SystemParams) -> Self {
        let config = SC::default_from_params(params.clone());
        Self {
            device: MetalDevice::new(params),
            config,
        }
    }
}

impl StarkEngine for BabyBearPoseidon2MetalEngine {
    type SC = SC;
    type PB = MetalBackend;
    type PD = MetalDevice;
    type TS = DuplexSpongeMetal;

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
        DuplexSpongeMetal::default()
    }

    fn prover_from_transcript(
        &self,
        transcript: DuplexSpongeMetal,
    ) -> Coordinator<SC, Self::PB, Self::PD, Self::TS> {
        Coordinator::new(MetalBackend, self.device.clone(), transcript)
    }
}
