use std::any::type_name;

use openvm_stark_backend::{
    config::StarkGenericConfig,
    proof::Proof,
    prover::{
        coordinator::Coordinator,
        types::{DeviceMultiStarkProvingKey, ProvingContext},
        Prover,
    },
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        log_up_params::log_up_security_params_baby_bear_100_bits,
        FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use p3_baby_bear::BabyBear;
use p3_field::Field;

use crate::{
    cuda::memory_manager::MemTracker,
    fri_log_up::FriLogUpPhaseGpu,
    gpu_device::{GpuConfig, GpuDevice},
    lde::GpuLdeDefault,
    prelude::SC,
    prover_backend::GpuBackend,
};

pub type MultiTraceStarkProverGPU = Coordinator<SC, GpuBackend, GpuDevice>;

pub struct GpuBabyBearPoseidon2Engine {
    engine: BabyBearPoseidon2Engine,
    device: GpuDevice,
}

impl StarkFriEngine for GpuBabyBearPoseidon2Engine {
    fn new(fri_params: FriParameters) -> Self {
        Self {
            engine: BabyBearPoseidon2Engine::new(fri_params),
            device: GpuDevice::new(
                0,
                GpuConfig::new(fri_params, BabyBear::GENERATOR),
                Some(FriLogUpPhaseGpu::new(
                    log_up_security_params_baby_bear_100_bits(),
                )),
            ),
        }
    }
    fn fri_params(&self) -> FriParameters {
        self.engine.fri_params()
    }
}

impl StarkEngine for GpuBabyBearPoseidon2Engine {
    type SC = BabyBearPoseidon2Config;
    type PB = GpuBackend;
    type PD = GpuDevice;

    fn config(&self) -> &SC {
        self.engine.config()
    }

    fn new_challenger(&self) -> <SC as StarkGenericConfig>::Challenger {
        self.engine.new_challenger()
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn prover(&self) -> MultiTraceStarkProverGPU {
        tracing::info!("LDE mode: {}", type_name::<GpuLdeDefault>());
        MultiTraceStarkProverGPU::new(
            GpuBackend::default(),
            self.device.clone(),
            self.new_challenger(),
        )
    }

    fn prove(
        &self,
        pk: &DeviceMultiStarkProvingKey<Self::PB>,
        ctx: ProvingContext<Self::PB>,
    ) -> Proof<Self::SC> {
        let mut mem = MemTracker::start("prove");
        mem.reset_peak();

        let mpk_view = pk.view(ctx.air_ids());
        let mut prover = self.prover();
        let proof = prover.prove(mpk_view, ctx);
        proof.into()
    }
}
