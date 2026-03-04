use std::marker::PhantomData;

use getset::MutGetters;
use openvm_stark_backend::{prover::Coordinator, StarkEngine, SystemParams};

use crate::{
    gpu_backend::GenericGpuBackend,
    hash_scheme::{DefaultHashScheme, GpuHashScheme},
    prelude::SC,
    sponge::DuplexSpongeGpu,
    GpuBackend, GpuDevice,
};

/// Generic GPU proving engine parameterised by a hash scheme.
///
/// Use the [`BabyBearPoseidon2GpuEngine`] type alias for the default Poseidon2 engine.
#[derive(MutGetters)]
pub struct GpuEngine<HS: GpuHashScheme> {
    #[getset(get_mut = "pub")]
    device: GpuDevice,
    #[getset(get_mut = "pub")]
    config: HS::SC,
}

/// Concrete GPU engine using the default BabyBear-Poseidon2 hash scheme.
pub type BabyBearPoseidon2GpuEngine = GpuEngine<DefaultHashScheme>;

impl<HS: GpuHashScheme> GpuEngine<HS> {
    pub fn new(params: SystemParams) -> Self {
        let config = HS::default_config(params.clone());
        Self {
            device: GpuDevice::new(params),
            config,
        }
    }
}

impl StarkEngine for GpuEngine<DefaultHashScheme> {
    type SC = SC;
    type PB = GpuBackend;
    type PD = GpuDevice;
    type TS = DuplexSpongeGpu;

    fn new(params: SystemParams) -> Self {
        GpuEngine::new(params)
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
        Coordinator::new(
            GenericGpuBackend::<DefaultHashScheme>(PhantomData),
            self.device.clone(),
            transcript,
        )
    }
}
