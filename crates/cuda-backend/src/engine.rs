use getset::MutGetters;
use openvm_stark_backend::{prover::Coordinator, StarkEngine, SystemParams};

use crate::{
    hash_scheme::{DefaultHashScheme, GpuHashScheme},
    prelude::SC,
    sponge::DuplexSpongeGpu,
    GpuBackend, GpuDevice,
};

#[cfg(feature = "baby-bear-bn254-poseidon2")]
use crate::{
    bn254_sponge::MultiField32ChallengerGpu, gpu_backend::GenericGpuBackend,
    hash_scheme::BabyBearBn254Poseidon2HashScheme,
};
#[cfg(feature = "baby-bear-bn254-poseidon2")]
use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config;

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

/// GPU engine using the BabyBear + BN254 Poseidon2 hash scheme (Groth16-friendly).
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub type BabyBearBn254Poseidon2GpuEngine = GpuEngine<BabyBearBn254Poseidon2HashScheme>;

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
        Coordinator::new(GpuBackend::default(), self.device.clone(), transcript)
    }
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl StarkEngine for GpuEngine<BabyBearBn254Poseidon2HashScheme> {
    type SC = BabyBearBn254Poseidon2Config;
    type PB = GenericGpuBackend<BabyBearBn254Poseidon2HashScheme>;
    type PD = GpuDevice;
    type TS = MultiField32ChallengerGpu;

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
        MultiField32ChallengerGpu::default()
    }

    fn prover_from_transcript(
        &self,
        transcript: MultiField32ChallengerGpu,
    ) -> Coordinator<BabyBearBn254Poseidon2Config, Self::PB, Self::PD, Self::TS> {
        Coordinator::new(
            GenericGpuBackend::default(),
            self.device.clone(),
            transcript,
        )
    }
}
