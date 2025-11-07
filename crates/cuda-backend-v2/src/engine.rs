use std::marker::PhantomData;

use openvm_cuda_backend::prelude::SC;
use stark_backend_v2::{
    StarkEngineV2, StarkWhirEngine, SystemParams,
    poseidon2::sponge::{DuplexSponge, FiatShamirTranscript},
    prover::CoordinatorV2,
};

use crate::{GpuBackendV2, GpuDeviceV2};

pub struct BabyBearPoseidon2GpuEngineV2<TS = DuplexSponge> {
    device: GpuDeviceV2,
    _transcript: PhantomData<TS>,
}

impl<TS> BabyBearPoseidon2GpuEngineV2<TS> {
    pub fn new(params: SystemParams) -> Self {
        Self {
            device: GpuDeviceV2::new(params),
            _transcript: PhantomData,
        }
    }
}

impl<TS> StarkEngineV2 for BabyBearPoseidon2GpuEngineV2<TS>
where
    TS: FiatShamirTranscript + Default,
{
    type SC = SC;
    type PB = GpuBackendV2;
    type PD = GpuDeviceV2;
    type TS = TS;

    fn device(&self) -> &Self::PD {
        &self.device
    }
    fn prover_from_transcript(
        &self,
        transcript: TS,
    ) -> CoordinatorV2<Self::PB, Self::PD, Self::TS> {
        CoordinatorV2::new(GpuBackendV2, self.device, transcript)
    }
}

impl<TS> StarkWhirEngine for BabyBearPoseidon2GpuEngineV2<TS>
where
    TS: FiatShamirTranscript + Default,
{
    fn new(params: SystemParams) -> Self {
        Self::new(params)
    }
}
