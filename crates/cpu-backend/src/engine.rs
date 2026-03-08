//! [`StarkEngine`] implementation using row-major [`CpuBackend`] and [`CpuDevice`].

use std::marker::PhantomData;

use openvm_stark_backend::{
    prover::Coordinator, FiatShamirTranscript, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_baby_bear::{default_babybear_poseidon2_16, Poseidon2BabyBear};

use crate::{CpuBackend, CpuDevice};

const WIDTH: usize = 16;
type SC = BabyBearPoseidon2Config;

/// Row-major CPU engine for BabyBear + Poseidon2.
///
/// Default transcript is [`CpuTranscript`], which uses Plonky3's `DuplexChallenger`
/// for SIMD-optimized proof-of-work grinding (~4x faster on NEON, ~8x on AVX2).
pub struct BabyBearPoseidon2CpuEngine<TS = crate::CpuTranscript> {
    device: CpuDevice<SC>,
    _transcript: PhantomData<TS>,
}

impl<TS> StarkEngine for BabyBearPoseidon2CpuEngine<TS>
where
    TS: FiatShamirTranscript<SC> + From<Poseidon2BabyBear<WIDTH>>,
{
    type SC = SC;
    type PB = CpuBackend<SC>;
    type PD = CpuDevice<SC>;
    type TS = TS;

    fn new(params: SystemParams) -> Self {
        let config = BabyBearPoseidon2Config::default_from_params(params);
        Self {
            device: CpuDevice::new(config),
            _transcript: PhantomData,
        }
    }

    fn config(&self) -> &SC {
        self.device.config()
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn initial_transcript(&self) -> Self::TS {
        TS::from(default_babybear_poseidon2_16())
    }

    fn prover_from_transcript(
        &self,
        transcript: TS,
    ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
        Coordinator::new(CpuBackend::new(), self.device.clone(), transcript)
    }
}
