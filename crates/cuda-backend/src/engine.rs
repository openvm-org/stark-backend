use openvm_cuda_common::memory_manager::MemTracker;
#[cfg(feature = "touchemall")]
use openvm_stark_backend::prover::types::AirProvingContext;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    keygen::MultiStarkKeygenBuilder,
    proof::Proof,
    prover::{
        coordinator::Coordinator,
        types::{DeviceMultiStarkProvingKey, ProvingContext},
        Prover,
    },
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{config_from_perm, default_perm, BabyBearPoseidon2Config},
        fri_params::{
            SecurityParameters, MAX_BATCH_SIZE_LOG_BLOWUP_1, MAX_BATCH_SIZE_LOG_BLOWUP_2,
            MAX_NUM_CONSTRAINTS,
        },
        FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::Field;

use crate::{
    fri_log_up::FriLogUpPhaseGpu,
    gpu_device::{GpuConfig, GpuDevice},
    prelude::{SC, WIDTH},
    prover_backend::GpuBackend,
};

pub type MultiTraceStarkProverGPU = Coordinator<SC, GpuBackend, GpuDevice>;

pub struct GpuBabyBearPoseidon2Engine {
    device: GpuDevice,
    config: BabyBearPoseidon2Config,
    perm: Poseidon2BabyBear<WIDTH>,
}

impl StarkFriEngine for GpuBabyBearPoseidon2Engine {
    fn new(fri_params: FriParameters) -> Self {
        let perm = default_perm();
        let security_params = SecurityParameters::new_baby_bear_100_bits(fri_params);
        Self {
            device: GpuDevice::new(
                GpuConfig::new(
                    fri_params,
                    BabyBear::GENERATOR,
                    security_params.deep_ali_params,
                ),
                Some(FriLogUpPhaseGpu::new(security_params.log_up_params)),
            ),
            config: config_from_perm(&perm, security_params),
            perm,
        }
    }
    fn fri_params(&self) -> FriParameters {
        self.device.config.fri
    }
}

impl StarkEngine for GpuBabyBearPoseidon2Engine {
    type SC = BabyBearPoseidon2Config;
    type PB = GpuBackend;
    type PD = GpuDevice;

    fn config(&self) -> &SC {
        &self.config
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(self.device.config.fri.max_constraint_degree())
    }

    fn new_challenger(&self) -> <SC as StarkGenericConfig>::Challenger {
        <SC as StarkGenericConfig>::Challenger::new(self.perm.clone())
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn keygen_builder(&self) -> MultiStarkKeygenBuilder<'_, Self::SC> {
        let mut builder = MultiStarkKeygenBuilder::new(self.config());
        builder.set_max_constraint_degree(self.max_constraint_degree().unwrap());
        let max_batch_size = if self.fri_params().log_blowup == 1 {
            MAX_BATCH_SIZE_LOG_BLOWUP_1
        } else {
            MAX_BATCH_SIZE_LOG_BLOWUP_2
        };
        builder.max_batch_size = Some(max_batch_size);
        builder.max_num_constraints = Some(MAX_NUM_CONSTRAINTS);

        builder
    }

    fn prover(&self) -> MultiTraceStarkProverGPU {
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
        #[cfg(feature = "touchemall")]
        {
            for (air_id, air_ctx) in ctx.per_air.iter() {
                check_trace_validity(air_ctx, &pk.per_air[*air_id].air_name);
            }
        }
        let mut prover = self.prover();
        let proof = prover.prove(mpk_view, ctx);
        proof.into()
    }
}

#[cfg(feature = "touchemall")]
pub fn check_trace_validity(proving_ctx: &AirProvingContext<GpuBackend>, name: &str) {
    use openvm_cuda_common::copy::MemCopyD2H;
    use openvm_stark_backend::prover::hal::MatrixDimensions;

    use crate::types::F;

    let trace = proving_ctx.common_main.as_ref().unwrap();
    let height = trace.height();
    let width = trace.width();
    let trace = trace.to_host().unwrap();
    for r in 0..height {
        for c in 0..width {
            let value = trace[c * height + r];
            let value_u32 = unsafe { *(&value as *const F as *const u32) };
            assert!(
                value_u32 != 0xffffffff,
                "potentially untouched value at ({r}, {c}) of a trace of size {height}x{width} for air {name}"
            );
        }
    }
}
