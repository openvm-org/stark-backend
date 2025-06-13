use std::{any::type_name, marker::PhantomData, sync::Arc};

use itertools::Itertools;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    keygen::types::MultiStarkProvingKey,
    proof::Proof,
    prover::{
        coordinator::Coordinator,
        hal::{DeviceDataTransporter, TraceCommitter},
        types::{
            AirProvingContext, DeviceMultiStarkProvingKey, ProofInput, ProvingContext,
            SingleCommitPreimage,
        },
        MultiTraceStarkProver, Prover,
    },
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        log_up_params::log_up_security_params_baby_bear_100_bits, FriParameters,
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
    types::{DeviceAirProofRawInput, DeviceProofInput},
};

pub type MultiTraceStarkProverGPU = Coordinator<SC, GpuBackend, GpuDevice>;

pub struct GpuBabyBearPoseidon2Engine {
    engine: BabyBearPoseidon2Engine,
    device: GpuDevice,
}

impl StarkFriEngine<SC> for GpuBabyBearPoseidon2Engine {
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

impl GpuBabyBearPoseidon2Engine {
    pub fn device(&self) -> GpuDevice {
        self.device.clone()
    }

    fn gpu_prover(&self) -> MultiTraceStarkProverGPU {
        MultiTraceStarkProverGPU::new(
            GpuBackend::default(),
            self.device.clone(),
            self.new_challenger(),
        )
    }

    pub fn gpu_prove(
        &self,
        mpk_view: DeviceMultiStarkProvingKey<'_, GpuBackend>,
        input: DeviceProofInput<GpuBackend>,
    ) -> Proof<SC> {
        let mut prover = self.gpu_prover();
        let ctx_per_air = input
            .per_air
            .into_iter()
            .map(|(air_id, input)| {
                let cached_mains = input
                    .cached_mains
                    .into_iter()
                    .map(|trace| {
                        let traces = [trace];
                        let (com, data) = prover.device.commit(&traces);
                        let [trace] = traces;
                        (
                            com,
                            SingleCommitPreimage {
                                trace,
                                data,
                                matrix_idx: 0,
                            },
                        )
                    })
                    .collect_vec();
                let air_ctx = AirProvingContext {
                    cached_mains,
                    common_main: input.common_main,
                    public_values: input.public_values.clone(),
                    cached_lifetime: PhantomData,
                };
                (air_id, air_ctx)
            })
            .collect();
        let ctx = ProvingContext {
            per_air: ctx_per_air,
        };
        let proof = Prover::prove(&mut prover, mpk_view, ctx);
        proof.into()
    }
}

impl StarkEngine<SC> for GpuBabyBearPoseidon2Engine {
    fn config(&self) -> &SC {
        self.engine.config()
    }

    fn new_challenger(&self) -> <SC as StarkGenericConfig>::Challenger {
        self.engine.new_challenger()
    }

    fn prover<'a>(&'a self) -> MultiTraceStarkProver<'a, SC>
    where
        Self: 'a,
    {
        unimplemented!("CPU prover should not be used")
    }

    fn prove(&self, mpk: &MultiStarkProvingKey<SC>, proof_input: ProofInput<SC>) -> Proof<SC> {
        tracing::info!("LDE mode: {}", type_name::<GpuLdeDefault>());
        let mut mem = MemTracker::start("prove");
        mem.reset_peak();
        let device = self.device();
        let air_ids = proof_input.per_air.iter().map(|(id, _)| *id).collect();
        let device_input = DeviceProofInput {
            per_air: proof_input
                .per_air
                .into_iter()
                .map(|(air_id, input)| {
                    (
                        air_id,
                        DeviceAirProofRawInput::<GpuBackend> {
                            cached_mains: input
                                .raw
                                .cached_mains
                                .into_iter()
                                .map(|trace| device.transport_matrix_to_device(&trace))
                                .collect(),
                            common_main: input.raw.common_main.map(|trace| {
                                let trace = Arc::new(trace);
                                device.transport_matrix_to_device(&trace)
                            }),
                            public_values: input.raw.public_values,
                        },
                    )
                })
                .collect(),
        };
        let mpk_view = device.transport_pk_to_device(mpk, air_ids);
        mem.tracing_info("after transport to device");
        self.gpu_prove(mpk_view, device_input)
    }
}
