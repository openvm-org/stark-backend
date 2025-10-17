use std::sync::Arc;

pub use openvm_stark_backend::engine::StarkEngine;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::VerificationData,
    p3_matrix::dense::RowMajorMatrix,
    prover::{types::AirProvingContext, DeviceDataTransporter},
    verifier::VerificationError,
    AirRef,
};
use tracing::Level;

use crate::config::{instrument::StarkHashStatistics, setup_tracing_with_log_level, FriParameters};

pub trait StarkEngineWithHashInstrumentation: StarkEngine {
    fn clear_instruments(&mut self);
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T>;
}

/// All necessary data to verify a Stark proof.
pub struct VerificationDataWithFriParams<SC: StarkGenericConfig> {
    pub data: VerificationData<SC>,
    pub fri_params: FriParameters,
}

/// Stark engine using Fri.
pub trait StarkFriEngine: StarkEngine + Sized {
    fn new(fri_params: FriParameters) -> Self;
    fn fri_params(&self) -> FriParameters;
    fn run_test(
        &self,
        airs: Vec<AirRef<Self::SC>>,
        ctx: Vec<AirProvingContext<Self::PB>>,
    ) -> Result<VerificationDataWithFriParams<Self::SC>, VerificationError> {
        setup_tracing_with_log_level(Level::WARN);
        let data = <Self as StarkEngine>::run_test_impl(self, airs, ctx)?;
        Ok(VerificationDataWithFriParams {
            data,
            fri_params: self.fri_params(),
        })
    }
    fn run_test_fast(
        airs: Vec<AirRef<Self::SC>>,
        ctx: Vec<AirProvingContext<Self::PB>>,
    ) -> Result<VerificationDataWithFriParams<Self::SC>, VerificationError> {
        let engine = Self::new(FriParameters::standard_fast());
        engine.run_test(airs, ctx)
    }
    /// Runs a single end-to-end test for a given set of AIRs and traces.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    /// This function should only be used on AIRs where the main trace is **not** partitioned.
    fn run_simple_test_impl(
        &self,
        chips: Vec<AirRef<Self::SC>>,
        traces: Vec<RowMajorMatrix<Val<Self::SC>>>,
        public_values: Vec<Vec<Val<Self::SC>>>,
    ) -> Result<VerificationDataWithFriParams<Self::SC>, VerificationError> {
        let traces = traces
            .into_iter()
            .map(|trace| self.device().transport_matrix_to_device(&Arc::new(trace)))
            .collect();
        self.run_test(
            chips,
            AirProvingContext::multiple_simple(traces, public_values),
        )
    }
    fn run_simple_test_fast(
        airs: Vec<AirRef<Self::SC>>,
        traces: Vec<RowMajorMatrix<Val<Self::SC>>>,
        public_values: Vec<Vec<Val<Self::SC>>>,
    ) -> Result<VerificationDataWithFriParams<Self::SC>, VerificationError> {
        let engine = Self::new(FriParameters::standard_fast());
        StarkFriEngine::run_simple_test_impl(&engine, airs, traces, public_values)
    }
    fn run_simple_test_no_pis_fast(
        airs: Vec<AirRef<Self::SC>>,
        traces: Vec<RowMajorMatrix<Val<Self::SC>>>,
    ) -> Result<VerificationDataWithFriParams<Self::SC>, VerificationError> {
        let pis = vec![vec![]; airs.len()];
        <Self as StarkFriEngine>::run_simple_test_fast(airs, traces, pis)
    }
}
