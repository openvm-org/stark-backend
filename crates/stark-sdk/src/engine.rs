pub use openvm_stark_backend::engine::StarkEngine;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::VerificationData,
    p3_matrix::dense::RowMajorMatrix,
    prover::types::AirProofInput,
    verifier::VerificationError,
    AirRef,
};
use tracing::Level;

use crate::config::{instrument::StarkHashStatistics, setup_tracing_with_log_level, FriParameters};

pub trait StarkEngineWithHashInstrumentation<SC: StarkGenericConfig>: StarkEngine<SC> {
    fn clear_instruments(&mut self);
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T>;
}

/// All necessary data to verify a Stark proof.
pub struct VerificationDataWithFriParams<SC: StarkGenericConfig> {
    pub data: VerificationData<SC>,
    pub fri_params: FriParameters,
}

/// Stark engine using Fri.
pub trait StarkFriEngine<SC: StarkGenericConfig>: StarkEngine<SC> + Sized {
    fn new(fri_parameters: FriParameters) -> Self;
    fn fri_params(&self) -> FriParameters;
    fn run_test(
        &self,
        airs: Vec<AirRef<SC>>,
        air_proof_inputs: Vec<AirProofInput<SC>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError>
    where
        AirProofInput<SC>: Send + Sync,
    {
        setup_tracing_with_log_level(Level::WARN);
        let data = <Self as StarkEngine<_>>::run_test_impl(self, airs, air_proof_inputs)?;
        Ok(VerificationDataWithFriParams {
            data,
            fri_params: self.fri_params(),
        })
    }
    fn run_test_fast(
        airs: Vec<AirRef<SC>>,
        air_proof_inputs: Vec<AirProofInput<SC>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError>
    where
        AirProofInput<SC>: Send + Sync,
    {
        let engine = Self::new(FriParameters::standard_fast());
        engine.run_test(airs, air_proof_inputs)
    }
    /// Runs a single end-to-end test for a given set of AIRs and traces.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    /// This function should only be used on AIRs where the main trace is **not** partitioned.
    fn run_simple_test_impl(
        &self,
        chips: Vec<AirRef<SC>>,
        traces: Vec<RowMajorMatrix<Val<SC>>>,
        public_values: Vec<Vec<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError>
    where
        AirProofInput<SC>: Send + Sync,
    {
        self.run_test(chips, AirProofInput::multiple_simple(traces, public_values))
    }
    fn run_simple_test_fast(
        airs: Vec<AirRef<SC>>,
        traces: Vec<RowMajorMatrix<Val<SC>>>,
        public_values: Vec<Vec<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let engine = Self::new(FriParameters::standard_fast());
        StarkFriEngine::<_>::run_simple_test_impl(&engine, airs, traces, public_values)
    }
    fn run_simple_test_no_pis_fast(
        airs: Vec<AirRef<SC>>,
        traces: Vec<RowMajorMatrix<Val<SC>>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        let pis = vec![vec![]; airs.len()];
        <Self as StarkFriEngine<SC>>::run_simple_test_fast(airs, traces, pis)
    }
}

#[macro_export]
macro_rules! collect_airs_and_inputs {
    ($($chip:expr),+ $(,)?) => {
        (
            vec![$($chip.air()),+],
            vec![$($chip.generate_air_proof_input()),+]
        )
    }
}
