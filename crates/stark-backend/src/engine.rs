use std::iter::zip;

use itertools::{izip, Itertools};
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_util::log2_strict_usize;

use crate::{
    air_builders::debug::debug_constraints_and_interactions,
    config::{StarkGenericConfig, Val},
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey, StarkProvingKey},
        MultiStarkKeygenBuilder,
    },
    proof::Proof,
    prover::{
        cpu::{CpuBackend, CpuDevice, PcsDataView},
        hal::{DeviceDataAdapter, TraceCommitter},
        types::{AirProofInput, AirProvingContext, CommittedTraceView, ProofInput, ProvingContext},
        MultiTraceStarkProver, Prover,
    },
    verifier::{MultiTraceStarkVerifier, VerificationError},
    AirRef,
};

/// Data for verifying a Stark proof.
pub struct VerificationData<SC: StarkGenericConfig> {
    pub vk: MultiStarkVerifyingKey<SC>,
    pub proof: Proof<SC>,
}

/// A helper trait to collect the different steps in multi-trace STARK
/// keygen and proving. Currently this trait is CPU specific.
pub trait StarkEngine<SC: StarkGenericConfig> {
    /// Stark config
    fn config(&self) -> &SC;

    /// Creates a new challenger with a deterministic state.
    /// Creating new challenger for prover and verifier separately will result in
    /// them having the same starting state.
    fn new_challenger(&self) -> SC::Challenger;

    fn keygen_builder(&self) -> MultiStarkKeygenBuilder<SC> {
        MultiStarkKeygenBuilder::new(self.config())
    }

    fn prover<'a>(&'a self) -> MultiTraceStarkProver<'a, SC>
    where
        Self: 'a,
    {
        MultiTraceStarkProver::new(
            CpuBackend::<SC>::default(),
            CpuDevice::new(self.config()),
            self.new_challenger(),
        )
    }

    fn verifier(&self) -> MultiTraceStarkVerifier<SC> {
        MultiTraceStarkVerifier::new(self.config())
    }

    // mpk can be removed if we use BaseAir trait to regenerate preprocessed traces
    fn debug(
        &self,
        airs: &[AirRef<SC>],
        pk: &[StarkProvingKey<SC>],
        proof_inputs: &[AirProofInput<SC>],
    ) {
        let (trace_views, pvs): (Vec<_>, Vec<_>) = proof_inputs
            .iter()
            .map(|input| {
                let mut views = input
                    .raw
                    .cached_mains
                    .iter()
                    .map(|trace| trace.as_view())
                    .collect_vec();
                if let Some(trace) = input.raw.common_main.as_ref() {
                    views.push(trace.as_view());
                }
                (views, input.raw.public_values.clone())
            })
            .unzip();
        debug_constraints_and_interactions(airs, pk, &trace_views, &pvs);
    }
    // TODO[jpw]: the following does not belong in this crate! dev tooling only

    /// Runs a single end-to-end test for a given set of AIRs and traces.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    /// This function should only be used on AIRs where the main trace is **not** partitioned.
    fn run_simple_test_impl(
        &self,
        airs: Vec<AirRef<SC>>,
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: Vec<Vec<Val<SC>>>,
    ) -> Result<VerificationData<SC>, VerificationError> {
        self.run_test_impl(airs, AirProofInput::multiple_simple(traces, public_values))
    }

    /// Runs a single end-to-end test for a given set of chips and traces partitions.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    fn run_test_impl(
        &self,
        airs: Vec<AirRef<SC>>,
        air_proof_inputs: Vec<AirProofInput<SC>>,
    ) -> Result<VerificationData<SC>, VerificationError> {
        let mut keygen_builder = self.keygen_builder();
        let air_ids = self.set_up_keygen_builder(&mut keygen_builder, &airs);
        let pk = keygen_builder.generate_pk();
        self.debug(&airs, &pk.per_air, &air_proof_inputs);
        let vk = pk.get_vk();
        let proof_input = ProofInput {
            per_air: izip!(air_ids, air_proof_inputs).collect(),
        };
        let proof = self.prove(&pk, proof_input);
        self.verify(&vk, &proof)?;
        Ok(VerificationData { vk, proof })
    }

    /// Add AIRs and get AIR IDs
    fn set_up_keygen_builder(
        &self,
        keygen_builder: &mut MultiStarkKeygenBuilder<'_, SC>,
        airs: &[AirRef<SC>],
    ) -> Vec<usize> {
        airs.iter()
            .map(|air| keygen_builder.add_air(air.clone()))
            .collect()
    }

    fn prove_then_verify(
        &self,
        mpk: &MultiStarkProvingKey<SC>,
        proof_input: ProofInput<SC>,
    ) -> Result<(), VerificationError> {
        let proof = self.prove(mpk, proof_input);
        self.verify(&mpk.get_vk(), &proof)
    }

    fn prove(&self, mpk: &MultiStarkProvingKey<SC>, proof_input: ProofInput<SC>) -> Proof<SC> {
        let mut prover = self.prover();
        let backend = prover.backend;
        let air_ids = proof_input.per_air.iter().map(|(id, _)| *id).collect();
        let ctx_per_air = proof_input
            .per_air
            .into_iter()
            .map(|(air_id, input)| {
                // Commit cached traces if they are not provided
                let cached_mains = if input.cached_mains_pdata.len() != input.raw.cached_mains.len()
                {
                    input
                        .raw
                        .cached_mains
                        .iter()
                        .map(|trace| {
                            let trace = backend.transport_matrix_to_device(trace);
                            let (com, data) = prover.device.commit(&[trace.clone()]);
                            (
                                com,
                                CommittedTraceView {
                                    trace,
                                    data,
                                    matrix_idx: 0,
                                },
                            )
                        })
                        .collect()
                } else {
                    zip(input.cached_mains_pdata, input.raw.cached_mains)
                        .map(|((com, data), trace)| {
                            let data_view = PcsDataView {
                                data,
                                log_trace_heights: vec![log2_strict_usize(trace.height()) as u8],
                            };
                            let view = CommittedTraceView {
                                trace,
                                data: data_view,
                                matrix_idx: 0,
                            };
                            (com, view)
                        })
                        .collect()
                };
                let air_ctx = AirProvingContext {
                    cached_mains,
                    common_main: input.raw.common_main,
                    public_values: input.raw.public_values,
                };
                (air_id, air_ctx)
            })
            .collect();
        let ctx = ProvingContext {
            per_air: ctx_per_air,
        };
        let mpk_view = backend.transport_pk_to_device(mpk, air_ids);
        let proof = Prover::prove(&mut prover, mpk_view, ctx);
        proof.into()
    }

    fn verify(
        &self,
        vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError> {
        let mut challenger = self.new_challenger();
        let verifier = self.verifier();
        verifier.verify(&mut challenger, vk, proof)
    }
}
