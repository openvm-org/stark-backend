use itertools::{zip_eq, Itertools};

use crate::{
    air_builders::debug::debug_constraints_and_interactions,
    config::{Com, PcsProof, RapPhaseSeqPartialProof, StarkGenericConfig, Val},
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey, StarkProvingKey},
        MultiStarkKeygenBuilder,
    },
    proof::{OpeningProof, Proof},
    prover::{
        coordinator::Coordinator,
        types::{AirProofRawInput, AirProvingContext, DeviceMultiStarkProvingKey, ProvingContext},
        DeviceDataTransporter, Prover, ProverBackend, ProverDevice,
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
pub trait StarkEngine
where
    <Self::PB as ProverBackend>::OpeningProof:
        Into<OpeningProof<PcsProof<Self::SC>, <Self::SC as StarkGenericConfig>::Challenge>>,
    <Self::PB as ProverBackend>::RapPartialProof: Into<Option<RapPhaseSeqPartialProof<Self::SC>>>,
{
    type SC: StarkGenericConfig;
    type PB: ProverBackend<
        Val = Val<Self::SC>,
        Challenge = <Self::SC as StarkGenericConfig>::Challenge,
        Commitment = Com<Self::SC>,
        Challenger = <Self::SC as StarkGenericConfig>::Challenger,
    >;
    type PD: ProverDevice<Self::PB> + DeviceDataTransporter<Self::SC, Self::PB>;

    /// Stark config
    fn config(&self) -> &Self::SC;

    /// During keygen, the circuit may be optimized but it will **try** to keep the
    /// constraint degree at most this value.
    fn max_constraint_degree(&self) -> Option<usize> {
        None
    }

    /// Creates a new challenger with a deterministic state.
    /// Creating new challenger for prover and verifier separately will result in
    /// them having the same starting state.
    fn new_challenger(&self) -> <Self::SC as StarkGenericConfig>::Challenger;

    fn keygen_builder(&self) -> MultiStarkKeygenBuilder<'_, Self::SC> {
        let mut builder = MultiStarkKeygenBuilder::new(self.config());
        if let Some(max_constraint_degree) = self.max_constraint_degree() {
            builder.set_max_constraint_degree(max_constraint_degree);
        }
        builder
    }

    fn device(&self) -> &Self::PD;

    fn prover(&self) -> Coordinator<Self::SC, Self::PB, Self::PD>;

    fn verifier(&self) -> MultiTraceStarkVerifier<'_, Self::SC> {
        MultiTraceStarkVerifier::new(self.config())
    }

    /// Add AIRs and get AIR IDs
    fn set_up_keygen_builder(
        &self,
        keygen_builder: &mut MultiStarkKeygenBuilder<'_, Self::SC>,
        airs: &[AirRef<Self::SC>],
    ) -> Vec<usize> {
        airs.iter()
            .map(|air| keygen_builder.add_air(air.clone()))
            .collect()
    }

    /// As a convenience, this function also transports the proving key from host to device.
    /// Note that the [Self::prove] function starts from a [DeviceMultiStarkProvingKey],
    /// which should be used if the proving key is already cached in device memory.
    fn prove_then_verify(
        &self,
        pk: &MultiStarkProvingKey<Self::SC>,
        ctx: ProvingContext<Self::PB>,
    ) -> Result<Proof<Self::SC>, VerificationError> {
        let pk_device = self.device().transport_pk_to_device(pk);
        let proof = self.prove(&pk_device, ctx);
        self.verify(&pk.get_vk(), &proof)?;
        Ok(proof)
    }

    fn prove(
        &self,
        pk: &DeviceMultiStarkProvingKey<Self::PB>,
        ctx: ProvingContext<Self::PB>,
    ) -> Proof<Self::SC> {
        let mpk_view = pk.view(ctx.air_ids());
        let mut prover = self.prover();
        let proof = prover.prove(mpk_view, ctx);
        proof.into()
    }

    fn verify(
        &self,
        vk: &MultiStarkVerifyingKey<Self::SC>,
        proof: &Proof<Self::SC>,
    ) -> Result<(), VerificationError> {
        let mut challenger = self.new_challenger();
        let verifier = self.verifier();
        verifier.verify(&mut challenger, vk, proof)
    }

    // mpk can be removed if we use BaseAir trait to regenerate preprocessed traces
    fn debug(
        &self,
        airs: &[AirRef<Self::SC>],
        pk: &[StarkProvingKey<Self::SC>],
        proof_inputs: &[AirProofRawInput<Val<Self::SC>>],
    ) {
        let (trace_views, pvs): (Vec<_>, Vec<_>) = proof_inputs
            .iter()
            .map(|input| {
                let mut views = input
                    .cached_mains
                    .iter()
                    .map(|trace| trace.as_view())
                    .collect_vec();
                if let Some(trace) = input.common_main.as_ref() {
                    views.push(trace.as_view());
                }
                (views, input.public_values.clone())
            })
            .unzip();
        debug_constraints_and_interactions(airs, pk, &trace_views, &pvs);
    }

    /// Runs a single end-to-end test for a given set of chips and traces partitions.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    fn run_test_impl(
        &self,
        airs: Vec<AirRef<Self::SC>>,
        ctx: Vec<AirProvingContext<Self::PB>>,
    ) -> Result<VerificationData<Self::SC>, VerificationError> {
        let mut keygen_builder = self.keygen_builder();
        let air_ids = self.set_up_keygen_builder(&mut keygen_builder, &airs);
        let pk = keygen_builder.generate_pk();
        let device = self.prover().device;
        let proof_inputs = ctx
            .iter()
            .map(|air_ctx| {
                let cached_mains = air_ctx
                    .cached_mains
                    .iter()
                    .map(|pre| device.transport_matrix_from_device_to_host(&pre.trace))
                    .collect_vec();
                let common_main = air_ctx
                    .common_main
                    .as_ref()
                    .map(|m| device.transport_matrix_from_device_to_host(m));
                let public_values = air_ctx.public_values.clone();
                AirProofRawInput {
                    cached_mains,
                    common_main,
                    public_values,
                }
            })
            .collect_vec();
        self.debug(&airs, &pk.per_air, &proof_inputs);
        let vk = pk.get_vk();
        let ctx = ProvingContext {
            per_air: zip_eq(air_ids, ctx).collect(),
        };
        let proof = self.prove_then_verify(&pk, ctx)?;
        Ok(VerificationData { vk, proof })
    }
}
