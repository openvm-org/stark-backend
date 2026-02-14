use std::sync::Arc;

use itertools::Itertools;

use crate::{
    air_builders::debug::{debug_constraints_and_interactions, AirProofRawInput},
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    proof::*,
    prover::{
        AirProvingContext, ColMajorMatrix, Coordinator, DeviceDataTransporter,
        DeviceMultiStarkProvingKey, MultiRapProver, OpeningProver, Prover, ProverBackend,
        ProverDevice, ProvingContext, StridedColMajorMatrixView,
    },
    verifier::{verify, VerifierError},
    AirRef, FiatShamirTranscript, StarkProtocolConfig, SystemParams,
};

/// Data for verifying a Stark proof.
#[derive(Debug)]
pub struct VerificationData<SC: StarkProtocolConfig> {
    pub vk: MultiStarkVerifyingKey<SC>,
    pub proof: Proof<SC>,
}

/// A helper trait to collect the different steps in multi-trace STARK
/// keygen and proving.
pub trait StarkEngine
where
    <Self::PD as MultiRapProver<Self::PB, Self::TS>>::Artifacts:
        Into<<Self::PD as OpeningProver<Self::PB, Self::TS>>::OpeningPoints>,
    <Self::PD as MultiRapProver<Self::PB, Self::TS>>::PartialProof:
        Into<(GkrProof<Self::SC>, BatchConstraintProof<Self::SC>)>,
    <Self::PD as OpeningProver<Self::PB, Self::TS>>::OpeningProof:
        Into<(StackingProof<Self::SC>, WhirProof<Self::SC>)>,
{
    type SC: StarkProtocolConfig;
    type PB: ProverBackend<
        Val = <Self::SC as StarkProtocolConfig>::F,
        Challenge = <Self::SC as StarkProtocolConfig>::EF,
        Commitment = <Self::SC as StarkProtocolConfig>::Digest,
    >;
    type PD: ProverDevice<Self::PB, Self::TS> + DeviceDataTransporter<Self::SC, Self::PB>;
    type TS: FiatShamirTranscript<Self::SC>;

    /// Constructor from only system parameters. In particular, the transcript and hasher must have
    /// a default configuration.
    fn new(params: SystemParams) -> Self;

    fn config(&self) -> &Self::SC;

    fn params(&self) -> &SystemParams {
        self.config().params()
    }

    fn device(&self) -> &Self::PD;

    /// Creates transcript with a deterministic initial state.
    fn initial_transcript(&self) -> Self::TS;

    fn prover_from_transcript(
        &self,
        transcript: Self::TS,
    ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS>;

    fn prover(&self) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
        let transcript = self.initial_transcript();
        self.prover_from_transcript(transcript)
    }

    fn keygen(
        &self,
        airs: &[AirRef<Self::SC>],
    ) -> (
        MultiStarkProvingKey<Self::SC>,
        MultiStarkVerifyingKey<Self::SC>,
    ) {
        let mut keygen_builder = MultiStarkKeygenBuilder::new(self.config().clone());
        for air in airs {
            keygen_builder.add_air(air.clone());
        }

        let pk = keygen_builder.generate_pk().unwrap();
        let vk = pk.get_vk();
        (pk, vk)
    }

    fn prove(
        &self,
        pk: &DeviceMultiStarkProvingKey<Self::PB>,
        ctx: ProvingContext<Self::PB>,
    ) -> Proof<Self::SC> {
        let mut prover = self.prover();
        prover.prove(pk, ctx)
    }

    /// Verifies using a default instantiation of the Fiat-Shamir transcript.
    fn verify(
        &self,
        vk: &MultiStarkVerifyingKey<Self::SC>,
        proof: &Proof<Self::SC>,
    ) -> Result<(), VerifierError<<Self::SC as StarkProtocolConfig>::EF>> {
        let mut transcript = self.initial_transcript();
        verify(self.config(), vk, proof, &mut transcript)
    }

    /// The indexing of AIR ID in `ctx` should be consistent with the order of `airs`. In
    /// particular, `airs` should correspond to the global proving key with all AIRs, including ones
    /// not present in the `ctx`.
    fn debug(&self, airs: &[AirRef<Self::SC>], ctx: &ProvingContext<Self::PB>) {
        let mut keygen_builder = MultiStarkKeygenBuilder::new(self.config().clone());
        for air in airs {
            keygen_builder.add_air(air.clone());
        }
        let pk = keygen_builder.generate_pk().unwrap();

        let transpose = |mat: ColMajorMatrix<<Self::SC as StarkProtocolConfig>::F>| {
            let row_major = StridedColMajorMatrixView::from(mat.as_view()).to_row_major_matrix();
            Arc::new(row_major)
        };
        let (inputs, used_airs, used_pks): (Vec<_>, Vec<_>, Vec<_>) = ctx
            .per_trace
            .iter()
            .map(|(air_id, trace_ctx)| {
                let common_main = self
                    .device()
                    .transport_matrix_from_device_to_host(&trace_ctx.common_main);
                let cached_mains = trace_ctx
                    .cached_mains
                    .iter()
                    .map(|cd| {
                        transpose(
                            self.device()
                                .transport_matrix_from_device_to_host(&cd.trace),
                        )
                    })
                    .collect_vec();
                let common_main = Some(transpose(common_main));
                let public_values = trace_ctx.public_values.clone();
                (
                    AirProofRawInput {
                        cached_mains,
                        common_main,
                        public_values,
                    },
                    airs[*air_id].clone(),
                    &pk.per_air[*air_id],
                )
            })
            .multiunzip();

        debug_constraints_and_interactions(&used_airs, &used_pks, &inputs);
    }

    /// Runs a single end-to-end test for a given set of chips and traces partitions.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    fn run_test(
        &self,
        airs: Vec<AirRef<Self::SC>>,
        ctxs: Vec<AirProvingContext<Self::PB>>,
    ) -> Result<VerificationData<Self::SC>, VerifierError<<Self::SC as StarkProtocolConfig>::EF>>
    {
        let (pk, vk) = self.keygen(&airs);
        let device = self.prover().device;
        let d_pk = device.transport_pk_to_device(&pk);
        let ctx = ProvingContext::new(ctxs.into_iter().enumerate().collect());
        let proof = self.prove(&d_pk, ctx);
        self.verify(&vk, &proof)?;
        Ok(VerificationData { vk, proof })
    }
}
