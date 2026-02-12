use std::sync::Arc;

use itertools::Itertools;

use crate::{
    air_builders::debug::{debug_constraints_and_interactions, AirProofRawInput},
    keygen::{
        types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
        MultiStarkKeygenBuilderV2,
    },
    proof::*,
    prover::{
        AirProvingContextV2, ColMajorMatrix, CoordinatorV2, DeviceDataTransporterV2,
        DeviceMultiStarkProvingKeyV2, MultiRapProver, OpeningProverV2, Prover, ProverBackendV2,
        ProverDeviceV2, ProvingContextV2, StridedColMajorMatrixView,
    },
    verifier::{verify, VerifierError},
    AirRef, FiatShamirTranscript, StarkProtocolConfig, SystemParams,
};

/// Data for verifying a Stark proof.
#[derive(Debug)]
pub struct VerificationDataV2<SC: StarkProtocolConfig> {
    pub vk: MultiStarkVerifyingKeyV2<SC>,
    pub proof: Proof<SC>,
}

/// A helper trait to collect the different steps in multi-trace STARK
/// keygen and proving. Currently this trait is CPU specific.
pub trait StarkEngineV2
where
    <Self::PD as MultiRapProver<Self::PB, Self::TS>>::Artifacts:
        Into<<Self::PD as OpeningProverV2<Self::PB, Self::TS>>::OpeningPoints>,
    <Self::PD as MultiRapProver<Self::PB, Self::TS>>::PartialProof:
        Into<(GkrProof<Self::SC>, BatchConstraintProof<Self::SC>)>,
    <Self::PD as OpeningProverV2<Self::PB, Self::TS>>::OpeningProof:
        Into<(StackingProof<Self::SC>, WhirProof<Self::SC>)>,
{
    type SC: StarkProtocolConfig;
    type PB: ProverBackendV2<
        Val = <Self::SC as StarkProtocolConfig>::F,
        Challenge = <Self::SC as StarkProtocolConfig>::EF,
        Commitment = <Self::SC as StarkProtocolConfig>::Digest,
    >;
    type PD: ProverDeviceV2<Self::PB, Self::TS> + DeviceDataTransporterV2<Self::SC, Self::PB>;
    type TS: FiatShamirTranscript<Self::SC> + Default;

    fn config(&self) -> &SystemParams {
        self.device().config()
    }

    fn device(&self) -> &Self::PD;

    fn prover_from_transcript(
        &self,
        transcript: Self::TS,
    ) -> CoordinatorV2<Self::SC, Self::PB, Self::PD, Self::TS>;

    fn prover(&self) -> CoordinatorV2<Self::SC, Self::PB, Self::PD, Self::TS> {
        self.prover_from_transcript(Self::TS::default())
    }

    fn keygen(
        &self,
        airs: &[AirRef<Self::SC>],
    ) -> (
        MultiStarkProvingKeyV2<Self::SC>,
        MultiStarkVerifyingKeyV2<Self::SC>,
    ) {
        let mut keygen_builder = MultiStarkKeygenBuilderV2::new(self.config().clone());
        for air in airs {
            keygen_builder.add_air(air.clone());
        }

        let pk = keygen_builder.generate_pk().unwrap();
        let vk = pk.get_vk();
        (pk, vk)
    }

    fn prove(
        &self,
        pk: &DeviceMultiStarkProvingKeyV2<Self::PB>,
        ctx: ProvingContextV2<Self::PB>,
    ) -> Proof<Self::SC>
    where
        Self::PB: ProverBackendV2<
            Val = <Self::SC as StarkProtocolConfig>::F,
            Challenge = <Self::SC as StarkProtocolConfig>::EF,
            Commitment = <Self::SC as StarkProtocolConfig>::Digest,
        >,
    {
        let mut prover = self.prover();
        prover.prove(pk, ctx)
    }

    fn verify(
        &self,
        vk: &MultiStarkVerifyingKeyV2<Self::SC>,
        proof: &Proof<Self::SC>,
    ) -> Result<(), VerifierError<<Self::SC as StarkProtocolConfig>::EF>>
    where
        <Self::SC as StarkProtocolConfig>::EF: p3_field::TwoAdicField,
    {
        let mut transcript = Self::TS::default();
        verify::<Self::SC, _>(vk, proof, &mut transcript)
    }

    /// The indexing of AIR ID in `ctx` should be consistent with the order of `airs`. In
    /// particular, `airs` should correspond to the global proving key with all AIRs, including ones
    /// not present in the `ctx`.
    fn debug(&self, airs: &[AirRef<Self::SC>], ctx: &ProvingContextV2<Self::PB>) {
        let mut keygen_builder = MultiStarkKeygenBuilderV2::new(self.config().clone());
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
            .map(|(air_id, air_ctx)| {
                let common_main = self
                    .device()
                    .transport_matrix_from_device_to_host(&air_ctx.common_main);
                let cached_mains = air_ctx
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
                let public_values = air_ctx.public_values.clone();
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
        ctxs: Vec<AirProvingContextV2<Self::PB>>,
    ) -> Result<VerificationDataV2<Self::SC>, VerifierError<<Self::SC as StarkProtocolConfig>::EF>>
    where
        Self::PB: ProverBackendV2<
            Val = <Self::SC as StarkProtocolConfig>::F,
            Challenge = <Self::SC as StarkProtocolConfig>::EF,
            Commitment = <Self::SC as StarkProtocolConfig>::Digest,
        >,
        <Self::SC as StarkProtocolConfig>::EF: p3_field::TwoAdicField;
}

/// [StarkEngineV2] that can be constructed from only system parameters.
pub trait DefaultStarkEngine: StarkEngineV2 {
    fn new(params: SystemParams) -> Self;
}
