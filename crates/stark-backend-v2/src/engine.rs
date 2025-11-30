// Replace engine.rs in v1
// TODO[jpw]: everything is currently assuming fixed types for:
// - F, EF, Digest, SystemParams
// We will make these generic in the future

use std::marker::PhantomData;

use openvm_stark_backend::{config::StarkGenericConfig, prover::Prover, AirRef};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    keygen::{
        types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
        MultiStarkKeygenBuilderV2,
    },
    poseidon2::sponge::{DuplexSponge, FiatShamirTranscript},
    proof::*,
    prover::{
        AirProvingContextV2, CoordinatorV2, CpuBackendV2, CpuDeviceV2, DeviceDataTransporterV2,
        DeviceMultiStarkProvingKeyV2, MultiRapProver, OpeningProverV2, ProverBackendV2,
        ProverDeviceV2, ProvingContextV2,
    },
    verifier::{verify, VerifierError},
    SystemParams,
};

/// Data for verifying a Stark proof.
#[derive(Debug)]
pub struct VerificationDataV2 {
    pub vk: MultiStarkVerifyingKeyV2,
    pub proof: Proof,
}

/// A helper trait to collect the different steps in multi-trace STARK
/// keygen and proving. Currently this trait is CPU specific.
pub trait StarkEngineV2
where
    <Self::PD as MultiRapProver<Self::PB, Self::TS>>::Artifacts:
        Into<<Self::PD as OpeningProverV2<Self::PB, Self::TS>>::OpeningPoints>,
    <Self::PD as MultiRapProver<Self::PB, Self::TS>>::PartialProof:
        Into<(GkrProof, BatchConstraintProof)>,
    <Self::PD as OpeningProverV2<Self::PB, Self::TS>>::OpeningProof:
        Into<(StackingProof, WhirProof)>,
{
    type SC: StarkGenericConfig<
        Pcs = <BabyBearPoseidon2Config as StarkGenericConfig>::Pcs,
        Challenge = <BabyBearPoseidon2Config as StarkGenericConfig>::Challenge,
        Challenger = <BabyBearPoseidon2Config as StarkGenericConfig>::Challenger,
    >;
    type PB: ProverBackendV2<Val = crate::F, Challenge = crate::EF, Commitment = crate::Digest>;
    type PD: ProverDeviceV2<Self::PB, Self::TS> + DeviceDataTransporterV2<Self::PB>;
    type TS: FiatShamirTranscript + Default;

    fn config(&self) -> SystemParams {
        self.device().config()
    }

    fn device(&self) -> &Self::PD;

    // TODO[jpw]: keygen builder

    fn prover_from_transcript(
        &self,
        transcript: Self::TS,
    ) -> CoordinatorV2<Self::PB, Self::PD, Self::TS>;

    fn prover(&self) -> CoordinatorV2<Self::PB, Self::PD, Self::TS> {
        self.prover_from_transcript(Self::TS::default())
    }

    fn keygen(
        &self,
        airs: &[AirRef<BabyBearPoseidon2Config>],
    ) -> (MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2) {
        let mut keygen_builder = MultiStarkKeygenBuilderV2::new(self.config());
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
    ) -> Proof {
        let mut prover = self.prover();
        prover.prove(pk, ctx)
    }

    fn verify(&self, vk: &MultiStarkVerifyingKeyV2, proof: &Proof) -> Result<(), VerifierError> {
        let mut transcript = Self::TS::default();
        verify(vk, proof, &mut transcript)
    }

    /// Runs a single end-to-end test for a given set of chips and traces partitions.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    fn run_test(
        &self,
        airs: Vec<AirRef<BabyBearPoseidon2Config>>,
        ctxs: Vec<AirProvingContextV2<Self::PB>>,
    ) -> Result<VerificationDataV2, VerifierError> {
        let (pk, vk) = self.keygen(&airs);
        let device = self.prover().device;
        let d_pk = device.transport_pk_to_device(&pk);
        let ctx = ProvingContextV2::new(ctxs.into_iter().enumerate().collect());
        let proof = self.prove(&d_pk, ctx);
        self.verify(&vk, &proof)?;
        Ok(VerificationDataV2 { vk, proof })
    }
}

pub struct BabyBearPoseidon2CpuEngineV2<TS = DuplexSponge> {
    device: CpuDeviceV2,
    _transcript: PhantomData<TS>,
}

impl<TS> BabyBearPoseidon2CpuEngineV2<TS> {
    pub fn new(params: SystemParams) -> Self {
        Self {
            device: CpuDeviceV2::new(params),
            _transcript: PhantomData,
        }
    }
}

impl<TS> StarkEngineV2 for BabyBearPoseidon2CpuEngineV2<TS>
where
    TS: FiatShamirTranscript + Default,
{
    type SC = BabyBearPoseidon2Config;
    type PB = CpuBackendV2;
    type PD = CpuDeviceV2;
    type TS = TS;

    fn device(&self) -> &Self::PD {
        &self.device
    }
    fn prover_from_transcript(
        &self,
        transcript: TS,
    ) -> CoordinatorV2<Self::PB, Self::PD, Self::TS> {
        CoordinatorV2::new(CpuBackendV2, self.device, transcript)
    }
}

// TODO[jpw]: move to stark-sdk
pub trait StarkWhirEngine: StarkEngineV2 {
    fn new(params: SystemParams) -> Self;
}

impl<TS> StarkWhirEngine for BabyBearPoseidon2CpuEngineV2<TS>
where
    TS: FiatShamirTranscript + Default,
{
    fn new(params: SystemParams) -> Self {
        Self::new(params)
    }
}
