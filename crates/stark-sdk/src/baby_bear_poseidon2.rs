use std::io::{Read, Result, Write};

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{
    codec::{
        decode_extension_field32, decode_prime_field32, encode_extension_field32,
        encode_prime_field32, DecodableConfig, EncodableConfig,
    },
    poseidon2::{sponge::Poseidon2Hasher, CHUNK},
    StarkProtocolConfig,
};

/// ZST config type for BabyBear + Poseidon2.
pub struct BabyBearPoseidon2ConfigV2;

impl StarkProtocolConfig for BabyBearPoseidon2ConfigV2 {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type Digest = [BabyBear; CHUNK];
    type H = Poseidon2Hasher;
}

impl EncodableConfig for BabyBearPoseidon2ConfigV2 {
    fn encode_base_field<W: Write>(val: &Self::F, writer: &mut W) -> Result<()> {
        encode_prime_field32(val, writer)
    }

    fn encode_extension_field<W: Write>(val: &Self::EF, writer: &mut W) -> Result<()> {
        encode_extension_field32::<Self::F, _, _>(val, writer)
    }

    fn encode_digest<W: Write>(digest: &Self::Digest, writer: &mut W) -> Result<()> {
        for val in digest {
            encode_prime_field32(val, writer)?;
        }
        Ok(())
    }
}

impl DecodableConfig for BabyBearPoseidon2ConfigV2 {
    fn decode_base_field<R: Read>(reader: &mut R) -> Result<Self::F> {
        decode_prime_field32(reader)
    }

    fn decode_extension_field<R: Read>(reader: &mut R) -> Result<Self::EF> {
        decode_extension_field32::<F, _, _>(reader)
    }

    fn decode_digest<R: Read>(reader: &mut R) -> Result<Self::Digest> {
        let mut result = Digest::default();
        for val in &mut result {
            *val = decode_prime_field32(reader)?;
        }
        Ok(result)
    }
}

// Convenience type aliases (for internal use in Phase 1)
pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, 4>;
pub const D_EF: usize = 4;
pub const DIGEST_SIZE: usize = CHUNK;
pub type Digest = [F; DIGEST_SIZE];

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
    TS: FiatShamirTranscript<BabyBearPoseidon2ConfigV2> + Default,
{
    type SC = BabyBearPoseidon2ConfigV2;
    type PB = CpuBackendV2<BabyBearPoseidon2ConfigV2>;
    type PD = CpuDeviceV2;
    type TS = TS;

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn prover_from_transcript(
        &self,
        transcript: TS,
    ) -> CoordinatorV2<Self::SC, Self::PB, Self::PD, Self::TS> {
        CoordinatorV2::new(CpuBackendV2::new(), self.device.clone(), transcript)
    }

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
        <Self::SC as StarkProtocolConfig>::EF: p3_field::TwoAdicField,
    {
        let (pk, vk) = self.keygen(&airs);
        let device = self.prover().device;
        let d_pk = device.transport_pk_to_device(&pk);
        let ctx = ProvingContextV2::new(ctxs.into_iter().enumerate().collect());
        let proof = self.prove(&d_pk, ctx);
        self.verify(&vk, &proof)?;
        Ok(VerificationDataV2 { vk, proof })
    }
}

impl<TS> StarkWhirEngine for BabyBearPoseidon2CpuEngineV2<TS>
where
    TS: FiatShamirTranscript<BabyBearPoseidon2ConfigV2> + Default,
{
    fn new(params: SystemParams) -> Self {
        Self::new(params)
    }
}
