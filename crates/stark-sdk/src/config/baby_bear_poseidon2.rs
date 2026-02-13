use std::{
    io::{self, Read, Write},
    marker::PhantomData,
    sync::OnceLock,
};

use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear, Poseidon2BabyBear};
use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing};
use stark_backend_v2::{
    codec::{
        decode_extension_field32, decode_prime_field32, encode_extension_field32,
        encode_prime_field32, DecodableConfig, EncodableConfig,
    },
    duplex_sponge,
    hasher::Hasher,
    p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation},
    prover::{
        AirProvingContextV2, CoordinatorV2, CpuBackendV2, CpuDeviceV2, DeviceDataTransporterV2,
        ProverBackendV2, ProvingContextV2,
    },
    verifier::VerifierError,
    AirRef, DefaultStarkEngine, FiatShamirTranscript, StarkEngineV2, StarkProtocolConfig,
    SystemParams, TranscriptLog, VerificationDataV2,
};

const RATE: usize = 8;
/// permutation width
const WIDTH: usize = 16; // rate + capacity
pub const CHUNK: usize = 8;
pub const DIGEST_SIZE: usize = CHUNK;

type Perm = Poseidon2BabyBear<WIDTH>;
// Generic over P: CryptographicPermutation<[F; WIDTH]>
type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_SIZE>;
type Compress<P> = TruncatedPermutation<P, 2, CHUNK, WIDTH>;
type PermHasher<P> = Hasher<F, Digest, Hash<P>, Compress<P>>;
// Defined below
type SC = BabyBearPoseidon2ConfigV2;

// Convenience type aliases
pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, 4>;
pub const D_EF: usize = 4;
pub type Digest = [F; DIGEST_SIZE];
pub type DuplexSponge = duplex_sponge::DuplexSponge<F, Perm, WIDTH, RATE>;
pub type DuplexSpongeRecorder = duplex_sponge::DuplexSpongeRecorder<F, Perm, WIDTH, RATE>;
pub type DuplexSpongeValidator = duplex_sponge::DuplexSpongeValidator<F, Perm, WIDTH, RATE>;

#[derive(Clone, Debug, derive_new::new)]
pub struct BabyBearPoseidon2ConfigV2 {
    params: SystemParams,
    hasher: PermHasher<Perm>,
}

impl StarkProtocolConfig for BabyBearPoseidon2ConfigV2 {
    type F = F;
    type EF = EF;
    type Digest = Digest;
    type Hasher = PermHasher<Perm>;

    fn params(&self) -> &SystemParams {
        &self.params
    }

    fn hasher(&self) -> &Self::Hasher {
        &self.hasher
    }
}

impl BabyBearPoseidon2ConfigV2 {
    pub fn default_from_params(params: SystemParams) -> Self {
        let perm = default_babybear_poseidon2_16();
        let hasher = Hasher::new(
            PaddingFreeSponge::new(perm.clone()),
            TruncatedPermutation::new(perm),
        );
        Self::new(params, hasher)
    }
}

impl EncodableConfig for BabyBearPoseidon2ConfigV2 {
    fn encode_base_field<W: Write>(val: &F, writer: &mut W) -> io::Result<()> {
        encode_prime_field32(val, writer)
    }

    fn encode_extension_field<W: Write>(val: &EF, writer: &mut W) -> io::Result<()> {
        encode_extension_field32::<Self::F, _, _>(val, writer)
    }

    fn encode_digest<W: Write>(digest: &Self::Digest, writer: &mut W) -> io::Result<()> {
        for val in digest {
            encode_prime_field32(val, writer)?;
        }
        Ok(())
    }
}

impl DecodableConfig for BabyBearPoseidon2ConfigV2 {
    fn decode_base_field<R: Read>(reader: &mut R) -> io::Result<F> {
        decode_prime_field32(reader)
    }

    fn decode_extension_field<R: Read>(reader: &mut R) -> io::Result<EF> {
        decode_extension_field32::<F, _, _>(reader)
    }

    fn decode_digest<R: Read>(reader: &mut R) -> io::Result<Digest> {
        let mut result = Digest::default();
        for val in &mut result {
            *val = decode_prime_field32(reader)?;
        }
        Ok(result)
    }
}

pub struct BabyBearPoseidon2CpuEngineV2<TS = DuplexSponge> {
    device: CpuDeviceV2<SC>,
    _transcript: PhantomData<TS>,
}

impl<TS> StarkEngineV2 for BabyBearPoseidon2CpuEngineV2<TS>
where
    TS: FiatShamirTranscript<SC> + From<Perm>,
{
    type SC = SC;
    type PB = CpuBackendV2<SC>;
    type PD = CpuDeviceV2<SC>;
    type TS = TS;

    fn config(&self) -> &SC {
        self.device.config()
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn initial_transcript(&self) -> Self::TS {
        TS::from(default_babybear_poseidon2_16())
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

impl<TS> DefaultStarkEngine for BabyBearPoseidon2CpuEngineV2<TS>
where
    TS: FiatShamirTranscript<BabyBearPoseidon2ConfigV2> + From<Perm>,
{
    fn new(params: SystemParams) -> Self {
        let config = BabyBearPoseidon2ConfigV2::default_from_params(params);
        Self {
            device: CpuDeviceV2::new(config),
            _transcript: PhantomData,
        }
    }
}

// Fixed Poseidon2 configuration
pub fn poseidon2_perm() -> &'static Poseidon2BabyBear<WIDTH> {
    static PERM: OnceLock<Poseidon2BabyBear<WIDTH>> = OnceLock::new();
    PERM.get_or_init(default_babybear_poseidon2_16)
}

pub fn poseidon2_compress_with_capacity(
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> ([F; CHUNK], [F; CHUNK]) {
    let mut state = [F::ZERO; WIDTH];
    state[..CHUNK].copy_from_slice(&left);
    state[CHUNK..].copy_from_slice(&right);
    poseidon2_perm().permute_mut(&mut state);
    (
        state[..CHUNK].try_into().unwrap(),
        state[CHUNK..].try_into().unwrap(),
    )
}

pub fn default_duplex_sponge() -> DuplexSponge {
    DuplexSponge::from(poseidon2_perm().clone())
}

pub fn default_duplex_sponge_recorder() -> DuplexSpongeRecorder {
    DuplexSpongeRecorder::from(poseidon2_perm().clone())
}

pub fn default_duplex_sponge_validator(
    logs: TranscriptLog<F, [F; WIDTH]>,
) -> DuplexSpongeValidator {
    DuplexSpongeValidator::new(poseidon2_perm().clone(), logs)
}

#[cfg(test)]
mod poseidon2_constant_tests {
    use p3_baby_bear::{
        BABYBEAR_RC16_EXTERNAL_FINAL, BABYBEAR_RC16_EXTERNAL_INITIAL, BABYBEAR_RC16_INTERNAL,
    };
    use zkhash::{
        ark_ff::PrimeField as _, fields::babybear::FpBabyBear as HorizenBabyBear,
        poseidon2::poseidon2_instance_babybear::RC16,
    };

    use super::*;

    fn horizen_to_p3(horizen_babybear: HorizenBabyBear) -> BabyBear {
        BabyBear::from_u64(horizen_babybear.into_bigint().0[0])
    }

    pub fn horizen_round_consts_16() -> ((Vec<[BabyBear; 16]>, Vec<[BabyBear; 16]>), Vec<BabyBear>)
    {
        let p3_rc16: Vec<Vec<BabyBear>> = RC16
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|babybear| horizen_to_p3(*babybear))
                    .collect()
            })
            .collect();

        let rounds_f = 8;
        let rounds_p = 13;
        let rounds_f_beginning = rounds_f / 2;
        let p_end = rounds_f_beginning + rounds_p;
        let initial: Vec<[BabyBear; 16]> = p3_rc16[..rounds_f_beginning]
            .iter()
            .cloned()
            .map(|round| round.try_into().unwrap())
            .collect();
        let terminal: Vec<[BabyBear; 16]> = p3_rc16[p_end..]
            .iter()
            .cloned()
            .map(|round| round.try_into().unwrap())
            .collect();
        let internal_round_constants: Vec<BabyBear> = p3_rc16[rounds_f_beginning..p_end]
            .iter()
            .map(|round| round[0])
            .collect();
        ((initial, terminal), internal_round_constants)
    }

    /// Uses HorizenLabs Poseidon2 round constants, but plonky3 Mat4 and also
    /// with a p3 Monty reduction factor.
    #[test]
    fn test_horizen_p3_rc_equality() {
        let ((external_initial, external_terminal), internal_constants) = horizen_round_consts_16();
        assert_eq!(external_initial, BABYBEAR_RC16_EXTERNAL_INITIAL.to_vec());
        assert_eq!(external_terminal, BABYBEAR_RC16_EXTERNAL_FINAL.to_vec());
        assert_eq!(internal_constants, BABYBEAR_RC16_INTERNAL.to_vec());
    }
}
