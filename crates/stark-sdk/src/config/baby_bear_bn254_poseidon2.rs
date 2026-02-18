use std::{marker::PhantomData, sync::OnceLock};

use openvm_stark_backend::{
    hasher::Hasher,
    p3_challenger::{CanObserve, CanSample, MultiField32Challenger},
    p3_symmetric::{self, MultiField32PaddingFreeSponge, TruncatedPermutation},
    prover::{Coordinator, CpuBackend, CpuDevice},
    FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams,
};
use p3_baby_bear::BabyBear;
// NOTE: plonky3's Bn254 is the type for scalar field of the BN254 curve. It is not the type
// for the curve itself.
pub use p3_bn254::Bn254 as Bn254Scalar;
use p3_bn254::Poseidon2Bn254;
use p3_field::extension::BinomialExtensionField;
use p3_poseidon2::ExternalLayerConstants;
use zkhash::{
    ark_ff::PrimeField as _, fields::bn256::FpBN256 as ark_FpBN256,
    poseidon2::poseidon2_instance_bn256::RC3,
};

const WIDTH: usize = 3;
/// Poseidon rate in F. <Poseidon RATE>(2) * <# of F in a N>(8) = 16
const BABY_BEAR_RATE: usize = 16;
const BN254_RATE: usize = 2;
/// Width in Bn254Fr
const DIGEST_WIDTH: usize = 1;

type Perm = Poseidon2Bn254<WIDTH>;
// Generic over P: CryptographicPermutation<[F; WIDTH]>
type H = MultiField32PaddingFreeSponge<F, Bn254Scalar, Perm, WIDTH, BABY_BEAR_RATE, DIGEST_WIDTH>;
type Compress = TruncatedPermutation<Perm, 2, DIGEST_WIDTH, WIDTH>;
type PermHasher = Hasher<F, Digest, H, Compress>;
// Defined below
type SC = BabyBearBn254Poseidon2Config;

// Convenience type aliases
pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, 4>;
pub const D_EF: usize = 4;
pub type Digest = [Bn254Scalar; DIGEST_WIDTH];

#[derive(Clone, Debug, derive_new::new)]
pub struct BabyBearBn254Poseidon2Config {
    params: SystemParams,
    hasher: PermHasher,
}

impl StarkProtocolConfig for BabyBearBn254Poseidon2Config {
    type F = F;
    type EF = EF;
    type Digest = Digest;
    type Hasher = PermHasher;

    fn params(&self) -> &SystemParams {
        &self.params
    }

    fn hasher(&self) -> &Self::Hasher {
        &self.hasher
    }
}

impl BabyBearBn254Poseidon2Config {
    pub fn new_from_perm(params: SystemParams, perm: Perm) -> Self {
        let hasher = Hasher::new(
            MultiField32PaddingFreeSponge::new(perm.clone()).unwrap(),
            TruncatedPermutation::new(perm),
        );
        Self { params, hasher }
    }

    pub fn default_from_params(params: SystemParams) -> Self {
        let perm = default_babybear_bn254_poseidon2();
        Self::new_from_perm(params, perm)
    }
}

#[derive(Clone, Debug)]
pub struct Transcript {
    inner: MultiField32Challenger<F, Bn254Scalar, Perm, WIDTH, BN254_RATE>,
}

impl FiatShamirTranscript<SC> for Transcript {
    fn observe(&mut self, value: F) {
        CanObserve::observe(&mut self.inner, value);
    }

    fn sample(&mut self) -> F {
        CanSample::sample(&mut self.inner)
    }

    fn observe_commit(&mut self, digest: Digest) {
        CanObserve::<p3_symmetric::Hash<_, _, _>>::observe(&mut self.inner, digest.into());
    }
}

impl From<Perm> for Transcript {
    fn from(perm: Perm) -> Self {
        Self {
            inner: MultiField32Challenger::new(perm).unwrap(),
        }
    }
}

pub struct BabyBearBn254Poseidon2CpuEngine<TS = Transcript> {
    device: CpuDevice<SC>,
    _transcript: PhantomData<TS>,
}

impl<TS> StarkEngine for BabyBearBn254Poseidon2CpuEngine<TS>
where
    TS: FiatShamirTranscript<SC> + From<Perm>,
{
    type SC = SC;
    type PB = CpuBackend<SC>;
    type PD = CpuDevice<SC>;
    type TS = TS;

    fn new(params: SystemParams) -> Self {
        let config = BabyBearBn254Poseidon2Config::default_from_params(params);
        Self {
            device: CpuDevice::new(config),
            _transcript: PhantomData,
        }
    }

    fn config(&self) -> &SC {
        self.device.config()
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn initial_transcript(&self) -> Self::TS {
        TS::from(default_babybear_bn254_poseidon2())
    }

    fn prover_from_transcript(
        &self,
        transcript: TS,
    ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
        Coordinator::new(CpuBackend::new(), self.device.clone(), transcript)
    }
}

pub fn default_babybear_bn254_poseidon2() -> Perm {
    static PERM: OnceLock<Perm> = OnceLock::new();
    PERM.get_or_init(default_perm).clone()
}

pub fn default_transcript() -> Transcript {
    Transcript::from(default_babybear_bn254_poseidon2())
}

/// The permutation for outer recursion.
fn default_perm() -> Perm {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 56;
    let mut round_constants = bn254_poseidon2_rc3();
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
    let terminal = round_constants.split_off(internal_end);
    let internal_round_constants = round_constants.split_off(ROUNDS_F / 2);
    let internal_round_constants = internal_round_constants
        .into_iter()
        .map(|vec| vec[0])
        .collect::<Vec<_>>();
    let initial = round_constants;

    let external_round_constants = ExternalLayerConstants::new(initial, terminal);
    Perm::new(external_round_constants, internal_round_constants)
}

fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254Scalar {
    let limbs_le = input.into_bigint().0;
    // arkworks limbs are little-endian u64s; convert to BigUint in little-endian
    let bytes = limbs_le
        .iter()
        .flat_map(|limb| limb.to_le_bytes())
        .collect::<Vec<_>>();
    let big = num_bigint::BigUint::from_bytes_le(&bytes);
    Bn254Scalar::from_biguint(big).expect("Invalid BN254 element")
}

fn bn254_poseidon2_rc3() -> Vec<[Bn254Scalar; 3]> {
    RC3.iter()
        .map(|vec| {
            vec.iter()
                .cloned()
                .map(bn254_from_ark_ff)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect()
}
