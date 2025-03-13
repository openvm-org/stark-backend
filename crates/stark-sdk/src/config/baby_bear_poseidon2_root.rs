use ff::PrimeField;
use openvm_stark_backend::{
    config::StarkConfig, interaction::fri_log_up::FriLogUpPhase,
    p3_challenger::MultiField32Challenger, p3_commit::ExtensionMmcs,
    p3_field::extension::BinomialExtensionField,
};
use p3_baby_bear::BabyBear;
use p3_bn254_fr::{Bn254Fr, FFBn254Fr, Poseidon2Bn254};
use p3_dft::Radix2DitParallel;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{CryptographicPermutation, MultiField32PaddingFreeSponge, TruncatedPermutation};
use zkhash::{
    ark_ff::{BigInteger, PrimeField as _},
    fields::bn256::FpBN256 as ark_FpBN256,
    poseidon2::poseidon2_instance_bn256::RC3,
};

use super::FriParameters;
use crate::{
    assert_sc_compatible_with_serde,
    config::{
        fri_params::SecurityParameters, log_up_params::log_up_security_params_baby_bear_100_bits,
    },
    engine::{StarkEngine, StarkFriEngine},
};

const WIDTH: usize = 3;
/// Poseidon rate in F. <Poseidon RATE>(2) * <# of F in a N>(8) = 16
const RATE: usize = 16;
const DIGEST_WIDTH: usize = 1;

/// A configuration for  recursion.
type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2Bn254<WIDTH>;
type Hash<P> = MultiField32PaddingFreeSponge<Val, Bn254Fr, P, WIDTH, RATE, DIGEST_WIDTH>;
type Compress<P> = TruncatedPermutation<P, 2, 1, WIDTH>;
type ValMmcs<P> = MerkleTreeMmcs<BabyBear, Bn254Fr, Hash<P>, Compress<P>, 1>;
type ChallengeMmcs<P> = ExtensionMmcs<Val, Challenge, ValMmcs<P>>;
type Dft = Radix2DitParallel<Val>;
type Challenger<P> = MultiField32Challenger<Val, Bn254Fr, P, WIDTH, 2>;
type Pcs<P> = TwoAdicFriPcs<Val, Dft, ValMmcs<P>, ChallengeMmcs<P>>;
type RapPhase<P> = FriLogUpPhase<Val, Challenge, Challenger<P>>;

pub type BabyBearPermutationRootConfig<P> =
    StarkConfig<Pcs<P>, RapPhase<P>, Challenge, Challenger<P>>;
pub type BabyBearPoseidon2RootConfig = BabyBearPermutationRootConfig<Perm>;
pub type BabyBearPoseidon2RootEngine = BabyBearPermutationRootEngine<Perm>;

assert_sc_compatible_with_serde!(BabyBearPoseidon2RootConfig);

pub struct BabyBearPermutationRootEngine<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    pub fri_params: FriParameters,
    pub config: BabyBearPermutationRootConfig<P>,
    pub perm: P,
    pub max_constraint_degree: usize,
}

impl<P> StarkEngine<BabyBearPermutationRootConfig<P>> for BabyBearPermutationRootEngine<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    fn config(&self) -> &BabyBearPermutationRootConfig<P> {
        &self.config
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(self.max_constraint_degree)
    }

    fn new_challenger(&self) -> Challenger<P> {
        Challenger::new(self.perm.clone()).unwrap()
    }
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine() -> BabyBearPoseidon2RootEngine {
    default_engine_impl(SecurityParameters::standard_fast())
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
fn default_engine_impl(security_params: SecurityParameters) -> BabyBearPoseidon2RootEngine {
    let perm = root_perm();
    engine_from_perm(perm, security_params)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config(perm: &Perm) -> BabyBearPoseidon2RootConfig {
    config_from_perm(perm, SecurityParameters::standard_fast())
}

pub fn engine_from_perm<P>(
    perm: P,
    security_params: SecurityParameters,
) -> BabyBearPermutationRootEngine<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    let fri_params = security_params.fri_params;
    let max_constraint_degree = fri_params.max_constraint_degree();
    let config = config_from_perm(&perm, security_params);
    BabyBearPermutationRootEngine {
        config,
        perm,
        fri_params,
        max_constraint_degree,
    }
}

pub fn config_from_perm<P>(
    perm: &P,
    security_params: SecurityParameters,
) -> BabyBearPermutationRootConfig<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    let hash = Hash::new(perm.clone()).unwrap();
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let SecurityParameters {
        fri_params,
        log_up_params,
    } = security_params;
    let fri_config = FriConfig {
        log_blowup: fri_params.log_blowup,
        log_final_poly_len: fri_params.log_final_poly_len,
        num_queries: fri_params.num_queries,
        proof_of_work_bits: fri_params.proof_of_work_bits,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let rap_phase = FriLogUpPhase::new(log_up_params);
    BabyBearPermutationRootConfig::new(pcs, rap_phase)
}

/// The permutation for outer recursion.
pub fn root_perm() -> Perm {
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

fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254Fr {
    let bytes = input.into_bigint().to_bytes_le();

    let mut res = <FFBn254Fr as ff::PrimeField>::Repr::default();

    for (i, digit) in res.as_mut().iter_mut().enumerate() {
        *digit = bytes[i];
    }

    let value = FFBn254Fr::from_repr(res);

    if value.is_some().into() {
        Bn254Fr {
            value: value.unwrap(),
        }
    } else {
        panic!("Invalid field element")
    }
}

fn bn254_poseidon2_rc3() -> Vec<[Bn254Fr; 3]> {
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

impl StarkFriEngine<BabyBearPoseidon2RootConfig> for BabyBearPoseidon2RootEngine {
    fn new(fri_params: FriParameters) -> Self {
        let security_params = SecurityParameters {
            fri_params,
            log_up_params: log_up_security_params_baby_bear_100_bits(),
        };
        default_engine_impl(security_params)
    }
    fn fri_params(&self) -> FriParameters {
        self.fri_params
    }
}
