use std::sync::Arc;

use openvm_stark_backend::{
    config::StarkConfig,
    interaction::fri_log_up::FriLogUpPhase,
    keygen::MultiStarkKeygenBuilder,
    p3_challenger::MultiField32Challenger,
    p3_commit::ExtensionMmcs,
    p3_field::extension::BinomialExtensionField,
    prover::{
        cpu::{CpuBackend, CpuDevice},
        MultiTraceStarkProver,
    },
};
use p3_baby_bear::BabyBear;
use p3_bn254::{Bn254, Poseidon2Bn254};
use p3_dft::Radix2DitParallel;
use p3_fri::{FriParameters as P3FriParameters, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{CryptographicPermutation, MultiField32PaddingFreeSponge, TruncatedPermutation};
use zkhash::{
    ark_ff::PrimeField as _, fields::bn256::FpBN256 as ark_FpBN256,
    poseidon2::poseidon2_instance_bn256::RC3,
};

use super::FriParameters;
use crate::{
    assert_sc_compatible_with_serde,
    config::fri_params::{
        SecurityParameters, MAX_BATCH_SIZE_LOG_BLOWUP_1, MAX_BATCH_SIZE_LOG_BLOWUP_2,
        MAX_NUM_CONSTRAINTS,
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
type Hash<P> = MultiField32PaddingFreeSponge<Val, Bn254, P, WIDTH, RATE, DIGEST_WIDTH>;
type Compress<P> = TruncatedPermutation<P, 2, 1, WIDTH>;
type ValMmcs<P> = MerkleTreeMmcs<BabyBear, Bn254, Hash<P>, Compress<P>, 1>;
type ChallengeMmcs<P> = ExtensionMmcs<Val, Challenge, ValMmcs<P>>;
type Dft = Radix2DitParallel<Val>;
type Challenger<P> = MultiField32Challenger<Val, Bn254, P, WIDTH, 2>;
type Pcs<P> = TwoAdicFriPcs<Val, Dft, ValMmcs<P>, ChallengeMmcs<P>>;
type RapPhase<P> = FriLogUpPhase<Val, Challenge, Challenger<P>>;

pub type BabyBearPermutationRootConfig<P> =
    StarkConfig<Pcs<P>, RapPhase<P>, Challenge, Challenger<P>>;
pub type BabyBearPoseidon2RootConfig = BabyBearPermutationRootConfig<Perm>;
pub type BabyBearPoseidon2RootEngine = BabyBearPermutationRootEngine<Perm>;

assert_sc_compatible_with_serde!(BabyBearPoseidon2RootConfig);

pub struct BabyBearPermutationRootEngine<P>
where
    P: CryptographicPermutation<[Bn254; WIDTH]> + Clone,
{
    pub fri_params: FriParameters,
    pub device: CpuDevice<BabyBearPermutationRootConfig<P>>,
    pub perm: P,
    pub max_constraint_degree: usize,
}

impl<P> StarkEngine for BabyBearPermutationRootEngine<P>
where
    P: CryptographicPermutation<[Bn254; WIDTH]> + Clone,
{
    type SC = BabyBearPermutationRootConfig<P>;
    type PB = CpuBackend<Self::SC>;
    type PD = CpuDevice<Self::SC>;

    fn config(&self) -> &BabyBearPermutationRootConfig<P> {
        &self.device.config
    }

    fn device(&self) -> &CpuDevice<BabyBearPermutationRootConfig<P>> {
        &self.device
    }

    fn keygen_builder(&self) -> MultiStarkKeygenBuilder<'_, Self::SC> {
        let mut builder = MultiStarkKeygenBuilder::new(self.config());
        builder.set_max_constraint_degree(self.max_constraint_degree);
        let max_batch_size = if self.fri_params.log_blowup == 1 {
            MAX_BATCH_SIZE_LOG_BLOWUP_1
        } else {
            MAX_BATCH_SIZE_LOG_BLOWUP_2
        };
        builder.max_batch_size = Some(max_batch_size);
        builder.max_num_constraints = Some(MAX_NUM_CONSTRAINTS);

        builder
    }

    fn prover(&self) -> MultiTraceStarkProver<BabyBearPermutationRootConfig<P>> {
        MultiTraceStarkProver::new(
            CpuBackend::default(),
            self.device.clone(),
            self.new_challenger(),
        )
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
    P: CryptographicPermutation<[Bn254; WIDTH]> + Clone,
{
    let fri_params = security_params.fri_params;
    let max_constraint_degree = fri_params.max_constraint_degree();
    let config = config_from_perm(&perm, security_params);
    BabyBearPermutationRootEngine {
        device: CpuDevice::new(Arc::new(config), fri_params.log_blowup),
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
    P: CryptographicPermutation<[Bn254; WIDTH]> + Clone,
{
    let hash = Hash::new(perm.clone()).unwrap();
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let SecurityParameters {
        fri_params,
        log_up_params,
        deep_ali_params,
    } = security_params;
    let fri_config = P3FriParameters {
        log_blowup: fri_params.log_blowup,
        log_final_poly_len: fri_params.log_final_poly_len,
        num_queries: fri_params.num_queries,
        commit_proof_of_work_bits: fri_params.commit_proof_of_work_bits,
        query_proof_of_work_bits: fri_params.query_proof_of_work_bits,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let challenger = Challenger::new(perm.clone()).unwrap();
    let rap_phase = FriLogUpPhase::new(log_up_params, fri_params.log_blowup);
    BabyBearPermutationRootConfig::new(pcs, challenger, rap_phase, deep_ali_params)
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

fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254 {
    let limbs_le = input.into_bigint().0;
    // arkworks limbs are little-endian u64s; convert to BigUint in little-endian
    let bytes = limbs_le
        .iter()
        .flat_map(|limb| limb.to_le_bytes())
        .collect::<Vec<_>>();
    let big = num_bigint::BigUint::from_bytes_le(&bytes);
    Bn254::from_biguint(big).expect("Invalid BN254 element")
}

fn bn254_poseidon2_rc3() -> Vec<[Bn254; 3]> {
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

impl StarkFriEngine for BabyBearPoseidon2RootEngine {
    fn new(fri_params: FriParameters) -> Self {
        let security_params = SecurityParameters::new_baby_bear_100_bits(fri_params);
        default_engine_impl(security_params)
    }
    fn fri_params(&self) -> FriParameters {
        self.fri_params
    }
}
