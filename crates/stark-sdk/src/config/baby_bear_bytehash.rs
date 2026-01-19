use std::sync::Arc;

use openvm_stark_backend::{
    config::StarkConfig,
    interaction::fri_log_up::FriLogUpPhase,
    p3_challenger::{HashChallenger, SerializingChallenger32},
    p3_commit::ExtensionMmcs,
    p3_field::extension::BinomialExtensionField,
    prover::{
        cpu::{CpuBackend, CpuDevice},
        MultiTraceStarkProver,
    },
};
use p3_baby_bear::BabyBear;
use p3_dft::Radix2DitParallel;
use p3_fri::{FriParameters as P3FriParameters, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, SerializingHasher};

use super::FriParameters;
use crate::{
    config::fri_params::SecurityParameters,
    engine::{StarkEngine, StarkFriEngine},
};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

// Generic over H: CryptographicHasher<u8, [u8; 32]>
type FieldHash<H> = SerializingHasher<H>;
type Compress<H> = CompressionFunctionFromHasher<H, 2, 32>;
// type InstrCompress<H> = Instrumented<Compress<H>>;

type ValMmcs<H> = MerkleTreeMmcs<Val, u8, FieldHash<H>, Compress<H>, 32>;
type ChallengeMmcs<H> = ExtensionMmcs<Val, Challenge, ValMmcs<H>>;
type Dft = Radix2DitParallel<Val>;
type Challenger<H> = SerializingChallenger32<Val, HashChallenger<u8, H, 32>>;

type Pcs<H> = TwoAdicFriPcs<Val, Dft, ValMmcs<H>, ChallengeMmcs<H>>;

type RapPhase<H> = FriLogUpPhase<Val, Challenge, Challenger<H>>;

pub type BabyBearByteHashConfig<H> = StarkConfig<Pcs<H>, RapPhase<H>, Challenge, Challenger<H>>;

pub struct BabyBearByteHashEngine<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    pub fri_params: FriParameters,
    pub device: CpuDevice<BabyBearByteHashConfig<H>>,
    pub byte_hash: H,
    pub max_constraint_degree: usize,
}

impl<H> StarkEngine for BabyBearByteHashEngine<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone + Send + Sync,
{
    type SC = BabyBearByteHashConfig<H>;
    type PB = CpuBackend<Self::SC>;
    type PD = CpuDevice<Self::SC>;

    fn config(&self) -> &BabyBearByteHashConfig<H> {
        &self.device.config
    }

    fn device(&self) -> &CpuDevice<BabyBearByteHashConfig<H>> {
        &self.device
    }

    fn prover(&self) -> MultiTraceStarkProver<BabyBearByteHashConfig<H>> {
        MultiTraceStarkProver::new(
            CpuBackend::default(),
            self.device.clone(),
            self.new_challenger(),
        )
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(self.max_constraint_degree)
    }

    fn new_challenger(&self) -> Challenger<H> {
        Challenger::from_hasher(vec![], self.byte_hash.clone())
    }
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine<H>(byte_hash: H) -> BabyBearByteHashEngine<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    engine_from_byte_hash(byte_hash, SecurityParameters::standard_fast())
}

pub fn engine_from_byte_hash<H>(
    byte_hash: H,
    security_params: SecurityParameters,
) -> BabyBearByteHashEngine<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    let fri_params = security_params.fri_params;
    let max_constraint_degree = fri_params.max_constraint_degree();
    let config = config_from_byte_hash(byte_hash.clone(), security_params);
    BabyBearByteHashEngine {
        device: CpuDevice::new(Arc::new(config), fri_params.log_blowup),
        byte_hash,
        fri_params,
        max_constraint_degree,
    }
}

pub fn config_from_byte_hash<H>(
    byte_hash: H,
    security_params: SecurityParameters,
) -> BabyBearByteHashConfig<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    let field_hash = FieldHash::new(byte_hash.clone());
    let compress = Compress::new(byte_hash.clone());
    let val_mmcs = ValMmcs::new(field_hash, compress);
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
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    let rap_phase = FriLogUpPhase::new(log_up_params, fri_params.log_blowup);
    BabyBearByteHashConfig::new(pcs, challenger, rap_phase, deep_ali_params)
}

pub trait BabyBearByteHashEngineWithDefaultHash<H>
where
    H: CryptographicHasher<u8, [u8; 32]> + Clone,
{
    fn default_hash() -> H;
}

impl<H: CryptographicHasher<u8, [u8; 32]> + Clone + Send + Sync> StarkFriEngine
    for BabyBearByteHashEngine<H>
where
    BabyBearByteHashEngine<H>: BabyBearByteHashEngineWithDefaultHash<H>,
{
    fn new(fri_params: FriParameters) -> Self {
        let security_params = SecurityParameters::new_baby_bear_100_bits(fri_params);
        engine_from_byte_hash(Self::default_hash(), security_params)
    }
    fn fri_params(&self) -> FriParameters {
        self.fri_params
    }
}
