//! Prove keccakf-air over BabyBear using poseidon2

use std::sync::Arc;

use eyre::eyre;
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_keccak_air::KeccakAir;
use rand::{rngs::StdRng, Rng, SeedableRng};
use stark_backend_v2::{
    baby_bear_poseidon2::BabyBearPoseidon2ConfigV2,
    interaction::LogUpSecurityParameters,
    poseidon2::sponge::DuplexSponge,
    prover::{AirProvingContextV2, ColMajorMatrix, DeviceDataTransporterV2, ProvingContextV2},
    verifier::verify,
    BabyBearPoseidon2CpuEngineV2, PartitionedBaseAir, StarkEngineV2, SystemParams, WhirConfig,
    WhirParams,
};
use tracing::info_span;

const NUM_PERMUTATIONS: usize = 1 << 10;

// Newtype to implement extended traits
struct TestAir(KeccakAir);

impl<F> BaseAir<F> for TestAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.0)
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for TestAir {}
impl<F: Field> PartitionedBaseAir<F> for TestAir {}

impl<AB: AirBuilder> Air<AB> for TestAir {
    fn eval(&self, builder: &mut AB) {
        self.0.eval(builder);
    }
}

fn log_up_security_params_baby_bear_100_bits() -> LogUpSecurityParameters {
    use p3_field::{extension::BinomialExtensionField, PrimeField32};
    let params = LogUpSecurityParameters {
        max_interaction_count: BabyBear::ORDER_U32,
        log_max_message_length: 7,
        pow_bits: 18,
    };
    assert!(params.bits_of_security::<BinomialExtensionField<BabyBear, 4>>() >= 100);
    params
}

fn main() -> eyre::Result<()> {
    let l_skip = 4;
    let n_stack = 17;
    let k_whir = 4;
    let whir_params = WhirParams {
        k: k_whir,
        log_final_poly_len: 2 * k_whir,
        query_phase_pow_bits: 20,
    };
    let log_blowup = 1;
    let whir = WhirConfig::new(log_blowup, l_skip + n_stack, whir_params, 100);
    let params = SystemParams {
        l_skip,
        n_stack,
        log_blowup,
        whir,
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree: 3,
    };

    let mut rng = StdRng::seed_from_u64(42);
    let air = TestAir(KeccakAir {});

    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let (pk, vk) = engine.keygen(&[Arc::new(air)]);

    let inputs = (0..NUM_PERMUTATIONS)
        .map(|_| rng.random())
        .collect::<Vec<_>>();
    let trace = info_span!("generate_trace")
        .in_scope(|| p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0));

    let air_ctx = AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace));
    let d_pk = engine.device().transport_pk_to_device(&pk);
    let proof = engine.prove(&d_pk, ProvingContextV2::new(vec![(0, air_ctx)]));

    verify::<BabyBearPoseidon2ConfigV2, _>(&vk, &proof, &mut DuplexSponge::default())
        .map_err(|e| eyre!("Proof failed to verify: {e}"))
}
