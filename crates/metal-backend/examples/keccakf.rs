//! Prove keccakf-air over BabyBear using poseidon2 on Metal

use std::sync::Arc;

use eyre::eyre;
use openvm_metal_backend::BabyBearPoseidon2MetalEngine;
use openvm_stark_sdk::{
    config::log_up_params::log_up_security_params_baby_bear_100_bits,
    openvm_stark_backend::{
        p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues},
        p3_field::Field,
        prover::{AirProvingContext, ColMajorMatrix, DeviceDataTransporter, ProvingContext},
        PartitionedBaseAir, StarkEngine, SystemParams, WhirConfig, WhirParams,
    },
    utils::setup_tracing,
};
use p3_keccak_air::KeccakAir;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing::info_span;

const NUM_PERMUTATIONS: usize = 1 << 10;

// Newtype to implement extended traits.
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

fn main() -> eyre::Result<()> {
    setup_tracing();
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

    let engine = BabyBearPoseidon2MetalEngine::new(params);
    let (pk, vk) = engine.keygen(&[Arc::new(air)]);

    let inputs = (0..NUM_PERMUTATIONS)
        .map(|_| rng.random())
        .collect::<Vec<_>>();
    let trace = info_span!("generate_trace").in_scope(|| {
        p3_keccak_air::generate_trace_rows::<openvm_stark_sdk::p3_baby_bear::BabyBear>(inputs, 0)
    });

    let trace_ctx = AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace));
    let d_pk = engine.device().transport_pk_to_device(&pk);
    let proving_ctx = ProvingContext::new(vec![(0, trace_ctx)]);
    let d_ctx = engine
        .device()
        .transport_proving_ctx_to_device(&proving_ctx);
    let proof = engine.prove(&d_pk, d_ctx);

    engine
        .verify(&vk, &proof)
        .map_err(|e| eyre!("Proof failed to verify: {e}"))
}
