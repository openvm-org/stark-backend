//! Prove keccakf-air over BabyBear using poseidon2 for FRI hash.

use std::sync::Arc;

use cuda_backend_v2::BabyBearPoseidon2GpuEngine;
use eyre::eyre;
use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    poseidon2::sponge::DuplexSponge,
    prover::{AirProvingContext, ColMajorMatrix, DeviceDataTransporter, ProvingContext},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    verifier::verify,
    StarkEngine, SystemParams, WhirConfig, WhirParams,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::log_up_params::log_up_security_params_baby_bear_100_bits,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing::trace_span;

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

    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<()> {
        let mut rng = StdRng::seed_from_u64(42);
        let air = TestAir(KeccakAir {});

        let engine = BabyBearPoseidon2GpuEngine::new(params);
        let (pk, vk) = engine.keygen(&[Arc::new(air)]);
        let air_idx = 0;

        let inputs = (0..NUM_PERMUTATIONS)
            .map(|_| rng.random())
            .collect::<Vec<_>>();
        let trace = trace_span!("generate_trace")
            .in_scope(|| p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0));
        let device = engine.device();
        let d_trace = device.transport_matrix_to_device(&ColMajorMatrix::from_row_major(&trace));

        let air_ctx = AirProvingContext::simple_no_pis(d_trace);
        let d_pk = device.transport_pk_to_device(&pk);
        let proof = engine.prove(&d_pk, ProvingContext::new(vec![(air_idx, air_ctx)]));

        verify(&vk, &proof, &mut DuplexSponge::default())
            .map_err(|e| eyre!("Proof failed to verify: {e}"))
    })
}
