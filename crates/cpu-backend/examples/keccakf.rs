//! Prove keccakf-air over BabyBear using the row-major CPU backend.
//!
//! Usage:
//!   cargo run -p openvm-cpu-backend --example keccakf --release

use std::sync::Arc;

use eyre::eyre;
use openvm_stark_backend::{
    prover::{AirProvingContext, DeviceDataTransporter, ProvingContext},
    PartitionedBaseAir, StarkEngine,
};
use openvm_stark_sdk::config::{
    app_params_with_128_bits_security, baby_bear_poseidon2::BabyBearPoseidon2CpuEngine,
};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_field::Field;
use p3_keccak_air::KeccakAir;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing::info_span;

const NUM_PERMUTATIONS: usize = 1 << 10;

// Newtype to implement extended traits required by SWIRL.
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
    // Initialize tracing (reads RUST_LOG env var).
    openvm_stark_sdk::utils::setup_tracing();

    let params = app_params_with_128_bits_security(21);

    let mut rng = StdRng::seed_from_u64(42);
    let air = TestAir(KeccakAir {});

    println!("Row-major CPU backend: BabyBearPoseidon2CpuEngine");
    println!("NUM_PERMUTATIONS = {NUM_PERMUTATIONS}");

    let engine: BabyBearPoseidon2CpuEngine = StarkEngine::new(params);
    let (pk, vk) = engine.keygen(&[Arc::new(air)]);

    let inputs: Vec<[u64; 25]> = (0..NUM_PERMUTATIONS).map(|_| rng.random()).collect();
    let trace = info_span!("generate_trace").in_scope(|| {
        p3_keccak_air::generate_trace_rows::<openvm_stark_sdk::p3_baby_bear::BabyBear>(inputs, 0)
    });

    // Row-major backend: wrap the RowMajorMatrix directly (no col-major conversion needed).
    let trace_ctx = AirProvingContext::simple_no_pis(trace);
    let d_pk = engine.device().transport_pk_to_device(&pk);
    let proof = engine
        .prove(&d_pk, ProvingContext::new(vec![(0, trace_ctx)]))
        .map_err(|e| eyre!("Proving failed: {e}"))?;

    engine
        .verify(&vk, &proof)
        .map_err(|e| eyre!("Proof failed to verify: {e}"))
}
