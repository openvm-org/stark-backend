//! Benchmark comparing row-major CPU backend vs reference (col-major) backend
//! on the keccakf-air.
//!
//! Usage:
//!   cargo bench -p openvm-cpu-backend --bench keccakf
//!
//! For a quick comparison without criterion overhead:
//!   cargo run -p openvm-cpu-backend --example keccakf --release

use std::sync::Arc;

use openvm_stark_backend::{
    prover::{AirProvingContext, ColMajorMatrix, DeviceDataTransporter, ProvingContext},
    PartitionedBaseAir, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security,
    baby_bear_poseidon2::{
        BabyBearPoseidon2CpuEngine, BabyBearPoseidon2RefEngine as ReferenceCpuEngine,
    },
};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_field::Field;
use p3_keccak_air::KeccakAir;
use rand::{rngs::StdRng, Rng, SeedableRng};

const NUM_PERMUTATIONS: usize = 1 << 10;

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

fn make_params() -> SystemParams {
    app_params_with_100_bits_security(21)
}

fn generate_inputs() -> Vec<[u64; 25]> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..NUM_PERMUTATIONS).map(|_| rng.random()).collect()
}

fn bench_reference_backend(params: SystemParams, inputs: Vec<[u64; 25]>) -> std::time::Duration {
    let air = TestAir(KeccakAir {});
    let engine: ReferenceCpuEngine = StarkEngine::new(params);
    let (pk, _vk) = engine.keygen(&[Arc::new(air)]);

    let trace =
        p3_keccak_air::generate_trace_rows::<openvm_stark_sdk::p3_baby_bear::BabyBear>(inputs, 0);
    let trace_ctx = AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace));
    let d_pk = engine.device().transport_pk_to_device(&pk);

    let start = std::time::Instant::now();
    let _proof = engine
        .prove(&d_pk, ProvingContext::new(vec![(0, trace_ctx)]))
        .unwrap();
    start.elapsed()
}

fn bench_rowmajor_backend(params: SystemParams, inputs: Vec<[u64; 25]>) -> std::time::Duration {
    let air = TestAir(KeccakAir {});
    let engine: BabyBearPoseidon2CpuEngine = StarkEngine::new(params);
    let (pk, _vk) = engine.keygen(&[Arc::new(air)]);

    let trace =
        p3_keccak_air::generate_trace_rows::<openvm_stark_sdk::p3_baby_bear::BabyBear>(inputs, 0);
    let trace_ctx = AirProvingContext::simple_no_pis(trace);
    let d_pk = engine.device().transport_pk_to_device(&pk);

    let start = std::time::Instant::now();
    let _proof = engine
        .prove(&d_pk, ProvingContext::new(vec![(0, trace_ctx)]))
        .unwrap();
    start.elapsed()
}

fn main() {
    openvm_stark_sdk::utils::setup_tracing();
    let params = make_params();
    let inputs = generate_inputs();

    println!("=== Keccakf Benchmark ({NUM_PERMUTATIONS} permutations) ===\n");

    // Warmup + benchmark reference (col-major) backend
    println!("Reference (col-major) backend...");
    let ref_duration = bench_reference_backend(params.clone(), inputs.clone());
    println!("  prove time: {ref_duration:?}");

    // Warmup + benchmark row-major backend
    println!("\nRow-major CPU backend...");
    let rm_duration = bench_rowmajor_backend(params, inputs);
    println!("  prove time: {rm_duration:?}");

    let speedup = ref_duration.as_secs_f64() / rm_duration.as_secs_f64();
    println!("\nSpeedup: {speedup:.2}x");
}
