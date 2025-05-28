use itertools::zip_eq;
use openvm_stark_backend::{
    engine::StarkEngine,
    prover::types::{AirProofInput, ProofInput},
};
use openvm_stark_sdk::{
    any_rap_arc_vec,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    dummy_airs::fib_air::{air::FibonacciAir, trace::generate_trace_rows},
    engine::StarkFriEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;
use stark_backend_gpu::engine::GpuBabyBearPoseidon2Engine;

const LOG_BLOWUP: usize = 2;
const LOG_TRACE_DEGREE: usize = 3;

// Public inputs:
const A: u32 = 0;
const B: u32 = 1;
const N: usize = 1usize << LOG_TRACE_DEGREE;

type Val = BabyBear;

fn get_fib_number(n: usize) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n - 1 {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

fn main() {
    setup_tracing();
    println!("test_single_fib_stark");

    let pis = [A, B, get_fib_number(N)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();
    let air = FibonacciAir;

    let trace = generate_trace_rows::<Val>(A, B, N);

    let airs = any_rap_arc_vec![air];
    let traces = vec![trace];
    let public_values = vec![pis];

    let air_proof_inputs = AirProofInput::multiple_simple(traces, public_values);

    let gpu_engine = GpuBabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = gpu_engine.keygen_builder();
    let air_ids = gpu_engine.set_up_keygen_builder(&mut keygen_builder, &airs);
    let pk = keygen_builder.generate_pk();
    // engine.debug(&airs, &pk.per_air, &air_proof_inputs);
    let proof_input = ProofInput {
        per_air: zip_eq(air_ids, air_proof_inputs).collect(),
    };

    // CPU    // CPU
    println!("\nStarting CPU proof");
    let cpu_engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let cpu_proof = cpu_engine.prove(&pk, proof_input.clone());
    cpu_engine.verify(&pk.get_vk(), &cpu_proof).unwrap();

    // GPU
    println!("\nStarting GPU proof");
    let gpu_proof = gpu_engine.prove(&pk, proof_input.clone());
    gpu_engine.verify(&pk.get_vk(), &gpu_proof).unwrap();
}
