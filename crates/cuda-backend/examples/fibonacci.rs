use std::sync::Arc;

use itertools::zip_eq;
use openvm_cuda_backend::{
    engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend, types::SC,
};
use openvm_stark_backend::{
    engine::StarkEngine,
    prover::{
        cpu::CpuBackend,
        hal::DeviceDataTransporter,
        types::{AirProvingContext, ProvingContext},
    },
};
use openvm_stark_sdk::{
    any_rap_arc_vec,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    dummy_airs::fib_air::{air::FibonacciAir, trace::generate_trace_rows},
    engine::StarkFriEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;

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

    let public_values = [A, B, get_fib_number(N)].map(BabyBear::from_u32).to_vec();
    let air = FibonacciAir;

    let cpu_trace = Arc::new(generate_trace_rows::<Val>(A, B, N));

    let airs = any_rap_arc_vec![air];

    let gpu_engine = GpuBabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let gpu_trace = gpu_engine.device().transport_matrix_to_device(&cpu_trace);

    let cpu_air_ctx = AirProvingContext::<CpuBackend<SC>>::simple(cpu_trace, public_values.clone());
    let gpu_air_ctx = AirProvingContext::<GpuBackend>::simple(gpu_trace, public_values);

    let mut keygen_builder = gpu_engine.keygen_builder();
    let air_ids = gpu_engine.set_up_keygen_builder(&mut keygen_builder, &airs);
    let pk_host = keygen_builder.generate_pk();
    let vk = pk_host.get_vk();
    let pk = gpu_engine.device().transport_pk_to_device(&pk_host);
    // engine.debug(&airs, &pk.per_air, &air_proof_inputs);
    let cpu_ctx = ProvingContext::new(zip_eq(air_ids.clone(), vec![cpu_air_ctx]).collect());
    let gpu_ctx = ProvingContext::new(zip_eq(air_ids, vec![gpu_air_ctx]).collect());

    // CPU    // CPU
    println!("\nStarting CPU proof");
    let cpu_engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let cpu_pk = cpu_engine.device().transport_pk_to_device(&pk_host);
    let cpu_proof = cpu_engine.prove(&cpu_pk, cpu_ctx);
    cpu_engine.verify(&vk, &cpu_proof).unwrap();

    // GPU
    println!("\nStarting GPU proof");
    let gpu_proof = gpu_engine.prove(&pk, gpu_ctx);
    gpu_engine.verify(&vk, &gpu_proof).unwrap();
}
