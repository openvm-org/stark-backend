//! Prove keccakf-air over BabyBear using poseidon2 for FRI hash.
//! Multi-threaded: runs multiple proofs across OS threads / CUDA streams.

use std::sync::Arc;

use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
use openvm_cuda_common::stream::current_stream_id;
use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    prover::{AirProvingContext, ColMajorMatrix, DeviceDataTransporter, ProvingContext},
    BaseAirWithPublicValues, PartitionedBaseAir, StarkEngine, SystemParams, WhirConfig, WhirParams,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge},
        log_up_params::log_up_security_params_baby_bear_100_bits,
    },
    utils::setup_tracing,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use rand::{rngs::StdRng, Rng, SeedableRng};
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

fn make_params() -> SystemParams {
    let l_skip = 4;
    let n_stack = 17;
    let w_stack = 64;
    let k_whir = 4;
    let whir_params = WhirParams {
        k: k_whir,
        log_final_poly_len: 2 * k_whir,
        query_phase_pow_bits: 20,
    };
    let log_blowup = 1;
    let whir = WhirConfig::new(log_blowup, l_skip + n_stack, whir_params, 100);
    SystemParams {
        l_skip,
        n_stack,
        w_stack,
        log_blowup,
        whir,
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree: 3,
    }
}

fn main() {
    setup_tracing();
    std::panic::set_hook(Box::new(|info| {
        eprintln!("Thread panicked: {}", info);
        std::process::abort();
    }));

    // NUM_THREADS: number of OS threads = CUDA streams to use
    let num_threads: usize = std::env::var("NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    // NUM_TASKS: number of proofs to run
    let num_tasks: usize = std::env::var("NUM_TASKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_threads * 4);

    // ----- CPU keygen once, shared by all threads -----
    let air = TestAir(KeccakAir {});
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(make_params());
    let (pk, vk) = engine.keygen(&[Arc::new(air)]);
    let pk = Arc::new(pk);
    let vk = Arc::new(vk);
    let air_idx = 0;

    // Base seed to derive per-task RNGs deterministically but independently.
    let mut master_rng = StdRng::seed_from_u64(42);
    let base_seed: u64 = master_rng.random();

    let tasks_per_thread = num_tasks.div_ceil(num_threads);
    let mut handles = Vec::new();

    for worker_idx in 0..num_threads {
        let pk = pk.clone();
        let vk = vk.clone();
        let start_task = worker_idx * tasks_per_thread;
        let end_task = std::cmp::min(start_task + tasks_per_thread, num_tasks);

        handles.push(std::thread::spawn(move || {
            for t in start_task..end_task {
                let task_seed = base_seed ^ ((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut rng = StdRng::seed_from_u64(task_seed);

                let inputs = (0..NUM_PERMUTATIONS)
                    .map(|_| rng.random())
                    .collect::<Vec<_>>();
                let trace = info_span!("generate_trace", task=%t)
                    .in_scope(|| p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0));

                println!("[task {t}] Starting GPU proof");
                let mut engine = BabyBearPoseidon2GpuEngine::new(make_params());
                // KeccakfAir has no interactions, so save memory heuristic is bad for zerocheck
                engine
                    .device_mut()
                    .prover_config_mut()
                    .zerocheck_save_memory = false;
                let device = engine.device();

                let d_pk = device.transport_pk_to_device(&pk);
                let d_trace =
                    device.transport_matrix_to_device(&ColMajorMatrix::from_row_major(&trace));
                let ctx =
                    ProvingContext::new(vec![(air_idx, AirProvingContext::simple_no_pis(d_trace))]);

                let proof = engine.prove(&d_pk, ctx).expect("proving failed");
                engine.verify(&vk, &proof).expect("verification failed");
                println!(
                    "[task {t} - stream {}] Proof verified",
                    current_stream_id().unwrap()
                );
            }
        }));
    }

    for handle in handles {
        handle.join().expect("worker thread panicked");
    }

    println!("\nAll {num_tasks} tasks completed on {num_threads} threads.");
}
