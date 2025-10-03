use std::sync::Arc;

use openvm_cuda_backend::engine::GpuBabyBearPoseidon2Engine;
use openvm_cuda_common::stream::current_stream_id;
use openvm_stark_backend::{
    engine::StarkEngine,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    prover::{
        hal::DeviceDataTransporter,
        types::{AirProvingContext, ProvingContext},
    },
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing::info_span;

struct TestAir(KeccakAir);

impl<F: Field> BaseAir<F> for TestAir {
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

const LOG_BLOWUP: usize = 2;
const NUM_PERMUTATIONS: usize = 1 << 10;

fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("Thread panicked: {}", info);
        std::process::abort();
    }));
    setup_tracing();

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

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        // ----- CPU keygen once, shared by all threads -----
        let air = TestAir(KeccakAir {});
        let engine_cpu = BabyBearPoseidon2Engine::new(
            FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
        );

        let mut keygen_builder = engine_cpu.keygen_builder();
        let air_id = keygen_builder.add_air(Arc::new(air));
        let pk_host = Arc::new(keygen_builder.generate_pk());
        let vk = Arc::new(pk_host.get_vk());

        // Base seed to derive per-thread RNGs deterministically but independently.
        let mut master_rng = create_seeded_rng();
        let base_seed: u64 = master_rng.gen();

        let mut handles = Vec::new();

        for t in 0..num_tasks {
            let pk_host = pk_host.clone();
            let vk = vk.clone();

            let handle = tokio::task::spawn(async move {
                let task_seed = base_seed ^ ((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut rng = StdRng::seed_from_u64(task_seed);

                // Per-thread random inputs + CPU trace
                let inputs = (0..NUM_PERMUTATIONS).map(|_| rng.gen()).collect::<Vec<_>>();
                let trace = info_span!("generate_trace", task=%t)
                    .in_scope(|| p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0));
                let cpu_trace = Arc::new(trace);

                // GPU: build a per-thread engine (uses per-thread default stream)
                println!("[task {t}] Starting GPU proof");
                let engine_gpu = GpuBabyBearPoseidon2Engine::new(
                    FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
                );

                // Move data to device and prove/verify
                let pk_dev = engine_gpu.device().transport_pk_to_device(&pk_host);
                let gpu_trace = engine_gpu.device().transport_matrix_to_device(&cpu_trace);
                let gpu_ctx = ProvingContext::new(vec![(
                    air_id,
                    AirProvingContext::simple_no_pis(gpu_trace),
                )]);

                let proof = engine_gpu.prove(&pk_dev, gpu_ctx);
                engine_gpu.verify(&vk, &proof).expect("verification failed");
                println!(
                    "[task {t} - stream {}] Proof verified ✅",
                    current_stream_id().unwrap()
                );
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("task failed");
        }
        println!("\nAll {num_tasks} tasks completed on {num_threads} threads.");
    });
}
