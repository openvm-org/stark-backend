//! Prove keccakf-air over BabyBear using poseidon2 for FRI hash.

use std::{fs::File, io::Write, path::Path, sync::Arc};

use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    prover::types::{AirProofInput, ProofInput},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    utils::metrics_span,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use pprof::{protos::Message, ProfilerGuardBuilder};
use rand::Rng;

const NUM_PERMUTATIONS: usize = 1 << 10;
const LOG_BLOWUP: usize = 1;

// Newtype to implement extended traits
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

fn main() {
    setup_tracing();

    let mut rng = create_seeded_rng();
    let air = TestAir(KeccakAir {});

    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = engine.keygen_builder();
    let air_id = keygen_builder.add_air(Arc::new(air));
    let pk = keygen_builder.generate_pk();

    // Start pprof profiling
    let guard = ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso", "rayon", "rayon_core"])
        .build()
        .expect("Failed to create profiler guard");

    let inputs = (0..NUM_PERMUTATIONS).map(|_| rng.gen()).collect::<Vec<_>>();
    let trace = metrics_span("generate_trace", || {
        p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0)
    });

    let proof = engine.prove(
        &pk,
        ProofInput::new(vec![(air_id, AirProofInput::simple_no_pis(trace))]),
    );

    // Create a frames post-processor function that filters out rayon calls
    fn frames_post_processor() -> impl Fn(&mut pprof::Frames) {
        |frames| {
            frames.frames.retain(|frame| {
                // Filter out frames where the name contains "rayon"
                !frame.iter().any(|s| s.name().contains("rayon"))
            });
        }
    }

    // Use the frames_post_processor when building the report
    match guard
        .report()
        .frames_post_processor(frames_post_processor())
        .build()
    {
        Ok(report) => {
            // Save pprof file
            let path = Path::new("stark_profile.pb");
            let mut file = File::create(path).expect("Failed to create profile file");
            let profile = report.pprof().expect("Failed to create pprof");
            let mut buf = Vec::new();
            profile.encode(&mut buf).expect("Failed to encode pprof");
            file.write_all(&buf).expect("Failed to write profile");

            // Generate flamegraph
            let flamegraph_path = "stark_flamegraph.svg";
            let mut flamegraph_file =
                File::create(flamegraph_path).expect("Failed to create flamegraph file");
            report
                .flamegraph(&mut flamegraph_file)
                .expect("Failed to create flamegraph");

            println!("Profile saved to stark_profile.pb");
            println!("Flamegraph saved to {flamegraph_path}");
        }
        Err(e) => println!("Failed to create pprof profile: {e}"),
    }

    engine.verify(&pk.get_vk(), &proof).unwrap();
}
