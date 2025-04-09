//! Prove keccakf-air over BabyBear using poseidon2 for FRI hash.

use std::sync::Arc;

use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    prover::types::{AirProofInput, ProofInput},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    utils::metrics_span,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, setup_tracing_with_log_level,
        FriParameters,
    },
    engine::StarkFriEngine,
    openvm_stark_backend::engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use rand::Rng;
use tracing_subscriber::EnvFilter;

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
    // setup_tracing();
    // setup_tracing_with_log_level(Level::DEBUG);
    // tracing_subscriber::fmt()
    //     .with_env_filter(EnvFilter::from_default_env())
    //     .init();

    let filter = EnvFilter::new("debug");

    // Set up a simple console subscriber with that filter
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true) // Show target (module path)
        .with_file(true) // Show file names in logs
        .with_line_number(true) // Show line numbers
        .init();

    tracing::debug!("Debug tracing initialized");

    tracing::info!("Starting proof generation");
    let mut rng = create_seeded_rng();
    let air = TestAir(KeccakAir {});

    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = engine.keygen_builder();
    let air_id = keygen_builder.add_air(Arc::new(air));
    let pk = keygen_builder.generate_pk();

    let inputs = (0..NUM_PERMUTATIONS).map(|_| rng.gen()).collect::<Vec<_>>();
    let trace = metrics_span("generate_trace", || {
        p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0)
    });

    let proof = engine.prove(
        &pk,
        ProofInput::new(vec![(air_id, AirProofInput::simple_no_pis(trace))]),
    );

    let proof_bytes = bitcode::serialize(&proof.opening.proof).unwrap();
    tracing::info!("Size of proof is {:?} bytes", proof_bytes.len());

    engine.verify(&pk.get_vk(), &proof).unwrap();
}
