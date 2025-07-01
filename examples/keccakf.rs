use std::sync::Arc;

use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    prover::types::{AirProofInput, ProofInput},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use rand::Rng;
use stark_backend_gpu::engine::GpuBabyBearPoseidon2Engine;
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
    setup_tracing();
    let mut rng = create_seeded_rng();
    let air = TestAir(KeccakAir {});

    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = engine.keygen_builder();
    let air_id = keygen_builder.add_air(Arc::new(air));
    let pk = keygen_builder.generate_pk();

    let inputs = (0..NUM_PERMUTATIONS).map(|_| rng.gen()).collect::<Vec<_>>();
    let trace = info_span!("generate_trace")
        .in_scope(|| p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0));
    let cpu_proof_input = ProofInput::new(vec![(air_id, AirProofInput::simple_no_pis(trace))]);
    let gpu_proof_input = cpu_proof_input.clone();

    // CPU
    println!("\nStarting CPU proof");
    let cpu_proof = engine.prove(&pk, cpu_proof_input);
    engine.verify(&pk.get_vk(), &cpu_proof).unwrap();

    // GPU
    println!("\nStarting GPU proof");
    let engine = GpuBabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let gpu_proof = engine.prove(&pk, gpu_proof_input);
    engine.verify(&pk.get_vk(), &gpu_proof).unwrap();
}
