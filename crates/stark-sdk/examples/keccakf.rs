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
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
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

    let inputs = (0..NUM_PERMUTATIONS).map(|_| rng.gen()).collect::<Vec<_>>();
    let trace = metrics_span("generate_trace", || {
        p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0)
    });

    let proof = engine.prove(
        &pk,
        ProofInput::new(vec![(air_id, AirProofInput::simple_no_pis(trace))]),
    );

    engine.verify(&pk.get_vk(), &proof).unwrap();
}
