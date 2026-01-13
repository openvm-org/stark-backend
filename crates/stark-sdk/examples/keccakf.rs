//! Prove keccakf-air over BabyBear using poseidon2 for FRI hash.

use std::sync::Arc;

use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    prover::types::{AirProvingContext, ProvingContext},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use rand::Rng;
use tracing::info_span;

const NUM_PERMUTATIONS: usize = 1 << 10;
const LOG_BLOWUP: usize = 1;

// Newtype to implement extended traits
struct TestAir(KeccakAir);

impl<F> BaseAir<F> for TestAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.0)
    }
}
impl<F> BaseAirWithPublicValues<F> for TestAir {}
impl<F> PartitionedBaseAir<F> for TestAir {}

impl<AB: AirBuilder> Air<AB> for TestAir {
    fn eval(&self, builder: &mut AB) {
        self.0.eval(builder);
    }
}

fn main() {
    run_with_metric_collection("OUTPUT_PATH", || {
        let mut rng = create_seeded_rng();
        let air = TestAir(KeccakAir {});

        let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_with_100_bits_security(
            LOG_BLOWUP,
        ));
        let mut keygen_builder = engine.keygen_builder();
        let air_id = keygen_builder.add_air(Arc::new(air));
        let pk = keygen_builder.generate_pk();

        let inputs = (0..NUM_PERMUTATIONS)
            .map(|_| rng.random())
            .collect::<Vec<_>>();
        let trace = info_span!("generate_trace")
            .in_scope(|| p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0));

        engine
            .prove_then_verify(
                &pk,
                ProvingContext::new(vec![(
                    air_id,
                    AirProvingContext::simple_no_pis(Arc::new(trace)),
                )]),
            )
            .unwrap();
    });
}
