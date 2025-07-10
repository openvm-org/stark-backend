//! Example of AIRs with interactions

use std::sync::Arc;

use openvm_stark_backend::{
    p3_matrix::dense::RowMajorMatrix,
    prover::types::{AirProvingContext, ProvingContext},
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    engine::StarkFriEngine,
    openvm_stark_backend::engine::StarkEngine,
    utils::to_field_vec,
};
use p3_baby_bear::BabyBear;

const LOG_BLOWUP: usize = 1;

type Val = BabyBear;

fn main() {
    setup_tracing();

    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = engine.keygen_builder();

    let sender_air = DummyInteractionAir::new(1, true, 0);
    let receiver_air = DummyInteractionAir::new(1, false, 0);
    let [sender_id, receiver_id] =
        [sender_air, receiver_air].map(|air| keygen_builder.add_air(Arc::new(air)));
    let pk = keygen_builder.generate_pk();

    // Mul  Val
    //   0    1
    //   7    4
    //   3    5
    // 546  889
    let sender_trace = Arc::new(RowMajorMatrix::new(
        to_field_vec::<Val>(vec![0, 1, 3, 5, 7, 4, 546, 889]),
        2,
    ));
    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   1  889
    //   0  456
    let receiver_trace = Arc::new(RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456,
        ]),
        2,
    ));

    engine
        .prove_then_verify(
            &pk,
            ProvingContext::new(vec![
                (sender_id, AirProvingContext::simple_no_pis(sender_trace)),
                (
                    receiver_id,
                    AirProvingContext::simple_no_pis(receiver_trace),
                ),
            ]),
        )
        .unwrap();
}
