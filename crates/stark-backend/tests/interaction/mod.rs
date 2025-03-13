use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    interaction::RapPhaseSeq,
    keygen::{types::LinearConstraint, MultiStarkKeygenBuilder},
    p3_field::FieldAlgebra,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, config,
    dummy_airs::interaction::{dummy_interaction_air::DummyInteractionAir, verify_interactions},
};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    fib_selector_air::{air::FibonacciSelectorAir, trace::generate_trace_rows},
    get_conditional_fib_number,
    utils::to_field_vec,
};

type Val = BabyBear;

#[test]
fn test_interaction_trace_height_constraints() {
    let log_trace_degree = 3;
    let n = 1usize << log_trace_degree;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let fib_air = FibonacciSelectorAir::new(sels.clone(), true);
    let mut sender_air = DummyInteractionAir::new(1, true, 0);
    sender_air.count_weight = 3;
    let mut sender_air_2 = DummyInteractionAir::new(1, true, 0);
    sender_air_2.count_weight = 1;
    let mut sender_air_3 = DummyInteractionAir::new(1, true, 1);
    sender_air_3.count_weight = 7;

    let perm = config::baby_bear_poseidon2::random_perm();
    let config = config::baby_bear_poseidon2::default_config(&perm);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    keygen_builder.add_air(Arc::new(fib_air));
    keygen_builder.add_air(Arc::new(sender_air));
    keygen_builder.add_air(Arc::new(sender_air_2));
    keygen_builder.add_air(Arc::new(sender_air_3));
    let pk = keygen_builder.generate_pk();
    let vk = pk.get_vk();

    assert_eq!(vk.trace_height_constraints.len(), 3);

    assert_eq!(
        &vk.trace_height_constraints[0],
        &LinearConstraint {
            coefficients: vec![0, 3, 1, 0], // fib_air does not contribute any count_weight
            threshold: BabyBear::ORDER_U32,
        }
    );
    assert_eq!(
        &vk.trace_height_constraints[1],
        &LinearConstraint {
            coefficients: vec![0, 0, 0, 7],
            threshold: BabyBear::ORDER_U32,
        }
    );
    assert_eq!(
        &vk.trace_height_constraints[2],
        &LinearConstraint {
            coefficients: vec![1, 1, 1, 1], // one interaction per AIR
            threshold: config
                .rap_phase_seq()
                .log_up_security_params()
                .max_interaction_count
        }
    );
}

#[test]
fn test_interaction_fib_selector_happy_path() {
    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let fib_res = get_conditional_fib_number(&sels);
    let pis = vec![a, b, fib_res]
        .into_iter()
        .map(Val::from_canonical_u32)
        .collect_vec();

    let air = FibonacciSelectorAir::new(sels.clone(), true);
    let trace = generate_trace_rows::<Val>(a, b, &sels);

    let mut curr_a = a;
    let mut curr_b = b;
    let mut vals = vec![];
    for sel in sels {
        vals.push(Val::from_bool(sel));
        if sel {
            let c = curr_a + curr_b;
            curr_a = curr_b;
            curr_b = c;
        }
        vals.push(Val::from_canonical_u32(curr_b));
    }
    let sender_trace = RowMajorMatrix::new(vals, 2);
    let sender_air = DummyInteractionAir::new(1, true, 0);
    verify_interactions(
        vec![trace, sender_trace],
        any_rap_arc_vec![air, sender_air],
        vec![pis, vec![]],
    )
    .expect("Verification failed");
}

#[test]
fn test_interaction_stark_multi_rows_happy_path() {
    // Mul  Val
    //   0    1
    //   7    4
    //   3    5
    // 546  889
    let sender_trace =
        RowMajorMatrix::new(to_field_vec::<Val>(vec![0, 1, 3, 5, 7, 4, 546, 889]), 2);
    let sender_air = DummyInteractionAir::new(1, true, 0);

    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   1  889
    //   0  456
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456,
        ]),
        2,
    );
    let receiver_air = DummyInteractionAir::new(1, false, 0);
    verify_interactions(
        vec![sender_trace, receiver_trace],
        any_rap_arc_vec![sender_air, receiver_air],
        vec![vec![], vec![]],
    )
    .expect("Verification failed");
}

#[test]
fn test_interaction_stark_multi_rows_neg() {
    // Mul  Val
    //   0    1
    //   3    5
    //   7    4
    // 546    0
    let sender_trace = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 7, 4, 546, 0]), 2);
    let sender_air = DummyInteractionAir::new(1, true, 0);

    // count of 0 is 545 != 546 in send.
    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545    0
    //   0    0
    //   0  456
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(vec![1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 0, 0, 0, 0, 456]),
        2,
    );
    let receiver_air = DummyInteractionAir::new(1, false, 0);
    let res = verify_interactions(
        vec![sender_trace, receiver_trace],
        any_rap_arc_vec![sender_air, receiver_air],
        vec![vec![], vec![]],
    );
    assert_eq!(res, Err(VerificationError::ChallengePhaseError));
}

#[test]
fn test_interaction_stark_all_0_sender_happy_path() {
    // Mul  Val
    //   0    1
    //   0  646
    //   0    0
    //   0  589
    let sender_trace = RowMajorMatrix::new(to_field_vec(vec![0, 1, 0, 5, 0, 4, 0, 889]), 2);
    let sender_air = DummyInteractionAir::new(1, true, 0);
    verify_interactions(
        vec![sender_trace],
        any_rap_arc_vec![sender_air],
        vec![vec![]],
    )
    .expect("Verification failed");
}

#[test]
fn test_interaction_stark_multi_senders_happy_path() {
    // Mul  Val
    //   0    1
    //   6    4
    //   3    5
    // 333  889
    let sender_trace1 = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 6, 4, 333, 889]), 2);
    // Mul  Val
    //   1    4
    // 213  889
    let sender_trace2 = RowMajorMatrix::new(to_field_vec(vec![1, 4, 213, 889]), 2);

    let sender_air = DummyInteractionAir::new(1, true, 0);

    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   1  889
    //   0  456
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456,
        ]),
        2,
    );
    let receiver_air = DummyInteractionAir::new(1, false, 0);
    verify_interactions(
        vec![sender_trace1, sender_trace2, receiver_trace],
        any_rap_arc_vec![sender_air, sender_air, receiver_air],
        vec![vec![]; 3],
    )
    .expect("Verification failed");
}

#[test]
fn test_interaction_stark_multi_senders_neg() {
    // Mul  Val
    //   0    1
    //   5    4
    //   3    5
    // 333  889
    let sender_trace1 = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 5, 4, 333, 889]), 2);
    // Mul  Val
    //   1    4
    // 213  889
    let sender_trace2 = RowMajorMatrix::new(to_field_vec(vec![1, 4, 213, 889]), 2);

    let sender_air = DummyInteractionAir::new(1, true, 0);

    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   1  889
    //   0  456
    let receiver_trace = RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 1, 889, 0, 456,
        ]),
        2,
    );
    let receiver_air = DummyInteractionAir::new(1, false, 0);
    let res = verify_interactions(
        vec![sender_trace1, sender_trace2, receiver_trace],
        any_rap_arc_vec![sender_air, sender_air, receiver_air],
        vec![vec![]; 3],
    );
    assert_eq!(res, Err(VerificationError::ChallengePhaseError));
}

#[test]
fn test_interaction_stark_multi_sender_receiver_happy_path() {
    // Mul  Val
    //   0    1
    //   6    4
    //   3    5
    // 333  889
    let sender_trace1 = RowMajorMatrix::new(to_field_vec(vec![0, 1, 3, 5, 6, 4, 333, 889]), 2);
    // Mul  Val
    //   1    4
    // 213  889
    let sender_trace2 = RowMajorMatrix::new(to_field_vec(vec![1, 4, 213, 889]), 2);

    let sender_air = DummyInteractionAir::new(1, true, 0);

    // Mul  Val
    //   1    5
    //   3    4
    //   4    4
    //   2    5
    //   0  123
    // 545  889
    //   0  289
    //   0  456
    let receiver_trace1 = RowMajorMatrix::new(
        to_field_vec(vec![
            1, 5, 3, 4, 4, 4, 2, 5, 0, 123, 545, 889, 0, 289, 0, 456,
        ]),
        2,
    );

    // Mul  Val
    //   1  889
    let receiver_trace2 = RowMajorMatrix::new(to_field_vec(vec![1, 889]), 2);
    let receiver_air = DummyInteractionAir::new(1, false, 0);
    verify_interactions(
        vec![
            sender_trace1,
            sender_trace2,
            receiver_trace1,
            receiver_trace2,
        ],
        any_rap_arc_vec![sender_air, sender_air, receiver_air, receiver_air],
        vec![vec![]; 4],
    )
    .expect("Verification failed");
}
