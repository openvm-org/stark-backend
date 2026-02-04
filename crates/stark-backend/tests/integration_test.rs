use openvm_stark_backend::{
    prover::{hal::DeviceDataTransporter, types::ProvingContext},
    utils::disable_debug_builder,
    Chip,
};
use openvm_stark_sdk::config::FriParameters;
/// Test utils
use openvm_stark_sdk::{
    any_rap_arc_vec, config,
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    dummy_airs::{
        fib_air::chip::FibonacciChip,
        interaction::dummy_interaction_air::{DummyInteractionChip, DummyInteractionData},
    },
    engine::StarkFriEngine,
    utils,
};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;

mod cached_lookup;
mod fib_triples_air;
pub mod interaction;
mod partitioned_sum_air;

#[test]
fn test_single_fib_stark() {
    use openvm_stark_sdk::dummy_airs::fib_air::{air::FibonacciAir, trace::generate_trace_rows};

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n)].map(BabyBear::from_u32).to_vec();
    let air = FibonacciAir;

    let trace = generate_trace_rows::<Val>(a, b, n);

    BabyBearPoseidon2Engine::run_simple_test_fast(any_rap_arc_vec![air], vec![trace], vec![pis])
        .expect("Verification failed");
}

#[test]
fn test_single_fib_triples_stark() {
    use fib_triples_air::{air::FibonacciAir, trace::generate_trace_rows};

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n + 1)]
        .map(BabyBear::from_u32)
        .to_vec();

    let air = FibonacciAir;

    let trace = generate_trace_rows::<Val>(a, b, n);

    BabyBearPoseidon2Engine::run_simple_test_fast(any_rap_arc_vec![air], vec![trace], vec![pis])
        .expect("Verification failed");
}

#[test]
fn test_single_fib_selector_stark() {
    use openvm_stark_sdk::dummy_airs::fib_selector_air::{
        air::FibonacciSelectorAir, trace::generate_trace_rows,
    };

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let pis = [a, b, get_conditional_fib_number(&sels)]
        .map(BabyBear::from_u32)
        .to_vec();

    let air = FibonacciSelectorAir::new(sels, false);

    let trace = generate_trace_rows::<Val>(a, b, air.sels());

    BabyBearPoseidon2Engine::run_simple_test_fast(any_rap_arc_vec![air], vec![trace], vec![pis])
        .expect("Verification failed");
}

#[test]
fn test_double_fib_starks() {
    use openvm_stark_sdk::dummy_airs::{
        fib_air, fib_air::air::FibonacciAir, fib_selector_air,
        fib_selector_air::air::FibonacciSelectorAir,
    };

    let log_n1 = 3;
    let log_n2 = 5;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n1 = 1usize << log_n1;
    let n2 = 1usize << log_n2;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n2).map(|i| i % 2 == 0).collect(); // Evens
    let pis1 = [a, b, get_fib_number(n1)].map(BabyBear::from_u32).to_vec();
    let pis2 = [a, b, get_conditional_fib_number(&sels)]
        .map(BabyBear::from_u32)
        .to_vec();

    let air1 = FibonacciAir;
    let air2 = FibonacciSelectorAir::new(sels, false);

    let trace1 = fib_air::trace::generate_trace_rows::<Val>(a, b, n1);
    let trace2 = fib_selector_air::trace::generate_trace_rows::<Val>(a, b, air2.sels());

    BabyBearPoseidon2Engine::run_simple_test_fast(
        any_rap_arc_vec![air1, air2],
        vec![trace1, trace2],
        vec![pis1, pis2],
    )
    .expect("Verification failed");
}

#[test]
fn test_optional_air() {
    use openvm_stark_backend::engine::StarkEngine;

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let fib_chip = FibonacciChip::new(0, 1, 8);
    let send_chip1 = DummyInteractionChip::new_without_partition(1, true, 0);
    let send_chip2 = DummyInteractionChip::new_with_partition(engine.device.clone(), 1, true, 0);
    let recv_chip1 = DummyInteractionChip::new_without_partition(1, false, 0);
    let mut keygen_builder = engine.keygen_builder();
    let fib_chip_id = keygen_builder.add_air(fib_chip.air());
    let send_chip1_id = keygen_builder.add_air(send_chip1.air());
    let send_chip2_id = keygen_builder.add_air(send_chip2.air());
    let recv_chip1_id = keygen_builder.add_air(recv_chip1.air());
    let pk = keygen_builder.generate_pk();
    let verifier = engine.verifier();

    // Case 1: All AIRs are present.
    {
        let fib_chip = fib_chip.clone();
        let mut send_chip1 = send_chip1.clone();
        let mut send_chip2 = send_chip2.clone();
        let mut recv_chip1 = recv_chip1.clone();
        send_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        send_chip2.load_data(DummyInteractionData {
            count: vec![1, 2, 8],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        recv_chip1.load_data(DummyInteractionData {
            count: vec![2, 4, 12],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        engine
            .prove_then_verify(
                &pk,
                ProvingContext {
                    per_air: vec![
                        (fib_chip_id, fib_chip.generate_proving_ctx(())),
                        (send_chip1_id, send_chip1.generate_proving_ctx(())),
                        (send_chip2_id, send_chip2.generate_proving_ctx(())),
                        (recv_chip1_id, recv_chip1.generate_proving_ctx(())),
                    ],
                },
            )
            .expect("Verification failed");
    }
    // Case 2: The second AIR is not presented.
    {
        let mut send_chip1 = send_chip1.clone();
        let mut recv_chip1 = recv_chip1.clone();
        send_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        recv_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        engine
            .prove_then_verify(
                &pk,
                ProvingContext {
                    per_air: vec![
                        (send_chip1_id, send_chip1.generate_proving_ctx(())),
                        (recv_chip1_id, recv_chip1.generate_proving_ctx(())),
                    ],
                },
            )
            .expect("Verification failed");
    }
    // Case 3: Negative - unbalanced interactions.
    {
        disable_debug_builder();
        let mut recv_chip1 = recv_chip1.clone();
        recv_chip1.load_data(DummyInteractionData {
            count: vec![1, 2, 4],
            fields: vec![vec![1], vec![2], vec![3]],
        });
        let pk_device = engine.device().transport_pk_to_device(&pk);
        let proof = engine.prove(
            &pk_device,
            ProvingContext {
                per_air: vec![(recv_chip1_id, recv_chip1.generate_proving_ctx(()))],
            },
        );
        let mut challenger = engine.new_challenger();
        assert!(verifier
            .verify(&mut challenger, &pk.get_vk(), &proof)
            .is_err());
    }
}

#[test]
fn test_vkey_methods() {
    use openvm_stark_backend::engine::StarkEngine;
    use openvm_stark_sdk::{
        config::{setup_tracing, FriParameters},
        dummy_airs::fib_air::air::FibonacciAir,
    };
    use p3_air::BaseAir;

    setup_tracing();

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());

    let fib_chip = FibonacciChip::new(0, 1, 8);
    let send_chip = DummyInteractionChip::new_without_partition(1, true, 0);
    let recv_chip = DummyInteractionChip::new_without_partition(1, false, 0);

    let mut keygen_builder = engine.keygen_builder();
    let _ = keygen_builder.add_air(fib_chip.air());
    let _ = keygen_builder.add_air(send_chip.air());
    let _ = keygen_builder.add_air(recv_chip.air());
    let pk = keygen_builder.generate_pk();

    let vk = pk.get_vk();

    let main_widths = vk.main_widths();
    assert_eq!(main_widths.len(), 3);
    assert_eq!(main_widths[0], BaseAir::<BabyBear>::width(&FibonacciAir));
    assert_eq!(main_widths[1], BaseAir::<BabyBear>::width(&send_chip.air));
    assert_eq!(main_widths[2], BaseAir::<BabyBear>::width(&recv_chip.air));

    let total_widths = vk.total_widths();
    assert_eq!(total_widths.len(), 3);
    assert_eq!(total_widths[0], BaseAir::<BabyBear>::width(&FibonacciAir));
    // 1 interaction -> 1 perm col + 1 cumsum col -> 8 cols in base field
    assert_eq!(
        total_widths[1],
        BaseAir::<BabyBear>::width(&send_chip.air) + 8
    );
    assert_eq!(
        total_widths[2],
        BaseAir::<BabyBear>::width(&recv_chip.air) + 8
    );

    let interactions = vk.num_interactions();
    assert_eq!(interactions.len(), 3);
    assert_eq!(interactions[0], 0);
    assert_eq!(interactions[1], 1);
    assert_eq!(interactions[2], 1);
}

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

fn get_conditional_fib_number(sels: &[bool]) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for &s in sels[0..sels.len() - 1].iter() {
        if s {
            let c = a + b;
            a = b;
            b = c;
        }
    }
    b
}
