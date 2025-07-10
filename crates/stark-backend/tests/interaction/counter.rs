use openvm_stark_backend::interaction::{
    counter::{DummyExpr, InteractionCounterBuilder},
    InteractionBuilder, LookupBus, PermutationCheckBus,
};
use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;

type F = BabyBear;

#[test]
fn test_counter_basic_functionality() {
    let mut counter = InteractionCounterBuilder::<F>::new();

    assert_eq!(counter.interaction_count(), 0);
    assert_eq!(counter.num_interactions(), 0);

    counter.push_interaction(0, vec![DummyExpr::new()], DummyExpr::new(), 1);
    assert_eq!(counter.interaction_count(), 1);

    counter.push_interaction(
        1,
        vec![DummyExpr::new(), DummyExpr::new()],
        DummyExpr::new(),
        2,
    );
    assert_eq!(counter.interaction_count(), 2);

    counter.push_interaction(2, Vec::<DummyExpr<F>>::new(), DummyExpr::new(), 0);
    assert_eq!(counter.interaction_count(), 3);
}

#[test]
fn test_counter_with_buses() {
    let mut counter = InteractionCounterBuilder::<F>::new();

    let lookup_bus = LookupBus::new(0);
    let perm_bus = PermutationCheckBus::new(1);

    lookup_bus.lookup_key(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
    assert_eq!(counter.interaction_count(), 1);

    lookup_bus.add_key_with_lookups(
        &mut counter,
        vec![DummyExpr::new(), DummyExpr::new()],
        DummyExpr::new(),
    );
    assert_eq!(counter.interaction_count(), 2);

    perm_bus.send(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
    assert_eq!(counter.interaction_count(), 3);

    perm_bus.receive(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
    assert_eq!(counter.interaction_count(), 4);
}

#[test]
fn test_counter_with_multiple_buses() {
    let mut counter = InteractionCounterBuilder::<F>::new();

    let bus1 = LookupBus::new(0);
    let bus2 = LookupBus::new(1);
    let bus3 = PermutationCheckBus::new(2);

    bus1.lookup_key(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
    bus2.lookup_key(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
    bus3.send(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
    bus1.add_key_with_lookups(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
    bus3.receive(&mut counter, vec![DummyExpr::new()], DummyExpr::new());

    assert_eq!(counter.interaction_count(), 5);
}

#[test]
fn test_counter_air_builder_noops() {
    use p3_air::AirBuilder;

    let mut counter = InteractionCounterBuilder::<F>::new();

    counter.assert_zero(DummyExpr::new());
    counter.assert_one(DummyExpr::new());
    counter.assert_eq(DummyExpr::new(), DummyExpr::new());

    assert_eq!(counter.interaction_count(), 0);

    counter
        .when(counter.is_first_row())
        .assert_zero(DummyExpr::new());
    counter.when_first_row().assert_one(DummyExpr::new());
    counter
        .when_last_row()
        .assert_eq(DummyExpr::new(), DummyExpr::new());
    counter.when_transition().assert_zero(DummyExpr::new());

    assert_eq!(counter.interaction_count(), 0);

    counter.push_interaction(0, vec![DummyExpr::new()], DummyExpr::new(), 1);
    assert_eq!(counter.interaction_count(), 1);
}

#[test]
#[should_panic(expected = "InteractionCounterBuilder only counts interactions, not store them")]
fn test_all_interactions_panics() {
    let counter = InteractionCounterBuilder::<F>::new();
    let _ = counter.all_interactions();
}

#[test]
fn test_counter_large_scale() {
    let mut counter = InteractionCounterBuilder::<F>::new();

    for i in 0..1000 {
        counter.push_interaction(i % 10, vec![DummyExpr::new()], DummyExpr::new(), 1);
    }

    assert_eq!(counter.interaction_count(), 1000);
}

#[test]
fn test_counter_mixed_field_types() {
    let mut counter_baby_bear = InteractionCounterBuilder::<BabyBear>::new();

    counter_baby_bear.push_interaction(0, vec![DummyExpr::new()], DummyExpr::new(), 1);
    assert_eq!(counter_baby_bear.interaction_count(), 1);
}

#[test]
fn test_dummy_expr_operations() {
    let expr1 = DummyExpr::<F>::new();
    let expr2 = DummyExpr::<F>::new();
    let field_val = F::from_canonical_u32(42);

    let _result = expr1 + expr2;
    let _result = expr1 - expr2;
    let _result = expr1 * expr2;
    let _result = -expr1;

    let _result = expr1 + field_val;
    let _result = expr1 - field_val;
    let _result = expr1 * field_val;

    let mut expr = DummyExpr::<F>::new();
    expr += expr2;
    expr -= expr2;
    expr *= expr2;
    expr += field_val;
    expr -= field_val;
    expr *= field_val;

    let exprs = [expr1, expr2];
    let _sum: DummyExpr<F> = exprs.iter().cloned().sum();
    let _product: DummyExpr<F> = exprs.iter().cloned().product();
}

#[test]
fn test_counter_simulating_air_evaluation() {
    let mut counter = InteractionCounterBuilder::<F>::new();

    let trace_height = 8;
    let bus = LookupBus::new(0);

    for _row in 0..trace_height {
        bus.add_key_with_lookups(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
    }

    assert_eq!(counter.interaction_count(), trace_height);
    assert_eq!(counter.num_interactions(), trace_height);
}

#[test]
fn test_counter_with_multiple_bus_types() {
    let mut counter = InteractionCounterBuilder::<F>::new();

    let lookup_bus = LookupBus::new(0);
    let perm_bus = PermutationCheckBus::new(1);

    for i in 0..5 {
        if i % 2 == 0 {
            lookup_bus.lookup_key(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
        } else {
            perm_bus.send(&mut counter, vec![DummyExpr::new()], DummyExpr::new());
        }
    }

    assert_eq!(counter.interaction_count(), 5);
}
