use std::borrow::Borrow;

use openvm_stark_backend::rap::{BaseAirWithPublicValues, PartitionedBaseAir};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_matrix::Matrix;

use super::columns::{FibonacciCols, NUM_FIBONACCI_COLS};

pub struct FibonacciAir;

impl<F> PartitionedBaseAir<F> for FibonacciAir {}
impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<F> BaseAirWithPublicValues<F> for FibonacciAir {
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &FibonacciCols<AB::Var> = (*local).borrow();
        let next: &FibonacciCols<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left.clone(), a);
        when_first_row.assert_eq(local.middle.clone(), b);
        when_first_row.assert_eq(
            local.right.clone(),
            local.left.clone() + local.middle.clone(),
        );

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.middle.clone(), next.left.clone());

        // b' <- c
        when_transition.assert_eq(local.right.clone(), next.middle.clone());

        // c' <- b + c
        when_transition.assert_eq(
            local.middle.clone() + local.right.clone(),
            next.right.clone(),
        );

        builder.when_last_row().assert_eq(local.right.clone(), x);
    }
}
