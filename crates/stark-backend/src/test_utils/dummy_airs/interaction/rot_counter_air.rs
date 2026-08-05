//! A padding-tolerant AIR with next-row references and interactions, mirroring the structure of a
//! typical VM chip: every constraint and interaction is gated on an `is_valid` selector so all-zero
//! padding rows (physical or virtual) satisfy the AIR and send nothing.
//!
//! Used to test proving with non-power-of-two trace heights (chunked stacking), in particular the
//! rotation opening claims across chunk boundaries.

use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::{
    interaction::{BusIndex, InteractionBuilder},
    PartitionedBaseAir,
};

/// Columns: `[is_valid, x]`. Valid rows are `(1, i)` for `i = 0..num_valid_rows`.
///
/// Constraints:
/// - `is_valid` is boolean;
/// - on transition rows, a valid next row must increment `x` (`next.is_valid * (next.x - x - 1) =
///   0`);
/// - on transition rows, a valid row cannot follow an invalid one (`(1 - is_valid) * next.is_valid
///   = 0`).
///
/// Sends `(x)` with count `is_valid` on `bus_index`.
#[derive(Debug, Clone, Copy)]
pub struct RotCounterAir {
    pub bus_index: BusIndex,
}

impl<F> BaseAir<F> for RotCounterAir {
    fn width(&self) -> usize {
        2
    }
}
impl<F> BaseAirWithPublicValues<F> for RotCounterAir {}
impl<F> PartitionedBaseAir<F> for RotCounterAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for RotCounterAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let (is_valid, x) = (local[0], local[1]);
        let (next_is_valid, next_x) = (next[0], next[1]);

        builder.assert_bool(is_valid);
        builder
            .when_transition()
            .assert_zero(next_is_valid * (next_x - x - AB::Expr::ONE));
        builder
            .when_transition()
            .assert_zero((AB::Expr::ONE - is_valid) * next_is_valid);

        builder.push_interaction(self.bus_index, [x.into()], is_valid.into(), 1);
    }
}
