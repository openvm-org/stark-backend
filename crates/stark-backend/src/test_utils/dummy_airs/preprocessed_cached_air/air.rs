use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues, PairBuilder};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{air_builders::PartitionedAirBuilder, PartitionedBaseAir};

#[derive(Clone)]
pub struct PreprocessedCachedAir {
    sels: Vec<bool>,
    num_cached_parts: usize,
}

impl PreprocessedCachedAir {
    pub fn new(sels: Vec<bool>, num_cached_parts: usize) -> Self {
        assert!(num_cached_parts > 0, "num_cached_parts must be at least 1");
        Self {
            sels,
            num_cached_parts,
        }
    }
}

impl<F: Field> PartitionedBaseAir<F> for PreprocessedCachedAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![1; self.num_cached_parts]
    }

    fn common_main_width(&self) -> usize {
        1
    }
}

impl<F: Field> BaseAir<F> for PreprocessedCachedAir {
    fn width(&self) -> usize {
        1 + self.num_cached_parts
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(RowMajorMatrix::new_col(
            self.sels.iter().map(|&sel| F::from_bool(sel)).collect(),
        ))
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for PreprocessedCachedAir {}

impl<AB: PartitionedAirBuilder + PairBuilder> Air<AB> for PreprocessedCachedAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let preprocessed_local = preprocessed
            .row_slice(0)
            .expect("preprocessed window should have one row")[0]
            .clone();
        let preprocessed_next = preprocessed
            .row_slice(1)
            .expect("preprocessed window should have two rows")[0]
            .clone();
        let common_local = builder
            .common_main()
            .row_slice(0)
            .expect("common main window should have one row")[0]
            .clone();
        let common_next = builder
            .common_main()
            .row_slice(1)
            .expect("common main window should have two rows")[0]
            .clone();
        let cached_rows: Vec<_> = builder
            .cached_mains()
            .iter()
            .map(|cached_main| {
                let local = cached_main
                    .row_slice(0)
                    .expect("cached main window should have one row")[0]
                    .clone();
                let next = cached_main
                    .row_slice(1)
                    .expect("cached main window should have two rows")[0]
                    .clone();
                (local, next)
            })
            .collect();

        debug_assert_eq!(cached_rows.len(), self.num_cached_parts);

        builder
            .assert_zero(preprocessed_local.clone() * (preprocessed_local.clone() - AB::Expr::ONE));
        builder.when_first_row().assert_zero(common_local.clone());
        builder
            .when_transition()
            .assert_one(common_next.clone() - common_local.clone());

        // Each cached partition increments the previous partition by the selector bit.
        let mut prev_local = common_local;
        let mut prev_next = common_next;
        for (cached_local, cached_next) in cached_rows {
            builder.assert_eq(
                prev_local.clone() + preprocessed_local.clone(),
                cached_local.clone(),
            );
            builder.when_transition().assert_eq(
                prev_next.clone() + preprocessed_next.clone(),
                cached_next.clone(),
            );
            prev_local = cached_local;
            prev_next = cached_next;
        }
    }
}
