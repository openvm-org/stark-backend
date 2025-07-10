use p3_air::{AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::{dense::DenseMatrix, Matrix};

use crate::interaction::{Interaction, InteractionBuilder};

/// A minimal AIR builder that only counts interactions.
/// All constraint operations are no-ops.
#[derive(Debug, Default)]
pub struct InteractionCounterBuilder<F: Field> {
    interaction_count: usize,
    _phantom: std::marker::PhantomData<F>,
}

/// Dummy expression type that implements required traits but performs no operations
#[derive(Clone, Copy, Debug, Default)]
pub struct DummyExpr<F: Field> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field> DummyExpr<F> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Field> From<F> for DummyExpr<F> {
    #[inline(always)]
    fn from(_: F) -> Self {
        Self::new()
    }
}

impl<F: Field> FieldAlgebra for DummyExpr<F> {
    type F = F;

    const ZERO: Self = Self {
        _phantom: std::marker::PhantomData,
    };
    const ONE: Self = Self {
        _phantom: std::marker::PhantomData,
    };
    const TWO: Self = Self {
        _phantom: std::marker::PhantomData,
    };
    const NEG_ONE: Self = Self {
        _phantom: std::marker::PhantomData,
    };

    #[inline(always)]
    fn from_f(_f: Self::F) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn from_bool(_b: bool) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn from_canonical_u8(_n: u8) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn from_canonical_u16(_n: u16) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn from_canonical_u32(_n: u32) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn from_canonical_u64(_n: u64) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn from_canonical_usize(_n: usize) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn from_wrapped_u32(_n: u32) -> Self {
        Self::new()
    }

    #[inline(always)]
    fn from_wrapped_u64(_n: u64) -> Self {
        Self::new()
    }
}

// Arithmetic operations
impl<F: Field> std::ops::Add for DummyExpr<F> {
    type Output = Self;
    #[inline(always)]
    fn add(self, _: Self) -> Self {
        Self::new()
    }
}

impl<F: Field> std::ops::Add<F> for DummyExpr<F> {
    type Output = Self;
    #[inline(always)]
    fn add(self, _: F) -> Self {
        Self::new()
    }
}

impl<F: Field> std::ops::Sub for DummyExpr<F> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, _: Self) -> Self {
        Self::new()
    }
}

impl<F: Field> std::ops::Sub<F> for DummyExpr<F> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, _: F) -> Self {
        Self::new()
    }
}

impl<F: Field> std::ops::Mul for DummyExpr<F> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, _: Self) -> Self {
        Self::new()
    }
}

impl<F: Field> std::ops::Mul<F> for DummyExpr<F> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, _: F) -> Self {
        Self::new()
    }
}

impl<F: Field> std::ops::Neg for DummyExpr<F> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new()
    }
}

impl<F: Field> std::ops::AddAssign for DummyExpr<F> {
    #[inline(always)]
    fn add_assign(&mut self, _: Self) {}
}

impl<F: Field> std::ops::AddAssign<F> for DummyExpr<F> {
    #[inline(always)]
    fn add_assign(&mut self, _: F) {}
}

impl<F: Field> std::ops::SubAssign for DummyExpr<F> {
    #[inline(always)]
    fn sub_assign(&mut self, _: Self) {}
}

impl<F: Field> std::ops::SubAssign<F> for DummyExpr<F> {
    #[inline(always)]
    fn sub_assign(&mut self, _: F) {}
}

impl<F: Field> std::ops::MulAssign for DummyExpr<F> {
    #[inline(always)]
    fn mul_assign(&mut self, _: Self) {}
}

impl<F: Field> std::ops::MulAssign<F> for DummyExpr<F> {
    #[inline(always)]
    fn mul_assign(&mut self, _: F) {}
}

impl<F: Field> std::iter::Sum for DummyExpr<F> {
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
        Self::new()
    }
}

impl<F: Field> std::iter::Product for DummyExpr<F> {
    #[inline(always)]
    fn product<I: Iterator<Item = Self>>(_iter: I) -> Self {
        Self::new()
    }
}

/// Dummy row iterator
pub struct DummyRowIter<F: Field> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field> Iterator for DummyRowIter<F> {
    type Item = DummyExpr<F>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

/// Dummy matrix type for the builder
pub struct DummyMatrix<F: Field> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field> Matrix<DummyExpr<F>> for DummyMatrix<F> {
    #[inline(always)]
    fn width(&self) -> usize {
        0
    }

    #[inline(always)]
    fn height(&self) -> usize {
        0
    }

    type Row<'a>
        = DummyRowIter<F>
    where
        Self: 'a;

    #[inline(always)]
    fn row(&self, _r: usize) -> Self::Row<'_> {
        DummyRowIter {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Field> InteractionCounterBuilder<F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            interaction_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn interaction_count(&self) -> usize {
        self.interaction_count
    }
}

impl<F: Field> BaseAir<F> for InteractionCounterBuilder<F> {
    #[inline(always)]
    fn width(&self) -> usize {
        0
    }

    #[inline(always)]
    fn preprocessed_trace(&self) -> Option<DenseMatrix<F>> {
        None
    }
}

impl<F: Field> AirBuilder for InteractionCounterBuilder<F> {
    type F = F;
    type Expr = DummyExpr<F>;
    type Var = DummyExpr<F>;
    type M = DummyMatrix<F>;

    #[inline(always)]
    fn main(&self) -> Self::M {
        DummyMatrix {
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn is_first_row(&self) -> Self::Expr {
        DummyExpr::new()
    }

    #[inline(always)]
    fn is_last_row(&self) -> Self::Expr {
        DummyExpr::new()
    }

    #[inline(always)]
    fn is_transition_window(&self, _size: usize) -> Self::Expr {
        DummyExpr::new()
    }

    #[inline(always)]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {}

    #[inline(always)]
    fn assert_one<I: Into<Self::Expr>>(&mut self, _x: I) {}

    #[inline(always)]
    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, _x: I1, _y: I2) {}
}

impl<F: Field> AirBuilderWithPublicValues for InteractionCounterBuilder<F> {
    type PublicVar = DummyExpr<F>;

    #[inline(always)]
    fn public_values(&self) -> &[Self::PublicVar] {
        &[]
    }
}

impl<F: Field> InteractionBuilder for InteractionCounterBuilder<F> {
    #[inline(always)]
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_index: u16,
        _fields: impl IntoIterator<Item = E>,
        _count: impl Into<Self::Expr>,
        _count_weight: u32,
    ) {
        self.interaction_count += 1;
    }

    #[inline(always)]
    fn num_interactions(&self) -> usize {
        self.interaction_count
    }

    fn all_interactions(&self) -> &[Interaction<Self::Expr>] {
        unimplemented!("InteractionCounterBuilder only counts interactions, not store them")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::{LookupBus, PermutationCheckBus};

    #[test]
    fn test_interaction_counter() {
        let mut builder = InteractionCounterBuilder::<p3_baby_bear::BabyBear>::new();

        // Test that initial count is 0
        assert_eq!(builder.num_interactions(), 0);

        // Push some interactions directly
        builder.push_interaction(0, vec![DummyExpr::new()], DummyExpr::new(), 1);
        assert_eq!(builder.num_interactions(), 1);

        builder.push_interaction(
            1,
            vec![DummyExpr::new(), DummyExpr::new()],
            DummyExpr::new(),
            2,
        );
        assert_eq!(builder.num_interactions(), 2);

        // Test using bus helpers
        let bus = LookupBus::new(3);
        bus.lookup_key(&mut builder, vec![DummyExpr::new()], DummyExpr::new());
        assert_eq!(builder.num_interactions(), 3);

        let perm_bus = PermutationCheckBus::new(4);
        perm_bus.send(
            &mut builder,
            vec![DummyExpr::new(), DummyExpr::new()],
            DummyExpr::new(),
        );
        assert_eq!(builder.num_interactions(), 4);
    }

    #[test]
    fn test_air_builder_operations_are_noops() {
        use p3_baby_bear::BabyBear;
        type F = BabyBear;

        let mut builder = InteractionCounterBuilder::<F>::new();

        // Test that all constraint operations don't affect interaction count
        builder.assert_zero(DummyExpr::new());
        builder.assert_one(DummyExpr::new());
        builder.assert_eq(DummyExpr::new(), DummyExpr::new());

        assert_eq!(builder.num_interactions(), 0);

        // Test conditional builders
        builder
            .when(builder.is_first_row())
            .assert_zero(DummyExpr::new());
        builder.when_first_row().assert_one(DummyExpr::new());
        builder
            .when_last_row()
            .assert_eq(DummyExpr::new(), DummyExpr::new());
        builder.when_transition().assert_zero(DummyExpr::new());

        assert_eq!(builder.num_interactions(), 0);

        // Only interactions should be counted
        builder.push_interaction(0, vec![DummyExpr::new()], DummyExpr::new(), 1);
        assert_eq!(builder.num_interactions(), 1);
    }

    #[test]
    fn test_with_real_air() {
        use p3_baby_bear::BabyBear;

        let mut builder = InteractionCounterBuilder::<BabyBear>::new();

        // Simulate what an AIR would do
        let bus1 = LookupBus::new(0);
        let bus2 = PermutationCheckBus::new(1);

        // Send some interactions
        bus1.lookup_key(&mut builder, vec![DummyExpr::new()], DummyExpr::new());
        bus1.add_key_with_lookups(
            &mut builder,
            vec![DummyExpr::new(), DummyExpr::new()],
            DummyExpr::new(),
        );
        bus2.send(
            &mut builder,
            vec![DummyExpr::new(), DummyExpr::new()],
            DummyExpr::new(),
        );

        assert_eq!(builder.num_interactions(), 3);
    }
}
