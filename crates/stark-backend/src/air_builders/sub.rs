// Copied from sp1/core under MIT license

use std::ops::{Deref, Range};

use p3_air::{AirBuilder, BaseAir};
use p3_matrix::Matrix;

/// A submatrix of a matrix.  The matrix will contain a subset of the columns of `self.inner`.
pub struct SubMatrixRowSlices<M: Matrix<T>, T: Clone + Send + Sync> {
    inner: M,
    column_range: Range<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<M: Matrix<T>, T: Clone + Send + Sync> SubMatrixRowSlices<M, T> {
    pub const fn new(inner: M, column_range: Range<usize>) -> Self {
        Self {
            inner,
            column_range,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Implement `Matrix` for `SubMatrixRowSlices`.
impl<M: Matrix<T>, T: Clone + Send + Sync> Matrix<T> for SubMatrixRowSlices<M, T> {
    #[inline]
    fn get(&self, r: usize, c: usize) -> Option<T> {
        let c = self.column_range.start + c;
        if c >= self.column_range.end {
            return None;
        }
        self.inner.get(r, c)
    }

    #[inline]
    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        let global_start = self.column_range.start + start;
        let global_end = self.column_range.start + end;
        self.inner
            .row_unchecked(r)
            .into_iter()
            .skip(global_start)
            .take(global_end - global_start)
    }

    #[inline]
    fn row_slice(&self, r: usize) -> Option<impl Deref<Target = [T]>> {
        self.inner.row(r).map(|row| {
            row.into_iter()
                .skip(self.column_range.start)
                .take(self.width())
                .collect::<Vec<_>>()
        })
    }

    #[inline]
    fn width(&self) -> usize {
        self.column_range.len()
    }

    #[inline]
    fn height(&self) -> usize {
        self.inner.height()
    }
}

/// A builder used to eval a sub-air.  This will handle enforcing constraints for a subset of a
/// trace matrix.  E.g. if a particular air needs to be enforced for a subset of the columns of
/// the trace, then the SubAirBuilder can be used.
pub struct SubAirBuilder<'a, AB: AirBuilder, SubAir: BaseAir<T>, T> {
    inner: &'a mut AB,
    column_range: Range<usize>,
    _phantom: std::marker::PhantomData<(SubAir, T)>,
}

impl<'a, AB: AirBuilder, SubAir: BaseAir<T>, T> SubAirBuilder<'a, AB, SubAir, T> {
    pub fn new(inner: &'a mut AB, column_range: Range<usize>) -> Self {
        Self {
            inner,
            column_range,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Implement `AirBuilder` for `SubAirBuilder`.
impl<AB: AirBuilder, SubAir: BaseAir<F>, F> AirBuilder for SubAirBuilder<'_, AB, SubAir, F> {
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type M = SubMatrixRowSlices<AB::M, Self::Var>;

    fn main(&self) -> Self::M {
        let matrix = self.inner.main();

        SubMatrixRowSlices::new(matrix, self.column_range.clone())
    }

    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x.into());
    }
}
