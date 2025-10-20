use openvm_stark_backend::{
    p3_matrix::{Matrix, dense::RowMajorMatrix},
    prover::MatrixDimensions,
};
use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Trait to consolidate different virtual matrices that may not be backed by an in-memory matrix
/// with a standard column-major or row-major layout.
///
/// Note: for performance, users should still be aware of the underlying memory layout. This trait
/// is just an organizational convenience.
pub trait MatrixView<F>: MatrixDimensions {
    fn get(&self, row_idx: usize, col_idx: usize) -> Option<&F> {
        if col_idx >= self.width() || row_idx >= self.height() {
            None
        } else {
            // SAFETY: bounds checked above
            Some(unsafe { self.get_unchecked(row_idx, col_idx) })
        }
    }
    /// Get a reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `col_idx` and `row_idx` are within the bounds of the matrix.
    unsafe fn get_unchecked(&self, row_idx: usize, col_idx: usize) -> &F;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColMajorMatrix<F> {
    pub values: Vec<F>,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ColMajorMatrixView<'a, F> {
    pub values: &'a [F],
    width: usize,
    height: usize,
}

impl<F> ColMajorMatrix<F> {
    pub fn new(values: Vec<F>, width: usize) -> Self {
        assert_eq!(values.len() % width, 0);
        let height = values.len() / width;
        assert!(height == 0 || height.is_power_of_two());
        Self {
            values,
            width,
            height,
        }
    }

    pub fn column(&self, col_idx: usize) -> &[F] {
        let start = col_idx * self.height;
        let end = start + self.height;
        &self.values[start..end]
    }

    pub fn columns(&self) -> impl Iterator<Item = &[F]> {
        self.values.chunks_exact(self.height)
    }

    pub fn as_view(&self) -> ColMajorMatrixView<'_, F> {
        ColMajorMatrixView {
            values: &self.values,
            width: self.width,
            height: self.height,
        }
    }

    // PERF[jpw]: currently this is not in-place transpose
    pub fn from_row_major(mat: &RowMajorMatrix<F>) -> Self
    where
        F: Field,
    {
        let mut values = F::zero_vec(mat.values.len());
        let width = mat.width;
        let height = mat.height();
        for r in 0..height {
            for c in 0..width {
                values[c * height + r] = mat.get(r, c);
            }
        }
        Self {
            values,
            width,
            height,
        }
    }
}

impl<F: Sync> ColMajorMatrix<F> {
    pub fn par_columns(&self) -> impl ParallelIterator<Item = &[F]> {
        self.values.par_chunks_exact(self.height)
    }
}

impl<F> MatrixDimensions for ColMajorMatrix<F> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
}

impl<F> MatrixView<F> for ColMajorMatrix<F> {
    unsafe fn get_unchecked(&self, row_idx: usize, col_idx: usize) -> &F {
        debug_assert!(col_idx < self.width);
        debug_assert!(row_idx < self.height);
        self.values.get_unchecked(col_idx * self.height + row_idx)
    }
}

impl<'a, F> ColMajorMatrixView<'a, F> {
    pub fn new(values: &'a [F], width: usize) -> Self {
        assert_eq!(values.len() % width, 0);
        let height = values.len() / width;
        assert!(height == 0 || height.is_power_of_two());
        Self {
            values,
            width,
            height,
        }
    }

    pub fn column(&self, col_idx: usize) -> &[F] {
        let start = col_idx * self.height;
        let end = start + self.height;
        &self.values[start..end]
    }

    pub fn columns(&self) -> impl Iterator<Item = &[F]> {
        self.values.chunks_exact(self.height)
    }
}

impl<F> MatrixDimensions for ColMajorMatrixView<'_, F> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
}

impl<F> MatrixView<F> for ColMajorMatrixView<'_, F> {
    unsafe fn get_unchecked(&self, row_idx: usize, col_idx: usize) -> &F {
        debug_assert!(col_idx < self.width);
        debug_assert!(row_idx < self.height);
        self.values
            .get_unchecked(col_maj_idx(row_idx, col_idx, self.height))
    }
}

#[inline(always)]
pub(crate) fn col_maj_idx(row_idx: usize, col_idx: usize, height: usize) -> usize {
    col_idx * height + row_idx
}
