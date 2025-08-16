use openvm_stark_backend::prover::hal::MatrixDimensions;

use crate::{base::DeviceMatrix, prelude::F};

pub(crate) mod ops;
use ops::*;

/// The top-level LDE abstraction, composed of general matrix access (dimensions),
/// trace access, and LDE behavior (which varies by mode).
pub trait GpuLde: MatrixDimensions + LdeCommon {
    /// Constructs a new LDE wrapper using the given trace matrix, blowup factor, and shift.
    ///
    /// - `added_bits` determines the blowup factor for LDE computation.
    /// - `shift` determines the coset used for LDE.
    fn new(matrix: DeviceMatrix<F>, added_bits: usize, shift: F) -> Self
    where
        Self: Sized;

    /// Returns the LDE matrix if domain size <= lde.height.
    fn take_lde(&self, domain_size: usize) -> DeviceMatrix<F>;

    /// Returns the LDE rows for the given indices.
    fn get_lde_rows(&self, row_indices: &[usize]) -> DeviceMatrix<F>;
}

pub trait LdeCommon {
    /// Returns the number of rows in the original trace matrix.
    fn trace_height(&self) -> usize;

    fn shift(&self) -> F;
}

#[derive(Clone)]
pub struct GpuLdeImpl {
    lde: DeviceMatrix<F>,
    added_bits: usize,
    shift: F,
}

impl MatrixDimensions for GpuLdeImpl {
    fn height(&self) -> usize {
        self.lde.height()
    }

    fn width(&self) -> usize {
        self.lde.width()
    }
}

impl LdeCommon for GpuLdeImpl {
    fn shift(&self) -> F {
        self.shift
    }

    fn trace_height(&self) -> usize {
        self.lde.height() >> self.added_bits
    }
}

impl GpuLde for GpuLdeImpl {
    fn new(matrix: DeviceMatrix<F>, added_bits: usize, shift: F) -> Self {
        if added_bits == 0 {
            return Self {
                lde: matrix,
                added_bits,
                shift,
            };
        }
        let trace_height = matrix.height();
        let lde_height = trace_height << added_bits;
        let lde = compute_lde_matrix(&matrix, lde_height, shift);
        Self {
            lde,
            added_bits,
            shift,
        }
    }

    fn take_lde(&self, domain_size: usize) -> DeviceMatrix<F> {
        assert!(self.height() >= domain_size);
        self.lde.clone()
    }

    fn get_lde_rows(&self, row_indices: &[usize]) -> DeviceMatrix<F> {
        assert!(!row_indices.is_empty());
        get_rows_from_matrix(&self.lde, row_indices)
    }
}
