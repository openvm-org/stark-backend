//! Row-major [ProverBackend] implementation.

use std::marker::PhantomData;

use openvm_stark_backend::{
    prover::{stacked_pcs::StackedPcsData, MatrixDimensions, ProverBackend},
    StarkProtocolConfig,
};
use p3_matrix::dense::RowMajorMatrix;

/// Newtype wrapper around `RowMajorMatrix<F>` implementing `MatrixDimensions`.
pub struct RowMajorMatrixWrapper<F> {
    pub inner: RowMajorMatrix<F>,
}

impl<F> RowMajorMatrixWrapper<F> {
    pub fn new(inner: RowMajorMatrix<F>) -> Self {
        Self { inner }
    }
}

impl<F: Clone + Send + Sync> MatrixDimensions for RowMajorMatrixWrapper<F> {
    fn height(&self) -> usize {
        p3_matrix::Matrix::height(&self.inner)
    }
    fn width(&self) -> usize {
        self.inner.width
    }
}

// SAFETY: RowMajorMatrix<F> is Send + Sync when F is
unsafe impl<F: Send> Send for RowMajorMatrixWrapper<F> {}
unsafe impl<F: Sync> Sync for RowMajorMatrixWrapper<F> {}

/// Row-major CPU prover backend.
///
/// Uses `RowMajorMatrixWrapper<SC::F>` as the matrix type for better cache locality
/// during constraint evaluation.
#[derive(Clone, Copy)]
pub struct CpuBackend<SC: StarkProtocolConfig>(PhantomData<SC>);

impl<SC: StarkProtocolConfig> CpuBackend<SC> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<SC: StarkProtocolConfig> Default for CpuBackend<SC> {
    fn default() -> Self {
        Self::new()
    }
}

impl<SC: StarkProtocolConfig> ProverBackend for CpuBackend<SC> {
    const CHALLENGE_EXT_DEGREE: u8 = SC::D_EF as u8;

    type Val = SC::F;
    type Challenge = SC::EF;
    type Commitment = SC::Digest;
    type Matrix = RowMajorMatrixWrapper<SC::F>;
    type OtherAirData = ();
    type PcsData = StackedPcsData<SC::F, SC::Digest>;
}
