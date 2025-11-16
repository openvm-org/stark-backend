use std::sync::Arc;

use getset::Getters;
use openvm_cuda_backend::{base::DeviceMatrix, ntt::batch_ntt};
use openvm_cuda_common::{
    copy::{MemCopyD2D, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::FieldAlgebra;

use crate::{
    EF, F, ProverError,
    cuda::poly::{
        eq_hypercube_nonoverlapping_stage_ext, eq_hypercube_stage_ext, mle_interpolate_stage,
        mle_interpolate_stage_ext,
    },
};

#[derive(derive_new::new)]
pub struct PleMatrix<F> {
    /// Evaluations on hyperprism D_n.
    pub evals: DeviceMatrix<F>,
    /// Stores the coefficient form of the univariate polynomial `f(Z, \vect x)` for each `\vect x
    /// in H_n`. Buffer size is the same as `evals`.
    pub mixed: DeviceBuffer<F>,
}

impl MatrixDimensions for PleMatrix<F> {
    fn width(&self) -> usize {
        self.evals.width()
    }

    fn height(&self) -> usize {
        self.evals.height()
    }
}

impl PleMatrix<F> {
    /// Creates a `PleMatrix`. This doubles the VRAM footprint to cache the `mixed` buffer.
    pub fn from_evals(l_skip: usize, evals: DeviceMatrix<F>) -> Result<Self, ProverError> {
        let width = evals.width();
        let height = evals.height();
        // D2D copy so we can do in-place iNTT
        let mixed = evals.buffer().device_copy()?;
        if l_skip > 0 {
            // For univariate coordinate, perform inverse NTT for each 2^l_skip chunk per column:
            // (width cols) * (height / 2^l_skip chunks per col). Use natural ordering.
            let num_uni_poly = (width * (height >> l_skip)) as u32;
            batch_ntt(&mixed, l_skip as u32, 0, num_uni_poly, true, true);
        }
        Ok(Self { evals, mixed })
    }

    pub fn to_evals(&self, l_skip: usize) -> Result<DeviceMatrix<F>, ProverError> {
        let width = self.width();
        let height = self.height();
        // D2D copy so we can do in-place NTT
        let evals = self.mixed.device_copy()?;
        if l_skip > 0 {
            // For univariate coordinate, perform NTT for each 2^l_skip chunk per column: (width
            // cols) * (height / 2^l_skip chunks per col). Use natural ordering.
            let num_uni_poly = (width * (height >> l_skip)) as u32;
            batch_ntt(&evals, l_skip as u32, 0, num_uni_poly, true, false);
        }
        Ok(DeviceMatrix::new(Arc::new(evals), height, width))
    }
}

/// Assumes that `evals` is column-major matrix of evaluations on a hypercube `H_n`.
/// In-place interpolates `evals` from evaluations to coefficient form.
pub fn mle_evals_to_coeffs_inplace(
    evals: &mut DeviceBuffer<F>,
    n: usize,
) -> Result<(), ProverError> {
    debug_assert!(evals.len().is_power_of_two());
    debug_assert!(evals.len() >= 1 << n);
    for log_step in 0..n {
        let step = 1 << log_step;
        // SAFETY: `f_evals` has length `2^n >= 2 * step`.
        unsafe {
            // Multilinear Eval -> Coeff
            mle_interpolate_stage(evals, step, true)?;
        }
    }
    Ok(())
}

/// Assumes that `evals` is column-major matrix of evaluations on a hypercube `H_n`.
/// In-place interpolates `evals` from evaluations to coefficient form.
pub fn mle_evals_to_coeffs_inplace_ext(
    evals: &mut DeviceBuffer<EF>,
    n: usize,
) -> Result<(), ProverError> {
    debug_assert!(evals.len().is_power_of_two());
    debug_assert!(evals.len() >= 1 << n);
    for log_step in 0..n {
        let step = 1 << log_step;
        // SAFETY: `f_evals` has length `2^n >= 2 * step`.
        unsafe {
            // Multilinear Eval -> Coeff
            mle_interpolate_stage_ext(evals, step, true)?;
        }
    }
    Ok(())
}

/// Given vector `x` in `F^n`, populates `out` with `eq_n(x, y)` for `y` on hypercube `H_n`.
///
/// Note: This function launches `n` CUDA kernels.
///
/// The multilinear equality polynomial is defined as:
/// ```text
///     eq(x, z) = \prod_{i=0}^{n-1} (x_i z_i + (1 - x_i)(1 - z_i)).
/// ```
///
/// # Safety
/// - `n` is set to the length of `xs`.
/// - `out` must have length `>= 2^n`.
pub unsafe fn evals_eq_hypercube(out: &mut DeviceBuffer<EF>, xs: &[EF]) -> Result<(), ProverError> {
    let n = xs.len();
    assert!(out.len() >= 1 << n);
    // Use memcpy instead of memset since EF will be in Montgomery form.
    [EF::ONE].copy_to(out)?;

    for (i, &x_i) in xs.iter().enumerate() {
        let step = 1 << i;
        eq_hypercube_stage_ext(out.as_mut_ptr(), x_i, step)?;
    }
    Ok(())
}

/// For a fixed `x` in `F^max_n`, stores the evaluations of `eq_n(x[..n], -)` on hypercube `H_n` for
/// `n = 0..=max_n`. The evaluations are stored contiguously in a single buffer of length `2^(max_n
/// + 1)`. We define `eq_0 = 1` always.
#[derive(Getters)]
pub struct EqEvalSegments<F> {
    /// Index 0 should never to read, but it will be initialized to zero.
    #[getset(get = "pub")]
    pub(crate) buffer: DeviceBuffer<F>,
    #[getset(get_copy = "pub")]
    max_n: usize,
}

impl<F> EqEvalSegments<F> {
    /// Returns start pointer for buffer of length `2^n` corresponding to evaluations of `eq(x[..n],
    /// -)` on `H_n`.
    pub fn get_ptr(&self, n: usize) -> *const F {
        assert!(n <= self.max_n);
        // SAFETY: `buffer` has length `2^{max_n + 1}`
        unsafe { self.buffer.as_ptr().add(1 << n) }
    }

    /// # Safety
    /// Caller must ensure that `buffer` has length `2^{max_n + 1}` and is properly initialized.
    pub unsafe fn from_raw_parts(buffer: DeviceBuffer<F>, max_n: usize) -> Self {
        Self { buffer, max_n }
    }
}

// Currently only implement kernels for EF.
impl EqEvalSegments<EF> {
    /// Creates a new `EqEvalSegments` instance with `max_n = x.len()`.
    pub fn new(x: &[EF]) -> Result<Self, ProverError> {
        let max_n = x.len();
        let mut buffer = DeviceBuffer::with_capacity(2 << max_n);
        // Index 0 should never to be used, but we initialize it to zero.
        // Index 1 is set to eq_0 = 1 for initial state
        [EF::ZERO, EF::ONE].copy_to(&mut buffer)?;
        // At step i, we populate `eq_{i+1}` starting at offset `2^{i+1}`
        for (i, &x_i) in x.iter().enumerate() {
            let step = 1 << i;
            // SAFETY:
            // - `buffer` has length `2^{max_n + 1}`
            // - `2 * 2^{i+1} <= 2^{max_n + 1}` ensures everything is in bounds
            unsafe {
                // The `eq_{i+1}` segment starts at offset `2^{i+1}`
                let dst = buffer.as_mut_ptr().add(2 * step);
                // The `eq_i` segment starts at offset `2^i`
                let src = buffer.as_ptr().add(step);
                eq_hypercube_nonoverlapping_stage_ext(dst, src, x_i, step as u32)?;
            }
        }
        Ok(Self { buffer, max_n })
    }
}
