use std::{cmp::max, sync::Arc};

use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::prover::MatrixDimensions;

use super::errors::FoldPleError;
use crate::{
    EF, F,
    cuda::logup_zerocheck::{compute_eq_sharp as compute_eq_sharp_ffi, fold_ple_from_evals},
};

/// Folds plain using mixed coefficients, folds rotation from evals.
/// - `mixed` should be mixed coefficient form of the _lifted_ trace.
/// - `trace_evals` should be unlifted (the original trace).
pub fn fold_ple_mixed_rotate(
    l_skip: usize,
    d_omega_skip_pows: &DeviceBuffer<F>,
    trace_evals: &DeviceMatrix<F>,
    d_inv_lagrange_denoms_r0: &DeviceBuffer<EF>,
) -> Result<DeviceMatrix<EF>, FoldPleError> {
    let width = trace_evals.width();
    let height = trace_evals.height();
    let num_x = max(height >> l_skip, 1);
    let folded_buf = DeviceBuffer::<EF>::with_capacity(num_x * width * 2);
    // SAFETY:
    // - We allocated `folded_buf` for `num_x * width * 2` elements.
    // - `trace_evals` is `height x width` unlighted matrix
    unsafe {
        fold_ple_evals_gpu(
            l_skip,
            d_omega_skip_pows,
            trace_evals,
            folded_buf.as_mut_ptr(),
            d_inv_lagrange_denoms_r0,
            false,
        )?;

        // Fold the rotation from evals
        fold_ple_evals_gpu(
            l_skip,
            d_omega_skip_pows,
            trace_evals,
            folded_buf.as_mut_ptr().add(num_x * width),
            d_inv_lagrange_denoms_r0,
            true,
        )?;
    }
    let folded = DeviceMatrix::new(Arc::new(folded_buf), num_x, width * 2);
    Ok(folded)
}

/// Folds PLE evaluations by interpolating univariate polynomials on coset D and evaluating at r_0.
/// Returns a single matrix of width `width`.
///
/// When `rotate` is true, returns the folding of the lift of the rotated matrix.
/// When `rotate` is false, returns the folding of the lift of the original matrix.
///
/// # Assumptions
/// - `mat` should be the unlifted original matrix of trace evaluations.
/// - `output` should be a valid pointer to a buffer of size at least `num_x * width` where `num_x =
///   max(height / 2^l_skip, 1)`.
pub unsafe fn fold_ple_evals_gpu(
    l_skip: usize,
    d_omega_skip_pows: &DeviceBuffer<F>,
    mat: &DeviceMatrix<F>,
    output: *mut EF,
    d_inv_lagrange_denoms_r0: &DeviceBuffer<EF>,
    rotate: bool,
) -> Result<(), FoldPleError> {
    let height = mat.height();
    let width = mat.width();

    if height == 0 || width == 0 {
        return Ok(());
    }

    let skip_domain = d_omega_skip_pows.len();
    debug_assert_eq!(skip_domain, 1 << l_skip);
    let lifted_height = max(skip_domain, height);
    let num_x = lifted_height / skip_domain;

    // Launch kernel
    unsafe {
        fold_ple_from_evals(
            mat.buffer(),
            output,
            d_omega_skip_pows,
            d_inv_lagrange_denoms_r0,
            height as u32,
            width as u32,
            l_skip as u32,
            num_x as u32,
            rotate,
        )?;
    }
    Ok(())
}

/// Multiply eq_xi by eq_r0 in-place and compute eq_sharp = original_eq_xi * eq_sharp_r0
/// This combines both operations in a single kernel call for efficiency
pub fn compute_eq_sharp_gpu(
    eq_xis: &mut DeviceBuffer<EF>,
    eq_r0: EF,
    eq_sharp_r0: EF,
) -> Result<DeviceBuffer<EF>, FoldPleError> {
    let count = eq_xis.len();

    // Allocate output buffer for eq_sharp
    let eq_sharp_buffer = DeviceBuffer::<EF>::with_capacity(count);

    // Call the combined kernel that mutates eq_xi in-place and computes eq_sharp
    // SAFETY: We have exclusive access to eq_xi via &mut DeviceMatrix, so mutating its buffer is
    // safe
    unsafe {
        compute_eq_sharp_ffi(eq_xis, &eq_sharp_buffer, eq_r0, eq_sharp_r0)?;
    }

    Ok(eq_sharp_buffer)
}
