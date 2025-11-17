use std::{cmp::max, sync::Arc};

use openvm_cuda_backend::base::{DeviceMatrix, DeviceMatrixView};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{FieldAlgebra, TwoAdicField};

use super::errors::FoldPleError;
use crate::{
    EF, F,
    cuda::{
        logup_zerocheck::{compute_eq_sharp as compute_eq_sharp_ffi, fold_ple_from_evals},
        sumcheck::fold_ple_from_coeffs,
    },
};

/// Folds plain using mixed coefficients, folds rotation from evals.
/// - `mixed` should be mixed coefficient form of the _lifted_ trace.
/// - `trace_evals` should be unlifted (the original trace).
pub fn fold_ple_mixed_rotate(
    l_skip: usize,
    trace_evals: &DeviceMatrix<F>,
    mixed: DeviceMatrixView<'_, F>,
    r_0: EF,
) -> Result<DeviceMatrix<EF>, FoldPleError> {
    let width = trace_evals.width();
    let height = trace_evals.height();
    let num_x = max(height >> l_skip, 1);
    let folded_buf = DeviceBuffer::<EF>::with_capacity(num_x * width * 2);
    debug_assert_eq!(mixed.width(), width);
    debug_assert_eq!(mixed.height(), num_x << l_skip);
    // SAFETY:
    // - We allocated `folded_buf` for `num_x * width * 2` elements.
    // - `mixed` is `(num_x * 2^l_skip) x width` matrix
    // - `trace_evals` is `height x width` unlighted matrix
    unsafe {
        if height >= (1 << l_skip) {
            // This means stride = 1 so we can fold the plain part from mixed coefficients
            fold_ple_from_coeffs(
                mixed.as_ptr(),
                folded_buf.as_mut_ptr(),
                num_x as u32,
                width as u32,
                1 << l_skip,
                r_0,
            )?;
        } else {
            debug_assert_eq!(num_x, 1);
            // Otherwise there is striding, so we can't use mixed form
            // Fold directly from plain evaluations
            fold_ple_evals_gpu(l_skip, trace_evals, folded_buf.as_mut_ptr(), r_0, false)?;
        }

        // Fold the rotation from evals
        fold_ple_evals_gpu(
            l_skip,
            trace_evals,
            folded_buf.as_mut_ptr().add(num_x * width),
            r_0,
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
    mat: &DeviceMatrix<F>,
    output: *mut EF,
    r_0: EF,
    rotate: bool,
) -> Result<(), FoldPleError> {
    let height = mat.height();
    let width = mat.width();

    if height == 0 || width == 0 {
        return Ok(());
    }

    let domain_size = 1usize << l_skip;
    let lifted_height = max(domain_size, height);
    let num_x = lifted_height >> l_skip;

    // Precompute denominators and numerators on host
    let omega = F::two_adic_generator(l_skip);
    let omega_pows: Vec<F> = omega.powers().take(domain_size).collect();

    // Compute numerators: Π_{j≠i} (r_0 - omega^j) for each i
    let numerators: Vec<EF> = (0..domain_size)
        .map(|i| {
            let mut num = EF::ONE;
            for (j, &omega_j) in omega_pows.iter().enumerate() {
                if j != i {
                    num *= r_0 - EF::from(omega_j);
                }
            }
            num
        })
        .collect();

    // Compute denominators for Lagrange basis: Π_{j≠i} (omega^i - omega^j) for each i
    let lagrange_denoms: Vec<EF> = (0..domain_size)
        .map(|i| {
            let mut denom = EF::ONE;
            let omega_i = omega_pows[i];
            for (j, &omega_j) in omega_pows.iter().enumerate() {
                if j != i {
                    denom *= EF::from(omega_i) - EF::from(omega_j);
                }
            }
            denom
        })
        .collect();
    let inv_lagrange_denoms: Vec<EF> = p3_field::batch_multiplicative_inverse(&lagrange_denoms);

    // Copy to device
    let d_numerators = numerators.to_device().map_err(FoldPleError::Copy)?;
    let d_inv_lagrange_denoms = inv_lagrange_denoms
        .to_device()
        .map_err(FoldPleError::Copy)?;

    // Launch kernel
    unsafe {
        fold_ple_from_evals(
            mat.buffer(),
            output,
            &d_numerators,
            &d_inv_lagrange_denoms,
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
    eq_xi: &mut DeviceMatrix<EF>,
    eq_r0: EF,
    eq_sharp_r0: EF,
) -> Result<DeviceMatrix<EF>, FoldPleError> {
    let height = eq_xi.height();
    let width = eq_xi.width();
    let count = height * width;

    // Allocate output buffer for eq_sharp
    let eq_sharp_buffer = DeviceBuffer::<EF>::with_capacity(count);

    // Call the combined kernel that mutates eq_xi in-place and computes eq_sharp
    // SAFETY: We have exclusive access to eq_xi via &mut DeviceMatrix, so mutating its buffer is
    // safe
    unsafe {
        compute_eq_sharp_ffi(eq_xi.buffer(), &eq_sharp_buffer, eq_r0, eq_sharp_r0)?;
    }

    Ok(DeviceMatrix::new(Arc::new(eq_sharp_buffer), height, width))
}
