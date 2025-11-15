use std::sync::Arc;

use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{FieldAlgebra, TwoAdicField};

use super::errors::FoldPleError;
use crate::{
    EF, F,
    cuda::logup_zerocheck::{compute_eq_sharp as compute_eq_sharp_ffi, fold_ple_from_evals},
};

/// Folds PLE evaluations by interpolating univariate polynomials on coset D and evaluating at r_0.
/// Returns a single matrix. When `rotate` is true, returns a doubled-width matrix:
/// Layout: [orig_col0...orig_colN, rot_col0...rot_colN] (width * 2 columns)
/// When `rotate` is false, returns a single-width matrix (width columns).
pub fn fold_ple_evals_gpu(
    l_skip: usize,
    mat: &DeviceMatrix<F>,
    r_0: EF,
    rotate: bool,
) -> Result<DeviceMatrix<EF>, FoldPleError> {
    let height = mat.height();
    let width = mat.width();

    if height == 0 || width == 0 {
        return Ok(DeviceMatrix::dummy());
    }

    let min_height = 1usize << l_skip;
    let lifted_height = if height > min_height {
        height
    } else {
        min_height
    };
    let domain_size = 1usize << l_skip;
    let new_height = lifted_height >> l_skip;

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

    // Allocate output buffer
    // When rotate=true: width * 2 columns (original + rotated)
    // When rotate=false: width columns (original only)
    let output_width = if rotate { width * 2 } else { width };
    let d_output = DeviceBuffer::<EF>::with_capacity(new_height * output_width);

    // Launch kernel
    unsafe {
        fold_ple_from_evals(
            mat.buffer(),
            &d_output,
            &d_numerators,
            &d_inv_lagrange_denoms,
            height as u32,
            width as u32,
            domain_size as u32,
            l_skip as u32,
            new_height as u32,
            rotate,
        )?;
    }

    Ok(DeviceMatrix::new(
        Arc::new(d_output),
        new_height,
        output_width,
    ))
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
