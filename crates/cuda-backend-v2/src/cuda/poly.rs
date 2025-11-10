use openvm_cuda_common::{copy::MemCopyD2H, d_buffer::DeviceBuffer, error::CudaError};

use crate::{D_EF, EF, F, ProverError};

extern "C" {
    fn _mle_interpolate_stage(
        buffer: *mut F,
        total_len: usize,
        step: u32,
        is_eval_to_coeff: bool,
    ) -> i32;

    fn _mle_interpolate_stage_ext(
        buffer: *mut EF,
        total_len: usize,
        step: u32,
        is_eval_to_coeff: bool,
    ) -> i32;

    fn _algebraic_batch_matrices(
        output: *mut EF,
        mats: *const *const F,
        mu_powers: *const EF,
        mu_idxs: *const u32,
        widths: *const u32,
        height: usize,
        num_mats: usize,
    ) -> i32;

    fn _eq_hypercube_stage_ext(out: *mut EF, x_i: EF, step: u32) -> i32;

    // `x` must be **device** buffer
    fn _batch_eq_hypercube_stage(
        out: *mut F,
        x: *const F,
        step: u32,
        width: u32,
        height: u32,
    ) -> i32;

    // `out` must be device ptr
    fn _eval_poly_at_point(coeffs: *const F, coeff_len: usize, x: F, out: *mut F) -> i32;

    // `out` must be device ptr
    fn _eval_poly_ext_at_point(base_coeffs: *const F, coeff_len: usize, x: EF, out: *mut EF)
    -> i32;
}

/// Does in-place interpolation on `buffer` from eval to coeff form in one coordinate, assuming the
/// associated polynomial is linear in that coordinate. Effectively it performs `v -= u` for `v, u`
/// that are `step` apart in the buffer.
///
/// The boolean `is_eval_to_coeff` indicates the direction of the transformation:
/// - If `true`, it performs eval to coeff transformation.
/// - If `false`, it performs coeff to eval transformation.
///
/// # Safety
/// - The `buffer` must be allocated and initialized on device in the default stream.
/// - `step` must be a power of two and less than or equal to half the length of the buffer.
pub unsafe fn mle_interpolate_stage(
    buffer: &mut DeviceBuffer<F>,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), CudaError> {
    CudaError::from_result(_mle_interpolate_stage(
        buffer.as_mut_ptr(),
        buffer.len(),
        step,
        is_eval_to_coeff,
    ))
}

/// Same as [mle_interpolate_stage] for extension field `EF`.
pub unsafe fn mle_interpolate_stage_ext(
    buffer: &mut DeviceBuffer<EF>,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), CudaError> {
    CudaError::from_result(_mle_interpolate_stage_ext(
        buffer.as_mut_ptr(),
        buffer.len(),
        step,
        is_eval_to_coeff,
    ))
}

/// Computes the algebraic batch of the column vectors from input matrices `mats`, in order.
/// In other words, `output = sum_i sum_{j=0..widths[i]} mu_powers[mu_idx[i]] * mats[i][j]`.
///
/// # Safety
/// - `output` must have length `>= height`.
/// - `mats[i]` must have length `>= widths[i] * height` for all `i`.
/// - `mats`, `mu_idxs`, `widths` must have length `>= num_mats`.
/// - `mu_idxs[i] < mu_powers.len()` for all `i`.
pub unsafe fn algebraic_batch_matrices(
    output: &mut DeviceBuffer<EF>,
    mat_ptrs: &DeviceBuffer<*const F>,
    mu_powers: &DeviceBuffer<EF>,
    mu_idxs: &DeviceBuffer<u32>,
    widths: &DeviceBuffer<u32>,
    height: usize,
    num_mats: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_algebraic_batch_matrices(
        output.as_mut_ptr(),
        mat_ptrs.as_ptr(),
        mu_powers.as_ptr(),
        mu_idxs.as_ptr(),
        widths.as_ptr(),
        height,
        num_mats,
    ))
}

/// Performs an in-place update of:
/// ```text
/// out[i + step] = out[i] * x_i
/// out[i] = out[i] * (1 - x_i)
/// ```
/// for `i` in `0..step`.
///
/// # Safety
/// - `out` is **device** pointer with length `>= 2 * step`.
pub unsafe fn eq_hypercube_stage_ext(out: *mut EF, x_i: EF, step: u32) -> Result<(), CudaError> {
    CudaError::from_result(_eq_hypercube_stage_ext(out, x_i, step))
}

/// Same as `eq_hypercube_stage`, over base field, but computes `eq` evals in batch for multiple
/// `x_i`.
///
/// # Safety
/// - `out` must be column-major matrix with size `>= width * height` where `width = x.len()`.
/// - `step < height` is a power of two.
pub unsafe fn batch_eq_hypercube_stage(
    out: &mut DeviceBuffer<F>,
    x: &DeviceBuffer<F>,
    step: u32,
    height: u32,
) -> Result<(), CudaError> {
    let width = x.len() as u32;
    debug_assert!(step < height);
    debug_assert!(out.len() <= (width * height) as usize);
    CudaError::from_result(_batch_eq_hypercube_stage(
        out.as_mut_ptr(),
        x.as_ptr(),
        step,
        width,
        height,
    ))
}

/// Evaluates a polynomial stored in `coeffs` (little-endian coefficients) at point `x`.
///
/// # Safety
/// - `coeff_len` specifies how many coefficients to read from `coeffs`. `coeff_len` must be `>0`.
#[allow(dead_code)]
pub unsafe fn eval_poly_at_point(coeffs: &DeviceBuffer<F>, x: F) -> Result<F, ProverError> {
    let d_out = DeviceBuffer::<F>::with_capacity(1);
    CudaError::from_result(_eval_poly_at_point(
        coeffs.as_ptr(),
        coeffs.len(),
        x,
        d_out.as_mut_ptr(),
    ))?;
    let out = d_out.to_host()?;
    Ok(out[0])
}

/// Evaluates a `EF`-polynomial stored in coefficient form as column-major `F`-matrix at point `x`.
///
/// # Safety
/// - `base_coeffs` is in column-major form for the **base** field `F`.
/// - `base_coeffs.len() >= coeff_len * D_EF`, where `D_EF` is the degree of extension for `EF`.
pub unsafe fn eval_poly_ext_at_point_from_base(
    base_coeffs: &DeviceBuffer<F>,
    coeff_len: usize,
    x: EF,
) -> Result<EF, ProverError> {
    debug_assert!(base_coeffs.len() >= coeff_len * D_EF);
    let d_out = DeviceBuffer::<EF>::with_capacity(1);
    CudaError::from_result(_eval_poly_ext_at_point(
        base_coeffs.as_ptr(),
        coeff_len,
        x,
        d_out.as_mut_ptr(),
    ))?;
    let out = d_out.to_host()?;
    Ok(out[0])
}
