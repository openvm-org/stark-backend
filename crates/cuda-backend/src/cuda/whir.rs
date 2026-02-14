use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{check, CudaError},
};

use crate::{
    prelude::{D_EF, EF, F},
    whir::BatchingTracePacket,
};

extern "C" {
    fn _whir_algebraic_batch_traces(
        output: *mut F,
        packets: *const BatchingTracePacket,
        mu_powers: *const EF,
        stacked_height: usize,
        num_packets: usize,
        skip_domain: u32,
    ) -> i32;

    pub fn _whir_sumcheck_coeff_moments_required_temp_buffer_size(height: u32) -> u32;

    fn _whir_sumcheck_coeff_moments_round(
        f_coeffs: *const EF,
        w_moments: *const EF,
        output: *mut EF,
        tmp_block_sums: *mut EF,
        height: u32,
    ) -> i32;

    fn _whir_fold_coeffs_and_moments(
        f_coeffs: *const EF,
        w_moments: *const EF,
        f_folded_coeffs: *mut EF,
        w_folded_moments: *mut EF,
        alpha: EF,
        height: u32,
    ) -> i32;

    fn _w_moments_accumulate(
        w_moments: *const EF,
        z0_pows2: *const EF,
        z_pows2: *const F,
        gamma: EF,
        num_queries: u32,
        log_height: u32,
        height: u32,
    ) -> i32;
}

/// # Safety
/// - `packets` must contain pointers `ptr` to device buffers valid for `height x width` elements.
///   The `stacked_row_start` must be the correct starting row index for the trace within the
///   stacked matrix.
/// - `mu_powers` must be defined for at least the sum of the stacked widths.
pub unsafe fn whir_algebraic_batch_traces(
    output: &mut DeviceBuffer<F>,
    packets: &DeviceBuffer<BatchingTracePacket>,
    mu_powers: &DeviceBuffer<EF>,
    skip_domain: u32,
) -> Result<(), CudaError> {
    debug_assert_eq!(output.len() % D_EF, 0);
    check(_whir_algebraic_batch_traces(
        output.as_mut_ptr(),
        packets.as_ptr(),
        mu_powers.as_ptr(),
        output.len() / D_EF,
        packets.len(),
        skip_domain,
    ))
}

/// Performs a WHIR sumcheck round with `f` in coefficient form and `w` in moment form.
///
/// # Safety
/// - `f_coeffs` and `w_moments` are the same length, which is a power of two.
/// - `output` has length at least 2.
/// - `tmp_block_sums` has length at least
///   `_whir_sumcheck_coeff_moments_required_temp_buffer_size(height)`.
pub unsafe fn whir_sumcheck_coeff_moments_round(
    f_coeffs: &DeviceBuffer<EF>,
    w_moments: &DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    height: u32,
) -> Result<(), CudaError> {
    debug_assert!(f_coeffs.len() >= height as usize);
    debug_assert!(w_moments.len() >= height as usize);
    #[cfg(debug_assertions)]
    {
        let len = tmp_block_sums.len();
        let required = _whir_sumcheck_coeff_moments_required_temp_buffer_size(height);
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    check(_whir_sumcheck_coeff_moments_round(
        f_coeffs.as_ptr(),
        w_moments.as_ptr(),
        output.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
        height,
    ))
}

/// Folds `f` (coefficient form) and `w` (moment form) for one WHIR round.
///
/// # Safety
/// - `f_coeffs` and `w_moments` have length `height`, which is a power of two.
/// - `f_folded_coeffs` and `w_folded_moments` have length `height / 2`.
pub unsafe fn whir_fold_coeffs_and_moments(
    f_coeffs: &DeviceBuffer<EF>,
    w_moments: &DeviceBuffer<EF>,
    f_folded_coeffs: &mut DeviceBuffer<EF>,
    w_folded_moments: &mut DeviceBuffer<EF>,
    alpha: EF,
    height: u32,
) -> Result<(), CudaError> {
    debug_assert!(f_coeffs.len() >= height as usize);
    debug_assert!(w_moments.len() >= height as usize);
    debug_assert!(f_folded_coeffs.len() >= (height as usize / 2));
    debug_assert!(w_folded_moments.len() >= (height as usize / 2));
    check(_whir_fold_coeffs_and_moments(
        f_coeffs.as_ptr(),
        w_moments.as_ptr(),
        f_folded_coeffs.as_mut_ptr(),
        w_folded_moments.as_mut_ptr(),
        alpha,
        height,
    ))
}

/// Updates WHIR moment vector in place by adding gamma-weighted query terms.
///
/// # Safety
/// - `w_moments` has length `height = 2^log_height`.
/// - `z0_pows2` has length `log_height`.
/// - `z_pows2` has length `num_queries * log_height`.
pub unsafe fn w_moments_accumulate(
    w_moments: &mut DeviceBuffer<EF>,
    z0_pows2: &DeviceBuffer<EF>,
    z_pows2: &DeviceBuffer<F>,
    gamma: EF,
    num_queries: u32,
    log_height: u32,
) -> Result<(), CudaError> {
    let height = w_moments.len();
    debug_assert_eq!(z0_pows2.len(), log_height as usize);
    debug_assert_eq!(z_pows2.len(), num_queries as usize * log_height as usize);
    check(_w_moments_accumulate(
        w_moments.as_mut_ptr(),
        z0_pows2.as_ptr(),
        z_pows2.as_ptr(),
        gamma,
        num_queries,
        log_height,
        height as u32,
    ))
}
