use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{CudaError, check},
};

use crate::{EF, F};

extern "C" {
    pub fn _whir_sumcheck_required_temp_buffer_size(height: u32) -> u32;

    fn _whir_sumcheck_mle_round(
        f_evals: *const EF,
        w_evals: *const EF,
        output: *mut EF,
        tmp_block_sums: *mut EF,
        height: u32,
    ) -> i32;

    fn _w_evals_accumulate(
        w_evals: *const EF,
        eq_z0: *const EF,
        eq_zs: *const F,
        gamma: EF,
        num_queries: u32,
        height: u32,
    ) -> i32;
}

/// Performs a sumcheck round in WHIR on the evaluations of `f` and `w` on the hypercube.
/// Writes the evaluations of the univariate polynomial `s` at `{1, 2}` to `output`.
/// The `tmp_block_sums` buffer is used for intermediate storage by the kernel.
///
/// # Safety
/// - `f_evals` and `w_evals` are the same length, which is a power of two.
/// - `output` has length at least 2.
/// - `tmp_block_sums` has length at least `num_blocks * 2` where `num_blocks = (height /
///   2).div_ceil(threads_per_block)`. Note that `threads_per_blocks` is by default `1024` but this
///   depends on the kernel launcher configuration.
pub unsafe fn whir_sumcheck_mle_round(
    f_evals: &DeviceBuffer<EF>,
    w_evals: &DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    height: u32,
) -> Result<(), CudaError> {
    debug_assert!(f_evals.len() >= height as usize);
    debug_assert!(w_evals.len() >= height as usize);
    #[cfg(debug_assertions)]
    {
        let len = tmp_block_sums.len();
        let required = unsafe { _whir_sumcheck_required_temp_buffer_size(height) };
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    check(_whir_sumcheck_mle_round(
        f_evals.as_ptr(),
        w_evals.as_ptr(),
        output.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
        height,
    ))
}

/// Special algebraic batching to update `w_evals` in place for WHIR.
///
/// # Safety
/// - `w_evals` has length `height`.
/// - `eq_z0` has length `height`.
/// - `eq_zs` has length `height * num_queries` and is a column major matrix of eq evaluations.
pub unsafe fn w_evals_accumulate(
    w_evals: &mut DeviceBuffer<EF>,
    eq_z0: &DeviceBuffer<EF>,
    eq_zs: &DeviceBuffer<F>,
    gamma: EF,
    num_queries: u32,
) -> Result<(), CudaError> {
    let height = w_evals.len();
    debug_assert_eq!(eq_zs.len(), height * num_queries as usize);
    debug_assert_eq!(eq_z0.len(), height);
    check(_w_evals_accumulate(
        w_evals.as_mut_ptr(),
        eq_z0.as_ptr(),
        eq_zs.as_ptr(),
        gamma,
        num_queries,
        height as u32,
    ))
}
