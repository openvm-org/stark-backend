use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{check, CudaError},
};

use crate::{
    poly::EqEvalSegments,
    prelude::{D_EF, EF, F},
    stacked_reduction::{UnstackedSlice, STACKED_REDUCTION_S_DEG},
};

/// Number of G outputs per z in round 0: G0, G1, G2
pub const NUM_G: usize = 3;

extern "C" {
    pub fn _stacked_reduction_r0_required_temp_buffer_size(
        trace_height: u32,
        trace_width: u32,
        l_skip: u32,
    ) -> u32;

    // SP_DEG=1: no z_packets needed, outputs [NUM_G * skip_domain] to be ADDed
    fn _stacked_reduction_sumcheck_round0(
        eq_r_ns: *const EF,
        trace_ptr: *const F,
        lambda_pows: *const EF,
        block_sums: *mut EF,
        output: *mut EF,
        trace_height: u32,
        trace_width: u32,
        l_skip: u32,
        num_x: u32,
    ) -> i32;

    fn _stacked_reduction_fold_ple(
        src: *const F,
        dst: *mut EF,
        omega_skip_pows: *const F,
        inv_lagrange_denoms: *const EF,
        trace_height: u32,
        trace_width: u32,
        l_skip: u32,
    ) -> i32;

    fn _initialize_k_rot_from_eq_segments(
        eq_r_ns: *const EF,
        k_rot_ns: *mut EF,
        k_rot_uni_0: EF,
        k_rot_uni_1: EF,
        max_n: u32,
    ) -> i32;

    fn _stacked_reduction_sumcheck_mle_round(
        q_evals: *const *const EF,
        eq_r_ns: *const EF,
        k_rot_ns: *const EF,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const EF,
        output: *mut u64, // atomic accumulator [S_DEG * D_EF]
        q_height: u32,
        window_len: u32,
        num_y: u32,
        sm_count: u32,
    ) -> i32;

    fn _stacked_reduction_sumcheck_mle_round_degenerate(
        q_evals: *const *const EF,
        eq_ub_ptr: *const EF,
        eq_r: EF,
        k_rot_r: EF,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const EF,
        output: *mut u64, // atomic accumulator [S_DEG * D_EF]
        q_height: u32,
        window_len: u32,
        l_skip: u32,
        round: u32,
    ) -> i32;
}

/// SP_DEG=1 round 0 kernel: computes G0, G1, G2 partial sums on identity coset only.
///
/// This kernel computes three partial sums per z-coordinate:
/// - G0(Z) = Σ_{col,x} coeff_eq[col] * eq_cube(x) * q_{col,x}(Z)
/// - G1(Z) = Σ_{col,x} coeff_rot[col] * eq_cube(x) * q_{col,x}(Z)
/// - G2(Z) = Σ_{col,x} coeff_rot[col] * (eq_cube(rot_prev(x)) - eq_cube(x)) * q_{col,x}(Z)
///
/// The reconstruction s₀(Z) = E0(Z)*G0(Z) + E1(Z)*G1(Z) + E2(Z)*G2(Z) happens on CPU.
///
/// The reduction is done in two stages:
/// 1. Block-level reduction to `block_sums` (shared memory within block)
/// 2. Final reduction kernel that ADDs block sums into `output` (for bucket accumulation)
///
/// # Safety
/// - `trace_ptr` must be a pointer to a device buffer valid for `height * width` elements.
/// - `lambda_pows` should be pointer to DeviceBuffer<EF>, at an offset.
///   - `lambda_pows` must be initialized for at least `2 * window_len` elements (for rotations).
/// - `block_sums` should have length `>= _stacked_reduction_r0_required_temp_buffer_size(...)`.
/// - `output` should have length `>= NUM_G * 2^l_skip` and be zero-initialized before first call.
///   Values are ADDED to output for bucket-based accumulation.
#[allow(clippy::too_many_arguments)]
pub unsafe fn stacked_reduction_sumcheck_round0(
    eq_r_ns: &EqEvalSegments<EF>,
    trace_ptr: *const F,
    lambda_pows: *const EF,
    block_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    height: usize,
    width: usize,
    l_skip: usize,
) -> Result<(), CudaError> {
    let output_size = NUM_G << l_skip;
    let num_x = (height >> l_skip).max(1) as u32;
    debug_assert!(output.len() >= output_size);

    check(_stacked_reduction_sumcheck_round0(
        eq_r_ns.buffer().as_ptr(),
        trace_ptr,
        lambda_pows,
        block_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        height as u32,
        width as u32,
        l_skip as u32,
        num_x,
    ))
}

/// Parallelizes barycentric interpolation across `2^l_skip` threads per cell.
///
/// Each kernel launch handles one trace. The caller should loop over traces and call this for each.
///
/// # Safety
/// - `src` must be a device pointer valid for `trace_height * trace_width` Fp elements.
/// - `dst` must be a device pointer valid for `new_height * trace_width` FpExt elements, where
///   `new_height = max(trace_height, 2^l_skip) / 2^l_skip`.
/// - `src` and `dst` memory regions must not overlap.
/// - `omega_skip_pows` must have length `>= 2^l_skip`.
/// - `inv_lagrange_denoms` must have length `>= 2^l_skip`.
/// - `l_skip` must be in range [0, 10] (skip_domain = 2^l_skip must be in [1, 1024]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn stacked_reduction_fold_ple(
    src: *const F,
    dst: *mut EF,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<EF>,
    trace_height: usize,
    trace_width: usize,
    l_skip: usize,
) -> Result<(), CudaError> {
    let skip_domain = 1 << l_skip;
    debug_assert!(omega_skip_pows.len() >= skip_domain);
    debug_assert!(inv_lagrange_denoms.len() >= skip_domain);
    check(_stacked_reduction_fold_ple(
        src,
        dst,
        omega_skip_pows.as_ptr(),
        inv_lagrange_denoms.as_ptr(),
        trace_height as u32,
        trace_width as u32,
        l_skip as u32,
    ))
}

/// Initializes the `k_rot` evaluations after round 0 for `n = 0..=max_n`.
///
/// # Safety
/// - `eq_r_ns` and `k_rot_ns` should both have length `2 * 2^max_n` and be arranged in segments.
/// - This function will not initialize `k_rot_ns[0]`. This entry should never be used, and it is
///   the caller's responsibility to initialize it if needed.
pub unsafe fn initialize_k_rot_from_eq_segments(
    eq_r_ns: &EqEvalSegments<EF>,
    k_rot_ns: &mut DeviceBuffer<EF>,
    k_rot_uni_0: EF,
    k_rot_uni_1: EF,
    max_n: u32,
) -> Result<(), CudaError> {
    debug_assert_eq!(eq_r_ns.buffer.len(), 2 << max_n);
    debug_assert_eq!(k_rot_ns.len(), eq_r_ns.buffer.len());

    check(_initialize_k_rot_from_eq_segments(
        eq_r_ns.buffer.as_ptr(),
        k_rot_ns.as_mut_ptr(),
        k_rot_uni_0,
        k_rot_uni_1,
        max_n,
    ))
}

/// Analog of [stacked_reduction_sumcheck_round0] for MLE sumcheck rounds.
/// One important distinction is that `k_rot_ns` needs to be provided separately from `eq_r_ns`, but
/// in the same segmented memory layout.
///
/// Uses warp-aggregated atomics for reduction into u64 accumulators. The caller is responsible for
/// zero-initializing `output` before calling and reducing modulo P after kernel completion.
///
/// # Safety
/// - `q_evals` must be pointers to DeviceBuffer<EF> matrices all of the same height `q_height`.
/// - `unstacked_cols` should be pointer to DeviceBuffer<UnstackedSlice>, at an offset.
///   - `unstacked_cols` must be initialized for at least `window_len` elements.
///   - for each `UnstackedSlice` in the `window_len` elements starting at `unstacked_cols`, the
///     `log_height` must all be the same and the `stacked_row_idx` must be a multiple of
///     `2^l_skip`.
/// - `lambda_pows` should be pointer to DeviceBuffer<EF>, at an offset.
///   - `lambda_pows` must be initialized for at least `2 * window_len` elements (for rotations).
/// - `unstacked_cols` pointers must be within bounds of `q_evals`.
/// - `output` must have length at least `S_DEG * D_EF = 8` and be zero-initialized.
#[allow(clippy::too_many_arguments)]
pub unsafe fn stacked_reduction_sumcheck_mle_round(
    q_evals: &DeviceBuffer<*const EF>,
    eq_r_ns: &EqEvalSegments<EF>,
    k_rot_ns: &EqEvalSegments<EF>,
    unstacked_cols: *const UnstackedSlice,
    lambda_pows: *const EF,
    output: &mut DeviceBuffer<u64>,
    q_height: usize,
    window_len: usize,
    num_y: usize,
    sm_count: u32,
) -> Result<(), CudaError> {
    debug_assert!(output.len() >= STACKED_REDUCTION_S_DEG * D_EF);

    check(_stacked_reduction_sumcheck_mle_round(
        q_evals.as_ptr(),
        eq_r_ns.buffer.as_ptr(),
        k_rot_ns.buffer.as_ptr(),
        unstacked_cols,
        lambda_pows,
        output.as_mut_ptr(),
        q_height as u32,
        window_len as u32,
        num_y as u32,
        sm_count,
    ))
}

/// Degenerate case for MLE sumcheck rounds.
///
/// Uses warp-aggregated atomics for reduction into u64 accumulators. The caller is responsible for
/// zero-initializing `output` before calling and reducing modulo P after kernel completion.
///
/// # Safety
/// - `output` must have length at least `S_DEG * D_EF = 8` and be zero-initialized.
#[allow(clippy::too_many_arguments)]
pub unsafe fn stacked_reduction_sumcheck_mle_round_degenerate(
    q_evals: &DeviceBuffer<*const EF>,
    eq_ub_ptr: &DeviceBuffer<EF>,
    eq_r: EF,
    k_rot_r: EF,
    unstacked_cols: *const UnstackedSlice,
    lambda_pows: *const EF,
    output: &mut DeviceBuffer<u64>,
    q_height: usize,
    window_len: usize,
    l_skip: usize,
    round: usize,
) -> Result<(), CudaError> {
    debug_assert!(output.len() >= STACKED_REDUCTION_S_DEG * D_EF);

    check(_stacked_reduction_sumcheck_mle_round_degenerate(
        q_evals.as_ptr(),
        eq_ub_ptr.as_ptr(),
        eq_r,
        k_rot_r,
        unstacked_cols,
        lambda_pows,
        output.as_mut_ptr(),
        q_height as u32,
        window_len as u32,
        l_skip as u32,
        round as u32,
    ))
}
