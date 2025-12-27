use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{CudaError, check},
};

use crate::{
    D_EF, EF, F,
    poly::EqEvalSegments,
    stacked_reduction::{
        Round0UniPacket, STACKED_REDUCTION_S_DEG, UnstackedPleFoldPacket, UnstackedSlice,
    },
};

extern "C" {
    pub fn _stacked_reduction_r0_required_temp_buffer_size(
        domain_size: u32,
        num_x: u32,
        thread_window_stride: u16,
    ) -> u32;

    fn _stacked_reduction_sumcheck_round0(
        eq_r_ns: *const EF,
        trace_ptr: *const F,
        lambda_pows: *const EF,
        z_packets: *const Round0UniPacket,
        omega_skip_pows: *const F,
        inv_lagrange_denoms: *const F,
        block_sums: *mut EF,
        output: *mut EF,
        height: u32,
        width: u32,
        log_domain_size: u32,
        l_skip: u32,
        num_x: u32,
        thread_window_stride: u16,
    ) -> i32;

    fn _stacked_reduction_fold_ple(
        trace_packets: *const UnstackedPleFoldPacket,
        omega_skip_pows: *const F,
        inv_lagrange_denoms: *const EF,
        l_skip: u32,
        num_packets: u16,
        max_new_height: u32,
        max_trace_width: u16,
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

/// Recall overall that we want to compute `Z -> sum_{H_{n_lift}} f(Z, \vec x)` for a specific `f`
/// on a DFT domain of size `domain_size` with `num_x = 2^{n_lift}`. The exact formula for `f` is
/// obtained by multiplying some `eq, \kappa_\rot` terms with prismalinear polynomial evaluations
/// `t_j(Z, \vec x)`. We need to upsample the `t_j(Z, \vec x)` evals on `(Z-domain) x hypercube` on
/// demand, which we use buffer slices `q_ptr[commit_idx]` to do.
///
/// We consider a "window" of `t_j`, with their slice locations given by `unstacked_cols` pointing
/// to the backing buffers `q_ptr`. Overall this means that we have a 3-dimensional array of
/// `t_{window_idx}(Z, x_int)` and we will match it to the 3-dimensional block/grid in CUDA. For
/// memory contiguity, we arrange so that
/// - cuda x-dim <-> Z coordinate
/// - cuda y-dim <-> x_int coordinate
/// - cuda z-dim <-> window_idx coordinate
///
/// We fix 1 z-thread per block, and the hard-code fixed constants for the x-threads per block and
/// y-threads per block. Now recall we want to sum over `x_int` and also over `window_idx` where we
/// algebraically batch over `window_idx` using `lambda_pows`. This means we must sum over cuda
/// y-dim and cuda z-dim but **not** over cuda x-dim.
///
/// The reduction is done in two stages:
/// 1. Block-level reduction to `block_sums` (shared memory within block)
/// 2. Final reduction kernel to combine block sums into `output`
///
/// Lastly, due to limitations on the CUDA grid dimensions, a single thread may need to sum over
/// multiple `x_int` and `window_idx` coordinates. This is achieved by separately striding in
/// `x_int` and `window_idx`. The `x_int` stride is configured automatically to `blockDim.y *
/// gridDim.y` to be the maximum possible. The `window_idx` stride is specified by the user input
/// parameter `thread_window_stride`.
///
/// # Safety
/// - `trace_ptr` must be a pointer to a device buffer valid for `height * width` elements.
/// - `lambda_pows` should be pointer to DeviceBuffer<EF>, at an offset.
///   - `lambda_pows` must be initialized for at least `2 * window_len` elements (for rotations).
/// - `z_packets` should be of length `2^log_domain_size`.
/// - `block_sums` should have length `>= _stacked_reduction_r0_required_temp_buffer_size(...)`.
/// - `output` should have length `>= domain_size`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn stacked_reduction_sumcheck_round0(
    eq_r_ns: &EqEvalSegments<EF>,
    trace_ptr: *const F,
    lambda_pows: *const EF,
    z_packets: &DeviceBuffer<Round0UniPacket>,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<F>,
    block_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    height: usize,
    width: usize,
    log_domain_size: usize,
    l_skip: usize,
    thread_window_stride: u16,
) -> Result<(), CudaError> {
    let domain_size = 1u32 << log_domain_size;
    let num_x = (height >> l_skip).max(1) as u32;
    debug_assert_eq!(z_packets.len(), domain_size as usize);
    debug_assert!(output.len() >= domain_size as usize);

    check(_stacked_reduction_sumcheck_round0(
        eq_r_ns.buffer().as_ptr(),
        trace_ptr,
        lambda_pows,
        z_packets.as_ptr(),
        omega_skip_pows.as_ptr(),
        inv_lagrange_denoms.as_ptr(),
        block_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        height as u32,
        width as u32,
        log_domain_size as u32,
        l_skip as u32,
        num_x,
        thread_window_stride,
    ))
}

/// # Safety
/// - The `src` pointers in `trace_packets` must be device pointers valid for `height x width` trace
///   matrices.
/// - The `dst` pointers in `trace_packets` must be device pointers valid for `new_height x width`
///   where `new_height = max(height, skip_domin) / skip_domin` for `skip_domain = 2^l_skip`.
///   Moreover these slices must be **disjoint**.
/// - The max trace width and length of `trace_packets` must not exceed `u16::MAX` due to kernel
///   grid sizes. This limitation can be removed with a kernel modification.
pub unsafe fn stacked_reduction_fold_ple(
    trace_packets: &DeviceBuffer<UnstackedPleFoldPacket>,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<EF>,
    l_skip: usize,
    max_new_height: usize,
    max_trace_width: usize,
) -> Result<(), CudaError> {
    let num_packets: u16 = trace_packets
        .len()
        .try_into()
        .expect("number of unstacked trace matrices must not exceed u16::MAX");
    debug_assert_eq!(inv_lagrange_denoms.len(), 1 << l_skip);
    let max_trace_width: u16 = max_trace_width
        .try_into()
        .expect("trace width must not exceed u16::MAX");
    check(_stacked_reduction_fold_ple(
        trace_packets.as_ptr(),
        omega_skip_pows.as_ptr(),
        inv_lagrange_denoms.as_ptr(),
        l_skip as u32,
        num_packets,
        max_new_height as u32,
        max_trace_width,
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
