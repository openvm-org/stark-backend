use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{CudaError, check},
};

use crate::{
    EF, F,
    poly::EqEvalSegments,
    stacked_reduction::{Round0UniPacket, UnstackedSlice},
};

extern "C" {
    pub fn _stacked_reduction_r0_required_temp_buffer_size(
        domain_size: u32,
        num_x: u32,
        thread_window_stride: u16,
    ) -> u32;

    fn _stacked_reduction_sumcheck_round0(
        q_upsampled_ptr: *const *const F,
        eq_r_ns: *const EF,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const EF,
        z_packets: *const Round0UniPacket,
        block_sums: *mut EF,
        output: *mut EF,
        upsampled_height: u32,
        log_domain_size: u32,
        l_skip: u32,
        window_len: u32,
        num_x: u32,
        thread_window_stride: u16,
    ) -> i32;

    fn _initialize_k_rot_from_eq_segments(
        eq_r_ns: *const EF,
        k_rot_ns: *mut EF,
        k_rot_uni_0: EF,
        k_rot_uni_1: EF,
        max_n: u32,
    ) -> i32;

    pub fn _stacked_reduction_mle_required_temp_buffer_size(
        num_y: u32,
        thread_window_stride: u16,
    ) -> u32;

    fn _stacked_reduction_sumcheck_mle_round(
        q_evals: *const *const EF,
        eq_r_ns: *const EF,
        k_rot_ns: *const EF,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const EF,
        block_sums: *mut EF,
        output: *mut EF,
        q_height: u32,
        window_len: u32,
        num_y: u32,
        thread_window_stride: u16,
    ) -> i32;

    fn _stacked_reduction_sumcheck_mle_round_degenerate(
        q_evals: *const *const EF,
        eq_ub_ptr: *const EF,
        eq_r: EF,
        k_rot_r: EF,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const EF,
        output: *mut EF,
        q_height: u32,
        window_len: u32,
        l_skip: u32,
        round: u32,
    ) -> i32;
}

/// Recall overall that we want to compute `Z -> sum_{H_{n_lift}} f(Z, \vec x)` for a specific `f`
/// on a DFT domain of size `domain_size` with `num_x = 2^{n_lift}`. The exact formula for `f` is
/// obtained by multiplying some `eq, \kappa_\rot` terms with prismalinear polynomial evaluations
/// `t_j(Z, \vec x)`. The setup is that we already have the "upsampled" `t_j(Z, \vec x)` evaluations
/// on `(Z-domain) x hypercube`, but these are represented as slices within the buffers
/// `q_upsampled_ptr[commit_idx]`.
///
/// We consider a "window" of `t_j`, with their slice locations given by `unstacked_cols` pointing
/// to the backing buffers `q_upsampled_ptr`. Overall this means that we have a 3-dimensional array
/// of `t_{window_idx}(Z, x_int)` and we will match it to the 3-dimensional block/grid in CUDA. For
/// memory contiguity, we arrange so that
/// - cuda x-dim <-> Z coordinate
/// - cuda y-dim <-> x_int coordinate
/// - cuda z-dim <-> window_idx coordinate
///
/// We fix 1 z-thread per block, and the hard-code fixed constants for the x-threads per block and
/// y-threads per block. Now recall we want to sum over `x_int` and also over `window_idx` where we
/// algebraically batch over `window_idx` using `lambda_pows`. This means we must sum over cuda
/// y-dim and cuda z-dim but **not** over cuda x-dim. To do so, we use a temporary buffer
/// `block_sums`, whose required size can be computed via
/// [_stacked_reduction_r0_required_temp_buffer_size]. To avoid excessive memory usage for
/// `block_sums`, we use shared memory to perform block reduction by summing over all threadIdx.y
/// within a block (without summing over threadIdx.x).
///
/// Lastly, due to limitations on the CUDA grid dimensions, a single thread may need to sum over
/// multiple `x_int` and `window_idx` coordinates. This is achieved by separately striding in
/// `x_int` and `window_idx`. The `x_int` stride is configured automatically to `blockDim.y *
/// gridDim.y` to be the maximum possible. The `window_idx` stride is specified by the user input
/// parameter `thread_window_stride`.
///
/// The `thread_window_stride` can be adjusted: smaller means more work per thread but smaller
/// required `block_sums` size. Larger means more parallelism but requires more `block_sums` size.
/// It must fit in `u16` for CUDA grid dimension limits.
///
/// # Safety
/// - Each pointer in `q_upsampled_ptr` must be to a device buffer of length `upsampled_height`.
/// - `unstacked_cols` should be pointer to DeviceBuffer<UnstackedSlice>, at an offset.
///   - `unstacked_cols` must be initialized for at least `window_len` elements.
///   - for each `UnstackedSlice` in the `window_len` elements starting at `unstacked_cols`, the
///     `log_height` must all be the same and the `stacked_row_idx` must be a multiple of
///     `2^l_skip`.
/// - `lambda_pows` should be pointer to DeviceBuffer<EF>, at an offset.
///   - `lambda_pows` must be initialized for at least `2 * window_len` elements (for rotations).
/// - `z_packets` should be of length `2^log_domain_size`.
/// - `block_sums` should have capacity at least
///   [`_stacked_reduction_r0_required_temp_buffer_size(domain_size, num_x,
///   thread_window_stride)`](_stacked_reduction_r0_required_temp_buffer_size).
/// - `output` should have length `>= 2^log_domain_size`.
pub unsafe fn stacked_reduction_sumcheck_round0(
    q_upsampled_ptr: &DeviceBuffer<*const F>,
    eq_r_ns: &EqEvalSegments<EF>,
    unstacked_cols: *const UnstackedSlice,
    lambda_pows: *const EF,
    z_packets: &DeviceBuffer<Round0UniPacket>,
    block_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    upsampled_height: usize,
    log_domain_size: usize,
    l_skip: usize,
    window_len: usize,
    num_x: usize,
    thread_window_stride: u16,
) -> Result<(), CudaError> {
    let domain_size = 1u32 << log_domain_size;
    let num_x = num_x as u32;
    debug_assert_eq!(z_packets.len(), domain_size as usize);
    debug_assert!(output.len() >= domain_size as usize);
    #[cfg(debug_assertions)]
    {
        let required_size = _stacked_reduction_r0_required_temp_buffer_size(
            domain_size,
            num_x,
            thread_window_stride,
        );
        assert!(
            block_sums.len() >= required_size as usize,
            "block_sums.len() = {}, required = {}",
            block_sums.len(),
            required_size
        );
    }
    check(_stacked_reduction_sumcheck_round0(
        q_upsampled_ptr.as_ptr(),
        eq_r_ns.buffer().as_ptr(),
        unstacked_cols,
        lambda_pows,
        z_packets.as_ptr(),
        block_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        upsampled_height as u32,
        log_domain_size as u32,
        l_skip as u32,
        window_len as u32,
        num_x,
        thread_window_stride,
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
/// # Safety
/// - `q_evals` must be pointers to DeviceBuffer<EF> matrices all of the same height `q_height`.
/// - `block_sums` needs enough size as specified by
///   [`_stacked_reduction_mle_required_temp_buffer_size`].
/// - `unstacked_cols` should be pointer to DeviceBuffer<UnstackedSlice>, at an offset.
///   - `unstacked_cols` must be initialized for at least `window_len` elements.
///   - for each `UnstackedSlice` in the `window_len` elements starting at `unstacked_cols`, the
///     `log_height` must all be the same and the `stacked_row_idx` must be a multiple of
///     `2^l_skip`.
/// - `lambda_pows` should be pointer to DeviceBuffer<EF>, at an offset.
///   - `lambda_pows` must be initialized for at least `2 * window_len` elements (for rotations).
/// - `unstacked_cols` pointers must be within bounds of `q_evals`.
/// - `output` must have length at least `s_deg = 2`.
pub unsafe fn stacked_reduction_sumcheck_mle_round(
    q_evals: &DeviceBuffer<*const EF>,
    eq_r_ns: &EqEvalSegments<EF>,
    k_rot_ns: &EqEvalSegments<EF>,
    unstacked_cols: *const UnstackedSlice,
    lambda_pows: *const EF,
    block_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    q_height: usize,
    window_len: usize,
    num_y: usize,
    thread_window_stride: u16,
) -> Result<(), CudaError> {
    debug_assert!(output.len() >= 2);
    #[cfg(debug_assertions)]
    {
        let required_size =
            _stacked_reduction_mle_required_temp_buffer_size(num_y as u32, thread_window_stride);
        let len = block_sums.len();
        assert!(
            len >= required_size as usize,
            "block_sums.len() = {len}, required = {required_size}"
        );
    }

    check(_stacked_reduction_sumcheck_mle_round(
        q_evals.as_ptr(),
        eq_r_ns.buffer.as_ptr(),
        k_rot_ns.buffer.as_ptr(),
        unstacked_cols,
        lambda_pows,
        block_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        q_height as u32,
        window_len as u32,
        num_y as u32,
        thread_window_stride,
    ))
}

pub unsafe fn stacked_reduction_sumcheck_mle_round_degenerate(
    q_evals: &DeviceBuffer<*const EF>,
    eq_ub_ptr: &DeviceBuffer<EF>,
    eq_r: EF,
    k_rot_r: EF,
    unstacked_cols: *const UnstackedSlice,
    lambda_pows: *const EF,
    output: &mut DeviceBuffer<EF>,
    q_height: usize,
    window_len: usize,
    l_skip: usize,
    round: usize,
) -> Result<(), CudaError> {
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
