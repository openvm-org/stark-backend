use p3_field::{Field, FieldAlgebra};

use super::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MainMatrixPtrs<T> {
    pub data: *const T,
    pub air_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockCtx {
    pub local_block_idx_x: u32,
    pub air_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EvalCtx {
    pub d_selectors: *const EF,
    pub d_preprocessed: MainMatrixPtrs<EF>,
    pub d_main: *const MainMatrixPtrs<EF>,
    pub d_public: *const F,
    pub d_intermediates: *mut EF,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZerocheckCtx {
    pub eval_ctx: EvalCtx,
    pub num_y: u32,
    pub d_eq_xi: *const EF,
    pub d_rules: *const std::ffi::c_void,
    pub rules_len: usize,
    pub d_used_nodes: *const usize,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupCtx {
    pub eval_ctx: EvalCtx,
    pub num_y: u32,
    pub d_eq_sharp: *const EF,
    pub d_challenges: *const EF,
    pub d_eq_3bs: *const EF,
    pub d_rules: *const std::ffi::c_void,
    pub rules_len: usize,
    pub d_used_nodes: *const usize,
    pub d_pair_idxs: *const u32,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

extern "C" {
    // gkr.cu
    fn _frac_build_tree_layer(layer: *mut Frac<EF>, layer_size: usize, revert: bool) -> i32;

    pub fn _frac_compute_round_temp_buffer_size(stride: u32) -> u32;

    fn _frac_compute_round(
        eq_xi: *const EF,
        pq_buffer: *mut Frac<EF>,
        eq_size: usize,
        pq_size: usize,
        lambda: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
    ) -> i32;

    fn _frac_fold_columns(buffer: *mut std::ffi::c_void, size: usize, r: EF) -> i32;

    fn _frac_fold_fpext_columns(
        buffer: *mut Frac<EF>,
        size: usize,
        r_or_r_inv: EF,
        revert: bool,
    ) -> i32;

    fn _frac_add_alpha(data: *mut std::ffi::c_void, len: usize, alpha: EF) -> i32;

    fn _frac_vector_scalar_multiply_ext_fp(frac_vec: *mut Frac<EF>, scalar: F, length: u32) -> i32;

    // utils.cu
    fn _fold_ple_from_evals(
        input_matrix: *const F,
        output_matrix: *mut EF,
        omega_skip_pows: *const F,
        inv_lagrange_denoms: *const EF,
        height: u32,
        width: u32,
        l_skip: u32,
        new_height: u32,
        rotate: bool,
    ) -> i32;

    fn _interpolate_columns(
        interpolated: *mut EF,
        columns: *const *const EF,
        s_deg: usize,
        num_y: usize,
        num_columns: usize,
    ) -> i32;

    fn _compute_eq_sharp(
        eq_xi: *mut std::ffi::c_void,
        eq_sharp: *mut std::ffi::c_void,
        eq_r0: EF,
        eq_sharp_r0: EF,
        count: u32,
    ) -> i32;

    fn _frac_matrix_vertically_repeat(
        out: *mut Frac<EF>,
        input: *const Frac<EF>,
        width: u32,
        lifted_height: u32,
        height: u32,
    ) -> i32;

    fn _frac_matrix_vertically_repeat_ext(
        out_numerators: *mut EF,
        out_denominators: *mut EF,
        in_numerators: *const EF,
        in_denominators: *const EF,
        width: u32,
        lifted_height: u32,
        height: u32,
    ) -> i32;

    // gkr_input.cu
    fn _logup_gkr_input_eval(
        is_global: bool,
        fracs: *mut Frac<EF>,
        preprocessed: *const std::ffi::c_void,
        partitioned_main: *const u64,
        challenges: *const std::ffi::c_void,
        intermediates: *const std::ffi::c_void,
        rules: *const std::ffi::c_void,
        used_nodes: *const usize,
        pair_idxs: *const u32,
        used_nodes_len: usize,
        height: u32,
        num_rows_per_tile: u32,
    ) -> i32;

    // logup_round0.cu
    pub fn _logup_r0_temp_sums_buffer_size(
        buffer_size: u32,
        large_domain: u32,
        num_x: u32,
        max_temp_bytes: usize,
    ) -> usize;

    pub fn _logup_r0_intermediates_buffer_size(
        buffer_size: u32,
        large_domain: u32,
        num_x: u32,
        max_temp_bytes: usize,
    ) -> usize;

    fn _logup_bary_eval_interactions_round0(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        selectors_cube: *const F,
        preprocessed: *const F,
        main_parts: *const *const F,
        omega_skip_pows: *const F,
        inv_lagrange_denoms: *const F,
        eq_sharp_uni: *const EF,
        eq_cube: *const EF,
        public_values: *const F,
        numer_weights: *const EF,
        denom_weights: *const EF,
        denom_sum_init: EF,
        d_rules: *const std::ffi::c_void,
        rules_len: usize,
        buffer_size: u32,
        d_intermediates: *mut F,
        large_domain: u32,
        skip_domain: u32,
        num_x: u32,
        height: u32,
        expansion_factor: u32,
        max_temp_bytes: usize,
    ) -> i32;

    // zerocheck_round0.cu
    pub fn _zerocheck_r0_temp_sums_buffer_size(
        buffer_size: u32,
        large_domain: u32,
        num_x: u32,
        max_temp_bytes: usize,
    ) -> usize;

    pub fn _zerocheck_r0_intermediates_buffer_size(
        buffer_size: u32,
        large_domain: u32,
        num_x: u32,
        max_temp_bytes: usize,
    ) -> usize;

    fn _zerocheck_bary_eval_constraints(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        selectors_cube: *const F,
        preprocessed: *const F,
        main_parts: *const *const F,
        omega_skip_pows: *const F,
        inv_lagrange_denoms: *const F,
        eq_uni: *const EF,
        eq_cube: *const EF,
        d_lambda_pows: *const EF,
        public_values: *const F,
        d_rules: *const std::ffi::c_void,
        rules_len: usize,
        d_used_nodes: *const usize,
        used_nodes_len: usize,
        lambda_len: usize,
        buffer_size: u32,
        d_intermediates: *mut F,
        large_domain: u32,
        skip_domain: u32,
        num_x: u32,
        height: u32,
        expansion_factor: u32,
        max_temp_bytes: usize,
    ) -> i32;

    fn _fold_selectors_round0(
        out: *mut EF,
        input: *const F,
        is_first: EF,
        is_last: EF,
        num_x: u32,
    ) -> i32;

    // mle.cu
    pub fn _zerocheck_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize;

    pub fn _zerocheck_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    pub fn _mle_eval_num_blocks(num_x: u32, num_y: u32) -> u32;

    fn _zerocheck_eval_mle(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        eq_xi: *const EF,
        selectors: *const EF,
        preprocessed: MainMatrixPtrs<EF>,
        main: *const MainMatrixPtrs<EF>,
        lambda_pows: *const EF,
        public_values: *const F,
        rules: *const std::ffi::c_void,
        rules_len: usize,
        used_nodes: *const usize,
        used_nodes_len: usize,
        lambda_len: usize,
        buffer_size: u32,
        intermediates: *mut EF,
        num_y: u32,
        num_x: u32,
    ) -> i32;

    fn _zerocheck_batch_eval_mle(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        zc_ctxs: *const ZerocheckCtx,
        air_block_offsets: *const u32,
        lambda_pows: *const EF,
        lambda_len: usize,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
    ) -> i32;

    pub fn _logup_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize;

    pub fn _logup_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize;

    // batch_mle.cu (batch kernels always use global intermediates when buffer_size > 0)
    pub fn _zerocheck_batch_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;
    pub fn _logup_batch_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    fn _logup_eval_mle(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        eq_sharp: *const EF,
        selectors: *const EF,
        preprocessed: MainMatrixPtrs<EF>,
        main: *const MainMatrixPtrs<EF>,
        challenges: *const EF,
        eq_3bs: *const EF,
        public_values: *const F,
        rules: *const std::ffi::c_void,
        used_nodes: *const usize,
        pair_idxs: *const u32,
        used_nodes_len: usize,
        buffer_size: u32,
        intermediates: *mut EF,
        num_y: u32,
        num_x: u32,
    ) -> i32;

    fn _logup_batch_eval_mle(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        block_ctxs: *const BlockCtx,
        logup_ctxs: *const LogupCtx,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
    ) -> i32;
}

pub unsafe fn interpolate_columns_gpu(
    interpolated: &DeviceBuffer<EF>,
    columns: &DeviceBuffer<*const EF>,
    s_deg: usize,
    num_y: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_interpolate_columns(
        interpolated.as_mut_ptr(),
        columns.as_ptr(),
        s_deg,
        num_y,
        columns.len(),
    ))
}

pub unsafe fn frac_build_tree_layer(
    layer: &mut DeviceBuffer<Frac<EF>>,
    layer_size: usize,
    revert: bool,
) -> Result<(), CudaError> {
    debug_assert!(layer.len() >= layer_size);
    CudaError::from_result(_frac_build_tree_layer(
        layer.as_mut_ptr(),
        layer_size,
        revert,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round(
    eq_xi: &DeviceBuffer<EF>,
    pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    eq_size: usize,
    pq_size: usize,
    lambda: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    #[cfg(debug_assertions)]
    {
        let len = tmp_block_sums.len();
        let required = _frac_compute_round_temp_buffer_size(eq_size as u32);
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    CudaError::from_result(_frac_compute_round(
        eq_xi.as_ptr(),
        pq_buffer.as_mut_ptr(),
        eq_size,
        pq_size,
        lambda,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
    ))
}

pub unsafe fn frac_fold_columns(
    buffer: &mut DeviceBuffer<EF>,
    size: usize,
    r: EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_fold_columns(buffer.as_mut_raw_ptr(), size, r))
}

/// Folds matrix of `Frac<EF>` but treats `input` and `output` as **row-major** matrices in
/// `Frac<EF>`. The numerator and denominator are folded pair-wise.
pub unsafe fn fold_ef_frac_columns(
    buffer: &mut DeviceBuffer<Frac<EF>>,
    size: usize,
    r: EF,
    revert: bool,
) -> Result<(), CudaError> {
    let r_or_r_inv = if revert {
        debug_assert!(r != EF::ONE);
        (EF::ONE - r).inverse()
    } else {
        r
    };
    CudaError::from_result(_frac_fold_fpext_columns(
        buffer.as_mut_ptr(),
        size,
        r_or_r_inv,
        revert,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn fold_ple_from_evals(
    input_matrix: &DeviceBuffer<F>,
    output_matrix: *mut EF,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<EF>,
    height: u32,
    width: u32,
    l_skip: u32,
    new_height: u32,
    rotate: bool,
) -> Result<(), CudaError> {
    CudaError::from_result(_fold_ple_from_evals(
        input_matrix.as_ptr(),
        output_matrix,
        omega_skip_pows.as_ptr(),
        inv_lagrange_denoms.as_ptr(),
        height,
        width,
        l_skip,
        new_height,
        rotate,
    ))
}

pub unsafe fn compute_eq_sharp(
    eq_xi: &DeviceBuffer<EF>,
    eq_sharp: &DeviceBuffer<EF>,
    eq_r0: EF,
    eq_sharp_r0: EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_compute_eq_sharp(
        eq_xi.as_mut_raw_ptr(),
        eq_sharp.as_mut_raw_ptr(),
        eq_r0,
        eq_sharp_r0,
        eq_xi.len() as u32,
    ))
}

/// # Safety
/// - `output` must be a pointer to a device buffer with capacity at least `num_partitions *
///   height`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_gkr_input_eval(
    is_global: bool,
    fracs: *mut Frac<EF>,
    preprocessed: &DeviceBuffer<F>,
    partitioned_main: &DeviceBuffer<u64>,
    challenges: &DeviceBuffer<EF>,
    intermediates: &DeviceBuffer<EF>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    pair_idxs: &DeviceBuffer<u32>,
    height: u32,
    num_rows_per_tile: u32,
) -> Result<(), CudaError> {
    debug_assert_eq!(used_nodes.len(), pair_idxs.len());
    CudaError::from_result(_logup_gkr_input_eval(
        is_global,
        fracs,
        preprocessed.as_raw_ptr(),
        partitioned_main.as_ptr(),
        challenges.as_raw_ptr(),
        intermediates.as_raw_ptr(),
        rules.as_raw_ptr(),
        used_nodes.as_ptr(),
        pair_idxs.as_ptr(),
        used_nodes.len(),
        height,
        num_rows_per_tile,
    ))
}

pub unsafe fn frac_add_alpha(data: &DeviceBuffer<Frac<EF>>, alpha: EF) -> Result<(), CudaError> {
    CudaError::from_result(_frac_add_alpha(data.as_mut_raw_ptr(), data.len(), alpha))
}

/// # Safety
/// - `buffer_size` does not refer to the capacity of `intermediates`. It refers to "how many DAG
///   nodes per row need to be buffered". The capacity is a multiple of `buffer_size` which is
///   runtime calculated based on `buffer_size`.
/// - `eq_cube` must be a pointer to device buffer with at least `num_x` elements representing
///   evaluations on hypercube.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_bary_eval_constraints(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    selectors_cube: &DeviceBuffer<F>,
    preprocessed: *const F,
    main_ptrs: &DeviceBuffer<*const F>,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<F>,
    eq_uni: &DeviceBuffer<EF>,
    eq_cube: *const EF,
    lambda_pows: &DeviceBuffer<EF>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<F>,
    large_domain: u32,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    max_temp_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_bary_eval_constraints(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        selectors_cube.as_ptr(),
        preprocessed,
        main_ptrs.as_ptr(),
        omega_skip_pows.as_ptr(),
        inv_lagrange_denoms.as_ptr(),
        eq_uni.as_ptr(),
        eq_cube,
        lambda_pows.as_ptr(),
        public_values.as_ptr(),
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        lambda_pows.len(),
        buffer_size,
        intermediates.as_mut_ptr(),
        large_domain,
        skip_domain,
        num_x,
        height,
        large_domain.next_power_of_two() / skip_domain,
        max_temp_bytes,
    ))
}

/// # Safety
/// - `buffer_size` does not refer to the capacity of `intermediates`. It refers to "how many DAG
///   nodes per row need to be buffered". The capacity is a multiple of `buffer_size` which is
///   runtime calculated based on `buffer_size`.
/// - `eq_cube` must be a pointer to device buffer with at least `num_x` elements representing
///   evaluations on hypercube.
/// - `output` will not be written to by this function. Only `tmp_sums_buffer` is written.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_bary_eval_interactions_round0(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    selectors_cube: &DeviceBuffer<F>,
    preprocessed: *const F,
    main_ptrs: &DeviceBuffer<*const F>,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<F>,
    eq_sharp_uni: &DeviceBuffer<EF>,
    eq_cube: *const EF,
    public_values: &DeviceBuffer<F>,
    numer_weights: &DeviceBuffer<EF>,
    denom_weights: &DeviceBuffer<EF>,
    denom_sum_init: EF,
    rules: &DeviceBuffer<u128>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<F>,
    large_domain: u32,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    max_temp_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_bary_eval_interactions_round0(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        selectors_cube.as_ptr(),
        preprocessed,
        main_ptrs.as_ptr(),
        omega_skip_pows.as_ptr(),
        inv_lagrange_denoms.as_ptr(),
        eq_sharp_uni.as_ptr(),
        eq_cube,
        public_values.as_ptr(),
        numer_weights.as_ptr(),
        denom_weights.as_ptr(),
        denom_sum_init,
        rules.as_raw_ptr(),
        rules.len(),
        buffer_size,
        intermediates.as_mut_ptr(),
        large_domain,
        skip_domain,
        num_x,
        height,
        large_domain.next_power_of_two() / skip_domain,
        max_temp_bytes,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_mle(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    eq_xi: *const EF,
    selectors: *const EF,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: &DeviceBuffer<MainMatrixPtrs<EF>>,
    lambda_pows: &DeviceBuffer<EF>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<EF>,
    num_y: u32,
    num_x: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        eq_xi,
        selectors,
        preprocessed,
        main_ptrs.as_ptr(),
        lambda_pows.as_ptr(),
        public_values.as_ptr(),
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        lambda_pows.len(),
        buffer_size,
        intermediates.as_mut_ptr(),
        num_y,
        num_x,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_batch_eval_mle(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    zc_ctxs: &DeviceBuffer<ZerocheckCtx>,
    air_block_offsets: &[u32],
    lambda_pows: &DeviceBuffer<EF>,
    lambda_len: usize,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_batch_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        zc_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        lambda_pows.as_ptr(),
        lambda_len,
        num_blocks,
        num_x,
        num_airs,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_eval_mle(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    eq_sharp: *const EF,
    selectors: *const EF,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: &DeviceBuffer<MainMatrixPtrs<EF>>,
    challenges: &DeviceBuffer<EF>,
    eq_3bs: &DeviceBuffer<EF>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    pair_idxs: &DeviceBuffer<u32>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<EF>,
    num_y: u32,
    num_x: u32,
) -> Result<(), CudaError> {
    debug_assert_eq!(used_nodes.len(), pair_idxs.len());
    CudaError::from_result(_logup_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        eq_sharp,
        selectors,
        preprocessed,
        main_ptrs.as_ptr(),
        challenges.as_ptr(),
        eq_3bs.as_ptr(),
        public_values.as_ptr(),
        rules.as_raw_ptr(),
        used_nodes.as_ptr(),
        pair_idxs.as_ptr(),
        used_nodes.len(),
        buffer_size,
        intermediates.as_mut_ptr(),
        num_y,
        num_x,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_batch_eval_mle(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    logup_ctxs: &DeviceBuffer<LogupCtx>,
    air_block_offsets: &[u32],
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_batch_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        logup_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
    ))
}

pub unsafe fn frac_vector_scalar_multiply_ext_fp(
    frac_vec: *mut Frac<EF>,
    scalar: F,
    length: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_vector_scalar_multiply_ext_fp(
        frac_vec, scalar, length,
    ))
}

/// Vertically repeats the rows of `input` and writes them to `out`. Both matrices are column-major.
///
/// # Safety
/// - `out` must be a pointer to `DeviceBuffer<F>` with length at least `lifted_height * width`.
/// - `input` must be a pointer to `DeviceBuffer<F>` with length at least `height * width`.
/// - `out` and `input` must not overlap.
pub unsafe fn frac_matrix_vertically_repeat(
    out: *mut Frac<EF>,
    input: *const Frac<EF>,
    width: u32,
    lifted_height: u32,
    height: u32,
) -> Result<(), CudaError> {
    debug_assert!(lifted_height > height);
    CudaError::from_result(_frac_matrix_vertically_repeat(
        out,
        input,
        width,
        lifted_height,
        height,
    ))
}

pub unsafe fn frac_matrix_vertically_repeat_ext(
    out_numerators: *mut EF,
    out_denominators: *mut EF,
    in_numerators: *const EF,
    in_denominators: *const EF,
    width: u32,
    lifted_height: u32,
    height: u32,
) -> Result<(), CudaError> {
    debug_assert!(lifted_height > height);
    CudaError::from_result(_frac_matrix_vertically_repeat_ext(
        out_numerators,
        out_denominators,
        in_numerators,
        in_denominators,
        width,
        lifted_height,
        height,
    ))
}

/// Create folded selectors around round 0 from hypercube evaluations and univariate factors.
///
/// Note: `is_transition` is not a product of univariate and hypercube factors.
pub unsafe fn fold_selectors_round0(
    out: *mut EF,
    input: *const F,
    is_first: EF,
    is_last: EF,
    num_x: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_fold_selectors_round0(
        out,
        input,
        is_first,
        is_last,
        num_x as u32,
    ))
}
