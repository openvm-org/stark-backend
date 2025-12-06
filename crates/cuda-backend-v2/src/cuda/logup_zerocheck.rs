use openvm_cuda_backend::base::DeviceMatrix;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{Field, FieldAlgebra};

use super::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MainMatrixPtrs<T> {
    pub data: *const T,
    pub air_width: u32,
}

extern "C" {
    // gkr.cu
    fn _frac_build_tree_layer(
        numerators: *mut EF,
        denominators: *mut EF,
        layer_size: usize,
        step: usize,
        revert: bool,
    ) -> i32;

    pub fn _frac_compute_round_temp_buffer_size(stride: u32) -> u32;

    fn _frac_compute_round(
        eq_xi: *const EF,
        pq_nums: *mut EF,
        pq_denoms: *mut EF,
        stride: usize,
        step: usize,
        pq_extra_step: usize,
        lambda: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
    ) -> i32;

    fn _frac_fold_columns(buffer: *mut std::ffi::c_void, stride: usize, step: usize, r: EF) -> i32;

    fn _frac_fold_ext_columns(
        buffer: *mut EF,
        stride: usize,
        step: usize,
        r_or_r_inv: EF,
        revert: bool,
    ) -> i32;

    fn _frac_extract_claims(
        data: *const std::ffi::c_void,
        stride: usize,
        out_device: *mut std::ffi::c_void,
    ) -> i32;

    fn _frac_add_alpha(data: *mut std::ffi::c_void, len: usize, alpha: EF) -> i32;

    fn _frac_vector_scalar_multiply_ext_fp(frac_vec: *mut Frac<EF>, scalar: F, length: u32) -> i32;

    fn _frac_add_alpha_mixed(denominators: *mut EF, len: usize, alpha: EF) -> i32;

    fn _frac_vector_scalar_multiply_ext(numerators: *mut EF, scalar: F, length: u32) -> i32;

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
        interpolated: *mut std::ffi::c_void,
        columns: *const usize,
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

    // interactions.cu
    fn _logup_gkr_input_eval(
        is_global: bool,
        numerators: *mut EF,
        denominators: *mut EF,
        preprocessed: *const std::ffi::c_void,
        partitioned_main: *const u64,
        challenges: *const std::ffi::c_void,
        intermediates: *const std::ffi::c_void,
        rules: *const std::ffi::c_void,
        used_nodes: *const usize,
        partition_lens: *const u32,
        num_partitions: usize,
        height: u32,
        num_rows_per_tile: u32,
    ) -> i32;
    fn _batch_constraints_eval_interactions_round0(
        output_numer: *mut std::ffi::c_void,
        output_denom: *mut std::ffi::c_void,
        selectors: *const std::ffi::c_void,
        selectors_width: u32,
        partitioned_main: *const MainMatrixPtrs<F>,
        main_count: u32,
        preprocessed: *const F,
        preprocessed_air_width: u32,
        eq_z: *const std::ffi::c_void,
        eq_x: *const std::ffi::c_void,
        eq_3b: *const std::ffi::c_void,
        public_values: *const std::ffi::c_void,
        public_len: u32,
        rules: *const std::ffi::c_void,
        rules_len: usize,
        used_nodes: *const usize,
        used_nodes_len: usize,
        buffer_size: u32,
        intermediates: *mut std::ffi::c_void,
        large_domain: u32,
        num_x: u32,
        num_rows_per_tile: u32,
        skip_stride: u32,
        challenges: *const std::ffi::c_void,
    ) -> i32;

    // interactions_bary.cu
    pub fn _logup_r0_temp_sums_buffer_size(buffer_size: u32, large_domain: u32, num_x: u32) -> u32;
    pub fn _logup_r0_intermediates_buffer_size(
        buffer_size: u32,
        large_domain: u32,
        num_x: u32,
    ) -> u32;
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
    ) -> i32;

    // constraints.cu
    fn _zerocheck_eval_constraints(
        output: *mut std::ffi::c_void,
        selectors: *const std::ffi::c_void,
        selectors_width: u32,
        partitioned_main: *const MainMatrixPtrs<F>,
        main_count: u32,
        preprocessed: *const F,
        preprocessed_air_width: u32,
        eq_z: *const std::ffi::c_void,
        eq_x: *const std::ffi::c_void,
        lambda_pows: *const std::ffi::c_void,
        lambda_indices: *const u32,
        public_values: *const std::ffi::c_void,
        public_len: u32,
        rules: *const std::ffi::c_void,
        rules_len: usize,
        used_nodes: *const usize,
        used_nodes_len: usize,
        lambda_len: usize,
        buffer_size: u32,
        intermediates: *mut std::ffi::c_void,
        large_domain: u32,
        num_x: u32,
        num_rows_per_tile: u32,
        skip_stride: u32,
    ) -> i32;
    fn _accumulate_constraints(
        output: *const std::ffi::c_void,
        sums: *mut std::ffi::c_void,
        large_domain: u32,
        num_x: u32,
    ) -> i32;
    fn _extract_component(
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        len: u32,
        component_idx: u32,
    ) -> i32;
    fn _assign_component(
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        len: u32,
        component_idx: u32,
    ) -> i32;

    // constraints_bary.cu
    pub fn _zerocheck_r0_temp_sums_buffer_size(
        buffer_size: u32,
        large_domain: u32,
        num_x: u32,
    ) -> u32;
    pub fn _zerocheck_r0_intermediates_buffer_size(
        buffer_size: u32,
        large_domain: u32,
        num_x: u32,
    ) -> u32;
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
        d_lambda_indices: *const u32,
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
    ) -> i32;
    fn _fold_selectors_round0(
        out: *mut EF,
        input: *const F,
        is_first: EF,
        is_last: EF,
        num_x: u32,
    ) -> i32;

    // mle.cu
    fn _zerocheck_eval_mle(
        output: *mut std::ffi::c_void,
        eq_xi: *const EF,
        selectors: *const EF,
        preprocessed: MainMatrixPtrs<EF>,
        main: *const MainMatrixPtrs<EF>,
        lambda_pows: *const std::ffi::c_void,
        lambda_indices: *const u32,
        public_values: *const std::ffi::c_void,
        public_len: u32,
        rules: *const std::ffi::c_void,
        rules_len: usize,
        used_nodes: *const usize,
        used_nodes_len: usize,
        lambda_len: usize,
        buffer_size: u32,
        intermediates: *mut std::ffi::c_void,
        num_y: u32,
        num_x: u32,
        num_rows_per_tile: u32,
    ) -> i32;
    fn _batch_constraints_eval_mle_interactions(
        output_numer: *mut std::ffi::c_void,
        output_denom: *mut std::ffi::c_void,
        eq_sharp: *const EF,
        selectors: *const EF,
        preprocessed: MainMatrixPtrs<EF>,
        main: *const MainMatrixPtrs<EF>,
        challenges: *const std::ffi::c_void,
        eq_3bs: *const std::ffi::c_void,
        public_values: *const std::ffi::c_void,
        public_len: u32,
        rules: *const std::ffi::c_void,
        rules_len: usize,
        used_nodes: *const usize,
        used_nodes_len: usize,
        buffer_size: u32,
        intermediates: *mut std::ffi::c_void,
        num_y: u32,
        num_x: u32,
        num_rows_per_tile: u32,
    ) -> i32;
    fn _reduce_hypercube_blocks(
        block_sums: *mut std::ffi::c_void,
        evaluated: *const std::ffi::c_void,
        s_deg: u32,
        num_y: u32,
    ) -> i32;
    fn _reduce_hypercube_final(
        output: *mut std::ffi::c_void,
        block_sums: *const std::ffi::c_void,
        s_deg: u32,
        num_blocks: u32,
    ) -> i32;
}

pub unsafe fn interpolate_columns_gpu(
    interpolated: &DeviceBuffer<EF>,
    columns: &DeviceBuffer<usize>,
    s_deg: usize,
    num_y: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_interpolate_columns(
        interpolated.as_mut_raw_ptr(),
        columns.as_ptr(),
        s_deg,
        num_y,
        columns.len(),
    ))
}

pub unsafe fn frac_build_tree_layer(
    numerators: &mut DeviceBuffer<EF>,
    denominators: &mut DeviceBuffer<EF>,
    layer_size: usize,
    step: usize,
    revert: bool,
) -> Result<(), CudaError> {
    debug_assert!(numerators.len() >= layer_size);
    debug_assert!(denominators.len() >= layer_size);
    CudaError::from_result(_frac_build_tree_layer(
        numerators.as_mut_ptr(),
        denominators.as_mut_ptr(),
        layer_size,
        step,
        revert,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round(
    eq_xi: &DeviceBuffer<EF>,
    pq_nums: &mut DeviceBuffer<EF>,
    pq_denoms: &mut DeviceBuffer<EF>,
    stride: usize,
    step: usize,
    pq_extra_step: usize,
    lambda: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    #[cfg(debug_assertions)]
    {
        let len = tmp_block_sums.len();
        let required = _frac_compute_round_temp_buffer_size(stride as u32);
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    CudaError::from_result(_frac_compute_round(
        eq_xi.as_ptr(),
        pq_nums.as_mut_ptr(),
        pq_denoms.as_mut_ptr(),
        stride,
        step,
        pq_extra_step,
        lambda,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
    ))
}

pub unsafe fn frac_fold_columns(
    buffer: &mut DeviceBuffer<EF>,
    stride: usize,
    step: usize,
    r: EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_fold_columns(buffer.as_mut_raw_ptr(), stride, step, r))
}

/// Folds matrix of `Frac<EF>` but treats `input` and `output` as **row-major** matrices in
/// `Frac<EF>`. The numerator and denominator are folded pair-wise.
pub unsafe fn fold_ef_columns(
    buffer: &mut DeviceBuffer<EF>,
    stride: usize,
    step: usize,
    r: EF,
    revert: bool,
) -> Result<(), CudaError> {
    let r_or_r_inv = if revert {
        debug_assert!(r != EF::ONE);
        (EF::ONE - r).inverse()
    } else {
        r
    };
    CudaError::from_result(_frac_fold_ext_columns(
        buffer.as_mut_ptr(),
        stride,
        step,
        r_or_r_inv,
        revert,
    ))
}

pub unsafe fn frac_extract_claims(
    data: &DeviceBuffer<EF>,
    stride: usize,
    out_device: &DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_extract_claims(
        data.as_raw_ptr(),
        stride,
        out_device.as_mut_raw_ptr(),
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
    numerators: *mut EF,
    denominators: *mut EF,
    preprocessed: &DeviceBuffer<F>,
    partitioned_main: &DeviceBuffer<u64>,
    challenges: &DeviceBuffer<EF>,
    intermediates: &DeviceBuffer<EF>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    partition_lens: &DeviceBuffer<u32>,
    num_partitions: usize,
    height: u32,
    num_rows_per_tile: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_gkr_input_eval(
        is_global,
        numerators,
        denominators,
        preprocessed.as_raw_ptr(),
        partitioned_main.as_ptr(),
        challenges.as_raw_ptr(),
        intermediates.as_raw_ptr(),
        rules.as_raw_ptr(),
        used_nodes.as_ptr(),
        partition_lens.as_ptr(),
        num_partitions,
        height,
        num_rows_per_tile,
    ))
}

pub unsafe fn frac_add_alpha(data: &DeviceBuffer<Frac<EF>>, alpha: EF) -> Result<(), CudaError> {
    CudaError::from_result(_frac_add_alpha(data.as_mut_raw_ptr(), data.len(), alpha))
}

pub unsafe fn frac_add_alpha_mixed(
    denominators: &DeviceBuffer<EF>,
    alpha: EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_add_alpha_mixed(
        denominators.as_mut_ptr(),
        denominators.len(),
        alpha,
    ))
}

/// # Safety
/// - `buffer_size` does not refer to the capacity of `intermediates`. It refers to "how many DAG
///   nodes per row need to be buffered". The capacity is a multiple of `buffer_size` which is
///   runtime calculated based on `buffer_size`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_constraints(
    output: &DeviceBuffer<EF>,
    selectors: &DeviceMatrix<F>,
    main_ptrs: &DeviceBuffer<MainMatrixPtrs<F>>,
    preprocessed: MainMatrixPtrs<F>,
    eq_z: &DeviceBuffer<EF>,
    eq_x: &DeviceMatrix<EF>,
    lambda_pows: &DeviceBuffer<EF>,
    lambda_indices: &DeviceBuffer<u32>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: Option<&DeviceBuffer<EF>>,
    large_domain: u32,
    num_x: u32,
    num_rows_per_tile: u32,
    skip_stride: u32,
) -> Result<(), CudaError> {
    let intermediates_ptr = intermediates
        .map(|buf| buf.as_mut_raw_ptr())
        .unwrap_or(std::ptr::null_mut());
    CudaError::from_result(_zerocheck_eval_constraints(
        output.as_mut_raw_ptr(),
        selectors.buffer().as_raw_ptr(),
        selectors.width() as u32,
        main_ptrs.as_ptr(),
        main_ptrs.len() as u32,
        preprocessed.data,
        preprocessed.air_width,
        eq_z.as_raw_ptr(),
        eq_x.buffer().as_raw_ptr(),
        lambda_pows.as_raw_ptr(),
        lambda_indices.as_ptr(),
        public_values.as_raw_ptr(),
        public_values.len() as u32,
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        lambda_pows.len(),
        buffer_size,
        intermediates_ptr,
        large_domain,
        num_x,
        num_rows_per_tile,
        skip_stride,
    ))
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
    lambda_indices: &DeviceBuffer<u32>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<F>,
    large_domain: u32,
    skip_domain: u32,
    num_x: u32,
    height: u32,
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
        lambda_indices.as_ptr(),
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
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_mle(
    output: &DeviceBuffer<EF>,
    eq_xi: *const EF,
    selectors: *const EF,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: &DeviceBuffer<MainMatrixPtrs<EF>>,
    lambda_pows: &DeviceBuffer<EF>,
    lambda_indices: &DeviceBuffer<u32>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: Option<&DeviceBuffer<EF>>,
    num_y: u32,
    num_x: u32,
    num_rows_per_tile: u32,
) -> Result<(), CudaError> {
    let intermediates_ptr = intermediates
        .map(|buf| buf.as_mut_raw_ptr())
        .unwrap_or(std::ptr::null_mut());
    CudaError::from_result(_zerocheck_eval_mle(
        output.as_mut_raw_ptr(),
        eq_xi,
        selectors,
        preprocessed,
        main_ptrs.as_ptr(),
        lambda_pows.as_raw_ptr(),
        lambda_indices.as_ptr(),
        public_values.as_raw_ptr(),
        public_values.len() as u32,
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        lambda_pows.len(),
        buffer_size,
        intermediates_ptr,
        num_y,
        num_x,
        num_rows_per_tile,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn batch_constraints_eval_mle_interactions(
    output_numer: &DeviceBuffer<EF>,
    output_denom: &DeviceBuffer<EF>,
    eq_sharp: *const EF,
    selectors: *const EF,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: &DeviceBuffer<MainMatrixPtrs<EF>>,
    challenges: &DeviceBuffer<EF>,
    eq_3bs: &DeviceBuffer<EF>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: Option<&DeviceBuffer<EF>>,
    num_y: u32,
    num_x: u32,
    num_rows_per_tile: u32,
) -> Result<(), CudaError> {
    let intermediates_ptr = intermediates
        .map(|buf| buf.as_mut_raw_ptr())
        .unwrap_or(std::ptr::null_mut());
    CudaError::from_result(_batch_constraints_eval_mle_interactions(
        output_numer.as_mut_raw_ptr(),
        output_denom.as_mut_raw_ptr(),
        eq_sharp,
        selectors,
        preprocessed,
        main_ptrs.as_ptr(),
        challenges.as_raw_ptr(),
        eq_3bs.as_raw_ptr(),
        public_values.as_raw_ptr(),
        public_values.len() as u32,
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        buffer_size,
        intermediates_ptr,
        num_y,
        num_x,
        num_rows_per_tile,
    ))
}

pub unsafe fn reduce_hypercube_blocks(
    block_sums: &mut DeviceBuffer<EF>,
    evaluated: &DeviceBuffer<EF>,
    s_deg: u32,
    num_y: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_reduce_hypercube_blocks(
        block_sums.as_mut_ptr() as *mut std::ffi::c_void,
        evaluated.as_ptr() as *const std::ffi::c_void,
        s_deg,
        num_y,
    ))
}

pub unsafe fn reduce_hypercube_final(
    output: &mut DeviceBuffer<EF>,
    block_sums: &DeviceBuffer<EF>,
    s_deg: u32,
    num_blocks: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_reduce_hypercube_final(
        output.as_mut_ptr() as *mut std::ffi::c_void,
        block_sums.as_ptr() as *const std::ffi::c_void,
        s_deg,
        num_blocks,
    ))
}

/// # Note
/// - all preprocessed, main_ptrs are assumed to be twice the width of the matrix and includes
///   rotation as second copy
#[allow(clippy::too_many_arguments)]
pub unsafe fn batch_constraints_eval_interactions_round0(
    output_numer: &DeviceBuffer<EF>,
    output_denom: &DeviceBuffer<EF>,
    selectors: &DeviceMatrix<F>,
    main_ptrs: &DeviceBuffer<MainMatrixPtrs<F>>,
    preprocessed: MainMatrixPtrs<F>,
    eq_sharp_z: &DeviceBuffer<EF>,
    eq_x: &DeviceMatrix<EF>,
    eq_3b: &DeviceBuffer<EF>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: Option<&DeviceBuffer<EF>>,
    large_domain: u32,
    num_x: u32,
    num_rows_per_tile: u32,
    skip_stride: u32,
    challenges: &DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    let intermediates_ptr = intermediates
        .map(|buf| buf.as_mut_raw_ptr())
        .unwrap_or(std::ptr::null_mut());
    CudaError::from_result(_batch_constraints_eval_interactions_round0(
        output_numer.as_mut_raw_ptr(),
        output_denom.as_mut_raw_ptr(),
        selectors.buffer().as_raw_ptr(),
        selectors.width() as u32,
        main_ptrs.as_ptr(),
        main_ptrs.len() as u32,
        preprocessed.data,
        preprocessed.air_width,
        eq_sharp_z.as_raw_ptr(),
        eq_x.buffer().as_raw_ptr(),
        eq_3b.as_raw_ptr(),
        public_values.as_raw_ptr(),
        public_values.len() as u32,
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        buffer_size,
        intermediates_ptr,
        large_domain,
        num_x,
        num_rows_per_tile,
        skip_stride,
        challenges.as_raw_ptr(),
    ))
}

pub unsafe fn accumulate_constraints(
    output: &DeviceBuffer<EF>,
    sums: &DeviceBuffer<EF>,
    large_domain: u32,
    num_x: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_accumulate_constraints(
        output.as_raw_ptr(),
        sums.as_mut_raw_ptr(),
        large_domain,
        num_x,
    ))
}

pub unsafe fn extract_component(
    input: &DeviceBuffer<EF>,
    output: &DeviceBuffer<F>,
    len: u32,
    component_idx: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_extract_component(
        input.as_raw_ptr(),
        output.as_mut_raw_ptr(),
        len,
        component_idx,
    ))
}

pub unsafe fn assign_component(
    input: &DeviceBuffer<F>,
    output: &DeviceBuffer<EF>,
    len: u32,
    component_idx: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_assign_component(
        input.as_raw_ptr(),
        output.as_mut_raw_ptr(),
        len,
        component_idx,
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

pub unsafe fn frac_vector_scalar_multiply_ext(
    numerators: *mut EF,
    scalar: F,
    length: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_vector_scalar_multiply_ext(numerators, scalar, length))
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
