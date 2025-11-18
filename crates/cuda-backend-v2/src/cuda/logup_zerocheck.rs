use std::ffi::c_void;

use openvm_cuda_backend::base::DeviceMatrix;
use openvm_stark_backend::prover::MatrixDimensions;

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
        out_layer: *mut Frac<EF>,
        in_layer: *const Frac<EF>,
        out_layer_size: usize,
    ) -> i32;
    fn _frac_compute_round(
        eq_xi: *const EF,
        pq: *const Frac<EF>,
        stride: usize,
        lambda: EF,
        out_device: *mut EF,
    ) -> i32;
    fn _frac_fold_columns(
        input: *const std::ffi::c_void,
        stride: usize,
        width: usize,
        r: EF,
        output: *mut std::ffi::c_void,
    ) -> i32;
    fn _frac_fold_frac_ext_columns(
        input: *const Frac<EF>,
        stride: usize,
        width: usize,
        r: EF,
        output: *mut Frac<EF>,
    ) -> i32;
    fn _frac_extract_claims(
        data: *const std::ffi::c_void,
        stride: usize,
        out_device: *mut std::ffi::c_void,
    ) -> i32;
    fn _frac_add_alpha(data: *mut std::ffi::c_void, len: usize, alpha: EF) -> i32;
    fn _frac_vector_scalar_multiply_ext_fp(frac_vec: *mut Frac<EF>, scalar: F, length: u32) -> i32;

    // utils.cu
    fn _fold_ple_from_evals(
        input_matrix: *const std::ffi::c_void,
        output_matrix: *mut std::ffi::c_void,
        numerators: *const std::ffi::c_void,
        inv_lagrange_denoms: *const std::ffi::c_void,
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

    // interactions.cu
    fn _logup_gkr_input_eval(
        is_global: bool,
        output: *mut std::ffi::c_void,
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
    out_layer: &mut DeviceBuffer<Frac<EF>>,
    in_layer: &DeviceBuffer<Frac<EF>>,
    out_layer_size: usize,
) -> Result<(), CudaError> {
    debug_assert!(out_layer.len() >= out_layer_size);
    debug_assert!(in_layer.len() >= 2 * out_layer_size);
    CudaError::from_result(_frac_build_tree_layer(
        out_layer.as_mut_ptr(),
        in_layer.as_ptr(),
        out_layer_size,
    ))
}

pub unsafe fn frac_compute_round(
    eq_xi: &DeviceBuffer<EF>,
    pq: &DeviceBuffer<Frac<EF>>,
    stride: usize,
    lambda: EF,
    out_device: &mut DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_compute_round(
        eq_xi.as_ptr(),
        pq.as_ptr(),
        stride,
        lambda,
        out_device.as_mut_ptr(),
    ))
}

pub unsafe fn frac_fold_columns(
    input: &DeviceBuffer<EF>,
    stride: usize,
    width: usize,
    r: EF,
    output: &DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_fold_columns(
        input.as_raw_ptr(),
        stride,
        width,
        r,
        output.as_mut_raw_ptr(),
    ))
}

/// Folds matrix of `Frac<EF>` but treats `input` and `output` as **row-major** matrices in
/// `Frac<EF>`. The numerator and denominator are folded pair-wise.
pub unsafe fn fold_frac_ext_columns(
    input: &DeviceBuffer<Frac<EF>>,
    stride: usize,
    width: usize,
    r: EF,
    output: &mut DeviceBuffer<Frac<EF>>,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_fold_frac_ext_columns(
        input.as_ptr(),
        stride,
        width,
        r,
        output.as_mut_ptr(),
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
    numerators: &DeviceBuffer<EF>,
    inv_lagrange_denoms: &DeviceBuffer<EF>,
    height: u32,
    width: u32,
    l_skip: u32,
    new_height: u32,
    rotate: bool,
) -> Result<(), CudaError> {
    CudaError::from_result(_fold_ple_from_evals(
        input_matrix.as_raw_ptr(),
        output_matrix as *mut c_void,
        numerators.as_raw_ptr(),
        inv_lagrange_denoms.as_raw_ptr(),
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
    output: *mut Frac<EF>,
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
        output as *mut c_void,
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
