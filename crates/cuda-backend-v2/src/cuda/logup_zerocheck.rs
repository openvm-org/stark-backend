use openvm_cuda_backend::base::DeviceMatrix;
use openvm_stark_backend::prover::MatrixDimensions;

use super::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MainMatrixPtrs {
    pub data: *const F,
    pub air_width: u32,
}
extern "C" {
    fn _frac_build_segment_tree(tree: *mut std::ffi::c_void, total_leaves: usize) -> i32;
    fn _frac_prepare_round(
        tree: *const std::ffi::c_void,
        segment_start: usize,
        eval_size: usize,
        pq_out: *mut std::ffi::c_void,
    ) -> i32;
    fn _frac_compute_round(
        eq_xi: *const std::ffi::c_void,
        pq: *const std::ffi::c_void,
        stride: usize,
        lambda: EF,
        out_device: *mut std::ffi::c_void,
    ) -> i32;
    fn _frac_fold_columns(
        input: *const std::ffi::c_void,
        stride: usize,
        width: usize,
        r: EF,
        output: *mut std::ffi::c_void,
    ) -> i32;
    fn _frac_extract_claims(
        data: *const std::ffi::c_void,
        stride: usize,
        out_device: *mut std::ffi::c_void,
    ) -> i32;
    fn _fold_ple_from_evals(
        input_matrix: *const std::ffi::c_void,
        output_matrix_orig: *mut std::ffi::c_void,
        output_matrix_rot: *mut std::ffi::c_void,
        numerators: *const std::ffi::c_void,
        inv_lagrange_denoms: *const std::ffi::c_void,
        height: u32,
        width: u32,
        domain_size: u32,
        l_skip: u32,
        new_height: u32,
        rotate: bool,
    ) -> i32;
    fn _compute_eq_sharp(
        eq_xi: *mut std::ffi::c_void,
        eq_sharp: *mut std::ffi::c_void,
        eq_r0: EF,
        eq_sharp_r0: EF,
        count: u32,
    ) -> i32;
    fn _zerocheck_eval_interactions_gkr(
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
    fn _frac_add_alpha(data: *mut std::ffi::c_void, len: usize, alpha: EF) -> i32;
    fn _zerocheck_eval_constraints(
        output: *mut std::ffi::c_void,
        selectors: *const std::ffi::c_void,
        selectors_width: u32,
        partitioned_main: *const MainMatrixPtrs,
        main_count: u32,
        preprocessed: *const F,
        preprocessed_width: u32,
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
    fn _zerocheck_eval_interactions_round0(
        output_numer: *mut std::ffi::c_void,
        output_denom: *mut std::ffi::c_void,
        selectors: *const std::ffi::c_void,
        selectors_width: u32,
        partitioned_main: *const MainMatrixPtrs,
        main_count: u32,
        preprocessed: *const F,
        preprocessed_width: u32,
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
    fn _frac_vector_scalar_multiply_ext_fp(frac_vec: *mut Frac<EF>, scalar: F, length: u32) -> i32;
}

pub unsafe fn frac_build_segment_tree(
    tree: &DeviceBuffer<Frac<EF>>,
    total_leaves: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_build_segment_tree(
        tree.as_mut_raw_ptr(),
        total_leaves,
    ))
}

pub unsafe fn frac_prepare_round(
    tree: &DeviceBuffer<Frac<EF>>,
    segment_start: usize,
    eval_size: usize,
    pq_out: &DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_prepare_round(
        tree.as_raw_ptr(),
        segment_start,
        eval_size,
        pq_out.as_mut_raw_ptr(),
    ))
}

pub unsafe fn frac_compute_round(
    eq_xi: &DeviceBuffer<EF>,
    pq: &DeviceBuffer<EF>,
    stride: usize,
    lambda: EF,
    out_device: &DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_compute_round(
        eq_xi.as_raw_ptr(),
        pq.as_raw_ptr(),
        stride,
        lambda,
        out_device.as_mut_raw_ptr(),
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
    output_matrix_orig: &DeviceBuffer<EF>,
    output_matrix_rot: &DeviceBuffer<EF>,
    numerators: &DeviceBuffer<EF>,
    inv_lagrange_denoms: &DeviceBuffer<EF>,
    height: u32,
    width: u32,
    domain_size: u32,
    l_skip: u32,
    new_height: u32,
    rotate: bool,
) -> Result<(), CudaError> {
    CudaError::from_result(_fold_ple_from_evals(
        input_matrix.as_raw_ptr(),
        output_matrix_orig.as_mut_raw_ptr(),
        output_matrix_rot.as_mut_raw_ptr(),
        numerators.as_raw_ptr(),
        inv_lagrange_denoms.as_raw_ptr(),
        height,
        width,
        domain_size,
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

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_interactions_gkr(
    is_global: bool,
    output: &DeviceBuffer<Frac<EF>>,
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
    CudaError::from_result(_zerocheck_eval_interactions_gkr(
        is_global,
        output.as_mut_raw_ptr(),
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

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_constraints(
    output: &DeviceBuffer<EF>,
    selectors: &DeviceMatrix<F>,
    main_ptrs: &DeviceBuffer<MainMatrixPtrs>,
    preprocessed: Option<&DeviceMatrix<F>>,
    eq_z: &DeviceBuffer<EF>,
    eq_x: &DeviceMatrix<EF>,
    lambda_pows: &DeviceBuffer<EF>,
    lambda_indices: &DeviceBuffer<u32>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    intermediates: Option<&DeviceBuffer<EF>>,
    large_domain: u32,
    num_x: u32,
    num_rows_per_tile: u32,
    skip_stride: u32,
) -> Result<(), CudaError> {
    let (pre_ptr, pre_width) = preprocessed
        .map(|matrix| (matrix.buffer().as_ptr(), matrix.width() as u32))
        .unwrap_or((std::ptr::null(), 0));
    let intermediates_ptr = intermediates
        .map(|buf| buf.as_mut_raw_ptr())
        .unwrap_or(std::ptr::null_mut());
    CudaError::from_result(_zerocheck_eval_constraints(
        output.as_mut_raw_ptr(),
        selectors.buffer().as_raw_ptr(),
        selectors.width() as u32,
        main_ptrs.as_ptr(),
        main_ptrs.len() as u32,
        pre_ptr,
        pre_width,
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
        intermediates.map(|buf| buf.len() as u32).unwrap_or(0),
        intermediates_ptr,
        large_domain,
        num_x,
        num_rows_per_tile,
        skip_stride,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_interactions_round0(
    output_numer: &DeviceBuffer<EF>,
    output_denom: &DeviceBuffer<EF>,
    selectors: &DeviceMatrix<F>,
    main_ptrs: &DeviceBuffer<MainMatrixPtrs>,
    preprocessed: Option<&DeviceMatrix<F>>,
    eq_z: &DeviceBuffer<EF>,
    eq_x: &DeviceMatrix<EF>,
    eq_3b: &DeviceBuffer<EF>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    intermediates: Option<&DeviceBuffer<EF>>,
    large_domain: u32,
    num_x: u32,
    num_rows_per_tile: u32,
    skip_stride: u32,
    challenges: &DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    let (pre_ptr, pre_width) = preprocessed
        .map(|matrix| (matrix.buffer().as_ptr(), matrix.width() as u32))
        .unwrap_or((std::ptr::null(), 0));
    let intermediates_ptr = intermediates
        .map(|buf| buf.as_mut_raw_ptr())
        .unwrap_or(std::ptr::null_mut());
    CudaError::from_result(_zerocheck_eval_interactions_round0(
        output_numer.as_mut_raw_ptr(),
        output_denom.as_mut_raw_ptr(),
        selectors.buffer().as_raw_ptr(),
        selectors.width() as u32,
        main_ptrs.as_ptr(),
        main_ptrs.len() as u32,
        pre_ptr,
        pre_width,
        eq_z.as_raw_ptr(),
        eq_x.buffer().as_raw_ptr(),
        eq_3b.as_raw_ptr(),
        public_values.as_raw_ptr(),
        public_values.len() as u32,
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        intermediates.map(|buf| buf.len() as u32).unwrap_or(0),
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
