//! LogUp/Zerocheck kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/logup_zerocheck.rs
//!
//! This is the largest dispatch module, mirroring the extensive CUDA FFI bindings
//! for GKR tree building, compute rounds, folding, zerocheck evaluation, logup
//! evaluation, monomial batching, and related operations.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use std::ffi::c_void;

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;

use crate::{
    monomial::{InteractionMonomialTerm, LambdaTerm, MonomialHeader, PackedVar},
    poly::SqrtEqLayers,
    prelude::{EF, F},
};

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};

// ============================================================================
// repr(C) context types (matching CUDA layout for kernel compatibility)
// ============================================================================

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
pub struct MonomialAirCtx {
    pub d_headers: *const MonomialHeader,
    pub d_variables: *const PackedVar,
    pub d_lambda_combinations: *const EF,
    pub num_monomials: u32,
    pub eval_ctx: EvalCoreCtx,
    pub d_eq_xi: *const EF,
    pub num_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EvalCoreCtx {
    pub d_selectors: *const EF,
    pub d_preprocessed: MainMatrixPtrs<EF>,
    pub d_main: *const MainMatrixPtrs<EF>,
    pub d_public: *const F,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZerocheckCtx {
    pub eval_ctx: EvalCoreCtx,
    pub d_intermediates: *mut EF,
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
    pub eval_ctx: EvalCoreCtx,
    pub d_intermediates: *mut EF,
    pub num_y: u32,
    pub d_eq_xi: *const EF,
    pub d_challenges: *const EF,
    pub d_eq_3bs: *const EF,
    pub d_rules: *const std::ffi::c_void,
    pub rules_len: usize,
    pub d_used_nodes: *const usize,
    pub d_pair_idxs: *const u32,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupMonomialCommonCtx {
    pub eval_ctx: EvalCoreCtx,
    pub d_eq_xi: *const EF,
    pub bus_term_sum: EF,
    pub num_y: u32,
    pub mono_blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupMonomialCtx {
    pub d_headers: *const MonomialHeader,
    pub d_variables: *const PackedVar,
    pub d_combinations: *const EF,
    pub num_monomials: u32,
}

// ============================================================================
// Buffer size query functions
// ============================================================================

pub fn frac_compute_round_temp_buffer_size(stride: u32) -> u32 {
    // Conservative estimate: ceil(stride / 256) * 2
    let blocks = (stride + 255) / 256;
    blocks * 2
}

pub fn logup_r0_temp_sums_buffer_size(
    buffer_size: u32,
    skip_domain: u32,
    num_x: u32,
    num_cosets: u32,
    _max_temp_bytes: usize,
) -> usize {
    // Conservative estimate
    let blocks = ((buffer_size * skip_domain * num_x) as usize + 255) / 256;
    blocks * num_cosets as usize * 2
}

pub fn logup_r0_intermediates_buffer_size(
    buffer_size: u32,
    skip_domain: u32,
    num_x: u32,
    num_cosets: u32,
    _max_temp_bytes: usize,
) -> usize {
    buffer_size as usize * skip_domain as usize * num_x as usize * num_cosets as usize
}

pub fn zerocheck_r0_temp_sums_buffer_size(
    buffer_size: u32,
    skip_domain: u32,
    num_x: u32,
    num_cosets: u32,
    _max_temp_bytes: usize,
) -> usize {
    let blocks = ((buffer_size * skip_domain * num_x) as usize + 255) / 256;
    blocks * num_cosets as usize * 2
}

pub fn zerocheck_r0_intermediates_buffer_size(
    buffer_size: u32,
    skip_domain: u32,
    num_x: u32,
    num_cosets: u32,
    _max_temp_bytes: usize,
) -> usize {
    buffer_size as usize * skip_domain as usize * num_x as usize * num_cosets as usize
}

pub fn zerocheck_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize {
    let blocks = (num_x as usize + 255) / 256;
    blocks * num_y as usize * 2
}

pub fn zerocheck_mle_intermediates_buffer_size(
    buffer_size: u32,
    num_x: u32,
    num_y: u32,
) -> usize {
    buffer_size as usize * num_x as usize * num_y as usize
}

pub fn logup_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize {
    let blocks = (num_x as usize + 255) / 256;
    blocks * num_y as usize * 2
}

pub fn logup_mle_intermediates_buffer_size(
    buffer_size: u32,
    num_x: u32,
    num_y: u32,
) -> usize {
    buffer_size as usize * num_x as usize * num_y as usize
}

pub fn zerocheck_batch_mle_intermediates_buffer_size(
    buffer_size: u32,
    num_x: u32,
    num_y: u32,
) -> usize {
    buffer_size as usize * num_x as usize * num_y as usize
}

pub fn logup_batch_mle_intermediates_buffer_size(
    buffer_size: u32,
    num_x: u32,
    num_y: u32,
) -> usize {
    buffer_size as usize * num_x as usize * num_y as usize
}

// ============================================================================
// GKR kernels
// ============================================================================

pub unsafe fn frac_build_tree_layer(
    layer: &mut MetalBuffer<Frac<EF>>,
    layer_size: usize,
    revert: bool,
) -> Result<(), MetalError> {
    debug_assert!(layer.len() >= layer_size);
    let pipeline = get_kernels().get_pipeline("frac_build_tree_layer")?;
    let (grid, group) = grid_size_1d(layer_size / 2, DEFAULT_THREADS_PER_GROUP);
    let layer_size_u32 = layer_size as u32;
    let revert_u32: u32 = if revert { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(layer.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &layer_size_u32 as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &revert_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn frac_compute_round(
    eq_xi: &SqrtEqLayers,
    pq_buffer: &MetalBuffer<Frac<EF>>,
    num_x: usize,
    lambda: EF,
    out_device: &mut MetalBuffer<EF>,
    tmp_block_sums: &mut MetalBuffer<EF>,
) -> Result<(), MetalError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    debug_assert!(pq_buffer.len() >= 2 * num_x);
    let pipeline = get_kernels().get_pipeline("frac_compute_round")?;
    let (grid, group) = grid_size_1d(num_x, DEFAULT_THREADS_PER_GROUP);
    let num_x_u32 = num_x as u32;
    let eq_low_cap = 1u32 << low_n;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_xi.low.layers[low_n].gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.high.layers[high_n].gpu_buffer()), 0);
        encoder.set_buffer(2, Some(pq_buffer.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &num_x_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &eq_low_cap as *const u32 as *const c_void);
        encoder.set_bytes(5, std::mem::size_of::<EF>() as u64, &lambda as *const EF as *const c_void);
        encoder.set_buffer(6, Some(out_device.gpu_buffer()), 0);
        encoder.set_buffer(7, Some(tmp_block_sums.gpu_buffer()), 0);
    })
}

pub unsafe fn frac_compute_round_and_revert(
    eq_xi: &SqrtEqLayers,
    layer: &mut MetalBuffer<Frac<EF>>,
    num_x: usize,
    lambda: EF,
    out_device: &mut MetalBuffer<EF>,
    tmp_block_sums: &mut MetalBuffer<EF>,
) -> Result<(), MetalError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    let pipeline = get_kernels().get_pipeline("frac_compute_round_and_revert")?;
    let (grid, group) = grid_size_1d(num_x, DEFAULT_THREADS_PER_GROUP);
    let num_x_u32 = num_x as u32;
    let eq_low_cap = 1u32 << low_n;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_xi.low.layers[low_n].gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.high.layers[high_n].gpu_buffer()), 0);
        encoder.set_buffer(2, Some(layer.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &num_x_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &eq_low_cap as *const u32 as *const c_void);
        encoder.set_bytes(5, std::mem::size_of::<EF>() as u64, &lambda as *const EF as *const c_void);
        encoder.set_buffer(6, Some(out_device.gpu_buffer()), 0);
        encoder.set_buffer(7, Some(tmp_block_sums.gpu_buffer()), 0);
    })
}

pub unsafe fn fold_ef_frac_columns(
    src: &MetalBuffer<Frac<EF>>,
    dst: &mut MetalBuffer<Frac<EF>>,
    size: usize,
    r: EF,
) -> Result<(), MetalError> {
    debug_assert!(src.len() >= size);
    debug_assert!(dst.len() >= size / 2);
    let pipeline = get_kernels().get_pipeline("frac_fold_fpext_columns")?;
    let (grid, group) = grid_size_1d(size / 4, DEFAULT_THREADS_PER_GROUP);
    let size_u32 = size as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(src.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(dst.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &size_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, std::mem::size_of::<EF>() as u64, &r as *const EF as *const c_void);
    })
}

pub unsafe fn fold_ef_frac_columns_inplace(
    buffer: &mut MetalBuffer<Frac<EF>>,
    size: usize,
    r: EF,
) -> Result<(), MetalError> {
    debug_assert!(buffer.len() >= size);
    let pipeline = get_kernels().get_pipeline("frac_fold_fpext_columns")?;
    let (grid, group) = grid_size_1d(size / 4, DEFAULT_THREADS_PER_GROUP);
    let size_u32 = size as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(buffer.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &size_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, std::mem::size_of::<EF>() as u64, &r as *const EF as *const c_void);
    })
}

pub unsafe fn frac_compute_round_and_fold(
    eq_xi: &SqrtEqLayers,
    src_pq_buffer: &MetalBuffer<Frac<EF>>,
    dst_pq_buffer: &mut MetalBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    out_device: &mut MetalBuffer<EF>,
    tmp_block_sums: &mut MetalBuffer<EF>,
) -> Result<(), MetalError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    let num_x = src_pq_size >> 2;
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    let pipeline = get_kernels().get_pipeline("frac_compute_round_and_fold")?;
    let (grid, group) = grid_size_1d(num_x, DEFAULT_THREADS_PER_GROUP);
    let src_pq_size_u32 = src_pq_size as u32;
    let eq_low_cap = 1u32 << low_n;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_xi.low.layers[low_n].gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.high.layers[high_n].gpu_buffer()), 0);
        encoder.set_buffer(2, Some(src_pq_buffer.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(dst_pq_buffer.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &src_pq_size_u32 as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &eq_low_cap as *const u32 as *const c_void);
        encoder.set_bytes(6, std::mem::size_of::<EF>() as u64, &lambda as *const EF as *const c_void);
        encoder.set_bytes(7, std::mem::size_of::<EF>() as u64, &r_prev as *const EF as *const c_void);
        encoder.set_buffer(8, Some(out_device.gpu_buffer()), 0);
        encoder.set_buffer(9, Some(tmp_block_sums.gpu_buffer()), 0);
    })
}

pub unsafe fn frac_compute_round_and_fold_inplace(
    eq_xi: &SqrtEqLayers,
    pq_buffer: &mut MetalBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    out_device: &mut MetalBuffer<EF>,
    tmp_block_sums: &mut MetalBuffer<EF>,
) -> Result<(), MetalError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    let num_x = src_pq_size >> 2;
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    let pipeline = get_kernels().get_pipeline("frac_compute_round_and_fold_inplace")?;
    let (grid, group) = grid_size_1d(num_x, DEFAULT_THREADS_PER_GROUP);
    let src_pq_size_u32 = src_pq_size as u32;
    let eq_low_cap = 1u32 << low_n;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_xi.low.layers[low_n].gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.high.layers[high_n].gpu_buffer()), 0);
        encoder.set_buffer(2, Some(pq_buffer.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &src_pq_size_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &eq_low_cap as *const u32 as *const c_void);
        encoder.set_bytes(5, std::mem::size_of::<EF>() as u64, &lambda as *const EF as *const c_void);
        encoder.set_bytes(6, std::mem::size_of::<EF>() as u64, &r_prev as *const EF as *const c_void);
        encoder.set_buffer(7, Some(out_device.gpu_buffer()), 0);
        encoder.set_buffer(8, Some(tmp_block_sums.gpu_buffer()), 0);
    })
}

pub unsafe fn frac_precompute_m_build_raw(
    _pq: *const Frac<EF>,
    rem_n: usize,
    w: usize,
    lambda: EF,
    r_prev: EF,
    inline_fold: bool,
    _eq_tail_low: *const EF,
    _eq_tail_high: *const EF,
    eq_tail_low_cap: usize,
    tail_tile: usize,
    _partial_out: *mut EF,
    partial_len: usize,
    _m_total: *mut EF,
) -> Result<(), MetalError> {
    debug_assert!(rem_n > 0);
    debug_assert!(w > 0 && w <= rem_n);
    debug_assert!(eq_tail_low_cap.is_power_of_two());
    debug_assert!(tail_tile > 0);
    let pipeline = get_kernels().get_pipeline("frac_precompute_m_build")?;
    let total = 1 << rem_n;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let rem_n_u32 = rem_n as u32;
    let w_u32 = w as u32;
    let inline_fold_u32: u32 = if inline_fold { 1 } else { 0 };
    let eq_tail_low_cap_u32 = eq_tail_low_cap as u32;
    let tail_tile_u32 = tail_tile as u32;
    let partial_len_u32 = partial_len as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &rem_n_u32 as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &w_u32 as *const u32 as *const c_void);
        encoder.set_bytes(2, std::mem::size_of::<EF>() as u64, &lambda as *const EF as *const c_void);
        encoder.set_bytes(3, std::mem::size_of::<EF>() as u64, &r_prev as *const EF as *const c_void);
        encoder.set_bytes(4, 4, &inline_fold_u32 as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &eq_tail_low_cap_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &tail_tile_u32 as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &partial_len_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn frac_precompute_m_eval_round_raw(
    _m_total: *const EF,
    w: usize,
    t: usize,
    _eq_r_prefix: *const EF,
    _eq_suffix: *const EF,
    _out: *mut EF,
) -> Result<(), MetalError> {
    debug_assert!(w > 0);
    debug_assert!(t < w);
    let pipeline = get_kernels().get_pipeline("frac_precompute_m_eval_round")?;
    let total = 1 << w;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let w_u32 = w as u32;
    let t_u32 = t as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &w_u32 as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &t_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn frac_multifold_raw(
    _src: *const Frac<EF>,
    _dst: *mut Frac<EF>,
    rem_n: usize,
    w: usize,
    _eq_r_window: *const EF,
) -> Result<(), MetalError> {
    debug_assert!(rem_n > 0);
    debug_assert!(w > 0 && w <= rem_n);
    let pipeline = get_kernels().get_pipeline("frac_multifold")?;
    let total = 1 << (rem_n - w);
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let rem_n_u32 = rem_n as u32;
    let w_u32 = w as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &rem_n_u32 as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &w_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn interpolate_columns_gpu(
    interpolated: &MetalBuffer<EF>,
    columns: &MetalBuffer<*const EF>,
    s_deg: usize,
    num_y: usize,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("interpolate_columns")?;
    let num_columns = columns.len();
    let total = num_columns * num_y;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let s_deg_u32 = s_deg as u32;
    let num_y_u32 = num_y as u32;
    let num_columns_u32 = num_columns as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(interpolated.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(columns.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &s_deg_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &num_y_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &num_columns_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn fold_ple_from_evals(
    input_matrix: &MetalBuffer<F>,
    _output_matrix: *mut EF,
    omega_skip_pows: &MetalBuffer<F>,
    inv_lagrange_denoms: &MetalBuffer<EF>,
    height: u32,
    width: u32,
    l_skip: u32,
    new_height: u32,
    rotate: bool,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("fold_ple_from_evals")?;
    let total = (new_height * width) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let rotate_u32: u32 = if rotate { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(input_matrix.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(omega_skip_pows.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(inv_lagrange_denoms.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &height as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &l_skip as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &new_height as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &rotate_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn frac_add_alpha(
    data: &MetalBuffer<Frac<EF>>,
    alpha: EF,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("frac_add_alpha")?;
    let len = data.len();
    let (grid, group) = grid_size_1d(len, DEFAULT_THREADS_PER_GROUP);
    let len_u32 = len as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(data.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(2, std::mem::size_of::<EF>() as u64, &alpha as *const EF as *const c_void);
    })
}

pub unsafe fn frac_vector_scalar_multiply_ext_fp(
    _frac_vec: *mut Frac<EF>,
    scalar: F,
    length: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("frac_vector_scalar_multiply_ext_fp")?;
    let (grid, group) = grid_size_1d(length as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, std::mem::size_of::<F>() as u64, &scalar as *const F as *const c_void);
        encoder.set_bytes(1, 4, &length as *const u32 as *const c_void);
    })
}

pub unsafe fn frac_matrix_vertically_repeat(
    _out: *mut Frac<EF>,
    _input: *const Frac<EF>,
    width: u32,
    lifted_height: u32,
    height: u32,
) -> Result<(), MetalError> {
    debug_assert!(lifted_height > height);
    let pipeline = get_kernels().get_pipeline("frac_matrix_vertically_repeat")?;
    let total = (lifted_height * width) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &lifted_height as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &height as *const u32 as *const c_void);
    })
}

pub unsafe fn frac_matrix_vertically_repeat_ext(
    _out_numerators: *mut EF,
    _out_denominators: *mut EF,
    _in_numerators: *const EF,
    _in_denominators: *const EF,
    width: u32,
    lifted_height: u32,
    height: u32,
) -> Result<(), MetalError> {
    debug_assert!(lifted_height > height);
    let pipeline = get_kernels().get_pipeline("frac_matrix_vertically_repeat_ext")?;
    let total = (lifted_height * width) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &lifted_height as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &height as *const u32 as *const c_void);
    })
}

pub unsafe fn fold_selectors_round0(
    _out: *mut EF,
    _input: *const F,
    is_first: EF,
    is_last: EF,
    num_x: usize,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("fold_selectors_round0")?;
    let (grid, group) = grid_size_1d(num_x, DEFAULT_THREADS_PER_GROUP);
    let num_x_u32 = num_x as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, std::mem::size_of::<EF>() as u64, &is_first as *const EF as *const c_void);
        encoder.set_bytes(1, std::mem::size_of::<EF>() as u64, &is_last as *const EF as *const c_void);
        encoder.set_bytes(2, 4, &num_x_u32 as *const u32 as *const c_void);
    })
}

// Precompute lambda combinations
pub unsafe fn precompute_lambda_combinations(
    out: &mut MetalBuffer<EF>,
    _headers: *const MonomialHeader,
    _lambda_terms: *const LambdaTerm<F>,
    lambda_pows: &MetalBuffer<EF>,
    num_monomials: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("precompute_lambda_combinations")?;
    let (grid, group) = grid_size_1d(num_monomials as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(lambda_pows.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &num_monomials as *const u32 as *const c_void);
    })
}

pub unsafe fn precompute_logup_numer_combinations(
    out: &mut MetalBuffer<EF>,
    _headers: *const MonomialHeader,
    _terms: *const InteractionMonomialTerm<F>,
    eq_3bs: &MetalBuffer<EF>,
    num_monomials: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("precompute_logup_numer_combinations")?;
    let (grid, group) = grid_size_1d(num_monomials as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_3bs.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &num_monomials as *const u32 as *const c_void);
    })
}

pub unsafe fn precompute_logup_denom_combinations(
    out: &mut MetalBuffer<EF>,
    _headers: *const MonomialHeader,
    _terms: *const InteractionMonomialTerm<F>,
    beta_pows: &MetalBuffer<EF>,
    eq_3bs: &MetalBuffer<EF>,
    num_monomials: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("precompute_logup_denom_combinations")?;
    let (grid, group) = grid_size_1d(num_monomials as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(beta_pows.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(eq_3bs.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &num_monomials as *const u32 as *const c_void);
    })
}
