//! LogUp/Zerocheck kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/logup_zerocheck.rs
//!
//! This is the largest dispatch module, mirroring the extensive CUDA FFI bindings
//! for GKR tree building, compute rounds, folding, zerocheck evaluation, logup
//! evaluation, monomial batching, and related operations.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use std::ffi::c_void;

use metal::Buffer as MetalRawBuffer;
use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use tracing::debug;

use crate::{
    monomial::{InteractionMonomialTerm, LambdaTerm, MonomialHeader},
    poly::SqrtEqLayers,
    prelude::{EF, F},
};

use super::{
    dispatch_sync, get_kernels, grid_size_1d, grid_size_2d, DEFAULT_THREADS_PER_GROUP, SIMD_SIZE,
};

const GKR_SP_DEG: usize = 2;
const GKR_INPUT_TASK_SIZE: u32 = 1 << 16;
const LOGUP_R0_BUFFER_THRESHOLD: u32 = 16;
const ZEROCHECK_R0_BUFFER_THRESHOLD: u32 = 16;
const ZEROCHECK_MONOMIAL_THREADS_PER_BLOCK: u32 = 256;
const ZEROCHECK_MONOMIAL_PAR_Y_THREADS_PER_BLOCK: u32 = 128;
const LOGUP_MONOMIAL_THREADS_PER_BLOCK: u32 = 128;

#[inline]
fn frac_q_offset_bytes(frac_len: usize) -> u64 {
    (frac_len * std::mem::size_of::<EF>()) as u64
}

#[inline]
fn frac_p_ptr(ptr: *const Frac<EF>) -> *const EF {
    ptr as *const EF
}

#[inline]
fn frac_q_ptr(ptr: *const Frac<EF>, frac_len: usize) -> *const EF {
    unsafe { (ptr as *const EF).add(frac_len) }
}

#[inline]
fn frac_p_ptr_mut(ptr: *mut Frac<EF>) -> *mut EF {
    ptr as *mut EF
}

#[inline]
fn frac_q_ptr_mut(ptr: *mut Frac<EF>, frac_len: usize) -> *mut EF {
    unsafe { (ptr as *mut EF).add(frac_len) }
}

#[inline]
fn block_shared_bytes(threads_per_group: usize) -> u64 {
    let simd_groups = threads_per_group.div_ceil(SIMD_SIZE);
    (simd_groups * std::mem::size_of::<EF>()) as u64
}

#[inline]
unsafe fn final_reduce_block_sums_to_buffer(
    block_sums: &MetalBuffer<EF>,
    output_buffer: &MetalRawBuffer,
    output_offset_bytes: u64,
    num_blocks: u32,
    d: u32,
    threads_per_group: usize,
) -> Result<(), MetalError> {
    if d == 0 {
        return Ok(());
    }
    let shared_bytes = block_shared_bytes(threads_per_group);
    let pipeline_reduce = get_kernels().get_pipeline("final_reduce_block_sums")?;
    let final_threads = d as usize * threads_per_group;
    let (grid_reduce, group_reduce) = grid_size_1d(final_threads, threads_per_group);
    dispatch_sync(&pipeline_reduce, grid_reduce, group_reduce, |encoder| {
        encoder.set_buffer(0, Some(block_sums.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(output_buffer), output_offset_bytes);
        encoder.set_bytes(2, 4, &num_blocks as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &d as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })
}

#[inline]
unsafe fn batched_final_reduce_block_sums_to_buffer(
    block_sums: &MetalBuffer<EF>,
    output_buffer: &MetalRawBuffer,
    output_offset_bytes: u64,
    segment_offsets: &MetalBuffer<u32>,
    num_segments: usize,
    d: u32,
    threads_per_group: usize,
) -> Result<(), MetalError> {
    if d == 0 || num_segments == 0 {
        return Ok(());
    }
    let shared_bytes = block_shared_bytes(threads_per_group);
    let pipeline_reduce = get_kernels().get_pipeline("batched_final_reduce_block_sums")?;
    let (grid_reduce, group_reduce) = grid_size_2d(
        num_segments * threads_per_group,
        d as usize,
        threads_per_group,
        1,
    );
    dispatch_sync(&pipeline_reduce, grid_reduce, group_reduce, |encoder| {
        encoder.set_buffer(0, Some(block_sums.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(output_buffer), output_offset_bytes);
        encoder.set_buffer(2, Some(segment_offsets.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &d as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })
}

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
pub struct ColumnPtr<T> {
    pub data: *const T,
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
    pub d_headers: u64,
    pub d_variables: u64,
    pub d_lambda_combinations: u64,
    pub num_monomials: u32,
    pub eval_ctx: EvalCoreCtx,
    pub d_eq_xi: u64,
    pub num_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EvalCoreCtx {
    pub d_selectors: u64,
    pub d_preprocessed: MainMatrixPtrs<EF>,
    pub d_main: u64,
    pub d_public: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZerocheckCtx {
    pub eval_ctx: EvalCoreCtx,
    pub d_intermediates: u64,
    pub num_y: u32,
    pub d_eq_xi: u64,
    pub d_rules: u64,
    pub rules_len: u32,
    pub d_used_nodes: u64,
    pub used_nodes_len: u32,
    pub buffer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupCtx {
    pub eval_ctx: EvalCoreCtx,
    pub d_intermediates: u64,
    pub num_y: u32,
    pub d_eq_xi: u64,
    pub d_challenges: u64,
    pub d_eq_3bs: u64,
    pub d_rules: u64,
    pub rules_len: u32,
    pub d_used_nodes: u64,
    pub d_pair_idxs: u64,
    pub used_nodes_len: u32,
    pub buffer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupMonomialCommonCtx {
    pub eval_ctx: EvalCoreCtx,
    pub d_eq_xi: u64,
    pub bus_term_sum: EF,
    pub num_y: u32,
    pub mono_blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupMonomialCtx {
    pub d_headers: u64,
    pub d_variables: u64,
    pub d_combinations: u64,
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
    if buffer_size <= ZEROCHECK_R0_BUFFER_THRESHOLD {
        return 0;
    }

    let count = (skip_domain as usize) * (num_x as usize);
    let max_threads = skip_domain.max(128) as usize;
    let threads_per_group = count.min(max_threads).max(skip_domain as usize);
    let num_blocks = count.div_ceil(threads_per_group);
    let total_threads = num_blocks * threads_per_group;

    buffer_size as usize * total_threads * num_cosets as usize
}

pub fn zerocheck_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize {
    let blocks = (num_x as usize + 255) / 256;
    blocks * num_y as usize * 2
}

pub fn zerocheck_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize {
    buffer_size as usize * num_x as usize * num_y as usize
}

pub fn logup_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize {
    let blocks = (num_x as usize + 255) / 256;
    blocks * num_y as usize * 2
}

pub fn logup_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize {
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
    let half_len = layer_size / 2;
    let (grid, group) = grid_size_1d(half_len, DEFAULT_THREADS_PER_GROUP);
    let half_len_u32 = half_len as u32;
    let revert_u32: u32 = if revert { 1 } else { 0 };
    let q_offset = frac_q_offset_bytes(layer.len());
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(layer.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(layer.gpu_buffer()), q_offset);
        encoder.set_bytes(2, 4, &half_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &revert_u32 as *const u32 as *const c_void);
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
    let threads_per_group = DEFAULT_THREADS_PER_GROUP;
    let work = (num_x / 2).max(1);
    let num_blocks = work.div_ceil(threads_per_group);
    let total_threads = num_blocks * threads_per_group;
    debug_assert!(tmp_block_sums.len() >= num_blocks * GKR_SP_DEG);
    debug_assert!(out_device.len() >= GKR_SP_DEG);
    let (grid, group) = grid_size_1d(total_threads, threads_per_group);
    let num_x_u32 = num_x as u32;
    let log_eq_low_cap = low_n as u32;
    let q_offset = frac_q_offset_bytes(pq_buffer.len());
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>()) as u64;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_xi.low.layers[low_n].gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.high.layers[high_n].gpu_buffer()), 0);
        encoder.set_buffer(2, Some(pq_buffer.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(pq_buffer.gpu_buffer()), q_offset);
        encoder.set_buffer(4, Some(tmp_block_sums.gpu_buffer()), 0);
        encoder.set_bytes(5, 4, &num_x_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &log_eq_low_cap as *const u32 as *const c_void);
        encoder.set_bytes(
            7,
            std::mem::size_of::<EF>() as u64,
            &lambda as *const EF as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    final_reduce_block_sums_to_buffer(
        tmp_block_sums,
        out_device.gpu_buffer(),
        0,
        num_blocks as u32,
        GKR_SP_DEG as u32,
        threads_per_group,
    )
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
    let threads_per_group = DEFAULT_THREADS_PER_GROUP;
    let work = (num_x / 2).max(1);
    let num_blocks = work.div_ceil(threads_per_group);
    let total_threads = num_blocks * threads_per_group;
    debug_assert!(tmp_block_sums.len() >= num_blocks * GKR_SP_DEG);
    debug_assert!(out_device.len() >= GKR_SP_DEG);
    let (grid, group) = grid_size_1d(total_threads, threads_per_group);
    let num_x_u32 = num_x as u32;
    let log_eq_low_cap = low_n as u32;
    let q_offset = frac_q_offset_bytes(layer.len());
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>()) as u64;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_xi.low.layers[low_n].gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.high.layers[high_n].gpu_buffer()), 0);
        encoder.set_buffer(2, Some(layer.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(layer.gpu_buffer()), q_offset);
        encoder.set_buffer(4, Some(tmp_block_sums.gpu_buffer()), 0);
        encoder.set_bytes(5, 4, &num_x_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &log_eq_low_cap as *const u32 as *const c_void);
        encoder.set_bytes(
            7,
            std::mem::size_of::<EF>() as u64,
            &lambda as *const EF as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    final_reduce_block_sums_to_buffer(
        tmp_block_sums,
        out_device.gpu_buffer(),
        0,
        num_blocks as u32,
        GKR_SP_DEG as u32,
        threads_per_group,
    )
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
    let quarter = size / 4;
    let (grid, group) = grid_size_1d(quarter, DEFAULT_THREADS_PER_GROUP);
    let quarter_u32 = quarter as u32;
    let src_q_offset = frac_q_offset_bytes(src.len());
    let dst_q_offset = frac_q_offset_bytes(dst.len());

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(src.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(dst.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &quarter_u32 as *const u32 as *const c_void);
        encoder.set_bytes(
            3,
            std::mem::size_of::<EF>() as u64,
            &r as *const EF as *const c_void,
        );
    })?;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(src.gpu_buffer()), src_q_offset);
        encoder.set_buffer(1, Some(dst.gpu_buffer()), dst_q_offset);
        encoder.set_bytes(2, 4, &quarter_u32 as *const u32 as *const c_void);
        encoder.set_bytes(
            3,
            std::mem::size_of::<EF>() as u64,
            &r as *const EF as *const c_void,
        );
    })
}

pub unsafe fn fold_ef_frac_columns_inplace(
    buffer: &mut MetalBuffer<Frac<EF>>,
    size: usize,
    r: EF,
) -> Result<(), MetalError> {
    debug_assert!(buffer.len() >= size);
    let pipeline = get_kernels().get_pipeline("frac_fold_fpext_columns")?;
    let quarter = size / 4;
    let (grid, group) = grid_size_1d(quarter, DEFAULT_THREADS_PER_GROUP);
    let quarter_u32 = quarter as u32;
    let q_offset = frac_q_offset_bytes(buffer.len());

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(buffer.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &quarter_u32 as *const u32 as *const c_void);
        encoder.set_bytes(
            3,
            std::mem::size_of::<EF>() as u64,
            &r as *const EF as *const c_void,
        );
    })?;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), q_offset);
        encoder.set_buffer(1, Some(buffer.gpu_buffer()), q_offset);
        encoder.set_bytes(2, 4, &quarter_u32 as *const u32 as *const c_void);
        encoder.set_bytes(
            3,
            std::mem::size_of::<EF>() as u64,
            &r as *const EF as *const c_void,
        );
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
    let threads_per_group = DEFAULT_THREADS_PER_GROUP;
    let work = (num_x / 2).max(1);
    let num_blocks = work.div_ceil(threads_per_group);
    let total_threads = num_blocks * threads_per_group;
    debug_assert!(tmp_block_sums.len() >= num_blocks * GKR_SP_DEG);
    debug_assert!(out_device.len() >= GKR_SP_DEG);
    let (grid, group) = grid_size_1d(total_threads, threads_per_group);
    let num_x_u32 = num_x as u32;
    let log_eq_low_cap = low_n as u32;
    let src_q_offset = frac_q_offset_bytes(src_pq_buffer.len());
    let dst_q_offset = frac_q_offset_bytes(dst_pq_buffer.len());
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>()) as u64;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_xi.low.layers[low_n].gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.high.layers[high_n].gpu_buffer()), 0);
        encoder.set_buffer(2, Some(src_pq_buffer.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(src_pq_buffer.gpu_buffer()), src_q_offset);
        encoder.set_buffer(4, Some(tmp_block_sums.gpu_buffer()), 0);
        encoder.set_buffer(5, Some(dst_pq_buffer.gpu_buffer()), 0);
        encoder.set_buffer(6, Some(dst_pq_buffer.gpu_buffer()), dst_q_offset);
        encoder.set_bytes(7, 4, &num_x_u32 as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &log_eq_low_cap as *const u32 as *const c_void);
        encoder.set_bytes(
            9,
            std::mem::size_of::<EF>() as u64,
            &lambda as *const EF as *const c_void,
        );
        encoder.set_bytes(
            10,
            std::mem::size_of::<EF>() as u64,
            &r_prev as *const EF as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    final_reduce_block_sums_to_buffer(
        tmp_block_sums,
        out_device.gpu_buffer(),
        0,
        num_blocks as u32,
        GKR_SP_DEG as u32,
        threads_per_group,
    )
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
    let threads_per_group = DEFAULT_THREADS_PER_GROUP;
    let work = (num_x / 2).max(1);
    let num_blocks = work.div_ceil(threads_per_group);
    let total_threads = num_blocks * threads_per_group;
    debug_assert!(tmp_block_sums.len() >= num_blocks * GKR_SP_DEG);
    debug_assert!(out_device.len() >= GKR_SP_DEG);
    let (grid, group) = grid_size_1d(total_threads, threads_per_group);
    let num_x_u32 = num_x as u32;
    let log_eq_low_cap = low_n as u32;
    let q_offset = frac_q_offset_bytes(pq_buffer.len());
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>()) as u64;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_xi.low.layers[low_n].gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.high.layers[high_n].gpu_buffer()), 0);
        encoder.set_buffer(2, Some(pq_buffer.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(pq_buffer.gpu_buffer()), q_offset);
        encoder.set_buffer(4, Some(tmp_block_sums.gpu_buffer()), 0);
        encoder.set_bytes(5, 4, &num_x_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &log_eq_low_cap as *const u32 as *const c_void);
        encoder.set_bytes(
            7,
            std::mem::size_of::<EF>() as u64,
            &lambda as *const EF as *const c_void,
        );
        encoder.set_bytes(
            8,
            std::mem::size_of::<EF>() as u64,
            &r_prev as *const EF as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    final_reduce_block_sums_to_buffer(
        tmp_block_sums,
        out_device.gpu_buffer(),
        0,
        num_blocks as u32,
        GKR_SP_DEG as u32,
        threads_per_group,
    )
}

pub unsafe fn frac_precompute_m_build_raw(
    _pq: *const Frac<EF>,
    rem_n: usize,
    w: usize,
    _lambda: EF,
    _r_prev: EF,
    _inline_fold: bool,
    _eq_tail_low: *const EF,
    _eq_tail_high: *const EF,
    eq_tail_low_cap: usize,
    tail_tile: usize,
    partial_out: &MetalBuffer<EF>,
    partial_len: usize,
    m_total: &MetalBuffer<EF>,
) -> Result<(), MetalError> {
    debug_assert!(rem_n > 0);
    debug_assert!(w > 0 && w <= rem_n);
    debug_assert!(eq_tail_low_cap.is_power_of_two());
    debug_assert!(tail_tile > 0);

    let pipeline = get_kernels().get_pipeline("frac_precompute_m_build")?;
    if partial_len == 0 {
        return Ok(());
    }

    // Kernel contract:
    //   buffer(0): partial[num_blocks * total_entries]
    //   buffer(1): m_total[total_entries]
    //   buffer(2): num_blocks
    //   buffer(3): total_entries
    let total_entries = (1usize << w) * (1usize << w);
    debug_assert!(total_entries > 0);
    debug_assert_eq!(partial_len % total_entries, 0);
    let num_blocks = partial_len / total_entries;
    debug_assert!(num_blocks > 0);
    debug_assert!(partial_out.len() >= partial_len);
    debug_assert!(m_total.len() >= total_entries);

    let (grid, group) = grid_size_1d(total_entries, DEFAULT_THREADS_PER_GROUP);
    let num_blocks_u32 = num_blocks as u32;
    let total_entries_u32 = total_entries as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(partial_out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(m_total.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &num_blocks_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &total_entries_u32 as *const u32 as *const c_void);
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
    columns: &MetalBuffer<ColumnPtr<EF>>,
    s_deg: usize,
    num_y: usize,
) -> Result<(), MetalError> {
    let num_columns = columns.len();
    if num_columns == 0 || num_y == 0 {
        return Ok(());
    }
    debug_assert_eq!(interpolated.len(), num_columns * s_deg * num_y);
    let pipeline = get_kernels().get_pipeline("interpolate_columns")?;
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

/// Interpolate one contiguous matrix block directly into the destination matrix.
///
/// Input layout is column-major `[width * height]` with `height == 2 * num_y`.
/// Output is written column-major into `interpolated` at `out_col_offset`.
pub unsafe fn interpolate_matrix_columns_gpu(
    input: &MetalBuffer<EF>,
    interpolated: &MetalBuffer<EF>,
    height: usize,
    s_deg: usize,
    num_y: usize,
    width: usize,
    out_col_offset: usize,
) -> Result<(), MetalError> {
    if width == 0 || num_y == 0 {
        return Ok(());
    }
    debug_assert_eq!(height, 2 * num_y);
    debug_assert!(input.len() >= width * height);
    debug_assert!(interpolated.len() >= (out_col_offset + width) * s_deg * num_y);

    let pipeline = get_kernels().get_pipeline("interpolate_matrix_columns")?;
    let total = width * num_y;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let height_u32 = height as u32;
    let s_deg_u32 = s_deg as u32;
    let num_y_u32 = num_y as u32;
    let width_u32 = width as u32;
    let out_col_offset_u32 = out_col_offset as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(input.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(interpolated.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &s_deg_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &num_y_u32 as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &width_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &out_col_offset_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn fold_ple_from_evals(
    input_matrix: &MetalBuffer<F>,
    output_matrix: &MetalBuffer<EF>,
    output_offset: usize,
    omega_skip_pows: &MetalBuffer<F>,
    inv_lagrange_denoms: &MetalBuffer<EF>,
    height: u32,
    width: u32,
    l_skip: u32,
    new_height: u32,
    rotate: bool,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("fold_ple_from_evals")?;
    let skip_domain = 1u32 << l_skip;
    let threads_per_group = if skip_domain as usize <= DEFAULT_THREADS_PER_GROUP {
        DEFAULT_THREADS_PER_GROUP
    } else {
        skip_domain as usize
    };
    let chunks_per_block = (threads_per_group / skip_domain as usize).max(1);
    let num_blocks = (new_height as usize).div_ceil(chunks_per_block);
    let (grid, group) = grid_size_1d(num_blocks * threads_per_group, threads_per_group);
    let output_offset_bytes = (output_offset * std::mem::size_of::<EF>()) as u64;
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>()) as u64;
    let rotate_u32: u32 = if rotate { 1 } else { 0 };
    for col in 0..width {
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input_matrix.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output_matrix.gpu_buffer()), output_offset_bytes);
            encoder.set_buffer(2, Some(omega_skip_pows.gpu_buffer()), 0);
            encoder.set_buffer(3, Some(inv_lagrange_denoms.gpu_buffer()), 0);
            encoder.set_bytes(4, 4, &height as *const u32 as *const c_void);
            encoder.set_bytes(5, 4, &skip_domain as *const u32 as *const c_void);
            encoder.set_bytes(6, 4, &l_skip as *const u32 as *const c_void);
            encoder.set_bytes(7, 4, &new_height as *const u32 as *const c_void);
            encoder.set_bytes(8, 4, &col as *const u32 as *const c_void);
            encoder.set_bytes(9, 4, &rotate_u32 as *const u32 as *const c_void);
            encoder.set_threadgroup_memory_length(0, shared_bytes);
        })?;
    }
    Ok(())
}

pub unsafe fn frac_add_alpha(data: &MetalBuffer<Frac<EF>>, alpha: EF) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("frac_add_alpha")?;
    let len = data.len();
    let (grid, group) = grid_size_1d(len, DEFAULT_THREADS_PER_GROUP);
    let len_u32 = len as u32;
    let q_offset = frac_q_offset_bytes(len);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(data.gpu_buffer()), q_offset);
        encoder.set_bytes(1, 4, &len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(
            2,
            std::mem::size_of::<EF>() as u64,
            &alpha as *const EF as *const c_void,
        );
    })
}

pub unsafe fn frac_vector_scalar_multiply_ext_fp(
    frac_vec: &MetalBuffer<Frac<EF>>,
    start: usize,
    scalar: F,
    length: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("frac_vector_scalar_multiply_ext_fp")?;
    let (grid, group) = grid_size_1d(length as usize, DEFAULT_THREADS_PER_GROUP);
    let start_offset_bytes = (start * std::mem::size_of::<EF>()) as u64;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(frac_vec.gpu_buffer()), start_offset_bytes);
        encoder.set_bytes(
            1,
            std::mem::size_of::<F>() as u64,
            &scalar as *const F as *const c_void,
        );
        encoder.set_bytes(2, 4, &length as *const u32 as *const c_void);
    })
}

pub unsafe fn frac_matrix_vertically_repeat(
    out: &MetalBuffer<Frac<EF>>,
    out_offset: usize,
    input: &MetalBuffer<Frac<EF>>,
    input_offset: usize,
    width: u32,
    lifted_height: u32,
    height: u32,
) -> Result<(), MetalError> {
    debug_assert!(lifted_height > height);
    let pipeline = get_kernels().get_pipeline("frac_matrix_vertically_repeat")?;
    let out_p_offset = (out_offset * std::mem::size_of::<EF>()) as u64;
    let out_q_offset =
        frac_q_offset_bytes(out.len()) + (out_offset * std::mem::size_of::<EF>()) as u64;
    let in_p_offset = (input_offset * std::mem::size_of::<EF>()) as u64;
    let in_q_offset =
        frac_q_offset_bytes(input.len()) + (input_offset * std::mem::size_of::<EF>()) as u64;
    let (grid, group) = grid_size_2d(lifted_height as usize, width as usize, 64, 1);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), out_p_offset);
        encoder.set_buffer(1, Some(out.gpu_buffer()), out_q_offset);
        encoder.set_buffer(2, Some(input.gpu_buffer()), in_p_offset);
        encoder.set_buffer(3, Some(input.gpu_buffer()), in_q_offset);
        encoder.set_bytes(4, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &lifted_height as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &height as *const u32 as *const c_void);
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
    out: &MetalBuffer<EF>,
    input: &MetalBuffer<F>,
    is_first: EF,
    is_last: EF,
    num_x: usize,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("fold_selectors_round0")?;
    let (grid, group) = grid_size_1d(num_x, DEFAULT_THREADS_PER_GROUP);
    let num_x_u32 = num_x as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(input.gpu_buffer()), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<EF>() as u64,
            &is_first as *const EF as *const c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<EF>() as u64,
            &is_last as *const EF as *const c_void,
        );
        encoder.set_bytes(4, 4, &num_x_u32 as *const u32 as *const c_void);
    })
}

// Precompute lambda combinations
pub unsafe fn precompute_lambda_combinations(
    out: &mut MetalBuffer<EF>,
    headers: &MetalBuffer<MonomialHeader>,
    lambda_terms: &MetalBuffer<LambdaTerm<F>>,
    lambda_pows: &MetalBuffer<EF>,
    num_monomials: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("precompute_lambda_combinations")?;
    let (grid, group) = grid_size_1d(num_monomials as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(headers.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(lambda_terms.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(lambda_pows.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &num_monomials as *const u32 as *const c_void);
    })
}

pub unsafe fn precompute_logup_numer_combinations(
    out: &mut MetalBuffer<EF>,
    headers: &MetalBuffer<MonomialHeader>,
    terms: &MetalBuffer<InteractionMonomialTerm<F>>,
    eq_3bs: &MetalBuffer<EF>,
    num_monomials: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("precompute_logup_numer_combinations")?;
    let (grid, group) = grid_size_1d(num_monomials as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(headers.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(terms.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(eq_3bs.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &num_monomials as *const u32 as *const c_void);
    })
}

pub unsafe fn precompute_logup_denom_combinations(
    out: &mut MetalBuffer<EF>,
    headers: &MetalBuffer<MonomialHeader>,
    terms: &MetalBuffer<InteractionMonomialTerm<F>>,
    beta_pows: &MetalBuffer<EF>,
    eq_3bs: &MetalBuffer<EF>,
    num_monomials: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("precompute_logup_denom_combinations")?;
    let (grid, group) = grid_size_1d(num_monomials as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(headers.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(terms.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(beta_pows.gpu_buffer()), 0);
        encoder.set_buffer(4, Some(eq_3bs.gpu_buffer()), 0);
        encoder.set_bytes(5, 4, &num_monomials as *const u32 as *const c_void);
    })
}

#[inline]
fn twiddles_for_l_skip(l_skip: u32) -> MetalBuffer<F> {
    if l_skip == 0 {
        return MetalBuffer::from_slice(&[F::ONE, F::ONE]);
    }
    let max_level = l_skip as usize;
    let total_size = (1usize << (max_level + 1)) - 2;
    let mut twiddles = Vec::with_capacity(total_size);
    for level in 1..=max_level {
        let root = F::two_adic_generator(level);
        let mut cur = F::ONE;
        for _ in 0..(1usize << level) {
            twiddles.push(cur);
            cur *= root;
        }
    }
    MetalBuffer::from_slice(&twiddles)
}

#[inline]
unsafe fn write_frac_split(buf: &MetalBuffer<Frac<EF>>, idx: usize, p: EF, q: EF) {
    let p_ptr = buf.as_mut_ptr() as *mut EF;
    let q_ptr = p_ptr.add(buf.len());
    *p_ptr.add(idx) = p;
    *q_ptr.add(idx) = q;
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_gkr_input_eval(
    is_global: bool,
    fracs: &MetalBuffer<Frac<EF>>,
    fracs_offset: usize,
    preprocessed: &MetalBuffer<F>,
    main_parts: &[&MetalBuffer<F>],
    main_part_ptrs: &MetalBuffer<*const F>,
    public_values: &MetalBuffer<F>,
    challenges: &MetalBuffer<EF>,
    intermediates: &MetalBuffer<EF>,
    rules: &MetalBuffer<u128>,
    used_nodes: &MetalBuffer<usize>,
    pair_idxs: &MetalBuffer<u32>,
    height: u32,
    num_rows_per_tile: u32,
) -> Result<(), MetalError> {
    debug_assert_eq!(used_nodes.len(), pair_idxs.len());
    let pipeline_name = if is_global {
        "evaluate_interactions_gkr_global"
    } else {
        "evaluate_interactions_gkr_local"
    };
    let pipeline = get_kernels().get_pipeline(pipeline_name)?;
    let count = if is_global {
        GKR_INPUT_TASK_SIZE
    } else {
        height
    };
    let (grid, group) = grid_size_1d(count as usize, DEFAULT_THREADS_PER_GROUP);

    let p_offset = (fracs_offset * std::mem::size_of::<EF>()) as u64;
    let q_offset = frac_q_offset_bytes(fracs.len()) + p_offset;
    let used_nodes_len = used_nodes.len() as u32;
    let rules_len = rules.len() as u32;
    let num_main_parts = main_parts.len() as u32;
    let total_threads = count;

    if is_global {
        dispatch_sync(&pipeline, grid, group, |encoder| {
            for part in main_parts {
                encoder.use_resource(part.gpu_buffer(), metal::MTLResourceUsage::Read);
            }
            encoder.set_buffer(0, Some(fracs.gpu_buffer()), p_offset);
            encoder.set_buffer(1, Some(fracs.gpu_buffer()), q_offset);
            encoder.set_buffer(2, Some(preprocessed.gpu_buffer()), 0);
            encoder.set_buffer(3, Some(public_values.gpu_buffer()), 0);
            encoder.set_buffer(4, Some(challenges.gpu_buffer()), 0);
            encoder.set_buffer(5, Some(intermediates.gpu_buffer()), 0);
            encoder.set_buffer(6, Some(rules.gpu_buffer()), 0);
            encoder.set_buffer(7, Some(used_nodes.gpu_buffer()), 0);
            encoder.set_buffer(8, Some(pair_idxs.gpu_buffer()), 0);
            encoder.set_bytes(9, 4, &used_nodes_len as *const u32 as *const c_void);
            encoder.set_bytes(10, 4, &height as *const u32 as *const c_void);
            encoder.set_bytes(11, 4, &rules_len as *const u32 as *const c_void);
            encoder.set_bytes(12, 4, &num_rows_per_tile as *const u32 as *const c_void);
            encoder.set_bytes(13, 4, &total_threads as *const u32 as *const c_void);
            encoder.set_buffer(14, Some(main_part_ptrs.gpu_buffer()), 0);
            encoder.set_bytes(15, 4, &num_main_parts as *const u32 as *const c_void);
        })
    } else {
        dispatch_sync(&pipeline, grid, group, |encoder| {
            for part in main_parts {
                encoder.use_resource(part.gpu_buffer(), metal::MTLResourceUsage::Read);
            }
            encoder.set_buffer(0, Some(fracs.gpu_buffer()), p_offset);
            encoder.set_buffer(1, Some(fracs.gpu_buffer()), q_offset);
            encoder.set_buffer(2, Some(preprocessed.gpu_buffer()), 0);
            encoder.set_buffer(3, Some(public_values.gpu_buffer()), 0);
            encoder.set_buffer(4, Some(challenges.gpu_buffer()), 0);
            encoder.set_buffer(5, Some(rules.gpu_buffer()), 0);
            encoder.set_buffer(6, Some(used_nodes.gpu_buffer()), 0);
            encoder.set_buffer(7, Some(pair_idxs.gpu_buffer()), 0);
            encoder.set_bytes(8, 4, &used_nodes_len as *const u32 as *const c_void);
            encoder.set_bytes(9, 4, &height as *const u32 as *const c_void);
            encoder.set_bytes(10, 4, &rules_len as *const u32 as *const c_void);
            encoder.set_bytes(11, 4, &num_rows_per_tile as *const u32 as *const c_void);
            encoder.set_bytes(12, 4, &total_threads as *const u32 as *const c_void);
            encoder.set_buffer(13, Some(main_part_ptrs.gpu_buffer()), 0);
            encoder.set_bytes(14, 4, &num_main_parts as *const u32 as *const c_void);
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_ntt_eval_constraints(
    output: &MetalBuffer<EF>,
    selectors_cube: &MetalBuffer<F>,
    preprocessed: &MetalBuffer<F>,
    main_ptrs: &MetalBuffer<*const F>,
    main_part_buffers: &[&MetalBuffer<F>],
    eq_cube: &MetalBuffer<EF>,
    lambda_pows: &MetalBuffer<EF>,
    public_values: &MetalBuffer<F>,
    rules: &MetalBuffer<u128>,
    used_nodes: &MetalBuffer<usize>,
    buffer_size: u32,
    intermediates: &MetalBuffer<F>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
) -> Result<(), MetalError> {
    let use_global_intermediates = buffer_size > ZEROCHECK_R0_BUFFER_THRESHOLD;
    let pipeline_name = if use_global_intermediates {
        "zerocheck_ntt_eval_constraints_global"
    } else {
        "zerocheck_ntt_eval_constraints"
    };
    let pipeline = get_kernels().get_pipeline(pipeline_name)?;
    let l_skip = skip_domain.trailing_zeros();
    let twiddles = twiddles_for_l_skip(l_skip);
    let count = (skip_domain as usize) * (num_x as usize);
    let max_threads = skip_domain.max(128) as usize;
    let threads_per_group = count.min(max_threads).max(skip_domain as usize);
    let num_blocks = count.div_ceil(threads_per_group);
    let total_threads = num_blocks * threads_per_group;
    let total_threads_u32 = total_threads as u32;
    let x_int_stride = (total_threads / skip_domain as usize) as u32;
    debug!(
        "zerocheck_r0 launch | count: {} | threads_per_group: {} | num_blocks: {} | x_int_stride: {} | skip_domain: {} | num_x: {}",
        count,
        threads_per_group,
        num_blocks,
        x_int_stride,
        skip_domain,
        num_x
    );
    let (grid, group) = grid_size_1d(total_threads, threads_per_group);
    let needs_tg_mem_flag: u32 = u32::from(skip_domain > 32);
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>())
        + if needs_tg_mem_flag == 1 {
            threads_per_group * std::mem::size_of::<F>()
        } else {
            0
        };
    let rules_len = rules.len() as u32;
    let used_nodes_len = used_nodes.len() as u32;
    let lambda_len = lambda_pows.len() as u32;
    if use_global_intermediates {
        debug_assert!(
            intermediates.len() >= buffer_size as usize * total_threads * num_cosets as usize
        );
    }

    for coset_idx in 0..num_cosets {
        let tmp = MetalBuffer::<EF>::with_capacity(num_blocks as usize * skip_domain as usize);
        if use_global_intermediates {
            dispatch_sync(&pipeline, grid, group, |encoder| {
                for part in main_part_buffers {
                    encoder.use_resource(part.gpu_buffer(), metal::MTLResourceUsage::Read);
                }
                encoder.set_buffer(0, Some(tmp.gpu_buffer()), 0);
                encoder.set_buffer(1, Some(selectors_cube.gpu_buffer()), 0);
                encoder.set_buffer(2, Some(preprocessed.gpu_buffer()), 0);
                encoder.set_buffer(3, Some(main_ptrs.gpu_buffer()), 0);
                encoder.set_buffer(4, Some(eq_cube.gpu_buffer()), 0);
                encoder.set_buffer(5, Some(lambda_pows.gpu_buffer()), 0);
                encoder.set_buffer(6, Some(public_values.gpu_buffer()), 0);
                encoder.set_buffer(7, Some(rules.gpu_buffer()), 0);
                encoder.set_buffer(8, Some(used_nodes.gpu_buffer()), 0);
                encoder.set_buffer(9, Some(twiddles.gpu_buffer()), 0);
                encoder.set_bytes(10, 4, &rules_len as *const u32 as *const c_void);
                encoder.set_bytes(11, 4, &used_nodes_len as *const u32 as *const c_void);
                encoder.set_bytes(12, 4, &lambda_len as *const u32 as *const c_void);
                encoder.set_bytes(13, 4, &buffer_size as *const u32 as *const c_void);
                encoder.set_bytes(14, 4, &skip_domain as *const u32 as *const c_void);
                encoder.set_bytes(15, 4, &num_x as *const u32 as *const c_void);
                encoder.set_bytes(16, 4, &height as *const u32 as *const c_void);
                encoder.set_bytes(17, 4, &coset_idx as *const u32 as *const c_void);
                encoder.set_bytes(
                    18,
                    std::mem::size_of::<F>() as u64,
                    &g_shift as *const F as *const c_void,
                );
                encoder.set_bytes(19, 4, &needs_tg_mem_flag as *const u32 as *const c_void);
                encoder.set_bytes(20, 4, &x_int_stride as *const u32 as *const c_void);
                encoder.set_buffer(21, Some(intermediates.gpu_buffer()), 0);
                encoder.set_bytes(22, 4, &total_threads_u32 as *const u32 as *const c_void);
                encoder.set_bytes(23, 4, &num_cosets as *const u32 as *const c_void);
                encoder.set_threadgroup_memory_length(0, shared_bytes as u64);
            })?;
        } else {
            dispatch_sync(&pipeline, grid, group, |encoder| {
                for part in main_part_buffers {
                    encoder.use_resource(part.gpu_buffer(), metal::MTLResourceUsage::Read);
                }
                encoder.set_buffer(0, Some(tmp.gpu_buffer()), 0);
                encoder.set_buffer(1, Some(selectors_cube.gpu_buffer()), 0);
                encoder.set_buffer(2, Some(preprocessed.gpu_buffer()), 0);
                encoder.set_buffer(3, Some(main_ptrs.gpu_buffer()), 0);
                encoder.set_buffer(4, Some(eq_cube.gpu_buffer()), 0);
                encoder.set_buffer(5, Some(lambda_pows.gpu_buffer()), 0);
                encoder.set_buffer(6, Some(public_values.gpu_buffer()), 0);
                encoder.set_buffer(7, Some(rules.gpu_buffer()), 0);
                encoder.set_buffer(8, Some(used_nodes.gpu_buffer()), 0);
                encoder.set_buffer(9, Some(twiddles.gpu_buffer()), 0);
                encoder.set_bytes(10, 4, &rules_len as *const u32 as *const c_void);
                encoder.set_bytes(11, 4, &used_nodes_len as *const u32 as *const c_void);
                encoder.set_bytes(12, 4, &lambda_len as *const u32 as *const c_void);
                encoder.set_bytes(13, 4, &buffer_size as *const u32 as *const c_void);
                encoder.set_bytes(14, 4, &skip_domain as *const u32 as *const c_void);
                encoder.set_bytes(15, 4, &num_x as *const u32 as *const c_void);
                encoder.set_bytes(16, 4, &height as *const u32 as *const c_void);
                encoder.set_bytes(17, 4, &coset_idx as *const u32 as *const c_void);
                encoder.set_bytes(
                    18,
                    std::mem::size_of::<F>() as u64,
                    &g_shift as *const F as *const c_void,
                );
                encoder.set_bytes(19, 4, &needs_tg_mem_flag as *const u32 as *const c_void);
                encoder.set_bytes(20, 4, &x_int_stride as *const u32 as *const c_void);
                encoder.set_threadgroup_memory_length(0, shared_bytes as u64);
            })?;
        }

        let out_offset_bytes =
            (coset_idx as usize * skip_domain as usize * std::mem::size_of::<EF>()) as u64;
        final_reduce_block_sums_to_buffer(
            &tmp,
            output.gpu_buffer(),
            out_offset_bytes,
            num_blocks as u32,
            skip_domain,
            threads_per_group,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_bary_eval_interactions_round0(
    output: &MetalBuffer<Frac<EF>>,
    selectors_cube: &MetalBuffer<F>,
    preprocessed: &MetalBuffer<F>,
    main_ptrs: &MetalBuffer<*const F>,
    main_part_buffers: &[&MetalBuffer<F>],
    eq_cube: &MetalBuffer<EF>,
    public_values: &MetalBuffer<F>,
    numer_weights: &MetalBuffer<EF>,
    denom_weights: &MetalBuffer<EF>,
    denom_sum_init: EF,
    rules: &MetalBuffer<u128>,
    buffer_size: u32,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
) -> Result<(), MetalError> {
    let use_global_intermediates = buffer_size > LOGUP_R0_BUFFER_THRESHOLD;
    let pipeline_name = if use_global_intermediates {
        "logup_r0_ntt_eval_interactions_global"
    } else {
        "logup_r0_ntt_eval_interactions"
    };
    let pipeline = get_kernels().get_pipeline(pipeline_name)?;
    let l_skip = skip_domain.trailing_zeros();
    let twiddles = twiddles_for_l_skip(l_skip);
    let count = (skip_domain as usize) * (num_x as usize);
    let max_threads = skip_domain.max(128) as usize;
    let threads_per_group = count.min(max_threads).max(skip_domain as usize);
    let num_blocks = count.div_ceil(threads_per_group);
    let total_threads = num_blocks * threads_per_group;
    let total_threads_u32 = total_threads as u32;
    let x_int_stride = (total_threads / skip_domain as usize) as u32;
    debug!(
        "logup_r0 launch | count: {} | threads_per_group: {} | num_blocks: {} | x_int_stride: {} | skip_domain: {} | num_x: {}",
        count,
        threads_per_group,
        num_blocks,
        x_int_stride,
        skip_domain,
        num_x
    );
    let (grid, group) = grid_size_1d(total_threads, threads_per_group);
    let needs_tg_mem_flag: u32 = u32::from(skip_domain > 32);
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>())
        + if needs_tg_mem_flag == 1 {
            threads_per_group * std::mem::size_of::<F>()
        } else {
            0
        };
    let rules_len = rules.len() as u32;
    let q_offset_bytes = frac_q_offset_bytes(output.len());
    let intermediates = if use_global_intermediates {
        MetalBuffer::<F>::with_capacity(buffer_size as usize * total_threads)
    } else {
        MetalBuffer::<F>::with_capacity(0)
    };

    for coset_idx in 0..num_cosets {
        let tmp_p = MetalBuffer::<EF>::with_capacity(num_blocks as usize * skip_domain as usize);
        let tmp_q = MetalBuffer::<EF>::with_capacity(num_blocks as usize * skip_domain as usize);
        let is_identity_coset_flag: u32 = u32::from(coset_idx == 0);
        if use_global_intermediates {
            dispatch_sync(&pipeline, grid, group, |encoder| {
                for part in main_part_buffers {
                    encoder.use_resource(part.gpu_buffer(), metal::MTLResourceUsage::Read);
                }
                encoder.set_buffer(0, Some(tmp_p.gpu_buffer()), 0);
                encoder.set_buffer(1, Some(tmp_q.gpu_buffer()), 0);
                encoder.set_buffer(2, Some(selectors_cube.gpu_buffer()), 0);
                encoder.set_buffer(3, Some(preprocessed.gpu_buffer()), 0);
                encoder.set_buffer(4, Some(main_ptrs.gpu_buffer()), 0);
                encoder.set_buffer(5, Some(eq_cube.gpu_buffer()), 0);
                encoder.set_buffer(6, Some(public_values.gpu_buffer()), 0);
                encoder.set_buffer(7, Some(numer_weights.gpu_buffer()), 0);
                encoder.set_buffer(8, Some(denom_weights.gpu_buffer()), 0);
                encoder.set_bytes(
                    9,
                    std::mem::size_of::<EF>() as u64,
                    &denom_sum_init as *const EF as *const c_void,
                );
                encoder.set_buffer(10, Some(rules.gpu_buffer()), 0);
                encoder.set_buffer(11, Some(twiddles.gpu_buffer()), 0);
                encoder.set_bytes(12, 4, &rules_len as *const u32 as *const c_void);
                encoder.set_bytes(13, 4, &buffer_size as *const u32 as *const c_void);
                encoder.set_bytes(14, 4, &skip_domain as *const u32 as *const c_void);
                encoder.set_bytes(15, 4, &num_x as *const u32 as *const c_void);
                encoder.set_bytes(16, 4, &height as *const u32 as *const c_void);
                encoder.set_bytes(17, 4, &coset_idx as *const u32 as *const c_void);
                encoder.set_bytes(
                    18,
                    std::mem::size_of::<F>() as u64,
                    &g_shift as *const F as *const c_void,
                );
                encoder.set_bytes(19, 4, &needs_tg_mem_flag as *const u32 as *const c_void);
                encoder.set_bytes(
                    20,
                    4,
                    &is_identity_coset_flag as *const u32 as *const c_void,
                );
                encoder.set_bytes(21, 4, &x_int_stride as *const u32 as *const c_void);
                encoder.set_buffer(22, Some(intermediates.gpu_buffer()), 0);
                encoder.set_bytes(23, 4, &total_threads_u32 as *const u32 as *const c_void);
                encoder.set_threadgroup_memory_length(0, shared_bytes as u64);
            })?;
        } else {
            dispatch_sync(&pipeline, grid, group, |encoder| {
                for part in main_part_buffers {
                    encoder.use_resource(part.gpu_buffer(), metal::MTLResourceUsage::Read);
                }
                encoder.set_buffer(0, Some(tmp_p.gpu_buffer()), 0);
                encoder.set_buffer(1, Some(tmp_q.gpu_buffer()), 0);
                encoder.set_buffer(2, Some(selectors_cube.gpu_buffer()), 0);
                encoder.set_buffer(3, Some(preprocessed.gpu_buffer()), 0);
                encoder.set_buffer(4, Some(main_ptrs.gpu_buffer()), 0);
                encoder.set_buffer(5, Some(eq_cube.gpu_buffer()), 0);
                encoder.set_buffer(6, Some(public_values.gpu_buffer()), 0);
                encoder.set_buffer(7, Some(numer_weights.gpu_buffer()), 0);
                encoder.set_buffer(8, Some(denom_weights.gpu_buffer()), 0);
                encoder.set_bytes(
                    9,
                    std::mem::size_of::<EF>() as u64,
                    &denom_sum_init as *const EF as *const c_void,
                );
                encoder.set_buffer(10, Some(rules.gpu_buffer()), 0);
                encoder.set_buffer(11, Some(twiddles.gpu_buffer()), 0);
                encoder.set_bytes(12, 4, &rules_len as *const u32 as *const c_void);
                encoder.set_bytes(13, 4, &buffer_size as *const u32 as *const c_void);
                encoder.set_bytes(14, 4, &skip_domain as *const u32 as *const c_void);
                encoder.set_bytes(15, 4, &num_x as *const u32 as *const c_void);
                encoder.set_bytes(16, 4, &height as *const u32 as *const c_void);
                encoder.set_bytes(17, 4, &coset_idx as *const u32 as *const c_void);
                encoder.set_bytes(
                    18,
                    std::mem::size_of::<F>() as u64,
                    &g_shift as *const F as *const c_void,
                );
                encoder.set_bytes(19, 4, &needs_tg_mem_flag as *const u32 as *const c_void);
                encoder.set_bytes(
                    20,
                    4,
                    &is_identity_coset_flag as *const u32 as *const c_void,
                );
                encoder.set_bytes(21, 4, &x_int_stride as *const u32 as *const c_void);
                encoder.set_threadgroup_memory_length(0, shared_bytes as u64);
            })?;
        }

        let out_offset_bytes =
            (coset_idx as usize * skip_domain as usize * std::mem::size_of::<EF>()) as u64;
        final_reduce_block_sums_to_buffer(
            &tmp_p,
            output.gpu_buffer(),
            out_offset_bytes,
            num_blocks as u32,
            skip_domain,
            threads_per_group,
        )?;
        final_reduce_block_sums_to_buffer(
            &tmp_q,
            output.gpu_buffer(),
            q_offset_bytes + out_offset_bytes,
            num_blocks as u32,
            skip_domain,
            threads_per_group,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_mle(
    output: &MetalBuffer<EF>,
    eq_xi: &MetalBuffer<EF>,
    selectors: &MetalBuffer<EF>,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: &MetalBuffer<MainMatrixPtrs<EF>>,
    lambda_pows: &MetalBuffer<EF>,
    public_values: &MetalBuffer<F>,
    rules: &MetalBuffer<u128>,
    used_nodes: &MetalBuffer<usize>,
    buffer_size: u32,
    _intermediates: &MetalBuffer<EF>,
    num_y: u32,
    num_x: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("zerocheck_mle")?;
    let preprocessed_ctx = MetalBuffer::from_slice(&[preprocessed]);
    let threads_per_group = num_y.min(128).max(1) as usize;
    let num_blocks = num_y.div_ceil(threads_per_group as u32);
    let grid = metal::MTLSize::new(
        num_blocks as u64 * threads_per_group as u64,
        num_x as u64,
        1,
    );
    let group = metal::MTLSize::new(threads_per_group as u64, 1, 1);
    let tmp = MetalBuffer::<EF>::with_capacity(num_blocks as usize * num_x as usize);
    let rules_len = rules.len() as u32;
    let used_nodes_len = used_nodes.len() as u32;
    let lambda_len = lambda_pows.len() as u32;
    let use_global_intermediates: u32 = 1;
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>()) as u64;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(tmp.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_xi.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(selectors.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(preprocessed_ctx.gpu_buffer()), 0);
        encoder.set_buffer(4, Some(main_ptrs.gpu_buffer()), 0);
        encoder.set_buffer(5, Some(lambda_pows.gpu_buffer()), 0);
        encoder.set_buffer(6, Some(public_values.gpu_buffer()), 0);
        encoder.set_buffer(7, Some(rules.gpu_buffer()), 0);
        encoder.set_buffer(8, Some(used_nodes.gpu_buffer()), 0);
        encoder.set_buffer(9, Some(_intermediates.gpu_buffer()), 0);
        encoder.set_bytes(10, 4, &rules_len as *const u32 as *const c_void);
        encoder.set_bytes(11, 4, &used_nodes_len as *const u32 as *const c_void);
        encoder.set_bytes(12, 4, &lambda_len as *const u32 as *const c_void);
        encoder.set_bytes(13, 4, &buffer_size as *const u32 as *const c_void);
        encoder.set_bytes(14, 4, &num_y as *const u32 as *const c_void);
        encoder.set_bytes(15, 4, &num_x as *const u32 as *const c_void);
        encoder.set_bytes(
            16,
            4,
            &use_global_intermediates as *const u32 as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    final_reduce_block_sums_to_buffer(
        &tmp,
        output.gpu_buffer(),
        0,
        num_blocks,
        num_x,
        threads_per_group,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_eval_mle(
    output: &MetalBuffer<Frac<EF>>,
    eq_xi: &MetalBuffer<EF>,
    selectors: &MetalBuffer<EF>,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: &MetalBuffer<MainMatrixPtrs<EF>>,
    challenges: &MetalBuffer<EF>,
    eq_3bs: &MetalBuffer<EF>,
    public_values: &MetalBuffer<F>,
    rules: &MetalBuffer<u128>,
    used_nodes: &MetalBuffer<usize>,
    pair_idxs: &MetalBuffer<u32>,
    buffer_size: u32,
    _intermediates: &MetalBuffer<EF>,
    num_y: u32,
    num_x: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("logup_mle")?;
    let preprocessed_ctx = MetalBuffer::from_slice(&[preprocessed]);
    let threads_per_group = num_y.min(128).max(1) as usize;
    let num_blocks = num_y.div_ceil(threads_per_group as u32);
    let grid = metal::MTLSize::new(
        num_blocks as u64 * threads_per_group as u64,
        num_x as u64,
        1,
    );
    let group = metal::MTLSize::new(threads_per_group as u64, 1, 1);
    let tmp_p = MetalBuffer::<EF>::with_capacity(num_blocks as usize * num_x as usize);
    let tmp_q = MetalBuffer::<EF>::with_capacity(num_blocks as usize * num_x as usize);
    let used_nodes_len = used_nodes.len() as u32;
    let rules_len = rules.len() as u32;
    let use_global_intermediates: u32 = 1;
    let shared_bytes = (threads_per_group * std::mem::size_of::<EF>()) as u64;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(tmp_p.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(tmp_q.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(eq_xi.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(selectors.gpu_buffer()), 0);
        encoder.set_buffer(4, Some(preprocessed_ctx.gpu_buffer()), 0);
        encoder.set_buffer(5, Some(main_ptrs.gpu_buffer()), 0);
        encoder.set_buffer(6, Some(challenges.gpu_buffer()), 0);
        encoder.set_buffer(7, Some(eq_3bs.gpu_buffer()), 0);
        encoder.set_buffer(8, Some(public_values.gpu_buffer()), 0);
        encoder.set_buffer(9, Some(rules.gpu_buffer()), 0);
        encoder.set_buffer(10, Some(used_nodes.gpu_buffer()), 0);
        encoder.set_buffer(11, Some(pair_idxs.gpu_buffer()), 0);
        encoder.set_buffer(12, Some(_intermediates.gpu_buffer()), 0);
        encoder.set_bytes(13, 4, &used_nodes_len as *const u32 as *const c_void);
        encoder.set_bytes(14, 4, &buffer_size as *const u32 as *const c_void);
        encoder.set_bytes(15, 4, &num_y as *const u32 as *const c_void);
        encoder.set_bytes(16, 4, &num_x as *const u32 as *const c_void);
        encoder.set_bytes(17, 4, &rules_len as *const u32 as *const c_void);
        encoder.set_bytes(
            18,
            4,
            &use_global_intermediates as *const u32 as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    final_reduce_block_sums_to_buffer(
        &tmp_p,
        output.gpu_buffer(),
        0,
        num_blocks,
        num_x,
        threads_per_group,
    )?;
    final_reduce_block_sums_to_buffer(
        &tmp_q,
        output.gpu_buffer(),
        frac_q_offset_bytes(output.len()),
        num_blocks,
        num_x,
        threads_per_group,
    )
}

pub unsafe fn zerocheck_batch_eval_mle(
    output: &MetalBuffer<EF>,
    block_ctxs: &MetalBuffer<BlockCtx>,
    zc_ctxs: &MetalBuffer<ZerocheckCtx>,
    air_block_offsets: &MetalBuffer<u32>,
    lambda_pows: &MetalBuffer<EF>,
    lambda_len: usize,
    num_x: u32,
    threads_per_block: u32,
    read_resources: &[MetalRawBuffer],
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("zerocheck_batch_mle")?;
    let num_blocks = block_ctxs.len();
    let num_airs = zc_ctxs.len();
    let grid = metal::MTLSize::new(
        num_blocks as u64 * threads_per_block as u64,
        num_x as u64,
        1,
    );
    let group = metal::MTLSize::new(threads_per_block as u64, 1, 1);
    let tmp = MetalBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
    let lambda_len_u32 = lambda_len as u32;
    let shared_bytes = (threads_per_block as usize * std::mem::size_of::<EF>()) as u64;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        for resource in read_resources {
            encoder.use_resource(
                resource,
                metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
            );
        }
        encoder.set_buffer(0, Some(tmp.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(block_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(zc_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(lambda_pows.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &lambda_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &num_x as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    batched_final_reduce_block_sums_to_buffer(
        &tmp,
        output.gpu_buffer(),
        0,
        air_block_offsets,
        num_airs,
        num_x,
        threads_per_block as usize,
    )
}

pub unsafe fn logup_batch_eval_mle(
    output: &MetalBuffer<Frac<EF>>,
    block_ctxs: &MetalBuffer<BlockCtx>,
    logup_ctxs: &MetalBuffer<LogupCtx>,
    air_block_offsets: &MetalBuffer<u32>,
    num_x: u32,
    threads_per_block: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("logup_batch_mle")?;
    let num_blocks = block_ctxs.len();
    let num_airs = logup_ctxs.len();
    let grid = metal::MTLSize::new(
        num_blocks as u64 * threads_per_block as u64,
        num_x as u64,
        1,
    );
    let group = metal::MTLSize::new(threads_per_block as u64, 1, 1);
    let tmp_p = MetalBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
    let tmp_q = MetalBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
    let shared_bytes = (threads_per_block as usize * std::mem::size_of::<EF>()) as u64;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(tmp_p.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(tmp_q.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(block_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(logup_ctxs.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &num_x as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    batched_final_reduce_block_sums_to_buffer(
        &tmp_p,
        output.gpu_buffer(),
        0,
        air_block_offsets,
        num_airs,
        num_x,
        threads_per_block as usize,
    )?;
    batched_final_reduce_block_sums_to_buffer(
        &tmp_q,
        output.gpu_buffer(),
        frac_q_offset_bytes(output.len()),
        air_block_offsets,
        num_airs,
        num_x,
        threads_per_block as usize,
    )
}

pub unsafe fn zerocheck_monomial_batched(
    output: &MetalBuffer<EF>,
    block_ctxs: &MetalBuffer<BlockCtx>,
    air_ctxs: &MetalBuffer<MonomialAirCtx>,
    air_offsets: &MetalBuffer<u32>,
    read_resources: &[MetalRawBuffer],
    num_x: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("zerocheck_monomial")?;
    let num_blocks = block_ctxs.len();
    let num_airs = air_ctxs.len();
    let threads_per_block = ZEROCHECK_MONOMIAL_THREADS_PER_BLOCK;
    let grid = metal::MTLSize::new(
        num_blocks as u64 * threads_per_block as u64,
        num_x as u64,
        1,
    );
    let group = metal::MTLSize::new(threads_per_block as u64, 1, 1);
    let tmp = MetalBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
    let shared_bytes = (threads_per_block as usize * std::mem::size_of::<EF>()) as u64;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        for resource in read_resources {
            encoder.use_resource(resource, metal::MTLResourceUsage::Read);
        }
        encoder.set_buffer(0, Some(tmp.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(block_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(air_ctxs.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &threads_per_block as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &num_x as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    #[cfg(debug_assertions)]
    if tracing::enabled!(tracing::Level::DEBUG) {
        let tmp_host = tmp.to_vec();
        let non_zero = tmp_host.iter().filter(|v| **v != EF::ZERO).count();
        let preview: Vec<_> = tmp_host.iter().take(tmp_host.len().min(8)).copied().collect();
        debug!(
            num_blocks,
            num_x,
            non_zero,
            ?preview,
            "zerocheck_monomial_tmp_preview"
        );
    }

    batched_final_reduce_block_sums_to_buffer(
        &tmp,
        output.gpu_buffer(),
        0,
        air_offsets,
        num_airs,
        num_x,
        threads_per_block as usize,
    )
}

pub unsafe fn zerocheck_monomial_par_y_batched(
    output: &MetalBuffer<EF>,
    block_ctxs: &MetalBuffer<BlockCtx>,
    air_ctxs: &MetalBuffer<MonomialAirCtx>,
    air_offsets: &MetalBuffer<u32>,
    read_resources: &[MetalRawBuffer],
    chunk_size: u32,
    num_x: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("zerocheck_monomial_par_y")?;
    let num_blocks = block_ctxs.len();
    let num_airs = air_ctxs.len();
    let threads_per_block = ZEROCHECK_MONOMIAL_PAR_Y_THREADS_PER_BLOCK;
    let grid = metal::MTLSize::new(
        num_blocks as u64 * threads_per_block as u64,
        num_x as u64,
        1,
    );
    let group = metal::MTLSize::new(threads_per_block as u64, 1, 1);
    let tmp = MetalBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
    let shared_bytes = (threads_per_block as usize * std::mem::size_of::<EF>()) as u64;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        for resource in read_resources {
            encoder.use_resource(resource, metal::MTLResourceUsage::Read);
        }
        encoder.set_buffer(0, Some(tmp.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(block_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(air_ctxs.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &threads_per_block as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &chunk_size as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &num_x as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    #[cfg(debug_assertions)]
    if tracing::enabled!(tracing::Level::DEBUG) {
        let tmp_host = tmp.to_vec();
        let non_zero = tmp_host.iter().filter(|v| **v != EF::ZERO).count();
        let preview: Vec<_> = tmp_host.iter().take(tmp_host.len().min(8)).copied().collect();
        debug!(
            num_blocks,
            num_x,
            chunk_size,
            non_zero,
            ?preview,
            "zerocheck_monomial_par_y_tmp_preview"
        );
    }

    batched_final_reduce_block_sums_to_buffer(
        &tmp,
        output.gpu_buffer(),
        0,
        air_offsets,
        num_airs,
        num_x,
        threads_per_block as usize,
    )
}

pub unsafe fn logup_monomial_batched(
    output: &MetalBuffer<Frac<EF>>,
    block_ctxs: &MetalBuffer<BlockCtx>,
    common_ctxs: &MetalBuffer<LogupMonomialCommonCtx>,
    numer_ctxs: &MetalBuffer<LogupMonomialCtx>,
    denom_ctxs: &MetalBuffer<LogupMonomialCtx>,
    air_offsets: &MetalBuffer<u32>,
    num_x: u32,
) -> Result<(), MetalError> {
    let numer_pipeline = get_kernels().get_pipeline("logup_monomial_numer")?;
    let denom_pipeline = get_kernels().get_pipeline("logup_monomial_denom")?;
    let num_blocks = block_ctxs.len();
    let num_airs = common_ctxs.len();
    let threads_per_block = LOGUP_MONOMIAL_THREADS_PER_BLOCK;
    let grid = metal::MTLSize::new(
        num_blocks as u64 * threads_per_block as u64,
        num_x as u64,
        1,
    );
    let group = metal::MTLSize::new(threads_per_block as u64, 1, 1);
    let tmp_p = MetalBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
    let tmp_q = MetalBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
    let shared_bytes = (threads_per_block as usize * std::mem::size_of::<EF>()) as u64;

    dispatch_sync(&numer_pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(tmp_p.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(block_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(common_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(numer_ctxs.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &num_x as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    dispatch_sync(&denom_pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(tmp_q.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(block_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(common_ctxs.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(denom_ctxs.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &num_x as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    batched_final_reduce_block_sums_to_buffer(
        &tmp_p,
        output.gpu_buffer(),
        0,
        air_offsets,
        num_airs,
        num_x,
        threads_per_block as usize,
    )?;
    batched_final_reduce_block_sums_to_buffer(
        &tmp_q,
        output.gpu_buffer(),
        frac_q_offset_bytes(output.len()),
        air_offsets,
        num_airs,
        num_x,
        threads_per_block as usize,
    )
}
