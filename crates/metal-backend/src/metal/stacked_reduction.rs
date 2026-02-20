//! Stacked reduction kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/stacked_reduction.rs

#![allow(clippy::too_many_arguments)]

use std::{cmp::min, ffi::c_void, mem};

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use crate::{
    poly::EqEvalSegments,
    prelude::{D_EF, EF, F},
    stacked_reduction::{UnstackedSlice, STACKED_REDUCTION_S_DEG},
};

use super::{
    dispatch_staged_sync, dispatch_sync, encode_dispatch, get_kernels, grid_size_1d, grid_size_2d,
    DEFAULT_THREADS_PER_GROUP, SIMD_SIZE,
};

/// Number of G outputs per z in round 0: G0, G1, G2
pub const NUM_G: usize = 3;
const MAX_GRID_DIM: u32 = 65_535;

#[inline]
fn div_ceil_u32(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

#[inline]
fn align_threads(total: usize, threads_per_group: usize) -> usize {
    let groups = (total + threads_per_group - 1) / threads_per_group;
    groups.max(1) * threads_per_group
}

pub fn stacked_reduction_r0_required_temp_buffer_size(
    trace_height: u32,
    trace_width: u32,
    l_skip: u32,
) -> u32 {
    let skip_domain = 1u32 << l_skip;
    let lifted_height = trace_height.max(skip_domain);
    let block_size = (DEFAULT_THREADS_PER_GROUP as u32).max(skip_domain);
    let grid_x = div_ceil_u32(lifted_height, block_size);
    grid_x * trace_width * skip_domain * NUM_G as u32
}

pub unsafe fn stacked_reduction_sumcheck_round0(
    eq_r_ns: &EqEvalSegments<EF>,
    trace: &MetalBuffer<F>,
    lambda_pows: &MetalBuffer<EF>,
    lambda_pows_offset: usize,
    block_sums: &mut MetalBuffer<EF>,
    output: &mut MetalBuffer<EF>,
    height: usize,
    width: usize,
    l_skip: usize,
) -> Result<(), MetalError> {
    let skip_domain = 1u32 << l_skip;
    let output_size = NUM_G * skip_domain as usize;
    let num_x = (height >> l_skip).max(1) as u32;
    let height_u32 = height as u32;
    let width_u32 = width as u32;
    let l_skip_u32 = l_skip as u32;
    debug_assert!(output.len() >= output_size);

    let block_size = (DEFAULT_THREADS_PER_GROUP as u32).max(skip_domain);
    let lifted_height = height_u32.max(skip_domain);
    let grid_x = div_ceil_u32(lifted_height, block_size);
    let grid_threads_x = (grid_x * block_size) as usize;
    let (grid, group) = grid_size_2d(grid_threads_x, width, block_size as usize, 1);
    let skip_mask = skip_domain - 1;
    let stride = (skip_domain / height_u32.max(1)).max(1);
    let log_stride = stride.ilog2();
    let lambda_pows_offset_bytes = (lambda_pows_offset * mem::size_of::<EF>()) as u64;
    let round0_shared_bytes = ((block_size as usize + 1) * NUM_G * mem::size_of::<EF>()) as u64;

    let pipeline_round0 = get_kernels().get_pipeline("stacked_reduction_sumcheck_round0")?;

    let num_blocks = grid_x * width_u32;
    let d = NUM_G as u32 * skip_domain;
    let reduce_threads_per_group = DEFAULT_THREADS_PER_GROUP;
    let reduce_total_threads = d as usize * reduce_threads_per_group;
    let (reduce_grid, reduce_group) = grid_size_1d(reduce_total_threads, reduce_threads_per_group);
    let reduce_shared_bytes =
        (((reduce_threads_per_group + SIMD_SIZE - 1) / SIMD_SIZE) * mem::size_of::<EF>()) as u64;
    let pipeline_reduce =
        get_kernels().get_pipeline("stacked_reduction_final_reduce_block_sums_add")?;
    dispatch_staged_sync("stacked_reduction_sumcheck_round0", |cmd_buffer| {
        encode_dispatch(cmd_buffer, &pipeline_round0, grid, group, |encoder| {
            encoder.set_buffer(0, Some(eq_r_ns.buffer.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(trace.gpu_buffer()), 0);
            encoder.set_buffer(2, Some(lambda_pows.gpu_buffer()), lambda_pows_offset_bytes);
            encoder.set_buffer(3, Some(block_sums.gpu_buffer()), 0);
            encoder.set_bytes(4, 4, &height_u32 as *const u32 as *const c_void);
            encoder.set_bytes(5, 4, &width_u32 as *const u32 as *const c_void);
            encoder.set_bytes(6, 4, &l_skip_u32 as *const u32 as *const c_void);
            encoder.set_bytes(7, 4, &skip_mask as *const u32 as *const c_void);
            encoder.set_bytes(8, 4, &num_x as *const u32 as *const c_void);
            encoder.set_bytes(9, 4, &log_stride as *const u32 as *const c_void);
            encoder.set_threadgroup_memory_length(0, round0_shared_bytes);
        });
        encode_dispatch(
            cmd_buffer,
            &pipeline_reduce,
            reduce_grid,
            reduce_group,
            |encoder| {
                encoder.set_buffer(0, Some(block_sums.gpu_buffer()), 0);
                encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
                encoder.set_bytes(2, 4, &num_blocks as *const u32 as *const c_void);
                encoder.set_bytes(3, 4, &d as *const u32 as *const c_void);
                encoder.set_threadgroup_memory_length(0, reduce_shared_bytes);
            },
        );
        Ok(2)
    })
}

pub unsafe fn stacked_reduction_fold_ple(
    src: &MetalBuffer<F>,
    dst: &MetalBuffer<EF>,
    dst_offset: usize,
    omega_skip_pows: &MetalBuffer<F>,
    inv_lagrange_denoms: &MetalBuffer<EF>,
    trace_height: usize,
    trace_width: usize,
    l_skip: usize,
) -> Result<(), MetalError> {
    let skip_domain = 1u32 << l_skip;
    debug_assert!(omega_skip_pows.len() >= skip_domain as usize);
    debug_assert!(inv_lagrange_denoms.len() >= skip_domain as usize);
    let trace_height_u32 = trace_height as u32;
    let trace_width_u32 = trace_width as u32;
    let new_height = trace_height_u32.max(skip_domain) / skip_domain;
    let block_size = (DEFAULT_THREADS_PER_GROUP as u32).max(skip_domain);
    let grid_x = div_ceil_u32(new_height, block_size);
    let grid_threads_x = (grid_x * block_size) as usize;
    let (grid, group) = grid_size_2d(grid_threads_x, trace_width, block_size as usize, 1);
    let dst_offset_bytes = (dst_offset * mem::size_of::<EF>()) as u64;

    let pipeline = get_kernels().get_pipeline("stacked_reduction_fold_ple")?;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(src.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(dst.gpu_buffer()), dst_offset_bytes);
        encoder.set_buffer(2, Some(omega_skip_pows.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(inv_lagrange_denoms.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &trace_height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &new_height as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &skip_domain as *const u32 as *const c_void);
    })
}

pub unsafe fn initialize_k_rot_from_eq_segments(
    eq_r_ns: &EqEvalSegments<EF>,
    k_rot_ns: &mut MetalBuffer<EF>,
    k_rot_uni_0: EF,
    k_rot_uni_1: EF,
    max_n: u32,
) -> Result<(), MetalError> {
    debug_assert_eq!(eq_r_ns.buffer.len(), 2 << max_n);
    debug_assert_eq!(k_rot_ns.len(), eq_r_ns.buffer.len());

    let pipeline = get_kernels().get_pipeline("initialize_k_rot_from_eq_segments")?;
    let max_x = 1usize << max_n;
    let aligned_x = align_threads(max_x, DEFAULT_THREADS_PER_GROUP);
    let (grid, group) = grid_size_2d(aligned_x, max_n as usize + 1, DEFAULT_THREADS_PER_GROUP, 1);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_r_ns.buffer.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(k_rot_ns.gpu_buffer()), 0);
        encoder.set_bytes(
            2,
            mem::size_of::<EF>() as u64,
            &k_rot_uni_0 as *const EF as *const c_void,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<EF>() as u64,
            &k_rot_uni_1 as *const EF as *const c_void,
        );
        encoder.set_bytes(4, 4, &max_n as *const u32 as *const c_void);
    })
}

pub unsafe fn stacked_reduction_sumcheck_mle_round(
    q_eval: &MetalBuffer<EF>,
    eq_r_ns: &EqEvalSegments<EF>,
    k_rot_ns: &EqEvalSegments<EF>,
    unstacked_cols: &MetalBuffer<UnstackedSlice>,
    unstacked_cols_offset: usize,
    lambda_pows: &MetalBuffer<EF>,
    lambda_pows_offset: usize,
    output: &mut MetalBuffer<u64>,
    q_height: usize,
    window_len: usize,
    num_y: usize,
    sm_count: u32,
) -> Result<(), MetalError> {
    debug_assert!(output.len() >= STACKED_REDUCTION_S_DEG * D_EF);

    let q_height_u32 = q_height as u32;
    let window_len_u32 = window_len as u32;
    let num_y_u32 = num_y as u32;
    let block_size = DEFAULT_THREADS_PER_GROUP as u32;
    let grid_x = div_ceil_u32(num_y_u32, block_size).max(1);
    let sm_count = sm_count.max(1);

    const WAVES_TARGET: u32 = 4;
    const ITERS_MIN: u32 = 4;
    const ITERS_MAX: u32 = 16;
    let stride_occ = div_ceil_u32(sm_count * WAVES_TARGET, grid_x);
    let stride_loop_lo = div_ceil_u32(window_len_u32, ITERS_MAX);
    let stride_loop_hi = div_ceil_u32(window_len_u32, ITERS_MIN);
    let lo = 1u32.max(stride_occ.max(stride_loop_lo));
    let hi = window_len_u32.min(MAX_GRID_DIM).min(stride_loop_hi);
    let grid_y = if lo <= hi {
        lo
    } else {
        lo.min(window_len_u32.min(MAX_GRID_DIM))
    }
    .max(1);

    let unstacked_cols_offset_bytes =
        (unstacked_cols_offset * mem::size_of::<UnstackedSlice>()) as u64;
    let lambda_pows_offset_bytes = (lambda_pows_offset * mem::size_of::<EF>()) as u64;
    let grid_threads_x = (grid_x * block_size) as usize;
    let (grid, group) = grid_size_2d(grid_threads_x, grid_y as usize, block_size as usize, 1);
    let shared_bytes =
        (((block_size as usize + SIMD_SIZE - 1) / SIMD_SIZE) * mem::size_of::<EF>()) as u64;

    let pipeline = get_kernels().get_pipeline("stacked_reduction_sumcheck_mle_round")?;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(q_eval.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_r_ns.buffer.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(k_rot_ns.buffer.gpu_buffer()), 0);
        encoder.set_buffer(
            3,
            Some(unstacked_cols.gpu_buffer()),
            unstacked_cols_offset_bytes,
        );
        encoder.set_buffer(4, Some(lambda_pows.gpu_buffer()), lambda_pows_offset_bytes);
        encoder.set_buffer(5, Some(output.gpu_buffer()), 0);
        encoder.set_bytes(6, 4, &q_height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &window_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &num_y_u32 as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })
}

pub unsafe fn stacked_reduction_sumcheck_mle_round_degenerate(
    q_eval: &MetalBuffer<EF>,
    eq_ub_ptr: &MetalBuffer<EF>,
    eq_r: EF,
    k_rot_r: EF,
    unstacked_cols: &MetalBuffer<UnstackedSlice>,
    unstacked_cols_offset: usize,
    lambda_pows: &MetalBuffer<EF>,
    lambda_pows_offset: usize,
    output: &mut MetalBuffer<u64>,
    q_height: usize,
    window_len: usize,
    l_skip: usize,
    round: usize,
) -> Result<(), MetalError> {
    debug_assert!(output.len() >= STACKED_REDUCTION_S_DEG * D_EF);

    let q_height_u32 = q_height as u32;
    let window_len_u32 = window_len as u32;
    let shift_factor = (l_skip + round) as u32;
    let block_size = min(window_len_u32.max(1), DEFAULT_THREADS_PER_GROUP as u32);
    let (grid, group) = grid_size_1d(block_size as usize, block_size as usize);
    let unstacked_cols_offset_bytes =
        (unstacked_cols_offset * mem::size_of::<UnstackedSlice>()) as u64;
    let lambda_pows_offset_bytes = (lambda_pows_offset * mem::size_of::<EF>()) as u64;
    let shared_bytes =
        (((block_size as usize + SIMD_SIZE - 1) / SIMD_SIZE) * mem::size_of::<EF>()) as u64;

    let pipeline = get_kernels().get_pipeline("stacked_reduction_sumcheck_mle_round_degenerate")?;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(q_eval.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_ub_ptr.gpu_buffer()), 0);
        encoder.set_bytes(
            2,
            mem::size_of::<EF>() as u64,
            &eq_r as *const EF as *const c_void,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<EF>() as u64,
            &k_rot_r as *const EF as *const c_void,
        );
        encoder.set_buffer(
            4,
            Some(unstacked_cols.gpu_buffer()),
            unstacked_cols_offset_bytes,
        );
        encoder.set_buffer(5, Some(lambda_pows.gpu_buffer()), lambda_pows_offset_bytes);
        encoder.set_buffer(6, Some(output.gpu_buffer()), 0);
        encoder.set_bytes(7, 4, &q_height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &window_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(9, 4, &shift_factor as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })
}
