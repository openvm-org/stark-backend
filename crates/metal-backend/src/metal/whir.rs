//! WHIR kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/whir.rs

#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use crate::prelude::{D_EF, EF, F};

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};

pub unsafe fn whir_algebraic_batch_trace(
    output: &mut MetalBuffer<F>,
    trace: &MetalBuffer<F>,
    mu_powers: &MetalBuffer<EF>,
    stacked_height: u32,
    trace_height: u32,
    trace_width: u32,
    stacked_row_start: u64,
    mu_idx_start: u32,
    skip_domain: u32,
) -> Result<(), MetalError> {
    debug_assert_eq!(output.len() % D_EF, 0);
    debug_assert_eq!(output.len() / D_EF, stacked_height as usize);
    debug_assert_eq!(
        trace.len(),
        trace_height as usize * trace_width as usize,
        "trace buffer length mismatch"
    );
    let pipeline = get_kernels().get_pipeline("whir_algebraic_batch_trace")?;
    let (grid, group) = grid_size_1d(stacked_height as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(output.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(trace.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(mu_powers.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &stacked_height as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &trace_height as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &trace_width as *const u32 as *const c_void);
        encoder.set_bytes(6, 8, &stacked_row_start as *const u64 as *const c_void);
        encoder.set_bytes(7, 4, &mu_idx_start as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &skip_domain as *const u32 as *const c_void);
    })
}

/// Returns the required temp buffer size for whir sumcheck coeff moments round.
pub fn whir_sumcheck_coeff_moments_required_temp_buffer_size(height: u32) -> u32 {
    // Match CUDA launcher: kernel_launch_params(height >> 1, 256), then * S_DEG(=2).
    let half_height = height >> 1;
    let blocks = half_height.div_ceil(DEFAULT_THREADS_PER_GROUP as u32);
    blocks * 2
}

pub unsafe fn whir_sumcheck_coeff_moments_round(
    f_coeffs: &MetalBuffer<EF>,
    w_moments: &MetalBuffer<EF>,
    output: &mut MetalBuffer<EF>,
    tmp_block_sums: &mut MetalBuffer<EF>,
    height: u32,
) -> Result<(), MetalError> {
    debug_assert!(f_coeffs.len() >= height as usize);
    debug_assert!(w_moments.len() >= height as usize);
    let half_height = height >> 1;
    if half_height == 0 {
        output.fill_zero();
        return Ok(());
    }

    let threads_per_group = DEFAULT_THREADS_PER_GROUP;
    let num_blocks = (half_height as usize).div_ceil(threads_per_group) as u32;
    debug_assert!(tmp_block_sums.len() >= num_blocks as usize * 2);
    let total_threads = num_blocks as usize * threads_per_group;
    let simd_groups = threads_per_group.div_ceil(32);
    let shared_bytes = (simd_groups * std::mem::size_of::<EF>()) as u64;

    let pipeline = get_kernels().get_pipeline("whir_sumcheck_coeff_moments_round")?;
    let (grid, group) = grid_size_1d(total_threads, threads_per_group);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(f_coeffs.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(w_moments.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(tmp_block_sums.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &height as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })?;

    let d: u32 = 2;
    let pipeline_reduce = get_kernels().get_pipeline("final_reduce_block_sums")?;
    let final_threads = d as usize * threads_per_group;
    let (grid_reduce, group_reduce) = grid_size_1d(final_threads, threads_per_group);
    dispatch_sync(&pipeline_reduce, grid_reduce, group_reduce, |encoder| {
        encoder.set_buffer(0, Some(tmp_block_sums.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &num_blocks as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &d as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })
}

pub unsafe fn whir_fold_coeffs_and_moments(
    f_coeffs: &MetalBuffer<EF>,
    w_moments: &MetalBuffer<EF>,
    f_folded_coeffs: &mut MetalBuffer<EF>,
    w_folded_moments: &mut MetalBuffer<EF>,
    alpha: EF,
    height: u32,
) -> Result<(), MetalError> {
    debug_assert!(f_coeffs.len() >= height as usize);
    debug_assert!(w_moments.len() >= height as usize);
    debug_assert!(f_folded_coeffs.len() >= (height as usize / 2));
    debug_assert!(w_folded_moments.len() >= (height as usize / 2));
    let pipeline = get_kernels().get_pipeline("whir_fold_coeffs_and_moments")?;
    let half_height = height >> 1;
    let (grid, group) = grid_size_1d(half_height as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(f_coeffs.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(w_moments.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(f_folded_coeffs.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(w_folded_moments.gpu_buffer()), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<EF>() as u64,
            &alpha as *const EF as *const c_void,
        );
        encoder.set_bytes(5, 4, &half_height as *const u32 as *const c_void);
    })
}

pub unsafe fn w_moments_accumulate(
    w_moments: &mut MetalBuffer<EF>,
    z0_pows2: &MetalBuffer<EF>,
    z_pows2: &MetalBuffer<F>,
    gamma: EF,
    num_queries: u32,
    log_height: u32,
) -> Result<(), MetalError> {
    let height = w_moments.len();
    debug_assert_eq!(z0_pows2.len(), log_height as usize);
    debug_assert_eq!(z_pows2.len(), num_queries as usize * log_height as usize);
    let pipeline = get_kernels().get_pipeline("w_moments_accumulate")?;
    let height_u32 = height as u32;
    let (grid, group) = grid_size_1d(height, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(w_moments.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(z0_pows2.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(z_pows2.gpu_buffer()), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<EF>() as u64,
            &gamma as *const EF as *const c_void,
        );
        encoder.set_bytes(4, 4, &num_queries as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &log_height as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &height_u32 as *const u32 as *const c_void);
    })
}
