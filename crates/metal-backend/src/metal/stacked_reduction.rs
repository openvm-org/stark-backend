//! Stacked reduction kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/stacked_reduction.rs

#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use crate::{
    poly::EqEvalSegments,
    prelude::{D_EF, EF, F},
    stacked_reduction::{UnstackedSlice, STACKED_REDUCTION_S_DEG},
};

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};

/// Number of G outputs per z in round 0: G0, G1, G2
pub const NUM_G: usize = 3;

pub fn stacked_reduction_r0_required_temp_buffer_size(
    trace_height: u32,
    trace_width: u32,
    l_skip: u32,
) -> u32 {
    // Conservative estimate matching CUDA kernel
    let num_x = (trace_height >> l_skip).max(1);
    let blocks = (trace_width * num_x + 255) / 256;
    blocks * NUM_G as u32 * (1 << l_skip)
}

pub unsafe fn stacked_reduction_sumcheck_round0(
    eq_r_ns: &EqEvalSegments<EF>,
    trace_ptr: *const F,
    lambda_pows: *const EF,
    block_sums: &mut MetalBuffer<EF>,
    output: &mut MetalBuffer<EF>,
    height: usize,
    width: usize,
    l_skip: usize,
) -> Result<(), MetalError> {
    let output_size = NUM_G << l_skip;
    let num_x = (height >> l_skip).max(1) as u32;
    debug_assert!(output.len() >= output_size);

    let pipeline = get_kernels().get_pipeline("stacked_reduction_sumcheck_round0")?;
    let total = height * width;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let height_u32 = height as u32;
    let width_u32 = width as u32;
    let l_skip_u32 = l_skip as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_r_ns.buffer.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(block_sums.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(output.gpu_buffer()), 0);
        encoder.set_bytes(3, 4, &height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &width_u32 as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &l_skip_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &num_x as *const u32 as *const c_void);
    })
}

pub unsafe fn stacked_reduction_fold_ple(
    src: *const F,
    dst: *mut EF,
    omega_skip_pows: &MetalBuffer<F>,
    inv_lagrange_denoms: &MetalBuffer<EF>,
    trace_height: usize,
    trace_width: usize,
    l_skip: usize,
) -> Result<(), MetalError> {
    let skip_domain = 1 << l_skip;
    debug_assert!(omega_skip_pows.len() >= skip_domain);
    debug_assert!(inv_lagrange_denoms.len() >= skip_domain);
    let pipeline = get_kernels().get_pipeline("stacked_reduction_fold_ple")?;
    let new_height = trace_height.max(skip_domain) / skip_domain;
    let total = new_height * trace_width;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let trace_height_u32 = trace_height as u32;
    let trace_width_u32 = trace_width as u32;
    let l_skip_u32 = l_skip as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(omega_skip_pows.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(inv_lagrange_denoms.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &trace_height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &trace_width_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &l_skip_u32 as *const u32 as *const c_void);
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
    let total = (2usize << max_n) - 1; // skip index 0
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(eq_r_ns.buffer.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(k_rot_ns.gpu_buffer()), 0);
        encoder.set_bytes(2, std::mem::size_of::<EF>() as u64, &k_rot_uni_0 as *const EF as *const c_void);
        encoder.set_bytes(3, std::mem::size_of::<EF>() as u64, &k_rot_uni_1 as *const EF as *const c_void);
        encoder.set_bytes(4, 4, &max_n as *const u32 as *const c_void);
    })
}

pub unsafe fn stacked_reduction_sumcheck_mle_round(
    q_evals: &MetalBuffer<*const EF>,
    eq_r_ns: &EqEvalSegments<EF>,
    k_rot_ns: &EqEvalSegments<EF>,
    unstacked_cols: *const UnstackedSlice,
    lambda_pows: *const EF,
    output: &mut MetalBuffer<u64>,
    q_height: usize,
    window_len: usize,
    num_y: usize,
    sm_count: u32,
) -> Result<(), MetalError> {
    debug_assert!(output.len() >= STACKED_REDUCTION_S_DEG * D_EF);

    let pipeline = get_kernels().get_pipeline("stacked_reduction_sumcheck_mle_round")?;
    let total = q_height * window_len;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let q_height_u32 = q_height as u32;
    let window_len_u32 = window_len as u32;
    let num_y_u32 = num_y as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(q_evals.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_r_ns.buffer.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(k_rot_ns.buffer.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(output.gpu_buffer()), 0);
        encoder.set_bytes(4, 4, &q_height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &window_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &num_y_u32 as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &sm_count as *const u32 as *const c_void);
    })
}

pub unsafe fn stacked_reduction_sumcheck_mle_round_degenerate(
    q_evals: &MetalBuffer<*const EF>,
    eq_ub_ptr: &MetalBuffer<EF>,
    eq_r: EF,
    k_rot_r: EF,
    unstacked_cols: *const UnstackedSlice,
    lambda_pows: *const EF,
    output: &mut MetalBuffer<u64>,
    q_height: usize,
    window_len: usize,
    l_skip: usize,
    round: usize,
) -> Result<(), MetalError> {
    debug_assert!(output.len() >= STACKED_REDUCTION_S_DEG * D_EF);

    let pipeline =
        get_kernels().get_pipeline("stacked_reduction_sumcheck_mle_round_degenerate")?;
    let total = q_height * window_len;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let q_height_u32 = q_height as u32;
    let window_len_u32 = window_len as u32;
    let l_skip_u32 = l_skip as u32;
    let round_u32 = round as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(q_evals.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(eq_ub_ptr.gpu_buffer()), 0);
        encoder.set_bytes(2, std::mem::size_of::<EF>() as u64, &eq_r as *const EF as *const c_void);
        encoder.set_bytes(3, std::mem::size_of::<EF>() as u64, &k_rot_r as *const EF as *const c_void);
        encoder.set_buffer(4, Some(output.gpu_buffer()), 0);
        encoder.set_bytes(5, 4, &q_height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &window_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &l_skip_u32 as *const u32 as *const c_void);
        encoder.set_bytes(8, 4, &round_u32 as *const u32 as *const c_void);
    })
}
