//! MLE interpolation kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/mle_interpolate.rs

#![allow(clippy::too_many_arguments)]

use std::{ffi::c_void, mem::size_of};

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use super::{
    dispatch_sync, get_kernels, grid_size_1d, grid_size_2d, DEFAULT_THREADS_PER_GROUP,
    LOG_SIMD_SIZE,
};
use crate::prelude::{EF, F};

pub unsafe fn mle_interpolate_stage(
    buffer: &mut MetalBuffer<F>,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("mle_interpolate_stage")?;
    let total_len = buffer.len();
    let (grid, group) = grid_size_1d(total_len / 2, DEFAULT_THREADS_PER_GROUP);
    let total_len_u32 = total_len as u32;
    let is_e2c: u32 = if is_eval_to_coeff { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &total_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &step as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &is_e2c as *const u32 as *const c_void);
    })
}

pub unsafe fn mle_interpolate_stage_ext(
    buffer: &mut MetalBuffer<EF>,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("mle_interpolate_stage_ext")?;
    let total_len = buffer.len();
    let (grid, group) = grid_size_1d(total_len / 2, DEFAULT_THREADS_PER_GROUP);
    let total_len_u32 = total_len as u32;
    let is_e2c: u32 = if is_eval_to_coeff { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &total_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &step as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &is_e2c as *const u32 as *const c_void);
    })
}

pub unsafe fn mle_interpolate_stage_2d(
    buffer: &mut MetalBuffer<F>,
    width: u16,
    height: u32,
    padded_height: u32,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), MetalError> {
    debug_assert!(height <= padded_height);
    debug_assert_eq!(padded_height % (step * 2), 0);
    let pipeline = get_kernels().get_pipeline("mle_interpolate_stage_2d")?;
    let span = step * 2;
    let (grid, group) = grid_size_2d(
        (height as usize) / 2,
        width as usize,
        DEFAULT_THREADS_PER_GROUP,
        1,
    );
    let is_e2c: u32 = if is_eval_to_coeff { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &padded_height as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &span as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &step as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &is_e2c as *const u32 as *const c_void);
    })
}

pub unsafe fn mle_interpolate_fused_2d(
    buffer: &mut MetalBuffer<F>,
    width: u16,
    padded_height: u32,
    log_stride: u32,
    start_step: u32,
    num_stages: u32,
    is_eval_to_coeff: bool,
    right_pad: bool,
) -> Result<(), MetalError> {
    debug_assert!((1..=LOG_SIMD_SIZE as u32).contains(&num_stages));
    debug_assert!((start_step << (num_stages - 1)) <= 16);
    let pipeline = get_kernels().get_pipeline("mle_interpolate_fused_2d")?;
    let meaningful_count = (padded_height >> log_stride) as usize;
    let (grid, group) = grid_size_2d(
        meaningful_count,
        width as usize,
        DEFAULT_THREADS_PER_GROUP,
        1,
    );
    let is_e2c: u32 = if is_eval_to_coeff { 1 } else { 0 };
    let rp: u32 = if right_pad { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &padded_height as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &log_stride as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &start_step as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &num_stages as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &is_e2c as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &rp as *const u32 as *const c_void);
    })
}

/// Tile log size for shared memory kernel (must match Metal kernel's MLE_SHARED_TILE_LOG_SIZE).
pub const MLE_SHARED_TILE_LOG_SIZE: u32 = 12;

pub unsafe fn mle_interpolate_shared_2d(
    buffer: &mut MetalBuffer<F>,
    width: u16,
    padded_height: u32,
    log_stride: u32,
    start_log_step: u32,
    end_log_step: u32,
    is_eval_to_coeff: bool,
    right_pad: bool,
) -> Result<(), MetalError> {
    debug_assert!(end_log_step < MLE_SHARED_TILE_LOG_SIZE);
    let pipeline = get_kernels().get_pipeline("mle_interpolate_shared_2d")?;
    let tile_size = 1u32 << MLE_SHARED_TILE_LOG_SIZE;
    let meaningful_count = padded_height >> log_stride;
    let num_tiles = meaningful_count.div_ceil(tile_size);
    let (grid, group) = grid_size_2d(
        num_tiles as usize * DEFAULT_THREADS_PER_GROUP,
        width as usize,
        DEFAULT_THREADS_PER_GROUP,
        1,
    );
    let shared_bytes = (tile_size + tile_size / 32) as usize * size_of::<F>();
    let is_e2c: u32 = if is_eval_to_coeff { 1 } else { 0 };
    let rp: u32 = if right_pad { 1 } else { 0 };
    let tile_log_size = MLE_SHARED_TILE_LOG_SIZE;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &padded_height as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &log_stride as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &start_log_step as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &end_log_step as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &is_e2c as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &rp as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &tile_log_size as *const u32 as *const c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes as u64);
    })
}
