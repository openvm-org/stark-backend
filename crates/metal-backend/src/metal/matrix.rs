//! Matrix kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/matrix.rs

#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use crate::prelude::{EF, F};

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};

pub unsafe fn matrix_transpose_fp(
    output: &MetalBuffer<F>,
    input: &MetalBuffer<F>,
    width: usize,
    height: usize,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("matrix_transpose_fp")?;
    let total = width * height;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let col_size = width as u32;
    let row_size = height as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(output.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(input.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &col_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &row_size as *const u32 as *const c_void);
    })
}

pub unsafe fn matrix_transpose_fpext(
    output: &MetalBuffer<EF>,
    input: &MetalBuffer<EF>,
    width: usize,
    height: usize,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("matrix_transpose_fpext")?;
    let total = width * height;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let col_size = width as u32;
    let row_size = height as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(output.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(input.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &col_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &row_size as *const u32 as *const c_void);
    })
}

pub unsafe fn matrix_get_rows_fp_kernel(
    output: &MetalBuffer<F>,
    input: &MetalBuffer<F>,
    row_indices: &MetalBuffer<u32>,
    matrix_width: u64,
    matrix_height: u64,
    row_indices_len: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("matrix_get_rows_fp")?;
    let total = (row_indices_len as u64 * matrix_width) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(output.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(input.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(row_indices.gpu_buffer()), 0);
        encoder.set_bytes(3, 8, &matrix_width as *const u64 as *const c_void);
        encoder.set_bytes(4, 8, &matrix_height as *const u64 as *const c_void);
        encoder.set_bytes(5, 4, &row_indices_len as *const u32 as *const c_void);
    })
}

pub unsafe fn split_ext_to_base_col_major_matrix(
    d_matrix: &mut MetalBuffer<F>,
    d_poly: &MetalBuffer<EF>,
    poly_len: u64,
    matrix_height: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("split_ext_to_base_col_major_matrix")?;
    let total = poly_len as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_matrix.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(d_poly.gpu_buffer()), 0);
        encoder.set_bytes(2, 8, &poly_len as *const u64 as *const c_void);
        encoder.set_bytes(3, 4, &matrix_height as *const u32 as *const c_void);
    })
}

pub unsafe fn batch_rotate_pad(
    output: *mut F,
    input: *const F,
    width: u32,
    num_x: u32,
    domain_size: u32,
    padded_size: u32,
) -> Result<(), MetalError> {
    debug_assert!(domain_size <= padded_size);
    debug_assert!(width.checked_mul(num_x).unwrap() < u16::MAX as u32 * u16::MAX as u32);
    let pipeline = get_kernels().get_pipeline("batch_rotate_pad")?;
    let total = (padded_size * width * num_x) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &num_x as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &domain_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &padded_size as *const u32 as *const c_void);
    })
}

pub unsafe fn lift_padded_matrix_evals(
    matrix: *mut F,
    width: u32,
    height: u32,
    lifted_height: u32,
    padded_height: u32,
) -> Result<(), MetalError> {
    debug_assert!(height <= lifted_height && lifted_height <= padded_height);
    let pipeline = get_kernels().get_pipeline("lift_padded_matrix_evals")?;
    let total = (lifted_height * width) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &height as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &lifted_height as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &padded_height as *const u32 as *const c_void);
    })
}

pub unsafe fn collapse_strided_matrix(
    output: *mut F,
    input: *const F,
    width: u32,
    height: u32,
    stride: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("collapse_strided_matrix")?;
    let total = (height * width) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &height as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &stride as *const u32 as *const c_void);
    })
}

pub unsafe fn batch_expand_pad(
    output: *mut F,
    input: *const F,
    poly_count: u32,
    out_size: u32,
    in_size: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("batch_expand_pad")?;
    let total = (out_size * poly_count) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &poly_count as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &out_size as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &in_size as *const u32 as *const c_void);
    })
}

pub unsafe fn batch_expand_pad_wide(
    out: *mut F,
    input: *const F,
    width: u32,
    padded_height: u32,
    height: u32,
) -> Result<(), MetalError> {
    debug_assert!(padded_height > height);
    let pipeline = get_kernels().get_pipeline("batch_expand_pad_wide")?;
    let total = (padded_height * width) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_bytes(0, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(1, 4, &padded_height as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &height as *const u32 as *const c_void);
    })
}
