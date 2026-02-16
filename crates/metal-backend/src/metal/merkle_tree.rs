//! Merkle tree / Poseidon2 kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/merkle_tree.rs

#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use crate::prelude::{Digest, EF, F};

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};

pub unsafe fn poseidon2_compressing_row_hashes(
    out: &mut MetalBuffer<Digest>,
    matrix: &MetalBuffer<F>,
    width: usize,
    query_stride: usize,
    log_rows_per_query: usize,
) -> Result<(), MetalError> {
    debug_assert!(matrix.len() >= width * (query_stride << log_rows_per_query));
    debug_assert!(out.len() >= query_stride);
    let pipeline = get_kernels().get_pipeline("poseidon2_compressing_row_hashes")?;
    let (grid, group) = grid_size_1d(query_stride, DEFAULT_THREADS_PER_GROUP);
    let width_u32 = width as u32;
    let query_stride_u32 = query_stride as u32;
    let log_rows_u32 = log_rows_per_query as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(matrix.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &width_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &query_stride_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &log_rows_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn poseidon2_compressing_row_hashes_ext(
    out: &mut MetalBuffer<Digest>,
    matrix: &MetalBuffer<EF>,
    width: usize,
    query_stride: usize,
    log_rows_per_query: usize,
) -> Result<(), MetalError> {
    debug_assert!(matrix.len() >= width * (query_stride << log_rows_per_query));
    debug_assert!(out.len() >= query_stride);
    let pipeline = get_kernels().get_pipeline("poseidon2_compressing_row_hashes_ext")?;
    let (grid, group) = grid_size_1d(query_stride, DEFAULT_THREADS_PER_GROUP);
    let width_u32 = width as u32;
    let query_stride_u32 = query_stride as u32;
    let log_rows_u32 = log_rows_per_query as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(matrix.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &width_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &query_stride_u32 as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &log_rows_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn poseidon2_strided_compress_layer(
    output: &mut MetalBuffer<Digest>,
    prev_layer: &MetalBuffer<Digest>,
    output_size: usize,
    stride: usize,
) -> Result<(), MetalError> {
    debug_assert!(stride > 0 && stride <= output_size);
    debug_assert!(output.len() >= output_size);
    debug_assert!(prev_layer.len() >= output_size * 2);
    let pipeline = get_kernels().get_pipeline("poseidon2_strided_compress_layer")?;
    let (grid, group) = grid_size_1d(output_size, DEFAULT_THREADS_PER_GROUP);
    let output_size_u32 = output_size as u32;
    let stride_u32 = stride as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(output.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(prev_layer.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &output_size_u32 as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &stride_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn poseidon2_adjacent_compress_layer(
    output: &mut MetalBuffer<Digest>,
    prev_layer: &MetalBuffer<Digest>,
    output_size: usize,
) -> Result<(), MetalError> {
    debug_assert!(output.len() >= output_size);
    debug_assert!(prev_layer.len() >= output_size * 2);
    let pipeline = get_kernels().get_pipeline("poseidon2_adjacent_compress_layer")?;
    let (grid, group) = grid_size_1d(output_size, DEFAULT_THREADS_PER_GROUP);
    let output_size_u32 = output_size as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(output.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(prev_layer.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &output_size_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn query_digest_layers(
    d_digest_matrix: &mut MetalBuffer<F>,
    d_layers_ptr: &MetalBuffer<u64>,
    d_indices: &MetalBuffer<u64>,
    num_query: u64,
    num_layer: u64,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("query_digest_layers")?;
    let total = (num_query * num_layer) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_digest_matrix.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(d_layers_ptr.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(d_indices.gpu_buffer()), 0);
        encoder.set_bytes(3, 8, &num_query as *const u64 as *const c_void);
        encoder.set_bytes(4, 8, &num_layer as *const u64 as *const c_void);
    })
}
