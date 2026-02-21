//! Polynomial kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/poly.rs

#![allow(clippy::too_many_arguments)]

use std::{ffi::c_void, mem::size_of};

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};
use crate::{
    prelude::{D_EF, EF, F},
    KernelError,
};

pub unsafe fn algebraic_batch_matrices(
    output: &mut MetalBuffer<EF>,
    mat_ptrs: &MetalBuffer<*const F>,
    mu_powers: &MetalBuffer<EF>,
    mu_idxs: &MetalBuffer<u32>,
    widths: &MetalBuffer<u32>,
    height: usize,
    num_mats: usize,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("algebraic_batch_matrices")?;
    let (grid, group) = grid_size_1d(height, DEFAULT_THREADS_PER_GROUP);
    let height_u32 = height as u32;
    let num_mats_u32 = num_mats as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(output.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(mat_ptrs.gpu_buffer()), 0);
        encoder.set_buffer(2, Some(mu_powers.gpu_buffer()), 0);
        encoder.set_buffer(3, Some(mu_idxs.gpu_buffer()), 0);
        encoder.set_buffer(4, Some(widths.gpu_buffer()), 0);
        encoder.set_bytes(5, 4, &height_u32 as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &num_mats_u32 as *const u32 as *const c_void);
    })
}

pub unsafe fn eq_hypercube_stage_ext(
    out: &MetalBuffer<EF>,
    out_offset: usize,
    x_i: EF,
    step: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("eq_hypercube_stage_ext")?;
    let (grid, group) = grid_size_1d(step as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(
            0,
            Some(out.gpu_buffer()),
            (out_offset * size_of::<EF>()) as u64,
        );
        encoder.set_bytes(
            1,
            size_of::<EF>() as u64,
            &x_i as *const EF as *const c_void,
        );
        encoder.set_bytes(2, 4, &step as *const u32 as *const c_void);
    })
}

pub unsafe fn mobius_eq_hypercube_stage_ext(
    out: &MetalBuffer<EF>,
    out_offset: usize,
    omega_i: EF,
    step: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("mobius_eq_hypercube_stage_ext")?;
    let (grid, group) = grid_size_1d(step as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(
            0,
            Some(out.gpu_buffer()),
            (out_offset * size_of::<EF>()) as u64,
        );
        encoder.set_bytes(
            1,
            size_of::<EF>() as u64,
            &omega_i as *const EF as *const c_void,
        );
        encoder.set_bytes(2, 4, &step as *const u32 as *const c_void);
    })
}

pub unsafe fn eq_hypercube_nonoverlapping_stage_ext(
    out: &MetalBuffer<EF>,
    out_offset: usize,
    input: &MetalBuffer<EF>,
    input_offset: usize,
    x_i: EF,
    step: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("eq_hypercube_nonoverlapping_stage_ext")?;
    let (grid, group) = grid_size_1d(step as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(
            0,
            Some(out.gpu_buffer()),
            (out_offset * size_of::<EF>()) as u64,
        );
        encoder.set_buffer(
            1,
            Some(input.gpu_buffer()),
            (input_offset * size_of::<EF>()) as u64,
        );
        encoder.set_bytes(
            2,
            size_of::<EF>() as u64,
            &x_i as *const EF as *const c_void,
        );
        encoder.set_bytes(3, 4, &step as *const u32 as *const c_void);
    })
}

pub unsafe fn eq_hypercube_interleaved_stage_ext(
    out: &MetalBuffer<EF>,
    out_offset: usize,
    input: &MetalBuffer<EF>,
    input_offset: usize,
    x_i: EF,
    step: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("eq_hypercube_interleaved_stage_ext")?;
    let (grid, group) = grid_size_1d(step as usize, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(
            0,
            Some(out.gpu_buffer()),
            (out_offset * size_of::<EF>()) as u64,
        );
        encoder.set_buffer(
            1,
            Some(input.gpu_buffer()),
            (input_offset * size_of::<EF>()) as u64,
        );
        encoder.set_bytes(
            2,
            size_of::<EF>() as u64,
            &x_i as *const EF as *const c_void,
        );
        encoder.set_bytes(3, 4, &step as *const u32 as *const c_void);
    })
}

pub unsafe fn batch_eq_hypercube_stage(
    out: &mut MetalBuffer<F>,
    x: &MetalBuffer<F>,
    step: u32,
    height: u32,
) -> Result<(), MetalError> {
    let width = x.len() as u32;
    debug_assert!(step < height);
    debug_assert!(out.len() <= (width * height) as usize);
    let pipeline = get_kernels().get_pipeline("batch_eq_hypercube_stage")?;
    let total = (width * height) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(x.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &step as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &width as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &height as *const u32 as *const c_void);
    })
}

pub unsafe fn eval_poly_ext_at_point_from_base(
    base_coeffs: &MetalBuffer<F>,
    coeff_len: usize,
    x: EF,
) -> Result<EF, KernelError> {
    debug_assert!(base_coeffs.len() >= coeff_len * D_EF);
    let d_out = MetalBuffer::<EF>::with_capacity(1);
    let pipeline = get_kernels()
        .get_pipeline("eval_poly_ext_at_point")
        .map_err(KernelError::Kernel)?;
    let mut threads_per_group = if coeff_len <= 256 {
        64usize
    } else if coeff_len <= 4096 {
        128usize
    } else if coeff_len <= 65536 {
        256usize
    } else {
        512usize
    };
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
    threads_per_group = threads_per_group.min(max_threads).max(32);
    let simd_groups = threads_per_group.div_ceil(32);
    let shared_bytes = (simd_groups * std::mem::size_of::<EF>()) as u64;
    let coeff_len_u32 = coeff_len as u32;
    let (grid, group) = grid_size_1d(threads_per_group, threads_per_group);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(base_coeffs.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &coeff_len_u32 as *const u32 as *const c_void);
        encoder.set_bytes(
            2,
            std::mem::size_of::<EF>() as u64,
            &x as *const EF as *const c_void,
        );
        encoder.set_buffer(3, Some(d_out.gpu_buffer()), 0);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    })
    .map_err(KernelError::Kernel)?;
    let out = d_out.to_vec();
    Ok(out[0])
}

pub fn vector_scalar_multiply_ext(vec: &mut MetalBuffer<EF>, scalar: EF) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("vector_scalar_multiply_ext")?;
    let length = vec.len() as u32;
    let (grid, group) = grid_size_1d(vec.len(), DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(vec.gpu_buffer()), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<EF>() as u64,
            &scalar as *const EF as *const c_void,
        );
        encoder.set_bytes(2, 4, &length as *const u32 as *const c_void);
    })
}

pub unsafe fn transpose_fp_to_fpext_vec(
    output: &mut MetalBuffer<EF>,
    input: &MetalBuffer<F>,
) -> Result<(), MetalError> {
    let height = output.len();
    debug_assert_eq!(height * D_EF, input.len());
    let pipeline = get_kernels().get_pipeline("transpose_fp_to_fpext_vec")?;
    let height_u32 = height as u32;
    let (grid, group) = grid_size_1d(height, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(output.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(input.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &height_u32 as *const u32 as *const c_void);
    })
}
