//! NTT kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/ntt.rs

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;

use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

use crate::prelude::{EF, F};

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};

pub unsafe fn generate_all_twiddles<T>(
    twiddles: &MetalBuffer<T>,
    inverse: bool,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("generate_all_twiddles")?;
    let (grid, group) = grid_size_1d(twiddles.len(), DEFAULT_THREADS_PER_GROUP);
    let inverse_u32: u32 = if inverse { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(twiddles.gpu_buffer()), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &inverse_u32 as *const u32 as *const c_void,
        );
    })
}

pub unsafe fn generate_partial_twiddles<T>(
    partial_twiddles: &MetalBuffer<T>,
    inverse: bool,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("generate_partial_twiddles")?;
    let (grid, group) = grid_size_1d(partial_twiddles.len(), DEFAULT_THREADS_PER_GROUP);
    let inverse_u32: u32 = if inverse { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(partial_twiddles.gpu_buffer()), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &inverse_u32 as *const u32 as *const c_void,
        );
    })
}

pub unsafe fn bit_rev(
    d_out: &MetalBuffer<F>,
    d_inp: &MetalBuffer<F>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("bit_rev")?;
    let total = (padded_poly_size * poly_count) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(d_inp.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &lg_domain_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &padded_poly_size as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &poly_count as *const u32 as *const c_void);
    })
}

pub unsafe fn bit_rev_ext(
    d_out: &MetalBuffer<EF>,
    d_inp: &MetalBuffer<EF>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("bit_rev_ext")?;
    let total = (padded_poly_size * poly_count) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(d_inp.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &lg_domain_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &padded_poly_size as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &poly_count as *const u32 as *const c_void);
    })
}

pub unsafe fn bit_rev_frac_ext(
    d_out: &MetalBuffer<(EF, EF)>,
    d_inp: &MetalBuffer<(EF, EF)>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("bit_rev_frac_ext")?;
    let total = (padded_poly_size * poly_count) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(d_inp.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &lg_domain_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &padded_poly_size as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &poly_count as *const u32 as *const c_void);
    })
}

pub unsafe fn ct_mixed_radix_narrow(
    d_inout: &MetalBuffer<F>,
    radix: u32,
    lg_domain_size: u32,
    stage: u32,
    iterations: u32,
    padded_poly_size: u32,
    poly_count: u32,
    is_intt: bool,
) -> Result<(), MetalError> {
    let pipeline = get_kernels().get_pipeline("ct_mixed_radix_narrow")?;
    let total = (padded_poly_size * poly_count) as usize;
    let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
    let is_intt_u32: u32 = if is_intt { 1 } else { 0 };
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_inout.gpu_buffer()), 0);
        encoder.set_bytes(1, 4, &radix as *const u32 as *const c_void);
        encoder.set_bytes(2, 4, &lg_domain_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &stage as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &iterations as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &padded_poly_size as *const u32 as *const c_void);
        encoder.set_bytes(6, 4, &poly_count as *const u32 as *const c_void);
        encoder.set_bytes(7, 4, &is_intt_u32 as *const u32 as *const c_void);
    })
}
