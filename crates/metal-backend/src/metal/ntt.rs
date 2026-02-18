//! NTT kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/ntt.rs

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::MTLSize;
use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};

use crate::prelude::{EF, F};

use super::{dispatch_sync, get_kernels, grid_size_1d, grid_size_2d, DEFAULT_THREADS_PER_GROUP};

const ROOT_TABLE_SIZE: usize = 28;
static FORWARD_ROOTS: OnceLock<MetalBuffer<F>> = OnceLock::new();
static INVERSE_ROOTS: OnceLock<MetalBuffer<F>> = OnceLock::new();

fn build_roots(inverse: bool) -> MetalBuffer<F> {
    let mut roots = Vec::with_capacity(ROOT_TABLE_SIZE);
    roots.push(F::ONE);
    for level in 1..ROOT_TABLE_SIZE {
        let mut root = F::two_adic_generator(level);
        if inverse {
            root = root.inverse();
        }
        roots.push(root);
    }
    MetalBuffer::from_slice(&roots)
}

fn ntt_roots(inverse: bool) -> &'static MetalBuffer<F> {
    if inverse {
        INVERSE_ROOTS.get_or_init(|| build_roots(true))
    } else {
        FORWARD_ROOTS.get_or_init(|| build_roots(false))
    }
}

pub fn ensure_ntt_roots_initialized(inverse: bool) {
    let _ = ntt_roots(inverse);
}

pub unsafe fn generate_all_twiddles<T>(
    twiddles: &MetalBuffer<T>,
    inverse: bool,
) -> Result<(), MetalError> {
    let _ = inverse;
    let pipeline = get_kernels().get_pipeline("generate_all_twiddles")?;
    let (grid, group) = grid_size_1d(twiddles.len(), DEFAULT_THREADS_PER_GROUP);
    let max_level: u32 = 31;
    let total_size: u32 = twiddles.len() as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(twiddles.gpu_buffer()), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &max_level as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &total_size as *const u32 as *const c_void,
        );
    })
}

pub unsafe fn generate_partial_twiddles<T>(
    partial_twiddles: &MetalBuffer<T>,
    inverse: bool,
) -> Result<(), MetalError> {
    let _ = inverse;
    let pipeline = get_kernels().get_pipeline("generate_partial_twiddles")?;
    let (grid, group) = grid_size_1d(partial_twiddles.len(), DEFAULT_THREADS_PER_GROUP);
    let level: u32 = 1;
    let window_size: u32 = partial_twiddles.len() as u32;
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(partial_twiddles.gpu_buffer()), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &level as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &window_size as *const u32 as *const c_void,
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
    let domain_size = 1usize << lg_domain_size;
    let (grid, group) = grid_size_2d(domain_size, poly_count as usize, 64, 4);
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
    let domain_size = 1usize << lg_domain_size;
    let (grid, group) = grid_size_2d(domain_size, poly_count as usize, 64, 4);
    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(d_inp.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &lg_domain_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &padded_poly_size as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &poly_count as *const u32 as *const c_void);
    })
}

pub unsafe fn bit_rev_frac_ext(
    d_out: &MetalBuffer<Frac<EF>>,
    d_inp: &MetalBuffer<Frac<EF>>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
) -> Result<(), MetalError> {
    debug_assert_eq!(d_out.len(), d_inp.len());
    let pipeline = get_kernels().get_pipeline("bit_rev_ext")?;
    let domain_size = 1usize << lg_domain_size;
    let (grid, group) = grid_size_2d(domain_size, poly_count as usize, 64, 4);
    let q_offset_out = (d_out.len() * std::mem::size_of::<EF>()) as u64;
    let q_offset_in = (d_inp.len() * std::mem::size_of::<EF>()) as u64;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_out.gpu_buffer()), 0);
        encoder.set_buffer(1, Some(d_inp.gpu_buffer()), 0);
        encoder.set_bytes(2, 4, &lg_domain_size as *const u32 as *const c_void);
        encoder.set_bytes(3, 4, &padded_poly_size as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &poly_count as *const u32 as *const c_void);
    })?;

    dispatch_sync(&pipeline, grid, group, |encoder| {
        encoder.set_buffer(0, Some(d_out.gpu_buffer()), q_offset_out);
        encoder.set_buffer(1, Some(d_inp.gpu_buffer()), q_offset_in);
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
    let _ = radix;
    let _ = padded_poly_size;
    let pipeline_name = if is_intt {
        "gs_mixed_radix_narrow"
    } else {
        "ntt_forward_step"
    };
    let pipeline = get_kernels().get_pipeline(pipeline_name)?;
    let roots = ntt_roots(is_intt);
    let n_bits = lg_domain_size;

    for round in 0..iterations {
        let s_bits = if is_intt {
            lg_domain_size - stage - round
        } else {
            stage + round + 1
        };
        let s_size = 1u64 << (s_bits - 1);
        let g_size = 1u64 << (n_bits - s_bits);
        let grid = MTLSize::new(s_size, g_size, poly_count as u64);
        let group = MTLSize::new(s_size.min(32), 1, 1);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(d_inout.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(roots.gpu_buffer()), 0);
            encoder.set_bytes(2, 4, &n_bits as *const u32 as *const c_void);
            encoder.set_bytes(3, 4, &s_bits as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &poly_count as *const u32 as *const c_void);
        })?;
    }

    Ok(())
}
