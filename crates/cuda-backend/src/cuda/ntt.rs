#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::cudaStream_t};

use crate::prelude::{EF, F};

pub const MAX_CUDA_NTT_LOG_DOMAIN_SIZE: u32 = 27;

// relate to supra/ntt_params.cu
extern "C" {
    fn _generate_all_twiddles(
        twiddles: *mut std::ffi::c_void,
        inverse: bool,
        stream: cudaStream_t,
    ) -> i32;
    fn _generate_partial_twiddles(
        partial_twiddles: *mut std::ffi::c_void,
        inverse: bool,
        stream: cudaStream_t,
    ) -> i32;
}

pub unsafe fn generate_all_twiddles<F>(
    twiddles: &DeviceBuffer<F>,
    inverse: bool,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_generate_all_twiddles(
        twiddles.as_mut_raw_ptr(),
        inverse,
        stream,
    ))
}

pub unsafe fn generate_partial_twiddles<F>(
    partial_twiddles: &DeviceBuffer<F>,
    inverse: bool,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_generate_partial_twiddles(
        partial_twiddles.as_mut_raw_ptr(),
        inverse,
        stream,
    ))
}

// relate to supra/ntt_bitrev.cu
extern "C" {
    fn _bit_rev(
        d_out: *mut std::ffi::c_void,
        d_inp: *const std::ffi::c_void,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _bit_rev_ext(
        d_out: *mut std::ffi::c_void,
        d_inp: *const std::ffi::c_void,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _bit_rev_frac_ext(
        d_out: *mut std::ffi::c_void,
        d_inp: *const std::ffi::c_void,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        stream: cudaStream_t,
    ) -> i32;

    /// Fused bitrev + K=2 tree build for a single frac_fpext_t buffer.
    /// Applies alpha to denominators and fuses tree layers 0 and 1 in shmem.
    /// Requires domain_size >= 256.
    fn _bit_rev_frac_ext_build_k2(
        inout: *mut std::ffi::c_void,
        real_len: usize,
        lg_domain_size: u32,
        alpha: EF,
        stream: cudaStream_t,
    ) -> i32;

    fn _bit_rev_zeta_fused(
        d_inout: *mut std::ffi::c_void,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        l_skip: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _bit_rev_expand_pad(
        d_out: *mut std::ffi::c_void,
        d_inp: *const std::ffi::c_void,
        lg_domain_size: u32,
        out_stride: u32,
        in_stride: u32,
        poly_count: u32,
        src_len: u32,
        stream: cudaStream_t,
    ) -> i32;
}

/// Fused subset-zeta transform (coeff-to-eval stages over the low `l_skip` index bits)
/// plus in-place bit-reversal permutation, per column. Produces the identical buffer
/// contents as `mle_interpolate_stages(0..l_skip-1)` followed by [`bit_rev`], in one
/// read+write pass.
///
/// # Safety
/// - `d_inout` must be a device buffer of `poly_count` columns with stride `padded_poly_size >=
///   2^lg_domain_size`.
/// - Requires `l_skip <= 6` and `2^lg_domain_size >= 4096` (Z-tile kernel bounds); the launcher
///   returns an error otherwise.
pub unsafe fn bit_rev_zeta_fused(
    d_inout: &DeviceBuffer<F>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    l_skip: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_bit_rev_zeta_fused(
        d_inout.as_mut_raw_ptr(),
        lg_domain_size,
        padded_poly_size,
        poly_count,
        l_skip,
        stream,
    ))
}

/// Fused zero-extend + bit-reversal: `out[bit_rev(i)] = if i < src_len { in[i] } else { 0 }`
/// per column. Produces the identical buffer contents as `batch_expand_pad` followed by
/// [`bit_rev`], in one pass. `d_out` and `d_inp` must not alias.
///
/// # Safety
/// - `d_out` must have `poly_count` columns of stride `out_stride >= 2^lg_domain_size`; `d_inp`
///   must have `poly_count` columns of stride `in_stride >= src_len`.
/// - Requires `2^lg_domain_size >= 4096` (Z-tile kernel bounds); the launcher returns an error
///   otherwise.
#[allow(clippy::too_many_arguments)]
pub unsafe fn bit_rev_expand_pad(
    d_out: &DeviceBuffer<F>,
    d_inp: &DeviceBuffer<F>,
    lg_domain_size: u32,
    out_stride: u32,
    in_stride: u32,
    poly_count: u32,
    src_len: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_bit_rev_expand_pad(
        d_out.as_mut_raw_ptr(),
        d_inp.as_raw_ptr(),
        lg_domain_size,
        out_stride,
        in_stride,
        poly_count,
        src_len,
        stream,
    ))
}

pub unsafe fn bit_rev(
    d_out: &DeviceBuffer<F>,
    d_inp: &DeviceBuffer<F>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_bit_rev(
        d_out.as_mut_raw_ptr(),
        d_inp.as_raw_ptr(),
        lg_domain_size,
        padded_poly_size,
        poly_count,
        stream,
    ))
}

pub unsafe fn bit_rev_ext(
    d_out: &DeviceBuffer<EF>,
    d_inp: &DeviceBuffer<EF>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_bit_rev_ext(
        d_out.as_mut_raw_ptr(),
        d_inp.as_raw_ptr(),
        lg_domain_size,
        padded_poly_size,
        poly_count,
        stream,
    ))
}

pub unsafe fn bit_rev_frac_ext(
    d_out: &DeviceBuffer<(EF, EF)>,
    d_inp: &DeviceBuffer<(EF, EF)>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_bit_rev_frac_ext(
        d_out.as_mut_raw_ptr(),
        d_inp.as_raw_ptr(),
        lg_domain_size,
        padded_poly_size,
        poly_count,
        stream,
    ))
}

pub unsafe fn bit_rev_frac_ext_build_k2(
    inout: &DeviceBuffer<(EF, EF)>,
    real_len: usize,
    lg_domain_size: u32,
    alpha: EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_bit_rev_frac_ext_build_k2(
        inout.as_mut_raw_ptr(),
        real_len,
        lg_domain_size,
        alpha,
        stream,
    ))
}

// relate to supra/ntt.cu
extern "C" {
    fn _ct_mixed_radix_narrow(
        d_inout: *mut std::ffi::c_void,
        radix: u32,
        lg_domain_size: u32,
        stage: u32,
        iterations: u32,
        padded_poly_size: u32,
        poly_count: u32,
        is_intt: bool,
        stream: cudaStream_t,
    ) -> i32;
}

pub unsafe fn ct_mixed_radix_narrow(
    d_inout: &DeviceBuffer<F>,
    radix: u32,
    lg_domain_size: u32,
    stage: u32,
    iterations: u32,
    padded_poly_size: u32,
    poly_count: u32,
    is_intt: bool,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    if lg_domain_size > MAX_CUDA_NTT_LOG_DOMAIN_SIZE {
        return Err(CudaError::new(1));
    }
    CudaError::from_result(_ct_mixed_radix_narrow(
        d_inout.as_mut_raw_ptr(),
        radix,
        lg_domain_size,
        stage,
        iterations,
        padded_poly_size,
        poly_count,
        is_intt,
        stream,
    ))
}
