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
        lg_domain_size: u32,
        alpha: EF,
        stream: cudaStream_t,
    ) -> i32;
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
    lg_domain_size: u32,
    alpha: EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_bit_rev_frac_ext_build_k2(
        inout.as_mut_raw_ptr(),
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
