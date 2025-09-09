#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

pub mod ntt {
    use super::*;

    // relate to supra/ntt_params.cu
    extern "C" {
        fn _generate_all_twiddles(twiddles: *mut std::ffi::c_void, inverse: bool) -> i32;
        fn _generate_partial_twiddles(
            partial_twiddles: *mut std::ffi::c_void,
            inverse: bool,
        ) -> i32;
    }

    pub unsafe fn generate_all_twiddles<F>(
        twiddles: &DeviceBuffer<F>,
        inverse: bool,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_generate_all_twiddles(twiddles.as_mut_raw_ptr(), inverse))
    }

    pub unsafe fn generate_partial_twiddles<F>(
        partial_twiddles: &DeviceBuffer<F>,
        inverse: bool,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_generate_partial_twiddles(
            partial_twiddles.as_mut_raw_ptr(),
            inverse,
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
        ) -> i32;
    }

    pub unsafe fn bit_rev<F>(
        d_out: &DeviceBuffer<F>,
        d_inp: &DeviceBuffer<F>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_bit_rev(
            d_out.as_mut_raw_ptr(),
            d_inp.as_raw_ptr(),
            lg_domain_size,
            padded_poly_size,
            poly_count,
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
            d_partial_twiddles: *const std::ffi::c_void,
            d_radix_twiddles: *const std::ffi::c_void,
            twiddles_offset: u32,
            is_intt: bool,
        ) -> i32;
    }

    pub unsafe fn ct_mixed_radix_narrow<F, T>(
        d_inout: &DeviceBuffer<F>,
        radix: u32,
        lg_domain_size: u32,
        stage: u32,
        iterations: u32,
        padded_poly_size: u32,
        poly_count: u32,
        d_partial_twiddles: &DeviceBuffer<T>,
        d_radix_twiddles: &DeviceBuffer<F>,
        twiddles_offset: u32,
        is_intt: bool,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_ct_mixed_radix_narrow(
            d_inout.as_mut_raw_ptr(),
            radix,
            lg_domain_size,
            stage,
            iterations,
            padded_poly_size,
            poly_count,
            d_partial_twiddles.as_raw_ptr(),
            d_radix_twiddles.as_raw_ptr(),
            twiddles_offset,
            is_intt,
        ))
    }
}
