#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::EF;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

pub mod sumcheck {
    use super::*;

    extern "C" {
        fn _sumcheck_mle_round(
            input_matrices: *const usize,
            output: *mut std::ffi::c_void,
            tmp_block_sums: *mut std::ffi::c_void,
            widths: *const u32,
            num_matrices: u32,
            height: u32,
            d: u32,
        ) -> i32;

        fn _fold_mle(
            input_matrices: *const usize,
            output_matrices: *const usize,
            widths: *const u32,
            num_matrices: u32,
            output_height: u32,
            r_val: EF,
        ) -> i32;

        fn _sumcheck_ple_round0(
            input_matrices: *const usize,
            output: *mut std::ffi::c_void,
            widths: *const u32,
            rotations: *const i32,
            num_matrices: u32,
            height: u32,
            domain_size: u32,
        ) -> i32;
    }

    pub unsafe fn sumcheck_mle_round<T>(
        input_matrices: &DeviceBuffer<usize>,
        output: &DeviceBuffer<T>,
        tmp_block_sums: &DeviceBuffer<T>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u32,
        height: u32,
        d: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_sumcheck_mle_round(
            input_matrices.as_ptr(),
            output.as_mut_raw_ptr(),
            tmp_block_sums.as_mut_raw_ptr(),
            widths.as_ptr(),
            num_matrices,
            height,
            d,
        ))
    }

    pub unsafe fn fold_mle(
        input_matrices: &DeviceBuffer<usize>,
        output_matrices: &DeviceBuffer<usize>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u32,
        output_height: u32,
        r_val: EF,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fold_mle(
            input_matrices.as_ptr(),
            output_matrices.as_ptr(),
            widths.as_ptr(),
            num_matrices,
            output_height,
            r_val,
        ))
    }

    // pub unsafe fn sumcheck_ple_round0<T>(
    //     input_matrices: &DeviceBuffer<usize>,
    //     output: &DeviceBuffer<T>,
    //     widths: &DeviceBuffer<u32>,
    //     rotations: &DeviceBuffer<i32>,
    //     num_matrices: u32,
    //     height: u32,
    //     domain_size: u32,
    // ) -> Result<(), CudaError> {
    //     CudaError::from_result(_sumcheck_ple_round0(
    //         input_matrices.as_ptr(),
    //         output.as_mut_raw_ptr(),
    //         widths.as_ptr(),
    //         rotations.as_ptr(),
    //         num_matrices,
    //         height,
    //         domain_size,
    //     ))
    // }
}
