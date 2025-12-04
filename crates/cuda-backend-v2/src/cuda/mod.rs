#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use stark_backend_v2::prover::fractional_sumcheck_gkr::Frac;

use crate::{EF, F};

pub mod logup_zerocheck;
pub mod matrix;
pub mod merkle_tree;
pub mod poly;
pub mod stacked_reduction;
pub mod whir;

pub mod sumcheck {
    use std::ffi::c_void;

    use super::*;
    use crate::poly::EqEvalSegments;

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

        fn _reduce_over_x_and_cols(
            input: *const std::ffi::c_void,
            output: *mut std::ffi::c_void,
            num_x: u32,
            num_cols: u32,
            large_domain_size: u32,
        ) -> i32;

        fn _fold_ple_from_coeffs(
            input_coeffs: *const std::ffi::c_void,
            output: *mut std::ffi::c_void,
            num_x: u32,
            width: u32,
            domain_size: u32,
            r: EF,
        ) -> i32;

        fn _fold_ple_from_evals_no_expansion(
            input_evals: *const std::ffi::c_void,
            output: *mut std::ffi::c_void,
            num_x: u32,
            width: u32,
            domain_size: u32,
            lagrange_coeffs: *const std::ffi::c_void,
        ) -> i32;

        fn _triangular_fold_mle(output: *mut EF, input: *const EF, r: EF, output_max_n: u32)
        -> i32;
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

    pub unsafe fn fold_ple_from_coeffs(
        input_coeffs: *const F, // Base field (F)
        output: *mut EF,        // Extension field (EF)
        num_x: u32,
        width: u32,
        domain_size: u32,
        r: EF,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fold_ple_from_coeffs(
            input_coeffs as *const c_void,
            output as *mut c_void,
            num_x,
            width,
            domain_size,
            r,
        ))
    }

    pub unsafe fn fold_ple_from_evals(
        input_evals: *const F, // Base field (F)
        output: *mut EF,       // Extension field (EF)
        num_x: u32,
        width: u32,
        domain_size: u32,
        lagrange_coeffs: *const EF,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fold_ple_from_evals_no_expansion(
            input_evals as *const c_void,
            output as *mut c_void,
            num_x,
            width,
            domain_size,
            lagrange_coeffs as *const c_void,
        ))
    }

    pub unsafe fn reduce_over_x_and_cols<T>(
        input: &DeviceBuffer<T>,
        output: &DeviceBuffer<T>,
        num_x: u32,
        num_cols: u32,
        large_domain_size: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_reduce_over_x_and_cols(
            input.as_raw_ptr(),
            output.as_mut_raw_ptr(),
            num_x,
            num_cols,
            large_domain_size,
        ))
    }

    /// Folds the segments of `input` onto `output` using random element `r`.
    ///
    /// # Safety
    /// - `output` must have max `n` equal to `output_max_n`, for total length `2 * 2^output_max_n`.
    /// - `input` must have length `2 * 2^{output_max_n + 1}`.
    pub unsafe fn triangular_fold_mle(
        output: &mut EqEvalSegments<EF>,
        input: &EqEvalSegments<EF>,
        r: EF,
        output_max_n: usize,
    ) -> Result<(), CudaError> {
        debug_assert_eq!(output.buffer.len(), 2 << output_max_n);
        debug_assert_eq!(input.buffer.len(), 4 << output_max_n);
        CudaError::from_result(_triangular_fold_mle(
            output.buffer.as_mut_ptr(),
            input.buffer.as_ptr(),
            r,
            output_max_n as u32,
        ))
    }
}
