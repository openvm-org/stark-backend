#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::cudaStream_t};
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;

use crate::prelude::{EF, F};

pub mod batch_ntt_small;
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub mod bn254_merkle_tree;
pub mod device_info;
pub mod logup_zerocheck;
pub mod matrix;
pub mod merkle_tree;
pub mod mle_interpolate;
pub mod ntt;
pub mod poly;
pub mod sponge;
pub mod stacked_reduction;
pub mod whir;

/// Log of warp size (32)
pub const LOG_WARP_SIZE: usize = 5;

pub mod sumcheck {
    use std::ffi::c_void;

    use super::*;
    use crate::poly::EqEvalSegments;

    const MAX_SUMCHECK_MLE_ROUND_D: u32 = 5;

    extern "C" {
        fn _sumcheck_mle_round(
            input_matrices: *const *const EF,
            output: *mut EF,
            tmp_block_sums: *mut EF,
            widths: *const u32,
            num_matrices: u32,
            height: u32,
            d: u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _fold_mle(
            input_matrices: *const *const EF,
            output_matrices: *const *mut EF,
            widths: *const u32,
            num_matrices: u16,
            output_height: u32,
            max_output_cells: u32,
            r_val: EF,
            stream: cudaStream_t,
        ) -> i32;

        fn _fold_mle_column(
            buffer: *mut std::ffi::c_void,
            size: usize,
            r: EF,
            stream: cudaStream_t,
        ) -> i32;

        fn _batch_fold_mle(
            input_matrices: *const *const EF,
            output_matrices: *const *mut EF,
            widths: *const u32,
            num_matrices: u16,
            log_output_heights: *const u8,
            max_output_cells: u32,
            r_val: EF,
            stream: cudaStream_t,
        ) -> i32;

        fn _batch_fold_mle_dev_challenge(
            input_matrices: *const *const EF,
            output_matrices: *const *mut EF,
            widths: *const u32,
            num_matrices: u16,
            log_output_heights: *const u8,
            max_output_cells: u32,
            r_dev: *const EF,
            stream: cudaStream_t,
        ) -> i32;

        fn _reduce_over_x_and_cols(
            input: *const std::ffi::c_void,
            output: *mut std::ffi::c_void,
            num_x: u32,
            num_cols: u32,
            large_domain_size: u32,
            stream: cudaStream_t,
        ) -> i32;

        fn _fold_ple_from_coeffs(
            input_coeffs: *const std::ffi::c_void,
            output: *mut std::ffi::c_void,
            num_x: u32,
            width: u32,
            domain_size: u32,
            r: EF,
            stream: cudaStream_t,
        ) -> i32;

        fn _triangular_fold_mle(
            output: *mut EF,
            input: *const EF,
            r: EF,
            output_max_n: u32,
            stream: cudaStream_t,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sumcheck_mle_round(
        input_matrices: &DeviceBuffer<*const EF>,
        output: &DeviceBuffer<EF>,
        tmp_block_sums: &DeviceBuffer<EF>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u32,
        height: u32,
        d: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        if d == 0 || d > MAX_SUMCHECK_MLE_ROUND_D {
            return Err(CudaError::new(1));
        }
        CudaError::from_result(_sumcheck_mle_round(
            input_matrices.as_ptr(),
            output.as_mut_ptr(),
            tmp_block_sums.as_mut_ptr(),
            widths.as_ptr(),
            num_matrices,
            height,
            d,
            stream,
        ))
    }

    /// # Safety
    /// - `input_matrices` must consist of pointers to device memory locations.
    /// - `output_matrices` must consist of pointers to device memory locations.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fold_mle(
        input_matrices: &DeviceBuffer<*const EF>,
        output_matrices: &DeviceBuffer<*mut EF>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u16,
        output_height: u32,
        max_output_cells: u32,
        r_val: EF,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fold_mle(
            input_matrices.as_ptr(),
            output_matrices.as_ptr(),
            widths.as_ptr(),
            num_matrices,
            output_height,
            max_output_cells,
            r_val,
            stream,
        ))
    }

    pub unsafe fn fold_mle_column(
        buffer: &mut DeviceBuffer<EF>,
        size: usize,
        r: EF,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fold_mle_column(buffer.as_mut_raw_ptr(), size, r, stream))
    }

    /// # Safety
    /// - `input_matrices` must consist of pointers to device memory locations.
    /// - `output_matrices` must consist of pointers to device memory locations.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn batch_fold_mle(
        input_matrices: &DeviceBuffer<*const EF>,
        output_matrices: &DeviceBuffer<*mut EF>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u16,
        log_output_heights: &DeviceBuffer<u8>,
        max_output_cells: u32,
        r_val: EF,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_batch_fold_mle(
            input_matrices.as_ptr(),
            output_matrices.as_ptr(),
            widths.as_ptr(),
            num_matrices,
            log_output_heights.as_ptr(),
            max_output_cells,
            r_val,
            stream,
        ))
    }

    /// Device-challenge variant of [`batch_fold_mle`], taking raw device pointers:
    /// the round challenge is read on-device from `r_dev` (graph-IR path), where
    /// the sampled challenge only exists as a runtime device buffer and must never
    /// be resolved on the host.
    ///
    /// Byte-for-byte equivalent to [`batch_fold_mle`] when `*r_dev == r_val`; see
    /// `dev_challenge_tests::batch_fold_mle_dev_challenge_matches_host_value`.
    ///
    /// # Safety
    /// - `input_matrices` must point to `num_matrices` pointers to device memory.
    /// - `output_matrices` must point to `num_matrices` pointers to device memory.
    /// - `widths` / `log_output_heights` must point to `num_matrices` device elements.
    /// - `r_dev` must point to at least one readable device `EF`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn batch_fold_mle_dev_challenge(
        input_matrices: *const *const EF,
        output_matrices: *const *mut EF,
        widths: *const u32,
        num_matrices: u16,
        log_output_heights: *const u8,
        max_output_cells: u32,
        r_dev: *const EF,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        debug_assert!(!input_matrices.is_null());
        debug_assert!(!output_matrices.is_null());
        debug_assert!(!widths.is_null());
        debug_assert!(!log_output_heights.is_null());
        debug_assert!(!r_dev.is_null());
        CudaError::from_result(_batch_fold_mle_dev_challenge(
            input_matrices,
            output_matrices,
            widths,
            num_matrices,
            log_output_heights,
            max_output_cells,
            r_dev,
            stream,
        ))
    }

    pub unsafe fn fold_ple_from_coeffs(
        input_coeffs: *const F, // Base field (F)
        output: *mut EF,        // Extension field (EF)
        num_x: u32,
        width: u32,
        domain_size: u32,
        r: EF,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fold_ple_from_coeffs(
            input_coeffs as *const c_void,
            output as *mut c_void,
            num_x,
            width,
            domain_size,
            r,
            stream,
        ))
    }

    pub unsafe fn reduce_over_x_and_cols<T>(
        input: &DeviceBuffer<T>,
        output: &DeviceBuffer<T>,
        num_x: u32,
        num_cols: u32,
        large_domain_size: u32,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_reduce_over_x_and_cols(
            input.as_raw_ptr(),
            output.as_mut_raw_ptr(),
            num_x,
            num_cols,
            large_domain_size,
            stream,
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
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        debug_assert_eq!(output.buffer.len(), 2 << output_max_n);
        debug_assert_eq!(input.buffer.len(), 4 << output_max_n);
        CudaError::from_result(_triangular_fold_mle(
            output.buffer.as_mut_ptr(),
            input.buffer.as_ptr(),
            r,
            output_max_n as u32,
            stream,
        ))
    }

    // -----------------------------------------------------------------------
    // Tests.

    /// Differential tests for the `_dev_challenge` sibling ABIs in this module.
    #[cfg(test)]
    mod dev_challenge_tests {
        use openvm_cuda_common::{
            common::get_device,
            copy::{MemCopyD2H, MemCopyH2D},
            d_buffer::DeviceBuffer,
            stream::{CudaStream, GpuDeviceCtx, StreamGuard},
        };
        use rand::{rngs::StdRng, Rng, SeedableRng};

        use super::{batch_fold_mle, batch_fold_mle_dev_challenge, EF};

        fn test_ctx() -> GpuDeviceCtx {
            GpuDeviceCtx {
                device_id: get_device().unwrap() as u32,
                stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
            }
        }

        /// Reference for `sumcheck::fold_mle` (`cuda/include/sumcheck.cuh:294`) on
        /// one column-major matrix: `out[col][row] = t0 + r * (t1 - t0)` over the
        /// even/odd row pair.
        fn host_fold(input: &[EF], height: usize, width: usize, r: EF) -> Vec<EF> {
            let out_height = height >> 1;
            let mut out = Vec::with_capacity(out_height * width);
            for col in 0..width {
                for row in 0..out_height {
                    let t0 = input[col * height + 2 * row];
                    let t1 = input[col * height + 2 * row + 1];
                    out.push(t0 + r * (t1 - t0));
                }
            }
            out
        }

        fn ef_bytes(v: &[EF]) -> &[u8] {
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
        }

        /// `_batch_fold_mle_dev_challenge` must produce bit-identical output to
        /// `_batch_fold_mle` when the device buffer holds the same challenge.
        ///
        /// The batch is deliberately ragged: unequal heights, unequal widths, and
        /// a height-one output (`log_output_height == 0`), so the 2-D dispatch over
        /// `log_output_heights[mat_idx]` is actually exercised.
        #[test]
        fn batch_fold_mle_dev_challenge_matches_host_value() {
            let ctx = test_ctx();
            let stream = ctx.stream.as_raw();
            let mut rng = StdRng::seed_from_u64(0x5EED_0002);

            // (input height, width); output height = height / 2.
            const SHAPES: [(usize, usize); 3] = [(8, 3), (4, 5), (2, 7)];
            let num_matrices = SHAPES.len();

            let log_output_heights: Vec<u8> = SHAPES
                .iter()
                .map(|(h, _)| ((h >> 1) as u32).ilog2() as u8)
                .collect();
            let widths: Vec<u32> = SHAPES.iter().map(|(_, w)| *w as u32).collect();
            let max_output_cells = SHAPES
                .iter()
                .map(|(h, w)| ((h >> 1) * w) as u32)
                .max()
                .unwrap();
            assert_eq!(log_output_heights, vec![2u8, 1, 0]);
            assert_eq!(max_output_cells, 12);

            // Shared read-only inputs; independent outputs for the two entry points.
            let host_inputs: Vec<Vec<EF>> = SHAPES
                .iter()
                .map(|(h, w)| (0..h * w).map(|_| rng.random()).collect())
                .collect();
            let inputs: Vec<DeviceBuffer<EF>> = host_inputs
                .iter()
                .map(|host| host.as_slice().to_device_on(&ctx).expect("H2D input"))
                .collect();
            let outs_a: Vec<DeviceBuffer<EF>> = SHAPES
                .iter()
                .map(|(h, w)| DeviceBuffer::<EF>::with_capacity_on((h >> 1) * w, &ctx))
                .collect();
            let outs_b: Vec<DeviceBuffer<EF>> = SHAPES
                .iter()
                .map(|(h, w)| DeviceBuffer::<EF>::with_capacity_on((h >> 1) * w, &ctx))
                .collect();

            let r: EF = rng.random();
            let r_dev = [r].as_slice().to_device_on(&ctx).expect("H2D r");

            let input_ptrs: Vec<*const EF> = inputs.iter().map(|b| b.as_ptr()).collect();
            let out_ptrs_a: Vec<*mut EF> = outs_a.iter().map(|b| b.as_mut_ptr()).collect();
            let out_ptrs_b: Vec<*mut EF> = outs_b.iter().map(|b| b.as_mut_ptr()).collect();

            let d_input_ptrs = input_ptrs.to_device_on(&ctx).expect("H2D input ptrs");
            let d_out_ptrs_a = out_ptrs_a.to_device_on(&ctx).expect("H2D out ptrs a");
            let d_out_ptrs_b = out_ptrs_b.to_device_on(&ctx).expect("H2D out ptrs b");
            let d_widths = widths.to_device_on(&ctx).expect("H2D widths");
            let d_log_output_heights = log_output_heights
                .to_device_on(&ctx)
                .expect("H2D log_output_heights");

            unsafe {
                batch_fold_mle(
                    &d_input_ptrs,
                    &d_out_ptrs_a,
                    &d_widths,
                    num_matrices as u16,
                    &d_log_output_heights,
                    max_output_cells,
                    r,
                    stream,
                )
                .expect("batch_fold_mle");
                batch_fold_mle_dev_challenge(
                    d_input_ptrs.as_ptr(),
                    d_out_ptrs_b.as_ptr(),
                    d_widths.as_ptr(),
                    num_matrices as u16,
                    d_log_output_heights.as_ptr(),
                    max_output_cells,
                    r_dev.as_ptr(),
                    stream,
                )
                .expect("batch_fold_mle_dev_challenge");
            }
            ctx.stream.synchronize().expect("sync");

            for (i, (a, b)) in outs_a.iter().zip(outs_b.iter()).enumerate() {
                let ha = a.to_host_on(&ctx).expect("D2H out_a");
                let hb = b.to_host_on(&ctx).expect("D2H out_b");
                // Teeth: the eager output must equal the host fold, so a pass
                // cannot come from two buffers of identical uninitialized bytes.
                let (h, w) = SHAPES[i];
                let want = host_fold(&host_inputs[i], h, w, r);
                assert_eq!(
                    ef_bytes(&ha),
                    ef_bytes(&want),
                    "matrix {i}: host-value fold differs from the reference fold"
                );
                assert_eq!(
                    ef_bytes(&ha),
                    ef_bytes(&hb),
                    "matrix {i}: dev-challenge fold differs from host-value fold"
                );
            }
        }
    }
}

// relate to prefix.cu
pub mod prefix {
    use super::*;

    extern "C" {
        fn _prefix_scan_block_ext(
            d_inout: *mut std::ffi::c_void,
            length: u64,
            round_stride: u64,
            block_num: u64,
            stream: cudaStream_t,
        ) -> i32;

        fn _prefix_scan_block_downsweep_ext(
            d_inout: *mut std::ffi::c_void,
            length: u64,
            round_stride: u64,
            stream: cudaStream_t,
        ) -> i32;

        fn _prefix_scan_epilogue_ext(
            d_inout: *mut std::ffi::c_void,
            length: u64,
            stream: cudaStream_t,
        ) -> i32;
    }

    pub unsafe fn prefix_scan_block_ext<T>(
        d_inout: &DeviceBuffer<T>,
        length: u64,
        round_stride: u64,
        block_num: u64,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_prefix_scan_block_ext(
            d_inout.as_mut_raw_ptr(),
            length,
            round_stride,
            block_num,
            stream,
        ))
    }

    pub unsafe fn prefix_scan_block_downsweep_ext<T>(
        d_inout: &DeviceBuffer<T>,
        length: u64,
        round_stride: u64,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_prefix_scan_block_downsweep_ext(
            d_inout.as_mut_raw_ptr(),
            length,
            round_stride,
            stream,
        ))
    }

    pub unsafe fn prefix_scan_epilogue_ext<T>(
        d_inout: &DeviceBuffer<T>,
        length: u64,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_prefix_scan_epilogue_ext(
            d_inout.as_mut_raw_ptr(),
            length,
            stream,
        ))
    }
}
