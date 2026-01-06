// Batch MLE evaluation for low num_y traces using monomial representation
//
// This module provides GPU-accelerated evaluation of constraint polynomials
// by parallelizing over monomials rather than (x_int, y_int) pairs.
// This is more efficient when num_y is small because it better utilizes GPU parallelism.

use std::mem::ManuallyDrop;

use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};

use super::monomial::DeviceMonomials;
use crate::cuda::kernels::batch_mle_low::batch_mle_low_kernel;

/// Threshold for using monomial-based evaluation.
/// Traces with `num_y <= BATCH_MLE_LOW_THRESHOLD` use the monomial kernel.
pub const BATCH_MLE_LOW_THRESHOLD: u32 = 32;

/// Size of extension field element in bytes (4 BabyBear elements * 4 bytes each)
const EF_SIZE_BYTES: usize = 16;

/// Evaluator for constraint polynomials using monomial representation.
///
/// This evaluator is designed for sumcheck batch MLE evaluation when the
/// hypercube dimension `num_y` is small. It parallelizes over monomials
/// instead of (x_int, y_int) pairs.
pub struct BatchMleLowEvaluator {
    /// Monomials stored on GPU
    monomials: DeviceMonomials,
    /// Temporary buffer for block reduction partial sums (stored as bytes)
    d_tmp_sums: DeviceBuffer<u8>,
    /// Number of elements (extension field) in tmp buffer
    tmp_len: usize,
}

/// Layout information for matrix evaluations on the GPU.
///
/// Matrices are stored in a flattened format:
/// `[part][var_idx][x_int * num_y + y_int]`
#[derive(Clone, Debug)]
pub struct MatrixLayout {
    /// Width of each matrix part (number of columns)
    pub widths: Vec<u32>,
    /// Cumulative offset into the flattened buffer for each part
    pub offsets: Vec<u32>,
    /// Total number of elements in the flattened buffer
    pub total_size: usize,
}

impl MatrixLayout {
    /// Create a new matrix layout from part widths.
    pub fn new(widths: &[usize], num_xy: usize) -> Self {
        let mut offsets = Vec::with_capacity(widths.len() + 1);
        offsets.push(0);
        let mut offset = 0u32;
        for &width in widths {
            offset += (width * num_xy) as u32;
            offsets.push(offset);
        }
        Self {
            widths: widths.iter().map(|&w| w as u32).collect(),
            offsets,
            total_size: offset as usize,
        }
    }

    /// Get device buffers for widths and offsets.
    pub fn to_device(&self) -> (DeviceBuffer<u32>, DeviceBuffer<u32>) {
        let d_widths = self.widths.as_slice().to_device().unwrap();
        let d_offsets = self.offsets.as_slice().to_device().unwrap();
        (d_widths, d_offsets)
    }
}

impl BatchMleLowEvaluator {
    /// Create a new evaluator from device monomials.
    ///
    /// # Arguments
    /// * `monomials` - Pre-expanded monomials stored on GPU
    /// * `max_num_x` - Maximum number of x evaluation points (s_deg)
    /// * `max_num_y` - Maximum number of y hypercube points
    pub fn new(monomials: DeviceMonomials, max_num_x: u32, max_num_y: u32) -> Self {
        // Calculate the number of blocks needed for reduction
        const BLOCK_SIZE: u32 = 256;
        let num_blocks_x = (monomials.num_monomials + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_num_xy = max_num_x * max_num_y;
        let tmp_len = (num_blocks_x * max_num_xy) as usize;

        // Allocate as bytes to be type-agnostic
        let d_tmp_sums = DeviceBuffer::with_capacity(tmp_len * EF_SIZE_BYTES);

        Self {
            monomials,
            d_tmp_sums,
            tmp_len,
        }
    }

    /// Evaluate the constraint polynomial at all (x_int, y_int) points.
    ///
    /// Returns `output[x_int * num_y + y_int] = sum_m coeff_m(λ) * ∏_i var_i(x_int, y_int) * eq_xi`
    ///
    /// # Arguments
    /// * `d_output` - Output buffer for results `[num_x * num_y]`
    /// * `d_lambda_pows` - Powers of λ for batching: `[λ^0, λ^1, ..., λ^{num_constraints-1}]`
    /// * `d_mat_evals_local` - Matrix evaluations for local row (offset=0)
    /// * `d_mat_evals_next` - Matrix evaluations for next row (offset=1)
    /// * `d_mat_widths` - Width of each matrix part
    /// * `d_mat_offsets` - Cumulative offset for each matrix part
    /// * `d_sels` - Selector evaluations `[is_first, is_last, is_transition][x_int * num_y + y_int]`
    /// * `d_eq_xi` - eq polynomial evaluations at (xi, x_int, y_int)
    /// * `num_x` - Number of x evaluation points
    /// * `num_y` - Number of y hypercube points
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<EF>(
        &self,
        d_output: &DeviceBuffer<EF>,
        d_lambda_pows: &DeviceBuffer<EF>,
        d_mat_evals_local: &DeviceBuffer<EF>,
        d_mat_evals_next: &DeviceBuffer<EF>,
        d_mat_widths: &DeviceBuffer<u32>,
        d_mat_offsets: &DeviceBuffer<u32>,
        d_sels: &DeviceBuffer<EF>,
        d_eq_xi: &DeviceBuffer<EF>,
        num_x: u32,
        num_y: u32,
    ) {
        // SAFETY: We reinterpret the byte buffer as EF elements.
        // This is safe because:
        // 1. The buffer was allocated with the correct byte size (tmp_len * EF_SIZE_BYTES)
        // 2. We use ManuallyDrop to prevent double-free
        // 3. EF is expected to be BinomialExtensionField<BabyBear, 4> which is 16 bytes
        let d_tmp_sums_typed: ManuallyDrop<DeviceBuffer<EF>> = unsafe {
            ManuallyDrop::new(DeviceBuffer::from_raw_parts(
                self.d_tmp_sums.as_mut_ptr() as *mut EF,
                self.tmp_len,
            ))
        };

        unsafe {
            batch_mle_low_kernel(
                d_output,
                &d_tmp_sums_typed,
                &self.monomials.d_data,
                &self.monomials.d_offsets,
                self.monomials.num_monomials,
                d_lambda_pows,
                d_mat_evals_local,
                d_mat_evals_next,
                d_mat_widths,
                d_mat_offsets,
                d_sels,
                d_eq_xi,
                num_x,
                num_y,
            )
            .expect("batch_mle_low kernel failed");
        }
    }

    /// Returns the number of monomials in this evaluator.
    pub fn num_monomials(&self) -> u32 {
        self.monomials.num_monomials
    }
}

/// Determine whether to use the monomial-based kernel for a given trace.
pub fn should_use_monomial_kernel(num_y: u32) -> bool {
    num_y <= BATCH_MLE_LOW_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_layout() {
        // 3 parts with widths 2, 5, 3 and num_xy = 4
        let widths = [2usize, 5, 3];
        let num_xy = 4;
        let layout = MatrixLayout::new(&widths, num_xy);

        assert_eq!(layout.widths, vec![2u32, 5, 3]);
        // Offsets should be cumulative:
        // offsets[0] = 0
        // offsets[1] = 2 * 4 = 8
        // offsets[2] = 8 + 5 * 4 = 28
        // offsets[3] = 28 + 3 * 4 = 40
        assert_eq!(layout.offsets, vec![0u32, 8, 28, 40]);
        assert_eq!(layout.total_size, 40);
    }

    #[test]
    fn test_threshold() {
        assert!(should_use_monomial_kernel(1));
        assert!(should_use_monomial_kernel(32));
        assert!(!should_use_monomial_kernel(33));
        assert!(!should_use_monomial_kernel(64));
    }
}
