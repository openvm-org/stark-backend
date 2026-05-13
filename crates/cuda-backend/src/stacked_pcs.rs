use std::{ffi::c_void, sync::Arc};

use getset::Getters;
use itertools::Itertools;
use openvm_cuda_common::{
    copy::cuda_memcpy_on, d_buffer::DeviceBuffer, memory_manager::MemTracker, stream::GpuDeviceCtx,
};
use openvm_stark_backend::{
    p3_util::log2_strict_usize,
    prover::{stacked_pcs::StackedLayout, MatrixDimensions},
};
use tracing::instrument;

use crate::{
    base::{DeviceMatrix, DeviceMatrixView},
    cuda::{
        batch_ntt_small::batch_ntt_small,
        matrix::{batch_expand_pad, batch_expand_pad_wide},
        ntt::bit_rev,
    },
    hash_scheme::GpuMerkleHash,
    merkle_tree::{MerkleTreeConstructor, MerkleTreeGpu},
    ntt::batch_ntt,
    poly::{mle_interpolate_stages, PleMatrix},
    prelude::F,
    GpuProverConfig, ProverError, RsCodeMatrixError, StackTracesError,
};

#[derive(Getters)]
pub struct StackedPcsDataGpu<F, Digest> {
    /// Layout of the unstacked collection of matrices within the stacked matrix.
    #[getset(get = "pub")]
    pub(crate) layout: StackedLayout,
    /// The stacked matrix with height `2^{l_skip + n_stack}`.
    /// This is optionally cached depending on the prover configuration:
    /// - Caching increases the peak GPU memory but avoids a recomputation during stacked
    ///   reduction.
    /// - Not caching means the stacked matrix computation is recomputed during stacked reduction,
    ///   but lowers the peak GPU memory.
    #[getset(get = "pub")]
    pub(crate) matrix: Option<PleMatrix<F>>,
    /// Merkle tree of the Reed-Solomon codewords of the stacked matrix.
    /// Depends on `k_whir` parameter.
    #[getset(get = "pub")]
    pub(crate) tree: MerkleTreeGpu<F, Digest>,
}

#[allow(clippy::type_complexity)]
#[instrument(level = "info", skip_all)]
pub fn stacked_commit<MH: GpuMerkleHash + MerkleTreeConstructor>(
    l_skip: usize,
    n_stack: usize,
    log_blowup: usize,
    k_whir: usize,
    traces: &[&DeviceMatrix<F>],
    prover_config: GpuProverConfig,
    device_ctx: &GpuDeviceCtx,
) -> Result<(MH::Digest, StackedPcsDataGpu<F, MH::Digest>), ProverError> {
    let mut mem = MemTracker::start("prover.stacked_commit");
    mem.tracing_info("before stacked_commit");
    mem.reset_peak();
    let layout = get_stacked_layout(l_skip, n_stack, traces);
    tracing::info!(
        height = layout.height(),
        width = layout.width(),
        "stacked_matrix_dimensions"
    );
    let opt_stacked_matrix = if prover_config.cache_stacked_matrix {
        Some(stack_traces(&layout, traces, device_ctx)?)
    } else {
        None
    };
    let rs_matrix = rs_code_matrix(log_blowup, &layout, traces, &opt_stacked_matrix, device_ctx)?;
    let tree = MerkleTreeGpu::<F, MH::Digest>::new_with_hash::<MH>(
        rs_matrix,
        1 << k_whir,
        prover_config.cache_rs_code_matrix,
        device_ctx,
    )?;
    let root = tree.root();
    let data = StackedPcsDataGpu {
        layout,
        matrix: opt_stacked_matrix,
        tree,
    };
    mem.emit_metrics();
    Ok((root, data))
}

/// The `traces` **must** already be in height-sorted order.
///
/// This function is generic in `F` and only relies on CUDA memory operations.
#[instrument(skip_all)]
pub fn stacked_matrix(
    l_skip: usize,
    n_stack: usize,
    traces: &[&DeviceMatrix<F>],
    device_ctx: &GpuDeviceCtx,
) -> Result<(PleMatrix<F>, StackedLayout), ProverError> {
    let layout = get_stacked_layout(l_skip, n_stack, traces);
    let matrix = stack_traces(&layout, traces, device_ctx)?;
    Ok((matrix, layout))
}

pub(crate) fn get_stacked_layout(
    l_skip: usize,
    n_stack: usize,
    traces: &[&DeviceMatrix<F>],
) -> StackedLayout {
    let sorted_meta = traces
        .iter()
        .map(|trace| {
            // height cannot be zero:
            let log_height = log2_strict_usize(trace.height());
            (trace.width(), log_height)
        })
        .collect_vec();
    debug_assert!(sorted_meta.is_sorted_by(|a, b| a.1 >= b.1));
    StackedLayout::new(l_skip, l_skip + n_stack, sorted_meta).unwrap()
}

pub(crate) fn stack_traces(
    layout: &StackedLayout,
    traces: &[&DeviceMatrix<F>],
    device_ctx: &GpuDeviceCtx,
) -> Result<PleMatrix<F>, StackTracesError> {
    let mem = MemTracker::start("prover.stack_traces");
    let l_skip = layout.l_skip();
    let height = layout.height();
    let width = layout.width();
    let mut q_evals =
        DeviceBuffer::<F>::with_capacity_on(width.checked_mul(height).unwrap(), device_ctx);
    stack_traces_into_expanded(layout, traces, &mut q_evals, height, device_ctx)?;
    mem.emit_metrics();
    Ok(PleMatrix::from_evals(
        l_skip, q_evals, height, width, device_ctx,
    ))
}

/// `buffer` should be the buffer to write the stacked traces into.
/// `buffer` should be a matrix with dimensions `padded_height x width` where `width` is the stacked
/// width and `padded_height` must be a multiple of the stacked height.
pub(crate) fn stack_traces_into_expanded(
    layout: &StackedLayout,
    traces: &[&DeviceMatrix<F>],
    buffer: &mut DeviceBuffer<F>,
    padded_height: usize,
    device_ctx: &GpuDeviceCtx,
) -> Result<(), StackTracesError> {
    let l_skip = layout.l_skip();
    debug_assert_eq!(padded_height % layout.height(), 0);
    debug_assert_eq!(buffer.len() % padded_height, 0);
    debug_assert_eq!(buffer.len() / padded_height, layout.width());
    buffer
        .fill_zero_on(device_ctx)
        .map_err(StackTracesError::FillZero)?;
    let mut idx = 0;
    while idx < layout.sorted_cols.len() {
        let (mat_idx, j, s) = &layout.sorted_cols[idx];
        let start = s.col_idx * padded_height + s.row_idx;
        let trace = traces[*mat_idx];
        let s_len = s.len(l_skip);
        debug_assert_eq!(trace.height(), 1 << s.log_height());
        if s.log_height() >= l_skip {
            debug_assert_eq!(trace.height(), s_len);
            let mut copy_len = s_len;
            let mut end = idx + 1;
            while end < layout.sorted_cols.len() {
                let (next_mat_idx, next_j, next_s) = &layout.sorted_cols[end];
                if *next_mat_idx != *mat_idx || next_s.log_height() != s.log_height() {
                    break;
                }
                let expected_j = *j + (end - idx);
                let next_len = next_s.len(l_skip);
                let next_start = next_s.col_idx * padded_height + next_s.row_idx;
                if *next_j != expected_j || next_len != s_len || next_start != start + copy_len {
                    break;
                }
                copy_len += next_len;
                end += 1;
            }

            // SAFETY: matrix buffers are allocated correctly with respect to dimensions.
            // The grouped columns are contiguous in both source and destination buffers.
            unsafe {
                let src = trace.buffer().as_ptr().add(*j * s_len);
                let dst = buffer.as_mut_ptr().add(start);
                cuda_memcpy_on::<true, true>(
                    dst as *mut c_void,
                    src as *const c_void,
                    copy_len * size_of::<F>(),
                    device_ctx,
                )?;
            }
            idx = end;
        } else {
            let stride = s.stride(l_skip);
            debug_assert_eq!(stride * trace.height(), s_len);
            // SAFETY: matrix buffers are allocated correctly
            // - `q_buf` has enough capacity by definition of stacked `layout`
            // - we abuse `batch_expand_pad` with `poly_count = trace.height()` to create a strided
            //   column of length `s_len = stride * trace.height()`
            unsafe {
                let src = trace.buffer().as_ptr().add(*j * trace.height());
                let dst = buffer.as_mut_ptr().add(start);
                batch_expand_pad_wide(
                    dst,
                    src,
                    trace.height() as u32,
                    stride as u32,
                    1,
                    device_ctx.stream.as_raw(),
                )
                .map_err(StackTracesError::BatchExpandPadWide)?;
            }
            idx += 1;
        }
    }
    Ok(())
}

/// Computes the Reed-Solomon codeword of each column vector of `eval_matrix` where the rate is
/// `2^{-log_blowup}`. The column vectors are treated as evaluations of a prismalinear extension on
/// a hyperprism.
///
/// Uses `stacked_matrix` if available, or else stacks `traces` directly into final codeword matrix
/// buffer.
#[instrument(skip_all)]
pub fn rs_code_matrix(
    log_blowup: usize,
    layout: &StackedLayout,
    traces: &[&DeviceMatrix<F>],
    stacked_matrix: &Option<PleMatrix<F>>,
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceMatrix<F>, RsCodeMatrixError> {
    let mem = MemTracker::start_and_reset_peak("prover.rs_code_matrix");
    let l_skip = layout.l_skip();
    let height = layout.height();
    let width = layout.width();
    debug_assert!(height >= (1 << l_skip));
    let codeword_height = height.checked_shl(log_blowup as u32).unwrap();
    let mut codewords = DeviceBuffer::<F>::with_capacity_on(codeword_height * width, device_ctx);
    // The following kernels together perform MLE interpolation followed by coset NTT for
    // `width` polys from `height -> codeword_height` size domains.
    if let Some(stacked_matrix) = stacked_matrix.as_ref() {
        // SAFETY: `codewords` is allocated for `width` polys of `codeword_height` each, and we
        // expand from `matrix.mixed` which is `width` polys of `height` each.
        unsafe {
            batch_expand_pad(
                codewords.as_mut_ptr(),
                stacked_matrix.mixed.as_ptr(),
                width as u32,
                codeword_height as u32,
                height as u32,
                device_ctx.stream.as_raw(),
            )
            .map_err(RsCodeMatrixError::BatchExpandPad)?;
        }
    } else {
        stack_traces_into_expanded(layout, traces, &mut codewords, codeword_height, device_ctx)
            .map_err(RsCodeMatrixError::StackTraces)?;
        // Currently codewords has the stacked matrix, batch expanded, in evaluation form on
        // hyperprism. We convert it to mixed form, i.e., unroll `PleMatrix::from_evals`.
        // PERF[jpw]: We do some wasted work on the padded zero part. A more specialized kernel
        // could be written to avoid this.
        if l_skip > 0 {
            // For univariate coordinate, perform inverse NTT for each 2^l_skip chunk per column:
            // (width cols) * (codeword_height / 2^l_skip chunks per col). Use natural ordering.
            let num_uni_poly = width * (codeword_height >> l_skip);
            unsafe {
                batch_ntt_small(
                    &mut codewords,
                    l_skip,
                    num_uni_poly,
                    true,
                    device_ctx.stream.as_raw(),
                )
                .map_err(RsCodeMatrixError::CustomBatchIntt)?;
            }
        }
    }
    // Eval-to-coeff RS encoding: Instead of full n-stage MLE evals→coeffs interpolation
    // (where n = log_height - l_skip), we only need l_skip stages of coeffs→evals
    // (subset-zeta transform) within each 2^l_skip chunk. This is a major optimization:
    // e.g. for log_height=17, l_skip=2: 2 stages instead of 15.
    let log_codeword_height = log2_strict_usize(codeword_height);

    // Apply l_skip stages of coeffs_to_evals within each 2^l_skip chunk.
    // After iNTT, each chunk holds Z-monomial coefficients. We apply the subset-zeta
    // transform to convert to hypercube evaluations over the Z-bit variables.
    if l_skip > 0 {
        // SAFETY: `codewords` is properly initialized and parameters are valid.
        // Steps 2^0, 2^1, ..., 2^(l_skip-1) stay within chunk boundaries.
        unsafe {
            mle_interpolate_stages(
                codewords.as_mut_ptr(),
                width,
                codeword_height as u32,
                log_blowup as u32,
                0,                 // start_log_step
                l_skip as u32 - 1, // end_log_step (inclusive)
                false,             // coeffs to evals (NOT eval to coeff)
                false,             // natural order
                device_ctx.stream.as_raw(),
            )
            .map_err(|error| RsCodeMatrixError::MleInterpolateStage2d { error, step: 1 })?;
        }
    }

    // Bit-reverse the entire buffer in-place (required for NTT)
    unsafe {
        bit_rev(
            &codewords,
            &codewords,
            log_codeword_height as u32,
            codeword_height as u32,
            width as u32,
            device_ctx.stream.as_raw(),
        )
        .map_err(RsCodeMatrixError::BitRev)?;
    }

    // Compute RS codeword via DFT on the smoothly-embedded domain.
    batch_ntt(
        &codewords,
        log_codeword_height as u32,
        0u32,
        width as u32,
        false, // bit-reversal already done
        false,
        device_ctx,
    );
    let code_matrix = DeviceMatrix::new(Arc::new(codewords), codeword_height, width);
    mem.emit_metrics();

    Ok(code_matrix)
}

impl<F, Digest> StackedPcsDataGpu<F, Digest> {
    /// Returns a view of the specified unstacked matrix in mixed form.
    ///
    /// # Notes
    /// - `width` must be the width of the unstacked matrix.
    /// - The unstacked matrix may be strided - this must be handled by the caller.
    pub fn mixed_view<'a>(
        &'a self,
        mat_idx: usize,
        width: usize,
    ) -> Option<DeviceMatrixView<'a, F>> {
        if let Some(matrix) = self.matrix.as_ref() {
            debug_assert_eq!(self.layout.width_of(mat_idx), width);
            let s = self
                .layout
                .get(mat_idx, 0)
                .unwrap_or_else(|| panic!("Invalid matrix index: {mat_idx}"));
            let l_skip = self.layout.l_skip();
            let lifted_height = s.len(l_skip);
            let offset = s.col_idx * matrix.height() + s.row_idx;
            // SAFETY:
            // - by definition of stacked layout and stacked matrix, `ptr` is valid and allocated
            //   for `lifted_height * width` elements.
            unsafe {
                let ptr = matrix.mixed.as_ptr().add(offset);
                Some(DeviceMatrixView::from_raw_parts(ptr, lifted_height, width))
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use openvm_cuda_common::{
        common::get_device,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::{
        prover::ColMajorMatrix,
        test_utils::{InteractionsFixture11, TestFixture},
    };
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{
        data_transporter::{transport_matrix_d2h_col_major, transport_matrix_h2d_col_major},
        prelude::{F, SC},
    };

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        }
    }

    #[test]
    fn test_stacked_matrix_manual_0() {
        let device_ctx = test_ctx();
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| {
                transport_matrix_h2d_col_major(&ColMajorMatrix::new(c, 1), &device_ctx).unwrap()
            })
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let l_skip = 0;
        let (stacked_mat, _layout) = stacked_matrix(0, 2, &mat_refs, &device_ctx).unwrap();
        assert_eq!(stacked_mat.height(), 4);
        assert_eq!(stacked_mat.width(), 2);
        let stacked_h_mat = transport_matrix_d2h_col_major(
            &stacked_mat.to_evals(l_skip, &device_ctx).unwrap(),
            &device_ctx,
        )
        .unwrap();
        assert_eq!(
            stacked_h_mat.values,
            [1, 2, 3, 4, 5, 6, 7, 0].map(F::from_u32).to_vec()
        );
    }

    #[test]
    fn test_stacked_matrix_manual_1() {
        let gpu_ctx = test_ctx();
        let proving_ctx = TestFixture::<SC>::generate_proving_ctx(&InteractionsFixture11);
        let [send_trace, rcv_trace] = [0, 1].map(|i| {
            transport_matrix_h2d_col_major(&proving_ctx.per_trace[i].1.common_main, &gpu_ctx)
                .unwrap()
        });
        let l_skip = 2;
        let n_stack = 8;
        let (stacked_mat, _layout) =
            stacked_matrix(l_skip, n_stack, &[&rcv_trace, &send_trace], &gpu_ctx).unwrap();
        assert_eq!(stacked_mat.height(), 1 << (l_skip + n_stack));
        assert_eq!(stacked_mat.width(), 1);
        let stacked_h_mat = transport_matrix_d2h_col_major(
            &stacked_mat.to_evals(l_skip, &gpu_ctx).unwrap(),
            &gpu_ctx,
        )
        .unwrap();
        let mut expected = vec![F::ZERO; 1 << (l_skip + n_stack)];
        expected[..24].copy_from_slice(
            &[
                1, 3, 4, 2, 0, 545, 1, 0, 5, 4, 4, 5, 123, 889, 889, 456, 0, 3, 7, 546, 1, 5, 4,
                889,
            ]
            .map(F::from_u32),
        );
        assert_eq!(stacked_h_mat.values, expected);
    }

    #[test]
    fn test_stacked_matrix_manual_strided_0() {
        let device_ctx = test_ctx();
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| {
                transport_matrix_h2d_col_major(&ColMajorMatrix::new(c, 1), &device_ctx).unwrap()
            })
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let l_skip = 2;
        let (stacked_mat, _layout) = stacked_matrix(l_skip, 0, &mat_refs, &device_ctx).unwrap();
        assert_eq!(stacked_mat.height(), 4);
        assert_eq!(stacked_mat.width(), 3);
        let stacked_h_mat = transport_matrix_d2h_col_major(
            &stacked_mat.to_evals(l_skip, &device_ctx).unwrap(),
            &device_ctx,
        )
        .unwrap();
        assert_eq!(
            stacked_h_mat.values,
            [1, 2, 3, 4, 5, 0, 6, 0, 7, 0, 0, 0]
                .map(F::from_u32)
                .to_vec()
        );
    }

    #[test]
    fn test_stacked_matrix_manual_strided_1() {
        let device_ctx = test_ctx();
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| {
                transport_matrix_h2d_col_major(&ColMajorMatrix::new(c, 1), &device_ctx).unwrap()
            })
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let l_skip = 3;
        let (stacked_mat, _layout) = stacked_matrix(l_skip, 0, &mat_refs, &device_ctx).unwrap();
        assert_eq!(stacked_mat.height(), 8);
        assert_eq!(stacked_mat.width(), 3);
        let stacked_h_mat = transport_matrix_d2h_col_major(
            &stacked_mat.to_evals(l_skip, &device_ctx).unwrap(),
            &device_ctx,
        )
        .unwrap();
        assert_eq!(
            stacked_h_mat.values,
            [
                [1, 0, 2, 0, 3, 0, 4, 0],
                [5, 0, 0, 0, 6, 0, 0, 0],
                [7, 0, 0, 0, 0, 0, 0, 0]
            ]
            .into_iter()
            .flatten()
            .map(F::from_u32)
            .collect_vec()
        );
    }
}
