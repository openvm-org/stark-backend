use std::{cmp::max, ffi::c_void, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::{
    base::DeviceMatrix, cuda::kernels::lde::batch_expand_pad, ntt::batch_ntt,
};
use openvm_cuda_common::{
    copy::{MemCopyD2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::{p3_util::log2_strict_usize, prover::MatrixDimensions};
use stark_backend_v2::prover::stacked_pcs::StackedLayout;
use tracing::instrument;

use crate::{
    Digest, F, ProverError,
    cuda::{matrix::batch_expand_pad_wide, poly::mle_interpolate_stage_2d},
    merkle_tree::MerkleTreeGpu,
    poly::PleMatrix,
};

#[derive(derive_new::new)]
pub struct StackedPcsDataGpu<F, Digest> {
    /// Layout of the unstacked collection of matrices within the stacked matrix.
    pub layout: StackedLayout,
    /// The stacked matrix with height `2^{l_skip + n_stack}`.
    pub matrix: PleMatrix<F>,
    /// Merkle tree of the Reed-Solomon codewords of the stacked matrix.
    /// Depends on `k_whir` parameter.
    pub tree: MerkleTreeGpu<F, Digest>,
}

#[instrument(level = "info", skip_all)]
pub fn stacked_commit(
    l_skip: usize,
    n_stack: usize,
    log_blowup: usize,
    k_whir: usize,
    traces: &[&DeviceMatrix<F>],
) -> Result<(Digest, StackedPcsDataGpu<F, Digest>), ProverError> {
    let (q_matrix, layout) = stacked_matrix(l_skip, n_stack, traces)?;
    let rs_matrix = rs_code_matrix(l_skip, log_blowup, &q_matrix)?;
    let tree = MerkleTreeGpu::<F, Digest>::new(rs_matrix, 1 << k_whir)?;
    let root = tree.root();
    let data = StackedPcsDataGpu::new(layout, q_matrix, tree);
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
) -> Result<(PleMatrix<F>, StackedLayout), ProverError> {
    let sorted_meta = traces
        .iter()
        .enumerate()
        .map(|(idx, trace)| {
            // height cannot be zero:
            let log_height = log2_strict_usize(trace.height());
            (idx, trace.width(), log_height)
        })
        .collect_vec();
    debug_assert!(sorted_meta.is_sorted_by(|a, b| a.2 >= b.2));
    let mut layout = StackedLayout::new(l_skip, l_skip + n_stack, sorted_meta);
    let total_cells: usize = traces
        .iter()
        .map(|t| max(t.height() * t.width(), 1 << l_skip))
        .sum();
    let height = 1usize << (l_skip + n_stack);
    let width = total_cells.div_ceil(height);

    let q_evals = DeviceBuffer::<F>::with_capacity(width.checked_mul(height).unwrap());
    q_evals.fill_zero()?;
    for (mat_idx, j, s) in &mut layout.sorted_cols {
        let start = s.col_idx * height + s.row_idx;
        let trace = traces[*mat_idx];
        let s_len = s.len(l_skip);
        debug_assert_eq!(trace.height(), 1 << s.log_height());
        if s.log_height() >= l_skip {
            debug_assert_eq!(trace.height(), s_len);
            // SAFETY: matrix buffers are allocated correctly with respect to dimensions
            // - `trace.height() = s_len` since `log_height >= l_skip`
            // - `q_buf` has enough capacity by definition of stacked `layout`
            unsafe {
                let src = trace.buffer().as_ptr().add(*j * s_len);
                let dst = q_evals.as_mut_ptr().add(start);
                // D2D memcpy
                cuda_memcpy::<true, true>(
                    dst as *mut c_void,
                    src as *const c_void,
                    s_len * size_of::<F>(),
                )?;
            }
        } else {
            let stride = s.stride(l_skip);
            debug_assert_eq!(stride * trace.height(), s_len);
            // SAFETY: matrix buffers are allocated correctly
            // - `q_buf` has enough capacity by definition of stacked `layout`
            // - we abuse `batch_expand_pad` with `poly_count = trace.height()` to create a strided
            //   column of length `s_len = stride * trace.height()`
            unsafe {
                let src = trace.buffer().as_ptr().add(*j * trace.height());
                let dst = q_evals.as_mut_ptr().add(start);
                batch_expand_pad_wide(dst, src, trace.height() as u32, stride as u32, 1)?;
            }
        }
    }
    // TODO: remove this once we delete evals in PleMatirx
    let q_mixed = q_evals.device_copy()?;
    if l_skip > 0 {
        // For univariate coordinate, perform inverse NTT for each 2^l_skip chunk of the full
        // `s_len` slice: Use natural ordering.
        batch_ntt(
            &q_mixed,
            l_skip as u32,
            0,
            (q_mixed.len() >> l_skip) as u32,
            true,
            true,
        );
    }
    Ok((
        PleMatrix::new(DeviceMatrix::new(Arc::new(q_evals), height, width), q_mixed),
        layout,
    ))
}

/// Computes the Reed-Solomon codeword of each column vector of `eval_matrix` where the rate is
/// `2^{-log_blowup}`. The column vectors are treated as evaluations of a prismalinear extension on
/// a hyperprism.
#[instrument(skip_all)]
pub fn rs_code_matrix(
    l_skip: usize,
    log_blowup: usize,
    matrix: &PleMatrix<F>,
) -> Result<DeviceMatrix<F>, ProverError> {
    let height = matrix.height();
    debug_assert!(height >= (1 << l_skip));
    let width = matrix.width();
    let codeword_height = height.checked_shl(log_blowup as u32).unwrap();
    let codewords = DeviceBuffer::<F>::with_capacity(codeword_height * width);
    // The following three kernels together perform MLE interpolation followed by coset NTT for
    // `width` polys from `height -> codeword_height` size domains.

    // SAFETY: `codewords` is allocated for `width` polys of `codeword_height` each, and we expand
    // from `matrix.mxied` which is `width` polys of `height` each.
    unsafe {
        batch_expand_pad(
            &codewords,
            &matrix.mixed,
            width as u32,
            codeword_height as u32,
            height as u32,
        )?;
    }
    let n = log2_strict_usize(height) - l_skip;
    // Go through coordinates X_1, ..., X_n and interpolate each one from s(0), s(1) -> s(0) +
    // (s(1) - s(0)) X_i
    for i in 0..n {
        let step = 1u32 << (l_skip + i);
        // SAFETY: `codewords` is properly initialized and step is in bounds.
        unsafe {
            mle_interpolate_stage_2d(
                codewords.as_mut_ptr(),
                width.try_into().unwrap(),
                height as u32,
                codeword_height as u32,
                step,
                true, // MLE evaluation-to-coefficient
            )?;
        }
    }
    // Compute RS codeword on a prismalinear polynomial in coefficient form:
    // We use that the coefficients are in a basis that exactly corresponds to the standard
    // monomial univariate basis. Hence RS codeword is just cosetDFT on the
    // relevant smooth domain
    batch_ntt(
        &codewords,
        log2_strict_usize(codeword_height) as u32,
        0u32,
        width as u32,
        true,
        false,
    );
    Ok(DeviceMatrix::new(
        Arc::new(codewords),
        codeword_height,
        width,
    ))
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_field::FieldAlgebra;
    use stark_backend_v2::{
        prover::ColMajorMatrix,
        test_utils::{InteractionsFixture11, TestFixture},
    };

    use super::*;
    use crate::{F, transport_matrix_d2h_col_major, transport_matrix_h2d_col_major};

    #[test]
    fn test_stacked_matrix_manual_0() {
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_canonical_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| transport_matrix_h2d_col_major(&ColMajorMatrix::new(c, 1)).unwrap())
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let l_skip = 0;
        let (stacked_mat, _layout) = stacked_matrix(0, 2, &mat_refs).unwrap();
        assert_eq!(stacked_mat.height(), 4);
        assert_eq!(stacked_mat.width(), 2);
        let stacked_h_mat =
            transport_matrix_d2h_col_major(&stacked_mat.to_evals(l_skip).unwrap()).unwrap();
        assert_eq!(
            stacked_h_mat.values,
            [1, 2, 3, 4, 5, 6, 7, 0].map(F::from_canonical_u32).to_vec()
        );
    }

    #[test]
    fn test_stacked_matrix_manual_1() {
        let ctx = InteractionsFixture11.generate_proving_ctx();
        let [send_trace, rcv_trace] = [0, 1]
            .map(|i| transport_matrix_h2d_col_major(&ctx.per_trace[i].1.common_main).unwrap());
        let l_skip = 2;
        let n_stack = 8;
        let (stacked_mat, _layout) =
            stacked_matrix(l_skip, n_stack, &[&rcv_trace, &send_trace]).unwrap();
        assert_eq!(stacked_mat.height(), 1 << (l_skip + n_stack));
        assert_eq!(stacked_mat.width(), 1);
        let stacked_h_mat =
            transport_matrix_d2h_col_major(&stacked_mat.to_evals(l_skip).unwrap()).unwrap();
        let mut expected = vec![F::ZERO; 1 << (l_skip + n_stack)];
        expected[..24].copy_from_slice(
            &[
                1, 3, 4, 2, 0, 545, 1, 0, 5, 4, 4, 5, 123, 889, 889, 456, 0, 3, 7, 546, 1, 5, 4,
                889,
            ]
            .map(F::from_canonical_u32),
        );
        assert_eq!(stacked_h_mat.values, expected);
    }

    #[test]
    fn test_stacked_matrix_manual_strided_0() {
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_canonical_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| transport_matrix_h2d_col_major(&ColMajorMatrix::new(c, 1)).unwrap())
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let l_skip = 2;
        let (stacked_mat, _layout) = stacked_matrix(l_skip, 0, &mat_refs).unwrap();
        assert_eq!(stacked_mat.height(), 4);
        assert_eq!(stacked_mat.width(), 3);
        let stacked_h_mat =
            transport_matrix_d2h_col_major(&stacked_mat.to_evals(l_skip).unwrap()).unwrap();
        assert_eq!(
            stacked_h_mat.values,
            [1, 2, 3, 4, 5, 0, 6, 0, 7, 0, 0, 0]
                .map(F::from_canonical_u32)
                .to_vec()
        );
    }

    #[test]
    fn test_stacked_matrix_manual_strided_1() {
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_canonical_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| transport_matrix_h2d_col_major(&ColMajorMatrix::new(c, 1)).unwrap())
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let l_skip = 3;
        let (stacked_mat, _layout) = stacked_matrix(l_skip, 0, &mat_refs).unwrap();
        assert_eq!(stacked_mat.height(), 8);
        assert_eq!(stacked_mat.width(), 3);
        let stacked_h_mat =
            transport_matrix_d2h_col_major(&stacked_mat.to_evals(l_skip).unwrap()).unwrap();
        assert_eq!(
            stacked_h_mat.values,
            [
                [1, 0, 2, 0, 3, 0, 4, 0],
                [5, 0, 0, 0, 6, 0, 0, 0],
                [7, 0, 0, 0, 0, 0, 0, 0]
            ]
            .into_iter()
            .flatten()
            .map(F::from_canonical_u32)
            .collect_vec()
        );
    }
}
