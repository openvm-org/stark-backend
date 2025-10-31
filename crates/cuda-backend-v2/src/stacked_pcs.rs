use std::{ffi::c_void, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::{
    base::DeviceMatrix, cuda::kernels::lde::batch_expand_pad, ntt::batch_ntt,
};
use openvm_cuda_common::{
    copy::{MemCopyD2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::{p3_util::log2_strict_usize, prover::MatrixDimensions};
use p3_field::Field;
use stark_backend_v2::prover::stacked_pcs::StackedLayout;
use tracing::instrument;

use crate::{
    Digest, F, ProverError, cuda::poly::ple_interpolate_stage, merkle_tree::MerkleTreeGpu,
};

#[derive(derive_new::new)]
pub struct StackedPcsDataGpu<F, Digest> {
    /// Layout of the unstacked collection of matrices within the stacked matrix.
    pub layout: StackedLayout,
    /// The stacked matrix with height `2^{l_skip + n_stack}`.
    pub matrix: DeviceMatrix<F>,
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
    let (q_trace, layout) = stacked_matrix(l_skip, n_stack, traces)?;
    let rs_matrix = rs_code_matrix(l_skip, log_blowup, &q_trace)?;
    let tree = MerkleTreeGpu::new(rs_matrix, 1 << k_whir);
    let root = tree.root();
    let data = StackedPcsDataGpu::new(layout, q_trace, tree);
    Ok((root, data))
}

/// The `traces` **must** already be in height-sorted order.
///
/// This function is generic in `F` and only relies on CUDA memory operations.
#[instrument(skip_all)]
pub fn stacked_matrix<F: Field>(
    l_skip: usize,
    n_stack: usize,
    traces: &[&DeviceMatrix<F>],
) -> Result<(DeviceMatrix<F>, StackedLayout), ProverError> {
    let sorted_meta = traces
        .iter()
        .enumerate()
        .map(|(idx, trace)| {
            // height cannot be zero:
            let prism_dim = log2_strict_usize(trace.height());
            let n = prism_dim.checked_sub(l_skip).expect("log_height < l_skip");
            (idx, trace.width(), l_skip + n)
        })
        .collect_vec();
    debug_assert!(sorted_meta.is_sorted_by(|a, b| a.2 >= b.2));
    let mut layout = StackedLayout::new(l_skip + n_stack, sorted_meta);
    let total_cells: usize = traces.iter().map(|t| t.height() * t.width()).sum();
    let height = 1usize << (l_skip + n_stack);
    let width = total_cells.div_ceil(height);

    let q_buf = DeviceBuffer::<F>::with_capacity(width.checked_mul(height).unwrap());
    q_buf.fill_zero()?;
    for (mat_idx, j, s) in &mut layout.sorted_cols {
        let start = s.col_idx * height + s.row_idx;
        let trace = traces[*mat_idx];
        let t_ht = trace.height();
        debug_assert_eq!(t_ht, 1 << s.log_height);
        // SAFETY: matrix buffers are allocated correctly with respect to dimensions
        unsafe {
            let src = trace.buffer().as_ptr().add(*j * t_ht);
            let dst = q_buf.as_mut_ptr().add(start);
            // D2D memcpy
            cuda_memcpy::<true, true>(
                dst as *mut c_void,
                src as *const c_void,
                t_ht * size_of::<F>(),
            )?;
        }
    }
    Ok((DeviceMatrix::new(Arc::new(q_buf), height, width), layout))
}

/// Computes the Reed-Solomon codeword of each column vector of `eval_matrix` where the rate is
/// `2^{-log_blowup}`. The column vectors are treated as evaluations of a prismalinear extension on
/// a hyperprism.
#[instrument(skip_all)]
pub fn rs_code_matrix(
    l_skip: usize,
    log_blowup: usize,
    eval_matrix: &DeviceMatrix<F>,
) -> Result<DeviceMatrix<F>, ProverError> {
    let height = eval_matrix.height();
    debug_assert!(height >= (1 << l_skip));
    let width = eval_matrix.width();
    let codeword_height = height.checked_shl(log_blowup as u32).unwrap();
    // D2D copy so we can do in-place Ple::from_evals
    let mut ple_coeffs = eval_matrix.buffer().device_copy()?;
    // For univariate coordinate, perform inverse NTT for each 2^l_skip chunk per column: (width
    // cols) * (height / 2^l_skip chunks per col). Use natural ordering.
    let num_uni_poly = (width * (height >> l_skip)) as u32;
    batch_ntt(&ple_coeffs, l_skip as u32, 0, num_uni_poly, true, true);
    let n = log2_strict_usize(height) - l_skip;
    let total_len = ple_coeffs.len();
    debug_assert_eq!(total_len, height * width);
    // Go through coordinates X_1, ..., X_n and interpolate each one from s(0), s(1) -> s(0) +
    // (s(1) - s(0)) X_i
    for i in 0..n {
        let step = 1u32 << (l_skip + i);
        let span = step << 1;
        debug_assert_eq!(height % (span as usize), 0);
        // SAFETY: `ple_coeffs` is properly initialized and step is in bounds.
        unsafe {
            ple_interpolate_stage(&mut ple_coeffs, step)?;
        }
    }
    let codewords = DeviceBuffer::<F>::with_capacity(codeword_height * width);
    // The following two kernels together perform coset NTT for `width` polys from `height ->
    // codeword_height` size domains. SAFETY: `codewords` is allocated for `width` polys of
    // `codeword_height` each, and we expand from `ple_coeffs` which is `width` polys of `height`
    // each.
    unsafe {
        batch_expand_pad(
            &codewords,
            &ple_coeffs,
            width as u32,
            codeword_height as u32,
            height as u32,
        )?;
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
        let (stacked_mat, _layout) = stacked_matrix(0, 2, &mat_refs).unwrap();
        assert_eq!(stacked_mat.height(), 4);
        assert_eq!(stacked_mat.width(), 2);
        let stacked_h_mat = transport_matrix_d2h_col_major(&stacked_mat).unwrap();
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
        let stacked_h_mat = transport_matrix_d2h_col_major(&stacked_mat).unwrap();
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
}
