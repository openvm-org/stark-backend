use getset::{CopyGetters, Getters};
use itertools::Itertools;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{Field, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    prover::{col_maj_idx, poly::Ple, ColMajorMatrix, MatrixView, StridedColMajorMatrixView},
    Digest, F,
};

#[derive(Clone, Serialize, Deserialize, Debug, CopyGetters)]
pub struct StackedLayout {
    /// The minimum log2 height of a stacked slice. When stacking columns with smaller height, the
    /// column is expanded to `2^l_skip` by striding.
    #[getset(get_copy = "pub")]
    l_skip: usize,
    /// Stacked height
    #[getset(get_copy = "pub")]
    height: usize,
    /// Stacked width
    #[getset(get_copy = "pub")]
    width: usize,
    /// The columns of the unstacked matrices in sorted order. Each entry `(matrix index, column
    /// index, coordinate)` contains the pointer `(matrix index, column index)` to a column of the
    /// unstacked collection of matrices as well as `coordinate` which is a pointer to where the
    /// column starts in the stacked matrix.
    pub sorted_cols: Vec<(
        usize, /* unstacked matrix index */
        usize, /* unstacked column index */
        StackedSlice,
    )>,
    /// `mat_starts[mat_idx]` is the index in `sorted_cols` where the matrix with index `mat_idx`
    /// starts.
    pub mat_starts: Vec<usize>,
}

/// Pointer to the location of a sub-column within the stacked matrix.
/// This struct contains length information, but information from [StackedLayout] (namely `l_skip`)
/// is needed to determine if this is a strided slice or not.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, CopyGetters, derive_new::new)]
pub struct StackedSlice {
    pub col_idx: usize,
    pub row_idx: usize,
    /// The true log height. If `>= l_skip`, no striding. Otherwise striding by `2^{l_skip -
    /// log_height}`.
    #[getset(get_copy = "pub")]
    log_height: usize,
}

impl StackedSlice {
    #[inline(always)]
    pub fn len(&self, l_skip: usize) -> usize {
        Self::_len(self.log_height, l_skip)
    }

    #[inline(always)]
    pub fn stride(&self, l_skip: usize) -> usize {
        1 << l_skip.saturating_sub(self.log_height)
    }

    #[inline(always)]
    fn _len(log_height: usize, l_skip: usize) -> usize {
        if l_skip <= log_height {
            1 << log_height
        } else {
            1 << l_skip
        }
    }
}

#[derive(Clone, Debug, Getters, CopyGetters, Serialize, Deserialize)]
pub struct MerkleTree<F, Digest> {
    /// The matrix that is used to form the leaves of the Merkle tree, which are
    /// in turn hashed into the bottom digest layer.
    ///
    /// This is typically the codeword matrix in hash-based PCS.
    #[getset(get = "pub")]
    pub(crate) backing_matrix: ColMajorMatrix<F>,
    #[getset(get = "pub")]
    pub(crate) digest_layers: Vec<Vec<Digest>>,
    #[getset(get_copy = "pub")]
    pub(crate) rows_per_query: usize,
}

#[derive(Clone, Serialize, Deserialize, derive_new::new)]
pub struct StackedPcsData<F, Digest> {
    /// Layout of the unstacked collection of matrices within the stacked matrix.
    pub layout: StackedLayout,
    /// The stacked matrix of evaluations with height `2^{l_skip + n_stack}`.
    pub matrix: ColMajorMatrix<F>,
    /// Merkle tree of the Reed-Solomon codewords of the stacked matrix.
    /// Depends on `k_whir` parameter.
    pub tree: MerkleTree<F, Digest>,
}

impl<F, Digest: Clone> StackedPcsData<F, Digest> {
    /// Returns the root of the Merkle tree.
    pub fn commit(&self) -> Digest {
        self.tree.root()
    }

    pub fn mat_view(&self, unstacked_mat_idx: usize) -> StridedColMajorMatrixView<'_, F> {
        self.layout.mat_view(unstacked_mat_idx, &self.matrix)
    }
}

#[instrument(level = "info", skip_all)]
pub fn stacked_commit(
    l_skip: usize,
    n_stack: usize,
    log_blowup: usize,
    k_whir: usize,
    traces: &[&ColMajorMatrix<F>],
) -> (Digest, StackedPcsData<F, Digest>) {
    let (q_trace, layout) = stacked_matrix(l_skip, n_stack, traces);
    let rs_matrix = rs_code_matrix(l_skip, log_blowup, &q_trace);
    let tree = MerkleTree::new(rs_matrix, 1 << k_whir);
    let root = tree.root();
    let data = StackedPcsData::new(layout, q_trace, tree);
    (root, data)
}

impl StackedLayout {
    /// Computes the layout of greedily stacking columns with dimension metadata given by `sorted`
    /// into a stacked matrix.
    /// - `l_skip` is a threshold log2 height: if a column has height less than `2^l_skip`, it is
    ///   stacked as a column of height `2^l_skip` with stride `2^{l_skip - log_height}`.
    /// - `log_stacked_height` is the log2 height of the stacked matrix.
    /// - `sorted` is Vec of `(width, log_height)` that must already be **sorted** in descending
    ///   order of `log_height`.
    pub fn new(
        l_skip: usize,
        log_stacked_height: usize,
        sorted: Vec<(usize /* width */, usize /* log_height */)>,
    ) -> Self {
        debug_assert!(l_skip <= log_stacked_height);
        debug_assert!(sorted.is_sorted_by(|a, b| a.1 >= b.1));
        let mut sorted_cols = Vec::with_capacity(sorted.len());
        let mut mat_starts = Vec::new();
        let mut col_idx = 0;
        let mut row_idx = 0;
        for (mat_idx, (width, log_ht)) in sorted.into_iter().enumerate() {
            mat_starts.push(sorted_cols.len());
            if width == 0 {
                continue;
            }
            assert!(
                log_ht <= log_stacked_height,
                "log_height={log_ht} > log_stacked_height={log_stacked_height}"
            );
            for j in 0..width {
                let slice_len = StackedSlice::_len(log_ht, l_skip);
                if row_idx + slice_len > (1 << log_stacked_height) {
                    assert_eq!(row_idx, 1 << log_stacked_height);
                    col_idx += 1;
                    row_idx = 0;
                }
                let slice = StackedSlice {
                    col_idx,
                    row_idx,
                    log_height: log_ht,
                };
                sorted_cols.push((mat_idx, j, slice));
                row_idx += slice_len;
            }
        }
        let stacked_width = col_idx + usize::from(row_idx != 0);
        debug_assert_eq!(
            stacked_width,
            sorted_cols
                .iter()
                .map(|(_, _, slice)| slice.col_idx + 1)
                .max()
                .unwrap_or(0)
        );
        Self {
            l_skip,
            height: 1 << log_stacked_height,
            width: stacked_width,
            sorted_cols,
            mat_starts,
        }
    }

    /// Raw unsafe constructor
    pub fn from_raw_parts(
        l_skip: usize,
        log_stacked_height: usize,
        sorted_cols: Vec<(usize, usize, StackedSlice)>,
    ) -> Self {
        let height = 1 << log_stacked_height;
        let width = sorted_cols
            .iter()
            .map(|(_, _, slice)| slice.col_idx + 1)
            .max()
            .unwrap_or(0);
        let mut mat_starts = Vec::new();
        for (idx, (mat_idx, _, _)) in sorted_cols.iter().enumerate() {
            if idx == 0 || *mat_idx + 1 != mat_starts.len() {
                assert_eq!(*mat_idx, mat_starts.len());
                mat_starts.push(idx);
            }
        }
        Self {
            l_skip,
            height,
            width,
            sorted_cols,
            mat_starts,
        }
    }

    pub fn unstacked_slices_iter(&self) -> impl Iterator<Item = &StackedSlice> {
        self.sorted_cols.iter().map(|(_, _, s)| s)
    }

    /// `(mat_idx, col_idx)` should be indexing into the unstacked collection of matrices.
    pub fn get(&self, mat_idx: usize, col_idx: usize) -> Option<&StackedSlice> {
        let idx = self.mat_starts[mat_idx];
        if idx + col_idx >= self.sorted_cols.len() {
            return None;
        }
        let (mat_idx1, col_idx1, s) = &self.sorted_cols[idx + col_idx];
        debug_assert_eq!(*mat_idx1, mat_idx);
        debug_assert_eq!(*col_idx1, col_idx);
        Some(s)
    }

    pub fn width_of(&self, mat_idx: usize) -> usize {
        let start_idx = self.mat_starts[mat_idx];
        debug_assert_eq!(self.sorted_cols[start_idx].0, mat_idx);
        debug_assert_eq!(self.sorted_cols[start_idx].1, 0);
        let next_idx = *self
            .mat_starts
            .get(mat_idx + 1)
            .unwrap_or(&self.sorted_cols.len());
        debug_assert_ne!(next_idx, usize::MAX);
        next_idx - start_idx
    }

    /// Due to the definition of stacking, in a column major matrix the lifted columns of the
    /// unstacked matrix will always be contiguous in memory within the stacked matrix, so we
    /// can return the sub-view.
    pub fn mat_view<'a, F>(
        &self,
        unstacked_mat_idx: usize,
        stacked_matrix: &'a ColMajorMatrix<F>,
    ) -> StridedColMajorMatrixView<'a, F> {
        let col_slices = self
            .sorted_cols
            .iter()
            .filter(|(m, _, _)| *m == unstacked_mat_idx)
            .collect_vec();
        let width = col_slices.len();
        let s = &col_slices[0].2;
        let lifted_height = s.len(self.l_skip);
        let stride = s.stride(self.l_skip);
        let start = col_maj_idx(s.row_idx, s.col_idx, stacked_matrix.height());
        StridedColMajorMatrixView::new(
            &stacked_matrix.values[start..start + lifted_height * width],
            width,
            stride,
        )
    }
}

/// The `traces` **must** already be in height-sorted order.
#[instrument(skip_all)]
pub fn stacked_matrix<F: Field>(
    l_skip: usize,
    n_stack: usize,
    traces: &[&ColMajorMatrix<F>],
) -> (ColMajorMatrix<F>, StackedLayout) {
    let sorted_meta = traces
        .iter()
        .map(|trace| {
            // height cannot be zero:
            let log_height = log2_strict_usize(trace.height());
            (trace.width(), log_height)
        })
        .collect_vec();
    let mut layout = StackedLayout::new(l_skip, l_skip + n_stack, sorted_meta);
    let total_cells: usize = traces
        .iter()
        .map(|t| t.height().max(1 << l_skip) * t.width())
        .sum();
    let height = 1usize << (l_skip + n_stack);
    let width = total_cells.div_ceil(height);

    let mut q_mat = F::zero_vec(width.checked_mul(height).unwrap());
    for (mat_idx, j, s) in &mut layout.sorted_cols {
        let start = s.col_idx * height + s.row_idx;
        let t_col = traces[*mat_idx].column(*j);
        debug_assert_eq!(t_col.len(), 1 << s.log_height);
        if s.log_height >= l_skip {
            q_mat[start..start + t_col.len()].copy_from_slice(t_col);
        } else {
            // t_col height is smaller than 2^l_skip, so we stride
            let stride = s.stride(l_skip);
            for (i, val) in t_col.iter().enumerate() {
                q_mat[start + i * stride] = *val;
            }
        }
    }
    (ColMajorMatrix::new(q_mat, width), layout)
}

/// Computes the Reed-Solomon codeword of each column vector of `eval_matrix` where the rate is
/// `2^{-log_blowup}`. The column vectors are treated as evaluations of a prismalinear extension on
/// a hyperprism.
#[instrument(skip_all)]
pub fn rs_code_matrix<F: TwoAdicField + Ord>(
    l_skip: usize,
    log_blowup: usize,
    eval_matrix: &ColMajorMatrix<F>,
) -> ColMajorMatrix<F> {
    let height = eval_matrix.height();
    let codewords: Vec<_> = eval_matrix
        .values
        .par_chunks_exact(height)
        .map(|column_evals| {
            let ple = Ple::from_evaluations(l_skip, column_evals);
            let mut coeffs = ple.coeffs;
            // Compute RS codeword on a prismalinear polynomial in coefficient form:
            // We use that the coefficients are in a basis that exactly corresponds to the standard
            // ith_basis_element univariate basis. Hence RS codeword is just cosetDFT on the
            // relevant smooth domain
            let dft = Radix2DitParallel::default();
            coeffs.resize(height.checked_shl(log_blowup as u32).unwrap(), F::ZERO);
            dft.dft(coeffs)
        })
        .collect::<Vec<_>>()
        .concat();

    ColMajorMatrix::new(codewords, eval_matrix.width())
}

impl<F, Digest> MerkleTree<F, Digest> {
    pub fn query_stride(&self) -> usize {
        self.digest_layers[0].len()
    }

    pub fn proof_depth(&self) -> usize {
        self.digest_layers.len() - 1
    }
}

impl<F, Digest: Clone> MerkleTree<F, Digest> {
    pub fn root(&self) -> Digest {
        self.digest_layers.last().unwrap()[0].clone()
    }

    pub fn query_merkle_proof(&self, query_idx: usize) -> Vec<Digest> {
        let stride = self.query_stride();
        assert!(
            query_idx < stride,
            "query_idx {query_idx} out of bounds for query_stride {stride}"
        );

        let mut idx = query_idx;
        let mut proof = Vec::with_capacity(self.proof_depth());
        for layer in self.digest_layers.iter().take(self.proof_depth()) {
            let sibling = layer[idx ^ 1].clone();
            proof.push(sibling);
            idx >>= 1;
        }
        proof
    }
}

mod poseidon2_merkle_tree {
    use p3_field::ExtensionField;

    use super::*;
    use crate::{
        poseidon2::sponge::{poseidon2_compress, poseidon2_hash_slice},
        Digest, F,
    };

    impl<EF> MerkleTree<EF, Digest>
    where
        EF: ExtensionField<F>,
    {
        #[instrument(name = "merkle_tree", skip_all)]
        pub fn new(matrix: ColMajorMatrix<EF>, rows_per_query: usize) -> Self {
            let height = matrix.height();
            assert!(height > 0);
            assert!(rows_per_query.is_power_of_two());
            let num_leaves = height.next_power_of_two();
            assert!(
                rows_per_query <= num_leaves,
                "rows_per_query ({rows_per_query}) must not exceed the number of Merkle leaves ({num_leaves})"
            );
            let row_hashes: Vec<_> = (0..num_leaves)
                .into_par_iter()
                .map(|r| {
                    let hash_input: Vec<F> = Self::row_iter(&matrix, r)
                        .flat_map(|ef| ef.as_basis_coefficients_slice().to_vec())
                        .collect();
                    poseidon2_hash_slice(&hash_input)
                })
                .collect();

            let query_stride = num_leaves / rows_per_query;
            let mut query_digest_layer = row_hashes;
            // For the first log2(rows_per_query) layers, we hash in `query_stride` pairs and don't
            // need to store the digest layers
            for _ in 0..log2_strict_usize(rows_per_query) {
                let prev_layer = query_digest_layer;
                query_digest_layer = (0..prev_layer.len() / 2)
                    .into_par_iter()
                    .map(|i| {
                        let x = i / query_stride;
                        let y = i % query_stride;
                        let left = prev_layer[2 * x * query_stride + y];
                        let right = prev_layer[(2 * x + 1) * query_stride + y];
                        poseidon2_compress(left, right)
                    })
                    .collect();
            }

            let mut digest_layers = vec![query_digest_layer];
            while digest_layers.last().unwrap().len() > 1 {
                let prev_layer = digest_layers.last().unwrap();
                let layer: Vec<_> = prev_layer
                    .par_chunks_exact(2)
                    .map(|pair| poseidon2_compress(pair[0], pair[1]))
                    .collect();
                digest_layers.push(layer);
            }

            Self {
                backing_matrix: matrix,
                digest_layers,
                rows_per_query,
            }
        }

        /// # Safety
        /// - Caller must ensure that `digest_layers` are correctly constructed Merkle hashes for
        ///   the Merkle tree.
        pub unsafe fn from_raw_parts(
            backing_matrix: ColMajorMatrix<EF>,
            digest_layers: Vec<Vec<Digest>>,
            rows_per_query: usize,
        ) -> Self {
            Self {
                backing_matrix,
                digest_layers,
                rows_per_query,
            }
        }

        /// Returns the ordered set of opened rows for the given query index.
        /// The rows are { query_idx + t * query_stride() } for t in 0..rows_per_query.
        pub fn get_opened_rows(&self, index: usize) -> Vec<Vec<EF>> {
            let query_stride = self.query_stride();
            assert!(
                index < query_stride,
                "index {index} out of bounds for query_stride {query_stride}"
            );

            let rows_per_query = self.rows_per_query;
            let width = self.backing_matrix.width();
            let mut preimage = Vec::with_capacity(rows_per_query);
            for row_offset in 0..rows_per_query {
                let row_idx = row_offset * query_stride + index;
                let row = Self::row_iter(&self.backing_matrix, row_idx).collect_vec();
                debug_assert_eq!(
                    row.len(),
                    width,
                    "row width mismatch: expected {width}, got {}",
                    row.len()
                );
                preimage.push(row);
            }
            preimage
        }

        fn row_iter(matrix: &ColMajorMatrix<EF>, index: usize) -> impl Iterator<Item = EF> + '_ {
            (0..matrix.width()).map(move |c| matrix.get(index, c).copied().unwrap_or(EF::ZERO))
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{prover::ColMajorMatrix, F};

    #[test]
    fn test_stacked_matrix_manual_0() {
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| ColMajorMatrix::new(c, 1))
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let (stacked_mat, layout) = stacked_matrix(0, 2, &mat_refs);
        assert_eq!(stacked_mat.height(), 4);
        assert_eq!(stacked_mat.width(), 2);
        assert_eq!(
            stacked_mat.values,
            [1, 2, 3, 4, 5, 6, 7, 0].map(F::from_u32).to_vec()
        );
        assert_eq!(layout.mat_starts, vec![0, 1, 2]);
    }

    #[test]
    fn test_stacked_matrix_manual_strided_0() {
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| ColMajorMatrix::new(c, 1))
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let (stacked_mat, _layout) = stacked_matrix(2, 0, &mat_refs);
        assert_eq!(stacked_mat.height(), 4);
        assert_eq!(stacked_mat.width(), 3);
        assert_eq!(
            stacked_mat.values,
            [1, 2, 3, 4, 5, 0, 6, 0, 7, 0, 0, 0]
                .map(F::from_u32)
                .to_vec()
        );
    }

    #[test]
    fn test_stacked_matrix_manual_strided_1() {
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| ColMajorMatrix::new(c, 1))
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let (stacked_mat, _layout) = stacked_matrix(3, 0, &mat_refs);
        assert_eq!(stacked_mat.height(), 8);
        assert_eq!(stacked_mat.width(), 3);
        assert_eq!(
            stacked_mat.values,
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
