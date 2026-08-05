use std::cmp::Reverse;

use getset::{CopyGetters, Getters};
use itertools::Itertools;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    hasher::MerkleHasher,
    prover::{
        col_maj_idx, error::StackedPcsError, poly::eval_to_coeff_rs_message, ColMajorMatrix,
        MatrixDimensions, MatrixView, StridedColMajorMatrixView,
    },
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
    /// The chunked columns of the unstacked matrices in stacking order (descending slice size).
    /// Each entry `(matrix index, column index, slice)` points to a power-of-two *chunk* of a
    /// column of the unstacked collection of matrices, as well as where the chunk lives in the
    /// stacked matrix.
    ///
    /// A matrix whose height is a power of two contributes exactly one slice per column. A matrix
    /// with non-power-of-two height `h` (which must be a multiple of `2^l_skip`) contributes one
    /// slice per column per set bit of `h` (its binary-ruler chunk decomposition), so a `(matrix
    /// index, column index)` pair may appear multiple times.
    pub sorted_cols: Vec<(
        usize, /* unstacked matrix index */
        usize, /* unstacked column index */
        StackedSlice,
    )>,
    /// Width of each unstacked matrix.
    pub mat_widths: Vec<usize>,
    /// Height (number of *used* rows, not padded to a power of two) of each unstacked matrix.
    pub mat_heights: Vec<usize>,
}

/// Pointer to the location of a sub-column chunk within the stacked matrix.
/// This struct contains length information, but information from [StackedLayout] (namely `l_skip`)
/// is needed to determine if this is a strided slice or not.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, CopyGetters, derive_new::new)]
pub struct StackedSlice {
    /// Column within the stacked matrix.
    pub col_idx: usize,
    /// Starting row within the stacked matrix. Always a multiple of `2^max(log_height, l_skip)`.
    pub row_idx: usize,
    /// The true log height of this chunk. If `>= l_skip`, no striding. Otherwise striding by
    /// `2^{l_skip - log_height}`.
    #[getset(get_copy = "pub")]
    log_height: usize,
    /// Row offset of this chunk within its unstacked column. Always a multiple of
    /// `2^log_height`; zero for power-of-two-height matrices (single chunk).
    pub row_offset: usize,
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

/// Decomposes a matrix height into its binary-ruler chunk decomposition `(row_offset,
/// log_height)`, in descending chunk size order.
///
/// - `height == 0` is rejected.
/// - `height < 2^l_skip` must be a power of two (a single strided chunk).
/// - `height >= 2^l_skip` must be a multiple of `2^l_skip`; each set bit of `height` becomes one
///   chunk. Because chunk sizes strictly decrease, each `row_offset` is automatically a multiple of
///   the chunk size.
pub fn chunk_decompose(
    l_skip: usize,
    height: usize,
) -> Result<impl Iterator<Item = (usize, usize)>, StackedPcsError> {
    if height == 0 {
        return Err(StackedPcsError::LayoutInvalidHeight { height, l_skip });
    }
    if height < (1 << l_skip) {
        if !height.is_power_of_two() {
            return Err(StackedPcsError::LayoutInvalidHeight { height, l_skip });
        }
    } else if height % (1 << l_skip) != 0 {
        return Err(StackedPcsError::LayoutInvalidHeight { height, l_skip });
    }
    let mut row_offset = 0usize;
    Ok((0..usize::BITS as usize)
        .rev()
        .filter(move |&a| height & (1 << a) != 0)
        .map(move |a| {
            let off = row_offset;
            row_offset += 1 << a;
            (off, a)
        }))
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
    pub fn commit(&self) -> Result<Digest, StackedPcsError> {
        self.tree.root()
    }

    pub fn mat_view(&self, unstacked_mat_idx: usize) -> StridedColMajorMatrixView<'_, F> {
        self.layout.mat_view(unstacked_mat_idx, &self.matrix)
    }
}

#[instrument(level = "info", skip_all)]
#[allow(clippy::type_complexity)]
pub fn stacked_commit<H: MerkleHasher>(
    hasher: &H,
    l_skip: usize,
    n_stack: usize,
    log_blowup: usize,
    k_whir: usize,
    traces: &[&ColMajorMatrix<H::F>],
) -> Result<(H::Digest, StackedPcsData<H::F, H::Digest>), StackedPcsError>
where
    H::F: TwoAdicField + Ord,
    H::Digest: Copy,
{
    let (q_trace, layout) = stacked_matrix(l_skip, n_stack, traces)?;
    let rs_matrix = rs_code_matrix(l_skip, log_blowup, &q_trace)?;
    let tree = MerkleTree::new(hasher, rs_matrix, 1 << k_whir)?;
    let root = tree.root()?;
    let data = StackedPcsData::new(layout, q_trace, tree);
    Ok((root, data))
}

impl StackedLayout {
    /// Computes the layout of greedily stacking columns with dimension metadata given by `sorted`
    /// into a stacked matrix.
    /// - `l_skip` is a threshold log2 height: if a column has height less than `2^l_skip`, it is
    ///   stacked as a column of height `2^l_skip` with stride `2^{l_skip - log_height}`.
    /// - `log_stacked_height` is the log2 height of the stacked matrix.
    /// - `sorted` is Vec of `(width, height)` that must already be **sorted** in descending order
    ///   of `height`. Heights need not be powers of two: a height `>= 2^l_skip` that is a multiple
    ///   of `2^l_skip` is decomposed into binary-ruler chunks, each stacked as its own slice.
    pub fn new(
        l_skip: usize,
        log_stacked_height: usize,
        sorted: Vec<(usize /* width */, usize /* height */)>,
    ) -> Result<Self, StackedPcsError> {
        debug_assert!(l_skip <= log_stacked_height);
        debug_assert!(sorted.is_sorted_by(|a, b| a.1 >= b.1));
        let mat_widths = sorted.iter().map(|&(w, _)| w).collect_vec();
        let mat_heights = sorted.iter().map(|&(_, h)| h).collect_vec();

        // Chunk units `(mat_idx, row_offset, chunk_log_height)` in stacking order: stable sort by
        // descending chunk size. Stability keeps chunks deterministic: ties broken by (matrix,
        // then descending chunk order within a matrix).
        let mut chunks: Vec<(usize, usize, usize)> = Vec::new();
        for (mat_idx, &(width, height)) in sorted.iter().enumerate() {
            if width == 0 {
                continue;
            }
            for (row_offset, log_ht) in chunk_decompose(l_skip, height)? {
                if log_ht > log_stacked_height {
                    return Err(StackedPcsError::LayoutHeightExceeded {
                        log_height: log_ht,
                        log_stacked_height,
                    });
                }
                chunks.push((mat_idx, row_offset, log_ht));
            }
        }
        chunks.sort_by_key(|&(_, _, log_ht)| Reverse(log_ht));

        let mut sorted_cols = Vec::new();
        let mut col_idx = 0;
        let mut row_idx = 0;
        for (mat_idx, row_offset, log_ht) in chunks {
            for j in 0..mat_widths[mat_idx] {
                let slice_len = StackedSlice::_len(log_ht, l_skip);
                if row_idx + slice_len > (1 << log_stacked_height) {
                    if row_idx != 1 << log_stacked_height {
                        return Err(StackedPcsError::LayoutRowOverflow {
                            col_idx,
                            stacked_height: 1 << log_stacked_height,
                        });
                    }
                    col_idx += 1;
                    row_idx = 0;
                }
                let slice = StackedSlice {
                    col_idx,
                    row_idx,
                    log_height: log_ht,
                    row_offset,
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
        Ok(Self {
            l_skip,
            height: 1 << log_stacked_height,
            width: stacked_width,
            sorted_cols,
            mat_widths,
            mat_heights,
        })
    }

    /// Raw unsafe constructor. The caller must guarantee that `sorted_cols` is consistent with
    /// `mat_widths`/`mat_heights` (see [StackedLayout::new]).
    pub fn from_raw_parts(
        l_skip: usize,
        log_stacked_height: usize,
        sorted_cols: Vec<(usize, usize, StackedSlice)>,
        mat_widths: Vec<usize>,
        mat_heights: Vec<usize>,
    ) -> Result<Self, StackedPcsError> {
        let height = 1 << log_stacked_height;
        let width = sorted_cols
            .iter()
            .map(|(_, _, slice)| slice.col_idx + 1)
            .max()
            .unwrap_or(0);
        for (mat_idx, _, _) in &sorted_cols {
            if *mat_idx >= mat_widths.len() {
                return Err(StackedPcsError::LayoutRawPartsMatIdx {
                    mat_idx: *mat_idx,
                    mat_starts_len: mat_widths.len(),
                });
            }
        }
        Ok(Self {
            l_skip,
            height,
            width,
            sorted_cols,
            mat_widths,
            mat_heights,
        })
    }

    /// Number of unstacked matrices in this layout.
    pub fn num_mats(&self) -> usize {
        self.mat_widths.len()
    }

    /// Index in `sorted_cols` where the slices of matrix `mat_idx` start.
    ///
    /// Only valid for layouts where every matrix has a single chunk per column (all power-of-two
    /// heights): there, slices are grouped per matrix in matrix order.
    pub fn mat_start(&self, mat_idx: usize) -> usize {
        debug_assert_eq!(
            self.sorted_cols.len(),
            self.num_claims(),
            "mat_start requires a single chunk per column"
        );
        self.claim_idx(mat_idx, 0)
    }

    pub fn unstacked_slices_iter(&self) -> impl Iterator<Item = &StackedSlice> {
        self.sorted_cols.iter().map(|(_, _, s)| s)
    }

    /// `(mat_idx, col_idx)` should be indexing into the unstacked collection of matrices.
    /// For a chunked column (non-power-of-two height matrix), returns the first (largest) chunk,
    /// which starts at `row_offset == 0`.
    pub fn get(&self, mat_idx: usize, col_idx: usize) -> Option<&StackedSlice> {
        self.sorted_cols
            .iter()
            .find(|(m, c, _)| *m == mat_idx && *c == col_idx)
            .map(|(_, _, s)| s)
    }

    pub fn width_of(&self, mat_idx: usize) -> usize {
        self.mat_widths[mat_idx]
    }

    /// The number of used (unpadded) rows of the unstacked matrix.
    pub fn height_of(&self, mat_idx: usize) -> usize {
        self.mat_heights[mat_idx]
    }

    /// Index of the opening claim for unstacked column `(mat_idx, col_idx)` when claims are
    /// ordered per matrix, then per column. Chunked slices of the same column share one claim.
    pub fn claim_idx(&self, mat_idx: usize, col_idx: usize) -> usize {
        self.mat_widths[..mat_idx].iter().sum::<usize>() + col_idx
    }

    /// Total number of opening claims (= total number of unstacked columns).
    pub fn num_claims(&self) -> usize {
        self.mat_widths.iter().sum()
    }

    /// Due to the definition of stacking, in a column major matrix the lifted columns of the
    /// unstacked matrix will always be contiguous in memory within the stacked matrix, so we
    /// can return the sub-view.
    ///
    /// Only valid for matrices with a single chunk per column (power-of-two height or height below
    /// `2^l_skip`), such as preprocessed and cached matrices.
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
        assert_eq!(
            width, self.mat_widths[unstacked_mat_idx],
            "mat_view requires a single chunk per column"
        );
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

/// The `traces` **must** already be in height-sorted (descending) order. Trace heights need not be
/// powers of two: any height `>= 2^l_skip` that is a multiple of `2^l_skip` is supported by
/// stacking its binary-ruler chunk decomposition; heights `< 2^l_skip` must be powers of two.
#[instrument(skip_all)]
pub fn stacked_matrix<F: Field>(
    l_skip: usize,
    n_stack: usize,
    traces: &[&ColMajorMatrix<F>],
) -> Result<(ColMajorMatrix<F>, StackedLayout), StackedPcsError> {
    let sorted_meta = traces
        .iter()
        .map(|trace| (trace.width(), trace.height()))
        .collect_vec();
    let mut layout = StackedLayout::new(l_skip, l_skip + n_stack, sorted_meta)?;
    let total_cells: usize = traces
        .iter()
        .map(|t| t.height().max(1 << l_skip) * t.width())
        .sum();
    let height = 1usize << (l_skip + n_stack);
    let width = total_cells.div_ceil(height);

    let mut q_mat = F::zero_vec(
        width
            .checked_mul(height)
            .ok_or(StackedPcsError::StackedMatrixOverflow)?,
    );
    for (mat_idx, j, s) in &mut layout.sorted_cols {
        let start = s.col_idx * height + s.row_idx;
        let t_col = traces[*mat_idx].column(*j);
        if s.log_height >= l_skip {
            let chunk = &t_col[s.row_offset..s.row_offset + (1 << s.log_height)];
            q_mat[start..start + chunk.len()].copy_from_slice(chunk);
        } else {
            // t_col height is smaller than 2^l_skip, so we stride
            debug_assert_eq!(s.row_offset, 0);
            debug_assert_eq!(t_col.len(), 1 << s.log_height);
            let stride = s.stride(l_skip);
            for (i, val) in t_col.iter().enumerate() {
                q_mat[start + i * stride] = *val;
            }
        }
    }
    Ok((ColMajorMatrix::new(q_mat, width), layout))
}

/// Computes the Reed-Solomon codeword of each column vector of `eval_matrix` where the rate is
/// `2^{-log_blowup}`. The column vectors are treated as evaluations of a prismalinear extension on
/// a hyperprism.
#[instrument(skip_all)]
pub fn rs_code_matrix<F: TwoAdicField + Ord>(
    l_skip: usize,
    log_blowup: usize,
    eval_matrix: &ColMajorMatrix<F>,
) -> Result<ColMajorMatrix<F>, StackedPcsError> {
    let height = eval_matrix.height();
    let rs_height = height
        .checked_shl(log_blowup as u32)
        .ok_or(StackedPcsError::RsCodeShiftOverflow { height, log_blowup })?;
    let codewords: Vec<_> = eval_matrix
        .values
        .par_chunks_exact(height)
        .map(|column_evals| {
            // Convert column evaluations on `D × {0,1}^n` directly into the eval-to-coeff RS
            // coefficient vector, avoiding redundant interpolation work.
            let mut coeffs = eval_to_coeff_rs_message(l_skip, column_evals);

            // Compute RS codeword on the resulting univariate polynomial in coefficient form.
            let dft = Radix2DitParallel::default();
            coeffs.resize(rs_height, F::ZERO);
            dft.dft(coeffs)
        })
        .collect::<Vec<_>>()
        .concat();

    Ok(ColMajorMatrix::new(codewords, eval_matrix.width()))
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
    pub fn root(&self) -> Result<Digest, StackedPcsError> {
        Ok(self
            .digest_layers
            .last()
            .ok_or(StackedPcsError::MerkleTreeNoRoot)?[0]
            .clone())
    }

    pub fn query_merkle_proof(&self, query_idx: usize) -> Result<Vec<Digest>, StackedPcsError> {
        let stride = self.query_stride();
        if query_idx >= stride {
            return Err(StackedPcsError::MerkleTreeQueryOutOfBounds {
                query_idx,
                query_stride: stride,
            });
        }

        let mut idx = query_idx;
        let mut proof = Vec::with_capacity(self.proof_depth());
        for layer in self.digest_layers.iter().take(self.proof_depth()) {
            let sibling = layer[idx ^ 1].clone();
            proof.push(sibling);
            idx >>= 1;
        }
        Ok(proof)
    }
}

impl<EF: Field, Digest> MerkleTree<EF, Digest>
where
    Digest: Copy + Send + Sync,
{
    #[instrument(name = "merkle_tree", skip_all)]
    pub fn new<H: MerkleHasher<Digest = Digest>>(
        hasher: &H,
        matrix: ColMajorMatrix<EF>,
        rows_per_query: usize,
    ) -> Result<Self, StackedPcsError>
    where
        EF: ExtensionField<H::F>,
    {
        let height = matrix.height();
        if height == 0 {
            return Err(StackedPcsError::MerkleTreeEmptyMatrix);
        }
        if !rows_per_query.is_power_of_two() {
            return Err(StackedPcsError::MerkleTreeRowsPerQueryNotPow2 { rows_per_query });
        }
        let num_leaves = height.next_power_of_two();
        if rows_per_query > num_leaves {
            return Err(StackedPcsError::MerkleTreeRowsPerQueryExceeded {
                rows_per_query,
                num_leaves,
            });
        }
        let row_hashes: Vec<_> = (0..num_leaves)
            .into_par_iter()
            .map(|r| {
                let hash_input: Vec<H::F> = Self::row_iter(&matrix, r)
                    .flat_map(|ef| ef.as_basis_coefficients_slice().to_vec())
                    .collect();
                hasher.hash_slice(&hash_input)
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
                    hasher.compress(left, right)
                })
                .collect();
        }

        let mut digest_layers = vec![query_digest_layer];
        while digest_layers
            .last()
            .ok_or(StackedPcsError::MerkleTreeNoRoot)?
            .len()
            > 1
        {
            let prev_layer = digest_layers
                .last()
                .ok_or(StackedPcsError::MerkleTreeNoRoot)?;
            let layer: Vec<_> = prev_layer
                .par_chunks_exact(2)
                .map(|pair| hasher.compress(pair[0], pair[1]))
                .collect();
            digest_layers.push(layer);
        }

        Ok(Self {
            backing_matrix: matrix,
            digest_layers,
            rows_per_query,
        })
    }

    /// Construct a `MerkleTree` from pre-computed parts without validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    /// - `digest_layers` form a valid Merkle tree over `backing_matrix`: the leaf layer contains
    ///   correct hashes of the matrix rows and each subsequent layer contains correct compressions
    ///   of consecutive pairs from the previous layer, terminating in a single root digest.
    /// - `rows_per_query` is a power of two and does not exceed the number of leaves (i.e.,
    ///   `backing_matrix.height().next_power_of_two()`).
    /// - The leaf layer length equals `backing_matrix.height().next_power_of_two() /
    ///   rows_per_query`.
    ///
    /// Violating these invariants will produce incorrect Merkle proofs or panics
    /// in downstream query/verification code.
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
    pub fn get_opened_rows(&self, index: usize) -> Result<Vec<Vec<EF>>, StackedPcsError> {
        let query_stride = self.query_stride();
        if index >= query_stride {
            return Err(StackedPcsError::MerkleTreeOpenedRowsOutOfBounds {
                index,
                query_stride,
            });
        }

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
        Ok(preimage)
    }

    fn row_iter(matrix: &ColMajorMatrix<EF>, index: usize) -> impl Iterator<Item = EF> + '_ {
        (0..matrix.width()).map(move |c| matrix.get(index, c).copied().unwrap_or(EF::ZERO))
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use openvm_stark_sdk::config::baby_bear_poseidon2::*;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::prover::ColMajorMatrix;

    #[test]
    fn test_stacked_matrix_manual_0() {
        let columns = [vec![1, 2, 3, 4], vec![5, 6], vec![7]]
            .map(|v| v.into_iter().map(F::from_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| ColMajorMatrix::new(c, 1))
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        let (stacked_mat, layout) = stacked_matrix(0, 2, &mat_refs).unwrap();
        assert_eq!(stacked_mat.height(), 4);
        assert_eq!(stacked_mat.width(), 2);
        assert_eq!(
            stacked_mat.values,
            [1, 2, 3, 4, 5, 6, 7, 0].map(F::from_u32).to_vec()
        );
        assert_eq!(layout.mat_widths, vec![1, 1, 1]);
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
        let (stacked_mat, _layout) = stacked_matrix(2, 0, &mat_refs).unwrap();
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
        let (stacked_mat, _layout) = stacked_matrix(3, 0, &mat_refs).unwrap();
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

    #[test]
    fn test_chunk_decompose() {
        // 12 = 8 + 4 with l_skip = 2
        let chunks = chunk_decompose(2, 12).unwrap().collect_vec();
        assert_eq!(chunks, vec![(0, 3), (8, 2)]);
        // power of two: single chunk
        let chunks = chunk_decompose(2, 16).unwrap().collect_vec();
        assert_eq!(chunks, vec![(0, 4)]);
        // below 2^l_skip: single strided chunk, must be a power of two
        let chunks = chunk_decompose(3, 2).unwrap().collect_vec();
        assert_eq!(chunks, vec![(0, 1)]);
        assert!(chunk_decompose(3, 3).is_err());
        assert!(chunk_decompose(2, 6).is_err());
        assert!(chunk_decompose(2, 0).is_err());
    }

    #[test]
    fn test_stacked_matrix_chunked() {
        // Trace 0: height 12 = 8 + 4, width 1; trace 1: height 4, width 1; l_skip = 2.
        let columns = [(1..=12).collect_vec(), vec![101, 102, 103, 104]]
            .map(|v| v.into_iter().map(F::from_u32).collect_vec());
        let mats = columns
            .into_iter()
            .map(|c| ColMajorMatrix::new(c, 1))
            .collect_vec();
        let mat_refs = mats.iter().collect_vec();
        // stacked height 8: chunks sizes [8, 4, 4] -> columns: [rows 1..8], [rows 9..12, 101..104]
        let (stacked_mat, layout) = stacked_matrix(2, 1, &mat_refs).unwrap();
        assert_eq!(stacked_mat.height(), 8);
        assert_eq!(stacked_mat.width(), 2);
        assert_eq!(
            stacked_mat.values,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 101, 102, 103, 104]
                .map(F::from_u32)
                .to_vec()
        );
        // slice bookkeeping: trace 0 has two chunks with row offsets 0 and 8
        let slices_mat0 = layout
            .sorted_cols
            .iter()
            .filter(|(m, _, _)| *m == 0)
            .map(|(_, _, s)| (s.row_offset, s.log_height(), s.col_idx, s.row_idx))
            .collect_vec();
        assert_eq!(slices_mat0, vec![(0, 3, 0, 0), (8, 2, 1, 0)]);
        assert_eq!(layout.claim_idx(1, 0), 1);
        assert_eq!(layout.height_of(0), 12);
    }
}
