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
    Digest, F,
    prover::{ColMajorMatrix, ColMajorMatrixView, MatrixView, col_maj_idx, poly::Ple},
};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StackedLayout {
    /// The columns of the unstacked matrices in sorted order. Each entry `(matrix index, column
    /// index, coordinate)` contains the pointer `(matrix index, column index)` to a column of the
    /// unstacked collection of matrices as well as `coordinate` which is a pointer to where the
    /// column starts in the stacked matrix.
    pub sorted_cols: Vec<(usize, usize, StackedSlice)>,
}

impl StackedLayout {
    pub fn unstacked_slices_iter(&self) -> impl Iterator<Item = &StackedSlice> {
        self.sorted_cols.iter().map(|(_, _, s)| s)
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct StackedSlice {
    pub col_idx: usize,
    pub row_idx: usize,
    pub log_height: usize,
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
    /// The stacked matrix with height `2^{l_skip + n_stack}`.
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
    /// `sorted` is Vec of `(matrix index, width, log_height)` that must already be
    /// **sorted** in descending order of `log_height`.
    pub fn new(log_stacked_height: usize, sorted: Vec<(usize, usize, usize)>) -> Self {
        debug_assert!(sorted.is_sorted_by(|a, b| a.2 >= b.2));
        let mut sorted_cols = Vec::new();
        let mut col_idx = 0;
        let mut row_idx = 0;
        for (mat_idx, width, log_ht) in sorted {
            if width == 0 {
                continue;
            }
            assert!(
                log_ht <= log_stacked_height,
                "log_height={log_ht} > log_stacked_height={log_stacked_height}"
            );
            for j in 0..width {
                if row_idx + (1 << log_ht) > (1 << log_stacked_height) {
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
                row_idx += 1 << log_ht;
            }
        }
        Self { sorted_cols }
    }

    /// `(mat_idx, col_idx)` should be indexing into the unstacked collection of matrices.
    pub fn get(&self, mat_idx: usize, col_idx: usize) -> Option<&StackedSlice> {
        // TODO[jpw]: re-organize StackedLayout so this is O(1)
        self.sorted_cols
            .iter()
            .find(|&&(m, j, _)| m == mat_idx && j == col_idx)
            .map(|(_, _, coord)| coord)
    }

    /// Due to the definition of stacking, in a column major matrix the columns of the unstacked
    /// matrix will always be contiguous in memory within the stacked matrix, so we can return the
    /// sub-view.
    pub fn mat_view<'a>(
        &self,
        unstacked_mat_idx: usize,
        stacked_matrix: ColMajorMatrixView<'a, F>,
    ) -> ColMajorMatrixView<'a, F> {
        let col_slices = self
            .sorted_cols
            .iter()
            .filter(|(m, _, _)| *m == unstacked_mat_idx)
            .collect_vec();
        let width = col_slices.len();
        let s = &col_slices[0].2;
        let height = 1 << s.log_height;
        let start = col_maj_idx(s.row_idx, s.col_idx, stacked_matrix.height());
        ColMajorMatrixView::new(&stacked_matrix.values[start..start + height * width], width)
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
        .enumerate()
        .map(|(idx, trace)| {
            // height cannot be zero:
            let prism_dim = log2_strict_usize(trace.height());
            let n = prism_dim.checked_sub(l_skip).expect("log_height < l_skip");
            (idx, trace.width(), l_skip + n)
        })
        .collect_vec();
    let mut layout = StackedLayout::new(l_skip + n_stack, sorted_meta);
    let total_cells: usize = traces.iter().map(|t| t.values.len()).sum();
    let height = 1usize << (l_skip + n_stack);
    let width = total_cells.div_ceil(height);

    let mut q_mat = F::zero_vec(width.checked_mul(height).unwrap());
    for (mat_idx, j, s) in &mut layout.sorted_cols {
        let start = s.col_idx * height + s.row_idx;
        let t_col = traces[*mat_idx].column(*j);
        debug_assert_eq!(t_col.len(), 1 << s.log_height);
        q_mat[start..start + t_col.len()].copy_from_slice(t_col);
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
            // monomial univariate basis. Hence RS codeword is just cosetDFT on the
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
        Digest, F,
        poseidon2::sponge::{poseidon2_compress, poseidon2_hash_slice, poseidon2_tree_compress},
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
                        .flat_map(|ef| ef.as_base_slice().to_vec())
                        .collect();
                    poseidon2_hash_slice(&hash_input)
                })
                .collect();

            let query_stride = num_leaves / rows_per_query;
            let query_digest_layer: Vec<Digest> = (0..query_stride)
                .into_par_iter()
                .map(|q| {
                    let sub_row_hashes = (0..rows_per_query)
                        .map(|t| row_hashes[q + t * query_stride])
                        .collect_vec();
                    poseidon2_tree_compress(sub_row_hashes)
                })
                .collect();

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

        fn row_iter<'a>(
            matrix: &'a ColMajorMatrix<EF>,
            index: usize,
        ) -> impl Iterator<Item = EF> + 'a {
            (0..matrix.width()).map(move |c| matrix.get(index, c).copied().unwrap_or(EF::ZERO))
        }
    }
}
