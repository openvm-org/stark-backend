use itertools::Itertools;
use openvm_metal_common::d_buffer::MetalBuffer;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    prover::MatrixDimensions,
};
use p3_field::PrimeCharacteristicRing;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    base::MetalMatrix,
    prelude::{Digest, F},
};

/// A Merkle tree backed by a `MetalMatrix` in unified memory.
///
/// This mirrors the CUDA `MerkleTreeGpu` but uses Metal's unified memory model.
/// Data is read directly from the `MetalBuffer` via pointer access rather than
/// being converted to `ColMajorMatrix`.
pub struct MerkleTreeMetal<F> {
    /// The RS codeword matrix that forms the leaves of the Merkle tree.
    /// Optionally cached depending on prover configuration:
    /// - Caching avoids recomputation of MLE eval-to-coeffs, batch_expand, and forward NTT.
    /// - Not caching reduces peak memory usage.
    pub(crate) backing_matrix: Option<MetalMatrix<F>>,
    /// Merkle tree digest layers, from bottom (query digest layer) to top (root).
    pub(crate) digest_layers: Vec<Vec<Digest>>,
    /// Number of rows grouped per Merkle query.
    pub(crate) rows_per_query: usize,
}

/// Read a single row from a column-major MetalBuffer.
///
/// In a column-major layout with dimensions `height x width`, element at
/// row `r`, column `c` is stored at offset `c * height + r`.
fn row_from_buffer(buffer: &MetalBuffer<F>, height: usize, width: usize, row_idx: usize) -> Vec<F> {
    let data = unsafe { std::slice::from_raw_parts(buffer.as_ptr(), buffer.len()) };
    (0..width).map(|c| data[c * height + row_idx]).collect()
}

impl MerkleTreeMetal<F> {
    /// Construct a new Merkle tree from a column-major `MetalMatrix`.
    ///
    /// The matrix data is read directly from unified memory via unsafe pointer access.
    /// Row hashes are computed using `hasher.hash_slice()`, then compressed per query group,
    /// and finally built into a binary Merkle tree using `hasher.compress()`.
    ///
    /// If `cache_backing_matrix` is false, the backing matrix is dropped after tree
    /// construction to save memory.
    #[instrument(name = "merkle_tree_metal", skip_all)]
    pub fn new<H: MerkleHasher<F = F, Digest = Digest>>(
        hasher: &H,
        backing_matrix: MetalMatrix<F>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
    ) -> Self {
        let height = backing_matrix.height();
        let width = backing_matrix.width();
        assert!(height > 0, "Matrix height must be positive");
        assert!(rows_per_query.is_power_of_two(), "rows_per_query must be a power of two");

        let num_leaves = height.next_power_of_two();
        assert!(
            rows_per_query <= num_leaves,
            "rows_per_query ({rows_per_query}) must not exceed the number of Merkle leaves ({num_leaves})"
        );

        // Hash each row of the matrix.
        // For rows beyond the actual height (padding to next power of two), hash an all-zeros row.
        let buffer = backing_matrix.buffer();
        let row_hashes: Vec<Digest> = (0..num_leaves)
            .map(|r| {
                if r < height {
                    let row = row_from_buffer(buffer, height, width, r);
                    hasher.hash_slice(&row)
                } else {
                    // Padding row: hash zeros
                    let zeros = vec![F::ZERO; width];
                    hasher.hash_slice(&zeros)
                }
            })
            .collect();

        // Compress row hashes per query group.
        // For the first log2(rows_per_query) layers, we compress in `query_stride` pairs.
        // These intermediate layers are not stored in digest_layers.
        let query_stride = num_leaves / rows_per_query;
        let mut query_digest_layer = row_hashes;
        for _ in 0..log2_strict_usize(rows_per_query) {
            let prev_layer = query_digest_layer;
            query_digest_layer = (0..prev_layer.len() / 2)
                .map(|i| {
                    let x = i / query_stride;
                    let y = i % query_stride;
                    let left = prev_layer[2 * x * query_stride + y];
                    let right = prev_layer[(2 * x + 1) * query_stride + y];
                    hasher.compress(left, right)
                })
                .collect();
        }

        // Build the remaining Merkle tree layers by adjacent compression.
        let mut digest_layers = vec![query_digest_layer];
        while digest_layers.last().unwrap().len() > 1 {
            let prev_layer = digest_layers.last().unwrap();
            let layer: Vec<Digest> = prev_layer
                .chunks_exact(2)
                .map(|pair| hasher.compress(pair[0], pair[1]))
                .collect();
            digest_layers.push(layer);
        }

        let backing = if cache_backing_matrix {
            Some(backing_matrix)
        } else {
            None
        };

        Self {
            backing_matrix: backing,
            digest_layers,
            rows_per_query,
        }
    }

    /// Returns the Merkle root digest.
    pub fn root(&self) -> Digest {
        self.digest_layers.last().unwrap()[0]
    }

    /// Returns the query stride: the number of distinct query indices.
    /// Equal to `num_leaves / rows_per_query`.
    pub fn query_stride(&self) -> usize {
        self.digest_layers[0].len()
    }

    /// Returns the depth of the Merkle proof (number of sibling digests).
    pub fn proof_depth(&self) -> usize {
        self.digest_layers.len() - 1
    }

    /// Returns the Merkle proof (sibling path) for a given query index.
    ///
    /// The proof consists of sibling digests from the bottom layer up to (but not
    /// including) the root layer.
    pub fn query_merkle_proof(&self, query_idx: usize) -> Vec<Digest> {
        let stride = self.query_stride();
        assert!(
            query_idx < stride,
            "query_idx {query_idx} out of bounds for query_stride {stride}"
        );

        let mut idx = query_idx;
        let mut proof = Vec::with_capacity(self.proof_depth());
        for layer in self.digest_layers.iter().take(self.proof_depth()) {
            let sibling = layer[idx ^ 1];
            proof.push(sibling);
            idx >>= 1;
        }
        proof
    }

    /// Returns the opened rows for a given query index.
    ///
    /// The rows are `{ query_idx + t * query_stride() }` for `t` in `0..rows_per_query`.
    /// Each row is read directly from the MetalBuffer in column-major order.
    ///
    /// Panics if the backing matrix has been dropped (i.e., not cached).
    pub fn get_opened_rows(&self, query_idx: usize) -> Vec<Vec<F>> {
        let backing = self
            .backing_matrix
            .as_ref()
            .expect("backing_matrix was not cached; cannot open rows");
        let query_stride = self.query_stride();
        assert!(
            query_idx < query_stride,
            "query_idx {query_idx} out of bounds for query_stride {query_stride}"
        );

        let height = backing.height();
        let width = backing.width();
        let buffer = backing.buffer();

        let mut opened = Vec::with_capacity(self.rows_per_query);
        for row_offset in 0..self.rows_per_query {
            let row_idx = row_offset * query_stride + query_idx;
            let row = row_from_buffer(buffer, height, width, row_idx);
            debug_assert_eq!(
                row.len(),
                width,
                "row width mismatch: expected {width}, got {}",
                row.len()
            );
            opened.push(row);
        }
        opened
    }

    /// Batch queries multiple trees at the same query indices for Merkle proofs.
    ///
    /// Returns `Vec<Vec<Vec<Digest>>>` indexed as `[tree_idx][query_idx][proof_layer]`.
    ///
    /// # Assumptions
    /// - All trees have the same proof depth.
    pub fn batch_query_merkle_proofs(
        trees: &[&Self],
        query_indices: &[usize],
    ) -> Vec<Vec<Vec<Digest>>> {
        let num_trees = trees.len();
        if num_trees == 0 {
            return Vec::new();
        }
        let depth = trees[0].proof_depth();
        debug_assert!(
            trees.iter().all(|tree| tree.proof_depth() == depth),
            "Merkle trees don't have the same depth"
        );

        (0..num_trees)
            .map(|tree_idx| {
                query_indices
                    .iter()
                    .map(|&idx| trees[tree_idx].query_merkle_proof(idx))
                    .collect_vec()
            })
            .collect_vec()
    }

    /// Batch open rows from multiple backing matrices at the same query indices.
    ///
    /// Returns `Vec<Vec<Vec<F>>>` indexed as `[matrix_idx][query_idx][row_data]`
    /// where `row_data` is the concatenated data for `rows_per_query` strided rows.
    ///
    /// This reads directly from the MetalBuffer in column-major order.
    pub fn batch_open_rows(
        backing_matrices: &[&MetalMatrix<F>],
        query_indices: &[usize],
        query_stride: usize,
        rows_per_query: usize,
    ) -> Vec<Vec<Vec<F>>> {
        backing_matrices
            .iter()
            .map(|matrix| {
                let height = matrix.height();
                let width = matrix.width();
                let buffer = matrix.buffer();

                query_indices
                    .iter()
                    .map(|&query_idx| {
                        debug_assert!(query_idx < query_stride);
                        let mut rows_data = Vec::with_capacity(rows_per_query * width);
                        for row_offset in 0..rows_per_query {
                            let row_idx = row_offset * query_stride + query_idx;
                            let row = row_from_buffer(buffer, height, width, row_idx);
                            rows_data.extend(row);
                        }
                        rows_data
                    })
                    .collect_vec()
            })
            .collect_vec()
    }
}
