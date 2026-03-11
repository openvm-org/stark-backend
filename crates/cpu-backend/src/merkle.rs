//! [CpuMerkleTree] — a Merkle tree backed by a [RowMajorMatrix].
//!
//! This mirrors the interface of [`openvm_stark_backend::prover::stacked_pcs::MerkleTree`]
//! but stores its codeword matrix in row-major layout for cache-friendly row hashing
//! and contiguous row access in query answering.

use openvm_stark_backend::prover::error::StackedPcsError;
use p3_matrix::dense::RowMajorMatrix;

/// Merkle tree with row-major codeword backing.
///
/// Each leaf corresponds to a row of `backing_matrix` (or a batch of rows when
/// `rows_per_query > 1`).  The `digest_layers` are built exactly as in
/// [`MerkleTree`](openvm_stark_backend::prover::stacked_pcs::MerkleTree), so
/// proof construction and verification are identical.
#[derive(Clone, Debug)]
pub struct CpuMerkleTree<F, Digest> {
    pub(crate) backing_matrix: RowMajorMatrix<F>,
    pub(crate) digest_layers: Vec<Vec<Digest>>,
    pub(crate) rows_per_query: usize,
}

impl<F, Digest> CpuMerkleTree<F, Digest> {
    /// Construct a `CpuMerkleTree` from pre-computed parts without validation.
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
        backing_matrix: RowMajorMatrix<F>,
        digest_layers: Vec<Vec<Digest>>,
        rows_per_query: usize,
    ) -> Self {
        Self {
            backing_matrix,
            digest_layers,
            rows_per_query,
        }
    }

    /// Returns a reference to the row-major codeword matrix.
    pub fn backing_matrix(&self) -> &RowMajorMatrix<F> {
        &self.backing_matrix
    }

    /// Returns a reference to the digest layers.
    pub fn digest_layers(&self) -> &Vec<Vec<Digest>> {
        &self.digest_layers
    }

    /// Number of rows batched into each Merkle leaf.
    pub fn rows_per_query(&self) -> usize {
        self.rows_per_query
    }

    /// Number of distinct query indices (= number of entries in the bottom digest layer).
    pub fn query_stride(&self) -> usize {
        self.digest_layers[0].len()
    }

    /// Depth of a Merkle proof (number of sibling digests).
    pub fn proof_depth(&self) -> usize {
        self.digest_layers.len() - 1
    }
}

impl<F, Digest: Clone> CpuMerkleTree<F, Digest> {
    /// Returns the Merkle root (the single element of the topmost digest layer).
    pub fn root(&self) -> Result<Digest, StackedPcsError> {
        Ok(self
            .digest_layers
            .last()
            .ok_or(StackedPcsError::MerkleTreeNoRoot)?[0]
            .clone())
    }

    /// Returns the Merkle authentication path for the given `query_idx`.
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

impl<F: Copy, Digest> CpuMerkleTree<F, Digest> {
    /// Returns the opened rows for the given query index.
    ///
    /// The rows are `{ index + t * query_stride() }` for `t` in `0..rows_per_query`.
    ///
    /// Because the backing matrix is row-major, each row is a contiguous slice —
    /// no scatter-gather is needed (unlike the column-major variant).
    pub fn get_opened_rows(&self, index: usize) -> Result<Vec<Vec<F>>, StackedPcsError> {
        let query_stride = self.query_stride();
        if index >= query_stride {
            return Err(StackedPcsError::MerkleTreeOpenedRowsOutOfBounds {
                index,
                query_stride,
            });
        }

        let width = self.backing_matrix.width;
        let height = self.backing_matrix.values.len() / width;
        let mut rows = Vec::with_capacity(self.rows_per_query);
        for t in 0..self.rows_per_query {
            let row_idx = t * query_stride + index;
            if row_idx < height {
                let start = row_idx * width;
                rows.push(self.backing_matrix.values[start..start + width].to_vec());
            } else {
                // Padding row for matrices whose height is not a power of two.
                rows.push(vec![]);
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_raw_parts_and_accessors() {
        let mat = RowMajorMatrix::new(vec![1u32, 2, 3, 4, 5, 6], 3);
        let digest_layers = vec![vec![10u32, 20], vec![30]];

        let tree = unsafe { CpuMerkleTree::from_raw_parts(mat, digest_layers, 1) };

        assert_eq!(tree.rows_per_query(), 1);
        assert_eq!(tree.backing_matrix().width, 3);
        assert_eq!(tree.digest_layers().len(), 2);
        assert_eq!(tree.query_stride(), 2);
        assert_eq!(tree.proof_depth(), 1);
    }

    #[test]
    fn test_root() {
        let mat = RowMajorMatrix::new(vec![1u32, 2, 3, 4], 2);
        let digest_layers = vec![vec![10u32, 20], vec![42]];

        let tree = unsafe { CpuMerkleTree::from_raw_parts(mat, digest_layers, 1) };
        assert_eq!(tree.root().unwrap(), 42);
    }

    #[test]
    fn test_root_no_layers() {
        let mat = RowMajorMatrix::new(vec![1u32, 2], 2);
        let digest_layers: Vec<Vec<u32>> = vec![];

        let tree = unsafe { CpuMerkleTree::from_raw_parts(mat, digest_layers, 1) };
        assert!(tree.root().is_err());
    }

    #[test]
    fn test_query_merkle_proof() {
        // 4 leaves -> 2 layers: [a, b, c, d] -> [ab, cd] -> [abcd]
        let mat = RowMajorMatrix::new(vec![0u32; 8], 2);
        let layer0 = vec![10u32, 20, 30, 40]; // 4 entries
        let layer1 = vec![100u32, 200]; // 2 entries
        let layer2 = vec![999u32]; // root
        let digest_layers = vec![layer0, layer1, layer2];

        let tree = unsafe { CpuMerkleTree::from_raw_parts(mat, digest_layers, 1) };
        assert_eq!(tree.query_stride(), 4);
        assert_eq!(tree.proof_depth(), 2);

        // Query index 0: siblings are layer0[1], layer1[1]
        let proof = tree.query_merkle_proof(0).unwrap();
        assert_eq!(proof, vec![20, 200]);

        // Query index 1: siblings are layer0[0], layer1[1]
        let proof = tree.query_merkle_proof(1).unwrap();
        assert_eq!(proof, vec![10, 200]);

        // Query index 2: siblings are layer0[3], layer1[0]
        let proof = tree.query_merkle_proof(2).unwrap();
        assert_eq!(proof, vec![40, 100]);

        // Out of bounds
        assert!(tree.query_merkle_proof(4).is_err());
    }

    #[test]
    fn test_get_opened_rows_single() {
        // 4 rows x 3 cols, rows_per_query = 1
        let mat = RowMajorMatrix::new(
            vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            3,
        );
        let digest_layers = vec![vec![0u32; 4], vec![0u32; 2], vec![0u32; 1]];

        let tree = unsafe { CpuMerkleTree::from_raw_parts(mat, digest_layers, 1) };
        assert_eq!(tree.query_stride(), 4);

        let rows = tree.get_opened_rows(0).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec![1, 2, 3]);

        let rows = tree.get_opened_rows(2).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec![7, 8, 9]);
    }

    #[test]
    fn test_get_opened_rows_batched() {
        // 4 rows x 2 cols, rows_per_query = 2
        // query_stride = 4 / 2 = 2 (from digest_layers[0].len())
        let mat = RowMajorMatrix::new(vec![1u32, 2, 3, 4, 5, 6, 7, 8], 2);
        let digest_layers = vec![vec![0u32; 2], vec![0u32; 1]];

        let tree = unsafe { CpuMerkleTree::from_raw_parts(mat, digest_layers, 2) };
        assert_eq!(tree.query_stride(), 2);

        // index=0: rows at 0 and 0 + 2 = 2
        let rows = tree.get_opened_rows(0).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![1, 2]);
        assert_eq!(rows[1], vec![5, 6]);

        // index=1: rows at 1 and 1 + 2 = 3
        let rows = tree.get_opened_rows(1).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![3, 4]);
        assert_eq!(rows[1], vec![7, 8]);

        // Out of bounds
        assert!(tree.get_opened_rows(2).is_err());
    }
}
