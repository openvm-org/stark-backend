//! [CpuStackedPcsData] — CPU-backend PCS data with RowMajor Merkle tree backing.
//!
//! The eval matrix stays `ColMajor` because stacked reduction accesses it
//! column-by-column (each unstacked trace column maps to a contiguous slice
//! in a stacked column).
//!
//! The Merkle tree uses `RowMajor` backing for cache-friendly row hashing
//! and contiguous row access in query answering.

use openvm_stark_backend::prover::{
    error::StackedPcsError, stacked_pcs::StackedLayout, ColMajorMatrix,
};

use crate::merkle::CpuMerkleTree;

/// CPU-backend PCS data with RowMajor Merkle tree backing.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct CpuStackedPcsData<F, Digest> {
    /// Layout of the unstacked collection of matrices within the stacked matrix.
    pub layout: StackedLayout,
    /// Stacked evaluation matrix (ColMajor for column-oriented access in stacked reduction).
    pub matrix: ColMajorMatrix<F>,
    /// Merkle tree with RowMajor RS codeword backing.
    pub tree: CpuMerkleTree<F, Digest>,
}

impl<F, Digest: Clone> CpuStackedPcsData<F, Digest> {
    pub fn new(
        layout: StackedLayout,
        matrix: ColMajorMatrix<F>,
        tree: CpuMerkleTree<F, Digest>,
    ) -> Self {
        Self {
            layout,
            matrix,
            tree,
        }
    }

    /// Returns the root of the Merkle tree.
    pub fn commit(&self) -> Result<Digest, StackedPcsError> {
        self.tree.root()
    }
}

#[cfg(test)]
mod tests {
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    /// Minimal smoke test: construct a CpuStackedPcsData and call `commit`.
    #[test]
    fn test_cpu_stacked_pcs_data_commit() {
        // Minimal StackedLayout (1 column, height 2)
        let layout = StackedLayout::new(0, 1, vec![(1, 1)]).unwrap();

        // ColMajor eval matrix: 1 column x 2 rows
        let matrix = ColMajorMatrix::new(vec![1u32, 2], 1);

        // Minimal CpuMerkleTree
        let rm = RowMajorMatrix::new(vec![10u32, 20], 1);
        let digest_layers = vec![vec![100u32], vec![42]];
        let tree = unsafe { CpuMerkleTree::from_raw_parts(rm, digest_layers, 2) };

        let data = CpuStackedPcsData::new(layout, matrix, tree);
        assert_eq!(data.commit().unwrap(), 42);
    }
}
