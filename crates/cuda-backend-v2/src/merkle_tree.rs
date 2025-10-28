use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use stark_backend_v2::prover::stacked_pcs::MerkleTree;
use tracing::instrument;

use crate::{Digest, F, gpu_backend::transport_matrix_d2h_col_major};

pub struct MerkleTreeGpu<F, Digest> {
    /// The matrix that is used to form the leaves of the Merkle tree, which are
    /// in turn hashed into the bottom digest layer.
    pub(crate) backing_matrix: DeviceMatrix<F>,
    pub(crate) digest_layers: Vec<DeviceBuffer<Digest>>,
    pub(crate) rows_per_leaf: usize,
}

impl<F, Digest> MerkleTreeGpu<F, Digest> {
    pub fn root(&self) -> Digest {
        let root = self.digest_layers.last().unwrap();
        assert_eq!(root.len(), 1, "Only one root is supported");
        root.to_host().unwrap().pop().unwrap()
    }
}

impl MerkleTreeGpu<F, Digest> {
    #[instrument(name = "merkle_tree", skip_all)]
    pub fn new(matrix: DeviceMatrix<F>, rows_per_leaf: usize) -> Self {
        // TODO[CUDA]: add kernel
        let tree = MerkleTree::<F, Digest>::new(
            transport_matrix_d2h_col_major(&matrix).unwrap(),
            rows_per_leaf,
        );
        let digest_layers = tree
            .digest_layers()
            .iter()
            .map(|layer| layer.to_device().unwrap())
            .collect();
        Self {
            backing_matrix: matrix,
            digest_layers,
            rows_per_leaf,
        }
    }
}
