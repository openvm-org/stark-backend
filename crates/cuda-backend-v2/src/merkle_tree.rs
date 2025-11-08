use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{copy::MemCopyD2H, d_buffer::DeviceBuffer, error::CudaError};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    Digest, F,
    cuda::merkle_tree::{
        poseidon2_adjacent_compress_layer, poseidon2_row_hashes, poseidon2_strided_compress_layer,
    },
};

pub struct MerkleTreeGpu<F, Digest> {
    /// The matrix that is used to form the leaves of the Merkle tree, which are
    /// in turn hashed into the bottom digest layer.
    pub(crate) backing_matrix: DeviceMatrix<F>,
    pub(crate) digest_layers: Vec<DeviceBuffer<Digest>>,
    pub(crate) rows_per_query: usize,
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
    pub fn new(matrix: DeviceMatrix<F>, rows_per_query: usize) -> Result<Self, CudaError> {
        let height = matrix.height();
        assert!(height.is_power_of_two());
        let k = log2_strict_usize(rows_per_query);
        assert!(
            rows_per_query <= height,
            "rows_per_query ({rows_per_query}) must not exceed height ({height})"
        );
        let mut row_hashes = DeviceBuffer::<Digest>::with_capacity(height);
        // SAFETY: row_hashes properly allocated
        unsafe {
            poseidon2_row_hashes(&mut row_hashes, &matrix.buffer(), matrix.width(), height)?;
        }

        let query_stride = height / rows_per_query;
        let mut query_digest_layer = row_hashes;
        // For the first log2(rows_per_query) layers, we hash in `query_stride` pairs and don't
        // need to store the digest layers
        for _i in 0..k {
            // PERF(memory): The memory manager doesn't allow easy resizing of buffers, so we simply
            // create a new buffer and drop the old one per layer. The memory manager should handle
            // this and effectively re-use the dropped buffer.
            let mut next_layer =
                DeviceBuffer::<Digest>::with_capacity(query_digest_layer.len() / 2);
            let next_layer_len = next_layer.len();
            // SAFETY:
            // - `next_layer` is properly allocated with half the size of `query_digest_layer` and
            //   does not overlap with it.
            // - `1 <= query_stride = 2^{-k} * height < 2^{-i} * height = query_digest_layer.len()`.
            unsafe {
                poseidon2_strided_compress_layer(
                    &mut next_layer,
                    &query_digest_layer,
                    next_layer_len,
                    query_stride,
                )?;
            }
            query_digest_layer = next_layer;
        }

        let mut digest_layers = vec![query_digest_layer];
        while digest_layers.last().unwrap().len() > 1 {
            let prev_layer = digest_layers.last().unwrap();
            let mut layer = DeviceBuffer::<Digest>::with_capacity(prev_layer.len() / 2);
            let layer_len = layer.len();
            // SAFETY:
            // - `layer` is properly allocated with half the size of `prev_layer` and does not
            //   overlap with it.
            unsafe {
                poseidon2_adjacent_compress_layer(&mut layer, prev_layer, layer_len)?;
            }
            digest_layers.push(layer);
        }

        Ok(Self {
            backing_matrix: matrix,
            digest_layers,
            rows_per_query,
        })
    }
}
