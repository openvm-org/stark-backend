use std::array::from_fn;

use itertools::Itertools;
use openvm_cuda_backend::{
    base::DeviceMatrix,
    cuda::kernels::{matrix::matrix_get_rows_fp_kernel, poseidon2::query_digest_layers_kernel},
};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::CudaError,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    DIGEST_SIZE, Digest, EF, F, ProverError,
    cuda::merkle_tree::{
        poseidon2_adjacent_compress_layer, poseidon2_row_hashes, poseidon2_row_hashes_ext,
        poseidon2_strided_compress_layer,
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

    pub fn query_stride(&self) -> usize {
        self.digest_layers[0].len()
    }

    pub fn proof_depth(&self) -> usize {
        self.digest_layers.len() - 1
    }
}

// Base field merkle tree
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
            poseidon2_row_hashes(&mut row_hashes, matrix.buffer(), matrix.width(), height)?;
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

    /// Batch queries multiple `trees` at _the same_ `query_indices` for merkle proofs.
    ///
    /// # Assumptions
    /// - All `trees` have the same depth.
    pub fn batch_query_merkle_proofs(
        trees: &[&Self],
        query_indices: &[usize],
    ) -> Result<
        Vec<
            // per tree
            Vec<
                // per query index
                Vec<Digest>, // merkle proof
            >,
        >,
        ProverError,
    > {
        // the way the kernel works is that it just treats each layer as a separate array and does
        // parallel accesses, so we just lay out all the layer pointers flattened into a vec
        let num_trees = trees.len();
        let num_queries = query_indices.len();
        let depth = trees[0].proof_depth();
        debug_assert!(
            trees.iter().all(|tree| tree.proof_depth() == depth),
            "Merkle trees don't have same depth"
        );
        let layers_ptr = trees
            .iter()
            .flat_map(|tree| {
                // skip root layer [depth]
                tree.digest_layers
                    .iter()
                    .take(depth)
                    .map(|layer| layer.as_ptr() as u64)
            })
            .collect_vec();
        let d_layers_ptr = layers_ptr.to_device()?;
        debug_assert_eq!(d_layers_ptr.len(), num_trees * depth);

        // [query_idx] is the top level grouping
        let indices = query_indices
            .iter()
            .flat_map(|&index| {
                (0..num_trees).flat_map(move |tree_idx| {
                    (0..depth).map(move |layer_idx| {
                        debug_assert!(index < trees[tree_idx].query_stride());
                        ((index >> layer_idx) ^ 1) as u64
                    })
                })
            })
            .collect_vec();
        let d_indices = indices.to_device()?;
        debug_assert_eq!(d_indices.len(), d_layers_ptr.len() * num_queries);

        let d_out =
            DeviceBuffer::<F>::with_capacity(d_layers_ptr.len() * num_queries * DIGEST_SIZE);
        // SAFETY:
        // - d_out has size num_trees * depth * num_queries * DIGEST_SIZE in `F` elements
        // - d_layers_ptr is size `num_trees * depth` device pointers
        // - d_indices is size `num_trees * depth * num_queries` indices for merkle proof sibling
        //   indices
        unsafe {
            query_digest_layers_kernel::<F>(
                &d_out,
                &d_layers_ptr,
                &d_indices,
                num_queries as u64,
                d_layers_ptr.len() as u64,
            )?;
        }
        let out = d_out.to_host()?;
        // Chunk up the array into expected Vec groupings
        let res = (0..num_trees)
            .map(|tree_idx| {
                (0..num_queries)
                    .map(|query_idx| {
                        // merkle proof for a single query for tree `tree_idx`
                        (0..depth)
                            .map(|layer_idx| -> Digest {
                                from_fn(|elem_idx| {
                                    out[(query_idx * num_trees * depth
                                        + tree_idx * depth
                                        + layer_idx)
                                        * DIGEST_SIZE
                                        + elem_idx]
                                })
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();
        Ok(res)
    }

    pub fn batch_open_rows(
        trees: &[&Self],
        query_indices: &[usize],
    ) -> Result<
        Vec<
            // per tree
            Vec<
                // per query index
                Vec<F>, // opened rows, concatenated for rows_per_query strided rows
            >,
        >,
        ProverError,
    > {
        let query_stride = trees[0].query_stride();
        debug_assert!(
            trees.iter().all(|tree| tree.query_stride() == query_stride),
            "Merkle trees don't have same layer size"
        );
        let rows_per_query = trees[0].rows_per_query;
        debug_assert!(
            trees
                .iter()
                .all(|tree| tree.rows_per_query == rows_per_query),
            "Merkle trees don't have same rows_per_query"
        );
        let row_idxs = query_indices
            .iter()
            .flat_map(|&query_idx| {
                debug_assert!(query_idx < query_stride);
                (0..rows_per_query)
                    .map(move |row_offset| (row_offset * query_stride + query_idx) as u32)
            })
            .collect_vec();
        let d_row_idxs = row_idxs.to_device()?;
        // PERF[jpw]: I did not batch across trees into a single kernel call because widths are
        // different so it was inconvenient. Make a new kernel if slow.
        // NOTE: par_iter requires cross-stream waits, not worth the effort
        trees
            .iter()
            .map(|tree| {
                let d_out =
                    DeviceBuffer::<F>::with_capacity(row_idxs.len() * tree.backing_matrix.width());
                let matrix = &tree.backing_matrix;
                // SAFETY:
                // - `output_rows` is allocated with row_idxs.len() * width
                // - row indices are within bounds
                unsafe {
                    matrix_get_rows_fp_kernel(
                        &d_out,
                        matrix.buffer(),
                        &d_row_idxs,
                        matrix.width() as u64,
                        matrix.height() as u64,
                        d_row_idxs.len() as u32,
                    )?;
                }
                let width = tree.backing_matrix.width();
                let out = d_out.to_host()?;
                let opened_rows_per_query = out
                    .chunks_exact(rows_per_query * width)
                    .map(|rows| rows.to_vec())
                    .collect_vec();
                Ok(opened_rows_per_query)
            })
            .collect::<Result<Vec<_>, ProverError>>()
    }
}

// Extension field merkle tree
// NOTE: this is currently unused because we tranpose DeviceMatrix<EF> to DeviceMatrix<F> beforehand
// in our use cases
impl MerkleTreeGpu<EF, Digest> {
    #[instrument(name = "merkle_tree_ext", skip_all)]
    pub fn new(matrix: DeviceMatrix<EF>, rows_per_query: usize) -> Result<Self, CudaError> {
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
            poseidon2_row_hashes_ext(&mut row_hashes, matrix.buffer(), matrix.width(), height)?;
        }

        // === Below this line is same as for MerkleTreeGpu<F, Digest> ===
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
