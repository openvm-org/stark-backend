use std::array::from_fn;

use itertools::Itertools;
use openvm_metal_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::MetalBuffer,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    base::MetalMatrix,
    metal::{
        matrix::matrix_get_rows_fp_kernel,
        merkle_tree::{
            poseidon2_adjacent_compress_layer, poseidon2_compressing_row_hashes,
            poseidon2_compressing_row_hashes_ext, query_digest_layers,
        },
    },
    prelude::{Digest, DIGEST_SIZE, EF, F},
    MerkleTreeError,
};

#[inline]
fn read_single_from_shared<T>(buf: &MetalBuffer<T>) -> T {
    debug_assert_eq!(buf.len(), 1, "Expected single-element buffer");
    unsafe { buf.as_ptr().read() }
}

#[inline]
fn build_non_root_layer_ptrs<Digest>(digest_layers: &[MetalBuffer<Digest>]) -> MetalBuffer<u64> {
    digest_layers
        .iter()
        .take(digest_layers.len().saturating_sub(1))
        .map(|layer| layer.as_device_ptr() as u64)
        .collect_vec()
        .to_device()
}

fn flatten_cached_layer_ptrs<F, Digest>(
    trees: &[&MerkleTreeMetal<F, Digest>],
    depth: usize,
) -> MetalBuffer<u64> {
    let out = MetalBuffer::<u64>::with_capacity(trees.len() * depth);
    let mut out_offset = 0usize;
    for tree in trees {
        debug_assert_eq!(tree.non_root_layer_ptrs.len(), depth);
        unsafe {
            std::ptr::copy_nonoverlapping(
                tree.non_root_layer_ptrs.as_ptr(),
                out.as_mut_ptr().add(out_offset),
                depth,
            );
        }
        out_offset += depth;
    }
    out
}

pub struct MerkleTreeMetal<F, Digest> {
    /// The matrix that is used to form the leaves of the Merkle tree.
    /// Optionally cached depending on the prover configuration.
    pub(crate) backing_matrix: Option<MetalMatrix<F>>,
    pub(crate) digest_layers: Vec<MetalBuffer<Digest>>,
    pub(crate) non_root_layer_ptrs: MetalBuffer<u64>,
    pub(crate) rows_per_query: usize,
    pub(crate) root: Digest,
}

impl<F, Digest> MerkleTreeMetal<F, Digest> {
    pub fn root(&self) -> Digest
    where
        Digest: Clone,
    {
        self.root.clone()
    }

    pub fn query_stride(&self) -> usize {
        self.digest_layers[0].len()
    }

    pub fn proof_depth(&self) -> usize {
        self.digest_layers.len() - 1
    }
}

// Base field merkle tree
impl MerkleTreeMetal<F, Digest> {
    #[instrument(name = "merkle_tree", skip_all)]
    pub fn new(
        matrix: MetalMatrix<F>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
    ) -> Result<Self, MerkleTreeError> {
        let height = matrix.height();
        assert!(height.is_power_of_two());
        let k = log2_strict_usize(rows_per_query);
        assert!(
            rows_per_query <= height,
            "rows_per_query ({rows_per_query}) must not exceed height ({height})"
        );
        let query_stride = height / rows_per_query;
        let mut query_digest_layer = MetalBuffer::<Digest>::with_capacity(query_stride);
        unsafe {
            poseidon2_compressing_row_hashes(
                &mut query_digest_layer,
                matrix.buffer(),
                matrix.width(),
                query_stride,
                k,
            )
            .map_err(MerkleTreeError::CompressingRowHashes)?;
        }
        // If not caching, drop the backing matrix at this point to save memory.
        let backing_matrix = cache_backing_matrix.then_some(matrix);

        let mut digest_layers = vec![query_digest_layer];
        while digest_layers.last().unwrap().len() > 1 {
            let prev_layer = digest_layers.last().unwrap();
            let mut layer = MetalBuffer::<Digest>::with_capacity(prev_layer.len() / 2);
            let layer_len = layer.len();
            let layer_idx = digest_layers.len();
            unsafe {
                poseidon2_adjacent_compress_layer(&mut layer, prev_layer, layer_len).map_err(
                    |error| MerkleTreeError::AdjacentCompressLayer {
                        error,
                        layer: layer_idx,
                    },
                )?;
            }
            digest_layers.push(layer);
        }
        let d_root = digest_layers.last().unwrap();
        assert_eq!(d_root.len(), 1, "Only one root is supported");
        let root = read_single_from_shared(d_root);
        let non_root_layer_ptrs = build_non_root_layer_ptrs(&digest_layers);

        Ok(Self {
            backing_matrix,
            digest_layers,
            non_root_layer_ptrs,
            rows_per_query,
            root,
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
        MerkleTreeError,
    > {
        let num_trees = trees.len();
        let num_queries = query_indices.len();
        let depth = trees[0].proof_depth();
        debug_assert!(
            trees.iter().all(|tree| tree.proof_depth() == depth),
            "Merkle trees don't have same depth"
        );
        let combined_layers_ptrs =
            (num_trees > 1).then(|| flatten_cached_layer_ptrs(trees, depth));
        let d_layers_ptr = combined_layers_ptrs
            .as_ref()
            .unwrap_or(&trees[0].non_root_layer_ptrs);
        debug_assert_eq!(d_layers_ptr.len(), num_trees * depth);

        let layers = trees
            .iter()
            .flat_map(|tree| {
                // skip root layer [depth]
                tree.digest_layers.iter().take(depth)
            })
            .collect_vec();

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
        let d_indices = indices.to_device();
        debug_assert_eq!(d_indices.len(), d_layers_ptr.len() * num_queries);

        let mut d_out =
            MetalBuffer::<F>::with_capacity(d_layers_ptr.len() * num_queries * DIGEST_SIZE);
        unsafe {
            query_digest_layers(
                &mut d_out,
                &d_layers_ptr,
                &layers,
                &d_indices,
                num_queries as u64,
                d_layers_ptr.len() as u64,
            )
            .map_err(MerkleTreeError::QueryDigestLayers)?;
        }
        let out = d_out.to_host();
        // Chunk up the array into expected Vec groupings
        let res = (0..num_trees)
            .map(|tree_idx| {
                (0..num_queries)
                    .map(|query_idx| {
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
        #[cfg(debug_assertions)]
        {
            for (tree_idx, tree) in trees.iter().enumerate() {
                let host_layers = tree
                    .digest_layers
                    .iter()
                    .take(depth)
                    .map(MetalBuffer::to_host)
                    .collect_vec();
                for (query_idx, &index) in query_indices.iter().enumerate() {
                    for layer_idx in 0..depth {
                        let sibling_idx = (index >> layer_idx) ^ 1;
                        debug_assert!(
                            res[tree_idx][query_idx][layer_idx]
                                == host_layers[layer_idx][sibling_idx]
                        );
                    }
                }
            }
        }
        Ok(res)
    }

    pub fn batch_open_rows(
        backing_matrices: &[&MetalMatrix<F>],
        query_indices: &[usize],
        query_stride: usize,
        rows_per_query: usize,
    ) -> Result<
        Vec<
            // per tree
            Vec<
                // per query index
                Vec<F>, // opened rows
            >,
        >,
        MerkleTreeError,
    > {
        let row_idxs = query_indices
            .iter()
            .flat_map(|&query_idx| {
                debug_assert!(query_idx < query_stride);
                (0..rows_per_query)
                    .map(move |row_offset| (row_offset * query_stride + query_idx) as u32)
            })
            .collect_vec();
        let d_row_idxs = row_idxs.to_device();
        backing_matrices
            .iter()
            .enumerate()
            .map(|(matrix_idx, matrix)| {
                let d_out = MetalBuffer::<F>::with_capacity(row_idxs.len() * matrix.width());
                unsafe {
                    matrix_get_rows_fp_kernel(
                        &d_out,
                        matrix.buffer(),
                        &d_row_idxs,
                        matrix.width() as u64,
                        matrix.height() as u64,
                        d_row_idxs.len() as u32,
                    )
                    .map_err(|error| MerkleTreeError::MatrixGetRows { error, matrix_idx })?;
                }
                let width = matrix.width();
                let out = d_out.to_host();
                let opened_rows_per_query = out
                    .chunks_exact(rows_per_query * width)
                    .map(|rows| rows.to_vec())
                    .collect_vec();
                #[cfg(debug_assertions)]
                {
                    let matrix_vals = matrix.buffer().to_host();
                    for (query_pos, &query_idx) in query_indices.iter().enumerate() {
                        let got_rows = &opened_rows_per_query[query_pos];
                        for row_offset in 0..rows_per_query {
                            let row_idx = row_offset * query_stride + query_idx;
                            let got = &got_rows[row_offset * width..(row_offset + 1) * width];
                            for col in 0..width {
                                let expected = matrix_vals[col * matrix.height() + row_idx];
                                debug_assert_eq!(got[col], expected);
                            }
                        }
                    }
                }
                Ok(opened_rows_per_query)
            })
            .collect::<Result<Vec<_>, MerkleTreeError>>()
    }
}

// Extension field merkle tree
impl MerkleTreeMetal<EF, Digest> {
    #[instrument(name = "merkle_tree_ext", skip_all)]
    pub fn new(
        matrix: MetalMatrix<EF>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
    ) -> Result<Self, MerkleTreeError> {
        let height = matrix.height();
        assert!(height.is_power_of_two());
        let k = log2_strict_usize(rows_per_query);
        assert!(
            rows_per_query <= height,
            "rows_per_query ({rows_per_query}) must not exceed height ({height})"
        );
        let query_stride = height / rows_per_query;
        let mut query_digest_layer = MetalBuffer::<Digest>::with_capacity(query_stride);
        unsafe {
            poseidon2_compressing_row_hashes_ext(
                &mut query_digest_layer,
                matrix.buffer(),
                matrix.width(),
                query_stride,
                k,
            )
            .map_err(MerkleTreeError::CompressingRowHashesExt)?;
        }
        // If not caching, drop the backing matrix at this point to save memory.
        let backing_matrix = cache_backing_matrix.then_some(matrix);

        let mut digest_layers = vec![query_digest_layer];
        while digest_layers.last().unwrap().len() > 1 {
            let prev_layer = digest_layers.last().unwrap();
            let mut layer = MetalBuffer::<Digest>::with_capacity(prev_layer.len() / 2);
            let layer_len = layer.len();
            let layer_idx = digest_layers.len();
            unsafe {
                poseidon2_adjacent_compress_layer(&mut layer, prev_layer, layer_len).map_err(
                    |error| MerkleTreeError::AdjacentCompressLayer {
                        error,
                        layer: layer_idx,
                    },
                )?;
            }
            digest_layers.push(layer);
        }
        let d_root = digest_layers.last().unwrap();
        assert_eq!(d_root.len(), 1, "Only one root is supported");
        let root = read_single_from_shared(d_root);
        let non_root_layer_ptrs = build_non_root_layer_ptrs(&digest_layers);

        Ok(Self {
            backing_matrix,
            digest_layers,
            non_root_layer_ptrs,
            rows_per_query,
            root,
        })
    }
}
