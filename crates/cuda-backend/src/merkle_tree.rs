use std::array::from_fn;

use itertools::Itertools;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
    stream::DeviceContext,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

#[cfg(feature = "baby-bear-bn254-poseidon2")]
use crate::cuda::bn254_merkle_tree::Bn254Digest;
use crate::{
    base::DeviceMatrix,
    cuda::{matrix::matrix_get_rows_fp_kernel, merkle_tree::query_digest_layers},
    hash_scheme::{GpuMerkleHash, Poseidon2MerkleHash},
    prelude::{Digest, DIGEST_SIZE, EF, F},
    MerkleTreeError,
};

/// Trait for reconstructing a digest from a flat slice of F elements as
/// produced by the `query_digest_layers` CUDA kernel.
///
/// Both `Digest = [F; 8]` (BabyBear Poseidon2) and `Bn254Digest = [Bn254Scalar; 1]`
/// occupy exactly `DIGEST_SIZE * size_of::<F>()` = 32 bytes, so the same kernel
/// can be reused for both.
pub trait BatchQueryMerkle: Copy + Sized + 'static {
    /// Reconstruct one digest from `DIGEST_SIZE` consecutive F-valued words in `out`
    /// starting at index `base`.
    fn reconstruct_from_f(out: &[F], base: usize) -> Self;
}

const MAX_MERKLE_ROWS_PER_QUERY: usize = 1024;

fn validate_merkle_rows_per_query(
    rows_per_query: usize,
    height: usize,
) -> Result<usize, MerkleTreeError> {
    // The generic Merkle constructor still treats "power of two" and "fits within the matrix
    // height" as caller invariants. The CUDA-specific maximum rows-per-query is returned as a
    // recoverable error because it depends on backend support, not generic Merkle correctness.
    let k = log2_strict_usize(rows_per_query);
    assert!(
        rows_per_query <= height,
        "rows_per_query ({rows_per_query}) must not exceed height ({height})"
    );
    if rows_per_query > MAX_MERKLE_ROWS_PER_QUERY {
        return Err(MerkleTreeError::UnsupportedRowsPerQuery {
            rows_per_query,
            max_rows_per_query: MAX_MERKLE_ROWS_PER_QUERY,
        });
    }
    Ok(k)
}

impl BatchQueryMerkle for Digest {
    fn reconstruct_from_f(out: &[F], base: usize) -> Self {
        from_fn(|i| out[base + i])
    }
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl BatchQueryMerkle for Bn254Digest {
    fn reconstruct_from_f(out: &[F], base: usize) -> Self {
        // Safety: [F; DIGEST_SIZE] and Bn254Digest have the same size (32 bytes).
        const _: () =
            assert!(std::mem::size_of::<Bn254Digest>() == DIGEST_SIZE * std::mem::size_of::<F>());
        let f_arr: [F; DIGEST_SIZE] = from_fn(|i| out[base + i]);
        unsafe { std::ptr::read_unaligned(f_arr.as_ptr() as *const Bn254Digest) }
    }
}

pub struct MerkleTreeGpu<F, Digest> {
    /// The matrix that is used to form the leaves of the Merkle tree, which are
    /// in turn hashed into the bottom digest layer.
    ///
    /// This matrix is optionally cached depending on the prover configuration:
    /// - Caching increases the peak GPU memory but avoids a recomputation of MLE eval-to-coeffs,
    ///   batch_expand, and forward NTT.
    /// - Not caching pays a performance penalty due to the above recomputation.
    pub(crate) backing_matrix: Option<DeviceMatrix<F>>,
    pub(crate) digest_layers: Vec<DeviceBuffer<Digest>>,
    pub(crate) rows_per_query: usize,
    pub(crate) root: Digest,
}

pub trait MerkleTreeConstructor: GpuMerkleHash {
    fn new_merkle_tree(
        matrix: DeviceMatrix<F>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
        device_ctx: &DeviceContext,
    ) -> Result<MerkleTreeGpu<F, Self::Digest>, MerkleTreeError>;
}

pub trait MerkleProofQueryDigest: BatchQueryMerkle + Copy + Send + Sync + 'static {
    fn batch_query_merkle_proofs(
        trees: &[&MerkleTreeGpu<F, Self>],
        query_indices: &[usize],
        device_ctx: &DeviceContext,
    ) -> Result<Vec<Vec<Vec<Self>>>, MerkleTreeError>;
}

impl<F, Digest> MerkleTreeGpu<F, Digest> {
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

// Base field merkle tree — generic constructor
impl<D: Copy + Send + Sync + 'static> MerkleTreeGpu<F, D> {
    /// Build a Merkle tree using the given hash scheme `MH`.
    ///
    /// This is the primary constructor; `new()` is a convenience wrapper that
    /// fixes `MH = Poseidon2MerkleHash`.
    #[instrument(name = "merkle_tree", skip_all)]
    pub fn new_with_hash<MH: MerkleTreeConstructor<Digest = D>>(
        matrix: DeviceMatrix<F>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
        device_ctx: &DeviceContext,
    ) -> Result<Self, MerkleTreeError> {
        MH::new_merkle_tree(matrix, rows_per_query, cache_backing_matrix, device_ctx)
    }

    fn new_generic_with_hash<MH: GpuMerkleHash<Digest = D>>(
        matrix: DeviceMatrix<F>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
        device_ctx: &DeviceContext,
    ) -> Result<Self, MerkleTreeError> {
        let mem = MemTracker::start("prover.merkle_tree");
        let height = matrix.height();
        assert!(height.is_power_of_two());
        let k = validate_merkle_rows_per_query(rows_per_query, height)?;
        let query_stride = height / rows_per_query;
        let mut query_digest_layer = DeviceBuffer::<D>::with_capacity_on(query_stride, device_ctx);
        // SAFETY: query_digest_layer properly allocated
        unsafe {
            MH::compress_rows(
                &mut query_digest_layer,
                matrix.buffer(),
                matrix.width(),
                query_stride,
                k,
                device_ctx,
            )
            .map_err(MerkleTreeError::CompressingRowHashes)?;
        }
        // If not caching, drop the backing matrix at this point to save memory.
        let backing_matrix = cache_backing_matrix.then_some(matrix);

        let mut digest_layers = vec![query_digest_layer];
        while digest_layers.last().unwrap().len() > 1 {
            let prev_layer = digest_layers.last().unwrap();
            let mut layer = DeviceBuffer::<D>::with_capacity_on(prev_layer.len() / 2, device_ctx);
            let layer_len = layer.len();
            let layer_idx = digest_layers.len();
            // SAFETY:
            // - `layer` is properly allocated with half the size of `prev_layer` and does not
            //   overlap with it.
            unsafe {
                MH::compress_layer(&mut layer, prev_layer, layer_len, device_ctx).map_err(
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
        let root = d_root.to_host_on(device_ctx)?.pop().unwrap();

        mem.emit_metrics();
        Ok(Self {
            backing_matrix,
            digest_layers,
            rows_per_query,
            root,
        })
    }

    #[instrument(name = "batch_open_rows", skip_all)]
    pub fn batch_open_rows(
        backing_matrices: &[&DeviceMatrix<F>],
        query_indices: &[usize],
        query_stride: usize,
        rows_per_query: usize,
        device_ctx: &DeviceContext,
    ) -> Result<
        Vec<
            // per tree
            Vec<
                // per query index
                Vec<F>, // opened rows, concatenated for rows_per_query strided rows
            >,
        >,
        MerkleTreeError,
    > {
        if query_indices.is_empty() {
            return Ok(vec![Vec::new(); backing_matrices.len()]);
        }
        let row_idxs = query_indices
            .iter()
            .flat_map(|&query_idx| {
                debug_assert!(query_idx < query_stride);
                (0..rows_per_query)
                    .map(move |row_offset| (row_offset * query_stride + query_idx) as u32)
            })
            .collect_vec();
        let d_row_idxs = row_idxs.to_device_on(device_ctx)?;
        // PERF[jpw]: I did not batch across trees into a single kernel call because widths are
        // different so it was inconvenient. Make a new kernel if slow.
        // NOTE: par_iter requires cross-stream waits, not worth the effort
        backing_matrices
            .iter()
            .enumerate()
            .map(|(matrix_idx, matrix)| {
                let d_out = DeviceBuffer::<F>::with_capacity_on(
                    row_idxs.len() * matrix.width(),
                    device_ctx,
                );
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
                        d_row_idxs.len(),
                        device_ctx.stream.as_raw(),
                    )
                    .map_err(|error| MerkleTreeError::MatrixGetRows { error, matrix_idx })?;
                }
                let width = matrix.width();
                let out =
                    info_span!("opened_rows_d2h").in_scope(|| d_out.to_host_on(device_ctx))?;
                let opened_rows_per_query = out
                    .chunks_exact(rows_per_query * width)
                    .map(|rows| rows.to_vec())
                    .collect_vec();
                Ok(opened_rows_per_query)
            })
            .collect::<Result<Vec<_>, MerkleTreeError>>()
    }
}

impl MerkleTreeConstructor for Poseidon2MerkleHash {
    fn new_merkle_tree(
        matrix: DeviceMatrix<F>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
        device_ctx: &DeviceContext,
    ) -> Result<MerkleTreeGpu<F, Self::Digest>, MerkleTreeError> {
        MerkleTreeGpu::<F, Self::Digest>::new_generic_with_hash::<Self>(
            matrix,
            rows_per_query,
            cache_backing_matrix,
            device_ctx,
        )
    }
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl MerkleTreeConstructor for crate::hash_scheme::Bn254Poseidon2MerkleHash {
    fn new_merkle_tree(
        matrix: DeviceMatrix<F>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
        device_ctx: &DeviceContext,
    ) -> Result<MerkleTreeGpu<F, Self::Digest>, MerkleTreeError> {
        MerkleTreeGpu::<F, Self::Digest>::new_generic_with_hash::<Self>(
            matrix,
            rows_per_query,
            cache_backing_matrix,
            device_ctx,
        )
    }
}

// Base field merkle tree — Poseidon2 default constructor
impl MerkleTreeGpu<F, Digest> {
    /// Build a Merkle tree using the default Poseidon2 hash.
    ///
    /// Equivalent to `new_with_hash::<Poseidon2MerkleHash>(...)`.
    pub fn new(
        matrix: DeviceMatrix<F>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
        device_ctx: &DeviceContext,
    ) -> Result<Self, MerkleTreeError> {
        Self::new_with_hash::<Poseidon2MerkleHash>(
            matrix,
            rows_per_query,
            cache_backing_matrix,
            device_ctx,
        )
    }
}

// Base field merkle tree — generic batch query (works for any BatchQueryMerkle digest)
impl<D: BatchQueryMerkle + Send + Sync + 'static> MerkleTreeGpu<F, D> {
    fn batch_query_proofs(
        trees: &[&Self],
        query_indices: &[usize],
        device_ctx: &DeviceContext,
    ) -> Result<Vec<Vec<Vec<D>>>, MerkleTreeError> {
        if trees.is_empty() {
            return Ok(Vec::new());
        }
        // The kernel treats each layer as a separate array and does parallel accesses;
        // we lay out all the layer pointers flattened into a vec.
        let num_trees = trees.len();
        let num_queries = query_indices.len();
        let depth = trees[0].proof_depth();
        debug_assert!(
            trees.iter().all(|tree| tree.proof_depth() == depth),
            "Merkle trees don't have same depth"
        );
        if num_queries == 0 {
            return Ok(vec![Vec::new(); num_trees]);
        }
        if depth == 0 {
            return Ok(vec![vec![Vec::new(); num_queries]; num_trees]);
        }
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
        let d_layers_ptr = layers_ptr.to_device_on(device_ctx)?;
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
        let d_indices = indices.to_device_on(device_ctx)?;
        debug_assert_eq!(d_indices.len(), d_layers_ptr.len() * num_queries);

        let mut d_out = DeviceBuffer::<F>::with_capacity_on(
            d_layers_ptr.len() * num_queries * DIGEST_SIZE,
            device_ctx,
        );
        // SAFETY:
        // - d_out has size num_trees * depth * num_queries * DIGEST_SIZE in `F` elements
        // - d_layers_ptr is size `num_trees * depth` device pointers
        // - d_indices is size `num_trees * depth * num_queries` indices for merkle proof sibling
        //   indices
        // - Both Digest=[F;8] and Bn254Digest=[Bn254Scalar;1] are 32 bytes == DIGEST_SIZE * 4, so
        //   the same kernel correctly copies the raw bytes for either digest type.
        unsafe {
            query_digest_layers(
                &mut d_out,
                &d_layers_ptr,
                &d_indices,
                num_queries,
                d_layers_ptr.len(),
                device_ctx.stream.as_raw(),
            )
            .map_err(MerkleTreeError::QueryDigestLayers)?;
        }
        let out = d_out.to_host_on(device_ctx)?;
        // Chunk up the array using D::reconstruct_from_f
        let res = (0..num_trees)
            .map(|tree_idx| {
                (0..num_queries)
                    .map(|query_idx| {
                        (0..depth)
                            .map(|layer_idx| {
                                let base =
                                    (query_idx * num_trees * depth + tree_idx * depth + layer_idx)
                                        * DIGEST_SIZE;
                                D::reconstruct_from_f(&out, base)
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();
        Ok(res)
    }

    /// Batch queries multiple `trees` at _the same_ `query_indices` for merkle proofs.
    ///
    /// # Assumptions
    /// - All `trees` have the same depth.
    pub fn batch_query_merkle_proofs(
        trees: &[&Self],
        query_indices: &[usize],
        device_ctx: &DeviceContext,
    ) -> Result<
        Vec<
            // per tree
            Vec<
                // per query index
                Vec<D>, // merkle proof
            >,
        >,
        MerkleTreeError,
    >
    where
        D: MerkleProofQueryDigest,
    {
        D::batch_query_merkle_proofs(trees, query_indices, device_ctx)
    }
}

impl MerkleProofQueryDigest for Digest {
    fn batch_query_merkle_proofs(
        trees: &[&MerkleTreeGpu<F, Self>],
        query_indices: &[usize],
        device_ctx: &DeviceContext,
    ) -> Result<Vec<Vec<Vec<Self>>>, MerkleTreeError> {
        MerkleTreeGpu::<F, Self>::batch_query_proofs(trees, query_indices, device_ctx)
    }
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl MerkleProofQueryDigest for Bn254Digest {
    fn batch_query_merkle_proofs(
        trees: &[&MerkleTreeGpu<F, Self>],
        query_indices: &[usize],
        device_ctx: &DeviceContext,
    ) -> Result<Vec<Vec<Vec<Self>>>, MerkleTreeError> {
        MerkleTreeGpu::<F, Self>::batch_query_proofs(trees, query_indices, device_ctx)
    }
}

// Extension field merkle tree — generic constructor
impl<D: Copy + Send + Sync + 'static> MerkleTreeGpu<EF, D> {
    /// Build a Merkle tree from an extension-field matrix using hash scheme `MH`.
    #[instrument(name = "merkle_tree_ext", skip_all)]
    pub fn new_with_hash<MH: GpuMerkleHash<Digest = D>>(
        matrix: DeviceMatrix<EF>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
        device_ctx: &DeviceContext,
    ) -> Result<Self, MerkleTreeError> {
        let height = matrix.height();
        assert!(height.is_power_of_two());
        let k = validate_merkle_rows_per_query(rows_per_query, height)?;
        let query_stride = height / rows_per_query;
        let mut query_digest_layer = DeviceBuffer::<D>::with_capacity_on(query_stride, device_ctx);
        // SAFETY: query_digest_layer properly allocated
        unsafe {
            MH::compress_rows_ext(
                &mut query_digest_layer,
                matrix.buffer(),
                matrix.width(),
                query_stride,
                k,
                device_ctx,
            )
            .map_err(MerkleTreeError::CompressingRowHashesExt)?;
        }
        // If not caching, drop the backing matrix at this point to save memory.
        let backing_matrix = cache_backing_matrix.then_some(matrix);

        let mut digest_layers = vec![query_digest_layer];
        while digest_layers.last().unwrap().len() > 1 {
            let prev_layer = digest_layers.last().unwrap();
            let mut layer = DeviceBuffer::<D>::with_capacity_on(prev_layer.len() / 2, device_ctx);
            let layer_len = layer.len();
            let layer_idx = digest_layers.len();
            // SAFETY:
            // - `layer` is properly allocated with half the size of `prev_layer` and does not
            //   overlap with it.
            unsafe {
                MH::compress_layer(&mut layer, prev_layer, layer_len, device_ctx).map_err(
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
        let root = d_root.to_host_on(device_ctx)?.pop().unwrap();

        Ok(Self {
            backing_matrix,
            digest_layers,
            rows_per_query,
            root,
        })
    }
}

// Extension field merkle tree — Poseidon2 default constructor
impl MerkleTreeGpu<EF, Digest> {
    /// Build a Merkle tree from an extension-field matrix using Poseidon2.
    ///
    /// NOTE: currently unused because we transpose `DeviceMatrix<EF>` to
    /// `DeviceMatrix<F>` beforehand in our use cases.
    pub fn new(
        matrix: DeviceMatrix<EF>,
        rows_per_query: usize,
        cache_backing_matrix: bool,
        device_ctx: &DeviceContext,
    ) -> Result<Self, MerkleTreeError> {
        Self::new_with_hash::<Poseidon2MerkleHash>(
            matrix,
            rows_per_query,
            cache_backing_matrix,
            device_ctx,
        )
    }
}
