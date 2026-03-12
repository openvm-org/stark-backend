//! [CpuMerkleTree] — a Merkle tree backed by a [RowMajorMatrix].
//!
//! This mirrors the interface of [`openvm_stark_backend::prover::stacked_pcs::MerkleTree`]
//! but stores its codeword matrix in row-major layout for cache-friendly row hashing
//! and contiguous row access in query answering.

use openvm_stark_backend::{
    hasher::MerkleHasher,
    prover::{error::StackedPcsError, ColMajorMatrix, MatrixDimensions},
};
use p3_baby_bear::BabyBear;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{device::eval_to_coeff_cpu, two_adic::DftTwiddles};

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

/// Reinterpret a `Vec<A>` as `Vec<B>` when both types have identical layout.
///
/// # Safety
/// The caller must guarantee that `A` and `B` have the same size, alignment,
/// and compatible memory representations (e.g. verified via `TypeId` checks).
pub(crate) unsafe fn reinterpret_vec<A, B>(v: Vec<A>) -> Vec<B> {
    debug_assert_eq!(std::mem::size_of::<A>(), std::mem::size_of::<B>());
    debug_assert_eq!(std::mem::align_of::<A>(), std::mem::align_of::<B>());
    let mut md = std::mem::ManuallyDrop::new(v);
    Vec::from_raw_parts(md.as_mut_ptr().cast::<B>(), md.len(), md.capacity())
}

/// Packed SIMD row hashing for BabyBear Poseidon2.
///
/// Hashes `F::Packing::WIDTH` rows simultaneously using packed field arithmetic.
/// On aarch64 NEON: 4 rows/hash, on x86 AVX2: 8 rows/hash, scalar fallback: 1 row/hash.
///
/// Constructs a fresh `PaddingFreeSponge` from `default_babybear_poseidon2_16()`,
/// which is deterministic and identical to the one in `BabyBearPoseidon2Config`.
pub(crate) fn hash_rows_packed_babybear(
    rm_vals: &[BabyBear],
    width: usize,
    codeword_height: usize,
    num_leaves: usize,
) -> Vec<[BabyBear; 8]> {
    use openvm_stark_backend::p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
    use p3_baby_bear::default_babybear_poseidon2_16;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};

    type P = <BabyBear as Field>::Packing;

    let perm = default_babybear_poseidon2_16();
    let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(perm);
    let pack_width = P::WIDTH;

    let mut digests = vec![[BabyBear::ZERO; 8]; num_leaves];

    digests
        .par_chunks_mut(pack_width)
        .enumerate()
        .for_each(|(chunk_idx, digest_chunk)| {
            let base_row = chunk_idx * pack_width;

            if digest_chunk.len() == pack_width {
                // SIMD: pack `pack_width` rows into packed field elements
                let packed_row: Vec<P> = (0..width)
                    .map(|col| {
                        P::from_fn(|lane| {
                            let row = base_row + lane;
                            if row < codeword_height {
                                rm_vals[row * width + col]
                            } else {
                                BabyBear::ZERO
                            }
                        })
                    })
                    .collect();

                // Hash produces [P; 8] — pack_width digests interleaved across lanes
                let packed_digest: [P; 8] = sponge.hash_slice(&packed_row);

                // Unpack individual digests from SIMD lanes
                for lane in 0..pack_width {
                    for d in 0..8 {
                        digest_chunk[lane][d] = packed_digest[d].as_slice()[lane];
                    }
                }
            } else {
                // Scalar fallback for partial final chunk
                for (lane, digest) in digest_chunk.iter_mut().enumerate() {
                    let row = base_row + lane;
                    if row < codeword_height {
                        *digest = sponge.hash_slice(&rm_vals[row * width..(row + 1) * width]);
                    }
                }
            }
        });

    digests
}

/// Build Merkle digest layers from row hashes, dispatching to packed SIMD for
/// BabyBear or scalar fallback otherwise.
///
/// This is the shared implementation used by both `rs_encode_and_merkle_cpu` and
/// `build_ef_merkle_tree_packed` in the WHIR module.
pub(crate) fn build_digest_layers<F, H>(
    row_hashes: Vec<H::Digest>,
    rows_per_query: usize,
    hasher: &H,
) -> Vec<Vec<H::Digest>>
where
    F: TwoAdicField + Ord + 'static,
    H: MerkleHasher<F = F>,
{
    use std::any::TypeId;
    if TypeId::of::<F>() == TypeId::of::<BabyBear>()
        && TypeId::of::<H::Digest>() == TypeId::of::<[BabyBear; 8]>()
    {
        // SAFETY: TypeId checks guarantee H::Digest = [BabyBear; 8].
        let bb_hashes: Vec<[BabyBear; 8]> = unsafe { reinterpret_vec(row_hashes) };
        let bb_layers = build_digest_layers_packed_babybear(bb_hashes, rows_per_query);
        bb_layers
            .into_iter()
            .map(|layer| unsafe { reinterpret_vec(layer) })
            .collect()
    } else {
        build_digest_layers_scalar(row_hashes, rows_per_query, hasher)
    }
}

pub(crate) fn hash_rows_with_padding<D, RowHashFn, PaddingHashFn>(
    num_leaves: usize,
    codeword_height: usize,
    row_hash_fn: RowHashFn,
    padding_hash_fn: PaddingHashFn,
) -> Vec<D>
where
    D: Send,
    RowHashFn: Fn(usize) -> D + Sync + Send,
    PaddingHashFn: Fn() -> D + Sync + Send,
{
    (0..num_leaves)
        .into_par_iter()
        .map(|r| {
            if r < codeword_height {
                row_hash_fn(r)
            } else {
                padding_hash_fn()
            }
        })
        .collect()
}

/// Fused RS encoding + Merkle tree construction with RowMajor backing.
///
/// Eliminates the Phase 6 transpose to col-major that the reference implementation requires,
/// since `CpuMerkleTree` stores the codeword matrix in row-major layout directly.
#[instrument(name = "rs_encode_and_merkle_cpu", skip_all)]
pub(crate) fn rs_encode_and_merkle_cpu<F, H>(
    hasher: &H,
    l_skip: usize,
    log_blowup: usize,
    eval_matrix: &ColMajorMatrix<F>,
    rows_per_query: usize,
) -> CpuMerkleTree<F, H::Digest>
where
    F: TwoAdicField + Ord + 'static,
    H: MerkleHasher<F = F>,
{
    use p3_dft::Radix2DitParallel;
    use p3_matrix::dense::RowMajorMatrix as P3RowMajorMatrix;

    let height = eval_matrix.height();
    let codeword_height = height.checked_shl(log_blowup as u32).unwrap();
    let width = eval_matrix.width();
    let twiddles = DftTwiddles::new(l_skip);

    // Phase 1: Convert PLE evaluations to coefficients (parallel per column).
    let coeff_vecs: Vec<Vec<F>> = tracing::info_span!("eval_to_coeff_phase").in_scope(|| {
        eval_matrix
            .values
            .par_chunks_exact(height)
            .map(|column_evals| {
                let mut coeffs = eval_to_coeff_cpu(column_evals, &twiddles);
                coeffs.resize(codeword_height, F::ZERO);
                coeffs
            })
            .collect()
    });

    // Phase 2: Transpose column vectors into a RowMajorMatrix for batch DFT.
    let rm_mat: P3RowMajorMatrix<F> = tracing::info_span!("transpose_to_rm").in_scope(|| {
        let mut rm_values = F::zero_vec(codeword_height * width);
        rm_values
            .par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, row)| {
                for (j, col) in coeff_vecs.iter().enumerate() {
                    row[j] = col[i];
                }
            });
        P3RowMajorMatrix::new(rm_values, width)
    });
    drop(coeff_vecs);

    // Phase 3: Batch DFT — single level of rayon parallelism + SIMD butterflies.
    let rm_result = tracing::info_span!("dft_batch").in_scope(|| {
        use p3_matrix::Matrix as _;
        Radix2DitParallel::default()
            .dft_batch(rm_mat)
            .to_row_major_matrix()
    });

    // Phase 4: Hash rows — use packed SIMD for BabyBear, scalar fallback otherwise.
    let num_leaves = codeword_height.next_power_of_two();
    let rm_vals = &rm_result.values;
    let row_hashes: Vec<H::Digest> = tracing::info_span!("row_hash").in_scope(|| {
        use std::any::TypeId;
        if TypeId::of::<F>() == TypeId::of::<BabyBear>()
            && TypeId::of::<H::Digest>() == TypeId::of::<[BabyBear; 8]>()
        {
            let bb_vals: &[BabyBear] = unsafe {
                std::slice::from_raw_parts(rm_vals.as_ptr().cast::<BabyBear>(), rm_vals.len())
            };
            let bb_digests = hash_rows_packed_babybear(bb_vals, width, codeword_height, num_leaves);
            unsafe { reinterpret_vec(bb_digests) }
        } else {
            let zero_row = vec![F::ZERO; width];
            hash_rows_with_padding(
                num_leaves,
                codeword_height,
                |r| hasher.hash_slice(&rm_vals[r * width..(r + 1) * width]),
                || hasher.hash_slice(&zero_row),
            )
        }
    });

    // Phase 5: Build Merkle digest layers.
    let digest_layers = tracing::info_span!("digest_layers")
        .in_scope(|| build_digest_layers::<F, H>(row_hashes, rows_per_query, hasher));

    // No Phase 6: RowMajor DFT result is stored directly as the backing matrix.
    // This eliminates the O(n*m) transpose to col-major.

    // SAFETY: digest_layers were just computed as correct Merkle hashes over rm_result
    // by hash_rows_packed_babybear and build_digest_layers_packed_babybear above.
    // rows_per_query is forwarded from the validated SystemParams.
    unsafe { CpuMerkleTree::from_raw_parts(rm_result, digest_layers, rows_per_query) }
}

/// Scalar fallback for building Merkle digest layers.
fn build_digest_layers_scalar<H: MerkleHasher>(
    row_hashes: Vec<H::Digest>,
    rows_per_query: usize,
    hasher: &H,
) -> Vec<Vec<H::Digest>> {
    let num_leaves = row_hashes.len();
    let query_stride = num_leaves / rows_per_query;
    let mut query_digest_layer = row_hashes;
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
    let mut layers = vec![query_digest_layer];
    while layers.last().unwrap().len() > 1 {
        let prev = layers.last().unwrap();
        let layer: Vec<_> = prev
            .par_chunks_exact(2)
            .map(|pair| hasher.compress(pair[0], pair[1]))
            .collect();
        layers.push(layer);
    }
    layers
}

/// Packed SIMD Merkle tree digest layer compression for BabyBear Poseidon2.
fn build_digest_layers_packed_babybear(
    row_hashes: Vec<[BabyBear; 8]>,
    rows_per_query: usize,
) -> Vec<Vec<[BabyBear; 8]>> {
    use openvm_stark_backend::p3_symmetric::{PseudoCompressionFunction, TruncatedPermutation};
    use p3_baby_bear::default_babybear_poseidon2_16;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};

    type P = <BabyBear as Field>::Packing;
    let pack_width = P::WIDTH;

    let perm = default_babybear_poseidon2_16();
    let compressor = TruncatedPermutation::<_, 2, 8, 16>::new(perm);

    let num_leaves = row_hashes.len();
    let query_stride = num_leaves / rows_per_query;

    // Phase 1: Query-stride interleaved layers.
    let mut prev_layer = row_hashes;
    for _ in 0..log2_strict_usize(rows_per_query) {
        let n = prev_layer.len() / 2;
        let qs = query_stride;
        let mut next_layer = vec![[BabyBear::ZERO; 8]; n];

        next_layer
            .par_chunks_mut(pack_width)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let base = chunk_idx * pack_width;
                let actual = out_chunk.len();

                if actual == pack_width {
                    let mut packed_input: [[P; 8]; 2] = [[P::default(); 8]; 2];
                    for d in 0..8 {
                        packed_input[0][d] = P::from_fn(|lane| {
                            let i = base + lane;
                            let x = i / qs;
                            let y = i % qs;
                            prev_layer[2 * x * qs + y][d]
                        });
                        packed_input[1][d] = P::from_fn(|lane| {
                            let i = base + lane;
                            let x = i / qs;
                            let y = i % qs;
                            prev_layer[(2 * x + 1) * qs + y][d]
                        });
                    }
                    let packed_result: [P; 8] = compressor.compress(packed_input);
                    for lane in 0..pack_width {
                        for d in 0..8 {
                            out_chunk[lane][d] = packed_result[d].as_slice()[lane];
                        }
                    }
                } else {
                    for lane in 0..actual {
                        let i = base + lane;
                        let x = i / qs;
                        let y = i % qs;
                        out_chunk[lane] = compressor.compress([
                            prev_layer[2 * x * qs + y],
                            prev_layer[(2 * x + 1) * qs + y],
                        ]);
                    }
                }
            });

        prev_layer = next_layer;
    }

    // Phase 2: Standard binary tree layers (adjacent pairs).
    let mut layers = vec![prev_layer];
    while layers.last().unwrap().len() > 1 {
        let n = layers.last().unwrap().len() / 2;
        let mut layer = vec![[BabyBear::ZERO; 8]; n];
        {
            let prev = layers.last().unwrap();
            layer
                .par_chunks_mut(pack_width)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let base = chunk_idx * pack_width;
                    let actual = out_chunk.len();

                    if actual == pack_width {
                        let mut packed_input: [[P; 8]; 2] = [[P::default(); 8]; 2];
                        for d in 0..8 {
                            packed_input[0][d] = P::from_fn(|lane| prev[2 * (base + lane)][d]);
                            packed_input[1][d] = P::from_fn(|lane| prev[2 * (base + lane) + 1][d]);
                        }
                        let packed_result: [P; 8] = compressor.compress(packed_input);
                        for lane in 0..pack_width {
                            for d in 0..8 {
                                out_chunk[lane][d] = packed_result[d].as_slice()[lane];
                            }
                        }
                    } else {
                        for lane in 0..actual {
                            let i = base + lane;
                            out_chunk[lane] = compressor.compress([prev[2 * i], prev[2 * i + 1]]);
                        }
                    }
                });
        }
        layers.push(layer);
    }

    layers
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
        let mat = RowMajorMatrix::new(vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3);
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
