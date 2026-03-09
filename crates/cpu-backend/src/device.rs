//! [CpuDevice] implementation: TraceCommitter, DeviceDataTransporter, MultiRapProver,
//! OpeningProver.

use getset::Getters;
use itertools::Itertools;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    keygen::types::MultiStarkProvingKey,
    poly_common::Squarable,
    proof::{BatchConstraintProof, GkrProof, StackingProof, WhirProof},
    prover::{
        stacked_pcs::{stacked_matrix, MerkleTree, StackedPcsData},
        stacked_reduction::prove_stacked_opening_reduction,
        ColMajorMatrix, CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey,
        DeviceStarkProvingKey, MatrixDimensions, MultiRapProver, OpeningProver, ProverDevice,
        ProvingContext, StridedColMajorMatrixView, TraceCommitter,
    },
    FiatShamirTranscript, StarkProtocolConfig, SystemParams,
};
use p3_baby_bear::BabyBear;
use p3_dft::TwoAdicSubgroupDft;
use p3_matrix::dense::RowMajorMatrix;
use p3_field::{ExtensionField, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{backend::CpuBackend, error::CpuProverError, stacked_reduction::StackedReductionCpuNew};

/// Row-major CPU prover device.
#[derive(Clone, Getters, derive_new::new)]
pub struct CpuDevice<SC> {
    #[getset(get = "pub")]
    config: SC,
}

impl<SC: StarkProtocolConfig> CpuDevice<SC> {
    pub fn params(&self) -> &SystemParams {
        self.config.params()
    }
}

impl<SC, TS> ProverDevice<CpuBackend<SC>, TS> for CpuDevice<SC>
where
    SC: StarkProtocolConfig,
    SC::F: Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    TS: FiatShamirTranscript<SC>,
{
    type Error = CpuProverError;
}

impl<SC: StarkProtocolConfig> TraceCommitter<CpuBackend<SC>> for CpuDevice<SC>
where
    SC::F: Ord,
{
    type Error = CpuProverError;

    #[instrument(level = "info", name = "trace_commit_cpu", skip_all)]
    fn commit(
        &self,
        traces: &[&RowMajorMatrix<SC::F>],
    ) -> Result<(SC::Digest, StackedPcsData<SC::F, SC::Digest>), Self::Error> {
        // Convert row-major to col-major for stacking commitment
        let col_major_traces: Vec<ColMajorMatrix<SC::F>> = traces
            .iter()
            .map(|rm| ColMajorMatrix::from_row_major(rm))
            .collect();
        let col_major_refs: Vec<&ColMajorMatrix<SC::F>> = col_major_traces.iter().collect();

        let params = self.params();
        let (q_trace, layout) = stacked_matrix(params.l_skip, params.n_stack, &col_major_refs)?;
        let tree = rs_encode_and_merkle_cpu(
            self.config().hasher(),
            params.l_skip,
            params.log_blowup,
            &q_trace,
            1 << params.k_whir(),
        );
        let root = tree.root()?;
        let data = StackedPcsData::new(layout, q_trace, tree);
        Ok((root, data))
    }
}

impl<SC, TS> MultiRapProver<CpuBackend<SC>, TS> for CpuDevice<SC>
where
    SC: StarkProtocolConfig,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
    TS: FiatShamirTranscript<SC>,
{
    type PartialProof = (GkrProof<SC>, BatchConstraintProof<SC>);
    type Artifacts = Vec<SC::EF>;

    type Error = CpuProverError;

    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
        ctx: &ProvingContext<CpuBackend<SC>>,
        _common_main_pcs_data: &StackedPcsData<SC::F, SC::Digest>,
    ) -> Result<((GkrProof<SC>, BatchConstraintProof<SC>), Vec<SC::EF>), Self::Error> {
        let (gkr_proof, batch_constraint_proof, r) =
            crate::logup_zerocheck::prove_zerocheck_and_logup::<SC, _>(transcript, mpk, ctx)?;
        Ok(((gkr_proof, batch_constraint_proof), r))
    }
}

impl<SC, TS> OpeningProver<CpuBackend<SC>, TS> for CpuDevice<SC>
where
    SC: StarkProtocolConfig,
    SC::F: Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    TS: FiatShamirTranscript<SC>,
{
    type OpeningProof = (StackingProof<SC>, WhirProof<SC>);
    type OpeningPoints = Vec<SC::EF>;

    type Error = CpuProverError;

    fn prove_openings(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
        ctx: ProvingContext<CpuBackend<SC>>,
        common_main_pcs_data: StackedPcsData<SC::F, SC::Digest>,
        r: Vec<SC::EF>,
    ) -> Result<(StackingProof<SC>, WhirProof<SC>), Self::Error> {
        let params = self.params();

        let need_rot_per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, _)| mpk.per_air[*air_idx].vk.params.need_rot)
            .collect_vec();

        let pre_cached_pcs_data_per_commit: Vec<_> = ctx
            .per_trace
            .iter()
            .flat_map(|(air_idx, trace_ctx)| {
                mpk.per_air[*air_idx]
                    .preprocessed_data
                    .iter()
                    .chain(&trace_ctx.cached_mains)
                    .map(|cd| cd.data.clone())
            })
            .collect();

        let mut stacked_per_commit = vec![&common_main_pcs_data];
        for data in &pre_cached_pcs_data_per_commit {
            stacked_per_commit.push(data);
        }
        let mut need_rot_per_commit = vec![need_rot_per_trace];
        for (air_idx, trace_ctx) in &ctx.per_trace {
            let need_rot = mpk.per_air[*air_idx].vk.params.need_rot;
            if mpk.per_air[*air_idx].preprocessed_data.is_some() {
                need_rot_per_commit.push(vec![need_rot]);
            }
            for _ in &trace_ctx.cached_mains {
                need_rot_per_commit.push(vec![need_rot]);
            }
        }
        let (stacking_proof, u_prisma) =
            prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpuNew<SC>>(
                self,
                transcript,
                params.n_stack,
                stacked_per_commit,
                need_rot_per_commit,
                &r,
            );

        let (&u0, u_rest) = u_prisma
            .split_first()
            .ok_or(openvm_stark_backend::prover::error::WhirProverError::UPrismaEmpty)?;
        let u_cube = u0
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(u_rest.iter().copied())
            .collect_vec();

        // Convert to col-major for WHIR
        let committed_mats = std::iter::once(&common_main_pcs_data)
            .chain(pre_cached_pcs_data_per_commit.iter().map(|d| d.as_ref()))
            .map(|d| (&d.matrix, &d.tree))
            .collect_vec();

        let whir_proof = crate::whir::prove_whir_opening_cpu::<SC, _>(
            transcript,
            self.config().hasher(),
            params.l_skip,
            params.log_blowup,
            &params.whir,
            &committed_mats,
            &u_cube,
        )?;
        Ok((stacking_proof, whir_proof))
    }
}

impl<SC: StarkProtocolConfig> DeviceDataTransporter<SC, CpuBackend<SC>> for CpuDevice<SC> {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKey<SC>,
    ) -> DeviceMultiStarkProvingKey<CpuBackend<SC>> {
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|d| {
                    let view: StridedColMajorMatrixView<'_, SC::F> = d.mat_view(0).into();
                    let row_major = view.to_row_major_matrix();
                    let trace = row_major;
                    CommittedTraceData {
                        commitment: d.commit().unwrap(),
                        trace,
                        data: d.clone(),
                    }
                });
                DeviceStarkProvingKey {
                    air_name: pk.air_name.clone(),
                    vk: pk.vk.clone(),
                    preprocessed_data,
                    other_data: (),
                }
            })
            .collect();
        DeviceMultiStarkProvingKey::new(
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.max_constraint_degree,
            mpk.params.clone(),
            mpk.vk_pre_hash,
        )
    }

    fn transport_matrix_to_device(
        &self,
        matrix: &ColMajorMatrix<SC::F>,
    ) -> RowMajorMatrix<SC::F> {
        let view: StridedColMajorMatrixView<'_, SC::F> = matrix.as_view().into();
        view.to_row_major_matrix()
    }

    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<SC::F, SC::Digest>,
    ) -> StackedPcsData<SC::F, SC::Digest> {
        pcs_data.clone()
    }

    fn transport_matrix_from_device_to_host(
        &self,
        matrix: &RowMajorMatrix<SC::F>,
    ) -> ColMajorMatrix<SC::F> {
        ColMajorMatrix::from_row_major(matrix)
    }
}

/// In-place PLE evaluation-to-coefficient conversion.
/// Uses inline DIF iDFT with precomputed twiddle factors to eliminate
/// the per-chunk allocation overhead of the shared `eval_to_coeff_rs_message`.
pub(crate) fn eval_to_coeff_cpu<F: TwoAdicField>(l_skip: usize, evals: &[F]) -> Vec<F> {
    let chunk_len = 1usize << l_skip;
    let mut buf = evals.to_vec();

    // Precompute twiddle factors for DIF iDFT of size chunk_len.
    // Layer l has block_size = chunk_len >> l, half_block twiddle factors.
    let omega_inv = F::two_adic_generator(l_skip).inverse();
    let two = F::ONE + F::ONE;
    let mut n_field = F::ONE;
    for _ in 0..l_skip {
        n_field *= two;
    }
    let n_inv = n_field.inverse();
    let twiddles: Vec<Vec<F>> = (0..l_skip)
        .map(|layer| {
            let half_block = chunk_len >> (layer + 1);
            let w_step = omega_inv.exp_power_of_2(layer);
            w_step.powers().take(half_block).collect()
        })
        .collect();

    // Phase 1: In-place DIF iDFT on each chunk, then bit-reverse + scale.
    for chunk in buf.chunks_exact_mut(chunk_len) {
        // DIF (Gentleman-Sande) butterflies: natural-order input → bit-reversed output.
        let mut block_size = chunk_len;
        for tw in &twiddles {
            let half = block_size >> 1;
            let mut k = 0;
            while k < chunk_len {
                for j in 0..half {
                    let u = chunk[k + j];
                    let v = chunk[k + j + half];
                    chunk[k + j] = u + v;
                    chunk[k + j + half] = (u - v) * tw[j];
                }
                k += block_size;
            }
            block_size = half;
        }

        // Bit-reverse permutation.
        for i in 0..chunk_len {
            let j = i.reverse_bits() >> (usize::BITS as u32 - l_skip as u32);
            if i < j {
                chunk.swap(i, j);
            }
        }

        // Scale by 1/N.
        for val in chunk.iter_mut() {
            *val *= n_inv;
        }
    }

    // Phase 2: Convert MLE coefficients to evaluations in-place.
    for chunk in buf.chunks_exact_mut(chunk_len) {
        mle_coeffs_to_evals_inplace(chunk);
    }

    buf
}

/// In-place conversion from multilinear polynomial coefficients to evaluations
/// on the boolean hypercube. Equivalent to `Mle::coeffs_to_evals_inplace`.
fn mle_coeffs_to_evals_inplace<F: TwoAdicField>(a: &mut [F]) {
    let n = log2_strict_usize(a.len());
    for b in 0..n {
        let step = 1usize << b;
        let span = step << 1;
        let mut i = 0;
        while i < a.len() {
            for j in 0..step {
                a[i + j + step] += a[i + j];
            }
            i += span;
        }
    }
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

/// Packed SIMD Merkle tree digest layer compression for BabyBear Poseidon2.
///
/// Compresses `F::Packing::WIDTH` independent (left, right) digest pairs simultaneously
/// using packed Poseidon2 permutation via `TruncatedPermutation`.
/// On aarch64/NEON: 4x throughput, on x86/AVX2: 8x, scalar fallback: 1x.
///
/// Handles both query-stride interleaved layers and standard binary tree layers.
pub(crate) fn build_digest_layers_packed_babybear(
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

/// Fused RS encoding + Merkle tree construction.
#[instrument(name = "rs_encode_and_merkle_cpu", skip_all)]
fn rs_encode_and_merkle_cpu<F, H>(
    hasher: &H,
    l_skip: usize,
    log_blowup: usize,
    eval_matrix: &ColMajorMatrix<F>,
    rows_per_query: usize,
) -> MerkleTree<F, H::Digest>
where
    F: TwoAdicField + Ord + 'static,
    H: MerkleHasher<F = F>,
{
    use p3_dft::Radix2DitParallel;
    use p3_matrix::dense::RowMajorMatrix as P3RowMajorMatrix;

    let height = eval_matrix.height();
    let codeword_height = height.checked_shl(log_blowup as u32).unwrap();
    let width = eval_matrix.width();

    // Phase 1: Convert PLE evaluations to coefficients (parallel per column).
    let coeff_vecs: Vec<Vec<F>> = tracing::info_span!("eval_to_coeff_phase").in_scope(|| {
        eval_matrix
            .values
            .par_chunks_exact(height)
            .map(|column_evals| {
                let mut coeffs = eval_to_coeff_cpu(l_skip, column_evals);
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
            // Packed SIMD hashing — 4x throughput on NEON, 8x on AVX2.
            // SAFETY: TypeId checks guarantee F = BabyBear and H::Digest = [BabyBear; 8],
            // so the pointer casts are between identical types (zero-cost reinterpretation).
            let bb_vals: &[BabyBear] = unsafe {
                std::slice::from_raw_parts(rm_vals.as_ptr().cast::<BabyBear>(), rm_vals.len())
            };
            let bb_digests = hash_rows_packed_babybear(bb_vals, width, codeword_height, num_leaves);
            let mut md = std::mem::ManuallyDrop::new(bb_digests);
            unsafe {
                Vec::from_raw_parts(md.as_mut_ptr().cast::<H::Digest>(), md.len(), md.capacity())
            }
        } else {
            (0..num_leaves)
                .into_par_iter()
                .map(|r| {
                    if r < codeword_height {
                        hasher.hash_slice(&rm_vals[r * width..(r + 1) * width])
                    } else {
                        hasher.hash_slice(&vec![F::ZERO; width])
                    }
                })
                .collect()
        }
    });

    // Phase 5: Build Merkle digest layers — packed SIMD for BabyBear, scalar fallback otherwise.
    let digest_layers = tracing::info_span!("digest_layers").in_scope(|| {
        use std::any::TypeId;
        if TypeId::of::<F>() == TypeId::of::<BabyBear>()
            && TypeId::of::<H::Digest>() == TypeId::of::<[BabyBear; 8]>()
        {
            // Packed SIMD compression — 4x throughput on NEON, 8x on AVX2.
            // SAFETY: TypeId checks guarantee H::Digest = [BabyBear; 8].
            let bb_hashes: Vec<[BabyBear; 8]> = unsafe {
                let mut md = std::mem::ManuallyDrop::new(row_hashes);
                Vec::from_raw_parts(
                    md.as_mut_ptr().cast::<[BabyBear; 8]>(),
                    md.len(),
                    md.capacity(),
                )
            };
            let bb_layers = build_digest_layers_packed_babybear(bb_hashes, rows_per_query);
            bb_layers
                .into_iter()
                .map(|layer| unsafe {
                    let mut md = std::mem::ManuallyDrop::new(layer);
                    Vec::from_raw_parts(
                        md.as_mut_ptr().cast::<H::Digest>(),
                        md.len(),
                        md.capacity(),
                    )
                })
                .collect()
        } else {
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
    });

    // Phase 6: Transpose to col-major for backing storage.
    let cm_matrix = tracing::info_span!("transpose_to_cm").in_scope(|| {
        let mut cm_values = F::zero_vec(codeword_height * width);
        cm_values
            .par_chunks_exact_mut(codeword_height)
            .enumerate()
            .for_each(|(j, col)| {
                for i in 0..codeword_height {
                    col[i] = rm_vals[i * width + j];
                }
            });
        ColMajorMatrix::new(cm_values, width)
    });

    // SAFETY: digest_layers were just computed as correct Merkle hashes over cm_matrix
    // by hash_rows_packed_babybear and build_digest_layers_packed_babybear above.
    // rows_per_query is forwarded from the validated SystemParams.
    unsafe { MerkleTree::from_raw_parts(cm_matrix, digest_layers, rows_per_query) }
}
