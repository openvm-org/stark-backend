use std::{iter::zip, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::{
    base::DeviceMatrix,
    cuda::kernels::{fri::split_ext_poly_to_base_col_major_matrix, lde::batch_expand_pad},
    ntt::batch_ntt,
};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use openvm_stark_backend::{
    poseidon2::sponge::FiatShamirTranscript,
    proof::{MerkleProof, WhirProof},
    prover::MatrixDimensions,
    SystemParams,
};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    cuda::{
        batch_ntt_small::batch_ntt_small,
        mle_interpolate::mle_interpolate_stage_ext,
        poly::{eval_poly_ext_at_point_from_base, transpose_fp_to_fpext_vec},
        whir::{
            _whir_sumcheck_coeff_moments_required_temp_buffer_size, w_moments_accumulate,
            whir_algebraic_batch_traces, whir_fold_coeffs_and_moments,
            whir_sumcheck_coeff_moments_round,
        },
    },
    merkle_tree::MerkleTreeGpu,
    poly::evals_eq_hypercube,
    sponge::DuplexSpongeGpu,
    stacked_pcs::rs_code_matrix,
    stacked_reduction::StackedPcsData2,
    Digest, WhirProverError, D_EF, EF, F,
};

#[repr(C)]
pub(crate) struct BatchingTracePacket {
    /// Pointer to trace device buffer
    ptr: *const F,
    /// Trace height
    height: u32,
    /// Trace width
    width: u32,
    /// Row start in the stacked matrix
    stacked_row_start: u32,
    /// Index in mu powers
    mu_idx: u32,
}

#[instrument(
    name = "prover.openings.whir",
    level = "info",
    skip_all,
    fields(phase = "prover")
)]
pub fn prove_whir_opening_gpu(
    params: &SystemParams,
    transcript: &mut DuplexSpongeGpu,
    mut stacked_per_commit: Vec<StackedPcsData2>,
    u: &[EF],
) -> Result<WhirProof, WhirProverError> {
    let mem = MemTracker::start("prover.prove_whir_opening");
    let l_skip = params.l_skip;
    let log_blowup = params.log_blowup;
    let k_whir = params.k_whir();
    let whir_params = params.whir();
    let log_final_poly_len = params.log_final_poly_len();

    let height = stacked_per_commit[0].layout().height();
    debug_assert!(stacked_per_commit
        .iter()
        .all(|d| d.layout().height() == height));
    let mut m = log2_strict_usize(height);
    assert_eq!(m, u.len());
    debug_assert!(m >= l_skip);

    // Proof-of-work grinding before μ batching challenge.
    // This amplifies soundness of the initial batching step.
    let mu_pow_witness = transcript.grind(whir_params.mu_pow_bits);
    // Sample randomness for algebraic batching.
    // We batch the codewords for \hat{q}_j together _before_ applying WHIR.
    let mu = transcript.sample_ext();
    let num_commits = stacked_per_commit.len();

    // The coefficient table of `\hat{f}` in the current WHIR round (MLE coefficient form).
    let mut f_ple_evals = DeviceBuffer::<F>::with_capacity(height * D_EF);
    // We algebraically batch all matrices together so we only need to interpolate one column vector
    {
        let mut packets = Vec::new();
        let mut total_stacked_width = 0u32;
        for stacked in &stacked_per_commit {
            let layout = stacked.layout();
            for (trace, &idx) in zip(&stacked.traces, &layout.mat_starts) {
                let (_, _, s) = layout.sorted_cols[idx];
                let packet = BatchingTracePacket {
                    ptr: trace.buffer().as_ptr(),
                    height: trace.height() as u32,
                    width: trace.width() as u32,
                    stacked_row_start: s.row_idx as u32,
                    mu_idx: total_stacked_width + s.col_idx as u32,
                };
                packets.push(packet);
            }
            total_stacked_width += layout.width() as u32;
        }
        let mu_powers = mu.powers().take(total_stacked_width as usize).collect_vec();
        let d_mu_powers = mu_powers.to_device()?;
        let d_packets = packets.to_device()?;
        // SAFETY:
        // - `mu_powers` has length `total_width`.
        // - `f_ple_evals` has capacity `height * D_EF` (stacked height in base coordinates).
        // - `d_packets` contain valid pointers and stacked row indices by construction.
        unsafe {
            whir_algebraic_batch_traces(&mut f_ple_evals, &d_packets, &d_mu_powers, 1 << l_skip)
                .map_err(WhirProverError::AlgebraicBatch)?;
        }
        for stacked in &mut stacked_per_commit {
            // drop traces to free device memory if RS codewords matrix exists
            if stacked.inner.tree.backing_matrix.is_some() {
                stacked.traces.clear();
            }
        }
    } // common_main_pcs_data.matrix has now been freed

    // Compute \hat{f} coefficients:
    //
    // Step 1: iDFT per chunk of size 2^l_skip. After this, each chunk holds univariate
    // coefficients c_z(x) of f(Z, x) for a fixed boolean assignment x in H_{m - l_skip}.
    unsafe {
        let num_poly = f_ple_evals.len() >> l_skip;
        batch_ntt_small(&mut f_ple_evals, l_skip, num_poly, true)
            .map_err(WhirProverError::CustomBatchIntt)?;
    }
    let mut f_coeffs = DeviceBuffer::<EF>::with_capacity(height);
    // SAFETY: `f_ple_evals` is constructed with length `height * D_EF`.
    unsafe {
        transpose_fp_to_fpext_vec(&mut f_coeffs, &f_ple_evals)
            .map_err(WhirProverError::Transpose)?;
    }
    drop(f_ple_evals);
    // Step 2: Within-chunk zeta (stages 0..l_skip). Applies the subset-zeta transform
    // over the Z-index bits, converting univariate coefficients (root-of-unity basis)
    // into hypercube evaluations over H_{l_skip}. Together with step 1, this computes
    // eval_to_coeff_rs_message, i.e. the MLE coefficient table of \hat{f}.
    for i in 0..l_skip {
        let step = 1u32 << i;
        // SAFETY: `f_coeffs` has length `2^m` with `m >= l_skip`.
        unsafe {
            mle_interpolate_stage_ext(&mut f_coeffs, step, false)
                .map_err(|error| WhirProverError::MleInterpolate { error, step })?;
        }
    }

    debug_assert_eq!((m - log_final_poly_len) % k_whir, 0);
    let num_whir_rounds = (m - log_final_poly_len) / k_whir;
    assert!(num_whir_rounds > 0);

    // We maintain moments of \hat{w}:
    // M[T] = sum_{x superset T} \hat{w}(x).
    // For initial \hat{w} = mobius_eq(u, -), these moments are exactly eq(u, -).
    let mut w_moments = DeviceBuffer::<EF>::with_capacity(1 << m);
    unsafe {
        evals_eq_hypercube(&mut w_moments, u).map_err(WhirProverError::EvalEq)?;
    }

    let mut whir_sumcheck_polys: Vec<[EF; 2]> = vec![];
    let mut codeword_commits = vec![];
    let mut ood_values = vec![];
    // per commitment, per whir query, per column
    let mut initial_round_opened_rows: Vec<Vec<Vec<Vec<F>>>> = vec![vec![]; num_commits];
    let mut initial_round_merkle_proofs: Vec<Vec<MerkleProof>> = vec![];
    let mut codeword_opened_values: Vec<Vec<Vec<EF>>> = vec![];
    let mut codeword_merkle_proofs: Vec<Vec<MerkleProof>> = vec![];
    let mut folding_pow_witnesses = vec![];
    let mut query_phase_pow_witnesses = vec![];
    let mut rs_tree = None;
    let mut log_rs_domain_size = m + log_blowup;
    let mut final_poly = None;

    let mut d_s_evals = DeviceBuffer::<EF>::with_capacity(2);
    let mut d_sumcheck_tmp = DeviceBuffer::<EF>::new();

    mem.tracing_info("before_whir_rounds");
    // We will drop `stacked_per_commit` and hence `common_main_pcs_data` after whir round 0.
    for (whir_round, round_params) in whir_params.rounds.iter().enumerate() {
        let is_last_round = whir_round == num_whir_rounds - 1;
        // Run k_whir rounds of sumcheck on `sum_{x in H_m} \hat{w}(\hat{f}(x), x)`
        for round in 0..k_whir {
            // Do not use f_coeffs.len() because it might have extra capacity.
            let f_height = 1 << (m - round);
            debug_assert!(
                f_coeffs.len() >= f_height,
                "f_coeffs has length {}, expected 2^{} for m={m}, round={round}",
                f_coeffs.len(),
                m - round
            );
            debug_assert!(w_moments.len() >= f_height);
            let output_height = f_height / 2;
            let tmp_buffer_capacity =
                unsafe { _whir_sumcheck_coeff_moments_required_temp_buffer_size(f_height as u32) };
            if d_sumcheck_tmp.len() < tmp_buffer_capacity as usize {
                d_sumcheck_tmp = DeviceBuffer::<EF>::with_capacity(tmp_buffer_capacity as usize);
            }
            let mut new_f_coeffs = DeviceBuffer::<EF>::with_capacity(output_height);
            let mut new_w_moments = DeviceBuffer::<EF>::with_capacity(output_height);
            // SAFETY:
            // - `d_s_evals` has length 2
            // - `d_sumcheck_tmp` has at least required scratch length
            unsafe {
                whir_sumcheck_coeff_moments_round(
                    &f_coeffs,
                    &w_moments,
                    &mut d_s_evals,
                    &mut d_sumcheck_tmp,
                    f_height as u32,
                )
                .map_err(|error| WhirProverError::SumcheckMleRound {
                    error,
                    whir_round,
                    round,
                })?;
            }
            let s_evals = d_s_evals.to_host()?;
            for &eval in &s_evals {
                transcript.observe_ext(eval);
            }
            whir_sumcheck_polys.push(s_evals.try_into().unwrap());

            folding_pow_witnesses.push(
                transcript
                    .grind_gpu(whir_params.folding_pow_bits)
                    .map_err(WhirProverError::FoldingGrind)?,
            );
            let alpha = transcript.sample_ext();

            // Fold `f` and `w` in coefficient/moment form with respect to `alpha`.
            // SAFETY:
            // - input buffers have length `f_height`.
            // - output buffers have length `f_height / 2`.
            unsafe {
                whir_fold_coeffs_and_moments(
                    &f_coeffs,
                    &w_moments,
                    &mut new_f_coeffs,
                    &mut new_w_moments,
                    alpha,
                    f_height as u32,
                )
                .map_err(|error| WhirProverError::FoldMle {
                    error,
                    whir_round,
                    round,
                })?;
            }
            f_coeffs = new_f_coeffs;
            w_moments = new_w_moments;
        }
        // Define g^ = f^(alpha, \cdot) and send matrix commit of RS(g^)
        // `f_coeffs` is the coefficient form of f^(alpha, \cdot).
        let f_height = 1 << (m - k_whir);
        debug_assert!(f_coeffs.len() >= f_height);
        debug_assert_eq!(size_of::<EF>() / size_of::<F>(), D_EF);
        let g_coeffs = DeviceBuffer::<F>::with_capacity(f_height * D_EF);
        // SAFETY: we allocated `f_coeffs.len() * D_EF` space for `g_coeffs` to do a 1-to-D_EF
        // (1-to-4) split
        unsafe {
            split_ext_poly_to_base_col_major_matrix(
                &g_coeffs,
                &f_coeffs,
                f_height as u64,
                f_height as u32,
            )
            .map_err(|error| WhirProverError::SplitExtPoly { error, whir_round })?;
        }
        let (g_tree, z_0) = if !is_last_round {
            let codeword_height = 1 << (log_rs_domain_size - 1);
            // `g: \mathcal{L}^{(2)} \to \mathbb F`
            let g_rs = DeviceBuffer::<F>::with_capacity(D_EF * codeword_height);
            // SAFETY:
            // - g_coeffs is a single EF polynomial, treated as 4 F-polynomials of height
            //   2^{m-k_whir}
            // - We resize each F-poly to RS domain size 2^{log_rs_domain_size - 1}, which is
            //   equivalent to resizing the EF-polynomial
            unsafe {
                batch_expand_pad::<F>(
                    &g_rs,
                    &g_coeffs,
                    D_EF as u32,
                    codeword_height as u32,
                    f_height as u32,
                )
                .map_err(|error| WhirProverError::BatchExpandPad { error, whir_round })?;

                batch_ntt(
                    &g_rs,
                    (log_rs_domain_size - 1) as u32,
                    0u32,
                    D_EF as u32,
                    true,
                    false,
                );
            }

            let g_tree = MerkleTreeGpu::<F, Digest>::new(
                DeviceMatrix::new(Arc::new(g_rs), codeword_height, D_EF),
                1 << k_whir,
                true,
            )
            .map_err(WhirProverError::MerkleTree)?;
            let g_commit = g_tree.root();
            transcript.observe_commit(g_commit);
            codeword_commits.push(g_commit);

            let z_0 = transcript.sample_ext();
            // SAFETY:
            // - `g_coeffs` is coefficient form of `\hat{g}`, which is degree `2^{m-k_whir}`.
            // - `g_coeffs` is F-column major matrix.
            let g_opened_value = unsafe {
                eval_poly_ext_at_point_from_base(&g_coeffs, 1 << (m - k_whir), z_0)
                    .map_err(|error| WhirProverError::EvalPolyAtPoint { error, whir_round })?
            };
            transcript.observe_ext(g_opened_value);
            ood_values.push(g_opened_value);

            (Some(g_tree), Some(z_0))
        } else {
            // Observe the final poly
            debug_assert_eq!(log_final_poly_len, m - k_whir);
            let final_poly_len = 1 << log_final_poly_len;
            let base_coeffs = g_coeffs.to_host()?;
            debug_assert_eq!(base_coeffs.len(), D_EF * final_poly_len);
            let mut coeffs = Vec::with_capacity(final_poly_len);
            for i in 0..final_poly_len {
                let coeff = EF::from_basis_coefficients_fn(|j| base_coeffs[j * final_poly_len + i]);
                transcript.observe_ext(coeff);
                coeffs.push(coeff);
            }
            final_poly = Some(coeffs);
            (None, None)
        };

        // omega is generator of RS domain `\mathcal{L}^{(2^k)}`
        let omega = F::two_adic_generator(log_rs_domain_size - k_whir);
        let num_queries = round_params.num_queries;
        let mut query_indices = Vec::with_capacity(num_queries);
        query_phase_pow_witnesses.push(
            transcript
                .grind_gpu(whir_params.query_phase_pow_bits)
                .map_err(WhirProverError::QueryPhaseGrind)?,
        );
        // Sample query indices first
        for _ in 0..num_queries {
            // This is the index of the leaf in the Merkle tree
            let index = transcript.sample_bits(log_rs_domain_size - k_whir);
            query_indices.push(index as usize);
        }
        if !is_last_round {
            codeword_opened_values.push(vec![]);
            codeword_merkle_proofs.push(vec![]);
        }
        if whir_round == 0 {
            // Vector to hold owned copies of backing matrices that are regenerated in the case they
            // were not cached
            let mut backing_mats_owned = vec![None; stacked_per_commit.len()];
            let mut backing_matrices = Vec::with_capacity(stacked_per_commit.len());
            let mut trees = Vec::with_capacity(stacked_per_commit.len());
            for (d, backing_mat_owned) in zip(&mut stacked_per_commit, &mut backing_mats_owned) {
                trees.push(&d.inner.tree);
                if let Some(matrix) = d.inner.tree.backing_matrix.as_ref() {
                    backing_matrices.push(matrix);
                } else {
                    let layout = d.layout();
                    let traces = d.traces.iter().collect_vec();
                    debug_assert!(!traces.is_empty());
                    let backing_matrix =
                        rs_code_matrix(log_blowup, layout, &traces, &d.inner.matrix)
                            .map_err(WhirProverError::RsCodeMatrix)?;
                    d.traces.clear();
                    *backing_mat_owned = Some(backing_matrix);
                    backing_matrices.push(backing_mat_owned.as_ref().unwrap());
                }
            }
            // Get merkle proofs for in-domain samples necessary to evaluate Fold(f, \vec
            // \alpha)(z_i)
            initial_round_merkle_proofs =
                MerkleTreeGpu::batch_query_merkle_proofs(&trees, &query_indices)
                    .map_err(WhirProverError::MerkleTree)?;

            let query_stride = trees[0].query_stride();
            debug_assert!(
                trees.iter().all(|tree| tree.query_stride() == query_stride),
                "Merkle trees don't have same layer size"
            );
            let num_rows_per_query = trees[0].rows_per_query;
            debug_assert!(
                trees
                    .iter()
                    .all(|tree| tree.rows_per_query == num_rows_per_query),
                "Merkle trees don't have same rows_per_query"
            );

            initial_round_opened_rows = MerkleTreeGpu::batch_open_rows(
                &backing_matrices,
                &query_indices,
                query_stride,
                num_rows_per_query,
            )
            .map_err(WhirProverError::MerkleTree)?
            .into_iter()
            .map(|rows_per_commit| {
                rows_per_commit
                    .into_iter()
                    .map(|rows| {
                        let width = rows.len() / num_rows_per_query;
                        rows.chunks_exact(width).map(|row| row.to_vec()).collect()
                    })
                    .collect()
            })
            .collect();
            debug_assert_eq!(
                Arc::strong_count(&stacked_per_commit[0].inner),
                1,
                "common_main_pcs_data should be owned"
            );
            stacked_per_commit.clear(); // this drops common_main_pcs_data
            mem.tracing_info("after_initial_whir_round");
        } else {
            let tree: &MerkleTreeGpu<F, Digest> = rs_tree.as_ref().unwrap();
            codeword_merkle_proofs[whir_round - 1] =
                MerkleTreeGpu::batch_query_merkle_proofs(&[tree], &query_indices)
                    .map_err(WhirProverError::MerkleTree)?
                    .pop()
                    .expect("exactly 1 tree");
            codeword_opened_values[whir_round - 1] = MerkleTreeGpu::batch_open_rows(
                &[tree.backing_matrix.as_ref().unwrap()],
                &query_indices,
                tree.query_stride(),
                tree.rows_per_query,
            )
            .map_err(WhirProverError::MerkleTree)?
            .pop()
            .unwrap()
            .into_iter()
            .map(EF::reconstitute_from_base)
            .collect();
        }
        rs_tree = g_tree;

        // We still sample on the last round to match the verifier, who uses a
        // final gamma to unify some logic. But we do not need to update
        // `w_moments`.
        let gamma = transcript.sample_ext();

        if !is_last_round {
            // Update moments of \hat{w}:
            // M(T) += gamma * z0^T + sum_i gamma^{i+1} * z_i^T.
            let log_height = (m - k_whir) as u32;
            let z0 = z_0.unwrap();
            let z_points = query_indices
                .iter()
                .map(|&index| omega.exp_u64(index as u64))
                .collect_vec();

            let mut z0_pows2 = Vec::with_capacity(log_height as usize);
            let mut z0_pow = z0;
            for _ in 0..log_height {
                z0_pows2.push(z0_pow);
                z0_pow = z0_pow.square();
            }

            let mut z_pows2 = Vec::with_capacity(num_queries * log_height as usize);
            for z in &z_points {
                let mut z_pow = *z;
                for _ in 0..log_height {
                    z_pows2.push(z_pow);
                    z_pow = z_pow.square();
                }
            }

            let d_z0_pows2 = z0_pows2.to_device()?;
            let d_z_pows2 = z_pows2.to_device()?;
            unsafe {
                w_moments_accumulate(
                    &mut w_moments,
                    &d_z0_pows2,
                    &d_z_pows2,
                    gamma,
                    num_queries as u32,
                    log_height,
                )
                .map_err(|error| WhirProverError::WMomentsAccumulate { error, whir_round })?;
            }
        }

        m -= k_whir;
        log_rs_domain_size -= 1;
    }

    mem.emit_metrics();
    Ok(WhirProof {
        mu_pow_witness,
        whir_sumcheck_polys,
        codeword_commits,
        ood_values,
        folding_pow_witnesses,
        query_phase_pow_witnesses,
        initial_round_opened_rows,
        initial_round_merkle_proofs,
        codeword_opened_values,
        codeword_merkle_proofs,
        final_poly: final_poly.unwrap(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use openvm_stark_backend::{
        keygen::types::MultiStarkProvingKeyV2,
        poly_common::Squarable,
        poseidon2::sponge::DuplexSponge,
        prover::{
            poly::Ple, stacked_pcs::stacked_commit, ColMajorMatrix, CpuBackendV2,
            DeviceDataTransporterV2, ProvingContextV2,
        },
        test_utils::{FibFixture, TestFixture},
        verifier::whir::{verify_whir, VerifyWhirError},
        BabyBearPoseidon2CpuEngineV2, SystemParams, WhirConfig, WhirParams, EF, F,
    };
    use openvm_stark_sdk::config::{
        log_up_params::log_up_security_params_baby_bear_100_bits, setup_tracing_with_log_level,
    };
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use test_case::test_case;
    use tracing::Level;

    use crate::{
        sponge::DuplexSpongeGpu, stacked_reduction::StackedPcsData2, whir::prove_whir_opening_gpu,
        GpuDeviceV2,
    };

    fn generate_random_z(params: &SystemParams, rng: &mut StdRng) -> (Vec<EF>, Vec<EF>) {
        let z_prism: Vec<_> = (0..params.n_stack + 1)
            .map(|_| EF::from_u64(rng.random()))
            .collect();

        let z_cube = {
            let z_cube = z_prism[0]
                .exp_powers_of_2()
                .take(params.l_skip)
                .chain(z_prism[1..].iter().copied())
                .collect_vec();
            debug_assert_eq!(z_cube.len(), params.n_stack + params.l_skip);
            z_cube
        };

        (z_prism, z_cube)
    }

    fn stacking_openings_for_matrix(
        params: &SystemParams,
        z_prism: &[EF],
        matrix: &ColMajorMatrix<F>,
    ) -> Vec<EF> {
        matrix
            .columns()
            .map(|col| {
                Ple::from_evaluations(params.l_skip, col).eval_at_point(
                    params.l_skip,
                    z_prism[0],
                    &z_prism[1..],
                )
            })
            .collect()
    }

    fn run_whir_test(
        params: SystemParams,
        pk: MultiStarkProvingKeyV2,
        ctx: ProvingContextV2<CpuBackendV2>,
    ) -> Result<(), VerifyWhirError> {
        let device = GpuDeviceV2::new(params.clone());
        let mut rng = StdRng::seed_from_u64(0);
        let (z_prism, z_cube) = generate_random_z(&params, &mut rng);

        let common_main_traces = ctx
            .common_main_traces()
            .map(|(_, trace)| trace)
            .collect_vec();
        let (common_main_commit, common_main_pcs_data) = stacked_commit(
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir(),
            &common_main_traces,
        );
        let d_common_main_traces = common_main_traces
            .iter()
            .map(|t| device.transport_matrix_to_device(t))
            .collect_vec();
        let d_common_main_pcs_data = device.transport_pcs_data_to_device(&common_main_pcs_data);

        let mut stacking_openings = vec![stacking_openings_for_matrix(
            &params,
            &z_prism,
            &common_main_pcs_data.matrix,
        )];
        let mut commits = vec![common_main_commit];
        let mut stacked_per_commit = vec![StackedPcsData2 {
            inner: Arc::new(d_common_main_pcs_data),
            traces: d_common_main_traces,
        }];
        for (air_id, air_ctx) in ctx.per_trace {
            for data in pk.per_air[air_id]
                .preprocessed_data
                .iter()
                .chain(air_ctx.cached_mains.iter().map(|cd| &cd.data))
            {
                let trace = device.transport_matrix_to_device(&data.mat_view(0).to_matrix());
                commits.push(data.commit());
                stacking_openings.push(stacking_openings_for_matrix(
                    &params,
                    &z_prism,
                    &data.matrix,
                ));
                let d_data = device.transport_pcs_data_to_device(data);
                stacked_per_commit.push(StackedPcsData2 {
                    inner: Arc::new(d_data),
                    traces: vec![trace],
                });
            }
        }

        let mut prover_sponge = DuplexSpongeGpu::default();

        let proof =
            prove_whir_opening_gpu(&params, &mut prover_sponge, stacked_per_commit, &z_cube)
                .unwrap();

        let mut verifier_sponge = DuplexSponge::default();
        verify_whir(
            &mut verifier_sponge,
            &params,
            &proof,
            &stacking_openings,
            &commits,
            &z_cube,
        )
    }

    fn run_whir_fib_test(params: SystemParams) -> Result<(), VerifyWhirError> {
        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params.clone());
        let fib = FibFixture::new(0, 1, 1 << params.log_stacked_height());
        let (pk, _vk) = fib.keygen(&engine);
        let ctx = fib.generate_proving_ctx();
        run_whir_test(params, pk, ctx)
    }

    fn whir_test_params(k_whir: usize, log_final_poly_len: usize) -> WhirParams {
        WhirParams {
            k: k_whir,
            log_final_poly_len,
            query_phase_pow_bits: 2,
        }
    }

    #[test_case(0, 1, 1, 0)]
    #[test_case(2, 1, 1, 2)]
    #[test_case(2, 1, 2, 0)]
    #[test_case(2, 1, 3, 1)]
    #[test_case(2, 1, 4, 0)]
    #[test_case(2, 2, 4, 0)]
    fn test_whir_single_fib(
        n_stack: usize,
        log_blowup: usize,
        k_whir: usize,
        log_final_poly_len: usize,
    ) -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let l_skip = 2;
        let whir = WhirConfig::new(
            log_blowup,
            l_skip + n_stack,
            whir_test_params(k_whir, log_final_poly_len),
            10,
        );
        let params = SystemParams {
            l_skip: 2,
            n_stack,
            log_blowup,
            whir,
            logup: log_up_security_params_baby_bear_100_bits(),
            max_constraint_degree: 3,
        };
        run_whir_fib_test(params)
    }

    // TODO: test multiple commitments
}
