use std::{iter::zip, sync::Arc};

use itertools::Itertools;
use openvm_metal_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::MetalBuffer,
};
use openvm_stark_backend::{
    proof::{MerkleProof, WhirProof},
    prover::MatrixDimensions,
    FiatShamirTranscript, SystemParams,
};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    base::MetalMatrix,
    merkle_tree::MerkleTreeMetal,
    metal::{
        batch_ntt_small::batch_ntt_small,
        matrix::{batch_expand_pad, split_ext_to_base_col_major_matrix},
        mle_interpolate::mle_interpolate_stage_ext,
        poly::{eval_poly_ext_at_point_from_base, transpose_fp_to_fpext_vec},
        whir::{
            w_moments_accumulate, whir_algebraic_batch_trace, whir_fold_coeffs_and_moments,
            whir_sumcheck_coeff_moments_required_temp_buffer_size,
            whir_sumcheck_coeff_moments_round,
        },
    },
    ntt::batch_ntt,
    poly::evals_eq_hypercube,
    prelude::{Digest, D_EF, EF, F, SC},
    sponge::DuplexSpongeMetal,
    stacked_pcs::rs_code_matrix,
    stacked_reduction::StackedPcsData2,
    WhirProverError,
};

#[instrument(
    name = "prover.openings.whir",
    level = "info",
    skip_all,
    fields(phase = "prover")
)]
pub fn prove_whir_opening_metal(
    params: &SystemParams,
    transcript: &mut DuplexSpongeMetal,
    mut stacked_per_commit: Vec<StackedPcsData2>,
    u: &[EF],
) -> Result<WhirProof<SC>, WhirProverError> {
    let l_skip = params.l_skip;
    let log_blowup = params.log_blowup;
    let k_whir = params.k_whir();
    let whir_params = params.whir();
    let log_final_poly_len = params.log_final_poly_len();

    let height = stacked_per_commit[0].inner.layout().height();
    debug_assert!(stacked_per_commit
        .iter()
        .all(|d| d.inner.layout().height() == height));
    let mut m = log2_strict_usize(height);
    assert_eq!(m, u.len());
    debug_assert!(m >= l_skip);

    // Proof-of-work grinding before mu batching challenge.
    // This amplifies soundness of the initial batching step.
    let mu_pow_witness = transcript
        .grind_gpu(whir_params.mu_pow_bits)
        .map_err(WhirProverError::MuGrind)?;
    // Sample randomness for algebraic batching.
    // We batch the codewords for \hat{q}_j together _before_ applying WHIR.
    let mu = transcript.sample_ext();
    let num_commits = stacked_per_commit.len();

    // The coefficient table of `\hat{f}` in the current WHIR round (MLE coefficient form).
    let mut f_ple_evals = MetalBuffer::<F>::with_capacity(height * D_EF);
    // We algebraically batch all matrices together so we only need to interpolate one column vector
    {
        f_ple_evals.fill_zero();
        let mut total_stacked_width = 0u32;
        for stacked in &stacked_per_commit {
            let layout = stacked.inner.layout();
            total_stacked_width += layout.width() as u32;
        }
        let mu_powers = mu.powers().take(total_stacked_width as usize).collect_vec();
        let d_mu_powers = mu_powers.to_device();
        let mut width_acc = 0u32;
        let stacked_height_u32 = height as u32;
        let skip_domain_u32 = 1u32 << l_skip;

        // SAFETY:
        // - `mu_powers` has length equal to sum of committed stacked widths.
        // - each trace buffer is valid for `trace.height() * trace.width()` base elements.
        // - output length is exactly `stacked_height * D_EF`.
        unsafe {
            for stacked in &stacked_per_commit {
                let layout = stacked.inner.layout();
                for (trace, &idx) in zip(&stacked.traces, &layout.mat_starts) {
                    let (_, _, s) = layout.sorted_cols[idx];
                    whir_algebraic_batch_trace(
                        &mut f_ple_evals,
                        trace.buffer(),
                        &d_mu_powers,
                        stacked_height_u32,
                        trace.height() as u32,
                        trace.width() as u32,
                        s.row_idx as u64,
                        width_acc + s.col_idx as u32,
                        skip_domain_u32,
                    )
                    .map_err(WhirProverError::AlgebraicBatch)?;
                }
                width_acc += layout.width() as u32;
            }
        }
        #[cfg(debug_assertions)]
        {
            let out = f_ple_evals.to_host();
            let stacked_height = f_ple_evals.len() / D_EF;
            debug_assert_eq!(stacked_height, height);
            let skip_domain = 1usize << l_skip;
            for row in 0..stacked_height {
                let mut expected = EF::ZERO;
                let mut width_acc = 0usize;
                for stacked in &stacked_per_commit {
                    let layout = stacked.inner.layout();
                    for (trace, &idx) in zip(&stacked.traces, &layout.mat_starts) {
                        let (_, _, s) = layout.sorted_cols[idx];
                        let h = trace.height();
                        let lifted_height = h.max(skip_domain);
                        let w = trace.width();
                        let row_start = s.row_idx;
                        let mu_idx_start = width_acc + s.col_idx;
                        let stride = (skip_domain / h).max(1);
                        let stacked_end = row_start + lifted_height * w;
                        if row >= stacked_end {
                            continue;
                        }
                        let offset_start = if row_start <= row { 0 } else { 1 };
                        let mut row_offset = offset_start * stacked_height + row;
                        while row_offset < stacked_end {
                            let offset = (row_offset - row) / stacked_height;
                            let tmp = row_offset - row_start;
                            let trace_col = tmp / lifted_height;
                            let strided_row = tmp % lifted_height;
                            let trace_val = if strided_row % stride == 0 {
                                let trace_row = strided_row / stride;
                                unsafe { *trace.buffer().as_ptr().add(trace_col * h + trace_row) }
                            } else {
                                F::ZERO
                            };
                            expected += mu_powers[mu_idx_start + offset] * trace_val;
                            row_offset += stacked_height;
                        }
                    }
                    width_acc += layout.width();
                }
                let coeffs = expected.as_basis_coefficients_slice();
                for i in 0..D_EF {
                    debug_assert_eq!(out[i * stacked_height + row], coeffs[i]);
                }
            }
        }
        for stacked in &mut stacked_per_commit {
            // drop traces to free device memory if RS codewords matrix exists
            if stacked.inner.tree.backing_matrix.is_some() {
                stacked.traces.clear();
            }
        }
    } // common_main_pcs_data.matrix has now been freed

    #[cfg(debug_assertions)]
    let f_ple_evals_before_transform = f_ple_evals.to_host();

    // Compute \hat{f} coefficients:
    //
    // Step 1: iDFT per chunk of size 2^l_skip. After this, each chunk holds univariate
    // coefficients c_z(x) of f(Z, x) for a fixed boolean assignment x in H_{m - l_skip}.
    unsafe {
        let num_poly = f_ple_evals.len() >> l_skip;
        batch_ntt_small(&mut f_ple_evals, l_skip, num_poly, true)
            .map_err(WhirProverError::CustomBatchIntt)?;
    }
    let mut f_coeffs = MetalBuffer::<EF>::with_capacity(height);
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
    #[cfg(debug_assertions)]
    {
        let coeffs = f_coeffs.to_host();
        let expected_lanes = (0..D_EF)
            .map(|i| {
                let evals_i = &f_ple_evals_before_transform[i * height..(i + 1) * height];
                openvm_stark_backend::prover::poly::eval_to_coeff_rs_message(l_skip, evals_i)
            })
            .collect_vec();
        for row in 0..height {
            let expected = EF::from_basis_coefficients_fn(|j| expected_lanes[j][row]);
            debug_assert_eq!(coeffs[row], expected);
        }
    }

    debug_assert_eq!((m - log_final_poly_len) % k_whir, 0);
    let num_whir_rounds = (m - log_final_poly_len) / k_whir;
    assert!(num_whir_rounds > 0);

    // We maintain moments of \hat{w}:
    // M[T] = sum_{x superset T} \hat{w}(x).
    // For initial \hat{w} = mobius_eq(u, -), these moments are exactly eq(u, -).
    let mut w_moments = MetalBuffer::<EF>::with_capacity(1 << m);
    unsafe {
        evals_eq_hypercube(&mut w_moments, u).map_err(WhirProverError::EvalEq)?;
    }
    #[cfg(debug_assertions)]
    {
        let w_host = w_moments.to_host();
        for (idx, got) in w_host.iter().copied().enumerate() {
            let mut expected = EF::ONE;
            for (bit, &u_bit) in u.iter().enumerate() {
                let b = ((idx >> bit) & 1) == 1;
                expected *= if b { u_bit } else { EF::ONE - u_bit };
            }
            debug_assert_eq!(got, expected);
        }
    }

    let mut whir_sumcheck_polys: Vec<[EF; 2]> = vec![];
    let mut codeword_commits = vec![];
    let mut ood_values = vec![];
    // per commitment, per whir query, per column
    let mut initial_round_opened_rows: Vec<Vec<Vec<Vec<F>>>> = vec![vec![]; num_commits];
    let mut initial_round_merkle_proofs: Vec<Vec<MerkleProof<Digest>>> = vec![];
    let mut codeword_opened_values: Vec<Vec<Vec<EF>>> = vec![];
    let mut codeword_merkle_proofs: Vec<Vec<MerkleProof<Digest>>> = vec![];
    let mut folding_pow_witnesses = vec![];
    let mut query_phase_pow_witnesses = vec![];
    let mut rs_tree = None;
    let mut log_rs_domain_size = m + log_blowup;
    let mut final_poly = None;

    let mut d_s_evals = MetalBuffer::<EF>::with_capacity(2);
    let mut d_sumcheck_tmp = MetalBuffer::<EF>::with_capacity(1);

    // We will drop `stacked_per_commit` and hence `common_main_pcs_data` after whir round 0.
    for (whir_round, round_params) in whir_params.rounds.iter().enumerate() {
        let is_last_round = whir_round == num_whir_rounds - 1;
        let mut alphas_round = Vec::with_capacity(k_whir);
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
                whir_sumcheck_coeff_moments_required_temp_buffer_size(f_height as u32);
            if d_sumcheck_tmp.len() < tmp_buffer_capacity as usize {
                d_sumcheck_tmp = MetalBuffer::<EF>::with_capacity(tmp_buffer_capacity as usize);
            }
            let mut new_f_coeffs = MetalBuffer::<EF>::with_capacity(output_height);
            let mut new_w_moments = MetalBuffer::<EF>::with_capacity(output_height);
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
            let s_evals = d_s_evals.to_host();
            #[cfg(debug_assertions)]
            {
                let f_host = f_coeffs.to_host();
                let w_host = w_moments.to_host();
                let mut expected = [EF::ZERO; 2];
                let three = EF::ONE + EF::ONE + EF::ONE;
                for y in 0..(f_height / 2) {
                    let idx0 = y << 1;
                    let idx1 = idx0 + 1;
                    let c0 = f_host[idx0];
                    let c1 = f_host[idx1];
                    let m0 = w_host[idx0];
                    let m1 = w_host[idx1];
                    let f1 = c0 + c1;
                    let f2 = c0 + c1 + c1;
                    let m2 = m1 * three - m0;
                    expected[0] += f1 * m1;
                    expected[1] += f2 * m2;
                }
                debug_assert_eq!(s_evals[0], expected[0]);
                debug_assert_eq!(s_evals[1], expected[1]);
            }
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
            alphas_round.push(alpha);

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
            #[cfg(debug_assertions)]
            {
                let f_in = f_coeffs.to_host();
                let w_in = w_moments.to_host();
                let f_out = new_f_coeffs.to_host();
                let w_out = new_w_moments.to_host();
                let one = EF::ONE;
                let one_minus_alpha = one - alpha;
                let two_alpha_minus_one = alpha + alpha - one;
                for y in 0..output_height {
                    let idx0 = y << 1;
                    let idx1 = idx0 + 1;
                    let c0 = f_in[idx0];
                    let c1 = f_in[idx1];
                    let m0 = w_in[idx0];
                    let m1 = w_in[idx1];
                    let expected_f = c0 + alpha * c1;
                    let expected_w = one_minus_alpha * m0 + two_alpha_minus_one * m1;
                    debug_assert_eq!(f_out[y], expected_f);
                    debug_assert_eq!(w_out[y], expected_w);
                }
            }
            f_coeffs = new_f_coeffs;
            w_moments = new_w_moments;
        }
        // Define g^ = f^(alpha, \cdot) and send matrix commit of RS(g^)
        // `f_coeffs` is the coefficient form of f^(alpha, \cdot).
        let f_height = 1 << (m - k_whir);
        debug_assert!(f_coeffs.len() >= f_height);
        debug_assert_eq!(size_of::<EF>() / size_of::<F>(), D_EF);
        let mut g_coeffs = MetalBuffer::<F>::with_capacity(f_height * D_EF);
        // SAFETY: we allocated `f_coeffs.len() * D_EF` space for `g_coeffs` to do a 1-to-D_EF
        // (1-to-4) split
        unsafe {
            split_ext_to_base_col_major_matrix(
                &mut g_coeffs,
                &f_coeffs,
                f_height as u64,
                f_height as u32,
            )
            .map_err(|error| WhirProverError::SplitExtPoly { error, whir_round })?;
        }
        #[cfg(debug_assertions)]
        {
            let f_host = f_coeffs.to_host();
            let g_host = g_coeffs.to_host();
            for (row, coeff) in f_host.iter().enumerate().take(f_height) {
                let basis = coeff.as_basis_coefficients_slice();
                for i in 0..D_EF {
                    debug_assert_eq!(g_host[i * f_height + row], basis[i]);
                }
            }
        }
        let (g_tree, z_0) = if !is_last_round {
            let codeword_height = 1 << (log_rs_domain_size - 1);
            // `g: \mathcal{L}^{(2)} \to \mathbb F`
            let g_rs = MetalBuffer::<F>::with_capacity(D_EF * codeword_height);
            // SAFETY:
            // - g_coeffs is a single EF polynomial, treated as 4 F-polynomials of height
            //   2^{m-k_whir}
            // - We resize each F-poly to RS domain size 2^{log_rs_domain_size - 1}, which is
            //   equivalent to resizing the EF-polynomial
            unsafe {
                batch_expand_pad(
                    &g_rs,
                    0,
                    &g_coeffs,
                    0,
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

            let g_tree = MerkleTreeMetal::<F, Digest>::new(
                MetalMatrix::new(Arc::new(g_rs), codeword_height, D_EF),
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
            #[cfg(debug_assertions)]
            {
                let base_coeffs = g_coeffs.to_host();
                let mut expected = EF::ZERO;
                for idx in (0..f_height).rev() {
                    let coeff = EF::from_basis_coefficients_fn(|j| base_coeffs[j * f_height + idx]);
                    expected = expected * z_0 + coeff;
                }
                debug_assert_eq!(g_opened_value, expected);
            }
            transcript.observe_ext(g_opened_value);
            ood_values.push(g_opened_value);

            (Some(g_tree), Some(z_0))
        } else {
            // Observe the final poly
            debug_assert_eq!(log_final_poly_len, m - k_whir);
            let final_poly_len = 1 << log_final_poly_len;
            let base_coeffs = g_coeffs.to_host();
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
                    let layout = d.inner.layout();
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
                MerkleTreeMetal::batch_query_merkle_proofs(&trees, &query_indices)
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

            initial_round_opened_rows = MerkleTreeMetal::batch_open_rows(
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
        } else {
            let tree: &MerkleTreeMetal<F, Digest> = rs_tree.as_ref().unwrap();
            codeword_merkle_proofs[whir_round - 1] =
                MerkleTreeMetal::batch_query_merkle_proofs(&[tree], &query_indices)
                    .map_err(WhirProverError::MerkleTree)?
                    .pop()
                    .expect("exactly 1 tree");
            codeword_opened_values[whir_round - 1] = MerkleTreeMetal::batch_open_rows(
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

            let d_z0_pows2 = z0_pows2.to_device();
            let d_z_pows2 = z_pows2.to_device();
            #[cfg(debug_assertions)]
            let old_w_moments = w_moments.to_host();
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
            #[cfg(debug_assertions)]
            {
                let new_w_moments = w_moments.to_host();
                for exponent in 0..(1usize << log_height) {
                    let mut z0_term = EF::ONE;
                    for bit in 0..(log_height as usize) {
                        if (exponent >> bit) & 1 == 1 {
                            z0_term *= z0_pows2[bit];
                        }
                    }
                    let mut acc = gamma * z0_term;
                    let mut gamma_pow = gamma;
                    for i in 0..num_queries {
                        gamma_pow *= gamma;
                        let base = i * log_height as usize;
                        let mut zi_pow = F::ONE;
                        for bit in 0..(log_height as usize) {
                            if (exponent >> bit) & 1 == 1 {
                                zi_pow *= z_pows2[base + bit];
                            }
                        }
                        acc += gamma_pow * EF::from(zi_pow);
                    }
                    debug_assert_eq!(new_w_moments[exponent], old_w_moments[exponent] + acc);
                }
            }
        }

        m -= k_whir;
        log_rs_domain_size -= 1;
    }

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
