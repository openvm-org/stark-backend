//! Batch sumcheck for ZeroCheck constraints and sumcheck for LogUp input layer MLEs

use std::{
    cmp::max,
    iter::zip,
    mem::{align_of, size_of, ManuallyDrop},
};

use itertools::Itertools;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{debug, info_span, instrument};

use crate::{
    calculate_n_logup,
    dft::Radix2BowersSerial,
    poly_common::{eq_sharp_uni_poly, eq_uni_poly, UnivariatePoly},
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, GkrProof, TensorLogupProof},
    prover::{
        fractional_sumcheck_gkr::{fractional_sumcheck, Frac},
        stacked_pcs::{stacked_commit, StackedLayout},
        sumcheck::sumcheck_round0_deg,
        ColMajorMatrix, CpuBackendV2, DeviceMultiStarkProvingKeyV2, MatrixView, ProvingContextV2,
    },
    EF, F,
};

mod cpu;
mod evaluator;
pub mod fractional_sumcheck_gkr;
mod single;

pub use cpu::LogupZerocheckCpu;
pub use single::*;

fn eval_mle_from_hypercube(evals: &[EF], point: &[EF]) -> EF {
    if evals.is_empty() {
        return EF::ZERO;
    }
    if point.is_empty() {
        return evals[0];
    }
    let mut cur = evals.to_vec();
    for &r in point {
        let next_len = cur.len() >> 1;
        for i in 0..next_len {
            let x0 = cur[i << 1];
            let x1 = cur[(i << 1) + 1];
            cur[i] = x0 + (x1 - x0) * r;
        }
        cur.truncate(next_len);
    }
    cur[0]
}

fn tensor_running_sum(evals: &[Frac<EF>], n_grid: usize, b_block: usize) -> Vec<EF> {
    let grid_size = 1 << n_grid;
    let block_size = 1 << b_block;
    if evals.is_empty() {
        return vec![EF::ZERO; grid_size];
    }
    debug_assert_eq!(evals.len(), grid_size * block_size);
    let mut running = Vec::with_capacity(grid_size);
    let mut acc = EF::ZERO;
    for x in 0..grid_size {
        let start = x * block_size;
        let end = start + block_size;
        let partial = evals[start..end]
            .iter()
            .map(|frac| frac.p * frac.q.inverse())
            .sum::<EF>();
        acc += partial;
        running.push(acc);
    }
    running
}

/// # Safety
/// Assumes extension element memory layout is `[F; EF::DIMENSION]`.
unsafe fn ext_col_major_to_base(ext_matrix: ColMajorMatrix<EF>) -> ColMajorMatrix<F> {
    let dim = <EF as BasedVectorSpace<F>>::DIMENSION;
    debug_assert_eq!(align_of::<EF>(), align_of::<F>());
    debug_assert_eq!(size_of::<EF>(), size_of::<F>() * dim);

    let width = ext_matrix.width() * dim;
    let mut values = ManuallyDrop::new(ext_matrix.values);
    let len = values.len() * dim;
    let cap = values.capacity() * dim;
    let ptr = values.as_mut_ptr();
    let base_values = Vec::from_raw_parts(ptr as *mut F, len, cap);
    ColMajorMatrix::new(base_values, width)
}

#[instrument(level = "info", skip_all)]
pub fn prove_zerocheck_and_logup<TS>(
    transcript: &mut TS,
    mpk: &DeviceMultiStarkProvingKeyV2<CpuBackendV2>,
    ctx: &ProvingContextV2<CpuBackendV2>,
) -> (GkrProof, BatchConstraintProof, Vec<EF>)
where
    TS: FiatShamirTranscript,
{
    let l_skip = mpk.params.l_skip;
    let constraint_degree = mpk.max_constraint_degree;
    let num_traces = ctx.per_trace.len();

    // Traces are sorted
    let n_max = log2_strict_usize(ctx.per_trace[0].1.common_main.height()).saturating_sub(l_skip);
    // Gather interactions metadata, including interactions stacked layout which depends on trace
    // heights
    let mut total_interactions = 0u64;
    let interactions_meta: Vec<_> = ctx
        .per_trace
        .iter()
        .map(|(air_idx, air_ctx)| {
            let pk = &mpk.per_air[*air_idx];

            let num_interactions = pk.vk.symbolic_constraints.interactions.len();
            let height = air_ctx.common_main.height();
            let log_height = log2_strict_usize(height);
            let log_lifted_height = log_height.max(l_skip);
            total_interactions += (num_interactions as u64) << log_lifted_height;
            (num_interactions, log_lifted_height)
        })
        .collect();
    // Implicitly, the width of this stacking should be 1
    let n_logup = calculate_n_logup(l_skip, total_interactions);
    debug!(%n_logup);
    // There's no stride threshold for `interactions_layout` because there's no univariate skip for
    // GKR
    let interactions_layout = StackedLayout::new(0, l_skip + n_logup, interactions_meta);

    // Grind to increase soundness of random sampling for LogUp
    let logup_pow_witness = transcript.grind(mpk.params.logup.pow_bits);
    let alpha_logup = transcript.sample_ext();
    let beta_logup = transcript.sample_ext();
    debug!(%alpha_logup, %beta_logup);

    let mut prover = LogupZerocheckCpu::new(
        mpk,
        ctx,
        n_logup,
        interactions_layout,
        alpha_logup,
        beta_logup,
    );
    // GKR
    // Compute logup input layer: these are the evaluations of \hat{p}, \hat{q} on the hypercube
    // `H_{l_skip + n_logup}`
    let has_interactions = !prover.interactions_layout.sorted_cols.is_empty();
    let gkr_input_evals = if !has_interactions {
        vec![]
    } else {
        // Per trace, a row major matrix of interaction evaluations
        // NOTE: these are the evaluations _without_ lifting
        // PERF[jpw]: we should write directly to the stacked `evals` in memory below
        let unstacked_interaction_evals = prover
            .eval_helpers
            .par_iter()
            .enumerate()
            .map(|(trace_idx, helper)| {
                let trace_ctx = &ctx.per_trace[trace_idx].1;
                let mats = helper.view_mats(trace_ctx);
                let height = trace_ctx.common_main.height();
                (0..height)
                    .into_par_iter()
                    .map(|i| {
                        let mut row_parts = Vec::with_capacity(mats.len() + 1);
                        let is_first = F::from_bool(i == 0);
                        let is_transition = F::from_bool(i != height - 1);
                        let is_last = F::from_bool(i == height - 1);
                        let sels = vec![is_first, is_transition, is_last];
                        row_parts.push(sels);
                        for (mat, is_rot) in &mats {
                            let offset = usize::from(*is_rot);
                            row_parts.push(
                                // SAFETY: %height ensures we never go out of bounds
                                (0..mat.width())
                                    .map(|j| unsafe {
                                        *mat.get_unchecked((i + offset) % height, j)
                                    })
                                    .collect_vec(),
                            );
                        }
                        helper.eval_interactions(&row_parts, &prover.beta_pows)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut evals = vec![Frac::default(); 1 << (l_skip + n_logup)];
        for (trace_idx, interaction_idx, s) in
            prover.interactions_layout.sorted_cols.iter().copied()
        {
            let pq_evals = &unstacked_interaction_evals[trace_idx];
            let height = pq_evals.len();
            debug_assert_eq!(s.col_idx, 0);
            // the interactions layout has internal striding threshold=0
            debug_assert_eq!(1 << s.log_height(), s.len(0));
            debug_assert_eq!(s.len(0) % height, 0);
            let norm_factor_denom = s.len(0) / height;
            let norm_factor = F::from_usize(norm_factor_denom).inverse();
            // We need to fill `evals` with the logup evaluations on the lifted trace, which is
            // the same as cyclic repeating of the unlifted evaluations
            evals[s.row_idx..s.row_idx + s.len(0)]
                .chunks_exact_mut(height)
                .for_each(|evals| {
                    evals
                        .par_iter_mut()
                        .zip(pq_evals)
                        .for_each(|(pq_eval, evals_at_z)| {
                            let (mut numer, denom) = evals_at_z[interaction_idx];
                            numer *= norm_factor;
                            *pq_eval = Frac::new(numer.into(), denom);
                        });
                });
        }
        // Prevent division by zero:
        evals.par_iter_mut().for_each(|frac| frac.q += alpha_logup);
        evals
    };

    let tensor_logup = if mpk.params.is_tensor_logup() {
        let total_rounds = l_skip + n_logup;
        let split = mpk.params.effective_tensor_split(total_rounds);
        let running_sum = tensor_running_sum(&gkr_input_evals, split.n_grid_eff, split.b_block);
        let running_sum_trace = ColMajorMatrix::new(running_sum.clone(), 1);
        // SAFETY: `EF` has contiguous basis-coefficient layout over `F`.
        let running_sum_trace_base = unsafe { ext_col_major_to_base(running_sum_trace) };
        let (running_sum_commit, _running_sum_data) = stacked_commit(
            mpk.params.l_skip,
            mpk.params.n_stack,
            mpk.params.log_blowup,
            mpk.params.k_whir(),
            &[&running_sum_trace_base],
        );
        transcript.observe_commit(running_sum_commit);

        let r_grid = (0..split.n_grid_eff)
            .map(|_| transcript.sample_ext())
            .collect_vec();
        let v_curr = eval_mle_from_hypercube(&running_sum, &r_grid);
        let mut r_grid_prev = r_grid.clone();
        if let Some(first) = r_grid_prev.first_mut() {
            *first -= EF::ONE;
        }
        let v_prev = eval_mle_from_hypercube(&running_sum, &r_grid_prev);
        transcript.observe_ext(v_curr);
        transcript.observe_ext(v_prev);

        Some(TensorLogupProof {
            running_sum_commit,
            v_curr,
            v_prev,
        })
    } else {
        None
    };

    let (frac_sum_proof, mut xi) = fractional_sumcheck(transcript, &gkr_input_evals, true);

    // Sample more for `\xi` in the edge case that some AIRs don't have interactions
    let n_global = max(n_max, n_logup);
    debug!(%n_global);
    while xi.len() != l_skip + n_global {
        xi.push(transcript.sample_ext());
    }
    debug!(?xi);
    prover.xi = xi;
    // we now have full \xi vector

    // begin batch sumcheck
    let mut sumcheck_round_polys = Vec::with_capacity(n_max);
    let mut r = Vec::with_capacity(n_max + 1);
    // batching randomness
    let lambda = transcript.sample_ext();
    debug!(%lambda);

    let sp_0_polys = prover.sumcheck_uni_round0_polys(ctx, lambda);
    let sp_0_deg = sumcheck_round0_deg(l_skip, constraint_degree);
    let s_deg = constraint_degree + 1;
    let s_0_deg = sumcheck_round0_deg(l_skip, s_deg);
    let large_uni_domain = (s_0_deg + 1).next_power_of_two();
    let dft = Radix2BowersSerial;
    let s_0_logup_polys = {
        let eq_sharp_uni = eq_sharp_uni_poly(&prover.xi[..l_skip]);
        let mut eq_coeffs = eq_sharp_uni.into_coeffs();
        eq_coeffs.resize(large_uni_domain, EF::ZERO);
        let eq_evals = dft.dft(eq_coeffs);

        let width = 2 * num_traces;
        let mut sp_coeffs_mat = EF::zero_vec(width * large_uni_domain);
        for (i, coeffs) in sp_0_polys[..2 * num_traces].iter().enumerate() {
            // NOTE: coeffs could have length longer than `sp_0_deg + 1` due to coset evaluation,
            // but trailing coefficients should be zero.
            for (j, &c_j) in coeffs.coeffs().iter().enumerate().take(sp_0_deg + 1) {
                // SAFETY:
                // - coeffs length is <= sp_0_deg + 1 <= s_0_deg < large_uni_domain
                // - sp_coeffs_mat allocated for width
                unsafe {
                    *sp_coeffs_mat.get_unchecked_mut(j * width + i) = c_j;
                }
            }
        }
        let mut s_evals = dft.dft_batch(RowMajorMatrix::new(sp_coeffs_mat, width));
        for (eq, row) in zip(eq_evals, s_evals.values.chunks_mut(width)) {
            for x in row {
                *x *= eq;
            }
        }
        dft.idft_batch(s_evals)
    };

    let skip_domain_size = F::from_usize(1 << l_skip);
    // logup sum claims (sum_{\hat p}, sum_{\hat q}) per present AIR
    let (numerator_term_per_air, denominator_term_per_air): (Vec<_>, Vec<_>) = (0..num_traces)
        .map(|trace_idx| {
            let [sum_claim_p, sum_claim_q] = [0, 1].map(|is_denom| {
                // Compute sum over D of s_0(Z) to get the sum claim
                (0..=s_0_deg)
                    .step_by(1 << l_skip)
                    .map(|j| unsafe {
                        // SAFETY: matrix is 2 * num_trace x large_uni_domain, s_0_deg <
                        // large_uni_domain
                        *s_0_logup_polys
                            .values
                            .get_unchecked(j * 2 * num_traces + 2 * trace_idx + is_denom)
                    })
                    .sum::<EF>()
                    * skip_domain_size
            });
            transcript.observe_ext(sum_claim_p);
            transcript.observe_ext(sum_claim_q);

            (sum_claim_p, sum_claim_q)
        })
        .unzip();

    let mu = transcript.sample_ext();
    debug!(%mu);
    let mu_pows = mu.powers().take(3 * num_traces).collect_vec();

    let s_0_zc_poly = {
        let eq_uni = eq_uni_poly::<F, _>(l_skip, prover.xi[0]);
        let mut eq_coeffs = eq_uni.into_coeffs();
        eq_coeffs.resize(large_uni_domain, EF::ZERO);
        let eq_evals = dft.dft(eq_coeffs);

        let mut sp_coeffs = EF::zero_vec(large_uni_domain);
        let mus = &mu_pows[2 * num_traces..];
        let polys = &sp_0_polys[2 * num_traces..];
        for (j, batch_coeff) in sp_coeffs.iter_mut().enumerate().take(sp_0_deg + 1) {
            for (&mu, poly) in zip(mus, polys) {
                *batch_coeff += mu * *poly.coeffs().get(j).unwrap_or(&EF::ZERO);
            }
        }
        let mut s_evals = dft.dft(sp_coeffs);
        for (eq, x) in zip(eq_evals, &mut s_evals) {
            *x *= eq;
        }
        dft.idft(s_evals)
    };

    // Algebraically batch
    let s_0_poly = UnivariatePoly::new(
        zip(
            s_0_logup_polys.values.chunks_exact(2 * num_traces),
            s_0_zc_poly,
        )
        .take(s_0_deg + 1)
        .map(|(logup_row, batched_zc)| {
            let coeff = batched_zc
                + zip(&mu_pows, logup_row)
                    .map(|(&mu_j, &x)| mu_j * x)
                    .sum::<EF>();
            transcript.observe_ext(coeff);
            coeff
        })
        .collect(),
    );

    let r_0 = transcript.sample_ext();
    r.push(r_0);
    debug!(round = 0, r_round = %r_0);
    prover.prev_s_eval = s_0_poly.eval_at_point(r_0);
    debug!("s_0(r_0) = {}", prover.prev_s_eval);

    prover.fold_ple_evals(ctx, r_0);

    // Sumcheck rounds:
    // - each round the prover needs to compute univariate polynomial `s_round`. This poly is linear
    //   since we are taking MLE of `evals`.
    // - at end of each round, sample random `r_round` in `EF`
    //
    // `s_round` is degree `s_deg` so we evaluate it at `0, ..., =s_deg`. The prover skips
    // evaluation at `0` because the verifier can infer it from the previous round's
    // `s_{round-1}(r)` claim. The degree is constraint_degree + 1, where + 1 is from eq term
    let _mle_rounds_span =
        info_span!("prover.batch_constraints.mle_rounds", phase = "prover").entered();
    debug!(%s_deg);
    for round in 1..=n_max {
        let sp_round_evals = prover.sumcheck_polys_eval(round, r[round - 1]);
        // From s'_T above, we can form s'_head(X) and s'_tail where s'_tail is constant
        // The desired polynomial s(X) for this round `j` is
        // s(X) = eq(\vec xi, \vec r_{j-1}) eq(xi_{}, X) s'_head(X) + s'_tail * X
        //
        // The head vs tail corresponds to the cutoff in front loaded batching where the coordinates
        // have been exhausted.
        //
        // In fact, we further need to split s'_head into s'_{head,zc} and s'_{head,logup} due to
        // different eq versus eq_sharp round 0 contributions.
        let tail_start = prover
            .n_per_trace
            .iter()
            .find_position(|&&n| round as isize > n)
            .map(|(i, _)| i)
            .unwrap_or(num_traces);
        let mut sp_head_zc = vec![EF::ZERO; constraint_degree];
        let mut sp_head_logup = vec![EF::ZERO; constraint_degree];
        let mut sp_tail = EF::ZERO;
        for trace_idx in 0..num_traces {
            let zc_idx = 2 * num_traces + trace_idx;
            let numer_idx = 2 * trace_idx;
            let denom_idx = numer_idx + 1;
            if trace_idx < tail_start {
                for i in 0..constraint_degree {
                    sp_head_zc[i] += mu_pows[zc_idx] * sp_round_evals[zc_idx][i];
                    sp_head_logup[i] += mu_pows[numer_idx] * sp_round_evals[numer_idx][i]
                        + mu_pows[denom_idx] * sp_round_evals[denom_idx][i];
                }
            } else {
                sp_tail += mu_pows[zc_idx] * sp_round_evals[zc_idx][0]
                    + mu_pows[numer_idx] * sp_round_evals[numer_idx][0]
                    + mu_pows[denom_idx] * sp_round_evals[denom_idx][0];
            }
        }
        // With eq(xi,r) contributions
        let mut sp_head_evals = vec![EF::ZERO; s_deg];
        for i in 0..constraint_degree {
            sp_head_evals[i + 1] = prover.eq_ns[round - 1] * sp_head_zc[i]
                + prover.eq_sharp_ns[round - 1] * sp_head_logup[i];
        }
        // We need to derive s'(0).
        // We use that s_j(0) + s_j(1) = s_{j-1}(r_{j-1})
        let xi_cur = prover.xi[l_skip + round - 1];
        {
            let eq_xi_0 = EF::ONE - xi_cur;
            let eq_xi_1 = xi_cur;
            sp_head_evals[0] =
                (prover.prev_s_eval - eq_xi_1 * sp_head_evals[1] - sp_tail) * eq_xi_0.inverse();
        }
        // s' has degree s_deg - 1
        let sp_head = UnivariatePoly::lagrange_interpolate(
            &(0..s_deg).map(F::from_usize).collect_vec(),
            &sp_head_evals,
        );
        // eq(xi, X) = (2 * xi - 1) * X + (1 - xi)
        // Compute s(X) = eq(xi, X) * s'_head(X) + s'_tail * X (s'_head now contains eq(..,r))
        // s(X) has degree s_deg
        let batch_s = {
            let mut coeffs = sp_head.into_coeffs();
            coeffs.push(EF::ZERO);
            let b = EF::ONE - xi_cur;
            let a = xi_cur - b;
            for i in (0..s_deg).rev() {
                coeffs[i + 1] = a * coeffs[i] + b * coeffs[i + 1];
            }
            coeffs[0] *= b;
            coeffs[1] += sp_tail;
            UnivariatePoly::new(coeffs)
        };
        let batch_s_evals = (1..=s_deg)
            .map(|i| batch_s.eval_at_point(EF::from_usize(i)))
            .collect_vec();
        for &eval in &batch_s_evals {
            transcript.observe_ext(eval);
        }
        sumcheck_round_polys.push(batch_s_evals);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);
        prover.prev_s_eval = batch_s.eval_at_point(r_round);

        prover.fold_mle_evals(round, r_round);
    }
    drop(_mle_rounds_span);
    assert_eq!(r.len(), n_max + 1);

    let column_openings = prover.into_column_openings();

    // Observe common main openings first, and then preprocessed/cached
    for openings in &column_openings {
        for (claim, claim_rot) in &openings[0] {
            transcript.observe_ext(*claim);
            transcript.observe_ext(*claim_rot);
        }
    }
    for openings in &column_openings {
        for part in openings.iter().skip(1) {
            for (claim, claim_rot) in part {
                transcript.observe_ext(*claim);
                transcript.observe_ext(*claim_rot);
            }
        }
    }

    let batch_constraint_proof = BatchConstraintProof {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs: s_0_poly.into_coeffs(),
        sumcheck_round_polys,
        column_openings,
    };
    let gkr_proof = GkrProof {
        logup_pow_witness,
        tensor_logup,
        q0_claim: frac_sum_proof.fractional_sum.1,
        claims_per_layer: frac_sum_proof.claims_per_layer,
        sumcheck_polys: frac_sum_proof.sumcheck_polys,
    };
    (gkr_proof, batch_constraint_proof, r)
}
