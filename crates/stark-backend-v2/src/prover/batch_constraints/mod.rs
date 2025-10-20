//! Batch sumcheck for ZeroCheck constraints and sumcheck for LogUp input layer MLEs

use std::{
    cmp::max,
    iter::{self, zip},
};

use itertools::Itertools;
use openvm_stark_backend::{
    air_builders::symbolic::{
        SymbolicConstraints, SymbolicExpressionNode, symbolic_variable::Entry,
    },
    parizip,
    prover::MatrixDimensions,
};
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_u64, log2_strict_usize};
use tracing::{debug, instrument};

use crate::{
    EF, F,
    poly_common::{UnivariatePoly, eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni},
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, GkrProof},
    prover::{
        ColMajorMatrix, CpuBackendV2, DeviceMultiStarkProvingKeyV2, ProvingContextV2,
        batch_constraints::single::EvalHelper,
        fractional_sumcheck_gkr::{Frac, fractional_sumcheck},
        poly::evals_eq_hypercube,
        stacked_pcs::StackedLayout,
        sumcheck::{
            batch_fold_mle_evals, batch_fold_ple_evals, fold_ple_evals, sumcheck_round_poly_evals,
            sumcheck_round0_deg, sumcheck_uni_round0_poly,
        },
    },
};

mod evaluator;
mod single;

#[instrument(level = "info", skip_all)]
pub fn prove_zerocheck_and_logup<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    mpk: &DeviceMultiStarkProvingKeyV2<CpuBackendV2>,
    ctx: &ProvingContextV2<CpuBackendV2>,
) -> (GkrProof, BatchConstraintProof, Vec<EF>) {
    let l_skip = mpk.params.l_skip;
    let omega_skip = F::two_adic_generator(l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();

    let constraint_degree = mpk.max_constraint_degree;
    let num_airs_present = ctx.per_trace.len();

    // Preparation stage: create Vec's all indexed by present AIRs (trace_idx)
    let (eval_helpers, mat_views_per_trace, interactions_meta, n_per_trace): (
        Vec<EvalHelper<F>>,
        Vec<_>,
        Vec<_>,
        Vec<_>,
    ) = ctx
        .per_trace
        .iter()
        .enumerate()
        .map(|(trace_idx, (air_id, air_ctx))| {
            let pk = &mpk.per_air[*air_id];
            let constraints = &pk.vk.symbolic_constraints.constraints;
            let public_values = &air_ctx.public_values;
            let preprocessed_trace = pk
                .preprocessed_data
                .as_ref()
                .map(|(_, d)| d.layout.mat_view(0, d.matrix.as_view()));
            let partitioned_main_trace = air_ctx
                .cached_mains
                .iter()
                .map(|(_, d)| d.layout.mat_view(0, d.matrix.as_view()))
                .chain(iter::once(air_ctx.common_main.as_view()))
                .collect_vec();
            // Scan constraints to see if we need `next` row and also check index bounds
            // so we don't need to check them per row.
            let mut rotation = 0;
            for node in &constraints.nodes {
                if let SymbolicExpressionNode::Variable(var) = node {
                    match var.entry {
                        Entry::Preprocessed { offset } => {
                            rotation = max(rotation, offset);
                            assert!(var.index < preprocessed_trace.as_ref().unwrap().width());
                        }
                        Entry::Main { part_index, offset } => {
                            rotation = max(rotation, offset);
                            assert!(
                                var.index < partitioned_main_trace[part_index].width(),
                                "col_index={} >= main partition {} width={}",
                                var.index,
                                part_index,
                                partitioned_main_trace[part_index].width()
                            );
                        }
                        Entry::Public => {
                            assert!(var.index < public_values.len());
                        }
                        _ => unreachable!("after_challenge not supported"),
                    }
                }
            }
            let needs_next = rotation > 0;
            let symbolic_constraints = SymbolicConstraints::from(&pk.vk.symbolic_constraints);
            let helper = EvalHelper {
                constraints_dag: &pk.vk.symbolic_constraints.constraints,
                interactions: symbolic_constraints.interactions,
                public_values,
                preprocessed_trace,
                needs_next,
            };
            let mat_views = helper.view_mats(air_ctx);

            let num_interactions = pk.vk.symbolic_constraints.interactions.len();
            let height = air_ctx.common_main.height();
            let log_height = log2_strict_usize(height);
            let meta = (trace_idx, num_interactions, log_height);
            (helper, mat_views, meta, log_height - l_skip)
        })
        .multiunzip();

    // Prepare for logup GKR
    let max_interaction_length: usize = mpk
        .per_air
        .iter()
        .flat_map(|pk| {
            pk.vk
                .symbolic_constraints
                .interactions
                .iter()
                .map(|int| int.message.len())
        })
        .max()
        .unwrap_or(0);
    let total_interaction_wt = interactions_meta
        .iter()
        .map(|(_, num_interactions, log_height)| {
            (*num_interactions as u64) << (log_height - l_skip)
        })
        .sum::<u64>();
    let logup_pow_witness = transcript.grind(mpk.params.logup_pow_bits);
    let alpha_logup = transcript.sample_ext();
    let beta_logup = transcript.sample_ext();
    debug!(%alpha_logup, %beta_logup);
    let beta_pows = beta_logup
        .powers()
        .take(max_interaction_length + 1)
        .collect_vec();
    // Implicitly, the width of this stacking should be 1
    let n_logup = log2_ceil_u64(total_interaction_wt) as usize;
    debug!(%n_logup);
    let interactions_layout = StackedLayout::new(l_skip + n_logup, interactions_meta);

    // For each trace, create selectors as a 3-column matrix of [is_first, is_transition, is_last]
    // PERF[jpw]: these can be shared for the same `n`
    let sels_per_trace = n_per_trace
        .iter()
        .map(|&n| {
            let height = 1 << (l_skip + n);
            let mut mat = F::zero_vec(3 * height);
            mat[0] = F::ONE;
            mat[height..2 * height - 1].fill(F::ONE);
            *mat.last_mut().unwrap() = F::ONE;
            ColMajorMatrix::new(mat, 3)
        })
        .collect_vec();

    let gkr_input_evals = if total_interaction_wt == 0 {
        vec![]
    } else {
        // Per trace, a row major matrix of interaction evaluations
        // PERF[jpw]: we should write directly to the stacked `evals` in memory below
        let unstacked_interaction_evals = eval_helpers
            .par_iter()
            .enumerate()
            .map(|(trace_idx, helper)| {
                let mats = &mat_views_per_trace[trace_idx];
                let n = n_per_trace[trace_idx];
                let sels = &sels_per_trace[trace_idx];
                let height = 1 << (l_skip + n);
                (0..height)
                    .into_par_iter()
                    .map(|i| {
                        let mut row_parts = Vec::with_capacity(mats.len() + 1);
                        row_parts.push(sels.columns().map(|col| col[i]).collect_vec());
                        for (mat, is_rot) in mats {
                            let offset = usize::from(*is_rot);
                            row_parts.push(
                                mat.columns()
                                    .map(|col| col[(i + offset) % height])
                                    .collect_vec(),
                            );
                        }
                        helper.eval_interactions(&row_parts, &beta_pows)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut evals = vec![Frac::default(); 1 << (l_skip + n_logup)];
        for (trace_idx, interaction_idx, s) in interactions_layout.sorted_cols.iter().copied() {
            assert_eq!(s.col_idx, 0);
            for (i, evals_at_z) in unstacked_interaction_evals[trace_idx].iter().enumerate() {
                let (numer, denom) = evals_at_z[interaction_idx];
                evals[s.row_idx + i] = Frac::new(numer.into(), denom);
            }
        }
        // Prevent division by zero:
        evals.par_iter_mut().for_each(|frac| frac.q += alpha_logup);
        evals
    };
    let (frac_sum_proof, mut xi) = fractional_sumcheck(transcript, &gkr_input_evals, true);
    // end fractional sumcheck (GKR)

    // begin batch sumcheck
    let mut n_global = n_per_trace.iter().copied().max().unwrap();
    if n_global < n_logup {
        n_global = n_logup;
    } else {
        while xi.len() != l_skip + n_global {
            xi.push(transcript.sample_ext());
        }
    }
    debug!(%n_global);
    debug!(?xi);
    // we now have full \xi vector

    let mut sumcheck_round_polys = Vec::with_capacity(n_global);
    let mut r = Vec::with_capacity(n_global + 1);
    // batching randomness
    let lambda = transcript.sample_ext();
    debug!(%lambda);
    let max_num_constraints = mpk
        .per_air
        .iter()
        .map(|pk| pk.vk.symbolic_constraints.constraints.constraint_idx.len())
        .max()
        .unwrap_or(0);
    let lambda_pows = lambda.powers().take(max_num_constraints).collect_vec();

    // PERF[jpw]: make Hashmap from unique n -> eq_n(xi, -)
    // NOTE: this is evaluations of `x -> eq_{H_n}(x, \xi[l_skip..l_skip + n])` on hypercube `H_n`.
    // We store the univariate component eq_D separately as an optimization.
    let mut eq_xi_per_trace: Vec<_> = n_per_trace
        .par_iter()
        .map(|&n| {
            // PERF[jpw]: might be able to share computations between eq_xi, eq_sharp computations
            // the eq(xi, -) evaluations on hyperprism for zerocheck
            let eq_xi = evals_eq_hypercube(&xi[l_skip..l_skip + n]);
            ColMajorMatrix::new(eq_xi, 1)
        })
        .collect();
    // For each trace, for each interaction \hat\sigma, the eq(ξ_3,b_{T,\hat\sigma}) term.
    // This is some weight per interaction that does not depend on the row.
    let eq_3b_per_trace = eval_helpers
        .par_iter()
        .enumerate()
        .map(|(trace_idx, helper)| {
            if helper.interactions.is_empty() {
                return vec![];
            }
            let n = n_per_trace[trace_idx];
            let mut b_vec = vec![F::ZERO; n_logup - n];
            (0..helper.interactions.len())
                .map(|i| {
                    let stacked_idx = interactions_layout.get(trace_idx, i).unwrap().row_idx;
                    debug_assert!(stacked_idx.trailing_zeros() as usize >= n + l_skip);
                    let mut b_int = stacked_idx >> (l_skip + n);
                    for b in &mut b_vec {
                        *b = F::from_bool(b_int & 1 == 1);
                        b_int >>= 1;
                    }
                    eval_eq_mle(&xi[l_skip + n..l_skip + n_logup], &b_vec)
                })
                .collect_vec()
        })
        .collect::<Vec<_>>();

    // +1 from eq term
    let s_deg = constraint_degree + 1;
    let s_0_deg = sumcheck_round0_deg(l_skip, s_deg);
    // PERF[jpw]: see Gruen, Section 3.2 and 4 on some ways to reduce the degree of the univariate
    // polynomial. We know s_0 is supposed to vanish on univariate skip domain `D` of size
    // `2^{l_skip}`. Hence `s_0 = Z_D * s'_0(Z)` where `Z_D = \prod_{z in D} (Z - z)` where
    // `s'_0` has degree `d * (2^{l_skip} - 1) - 1`. We can evaluate s'_0 on a coset and then
    // perform coset iDFT to get `s'_0` coefficients. We need to use a coset to ensure
    // disjointness from `D`.
    let s_0_zerochecks = eval_helpers
        .par_iter()
        .enumerate()
        .map(|(trace_idx, helper)| {
            let n = n_per_trace[trace_idx];
            let mats = &mat_views_per_trace[trace_idx];
            let eq_xi = eq_xi_per_trace[trace_idx].column(0);
            let sels = sels_per_trace[trace_idx].as_view();
            let mut parts = vec![(sels, false)];
            parts.extend_from_slice(mats);

            // degree is constraint_degree + 1 due to eq term
            let [s_0] = sumcheck_uni_round0_poly(l_skip, n, s_deg, &parts, |z, x, row_parts| {
                // PERF[jpw]: we are limited by the closure interface but `eq_uni` should be
                // cached
                let eq = eval_eq_uni(l_skip, xi[0], z.into()) * eq_xi[x];
                let constraint_eval = helper.acc_constraints(row_parts, &lambda_pows);
                [eq * constraint_eval]
            });
            debug_assert_eq!(
                omega_skip_pows
                    .iter()
                    .map(|z| s_0.eval_at_point(EF::from(*z)))
                    .sum::<EF>(),
                EF::ZERO,
                "Zerocheck sum is not zero for air_id: {}",
                ctx.per_trace[trace_idx].0
            );
            s_0
        })
        .collect::<Vec<_>>();
    // Reminder: sum claims for zerocheck are zero, per AIR

    // We interpolate each logup round 0 sumcheck poly because we need to use it to compute
    // sum_{\hat{p}, T, I}, sum_{\hat{q}, T, I} per trace.
    let s_0_logups = eval_helpers
        .par_iter()
        .enumerate()
        .flat_map(|(trace_idx, helper)| {
            if helper.interactions.is_empty() {
                return [(); 2].map(|_| UnivariatePoly::new(vec![EF::ZERO; s_0_deg + 1]));
            }
            let n = n_per_trace[trace_idx];
            let mats = &mat_views_per_trace[trace_idx];
            let eq_xi = eq_xi_per_trace[trace_idx].column(0);
            let eq_3bs = &eq_3b_per_trace[trace_idx];
            let sels = sels_per_trace[trace_idx].as_view();
            let mut parts = vec![(sels, false)];
            parts.extend_from_slice(mats);

            // degree is constraint_degree + 1 due to eq term
            sumcheck_uni_round0_poly(l_skip, n, s_deg, &parts, |z, x, row_parts| {
                let eq_sharp =
                    eval_eq_sharp_uni(&omega_skip_pows, &xi[..l_skip], z.into()) * eq_xi[x];
                let [numer, denom] = helper.acc_interactions(row_parts, &beta_pows, eq_3bs);
                [eq_sharp * numer, eq_sharp * denom]
            })
        })
        .collect::<Vec<_>>();
    // logup sum claims (sum_{\hat p}, sum_{\hat q}) per present AIR
    let (numerator_term_per_air, denominator_term_per_air): (Vec<_>, Vec<_>) = s_0_logups
        .chunks_exact(2)
        .map(|frac| {
            // PERF[jpw]: use some batch evaluation algorithm
            let [sum_claim_p, sum_claim_q] = [&frac[0], &frac[1]].map(|s_0| {
                omega_skip_pows
                    .iter()
                    .map(|z| s_0.eval_at_point(EF::from(*z)))
                    .sum::<EF>()
            });
            transcript.observe_ext(sum_claim_p);
            transcript.observe_ext(sum_claim_q);

            (sum_claim_p, sum_claim_q)
        })
        .unzip();

    let mu = transcript.sample_ext();
    debug!(%mu);

    let mu_pows = mu.powers().take(3 * num_airs_present).collect_vec();
    let univariate_round_coeffs = (0..=s_0_deg)
        .map(|i| {
            let eval = iter::empty()
                .chain(&s_0_logups)
                .chain(&s_0_zerochecks)
                .enumerate()
                .map(|(j, s_0)| mu_pows[j] * s_0.0[i])
                .sum::<EF>();
            transcript.observe_ext(eval);
            eval
        })
        .collect_vec();

    let r_0 = transcript.sample_ext();
    r.push(r_0);
    debug!(round = 0, r_round = %r_0);

    // "Fold" all PLE evaluations by interpolating and evaluating at `r_0`.
    // NOTE: after this folding, \hat{T} and \hat{T_{rot}} will be treated as completely distinct
    // matrices.
    let mut mat_evals_per_trace = mat_views_per_trace
        .into_par_iter()
        .map(|mats| {
            mats.into_par_iter()
                .map(|(mat, is_rot)| fold_ple_evals(l_skip, mat, is_rot, r_0))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut sels_per_trace = batch_fold_ple_evals(l_skip, sels_per_trace, false, r_0);
    let eq_r0 = eval_eq_uni(l_skip, xi[0], r_0);
    let eq_sharp_r0 = eval_eq_sharp_uni(&omega_skip_pows, &xi[..l_skip], r_0);
    // Define eq^\sharp_D(xi[0], r0) * eq_{H_n}(xi[1..1+n], x) and also update eq_D(xi[0], r0) *
    // eq_{H_n}(xi[1..1+n], x)
    let mut eq_sharp_per_trace: Vec<_> = eq_xi_per_trace
        .par_iter_mut()
        .map(|eq| {
            let eq_sharp_evals = eq
                .values
                .par_iter_mut()
                .map(|x| {
                    let eq = *x;
                    *x *= eq_r0;
                    eq * eq_sharp_r0
                })
                .collect();
            ColMajorMatrix::new(eq_sharp_evals, 1)
        })
        .collect();
    // Sumcheck rounds:
    // - each round the prover needs to compute univariate polynomial `s_round`. This poly is linear
    //   since we are taking MLE of `evals`.
    // - at end of each round, sample random `r_round` in `EF`
    //
    // `s_round` is degree `s_deg` so we evaluate it at `0, ..., =s_deg`. The prover skips
    // evaluation at `0` because the verifier can infer it from the previous round's
    // `s_{round-1}(r)` claim. The degree is constraint_degree + 1, where + 1 is from eq term
    debug!(%s_deg);
    // Stores \hat{f}(\vec r_n) * r_{n+1} .. r_{round-1} for polys f that are "done" in the batch
    // sumcheck
    let mut zerocheck_tilde_evals = vec![EF::ZERO; num_airs_present];
    let mut logup_tilde_evals = vec![[EF::ZERO; 2]; num_airs_present];
    for round in 1..=n_global {
        let s_zerocheck_evals: Vec<Vec<EF>> = parizip!(&eval_helpers, &mut zerocheck_tilde_evals)
            .enumerate()
            .map(|(trace_idx, (helper, tilde_eval))| {
                let n = n_per_trace[trace_idx];
                let mats = &mat_evals_per_trace[trace_idx];
                let sels = &sels_per_trace[trace_idx];
                let eq_xi = &eq_xi_per_trace[trace_idx];

                if round > n {
                    if round == n + 1 {
                        // Evaluate \hat{f}(\vec r_n)
                        let parts = iter::once(sels)
                            .chain(mats)
                            .map(|mat| mat.columns().map(|c| c[0]).collect_vec())
                            .collect_vec();
                        *tilde_eval =
                            eq_xi.column(0)[0] * helper.acc_constraints(&parts, &lambda_pows);
                    } else {
                        *tilde_eval *= r[round - 1];
                    };
                    (1..=s_deg)
                        .map(|x| *tilde_eval * F::from_canonical_usize(x))
                        .collect()
                } else {
                    let parts = iter::empty()
                        .chain([eq_xi, sels])
                        .chain(mats)
                        .map(|m| m.as_view())
                        .collect_vec();
                    let [s] = sumcheck_round_poly_evals(
                        n - (round - 1),
                        s_deg,
                        &parts,
                        |_x, _y, row_parts| {
                            let eq = row_parts[0][0];
                            let constraint_eval =
                                helper.acc_constraints(&row_parts[1..], &lambda_pows);
                            [eq * constraint_eval]
                        },
                    );
                    s
                }
            })
            .collect();

        let s_logup_evals: Vec<Vec<EF>> = parizip!(&eval_helpers, &mut logup_tilde_evals)
            .enumerate()
            .flat_map(|(trace_idx, (helper, tilde_eval))| {
                if helper.interactions.is_empty() {
                    return [vec![EF::ZERO; s_deg], vec![EF::ZERO; s_deg]];
                }
                let n = n_per_trace[trace_idx];
                let mats = &mat_evals_per_trace[trace_idx];
                let sels = &sels_per_trace[trace_idx];
                let eq_sharp = &eq_sharp_per_trace[trace_idx];
                let eq_3bs = &eq_3b_per_trace[trace_idx];

                if round > n {
                    if round == n + 1 {
                        // Evaluate \hat{f}(\vec r_n)
                        let parts = iter::once(sels)
                            .chain(mats)
                            .map(|mat| mat.columns().map(|c| c[0]).collect_vec())
                            .collect_vec();
                        let eq = eq_sharp.column(0)[0];
                        *tilde_eval = helper
                            .acc_interactions(&parts, &beta_pows, eq_3bs)
                            .map(|x| eq * x);
                    } else {
                        for x in tilde_eval.iter_mut() {
                            *x *= r[round - 1];
                        }
                    };
                    tilde_eval.map(|tilde_eval| {
                        (1..=s_deg)
                            .map(|x| tilde_eval * F::from_canonical_usize(x))
                            .collect()
                    })
                } else {
                    let parts = iter::empty()
                        .chain([eq_sharp, sels])
                        .chain(mats)
                        .map(|m| m.as_view())
                        .collect_vec();
                    sumcheck_round_poly_evals(
                        n - (round - 1),
                        s_deg,
                        &parts,
                        |_x, _y, row_parts| {
                            let eq_sharp = row_parts[0][0];
                            helper
                                .acc_interactions(&row_parts[1..], &beta_pows, eq_3bs)
                                .map(|eval| eq_sharp * eval)
                        },
                    )
                }
            })
            .collect();
        let batch_s_evals = (0..s_deg)
            .map(|i| {
                iter::empty()
                    .chain(&s_logup_evals)
                    .chain(&s_zerocheck_evals)
                    .enumerate()
                    .fold(EF::ZERO, |coeff, (j, evals)| coeff + mu_pows[j] * evals[i])
            })
            .collect_vec();
        for &eval in &batch_s_evals {
            transcript.observe_ext(eval);
        }
        sumcheck_round_polys.push(batch_s_evals);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);

        // Fold all MLE evaluations
        mat_evals_per_trace = mat_evals_per_trace
            .into_iter()
            .map(|mats| batch_fold_mle_evals(mats, r_round))
            .collect_vec();
        sels_per_trace = batch_fold_mle_evals(sels_per_trace, r_round);
        eq_xi_per_trace = batch_fold_mle_evals(eq_xi_per_trace, r_round);
        eq_sharp_per_trace = batch_fold_mle_evals(eq_sharp_per_trace, r_round);
    }
    #[cfg(debug_assertions)]
    if tracing::enabled!(tracing::Level::DEBUG) {
        use crate::prover::poly::{Ple, PleMatrix};
        // Heavy debugging: interpolation the trace matrices to prismalinear coefficient form and
        // honestly evaluate at r
        for trace_idx in 0..num_airs_present {
            debug!(%trace_idx, eq_xi_r = %eq_xi_per_trace[trace_idx].column(0)[0]);
            debug!(%trace_idx, eq_sharp_xi_r = %eq_sharp_per_trace[trace_idx].column(0)[0]);
            let n = n_per_trace[trace_idx];
            let mat_views = eval_helpers[trace_idx].view_mats(&ctx.per_trace[trace_idx].1);
            // Comes in pairs of (mat, mat_rot)
            let mats_evals_r = &mat_evals_per_trace[trace_idx];
            assert_eq!(mats_evals_r.len(), mat_views.len());
            for (mat_evals_r, (evals, is_rot)) in zip(mats_evals_r, mat_views) {
                if !is_rot {
                    let ple_mat = PleMatrix::from_evaluations(l_skip, &evals);
                    for (eval_r, ple) in zip(mat_evals_r.columns(), ple_mat.columns) {
                        assert_eq!(eval_r[0], ple.eval_at_point(l_skip, r[0], &r[1..1 + n]));
                    }
                } else {
                    let height = evals.height();
                    for (eval_r, col_evals) in zip(mat_evals_r.columns(), evals.columns()) {
                        let rot_evals = (0..height)
                            .map(|i| col_evals[(i + 1) % height])
                            .collect_vec();
                        let ple = Ple::from_evaluations(l_skip, &rot_evals);
                        assert_eq!(eval_r[0], ple.eval_at_point(l_skip, r[0], &r[1..1 + n]));
                    }
                }
            }
        }
    }

    assert_eq!(r.len(), n_global + 1);
    let mut column_openings = Vec::with_capacity(num_airs_present);
    // At the end, we've folded all MLEs so they only have one row equal to evaluation at `\vec r`.
    for mut mat_evals in mat_evals_per_trace {
        // Order of mats is:
        // - preprocessed (if has_preprocessed),
        // - preprocessed_rot (if has_preprocessed),
        // - cached(0), cached(0)_rot, ...
        // - common_main
        // - common_main_rot
        // For column openings, we pop common_main, common_main_rot and put it at the front
        assert_eq!(mat_evals.len() % 2, 0); // always include rot for now
        let common_main_rot = mat_evals.pop().unwrap();
        let common_main = mat_evals.pop().unwrap();
        let openings_of_air = iter::once(&[common_main, common_main_rot] as &[_])
            .chain(mat_evals.chunks_exact(2))
            .map(|pair| {
                zip(pair[0].columns(), pair[1].columns())
                    .map(|(claim, claim_rot)| {
                        assert_eq!(claim.len(), 1);
                        assert_eq!(claim_rot.len(), 1);
                        let claim = claim[0];
                        let claim_rot = claim_rot[0];
                        transcript.observe_ext(claim);
                        transcript.observe_ext(claim_rot);

                        (claim, claim_rot)
                    })
                    .collect_vec()
            })
            .collect_vec();
        column_openings.push(openings_of_air);
    }

    let batch_constraint_proof = BatchConstraintProof {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs,
        sumcheck_round_polys,
        column_openings,
    };
    let gkr_proof = GkrProof {
        logup_pow_witness,
        q0_claim: frac_sum_proof.fractional_sum.1,
        claims_per_layer: frac_sum_proof.claims_per_layer,
        sumcheck_polys: frac_sum_proof.sumcheck_polys,
    };
    (gkr_proof, batch_constraint_proof, r)
}
