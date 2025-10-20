//! Stacked opening reduction

use std::{collections::HashMap, iter::zip};

use itertools::Itertools;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use tracing::{debug, instrument};

use crate::{
    EF, F,
    poly_common::{eval_eq_mle, eval_eq_uni, eval_eq_uni_at_one},
    poseidon2::sponge::FiatShamirTranscript,
    proof::StackingProof,
    prover::{
        ColMajorMatrix, ColMajorMatrixView, MatrixView,
        poly::evals_eq_hypercube,
        stacked_pcs::StackedLayout,
        sumcheck::{
            batch_fold_mle_evals, fold_mle_evals, fold_ple_evals, sumcheck_round_poly_evals,
            sumcheck_round0_deg, sumcheck_uni_round0_poly,
        },
    },
};

/// Batch sumcheck to reduce trace openings, including rotations, to stacked matrix opening.
///
/// The `stacked_matrix, stacked_layout` should be the result of stacking the `traces` with
/// parameters `l_skip` and `n_stack`.
#[instrument(level = "info", skip_all)]
pub fn stacked_opening_reduction<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    l_skip: usize,
    n_stack: usize,
    stacked_per_commit: &[(&ColMajorMatrix<F>, &StackedLayout)],
    r: &[EF],
) -> (StackingProof, Vec<EF>) {
    // Batching randomness
    let lambda = transcript.sample_ext();
    let lambda_pows = lambda
        .powers()
        .take(
            stacked_per_commit
                .iter()
                .map(|(_, l)| {
                    assert!(!l.sorted_cols.is_empty());
                    l.sorted_cols.len() * 2
                })
                .sum(),
        )
        .collect_vec();

    let mut round_polys_eval = Vec::with_capacity(n_stack);

    // Flattened list of unstacked trace column slices for convenience
    let trace_views = stacked_per_commit
        .iter()
        .enumerate()
        .flat_map(|(com_idx, (_, layout))| {
            layout.unstacked_slices_iter().map(move |s| (com_idx, *s))
        })
        .collect_vec();
    let mut ht_diff_idxs = Vec::new();
    let mut eq_r_per_lht: HashMap<usize, ColMajorMatrix<EF>> = HashMap::new();
    let mut last_height = 0;
    for (i, (_, s)) in trace_views.iter().enumerate() {
        let n = s.log_height - l_skip;
        if i == 0 || s.log_height != last_height {
            ht_diff_idxs.push(i);
            last_height = s.log_height;
        }
        eq_r_per_lht
            .entry(s.log_height)
            .or_insert_with(|| ColMajorMatrix::new(evals_eq_hypercube(&r[1..1 + n]), 1));
    }
    ht_diff_idxs.push(trace_views.len());

    // +1 from eq term
    let s_0_deg = sumcheck_round0_deg(l_skip, 2);
    // We want to compute algebraic batching, via \lambda,
    // for each (T, j) pair of (trace, column) of the univariate polynomials
    // ```text
    // Z -> sum_{x in H_{n_stack}} q(Z,x) eq((Z,x[..n_T]), r[..1+n_T]) eq(x[n_T..], b_{T,j})
    // Z -> sum_{x in H_{n_stack}} q(Z,x) \kappa_\rot((Z,x[..n_T]), r[..1+n_T]) eq(x[n_T..], b_{T,j})
    // ```
    // where `b_{T,j}` is length `n_stack - n_T` binary encoding of `StackedSlice.row_idx >> (l_skip
    // + n_T)`. Note that since x is in the hypercube, by definition of `eq` the above
    // simplifies to
    // ```text
    // Z -> sum_{x in H_{n_T}} q(Z,x,b_{T,j}) eq(Z,r_0) eq(x[..n_T], r[1..1+n_T])
    // Z -> sum_{x in H_{n_T}} q(Z,x,b_{T,j}) \kappa_rot((Z, x[..n_T]), (r_0, r[1..1+n_T]))
    // ```
    // where we also simplified the other `eq` term.
    // We further simplify the second using equation
    // ```text
    // \kappa_rot((Z, x[..n_T]), (r_0, r[1..1+n_T])) =
    // eq_D(Z,omega_D r_0) eq(x[..n_T], r[1..1+n_T]) + eq_D(Z,1)eq_D(omega_D r_0,1) ( kappa_rot(x[..n_T], r[1..1+n_T]) - eq(x[..n_T], r[1..1+n_T]) )
    // ```
    // We compute the last polynomial in our usual way, by considering `q(\vec Z, b_{T,j})` as a
    // prismalinear polynomial and using its evaluations on `D_{n_T}`.
    let omega_skip = F::two_adic_generator(l_skip);
    let eq_const = eval_eq_uni_at_one(l_skip, r[0] * omega_skip);
    let s_0_polys: Vec<_> = ht_diff_idxs
        .par_windows(2)
        .flat_map(|window| {
            let t_window = &trace_views[window[0]..window[1]];
            let log_height = t_window[0].1.log_height;
            let n = log_height - l_skip;
            let eq_rs = eq_r_per_lht.get(&log_height).unwrap().column(0);
            debug_assert_eq!(eq_rs.len(), 1 << n);
            // Prepare the q subslice eval views
            let t_cols = t_window
                .iter()
                .map(|(com_idx, s)| {
                    debug_assert_eq!(s.log_height, log_height);
                    let q = &stacked_per_commit[*com_idx].0;
                    let t_col = &q.column(s.col_idx)[s.row_idx..s.row_idx + (1 << s.log_height)];
                    (ColMajorMatrixView::new(t_col, 1), false)
                })
                .collect_vec();
            sumcheck_uni_round0_poly(l_skip, n, 2, &t_cols, |z, x, evals| {
                let eq_cube = eq_rs[x];
                let eq_uni_r0 = eval_eq_uni(l_skip, z.into(), r[0]);
                let eq_uni_r0_rot = eval_eq_uni(l_skip, z.into(), r[0] * omega_skip);
                let eq_uni_1 = eval_eq_uni_at_one(l_skip, z);
                let k_rot_cube = eq_rs[rot_prev(x, n)];

                let eq = eq_uni_r0 * eq_cube;
                let k_rot = eq_uni_r0_rot * eq_cube + eq_const * eq_uni_1 * (k_rot_cube - eq_cube);
                zip(
                    lambda_pows[2 * window[0]..2 * window[1]].chunks_exact(2),
                    evals,
                )
                .fold([EF::ZERO; 2], |mut acc, (lambdas, eval)| {
                    let q = eval[0];
                    acc[0] += lambdas[0] * eq * q;
                    acc[1] += lambdas[1] * k_rot * q;
                    acc
                })
            })
        })
        .collect();
    let s_0_coeffs = (0..=s_0_deg)
        .map(|i| s_0_polys.iter().map(|evals| evals.0[i]).sum::<EF>())
        .collect_vec();
    for &coeff in &s_0_coeffs {
        transcript.observe_ext(coeff);
    }
    // end round 0

    let mut u_vec = Vec::with_capacity(n_stack + 1);
    let u_0 = transcript.sample_ext();
    u_vec.push(u_0);
    debug!(round = 0, u_round = %u_0);

    let mut q_evals = stacked_per_commit
        .iter()
        .map(|(mat, _)| fold_ple_evals(l_skip, mat.as_view(), false, u_0))
        .collect_vec();
    // fold PLEs into MLEs for \eq and \kappa_\rot, using u_0
    let eq_uni_u0r0 = eval_eq_uni(l_skip, u_0, r[0]);
    let eq_uni_u0r0_rot = eval_eq_uni(l_skip, u_0, r[0] * omega_skip);
    let eq_uni_u01 = eval_eq_uni_at_one(l_skip, u_0);
    // \kappa_\rot(x, r) = eq(rot^{-1}(x), r)
    let mut k_rot_r_per_lht: HashMap<usize, ColMajorMatrix<EF>> = eq_r_per_lht
        .par_iter_mut()
        .map(|(lht, mat)| {
            let n = *lht - l_skip;
            debug_assert_eq!(mat.values.len(), 1 << n);
            // folded \kappa_\rot evals
            let evals: Vec<_> = (0..1 << n)
                .into_par_iter()
                .map(|x| {
                    let eq_cube = unsafe { *mat.get_unchecked(x, 0) };
                    let k_rot_cube = unsafe { *mat.get_unchecked(rot_prev(x, n), 0) };
                    eq_uni_u0r0_rot * eq_cube + eq_const * eq_uni_u01 * (k_rot_cube - eq_cube)
                })
                .collect();
            // update \eq with the univariate factor:
            mat.values.par_iter_mut().for_each(|v| {
                *v *= eq_uni_u0r0;
            });
            (*lht, ColMajorMatrix::new(evals, 1))
        })
        .collect();

    let s_deg = 2;
    // Stores eq(u[1+n_T..round-1], b_{T,j}[..round-n_T-1])
    let mut eq_ub_per_trace = vec![EF::ONE; trace_views.len()];
    #[allow(clippy::needless_range_loop)]
    for round in 1..=n_stack {
        // We want to compute algebraic batching, via \lambda,
        // for each (T, j) pair of (trace, column) of the univariate polynomials
        // ```
        // X -> sum_{y in H_{n_stack-round}} q(u[..round],X,y) eq((u[..round],X,y[..n_T-round]), r[..1+n_T]) eq(y[n_T-round..], b_{T,j})
        //      = sum_{y in H_{n_T-round}} q(u[..round],X,y,b_{T,j}) eq((u[..round],X,y), r[..1+n_T])
        // X -> sum_{y in H_{n_stack-round}} q(u[..round],X,y) \kappa_\rot((u[..round],X,y[..n_T]), r[..1+n_T]) eq(y[n_T..], b_{T,j})
        // ```
        // if `round <= n_T`. Otherwise we compute
        // ```
        // X -> sum_{y in H_{n_stack-round}} q(u[..round],X,y) eq((u[..1+n_T], r[..1+n_T]) eq((u[1+n_T..round],X,y[round..]), b_{T,j})
        //      = q(u[..round], Z, b_{T,j}[round-n_T..]) eq((u[..1+n_T], r[..1+n_T]) eq((u[1+n_T..round],X), b_{T,j}[..round-n_T])
        // X -> sum_{y in H_{n_stack-round}} q(u[..round],X,y) \kappa_\rot(u[..1+n_T], r[..1+n_T]) eq((u[1+n_T..round],X,y[round..]), b_{T,j})
        // ```
        let s_evals: Vec<_> = ht_diff_idxs
            .par_windows(2)
            .flat_map(|window| {
                let t_views = &trace_views[window[0]..window[1]];
                let log_height = t_views[0].1.log_height;
                let n = log_height - l_skip; // n_T
                let hypercube_dim = n.saturating_sub(round);
                let eq_rs = eq_r_per_lht.get(&log_height).unwrap().column(0);
                let k_rot_rs = k_rot_r_per_lht.get(&log_height).unwrap().column(0);
                debug_assert_eq!(eq_rs.len(), 1 << n.saturating_sub(round - 1));
                debug_assert_eq!(k_rot_rs.len(), 1 << n.saturating_sub(round - 1));
                // Prepare the q subslice eval views
                let t_cols = t_views
                    .iter()
                    .map(|(com_idx, s)| {
                        debug_assert_eq!(s.log_height, log_height);
                        let q = &q_evals[*com_idx];
                        let row_start = if round <= n {
                            (s.row_idx >> log_height) << (hypercube_dim + 1)
                        } else {
                            (s.row_idx >> (log_height + round - n)) << 1
                        };
                        let t_col =
                            &q.column(s.col_idx)[row_start..row_start + (2 << hypercube_dim)];
                        ColMajorMatrixView::new(t_col, 1)
                    })
                    .collect_vec();
                sumcheck_round_poly_evals(hypercube_dim + 1, s_deg, &t_cols, |x, y, evals| {
                    evals
                        .iter()
                        .enumerate()
                        .fold([EF::ZERO; 2], |mut acc, (i, eval)| {
                            let t_idx = window[0] + i;
                            let q = eval[0];
                            let mut eq_ub = eq_ub_per_trace[t_idx];
                            let (eq, k_rot) = if round > n {
                                // Extra contribution of eq(X, b_{T,j}[round-n_T-1])
                                let b = (trace_views[t_idx].1.row_idx >> (l_skip + round - 1)) & 1;
                                eq_ub *= eval_eq_mle(&[x], &[F::from_bool(b == 1)]);
                                debug_assert_eq!(y, 0);
                                (eq_rs[0] * eq_ub, k_rot_rs[0] * eq_ub)
                            } else {
                                // linearly interpolate eq(-, r[..1+n_T]), \kappa_\rot(-,
                                // r[..1+n_T])
                                let eq_r = eq_rs[y << 1] * (EF::ONE - x) + eq_rs[(y << 1) + 1] * x;
                                let k_rot_r =
                                    k_rot_rs[y << 1] * (EF::ONE - x) + k_rot_rs[(y << 1) + 1] * x;
                                (eq_r * eq_ub, k_rot_r * eq_ub)
                            };
                            acc[0] += lambda_pows[t_idx * 2] * q * eq;
                            acc[1] += lambda_pows[t_idx * 2 + 1] * q * k_rot;
                            acc
                        })
                })
            })
            .collect();
        let batch_s_evals = (0..s_deg)
            .map(|i| s_evals.iter().map(|evals| evals[i]).sum::<EF>())
            .collect_vec();
        for &eval in &batch_s_evals {
            transcript.observe_ext(eval);
        }
        round_polys_eval.push(batch_s_evals);

        let u_round = transcript.sample_ext();
        u_vec.push(u_round);
        debug!(%round, %u_round);

        q_evals = batch_fold_mle_evals(q_evals, u_round);
        eq_r_per_lht = eq_r_per_lht
            .into_par_iter()
            .map(|(lht, mat)| (lht, fold_mle_evals(mat, u_round)))
            .collect();
        k_rot_r_per_lht = k_rot_r_per_lht
            .into_par_iter()
            .map(|(lht, mat)| (lht, fold_mle_evals(mat, u_round)))
            .collect();
        for ((_, s), eq_ub) in zip(&trace_views, &mut eq_ub_per_trace) {
            let n = s.log_height - l_skip;
            if round > n {
                // Folding above did nothing, and we update the eq(u[1+n_T..=round],
                // b_{T,j}[..=round-n_T-1]) value
                let b = (s.row_idx >> (l_skip + round - 1)) & 1;
                *eq_ub *= eval_eq_mle(&[u_round], &[F::from_bool(b == 1)]);
            }
        }
    }
    let stacking_openings = q_evals
        .into_iter()
        .map(|q| {
            debug_assert_eq!(q.height(), 1);
            for &q_j in &q.values {
                transcript.observe_ext(q_j);
            }
            q.values
        })
        .collect_vec();
    let proof = StackingProof {
        univariate_round_coeffs: s_0_coeffs,
        sumcheck_round_polys: round_polys_eval
            .into_iter()
            .map(|evals| evals.try_into().unwrap())
            .collect(),
        stacking_openings,
    };
    (proof, u_vec)
}

/// `x_int` is the integer representation of point on H_n.
fn rot_prev(x_int: usize, n: usize) -> usize {
    debug_assert!(x_int < (1 << n));
    if x_int == 0 { (1 << n) - 1 } else { x_int - 1 }
}
