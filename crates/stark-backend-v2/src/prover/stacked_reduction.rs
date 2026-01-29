//! Stacked opening reduction

use std::{array::from_fn, collections::HashMap, iter::zip, mem::take};

use itertools::Itertools;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use tracing::{debug, instrument};

use crate::{
    poly_common::{eval_eq_mle, eval_eq_uni, eval_eq_uni_at_one, eval_in_uni, UnivariatePoly},
    poseidon2::sponge::FiatShamirTranscript,
    proof::StackingProof,
    prover::{
        poly::evals_eq_hypercube,
        stacked_pcs::{StackedPcsData, StackedSlice},
        sumcheck::{
            batch_fold_mle_evals, fold_mle_evals, fold_ple_evals, sumcheck_round0_deg,
            sumcheck_round_poly_evals, sumcheck_uni_round0_poly,
        },
        ColMajorMatrix, ColMajorMatrixView, CpuBackendV2, CpuDeviceV2, MatrixView, ProverBackendV2,
    },
    Digest, EF, F,
};

/// Helper trait for proving the reduction of column opening claims and column rotation opening
/// claims to opening claims of column polynomials of the stacked matrix.
///
/// Returns the reduction proof and the random vector `u` of length `1 + n_stack`.
pub trait StackedReductionProver<'a, PB: ProverBackendV2, PD> {
    /// We only provide a view to the stacked `PcsData` per commitment because the WHIR prover will
    /// still use the PLE evaluations of the stacked matrices later. The order of
    /// `stacked_per_commit` is `common_main, preprocessed for trace_idx=0 (if any), cached_0 for
    /// trace_idx=0, ..., preprocessed for trace_idx=1 (if any), ...`.
    ///
    /// The `lambda` is the batching randomness for the batch sumcheck.
    fn new(
        device: &'a PD,
        stacked_per_commit: Vec<&'a PB::PcsData>,
        need_rot_per_commit: Vec<Vec<bool>>,
        r: &[PB::Challenge],
        lambda: PB::Challenge,
    ) -> Self;

    /// Return the `s_0` batched polynomial from univariate round 0 of sumcheck.
    fn batch_sumcheck_uni_round0_poly(&mut self) -> UnivariatePoly<PB::Challenge>;

    fn fold_ple_evals(&mut self, u_0: PB::Challenge);

    fn batch_sumcheck_poly_eval(
        &mut self,
        round: usize,
        u_prev: PB::Challenge,
    ) -> [PB::Challenge; 2];

    fn fold_mle_evals(&mut self, round: usize, u_round: PB::Challenge);

    fn into_stacked_openings(self) -> Vec<Vec<PB::Challenge>>;
}

/// Batch sumcheck to reduce trace openings, including rotations, to stacked matrix opening.
///
/// The `stacked_matrix, stacked_layout` should be the result of stacking the `traces` with
/// parameters `l_skip` and `n_stack`.
#[instrument(level = "info", skip_all)]
pub fn prove_stacked_opening_reduction<'a, PB, PD, TS, SRP>(
    device: &'a PD,
    transcript: &mut TS,
    n_stack: usize,
    stacked_per_commit: Vec<&'a PB::PcsData>,
    need_rot_per_commit: Vec<Vec<bool>>,
    r: &[PB::Challenge],
) -> (StackingProof, Vec<PB::Challenge>)
where
    PB: ProverBackendV2<Val = F, Challenge = EF>,
    TS: FiatShamirTranscript,
    SRP: StackedReductionProver<'a, PB, PD>,
{
    // Batching randomness
    let lambda = transcript.sample_ext();

    let mut prover = SRP::new(device, stacked_per_commit, need_rot_per_commit, r, lambda);
    let s_0 = prover.batch_sumcheck_uni_round0_poly();
    for &coeff in s_0.coeffs() {
        transcript.observe_ext(coeff);
    }

    let mut u_vec = Vec::with_capacity(n_stack + 1);
    let u_0 = transcript.sample_ext();
    u_vec.push(u_0);
    debug!(round = 0, u_round = %u_0);

    prover.fold_ple_evals(u_0);
    // end round 0

    let mut sumcheck_round_polys = Vec::with_capacity(n_stack);

    #[allow(clippy::needless_range_loop)]
    for round in 1..=n_stack {
        let batch_s_evals = prover.batch_sumcheck_poly_eval(round, u_vec[round - 1]);

        for &eval in &batch_s_evals {
            transcript.observe_ext(eval);
        }
        sumcheck_round_polys.push(batch_s_evals);

        let u_round = transcript.sample_ext();
        u_vec.push(u_round);
        debug!(%round, %u_round);

        prover.fold_mle_evals(round, u_round);
    }
    let stacking_openings = prover.into_stacked_openings();
    for claims_for_com in &stacking_openings {
        for &claim in claims_for_com {
            transcript.observe_ext(claim);
        }
    }
    let proof = StackingProof {
        univariate_round_coeffs: s_0.0,
        sumcheck_round_polys,
        stacking_openings,
    };
    (proof, u_vec)
}

pub struct StackedReductionCpu<'a> {
    l_skip: usize,
    omega_skip: F,

    r_0: EF,
    lambda_pows: Vec<EF>,
    eq_const: EF,

    stacked_per_commit: Vec<&'a StackedPcsData<F, Digest>>,
    trace_views: Vec<TraceViewMeta>,
    ht_diff_idxs: Vec<usize>,

    eq_r_per_lht: HashMap<usize, ColMajorMatrix<EF>>,

    // After round 0:
    k_rot_r_per_lht: HashMap<usize, ColMajorMatrix<EF>>,
    q_evals: Vec<ColMajorMatrix<EF>>,
    /// Stores eq(u[1+n_T..round-1], b_{T,j}[..round-n_T-1])
    eq_ub_per_trace: Vec<EF>,
}

struct TraceViewMeta {
    com_idx: usize,
    slice: StackedSlice,
    lambda_eq_idx: usize,
    lambda_rot_idx: usize,
    need_rot: bool,
}

impl<'a> StackedReductionProver<'a, CpuBackendV2, CpuDeviceV2> for StackedReductionCpu<'a> {
    fn new(
        device: &CpuDeviceV2,
        stacked_per_commit: Vec<&'a StackedPcsData<F, Digest>>,
        need_rot_per_commit: Vec<Vec<bool>>,
        r: &[EF],
        lambda: EF,
    ) -> Self {
        let l_skip = device.config().l_skip;
        let omega_skip = F::two_adic_generator(l_skip);

        let mut trace_views = Vec::new();
        let mut lambda_idx = 0usize;
        for (com_idx, d) in stacked_per_commit.iter().enumerate() {
            let need_rot_for_commit = &need_rot_per_commit[com_idx];
            debug_assert_eq!(need_rot_for_commit.len(), d.layout.mat_starts.len());
            for &(mat_idx, _col_idx, slice) in &d.layout.sorted_cols {
                let need_rot = need_rot_for_commit[mat_idx];
                let lambda_eq_idx = lambda_idx;
                let lambda_rot_idx = lambda_idx + 1;
                lambda_idx += 2;
                trace_views.push(TraceViewMeta {
                    com_idx,
                    slice,
                    lambda_eq_idx,
                    lambda_rot_idx,
                    need_rot,
                });
            }
        }
        let lambda_pows = lambda.powers().take(lambda_idx).collect_vec();

        let mut ht_diff_idxs = Vec::new();
        let mut eq_r_per_lht: HashMap<usize, ColMajorMatrix<EF>> = HashMap::new();
        let mut last_height = 0;
        for (i, tv) in trace_views.iter().enumerate() {
            let n_lift = tv.slice.log_height().saturating_sub(l_skip);
            if i == 0 || tv.slice.log_height() != last_height {
                ht_diff_idxs.push(i);
                last_height = tv.slice.log_height();
            }
            eq_r_per_lht
                .entry(tv.slice.log_height())
                .or_insert_with(|| ColMajorMatrix::new(evals_eq_hypercube(&r[1..1 + n_lift]), 1));
        }
        ht_diff_idxs.push(trace_views.len());

        let eq_const = eval_eq_uni_at_one(l_skip, r[0] * omega_skip);
        let eq_ub_per_trace = vec![EF::ONE; trace_views.len()];

        Self {
            l_skip,
            omega_skip,
            r_0: r[0],
            lambda_pows,
            eq_const,
            stacked_per_commit,
            trace_views,
            ht_diff_idxs,
            eq_r_per_lht,
            q_evals: vec![],
            k_rot_r_per_lht: HashMap::new(),
            eq_ub_per_trace,
        }
    }

    fn batch_sumcheck_uni_round0_poly(&mut self) -> UnivariatePoly<EF> {
        let l_skip = self.l_skip;
        let omega_skip = self.omega_skip;
        // +1 from eq term
        let s_0_deg = sumcheck_round0_deg(l_skip, 2);
        // We want to compute algebraic batching, via \lambda,
        // for each (T, j) pair of (trace, column) of the univariate polynomials
        // ```text
        // Z -> sum_{x in H_{n_stack}} q(Z,x) in_{D,n_T}(Z) eq_{D_{n_T}}((Z,x[..\tilde n_T]), r[..1+\tilde n_T]) eq(x[\tilde n_T..], b_{T,j})
        // Z -> sum_{x in H_{n_stack}} q(Z,x) in_{D,n_T}(Z) \kappa_{\rot, D_{n_T}}((Z,x[..\tilde n_T]), r[..1+\tilde n_T]) eq(x[\tilde n_T..], b_{T,j})
        // ```
        // where `b_{T,j}` is length `n_stack - n_T` binary encoding of `StackedSlice.row_idx >>
        // (l_skip
        // + n_T)`. Note that since x is in the hypercube, by definition of `eq` the above
        // simplifies to
        // ```text
        // Z -> sum_{x in H_{n_T}} q(Z,x,b_{T,j}) (in_{D,n_T}(Z) eq(Z,r_0)) eq(x[..\tilde n_T], r[1..1+\tilde n_T])
        // Z -> sum_{x in H_{n_T}} q(Z,x,b_{T,j}) in_{D,n_T}(Z) \kappa_rot((Z, x[..n_T]), (r_0, r[1..1+n_T]))
        // ```
        // where we also simplified the other `eq` term.
        // We further simplify the second using equation
        // ```text
        // \kappa_rot((Z, x[..n_T]), (r_0, r[1..1+n_T])) =
        // eq_D(Z,omega_D r_0) eq(x[..n_T], r[1..1+n_T]) + eq_D(Z,1)eq_D(omega_D r_0,1) ( kappa_rot(x[..n_T], r[1..1+n_T]) - eq(x[..n_T], r[1..1+n_T]) )
        // ```
        // We compute the last polynomial in our usual way, by considering `q(\vec Z, b_{T,j})` as a
        // prismalinear polynomial and using its evaluations on `D_{n_T}`.
        let s_0_polys: Vec<_> = self
            .ht_diff_idxs
            .par_windows(2)
            .flat_map(|window| {
                let t_window = &self.trace_views[window[0]..window[1]];
                let log_height = t_window[0].slice.log_height();
                let n = log_height as isize - l_skip as isize;
                let n_lift = n.max(0) as usize;
                let eq_rs = self.eq_r_per_lht.get(&log_height).unwrap().column(0);
                debug_assert_eq!(eq_rs.len(), 1 << n_lift);
                // Prepare the q subslice eval views
                let q_t_cols = t_window
                    .iter()
                    .map(|tv| {
                        debug_assert_eq!(tv.slice.log_height(), log_height);
                        let q = &self.stacked_per_commit[tv.com_idx].matrix;
                        let s = tv.slice;
                        let q_t_col = &q.column(s.col_idx)[s.row_idx..s.row_idx + s.len(l_skip)];
                        // NOTE: even if s.stride(l_skip) != 1, we use the full non-strided column
                        // subslice. The sumcheck will not depend on the values outside of the
                        // stride because of the `in_{D, n_T}` indicator below.
                        (ColMajorMatrixView::new(q_t_col, 1).into(), false)
                    })
                    .collect_vec();
                sumcheck_uni_round0_poly(l_skip, n_lift, 2, &q_t_cols, |z, x, evals| {
                    let eq_cube = eq_rs[x];
                    let (l, omega, r_uni) = if n.is_negative() {
                        (
                            l_skip.wrapping_add_signed(n),
                            omega_skip.exp_power_of_2(-n as usize),
                            self.r_0.exp_power_of_2(-n as usize),
                        )
                    } else {
                        (l_skip, omega_skip, self.r_0)
                    };
                    let ind = eval_in_uni(l_skip, n, z);
                    let eq_uni_r0 = eval_eq_uni(l, z.into(), r_uni);
                    let eq_uni_r0_rot = eval_eq_uni(l, z.into(), r_uni * omega);
                    // eq_uni_1, k_rot_cube are only used when n > 0
                    let eq_uni_1 = eval_eq_uni_at_one(l_skip, z);
                    let k_rot_cube = eq_rs[rot_prev(x, n_lift)];

                    let eq = eq_uni_r0 * eq_cube;
                    let k_rot =
                        eq_uni_r0_rot * eq_cube + self.eq_const * eq_uni_1 * (k_rot_cube - eq_cube);
                    zip(t_window, evals).fold([EF::ZERO; 2], |mut acc, (tv, eval)| {
                        let q = eval[0];
                        acc[0] += self.lambda_pows[tv.lambda_eq_idx] * eq * q * ind;
                        if tv.need_rot {
                            acc[1] += self.lambda_pows[tv.lambda_rot_idx] * k_rot * q * ind;
                        }
                        acc
                    })
                })
            })
            .collect();
        let s_0_coeffs = (0..=s_0_deg)
            .map(|i| s_0_polys.iter().map(|evals| evals.coeffs()[i]).sum::<EF>())
            .collect_vec();
        UnivariatePoly::new(s_0_coeffs)
    }

    fn fold_ple_evals(&mut self, u_0: EF) {
        let l_skip = self.l_skip;
        let r_0 = self.r_0;
        let omega_skip = self.omega_skip;
        self.q_evals = self
            .stacked_per_commit
            .iter()
            .map(|d| fold_ple_evals(l_skip, d.matrix.as_view().into(), false, u_0))
            .collect_vec();
        // fold PLEs into MLEs for \eq and \kappa_\rot, using u_0
        let eq_uni_u0r0 = eval_eq_uni(l_skip, u_0, r_0);
        let eq_uni_u0r0_rot = eval_eq_uni(l_skip, u_0, r_0 * omega_skip);
        let eq_uni_u01 = eval_eq_uni_at_one(l_skip, u_0);
        // \kappa_\rot(x, r) = eq(rot^{-1}(x), r)
        self.k_rot_r_per_lht = self
            .eq_r_per_lht
            .par_iter_mut()
            .map(|(&log_height, mat)| {
                let n = log_height as isize - l_skip as isize;
                let n_lift = n.max(0) as usize;
                debug_assert_eq!(mat.values.len(), 1 << n_lift);
                let ind = eval_in_uni(l_skip, n, u_0);
                let (eq_uni, eq_uni_rot) = if n.is_negative() {
                    let omega = omega_skip.exp_power_of_2(-n as usize);
                    let r = r_0.exp_power_of_2(-n as usize);
                    let l = l_skip.wrapping_add_signed(n);
                    (eval_eq_uni(l, u_0, r), eval_eq_uni(l, u_0, r * omega))
                } else {
                    (eq_uni_u0r0, eq_uni_u0r0_rot)
                };
                // folded \kappa_\rot evals
                let evals: Vec<_> = (0..1 << n_lift)
                    .into_par_iter()
                    .map(|x| {
                        let eq_cube = unsafe { *mat.get_unchecked(x, 0) };
                        let k_rot_cube = unsafe { *mat.get_unchecked(rot_prev(x, n_lift), 0) };
                        ind * (eq_uni_rot * eq_cube
                            + self.eq_const * eq_uni_u01 * (k_rot_cube - eq_cube))
                    })
                    .collect();
                // update \eq with the univariate factor:
                mat.values.par_iter_mut().for_each(|v| {
                    *v *= ind * eq_uni;
                });
                (log_height, ColMajorMatrix::new(evals, 1))
            })
            .collect();
    }

    fn batch_sumcheck_poly_eval(&mut self, round: usize, _u_prev: EF) -> [EF; 2] {
        let l_skip = self.l_skip;
        let s_deg = 2;
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
        //      = q(u[..round], X, b_{T,j}[round-n_T..]) eq((u[..1+n_T], r[..1+n_T]) eq((u[1+n_T..round],X), b_{T,j}[..round-n_T])
        // X -> sum_{y in H_{n_stack-round}} q(u[..round],X,y) \kappa_\rot(u[..1+n_T], r[..1+n_T]) eq((u[1+n_T..round],X,y[round..]), b_{T,j})
        // ```
        let s_evals: Vec<_> = self
            .ht_diff_idxs
            .par_windows(2)
            .flat_map(|window| {
                let t_views = &self.trace_views[window[0]..window[1]];
                let log_height = t_views[0].slice.log_height();
                let n_lift = log_height.saturating_sub(l_skip); // \tilde{n}_T
                let hypercube_dim = n_lift.saturating_sub(round);
                let eq_rs = self.eq_r_per_lht.get(&log_height).unwrap().column(0);
                let k_rot_rs = self.k_rot_r_per_lht.get(&log_height).unwrap().column(0);
                debug_assert_eq!(eq_rs.len(), 1 << n_lift.saturating_sub(round - 1));
                debug_assert_eq!(k_rot_rs.len(), 1 << n_lift.saturating_sub(round - 1));
                // Prepare the q subslice eval views
                let t_cols = t_views
                    .iter()
                    .map(|tv| {
                        debug_assert_eq!(tv.slice.log_height(), log_height);
                        // q(u[..round], X, b_{T,j}[round-\tilde n_T..])
                        // q_evals has been folded already
                        let q = &self.q_evals[tv.com_idx];
                        let s = tv.slice;
                        let row_start = if round <= n_lift {
                            // round >= 1 so n_lift >= 1
                            (s.row_idx >> log_height) << (hypercube_dim + 1)
                        } else {
                            (s.row_idx >> (l_skip + round)) << 1
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
                            let tv = &self.trace_views[t_idx];
                            let q = eval[0];
                            let mut eq_ub = self.eq_ub_per_trace[t_idx];
                            let (eq, k_rot) = if round > n_lift {
                                // Extra contribution of eq(X, b_{T,j}[round-n_T-1])
                                let b = (tv.slice.row_idx >> (l_skip + round - 1)) & 1;
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
                            acc[0] += self.lambda_pows[tv.lambda_eq_idx] * q * eq;
                            if tv.need_rot {
                                acc[1] += self.lambda_pows[tv.lambda_rot_idx] * q * k_rot;
                            }
                            acc
                        })
                })
            })
            .collect();
        from_fn(|i| s_evals.iter().map(|evals| evals[i]).sum::<EF>())
    }

    fn fold_mle_evals(&mut self, round: usize, u_round: EF) {
        let l_skip = self.l_skip;
        self.q_evals = batch_fold_mle_evals(take(&mut self.q_evals), u_round);
        self.eq_r_per_lht = take(&mut self.eq_r_per_lht)
            .into_par_iter()
            .map(|(lht, mat)| (lht, fold_mle_evals(mat, u_round)))
            .collect();
        self.k_rot_r_per_lht = take(&mut self.k_rot_r_per_lht)
            .into_par_iter()
            .map(|(lht, mat)| (lht, fold_mle_evals(mat, u_round)))
            .collect();
        for (tv, eq_ub) in zip(&self.trace_views, &mut self.eq_ub_per_trace) {
            let s = tv.slice;
            let n_lift = s.log_height().saturating_sub(l_skip);
            if round > n_lift {
                // Folding above did nothing, and we update the eq(u[1+n_T..=round],
                // b_{T,j}[..=round-n_T-1]) value
                let b = (s.row_idx >> (l_skip + round - 1)) & 1;
                *eq_ub *= eval_eq_mle(&[u_round], &[F::from_bool(b == 1)]);
            }
        }
    }

    fn into_stacked_openings(self) -> Vec<Vec<EF>> {
        self.q_evals
            .into_iter()
            .map(|q| {
                debug_assert_eq!(q.height(), 1);
                q.values
            })
            .collect()
    }
}

/// `x_int` is the integer representation of point on H_n.
fn rot_prev(x_int: usize, n: usize) -> usize {
    debug_assert!(x_int < (1 << n));
    if x_int == 0 {
        (1 << n) - 1
    } else {
        x_int - 1
    }
}
