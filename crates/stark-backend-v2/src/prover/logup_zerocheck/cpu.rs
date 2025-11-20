use std::{
    cmp::max,
    iter::{self, zip},
    mem::take,
};

use itertools::Itertools;
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_variable::Entry, SymbolicConstraints, SymbolicExpressionNode,
    },
    parizip,
    prover::MatrixDimensions,
};
use p3_field::{Field, FieldAlgebra, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::debug;

use crate::{
    poly_common::{eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni, UnivariatePoly},
    poseidon2::sponge::FiatShamirTranscript,
    prover::{
        fractional_sumcheck_gkr::{fractional_sumcheck, Frac, FracSumcheckProof},
        logup_zerocheck::EvalHelper,
        poly::evals_eq_hypercube,
        stacked_pcs::StackedLayout,
        sumcheck::{
            batch_fold_mle_evals, batch_fold_ple_evals, fold_ple_evals, sumcheck_round0_deg,
            sumcheck_round_poly_evals, sumcheck_uni_round0_poly,
        },
        ColMajorMatrix, CpuBackendV2, CpuDeviceV2, DeviceMultiStarkProvingKeyV2,
        LogupZerocheckProver, MatrixView, ProverBackendV2, ProvingContextV2,
    },
    EF, F,
};

pub struct LogupZerocheckCpu<'a> {
    pub alpha_logup: EF,
    pub beta_pows: Vec<EF>,

    pub l_skip: usize,
    pub n_logup: usize,
    pub n_max: usize,

    pub omega_skip: F,
    pub omega_skip_pows: Vec<F>,

    pub interactions_layout: StackedLayout,
    eval_helpers: Vec<EvalHelper<'a, F>>,
    /// Max constraint degree across constraints and interactions
    pub constraint_degree: usize,
    pub n_per_trace: Vec<isize>,
    max_num_constraints: usize,
    s_deg: usize,

    // Available after GKR:
    pub xi: Vec<EF>,
    lambda_pows: Vec<EF>,

    pub eq_xi_per_trace: Vec<ColMajorMatrix<EF>>,
    pub eq_sharp_per_trace: Vec<ColMajorMatrix<EF>>,
    eq_3b_per_trace: Vec<Vec<EF>>,
    // TODO[jpw]: delete these
    sels_per_trace_base: Vec<ColMajorMatrix<F>>,
    // After univariate round 0:
    pub mat_evals_per_trace: Vec<Vec<ColMajorMatrix<EF>>>,
    pub sels_per_trace: Vec<ColMajorMatrix<EF>>,
    // Stores \hat{f}(\vec r_n) * r_{n+1} .. r_{round-1} for polys f that are "done" in the batch
    // sumcheck
    zerocheck_tilde_evals: Vec<EF>,
    logup_tilde_evals: Vec<[EF; 2]>,
}

impl<'a, TS> LogupZerocheckProver<'a, CpuBackendV2, CpuDeviceV2, TS> for LogupZerocheckCpu<'a>
where
    TS: FiatShamirTranscript,
{
    fn prove_logup_gkr(
        _device: &'a CpuDeviceV2,
        transcript: &mut TS,
        pk: &'a DeviceMultiStarkProvingKeyV2<CpuBackendV2>,
        ctx: &ProvingContextV2<CpuBackendV2>,
        _common_main_data: &'a <CpuBackendV2 as ProverBackendV2>::PcsData,
        n_logup: usize,
        interactions_layout: StackedLayout,
        alpha_logup: EF,
        beta_logup: EF,
    ) -> (Self, FracSumcheckProof<EF>) {
        let l_skip = pk.params.l_skip;
        let omega_skip = F::two_adic_generator(l_skip);
        let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();
        let num_airs_present = ctx.per_trace.len();

        let constraint_degree = pk.max_constraint_degree;
        // +1 from eq term
        let s_deg = constraint_degree + 1;
        let max_interaction_length = ctx
            .per_trace
            .iter()
            .flat_map(|(air_idx, _)| {
                pk.per_air[*air_idx]
                    .vk
                    .symbolic_constraints
                    .interactions
                    .iter()
                    .map(|i| i.message.len())
            })
            .max()
            .unwrap_or(0);
        let beta_pows = beta_logup
            .powers()
            .take(max_interaction_length + 1)
            .collect_vec();

        let n_per_trace: Vec<isize> = ctx
            .common_main_traces()
            .map(|(_, t)| log2_strict_usize(t.height()) as isize - l_skip as isize)
            .collect_vec();
        let n_max: usize = n_per_trace[0].max(0) as usize;

        let eval_helpers: Vec<EvalHelper<F>> = ctx
            .per_trace
            .iter()
            .map(|(air_idx, air_ctx)| {
                let pk = &pk.per_air[*air_idx];
                let constraints = &pk.vk.symbolic_constraints.constraints;
                let public_values = air_ctx.public_values.clone();
                let preprocessed_trace =
                    pk.preprocessed_data.as_ref().map(|cd| cd.data.mat_view(0));
                let partitioned_main_trace = air_ctx
                    .cached_mains
                    .iter()
                    .map(|cd| cd.data.mat_view(0))
                    .chain(iter::once(air_ctx.common_main.as_view().into()))
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
                EvalHelper {
                    constraints_dag: &pk.vk.symbolic_constraints.constraints,
                    interactions: symbolic_constraints.interactions,
                    public_values,
                    preprocessed_trace,
                    needs_next,
                }
            })
            .collect(); // end of preparation / loading of constraints
        let max_num_constraints = pk
            .per_air
            .iter()
            .map(|pk| pk.vk.symbolic_constraints.constraints.constraint_idx.len())
            .max()
            .unwrap_or(0);

        // Compute logup input layer: these are the evaluations of \hat{p}, \hat{q} on the hypercube
        // `H_{l_skip + n_logup}`
        let has_interactions = !interactions_layout.sorted_cols.is_empty();
        let gkr_input_evals = if !has_interactions {
            vec![]
        } else {
            // Per trace, a row major matrix of interaction evaluations
            // NOTE: these are the evaluations _without_ lifting
            // PERF[jpw]: we should write directly to the stacked `evals` in memory below
            let unstacked_interaction_evals = eval_helpers
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
                            helper.eval_interactions(&row_parts, &beta_pows)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut evals = vec![Frac::default(); 1 << (l_skip + n_logup)];
            for (trace_idx, interaction_idx, s) in interactions_layout.sorted_cols.iter().copied() {
                let pq_evals = &unstacked_interaction_evals[trace_idx];
                let height = pq_evals.len();
                debug_assert_eq!(s.col_idx, 0);
                // the interactions layout has internal striding threshold=0
                debug_assert_eq!(1 << s.log_height(), s.len(0));
                debug_assert_eq!(s.len(0) % height, 0);
                let norm_factor_denom = s.len(0) / height;
                let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
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

        // GKR
        let (frac_sum_proof, mut xi) = fractional_sumcheck(transcript, &gkr_input_evals, true);

        // Sample more for `\xi` in the edge case that some AIRs don't have interactions
        let n_global = max(n_max, n_logup);
        debug!(%n_global);
        while xi.len() != l_skip + n_global {
            xi.push(transcript.sample_ext());
        }
        debug!(?xi);
        // we now have full \xi vector

        let zerocheck_tilde_evals = vec![EF::ZERO; num_airs_present];
        let logup_tilde_evals = vec![[EF::ZERO; 2]; num_airs_present];
        let prover = Self {
            alpha_logup,
            beta_pows,
            l_skip,
            n_logup,
            n_max,
            omega_skip,
            omega_skip_pows,
            interactions_layout,
            constraint_degree,
            max_num_constraints,
            s_deg,
            n_per_trace,
            eval_helpers,
            xi,
            lambda_pows: vec![],
            sels_per_trace_base: vec![],
            eq_xi_per_trace: vec![],
            eq_sharp_per_trace: vec![],
            eq_3b_per_trace: vec![],
            mat_evals_per_trace: vec![],
            sels_per_trace: vec![],
            zerocheck_tilde_evals,
            logup_tilde_evals,
        };
        (prover, frac_sum_proof)
    }

    fn sumcheck_uni_round0_polys(
        &mut self,
        ctx: &ProvingContextV2<CpuBackendV2>,
        lambda: EF,
    ) -> Vec<UnivariatePoly<EF>> {
        let n_logup = self.n_logup;
        let l_skip = self.l_skip;
        let xi = &self.xi;
        self.lambda_pows = lambda.powers().take(self.max_num_constraints).collect_vec();
        let s_deg = self.s_deg;
        let s_0_deg = sumcheck_round0_deg(l_skip, s_deg);

        // For each trace, for each interaction \hat\sigma, the eq(ξ_3,b_{T,\hat\sigma}) term.
        // This is some weight per interaction that does not depend on the row.
        self.eq_3b_per_trace = self
            .eval_helpers
            .par_iter()
            .zip(&self.n_per_trace)
            .enumerate()
            .map(|(trace_idx, (helper, &n))| {
                // Everything for logup is done with respect to lifted traces
                // Note: `n_lift = \tilde{n}` from the paper
                let n_lift = n.max(0) as usize;
                if helper.interactions.is_empty() {
                    return vec![];
                }
                let mut b_vec = vec![F::ZERO; n_logup - n_lift];
                (0..helper.interactions.len())
                    .map(|i| {
                        // PERF[jpw]: interactions_layout.get is linear
                        let stacked_idx =
                            self.interactions_layout.get(trace_idx, i).unwrap().row_idx;
                        debug_assert!(stacked_idx.trailing_zeros() as usize >= n_lift + l_skip);
                        let mut b_int = stacked_idx >> (l_skip + n_lift);
                        for b in &mut b_vec {
                            *b = F::from_bool(b_int & 1 == 1);
                            b_int >>= 1;
                        }
                        eval_eq_mle(&xi[l_skip + n_lift..l_skip + n_logup], &b_vec)
                    })
                    .collect_vec()
            })
            .collect::<Vec<_>>();

        // PERF[jpw]: make Hashmap from unique n -> eq_n(xi, -)
        // NOTE: this is evaluations of `x -> eq_{H_{\tilde n}}(x, \xi[l_skip..l_skip + \tilde n])`
        // on hypercube `H_{\tilde n}`. We store the univariate component eq_D separately as
        // an optimization.
        self.eq_xi_per_trace = self
            .n_per_trace
            .par_iter()
            .map(|&n| {
                let n_lift = n.max(0) as usize;
                // PERF[jpw]: might be able to share computations between eq_xi, eq_sharp
                // computations the eq(xi, -) evaluations on hyperprism for
                // zerocheck
                let eq_xi = evals_eq_hypercube(&xi[l_skip..l_skip + n_lift]);
                ColMajorMatrix::new(eq_xi, 1)
            })
            .collect();

        // For each trace, create selectors as a 3-column matrix of _the lifts of_ [is_first,
        // is_transition, is_last]
        //
        // PERF[jpw]: I think it's better to not save these and just
        // interpolate directly using the formulas for selectors
        self.sels_per_trace_base = self
            .n_per_trace
            .iter()
            .map(|&n| {
                let log_height = l_skip.checked_add_signed(n).unwrap();
                let height = 1 << log_height;
                let lifted_height = height.max(1 << l_skip);
                let mut mat = F::zero_vec(3 * lifted_height);
                mat[lifted_height..2 * lifted_height].fill(F::ONE);
                for i in (0..lifted_height).step_by(height) {
                    mat[i] = F::ONE; // is_first
                    mat[lifted_height + i + height - 1] = F::ZERO; // is_transition
                    mat[2 * lifted_height + i + height - 1] = F::ONE; // is_last
                }
                ColMajorMatrix::new(mat, 3)
            })
            .collect_vec();

        // PERF[jpw]: see Gruen, Section 3.2 and 4 on some ways to reduce the degree of the
        // univariate polynomial. We know s_0 is supposed to vanish on univariate skip
        // domain `D` of size `2^{l_skip}`. Hence `s_0 = Z_D * s'_0(Z)` where `Z_D =
        // \prod_{z in D} (Z - z)` where `s'_0` has degree `d * (2^{l_skip} - 1) - 1`. We
        // can evaluate s'_0 on a coset and then perform coset iDFT to get `s'_0`
        // coefficients. We need to use a coset to ensure disjointness from `D`.
        let s_0_zerochecks = self
            .eval_helpers
            .par_iter()
            .enumerate()
            .map(|(trace_idx, helper)| {
                let trace_ctx = &ctx.per_trace[trace_idx].1;
                let n_lift = log2_strict_usize(trace_ctx.height()).saturating_sub(l_skip);
                let mats = &helper.view_mats(trace_ctx);
                let eq_xi = self.eq_xi_per_trace[trace_idx].column(0);
                let sels = self.sels_per_trace_base[trace_idx].as_view();
                let mut parts = vec![(sels.into(), false)];
                parts.extend_from_slice(mats);

                // degree is constraint_degree + 1 due to eq term
                let [s_0] =
                    sumcheck_uni_round0_poly(l_skip, n_lift, s_deg, &parts, |z, x, row_parts| {
                        // PERF[jpw]: we are limited by the closure interface but `eq_uni` should be
                        // cached
                        let eq = eval_eq_uni(l_skip, xi[0], z.into()) * eq_xi[x];
                        let constraint_eval = helper.acc_constraints(row_parts, &self.lambda_pows);
                        [eq * constraint_eval]
                    });
                debug_assert_eq!(
                    s_0.coeffs()
                        .iter()
                        .step_by(1 << l_skip)
                        .copied()
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
        let s_0_logups = self
            .eval_helpers
            .par_iter()
            .enumerate()
            .flat_map(|(trace_idx, helper)| {
                if helper.interactions.is_empty() {
                    return [(); 2].map(|_| UnivariatePoly::new(vec![EF::ZERO; s_0_deg + 1]));
                }
                let trace_ctx = &ctx.per_trace[trace_idx].1;
                let log_height = log2_strict_usize(trace_ctx.height());
                let n_lift = log_height.saturating_sub(l_skip);
                let mats = &helper.view_mats(trace_ctx);
                let eq_xi = self.eq_xi_per_trace[trace_idx].column(0);
                let eq_3bs = &self.eq_3b_per_trace[trace_idx];
                let sels = self.sels_per_trace_base[trace_idx].as_view();
                let mut parts = vec![(sels.into(), false)];
                parts.extend_from_slice(mats);
                let norm_factor_denom = 1 << l_skip.saturating_sub(log_height);
                let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();

                // degree is constraint_degree + 1 due to eq term
                let [mut numer, denom] =
                    sumcheck_uni_round0_poly(l_skip, n_lift, s_deg, &parts, |z, x, row_parts| {
                        let eq_sharp =
                            eval_eq_sharp_uni(&self.omega_skip_pows, &xi[..l_skip], z.into())
                                * eq_xi[x];
                        let [numer, denom] =
                            helper.acc_interactions(row_parts, &self.beta_pows, eq_3bs);
                        [eq_sharp * numer, eq_sharp * denom]
                    });
                for p in numer.coeffs_mut() {
                    *p *= norm_factor;
                }
                [numer, denom]
            })
            .collect::<Vec<_>>();

        s_0_logups.into_iter().chain(s_0_zerochecks).collect()
    }

    fn fold_ple_evals(&mut self, ctx: ProvingContextV2<CpuBackendV2>, r_0: EF) {
        let l_skip = self.l_skip;
        // "Fold" all PLE evaluations by interpolating and evaluating at `r_0`.
        // NOTE: after this folding, \hat{T} and \hat{T_{rot}} will be treated as completely
        // distinct matrices.
        self.mat_evals_per_trace = self
            .eval_helpers
            .par_iter()
            .zip(ctx.per_trace.into_par_iter())
            .map(|(helper, (_, trace_ctx))| {
                let mats = helper.view_mats(&trace_ctx);
                mats.into_par_iter()
                    .map(|(mat, is_rot)| fold_ple_evals(l_skip, mat, is_rot, r_0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.sels_per_trace =
            batch_fold_ple_evals(l_skip, take(&mut self.sels_per_trace_base), false, r_0);
        let eq_r0 = eval_eq_uni(l_skip, self.xi[0], r_0);
        let eq_sharp_r0 = eval_eq_sharp_uni(&self.omega_skip_pows, &self.xi[..l_skip], r_0);
        // Define eq^\sharp_D(xi[0], r0) * eq_{H_n}(xi[1..1+n], x) and also update eq_D(xi[0], r0) *
        // eq_{H_n}(xi[1..1+n], x)
        self.eq_sharp_per_trace = self
            .eq_xi_per_trace
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
    }

    fn sumcheck_polys_eval(&mut self, round: usize, r_prev: EF) -> Vec<Vec<EF>> {
        // PERF[jpw]: use per AIR s_deg
        let s_deg = self.s_deg;
        let s_zerocheck_evals: Vec<Vec<EF>> = parizip!(
            &self.eval_helpers,
            &mut self.zerocheck_tilde_evals,
            &self.n_per_trace,
            &self.mat_evals_per_trace,
            &self.sels_per_trace,
            &self.eq_xi_per_trace
        )
        .map(|(helper, tilde_eval, &n, mats, sels, eq_xi)| {
            let n_lift = n.max(0) as usize;
            if round > n_lift {
                if round == n_lift + 1 {
                    // Evaluate \hat{f}(\vec r_n)
                    let parts = iter::once(sels)
                        .chain(mats)
                        .map(|mat| mat.columns().map(|c| c[0]).collect_vec())
                        .collect_vec();
                    *tilde_eval =
                        eq_xi.column(0)[0] * helper.acc_constraints(&parts, &self.lambda_pows);
                } else {
                    *tilde_eval *= r_prev;
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
                    n_lift - (round - 1),
                    s_deg,
                    &parts,
                    |_x, _y, row_parts| {
                        let eq = row_parts[0][0];
                        let constraint_eval =
                            helper.acc_constraints(&row_parts[1..], &self.lambda_pows);
                        [eq * constraint_eval]
                    },
                );
                s
            }
        })
        .collect();

        let s_logup_evals: Vec<Vec<EF>> = parizip!(
            &self.eval_helpers,
            &mut self.logup_tilde_evals,
            &self.n_per_trace,
            &self.mat_evals_per_trace,
            &self.sels_per_trace,
            &self.eq_sharp_per_trace,
            &self.eq_3b_per_trace
        )
        .flat_map(|(helper, tilde_eval, &n, mats, sels, eq_sharp, eq_3bs)| {
            if helper.interactions.is_empty() {
                return [vec![EF::ZERO; s_deg], vec![EF::ZERO; s_deg]];
            }
            let n_lift = n.max(0) as usize;
            let norm_factor_denom = 1 << (-n).max(0);
            let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
            if round > n_lift {
                if round == n_lift + 1 {
                    // Evaluate \hat{f}(\vec r_n)
                    let parts = iter::once(sels)
                        .chain(mats)
                        .map(|mat| mat.columns().map(|c| c[0]).collect_vec())
                        .collect_vec();
                    let eq = eq_sharp.column(0)[0];
                    *tilde_eval = helper
                        .acc_interactions(&parts, &self.beta_pows, eq_3bs)
                        .map(|x| eq * x);
                    tilde_eval[0] *= norm_factor;
                } else {
                    for x in tilde_eval.iter_mut() {
                        *x *= r_prev;
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
                let [mut numer, denom] = sumcheck_round_poly_evals(
                    n_lift - (round - 1),
                    s_deg,
                    &parts,
                    |_x, _y, row_parts| {
                        let eq_sharp = row_parts[0][0];
                        helper
                            .acc_interactions(&row_parts[1..], &self.beta_pows, eq_3bs)
                            .map(|eval| eq_sharp * eval)
                    },
                );
                for p in &mut numer {
                    *p *= norm_factor;
                }
                [numer, denom]
            }
        })
        .collect();

        s_logup_evals.into_iter().chain(s_zerocheck_evals).collect()
    }

    fn fold_mle_evals(&mut self, _round: usize, r_round: EF) {
        self.mat_evals_per_trace = take(&mut self.mat_evals_per_trace)
            .into_iter()
            .map(|mats| batch_fold_mle_evals(mats, r_round))
            .collect_vec();
        self.sels_per_trace = batch_fold_mle_evals(take(&mut self.sels_per_trace), r_round);
        self.eq_xi_per_trace = batch_fold_mle_evals(take(&mut self.eq_xi_per_trace), r_round);
        self.eq_sharp_per_trace = batch_fold_mle_evals(take(&mut self.eq_sharp_per_trace), r_round);

        #[allow(unused_variables)]
        #[cfg(debug_assertions)]
        if tracing::enabled!(tracing::Level::DEBUG) && _round == self.n_max {
            use itertools::izip;

            for (trace_idx, (helper, &n, mats, sels, eq_xi)) in izip!(
                &self.eval_helpers,
                &self.n_per_trace,
                &self.mat_evals_per_trace,
                &self.sels_per_trace,
                &self.eq_xi_per_trace
            )
            .enumerate()
            {
                let parts = iter::once(sels)
                    .chain(mats)
                    .map(|mat| mat.columns().map(|c| c[0]).collect_vec())
                    .collect_vec();
                let expr = helper.acc_constraints(&parts, &self.lambda_pows);
                debug!(%trace_idx, %expr, "constraints_eval");
            }

            for (trace_idx, (helper, &n, mats, sels, eq_sharp, eq_3bs)) in izip!(
                &self.eval_helpers,
                &self.n_per_trace,
                &self.mat_evals_per_trace,
                &self.sels_per_trace,
                &self.eq_sharp_per_trace,
                &self.eq_3b_per_trace
            )
            .enumerate()
            {
                if helper.interactions.is_empty() {
                    continue;
                }
                let parts = iter::once(sels)
                    .chain(mats)
                    .map(|mat| mat.columns().map(|c| c[0]).collect_vec())
                    .collect_vec();
                let [num, denom] = helper.acc_interactions(&parts, &self.beta_pows, eq_3bs);

                debug!(%trace_idx, %num, %denom, "interactions_eval");
            }
        }
    }

    fn into_column_openings(mut self) -> Vec<Vec<Vec<(EF, EF)>>> {
        let num_airs_present = self.mat_evals_per_trace.len();
        let mut column_openings = Vec::with_capacity(num_airs_present);
        // At the end, we've folded all MLEs so they only have one row equal to evaluation at `\vec
        // r`.
        for mut mat_evals in take(&mut self.mat_evals_per_trace) {
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
                            (claim[0], claim_rot[0])
                        })
                        .collect_vec()
                })
                .collect_vec();
            column_openings.push(openings_of_air);
        }
        column_openings
    }
}
