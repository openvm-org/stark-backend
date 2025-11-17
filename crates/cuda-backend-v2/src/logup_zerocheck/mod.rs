use std::{
    cmp::max,
    iter::{self},
    sync::Arc,
};

use itertools::{Itertools, izip};
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::{air_builders::symbolic::SymbolicConstraints, prover::MatrixDimensions};
use p3_field::{Field, FieldAlgebra, TwoAdicField};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use stark_backend_v2::{
    poly_common::{UnivariatePoly, eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni},
    poseidon2::sponge::FiatShamirTranscript,
    prover::{
        ColMajorMatrix, DeviceMultiStarkProvingKeyV2, LogupZerocheckProver, ProvingContextV2,
        fractional_sumcheck_gkr::FracSumcheckProof, poly::evals_eq_hypercube,
        stacked_pcs::StackedLayout, sumcheck::sumcheck_round0_deg,
    },
};
use tracing::{debug, instrument};

use crate::{
    Digest, EF, F, GpuBackendV2, GpuDeviceV2,
    cuda::{
        logup_zerocheck::{MainMatrixPtrs, interpolate_columns_gpu},
        sumcheck::fold_mle,
    },
    gpu_backend::transport_matrix_d2h_col_major,
    logup_zerocheck::{fold_ple::fold_ple_mixed_rotate, round0::prepare_round0_trace_input},
    stacked_pcs::StackedPcsDataGpu,
};

mod dag_scheduling;
mod errors;
mod fold_ple;
/// Fraction sumcheck via GKR
mod fractional;
/// Logup interaction evaluations for GKR input
mod gkr_input;
mod mle_round;
mod round0;
mod state;

pub use errors::*;
use fold_ple::{compute_eq_sharp_gpu, fold_ple_evals_gpu};
use fractional::{fractional_sumcheck_gpu, initialize_segment_tree};
use gkr_input::{collect_trace_interactions, log_gkr_input_evals};
use mle_round::{evaluate_mle_constraints_gpu, evaluate_mle_interactions_gpu};
use round0::{evaluate_round0_constraints_gpu, evaluate_round0_interactions_gpu};
use state::{FractionalGkrState, Round0Buffers};

#[allow(dead_code)]
pub struct LogupZerocheckGpu<'a> {
    pub alpha_logup: EF,
    pub beta_pows: Vec<EF>,

    pub l_skip: usize,
    n_logup: usize,
    n_max: usize,
    n_global: usize,

    pub omega_skip: F,
    pub omega_skip_pows: Vec<F>,

    pub interactions_layout: StackedLayout,
    pub constraint_degree: usize,
    n_per_trace: Vec<isize>,
    max_num_constraints: usize,
    s_deg: usize,
    // Available after GKR:
    pub xi: Vec<EF>,
    pub lambda_pows: Option<DeviceBuffer<EF>>,

    eq_xi_per_trace: Vec<DeviceMatrix<EF>>,
    eq_sharp_per_trace: Vec<DeviceMatrix<EF>>,
    eq_3b_per_trace: Vec<DeviceBuffer<EF>>,
    // TODO: ask jpw why we need to delete these
    sels_per_trace_base: Vec<DeviceMatrix<F>>,
    // After univariate round 0:
    mat_evals_per_trace: Vec<Vec<DeviceMatrix<EF>>>,
    sels_per_trace: Vec<DeviceMatrix<EF>>,
    // Store public_values per trace (similar to CPU's EvalHelper)
    public_values_per_trace: Vec<DeviceBuffer<F>>,
    air_indices_per_trace: Vec<usize>,
    zerocheck_tilde_evals: Vec<EF>,
    logup_tilde_evals: Vec<[EF; 2]>,

    fractional_state: FractionalGkrState, //?
    // round0: Round0Buffers,
    pk: &'a DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    device: &'a GpuDeviceV2,
    common_main_pcs_data: &'a StackedPcsDataGpu<F, Digest>,
}

impl<'a, TS> LogupZerocheckProver<'a, GpuBackendV2, GpuDeviceV2, TS> for LogupZerocheckGpu<'a>
where
    TS: FiatShamirTranscript,
{
    // TODO: modify trait so we can drop common main buffers in `ctx`: they are extraneous given
    // `common_main_pcs_data`
    #[instrument(skip_all)]
    fn prove_logup_gkr(
        device: &'a GpuDeviceV2,
        transcript: &mut TS,
        pk: &'a DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        ctx: &ProvingContextV2<GpuBackendV2>,
        common_main_pcs_data: &'a StackedPcsDataGpu<F, Digest>,
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
        let s_deg = constraint_degree + 1;

        let max_interaction_length = pk
            .per_air
            .iter()
            .flat_map(|air_pk| air_pk.vk.symbolic_constraints.interactions.iter())
            .map(|interaction| interaction.message.len())
            .max()
            .unwrap_or(0);
        let beta_pows = beta_logup
            .powers()
            .take(max_interaction_length + 1)
            .collect_vec();

        let n_per_trace: Vec<isize> = ctx
            .common_main_traces()
            .map(|(_, t)| log2_strict_usize(t.height()) as isize - l_skip as isize)
            .collect();
        let n_max = n_per_trace[0].max(0) as usize;

        let max_num_constraints = pk
            .per_air
            .iter()
            .map(|air_pk| {
                air_pk
                    .vk
                    .symbolic_constraints
                    .constraints
                    .constraint_idx
                    .len()
            })
            .max()
            .unwrap_or(0);

        let has_interactions = !interactions_layout.sorted_cols.is_empty();

        // Collect interaction metadata for GPU execution (evaluations still run on CPU for now).
        let trace_interactions = collect_trace_interactions(pk, ctx, &interactions_layout);
        let mut fractional_state = FractionalGkrState {
            trace_interactions,
            ..Default::default()
        };

        let total_leaves = 1 << (l_skip + n_logup);
        let leaves = if has_interactions {
            log_gkr_input_evals(
                &mut fractional_state,
                pk,
                ctx,
                l_skip,
                alpha_logup,
                &beta_pows,
                total_leaves,
            )
            .expect("failed to evaluate interactions on device")
        } else {
            DeviceBuffer::new()
        };

        initialize_segment_tree(&mut fractional_state, leaves)
            .expect("failed to build fractional segment tree on device");

        let (frac_sum_proof, mut xi) = fractional_sumcheck_gpu(transcript, &fractional_state, true)
            .expect("failed to run fractional sumcheck on GPU");
        // Drop segment tree buffers
        fractional_state.input_evals.take();
        fractional_state.segment_tree.take();

        let n_global = max(n_max, n_logup);
        debug!(%n_global);
        while xi.len() != l_skip + n_global {
            xi.push(transcript.sample_ext());
        }
        debug!(?xi);

        let prover = Self {
            alpha_logup,
            beta_pows,
            l_skip,
            n_logup,
            n_max,
            n_global,
            omega_skip,
            omega_skip_pows,
            interactions_layout,
            constraint_degree,
            n_per_trace,
            max_num_constraints,
            s_deg,
            xi,
            lambda_pows: None,
            eq_xi_per_trace: vec![],
            eq_sharp_per_trace: vec![],
            eq_3b_per_trace: vec![],
            sels_per_trace_base: vec![],
            mat_evals_per_trace: vec![],
            sels_per_trace: vec![],
            public_values_per_trace: ctx
                .per_trace
                .iter()
                .map(|(_, air_ctx)| {
                    if air_ctx.public_values.is_empty() {
                        DeviceBuffer::new()
                    } else {
                        air_ctx
                            .public_values
                            .to_device()
                            .expect("failed to copy public values to device")
                    }
                })
                .collect_vec(),
            air_indices_per_trace: ctx
                .per_trace
                .iter()
                .map(|(air_idx, _)| *air_idx)
                .collect_vec(),
            zerocheck_tilde_evals: vec![EF::ZERO; num_airs_present],
            logup_tilde_evals: vec![[EF::ZERO; 2]; num_airs_present],
            fractional_state,
            pk,
            device,
            common_main_pcs_data,
        };

        (prover, frac_sum_proof)
    }

    #[instrument(
        name = "LogupZerocheck::sumcheck_uni_round0_polys",
        level = "debug",
        skip_all
    )]
    fn sumcheck_uni_round0_polys(
        &mut self,
        ctx: &ProvingContextV2<GpuBackendV2>,
        lambda: EF,
    ) -> Vec<UnivariatePoly<EF>> {
        let n_logup = self.n_logup;
        let l_skip = self.l_skip;
        let xi = &self.xi;
        let h_lambda_pows = lambda.powers().take(self.max_num_constraints).collect_vec();
        self.lambda_pows = Some(if !h_lambda_pows.is_empty() {
            h_lambda_pows
                .to_device()
                .expect("failed to copy lambda powers to device")
        } else {
            DeviceBuffer::new()
        });
        let s_deg = self.s_deg;
        let s_0_deg = sumcheck_round0_deg(l_skip, s_deg);
        let num_present_airs = ctx.per_trace.len();
        debug_assert_eq!(num_present_airs, self.n_per_trace.len());

        self.eq_3b_per_trace = self
            .fractional_state
            .trace_interactions
            .iter()
            .enumerate()
            .map(|(trace_idx, maybe_meta)| {
                if let Some(meta) = maybe_meta {
                    let n = self.n_per_trace[trace_idx];
                    let n_lift = n.max(0) as usize;
                    let mut b_vec = vec![F::ZERO; n_logup - n_lift];
                    let mut weights = Vec::with_capacity(meta.interactions.len());
                    for (interaction_idx, _) in meta.interactions.iter().enumerate() {
                        let stacked_idx = self
                            .interactions_layout
                            .get(trace_idx, interaction_idx)
                            .unwrap()
                            .row_idx;
                        let mut b_int = stacked_idx >> (self.l_skip + n_lift);
                        for bit in &mut b_vec {
                            *bit = F::from_bool(b_int & 1 == 1);
                            b_int >>= 1;
                        }
                        let weight = eval_eq_mle(
                            &self.xi[self.l_skip + n_lift..self.l_skip + self.n_logup],
                            &b_vec,
                        );
                        weights.push(weight);
                    }
                    if !weights.is_empty() {
                        weights
                            .to_device()
                            .expect("failed to copy eq_3b_per_trace to device")
                    } else {
                        DeviceBuffer::new()
                    }
                } else {
                    DeviceBuffer::new()
                }
            })
            .collect_vec();

        self.eq_xi_per_trace = self
            .n_per_trace
            .iter()
            .map(|&n| {
                let n_lift = n.max(0) as usize;
                let h_eq_xi = evals_eq_hypercube(&xi[l_skip..l_skip + n_lift]);
                if !h_eq_xi.is_empty() {
                    let d_eq_xi = h_eq_xi
                        .to_device()
                        .expect("failed to copy eq_xi_per_trace to device");
                    DeviceMatrix::new(Arc::new(d_eq_xi), h_eq_xi.len(), 1)
                } else {
                    DeviceMatrix::dummy()
                }
            })
            .collect_vec();

        self.sels_per_trace_base = self
            .n_per_trace
            .iter()
            .map(|&n| {
                let log_height = l_skip.checked_add_signed(n).unwrap();
                let height = 1 << log_height;
                let lifted_height = height.max(1 << l_skip);
                let mut cols = F::zero_vec(3 * lifted_height);
                cols[lifted_height..2 * lifted_height].fill(F::ONE);
                for i in (0..lifted_height).step_by(height) {
                    cols[i] = F::ONE; // is_first
                    cols[lifted_height + i + height - 1] = F::ZERO; // is_transition
                    cols[2 * lifted_height + i + height - 1] = F::ONE; // is_last
                }
                let d_cols = cols
                    .to_device()
                    .expect("failed to copy selectors base to device");
                DeviceMatrix::new(Arc::new(d_cols), lifted_height, 3)
            })
            .collect_vec();

        let buffers = Round0Buffers {
            selectors_base: self.sels_per_trace_base.clone(),
            eq_xi: self.eq_xi_per_trace.clone(),
        };

        // All (numer, denom) pairs per present AIR for logup, followed by 1 zerocheck poly per
        // present AIR
        let mut batch_polys =
            vec![UnivariatePoly::new(vec![EF::ZERO; s_0_deg + 1]); 3 * num_present_airs];
        let log_large_domain = log2_ceil_usize(s_0_deg + 1);
        let large_domain = 1 << log_large_domain;
        assert!(!xi.is_empty(), "xi vector must not be empty");
        let omega = F::two_adic_generator(log_large_domain);
        let eq_z_host: Vec<EF> = omega
            .powers()
            .take(large_domain)
            .map(|z| eval_eq_uni(l_skip, xi[0], z.into()))
            .collect();
        let d_eq_z = eq_z_host.to_device().unwrap();
        // Precompute eq_sharp_z (using eq_sharp instead of eq_z)
        let eq_sharp_z_host: Vec<EF> = omega
            .powers()
            .take(large_domain)
            .map(|z| eval_eq_sharp_uni(&self.omega_skip_pows, &xi[..l_skip], z.into()))
            .collect();
        let d_eq_sharp_z = eq_sharp_z_host.to_device().unwrap();
        // Loop through one AIR at a time; it is more efficient to do everything for one AIR
        // together
        let d_lambda_pows = self
            .lambda_pows
            .as_ref()
            .expect("lambda powers must be set before round-0 evaluation");

        for (trace_idx, ((air_idx, air_ctx), n, selectors_base, eq_x, eq_3b)) in izip!(
            &ctx.per_trace,
            &self.n_per_trace,
            &buffers.selectors_base,
            &buffers.eq_xi,
            &self.eq_3b_per_trace,
        )
        .enumerate()
        {
            let single_pk = &self.pk.per_air[*air_idx];
            let trace_input = prepare_round0_trace_input(
                l_skip,
                log_large_domain,
                single_pk,
                air_ctx,
                &self.common_main_pcs_data,
                selectors_base,
                eq_x.clone(),
                &self.public_values_per_trace,
                trace_idx,
            )
            .unwrap();

            let sum_buffer = evaluate_round0_constraints_gpu(
                l_skip,
                large_domain,
                single_pk,
                &trace_input,
                d_lambda_pows,
                &d_eq_z,
            )
            .expect("failed to evaluate round-0 constraints on device");
            if !sum_buffer.is_empty() {
                let host_sums = sum_buffer
                    .to_host()
                    .expect("failed to copy aggregated constraint sums back to host");
                batch_polys[2 * num_present_airs + trace_idx] =
                    UnivariatePoly::from_evals_idft(&host_sums);
            }

            let (sum_numer, sum_denom) = evaluate_round0_interactions_gpu(
                l_skip,
                large_domain,
                single_pk,
                &trace_input,
                &d_eq_sharp_z,
                eq_3b,
                &self.beta_pows,
            )
            .expect("failed to evaluate round-0 interactions on device");
            if !sum_numer.is_empty() && !sum_denom.is_empty() {
                let [mut numer_poly, denom_poly] = [sum_numer, sum_denom].map(|sum| {
                    let evals = sum
                        .to_host()
                        .expect("failed to copy interaction sums back to host");
                    UnivariatePoly::from_evals_idft(&evals)
                });
                if n.is_negative() {
                    // normalize for lifting
                    let norm_factor = F::from_canonical_u32(1 << n.unsigned_abs()).inverse();
                    for s in numer_poly.coeffs_mut() {
                        *s *= norm_factor;
                    }
                }
                batch_polys[2 * trace_idx] = numer_poly;
                batch_polys[2 * trace_idx + 1] = denom_poly;
            }
        }

        let gpu_polys: Vec<UnivariatePoly<EF>> = batch_polys
            .into_iter()
            .map(|mut poly| {
                debug_assert!(
                    poly.coeffs()[s_0_deg + 1..]
                        .iter()
                        .all(|&coeff| coeff == EF::ZERO)
                );
                poly.coeffs_mut().truncate(s_0_deg + 1);
                poly
            })
            .collect();

        gpu_polys
    }

    #[instrument(name = "LogupZerocheck::fold_ple_evals", level = "debug", skip_all)]
    fn fold_ple_evals(&mut self, ctx: ProvingContextV2<GpuBackendV2>, r_0: EF) {
        let l_skip = self.l_skip;

        // GPU folding for mat_evals_per_trace
        // We drop (free) old traces from ctx as we go
        self.mat_evals_per_trace = ctx
            .per_trace
            .into_iter()
            .enumerate()
            .map(|(trace_idx, (air_idx, air_ctx))| {
                let air_pk = &self.pk.per_air[air_idx];
                let mut results: Vec<DeviceMatrix<EF>> = Vec::new();

                // Preprocessed (if exists)
                if let Some(committed) = &air_pk.preprocessed_data {
                    let trace = &committed.trace;
                    let width = trace.width();
                    let folded = fold_ple_mixed_rotate(
                        l_skip,
                        trace,
                        committed.data.mixed_view(0, width),
                        r_0,
                    )
                    .unwrap();
                    results.push(folded);
                }

                // Cached mains
                for committed in air_ctx.cached_mains {
                    let trace = committed.trace;
                    let width = trace.width();
                    let folded = fold_ple_mixed_rotate(
                        l_skip,
                        &trace,
                        committed.data.mixed_view(0, width),
                        r_0,
                    )
                    .unwrap();
                    drop(trace);
                    results.push(folded);
                }

                // Common main
                let trace = air_ctx.common_main;
                let width = trace.width();
                let folded = fold_ple_mixed_rotate(
                    l_skip,
                    &trace,
                    self.common_main_pcs_data.mixed_view(trace_idx, width),
                    r_0,
                )
                .unwrap();
                drop(trace);
                results.push(folded);

                results
            })
            .collect();

        // GPU folding for sels_per_trace (rotate=false, only need offset=0)
        self.sels_per_trace = std::mem::take(&mut self.sels_per_trace_base)
            .into_iter()
            .map(|mat| {
                let num_x = mat.height() >> l_skip;
                let width = mat.width();
                debug_assert_ne!(num_x, 0);
                let folded_buf = DeviceBuffer::<EF>::with_capacity(num_x * width);
                unsafe {
                    fold_ple_evals_gpu(l_skip, &mat, folded_buf.as_mut_ptr(), r_0, false).unwrap()
                };
                DeviceMatrix::new(Arc::new(folded_buf), num_x, width)
            })
            .collect();

        // GPU scalar multiplication for eq_xi_per_trace and eq_sharp_per_trace
        // Compute scalars on CPU (small computation)
        let eq_r0 = eval_eq_uni(l_skip, self.xi[0], r_0);
        let eq_sharp_r0 = eval_eq_sharp_uni(&self.omega_skip_pows, &self.xi[..l_skip], r_0);

        // Mutate eq_xi_per_trace in-place and compute eq_sharp_per_trace in a single kernel call
        // This matches CPU behavior: eq_xi *= eq_r0, eq_sharp = original_eq_xi * eq_sharp_r0
        self.eq_sharp_per_trace = self
            .eq_xi_per_trace
            .iter_mut()
            .map(|eq_xi| {
                compute_eq_sharp_gpu(eq_xi, eq_r0, eq_sharp_r0)
                    .expect("failed to multiply eq_xi and compute eq_sharp on GPU")
            })
            .collect();
    }

    #[instrument(
        name = "LogupZerocheck::sumcheck_polys_eval",
        level = "debug",
        skip_all
    )]
    fn sumcheck_polys_eval(&mut self, round: usize, r_prev: EF) -> Vec<Vec<EF>> {
        let s_deg = self.s_deg;
        let lambda_pows = self.lambda_pows.as_ref().expect("lambda_pows must be set");

        let mut s_logup_evals = Vec::new();
        let mut s_zerocheck_evals = Vec::new();
        // Process zerocheck and logup together per trace
        izip!(
            self.n_per_trace.iter(),
            self.zerocheck_tilde_evals.iter_mut(),
            self.logup_tilde_evals.iter_mut(),
            self.mat_evals_per_trace.iter(),
            self.sels_per_trace.iter(),
            self.eq_xi_per_trace.iter(),
            self.eq_sharp_per_trace.iter(),
            self.eq_3b_per_trace.iter(),
            self.public_values_per_trace.iter(),
            self.air_indices_per_trace.iter()
        )
        .for_each(
            |(
                &n,
                zc_tilde_eval,
                logup_tilde_eval,
                mats,
                sels,
                eq_xi,
                eq_sharp,
                eq_3bs,
                public_vals,
                &air_idx,
            )| {
                let mut results = vec![vec![EF::ZERO; s_deg]; 3]; // logup numer, logup denom, zerocheck

                let constraints_dag = &self.pk.per_air[air_idx].vk.symbolic_constraints;
                let constraints = SymbolicConstraints::from(constraints_dag);
                let has_constraints = !constraints.constraints.is_empty();
                let has_interactions = !constraints.interactions.is_empty();

                if has_interactions || has_constraints {
                    let n_lift = n.max(0) as usize;
                    let norm_factor_denom = 1 << (-n).max(0);
                    let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
                    let has_preprocessed = self.pk.per_air[air_idx].preprocessed_data.is_some();

                    if round > n_lift {
                        // Case A.1 round == n_lift + 1
                        if round == n_lift + 1 {
                            let (prep_ptr, first_main_idx) = if has_preprocessed {
                                (
                                    MainMatrixPtrs {
                                        data: mats[0].buffer().as_ptr(),
                                        air_width: mats[0].width() as u32 / 2,
                                    },
                                    1,
                                )
                            } else {
                                (
                                    MainMatrixPtrs {
                                        data: std::ptr::null(),
                                        air_width: 0,
                                    },
                                    0,
                                )
                            };
                            let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats[first_main_idx..]
                                .iter()
                                .map(|m| MainMatrixPtrs {
                                    data: m.buffer().as_ptr(),
                                    air_width: m.width() as u32 / 2,
                                })
                                .collect_vec();
                            if has_constraints {
                                let zc_evals = evaluate_mle_constraints_gpu(
                                    eq_xi.buffer().as_ptr(),
                                    sels.buffer().as_ptr(),
                                    prep_ptr,
                                    &main_ptrs,
                                    public_vals,
                                    lambda_pows,
                                    constraints_dag,
                                    1,
                                    1,
                                );
                                assert_eq!(zc_evals.len(), 1);
                                *zc_tilde_eval = zc_evals[0];
                            }
                            if has_interactions {
                                let [logup_numer_evals, logup_denom_evals] =
                                    evaluate_mle_interactions_gpu(
                                        eq_sharp.buffer().as_ptr(),
                                        sels.buffer().as_ptr(),
                                        prep_ptr,
                                        &main_ptrs,
                                        public_vals,
                                        &self.beta_pows,
                                        eq_3bs,
                                        constraints_dag,
                                        1,
                                        1,
                                    );
                                assert_eq!(logup_numer_evals.len(), 1);
                                assert_eq!(logup_denom_evals.len(), 1);
                                logup_tilde_eval[0] = logup_numer_evals[0] * norm_factor;
                                logup_tilde_eval[1] = logup_denom_evals[0];
                            }
                        } else {
                            // Case A.2 round > n_lift + 1
                            if has_constraints {
                                *zc_tilde_eval *= r_prev;
                            }
                            if has_interactions {
                                for x in logup_tilde_eval.iter_mut() {
                                    *x *= r_prev;
                                }
                            }
                        }
                        if has_constraints {
                            results[2] = (1..=s_deg)
                                .map(|x| *zc_tilde_eval * F::from_canonical_usize(x))
                                .collect();
                        }
                        if has_interactions {
                            let [numer, denom] = logup_tilde_eval.map(|tilde_eval| {
                                (1..=s_deg)
                                    .map(|x| tilde_eval * F::from_canonical_usize(x))
                                    .collect()
                            });
                            results[0] = numer;
                            results[1] = denom;
                        }
                    } else {
                        // Case B: round <= n_lift
                        let n = n_lift.saturating_sub(round - 1);
                        let height = 1 << n;
                        let mut columns: Vec<usize> = Vec::new();
                        if has_constraints {
                            debug_assert_eq!(eq_xi.height(), height);
                            debug_assert_eq!(eq_xi.width(), 1);
                            columns.push(eq_xi.buffer().as_ptr() as usize);
                        }
                        if has_interactions {
                            debug_assert_eq!(eq_sharp.height(), height);
                            debug_assert_eq!(eq_sharp.width(), 1);
                            columns.push(eq_sharp.buffer().as_ptr() as usize);
                        }
                        columns.extend(
                            iter::once(sels)
                                .chain(mats.iter())
                                .flat_map(|m| {
                                    assert_eq!(m.height(), height);
                                    (0..m.width()).map(|col| {
                                        m.buffer().as_ptr().wrapping_add(col * m.height()) as usize
                                    })
                                })
                                .collect_vec(),
                        );
                        let num_columns = columns.len();
                        let num_y = height / 2;
                        let interpolated =
                            DeviceMatrix::<EF>::with_capacity(s_deg * num_y, num_columns);
                        let d_columns = columns
                            .to_device()
                            .expect("failed to copy column ptrs to device");
                        unsafe {
                            interpolate_columns_gpu(
                                interpolated.buffer(),
                                &d_columns,
                                s_deg,
                                num_y,
                            )
                            .expect("failed to interpolate columns on GPU");
                        }
                        // interpolated columns layout:
                        // [eq_xi (x1), eq_sharp (x1), sels (x3), prep? (x1), main[i] (x1)]

                        // EVALUATION:
                        let interpolated_height = interpolated.height();
                        let mats_widths: Vec<usize> = mats.iter().map(|m| m.width() / 2).collect();
                        let mut widths_so_far = 0;
                        let eq_xi_ptr = if has_constraints {
                            let ptr = interpolated
                                .buffer()
                                .as_ptr()
                                .wrapping_add(widths_so_far * interpolated_height);
                            widths_so_far += 1;
                            ptr
                        } else {
                            std::ptr::null()
                        };
                        let eq_sharp_ptr = if has_interactions {
                            let ptr = interpolated
                                .buffer()
                                .as_ptr()
                                .wrapping_add(widths_so_far * interpolated_height);
                            widths_so_far += 1;
                            ptr
                        } else {
                            std::ptr::null()
                        };
                        let sels_ptr = interpolated
                            .buffer()
                            .as_ptr()
                            .wrapping_add(widths_so_far * interpolated_height);
                        widths_so_far += 3;
                        let (prep_ptr, first_main_idx) = if has_preprocessed {
                            (
                                MainMatrixPtrs {
                                    data: interpolated
                                        .buffer()
                                        .as_ptr()
                                        .wrapping_add(widths_so_far * interpolated_height),
                                    air_width: mats_widths[0] as u32,
                                },
                                1,
                            )
                        } else {
                            (
                                MainMatrixPtrs {
                                    data: std::ptr::null(),
                                    air_width: 0,
                                },
                                0,
                            )
                        };
                        widths_so_far += 2 * prep_ptr.air_width as usize;
                        let main_ptrs: Vec<MainMatrixPtrs<EF>> = mats_widths[first_main_idx..]
                            .iter()
                            .map(|m_width| {
                                let main_ptr = MainMatrixPtrs {
                                    data: interpolated
                                        .buffer()
                                        .as_ptr()
                                        .wrapping_add(widths_so_far * interpolated_height),
                                    air_width: *m_width as u32,
                                };
                                widths_so_far += m_width * 2;
                                main_ptr
                            })
                            .collect_vec();
                        debug_assert_eq!(widths_so_far, interpolated.width());
                        if has_constraints {
                            results[2] = evaluate_mle_constraints_gpu(
                                eq_xi_ptr,
                                sels_ptr,
                                prep_ptr,
                                &main_ptrs,
                                public_vals,
                                lambda_pows,
                                constraints_dag,
                                interpolated_height,
                                s_deg,
                            );
                        }
                        if has_interactions {
                            let [numer_evals, denom_evals] = evaluate_mle_interactions_gpu(
                                eq_sharp_ptr,
                                sels_ptr,
                                prep_ptr,
                                &main_ptrs,
                                public_vals,
                                &self.beta_pows,
                                eq_3bs,
                                constraints_dag,
                                interpolated_height,
                                s_deg,
                            );

                            // Apply normalization to numer only (same as CPU)
                            let mut numer_normalized = numer_evals;
                            for p in numer_normalized.iter_mut() {
                                *p *= norm_factor;
                            }
                            results[0] = numer_normalized;
                            results[1] = denom_evals;
                        }
                    }
                }
                s_logup_evals.push(results[0].clone());
                s_logup_evals.push(results[1].clone());
                s_zerocheck_evals.push(results[2].clone());
            },
        );

        s_logup_evals.into_iter().chain(s_zerocheck_evals).collect()
    }

    #[instrument(name = "LogupZerocheck::fold_mle_evals", level = "debug", skip_all)]
    fn fold_mle_evals(&mut self, _round: usize, r_round: EF) {
        // Fold mat_evals_per_trace: Vec<Vec<DeviceMatrix<EF>>>
        self.mat_evals_per_trace = std::mem::take(&mut self.mat_evals_per_trace)
            .into_iter()
            .map(|mats| {
                if mats.is_empty() {
                    return mats;
                }
                let height = mats[0].height();
                if height <= 1 {
                    return mats;
                }
                let output_height = height >> 1;

                // Prepare input/output pointers and widths
                let num_matrices = mats.len() as u32;
                let input_ptrs: Vec<usize> =
                    mats.iter().map(|m| m.buffer().as_ptr() as usize).collect();
                let widths: Vec<u32> = mats.iter().map(|m| m.width() as u32).collect();

                // Allocate output matrices (preserve doubled-width structure)
                let output_mats: Vec<DeviceMatrix<EF>> = mats
                    .iter()
                    .map(|m| DeviceMatrix::with_capacity(output_height, m.width()))
                    .collect();
                let output_ptrs: Vec<usize> = output_mats
                    .iter()
                    .map(|m| m.buffer().as_ptr() as usize)
                    .collect();

                // Copy to device buffers
                let d_input_ptrs = input_ptrs
                    .to_device()
                    .expect("failed to copy input ptrs to device");
                let d_output_ptrs = output_ptrs
                    .to_device()
                    .expect("failed to copy output ptrs to device");
                let d_widths = widths.to_device().expect("failed to copy widths to device");

                // Launch fold_mle kernel
                unsafe {
                    fold_mle(
                        &d_input_ptrs,
                        &d_output_ptrs,
                        &d_widths,
                        num_matrices,
                        output_height as u32,
                        r_round,
                    )
                    .expect("failed to fold MLE matrices on GPU");
                }

                output_mats
            })
            .collect();

        // Fold sels_per_trace: Vec<DeviceMatrix<EF>>
        self.sels_per_trace = std::mem::take(&mut self.sels_per_trace)
            .into_iter()
            .map(|mat| {
                let height = mat.height();
                if height <= 1 {
                    return mat;
                }
                let output_height = height >> 1;
                let width = mat.width();

                let input_ptr = mat.buffer().as_ptr() as usize;
                let output_mat = DeviceMatrix::with_capacity(output_height, width);
                let output_ptr = output_mat.buffer().as_ptr() as usize;

                let d_input_ptrs = [input_ptr]
                    .to_device()
                    .expect("failed to copy input ptr to device");
                let d_output_ptrs = [output_ptr]
                    .to_device()
                    .expect("failed to copy output ptr to device");
                let d_widths = [width as u32]
                    .to_device()
                    .expect("failed to copy width to device");

                unsafe {
                    fold_mle(
                        &d_input_ptrs,
                        &d_output_ptrs,
                        &d_widths,
                        1,
                        output_height as u32,
                        r_round,
                    )
                    .expect("failed to fold MLE selectors on GPU");
                }

                output_mat
            })
            .collect();

        // Fold eq_xi_per_trace: Vec<DeviceMatrix<EF>>
        self.eq_xi_per_trace = std::mem::take(&mut self.eq_xi_per_trace)
            .into_iter()
            .map(|mat| {
                let height = mat.height();
                if height <= 1 {
                    return mat;
                }
                let output_height = height >> 1;
                let width = mat.width();

                let input_ptr = mat.buffer().as_ptr() as usize;
                let output_mat = DeviceMatrix::with_capacity(output_height, width);
                let output_ptr = output_mat.buffer().as_ptr() as usize;

                let d_input_ptrs = [input_ptr]
                    .to_device()
                    .expect("failed to copy input ptr to device");
                let d_output_ptrs = [output_ptr]
                    .to_device()
                    .expect("failed to copy output ptr to device");
                let d_widths = [width as u32]
                    .to_device()
                    .expect("failed to copy width to device");

                unsafe {
                    fold_mle(
                        &d_input_ptrs,
                        &d_output_ptrs,
                        &d_widths,
                        1,
                        output_height as u32,
                        r_round,
                    )
                    .expect("failed to fold MLE eq_xi on GPU");
                }

                output_mat
            })
            .collect();

        // Fold eq_sharp_per_trace: Vec<DeviceMatrix<EF>>
        self.eq_sharp_per_trace = std::mem::take(&mut self.eq_sharp_per_trace)
            .into_iter()
            .map(|mat| {
                let height = mat.height();
                if height <= 1 {
                    return mat;
                }
                let output_height = height >> 1;
                let width = mat.width();

                let input_ptr = mat.buffer().as_ptr() as usize;
                let output_mat = DeviceMatrix::with_capacity(output_height, width);
                let output_ptr = output_mat.buffer().as_ptr() as usize;

                let d_input_ptrs = [input_ptr]
                    .to_device()
                    .expect("failed to copy input ptr to device");
                let d_output_ptrs = [output_ptr]
                    .to_device()
                    .expect("failed to copy output ptr to device");
                let d_widths = [width as u32]
                    .to_device()
                    .expect("failed to copy width to device");

                unsafe {
                    fold_mle(
                        &d_input_ptrs,
                        &d_output_ptrs,
                        &d_widths,
                        1,
                        output_height as u32,
                        r_round,
                    )
                    .expect("failed to fold MLE eq_sharp on GPU");
                }

                output_mat
            })
            .collect();
    }

    #[instrument(
        name = "LogupZerocheck::into_column_openings",
        level = "debug",
        skip_all
    )]
    fn into_column_openings(mut self) -> Vec<Vec<Vec<(EF, EF)>>> {
        let num_airs_present = self.mat_evals_per_trace.len();
        let mut column_openings = Vec::with_capacity(num_airs_present);

        // At the end, we've folded all MLEs so they only have one row equal to evaluation at `\vec
        // r`.
        for mat_evals in std::mem::take(&mut self.mat_evals_per_trace) {
            // GPU matrices are doubled-width (original + rotated), so we need to split them
            // First, copy all matrices to host and split them
            let mut split_mats: Vec<ColMajorMatrix<EF>> = mat_evals
                .into_iter()
                .flat_map(|mat| {
                    let mat_host = transport_matrix_d2h_col_major(&mat)
                        .expect("failed to copy GPU matrix to host");
                    let width = mat_host.width();
                    let height = mat_host.height();
                    debug_assert_eq!(height, 1, "Matrices should have height=1 after folding");
                    debug_assert_eq!(
                        width % 2,
                        0,
                        "GPU matrices should have doubled width (original + rotated)"
                    );
                    let air_width = width / 2;

                    // Split doubled-width matrix into original and rotated parts
                    let values = &mat_host.values;
                    let orig: Vec<EF> = (0..air_width)
                        .map(|col| values[col * height]) // height=1, so values[col]
                        .collect();
                    let rot: Vec<EF> = (air_width..width)
                        .map(|col| values[col * height]) // height=1, so values[col]
                        .collect();

                    vec![
                        ColMajorMatrix::new(orig, air_width),
                        ColMajorMatrix::new(rot, air_width),
                    ]
                })
                .collect();

            // Order of mats after splitting is:
            // - preprocessed (if has_preprocessed),
            // - preprocessed_rot (if has_preprocessed),
            // - cached(0), cached(0)_rot, ...
            // - common_main
            // - common_main_rot
            // For column openings, we pop common_main, common_main_rot and put it at the front
            assert_eq!(
                split_mats.len() % 2,
                0,
                "Should have even number of matrices after splitting"
            );
            let common_main_rot = split_mats.pop().unwrap();
            let common_main = split_mats.pop().unwrap();

            let openings_of_air = iter::once(&[common_main, common_main_rot] as &[_])
                .chain(split_mats.chunks_exact(2))
                .map(|pair| {
                    std::iter::zip(pair[0].columns(), pair[1].columns())
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
