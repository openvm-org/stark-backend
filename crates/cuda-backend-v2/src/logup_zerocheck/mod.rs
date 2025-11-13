use std::{cmp::max, sync::Arc};

use itertools::{Itertools, izip};
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{Field, FieldAlgebra, TwoAdicField};
use p3_util::log2_strict_usize;
use stark_backend_v2::{
    poly_common::{UnivariatePoly, eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni},
    poseidon2::sponge::FiatShamirTranscript,
    prover::{
        AirProvingContextV2, CpuBackendV2, CpuDeviceV2, DeviceMultiStarkProvingKeyV2,
        DeviceStarkProvingKeyV2, LogupZerocheckCpu, LogupZerocheckProver, ProvingContextV2,
        fractional_sumcheck_gkr::FracSumcheckProof, poly::evals_eq_hypercube,
        stacked_pcs::StackedLayout, sumcheck::sumcheck_round0_deg,
    },
};
use tracing::debug;

use crate::{
    Digest, EF, F, GpuBackendV2, GpuDeviceV2,
    gpu_backend::{transport_committed_trace_data_to_host, transport_matrix_d2h_col_major},
    stacked_pcs::StackedPcsDataGpu,
};

mod dag_scheduling;
mod errors;
mod fold_ple;
mod fractional;
mod interactions;
mod matrix_utils;
mod round0;
mod state;

pub use errors::*;
use fold_ple::{compute_eq_sharp_gpu, fold_ple_evals_gpu};
use fractional::{fractional_sumcheck_gpu, initialize_segment_tree};
use interactions::{collect_trace_interactions, evaluate_interactions_gpu};
use matrix_utils::unstack_matrix_round0;
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

    zerocheck_tilde_evals: Vec<EF>,
    logup_tilde_evals: Vec<[EF; 2]>,

    fractional_state: FractionalGkrState, //?
    // round0: Round0Buffers,
    pk: &'a DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    device: &'a GpuDeviceV2,
    common_main_pcs_data: &'a StackedPcsDataGpu<F, Digest>,
    cpu_fallback: CpuFallback<'a>,
}

impl<'a, TS> LogupZerocheckProver<'a, GpuBackendV2, GpuDeviceV2, TS> for LogupZerocheckGpu<'a>
where
    TS: FiatShamirTranscript,
{
    // TODO: modify trait so we can drop common main buffers in `ctx`: they are extraneous given
    // `common_main_pcs_data`
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
        let trace_interactions = collect_trace_interactions(pk, ctx, &interactions_layout, l_skip);
        let mut fractional_state = FractionalGkrState {
            trace_interactions,
            ..Default::default()
        };

        // CPU fallback: transport inputs and reuse the CPU implementation for now.
        let cpu_pk = transport_device_pk_to_host(pk);
        let cpu_device = CpuDeviceV2::new(device.config());
        let cpu_ctx = transport_proving_context_ref_to_host(ctx);

        let total_leaves = 1 << (l_skip + n_logup);
        let leaves = if has_interactions {
            evaluate_interactions_gpu(
                &fractional_state,
                pk,
                ctx,
                common_main_pcs_data,
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

        let mut transcript_shadow = transcript.clone();

        let (frac_sum_proof, mut xi) = fractional_sumcheck_gpu(transcript, &fractional_state, true)
            .expect("failed to run fractional sumcheck on GPU");

        let n_global = max(n_max, n_logup);
        debug!(%n_global);
        while xi.len() != l_skip + n_global {
            xi.push(transcript.sample_ext());
        }
        debug!(?xi);

        // TODO: delete after done with gpu
        let (cpu_prover, cpu_frac_proof) = LogupZerocheckCpu::prove_logup_gkr(
            unsafe { &*(&cpu_device as *const _) },
            &mut transcript_shadow,
            unsafe { &*(&cpu_pk as *const _) },
            &cpu_ctx,
            unsafe { &*(std::ptr::dangling() as *const _) }, // UNSAFE: CPU doesn't use this ptr
            n_logup,
            interactions_layout.clone(),
            alpha_logup,
            beta_logup,
        );

        debug_assert_eq!(frac_sum_proof.fractional_sum, cpu_frac_proof.fractional_sum);
        debug_assert_eq!(
            frac_sum_proof.claims_per_layer.len(),
            cpu_frac_proof.claims_per_layer.len()
        );
        debug_assert_eq!(
            frac_sum_proof.sumcheck_polys.len(),
            cpu_frac_proof.sumcheck_polys.len()
        );
        for (gpu_claim, cpu_claim) in frac_sum_proof
            .claims_per_layer
            .iter()
            .zip(cpu_frac_proof.claims_per_layer.iter())
        {
            debug_assert_eq!(gpu_claim.p_xi_0, cpu_claim.p_xi_0);
            debug_assert_eq!(gpu_claim.q_xi_0, cpu_claim.q_xi_0);
            debug_assert_eq!(gpu_claim.p_xi_1, cpu_claim.p_xi_1);
            debug_assert_eq!(gpu_claim.q_xi_1, cpu_claim.q_xi_1);
        }
        for (gpu_layer, cpu_layer) in frac_sum_proof
            .sumcheck_polys
            .iter()
            .zip(cpu_frac_proof.sumcheck_polys.iter())
        {
            debug_assert_eq!(gpu_layer.len(), cpu_layer.len());
            for (gpu_round, cpu_round) in gpu_layer.iter().zip(cpu_layer.iter()) {
                debug_assert_eq!(gpu_round, cpu_round);
            }
        }

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
            zerocheck_tilde_evals: vec![EF::ZERO; num_airs_present],
            logup_tilde_evals: vec![[EF::ZERO; 2]; num_airs_present],
            fractional_state,
            pk,
            device,
            common_main_pcs_data,
            cpu_fallback: CpuFallback::new(cpu_prover, cpu_device, cpu_pk),
        };

        (prover, frac_sum_proof)
    }

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
            public_values: ctx
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
        };

        let sum_buffers = evaluate_round0_constraints_gpu(
            self.l_skip,
            self.s_deg,
            &self.n_per_trace,
            self.pk,
            ctx,
            self.common_main_pcs_data,
            &buffers,
            &self.xi,
            self.lambda_pows
                .as_ref()
                .expect("lambda powers must be set before round-0 evaluation"),
        )
        .expect("failed to evaluate round-0 constraints on device");

        let zerocheck_polys = sum_buffers
            .iter()
            .map(|sum_buffer| {
                if sum_buffer.is_empty() {
                    UnivariatePoly::new(vec![EF::ZERO; s_0_deg + 1])
                } else {
                    let host_sums = sum_buffer
                        .to_host()
                        .expect("failed to copy aggregated constraint sums back to host");
                    UnivariatePoly::from_evals_idft(&host_sums)
                }
            })
            .collect_vec();

        let logup_polys = if self.eq_3b_per_trace.iter().all(|eq_3b| eq_3b.is_empty()) {
            (0..zerocheck_polys.len() * 2)
                .map(|_| UnivariatePoly::new(vec![EF::ZERO; s_0_deg + 1]))
                .collect_vec()
        } else {
            let (sums_numer, sums_denom) = evaluate_round0_interactions_gpu(
                self.l_skip,
                self.s_deg,
                &self.n_per_trace,
                self.pk,
                ctx,
                self.common_main_pcs_data,
                &buffers,
                &self.xi,
                &self.omega_skip_pows,
                &self.beta_pows,
                &self.eq_3b_per_trace,
            )
            .expect("failed to evaluate round-0 interactions on device");

            // Convert to polynomials: each trace has 2 polynomials (numer, denom)
            // Ensure we have the same number of traces as CPU expects
            assert_eq!(
                sums_numer.len(),
                ctx.per_trace.len(),
                "Mismatch in number of traces"
            );
            debug_assert_eq!(sums_numer.len(), self.n_per_trace.len());
            let mut logup_polys = Vec::with_capacity(sums_numer.len() * 2);
            for (sum_numer, sum_denom, n) in izip!(sums_numer, sums_denom, &self.n_per_trace) {
                let mut numer_poly = if sum_numer.is_empty() {
                    UnivariatePoly::new(vec![EF::ZERO; s_0_deg + 1])
                } else {
                    let host_sums = sum_numer
                        .to_host()
                        .expect("failed to copy interaction numer sums back to host");
                    UnivariatePoly::from_evals_idft(&host_sums)
                };
                let denom_poly = if sum_denom.is_empty() {
                    UnivariatePoly::new(vec![EF::ZERO; s_0_deg + 1])
                } else {
                    let host_sums = sum_denom
                        .to_host()
                        .expect("failed to copy interaction denom sums back to host");
                    UnivariatePoly::from_evals_idft(&host_sums)
                };
                if n.is_negative() {
                    // normalize for lifting
                    let norm_factor = F::from_canonical_u32(1 << n.unsigned_abs()).inverse();
                    for s in numer_poly.coeffs_mut() {
                        *s *= norm_factor;
                    }
                }
                logup_polys.push(numer_poly);
                logup_polys.push(denom_poly);
            }
            logup_polys
        };

        // TODO: remove this after GPU-ising all
        let cpu_polys = self
            .cpu_fallback
            .sumcheck_uni_round0_polys::<TS>(ctx, lambda);

        logup_polys
            .into_iter()
            .chain(zerocheck_polys)
            .map(|mut poly| {
                debug_assert!(
                    poly.coeffs()[s_0_deg + 1..]
                        .iter()
                        .all(|&coeff| coeff == EF::ZERO)
                );
                poly.coeffs_mut().truncate(s_0_deg + 1);
                poly
            })
            .collect()
    }

    fn fold_ple_evals(&mut self, ctx: ProvingContextV2<GpuBackendV2>, r_0: EF) {
        let l_skip = self.l_skip;

        // GPU folding for mat_evals_per_trace
        self.mat_evals_per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, air_ctx)| {
                let air_pk = &self.pk.per_air[*air_idx];
                let mut results = Vec::new();

                // Preprocessed (if exists)
                if let Some(preprocessed_data) = &air_pk.preprocessed_data {
                    let (mat, _) = unstack_matrix_round0(preprocessed_data.data.as_ref(), 0)
                        .expect("failed to unstack preprocessed");
                    let (folded_0, folded_1) = fold_ple_evals_gpu(l_skip, &mat, r_0, true)
                        .expect("failed to fold PLE on GPU");
                    results.extend([folded_0, folded_1]);
                }

                // Cached mains
                for cached_data in &air_ctx.cached_mains {
                    let (mat, _) = unstack_matrix_round0(cached_data.data.as_ref(), 0)
                        .expect("failed to unstack cached");
                    let (folded_0, folded_1) = fold_ple_evals_gpu(l_skip, &mat, r_0, true)
                        .expect("failed to fold PLE on GPU");
                    results.extend([folded_0, folded_1]);
                }

                // Common main
                let (folded_0, folded_1) =
                    fold_ple_evals_gpu(l_skip, &air_ctx.common_main, r_0, true)
                        .expect("failed to fold PLE on GPU");
                results.extend([folded_0, folded_1]);

                results
            })
            .collect();

        // GPU folding for sels_per_trace (rotate=false, only need offset=0)
        self.sels_per_trace = std::mem::take(&mut self.sels_per_trace_base)
            .into_iter()
            .map(|mat| {
                fold_ple_evals_gpu(l_skip, &mat, r_0, false)
                    .expect("failed to fold selectors on GPU")
                    .0 // Only use the first result (offset=0)
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

        // CPU fallback for comparison. TODO: remove this after GPU-ising all
        self.cpu_fallback.fold_ple_evals::<TS>(ctx, r_0);

        // Compare GPU vs CPU results for mat_evals_per_trace
        let gpu_host: Vec<Vec<Vec<EF>>> = self
            .mat_evals_per_trace
            .iter()
            .map(|trace_mats| {
                trace_mats
                    .iter()
                    .map(|mat| {
                        transport_matrix_d2h_col_major(mat)
                            .expect("failed to copy GPU matrix to host")
                            .values
                    })
                    .collect()
            })
            .collect();
        let cpu_host: Vec<Vec<Vec<EF>>> = self
            .cpu_fallback
            .prover
            .mat_evals_per_trace
            .iter()
            .map(|trace_mats| trace_mats.iter().map(|mat| mat.values.clone()).collect())
            .collect();
        assert_eq!(gpu_host, cpu_host);

        // Compare GPU vs CPU results for sels_per_trace
        let gpu_sels: Vec<Vec<EF>> = self
            .sels_per_trace
            .iter()
            .map(|mat| {
                transport_matrix_d2h_col_major(mat)
                    .expect("failed to copy GPU matrix to host")
                    .values
            })
            .collect();
        let cpu_sels: Vec<Vec<EF>> = self
            .cpu_fallback
            .prover
            .sels_per_trace
            .iter()
            .map(|mat| mat.values.clone())
            .collect();
        assert_eq!(gpu_sels, cpu_sels);

        // Compare GPU vs CPU results for eq_xi_per_trace
        let gpu_eq_xi: Vec<Vec<EF>> = self
            .eq_xi_per_trace
            .iter()
            .map(|mat| {
                transport_matrix_d2h_col_major(mat)
                    .expect("failed to copy GPU matrix to host")
                    .values
            })
            .collect();
        let cpu_eq_xi: Vec<Vec<EF>> = self
            .cpu_fallback
            .prover
            .eq_xi_per_trace
            .iter()
            .map(|mat| mat.values.clone())
            .collect();
        assert_eq!(gpu_eq_xi, cpu_eq_xi);

        // Compare GPU vs CPU results for eq_sharp_per_trace
        let gpu_eq_sharp: Vec<Vec<EF>> = self
            .eq_sharp_per_trace
            .iter()
            .map(|mat| {
                transport_matrix_d2h_col_major(mat)
                    .expect("failed to copy GPU matrix to host")
                    .values
            })
            .collect();
        let cpu_eq_sharp: Vec<Vec<EF>> = self
            .cpu_fallback
            .prover
            .eq_sharp_per_trace
            .iter()
            .map(|mat| mat.values.clone())
            .collect();
        assert_eq!(gpu_eq_sharp, cpu_eq_sharp);
    }

    fn sumcheck_polys_eval(&mut self, round: usize, r_prev: EF) -> Vec<Vec<EF>> {
        self.cpu_fallback.sumcheck_polys_eval::<TS>(round, r_prev)
    }

    fn fold_mle_evals(&mut self, round: usize, r_round: EF) {
        self.cpu_fallback.fold_mle_evals::<TS>(round, r_round);
    }

    fn into_column_openings(self) -> Vec<Vec<Vec<(EF, EF)>>> {
        self.cpu_fallback.into_column_openings::<TS>()
    }
}

struct CpuFallback<'a> {
    prover: LogupZerocheckCpu<'a>,
    _device: CpuDeviceV2,
    _pk: DeviceMultiStarkProvingKeyV2<CpuBackendV2>,
}

impl<'a> CpuFallback<'a> {
    fn new(
        prover: LogupZerocheckCpu<'a>,
        device: CpuDeviceV2,
        pk: DeviceMultiStarkProvingKeyV2<CpuBackendV2>,
    ) -> Self {
        Self {
            prover,
            _device: device,
            _pk: pk,
        }
    }

    fn sumcheck_uni_round0_polys<TS: FiatShamirTranscript>(
        &mut self,
        ctx: &ProvingContextV2<GpuBackendV2>,
        lambda: EF,
    ) -> Vec<UnivariatePoly<EF>> {
        let cpu_ctx = transport_proving_context_ref_to_host(ctx);
        LogupZerocheckProver::<_, _, TS>::sumcheck_uni_round0_polys(
            &mut self.prover,
            &cpu_ctx,
            lambda,
        )
    }

    fn fold_ple_evals<TS: FiatShamirTranscript>(
        &mut self,
        ctx: ProvingContextV2<GpuBackendV2>,
        r_0: EF,
    ) {
        let cpu_ctx = transport_proving_context_ref_to_host(&ctx);
        LogupZerocheckProver::<_, _, TS>::fold_ple_evals(&mut self.prover, cpu_ctx, r_0);
    }

    fn sumcheck_polys_eval<TS: FiatShamirTranscript>(
        &mut self,
        round: usize,
        r_prev: EF,
    ) -> Vec<Vec<EF>> {
        LogupZerocheckProver::<_, _, TS>::sumcheck_polys_eval(&mut self.prover, round, r_prev)
    }

    fn fold_mle_evals<TS: FiatShamirTranscript>(&mut self, round: usize, r_round: EF) {
        LogupZerocheckProver::<_, _, TS>::fold_mle_evals(&mut self.prover, round, r_round);
    }

    fn into_column_openings<TS: FiatShamirTranscript>(self) -> Vec<Vec<Vec<(EF, EF)>>> {
        LogupZerocheckProver::<_, _, TS>::into_column_openings(self.prover)
    }
}

fn transport_device_pk_to_host(
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
) -> DeviceMultiStarkProvingKeyV2<CpuBackendV2> {
    let per_air = pk
        .per_air
        .iter()
        .map(|air_pk| {
            let preprocessed_data = air_pk
                .preprocessed_data
                .as_ref()
                .map(transport_committed_trace_data_to_host);
            DeviceStarkProvingKeyV2 {
                air_name: air_pk.air_name.clone(),
                vk: air_pk.vk.clone(),
                preprocessed_data,
            }
        })
        .collect();

    DeviceMultiStarkProvingKeyV2::new(
        per_air,
        pk.trace_height_constraints.clone(),
        pk.max_constraint_degree,
        pk.params,
        pk.vk_pre_hash,
    )
}

pub(crate) fn transport_proving_context_ref_to_host(
    ctx: &ProvingContextV2<GpuBackendV2>,
) -> ProvingContextV2<CpuBackendV2> {
    let per_trace = ctx
        .per_trace
        .iter()
        .map(|(air_idx, air_ctx)| (*air_idx, transport_air_context_to_host(air_ctx)))
        .collect();
    ProvingContextV2::new(per_trace)
}

fn transport_air_context_to_host(
    air_ctx: &AirProvingContextV2<GpuBackendV2>,
) -> AirProvingContextV2<CpuBackendV2> {
    let cached_mains = air_ctx
        .cached_mains
        .iter()
        .map(transport_committed_trace_data_to_host)
        .collect();
    let common_main = transport_matrix_d2h_col_major(&air_ctx.common_main).unwrap();
    let public_values = air_ctx.public_values.clone();
    AirProvingContextV2::new(cached_mains, common_main, public_values)
}
