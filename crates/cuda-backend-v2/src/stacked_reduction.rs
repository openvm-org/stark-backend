use std::{array::from_fn, cmp::max, ffi::c_void, iter::zip, mem, sync::Arc};

use itertools::{Itertools, zip_eq};
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use openvm_stark_backend::{p3_matrix::dense::RowMajorMatrix, prover::MatrixDimensions};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use stark_backend_v2::{
    dft::Radix2BowersSerial,
    poly_common::{
        Squarable, UnivariatePoly, eq_uni_poly, eval_eq_mle, eval_eq_uni, eval_eq_uni_at_one,
        eval_in_uni,
    },
    poseidon2::sponge::FiatShamirTranscript,
    proof::StackingProof,
    prover::{
        DeviceMultiStarkProvingKeyV2, ProvingContextV2, stacked_pcs::StackedLayout,
        sumcheck::sumcheck_round0_deg,
    },
};
use tracing::{debug, info_span, instrument};

use crate::{
    D_EF, Digest, EF, F, GpuBackendV2, GpuDeviceV2, ProverError,
    cuda::{
        batch_ntt_small::ensure_device_ntt_twiddles_initialized,
        poly::vector_scalar_multiply_ext,
        stacked_reduction::{
            _stacked_reduction_r0_required_temp_buffer_size, NUM_G,
            initialize_k_rot_from_eq_segments, stacked_reduction_fold_ple,
            stacked_reduction_sumcheck_mle_round, stacked_reduction_sumcheck_mle_round_degenerate,
            stacked_reduction_sumcheck_round0,
        },
        sumcheck::{fold_mle, triangular_fold_mle},
    },
    poly::EqEvalSegments,
    sponge::DuplexSpongeGpu,
    stacked_pcs::StackedPcsDataGpu,
    utils::{compute_barycentric_inv_lagrange_denoms, reduce_raw_u64_to_ef},
};

/// Degree of the sumcheck polynomial for stacked reduction.
pub const STACKED_REDUCTION_S_DEG: usize = 2;

pub struct StackedReductionGpu {
    sm_count: u32,

    l_skip: usize,
    n_stack: usize,

    omega_skip: F,
    omega_skip_pows: Vec<F>,
    d_omega_skip_pows: DeviceBuffer<F>,

    r_0: EF,
    d_lambda_pows: DeviceBuffer<EF>,
    eq_const: EF,

    pub(crate) stacked_per_commit: Vec<StackedPcsData2>,
    d_q_widths: DeviceBuffer<u32>,
    q_width_max: u32,
    d_q_eval_ptrs: DeviceBuffer<*const EF>,

    trace_ptrs: Vec<(
        *const F, /* trace_ptr */
        usize,    /* height */
        usize,    /* width */
    )>,
    unstacked_cols: Vec<UnstackedSlice>,
    d_unstacked_cols: DeviceBuffer<UnstackedSlice>,
    // boundary indices where heights change. all columns between two boundaries must have the same
    // height. We can have multiple chunks of the same height if that is needed to reduce peak GPU
    // memory.
    ht_diff_idxs: Vec<usize>,
    n_max: usize,

    // Initially holds eq(r[1..=n], H_n) for n=0..=n_max but gets updated after each sumcheck round
    // by some custom folding
    eq_r_ns: EqEvalSegments<EF>,

    // == After round 0 ==
    q_evals: Vec<DeviceBuffer<EF>>, // get width from stacked_per_commit
    // Stores folded eq values that won't change anymore (no more folding)
    // Corresponds to log_height in 0..l_skip+round-1 _before_ round `round`. Gets updated with one
    // new element after each round.
    eq_stable: Vec<EF>,
    k_rot_stable: Vec<EF>,

    /// Stores the folded k_rot evaluations for `\kappa_\rot(x, r) = eq_n(rot^{-1}(x), r)` for each
    /// `n` after each round. We use the [EqEvalSegments] type to guard the segment-based
    /// memory layout.
    k_rot_ns: EqEvalSegments<EF>,
    /// Stores eq(u[1+n_T..round-1], b_{T,j}[..round-n_T-1])
    eq_ub_per_trace: Vec<EF>,
    d_eq_ub: DeviceBuffer<EF>,

    d_block_sums: DeviceBuffer<EF>,
    d_accum: DeviceBuffer<u64>,
    d_input_ptrs: DeviceBuffer<*const EF>,
    d_output_ptrs: DeviceBuffer<*mut EF>,

    mem: MemTracker,
}

/// A struct for holding stacked pcs data. We only need the `MerkleTreeGpu` from `StackedPcsDataGpu`
/// but we wrap the latter in an `Arc` to provide uniformity in dealing with common main and
/// preprocessed/cached traces. This struct stores the unstacked `traces`, in prismalinear
/// evaluation form, corresponding to the stacked pcs data.
pub struct StackedPcsData2 {
    pub(crate) inner: Arc<StackedPcsDataGpu<F, Digest>>,
    /// The unstacked traces corresponding to `inner`'s commitment.
    pub(crate) traces: Vec<DeviceMatrix<F>>,
}

impl StackedPcsData2 {
    /// # Safety
    /// `traces` must be the traces that were committed to in `pcs_data`.
    pub unsafe fn from_raw(
        pcs_data: Arc<StackedPcsDataGpu<F, Digest>>,
        traces: Vec<DeviceMatrix<F>>,
    ) -> Self {
        Self {
            inner: pcs_data,
            traces,
        }
    }

    pub fn layout(&self) -> &StackedLayout {
        &self.inner.layout
    }
}

/// Pointer with length to location in a big device buffer. The device buffer is identified by
/// `commit_idx`. Due to pecuarlities with `l_skip`, the length of the slice is defined as
/// `max(2^log_height, 2^l_skip)`. In other words, this is a pointer to the slice
/// `q[commit_idx].column(stacked_col_idx)[stacked_row_idx..stacked_row_idx + len]`.
///
/// # Safety
/// - this type is `repr(C)` as it will cross FFI boundaries for CUDA usage.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnstackedSlice {
    commit_idx: u32,
    log_height: u32,
    stacked_row_idx: u32,
    stacked_col_idx: u32,
}

impl StackedReductionGpu {
    fn log_stacked_height(&self, round: usize) -> usize {
        self.n_stack - (round - 1)
    }

    fn stacked_height(&self, round: usize) -> usize {
        1 << self.log_stacked_height(round)
    }

    /// Current maximum `n` supported by eq_r_ns
    fn cur_max_n(&self, round: usize) -> usize {
        self.n_max - (round - 1)
    }
}

/// Batch sumcheck to reduce trace openings, including rotations, to stacked matrix opening.
///
/// The `stacked_matrix, stacked_layout` should be the result of stacking the `traces` with
/// parameters `l_skip` and `n_stack`.
#[instrument(
    name = "prover.openings.stacked_reduction",
    level = "info",
    skip_all,
    fields(phase = "prover")
)]
pub fn prove_stacked_opening_reduction_gpu(
    device: &GpuDeviceV2,
    transcript: &mut DuplexSpongeGpu,
    mpk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: ProvingContextV2<GpuBackendV2>,
    common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
    r: &[EF],
) -> Result<(StackingProof, Vec<EF>, Vec<StackedPcsData2>), ProverError> {
    let n_stack = device.config.n_stack;
    // Batching randomness
    let lambda = transcript.sample_ext();

    let _round0_span =
        info_span!("prover.openings.stacked_reduction.round0", phase = "prover").entered();
    let mut prover =
        StackedReductionGpu::new(mpk, ctx, common_main_pcs_data, r, lambda, device.sm_count())?;

    // Round 0: univariate sumcheck
    let s_0 = prover.batch_sumcheck_uni_round0_poly();
    for &coeff in s_0.coeffs() {
        transcript.observe_ext(coeff);
    }

    let mut u_vec = Vec::with_capacity(n_stack + 1);
    let u_0 = transcript.sample_ext();
    u_vec.push(u_0);
    debug!(round = 0, u_round = %u_0);

    prover.fold_ple_evals(u_0);
    drop(_round0_span);
    // end round 0

    let mut sumcheck_round_polys = Vec::with_capacity(n_stack);

    // Rounds 1..=n_stack: MLE sumcheck
    let _mle_rounds_span = info_span!(
        "prover.openings.stacked_reduction.mle_rounds",
        phase = "prover"
    )
    .entered();
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
    let stacking_openings = prover.get_stacked_openings();
    for claims_for_com in &stacking_openings {
        for &claim in claims_for_com {
            transcript.observe_ext(claim);
        }
    }
    drop(_mle_rounds_span);
    let proof = StackingProof {
        univariate_round_coeffs: s_0.into_coeffs(),
        sumcheck_round_polys,
        stacking_openings,
    };
    Ok((proof, u_vec, prover.stacked_per_commit))
}

impl StackedReductionGpu {
    #[instrument("stacked_reduction_new", level = "debug", skip_all)]
    fn new(
        mpk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        ctx: ProvingContextV2<GpuBackendV2>,
        common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
        r: &[EF],
        lambda: EF,
        sm_count: u32,
    ) -> Result<Self, ProverError> {
        ensure_device_ntt_twiddles_initialized();
        let mem = MemTracker::start("prover.stacked_reduction_new");
        let l_skip = mpk.params.l_skip;
        let n_stack = mpk.params.n_stack;

        let omega_skip = F::two_adic_generator(l_skip);
        let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();
        let d_omega_skip_pows = omega_skip_pows.to_device().unwrap();

        // NOTE: DeviceMatrix contains an Arc, so clone is lightweight [for now].
        // PERF[jpw]: stack the traces all at once. To save memory, we could either stack and drop
        // traces as we go or even at commit time only store the stacked matrix and drop the traces
        // much earlier.
        let common_main_traces = ctx
            .per_trace
            .iter()
            .map(|(_, air_ctx)| air_ctx.common_main.clone())
            .collect_vec();
        // SAFETY: common_main_traces commits to common_main_pcs_data
        let common_main_stacked = unsafe {
            StackedPcsData2::from_raw(Arc::new(common_main_pcs_data), common_main_traces)
        };
        let mut stacked_per_commit = vec![common_main_stacked];
        for (air_idx, air_ctx) in ctx.per_trace {
            for committed in mpk.per_air[air_idx]
                .preprocessed_data
                .iter()
                .chain(air_ctx.cached_mains.iter())
            {
                // SAFETY: committed.trace commits to committed.data
                let stacked = unsafe {
                    StackedPcsData2::from_raw(committed.data.clone(), vec![committed.trace.clone()])
                };
                stacked_per_commit.push(stacked);
            }
        }

        debug_assert!(
            stacked_per_commit
                .iter()
                .all(|d| d.layout().height() == 1 << (l_skip + n_stack))
        );
        let q_widths = stacked_per_commit
            .iter()
            .map(|d| d.layout().width() as u32)
            .collect_vec();
        let q_width_max = *q_widths.iter().max().unwrap();
        let d_q_widths = q_widths.to_device().unwrap();

        let total_num_col_openings = stacked_per_commit
            .iter()
            .map(|d| d.layout().sorted_cols.len() * 2) // 2 for [plain, rotated]
            .sum();
        let lambda_pows = lambda.powers().take(total_num_col_openings).collect_vec();
        let d_lambda_pows = lambda_pows.to_device().unwrap();

        let mut unstacked_cols = Vec::with_capacity(total_num_col_openings);
        let mut ht_diff_idxs = Vec::new();
        let mut trace_ptrs = Vec::new();
        for (commit_idx, stacked) in stacked_per_commit.iter().enumerate() {
            let layout = stacked.layout();
            for (trace, &idx) in zip_eq(&stacked.traces, &layout.mat_starts) {
                debug_assert_ne!(trace.width(), 0);
                debug_assert_ne!(trace.height(), 0);
                ht_diff_idxs.push(unstacked_cols.len());
                trace_ptrs.push((trace.buffer().as_ptr(), trace.height(), trace.width()));
                for j in 0..trace.width() {
                    let (_, _j, s) = layout.sorted_cols[idx + j];
                    debug_assert_eq!(_j, j);
                    debug_assert_eq!(1 << s.log_height(), trace.height());
                    unstacked_cols.push(UnstackedSlice {
                        commit_idx: commit_idx as u32,
                        log_height: s.log_height() as u32,
                        stacked_row_idx: s.row_idx as u32,
                        stacked_col_idx: s.col_idx as u32,
                    });
                }
            }
        }
        debug_assert_eq!(unstacked_cols.len(), total_num_col_openings / 2);
        ht_diff_idxs.push(unstacked_cols.len());

        let d_unstacked_cols = unstacked_cols.to_device().unwrap(); // TODO: handle error
        let max_window_len = ht_diff_idxs
            .windows(2)
            .map(|window| window[1] - window[0])
            .max()
            .unwrap_or(0);

        // layout per commit is sorted, first height is largest
        let n_max = r.len() - 1;
        debug_assert_eq!(
            n_max,
            stacked_per_commit
                .iter()
                .map(|d| d.layout().sorted_cols[0].2.log_height())
                .max()
                .unwrap_or(0)
                .saturating_sub(l_skip)
        );
        let eq_r_ns = EqEvalSegments::new(&r[1..]).unwrap(); // TODO: return error

        let eq_const = eval_eq_uni_at_one(l_skip, r[0] * omega_skip);
        let eq_ub_per_trace = vec![EF::ONE; unstacked_cols.len()];
        let d_q_eval_ptrs = if stacked_per_commit.is_empty() {
            DeviceBuffer::new()
        } else {
            DeviceBuffer::with_capacity(stacked_per_commit.len())
        };
        let d_input_ptrs = if stacked_per_commit.is_empty() {
            DeviceBuffer::new()
        } else {
            DeviceBuffer::with_capacity(stacked_per_commit.len())
        };
        let d_output_ptrs = if stacked_per_commit.is_empty() {
            DeviceBuffer::new()
        } else {
            DeviceBuffer::with_capacity(stacked_per_commit.len())
        };
        let d_accum = DeviceBuffer::<u64>::with_capacity(STACKED_REDUCTION_S_DEG * D_EF);
        let d_eq_ub = if max_window_len > 0 {
            DeviceBuffer::with_capacity(max_window_len)
        } else {
            DeviceBuffer::new()
        };

        Ok(Self {
            sm_count,
            l_skip,
            n_stack,
            omega_skip,
            omega_skip_pows,
            d_omega_skip_pows,
            r_0: r[0],
            d_lambda_pows,
            eq_const,
            stacked_per_commit,
            d_q_widths,
            q_width_max,
            d_q_eval_ptrs,
            trace_ptrs,
            unstacked_cols,
            d_unstacked_cols,
            ht_diff_idxs,
            n_max,
            eq_r_ns,
            q_evals: vec![],
            eq_stable: vec![],
            k_rot_stable: vec![],
            // SAFETY: This is unused in round 0 and will be initialized properly after round 0.
            k_rot_ns: unsafe { EqEvalSegments::from_raw_parts(DeviceBuffer::new(), 0) },
            eq_ub_per_trace,
            d_eq_ub,
            d_block_sums: DeviceBuffer::new(),
            d_accum,
            d_input_ptrs,
            d_output_ptrs,
            mem,
        })
    }

    /// SP_DEG=1 round 0: computes G0, G1, G2 on identity coset, then reconstructs s_0 on CPU.
    ///
    /// Key insight: Instead of computing s₀(Z) directly on 2 cosets with in-kernel NTT,
    /// we compute three partial sums G0, G1, G2 on the identity coset only, then
    /// reconstruct s₀ via CPU-side NTT-based polynomial multiplication.
    #[instrument(
        "stacked_reduction_sumcheck",
        level = "debug",
        skip_all,
        fields(round = 0)
    )]
    fn batch_sumcheck_uni_round0_poly(&mut self) -> UnivariatePoly<EF> {
        let l_skip = self.l_skip;
        let skip_domain = 1 << l_skip;
        let s_0_deg = sumcheck_round0_deg(l_skip, STACKED_REDUCTION_S_DEG);

        // Accumulation buffers for G0, G1, G2 (on identity coset)
        // d_g_pos: for n >= 0 traces
        let mut d_g_pos = DeviceBuffer::<EF>::with_capacity(NUM_G * skip_domain);
        d_g_pos.fill_zero().unwrap();

        // d_g_neg[k]: for traces with |n| = k+1, where n < 0 and k in 0..l_skip
        let mut d_g_neg: Vec<DeviceBuffer<EF>> = (0..l_skip)
            .map(|_| {
                let b = DeviceBuffer::with_capacity(NUM_G * skip_domain);
                b.fill_zero().unwrap();
                b
            })
            .collect();

        // Process each trace - call kernel for each, accumulating into appropriate bucket
        for ((trace_ptr, trace_height, trace_width), window) in zip(
            mem::take(&mut self.trace_ptrs),
            self.ht_diff_idxs.windows(2),
        ) {
            debug_assert_eq!(window[1] - window[0], trace_width);
            let log_height = trace_height.ilog2();
            let n = log_height as isize - l_skip as isize;

            // Select output bucket based on n
            let d_g_output = if n >= 0 {
                &mut d_g_pos
            } else {
                &mut d_g_neg[(-n - 1) as usize]
            };

            // Allocate block_sums buffer for intermediate reduction
            let block_sums_len = unsafe {
                _stacked_reduction_r0_required_temp_buffer_size(
                    trace_height as u32,
                    trace_width as u32,
                    l_skip as u32,
                )
            } as usize;

            if block_sums_len > self.d_block_sums.len() {
                self.d_block_sums = DeviceBuffer::<EF>::with_capacity(block_sums_len);
            }

            unsafe {
                // 2 per column for (eq, k_rot) - coeff_eq and coeff_rot
                let lambda_pows_ptr = self.d_lambda_pows.as_ptr().add(2 * window[0]);

                stacked_reduction_sumcheck_round0(
                    &self.eq_r_ns,
                    trace_ptr,
                    lambda_pows_ptr,
                    &mut self.d_block_sums,
                    d_g_output,
                    trace_height,
                    trace_width,
                    l_skip,
                )
                .unwrap();
            };
        }

        // CPU reconstruction: s₀(Z) = E0(Z)*G0(Z) + E1(Z)*G1(Z) + E2(Z)*G2(Z)
        let s_0 = self.reconstruct_s0_from_g(d_g_pos, d_g_neg, s_0_deg);
        self.mem.tracing_info("stacked_reduction_sumcheck round 0");

        s_0
    }

    /// Reconstructs s_0 from G0, G1, G2 using NTT-based polynomial multiplication.
    ///
    /// For n >= 0:
    /// - E0(Z) = eq_uni(l_skip, Z, r0)
    /// - E1(Z) = eq_uni(l_skip, Z, r0*ω_skip)
    /// - E2(Z) = eq_const * eq_uni_at_one(l_skip, Z)
    ///
    /// For n < 0: multiply each E by ind(Z) = eval_in_uni(l_skip, n, Z)
    fn reconstruct_s0_from_g(
        &self,
        d_g_pos: DeviceBuffer<EF>,
        d_g_neg: Vec<DeviceBuffer<EF>>,
        s_0_deg: usize,
    ) -> UnivariatePoly<EF> {
        let l_skip = self.l_skip;
        let skip_domain = 1 << l_skip;
        let large_uni_domain = (s_0_deg + 1).next_power_of_two(); // 2 * skip_domain
        let dft = Radix2BowersSerial;

        // Accumulate s_0 coefficients across all buckets
        let mut s_0_coeffs = vec![EF::ZERO; large_uni_domain];

        // --- Process n >= 0 bucket ---
        let g_pos = d_g_pos.to_host().unwrap();
        if !g_pos.iter().all(|&x| x == EF::ZERO) {
            // Build E polynomials for n >= 0
            let e0 = eq_uni_poly::<F, EF>(l_skip, self.r_0);
            let e1 = eq_uni_poly::<F, EF>(l_skip, self.r_0 * self.omega_skip);
            let e2 = eq_uni_at_one_poly(l_skip, self.eq_const);

            // NTT-based multiplication: s_0 += E0*G0 + E1*G1 + E2*G2
            Self::ntt_multiply_and_add(
                &dft,
                large_uni_domain,
                [e0.coeffs(), e1.coeffs(), e2.coeffs()],
                [
                    &g_pos[0..skip_domain],
                    &g_pos[skip_domain..2 * skip_domain],
                    &g_pos[2 * skip_domain..3 * skip_domain],
                ],
                &mut s_0_coeffs,
            );
        }

        // --- Process n < 0 buckets ---
        for (bucket_idx, d_g_neg_bucket) in d_g_neg.into_iter().enumerate() {
            let n_abs = bucket_idx + 1;
            let g_neg = d_g_neg_bucket.to_host().unwrap();
            if g_neg.iter().all(|&x| x == EF::ZERO) {
                continue;
            }

            // Adjusted parameters for n < 0
            let l = l_skip - n_abs;
            let omega_l = self.omega_skip.exp_power_of_2(n_abs);
            let r_uni = self.r_0.exp_power_of_2(n_abs);

            // Build E polynomials with indicator factor
            let ind = build_indicator_poly(l_skip, -(n_abs as isize));
            let e0_base = eq_uni_poly::<F, EF>(l, r_uni);
            let e1_base = eq_uni_poly::<F, EF>(l, r_uni * omega_l);
            let e2_base = eq_uni_at_one_poly(l, self.eq_const);

            // E_neg = E_base * ind (polynomial multiplication)
            let e0_neg = poly_multiply_ntt(&dft, e0_base.coeffs(), ind.coeffs(), skip_domain);
            let e1_neg = poly_multiply_ntt(&dft, e1_base.coeffs(), ind.coeffs(), skip_domain);
            let e2_neg = poly_multiply_ntt(&dft, e2_base.coeffs(), ind.coeffs(), skip_domain);

            Self::ntt_multiply_and_add(
                &dft,
                large_uni_domain,
                [&e0_neg, &e1_neg, &e2_neg],
                [
                    &g_neg[0..skip_domain],
                    &g_neg[skip_domain..2 * skip_domain],
                    &g_neg[2 * skip_domain..3 * skip_domain],
                ],
                &mut s_0_coeffs,
            );
        }

        s_0_coeffs.truncate(s_0_deg + 1);
        UnivariatePoly::new(s_0_coeffs)
    }

    /// NTT-based polynomial multiplication following logup pattern.
    /// Computes: out += sum_i E[i] * G[i]
    fn ntt_multiply_and_add(
        dft: &Radix2BowersSerial,
        domain_size: usize,
        e_coeffs: [&[EF]; 3],
        g_evals: [&[EF]; 3], // G evaluations on identity coset
        out: &mut [EF],
    ) {
        // 1. iDFT G evaluations to get G coefficients
        let g_coeffs: [Vec<EF>; 3] = std::array::from_fn(|i| dft.idft(g_evals[i].to_vec()));

        // 2. Prepare coefficient matrices, resize to domain_size
        let mut e_padded = vec![EF::ZERO; domain_size * 3];
        let mut g_padded = vec![EF::ZERO; domain_size * 3];
        for i in 0..3 {
            for (j, &c) in e_coeffs[i].iter().enumerate() {
                e_padded[j * 3 + i] = c;
            }
            for (j, &c) in g_coeffs[i].iter().enumerate() {
                g_padded[j * 3 + i] = c;
            }
        }

        // 3. DFT batch to evaluation domain
        let e_evals_mat = dft.dft_batch(RowMajorMatrix::new(e_padded, 3));
        let g_evals_mat = dft.dft_batch(RowMajorMatrix::new(g_padded, 3));

        // 4. Pointwise multiply and sum: s[j] = sum_i e[j][i] * g[j][i]
        let mut s_evals = vec![EF::ZERO; domain_size];
        for j in 0..domain_size {
            for i in 0..3 {
                s_evals[j] += e_evals_mat.values[j * 3 + i] * g_evals_mat.values[j * 3 + i];
            }
        }

        // 5. iDFT to get product coefficients
        let s_coeffs = dft.idft(s_evals);

        // 6. Add to output
        for (o, c) in out.iter_mut().zip(s_coeffs) {
            *o += c;
        }
    }

    #[instrument("stacked_reduction_fold_ple", level = "debug", skip_all)]
    fn fold_ple_evals(&mut self, u_0: EF) {
        let l_skip = self.l_skip;
        let n_stack = self.n_stack;
        let r_0 = self.r_0;
        let omega_skip = self.omega_skip;
        let n_max = self.n_max;
        self.q_evals.clear();

        // Precompute Lagrange denominators once (shared across all traces)
        let skip_domain = 1 << l_skip;
        let inv_lagrange_denoms =
            compute_barycentric_inv_lagrange_denoms(l_skip, &self.omega_skip_pows, u_0);
        let d_inv_lagrange_denoms = inv_lagrange_denoms.to_device().unwrap();

        for stacked in &self.stacked_per_commit {
            let layout = stacked.layout();
            let num_x = 1 << n_stack;
            let stacked_width = layout.width();
            debug_assert_eq!(layout.height(), 1 << (l_skip + n_stack));
            let folded_evals = DeviceBuffer::<EF>::with_capacity(num_x * stacked_width);
            // We must fill with zeros because some parts will be left empty due to stacking
            folded_evals.fill_zero().unwrap();
            let mut dst_offset = 0;
            for trace in &stacked.traces {
                if trace.width() == 0 || trace.height() == 0 {
                    continue;
                }
                let new_height = max(trace.height(), skip_domain) / skip_domain;

                // Launch single-trace kernel for this trace
                // SAFETY:
                // - `trace.buffer()` is a valid device pointer for `trace.height() * trace.width()`
                //   elements
                // - `folded_evals` at `dst_offset` is valid for `new_height * trace.width()`
                //   elements since we allocated `num_x * stacked_width` and traces fill
                //   contiguously
                // - `d_omega_skip_pows` and `d_inv_lagrange_denoms` have length `>= skip_domain`
                unsafe {
                    let dst = folded_evals.as_mut_ptr().add(dst_offset);
                    stacked_reduction_fold_ple(
                        trace.buffer().as_ptr(),
                        dst,
                        &self.d_omega_skip_pows,
                        &d_inv_lagrange_denoms,
                        trace.height(),
                        trace.width(),
                        l_skip,
                    )
                    .unwrap();
                }

                dst_offset += new_height * trace.width();
            }
            self.q_evals.push(folded_evals);
        }

        // fold PLEs into MLEs for \eq and \kappa_\rot, using u_0
        let eq_uni_u0r0 = eval_eq_uni(l_skip, u_0, r_0);
        let eq_uni_u0r0_rot = eval_eq_uni(l_skip, u_0, r_0 * omega_skip);
        let eq_uni_u01 = eval_eq_uni_at_one(l_skip, u_0);
        debug_assert_eq!(self.eq_r_ns.buffer.len(), 2 << n_max);
        self.k_rot_ns.buffer = DeviceBuffer::with_capacity(2 << n_max);
        [EF::ZERO].copy_to(&mut self.k_rot_ns.buffer).unwrap();
        unsafe {
            // SAFETY:
            // - We allocated `k_rot_ns` with same capacity as `eq_r_ns` above.
            initialize_k_rot_from_eq_segments(
                &self.eq_r_ns,
                &mut self.k_rot_ns.buffer,
                eq_uni_u0r0_rot,
                self.eq_const * eq_uni_u01,
                n_max as u32,
            )
            .unwrap();
        }
        vector_scalar_multiply_ext(&mut self.eq_r_ns.buffer, eq_uni_u0r0).unwrap();

        // Compute the special eq values for n = -l_skip..0
        // First in order -n = 1..=l_skip, then reverse to order n = -l_skip..0 corresponding to
        // log_height = 0..l_skip
        (self.eq_stable, self.k_rot_stable) =
            zip(r_0.exp_powers_of_2(), omega_skip.exp_powers_of_2())
                .enumerate()
                .skip(1)
                .take(l_skip)
                .map(|(n_abs, (r, omega_l))| {
                    let l = l_skip - n_abs;
                    let eq_uni = eval_eq_uni(l, u_0, r);
                    let eq_uni_rot = eval_eq_uni(l, u_0, r * omega_l);
                    let ind = eval_in_uni(l_skip, -(n_abs as isize), u_0);
                    (ind * eq_uni, ind * eq_uni_rot)
                })
                .unzip();
        self.eq_stable.reverse();
        self.k_rot_stable.reverse();
    }

    #[instrument("stacked_reduction_sumcheck", level = "debug", skip_all, fields(round = round))]
    fn batch_sumcheck_poly_eval(
        &mut self,
        round: usize,
        _u_prev: EF,
    ) -> [EF; STACKED_REDUCTION_S_DEG] {
        let l_skip = self.l_skip;

        let q_eval_ptrs = self.q_evals.iter().map(|q| q.as_ptr()).collect_vec();
        q_eval_ptrs.copy_to(&mut self.d_q_eval_ptrs).unwrap();

        if self.n_max >= (round - 1) {
            // Move stable eq, k_rot to stable vectors
            let mut tmp = [EF::ZERO];
            debug_assert_eq!(self.eq_stable.len(), l_skip + round - 1);
            debug_assert_eq!(self.k_rot_stable.len(), l_skip + round - 1);
            debug_assert!(self.eq_r_ns.buffer.len() > 1);
            debug_assert!(self.k_rot_ns.buffer.len() > 1);
            // SAFETY: size of eq_r_ns, k_rot_ns is currently 2 * 2^{n_max - round + 1}
            unsafe {
                // D2H copy of single EF element
                cuda_memcpy::<true, false>(
                    tmp.as_mut_ptr() as *mut c_void,
                    self.eq_r_ns.get_ptr(0) as *const c_void,
                    size_of::<EF>(),
                )
                .unwrap();

                self.eq_stable.push(tmp[0]);

                // D2H copy of single EF element
                cuda_memcpy::<true, false>(
                    tmp.as_mut_ptr() as *mut c_void,
                    self.k_rot_ns.get_ptr(0) as *const c_void,
                    size_of::<EF>(),
                )
                .unwrap();

                self.k_rot_stable.push(tmp[0]);
            }
        }
        let mut s_evals_batch = Vec::with_capacity(self.ht_diff_idxs.len() - 1);
        for window in self.ht_diff_idxs.windows(2) {
            let window_len = window[1] - window[0];
            // SAFETY: in bounds by construction of ht_diff_idxs
            let unstacked_cols_ptr = unsafe { self.d_unstacked_cols.as_ptr().add(window[0]) };
            // 2 per column for (eq, k_rot)
            // SAFETY: in bounds by construction of lambda_pows
            let lambda_pows_ptr = unsafe { self.d_lambda_pows.as_ptr().add(2 * window[0]) };

            let log_height = self.unstacked_cols[window[0]].log_height as usize;

            // Zero-initialize accumulator for atomic adds
            self.d_accum.fill_zero().unwrap();

            if log_height < l_skip + round {
                // We are in the eq, k_rot stable regime
                // This includes all n < 0 cases
                // In this case, the `s` poly contribution is a constant and we don't need to
                // interpolate
                let eq_r = self.eq_stable[log_height];
                let k_rot_r = self.k_rot_stable[log_height];
                // PERF[jpw]: most of eq_ub can be incorporated into eq_stable, k_rot_stable, so
                // this transfer can be minimized.
                let eq_ub_slice = &self.eq_ub_per_trace[window[0]..window[1]];
                if eq_ub_slice.len() > self.d_eq_ub.len() {
                    self.d_eq_ub = DeviceBuffer::with_capacity(eq_ub_slice.len());
                }
                eq_ub_slice.copy_to(&mut self.d_eq_ub).unwrap();
                let stacked_height = self.stacked_height(round);
                unsafe {
                    stacked_reduction_sumcheck_mle_round_degenerate(
                        &self.d_q_eval_ptrs,
                        &self.d_eq_ub,
                        eq_r,
                        k_rot_r,
                        unstacked_cols_ptr,
                        lambda_pows_ptr,
                        &mut self.d_accum,
                        stacked_height,
                        window_len,
                        l_skip,
                        round,
                    )
                    .unwrap();
                }
            } else {
                let hypercube_dim = log_height - l_skip - round;
                let num_y = 1 << hypercube_dim;
                // Allow the CUDA launcher to auto-tune grid.y (thread_window_stride) based on
                // (num_y, window_len) and device SM count.

                let stacked_height = self.stacked_height(round);
                unsafe {
                    stacked_reduction_sumcheck_mle_round(
                        &self.d_q_eval_ptrs,
                        &self.eq_r_ns,
                        &self.k_rot_ns,
                        unstacked_cols_ptr,
                        lambda_pows_ptr,
                        &mut self.d_accum,
                        stacked_height,
                        window_len,
                        num_y,
                        self.sm_count,
                    )
                    .unwrap();
                };
            }

            // D2H copy and reduce modulo P
            let h_accum = self.d_accum.to_host().unwrap();
            let evals = reduce_raw_u64_to_ef(&h_accum);
            s_evals_batch.push(evals);
        }

        from_fn(|i| s_evals_batch.iter().map(|evals| evals[i]).sum::<EF>())
    }

    #[instrument("stacked_reduction_fold_mle", level = "debug", skip_all, fields(round = round))]
    fn fold_mle_evals(&mut self, round: usize, u_round: EF) {
        debug_assert!(round <= self.n_stack);
        let l_skip = self.l_skip;
        let (folded_q_evals, input_ptrs, output_ptrs): (Vec<_>, Vec<_>, Vec<_>) = self
            .q_evals
            .iter()
            .map(|q| {
                let folded = DeviceBuffer::with_capacity(q.len() >> 1);
                let output_ptr = folded.as_mut_ptr();
                (folded, q.as_ptr(), output_ptr)
            })
            .multiunzip();
        input_ptrs.copy_to(&mut self.d_input_ptrs).unwrap();
        output_ptrs.copy_to(&mut self.d_output_ptrs).unwrap();

        // SAFETY:
        // - `d_input_ptrs` points to matrices with widths specified by `d_q_widths` and heights
        //   `stacked_height(round) = stacked_height(round + 1) * 2`.
        // - `d_output_ptrs` points to matrices just allocated with widths specified by `d_q_widths`
        //   and heights `stacked_height(round + 1)`.
        let output_height = self.stacked_height(round + 1) as u32;
        unsafe {
            fold_mle(
                &self.d_input_ptrs,
                &self.d_output_ptrs,
                &self.d_q_widths,
                self.q_evals.len().try_into().unwrap(),
                self.stacked_height(round + 1) as u32,
                self.q_width_max * output_height,
                u_round,
            )
            .unwrap();
        }
        self.q_evals = folded_q_evals;

        if self.n_max >= (round - 1) {
            let input_max_n = self.cur_max_n(round);
            let output_max_n = input_max_n.saturating_sub(1);
            let output_len = 1 << input_max_n;

            let mut buffer = DeviceBuffer::<EF>::with_capacity(output_len);
            [EF::ZERO].copy_to(&mut buffer).unwrap();
            // SAFETY:
            // - eq_r_ns has max_n equal to input_max_n
            // - we allocate output for half the size of eq_r_ns
            unsafe {
                let mut output = EqEvalSegments::from_raw_parts(buffer, output_max_n);
                if input_max_n != 0 {
                    triangular_fold_mle(&mut output, &self.eq_r_ns, u_round, output_max_n).unwrap();
                }
                self.eq_r_ns = output;
            }

            let mut buffer = DeviceBuffer::<EF>::with_capacity(output_len);
            [EF::ZERO].copy_to(&mut buffer).unwrap();
            // SAFETY:
            // - k_rot_ns has max_n equal to input_max_n
            // - we allocate output for half the size of eq_r_ns
            unsafe {
                let mut output = EqEvalSegments::from_raw_parts(buffer, output_max_n);
                if input_max_n != 0 {
                    triangular_fold_mle(&mut output, &self.k_rot_ns, u_round, output_max_n)
                        .unwrap();
                }
                self.k_rot_ns = output;
            }
        } else {
            assert_eq!(self.eq_r_ns.buffer.len(), 1);
            assert_eq!(self.k_rot_ns.buffer.len(), 1);
        }
        for (s, eq_ub) in zip(&self.unstacked_cols, &mut self.eq_ub_per_trace) {
            if round + l_skip > s.log_height as usize {
                // Folding above did nothing, and we update the eq(u[1+n_T..=round],
                // b_{T,j}[..=round-n_T-1]) value
                debug_assert_eq!(s.stacked_row_idx % (1 << s.log_height), 0);
                let b = (s.stacked_row_idx >> (l_skip + round - 1)) & 1;
                *eq_ub *= eval_eq_mle(&[u_round], &[F::from_bool(b == 1)]);
            }
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn get_stacked_openings(&self) -> Vec<Vec<EF>> {
        self.q_evals.iter().map(|q| q.to_host().unwrap()).collect()
    }
}

/// Build indicator polynomial: ind(Z) = sum_{k=0}^{2^{n_abs}-1} Z^{k * 2^l} / 2^{n_abs}
fn build_indicator_poly(l_skip: usize, n: isize) -> UnivariatePoly<EF> {
    let n_abs = (-n) as usize;
    let l = l_skip - n_abs;
    let scale = EF::ONE.halve().exp_u64(n_abs as u64);
    let mut coeffs = vec![EF::ZERO; 1 << l_skip];
    for k in 0..(1 << n_abs) {
        coeffs[k * (1 << l)] = scale;
    }
    UnivariatePoly::new(coeffs)
}

/// eq_uni_at_one polynomial: eq_D(Z, 1) as a polynomial in Z
///
/// All coefficients are n_inv * scale where n_inv = 1 / 2^l
fn eq_uni_at_one_poly(l: usize, scale: EF) -> UnivariatePoly<EF> {
    let n_inv = F::ONE.halve().exp_u64(l as u64);
    UnivariatePoly::new(vec![EF::from(n_inv) * scale; 1 << l])
}

/// NTT-based polynomial multiplication
fn poly_multiply_ntt(dft: &Radix2BowersSerial, a: &[EF], b: &[EF], min_size: usize) -> Vec<EF> {
    let size = (a.len() + b.len() - 1).max(min_size).next_power_of_two();
    let mut a_pad = a.to_vec();
    a_pad.resize(size, EF::ZERO);
    let mut b_pad = b.to_vec();
    b_pad.resize(size, EF::ZERO);
    let a_evals = dft.dft(a_pad);
    let b_evals = dft.dft(b_pad);
    let c_evals: Vec<EF> = a_evals
        .into_iter()
        .zip(b_evals)
        .map(|(a, b)| a * b)
        .collect();
    dft.idft(c_evals)
}
