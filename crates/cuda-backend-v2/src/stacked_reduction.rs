use std::{array::from_fn, ffi::c_void, iter::zip, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::{base::DeviceMatrix, ntt::batch_ntt};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_util::log2_ceil_usize;
use stark_backend_v2::{
    poly_common::{
        Squarable, UnivariatePoly, eval_eq_mle, eval_eq_uni, eval_eq_uni_at_one, eval_in_uni,
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
    Digest, EF, F, GpuBackendV2, GpuDeviceV2, ProverError,
    cuda::{
        matrix::batch_expand_pad_wide,
        poly::vector_scalar_multiply_ext,
        stacked_reduction::{
            _stacked_reduction_mle_required_temp_buffer_size,
            _stacked_reduction_r0_required_temp_buffer_size, initialize_k_rot_from_eq_segments,
            stacked_reduction_sumcheck_mle_round, stacked_reduction_sumcheck_mle_round_degenerate,
            stacked_reduction_sumcheck_round0,
        },
        sumcheck::{fold_mle, fold_ple_from_coeffs, triangular_fold_mle},
    },
    poly::{EqEvalSegments, PleMatrix},
    stacked_pcs::{StackedPcsDataGpu, stack_traces},
};

pub struct StackedReductionGpu {
    l_skip: usize,
    omega_skip: F,
    n_stack: usize,

    r_0: EF,
    d_lambda_pows: DeviceBuffer<EF>,
    eq_const: EF,

    pub(crate) stacked_per_commit: Vec<StackedPcsData2>,
    d_q_widths: DeviceBuffer<u32>,

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

    mem: MemTracker,
}

/// A struct for holding stacked pcs data. Since `StackedPcsDataGpu` may not contain `matrix` if
/// prover configuration does not cache it, we generate the matrix from traces and store in
/// `stacked_matrix`. It will be guaranateed that either `inner.matrix` or `stacked_matrix` is
/// present.
pub struct StackedPcsData2 {
    pub(crate) inner: Arc<StackedPcsDataGpu<F, Digest>>,
    pub(crate) stacked_matrix: Option<PleMatrix<F>>,
}

impl StackedPcsData2 {
    /// # Safety
    /// `traces` must be the traces that were committed to in `pcs_data`.
    pub unsafe fn from_raw(
        pcs_data: Arc<StackedPcsDataGpu<F, Digest>>,
        traces: &[&DeviceMatrix<F>],
    ) -> Result<Self, ProverError> {
        if pcs_data.matrix.is_some() {
            Ok(Self {
                inner: pcs_data,
                stacked_matrix: None,
            })
        } else {
            let layout = &pcs_data.layout;
            let matrix = stack_traces(layout, traces)?;
            Ok(Self {
                inner: pcs_data,
                stacked_matrix: Some(matrix),
            })
        }
    }

    pub fn layout(&self) -> &StackedLayout {
        &self.inner.layout
    }

    pub fn matrix(&self) -> &PleMatrix<F> {
        self.inner
            .matrix
            .as_ref()
            .or(self.stacked_matrix.as_ref())
            .unwrap()
    }
}

/// Pointer with length to location in a big device buffer. The device buffer is identified by
/// `commit_idx`. Due to pecuarlities with `l_skip`, the length of the slice is defined as
/// `max(2^log_height, 2^l_skip)`. In other words, this is a pointer to the slice
/// `q[commit_idx].column(stacked_col_idx)[stacked_row_idx..stacked_row_idx + len]`.
///
/// Note: this type is `repr(C)` as it will cross FFI boundaries for CUDA usage.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnstackedSlice {
    commit_idx: u32,
    stacked_row_idx: u32,
    stacked_col_idx: u32,
    log_height: u32,
}

/// Data that only needs to be computed per univariate coordinate z (2^log_domain_size many) in
/// round 0 sumcheck. Seems easiest to just compute on CPU and H2D transfer.
///
/// Terms include the `ind` factor for `n < 0`.
#[repr(C)]
pub(crate) struct Round0UniPacket {
    eq_uni: EF,  // to multiply by eq_cube
    k_rot_0: EF, // to multiply be eq_cube
    k_rot_1: EF, // to multiply by k_rot_cube - eq_cube
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
pub fn prove_stacked_opening_reduction_gpu<TS>(
    device: &GpuDeviceV2,
    transcript: &mut TS,
    mpk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: ProvingContextV2<GpuBackendV2>,
    common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
    r: &[EF],
) -> Result<(StackingProof, Vec<EF>, Vec<StackedPcsData2>), ProverError>
where
    TS: FiatShamirTranscript,
{
    let n_stack = device.config.n_stack;
    // Batching randomness
    let lambda = transcript.sample_ext();

    let _round0_span = info_span!("stacked_reduction.round0", phase = "prover").entered();
    let mut prover = StackedReductionGpu::new(mpk, ctx, common_main_pcs_data, r, lambda)?;

    // Round 0: univariate sumcheck
    let s_0 = prover.batch_sumcheck_uni_round0_poly();
    for &coeff in s_0.coeffs() {
        transcript.observe_ext(coeff);
    }
    drop(_round0_span);

    let mut u_vec = Vec::with_capacity(n_stack + 1);
    let u_0 = transcript.sample_ext();
    u_vec.push(u_0);
    debug!(round = 0, u_round = %u_0);

    prover.fold_ple_evals(u_0);
    // end round 0

    let mut sumcheck_round_polys = Vec::with_capacity(n_stack);

    // Rounds 1..=n_stack: MLE sumcheck
    let _mle_rounds_span = info_span!("stacked_reduction.mle_rounds", phase = "prover").entered();
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
    drop(_mle_rounds_span);
    let stacking_openings = prover.get_stacked_openings();
    for claims_for_com in &stacking_openings {
        for &claim in claims_for_com {
            transcript.observe_ext(claim);
        }
    }
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
        mut ctx: ProvingContextV2<GpuBackendV2>,
        common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
        r: &[EF],
        lambda: EF,
    ) -> Result<Self, ProverError> {
        let mem = MemTracker::start("prover.stacked_reduction_new");
        let l_skip = mpk.params.l_skip;
        let omega_skip = F::two_adic_generator(l_skip);
        let n_stack = mpk.params.n_stack;

        // PERF[jpw]: stack the traces all at once. To save memory, we could either stack and drop
        // traces as we go or even at commit time only store the stacked matrix and drop the traces
        // much earlier.
        let common_main_traces = ctx
            .per_trace
            .iter()
            .map(|(_, air_ctx)| &air_ctx.common_main)
            .collect_vec();
        // SAFETY: common_main_traces commits to common_main_pcs_data
        let common_main_stacked = unsafe {
            StackedPcsData2::from_raw(Arc::new(common_main_pcs_data), &common_main_traces)?
        };
        let mut stacked_per_commit = vec![common_main_stacked];
        // Drop all traces
        for (_, air_ctx) in &mut ctx.per_trace {
            air_ctx.common_main = DeviceMatrix::dummy();
        }
        for (air_idx, air_ctx) in ctx.per_trace {
            for committed in mpk.per_air[air_idx]
                .preprocessed_data
                .iter()
                .chain(air_ctx.cached_mains.iter())
            {
                // SAFETY: committed.trace commits to committed.data
                let stacked = unsafe {
                    StackedPcsData2::from_raw(committed.data.clone(), &[&committed.trace])?
                };
                stacked_per_commit.push(stacked);
            }
        }
        // ctx is dropped as this point

        debug_assert!(
            stacked_per_commit
                .iter()
                .all(|d| d.matrix().height() == 1 << (l_skip + n_stack))
        );
        let widths = stacked_per_commit
            .iter()
            .map(|d| d.matrix().width() as u32)
            .collect_vec();
        let d_q_widths = widths.to_device().unwrap();

        let total_num_col_openings = stacked_per_commit
            .iter()
            .map(|d| d.layout().sorted_cols.len() * 2) // 2 for [plain, rotated]
            .sum();
        let lambda_pows = lambda.powers().take(total_num_col_openings).collect_vec();
        let d_lambda_pows = lambda_pows.to_device().unwrap();

        let unstacked_cols = stacked_per_commit
            .iter()
            .enumerate()
            .flat_map(|(com_idx, d)| {
                d.layout()
                    .unstacked_slices_iter()
                    .map(move |s| UnstackedSlice {
                        commit_idx: com_idx as u32,
                        stacked_row_idx: s.row_idx as u32,
                        stacked_col_idx: s.col_idx as u32,
                        log_height: s.log_height() as u32,
                    })
            })
            .collect_vec();
        let d_unstacked_cols = unstacked_cols.to_device().unwrap(); // TODO: handle error

        let mut ht_diff_idxs = Vec::new();
        let mut last_height = 0;
        for (i, s) in unstacked_cols.iter().enumerate() {
            if i == 0 || s.log_height != last_height {
                ht_diff_idxs.push(i);
                last_height = s.log_height;
            }
        }
        ht_diff_idxs.push(unstacked_cols.len());

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

        Ok(Self {
            l_skip,
            omega_skip,
            n_stack,
            r_0: r[0],
            d_lambda_pows,
            eq_const,
            stacked_per_commit,
            d_q_widths,
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
            mem,
        })
    }

    #[instrument(
        "stacked_reduction_sumcheck",
        level = "debug",
        skip_all,
        fields(round = 0)
    )]
    fn batch_sumcheck_uni_round0_poly(&mut self) -> UnivariatePoly<EF> {
        let l_skip = self.l_skip;
        let omega_skip = self.omega_skip;
        // +1 from eq term
        let s_0_deg = sumcheck_round0_deg(l_skip, 2);

        let log_domain_size = log2_ceil_usize(s_0_deg + 1);
        let domain_size = 1 << log_domain_size;
        let mut d_s0_evals = DeviceBuffer::<EF>::with_capacity(domain_size);
        // TODO/PERF[jpw]: Can upsampling be cached and shared between RS-code, Logup/Zerocheck,
        // StackedReduction? The latter domain size varies by l_skip and constraint degree (for
        // Logup/Zerocheck)
        let log_expansion = log_domain_size - l_skip;
        // NOTE: upsampling entire matrix altogether for convenience and best parallelism. This uses
        // `size_of(q) * 2` extra memory. We could instead do upsampling in chunks when iterating
        // over `ht_diff_idxs`
        let q_upsampled_evals = self
            .stacked_per_commit
            .iter()
            .map(|pcs_data| {
                let q_mixed = &pcs_data.matrix().mixed;
                let len = q_mixed.len();
                let upsampled = DeviceBuffer::with_capacity(len << log_expansion);
                // SAFETY:
                // - We allocated `upsampled` with `len * 2^(log_domain_size - l_skip)`.
                // - We chunk `q_mixed` into `len / 2^l_skip` univariate polynomials of size
                //   `2^l_skip`.
                // - By definition `q_mixed` is in coefficient form for these univariate
                //   polynomials.
                // - We expand each one from size `2^l_skip` to `2^log_domain_size
                // - We apply batch NTT of size 2^log_domain_size on the expanded polynomials.
                debug_assert_eq!(len % (1 << l_skip), 0);
                let num_polys = (len >> l_skip) as u32;
                unsafe {
                    batch_expand_pad_wide(
                        upsampled.as_mut_ptr(),
                        q_mixed.as_ptr(),
                        num_polys,
                        domain_size as u32,
                        1 << l_skip,
                    )?;
                    batch_ntt(
                        &upsampled,
                        log_domain_size as u32,
                        0,
                        num_polys,
                        true,
                        false, /* forward NTT */
                    );
                }
                Ok(upsampled)
            })
            .collect::<Result<Vec<_>, ProverError>>()
            .unwrap();
        let upsampled_height = self.stacked_per_commit[0].matrix().height() << log_expansion;
        let q_upsampled_ptrs = q_upsampled_evals.iter().map(|q| q.as_ptr()).collect_vec();
        let d_q_upsampled_ptrs = q_upsampled_ptrs.to_device().unwrap();

        let omega = F::two_adic_generator(log_domain_size);
        // Default packets for n >= 0
        let default_packets = omega
            .powers()
            .take(1 << log_domain_size)
            .map(|z| {
                let eq_uni_r0 = eval_eq_uni(l_skip, z.into(), self.r_0);
                let eq_uni_r0_rot = eval_eq_uni(l_skip, z.into(), self.r_0 * omega_skip);
                let eq_uni_1 = eval_eq_uni_at_one(l_skip, z);
                Round0UniPacket {
                    eq_uni: eq_uni_r0,
                    k_rot_0: eq_uni_r0_rot,
                    k_rot_1: self.eq_const * eq_uni_1,
                }
            })
            .collect_vec();
        let d_default_packets = default_packets.to_device().unwrap();

        // The main point is that stacking reduction is a batch sumcheck over D_{n_stack}, batching
        // over all unstacked columns. However instead of naively computing the batching in that
        // way, we use the special definition of the sumcheck in terms of unstacked traces to sum
        // over smaller domains on a per-trace basis.
        let mut s_0_evals_batch = Vec::with_capacity(self.ht_diff_idxs.len() - 1);
        for window in self.ht_diff_idxs.windows(2) {
            let window_len = window[1] - window[0];
            let log_height = self.unstacked_cols[window[0]].log_height;
            let n = log_height as isize - l_skip as isize;
            let n_lift = n.max(0) as usize;

            let d_special_packets;
            let z_packets = if n.is_negative() {
                let l = l_skip.wrapping_add_signed(n);
                let omega_l = omega_skip.exp_power_of_2(-n as usize);
                let r_uni = self.r_0.exp_power_of_2(-n as usize);
                let z_packets = omega
                    .powers()
                    .enumerate()
                    .take(1 << log_domain_size)
                    .map(|(i, z)| {
                        let ind = eval_in_uni(l_skip, n, z);
                        let eq_uni_r0 = eval_eq_uni(l, z.into(), r_uni);
                        let eq_uni_r0_rot = eval_eq_uni(l, z.into(), r_uni * omega_l);

                        // default_packets[i].k_rot_1 = self.eq_const * eq_uni_1 doesn't change with
                        // n<0
                        Round0UniPacket {
                            eq_uni: eq_uni_r0 * ind,
                            k_rot_0: eq_uni_r0_rot * ind,
                            k_rot_1: default_packets[i].k_rot_1 * ind,
                        }
                    })
                    .collect_vec();
                d_special_packets = z_packets.to_device().unwrap();
                &d_special_packets
            } else {
                &d_default_packets
            };

            let num_x = 1 << n_lift;
            // PERF[jpw]: we choose the largest stride possible for more parallelism. Adjust if peak
            // memory too high.
            let thread_window_stride = window_len as u16;
            unsafe {
                let block_sums_len = _stacked_reduction_r0_required_temp_buffer_size(
                    domain_size as u32,
                    num_x as u32,
                    thread_window_stride,
                );
                // PERF[jpw]: buffer could be reused for q PLE folding below
                // Currently `CUBE_THREADS = 8` so this is at least D_EF/8 = 1/2 as much memory as
                // q_upsampled itself
                let mut block_sums = DeviceBuffer::with_capacity(block_sums_len as usize);
                let unstacked_cols_ptr = self.d_unstacked_cols.as_ptr().add(window[0]);
                // 2 per column for (eq, k_rot)
                let lambda_pows_ptr = self.d_lambda_pows.as_ptr().add(2 * window[0]);

                stacked_reduction_sumcheck_round0(
                    &d_q_upsampled_ptrs,
                    &self.eq_r_ns,
                    unstacked_cols_ptr,
                    lambda_pows_ptr,
                    z_packets,
                    &mut block_sums,
                    &mut d_s0_evals,
                    upsampled_height,
                    log_domain_size,
                    l_skip,
                    window_len,
                    num_x,
                    thread_window_stride,
                )
                .unwrap();
            };

            let evals = d_s0_evals.to_host().unwrap();
            debug_assert!(
                UnivariatePoly::from_evals_idft(&evals).coeffs()[s_0_deg + 1..]
                    .iter()
                    .all(|c| *c == EF::ZERO),
                "s_0 degree exceeds {s_0_deg}"
            );

            s_0_evals_batch.push(evals);
        }
        let s_0_evals = (0..domain_size)
            .map(|i| s_0_evals_batch.iter().map(|evals| evals[i]).sum::<EF>())
            .collect_vec();
        let mut s_0 = UnivariatePoly::from_evals_idft(&s_0_evals);
        debug_assert!(s_0.coeffs()[s_0_deg + 1..].iter().all(|c| *c == EF::ZERO));
        s_0.coeffs_mut().truncate(s_0_deg + 1);
        self.mem.tracing_info("stacked_reduction_sumcheck round 0");

        s_0
    }

    #[instrument("stacked_reduction_fold_ple", level = "debug", skip_all)]
    fn fold_ple_evals(&mut self, u_0: EF) {
        let l_skip = self.l_skip;
        let n_stack = self.n_stack;
        let r_0 = self.r_0;
        let omega_skip = self.omega_skip;
        let n_max = self.n_max;
        // fold the Q stacked matrices from mixed coefficient form
        self.q_evals = self
            .stacked_per_commit
            .iter()
            .map(|d| {
                let coeffs = &d.matrix().mixed;
                let num_x = 1 << n_stack;
                let width = d.matrix().width();
                debug_assert_eq!(coeffs.len(), width << (l_skip + n_stack));
                let folded_evals = DeviceBuffer::with_capacity(coeffs.len() >> l_skip);
                unsafe {
                    fold_ple_from_coeffs(
                        coeffs.as_ptr(),
                        folded_evals.as_mut_ptr(),
                        num_x,
                        width as u32,
                        1 << l_skip,
                        u_0,
                    )
                    .unwrap();
                }
                folded_evals
            })
            .collect();
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
    fn batch_sumcheck_poly_eval(&mut self, round: usize, _u_prev: EF) -> [EF; 2] {
        let l_skip = self.l_skip;
        let s_deg = 2;
        let mut d_s_evals = DeviceBuffer::<EF>::with_capacity(s_deg);
        let q_eval_ptrs = self.q_evals.iter().map(|q| q.as_ptr()).collect_vec();
        let d_q_eval_ptrs = q_eval_ptrs.to_device().unwrap();

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
            if log_height < l_skip + round {
                // We are in the eq, k_rot stable regime
                // This includes all n < 0 cases
                // In this case, the `s` poly contribution is a constant and we don't need to
                // interpolate
                let eq_r = self.eq_stable[log_height];
                let k_rot_r = self.k_rot_stable[log_height];
                // PERF[jpw]: most of eq_ub can be incorporated into eq_stable, k_rot_stable, so
                // this transfer can be minimized.
                let d_eq_ub = self.eq_ub_per_trace[window[0]..window[1]]
                    .to_device()
                    .unwrap();
                unsafe {
                    stacked_reduction_sumcheck_mle_round_degenerate(
                        &d_q_eval_ptrs,
                        &d_eq_ub,
                        eq_r,
                        k_rot_r,
                        unstacked_cols_ptr,
                        lambda_pows_ptr,
                        &mut d_s_evals,
                        self.stacked_height(round),
                        window_len,
                        l_skip,
                        round,
                    )
                    .unwrap();
                }
                let evals = d_s_evals.to_host().unwrap();
                s_evals_batch.push(evals);
            } else {
                let hypercube_dim = log_height - l_skip - round;
                let num_y = 1 << hypercube_dim;
                // PERF[jpw]: we choose the largest stride possible for more parallelism. Adjust if
                // peak memory too high.
                let thread_window_stride = window_len as u16;

                unsafe {
                    let block_sums_len = _stacked_reduction_mle_required_temp_buffer_size(
                        num_y as u32,
                        thread_window_stride,
                    );
                    let mut block_sums = DeviceBuffer::with_capacity(block_sums_len as usize);

                    stacked_reduction_sumcheck_mle_round(
                        &d_q_eval_ptrs,
                        &self.eq_r_ns,
                        &self.k_rot_ns,
                        unstacked_cols_ptr,
                        lambda_pows_ptr,
                        &mut block_sums,
                        &mut d_s_evals,
                        self.stacked_height(round),
                        window_len,
                        num_y,
                        thread_window_stride,
                    )
                    .unwrap();
                };
                let evals = d_s_evals.to_host().unwrap();
                s_evals_batch.push(evals);
            }
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
                let output_ptr = folded.as_mut_ptr() as usize;
                (folded, q.as_ptr() as usize, output_ptr)
            })
            .multiunzip();
        let d_input_ptrs = input_ptrs.to_device().unwrap();
        let d_output_ptrs = output_ptrs.to_device().unwrap();

        // SAFETY:
        // - `d_input_ptrs` points to matrices with widths specified by `d_q_widths` and heights
        //   `stacked_height(round) = stacked_height(round + 1) * 2`.
        // - `d_output_ptrs` points to matrices just allocated with widths specified by `d_q_widths`
        //   and heights `stacked_height(round + 1)`.
        unsafe {
            fold_mle(
                &d_input_ptrs,
                &d_output_ptrs,
                &self.d_q_widths,
                self.q_evals.len() as u32,
                self.stacked_height(round + 1) as u32,
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
