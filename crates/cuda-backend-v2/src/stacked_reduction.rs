use std::{array::from_fn, ffi::c_void, iter::zip};

use itertools::Itertools;
use openvm_cuda_backend::{cuda::kernels::lde::batch_expand_pad, ntt::batch_ntt};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_util::log2_ceil_usize;
use stark_backend_v2::{
    poly_common::{
        Squarable, UnivariatePoly, eval_eq_mle, eval_eq_uni, eval_eq_uni_at_one, eval_in_uni,
    },
    prover::{stacked_reduction::StackedReductionProver, sumcheck::sumcheck_round0_deg},
};
use tracing::instrument;

use crate::{
    Digest, EF, F, GpuBackendV2, GpuDeviceV2, ProverError,
    cuda::{
        poly::vector_scalar_multiply_ext,
        stacked_reduction::{
            _stacked_reduction_mle_required_temp_buffer_size,
            _stacked_reduction_r0_required_temp_buffer_size, initialize_k_rot_from_eq_segments,
            stacked_reduction_sumcheck_mle_round, stacked_reduction_sumcheck_mle_round_degenerate,
            stacked_reduction_sumcheck_round0,
        },
        sumcheck::{fold_mle, fold_ple_from_coeffs, triangular_fold_mle},
    },
    poly::EqEvalSegments,
    stacked_pcs::StackedPcsDataGpu,
};

pub struct StackedReductionGpu<'a> {
    l_skip: usize,
    omega_skip: F,
    n_stack: usize,

    r_0: EF,
    d_lambda_pows: DeviceBuffer<EF>,
    eq_const: EF,

    stacked_per_commit: Vec<&'a StackedPcsDataGpu<F, Digest>>,
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

impl<'a> StackedReductionGpu<'a> {
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

impl<'a> StackedReductionProver<'a, GpuBackendV2, GpuDeviceV2> for StackedReductionGpu<'a> {
    #[instrument("StackedReductionGpu::new", skip_all)]
    fn new(
        device: &'a GpuDeviceV2,
        stacked_per_commit: Vec<&'a StackedPcsDataGpu<F, Digest>>,
        r: &[EF],
        lambda: EF,
    ) -> Self {
        let l_skip = device.config.l_skip;
        let omega_skip = F::two_adic_generator(l_skip);
        let n_stack = device.config.n_stack;
        debug_assert!(
            stacked_per_commit
                .iter()
                .all(|d| d.matrix.height() == 1 << (l_skip + n_stack))
        );
        let widths = stacked_per_commit
            .iter()
            .map(|d| d.matrix.width() as u32)
            .collect_vec();
        let d_q_widths = widths.to_device().unwrap();

        let total_num_col_openings = stacked_per_commit
            .iter()
            .map(|d| d.layout.sorted_cols.len() * 2) // 2 for [plain, rotated]
            .sum();
        let lambda_pows = lambda.powers().take(total_num_col_openings).collect_vec();
        let d_lambda_pows = lambda_pows.to_device().unwrap();

        let unstacked_cols = stacked_per_commit
            .iter()
            .enumerate()
            .flat_map(|(com_idx, d)| {
                d.layout
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
                .map(|d| d.layout.sorted_cols[0].2.log_height())
                .max()
                .unwrap_or(0)
                .saturating_sub(l_skip)
        );
        let eq_r_ns = EqEvalSegments::new(&r[1..]).unwrap(); // TODO: return error

        let eq_const = eval_eq_uni_at_one(l_skip, r[0] * omega_skip);
        let eq_ub_per_trace = vec![EF::ONE; unstacked_cols.len()];

        Self {
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
        }
    }

    #[instrument("stacked_reduction_sumcheck", skip_all, fields(round = 0))]
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
                let q_mixed = &pcs_data.matrix.mixed;
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
                    batch_expand_pad(
                        &upsampled,
                        &q_mixed,
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
        let upsampled_height = self.stacked_per_commit[0].matrix.height() << log_expansion;
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
                    &z_packets,
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

        s_0
    }

    #[instrument("stacked_reduction_fold_ple", skip_all)]
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
                let coeffs = &d.matrix.mixed;
                let num_x = 1 << n_stack;
                let width = d.matrix.width();
                debug_assert_eq!(coeffs.len(), width << (l_skip + n_stack));
                let folded_evals = DeviceBuffer::with_capacity(coeffs.len() >> l_skip);
                unsafe {
                    fold_ple_from_coeffs(
                        &coeffs,
                        &folded_evals,
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

    #[instrument("stacked_reduction_into_stacked_openings", skip_all)]
    fn into_stacked_openings(self) -> Vec<Vec<EF>> {
        self.q_evals
            .into_iter()
            .map(|q| q.to_host().unwrap())
            .collect()
    }
}
