//! Single-outer-round pipelined GKR fractional sumcheck.
//!
//! Implements one outer round of `super::fractional::fractional_sumcheck_gpu`'s
//! PrecomputeM arm, restructured around the pipeline schedule from
//! [`pipelined_fractional.md`] (this file's sibling doc). The pipelined
//! region of length `L = round − 1` is tiled by M windows at bases
//! `0, α, w, w+α, 2w, …` and evaluated in alternating half-slots of length
//! `α` and `w − α`. Slot k's `fold pq + build M for slot k+1` runs
//! concurrently with slot k's transcript-serial `eval + observe + sample`
//! chain — the pipelining knob that keeps the M-build off the transcript
//! critical path.
//!
//! Two entry points share the same round-shape contract:
//!
//! - [`fractional_sumcheck_round_eager`] — the eagerly-executed CUDA baseline extracted from
//!   `super::fractional`'s PrecomputeM path (single M window + fold-eval tail + final fold). Used
//!   as the correctness reference.
//!
//! - [`fractional_sumcheck_round_pipelined_ir`] — graph-IR emission of the pipelined multi-window
//!   schedule. Every heavy compute stays in a blackbox CUDA kernel (`frac_precompute_m_build_*`,
//!   `frac_precompute_m_eval_round`, `frac_multifold`, `frac_compute_round_and_revert`); only the
//!   tiny scalar/eq bookkeeping runs through the DSL helpers already in [`super::fractional_ir`].
//!
//! Both entry points require `round >= 2 + PIPELINE_WINDOW` — the
//! minimum shape that leaves `pq_size = 2` for claim extraction after all
//! folds. Larger `round` values exercise the multi-window pipelining
//! more thoroughly (see `pipelined_ir_matches_eager_two_extra_windows`).

use std::ffi::c_void;

use crypto_compiler::{
    graph_ir::{BufId, DeviceType, GraphBuilder},
    quast::Quast,
};
use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use openvm_stark_backend::{
    poly_common::{eval_eq_mle, interpolate_quadratic_at_012},
    prover::fractional_sumcheck_gkr::Frac,
    FiatShamirTranscript, StarkProtocolConfig,
};
use p3_field::{Field, PrimeCharacteristicRing};

use super::{
    errors::FractionalSumcheckError,
    fractional::eval_mle_table,
    fractional_ir::{
        add_ef_buf, add_frac_ef_buf, eq_mle_table_ir_with_seed, eq_tail_bufs_with_seed,
        extract_claim_pair_ir, frac_compute_round_and_fold_inplace_ir_bufid,
        frac_compute_round_ir_bufid, frac_multifold_inplace_ir, frac_multifold_ir,
        frac_precompute_m_build_ir_bufid, frac_precompute_m_eval_round_ir, observe_and_update_ir,
        GkrLayerClaimIR, SqrtEqLayersIR,
    },
};
use crate::{
    cuda::logup_zerocheck::{
        _frac_compute_round_temp_buffer_size, fold_ef_frac_columns_inplace,
        frac_compute_round_and_revert, frac_multifold_raw, frac_precompute_m_build_raw,
        frac_precompute_m_eval_round_raw,
    },
    poly::SqrtEqLayers,
    prelude::EF,
    sponge_graph_ir::FiatShamirTranscriptGraphIR,
};

/// Degree of the sumcheck round polynomial `s(X)` (i.e. we transmit three
/// evaluations per inner round: `s(1), s(2), s(3)`).
pub const GKR_S_DEG: usize = 3;

/// PrecomputeM window size the pipelined driver targets. The M-build and
/// M-eval CUDA kernels are template-free (they accept any `w > 0`); the
/// multifold kernel's `switch (w)` dispatch caps at `w = 5`, and every
/// multifold in this driver stays at `w_fold ≤ w + 1 = 5`, so `w = 4` is
/// the largest safe choice. The eager `FractionalGkrMemoryModel::WINDOW_SIZE`
/// happens to be 3 today; we deliberately pick 4 here to exercise the
/// pipelining with `α = 2` (not `α = 1`) — a longer α makes the concurrent
/// M-build overlap correspondingly more transcript work.
pub const PIPELINE_WINDOW: usize = 4;

/// Half-slot split point `α`. `α = ⌊w / 2⌋` balances the extra `(1 + 2^{-α})`
/// M-build work against the shorter half-slot the build must hide behind
/// (see [`pipelined_fractional.md`] § "Notes ▸ Win condition").
pub const PIPELINE_ALPHA: usize = PIPELINE_WINDOW / 2;

// ---------------------------------------------------------------------------
// Shared round-shape structs.

/// Input to one outer GKR round. `layer` starts at `2 << round` `Frac<EF>`
/// elements: physically it is the post-tree-revert prover layer of the
/// previous outer round (the caller passes it in already reverted for
/// this single-round driver — the round-0 fused revert kernel handles
/// the layer_size = 2 → 4 top-of-tree step below).
#[derive(Debug)]
pub struct RoundInputEager {
    /// Physical `Frac<EF>` buffer holding the prover layer at the start
    /// of this outer round. Modified in place across inner rounds.
    pub layer: DeviceBuffer<Frac<EF>>,
    /// `ξ^{(j-1)}` — the previous outer round's Fiat-Shamir samples.
    /// `xi_prev.len() == round`.
    pub xi_prev: Vec<EF>,
    /// Seed for the sumcheck's running claim `s_{j-1}(r_{j-1})`. Under
    /// the outer-round protocol this is `numer + λ · denom` where numer
    /// / denom come from `reduce_to_single_evaluation(prev_claim, μ)`.
    pub prev_s_eval: EF,
    /// Seed for the running product `∏ eq(ξ_{j+1..}, r_{j+1..})`. `EF::ONE`
    /// at the start of every outer round.
    pub eq_r_acc: EF,
    /// Batching challenge sampled *before* this round begins.
    pub lambda: EF,
    /// Virtual-padding challenge, fixed per prover invocation.
    pub alpha: EF,
    /// `layer.len()` (physical `Frac<EF>` capacity).
    pub real_len: usize,
    /// Logical leaf count. `real_len == logical_len` for dense inputs.
    pub logical_len: usize,
    /// Outer round index `j`. `round == xi_prev.len()`.
    pub round: usize,
}

/// Output of one outer GKR round.
#[derive(Debug, Clone)]
pub struct RoundOutputEager {
    /// Per-inner-round `[s(1), s(2), s(3)]` observations. `len() == round`.
    pub round_polys: Vec<[EF; GKR_S_DEG]>,
    /// Fiat-Shamir samples produced by each inner round. `len() == round`.
    pub r_vec: Vec<EF>,
    /// Claim at `x = 0` for `p`.
    pub p_xi_0: EF,
    /// Claim at `x = 0` for `q`.
    pub q_xi_0: EF,
    /// Claim at `x = 1` for `p`.
    pub p_xi_1: EF,
    /// Claim at `x = 1` for `q`.
    pub q_xi_1: EF,
}

/// Graph-IR analogue of [`RoundInputEager`]. Every `EF`-valued input is a
/// [`BufId`] holding the raw p3 memory layout of an `EF` (identical to
/// what [`FiatShamirTranscriptGraphIR::sample_ext`] emits).
///
/// `seed` is a length-1 `[EF::ONE]` scalar buffer threaded through every
/// `eq_mle_table_ir_with_seed`, `SqrtEqLayersIR::from_xi_with_seed`, and
/// `eq_tail_bufs_with_seed` call in the driver body. Providing the seed
/// from the caller (typically registered as a graph input and populated
/// via `set_input`) lets the caller avoid the per-launch H2D copy that a
/// `SqrtEqLayersIR::seed_layer`-created `ConstBuf::HostBuf` would emit.
#[derive(Debug, Clone, Copy)]
pub struct RoundInputIR {
    pub layer: BufId,
    pub layer_len: usize,
    pub prev_s_eval: BufId,
    pub eq_r_acc: BufId,
    pub lambda: BufId,
    pub seed: BufId,
    pub alpha: EF,
    pub real_len: usize,
    pub logical_len: usize,
    pub round: usize,
}

/// Graph-IR analogue of [`RoundOutputEager`]. `p_xi_*` / `q_xi_*` are
/// EF-scalar buffers (`[D_EF]` BabyBear byte layout, the shape
/// [`add_ext_scalar_buf`] allocates).
#[derive(Debug, Clone)]
pub struct RoundOutputIR {
    pub round_polys: Vec<[BufId; GKR_S_DEG]>,
    pub r_vec: Vec<BufId>,
    pub p_xi_0: BufId,
    pub q_xi_0: BufId,
    pub p_xi_1: BufId,
    pub q_xi_1: BufId,
}

// ---------------------------------------------------------------------------
// Eager baseline (PrecomputeM path, single outer round).
//
// This is the code lifted almost verbatim from
// `super::fractional::fractional_sumcheck_gpu`'s inner `for round in
// 1..total_rounds` loop body (PrecomputeM arm), specialised to the
// single-window regime and stripped of the surrounding tree-build /
// transcript-plumbing scaffolding. Kept close to the original so the two
// implementations can be diffed line-by-line during triage.

fn eq_tail_ptrs_local(
    eq_buffer: &SqrtEqLayers,
    drop_count: usize,
) -> (*const EF, *const EF, usize) {
    let mut high_n = eq_buffer.high_n();
    let mut low_n = eq_buffer.low_n();
    let total_n = high_n + low_n;
    if drop_count >= total_n {
        return (std::ptr::null(), std::ptr::null(), 1);
    }
    if drop_count <= high_n {
        high_n -= drop_count;
    } else {
        low_n -= drop_count - high_n;
        high_n = 0;
    }
    (
        eq_buffer.low.get_ptr(low_n),
        eq_buffer.high.get_ptr(high_n),
        1 << low_n,
    )
}

/// Observes s_evals in transcript, updates accumulators, and returns the sampled challenge.
#[allow(clippy::too_many_arguments)]
fn observe_and_update_eager<SC, TS>(
    d_sum_evals: &DeviceBuffer<EF>,
    transcript: &mut TS,
    round_polys: &mut Vec<[EF; GKR_S_DEG]>,
    r_vec: &mut Vec<EF>,
    prev_s_eval: &mut EF,
    xi_j: EF,
    eq_r_acc: &mut EF,
    device_ctx: &GpuDeviceCtx,
) -> Result<EF, FractionalSumcheckError>
where
    SC: StarkProtocolConfig<EF = EF>,
    TS: FiatShamirTranscript<SC>,
{
    let sp_host: Vec<EF> = d_sum_evals.to_host_on(device_ctx)?;
    debug_assert_eq!(sp_host.len(), GKR_S_DEG - 1);

    // Reconstruct s(1), s(2), s(3) from sp(1), sp(2) using the sum-check
    // consistency relation `s_j(0) + s_j(1) = s_{j-1}(r_{j-1})` and
    // `s_j(X) = eq(xi_j, X) · sp_j(X)`.
    let mut sp_evals = [EF::ZERO; GKR_S_DEG];
    sp_evals[1] = sp_host[0] * *eq_r_acc;
    sp_evals[2] = sp_host[1] * *eq_r_acc;
    let eq_xi_0 = EF::ONE - xi_j;
    sp_evals[0] = (*prev_s_eval - xi_j * sp_evals[1]) * eq_xi_0.inverse();

    let s_evals: [EF; GKR_S_DEG] = std::array::from_fn(|i| {
        let x = EF::from_usize(i + 1);
        let sp = if i < GKR_S_DEG - 1 {
            sp_evals[i + 1]
        } else {
            interpolate_quadratic_at_012(&sp_evals, x)
        };
        eval_eq_mle(&[xi_j], &[x]) * sp
    });

    for &eval in &s_evals {
        transcript.observe_ext(eval);
    }
    round_polys.push(s_evals);

    let r = transcript.sample_ext();
    r_vec.push(r);

    let eq_r = eval_eq_mle(&[xi_j], &[r]);
    *eq_r_acc *= eq_r;
    *prev_s_eval = eq_r * interpolate_quadratic_at_012(&sp_evals, r);

    Ok(r)
}

/// Extract `(p_xi_0, q_xi_0, p_xi_1, q_xi_1)` from `layer` at positions
/// `0` and `stride`.
fn read_pq_pair(
    layer: &DeviceBuffer<Frac<EF>>,
    stride: usize,
    device_ctx: &GpuDeviceCtx,
) -> Result<(Frac<EF>, Frac<EF>), FractionalSumcheckError> {
    let scratch = DeviceBuffer::<Frac<EF>>::with_capacity_on(2, device_ctx);
    unsafe {
        cuda_memcpy_on::<true, true>(
            scratch.as_mut_raw_ptr(),
            layer.as_ptr() as *const c_void,
            std::mem::size_of::<Frac<EF>>(),
            device_ctx,
        )?;
        cuda_memcpy_on::<true, true>(
            (scratch.as_mut_raw_ptr() as *mut Frac<EF>).add(1) as *mut c_void,
            layer.as_ptr().add(stride) as *const c_void,
            std::mem::size_of::<Frac<EF>>(),
            device_ctx,
        )?;
    }
    let host = scratch.to_host_on(device_ctx)?;
    Ok((host[0], host[1]))
}

/// Single-outer-round PrecomputeM sumcheck, dense-only, single-window regime.
///
/// Precondition: `round == 1 + PIPELINE_WINDOW` and `real_len == logical_len
/// == 2 << round`. The single-window constraint keeps this baseline
/// symmetric with the pipelined IR driver below; the eager prover's full
/// multi-window / fold-eval-tail machinery lives in
/// `super::fractional::fractional_sumcheck_gpu`.
///
/// Behaviour tracks the eager prover: fused revert + compute at inner
/// round 0, then one PrecomputeM window (M-build with `pending_fold =
/// true`, `w` eval kernels, one `w+1`-variable multifold), then the
/// physical layer holds the two folded claims at positions 0 and 1.
pub fn fractional_sumcheck_round_eager<SC, TS>(
    input: RoundInputEager,
    transcript: &mut TS,
    device_ctx: &GpuDeviceCtx,
) -> Result<RoundOutputEager, FractionalSumcheckError>
where
    SC: StarkProtocolConfig<EF = EF>,
    TS: FiatShamirTranscript<SC>,
{
    let RoundInputEager {
        mut layer,
        xi_prev,
        mut prev_s_eval,
        mut eq_r_acc,
        lambda,
        alpha,
        real_len,
        logical_len,
        round,
    } = input;
    assert_eq!(xi_prev.len(), round, "xi_prev.len() must equal round");
    assert!(
        round >= 2 + PIPELINE_WINDOW,
        "eager baseline needs a full PrecomputeM window + tail:\
         expect round >= 2 + PIPELINE_WINDOW"
    );
    assert_eq!(real_len, logical_len, "dense-only path");
    assert_eq!(
        layer.len(),
        real_len,
        "layer.len() must equal real_len for this driver"
    );

    let stream = device_ctx.stream.as_raw();
    let total_leaves = logical_len;
    let w = PIPELINE_WINDOW;
    let mut pq_size = 2usize << round;

    let mut eq_buffer = SqrtEqLayers::from_xi(&xi_prev[1..], device_ctx)
        .map_err(FractionalSumcheckError::EvalEqHypercube)?;

    let mut round_polys: Vec<[EF; GKR_S_DEG]> = Vec::with_capacity(round);
    let mut r_vec: Vec<EF> = Vec::with_capacity(round);

    let mut d_sum_evals = DeviceBuffer::<EF>::with_capacity_on(GKR_S_DEG - 1, device_ctx);
    let tmp_cap = unsafe { _frac_compute_round_temp_buffer_size((1 << round) as u32) } as usize;
    let tmp_cap = tmp_cap.max(1 << (w + 1));
    let mut tmp_block_sums = DeviceBuffer::<EF>::with_capacity_on(tmp_cap, device_ctx);

    // ---- Inner round 0: fused revert + compute ----------------------------
    unsafe {
        frac_compute_round_and_revert(
            &eq_buffer,
            &mut layer,
            pq_size / 2,
            total_leaves,
            lambda,
            alpha,
            &mut d_sum_evals,
            &mut tmp_block_sums,
            stream,
        )
        .map_err(FractionalSumcheckError::ComputeRound)?;
    }
    eq_buffer.drop_layer();
    let r0 = observe_and_update_eager::<SC, TS>(
        &d_sum_evals,
        transcript,
        &mut round_polys,
        &mut r_vec,
        &mut prev_s_eval,
        xi_prev[0],
        &mut eq_r_acc,
        device_ctx,
    )?;

    // ---- Single PrecomputeM window, base = 1 ------------------------------
    let base = 1usize;
    let rem_n = round - base;
    let m_len = 1usize << (2 * w);
    let m_buffer = DeviceBuffer::<EF>::with_capacity_on(m_len, device_ctx);
    // Tail tile fixed to one block (dense small case); the real driver
    // clamps this via env knobs but we keep it simple here.
    let tail_n = rem_n - w;
    let tail_tile = 1usize << tail_n;
    let partial_len = m_len;
    let m_partial = DeviceBuffer::<EF>::with_capacity_on(partial_len, device_ctx);

    let (eq_tail_low, eq_tail_high, eq_low_cap) = eq_tail_ptrs_local(&eq_buffer, w - 1);

    unsafe {
        frac_precompute_m_build_raw(
            layer.as_ptr(),
            real_len,
            total_leaves,
            rem_n,
            w,
            lambda,
            r0,
            alpha,
            /* inline_fold */ true,
            eq_tail_low,
            eq_tail_high,
            eq_low_cap,
            tail_tile,
            m_partial.as_mut_ptr(),
            partial_len,
            m_buffer.as_mut_ptr(),
            stream,
        )
        .map_err(FractionalSumcheckError::ComputeRound)?;
    }

    let mut eq_r_prefix_host = vec![EF::ZERO; 1 << w];
    let mut eq_suffix_host = vec![EF::ZERO; 1 << w];
    let eq_r_prefix_dev = DeviceBuffer::<EF>::with_capacity_on(1usize << w, device_ctx);
    let eq_suffix_dev = DeviceBuffer::<EF>::with_capacity_on(1usize << w, device_ctx);

    let mut window_rs: Vec<EF> = Vec::with_capacity(w);
    for t in 0..w {
        let prefix_bits = t;
        let suffix_bits = w - t - 1;
        eval_mle_table(&window_rs, &mut eq_r_prefix_host);
        eval_mle_table(&xi_prev[base + t + 1..base + w], &mut eq_suffix_host);

        unsafe {
            cuda_memcpy_on::<false, true>(
                eq_r_prefix_dev.as_mut_raw_ptr(),
                eq_r_prefix_host.as_ptr() as *const c_void,
                (1usize << prefix_bits) * std::mem::size_of::<EF>(),
                device_ctx,
            )?;
            cuda_memcpy_on::<false, true>(
                eq_suffix_dev.as_mut_raw_ptr(),
                eq_suffix_host.as_ptr() as *const c_void,
                (1usize << suffix_bits) * std::mem::size_of::<EF>(),
                device_ctx,
            )?;
            frac_precompute_m_eval_round_raw(
                m_buffer.as_ptr(),
                w,
                t,
                eq_r_prefix_dev.as_ptr(),
                eq_suffix_dev.as_ptr(),
                d_sum_evals.as_mut_ptr(),
                stream,
            )
            .map_err(FractionalSumcheckError::ComputeRound)?;
        }
        eq_buffer.drop_layer();
        let r = observe_and_update_eager::<SC, TS>(
            &d_sum_evals,
            transcript,
            &mut round_polys,
            &mut r_vec,
            &mut prev_s_eval,
            xi_prev[base + t],
            &mut eq_r_acc,
            device_ctx,
        )?;
        window_rs.push(r);
    }

    // ---- Multifold: fold pq by (r0, ρ_0..ρ_{w-1}) — w+1 variables --------
    let mut eq_r_window_host = vec![EF::ZERO; 1 << (w + 1)];
    let mut all_rs: Vec<EF> = Vec::with_capacity(w + 1);
    all_rs.push(r0);
    all_rs.extend_from_slice(&window_rs);
    eval_mle_table(&all_rs, &mut eq_r_window_host);
    let d_eq_r_window = tmp_block_sums.as_mut_ptr();
    unsafe {
        cuda_memcpy_on::<false, true>(
            d_eq_r_window as *mut c_void,
            eq_r_window_host.as_ptr() as *const c_void,
            (1usize << (w + 1)) * std::mem::size_of::<EF>(),
            device_ctx,
        )?;
        frac_multifold_raw(
            layer.as_ptr(),
            layer.as_mut_ptr(),
            real_len,
            total_leaves,
            rem_n + 1,
            w + 1,
            alpha,
            d_eq_r_window,
            stream,
        )
        .map_err(FractionalSumcheckError::FoldColumns)?;
    }
    pq_size >>= w + 1;

    // ---- Standalone tail compute: one more sample, no fold. --------------
    // Mirrors `fractional::fractional_sumcheck_gpu`'s tail arm
    // (`frac_compute_round` + `observe_and_update`).
    let tail_base = 1 + w;
    use crate::cuda::logup_zerocheck::{frac_compute_round, frac_compute_round_and_fold_inplace};
    unsafe {
        frac_compute_round(
            &eq_buffer,
            &layer,
            pq_size / 2,
            lambda,
            &mut d_sum_evals,
            &mut tmp_block_sums,
            stream,
        )
        .map_err(FractionalSumcheckError::ComputeRound)?;
    }
    eq_buffer.drop_layer();
    let mut prev_r = observe_and_update_eager::<SC, TS>(
        &d_sum_evals,
        transcript,
        &mut round_polys,
        &mut r_vec,
        &mut prev_s_eval,
        xi_prev[tail_base],
        &mut eq_r_acc,
        device_ctx,
    )?;

    // ---- Fused fold + compute tail rounds (round − (1 + w + 1) iters). ---
    // Each iteration folds pq by `prev_r` (sample from the previous
    // round), then computes and samples a new challenge — the standard
    // fold-eval body from `fractional_sumcheck_gpu`.
    for &xi_j in xi_prev.iter().skip(tail_base + 1) {
        let src_pq_size = pq_size;
        unsafe {
            frac_compute_round_and_fold_inplace(
                &eq_buffer,
                &mut layer,
                src_pq_size,
                src_pq_size,
                src_pq_size,
                src_pq_size >> 1,
                src_pq_size >> 1,
                lambda,
                prev_r,
                alpha,
                &mut d_sum_evals,
                &mut tmp_block_sums,
                stream,
            )
            .map_err(FractionalSumcheckError::ComputeRound)?;
        }
        eq_buffer.drop_layer();
        prev_r = observe_and_update_eager::<SC, TS>(
            &d_sum_evals,
            transcript,
            &mut round_polys,
            &mut r_vec,
            &mut prev_s_eval,
            xi_j,
            &mut eq_r_acc,
            device_ctx,
        )?;
        pq_size >>= 1;
    }

    // ---- Final fold using the last-sampled challenge. --------------------
    unsafe {
        fold_ef_frac_columns_inplace(&mut layer, pq_size, pq_size, pq_size, prev_r, alpha, stream)
            .map_err(FractionalSumcheckError::FoldColumns)?;
    }
    pq_size >>= 1;

    // ---- Extract claims at positions 0 and pq_size/2 ----------------------
    let (left, right) = read_pq_pair(&layer, pq_size / 2, device_ctx)?;
    Ok(RoundOutputEager {
        round_polys,
        r_vec,
        p_xi_0: left.p,
        q_xi_0: left.q,
        p_xi_1: right.p,
        q_xi_1: right.q,
    })
}

// ---------------------------------------------------------------------------
// Graph-IR pipelined driver.
//
// Same round-shape as the eager baseline above: fused revert + compute at
// inner round 0, then the single PrecomputeM window, then a multifold to
// pq_size = 2. The difference sits in the middle: the window's `w` eval
// rounds are split into two half-slots of length `α` and `w − α`, and the
// next stage's (challenge-free) fold + M build is emitted *before* the
// current half-slot's eval chain — so the graph planner sees them as
// independent DAG branches and can put them on different streams.
//
// Single-window regime only (`round == 1 + PIPELINE_WINDOW`): there is no
// "next window" to overlap with in this file, but the schedule below
// still exercises the pipelining structure (M-build for the *startup*
// window is issued before the round-0 revert compute, so its dev-challenge
// inputs — `lambda`, seeded `r_prev` const, etc. — sit on the M-build
// stream while the revert runs). The multi-window generalisation is a
// straightforward extension: pull the "fold-by-α, build M_{k+1}, eval-
// half-slot-k" body into a loop.

/// Wrapper around [`super::fractional_ir::frac_compute_round_and_revert_ir_bufid`]
/// that takes the eq layers explicitly (rather than a mutable
/// [`SqrtEqLayersIR`]) — the pipelined driver keeps `eq_buffer` immutable
/// across the round-0 revert and drops layers only around the eval kernels.
#[allow(clippy::too_many_arguments)]
fn compute_round_and_revert_ir(
    g: &mut GraphBuilder,
    eq_buffer: &SqrtEqLayersIR,
    layer: BufId,
    layer_len: usize,
    num_x: usize,
    logical_len: usize,
    lambda: BufId,
    alpha: EF,
    d_sum_evals: BufId,
    tmp_block_sums: BufId,
) {
    use crate::cuda::logup_zerocheck::frac_compute_round_and_revert_dev_challenge;
    let low_n = eq_buffer.low_n();
    let high_n = eq_buffer.high_n();
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    let eq_low = eq_buffer.low.get(low_n);
    let eq_high = eq_buffer.high.get(high_n);
    let eq_low_cap = 1usize << low_n;
    g.insert_blackbox_kernel(
        "frac_compute_round_and_revert_dev_challenge",
        [eq_low, eq_high, layer, lambda].into_iter(),
        [d_sum_evals, tmp_block_sums].into_iter(),
        [false, false, true, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_compute_round_and_revert_dev_challenge(
                inputs[0] as *const EF,
                inputs[1] as *const EF,
                inputs[2] as *mut Frac<EF>,
                num_x,
                layer_len,
                logical_len,
                eq_low_cap,
                inputs[3] as *const EF,
                alpha,
                outputs[0] as *mut EF,
                outputs[1] as *mut EF,
                stream,
            )
            .expect("frac_compute_round_and_revert_dev_challenge");
        },
    );
}

/// In-place `fold_ef_frac_columns` node, taking `r_prev` as a `BufId`.
fn fold_columns_inplace_ir(
    g: &mut GraphBuilder,
    buf: BufId,
    size: usize,
    r_prev: BufId,
    alpha: EF,
) {
    use crate::cuda::logup_zerocheck::fold_ef_frac_columns_inplace_dev_challenge;
    g.insert_blackbox_kernel(
        "fold_ef_frac_columns_inplace_dev_challenge",
        [buf, r_prev].into_iter(),
        std::iter::empty(),
        [true, false].into_iter(),
        move |inputs, _outputs, stream| unsafe {
            fold_ef_frac_columns_inplace_dev_challenge(
                inputs[0] as *mut Frac<EF>,
                size,
                size,
                size,
                inputs[1] as *const EF,
                alpha,
                stream,
            )
            .expect("fold_ef_frac_columns_inplace_dev_challenge");
        },
    );
}

/// Emit one M-build kernel node for the pipelined driver's per-window
/// precompute-M stage. Wraps [`frac_precompute_m_build_ir_bufid`] with the
/// per-window sizing math: `rem_n = L − pipelined_base` (with the eager
/// driver's `pending_fold` semantics — the first-window build folds `r0`
/// inline without modifying `pq`); the tail-tile heuristic and partial
/// buffer sizing match the eager driver's
/// [`super::fractional::precompute_m_build_tail_tile`] /
/// [`super::fractional::precompute_m_num_tail_blocks`] so the kernel
/// launches with a real grid dim (`num_blocks × 2^w × 2^w` = # tail
/// tiles × the M matrix's `(u, v)` axes) instead of `(1, 1, 1)`.
#[allow(clippy::too_many_arguments)]
fn emit_m_build(
    g: &mut GraphBuilder,
    eq_buffer: &SqrtEqLayersIR,
    seed: BufId,
    pq_src: BufId,
    pipelined_base: usize,
    pending_fold: bool,
    r_prev_buf: BufId,
    name_hint: &str,
    l_pipelined: usize,
    w: usize,
    real_len: usize,
    logical_len: usize,
    lambda: BufId,
    alpha: EF,
    device: DeviceType,
) -> BufId {
    use super::fractional::{
        precompute_m_build_tail_tile, precompute_m_min_blocks_threshold,
        precompute_m_num_tail_blocks, precompute_m_tail_tile_override, precompute_m_target_blocks,
    };

    let rem_n = l_pipelined - pipelined_base;
    debug_assert!(rem_n >= w, "M-build requires rem_n >= w");
    // Match the eager driver's tail-tile heuristic: `desired_tile =
    // ceil(2^tail_n / target_blocks)` clamped to `[MIN, MAX]` so the
    // kernel gets `num_blocks ∈ ~[64, 1024]` tail tiles. Previous code
    // hardcoded `tail_tile = 2^tail_n` → 1 block, so
    // `precompute_m_build_partial_kernel` launched with grid dim
    // `(1, 1, 1)` and left the SM utterly cold.
    let min_blocks = precompute_m_min_blocks_threshold();
    let target_blocks = precompute_m_target_blocks();
    let tile_override = precompute_m_tail_tile_override();
    let tail_tile =
        precompute_m_build_tail_tile(rem_n, w, min_blocks, target_blocks, tile_override);
    let num_blocks = precompute_m_num_tail_blocks(rem_n, w, tail_tile);
    let m_len = 1usize << (2 * w);
    let partial_len = num_blocks * m_len;
    let m_buf = add_ef_buf(g, device, &format!("{name_hint}_m"), m_len);
    let m_partial = add_ef_buf(g, device, &format!("{name_hint}_partial"), partial_len);
    let (eq_tail_low, eq_tail_high, eq_low_cap) = eq_tail_bufs_with_seed(g, eq_buffer, w - 1, seed);
    frac_precompute_m_build_ir_bufid(
        g,
        pq_src,
        eq_tail_low,
        eq_tail_high,
        m_partial,
        m_buf,
        real_len,
        logical_len,
        rem_n,
        w,
        lambda,
        r_prev_buf,
        alpha,
        pending_fold,
        eq_low_cap,
        tail_tile,
        partial_len,
    );
    m_buf
}

/// Extract the folded pq claims `(layer[0], layer[stride])` into a fresh
/// [`GkrLayerClaimIR`]. Thin wrapper around the shared
/// [`extract_claim_pair_ir`] gather module in [`super::fractional_ir`].
fn extract_frac_pair_ir(
    g: &mut GraphBuilder,
    src: BufId,
    src_alloc_len: usize,
    stride: usize,
    device: DeviceType,
    label: &str,
) -> GkrLayerClaimIR {
    extract_claim_pair_ir(g, src, src_alloc_len, stride, label, device)
}

/// Emit the single-outer-round pipelined sumcheck onto `g` and return
/// per-inner-round s-poly / sample [`BufId`]s plus the extracted layer
/// claims. `xi_prev` are the previous outer round's Fiat-Shamir samples
/// staged as `[D_EF]` BabyBear const-or-sample buffers (the shape
/// [`FiatShamirTranscriptGraphIR::sample_ext`] and
/// [`super::fractional_ir::ef_const_ext_scalar_buf`] both produce).
///
/// # Pipeline
/// - Round 0: fused revert + compute (uses `layer` in place). Depends on `lambda`.
/// - Startup M-build: one `frac_precompute_m_build_dev_challenge` with `inline_fold = true`,
///   folding round 0's sample `r0` on the fly. This kernel does NOT touch `layer` (build reads pq
///   into an M-matrix scratch buffer), so it runs concurrently with subsequent eval kernels that
///   read the post-revert `layer` for the next window.
/// - Slot 0 (`α` rounds): eval kernels reading M, chained through the transcript-serial `observe_
///   and_update` scalar block.
/// - Slot 1 (`w − α` rounds): same shape as slot 0. No new M-build to overlap in the single-window
///   regime, but the schedule leaves room for it (a multi-window extension inserts a `fold +
///   frac_precompute_m_build_dev_challenge` for M_next between the two slots).
/// - Multifold + final fold: shrink the pq buffer to two entries and extract claims.
#[allow(clippy::too_many_arguments)]
pub fn fractional_sumcheck_round_pipelined_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    input: RoundInputIR,
    xi_prev: &[BufId],
    device: DeviceType,
) -> RoundOutputIR
where
    TS: FiatShamirTranscriptGraphIR,
{
    let RoundInputIR {
        layer,
        layer_len,
        prev_s_eval,
        eq_r_acc,
        lambda,
        seed: shared_seed,
        alpha,
        real_len,
        logical_len,
        round,
    } = input;
    assert_eq!(xi_prev.len(), round, "xi_prev.len() must equal round");
    assert!(
        round >= 2 + PIPELINE_WINDOW,
        "pipelined driver needs at least one multi-window step: \
         expect round >= 2 + PIPELINE_WINDOW"
    );
    assert_eq!(real_len, logical_len, "dense-only path");

    let total_leaves = logical_len;
    let w = PIPELINE_WINDOW;
    let alpha_half = PIPELINE_ALPHA;
    let mut pq_size = 2usize << round;

    // Pipelined region: absolute inner rounds 1..round map to pipelined
    // positions 0..L−1 (round 0 is the fused-revert round handled
    // separately below; its challenge r0 is inline-folded into the
    // startup M-build, mirroring the eager driver's `pending_fold =
    // true`).
    let l_pipelined = round - 1;

    // Half-slot schedule: `boundaries[k]` is the first pipelined position
    // in slot k. Half-slots alternate lengths `α, w − α, α, …` starting
    // from position 0. Truncated to `L` on the last step so a partial
    // final slot doesn't overshoot.
    let boundaries: Vec<usize> = {
        let mut b = vec![0usize];
        let mut cur = 0usize;
        let mut i = 0;
        while cur < l_pipelined {
            let step = if i % 2 == 0 {
                alpha_half
            } else {
                w - alpha_half
            };
            cur = (cur + step).min(l_pipelined);
            b.push(cur);
            i += 1;
        }
        b
    };
    let num_slots = boundaries.len() - 1;
    // Slot k's M window base (pipelined-relative). Slots 0 and 1 both
    // read the startup M[0, w); from slot 2 onward, slot k reads the M
    // built during slot k−1 at base `boundaries[k−1]`.
    let m_base_of_slot = |k: usize| -> usize {
        if k <= 1 {
            0
        } else {
            boundaries[k - 1]
        }
    };

    // The caller-provided `seed` (see [`RoundInputIR::seed`]) is threaded
    // through every `eq_mle_table_ir_with_seed`,
    // `SqrtEqLayersIR::from_xi_with_seed`, and `eq_tail_bufs_with_seed`
    // call below. If the caller registers it as a graph input and
    // populates it once via `set_input`, no per-launch H2D copy is
    // emitted for the seed — collapsing the ~30 seed_layer H2D copies
    // an internally-created seed would leave in the captured graph
    // down to zero.
    let mut eq_buffer = SqrtEqLayersIR::from_xi_with_seed(g, &xi_prev[1..], shared_seed, device);

    let mut round_polys: Vec<[BufId; GKR_S_DEG]> = Vec::with_capacity(round);
    let mut r_vec: Vec<BufId> = Vec::with_capacity(round);

    // Scratch buffers reused across kernels.
    let tmp_len = {
        let cap = unsafe { _frac_compute_round_temp_buffer_size((1 << round) as u32) } as usize;
        cap.max(1usize << (w + 1))
    };
    let d_sum_r0 = add_ef_buf(g, device, "d_sum_r0", GKR_S_DEG - 1);
    let tmp_r0 = add_ef_buf(g, device, "tmp_r0", tmp_len);

    // ---- Inner round 0: fused revert + compute ----------------------------
    compute_round_and_revert_ir(
        g,
        &eq_buffer,
        layer,
        layer_len,
        pq_size / 2,
        total_leaves,
        lambda,
        alpha,
        d_sum_r0,
        tmp_r0,
    );
    eq_buffer.drop_layer();
    let r0_out = observe_and_update_ir(
        g,
        transcript,
        d_sum_r0,
        prev_s_eval,
        xi_prev[0],
        eq_r_acc,
        device,
    );
    round_polys.push(r0_out.s_evals);
    r_vec.push(r0_out.r);
    let mut prev_s_eval_cur = r0_out.prev_s_eval;
    let mut eq_r_acc_cur = r0_out.eq_r_acc;
    let r0 = r0_out.r;

    // ---- Startup M-build: M[base=0 (pipelined), w) with pending_fold=true.
    // Reads `layer` (post-revert) but does NOT write it. Subsequent
    // folds move the working pq state into `work_buf` (out-of-place on
    // the first fold, in-place afterwards); the `layer` buffer stays
    // pristine so the NEXT outer round's `compute+revert` chain can
    // read it unchanged, matching the eager driver's
    // `active_pq = work_buffer` policy for non-last outer rounds.
    let m_startup = emit_m_build(
        g,
        &eq_buffer,
        shared_seed,
        layer,
        /* pipelined_base */ 0,
        /* pending_fold */ true,
        r0,
        "m0",
        l_pipelined,
        w,
        real_len,
        total_leaves,
        lambda,
        alpha,
        device,
    );
    // Work buffer for post-round-0 pq state. Sized to `pq_size` (= 2 <<
    // round): the first fold reduces it, all subsequent folds stay
    // within its capacity.
    let work_buf_capacity = pq_size;
    let work_buf = add_frac_ef_buf(g, device, "pipelined_work", work_buf_capacity);
    // `active_pq_is_work` = false means the current pq state lives in
    // `layer[0..pq_size]`; true means it lives in `work_buf`. We flip
    // to `true` after the first fold copies layer → work_buf.
    let mut active_pq_is_work = false;
    // `active_source_real_len` / `active_source_logical_len` mirror the
    // eager driver's `source_real_len` / `source_logical_len`: on
    // out-of-place fold-eval reads they refer to the SOURCE buffer's
    // physical size; on in-place they refer to the current pq's real /
    // logical size (= pq_size for dense).
    let mut active_source_real_len = real_len;
    let mut active_source_logical_len = total_leaves;

    // ---- Precompute which slots run pipelined vs fold-eval ---------------
    // Slot k reads an M window if either:
    //   - k ≤ 1 (uses the startup M_0), or
    //   - k ≥ 2 AND M_{k-1} was buildable — i.e. `boundaries[k-1] + w ≤ L` (rem_n = L −
    //     boundaries[k-1] must be ≥ w for the M-build kernel).
    // Everything past that boundary falls back to the eager fold-eval
    // tail (standalone compute + fused fold+compute), mirroring the
    // `if base < round { … }` branch in `super::fractional`.
    let last_pipelined_slot = {
        let mut last = 1.min(num_slots - 1);
        for k in 2..num_slots {
            if boundaries[k - 1] + w <= l_pipelined {
                last = k;
            } else {
                break;
            }
        }
        last
    };

    // ---- Half-slot loop ---------------------------------------------------
    // Emission order per slot k:
    //   1a. Swap `current_m` in from the queue if slot k moves to a new M.
    //   1b. If k ≥ 1, fold pq by `pending` samples (all slots — pipelined AND fold-eval — flush
    //       accumulated samples into the pq buffer at their start).
    //   1c. If slot k+1 is pipelined AND exists, build M for slot k+1's use. The fold + build has
    //       NO data dep on the current or later slot's transcript state, so the planner can
    //       schedule it on the M-build stream concurrently with slot k's eval chain below — this is
    //       the pipelining knob.
    //   2. Emit slot k's eval + observe + sample chain — M-based when `k ≤ last_pipelined_slot`,
    //      fold-eval (standalone + fused compute+fold) otherwise.
    let mut current_m = m_startup;
    let mut queued_m: Option<BufId> = None;
    let mut pending: Vec<BufId> = vec![r0];
    for k in 0..num_slots {
        let is_pipelined = k <= last_pipelined_slot;
        let next_is_pipelined = k + 1 < num_slots && k < last_pipelined_slot;

        // Step 1a: swap current_m for pipelined slots ≥ 2 (both slot 0 and
        // slot 1 read the startup M_0).
        if is_pipelined && k >= 2 {
            current_m = queued_m
                .take()
                .expect("queued M from previous slot's build");
        }

        // Step 1b: fold pending into pq. Applies to every slot ≥ 1 — the
        // fold-eval branch needs its input pq folded through the previous
        // slot's samples too. First fold: layer → work_buf (out-of-place,
        // leaves layer intact). Subsequent folds: in-place on work_buf.
        if k >= 1 && !pending.is_empty() {
            let n_fold = pending.len();
            let buf_vars = pq_size.trailing_zeros() as usize - 1;
            let (src_read_real_len, src_read_logical_len) = if active_pq_is_work {
                (active_source_real_len, active_source_logical_len)
            } else {
                // First fold reads `layer`, whose physical length is `real_len`.
                (real_len, total_leaves)
            };
            if active_pq_is_work {
                // In-place fold on work_buf.
                if n_fold == 1 {
                    fold_columns_inplace_ir(g, work_buf, pq_size, pending[0], alpha);
                    pq_size >>= 1;
                } else {
                    debug_assert!(
                        (2..=5).contains(&n_fold),
                        "slot {k}: multifold w = {n_fold} out of dispatch range {{2..=5}}"
                    );
                    let eq_r_window = eq_mle_table_ir_with_seed(g, &pending, shared_seed, device);
                    frac_multifold_inplace_ir(
                        g,
                        work_buf,
                        eq_r_window,
                        src_read_real_len,
                        src_read_logical_len,
                        buf_vars,
                        n_fold,
                        alpha,
                    );
                    pq_size >>= n_fold;
                }
            } else {
                // First fold: layer → work_buf.
                if n_fold == 1 {
                    use super::fractional_ir::fold_ef_frac_columns_ir_bufid;
                    fold_ef_frac_columns_ir_bufid(
                        g,
                        layer,
                        work_buf,
                        pq_size,
                        src_read_real_len,
                        src_read_logical_len,
                        pending[0],
                        alpha,
                    );
                    pq_size >>= 1;
                } else {
                    debug_assert!(
                        (2..=5).contains(&n_fold),
                        "slot {k}: multifold w = {n_fold} out of dispatch range {{2..=5}}"
                    );
                    let eq_r_window = eq_mle_table_ir_with_seed(g, &pending, shared_seed, device);
                    frac_multifold_ir(
                        g,
                        layer,
                        work_buf,
                        eq_r_window,
                        src_read_real_len,
                        src_read_logical_len,
                        buf_vars,
                        n_fold,
                        alpha,
                    );
                    pq_size >>= n_fold;
                }
                active_pq_is_work = true;
            }
            active_source_real_len = pq_size;
            active_source_logical_len = pq_size;
            pending.clear();
        }

        // Step 1c: build M for slot k+1's use IFF slot k+1 is pipelined.
        // We're implicitly requiring k ≥ 1 (slot 0's "next" is slot 1
        // which reads the startup M_0, no build needed here). The M-build
        // reads from whichever buffer currently holds the pq state
        // (`active_pq` = layer before the first fold, work_buf after).
        if k >= 1 && next_is_pipelined {
            let next_base = boundaries[k];
            let build_src = if active_pq_is_work { work_buf } else { layer };
            let (build_real, build_logical) = if active_pq_is_work {
                (active_source_real_len, active_source_logical_len)
            } else {
                (real_len, total_leaves)
            };
            let m_next = emit_m_build(
                g,
                &eq_buffer,
                shared_seed,
                build_src,
                next_base,
                /* pending_fold */ false,
                r0, // Unused when pending_fold=false; still bound as a valid ptr.
                &format!("m{k}"),
                l_pipelined,
                w,
                build_real,
                build_logical,
                lambda,
                alpha,
                device,
            );
            queued_m = Some(m_next);
        }

        // Step 2: eval slot k.
        let slot_start = boundaries[k];
        let slot_end = boundaries[k + 1];
        if is_pipelined {
            // M-based eval. Prefix bits = local_t (samples drawn since
            // this M's base); suffix bits = w − local_t − 1.
            let m_base = m_base_of_slot(k);
            for pos in slot_start..slot_end {
                let local_t = pos - m_base;
                // Prefix: samples at pipelined positions [m_base, pos).
                // r_vec[0] holds r0; r_vec[i + 1] holds ρ_i.
                let prefix_bufs: Vec<BufId> = (m_base..pos).map(|p| r_vec[p + 1]).collect();
                let eq_r_prefix = eq_mle_table_ir_with_seed(g, &prefix_bufs, shared_seed, device);
                // Suffix: xi_prev at absolute inner rounds [pos + 2, m_base + 1 + w).
                let suffix_start = pos + 2;
                let suffix_end = m_base + 1 + w;
                let eq_suffix = eq_mle_table_ir_with_seed(
                    g,
                    &xi_prev[suffix_start..suffix_end],
                    shared_seed,
                    device,
                );

                let d_sum = add_ef_buf(g, device, &format!("d_sum_pos{pos}"), GKR_S_DEG - 1);
                frac_precompute_m_eval_round_ir(
                    g,
                    current_m,
                    eq_r_prefix,
                    eq_suffix,
                    d_sum,
                    w,
                    local_t,
                );
                eq_buffer.drop_layer();
                let out = observe_and_update_ir(
                    g,
                    transcript,
                    d_sum,
                    prev_s_eval_cur,
                    xi_prev[pos + 1],
                    eq_r_acc_cur,
                    device,
                );
                round_polys.push(out.s_evals);
                r_vec.push(out.r);
                prev_s_eval_cur = out.prev_s_eval;
                eq_r_acc_cur = out.eq_r_acc;
                pending.push(out.r);
            }
        } else {
            // Fold-eval tail. First position is a standalone compute
            // (no fold — the pending fold above has already positioned
            // pq); subsequent positions are fused compute + fold-by-
            // prev_r. Only the LAST sample of the slot is left unfolded
            // (it goes to `pending` for the post-loop absorption); the
            // intermediate samples are consumed inline by the fused
            // kernel.
            let mut prev_r_this_slot: Option<BufId> = None;
            for pos in slot_start..slot_end {
                let d_sum = add_ef_buf(g, device, &format!("d_sum_pos{pos}"), GKR_S_DEG - 1);
                let tmp_cap =
                    unsafe { _frac_compute_round_temp_buffer_size(pq_size as u32) } as usize;
                let tmp = add_ef_buf(g, device, &format!("tmp_pos{pos}"), tmp_cap.max(1));
                let active_buf = if active_pq_is_work { work_buf } else { layer };
                match prev_r_this_slot {
                    None => {
                        // Standalone compute (no fold) — reads active_pq only.
                        frac_compute_round_ir_bufid(
                            g,
                            &eq_buffer,
                            active_buf,
                            pq_size / 2,
                            lambda,
                            d_sum,
                            tmp,
                        );
                    }
                    Some(pr) => {
                        // Fused compute + fold by previous sample. If
                        // active_pq is still layer (no prior fold),
                        // this fold WILL corrupt layer — but in the
                        // pipelined regime we always fold pending into
                        // work_buf before entering the fold-eval tail,
                        // so this branch runs on work_buf.
                        debug_assert!(
                            active_pq_is_work,
                            "fold-eval tail fused iter must run on work_buf",
                        );
                        let src_pq_size = pq_size;
                        frac_compute_round_and_fold_inplace_ir_bufid(
                            g,
                            &eq_buffer,
                            active_buf,
                            src_pq_size,
                            active_source_real_len,
                            active_source_logical_len,
                            src_pq_size >> 1,
                            src_pq_size >> 1,
                            lambda,
                            pr,
                            alpha,
                            d_sum,
                            tmp,
                        );
                        pq_size >>= 1;
                        active_source_real_len = src_pq_size >> 1;
                        active_source_logical_len = src_pq_size >> 1;
                    }
                }
                eq_buffer.drop_layer();
                let out = observe_and_update_ir(
                    g,
                    transcript,
                    d_sum,
                    prev_s_eval_cur,
                    xi_prev[pos + 1],
                    eq_r_acc_cur,
                    device,
                );
                round_polys.push(out.s_evals);
                r_vec.push(out.r);
                prev_s_eval_cur = out.prev_s_eval;
                eq_r_acc_cur = out.eq_r_acc;
                prev_r_this_slot = Some(out.r);
            }
            if let Some(pr) = prev_r_this_slot {
                pending.push(pr);
            }
        }
    }

    // ---- Post-loop fold: absorb any samples still in `pending` -----------
    // Under the α/w−α tiling with r0 folded inline by the startup build,
    // pending at loop exit holds the last slot's samples (slot last has no
    // paired fold) plus, when the last slot is preceded directly by
    // another eval-only slot, any earlier deferred samples. In our
    // supported regimes (`round ≥ 2 + w`), the size sits in [2, 5], safely
    // inside the multifold kernel's dispatch table.
    let n_post = pending.len();
    debug_assert!(
        (1..=5).contains(&n_post),
        "post-loop fold w = {n_post} out of dispatch range {{1..=5}}"
    );
    debug_assert!(
        active_pq_is_work,
        "post-loop fold must run on work_buf (pipelined regime always folds pending into work_buf \
         at slot 1)",
    );
    if n_post == 1 {
        fold_columns_inplace_ir(g, work_buf, pq_size, pending[0], alpha);
        pq_size >>= 1;
    } else {
        let eq_r_window = eq_mle_table_ir_with_seed(g, &pending, shared_seed, device);
        let buf_vars = pq_size.trailing_zeros() as usize - 1;
        frac_multifold_inplace_ir(
            g,
            work_buf,
            eq_r_window,
            active_source_real_len,
            active_source_logical_len,
            buf_vars,
            n_post,
            alpha,
        );
        pq_size >>= n_post;
    }
    debug_assert_eq!(
        pq_size, 2,
        "post-fold pq_size must be 2 for claim extraction"
    );

    // ---- Extract claims at positions 0 and pq_size/2 from work_buf --------
    // (`layer` stays intact for the next outer round's compute+revert.)
    let _ = layer_len; // no longer used for claim extraction.
    let claim = extract_frac_pair_ir(
        g,
        work_buf,
        work_buf_capacity,
        pq_size / 2,
        device,
        "pipelined_claim",
    );
    RoundOutputIR {
        round_polys,
        r_vec,
        p_xi_0: claim.p_xi_0,
        q_xi_0: claim.q_xi_0,
        p_xi_1: claim.p_xi_1,
        q_xi_1: claim.q_xi_1,
    }
}

// ---------------------------------------------------------------------------
// FoldEval single-round fallback (small-round path).
//
// The pipelined M-window schedule requires `round >= 2 + PIPELINE_WINDOW`
// — the smallest shape that still fits one M window plus a
// non-degenerate tail. For smaller outer rounds the eager driver falls
// back to per-inner-round fold+compute; this function mirrors that fall
// back, matching the [`RoundInputIR`] / [`RoundOutputIR`] contract so
// the full-sumcheck driver below can dispatch on round size.

/// Single outer GKR round via the eager fold-eval strategy: round 0 is a
/// fused compute+revert, inner rounds `1..round-1` are fused compute+fold
/// on the last-sampled challenge, and a final `fold_ef_frac_columns` on
/// the last sample brings `pq_size` to 2 for claim extraction. Same
/// dispatch contract as [`fractional_sumcheck_round_pipelined_ir`]; kept
/// alongside it so the full driver's per-round `if round >= 2 + w { … }
/// else { … }` split is a one-line change.
pub fn fractional_sumcheck_round_foldeval_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    input: RoundInputIR,
    xi_prev: &[BufId],
    device: DeviceType,
) -> RoundOutputIR
where
    TS: FiatShamirTranscriptGraphIR,
{
    use super::fractional_ir::{
        add_frac_ef_buf, do_fused_sumcheck_round_inplace_ir, do_fused_sumcheck_round_ir,
        do_sumcheck_round_and_revert_ir, fold_ef_frac_columns_inplace_ir_bufid,
        fold_ef_frac_columns_ir_bufid,
    };

    let RoundInputIR {
        layer,
        layer_len,
        prev_s_eval,
        eq_r_acc,
        lambda,
        seed: shared_seed,
        alpha,
        real_len,
        logical_len,
        round,
    } = input;
    assert_eq!(xi_prev.len(), round, "xi_prev.len() must equal round");
    assert!(round >= 1, "round must be >= 1");
    assert_eq!(real_len, logical_len, "dense-only path");

    let mut pq_size = 2usize << round;
    let mut eq_buffer = SqrtEqLayersIR::from_xi_with_seed(g, &xi_prev[1..], shared_seed, device);

    let mut round_polys: Vec<[BufId; GKR_S_DEG]> = Vec::with_capacity(round);
    let mut r_vec: Vec<BufId> = Vec::with_capacity(round);

    // Round 0: fused compute + revert (modifies `layer` in place — this
    // is expected: each outer round's revert consumes the previous
    // round's revert output and produces the next tree level's contents
    // at `layer[0..pq_size]`).
    let tmp_cap = unsafe { _frac_compute_round_temp_buffer_size((1 << round) as u32) } as usize;
    let d_sum_r0 = add_ef_buf(g, device, "d_sum_r0_fe", GKR_S_DEG - 1);
    let tmp_r0 = add_ef_buf(g, device, "tmp_r0_fe", tmp_cap.max(1));
    let out0 = do_sumcheck_round_and_revert_ir(
        g,
        transcript,
        &mut eq_buffer,
        layer,
        layer_len,
        pq_size,
        logical_len,
        lambda,
        alpha,
        d_sum_r0,
        tmp_r0,
        prev_s_eval,
        xi_prev[0],
        eq_r_acc,
        device,
    );
    round_polys.push(out0.s_evals);
    r_vec.push(out0.r);
    let mut prev_s_eval_cur = out0.prev_s_eval;
    let mut eq_r_acc_cur = out0.eq_r_acc;
    let mut prev_r = out0.r;

    // Ping-pong buffer: fold-eval iters and the final fold go here so
    // `layer[0..pq_size]` stays intact for the NEXT outer round's
    // `compute+revert` (which expects its input untouched by prior
    // rounds' folds). This mirrors the eager driver's `BufferScheduler`
    // choice `LayerToWork → InPlaceWork` for non-last outer rounds.
    let work_len = pq_size; // Enough for the pre-fold state after round 0.
    let work_buf = add_frac_ef_buf(g, device, "fe_work", work_len);

    // Fused inner rounds `1..round-1`. First iter reads `layer`, writes
    // `work_buf`; subsequent iters run in place on `work_buf`. Track
    // `source_real_len` / `source_logical_len` starting from
    // (`real_len`, `logical_len`) and update to the post-fold size after
    // each iteration (matches the eager driver — the compute kernel
    // treats these as the buffer's physical size in its stride math).
    let mut source_real_len = real_len;
    let mut source_logical_len = logical_len;
    let mut in_work = false;
    for (iter_idx, &xi_j) in xi_prev.iter().skip(1).enumerate() {
        let src_pq_size = pq_size;
        let dst_real_len = src_pq_size >> 1;
        let dst_logical_len = src_pq_size >> 1;
        let tmp_cap_i =
            unsafe { _frac_compute_round_temp_buffer_size((src_pq_size >> 2) as u32) } as usize;
        let d_sum = add_ef_buf(g, device, "d_sum_fe", GKR_S_DEG - 1);
        let tmp = add_ef_buf(g, device, "tmp_fe", tmp_cap_i.max(1));
        let out = if iter_idx == 0 {
            // LayerToWork: read layer[0..src_pq_size], write work_buf[0..dst_pq_size].
            do_fused_sumcheck_round_ir(
                g,
                transcript,
                &mut eq_buffer,
                layer,
                work_buf,
                src_pq_size,
                source_real_len,
                source_logical_len,
                lambda,
                prev_r,
                alpha,
                d_sum,
                tmp,
                prev_s_eval_cur,
                xi_j,
                eq_r_acc_cur,
                device,
            )
        } else {
            // InPlaceWork: read/write work_buf.
            do_fused_sumcheck_round_inplace_ir(
                g,
                transcript,
                &mut eq_buffer,
                work_buf,
                src_pq_size,
                source_real_len,
                source_logical_len,
                dst_real_len,
                dst_logical_len,
                lambda,
                prev_r,
                alpha,
                d_sum,
                tmp,
                prev_s_eval_cur,
                xi_j,
                eq_r_acc_cur,
                device,
            )
        };
        round_polys.push(out.s_evals);
        r_vec.push(out.r);
        prev_s_eval_cur = out.prev_s_eval;
        eq_r_acc_cur = out.eq_r_acc;
        prev_r = out.r;
        pq_size >>= 1;
        source_real_len = dst_real_len;
        source_logical_len = dst_logical_len;
        in_work = true;
    }

    // Final fold on the last-sampled challenge. If we had 0 fused iters
    // (round == 1), fold layer[0..pq_size] → work_buf[0..pq_size/2]
    // (out-of-place, so `layer[0..pq_size]` stays intact). Otherwise
    // fold in-place on work_buf.
    let (claim_buf, claim_buf_len) = if in_work {
        fold_ef_frac_columns_inplace_ir_bufid(
            g,
            work_buf,
            pq_size,
            source_real_len,
            source_logical_len,
            prev_r,
            alpha,
        );
        (work_buf, work_len)
    } else {
        fold_ef_frac_columns_ir_bufid(
            g,
            layer,
            work_buf,
            pq_size,
            source_real_len,
            source_logical_len,
            prev_r,
            alpha,
        );
        (work_buf, work_len)
    };
    pq_size >>= 1;
    debug_assert_eq!(
        pq_size, 2,
        "post-fold pq_size must be 2 for claim extraction"
    );
    // Silence unused-write warnings on the last iter's carry state.
    let _ = (prev_s_eval_cur, eq_r_acc_cur);

    let claim = extract_frac_pair_ir(
        g,
        claim_buf,
        claim_buf_len,
        pq_size / 2,
        device,
        "foldeval_claim",
    );
    RoundOutputIR {
        round_polys,
        r_vec,
        p_xi_0: claim.p_xi_0,
        q_xi_0: claim.q_xi_0,
        p_xi_1: claim.p_xi_1,
        q_xi_1: claim.q_xi_1,
    }
}

// ---------------------------------------------------------------------------
// Full sumcheck driver: pipelined per outer round.

/// Full GKR fractional sumcheck built out of the α-tiled pipelined
/// single-outer-round driver above. Mirrors
/// [`super::fractional::fractional_sumcheck_gpu`]:
///
/// 1. `build_segment_tree_ir` folds `leaves` into the top-of-tree revert
///    and observes `root.p` / `root.q` (dense inputs only).
/// 2. Extract layer-0 claims and sample the outer challenge `μ_1`.
/// 3. Outer loop `round in 1..total_rounds` — each round dispatches to
///    [`fractional_sumcheck_round_pipelined_ir`] when `round >= 2 +
///    PIPELINE_WINDOW`, and to [`fractional_sumcheck_round_foldeval_ir`]
///    otherwise. Between rounds, the layer buffer state is threaded
///    forward (kernels write in place), and the transcript samples the
///    outer `μ_j`.
///
/// `seed` is the shared `[EF::ONE]` scalar buffer threaded through every
/// eq-layer construction inside the driver body (see [`RoundInputIR::seed`]);
/// the caller typically registers it as a graph input so no per-launch
/// H2D is emitted for it.
#[allow(clippy::too_many_arguments)]
pub fn fractional_sumcheck_gpu_pipelined_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    leaves: BufId,
    sizes: super::fractional::FractionalInputSize,
    alpha: EF,
    assert_zero: bool,
    seed: BufId,
    device: DeviceType,
) -> Result<super::fractional_ir::FracSumcheckProofIR, FractionalSumcheckError>
where
    TS: FiatShamirTranscriptGraphIR,
{
    use p3_util::log2_strict_usize;

    use super::fractional_ir::{
        build_segment_tree_ir, claim_combine_ir, extract_claim_pair_ir,
        reduce_to_single_evaluation_ir, FracSumcheckProofIR, GkrLayerClaimIR,
    };

    let real_len = sizes.real_len;
    let total_leaves = sizes.logical_len;
    assert_eq!(
        real_len, total_leaves,
        "dense-only path (pipelined driver assumes real_len == logical_len)"
    );
    assert!(real_len > 0, "fractional sumcheck requires nonempty input");

    // Segment-tree build (mutates `leaves` in place through the tree
    // layers, ending with the layer_size = 2 revert that seeds outer
    // round 1). Observes `root.p` / `root.q` and validates `assert_zero`
    // via the shared helper.
    let tree = build_segment_tree_ir(g, transcript, leaves, sizes, alpha, assert_zero, device)?;

    let total_rounds = log2_strict_usize(total_leaves);

    // Layer-0 claims: (layer[0], layer[1]) after the top-of-tree revert.
    let first_claim = extract_claim_pair_ir(g, leaves, real_len, 1, "claim0", device);
    let mut claims_per_layer: Vec<GkrLayerClaimIR> = Vec::with_capacity(total_rounds);
    claims_per_layer.push(first_claim);
    for buf in first_claim.as_array() {
        transcript.observe_ext(g, buf);
    }
    let mu_1 = transcript.sample_ext(g);
    let mut xi_prev: Vec<BufId> = vec![mu_1];
    let mut sumcheck_polys: Vec<Vec<[BufId; GKR_S_DEG]>> = Vec::with_capacity(total_rounds);

    let layer = leaves;
    let layer_len = real_len;
    let w = PIPELINE_WINDOW;

    // Outer-round loop: pipelined for `round >= 2 + w`, fold-eval below.
    for round in 1..total_rounds {
        let prev_claim = *claims_per_layer.last().unwrap();
        let (numer, denom) =
            reduce_to_single_evaluation_ir(g, prev_claim, /* mu */ xi_prev[0], device);
        let lambda = transcript.sample_ext(g);
        let prev_s_eval = claim_combine_ir(g, numer, denom, lambda, device);

        // `eq_r_acc` seeds at `EF::ONE` = the value stored in `seed`, so
        // we reuse the seed BufId. Kernels only read this scalar; they
        // never write it, so the shared BufId is safe here.
        let round_input = RoundInputIR {
            layer,
            layer_len,
            prev_s_eval,
            eq_r_acc: seed,
            lambda,
            seed,
            alpha,
            real_len,
            logical_len: total_leaves,
            round,
        };

        let out = if round >= 2 + w {
            fractional_sumcheck_round_pipelined_ir(g, transcript, round_input, &xi_prev, device)
        } else {
            fractional_sumcheck_round_foldeval_ir(g, transcript, round_input, &xi_prev, device)
        };

        let claim = GkrLayerClaimIR {
            p_xi_0: out.p_xi_0,
            q_xi_0: out.q_xi_0,
            p_xi_1: out.p_xi_1,
            q_xi_1: out.q_xi_1,
        };
        claims_per_layer.push(claim);
        for buf in claim.as_array() {
            transcript.observe_ext(g, buf);
        }
        let mu = transcript.sample_ext(g);
        xi_prev = std::iter::once(mu).chain(out.r_vec).collect();
        sumcheck_polys.push(out.round_polys);
    }

    Ok(FracSumcheckProofIR {
        fractional_sum: (tree.root_p, tree.root_q),
        claims_per_layer,
        sumcheck_polys,
        final_randomness: xi_prev,
    })
}

// ---------------------------------------------------------------------------
// Layout helpers (kept out of the driver body for reuse in tests).

/// `BufInfo` for a plain `Frac<EF>` device buffer of exact length `n`,
/// used when we need to allocate a work / claim buffer inline. Kept here
/// (rather than in [`super::fractional_ir`]) because the pipelined
/// driver is the only caller that pins the concrete size rather than
/// leaning on the fusion planner.
#[allow(dead_code)]
pub(crate) fn frac_ef_buf_exact(g: &mut GraphBuilder, device: DeviceType, name: &str, n: usize) {
    let byte_size = n * std::mem::size_of::<Frac<EF>>();
    let _ = g.add_buf(crypto_compiler::graph_ir::BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst(byte_size as i64),
        concrete_size: byte_size,
        elem_size: std::mem::size_of::<Frac<EF>>(),
    });
}

// ---------------------------------------------------------------------------
// Tests.

#[cfg(test)]
mod tests {
    use crypto_compiler::graph_ir::{ConstBuf, DeviceType, GraphBuilder};
    use openvm_cuda_common::{
        common::get_device,
        copy::MemCopyH2D,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::{
        fractional_sumcheck_round_eager, fractional_sumcheck_round_foldeval_ir,
        fractional_sumcheck_round_pipelined_ir, RoundInputEager, RoundInputIR, RoundOutputEager,
        GKR_S_DEG, PIPELINE_ALPHA, PIPELINE_WINDOW,
    };
    use crate::{
        logup_zerocheck::{
            frac_bench_utils::cc_compiler,
            fractional_ir::{
                add_ext_scalar_buf, add_frac_ef_buf, ef_const_ext_scalar_buf, SqrtEqLayersIR,
            },
        },
        prelude::{EF, SC},
        sponge::DuplexSpongeGpu,
        sponge_graph_ir::DuplexSpongeGpuIR,
    };

    #[link(name = "cudart")]
    extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
    }

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        }
    }

    fn make_host_leaves(len: usize, seed: u64) -> Vec<Frac<EF>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..len)
            .map(|_| Frac {
                p: rng.random::<EF>(),
                q: rng.random::<EF>(),
            })
            .collect()
    }

    fn frac_bytes(leaves: &[Frac<EF>]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(leaves.as_ptr() as *const u8, std::mem::size_of_val(leaves))
        }
    }

    fn ef_from_bytes(bytes: &[u8]) -> EF {
        assert_eq!(bytes.len(), std::mem::size_of::<EF>());
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const EF) }
    }

    /// Stage `leaves` as a mutable working buffer seeded from a const.
    fn frac_working_buf(
        g: &mut GraphBuilder,
        name: &str,
        leaves: &[Frac<EF>],
    ) -> crypto_compiler::graph_ir::BufId {
        let device = DeviceType::Cuda(0);
        let init = add_frac_ef_buf(g, device, &format!("{name}_init"), leaves.len());
        g.insert_const(init, ConstBuf::HostBuf(frac_bytes(leaves).to_vec()));
        let buf = add_frac_ef_buf(g, device, name, leaves.len());
        g.insert_memcpy(init, buf);
        buf
    }

    /// Run the eager baseline for the single-window regime and return the
    /// full round output.
    #[allow(clippy::too_many_arguments)]
    fn run_eager(
        leaves: &[Frac<EF>],
        xi_prev: &[EF],
        prev_s_eval: EF,
        eq_r_acc: EF,
        lambda: EF,
        alpha: EF,
        round: usize,
        ctx: &GpuDeviceCtx,
    ) -> RoundOutputEager {
        let layer = leaves.to_device_on(ctx).expect("H2D");
        let n = leaves.len();
        let mut transcript = DuplexSpongeGpu::default();
        let out = fractional_sumcheck_round_eager::<SC, _>(
            RoundInputEager {
                layer,
                xi_prev: xi_prev.to_vec(),
                prev_s_eval,
                eq_r_acc,
                lambda,
                alpha,
                real_len: n,
                logical_len: n,
                round,
            },
            &mut transcript,
            ctx,
        )
        .expect("eager round");
        ctx.stream.synchronize().expect("sync");
        out
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn run_pipelined_ir(
        leaves: &[Frac<EF>],
        xi_prev: &[EF],
        prev_s_eval: EF,
        eq_r_acc: EF,
        lambda: EF,
        alpha: EF,
        round: usize,
        ctx: &GpuDeviceCtx,
    ) -> (Vec<[EF; GKR_S_DEG]>, Vec<EF>, EF, EF, EF, EF) {
        let device = DeviceType::Cuda(0);
        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let layer = frac_working_buf(&mut g, "pipelined_layer", leaves);
        let xi_bufs: Vec<crypto_compiler::graph_ir::BufId> = xi_prev
            .iter()
            .enumerate()
            .map(|(j, v)| ef_const_ext_scalar_buf(&mut g, device, &format!("xi_{j}"), *v))
            .collect();
        let lambda_buf = ef_const_ext_scalar_buf(&mut g, device, "lambda", lambda);
        let prev_buf = ef_const_ext_scalar_buf(&mut g, device, "prev_s_eval", prev_s_eval);
        let eqacc_buf = ef_const_ext_scalar_buf(&mut g, device, "eq_r_acc", eq_r_acc);
        let seed_buf = SqrtEqLayersIR::seed_layer(&mut g, device);
        let out = fractional_sumcheck_round_pipelined_ir(
            &mut g,
            &mut transcript,
            RoundInputIR {
                layer,
                layer_len: leaves.len(),
                prev_s_eval: prev_buf,
                eq_r_acc: eqacc_buf,
                lambda: lambda_buf,
                seed: seed_buf,
                alpha,
                real_len: leaves.len(),
                logical_len: leaves.len(),
                round,
            },
            &xi_bufs,
            device,
        );

        // Register all outputs, chained via memcpy so the graph runtime
        // makes them addressable.
        let mut exports: Vec<crypto_compiler::graph_ir::BufId> = Vec::new();
        let mut export =
            |g: &mut GraphBuilder, src: crypto_compiler::graph_ir::BufId, name: &str| {
                let dst = add_ext_scalar_buf(g, device, name);
                g.insert_memcpy(src, dst);
                g.register_output(dst);
                exports.push(dst);
            };
        for (i, s) in out.round_polys.iter().enumerate() {
            for (k, &b) in s.iter().enumerate() {
                export(&mut g, b, &format!("s_{i}_{k}"));
            }
        }
        for (i, &b) in out.r_vec.iter().enumerate() {
            export(&mut g, b, &format!("r_{i}"));
        }
        export(&mut g, out.p_xi_0, "p_xi_0");
        export(&mut g, out.q_xi_0, "q_xi_0");
        export(&mut g, out.p_xi_1, "p_xi_1");
        export(&mut g, out.q_xi_1, "q_xi_1");

        let mut exe = cc_compiler(device).compile(g).expect("graph compile");
        exe.run(ctx).expect("graph run");
        let vals: Vec<EF> = exports
            .iter()
            .map(|&bid| {
                let idx = (0..exe.num_outputs())
                    .find(|&i| exe.output_buf_id(i) == bid)
                    .expect("output buf");
                let bytes = exe.get_output(idx).to_host_on(ctx).expect("D2H");
                ef_from_bytes(&bytes)
            })
            .collect();
        let mut it = vals.into_iter();
        let round_polys: Vec<[EF; GKR_S_DEG]> = (0..round)
            .map(|_| std::array::from_fn(|_| it.next().unwrap()))
            .collect();
        let r_vec: Vec<EF> = (0..round).map(|_| it.next().unwrap()).collect();
        let p_xi_0 = it.next().unwrap();
        let q_xi_0 = it.next().unwrap();
        let p_xi_1 = it.next().unwrap();
        let q_xi_1 = it.next().unwrap();
        assert!(it.next().is_none(), "leftover values");
        (round_polys, r_vec, p_xi_0, q_xi_0, p_xi_1, q_xi_1)
    }

    fn random_case_with_round(
        round: usize,
        seed: u64,
    ) -> (Vec<Frac<EF>>, Vec<EF>, EF, EF, EF, EF, usize) {
        let n = 2usize << round; // pq_size at round start.
        let mut rng = StdRng::seed_from_u64(seed);
        let leaves = make_host_leaves(n, seed ^ 0xF00D);
        let xi_prev: Vec<EF> = (0..round).map(|_| rng.random::<EF>()).collect();
        let prev_s_eval: EF = rng.random();
        let eq_r_acc = EF::ONE;
        let lambda: EF = rng.random();
        let alpha: EF = rng.random();
        (leaves, xi_prev, prev_s_eval, eq_r_acc, lambda, alpha, round)
    }

    #[allow(dead_code)]
    fn random_case(seed: u64) -> (Vec<Frac<EF>>, Vec<EF>, EF, EF, EF, EF, usize) {
        // Default perf-test size: `j = 12` — 5 pipelined slots (one
        // startup M + three concurrent mid-loop M builds) plus one
        // fold-eval tail slot. Big enough that the pipelined + fold-eval
        // transition is exercised on the timing path.
        random_case_with_round(12, seed)
    }

    fn assert_pipelined_matches_eager(round: usize, seed: u64) {
        let ctx = test_ctx();
        let (leaves, xi_prev, prev_s_eval, eq_r_acc, lambda, alpha, round) =
            random_case_with_round(round, seed);
        let want = run_eager(
            &leaves,
            &xi_prev,
            prev_s_eval,
            eq_r_acc,
            lambda,
            alpha,
            round,
            &ctx,
        );
        let (got_polys, got_rs, gp0, gq0, gp1, gq1) = run_pipelined_ir(
            &leaves,
            &xi_prev,
            prev_s_eval,
            eq_r_acc,
            lambda,
            alpha,
            round,
            &ctx,
        );
        assert_eq!(
            got_polys, want.round_polys,
            "round={round} seed={seed:#x}: round_polys"
        );
        assert_eq!(got_rs, want.r_vec, "round={round} seed={seed:#x}: r_vec");
        assert_eq!(gp0, want.p_xi_0, "round={round} seed={seed:#x}: p_xi_0");
        assert_eq!(gq0, want.q_xi_0, "round={round} seed={seed:#x}: q_xi_0");
        assert_eq!(gp1, want.p_xi_1, "round={round} seed={seed:#x}: p_xi_1");
        assert_eq!(gq1, want.q_xi_1, "round={round} seed={seed:#x}: q_xi_1");
    }

    #[test]
    fn pipelined_ir_matches_eager_round_8() {
        // j = 8, w = 4, α = 2 → L = 7, boundaries [0, 2, 4, 6, 7],
        // 4 half-slots. Slots 0–2 pipelined (M_0 built at startup,
        // M_1 at base 2 built during slot 1); slot 3 falls back to
        // fold-eval (M_2 at base 4 would need rem_n = 3 < w).
        for seed in [0xC0FFEE_u64, 0xDEAD_BEEF, 0x1234_5678] {
            assert_pipelined_matches_eager(8, seed);
        }
    }

    #[test]
    fn pipelined_ir_matches_eager_round_12() {
        // j = 12, w = 4, α = 2 → L = 11, boundaries [0, 2, 4, 6, 8, 10, 11],
        // 6 half-slots. Slots 0–4 pipelined (M_0 startup + M_1..M_3
        // built during slots 1–3); slot 5 falls back to fold-eval
        // (M_4 at base 8 would need rem_n = 3 < w). Exercises multiple
        // mid-loop M builds and the `queued_m → current_m` swap across
        // several iterations.
        for seed in [0xABCD_1234_u64, 0xFEED_C0DE] {
            assert_pipelined_matches_eager(12, seed);
        }
    }

    // ---- Bench: eager vs pipelined-IR at `j = FRAC_ROUND` (default 12) ---
    //
    // Structure mirrors [`crate::logup_zerocheck::fractional_ir::tests::
    // bench_fractional_sumcheck_eager_vs_ir`]:
    //   1. Setup — build + compile the pipelined graph exe (always run with `CC_FUSION=v2`; v1
    //      is deprecated). Set the input, run one warmup, verify correctness against the eager
    //      baseline, then capture a CUDA graph. Everything here happens *before*
    //      `cudaProfilerStart`, so the profile contains only measured kernel work.
    //   2. Timed pass — a single `cudaProfilerStart / Stop` window wraps `ITERS` eager runs
    //      followed by `ITERS` pipelined `launch_graph` runs. Each iteration is bracketed by its
    //      own NVTX range so the profile shows one bar per run.
    //
    // Env vars:
    //   - `FRAC_ROUND` — outer-round index `j` (default 12); pq_size at round start = `2 << j`.
    //   - `NSYS_ENABLED=1` — flip on the `cudaProfilerStart/Stop` + NVTX push/pop calls (no-op
    //     otherwise, so unattended `cargo nextest` runs still time cleanly).
    //   - `CC_FUSION=v2` — v2 fusion (v1 is deprecated). Other `cc_compiler` knobs (`CC_STREAMS`,
    //     `CC_FUSION_SOLVER_SECS`, etc.) flow through as usual.
    //
    // Output registration: every s-poly BufId (`round_polys[i][k]` for `i ∈ 0..j`, `k ∈ 0..3`) and
    // every Fiat-Shamir sample `r_vec[i]` is `register_output`-ed directly — no post-driver
    // `insert_memcpy` — plus the four claim BufIds so the correctness gate can materialise them.
    // The graph runtime keeps registered BufIds live at their producing kernels' output slots.
    //
    // Recommended nsys invocation (matches the AGENTS.md profiling rules):
    //   NSYS_ENABLED=1 CC_FUSION=v2 FRAC_ROUND=12 \
    //     nsys profile --capture-range=cudaProfilerApi \
    //       --cuda-graph-trace=node --gpu-metrics-devices=visible \
    //       --trace=cuda,nvtx -o pipelined_bench \
    //     cargo nextest run -p openvm-cuda-backend --features graph-ir \
    //       --run-ignored all --no-capture \
    //       -E 'test(bench_pipelined_ir_vs_eager)'
    #[test]
    #[ignore = "benchmark; run explicitly with --run-ignored"]
    fn bench_pipelined_ir_vs_eager() {
        use std::time::Instant;

        use crypto_compiler::graph_exe::GraphExe;

        use super::super::frac_bench_utils::cc_compiler;

        const WARMUP: usize = 2;
        const ITERS: usize = 5;

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);

        // ---- Input configuration -----------------------------------------
        let round: usize = std::env::var("FRAC_ROUND")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(12);
        assert!(
            round >= 2 + PIPELINE_WINDOW,
            "FRAC_ROUND must be >= 2 + PIPELINE_WINDOW = {}",
            2 + PIPELINE_WINDOW,
        );
        let nsys_enabled = std::env::var_os("NSYS_ENABLED").is_some();
        let (leaves, xi_prev, prev_s_eval, eq_r_acc, lambda, alpha, _) =
            random_case_with_round(round, 0xACE1_5EED_u64);

        println!(
            "=== pipelined GKR single-round bench: j = round = {round}, \
             pq_size = 2 << j = {} elements, w = {PIPELINE_WINDOW}, α = {PIPELINE_ALPHA} ===",
            leaves.len(),
        );

        // ---- Build + compile pipelined exe (before profiler window) -----
        //
        // Every EF-scalar the pipelined driver reads (`xi_prev[..]`,
        // `lambda`, `prev_s_eval`, `eq_r_acc`, `seed`) is registered as a
        // graph input rather than baked as a `ConstBuf::HostBuf` const.
        // Inputs are populated exactly once via `set_input` (D2D copy into
        // the graph's pool slot) and persist across every `launch_graph`
        // replay, so the captured CUDA graph contains **no** per-launch
        // H2D copies for these scalars. Baking them as `HostBuf` consts
        // would emit one `cudaMemcpyAsync(HostToDevice)` per const per
        // launch, exactly the "minor H2D" transfers this bench is
        // designed to avoid.
        let t_build = Instant::now();
        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let input_buf = add_frac_ef_buf(&mut g, device, "leaves_in", leaves.len());
        let layer_buf = add_frac_ef_buf(&mut g, device, "layer_work", leaves.len());
        g.insert_memcpy(input_buf, layer_buf);
        g.register_input(input_buf);

        // Every scalar becomes its own registered input. Order in the
        // exe's input list: [leaves, xi_0..xi_{j-1}, lambda, prev_s_eval,
        // eq_r_acc, seed] — matched by the `set_input` block below.
        let xi_bufs: Vec<crypto_compiler::graph_ir::BufId> = xi_prev
            .iter()
            .enumerate()
            .map(|(j, _v)| {
                let b = add_ext_scalar_buf(&mut g, device, &format!("xi_{j}_in"));
                g.register_input(b);
                b
            })
            .collect();
        let lambda_buf = add_ext_scalar_buf(&mut g, device, "lambda_in");
        g.register_input(lambda_buf);
        let prev_buf = add_ext_scalar_buf(&mut g, device, "prev_s_eval_in");
        g.register_input(prev_buf);
        let eqacc_buf = add_ext_scalar_buf(&mut g, device, "eq_r_acc_in");
        g.register_input(eqacc_buf);
        let seed_buf = add_ext_scalar_buf(&mut g, device, "seed_in");
        g.register_input(seed_buf);

        let out = fractional_sumcheck_round_pipelined_ir(
            &mut g,
            &mut transcript,
            RoundInputIR {
                layer: layer_buf,
                layer_len: leaves.len(),
                prev_s_eval: prev_buf,
                eq_r_acc: eqacc_buf,
                lambda: lambda_buf,
                seed: seed_buf,
                alpha,
                real_len: leaves.len(),
                logical_len: leaves.len(),
                round,
            },
            &xi_bufs,
            device,
        );

        // Register every s-poly BufId, every r_vec sample, and the four
        // claim BufIds directly as outputs (no `insert_memcpy` between
        // the driver's writers and the graph output slots).
        for s in &out.round_polys {
            for &b in s.iter() {
                g.register_output(b);
            }
        }
        for &b in &out.r_vec {
            g.register_output(b);
        }
        for b in [out.p_xi_0, out.q_xi_0, out.p_xi_1, out.q_xi_1] {
            g.register_output(b);
        }
        let n_nodes = g.nodes.len();
        let build_ms = t_build.elapsed().as_secs_f64() * 1e3;

        let t_compile = Instant::now();
        let mut exe: GraphExe = cc_compiler(device).compile(g).expect("graph compile");
        let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
        if let Some(v2) = exe.fusion_report().and_then(|r| r.v2.as_ref()) {
            println!(
                "fusion v2: nodes {} -> {}, inserted={}, selected={}, fallback={:?}",
                v2.nodes_before,
                v2.nodes_after,
                v2.candidates_inserted,
                v2.selected_from_solver,
                v2.fallback_reason,
            );
        }
        println!(
            "graph build: {build_ms:>8.2} ms ({n_nodes} nodes); compile: {compile_ms:>8.2} ms \
             ({} unique modules, {} loaded from cache)",
            exe.num_unique_modules(),
            exe.num_cached_modules(),
        );

        // ---- H2D + set_input: one D2D copy per input into its pool slot,
        // done exactly once before the profiler window. The captured
        // graph reads directly from these slots and does not re-upload.
        let n_inputs = round + 5; // leaves + `round` xi + lambda + prev_s_eval + eq_r_acc + seed
        assert_eq!(
            exe.num_inputs(),
            n_inputs,
            "expected {n_inputs} registered inputs",
        );
        let d_leaves = frac_bytes(&leaves).to_device_on(&ctx).expect("H2D leaves");
        exe.set_input(&ctx, 0, &d_leaves).expect("set_input leaves");
        // Helper: upload an `EF` as a `DeviceBuffer<u8>` sized to `size_of::<EF>()`.
        let ef_to_dev = |v: EF| -> openvm_cuda_common::d_buffer::DeviceBuffer<u8> {
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(&v as *const EF as *const u8, std::mem::size_of::<EF>())
                    .to_vec()
            };
            bytes.as_slice().to_device_on(&ctx).expect("H2D EF scalar")
        };
        let d_xis: Vec<_> = xi_prev.iter().copied().map(ef_to_dev).collect();
        for (j, d) in d_xis.iter().enumerate() {
            exe.set_input(&ctx, 1 + j, d).expect("set_input xi");
        }
        let d_lambda = ef_to_dev(lambda);
        let d_prev = ef_to_dev(prev_s_eval);
        let d_eqacc = ef_to_dev(eq_r_acc);
        let d_seed = ef_to_dev(EF::ONE);
        exe.set_input(&ctx, 1 + round, &d_lambda)
            .expect("set_input lambda");
        exe.set_input(&ctx, 2 + round, &d_prev)
            .expect("set_input prev_s_eval");
        exe.set_input(&ctx, 3 + round, &d_eqacc)
            .expect("set_input eq_r_acc");
        exe.set_input(&ctx, 4 + round, &d_seed)
            .expect("set_input seed");
        ctx.stream.synchronize().expect("sync after set_input");

        // ---- Correctness check (before profiler window) -----------------
        // One graph run to materialise every output; then compare against
        // the eager reference artifact-by-artifact.
        exe.run(&ctx).expect("correctness graph run");
        ctx.stream.synchronize().expect("sync");
        let read_ef_from_bufid = |bid: crypto_compiler::graph_ir::BufId, exe: &GraphExe| -> EF {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == bid)
                .expect("registered output BufId");
            ef_from_bytes(&exe.get_output(idx).to_host_on(&ctx).expect("D2H"))
        };
        let got_polys: Vec<[EF; GKR_S_DEG]> = out
            .round_polys
            .iter()
            .map(|s| std::array::from_fn(|k| read_ef_from_bufid(s[k], &exe)))
            .collect();
        let got_rs: Vec<EF> = out
            .r_vec
            .iter()
            .map(|&b| read_ef_from_bufid(b, &exe))
            .collect();
        let gp0 = read_ef_from_bufid(out.p_xi_0, &exe);
        let gq0 = read_ef_from_bufid(out.q_xi_0, &exe);
        let gp1 = read_ef_from_bufid(out.p_xi_1, &exe);
        let gq1 = read_ef_from_bufid(out.q_xi_1, &exe);

        let want = run_eager(
            &leaves,
            &xi_prev,
            prev_s_eval,
            eq_r_acc,
            lambda,
            alpha,
            round,
            &ctx,
        );
        assert_eq!(got_polys, want.round_polys, "round_polys mismatch");
        assert_eq!(got_rs, want.r_vec, "r_vec mismatch");
        assert_eq!(gp0, want.p_xi_0, "p_xi_0 mismatch");
        assert_eq!(gq0, want.q_xi_0, "q_xi_0 mismatch");
        assert_eq!(gp1, want.p_xi_1, "p_xi_1 mismatch");
        assert_eq!(gq1, want.q_xi_1, "q_xi_1 mismatch");
        println!(
            "correctness: pipelined vs eager match ({} s-polys, {} samples, 4 claims)",
            got_polys.len(),
            got_rs.len(),
        );

        // ---- Warmup (outside profiler window) ---------------------------
        for _ in 0..WARMUP {
            let _ = run_eager(
                &leaves,
                &xi_prev,
                prev_s_eval,
                eq_r_acc,
                lambda,
                alpha,
                round,
                &ctx,
            );
            exe.run(&ctx).expect("graph warmup");
        }
        ctx.stream.synchronize().expect("sync post-warmup");

        // Capture the CUDA graph so timed pipelined iterations are pure
        // `cudaGraphLaunch` replays (no per-node host dispatch overhead
        // inside the profile window).
        let t_cap = Instant::now();
        exe.capture_graph(&ctx).expect("graph capture");
        exe.launch_graph(&ctx).expect("graph capture warmup");
        ctx.stream.synchronize().expect("sync post-capture");
        println!(
            "cuda graph capture: {:>8.2} ms",
            t_cap.elapsed().as_secs_f64() * 1e3,
        );

        // ---- Timed pass inside a single cudaProfilerStart/Stop window ---
        let mut eager_ms: Vec<f64> = Vec::with_capacity(ITERS);
        let mut pipelined_ms: Vec<f64> = Vec::with_capacity(ITERS);
        if nsys_enabled {
            unsafe { cudaProfilerStart() };
        }
        for i in 0..ITERS {
            ctx.stream.synchronize().expect("sync pre-eager");
            let t0 = Instant::now();
            if nsys_enabled {
                nvtx::range_push!("eager j={round} iter={i}");
            }
            let _ = run_eager(
                &leaves,
                &xi_prev,
                prev_s_eval,
                eq_r_acc,
                lambda,
                alpha,
                round,
                &ctx,
            );
            ctx.stream.synchronize().expect("sync post-eager");
            if nsys_enabled {
                nvtx::range_pop!();
            }
            eager_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        for i in 0..ITERS {
            ctx.stream.synchronize().expect("sync pre-pipelined");
            let t0 = Instant::now();
            if nsys_enabled {
                nvtx::range_push!("pipelined j={round} iter={i}");
            }
            exe.launch_graph(&ctx).expect("pipelined launch_graph");
            ctx.stream.synchronize().expect("sync post-pipelined");
            if nsys_enabled {
                nvtx::range_pop!();
            }
            pipelined_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        if nsys_enabled {
            unsafe { cudaProfilerStop() };
        }

        let eager_mean = eager_ms.iter().sum::<f64>() / ITERS as f64;
        let pipelined_mean = pipelined_ms.iter().sum::<f64>() / ITERS as f64;
        println!(
            "\n--- pipelined GKR single-round: j = {round} ---\n\
             eager     : {eager_ms:>8.3?} ms (mean {eager_mean:.3} ms)\n\
             pipelined : {pipelined_ms:>8.3?} ms (mean {pipelined_mean:.3} ms, \
             {:.3}x eager)",
            pipelined_mean / eager_mean,
        );
    }

    // ---- Full-proof bench: pipelined multi-round driver vs eager ----------
    //
    // Same structure as `bench_pipelined_ir_vs_eager` but for the FULL
    // fractional-sumcheck proof: builds
    // [`fractional_sumcheck_gpu_pipelined_ir`] against the eager
    // [`super::super::super::fractional::fractional_sumcheck_gpu`]
    // baseline. Correctness runs first (every proof artifact — root_sum,
    // per-layer claims, sumcheck polys, final randomness — is registered
    // as a graph output BufId directly; no `insert_memcpy` after the
    // driver), then eager + graph iterations run inside one
    // `cudaProfilerStart / Stop` window with per-iteration NVTX ranges.
    //
    // Env vars:
    //   - `FRAC_LOG_N` — comma-separated log2(leaf count), first entry taken (default 16).
    //   - `NSYS_ENABLED=1` — flip on the profiler + NVTX calls.
    //   - `CC_FUSION=v2` — v2 fusion (v1 is deprecated; always run with v2). Other `cc_compiler`
    //     knobs (`CC_STREAMS`, `CC_FUSION_SOLVER_SECS`, `CC_FUSION_MAX_ALTS`, etc.) flow through
    //     as usual.
    //
    // Recommended nsys invocation (per AGENTS.md):
    //   NSYS_ENABLED=1 CC_FUSION=v2 FRAC_LOG_N=16 \
    //     nsys profile --capture-range=cudaProfilerApi \
    //       --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    //       --trace=cuda,nvtx -o pipelined_full_bench \
    //     cargo nextest run -p openvm-cuda-backend --features graph-ir \
    //       --run-ignored all --no-capture \
    //       -E 'test(bench_pipelined_full_sumcheck_vs_eager)'
    #[test]
    #[ignore = "benchmark; run explicitly with --run-ignored"]
    fn bench_pipelined_full_sumcheck_vs_eager() {
        use std::time::Instant;

        use crypto_compiler::graph_exe::GraphExe;
        use openvm_cuda_common::memory_manager::MemTracker;
        use openvm_stark_backend::prover::fractional_sumcheck_gkr::FracSumcheckProof;
        use p3_util::log2_strict_usize;

        use super::super::{
            frac_bench_utils::{cc_compiler, frac_log_n_single},
            fractional::fractional_sumcheck_gpu,
            fractional_ir::FracSumcheckProofIR,
        };
        use crate::logup_zerocheck::fractional_ir_pipelined::fractional_sumcheck_gpu_pipelined_ir;

        const WARMUP: usize = 2;
        const ITERS: usize = 3;

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);

        // ---- Input configuration ---------------------------------------
        let log_n: usize = frac_log_n_single(16);
        let n = 1usize << log_n;
        let sizes = super::super::FractionalInputSize::new(n, n);
        let leaves = make_host_leaves(n, 0x5EED_F011_u64 ^ log_n as u64);
        let mut rng = StdRng::seed_from_u64(0xA1FA ^ log_n as u64);
        let alpha: EF = rng.random();
        let nsys_enabled = std::env::var_os("NSYS_ENABLED").is_some();

        println!(
            "=== pipelined full fractional sumcheck bench: n = 2^{log_n} = {n} leaves, \
             w = {}, α = {} ===",
            PIPELINE_WINDOW, PIPELINE_ALPHA,
        );

        // ---- Eager reference proof (also warms up eager code paths) ----
        let t_e = Instant::now();
        let mut sponge_ref = DuplexSpongeGpu::default();
        let d_leaves_ref: openvm_cuda_common::d_buffer::DeviceBuffer<Frac<EF>> =
            leaves.as_slice().to_device_on(&ctx).expect("H2D");
        let mut mem_ref = MemTracker::start("bench.full_pipelined_eager");
        let (eager_proof, eager_xi): (FracSumcheckProof<SC>, Vec<EF>) =
            fractional_sumcheck_gpu::<SC, _>(
                &mut sponge_ref,
                d_leaves_ref,
                sizes,
                alpha,
                false,
                &mut mem_ref,
                &ctx,
            )
            .expect("eager warmup");
        ctx.stream.synchronize().expect("sync");
        println!(
            "[bench] eager warmup: {} ms, {} claim layers, {} total sumcheck polys",
            t_e.elapsed().as_secs_f64() * 1e3,
            eager_proof.claims_per_layer.len(),
            eager_proof
                .sumcheck_polys
                .iter()
                .map(|l| l.len())
                .sum::<usize>(),
        );

        // ---- Build + compile pipelined full-sumcheck graph exe ---------
        //
        // Every EF-scalar the driver needs (`alpha` is a compile-time
        // constant baked into kernel signatures, but the shared seed is
        // a graph input). Fresh `leaves` input BufId → memcpy into a
        // writable `layer` copy that the driver mutates through segment
        // tree build + all outer rounds.
        let t_build = Instant::now();
        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let input_buf = add_frac_ef_buf(&mut g, device, "leaves_in", n);
        let layer_buf = add_frac_ef_buf(&mut g, device, "layer_work", n);
        g.insert_memcpy(input_buf, layer_buf);
        g.register_input(input_buf);
        let seed_buf = add_ext_scalar_buf(&mut g, device, "seed_in");
        g.register_input(seed_buf);

        let proof_ir: FracSumcheckProofIR = fractional_sumcheck_gpu_pipelined_ir(
            &mut g,
            &mut transcript,
            layer_buf,
            sizes,
            alpha,
            /* assert_zero */ false,
            seed_buf,
            device,
        )
        .expect("fractional_sumcheck_gpu_pipelined_ir");

        // Register EVERY proof artifact BufId directly as a graph output
        // (no `insert_memcpy` between the driver's writers and the graph
        // output slots) — per the user's "no memcpy, register directly"
        // rule. Order matches `reshape_proof_efs` below.
        let mut output_bids: Vec<crypto_compiler::graph_ir::BufId> = Vec::new();
        {
            let (rp, rq) = proof_ir.fractional_sum;
            g.register_output(rp);
            output_bids.push(rp);
            g.register_output(rq);
            output_bids.push(rq);
            for claim in &proof_ir.claims_per_layer {
                for b in claim.as_array() {
                    g.register_output(b);
                    output_bids.push(b);
                }
            }
            for round_polys in &proof_ir.sumcheck_polys {
                for s in round_polys {
                    for &b in s.iter() {
                        g.register_output(b);
                        output_bids.push(b);
                    }
                }
            }
            for &b in &proof_ir.final_randomness {
                g.register_output(b);
                output_bids.push(b);
            }
        }
        let n_nodes = g.nodes.len();
        let build_ms = t_build.elapsed().as_secs_f64() * 1e3;

        let t_compile = Instant::now();
        let mut exe: GraphExe = cc_compiler(device).compile(g).expect("graph compile");
        let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
        if let Some(v2) = exe.fusion_report().and_then(|r| r.v2.as_ref()) {
            println!(
                "fusion v2: nodes {} -> {}, inserted={}, selected={}, fallback={:?}",
                v2.nodes_before,
                v2.nodes_after,
                v2.candidates_inserted,
                v2.selected_from_solver,
                v2.fallback_reason,
            );
        }
        println!(
            "graph build: {build_ms:>8.2} ms ({n_nodes} nodes); compile: {compile_ms:>8.2} ms \
             ({} unique modules, {} loaded from cache)",
            exe.num_unique_modules(),
            exe.num_cached_modules(),
        );

        // ---- H2D + set_input (once, before the profiler window) --------
        assert_eq!(exe.num_inputs(), 2, "expected [leaves_in, seed_in] inputs");
        let d_leaves = frac_bytes(&leaves).to_device_on(&ctx).expect("H2D leaves");
        exe.set_input(&ctx, 0, &d_leaves).expect("set_input leaves");
        let seed_bytes: Vec<u8> = unsafe {
            let one = EF::ONE;
            std::slice::from_raw_parts(&one as *const EF as *const u8, std::mem::size_of::<EF>())
                .to_vec()
        };
        let d_seed = seed_bytes.as_slice().to_device_on(&ctx).expect("H2D seed");
        exe.set_input(&ctx, 1, &d_seed).expect("set_input seed");
        ctx.stream.synchronize().expect("sync after set_input");

        // ---- Correctness: run graph, compare full proof to eager ------
        exe.run(&ctx).expect("correctness run");
        ctx.stream.synchronize().expect("sync post-correctness");
        let read_ef_bid = |bid: crypto_compiler::graph_ir::BufId, exe: &GraphExe| -> EF {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == bid)
                .expect("registered output BufId");
            ef_from_bytes(&exe.get_output(idx).to_host_on(&ctx).expect("D2H"))
        };

        // Reshape graph outputs matching the eager `FracSumcheckProof`.
        let got_sum = (
            read_ef_bid(proof_ir.fractional_sum.0, &exe),
            read_ef_bid(proof_ir.fractional_sum.1, &exe),
        );
        let got_claims: Vec<[EF; 4]> = proof_ir
            .claims_per_layer
            .iter()
            .map(|c| std::array::from_fn(|k| read_ef_bid(c.as_array()[k], &exe)))
            .collect();
        let got_polys: Vec<Vec<[EF; GKR_S_DEG]>> = proof_ir
            .sumcheck_polys
            .iter()
            .map(|round_polys| {
                round_polys
                    .iter()
                    .map(|s| std::array::from_fn(|k| read_ef_bid(s[k], &exe)))
                    .collect()
            })
            .collect();
        let got_xi: Vec<EF> = proof_ir
            .final_randomness
            .iter()
            .map(|&b| read_ef_bid(b, &exe))
            .collect();

        assert_eq!(
            got_sum, eager_proof.fractional_sum,
            "fractional_sum mismatch"
        );
        assert_eq!(
            got_claims.len(),
            eager_proof.claims_per_layer.len(),
            "claims_per_layer length mismatch",
        );
        for (i, (got, want)) in got_claims
            .iter()
            .zip(&eager_proof.claims_per_layer)
            .enumerate()
        {
            assert_eq!(
                *got,
                [want.p_xi_0, want.q_xi_0, want.p_xi_1, want.q_xi_1],
                "layer {i} claims mismatch",
            );
        }
        assert_eq!(
            got_polys, eager_proof.sumcheck_polys,
            "sumcheck_polys mismatch"
        );
        assert_eq!(got_xi, eager_xi, "final_randomness mismatch");
        let total_rounds = log2_strict_usize(n);
        println!(
            "correctness: full proof matches eager ({} claim layers, {} inner sumcheck rounds, \
             {} final randomness)",
            got_claims.len(),
            got_polys.iter().map(|r| r.len()).sum::<usize>(),
            got_xi.len(),
        );

        // ---- Warmup (outside profiler window) --------------------------
        for _ in 0..WARMUP {
            // Fresh eager transcript state per warmup pass (matches timed loop below).
            let mut sp = DuplexSpongeGpu::default();
            let d_l: openvm_cuda_common::d_buffer::DeviceBuffer<Frac<EF>> =
                leaves.as_slice().to_device_on(&ctx).expect("H2D warmup");
            let mut m = MemTracker::start("bench.full_warm");
            let _ =
                fractional_sumcheck_gpu::<SC, _>(&mut sp, d_l, sizes, alpha, false, &mut m, &ctx)
                    .expect("eager warmup");
            exe.run(&ctx).expect("graph warmup");
        }
        ctx.stream.synchronize().expect("sync post-warmup");

        // Capture CUDA graph so timed graph iterations are pure replays.
        let t_cap = Instant::now();
        exe.capture_graph(&ctx).expect("capture_graph");
        exe.launch_graph(&ctx).expect("capture warmup launch");
        ctx.stream.synchronize().expect("sync post-capture");
        println!(
            "cuda graph capture: {:>8.2} ms (total_rounds = {})",
            t_cap.elapsed().as_secs_f64() * 1e3,
            total_rounds,
        );

        // ---- Timed pass inside cudaProfilerStart/Stop -----------------
        let mut eager_ms: Vec<f64> = Vec::with_capacity(ITERS);
        let mut pipelined_ms: Vec<f64> = Vec::with_capacity(ITERS);
        if nsys_enabled {
            unsafe { cudaProfilerStart() };
        }
        for i in 0..ITERS {
            let mut sp = DuplexSpongeGpu::default();
            let d_l: openvm_cuda_common::d_buffer::DeviceBuffer<Frac<EF>> =
                leaves.as_slice().to_device_on(&ctx).expect("H2D iter");
            let mut m = MemTracker::start("bench.full_eager");
            ctx.stream.synchronize().expect("sync pre-eager");
            let t0 = Instant::now();
            if nsys_enabled {
                nvtx::range_push!("eager log_n={log_n} iter={i}");
            }
            let _ =
                fractional_sumcheck_gpu::<SC, _>(&mut sp, d_l, sizes, alpha, false, &mut m, &ctx)
                    .expect("eager iter");
            ctx.stream.synchronize().expect("sync post-eager");
            if nsys_enabled {
                nvtx::range_pop!();
            }
            eager_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        for i in 0..ITERS {
            ctx.stream.synchronize().expect("sync pre-pipelined");
            let t0 = Instant::now();
            if nsys_enabled {
                nvtx::range_push!("pipelined log_n={log_n} iter={i}");
            }
            exe.launch_graph(&ctx).expect("launch_graph iter");
            ctx.stream.synchronize().expect("sync post-pipelined");
            if nsys_enabled {
                nvtx::range_pop!();
            }
            pipelined_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        if nsys_enabled {
            unsafe { cudaProfilerStop() };
        }

        let eager_mean = eager_ms.iter().sum::<f64>() / ITERS as f64;
        let pipelined_mean = pipelined_ms.iter().sum::<f64>() / ITERS as f64;
        println!(
            "\n--- pipelined full fractional sumcheck: n = 2^{log_n} = {n} ---\n\
             eager     : {eager_ms:>8.3?} ms (mean {eager_mean:.3} ms)\n\
             pipelined : {pipelined_ms:>8.3?} ms (mean {pipelined_mean:.3} ms, \
             {:.3}x eager)",
            pipelined_mean / eager_mean,
        );
    }

    // ---- Sweep bench: per-outer-round-j single-round exes, one nsys window --
    //
    // For each `j` in `FRAC_ROUNDS` (default `4,10,16,20,24`), build the
    // single-outer-round graph (`fractional_sumcheck_round_pipelined_ir`
    // for `j >= 2 + PIPELINE_WINDOW`, `fractional_sumcheck_round_foldeval_ir`
    // otherwise), compile, capture the CUDA graph, and warm up — all
    // outside the profiler window. Then inside a single
    // `cudaProfilerStart / Stop`, launch each captured graph `ITERS`
    // times, each iteration wrapped in an NVTX range `pipelined j=X
    // iter=Y`. The profile contains only measured kernel work; compile
    // + capture + warmup costs are excluded.
    //
    // Env vars:
    //   - `FRAC_ROUNDS` — comma-separated `j` values (default `"4,10,16,20,24"`).
    //   - `NSYS_ENABLED=1` — flip on the profiler + NVTX calls.
    //   - `CC_FUSION=v2` — v2 fusion (v1 is deprecated).
    //
    // Recommended nsys invocation:
    //   NSYS_ENABLED=1 CC_FUSION=v2 FRAC_ROUNDS=4,10,16,20,24 \
    //     nsys profile --capture-range=cudaProfilerApi \
    //       --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    //       --trace=cuda,nvtx -o pipelined_sweep \
    //     cargo nextest run -p openvm-cuda-backend --features graph-ir \
    //       --run-ignored all --no-capture \
    //       -E 'test(bench_pipelined_ir_sweep)'
    #[test]
    #[ignore = "benchmark; run explicitly with --run-ignored"]
    fn bench_pipelined_ir_sweep() {
        use std::time::Instant;

        use crypto_compiler::graph_exe::GraphExe;

        use super::super::frac_bench_utils::cc_compiler;

        const WARMUP: usize = 2;
        const ITERS: usize = 3;

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);

        // Parse env: comma-separated round values, default `4,10,16,20,24`.
        let js: Vec<usize> = std::env::var("FRAC_ROUNDS")
            .unwrap_or_else(|_| "4,10,16,20,24".into())
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .expect("FRAC_ROUNDS entry must be a non-negative integer")
            })
            .collect();
        let nsys_enabled = std::env::var_os("NSYS_ENABLED").is_some();

        // Per-j state carried from setup pass into the profiler window.
        // `d_layer_snapshot` holds the pre-round pq buffer on-device; each
        // timed iter starts by copying it into the exe's `layer_work`
        // pool slot via `set_input(0, ...)`, so the reset D2D happens
        // OUTSIDE the captured graph (and outside the NVTX range).
        struct PerRound {
            j: usize,
            exe: GraphExe,
            n_nodes: usize,
            build_ms: f64,
            compile_ms: f64,
            capture_ms: f64,
            d_layer_snapshot: openvm_cuda_common::d_buffer::DeviceBuffer<u8>,
        }
        let mut states: Vec<PerRound> = Vec::with_capacity(js.len());

        // ---- Setup pass: build/compile/capture/warmup outside profiler ---
        for &j in &js {
            let n = 2usize << j; // pq_size at round start.
            println!(
                "\n=== [setup] j = {j}, n = 2 << j = {n} elements \
                 (w = {PIPELINE_WINDOW}, α = {PIPELINE_ALPHA}) ==="
            );

            // Deterministic per-j input (so the same seed reproduces
            // the same graph across sweeps — good for kernel-cache
            // hits in a rerun).
            let (leaves, xi_prev, prev_s_eval, eq_r_acc, lambda, alpha, _) =
                random_case_with_round(j, 0xACE1_5EED_u64 ^ j as u64);

            // Build.
            let t_build = Instant::now();
            let mut g = GraphBuilder::new();
            let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
            // Register `layer_buf` directly as the graph input (no
            // in-graph memcpy) — the pipelined driver mutates it in
            // place. Between timed iters the harness calls
            // `exe.set_input(0, &d_leaves)` to reset the layer,
            // which issues a D2D on the ctx stream OUTSIDE the
            // captured graph (and outside the NVTX range), so the
            // profile's `pipelined j=X iter=Y` bar measures only
            // kernel + graph-scheduler work — no 1 GB memcpy inside
            // the graph the way an `insert_memcpy(input_buf,
            // layer_buf)` node would produce.
            let layer_buf = add_frac_ef_buf(&mut g, device, "layer_work", n);
            g.register_input(layer_buf);
            // Every scalar becomes its own registered input so no
            // per-launch H2D is emitted inside the captured graph.
            let xi_bufs: Vec<crypto_compiler::graph_ir::BufId> = xi_prev
                .iter()
                .enumerate()
                .map(|(idx, _)| {
                    let b = add_ext_scalar_buf(&mut g, device, &format!("xi_{idx}_in"));
                    g.register_input(b);
                    b
                })
                .collect();
            let lambda_buf = add_ext_scalar_buf(&mut g, device, "lambda_in");
            g.register_input(lambda_buf);
            let prev_buf = add_ext_scalar_buf(&mut g, device, "prev_s_eval_in");
            g.register_input(prev_buf);
            let eqacc_buf = add_ext_scalar_buf(&mut g, device, "eq_r_acc_in");
            g.register_input(eqacc_buf);
            let seed_buf = add_ext_scalar_buf(&mut g, device, "seed_in");
            g.register_input(seed_buf);

            let round_input = RoundInputIR {
                layer: layer_buf,
                layer_len: n,
                prev_s_eval: prev_buf,
                eq_r_acc: eqacc_buf,
                lambda: lambda_buf,
                seed: seed_buf,
                alpha,
                real_len: n,
                logical_len: n,
                round: j,
            };
            let out = if j >= 2 + PIPELINE_WINDOW {
                fractional_sumcheck_round_pipelined_ir(
                    &mut g,
                    &mut transcript,
                    round_input,
                    &xi_bufs,
                    device,
                )
            } else {
                fractional_sumcheck_round_foldeval_ir(
                    &mut g,
                    &mut transcript,
                    round_input,
                    &xi_bufs,
                    device,
                )
            };
            // Register every proof artifact BufId directly — no memcpy layer.
            for s in &out.round_polys {
                for &b in s.iter() {
                    g.register_output(b);
                }
            }
            for &b in &out.r_vec {
                g.register_output(b);
            }
            for b in [out.p_xi_0, out.q_xi_0, out.p_xi_1, out.q_xi_1] {
                g.register_output(b);
            }
            let n_nodes = g.nodes.len();
            let build_ms = t_build.elapsed().as_secs_f64() * 1e3;

            // Compile.
            let t_compile = Instant::now();
            let mut exe: GraphExe = cc_compiler(device).compile(g).expect("graph compile");
            let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
            if let Some(v2) = exe.fusion_report().and_then(|r| r.v2.as_ref()) {
                println!(
                    "[setup j={j}] fusion v2: nodes {} -> {}, inserted={}, selected={}, \
                     fallback={:?}",
                    v2.nodes_before,
                    v2.nodes_after,
                    v2.candidates_inserted,
                    v2.selected_from_solver,
                    v2.fallback_reason,
                );
            }
            println!(
                "[setup j={j}] build {build_ms:>8.2} ms ({n_nodes} nodes); \
                 compile {compile_ms:>8.2} ms ({} unique modules, {} from cache)",
                exe.num_unique_modules(),
                exe.num_cached_modules(),
            );

            // Set inputs (once, outside profiler window). Order: leaves,
            // xi_0..xi_{j-1}, lambda, prev_s_eval, eq_r_acc, seed.
            let d_leaves = frac_bytes(&leaves).to_device_on(&ctx).expect("H2D leaves");
            exe.set_input(&ctx, 0, &d_leaves).expect("set_input leaves");
            let ef_to_dev = |v: EF| -> openvm_cuda_common::d_buffer::DeviceBuffer<u8> {
                let bytes: Vec<u8> = unsafe {
                    std::slice::from_raw_parts(
                        &v as *const EF as *const u8,
                        std::mem::size_of::<EF>(),
                    )
                    .to_vec()
                };
                bytes.as_slice().to_device_on(&ctx).expect("H2D EF")
            };
            let d_xis: Vec<_> = xi_prev.iter().copied().map(ef_to_dev).collect();
            for (idx, d) in d_xis.iter().enumerate() {
                exe.set_input(&ctx, 1 + idx, d).expect("set_input xi");
            }
            let d_lambda = ef_to_dev(lambda);
            let d_prev = ef_to_dev(prev_s_eval);
            let d_eqacc = ef_to_dev(eq_r_acc);
            let d_seed = ef_to_dev(EF::ONE);
            exe.set_input(&ctx, 1 + j, &d_lambda)
                .expect("set_input lambda");
            exe.set_input(&ctx, 2 + j, &d_prev)
                .expect("set_input prev_s_eval");
            exe.set_input(&ctx, 3 + j, &d_eqacc)
                .expect("set_input eq_r_acc");
            exe.set_input(&ctx, 4 + j, &d_seed).expect("set_input seed");
            ctx.stream.synchronize().expect("sync set_input");

            // Warmup (both to prime driver-side setup + to fault in
            // the pool allocation).
            for _ in 0..WARMUP {
                exe.run(&ctx).expect("warmup run");
            }
            ctx.stream.synchronize().expect("sync post-warmup");

            // Capture CUDA graph so timed iters are pure `cudaGraphLaunch` replays.
            let t_cap = Instant::now();
            exe.capture_graph(&ctx).expect("capture_graph");
            exe.launch_graph(&ctx).expect("capture warmup launch");
            ctx.stream.synchronize().expect("sync post-capture");
            let capture_ms = t_cap.elapsed().as_secs_f64() * 1e3;
            println!("[setup j={j}] capture {capture_ms:>8.2} ms");

            states.push(PerRound {
                j,
                exe,
                n_nodes,
                build_ms,
                compile_ms,
                capture_ms,
                d_layer_snapshot: d_leaves,
            });
        }

        // ---- Timed pass inside cudaProfilerStart/Stop ------------------
        // One profiler window across the whole sweep; each j gets ITERS
        // NVTX ranges of the form `pipelined j=X iter=Y`.
        //
        // Per-iter shape (in order):
        //   1. `set_input(0, d_layer_snapshot)` — D2D reset of the exe's `layer_work` pool slot.
        //      Runs on the ctx stream; not part of the captured graph, so no memcpy node lands
        //      inside `pipelined j=X iter=Y`.
        //   2. `device_synchronize()` — ensures the reset D2D completes before the NVTX range
        //      opens, so kernel timing inside the range is not skewed by an overlapping copy.
        //   3. NVTX push → `launch_graph` → `device_synchronize` → NVTX pop.
        //
        // `device_synchronize` (instead of `ctx.stream.synchronize`) forces
        // the CPU to wait for ALL streams (the graph uses 8 streams via
        // the v2 planner), so the NVTX range's end timestamp reflects the
        // last kernel's completion on any stream — not just the ctx
        // stream's tail.
        use openvm_cuda_common::stream::device_synchronize;
        let mut timings: Vec<(usize, Vec<f64>)> = Vec::with_capacity(states.len());
        if nsys_enabled {
            unsafe { cudaProfilerStart() };
        }
        for st in states.iter_mut() {
            let mut ms: Vec<f64> = Vec::with_capacity(ITERS);
            for i in 0..ITERS {
                // Reset the exe's layer_work pool slot from the snapshot
                // BEFORE the NVTX range opens — the D2D lands outside the
                // captured graph and outside the timed section.
                st.exe
                    .set_input(&ctx, 0, &st.d_layer_snapshot)
                    .expect("set_input reset");
                device_synchronize().expect("device sync pre-launch");
                let t0 = Instant::now();
                if nsys_enabled {
                    nvtx::range_push!("pipelined j={} iter={}", st.j, i);
                }
                st.exe.launch_graph(&ctx).expect("launch_graph iter");
                device_synchronize().expect("device sync post-launch");
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            timings.push((st.j, ms));
        }
        if nsys_enabled {
            unsafe { cudaProfilerStop() };
        }

        // ---- Report ---------------------------------------------------
        println!("\n=== pipelined single-round sweep ===");
        println!(
            "{:>4}  {:>12}  {:>10}  {:>10}  {:>10}  {}",
            "j", "n=2<<j", "build ms", "compile ms", "capture ms", "iter ms"
        );
        for (st, (j, ms)) in states.iter().zip(timings.iter()) {
            let mean = ms.iter().sum::<f64>() / ms.len() as f64;
            let n = 2usize << st.j;
            debug_assert_eq!(*j, st.j);
            println!(
                "{:>4}  {:>12}  {:>10.2}  {:>10.2}  {:>10.2}  {ms:>8.3?} (mean {mean:.3} ms, \
                 {} nodes)",
                st.j, n, st.build_ms, st.compile_ms, st.capture_ms, st.n_nodes,
            );
        }
    }

    // ---- Full-sumcheck sweep: eager + default IR + pipelined ---------------
    //
    // For each `log_n` in `FRAC_LOG_N` (default `16,20,24`), build three
    // driver states side-by-side:
    //   - eager  = `fractional_sumcheck_gpu` (baseline)
    //   - ir     = `fractional_sumcheck_gpu_ir` (the default graph-IR driver, no pipelining)
    //   - pipe   = `fractional_sumcheck_gpu_pipelined_ir` (α-tiled pipelined driver)
    //
    // Setup (build, compile, `capture_graph`, warmup, correctness check
    // against the eager reference) runs BEFORE `cudaProfilerStart` for
    // every log_n × driver combination. Inside a single
    // `cudaProfilerStart / Stop` window, each iter is bracketed by its
    // own NVTX range — `{eager,ir,pipe} log_n=X iter=Y`. Between iters
    // the per-driver device-side leaves are reset from a device-side
    // snapshot via a D2D copy on the ctx stream, OUTSIDE the NVTX range
    // (so no memcpy shows up inside the timed section). Every iter is
    // fenced with `cudaDeviceSynchronize` (not per-stream sync) so the
    // NVTX range's end timestamp reflects the last kernel completion
    // across every stream the v2 planner used.
    //
    // Env vars:
    //   - `FRAC_LOG_N` — comma-separated log2(leaf count) (default `16,20,24`).
    //   - `NSYS_ENABLED=1` — enable the profiler + NVTX calls.
    //   - `CC_FUSION=v2` — v2 fusion (v1 deprecated).
    //   - `CC_SCHEDULER={list_v1,list_v2}` — flow through cc_scheduler_mode.
    //
    // Recommended nsys invocation:
    //   NSYS_ENABLED=1 CC_FUSION=v2 FRAC_LOG_N=16,20,24 \
    //     nsys profile --capture-range=cudaProfilerApi \
    //       --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    //       --trace=cuda,nvtx -o pipelined_full_sweep \
    //     cargo nextest run -p openvm-cuda-backend --features graph-ir \
    //       --run-ignored all --no-capture \
    //       -E 'test(bench_pipelined_full_sumcheck_sweep)'
    #[test]
    #[ignore = "benchmark; run explicitly with --run-ignored"]
    fn bench_pipelined_full_sumcheck_sweep() {
        use std::time::Instant;

        use crypto_compiler::graph_exe::GraphExe;
        use openvm_cuda_common::{d_buffer::DeviceBuffer, memory_manager::MemTracker};
        use openvm_stark_backend::prover::fractional_sumcheck_gkr::FracSumcheckProof;
        use p3_util::log2_strict_usize;

        use super::super::{
            frac_bench_utils::{cc_compiler, frac_log_ns},
            fractional::fractional_sumcheck_gpu,
            fractional_ir::{fractional_sumcheck_gpu_ir, FracSumcheckProofIR},
            FractionalInputSize,
        };
        use crate::logup_zerocheck::fractional_ir_pipelined::fractional_sumcheck_gpu_pipelined_ir;

        const WARMUP: usize = 2;
        const ITERS: usize = 3;

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let log_ns: Vec<usize> = frac_log_ns("16,20,24");
        let nsys_enabled = std::env::var_os("NSYS_ENABLED").is_some();
        use openvm_cuda_common::stream::device_synchronize;

        // Per-log_n state: three exes (eager reference proof lives on
        // host; ir + pipelined are graph-compiled), plus a device-side
        // leaves snapshot used to reset each driver's working buffer
        // between iters (outside NVTX ranges, on ctx stream).
        #[allow(dead_code)]
        struct PerCase {
            log_n: usize,
            n: usize,
            sizes: FractionalInputSize,
            leaves_host: Vec<Frac<EF>>,
            alpha: EF,
            // Device snapshot of the leaves — persistent, one H2D at setup.
            // Used to feed both IR exes' `layer_work` slot each iter, and
            // to cheaply produce a fresh working buffer for the eager
            // driver via D2D (host `leaves` are 1 GB at log_n=24; the
            // per-iter H2D would otherwise cost ~15-20 ms each).
            d_leaves_snapshot_bytes: DeviceBuffer<u8>,
            // Default graph-IR driver.
            ir_exe: GraphExe,
            ir_build_ms: f64,
            ir_compile_ms: f64,
            ir_capture_ms: f64,
            ir_n_nodes: usize,
            // Pipelined driver.
            pipe_exe: GraphExe,
            pipe_build_ms: f64,
            pipe_compile_ms: f64,
            pipe_capture_ms: f64,
            pipe_n_nodes: usize,
            // Eager reference proof (for correctness gating outside profiler).
            eager_proof: FracSumcheckProof<SC>,
            eager_xi: Vec<EF>,
        }

        let mut cases: Vec<PerCase> = Vec::with_capacity(log_ns.len());

        for &log_n in &log_ns {
            let n = 1usize << log_n;
            let sizes = FractionalInputSize::new(n, n);
            let leaves = make_host_leaves(n, 0x5EED_F0F0_u64 ^ log_n as u64);
            let mut rng = StdRng::seed_from_u64(0xA1FA ^ log_n as u64);
            let alpha: EF = rng.random();
            println!(
                "\n=== [setup] full sumcheck: n = 2^{log_n} = {n} leaves ({} MiB Frac<EF>) ===",
                (n * std::mem::size_of::<Frac<EF>>()) >> 20,
            );

            // ---- Eager reference (also warms up eager code paths). ----
            let d_leaves_ref: DeviceBuffer<Frac<EF>> =
                leaves.as_slice().to_device_on(&ctx).expect("H2D eager ref");
            let mut sponge_ref = DuplexSpongeGpu::default();
            let mut mem_ref = MemTracker::start("bench.full_sweep_eager_ref");
            let t_e = Instant::now();
            let (eager_proof, eager_xi): (FracSumcheckProof<SC>, Vec<EF>) =
                fractional_sumcheck_gpu::<SC, _>(
                    &mut sponge_ref,
                    d_leaves_ref,
                    sizes,
                    alpha,
                    false,
                    &mut mem_ref,
                    &ctx,
                )
                .expect("eager reference");
            device_synchronize().expect("sync eager ref");
            println!(
                "[setup log_n={log_n}] eager reference: {:>8.2} ms",
                t_e.elapsed().as_secs_f64() * 1e3,
            );

            // Persistent device snapshot for D2D resets in the timed loop.
            let d_leaves_snapshot_bytes: DeviceBuffer<u8> = frac_bytes(&leaves)
                .to_device_on(&ctx)
                .expect("H2D snapshot");
            device_synchronize().expect("sync snapshot");

            // ---- Default IR driver -----------------------------------
            let t_build = Instant::now();
            let mut g = GraphBuilder::new();
            let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
            // Register `layer_work` directly as the graph input. No
            // in-graph `insert_memcpy(input_buf, layer_buf)` — the
            // harness resets `layer_work` via `set_input` between
            // iters, outside the NVTX range. Same pattern as the
            // pipelined path below.
            let ir_layer = add_frac_ef_buf(&mut g, device, "ir_layer_work", n);
            g.register_input(ir_layer);
            let ir_proof: FracSumcheckProofIR = fractional_sumcheck_gpu_ir(
                &mut g,
                &mut transcript,
                ir_layer,
                sizes,
                alpha,
                /* assert_zero */ false,
                device,
            )
            .expect("fractional_sumcheck_gpu_ir");
            let ir_output_bids = register_proof_artifacts(&mut g, &ir_proof);
            let ir_n_nodes = g.nodes.len();
            let ir_build_ms = t_build.elapsed().as_secs_f64() * 1e3;
            let t_compile = Instant::now();
            let mut ir_exe: GraphExe = cc_compiler(device).compile(g).expect("compile ir");
            let ir_compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
            if let Some(v2) = ir_exe.fusion_report().and_then(|r| r.v2.as_ref()) {
                println!(
                    "[setup log_n={log_n} ir] fusion v2: {} -> {} nodes, selected={}",
                    v2.nodes_before, v2.nodes_after, v2.selected_from_solver,
                );
            }
            println!(
                "[setup log_n={log_n} ir] build {ir_build_ms:>8.2} ms ({ir_n_nodes} nodes); \
                 compile {ir_compile_ms:>8.2} ms",
            );
            assert_eq!(ir_exe.num_inputs(), 1, "ir_exe should have leaves_in only");
            // Reset layer_work BEFORE every warmup / correctness / capture
            // launch — the graph mutates it in place, so any run past
            // the first would otherwise read modified state and produce
            // a stale proof.
            for _ in 0..WARMUP {
                ir_exe
                    .set_input(&ctx, 0, &d_leaves_snapshot_bytes)
                    .expect("ir warmup reset");
                device_synchronize().expect("sync ir warmup reset");
                ir_exe.run(&ctx).expect("ir warmup run");
                device_synchronize().expect("sync ir warmup");
            }
            // Correctness check: fresh input, run, read outputs.
            ir_exe
                .set_input(&ctx, 0, &d_leaves_snapshot_bytes)
                .expect("ir correctness reset");
            device_synchronize().expect("sync ir correctness reset");
            ir_exe.run(&ctx).expect("ir correctness run");
            device_synchronize().expect("sync ir correctness");
            check_proof_matches_eager(
                &ir_exe,
                &ir_proof,
                &ir_output_bids,
                &eager_proof,
                &eager_xi,
                &ctx,
                &format!("ir log_n={log_n}"),
            );
            // Capture — the capture_graph call itself executes the graph
            // once (stream-capture mode), so reset first.
            ir_exe
                .set_input(&ctx, 0, &d_leaves_snapshot_bytes)
                .expect("ir capture reset");
            device_synchronize().expect("sync ir capture reset");
            let t_cap = Instant::now();
            ir_exe.capture_graph(&ctx).expect("capture ir");
            device_synchronize().expect("sync ir capture");
            let ir_capture_ms = t_cap.elapsed().as_secs_f64() * 1e3;
            // One post-capture launch to prime driver setup.
            ir_exe
                .set_input(&ctx, 0, &d_leaves_snapshot_bytes)
                .expect("ir capture warmup reset");
            device_synchronize().expect("sync ir capture warmup reset");
            ir_exe.launch_graph(&ctx).expect("ir capture warmup launch");
            device_synchronize().expect("sync ir capture warmup");
            println!("[setup log_n={log_n} ir] capture {ir_capture_ms:>8.2} ms");

            // ---- Pipelined driver ------------------------------------
            let t_build = Instant::now();
            let mut g = GraphBuilder::new();
            let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
            let pipe_layer = add_frac_ef_buf(&mut g, device, "pipe_layer_work", n);
            g.register_input(pipe_layer);
            let pipe_seed = add_ext_scalar_buf(&mut g, device, "pipe_seed_in");
            g.register_input(pipe_seed);
            let pipe_proof: FracSumcheckProofIR = fractional_sumcheck_gpu_pipelined_ir(
                &mut g,
                &mut transcript,
                pipe_layer,
                sizes,
                alpha,
                /* assert_zero */ false,
                pipe_seed,
                device,
            )
            .expect("fractional_sumcheck_gpu_pipelined_ir");
            let pipe_output_bids = register_proof_artifacts(&mut g, &pipe_proof);
            let pipe_n_nodes = g.nodes.len();
            let pipe_build_ms = t_build.elapsed().as_secs_f64() * 1e3;
            let t_compile = Instant::now();
            let mut pipe_exe: GraphExe = cc_compiler(device).compile(g).expect("compile pipe");
            let pipe_compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
            if let Some(v2) = pipe_exe.fusion_report().and_then(|r| r.v2.as_ref()) {
                println!(
                    "[setup log_n={log_n} pipe] fusion v2: {} -> {} nodes, selected={}",
                    v2.nodes_before, v2.nodes_after, v2.selected_from_solver,
                );
            }
            println!(
                "[setup log_n={log_n} pipe] build {pipe_build_ms:>8.2} ms ({pipe_n_nodes} nodes); \
                 compile {pipe_compile_ms:>8.2} ms",
            );
            assert_eq!(
                pipe_exe.num_inputs(),
                2,
                "pipe_exe should have [layer, seed]"
            );
            let seed_bytes: Vec<u8> = unsafe {
                let one = EF::ONE;
                std::slice::from_raw_parts(
                    &one as *const EF as *const u8,
                    std::mem::size_of::<EF>(),
                )
                .to_vec()
            };
            let d_seed = seed_bytes.as_slice().to_device_on(&ctx).expect("H2D seed");
            // seed is read-only within the graph (const), set once.
            pipe_exe
                .set_input(&ctx, 1, &d_seed)
                .expect("set_input pipe seed");
            device_synchronize().expect("sync pipe seed");
            for _ in 0..WARMUP {
                pipe_exe
                    .set_input(&ctx, 0, &d_leaves_snapshot_bytes)
                    .expect("pipe warmup reset");
                device_synchronize().expect("sync pipe warmup reset");
                pipe_exe.run(&ctx).expect("pipe warmup run");
                device_synchronize().expect("sync pipe warmup");
            }
            pipe_exe
                .set_input(&ctx, 0, &d_leaves_snapshot_bytes)
                .expect("pipe correctness reset");
            device_synchronize().expect("sync pipe correctness reset");
            pipe_exe.run(&ctx).expect("pipe correctness run");
            device_synchronize().expect("sync pipe correctness");
            check_proof_matches_eager(
                &pipe_exe,
                &pipe_proof,
                &pipe_output_bids,
                &eager_proof,
                &eager_xi,
                &ctx,
                &format!("pipe log_n={log_n}"),
            );
            pipe_exe
                .set_input(&ctx, 0, &d_leaves_snapshot_bytes)
                .expect("pipe capture reset");
            device_synchronize().expect("sync pipe capture reset");
            let t_cap = Instant::now();
            pipe_exe.capture_graph(&ctx).expect("capture pipe");
            device_synchronize().expect("sync pipe capture");
            let pipe_capture_ms = t_cap.elapsed().as_secs_f64() * 1e3;
            pipe_exe
                .set_input(&ctx, 0, &d_leaves_snapshot_bytes)
                .expect("pipe capture warmup reset");
            device_synchronize().expect("sync pipe capture warmup reset");
            pipe_exe
                .launch_graph(&ctx)
                .expect("pipe capture warmup launch");
            device_synchronize().expect("sync pipe capture warmup");
            println!("[setup log_n={log_n} pipe] capture {pipe_capture_ms:>8.2} ms");

            cases.push(PerCase {
                log_n,
                n,
                sizes,
                leaves_host: leaves,
                alpha,
                d_leaves_snapshot_bytes,
                ir_exe,
                ir_build_ms,
                ir_compile_ms,
                ir_capture_ms,
                ir_n_nodes,
                pipe_exe,
                pipe_build_ms,
                pipe_compile_ms,
                pipe_capture_ms,
                pipe_n_nodes,
                eager_proof,
                eager_xi,
            });
        }

        // ---- Timed pass inside cudaProfilerStart/Stop ------------------
        // 3 drivers × N log_n × ITERS iters, each in its own NVTX range.
        // Between iters we run a D2D reset (for IR + pipe) or a fresh H2D
        // clone (for eager) OUTSIDE the NVTX range, followed by
        // `cudaDeviceSynchronize` so the range starts cleanly.
        let mut timings: Vec<(usize, Vec<f64>, Vec<f64>, Vec<f64>)> =
            Vec::with_capacity(cases.len());
        if nsys_enabled {
            unsafe { cudaProfilerStart() };
        }
        for st in cases.iter_mut() {
            let mut eager_ms: Vec<f64> = Vec::with_capacity(ITERS);
            let mut ir_ms: Vec<f64> = Vec::with_capacity(ITERS);
            let mut pipe_ms: Vec<f64> = Vec::with_capacity(ITERS);

            // Eager iters — a fresh working DeviceBuffer<Frac<EF>> per iter
            // (the eager driver moves it in). We produce it by cloning the
            // host `leaves_host` via `to_device_on` OUTSIDE the NVTX
            // range; sponge + MemTracker init are also outside.
            for i in 0..ITERS {
                let d_leaves: DeviceBuffer<Frac<EF>> = st
                    .leaves_host
                    .as_slice()
                    .to_device_on(&ctx)
                    .expect("H2D eager iter leaves");
                let mut sponge = DuplexSpongeGpu::default();
                let mut mem = MemTracker::start("bench.full_sweep_eager");
                device_synchronize().expect("sync pre-eager");

                let t0 = Instant::now();
                if nsys_enabled {
                    nvtx::range_push!("eager log_n={} iter={}", st.log_n, i);
                }
                let _ = fractional_sumcheck_gpu::<SC, _>(
                    &mut sponge,
                    d_leaves,
                    st.sizes,
                    st.alpha,
                    false,
                    &mut mem,
                    &ctx,
                )
                .expect("eager iter");
                device_synchronize().expect("sync post-eager");
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                eager_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }

            // Default IR iters — reset layer_work via set_input(0, snapshot).
            for i in 0..ITERS {
                st.ir_exe
                    .set_input(&ctx, 0, &st.d_leaves_snapshot_bytes)
                    .expect("ir reset");
                device_synchronize().expect("sync pre-ir");

                let t0 = Instant::now();
                if nsys_enabled {
                    nvtx::range_push!("ir log_n={} iter={}", st.log_n, i);
                }
                st.ir_exe.launch_graph(&ctx).expect("ir launch");
                device_synchronize().expect("sync post-ir");
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                ir_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }

            // Pipelined iters — reset layer_work via set_input(0, snapshot).
            for i in 0..ITERS {
                st.pipe_exe
                    .set_input(&ctx, 0, &st.d_leaves_snapshot_bytes)
                    .expect("pipe reset");
                device_synchronize().expect("sync pre-pipe");

                let t0 = Instant::now();
                if nsys_enabled {
                    nvtx::range_push!("pipe log_n={} iter={}", st.log_n, i);
                }
                st.pipe_exe.launch_graph(&ctx).expect("pipe launch");
                device_synchronize().expect("sync post-pipe");
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                pipe_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }

            timings.push((st.log_n, eager_ms, ir_ms, pipe_ms));
        }
        if nsys_enabled {
            unsafe { cudaProfilerStop() };
        }

        // ---- Report ---------------------------------------------------
        println!("\n=== full-sumcheck sweep (mean ms across {ITERS} iters) ===");
        println!(
            "{:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
            "log_n", "n", "eager", "ir", "pipe", "pipe/eager",
        );
        for (log_n, eager, ir, pipe) in &timings {
            let em = eager.iter().sum::<f64>() / eager.len() as f64;
            let im = ir.iter().sum::<f64>() / ir.len() as f64;
            let pm = pipe.iter().sum::<f64>() / pipe.len() as f64;
            let n = 1usize << *log_n;
            println!(
                "{log_n:>6}  {n:>8}  {em:>8.3}  {im:>8.3}  {pm:>8.3}  {:>8.3}x",
                pm / em,
            );
        }
        let _ = log2_strict_usize;
    }

    /// Register every proof-artifact BufId in `proof_ir` as a graph
    /// output. No `insert_memcpy` between the driver's writers and the
    /// output slots — the graph runtime keeps registered BufIds live
    /// at their producing kernels' output slots. Order matches the
    /// reshape used by the correctness check.
    fn register_proof_artifacts(
        g: &mut GraphBuilder,
        proof: &crate::logup_zerocheck::fractional_ir::FracSumcheckProofIR,
    ) -> Vec<crypto_compiler::graph_ir::BufId> {
        let mut out: Vec<crypto_compiler::graph_ir::BufId> = Vec::new();
        let (rp, rq) = proof.fractional_sum;
        g.register_output(rp);
        out.push(rp);
        g.register_output(rq);
        out.push(rq);
        for claim in &proof.claims_per_layer {
            for b in claim.as_array() {
                g.register_output(b);
                out.push(b);
            }
        }
        for round_polys in &proof.sumcheck_polys {
            for s in round_polys {
                for &b in s.iter() {
                    g.register_output(b);
                    out.push(b);
                }
            }
        }
        for &b in &proof.final_randomness {
            g.register_output(b);
            out.push(b);
        }
        out
    }

    /// Read every registered proof-artifact BufId back from the exe and
    /// compare artifact-by-artifact against the eager reference. Panics
    /// with a labelled context on mismatch.
    fn check_proof_matches_eager(
        exe: &crypto_compiler::graph_exe::GraphExe,
        proof: &crate::logup_zerocheck::fractional_ir::FracSumcheckProofIR,
        _output_bids: &[crypto_compiler::graph_ir::BufId],
        eager_proof: &openvm_stark_backend::prover::fractional_sumcheck_gkr::FracSumcheckProof<SC>,
        eager_xi: &[EF],
        ctx: &GpuDeviceCtx,
        label: &str,
    ) {
        let read_ef = |bid: crypto_compiler::graph_ir::BufId| -> EF {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == bid)
                .expect("registered output BufId");
            ef_from_bytes(&exe.get_output(idx).to_host_on(ctx).expect("D2H"))
        };
        let got_sum = (
            read_ef(proof.fractional_sum.0),
            read_ef(proof.fractional_sum.1),
        );
        assert_eq!(
            got_sum, eager_proof.fractional_sum,
            "[{label}] fractional_sum mismatch"
        );
        assert_eq!(
            proof.claims_per_layer.len(),
            eager_proof.claims_per_layer.len(),
            "[{label}] claims layer count mismatch",
        );
        for (i, (got, want)) in proof
            .claims_per_layer
            .iter()
            .zip(&eager_proof.claims_per_layer)
            .enumerate()
        {
            let got_arr: [EF; 4] = std::array::from_fn(|k| read_ef(got.as_array()[k]));
            assert_eq!(
                got_arr,
                [want.p_xi_0, want.q_xi_0, want.p_xi_1, want.q_xi_1],
                "[{label}] layer {i} claims mismatch",
            );
        }
        let got_polys: Vec<Vec<[EF; GKR_S_DEG]>> = proof
            .sumcheck_polys
            .iter()
            .map(|rp| {
                rp.iter()
                    .map(|s| std::array::from_fn(|k| read_ef(s[k])))
                    .collect()
            })
            .collect();
        assert_eq!(
            got_polys, eager_proof.sumcheck_polys,
            "[{label}] sumcheck_polys mismatch"
        );
        let got_xi: Vec<EF> = proof.final_randomness.iter().map(|&b| read_ef(b)).collect();
        assert_eq!(
            got_xi.as_slice(),
            eager_xi,
            "[{label}] final_randomness mismatch"
        );
        println!(
            "[{label}] correctness OK ({} claim layers, {} sumcheck polys, {} final xi)",
            proof.claims_per_layer.len(),
            proof.sumcheck_polys.iter().map(|r| r.len()).sum::<usize>(),
            proof.final_randomness.len(),
        );
    }
}
