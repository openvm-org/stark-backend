use std::{array::from_fn, convert::TryInto, env, ffi::c_void, mem::transmute};

use openvm_cuda_common::{
    copy::{cuda_memcpy, MemCopyD2H},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use openvm_stark_backend::{
    poly_common::{eval_eq_mle, interpolate_linear_at_01, interpolate_quadratic_at_012},
    proof::GkrLayerClaims,
    prover::fractional_sumcheck_gkr::{Frac, FracSumcheckProof},
    FiatShamirTranscript,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_util::log2_strict_usize;
use tracing::{debug_span, instrument};

use super::errors::FractionalSumcheckError;
use crate::{
    cuda::{
        logup_zerocheck::{
            _frac_compute_round_temp_buffer_size, fold_ef_frac_columns,
            fold_ef_frac_columns_inplace, frac_build_tree_layer, frac_compute_round,
            frac_compute_round_and_fold, frac_compute_round_and_fold_inplace,
            frac_compute_round_and_revert, frac_multifold_raw, frac_precompute_m_build_raw,
            frac_precompute_m_eval_round_raw,
        },
        ntt::bit_rev_frac_ext,
    },
    poly::SqrtEqLayers,
    prelude::{EF, SC},
    sponge::DuplexSpongeGpu,
};

const GKR_S_DEG: usize = 3;
const GKR_WINDOW_SIZE: usize = 3;
const GKR_WINDOW_DEFAULT_MIN_N: usize = 22;

/// Describes which buffer operation to use for the next fused compute+fold round.
#[derive(Debug, Clone, Copy)]
enum BufferTarget {
    /// Out-of-place: layer → work_buffer
    LayerToWork,
    /// Out-of-place: work_buffer → layer
    WorkToLayer,
    /// In-place on layer buffer
    InPlaceLayer,
    /// In-place on work_buffer
    InPlaceWork,
}

/// Encapsulates ping-pong buffer scheduling state for GKR inner rounds.
///
/// This struct manages the decision of whether to use in-place or out-of-place
/// (ping-pong) kernel variants based on buffer capacities and current data location.
struct BufferScheduler {
    /// True if data currently resides in work_buffer, false if in layer.
    data_in_work_buffer: bool,
    /// Maximum capacity of work_buffer in elements.
    work_buffer_cap: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GkrRoundStrategy {
    FoldEval,
    PrecomputeM,
}

const PRECOMPUTE_M_TAIL_TILE: usize = 4096;
const PRECOMPUTE_M_MIN_TAIL_TILE: usize = 256;
const PRECOMPUTE_M_DEFAULT_MIN_BLOCKS: usize = 64;
const PRECOMPUTE_M_DEFAULT_TARGET_BLOCKS: usize = 1024;

impl BufferScheduler {
    /// Creates a new scheduler with data initially in layer buffer.
    fn new(work_buffer_cap: usize) -> Self {
        Self {
            data_in_work_buffer: false,
            work_buffer_cap,
        }
    }

    /// Returns true if we can use ping-pong (out-of-place) for the given post-fold size.
    fn can_pingpong(&self, post_fold_size: usize) -> bool {
        post_fold_size <= self.work_buffer_cap
    }

    /// Determines the next buffer target for a fused compute+fold operation.
    ///
    /// For last outer round: uses ping-pong when possible for __restrict__ optimization.
    /// For non-last outer rounds: preserves layer for tree revert operations.
    fn next_target(&mut self, post_fold_size: usize, last_outer_round: bool) -> BufferTarget {
        let can_pingpong = self.can_pingpong(post_fold_size);

        if last_outer_round {
            if can_pingpong {
                // Ping-pong to other buffer
                if self.data_in_work_buffer {
                    self.data_in_work_buffer = false;
                    BufferTarget::WorkToLayer
                } else {
                    self.data_in_work_buffer = true;
                    BufferTarget::LayerToWork
                }
            } else {
                // In-place on layer (data must be in layer for early rounds of last outer round)
                debug_assert!(
                    !self.data_in_work_buffer,
                    "in-place path requires data in layer"
                );
                BufferTarget::InPlaceLayer
            }
        } else {
            // Non-last outer round: preserve layer for tree revert
            if self.data_in_work_buffer {
                // Already in work_buffer, stay there in-place
                BufferTarget::InPlaceWork
            } else {
                // Data in layer, move to work_buffer (out-of-place)
                self.data_in_work_buffer = true;
                BufferTarget::LayerToWork
            }
        }
    }

    /// Returns the buffer target for the final fold (no fused compute).
    fn final_fold_target(&self, last_outer_round: bool) -> BufferTarget {
        if last_outer_round {
            if self.data_in_work_buffer {
                BufferTarget::InPlaceWork
            } else {
                BufferTarget::InPlaceLayer
            }
        } else {
            // Non-last outer round: fold into work_buffer to preserve layer
            if self.data_in_work_buffer {
                BufferTarget::InPlaceWork
            } else {
                BufferTarget::LayerToWork
            }
        }
    }
}

fn precompute_m_enabled() -> bool {
    !matches!(
        env::var("SWIRL_CUDA_GKR_PRECOMPUTE_M"),
        Ok(val) if matches!(val.as_str(), "0" | "false" | "FALSE" | "no" | "NO")
    )
}

fn precompute_m_min_blocks_threshold() -> usize {
    env::var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(PRECOMPUTE_M_DEFAULT_MIN_BLOCKS)
        .max(1)
}

fn precompute_m_num_tail_blocks(rem_n: usize, w: usize, tail_tile: usize) -> usize {
    let tail_n = rem_n - w;
    (1usize << tail_n).div_ceil(tail_tile)
}

fn precompute_m_target_blocks() -> usize {
    env::var("SWIRL_CUDA_GKR_PRECOMPUTE_M_TARGET_BLOCKS")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(PRECOMPUTE_M_DEFAULT_TARGET_BLOCKS)
        .max(1)
}

fn precompute_m_tail_tile_override() -> Option<usize> {
    env::var("SWIRL_CUDA_GKR_PRECOMPUTE_M_TAIL_TILE")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .map(|v| v.clamp(PRECOMPUTE_M_MIN_TAIL_TILE, PRECOMPUTE_M_TAIL_TILE))
}

fn precompute_m_min_n() -> usize {
    env::var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(GKR_WINDOW_DEFAULT_MIN_N)
}

fn precompute_m_build_tail_tile(
    rem_n: usize,
    w: usize,
    min_blocks_threshold: usize,
    target_blocks: usize,
    tail_tile_override: Option<usize>,
) -> usize {
    if let Some(tile) = tail_tile_override {
        return tile;
    }
    let tail_n = rem_n - w;
    let k = 1usize << tail_n;
    let target_blocks = target_blocks.max(min_blocks_threshold).max(1);
    let desired_tile = k.div_ceil(target_blocks).max(1);
    desired_tile.clamp(PRECOMPUTE_M_MIN_TAIL_TILE, PRECOMPUTE_M_TAIL_TILE)
}

fn choose_precompute_m_window_w(
    rem_n: usize,
    rounds_left: usize,
    min_blocks_threshold: usize,
    target_blocks: usize,
    tail_tile_override: Option<usize>,
    min_n: usize,
) -> Option<usize> {
    if rem_n < min_n || rounds_left < GKR_WINDOW_SIZE {
        return None;
    }
    let w = GKR_WINDOW_SIZE;
    let tail_tile = precompute_m_build_tail_tile(
        rem_n,
        w,
        min_blocks_threshold,
        target_blocks,
        tail_tile_override,
    );
    (precompute_m_num_tail_blocks(rem_n, w, tail_tile) >= min_blocks_threshold).then_some(w)
}

fn choose_round_strategy(
    round: usize,
    precompute_m_env: bool,
    precompute_m_min_blocks_threshold: usize,
    precompute_m_target_blocks: usize,
    precompute_m_tail_tile_override: Option<usize>,
    precompute_m_min_n: usize,
) -> GkrRoundStrategy {
    if !precompute_m_env {
        return GkrRoundStrategy::FoldEval;
    }
    let start_base = 1usize;
    let stop = round.div_ceil(2);
    let rem_n = round - start_base;
    let rounds_left = stop - start_base;
    if choose_precompute_m_window_w(
        rem_n,
        rounds_left,
        precompute_m_min_blocks_threshold,
        precompute_m_target_blocks,
        precompute_m_tail_tile_override,
        precompute_m_min_n,
    )
    .is_none()
    {
        return GkrRoundStrategy::FoldEval;
    }

    GkrRoundStrategy::PrecomputeM
}

fn eval_mle_table(points: &[EF], out: &mut [EF]) {
    // w <= 5 so CPU builds are trivial; avoid GPU kernel/alloc overhead for tiny tables.
    let n = points.len();
    let size = 1usize << n;
    debug_assert!(out.len() >= size);
    for (bits, dst) in out.iter_mut().enumerate().take(size) {
        let mut acc = EF::ONE;
        for (i, &x) in points.iter().enumerate() {
            let bit = ((bits >> (n - 1 - i)) & 1) == 1;
            acc *= if bit { x } else { EF::ONE - x };
        }
        *dst = acc;
    }
}

/// Get low/high eq pointers for the tail portion of the eq buffer, skipping `drop_count` layers.
/// See gkr-prover.md § "Eq buffer sqrt decomposition".
fn eq_tail_ptrs(
    eq_buffer: &SqrtEqLayers,
    drop_count: usize,
) -> (*const EF, *const EF, usize, usize) {
    let mut high_n = eq_buffer.high_n();
    let mut low_n = eq_buffer.low_n();
    let total_n = high_n + low_n;
    if drop_count >= total_n {
        return (std::ptr::null(), std::ptr::null(), 1, 0);
    }
    if drop_count <= high_n {
        high_n -= drop_count;
    } else {
        low_n -= drop_count - high_n;
        high_n = 0;
    }
    let tail_n = low_n + high_n;
    if tail_n == 0 {
        (std::ptr::null(), std::ptr::null(), 1, 0)
    } else {
        (
            eq_buffer.low.get_ptr(low_n),
            eq_buffer.high.get_ptr(high_n),
            1 << low_n,
            tail_n,
        )
    }
}

fn copy_to_device_ptr<T: Copy>(dst: *mut T, src: &[T]) -> Result<(), FractionalSumcheckError> {
    if src.is_empty() {
        return Ok(());
    }
    unsafe {
        cuda_memcpy::<false, true>(
            dst as *mut c_void,
            src.as_ptr() as *const c_void,
            std::mem::size_of_val(src),
        )?;
    }
    Ok(())
}

/// Observes s_evals in transcript, updates accumulators, and returns the sampled challenge.
#[allow(clippy::too_many_arguments)]
fn observe_and_update(
    d_sum_evals: &DeviceBuffer<EF>,
    transcript: &mut DuplexSpongeGpu,
    round_polys_eval: &mut Vec<[EF; GKR_S_DEG]>,
    r_vec: &mut Vec<EF>,
    prev_s_eval: &mut EF,
    xi_j: EF,
    eq_r_acc: &mut EF,
) -> Result<EF, FractionalSumcheckError> {
    let (s_evals, sp_evals) = reconstruct_s_evals(d_sum_evals, *prev_s_eval, xi_j, *eq_r_acc)?;

    for &eval in &s_evals {
        transcript.observe_ext(eval);
    }
    round_polys_eval.push(s_evals);

    let r = transcript.sample_ext();
    r_vec.push(r);

    let eq_r = eval_eq_mle(&[xi_j], &[r]);
    *eq_r_acc *= eq_r;
    *prev_s_eval = eq_r * interpolate_quadratic_at_012(&sp_evals, r);

    Ok(r)
}

/// Fused revert + compute round: reverts the tree layer and computes s'_0(1) and s'_0(2).
///
/// See gkr-prover.md § "Sumcheck round strategies" for context.
#[allow(clippy::too_many_arguments)]
fn do_sumcheck_round_and_revert(
    eq_buffer: &mut SqrtEqLayers,
    layer: &mut DeviceBuffer<Frac<EF>>,
    pq_size: usize,
    lambda: EF,
    transcript: &mut DuplexSpongeGpu,
    d_sum_evals: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    round_polys_eval: &mut Vec<[EF; GKR_S_DEG]>,
    r_vec: &mut Vec<EF>,
    prev_s_eval: &mut EF,
    xi_j: EF,
    eq_r_acc: &mut EF,
) -> Result<EF, FractionalSumcheckError> {
    unsafe {
        frac_compute_round_and_revert(
            eq_buffer,
            layer,
            pq_size / 2,
            lambda,
            d_sum_evals,
            tmp_block_sums,
        )
        .map_err(FractionalSumcheckError::ComputeRound)?;
    }
    eq_buffer.drop_layer();
    observe_and_update(
        d_sum_evals,
        transcript,
        round_polys_eval,
        r_vec,
        prev_s_eval,
        xi_j,
        eq_r_acc,
    )
}

/// Fused compute round: computes s' polynomial AND folds the pq_buffer for next round.
///
/// This kernel fuses the fold operation (using `r_prev` from the previous round) into the current
/// round's compute, eliminating one kernel launch and reducing memory traffic.
#[allow(clippy::too_many_arguments)]
fn do_fused_sumcheck_round(
    eq_buffer: &mut SqrtEqLayers,
    src_pq_buffer: &DeviceBuffer<Frac<EF>>,
    dst_pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    transcript: &mut DuplexSpongeGpu,
    d_sum_evals: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    round_polys_eval: &mut Vec<[EF; GKR_S_DEG]>,
    r_vec: &mut Vec<EF>,
    prev_s_eval: &mut EF,
    xi_j: EF,
    eq_r_acc: &mut EF,
) -> Result<EF, FractionalSumcheckError> {
    unsafe {
        frac_compute_round_and_fold(
            eq_buffer,
            src_pq_buffer,
            dst_pq_buffer,
            src_pq_size,
            lambda,
            r_prev,
            d_sum_evals,
            tmp_block_sums,
        )
        .map_err(FractionalSumcheckError::ComputeRound)?;
    }
    eq_buffer.drop_layer();
    observe_and_update(
        d_sum_evals,
        transcript,
        round_polys_eval,
        r_vec,
        prev_s_eval,
        xi_j,
        eq_r_acc,
    )
}

/// In-place variant of [`do_fused_sumcheck_round`]. Reads and writes to the same buffer.
#[allow(clippy::too_many_arguments)]
fn do_fused_sumcheck_round_inplace(
    eq_buffer: &mut SqrtEqLayers,
    pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    transcript: &mut DuplexSpongeGpu,
    d_sum_evals: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    round_polys_eval: &mut Vec<[EF; GKR_S_DEG]>,
    r_vec: &mut Vec<EF>,
    prev_s_eval: &mut EF,
    xi_j: EF,
    eq_r_acc: &mut EF,
) -> Result<EF, FractionalSumcheckError> {
    unsafe {
        frac_compute_round_and_fold_inplace(
            eq_buffer,
            pq_buffer,
            src_pq_size,
            lambda,
            r_prev,
            d_sum_evals,
            tmp_block_sums,
        )
        .map_err(FractionalSumcheckError::ComputeRound)?;
    }
    eq_buffer.drop_layer();
    observe_and_update(
        d_sum_evals,
        transcript,
        round_polys_eval,
        r_vec,
        prev_s_eval,
        xi_j,
        eq_r_acc,
    )
}

/// GKR fractional sumcheck prover. See `docs/cuda-backend/gkr-prover.md` (repo root) for the
/// protocol and implementation details.
#[instrument(skip_all)]
pub fn fractional_sumcheck_gpu(
    transcript: &mut DuplexSpongeGpu,
    leaves: DeviceBuffer<Frac<EF>>,
    alpha: EF,
    assert_zero: bool,
    mem: &mut MemTracker,
) -> Result<(FracSumcheckProof<SC>, Vec<EF>), FractionalSumcheckError> {
    let mut layer = leaves;
    if layer.is_empty() {
        return Ok((
            FracSumcheckProof {
                fractional_sum: (EF::ZERO, EF::ONE),
                claims_per_layer: vec![],
                sumcheck_polys: vec![],
            },
            vec![],
        ));
    };
    let total_leaves = layer.len();
    // total_rounds = l_skip + n_logup
    let total_rounds = log2_strict_usize(total_leaves);
    assert!(total_rounds > 0, "n_logup > 0 when there are interactions");
    // Build segment tree.
    // - We only maintain the current layer
    // - Input layer uses separate F and EF buffers to save memory
    // - First tree layer converts (F, EF) to FracExt (EF, EF)

    // We store it in bit-reversal order for coalesced memory accesses.
    unsafe {
        // SAFETY: Frac<EF> has exact same memory layout and alignment as (EF, EF).
        let buf = transmute::<&DeviceBuffer<Frac<EF>>, &DeviceBuffer<(EF, EF)>>(&layer);
        bit_rev_frac_ext(
            buf,
            buf,
            total_rounds as u32,
            total_leaves.try_into().unwrap(),
            1,
        )
        .map_err(FractionalSumcheckError::BitReversal)?;
    }

    // Build segment tree with fused alpha application
    for i in 0..total_rounds {
        unsafe {
            if i == 0 {
                // Fuse alpha into first tree layer (applies to both halves)
                frac_build_tree_layer(&mut layer, total_leaves >> i, false, alpha, true)
                    .map_err(FractionalSumcheckError::SegmentTree)?;
            } else {
                frac_build_tree_layer(&mut layer, total_leaves >> i, false, EF::ZERO, false)
                    .map_err(FractionalSumcheckError::SegmentTree)?;
            }
        }
    }
    mem.emit_metrics_with_label("frac_sumcheck.segment_tree");
    mem.tracing_info("fractional_sumcheck_gkr: after building segment tree");
    let mut copy_scratch = DeviceBuffer::<Frac<EF>>::with_capacity(1);
    let root = copy_from_device(&layer, 0, &mut copy_scratch)?;
    unsafe {
        frac_build_tree_layer(&mut layer, 2, true, EF::ZERO, false)
            .map_err(FractionalSumcheckError::SegmentTree)?;
    }
    if assert_zero {
        if root.p != EF::ZERO {
            return Err(FractionalSumcheckError::NonzeroRootSum {
                p: root.p,
                q: root.q,
            });
        }
    } else {
        transcript.observe_ext(root.p);
    }
    transcript.observe_ext(root.q);

    let mut claims_per_layer = Vec::with_capacity(total_rounds);
    let mut sumcheck_polys = Vec::with_capacity(total_rounds);

    let first_left = copy_from_device(&layer, 0, &mut copy_scratch)?;
    let first_right = copy_from_device(&layer, 1, &mut copy_scratch)?;
    claims_per_layer.push(GkrLayerClaims {
        p_xi_0: first_left.p,
        q_xi_0: first_left.q,
        p_xi_1: first_right.p,
        q_xi_1: first_right.q,
    });
    for value in [
        claims_per_layer[0].p_xi_0,
        claims_per_layer[0].q_xi_0,
        claims_per_layer[0].p_xi_1,
        claims_per_layer[0].q_xi_1,
    ] {
        transcript.observe_ext(value);
    }
    let mu_1 = transcript.sample_ext();
    let mut xi_prev = vec![mu_1];
    let mut d_sum_evals = DeviceBuffer::<EF>::with_capacity(2);

    let precompute_m_env = precompute_m_enabled();

    // Work buffer to avoid revert operations on layer. Only needed for non-last rounds.
    // For the last round (round == total_rounds - 1), we fold in-place on layer.
    // When precompute-M is active, the first multifold folds w+1 variables at once
    // (pending r_prev + w window challenges), so the output is total_leaves >> (2 + w).
    // Fold-eval fallback rounds (rem_n < min_n) need at most 2^min_n elements.
    let max_work_size = if total_rounds > 2 {
        if precompute_m_env {
            (total_leaves >> (2 + GKR_WINDOW_SIZE)).max(1 << GKR_WINDOW_DEFAULT_MIN_N)
        } else {
            total_leaves >> 2
        }
    } else {
        0
    };
    let mut work_buffer = DeviceBuffer::<Frac<EF>>::with_capacity(max_work_size);
    let max_tmp_buffer_capacity = if total_rounds > 1 {
        (unsafe { _frac_compute_round_temp_buffer_size((1 << (total_rounds - 1)) as u32) }) as usize
    } else {
        0
    };
    let mut tmp_block_sums = if max_tmp_buffer_capacity > 0 {
        DeviceBuffer::<EF>::with_capacity(max_tmp_buffer_capacity)
    } else {
        DeviceBuffer::new()
    };
    let precompute_m_min_blocks_threshold = precompute_m_min_blocks_threshold();
    let precompute_m_target_blocks = precompute_m_target_blocks();
    let precompute_m_tail_tile_override = precompute_m_tail_tile_override();
    let precompute_m_min_n = precompute_m_min_n();
    let mut m_buffer = DeviceBuffer::<EF>::new();
    let mut m_partial_buffer = DeviceBuffer::<EF>::new();
    let mut eq_r_prefix_buffer = DeviceBuffer::<EF>::new();
    let mut eq_suffix_buffer = DeviceBuffer::<EF>::new();

    for round in 1..total_rounds {
        let gkr_round_span = debug_span!("GKR", round).entered();

        // Note: frac_build_tree_layer revert is now fused into do_sumcheck_round_and_revert below.

        debug_assert_eq!(xi_prev.len(), round);
        // eq_buffer stores eq(xi_prev[j..], x) for x in H_{xi_prev.len()-j} for
        // j=1,...,xi_prev.len()-1.
        let mut eq_buffer = SqrtEqLayers::from_xi(&xi_prev[1..])
            .map_err(FractionalSumcheckError::EvalEqHypercube)?;

        let mut round_polys_eval = Vec::with_capacity(round);
        let mut r_vec = Vec::with_capacity(round);
        let mut pq_size = 2 << round;

        let lambda = transcript.sample_ext();

        let tmp_buffer_capacity =
            unsafe { _frac_compute_round_temp_buffer_size((1 << round) as u32) } as usize;
        if tmp_buffer_capacity > tmp_block_sums.len() {
            tmp_block_sums = DeviceBuffer::<EF>::with_capacity(tmp_buffer_capacity);
        }

        let last_outer_round = round == total_rounds - 1;
        debug_assert!(round > 0);
        let backend = choose_round_strategy(
            round,
            precompute_m_env,
            precompute_m_min_blocks_threshold,
            precompute_m_target_blocks,
            precompute_m_tail_tile_override,
            precompute_m_min_n,
        );

        // In round `j`, contains `s_{j-1}(r_{j-1})`. Starts with the sumcheck's sum claim.
        let (numer_claim, denom_claim) =
            reduce_to_single_evaluation(claims_per_layer.last().unwrap(), /* mu */ xi_prev[0]);
        let mut prev_s_eval = numer_claim + lambda * denom_claim;
        let mut eq_r_acc = EF::ONE;

        // Round 0: compute + revert fused. The pq_buffer fold will be fused into next round's
        // compute. This fuses frac_build_tree_layer(revert=true) with the first inner round
        // compute.
        let r0 = do_sumcheck_round_and_revert(
            &mut eq_buffer,
            &mut layer,
            pq_size,
            lambda,
            transcript,
            &mut d_sum_evals,
            &mut tmp_block_sums,
            &mut round_polys_eval,
            &mut r_vec,
            &mut prev_s_eval,
            xi_prev[0],
            &mut eq_r_acc,
        )?;

        // Fused rounds 1..(round-1): compute + fold using prev_r.
        let mut prev_r = r0;
        let active: &mut DeviceBuffer<Frac<EF>>;

        match backend {
            GkrRoundStrategy::FoldEval => {
                // Existing fused path.
                let mut scheduler = BufferScheduler::new(max_work_size);
                for &xi_j in xi_prev.iter().skip(1) {
                    let src_pq_size = pq_size;
                    let post_fold_size = pq_size >> 1;

                    let r = match scheduler.next_target(post_fold_size, last_outer_round) {
                        BufferTarget::LayerToWork => do_fused_sumcheck_round(
                            &mut eq_buffer,
                            &layer,
                            &mut work_buffer,
                            src_pq_size,
                            lambda,
                            prev_r,
                            transcript,
                            &mut d_sum_evals,
                            &mut tmp_block_sums,
                            &mut round_polys_eval,
                            &mut r_vec,
                            &mut prev_s_eval,
                            xi_j,
                            &mut eq_r_acc,
                        )?,
                        BufferTarget::WorkToLayer => do_fused_sumcheck_round(
                            &mut eq_buffer,
                            &work_buffer,
                            &mut layer,
                            src_pq_size,
                            lambda,
                            prev_r,
                            transcript,
                            &mut d_sum_evals,
                            &mut tmp_block_sums,
                            &mut round_polys_eval,
                            &mut r_vec,
                            &mut prev_s_eval,
                            xi_j,
                            &mut eq_r_acc,
                        )?,
                        BufferTarget::InPlaceLayer => do_fused_sumcheck_round_inplace(
                            &mut eq_buffer,
                            &mut layer,
                            src_pq_size,
                            lambda,
                            prev_r,
                            transcript,
                            &mut d_sum_evals,
                            &mut tmp_block_sums,
                            &mut round_polys_eval,
                            &mut r_vec,
                            &mut prev_s_eval,
                            xi_j,
                            &mut eq_r_acc,
                        )?,
                        BufferTarget::InPlaceWork => do_fused_sumcheck_round_inplace(
                            &mut eq_buffer,
                            &mut work_buffer,
                            src_pq_size,
                            lambda,
                            prev_r,
                            transcript,
                            &mut d_sum_evals,
                            &mut tmp_block_sums,
                            &mut round_polys_eval,
                            &mut r_vec,
                            &mut prev_s_eval,
                            xi_j,
                            &mut eq_r_acc,
                        )?,
                    };

                    pq_size >>= 1;
                    prev_r = r;
                }

                // Final fold after last r (no next compute to fuse with).
                active = match scheduler.final_fold_target(last_outer_round) {
                    BufferTarget::InPlaceWork => {
                        unsafe {
                            fold_ef_frac_columns_inplace(&mut work_buffer, pq_size, prev_r)
                                .map_err(FractionalSumcheckError::FoldColumns)?;
                        }
                        &mut work_buffer
                    }
                    BufferTarget::InPlaceLayer => {
                        unsafe {
                            fold_ef_frac_columns_inplace(&mut layer, pq_size, prev_r)
                                .map_err(FractionalSumcheckError::FoldColumns)?;
                        }
                        &mut layer
                    }
                    BufferTarget::LayerToWork => {
                        unsafe {
                            fold_ef_frac_columns(&layer, &mut work_buffer, pq_size, prev_r)
                                .map_err(FractionalSumcheckError::FoldColumns)?;
                        }
                        &mut work_buffer
                    }
                    BufferTarget::WorkToLayer => unreachable!(),
                };
                pq_size >>= 1;
            }
            GkrRoundStrategy::PrecomputeM => {
                let base = 1usize;
                let stop = round.div_ceil(2);

                // First window reads from `layer` with pending_fold=true
                // (M-build folds prev_r inline). Multifold writes to
                // `active_pq` (work_buffer for non-last rounds, layer for
                // last). After the first window, subsequent windows
                // read/write active_pq with pending_fold=false.
                let mut pending_fold = true;
                let layer_read_ptr = layer.as_ptr();
                let active_pq = if last_outer_round {
                    &mut layer
                } else {
                    &mut work_buffer
                };

                // w+1 to accommodate r_prev prepended to window challenges
                // on the first iteration (inline fold on last outer round).
                let mut eq_r_window_host = vec![EF::ZERO; 1 << (GKR_WINDOW_SIZE + 1)];
                let mut eq_r_prefix_host = vec![EF::ZERO; 1 << GKR_WINDOW_SIZE];
                let mut eq_suffix_host = vec![EF::ZERO; 1 << GKR_WINDOW_SIZE];

                if eq_r_prefix_buffer.is_empty() {
                    eq_r_prefix_buffer =
                        DeviceBuffer::<EF>::with_capacity(1usize << GKR_WINDOW_SIZE);
                }
                if eq_suffix_buffer.is_empty() {
                    eq_suffix_buffer = DeviceBuffer::<EF>::with_capacity(1usize << GKR_WINDOW_SIZE);
                }

                let mut base = base;
                while base < stop {
                    let rem_n = round - base;
                    let rounds_left = stop - base;
                    let Some(w) = choose_precompute_m_window_w(
                        rem_n,
                        rounds_left,
                        precompute_m_min_blocks_threshold,
                        precompute_m_target_blocks,
                        precompute_m_tail_tile_override,
                        precompute_m_min_n,
                    ) else {
                        break;
                    };
                    if m_buffer.is_empty() {
                        let max_m_len = 1usize << (2 * GKR_WINDOW_SIZE);
                        m_buffer = DeviceBuffer::<EF>::with_capacity(max_m_len);
                    }
                    let m_ptr = m_buffer.as_mut_ptr();
                    // Reuse tmp_block_sums for eq_r_window upload.
                    // Safe: M eval rounds finish before multifold needs eq_r_window,
                    // and tmp_block_sums is not needed until next fold-eval round.
                    let max_eq_r_window_len = 1usize << (GKR_WINDOW_SIZE + 1);
                    debug_assert!(
                        tmp_block_sums.len() >= max_eq_r_window_len,
                        "tmp_block_sums too small for eq_r_window: {} < {}",
                        tmp_block_sums.len(),
                        max_eq_r_window_len,
                    );
                    let d_eq_r_window = tmp_block_sums.as_mut_ptr();

                    // When pending_fold is true, the buffer has rem_n+1 variables
                    // and the kernel folds inline. The effective tail dimension
                    // for tiling is the same either way (rem_n - w).
                    let tail_tile = precompute_m_build_tail_tile(
                        rem_n,
                        w,
                        precompute_m_min_blocks_threshold,
                        precompute_m_target_blocks,
                        precompute_m_tail_tile_override,
                    );
                    let num_blocks = precompute_m_num_tail_blocks(rem_n, w, tail_tile);
                    let m_len = (1usize << w) * (1usize << w);
                    let partial_len = num_blocks * m_len;
                    if partial_len > m_partial_buffer.len() {
                        m_partial_buffer = DeviceBuffer::<EF>::with_capacity(partial_len);
                    }

                    let (eq_tail_low, eq_tail_high, eq_low_cap, _) =
                        eq_tail_ptrs(&eq_buffer, w - 1);

                    let r_fold = prev_r; // save before eval loop overwrites prev_r
                    let build_src = if pending_fold {
                        layer_read_ptr
                    } else {
                        active_pq.as_ptr()
                    };
                    unsafe {
                        frac_precompute_m_build_raw(
                            build_src,
                            rem_n,
                            w,
                            lambda,
                            r_fold,
                            pending_fold, // inline fold only on first iteration
                            eq_tail_low,
                            eq_tail_high,
                            eq_low_cap,
                            tail_tile,
                            m_partial_buffer.as_mut_ptr(),
                            partial_len,
                            m_ptr,
                        )
                        .map_err(FractionalSumcheckError::ComputeRound)?;
                    }

                    let mut window_rs = Vec::with_capacity(w);
                    for t in 0..w {
                        let prefix_bits = t;
                        let suffix_bits = w - t - 1;
                        eval_mle_table(&window_rs, &mut eq_r_prefix_host);
                        eval_mle_table(&xi_prev[base + t + 1..base + w], &mut eq_suffix_host);

                        copy_to_device_ptr(
                            eq_r_prefix_buffer.as_mut_ptr(),
                            &eq_r_prefix_host[..(1usize << prefix_bits)],
                        )?;
                        copy_to_device_ptr(
                            eq_suffix_buffer.as_mut_ptr(),
                            &eq_suffix_host[..(1usize << suffix_bits)],
                        )?;
                        unsafe {
                            frac_precompute_m_eval_round_raw(
                                m_ptr,
                                w,
                                t,
                                eq_r_prefix_buffer.as_ptr(),
                                eq_suffix_buffer.as_ptr(),
                                d_sum_evals.as_mut_ptr(),
                            )
                            .map_err(FractionalSumcheckError::ComputeRound)?;
                        }
                        eq_buffer.drop_layer();
                        let r = observe_and_update(
                            &d_sum_evals,
                            transcript,
                            &mut round_polys_eval,
                            &mut r_vec,
                            &mut prev_s_eval,
                            xi_prev[base + t],
                            &mut eq_r_acc,
                        )?;
                        prev_r = r;
                        window_rs.push(r);
                    }

                    // Compute eq_r_window for the multifold.
                    let (buf_vars, w_fold) = if pending_fold {
                        let mut all_rs = Vec::with_capacity(w + 1);
                        all_rs.push(r_fold);
                        all_rs.extend_from_slice(&window_rs);
                        eval_mle_table(&all_rs, &mut eq_r_window_host);
                        copy_to_device_ptr(d_eq_r_window, &eq_r_window_host[..(1 << (w + 1))])?;
                        (rem_n + 1, w + 1)
                    } else {
                        eval_mle_table(&window_rs, &mut eq_r_window_host);
                        copy_to_device_ptr(d_eq_r_window, &eq_r_window_host[..(1 << w)])?;
                        (rem_n, w)
                    };

                    let multifold_src = if pending_fold {
                        layer_read_ptr
                    } else {
                        active_pq.as_ptr()
                    };
                    unsafe {
                        frac_multifold_raw(
                            multifold_src,
                            active_pq.as_mut_ptr(),
                            buf_vars,
                            w_fold,
                            d_eq_r_window,
                        )
                        .map_err(FractionalSumcheckError::FoldColumns)?;
                    }
                    pq_size >>= w_fold;
                    pending_fold = false;
                    base += w;
                }

                if base < round {
                    // First tail round is standalone compute,
                    // subsequent rounds are fold+compute.
                    unsafe {
                        frac_compute_round(
                            &eq_buffer,
                            active_pq,
                            pq_size / 2,
                            lambda,
                            &mut d_sum_evals,
                            &mut tmp_block_sums,
                        )
                        .map_err(FractionalSumcheckError::ComputeRound)?;
                    }
                    eq_buffer.drop_layer();
                    prev_r = observe_and_update(
                        &d_sum_evals,
                        transcript,
                        &mut round_polys_eval,
                        &mut r_vec,
                        &mut prev_s_eval,
                        xi_prev[base],
                        &mut eq_r_acc,
                    )?;

                    for &xi_j in xi_prev.iter().skip(base + 1) {
                        let src_pq_size = pq_size;
                        prev_r = do_fused_sumcheck_round_inplace(
                            &mut eq_buffer,
                            active_pq,
                            src_pq_size,
                            lambda,
                            prev_r,
                            transcript,
                            &mut d_sum_evals,
                            &mut tmp_block_sums,
                            &mut round_polys_eval,
                            &mut r_vec,
                            &mut prev_s_eval,
                            xi_j,
                            &mut eq_r_acc,
                        )?;
                        pq_size >>= 1;
                    }
                }

                unsafe {
                    fold_ef_frac_columns_inplace(active_pq, pq_size, prev_r)
                        .map_err(FractionalSumcheckError::FoldColumns)?;
                }
                active = active_pq;
                pq_size >>= 1;
            }
        }

        let pq_host = [
            copy_from_device(active, 0, &mut copy_scratch)?,
            copy_from_device(active, pq_size / 2, &mut copy_scratch)?,
        ];

        claims_per_layer.push(GkrLayerClaims {
            p_xi_0: pq_host[0].p,
            q_xi_0: pq_host[0].q,
            p_xi_1: pq_host[1].p,
            q_xi_1: pq_host[1].q,
        });

        transcript.observe_ext(claims_per_layer[round].p_xi_0);
        transcript.observe_ext(claims_per_layer[round].q_xi_0);
        transcript.observe_ext(claims_per_layer[round].p_xi_1);
        transcript.observe_ext(claims_per_layer[round].q_xi_1);

        let mu = transcript.sample_ext();
        xi_prev = [vec![mu], r_vec].concat();

        sumcheck_polys.push(round_polys_eval);
        gkr_round_span.exit();
    }
    mem.emit_metrics_with_label("frac_sumcheck.gkr_rounds");
    mem.tracing_info("after_fractional_sumcheck_gkr");

    Ok((
        FracSumcheckProof {
            fractional_sum: (root.p, root.q),
            claims_per_layer,
            sumcheck_polys,
        },
        xi_prev,
    ))
}

fn copy_from_device<T: Copy>(
    buf: &DeviceBuffer<T>,
    index: usize,
    scratch: &mut DeviceBuffer<T>,
) -> Result<T, FractionalSumcheckError> {
    debug_assert!(!scratch.is_empty());
    unsafe {
        cuda_memcpy::<true, true>(
            scratch.as_mut_raw_ptr(),
            buf.as_ptr().add(index) as *const std::ffi::c_void,
            std::mem::size_of::<T>(),
        )?;
    }
    let host = scratch.to_host()?;
    Ok(host[0])
}

/// Reduces claims to a single evaluation point using linear interpolation.
fn reduce_to_single_evaluation(claims: &GkrLayerClaims<SC>, mu: EF) -> (EF, EF) {
    let numer = interpolate_linear_at_01(&[claims.p_xi_0, claims.p_xi_1], mu);
    let denom = interpolate_linear_at_01(&[claims.q_xi_0, claims.q_xi_1], mu);
    (numer, denom)
}

/// Reconstructs the full s(1,2,3) evaluations from s'(1,2) evaluations returned by GPU.
///
/// Goal: compute s({1,2,3}) from s(X) = eq(xi_j, X) * s'(X).
/// Reconstruct the full round polynomial s_t from GPU-computed s'_t(1), s'_t(2).
///
/// See gkr-prover.md § "Sumcheck round implementation" for the derivation.
fn reconstruct_s_evals(
    d_sum_evals: &DeviceBuffer<EF>,
    prev_s_eval: EF,
    xi_j: EF,
    eq_r_acc: EF,
) -> Result<([EF; GKR_S_DEG], [EF; GKR_S_DEG]), FractionalSumcheckError> {
    let sp_vec = d_sum_evals.to_host()?;
    debug_assert_eq!(sp_vec.len(), GKR_S_DEG - 1);

    // sp_evals holds evaluations of degree 2 poly `eq(xi_{j+1..}, r_{j+1..}) * s'(X)` at {0,1,2}
    let mut sp_evals = [EF::ZERO; GKR_S_DEG];
    sp_evals[1] = sp_vec[0] * eq_r_acc;
    sp_evals[2] = sp_vec[1] * eq_r_acc;

    // We use that s_j(0) + s_j(1) = s_{j-1}(r_{j-1})
    // s_j(X) = eq(xi_j, X) * sp_j(X)
    // s_j(0) = (1 - xi_j) * sp_j(0)
    // s_j(1) = xi_j * sp_j(1)
    // So: (1 - xi_j) * sp_j(0) + xi_j * sp_j(1) = prev_s_eval
    // xi_j is randomly sampled so 1 - xi_j should be invertible
    let eq_xi_0 = EF::ONE - xi_j;
    debug_assert_ne!(eq_xi_0, EF::ZERO);
    let eq_xi_1 = xi_j;
    sp_evals[0] = (prev_s_eval - eq_xi_1 * sp_evals[1]) * eq_xi_0.inverse();

    let s_evals: [EF; GKR_S_DEG] = from_fn(|i| {
        // evaluate s at X = i + 1 (skip 0 evaluation)
        let x = EF::from_usize(i + 1);
        let sp_eval = if i < GKR_S_DEG - 1 {
            sp_evals[i + 1]
        } else {
            interpolate_quadratic_at_012(&sp_evals, x)
        };
        eval_eq_mle(&[xi_j], &[x]) * sp_eval
    });

    Ok((s_evals, sp_evals))
}

/// Generate random fractional leaves on device for benchmarking.
pub fn make_synthetic_leaves(n: usize) -> Result<DeviceBuffer<Frac<EF>>, FractionalSumcheckError> {
    use openvm_cuda_common::copy::cuda_memcpy;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let size = 1usize << n;
    let mut rng = StdRng::seed_from_u64(42);
    let host: Vec<(EF, EF)> = (0..size)
        .map(|_| (rng.random::<EF>(), rng.random::<EF>()))
        .collect();
    let d_leaves = DeviceBuffer::<Frac<EF>>::with_capacity(size);
    unsafe {
        cuda_memcpy::<false, true>(
            d_leaves.as_mut_raw_ptr(),
            host.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(host.as_slice()),
        )?;
    }
    Ok(d_leaves)
}

#[cfg(test)]
mod tests {
    use openvm_cuda_common::{memory_manager::MemTracker, stream::current_stream_sync};
    use p3_field::PrimeCharacteristicRing;

    use super::{
        fractional_sumcheck_gpu, make_synthetic_leaves, DuplexSpongeGpu, FractionalSumcheckError,
        GkrRoundStrategy, EF, SC,
    };

    /// Run fractional sumcheck with a given round strategy and return the proof + final randomness.
    fn run_with_strategy(
        n: usize,
        strategy: GkrRoundStrategy,
    ) -> Result<(super::FracSumcheckProof<SC>, Vec<EF>), FractionalSumcheckError> {
        // SAFETY: test sets process env; run with --test-threads=1.
        let enable_precompute_m = matches!(strategy, GkrRoundStrategy::PrecomputeM);
        unsafe {
            std::env::set_var(
                "SWIRL_CUDA_GKR_PRECOMPUTE_M",
                if enable_precompute_m { "1" } else { "0" },
            );
        }
        let mut transcript = DuplexSpongeGpu::default();
        let leaves = make_synthetic_leaves(n)?;
        let mut mem = MemTracker::start("test.precompute_m");
        let result = fractional_sumcheck_gpu(&mut transcript, leaves, EF::ZERO, false, &mut mem)?;
        current_stream_sync().expect("sync");
        Ok(result)
    }

    fn assert_proofs_equal(
        a: &(super::FracSumcheckProof<SC>, Vec<EF>),
        b: &(super::FracSumcheckProof<SC>, Vec<EF>),
    ) {
        assert_eq!(
            a.0.fractional_sum, b.0.fractional_sum,
            "fractional_sum mismatch"
        );
        assert_eq!(
            a.0.claims_per_layer, b.0.claims_per_layer,
            "claims_per_layer mismatch"
        );
        assert_eq!(
            a.0.sumcheck_polys, b.0.sumcheck_polys,
            "sumcheck_polys mismatch"
        );
        assert_eq!(a.1, b.1, "final randomness mismatch");
    }

    /// Compares precompute-M against FoldEval at n=24,25,26.
    /// n=24: top layer rem_n=22 (one window).
    /// n=25: top layer rem_n=23 (one window, more tail).
    /// n=26: top layer rem_n=24 (one window, even more tail).
    #[test]
    fn test_precompute_m_matches_fused() -> Result<(), FractionalSumcheckError> {
        for n in [24, 25, 26] {
            eprintln!("--- testing n={n} ---");
            let fused = run_with_strategy(n, GkrRoundStrategy::FoldEval)?;
            let precompute = run_with_strategy(n, GkrRoundStrategy::PrecomputeM)?;
            assert_proofs_equal(&fused, &precompute);
        }
        Ok(())
    }

    /// Compares precompute-M against FoldEval with lowered thresholds to force multi-window
    /// iteration at small n. At n=16, round=15: stop=8, window 1 at base=1 (rem_n=14),
    /// window 2 at base=4 (rem_n=11), then fold-eval tail for the remaining round.
    #[test]
    fn test_precompute_m_multi_window_matches_fused() -> Result<(), FractionalSumcheckError> {
        unsafe {
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N", "8");
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS", "1");
        }
        let fused = run_with_strategy(16, GkrRoundStrategy::FoldEval)?;
        let precompute = run_with_strategy(16, GkrRoundStrategy::PrecomputeM)?;
        assert_proofs_equal(&fused, &precompute);
        unsafe {
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N");
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS");
        }
        Ok(())
    }
}
