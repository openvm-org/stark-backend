use std::{array::from_fn, env};

use openvm_metal_common::d_buffer::MetalBuffer;
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
    metal::{
        logup_zerocheck::{
            fold_ef_frac_columns, fold_ef_frac_columns_inplace, frac_build_tree_layer,
            frac_compute_round, frac_compute_round_and_fold, frac_compute_round_and_fold_inplace,
            frac_compute_round_and_revert, frac_compute_round_temp_buffer_size, frac_multifold_raw,
            frac_precompute_m_build_raw, frac_precompute_m_eval_round_raw,
        },
        ntt::bit_rev_frac_ext,
    },
    poly::SqrtEqLayers,
    prelude::{EF, SC},
    sponge::DuplexSpongeMetal,
};

const GKR_S_DEG: usize = 3;
const GKR_WINDOW_SIZE: usize = 3;
const GKR_WINDOW_DEFAULT_MIN_N: usize = 22;

/// Describes which buffer operation to use for the next fused compute+fold round.
#[derive(Debug, Clone, Copy)]
enum BufferTarget {
    /// Out-of-place: layer -> work_buffer
    LayerToWork,
    /// Out-of-place: work_buffer -> layer
    WorkToLayer,
    /// In-place on layer buffer
    InPlaceLayer,
    /// In-place on work_buffer
    InPlaceWork,
}

/// Encapsulates ping-pong buffer scheduling state for GKR inner rounds.
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
    fn new(work_buffer_cap: usize) -> Self {
        Self {
            data_in_work_buffer: false,
            work_buffer_cap,
        }
    }

    fn can_pingpong(&self, post_fold_size: usize) -> bool {
        post_fold_size <= self.work_buffer_cap
    }

    fn next_target(&mut self, post_fold_size: usize, last_outer_round: bool) -> BufferTarget {
        let can_pingpong = self.can_pingpong(post_fold_size);

        if last_outer_round {
            if can_pingpong {
                if self.data_in_work_buffer {
                    self.data_in_work_buffer = false;
                    BufferTarget::WorkToLayer
                } else {
                    self.data_in_work_buffer = true;
                    BufferTarget::LayerToWork
                }
            } else {
                debug_assert!(
                    !self.data_in_work_buffer,
                    "in-place path requires data in layer"
                );
                BufferTarget::InPlaceLayer
            }
        } else if self.data_in_work_buffer {
            BufferTarget::InPlaceWork
        } else {
            self.data_in_work_buffer = true;
            BufferTarget::LayerToWork
        }
    }

    fn final_fold_target(&self, last_outer_round: bool) -> BufferTarget {
        if last_outer_round {
            if self.data_in_work_buffer {
                BufferTarget::InPlaceWork
            } else {
                BufferTarget::InPlaceLayer
            }
        } else if self.data_in_work_buffer {
            BufferTarget::InPlaceWork
        } else {
            BufferTarget::LayerToWork
        }
    }
}

fn precompute_m_enabled() -> bool {
    matches!(
        env::var("SWIRL_METAL_GKR_PRECOMPUTE_M"),
        Ok(val) if matches!(val.as_str(), "1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn precompute_m_min_blocks_threshold() -> usize {
    env::var("SWIRL_METAL_GKR_PRECOMPUTE_M_MIN_BLOCKS")
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
    env::var("SWIRL_METAL_GKR_PRECOMPUTE_M_TARGET_BLOCKS")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(PRECOMPUTE_M_DEFAULT_TARGET_BLOCKS)
        .max(1)
}

fn precompute_m_tail_tile_override() -> Option<usize> {
    env::var("SWIRL_METAL_GKR_PRECOMPUTE_M_TAIL_TILE")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .map(|v| v.clamp(PRECOMPUTE_M_MIN_TAIL_TILE, PRECOMPUTE_M_TAIL_TILE))
}

fn precompute_m_min_n() -> usize {
    env::var("SWIRL_METAL_GKR_PRECOMPUTE_M_MIN_N")
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
    // On Metal with unified memory, we can just memcpy directly
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
    Ok(())
}

/// Observes s_evals in transcript, updates accumulators, and returns the sampled challenge.
#[allow(clippy::too_many_arguments)]
fn observe_and_update(
    d_sum_evals: &MetalBuffer<EF>,
    transcript: &mut DuplexSpongeMetal,
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

/// Fused revert + compute round.
#[allow(clippy::too_many_arguments)]
fn do_sumcheck_round_and_revert(
    eq_buffer: &mut SqrtEqLayers,
    layer: &mut MetalBuffer<Frac<EF>>,
    pq_size: usize,
    lambda: EF,
    transcript: &mut DuplexSpongeMetal,
    d_sum_evals: &mut MetalBuffer<EF>,
    tmp_block_sums: &mut MetalBuffer<EF>,
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

/// Fused compute round with fold.
#[allow(clippy::too_many_arguments)]
fn do_fused_sumcheck_round(
    eq_buffer: &mut SqrtEqLayers,
    src_pq_buffer: &MetalBuffer<Frac<EF>>,
    dst_pq_buffer: &mut MetalBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    transcript: &mut DuplexSpongeMetal,
    d_sum_evals: &mut MetalBuffer<EF>,
    tmp_block_sums: &mut MetalBuffer<EF>,
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

/// In-place variant of [`do_fused_sumcheck_round`].
#[allow(clippy::too_many_arguments)]
fn do_fused_sumcheck_round_inplace(
    eq_buffer: &mut SqrtEqLayers,
    pq_buffer: &mut MetalBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    transcript: &mut DuplexSpongeMetal,
    d_sum_evals: &mut MetalBuffer<EF>,
    tmp_block_sums: &mut MetalBuffer<EF>,
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

/// GKR fractional sumcheck prover for Metal.
#[instrument(skip_all)]
pub fn fractional_sumcheck_metal(
    transcript: &mut DuplexSpongeMetal,
    leaves: MetalBuffer<Frac<EF>>,
    assert_zero: bool,
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
    let total_rounds = log2_strict_usize(total_leaves);
    assert!(total_rounds > 0, "n_logup > 0 when there are interactions");

    // Bit-reverse for coalesced memory accesses.
    unsafe {
        bit_rev_frac_ext(&layer, &layer, total_rounds as u32, total_leaves as u32, 1)
            .map_err(FractionalSumcheckError::BitReversal)?;
    }

    // Build segment tree.
    for i in 0..total_rounds {
        unsafe {
            frac_build_tree_layer(&mut layer, total_leaves >> i, false)
                .map_err(FractionalSumcheckError::SegmentTree)?;
        }
    }

    let root = copy_from_device(&layer, 0)?;
    unsafe {
        frac_build_tree_layer(&mut layer, 2, true).map_err(FractionalSumcheckError::SegmentTree)?;
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

    let first_left = copy_from_device(&layer, 0)?;
    let first_right = copy_from_device(&layer, 1)?;
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
    let mut d_sum_evals = MetalBuffer::<EF>::with_capacity(2);

    let precompute_m_env = precompute_m_enabled();

    let max_work_size = if total_rounds > 2 {
        if precompute_m_env {
            (total_leaves >> (2 + GKR_WINDOW_SIZE)).max(1 << GKR_WINDOW_DEFAULT_MIN_N)
        } else {
            total_leaves >> 2
        }
    } else {
        0
    };
    let mut work_buffer = MetalBuffer::<Frac<EF>>::with_capacity(max_work_size);
    let max_tmp_buffer_capacity = if total_rounds > 1 {
        frac_compute_round_temp_buffer_size((1 << (total_rounds - 1)) as u32) as usize
    } else {
        0
    };
    let mut tmp_block_sums = if max_tmp_buffer_capacity > 0 {
        MetalBuffer::<EF>::with_capacity(max_tmp_buffer_capacity)
    } else {
        MetalBuffer::<EF>::with_capacity(0)
    };
    let precompute_m_min_blocks_threshold = precompute_m_min_blocks_threshold();
    let precompute_m_target_blocks = precompute_m_target_blocks();
    let precompute_m_tail_tile_override = precompute_m_tail_tile_override();
    let precompute_m_min_n = precompute_m_min_n();
    let mut m_buffer = MetalBuffer::<EF>::with_capacity(0);
    let mut m_partial_buffer = MetalBuffer::<EF>::with_capacity(0);
    let mut eq_r_prefix_buffer = MetalBuffer::<EF>::with_capacity(0);
    let mut eq_suffix_buffer = MetalBuffer::<EF>::with_capacity(0);

    for round in 1..total_rounds {
        let gkr_round_span = debug_span!("GKR", round).entered();

        debug_assert_eq!(xi_prev.len(), round);
        let mut eq_buffer = SqrtEqLayers::from_xi(&xi_prev[1..])
            .map_err(FractionalSumcheckError::EvalEqHypercube)?;

        let mut round_polys_eval = Vec::with_capacity(round);
        let mut r_vec = Vec::with_capacity(round);
        let mut pq_size = 2 << round;

        let lambda = transcript.sample_ext();

        let tmp_buffer_capacity = frac_compute_round_temp_buffer_size((1 << round) as u32) as usize;
        if tmp_buffer_capacity > tmp_block_sums.len() {
            tmp_block_sums = MetalBuffer::<EF>::with_capacity(tmp_buffer_capacity);
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

        let (numer_claim, denom_claim) =
            reduce_to_single_evaluation(claims_per_layer.last().unwrap(), xi_prev[0]);
        let mut prev_s_eval = numer_claim + lambda * denom_claim;
        let mut eq_r_acc = EF::ONE;

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

        let mut prev_r = r0;
        let active: &mut MetalBuffer<Frac<EF>>;

        match backend {
            GkrRoundStrategy::FoldEval => {
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

                let mut pending_fold = true;
                let layer_read_ptr = layer.as_ptr();
                let active_pq = if last_outer_round {
                    &mut layer
                } else {
                    &mut work_buffer
                };

                let mut eq_r_window_host = vec![EF::ZERO; 1 << (GKR_WINDOW_SIZE + 1)];
                let mut eq_r_prefix_host = vec![EF::ZERO; 1 << GKR_WINDOW_SIZE];
                let mut eq_suffix_host = vec![EF::ZERO; 1 << GKR_WINDOW_SIZE];

                if eq_r_prefix_buffer.is_empty() {
                    eq_r_prefix_buffer =
                        MetalBuffer::<EF>::with_capacity(1usize << GKR_WINDOW_SIZE);
                }
                if eq_suffix_buffer.is_empty() {
                    eq_suffix_buffer = MetalBuffer::<EF>::with_capacity(1usize << GKR_WINDOW_SIZE);
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
                        m_buffer = MetalBuffer::<EF>::with_capacity(max_m_len);
                    }
                    let m_ptr = m_buffer.as_mut_ptr();
                    // Reuse tmp_block_sums for eq_r_window upload.
                    let max_eq_r_window_len = 1usize << (GKR_WINDOW_SIZE + 1);
                    debug_assert!(
                        tmp_block_sums.len() >= max_eq_r_window_len,
                        "tmp_block_sums too small for eq_r_window: {} < {}",
                        tmp_block_sums.len(),
                        max_eq_r_window_len,
                    );
                    let d_eq_r_window = tmp_block_sums.as_mut_ptr();

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
                        m_partial_buffer = MetalBuffer::<EF>::with_capacity(partial_len);
                    }

                    let (eq_tail_low, eq_tail_high, eq_low_cap, _) =
                        eq_tail_ptrs(&eq_buffer, w - 1);

                    let r_fold = prev_r;
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
                            pending_fold,
                            eq_tail_low,
                            eq_tail_high,
                            eq_low_cap,
                            tail_tile,
                            &m_partial_buffer,
                            partial_len,
                            &m_buffer,
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
            copy_from_device(active, 0)?,
            copy_from_device(active, pq_size / 2)?,
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

    Ok((
        FracSumcheckProof {
            fractional_sum: (root.p, root.q),
            claims_per_layer,
            sumcheck_polys,
        },
        xi_prev,
    ))
}

/// Copy a single element from device buffer at the given index.
/// On Metal with unified memory, this is a direct pointer read.
fn copy_from_device(
    buf: &MetalBuffer<Frac<EF>>,
    index: usize,
) -> Result<Frac<EF>, FractionalSumcheckError> {
    let p_ptr = buf.as_ptr() as *const EF;
    let q_ptr = unsafe { p_ptr.add(buf.len()) };
    Ok(unsafe { Frac::new(*p_ptr.add(index), *q_ptr.add(index)) })
}

/// Reduces claims to a single evaluation point using linear interpolation.
fn reduce_to_single_evaluation(claims: &GkrLayerClaims<SC>, mu: EF) -> (EF, EF) {
    let numer = interpolate_linear_at_01(&[claims.p_xi_0, claims.p_xi_1], mu);
    let denom = interpolate_linear_at_01(&[claims.q_xi_0, claims.q_xi_1], mu);
    (numer, denom)
}

/// Reconstructs the full s(1,2,3) evaluations from s'(1,2) evaluations returned by GPU.
fn reconstruct_s_evals(
    d_sum_evals: &MetalBuffer<EF>,
    prev_s_eval: EF,
    xi_j: EF,
    eq_r_acc: EF,
) -> Result<([EF; GKR_S_DEG], [EF; GKR_S_DEG]), FractionalSumcheckError> {
    let sp_vec = d_sum_evals.to_vec();
    debug_assert_eq!(sp_vec.len(), GKR_S_DEG - 1);

    let mut sp_evals = [EF::ZERO; GKR_S_DEG];
    sp_evals[1] = sp_vec[0] * eq_r_acc;
    sp_evals[2] = sp_vec[1] * eq_r_acc;

    let eq_xi_0 = EF::ONE - xi_j;
    debug_assert_ne!(eq_xi_0, EF::ZERO);
    let eq_xi_1 = xi_j;
    sp_evals[0] = (prev_s_eval - eq_xi_1 * sp_evals[1]) * eq_xi_0.inverse();

    let s_evals: [EF; GKR_S_DEG] = from_fn(|i| {
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

#[cfg(test)]
mod tests {
    use openvm_metal_common::d_buffer::MetalBuffer;
    use openvm_stark_backend::prover::fractional_sumcheck_gkr::{fractional_sumcheck, Frac};
    use openvm_stark_sdk::config::baby_bear_poseidon2::default_duplex_sponge;
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::{fractional_sumcheck_metal, DuplexSpongeMetal, EF, SC};

    fn frac_vec_to_split_device(leaves: &[Frac<EF>]) -> MetalBuffer<Frac<EF>> {
        let buf = MetalBuffer::<Frac<EF>>::with_capacity(leaves.len());
        if leaves.is_empty() {
            return buf;
        }
        let p_ptr = buf.as_mut_ptr() as *mut EF;
        let q_ptr = unsafe { p_ptr.add(leaves.len()) };
        for (i, leaf) in leaves.iter().enumerate() {
            unsafe {
                *p_ptr.add(i) = leaf.p;
                *q_ptr.add(i) = leaf.q;
            }
        }
        buf
    }

    #[test]
    fn test_fractional_sumcheck_matches_cpu_n14() {
        let n = 14usize;
        let size = 1usize << n;
        let mut rng = StdRng::seed_from_u64(42);
        let leaves: Vec<Frac<EF>> = (0..size)
            .map(|_| Frac::new(rng.random::<EF>(), rng.random::<EF>()))
            .collect();

        let mut transcript_metal = DuplexSpongeMetal::default();
        let metal_leaves = frac_vec_to_split_device(&leaves);
        let (metal_proof, metal_xi) =
            fractional_sumcheck_metal(&mut transcript_metal, metal_leaves, false)
                .expect("metal fractional_sumcheck failed");

        let mut transcript_cpu = default_duplex_sponge();
        let (cpu_proof, cpu_xi) = fractional_sumcheck::<SC, _>(&mut transcript_cpu, &leaves, false);

        assert_eq!(metal_proof.fractional_sum, cpu_proof.fractional_sum);
        assert_eq!(metal_proof.claims_per_layer, cpu_proof.claims_per_layer);
        assert_eq!(metal_proof.sumcheck_polys, cpu_proof.sumcheck_polys);
        assert_eq!(metal_xi, cpu_xi);
    }

    #[test]
    fn test_fractional_sumcheck_matches_cpu_n16() {
        let n = 16usize;
        let size = 1usize << n;
        let mut rng = StdRng::seed_from_u64(1337);
        let leaves: Vec<Frac<EF>> = (0..size)
            .map(|_| Frac::new(rng.random::<EF>(), rng.random::<EF>()))
            .collect();

        let mut transcript_metal = DuplexSpongeMetal::default();
        let metal_leaves = frac_vec_to_split_device(&leaves);
        let (metal_proof, metal_xi) =
            fractional_sumcheck_metal(&mut transcript_metal, metal_leaves, false)
                .expect("metal fractional_sumcheck failed");

        let mut transcript_cpu = default_duplex_sponge();
        let (cpu_proof, cpu_xi) = fractional_sumcheck::<SC, _>(&mut transcript_cpu, &leaves, false);

        assert_eq!(metal_proof.fractional_sum, cpu_proof.fractional_sum);
        assert_eq!(metal_proof.claims_per_layer, cpu_proof.claims_per_layer);
        assert_eq!(metal_proof.sumcheck_polys, cpu_proof.sumcheck_polys);
        assert_eq!(metal_xi, cpu_xi);
    }

    #[test]
    fn test_fractional_sumcheck_matches_cpu_n16_padded_tail() {
        let n = 16usize;
        let size = 1usize << n;
        let active = 60usize * 1024usize; // mirrors self-interaction fixture density
        let mut rng = StdRng::seed_from_u64(4242);
        let alpha = EF::from_u32(17);
        let mut leaves = vec![Frac::new(EF::ZERO, EF::ZERO); size];
        for leaf in leaves.iter_mut().take(active) {
            *leaf = Frac::new(rng.random::<EF>(), rng.random::<EF>());
        }
        for leaf in &mut leaves {
            leaf.q += alpha;
        }

        let mut transcript_metal = DuplexSpongeMetal::default();
        let metal_leaves = frac_vec_to_split_device(&leaves);
        let (metal_proof, metal_xi) =
            fractional_sumcheck_metal(&mut transcript_metal, metal_leaves, false)
                .expect("metal fractional_sumcheck failed");

        let mut transcript_cpu = default_duplex_sponge();
        let (cpu_proof, cpu_xi) = fractional_sumcheck::<SC, _>(&mut transcript_cpu, &leaves, false);

        assert_eq!(metal_proof.fractional_sum, cpu_proof.fractional_sum);
        assert_eq!(metal_proof.claims_per_layer, cpu_proof.claims_per_layer);
        assert_eq!(metal_proof.sumcheck_polys, cpu_proof.sumcheck_polys);
        assert_eq!(metal_xi, cpu_xi);
    }

    #[test]
    fn test_frac_precompute_m_build_reduces_partials() {
        let w = 2usize;
        let rem_n = 4usize;
        let total_entries = (1usize << w) * (1usize << w);
        let num_blocks = 3usize;
        let partial_len = total_entries * num_blocks;

        let partial = MetalBuffer::<EF>::with_capacity(partial_len);
        let m_total = MetalBuffer::<EF>::with_capacity(total_entries);

        let partial_host = (0..partial_len)
            .map(|i| EF::from_u32((i + 1) as u32))
            .collect::<Vec<_>>();
        partial.copy_from_slice(&partial_host);

        unsafe {
            crate::metal::logup_zerocheck::frac_precompute_m_build_raw(
                std::ptr::null(),
                rem_n,
                w,
                EF::ZERO,
                EF::ZERO,
                false,
                std::ptr::null(),
                std::ptr::null(),
                1,
                1,
                &partial,
                partial_len,
                &m_total,
            )
            .expect("frac_precompute_m_build_raw failed");
        }

        let actual = m_total.to_vec();
        let expected = (0..total_entries)
            .map(|entry| {
                (0..num_blocks).fold(EF::ZERO, |acc, block| {
                    acc + partial_host[block * total_entries + entry]
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
