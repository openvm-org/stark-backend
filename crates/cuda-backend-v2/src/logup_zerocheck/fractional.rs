use std::{convert::TryInto, mem::transmute};

use openvm_cuda_backend::cuda::ntt::bit_rev_frac_ext;
use openvm_cuda_common::{
    copy::{MemCopyD2H, cuda_memcpy},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use p3_field::PrimeCharacteristicRing;
use p3_util::log2_strict_usize;
use stark_backend_v2::{
    poseidon2::sponge::FiatShamirTranscript,
    proof::GkrLayerClaims,
    prover::fractional_sumcheck_gkr::{Frac, FracSumcheckProof},
};
use tracing::{debug_span, instrument};

use super::errors::FractionalSumcheckError;
use crate::{
    EF,
    cuda::logup_zerocheck::{
        _frac_compute_round_temp_buffer_size, fold_ef_frac_columns, fold_ef_frac_columns_inplace,
        frac_build_tree_layer, frac_compute_round_and_fold, frac_compute_round_and_fold_inplace,
        frac_compute_round_and_revert,
    },
    poly::SqrtHyperBuffer,
    sponge::DuplexSpongeGpu,
};

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

/// Fused revert + compute round: reverts the tree layer and computes sumcheck polynomial.
///
/// This kernel fuses `frac_build_tree_layer(revert=true)` with the first inner round compute,
/// eliminating one kernel launch per outer round.
#[allow(clippy::too_many_arguments)]
fn do_sumcheck_round_and_revert(
    eq_buffer: &SqrtHyperBuffer,
    layer: &mut DeviceBuffer<Frac<EF>>,
    pq_size: usize,
    lambda: EF,
    transcript: &mut DuplexSpongeGpu,
    d_sum_evals: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    round_polys_eval: &mut Vec<[EF; 3]>,
    r_vec: &mut Vec<EF>,
) -> Result<EF, FractionalSumcheckError> {
    unsafe {
        frac_compute_round_and_revert(
            eq_buffer,
            layer,
            pq_size,
            lambda,
            d_sum_evals,
            tmp_block_sums,
        )
        .map_err(FractionalSumcheckError::ComputeRound)?;
    }
    let s_vec = d_sum_evals.to_host()?;
    let s_evals: [EF; 3] = s_vec
        .try_into()
        .expect("sumcheck round produced unexpected number of evaluations");
    for &eval in &s_evals {
        transcript.observe_ext(eval);
    }
    round_polys_eval.push(s_evals);

    let r = transcript.sample_ext();
    r_vec.push(r);
    Ok(r)
}

/// Fused compute round: computes sumcheck polynomial AND folds the pq_buffer for next round.
///
/// This kernel fuses the fold operation (using `r_prev` from the previous round) into the current
/// round's compute, eliminating one kernel launch and reducing memory traffic.
///
/// The eq_buffer should already be folded to the correct size for this round (eq_size =
/// src_pq_size/2).
#[allow(clippy::too_many_arguments)]
fn do_fused_sumcheck_round(
    eq_buffer: &SqrtHyperBuffer,
    src_pq_buffer: &DeviceBuffer<Frac<EF>>,
    dst_pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    transcript: &mut DuplexSpongeGpu,
    d_sum_evals: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    round_polys_eval: &mut Vec<[EF; 3]>,
    r_vec: &mut Vec<EF>,
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
    let s_vec = d_sum_evals.to_host()?;
    let s_evals: [EF; 3] = s_vec
        .try_into()
        .expect("sumcheck round produced unexpected number of evaluations");
    for &eval in &s_evals {
        transcript.observe_ext(eval);
    }
    round_polys_eval.push(s_evals);

    let r = transcript.sample_ext();
    r_vec.push(r);
    Ok(r)
}

/// In-place variant of [`do_fused_sumcheck_round`]. Reads and writes to the same buffer.
#[allow(clippy::too_many_arguments)]
fn do_fused_sumcheck_round_inplace(
    eq_buffer: &SqrtHyperBuffer,
    pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    transcript: &mut DuplexSpongeGpu,
    d_sum_evals: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    round_polys_eval: &mut Vec<[EF; 3]>,
    r_vec: &mut Vec<EF>,
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
    let s_vec = d_sum_evals.to_host()?;
    let s_evals: [EF; 3] = s_vec
        .try_into()
        .expect("sumcheck round produced unexpected number of evaluations");
    for &eval in &s_evals {
        transcript.observe_ext(eval);
    }
    round_polys_eval.push(s_evals);

    let r = transcript.sample_ext();
    r_vec.push(r);
    Ok(r)
}

#[instrument(skip_all)]
pub fn fractional_sumcheck_gpu(
    transcript: &mut DuplexSpongeGpu,
    leaves: DeviceBuffer<Frac<EF>>,
    assert_zero: bool,
    mem: &mut MemTracker,
) -> Result<(FracSumcheckProof<EF>, Vec<EF>), FractionalSumcheckError> {
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

    for i in 0..total_rounds {
        unsafe {
            frac_build_tree_layer(&mut layer, total_leaves >> i, false)
                .map_err(FractionalSumcheckError::SegmentTree)?;
        }
    }
    mem.emit_metrics_with_label("frac_sumcheck.segment_tree");
    mem.tracing_info("fractional_sumcheck_gkr: after building segment tree");
    let mut copy_scratch = DeviceBuffer::<Frac<EF>>::with_capacity(1);
    let root = copy_from_device(&layer, 0, &mut copy_scratch)?;
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
    let mut d_sum_evals = DeviceBuffer::<EF>::with_capacity(3);

    // Work buffer to avoid revert operations on layer. Only needed for non-last rounds.
    // For the last round (round == total_rounds - 1), we fold in-place on layer.
    // For non-last rounds, max pq_size is 2 << (total_rounds - 2) = total_leaves / 2,
    // and after first fold we have total_leaves / 4 elements.
    let max_work_size = if total_rounds > 2 {
        total_leaves >> 2
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

    for round in 1..total_rounds {
        let gkr_round_span = debug_span!("GKR", round).entered();

        // Note: frac_build_tree_layer revert is now fused into do_sumcheck_round_and_revert below.

        xi_prev.reverse();
        let mut eq_buffer =
            SqrtHyperBuffer::from_xi(&xi_prev).map_err(FractionalSumcheckError::EvalEqHypercube)?;

        let mut round_polys_eval = Vec::with_capacity(round);
        let mut r_vec = Vec::with_capacity(round);
        let mut pq_size = 2 << round;

        let lambda = transcript.sample_ext();

        let tmp_buffer_capacity =
            unsafe { _frac_compute_round_temp_buffer_size(eq_buffer.size as u32) } as usize;
        if tmp_buffer_capacity > tmp_block_sums.len() {
            tmp_block_sums = DeviceBuffer::<EF>::with_capacity(tmp_buffer_capacity);
        }

        let last_outer_round = round == total_rounds - 1;
        debug_assert!(round > 0);

        // Round 0: compute + revert fused. The pq_buffer fold will be fused into next round's
        // compute. This fuses frac_build_tree_layer(revert=true) with the first inner round
        // compute.
        let r0 = do_sumcheck_round_and_revert(
            &eq_buffer,
            &mut layer,
            pq_size,
            lambda,
            transcript,
            &mut d_sum_evals,
            &mut tmp_block_sums,
            &mut round_polys_eval,
            &mut r_vec,
        )?;
        eq_buffer
            .fold_columns(r0)
            .map_err(FractionalSumcheckError::FoldColumns)?;

        // Fused rounds 1..(round-1): compute + fold using prev_r.
        // BufferScheduler manages ping-pong vs in-place decisions.
        let mut prev_r = r0;
        let mut scheduler = BufferScheduler::new(max_work_size);

        for _inner_round in 1..round {
            let src_pq_size = pq_size;
            let post_fold_size = pq_size >> 1;

            let r = match scheduler.next_target(post_fold_size, last_outer_round) {
                BufferTarget::LayerToWork => do_fused_sumcheck_round(
                    &eq_buffer,
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
                )?,
                BufferTarget::WorkToLayer => do_fused_sumcheck_round(
                    &eq_buffer,
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
                )?,
                BufferTarget::InPlaceLayer => do_fused_sumcheck_round_inplace(
                    &eq_buffer,
                    &mut layer,
                    src_pq_size,
                    lambda,
                    prev_r,
                    transcript,
                    &mut d_sum_evals,
                    &mut tmp_block_sums,
                    &mut round_polys_eval,
                    &mut r_vec,
                )?,
                BufferTarget::InPlaceWork => do_fused_sumcheck_round_inplace(
                    &eq_buffer,
                    &mut work_buffer,
                    src_pq_size,
                    lambda,
                    prev_r,
                    transcript,
                    &mut d_sum_evals,
                    &mut tmp_block_sums,
                    &mut round_polys_eval,
                    &mut r_vec,
                )?,
            };

            eq_buffer
                .fold_columns(r)
                .map_err(FractionalSumcheckError::FoldColumns)?;
            pq_size >>= 1;
            prev_r = r;
        }

        // Final fold after last r (no next compute to fuse with).
        let active: &mut DeviceBuffer<Frac<EF>> =
            match scheduler.final_fold_target(last_outer_round) {
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
                // WorkToLayer is only used during inner rounds (ping-pong), never for final fold
                BufferTarget::WorkToLayer => unreachable!(),
            };
        pq_size >>= 1;

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
    debug_assert!(scratch.len() >= 1);
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
