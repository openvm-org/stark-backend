use std::{convert::TryInto, mem::transmute};

use openvm_cuda_backend::cuda::ntt::bit_rev_frac_ext;
use openvm_cuda_common::{
    copy::{MemCopyD2H, cuda_memcpy},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use p3_field::FieldAlgebra;
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
        frac_build_tree_layer, frac_compute_round,
    },
    poly::SqrtHyperBuffer,
    sponge::DuplexSpongeGpu,
};

/// Compute one sumcheck round: evaluate polynomial, observe in transcript, sample challenge.
/// Returns the sampled challenge `r`.
#[allow(clippy::too_many_arguments)]
fn do_sumcheck_round(
    eq_buffer: &SqrtHyperBuffer,
    pq_buffer: &DeviceBuffer<Frac<EF>>,
    pq_size: usize,
    lambda: EF,
    transcript: &mut DuplexSpongeGpu,
    d_sum_evals: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    round_polys_eval: &mut Vec<[EF; 3]>,
    r_vec: &mut Vec<EF>,
) -> Result<EF, FractionalSumcheckError> {
    unsafe {
        frac_compute_round(eq_buffer, pq_buffer, pq_size, lambda, d_sum_evals, tmp_block_sums)
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
        (unsafe { _frac_compute_round_temp_buffer_size((1 << (total_rounds - 1)) as u32) })
            as usize
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

        unsafe {
            frac_build_tree_layer(&mut layer, 2 << round, true)
                .map_err(FractionalSumcheckError::SegmentTree)?;
        }

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

        // Round 0: always reads from `layer`, writes to either layer or work_buffer
        {
            let r = do_sumcheck_round(
                &eq_buffer,
                &layer,
                pq_size,
                lambda,
                transcript,
                &mut d_sum_evals,
                &mut tmp_block_sums,
                &mut round_polys_eval,
                &mut r_vec,
            )?;
            eq_buffer
                .fold_columns(r)
                .map_err(FractionalSumcheckError::FoldColumns)?;

            if last_outer_round {
                unsafe {
                    fold_ef_frac_columns_inplace(&mut layer, pq_size, r)
                        .map_err(FractionalSumcheckError::FoldColumns)?;
                }
            } else {
                unsafe {
                    fold_ef_frac_columns(&layer, &mut work_buffer, pq_size, r)
                        .map_err(FractionalSumcheckError::FoldColumns)?;
                }
            }

            pq_size >>= 1;
        }

        // After the first step:
        // - if last_outer_round: we keep folding in-place on `layer`
        // - else: we fold in-place on `work_buffer`
        let active: &mut DeviceBuffer<Frac<EF>> = if last_outer_round {
            &mut layer
        } else {
            &mut work_buffer
        };

        // Remaining rounds: always read + fold in-place on `active`
        for _ in 1..round {
            let r = do_sumcheck_round(
                &eq_buffer,
                active,
                pq_size,
                lambda,
                transcript,
                &mut d_sum_evals,
                &mut tmp_block_sums,
                &mut round_polys_eval,
                &mut r_vec,
            )?;
            eq_buffer
                .fold_columns(r)
                .map_err(FractionalSumcheckError::FoldColumns)?;
            unsafe {
                fold_ef_frac_columns_inplace(active, pq_size, r)
                    .map_err(FractionalSumcheckError::FoldColumns)?;
            }
            pq_size >>= 1;
        }

        let pq_host = [
            copy_from_device(&*active, 0, &mut copy_scratch)?,
            copy_from_device(&*active, pq_size / 2, &mut copy_scratch)?,
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
