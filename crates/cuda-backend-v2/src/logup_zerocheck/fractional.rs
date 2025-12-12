use std::convert::TryInto;

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
    cuda::{logup_zerocheck::{
        _frac_compute_round_temp_buffer_size, fold_ef_frac_columns, frac_build_tree_layer,
        frac_compute_round, frac_fold_columns,
    }, matrix::bitrev},
    poly::evals_eq_hypercube,
};

#[instrument(skip_all)]
pub fn fractional_sumcheck_gpu<TS: FiatShamirTranscript>(
    transcript: &mut TS,
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
        bitrev(&mut layer, total_leaves).map_err(FractionalSumcheckError::BitReversal)?;
    }

    for i in 0..total_rounds {
        unsafe {
            frac_build_tree_layer(&mut layer, total_leaves >> i, false)
                .map_err(FractionalSumcheckError::SegmentTree)?;
        }
    }
    mem.emit_metrics_with_label("frac_sumcheck.segment_tree");
    mem.tracing_info("fractional_sumcheck_gkr: after building segment tree");
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
    let mut d_sum_evals = DeviceBuffer::<EF>::with_capacity(3);

    for round in 1..total_rounds {
        let gkr_round_span = debug_span!("GKR", round).entered();

        unsafe {
            frac_build_tree_layer(&mut layer, 2 << round, true)
                .map_err(FractionalSumcheckError::SegmentTree)?;
        }

        let mut eq_buffer = DeviceBuffer::<EF>::with_capacity(1 << xi_prev.len());
        xi_prev.reverse();
        unsafe {
            evals_eq_hypercube(&mut eq_buffer, &xi_prev).map_err(|e| match e {
                crate::ProverError::Cuda(cuda_err) => {
                    FractionalSumcheckError::EvalEqHypercube(cuda_err)
                }
                crate::ProverError::MemCopy(mem_err) => FractionalSumcheckError::Copy(mem_err),
            })?;
        }

        let mut round_polys_eval = Vec::with_capacity(round);
        let mut r_vec = Vec::with_capacity(round);
        let mut eq_size = 1 << round;
        let mut pq_size = 2 << round;

        let lambda = transcript.sample_ext();

        let tmp_buffer_capacity = unsafe { _frac_compute_round_temp_buffer_size(eq_size as u32) };
        // NOTE: we re-use the buffer across sumcheck rounds below. This requires that the
        // temp_buffer_size only decreases as stride decreases.
        let mut tmp_block_sums = DeviceBuffer::<EF>::with_capacity(tmp_buffer_capacity as usize);

        for _sum_round in 0..round {
            unsafe {
                frac_compute_round(
                    &eq_buffer,
                    &mut layer,
                    eq_size,
                    pq_size,
                    lambda,
                    &mut d_sum_evals,
                    &mut tmp_block_sums,
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

            let r_round = transcript.sample_ext();
            r_vec.push(r_round);

            unsafe {
                frac_fold_columns(&mut eq_buffer, eq_size, r_round)
                    .map_err(FractionalSumcheckError::FoldColumns)?;
                fold_ef_frac_columns(&mut layer, pq_size, r_round, false)
                    .map_err(FractionalSumcheckError::FoldColumns)?;
            }
            eq_size >>= 1;
            pq_size >>= 1;
        }
        let pq_host = [
            copy_from_device(&layer, 0)?,
            copy_from_device(&layer, pq_size / 2)?,
        ];
        for pq_revert_round in (0..round).rev() {
            pq_size <<= 1;
            unsafe {
                fold_ef_frac_columns(&mut layer, pq_size, r_vec[pq_revert_round], true)
                    .map_err(FractionalSumcheckError::FoldColumns)?;
            }
        }

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
) -> Result<T, FractionalSumcheckError> {
    let scratch = DeviceBuffer::<T>::with_capacity(1);
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
