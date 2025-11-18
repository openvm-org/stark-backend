use std::convert::TryInto;

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
    memory_manager::MemTracker,
};
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;
use stark_backend_v2::{
    poseidon2::sponge::FiatShamirTranscript,
    proof::GkrLayerClaims,
    prover::{
        fractional_sumcheck_gkr::{Frac, FracSumcheckProof},
        poly::evals_eq_hypercube,
    },
};
use tracing::{debug_span, instrument};

use super::errors::FractionalSumcheckError;
use crate::{
    EF,
    cuda::logup_zerocheck::{
        fold_frac_ext_columns, frac_build_tree_layer, frac_compute_round, frac_fold_columns,
    },
};

#[instrument(skip_all)]
pub fn fractional_sumcheck_gpu<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    input_evals: DeviceBuffer<Frac<EF>>,
    assert_zero: bool,
    mem: &MemTracker,
) -> Result<(FracSumcheckProof<EF>, Vec<EF>), FractionalSumcheckError> {
    if input_evals.is_empty() {
        return Ok((
            FracSumcheckProof {
                fractional_sum: (EF::ZERO, EF::ONE),
                claims_per_layer: vec![],
                sumcheck_polys: vec![],
            },
            vec![],
        ));
    };
    let total_leaves = input_evals.len();
    // total_rounds = l_skip + n_logup
    let total_rounds = log2_strict_usize(total_leaves);
    assert!(total_rounds > 0, "n_logup > 0 when there are interactions");
    // Build segment tree.
    // - We keep layers as separate buffers so we can drop them earlier
    // - Input layer is not in the tree to specialize the fact it has nodes of type (F, EF)
    let mut tree = Vec::with_capacity(total_rounds);
    let mut out_layer = DeviceBuffer::<Frac<EF>>::with_capacity(total_leaves / 2);
    unsafe {
        frac_build_tree_layer(&mut out_layer, &input_evals, total_leaves / 2)
            .map_err(FractionalSumcheckError::SegmentTree)?;
    }
    tree.push(out_layer);
    for i in (0..total_rounds - 1).rev() {
        let input_layer = tree.last().unwrap();
        debug_assert_eq!(input_layer.len(), 1 << (i + 1));
        let mut out_layer = DeviceBuffer::<Frac<EF>>::with_capacity(input_layer.len() / 2);
        unsafe {
            frac_build_tree_layer(&mut out_layer, input_layer, input_layer.len() / 2)
                .map_err(FractionalSumcheckError::SegmentTree)?;
        }
        tree.push(out_layer);
    }
    mem.tracing_info("fractional_sumcheck_gkr: after building segment tree");
    let root_layer = tree.pop().unwrap();
    let root = copy_frac_from_device(&root_layer, 0)?;
    if assert_zero {
        debug_assert_eq!(root.p, EF::ZERO);
    } else {
        transcript.observe_ext(root.p);
    }
    transcript.observe_ext(root.q);

    let mut claims_per_layer = Vec::with_capacity(total_rounds);
    let mut sumcheck_polys = Vec::with_capacity(total_rounds);

    let (first_left, first_right) = if total_rounds == 1 {
        (
            copy_frac_from_device(&input_evals, 0)?,
            copy_frac_from_device(&input_evals, 1)?,
        )
    } else {
        let layer = tree.pop().unwrap();
        (
            copy_frac_from_device(&layer, 0)?,
            copy_frac_from_device(&layer, 1)?,
        )
    };
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
    let mut input_evals = Some(input_evals);

    for round in 1..total_rounds {
        let gkr_round_span = debug_span!("GKR round {round}").entered();
        let eval_size = 1 << round;

        let mut pq_buffer = if round != total_rounds - 1 {
            tree.pop().unwrap()
        } else {
            input_evals.take().unwrap()
        };

        let eq_host = evals_eq_hypercube(&xi_prev);
        let eq_buffer = eq_host.to_device()?;
        let mut eq_buffer = eq_buffer;

        let mut round_polys_eval = Vec::with_capacity(round);
        let mut r_vec = Vec::with_capacity(round);
        let mut stride = eval_size;

        let lambda = transcript.sample_ext();

        for _sum_round in 0..round {
            unsafe {
                frac_compute_round(&eq_buffer, &pq_buffer, stride, lambda, &mut d_sum_evals)
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

            let next_eq = DeviceBuffer::<EF>::with_capacity(stride >> 1);
            let mut next_pq = DeviceBuffer::<Frac<EF>>::with_capacity(stride);
            unsafe {
                frac_fold_columns(&eq_buffer, stride, 1, r_round, &next_eq)
                    .map_err(FractionalSumcheckError::FoldColumns)?;
                fold_frac_ext_columns(&pq_buffer, stride, 2, r_round, &mut next_pq)
                    .map_err(FractionalSumcheckError::FoldColumns)?;
            }
            eq_buffer = next_eq;
            pq_buffer = next_pq;
            stride >>= 1;
        }
        debug_assert_eq!(pq_buffer.len(), 2);
        let pq_host = pq_buffer.to_host().map_err(FractionalSumcheckError::Copy)?;

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

fn copy_frac_from_device(
    buf: &DeviceBuffer<Frac<EF>>,
    index: usize,
) -> Result<Frac<EF>, FractionalSumcheckError> {
    let scratch = DeviceBuffer::<Frac<EF>>::with_capacity(1);
    unsafe {
        cuda_memcpy::<true, true>(
            scratch.as_mut_raw_ptr(),
            buf.as_ptr().add(index) as *const std::ffi::c_void,
            std::mem::size_of::<Frac<EF>>(),
        )?;
    }
    let host = scratch.to_host()?;
    Ok(host[0])
}
