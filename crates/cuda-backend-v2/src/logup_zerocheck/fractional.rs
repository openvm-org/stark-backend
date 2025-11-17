use std::{convert::TryInto, ffi::c_void};

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
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
use tracing::instrument;

use super::{
    errors::{FractionalSegmentTreeError, FractionalSumcheckError},
    state::FractionalGkrState,
};
use crate::{
    EF,
    cuda::logup_zerocheck::{
        frac_build_segment_tree, frac_compute_round, frac_extract_claims, frac_fold_columns,
        frac_prepare_round,
    },
};

pub fn initialize_segment_tree(
    state: &mut FractionalGkrState,
    evals: DeviceBuffer<Frac<EF>>,
) -> Result<(), FractionalSegmentTreeError> {
    if evals.is_empty() {
        state.input_evals = None;
        state.segment_tree = None;
        state.total_rounds = 0;
        return Ok(());
    }

    let total_leaves = evals.len();
    let total_rounds = log2_strict_usize(total_leaves);
    debug_assert_eq!(1usize << total_rounds, total_leaves);

    let tree_len = 1 << (total_rounds + 1);
    let leaf_offset = 1 << total_rounds;

    let tree_device = DeviceBuffer::<Frac<EF>>::with_capacity(tree_len);
    tree_device.fill_zero()?;
    unsafe {
        cuda_memcpy::<true, true>(
            tree_device.as_mut_ptr().add(leaf_offset) as *mut c_void,
            evals.as_ptr() as *const c_void,
            total_leaves * std::mem::size_of::<Frac<EF>>(),
        )?;
        frac_build_segment_tree(&tree_device, total_leaves)?;
    }

    state.total_rounds = total_rounds;
    state.segment_tree = Some(tree_device);
    state.input_evals = Some(evals);

    Ok(())
}

fn copy_frac_from_device(
    tree: &DeviceBuffer<Frac<EF>>,
    index: usize,
) -> Result<[EF; 2], FractionalSumcheckError> {
    let scratch = DeviceBuffer::<Frac<EF>>::with_capacity(1);
    unsafe {
        cuda_memcpy::<true, true>(
            scratch.as_mut_raw_ptr(),
            tree.as_ptr().add(index) as *const std::ffi::c_void,
            std::mem::size_of::<Frac<EF>>(),
        )?;
    }
    let host = scratch.to_host()?;
    Ok([host[0].p, host[0].q])
}

#[instrument(skip_all)]
pub fn fractional_sumcheck_gpu<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    state: &FractionalGkrState,
    assert_zero: bool,
) -> Result<(FracSumcheckProof<EF>, Vec<EF>), FractionalSumcheckError> {
    let total_rounds = state.total_rounds;
    let Some(tree) = &state.segment_tree else {
        return Ok((
            FracSumcheckProof {
                fractional_sum: (EF::ZERO, EF::ONE),
                claims_per_layer: vec![],
                sumcheck_polys: vec![],
            },
            vec![],
        ));
    };

    let root = copy_frac_from_device(tree, 1)?;
    if assert_zero {
        debug_assert_eq!(root[0], EF::ZERO);
    } else {
        transcript.observe_ext(root[0]);
    }
    transcript.observe_ext(root[1]);

    if total_rounds == 0 {
        return Ok((
            FracSumcheckProof {
                fractional_sum: (root[0], root[1]),
                claims_per_layer: vec![],
                sumcheck_polys: vec![],
            },
            vec![],
        ));
    }

    let mut claims_per_layer = Vec::with_capacity(total_rounds);
    let mut sumcheck_polys = Vec::with_capacity(total_rounds);

    let first_left = copy_frac_from_device(tree, 2)?;
    let first_right = copy_frac_from_device(tree, 3)?;
    claims_per_layer.push(GkrLayerClaims {
        p_xi_0: first_left[0],
        q_xi_0: first_left[1],
        p_xi_1: first_right[0],
        q_xi_1: first_right[1],
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

    for round in 1..total_rounds {
        let eval_size = 1 << round;
        let segment_start = 2 * eval_size;

        let mut pq_buffer = DeviceBuffer::<EF>::with_capacity(4 * eval_size);
        unsafe {
            frac_prepare_round(tree, segment_start, eval_size, &pq_buffer)
                .map_err(FractionalSumcheckError::PrepareRound)?;
        }

        let eq_host = evals_eq_hypercube(&xi_prev);
        let eq_buffer = eq_host.to_device()?;
        let mut eq_buffer = eq_buffer;

        let mut round_polys_eval = Vec::with_capacity(round);
        let mut r_vec = Vec::with_capacity(round);
        let mut stride = eval_size;
        let sum_evals_device = DeviceBuffer::<EF>::with_capacity(3);

        let lambda = transcript.sample_ext();

        for _sum_round in 0..round {
            unsafe {
                frac_compute_round(&eq_buffer, &pq_buffer, stride, lambda, &sum_evals_device)
                    .map_err(FractionalSumcheckError::ComputeRound)?;
            }
            let s_vec = sum_evals_device.to_host()?;
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
            let next_pq = DeviceBuffer::<EF>::with_capacity(4 * (stride >> 1));
            unsafe {
                frac_fold_columns(&eq_buffer, stride, 1, r_round, &next_eq)
                    .map_err(FractionalSumcheckError::FoldColumns)?;
                frac_fold_columns(&pq_buffer, stride, 4, r_round, &next_pq)
                    .map_err(FractionalSumcheckError::FoldColumns)?;
            }
            eq_buffer = next_eq;
            pq_buffer = next_pq;
            stride >>= 1;
        }

        let claim_device = DeviceBuffer::<EF>::with_capacity(4);
        unsafe {
            frac_extract_claims(&pq_buffer, stride, &claim_device)
                .map_err(FractionalSumcheckError::ExtractClaims)?;
        }
        let claim_vec = claim_device.to_host()?;
        let claim_values: [EF; 4] = claim_vec
            .try_into()
            .expect("claim extraction produced unexpected number of values");

        claims_per_layer.push(GkrLayerClaims {
            p_xi_0: claim_values[0],
            q_xi_0: claim_values[1],
            p_xi_1: claim_values[2],
            q_xi_1: claim_values[3],
        });
        for &value in &claim_values {
            transcript.observe_ext(value);
        }

        sumcheck_polys.push(round_polys_eval);

        let mu = transcript.sample_ext();
        xi_prev = std::iter::once(mu).chain(r_vec.iter().copied()).collect();
    }

    Ok((
        FracSumcheckProof {
            fractional_sum: (root[0], root[1]),
            claims_per_layer,
            sumcheck_polys,
        },
        xi_prev,
    ))
}
