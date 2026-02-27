use core::cmp::Reverse;

use itertools::{izip, Itertools};
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use thiserror::Error;

use crate::{
    keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
    poly_common::Squarable,
    proof::Proof,
    verifier::{
        batch_constraints::{verify_zerocheck_and_logup, BatchConstraintError},
        proof_shape::{verify_proof_shape, ProofShapeError},
        stacked_reduction::{verify_stacked_reduction, StackedReductionError},
        whir::{verify_whir, VerifyWhirError},
    },
    FiatShamirTranscript, StarkProtocolConfig,
};

#[derive(Error, Debug, PartialEq, Eq)]
pub enum VerifierError<EF: core::fmt::Debug + core::fmt::Display + PartialEq + Eq> {
    #[error("Protocol and VerifyingKey has mismatch in SystemParams")]
    SystemParamsMismatch,

    #[error("Trace heights are too large")]
    TraceHeightsTooLarge,

    /// A proof without any traces is always considered invalid.
    #[error("Proof has no traces")]
    EmptyTraces,

    #[error("Proof shape verification failed: {0}")]
    ProofShapeError(#[from] ProofShapeError),

    #[error("Batch constraint verification failed: {0}")]
    BatchConstraintError(#[from] BatchConstraintError<EF>),

    #[error("Stacked reduction verification failed: {0}")]
    StackedReductionError(#[from] StackedReductionError<EF>),

    #[error("Whir verification failed: {0}")]
    WhirError(#[from] VerifyWhirError),
}

pub mod batch_constraints;
pub mod evaluator;
pub mod fractional_sumcheck_gkr;
pub mod proof_shape;
pub mod stacked_reduction;
pub mod sumcheck;
pub mod whir;

pub fn verify<SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>>(
    config: &SC,
    mvk: &MultiStarkVerifyingKey<SC>,
    proof: &Proof<SC>,
    transcript: &mut TS,
) -> Result<(), VerifierError<SC::EF>>
where
    SC::EF: p3_field::TwoAdicField,
{
    if config.params() != &mvk.inner.params {
        return Err(VerifierError::SystemParamsMismatch);
    }
    let &Proof {
        common_main_commit,
        trace_vdata,
        public_values,
        gkr_proof,
        batch_constraint_proof,
        stacking_proof,
        whir_proof,
    } = &proof;
    let &MultiStarkVerifyingKey {
        inner: mvk,
        pre_hash: mvk_pre_hash,
    } = &mvk;
    let &MultiStarkVerifyingKey0 {
        params,
        per_air,
        trace_height_constraints,
    } = &mvk;
    let l_skip = params.l_skip;

    let num_airs = per_air.len();
    let num_traces = trace_vdata.iter().flatten().collect_vec().len();
    if num_traces == 0 {
        return Err(VerifierError::EmptyTraces);
    }
    // We verify the proof shape early to return error and prevent later panics
    let layouts = verify_proof_shape::<SC>(mvk, proof)?;

    let mut trace_id_to_air_id: Vec<usize> = (0..num_airs).collect();
    trace_id_to_air_id.sort_by_key(|&air_id| {
        (
            trace_vdata[air_id].is_none(),
            trace_vdata[air_id]
                .as_ref()
                .map(|vdata| Reverse(vdata.log_height)),
            air_id,
        )
    });
    trace_id_to_air_id.truncate(num_traces);

    for constraint in trace_height_constraints {
        let sum = trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                let log_height = trace_vdata[air_id].as_ref().unwrap().log_height;
                // Proof shape will check n <= n_stack is in bounds
                (1 << log_height.max(l_skip)) as u64 * constraint.coefficients[air_id] as u64
            })
            .sum::<u64>();
        if sum >= constraint.threshold as u64 {
            return Err(VerifierError::TraceHeightsTooLarge);
        }
    }

    let omega_skip = SC::F::two_adic_generator(l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();

    // Preamble
    transcript.observe_commit(*mvk_pre_hash);
    transcript.observe_commit(proof.common_main_commit);

    for (trace_vdata, avk, pvs) in izip!(&proof.trace_vdata, per_air, &proof.public_values) {
        let is_air_present = trace_vdata.is_some();

        if !avk.is_required {
            transcript.observe(SC::F::from_bool(is_air_present));
        }
        if let Some(trace_vdata) = trace_vdata {
            if let Some(pdata) = avk.preprocessed_data.as_ref() {
                transcript.observe_commit(pdata.commit);
            } else {
                transcript.observe(SC::F::from_usize(trace_vdata.log_height));
            }
            debug_assert_eq!(
                avk.params.width.cached_mains.len(),
                trace_vdata.cached_commitments.len()
            );
            for commit in &trace_vdata.cached_commitments {
                transcript.observe_commit(*commit);
            }
            debug_assert_eq!(avk.params.num_public_values, pvs.len());
        }
        for pv in pvs {
            transcript.observe(*pv);
        }
    }

    let n_per_trace: Vec<isize> = trace_id_to_air_id
        .iter()
        .map(|&air_id| trace_vdata[air_id].as_ref().unwrap().log_height as isize - l_skip as isize)
        .collect();
    let r = verify_zerocheck_and_logup::<SC, TS>(
        transcript,
        mvk,
        public_values,
        gkr_proof,
        batch_constraint_proof,
        &trace_id_to_air_id,
        &n_per_trace,
        &omega_skip_pows,
    )?;

    let need_rot_per_trace = trace_id_to_air_id
        .iter()
        .map(|&air_id| per_air[air_id].params.need_rot)
        .collect_vec();
    let mut need_rot_per_commit = vec![need_rot_per_trace];
    for &air_id in &trace_id_to_air_id {
        let need_rot = per_air[air_id].params.need_rot;
        if per_air[air_id].preprocessed_data.is_some() {
            need_rot_per_commit.push(vec![need_rot]);
        }
        let cached_len = trace_vdata[air_id]
            .as_ref()
            .unwrap()
            .cached_commitments
            .len();
        for _ in 0..cached_len {
            need_rot_per_commit.push(vec![need_rot]);
        }
    }

    let u_prism = verify_stacked_reduction::<SC, TS>(
        transcript,
        stacking_proof,
        &layouts,
        &need_rot_per_commit,
        l_skip,
        params.n_stack,
        &proof.batch_constraint_proof.column_openings,
        &r,
        &omega_skip_pows,
    )?;

    let (&u0, u_rest) = u_prism.split_first().unwrap();
    let u_cube = u0
        .exp_powers_of_2()
        .take(l_skip)
        .chain(u_rest.iter().copied())
        .collect_vec();

    let mut commits = vec![*common_main_commit];
    for &air_id in trace_id_to_air_id.iter() {
        if let Some(preprocessed) = &per_air[air_id].preprocessed_data {
            commits.push(preprocessed.commit);
        }
        commits.extend(&trace_vdata[air_id].as_ref().unwrap().cached_commitments);
    }

    verify_whir::<SC, TS>(
        transcript,
        config,
        whir_proof,
        &stacking_proof.stacking_openings,
        &commits,
        &u_cube,
    )?;

    Ok(())
}
