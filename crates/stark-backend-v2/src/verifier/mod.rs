use core::{cmp::Reverse, iter::zip};

use itertools::{Itertools, izip};
use p3_field::{FieldAlgebra, TwoAdicField};
use thiserror::Error;

use crate::{
    F,
    keygen::types::{MultiStarkVerifyingKey0V2, MultiStarkVerifyingKeyV2},
    poly_common::Squarable,
    poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
    verifier::{
        batch_constraints::{BatchConstraintError, verify_zerocheck_and_logup},
        proof_shape::{ProofShapeError, verify_proof_shape},
        stacked_reduction::{StackedReductionError, verify_stacked_reduction},
        whir::{VerifyWhirError, verify_whir},
    },
};

#[derive(Error, Debug)]
pub enum VerifierError {
    #[error("Trace heights are too large")]
    TraceHeightsTooLarge,

    #[error("Proof shape verification failed: {0}")]
    ProofShapeError(#[from] ProofShapeError),

    #[error("Batch constraint verification failed: {0}")]
    BatchConstraintError(#[from] BatchConstraintError),

    #[error("Stacked reduction verification failed: {0}")]
    StackedReductionError(#[from] StackedReductionError),

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

pub fn verify<TS: FiatShamirTranscript>(
    mvk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    transcript: &mut TS,
) -> Result<(), VerifierError> {
    let &Proof {
        common_main_commit,
        trace_vdata,
        public_values,
        gkr_proof,
        batch_constraint_proof,
        stacking_proof,
        whir_proof,
    } = &proof;
    let &MultiStarkVerifyingKeyV2 {
        inner: mvk,
        pre_hash: mvk_pre_hash,
    } = &mvk;
    let &MultiStarkVerifyingKey0V2 {
        params,
        per_air,
        trace_height_constraints,
        max_constraint_degree: _,
    } = &mvk;

    let num_airs = per_air.len();

    let mut trace_id_to_air_id: Vec<usize> = (0..num_airs).collect();
    trace_id_to_air_id.sort_by_key(|&air_id| {
        (
            trace_vdata[air_id].is_none(),
            trace_vdata[air_id]
                .as_ref()
                .map(|vdata| Reverse(vdata.hypercube_dim)),
        )
    });
    let num_traces = trace_vdata.iter().flatten().collect_vec().len();
    trace_id_to_air_id.truncate(num_traces);

    let n_per_trace: Vec<usize> = trace_id_to_air_id
        .iter()
        .map(|&air_id| trace_vdata[air_id].as_ref().unwrap().hypercube_dim)
        .collect();

    for constraint in trace_height_constraints {
        let sum = zip(trace_id_to_air_id.iter(), n_per_trace.iter())
            .map(|(&air_id, n)| {
                (1 << (n + params.l_skip)) as u64 * constraint.coefficients[air_id] as u64
            })
            .sum::<u64>();
        if sum >= constraint.threshold as u64 {
            return Err(VerifierError::TraceHeightsTooLarge);
        }
    }

    let omega_skip = F::two_adic_generator(params.l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << params.l_skip).collect_vec();

    // Preamble
    transcript.observe_commit(*mvk_pre_hash);
    transcript.observe_commit(proof.common_main_commit);

    for (trace_vdata, avk, pvs) in izip!(&proof.trace_vdata, per_air, &proof.public_values) {
        let is_air_present = trace_vdata.is_some();

        if !avk.is_required {
            transcript.observe(F::from_bool(is_air_present));
        }
        if let Some(trace_vdata) = trace_vdata {
            if let Some(pdata) = avk.preprocessed_data.as_ref() {
                transcript.observe_commit(pdata.commit);
            } else {
                transcript.observe(F::from_canonical_usize(trace_vdata.hypercube_dim));
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

    let layouts = verify_proof_shape(mvk, proof)?;

    let r = verify_zerocheck_and_logup(
        transcript,
        mvk,
        public_values,
        gkr_proof,
        batch_constraint_proof,
        &trace_id_to_air_id,
        &n_per_trace,
        &omega_skip_pows,
    )?;

    let u_prism = verify_stacked_reduction(
        transcript,
        stacking_proof,
        &layouts,
        params.l_skip,
        params.n_stack,
        &proof.batch_constraint_proof.column_openings,
        &r,
        &omega_skip_pows,
    )?;

    let (&u0, u_rest) = u_prism.split_first().unwrap();
    let u_cube = u0
        .exp_powers_of_2()
        .take(params.l_skip)
        .chain(u_rest.iter().copied())
        .collect_vec();

    let mut commits = vec![*common_main_commit];
    for &air_id in trace_id_to_air_id.iter() {
        if let Some(preprocessed) = &per_air[air_id].preprocessed_data {
            commits.push(preprocessed.commit);
        }
        commits.extend(&trace_vdata[air_id].as_ref().unwrap().cached_commitments);
    }

    verify_whir(
        transcript,
        params,
        whir_proof,
        &stacking_proof.stacking_openings,
        &commits,
        &u_cube,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::config::setup_tracing_with_log_level;
    use tracing::Level;

    use crate::{
        BabyBearPoseidon2CpuEngineV2,
        keygen::types::SystemParams,
        test_utils::{
            CachedFixture11, DuplexSpongeRecorder, DuplexSpongeValidator, FibFixture,
            InteractionsFixture11, PreprocessedFibFixture, TestFixture, test_system_params_small,
        },
        verifier::{VerifierError, verify},
    };

    #[test]
    fn test_fib_air_roundtrip() -> Result<(), VerifierError> {
        setup_tracing_with_log_level(Level::INFO);

        let params = SystemParams {
            l_skip: 2,
            n_stack: 8,
            log_blowup: 1,
            k_whir: 4,
            num_whir_queries: 100,
            log_final_poly_len: 2,
            logup_pow_bits: 1,
            whir_pow_bits: 1,
        };
        let log_trace_degree = params.l_skip + params.n_stack;
        let fib = FibFixture::new(0, 1, 1 << log_trace_degree);

        let engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let (pk, vk) = fib.keygen(&engine);
        let mut recorder = DuplexSpongeRecorder::default();
        let proof = fib.prove_from_transcript(&engine, &pk, &mut recorder);

        let mut validator_sponge = DuplexSpongeValidator::new(recorder.history);
        verify(&vk, &proof, &mut validator_sponge)
    }

    #[test]
    fn test_dummy_interactions_roundtrip() -> Result<(), VerifierError> {
        let params = test_system_params_small();
        let engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let fx = InteractionsFixture11;
        let (pk, vk) = fx.keygen(&engine);

        let mut recorder = DuplexSpongeRecorder::default();
        let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

        let mut validator_sponge = DuplexSpongeValidator::new(recorder.history);
        verify(&vk, &proof, &mut validator_sponge)
    }

    #[test]
    fn test_cached_trace_roundtrip() -> Result<(), VerifierError> {
        let params = test_system_params_small();
        let engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let fx = CachedFixture11::new(params);
        let (pk, vk) = fx.keygen(&engine);

        let mut recorder = DuplexSpongeRecorder::default();
        let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

        let mut validator_sponge = DuplexSpongeValidator::new(recorder.history);
        verify(&vk, &proof, &mut validator_sponge)
    }

    #[test]
    fn test_preprocessed_trace_roundtrip() -> Result<(), VerifierError> {
        use itertools::Itertools;
        let params = test_system_params_small();
        let engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let log_trace_degree = 8;
        let height = 1 << log_trace_degree;
        let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
        let fx = PreprocessedFibFixture::new(0, 1, sels);
        let (pk, vk) = fx.keygen(&engine);

        let mut recorder = DuplexSpongeRecorder::default();
        let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

        let mut validator_sponge = DuplexSpongeValidator::new(recorder.history);
        verify(&vk, &proof, &mut validator_sponge)
    }
}
