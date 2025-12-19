use core::cmp::Reverse;

use itertools::{izip, Itertools};
use p3_field::{FieldAlgebra, TwoAdicField};
use thiserror::Error;

use crate::{
    keygen::types::{MultiStarkVerifyingKey0V2, MultiStarkVerifyingKeyV2},
    poly_common::Squarable,
    poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
    verifier::{
        batch_constraints::{verify_zerocheck_and_logup, BatchConstraintError},
        proof_shape::{verify_proof_shape, ProofShapeError},
        stacked_reduction::{verify_stacked_reduction, StackedReductionError},
        whir::{verify_whir, VerifyWhirError},
    },
    F,
};

#[derive(Error, Debug, PartialEq, Eq)]
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
    } = &mvk;
    let l_skip = params.l_skip;

    let num_airs = per_air.len();

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
    let num_traces = trace_vdata.iter().flatten().collect_vec().len();
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

    let omega_skip = F::two_adic_generator(l_skip);
    let omega_skip_pows = omega_skip.powers().take(1 << l_skip).collect_vec();

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
                transcript.observe(F::from_canonical_usize(trace_vdata.log_height));
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

    let n_per_trace: Vec<isize> = trace_id_to_air_id
        .iter()
        .map(|&air_id| trace_vdata[air_id].as_ref().unwrap().log_height as isize - l_skip as isize)
        .collect();
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
    use openvm_stark_sdk::config::{
        log_up_params::log_up_security_params_baby_bear_100_bits, setup_tracing_with_log_level,
    };
    use test_case::test_case;
    use tracing::Level;

    use crate::{
        poseidon2::sponge::{DuplexSpongeRecorder, TranscriptHistory},
        test_utils::{
            test_system_params_small, CachedFixture11, DuplexSpongeValidator, FibFixture,
            InteractionsFixture11, PreprocessedFibFixture, TestFixture,
        },
        verifier::{verify, VerifierError},
        BabyBearPoseidon2CpuEngineV2, SystemParams, WhirConfig, WhirParams,
    };

    #[test_case(2, 10)]
    #[test_case(2, 1; "where log_trace_degree=1 less than l_skip=2")]
    #[test_case(2, 0; "where log_trace_degree=0 less than l_skip=2")]
    #[test_case(3, 2; "where log_trace_degree=2 less than l_skip=3")]
    fn test_fib_air_roundtrip(l_skip: usize, log_trace_degree: usize) -> Result<(), VerifierError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let n_stack = 8;
        let k_whir = 4;
        let whir_params = WhirParams {
            k: k_whir,
            log_final_poly_len: k_whir,
            query_phase_pow_bits: 1,
        };
        let log_blowup = 1;
        let whir = WhirConfig::new(log_blowup, l_skip + n_stack, whir_params, 80);
        let params = SystemParams {
            l_skip,
            n_stack,
            log_blowup,
            whir,
            logup: log_up_security_params_baby_bear_100_bits(),
            max_constraint_degree: 3,
        };
        let fib = FibFixture::new(0, 1, 1 << log_trace_degree);

        let engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let (pk, vk) = fib.keygen(&engine);
        let mut recorder = DuplexSpongeRecorder::default();
        let proof = fib.prove_from_transcript(&engine, &pk, &mut recorder);

        let mut validator_sponge = DuplexSpongeValidator::new(recorder.into_log());
        verify(&vk, &proof, &mut validator_sponge)
    }

    #[test_case(2, 8, 3)]
    #[test_case(5, 5, 4)]
    fn test_dummy_interactions_roundtrip(
        l_skip: usize,
        n_stack: usize,
        k_whir: usize,
    ) -> Result<(), VerifierError> {
        let params = test_system_params_small(l_skip, n_stack, k_whir, 3);
        let engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let fx = InteractionsFixture11;
        let (pk, vk) = fx.keygen(&engine);

        let mut recorder = DuplexSpongeRecorder::default();
        let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

        let mut validator_sponge = DuplexSpongeValidator::new(recorder.into_log());
        verify(&vk, &proof, &mut validator_sponge)
    }

    #[test_case(2, 8, 3)]
    #[test_case(5, 5, 4)]
    #[test_case(5, 8, 3)]
    fn test_cached_trace_roundtrip(
        l_skip: usize,
        n_stack: usize,
        k_whir: usize,
    ) -> Result<(), VerifierError> {
        setup_tracing_with_log_level(Level::DEBUG);
        let params = test_system_params_small(l_skip, n_stack, k_whir, 3);
        let engine = BabyBearPoseidon2CpuEngineV2::new(params.clone());
        let fx = CachedFixture11::new(params);
        let (pk, vk) = fx.keygen(&engine);

        let mut recorder = DuplexSpongeRecorder::default();
        let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

        let mut validator_sponge = DuplexSpongeValidator::new(recorder.into_log());
        verify(&vk, &proof, &mut validator_sponge)
    }

    #[test_case(2, 8, 3)]
    #[test_case(5, 5, 4)]
    fn test_preprocessed_trace_roundtrip(
        l_skip: usize,
        n_stack: usize,
        k_whir: usize,
    ) -> Result<(), VerifierError> {
        use itertools::Itertools;
        let params = test_system_params_small(l_skip, n_stack, k_whir, 3);
        let engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let log_trace_degree = 8;
        let height = 1 << log_trace_degree;
        let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
        let fx = PreprocessedFibFixture::new(0, 1, sels);
        let (pk, vk) = fx.keygen(&engine);

        let mut recorder = DuplexSpongeRecorder::default();
        let proof = fx.prove_from_transcript(&engine, &pk, &mut recorder);

        let mut validator_sponge = DuplexSpongeValidator::new(recorder.into_log());
        verify(&vk, &proof, &mut validator_sponge)
    }
}
