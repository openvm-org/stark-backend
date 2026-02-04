use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_util::log2_strict_usize;
use tracing::debug;

use crate::{
    poseidon2::sponge::FiatShamirTranscript,
    prover::sumcheck::{SumcheckCubeProof, SumcheckPrismProof},
    EF,
};

pub fn verify_sumcheck_multilinear<F: Field, TS: FiatShamirTranscript>(
    transcript: &mut TS,
    proof: &SumcheckCubeProof<EF>,
) -> Result<(), String>
where
    EF: ExtensionField<F>,
{
    let SumcheckCubeProof {
        sum_claim,
        round_polys_eval,
        eval_claim,
    } = proof;
    let n = round_polys_eval.len();

    let mut cur_sum = *sum_claim;
    #[allow(clippy::needless_range_loop)]
    for round in 0..n {
        assert_eq!(round_polys_eval[round].len(), 1);
        let s_1 = round_polys_eval[round][0];
        let s_0 = cur_sum - s_1;

        if round == 0 {
            transcript.observe_ext(*sum_claim);
        }
        transcript.observe_ext(s_1);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        cur_sum = s_0 + (s_1 - s_0) * r_round;
    }
    if cur_sum != *eval_claim {
        return Err("The provided evaluations are inconsistent".to_string());
    }

    transcript.observe_ext(*eval_claim);

    Ok(())
}

pub fn verify_sumcheck_prismalinear<F: Field, TS: FiatShamirTranscript>(
    transcript: &mut TS,
    l_skip: usize,
    proof: &SumcheckPrismProof<EF>,
) -> Result<(), String>
where
    EF: ExtensionField<F>,
{
    let SumcheckPrismProof {
        sum_claim,
        s_0,
        round_polys_eval,
        eval_claim,
    } = proof;
    let n = round_polys_eval.len();

    if log2_strict_usize(s_0.0.len()) != l_skip {
        return Err(format!(
            "Wrong proof shape: `s_0` must have length 2^l_skip = {}, but has {}",
            (1 << l_skip),
            s_0.0.len()
        ));
    }

    transcript.observe_ext(*sum_claim);
    for x in s_0.0.iter() {
        transcript.observe_ext(*x);
    }
    let r_0 = transcript.sample_ext();
    debug!(round = 0, r_round = %r_0);

    if *sum_claim != s_0.0[0] * EF::from_usize(s_0.0.len()) {
        return Err(format!(
            "`sum_claim` does not equal the sum of `s_0` at all the roots of unity: {} != {}",
            *sum_claim,
            s_0.0[0] * EF::from_usize(s_0.0.len())
        ));
    }

    let mut cur_sum = s_0.eval_at_point(r_0);
    #[allow(clippy::needless_range_loop)]
    for round in 0..n {
        debug!(%round, %cur_sum);
        assert_eq!(round_polys_eval[round].len(), 1);
        let s_1 = round_polys_eval[round][0];
        let s_0 = cur_sum - s_1;

        transcript.observe_ext(s_1);
        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        cur_sum = s_0 + (s_1 - s_0) * r_round;
    }

    if cur_sum != *eval_claim {
        return Err("The provided evaluations are inconsistent".to_string());
    }

    transcript.observe_ext(*eval_claim);

    Ok(())
}
