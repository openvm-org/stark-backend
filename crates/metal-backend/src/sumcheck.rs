// Sumcheck prover
#![allow(dead_code)]

use openvm_metal_common::{copy::MemCopyH2D, d_buffer::MetalBuffer};
use openvm_stark_backend::{
    p3_util,
    p3_util::log2_strict_usize,
    poly_common::UnivariatePoly,
    prover::sumcheck::{SumcheckCubeProof, SumcheckPrismProof},
    FiatShamirTranscript,
};
use p3_field::{ExtensionField, Field};
use tracing::{debug, info_span, instrument};

use crate::{
    metal::{
        batch_ntt_small::batch_ntt_small,
        matrix::batch_expand_pad_wide,
        sumcheck::{
            fold_mle_single, fold_ple_from_coeffs, reduce_over_x_and_cols,
            sumcheck_mle_round_single,
        },
    },
    prelude::*,
    sponge::DuplexSpongeMetal,
};

/// Metal implementation of multilinear sumcheck
///
/// Memory strategy: Ping-pong buffers (buffer_a <-> buffer_b) alternate each round
/// - Round 0 (even): reads buffer_a, writes buffer_b
/// - Round 1 (odd):  reads buffer_b, writes buffer_a
/// - Final result in buffer determined by parity of n
/// - Memory footprint: ~1.5 * evals.len() * sizeof(EF)
#[instrument(name = "sumcheck_multilinear_metal", level = "info", skip_all)]
pub fn sumcheck_multilinear_metal<F: Field>(
    transcript: &mut DuplexSpongeMetal,
    evals: &[F],
) -> (SumcheckCubeProof<EF>, Vec<EF>)
where
    EF: ExtensionField<F>,
{
    let n = log2_strict_usize(evals.len());
    let mut round_polys_eval = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);

    // Compute sum claim
    let sum_claim: EF = evals.iter().copied().sum::<F>().into();
    transcript.observe_ext(sum_claim);

    // Convert to extension field
    let evals_ext: Vec<EF> = evals.iter().map(|&x| EF::from(x)).collect();

    // Setup
    let mut current_height = 1 << n;
    let d = 1;
    const WD: usize = 1; // Number of output polynomials

    // Allocate ping-pong buffers
    let total_size = current_height;
    let d_buffer_a = MetalBuffer::<EF>::with_capacity(total_size);
    evals_ext.copy_to(&d_buffer_a).unwrap();
    let d_buffer_b = MetalBuffer::<EF>::with_capacity(total_size / 2);

    // Buffer for round output [d * WD]
    let d_round_output = MetalBuffer::<EF>::with_capacity(d * WD);

    // Sumcheck rounds
    for round in 0..n {
        // Ping-pong buffers
        let (input_buf, output_buf) = if round % 2 == 0 {
            (&d_buffer_a, &d_buffer_b)
        } else {
            (&d_buffer_b, &d_buffer_a)
        };

        // Call sumcheck_mle_round kernel (uses output_buf as tmp)
        unsafe {
            sumcheck_mle_round_single(
                input_buf,
                &d_round_output,
                output_buf, // Reuse output buffer as tmp!
                current_height as u32,
            )
            .unwrap();
        }

        // Read result back (unified memory: direct read)
        let h_round_output = d_round_output.to_vec();

        // Observe in transcript
        let s = h_round_output[0..d].to_vec();

        assert_eq!(s.len(), d);
        transcript.observe_ext(s[0]);
        round_polys_eval.push(s);

        // Sample challenge from transcript
        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);

        // Fold using the challenge
        let output_height = (current_height >> 1) as u32;
        unsafe {
            fold_mle_single(input_buf, output_buf, output_height, r_round).unwrap();
        }

        current_height >>= 1;
    }

    // After all rounds, get final evaluation claim
    let final_buf = if n % 2 == 1 { &d_buffer_b } else { &d_buffer_a };
    let eval_claim_vec = final_buf.to_vec();
    let eval_claim = eval_claim_vec[0];

    transcript.observe_ext(eval_claim);

    (
        SumcheckCubeProof {
            sum_claim,
            round_polys_eval,
            eval_claim,
        },
        r,
    )
}

/// Metal implementation of prismalinear sumcheck with univariate skip
///
/// Memory strategy:
/// - Round 0: Uses DFT/iDFT pipeline, reuses buffers between steps
/// - Rounds 1..n: Standard MLE rounds with ping-pong buffers
/// - d_evals gets modified by iDFT in round 0, then reused for fold_ple
/// - Memory footprint: if evals.len() = 2^(l_skip +n), then 2 * evals.len() * sizeof(F) + 1.5 * 2^n
///   * sizeof(EF)
///
/// NOTE: batch_ntt expects a concrete type BabyBear, so generic type parameters are removed
#[instrument(name = "sumcheck_prismalinear_metal", level = "info", skip_all)]
pub fn sumcheck_prismalinear_metal(
    transcript: &mut DuplexSpongeMetal,
    l_skip: usize,
    evals: &[F],
) -> (SumcheckPrismProof<EF>, Vec<EF>) {
    let prism_dim = p3_util::log2_strict_usize(evals.len());
    assert!(prism_dim >= l_skip);
    let n = prism_dim - l_skip;

    let mut round_polys_eval = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n + 1);

    // Compute sum claim
    let sum_claim: EF = evals.iter().copied().sum::<F>().into();
    transcript.observe_ext(sum_claim);

    // Setup
    let domain_size = 1 << l_skip;
    let num_x = 1 << n;
    let width = 1;
    let d = 1;
    let s_deg = d * (domain_size - 1);
    let log_large_domain = p3_util::log2_ceil_usize(s_deg + 1);
    let large_domain_size = 1 << log_large_domain;

    // ========== Round 0: Special PLE round ==========
    let _round0_span = info_span!("sumcheck_prismalinear.round0").entered();

    let mut d_coeffs = evals.to_device();
    let mut d_s0_coeffs = MetalBuffer::<F>::with_capacity(large_domain_size);

    // Step 1: iDFT on skip domain (reinterpreting dimensions).
    unsafe {
        batch_ntt_small(&mut d_coeffs, l_skip, num_x * width, true).unwrap();
    }

    if domain_size == large_domain_size {
        unsafe {
            reduce_over_x_and_cols(
                &d_coeffs,
                &d_s0_coeffs,
                num_x as u32,
                width as u32,
                large_domain_size as u32,
            )
            .unwrap();
        }
    } else {
        let mut d_coeffs_large = MetalBuffer::<F>::with_capacity(num_x * width * large_domain_size);

        unsafe {
            batch_expand_pad_wide(
                &d_coeffs_large,
                0,
                &d_coeffs,
                0,
                (num_x * width) as u32,
                large_domain_size as u32,
                domain_size as u32,
            )
            .unwrap();
            batch_ntt_small(&mut d_coeffs_large, log_large_domain, num_x * width, false).unwrap();
            reduce_over_x_and_cols(
                &d_coeffs_large,
                &d_s0_coeffs,
                num_x as u32,
                width as u32,
                large_domain_size as u32,
            )
            .unwrap();
            batch_ntt_small(&mut d_s0_coeffs, log_large_domain, 1, true).unwrap();
        }
    }

    let s0_coeffs_host: Vec<F> = d_s0_coeffs.to_vec();
    let s0_coeffs_ext: Vec<EF> = s0_coeffs_host[0..=s_deg]
        .iter()
        .map(|&x| EF::from(x))
        .collect();
    let s_0 = UnivariatePoly::new(s0_coeffs_ext.clone());
    for &coeff in &s0_coeffs_ext {
        transcript.observe_ext(coeff);
    }

    let r_0 = transcript.sample_ext();
    debug!(round = 0, r_round = %r_0);
    r.push(r_0);

    let d_folded = MetalBuffer::<EF>::with_capacity(num_x);
    unsafe {
        fold_ple_from_coeffs(
            &d_coeffs,
            &d_folded,
            num_x as u32,
            width as u32,
            domain_size as u32,
            r_0,
        )
        .unwrap();
    }
    drop(d_coeffs);
    drop(_round0_span);

    // ========== Rounds 1..n: Regular MLE rounds ==========
    let _mle_rounds_span = info_span!("sumcheck_prismalinear.mle_rounds").entered();
    let mut current_height = num_x;
    let d_buffer_a = d_folded;
    let d_buffer_b = MetalBuffer::<EF>::with_capacity(current_height / 2);
    let d_round_output = MetalBuffer::<EF>::with_capacity(d);

    for round in 1..=n {
        let (input_buf, output_buf) = if round % 2 == 1 {
            (&d_buffer_a, &d_buffer_b)
        } else {
            (&d_buffer_b, &d_buffer_a)
        };

        unsafe {
            sumcheck_mle_round_single(
                input_buf,
                &d_round_output,
                output_buf,
                current_height as u32,
            )
            .unwrap();
        }

        let h_round_output = d_round_output.to_vec();
        let s = h_round_output[0..d].to_vec();
        assert_eq!(s.len(), d);
        transcript.observe_ext(s[0]);
        round_polys_eval.push(s);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);

        unsafe {
            fold_mle_single(input_buf, output_buf, (current_height >> 1) as u32, r_round).unwrap();
        }

        current_height >>= 1;
    }
    drop(_mle_rounds_span);

    let final_buf = if n % 2 == 1 { &d_buffer_b } else { &d_buffer_a };
    let eval_claim_vec = final_buf.to_vec();
    let eval_claim = eval_claim_vec[0];

    transcript.observe_ext(eval_claim);

    (
        SumcheckPrismProof {
            sum_claim,
            s_0,
            round_polys_eval,
            eval_claim,
        },
        r,
    )
}
