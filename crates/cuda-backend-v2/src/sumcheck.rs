use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::p3_util::log2_strict_usize;
use p3_field::{ExtensionField, Field, FieldExtensionAlgebra};
use stark_backend_v2::{
    EF, poseidon2::sponge::FiatShamirTranscript, prover::sumcheck::SumcheckCubeProof,
};
use tracing::debug;

use crate::cuda::sumcheck::{fold_mle, sumcheck_mle_round};

pub fn sumcheck_multilinear_gpu<F: Field, TS: FiatShamirTranscript>(
    transcript: &mut TS,
    evals: &[F],
) -> (SumcheckCubeProof<EF>, Vec<EF>)
where
    EF: ExtensionField<F>,
{
    let n = log2_strict_usize(evals.len());
    let mut round_polys_eval = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);

    // Convert to extension field and compute sum claim
    let evals_ext: Vec<EF> = evals.iter().map(|&x| EF::from_base(x)).collect();

    let sum_claim: EF = evals.iter().fold(F::ZERO, |acc, &x| acc + x).into();
    transcript.observe_ext(sum_claim);

    // Allocate GPU buffers
    let mut current_height = 1 << n;
    let width = 1;
    let num_matrices = 1;
    let d = 1; // Degree for MLE
    const WD: usize = 1; // Number of output polynomials

    // Allocate ping-pong buffers
    let total_size = width * current_height;
    let mut d_buffer_a = evals_ext.to_device().unwrap();
    let mut d_buffer_b = DeviceBuffer::<EF>::with_capacity(total_size / 2);

    // Set up pointer arrays on device
    let mut d_input_ptrs = DeviceBuffer::<usize>::with_capacity(num_matrices);
    let mut d_output_ptrs = DeviceBuffer::<usize>::with_capacity(num_matrices);
    let d_widths = vec![width as u32].to_device().unwrap();

    // Buffer for round output [d * WD]
    let d_round_output = DeviceBuffer::<EF>::with_capacity(d * WD);

    // Sumcheck rounds
    for round in 0..n {
        // Ping-pong buffers
        let (input_buf, output_buf) = if round % 2 == 0 {
            (&d_buffer_a, &mut d_buffer_b)
        } else {
            (&d_buffer_b, &mut d_buffer_a)
        };

        // Update pointer arrays
        let input_ptr = input_buf.as_ptr() as usize;
        let output_ptr = output_buf.as_mut_ptr() as usize;

        [input_ptr].copy_to(&mut d_input_ptrs).unwrap();
        [output_ptr].copy_to(&mut d_output_ptrs).unwrap();

        // Call sumcheck_mle_round kernel (uses output_buf as tmp)
        unsafe {
            sumcheck_mle_round(
                &d_input_ptrs,
                &d_round_output,
                &output_buf, // Reuse output buffer as tmp!
                &d_widths,
                num_matrices as u32,
                current_height as u32,
                d as u32,
            )
            .unwrap();
        }

        // Copy result back to host
        let h_round_output = d_round_output.to_host().unwrap();

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
        unsafe {
            fold_mle(
                &d_input_ptrs,
                &d_output_ptrs,
                &d_widths,
                num_matrices as u32,
                (current_height >> 1) as u32,
                r_round,
            )
            .unwrap();
        }

        current_height >>= 1;
    }

    // After all rounds, get final evaluation claim
    let final_buf = if n % 2 == 1 { &d_buffer_b } else { &d_buffer_a };
    let eval_claim_vec = final_buf.to_host().unwrap();
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
