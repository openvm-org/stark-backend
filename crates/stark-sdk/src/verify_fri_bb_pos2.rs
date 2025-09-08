//! Single file with full verification algorithm using FRI
//! for BabyBear, Poseidon2

use std::{array::from_fn, iter::zip, sync::OnceLock};

use itertools::Itertools;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_field::{
        extension::BinomialExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra,
        PrimeField32, TwoAdicField,
    },
    p3_util::log2_strict_usize,
    verifier::VerificationError::{self, InvalidProofShape},
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_symmetric::Permutation;

use crate::config::{
    baby_bear_poseidon2::{default_perm, BabyBearPoseidon2Config},
    FriParameters,
};

const WIDTH: usize = 16;
const CHUNK: usize = 8;
const D: usize = 4;
const MAX_TWO_ADICITY: usize = 27;

type VerifyingKey = MultiStarkVerifyingKey<BabyBearPoseidon2Config>;
type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

// We avoid using the p3 Proof<SC> struct to avoid associated types.
// Instead we unroll the struct without generics so all types are concrete.
//
// We specialize to the case with one after-challenge round, which we will call `perm` (i.e.,
// after_challenge[0] =: perm).
pub struct Proof {
    // Commitments
    pub cached_main_commitments: Vec<[F; CHUNK]>,
    pub common_main_commitment: [F; CHUNK],
    pub perm_commitment: [F; CHUNK],
    pub quotient_commitment: [F; CHUNK],

    // FRI PCS opening proof
    pub fri_proof: FriProof,
    /// - each round (preprocessed, cached_main[..], common_main, perm, quotient)
    ///   - each matrix
    ///     - each out-of-domain point to open at
    ///       - evaluation for each column poly at that point, valued in EF
    pub opened_values: Vec<Vec<Vec<Vec<EF>>>>,
    /// Per-AIR proof data
    pub per_air: Vec<AirProofData>,
    /// Logup proof-of-work witness from grinding
    pub logup_pow_witness: F,
}

pub struct FriProof {
    /// Merkle roots from intermediate FRI folds
    pub commit_phase_commits: Vec<[F; CHUNK]>,
    /// Length is num queries
    pub query_proofs: Vec<QueryProof>,
    /// Final polynomial at end of FRI should be a constant
    pub final_poly: EF,
    /// Proof-of-work witness from grinding
    pub pow_witness: F,
}

pub struct QueryProof {
    pub input_proof: Vec<BatchOpening<F>>,
    pub commit_phase_openings: Vec<CommitPhaseProofStep>,
}
/// MMCS merkle proof into a single merkle root at a single query index
pub struct BatchOpening<F> {
    /// Per matrix, per column
    pub opened_values: Vec<Vec<F>>,
    /// merkle proof
    pub opening_proof: Vec<[F; CHUNK]>,
}
pub struct CommitPhaseProofStep {
    pub sibling_value: EF,
    /// merkle proof
    pub opening_proof: Vec<[F; CHUNK]>,
}

pub struct AirProofData {
    /// AIR ID in the vkey. For optional AIR handling
    pub air_id: usize,
    /// height of trace matrix.
    pub degree: usize,
    pub log_up_cumulative_sum: EF,
    // The public values to expose to the verifier
    pub public_values: Vec<F>,
}

/// Poseidon2 sponge state.
/// Duplex refers to having both observe (absorb) index (from 0->7) and sample index (from 8->0).
/// When sampling, sample at `sample_idx - 1 >= 0` or squeeze first and then sample.
pub struct DuplexSponge {
    /// Poseidon2 state
    state: [F; WIDTH],
    /// Invariant to be preserved: 0 <= absorb_idx < CHUNK
    absorb_idx: usize,
    /// Invariant to be preserved: 0 <= sample_idx <= CHUNK
    sample_idx: usize,
}

// Fixed Poseidon2 configuration
static PERM: OnceLock<Poseidon2BabyBear<WIDTH>> = OnceLock::new();
fn poseidon2_perm() -> &'static Poseidon2BabyBear<WIDTH> {
    PERM.get_or_init(|| default_perm())
}

impl DuplexSponge {
    pub fn observe(&mut self, value: F) {
        self.state[self.absorb_idx] = value;
        self.absorb_idx += 1;
        if self.absorb_idx == CHUNK {
            poseidon2_perm().permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = CHUNK;
        }
        self.absorb_idx = (self.absorb_idx + 1) % CHUNK;
    }

    pub fn sample(&mut self) -> F {
        if self.absorb_idx != 0 || self.sample_idx == 0 {
            poseidon2_perm().permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = CHUNK;
        }
        self.sample_idx -= 1;
        self.state[self.sample_idx]
    }

    pub fn observe_ext(&mut self, value: EF) {
        // for i in 0..D
        for &base_val in value.as_base_slice() {
            self.observe(base_val);
        }
    }

    pub fn sample_ext(&mut self) -> EF {
        let slice: [F; D] = from_fn(|_| self.sample());
        EF::from_base_slice(&slice)
    }

    pub fn sample_bits(&mut self, bits: usize) -> u32 {
        assert!(bits < (u32::BITS as usize));
        assert!((1 << bits) < F::ORDER_U32);
        let rand_f: F = self.sample();
        let rand_u32 = rand_f.as_canonical_u32();
        rand_u32 & ((1 << bits) - 1)
    }
}

pub fn verify(mvk: &VerifyingKey, proof: Proof) -> Result<(), VerificationError> {
    Ok(())
}

/// Polynomial commitment opening of `rounds` via FRI over BabyBear (2-adic pcs)
pub fn verify_fri_pcs(
    params: &FriParameters,
    // For each commitment:
    per_commitment: Vec<(
        // Commitment
        [F; CHUNK],
        // for each matrix:
        Vec<(
            // matrix trace height
            usize,
            // for each point:
            Vec<(
                // the out-of-domain point to evaluate at,
                EF,
                // evaluations at the point for each column polynomial of the matrix
                Vec<EF>,
            )>,
        )>,
    )>,
    proof: FriProof,
    sponge: &mut DuplexSponge,
) -> Result<(), VerificationError> {
    // Write evaluations to challenger
    for (_commitment, per_mat) in per_commitment.iter() {
        for (_trace_height, per_ood_point) in per_mat.iter() {
            for (_ood_point, per_col) in per_ood_point.iter() {
                for &evaluation in per_col.iter() {
                    sponge.observe_ext(evaluation);
                }
            }
        }
    }

    // Batch combination challenge
    let alpha: EF = sponge.sample_ext();

    // !Unvalidated: height is based on merkle proof length
    // - global refers to fact that it's max height across matrices across all commitments
    // `commit_phase_commits.len()` is the number of folding steps, so the maximum polynomial degree
    // will be `commit_phase_commits.len() + self.fri.log_final_poly_len` and so, as the same
    // blow-up is used for all polynomials, the log of the maximum matrix height is:
    let log_global_max_height = proof.commit_phase_commits.len() + params.log_blowup; // + params.log_final_poly_len;

    // Generate all of the random challenges for the FRI rounds.
    let betas: Vec<EF> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            for i in 0..CHUNK {
                sponge.observe(comm[i]);
            }
            sponge.sample_ext()
        })
        .collect();

    // Ensure that the final polynomial has the expected degree.
    if 0 != params.log_final_poly_len {
        return Err(InvalidProofShape);
    }

    // Observe final polynomial (= constant).
    sponge.observe_ext(proof.final_poly);

    // Ensure that we have the expected number of FRI query proofs.
    if proof.query_proofs.len() != params.num_queries {
        return Err(InvalidProofShape);
    }

    // Check PoW.
    {
        sponge.observe(proof.pow_witness);
        if sponge.sample_bits(params.proof_of_work_bits) != 0 {
            return Err(VerificationError::InvalidOpeningArgument(
                "InvalidPoW".to_string(),
            ));
        }
    }

    // The log of the final domain size.
    let log_final_height = params.log_blowup; // + params.log_final_poly_len = 0;

    for QueryProof {
        input_proof,
        commit_phase_openings,
    } in &proof.query_proofs
    {
        if proof.commit_phase_commits.len() != commit_phase_openings.len() {
            return Err(InvalidProofShape);
        }
        if input_proof.len() != per_commitment.len() {
            return Err(InvalidProofShape);
        }

        // For each query proof, we start by generating the random index.
        let index = sponge.sample_bits(log_global_max_height); // p3 note: folding.extra_query_index_bits() = 0;

        // ================= FRI Reduced Opening ============================
        // Next we open all polynomials `f` at the relevant index and combine them into our FRI
        // inputs.
        // For each log_height, we store the alpha power and compute the reduced opening.
        // log_height -> (alpha_pow, reduced_opening)
        let mut reduced_openings = [EF::ZERO; MAX_TWO_ADICITY + 1];
        let mut alpha_pows = [EF::ONE; MAX_TWO_ADICITY + 1];

        // Reminder: input_proof contains poly evaluations on in-domain points that changes per
        // query; per_commitment contains poly evaluations on out-of-domain points independent of
        // query

        // For each batch commitment and opening proof
        for (batch_opening, (batch_commit, per_mat)) in zip(input_proof, &per_commitment) {
            // number of matrices in the batch
            if batch_opening.opened_values.len() != per_mat.len() {
                return Err(InvalidProofShape);
            }
            // Find the height of each matrix in the batch.
            // Currently we only check domain.size() as the shift is
            // assumed to always be Val::GENERATOR.
            // !Unsorted
            let batch_heights = per_mat
                .iter()
                .map(|(trace_height, _)| trace_height << params.log_blowup)
                .collect_vec();

            // If the maximum height of the batch is smaller than the global max height,
            // we need to correct the index by right shifting it.
            // If the batch is empty, we set the index to 0.
            let reduced_index = batch_heights
                .iter()
                .max()
                .map(|&h| index >> (log_global_max_height - log2_strict_usize(h)))
                .unwrap_or(0);

            verify_batch(
                *batch_commit,
                &batch_heights,
                reduced_index,
                batch_opening.into(),
            )?;

            // For each matrix in the commitment
            for (mat_opening, (trace_height, per_ood_point)) in
                zip(&batch_opening.opened_values, per_mat)
            {
                // log(lde_height)
                let log_height = log2_strict_usize(*trace_height) + params.log_blowup;
                if log_height > MAX_TWO_ADICITY {
                    return Err(InvalidProofShape);
                }

                let bits_reduced = log_global_max_height - log_height;
                let rev_reduced_index = (index >> bits_reduced)
                    .reverse_bits()
                    .overflowing_shr(u32::BITS - log_height as u32)
                    .0;

                // Compute gh^i
                let x = F::GENERATOR
                    * F::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

                let alpha_pow = &mut alpha_pows[log_height];
                let ro = &mut reduced_openings[log_height];

                // For each polynomial `f` in our matrix, compute `(f(z) - f(x))/(z - x)`,
                // scale by the appropriate alpha power and add to the reduced opening for this
                // log_height.
                for (z, ps_at_z) in per_ood_point {
                    // number of columns in the matrix
                    if mat_opening.len() != ps_at_z.len() {
                        return Err(InvalidProofShape);
                    }
                    // z is the out-of-domain point
                    // ps_at_z are evaluations of column polys at z
                    let quotient = (*z - x).inverse();
                    for (&p_at_x, &p_at_z) in zip(mat_opening, ps_at_z) {
                        // Note we just checked batch proofs to ensure p_at_x is correct.
                        // x, z were sent by the verifier.
                        // ps_at_z was sent to the verifier and we are using fri to prove it is
                        // correct.
                        *ro += *alpha_pow * (p_at_z - p_at_x) * quotient;
                        *alpha_pow *= alpha;
                    }
                }
            }

            // `reduced_openings` would have a log_height = log_blowup entry only if there was a
            // trace matrix of height 1. In this case `f` is constant, so `f(zeta) - f(x))/(zeta -
            // x)` must equal `0`.
            if !reduced_openings[params.log_blowup].is_zero() {
                return Err(VerificationError::InvalidOpeningArgument(
                    "FinalPolyMismatch".to_string(),
                ));
            }
        }

        // Return reduced openings descending by log_height.
        let ro: Vec<(usize, EF)> = reduced_openings
            .into_iter()
            .take(log_global_max_height + 1)
            .enumerate()
            .rev()
            .collect();

        let mut domain_index = index; // folding.extra_query_index_bits() = 0;

        // Starting at the evaluation at `index` of the initial domain,
        // perform FRI folds until the domain size reaches the final domain size.
        // Check after each fold that the pair of sibling evaluations at the current
        // node match the commitment.
        let folded_eval = verify_query(
            params,
            &mut domain_index,
            zip(
                zip(&betas, &proof.commit_phase_commits), /* betas.len() =
                                                           * proof.commit_phase_commits.len() by
                                                           * definition */
                commit_phase_openings,
            ),
            ro,
            log_global_max_height,
            log_final_height,
        )?;

        // Assuming all the checks passed, the final check is to ensure that the folded evaluation
        // matches the evaluation of the final polynomial sent by the prover.

        // This is simplified because we required final_poly to be a constant:
        let eval = proof.final_poly;

        if eval != folded_eval {
            return Err(VerificationError::InvalidOpeningArgument(
                "FinalPolyMismatch".to_string(),
            ));
        }
    }
    Ok(())
}
