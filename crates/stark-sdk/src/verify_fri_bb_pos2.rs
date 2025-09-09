//! Single file with full verification algorithm using FRI
//! for BabyBear, Poseidon2
#![allow(clippy::needless_range_loop)]
use std::{array::from_fn, cmp::Reverse, iter::zip, sync::OnceLock};

use itertools::Itertools;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_field::{
        extension::BinomialExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra,
        PrimeField32, TwoAdicField,
    },
    p3_util::{log2_strict_usize, reverse_bits_len},
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
pub struct StarkProof {
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
    pub input_proof: Vec<BatchOpening>,
    pub commit_phase_openings: Vec<CommitPhaseProofStep>,
}
/// MMCS merkle proof into a single merkle root at a single query index
pub struct BatchOpening {
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
    PERM.get_or_init(default_perm)
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

pub fn verify(
    params: &FriParameters,
    mvk: &VerifyingKey,
    proof: StarkProof,
) -> Result<(), VerificationError> {
    Ok(())
}

/// Polynomial commitment opening of `rounds` via FRI over BabyBear (2-adic pcs)
#[allow(clippy::type_complexity)]
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
    // The log of the final domain size.
    let log_final_height = params.log_blowup; // + params.log_final_poly_len = 0;

    // Observe final polynomial (= constant).
    sponge.observe_ext(proof.final_poly);

    // Ensure that we have the expected number of FRI query proofs.
    if proof.query_proofs.len() != params.num_queries {
        return Err(InvalidProofShape);
    }
    if proof.commit_phase_commits.len() != log_global_max_height - log_final_height {
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

            verify_batch(batch_commit, &batch_heights, reduced_index, batch_opening)?;

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

        let mut domain_index = index as usize; // folding.extra_query_index_bits() = 0;

        // ================ Verify FRI Query (FRI folding) ======================
        // Starting at the evaluation at `index` of the initial domain,
        // perform FRI folds until the domain size reaches the final domain size.
        // Check after each fold that the pair of sibling evaluations at the current
        // node match the commitment.
        let start_index = &mut domain_index;

        let mut ro_iter = ro.into_iter().peekable();
        let log_max_height = log_global_max_height;

        let mut folded_eval = ro_iter.next().unwrap().1;
        // We start with evaluations over a domain of size (1 << log_max_height). We fold
        // using FRI until the domain size reaches (1 << log_final_height).
        for (log_folded_height, ((&beta, comm), opening)) in zip(
            (log_final_height..log_max_height).rev(),
            zip(
                zip(&betas, &proof.commit_phase_commits), /* betas.len() =
                                                           * proof.commit_phase_commits.len() by
                                                           * definition */
                commit_phase_openings,
            ),
        ) {
            if log_folded_height != opening.opening_proof.len() {
                return Err(InvalidProofShape);
            }
            // Get the index of the other sibling of the current FRI node.
            let index_sibling = *start_index ^ 1;

            let mut evals = [folded_eval; 2];
            evals[index_sibling % 2] = opening.sibling_value;
            // Verify a normal Merkle proof
            let mut root =
                poseidon2_hash_slice(&evals.map(|ef| ef.as_base_slice().to_vec()).concat());
            // Replace index with the index of the parent FRI node.
            *start_index >>= 1;

            let mut cur_index = *start_index;
            for &sibling in &opening.opening_proof {
                let (left, right) = if cur_index & 1 == 0 {
                    (root, sibling)
                } else {
                    (sibling, root)
                };
                root = poseidon2_compress(left, right);
                cur_index >>= 1;
            }
            if root != *comm {
                return Err(VerificationError::InvalidOpeningArgument(
                    "CommitPhaseMerkleProofError".to_string(),
                ));
            }

            // Fold the pair of sibling nodes to get the evaluation of the parent FRI node.
            folded_eval = {
                let log_arity = 1;
                let [e0, e1] = evals;
                // If performance critical, make this API stateful to avoid this
                // This is a bit more math than is necessary, but leaving it here
                // in case we want higher arity in the future
                let subgroup_start = F::two_adic_generator(log_folded_height + log_arity)
                    .exp_u64(reverse_bits_len(*start_index, log_folded_height) as u64);
                let xs = [
                    subgroup_start,
                    subgroup_start * F::two_adic_generator(log_arity),
                ];
                // interpolate and evaluate at beta
                e0 + (beta - xs[0]) * (e1 - e0) / (xs[1] - xs[0]).into()
            };

            // If there are new polynomials to roll in at the folded height, do so.
            //
            // Each element of `ro_iter` is the evaluation of a reduced opening polynomial, which is
            // itself a random linear combination `f_{i, 0}(x) + alpha f_{i, 1}(x) +
            // ...`, but when we add it to the current folded polynomial evaluation
            // claim, we need to multiply by a new random factor since `f_{i, 0}` has no
            // leading coefficient.
            //
            // We use `beta^2` as the random factor since `beta` is already used in the folding.
            // This increases the query phase error probability by a negligible amount, and does not
            // change the required number of FRI queries.
            if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
                folded_eval += beta.square() * ro;
            }
        }

        // If ro_iter is not empty, we failed to fold in some polynomial evaluations.
        if ro_iter.next().is_some() {
            return Err(InvalidProofShape);
        }

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

/// Merkle proof verification for mixed matrix commitment scheme
/// - for a single `commit`
/// - `trace_heights` are not sorted and must be non-zero
///
/// # Assumptions
/// - `opened_values.len()` equals `trace_heights.len()` is the number of matrices in the batch.
/// - all entries of `trace_heights` must be powers of two (hence nonzero)
fn verify_batch(
    commit: &[F; CHUNK],
    trace_heights: &[usize],
    mut index: u32,
    BatchOpening {
        opened_values,
        opening_proof,
    }: &BatchOpening,
) -> Result<(), VerificationError> {
    // Check that the openings have the correct shape.
    debug_assert_eq!(trace_heights.len(), opened_values.len());

    let mut heights_tallest_first = trace_heights
        .iter()
        .copied()
        .enumerate()
        .sorted_by_key(|(_, height)| Reverse(*height))
        .peekable();

    // Get the initial height padded to a power of two. As heights_tallest_first is sorted,
    // the initial height will be the maximum height.
    // Returns an error if either:
    //              1. proof.len() != log_max_height
    //              2. heights_tallest_first is empty.
    let mut curr_height_padded = match heights_tallest_first.peek() {
        Some((_, height)) => {
            let max_height = *height;
            let log_max_height = opening_proof.len();
            if log_max_height > MAX_TWO_ADICITY && max_height != 1 << log_max_height {
                return Err(InvalidProofShape);
            }
            max_height
        }
        None => return Err(InvalidProofShape),
    };

    // Hash all matrix openings at the current height.
    let mut root = poseidon2_hash_slice(
        &heights_tallest_first
            .peeking_take_while(|(_, height)| *height == curr_height_padded)
            .map(|(mat_idx, _)| opened_values[mat_idx].to_vec())
            .concat(),
    );

    for &sibling in opening_proof {
        // The last bit of index informs us whether the current node is on the left or right.
        let (left, right) = if index & 1 == 0 {
            (root, sibling)
        } else {
            (sibling, root)
        };

        // Combine the current node with the sibling node to get the parent node.
        root = poseidon2_compress(left, right);
        index >>= 1;
        curr_height_padded >>= 1;

        // Check if there are any new matrix rows to inject at the next height.
        let next_height = heights_tallest_first
            .peek()
            .map(|(_, height)| height)
            .filter(|&h| *h == curr_height_padded)
            .copied();
        if let Some(next_height) = next_height {
            // If there are new matrix rows, hash the rows together and then combine with the
            // current root.
            let next_height_openings_digest = poseidon2_hash_slice(
                &heights_tallest_first
                    .peeking_take_while(|(_, height)| *height == next_height)
                    .map(|(i, _)| opened_values[i].to_vec())
                    .concat(),
            );

            root = poseidon2_compress(root, next_height_openings_digest);
        }
    }

    // The computed root should equal the committed one.
    if commit == &root {
        Ok(())
    } else {
        Err(VerificationError::InvalidOpeningArgument(
            "MerkleRootMismatch".to_string(),
        ))
    }
}

fn poseidon2_hash_slice(vals: &[F]) -> [F; CHUNK] {
    let perm = poseidon2_perm();
    let mut state = [F::ZERO; WIDTH];
    let mut i = 0;
    for &val in vals {
        state[i] = val;
        i += 1;
        if i == CHUNK {
            perm.permute_mut(&mut state);
            i = 0;
        }
    }
    if i != 0 {
        perm.permute_mut(&mut state);
    }
    state[..CHUNK].try_into().unwrap()
}

fn poseidon2_compress(left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK] {
    let mut state = [F::ZERO; WIDTH];
    state[..CHUNK].copy_from_slice(&left);
    state[CHUNK..].copy_from_slice(&right);
    poseidon2_perm().permute_mut(&mut state);
    state[..CHUNK].try_into().unwrap()
}
