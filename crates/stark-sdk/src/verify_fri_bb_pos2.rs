//! Single file with full verification algorithm using FRI
//! for BabyBear, Poseidon2
#![allow(clippy::needless_range_loop)]
use std::{array::from_fn, cmp::Reverse, iter::zip, marker::PhantomData, sync::OnceLock};

use itertools::Itertools;
use openvm_stark_backend::{
    keygen::{types::MultiStarkVerifyingKey, view::MultiStarkVerifyingKeyView},
    p3_field::{
        extension::BinomialExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra,
        PrimeField32, TwoAdicField,
    },
    p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair},
    p3_util::{log2_strict_usize, reverse_bits_len},
    proof::Proof,
    verifier::{
        folder::VerifierConstraintFolder,
        GenericVerifierConstraintFolder,
        VerificationError::{self, InvalidProofShape},
    },
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_symmetric::Permutation;

use crate::config::{
    baby_bear_poseidon2::{default_perm, BabyBearPoseidon2Config},
    fri_params::SecurityParameters,
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
    /// NOTE[jpw]: it is likely still better to separate this by commit type:
    /// - each commit (preprocessed, cached_main[..], common_main, perm, quotient)
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
    /// log2 of height of trace matrix.
    pub log_trace_height: usize,
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
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            state: [F::ZERO; WIDTH],
            absorb_idx: 0,
            sample_idx: CHUNK,
        }
    }

    pub fn observe(&mut self, value: F) {
        self.state[self.absorb_idx] = value;
        self.absorb_idx += 1;
        if self.absorb_idx == CHUNK {
            poseidon2_perm().permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = CHUNK;
        }
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

    pub fn observe_commit(&mut self, digest: [F; CHUNK]) {
        for x in digest {
            self.observe(x);
        }
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

/// Verifies STARK proof.
///
/// The implementation of this function handles verification logic for multiple AIR traces and
/// reduction of LogUp and DEEP-ALI (quotient polynomials) to FRI PCS opening which is then done via
/// a single call to [verify_fri_pcs].
pub fn verify_stark(
    params: &SecurityParameters,
    global_vk: &VerifyingKey,
    proof: StarkProof,
) -> Result<(), VerificationError> {
    // Note: construction of view panics if any air_id exceeds number of AIRs in provided
    // `MultiStarkVerifyingKey`
    let air_ids = proof.per_air.iter().map(|p| p.air_id).collect_vec();
    // NOTE: from now on, mvk is a view of only the AIRs used in the proof
    let mvk = MultiStarkVerifyingKeyView::new(
        air_ids
            .iter()
            .map(|&id| &global_vk.inner.per_air[id])
            .collect(),
        &global_vk.inner.trace_height_constraints,
        global_vk.pre_hash,
    );
    let mut sponge = DuplexSponge::new();
    // Note: this is same as setting initial sponge state and then calling poseidon2_permute once
    sponge.observe_commit(mvk.pre_hash.into());
    let num_used_airs = air_ids.len();
    sponge.observe(F::from_canonical_usize(num_used_airs));
    for &air_id in &air_ids {
        sponge.observe(F::from_canonical_usize(air_id));
    }
    // Enforce trace height linear inequalities
    for constraint in mvk.trace_height_constraints {
        let sum = proof
            .per_air
            .iter()
            .map(|ap| (constraint.coefficients[ap.air_id] as u64) << ap.log_trace_height)
            .sum::<u64>();
        if sum >= constraint.threshold as u64 {
            return Err(VerificationError::InvalidProofShape);
        }
    }
    // (T01a): Check that all `air_id`s are different and contained in `MultiStarkVerifyingKey`
    {
        let mut air_ids = air_ids;
        air_ids.sort();
        for ids in air_ids.windows(2) {
            if ids[0] >= ids[1] {
                return Err(VerificationError::DuplicateAirs);
            }
        }
    }

    let public_values = proof
        .per_air
        .iter()
        .map(|p| p.public_values.clone())
        .collect_vec();
    // (T03a): verify shape of public values
    {
        // proof.per_air.len() = mvk.per_air.len() by definition of mvk view
        for (pvs_per_air, vk) in zip(&public_values, &mvk.per_air) {
            if pvs_per_air.len() != vk.params.num_public_values {
                return Err(VerificationError::InvalidProofShape);
            }
        }
    }
    // Challenger must observe public values
    for pvs in &public_values {
        for &pv in pvs {
            sponge.observe(pv);
        }
    }

    for preprocessed_commit in mvk.flattened_preprocessed_commits() {
        sponge.observe_commit(preprocessed_commit.into());
    }

    // (T04a): validate shapes of `main_trace_commits`:
    {
        let num_cached_mains = mvk
            .per_air
            .iter()
            .map(|vk| vk.params.width.cached_mains.len())
            .sum::<usize>();
        if proof.cached_main_commitments.len() != num_cached_mains {
            return Err(VerificationError::InvalidProofShape);
        }
    }
    // Observe main trace commitments
    for cached_main_com in &proof.cached_main_commitments {
        sponge.observe_commit(*cached_main_com);
    }
    sponge.observe_commit(proof.common_main_commitment);
    for log_trace_height in proof.per_air.iter().map(|p| p.log_trace_height) {
        sponge.observe(F::from_canonical_usize(log_trace_height));
    }

    // ====================== (Partial) Verification of logUp cumulative sums ==============
    // Partial because the PCS opening is done with FRI verify

    // Assumption: valid mvk has num_phases consistent between num_challenges_to_sample and
    // exposed_values
    let num_phases = mvk.num_phases();
    debug_assert!(num_phases <= 1, "Only support one challenge phase");
    let mut logup_sum = EF::ZERO;
    let perm_challenges: [EF; 2] = if num_phases != 0 {
        // Check logUp PoW
        sponge.observe(proof.logup_pow_witness);
        if sponge.sample_bits(params.log_up_params.log_up_pow_bits) != 0 {
            return Err(VerificationError::InvalidOpeningArgument(
                "InvalidLogUpPoW".to_string(),
            ));
        }

        let challenges = from_fn(|_| sponge.sample_ext());

        for i in 0..num_used_airs {
            if !mvk.per_air[i]
                .params
                .num_exposed_values_after_challenge
                .is_empty()
            {
                // The only exposed value is the cumulative sum for that AIR
                debug_assert_eq!(
                    mvk.per_air[i]
                        .params
                        .num_exposed_values_after_challenge
                        .len(),
                    1
                );
                debug_assert_eq!(
                    mvk.per_air[i].params.num_exposed_values_after_challenge[0],
                    1
                );
                sponge.observe_ext(proof.per_air[i].log_up_cumulative_sum);
                logup_sum += proof.per_air[i].log_up_cumulative_sum;
            }
        }
        sponge.observe_commit(proof.perm_commitment);

        challenges
    } else {
        [EF::ZERO; 2]
    };

    if logup_sum != EF::ZERO {
        return Err(VerificationError::ChallengePhaseError);
    };

    // Draw `alpha` challenge
    let alpha: EF = sponge.sample_ext();
    tracing::debug!("alpha: {alpha:?}");

    // Observe quotient commitments
    sponge.observe_commit(proof.quotient_commitment);

    // Draw `zeta` challenge
    let zeta: EF = sponge.sample_ext();
    tracing::debug!("zeta: {zeta:?}");

    // Verify all opening proofs
    let trace_height_and_adj_openings =
        |log_trace_height: usize, width: usize, zeta: EF, per_ood_point: &[Vec<EF>]| {
            if log_trace_height > MAX_TWO_ADICITY {
                return Err(InvalidProofShape);
            }
            // Only use this for adjacent openings
            if per_ood_point.len() != 2
                || per_ood_point[0].len() != width
                || per_ood_point[1].len() != width
            {
                return Err(InvalidProofShape);
            }
            let omega = F::two_adic_generator(log_trace_height);
            Ok((
                1usize << log_trace_height,
                vec![
                    (zeta, per_ood_point[0].to_vec()),
                    (zeta * omega, per_ood_point[1].to_vec()),
                ],
            ))
        };
    // Compute the out-of-domain points and attach to claimed opening values
    // 1. First the preprocessed trace openings
    // Assumption: each AIR with preprocessed trace has its own commitment and opening values
    // T05a: validate `opened_values` shape
    let mut per_commitment = Vec::new();
    let mut commit_idx = 0;
    for i in 0..num_used_airs {
        // Each preprocessed trace, if it exists, is its own commitment
        if let Some(com) = mvk.per_air[i].preprocessed_data.as_ref().map(|d| d.commit) {
            let com: [F; CHUNK] = com.into();
            if commit_idx > proof.opened_values.len() || proof.opened_values[commit_idx].len() != 1
            {
                return Err(InvalidProofShape);
            }
            per_commitment.push((
                com,
                vec![trace_height_and_adj_openings(
                    proof.per_air[i].log_trace_height,
                    mvk.per_air[i].params.width.preprocessed.unwrap_or(0),
                    zeta,
                    &proof.opened_values[commit_idx][0],
                )?],
            ));
            commit_idx += 1;
        }
    }
    // 2. Then the main trace openings
    let mut main_commit_idx = 0;
    // All commits except the last one are cached main traces.
    for i in 0..num_used_airs {
        for &cached_main_width in &mvk.per_air[i].params.width.cached_mains {
            if main_commit_idx > proof.cached_main_commitments.len() {
                return Err(InvalidProofShape);
            }
            let commit = proof.cached_main_commitments[main_commit_idx];
            if commit_idx > proof.opened_values.len() || proof.opened_values[commit_idx].len() != 1
            {
                return Err(InvalidProofShape);
            }
            per_commitment.push((
                commit,
                vec![trace_height_and_adj_openings(
                    proof.per_air[i].log_trace_height,
                    cached_main_width,
                    zeta,
                    &proof.opened_values[commit_idx][0],
                )?],
            ));
            main_commit_idx += 1;
            commit_idx += 1;
        }
    }
    if main_commit_idx != proof.cached_main_commitments.len() {
        return Err(InvalidProofShape);
    }
    let common_main_commit_idx = main_commit_idx;
    // In the last commit, each matrix corresponds to an AIR with a common main trace.
    {
        if commit_idx > proof.opened_values.len() {
            return Err(InvalidProofShape);
        }
        if proof.opened_values[commit_idx].len() != num_used_airs {
            return Err(InvalidProofShape);
        }
        let commit = proof.common_main_commitment;
        let mut per_mat = Vec::with_capacity(num_used_airs);
        for i in 0..num_used_airs {
            let width = mvk.per_air[i].params.width.common_main;
            // Currently do not support AIR without common main
            debug_assert!(width > 0);
            per_mat.push(trace_height_and_adj_openings(
                proof.per_air[i].log_trace_height,
                width,
                zeta,
                &proof.opened_values[commit_idx][i],
            )?);
        }
        per_commitment.push((commit, per_mat));
        commit_idx += 1;
    }
    // 3. Then perm commit
    // All AIRs with interactions should have perm trace.
    if num_phases != 0 {
        if commit_idx > proof.opened_values.len() {
            return Err(InvalidProofShape);
        }
        let commit = proof.perm_commitment;
        let mut mat_idx = 0;
        let mut per_mat = Vec::with_capacity(num_used_airs);
        for i in 0..num_used_airs {
            if !mvk.per_air[i].has_interaction() {
                continue;
            }
            let width = mvk.per_air[i].params.width.after_challenge[0] * D;
            per_mat.push(trace_height_and_adj_openings(
                proof.per_air[i].log_trace_height,
                width,
                zeta,
                &proof.opened_values[commit_idx][mat_idx],
            )?);
            mat_idx += 1;
        }
        if mat_idx != proof.opened_values[commit_idx].len() {
            return Err(InvalidProofShape);
        }
        per_commitment.push((commit, per_mat));
        commit_idx += 1;
    }
    let quotient_commit_idx = commit_idx;
    // 4. Finally quotient chunks are all in one commitment, but out-of-domain points to open at
    //    have shifts
    if commit_idx + 1 != proof.opened_values.len() {
        return Err(InvalidProofShape);
    }
    {
        if proof.opened_values[commit_idx].len() != num_used_airs {
            return Err(InvalidProofShape);
        }
        let commit = proof.quotient_commitment;
        let mut per_mat = Vec::new(); // num_used_airs * (quotient_degree per air) * D
        for i in 0..num_used_airs {
            if proof.opened_values[commit_idx][i].len() != mvk.per_air[i].quotient_degree as usize {
                return Err(InvalidProofShape);
            }
            for j in 0..mvk.per_air[i].quotient_degree as usize {
                if proof.opened_values[commit_idx][i][j].len() != D {
                    return Err(InvalidProofShape);
                }
                per_mat.push((
                    1usize << proof.per_air[i].log_trace_height,
                    vec![(zeta, proof.opened_values[commit_idx][i][j].to_vec())],
                ));
            }
        }
        per_commitment.push((commit, per_mat));
    }

    verify_fri_pcs(
        &params.fri_params,
        per_commitment,
        proof.fri_proof,
        &mut sponge,
    )?;

    commit_idx = 0usize;
    let mut perm_matrix_idx = 0usize;

    // ================== Verify each RAP's constraints ====================
    for idx in 0..num_used_airs {
        let vk = &mvk.per_air[idx];
        let preprocessed_values = vk.preprocessed_data.is_some().then(|| {
            let values = &proof.opened_values[commit_idx][0];
            commit_idx += 1;
            values
        });
        let mut partitioned_main_values = Vec::with_capacity(vk.num_cached_mains() + 1);
        for _ in 0..vk.num_cached_mains() {
            partitioned_main_values.push(&proof.opened_values[commit_idx][0]);
            commit_idx += 1;
        }
        partitioned_main_values.push(&proof.opened_values[common_main_commit_idx][idx]);
        // loop through challenge phases of this single RAP
        let perm_values = if vk.has_interaction() {
            let perm_commit_idx = common_main_commit_idx + 1;
            let values = &proof.opened_values[perm_commit_idx][perm_matrix_idx];
            perm_matrix_idx += 1;
            values
        } else {
            &vec![vec![]; 2]
        };
        let quotient_values = &proof.opened_values[quotient_commit_idx][idx];
        // ============ verify single RAP constraints =================
        let log_trace_height = proof.per_air[idx].log_trace_height;
        let omega = F::two_adic_generator(log_trace_height);
        let mut shift = F::GENERATOR;

        let quotient_degree = vk.quotient_degree as usize;
        let mut quotient_domain_shifts = Vec::with_capacity(quotient_degree);
        for _ in 0..quotient_degree {
            quotient_domain_shifts.push(shift);
            shift *= omega;
        }
        let zps = (0..quotient_degree)
            .map(|ch_i| {
                let mut prod = EF::ONE;
                for j in 0..quotient_degree {
                    if j != ch_i {
                        prod *= (zeta * quotient_domain_shifts[j].inverse())
                            .exp_power_of_2(log_trace_height)
                            - EF::ONE;
                        prod *= (EF::from(
                            (quotient_domain_shifts[ch_i] * quotient_domain_shifts[j].inverse())
                                .exp_power_of_2(log_trace_height),
                        ) - EF::ONE)
                            .inverse();
                    }
                }
                prod
            })
            .collect_vec();

        // quotient poly evaluation at zeta
        let mut quotient_eval = EF::ZERO;
        for ch_i in 0..quotient_degree {
            for e_i in 0..D {
                quotient_eval += zps[ch_i]
                    * <EF as FieldExtensionAlgebra<F>>::monomial(e_i)
                    * quotient_values[ch_i][e_i];
            }
        }

        // Lagrange selectors:
        let z_h = zeta.exp_power_of_2(log_trace_height) - EF::ONE;
        let is_first_row = z_h / (zeta - EF::ONE);
        let is_last_row = z_h / (zeta - omega.inverse());
        let is_transition = zeta - omega.inverse();
        let inv_zeroifier = z_h.inverse();

        // Evaluation by traversal of constraint DAG: currently going to use existing recursive
        // traversal
        let (preprocessed_local, preprocessed_next) = preprocessed_values
            .as_ref()
            .map(|values| (values[0].as_slice(), values[1].as_slice()))
            .unwrap_or((&[], &[]));
        let preprocessed = VerticalPair::new(
            RowMajorMatrixView::new_row(preprocessed_local),
            RowMajorMatrixView::new_row(preprocessed_next),
        );
        let partitioned_main: Vec<_> = partitioned_main_values
            .into_iter()
            .map(|values| {
                VerticalPair::new(
                    RowMajorMatrixView::new_row(&values[0]),
                    RowMajorMatrixView::new_row(&values[1]),
                )
            })
            .collect();
        let [perm_local, perm_next] = [0, 1].map(|i| {
            perm_values[i]
                .chunks_exact(D)
                .map(|chunk| {
                    // some interpolation to go from evaluations of base field poly to evaluation of
                    // extension field poly
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(e_i, &c)| <EF as FieldExtensionAlgebra<F>>::monomial(e_i) * c)
                        .sum()
                })
                .collect_vec()
        });
        let perm = VerticalPair::new(
            RowMajorMatrixView::new_row(&perm_local),
            RowMajorMatrixView::new_row(&perm_next),
        );

        let mut folder: VerifierConstraintFolder<'_, BabyBearPoseidon2Config> =
            GenericVerifierConstraintFolder {
                preprocessed,
                partitioned_main,
                after_challenge: vec![perm],
                is_first_row,
                is_last_row,
                is_transition,
                alpha,
                accumulator: EF::ZERO,
                challenges: &[perm_challenges.to_vec()],
                public_values: &proof.per_air[idx].public_values,
                exposed_values_after_challenge: &[vec![proof.per_air[idx].log_up_cumulative_sum]],
                _marker: PhantomData,
            };
        folder.eval_constraints(&vk.symbolic_constraints.constraints);

        let folded_constraints = folder.accumulator;
        // Finally, check that
        //     folded_constraints(zeta) / Z_H(zeta) = quotient(zeta)
        if folded_constraints * inv_zeroifier != quotient_eval {
            return Err(VerificationError::OodEvaluationMismatch);
        }
    }

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
        .map(|&comm| {
            sponge.observe_commit(comm);
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
            .skip(params.log_blowup)
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

impl From<Proof<BabyBearPoseidon2Config>> for StarkProof {
    fn from(mut proof: Proof<BabyBearPoseidon2Config>) -> Self {
        let common_main_commitment = proof.commitments.main_trace.pop().unwrap().into();
        let cached_main_commitments = proof
            .commitments
            .main_trace
            .into_iter()
            .map(|com| com.into())
            .collect();
        let perm_commitment = proof
            .commitments
            .after_challenge
            .first()
            .map(|&com| com.into())
            .unwrap_or([F::ZERO; CHUNK]);
        let quotient_commitment = proof.commitments.quotient.into();
        let p3_fri_proof = proof.opening.proof;
        let fri_proof = FriProof {
            commit_phase_commits: p3_fri_proof
                .commit_phase_commits
                .into_iter()
                .map(|com| com.into())
                .collect(),
            query_proofs: p3_fri_proof
                .query_proofs
                .into_iter()
                .map(|qp| QueryProof {
                    input_proof: qp
                        .input_proof
                        .into_iter()
                        .map(|bo| BatchOpening {
                            opened_values: bo.opened_values,
                            opening_proof: bo.opening_proof,
                        })
                        .collect(),
                    commit_phase_openings: qp
                        .commit_phase_openings
                        .into_iter()
                        .map(|step| CommitPhaseProofStep {
                            sibling_value: step.sibling_value,
                            opening_proof: step.opening_proof,
                        })
                        .collect(),
                })
                .collect(),
            final_poly: p3_fri_proof.final_poly[0],
            pow_witness: p3_fri_proof.pow_witness,
        };

        let mut opened_values = Vec::new();
        for prep in proof.opening.values.preprocessed {
            opened_values.push(vec![vec![prep.local, prep.next]]);
        }
        for m in proof.opening.values.main {
            opened_values.push(m.into_iter().map(|v| vec![v.local, v.next]).collect());
        }
        assert!(proof.opening.values.after_challenge.len() <= 1);
        for perm in proof.opening.values.after_challenge {
            opened_values.push(perm.into_iter().map(|v| vec![v.local, v.next]).collect());
        }
        opened_values.push(proof.opening.values.quotient);

        let per_air = proof
            .per_air
            .into_iter()
            .map(|p| AirProofData {
                air_id: p.air_id,
                log_trace_height: log2_strict_usize(p.degree),
                log_up_cumulative_sum: p
                    .exposed_values_after_challenge
                    .first()
                    .map(|v| v[0])
                    .unwrap_or_default(),
                public_values: p.public_values,
            })
            .collect();

        Self {
            cached_main_commitments,
            common_main_commitment,
            perm_commitment,
            quotient_commitment,
            fri_proof,
            opened_values,
            per_air,
            logup_pow_witness: proof
                .rap_phase_seq_proof
                .as_ref()
                .map(|p| p.logup_pow_witness)
                .unwrap_or_default(),
        }
    }
}
