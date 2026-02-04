use std::cmp::{max, Reverse};

use itertools::{izip, Itertools};
use thiserror::Error;

use crate::{
    block_sumcheck_sizes, calculate_n_logup, gkr_block_len,
    keygen::types::MultiStarkVerifyingKey0V2, proof::Proof, prover::stacked_pcs::StackedLayout,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProofShapeError {
    #[error("Invalid VData: {0}")]
    InvalidVData(ProofShapeVDataError),
    #[error("Invalid GkrProof shape: {0}")]
    InvalidGkrProofShape(GkrProofShapeError),
    #[error("Invalid BatchConstraintProof shape: {0}")]
    InvalidBatchConstraintProofShape(BatchProofShapeError),
    #[error("Invalid StackingProof shape: {0}")]
    InvalidStackingProofShape(StackingProofShapeError),
    #[error("Invalid WhirProof shape: {0}")]
    InvalidWhirProofShape(WhirProofShapeError),
}

impl ProofShapeError {
    fn invalid_vdata<T>(err: ProofShapeVDataError) -> Result<T, Self> {
        Err(Self::InvalidVData(err))
    }

    fn invalid_gkr<T>(err: GkrProofShapeError) -> Result<T, Self> {
        Err(Self::InvalidGkrProofShape(err))
    }

    fn invalid_batch_constraint<T>(err: BatchProofShapeError) -> Result<T, Self> {
        Err(Self::InvalidBatchConstraintProofShape(err))
    }

    fn invalid_stacking<T>(err: StackingProofShapeError) -> Result<T, Self> {
        Err(Self::InvalidStackingProofShape(err))
    }

    fn invalid_whir<T>(err: WhirProofShapeError) -> Result<T, Self> {
        Err(Self::InvalidWhirProofShape(err))
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProofShapeVDataError {
    #[error("Proof trace_vdata length ({len}) does not match number of AIRs ({num_airs})")]
    InvalidVDataLength { len: usize, num_airs: usize },
    #[error("Proof public_values length ({len}) does not match number of AIRs ({num_airs})")]
    InvalidPublicValuesLength { len: usize, num_airs: usize },
    #[error("AIR {air_idx} is required, but trace_vdata[{air_idx}] is None")]
    RequiredAirNoVData { air_idx: usize },
    #[error("AIR {air_idx} has no TraceVData, but a non-zero amount of public values")]
    PublicValuesNoVData { air_idx: usize },
    #[error(
        "TraceVata for AIR {air_idx} should have {expected} cached commitments, but has {actual}"
    )]
    InvalidCachedCommitments {
        air_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error("AIR {air_idx} should have log_height <= {}, but has {actual} (l_skip = {l_skip}, n_stack = {n_stack}", l_skip + n_stack)]
    LogHeightOutOfBounds {
        air_idx: usize,
        l_skip: usize,
        n_stack: usize,
        actual: usize,
    },
    #[error("AIR {air_idx} should have {expected} public values, but has {actual}")]
    InvalidPublicValues {
        air_idx: usize,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GkrProofShapeError {
    #[error(
        "claims_per_layer should have num_gkr_rounds = {expected} claims, but it has {actual}"
    )]
    InvalidClaimsPerLayer { expected: usize, actual: usize },
    #[error(
        "block_sumcheck_polys should have num_gkr_rounds.saturating_sub(1) = {expected} layers, but it has {actual}"
    )]
    InvalidBlockSumcheckPolys { expected: usize, actual: usize },
    #[error("Block sumcheck entry for round {round} block {block} has invalid length {len}")]
    InvalidBlockSumcheckPolyLen {
        round: usize,
        block: usize,
        len: usize,
    },
    #[error("Round {round} should have {expected} sumcheck variables, but got {actual}")]
    InvalidBlockSumcheckRoundVars {
        round: usize,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BatchProofShapeError {
    #[error("numerator_term_per_air should have num_airs = {expected} terms, but it has {actual}")]
    InvalidNumeratorTerms { expected: usize, actual: usize },
    #[error(
        "denominator_term_per_air should have num_airs = {expected} terms, but it has {actual}"
    )]
    InvalidDenominatorTerms { expected: usize, actual: usize },
    #[error(
        "univariate_round_coeffs should have (max_constraint_degree + 1) * (2^l_skip - 1) + 1 = {expected} coefficients, but it has {actual}"
    )]
    InvalidUnivariateRoundCoeffs { expected: usize, actual: usize },
    #[error(
        "sumcheck_round_polys should have n_global = {expected} polynomials, but it has {actual}"
    )]
    InvalidSumcheckRoundPolys { expected: usize, actual: usize },
    #[error(
        "column_openings should have num_airs = {expected} sets of openings, but it has {actual}"
    )]
    InvalidColumnOpeningsAirs { expected: usize, actual: usize },
    #[error(
        "sumcheck_round_polys[{round}] should have degree = {expected} evaluations, but it has {actual}"
    )]
    InvalidSumcheckRoundPolyEvals {
        round: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "AIR {air_idx} has {expected} parts, but there are {actual} sets of per-part column openings"
    )]
    InvalidColumnOpeningsPerAir {
        air_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "There should be {expected} column opening pairs for AIR {air_idx}'s main trace, but instead there are {actual}"
    )]
    InvalidColumnOpeningsPerAirMain {
        air_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "There should be {expected} column opening pairs for AIR {air_idx}'s preprocessed trace, but instead there are {actual}"
    )]
    InvalidColumnOpeningsPerAirPreprocessed {
        air_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "There should be {expected} column opening pairs for AIR {air_idx}'s cached trace {cached_idx}, but instead there are {actual}"
    )]
    InvalidColumnOpeningsPerAirCached {
        air_idx: usize,
        cached_idx: usize,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StackingProofShapeError {
    #[error(
        "univariate_round_coeffs should have 2 * ((1 << mvk.params.l_skip) - 1) + 1 = {expected} coefficients, but it has {actual}"
    )]
    InvalidUnivariateRoundCoeffs { expected: usize, actual: usize },
    #[error(
        "sumcheck_round_polys should have n_stack = {expected} polynomials, but it has {actual}"
    )]
    InvalidSumcheckRoundPolys { expected: usize, actual: usize },
    #[error(
        "There should be {expected} sets of per-commit stacking openings, but instead there are {actual}"
    )]
    InvalidStackOpenings { expected: usize, actual: usize },
    #[error(
        "Stacked matrix {commit_idx} should have {expected} stacking openings, but instead there are {actual}"
    )]
    InvalidStackOpeningsPerMatrix {
        commit_idx: usize,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WhirProofShapeError {
    #[error(
        "whir_sumcheck_polys should have num_whir_sumcheck_rounds = {expected} polynomials, but it has {actual}"
    )]
    InvalidSumcheckPolys { expected: usize, actual: usize },
    #[error("final_poly should have len = {expected}, but it has {actual}")]
    InvalidFinalPolyLen { expected: usize, actual: usize },
    #[error(
        "There should be num_whir_rounds = {expected} codeword commits, but there are {actual}"
    )]
    InvalidCodewordCommits { expected: usize, actual: usize },
    #[error(
        "There should be num_whir_rounds = {expected} out-of-domain values, but there are {actual}"
    )]
    InvalidOodValues { expected: usize, actual: usize },
    #[error(
        "There should be num_whir_sumcheck_rounds = {expected} folding PoW witnesses, but there are {actual}"
    )]
    InvalidFoldingPowWitnesses { expected: usize, actual: usize },
    #[error(
        "There should be num_whir_rounds = {expected} query phase PoW witnesses, but there are {actual}"
    )]
    InvalidQueryPhasePowWitnesses { expected: usize, actual: usize },
    #[error(
        "There should be num_commits = {expected} sets of initial round opened rows, but there are {actual}"
    )]
    InvalidInitialRoundOpenedRows { expected: usize, actual: usize },
    #[error(
        "There should be num_commits = {expected} sets of initial round merkle proofs, but there are {actual}"
    )]
    InvalidInitialRoundMerkleProofs { expected: usize, actual: usize },
    #[error(
        "There should be num_whir_rounds = {expected} sets of non-initial round opened rows, but there are {actual}"
    )]
    InvalidCodewordOpenedRows { expected: usize, actual: usize },
    #[error(
        "There should be num_whir_rounds = {expected} sets of non-initial round merkle proofs, but there are {actual}"
    )]
    InvalidCodewordMerkleProofs { expected: usize, actual: usize },
    #[error(
        "There should be num_whir_queries = {expected} initial round opened rows for commit {commit_idx}, but there are {actual}"
    )]
    InvalidInitialRoundOpenedRowsQueries {
        commit_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "There should be num_whir_queries = {expected} initial round merkle proofs for commit {commit_idx}, but there are {actual}"
    )]
    InvalidInitialRoundMerkleProofsQueries {
        commit_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Initial round opened row {opened_idx} for commit {commit_idx} should have length {expected}, but it has length {actual}"
    )]
    InvalidInitialRoundOpenedRowK {
        opened_idx: usize,
        commit_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Initial round opened row {row_idx} for commit {commit_idx} should have width {expected}, but it has width {actual}"
    )]
    InvalidInitialRoundOpenedRowWidth {
        row_idx: usize,
        commit_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Initial round merkle proof {opened_idx} for commit {commit_idx} should have depth {expected}, but it has depth {actual}"
    )]
    InvalidInitialRoundMerkleProofDepth {
        opened_idx: usize,
        commit_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "There should be num_whir_queries = {expected} round {round} opened rows, but there are {actual}"
    )]
    InvalidCodewordOpenedRowsQueries {
        round: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "There should be num_whir_queries = {expected} round {round} merkle proofs, but there are {actual}"
    )]
    InvalidCodewordMerkleProofsQueries {
        round: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Round {round} opened row {opened_idx} should have length {expected}, but it has length {actual}"
    )]
    InvalidCodewordOpenedValues {
        round: usize,
        opened_idx: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Round {round} merkle proof {opened_idx} should have depth {expected}, but it has depth {actual}"
    )]
    InvalidCodewordMerkleProofDepth {
        round: usize,
        opened_idx: usize,
        expected: usize,
        actual: usize,
    },
}

pub fn verify_proof_shape(
    mvk: &MultiStarkVerifyingKey0V2,
    proof: &Proof,
) -> Result<Vec<StackedLayout>, ProofShapeError> {
    // TRACE HEIGHTS AND PUBLIC VALUES
    let num_airs = mvk.per_air.len();
    let l_skip = mvk.params.l_skip;

    if proof.trace_vdata.len() != num_airs {
        return ProofShapeError::invalid_vdata(ProofShapeVDataError::InvalidVDataLength {
            len: proof.trace_vdata.len(),
            num_airs,
        });
    } else if proof.public_values.len() != num_airs {
        return ProofShapeError::invalid_vdata(ProofShapeVDataError::InvalidPublicValuesLength {
            len: proof.public_values.len(),
            num_airs,
        });
    }

    for (air_idx, (vk, vdata, pvs)) in
        izip!(&mvk.per_air, &proof.trace_vdata, &proof.public_values).enumerate()
    {
        if vdata.is_none() {
            if vk.is_required {
                return ProofShapeError::invalid_vdata(ProofShapeVDataError::RequiredAirNoVData {
                    air_idx,
                });
            } else if !pvs.is_empty() {
                return ProofShapeError::invalid_vdata(ProofShapeVDataError::PublicValuesNoVData {
                    air_idx,
                });
            }
        } else {
            let vdata = vdata.as_ref().unwrap();
            if vdata.cached_commitments.len() != vk.num_cached_mains() {
                return ProofShapeError::invalid_vdata(
                    ProofShapeVDataError::InvalidCachedCommitments {
                        air_idx,
                        expected: vk.num_cached_mains(),
                        actual: vdata.cached_commitments.len(),
                    },
                );
            } else if vdata.log_height > l_skip + mvk.params.n_stack {
                return ProofShapeError::invalid_vdata(
                    ProofShapeVDataError::LogHeightOutOfBounds {
                        air_idx,
                        l_skip,
                        n_stack: mvk.params.n_stack,
                        actual: vdata.log_height,
                    },
                );
            } else if vk.params.num_public_values != pvs.len() {
                return ProofShapeError::invalid_vdata(ProofShapeVDataError::InvalidPublicValues {
                    air_idx,
                    expected: vk.params.num_public_values,
                    actual: pvs.len(),
                });
            }
        }
    }

    let per_trace = mvk
        .per_air
        .iter()
        .zip(&proof.trace_vdata)
        .enumerate()
        .filter_map(|(air_idx, (vk, vdata))| vdata.as_ref().map(|vdata| (air_idx, vk, vdata)))
        .sorted_by_key(|(_, _, vdata)| Reverse(vdata.log_height))
        .collect_vec();
    let num_airs_present = per_trace.len();

    // GKR PROOF SHAPE
    let total_interactions = per_trace.iter().fold(0u64, |acc, (_, vk, vdata)| {
        acc + ((vk.num_interactions() as u64) << max(vdata.log_height, l_skip))
    });
    let n_logup = calculate_n_logup(l_skip, total_interactions);
    let num_gkr_rounds = if total_interactions == 0 {
        0
    } else {
        l_skip + n_logup
    };

    if proof.gkr_proof.claims_per_layer.len() != num_gkr_rounds {
        return ProofShapeError::invalid_gkr(GkrProofShapeError::InvalidClaimsPerLayer {
            expected: num_gkr_rounds,
            actual: proof.gkr_proof.claims_per_layer.len(),
        });
    } else if proof.gkr_proof.block_sumcheck_polys.len() != num_gkr_rounds.saturating_sub(1) {
        return ProofShapeError::invalid_gkr(GkrProofShapeError::InvalidBlockSumcheckPolys {
            expected: num_gkr_rounds.saturating_sub(1),
            actual: proof.gkr_proof.block_sumcheck_polys.len(),
        });
    }

    for (round_idx, blocks) in proof.gkr_proof.block_sumcheck_polys.iter().enumerate() {
        let round = round_idx + 1;
        let mut sizes = block_sumcheck_sizes(round);
        let mut consumed = 0usize;
        for (block_idx, block) in blocks.iter().enumerate() {
            let Some(k) = sizes.next() else {
                return ProofShapeError::invalid_gkr(
                    GkrProofShapeError::InvalidBlockSumcheckRoundVars {
                        round,
                        expected: round,
                        actual: round + 1,
                    },
                );
            };
            let expected_len = gkr_block_len(k);
            if block.len() != expected_len {
                return ProofShapeError::invalid_gkr(
                    GkrProofShapeError::InvalidBlockSumcheckPolyLen {
                        round,
                        block: block_idx,
                        len: block.len(),
                    },
                );
            }
            consumed += k;
        }
        if sizes.next().is_some() || consumed != round {
            return ProofShapeError::invalid_gkr(
                GkrProofShapeError::InvalidBlockSumcheckRoundVars {
                    round,
                    expected: round,
                    actual: consumed,
                },
            );
        }
    }

    // BATCH CONSTRAINTS PROOF SHAPE
    let batch_proof = &proof.batch_constraint_proof;

    let n_max = per_trace[0].2.log_height.saturating_sub(l_skip);

    let s_0_deg = (mvk.max_constraint_degree() + 1) * ((1 << l_skip) - 1);
    if batch_proof.numerator_term_per_air.len() != num_airs_present {
        return ProofShapeError::invalid_batch_constraint(
            BatchProofShapeError::InvalidNumeratorTerms {
                expected: num_airs_present,
                actual: batch_proof.numerator_term_per_air.len(),
            },
        );
    } else if batch_proof.denominator_term_per_air.len() != num_airs_present {
        return ProofShapeError::invalid_batch_constraint(
            BatchProofShapeError::InvalidDenominatorTerms {
                expected: num_airs_present,
                actual: batch_proof.denominator_term_per_air.len(),
            },
        );
    } else if batch_proof.univariate_round_coeffs.len() != s_0_deg + 1 {
        return ProofShapeError::invalid_batch_constraint(
            BatchProofShapeError::InvalidUnivariateRoundCoeffs {
                expected: s_0_deg + 1,
                actual: batch_proof.univariate_round_coeffs.len(),
            },
        );
    } else if batch_proof.sumcheck_round_polys.len() != n_max {
        return ProofShapeError::invalid_batch_constraint(
            BatchProofShapeError::InvalidSumcheckRoundPolys {
                expected: n_max,
                actual: batch_proof.sumcheck_round_polys.len(),
            },
        );
    } else if batch_proof.column_openings.len() != num_airs_present {
        return ProofShapeError::invalid_batch_constraint(
            BatchProofShapeError::InvalidColumnOpeningsAirs {
                expected: num_airs_present,
                actual: batch_proof.column_openings.len(),
            },
        );
    }

    for (i, evals) in batch_proof.sumcheck_round_polys.iter().enumerate() {
        if evals.len() != mvk.max_constraint_degree() + 1 {
            return ProofShapeError::invalid_batch_constraint(
                BatchProofShapeError::InvalidSumcheckRoundPolyEvals {
                    round: i,
                    expected: mvk.max_constraint_degree() + 1,
                    actual: evals.len(),
                },
            );
        }
    }

    for (part_openings, &(air_idx, vk, _)) in batch_proof.column_openings.iter().zip(&per_trace) {
        let need_rot = mvk.per_air[air_idx].params.need_rot;
        let openings_per_col = if need_rot { 2 } else { 1 };
        if part_openings.len() != vk.num_parts() {
            return ProofShapeError::invalid_batch_constraint(
                BatchProofShapeError::InvalidColumnOpeningsPerAir {
                    air_idx,
                    expected: vk.num_parts(),
                    actual: part_openings.len(),
                },
            );
        } else if part_openings[0].len() != vk.params.width.common_main * openings_per_col {
            return ProofShapeError::invalid_batch_constraint(
                BatchProofShapeError::InvalidColumnOpeningsPerAirMain {
                    air_idx,
                    expected: vk.params.width.common_main,
                    actual: part_openings[0].len(),
                },
            );
        } else if let Some(preprocessed_width) = &vk.params.width.preprocessed {
            if part_openings[1].len() != *preprocessed_width * openings_per_col {
                return ProofShapeError::invalid_batch_constraint(
                    BatchProofShapeError::InvalidColumnOpeningsPerAirPreprocessed {
                        air_idx,
                        expected: *preprocessed_width,
                        actual: part_openings[1].len(),
                    },
                );
            }
        }

        let cached_openings = &part_openings[1 + (vk.preprocessed_data.is_some() as usize)..];
        for (cached_idx, (col_opening, &width)) in cached_openings
            .iter()
            .zip(&vk.params.width.cached_mains)
            .enumerate()
        {
            if col_opening.len() != width * openings_per_col {
                return ProofShapeError::invalid_batch_constraint(
                    BatchProofShapeError::InvalidColumnOpeningsPerAirCached {
                        air_idx,
                        cached_idx,
                        expected: width,
                        actual: col_opening.len(),
                    },
                );
            }
        }
    }

    // STACKING PROOF SHAPE
    let stacking_proof = &proof.stacking_proof;

    let s_0_deg = 2 * ((1 << l_skip) - 1);
    if stacking_proof.univariate_round_coeffs.len() != s_0_deg + 1 {
        return ProofShapeError::invalid_stacking(
            StackingProofShapeError::InvalidUnivariateRoundCoeffs {
                expected: s_0_deg + 1,
                actual: stacking_proof.univariate_round_coeffs.len(),
            },
        );
    } else if stacking_proof.sumcheck_round_polys.len() != mvk.params.n_stack {
        return ProofShapeError::invalid_stacking(
            StackingProofShapeError::InvalidSumcheckRoundPolys {
                expected: mvk.params.n_stack,
                actual: stacking_proof.sumcheck_round_polys.len(),
            },
        );
    }

    let common_main_layout = StackedLayout::new(
        l_skip,
        mvk.params.n_stack + l_skip,
        per_trace
            .iter()
            .map(|(_, vk, vdata)| (vk.params.width.common_main, vdata.log_height))
            .collect_vec(),
    );

    let other_layouts = per_trace
        .iter()
        .flat_map(|(_, vk, vdata)| {
            vk.params
                .width
                .preprocessed
                .iter()
                .chain(&vk.params.width.cached_mains)
                .copied()
                .map(|width| (width, vdata.log_height))
                .collect_vec()
        })
        .map(|sorted| StackedLayout::new(l_skip, mvk.params.n_stack + l_skip, vec![sorted]))
        .collect_vec();

    let layouts = [common_main_layout]
        .into_iter()
        .chain(other_layouts)
        .collect_vec();

    if stacking_proof.stacking_openings.len() != layouts.len() {
        return ProofShapeError::invalid_stacking(StackingProofShapeError::InvalidStackOpenings {
            expected: layouts.len(),
            actual: stacking_proof.stacking_openings.len(),
        });
    }

    for (commit_idx, (openings, layout)) in stacking_proof
        .stacking_openings
        .iter()
        .zip(&layouts)
        .enumerate()
    {
        let stacked_matrix_width = layout.sorted_cols.last().unwrap().2.col_idx + 1;
        if openings.len() != stacked_matrix_width {
            return ProofShapeError::invalid_stacking(
                StackingProofShapeError::InvalidStackOpeningsPerMatrix {
                    commit_idx,
                    expected: stacked_matrix_width,
                    actual: openings.len(),
                },
            );
        }
    }

    // WHIR PROOF SHAPE
    let whir_proof = &proof.whir_proof;

    let log_stacked_height = mvk.params.log_stacked_height();
    let num_whir_rounds = mvk.params.num_whir_rounds();
    let num_whir_sumcheck_rounds = mvk.params.num_whir_sumcheck_rounds();
    let k_whir = mvk.params.k_whir();
    debug_assert_ne!(num_whir_rounds, 0);

    if whir_proof.whir_sumcheck_polys.len() != num_whir_sumcheck_rounds {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidSumcheckPolys {
            expected: num_whir_sumcheck_rounds,
            actual: whir_proof.whir_sumcheck_polys.len(),
        });
    } else if whir_proof.codeword_commits.len() != num_whir_rounds - 1 {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidCodewordCommits {
            expected: num_whir_rounds - 1,
            actual: whir_proof.codeword_commits.len(),
        });
    } else if whir_proof.ood_values.len() != num_whir_rounds - 1 {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidOodValues {
            expected: num_whir_rounds - 1,
            actual: whir_proof.ood_values.len(),
        });
    } else if whir_proof.folding_pow_witnesses.len() != num_whir_sumcheck_rounds {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidFoldingPowWitnesses {
            expected: num_whir_sumcheck_rounds,
            actual: whir_proof.folding_pow_witnesses.len(),
        });
    } else if whir_proof.query_phase_pow_witnesses.len() != num_whir_rounds {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidQueryPhasePowWitnesses {
            expected: num_whir_rounds,
            actual: whir_proof.query_phase_pow_witnesses.len(),
        });
    } else if whir_proof.initial_round_opened_rows.len() != layouts.len() {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidInitialRoundOpenedRows {
            expected: layouts.len(),
            actual: whir_proof.initial_round_opened_rows.len(),
        });
    } else if whir_proof.initial_round_merkle_proofs.len() != layouts.len() {
        return ProofShapeError::invalid_whir(
            WhirProofShapeError::InvalidInitialRoundMerkleProofs {
                expected: layouts.len(),
                actual: whir_proof.initial_round_merkle_proofs.len(),
            },
        );
    } else if whir_proof.codeword_opened_values.len() != num_whir_rounds - 1 {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidCodewordOpenedRows {
            expected: num_whir_rounds - 1,
            actual: whir_proof.codeword_opened_values.len(),
        });
    } else if whir_proof.codeword_merkle_proofs.len() != num_whir_rounds - 1 {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidCodewordMerkleProofs {
            expected: num_whir_rounds - 1,
            actual: whir_proof.codeword_merkle_proofs.len(),
        });
    } else if whir_proof.final_poly.len() != 1 << mvk.params.log_final_poly_len() {
        return ProofShapeError::invalid_whir(WhirProofShapeError::InvalidFinalPolyLen {
            expected: 1 << mvk.params.log_final_poly_len(),
            actual: whir_proof.final_poly.len(),
        });
    }

    let initial_whir_round_num_queries = mvk.params.whir.rounds[0].num_queries;
    for (commit_idx, (opened_rows, merkle_proofs)) in whir_proof
        .initial_round_opened_rows
        .iter()
        .zip(&whir_proof.initial_round_merkle_proofs)
        .enumerate()
    {
        if opened_rows.len() != initial_whir_round_num_queries {
            return ProofShapeError::invalid_whir(
                WhirProofShapeError::InvalidInitialRoundOpenedRowsQueries {
                    commit_idx,
                    expected: initial_whir_round_num_queries,
                    actual: opened_rows.len(),
                },
            );
        } else if merkle_proofs.len() != initial_whir_round_num_queries {
            return ProofShapeError::invalid_whir(
                WhirProofShapeError::InvalidInitialRoundMerkleProofsQueries {
                    commit_idx,
                    expected: initial_whir_round_num_queries,
                    actual: merkle_proofs.len(),
                },
            );
        }
        let width = stacking_proof.stacking_openings[commit_idx].len();
        for (opened_idx, rows) in opened_rows.iter().enumerate() {
            if rows.len() != 1 << k_whir {
                return ProofShapeError::invalid_whir(
                    WhirProofShapeError::InvalidInitialRoundOpenedRowK {
                        opened_idx,
                        commit_idx,
                        expected: 1 << k_whir,
                        actual: rows.len(),
                    },
                );
            }
            for (row_idx, row) in rows.iter().enumerate() {
                if row.len() != width {
                    return ProofShapeError::invalid_whir(
                        WhirProofShapeError::InvalidInitialRoundOpenedRowWidth {
                            row_idx,
                            commit_idx,
                            expected: width,
                            actual: row.len(),
                        },
                    );
                }
            }
        }

        let merkle_depth = (log_stacked_height + mvk.params.log_blowup).saturating_sub(k_whir);
        for (opened_idx, proof) in merkle_proofs.iter().enumerate() {
            if proof.len() != merkle_depth {
                return ProofShapeError::invalid_whir(
                    WhirProofShapeError::InvalidInitialRoundMerkleProofDepth {
                        opened_idx,
                        commit_idx,
                        expected: merkle_depth,
                        actual: proof.len(),
                    },
                );
            }
        }
    }

    for (round_minus_one, (opened_values_per_query, merkle_proofs)) in whir_proof
        .codeword_opened_values
        .iter()
        .zip(&whir_proof.codeword_merkle_proofs)
        .take(num_whir_rounds - 1)
        .enumerate()
    {
        let round = round_minus_one + 1;
        let num_queries = mvk.params.whir.rounds[round].num_queries;
        if opened_values_per_query.len() != num_queries {
            return ProofShapeError::invalid_whir(
                WhirProofShapeError::InvalidCodewordOpenedRowsQueries {
                    round,
                    expected: num_queries,
                    actual: opened_values_per_query.len(),
                },
            );
        } else if merkle_proofs.len() != num_queries {
            return ProofShapeError::invalid_whir(
                WhirProofShapeError::InvalidCodewordMerkleProofsQueries {
                    round,
                    expected: num_queries,
                    actual: merkle_proofs.len(),
                },
            );
        }

        for (opened_idx, opened_values) in opened_values_per_query.iter().enumerate() {
            if opened_values.len() != 1 << mvk.params.k_whir() {
                return ProofShapeError::invalid_whir(
                    WhirProofShapeError::InvalidCodewordOpenedValues {
                        round,
                        opened_idx,
                        expected: 1 << mvk.params.k_whir(),
                        actual: opened_values.len(),
                    },
                );
            }
        }

        let merkle_depth = log_stacked_height + mvk.params.log_blowup - k_whir - round;
        for (opened_idx, proof) in merkle_proofs.iter().enumerate() {
            if proof.len() != merkle_depth {
                return ProofShapeError::invalid_whir(
                    WhirProofShapeError::InvalidCodewordMerkleProofDepth {
                        round,
                        opened_idx,
                        expected: merkle_depth,
                        actual: proof.len(),
                    },
                );
            }
        }
    }

    Ok(layouts)
}
