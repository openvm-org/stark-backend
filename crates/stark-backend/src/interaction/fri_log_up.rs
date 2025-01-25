use std::{
    array,
    borrow::Borrow,
    cmp::max,
    iter::{self, zip},
    marker::PhantomData,
};

use itertools::Itertools;
use p3_air::ExtensionBuilder;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{ExtensionField, Field, FieldAlgebra};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{PairTraceView, SymbolicInteraction};
use crate::{
    air_builders::symbolic::{symbolic_expression::SymbolicEvaluator, SymbolicConstraints},
    interaction::{
        trace::Evaluator,
        utils::{generate_betas, generate_rlc_elements},
        InteractionBuilder, InteractionType, RapPhaseProverData, RapPhaseSeq, RapPhaseSeqKind,
        RapPhaseVerifierData,
    },
    parizip,
    rap::PermutationAirBuilderWithExposedValues,
    utils::metrics_span,
};

#[derive(Default)]
pub struct FriLogUpPhase<F, Challenge, Challenger> {
    _marker: PhantomData<(F, Challenge, Challenger)>,
}

impl<F, Challenge, Challenger> FriLogUpPhase<F, Challenge, Challenger> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[derive(Error, Debug)]
pub enum FriLogUpError {
    #[error("non-zero cumulative sum")]
    NonZeroCumulativeSum,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FriLogUpProvingKey {
    interaction_partitions: Vec<Vec<usize>>,
}

impl<F: Field, Challenge, Challenger> RapPhaseSeq<F, Challenge, Challenger>
    for FriLogUpPhase<F, Challenge, Challenger>
where
    F: Field,
    Challenge: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    type PartialProof = ();
    type Error = FriLogUpError;
    const ID: RapPhaseSeqKind = RapPhaseSeqKind::FriLogUp;

    fn partially_prove(
        &self,
        challenger: &mut Challenger,
        constraints_per_air: &[&SymbolicConstraints<F>],
        params_per_air: &[&FriLogUpProvingKey],
        trace_view_per_air: &[PairTraceView<F>],
    ) -> Option<(Self::PartialProof, RapPhaseProverData<Challenge>)> {
        let has_any_interactions = constraints_per_air
            .iter()
            .any(|constraints| !constraints.interactions.is_empty());

        if !has_any_interactions {
            return None;
        }

        let challenges: [Challenge; STARK_LU_NUM_CHALLENGES] =
            array::from_fn(|_| challenger.sample_ext_element::<Challenge>());

        let after_challenge_trace_per_air = metrics_span("generate_perm_trace_time_ms", || {
            Self::generate_after_challenge_traces_per_air(
                &challenges,
                constraints_per_air,
                params_per_air,
                trace_view_per_air,
            )
        });
        let cumulative_sum_per_air = Self::extract_cumulative_sums(&after_challenge_trace_per_air);

        // Challenger needs to observe what is exposed (cumulative_sums)
        for cumulative_sum in cumulative_sum_per_air.iter().flatten() {
            challenger.observe_slice(cumulative_sum.as_base_slice());
        }

        let exposed_values_per_air = cumulative_sum_per_air
            .iter()
            .map(|csum| csum.map(|csum| vec![csum]))
            .collect_vec();

        Some((
            (),
            RapPhaseProverData {
                challenges: challenges.to_vec(),
                after_challenge_trace_per_air,
                exposed_values_per_air,
            },
        ))
    }

    fn partially_verify<Commitment: Clone>(
        &self,
        challenger: &mut Challenger,
        _partial_proof: Option<&Self::PartialProof>,
        exposed_values_per_phase_per_air: &[Vec<Vec<Challenge>>],
        commitment_per_phase: &[Commitment],
        _permutation_opened_values: &[Vec<Vec<Vec<Challenge>>>],
    ) -> (RapPhaseVerifierData<Challenge>, Result<(), Self::Error>)
    where
        Challenger: CanObserve<Commitment>,
    {
        if exposed_values_per_phase_per_air
            .iter()
            .all(|exposed_values_per_phase_per_air| exposed_values_per_phase_per_air.is_empty())
        {
            return (
                RapPhaseVerifierData {
                    challenges_per_phase: vec![],
                },
                Ok(()),
            );
        }

        let challenges: [Challenge; STARK_LU_NUM_CHALLENGES] =
            array::from_fn(|_| challenger.sample_ext_element::<Challenge>());

        for exposed_values_per_phase in exposed_values_per_phase_per_air.iter() {
            if let Some(exposed_values) = exposed_values_per_phase.first() {
                for exposed_value in exposed_values {
                    challenger.observe_slice(exposed_value.as_base_slice());
                }
            }
        }

        challenger.observe(commitment_per_phase[0].clone());

        let cumulative_sums = exposed_values_per_phase_per_air
            .iter()
            .map(|exposed_values_per_phase| {
                assert!(
                    exposed_values_per_phase.len() <= 1,
                    "Verifier does not support more than 1 challenge phase"
                );
                exposed_values_per_phase.first().map(|exposed_values| {
                    assert_eq!(
                        exposed_values.len(),
                        1,
                        "Only exposed value should be cumulative sum"
                    );
                    exposed_values[0]
                })
            })
            .collect_vec();

        // Check cumulative sum
        let sum: Challenge = cumulative_sums
            .into_iter()
            .map(|c| c.unwrap_or(Challenge::ZERO))
            .sum();

        let result = if sum == Challenge::ZERO {
            Ok(())
        } else {
            Err(Self::Error::NonZeroCumulativeSum)
        };
        let verifier_data = RapPhaseVerifierData {
            challenges_per_phase: vec![challenges.to_vec()],
        };
        (verifier_data, result)
    }
}

pub const STARK_LU_NUM_CHALLENGES: usize = 2;
pub const STARK_LU_NUM_EXPOSED_VALUES: usize = 1;

impl<F, Challenge, Challenger> FriLogUpPhase<F, Challenge, Challenger>
where
    F: Field,
    Challenge: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    /// Returns a list of optional tuples of (permutation trace,cumulative sum) for each AIR.
    fn generate_after_challenge_traces_per_air(
        challenges: &[Challenge; STARK_LU_NUM_CHALLENGES],
        constraints_per_air: &[&SymbolicConstraints<F>],
        params_per_air: &[&FriLogUpProvingKey],
        trace_view_per_air: &[PairTraceView<F>],
    ) -> Vec<Option<RowMajorMatrix<Challenge>>> {
        parizip!(constraints_per_air, trace_view_per_air, params_per_air)
            .map(|(constraints, trace_view, params)| {
                Self::generate_after_challenge_trace(
                    &constraints.interactions,
                    trace_view,
                    challenges,
                    &params.interaction_partitions,
                )
            })
            .collect::<Vec<_>>()
    }

    fn extract_cumulative_sums(
        perm_traces: &[Option<RowMajorMatrix<Challenge>>],
    ) -> Vec<Option<Challenge>> {
        perm_traces
            .iter()
            .map(|perm_trace| {
                perm_trace.as_ref().map(|perm_trace| {
                    *perm_trace
                        .row_slice(perm_trace.height() - 1)
                        .last()
                        .unwrap()
                })
            })
            .collect()
    }

    // Copied from valida/machine/src/chip.rs, modified to allow partitioned main trace
    /// Generate the permutation trace for a chip given the main trace.
    /// The permutation randomness is only available after the main trace from all chips
    /// involved in interactions have been committed.
    ///
    /// - `partitioned_main` is the main trace, partitioned into several matrices of the same height
    ///
    /// Returns the permutation trace as a matrix of extension field elements.
    ///
    /// ## Panics
    /// - If `partitioned_main` is empty.
    pub fn generate_after_challenge_trace(
        all_interactions: &[SymbolicInteraction<F>],
        trace_view: &PairTraceView<F>,
        permutation_randomness: &[Challenge; STARK_LU_NUM_CHALLENGES],
        interaction_partitions: &[Vec<usize>],
    ) -> Option<RowMajorMatrix<Challenge>>
    where
        F: Field,
        Challenge: ExtensionField<F>,
    {
        if all_interactions.is_empty() {
            return None;
        }
        let &[alpha, beta] = permutation_randomness;

        let alphas = generate_rlc_elements(alpha, all_interactions);
        let betas = generate_betas(beta, all_interactions);

        // Compute the reciprocal columns
        //
        // For every row we do the following
        // We first compute the reciprocals: r_1, r_2, ..., r_n, where
        // r_i = \frac{1}{\alpha^i + \sum_j \beta^j * f_{i, j}}, where
        // f_{i, j} is the jth main trace column for the ith interaction
        //
        // We then bundle every interaction_chunk_size interactions together
        // to get the value perm_i = \sum_{i \in bundle} r_i * m_i, where m_i
        // is the signed count for the interaction.
        //
        // Finally, the last column, \phi, of every row is the running sum of
        // all the previous perm values
        //
        // Row: | perm_1 | perm_2 | perm_3 | ... | perm_s | phi |, where s
        // is the number of bundles
        let num_interactions = all_interactions.len();
        let height = trace_view.partitioned_main[0].height();

        // Note: we could precompute this and include in the proving key, but this should be
        // a fast scan and only done once per AIR and not per row, so it is more ergonomic to compute
        // on the fly. If we introduce a more advanced chunking algorithm, then we will need to
        // cache the chunking information in the proving key.
        let chunk_end = find_interaction_chunks(all_interactions, max_constraint_degree);
        let chunk_split_at = iter::once(chunk_end[0])
            .chain(chunk_end.windows(2).map(|w| w[1] - w[0]))
            .collect_vec();
        let perm_width = chunk_end.len() + 1;
        let mut perm_values = Challenge::zero_vec(height * perm_width);
        debug_assert!(
            trace_view
                .partitioned_main
                .iter()
                .all(|m| m.height() == height),
            "All main trace parts must have same height"
        );

        // To optimize memory and parallelism, we split the trace rows into chunks
        // based on the number of cpu threads available, and then do all
        // computations necessary for that chunk within a single thread.
        #[cfg(feature = "parallel")]
        let num_threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let num_threads = 1;

        let preprocessed = trace_view.preprocessed.as_ref().map(|m| m.as_view());
        let partitioned_main = trace_view
            .partitioned_main
            .iter()
            .map(|m| m.as_view())
            .collect_vec();
        let evaluator = |local_index: usize| Evaluator {
            preprocessed: &preprocessed,
            partitioned_main: &partitioned_main,
            public_values: &trace_view.public_values,
            height,
            local_index,
        };
        let height_per_thread = height.div_ceil(num_threads);
        perm_values
            .par_chunks_mut(height_per_thread * perm_width)
            .enumerate()
            .for_each(|(thread_idx, perm_values)| {
                // perm_values is now local_height x perm_width row-major matrix
                let num_rows = perm_values.len() / perm_width;
                // the interaction chunking requires more memory because we must
                // allocate separate memory for the denominators and reciprocals
                let mut denoms = Challenge::zero_vec(num_rows * num_interactions);
                let row_offset = thread_idx * height_per_thread;
                // compute the denominators to be inverted:
                for (n, denom_row) in denoms.chunks_exact_mut(num_interactions).enumerate() {
                    let evaluator = evaluator(row_offset + n);
                    for (denom, interaction) in denom_row.iter_mut().zip(all_interactions.iter()) {
                        let alpha = alphas[interaction.bus_index];
                        debug_assert!(interaction.fields.len() <= betas.len());
                        let mut fields = interaction.fields.iter();
                        *denom = alpha
                            + evaluator
                                .eval_expr(fields.next().expect("fields should not be empty"));
                        for (expr, &beta) in fields.zip(betas.iter().skip(1)) {
                            *denom += beta * evaluator.eval_expr(expr);
                        }
                    }
                }

                // Zero should be vanishingly unlikely if alpha, beta are properly pseudo-randomized
                // The logup reciprocals should never be zero, so trace generation should panic if
                // trying to divide by zero.
                let reciprocals = p3_field::batch_multiplicative_inverse(&denoms);
                drop(denoms);
                // For loop over rows in same thread:
                // This block should already be in a single thread, but rayon is able
                // to do more magic sometimes
                perm_values
                    .par_chunks_exact_mut(perm_width)
                    .zip(reciprocals.par_chunks_exact(num_interactions))
                    .enumerate()
                    .for_each(|(n, (perm_row, reciprocals))| {
                        debug_assert_eq!(perm_row.len(), perm_width);
                        debug_assert_eq!(reciprocals.len(), num_interactions);
                        let mut remaining_reciprocals = reciprocals;
                        let mut remaining_interactions = all_interactions;

                        let evaluator = evaluator(row_offset + n);
                        let mut row_sum = Challenge::ZERO;
                        for (&mid, perm_val) in zip(chunk_split_at.iter(), perm_row.iter_mut()) {
                            let reciprocal_chunk;
                            let interaction_chunk;
                            (reciprocal_chunk, remaining_reciprocals) =
                                remaining_reciprocals.split_at(mid);
                            (interaction_chunk, remaining_interactions) =
                                remaining_interactions.split_at(mid);
                            for (reciprocal, interaction) in
                                zip(reciprocal_chunk, interaction_chunk)
                            {
                                let mut interaction_val =
                                    *reciprocal * evaluator.eval_expr(&interaction.count);
                                if interaction.interaction_type == InteractionType::Receive {
                                    interaction_val = -interaction_val;
                                }
                                *perm_val += interaction_val;
                            }
                            row_sum += *perm_val;
                        }

                        perm_row[perm_width - 1] = row_sum;
                    });
            });

        // At this point, the trace matrix is complete except that the last column
        // has the row sum but not the partial sum
        tracing::trace_span!("compute logup partial sums").in_scope(|| {
            let mut phi = Challenge::ZERO;
            for perm_chunk in perm_values.chunks_exact_mut(perm_width) {
                phi += *perm_chunk.last().unwrap();
                *perm_chunk.last_mut().unwrap() = phi;
            }
        });

        Some(RowMajorMatrix::new(perm_values, perm_width))
    }
}

// Initial version taken from valida/machine/src/chip.rs under MIT license.
//
/// The permutation row consists of 1 column for each bundle of interactions
/// and one column for the partial sum of log derivative. These columns are trace columns
/// "after challenge" phase 0, and they are valued in the extension field.
/// For more details, see the comment in the trace.rs file
pub fn eval_fri_log_up_phase<AB>(
    builder: &mut AB,
    symbolic_interactions: &[SymbolicInteraction<AB::F>],
    max_constraint_degree: usize,
) where
    AB: InteractionBuilder + PermutationAirBuilderWithExposedValues,
{
    let exposed_values = builder.permutation_exposed_values();
    // There are interactions, add constraints for the virtual columns
    assert_eq!(
        exposed_values.len(),
        1,
        "Should have one exposed value for cumulative_sum"
    );
    let cumulative_sum = exposed_values[0];

    let rand_elems = builder.permutation_randomness();

    let perm = builder.permutation();
    let (perm_local, perm_next) = (perm.row_slice(0), perm.row_slice(1));
    let perm_local: &[AB::VarEF] = (*perm_local).borrow();
    let perm_next: &[AB::VarEF] = (*perm_next).borrow();

    let all_interactions = builder.all_interactions().to_vec();
    let chunk_end = find_interaction_chunks(symbolic_interactions, max_constraint_degree);
    debug_assert_eq!(chunk_end.len() + 1, perm_local.len());

    let phi_local = *perm_local.last().unwrap();
    let phi_next = *perm_next.last().unwrap();

    let alphas = generate_rlc_elements(rand_elems[0].into(), &all_interactions);
    let betas = generate_betas(rand_elems[1].into(), &all_interactions);

    let phi_lhs = phi_next.into() - phi_local.into();
    let mut phi_rhs = AB::ExprEF::ZERO;
    let mut phi_0 = AB::ExprEF::ZERO;

    for (chunk_idx, &end) in chunk_end.iter().enumerate() {
        let start = if chunk_idx == 0 {
            0
        } else {
            chunk_end[chunk_idx - 1]
        };
        let interaction_chunk = &all_interactions[start..end];

        let denoms_per_chunk = interaction_chunk
            .iter()
            .map(|interaction| {
                assert!(!interaction.fields.is_empty(), "fields should not be empty");
                let mut field_hash = AB::ExprEF::ZERO;
                for (field, beta) in interaction.fields.iter().zip(betas.iter()) {
                    field_hash += beta.clone() * field.clone();
                }
                field_hash + alphas[interaction.bus_index].clone()
            })
            .collect_vec();

        let mut row_lhs: AB::ExprEF = perm_local[chunk_idx].into();
        for denom in denoms_per_chunk.iter() {
            row_lhs *= denom.clone();
        }

        let mut row_rhs = AB::ExprEF::ZERO;
        for (i, interaction) in interaction_chunk.iter().cloned().enumerate() {
            let mut term: AB::ExprEF = interaction.count.into();
            if interaction.interaction_type == InteractionType::Receive {
                term = -term;
            }
            for (j, denom) in denoms_per_chunk.iter().enumerate() {
                if i != j {
                    term *= denom.clone();
                }
            }
            row_rhs += term;
        }

        // Some analysis on the degrees of row_lhs and row_rhs:
        //
        // Let max_field_degree be the maximum degree of all fields across all interactions
        // for the AIR. Define max_count_degree similarly for the counts of the interactions.
        //
        // By construction, the degree of row_lhs is bounded by 1 + sum_i(max_field_degree),
        // and the degree of row_rhs is bounded by max_i(max_count_degree_i + sum_{j!=i}(max_field_degree_j))
        builder.assert_eq_ext(row_lhs, row_rhs);

        phi_0 += perm_local[chunk_idx].into();
        phi_rhs += perm_next[chunk_idx].into();
    }

    // Running sum constraints
    builder.when_transition().assert_eq_ext(phi_lhs, phi_rhs);
    builder
        .when_first_row()
        .assert_eq_ext(*perm_local.last().unwrap(), phi_0);
    builder
        .when_last_row()
        .assert_eq_ext(*perm_local.last().unwrap(), cumulative_sum);
}

/// We can chunk interactions, where the degree of the dominating logup constraint is bounded by
///
///     logup_degree = max(
///         1 + sum_i(max_field_degree_i),
///         max_i(count_degree_i + sum_{j!=i}(max_field_degree_j))
///     )
/// where i,j refer to interactions in the chunk.
///
/// More details about this can be found in the function [eval_fri_log_up_phase].
///
/// We greedily pack interactions into chunks while making sure the constraint
/// degree does not exceed `max_constraint_degree` (if possible).
/// `max_constraint_degree` is the maximum constraint degree across all AIRs.
/// Greedy packing means that we do not reorder the interactions.
///
/// Returns `chunk_end: Vec<usize>` where `num_chunks = chunk_end.len()` and the interactions
/// are chunked into `0..chunk_end[0], chunk_end[0]..chunk_end[1], ...`.
/// If `max_constraint_degree == 0`, then `num_chunks = interactions.len()` and no chunking is done.
///
/// ## Panics
/// If `max_constraint_degree > 0` and there are interactions that cannot fit in a singleton chunk.
pub(crate) fn find_interaction_chunks<F: Field>(
    interactions: &[SymbolicInteraction<F>],
    max_constraint_degree: usize,
) -> Vec<usize> {
    if interactions.is_empty() {
        return vec![];
    }
    let mut running_sum_field_degree = 0;
    let mut numerator_max_degree = 0;
    let mut chunk_idx = vec![];
    for (i, interaction) in interactions.iter().enumerate() {
        let field_degree = interaction
            .fields
            .iter()
            .map(|f| f.degree_multiple())
            .max()
            .unwrap_or(0);
        let count_degree = interaction.count.degree_multiple();
        // Can we add this interaction to the current chunk?
        let new_num_max_degree = max(
            numerator_max_degree + field_degree,
            count_degree + running_sum_field_degree,
        );
        let new_denom_degree = running_sum_field_degree + field_degree;
        if max(new_num_max_degree, new_denom_degree + 1) <= max_constraint_degree {
            // include in current chunk
            numerator_max_degree = new_num_max_degree;
            running_sum_field_degree += field_degree;
        } else {
            // seal current chunk + start new chunk
            if i > 0 {
                // if i == 0, that means the interaction exceeds the max_constraint_degree
                chunk_idx.push(i);
            }
            numerator_max_degree = count_degree;
            running_sum_field_degree = field_degree;
            if max_constraint_degree > 0
                && max(count_degree, field_degree + 1) > max_constraint_degree
            {
                panic!("Interaction with field_degree={field_degree}, count_degree={count_degree} exceeds max_constraint_degree={max_constraint_degree}");
            }
        }
    }
    // the last interaction is in a chunk that has not been sealed
    chunk_idx.push(interactions.len());
    chunk_idx
}
