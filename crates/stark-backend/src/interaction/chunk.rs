use std::{cmp::max, iter::zip, mem};

use p3_field::Field;
use serde::{Deserialize, Serialize};

use super::SymbolicInteraction;
use crate::air_builders::symbolic::{
    symbolic_expression::SymbolicExpression, symbolic_variable::SymbolicVariable,
};

/// See [`find_interaction_chunks`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InteractionChunks {
    indices_by_chunk: Vec<Vec<usize>>,
}

impl InteractionChunks {
    /// Returns disjoint sets of indices specifying the interaction chunks.
    /// The indices are with respect to the `symbolic_interactions` list.
    pub fn indices_by_chunk(&self) -> &[Vec<usize>] {
        &self.indices_by_chunk
    }

    pub fn num_chunks(&self) -> usize {
        self.indices_by_chunk.len()
    }

    /// Returns a symbolic expresion pair consisting of `(numerator, denominator)` for the logup
    /// fraction corresponding to each interaction chunk. The returned vector has length equal to
    /// `num_chunks()`.
    ///
    /// The provided symbolic `interactions` must be in the same order as what was used to compute
    /// the interaction chunks.
    pub fn symbolic_logup_fractions<F: Field>(
        &self,
        interactions: &[SymbolicInteraction<F>],
        beta_pows: &[SymbolicVariable<F>],
    ) -> Vec<(SymbolicExpression<F>, SymbolicExpression<F>)> {
        self.indices_by_chunk
            .iter()
            .map(|indices| {
                let interactions_in_chunk = indices
                    .iter()
                    .map(|&i| interactions[i].clone())
                    .collect::<Vec<_>>();
                symbolic_logup_fraction(interactions_in_chunk, beta_pows)
            })
            .collect()
    }
}

/// We can chunk interactions, where the degree of the dominating logup constraint is bounded by
///
/// logup_degree = max(
///     sum_i(max_field_degree_i),
///     max_i(count_degree_i + sum_{j!=i}(max_field_degree_j))
/// )
/// where i,j refer to interactions in the chunk.
///
/// We pack interactions into chunks while making sure the constraint
/// degree does not exceed `max_constraint_degree` (if possible).
/// `max_constraint_degree` is the maximum constraint degree across all AIRs.
/// Interactions may be reordered in the process.
///
/// Returns [InteractionChunks] which consists of `indices_by_chunk: Vec<Vec<usize>>` where
/// `num_chunks = indices_by_chunk.len()`.
/// This function guarantees that the `indices_by_chunk` forms a (disjoint) partition of the
/// indices `0..interactions.len()`. For `chunk_idx`, the array `indices_by_chunk[chunk_idx]`
/// contains the indices of interactions that are in the `chunk_idx`-th chunk.
///
/// If `max_constraint_degree == 0`, then `num_chunks = interactions.len()` and no chunking is done.
///
/// ## Note
/// This function is only intended for use in preprocessing, and is not used in proving.
///
/// ## Panics
/// If `max_constraint_degree > 0` and there are interactions that cannot fit in a singleton
///   chunk.
pub fn find_interaction_chunks<F: Field>(
    interactions: &[SymbolicInteraction<F>],
    max_constraint_degree: usize,
) -> InteractionChunks {
    if interactions.is_empty() {
        return InteractionChunks::default();
    }
    let mut interaction_idxs: Vec<usize> = (0..interactions.len()).collect();
    interaction_idxs.sort_by(|&i, &j| {
        let a = &interactions[i];
        let b = &interactions[j];
        a.max_message_degree()
            .cmp(&b.max_message_degree())
            .then(a.count.degree_multiple().cmp(&b.count.degree_multiple()))
    });
    // Now we greedily pack
    let mut running_sum_field_degree = 0;
    let mut numerator_max_degree = 0;
    let mut indices_by_chunk = vec![];
    let mut cur_chunk = vec![];
    for interaction_idx in interaction_idxs {
        let interaction = &interactions[interaction_idx];
        let msg_degree = interaction.max_message_degree();
        let count_degree = interaction.count.degree_multiple();
        // Can we add this interaction to the current chunk?
        let new_num_max_degree = max(
            numerator_max_degree + msg_degree,
            count_degree + running_sum_field_degree,
        );
        let new_denom_degree = running_sum_field_degree + msg_degree;
        if max(new_num_max_degree, new_denom_degree) <= max_constraint_degree {
            // include in current chunk
            cur_chunk.push(interaction_idx);
            numerator_max_degree = new_num_max_degree;
            running_sum_field_degree += msg_degree;
        } else {
            // seal current chunk + start new chunk
            if !cur_chunk.is_empty() {
                // if i == 0, that means the interaction exceeds the max_constraint_degree
                indices_by_chunk.push(mem::take(&mut cur_chunk));
            }
            cur_chunk.push(interaction_idx);
            numerator_max_degree = count_degree;
            running_sum_field_degree = msg_degree;
            if max_constraint_degree > 0 && max(count_degree, msg_degree) > max_constraint_degree {
                panic!("Interaction with field_degree={msg_degree}, count_degree={count_degree} exceeds max_constraint_degree={max_constraint_degree}");
            }
        }
    }
    // the last interaction is in a chunk that has not been sealed
    assert!(!cur_chunk.is_empty());
    indices_by_chunk.push(cur_chunk);

    InteractionChunks { indices_by_chunk }
}

/// Computes the sum of the logup fractions corresponding to symbolic `interactions` provided. The
/// sum is expressed as a single `(numerator, denominator)` symbolic expression pair representing
/// the fraction.
///
/// The `beta_pows` should be of length at least `max_message_length + 1` and consist of
/// `SymbolicVariable`s with `Entry::Challenge`.
pub fn symbolic_logup_fraction<F: Field>(
    interactions: Vec<SymbolicInteraction<F>>,
    beta_pows: &[SymbolicVariable<F>],
) -> (SymbolicExpression<F>, SymbolicExpression<F>) {
    let mut frac: Option<(SymbolicExpression<F>, SymbolicExpression<F>)> = None;

    for interaction in interactions {
        let logup_num = interaction.count;
        let msg_len = interaction.message.len();
        assert!(msg_len <= beta_pows.len());
        let b = F::from_canonical_u32(interaction.bus_index as u32 + 1);
        let logup_denom = zip(interaction.message, beta_pows)
            .fold(beta_pows[msg_len] * b, |h_beta, (msg_j, &beta_j)| {
                h_beta + msg_j * beta_j
            });

        frac = Some(match frac {
            None => (logup_num, logup_denom),
            Some((num, denom)) => (
                num * logup_denom.clone() + logup_num * denom.clone(),
                denom * logup_denom,
            ),
        });
    }

    frac.unwrap()
}
