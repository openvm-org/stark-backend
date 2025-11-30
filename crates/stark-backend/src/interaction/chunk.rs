use std::{cmp::max, mem};

use itertools::Itertools;
use p3_field::Field;
use serde::{Deserialize, Serialize};

use super::SymbolicInteraction;

#[derive(Clone, Default, Serialize, Deserialize)]
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
}

/// We can chunk interactions, where the degree of the dominating logup constraint is bounded by
///
/// logup_degree = max(
///     1 + sum_i(max_field_degree_i),
///     max_i(count_degree_i + sum_{j!=i}(max_field_degree_j))
/// )
/// where i,j refer to interactions in the chunk.
///
/// More details about this can be found in the function [eval_fri_log_up_phase].
///
/// We pack interactions into chunks while making sure the constraint
/// degree does not exceed `max_constraint_degree` (if possible).
/// `max_constraint_degree` is the maximum constraint degree across all AIRs.
/// Interactions may be reordered in the process.
///
/// Returns [FriLogUpProvingKey] which consists of `interaction_partitions: Vec<Vec<usize>>` where
/// `num_chunks = interaction_partitions.len()`.
/// This function guarantees that the `interaction_partitions` forms a (disjoint) partition of the
/// indices `0..interactions.len()`. For `chunk_idx`, the array `interaction_partitions[chunk_idx]`
/// contains the indices of interactions that are in the `chunk_idx`-th chunk.
///
/// If `max_constraint_degree == 0`, then `num_chunks = interactions.len()` and no chunking is done.
///
/// ## Note
/// This function is only intended for use in preprocessing, and is not used in proving.
///
/// ## Panics
/// If `max_constraint_degree > 0` and there are interactions that cannot fit in a singleton chunk.
pub fn find_interaction_chunks<F: Field>(
    interactions: &[SymbolicInteraction<F>],
    max_constraint_degree: usize,
) -> InteractionChunks {
    if interactions.is_empty() {
        return InteractionChunks::default();
    }
    // First, we sort interaction indices by ascending max field degree
    let max_field_degree = |i: usize| {
        interactions[i]
            .message
            .iter()
            .map(|f| f.degree_multiple())
            .max()
            .unwrap_or(0)
    };
    let mut interaction_idxs = (0..interactions.len()).collect_vec();
    interaction_idxs.sort_by(|&i, &j| {
        let field_cmp = max_field_degree(i).cmp(&max_field_degree(j));
        if field_cmp == std::cmp::Ordering::Equal {
            interactions[i]
                .count
                .degree_multiple()
                .cmp(&interactions[j].count.degree_multiple())
        } else {
            field_cmp
        }
    });
    // Now we greedily pack
    let mut running_sum_field_degree = 0;
    let mut numerator_max_degree = 0;
    let mut indices_by_chunk = vec![];
    let mut cur_chunk = vec![];
    for interaction_idx in interaction_idxs {
        let field_degree = max_field_degree(interaction_idx);
        let count_degree = interactions[interaction_idx].count.degree_multiple();
        // Can we add this interaction to the current chunk?
        let new_num_max_degree = max(
            numerator_max_degree + field_degree,
            count_degree + running_sum_field_degree,
        );
        let new_denom_degree = running_sum_field_degree + field_degree;
        if max(new_num_max_degree, new_denom_degree + 1) <= max_constraint_degree {
            // include in current chunk
            cur_chunk.push(interaction_idx);
            numerator_max_degree = new_num_max_degree;
            running_sum_field_degree += field_degree;
        } else {
            // seal current chunk + start new chunk
            if !cur_chunk.is_empty() {
                // if i == 0, that means the interaction exceeds the max_constraint_degree
                indices_by_chunk.push(mem::take(&mut cur_chunk));
            }
            cur_chunk.push(interaction_idx);
            numerator_max_degree = count_degree;
            running_sum_field_degree = field_degree;
            if max_constraint_degree > 0 && max(count_degree, field_degree) > max_constraint_degree
            {
                panic!("Interaction with field_degree={field_degree}, count_degree={count_degree} exceeds max_constraint_degree={max_constraint_degree}");
            }
        }
    }
    // the last interaction is in a chunk that has not been sealed
    assert!(!cur_chunk.is_empty());
    indices_by_chunk.push(cur_chunk);

    InteractionChunks { indices_by_chunk }
}
