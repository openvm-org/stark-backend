use std::cmp::max;

use itertools::Itertools;
use p3_commit::PolynomialSpace;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PackedValue};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use super::{
    evaluator::{ProverConstraintEvaluator, ViewPair},
    QuotientChunk,
};
use crate::{
    air_builders::symbolic::{
        symbolic_variable::Entry, SymbolicExpressionDag, SymbolicExpressionNode,
    },
    config::{Domain, PackedChallenge, PackedVal, StarkGenericConfig, Val},
    prover::cpu::transmute_to_base,
};

// Starting reference: p3_uni_stark::prover::quotient_values
// (many changes have been made since then)
/// Computes evaluation of DEEP quotient polynomial on the quotient domain for a single RAP (single trace matrix).
///
/// Designed to be general enough to support RAP with multiple rounds of challenges.
///
/// **Note**: This function assumes that the
/// `quotient_domain.split_evals(quotient_degree, quotient_flat)` function from Plonky3 works
/// as follows (currently true for all known implementations):
/// The quotient polynomial will is treated as long columns of the form
/// ```
/// [q_0]
/// [q_1]
/// ...
/// [q_{quotient_degree - 1}]
/// ```
/// where each `q_i` is column of length `trace_height` of extension field elements.
/// We treat them as separate base field matrices
/// ```
/// [q_0], [q_1], ..., [q_{quotient_degree - 1}]
/// ```
/// Each matrix is a "chunk".
#[allow(clippy::too_many_arguments)]
#[instrument(
    name = "compute single RAP quotient polynomial",
    level = "trace",
    skip_all
)]
pub fn compute_single_rap_quotient_values<'a, SC, M>(
    constraints: &SymbolicExpressionDag<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_trace_on_quotient_domain: Option<M>,
    partitioned_main_lde_on_quotient_domain: Vec<M>,
    after_challenge_lde_on_quotient_domain: Vec<M>,
    // For each challenge round, the challenges drawn
    challenges: &'a [Vec<PackedChallenge<SC>>],
    alpha: SC::Challenge,
    public_values: &'a [Val<SC>],
    // Values exposed to verifier after challenge round i
    exposed_values_after_challenge: &'a [Vec<PackedChallenge<SC>>],
    extra_capacity_bits: usize,
) -> Vec<QuotientChunk<SC>>
where
    SC: StarkGenericConfig,
    M: Matrix<Val<SC>>,
{
    let quotient_size = quotient_domain.size();
    let trace_height = trace_domain.size();
    assert!(partitioned_main_lde_on_quotient_domain
        .iter()
        .all(|m| m.height() >= quotient_size));
    assert!(after_challenge_lde_on_quotient_domain
        .iter()
        .all(|m| m.height() >= quotient_size));
    let preprocessed_width = preprocessed_trace_on_quotient_domain
        .as_ref()
        .map(|m| m.width())
        .unwrap_or(0);
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_size) - log2_strict_usize(trace_height);
    let quotient_degree = 1 << qdb;
    debug_assert_eq!(quotient_size, trace_height * quotient_degree);
    // The input values are evaluations on quotient domain, so the trace evaluations are spaced quotient_degree apart
    let next_step = quotient_degree;

    let ext_degree = SC::Challenge::D;

    let mut alpha_powers = alpha
        .powers()
        .take(constraints.constraint_idx.len())
        .map(PackedChallenge::<SC>::from_f)
        .collect_vec();
    // We want alpha powers to have highest power first, because of how accumulator "folding" works
    // So this will be alpha^{num_constraints - 1}, ..., alpha^0
    alpha_powers.reverse();

    // assert!(quotient_size >= PackedVal::<SC>::WIDTH);
    // We take PackedVal::<SC>::WIDTH worth of values at a time from a quotient_size slice, so we need to
    // pad with default values in the case where quotient_size is smaller than PackedVal::<SC>::WIDTH.
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_zeroifier.push(Val::<SC>::default());
    }

    // Scan constraints to see if we need `next` row and also check index bounds
    // so we don't need to check them per row.
    let mut rotation = 0;
    for node in &constraints.nodes {
        if let SymbolicExpressionNode::Variable(var) = node {
            match var.entry {
                Entry::Preprocessed { offset } => {
                    rotation = max(rotation, offset);
                    assert!(var.index < preprocessed_width);
                    assert!(
                        preprocessed_trace_on_quotient_domain
                            .as_ref()
                            .unwrap()
                            .height()
                            >= quotient_size
                    );
                }
                Entry::Main { part_index, offset } => {
                    rotation = max(rotation, offset);
                    assert!(
                        var.index < partitioned_main_lde_on_quotient_domain[part_index].width()
                    );
                }
                Entry::Public => {
                    assert!(var.index < public_values.len());
                }
                Entry::Permutation { offset } => {
                    rotation = max(rotation, offset);
                    let ext_width = after_challenge_lde_on_quotient_domain
                        .first()
                        .expect("Challenge phase not supported")
                        .width()
                        / ext_degree;
                    assert!(var.index < ext_width);
                }
                Entry::Challenge => {
                    assert!(
                        var.index
                            < challenges
                                .first()
                                .expect("Challenge phase not supported")
                                .len()
                    );
                }
                Entry::Exposed => {
                    assert!(
                        var.index
                            < exposed_values_after_challenge
                                .first()
                                .expect("Challenge phase not supported")
                                .len()
                    );
                }
            }
        }
    }
    let needs_next = rotation > 0;

    let qc_domains = quotient_domain.split_domains(quotient_degree);
    qc_domains
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, chunk_domain)| {
            // This will be evaluations of the quotient poly on the `chunk_domain`, where `chunk_domain.size() = trace_height`. We reserve extra capacity for the coset lde in the pcs.commit of this chunk.
            let mut chunk = SC::Challenge::zero_vec(trace_height << extra_capacity_bits);
            chunk.truncate(trace_height);
            // We parallel iterate over "fat" rows, which are consecutive rows packed for SIMD.
            // Use par_chunks instead of par_chunks_exact in case trace_height is not a multiple of PackedVal::WIDTH
            chunk
                .par_chunks_mut(PackedVal::<SC>::WIDTH)
                .enumerate()
                .for_each(|(fat_row_idx, packed_ef_mut)| {
                    let i_start = chunk_idx * trace_height + fat_row_idx * PackedVal::<SC>::WIDTH;
                    let wrap = |i| i % quotient_size;
                    let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

                    let [row_idx_local, row_idx_next] = [0, next_step].map(|shift| {
                        (0..PackedVal::<SC>::WIDTH)
                            .map(|offset| wrap(i_start + offset + shift))
                            .collect::<Vec<_>>()
                    });
                    let row_idx_local = Some(row_idx_local);
                    let row_idx_next = needs_next.then_some(row_idx_next);

                    let is_first_row =
                        *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
                    let is_last_row =
                        *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
                    let is_transition =
                        *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
                    let inv_zeroifier =
                        *PackedVal::<SC>::from_slice(&sels.inv_zeroifier[i_range.clone()]);

                    // Vertically pack rows of each matrix,
                    // skipping `next` if above scan showed no constraints need it:

                    let [preprocessed_local, preprocessed_next] = [&row_idx_local, &row_idx_next]
                        .map(|wrapped_idx| {
                            wrapped_idx.as_ref().map(|wrapped_idx| {
                                (0..preprocessed_width)
                                    .map(|col| {
                                        PackedVal::<SC>::from_fn(|offset| {
                                            preprocessed_trace_on_quotient_domain
                                                .as_ref()
                                                .unwrap()
                                                .get(wrapped_idx[offset], col)
                                        })
                                    })
                                    .collect_vec()
                            })
                        });
                    let preprocessed_pair =
                        ViewPair::new(preprocessed_local.unwrap(), preprocessed_next);

                    let partitioned_main_pairs = partitioned_main_lde_on_quotient_domain
                        .iter()
                        .map(|lde| {
                            let width = lde.width();
                            let [local, next] =
                                [&row_idx_local, &row_idx_next].map(|wrapped_idx| {
                                    wrapped_idx.as_ref().map(|wrapped_idx| {
                                        (0..width)
                                            .map(|col| {
                                                PackedVal::<SC>::from_fn(|offset| {
                                                    lde.get(wrapped_idx[offset], col)
                                                })
                                            })
                                            .collect_vec()
                                    })
                                });
                            ViewPair::new(local.unwrap(), next)
                        })
                        .collect_vec();

                    let after_challenge_pairs = after_challenge_lde_on_quotient_domain
                        .iter()
                        .map(|lde| {
                            // Width in base field with extension field elements flattened
                            let base_width = lde.width();
                            let [local, next] =
                                [&row_idx_local, &row_idx_next].map(|wrapped_idx| {
                                    wrapped_idx.as_ref().map(|wrapped_idx| {
                                        (0..base_width)
                                            .step_by(ext_degree)
                                            .map(|col| {
                                                PackedChallenge::<SC>::from_base_fn(|i| {
                                                    PackedVal::<SC>::from_fn(|offset| {
                                                        lde.get(wrapped_idx[offset], col + i)
                                                    })
                                                })
                                            })
                                            .collect_vec()
                                    })
                                });
                            ViewPair::new(local.unwrap(), next)
                        })
                        .collect_vec();

                    let evaluator: ProverConstraintEvaluator<SC> = ProverConstraintEvaluator {
                        preprocessed: preprocessed_pair,
                        partitioned_main: partitioned_main_pairs,
                        after_challenge: after_challenge_pairs,
                        challenges,
                        is_first_row,
                        is_last_row,
                        is_transition,
                        public_values,
                        exposed_values_after_challenge,
                    };
                    let accumulator = evaluator.accumulate(constraints, &alpha_powers);
                    // quotient(x) = constraints(x) / Z_H(x)
                    let quotient: PackedChallenge<SC> = accumulator * inv_zeroifier;

                    // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
                    for (idx_in_packing, ef) in packed_ef_mut.iter_mut().enumerate() {
                        *ef = SC::Challenge::from_base_fn(|coeff_idx| {
                            quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing]
                        });
                    }
                });
            // Flatten from extension field elements to base field elements
            // SAFETY: `Challenge` is assumed to be extension field of `F`
            // with memory layout `[F; Challenge::D]`
            let matrix = unsafe { transmute_to_base(RowMajorMatrix::new_col(chunk)) };
            QuotientChunk {
                domain: chunk_domain,
                matrix,
            }
        })
        .collect()
}
