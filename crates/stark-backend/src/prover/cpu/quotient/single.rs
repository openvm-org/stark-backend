use std::cmp::{max, min};

use itertools::Itertools;
use p3_commit::PolynomialSpace;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PackedValue};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
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
    utils::parallelize_chunks,
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
/// The evaluations of the quotient polynomial on the quotient domain (shift of a subgroup) is viewed as a long column of the form
/// ```ignore
/// [q_{0,0}]
/// [q_{1,0}]
/// ...
/// [q_{quotient_degree - 1,0}]
/// [q_{0,1}]
/// ...
/// [q_{quotient_degree - 1, trace_height - 1}]
/// ```
/// which is "vertically strided" with stride `quotient_degree`.
/// We regroup them into evaluations on cosets of the trace domain subgroup as separate base field matrices
/// ```ignore
/// [q_{0,0}               ]   [q_{1,0}               ]  ...  [q_{quotient_degree - 1,0}               ]
/// [q_{0,1}               ]   [q_{1,1}               ]  ...  [q_{quotient_degree - 1,1}               ]
/// ...
/// [q_{0,trace_height - 1}]   [q_{1,trace_height - 1}]  ...  [q_{quotient_degree - 1,trace_height - 1}]
/// ```
/// where `q_{0,*}` and `q_{1,*}` are separate matrices. Each matrix is called a "chunk".
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
    let sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_size) - log2_strict_usize(trace_height);
    let quotient_degree = 1 << qdb;
    debug_assert_eq!(quotient_size, trace_height * quotient_degree);

    let ext_degree = SC::Challenge::D;

    let mut alpha_powers = alpha
        .powers()
        .take(constraints.constraint_idx.len())
        .map(PackedChallenge::<SC>::from_f)
        .collect_vec();
    // We want alpha powers to have highest power first, because of how accumulator "folding" works
    // So this will be alpha^{num_constraints - 1}, ..., alpha^0
    alpha_powers.reverse();

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
            // If trace_height is smaller than PackedVal::<SC>::WIDTH, we just don't parallelize
            let simd_width = min(trace_height, PackedVal::<SC>::WIDTH);
            parallelize_chunks(&mut chunk, simd_width, |chunk, start_row_idx| {
                debug_assert_eq!(start_row_idx % PackedVal::<SC>::WIDTH, 0);

                // Pre-allocate vectors
                let mut row_idx_local = Vec::with_capacity(PackedVal::<SC>::WIDTH);
                let mut row_idx_next = Vec::with_capacity(PackedVal::<SC>::WIDTH);

                fn new_view_pair<T>(width: usize, needs_next: bool) -> ViewPair<T> {
                    let local = Vec::with_capacity(width);
                    let next = needs_next.then(|| Vec::with_capacity(width));
                    ViewPair::new(local, next)
                }

                let mut preprocessed_pair: ViewPair<PackedVal<SC>> =
                    new_view_pair(preprocessed_width, needs_next);
                let mut partitioned_main_pairs: Vec<ViewPair<PackedVal<SC>>> =
                    partitioned_main_lde_on_quotient_domain
                        .iter()
                        .map(|lde| new_view_pair(lde.width(), needs_next))
                        .collect();
                let mut after_challenge_pairs: Vec<ViewPair<PackedChallenge<SC>>> =
                    after_challenge_lde_on_quotient_domain
                        .iter()
                        .map(|lde| new_view_pair(lde.width() / ext_degree, needs_next))
                        .collect();
                let mut node_exprs = Vec::with_capacity(constraints.nodes.len());

                // Use chunks instead of chunks_exact in case trace_height is not a multiple of PackedVal::WIDTH
                for (local_fat_row_idx, packed_ef_mut) in
                    chunk.chunks_mut(PackedVal::<SC>::WIDTH).enumerate()
                {
                    let row_idx = start_row_idx + local_fat_row_idx * PackedVal::<SC>::WIDTH;
                    // `packed_ef_mut` is a vertical sub-column, index `offset` of `packed_ef_mut`
                    // is supposed to be the `chunk_row_idx = row_idx + offset` row of the chunk matrix
                    // which is the `chunk_idx + chunk_row_idx * quotient_degree`th row of the evaluation of quotient polynomial on the quotient domain
                    // PERF[jpw]: This may not be cache friendly - would it be better to generate the quotient values in order first and then do some in-place permutation?
                    let quot_row_idx =
                        |offset| (chunk_idx + (row_idx + offset) * quotient_degree) % quotient_size;

                    row_idx_local.clear();
                    row_idx_next.clear();
                    for offset in 0..PackedVal::<SC>::WIDTH {
                        row_idx_local.push(quot_row_idx(offset));
                        row_idx_next.push(quot_row_idx(offset + 1));
                    }

                    let is_first_row =
                        PackedVal::<SC>::from_fn(|offset| sels.is_first_row[quot_row_idx(offset)]);
                    let is_last_row =
                        PackedVal::<SC>::from_fn(|offset| sels.is_last_row[quot_row_idx(offset)]);
                    let is_transition =
                        PackedVal::<SC>::from_fn(|offset| sels.is_transition[quot_row_idx(offset)]);
                    let inv_zeroifier =
                        PackedVal::<SC>::from_fn(|offset| sels.inv_zeroifier[quot_row_idx(offset)]);

                    // Vertically pack rows of each matrix,
                    // skipping `next` if above scan showed no constraints need it:
                    for (wrapped_idx, row_buf) in [
                        (&row_idx_local, Some(&mut preprocessed_pair.local)),
                        (&row_idx_next, Option::as_mut(&mut preprocessed_pair.next)),
                    ] {
                        if let Some(row_buf) = row_buf {
                            row_buf.clear();
                            for col in 0..preprocessed_width {
                                row_buf.push(PackedVal::<SC>::from_fn(|offset| {
                                    preprocessed_trace_on_quotient_domain
                                        .as_ref()
                                        .unwrap()
                                        .get(wrapped_idx[offset], col)
                                }));
                            }
                        }
                    }

                    for (lde, view_pair) in partitioned_main_lde_on_quotient_domain
                        .iter()
                        .zip(partitioned_main_pairs.iter_mut())
                    {
                        let width = lde.width();
                        for (wrapped_idx, row_buf) in [
                            (&row_idx_local, Some(&mut view_pair.local)),
                            (&row_idx_next, Option::as_mut(&mut view_pair.next)),
                        ] {
                            if let Some(row_buf) = row_buf {
                                row_buf.clear();
                                for col in 0..width {
                                    row_buf.push(PackedVal::<SC>::from_fn(|offset| {
                                        lde.get(wrapped_idx[offset], col)
                                    }));
                                }
                            }
                        }
                    }

                    for (lde, view_pair) in after_challenge_lde_on_quotient_domain
                        .iter()
                        .zip(after_challenge_pairs.iter_mut())
                    {
                        // Width in base field with extension field elements flattened
                        let base_width = lde.width();
                        for (wrapped_idx, row_buf) in [
                            (&row_idx_local, Some(&mut view_pair.local)),
                            (&row_idx_next, Option::as_mut(&mut view_pair.next)),
                        ] {
                            if let Some(row_buf) = row_buf {
                                row_buf.clear();
                                for col in (0..base_width).step_by(ext_degree) {
                                    row_buf.push(PackedChallenge::<SC>::from_base_fn(|i| {
                                        PackedVal::<SC>::from_fn(|offset| {
                                            lde.get(wrapped_idx[offset], col + i)
                                        })
                                    }));
                                }
                            }
                        }
                    }

                    let evaluator: ProverConstraintEvaluator<SC> = ProverConstraintEvaluator {
                        preprocessed: &preprocessed_pair,
                        partitioned_main: &partitioned_main_pairs,
                        after_challenge: &after_challenge_pairs,
                        challenges,
                        is_first_row,
                        is_last_row,
                        is_transition,
                        public_values,
                        exposed_values_after_challenge,
                    };
                    // SAFETY: `constraints.nodes` should be in topological order
                    let accumulator = unsafe {
                        evaluator.accumulate(constraints, &alpha_powers, &mut node_exprs)
                    };
                    // quotient(x) = constraints(x) / Z_H(x)
                    let quotient: PackedChallenge<SC> = accumulator * inv_zeroifier;

                    // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
                    for (idx_in_packing, ef) in packed_ef_mut.iter_mut().enumerate() {
                        *ef = SC::Challenge::from_base_fn(|coeff_idx| {
                            quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing]
                        });
                    }
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
