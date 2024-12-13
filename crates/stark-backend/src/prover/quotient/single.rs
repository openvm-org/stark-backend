use std::{cmp::min, iter, ops::Range};

use itertools::Itertools;
use p3_commit::PolynomialSpace;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PackedValue};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use super::evaluator::{ProverConstraintEvaluator, ViewPair};
use crate::{
    air_builders::symbolic::{
        dag::{SymbolicExpressionDag, SymbolicExpressionNode},
        symbolic_variable::Entry,
    },
    config::{Domain, PackedChallenge, PackedVal, StarkGenericConfig, Val},
    interaction::{
        gkr_log_up::{cyclic_selectors_on_coset, fold_multilinear_lagrange_col_constraints},
        RapPhaseSeq, RapPhaseSeqKind,
    },
};

pub(crate) struct TraceEvals<T> {
    pub(crate) preprocessed: RowMajorMatrix<T>,
    pub(crate) partitioned_main: Vec<RowMajorMatrix<T>>,
    pub(crate) after_challenge: Vec<RowMajorMatrix<T>>,
}

// Starting reference: p3_uni_stark::prover::quotient_values
// TODO: make this into a trait that is auto-implemented so we can dynamic dispatch the trait
/// Computes evaluation of DEEP quotient polynomial on the quotient domain for a single RAP (single trace matrix).
///
/// Designed to be general enough to support RAP with multiple rounds of challenges.
#[allow(clippy::too_many_arguments)]
#[instrument(
    name = "compute single RAP quotient polynomial",
    level = "trace",
    skip_all
)]
pub(crate) fn compute_single_rap_quotient_values<'a, SC>(
    constraints: &SymbolicExpressionDag<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_evals_on_quotient_domain: TraceEvals<Val<SC>>,
    // For each challenge round, the challenges drawn
    challenges: &[Vec<PackedChallenge<SC>>],
    alpha: SC::Challenge,
    public_values: &'a [Val<SC>],
    // Values exposed to verifier after challenge round i
    exposed_values_after_challenge: &'a [&'a [PackedChallenge<SC>]],
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
{
    let is_gkr = matches!(SC::RapPhaseSeq::KIND, RapPhaseSeqKind::GkrLogUp);

    let log_trace_size = log2_strict_usize(trace_domain.size());
    let quotient_size = quotient_domain.size();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_size) - log_trace_size;
    let next_step = 1 << qdb;

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

    let gkr_folder = is_gkr.then(|| {
        let mut cyclic_sels = cyclic_selectors_on_coset(trace_domain, quotient_domain);
        for _ in quotient_size..PackedVal::<SC>::WIDTH {
            for is_pow2k_sel in cyclic_sels.iter_mut() {
                is_pow2k_sel.push(Val::<SC>::default());
            }
        }
        let next_steps = (0..log_trace_size).map(|i| 1u64 << (qdb + i)).collect_vec();
        GkrLogUpAdapterFolder::<SC> {
            after_challenge: &trace_evals_on_quotient_domain.after_challenge,
            challenges,
            cyclic_sels,
            next_steps,
            alpha,
        }
    });

    let rotation = compute_max_rotation_and_verify_bounds(
        constraints,
        &trace_evals_on_quotient_domain,
        public_values,
        challenges,
        exposed_values_after_challenge,
        ext_degree,
    );
    let needs_next = rotation > 0;

    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let wrap = |i| i % quotient_size;
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

            let row_idx_local = i_range.clone().map(wrap).collect::<Vec<_>>();
            let row_idx_next = needs_next.then(|| {
                i_range
                    .clone()
                    .map(|i| wrap(i + next_step))
                    .collect::<Vec<_>>()
            });

            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_zeroifier = *PackedVal::<SC>::from_slice(&sels.inv_zeroifier[i_range.clone()]);

            // Vertically pack rows of each matrix,
            // skipping `next` if above scan showed no constraints need it:
            let preprocessed = pack_rows::<SC>(
                &trace_evals_on_quotient_domain.preprocessed,
                &row_idx_local,
                row_idx_next.as_deref(),
            );
            let partitioned_main = trace_evals_on_quotient_domain
                .partitioned_main
                .iter()
                .map(|lde| pack_rows::<SC>(lde, &row_idx_local, row_idx_next.as_deref()))
                .collect_vec();
            let after_challenge = trace_evals_on_quotient_domain
                .after_challenge
                .iter()
                .map(|lde| pack_challenge_rows::<SC>(lde, &row_idx_local, row_idx_next.as_deref()))
                .collect_vec();

            let evaluator = ProverConstraintEvaluator::<SC> {
                preprocessed,
                partitioned_main,
                after_challenge,
                challenges,
                is_first_row,
                is_last_row,
                is_transition,
                public_values,
                exposed_values_after_challenge,
            };
            let mut accumulator = evaluator.accumulate(constraints, &alpha_powers);

            if let Some(gkr_folder) = gkr_folder.as_ref() {
                gkr_folder.fold_constraints(&mut accumulator, &i_range, row_idx_local);
            }

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient: PackedChallenge<SC> = accumulator * inv_zeroifier;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            let width = min(PackedVal::<SC>::WIDTH, quotient_size);
            unpack_challenges::<SC>(&quotient, width)
        })
        .collect()
}

struct GkrLogUpAdapterFolder<'a, SC: StarkGenericConfig> {
    pub after_challenge: &'a [RowMajorMatrix<Val<SC>>],
    pub challenges: &'a [Vec<PackedChallenge<SC>>],
    pub cyclic_sels: Vec<Vec<Val<SC>>>,
    pub next_steps: Vec<u64>,
    pub alpha: SC::Challenge,
}

impl<SC: StarkGenericConfig> GkrLogUpAdapterFolder<'_, SC> {
    fn fold_constraints(
        &self,
        accumulator: &mut PackedChallenge<SC>,
        i_range: &Range<usize>,
        row_idx_local: Vec<usize>,
    ) {
        if self.after_challenge.is_empty() {
            return;
        }

        assert_eq!(self.after_challenge.len(), 1);
        assert_eq!(self.challenges.len(), 1);

        let after_challenge = &self.after_challenge[0];
        let challenges = &self.challenges[0];
        let quotient_size = after_challenge.height();

        let is_cyclic_sel = self
            .cyclic_sels
            .iter()
            .map(|sels| *PackedVal::<SC>::from_slice(&sels[i_range.clone()]))
            .collect_vec();

        let row_idx_rots = self
            .next_steps
            .iter()
            .map(|step| {
                i_range
                    .clone()
                    .map(|i| ((i as u64 + step) % quotient_size as u64) as usize)
                    .collect_vec()
            })
            .collect_vec();
        let row_idx_rots = iter::once(row_idx_local).chain(row_idx_rots).collect_vec();
        let after_challenge_window = row_idx_rots
            .iter()
            .flat_map(|wrapped_idx| {
                (0..after_challenge.width)
                    .step_by(SC::Challenge::D)
                    .map(|col| {
                        PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                *mat_get_unchecked(after_challenge, wrapped_idx[offset], col + i)
                            })
                        })
                    })
                    .collect_vec()
            })
            .collect_vec();
        let after_challenge_window = RowMajorMatrix::new(
            after_challenge_window,
            after_challenge.width / SC::Challenge::D,
        );

        let log_trace_size = after_challenge_window.height() - 1;
        let r = &challenges[challenges.len() - log_trace_size..];

        fold_multilinear_lagrange_col_constraints(
            accumulator,
            self.alpha,
            &after_challenge_window,
            &is_cyclic_sel,
            r,
            0,
        );
    }
}

fn compute_max_rotation_and_verify_bounds<Val, Challenge>(
    constraints: &SymbolicExpressionDag<Val>,
    trace_evals_on_quotient_domain: &TraceEvals<Val>,
    public_values: &[Val],
    challenges: &[Vec<Challenge>],
    exposed_values_after_challenge: &[&[Challenge]],
    ext_degree: usize,
) -> usize {
    let mut rotation = 0;
    for node in &constraints.nodes {
        if let SymbolicExpressionNode::Variable(var) = node {
            match var.entry {
                Entry::Preprocessed { offset } => {
                    rotation = rotation.max(offset);
                    assert!(var.index < trace_evals_on_quotient_domain.preprocessed.width);
                }
                Entry::Main { part_index, offset } => {
                    rotation = rotation.max(offset);
                    assert!(
                        var.index
                            < trace_evals_on_quotient_domain.partitioned_main[part_index].width
                    );
                }
                Entry::Public => {
                    assert!(var.index < public_values.len());
                }
                Entry::Permutation { offset } => {
                    rotation = rotation.max(offset);
                    let ext_width = trace_evals_on_quotient_domain
                        .after_challenge
                        .first()
                        .expect("Challenge phase not supported")
                        .width
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
    rotation
}

fn pack_rows<SC: StarkGenericConfig>(
    matrix: &RowMajorMatrix<Val<SC>>,
    row_idx_local: &[usize],
    row_idx_next: Option<&[usize]>,
) -> ViewPair<PackedVal<SC>> {
    let [local, next] = [Some(row_idx_local), row_idx_next].map(|wrapped_idx| {
        wrapped_idx.map(|wrapped_idx| {
            (0..matrix.width)
                .map(|col| {
                    PackedVal::<SC>::from_fn(|offset| {
                        *mat_get_unchecked(matrix, wrapped_idx[offset], col)
                    })
                })
                .collect_vec()
        })
    });
    ViewPair::new(local.unwrap(), next)
}

fn pack_challenge_rows<SC: StarkGenericConfig>(
    matrix: &RowMajorMatrix<Val<SC>>,
    row_idx_local: &[usize],
    row_idx_next: Option<&[usize]>,
) -> ViewPair<PackedChallenge<SC>> {
    let [local, next] = [Some(row_idx_local), row_idx_next].map(|wrapped_idx| {
        wrapped_idx.map(|wrapped_idx| {
            (0..matrix.width)
                .step_by(SC::Challenge::D)
                .map(|col| {
                    PackedChallenge::<SC>::from_base_fn(|i| {
                        PackedVal::<SC>::from_fn(|offset| {
                            *mat_get_unchecked(matrix, wrapped_idx[offset], col + i)
                        })
                    })
                })
                .collect_vec()
        })
    });
    ViewPair::new(local.unwrap(), next)
}

/// Transposes a vector of packed values into a vector of challenges.
fn unpack_challenges<SC>(quotient_evals: &PackedChallenge<SC>, width: usize) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
{
    (0..width)
        .map(|idx_in_packing| {
            let quotient_value = (0..SC::Challenge::D)
                .map(|coeff_idx| {
                    quotient_evals.as_base_slice()[coeff_idx].as_slice()[idx_in_packing]
                })
                .collect::<Vec<_>>();
            SC::Challenge::from_base_slice(&quotient_value)
        })
        .collect()
}

fn mat_get_unchecked<T>(mat: &RowMajorMatrix<T>, r: usize, c: usize) -> &T {
    unsafe { mat.values.get_unchecked(r * mat.width + c) }
}
