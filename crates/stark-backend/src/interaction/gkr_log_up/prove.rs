use std::iter;

use itertools::Itertools;
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use std::cmp::max;

use crate::{
    air_builders::symbolic::symbolic_expression::{SymbolicEvaluator, SymbolicExpression},
    gkr::{
        GkrArtifact,
        Layer::{self, LogUpGeneric},
    },
    interaction::{
        gkr_log_up::{num_interaction_dimensions, GkrAuxData, GkrLogUpPhase},
        trace::Evaluator,
        PairTraceView, SymbolicInteraction,
    },
    parizip,
    poly::multi::{hypercube_eq_over_y, Mle},
    utils::metrics_span_increment,
};

/// A struct that holds the interaction-related data for a single GKR instance:
/// - `numerators`: A matrix of the numerators used in rational sumcheck.
/// - `denominators`: A matrix of the denominators used in rational sumcheck.
pub(super) struct GkrLogUpInstance<EF> {
    numerators: DenseMatrix<EF>,
    denominators: DenseMatrix<EF>,
}

impl<EF: Field> GkrLogUpInstance<EF> {
    /// Create a new `GkrInstance` from the given interactions and their associated evaluation context.
    ///
    /// # Parameters
    /// - `trace_view`: Provides access to the evaluated expressions of the trace at each row.
    /// - `interactions`: The interactions for a given AIR.
    /// - `alpha`: The alpha challenge for LogUp random evaluation.
    /// - `beta_pows`: Precomputed beta powers, one per interaction field.
    ///
    /// # Returns
    /// A `GkrInstance` containing the padded count and sigma matrices and the corresponding bus indices.
    pub fn from_interactions<F: Field>(
        trace_view: &PairTraceView<'_, F>,
        interactions: &[SymbolicInteraction<F>],
        alpha: EF,
        beta_pows: &[EF],
    ) -> Self
    where
        EF: ExtensionField<F>,
    {
        assert!(!interactions.is_empty());

        let height = trace_view.partitioned_main[0].height();
        let num_padded_interactions = 1 << num_interaction_dimensions(interactions.len());

        let partitioned_main_view = trace_view
            .partitioned_main
            .iter()
            .map(|mat| mat.as_view())
            .collect_vec();
        let preprocessed_view = trace_view.preprocessed.map(|mat| mat.as_view());

        let total = height * num_padded_interactions;
        let mut numerator_data = EF::zero_vec(total);
        let mut denominator_data = vec![EF::ONE; total];

        numerator_data
            .par_chunks_mut(num_padded_interactions)
            .zip(denominator_data.par_chunks_mut(num_padded_interactions))
            .enumerate()
            .for_each(|(row, (numer_row, denom_row))| {
                let evaluator = Evaluator {
                    preprocessed: preprocessed_view,
                    partitioned_main: &partitioned_main_view,
                    public_values: &trace_view.public_values,
                    height,
                    local_index: row,
                };

                for (col, interaction) in interactions.iter().enumerate() {
                    numer_row[col] = EF::from_base(evaluator.eval_expr(&interaction.count));

                    let b = F::from_canonical_u32(interaction.bus_index as u32 + 1);
                    let sigma = interaction
                        .message
                        .iter()
                        .chain(iter::once(&SymbolicExpression::Constant(b)))
                        .zip(beta_pows)
                        .fold(EF::ZERO, |acc, (expr, &beta)| {
                            acc + beta * evaluator.eval_expr(expr)
                        });

                    denom_row[col] = alpha + sigma;
                }
            });

        GkrLogUpInstance {
            numerators: RowMajorMatrix::new(numerator_data, num_padded_interactions),
            denominators: RowMajorMatrix::new(denominator_data, num_padded_interactions),
        }
    }

    pub fn build_gkr_input_layer(&self) -> Layer<EF> {
        debug_assert_eq!(self.numerators.height(), self.denominators.height());
        debug_assert_eq!(self.numerators.width(), self.denominators.width());

        LogUpGeneric {
            numerators: Mle::new(self.numerators.transpose().values),
            denominators: Mle::new(self.denominators.transpose().values),
        }
    }
}

impl<F, EF, Challenger> GkrLogUpPhase<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    pub(super) fn build_gkr_instances(
        trace_view_per_air: &[PairTraceView<F>],
        interactions_per_air: &[Vec<SymbolicInteraction<F>>],
        alpha: EF,
        beta_pows: &[EF],
    ) -> Vec<GkrLogUpInstance<EF>> {
        interactions_per_air
            .par_iter()
            .zip_eq(trace_view_per_air.par_iter())
            .filter_map(|(interactions, trace_view)| {
                if interactions.is_empty() {
                    None
                } else {
                    Some(GkrLogUpInstance::from_interactions(
                        trace_view,
                        interactions,
                        alpha,
                        beta_pows,
                    ))
                }
            })
            .collect()
    }

    pub(super) fn generate_aux_per_air(
        challenger: &mut Challenger,
        interactions_per_air: &[Vec<SymbolicInteraction<F>>],
        trace_view_per_air: &[PairTraceView<F>],
        gamma: EF,
        gkr_instances: &[GkrLogUpInstance<EF>],
        gkr_artifact: &GkrArtifact<EF>,
    ) -> GkrAuxData<EF> {
        let ood_point = &gkr_artifact.ood_point;

        let max_interactions = interactions_per_air
            .iter()
            .map(|v| 1 << max(log2_ceil_usize(v.len()), 1))
            .max()
            .unwrap();
        let gamma_pows = gamma.powers().take(2 * max_interactions).collect_vec();

        let trace_view_per_air_filtered = interactions_per_air
            .iter()
            .zip(trace_view_per_air.iter())
            .filter_map(|(interactions, view)| {
                if interactions.is_empty() {
                    None
                } else {
                    Some(view)
                }
            })
            .collect_vec();

        let results: Vec<_> = parizip!(
            trace_view_per_air_filtered,
            gkr_instances,
            &gkr_artifact.n_variables_by_instance
        )
        .map(|(trace_view, gkr_instance, &n_vars)| {
            let height = trace_view.partitioned_main[0].height();
            let log_height = log2_strict_usize(height);

            let instance_ood = &ood_point[ood_point.len() - n_vars..];
            let (_, r) = instance_ood.split_at(n_vars - log_height);
            debug_assert_eq!(r.len(), log_height);

            let (after_challenge_trace, exposed_values, numer_mle_claims, denom_mle_claims) =
                Self::generate_aux(gkr_instance, &gamma_pows, r);

            (
                after_challenge_trace,
                exposed_values,
                numer_mle_claims,
                denom_mle_claims,
            )
        })
        .collect();

        let mut results_iter = results.into_iter();

        let n_airs = interactions_per_air.len();
        let mut after_challenge_trace_per_air = Vec::with_capacity(n_airs);
        let mut exposed_values_per_air = Vec::with_capacity(n_airs);
        let mut numer_mle_claims_per_instance = Vec::with_capacity(n_airs);
        let mut denom_mle_claims_per_instance = Vec::with_capacity(n_airs);

        for interactions in interactions_per_air.iter() {
            if interactions.is_empty() {
                after_challenge_trace_per_air.push(None);
                exposed_values_per_air.push(None);
            } else {
                let (after_trace, exposed, numer_mle, denom_mle) = results_iter.next().unwrap();

                for (numer_mle_claim, denom_mle_claim) in numer_mle.iter().zip(denom_mle.iter()) {
                    challenger.observe_ext_element(*numer_mle_claim);
                    challenger.observe_ext_element(*denom_mle_claim);
                }
                after_challenge_trace_per_air.push(Some(after_trace));
                exposed_values_per_air.push(Some(exposed));
                numer_mle_claims_per_instance.push(numer_mle);
                denom_mle_claims_per_instance.push(denom_mle);
            }
        }
        GkrAuxData {
            after_challenge_trace_per_air,
            exposed_values_per_air,
            numer_mle_claims_per_instance,
            denom_mle_claims_per_instance,
        }
    }

    fn generate_aux(
        gkr_instance: &GkrLogUpInstance<EF>,
        gamma_pows: &[EF],
        r: &[EF],
    ) -> (DenseMatrix<EF>, Vec<EF>, Vec<EF>, Vec<EF>) {
        let numer_mle_claims: Vec<_> = gkr_instance
            .numerators
            .transpose()
            .par_rows_mut()
            .map(|row| Mle::eval_slice(row, r))
            .collect();

        let denom_mle_claims: Vec<_> = gkr_instance
            .denominators
            .transpose()
            .par_rows_mut()
            .map(|row| Mle::eval_slice(row, r))
            .collect();

        let (after_challenge_trace, exposed_values) =
            metrics_span_increment("generate_perm_trace_time_ms", || {
                let s_at_rows: Vec<EF> = gkr_instance
                    .numerators
                    .par_row_slices()
                    .zip(gkr_instance.denominators.par_row_slices())
                    .map(|(count_row, sigma_row)| {
                        itertools::interleave(count_row, sigma_row)
                            .zip(gamma_pows)
                            .fold(EF::ZERO, |acc, (&val, &gamma_pow)| acc + gamma_pow * val)
                    })
                    .collect();

                // TODO: Precompute these per height.
                let eqs_at_r = hypercube_eq_over_y(r);

                let mut partial_sum = EF::ZERO;
                let mut after_challenge_trace_data = Vec::with_capacity(eqs_at_r.len() * 3);
                for (eq_at_r, s_at_row) in eqs_at_r.iter().zip(s_at_rows.iter()) {
                    after_challenge_trace_data.push(*eq_at_r);
                    after_challenge_trace_data.push(*s_at_row);
                    after_challenge_trace_data.push(partial_sum);

                    partial_sum += *eq_at_r * *s_at_row;
                }
                let after_challenge_trace = RowMajorMatrix::new(after_challenge_trace_data, 3);
                (after_challenge_trace, vec![partial_sum])
            });

        (
            after_challenge_trace,
            exposed_values,
            numer_mle_claims,
            denom_mle_claims,
        )
    }
}
