use std::iter;

use itertools::{izip, Itertools};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{
    air_builders::symbolic::{
        symbolic_expression::{SymbolicEvaluator, SymbolicExpression},
        SymbolicConstraints,
    },
    gkr::{GkrArtifact, Layer, Layer::LogUpGeneric},
    interaction::{
        gkr_log_up::{num_interaction_dimensions, GkrAuxData, GkrLogUpPhase},
        trace::Evaluator,
        PairTraceView, SymbolicInteraction,
    },
    poly::multi::{hypercube_eq_over_y, Mle},
};

/// A struct that holds the interaction-related data for a single GKR instance:
/// - `counts`: A matrix of the count values for each interaction evaluated at each row.
/// - `sigmas`: A matrix of the sigma values for each interaction evaluated at each row.
/// - `bus_indices`: The bus indices for each interaction (padded as needed).
pub(super) struct GkrLogUpInstance<EF> {
    counts: DenseMatrix<EF>,
    sigmas: DenseMatrix<EF>,
}

impl<EF: Field> GkrLogUpInstance<EF> {
    /// Create a new `GkrInstance` from the given interactions and their associated evaluation context.
    ///
    /// # Parameters
    /// - `trace_view`: Provides access to the evaluated expressions of the trace at each row.
    /// - `interactions`: The interactions for a given AIR.
    /// - `beta_pows`: Precomputed beta powers, one per interaction field.
    ///
    /// # Returns
    /// A `GkrInstance` containing the padded count and sigma matrices and the corresponding bus indices.
    pub fn from_interactions<F: Field>(
        trace_view: &PairTraceView<'_, F>,
        interactions: &[SymbolicInteraction<F>],
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
        let mut count_data = vec![EF::ZERO; total];
        let mut sigma_data = vec![EF::ZERO; total];

        count_data
            .par_chunks_mut(num_padded_interactions)
            .zip(sigma_data.par_chunks_mut(num_padded_interactions))
            .enumerate()
            .for_each(|(row, (count_row, sigma_row))| {
                let evaluator = Evaluator {
                    preprocessed: preprocessed_view,
                    partitioned_main: &partitioned_main_view,
                    public_values: &trace_view.public_values,
                    height,
                    local_index: row,
                };

                for col in 0..interactions.len() {
                    let interaction = &interactions[col];

                    count_row[col] = EF::from_base(evaluator.eval_expr(&interaction.count));

                    let b = F::from_canonical_u32(interaction.bus_index as u32 + 1);
                    let sigma = interaction
                        .message
                        .iter()
                        .chain(iter::once(&SymbolicExpression::Constant(b)))
                        .zip(beta_pows)
                        .fold(EF::ZERO, |acc, (expr, &beta)| acc + beta * evaluator.eval_expr(expr));

                    sigma_row[col] = sigma;
                }
            });

        GkrLogUpInstance {
            counts: RowMajorMatrix::new(count_data, num_padded_interactions),
            sigmas: RowMajorMatrix::new(sigma_data, num_padded_interactions),
        }
    }

    pub fn build_gkr_input_layer(&self, alpha: EF) -> Layer<EF> {
        debug_assert_eq!(self.counts.height(), self.sigmas.height());
        debug_assert_eq!(self.counts.width(), self.sigmas.width());

        let width = self.counts.width();
        let height = self.counts.height();
        let total = width * height;

        let (numerators, denominators): (Vec<EF>, Vec<EF>) = (0..total)
            .into_par_iter()
            .map(|i| {
                let row = i % height;
                let col = i / height;
                let numerator = self.counts.get(row, col);
                let denominator = alpha + self.sigmas.get(row, col);
                (numerator, denominator)
            })
            .unzip();

        LogUpGeneric {
            numerators: Mle::from_vec(numerators),
            denominators: Mle::from_vec(denominators),
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
                        &interactions,
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
            .map(|v| v.len())
            .max()
            .unwrap();
        let gamma_pows = gamma.powers().take(2 * max_interactions).collect_vec();

        let trace_view_per_air_filtered =
            interactions_per_air
                .iter()
                .zip(trace_view_per_air.iter())
                .filter_map(|(interactions, view)| {
                    if interactions.is_empty() {
                        None
                    } else {
                        Some(view)
                    }
                }).collect_vec();

        let results: Vec<_> = trace_view_per_air_filtered.par_iter()
            .zip(gkr_instances.par_iter())
            .zip(gkr_artifact.n_variables_by_instance.par_iter())
            .map(|((trace_view, gkr_instance), &n_vars)| {
                let height = trace_view.partitioned_main[0].height();
                let log_height = log2_strict_usize(height);

                let instance_ood = &ood_point[ood_point.len() - n_vars..];
                let (_, r) = instance_ood.split_at(n_vars - log_height);
                debug_assert_eq!(r.len(), log_height);

                let (after_challenge_trace, exposed_values, count_mle_claims, sigma_mle_claims) =
                    Self::generate_aux(gkr_instance, &gamma_pows, r);

                (
                    after_challenge_trace,
                    exposed_values,
                    count_mle_claims,
                    sigma_mle_claims,
                )
            })
            .collect();

        let mut results_iter = results.into_iter();

        let n_airs = interactions_per_air.len();
        let mut after_challenge_trace_per_air = Vec::with_capacity(n_airs);
        let mut exposed_values_per_air = Vec::with_capacity(n_airs);
        let mut count_mle_claims_per_instance = Vec::with_capacity(n_airs);
        let mut sigma_mle_claims_per_instance = Vec::with_capacity(n_airs);

        for interactions in interactions_per_air.iter() {
            if interactions.is_empty() {
                after_challenge_trace_per_air.push(None);
                exposed_values_per_air.push(None);
            } else {
                let
                    (after_trace, exposed, count_mle, sigma_mle) = results_iter.next().unwrap();

                for (count_mle_claim, sigma_mle_claim) in count_mle.iter().zip(sigma_mle.iter()) {
                    challenger.observe_ext_element(*count_mle_claim);
                    challenger.observe_ext_element(*sigma_mle_claim);
                }
                after_challenge_trace_per_air.push(Some(after_trace));
                exposed_values_per_air.push(Some(exposed));
                count_mle_claims_per_instance.push(count_mle);
                sigma_mle_claims_per_instance.push(sigma_mle);
            }
        }
        GkrAuxData {
            after_challenge_trace_per_air,
            exposed_values_per_air,
            count_mle_claims_per_instance,
            sigma_mle_claims_per_instance,
        }
    }

    fn generate_aux(
        gkr_instance: &GkrLogUpInstance<EF>,
        gamma_pows: &[EF],
        r: &[EF],
    ) -> (DenseMatrix<EF>, Vec<EF>, Vec<EF>, Vec<EF>) {
        // For each column (corresponding to a single interaction), we compute the MLE.
        let count_mle_claims: Vec<_> = gkr_instance
            .counts
            .transpose()
            .par_row_slices()
            .map(|row| Mle::eval_slice(row, r))
            .collect();
        let sigma_mle_claims: Vec<_> = gkr_instance
            .sigmas
            .transpose()
            .par_row_slices()
            .map(|row| Mle::eval_slice(row, r))
            .collect();

        let s_at_rows: Vec<EF> = gkr_instance
            .counts
            .par_row_slices()
            .zip(gkr_instance.sigmas.par_row_slices())
            .map(|(count_row, sigma_row)| {
                itertools::interleave(count_row, sigma_row)
                    .zip(gamma_pows)
                    .fold(EF::ZERO, |acc, (&val, &gamma_pow)| acc + gamma_pow * val)
            })
            .collect();

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
        (
            after_challenge_trace,
            vec![partial_sum],
            count_mle_claims,
            sigma_mle_claims,
        )
    }
}
