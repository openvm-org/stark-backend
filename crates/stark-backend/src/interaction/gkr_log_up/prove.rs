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
    poly::{
        multi::{hypercube_eq_partial, Mle},
        uni::random_linear_combination_iter,
    },
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
        let (count_data, sigma_data): (Vec<EF>, Vec<EF>) = (0..total)
            .into_par_iter()
            .map(|i| {
                let row = i / num_padded_interactions;
                let col = i % num_padded_interactions;
                if col >= interactions.len() {
                    return (EF::ZERO, EF::ZERO);
                }
                let interaction = &interactions[col];

                let evaluator = Evaluator {
                    preprocessed: preprocessed_view,
                    partitioned_main: &partitioned_main_view,
                    public_values: &trace_view.public_values,
                    height,
                    local_index: row,
                };

                let count = evaluator.eval_expr(&interaction.count);
                let count_val = EF::from_base(count);

                let b = F::from_canonical_u32(interaction.bus_index as u32 + 1);
                let sigma = interaction
                    .message
                    .iter()
                    .chain(iter::once(&SymbolicExpression::Constant(b)))
                    .zip(beta_pows)
                    .fold(EF::ZERO, |acc, (field, &beta)| {
                        acc + beta * evaluator.eval_expr(field)
                    });

                (count_val, sigma)
            })
            .unzip();

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
        constraints_per_air: &[&SymbolicConstraints<F>],
        beta_pows: &[EF],
    ) -> Vec<GkrLogUpInstance<EF>> {
        constraints_per_air
            .par_iter()
            .zip_eq(trace_view_per_air.par_iter())
            .filter_map(|(constraints, trace_view)| {
                if constraints.interactions.is_empty() {
                    None
                } else {
                    Some(GkrLogUpInstance::from_interactions(
                        trace_view,
                        &constraints.interactions,
                        beta_pows,
                    ))
                }
            })
            .collect()
    }

    pub(super) fn generate_aux_per_air(
        challenger: &mut Challenger,
        constraints_per_air: &[&SymbolicConstraints<F>],
        trace_view_per_air: &[PairTraceView<F>],
        gamma: EF,
        gkr_instances: &[GkrLogUpInstance<EF>],
        gkr_artifact: &GkrArtifact<EF>,
    ) -> GkrAuxData<EF> {
        let ood_point = &gkr_artifact.ood_point;

        let mut after_challenge_trace_per_air = Vec::with_capacity(constraints_per_air.len());
        let mut exposed_values_per_air = Vec::with_capacity(constraints_per_air.len());
        let mut count_mle_claims_per_instance = Vec::with_capacity(constraints_per_air.len());
        let mut sigma_mle_claims_per_instance = Vec::with_capacity(constraints_per_air.len());

        let mut i = 0;
        for (constraints, trace_view) in izip!(constraints_per_air, trace_view_per_air,) {
            let interactions = &constraints.interactions;

            if interactions.is_empty() {
                after_challenge_trace_per_air.push(None);
                exposed_values_per_air.push(None);
                continue;
            }

            let gkr_instance = &gkr_instances[i];
            let height = trace_view.partitioned_main[0].height();
            let log_height = log2_strict_usize(height);
            let n_vars = gkr_artifact.n_variables_by_instance[i];

            let instance_ood = &ood_point[ood_point.len() - n_vars..];
            let (_, r) = instance_ood.split_at(n_vars - log_height);

            debug_assert_eq!(r.len(), log_height);

            let (after_challenge_trace, exposed_values, count_mle_claims, sigma_mle_claims) =
                Self::generate_aux(gkr_instance, gamma, r);

            // Send MLE claims to channel.
            for (count_mle_claim, sigma_mle_claim) in izip!(&count_mle_claims, &sigma_mle_claims) {
                challenger.observe_ext_element(*count_mle_claim);
                challenger.observe_ext_element(*sigma_mle_claim);
            }

            after_challenge_trace_per_air.push(Some(after_challenge_trace));
            exposed_values_per_air.push(Some(exposed_values));

            count_mle_claims_per_instance.push(count_mle_claims);
            sigma_mle_claims_per_instance.push(sigma_mle_claims);

            i += 1;
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
        gamma: EF,
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

        let eqs_at_r = hypercube_eq_partial(r);
        let mut partial_sum = EF::ZERO;

        let mut after_challenge_trace_data = vec![];
        for (count_row, sigma_row, eq_at_r) in izip!(
            gkr_instance.counts.row_slices(),
            gkr_instance.sigmas.row_slices(),
            eqs_at_r
        ) {
            let s_at_row = random_linear_combination_iter(
                &mut itertools::interleave(count_row, sigma_row),
                gamma,
            );

            after_challenge_trace_data.push(eq_at_r);
            after_challenge_trace_data.push(s_at_row);
            after_challenge_trace_data.push(partial_sum);

            partial_sum += eq_at_r * s_at_row;
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
