use std::{iter, iter::zip, marker::PhantomData};

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::PolynomialSpace;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use serde::{Deserialize, Serialize};

use crate::{
    air_builders::symbolic::{symbolic_expression::SymbolicEvaluator, SymbolicConstraints},
    gkr,
    gkr::{GkrBatchProof, Layer, Layer::LogUpMultiplicities},
    interaction::{
        trace::Evaluator,
        utils::{generate_betas, generate_rlc_elements},
        HasInteractionChunkSize, RapPhaseProverData, RapPhaseSeq, RapPhaseSeqKind,
        RapPhaseVerifierData, SymbolicInteraction,
    },
    poly::{multi::Mle, uni::random_linear_combination},
    prover::PairTraceView,
};

struct GkrLogUpPhase<F, EF, Challenger, Pcs> {
    pcs: Pcs,
    _phantom: PhantomData<(F, EF, Challenger)>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GkrLogUpProvingKey;

#[derive(Clone, Serialize, Deserialize)]
struct GkrLogUpProof<T> {
    /// The rational sumcheck proof that can be verified via [gkr::partially_verify].
    gkr_proof: GkrBatchProof<T>,
    /// The purported evaluations of the count MLEs at `r`, per AIR, per interaction.
    count_mle_claims_per_air: Vec<Vec<T>>,
    /// The purported evaluations of the sigma MLEs at `r`, per AIR, per interaction.
    sigma_mle_claims_per_air: Vec<Vec<T>>,
}

impl HasInteractionChunkSize for GkrLogUpProvingKey {
    fn interaction_chunk_size(&self) -> usize {
        0
    }
}

impl<F, EF, Challenger, Pcs> RapPhaseSeq<F, EF, Challenger>
for GkrLogUpPhase<F, EF, Challenger, Pcs>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Pcs: p3_commit::Pcs<EF, Challenger>,
    <Pcs as p3_commit::Pcs<EF, Challenger>>::Domain: PolynomialSpace<Val = F>,
{
    type PartialProof = GkrLogUpProof<EF>;
    type ProvingKey = GkrLogUpProvingKey;
    type Error = ();
    const KIND: RapPhaseSeqKind = RapPhaseSeqKind::GkrLogUp;

    fn generate_pk_per_air(
        &self,
        symbolic_constraints_per_air: Vec<SymbolicConstraints<F>>,
    ) -> Vec<Self::ProvingKey> {
        vec![GkrLogUpProvingKey; symbolic_constraints_per_air.len()]
    }

    fn partially_prove(
        &self,
        challenger: &mut Challenger,
        _params_per_air: &[Self::ProvingKey],
        constraints_per_air: &[&SymbolicConstraints<F>],
        trace_view_per_air: &[PairTraceView<'_, F>],
    ) -> Option<(Self::PartialProof, RapPhaseProverData<EF>)> {
        let all_interactions = constraints_per_air
            .iter()
            .flat_map(|c| c.interactions.clone())
            .collect_vec();
        if all_interactions.is_empty() {
            return None;
        }

        let alpha = challenger.sample_ext_element::<EF>();
        let beta = challenger.sample_ext_element::<EF>();
        let gamma = challenger.sample_ext_element::<EF>();

        let alpha_pows = generate_rlc_elements(alpha, &all_interactions);
        let beta_pows = generate_betas(beta, &all_interactions);

        // counts_per_air_per_row[i] and sigmas_per_air_per_row[i] are matrices with width equal to
        // the # interactions in the ith AIR. These matrices are *not* part of any trace and are
        // *not* committed to.
        let (counts_per_air, sigmas_per_air, bus_indices_per_air): (Vec<_>, Vec<_>, Vec<_>) =
            constraints_per_air
                .iter()
                .zip(trace_view_per_air)
                .map(|(constraints, trace_view)| {
                    let (counts_per_row, sigmas_per_row) =
                        Self::make_intxn_cols(&beta_pows, &constraints.interactions, trace_view);
                    let bus_indices = constraints
                        .interactions
                        .iter()
                        .map(|interaction| interaction.bus_index)
                        .collect_vec();
                    (counts_per_row, sigmas_per_row, bus_indices)
                })
                .multiunzip();

        // Make the GKR input layers. Note that AIRs with no interactions are filtered out.
        let input_layers: Vec<_> = izip!(&counts_per_air, &sigmas_per_air, bus_indices_per_air)
            .filter_map(|(counts_per_row, sigmas_per_row, bus_indices)| {
                Self::make_gkr_input_layer(
                    &alpha_pows,
                    counts_per_row.as_view(),
                    sigmas_per_row.as_view(),
                    &bus_indices,
                )
            })
            .collect();

        // Produce the GKR proof.
        let (gkr_proof, gkr_artifact) = gkr::prove_batch(challenger, input_layers);

        // The `ood_point` is shared among the whole batch, so some instances sit out for the
        // initial rounds (corresponding to the initial coordinates).
        let ood_point = gkr_artifact.ood_point;

        let mut i = 0;
        let mut after_challenge_trace_per_air = vec![];
        let mut exposed_values_per_air = vec![];
        let mut count_mle_claims_per_air = vec![];
        let mut sigma_mle_claims_per_air = vec![];

        for (constraints, trace_view, counts, sigmas) in izip!(
            constraints_per_air,
            trace_view_per_air,
            counts_per_air,
            sigmas_per_air
        ) {
            let interactions = &constraints.interactions;

            if interactions.is_empty() {
                after_challenge_trace_per_air.push(None);
                exposed_values_per_air.push(None);
            } else {
                // For each column (corresponding to a single interaction), we compute the MLE.
                let count_mles = counts
                    .transpose()
                    .rows()
                    .map(|row| Mle::new(row.collect()))
                    .collect_vec();
                let sigma_mles = sigmas
                    .transpose()
                    .rows()
                    .map(|row| Mle::new(row.collect()))
                    .collect_vec();

                let height = trace_view.height();
                let log_height = log2_strict_usize(height);
                let n_vars = gkr_artifact.n_variables_by_instance[i];

                debug_assert_eq!(n_vars, log_height + log2_ceil_usize(interactions.len()));
                let instance_ood = &ood_point[ood_point.len() - n_vars..];
                let (r, z) = instance_ood.split_at(log_height);

                debug_assert_eq!(r.len(), log_height);
                debug_assert_eq!(z.len(), log2_ceil_usize(interactions.len()));

                let count_mle_claims = count_mles.iter().map(|mle| mle.eval(r)).collect_vec();
                let sigma_mle_claims = sigma_mles.iter().map(|mle| mle.eval(r)).collect_vec();

                // Send MLE claims to channel.
                for (count_mle_claim, sigma_mle_claim) in
                    izip!(&count_mle_claims, &sigma_mle_claims)
                {
                    challenger.observe_ext_element(*count_mle_claim);
                    challenger.observe_ext_element(*sigma_mle_claim);
                }

                let eqs_at_r = eqs_at_r(r);
                let mut partial_sum = EF::ZERO;
                let mut partial_sums = vec![];

                let mut after_challenge_trace = vec![];
                for (k, (count_row, sigma_row)) in zip(counts.rows(), sigmas.rows()).enumerate() {
                    let s_eval =
                        random_linear_combination(&count_row.chain(sigma_row).collect_vec(), gamma);

                    partial_sum += eqs_at_r[k] * s_eval;
                    partial_sums.push(partial_sum);

                    after_challenge_trace.push(eqs_at_r[k]);
                    after_challenge_trace.push(partial_sum);
                }
                after_challenge_trace_per_air
                    .push(Some(RowMajorMatrix::new(after_challenge_trace, 2)));
                exposed_values_per_air.push(Some(vec![partial_sum]));
                count_mle_claims_per_air.push(count_mle_claims);
                sigma_mle_claims_per_air.push(sigma_mle_claims);

                i += 1;
            }
        }

        Some((
            GkrLogUpProof {
                gkr_proof,
                count_mle_claims_per_air,
                sigma_mle_claims_per_air,
            },
            RapPhaseProverData {
                challenges: vec![alpha, beta, gamma],
                after_challenge_trace_per_air,
                exposed_values_per_air,
            },
        ))
    }

    fn extra_opening_points<Domain>(
        &self,
        zeta: EF,
        domains_per_air: &[Domain],
    ) -> Vec<Vec<Vec<EF>>>
    where
        Domain: PolynomialSpace<Val = F>,
    {
        let mut points_per_air = vec![];
        for domain in domains_per_air {
            let g = EF::from_base(F::two_adic_generator(domain.size()));

            // zeta and zeta * g are already included; start at zeta * g^2.
            let mut cur = g * g;
            let mut points = vec![];
            while cur != zeta {
                points.push(cur);
                cur *= g;
            }
            points_per_air.push(points);
        }
        // one phase
        vec![points_per_air]
    }

    fn partially_verify<Commitment: Clone>(
        &self,
        challenger: &mut Challenger,
        partial_proof: Option<&Self::PartialProof>,
        exposed_values_per_air_per_phase: &[Vec<Vec<EF>>],
        commitments_per_phase: &[Commitment],
        after_challenge_opened_values: &[Vec<Vec<Vec<EF>>>],
    ) -> (RapPhaseVerifierData<EF>, Result<(), Self::Error>)
    where
        Challenger: CanObserve<Commitment>,
    {
        todo!()
    }
}

impl<F, EF, Challenger, Pcs> GkrLogUpPhase<F, EF, Challenger, Pcs>
where
    F: Field,
    EF: ExtensionField<F>,
    Pcs: p3_commit::Pcs<EF, Challenger>,
    <Pcs as p3_commit::Pcs<EF, Challenger>>::Domain: PolynomialSpace<Val = F>,
{
    /// Computes the input layer for the rational sum circuit.
    ///
    /// Returns `None` if there are no interactions.
    fn make_gkr_input_layer(
        alpha_pows: &[EF],
        counts: RowMajorMatrixView<EF>,
        sigmas: RowMajorMatrixView<EF>,
        bus_indices: &[usize],
    ) -> Option<Layer<EF>> {
        let pad_len = bus_indices.len().next_power_of_two() - bus_indices.len();

        let mut numerators = vec![];
        let mut denominators = vec![];
        for (counts, sigmas) in zip(counts.rows(), sigmas.rows()) {
            numerators.extend(counts);
            numerators.extend(iter::repeat_n(EF::ZERO, pad_len));

            denominators
                .extend(zip(sigmas, bus_indices).map(|(sigma, b_i)| alpha_pows[*b_i] + sigma));
            denominators.extend(iter::repeat_n(EF::ONE, pad_len));
        }
        debug_assert_eq!(numerators.len(), denominators.len());

        Some(LogUpMultiplicities {
            numerators: Mle::new(numerators),
            denominators: Mle::new(denominators),
        })
    }

    /// Computes (virtual) columns for the count and hashed fields.
    ///
    /// Returns (counts, sigmas), where `counts[j][i]` is the count of the ith interaction evaluated
    /// on the jth row, and similarly for `sigmas`.
    fn make_intxn_cols(
        beta_pows: &[EF],
        interactions: &[SymbolicInteraction<F>],
        trace_view: &PairTraceView<'_, F>,
    ) -> (DenseMatrix<EF>, DenseMatrix<EF>) {
        let height = trace_view.height();

        let mut counts = vec![];
        let mut sigmas = vec![];

        for n in 0..height {
            let evaluator = Evaluator {
                preprocessed: trace_view.preprocessed,
                partitioned_main: trace_view.partitioned_main,
                public_values: trace_view.public_values,
                height,
                local_index: n,
            };
            for interaction in interactions {
                let count = evaluator.eval_expr(&interaction.count);
                counts.push(EF::from_base(count));

                let sigma = zip(&interaction.fields, beta_pows)
                    .fold(EF::ZERO, |acc, (expr, &beta)| {
                        acc + beta * evaluator.eval_expr(expr)
                    });
                sigmas.push(sigma);
            }
        }
        (
            RowMajorMatrix::new(counts, interactions.len()),
            RowMajorMatrix::new(sigmas, interactions.len()),
        )
    }
}

/// Computes eq_n(r, x) for fixed r and each x in {0, 1}^n, where
///          eq_n(r, x) = sum_i r_i x_i + (1 - r_i)(1 - x_i).
fn eqs_at_r<F: Field>(r: &[F]) -> Vec<F> {
    let height = 1 << r.len();

    let mut eqs_at_r = vec![];

    // This runs in time O(n 2^n) but can be optimized to O(2^n).
    for k in 0..height {
        let mut eq_eval = F::ONE;
        let mut k_left = k;
        for &r_i in r {
            let k_i = F::from_canonical_usize(k_left % 2);
            eq_eval *= r_i * k_i + (F::ONE - r_i) * (F::ONE - k_i);
            k_left /= 2;
        }
        eqs_at_r.push(eq_eval);
    }
    eqs_at_r
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_field::AbstractField;
    use rand::Rng;

    use crate::interaction::gkr_log_up::eqs_at_r;

    type F = BabyBear;

    #[test]
    fn test_eqs_at_r_binary() {
        // Try on a hypercube point.
        let r = vec![F::ZERO, F::ONE, F::ZERO, F::ZERO];
        let eqs = eqs_at_r(&r);

        let mut expected = vec![F::ZERO; 16];
        expected[2] = F::ONE;
        assert_eq!(eqs, expected);
    }

    #[test]
    fn test_eqs_at_r_dim_2() {
        let mut rng = create_seeded_rng();

        let a = rng.gen::<F>();
        let b = rng.gen::<F>();
        let c = rng.gen::<F>();
        let r = vec![a, b, c];

        let eqs = eqs_at_r(&r);

        let expected = vec![
            (F::ONE - a) * (F::ONE - b) * (F::ONE - c),
            a * (F::ONE - b) * (F::ONE - c),
            (F::ONE - a) * b * (F::ONE - c),
            a * b * (F::ONE - c),
            (F::ONE - a) * (F::ONE - b) * c,
            a * (F::ONE - b) * c,
            (F::ONE - a) * b * c,
            a * b * c,
        ];
        assert_eq!(eqs, expected);
    }
}