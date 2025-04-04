mod prove;
mod verify;

use std::{array, borrow::Borrow, cmp::max, fmt::Debug, iter, iter::zip, marker::PhantomData};

use itertools::{izip, Itertools};
use p3_air::ExtensionBuilder;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::PolynomialSpace;
use p3_field::{
    cyclic_subgroup_coset_known_order, ExtensionField, Field, FieldAlgebra, TwoAdicField,
};
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use rayon::iter::IntoParallelRefIterator;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::interaction::fri_log_up::{FriLogUpError, STARK_LU_NUM_CHALLENGES};
use crate::interaction::{LogUpSecurityParameters, PairTraceView};
use crate::{
    air_builders::symbolic::SymbolicConstraints,
    gkr,
    gkr::{Gate, GkrBatchProof},
    interaction::{
        utils::generate_betas, InteractionBuilder, RapPhaseProverData, RapPhaseSeq,
        RapPhaseSeqKind, RapPhaseVerifierData,
    },
    rap::PermutationAirBuilderWithExposedValues,
};

pub struct GkrLogUpPhase<F, EF, Challenger> {
    // FIXME: USE THIS IN POW
    log_up_params: LogUpSecurityParameters,
    _phantom: PhantomData<(F, EF, Challenger)>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GkrLogUpProvingKey;

#[derive(Clone, Serialize, Deserialize)]
pub struct GkrLogUpPartialProof<T, Witness> {
    /// The rational sumcheck proof that can be verified via [gkr::partially_verify].
    pub gkr_proof: GkrBatchProof<T>,
    /// The purported evaluations of the count MLEs at `r`, per AIR, per interaction.
    pub count_mle_claims_per_instance: Vec<Vec<T>>,
    /// The purported evaluations of the sigma MLEs at `r`, per AIR, per interaction.
    pub sigma_mle_claims_per_instance: Vec<Vec<T>>,
    pub logup_pow_witness: Witness,
}

#[derive(Error, Debug)]
pub enum GkrLogUpError<F> {
    #[error("non-zero cumulative sum")]
    NonZeroCumulativeSum,
    #[error("missing GKR proof")]
    MissingGkrProof,
    #[error("missing GKR proof")]
    MalformedGkrLogUpProof,
    #[error("missing GKR proof")]
    LagrangePolynomialCheckFailed,
    #[error("GKR error: {0}")]
    GkrError(#[from] gkr::GkrError<F>),
    #[error("invalid proof of work witness")]
    InvalidPowWitness,
}

struct GkrAuxData<T> {
    after_challenge_trace_per_air: Vec<Option<DenseMatrix<T, Vec<T>>>>,
    exposed_values_per_air: Vec<Option<Vec<T>>>,
    count_mle_claims_per_instance: Vec<Vec<T>>,
    sigma_mle_claims_per_instance: Vec<Vec<T>>,
}

impl<F, EF, Challenger> GkrLogUpPhase<F, EF, Challenger> {
    pub fn new(log_up_params: LogUpSecurityParameters) -> Self {
        Self {
            log_up_params,
            _phantom: PhantomData,
        }
    }
}

impl<F, EF, Challenger> RapPhaseSeq<F, EF, Challenger> for GkrLogUpPhase<F, EF, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    type PartialProof = GkrLogUpPartialProof<EF, F>;
    type PartialProvingKey = GkrLogUpProvingKey;
    type Error = GkrLogUpError<EF>;
    // The random evaluation point `r` from batch GKR.
    const KIND: RapPhaseSeqKind = RapPhaseSeqKind::GkrLogUp;

    fn generate_pk_per_air(
        &self,
        symbolic_constraints_per_air: &[SymbolicConstraints<F>],
        _max_constraint_degree: usize,
    ) -> Vec<Self::PartialProvingKey> {
        vec![GkrLogUpProvingKey; symbolic_constraints_per_air.len()]
    }

    fn log_up_security_params(&self) -> &LogUpSecurityParameters {
        &self.log_up_params
    }

    fn partially_prove(
        &self,
        challenger: &mut Challenger,
        constraints_per_air: &[&SymbolicConstraints<F>],
        _params_per_air: &[&Self::PartialProvingKey],
        trace_view_per_air: &[PairTraceView<'_, F>],
    ) -> Option<(Self::PartialProof, RapPhaseProverData<EF>)> {
        let all_interactions = constraints_per_air
            .iter()
            .flat_map(|c| c.interactions.clone())
            .collect_vec();
        if all_interactions.is_empty() {
            return None;
        }

        let logup_pow_witness = challenger.grind(self.log_up_params.log_up_pow_bits);

        let (alpha, beta) = self.generate_challenges(challenger);
        let beta_pows = generate_betas(beta, &all_interactions);

        // Build GKR instances.
        let gkr_instances: Vec<_> =
            Self::build_gkr_instances(trace_view_per_air, constraints_per_air, &beta_pows);

        // Construct input layers and run GKR proof.
        let input_layers: Vec<_> = gkr_instances
            .par_iter()
            .map(|gkr_instance| gkr_instance.build_gkr_input_layer(alpha))
            .collect();

        let (gkr_proof, gkr_artifact) = gkr::prove_batch(challenger, input_layers);

        // Generate after challenge trace, exposed values, and MLE claims for each instance.
        let GkrAuxData {
            after_challenge_trace_per_air,
            exposed_values_per_air,
            count_mle_claims_per_instance,
            sigma_mle_claims_per_instance,
        } = Self::generate_aux_per_air(
            challenger,
            constraints_per_air,
            trace_view_per_air,
            alpha,
            &gkr_instances,
            &gkr_artifact,
        );

        let mut challenges = vec![beta, alpha];
        challenges.extend_from_slice(&gkr_artifact.ood_point);

        Some((
            GkrLogUpPartialProof {
                gkr_proof,
                count_mle_claims_per_instance,
                sigma_mle_claims_per_instance,
                logup_pow_witness,
            },
            RapPhaseProverData {
                challenges_per_phase: vec![challenges],
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
        if domains_per_air.is_empty() {
            return vec![];
        }
        let points_per_air = domains_per_air
            .iter()
            .map(|domain| extra_opening_points(zeta, domain.size()))
            .collect_vec();
        // one phase
        vec![points_per_air]
    }

    fn partially_verify(
        &self,
        challenger: &mut Challenger,
        partial_proof: Option<&Self::PartialProof>,
        constraints_per_air: &[SymbolicConstraints<F>],
        exposed_values_per_air_per_phase: &[Vec<Vec<EF>>],
    ) -> Result<RapPhaseVerifierData<EF>, Self::Error> {
        let all_interactions = constraints_per_air
            .iter()
            .flat_map(|c| c.interactions.clone())
            .collect_vec();

        if all_interactions.is_empty() {
            // TODO[zach]: Should we check the other parameters?
            return Ok(RapPhaseVerifierData {
                challenges_per_phase: vec![],
            });
        }

        let partial_proof = partial_proof.ok_or(Self::Error::MissingGkrProof)?;
        if !challenger.check_witness(
            self.log_up_params.log_up_pow_bits,
            partial_proof.logup_pow_witness,
        ) {
            return Err(GkrLogUpError::InvalidPowWitness);
        }

        let (alpha, beta) = self.generate_challenges(challenger);

        let gkr_proof = &partial_proof.gkr_proof;

        let n_instances = constraints_per_air
            .iter()
            .filter(|c| !c.interactions.is_empty())
            .count();
        if n_instances != gkr_proof.output_claims_by_instance.len() {
            return Err(Self::Error::MalformedGkrLogUpProof);
        }
        if n_instances != gkr_proof.layer_masks_by_instance.len() {
            return Err(Self::Error::MalformedGkrLogUpProof);
        }

        let gkr_artifact =
            gkr::partially_verify_batch(vec![Gate::LogUp; n_instances], gkr_proof, challenger)
                .map_err(|e| Self::Error::GkrError(e))?;

        for (count_mle_claims, sigma_mle_claims) in izip!(
            partial_proof.count_mle_claims_per_instance.iter(),
            partial_proof.sigma_mle_claims_per_instance.iter()
        ) {
            for (count_mle_claim, sigma_mle_claim) in izip!(count_mle_claims, sigma_mle_claims) {
                challenger.observe_ext_element(*count_mle_claim);
                challenger.observe_ext_element(*sigma_mle_claim);
            }
        }

        let bus_indices_per_air = constraints_per_air
            .iter()
            .map(|c| c.interactions.iter().map(|i| i.bus_index).collect_vec())
            .collect_vec();

        self.verify_all_instances(
            &gkr_artifact,
            &bus_indices_per_air,
            exposed_values_per_air_per_phase,
            partial_proof,
            alpha,
            alpha,
        )?;

        let mut j = 0;

        for (bus_indices, exposed_values_per_phase) in zip(
            bus_indices_per_air.iter(),
            exposed_values_per_air_per_phase.iter(),
        ) {
            if bus_indices.is_empty() {
                continue;
            }

            let interaction_dims = num_interaction_dimensions(bus_indices.len());
            let padded_len = 1 << interaction_dims;

            let claims_to_verify = &gkr_artifact.claims_to_verify_by_instance[j];
            debug_assert_eq!(claims_to_verify.len(), 2);
            let numerator_claim = claims_to_verify[0];
            let denominator_claim = claims_to_verify[1];

            let count_mle_claims = &partial_proof.count_mle_claims_per_instance[j];
            let sigma_mle_claims = &partial_proof.sigma_mle_claims_per_instance[j];

            if count_mle_claims.len() != padded_len || sigma_mle_claims.len() != padded_len {
                return Err(Self::Error::MalformedGkrLogUpProof);
            }

            let n_vars = gkr_artifact.n_variables_by_instance[j];

            let ood_point = &gkr_artifact.ood_point;
            let instance_ood = &ood_point[ood_point.len() - n_vars..];
            // Constraints involving `r` are folded into the quotient polynomial, so we don't need to check it here.
            let (z, _r) = instance_ood.split_at(interaction_dims);

            if exposed_values_per_phase.len() != 1 || exposed_values_per_phase[0].len() != 1 {
                return Err(Self::Error::MalformedGkrLogUpProof);
            }
            let actual_sr = exposed_values_per_phase[0][0];

            Self::verify_instance_claims(
                z,
                numerator_claim,
                denominator_claim,
                actual_sr,
                count_mle_claims,
                sigma_mle_claims,
                bus_indices,
                alpha,
                alpha, // using alpha as gamma—check this
            )?;

            j += 1;
        }
        debug_assert_eq!(j, gkr_artifact.claims_to_verify_by_instance.len());

        let output_sum = gkr_proof
            .output_claims_by_instance
            .iter()
            .map(|c| c[0] / c[1])
            .sum::<EF>();
        if output_sum != EF::ZERO {
            return Err(Self::Error::NonZeroCumulativeSum);
        }

        let mut challenges = vec![beta, alpha];
        challenges.extend_from_slice(&gkr_artifact.ood_point);

        Ok(RapPhaseVerifierData {
            challenges_per_phase: vec![challenges],
        })
    }
}

/// Assumes TwoAdicField
pub(crate) fn cyclic_selectors_at_point<F, EF, Domain>(domain: Domain, point: EF) -> Vec<EF>
where
    F: Field, // TwoAdicField,
    EF: ExtensionField<F>,
    Domain: PolynomialSpace<Val = F>,
{
    let unshifted_point = point * domain.first_point().inverse();
    let log_n = log2_strict_usize(domain.size());
    let z_h = unshifted_point.exp_power_of_2(log_n) - EF::ONE;
    (0..=log_n)
        .map(|k| z_h / (unshifted_point.exp_power_of_2(k) - EF::ONE))
        .collect()
}

/// Assumes TwoAdicField
pub(crate) fn cyclic_selectors_on_coset<Domain, CosetDomain>(
    domain: Domain,
    coset: CosetDomain,
) -> Vec<Vec<Domain::Val>>
where
    Domain: PolynomialSpace,
    CosetDomain: PolynomialSpace<Val = Domain::Val>,
{
    let one = Domain::Val::ONE;

    let domain_shift = domain.first_point();
    let domain_log_n = log2_strict_usize(domain.size());
    let coset_shift = coset.first_point();
    let coset_log_n = log2_strict_usize(coset.size());
    let coset_gen = coset.next_point(one).unwrap(); // a bit of a hack so that we don't need TwoAdicField trait bound

    assert_eq!(domain_shift, one);
    assert_ne!(coset_shift, one);
    assert!(coset_log_n >= domain_log_n);

    let xs =
        cyclic_subgroup_coset_known_order(coset_gen, coset_shift, 1 << coset_log_n).collect_vec();

    // TODO: Optimize this.
    (0..domain_log_n)
        .map(|k| {
            xs.iter()
                .map(|&x| (x.exp_power_of_2(domain_log_n) - one) / (x.exp_power_of_2(k) - one))
                .collect_vec()
        })
        .collect_vec()
}

impl<F, EF, Challenger> GkrLogUpPhase<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    fn generate_challenges(&self, challenger: &mut Challenger) -> (EF, EF) {
        let alpha: EF = challenger.sample_ext_element();
        let beta: EF = challenger.sample_ext_element();
        (alpha, beta)
    }
}

fn extra_opening_points<F, EF>(zeta: EF, domain_size: usize) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let g = EF::from_base(F::two_adic_generator(log2_strict_usize(domain_size)));

    let mut cur = g * g;
    let mut points = vec![];
    while cur != EF::ONE {
        points.push(zeta * cur);
        cur *= cur;
    }
    points
}

fn num_interaction_dimensions(num_interactions: usize) -> usize {
    assert!(num_interactions > 0);
    max(log2_ceil_usize(num_interactions), 1)
}

#[derive(Debug, Clone)]
struct GkrLogUpPermutationCols<T> {
    cr: T,
    s: T,
    cum_sum: T,
}

impl<T> Borrow<GkrLogUpPermutationCols<T>> for [T] {
    fn borrow(&self) -> &GkrLogUpPermutationCols<T> {
        debug_assert_eq!(self.len(), 3);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<GkrLogUpPermutationCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

pub fn eval_gkr_log_up_phase<AB>(builder: &mut AB)
where
    AB: InteractionBuilder + PermutationAirBuilderWithExposedValues,
{
    let &[beta, gamma] = builder.permutation_randomness() else {
        panic!("PermutationAirBuilderWithExposedValues requires 2 randomness elements");
    };

    let all_interactions = builder.all_interactions();

    let mut gamma_pows = gamma.into().powers().take(2 * all_interactions.len());
    let beta_pows = generate_betas(beta.into(), all_interactions);

    let mut s_next = AB::ExprEF::ZERO;
    for interaction in all_interactions {
        s_next += gamma_pows.next().unwrap() * AB::ExprEF::from(interaction.count.clone());

        let b = AB::Expr::from_canonical_u32(interaction.bus_index as u32 + 1);
        let message = interaction.message.iter().chain(iter::once(&b));

        let sigma = zip(message, &beta_pows).fold(AB::ExprEF::ZERO, |acc, (field, beta)| {
            acc + beta.clone() * field.clone()
        });
        s_next += gamma_pows.next().unwrap() * sigma;
    }

    let exposed_values = builder.permutation_exposed_values();
    let cumulative_sum = exposed_values[0];

    let perm = builder.permutation();
    let (local, next) = (perm.row_slice(0), perm.row_slice(1));
    let local: &GkrLogUpPermutationCols<AB::VarEF> = (*local).borrow();
    let next: &GkrLogUpPermutationCols<AB::VarEF> = (*next).borrow();

    builder
        .when_first_row()
        .assert_zero_ext(local.cum_sum.into());
    builder.assert_eq_ext(s_next, local.s);
    builder.when_transition().assert_eq_ext(
        local.cr.into() * local.s.into() + local.cum_sum.into(),
        next.cum_sum.into(),
    );
    builder.when_last_row().assert_eq_ext(
        local.cr.into() * local.s.into() + local.cum_sum.into(),
        cumulative_sum,
    );
}

pub(crate) fn fold_multilinear_lagrange_col_constraints<Var, EF, VarEF, ExprEF, M>(
    running_sum: &mut ExprEF,
    alpha: EF,
    perm: &M,
    is_cyclic_sel: &[Var],
    r: &[VarEF],
    col: usize,
) where
    EF: Field,
    Var: Into<ExprEF> + Copy + Send + Sync + Debug,
    VarEF: FieldAlgebra<F = EF> + Into<ExprEF> + Copy + Send + Sync,
    ExprEF: FieldAlgebra<F = EF>,
    M: Matrix<VarEF>,
{
    if r.is_empty() {
        return;
    }

    let log_n = perm.height() - 1;
    assert_eq!(r.len(), log_n);

    let local = perm.row_slice(0);
    let local: &[VarEF] = (*local).borrow();

    let mut r_prod: ExprEF = ExprEF::ONE;

    for k in 0..log_n {
        let rot = perm.row_slice(log_n - k);
        let rot: &[VarEF] = (*rot).borrow();

        // if bin(i)_t = 0 and bin(j)_t = 1, and otherwise bin(i) and bin(j) are the same, then
        //    r_t * evals[i] = (1 - r_t) * evals[j]
        // it is sufficient to only enforce the constraint on rows i for which the last log_n - k bits of i are all zero
        *running_sum *= ExprEF::from_f(alpha);
        *running_sum += is_cyclic_sel[k].into()
            * ((local[col] * r[k]).into() - rot[col].into() * (ExprEF::ONE - r[k].into()));

        r_prod *= ExprEF::ONE - r[k].into();
    }
    // L_r(0) = (1 - r_0) * ... * (1 - r_{n-1})
    *running_sum *= ExprEF::from_f(alpha);
    *running_sum += is_cyclic_sel[0].into() * (local[col].into() - r_prod);
}

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_baby_bear::BabyBear;
    use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
    use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
    use p3_matrix::dense::RowMajorMatrix;
    use rand::Rng;

    use crate::{
        interaction::gkr_log_up::{
            cyclic_selectors_at_point, cyclic_selectors_on_coset, extra_opening_points,
            fold_multilinear_lagrange_col_constraints,
        },
        p3_field::extension::BinomialExtensionField,
        poly::multi::hypercube_eq_partial,
    };

    #[test]
    fn test_extra_opening_points() {
        type EF = BinomialExtensionField<BabyBear, 4>;

        let mut rng = create_seeded_rng();

        let zeta: EF = rng.gen();
        let log_domain_size = 4;
        let domain_size = 1 << log_domain_size;
        let extra_points = extra_opening_points::<BabyBear, EF>(zeta, domain_size);

        let g = EF::from_base(BabyBear::two_adic_generator(log_domain_size));

        // Need to open zeta * g, zeta * g^2, zeta * g^4, and zeta * g^8.
        // But zeta * g is already opened by the STARK, so we don't request it here.
        assert_eq!(extra_points.len(), 3);

        assert_eq!(extra_points[0], g * g * zeta);
        assert_eq!(extra_points[1], g.exp_u64(4) * zeta);
        assert_eq!(extra_points[2], g.exp_u64(8) * zeta);
    }

    #[test]
    fn test_fold_multilinear_lagrange_col_constraints() {
        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let mut rng = create_seeded_rng();

        let mut running_sum = EF::ZERO;
        let alpha = EF::TWO;

        let r: [EF; 5] = rng.gen();
        let n = 1 << r.len();
        let cr = hypercube_eq_partial(&r);

        for i in 0..n {
            let perm_window = vec![
                cr[i % n],
                cr[(i + 1) % n],
                cr[(i + 2) % n],
                cr[(i + 4) % n],
                cr[(i + 8) % n],
                cr[(i + 16) % n],
            ];
            let perm = RowMajorMatrix::new(perm_window, 1);

            // this isn't the actual selector, but it has the same zeros
            let is_cyclic_row = [
                EF::from_bool((i & 31) == 0),
                EF::from_bool((i & 15) == 0),
                EF::from_bool((i & 7) == 0),
                EF::from_bool((i & 3) == 0),
                EF::from_bool((i & 1) == 0),
            ];
            fold_multilinear_lagrange_col_constraints(
                &mut running_sum,
                alpha,
                &perm,
                &is_cyclic_row,
                &r,
                0,
            );
            assert_eq!(running_sum, EF::ZERO);
        }
    }

    #[test]
    fn test_cyclic_selectors_at_point() {
        type Val = BabyBear;
        type Challenge = BinomialExtensionField<Val, 4>;

        let mut rng = create_seeded_rng();

        let domain = TwoAdicMultiplicativeCoset {
            log_n: 3,
            shift: Val::TWO,
        };
        let shift_inv = domain.shift.inverse();

        let zeta: Challenge = rng.gen();
        let cyclic_sels = cyclic_selectors_at_point(domain, zeta);
        assert_eq!(cyclic_sels.len(), 4);

        let sels = domain.selectors_at_point(zeta);
        assert_eq!(cyclic_sels[0], sels.is_first_row);

        let z_h = (zeta * shift_inv).exp_power_of_2(domain.log_n) - Challenge::ONE;
        assert_eq!(cyclic_sels[0], z_h / ((zeta * shift_inv) - Challenge::ONE));
        assert_eq!(
            cyclic_sels[1],
            z_h / ((zeta * shift_inv).square() - Challenge::ONE)
        );
        assert_eq!(
            cyclic_sels[2],
            z_h / ((zeta * shift_inv).square().square() - Challenge::ONE)
        );
        assert_eq!(
            cyclic_sels[3],
            z_h / ((zeta * shift_inv).square().square().square() - Challenge::ONE)
        );
    }

    #[test]
    fn test_cyclic_selectors_on_coset() {
        type Val = BabyBear;

        let domain = TwoAdicMultiplicativeCoset {
            log_n: 3,
            shift: Val::ONE,
        };
        let coset = TwoAdicMultiplicativeCoset {
            log_n: 5,
            shift: Val::TWO,
        };

        let cyclic_sels = cyclic_selectors_on_coset(domain, coset);

        let sels = domain.selectors_on_coset(coset);
        assert_eq!(sels.is_first_row, cyclic_sels[0]);

        let s = coset.shift;
        let h = Val::two_adic_generator(coset.log_n);

        assert_eq!(cyclic_sels.len(), domain.log_n);

        let mut cur = s;
        for i in 0..(1 << coset.log_n) {
            // (x^8 - 1) / (x - 1)
            assert_eq!(
                cyclic_sels[0][i],
                (cur.exp_power_of_2(domain.log_n) - Val::ONE) / (cur - Val::ONE)
            );
            // (x^8 - 1) / (x^2 - 1)
            assert_eq!(
                cyclic_sels[1][i],
                (cur.exp_power_of_2(domain.log_n) - Val::ONE) / (cur.square() - Val::ONE)
            );
            // (x^8 - 1) / (x^4 - 1)
            assert_eq!(
                cyclic_sels[2][i],
                (cur.exp_power_of_2(domain.log_n) - Val::ONE) / (cur.square().square() - Val::ONE)
            );
            cur *= h;
        }
    }
}
