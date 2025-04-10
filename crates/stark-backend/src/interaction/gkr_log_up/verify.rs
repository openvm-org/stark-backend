use std::iter::zip;

use itertools::izip;
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};

use crate::{
    gkr,
    interaction::{
        gkr_log_up::{
            num_interaction_dimensions, GkrLogUpError, GkrLogUpPartialProof, GkrLogUpPhase,
        },
        BusIndex,
    },
    poly::multi::hypercube_eq_over_y,
};

impl<F, EF, Challenger> GkrLogUpPhase<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    /// Verifies the following:
    /// * numerator_claim in terms of numer_mle_claims
    /// * denominator_claim in terms of denom_mle_claims
    /// * numer_mle_claims and denom_mle_claims in terms of actual_sr
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_instance_claims(
        z: &[EF],
        numerator_claim: EF,
        denominator_claim: EF,
        actual_sr: EF,
        numerator_mle_claims: &[EF],
        denominator_mle_claims: &[EF],
        gamma: EF,
    ) -> Result<(), GkrLogUpError<EF>> {
        // TDOO: Can cache this per height.
        let eqs_at_z = hypercube_eq_over_y(z);

        let mut expected_numerator = EF::ZERO;
        let mut expected_denominator = EF::ZERO;
        let mut expected_sr = EF::ZERO;

        let mut gamma_pow = EF::ONE;

        for (&numer_j, &denom_j, eq_at_z) in
            izip!(numerator_mle_claims, denominator_mle_claims, eqs_at_z)
        {
            expected_numerator += eq_at_z * numer_j;
            expected_denominator += eq_at_z * denom_j;

            expected_sr += gamma_pow * numer_j;
            gamma_pow *= gamma;
            expected_sr += gamma_pow * denom_j;
            gamma_pow *= gamma;
        }
        if expected_numerator != numerator_claim {
            return Err(GkrLogUpError::MalformedGkrLogUpProof);
        }
        if expected_denominator != denominator_claim {
            return Err(GkrLogUpError::MalformedGkrLogUpProof);
        }
        if expected_sr != actual_sr {
            return Err(GkrLogUpError::MalformedGkrLogUpProof);
        }
        Ok(())
    }

    /// Verify all AIR instances using the provided data, ensuring MLE claims match expectations.
    pub(super) fn verify_all_instances(
        &self,
        gkr_artifact: &gkr::GkrArtifact<EF>,
        bus_indices_per_instance: &[Vec<BusIndex>],
        exposed_values_per_air_per_phase: &[Vec<Vec<EF>>],
        partial_proof: &GkrLogUpPartialProof<EF, F>,
        gamma: EF,
    ) -> Result<(), GkrLogUpError<EF>> {
        let mut j = 0;

        for (bus_indices, exposed_values_per_phase) in zip(
            bus_indices_per_instance.iter(),
            exposed_values_per_air_per_phase.iter(),
        ) {
            if bus_indices.is_empty() {
                // No interactions means no verification needed for this AIR.
                continue;
            }

            let n_vars = gkr_artifact.n_variables_by_instance[j];
            let ood_point = &gkr_artifact.ood_point;
            let instance_ood = &ood_point[ood_point.len() - n_vars..];

            self.verify_single_instance(
                bus_indices,
                exposed_values_per_phase,
                instance_ood,
                &partial_proof.numer_mle_claims_per_instance[j],
                &partial_proof.denom_mle_claims_per_instance[j],
                &gkr_artifact.claims_to_verify_by_instance[j],
                gamma,
            )?;

            j += 1;
        }

        debug_assert_eq!(j, gkr_artifact.claims_to_verify_by_instance.len());
        Ok(())
    }

    /// Verify a single AIR instance:
    ///  - Check claim lengths
    ///  - Validate claims against the numerator/denominator from GKR
    ///  - Check after-challenge opened values
    #[allow(clippy::too_many_arguments)]
    fn verify_single_instance(
        &self,
        bus_indices: &[BusIndex],
        exposed_values_per_phase: &[Vec<EF>],
        ood_point: &[EF],
        numer_mle_claims: &[EF],
        denom_mle_claims: &[EF],
        gkr_claims_to_verify: &[EF],
        gamma: EF,
    ) -> Result<(), GkrLogUpError<EF>> {
        if exposed_values_per_phase.len() != 1 || exposed_values_per_phase[0].len() != 1 {
            return Err(GkrLogUpError::MalformedGkrLogUpProof);
        }

        let interactions_dim = num_interaction_dimensions(bus_indices.len());
        let interactions_size = 1 << interactions_dim;

        // Check MLE claim lengths match the padded length.
        if numer_mle_claims.len() != interactions_size
            || denom_mle_claims.len() != interactions_size
        {
            return Err(GkrLogUpError::MalformedGkrLogUpProof);
        }

        // `gkr_claims_to_verify`  comes from artifact, not proof, so we don't need to check it.
        debug_assert_eq!(gkr_claims_to_verify.len(), 2);

        let numerator_claim = gkr_claims_to_verify[0];
        let denominator_claim = gkr_claims_to_verify[1];

        // Constraints involving `r` are folded into the quotient polynomial, so we don't need to check it here.
        let (z, _r) = ood_point.split_at(interactions_dim);

        let actual_sr = exposed_values_per_phase[0][0];

        Self::verify_instance_claims(
            z,
            numerator_claim,
            denominator_claim,
            actual_sr,
            numer_mle_claims,
            denom_mle_claims,
            gamma,
        )?;

        Ok(())
    }
}
