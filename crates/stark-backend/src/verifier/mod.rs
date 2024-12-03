use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, AbstractField};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    config::{Domain, StarkGenericConfig},
    keygen::{types::MultiStarkVerifyingKey, view::MultiStarkVerifyingKeyView},
    prover::{opener::AdjacentOpenedValues, types::Proof},
    verifier::constraints::verify_single_rap_constraints,
};

pub mod constraints;
mod error;

pub use error::*;

use crate::config::Val;

/// Verifies a partitioned proof of multi-matrix AIRs.
pub struct MultiTraceStarkVerifier<'c, SC: StarkGenericConfig> {
    config: &'c SC,
}

impl<'c, SC: StarkGenericConfig> MultiTraceStarkVerifier<'c, SC> {
    pub fn new(config: &'c SC) -> Self {
        Self { config }
    }
    /// Verify collection of InteractiveAIRs and check the permutation
    /// cumulative sum is equal to zero across all AIRs.
    #[instrument(name = "MultiTraceStarkVerifier::verify", level = "debug", skip_all)]
    pub fn verify(
        &self,
        challenger: &mut SC::Challenger,
        mvk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError> {
        let mvk = mvk.view(&proof.get_air_ids());
        let cumulative_sums = proof
            .per_air
            .iter()
            .map(|p| {
                assert!(
                    p.exposed_values_after_challenge.len() <= 1,
                    "Verifier does not support more than 1 challenge phase"
                );
                p.exposed_values_after_challenge.first().map(|values| {
                    assert_eq!(
                        values.len(),
                        1,
                        "Only exposed value should be cumulative sum"
                    );
                    values[0]
                })
            })
            .collect_vec();

        self.verify_raps(challenger, &mvk, proof)?;

        // Check cumulative sum
        let sum: SC::Challenge = cumulative_sums
            .into_iter()
            .map(|c| c.unwrap_or(SC::Challenge::ZERO))
            .sum();
        if sum != SC::Challenge::ZERO {
            return Err(VerificationError::NonZeroCumulativeSum);
        }
        Ok(())
    }

    /// Verify general RAPs without checking any relations (e.g., cumulative sum) between exposed values of different RAPs.
    ///
    /// Public values is a global list shared across all AIRs.
    ///
    /// - `num_challenges_to_sample[i]` is the number of challenges to sample in the trace challenge phase corresponding to `proof.commitments.after_challenge[i]`. This must have length equal
    /// to `proof.commitments.after_challenge`.
    #[instrument(level = "debug", skip_all)]
    pub fn verify_raps(
        &self,
        challenger: &mut SC::Challenger,
        mvk: &MultiStarkVerifyingKeyView<SC>,
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError> {
        let public_values = proof.get_public_values();
        // Challenger must observe public values
        for pis in &public_values {
            challenger.observe_slice(pis);
        }

        // TODO: valid shape check from verifying key
        for preprocessed_commit in mvk.flattened_preprocessed_commits() {
            challenger.observe(preprocessed_commit);
        }

        // Observe main trace commitments
        challenger.observe_slice(&proof.commitments.main_trace);
        challenger.observe_slice(
            &proof
                .per_air
                .iter()
                .map(|ap| Val::<SC>::from_canonical_usize(log2_strict_usize(ap.degree)))
                .collect_vec(),
        );

        let mut challenges = Vec::new();
        for (phase_idx, (&num_to_sample, commit)) in mvk
            .num_challenges_per_phase()
            .iter()
            .zip_eq(&proof.commitments.after_challenge)
            .enumerate()
        {
            // Sample challenges needed in this phase
            challenges.push(
                (0..num_to_sample)
                    .map(|_| challenger.sample_ext_element::<SC::Challenge>())
                    .collect_vec(),
            );
            // For each RAP, the exposed values in current phase
            for air_proof in &proof.per_air {
                let exposed_values = air_proof.exposed_values_after_challenge.get(phase_idx);
                if let Some(values) = exposed_values {
                    // Observe exposed values (in ext field)
                    for value in values {
                        challenger.observe_slice(value.as_base_slice());
                    }
                }
            }
            // Observe single commitment to all trace matrices in this phase
            challenger.observe(commit.clone());
        }

        // Draw `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        // Observe quotient commitments
        challenger.observe(proof.commitments.quotient.clone());

        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        let pcs = self.config.pcs();
        // Build domains
        let (domains, quotient_chunks_domains): (Vec<_>, Vec<Vec<_>>) = mvk
            .per_air
            .iter()
            .zip_eq(&proof.per_air)
            .map(|(vk, air_proof)| {
                let degree = air_proof.degree;
                let quotient_degree = vk.quotient_degree;
                let domain = pcs.natural_domain_for_degree(degree);
                let quotient_domain = domain.create_disjoint_domain(degree * quotient_degree);
                let qc_domains = quotient_domain.split_domains(quotient_degree);
                (domain, qc_domains)
            })
            .unzip();
        // Verify all opening proofs
        let opened_values = &proof.opening.values;
        let trace_domain_and_openings =
            |domain: Domain<SC>,
             zeta: SC::Challenge,
             values: &AdjacentOpenedValues<SC::Challenge>| {
                (
                    domain,
                    vec![
                        (zeta, values.local.clone()),
                        (domain.next_point(zeta).unwrap(), values.next.clone()),
                    ],
                )
            };
        // Build the opening rounds
        // 1. First the preprocessed trace openings
        // Assumption: each AIR with preprocessed trace has its own commitment and opening values
        let mut rounds: Vec<_> = mvk
            .preprocessed_commits()
            .into_iter()
            .zip_eq(&domains)
            .flat_map(|(commit, domain)| commit.map(|commit| (commit, *domain)))
            .zip_eq(&opened_values.preprocessed)
            .map(|((commit, domain), values)| {
                let domain_and_openings = trace_domain_and_openings(domain, zeta, values);
                (commit, vec![domain_and_openings])
            })
            .collect();

        // 2. Then the main trace openings

        let num_main_commits = opened_values.main.len();
        assert_eq!(num_main_commits, proof.commitments.main_trace.len());
        let mut main_commit_idx = 0;
        // All commits except the last one are cached main traces.
        izip!(&mvk.per_air, &domains).for_each(|(vk, domain)| {
            for _ in 0..vk.num_cached_mains() {
                let commit = proof.commitments.main_trace[main_commit_idx].clone();
                let value = &opened_values.main[main_commit_idx][0];
                let domains_and_openings = vec![trace_domain_and_openings(*domain, zeta, value)];
                rounds.push((commit.clone(), domains_and_openings));
                main_commit_idx += 1;
            }
        });
        // In the last commit, each matrix corresponds to an AIR with a common main trace.
        {
            let values_per_mat = &opened_values.main[main_commit_idx];
            let commit = proof.commitments.main_trace[main_commit_idx].clone();
            let domains_and_openings = mvk
                .per_air
                .iter()
                .zip_eq(&domains)
                .filter_map(|(vk, domain)| {
                    if vk.has_common_main() {
                        Some(*domain)
                    } else {
                        None
                    }
                })
                .zip_eq(values_per_mat)
                .map(|(domain, values)| trace_domain_and_openings(domain, zeta, values))
                .collect_vec();
            rounds.push((commit.clone(), domains_and_openings));
        }

        // 3. Then after_challenge trace openings, at most 1 phase for now.
        // All AIRs with interactions should an after challenge trace.
        let after_challenge_domain_per_air = mvk
            .per_air
            .iter()
            .zip_eq(&domains)
            .filter_map(|(vk, domain)| {
                if vk.has_interaction() {
                    Some(*domain)
                } else {
                    None
                }
            })
            .collect_vec();
        if after_challenge_domain_per_air.is_empty() {
            assert_eq!(proof.commitments.after_challenge.len(), 0);
            assert_eq!(opened_values.after_challenge.len(), 0);
        } else {
            let after_challenge_commit = proof.commitments.after_challenge[0].clone();
            let domains_and_openings = after_challenge_domain_per_air
                .into_iter()
                .zip_eq(&opened_values.after_challenge[0])
                .map(|(domain, values)| trace_domain_and_openings(domain, zeta, values))
                .collect_vec();
            rounds.push((after_challenge_commit, domains_and_openings));
        }

        let quotient_domains_and_openings = opened_values
            .quotient
            .iter()
            .zip_eq(&quotient_chunks_domains)
            .flat_map(|(chunk, quotient_chunks_domains_per_air)| {
                chunk
                    .iter()
                    .zip_eq(quotient_chunks_domains_per_air)
                    .map(|(values, &domain)| (domain, vec![(zeta, values.clone())]))
            })
            .collect_vec();
        rounds.push((
            proof.commitments.quotient.clone(),
            quotient_domains_and_openings,
        ));

        pcs.verify(rounds, &proof.opening.proof, challenger)
            .map_err(|e| VerificationError::InvalidOpeningArgument(format!("{:?}", e)))?;

        let mut preprocessed_idx = 0usize; // preprocessed commit idx
        let num_phases = mvk.num_phases();
        let mut after_challenge_idx = vec![0usize; num_phases];
        let mut cached_main_commit_idx = 0;
        let mut common_main_matrix_idx = 0;

        // Verify each RAP's constraints
        for (domain, qc_domains, quotient_chunks, vk, air_proof) in izip!(
            domains,
            quotient_chunks_domains,
            &opened_values.quotient,
            &mvk.per_air,
            &proof.per_air
        ) {
            let preprocessed_values = vk.preprocessed_data.as_ref().map(|_| {
                let values = &opened_values.preprocessed[preprocessed_idx];
                preprocessed_idx += 1;
                values
            });
            let mut partitioned_main_values = Vec::with_capacity(vk.num_cached_mains());
            for _ in 0..vk.num_cached_mains() {
                partitioned_main_values.push(&opened_values.main[cached_main_commit_idx][0]);
                cached_main_commit_idx += 1;
            }
            if vk.has_common_main() {
                partitioned_main_values
                    .push(&opened_values.main.last().unwrap()[common_main_matrix_idx]);
                common_main_matrix_idx += 1;
            }
            // loop through challenge phases of this single RAP
            let after_challenge_values = if vk.has_interaction() {
                (0..num_phases)
                    .map(|phase_idx| {
                        let matrix_idx = after_challenge_idx[phase_idx];
                        after_challenge_idx[phase_idx] += 1;
                        &opened_values.after_challenge[phase_idx][matrix_idx]
                    })
                    .collect_vec()
            } else {
                vec![]
            };
            verify_single_rap_constraints::<SC>(
                &vk.symbolic_constraints.constraints,
                preprocessed_values,
                partitioned_main_values,
                after_challenge_values,
                quotient_chunks,
                domain,
                &qc_domains,
                zeta,
                alpha,
                &challenges,
                &air_proof.public_values,
                &air_proof.exposed_values_after_challenge,
            )?;
        }

        Ok(())
    }
}
