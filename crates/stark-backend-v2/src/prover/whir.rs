use std::{iter::once, sync::Arc};

use itertools::Itertools;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    poly_common::Squarable,
    proof::{MerkleProof, WhirProof},
    prover::{
        poly::{eval_to_coeff_rs_message, evals_eq_hypercube, evals_mobius_eq_hypercube, Mle},
        stacked_pcs::{MerkleTree, StackedPcsData},
        ColMajorMatrix, CpuBackendV2, CpuDeviceV2, MatrixDimensions, ProverBackendV2,
    },
    FiatShamirTranscript, StarkProtocolConfig, WhirConfig,
};

pub trait WhirProver<SC: StarkProtocolConfig, PB: ProverBackendV2, PD, TS> {
    /// Prove the WHIR protocol for a collection of MLE polynomials \hat{q}_j, each in n variables,
    /// at a single vector `u \in \Fext^n`.
    ///
    /// This means applying WHIR with weight polynomial
    /// `\hat{w}(Z, \vec X) = Z * mobius_eq_poly(u)(\vec X)`, where `mobius_eq_poly(u)` is the
    /// Möbius-adjusted equality polynomial for eval-to-coeff RS encoding.
    ///
    /// The matrices in `common_main_pcs_data` and `pre_cached_pcs_data_per_commit` must all have
    /// the same height.
    fn prove_whir(
        &self,
        transcript: &mut TS,
        common_main_pcs_data: PB::PcsData,
        pre_cached_pcs_data_per_commit: Vec<Arc<PB::PcsData>>,
        u_cube: &[PB::Challenge],
    ) -> WhirProof<SC>;
}

impl<SC, TS> WhirProver<SC, CpuBackendV2<SC>, CpuDeviceV2, TS> for CpuDeviceV2
where
    SC: StarkProtocolConfig,
    SC::F: TwoAdicField + Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    TS: FiatShamirTranscript<SC>,
{
    #[instrument(level = "info", skip_all)]
    fn prove_whir(
        &self,
        transcript: &mut TS,
        common_main_pcs_data: StackedPcsData<SC::F, SC::Digest>,
        pre_cached_pcs_data_per_commit: Vec<Arc<StackedPcsData<SC::F, SC::Digest>>>,
        u_cube: &[SC::EF],
    ) -> WhirProof<SC> {
        let params = self.config();
        let committed_mats = once(&common_main_pcs_data)
            .chain(pre_cached_pcs_data_per_commit.iter().map(|d| d.as_ref()))
            .map(|d| (&d.matrix, &d.tree))
            .collect_vec();
        prove_whir_opening::<SC, _>(
            transcript,
            params.l_skip,
            params.log_blowup,
            &params.whir,
            &committed_mats,
            u_cube,
        )
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prove_whir_opening<SC, TS>(
    transcript: &mut TS,
    l_skip: usize,
    log_blowup: usize,
    whir_params: &WhirConfig,
    committed_mats: &[(&ColMajorMatrix<SC::F>, &MerkleTree<SC::F, SC::Digest>)],
    u: &[SC::EF],
) -> WhirProof<SC>
where
    SC: StarkProtocolConfig,
    SC::F: TwoAdicField + Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    TS: FiatShamirTranscript<SC>,
{
    // Sample randomness for algebraic batching.
    // We batch the codewords for \hat{q}_j together _before_ applying WHIR.
    let mu = transcript.sample_ext();
    let total_width = committed_mats.iter().map(|(mat, _)| mat.width()).sum();
    let mu_powers = mu.powers().take(total_width).collect_vec();

    let height = committed_mats[0].0.height();
    debug_assert!(committed_mats.iter().all(|(mat, _)| mat.height() == height));
    let mut m = log2_strict_usize(height);

    let k_whir = whir_params.k;
    let num_whir_rounds = whir_params.num_whir_rounds();
    let num_sumcheck_rounds = whir_params.num_sumcheck_rounds();

    let mles: Vec<Vec<SC::F>> = committed_mats
        .par_iter()
        .flat_map(|(mat, _)| {
            mat.par_columns().map(|col| {
                // Convert column evaluations directly into eval-to-coeff RS coefficients, then
                // interpret them as MLE coefficients of HatF and compute HatF hypercube evals.
                let mut x = eval_to_coeff_rs_message(l_skip, col);
                Mle::coeffs_to_evals_inplace(&mut x);
                x
            })
        })
        .collect();

    // The evaluations of `\hat{f}` in the current WHIR round on the hypercube `H_m`.
    let mut f_evals: Vec<_> = (0..1 << m)
        .into_par_iter()
        .map(|i| {
            mles.iter()
                .zip(mu_powers.iter())
                .fold(SC::EF::ZERO, |acc, (mle_j, mu_j)| acc + *mu_j * mle_j[i])
        })
        .collect();

    // We assume `\hat{w}` in a WHIR round is always multilinear and maintain its
    // evaluations on `H_m`.
    let mut w_evals = evals_mobius_eq_hypercube(u);

    let mut whir_sumcheck_polys: Vec<[SC::EF; 2]> = Vec::with_capacity(num_sumcheck_rounds);
    let mut codeword_commits = vec![];
    let mut ood_values = vec![];
    // per commitment, per whir query, per column
    let mut initial_round_opened_rows: Vec<Vec<Vec<Vec<SC::F>>>> =
        vec![vec![]; committed_mats.len()];
    let mut initial_round_merkle_proofs: Vec<Vec<MerkleProof<SC::Digest>>> =
        vec![vec![]; committed_mats.len()];
    let mut codeword_opened_values: Vec<Vec<Vec<SC::EF>>> = Vec::with_capacity(num_whir_rounds - 1);
    let mut codeword_merkle_proofs: Vec<Vec<MerkleProof<SC::Digest>>> =
        Vec::with_capacity(num_whir_rounds - 1);
    let mut folding_pow_witnesses = Vec::with_capacity(num_sumcheck_rounds);
    let mut query_phase_pow_witnesses = Vec::with_capacity(num_whir_rounds);
    let mut rs_tree = None;
    let mut log_rs_domain_size = m + log_blowup;
    let mut final_poly = None;
    for (whir_round, round_params) in whir_params.rounds.iter().enumerate() {
        let is_last_round = whir_round == num_whir_rounds - 1;
        // Run k_whir rounds of sumcheck on `sum_{x in H_m} \hat{w}(\hat{f}(x), x)`
        for round in 0..k_whir {
            debug_assert_eq!(f_evals.len(), 1 << (m - round));

            // \hat{f} * eq has degree 2
            let s_deg = 2;
            let s_evals = (1..=s_deg)
                .map(|x| {
                    let x = SC::F::from_usize(x);
                    let hypercube_dim = m - round - 1;
                    (0..(1usize << hypercube_dim))
                        .map(|y| {
                            let f_0 = f_evals[y << 1];
                            let f_1 = f_evals[(y << 1) + 1];
                            let f_x = f_0 + (f_1 - f_0) * x;
                            let w_0 = w_evals[y << 1];
                            let w_1 = w_evals[(y << 1) + 1];
                            let w_x = w_0 + (w_1 - w_0) * x;
                            f_x * w_x
                        })
                        .fold(SC::EF::ZERO, |acc, x| acc + x)
                })
                .collect_vec();

            for &eval in &s_evals {
                transcript.observe_ext(eval);
            }
            whir_sumcheck_polys.push(s_evals.try_into().unwrap());

            folding_pow_witnesses.push(transcript.grind(whir_params.folding_pow_bits));
            // Folding randomness
            let alpha = transcript.sample_ext();

            // Fold the evaluations
            let half = f_evals.len() / 2;
            for y in 0..half {
                let eval_0 = f_evals[y << 1];
                let eval_1 = f_evals[(y << 1) + 1];
                // Linear interpolation at r_round
                f_evals[y] = eval_0 + alpha * (eval_1 - eval_0);

                let eval_0 = w_evals[y << 1];
                let eval_1 = w_evals[(y << 1) + 1];
                w_evals[y] = eval_0 + alpha * (eval_1 - eval_0);
            }
            f_evals.truncate(half);
            w_evals.truncate(half);
        }
        // Define g^ = f^(alpha, \cdot) and send matrix commit of RS(g^)
        // f_evals is the evaluations of f^(alpha, \cdot) on hypercube
        let g_mle = Mle::from_evaluations(&f_evals);
        let (g_tree, z_0) = if !is_last_round {
            let dft = Radix2DitParallel::default();
            let mut g_coeffs = g_mle.coeffs().to_vec();
            debug_assert_eq!(g_coeffs.len(), 1 << (m - k_whir));
            g_coeffs.resize(1 << (log_rs_domain_size - 1), SC::EF::ZERO);
            // `g: \mathcal{L}^{(2)} \to \mathbb F`
            let g_rs = dft.dft(g_coeffs);
            let g_tree = MerkleTree::new::<SC::H>(ColMajorMatrix::new(g_rs, 1), 1 << k_whir);
            let g_commit = g_tree.root();
            transcript.observe_commit(g_commit);
            codeword_commits.push(g_commit);

            let z_0 = transcript.sample_ext();
            let z_0_vec = z_0.exp_powers_of_2().take(m - k_whir).collect_vec();
            let g_opened_value = g_mle.eval_at_point(&z_0_vec);
            transcript.observe_ext(g_opened_value);
            ood_values.push(g_opened_value);

            (Some(g_tree), Some(z_0))
        } else {
            let coeffs = g_mle.into_coeffs();
            for coeff in &coeffs {
                transcript.observe_ext(*coeff);
            }
            final_poly = Some(coeffs);
            (None, None)
        };

        // omega is generator of RS domain `\mathcal{L}^{(2^k)}`
        let omega = SC::F::two_adic_generator(log_rs_domain_size - k_whir);
        let num_queries = round_params.num_queries;
        let mut query_indices = Vec::with_capacity(num_queries);
        query_phase_pow_witnesses.push(transcript.grind(whir_params.query_phase_pow_bits));
        // Sample query indices first
        for _ in 0..num_queries {
            // This is the index of the leaf in the Merkle tree
            let index = transcript.sample_bits(log_rs_domain_size - k_whir);
            query_indices.push(index as usize);
        }
        let mut zs = Vec::with_capacity(num_queries);
        if !is_last_round {
            codeword_opened_values.push(vec![]);
            codeword_merkle_proofs.push(vec![]);
        }
        for (query_idx, index) in query_indices.into_iter().enumerate() {
            let z_i = omega.exp_u64(index as u64);
            // Get merkle proofs for in-domain samples necessary to evaluate Fold(f, \vec
            // \alpha)(z_i)
            zs.push(z_i);

            let depth = log_rs_domain_size.saturating_sub(k_whir);
            // Row openings are different between first WHIR round (width > 1) and other rounds
            // (width = 1):
            // NOTE: merkle proof is deterministic from the index and merkle root, so the opened_row
            // and merkle proof are both hinted and not observed by the transcript.
            if whir_round == 0 {
                #[allow(clippy::needless_range_loop)]
                for com_idx in 0..committed_mats.len() {
                    debug_assert_eq!(initial_round_merkle_proofs[com_idx].len(), query_idx);
                    let tree = &committed_mats[com_idx].1;
                    assert_eq!(tree.backing_matrix.height(), 1 << log_rs_domain_size);
                    let opened_rows = tree.get_opened_rows(index);
                    initial_round_opened_rows[com_idx].push(opened_rows);
                    debug_assert_eq!(tree.proof_depth(), depth);
                    let proof = tree.query_merkle_proof(index);
                    debug_assert_eq!(proof.len(), depth);
                    initial_round_merkle_proofs[com_idx].push(proof);
                }
            } else {
                let tree: &MerkleTree<SC::EF, SC::Digest> = rs_tree.as_ref().unwrap();
                assert_eq!(tree.backing_matrix.width(), 1);
                let opened_rows = tree
                    .get_opened_rows(index)
                    .into_iter()
                    .flatten()
                    .collect_vec();
                codeword_opened_values[whir_round - 1].push(opened_rows);
                debug_assert_eq!(tree.proof_depth(), depth);
                let proof = tree.query_merkle_proof(index);
                debug_assert_eq!(proof.len(), depth);
                codeword_merkle_proofs[whir_round - 1].push(proof);
            }
        }
        rs_tree = g_tree;

        // We still sample on the last round to match the verifier, who uses a
        // final gamma to unify some logic. But we do not need to update
        // `w_evals`.
        let gamma = transcript.sample_ext();

        if !is_last_round {
            // Update \hat{w}
            w_evals_accumulate::<SC::EF, SC::EF>(&mut w_evals, z_0.unwrap(), gamma);
            for (z_i, gamma_pow) in zs.into_iter().zip(gamma.powers().skip(2)) {
                w_evals_accumulate::<SC::F, SC::EF>(&mut w_evals, z_i, gamma_pow);
            }
        }

        m -= k_whir;
        log_rs_domain_size -= 1;
    }

    WhirProof::<SC> {
        whir_sumcheck_polys,
        codeword_commits,
        ood_values,
        folding_pow_witnesses,
        query_phase_pow_witnesses,
        initial_round_opened_rows,
        initial_round_merkle_proofs,
        codeword_opened_values,
        codeword_merkle_proofs,
        final_poly: final_poly.unwrap(),
    }
}

/// Given hypercube evaluations `w_evals` of `\hat{w}` on `H_t`, this updates the evaluations
/// in place to be the evaluations of `\hat{w}'(x) = \hat{w}(x) + γ * eq(x, pow(z))`.
fn w_evals_accumulate<F: Field, EF: ExtensionField<F>>(w_evals: &mut [EF], z: F, gamma: EF) {
    let dim = log2_strict_usize(w_evals.len());
    let z_pows = z.exp_powers_of_2().take(dim).collect_vec();
    let evals = evals_eq_hypercube(&z_pows);
    for (w, x) in w_evals.iter_mut().zip(evals.into_iter()) {
        *w += gamma * x;
    }
}
