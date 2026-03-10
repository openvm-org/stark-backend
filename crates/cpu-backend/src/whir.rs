//! Optimized WHIR opening prover for the CPU backend.
//!
//! Key optimizations over the shared `prove_whir_opening`:
//! - Uses allocation-free `eval_to_coeff_cpu` for MLE conversion
//! - Parallelized sumcheck inner loop and folding
//! - Fused w_evals accumulation without intermediate vector allocation
//! - Early drop of large intermediate data structures

use itertools::Itertools;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    poly_common::Squarable,
    proof::{MerkleProof, WhirProof},
    prover::{
        error::WhirProverError,
        poly::{evals_mobius_eq_hypercube, Mle},
        stacked_pcs::MerkleTree,
        ColMajorMatrix, MatrixDimensions,
    },
    FiatShamirTranscript, StarkProtocolConfig, WhirConfig,
};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    device::{build_digest_layers, eval_to_coeff_cpu, reinterpret_vec},
    two_adic::DftTwiddles,
};

/// Optimized WHIR opening prover for the CPU backend.
///
/// Same protocol as `prove_whir_opening` in stark-backend, but with:
/// - Allocation-free MLE conversion via `eval_to_coeff_cpu`
/// - Parallelized sumcheck evaluation and folding
/// - Fused w_evals accumulation (no intermediate Vec allocation)
/// - Early memory deallocation
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[instrument(name = "prove_whir_opening_cpu", skip_all)]
pub fn prove_whir_opening_cpu<SC, TS>(
    transcript: &mut TS,
    hasher: &SC::Hasher,
    l_skip: usize,
    log_blowup: usize,
    whir_params: &WhirConfig,
    committed_mats: &[(&ColMajorMatrix<SC::F>, &MerkleTree<SC::F, SC::Digest>)],
    u: &[SC::EF],
) -> Result<WhirProof<SC>, WhirProverError>
where
    SC: StarkProtocolConfig,
    SC::F: TwoAdicField + Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    TS: FiatShamirTranscript<SC>,
{
    // PoW grinding before μ batching challenge.
    let mu_pow_witness = transcript.grind(whir_params.mu_pow_bits);

    let mu: SC::EF = transcript.sample_ext();
    let total_width: usize = committed_mats.iter().map(|(mat, _)| mat.width()).sum();
    let mu_powers: Vec<SC::EF> = mu.powers().take(total_width).collect_vec();

    let height = committed_mats[0].0.height();
    debug_assert!(committed_mats.iter().all(|(mat, _)| mat.height() == height));
    let mut m = log2_strict_usize(height);

    let k_whir = whir_params.k;
    let num_whir_rounds = whir_params.num_whir_rounds();
    let num_sumcheck_rounds = whir_params.num_sumcheck_rounds();
    let twiddles = DftTwiddles::new(l_skip);

    // Phase 1: MLE conversion using allocation-free eval_to_coeff_cpu
    let mles: Vec<Vec<SC::F>> = tracing::info_span!("whir_mle_conversion").in_scope(|| {
        committed_mats
            .par_iter()
            .flat_map(|(mat, _)| {
                mat.par_columns().map(|col| {
                    let mut x = eval_to_coeff_cpu(col, &twiddles);
                    Mle::coeffs_to_evals_inplace(&mut x);
                    x
                })
            })
            .collect()
    });

    // Phase 2: Batch into f_evals, then drop mles to free ~300MB
    let mut f_evals: Vec<SC::EF> = tracing::info_span!("whir_f_evals_batch").in_scope(|| {
        (0..1usize << m)
            .into_par_iter()
            .map(|i| {
                mles.iter()
                    .zip(mu_powers.iter())
                    .fold(SC::EF::ZERO, |acc, (mle_j, mu_j)| acc + *mu_j * mle_j[i])
            })
            .collect()
    });
    drop(mles);

    let mut w_evals = evals_mobius_eq_hypercube(u);

    let mut whir_sumcheck_polys: Vec<[SC::EF; 2]> = Vec::with_capacity(num_sumcheck_rounds);
    let mut codeword_commits = vec![];
    let mut ood_values = vec![];
    let mut initial_round_opened_rows: Vec<Vec<Vec<Vec<SC::F>>>> =
        vec![vec![]; committed_mats.len()];
    let mut initial_round_merkle_proofs: Vec<Vec<MerkleProof<SC::Digest>>> =
        vec![vec![]; committed_mats.len()];
    let mut codeword_opened_values: Vec<Vec<Vec<SC::EF>>> = Vec::with_capacity(num_whir_rounds - 1);
    let mut codeword_merkle_proofs: Vec<Vec<MerkleProof<SC::Digest>>> =
        Vec::with_capacity(num_whir_rounds - 1);
    let mut folding_pow_witnesses = Vec::with_capacity(num_sumcheck_rounds);
    let mut query_phase_pow_witnesses = Vec::with_capacity(num_whir_rounds);
    let mut rs_tree: Option<MerkleTree<SC::EF, SC::Digest>> = None;
    let mut log_rs_domain_size = m + log_blowup;
    let mut final_poly = None;

    for (whir_round, round_params) in whir_params.rounds.iter().enumerate() {
        let is_last_round = whir_round == num_whir_rounds - 1;

        // Sumcheck rounds with parallelized evaluation and folding
        tracing::info_span!("whir_sumcheck", round = whir_round).in_scope(|| {
            for round in 0..k_whir {
                debug_assert_eq!(f_evals.len(), 1 << (m - round));

                let s_deg = 2;
                let hypercube_dim = m - round - 1;

                // Parallelized sumcheck evaluation
                let s_evals: Vec<SC::EF> = (1..=s_deg)
                    .map(|x_idx| {
                        let x: SC::F = SC::F::from_usize(x_idx);
                        (0..(1usize << hypercube_dim))
                            .into_par_iter()
                            .map(|y| {
                                let f_0 = f_evals[y << 1];
                                let f_1 = f_evals[(y << 1) + 1];
                                let f_x = f_0 + (f_1 - f_0) * x;
                                let w_0 = w_evals[y << 1];
                                let w_1 = w_evals[(y << 1) + 1];
                                let w_x = w_0 + (w_1 - w_0) * x;
                                f_x * w_x
                            })
                            .sum::<SC::EF>()
                    })
                    .collect();

                for &eval in &s_evals {
                    transcript.observe_ext(eval);
                }
                whir_sumcheck_polys.push(s_evals.try_into().unwrap());

                folding_pow_witnesses.push(transcript.grind(whir_params.folding_pow_bits));
                let alpha: SC::EF = transcript.sample_ext();

                // Parallelized folding
                let half = f_evals.len() / 2;
                let new_f: Vec<SC::EF> = (0..half)
                    .into_par_iter()
                    .map(|y| {
                        let eval_0 = f_evals[y << 1];
                        let eval_1 = f_evals[(y << 1) + 1];
                        eval_0 + alpha * (eval_1 - eval_0)
                    })
                    .collect();
                let new_w: Vec<SC::EF> = (0..half)
                    .into_par_iter()
                    .map(|y| {
                        let eval_0 = w_evals[y << 1];
                        let eval_1 = w_evals[(y << 1) + 1];
                        eval_0 + alpha * (eval_1 - eval_0)
                    })
                    .collect();
                f_evals = new_f;
                w_evals = new_w;
            }
        });

        // Build g_mle from folded evaluations
        let g_mle = Mle::from_evaluations(&f_evals);
        let (g_tree, z_0) = if !is_last_round {
            let g_tree =
                tracing::info_span!("whir_dft_merkle", round = whir_round).in_scope(|| {
                    let dft = Radix2DitParallel::default();
                    let mut g_coeffs = g_mle.coeffs().to_vec();
                    debug_assert_eq!(g_coeffs.len(), 1 << (m - k_whir));
                    g_coeffs.resize(1 << (log_rs_domain_size - 1), SC::EF::ZERO);
                    let g_rs = dft.dft(g_coeffs);
                    build_ef_merkle_tree_packed::<SC>(hasher, g_rs, 1 << k_whir)
                });
            let g_commit = g_tree.root()?;
            transcript.observe_commit(g_commit);
            codeword_commits.push(g_commit);

            let z_0: SC::EF = transcript.sample_ext();
            let z_0_vec: Vec<SC::EF> = z_0.exp_powers_of_2().take(m - k_whir).collect_vec();
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

        // Query phase
        let omega: SC::F = SC::F::two_adic_generator(log_rs_domain_size - k_whir);
        let num_queries = round_params.num_queries;
        let mut query_indices = Vec::with_capacity(num_queries);
        query_phase_pow_witnesses.push(transcript.grind(whir_params.query_phase_pow_bits));
        for _ in 0..num_queries {
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
            zs.push(z_i);

            let depth = log_rs_domain_size.saturating_sub(k_whir);
            if whir_round == 0 {
                #[allow(clippy::needless_range_loop)]
                for com_idx in 0..committed_mats.len() {
                    debug_assert_eq!(initial_round_merkle_proofs[com_idx].len(), query_idx);
                    let tree = &committed_mats[com_idx].1;
                    assert_eq!(tree.backing_matrix().height(), 1 << log_rs_domain_size);
                    let opened_rows = tree.get_opened_rows(index)?;
                    initial_round_opened_rows[com_idx].push(opened_rows);
                    debug_assert_eq!(tree.proof_depth(), depth);
                    let proof = tree.query_merkle_proof(index)?;
                    debug_assert_eq!(proof.len(), depth);
                    initial_round_merkle_proofs[com_idx].push(proof);
                }
            } else {
                let tree: &MerkleTree<SC::EF, SC::Digest> = rs_tree.as_ref().unwrap();
                assert_eq!(tree.backing_matrix().width(), 1);
                let opened_rows = tree
                    .get_opened_rows(index)?
                    .into_iter()
                    .flatten()
                    .collect_vec();
                codeword_opened_values[whir_round - 1].push(opened_rows);
                debug_assert_eq!(tree.proof_depth(), depth);
                let proof = tree.query_merkle_proof(index)?;
                debug_assert_eq!(proof.len(), depth);
                codeword_merkle_proofs[whir_round - 1].push(proof);
            }
        }
        rs_tree = g_tree;

        let gamma: SC::EF = transcript.sample_ext();

        if !is_last_round {
            // Fused w_evals accumulation without intermediate allocation
            tracing::info_span!("whir_w_evals_accum", round = whir_round).in_scope(|| {
                // z_0 is in extension field — must use EF arithmetic
                w_evals_accumulate_ef::<SC>(&mut w_evals, z_0.unwrap(), gamma);
                // z_i values are in base field — use base field eq + EF accumulate
                for (z_i, gamma_pow) in zs.into_iter().zip(gamma.powers().skip(2)) {
                    w_evals_accumulate_base::<SC>(&mut w_evals, z_i, gamma_pow);
                }
            });
        }

        m -= k_whir;
        log_rs_domain_size -= 1;
    }

    Ok(WhirProof::<SC> {
        mu_pow_witness,
        whir_sumcheck_polys,
        codeword_commits,
        ood_values,
        folding_pow_witnesses,
        query_phase_pow_witnesses,
        initial_round_opened_rows,
        initial_round_merkle_proofs,
        codeword_opened_values,
        codeword_merkle_proofs,
        final_poly: final_poly.ok_or(WhirProverError::FinalPolyNone)?,
    })
}

/// w_evals accumulation when z is in the extension field.
/// Computes eq evaluations in EF and accumulates into w_evals.
fn w_evals_accumulate_ef<SC: StarkProtocolConfig>(w_evals: &mut [SC::EF], z: SC::EF, gamma: SC::EF)
where
    SC::EF: ExtensionField<SC::F>,
{
    let dim = log2_strict_usize(w_evals.len());
    let z_pows: Vec<SC::EF> = z.exp_powers_of_2().take(dim).collect_vec();
    let len = w_evals.len();

    let mut eq = SC::EF::zero_vec(len);
    eq[0] = gamma;
    fill_eq_evals(&mut eq, &z_pows);
    w_evals
        .par_iter_mut()
        .zip(eq.par_iter())
        .for_each(|(w, e)| {
            *w += *e;
        });
}

/// Optimized w_evals accumulation when z is in the base field.
/// Computes eq evaluations in base field F (F×F is ~4x cheaper than EF×EF),
/// then multiplies by gamma (EF×F) and accumulates into w_evals.
fn w_evals_accumulate_base<SC: StarkProtocolConfig>(w_evals: &mut [SC::EF], z: SC::F, gamma: SC::EF)
where
    SC::F: TwoAdicField,
    SC::EF: ExtensionField<SC::F>,
{
    let dim = log2_strict_usize(w_evals.len());
    let z_pows: Vec<SC::F> = z.exp_powers_of_2().take(dim).collect_vec();
    let len = w_evals.len();

    // Compute eq evaluations entirely in base field F (cheap F×F multiplies)
    let mut eq = SC::F::zero_vec(len);
    eq[0] = SC::F::ONE;
    fill_eq_evals(&mut eq, &z_pows);
    // Accumulate: w_evals[i] += gamma * eq[i]  (EF × F, not EF × EF)
    w_evals
        .par_iter_mut()
        .zip(eq.par_iter())
        .for_each(|(w, e)| {
            *w += gamma * *e;
        });
}

fn fill_eq_evals<F: TwoAdicField>(eq: &mut [F], z_pows: &[F]) {
    for (j, &z_j) in z_pows.iter().enumerate() {
        let one_minus_z = F::ONE - z_j;
        let step = 1usize << j;
        let span = step << 1;
        let active_len = 2usize << j;
        if active_len >= 1024 {
            eq[..active_len]
                .par_chunks_exact_mut(span)
                .for_each(|chunk| {
                    let (lo_half, hi_half) = chunk.split_at_mut(step);
                    for (lo, hi) in lo_half.iter_mut().zip(hi_half.iter_mut()) {
                        let prev = *lo;
                        *hi = prev * z_j;
                        *lo = prev * one_minus_z;
                    }
                });
        } else {
            for i in (0..active_len).step_by(span) {
                for k in 0..step {
                    let prev = eq[i + k];
                    eq[i + k] = prev * one_minus_z;
                    eq[i + k + step] = prev * z_j;
                }
            }
        }
    }
}

/// Build a Merkle tree from extension field codeword values using packed SIMD hashing.
fn build_ef_merkle_tree_packed<SC>(
    hasher: &SC::Hasher,
    g_rs: Vec<SC::EF>,
    rows_per_query: usize,
) -> MerkleTree<SC::EF, SC::Digest>
where
    SC: StarkProtocolConfig,
    SC::F: TwoAdicField + Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
{
    use std::any::TypeId;

    let height = g_rs.len();
    let num_leaves = height.next_power_of_two();
    let ef_width = SC::D_EF;

    // Phase 1: Leaf hashing — packed SIMD for BabyBear, allocation-free fallback otherwise.
    let row_hashes: Vec<SC::Digest> = tracing::info_span!("merkle_tree").in_scope(|| {
        if TypeId::of::<SC::F>() == TypeId::of::<BabyBear>()
            && TypeId::of::<SC::Digest>() == TypeId::of::<[BabyBear; 8]>()
        {
            // SAFETY: BinomialExtensionField<BabyBear, 4> is #[repr(C)] with value: [BabyBear; 4],
            // so the memory layout is contiguous BabyBear values.
            let bb_vals: &[BabyBear] = unsafe {
                std::slice::from_raw_parts(g_rs.as_ptr().cast::<BabyBear>(), height * ef_width)
            };
            let bb_digests =
                crate::device::hash_rows_packed_babybear(bb_vals, ef_width, height, num_leaves);
            // SAFETY: TypeId checks guarantee SC::Digest = [BabyBear; 8].
            unsafe { reinterpret_vec(bb_digests) }
        } else {
            (0..num_leaves)
                .into_par_iter()
                .map(|r| {
                    if r < height {
                        hasher.hash_slice(g_rs[r].as_basis_coefficients_slice())
                    } else {
                        hasher.hash_slice(&vec![SC::F::ZERO; ef_width])
                    }
                })
                .collect()
        }
    });

    // Phase 2: Build Merkle digest layers using shared helper.
    let layers = build_digest_layers::<SC::F, SC::Hasher>(row_hashes, rows_per_query, hasher);

    let backing_matrix = ColMajorMatrix::new(g_rs, 1);
    // SAFETY: layers were just computed as correct Merkle hashes over backing_matrix.
    unsafe { MerkleTree::from_raw_parts(backing_matrix, layers, rows_per_query) }
}
