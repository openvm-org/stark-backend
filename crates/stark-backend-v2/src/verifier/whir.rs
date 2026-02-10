use core::iter::zip;

use itertools::{izip, Itertools};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use thiserror::Error;
use tracing::instrument;

use crate::{
    poly_common::{
        eval_eq_mle, eval_mle_evals_at_point, eval_mobius_eq_mle, horner_eval,
        interpolate_quadratic_at_012, Squarable,
    },
    poseidon2::sponge::{poseidon2_compress, poseidon2_hash_slice, poseidon2_tree_compress},
    proof::WhirProof,
    baby_bear_poseidon2::{BabyBearPoseidon2ConfigV2, Digest, EF, F},
    FiatShamirTranscript, SystemParams,
};

#[inline]
fn ensure(cond: bool, err: VerifyWhirError) -> Result<(), VerifyWhirError> {
    if cond {
        Ok(())
    } else {
        Err(err)
    }
}

/// Verify a WHIR proof.
///
/// Assumes that all inputs have already been checked to have the correct sizes.
#[instrument(level = "debug", skip_all)]
pub fn verify_whir<TS: FiatShamirTranscript<BabyBearPoseidon2ConfigV2>>(
    transcript: &mut TS,
    params: &SystemParams,
    whir_proof: &WhirProof,
    stacking_openings: &[Vec<EF>],
    commitments: &[Digest],
    u: &[EF],
) -> Result<(), VerifyWhirError> {
    let widths = stacking_openings
        .iter()
        .map(|v| v.len())
        .collect::<Vec<_>>();

    let mu = transcript.sample_ext();

    let WhirProof {
        whir_sumcheck_polys,
        codeword_commits,
        ood_values,
        initial_round_opened_rows,
        initial_round_merkle_proofs,
        codeword_opened_values,
        codeword_merkle_proofs,
        folding_pow_witnesses,
        query_phase_pow_witnesses,
        final_poly,
    } = whir_proof;

    let m = params.l_skip + params.n_stack;
    let k_whir = params.k_whir();
    debug_assert_eq!((m - params.log_final_poly_len()) % k_whir, 0);
    let num_whir_rounds = params.num_whir_rounds();
    let mut log_rs_domain_size = m + params.log_blowup;
    debug_assert!(params.num_whir_sumcheck_rounds() <= m);
    debug_assert_eq!(
        folding_pow_witnesses.len(),
        params.num_whir_sumcheck_rounds()
    );

    let mut sumcheck_poly_iter = whir_sumcheck_polys.iter();
    let mut folding_pow_iter = folding_pow_witnesses.iter();
    let mu_pows: Vec<_> = mu.powers().take(widths.iter().sum::<usize>()).collect();
    let mut claim = stacking_openings
        .iter()
        .flatten()
        .zip(mu_pows.iter())
        .fold(EF::ZERO, |acc, (&opening, &mu_pow)| acc + mu_pow * opening);

    let mut gammas = Vec::with_capacity(num_whir_rounds);
    let mut zs = Vec::with_capacity(num_whir_rounds);
    let mut z0s = Vec::with_capacity(num_whir_rounds);
    let mut alphas = Vec::with_capacity(m);

    debug_assert_eq!(query_phase_pow_witnesses.len(), num_whir_rounds);
    for (whir_round, (query_phase_pow_witness, round_params)) in
        zip(query_phase_pow_witnesses, &params.whir.rounds).enumerate()
    {
        // A WHIR round consists of the following steps:
        // 1) Run k rounds of sumcheck to obtain polynomial f'.
        // 2) On non-final rounds, observe commitment f' on shifted domain.
        // 3) On non-final rounds, sample OOD point z0 and observe claim y0 =?= f'(z0).
        // 4) Sample in-domain queries z_i and compute f'(z_i) from openings. On the first round,
        //    the codeword is not committed directly; instead it is derived from the stacking
        //    commitments. In all other rounds, the previous codeword is committed directly.
        // 5) On non-final rounds, sample batching parameter gamma to define next codeword and
        //    derive new WHIR constraint target (`claim`).

        let is_initial_round = whir_round == 0;
        let is_final_round = whir_round == num_whir_rounds - 1;

        let mut alphas_round = Vec::with_capacity(k_whir);

        for _ in 0..k_whir {
            if let Some(evals) = sumcheck_poly_iter.next() {
                let &[ev1, ev2] = evals;

                transcript.observe_ext(ev1);
                transcript.observe_ext(ev2);

                let pow_witness = *folding_pow_iter.next().unwrap();
                if !transcript.check_witness(params.whir.folding_pow_bits, pow_witness) {
                    return Err(VerifyWhirError::FoldingPoWInvalid);
                }
                let alpha = transcript.sample_ext();
                alphas_round.push(alpha);

                let ev0 = claim - ev1;
                claim = interpolate_quadratic_at_012(&[ev0, ev1, ev2], alpha);
            }
        }

        let y0 = if is_final_round {
            // Observe the final polynomial before the queries on the final
            // round.
            for coeff in final_poly {
                transcript.observe_ext(*coeff);
            }
            None
        } else {
            let commit = codeword_commits[whir_round];
            transcript.observe_commit(commit);

            let z0 = transcript.sample_ext();
            z0s.push(z0);

            let y0 = ood_values[whir_round];
            transcript.observe_ext(y0);
            Some(y0)
        };

        if !transcript.check_witness(params.whir.query_phase_pow_bits, *query_phase_pow_witness) {
            return Err(VerifyWhirError::QueryPhasePoWInvalid);
        }

        let num_queries = round_params.num_queries;
        let query_indices =
            (0..num_queries).map(|_| transcript.sample_bits(log_rs_domain_size - k_whir));

        let mut zs_round = Vec::with_capacity(num_queries);
        let mut ys_round = Vec::with_capacity(num_queries);

        let omega = F::two_adic_generator(log_rs_domain_size);
        for (query_idx, index) in query_indices.into_iter().enumerate() {
            let zi_root = omega.exp_u64(index);
            let zi = zi_root.exp_power_of_2(k_whir);

            let yi = if is_initial_round {
                let mut codeword_vals = vec![EF::ZERO; 1 << k_whir];
                let mut mu_pow_iter = mu_pows.iter();
                for (&commit, &width, opened_rows_per_query, merkle_proofs) in izip!(
                    commitments,
                    &widths,
                    initial_round_opened_rows,
                    initial_round_merkle_proofs
                ) {
                    let opened_rows = &opened_rows_per_query[query_idx];
                    let leaf_hashes = opened_rows
                        .iter()
                        .map(|opened_row| poseidon2_hash_slice(opened_row))
                        .collect_vec();
                    let query_digest = poseidon2_tree_compress(leaf_hashes);
                    let merkle_proof = &merkle_proofs[query_idx];
                    merkle_verify(commit, index as u32, query_digest, merkle_proof)?;

                    for c in 0..width {
                        let mu_pow = mu_pow_iter.next().unwrap(); // ok; mu_pows has total_width length
                        for j in 0..(1 << k_whir) {
                            codeword_vals[j] += *mu_pow * opened_rows[j][c];
                        }
                    }
                }
                binary_k_fold(codeword_vals, &alphas_round, zi_root)
            } else {
                let opened_values = codeword_opened_values[whir_round - 1][query_idx].clone();
                let merkle_proof = &codeword_merkle_proofs[whir_round - 1][query_idx];
                let leaf_hashes = opened_values
                    .iter()
                    .map(|opened_value| {
                        poseidon2_hash_slice(opened_value.as_basis_coefficients_slice())
                    })
                    .collect_vec();
                let query_digest = poseidon2_tree_compress(leaf_hashes);
                merkle_verify(
                    codeword_commits[whir_round - 1],
                    index as u32,
                    query_digest,
                    merkle_proof,
                )?;
                binary_k_fold(opened_values, &alphas_round, zi_root)
            };
            zs_round.push(zi);
            ys_round.push(yi);
        }
        // We sample `gamma` even in the final round. There are no observations
        // after this challenge and strictly serves to unify the verifier logic.
        // Rather than checking that `final_poly(zi) = yi` for all `i` in the
        // last round, we accumulate them into `claim`. The final WHIR check
        // automatically performs this check for us (now with high probability).
        let gamma = transcript.sample_ext();
        if let Some(y0) = y0 {
            claim += y0 * gamma;
        }
        for (yi, gamma_pow) in ys_round.iter().zip(gamma.powers().skip(2)) {
            claim += *yi * gamma_pow;
        }
        gammas.push(gamma);
        zs.push(zs_round);
        alphas.extend(alphas_round);

        log_rs_domain_size -= 1;
    }
    debug_assert!(sumcheck_poly_iter.next().is_none());

    ensure(
        final_poly.len() == 1 << params.log_final_poly_len(),
        VerifyWhirError::FinalPolyDegree,
    )?;

    debug_assert_eq!(alphas.len(), k_whir * num_whir_rounds);
    debug_assert_eq!(z0s.len(), num_whir_rounds - 1);
    debug_assert_eq!(zs.len(), num_whir_rounds);
    debug_assert_eq!(gammas.len(), num_whir_rounds);

    // Here we perform the final WHIR check, which requires us to compute
    //
    //  sum_{b in H_{m-t}} f(b) (mobius_eq(u, alpha || b) +
    //                           sum_i sum_j gamma_{i,j} eq(pow(z_i) alpha[ki..] || b)),
    //
    // where || denotes concatenation.
    //
    // If we let u' = u[..t] and u'' = u[t..], then by factoring we can rewrite the term
    // ```text
    // sum_{b in H_{m-t}} f(b) mobius_eq(u, alpha || b) = mobius_eq(u', alpha) *
    //                                                    sum_{b in H_{m-t}} f(b) mobius_eq(u'',b)
    // ```
    //
    // For multilinear f with coefficient table `c[S]`, we have the identity:
    //   sum_{b} f(b) mobius_eq(u'', b) = sum_{S} c[S] eq(u'', S),
    // i.e. it is the MLE of the table `c` evaluated at `u''`.
    //
    // Similar algebra allows us to control the terms with eq(pow(z_i)). Note that here we actually
    // end up with f(pow(z_i^{2^p})) for some power p, which is a univariate evaluation.
    let t = k_whir * num_whir_rounds;
    let prefix = eval_mobius_eq_mle(&u[..t], &alphas[..t]);
    let suffix_sum = eval_mle_evals_at_point(&mut final_poly.clone(), &u[t..]);
    let mut acc = prefix * suffix_sum;
    let mut j = k_whir;
    for i in 0..num_whir_rounds {
        let zis = &zs[i];
        let gamma = gammas[i];
        let alpha_slc = &alphas[j..t];
        let slc_len = (t - j) + 1;

        if i != num_whir_rounds - 1 {
            let z0_pow = z0s[i].exp_powers_of_2().take(slc_len).collect_vec();
            let (z0_pow_max, z0_pow_left) = z0_pow.split_last().unwrap();
            acc += gamma
                * eval_eq_mle(alpha_slc, z0_pow_left)
                * horner_eval::<EF, EF, EF>(final_poly, *z0_pow_max);
        }

        debug_assert_eq!(zis.len(), params.whir.rounds[i].num_queries);
        for (zi, gamma_pow) in zip(zis, gamma.powers().skip(2)) {
            let zi_pow = zi.exp_powers_of_2().take(slc_len).collect_vec();
            let (zi_pow_max, zi_pow_left) = zi_pow.split_last().unwrap();
            acc += gamma_pow
                * eval_eq_mle(alpha_slc, zi_pow_left)
                * horner_eval::<EF, F, EF>(final_poly, *zi_pow_max);
        }
        j += k_whir;
    }
    ensure(acc == claim, VerifyWhirError::FinalPolyConstraint)
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum VerifyWhirError {
    #[error("final polynomial has wrong degree")]
    FinalPolyDegree,
    #[error("folding proof-of-work witness check failed")]
    FoldingPoWInvalid,
    #[error("query phase proof-of-work witness check failed")]
    QueryPhasePoWInvalid,
    #[error("final polynomial doesn't explain queries")]
    FinalPolyQueryMismatch,
    #[error("final poly is not in the final constrained RS code")]
    FinalPolyConstraint,
    #[error("merkle verification failed")]
    MerkleVerify,
}

/// Evaluates the k-fold binary fold of `f` at `x^{2^k}` given its evaluations
/// `values` on the coset `H = {x, ωx, …, ω^{2^k-1}x}` and fold points `alphas`.
///
/// Let `g₀ = f`. For `i >= 1` define
///
///   gᵢ(Y) = fold(g_{i-1}; α_{i-1})(Y),
///
/// where
///
///   fold(h; α)(X²) = h(X) + (α - X) * (h(X) - h(-X)) / (2X).
///
/// If `values = [f(x), f(ωx), …, f(ω^{2^k-1}x)]`, then
/// `binary_k_fold(values, alphas, x)` returns `g_k(x^{2^k})`.
pub fn binary_k_fold(mut values: Vec<EF>, alphas: &[EF], x: F) -> EF {
    let n = values.len();
    let k = alphas.len();
    debug_assert_eq!(n, 1 << k);

    let omega_k = F::two_adic_generator(k);
    let omega_k_inv = omega_k.inverse();

    let tw = omega_k.powers().take(1 << (k - 1)).collect_vec();
    let inv_tw = omega_k_inv.powers().take(1 << (k - 1)).collect_vec();

    for (j, (&alpha, x_pow, x_inv_pow)) in izip!(
        alphas.iter(),
        x.exp_powers_of_2(),
        x.inverse().exp_powers_of_2()
    )
    .enumerate()
    {
        let m = n >> (j + 1);
        let (lo, hi) = values.split_at_mut(m);

        for i in 0..m {
            let t = tw[i << j] * x_pow;
            let t_inv = inv_tw[i << j] * x_inv_pow;
            lo[i] += (alpha - t) * (lo[i] - hi[i]) * t_inv.halve();
        }
    }
    values[0]
}

pub fn merkle_verify(
    root: Digest,
    mut idx: u32,
    leaf_hash: Digest,
    merkle_proof: &[Digest],
) -> Result<(), VerifyWhirError> {
    let mut cur = leaf_hash;
    for &sibling in merkle_proof {
        cur = if idx & 1 == 0 {
            poseidon2_compress(cur, sibling)
        } else {
            poseidon2_compress(sibling, cur)
        };
        idx >>= 1;
    }
    if root != cur {
        Err(VerifyWhirError::MerkleVerify)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use openvm_stark_backend::prover::MatrixDimensions;
    use openvm_stark_sdk::config::{
        log_up_params::log_up_security_params_baby_bear_100_bits, setup_tracing_with_log_level,
    };
    use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use test_case::test_case;
    use tracing::Level;

    use super::*;
    use crate::{
        poly_common::Squarable,
        poseidon2::sponge::{DuplexSponge, DuplexSpongeRecorder, TranscriptHistory},
        prover::{
            poly::Ple, stacked_pcs::stacked_commit, whir::prove_whir_opening, ColMajorMatrix,
            CpuBackendV2, DeviceMultiStarkProvingKeyV2, ProvingContextV2,
        },
        test_utils::{test_whir_config_small, DuplexSpongeValidator, FibFixture, TestFixture},
        verifier::whir::{binary_k_fold, verify_whir, VerifyWhirError},
        WhirConfig, WhirRoundConfig,
    };
    use crate::baby_bear_poseidon2::{BabyBearPoseidon2ConfigV2, EF, F};

    type SCV2 = BabyBearPoseidon2ConfigV2;

    fn generate_random_z(params: &SystemParams, rng: &mut StdRng) -> (Vec<EF>, Vec<EF>) {
        let z_prism: Vec<_> = (0..params.n_stack + 1)
            .map(|_| EF::from_u64(rng.random()))
            .collect();

        let z_cube = {
            let z_cube = z_prism[0]
                .exp_powers_of_2()
                .take(params.l_skip)
                .chain(z_prism[1..].iter().copied())
                .collect_vec();
            debug_assert_eq!(z_cube.len(), params.n_stack + params.l_skip);
            z_cube
        };

        (z_prism, z_cube)
    }

    fn stacking_openings_for_matrix(
        params: &SystemParams,
        z_prism: &[EF],
        matrix: &ColMajorMatrix<F>,
    ) -> Vec<EF> {
        matrix
            .columns()
            .map(|col| {
                Ple::from_evaluations(params.l_skip, col).eval_at_point(
                    params.l_skip,
                    z_prism[0],
                    &z_prism[1..],
                )
            })
            .collect()
    }

    fn run_whir_test(
        params: SystemParams,
        pk: DeviceMultiStarkProvingKeyV2<CpuBackendV2<SCV2>>,
        ctx: &ProvingContextV2<CpuBackendV2<SCV2>>,
    ) -> Result<(), VerifyWhirError> {
        let (common_main_commit, common_main_pcs_data) = {
            let traces = ctx
                .common_main_traces()
                .map(|(_, trace)| trace)
                .collect_vec();
            stacked_commit(
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir(),
                &traces,
            )
        };

        let mut commits = vec![common_main_commit];
        let mut committed_mats = vec![(&common_main_pcs_data.matrix, &common_main_pcs_data.tree)];
        for (air_id, air_ctx) in &ctx.per_trace {
            let pcs_datas = pk.per_air[*air_id]
                .preprocessed_data
                .iter()
                .chain(&air_ctx.cached_mains);
            for cd in pcs_datas {
                let data = &cd.data;
                committed_mats.push((&data.matrix, &data.tree));
                commits.push(data.commit());
            }
        }

        let mut rng = StdRng::seed_from_u64(0);

        let (z_prism, z_cube) = generate_random_z(&params, &mut rng);

        let mut prover_sponge = DuplexSpongeRecorder::default();

        let proof = prove_whir_opening(
            &mut prover_sponge,
            params.l_skip,
            params.log_blowup,
            params.whir(),
            &committed_mats,
            &z_cube,
        );

        let stacking_openings = committed_mats
            .iter()
            .map(|(matrix, _)| stacking_openings_for_matrix(&params, &z_prism, matrix))
            .collect_vec();

        let mut verifier_sponge = DuplexSpongeValidator::new(prover_sponge.into_log());
        verify_whir(
            &mut verifier_sponge,
            &params,
            &proof,
            &stacking_openings,
            &commits,
            &z_cube,
        )
    }

    fn run_whir_fib_test(params: SystemParams) -> Result<(), VerifyWhirError> {
        use crate::{
            poseidon2::sponge::DuplexSponge, prover::DeviceDataTransporterV2,
            BabyBearPoseidon2CpuEngineV2, StarkEngineV2,
        };
        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params.clone());
        let fib = FibFixture::new(0, 1, 1 << params.log_stacked_height());
        let (pk, _vk) = fib.keygen(&engine);
        let pk = engine.device().transport_pk_to_device(&pk);
        let ctx = fib.generate_proving_ctx();
        run_whir_test(params, pk, &ctx)
    }

    #[test_case(0, 1, 1, 0)]
    #[test_case(2, 1, 1, 2)]
    #[test_case(2, 1, 2, 0)]
    #[test_case(2, 1, 3, 1)]
    #[test_case(2, 1, 4, 0)]
    #[test_case(2, 2, 4, 0)]
    fn test_whir_single_fib(
        n_stack: usize,
        log_blowup: usize,
        k_whir: usize,
        log_final_poly_len: usize,
    ) -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);
        let l_skip = 2;
        let whir = test_whir_config_small(log_blowup, l_skip + n_stack, k_whir, log_final_poly_len);

        let params = SystemParams {
            l_skip,
            n_stack,
            log_blowup,
            whir,
            logup: log_up_security_params_baby_bear_100_bits(),
            max_constraint_degree: 3,
        };
        run_whir_fib_test(params)
    }

    #[test]
    fn test_fold_single() {
        let mut rng = StdRng::seed_from_u64(0);

        let a0 = EF::from_u32(rng.random());
        let a1 = EF::from_u32(rng.random());
        let alpha = EF::from_u32(rng.random());
        let x = F::from_u32(rng.random());

        let result = binary_k_fold(vec![a0, a1], &[alpha], x);
        assert_eq!(result, a0 + (alpha - x) * (a0 - a1) * x.double().inverse());
    }

    #[test]
    fn test_fold_double() {
        let mut rng = StdRng::seed_from_u64(0);

        let a0 = EF::from_u32(rng.random());
        let a1 = EF::from_u32(rng.random());
        let a2 = EF::from_u32(rng.random());
        let a3 = EF::from_u32(rng.random());
        let alpha0 = EF::from_u32(rng.random());
        let alpha1 = EF::from_u32(rng.random());

        let x = F::from_u32(rng.random());

        let result = binary_k_fold(vec![a0, a1, a2, a3], &[alpha0, alpha1], x);
        let tw = F::two_adic_generator(2);

        let b0 = a0 + (alpha0 - x) * (a0 - a2) * x.double().inverse();
        let b1 = a1 + (alpha0 - (tw * x)) * (a1 - a3) * (tw * x).double().inverse();
        let x2 = x.square();
        let expected = b0 + (alpha1 - x2) * (b0 - b1) * x2.double().inverse();

        assert_eq!(result, expected);
    }

    fn whir_test_config(k_whir: usize) -> WhirConfig {
        WhirConfig {
            k: k_whir,
            rounds: vec![
                WhirRoundConfig { num_queries: 6 },
                WhirRoundConfig { num_queries: 5 },
            ],
            query_phase_pow_bits: 1,
            folding_pow_bits: 1,
        }
    }

    #[test]
    fn test_whir_multiple_commitments() -> Result<(), VerifyWhirError> {
        setup_tracing_with_log_level(Level::DEBUG);

        let mut rng = StdRng::seed_from_u64(42);

        let params = SystemParams {
            l_skip: 3,
            n_stack: 3,
            log_blowup: 1,
            whir: whir_test_config(2),
            logup: log_up_security_params_baby_bear_100_bits(),
            max_constraint_degree: 3,
        };

        let n_rows = 1 << (params.n_stack + params.l_skip);

        let mut matrices = vec![];
        let mut commits = vec![];
        let mut trees = vec![];

        let num_commitments = 5;
        for _ in 0..num_commitments {
            let n_cols = (rng.random::<u64>() % 10 + 3) as usize;
            let data = (0..n_rows * n_cols)
                .map(|_| F::from_u64(rng.random()))
                .collect_vec();
            let mat = ColMajorMatrix::new(data, n_cols);

            let (commit, pcs_data) = stacked_commit(
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir(),
                &[&mat],
            );

            matrices.push(mat);
            commits.push(commit);
            trees.push(pcs_data.tree);
        }

        debug_assert_eq!(matrices[0].height(), 1 << (params.n_stack + params.l_skip));

        let (z_prism, z_cube) = generate_random_z(&params, &mut rng);

        let mut prover_sponge = DuplexSpongeRecorder::default();

        let committed_mats = matrices.iter().zip(trees.iter()).collect_vec();
        let proof = prove_whir_opening(
            &mut prover_sponge,
            params.l_skip,
            params.log_blowup,
            params.whir(),
            &committed_mats,
            &z_cube,
        );

        let stacking_openings: Vec<Vec<EF>> = matrices
            .iter()
            .map(|mat| stacking_openings_for_matrix(&params, &z_prism, mat))
            .collect();

        let mut verifier_sponge = DuplexSpongeValidator::new(prover_sponge.into_log());
        verify_whir(
            &mut verifier_sponge,
            &params,
            &proof,
            &stacking_openings,
            &commits,
            &z_cube,
        )
    }

    #[test]
    fn test_whir_multiple_commitments_negative() {
        setup_tracing_with_log_level(Level::DEBUG);

        let mut rng = StdRng::seed_from_u64(42);

        let params = SystemParams {
            l_skip: 3,
            n_stack: 3,
            log_blowup: 1,
            whir: whir_test_config(2),
            logup: log_up_security_params_baby_bear_100_bits(),
            max_constraint_degree: 3,
        };

        let n_rows = 1 << (params.n_stack + params.l_skip);

        let mut matrices = vec![];
        let mut commits = vec![];
        let mut trees = vec![];

        let num_commitments = 5;
        for _ in 0..num_commitments {
            let n_cols = (rng.random::<u64>() % 10 + 3) as usize;
            let data = (0..n_rows * n_cols)
                .map(|_| F::from_u64(rng.random()))
                .collect_vec();
            let mat = ColMajorMatrix::new(data, n_cols);

            let (commit, pcs_data) = stacked_commit(
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir(),
                &[&mat],
            );

            matrices.push(mat);
            commits.push(commit);
            trees.push(pcs_data.tree);
        }

        debug_assert_eq!(matrices[0].height(), 1 << (params.n_stack + params.l_skip));

        let (z_prism, z_cube) = generate_random_z(&params, &mut rng);

        let mut prover_sponge = DuplexSponge::default();
        let mut verifier_sponge = DuplexSponge::default();

        let committed_mats = matrices.iter().zip(trees.iter()).collect_vec();
        let proof = prove_whir_opening(
            &mut prover_sponge,
            params.l_skip,
            params.log_blowup,
            params.whir(),
            &committed_mats,
            &z_cube,
        );

        let mut stacking_openings: Vec<Vec<EF>> = matrices
            .iter()
            .map(|mat| stacking_openings_for_matrix(&params, &z_prism, mat))
            .collect();

        // change an opening to test soundness
        stacking_openings[1][2] = EF::ONE;

        assert!(matches!(
            verify_whir(
                &mut verifier_sponge,
                &params,
                &proof,
                &stacking_openings,
                &commits,
                &z_cube,
            ),
            Err(VerifyWhirError::FinalPolyConstraint)
        ));
    }
}
