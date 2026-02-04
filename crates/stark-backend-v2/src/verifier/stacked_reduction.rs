use std::iter::zip;

use itertools::Itertools;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;
use tracing::{debug, instrument};

use crate::{
    poly_common::{
        eval_eq_mle, eval_eq_prism, eval_in_uni, eval_rot_kernel_prism, horner_eval,
        interpolate_quadratic_at_012,
    },
    poseidon2::sponge::FiatShamirTranscript,
    proof::StackingProof,
    prover::stacked_pcs::StackedLayout,
    EF, F,
};

#[derive(Error, Debug, PartialEq, Eq)]
pub enum StackedReductionError {
    #[error("s_0 does not match s_0 polynomial evaluation sum: {s_0} != {s_0_sum_eval}")]
    S0Mismatch { s_0: EF, s_0_sum_eval: EF },

    #[error("s_n(u_n) does not match claimed q(u) sum: {claim} != {final_sum}")]
    FinalSumMismatch { claim: EF, final_sum: EF },
}

/// `has_preprocessed` must be per present trace in sorted AIR order.
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip_all)]
pub fn verify_stacked_reduction<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    proof: &StackingProof,
    layouts: &[StackedLayout],
    l_skip: usize,
    n_stack: usize,
    column_openings: &Vec<Vec<Vec<(EF, EF)>>>,
    r: &[EF],
    omega_shift_pows: &[F],
) -> Result<Vec<EF>, StackedReductionError> {
    /*
     * SETUP
     *
     * We start by setting up for the rounds below. Most importantly, we need to ensure that the
     * order we process column_openings is the same as the stacked reduction prover. The prover
     * orders the claims per commit -> per column (as in layouts), but column_openings is per AIR
     * -> per part (common main, preprocessed, then cached) -> per column. Note that the verifier
     * needs to compute and pass in has_preprocessed, which is expected to be sorted in the same
     * way column_openings is (i.e. sorted by trace height).
     */
    let omega_order = omega_shift_pows.len();
    let omega_order_f = F::from_usize(omega_order);

    let t_claims_len = layouts
        .iter()
        .map(|l| l.sorted_cols.len() * 2)
        .sum::<usize>();
    let mut t_claims = Vec::with_capacity(t_claims_len);

    // common main columns
    column_openings.iter().for_each(|parts| {
        t_claims.extend(parts[0].iter().flat_map(|(t, t_rot)| [*t, *t_rot]));
    });

    // preprocessed and cached columns
    column_openings.iter().for_each(|parts| {
        t_claims.extend(
            parts
                .iter()
                .skip(1)
                .flat_map(|cols| cols.iter().flat_map(|(t, t_rot)| [*t, *t_rot])),
        );
    });

    assert_eq!(t_claims.len(), t_claims_len);
    debug!(?t_claims);

    let lambda = transcript.sample_ext();
    let lambda_powers = lambda.powers().take(t_claims_len).collect_vec();

    /*
     * INITIAL UNIVARIATE ROUND
     *
     * In this round we compute s_0 = sum_i (t_i * lambda^i) from the column opening claims t_i
     * and compare it to the s_1 polynomial in proof. If the polynomial was correctly computed,
     * then we should have s_0 == sum_{z in D} s_1(z).
     *
     * Note that we abuse the properties of multiplicative subgroup D to speed up the computation
     * of sum_{z in D} s_1(z). Suppose s_1(x) = a_0 + a_1 * x + ... a_k * x^k. Because we have
     * omega^{|D|} == 1, sum_{z in D} s_1(z) = |D| * (a_0 + a_{|D|} + ...).
     */
    let s_0 = zip(&t_claims, &lambda_powers)
        .map(|(&t_i, &lambda_i)| t_i * lambda_i)
        .sum::<EF>();
    let s_0_sum_eval = proof
        .univariate_round_coeffs
        .iter()
        .step_by(omega_order)
        .copied()
        .sum::<EF>()
        * omega_order_f;

    if s_0 != s_0_sum_eval {
        return Err(StackedReductionError::S0Mismatch { s_0, s_0_sum_eval });
    }

    for coeffs in &proof.univariate_round_coeffs {
        transcript.observe_ext(*coeffs);
    }

    let mut u = vec![EF::ZERO; n_stack + 1];
    u[0] = transcript.sample_ext();
    debug!(round = 0, u_round = %u[0]);

    let mut s_j_0 = s_0;
    let mut claim = horner_eval(&proof.univariate_round_coeffs, u[0]);

    /*
     * SUMCHECK ROUNDS 1 TO N
     *
     * We sample size n_stack vector u using the transcript, and run the verifier sumcheck for
     * rounds 1 to n_stack. We start by evaluating the univariate round polynomial at u_0, which
     * we store as s_0(u_0). We then evaluate s_j(0) = s_{j - 1}(u_{j - 1}) - s_j(1) for each j,
     * which we then use with s_j(1) and s_j(2) to interpolate s_j(u_j).
     */

    u.iter_mut().enumerate().skip(1).for_each(|(j, u_j)| {
        let s_j_1 = proof.sumcheck_round_polys[j - 1][0];
        let s_j_2 = proof.sumcheck_round_polys[j - 1][1];
        transcript.observe_ext(s_j_1);
        transcript.observe_ext(s_j_2);
        *u_j = transcript.sample_ext();
        s_j_0 = claim - s_j_1;
        claim = interpolate_quadratic_at_012(&[s_j_0, s_j_1, s_j_2], *u_j);
        debug!(round = %j, sum_claim = %claim);
    });

    /*
     * FINAL VERIFICATION
     *
     * Finally, to verify that the claims about t_i(r) were properly reduced we assert that the
     * final s_{n_stack}(u_{n_stack}) == sum_j (lambda^j * q_{j'}(u) * h(u, r, b_j)), where each
     * j maps to some (non-unique) j' and h(u, r, b_j) is either (a) eq(u_{n_j}, r_{n_j}) *
     * eq(u_{> n_j}, b_j) or (b) rot(u_{n_j}, r_{n_j}) * eq(u_{> n_j}, b_j).
     *
     * It is up to the verifier to compute each h(u, r, b_j). Let q_coeffs[j'] be the sum of all
     * lambda^j * h(u, r, b_j) such that j maps to j' - given claims q_{j'}(u), we thus want to
     * assert s_{n_stack}(u_{n_stack}) == sum_{j'} q_{j'}(u) * q_coeffs[j'].
     */
    let mut q_coeffs = proof
        .stacking_openings
        .iter()
        .map(|vec| vec![EF::ZERO; vec.len()])
        .collect_vec();

    let mut j = 0usize;
    layouts
        .iter()
        .zip(q_coeffs.iter_mut())
        .for_each(|(layout, coeffs)| {
            layout.sorted_cols.iter().for_each(|&(_, _, s)| {
                let n = s.log_height() as isize - l_skip as isize;
                let n_lift = n.max(0) as usize;
                let b = (l_skip + n_lift..l_skip + n_stack)
                    .map(|j| F::from_bool((s.row_idx >> j) & 1 == 1))
                    .collect_vec();
                let eq_mle = eval_eq_mle(&u[n_lift + 1..], &b);
                let ind = eval_in_uni(l_skip, n, u[0]);
                let (l, rs_n) = if n.is_negative() {
                    (
                        l_skip.wrapping_add_signed(n),
                        &[r[0].exp_power_of_2(-n as usize)] as &[_],
                    )
                } else {
                    (l_skip, &r[..=n_lift])
                };
                let eq_prism = eval_eq_prism(l, &u[..=n_lift], rs_n);
                let rot_kernel_prism = eval_rot_kernel_prism(l, &u[..=n_lift], rs_n);
                coeffs[s.col_idx] += eq_mle
                    * (lambda_powers[j] * eq_prism + lambda_powers[j + 1] * rot_kernel_prism)
                    * ind;
                j += 2;
            });
        });

    let final_sum = q_coeffs.iter().zip(proof.stacking_openings.iter()).fold(
        EF::ZERO,
        |acc, (q_coeff_vec, q_j_vec)| {
            acc + q_coeff_vec
                .iter()
                .zip(q_j_vec.iter())
                .fold(EF::ZERO, |acc, (&q_coeff, &q_j)| {
                    transcript.observe_ext(q_j);
                    acc + (q_coeff * q_j)
                })
        },
    );

    if claim != final_sum {
        return Err(StackedReductionError::FinalSumMismatch { claim, final_sum });
    }

    Ok(u)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_dft::{Radix2Bowers, TwoAdicSubgroupDft};
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32, TwoAdicField};
    use p3_util::log2_ceil_usize;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{poseidon2::sponge::DuplexSponge, prover::stacked_pcs::StackedSlice};

    const N_STACK: usize = 4;
    const L_SKIP: usize = 2;

    struct StackedReductionTestCase {
        pub transcript: DuplexSponge,
        pub proof: StackingProof,
        pub layouts: Vec<StackedLayout>,
        pub column_openings: Vec<Vec<Vec<(EF, EF)>>>,
        pub r: Vec<EF>,
        pub omega_pows: Vec<F>,
    }

    fn compute_t<const ROT: bool>(
        q: impl Fn(&[EF]) -> EF,
        r: &[EF],
        b: &[F],
        u: &[EF],
        x: EF,
        round: usize,
        l_skip: usize,
    ) -> EF {
        let n_t = N_STACK - b.len();
        let mut sum = EF::ZERO;
        for i in 0..(1 << (N_STACK - round)) {
            let hypercube = (0..(N_STACK - round))
                .map(|bit_idx| EF::from_usize((i >> bit_idx) & 1))
                .collect_vec();
            let z = u
                .iter()
                .take(round)
                .chain([x].iter())
                .chain(hypercube.iter())
                .copied()
                .collect_vec();
            let eq_or_rot_eval = if ROT {
                eval_rot_kernel_prism(l_skip, &z[..=n_t], &r[..=n_t])
            } else {
                eval_eq_prism(l_skip, &z[..=n_t], &r[..=n_t])
            };
            sum += q(&z) * eq_or_rot_eval * eval_eq_mle(&z[n_t + 1..], b);
        }
        sum
    }

    fn generate_random_linear_q(rng: &mut StdRng) -> impl Fn(&[EF]) -> EF {
        let coeffs = (0..=N_STACK)
            .map(|_| EF::from_usize(rng.random_range(0usize..100)))
            .collect_vec();
        move |vals: &[EF]| {
            coeffs
                .iter()
                .zip(vals)
                .fold(EF::ZERO, |acc, (&coeff, &val)| acc + coeff * val)
        }
    }

    fn generate_single_column_test_case() -> StackedReductionTestCase {
        let mut rng = StdRng::from_seed([42; 32]);
        let omega_pows = F::two_adic_generator(L_SKIP)
            .powers()
            .take(1 << L_SKIP)
            .collect_vec();

        let slice = StackedSlice::new(0, 0, L_SKIP + N_STACK);
        let layout = StackedLayout::from_raw_parts(L_SKIP, vec![(0, 0, slice)]);

        let q = generate_random_linear_q(&mut rng);
        let r = (0..=N_STACK)
            .map(|_| EF::from_u32(rng.random_range(0..F::ORDER_U32)))
            .collect_vec();
        let n = slice.log_height() - L_SKIP;
        let b = (L_SKIP + n..L_SKIP + N_STACK)
            .map(|j| F::from_bool((slice.row_idx >> j) & 1 == 1))
            .collect_vec();
        let mut u = vec![];

        let t = omega_pows.iter().fold(EF::ZERO, |acc, &omega| {
            acc + compute_t::<false>(&q, &r, &b, &u, EF::from(omega), 0, L_SKIP)
        });
        let t_rot = omega_pows.iter().fold(EF::ZERO, |acc, &omega| {
            acc + compute_t::<true>(&q, &r, &b, &u, EF::from(omega), 0, L_SKIP)
        });
        let column_openings = vec![vec![vec![(t, t_rot)]]];

        let mut transcript = DuplexSponge::default();
        let lambda = transcript.sample_ext();

        let s_0_deg = 2 * ((1 << L_SKIP) - 1);
        let log_dft_size = log2_ceil_usize(s_0_deg + 1);
        let two_adic_gen = F::two_adic_generator(log_dft_size);

        let univariate_round_evals = two_adic_gen
            .powers()
            .take(1 << log_dft_size)
            .map(|z| {
                compute_t::<false>(&q, &r, &b, &u, z.into(), 0, L_SKIP)
                    + lambda * compute_t::<true>(&q, &r, &b, &u, z.into(), 0, L_SKIP)
            })
            .collect_vec();
        let univariate_round_coeffs = Radix2Bowers.coset_idft(univariate_round_evals, EF::ONE);

        for coeffs in &univariate_round_coeffs {
            transcript.observe_ext(*coeffs);
        }

        let mut sumcheck_round_polys = vec![];
        u.push(transcript.sample_ext());
        for round in 1..=N_STACK {
            sumcheck_round_polys.push([
                compute_t::<false>(&q, &r, &b, &u, EF::ONE, round, L_SKIP)
                    + lambda * compute_t::<true>(&q, &r, &b, &u, EF::ONE, round, L_SKIP),
                compute_t::<false>(&q, &r, &b, &u, EF::TWO, round, L_SKIP)
                    + lambda * compute_t::<true>(&q, &r, &b, &u, EF::TWO, round, L_SKIP),
            ]);
            transcript.observe_ext(sumcheck_round_polys[round - 1][0]);
            transcript.observe_ext(sumcheck_round_polys[round - 1][1]);
            u.push(transcript.sample_ext());
        }

        let q_at_u = q(&u);
        transcript.observe_ext(q_at_u);

        let proof = StackingProof {
            univariate_round_coeffs,
            sumcheck_round_polys,
            stacking_openings: vec![vec![q_at_u]],
        };

        StackedReductionTestCase {
            transcript: DuplexSponge::default(),
            proof,
            layouts: vec![layout],
            column_openings,
            r,
            omega_pows,
        }
    }

    #[test]
    fn verify_single_column_test() {
        let mut test_case = generate_single_column_test_case();
        verify_stacked_reduction(
            &mut test_case.transcript,
            &test_case.proof,
            &test_case.layouts,
            L_SKIP,
            N_STACK,
            &test_case.column_openings,
            &test_case.r,
            &test_case.omega_pows,
        )
        .unwrap();
    }

    #[test]
    fn single_column_univariate_round_negative_test() {
        let mut test_case = generate_single_column_test_case();
        test_case.proof.univariate_round_coeffs[0] += EF::ONE;
        verify_stacked_reduction(
            &mut test_case.transcript,
            &test_case.proof,
            &test_case.layouts,
            L_SKIP,
            N_STACK,
            &test_case.column_openings,
            &test_case.r,
            &test_case.omega_pows,
        )
        .unwrap_err();
    }

    #[test]
    fn single_column_sumcheck_rounds_negative_test() {
        let mut test_case = generate_single_column_test_case();
        test_case.proof.sumcheck_round_polys[N_STACK - 1][0] += EF::ONE;
        verify_stacked_reduction(
            &mut test_case.transcript,
            &test_case.proof,
            &test_case.layouts,
            L_SKIP,
            N_STACK,
            &test_case.column_openings,
            &test_case.r,
            &test_case.omega_pows,
        )
        .unwrap_err();
    }

    #[test]
    fn single_column_stacking_openings_negative_test() {
        let mut test_case = generate_single_column_test_case();
        test_case.proof.stacking_openings[0][0] += EF::ONE;
        verify_stacked_reduction(
            &mut test_case.transcript,
            &test_case.proof,
            &test_case.layouts,
            L_SKIP,
            N_STACK,
            &test_case.column_openings,
            &test_case.r,
            &test_case.omega_pows,
        )
        .unwrap_err();
    }
}
