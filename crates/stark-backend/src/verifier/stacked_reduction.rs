use std::iter::zip;

use itertools::Itertools;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use thiserror::Error;
use tracing::{debug, instrument};

use crate::{
    poly_common::{
        eval_eq_mle, eval_eq_prism, eval_in_uni, eval_rot_kernel_prism, horner_eval,
        interpolate_quadratic_at_012,
    },
    proof::{column_openings_by_rot, StackingProof},
    prover::stacked_pcs::StackedLayout,
    FiatShamirTranscript, StarkProtocolConfig,
};

#[derive(Error, Debug, PartialEq, Eq)]
pub enum StackedReductionError<EF: core::fmt::Debug + core::fmt::Display + PartialEq + Eq> {
    #[error("s_0 does not match s_0 polynomial evaluation sum: {s_0} != {s_0_sum_eval}")]
    S0Mismatch { s_0: EF, s_0_sum_eval: EF },

    #[error("s_n(u_n) does not match claimed q(u) sum: {claim} != {final_sum}")]
    FinalSumMismatch { claim: EF, final_sum: EF },
}

/// `has_preprocessed` must be per present trace in sorted AIR order.
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip_all)]
pub fn verify_stacked_reduction<SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>>(
    transcript: &mut TS,
    proof: &StackingProof<SC>,
    layouts: &[StackedLayout],
    need_rot_per_commit: &[Vec<bool>],
    l_skip: usize,
    n_stack: usize,
    column_openings: &Vec<Vec<Vec<SC::EF>>>,
    r: &[SC::EF],
    omega_shift_pows: &[SC::F],
) -> Result<Vec<SC::EF>, StackedReductionError<SC::EF>>
where
    SC::EF: TwoAdicField,
{
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
    let omega_order_f = SC::F::from_usize(omega_order);

    debug_assert_eq!(layouts.len(), need_rot_per_commit.len());
    let mut lambda_idx = 0usize;
    let lambda_indices_per_layout: Vec<Vec<(usize, bool)>> = layouts
        .iter()
        .enumerate()
        .map(|(commit_idx, layout)| {
            let need_rot_for_commit = &need_rot_per_commit[commit_idx];
            debug_assert_eq!(need_rot_for_commit.len(), layout.mat_starts.len());
            layout
                .sorted_cols
                .iter()
                .map(|&(mat_idx, _col_idx, _slice)| {
                    lambda_idx += 1;
                    (lambda_idx - 1, need_rot_for_commit[mat_idx])
                })
                .collect_vec()
        })
        .collect_vec();
    let t_claims_len = lambda_idx;
    let mut t_claims = Vec::with_capacity(t_claims_len);

    // common main columns (commit 0)
    for (trace_idx, parts) in column_openings.iter().enumerate() {
        let need_rot = need_rot_per_commit[0][trace_idx];
        t_claims.extend(column_openings_by_rot(&parts[0], need_rot));
    }

    // preprocessed and cached columns (commits 1..)
    let mut commit_idx = 1usize;
    for parts in column_openings {
        for cols in parts.iter().skip(1) {
            let need_rot = need_rot_per_commit[commit_idx][0];
            t_claims.extend(column_openings_by_rot(cols, need_rot));
            commit_idx += 1;
        }
    }

    assert_eq!(t_claims.len(), t_claims_len);
    debug!(?t_claims);

    let lambda = transcript.sample_ext();
    let lambda_sqr_powers = (lambda * lambda).powers().take(t_claims_len).collect_vec();

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
    let s_0 = zip(&t_claims, &lambda_sqr_powers)
        .map(|(&t_i, &lambda_i)| (t_i.0 + t_i.1 * lambda) * lambda_i)
        .sum::<SC::EF>();
    let s_0_sum_eval = proof
        .univariate_round_coeffs
        .iter()
        .step_by(omega_order)
        .copied()
        .sum::<SC::EF>()
        * omega_order_f;

    if s_0 != s_0_sum_eval {
        return Err(StackedReductionError::S0Mismatch { s_0, s_0_sum_eval });
    }

    for coeffs in &proof.univariate_round_coeffs {
        transcript.observe_ext(*coeffs);
    }

    let mut u = vec![SC::EF::ZERO; n_stack + 1];
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
        .map(|vec| vec![SC::EF::ZERO; vec.len()])
        .collect_vec();

    layouts
        .iter()
        .enumerate()
        .zip(q_coeffs.iter_mut())
        .for_each(|((commit_idx, layout), coeffs)| {
            let lambda_indices = &lambda_indices_per_layout[commit_idx];
            layout
                .sorted_cols
                .iter()
                .enumerate()
                .for_each(|(col_idx, &(_, _, s))| {
                    let (lambda_idx, need_rot) = lambda_indices[col_idx];
                    let n = s.log_height() as isize - l_skip as isize;
                    let n_lift = n.max(0) as usize;
                    let b = (l_skip + n_lift..l_skip + n_stack)
                        .map(|j| SC::F::from_bool((s.row_idx >> j) & 1 == 1))
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
                    let mut batched = lambda_sqr_powers[lambda_idx] * eq_prism;
                    if need_rot {
                        let rot_kernel_prism = eval_rot_kernel_prism(l, &u[..=n_lift], rs_n);
                        batched += lambda_sqr_powers[lambda_idx] * lambda * rot_kernel_prism;
                    }
                    coeffs[s.col_idx] += eq_mle * batched * ind;
                });
        });

    let final_sum = q_coeffs.iter().zip(proof.stacking_openings.iter()).fold(
        SC::EF::ZERO,
        |acc, (q_coeff_vec, q_j_vec)| {
            acc + q_coeff_vec.iter().zip(q_j_vec.iter()).fold(
                SC::EF::ZERO,
                |acc, (&q_coeff, &q_j)| {
                    transcript.observe_ext(q_j);
                    acc + (q_coeff * q_j)
                },
            )
        },
    );

    if claim != final_sum {
        return Err(StackedReductionError::FinalSumMismatch { claim, final_sum });
    }

    Ok(u)
}
