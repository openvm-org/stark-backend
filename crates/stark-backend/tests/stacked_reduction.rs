//! Verifier-only unit tests for stacked reduction (`verify_stacked_reduction`).
//!
//! These hand-construct a `StackingProof` from first principles — computing
//! `compute_t` over the hypercube and applying DFT for the univariate round —
//! then verify it. Negative tests tamper with individual proof components
//! (univariate coeffs, sumcheck polys, stacking openings) to confirm rejection.
//!
//! No engine or prover backend is involved; this is a self-contained verifier
//! correctness test, so it is not in the shared backend test suite.

use itertools::Itertools;
use openvm_stark_backend::{
    poly_common::{eval_eq_mle, eval_eq_prism, eval_rot_kernel_prism},
    proof::StackingProof,
    prover::stacked_pcs::{StackedLayout, StackedSlice},
    verifier::stacked_reduction::verify_stacked_reduction,
    FiatShamirTranscript,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::*;
use p3_dft::{Radix2Bowers, TwoAdicSubgroupDft};
use p3_field::{PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, Rng, SeedableRng};

type SC = BabyBearPoseidon2Config;

const N_STACK: usize = 4;
const L_SKIP: usize = 2;

struct StackedReductionTestCase {
    pub transcript: DuplexSponge,
    pub proof: StackingProof<SC>,
    pub layouts: Vec<StackedLayout>,
    pub need_rot_per_commit: Vec<Vec<bool>>,
    pub column_openings: Vec<Vec<Vec<EF>>>,
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
    let layout =
        StackedLayout::from_raw_parts(L_SKIP, L_SKIP + N_STACK, vec![(0, 0, slice)]).unwrap();

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
    let column_openings = vec![vec![vec![t, t_rot]]];

    let mut transcript = default_duplex_sponge();
    let lambda = FiatShamirTranscript::<SC>::sample_ext(&mut transcript);

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
        FiatShamirTranscript::<SC>::observe_ext(&mut transcript, *coeffs);
    }

    let mut sumcheck_round_polys = vec![];
    u.push(FiatShamirTranscript::<SC>::sample_ext(&mut transcript));
    for round in 1..=N_STACK {
        sumcheck_round_polys.push([
            compute_t::<false>(&q, &r, &b, &u, EF::ONE, round, L_SKIP)
                + lambda * compute_t::<true>(&q, &r, &b, &u, EF::ONE, round, L_SKIP),
            compute_t::<false>(&q, &r, &b, &u, EF::TWO, round, L_SKIP)
                + lambda * compute_t::<true>(&q, &r, &b, &u, EF::TWO, round, L_SKIP),
        ]);
        FiatShamirTranscript::<SC>::observe_ext(
            &mut transcript,
            sumcheck_round_polys[round - 1][0],
        );
        FiatShamirTranscript::<SC>::observe_ext(
            &mut transcript,
            sumcheck_round_polys[round - 1][1],
        );
        u.push(FiatShamirTranscript::<SC>::sample_ext(&mut transcript));
    }

    let q_at_u = q(&u);
    FiatShamirTranscript::<SC>::observe_ext(&mut transcript, q_at_u);

    let proof = StackingProof::<SC> {
        univariate_round_coeffs,
        sumcheck_round_polys,
        stacking_openings: vec![vec![q_at_u]],
    };

    StackedReductionTestCase {
        transcript: default_duplex_sponge(),
        proof,
        layouts: vec![layout],
        need_rot_per_commit: vec![vec![true]],
        column_openings,
        r,
        omega_pows,
    }
}

#[test]
fn verify_single_column_test() -> eyre::Result<()> {
    let mut test_case = generate_single_column_test_case();
    verify_stacked_reduction::<SC, _>(
        &mut test_case.transcript,
        &test_case.proof,
        &test_case.layouts,
        &test_case.need_rot_per_commit,
        L_SKIP,
        N_STACK,
        &test_case.column_openings,
        &test_case.r,
        &test_case.omega_pows,
    )?;
    Ok(())
}

#[test]
fn single_column_univariate_round_negative_test() {
    let mut test_case = generate_single_column_test_case();
    test_case.proof.univariate_round_coeffs[0] += EF::ONE;
    verify_stacked_reduction::<SC, _>(
        &mut test_case.transcript,
        &test_case.proof,
        &test_case.layouts,
        &test_case.need_rot_per_commit,
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
    verify_stacked_reduction::<SC, _>(
        &mut test_case.transcript,
        &test_case.proof,
        &test_case.layouts,
        &test_case.need_rot_per_commit,
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
    verify_stacked_reduction::<SC, _>(
        &mut test_case.transcript,
        &test_case.proof,
        &test_case.layouts,
        &test_case.need_rot_per_commit,
        L_SKIP,
        N_STACK,
        &test_case.column_openings,
        &test_case.r,
        &test_case.omega_pows,
    )
    .unwrap_err();
}
