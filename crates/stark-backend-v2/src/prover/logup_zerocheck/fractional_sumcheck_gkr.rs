use std::ops::Add;

use p3_field::{Field, PrimeCharacteristicRing};
use p3_util::log2_strict_usize;
use tracing::{debug, instrument};

use crate::{
    block_sumcheck_sizes,
    poseidon2::sponge::FiatShamirTranscript,
    proof::GkrLayerClaims,
    prover::{
        poly::evals_eq_hypercube,
        sumcheck::{fold_mle_evals, gkr_block_sumcheck_poly_evals},
        ColMajorMatrix,
    },
    EF,
};

/// Proof for fractional sumcheck protocol
pub struct FracSumcheckProof<EF> {
    /// The fractional sum p_0 / q_0
    pub fractional_sum: (EF, EF),
    /// The claims for p_j(0, rho), p_j(1, rho), q_j(0, rho), and q_j(1, rho) for each layer j > 0.
    pub claims_per_layer: Vec<GkrLayerClaims>,
    /// Block sumcheck polynomials per layer. Each inner-most Vec has 4^k - 1 elements
    /// where k is the block size for that block, except maybe the last layer.
    pub block_sumcheck_polys: Vec<Vec<Vec<EF>>>,
}

#[derive(Clone, Copy, Debug, Default, derive_new::new)]
#[repr(C)]
pub struct Frac<EF> {
    // PERF[jpw]: in the initial round, we can keep `p` in base field
    pub p: EF,
    pub q: EF,
}

impl<EF: Field> Add<Frac<EF>> for Frac<EF> {
    type Output = Frac<EF>;

    fn add(self, other: Frac<EF>) -> Self::Output {
        Frac {
            p: self.p * other.q + self.q * other.p,
            q: self.q * other.q,
        }
    }
}

/// Runs the fractional sumcheck protocol using GKR layered circuit.
///
/// # Arguments
/// * `transcript` - The Fiat-Shamir transcript
/// * `evals` - list of `(p, q)` pairs of fractions in projective coordinates representing
///   evaluations on the hypercube
/// * `assert_zero` - Whether to assert that the final sum is zero. If `true`, then the transcript
///   will not observe the numerator of the final sum.
///
/// # Returns
/// The fractional sumcheck proof and the final random evaluation vector.
#[instrument(level = "info", skip_all)]
pub fn fractional_sumcheck<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    evals: &[Frac<EF>],
    assert_zero: bool,
) -> (FracSumcheckProof<EF>, Vec<EF>) {
    if evals.is_empty() {
        return (
            FracSumcheckProof {
                fractional_sum: (EF::ZERO, EF::ONE),
                claims_per_layer: vec![],
                block_sumcheck_polys: vec![],
            },
            vec![],
        );
    }
    // total_rounds = l_skip + n_logup
    let total_rounds = log2_strict_usize(evals.len());
    // block sumcheck polys for layers j=2,...,total_rounds
    let mut block_sumcheck_polys = Vec::with_capacity(total_rounds);

    // segment tree: layer i=0,...,total_rounds starts at 2^i (index 0 unused)
    let mut tree_evals: Vec<Frac<EF>> = vec![Frac::default(); 2 << total_rounds];
    tree_evals[(1 << total_rounds)..].copy_from_slice(evals);

    for node_idx in (1..(1 << total_rounds)).rev() {
        tree_evals[node_idx] = tree_evals[2 * node_idx] + tree_evals[2 * node_idx + 1];
    }
    let frac_sum = tree_evals[1];
    if assert_zero {
        assert_eq!(frac_sum.p, EF::ZERO);
    } else {
        transcript.observe_ext(frac_sum.p);
    }
    transcript.observe_ext(frac_sum.q);

    // Index i is for layer i+1
    let mut claims_per_layer: Vec<GkrLayerClaims> = Vec::with_capacity(total_rounds);

    // Process each GKR round
    // `j = round + 1` goes from `1, ..., total_rounds`
    //
    // Round `j = 1` is special since "sumcheck" is directly checked by verifier
    claims_per_layer.push(GkrLayerClaims {
        p_xi_0: tree_evals[2].p,
        q_xi_0: tree_evals[2].q,
        p_xi_1: tree_evals[3].p,
        q_xi_1: tree_evals[3].q,
    });
    transcript.observe_ext(claims_per_layer[0].p_xi_0);
    transcript.observe_ext(claims_per_layer[0].q_xi_0);
    transcript.observe_ext(claims_per_layer[0].p_xi_1);
    transcript.observe_ext(claims_per_layer[0].q_xi_1);
    let mu_1 = transcript.sample_ext();
    debug!(gkr_round = 0, mu = %mu_1);
    // ξ^{(j-1)}
    let mut xi_prev = vec![mu_1];

    // GKR rounds
    for round in 1..total_rounds {
        // Number of hypercube points
        let eval_size = 1 << round;
        // We apply batch sumcheck to the polynomials
        // \eq(ξ^{(j-1)}, Y) (\hat p_j(0, Y) \hat q_j(1, Y) + \hat p_j(1, Y) \hat q_j(0, Y))
        // \eq(ξ^{(j-1)}, Y) (\hat q_j(0, Y) \hat q_j(1, Y))
        // Note: these are polynomials of degree 3 in each Y_i coordinate.

        // Sample λ_j for batching
        let lambda = transcript.sample_ext();
        debug!(gkr_round = round, %lambda);

        // Columns are p_j0, q_j0, p_j1, q_j1
        // PERF: use a view instead of re-allocating memory
        let mut pq_j_evals = EF::zero_vec(4 * eval_size);
        let segment = &tree_evals[2 * eval_size..4 * eval_size];
        for x in 0..eval_size {
            pq_j_evals[x] = segment[2 * x].p;
            pq_j_evals[eval_size + x] = segment[2 * x].q;
            pq_j_evals[2 * eval_size + x] = segment[2 * x + 1].p;
            pq_j_evals[3 * eval_size + x] = segment[2 * x + 1].q;
        }
        let mut pq_j_evals = ColMajorMatrix::new(pq_j_evals, 4);
        let mut eq_xis = ColMajorMatrix::new(evals_eq_hypercube(&xi_prev), 1);

        // Block sumcheck where the round polynomial is evaluated on {0,1,2,3}^k
        let (round_blocks_eval, rho) = {
            let n = round;
            let mut round_blocks_eval = Vec::with_capacity(n);
            let mut r_vec = Vec::with_capacity(n);

            for (block_idx, k) in block_sumcheck_sizes(n).enumerate() {
                let block_evals = gkr_block_sumcheck_poly_evals(&eq_xis, &pq_j_evals, lambda, k);
                for &eval in &block_evals {
                    transcript.observe_ext(eval);
                }

                for _ in 0..k {
                    let r_round = transcript.sample_ext();
                    r_vec.push(r_round);

                    pq_j_evals = fold_mle_evals(pq_j_evals, r_round);
                    eq_xis = fold_mle_evals(eq_xis, r_round);
                }
                debug!(
                    gkr_round = round,
                    block = block_idx,
                    k,
                    "block_sumcheck_round"
                );

                round_blocks_eval.push(block_evals);
            }
            (round_blocks_eval, r_vec)
        };
        claims_per_layer.push(GkrLayerClaims {
            p_xi_0: pq_j_evals.column(0)[0],
            q_xi_0: pq_j_evals.column(1)[0],
            p_xi_1: pq_j_evals.column(2)[0],
            q_xi_1: pq_j_evals.column(3)[0],
        });
        transcript.observe_ext(claims_per_layer[round].p_xi_0);
        transcript.observe_ext(claims_per_layer[round].q_xi_0);
        transcript.observe_ext(claims_per_layer[round].p_xi_1);
        transcript.observe_ext(claims_per_layer[round].q_xi_1);

        // Sample μ_j for reduction to single evaluation point
        let mu = transcript.sample_ext();
        debug!(gkr_round = round, %mu);

        // Update ξ^{(j)} = (μ_j, ρ^{(j-1)})
        xi_prev = [vec![mu], rho].concat();

        block_sumcheck_polys.push(round_blocks_eval);
    }

    (
        FracSumcheckProof {
            fractional_sum: (frac_sum.p, frac_sum.q),
            claims_per_layer,
            block_sumcheck_polys,
        },
        xi_prev,
    )
}
