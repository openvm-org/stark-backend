use std::ops::Add;

use p3_field::{Field, PrimeCharacteristicRing};
use p3_util::log2_strict_usize;
use tracing::{debug, instrument};

use crate::{
    poseidon2::sponge::FiatShamirTranscript,
    proof::GkrLayerClaims,
    prover::{
        poly::evals_eq_hypercube,
        sumcheck::{fold_mle_evals, sumcheck_round_poly_evals},
        ColMajorMatrix,
    },
    EF,
};

/// Proof for fractional sumcheck protocol with data-parallel GKR.
///
/// When `n_grid = 0`, there is exactly one grid point and this reduces to the
/// non-parallel case.
pub struct FracSumcheckProof<EF> {
    /// Per grid point g, (v_{p,g}, v_{q,g}) -- the inner fractional sum claims.
    /// Length is `n_grid_size = 2^{n_grid}`. Empty when evals is empty.
    pub grid_claims: Vec<(EF, EF)>,
    /// Per GKR layer j > 0, per grid point g, the claims for
    /// p_j(0, rho), p_j(1, rho), q_j(0, rho), q_j(1, rho).
    /// Outer length is `total_rounds_block`, inner length is `n_grid_size`.
    pub claims_per_layer: Vec<Vec<GkrLayerClaims>>,
    /// Sumcheck polynomials for each layer j >= 2, for each sumcheck round,
    /// given by their evaluations on {1, 2, 3}.
    /// Layer j (1-indexed, starting from 2) has j-1 sumcheck sub-rounds.
    pub sumcheck_polys: Vec<Vec<[EF; 3]>>,
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

/// Runs the data-parallel fractional sumcheck protocol using GKR layered circuit.
///
/// The evaluations `evals` has length `2^{total_rounds}` where
/// `total_rounds = l_skip + n_logup`. The parameter `n_grid` specifies the
/// grid dimension: the evaluations are split into `n_grid_size = 2^{n_grid}` blocks
/// of size `block_size = 2^{total_rounds_block}` where
/// `total_rounds_block = total_rounds - n_grid`.
///
/// Each grid point gets its own independent segment tree and GKR claims.
/// The sumcheck batches across all grid points using weight vectors initialized
/// from interleaved powers of a single challenge γ: ω_p[g] = γ^{2g}, ω_q[g] = γ^{2g+1}.
///
/// # Arguments
/// * `transcript` - The Fiat-Shamir transcript
/// * `evals` - list of `(p, q)` pairs of fractions in projective coordinates representing
///   evaluations on the hypercube `H_{total_rounds}`
/// * `assert_zero` - Whether to assert that the fractional sum is zero.
/// * `n_logup_grid` - The system parameter for grid dimension, used to specify "early stopping" for
///   GKR. The effective grid dimension `n_grid` will be set to `min(total_rounds, n_logup_grid)`.
///   When `n_grid = 0`, there is one grid point and behavior matches GKR without early stopping.
///
/// # Returns
/// The fractional sumcheck proof and the final random evaluation vector
/// xi = (xi_block, xi_grid) of length `total_rounds`.
#[instrument(level = "info", skip_all)]
pub fn fractional_sumcheck<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    evals: &[Frac<EF>],
    assert_zero: bool,
    n_logup_grid: usize,
) -> (FracSumcheckProof<EF>, Vec<EF>) {
    if evals.is_empty() {
        return (
            FracSumcheckProof {
                grid_claims: vec![],
                claims_per_layer: vec![],
                sumcheck_polys: vec![],
            },
            vec![],
        );
    }

    // total_rounds is total number of outer GKR rounds without early stopping
    let total_rounds = log2_strict_usize(evals.len());
    // We borrow the (grid, block) terminology from CUDA
    let n_grid = n_logup_grid.min(total_rounds);
    let grid_size = 1usize << n_grid;
    let n_block = total_rounds - n_grid;

    // Build per-grid-point segment trees.
    // Each tree has (2 << total_rounds_block) entries with index 0 unused.
    // segment tree: layer i=0,...,total_rounds starts at 2^i
    let mut tree_evals: Vec<Frac<EF>> = vec![Frac::default(); 2 << total_rounds];
    tree_evals[(1 << total_rounds)..].copy_from_slice(evals);
    // indices 0..2^n_grid will be unused
    for node_idx in (grid_size..(1 << total_rounds)).rev() {
        tree_evals[node_idx] = tree_evals[2 * node_idx] + tree_evals[2 * node_idx + 1];
    }

    // Compute per-grid-point root claims: grid_claims[g] = (tree_g[1].p, tree_g[1].q)
    let grid_claims: Vec<(EF, EF)> = tree_evals
        .iter()
        .skip(grid_size)
        .take(grid_size)
        .map(|frac| (frac.p, frac.q))
        .collect();

    // Check assert_zero condition: sum of p_g/q_g = 0 in projective coordinates.
    // This means the projective sum of all grid roots should have p = 0.
    #[cfg(debug_assertions)]
    if assert_zero {
        let total_sum = grid_claims
            .iter()
            .map(|&(p, q)| p * q.inverse())
            .sum::<EF>();
        assert_eq!(total_sum, EF::ZERO, "fractional sum zero-check failed");
    }

    // Observe grid claims in transcript.
    // When assert_zero=true and the first p is not observed because it can be derived as -q_0 *
    // (sum_{i>1} p_i / q_i).
    for (g, &(p_g, q_g)) in grid_claims.iter().enumerate() {
        if !assert_zero || g != 0 {
            transcript.observe_ext(p_g);
        }
        transcript.observe_ext(q_g);
    }

    if n_block == 0 {
        // No GKR rounds needed; the grid claims are the leaf evaluations themselves.
        // Sample xi_grid for the grid check
        let xi_grid: Vec<EF> = (0..n_grid).map(|_| transcript.sample_ext()).collect();
        return (
            FracSumcheckProof {
                grid_claims,
                claims_per_layer: vec![],
                sumcheck_polys: vec![],
            },
            xi_grid, // xi_block is empty, so xi = xi_grid
        );
    }
    // claims_per_layer[n_block][n_grid]
    let mut claims_per_layer: Vec<Vec<GkrLayerClaims>> = Vec::with_capacity(n_block);
    let mut sumcheck_polys: Vec<Vec<[EF; 3]>> = Vec::with_capacity(n_block.saturating_sub(1));

    // Process each GKR round: `j = round + 1` goes from `1, ..., n_block`

    // Round `j = 1` is special since "sumcheck" is directly checked by verifier
    let layer_1_claims: Vec<GkrLayerClaims> = (grid_size..2 * grid_size)
        .map(|g_idx| {
            let p_xi_0 = tree_evals[2 * g_idx].p;
            let q_xi_0 = tree_evals[2 * g_idx].q;
            let p_xi_1 = tree_evals[2 * g_idx + 1].p;
            let q_xi_1 = tree_evals[2 * g_idx + 1].q;

            transcript.observe_ext(p_xi_0);
            transcript.observe_ext(q_xi_0);
            transcript.observe_ext(p_xi_1);
            transcript.observe_ext(q_xi_1);

            GkrLayerClaims {
                p_xi_0,
                q_xi_0,
                p_xi_1,
                q_xi_1,
            }
        })
        .collect();
    claims_per_layer.push(layer_1_claims);
    // Sample mu_1 for reduction to single evaluation point
    let mu_1 = transcript.sample_ext();
    debug!(gkr_round = 0, mu = %mu_1);
    // ξ^{(j-1)}
    let mut xi_prev = vec![mu_1];

    for round in 1..n_block {
        // Number of hypercube points per grid point at this layer
        let eval_size = 1 << round;
        // We apply batch sumcheck to the polynomials
        // ```text
        // \eq(ξ^{(j-1)}, Y) (\hat p_j(0, Y, g) \hat q_j(1, Y, g) + \hat p_j(1, Y, g) \hat q_j(0, Y, g))
        // \eq(ξ^{(j-1)}, Y) (\hat q_j(0, Y, g) \hat q_j(1, Y, g))
        // ```
        // over all g in 0..grid_size
        // Note: these are polynomials of degree 3 in each Y_i coordinate.

        // Sample λ_j for batching
        let lambda = transcript.sample_ext();
        debug!(gkr_round = round, %lambda);

        // Build columns: for each grid point g, 4 columns (p_j_g_0, q_j_g_0, p_j_g_1, q_j_g_1).
        // Total pq columns: 4 * n_grid_size. Plus 1 eq_xis column.
        let num_pq_cols = 4 * grid_size;
        let mut pq_j_evals = EF::zero_vec(num_pq_cols * eval_size);
        for g in 0..grid_size {
            let segment = &tree_evals[(grid_size + g) << (round + 1)..][..2 * eval_size];
            let col_base = 4 * g;
            for x in 0..eval_size {
                pq_j_evals[(col_base) * eval_size + x] = segment[2 * x].p;
                pq_j_evals[(col_base + 1) * eval_size + x] = segment[2 * x].q;
                pq_j_evals[(col_base + 2) * eval_size + x] = segment[2 * x + 1].p;
                pq_j_evals[(col_base + 3) * eval_size + x] = segment[2 * x + 1].q;
            }
        }
        let mut pq_j_evals = ColMajorMatrix::new(pq_j_evals, num_pq_cols);
        let mut eq_xis = ColMajorMatrix::new(evals_eq_hypercube(&xi_prev), 1);

        // Batch sumcheck where the round polynomials are evaluated at {1, 2, 3}
        let (round_polys_eval, rho) = {
            let n = round;
            let mut round_polys_eval = Vec::with_capacity(n);
            let mut r_vec = Vec::with_capacity(n);

            for sumcheck_round in 0..n {
                // The combined sumcheck polynomial sums over all grid points:
                //   s(X) = sum_g [eq(xi_prev, Y) * (
                //     lambda^{2*g} * (p_0g * q_1g + p_1g * q_0g)
                //     + lambda^{2*g+1} * (q_0g * q_1g)
                //   )]
                // This is degree 3 in each Y_i (eq is degree 1, cross-term is degree 2).
                let [s_evals] = sumcheck_round_poly_evals(
                    n - sumcheck_round,
                    3,
                    &[eq_xis.as_view(), pq_j_evals.as_view()],
                    |_x, _y, row| {
                        let eq_xi = row[0][0];
                        let pq_row = row[1].as_slice();
                        let mut val = EF::ZERO;
                        let mut lambda_pow = EF::ONE;
                        for g in 0..grid_size {
                            let base = 4 * g;
                            let p_j0 = pq_row[base];
                            let q_j0 = pq_row[base + 1];
                            let p_j1 = pq_row[base + 2];
                            let q_j1 = pq_row[base + 3];
                            let p_cross = p_j0 * q_j1 + p_j1 * q_j0;
                            let q_cross = q_j0 * q_j1;
                            let lambda_p = lambda_pow;
                            lambda_pow *= lambda;
                            val += lambda_p * p_cross + lambda_pow * q_cross;
                            lambda_pow *= lambda;
                        }
                        [eq_xi * val]
                    },
                );
                let s_evals: [EF; 3] = s_evals.try_into().unwrap();
                for &eval in &s_evals {
                    transcript.observe_ext(eval);
                }
                round_polys_eval.push(s_evals);

                let r_round = transcript.sample_ext();
                pq_j_evals = fold_mle_evals(pq_j_evals, r_round);
                eq_xis = fold_mle_evals(eq_xis, r_round);
                r_vec.push(r_round);
                debug!(gkr_round = round, %sumcheck_round, %r_round);
            }
            (round_polys_eval, r_vec)
        };

        // Read off per-grid claims from folded columns
        let layer_claims: Vec<GkrLayerClaims> = (0..grid_size)
            .map(|g| {
                let base = 4 * g;
                GkrLayerClaims {
                    p_xi_0: pq_j_evals.column(base)[0],
                    q_xi_0: pq_j_evals.column(base + 1)[0],
                    p_xi_1: pq_j_evals.column(base + 2)[0],
                    q_xi_1: pq_j_evals.column(base + 3)[0],
                }
            })
            .collect();

        // Observe per-grid claims
        for claims in &layer_claims {
            transcript.observe_ext(claims.p_xi_0);
            transcript.observe_ext(claims.q_xi_0);
            transcript.observe_ext(claims.p_xi_1);
            transcript.observe_ext(claims.q_xi_1);
        }
        claims_per_layer.push(layer_claims);

        // Sample mu for reduction to single evaluation point
        let mu = transcript.sample_ext();
        debug!(gkr_round = round, %mu);

        // Update ξ^{(j)} = (μ_j, ρ^{(j-1)})
        xi_prev = [vec![mu], rho].concat();

        sumcheck_polys.push(round_polys_eval);
    }

    // Sample xi_grid for the grid check
    let xi_grid: Vec<EF> = (0..n_grid).map(|_| transcript.sample_ext()).collect();
    // Full xi = (xi_block, xi_grid)
    xi_prev.extend(xi_grid);

    (
        FracSumcheckProof {
            grid_claims,
            claims_per_layer,
            sumcheck_polys,
        },
        xi_prev,
    )
}
