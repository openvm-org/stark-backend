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
/// The evaluations `evals` has length `2^{total_rounds_full}` where
/// `total_rounds_full = l_skip + n_logup`. The parameter `n_grid` specifies the
/// grid dimension: the evaluations are split into `n_grid_size = 2^{n_grid}` blocks
/// of size `block_size = 2^{total_rounds_block}` where
/// `total_rounds_block = total_rounds_full - n_grid`.
///
/// Each grid point gets its own independent segment tree and GKR claims.
/// The sumcheck batches across all grid points using weight vectors initialized
/// from interleaved powers of a single challenge γ: ω_p[g] = γ^{2g}, ω_q[g] = γ^{2g+1}.
///
/// # Arguments
/// * `transcript` - The Fiat-Shamir transcript
/// * `evals` - list of `(p, q)` pairs of fractions in projective coordinates representing
///   evaluations on the hypercube `H_{total_rounds_full}`
/// * `assert_zero` - Whether to assert that the fractional sum is zero.
/// * `n_grid` - The effective grid dimension. The caller should compute
///   `min(n_logup, n_logup_grid)`. When 0, there is one grid point and
///   behavior matches the single-instance case.
///
/// # Returns
/// The fractional sumcheck proof and the final random evaluation vector
/// xi = (xi_block, xi_grid) of length `total_rounds_full = total_rounds_block + n_grid`.
/// When `n_grid = 0`, xi_grid is empty and xi = xi_block.
#[instrument(level = "info", skip_all)]
pub fn fractional_sumcheck<TS: FiatShamirTranscript>(
    transcript: &mut TS,
    evals: &[Frac<EF>],
    assert_zero: bool,
    n_grid: usize,
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

    let total_rounds_full = log2_strict_usize(evals.len());
    debug_assert!(
        n_grid <= total_rounds_full,
        "n_grid ({n_grid}) must be <= total_rounds_full ({total_rounds_full})"
    );
    let n_grid_size = 1usize << n_grid;
    let total_rounds_block = total_rounds_full - n_grid;
    let block_size = 1usize << total_rounds_block;

    // Build per-grid-point segment trees.
    // Each tree has (2 << total_rounds_block) entries with index 0 unused.
    let tree_size = 2 << total_rounds_block;
    let mut trees: Vec<Vec<Frac<EF>>> = Vec::with_capacity(n_grid_size);
    for g in 0..n_grid_size {
        let mut tree: Vec<Frac<EF>> = vec![Frac::default(); tree_size];
        let block_start = g * block_size;
        tree[(1 << total_rounds_block)..].copy_from_slice(
            &evals[block_start..block_start + block_size],
        );
        for node_idx in (1..(1 << total_rounds_block)).rev() {
            tree[node_idx] = tree[2 * node_idx] + tree[2 * node_idx + 1];
        }
        trees.push(tree);
    }

    // Compute per-grid-point root claims: grid_claims[g] = (tree_g[1].p, tree_g[1].q)
    let grid_claims: Vec<(EF, EF)> =
        trees.iter().map(|tree| (tree[1].p, tree[1].q)).collect();

    // Check assert_zero condition: sum of p_g/q_g = 0 in projective coordinates.
    // This means the projective sum of all grid roots should have p = 0.
    if assert_zero {
        let total_sum = trees
            .iter()
            .map(|tree| tree[1])
            .reduce(|a, b| a + b)
            .unwrap();
        assert_eq!(total_sum.p, EF::ZERO, "fractional sum zero-check failed");
    }

    // Observe grid claims in transcript.
    // When assert_zero=true and there is only one grid point, p=0 is implicit
    // and not observed. When there are multiple grid points, individual p_g values
    // are non-zero and must be observed to bind them before xi_grid is sampled.
    for &(p_g, q_g) in &grid_claims {
        if !assert_zero || n_grid_size > 1 {
            transcript.observe_ext(p_g);
        }
        transcript.observe_ext(q_g);
    }

    if total_rounds_block == 0 {
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

    let mut claims_per_layer: Vec<Vec<GkrLayerClaims>> =
        Vec::with_capacity(total_rounds_block);
    let mut sumcheck_polys: Vec<Vec<[EF; 3]>> =
        Vec::with_capacity(total_rounds_block.saturating_sub(1));

    // Vectorized GKR Setup: sample gamma and initialize weight vectors using
    // interleaved powers: omega_p[g] = gamma^{2g}, omega_q[g] = gamma^{2g+1}.
    // This ensures omega_p[0]=1 != omega_q[0]=gamma, so p and q are always
    // independently bound (avoiding the soundness issue of equal weights at g=0).
    let gamma: EF = transcript.sample_ext();
    debug!(%gamma);
    let gamma_sq = gamma * gamma;
    let mut omega_p: Vec<EF> = {
        let mut v = Vec::with_capacity(n_grid_size);
        let mut cur = EF::ONE; // gamma^0
        for _ in 0..n_grid_size {
            v.push(cur);
            cur *= gamma_sq; // gamma^0, gamma^2, gamma^4, ...
        }
        v
    };
    let mut omega_q: Vec<EF> = {
        let mut v = Vec::with_capacity(n_grid_size);
        let mut cur = gamma; // gamma^1
        for _ in 0..n_grid_size {
            v.push(cur);
            cur *= gamma_sq; // gamma^1, gamma^3, gamma^5, ...
        }
        v
    };

    // =========================================================================
    // Round j=1 (special: 0 sumcheck sub-rounds)
    // =========================================================================
    // For each grid point g, layer 1 claims come from tree nodes 2 and 3.
    let layer_1_claims: Vec<GkrLayerClaims> = trees
        .iter()
        .map(|tree| GkrLayerClaims {
            p_xi_0: tree[2].p,
            q_xi_0: tree[2].q,
            p_xi_1: tree[3].p,
            q_xi_1: tree[3].q,
        })
        .collect();

    // Observe all layer-1 claims
    for claims in &layer_1_claims {
        transcript.observe_ext(claims.p_xi_0);
        transcript.observe_ext(claims.q_xi_0);
        transcript.observe_ext(claims.p_xi_1);
        transcript.observe_ext(claims.q_xi_1);
    }
    claims_per_layer.push(layer_1_claims);

    // Sample mu_1 for reduction to single evaluation point
    let mu_1 = transcript.sample_ext();
    debug!(gkr_round = 0, mu = %mu_1);

    // Weight update for round 1
    for (g, claims) in claims_per_layer[0].iter().enumerate() {
        let cp0 = omega_p[g] * claims.q_xi_1;
        let cp1 = omega_p[g] * claims.q_xi_0;
        let cq0 = omega_p[g] * claims.p_xi_1 + omega_q[g] * claims.q_xi_1;
        let cq1 = omega_p[g] * claims.p_xi_0 + omega_q[g] * claims.q_xi_0;
        omega_p[g] = (EF::ONE - mu_1) * cp0 + mu_1 * cp1;
        omega_q[g] = (EF::ONE - mu_1) * cq0 + mu_1 * cq1;
    }

    let mut xi_prev = vec![mu_1];

    // =========================================================================
    // Rounds j=2,...,total_rounds_block (with sumcheck)
    // =========================================================================
    for round in 1..total_rounds_block {
        // Number of hypercube points per grid point at this layer
        let eval_size = 1 << round;

        // Build columns: for each grid point g, 4 columns (p_j_g_0, q_j_g_0, p_j_g_1, q_j_g_1).
        // Total pq columns: 4 * n_grid_size. Plus 1 eq_xis column.
        let num_pq_cols = 4 * n_grid_size;
        let mut pq_j_evals = EF::zero_vec(num_pq_cols * eval_size);
        for (g, tree) in trees.iter().enumerate() {
            let segment = &tree[2 * eval_size..4 * eval_size];
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
                //     omega_p[g] * (p_g_0 * q_g_1 + p_g_1 * q_g_0)
                //     + omega_q[g] * (q_g_0 * q_g_1)
                //   )]
                // This is degree 3 in each Y_i (eq is degree 1, cross-term is degree 2).
                let n_gs = n_grid_size; // capture for closure
                let op = &omega_p;
                let oq = &omega_q;
                let [s_evals] = sumcheck_round_poly_evals(
                    n - sumcheck_round,
                    3,
                    &[eq_xis.as_view(), pq_j_evals.as_view()],
                    |_x, _y, row| {
                        let eq_xi = row[0][0];
                        let pq_row = row[1].as_slice();
                        let mut val = EF::ZERO;
                        for g in 0..n_gs {
                            let base = 4 * g;
                            let p_j0 = pq_row[base];
                            let q_j0 = pq_row[base + 1];
                            let p_j1 = pq_row[base + 2];
                            let q_j1 = pq_row[base + 3];
                            let p_cross = p_j0 * q_j1 + p_j1 * q_j0;
                            let q_cross = q_j0 * q_j1;
                            val += op[g] * p_cross + oq[g] * q_cross;
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
        let layer_claims: Vec<GkrLayerClaims> = (0..n_grid_size)
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

        // Weight update
        let layer_ref = claims_per_layer.last().unwrap();
        for (g, claims) in layer_ref.iter().enumerate() {
            let cp0 = omega_p[g] * claims.q_xi_1;
            let cp1 = omega_p[g] * claims.q_xi_0;
            let cq0 = omega_p[g] * claims.p_xi_1 + omega_q[g] * claims.q_xi_1;
            let cq1 = omega_p[g] * claims.p_xi_0 + omega_q[g] * claims.q_xi_0;
            omega_p[g] = (EF::ONE - mu) * cp0 + mu * cp1;
            omega_q[g] = (EF::ONE - mu) * cq0 + mu * cq1;
        }

        // Update xi_prev = (mu, rho)
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
