//! Soundness calculator for the SWIRL proof system.
//!
//! The SWIRL proof system consists of the following components:
//! 1. LogUp GKR - Fractional sumcheck for interaction constraints
//! 2. ZeroCheck - Batched constraint verification across AIRs
//! 3. Stacked Reduction - Reduces trace evaluations to stacked polynomial evaluations
//! 4. WHIR - Polynomial commitment opening via FRI-like folding
//!
//! Each component contributes to the overall soundness error, and the total security
//! is the minimum across all components.

use std::f64;

use crate::{
    config::{ProximityRegime, SystemParams, WhirConfig},
    WhirProximityStrategy,
};

#[derive(Clone, Debug)]
pub struct SoundnessCalculator {
    /// Security bits from LogUp α/β sampling and PoW.
    pub logup_bits: f64,
    /// Security bits from GKR sumcheck rounds.
    pub gkr_sumcheck_bits: f64,
    /// Security bits from GKR μ/λ batching per layer.
    pub gkr_batching_bits: f64,
    /// Security bits from ZeroCheck sumcheck (univariate and multilinear rounds).
    pub zerocheck_sumcheck_bits: f64,
    /// Security bits from constraint batching (λ and μ via Schwartz-Zippel).
    pub constraint_batching_bits: f64,
    /// Security bits from stacked reduction sumcheck.
    pub stacked_reduction_bits: f64,
    /// Security bits from WHIR (minimum across all rounds and error sources).
    pub whir_bits: f64,
    /// Detailed WHIR soundness breakdown.
    pub whir_details: WhirSoundnessCalculator,
    /// Total security bits (minimum of all components).
    pub total_bits: f64,
}

/// WHIR soundness breakdown by error source.
#[derive(Clone, Debug)]
pub struct WhirSoundnessCalculator {
    /// Security bits from μ batching (initial polynomial batching).
    pub mu_batching_bits: f64,
    /// Minimum round-by-round security bits across folding rounds, i.e. `ε_fold`.
    pub fold_rbr_bits: f64,
    /// Security bits from proximity gaps (folding soundness).
    pub proximity_gaps_bits: f64,
    /// Security bits from sumcheck within WHIR rounds.
    pub sumcheck_bits: f64,
    /// Security bits from out-of-domain sampling.
    pub ood_rbr_bits: f64,
    /// Minimum round-by-round security bits across shift/final rounds, i.e. `ε_shift` / `ε_fin`.
    pub shift_rbr_bits: f64,
    /// Security bits from query sampling.
    pub query_bits: f64,
    /// Security bits from γ batching (combining query and OOD claims).
    pub gamma_batching_bits: f64,
}

#[derive(Clone, Debug)]
pub struct ProximityGapSecurity {
    pub log2_err: f64,
    pub log2_list_size: f64,
}

impl SoundnessCalculator {
    /// Calculates soundness for the given system parameters.
    ///
    /// # Arguments
    /// * `params` - System parameters including WHIR config and LogUp parameters.
    /// * `challenge_field_bits` - Bits in the challenge field. For BabyBear4: ~124 bits.
    /// * `max_num_constraints_per_air` - Maximum constraints in any single AIR.
    /// * `num_airs` - Number of AIRs being batched.
    /// * `max_constraint_degree` - Maximum degree of any constraint polynomial.
    /// * `max_log_trace_height` - Maximum log₂(trace height) across all AIRs.
    /// * `num_trace_columns` - Total columns batched in stacked reduction.
    /// * `num_stacked_columns` - Total columns across all commitments (for WHIR μ batching).
    /// * `n_logup` - GKR depth (log₂ of total interactions), or 0 if no interactions.
    /// * `proximity_regime` - Unique decoding or other regimes (for WHIR-related calculations).
    #[allow(clippy::too_many_arguments)]
    pub fn calculate(
        params: &SystemParams,
        challenge_field_bits: f64,
        max_num_constraints_per_air: usize,
        num_airs: usize,
        max_constraint_degree: usize,
        max_log_trace_height: usize,
        num_trace_columns: usize,
        num_stacked_columns: usize,
        n_logup: usize,
    ) -> Self {
        let init_prox_gap = Self::whir_proximity_gap_security(
            params.whir.proximity.initial_round(),
            challenge_field_bits,
            params.log_stacked_height(),
            params.log_blowup,
            num_stacked_columns,
        );
        // log2_list_size is log2(L_{PCS}) where L_{PCS} is the list size with respect to the
        // proximity radius of the _initial_ WHIR round.
        let log2_list_size = init_prox_gap.log2_list_size;
        let logup_bits =
            Self::calculate_logup_soundness(params, challenge_field_bits, log2_list_size);

        let gkr_sumcheck_bits =
            Self::calculate_gkr_sumcheck_soundness(challenge_field_bits, params.l_skip, n_logup);

        let gkr_batching_bits =
            Self::calculate_gkr_batching_soundness(challenge_field_bits, params.l_skip, n_logup);

        let zerocheck_sumcheck_bits = Self::calculate_zerocheck_sumcheck_soundness(
            challenge_field_bits,
            max_constraint_degree,
            params.l_skip,
            max_log_trace_height,
            log2_list_size,
        );

        let constraint_batching_bits = Self::calculate_constraint_batching_soundness(
            challenge_field_bits,
            max_num_constraints_per_air,
            num_airs,
            log2_list_size,
        );

        let stacked_reduction_bits = Self::calculate_stacked_reduction_soundness(
            challenge_field_bits,
            num_trace_columns,
            params.l_skip,
            params.n_stack,
            log2_list_size,
        );

        let (whir_bits, whir_details) = Self::calculate_whir_soundness(
            params,
            challenge_field_bits,
            num_stacked_columns,
            params.whir.proximity,
        );

        let total_bits = logup_bits
            .min(gkr_sumcheck_bits)
            .min(gkr_batching_bits)
            .min(zerocheck_sumcheck_bits)
            .min(constraint_batching_bits)
            .min(stacked_reduction_bits)
            .min(whir_bits);

        Self {
            logup_bits,
            gkr_sumcheck_bits,
            gkr_batching_bits,
            zerocheck_sumcheck_bits,
            constraint_batching_bits,
            stacked_reduction_bits,
            whir_bits,
            whir_details,
            total_bits,
        }
    }

    /// LogUp soundness from α/β sampling.
    ///
    /// - α: Random evaluation point to test whether Σ p(y)/q(y) = 0. If interactions are
    ///   unbalanced, the sum is a non-zero rational function with a bounded number of roots. By
    ///   Schwartz-Zippel, a random α detects this with high probability.
    /// - β: Random challenge for compressing interaction messages into field elements. Degeneracy
    ///   would allow distinct message tuples to collide.
    ///
    /// Security = |F_ext| - log₂(2 × max_interaction_count) - log_max_message_length + pow_bits
    ///
    /// Reference: Section 4 of docs/Soundness_of_Interactions_via_LogUp.pdf
    fn calculate_logup_soundness(
        params: &SystemParams,
        challenge_field_bits: f64,
        log2_list_size: f64,
    ) -> f64 {
        let logup = &params.logup;
        let max_interaction_count_bits = (2.0 * logup.max_interaction_count as f64).log2();

        log2_list_size + challenge_field_bits
            - max_interaction_count_bits
            - logup.log_max_message_length as f64
            + logup.pow_bits as f64
    }

    /// GKR sumcheck soundness (per-round).
    ///
    /// The GKR protocol has a triangular sumcheck structure where round j has j sub-rounds.
    /// Each sub-round uses degree-3 interpolation, giving per-round error = 3 / |F_ext|.
    ///
    /// Security is determined by the worst round: |F_ext| - log₂(3)
    fn calculate_gkr_sumcheck_soundness(
        challenge_field_bits: f64,
        l_skip: usize,
        n_logup: usize,
    ) -> f64 {
        let total_rounds = l_skip + n_logup;
        assert!(total_rounds >= 1, "GKR requires at least 1 round");

        // Each sub-round has degree 3; security = field_bits - log2(degree)
        let degree_per_subround = 3;
        challenge_field_bits - (degree_per_subround as f64).log2()
    }

    /// GKR batching soundness from μ and λ challenges per layer.
    ///
    /// Each layer samples:
    /// - μ: Reduces four evaluation claims to two via linear interpolation (degree 1)
    /// - λ: Batches numerator and denominator claims (degree 1)
    ///
    /// Per-round security = |F_ext| - log₂(degree) = |F_ext| - log₂(1) = |F_ext|
    fn calculate_gkr_batching_soundness(
        challenge_field_bits: f64,
        _l_skip: usize,
        _n_logup: usize,
    ) -> f64 {
        // Each μ/λ challenge is a degree-1 polynomial test (linear interpolation)
        let degree = 1;
        challenge_field_bits - (degree as f64).log2()
    }

    /// ZeroCheck sumcheck soundness (per-round).
    ///
    /// Two phases with different per-round degrees:
    ///
    /// 1. **Univariate round** over coset domain (size 2^l_skip):
    ///    - Degree: (max_constraint_degree + 1) × (2^l_skip - 1)
    ///
    /// 2. **Multilinear rounds** (n_max = max_log_trace_height - l_skip):
    ///    - Per-round degree: max_constraint_degree + 1
    ///
    /// 3. **Polynomial identity testing at r**: After sumcheck completes, trace polynomials are
    ///    evaluated at random r. If the prover's trace differs from the committed trace, this is
    ///    caught by Schwartz-Zippel. Trace polynomials have deg_Z ≤ 2^l_skip - 1 and deg_{X_i} ≤ 1.
    ///    Error ≤ (2^l_skip - 1 + n_max) / |F_ext|
    fn calculate_zerocheck_sumcheck_soundness(
        challenge_field_bits: f64,
        max_constraint_degree: usize,
        l_skip: usize,
        max_log_trace_height: usize,
        log2_list_size: f64,
    ) -> f64 {
        let univariate_degree = (max_constraint_degree + 1) * ((1 << l_skip) - 1);
        let multilinear_degree = max_constraint_degree + 1;

        let worst_degree = univariate_degree.max(multilinear_degree);
        let sumcheck_bits = challenge_field_bits - (worst_degree as f64).log2();

        // Polynomial identity testing: trace polynomial has deg_Z ≤ 2^l_skip - 1, deg_{X_i} ≤ 1
        // n_max = max_log_trace_height - l_skip multilinear variables
        let n_max = max_log_trace_height.saturating_sub(l_skip);
        let poly_degree_sum = (1 << l_skip) - 1 + n_max;
        let poly_identity_bits = challenge_field_bits - (poly_degree_sum as f64).log2();

        log2_list_size + sumcheck_bits.min(poly_identity_bits)
    }

    /// Constraint batching soundness via Schwartz-Zippel.
    ///
    /// Two batching levels:
    /// - λ: Within each AIR, batching n constraints. Error ≤ n / |F_ext|
    /// - μ: Across AIRs, batching 3k sum claims (ZeroCheck + LogUp numerator + LogUp denominator
    ///   per AIR). Error ≤ 3k / |F_ext|
    fn calculate_constraint_batching_soundness(
        challenge_field_bits: f64,
        max_num_constraints_per_air: usize,
        num_airs: usize,
        log2_list_size: f64,
    ) -> f64 {
        let lambda_batching_bits =
            challenge_field_bits - (max_num_constraints_per_air as f64).log2();
        // Each AIR contributes 3 sum claims to the batch sumcheck:
        // 1. ZeroCheck (constraint satisfaction)
        // 2. LogUp numerator (p̂(ξ) input layer)
        // 3. LogUp denominator (q̂(ξ) input layer)
        let mu_batching_bits = challenge_field_bits - (3.0 * num_airs as f64).log2();

        log2_list_size + lambda_batching_bits.min(mu_batching_bits)
    }

    /// Stacked reduction soundness.
    ///
    /// Reduces trace evaluations at point r to stacked polynomial evaluations at point u.
    ///
    /// Note: Trace heights do not appear directly; polynomial degrees are determined by the
    /// stacking structure (l_skip, n_stack), not individual trace heights.
    ///
    /// Error sources:
    /// 1. **λ batching**: 2 claims per column (T(r) and T_rot(r)). Error = 2n / |F_ext|
    /// 2. **Univariate round**: Degree 2×(2^l_skip - 1). Per-round error = degree / |F_ext|
    /// 3. **Multilinear rounds**: n_stack rounds, each with degree 2. Per-round error = 2 / |F_ext|
    fn calculate_stacked_reduction_soundness(
        challenge_field_bits: f64,
        num_trace_columns: usize,
        l_skip: usize,
        _n_stack: usize,
        log2_list_size: f64,
    ) -> f64 {
        let batching_bits = challenge_field_bits - (2.0 * num_trace_columns as f64).log2();

        let univariate_degree = 2 * ((1 << l_skip) - 1);
        let univariate_bits = challenge_field_bits - (univariate_degree as f64).log2();

        // Degree 2 per round => log2(2) = 1.
        let multilinear_bits = challenge_field_bits - 1.0;

        log2_list_size + batching_bits.min(univariate_bits).min(multilinear_bits)
    }

    /// WHIR soundness analysis based on the WHIR paper (ePrint 2024/1586).
    ///
    /// Error sources (formulas depend on the proximity regime):
    /// 1. **Fold error** (sumcheck + proximity gap per sub-round)
    /// 2. **OOD error** (non-final rounds)
    /// 3. **Shift/final error** (query sampling + γ batching)
    /// 5. **Initial μ batching**
    fn calculate_whir_soundness(
        params: &SystemParams,
        challenge_field_bits: f64,
        num_stacked_columns: usize,
        proximity: WhirProximityStrategy,
    ) -> (f64, WhirSoundnessCalculator) {
        let whir = &params.whir;
        let k_whir = whir.k;
        let log_stacked_height = params.log_stacked_height();
        let num_whir_rounds = whir.rounds.len();

        let mut min_query_bits = f64::INFINITY;
        let mut min_prox_gaps_bits = f64::INFINITY;
        let mut min_sumcheck_bits = f64::INFINITY;
        let mut min_ood_bits = f64::INFINITY;
        let mut min_gamma_batching_bits = f64::INFINITY;
        let mut min_fold_rbr_bits = f64::INFINITY;
        let mut min_shift_rbr_bits = f64::INFINITY;

        assert!(
            num_stacked_columns >= 2,
            "WHIR requires at least 2 stacked columns for μ batching"
        );
        let mu_security = Self::whir_proximity_gap_security(
            proximity.initial_round(),
            challenge_field_bits,
            log_stacked_height,
            params.log_blowup,
            num_stacked_columns,
        );
        let mu_batching_bits = mu_security.log2_err + whir.mu_pow_bits as f64;
        let mut min_rbr_bits = mu_batching_bits;

        let mut log_inv_rate = params.log_blowup;
        let mut current_log_degree = log_stacked_height;

        for (round, round_config) in whir.rounds.iter().enumerate() {
            let proximity_regime = proximity.in_round(round);
            let is_final_round = round == num_whir_rounds - 1;
            let next_rate = log_inv_rate + (k_whir - 1);
            // log2(list size) only depends on proximity regime, will not change depending on the
            // sub-round
            let mut log2_list_size: Option<f64> = None;

            for _ in 0..k_whir {
                current_log_degree -= 1;

                let prox_gaps = Self::whir_proximity_gap_security(
                    proximity_regime,
                    challenge_field_bits,
                    current_log_degree,
                    log_inv_rate,
                    2,
                );
                if let Some(l2) = log2_list_size.as_ref() {
                    debug_assert!((*l2 - prox_gaps.log2_list_size).abs() < 1e-6);
                } else {
                    log2_list_size = Some(prox_gaps.log2_list_size);
                }
                let prox_gaps_bits = prox_gaps.log2_err + whir.folding_pow_bits as f64;
                min_prox_gaps_bits = min_prox_gaps_bits.min(prox_gaps_bits);

                let sumcheck_bits = Self::whir_sumcheck_security(
                    challenge_field_bits,
                    whir.folding_pow_bits,
                    log2_list_size.unwrap(),
                );
                min_sumcheck_bits = min_sumcheck_bits.min(sumcheck_bits);

                // Theorem 5.2: ε_fold = d * ℓ_{i,s-1} / |F| + err*.
                let fold_rbr_bits = Self::combine_security_bits(sumcheck_bits, prox_gaps_bits);
                min_fold_rbr_bits = min_fold_rbr_bits.min(fold_rbr_bits);
                min_rbr_bits = min_rbr_bits.min(fold_rbr_bits);
            }

            // Query error (all rounds), protected by query_phase_pow_bits.
            let query_bits = proximity_regime
                .whir_query_security_bits(round_config.num_queries, log_inv_rate)
                + whir.query_phase_pow_bits as f64;
            min_query_bits = min_query_bits.min(query_bits);

            // ε_shift and ε_out reference m_i, ℓ_{i,0} where `i = round + 1` refers to the *next*
            // round.
            let next_log2_list_size = Self::whir_proximity_gap_security(
                proximity.in_round(round + 1),
                challenge_field_bits,
                current_log_degree,
                next_rate,
                2,
            )
            .log2_list_size;
            // In-domain γ batching (not protected by PoW; Merkle proofs are observed before γ).
            // NOTE[jpw] For now we use the original paper where this is fixed to 1. <https://github.com/WizardOfMenlo/whir/blob/cf1599b56ff50e09142ebe6d2e2fbd86875c9986/src/whir/parameters.rs#L373> now varies this to increase security in LDR.
            const NUM_OOD_SAMPLES: usize = 1;
            let batch_size = round_config.num_queries + NUM_OOD_SAMPLES;
            debug_assert!(batch_size > 0);
            let gamma_batching_bits = Self::whir_gamma_batching_security(
                challenge_field_bits,
                batch_size,
                next_log2_list_size,
            );
            min_gamma_batching_bits = min_gamma_batching_bits.min(gamma_batching_bits);

            // Theorem 5.2 / Claim 5.4: ε_shift = (1 - δ)^t + ℓ_{i,0} * (t + 1) / |F|. The
            // implementation keeps the same additive structure, with the final round
            // batching only the query claims.
            let shift_rbr_bits = Self::combine_security_bits(query_bits, gamma_batching_bits);
            min_shift_rbr_bits = min_shift_rbr_bits.min(shift_rbr_bits);
            min_rbr_bits = min_rbr_bits.min(shift_rbr_bits);

            if !is_final_round {
                // OOD error (not protected by PoW; sampled after commitment observed).
                // This is OOD sample on f_i for the *next* round `i = round + 1` after folding. So
                // `m_i = current_log_degree` (with the present round `round`'s foldings)
                let ood_bits = Self::whir_ood_security(
                    next_log2_list_size,
                    challenge_field_bits,
                    current_log_degree,
                );
                min_ood_bits = min_ood_bits.min(ood_bits);
                min_rbr_bits = min_rbr_bits.min(ood_bits);

                tracing::debug!(
                    "WHIR round {} | rate=2^-{} | queries={} | query={:.1} | prox_gaps={:.1} | sumcheck={:.1} | shift={:.1} | ood={:.1} | gamma={:.1}",
                    round, log_inv_rate, round_config.num_queries, query_bits,
                    min_prox_gaps_bits, min_sumcheck_bits, shift_rbr_bits, ood_bits,
                    min_gamma_batching_bits,
                );
            } else {
                tracing::debug!(
                    "WHIR round {} (final) | rate=2^-{} | queries={} | query={:.1} | prox_gaps={:.1} | sumcheck={:.1} | final={:.1} | gamma={:.1}",
                    round, log_inv_rate, round_config.num_queries, query_bits,
                    min_prox_gaps_bits, min_sumcheck_bits, shift_rbr_bits,
                    min_gamma_batching_bits,
                );
            }

            log_inv_rate = next_rate;
        }

        let details = WhirSoundnessCalculator {
            mu_batching_bits,
            fold_rbr_bits: min_fold_rbr_bits,
            ood_rbr_bits: min_ood_bits,
            shift_rbr_bits: min_shift_rbr_bits,
            // The following are part of above rbr error, but kept for detailed analysis
            query_bits: min_query_bits,
            proximity_gaps_bits: min_prox_gaps_bits,
            sumcheck_bits: min_sumcheck_bits,
            gamma_batching_bits: min_gamma_batching_bits,
        };

        let min_security = min_rbr_bits;

        (min_security, details)
    }

    /// Computes WHIR proximity gap security bits.
    ///
    /// Per WHIR paper: err*(C, ℓ, δ) = (ℓ - 1) · 2^m / (ρ · |F|)
    ///
    /// - `UniqueDecoding`: Security bits = |F_ext| - log₂(ℓ - 1) - log₂(degree) - log₂(1/rate)
    /// - `ListDecoding { m }`: Uses BCHKS25/TR25-169 Theorem 1.5 (contrapositive) to bound the "bad
    ///   z set" size by `a`, with Haböck's global extension introducing a linear factor `(ℓ - 1)`.
    pub fn whir_proximity_gap_security(
        proximity_regime: ProximityRegime,
        challenge_field_bits: f64,
        log_degree: usize,
        log_inv_rate: usize,
        batch_size: usize,
    ) -> ProximityGapSecurity {
        debug_assert!(batch_size > 1, "batch_size must be > 1 for err*");
        match proximity_regime {
            ProximityRegime::UniqueDecoding => {
                let log2_err = challenge_field_bits
                    - ((batch_size - 1) as f64).log2()
                    - log_degree as f64
                    - log_inv_rate as f64;
                ProximityGapSecurity {
                    log2_err,
                    log2_list_size: 0.0,
                }
            }
            ProximityRegime::ListDecoding { m } => {
                let (log2_a, log2_list_size) =
                    Self::log2_a_bound_bchks25(log_degree, log_inv_rate, m);
                // Haböck global mutual correlated agreement: error ∝ (ℓ - 1) * a / |F|.
                let log2_err = challenge_field_bits - ((batch_size - 1) as f64).log2() - log2_a;
                ProximityGapSecurity {
                    log2_err,
                    log2_list_size,
                }
            }
        }
    }

    /// Numerically stable `log2(2^x + 2^y)`.
    #[inline]
    fn log2_add(log2_x: f64, log2_y: f64) -> f64 {
        if log2_x.is_infinite() && log2_x.is_sign_positive() {
            return log2_x;
        }
        if log2_y.is_infinite() && log2_y.is_sign_positive() {
            return log2_y;
        }
        if log2_x.is_nan() || log2_y.is_nan() {
            return f64::NAN;
        }

        let (hi, lo) = if log2_x >= log2_y {
            (log2_x, log2_y)
        } else {
            (log2_y, log2_x)
        };
        let ratio = (lo - hi).exp2();
        hi + (1.0 + ratio).log2()
    }

    /// Combines two additive error terms `2^-a + 2^-b` into security bits `-log2(error)`.
    #[inline]
    fn combine_security_bits(bits_a: f64, bits_b: f64) -> f64 {
        if bits_a.is_infinite() && bits_a.is_sign_positive() {
            return bits_b;
        }
        if bits_b.is_infinite() && bits_b.is_sign_positive() {
            return bits_a;
        }
        if bits_a.is_nan() || bits_b.is_nan() {
            return f64::NAN;
        }

        -Self::log2_add(-bits_a, -bits_b)
    }

    #[inline]
    fn bchks25_log2_a_from_log2_degrees(
        log2_d_x: f64,
        log2_d_y: f64,
        log2_d_z: f64,
        log2_agreement_term: f64,
    ) -> f64 {
        // Equation (13): a > 2*D_X*D_Y^2*D_Z + agreement_term*D_Y
        let log2_term_poly = 1.0 + log2_d_x + 2.0 * log2_d_y + log2_d_z;
        let log2_term_gamma = log2_d_y + log2_agreement_term;
        Self::log2_add(log2_term_poly, log2_term_gamma)
    }

    /// Reference closed-form degrees from BCHKS25 Lemma 3.1, Equations (7), (8), (9).
    /// For `m < 3`, use `D_Z = max(D_Y, Equation (9) value)` as noted below Lemma 3.1.
    ///
    /// Returns (log2(D_X), log2(D_Y), log2(D_Z))
    fn bchks25_log2_degrees(
        log_degree: usize,
        log_inv_rate: usize,
        m: usize,
        _gamma: f64,
    ) -> (f64, f64, f64) {
        #[cfg(feature = "soundness-bchks25-optimized")]
        if let Some((degrees, _)) = bchks25_brute_force_params::bchks25_optimal_degrees_bruteforce(
            log_degree,
            log_inv_rate,
            m,
            _gamma,
        ) {
            debug_assert!(degrees.d_x > 0 && degrees.d_y > 0 && degrees.d_z > 0);
            return (
                (degrees.d_x as f64).log2(),
                (degrees.d_y as f64).log2(),
                (degrees.d_z as f64).log2(),
            );
        }

        Self::bchks25_reference_log2_degrees(log_degree, log_inv_rate, m)
    }

    fn bchks25_reference_log2_degrees(
        log_degree: usize,
        log_inv_rate: usize,
        m: usize,
    ) -> (f64, f64, f64) {
        let m_bar = m.max(1) as f64 + 0.5;
        let log2_m_bar = m_bar.log2();
        let log2_n = (log_degree + log_inv_rate) as f64;
        let log2_3 = 3.0_f64.log2();
        let log2_rho = -(log_inv_rate as f64);

        // D_X = (m + 1/2) * sqrt(k * n)
        // D_Y = (m + 1/2) * sqrt(n / k)
        // D_Z (Equation 9) = ((m + 1/2)^2 * n) / (3 * k)
        let log2_d_x = log2_m_bar + log2_n + 0.5 * log2_rho;
        let log2_d_y = log2_m_bar - 0.5 * log2_rho;
        let log2_d_z = 2.0 * log2_m_bar - log2_3 - log2_rho;
        let log2_d_z = log2_d_y.max(log2_d_z);

        (log2_d_x, log2_d_y, log2_d_z)
    }

    /// Computes `log2(a_bound)` from BCHKS25/TR25-169 Theorem 1.5 (contrapositive), where
    /// `a_bound = ceil(a).max(1)`.
    ///
    /// We use Lemma 3.1 and the bounds on `a` in terms of `D_X, D_Y, D_Z` from Section 3.2 and
    /// Equation (13). As noted in the paragraph after Lemma 3.1, the parameters chosen in the
    /// Lemma 3.1 statement are not optimal and chosen to provide cleaner expressions.
    /// When feature `soundness-bchks25-optimized` is enabled,
    /// we do a brute-force search in `bchks25_optimal_degrees_bruteforce` to find parameters that
    /// meet the conditions for the proof of Lemma 3.1 to be applied to find the polynomial `Q`.
    /// When the feature is not enabled, a closed-form Lemma 3.1 degree computation is used.
    ///
    /// Parameters are mapped as:
    /// - `num_variables = log_degree`
    /// - `rho = 2^{-log_inv_rate}`
    /// - `n = 2^{log_degree + log_inv_rate}`
    ///
    /// We set the theorem slack `η` from the provided multiplicity `m` as
    /// `η = sqrt(rho) / (2m)`, so that `m = ceil(sqrt(rho)/(2η))`.
    ///
    /// Returns `log2(a_bound), log2(list_size)`.
    fn log2_a_bound_bchks25(log_degree: usize, log_inv_rate: usize, m: usize) -> (f64, f64) {
        const INVALID: (f64, f64) = (f64::INFINITY, f64::INFINITY);
        let m_eff = m.max(1);
        let log2_rho = -(log_inv_rate as f64);
        let rho = log2_rho.exp2();
        if rho <= 0.0 || !rho.is_finite() {
            return INVALID;
        }
        if m_eff == 1 && rho >= (4.0 / 9.0) {
            // For m=1: gamma = 1 - sqrt(rho) - sqrt(rho)/(2m) = 1 - 3*sqrt(rho)/2.
            // Gamma must be positive to apply the Section 3.2 argument.
            return INVALID;
        }

        let sqrt_rho = rho.sqrt();
        let eta = sqrt_rho / (2.0 * m_eff as f64);
        let gamma = 1.0 - sqrt_rho - eta;
        if eta <= 0.0 || gamma <= 0.0 || gamma >= 1.0 - sqrt_rho {
            // Invalid theorem regime => no security from this term.
            return INVALID;
        }

        let log2_n = (log_degree + log_inv_rate) as f64;
        let (log2_a_real, log2_list_size) = {
            // Fallback for extreme parameter regimes where exact integer search is not
            // representable.
            let (log2_d_x, log2_d_y, log2_d_z) =
                Self::bchks25_log2_degrees(log_degree, log_inv_rate, m_eff, gamma);
            let log2_gamma_n_plus_1 = Self::log2_add(gamma.log2() + log2_n, 0.0);
            let log2_a = Self::bchks25_log2_a_from_log2_degrees(
                log2_d_x,
                log2_d_y,
                log2_d_z,
                log2_gamma_n_plus_1,
            );
            // Note: we could take log2(floor(2^log2_d_y)) for a tighter list size bound
            (log2_a, log2_d_y)
        };
        if !log2_a_real.is_finite() {
            return INVALID;
        }

        // Clamp `a_bound >= 1` => `log2(a_bound) >= 0`.
        let log2_a_real = log2_a_real.max(0.0);

        // If `a` is small enough, apply `ceil` in normal space for exactness.
        let a = log2_a_real.exp2();
        let a_bound = a.ceil().max(1.0);
        (a_bound.log2(), log2_list_size)
    }

    /// Computes WHIR sumcheck security bits for a sub-round.
    ///
    /// Sumcheck error is d * ℓ_{i,s-1} / |F|, d^*:= 1 + deg_Z(w0) + max_i deg_{X_i}(w0) and d :=
    /// max{d^*,3}.
    ///
    /// Security bits = |F_ext| - log₂(3) - log2(ℓ) + folding_pow_bits
    fn whir_sumcheck_security(
        challenge_field_bits: f64,
        folding_pow_bits: usize,
        log2_list_size: f64,
    ) -> f64 {
        // For our use case, w0 has degree 1 in each variable, so d = 3.
        let sumcheck_degree: f64 = 3.0;
        challenge_field_bits - sumcheck_degree.log2() - log2_list_size + folding_pow_bits as f64
    }

    /// Computes WHIR out-of-domain (OOD) security bits.
    ///
    /// OOD error is 2^{m_i - 1} ℓ_{i,0}^2 / |F| where m_i is the log_degree at the start of WHIR
    /// round `i`.
    ///
    /// Security bits = |F_ext| - log_degree + 1 - 2 * log2(ℓ_{i,0})
    fn whir_ood_security(
        log2_list_size: f64,
        challenge_field_bits: f64,
        log_degree_at_round_start: usize,
    ) -> f64 {
        let base_bits = challenge_field_bits - log_degree_at_round_start as f64 + 1.0;
        base_bits - 2.0 * log2_list_size
    }

    /// Computes WHIR in-domain γ batching security bits.
    ///
    /// Theorem 5.6 / Claim 5.4: batching `t` claims against a list of size `ℓ` gives
    /// error `ℓ * t / |F|`.
    fn whir_gamma_batching_security(
        challenge_field_bits: f64,
        batch_size: usize,
        log2_list_size: f64,
    ) -> f64 {
        debug_assert!(batch_size > 0, "batch_size must be > 0 for gamma batching");
        challenge_field_bits - (batch_size as f64).log2() - log2_list_size
    }
}

/// Prints a detailed soundness report to stdout.
#[allow(clippy::too_many_arguments)]
pub fn print_soundness_report(
    params: &SystemParams,
    challenge_field_bits: f64,
    max_num_constraints_per_air: usize,
    num_airs: usize,
    max_constraint_degree: usize,
    max_log_trace_height: usize,
    num_trace_columns: usize,
    num_stacked_columns: usize,
    n_logup: usize,
    proximity_regime: ProximityRegime,
) {
    let soundness = SoundnessCalculator::calculate(
        params,
        challenge_field_bits,
        max_num_constraints_per_air,
        num_airs,
        max_constraint_degree,
        max_log_trace_height,
        num_trace_columns,
        num_stacked_columns,
        n_logup,
    );

    println!("=== V2 Proof System Soundness Report ===");
    println!();
    println!("System Parameters:");
    println!("  l_skip: {}", params.l_skip);
    println!("  n_stack: {}", params.n_stack);
    println!("  log_blowup: {}", params.log_blowup);
    println!("  WHIR k: {}", params.whir.k);
    println!("  WHIR rounds: {}", params.whir.rounds.len());
    println!("  WHIR mu_pow_bits: {}", params.whir.mu_pow_bits);
    println!(
        "  WHIR query_phase_pow_bits: {}",
        params.whir.query_phase_pow_bits
    );
    println!("  WHIR folding_pow_bits: {}", params.whir.folding_pow_bits);
    println!("  LogUp pow_bits: {}", params.logup.pow_bits);
    println!(
        "  LogUp max_interaction_count: {}",
        params.logup.max_interaction_count
    );
    println!(
        "  LogUp log_max_message_length: {}",
        params.logup.log_max_message_length
    );
    println!("  max_constraint_degree: {}", params.max_constraint_degree);
    println!();
    println!("Proving Context:");
    println!("  challenge_field_bits: {:.0}", challenge_field_bits);
    println!(
        "  max_num_constraints_per_air: {}",
        max_num_constraints_per_air
    );
    println!("  num_airs: {}", num_airs);
    println!("  max_constraint_degree: {}", max_constraint_degree);
    println!("  max_log_trace_height: {}", max_log_trace_height);
    println!("  num_trace_columns: {}", num_trace_columns);
    println!("  num_stacked_columns: {}", num_stacked_columns);
    println!("  n_logup (GKR depth): {}", n_logup);
    println!();
    println!("Security Analysis (bits):");
    println!("  LogUp (α/β + PoW):           {:.1}", soundness.logup_bits);
    println!(
        "  GKR sumcheck:                {:.1}",
        soundness.gkr_sumcheck_bits
    );
    println!(
        "  GKR batching (μ/λ):          {:.1}",
        soundness.gkr_batching_bits
    );
    println!(
        "  ZeroCheck sumcheck:          {:.1}",
        soundness.zerocheck_sumcheck_bits
    );
    println!(
        "  Constraint batching:         {:.1}",
        soundness.constraint_batching_bits
    );
    println!(
        "  Stacked reduction:           {:.1}",
        soundness.stacked_reduction_bits
    );
    println!("  WHIR (round-by-round min):   {:.1}", soundness.whir_bits);
    println!();
    println!(
        "  TOTAL SECURITY:              {:.1} bits",
        soundness.total_bits
    );
    println!();

    println!("WHIR Error Source Breakdown:");
    let whir = &soundness.whir_details;
    println!("  Query error:          {:.1} bits", whir.query_bits);
    println!(
        "  Proximity gaps:       {:.1} bits",
        whir.proximity_gaps_bits
    );
    println!("  Sumcheck error:       {:.1} bits", whir.sumcheck_bits);
    println!("  Min ε_fold:           {:.1} bits", whir.fold_rbr_bits);
    println!("  OOD error:            {:.1} bits", whir.ood_rbr_bits);
    println!(
        "  γ batching error:     {:.1} bits",
        whir.gamma_batching_bits
    );
    println!("  Min ε_shift/ε_fin:    {:.1} bits", whir.shift_rbr_bits);
    println!("  μ batching error:     {:.1} bits", whir.mu_batching_bits);
    println!();

    println!("WHIR Round Breakdown:");
    let k_whir = params.whir.k;
    let mut log_inv_rate = params.log_blowup;
    for (round, round_config) in params.whir.rounds.iter().enumerate() {
        let query_sec =
            proximity_regime.whir_query_security_bits(round_config.num_queries, log_inv_rate);
        println!(
            "  Round {} | rate=2^-{:<2} | queries={:<3} | query_sec={:5.1} | pow={} | fold_pow={}",
            round,
            log_inv_rate,
            round_config.num_queries,
            query_sec,
            params.whir.query_phase_pow_bits,
            params.whir.folding_pow_bits
        );
        log_inv_rate += k_whir - 1;
    }
}

/// Calculates the minimum WHIR queries needed for a target security level.
pub fn min_whir_queries(
    proximity_regime: ProximityRegime,
    target_security_bits: usize,
    log_inv_rate: usize,
) -> usize {
    WhirConfig::queries(proximity_regime, target_security_bits, log_inv_rate)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeField64;

    use super::*;
    use crate::{config::WhirRoundConfig, interaction::LogUpSecurityParameters};

    fn babybear_quartic_extension_bits() -> f64 {
        4.0 * (BabyBear::ORDER_U64 as f64).log2()
    }
    // ==========================================================================
    // Test fixtures
    // ==========================================================================

    fn test_params() -> SystemParams {
        SystemParams {
            l_skip: 3,
            n_stack: 8,
            w_stack: 64,
            log_blowup: 1,
            whir: WhirConfig {
                k: 4,
                rounds: vec![
                    WhirRoundConfig { num_queries: 36 },
                    WhirRoundConfig { num_queries: 18 },
                ],
                mu_pow_bits: 16,
                query_phase_pow_bits: 16,
                folding_pow_bits: 10,
                proximity: WhirProximityStrategy::UniqueDecoding,
            },
            logup: LogUpSecurityParameters {
                max_interaction_count: 1 << 20,
                log_max_message_length: 4,
                pow_bits: 16,
            },
            max_constraint_degree: 5,
        }
    }

    // ==========================================================================
    // Unit tests
    // ==========================================================================
    const TARGET_SECURITY_BITS: usize = 100;

    #[test]
    fn test_soundness_calculation() {
        let params = test_params();
        let soundness = SoundnessCalculator::calculate(
            &params,
            babybear_quartic_extension_bits(),
            1000,
            50,
            4,
            24,
            200,
            10,
            15,
        );

        assert!(soundness.logup_bits > 0.0);
        assert!(soundness.gkr_sumcheck_bits > 0.0);
        assert!(soundness.gkr_batching_bits > 0.0);
        assert!(soundness.zerocheck_sumcheck_bits > 0.0);
        assert!(soundness.constraint_batching_bits > 0.0);
        assert!(soundness.stacked_reduction_bits > 0.0);
        assert!(soundness.whir_bits > 0.0);
        assert!(soundness.total_bits > 0.0);

        let expected_total = soundness
            .logup_bits
            .min(soundness.gkr_sumcheck_bits)
            .min(soundness.gkr_batching_bits)
            .min(soundness.zerocheck_sumcheck_bits)
            .min(soundness.constraint_batching_bits)
            .min(soundness.stacked_reduction_bits)
            .min(soundness.whir_bits);
        assert!((soundness.total_bits - expected_total).abs() < 0.001);
    }

    #[test]
    fn test_whir_query_calculation() {
        let queries = min_whir_queries(ProximityRegime::UniqueDecoding, TARGET_SECURITY_BITS, 1);
        assert!(queries > 0);
    }

    #[test]
    fn test_logup_soundness() {
        let params = test_params();
        let security = SoundnessCalculator::calculate_logup_soundness(
            &params,
            babybear_quartic_extension_bits(),
            0.0,
        );
        assert!(security > TARGET_SECURITY_BITS as f64);
    }

    #[test]
    fn test_whir_unique_decoding_security() {
        // rate = 0.5: error = 0.75, security per query ≈ 0.415 bits
        let security = ProximityRegime::UniqueDecoding.whir_query_security_bits(100, 1);
        assert!(
            (security - 41.5).abs() < 1.0,
            "Expected ~41.5, got {}",
            security
        );

        // rate = 0.25: error = 0.625, security per query ≈ 0.678 bits
        let security_blowup2 = ProximityRegime::UniqueDecoding.whir_query_security_bits(100, 2);
        assert!(
            (security_blowup2 - 67.8).abs() < 1.0,
            "Expected ~67.8, got {}",
            security_blowup2
        );
    }

    #[test]
    fn test_whir_gamma_batching_uses_list_size_and_full_batch_size() {
        let security = SoundnessCalculator::whir_gamma_batching_security(100.0, 5, 3.0);
        let expected = 100.0 - 5.0_f64.log2() - 3.0;
        assert!((security - expected).abs() < 1e-9);
    }

    #[test]
    fn test_combine_security_bits_sums_errors_before_taking_log() {
        let combined = SoundnessCalculator::combine_security_bits(100.0, 100.0);
        let expected = 99.0;
        assert!((combined - expected).abs() < 1e-9);
    }

    #[test]
    fn test_bchks25_reference_m2_enforces_dz_ge_dy() {
        let (_log2_d_x, log2_d_y, log2_d_z) =
            SoundnessCalculator::bchks25_reference_log2_degrees(24, 2, 2);
        assert!(log2_d_z >= log2_d_y);
    }

    #[test]
    fn test_bchks25_m1_requires_rho_below_four_ninths() {
        // log_inv_rate=1 => rho=1/2 > 4/9, so m=1 regime is invalid.
        let invalid = SoundnessCalculator::log2_a_bound_bchks25(12, 1, 1);
        assert!(invalid.0.is_infinite() && invalid.1.is_infinite());

        // log_inv_rate=2 => rho=1/4 < 4/9, so m=1 regime is admissible.
        let valid = SoundnessCalculator::log2_a_bound_bchks25(12, 2, 1);
        assert!(valid.0.is_finite() && valid.1.is_finite());
    }
}

/// The D_X, D_Y, D_Z given in [BCHKS25] Lemma 3.1 are not optimal (as noted by the authors) and `m
/// < 3` assumption is not needed. We can perform a brute force search over possible values of D_X,
/// D_Y, D_Z that satisfy properties to allow the proof of Lemma 3.1 to go through.
///
/// Everything in this module is still backed by proven results.
#[allow(dead_code)]
#[cfg(feature = "soundness-bchks25-optimized")]
mod bchks25_brute_force_params {
    use crate::soundness::SoundnessCalculator;

    const BCHKS25_DY_SEARCH_MIN_MAX: u128 = 9;
    const BCHKS25_DY_SEARCH_REF_MULTIPLIER: u128 = 4;
    const BCHKS25_DY_SEARCH_HARD_MAX: u128 = 4096;
    const BCHKS25_DZ_SEARCH_MAX: u128 = 500_000;

    #[derive(Clone, Copy, Debug)]
    pub struct Bchks25Degrees {
        pub d_x: u128,
        pub d_y: u128,
        // Integer index representation for Z-degree support:
        // `j + h < D_Z` is represented as `0 <= h <= d_z - j`, so `d_z = ceil(D_Z) - 1`.
        pub d_z: u128,
    }

    /// We find optimal parameters `D_X, D_Y, D_Z` that minimize Equation (13) in BCHKS25 **and**
    /// satisfy the conditions necessary for the proof of Lemma 3.1 to carry through:
    /// - `D_X >= k * D_Y`
    /// - `D_Z >= D_Y`
    /// - for `m < 3`, additionally `D_Z >=` Equation (9), i.e. `D_Z = max(D_Y, Equation (9) value)`
    /// - `D_Y >= m - 1`
    /// - `D_X <= (1 - gamma) * m * n` (Section 3.2 precondition before applying Equation (13))
    /// - `n_vars > n_eqs` where `n_eqs` is given by Equation (11) and `n_vars = \sum_0^{ceil(D_Y) -
    ///   1} (ceil(D_X) - kj)(ceil(D_Z) - j)`
    /// ```text
    /// n_{\mathrm{vars}}=\sum_{j=0}^{\lceil D_Y\rceil-1}(\lceil D_X\rceil-kj)(\lceil D_Z\rceil-j)
    /// n_{\mathrm{eqs}}=n\sum_{s=0}^{m-1}(\lceil D_Z\rceil-s)(m-s)
    /// ```
    ///
    /// Brute-force search for degrees minimizing Equation (13):
    /// - scan `D_Y in [max(1, m - 1), D_Y_max]`
    /// - `D_Y_max` is chosen from a scaled Lemma 3.1 reference degree (with a hard cap to keep
    ///   computation bounded)
    /// - scan candidate `D_X >= k * D_Y` up to the Section 3.2 limit
    /// - solve directly for the smallest valid `D_Z` for each `(D_X, D_Y)`
    pub fn bchks25_optimal_degrees_bruteforce(
        log_degree: usize,
        log_inv_rate: usize,
        m: usize,
        gamma: f64,
    ) -> Option<(Bchks25Degrees, f64)> {
        if !gamma.is_finite() || gamma <= 0.0 {
            return None;
        }

        let log_n = log_degree.checked_add(log_inv_rate)?;
        if log_n >= u128::BITS as usize || log_degree >= u128::BITS as usize {
            return None;
        }

        let n = 1_u128.checked_shl(log_n as u32)?;
        let k = 1_u128.checked_shl(log_degree as u32)?;
        let m_u = m as u128;

        let agreements_plus_one = (gamma * n as f64).ceil() + 1.0;
        if !agreements_plus_one.is_finite() || agreements_plus_one <= 0.0 {
            return None;
        }
        let log2_agreements_plus_one = agreements_plus_one.log2();
        let max_d_x_for_gamma = (1.0 - gamma) * (m_u as f64) * (n as f64);
        if !max_d_x_for_gamma.is_finite() || max_d_x_for_gamma <= 0.0 {
            return None;
        }

        let mut best: Option<(Bchks25Degrees, f64)> = None;
        let d_y_start = 1_u128.max(m_u.saturating_sub(1));
        let d_y_end = bchks25_dy_search_upper_bound(log_degree, log_inv_rate, m);
        let d_z_floor = if m < 3 {
            bchks25_dz_eq9_index_lower_bound(log_inv_rate, m)?
        } else {
            0
        };
        if d_y_start > d_y_end {
            return None;
        }

        for d_y in d_y_start..=d_y_end {
            let Some(d_x_base) = k.checked_mul(d_y) else {
                continue;
            };
            if (d_x_base as f64) >= max_d_x_for_gamma {
                continue;
            }

            let d_x_upper = max_d_x_for_gamma.ceil() as u128;
            let mut d_x_candidates = Vec::with_capacity(24);
            d_x_candidates.push(d_x_base);
            if let Some(v) = d_x_base.checked_add(1) {
                d_x_candidates.push(v);
            }

            let d_y_plus_1 = d_y.checked_add(1)?;
            let k_term = k
                .checked_mul(d_y)?
                .checked_mul(d_y_plus_1)?
                .checked_div(2)?;
            let a_e = n.checked_mul(m_u.checked_mul(m_u.checked_add(1)?)?.checked_div(2)?)?;
            let d_x_slope_cross = a_e.checked_add(k_term)?.checked_div(d_y_plus_1)?;
            for off in [0_u128, 1, 2] {
                if d_x_slope_cross >= off {
                    d_x_candidates.push(d_x_slope_cross - off);
                }
                if let Some(v) = d_x_slope_cross.checked_add(off) {
                    d_x_candidates.push(v);
                }
            }

            if d_x_upper > d_x_base {
                let step = ((d_x_upper - d_x_base) / 16).max(1);
                let mut cur = d_x_base;
                while cur <= d_x_upper {
                    d_x_candidates.push(cur);
                    let Some(next) = cur.checked_add(step) else {
                        break;
                    };
                    if next <= cur {
                        break;
                    }
                    cur = next;
                }
                d_x_candidates.push(d_x_upper);
                d_x_candidates.push(d_x_upper.saturating_sub(1));
            }

            d_x_candidates.sort_unstable();
            d_x_candidates.dedup();

            for d_x in d_x_candidates {
                if d_x < d_x_base || (d_x as f64) >= max_d_x_for_gamma {
                    continue;
                }

                let Some(d_z) = bchks25_min_dz_for_dx_dy(k, n, m_u, d_x, d_y, d_z_floor) else {
                    continue;
                };

                let log2_d_x = (d_x as f64).log2();
                let log2_d_y = (d_y as f64).log2();
                let log2_d_z = (d_z as f64).log2();
                let log2_a = SoundnessCalculator::bchks25_log2_a_from_log2_degrees(
                    log2_d_x,
                    log2_d_y,
                    log2_d_z,
                    log2_agreements_plus_one,
                );
                if !log2_a.is_finite() {
                    continue;
                }

                let candidate = (Bchks25Degrees { d_x, d_y, d_z }, log2_a);
                match best {
                    None => best = Some(candidate),
                    Some((best_deg, best_log2_a)) => {
                        // Stable tie-breaking keeps smaller degrees if scores are effectively
                        // equal.
                        let better = log2_a + 1e-12 < best_log2_a
                            || ((log2_a - best_log2_a).abs() <= 1e-12
                                && (d_y < best_deg.d_y
                                    || (d_y == best_deg.d_y
                                        && (d_z < best_deg.d_z
                                            || (d_z == best_deg.d_z && d_x < best_deg.d_x)))));
                        if better {
                            best = Some(candidate);
                        }
                    }
                }
            }
        }
        best
    }

    fn bchks25_dy_search_upper_bound(log_degree: usize, log_inv_rate: usize, m: usize) -> u128 {
        let (_log2_d_x, log2_d_y, _log2_d_z) =
            SoundnessCalculator::bchks25_reference_log2_degrees(log_degree, log_inv_rate, m);
        if !log2_d_y.is_finite() {
            return BCHKS25_DY_SEARCH_HARD_MAX;
        }
        let ref_d_y = log2_d_y.exp2().ceil();
        if !ref_d_y.is_finite() || ref_d_y <= 0.0 {
            return BCHKS25_DY_SEARCH_HARD_MAX;
        }
        let ref_scaled = (ref_d_y as u128).saturating_mul(BCHKS25_DY_SEARCH_REF_MULTIPLIER);
        ref_scaled.clamp(BCHKS25_DY_SEARCH_MIN_MAX, BCHKS25_DY_SEARCH_HARD_MAX)
    }

    /// Index-space lower bound from Equation (9), where `d_z = ceil(D_Z)-1`.
    fn bchks25_dz_eq9_index_lower_bound(log_inv_rate: usize, m: usize) -> Option<u128> {
        let m_u = m.max(1) as u128;
        if log_inv_rate >= u128::BITS as usize {
            return None;
        }

        // Equation (9): D_Z = ((m + 1/2)^2 * (n/k)) / 3
        // with n/k = 2^{log_inv_rate}. So:
        // D_Z = ((2m+1)^2 * 2^{log_inv_rate}) / 12.
        let n_over_k = 1_u128.checked_shl(log_inv_rate as u32)?;
        let two_m_plus_1 = m_u.checked_mul(2)?.checked_add(1)?;
        let numerator = two_m_plus_1
            .checked_mul(two_m_plus_1)?
            .checked_mul(n_over_k)?;
        let ceil_d_z = numerator.checked_add(11)?.checked_div(12)?;
        Some(ceil_d_z.saturating_sub(1))
    }

    /// Counts interpolation variables for fixed `D_X, D_Y, D_Z`.
    ///
    /// Here `d_x`, `d_y` and `d_z` are index-space bounds
    /// (`ceil(D_X)-1`, `ceil(D_Y)-1`, `ceil(D_Z)-1`), not real degrees.
    /// This matches the lattice count used in Lemma 3.1:
    /// `j + h < D_Z` contributes `(d_z - j + 1)` monomials in `Z`.
    #[cfg(test)]
    fn bchks25_num_vars(k: u128, d_x: u128, d_y: u128, d_z: u128) -> Option<u128> {
        if d_z < d_y {
            return Some(0);
        }

        let mut total = 0_u128;
        for j in 0..=d_y {
            let x_terms = d_x.checked_sub(k.checked_mul(j)?)?.checked_add(1)?;
            let z_terms = (d_z - j).checked_add(1)?;
            let add = x_terms.checked_mul(z_terms)?;
            total = total.checked_add(add)?;
        }
        Some(total)
    }

    /// Equation (11) RHS (closed form):
    /// `n_eqs = n * ( m(m+1)/2 * ceil(D_Z) - (m^3-m)/6 )`.
    ///
    /// We store `d_z = ceil(D_Z)-1` (index-space convention), therefore `ceil(D_Z)=d_z+1`.
    #[cfg(test)]
    fn bchks25_num_eqs_eq11(n: u128, m: u128, d_z: u128) -> Option<u128> {
        let ceil_d_z = d_z.checked_add(1)?;
        let m_plus_1 = m.checked_add(1)?;
        let m_choose_2_scaled = m.checked_mul(m_plus_1)?.checked_div(2)?;
        let m_cubic_minus_m_over_6 = m
            .checked_mul(m)?
            .checked_mul(m)?
            .checked_sub(m)?
            .checked_div(6)?;
        let inner = m_choose_2_scaled
            .checked_mul(ceil_d_z)?
            .checked_sub(m_cubic_minus_m_over_6)?;
        n.checked_mul(inner)
    }

    /// Solves for the minimal index-space `d_z` satisfying
    /// `n_vars(d_z) > n_eqs(d_z)` (Equation (11)).
    ///
    /// Both sides are affine in `d_z`, so this is a 1D linear inequality with bounds
    /// `d_z in [max(d_y, m-1), BCHKS25_DZ_SEARCH_MAX]` and does not require binary search.
    ///
    /// The returned value corresponds to real-degree parameter `D_Z` via `d_z = ceil(D_Z)-1`.
    fn bchks25_min_dz_for_dx_dy(
        k: u128,
        n: u128,
        m: u128,
        d_x: u128,
        d_y: u128,
        d_z_floor: u128,
    ) -> Option<u128> {
        let low = d_y.max(m.saturating_sub(1)).max(d_z_floor);
        let high = BCHKS25_DZ_SEARCH_MAX;
        if low > high {
            return None;
        }

        // n_vars(d_z) for fixed (d_x, d_y):
        // n_vars = sum_{j=0}^{d_y} (d_x - k*j + 1) * (d_z - j + 1)
        //        = A_v * (d_z + 1) - B_v
        //
        let d_y_plus_1 = d_y.checked_add(1)?;
        let d_y_d_y_plus_1_over_2 = d_y.checked_mul(d_y_plus_1)?.checked_div(2)?;
        let d_y_d_y_plus_1_two_d_y_plus_1_over_6 = d_y
            .checked_mul(d_y)?
            .checked_add(d_y)?
            .checked_mul(d_y.checked_mul(2)?.checked_add(1)?)?
            .checked_div(6)?;
        let a_v = d_y_plus_1
            .checked_mul(d_x.checked_add(1)?)?
            .checked_sub(k.checked_mul(d_y_d_y_plus_1_over_2)?)?;
        let b_v = d_x
            .checked_add(1)?
            .checked_mul(d_y_d_y_plus_1_over_2)?
            .checked_sub(k.checked_mul(d_y_d_y_plus_1_two_d_y_plus_1_over_6)?)?;
        if a_v == 0 {
            return None;
        }

        // n_eqs(d_z) = A_e * (d_z + 1) - B_e from Equation (11) closed form.
        let m_plus_1 = m.checked_add(1)?;
        let a_e = n.checked_mul(m.checked_mul(m_plus_1)?.checked_div(2)?)?;
        let b_e = n.checked_mul(
            m.checked_mul(m)?
                .checked_mul(m)?
                .checked_sub(m)?
                .checked_div(6)?,
        )?;

        let is_valid = |d_z: u128| -> Option<bool> {
            let x = d_z.checked_add(1)?;
            let n_vars = a_v.checked_mul(x)?.checked_sub(b_v)?;
            let n_eqs = a_e.checked_mul(x)?.checked_sub(b_e)?;
            Some(n_vars > n_eqs)
        };

        match a_v.cmp(&a_e) {
            core::cmp::Ordering::Greater => {
                // Increasing predicate: solve exactly, then clamp to bounds.
                let slope = a_v.checked_sub(a_e)?;
                let candidate = if b_v < b_e {
                    low
                } else {
                    let rhs = b_v.checked_sub(b_e)?;
                    let x_min = rhs.checked_div(slope)?.checked_add(1)?;
                    let d_min = x_min.checked_sub(1)?;
                    low.max(d_min)
                };
                if candidate > high {
                    return None;
                }
                if is_valid(candidate)? {
                    Some(candidate)
                } else {
                    None
                }
            }
            core::cmp::Ordering::Equal => {
                if b_v < b_e {
                    Some(low)
                } else {
                    None
                }
            }
            core::cmp::Ordering::Less => {
                // Decreasing predicate: if low is not valid, no later value can be valid.
                if is_valid(low)? {
                    Some(low)
                } else {
                    None
                }
            }
        }
    }

    #[test]
    fn test_bchks25_optimizer_finds_minimal_valid_dz() {
        let log_degree = 14;
        let log_inv_rate = 5;
        let m = 1;

        let rho = (-(log_inv_rate as f64)).exp2();
        let sqrt_rho = rho.sqrt();
        let eta = sqrt_rho / (2.0 * m as f64);
        let gamma = 1.0 - sqrt_rho - eta;

        let (degrees, _log2_a) =
            bchks25_optimal_degrees_bruteforce(log_degree, log_inv_rate, m, gamma)
                .expect("optimizer should find valid degrees");
        let n = 1_u128 << (log_degree + log_inv_rate);
        let k = 1_u128 << log_degree;
        let m_u = m as u128;

        assert!(degrees.d_y >= m_u.saturating_sub(1));
        let max_d_x_for_gamma = (1.0 - gamma) * (m as f64) * (n as f64);
        assert!(degrees.d_x >= k * degrees.d_y);
        assert!((degrees.d_x as f64) < max_d_x_for_gamma);
        let d_z_floor = bchks25_dz_eq9_index_lower_bound(log_inv_rate, m)
            .expect("d_z floor should be representable");
        assert!(degrees.d_z >= degrees.d_y.max(d_z_floor));
        let vars = bchks25_num_vars(k, degrees.d_x, degrees.d_y, degrees.d_z)
            .expect("vars should fit in u128");
        let eqs = bchks25_num_eqs_eq11(n, m_u, degrees.d_z).expect("eqs should fit in u128");
        assert!(vars > eqs);
        let low = degrees.d_y.max(m_u.saturating_sub(1)).max(d_z_floor);
        if degrees.d_z > low {
            let prev_vars = bchks25_num_vars(k, degrees.d_x, degrees.d_y, degrees.d_z - 1)
                .expect("vars should fit in u128");
            let prev_eqs =
                bchks25_num_eqs_eq11(n, m_u, degrees.d_z - 1).expect("eqs should fit in u128");
            assert!(prev_vars <= prev_eqs);
        }
    }

    #[test]
    fn test_bchks25_eq11_closed_form_matches_expanded_sum() {
        let n = 1_u128 << 12;
        for m in 2_u128..=8 {
            for d_z in (m - 1)..=(m + 12) {
                let closed_form =
                    bchks25_num_eqs_eq11(n, m, d_z).expect("closed-form n_eqs should fit in u128");
                let ceil_d_z = d_z + 1;
                let mut expanded_sum = 0_u128;
                for s in 0..m {
                    expanded_sum += (ceil_d_z - s) * (m - s);
                }
                let expanded = n * expanded_sum;
                assert_eq!(closed_form, expanded);
            }
        }
    }

    #[test]
    fn test_bchks25_min_dz_direct_solve_matches_linear_scan() {
        let cases = [
            // Increasing predicate case.
            (
                1_u128 << 20,
                1_u128 << 22,
                3_u128,
                2_u128,
                (1_u128 << 20) * 2,
            ),
            // Decreasing predicate case.
            (
                1_u128 << 12,
                1_u128 << 24,
                6_u128,
                5_u128,
                (1_u128 << 12) * 5,
            ),
            // Near-flat-ish case.
            (
                1_u128 << 16,
                1_u128 << 20,
                4_u128,
                3_u128,
                (1_u128 << 16) * 3 + 1,
            ),
            // m < 3 with Equation (9) lower bound active.
            (
                1_u128 << 12,
                1_u128 << 17,
                1_u128,
                2_u128,
                (1_u128 << 12) * 2 + 17,
            ),
        ];

        for (k, n, m, d_y, d_x) in cases {
            assert!(d_x >= k * d_y);
            let log_inv_rate = (n / k).ilog2() as usize;
            let d_z_floor = if m < 3 {
                bchks25_dz_eq9_index_lower_bound(log_inv_rate, m as usize)
                    .expect("d_z floor should be representable")
            } else {
                0
            };
            let got = bchks25_min_dz_for_dx_dy(k, n, m, d_x, d_y, d_z_floor);
            let low = d_y.max(m.saturating_sub(1)).max(d_z_floor);
            let expected = (low..=BCHKS25_DZ_SEARCH_MAX).find(|&d_z| {
                let vars = bchks25_num_vars(k, d_x, d_y, d_z).expect("vars should fit");
                let eqs = bchks25_num_eqs_eq11(n, m, d_z).expect("eqs should fit");
                vars > eqs
            });
            assert_eq!(got, expected);
        }
    }
}
