//! Soundness calculator for the v2 proof system.
//!
//! The v2 proof system consists of the following components:
//! 1. LogUp GKR - Fractional sumcheck for interaction constraints
//! 2. ZeroCheck - Batched constraint verification across AIRs
//! 3. Stacked Reduction - Reduces trace evaluations to stacked polynomial evaluations
//! 4. WHIR - Polynomial commitment opening via FRI-like folding
//!
//! Each component contributes to the overall soundness error, and the total security
//! is the minimum across all components.

use p3_field::PrimeField32;

use crate::{
    config::{SoundnessType, SystemParams, WhirConfig},
    D_EF, F,
};

/// Returns the number of bits in the extension field used for challenges.
///
/// For BabyBear4: log₂(p) × 4 ≈ 30.9 × 4 ≈ 123.6 bits.
pub fn challenge_field_bits() -> f64 {
    (F::ORDER_U32 as f64).log2() * D_EF as f64
}

/// Soundness analysis for the v2 proof system.
#[derive(Clone, Debug)]
pub struct SoundnessAnalysis {
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
    pub whir_details: WhirSoundnessAnalysis,
    /// Total security bits (minimum of all components).
    pub total_bits: f64,
}

/// Detailed breakdown of WHIR soundness by error source.
#[derive(Clone, Debug)]
pub struct WhirSoundnessAnalysis {
    /// Security bits from query sampling (unique decoding regime).
    pub query_bits: f64,
    /// Security bits from proximity gaps (folding soundness).
    pub proximity_gaps_bits: f64,
    /// Security bits from sumcheck within WHIR rounds.
    pub sumcheck_bits: f64,
    /// Security bits from out-of-domain sampling.
    pub ood_bits: f64,
    /// Security bits from γ batching (combining query and OOD claims).
    pub gamma_batching_bits: f64,
    /// Security bits from μ batching (initial polynomial batching).
    pub mu_batching_bits: f64,
}

impl SoundnessAnalysis {
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
        let logup_bits = Self::calculate_logup_soundness(params, challenge_field_bits);

        let gkr_sumcheck_bits =
            Self::calculate_gkr_sumcheck_soundness(challenge_field_bits, params.l_skip, n_logup);

        let gkr_batching_bits =
            Self::calculate_gkr_batching_soundness(challenge_field_bits, params.l_skip, n_logup);

        let zerocheck_sumcheck_bits = Self::calculate_zerocheck_sumcheck_soundness(
            challenge_field_bits,
            max_constraint_degree,
            params.l_skip,
            max_log_trace_height,
        );

        let constraint_batching_bits = Self::calculate_constraint_batching_soundness(
            challenge_field_bits,
            max_num_constraints_per_air,
            num_airs,
        );

        let stacked_reduction_bits = Self::calculate_stacked_reduction_soundness(
            challenge_field_bits,
            num_trace_columns,
            params.l_skip,
            params.n_stack,
        );

        let (whir_bits, whir_details) =
            Self::calculate_whir_soundness(params, challenge_field_bits, num_stacked_columns);

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
    fn calculate_logup_soundness(params: &SystemParams, challenge_field_bits: f64) -> f64 {
        let logup = &params.logup;
        let max_interaction_count_bits = (2.0 * logup.max_interaction_count as f64).log2();

        challenge_field_bits - max_interaction_count_bits - logup.log_max_message_length as f64
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
    ///
    /// Security is the minimum across all sources.
    fn calculate_zerocheck_sumcheck_soundness(
        challenge_field_bits: f64,
        max_constraint_degree: usize,
        l_skip: usize,
        max_log_trace_height: usize,
    ) -> f64 {
        let univariate_degree = (max_constraint_degree + 1) * ((1 << l_skip) - 1);
        let multilinear_degree = max_constraint_degree + 1;

        // Per-round sumcheck security (univariate round is typically the bottleneck)
        let worst_degree = univariate_degree.max(multilinear_degree);
        let sumcheck_bits = challenge_field_bits - (worst_degree as f64).log2();

        // Polynomial identity testing: trace polynomial has deg_Z ≤ 2^l_skip - 1, deg_{X_i} ≤ 1
        // n_max = max_log_trace_height - l_skip multilinear variables
        let n_max = max_log_trace_height.saturating_sub(l_skip);
        let poly_degree_sum = (1 << l_skip) - 1 + n_max;
        let poly_identity_bits = challenge_field_bits - (poly_degree_sum as f64).log2();

        sumcheck_bits.min(poly_identity_bits)
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
    ) -> f64 {
        let lambda_batching_bits =
            challenge_field_bits - (max_num_constraints_per_air as f64).log2();
        // Each AIR contributes 3 sum claims to the batch sumcheck:
        // 1. ZeroCheck (constraint satisfaction)
        // 2. LogUp numerator (p̂(ξ) input layer)
        // 3. LogUp denominator (q̂(ξ) input layer)
        let mu_batching_bits = challenge_field_bits - (3.0 * num_airs as f64).log2();

        lambda_batching_bits.min(mu_batching_bits)
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
    /// 4. **Polynomial identity testing at u**: If the prover committed to a wrong stacked
    ///    polynomial q̂' ≠ q̂, evaluating at random u catches this. The stacked polynomial has deg_Z
    ///    ≤ 2^l_skip - 1 and deg_{X_i} ≤ 1. Error ≤ (2^l_skip - 1 + n_stack) / |F_ext|
    ///
    /// Security is the minimum across all sources.
    fn calculate_stacked_reduction_soundness(
        challenge_field_bits: f64,
        num_trace_columns: usize,
        l_skip: usize,
        n_stack: usize,
    ) -> f64 {
        // λ batching: single Schwartz-Zippel application
        let batching_bits = challenge_field_bits - (2.0 * num_trace_columns as f64).log2();

        // Univariate sumcheck round
        let univariate_degree = 2 * ((1 << l_skip) - 1);
        let univariate_bits = challenge_field_bits - (univariate_degree as f64).log2();

        // Multilinear sumcheck rounds: each has degree 2
        let multilinear_bits = challenge_field_bits - 1.0; // log2(2) = 1

        // Polynomial identity testing: stacked polynomial has deg_Z ≤ 2^l_skip - 1, deg_{X_i} ≤ 1
        // By Schwartz-Zippel, error ≤ (sum of degrees) / |F_ext|
        let poly_degree_sum = (1 << l_skip) - 1 + n_stack;
        let poly_identity_bits = challenge_field_bits - (poly_degree_sum as f64).log2();

        // Security is the minimum across all sources
        batching_bits
            .min(univariate_bits)
            .min(multilinear_bits)
            .min(poly_identity_bits)
    }

    /// WHIR soundness analysis based on the WHIR paper (ePrint 2024/1586).
    ///
    /// Error sources (in unique decoding regime with list size ℓ = 1):
    ///
    /// 1. **Fold error** (per sumcheck sub-round s within WHIR round i):
    ///    - Sumcheck: 2/|F| for s=0, 3/|F| for s>0
    ///    - Proximity gap: 2^{m_s} / (ρ · |F|) where m_s is log_degree, ρ is rate
    ///    - Protected by folding_pow_bits
    ///
    /// 2. **OOD error** (per non-final round):
    ///    - Error: 2^{m_i - 1} / |F| where m_i is log_degree at start of round
    ///    - NOT protected by PoW (sampled after commitment observed)
    ///
    /// 3. **Shift error** (per non-final round):
    ///    - Query proximity: ((1 + ρ) / 2)^t where t = num_queries
    ///    - In-domain batching: (t - 1) / |F|
    ///    - Query indices protected by query_phase_pow_bits
    ///    - γ batching NOT protected by PoW (Merkle proofs observed before γ)
    ///
    /// 4. **Final polynomial error** (final round only):
    ///    - Query proximity: ((1 + ρ) / 2)^t
    ///    - Our implementation also does γ batching: (t - 1) / |F|
    ///    - Query indices protected by query_phase_pow_bits
    ///
    /// 5. **Initial μ batching**: num_stacked_columns / |F|
    fn calculate_whir_soundness(
        params: &SystemParams,
        challenge_field_bits: f64,
        num_stacked_columns: usize,
    ) -> (f64, WhirSoundnessAnalysis) {
        let whir = &params.whir;
        let k_whir = whir.k;
        let log_stacked_height = params.log_stacked_height();
        let num_whir_rounds = whir.rounds.len();

        let mut min_query_bits = f64::INFINITY;
        let mut min_prox_gaps_bits = f64::INFINITY;
        let mut min_sumcheck_bits = f64::INFINITY;
        let mut min_ood_bits = f64::INFINITY;
        let mut min_gamma_batching_bits = f64::INFINITY;

        assert!(
            num_stacked_columns >= 2,
            "WHIR requires at least 2 stacked columns for μ batching"
        );
        // Initial μ batching: err*(C, num_stacked_columns, δ) = (n-1) · 2^m / (ρ · |F|)
        // where m = log_stacked_height (the RS code blocklength), ρ = 2^{-log_blowup}
        // Security = |F_ext| - log₂(num_stacked_columns - 1) - log_stacked_height - log_blowup + mu_pow_bits
        let mu_batching_bits = challenge_field_bits
            - ((num_stacked_columns - 1) as f64).log2()
            - log_stacked_height as f64
            - params.log_blowup as f64
            + whir.mu_pow_bits as f64;

        let mut log_inv_rate = params.log_blowup;
        let mut current_log_degree = log_stacked_height;

        for (round, round_config) in whir.rounds.iter().enumerate() {
            let is_final_round = round == num_whir_rounds - 1;
            let next_rate = log_inv_rate + (k_whir - 1);

            // Sumcheck sub-rounds within this WHIR round
            for sub_round in 0..k_whir {
                current_log_degree -= 1;

                // Proximity gap: err*(C, 2, δ) = 2^m / (ρ · |F|)
                // Security = |F_ext| - log_degree - log_inv_rate + folding_pow
                let prox_gaps_bits = challenge_field_bits
                    - current_log_degree as f64
                    - log_inv_rate as f64
                    + whir.folding_pow_bits as f64;
                min_prox_gaps_bits = min_prox_gaps_bits.min(prox_gaps_bits);

                // Sumcheck error: 2/|F| for sub_round 0, 3/|F| for others
                let sumcheck_degree: f64 = if sub_round == 0 { 2.0 } else { 3.0 };
                let sumcheck_bits =
                    challenge_field_bits - sumcheck_degree.log2() + whir.folding_pow_bits as f64;
                min_sumcheck_bits = min_sumcheck_bits.min(sumcheck_bits);
            }

            // Query error (all rounds): ((1 + ρ) / 2)^t
            // Protected by query_phase_pow_bits
            let query_bits = Self::whir_query_security(
                SoundnessType::UniqueDecoding,
                round_config.num_queries,
                log_inv_rate,
            ) + whir.query_phase_pow_bits as f64;
            min_query_bits = min_query_bits.min(query_bits);

            // In-domain γ batching: (t - 1) / |F| for t queries
            // NOT protected by PoW - Merkle proofs are observed before γ is sampled
            // Note: We use (t - 1) per WHIR paper analysis
            let num_queries = round_config.num_queries;
            if num_queries > 1 {
                let gamma_batching_bits =
                    challenge_field_bits - ((num_queries - 1) as f64).log2();
                min_gamma_batching_bits = min_gamma_batching_bits.min(gamma_batching_bits);
            }

            if !is_final_round {
                // OOD error: 2^{m - 1} / |F| where m is log_degree at start of round
                // Security = |F_ext| - (log_degree - 1) = |F_ext| - log_degree + 1
                // NOT protected by PoW (sampled after commitment observed)
                // Note: current_log_degree has already been decremented by k_whir,
                // so we need to use (current_log_degree + k_whir) for the degree at round start
                let log_degree_at_round_start = current_log_degree + k_whir;
                let ood_bits = challenge_field_bits - log_degree_at_round_start as f64 + 1.0;
                min_ood_bits = min_ood_bits.min(ood_bits);

                tracing::debug!(
                    "WHIR round {} | rate=2^-{} | queries={} | query={:.1} | prox_gaps={:.1} | sumcheck={:.1} | ood={:.1} | gamma={:.1}",
                    round, log_inv_rate, round_config.num_queries, query_bits,
                    min_prox_gaps_bits, min_sumcheck_bits, ood_bits, min_gamma_batching_bits,
                );
            } else {
                tracing::debug!(
                    "WHIR round {} (final) | rate=2^-{} | queries={} | query={:.1} | prox_gaps={:.1} | sumcheck={:.1} | gamma={:.1}",
                    round, log_inv_rate, round_config.num_queries, query_bits,
                    min_prox_gaps_bits, min_sumcheck_bits, min_gamma_batching_bits,
                );
            }

            log_inv_rate = next_rate;
        }

        let details = WhirSoundnessAnalysis {
            query_bits: min_query_bits,
            proximity_gaps_bits: min_prox_gaps_bits,
            sumcheck_bits: min_sumcheck_bits,
            ood_bits: min_ood_bits,
            gamma_batching_bits: min_gamma_batching_bits,
            mu_batching_bits,
        };

        let min_security = min_query_bits
            .min(min_prox_gaps_bits)
            .min(min_sumcheck_bits)
            .min(min_ood_bits)
            .min(min_gamma_batching_bits)
            .min(mu_batching_bits);

        (min_security, details)
    }

    /// Computes WHIR query security bits.
    ///
    /// In unique decoding regime:
    /// Error probability per query = (1 + ρ) / 2, where ρ is the rate.
    /// Security bits = -n × log₂((1 + ρ) / 2)
    pub fn whir_query_security(
        soundness_type: SoundnessType,
        num_queries: usize,
        log_inv_rate: usize,
    ) -> f64 {
        match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1.0 / (1 << log_inv_rate) as f64;
                let denom = (0.5 * (1.0 + rate)).log2();
                -(num_queries as f64) * denom
            }
        }
    }

    /// Computes WHIR proximity gap security bits.
    ///
    /// Per WHIR paper: err*(C, ℓ, δ) = (ℓ - 1) · 2^m / (ρ · |F|)
    /// For ℓ = 2 (batching 2 things in fold): err* = 2^m / (ρ · |F|)
    ///
    /// In unique decoding regime:
    /// Security bits = |F_ext| - log₂(degree) - log₂(1/rate) = |F_ext| - log_degree - log_inv_rate
    pub fn whir_proximity_gap_security(
        soundness_type: SoundnessType,
        challenge_field_bits: f64,
        log_degree: usize,
        log_inv_rate: usize,
    ) -> f64 {
        match soundness_type {
            SoundnessType::UniqueDecoding => {
                challenge_field_bits - log_degree as f64 - log_inv_rate as f64
            }
        }
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
) {
    let soundness = SoundnessAnalysis::calculate(
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
    println!("  WHIR (min across sources):   {:.1}", soundness.whir_bits);
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
    println!("  OOD error:            {:.1} bits", whir.ood_bits);
    println!(
        "  γ batching error:     {:.1} bits",
        whir.gamma_batching_bits
    );
    println!("  μ batching error:     {:.1} bits", whir.mu_batching_bits);
    println!();

    println!("WHIR Round Breakdown:");
    let k_whir = params.whir.k;
    let mut log_inv_rate = params.log_blowup;
    for (round, round_config) in params.whir.rounds.iter().enumerate() {
        let query_sec = SoundnessAnalysis::whir_query_security(
            SoundnessType::UniqueDecoding,
            round_config.num_queries,
            log_inv_rate,
        );
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
    soundness_type: SoundnessType,
    target_security_bits: usize,
    log_inv_rate: usize,
) -> usize {
    WhirConfig::queries(soundness_type, target_security_bits, log_inv_rate)
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::interaction::LogUpSecurityParameters;
    use openvm_stark_sdk::config::log_up_params::log_up_security_params_baby_bear_100_bits;

    use super::*;
    use crate::config::WhirRoundConfig;

    // ==========================================================================
    // Test fixtures
    // ==========================================================================

    fn test_params() -> SystemParams {
        SystemParams {
            l_skip: 3,
            n_stack: 8,
            log_blowup: 1,
            whir: WhirConfig {
                k: 4,
                rounds: vec![
                    WhirRoundConfig { num_queries: 36 },
                    WhirRoundConfig { num_queries: 18 },
                ],
                mu_pow_bits: 20,
                query_phase_pow_bits: 16,
                folding_pow_bits: 10,
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
    // Production configurations
    //
    // These must be kept in sync with v2-proof-system/crates/sdk/src/config.rs:
    //   - default_app_params()
    //   - default_leaf_params()
    //   - default_internal_params()
    //   - default_compression_params()
    //
    // When production params change in the SDK, update these values accordingly.
    // ==========================================================================

    // From SDK config.rs constants
    const WHIR_MAX_LOG_FINAL_POLY_LEN: usize = 10;
    const WHIR_POW_BITS: usize = 20;
    const SECURITY_LEVEL: usize = 100;

    // From SDK config.rs: DEFAULT_APP_L_SKIP, DEFAULT_APP_LOG_BLOWUP, etc.
    const DEFAULT_APP_L_SKIP: usize = 4;
    const DEFAULT_APP_LOG_BLOWUP: usize = 1;
    const DEFAULT_LEAF_LOG_BLOWUP: usize = 2;
    const DEFAULT_INTERNAL_LOG_BLOWUP: usize = 2;
    const DEFAULT_COMPRESSION_LOG_BLOWUP: usize = 4;

    fn production_system_params(
        log_blowup: usize,
        l_skip: usize,
        n_stack: usize,
        log_final_poly_len: usize,
    ) -> SystemParams {
        let k_whir = 4;
        let max_constraint_degree = 4;
        let log_stacked_height = l_skip + n_stack;

        SystemParams {
            l_skip,
            n_stack,
            log_blowup,
            whir: WhirConfig::new(
                log_blowup,
                log_stacked_height,
                crate::config::WhirParams {
                    k: k_whir,
                    log_final_poly_len,
                    query_phase_pow_bits: WHIR_POW_BITS,
                },
                SECURITY_LEVEL,
            ),
            logup: log_up_security_params_baby_bear_100_bits(),
            max_constraint_degree,
        }
    }

    /// App VM params: from SDK default_app_params(DEFAULT_APP_LOG_BLOWUP, DEFAULT_APP_L_SKIP, 20)
    fn app_params() -> SystemParams {
        production_system_params(
            DEFAULT_APP_LOG_BLOWUP,
            DEFAULT_APP_L_SKIP,
            20,
            WHIR_MAX_LOG_FINAL_POLY_LEN,
        )
    }

    /// Leaf params: from SDK default_leaf_params(DEFAULT_LEAF_LOG_BLOWUP)
    /// l_skip=2, n_stack=18
    fn leaf_params() -> SystemParams {
        production_system_params(DEFAULT_LEAF_LOG_BLOWUP, 2, 18, WHIR_MAX_LOG_FINAL_POLY_LEN)
    }

    /// Internal params: from SDK default_internal_params(DEFAULT_INTERNAL_LOG_BLOWUP)
    /// l_skip=2, n_stack=17
    fn internal_params() -> SystemParams {
        production_system_params(
            DEFAULT_INTERNAL_LOG_BLOWUP,
            2,
            17,
            WHIR_MAX_LOG_FINAL_POLY_LEN,
        )
    }

    /// Compression params: from SDK default_compression_params(DEFAULT_COMPRESSION_LOG_BLOWUP)
    /// l_skip=2, n_stack=20, log_final_poly_len=11 (different from others!)
    fn compression_params() -> SystemParams {
        production_system_params(DEFAULT_COMPRESSION_LOG_BLOWUP, 2, 20, 11)
    }

    // ==========================================================================
    // Circuit parameter upper bounds for soundness analysis
    //
    // These are conservative estimates based on actual production values.
    // Stacking can only reduce width, so num_columns is an upper bound on stacked columns.
    // ==========================================================================

    /// Upper bound on n_logup derived from circuit parameters.
    ///
    /// n_logup = ceil_log2(total_interactions) - l_skip, where:
    /// - total_interactions ≤ num_airs × max_interactions_per_air × 2^max_log_height
    /// - total_interactions must fit in u32 (enforced by verifier)
    ///
    /// So: n_logup ≤ min(32 - l_skip, log2(num_airs × max_interactions) + max_log_height - l_skip)
    fn n_logup_bound(
        l_skip: usize,
        num_airs: usize,
        max_interactions_per_air: usize,
        max_log_height: usize,
    ) -> usize {
        let field_bound = 32 - l_skip;
        let param_bound = (num_airs as f64).log2().ceil() as usize
            + (max_interactions_per_air as f64).log2().ceil() as usize
            + max_log_height
            - l_skip;
        field_bound.min(param_bound)
    }

    // App VM: large circuits with many AIRs
    // Actual: max_constraints=4513 (keccak), num_airs=73, max_interactions=832
    const APP_MAX_CONSTRAINTS: usize = 5000;
    const APP_NUM_AIRS: usize = 100;
    const APP_MAX_LOG_HEIGHT: usize = 24;
    const APP_NUM_COLUMNS: usize = 30000;
    const APP_MAX_INTERACTIONS_PER_AIR: usize = 1000;

    // Recursion circuits: smaller, fixed structure
    // Actual: num_airs=42
    const RECURSION_MAX_CONSTRAINTS: usize = 1000;
    const RECURSION_NUM_AIRS: usize = 50;
    const RECURSION_NUM_COLUMNS: usize = 2000;
    const RECURSION_MAX_INTERACTIONS_PER_AIR: usize = 100; // estimate, needs verification

    // ==========================================================================
    // Unit tests
    // ==========================================================================

    #[test]
    fn test_soundness_calculation() {
        let params = test_params();
        let soundness = SoundnessAnalysis::calculate(
            &params,
            challenge_field_bits(),
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
        let queries = min_whir_queries(SoundnessType::UniqueDecoding, 100, 1);
        assert!(queries > 0);
    }

    #[test]
    fn test_logup_soundness() {
        let params = test_params();
        let security =
            SoundnessAnalysis::calculate_logup_soundness(&params, challenge_field_bits());
        assert!(security > 100.0);
    }

    #[test]
    fn test_whir_unique_decoding_security() {
        // rate = 0.5: error = 0.75, security per query ≈ 0.415 bits
        let security =
            SoundnessAnalysis::whir_query_security(SoundnessType::UniqueDecoding, 100, 1);
        assert!(
            (security - 41.5).abs() < 1.0,
            "Expected ~41.5, got {}",
            security
        );

        // rate = 0.25: error = 0.625, security per query ≈ 0.678 bits
        let security_blowup2 =
            SoundnessAnalysis::whir_query_security(SoundnessType::UniqueDecoding, 100, 2);
        assert!(
            (security_blowup2 - 67.8).abs() < 1.0,
            "Expected ~67.8, got {}",
            security_blowup2
        );
    }

    // ==========================================================================
    // Production configuration tests
    // ==========================================================================

    fn check_soundness(
        name: &str,
        params: &SystemParams,
        max_constraints: usize,
        num_airs: usize,
        max_log_height: usize,
        num_columns: usize,
        n_logup: usize,
    ) -> SoundnessAnalysis {
        // num_columns is used for both num_trace_columns and num_stacked_columns.
        // This is conservative since num_trace_columns >= num_stacked_columns
        // (stacking can only reduce width).
        let soundness = SoundnessAnalysis::calculate(
            params,
            challenge_field_bits(),
            max_constraints,
            num_airs,
            params.max_constraint_degree,
            max_log_height,
            num_columns,
            num_columns,
            n_logup,
        );

        println!("\n=== {} Soundness ===", name);
        println!(
            "Config: l_skip={}, n_stack={}, log_blowup={}, k_whir={}",
            params.l_skip, params.n_stack, params.log_blowup, params.whir.k
        );
        println!(
            "Context: max_constraints={}, num_airs={}, max_log_height={}, num_columns={}, n_logup={}",
            max_constraints, num_airs, max_log_height, num_columns, n_logup
        );
        println!();
        println!("LogUp (α/β + PoW):   {:.1} bits", soundness.logup_bits);
        println!(
            "GKR sumcheck:        {:.1} bits",
            soundness.gkr_sumcheck_bits
        );
        println!(
            "GKR batching (μ/λ):  {:.1} bits",
            soundness.gkr_batching_bits
        );
        println!(
            "ZeroCheck sumcheck:  {:.1} bits",
            soundness.zerocheck_sumcheck_bits
        );
        println!(
            "Constraint batching: {:.1} bits",
            soundness.constraint_batching_bits
        );
        println!(
            "Stacked reduction:   {:.1} bits",
            soundness.stacked_reduction_bits
        );
        println!("WHIR:                {:.1} bits", soundness.whir_bits);
        println!("TOTAL:               {:.1} bits", soundness.total_bits);

        println!("\nWHIR Error Source Breakdown:");
        let whir = &soundness.whir_details;
        println!("  Query error:          {:.1} bits", whir.query_bits);
        println!(
            "  Proximity gaps:       {:.1} bits",
            whir.proximity_gaps_bits
        );
        println!("  Sumcheck error:       {:.1} bits", whir.sumcheck_bits);
        println!("  OOD error:            {:.1} bits", whir.ood_bits);
        println!(
            "  γ batching error:     {:.1} bits",
            whir.gamma_batching_bits
        );
        println!("  μ batching error:     {:.1} bits", whir.mu_batching_bits);

        soundness
    }

    #[test]
    fn test_app_vm_security() {
        let params = app_params();
        let n_logup = n_logup_bound(
            params.l_skip,
            APP_NUM_AIRS,
            APP_MAX_INTERACTIONS_PER_AIR,
            APP_MAX_LOG_HEIGHT,
        );
        let soundness = check_soundness(
            "App VM",
            &params,
            APP_MAX_CONSTRAINTS,
            APP_NUM_AIRS,
            APP_MAX_LOG_HEIGHT,
            APP_NUM_COLUMNS,
            n_logup,
        );
        assert!(
            soundness.total_bits >= 100.0,
            "App VM: got {:.1} bits",
            soundness.total_bits
        );
    }

    #[test]
    fn test_leaf_aggregation_security() {
        let params = leaf_params();
        let max_log_height = 20;
        let n_logup = n_logup_bound(
            params.l_skip,
            RECURSION_NUM_AIRS,
            RECURSION_MAX_INTERACTIONS_PER_AIR,
            max_log_height,
        );
        let soundness = check_soundness(
            "Leaf Aggregation",
            &params,
            RECURSION_MAX_CONSTRAINTS,
            RECURSION_NUM_AIRS,
            max_log_height,
            RECURSION_NUM_COLUMNS,
            n_logup,
        );
        assert!(
            soundness.total_bits >= 100.0,
            "Leaf: got {:.1} bits",
            soundness.total_bits
        );
    }

    #[test]
    fn test_internal_aggregation_security() {
        let params = internal_params();
        let max_log_height = 19;
        let n_logup = n_logup_bound(
            params.l_skip,
            RECURSION_NUM_AIRS,
            RECURSION_MAX_INTERACTIONS_PER_AIR,
            max_log_height,
        );
        let soundness = check_soundness(
            "Internal Aggregation",
            &params,
            RECURSION_MAX_CONSTRAINTS,
            RECURSION_NUM_AIRS,
            max_log_height,
            RECURSION_NUM_COLUMNS,
            n_logup,
        );
        assert!(
            soundness.total_bits >= 100.0,
            "Internal: got {:.1} bits",
            soundness.total_bits
        );
    }

    #[test]
    fn test_compression_security() {
        let params = compression_params();
        let max_log_height = 22;
        let n_logup = n_logup_bound(
            params.l_skip,
            RECURSION_NUM_AIRS,
            RECURSION_MAX_INTERACTIONS_PER_AIR,
            max_log_height,
        );
        let soundness = check_soundness(
            "Compression",
            &params,
            RECURSION_MAX_CONSTRAINTS,
            RECURSION_NUM_AIRS,
            max_log_height,
            RECURSION_NUM_COLUMNS,
            n_logup,
        );
        assert!(
            soundness.total_bits >= 100.0,
            "Compression: got {:.1} bits",
            soundness.total_bits
        );
    }

    #[test]
    fn test_all_production_configs() {
        println!("\n========== ALL PRODUCTION CONFIGS ==========");

        let app = app_params();
        let leaf = leaf_params();
        let internal = internal_params();
        let compression = compression_params();

        // (name, params, max_constraints, num_airs, max_log_height, num_columns,
        // max_interactions_per_air)
        let configs: [(&str, &SystemParams, usize, usize, usize, usize, usize); 4] = [
            (
                "App VM",
                &app,
                APP_MAX_CONSTRAINTS,
                APP_NUM_AIRS,
                APP_MAX_LOG_HEIGHT,
                APP_NUM_COLUMNS,
                APP_MAX_INTERACTIONS_PER_AIR,
            ),
            (
                "Leaf",
                &leaf,
                RECURSION_MAX_CONSTRAINTS,
                RECURSION_NUM_AIRS,
                20,
                RECURSION_NUM_COLUMNS,
                RECURSION_MAX_INTERACTIONS_PER_AIR,
            ),
            (
                "Internal",
                &internal,
                RECURSION_MAX_CONSTRAINTS,
                RECURSION_NUM_AIRS,
                19,
                RECURSION_NUM_COLUMNS,
                RECURSION_MAX_INTERACTIONS_PER_AIR,
            ),
            (
                "Compression",
                &compression,
                RECURSION_MAX_CONSTRAINTS,
                RECURSION_NUM_AIRS,
                22,
                RECURSION_NUM_COLUMNS,
                RECURSION_MAX_INTERACTIONS_PER_AIR,
            ),
        ];

        for (
            name,
            params,
            max_constraints,
            num_airs,
            max_log_height,
            num_columns,
            max_interactions,
        ) in configs
        {
            let n_logup = n_logup_bound(params.l_skip, num_airs, max_interactions, max_log_height);
            let soundness = check_soundness(
                name,
                params,
                max_constraints,
                num_airs,
                max_log_height,
                num_columns,
                n_logup,
            );
            assert!(
                soundness.total_bits >= 100.0,
                "{}: got {:.1} bits",
                name,
                soundness.total_bits
            );
        }

        println!("\n========== ALL CONFIGS PASSED ==========");
    }
}
