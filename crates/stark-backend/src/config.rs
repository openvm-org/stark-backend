use core::fmt::Debug;

use getset::Getters;
use p3_field::{BasedVectorSpace, ExtensionField, PrimeField64, TwoAdicField};
use serde::{Deserialize, Serialize};

use crate::{hasher::MerkleHasher, interaction::LogUpSecurityParameters};

pub const DEFAULT_K_WHIR: usize = 4;

/// Trait that holds the associated types for the SWIRL protocol. These are the types needed by the
/// verifier and must be independent of the prover backend.
///
/// This trait only holds the associated types and the struct implementing the trait does not hold
/// the system parameters. System parameters are specified and stored separated in [SystemParams].
///
/// The trait has an **implicit** associated Fiat-Shamir transcript type, including the hash used.
/// There is no explicit associated type because the concrete implementation of the transcript may
/// differ between prover and verifier and the verifier may further employ different implementations
/// for logging or debugging purposes. The trait controlling concrete implementations of the
/// transcript is specified by [`FiatShamirTranscript`](crate::FiatShamirTranscript).
pub trait StarkProtocolConfig: 'static + Clone + Send + Sync {
    /// The prime base field.
    type F: TwoAdicField + PrimeField64;
    /// The extension field, used for random challenges.
    type EF: TwoAdicField + ExtensionField<Self::F>;
    /// The digest type used for commitments.
    type Digest: Copy
        + Send
        + Sync
        + Debug
        + Default
        + PartialEq
        + Eq
        + Serialize
        + for<'de> Deserialize<'de>;
    /// The merkle tree hasher used by the polynomial commitment scheme.
    type Hasher: MerkleHasher<F = Self::F, Digest = Self::Digest>;

    /// Degree of the extension field.
    const D_EF: usize = <Self::EF as BasedVectorSpace<Self::F>>::DIMENSION;

    fn params(&self) -> &SystemParams;

    fn hasher(&self) -> &Self::Hasher;
}

/// Type alias for backwards compatibility. New implementations should use `SC::F`.
pub type Val<SC> = <SC as StarkProtocolConfig>::F;
/// Type alias for backwards compatibility. New implementations should use `SC::Digest`.
pub type Com<SC> = <SC as StarkProtocolConfig>::Digest;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Getters)]
pub struct SystemParams {
    pub l_skip: usize,
    pub n_stack: usize,
    /// Maximum number of stacked polynomials (i.e. stacked matrix width). This implies a max
    /// stacked cell count of `w_stack * 2^(n_stack + l_skip)`.
    pub w_stack: usize,
    /// `-log_2` of the rate for the initial Reed-Solomon code.
    pub log_blowup: usize,
    #[getset(get = "pub")]
    pub whir: WhirConfig,
    pub logup: LogUpSecurityParameters,
    /// Global max constraint degree enforced across all AIR and Interaction constraints
    pub max_constraint_degree: usize,
}

impl SystemParams {
    pub fn logup_pow_bits(&self) -> usize {
        self.logup.pow_bits
    }

    pub fn k_whir(&self) -> usize {
        self.whir.k
    }

    #[inline]
    pub fn log_stacked_height(&self) -> usize {
        self.l_skip + self.n_stack
    }

    #[inline]
    pub fn log_final_poly_len(&self) -> usize {
        self.whir.log_final_poly_len(self.log_stacked_height())
    }

    #[inline]
    pub fn num_whir_rounds(&self) -> usize {
        self.whir.num_whir_rounds()
    }

    #[inline]
    pub fn num_whir_sumcheck_rounds(&self) -> usize {
        self.whir.num_sumcheck_rounds()
    }

    /// Constructor with many configuration options. Only `k_whir`, `max_constraint_degree`,
    /// `query_phase_pow_bits` are preset with constants.
    ///
    /// The `security_bits` is the target bits of security. It is used to select the number of WHIR
    /// queries, but does **not** guarantee the target security level is achieved by the overall
    /// protocol using these parameters. Use the soundness calculator to ensure the target security
    /// level is met.
    ///
    /// This function should only be used for internal cryptography libraries. Most users should
    /// instead use preset parameters provided in SDKs.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        log_blowup: usize,
        l_skip: usize,
        n_stack: usize,
        w_stack: usize,
        log_final_poly_len: usize,
        folding_pow_bits: usize,
        mu_pow_bits: usize,
        proximity: WhirProximityStrategy,
        security_bits: usize,
        logup: LogUpSecurityParameters,
    ) -> SystemParams {
        const WHIR_QUERY_PHASE_POW_BITS: usize = 20;

        let k_whir = DEFAULT_K_WHIR;
        let max_constraint_degree = 4;
        let log_stacked_height = l_skip + n_stack;

        SystemParams {
            l_skip,
            n_stack,
            w_stack,
            log_blowup,
            whir: WhirConfig::new(
                log_blowup,
                log_stacked_height,
                WhirParams {
                    k: k_whir,
                    log_final_poly_len,
                    query_phase_pow_bits: WHIR_QUERY_PHASE_POW_BITS,
                    proximity,
                    folding_pow_bits,
                    mu_pow_bits,
                },
                security_bits,
            ),
            logup,
            max_constraint_degree,
        }
    }
}

/// Configurable parameters that are used to determine the [WhirConfig] for a target security level.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhirParams {
    pub k: usize,
    /// WHIR rounds will stop as soon as `log2` of the final polynomial length is `<=
    /// log_final_poly_len`.
    pub log_final_poly_len: usize,
    pub query_phase_pow_bits: usize,
    /// Proximity regime to use within each WHIR round. This is used to determine the number of
    /// queries in each round for a target security level.
    pub proximity: WhirProximityStrategy,
    /// Number of bits of grinding to increase security of each folding step.
    pub folding_pow_bits: usize,
    /// Number of bits of grinding before sampling the μ batching challenge.
    pub mu_pow_bits: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhirConfig {
    /// Constant folding factor. This means that `2^k` terms are folded per round.
    pub k: usize,
    pub rounds: Vec<WhirRoundConfig>,
    /// Number of bits of grinding before sampling the μ batching challenge.
    pub mu_pow_bits: usize,
    /// Number of bits of grinding for the query phase of each WHIR round.
    /// The PoW bits can vary per round, but for simplicity we use the same number for all rounds.
    pub query_phase_pow_bits: usize,
    /// Number of bits of grinding before sampling folding randomness in each WHIR round.
    /// The folding PoW bits can vary per round, but for simplicity (and efficiency of the
    /// recursion circuit) we use the same number for all rounds.
    pub folding_pow_bits: usize,
    /// Proximity regimes for WHIR rounds. We use only proven error bounds.
    ///
    /// Note: this field is not needed by the verifier as it is only used to determine the number
    /// of queries in the `rounds` field. However we store it in `WhirConfig` for security analysis
    /// purposes.
    pub proximity: WhirProximityStrategy,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhirRoundConfig {
    /// Number of in-domain queries sampled in this WHIR round.
    pub num_queries: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum WhirProximityStrategy {
    /// Unique decoding regime in every WHIR round.
    UniqueDecoding,
    /// Unique decoding regime in the initial `list_start_round` WHIR rounds. Then list decoding
    /// regime with the same proximity radius derived from `m` (see `ListDecoding`) for all
    /// subsequent WHIR rounds (0-indices `>= list_start_round`). Note that a WHIR round
    /// consists of codewords of different degrees with the same rate. The WHIR round changes
    /// when the rate changes.
    SplitUniqueList { m: usize, list_start_round: usize },
    /// List decoding regime in every WHIR round, with the same proximity radius derived from `m`,
    /// where `m = ceil(sqrt(\rho) / 2 \eta), \eta = 1 - sqrt(\rho) - \delta, where \delta is the
    /// proximity radius.
    ListDecoding { m: usize },
}

impl WhirProximityStrategy {
    pub fn initial_round(&self) -> ProximityRegime {
        self.in_round(0)
    }

    pub fn in_round(&self, whir_round: usize) -> ProximityRegime {
        match *self {
            Self::UniqueDecoding => ProximityRegime::UniqueDecoding,
            Self::SplitUniqueList {
                m,
                list_start_round,
            } => {
                if whir_round < list_start_round {
                    ProximityRegime::UniqueDecoding
                } else {
                    ProximityRegime::ListDecoding { m }
                }
            }
            Self::ListDecoding { m } => ProximityRegime::ListDecoding { m },
        }
    }
}

/// Defines the proximity regime for the proof system.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProximityRegime {
    /// Unique decoding guarantees a single valid witness.
    UniqueDecoding,
    /// List decoding bounded by multiplicity `m`.
    ListDecoding { m: usize },
}

impl ProximityRegime {
    /// Returns total security bits for `num_queries` WHIR queries.
    ///
    /// This treats the per-query error as an upper bound on the maximum agreement.
    ///
    /// - `UniqueDecoding`: max agreement is `(1 + ρ) / 2`.
    /// - `ListDecoding { m }`: finite-multiplicity Guruswami-Sudan threshold, `sqrt(ρ) (1 +
    ///   1/(2m)).
    pub fn whir_query_security_bits(&self, num_queries: usize, log_inv_rate: usize) -> f64 {
        let rho = 2.0_f64.powf(-(log_inv_rate as f64));
        let max_agreement = match *self {
            ProximityRegime::UniqueDecoding => (1.0 + rho) / 2.0,
            ProximityRegime::ListDecoding { m } => {
                let m = m.max(1) as f64;
                // The stronger bound of sqrt(ρ(1 + 1/m)) could be used for the Guruswami-Sudan
                // threshold, but to match the explicit statement in the WHIR paper, we use the
                // weaker Taylor series expansion to sqrt(ρ) * (1 + 1/(2m))
                rho.sqrt() * (1.0 + 1.0 / (2.0 * m))
            }
        };

        // Keep the `log2` well-defined.
        let max_agreement = max_agreement.clamp(f64::MIN_POSITIVE, 1.0);
        -(num_queries as f64) * max_agreement.log2()
    }

    /// Returns the per-query security bits for WHIR query sampling.
    pub fn whir_per_query_security_bits(&self, log_inv_rate: usize) -> f64 {
        self.whir_query_security_bits(1, log_inv_rate)
    }
}

impl WhirConfig {
    /// Sets parameters targeting `security_bits` of provable security (including grinding), using
    /// the given proximity regime.
    pub fn new(
        log_blowup: usize,
        log_stacked_height: usize,
        whir_params: WhirParams,
        security_bits: usize,
    ) -> Self {
        let query_phase_pow_bits = whir_params.query_phase_pow_bits;
        let protocol_security_level = security_bits.saturating_sub(query_phase_pow_bits);
        let k_whir = whir_params.k;
        let num_rounds = log_stacked_height
            .saturating_sub(whir_params.log_final_poly_len)
            .div_ceil(k_whir);
        let mut log_inv_rate = log_blowup;
        let proximity = whir_params.proximity;

        let mut round_parameters = Vec::with_capacity(num_rounds);
        for round in 0..num_rounds {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let next_rate = log_inv_rate + (k_whir - 1);

            let num_queries = Self::queries(
                proximity.in_round(round),
                protocol_security_level,
                log_inv_rate,
            );
            round_parameters.push(WhirRoundConfig { num_queries });

            log_inv_rate = next_rate;
        }

        Self {
            k: k_whir,
            rounds: round_parameters,
            mu_pow_bits: whir_params.mu_pow_bits,
            query_phase_pow_bits,
            folding_pow_bits: whir_params.folding_pow_bits,
            proximity,
        }
    }

    #[inline]
    pub fn log_final_poly_len(&self, log_stacked_height: usize) -> usize {
        log_stacked_height - self.num_whir_rounds() * self.k
    }

    pub fn num_whir_rounds(&self) -> usize {
        self.rounds.len()
    }

    #[inline]
    pub fn num_sumcheck_rounds(&self) -> usize {
        self.num_whir_rounds() * self.k
    }

    /// Pure function to calculate the number of queries necessary for a given WHIR round.
    /// - `protocol_security_level` refers to the target bits of security without grinding.
    /// - `log_inv_rate` is the log blowup for the WHIR round we want to calculate the number of
    ///   queries for.
    // Source: https://github.com/WizardOfMenlo/whir/blob/cf1599b56ff50e09142ebe6d2e2fbd86875c9986/src/whir/parameters.rs#L457
    pub fn queries(
        proximity_regime: ProximityRegime,
        protocol_security_level: usize,
        log_inv_rate: usize,
    ) -> usize {
        let per_query_bits = proximity_regime.whir_per_query_security_bits(log_inv_rate);
        let num_queries_f = (protocol_security_level as f64) / per_query_bits;
        num_queries_f.ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whir_list_decoding_query_bits_monotone_in_num_queries() {
        let regime = ProximityRegime::ListDecoding { m: 2 };
        let sec_10 = regime.whir_query_security_bits(10, 1);
        let sec_20 = regime.whir_query_security_bits(20, 1);
        assert!(sec_20 > sec_10);
        assert!(sec_10 > 0.0);
    }
}
