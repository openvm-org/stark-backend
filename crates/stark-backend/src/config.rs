use core::fmt::Debug;

use getset::Getters;
use p3_field::{BasedVectorSpace, ExtensionField, PrimeField64, TwoAdicField};
use serde::{Deserialize, Serialize};

use crate::{hasher::MerkleHasher, interaction::LogUpSecurityParameters};

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
}

/// Configurable parameters that are used to determine the [WhirConfig] for a target security level.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhirParams {
    pub k: usize,
    /// WHIR rounds will stop as soon as `log2` of the final polynomial length is `<=
    /// log_final_poly_len`.
    pub log_final_poly_len: usize,
    pub query_phase_pow_bits: usize,
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
    /// Regime for proximity gaps (unique decoding or list decoding). We use only proven error
    /// bounds.
    pub proximity_regime: ProximityRegime,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhirRoundConfig {
    /// Number of in-domain queries sampled in this WHIR round.
    pub num_queries: usize,
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
    /// - `ListDecoding { m }`: Johnson bound, `sqrt(ρ(1 + 1/m)) + ε` for a tiny `ε`, to keep the
    ///   proximity threshold strict.
    pub fn whir_query_security_bits(&self, num_queries: usize, log_inv_rate: usize) -> f64 {
        let rho = 2.0_f64.powf(-(log_inv_rate as f64));
        let max_agreement = match *self {
            ProximityRegime::UniqueDecoding => (1.0 + rho) / 2.0,
            ProximityRegime::ListDecoding { m } => {
                let m = m.max(1) as f64;
                // Johnson bound with a tiny epsilon to ensure strict proximity.
                (rho * (1.0 + 1.0 / m)).sqrt() + 1e-6
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
        proximity_regime: ProximityRegime,
    ) -> Self {
        let query_phase_pow_bits = whir_params.query_phase_pow_bits;
        let protocol_security_level = security_bits.saturating_sub(query_phase_pow_bits);
        let k_whir = whir_params.k;
        let num_rounds = log_stacked_height
            .saturating_sub(whir_params.log_final_poly_len)
            .div_ceil(k_whir);
        let mut log_inv_rate = log_blowup;

        // A safe setting for BabyBear and ~200 columns
        // TODO[jpw]: use rbr_soundness_queries_combination
        const FOLDING_POW_BITS: usize = 10;

        let mut round_parameters = Vec::with_capacity(num_rounds);
        for _round in 0..num_rounds {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let next_rate = log_inv_rate + (k_whir - 1);

            let num_queries =
                Self::queries(proximity_regime, protocol_security_level, log_inv_rate);
            round_parameters.push(WhirRoundConfig { num_queries });

            log_inv_rate = next_rate;
        }

        const MU_POW_BITS: usize = 20;

        Self {
            k: k_whir,
            rounds: round_parameters,
            mu_pow_bits: MU_POW_BITS,
            query_phase_pow_bits,
            folding_pow_bits: FOLDING_POW_BITS,
            proximity_regime,
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
