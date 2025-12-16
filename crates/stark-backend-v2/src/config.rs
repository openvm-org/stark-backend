use getset::Getters;
use openvm_stark_backend::interaction::LogUpSecurityParameters;
use serde::{Deserialize, Serialize};

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
    /// Number of bits of grinding for the query phase of each WHIR round.
    /// The PoW bits can vary per round, but for simplicity we use the same number for all rounds.
    pub query_phase_pow_bits: usize,
    /// Number of bits of grinding before sampling folding randomness in each WHIR round.
    /// The folding PoW bits can vary per round, but for simplicity (and efficiency of the
    /// recursion circuit) we use the same number for all rounds.
    pub folding_pow_bits: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhirRoundConfig {
    pub num_queries: usize,
}

/// Defines the soundness type for the proof system.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SoundnessType {
    /// Unique decoding guarantees a single valid witness.
    UniqueDecoding,
}

impl WhirConfig {
    /// Sets parameters targeting 100-bits of provable security, with grinding, using the unique
    /// decoding regime.
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

        // A safe setting for BabyBear and ~200 columns
        // TODO[jpw]: use rbr_soundness_queries_combination
        const FOLDING_POW_BITS: usize = 10;

        let mut round_parameters = Vec::with_capacity(num_rounds);
        for _round in 0..num_rounds {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let next_rate = log_inv_rate + (k_whir - 1);

            let num_queries = Self::queries(
                SoundnessType::UniqueDecoding,
                protocol_security_level,
                log_inv_rate,
            );
            round_parameters.push(WhirRoundConfig { num_queries });

            log_inv_rate = next_rate;
        }

        Self {
            k: k_whir,
            rounds: round_parameters,
            query_phase_pow_bits,
            folding_pow_bits: FOLDING_POW_BITS,
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
        soundness_type: SoundnessType,
        protocol_security_level: usize,
        log_inv_rate: usize,
    ) -> usize {
        let num_queries_f = match soundness_type {
            SoundnessType::UniqueDecoding => {
                let rate = 1. / f64::from(1 << log_inv_rate);
                let denom = (0.5 * (1. + rate)).log2();

                -(protocol_security_level as f64) / denom
            }
        };
        num_queries_f.ceil() as usize
    }
}
