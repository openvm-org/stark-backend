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
    pub whir: WhirParams,
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhirParams {
    /// Constant folding factor. This means that `2^k` terms are folded per round.
    pub k: usize,
    pub rounds: Vec<WhirRoundParams>,
    /// Number of bits of grinding for the query phase of each WHIR round.
    /// The PoW bits can vary per round, but for simplicity we use the same number for all rounds.
    pub query_phase_pow_bits: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WhirRoundParams {
    pub folding_pow_bits: usize,
    pub num_queries: usize,
}

/// Defines the soundness type for the proof system.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SoundnessType {
    /// Unique decoding guarantees a single valid witness.
    UniqueDecoding,
}

impl WhirParams {
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
