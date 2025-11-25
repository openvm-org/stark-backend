use openvm_stark_backend::interaction::LogUpSecurityParameters;
use serde::{Deserialize, Serialize};

use crate::config::log_up_params::log_up_security_params_baby_bear_100_bits;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FriParameters {
    pub log_blowup: usize,
    pub log_final_poly_len: usize,
    pub num_queries: usize,
    pub query_proof_of_work_bits: usize,
    pub commit_proof_of_work_bits: usize,
}

impl FriParameters {
    /// Conjectured bits of security.
    /// See ethSTARK paper (<https://eprint.iacr.org/2021/582.pdf>) section 5.10.1 equation (19)
    ///
    /// `challenge_field_bits` is the number of bits in the challenge field (extension field) of the
    /// STARK config.
    pub fn get_conjectured_security_bits(&self, challenge_field_bits: usize) -> usize {
        let fri_query_security_bits =
            self.num_queries * self.log_blowup + self.query_proof_of_work_bits;
        // The paper says min(fri_field_bits, fri_query_security_bits) - 1 but plonky2 (https://github.com/0xPolygonZero/plonky2/blob/41dc325e61ab8d4c0491e68e667c35a4e8173ffa/starky/src/config.rs#L86C1-L87C1) omits the -1
        challenge_field_bits.min(fri_query_security_bits)
    }

    pub fn standard_fast() -> Self {
        standard_fri_params_with_100_bits_security(1)
    }

    #[deprecated(note = "use standard_with_100_bits_security instead")]
    pub fn standard_with_100_bits_conjectured_security(log_blowup: usize) -> Self {
        #[allow(deprecated)]
        standard_fri_params_with_100_bits_conjectured_security(log_blowup)
    }

    pub fn standard_with_100_bits_security(log_blowup: usize) -> Self {
        standard_fri_params_with_100_bits_security(log_blowup)
    }

    pub fn max_constraint_degree(&self) -> usize {
        (1 << self.log_blowup) + 1
    }

    /// New FRI parameters for testing usage with the specific `log_blowup`.
    /// If the environment variable `OPENVM_FAST_TEST` is set to "1", then the parameters are **not
    /// secure** and meant for fast testing only.
    ///
    /// In production, use `Self::standard_with_100_bits_security` instead.
    pub fn new_for_testing(log_blowup: usize) -> Self {
        if let Ok("1") = std::env::var("OPENVM_FAST_TEST").as_deref() {
            Self {
                log_blowup,
                log_final_poly_len: 0,
                num_queries: 2,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
            }
        } else {
            Self::standard_with_100_bits_security(log_blowup)
        }
    }
}

/// Pre-defined FRI parameters with 100 bits of provable security, meaning we do
/// not rely on any conjectures about Reed–Solomon codes (e.g., about proximity
/// gaps) or the ethSTARK Toy Problem Conjecture.
///
/// The value `num_queries` is chosen so that the verifier accepts a δ-far
/// codeword for δ = (1 - 2**(-log_blowup)) with probability at most 2^{-80}.
/// I.e., we target the unique-decoding radius. We require 20 PoW bits
/// just before the query phase begins to boost the soundness to 100 bits.
///
/// Assumes that:
/// - the challenge field has size at least 2^123
/// - for `log_blowup = 1`, multi-FRI will be run with at most width 30000 at any level
/// - for `log_blowup > 1`, multi-FRI will be run with at most width 2000 at any level
pub fn standard_fri_params_with_100_bits_security(log_blowup: usize) -> FriParameters {
    let fri_params = match log_blowup {
        1 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 193,
            proof_of_work_bits: 20,
        },
        2 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 118,
            proof_of_work_bits: 20,
        },
        3 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 97,
            proof_of_work_bits: 20,
        },
        4 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 88,
            proof_of_work_bits: 20,
        },
        _ => todo!("No standard FRI params defined for log blowup {log_blowup}",),
    };
    tracing::debug!("FRI parameters | log_blowup: {log_blowup:<2} | num_queries: {:<2} | proof_of_work_bits: {:<2}", fri_params.num_queries, fri_params.proof_of_work_bits);
    fri_params
}

/// Pre-defined FRI parameters with 100 bits of conjectured security.
/// Security bits calculated following ethSTARK (<https://eprint.iacr.org/2021/582.pdf>) 5.10.1 eq (19)
///
/// Assumes that the challenge field used as more than 100 bits.
#[deprecated(note = "use standard_fri_params_with_100_bits_security instead")]
pub fn standard_fri_params_with_100_bits_conjectured_security(log_blowup: usize) -> FriParameters {
    let fri_params = match log_blowup {
        // plonky2 standard fast config uses num_queries=84: https://github.com/0xPolygonZero/plonky2/blob/41dc325e61ab8d4c0491e68e667c35a4e8173ffa/starky/src/config.rs#L49
        // plonky3's default is num_queries=100, so we will use that. See https://github.com/Plonky3/Plonky3/issues/380 for related security discussion.
        1 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 100,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 16,
        },
        2 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 44,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 16,
        },
        // plonky2 standard recursion config: https://github.com/0xPolygonZero/plonky2/blob/41dc325e61ab8d4c0491e68e667c35a4e8173ffa/plonky2/src/plonk/circuit_data.rs#L101
        3 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 30,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 16,
        },
        4 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 23,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 16,
        },
        _ => todo!("No standard FRI params defined for log blowup {log_blowup}",),
    };
    assert!(fri_params.get_conjectured_security_bits(100) >= 100);
    tracing::debug!("FRI parameters | log_blowup: {log_blowup:<2} | num_queries: {:<2} | commit_pow_bits: {:<2}, query_pow_bits: {:<2}", fri_params.num_queries, fri_params.commit_proof_of_work_bits, fri_params.query_proof_of_work_bits);
    fri_params
}

#[derive(Clone, Debug)]
pub struct SecurityParameters {
    pub fri_params: FriParameters,
    pub log_up_params: LogUpSecurityParameters,
}

impl SecurityParameters {
    pub fn standard_fast() -> Self {
        Self {
            fri_params: FriParameters::standard_fast(),
            log_up_params: log_up_security_params_baby_bear_100_bits(),
        }
    }
    pub fn standard_100_bits_with_fri_log_blowup(log_blowup: usize) -> Self {
        Self {
            fri_params: FriParameters::standard_with_100_bits_security(log_blowup),
            log_up_params: log_up_security_params_baby_bear_100_bits(),
        }
    }
}
