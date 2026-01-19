use openvm_stark_backend::{
    config::DeepAliParameters, interaction::LogUpSecurityParameters, p3_field::Field,
};
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

    /// We (via Plonky3) use multi-FRI, whose security in the unique decoding regime can be bounded
    /// above by the security of batch FRI by considering all polynomials in the largest domain.
    ///
    /// We use batch FRI with <=2 opening points (the second point for rotation openings). The
    /// `num_batch_columns` is the number of columns to be batched across all domain sizes. A trace
    /// polynomial over the base field opened at 2 points gets a contribution of 2. A trace
    /// polynomial over the extension field opened at 2 points gets a contribution of 8.
    pub fn security_bits_fri(
        &self,
        challenge_field_bits: f64,
        num_batch_columns: usize,
        max_log_domain_size: usize,
    ) -> f64 {
        let commit_bits = self.security_bits_fri_commit_phase(
            challenge_field_bits,
            num_batch_columns,
            max_log_domain_size,
        );
        tracing::debug!("FRI commit phase security bits: {commit_bits}");
        let query_bits = self.security_bits_fri_query_phase();
        tracing::debug!("FRI query phase security bits: {query_bits}");
        -(0.5_f64.powf(commit_bits) + 0.5_f64.powf(query_bits)).log2()
    }

    /// Batch FRI error according to <https://eprint.iacr.org/2022/1216.pdf> in the unique decoding regime.
    ///
    /// We assume arity-2 folding.
    pub fn security_bits_fri_commit_phase(
        &self,
        challenge_field_bits: f64,
        num_batch_columns: usize,
        max_log_domain_size: usize,
    ) -> f64 {
        // Using formula (1) from https://eprint.iacr.org/2022/1216.pdf on correlated agreement in UDR
        let batch_term = num_batch_columns - 1;
        let fold_term = 2;
        challenge_field_bits + (self.commit_proof_of_work_bits as f64)
            - max_log_domain_size as f64
            - ((batch_term + fold_term) as f64).log2()
    }

    pub fn security_bits_fri_query_phase(&self) -> f64 {
        let rho = (2.0_f64).powi(-(self.log_blowup as i32));
        let theta = (1.0 - rho) / 2.0;
        -(1.0 - theta).powi(self.num_queries as i32).log2() + self.query_proof_of_work_bits as f64
    }

    /// The bits of security from DEEP-ALI.
    /// Note that unlike in Theorem 8 of <https://eprint.iacr.org/2022/1216.pdf>, we do not include the rotation in the denominator of the quotient polynomial. Instead the quotient polynomial denominator consists only of `Z_H(X) = X^{trace_height} - 1`. The rotation opening point is handled as part of FRI PCS opening as an additional opening point. This means that for us `k^+ = k + 1`. We also include degree of selectors within the `max_constraint_degree` parameter.
    pub fn security_bits_deep_ali(
        &self,
        params: &DeepAliParameters,
        challenge_field_bits: f64,
        max_log_domain_size: usize,
        max_constraint_degree: usize,
        max_num_constraints: usize,
    ) -> f64 {
        assert!(max_constraint_degree <= (1 << self.log_blowup) + 1);
        let trace_length = 2.0_f64.powi(max_log_domain_size.saturating_sub(self.log_blowup) as i32);
        let domain_size = 2.0_f64.powi(max_log_domain_size as i32);
        let field_size = 2.0_f64.powf(challenge_field_bits);
        // Schwartz-Zippel applied to constraint polynomial, random out-of-domain point must
        // subgroup H and codeword evaluation coset D
        let e_deep = (max_constraint_degree as f64) * (trace_length - 1.0)
            / (field_size - trace_length - domain_size);
        // ALI error is from algebraic batching
        let e_ali = (max_num_constraints as f64) / field_size;
        -(e_deep + e_ali).log2() + (params.deep_pow_bits as f64)
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
            num_queries: 198,
            commit_proof_of_work_bits: 20,
            query_proof_of_work_bits: 20,
        },
        2 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 120,
            commit_proof_of_work_bits: 17,
            query_proof_of_work_bits: 20,
        },
        3 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 99,
            commit_proof_of_work_bits: 17,
            query_proof_of_work_bits: 20,
        },
        4 => FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            num_queries: 90,
            commit_proof_of_work_bits: 17,
            query_proof_of_work_bits: 20,
        },
        _ => todo!("No standard FRI params defined for log blowup {log_blowup}",),
    };
    tracing::debug!("FRI parameters | log_blowup: {log_blowup:<2} | num_queries: {:<2} | commit_proof_of_work_bits: {:<2} | query_proof_of_work_bits: {:<2}", fri_params.num_queries, fri_params.commit_proof_of_work_bits, fri_params.query_proof_of_work_bits);
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
    pub deep_ali_params: DeepAliParameters,
}

impl SecurityParameters {
    pub fn standard_fast() -> Self {
        Self::new_baby_bear_100_bits(FriParameters::standard_fast())
    }

    pub fn standard_100_bits_with_fri_log_blowup(log_blowup: usize) -> Self {
        Self::new_baby_bear_100_bits(FriParameters::standard_with_100_bits_security(log_blowup))
    }

    pub fn new_baby_bear_100_bits(fri_params: FriParameters) -> Self {
        Self {
            fri_params,
            log_up_params: log_up_security_params_baby_bear_100_bits(),
            deep_ali_params: deep_ali_security_params_baby_bear_100_bits(),
        }
    }

    pub fn security_bits<EF: Field>(
        &self,
        challenge_field_bits: f64,
        num_batch_columns: usize,
        max_log_domain_size: usize,
        max_constraint_degree: usize,
        max_num_constraints: usize,
    ) -> f64 {
        let fri_bits = self.fri_params.security_bits_fri(
            challenge_field_bits,
            num_batch_columns,
            max_log_domain_size,
        );
        tracing::debug!("FRI security bits: {fri_bits}");
        let deep_ali_bits = self.fri_params.security_bits_deep_ali(
            &self.deep_ali_params,
            challenge_field_bits,
            max_log_domain_size,
            max_constraint_degree,
            max_num_constraints,
        );
        tracing::debug!("DEEP-ALI security bits: {deep_ali_bits}");
        let logup_bits = self.log_up_params.bits_of_security::<EF>() as i32;
        tracing::debug!("LogUp security bits: {logup_bits}");

        // We take a union bound of errors following the literature, although we note that in the
        // non-interactive setting, round-by-round soundness may be more appropriate (cf. <https://eprint.iacr.org/2025/1993.pdf>)
        -(0.5_f64.powf(fri_bits) + 0.5_f64.powf(deep_ali_bits) + 0.5_f64.powi(logup_bits)).log2()
    }
}

pub fn deep_ali_security_params_baby_bear_100_bits() -> DeepAliParameters {
    DeepAliParameters { deep_pow_bits: 7 }
}

#[cfg(test)]
mod security_tests {
    use openvm_stark_backend::p3_field::{
        extension::BinomialExtensionField, PrimeField64, TwoAdicField,
    };
    use p3_baby_bear::BabyBear;
    use tracing::Level;

    use super::*;
    use crate::config::setup_tracing_with_log_level;

    #[test]
    fn test_params_provable_security() {
        setup_tracing_with_log_level(Level::DEBUG);

        let challenge_field_bits = (BabyBear::ORDER_U64 as f64).powi(4).log2();
        for log_blowup in 1..=4 {
            let params = SecurityParameters::standard_100_bits_with_fri_log_blowup(log_blowup);
            let num_batch_columns = if log_blowup == 1 { 30_000 } else { 2_000 };
            let max_num_constraints = 15_000;
            let max_constraint_degree = (1 << log_blowup) + 1;
            let max_log_domain_size = BabyBear::TWO_ADICITY;
            let bits = params.security_bits::<BinomialExtensionField<BabyBear, 4>>(
                challenge_field_bits,
                num_batch_columns,
                max_log_domain_size,
                max_constraint_degree,
                max_num_constraints,
            );
            assert!(
                bits >= 100.0,
                "log_blowup: {log_blowup} has {bits} bits of security"
            );
        }
    }
}
