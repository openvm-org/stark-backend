use p3_util::log2_ceil_usize;

use super::{base_field_order, challenge_field_bits, SoundnessCalculator};
use crate::{keygen::types::MultiStarkVerifyingKey, StarkProtocolConfig};

impl SoundnessCalculator {
    /// Computes a conservative soundness estimate for the verifier defined by the given `vk`.
    ///
    /// The verifying key does not fix a single proof shape: optional AIRs may be absent and trace
    /// heights can vary within verifier-enforced bounds. This function therefore uses verifier-side
    /// upper bounds derived from the key.
    pub fn calculate_from_vk<SC: StarkProtocolConfig>(vk: &MultiStarkVerifyingKey<SC>) -> Self {
        let params = &vk.inner.params;
        let num_airs = vk.inner.per_air.len();
        let mut max_constraints_per_air = 0;
        let mut max_interactions_per_air = 0;
        let mut num_trace_columns_bound = 0;

        for air_vk in &vk.inner.per_air {
            max_constraints_per_air = max_constraints_per_air
                .max(air_vk.symbolic_constraints.constraints.constraint_idx.len());
            max_interactions_per_air =
                max_interactions_per_air.max(air_vk.symbolic_constraints.interactions.len());
            num_trace_columns_bound += air_vk.params.width.total_width();
        }

        let n_logup = calculate_n_logup_bound_from_interaction_limit(
            params.l_skip,
            params.logup.max_interaction_count as usize,
        )
        .min(calculate_n_logup_bound_from_vk_shape(
            params.l_skip,
            num_airs,
            max_interactions_per_air,
            params.log_stacked_height(),
        ));

        Self::calculate(
            params,
            base_field_order::<SC>(),
            challenge_field_bits::<SC>(),
            max_constraints_per_air,
            num_airs,
            params.max_constraint_degree,
            params.log_stacked_height(),
            num_trace_columns_bound,
            params.w_stack,
            n_logup,
        )
    }
}

fn calculate_n_logup_bound_from_interaction_limit(
    l_skip: usize,
    max_interaction_count: usize,
) -> usize {
    if max_interaction_count == 0 {
        return 0;
    }
    log2_ceil_usize(max_interaction_count).saturating_sub(l_skip)
}

fn calculate_n_logup_bound_from_vk_shape(
    l_skip: usize,
    num_airs: usize,
    max_interactions_per_air: usize,
    max_log_trace_height: usize,
) -> usize {
    if num_airs == 0 || max_interactions_per_air == 0 {
        return 0;
    }

    (log2_ceil_usize(num_airs) + log2_ceil_usize(max_interactions_per_air) + max_log_trace_height)
        .saturating_sub(l_skip)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing};

    use crate::{
        air_builders::symbolic::{
            SymbolicConstraintsDag, SymbolicExpressionDag, SymbolicExpressionNode,
        },
        hasher::MerkleHasher,
        interaction::Interaction,
        keygen::types::{
            MultiStarkVerifyingKey, MultiStarkVerifyingKey0, StarkVerifyingKey,
            StarkVerifyingParams, TraceWidth,
        },
        soundness::{base_field_order, challenge_field_bits, SoundnessCalculator},
        test_utils::default_test_params_small,
        StarkProtocolConfig,
    };

    #[derive(Clone, Debug)]
    struct DummyHasher;

    impl MerkleHasher for DummyHasher {
        type F = BabyBear;
        type Digest = [BabyBear; 1];

        fn hash_slice(&self, _vals: &[Self::F]) -> Self::Digest {
            [BabyBear::ZERO]
        }

        fn compress(&self, _left: Self::Digest, _right: Self::Digest) -> Self::Digest {
            [BabyBear::ZERO]
        }
    }

    #[derive(Clone, Debug)]
    struct DummyConfig {
        params: crate::SystemParams,
    }

    impl StarkProtocolConfig for DummyConfig {
        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;
        type Digest = [BabyBear; 1];
        type Hasher = DummyHasher;

        fn params(&self) -> &crate::SystemParams {
            &self.params
        }

        fn hasher(&self) -> &Self::Hasher {
            static HASHER: DummyHasher = DummyHasher;
            &HASHER
        }
    }

    fn constraints_with_counts(
        num_constraints: usize,
        num_interactions: usize,
    ) -> SymbolicConstraintsDag<BabyBear> {
        let mut nodes = Vec::new();
        let mut constraint_idx = Vec::new();
        for i in 0..num_constraints {
            nodes.push(SymbolicExpressionNode::Constant(BabyBear::from_usize(
                i + 1,
            )));
            constraint_idx.push(i);
        }
        let base_idx = nodes.len();
        let interactions = (0..num_interactions)
            .map(|i| {
                nodes.push(SymbolicExpressionNode::Constant(BabyBear::from_usize(
                    i + 11,
                )));
                nodes.push(SymbolicExpressionNode::Constant(BabyBear::ONE));
                Interaction {
                    message: vec![base_idx + 2 * i],
                    count: base_idx + 2 * i + 1,
                    bus_index: 0,
                    count_weight: 0,
                }
            })
            .collect();

        SymbolicConstraintsDag {
            constraints: SymbolicExpressionDag {
                nodes,
                constraint_idx,
            },
            interactions,
        }
    }

    fn test_vk() -> MultiStarkVerifyingKey<DummyConfig> {
        let params = default_test_params_small();
        MultiStarkVerifyingKey::<DummyConfig> {
            inner: MultiStarkVerifyingKey0 {
                params,
                per_air: vec![
                    StarkVerifyingKey {
                        preprocessed_data: None,
                        params: StarkVerifyingParams {
                            width: TraceWidth {
                                preprocessed: None,
                                cached_mains: vec![],
                                common_main: 2,
                            },
                            num_public_values: 3,
                            need_rot: false,
                        },
                        symbolic_constraints: constraints_with_counts(4, 0),
                        max_constraint_degree: 1,
                        is_required: true,
                        unused_variables: vec![],
                    },
                    StarkVerifyingKey {
                        preprocessed_data: None,
                        params: StarkVerifyingParams {
                            width: TraceWidth {
                                preprocessed: None,
                                cached_mains: vec![],
                                common_main: 2,
                            },
                            num_public_values: 0,
                            need_rot: false,
                        },
                        symbolic_constraints: constraints_with_counts(0, 1),
                        max_constraint_degree: 1,
                        is_required: true,
                        unused_variables: vec![],
                    },
                    StarkVerifyingKey {
                        preprocessed_data: None,
                        params: StarkVerifyingParams {
                            width: TraceWidth {
                                preprocessed: None,
                                cached_mains: vec![],
                                common_main: 2,
                            },
                            num_public_values: 0,
                            need_rot: false,
                        },
                        symbolic_constraints: constraints_with_counts(0, 1),
                        max_constraint_degree: 1,
                        is_required: true,
                        unused_variables: vec![],
                    },
                ],
                trace_height_constraints: vec![crate::keygen::types::LinearConstraint {
                    coefficients: vec![0, 1, 1],
                    threshold: 1 << 30,
                }],
            },
            pre_hash: [BabyBear::ZERO],
        }
    }

    #[test]
    fn calculates_vk_soundness_from_verifier_bounds() {
        let vk = test_vk();
        let params = vk.inner.params.clone();
        let soundness = SoundnessCalculator::calculate_from_vk(&vk);

        let expected = SoundnessCalculator::calculate(
            &params,
            base_field_order::<DummyConfig>(),
            challenge_field_bits::<DummyConfig>(),
            4,
            3,
            params.max_constraint_degree,
            params.log_stacked_height(),
            6,
            params.w_stack,
            10,
        );

        assert_eq!(soundness.total_bits, expected.total_bits);
        assert_eq!(
            soundness.constraint_batching_bits,
            expected.constraint_batching_bits
        );
    }
}
