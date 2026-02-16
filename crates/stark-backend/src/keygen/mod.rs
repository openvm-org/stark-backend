use std::{cmp::max, collections::HashMap, sync::Arc};

use itertools::Itertools;
use p3_air::BaseAir;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    air_builders::symbolic::{
        get_symbolic_builder, max_constraint_degree_round0,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraintsDag, SymbolicExpressionNode, SymbolicRapBuilder,
    },
    hasher::MerkleHasher,
    keygen::types::{
        KeygenError, LinearConstraint, MultiStarkProvingKey, MultiStarkVerifyingKey0,
        StarkProvingKey, StarkVerifyingKey, StarkVerifyingParams, TraceWidth,
        VerifierSinglePreprocessedData,
    },
    prover::{
        stacked_pcs::{stacked_commit, StackedPcsData},
        ColMajorMatrix, MatrixDimensions,
    },
    AirRef, AnyAir, StarkProtocolConfig, SystemParams,
};

pub mod types;

struct AirKeygenBuilder<SC: StarkProtocolConfig> {
    pub is_required: bool,
    air: AirRef<SC>,
    prep_keygen_data: PrepKeygenData<SC>,
}

/// Stateful builder to create multi-stark proving and verifying keys
/// for system of multiple RAPs with multiple multi-matrix commitments
pub struct MultiStarkKeygenBuilder<SC: StarkProtocolConfig> {
    pub config: SC,
    /// Information for partitioned AIRs.
    partitioned_airs: Vec<AirKeygenBuilder<SC>>,
}

impl<SC: StarkProtocolConfig> MultiStarkKeygenBuilder<SC> {
    pub fn new(config: SC) -> Self {
        Self {
            config,
            partitioned_airs: vec![],
        }
    }

    pub fn params(&self) -> &SystemParams {
        self.config.params()
    }

    /// Default way to add a single Interactive AIR.
    /// Returns `air_id`
    pub fn add_air(&mut self, air: AirRef<SC>) -> usize {
        self.add_air_impl(air, false)
    }

    pub fn add_required_air(&mut self, air: AirRef<SC>) -> usize {
        self.add_air_impl(air, true)
    }

    #[instrument(level = "debug", skip_all, fields(name = air.name(), is_required = is_required))]
    fn add_air_impl(&mut self, air: AirRef<SC>, is_required: bool) -> usize {
        self.partitioned_airs
            .push(AirKeygenBuilder::new(&self.config, air, is_required));
        self.partitioned_airs.len() - 1
    }

    /// Consume the builder and generate proving key.
    /// The verifying key can be obtained from the proving key.
    pub fn generate_pk(self) -> Result<MultiStarkProvingKey<SC>, KeygenError> {
        let params = self.params().clone();
        let max_constraint_degree = params.max_constraint_degree;
        let pk_per_air: Vec<_> = self
            .partitioned_airs
            .into_iter()
            .map(|keygen_builder| {
                // Second pass: get final constraints, where RAP phase constraints may have changed
                keygen_builder.generate_pk(max_constraint_degree)
            })
            .collect::<Result<Vec<_>, KeygenError>>()?;

        let mut air_max_constraint_degree = 0;
        for pk in pk_per_air.iter() {
            let width = &pk.vk.params.width;
            tracing::info!("{:<20} | Constraint Deg = {:<2} | Prep Cols = {:<2} | Main Cols = {:<8} | {:4} Constraints | {:3} Interactions",
                pk.air_name,
                pk.vk.max_constraint_degree,
                width.preprocessed.unwrap_or(0),
                format!("{:?}",width.main_widths()),
                pk.vk.symbolic_constraints.constraints.constraint_idx.len(),
                pk.vk.symbolic_constraints.interactions.len(),
            );
            air_max_constraint_degree = max(air_max_constraint_degree, pk.vk.max_constraint_degree);
            tracing::debug!(
                "On Buses {:?}",
                pk.vk
                    .symbolic_constraints
                    .interactions
                    .iter()
                    .map(|i| i.bus_index)
                    .collect_vec()
            );
            #[cfg(feature = "metrics")]
            {
                let labels = [("air_name", pk.air_name.clone())];
                metrics::counter!("constraint_deg", &labels)
                    .absolute(pk.vk.max_constraint_degree as u64);
                // column info will be logged by prover later
                metrics::counter!("constraints", &labels)
                    .absolute(pk.vk.symbolic_constraints.constraints.constraint_idx.len() as u64);
                metrics::counter!("interactions", &labels)
                    .absolute(pk.vk.symbolic_constraints.interactions.len() as u64);
            }
        }
        if max_constraint_degree != air_max_constraint_degree as usize {
            tracing::warn!(
            "Actual max constraint degree across all AIRs ({air_max_constraint_degree}) does not match configured max constraint degree ({max_constraint_degree})",
        );
        }

        let num_airs = pk_per_air.len();
        let base_order = SC::F::order().to_u32_digits()[0];
        let mut count_weight_per_air_per_bus_index = HashMap::new();

        let mut num_interactions_per_air: Vec<u32> = Vec::with_capacity(num_airs);
        // We compute the a_i's for the constraints of the form a_0 n_0 + ... + a_{k-1} n_{k-1} <
        // a_k, First the constraints that the total number of interactions on each bus is
        // at most the base field order.
        for (air_idx, pk) in pk_per_air.iter().enumerate() {
            let constraints = &pk.vk.symbolic_constraints;
            num_interactions_per_air.push(constraints.interactions.len().try_into().unwrap());
            for interaction in &constraints.interactions {
                // Also make sure that this of interaction is valid given the security params.
                // +1 because of the bus
                let max_msg_len = params.logup.max_message_length();
                // plus one because of the bus
                let total_message_length = interaction.message.len() + 1;
                assert!(
                    total_message_length <= max_msg_len,
                    "interaction message with bus has length {}, which is more than max {max_msg_len}",
                    total_message_length,
                );

                let b = interaction.bus_index;
                let constraint = count_weight_per_air_per_bus_index
                    .entry(b)
                    .or_insert_with(|| LinearConstraint {
                        coefficients: vec![0; num_airs],
                        threshold: base_order,
                    });
                constraint.coefficients[air_idx] += interaction.count_weight;
            }
        }

        // Sorting by bus index is not necessary, but makes debugging/testing easier.
        let mut trace_height_constraints = count_weight_per_air_per_bus_index
            .into_iter()
            .sorted_by_key(|(bus_index, _)| *bus_index)
            .map(|(_, constraint)| constraint)
            .collect_vec();

        let log_up_security_params = params.logup;

        // Add a constraint for the total number of interactions.
        trace_height_constraints.push(LinearConstraint {
            coefficients: num_interactions_per_air,
            threshold: log_up_security_params.max_interaction_count,
        });

        let pre_vk: MultiStarkVerifyingKey0<SC> = MultiStarkVerifyingKey0 {
            params: params.clone(),
            per_air: pk_per_air.iter().map(|pk| pk.vk.clone()).collect(),
            trace_height_constraints: trace_height_constraints.clone(),
        };
        // To protect against weak Fiat-Shamir, we hash the "pre"-verifying key and include it in
        // the final verifying key. This just needs to commit to the verifying key and does
        // not need to be verified by the verifier, so we just use bincode to serialize it.
        let vk_bytes = bitcode::serialize(&pre_vk).unwrap();
        tracing::debug!("pre-vkey: {} bytes", vk_bytes.len());
        // Purely to get type compatibility and convenience, we hash using the native hash
        let vk_pre_hash = self
            .config
            .hasher()
            .hash_slice(&vk_bytes.into_iter().map(SC::F::from_u8).collect_vec());

        Ok(MultiStarkProvingKey {
            params,
            per_air: pk_per_air,
            trace_height_constraints,
            max_constraint_degree,
            vk_pre_hash,
        })
    }
}

impl<SC: StarkProtocolConfig> AirKeygenBuilder<SC> {
    pub fn new(config: &SC, air: AirRef<SC>, is_required: bool) -> Self {
        let prep_keygen_data = PrepKeygenData::new(config.hasher(), config.params(), air.as_ref());
        Self {
            is_required,
            air,
            prep_keygen_data,
        }
    }

    /// `max_constraint_degree` is the global max constraint degree. If this AIR's constraint degree
    /// exceeds it, an error will be returned.
    pub fn generate_pk(
        self,
        max_constraint_degree: usize,
    ) -> Result<StarkProvingKey<SC>, KeygenError> {
        let air_name = self.air.name();

        let symbolic_builder = self.get_symbolic_builder();
        let width = symbolic_builder.width();
        let num_public_values = symbolic_builder.num_public_values();

        let symbolic_constraints = symbolic_builder.constraints();
        let constraint_degree = symbolic_constraints.max_constraint_degree();
        if constraint_degree > max_constraint_degree {
            return Err(KeygenError::MaxConstraintDegreeExceeded {
                name: air_name.clone(),
                degree: constraint_degree,
                max_degree: max_constraint_degree,
            });
        }

        let Self {
            prep_keygen_data:
                PrepKeygenData {
                    verifier_data: preprocessed_vdata,
                    prover_data: prep_prover_data,
                },
            ..
        } = self;

        let dag = SymbolicConstraintsDag::from(symbolic_constraints);
        let max_uni_constraint_degree = max_constraint_degree_round0(&dag);
        let max_rotation = dag.constraints.max_rotation();
        debug_assert!(max_rotation <= 1);
        let need_rot = max_rotation == 1;
        let vparams = StarkVerifyingParams {
            width,
            num_public_values,
            need_rot,
        };
        assert!(vparams.width.after_challenge.is_empty());

        let unused_variables = find_unused_vars(&dag, &vparams.width, need_rot);
        let vk = StarkVerifyingKey {
            preprocessed_data: preprocessed_vdata,
            params: vparams,
            symbolic_constraints: dag,
            max_constraint_degree: constraint_degree
                .try_into()
                .expect("constraint degree should fit in u8"),
            max_uni_constraint_degree: max_uni_constraint_degree
                .try_into()
                .expect("univariate constraint degree should fit in u8"),
            is_required: self.is_required,
            unused_variables,
        };
        Ok(StarkProvingKey {
            air_name,
            vk,
            preprocessed_data: prep_prover_data,
        })
    }

    pub fn get_symbolic_builder(&self) -> SymbolicRapBuilder<SC::F> {
        let width = TraceWidth {
            preprocessed: self.prep_keygen_data.width(),
            cached_mains: self.air.cached_main_widths(),
            common_main: self.air.common_main_width(),
            after_challenge: vec![],
        };
        get_symbolic_builder(self.air.as_ref(), &width, &[], &[])
    }
}

pub(super) struct PrepKeygenData<SC: StarkProtocolConfig> {
    pub verifier_data: Option<VerifierSinglePreprocessedData<SC::Digest>>,
    pub prover_data: Option<Arc<StackedPcsData<SC::F, SC::Digest>>>,
}

impl<SC: StarkProtocolConfig> PrepKeygenData<SC> {
    fn new(hasher: &SC::Hasher, params: &SystemParams, air: &dyn AnyAir<SC>) -> Self {
        let preprocessed_trace = BaseAir::<SC::F>::preprocessed_trace(air);
        let vpdata_opt = preprocessed_trace.map(|trace| {
            let trace = ColMajorMatrix::from_row_major(&trace);
            let (commit, data) = stacked_commit(
                hasher,
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir(),
                &[&trace],
            );
            debug_assert_eq!(trace.width(), data.mat_view(0).width());
            let vdata = VerifierSinglePreprocessedData {
                commit,
                hypercube_dim: log2_strict_usize(trace.height()) as isize - params.l_skip as isize,
                stacking_width: data.matrix.width(),
            };
            let pdata = Arc::new(data);
            (vdata, pdata)
        });
        if let Some((vdata, pdata)) = vpdata_opt {
            Self {
                prover_data: Some(pdata),
                verifier_data: Some(vdata),
            }
        } else {
            Self {
                prover_data: None,
                verifier_data: None,
            }
        }
    }

    fn width(&self) -> Option<usize> {
        self.prover_data.as_ref().map(|d| d.mat_view(0).width())
    }
}

pub(crate) fn find_unused_vars<F: Field>(
    constraints: &SymbolicConstraintsDag<F>,
    width: &TraceWidth,
    need_rot: bool,
) -> Vec<SymbolicVariable<F>> {
    let preprocessed_width = width.preprocessed.unwrap_or(0);
    let mut preprocessed_present = vec![vec![false; 2]; preprocessed_width];

    let mut main_present = vec![];
    for width in width.main_widths() {
        main_present.push(vec![vec![false; 2]; width]);
    }

    for node in &constraints.constraints.nodes {
        let SymbolicExpressionNode::Variable(var) = node else {
            continue;
        };

        match var.entry {
            Entry::Preprocessed { offset } => {
                preprocessed_present[var.index][offset] = true;
            }
            Entry::Main { part_index, offset } => {
                main_present[part_index][var.index][offset] = true;
            }
            Entry::Public => {}
            Entry::Challenge | Entry::Exposed | Entry::Permutation { .. } => unreachable!(),
        }
    }

    let mut missing = vec![];
    for (index, presents) in preprocessed_present.iter().enumerate() {
        for (offset, present) in presents.iter().enumerate() {
            if !present && (offset == 0 || need_rot) {
                missing.push(SymbolicVariable::new(Entry::Preprocessed { offset }, index));
            }
        }
    }
    for (part_index, present_per_part) in main_present.iter().enumerate() {
        for (index, presents) in present_per_part.iter().enumerate() {
            for (offset, present) in presents.iter().enumerate() {
                if !present && (offset == 0 || need_rot) {
                    missing.push(SymbolicVariable::new(
                        Entry::Main { part_index, offset },
                        index,
                    ));
                }
            }
        }
    }
    missing
}
