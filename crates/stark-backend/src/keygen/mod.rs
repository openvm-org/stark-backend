use std::sync::Arc;

use itertools::Itertools;
use p3_commit::Pcs;
use p3_field::FieldExtensionAlgebra;
use p3_matrix::Matrix;
use tracing::instrument;

use crate::{
    air_builders::symbolic::{get_symbolic_builder, SymbolicRapBuilder},
    config::{Com, StarkGenericConfig, Val},
    interaction::{RapPhaseSeq, RapPhaseSeqKind},
    keygen::types::{
        MultiStarkProvingKey, ProverOnlySinglePreprocessedData, StarkProvingKey, StarkVerifyingKey,
        TraceWidth, VerifierSinglePreprocessedData,
    },
    rap::AnyRap,
};

pub mod types;
pub(crate) mod view;

struct AirKeygenBuilder<SC: StarkGenericConfig> {
    air: Arc<dyn AnyRap<SC>>,
    rap_phase_seq_kind: RapPhaseSeqKind,
    prep_keygen_data: PrepKeygenData<SC>,
}

/// Stateful builder to create multi-stark proving and verifying keys
/// for system of multiple RAPs with multiple multi-matrix commitments
pub struct MultiStarkKeygenBuilder<'a, SC: StarkGenericConfig> {
    pub config: &'a SC,
    /// Information for partitioned AIRs.
    partitioned_airs: Vec<AirKeygenBuilder<SC>>,
    max_constraint_degree: usize,
}

impl<'a, SC: StarkGenericConfig> MultiStarkKeygenBuilder<'a, SC> {
    pub fn new(config: &'a SC) -> Self {
        Self {
            config,
            partitioned_airs: vec![],
            max_constraint_degree: 0,
        }
    }

    pub fn set_max_constraint_degree(&mut self, max_constraint_degree: usize) {
        self.max_constraint_degree = max_constraint_degree;
    }

    /// Default way to add a single Interactive AIR.
    /// Returns `air_id`
    #[instrument(level = "debug", skip_all)]
    pub fn add_air(&mut self, air: Arc<dyn AnyRap<SC>>) -> usize {
        self.partitioned_airs.push(AirKeygenBuilder::new(
            self.config.pcs(),
            SC::RapPhaseSeq::ID,
            air,
        ));
        self.partitioned_airs.len() - 1
    }

    /// Consume the builder and generate proving key.
    /// The verifying key can be obtained from the proving key.
    pub fn generate_pk(mut self) -> MultiStarkProvingKey<SC> {
        let air_max_constraint_degree = self
            .partitioned_airs
            .iter()
            .map(|keygen_builder| {
                let max_constraint_degree = keygen_builder.max_constraint_degree();
                tracing::debug!(
                    "{} has constraint degree {}",
                    keygen_builder.air.name(),
                    max_constraint_degree
                );
                max_constraint_degree
            })
            .max()
            .unwrap();
        tracing::info!(
            "Max constraint (excluding logup constraints) degree across all AIRs: {}",
            air_max_constraint_degree
        );
        if self.max_constraint_degree != 0 {
            assert!(air_max_constraint_degree <= self.max_constraint_degree);
        } else {
            self.max_constraint_degree = air_max_constraint_degree;
        }

        let pk_per_air: Vec<_> = self
            .partitioned_airs
            .iter()
            .map(|keygen_builder| keygen_builder.generate_pk(self.max_constraint_degree))
            .collect();

        for pk in pk_per_air.iter() {
            let width = &pk.vk.params.width;
            tracing::info!("{:<20} | Quotient Deg = {:<2} | Prep Cols = {:<2} | Main Cols = {:<8} | Perm Cols = {:<4} | {:4} Constraints | {:3} Interactions On Buses {:?}",
                pk.air_name,
                pk.vk.quotient_degree,
                width.preprocessed.unwrap_or(0),
                format!("{:?}",width.main_widths()),
                format!("{:?}",width.after_challenge.iter().map(|&x| x * <SC::Challenge as FieldExtensionAlgebra<Val<SC>>>::D).collect_vec()),
                pk.vk.symbolic_constraints.constraints.constraint_idx.len(),
                pk.vk.symbolic_constraints.interactions.len(),
                pk.vk
                    .symbolic_constraints
                    .interactions
                    .iter()
                    .map(|i| i.bus_index)
                    .collect_vec()
            );
            #[cfg(feature = "bench-metrics")]
            {
                let labels = [("air_name", pk.air_name.clone())];
                metrics::counter!("quotient_deg", &labels).absolute(pk.vk.quotient_degree as u64);
                // column info will be logged by prover later
                metrics::counter!("constraints", &labels)
                    .absolute(pk.vk.symbolic_constraints.constraints.constraint_idx.len() as u64);
                metrics::counter!("interactions", &labels)
                    .absolute(pk.vk.symbolic_constraints.interactions.len() as u64);
            }
        }

        MultiStarkProvingKey {
            per_air: pk_per_air,
            max_constraint_degree: self.max_constraint_degree,
        }
    }
}

impl<SC: StarkGenericConfig> AirKeygenBuilder<SC> {
    fn new(pcs: &SC::Pcs, rap_phase_seq_kind: RapPhaseSeqKind, air: Arc<dyn AnyRap<SC>>) -> Self {
        let prep_keygen_data = compute_prep_data_for_air(pcs, air.as_ref());
        AirKeygenBuilder {
            air,
            rap_phase_seq_kind,
            prep_keygen_data,
        }
    }

    fn max_constraint_degree(&self) -> usize {
        self.get_symbolic_builder(None)
            .constraints()
            .max_constraint_degree()
    }

    fn generate_pk(self, max_constraint_degree: usize) -> StarkProvingKey<SC> {
        let air_name = self.air.name();

        let symbolic_builder = self.get_symbolic_builder(Some(max_constraint_degree));
        let params = symbolic_builder.params();
        let symbolic_constraints = symbolic_builder.constraints();
        let log_quotient_degree = symbolic_constraints.get_log_quotient_degree();
        let quotient_degree = 1 << log_quotient_degree;

        let Self {
            prep_keygen_data:
                PrepKeygenData {
                    verifier_data: prep_verifier_data,
                    prover_data: prep_prover_data,
                },
            ..
        } = self;

        let vk: StarkVerifyingKey<Val<SC>, Com<SC>> = StarkVerifyingKey {
            preprocessed_data: prep_verifier_data,
            params,
            symbolic_constraints: symbolic_constraints.into(),
            quotient_degree,
            rap_phase_seq_kind: self.rap_phase_seq_kind,
        };
        StarkProvingKey {
            air_name,
            vk,
            preprocessed_data: prep_prover_data,
        }
    }

    fn get_symbolic_builder(
        &self,
        max_constraint_degree: Option<usize>,
    ) -> SymbolicRapBuilder<Val<SC>> {
        let width = TraceWidth {
            preprocessed: self.prep_keygen_data.width(),
            cached_mains: self.air.cached_main_widths(),
            common_main: self.air.common_main_width(),
            after_challenge: vec![],
        };
        get_symbolic_builder(
            self.air.as_ref(),
            &width,
            &[],
            &[],
            SC::RapPhaseSeq::ID,
            max_constraint_degree.unwrap_or(0),
        )
    }
}

pub(super) struct PrepKeygenData<SC: StarkGenericConfig> {
    pub verifier_data: Option<VerifierSinglePreprocessedData<Com<SC>>>,
    pub prover_data: Option<ProverOnlySinglePreprocessedData<SC>>,
}

impl<SC: StarkGenericConfig> PrepKeygenData<SC> {
    pub fn width(&self) -> Option<usize> {
        self.prover_data.as_ref().map(|d| d.trace.width())
    }
}

fn compute_prep_data_for_air<SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    air: &dyn AnyRap<SC>,
) -> PrepKeygenData<SC> {
    let preprocessed_trace = air.preprocessed_trace();
    let vpdata_opt = preprocessed_trace.map(|trace| {
        let domain = pcs.natural_domain_for_degree(trace.height());
        let (commit, data) = pcs.commit(vec![(domain, trace.clone())]);
        let vdata = VerifierSinglePreprocessedData { commit };
        let pdata = ProverOnlySinglePreprocessedData {
            trace: Arc::new(trace),
            data: Arc::new(data),
        };
        (vdata, pdata)
    });
    if let Some((vdata, pdata)) = vpdata_opt {
        PrepKeygenData {
            prover_data: Some(pdata),
            verifier_data: Some(vdata),
        }
    } else {
        PrepKeygenData {
            prover_data: None,
            verifier_data: None,
        }
    }
}
