use trace::TraceCommitter;

use crate::config::StarkGenericConfig;

/// Polynomial opening proofs
pub mod opener;
/// Computation of DEEP quotient polynomial and commitment
pub mod quotient;
/// Trace commitment computation
pub mod trace;

/// Proves multiple chips with interactions together.
/// This prover implementation is specialized for Interactive AIRs.
pub struct MultiTraceStarkProver<'c, SC: StarkGenericConfig> {
    pub config: &'c SC,
}

impl<'c, SC: StarkGenericConfig> MultiTraceStarkProver<'c, SC> {
    pub fn new(config: &'c SC) -> Self {
        Self { config }
    }

    pub fn pcs(&self) -> &SC::Pcs {
        self.config.pcs()
    }

    pub fn committer(&self) -> TraceCommitter<SC> {
        TraceCommitter::new(self.pcs())
    }
}

// fn commit_perm_traces<SC: StarkGenericConfig>(
//     pcs: &SC::Pcs,
//     perm_traces: Vec<Option<RowMajorMatrix<SC::Challenge>>>,
//     domain_per_air: &[Domain<SC>],
// ) -> Option<ProverTraceData<SC>> {
//     let flattened_traces_with_domains: Vec<_> = perm_traces
//         .into_iter()
//         .zip_eq(domain_per_air)
//         .flat_map(|(perm_trace, domain)| perm_trace.map(|trace| (*domain, trace.flatten_to_base())))
//         .collect();
//     // Only commit if there are permutation traces
//     if !flattened_traces_with_domains.is_empty() {
//         let (commit, data) = pcs.commit(flattened_traces_with_domains);
//         Some(ProverTraceData {
//             commit,
//             data: data.into(),
//         })
//     } else {
//         None
//     }
// }
