use std::{cmp::Reverse, sync::Arc};

use derivative::Derivative;

use crate::{
    keygen::types::{
        LinearConstraint, MultiStarkVerifyingKey, MultiStarkVerifyingKey0, StarkVerifyingKey,
    },
    proof::TraceVData,
    prover::{MatrixDimensions, ProverBackend},
    StarkProtocolConfig, SystemParams,
};

/// The committed trace data for a single trace matrix. This type is used to store prover data for
/// both preprocessed trace and cached trace.
#[derive(Derivative)]
#[derivative(Clone(bound = "PB::Matrix: Clone"))]
pub struct CommittedTraceData<PB: ProverBackend> {
    /// The polynomial commitment.
    pub commitment: PB::Commitment,
    /// The trace matrix, unstacked, in evaluation form.
    pub trace: PB::Matrix,
    /// The PCS data for a single committed trace matrix.
    pub data: Arc<PB::PcsData>,
}

/// The proving key for a circuit consisting of multiple AIRs, after prover-specific data has been
/// transferred to device. The host data (e.g., vkey) is owned by this struct.
///
/// Ordering is always by AIR ID and includes all AIRs, including ones that may have empty traces.
#[derive(derive_new::new)]
pub struct DeviceMultiStarkProvingKey<PB: ProverBackend> {
    pub per_air: Vec<DeviceStarkProvingKey<PB>>,
    pub trace_height_constraints: Vec<LinearConstraint>,
    /// Maximum degree of constraints across all AIRs
    pub max_constraint_degree: usize,
    pub params: SystemParams,
    pub vk_pre_hash: PB::Commitment,
}

/// The proving key after prover-specific data has been transferred to device. The host data (e.g.,
/// vkey) is owned by this struct.
pub struct DeviceStarkProvingKey<PB: ProverBackend> {
    /// Type name of the AIR, for display purposes only
    pub air_name: String,
    pub vk: StarkVerifyingKey<PB::Val, PB::Commitment>,
    /// Prover only data for preprocessed trace
    pub preprocessed_data: Option<CommittedTraceData<PB>>,
    pub other_data: PB::OtherAirData,
}

#[derive(derive_new::new)]
pub struct ProvingContext<PB: ProverBackend> {
    /// For each AIR with non-empty trace, the pair of (AIR ID, [AirProvingContext]), where AIR
    /// ID is with respect to the vkey ordering.
    pub per_trace: Vec<(usize, AirProvingContext<PB>)>,
}

#[derive(derive_new::new)]
pub struct AirProvingContext<PB: ProverBackend> {
    /// Cached main trace matrices as `PcsData`. The original trace matrix should be extractable as
    /// a view from the `PcsData`. The `PcsData` should also contain the commitment value. Cached
    /// trace commitments have a single matrix per commitment.
    ///
    /// The `PcsData` is kept inside an `Arc` to emphasize that this data is cached and may be
    /// shared between multiple proving contexts. In particular, it is not typically safe to mutate
    /// the data during a proving job.
    pub cached_mains: Vec<CommittedTraceData<PB>>,
    /// Common main trace matrix
    pub common_main: PB::Matrix,
    /// Public values
    pub public_values: Vec<PB::Val>,
}

/// Proof on the host, with respect to the host types in the generic `PB`.
pub struct HostProof<SC: StarkProtocolConfig, PB: ProverBackend, ConstraintsProof, OpeningProof> {
    /// The commitment to the data in common_main.
    pub common_main_commit: PB::Commitment,

    /// For each AIR in vkey order, the corresponding trace shape, or None if
    /// the trace is empty. In a valid proof, if `vk.per_air[i].is_required`,
    /// then `trace_vdata[i]` must be `Some(_)`.
    pub trace_vdata: Vec<Option<TraceVData<SC>>>,

    /// For each AIR in vkey order, the public values. Public values should be empty if the AIR has
    /// an empty trace.
    pub public_values: Vec<Vec<PB::Val>>,

    pub constraints_proof: ConstraintsProof,
    /// Opening proof for multiple polynomials over mixed sized domains
    pub opening_proof: OpeningProof,
}

impl<PB: ProverBackend> CommittedTraceData<PB> {
    #[inline(always)]
    pub fn height(&self) -> usize {
        self.trace.height()
    }
}

impl<PB: ProverBackend> DeviceMultiStarkProvingKey<PB> {
    pub fn get_vk<SC>(&self) -> MultiStarkVerifyingKey<SC>
    where
        SC: StarkProtocolConfig<F = PB::Val, Digest = PB::Commitment>,
    {
        let per_air = self.per_air.iter().map(|pk| pk.vk.clone()).collect();
        let inner = MultiStarkVerifyingKey0 {
            params: self.params.clone(),
            per_air,
            trace_height_constraints: self.trace_height_constraints.clone(),
        };
        MultiStarkVerifyingKey {
            inner,
            pre_hash: self.vk_pre_hash.clone(),
        }
    }
}

impl<PB: ProverBackend> IntoIterator for ProvingContext<PB> {
    type Item = (usize, AirProvingContext<PB>);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.per_trace.into_iter()
    }
}

impl<PB: ProverBackend> ProvingContext<PB> {
    pub fn common_main_traces(&self) -> impl Iterator<Item = (usize, &PB::Matrix)> {
        self.per_trace
            .iter()
            .map(|(air_idx, trace_ctx)| (*air_idx, &trace_ctx.common_main))
    }

    // Returns `self` with the trace data sorted to be descending in height for column stacking. For
    // equal heights, traces are sorted in ascending order of AIR index.
    pub fn into_sorted(mut self) -> Self {
        self.sort_for_stacking();
        self
    }

    // Stable sort the trace data to be descending in height: this is needed for stacking. For
    // equal heights, sort in ascending order of AIR index.
    pub fn sort_for_stacking(&mut self) {
        self.per_trace.sort_by_key(|(air_idx, trace_ctx)| {
            (Reverse(trace_ctx.common_main.height()), *air_idx)
        });
    }
}

impl<PB: ProverBackend> AirProvingContext<PB> {
    pub fn simple(common_main_trace: PB::Matrix, public_values: Vec<PB::Val>) -> Self {
        Self::new(vec![], common_main_trace, public_values)
    }
    pub fn simple_no_pis(common_main_trace: PB::Matrix) -> Self {
        Self::simple(common_main_trace, vec![])
    }

    /// Return the height of the main trace.
    pub fn height(&self) -> usize {
        self.common_main.height()
    }
}
