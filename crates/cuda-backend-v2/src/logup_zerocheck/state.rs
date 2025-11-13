use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use stark_backend_v2::prover::fractional_sumcheck_gkr::Frac;

use super::interactions::TraceInteractionMeta;
use crate::{EF, F};

#[derive(Default)]
pub struct Round0Buffers {
    pub selectors_base: Vec<DeviceMatrix<F>>,
    pub eq_xi: Vec<DeviceMatrix<EF>>,
    pub public_values: Vec<DeviceBuffer<F>>,
}

#[derive(Default)]
pub struct FractionalGkrState {
    pub trace_interactions: Vec<Option<TraceInteractionMeta>>,
    pub input_evals: Option<DeviceBuffer<Frac<EF>>>,
    pub segment_tree: Option<DeviceBuffer<Frac<EF>>>,
    pub total_rounds: usize,
}
