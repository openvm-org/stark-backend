//! Type conversion utilities between Metal and CPU backend types.
//!
//! Since Metal uses unified memory (StorageModeShared), conversions are efficient:
//! reading MetalMatrix data is a direct memory read, no GPU sync needed for CPU-only ops.

use std::sync::Arc;

use openvm_metal_common::copy::MemCopyD2H;
use openvm_stark_backend::prover::{
    AirProvingContext, ColMajorMatrix, CommittedTraceData, CpuBackend, CpuDevice,
    DeviceMultiStarkProvingKey, DeviceStarkProvingKey, MatrixDimensions, ProvingContext,
};

use crate::{
    base::MetalMatrix,
    prelude::{F, SC},
    MetalBackend,
};

/// Convert a MetalMatrix to a ColMajorMatrix by reading from unified memory.
pub fn matrix_to_cpu(m: &MetalMatrix<F>) -> ColMajorMatrix<F> {
    ColMajorMatrix::new(m.to_host(), m.width())
}

/// Convert committed trace data from Metal to CPU backend.
fn committed_data_to_cpu(
    cd: &CommittedTraceData<MetalBackend>,
) -> CommittedTraceData<CpuBackend<SC>> {
    CommittedTraceData {
        commitment: cd.commitment,
        trace: matrix_to_cpu(&cd.trace),
        data: Arc::new(cd.data.inner.clone()),
    }
}

/// Convert DeviceMultiStarkProvingKey from Metal to CPU backend.
pub fn mpk_to_cpu(
    mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
) -> DeviceMultiStarkProvingKey<CpuBackend<SC>> {
    let per_air = mpk
        .per_air
        .iter()
        .map(|pk| {
            let preprocessed_data = pk.preprocessed_data.as_ref().map(committed_data_to_cpu);
            DeviceStarkProvingKey {
                air_name: pk.air_name.clone(),
                vk: pk.vk.clone(),
                preprocessed_data,
                other_data: (),
            }
        })
        .collect();
    DeviceMultiStarkProvingKey::new(
        per_air,
        mpk.trace_height_constraints.clone(),
        mpk.max_constraint_degree,
        mpk.params.clone(),
        mpk.vk_pre_hash,
    )
}

/// Convert ProvingContext from Metal to CPU backend.
pub fn ctx_to_cpu(ctx: &ProvingContext<MetalBackend>) -> ProvingContext<CpuBackend<SC>> {
    let per_trace = ctx
        .per_trace
        .iter()
        .map(|(air_idx, trace_ctx)| {
            let cached_mains = trace_ctx
                .cached_mains
                .iter()
                .map(committed_data_to_cpu)
                .collect();
            let common_main = matrix_to_cpu(&trace_ctx.common_main);
            let public_values = trace_ctx.public_values.clone();
            (
                *air_idx,
                AirProvingContext::new(cached_mains, common_main, public_values),
            )
        })
        .collect();
    ProvingContext::new(per_trace)
}

/// Create a CpuDevice from SystemParams (for delegating prove calls).
pub fn make_cpu_device(params: &openvm_stark_backend::SystemParams) -> CpuDevice<SC> {
    let config = SC::default_from_params(params.clone());
    CpuDevice::new(config)
}
