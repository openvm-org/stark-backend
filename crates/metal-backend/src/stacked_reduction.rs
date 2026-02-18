//! Metal-native stacked opening reduction.
//!
//! Implements the batch sumcheck protocol for opening reduction, mirroring
//! the CUDA `stacked_reduction.rs` module structure.

use itertools::Itertools;
use openvm_stark_backend::{
    proof::StackingProof,
    prover::{
        stacked_pcs::StackedPcsData,
        stacked_reduction::{prove_stacked_opening_reduction, StackedReductionCpu},
        CpuDevice, DeviceMultiStarkProvingKey, ProvingContext,
    },
};
use tracing::instrument;

use crate::{
    prelude::{Digest, EF, F, SC},
    sponge::DuplexSpongeMetal,
    MetalBackend, MetalDevice, StackedPcsDataMetal,
};

/// Metal-native stacked opening reduction.
///
/// Converts Metal types to CPU types internally and delegates to the CPU algorithm.
/// This is correct because Metal uses unified memory (data is directly accessible).
#[instrument(name = "metal.stacked_reduction", skip_all)]
pub fn prove_stacked_opening_reduction_metal(
    device: &MetalDevice,
    transcript: &mut DuplexSpongeMetal,
    mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
    ctx: &ProvingContext<MetalBackend>,
    common_main_pcs_data: &StackedPcsDataMetal,
    r: &[EF],
) -> (StackingProof<SC>, Vec<EF>) {
    let config = device.config();

    let need_rot_per_trace = ctx
        .per_trace
        .iter()
        .map(|(air_idx, _)| mpk.per_air[*air_idx].vk.params.need_rot)
        .collect_vec();

    // Collect preprocessed/cached PCS data
    let pre_cached_pcs_data_per_commit: Vec<_> = ctx
        .per_trace
        .iter()
        .flat_map(|(air_idx, trace_ctx)| {
            mpk.per_air[*air_idx]
                .preprocessed_data
                .iter()
                .chain(&trace_ctx.cached_mains)
                .map(|cd| cd.data.clone())
        })
        .collect();

    let cpu_common_main = common_main_pcs_data.inner();
    let cpu_pre_cached: Vec<_> = pre_cached_pcs_data_per_commit
        .iter()
        .map(|d| d.inner())
        .collect();

    let mut stacked_per_commit: Vec<&StackedPcsData<F, Digest>> = vec![cpu_common_main];
    for data in &cpu_pre_cached {
        stacked_per_commit.push(data);
    }

    let mut need_rot_per_commit = vec![need_rot_per_trace];
    for (air_idx, trace_ctx) in &ctx.per_trace {
        let need_rot = mpk.per_air[*air_idx].vk.params.need_rot;
        if mpk.per_air[*air_idx].preprocessed_data.is_some() {
            need_rot_per_commit.push(vec![need_rot]);
        }
        for _ in &trace_ctx.cached_mains {
            need_rot_per_commit.push(vec![need_rot]);
        }
    }

    // Use CPU stacked reduction via the generic interface
    let cpu_device = make_cpu_device(config);
    prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpu<SC>>(
        &cpu_device,
        transcript,
        config.n_stack,
        stacked_per_commit,
        need_rot_per_commit,
        r,
    )
}

/// Create a CpuDevice for algorithm execution.
fn make_cpu_device(params: &openvm_stark_backend::SystemParams) -> CpuDevice<SC> {
    let config = SC::default_from_params(params.clone());
    CpuDevice::new(config)
}
