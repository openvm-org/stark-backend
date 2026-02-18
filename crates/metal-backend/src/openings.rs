//! Metal-native opening proof (stacked reduction + WHIR).
//!
//! Combines the stacked opening reduction and WHIR opening proof into a single
//! Metal-native function, avoiding the need to pass CPU types through metal_backend.rs.

use itertools::Itertools;
use openvm_stark_backend::{
    poly_common::Squarable,
    proof::{StackingProof, WhirProof},
    prover::{
        stacked_pcs::StackedPcsData,
        stacked_reduction::{prove_stacked_opening_reduction, StackedReductionCpu},
        whir::prove_whir_opening,
        CpuDevice, DeviceMultiStarkProvingKey, ProvingContext,
    },
    StarkProtocolConfig,
};
use tracing::instrument;

use crate::{
    prelude::{Digest, EF, F, SC},
    sponge::DuplexSpongeMetal,
    MetalBackend, MetalDevice, StackedPcsDataMetal,
};

/// Metal-native combined opening proof (stacked reduction + WHIR).
///
/// This function handles the full opening proof pipeline, keeping CPU types
/// internal to this module.
#[instrument(name = "metal.openings", skip_all)]
pub fn prove_openings_metal(
    device: &MetalDevice,
    transcript: &mut DuplexSpongeMetal,
    mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
    ctx: ProvingContext<MetalBackend>,
    common_main_pcs_data: StackedPcsDataMetal,
    r: Vec<EF>,
) -> (StackingProof<SC>, WhirProof<SC>) {
    let config = device.config();
    let sc = SC::default_from_params(config.clone());
    let hasher = sc.hasher();

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

    // Access the inner CPU data from StackedPcsDataMetal
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

    // Run stacked opening reduction
    let cpu_device = make_cpu_device(config);
    let (stacking_proof, u_prisma) =
        prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpu<SC>>(
            &cpu_device,
            transcript,
            config.n_stack,
            stacked_per_commit.clone(),
            need_rot_per_commit,
            &r,
        );

    let (&u0, u_rest) = u_prisma.split_first().unwrap();
    let u_cube = u0
        .exp_powers_of_2()
        .take(config.l_skip)
        .chain(u_rest.iter().copied())
        .collect_vec();

    // Run WHIR opening proof using the same PCS data
    let committed_mats: Vec<_> = stacked_per_commit
        .iter()
        .map(|d| (&d.matrix, &d.tree))
        .collect();

    let whir_proof = prove_whir_opening::<SC, _>(
        transcript,
        hasher,
        config.l_skip,
        config.log_blowup,
        &config.whir,
        &committed_mats,
        &u_cube,
    );

    (stacking_proof, whir_proof)
}

/// Create a CpuDevice for algorithm execution.
fn make_cpu_device(params: &openvm_stark_backend::SystemParams) -> CpuDevice<SC> {
    let config = SC::default_from_params(params.clone());
    CpuDevice::new(config)
}
