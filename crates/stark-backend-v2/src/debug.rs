// Merge into debug/mod.rs in v1:
// - debug_constraints
// - debug_constraints_and_interactions
use std::sync::Arc;

use itertools::{izip, Itertools};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    air_builders::{
        debug::{check_constraints, check_logup, USE_DEBUG_BUILDER},
        symbolic::SymbolicConstraints,
    },
    keygen::{types::StarkProvingKeyV2, MultiStarkKeygenBuilderV2},
    prover::{
        ColMajorMatrix, DeviceDataTransporterV2, ProverBackendV2, ProvingContextV2,
        StridedColMajorMatrixView,
    },
    AirRef, StarkProtocolConfig, SystemParams,
};

/// Raw input data for debugging a single AIR.
pub struct AirProofRawInput<F> {
    pub cached_mains: Vec<Arc<RowMajorMatrix<F>>>,
    pub common_main: Option<Arc<RowMajorMatrix<F>>>,
    pub public_values: Vec<F>,
}

// TODO[jpw]: move into StarkEngineV2::debug default implementation after `SC` is made generic.
/// `airs` should be the full list of all AIRs, not just used AIRs.
pub fn debug_impl<SC, PB, PD>(
    config: SystemParams,
    device: &PD,
    airs: &[AirRef<SC>],
    ctx: &ProvingContextV2<PB>,
)
where
    SC: StarkProtocolConfig,
    PB: ProverBackendV2<
        Val = SC::F,
        Challenge = SC::EF,
        Commitment = SC::Digest,
    >,
    PD: DeviceDataTransporterV2<SC, PB>,
{
    let mut keygen_builder = MultiStarkKeygenBuilderV2::new(config);
    for air in airs {
        keygen_builder.add_air(air.clone());
    }
    let pk = keygen_builder.generate_pk().unwrap();

    let transpose = |mat: ColMajorMatrix<SC::F>| {
        let row_major = StridedColMajorMatrixView::from(mat.as_view()).to_row_major_matrix();
        Arc::new(row_major)
    };
    let (inputs, used_airs, used_pks): (Vec<_>, Vec<_>, Vec<_>) = ctx
        .per_trace
        .iter()
        .map(|(air_id, air_ctx)| {
            // Transfer from device **back** to host so the debugger can read the data.
            let common_main = device.transport_matrix_from_device_to_host(&air_ctx.common_main);
            let cached_mains = air_ctx
                .cached_mains
                .iter()
                .map(|cd| transpose(device.transport_matrix_from_device_to_host(&cd.trace)))
                .collect_vec();
            let common_main = Some(transpose(common_main));
            let public_values = air_ctx.public_values.clone();
            (
                AirProofRawInput {
                    cached_mains,
                    common_main,
                    public_values,
                },
                airs[*air_id].clone(),
                &pk.per_air[*air_id],
            )
        })
        .multiunzip();

    debug_constraints_and_interactions(&used_airs, &used_pks, &inputs);
}

/// The debugging will check the main AIR constraints and then separately check LogUp constraints by
/// checking the actual multiset equalities. Currently it will not debug check any after challenge
/// phase constraints for implementation simplicity.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn debug_constraints_and_interactions<SC: StarkProtocolConfig>(
    airs: &[AirRef<SC>],
    pk: &[&StarkProvingKeyV2<SC>],
    inputs: &[AirProofRawInput<SC::F>],
) {
    USE_DEBUG_BUILDER.with(|debug| {
        if *debug.lock().unwrap() {
            let (main_parts_per_air, pvs_per_air): (Vec<_>, Vec<_>) = inputs
                .iter()
                .map(|input| {
                    let mut main_parts = input
                        .cached_mains
                        .iter()
                        .map(|trace| trace.as_view())
                        .collect_vec();
                    if let Some(trace) = input.common_main.as_ref() {
                        main_parts.push(trace.as_view());
                    }
                    (main_parts, input.public_values.clone())
                })
                .unzip();
            let preprocessed = izip!(airs, pk, &main_parts_per_air, &pvs_per_air)
                .map(|(air, pk, main_parts, pvs)| {
                    let preprocessed_trace = pk
                        .preprocessed_data
                        .as_ref()
                        .map(|data| data.mat_view(0).to_row_major_matrix());
                    tracing::debug!("Checking constraints for {}", air.name());
                    check_constraints(
                        air.as_ref(),
                        &air.name(),
                        &preprocessed_trace.as_ref().map(|t| t.as_view()),
                        main_parts,
                        pvs,
                    );
                    preprocessed_trace
                })
                .collect_vec();

            let (air_names, interactions): (Vec<_>, Vec<_>) = pk
                .iter()
                .map(|pk| {
                    let sym_constraints = SymbolicConstraints::from(&pk.vk.symbolic_constraints);
                    (pk.air_name.clone(), sym_constraints.interactions)
                })
                .unzip();
            let preprocessed_views = preprocessed
                .iter()
                .map(|t| t.as_ref().map(|t| t.as_view()))
                .collect_vec();
            check_logup(
                &air_names,
                &interactions,
                &preprocessed_views,
                &main_parts_per_air,
                &pvs_per_air,
            );
        }
    });
}
