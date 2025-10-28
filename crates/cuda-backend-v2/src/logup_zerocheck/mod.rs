use std::{marker::PhantomData, sync::Arc};

use stark_backend_v2::{
    poly_common::UnivariatePoly,
    poseidon2::sponge::FiatShamirTranscript,
    prover::{
        AirProvingContextV2, CpuBackendV2, CpuDeviceV2, DeviceMultiStarkProvingKeyV2,
        DeviceStarkProvingKeyV2, LogupZerocheckCpu, LogupZerocheckProver, ProvingContextV2,
        fractional_sumcheck_gkr::FracSumcheckProof,
        stacked_pcs::{StackedLayout, StackedPcsData},
    },
};

use crate::{
    Digest, EF, F, GpuBackendV2, GpuDeviceV2,
    gpu_backend::{transport_matrix_d2h_col_major, transport_stacked_pcs_data_to_host},
    stacked_pcs::StackedPcsDataGpu,
};

pub struct LogupZerocheckGpu<'a> {
    // TODO[CUDA]: remove cpu types after gpu implementation done. In the meantime, progressively
    // migrate code from cpu to gpu
    cpu: LogupZerocheckCpu<'a>,
    __cpu_device: CpuDeviceV2,
    __cpu_pk: DeviceMultiStarkProvingKeyV2<CpuBackendV2>,
    _device: PhantomData<&'a GpuDeviceV2>,
}

impl<'a, TS> LogupZerocheckProver<'a, GpuBackendV2, GpuDeviceV2, TS> for LogupZerocheckGpu<'a>
where
    TS: FiatShamirTranscript,
{
    fn prove_logup_gkr(
        device: &'a GpuDeviceV2,
        transcript: &mut TS,
        pk: &'a DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        ctx: &ProvingContextV2<GpuBackendV2>,
        n_logup: usize,
        interactions_layout: StackedLayout,
        alpha_logup: EF,
        beta_logup: EF,
    ) -> (Self, FracSumcheckProof<EF>) {
        let __cpu_pk = transport_device_pk_to_host(pk);
        let __cpu_device = CpuDeviceV2::new(device.config());

        let cpu_ctx = transport_proving_context_ref_to_host(ctx);
        let (cpu, frac_sum_proof) = LogupZerocheckCpu::prove_logup_gkr(
            unsafe { &*(&__cpu_device as *const _) },
            transcript,
            unsafe { &*(&__cpu_pk as *const _) },
            &cpu_ctx,
            n_logup,
            interactions_layout,
            alpha_logup,
            beta_logup,
        );

        (
            Self {
                cpu,
                __cpu_device,
                __cpu_pk,
                _device: PhantomData,
            },
            frac_sum_proof,
        )
    }

    fn n_global(&self) -> usize {
        LogupZerocheckProver::<_, _, TS>::n_global(&self.cpu)
    }

    fn sumcheck_uni_round0_polys(
        &mut self,
        ctx: &ProvingContextV2<GpuBackendV2>,
        lambda: EF,
    ) -> Vec<UnivariatePoly<EF>> {
        let cpu_ctx = transport_proving_context_ref_to_host(ctx);
        LogupZerocheckProver::<_, _, TS>::sumcheck_uni_round0_polys(&mut self.cpu, &cpu_ctx, lambda)
    }

    fn fold_ple_evals(&mut self, ctx: ProvingContextV2<GpuBackendV2>, r_0: EF) {
        let cpu_ctx = transport_proving_context_ref_to_host(&ctx);
        LogupZerocheckProver::<_, _, TS>::fold_ple_evals(&mut self.cpu, cpu_ctx, r_0);
    }

    fn sumcheck_polys_eval(&mut self, round: usize, r_prev: EF) -> Vec<Vec<EF>> {
        LogupZerocheckProver::<_, _, TS>::sumcheck_polys_eval(&mut self.cpu, round, r_prev)
    }

    fn fold_mle_evals(&mut self, round: usize, r_round: EF) {
        LogupZerocheckProver::<_, _, TS>::fold_mle_evals(&mut self.cpu, round, r_round)
    }

    fn into_column_openings(self) -> Vec<Vec<Vec<(EF, EF)>>> {
        LogupZerocheckProver::<_, _, TS>::into_column_openings(self.cpu)
    }
}

fn transport_device_pk_to_host(
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
) -> DeviceMultiStarkProvingKeyV2<CpuBackendV2> {
    let per_air = pk
        .per_air
        .iter()
        .map(|air_pk| {
            let preprocessed_data = air_pk.preprocessed_data.as_ref().map(|(commit, data)| {
                let host = transport_pcs_arc_to_host(data);
                (*commit, host)
            });
            DeviceStarkProvingKeyV2 {
                air_name: air_pk.air_name.clone(),
                vk: air_pk.vk.clone(),
                preprocessed_data,
            }
        })
        .collect();

    DeviceMultiStarkProvingKeyV2::new(
        per_air,
        pk.trace_height_constraints.clone(),
        pk.max_constraint_degree,
        pk.params,
        pk.vk_pre_hash,
    )
}

fn transport_proving_context_ref_to_host(
    ctx: &ProvingContextV2<GpuBackendV2>,
) -> ProvingContextV2<CpuBackendV2> {
    let per_trace = ctx
        .per_trace
        .iter()
        .map(|(air_idx, air_ctx)| (*air_idx, transport_air_context_to_host(air_ctx)))
        .collect();
    ProvingContextV2::new(per_trace)
}

fn transport_air_context_to_host(
    air_ctx: &AirProvingContextV2<GpuBackendV2>,
) -> AirProvingContextV2<CpuBackendV2> {
    let cached_mains = air_ctx
        .cached_mains
        .iter()
        .map(|(commit, data)| (*commit, transport_pcs_arc_to_host(data)))
        .collect();
    let common_main = transport_matrix_d2h_col_major(&air_ctx.common_main).unwrap();
    let public_values = air_ctx.public_values.clone();
    AirProvingContextV2::new(cached_mains, common_main, public_values)
}

fn transport_pcs_arc_to_host(
    data: &Arc<StackedPcsDataGpu<F, Digest>>,
) -> Arc<StackedPcsData<F, Digest>> {
    Arc::new(transport_stacked_pcs_data_to_host(data.as_ref()))
}
