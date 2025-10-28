use std::marker::PhantomData;

use openvm_stark_backend::prover::MatrixDimensions;
use stark_backend_v2::{
    poly_common::UnivariatePoly,
    prover::{
        CpuDeviceV2,
        stacked_pcs::StackedPcsData,
        stacked_reduction::{StackedReductionCpu, StackedReductionProver},
    },
};

use crate::{
    Digest, EF, F, GpuBackendV2, GpuDeviceV2, gpu_backend::transport_stacked_pcs_data_to_host,
    stacked_pcs::StackedPcsDataGpu,
};

pub struct StackedReductionGpu<'a> {
    // TODO[CUDA]: delete all cpu types once gpu implementation is done
    __cpu_device: CpuDeviceV2,
    cpu: StackedReductionCpu<'a>,
    __host_pcs_data: Vec<StackedPcsData<F, Digest>>,
    _device: PhantomData<&'a GpuDeviceV2>,
}

impl<'a> StackedReductionProver<'a, GpuBackendV2, GpuDeviceV2> for StackedReductionGpu<'a> {
    fn new(
        device: &'a GpuDeviceV2,
        stacked_per_commit: Vec<&'a StackedPcsDataGpu<F, Digest>>,
        r: &[EF],
        lambda: EF,
    ) -> Self {
        let l_skip = device.config.l_skip;
        let n_stack = device.config.n_stack;
        debug_assert!(
            stacked_per_commit
                .iter()
                .all(|d| d.matrix.height() == 1 << (l_skip + n_stack))
        );
        let __host_pcs_data: Vec<StackedPcsData<F, Digest>> = stacked_per_commit
            .into_iter()
            .map(transport_stacked_pcs_data_to_host)
            .collect();

        // Remove lifetimes because we store the underlying data in the struct
        let host_refs: Vec<&StackedPcsData<F, Digest>> = __host_pcs_data
            .iter()
            .map(|d| unsafe { &*(d as *const _) })
            .collect();

        let __cpu_device = CpuDeviceV2::new(device.config());
        let cpu = StackedReductionCpu::new(
            unsafe { &*(&__cpu_device as *const _) },
            host_refs,
            r,
            lambda,
        );

        Self {
            __cpu_device,
            cpu,
            __host_pcs_data,
            _device: PhantomData,
        }
    }

    fn batch_sumcheck_uni_round0_poly(&mut self) -> UnivariatePoly<EF> {
        self.cpu.batch_sumcheck_uni_round0_poly()
    }

    fn fold_ple_evals(&mut self, u_0: EF) {
        self.cpu.fold_ple_evals(u_0);
    }

    fn batch_sumcheck_poly_eval(&mut self, round: usize, u_prev: EF) -> [EF; 2] {
        self.cpu.batch_sumcheck_poly_eval(round, u_prev)
    }

    fn fold_mle_evals(&mut self, round: usize, u_round: EF) {
        self.cpu.fold_mle_evals(round, u_round);
    }

    fn into_stacked_openings(self) -> Vec<Vec<EF>> {
        self.cpu.into_stacked_openings()
    }
}
