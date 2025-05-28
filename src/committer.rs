use openvm_stark_backend::prover::hal::MatrixDimensions;
use p3_baby_bear::BabyBear;
use p3_util::log2_strict_usize;

use crate::{base::DeviceMatrix, gpu_device::GpuDevice, lde::GpuLde, merkle_tree::GpuMerkleTree};

impl GpuDevice {
    pub fn commit_trace<LDE: GpuLde>(
        &self,
        trace: DeviceMatrix<BabyBear>,
    ) -> (Vec<u8>, GpuMerkleTree<LDE>) {
        let log_height: u8 = log2_strict_usize(trace.height()).try_into().unwrap();
        let lde = LDE::new(self, trace, 0, self.config.shift);
        (
            vec![log_height],
            GpuMerkleTree::new(vec![lde], self).unwrap(),
        )
    }

    /// Commit a trace to a GPU device.
    pub fn commit_traces_with_lde<LDE: GpuLde>(
        &self,
        traces_with_shifts: Vec<(DeviceMatrix<BabyBear>, BabyBear)>,
        log_blowup: usize,
        keep_traces: bool,
    ) -> (Vec<u8>, GpuMerkleTree<LDE>) {
        let (log_trace_heights, ldes): (Vec<u8>, Vec<LDE>) = traces_with_shifts
            .into_iter()
            .map(|(trace, shift)| {
                let height = trace.height();
                let log_height: u8 = log2_strict_usize(height).try_into().unwrap();
                let mut lde = LDE::new(self, trace, log_blowup, shift);
                if !keep_traces {
                    lde.to_coefficient_form();
                }
                (log_height, lde)
            })
            .collect();

        (log_trace_heights, GpuMerkleTree::new(ldes, self).unwrap())
    }
}
