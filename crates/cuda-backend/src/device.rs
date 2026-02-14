use getset::{CopyGetters, Getters, MutGetters};
use openvm_cuda_common::common::get_device;
use openvm_stark_backend::SystemParams;

use crate::cuda::{
    batch_ntt_small::ensure_device_ntt_twiddles_initialized, device_info::get_sm_count,
};

#[derive(Clone, Getters, CopyGetters, MutGetters)]
pub struct GpuDevice {
    #[getset(get = "pub")]
    pub(crate) config: SystemParams,
    #[getset(get = "pub", get_mut = "pub")]
    pub(crate) prover_config: GpuProverConfig,
    pub id: u32,
    #[getset(get_copy = "pub")]
    pub sm_count: u32,
}

#[derive(Clone, Copy)]
pub struct GpuProverConfig {
    pub cache_stacked_matrix: bool,
    pub cache_rs_code_matrix: bool,
    pub zerocheck_save_memory: bool,
}

impl GpuDevice {
    pub fn new(config: SystemParams) -> Self {
        ensure_device_ntt_twiddles_initialized();

        let prover_config = GpuProverConfig {
            zerocheck_save_memory: config.log_blowup == 1,
            ..Default::default()
        };
        let id = get_device().unwrap() as u32;
        let sm_count = get_sm_count(id).expect("failed to get SM count");
        Self {
            config,
            prover_config,
            id,
            sm_count,
        }
    }

    pub fn with_cache_rs_code_matrix(mut self, cache_rs_code_matrix: bool) -> Self {
        self.prover_config.cache_rs_code_matrix = cache_rs_code_matrix;
        self
    }

    pub fn set_cache_rs_code_matrix(&mut self, cache_rs_code_matrix: bool) {
        self.prover_config.cache_rs_code_matrix = cache_rs_code_matrix;
    }
}

/// Default configuration is to reduce peak memory usage when there is not a significant performance
/// trade-off. The Reed-Solomon code computation does incur a performance penalty, so we cache it.
impl Default for GpuProverConfig {
    fn default() -> Self {
        Self {
            cache_stacked_matrix: false,
            cache_rs_code_matrix: true,
            zerocheck_save_memory: true,
        }
    }
}
