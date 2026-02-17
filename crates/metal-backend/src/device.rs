use getset::{CopyGetters, Getters, MutGetters};
use openvm_stark_backend::SystemParams;

use crate::metal::device_info::get_gpu_core_count;

#[derive(Clone, Getters, CopyGetters, MutGetters)]
pub struct MetalDevice {
    #[getset(get = "pub")]
    pub(crate) config: SystemParams,
    #[getset(get = "pub", get_mut = "pub")]
    pub(crate) prover_config: MetalProverConfig,
    #[getset(get_copy = "pub")]
    pub sm_count: u32,
}

#[derive(Clone, Copy)]
pub struct MetalProverConfig {
    pub cache_stacked_matrix: bool,
    pub cache_rs_code_matrix: bool,
    pub zerocheck_save_memory: bool,
}

impl MetalDevice {
    pub fn new(config: SystemParams) -> Self {
        let prover_config = MetalProverConfig {
            zerocheck_save_memory: config.log_blowup == 1,
            ..Default::default()
        };
        let sm_count = get_gpu_core_count(0).unwrap();
        Self {
            config,
            prover_config,
            sm_count,
        }
    }
}

impl Default for MetalProverConfig {
    fn default() -> Self {
        Self {
            cache_stacked_matrix: false,
            cache_rs_code_matrix: true,
            zerocheck_save_memory: true,
        }
    }
}
