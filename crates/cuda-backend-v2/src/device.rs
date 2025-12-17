use getset::Getters;
use openvm_cuda_common::common::get_device;
use stark_backend_v2::SystemParams;

#[derive(Clone, Getters)]
pub struct GpuDeviceV2 {
    #[getset(get = "pub")]
    pub(crate) config: SystemParams,
    pub(crate) prover_config: GpuProverConfig,
    pub id: u32,
}

#[derive(Clone, Copy)]
pub struct GpuProverConfig {
    pub cache_stacked_matrix: bool,
    pub cache_rs_code_matrix: bool,
}

impl GpuDeviceV2 {
    pub fn new(config: SystemParams) -> Self {
        let mut prover_config = GpuProverConfig::default();
        // Heuristic to determine whether to prioritize saving memory or not
        if config.log_blowup > 1 {
            prover_config.cache_rs_code_matrix = true;
        }
        Self {
            config,
            prover_config,
            id: get_device().unwrap() as u32,
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
/// trade-off. The Reed-Solomon code computation does incur a performance penalty, but by default we
/// still opt to not cache to save memory.
impl Default for GpuProverConfig {
    fn default() -> Self {
        Self {
            cache_stacked_matrix: false,
            cache_rs_code_matrix: false,
        }
    }
}
