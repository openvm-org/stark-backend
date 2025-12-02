use getset::CopyGetters;
use openvm_cuda_common::common::get_device;
use stark_backend_v2::SystemParams;

#[derive(Clone, Copy, CopyGetters)]
pub struct GpuDeviceV2 {
    #[getset(get_copy = "pub")]
    pub(crate) config: SystemParams,
    pub(crate) prover_config: GpuProverConfig,
    pub id: u32,
}

#[derive(Clone, Copy)]
pub struct GpuProverConfig {
    pub cache_stacked_matrix: bool,
}

impl GpuDeviceV2 {
    pub fn new(config: SystemParams) -> Self {
        Self {
            config,
            prover_config: GpuProverConfig::default(),
            id: get_device().unwrap() as u32,
        }
    }
}

/// Default configuration is to reduce peak memory usage when there is not a significant performance
/// trade-off. The Reed-Solomon code computation is still cached since it is a significant
/// trade-off.
impl Default for GpuProverConfig {
    fn default() -> Self {
        Self {
            cache_stacked_matrix: false,
        }
    }
}
