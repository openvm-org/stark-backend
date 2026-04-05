use getset::{CopyGetters, Getters, MutGetters};
use openvm_cuda_common::{
    common::get_device,
    stream::{CudaStream, DeviceContext, StreamGuard},
};
use openvm_stark_backend::SystemParams;

use crate::cuda::{
    batch_ntt_small::{ensure_device_ntt_twiddles_initialized, validate_gpu_l_skip},
    device_info::get_sm_count,
};

/// Pure device configuration — no stream, no CUDA runtime state.
/// Renamed from the old `GpuDevice`.
#[derive(Clone, Getters, CopyGetters, MutGetters)]
pub struct GpuDeviceConfig {
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

/// Stream-owning device handle. Wraps [`GpuDeviceConfig`] with a
/// [`DeviceContext`] that carries the explicit CUDA stream.
#[derive(Clone)]
pub struct GpuDevice {
    pub config: GpuDeviceConfig,
    pub ctx: DeviceContext,
}

impl GpuDevice {
    pub fn new(params: SystemParams) -> Result<Self, openvm_cuda_common::error::CudaError> {
        validate_gpu_l_skip(params.l_skip)
            .expect("GPU backend requires l_skip <= 10 for current CUDA kernels");
        ensure_device_ntt_twiddles_initialized()
            .expect("failed to initialize small-NTT twiddles for current CUDA device");

        let prover_config = GpuProverConfig {
            zerocheck_save_memory: params.log_blowup == 1,
            ..Default::default()
        };
        let id = get_device().unwrap() as u32;
        let ctx = DeviceContext {
            device_id: id,
            stream: StreamGuard::new(CudaStream::new_non_blocking()?),
        };
        let sm_count = get_sm_count(id, ctx.stream.as_raw()).expect("failed to get SM count");
        let config = GpuDeviceConfig {
            config: params,
            prover_config,
            id,
            sm_count,
        };

        Ok(Self { config, ctx })
    }

    // Delegate accessors to inner config
    pub fn config(&self) -> &SystemParams {
        &self.config.config
    }

    pub fn prover_config(&self) -> &GpuProverConfig {
        &self.config.prover_config
    }

    pub fn prover_config_mut(&mut self) -> &mut GpuProverConfig {
        &mut self.config.prover_config
    }

    pub fn sm_count(&self) -> u32 {
        self.config.sm_count
    }

    pub fn id(&self) -> u32 {
        self.config.id
    }

    pub fn with_cache_rs_code_matrix(mut self, cache_rs_code_matrix: bool) -> Self {
        self.config.prover_config.cache_rs_code_matrix = cache_rs_code_matrix;
        self
    }

    pub fn set_cache_rs_code_matrix(&mut self, cache_rs_code_matrix: bool) {
        self.config.prover_config.cache_rs_code_matrix = cache_rs_code_matrix;
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
