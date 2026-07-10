use std::sync::{Arc, Mutex};

use getset::{CopyGetters, Getters, MutGetters};
use openvm_cuda_common::{
    common::get_device,
    stream::{CudaStream, GpuDeviceCtx, StreamGuard},
};
use openvm_stark_backend::{
    memory_metering::ProvingMemoryConfig, StarkProtocolConfig, SystemParams,
};

use crate::{
    cuda::{
        batch_ntt_small::{ensure_device_ntt_twiddles_initialized, validate_gpu_l_skip},
        device_info::get_sm_count,
    },
    stacked_pcs::RsCodewordPrefetch,
};

/// Pure device configuration — no stream, no CUDA runtime state.
/// Renamed from the old `GpuDevice`.
#[derive(Clone, Getters, CopyGetters, MutGetters)]
pub struct GpuDeviceConfig {
    #[getset(get = "pub")]
    pub(crate) params: SystemParams,
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
    /// When the RS codeword is not cached on device, re-encode it for the
    /// WHIR initial round ahead of time on a low-priority auxiliary stream
    /// (launched after the GKR fractional sumcheck), so the encode overlaps
    /// the sumcheck phases instead of running serially inside WHIR. Has no
    /// effect if `cache_rs_code_matrix` is set.
    pub prefetch_rs_code_matrix: bool,
    pub zerocheck_save_memory: bool,
}

impl GpuProverConfig {
    pub fn proving_memory_config<SC: StarkProtocolConfig>(
        &self,
        config: &SC,
    ) -> ProvingMemoryConfig {
        // Interaction memory is estimated from the CUDA fractional-GKR buffer model in
        // `openvm_stark_backend::memory_metering`. Update that estimate when changing GKR
        // input layout, work-buffer sizing, or scratch allocations.
        ProvingMemoryConfig::from_protocol_config(config, self.cache_rs_code_matrix)
    }
}

/// Stream-owning device handle. Wraps [`GpuDeviceConfig`] with a
/// [`GpuDeviceCtx`] that carries the explicit CUDA stream.
#[derive(Clone)]
pub struct GpuDevice {
    pub config: GpuDeviceConfig,
    pub device_ctx: GpuDeviceCtx,
    /// Low-priority stream for `prefetch_rs_code_matrix`: its kernels yield
    /// to the main proving stream and only soak idle bubbles.
    pub(crate) aux_ctx: GpuDeviceCtx,
    /// Handoff slot from the prefetch launch (during RAP constraints) to the
    /// WHIR opening prover. Assumes proofs on one `GpuDevice` do not run
    /// concurrently, matching the single main-stream design.
    pub(crate) rs_prefetch: Arc<Mutex<Option<RsCodewordPrefetch>>>,
}

impl GpuDevice {
    pub fn new(params: SystemParams) -> Result<Self, openvm_cuda_common::error::CudaError> {
        validate_gpu_l_skip(params.l_skip)
            .expect("GPU backend requires l_skip <= 9 for current CUDA kernels");
        ensure_device_ntt_twiddles_initialized()
            .expect("failed to initialize small-NTT twiddles for current CUDA device");

        let prover_config = GpuProverConfig {
            zerocheck_save_memory: params.log_blowup == 1,
            ..Default::default()
        };
        let id = get_device().unwrap() as u32;
        let device_ctx = GpuDeviceCtx::for_device(id)?;
        let aux_ctx = GpuDeviceCtx {
            device_id: id,
            stream: StreamGuard::new(CudaStream::new_low_priority()?),
        };
        let sm_count = get_sm_count(id).expect("failed to get SM count");
        let config = GpuDeviceConfig {
            params,
            prover_config,
            id,
            sm_count,
        };

        Ok(Self {
            config,
            device_ctx,
            aux_ctx,
            rs_prefetch: Arc::new(Mutex::new(None)),
        })
    }

    // Delegate accessors to inner config
    pub fn params(&self) -> &SystemParams {
        &self.config.params
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

    pub fn with_prefetch_rs_code_matrix(mut self, prefetch_rs_code_matrix: bool) -> Self {
        self.config.prover_config.prefetch_rs_code_matrix = prefetch_rs_code_matrix;
        self
    }

    pub fn set_prefetch_rs_code_matrix(&mut self, prefetch_rs_code_matrix: bool) {
        self.config.prover_config.prefetch_rs_code_matrix = prefetch_rs_code_matrix;
    }
}

/// Default GPU prover cache settings.
impl Default for GpuProverConfig {
    fn default() -> Self {
        Self {
            cache_stacked_matrix: false,
            cache_rs_code_matrix: false,
            prefetch_rs_code_matrix: true,
            zerocheck_save_memory: true,
        }
    }
}
