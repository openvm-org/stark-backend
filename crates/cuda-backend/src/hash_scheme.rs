use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use openvm_stark_backend::{StarkProtocolConfig, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, Digest as BabyBearPoseidon2Digest,
};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    cuda::merkle_tree::{
        poseidon2_adjacent_compress_layer, poseidon2_compressing_row_hashes,
        poseidon2_compressing_row_hashes_ext,
    },
    sponge::{DuplexSpongeGpu, GpuFiatShamirTranscript},
    types::{EF, F},
};

/// Dispatch trait for GPU Merkle hash kernels.
///
/// Each implementation routes the three kernel entry points
/// (`compress_rows`, `compress_rows_ext`, `compress_layer`) to the
/// appropriate CUDA FFI wrappers, and declares the concrete `Digest` type
/// those kernels produce.
pub trait GpuMerkleHash: Copy + Clone + Send + Sync + 'static {
    type Digest: Copy + Clone + Send + Sync + Serialize + DeserializeOwned + 'static;

    /// Compress rows of a base-field matrix into digest leaves.
    unsafe fn compress_rows(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<F>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError>;

    /// Compress rows of an extension-field matrix into digest leaves.
    unsafe fn compress_rows_ext(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<EF>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError>;

    /// Compress adjacent pairs of digests to build an inner Merkle layer.
    unsafe fn compress_layer(
        output: &mut DeviceBuffer<Self::Digest>,
        prev_layer: &DeviceBuffer<Self::Digest>,
        output_size: usize,
    ) -> Result<(), CudaError>;
}

/// Binding trait that couples a `StarkProtocolConfig`, a Merkle hash scheme,
/// and a transcript type into a single coherent GPU proving configuration.
pub trait GpuHashScheme: Copy + Clone + Send + Sync + 'static {
    type SC: StarkProtocolConfig<F = F, EF = EF, Digest = Self::Digest>;
    type Digest: Copy + Clone + Send + Sync + Serialize + DeserializeOwned + 'static;
    type Transcript: GpuFiatShamirTranscript<Self::SC>
        + Default
        + Clone
        + Send
        + Sync
        + 'static;
    type MerkleHash: GpuMerkleHash<Digest = Self::Digest>;

    fn default_config(params: SystemParams) -> Self::SC;

    fn default_transcript() -> Self::Transcript;
}

// ---------------------------------------------------------------------------
// Poseidon2 / BabyBear concrete implementations
// ---------------------------------------------------------------------------

/// Poseidon2 Merkle hash over BabyBear — delegates to the existing CUDA FFI.
#[derive(Clone, Copy, Debug, Default)]
pub struct Poseidon2MerkleHash;

impl GpuMerkleHash for Poseidon2MerkleHash {
    type Digest = BabyBearPoseidon2Digest;

    unsafe fn compress_rows(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<F>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError> {
        poseidon2_compressing_row_hashes(out, matrix, width, query_stride, log_rows_per_query)
    }

    unsafe fn compress_rows_ext(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<EF>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError> {
        poseidon2_compressing_row_hashes_ext(out, matrix, width, query_stride, log_rows_per_query)
    }

    unsafe fn compress_layer(
        output: &mut DeviceBuffer<Self::Digest>,
        prev_layer: &DeviceBuffer<Self::Digest>,
        output_size: usize,
    ) -> Result<(), CudaError> {
        poseidon2_adjacent_compress_layer(output, prev_layer, output_size)
    }
}

/// BabyBear Poseidon2 hash scheme — the only scheme implemented in this crate.
#[derive(Clone, Copy, Debug, Default)]
pub struct BabyBearPoseidon2HashScheme;

impl GpuHashScheme for BabyBearPoseidon2HashScheme {
    type SC = BabyBearPoseidon2Config;
    type Digest = BabyBearPoseidon2Digest;
    type Transcript = DuplexSpongeGpu;
    type MerkleHash = Poseidon2MerkleHash;

    fn default_config(params: SystemParams) -> Self::SC {
        Self::SC::default_from_params(params)
    }

    fn default_transcript() -> Self::Transcript {
        Self::Transcript::default()
    }
}

pub type DefaultHashScheme = BabyBearPoseidon2HashScheme;
