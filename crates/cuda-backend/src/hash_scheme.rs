use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::GpuDeviceCtx};
use openvm_stark_backend::{StarkProtocolConfig, SystemParams};
#[cfg(feature = "baby-bear-bn254-poseidon2")]
use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::{
    BabyBearBn254Poseidon2Config, Digest as Bn254Digest,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, Digest as BabyBearPoseidon2Digest,
};
use serde::{de::DeserializeOwned, Serialize};

#[cfg(feature = "baby-bear-bn254-poseidon2")]
use crate::{
    bn254_sponge::MultiFieldTranscriptGpu,
    cuda::bn254_merkle_tree::{
        bn254_poseidon2_adjacent_compress_layer, bn254_poseidon2_compressing_row_hashes,
        bn254_poseidon2_compressing_row_hashes_ext,
    },
};
use crate::{
    cuda::merkle_tree::{
        poseidon2_adjacent_compress_layer, poseidon2_compressing_row_hashes,
        poseidon2_compressing_row_hashes_ext,
    },
    merkle_tree::BatchQueryMerkle,
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
    type Digest: Copy
        + Clone
        + PartialEq
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + BatchQueryMerkle
        + 'static;

    /// Compress rows of a base-field matrix into digest leaves.
    ///
    /// # Safety
    ///
    /// `out` must be allocated with capacity `query_stride` and `matrix` must
    /// contain `width * query_stride * (1 << log_rows_per_query)` valid elements.
    unsafe fn compress_rows(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<F>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError>;

    /// Compress rows of an extension-field matrix into digest leaves.
    ///
    /// # Safety
    ///
    /// `out` must be allocated with capacity `query_stride` and `matrix` must
    /// contain `width * query_stride * (1 << log_rows_per_query)` valid elements.
    unsafe fn compress_rows_ext(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<EF>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError>;

    /// Compress adjacent pairs of digests to build an inner Merkle layer.
    ///
    /// # Safety
    ///
    /// `output` must be allocated with capacity `output_size`, `prev_layer` must
    /// contain at least `output_size * 2` valid elements, and the two buffers
    /// must not overlap.
    unsafe fn compress_layer(
        output: &mut DeviceBuffer<Self::Digest>,
        prev_layer: &DeviceBuffer<Self::Digest>,
        output_size: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError>;
}

/// Binding trait that couples a `StarkProtocolConfig`, a Merkle hash scheme,
/// and a transcript type into a single coherent GPU proving configuration.
pub trait GpuHashScheme: Copy + Clone + Send + Sync + 'static {
    type SC: StarkProtocolConfig<F = F, EF = EF, Digest = Self::Digest>;
    type Digest: Copy
        + Clone
        + PartialEq
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + BatchQueryMerkle
        + 'static;
    type Transcript: GpuFiatShamirTranscript<Self::SC> + Default + Clone + Send + Sync + 'static;
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
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError> {
        poseidon2_compressing_row_hashes(
            out,
            matrix,
            width,
            query_stride,
            log_rows_per_query,
            device_ctx.stream.as_raw(),
        )
    }

    unsafe fn compress_rows_ext(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<EF>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError> {
        poseidon2_compressing_row_hashes_ext(
            out,
            matrix,
            width,
            query_stride,
            log_rows_per_query,
            device_ctx.stream.as_raw(),
        )
    }

    unsafe fn compress_layer(
        output: &mut DeviceBuffer<Self::Digest>,
        prev_layer: &DeviceBuffer<Self::Digest>,
        output_size: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError> {
        poseidon2_adjacent_compress_layer(
            output,
            prev_layer,
            output_size,
            device_ctx.stream.as_raw(),
        )
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

// ---------------------------------------------------------------------------
// BN254 Poseidon2 concrete implementations
// ---------------------------------------------------------------------------

#[cfg(feature = "baby-bear-bn254-poseidon2")]
/// BN254 Poseidon2 Merkle hash — delegates to the BN254 CUDA FFI.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bn254Poseidon2MerkleHash;

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl GpuMerkleHash for Bn254Poseidon2MerkleHash {
    // `Bn254Digest` from stark-sdk = `[Bn254Scalar; 1]`, which is the same concrete type as
    // `Bn254Digest` in `cuda::bn254_merkle_tree` — both are type aliases for `[p3_bn254::Bn254;
    // 1]`.
    type Digest = Bn254Digest;

    unsafe fn compress_rows(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<F>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError> {
        bn254_poseidon2_compressing_row_hashes(
            out,
            matrix,
            width,
            query_stride,
            log_rows_per_query,
            device_ctx.stream.as_raw(),
        )
    }

    unsafe fn compress_rows_ext(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<EF>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError> {
        bn254_poseidon2_compressing_row_hashes_ext(
            out,
            matrix,
            width,
            query_stride,
            log_rows_per_query,
            device_ctx.stream.as_raw(),
        )
    }

    unsafe fn compress_layer(
        output: &mut DeviceBuffer<Self::Digest>,
        prev_layer: &DeviceBuffer<Self::Digest>,
        output_size: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), CudaError> {
        bn254_poseidon2_adjacent_compress_layer(
            output,
            prev_layer,
            output_size,
            device_ctx.stream.as_raw(),
        )
    }
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
/// BabyBear + BN254 Poseidon2 hash scheme (Groth16-friendly transcript).
#[derive(Clone, Copy, Debug, Default)]
pub struct BabyBearBn254Poseidon2HashScheme;

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl GpuHashScheme for BabyBearBn254Poseidon2HashScheme {
    type SC = BabyBearBn254Poseidon2Config;
    type Digest = Bn254Digest;
    type Transcript = MultiFieldTranscriptGpu;
    type MerkleHash = Bn254Poseidon2MerkleHash;

    fn default_config(params: SystemParams) -> Self::SC {
        Self::SC::default_from_params(params)
    }

    fn default_transcript() -> Self::Transcript {
        Self::Transcript::default()
    }
}
