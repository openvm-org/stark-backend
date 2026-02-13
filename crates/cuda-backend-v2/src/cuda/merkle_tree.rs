use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::{Digest, EF, F};

extern "C" {
    fn _poseidon2_compressing_row_hashes(
        out: *mut Digest,
        matrix: *const F,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> i32;

    fn _poseidon2_compressing_row_hashes_ext(
        out: *mut Digest,
        matrix: *const EF,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> i32;

    fn _poseidon2_strided_compress_layer(
        output: *mut Digest,
        prev_layer: *const Digest,
        output_size: usize,
        stride: usize,
    ) -> i32;

    fn _poseidon2_adjacent_compress_layer(
        output: *mut Digest,
        prev_layer: *const Digest,
        output_size: usize,
    ) -> i32;

    fn _query_digest_layers(
        d_digest_matrix: *mut F,
        d_layers_ptr: *const u64,
        d_indices: *const u64,
        num_query: u64,
        num_layer: u64,
    ) -> i32;
}

/// Computes row hashes of `matrix` of dimensions `width` x `height` using Poseidon2 and then takes
/// merkle root of `2^log_rows_per_query` strided row hashes and writes the digests to `out`, where
/// `height = query_stride * 2^log_rows_per_query`. Memory layout expects `matrix` to be column
/// major in `F`, and `out` is buffer of `Digest`. Digests are written in order of rows.
///
/// # Safety
/// - `out` must have length `>= query_stride` in `Digest` elements.
/// - `out` and `matrix` must be non-overlapping.
pub unsafe fn poseidon2_compressing_row_hashes(
    out: &mut DeviceBuffer<Digest>,
    matrix: &DeviceBuffer<F>,
    width: usize,
    query_stride: usize,
    log_rows_per_query: usize,
) -> Result<(), CudaError> {
    debug_assert!(matrix.len() >= width * (query_stride << log_rows_per_query));
    debug_assert!(out.len() >= query_stride);
    CudaError::from_result(_poseidon2_compressing_row_hashes(
        out.as_mut_ptr(),
        matrix.as_ptr(),
        width,
        query_stride,
        log_rows_per_query,
    ))
}

/// Computes row hashes of `matrix` of dimensions `width` x `height` using Poseidon2 and then takes
/// merkle root of `2^log_rows_per_query` strided row hashes and writes the digests to `out`, where
/// `height = query_stride * 2^log_rows_per_query`. Memory layout expects `matrix` to be column
/// major in `EF`, and `out` is buffer of `Digest`. Digests are written in order of rows.
///
/// Note: `matrix` column major in `EF` means that `EF::D` base field elements are contiguous in
/// memory.
///
/// # Safety
/// - `out` must have length `>= query_stride` in `Digest` elements.
/// - `out` and `matrix` must be non-overlapping.
pub unsafe fn poseidon2_compressing_row_hashes_ext(
    out: &mut DeviceBuffer<Digest>,
    matrix: &DeviceBuffer<EF>,
    width: usize,
    query_stride: usize,
    log_rows_per_query: usize,
) -> Result<(), CudaError> {
    debug_assert!(matrix.len() >= width * (query_stride << log_rows_per_query));
    debug_assert!(out.len() >= query_stride);
    CudaError::from_result(_poseidon2_compressing_row_hashes_ext(
        out.as_mut_ptr(),
        matrix.as_ptr(),
        width,
        query_stride,
        log_rows_per_query,
    ))
}

/// Writes a new digest layer of `output_size` elements of type `Digest` to `output` by apply
/// poseidon2 compress function to pairs of digest elements from `prev_layer`. The pairs to compress
/// are selected with a `stride` between them.
///
/// # Safety
/// - `output` must have length `>= output_size` in `Digest` elements.
/// - `prev_layer` must have length `>= output_size * 2` in `Digest` elements.
/// - `output` and `prev_layer` must be non-overlapping.
/// - `stride` must be a power of two and `stride <= output_size`.
pub unsafe fn poseidon2_strided_compress_layer(
    output: &mut DeviceBuffer<Digest>,
    prev_layer: &DeviceBuffer<Digest>,
    output_size: usize,
    stride: usize,
) -> Result<(), CudaError> {
    debug_assert!(stride > 0 && stride <= output_size);
    debug_assert!(output.len() >= output_size);
    debug_assert!(prev_layer.len() >= output_size * 2);
    CudaError::from_result(_poseidon2_strided_compress_layer(
        output.as_mut_ptr(),
        prev_layer.as_ptr(),
        output_size,
        stride,
    ))
}

/// Writes a new digest layer of `output_size` elements of type `Digest` to `output` by apply
/// poseidon2 compress function to adjacent pairs of digest elements from `prev_layer`.
///
/// # Safety
/// - `output` must have length `>= output_size` in `Digest` elements.
/// - `prev_layer` must have length `>= output_size * 2` in `Digest` elements.
/// - `output` and `prev_layer` must be non-overlapping.
pub unsafe fn poseidon2_adjacent_compress_layer(
    output: &mut DeviceBuffer<Digest>,
    prev_layer: &DeviceBuffer<Digest>,
    output_size: usize,
) -> Result<(), CudaError> {
    debug_assert!(output.len() >= output_size);
    debug_assert!(prev_layer.len() >= output_size * 2);
    CudaError::from_result(_poseidon2_adjacent_compress_layer(
        output.as_mut_ptr(),
        prev_layer.as_ptr(),
        output_size,
    ))
}

pub unsafe fn query_digest_layers(
    d_digest_matrix: &mut DeviceBuffer<F>,
    d_layers_ptr: &DeviceBuffer<u64>,
    d_indices: &DeviceBuffer<u64>,
    num_query: u64,
    num_layer: u64,
) -> Result<(), CudaError> {
    CudaError::from_result(_query_digest_layers(
        d_digest_matrix.as_mut_ptr(),
        d_layers_ptr.as_ptr(),
        d_indices.as_ptr(),
        num_query,
        num_layer,
    ))
}
