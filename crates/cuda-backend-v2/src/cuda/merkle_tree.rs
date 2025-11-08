use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::{Digest, F};

extern "C" {
    fn _poseidon2_row_hashes(
        out: *mut std::ffi::c_void,
        matrix: *const std::ffi::c_void,
        width: usize,
        height: usize,
    ) -> i32;

    // TODO EF version

    fn _poseidon2_strided_compress_layer(
        output: *mut std::ffi::c_void,
        prev_layer: *const std::ffi::c_void,
        output_size: usize,
        stride: usize,
    ) -> i32;

    fn _poseidon2_adjacent_compress_layer(
        output: *mut std::ffi::c_void,
        prev_layer: *const std::ffi::c_void,
        output_size: usize,
    ) -> i32;
}

/// Computes row hashes of `matrix` of dimensions `width` x `height` using Poseidon2 and writes the
/// digests to `out`. Memory layout expects `matrix` to be column major in `F`, and `out` is buffer
/// of `Digest`. Digests are written in order of rows.
///
/// # Safety
/// - `out` must have length `>= height` in `Digest` elements.
/// - `out` and `matrix` must be non-overlapping.
pub unsafe fn poseidon2_row_hashes(
    out: &mut DeviceBuffer<Digest>,
    matrix: &DeviceBuffer<F>,
    width: usize,
    height: usize,
) -> Result<(), CudaError> {
    debug_assert!(matrix.len() >= width * height);
    debug_assert!(out.len() >= height);
    CudaError::from_result(_poseidon2_row_hashes(
        out.as_mut_raw_ptr(),
        matrix.as_raw_ptr(),
        width,
        height,
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
        output.as_mut_raw_ptr(),
        prev_layer.as_raw_ptr(),
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
        output.as_mut_raw_ptr(),
        prev_layer.as_raw_ptr(),
        output_size,
    ))
}
