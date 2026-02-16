//! Metal kernel dispatch infrastructure.
//!
//! This module replaces the CUDA backend's `extern "C"` FFI bindings with Metal compute
//! pipeline dispatch. Each sub-module provides safe Rust wrappers around Metal kernel
//! launches, analogous to the CUDA backend's per-file FFI bindings.

pub mod batch_ntt_small;
pub mod device_info;
pub mod logup_zerocheck;
pub mod matrix;
pub mod merkle_tree;
pub mod mle_interpolate;
pub mod ntt;
pub mod poly;
pub mod sponge;
pub mod stacked_reduction;
pub mod whir;

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, Library,
    MTLSize,
};
use openvm_metal_common::device::get_context;
use openvm_metal_common::error::MetalError;

/// Metal equivalent of CUDA's warp size for dispatch calculations.
/// Apple GPUs use SIMD group size of 32.
pub const SIMD_SIZE: usize = 32;

/// Log of SIMD size.
pub const LOG_SIMD_SIZE: usize = 5;

/// Embedded Metal shader library bytes, compiled at build time.
static METAL_LIB_BYTES: &[u8] = include_bytes!(env!("METAL_KERNELS_PATH"));

/// Cached kernel pipelines, keyed by kernel function name.
static KERNELS: OnceLock<MetalKernels> = OnceLock::new();

/// Holds the compiled Metal library and cached compute pipeline states.
pub struct MetalKernels {
    library: Library,
    pipelines: std::sync::Mutex<HashMap<String, ComputePipelineState>>,
}

impl MetalKernels {
    fn new() -> Result<Self, MetalError> {
        let ctx = get_context();
        let library = ctx
            .device
            .new_library_with_data(METAL_LIB_BYTES)
            .map_err(|e| MetalError::LibraryCreation(e.to_string()))?;
        Ok(Self {
            library,
            pipelines: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Gets or creates a compute pipeline state for the named kernel function.
    pub fn get_pipeline(&self, name: &str) -> Result<ComputePipelineState, MetalError> {
        let mut cache = self.pipelines.lock().unwrap();
        if let Some(pipeline) = cache.get(name) {
            return Ok(pipeline.clone());
        }

        let func = self
            .library
            .get_function(name, None)
            .map_err(|_| MetalError::KernelNotFound(name.to_string()))?;

        let ctx = get_context();
        let pipeline = ctx
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

        cache.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }
}

/// Returns a reference to the global MetalKernels instance, initializing on first call.
pub fn get_kernels() -> &'static MetalKernels {
    KERNELS.get_or_init(|| MetalKernels::new().expect("Failed to initialize Metal kernels"))
}

/// Helper: compute 1D dispatch grid size given total threads and threads-per-threadgroup.
#[inline]
pub fn grid_size_1d(total_threads: usize, threads_per_group: usize) -> (MTLSize, MTLSize) {
    let grid = MTLSize::new(total_threads as u64, 1, 1);
    let group = MTLSize::new(threads_per_group as u64, 1, 1);
    (grid, group)
}

/// Helper: compute 2D dispatch grid size.
#[inline]
pub fn grid_size_2d(
    width: usize,
    height: usize,
    group_w: usize,
    group_h: usize,
) -> (MTLSize, MTLSize) {
    let grid = MTLSize::new(width as u64, height as u64, 1);
    let group = MTLSize::new(group_w as u64, group_h as u64, 1);
    (grid, group)
}

/// Default threads per threadgroup for 1D dispatches.
pub const DEFAULT_THREADS_PER_GROUP: usize = 256;

/// Sets a buffer argument on a compute command encoder.
///
/// # Safety
/// The pointer must be valid for `size_bytes` bytes and remain valid until the command
/// buffer completes.
#[inline]
pub unsafe fn set_buffer_from_ptr(
    encoder: &ComputeCommandEncoderRef,
    index: u64,
    ptr: *const c_void,
    size_bytes: usize,
) {
    // For Metal with StorageModeShared, we need to find the MTLBuffer that contains this pointer.
    // However, in practice we should use set_bytes for small data or set_buffer for MetalBuffers.
    // This function is used for small inline data (constants/uniforms).
    encoder.set_bytes(index, size_bytes as u64, ptr);
}

/// Dispatches a compute kernel with the given pipeline, grid, and group sizes.
/// This is synchronous: it commits the command buffer and waits for completion.
pub fn dispatch_sync(
    pipeline: &ComputePipelineState,
    grid: MTLSize,
    group: MTLSize,
    encode_fn: impl FnOnce(&ComputeCommandEncoderRef),
) -> Result<(), MetalError> {
    let ctx = get_context();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encode_fn(encoder);
    encoder.dispatch_threads(grid, group);
    encoder.end_encoding();
    openvm_metal_common::command::sync_and_check(cmd_buffer)
}

/// Dispatches a compute kernel, returning the command buffer reference for deferred sync.
/// The caller is responsible for calling `sync_and_check` on the returned command buffer.
pub fn dispatch_async<'a>(
    pipeline: &ComputePipelineState,
    grid: MTLSize,
    group: MTLSize,
    encode_fn: impl FnOnce(&ComputeCommandEncoderRef),
) -> &'a CommandBufferRef {
    let ctx = get_context();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encode_fn(encoder);
    encoder.dispatch_threads(grid, group);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer
}

/// Dispatches multiple kernels in sequence on the same command buffer, then syncs.
pub fn dispatch_multi_sync(
    dispatches: &[(
        &ComputePipelineState,
        MTLSize,
        MTLSize,
        &dyn Fn(&ComputeCommandEncoderRef),
    )],
) -> Result<(), MetalError> {
    let ctx = get_context();
    let cmd_buffer = ctx.queue.new_command_buffer();
    for (pipeline, grid, group, encode_fn) in dispatches {
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encode_fn(encoder);
        encoder.dispatch_threads(*grid, *group);
        encoder.end_encoding();
    }
    openvm_metal_common::command::sync_and_check(cmd_buffer)
}

/// Module-level sumcheck dispatch functions, ported from cuda/mod.rs sumcheck module.
pub mod sumcheck {
    use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

    use crate::{
        poly::EqEvalSegments,
        prelude::{EF, F},
    };

    use super::*;

    pub unsafe fn sumcheck_mle_round(
        input_matrices: &MetalBuffer<*const EF>,
        output: &MetalBuffer<EF>,
        tmp_block_sums: &MetalBuffer<EF>,
        widths: &MetalBuffer<u32>,
        num_matrices: u32,
        height: u32,
        d: u32,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("sumcheck_mle_round")?;
        let (grid, group) = grid_size_1d(height as usize, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input_matrices.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
            encoder.set_buffer(2, Some(tmp_block_sums.gpu_buffer()), 0);
            encoder.set_buffer(3, Some(widths.gpu_buffer()), 0);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &num_matrices as *const u32 as *const c_void);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &height as *const u32 as *const c_void);
            encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &d as *const u32 as *const c_void);
        })
    }

    pub unsafe fn fold_mle(
        input_matrices: &MetalBuffer<*const EF>,
        output_matrices: &MetalBuffer<*mut EF>,
        widths: &MetalBuffer<u32>,
        num_matrices: u16,
        output_height: u32,
        max_output_cells: u32,
        r_val: EF,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("fold_mle")?;
        let (grid, group) = grid_size_1d(max_output_cells as usize, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input_matrices.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output_matrices.gpu_buffer()), 0);
            encoder.set_buffer(2, Some(widths.gpu_buffer()), 0);
            encoder.set_bytes(3, std::mem::size_of::<u16>() as u64, &num_matrices as *const u16 as *const c_void);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &output_height as *const u32 as *const c_void);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &max_output_cells as *const u32 as *const c_void);
            encoder.set_bytes(6, std::mem::size_of::<EF>() as u64, &r_val as *const EF as *const c_void);
        })
    }

    pub unsafe fn fold_mle_column(
        buffer: &mut MetalBuffer<EF>,
        size: usize,
        r: EF,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("fold_mle_column")?;
        let (grid, group) = grid_size_1d(size / 2, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
            let size_u32 = size as u32;
            encoder.set_bytes(1, std::mem::size_of::<u32>() as u64, &size_u32 as *const u32 as *const c_void);
            encoder.set_bytes(2, std::mem::size_of::<EF>() as u64, &r as *const EF as *const c_void);
        })
    }

    pub unsafe fn batch_fold_mle(
        input_matrices: &MetalBuffer<*const EF>,
        output_matrices: &MetalBuffer<*mut EF>,
        widths: &MetalBuffer<u32>,
        num_matrices: u16,
        log_output_heights: &MetalBuffer<u8>,
        max_output_cells: u32,
        r_val: EF,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("batch_fold_mle")?;
        let (grid, group) = grid_size_1d(max_output_cells as usize, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input_matrices.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output_matrices.gpu_buffer()), 0);
            encoder.set_buffer(2, Some(widths.gpu_buffer()), 0);
            encoder.set_bytes(3, std::mem::size_of::<u16>() as u64, &num_matrices as *const u16 as *const c_void);
            encoder.set_buffer(4, Some(log_output_heights.gpu_buffer()), 0);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &max_output_cells as *const u32 as *const c_void);
            encoder.set_bytes(6, std::mem::size_of::<EF>() as u64, &r_val as *const EF as *const c_void);
        })
    }

    pub unsafe fn fold_ple_from_coeffs(
        input_coeffs: *const F,
        output: *mut EF,
        num_x: u32,
        width: u32,
        domain_size: u32,
        r: EF,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("fold_ple_from_coeffs")?;
        let total = (domain_size * width * num_x) as usize;
        let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            // These are raw pointers into MetalBuffers - we need to use set_bytes for the scalar
            // params, and for the pointers we need to find the underlying Metal buffer.
            // For now, use set_bytes for scalars and pass buffer offsets.
            // Note: This is a simplification - in a real dispatch, the caller should pass
            // the MetalBuffers directly. This stub maintains the same API as CUDA for now.
            encoder.set_bytes(0, std::mem::size_of::<u32>() as u64, &num_x as *const u32 as *const c_void);
            encoder.set_bytes(1, std::mem::size_of::<u32>() as u64, &width as *const u32 as *const c_void);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &domain_size as *const u32 as *const c_void);
            encoder.set_bytes(3, std::mem::size_of::<EF>() as u64, &r as *const EF as *const c_void);
        })
    }

    pub unsafe fn reduce_over_x_and_cols<T>(
        input: &MetalBuffer<T>,
        output: &MetalBuffer<T>,
        num_x: u32,
        num_cols: u32,
        large_domain_size: u32,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("reduce_over_x_and_cols")?;
        let total = (num_cols * large_domain_size) as usize;
        let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &num_x as *const u32 as *const c_void);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &num_cols as *const u32 as *const c_void);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &large_domain_size as *const u32 as *const c_void);
        })
    }

    pub unsafe fn triangular_fold_mle(
        output: &mut EqEvalSegments<EF>,
        input: &EqEvalSegments<EF>,
        r: EF,
        output_max_n: usize,
    ) -> Result<(), MetalError> {
        debug_assert_eq!(output.buffer.len(), 2 << output_max_n);
        debug_assert_eq!(input.buffer.len(), 4 << output_max_n);
        let pipeline = get_kernels().get_pipeline("triangular_fold_mle")?;
        let total = 2 << output_max_n;
        let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
        let output_max_n_u32 = output_max_n as u32;
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(output.buffer.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(input.buffer.gpu_buffer()), 0);
            encoder.set_bytes(2, std::mem::size_of::<EF>() as u64, &r as *const EF as *const c_void);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &output_max_n_u32 as *const u32 as *const c_void);
        })
    }
}

/// Module-level prefix scan dispatch functions, ported from cuda/mod.rs prefix module.
pub mod prefix {
    use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

    use super::*;

    pub unsafe fn prefix_scan_block_ext<T>(
        d_inout: &MetalBuffer<T>,
        length: u64,
        round_stride: u64,
        block_num: u64,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("prefix_scan_block_ext")?;
        let (grid, group) = grid_size_1d(block_num as usize * SIMD_SIZE, SIMD_SIZE);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(d_inout.gpu_buffer()), 0);
            encoder.set_bytes(1, std::mem::size_of::<u64>() as u64, &length as *const u64 as *const c_void);
            encoder.set_bytes(2, std::mem::size_of::<u64>() as u64, &round_stride as *const u64 as *const c_void);
            encoder.set_bytes(3, std::mem::size_of::<u64>() as u64, &block_num as *const u64 as *const c_void);
        })
    }

    pub unsafe fn prefix_scan_block_downsweep_ext<T>(
        d_inout: &MetalBuffer<T>,
        length: u64,
        round_stride: u64,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("prefix_scan_block_downsweep_ext")?;
        let total = length as usize;
        let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(d_inout.gpu_buffer()), 0);
            encoder.set_bytes(1, std::mem::size_of::<u64>() as u64, &length as *const u64 as *const c_void);
            encoder.set_bytes(2, std::mem::size_of::<u64>() as u64, &round_stride as *const u64 as *const c_void);
        })
    }

    pub unsafe fn prefix_scan_epilogue_ext<T>(
        d_inout: &MetalBuffer<T>,
        length: u64,
    ) -> Result<(), MetalError> {
        let pipeline = get_kernels().get_pipeline("prefix_scan_epilogue_ext")?;
        let total = length as usize;
        let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(d_inout.gpu_buffer()), 0);
            encoder.set_bytes(1, std::mem::size_of::<u64>() as u64, &length as *const u64 as *const c_void);
        })
    }
}
