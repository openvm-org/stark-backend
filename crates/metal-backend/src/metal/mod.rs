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

use metal::{CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, Library, MTLSize};
use openvm_metal_common::device::get_context;
use openvm_metal_common::error::MetalError;
use tracing::debug;

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

        let mut candidates = vec![name.to_string()];
        if let Some(alias) = kernel_name_alias(name) {
            if alias != name {
                candidates.push(alias.to_string());
                candidates.push(format!("{alias}_kernel"));
            }
        }
        let with_kernel_suffix = format!("{name}_kernel");
        if !candidates.iter().any(|s| s == &with_kernel_suffix) {
            candidates.push(with_kernel_suffix);
        }

        for candidate in candidates {
            if let Ok(func) = self.library.get_function(&candidate, None) {
                let ctx = get_context();
                let pipeline = ctx
                    .device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;
                cache.insert(name.to_string(), pipeline.clone());
                cache.insert(candidate, pipeline.clone());
                return Ok(pipeline);
            }
        }

        Err(MetalError::KernelNotFound(name.to_string()))
    }
}

fn kernel_name_alias(name: &str) -> Option<&'static str> {
    match name {
        "bit_rev" => Some("bit_reverse"),
        "bit_rev_ext" => Some("bit_reverse_ext"),
        "bit_rev_frac_ext" => Some("bit_reverse_frac_ext"),
        "sponge_grind" => Some("grind"),
        "split_ext_to_base_col_major_matrix" => Some("split_ext_to_base_col_major"),
        "stacked_reduction_sumcheck_round0" => Some("stacked_reduction_round0_block_sum"),
        "frac_add_alpha" => Some("add_alpha"),
        "frac_compute_round" => Some("compute_round_block_sum"),
        "frac_compute_round_and_revert" => Some("compute_round_and_revert"),
        "frac_compute_round_and_fold" => Some("compute_round_and_fold"),
        "frac_compute_round_and_fold_inplace" => Some("compute_round_and_fold_inplace"),
        "frac_fold_fpext_columns" => Some("fold_ef_columns"),
        "frac_multifold" => Some("multifold"),
        "frac_precompute_m_build" => Some("precompute_m_reduce_partials"),
        "frac_precompute_m_eval_round" => Some("precompute_m_eval_round"),
        "frac_vector_scalar_multiply_ext_fp" => Some("frac_vector_scalar_multiply"),
        _ => None,
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
#[inline]
pub fn encode_dispatch(
    cmd_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    grid: MTLSize,
    group: MTLSize,
    encode_fn: impl FnOnce(&ComputeCommandEncoderRef),
) {
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encode_fn(encoder);
    encoder.dispatch_threads(grid, group);
    encoder.end_encoding();
}

/// Encodes a staged sequence of dispatches onto a single command buffer, then syncs once.
///
/// The stage closure should return the number of kernel dispatches it encoded for tracing.
pub fn dispatch_staged_sync(
    stage_name: &str,
    encode_stage: impl FnOnce(&CommandBufferRef) -> Result<usize, MetalError>,
) -> Result<(), MetalError> {
    let ctx = get_context();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let kernel_count = encode_stage(cmd_buffer)?;
    debug!(
        stage = stage_name,
        kernel_count,
        sync_count = 1usize,
        "metal_dispatch_stage"
    );
    openvm_metal_common::command::sync_and_check(cmd_buffer)
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
    encode_dispatch(cmd_buffer, pipeline, grid, group, encode_fn);
    openvm_metal_common::command::sync_and_check(cmd_buffer)
}

/// Dispatches a compute kernel, returning the command buffer reference for deferred sync.
/// The caller is responsible for calling `sync_and_check` on the returned command buffer.
pub fn dispatch_async(
    pipeline: &ComputePipelineState,
    grid: MTLSize,
    group: MTLSize,
    encode_fn: impl FnOnce(&ComputeCommandEncoderRef),
) -> &CommandBufferRef {
    let ctx = get_context();
    let cmd_buffer = ctx.queue.new_command_buffer();
    encode_dispatch(cmd_buffer, pipeline, grid, group, encode_fn);
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
    dispatch_staged_sync("dispatch_multi_sync", |cmd_buffer| {
        for (pipeline, grid, group, encode_fn) in dispatches {
            encode_dispatch(cmd_buffer, pipeline, *grid, *group, |encoder| encode_fn(encoder));
        }
        Ok(dispatches.len())
    })
}

/// Module-level sumcheck dispatch functions, ported from cuda/mod.rs sumcheck module.
pub mod sumcheck {
    use std::{ffi::c_void, mem};

    use openvm_metal_common::{d_buffer::MetalBuffer, error::MetalError};

    use crate::{
        poly::EqEvalSegments,
        prelude::{EF, F},
    };

    use super::*;

    fn block_shared_bytes(threads_per_group: usize) -> u64 {
        let simd_groups = (threads_per_group + SIMD_SIZE - 1) / SIMD_SIZE;
        (simd_groups * mem::size_of::<EF>()) as u64
    }

    fn align_threads(total: usize, threads_per_group: usize) -> usize {
        let groups = (total + threads_per_group - 1) / threads_per_group;
        groups.max(1) * threads_per_group
    }

    pub unsafe fn sumcheck_mle_round(
        input_matrices: &MetalBuffer<u64>,
        output: &MetalBuffer<EF>,
        tmp_block_sums: &MetalBuffer<EF>,
        widths: &MetalBuffer<u32>,
        num_matrices: u32,
        height: u32,
        d: u32,
    ) -> Result<(), MetalError> {
        if d == 0 {
            output.fill_zero();
            return Ok(());
        }

        let half_height = (height as usize) / 2;
        if half_height == 0 {
            output.fill_zero();
            return Ok(());
        }

        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let num_blocks = ((half_height + threads_per_group - 1) / threads_per_group) as u32;
        debug_assert!(tmp_block_sums.len() >= num_blocks as usize * d as usize);
        let total_threads = num_blocks as usize * threads_per_group;
        let shared_bytes = block_shared_bytes(threads_per_group);

        let pipeline = get_kernels().get_pipeline("sumcheck_mle_round")?;
        let (grid, group) = grid_size_1d(total_threads, threads_per_group);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input_matrices.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(tmp_block_sums.gpu_buffer()), 0);
            encoder.set_buffer(2, Some(widths.gpu_buffer()), 0);
            encoder.set_bytes(3, 4, &num_matrices as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &height as *const u32 as *const c_void);
            encoder.set_bytes(5, 4, &d as *const u32 as *const c_void);
            encoder.set_threadgroup_memory_length(0, shared_bytes);
        })?;

        let pipeline_reduce = get_kernels().get_pipeline("final_reduce_block_sums")?;
        let final_threads = d as usize * threads_per_group;
        let (grid_reduce, group_reduce) = grid_size_1d(final_threads, threads_per_group);
        dispatch_sync(&pipeline_reduce, grid_reduce, group_reduce, |encoder| {
            encoder.set_buffer(0, Some(tmp_block_sums.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
            encoder.set_bytes(2, 4, &num_blocks as *const u32 as *const c_void);
            encoder.set_bytes(3, 4, &d as *const u32 as *const c_void);
            encoder.set_threadgroup_memory_length(0, shared_bytes);
        })
    }

    pub unsafe fn sumcheck_mle_round_single(
        input: &MetalBuffer<EF>,
        output: &MetalBuffer<EF>,
        tmp_block_sums: &MetalBuffer<EF>,
        height: u32,
    ) -> Result<(), MetalError> {
        let half_height = (height as usize) / 2;
        if half_height == 0 {
            output.fill_zero();
            return Ok(());
        }

        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let num_blocks = ((half_height + threads_per_group - 1) / threads_per_group) as u32;
        debug_assert!(tmp_block_sums.len() >= num_blocks as usize);
        let total_threads = num_blocks as usize * threads_per_group;
        let shared_bytes = block_shared_bytes(threads_per_group);

        let pipeline = get_kernels().get_pipeline("sumcheck_mle_round_single")?;
        let (grid, group) = grid_size_1d(total_threads, threads_per_group);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(tmp_block_sums.gpu_buffer()), 0);
            encoder.set_bytes(2, 4, &height as *const u32 as *const c_void);
            encoder.set_threadgroup_memory_length(0, shared_bytes);
        })?;

        let d: u32 = 1;
        let pipeline_reduce = get_kernels().get_pipeline("final_reduce_block_sums")?;
        let final_threads = d as usize * threads_per_group;
        let (grid_reduce, group_reduce) = grid_size_1d(final_threads, threads_per_group);
        dispatch_sync(&pipeline_reduce, grid_reduce, group_reduce, |encoder| {
            encoder.set_buffer(0, Some(tmp_block_sums.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
            encoder.set_bytes(2, 4, &num_blocks as *const u32 as *const c_void);
            encoder.set_bytes(3, 4, &d as *const u32 as *const c_void);
            encoder.set_threadgroup_memory_length(0, shared_bytes);
        })
    }

    pub unsafe fn fold_mle(
        input_matrices: &MetalBuffer<u64>,
        output_matrices: &MetalBuffer<u64>,
        widths: &MetalBuffer<u32>,
        num_matrices: u16,
        output_height: u32,
        _max_output_cells: u32,
        r_val: EF,
    ) -> Result<(), MetalError> {
        let num_matrices_usize = num_matrices as usize;
        if num_matrices_usize == 0 || output_height == 0 {
            return Ok(());
        }

        let widths_slice = unsafe { widths.as_slice() };
        debug_assert!(widths_slice.len() >= num_matrices_usize);
        let max_width = widths_slice[..num_matrices_usize]
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize;
        if max_width == 0 {
            return Ok(());
        }

        debug_assert!(output_height.is_power_of_two());
        let log_output_height = output_height.trailing_zeros() as u8;
        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let total_threads = max_width * output_height as usize;
        let aligned_width = align_threads(total_threads, threads_per_group);
        let (grid, group) = grid_size_2d(aligned_width, num_matrices_usize, threads_per_group, 1);
        let pipeline = get_kernels().get_pipeline("fold_mle")?;
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input_matrices.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output_matrices.gpu_buffer()), 0);
            encoder.set_buffer(2, Some(widths.gpu_buffer()), 0);
            encoder.set_bytes(3, 1, &log_output_height as *const u8 as *const c_void);
            encoder.set_bytes(
                4,
                mem::size_of::<EF>() as u64,
                &r_val as *const EF as *const c_void,
            );
        })
    }

    pub unsafe fn fold_mle_single(
        input: &MetalBuffer<EF>,
        output: &MetalBuffer<EF>,
        output_height: u32,
        r_val: EF,
    ) -> Result<(), MetalError> {
        if output_height == 0 {
            return Ok(());
        }
        debug_assert!(output_height.is_power_of_two());
        let log_output_height = output_height.trailing_zeros() as u8;

        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let total_threads = align_threads(output_height as usize, threads_per_group);
        let (grid, group) = grid_size_1d(total_threads, threads_per_group);
        let pipeline = get_kernels().get_pipeline("fold_mle_single")?;
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
            encoder.set_bytes(2, 1, &log_output_height as *const u8 as *const c_void);
            encoder.set_bytes(
                3,
                mem::size_of::<EF>() as u64,
                &r_val as *const EF as *const c_void,
            );
        })
    }

    pub unsafe fn fold_mle_matrix(
        input: &MetalBuffer<EF>,
        output: &MetalBuffer<EF>,
        width: u32,
        output_height: u32,
        r_val: EF,
    ) -> Result<(), MetalError> {
        if output_height == 0 || width == 0 {
            return Ok(());
        }
        debug_assert!(output_height.is_power_of_two());
        let log_output_height = output_height.trailing_zeros() as u8;

        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let total = (output_height as usize) * (width as usize);
        let aligned = align_threads(total, threads_per_group);
        let (grid, group) = grid_size_1d(aligned, threads_per_group);
        let pipeline = get_kernels().get_pipeline("fold_mle_matrix")?;
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
            encoder.set_bytes(2, 4, &width as *const u32 as *const c_void);
            encoder.set_bytes(3, 1, &log_output_height as *const u8 as *const c_void);
            encoder.set_bytes(
                4,
                mem::size_of::<EF>() as u64,
                &r_val as *const EF as *const c_void,
            );
        })
    }

    pub unsafe fn fold_mle_column(
        buffer: &mut MetalBuffer<EF>,
        size: usize,
        r: EF,
    ) -> Result<(), MetalError> {
        if size < 2 {
            return Ok(());
        }

        let half = size / 2;
        let pipeline = get_kernels().get_pipeline("fold_mle_column")?;
        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let total_threads = align_threads(half, threads_per_group);
        let (grid, group) = grid_size_1d(total_threads, threads_per_group);
        let half_u32 = half as u32;
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(buffer.gpu_buffer()), 0);
            encoder.set_bytes(1, 4, &half_u32 as *const u32 as *const c_void);
            encoder.set_bytes(
                2,
                mem::size_of::<EF>() as u64,
                &r as *const EF as *const c_void,
            );
        })
    }

    pub unsafe fn batch_fold_mle(
        input_matrices: &MetalBuffer<u64>,
        output_matrices: &MetalBuffer<u64>,
        widths: &MetalBuffer<u32>,
        num_matrices: u16,
        log_output_heights: &MetalBuffer<u8>,
        _max_output_cells: u32,
        r_val: EF,
    ) -> Result<(), MetalError> {
        let num_matrices_usize = num_matrices as usize;
        if num_matrices_usize == 0 {
            return Ok(());
        }

        let widths_slice = unsafe { widths.as_slice() };
        debug_assert!(widths_slice.len() >= num_matrices_usize);
        let max_width = widths_slice[..num_matrices_usize]
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize;
        if max_width == 0 {
            return Ok(());
        }

        let log_heights_slice = unsafe { log_output_heights.as_slice() };
        debug_assert!(log_heights_slice.len() >= num_matrices_usize);
        let max_log = log_heights_slice[..num_matrices_usize]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let max_output_height = 1usize << (max_log as usize);
        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let total_threads = max_width * max_output_height;
        let aligned_width = align_threads(total_threads, threads_per_group);
        let (grid, group) = grid_size_2d(aligned_width, num_matrices_usize, threads_per_group, 1);
        let pipeline = get_kernels().get_pipeline("batch_fold_mle")?;
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input_matrices.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output_matrices.gpu_buffer()), 0);
            encoder.set_buffer(2, Some(widths.gpu_buffer()), 0);
            encoder.set_buffer(3, Some(log_output_heights.gpu_buffer()), 0);
            encoder.set_bytes(
                4,
                mem::size_of::<EF>() as u64,
                &r_val as *const EF as *const c_void,
            );
        })
    }

    pub unsafe fn fold_ple_from_coeffs(
        input_coeffs: &MetalBuffer<F>,
        output: &MetalBuffer<EF>,
        num_x: u32,
        width: u32,
        domain_size: u32,
        r: EF,
    ) -> Result<(), MetalError> {
        let total_polys = (num_x * width) as usize;
        if total_polys == 0 {
            return Ok(());
        }
        debug_assert!(input_coeffs.len() >= total_polys * domain_size as usize);
        debug_assert!(output.len() >= total_polys);

        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let total_threads = align_threads(total_polys, threads_per_group);
        let (grid, group) = grid_size_1d(total_threads, threads_per_group);
        let pipeline = get_kernels().get_pipeline("fold_ple_from_coeffs")?;
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input_coeffs.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
            encoder.set_bytes(2, 4, &num_x as *const u32 as *const c_void);
            encoder.set_bytes(3, 4, &width as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &domain_size as *const u32 as *const c_void);
            encoder.set_bytes(
                5,
                mem::size_of::<EF>() as u64,
                &r as *const EF as *const c_void,
            );
        })
    }

    pub unsafe fn reduce_over_x_and_cols(
        input: &MetalBuffer<F>,
        output: &MetalBuffer<F>,
        num_x: u32,
        num_cols: u32,
        large_domain_size: u32,
    ) -> Result<(), MetalError> {
        if large_domain_size == 0 {
            return Ok(());
        }

        let pipeline = get_kernels().get_pipeline("reduce_over_x_and_cols")?;
        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let total_threads = large_domain_size as usize;
        let aligned = align_threads(total_threads, threads_per_group);
        let (grid, group) = grid_size_1d(aligned, threads_per_group);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(input.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(output.gpu_buffer()), 0);
            encoder.set_bytes(2, 4, &num_x as *const u32 as *const c_void);
            encoder.set_bytes(3, 4, &num_cols as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &large_domain_size as *const u32 as *const c_void);
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
        let n_levels = output_max_n + 1;
        if n_levels == 0 {
            return Ok(());
        }
        let max_x = 1usize << output_max_n;
        let threads_per_group = DEFAULT_THREADS_PER_GROUP;
        let aligned = align_threads(max_x, threads_per_group);
        let (grid, group) = grid_size_2d(aligned, n_levels, threads_per_group, 1);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(output.buffer.gpu_buffer()), 0);
            encoder.set_buffer(1, Some(input.buffer.gpu_buffer()), 0);
            encoder.set_bytes(
                2,
                mem::size_of::<EF>() as u64,
                &r as *const EF as *const c_void,
            );
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
            encoder.set_bytes(
                1,
                std::mem::size_of::<u64>() as u64,
                &length as *const u64 as *const c_void,
            );
            encoder.set_bytes(
                2,
                std::mem::size_of::<u64>() as u64,
                &round_stride as *const u64 as *const c_void,
            );
            encoder.set_bytes(
                3,
                std::mem::size_of::<u64>() as u64,
                &block_num as *const u64 as *const c_void,
            );
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
            encoder.set_bytes(
                1,
                std::mem::size_of::<u64>() as u64,
                &length as *const u64 as *const c_void,
            );
            encoder.set_bytes(
                2,
                std::mem::size_of::<u64>() as u64,
                &round_stride as *const u64 as *const c_void,
            );
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
            encoder.set_bytes(
                1,
                std::mem::size_of::<u64>() as u64,
                &length as *const u64 as *const c_void,
            );
        })
    }
}
