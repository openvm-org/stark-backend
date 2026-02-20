use std::{mem::size_of, sync::Arc};

use getset::Getters;
use itertools::Itertools;
use openvm_metal_common::{command, d_buffer::MetalBuffer, device::get_context};
use openvm_stark_backend::{
    p3_util::log2_strict_usize,
    prover::{stacked_pcs::StackedLayout, MatrixDimensions},
};
use tracing::instrument;

use crate::{
    base::{MetalMatrix, MetalMatrixView},
    merkle_tree::MerkleTreeMetal,
    metal::{
        batch_ntt_small::batch_ntt_small,
        matrix::{batch_expand_pad, batch_expand_pad_wide},
        ntt::bit_rev,
    },
    ntt::batch_ntt,
    poly::{mle_interpolate_stages, PleMatrix},
    prelude::{Digest, F},
    MetalProverConfig, ProverError, RsCodeMatrixError, StackTracesError,
};

#[derive(Getters)]
pub struct StackedPcsDataMetal<F, Digest> {
    #[getset(get = "pub")]
    pub(crate) layout: StackedLayout,
    #[getset(get = "pub")]
    pub(crate) matrix: Option<PleMatrix<F>>,
    #[getset(get = "pub")]
    pub(crate) tree: MerkleTreeMetal<F, Digest>,
}

#[instrument(level = "info", skip_all)]
pub fn stacked_commit(
    l_skip: usize,
    n_stack: usize,
    log_blowup: usize,
    k_whir: usize,
    traces: &[&MetalMatrix<F>],
    prover_config: MetalProverConfig,
) -> Result<(Digest, StackedPcsDataMetal<F, Digest>), ProverError> {
    let layout = get_stacked_layout(l_skip, n_stack, traces);
    tracing::info!(
        height = layout.height(),
        width = layout.width(),
        "stacked_matrix_dimensions"
    );
    let opt_stacked_matrix = if prover_config.cache_stacked_matrix {
        Some(stack_traces(&layout, traces)?)
    } else {
        None
    };
    let rs_matrix = rs_code_matrix(log_blowup, &layout, traces, &opt_stacked_matrix)?;
    let tree = MerkleTreeMetal::<F, Digest>::new(
        rs_matrix,
        1 << k_whir,
        prover_config.cache_rs_code_matrix,
    )?;
    let root = tree.root();
    let data = StackedPcsDataMetal {
        layout,
        matrix: opt_stacked_matrix,
        tree,
    };
    Ok((root, data))
}

#[instrument(skip_all)]
pub fn stacked_matrix(
    l_skip: usize,
    n_stack: usize,
    traces: &[&MetalMatrix<F>],
) -> Result<(PleMatrix<F>, StackedLayout), ProverError> {
    let layout = get_stacked_layout(l_skip, n_stack, traces);
    let matrix = stack_traces(&layout, traces)?;
    Ok((matrix, layout))
}

pub(crate) fn get_stacked_layout(
    l_skip: usize,
    n_stack: usize,
    traces: &[&MetalMatrix<F>],
) -> StackedLayout {
    let sorted_meta = traces
        .iter()
        .map(|trace| {
            let log_height = log2_strict_usize(trace.height());
            (trace.width(), log_height)
        })
        .collect_vec();
    debug_assert!(sorted_meta.is_sorted_by(|a, b| a.1 >= b.1));
    StackedLayout::new(l_skip, l_skip + n_stack, sorted_meta)
}

pub(crate) fn stack_traces(
    layout: &StackedLayout,
    traces: &[&MetalMatrix<F>],
) -> Result<PleMatrix<F>, StackTracesError> {
    let l_skip = layout.l_skip();
    let height = layout.height();
    let width = layout.width();
    let mut q_evals = MetalBuffer::<F>::with_capacity(width.checked_mul(height).unwrap());
    stack_traces_into_expanded(layout, traces, &mut q_evals, height)?;
    Ok(PleMatrix::from_evals(l_skip, q_evals, height, width))
}

pub(crate) fn stack_traces_into_expanded(
    layout: &StackedLayout,
    traces: &[&MetalMatrix<F>],
    buffer: &mut MetalBuffer<F>,
    padded_height: usize,
) -> Result<(), StackTracesError> {
    #[derive(Clone, Copy)]
    struct BlitCopyTask {
        mat_idx: usize,
        src_offset: usize,
        dst_offset: usize,
        len: usize,
    }

    let l_skip = layout.l_skip();
    debug_assert_eq!(padded_height % layout.height(), 0);
    debug_assert_eq!(buffer.len() % padded_height, 0);
    debug_assert_eq!(buffer.len() / padded_height, layout.width());
    buffer.fill_zero();
    let mut batched_blit_tasks = Vec::<BlitCopyTask>::new();
    let mut blit_copy_count = 0usize;
    let mut blit_sync_count = 0usize;
    let mut expand_pad_dispatch_count = 0usize;
    let mut flush_batched_blits =
        |tasks: &mut Vec<BlitCopyTask>| -> Result<(), StackTracesError> {
            if tasks.is_empty() {
                return Ok(());
            }

            let ctx = get_context();
            command::blit_operation(&ctx.queue, |blit| {
                for task in tasks.iter() {
                    let src = traces[task.mat_idx].buffer();
                    debug_assert!(task.src_offset + task.len <= src.len());
                    debug_assert!(task.dst_offset + task.len <= buffer.len());
                    let src_offset_bytes = (task.src_offset * size_of::<F>()) as u64;
                    let dst_offset_bytes = (task.dst_offset * size_of::<F>()) as u64;
                    let copy_bytes = (task.len * size_of::<F>()) as u64;
                    blit.copy_from_buffer(
                        src.gpu_buffer(),
                        src_offset_bytes,
                        buffer.gpu_buffer(),
                        dst_offset_bytes,
                        copy_bytes,
                    );
                }
            })
            .map_err(StackTracesError::BlitCopy)?;
            tasks.clear();
            blit_sync_count += 1;
            Ok(())
        };

    for (mat_idx, j, s) in &layout.sorted_cols {
        let start = s.col_idx * padded_height + s.row_idx;
        let trace = traces[*mat_idx];
        let s_len = s.len(l_skip);
        debug_assert_eq!(trace.height(), 1 << s.log_height());
        if s.log_height() >= l_skip {
            debug_assert_eq!(trace.height(), s_len);
            batched_blit_tasks.push(BlitCopyTask {
                mat_idx: *mat_idx,
                src_offset: *j * s_len,
                dst_offset: start,
                len: s_len,
            });
            blit_copy_count += 1;
        } else {
            flush_batched_blits(&mut batched_blit_tasks)?;
            expand_pad_dispatch_count += 1;
            let stride = s.stride(l_skip);
            debug_assert_eq!(stride * trace.height(), s_len);
            unsafe {
                batch_expand_pad_wide(
                    buffer,
                    start,
                    trace.buffer(),
                    *j * trace.height(),
                    trace.height() as u32,
                    stride as u32,
                    1,
                )
                .map_err(StackTracesError::BatchExpandPadWide)?;
            }
        }
    }
    flush_batched_blits(&mut batched_blit_tasks)?;
    tracing::debug!(
        blit_copy_count,
        blit_sync_count,
        expand_pad_dispatch_count,
        "stack_traces_into_expanded_dispatch_stats"
    );
    Ok(())
}

#[instrument(skip_all)]
pub fn rs_code_matrix(
    log_blowup: usize,
    layout: &StackedLayout,
    traces: &[&MetalMatrix<F>],
    stacked_matrix: &Option<PleMatrix<F>>,
) -> Result<MetalMatrix<F>, RsCodeMatrixError> {
    let l_skip = layout.l_skip();
    let height = layout.height();
    let width = layout.width();
    debug_assert!(height >= (1 << l_skip));
    let codeword_height = height.checked_shl(log_blowup as u32).unwrap();
    let mut used_compact_mixed_path = false;
    let mut batch_expand_dispatch_count = 0usize;
    let mut batch_ntt_small_dispatch_count = 0usize;
    let mut stack_padded_height = codeword_height;
    let mut codewords = MetalBuffer::<F>::with_capacity(codeword_height * width);
    if let Some(stacked_matrix) = stacked_matrix.as_ref() {
        unsafe {
            batch_expand_pad(
                &codewords,
                0,
                &stacked_matrix.mixed,
                0,
                width as u32,
                codeword_height as u32,
                height as u32,
            )
            .map_err(RsCodeMatrixError::BatchExpandPad)?;
        }
        batch_expand_dispatch_count += 1;
    } else if l_skip > 0 && log_blowup > 0 {
        // Preserve the existing mixed-coefficient semantics while avoiding an expanded-height
        // `batch_ntt_small` over zero-padded tails.
        used_compact_mixed_path = true;
        stack_padded_height = height;
        let mut stacked_mixed = MetalBuffer::<F>::with_capacity(height * width);
        stack_traces_into_expanded(layout, traces, &mut stacked_mixed, height)
            .map_err(RsCodeMatrixError::StackTraces)?;
        let num_uni_poly = width * (height >> l_skip);
        unsafe {
            batch_ntt_small(&mut stacked_mixed, l_skip, num_uni_poly, true)
                .map_err(RsCodeMatrixError::CustomBatchIntt)?;
            batch_expand_pad(
                &codewords,
                0,
                &stacked_mixed,
                0,
                width as u32,
                codeword_height as u32,
                height as u32,
            )
            .map_err(RsCodeMatrixError::BatchExpandPad)?;
        }
        batch_ntt_small_dispatch_count += 1;
        batch_expand_dispatch_count += 1;
    } else {
        stack_traces_into_expanded(layout, traces, &mut codewords, codeword_height)
            .map_err(RsCodeMatrixError::StackTraces)?;
        if l_skip > 0 {
            let num_uni_poly = width * (codeword_height >> l_skip);
            unsafe {
                batch_ntt_small(&mut codewords, l_skip, num_uni_poly, true)
                    .map_err(RsCodeMatrixError::CustomBatchIntt)?;
            }
            batch_ntt_small_dispatch_count += 1;
        }
    }
    tracing::debug!(
        cached_stacked_matrix = stacked_matrix.is_some(),
        used_compact_mixed_path,
        stack_padded_height,
        codeword_height,
        batch_expand_dispatch_count,
        batch_ntt_small_dispatch_count,
        "rs_code_matrix_dispatch_stats"
    );
    let log_codeword_height = log2_strict_usize(codeword_height);

    if l_skip > 0 {
        unsafe {
            mle_interpolate_stages(
                &mut codewords,
                width as u16,
                codeword_height as u32,
                log_blowup as u32,
                0,
                l_skip as u32 - 1,
                false,
                false,
            )
            .map_err(|error| RsCodeMatrixError::MleInterpolateStage2d { error, step: 1 })?;
        }
    }

    unsafe {
        bit_rev(
            &codewords,
            &codewords,
            log_codeword_height as u32,
            codeword_height as u32,
            width as u32,
        )
        .map_err(RsCodeMatrixError::BitRev)?;
    }

    batch_ntt(
        &codewords,
        log_codeword_height as u32,
        0u32,
        width as u32,
        false,
        false,
    );
    let code_matrix = MetalMatrix::new(Arc::new(codewords), codeword_height, width);

    Ok(code_matrix)
}

impl<F, Digest> StackedPcsDataMetal<F, Digest> {
    pub fn mixed_view<'a>(
        &'a self,
        mat_idx: usize,
        width: usize,
    ) -> Option<MetalMatrixView<'a, F>> {
        if let Some(matrix) = self.matrix.as_ref() {
            debug_assert_eq!(self.layout.width_of(mat_idx), width);
            let s = self
                .layout
                .get(mat_idx, 0)
                .unwrap_or_else(|| panic!("Invalid matrix index: {mat_idx}"));
            let l_skip = self.layout.l_skip();
            let lifted_height = s.len(l_skip);
            let offset = s.col_idx * matrix.height() + s.row_idx;
            unsafe {
                let ptr = matrix.mixed.as_ptr().add(offset);
                Some(MetalMatrixView::from_raw_parts(ptr, lifted_height, width))
            }
        } else {
            None
        }
    }
}
