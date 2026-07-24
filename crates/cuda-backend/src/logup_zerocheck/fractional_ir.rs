//! Graph-IR port of [`super::fractional::fractional_sumcheck_gpu`].
//!
//! This module mirrors the eagerly-executed CUDA GKR fractional sumcheck
//! prover in `fractional.rs`, but records every kernel launch as a node on a
//! [`GraphBuilder`] instead of running it immediately. The transcript is
//! expressed through [`FiatShamirTranscriptGraphIR`], so every `observe_ext`
//! / `sample_ext` becomes a graph node as well and the whole prover can be
//! compiled and re-executed via `GraphCompiler` / `GraphExe`.
//!
//! # Status
//!
//! This is a skeleton. The kernel wrappers below insert
//! [`GraphNode::BlackboxKernel`] nodes that call the same underlying `_frac_*`
//! CUDA functions used by `fractional.rs`; the top-level driver
//! [`fractional_sumcheck_gpu_ir`] sketches the control flow but leaves the
//! per-round scheduling and challenge-BufId threading as `todo!()`s.
//!
//! # Challenges as BufIds
//!
//! The host `fractional_sumcheck_gpu` reads Fiat-Shamir challenges
//! (`lambda`, `r_prev`, `xi_j`, …) back to the host and passes them as
//! plain `EF` values to each `_frac_*` kernel. The graph-IR port cannot do
//! that: challenges live on the device as [`BufId`]s produced by
//! `sample_ext`, and inserting them into the closure captured `EF` values
//! would force a synchronous D2H sync from inside a kernel launch closure
//! (forbidden by `insert_blackbox_kernel`'s async contract).
//!
//! For now every kernel wrapper still takes challenges by value so the
//! signatures stay 1:1 with `crate::cuda::logup_zerocheck::frac_*`; a
//! follow-up will introduce graph-friendly kernel variants that read the
//! challenges directly from `BufId`-resident EF slots.

use std::{
    ffi::c_void,
    mem::{forget, size_of},
};

use crypto_compiler::graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder};
use openvm_cuda_common::{copy::cudaMemcpyKind, d_buffer::DeviceBuffer, stream::cudaStream_t};
use openvm_stark_backend::{prover::fractional_sumcheck_gkr::Frac, StarkProtocolConfig};
use p3_util::log2_strict_usize;

use super::{errors::FractionalSumcheckError, fractional::FractionalInputSize};
use crate::{
    cuda::{
        logup_zerocheck::{
            fold_ef_frac_columns, fold_ef_frac_columns_inplace, frac_add_alpha,
            frac_build_tree_layer, frac_build_tree_two_layers, frac_compute_round,
            frac_compute_round_and_fold, frac_compute_round_and_fold_inplace,
            frac_compute_round_and_revert, frac_multifold_raw, frac_precompute_m_build_raw,
            frac_precompute_m_eval_round_raw,
        },
        ntt::{bit_rev_frac_ext, bit_rev_frac_ext_build_k2},
    },
    poly::SqrtEqLayers,
    prelude::EF,
    sponge_graph_ir::FiatShamirTranscriptGraphIR,
    types::D_EF,
};

// Redeclared here so kernel-launch closures can issue device-to-device copies
// with just the raw `cudaStream_t` they receive (the safe wrappers in
// `openvm_cuda_common::copy` all take a `&GpuDeviceCtx`).
#[link(name = "cudart")]
extern "C" {
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// Buffer allocation helpers.

/// Byte size of a `Frac<EF>` element (two `EF`s = 8 base-field elements).
#[allow(dead_code)]
pub(crate) const FRAC_EF_BYTES: usize = size_of::<Frac<EF>>();
/// Byte size of a bare `EF` element.
#[allow(dead_code)]
pub(crate) const EF_BYTES: usize = size_of::<EF>();

/// Allocate a device buffer of `n` `Frac<EF>` elements on `device`.
#[allow(dead_code)]
pub(crate) fn add_frac_ef_buf(
    g: &mut GraphBuilder,
    device: DeviceType,
    name: &str,
    n: usize,
) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: crypto_compiler::quast::Quast::cst((n * FRAC_EF_BYTES) as i64),
        elem_size: FRAC_EF_BYTES,
    })
}

/// Allocate a device buffer of `n` `EF` elements on `device`.
#[allow(dead_code)]
pub(crate) fn add_ef_buf(g: &mut GraphBuilder, device: DeviceType, name: &str, n: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: crypto_compiler::quast::Quast::cst((n * EF_BYTES) as i64),
        elem_size: EF_BYTES,
    })
}

/// Allocate a `[D_EF]`-shaped `BabyBear` buffer (`D_EF * 4` bytes,
/// `elem_size = 4`) — the exact shape [`FiatShamirTranscriptGraphIR`]'s
/// `observe_ext` / `sample_ext` expect.
///
/// The bytes stored are those of an `EF` value (four `F`s in coefficient
/// order); the transcript reads them raw. See the module docs about the
/// canonical-vs-Montgomery representation caveat when the source is a CUDA
/// buffer.
pub(crate) fn add_ext_scalar_buf(g: &mut GraphBuilder, device: DeviceType, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: crypto_compiler::quast::Quast::cst((D_EF as i64) * 4),
        elem_size: 4,
    })
}

// ---------------------------------------------------------------------------
// Blackbox kernel wrappers.
//
// Each `*_ir` function has the same rough shape as the corresponding safe
// wrapper in `crate::cuda::logup_zerocheck` / `crate::cuda::ntt`: buffer
// arguments come in as [`BufId`]s, scalar/config arguments are captured by
// value in the closure, and the closure reconstructs `DeviceBuffer`s from the
// raw pointers the graph runtime hands it, calls the underlying kernel, and
// `mem::forget`s the wrappers to avoid the borrowed pointers being freed.

/// Insert a fused bitrev + K=2 tree-build kernel node. Modifies `layer` in
/// place. `layer_len` is the number of `Frac<EF>` elements owned by the
/// buffer (used only to reconstruct a borrowed [`DeviceBuffer`] view).
pub fn bit_rev_frac_ext_build_k2_ir(
    g: &mut GraphBuilder,
    layer: BufId,
    layer_len: usize,
    real_len: usize,
    total_rounds: u32,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "bit_rev_frac_ext_build_k2",
        std::iter::once(layer),
        std::iter::empty(),
        std::iter::once(true),
        move |inputs, _outputs, stream| unsafe {
            let ptr = inputs[0] as *mut (EF, EF);
            let buf = DeviceBuffer::<(EF, EF)>::from_raw_parts(ptr, layer_len);
            bit_rev_frac_ext_build_k2(&buf, real_len, total_rounds, alpha, stream)
                .expect("bit_rev_frac_ext_build_k2");
            forget(buf);
        },
    );
}

/// Insert an out-of-place bitrev kernel node for `Frac<EF>` (viewed as
/// `(EF, EF)`).
#[allow(clippy::too_many_arguments)]
pub fn bit_rev_frac_ext_ir(
    g: &mut GraphBuilder,
    src: BufId,
    dst: BufId,
    src_len: usize,
    dst_len: usize,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
) {
    g.insert_blackbox_kernel(
        "bit_rev_frac_ext",
        std::iter::once(src),
        std::iter::once(dst),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            let src = DeviceBuffer::<(EF, EF)>::from_raw_parts(inputs[0] as *mut (EF, EF), src_len);
            let dst =
                DeviceBuffer::<(EF, EF)>::from_raw_parts(outputs[0] as *mut (EF, EF), dst_len);
            bit_rev_frac_ext(
                &dst,
                &src,
                lg_domain_size,
                padded_poly_size,
                poly_count,
                stream,
            )
            .expect("bit_rev_frac_ext");
            forget(src);
            forget(dst);
        },
    );
}

/// Insert an in-place bitrev kernel node for `Frac<EF>` (viewed as
/// `(EF, EF)`). Modifies `buf` in place — this is the shape the
/// `total_leaves <= 1024` fallback path in
/// `super::fractional::fractional_sumcheck_gpu` uses.
pub fn bit_rev_frac_ext_inplace_ir(
    g: &mut GraphBuilder,
    buf: BufId,
    buf_len: usize,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
) {
    g.insert_blackbox_kernel(
        "bit_rev_frac_ext_inplace",
        std::iter::once(buf),
        std::iter::empty(),
        std::iter::once(true),
        move |inputs, _outputs, stream| unsafe {
            let view =
                DeviceBuffer::<(EF, EF)>::from_raw_parts(inputs[0] as *mut (EF, EF), buf_len);
            bit_rev_frac_ext(
                &view,
                &view,
                lg_domain_size,
                padded_poly_size,
                poly_count,
                stream,
            )
            .expect("bit_rev_frac_ext (in-place)");
            forget(view);
        },
    );
}

/// Insert a small kernel node that copies the `.p` and `.q` fields of
/// `layer[0]` (a `Frac<EF>`) into two freshly allocated `[D_EF]`-shaped
/// buffers. The output shape matches what
/// [`FiatShamirTranscriptGraphIR::observe_ext`] expects.
///
/// This is a simple pair of device-to-device `cudaMemcpyAsync`s issued on
/// the graph stream — no compute, no reads back to host.
pub fn extract_root_pq_ir(g: &mut GraphBuilder, layer: BufId, root_p: BufId, root_q: BufId) {
    g.insert_blackbox_kernel(
        "extract_root_pq",
        std::iter::once(layer),
        [root_p, root_q].into_iter(),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            let src_p = inputs[0] as *const u8;
            let src_q = src_p.add(EF_BYTES);
            let dst_p = outputs[0] as *mut u8;
            let dst_q = outputs[1] as *mut u8;
            let code = cudaMemcpyAsync(
                dst_p as *mut c_void,
                src_p as *const c_void,
                EF_BYTES,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            );
            assert_eq!(code, 0, "extract_root_pq: cudaMemcpyAsync (p) failed");
            let code = cudaMemcpyAsync(
                dst_q as *mut c_void,
                src_q as *const c_void,
                EF_BYTES,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            );
            assert_eq!(code, 0, "extract_root_pq: cudaMemcpyAsync (q) failed");
        },
    );
}

/// Insert a single-layer segment-tree build/revert kernel node.
#[allow(clippy::too_many_arguments)]
pub fn frac_build_tree_layer_ir(
    g: &mut GraphBuilder,
    layer: BufId,
    layer_len: usize,
    layer_size: usize,
    logical_len: usize,
    revert: bool,
    alpha: EF,
    apply_alpha: bool,
) {
    g.insert_blackbox_kernel(
        "frac_build_tree_layer",
        std::iter::once(layer),
        std::iter::empty(),
        std::iter::once(true),
        move |inputs, _outputs, stream| unsafe {
            let mut buf =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, layer_len);
            frac_build_tree_layer(
                &mut buf,
                layer_size,
                logical_len,
                revert,
                alpha,
                apply_alpha,
                stream,
            )
            .expect("frac_build_tree_layer");
            forget(buf);
        },
    );
}

/// Insert a fused two-layer segment-tree build kernel node.
pub fn frac_build_tree_two_layers_ir(
    g: &mut GraphBuilder,
    layer: BufId,
    layer_len: usize,
    half_i1: usize,
    logical_len: usize,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "frac_build_tree_two_layers",
        std::iter::once(layer),
        std::iter::empty(),
        std::iter::once(true),
        move |inputs, _outputs, stream| unsafe {
            let mut buf =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, layer_len);
            frac_build_tree_two_layers(&mut buf, half_i1, logical_len, alpha, stream)
                .expect("frac_build_tree_two_layers");
            forget(buf);
        },
    );
}

/// Insert an alpha-multiply-into-denominator kernel node. Modifies the
/// `count` `Frac<EF>` elements starting at `offset` inside `buf`.
///
/// Mirrors the "second half only" call in
/// `super::fractional::fractional_sumcheck_gpu` where alpha is applied to
/// the upper `total_leaves / 2` entries of a freshly bitrev'd input layer.
pub fn frac_add_alpha_ir(g: &mut GraphBuilder, buf: BufId, offset: usize, count: usize, alpha: EF) {
    g.insert_blackbox_kernel(
        "frac_add_alpha",
        std::iter::once(buf),
        std::iter::empty(),
        std::iter::once(true),
        move |inputs, _outputs, stream| unsafe {
            let base = inputs[0] as *mut Frac<EF>;
            let view = DeviceBuffer::<Frac<EF>>::from_raw_parts(base.add(offset), count);
            frac_add_alpha(&view, alpha, stream).expect("frac_add_alpha");
            forget(view);
        },
    );
}

/// Insert a `frac_compute_round` kernel node. Reads `pq_buffer` and writes
/// `out_device` + `tmp_block_sums` (both are outputs).
///
/// `eq_xi` is passed through the closure: `SqrtEqLayers` owns its own
/// `Arc`d device buffers, so cloning is cheap and safe. Callers are
/// responsible for keeping the `SqrtEqLayers` alive for as long as the
/// graph node may execute.
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_ir(
    g: &mut GraphBuilder,
    eq_xi: SqrtEqLayers,
    pq_buffer: BufId,
    pq_buffer_len: usize,
    out_device: BufId,
    out_device_len: usize,
    tmp_block_sums: BufId,
    tmp_block_sums_len: usize,
    num_x: usize,
    lambda: EF,
) {
    g.insert_blackbox_kernel(
        "frac_compute_round",
        std::iter::once(pq_buffer),
        [out_device, tmp_block_sums].into_iter(),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            let pq =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, pq_buffer_len);
            let mut out = DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, out_device_len);
            let mut tmp =
                DeviceBuffer::<EF>::from_raw_parts(outputs[1] as *mut EF, tmp_block_sums_len);
            frac_compute_round(&eq_xi, &pq, num_x, lambda, &mut out, &mut tmp, stream)
                .expect("frac_compute_round");
            forget(pq);
            forget(out);
            forget(tmp);
        },
    );
}

/// Insert a fused `frac_compute_round_and_revert` kernel node. Modifies
/// `layer` in place, writes `out_device` and `tmp_block_sums`.
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_and_revert_ir(
    g: &mut GraphBuilder,
    eq_xi: SqrtEqLayers,
    layer: BufId,
    layer_len: usize,
    out_device: BufId,
    out_device_len: usize,
    tmp_block_sums: BufId,
    tmp_block_sums_len: usize,
    num_x: usize,
    logical_len: usize,
    lambda: EF,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "frac_compute_round_and_revert",
        std::iter::once(layer),
        [out_device, tmp_block_sums].into_iter(),
        std::iter::once(true),
        move |inputs, outputs, stream| unsafe {
            let mut layer_buf =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, layer_len);
            let mut out = DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, out_device_len);
            let mut tmp =
                DeviceBuffer::<EF>::from_raw_parts(outputs[1] as *mut EF, tmp_block_sums_len);
            frac_compute_round_and_revert(
                &eq_xi,
                &mut layer_buf,
                num_x,
                logical_len,
                lambda,
                alpha,
                &mut out,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_revert");
            forget(layer_buf);
            forget(out);
            forget(tmp);
        },
    );
}

/// Insert an out-of-place fold kernel node.
#[allow(clippy::too_many_arguments)]
pub fn fold_ef_frac_columns_ir(
    g: &mut GraphBuilder,
    src: BufId,
    src_len: usize,
    dst: BufId,
    dst_len: usize,
    size: usize,
    real_len: usize,
    logical_len: usize,
    r: EF,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "fold_ef_frac_columns",
        std::iter::once(src),
        std::iter::once(dst),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            let src_buf =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, src_len);
            let mut dst_buf =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(outputs[0] as *mut Frac<EF>, dst_len);
            fold_ef_frac_columns(
                &src_buf,
                &mut dst_buf,
                size,
                real_len,
                logical_len,
                r,
                alpha,
                stream,
            )
            .expect("fold_ef_frac_columns");
            forget(src_buf);
            forget(dst_buf);
        },
    );
}

/// Insert an in-place fold kernel node.
#[allow(clippy::too_many_arguments)]
pub fn fold_ef_frac_columns_inplace_ir(
    g: &mut GraphBuilder,
    buf: BufId,
    buf_len: usize,
    size: usize,
    real_len: usize,
    logical_len: usize,
    r: EF,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "fold_ef_frac_columns_inplace",
        std::iter::once(buf),
        std::iter::empty(),
        std::iter::once(true),
        move |inputs, _outputs, stream| unsafe {
            let mut dev =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, buf_len);
            fold_ef_frac_columns_inplace(&mut dev, size, real_len, logical_len, r, alpha, stream)
                .expect("fold_ef_frac_columns_inplace");
            forget(dev);
        },
    );
}

/// Insert an out-of-place fused compute-round + fold kernel node.
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_and_fold_ir(
    g: &mut GraphBuilder,
    eq_xi: SqrtEqLayers,
    src_pq_buffer: BufId,
    src_pq_len: usize,
    dst_pq_buffer: BufId,
    dst_pq_len: usize,
    out_device: BufId,
    out_device_len: usize,
    tmp_block_sums: BufId,
    tmp_block_sums_len: usize,
    src_pq_size: usize,
    real_len: usize,
    logical_len: usize,
    lambda: EF,
    r_prev: EF,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "frac_compute_round_and_fold",
        std::iter::once(src_pq_buffer),
        [dst_pq_buffer, out_device, tmp_block_sums].into_iter(),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            let src =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, src_pq_len);
            let mut dst =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(outputs[0] as *mut Frac<EF>, dst_pq_len);
            let mut out = DeviceBuffer::<EF>::from_raw_parts(outputs[1] as *mut EF, out_device_len);
            let mut tmp =
                DeviceBuffer::<EF>::from_raw_parts(outputs[2] as *mut EF, tmp_block_sums_len);
            frac_compute_round_and_fold(
                &eq_xi,
                &src,
                &mut dst,
                src_pq_size,
                real_len,
                logical_len,
                lambda,
                r_prev,
                alpha,
                &mut out,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_fold");
            forget(src);
            forget(dst);
            forget(out);
            forget(tmp);
        },
    );
}

/// Insert an in-place fused compute-round + fold kernel node.
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_and_fold_inplace_ir(
    g: &mut GraphBuilder,
    eq_xi: SqrtEqLayers,
    pq_buffer: BufId,
    pq_buffer_len: usize,
    out_device: BufId,
    out_device_len: usize,
    tmp_block_sums: BufId,
    tmp_block_sums_len: usize,
    src_pq_size: usize,
    real_len: usize,
    logical_len: usize,
    dst_real_len: usize,
    dst_logical_len: usize,
    lambda: EF,
    r_prev: EF,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "frac_compute_round_and_fold_inplace",
        std::iter::once(pq_buffer),
        [out_device, tmp_block_sums].into_iter(),
        std::iter::once(true),
        move |inputs, outputs, stream| unsafe {
            let mut pq =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, pq_buffer_len);
            let mut out = DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, out_device_len);
            let mut tmp =
                DeviceBuffer::<EF>::from_raw_parts(outputs[1] as *mut EF, tmp_block_sums_len);
            frac_compute_round_and_fold_inplace(
                &eq_xi,
                &mut pq,
                src_pq_size,
                real_len,
                logical_len,
                dst_real_len,
                dst_logical_len,
                lambda,
                r_prev,
                alpha,
                &mut out,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_fold_inplace");
            forget(pq);
            forget(out);
            forget(tmp);
        },
    );
}

/// Insert a `frac_precompute_m_build` kernel node.
///
/// The eq tail buffers (`eq_tail_low`, `eq_tail_high`) are passed as
/// [`BufId`]s so the graph runtime resolves them at execution time. The
/// caller is responsible for pointing them at the tail slices of a
/// [`SqrtEqLayers`] (see `super::fractional::eq_tail_ptrs`).
#[allow(clippy::too_many_arguments)]
pub fn frac_precompute_m_build_ir(
    g: &mut GraphBuilder,
    pq: BufId,
    _pq_len: usize,
    eq_tail_low: BufId,
    _eq_tail_low_len: usize,
    eq_tail_high: BufId,
    _eq_tail_high_len: usize,
    m_partial: BufId,
    m_total: BufId,
    _m_total_len: usize,
    real_len: usize,
    logical_len: usize,
    rem_n: usize,
    w: usize,
    lambda: EF,
    r_prev: EF,
    alpha: EF,
    inline_fold: bool,
    eq_tail_low_cap: usize,
    tail_tile: usize,
    partial_len: usize,
) {
    g.insert_blackbox_kernel(
        "frac_precompute_m_build",
        [pq, eq_tail_low, eq_tail_high].into_iter(),
        [m_partial, m_total].into_iter(),
        [false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_precompute_m_build_raw(
                inputs[0] as *const Frac<EF>,
                real_len,
                logical_len,
                rem_n,
                w,
                lambda,
                r_prev,
                alpha,
                inline_fold,
                inputs[1] as *const EF,
                inputs[2] as *const EF,
                eq_tail_low_cap,
                tail_tile,
                outputs[0] as *mut EF,
                partial_len,
                outputs[1] as *mut EF,
                stream,
            )
            .expect("frac_precompute_m_build");
        },
    );
}

/// Insert a `frac_precompute_m_eval_round` kernel node.
#[allow(clippy::too_many_arguments)]
pub fn frac_precompute_m_eval_round_ir(
    g: &mut GraphBuilder,
    m_total: BufId,
    eq_r_prefix: BufId,
    eq_suffix: BufId,
    out: BufId,
    w: usize,
    t: usize,
) {
    g.insert_blackbox_kernel(
        "frac_precompute_m_eval_round",
        [m_total, eq_r_prefix, eq_suffix].into_iter(),
        std::iter::once(out),
        [false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_precompute_m_eval_round_raw(
                inputs[0] as *const EF,
                w,
                t,
                inputs[1] as *const EF,
                inputs[2] as *const EF,
                outputs[0] as *mut EF,
                stream,
            )
            .expect("frac_precompute_m_eval_round");
        },
    );
}

/// Insert a `frac_multifold` kernel node. `eq_r_window` is a host-uploaded
/// EF table (see `super::fractional`); pass it in as a [`BufId`].
#[allow(clippy::too_many_arguments)]
pub fn frac_multifold_ir(
    g: &mut GraphBuilder,
    src: BufId,
    dst: BufId,
    eq_r_window: BufId,
    real_len: usize,
    logical_len: usize,
    rem_n: usize,
    w: usize,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "frac_multifold",
        [src, eq_r_window].into_iter(),
        std::iter::once(dst),
        [false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_multifold_raw(
                inputs[0] as *const Frac<EF>,
                outputs[0] as *mut Frac<EF>,
                real_len,
                logical_len,
                rem_n,
                w,
                alpha,
                inputs[1] as *const EF,
                stream,
            )
            .expect("frac_multifold");
        },
    );
}

// ---------------------------------------------------------------------------
// Segment-tree build.

/// Buffers the segment-tree build phase writes; consumed by the caller to
/// keep observing / checking the root and to chain into the outer GKR loop.
#[derive(Debug, Clone, Copy)]
pub struct SegmentTreeBuildOut {
    /// `[D_EF]`-shaped BabyBear buffer holding the bytes of `root.p`.
    pub root_p: BufId,
    /// `[D_EF]`-shaped BabyBear buffer holding the bytes of `root.q`.
    pub root_q: BufId,
}

/// Emit the segment-tree build phase of the fractional GKR prover onto `g`.
///
/// Mirrors lines 665–765 of [`super::fractional::fractional_sumcheck_gpu`]:
/// bit-reversal + layer 0/1 fusion (or the ≤1024 fallback), the two-layer
/// build loop up to the root, a `frac_add_alpha` on the second half for
/// non-virtual inputs, extraction of `root.p` / `root.q`, and finally the
/// top-of-tree revert (layer_size = 2) used to seed the first inner
/// sumcheck round. The extracted `root.p` and `root.q` are observed into
/// `transcript` (`root.p` is skipped when `assert_zero` is true, matching
/// the host prover).
///
/// # Arguments
/// - `layer` must be a `Frac<EF>` buffer of exactly `sizes.real_len` elements (`sizes.real_len *
///   FRAC_EF_BYTES` bytes), already populated with the input leaves by an upstream node (e.g. an
///   `insert_memcpy` from a graph input, or the caller's leaf-generation kernel).
///
/// # One kernel per node
/// Every underlying `_frac_*` / `_bit_rev_frac_*` launch surfaces as its own
/// [`GraphNode::BlackboxKernel`], composed here via the per-kernel wrappers
/// above (`bit_rev_frac_ext_build_k2_ir`, `frac_build_tree_layer_ir`,
/// `frac_build_tree_two_layers_ir`, `frac_add_alpha_ir`,
/// `extract_root_pq_ir`). Multiple in-place modifiers of `layer` are
/// serialized by insertion order in `crypto_compiler::planner`, so no
/// fusion is required.
///
/// # Representation caveat (`assert_zero`)
/// The host prover reads `root.p` back to check it against `EF::ZERO` and
/// errors if non-zero. The graph-IR analogue would need a check kernel and
/// a way to surface the failure at `GraphExe::run` time; that is TODO. For
/// now `assert_zero` only controls whether `root.p` is observed.
///
/// # Representation caveat (`root_p` / `root_q`)
/// The bytes in the CUDA `layer` buffer are BabyBears in Montgomery form
/// (that is how the `Frac<EF>` kernels store them). The graph-IR sponge in
/// [`crate::sponge_graph_ir`] treats `[D_EF]`-shaped BabyBear inputs as
/// *canonical* BabyBears. Observing the raw extracted bytes therefore
/// advances the sponge state along a different trajectory than the host
/// `DuplexSpongeGpu` would for the same fractional root — the transcript
/// state produced here is not directly comparable to the host prover's.
/// A Montgomery→canonical conversion node is TODO.
#[allow(clippy::too_many_arguments)]
pub fn build_segment_tree_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    layer: BufId,
    sizes: FractionalInputSize,
    alpha: EF,
    assert_zero: bool,
    device: DeviceType,
) -> Result<SegmentTreeBuildOut, FractionalSumcheckError>
where
    TS: FiatShamirTranscriptGraphIR,
{
    let real_len = sizes.real_len;
    let total_leaves = sizes.logical_len;
    assert!(real_len > 0, "real_len must be nonzero");
    assert!(
        total_leaves.is_power_of_two(),
        "logical_len must be a power of two"
    );
    assert!(
        real_len <= total_leaves,
        "real_len must not exceed logical_len"
    );
    assert!(
        total_leaves / 2 <= real_len,
        "virtual padding requires logical_len / 2 <= real_len"
    );

    let total_rounds = log2_strict_usize(total_leaves);
    assert!(total_rounds > 0, "n_logup > 0 when there are interactions");

    let virtual_input = real_len < total_leaves;
    let layer_len = real_len;

    // ---- Bit-reversal + first tree layers ---------------------------------
    let start_layer_i = if total_leaves > 1024 {
        bit_rev_frac_ext_build_k2_ir(g, layer, layer_len, real_len, total_rounds as u32, alpha);
        2
    } else {
        if !virtual_input {
            bit_rev_frac_ext_inplace_ir(
                g,
                layer,
                layer_len,
                total_rounds as u32,
                total_leaves.try_into().expect("total_leaves fits in u32"),
                1,
            );
        }
        frac_build_tree_layer_ir(
            g,
            layer,
            layer_len,
            total_leaves,
            total_leaves,
            false,
            alpha,
            true,
        );
        if !virtual_input {
            let half = total_leaves / 2;
            frac_add_alpha_ir(g, layer, half, half, alpha);
        }
        1
    };

    // ---- Remaining layers: pair up via two-layer kernels ------------------
    let mut i = start_layer_i;
    while i + 1 < total_rounds {
        let half_i1 = total_leaves >> (i + 2);
        frac_build_tree_two_layers_ir(g, layer, layer_len, half_i1, total_leaves, alpha);
        i += 2;
    }
    if i < total_rounds {
        frac_build_tree_layer_ir(
            g,
            layer,
            layer_len,
            total_leaves >> i,
            total_leaves,
            false,
            alpha,
            false,
        );
    }

    // ---- Extract root (read-only on layer), then revert layer_size=2 ------
    //
    // The extract must precede the revert node. Both `extract_root_pq_ir`
    // (as a read-only reader of `layer`) and `frac_build_tree_layer_ir`
    // (as an in-place modifier of `layer`) are serialized by their
    // graph-insertion order in the planner.
    let root_p = add_ext_scalar_buf(g, device, "frac_root_p");
    let root_q = add_ext_scalar_buf(g, device, "frac_root_q");
    extract_root_pq_ir(g, layer, root_p, root_q);

    frac_build_tree_layer_ir(g, layer, layer_len, 2, total_leaves, true, alpha, false);

    if !assert_zero {
        transcript.observe_ext(g, root_p);
    }
    // TODO: when `assert_zero == true`, emit a check kernel that fails
    // `GraphExe::run` if `root.p != 0`. For now we simply skip the observe,
    // matching the host prover's branch.
    transcript.observe_ext(g, root_q);

    Ok(SegmentTreeBuildOut { root_p, root_q })
}

// ---------------------------------------------------------------------------
// Top-level driver skeleton.

/// Handles to the artifacts the graph-IR sumcheck prover produces on the
/// device. Since sumcheck polynomials, per-layer claims and final
/// randomness are constructed layer-by-layer as challenges are sampled,
/// they surface here as [`BufId`] lists rather than concrete `EF` values.
///
/// See [`super::fractional::fractional_sumcheck_gpu`]'s return type
/// ([`FracSumcheckProof`] + `Vec<EF>`) for the concrete meaning of each
/// entry — the graph-IR analogue holds the corresponding device buffers.
#[derive(Debug, Default)]
pub struct FracSumcheckProofIR {
    /// `[p_root, q_root]` in a single `[2]`-shaped EF buffer.
    pub fractional_sum: Option<BufId>,
    /// Per-layer claims — each entry holds the four `EF` values
    /// `(p_xi_0, q_xi_0, p_xi_1, q_xi_1)` as a `[4]`-shaped EF buffer.
    pub claims_per_layer: Vec<BufId>,
    /// Per-layer sumcheck polynomials. Outer vector: one entry per GKR
    /// layer (1..total_rounds). Inner vector: one `[3]`-shaped EF buffer
    /// per sumcheck round in that layer, holding s({1,2,3}).
    pub sumcheck_polys: Vec<Vec<BufId>>,
    /// Final randomness `xi_prev` as a single `[n]`-shaped EF buffer.
    pub final_randomness: Option<BufId>,
}

/// Graph-IR analogue of [`super::fractional::fractional_sumcheck_gpu`].
///
/// Records the entire GKR fractional sumcheck computation as nodes on `g`,
/// with `transcript` handling every observe/sample as its own graph node.
/// The concrete GPU computation happens later, when the graph is compiled
/// (via `GraphCompiler`) and run (via `GraphExe`).
///
/// # Arguments
/// - `leaves`: input `Frac<EF>` buffer of length `sizes.real_len`.
/// - `sizes`: real / logical fractional input sizes.
/// - `alpha`: virtual-padding scalar. Passed as a plain `EF` for now (see module docs); a follow-up
///   may accept it as a [`BufId`] too.
/// - `assert_zero`: if true, error if `root.p != 0` at runtime. In graph form this becomes a
///   `BlackboxKernel` that panics on nonzero; we don't emit it yet.
///
/// # Return value
/// Emits nodes into `g` and hands back device-resident handles to every
/// output the host prover would have returned. Actual `EF` values are
/// available after `GraphExe::run`.
///
/// # Status
/// Skeleton — the segment-tree phase, the layer loop, and the per-round
/// scheduling are all `todo!()`. The kernel wrappers above cover every
/// underlying `_frac_*` launch this function needs.
#[allow(clippy::too_many_arguments)]
pub fn fractional_sumcheck_gpu_ir<SC, TS>(
    _g: &mut GraphBuilder,
    _transcript: &mut TS,
    _leaves: BufId,
    _sizes: FractionalInputSize,
    _alpha: EF,
    _assert_zero: bool,
    _device: DeviceType,
) -> Result<FracSumcheckProofIR, FractionalSumcheckError>
where
    SC: StarkProtocolConfig<EF = EF>,
    TS: FiatShamirTranscriptGraphIR,
{
    // ---- Segment-tree build ------------------------------------------------
    //
    // Mirrors `fractional_sumcheck_gpu`'s prologue: for `total_leaves >
    // 1024` fuse bitrev + tree layers 0–1 via `bit_rev_frac_ext_build_k2_ir`;
    // otherwise emit `bit_rev_frac_ext_ir` + `frac_build_tree_layer_ir` and
    // an `frac_add_alpha_ir` on the second half for non-virtual inputs.
    // Then loop `frac_build_tree_two_layers_ir` (with a final
    // `frac_build_tree_layer_ir` if odd).
    //
    // Emit a `frac_build_tree_layer_ir(revert=true)` for the top layer, so
    // subsequent inner rounds can consume the reverted layer.

    // ---- Root read + first observes ---------------------------------------
    //
    // The host prover copies `root` (Frac<EF>) back to the host and calls
    // `transcript.observe_ext(root.p)` + `transcript.observe_ext(root.q)`.
    // In the graph, root.p / root.q live in the top-of-tree buffer slot and
    // must be exposed as [1, D_EF]-shaped `BufId`s via a small
    // "extract_ef_from_frac_slot" blackbox kernel; then we call
    // `transcript.observe_ext(g, buf)` on each.
    //
    // Same pattern for `assert_zero == true`: emit a blackbox check kernel
    // that returns an error at run time if `root.p != 0`.

    // ---- Outer GKR loop ---------------------------------------------------
    //
    // For `round in 1..total_rounds`:
    //   - Build eq buffer `SqrtEqLayers::from_xi(&xi_prev[1..])`. In graph form, xi_prev entries
    //     are BufIds; we need a graph-friendly `SqrtEqLayers` builder (out of scope for the
    //     skeleton).
    //   - Sample `lambda` via `transcript.sample_ext(g)`.
    //   - Reduce claims to a single evaluation (linear interpolation on `claims_per_layer.last()`),
    //     producing `prev_s_eval` BufId; this needs a small blackbox interpolation kernel.
    //   - Round 0: `frac_compute_round_and_revert_ir` on `layer`. Reconstruct s_evals via
    //     `reconstruct_s_evals` (blackbox kernel that also drops one eq layer),
    //     `transcript.observe_ext` each s_eval, `transcript.sample_ext` -> `r0` BufId.
    //   - Inner rounds: pick between `frac_compute_round_and_fold_ir` / `_inplace_ir` per the same
    //     `BufferScheduler` logic as the host prover. Terminal fold uses `fold_ef_frac_columns_ir`
    //     or `_inplace_ir`.
    //   - `PrecomputeM` variant: `frac_precompute_m_build_ir` -> inner
    //     `frac_precompute_m_eval_round_ir` loop -> `frac_multifold_ir` for each window, then the
    //     fold-eval tail.
    //   - Copy pq[0] and pq[pq_size/2] out into a [4]-shaped EF BufId, observe each of its four
    //     entries into the transcript, sample `mu` and prepend to r_vec to form the next `xi_prev`.
    //
    // Every EF-valued challenge above (`lambda`, `r_prev`, `xi_j`, `mu`) is
    // a BufId in graph form. The current kernel wrappers still take EF by
    // value, so a follow-up will introduce `_bufid` variants that read
    // challenges from device slots (see module docs).

    todo!("fractional_sumcheck_gpu_ir: skeleton only — see doc comments")
}

// ---------------------------------------------------------------------------
// Tests.

#[cfg(test)]
mod tests {
    use std::mem::transmute;

    use crypto_compiler::{
        graph_exe::GraphCompiler,
        graph_ir::{DeviceType, GraphBuilder},
        runtime::CompileOptions,
    };
    use openvm_cuda_common::{
        common::get_device,
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{
        cuda::{
            logup_zerocheck::{frac_add_alpha, frac_build_tree_layer, frac_build_tree_two_layers},
            ntt::{bit_rev_frac_ext, bit_rev_frac_ext_build_k2},
        },
        prelude::EF,
        sponge_graph_ir::DuplexSpongeGpuIR,
    };

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        }
    }

    fn make_host_leaves(len: usize, seed: u64) -> Vec<Frac<EF>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..len)
            .map(|_| Frac {
                p: rng.random::<EF>(),
                q: rng.random::<EF>(),
            })
            .collect()
    }

    /// Byte view of a `[Frac<EF>]` slice for D2H / H2D transfers.
    fn frac_bytes(leaves: &[Frac<EF>]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(leaves.as_ptr() as *const u8, std::mem::size_of_val(leaves))
        }
    }

    /// Copy leaves to a fresh device buffer of type `Frac<EF>`.
    fn leaves_to_device(leaves: &[Frac<EF>], ctx: &GpuDeviceCtx) -> DeviceBuffer<Frac<EF>> {
        let dev = DeviceBuffer::<Frac<EF>>::with_capacity_on(leaves.len(), ctx);
        unsafe {
            openvm_cuda_common::copy::cuda_memcpy_on::<false, true>(
                dev.as_mut_raw_ptr(),
                leaves.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(leaves),
                ctx,
            )
            .expect("H2D copy");
        }
        dev
    }

    /// Host reference: run the segment-tree build phase of
    /// `super::super::fractional::fractional_sumcheck_gpu` (lines
    /// 665–753, up to and including the top-of-tree revert of
    /// `layer_size = 2`) using the same `_frac_*` kernels.
    fn host_segment_tree(
        leaves: &[Frac<EF>],
        sizes: FractionalInputSize,
        alpha: EF,
        ctx: &GpuDeviceCtx,
    ) -> Vec<Frac<EF>> {
        let mut layer = leaves_to_device(leaves, ctx);
        let real_len = sizes.real_len;
        let total_leaves = sizes.logical_len;
        let total_rounds = log2_strict_usize(total_leaves);
        let virtual_input = real_len < total_leaves;
        let stream = ctx.stream.as_raw();

        let start_layer_i = if total_leaves > 1024 {
            unsafe {
                let buf = transmute::<&DeviceBuffer<Frac<EF>>, &DeviceBuffer<(EF, EF)>>(&layer);
                bit_rev_frac_ext_build_k2(buf, real_len, total_rounds as u32, alpha, stream)
                    .expect("bit_rev_frac_ext_build_k2");
            }
            2
        } else {
            unsafe {
                if !virtual_input {
                    let buf = transmute::<&DeviceBuffer<Frac<EF>>, &DeviceBuffer<(EF, EF)>>(&layer);
                    bit_rev_frac_ext(
                        buf,
                        buf,
                        total_rounds as u32,
                        total_leaves.try_into().unwrap(),
                        1,
                        stream,
                    )
                    .expect("bit_rev_frac_ext");
                }
                frac_build_tree_layer(
                    &mut layer,
                    total_leaves,
                    total_leaves,
                    false,
                    alpha,
                    true,
                    stream,
                )
                .expect("frac_build_tree_layer initial");
                if !virtual_input {
                    let half = total_leaves / 2;
                    let second_half_ptr = layer.as_mut_raw_ptr() as *mut Frac<EF>;
                    let second_half_buf =
                        DeviceBuffer::<Frac<EF>>::from_raw_parts(second_half_ptr.add(half), half);
                    frac_add_alpha(&second_half_buf, alpha, stream).expect("frac_add_alpha");
                    std::mem::forget(second_half_buf);
                }
            }
            1
        };

        let mut i = start_layer_i;
        while i + 1 < total_rounds {
            let half_i1 = total_leaves >> (i + 2);
            unsafe {
                frac_build_tree_two_layers(&mut layer, half_i1, total_leaves, alpha, stream)
                    .expect("frac_build_tree_two_layers");
            }
            i += 2;
        }
        if i < total_rounds {
            unsafe {
                frac_build_tree_layer(
                    &mut layer,
                    total_leaves >> i,
                    total_leaves,
                    false,
                    alpha,
                    false,
                    stream,
                )
                .expect("frac_build_tree_layer tail");
            }
        }
        unsafe {
            frac_build_tree_layer(&mut layer, 2, total_leaves, true, alpha, false, stream)
                .expect("frac_build_tree_layer revert");
        }
        ctx.stream.synchronize().expect("sync");
        layer.to_host_on(ctx).expect("D2H")
    }

    /// Build a graph that mirrors `host_segment_tree` and run it. Returns
    /// the layer buffer contents (`real_len` `Frac<EF>` elements) as bytes.
    fn ir_segment_tree(
        leaves: &[Frac<EF>],
        sizes: FractionalInputSize,
        alpha: EF,
        ctx: &GpuDeviceCtx,
    ) -> Vec<u8> {
        let device = DeviceType::Cuda(0);
        let real_len = sizes.real_len;
        let mut g = GraphBuilder::new();
        let mut sponge = DuplexSpongeGpuIR::new(&mut g, device);

        let leaves_in = add_frac_ef_buf(&mut g, device, "leaves_in", real_len);
        let layer = add_frac_ef_buf(&mut g, device, "layer", real_len);
        g.insert_memcpy(leaves_in, layer);

        build_segment_tree_ir(&mut g, &mut sponge, layer, sizes, alpha, false, device)
            .expect("build_segment_tree_ir");

        let layer_out = add_frac_ef_buf(&mut g, device, "layer_out", real_len);
        g.insert_memcpy(layer, layer_out);

        let mut exe = GraphCompiler::new()
            .device(device)
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");

        // Wire caller-visible input/output buffers.
        let leaves_dev_bytes: DeviceBuffer<u8> = frac_bytes(leaves)
            .to_device_on(ctx)
            .expect("H2D input leaves");
        let inputs = vec![leaves_dev_bytes];
        let mut outputs: Vec<DeviceBuffer<u8>> = (0..exe.num_outputs())
            .map(|i| DeviceBuffer::<u8>::with_capacity_on(exe.output_size(i), ctx))
            .collect();
        let mut scratch = DeviceBuffer::<u8>::with_capacity_on(exe.scratch_bytes().max(1), ctx);
        exe.run(ctx, &inputs, &mut outputs, &mut scratch)
            .expect("graph run");

        let out_idx = (0..exe.num_outputs())
            .find(|&i| exe.output_buf_id(i) == layer_out)
            .expect("layer_out is a graph output");
        outputs[out_idx].to_host_on(ctx).expect("D2H layer_out")
    }

    fn assert_layer_matches(sizes: FractionalInputSize, alpha: EF, seed: u64) {
        let ctx = test_ctx();
        let leaves = make_host_leaves(sizes.real_len, seed);
        let want = host_segment_tree(&leaves, sizes, alpha, &ctx);
        let got_bytes = ir_segment_tree(&leaves, sizes, alpha, &ctx);
        let want_bytes = frac_bytes(&want);
        assert_eq!(
            got_bytes.len(),
            want_bytes.len(),
            "layer byte-length mismatch: got {}, want {}",
            got_bytes.len(),
            want_bytes.len()
        );
        assert_eq!(
            got_bytes, want_bytes,
            "segment-tree layer buffers differ for sizes={sizes:?}, alpha={alpha:?}"
        );
    }

    #[test]
    fn build_segment_tree_ir_matches_host_dense_small() {
        // Small path — exercises the ≤1024 fallback (separate bitrev + layer 0
        // + frac_add_alpha) and the odd-tail single-layer branch.
        for log_n in [2usize, 4, 5, 8] {
            let n = 1usize << log_n;
            assert_layer_matches(FractionalInputSize::dense(n), EF::from_u32(7), 20260430);
        }
    }

    #[test]
    fn build_segment_tree_ir_matches_host_dense_large() {
        // Large path — exercises `bit_rev_frac_ext_build_k2`.
        for log_n in [11usize, 12] {
            let n = 1usize << log_n;
            assert_layer_matches(FractionalInputSize::dense(n), EF::from_u32(7), 20260430);
        }
    }

    #[test]
    fn build_segment_tree_ir_matches_host_virtual() {
        // Virtual-input path — real_len < logical_len.
        for (real_len, logical_len) in [
            (3usize, 4),
            (5, 8),
            (9, 16),
            (17, 32),
            (33, 64),
            (1500, 2048),
        ] {
            assert_layer_matches(
                FractionalInputSize::new(real_len, logical_len),
                EF::from_u32(7),
                20260430,
            );
        }
    }
}
