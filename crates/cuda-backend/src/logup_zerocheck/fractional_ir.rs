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
//! The kernel wrappers below insert [`GraphNode::BlackboxKernel`] nodes that
//! call the same underlying `_frac_*` CUDA functions used by `fractional.rs`,
//! and the top-level driver [`fractional_sumcheck_gpu_ir`] mirrors the eager
//! prover round for round, including both the FoldEval and PrecomputeM round
//! strategies (chosen per round by the same env-driven
//! `super::fractional::choose_round_strategy`). The eager prover's host-side
//! `eval_mle_table` becomes a chain of on-device eq-hypercube stage kernels
//! ([`eq_mle_table_ir`]) so challenges never leave the device.
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
//! Kernel wrappers therefore come in two flavors: the plain `*_ir`
//! wrappers capture challenges as `EF` by value at graph-build time (only
//! usable when the challenge does not depend on a kernel output), and the
//! `*_ir_bufid` wrappers bind each challenge as an extra kernel input
//! backed by a `_dev_challenge` CUDA entry point that reads it from device
//! memory on kernel entry (see `gpu_ir_porting_guide.md` § "Kernels that
//! took EF by value"). The `do_*_ir` round composites use the `_bufid`
//! flavor exclusively.

use std::{
    cell::OnceCell,
    mem::{forget, size_of},
    sync::Arc,
};

use crypto_compiler::{
    field_ext::ef_inverse_coeffs,
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder},
    ir::{IRBuilder, Module, NodeId, ScalarType},
};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::prover::fractional_sumcheck_gkr::{Frac, FractionalGkrMemoryModel};
use p3_util::log2_strict_usize;

use super::{
    errors::FractionalSumcheckError,
    fractional::{
        choose_precompute_m_window_w, choose_round_strategy, folded_virtual_support_len,
        precompute_m_build_tail_tile, precompute_m_enabled, precompute_m_min_blocks_threshold,
        precompute_m_min_n, precompute_m_num_tail_blocks, precompute_m_tail_tile_override,
        precompute_m_target_blocks, virtual_padding_q, BufferScheduler, BufferTarget,
        FractionalInputSize, GkrRoundStrategy,
    },
    fractional_ir_dsl::frac_precompute_m_eval_round_ir_dsl,
};
use crate::{
    cuda::{
        logup_zerocheck::{
            _frac_compute_round_temp_buffer_size, fold_ef_frac_columns,
            fold_ef_frac_columns_dev_challenge, fold_ef_frac_columns_inplace,
            fold_ef_frac_columns_inplace_dev_challenge, frac_add_alpha, frac_build_tree_layer,
            frac_build_tree_two_layers, frac_compute_round, frac_compute_round_and_fold,
            frac_compute_round_and_fold_dev_challenge, frac_compute_round_and_fold_inplace,
            frac_compute_round_and_fold_inplace_dev_challenge, frac_compute_round_and_revert,
            frac_compute_round_and_revert_dev_challenge, frac_compute_round_dev_challenge,
            frac_multifold_raw, frac_precompute_m_build_dev_challenge_raw,
            frac_precompute_m_build_raw, frac_precompute_m_eval_round_raw,
        },
        ntt::{bit_rev_frac_ext, bit_rev_frac_ext_build_k2},
    },
    poly::SqrtEqLayers,
    prelude::EF,
    sponge_graph_ir::FiatShamirTranscriptGraphIR,
    types::D_EF,
};

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
/// order, Montgomery-encoded — the raw p3 / CUDA memory layout, which the
/// DSL's Montgomery codegen reads directly); the transcript reads them raw.
///
/// `elem_size` is 16 (not 4): structured kernels also bind these buffers as
/// `[1]`-shaped `ScalarType::FpExt` tensors, whose loads/stores are 128-bit
/// vector instructions — the memory planner aligns buffer offsets to
/// `elem_size`, and 16-byte alignment satisfies the BabyBear view too.
pub(crate) fn add_ext_scalar_buf(g: &mut GraphBuilder, device: DeviceType, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: crypto_compiler::quast::Quast::cst((D_EF as i64) * 4),
        elem_size: (D_EF * 4),
    })
}

/// Allocate an EF-scalar buffer (see [`add_ext_scalar_buf`]) holding the
/// bytes of a host `EF` constant.
///
/// The buffer must stay read-only: const buffers are staged once by the
/// graph runtime, so no kernel may mutate them.
pub(crate) fn ef_const_ext_scalar_buf(
    g: &mut GraphBuilder,
    device: DeviceType,
    name: &str,
    value: EF,
) -> BufId {
    use crypto_compiler::graph_ir::ConstBuf;
    let buf = add_ext_scalar_buf(g, device, name);
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(&value as *const EF as *const u8, size_of::<EF>()).to_vec()
    };
    g.insert_const(buf, ConstBuf::HostBuf(bytes));
    buf
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

/// Insert a structured kernel that copies the `.p` and `.q` fields of
/// `layer[0]` (a `Frac<EF>`) into two freshly allocated `[D_EF]`-shaped
/// BabyBear buffers. The output shape matches what
/// [`FiatShamirTranscriptGraphIR::observe_ext`] expects.
///
/// The `layer` buffer is passed in as a `Frac<EF>` array; the kernel views
/// it as `[real_len, 8]` BabyBears (each `Frac<EF>` occupies eight
/// `F`-sized slots: four for `p`, four for `q`) and gathers slots
/// `layer[0, 0..D_EF]` into `root_p` and slots `layer[0, D_EF..2*D_EF]`
/// into `root_q`.
pub fn extract_root_pq_ir(
    g: &mut GraphBuilder,
    layer: BufId,
    real_len: usize,
    root_p: BufId,
    root_q: BufId,
) {
    g.insert_kernel(
        build_extract_root_module(real_len),
        [layer],
        [root_p, root_q],
    );
}

/// Builds the `ir::Module` used by [`extract_root_pq_ir`]. The module has
/// one input (`layer: [real_len, 8] BabyBear`) and a tuple of two
/// `[D_EF] BabyBear` outputs, each written by a `compute(D_EF, ..)` loop
/// that reads its element from a constant slot of `layer[0]`.
fn build_extract_root_module(real_len: usize) -> Module {
    let mut b = IRBuilder::new();
    let layer = b.input("layer", ScalarType::BabyBear, vec![real_len, 8]);
    let root_p = b.compute(D_EF, |b, i| {
        let zero = b.const_u32(0);
        b.index(layer, &[zero, i])
    });
    let root_q = b.compute(D_EF, |b, i| {
        let zero = b.const_u32(0);
        let d_ef = b.const_u32(D_EF as u32);
        let j = b.add(i, d_ef);
        b.index(layer, &[zero, j])
    });
    let out = b.tuple(&[root_p, root_q]);
    b.finish(format!("extract_root_pq_{real_len}"), out)
}

/// Extract the claim pair `(pq[0], pq[idx1])` from a dense `Frac<EF>`
/// buffer into a fresh [`GkrLayerClaimIR`].
///
/// Generalizes [`extract_root_pq_ir`]: the structured kernel binds `pq` as
/// a `[pq_alloc_len, 8]` BabyBear tensor — its full *physical* allocation,
/// so the binding shape matches the `BufId` size even when the active
/// prefix is shorter — and gathers the `p` / `q` halves of rows `0` and
/// `idx1` into the four EF-scalar claim buffers.
pub fn extract_claim_pair_ir(
    g: &mut GraphBuilder,
    pq: BufId,
    pq_alloc_len: usize,
    idx1: usize,
    name_prefix: &str,
    device: DeviceType,
) -> GkrLayerClaimIR {
    assert!(idx1 < pq_alloc_len, "claim index out of bounds");
    let claim = GkrLayerClaimIR::alloc(g, device, name_prefix);
    g.insert_kernel(
        build_extract_claim_pair_module(pq_alloc_len, idx1),
        [pq],
        claim.as_array(),
    );
    claim
}

/// Builds the `ir::Module` used by [`extract_claim_pair_ir`]: one
/// `[alloc_len, 8]` BabyBear input and four `[D_EF]` BabyBear outputs
/// gathering the `(row, slot-offset)` pairs `(0, 0)`, `(0, D_EF)`,
/// `(idx1, 0)`, `(idx1, D_EF)` — i.e. `p_xi_0, q_xi_0, p_xi_1, q_xi_1`.
fn build_extract_claim_pair_module(alloc_len: usize, idx1: usize) -> Module {
    let mut b = IRBuilder::new();
    let pq = b.input("pq", ScalarType::BabyBear, vec![alloc_len, 8]);
    let gather = |b: &mut IRBuilder, row: u32, offset: u32| {
        b.compute(D_EF, move |b, i| {
            let row_c = b.const_u32(row);
            let off_c = b.const_u32(offset);
            let j = b.add(i, off_c);
            b.index(pq, &[row_c, j])
        })
    };
    let p0 = gather(&mut b, 0, 0);
    let q0 = gather(&mut b, 0, D_EF as u32);
    let p1 = gather(&mut b, idx1 as u32, 0);
    let q1 = gather(&mut b, idx1 as u32, D_EF as u32);
    let out = b.tuple(&[p0, q0, p1, q1]);
    b.finish(format!("extract_claim_pair_{alloc_len}_{idx1}"), out)
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

/// In-place variant of [`frac_multifold_ir`]: reads and writes `buf`.
/// The multifold kernel tolerates `src == dst` aliasing on dense buffers
/// (each output thread only overwrites slots it alone reads — see the
/// eager PrecomputeM arm in `super::fractional`); virtual-compact sources
/// must use the out-of-place form instead.
#[allow(clippy::too_many_arguments)]
pub fn frac_multifold_inplace_ir(
    g: &mut GraphBuilder,
    buf: BufId,
    eq_r_window: BufId,
    real_len: usize,
    logical_len: usize,
    rem_n: usize,
    w: usize,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "frac_multifold_inplace",
        [buf, eq_r_window].into_iter(),
        std::iter::empty(),
        [true, false].into_iter(),
        move |inputs, _outputs, stream| unsafe {
            frac_multifold_raw(
                inputs[0] as *const Frac<EF>,
                inputs[0] as *mut Frac<EF>,
                real_len,
                logical_len,
                rem_n,
                w,
                alpha,
                inputs[1] as *const EF,
                stream,
            )
            .expect("frac_multifold_inplace");
        },
    );
}

/// Insert a single stage of eq-hypercube expansion: consume `src`
/// (`step` `EF` elements holding the current `eq(x[0..i], -)` layer) and
/// write `dst` (`2*step` elements) holding `eq(x[0..i+1], -)` with `x_i`
/// inserted *from the back* (the same "nonoverlapping" convention used by
/// [`crate::poly::SqrtEqLayers::from_xi`]).
///
/// `x_i` is a `[D_EF]`-shaped `BabyBear` [`BufId`] (the shape
/// [`FiatShamirTranscriptGraphIR::sample_ext`] produces) — the challenge
/// stays on-device throughout. The launch is a structured
/// [`crate::graph_ir::GraphNode::Kernel`] so there is no host-side
/// D2H sync inside a blackbox closure.
///
/// # Layout: for each output index `j`
/// ```text
/// out[j] = src[j]        * (1 - x_i)   if j <  step
///        = src[j - step] * x_i         if j >= step
/// ```
pub fn eq_hypercube_nonoverlapping_stage_ext_ir(
    g: &mut GraphBuilder,
    src: BufId,
    dst: BufId,
    x_i: BufId,
    step: usize,
) {
    g.insert_kernel(build_eq_hypercube_stage_module(step), [src, x_i], [dst]);
}

/// The `ir::Module` behind [`eq_hypercube_nonoverlapping_stage_ext_ir`].
///
/// The module has two inputs — `src: [step] FpExt` and
/// `x_i: [D_EF] BabyBear` — and one `[2*step] FpExt` output produced by
/// a single `compute(2*step, ..)` loop.
///
/// The four `x_i` base-field coefficients are lifted to `FpExt` and
/// recombined against the extension basis `{1, t, t², t³}` once per
/// kernel invocation, then re-used inside every thread. `1 - x_i` is
/// computed the same way.
fn build_eq_hypercube_stage_module(step: usize) -> Module {
    let mut b = IRBuilder::new();
    let src = b.input("src", ScalarType::FpExt, vec![step]);
    let x_i = b.input("x_i", ScalarType::BabyBear, vec![D_EF]);

    // x_i as an FpExt scalar: lift each of the four BabyBear coeffs and
    // recombine against `{1, t, t², t³}`.
    let x_i_ext = {
        let coeffs = load_ext_coeffs(&mut b, x_i);
        let combined = fpext_from_coeffs(&mut b, coeffs);
        b.let_bound(combined)
    };
    let one_ext = b.const_fpext([1, 0, 0, 0]);
    let one_minus_x = b.sub(one_ext, x_i_ext);
    let one_minus_x = b.let_bound(one_minus_x);

    let body = b.compute(2 * step, |b, j| {
        // src index is `j mod step`; over `j ∈ [0, 2*step)` and
        // `step = 2^k` this is `j - (j / step) * step`, which the
        // DSL's index-expression checker recognizes as quasi-affine
        // (floor-div of a compute var, times a constant, subtracted
        // from the var itself).
        let step_c = b.const_u32(step as u32);
        let quot = b.div(j, step_c);
        let offset = b.mul(quot, step_c);
        let src_idx = b.sub(j, offset);
        let src_val = b.index(src, &[src_idx]);
        // Coefficient branches on the value of `j / step`, not on an
        // index — `select` on FpExt is fine.
        let is_lower = b.lt(j, step_c);
        let coeff = b.select(is_lower, one_minus_x, x_i_ext);
        b.mul(src_val, coeff)
    });
    b.finish(
        format!("eq_hypercube_nonoverlapping_stage_ext_{step}"),
        body,
    )
}

// ---------------------------------------------------------------------------
// Layer claim + eq-layer types.

/// Graph-IR mirror of one `openvm_stark_backend::proof::GkrLayerClaims`
/// entry: the four `EF` values `(p_xi_0, q_xi_0, p_xi_1, q_xi_1)` produced
/// per GKR layer, each stored on-device in its own `[D_EF]`-shaped
/// BabyBear buffer (see [`add_ext_scalar_buf`]).
///
/// This is the *singleton* form — one struct per GKR layer. The outer
/// prover keeps a `Vec<GkrLayerClaimIR>` in place of the eager prover's
/// `Vec<GkrLayerClaims<SC>>`.
#[derive(Debug, Clone, Copy)]
pub struct GkrLayerClaimIR {
    pub p_xi_0: BufId,
    pub q_xi_0: BufId,
    pub p_xi_1: BufId,
    pub q_xi_1: BufId,
}

impl GkrLayerClaimIR {
    /// Allocate the four `[D_EF]` BabyBear buffers a layer needs.
    pub fn alloc(g: &mut GraphBuilder, device: DeviceType, name_prefix: &str) -> Self {
        Self {
            p_xi_0: add_ext_scalar_buf(g, device, &format!("{name_prefix}_p_xi_0")),
            q_xi_0: add_ext_scalar_buf(g, device, &format!("{name_prefix}_q_xi_0")),
            p_xi_1: add_ext_scalar_buf(g, device, &format!("{name_prefix}_p_xi_1")),
            q_xi_1: add_ext_scalar_buf(g, device, &format!("{name_prefix}_q_xi_1")),
        }
    }

    /// The four buffers in the same order the host prover observes them
    /// into the transcript (`p_xi_0`, `q_xi_0`, `p_xi_1`, `q_xi_1`).
    pub fn as_array(&self) -> [BufId; 4] {
        [self.p_xi_0, self.q_xi_0, self.p_xi_1, self.q_xi_1]
    }
}

/// Graph-IR mirror of [`crate::poly::EqEvalLayers<EF>`]: a sequence of
/// device buffers where `layers[n]` holds the `2^n` values of
/// `eq(x[..n], -)` evaluated on the boolean hypercube `H_n`, together
/// with the shared length-1 seed layer (`eq() == 1`) at index 0.
///
/// See [`SqrtEqLayersIR`] for how these are built as `xi` is folded in.
#[derive(Debug, Clone)]
pub struct EqEvalLayersIR {
    /// `layers[i]` is a `[2^i]`-shaped `EF` buffer (`2^i * EF_BYTES` bytes,
    /// `elem_size = EF_BYTES`). `layers[0]` is a shared constant seed
    /// buffer holding `EF::ONE`.
    pub layers: Vec<BufId>,
}

impl EqEvalLayersIR {
    /// Length of `layers` minus one, matching
    /// [`crate::poly::EqEvalLayers::layers`]'s "`n + 1` layers for `n`
    /// hypercube variables" convention.
    pub fn n(&self) -> usize {
        self.layers.len() - 1
    }

    /// Buffer for the `2^n`-sized layer.
    pub fn get(&self, n: usize) -> BufId {
        self.layers[n]
    }

    /// Emit the layer-building kernels for `x` starting from `layer_0`
    /// (typically the shared seed produced by [`SqrtEqLayersIR::seed_layer`]).
    /// Mirrors [`crate::poly::EqEvalLayers::new_with_one`]: inserts each
    /// `x_i` from the back, i.e. uses the "nonoverlapping" stage kernel.
    ///
    /// Each `x_i` is a `[D_EF]`-shaped `BabyBear` [`BufId`] — the shape
    /// [`FiatShamirTranscriptGraphIR::sample_ext`] produces.
    fn new_with_one(
        g: &mut GraphBuilder,
        x: impl IntoIterator<Item = BufId>,
        layer_0: BufId,
        device: DeviceType,
    ) -> Self {
        let mut layers = vec![layer_0];
        for (i, x_i) in x.into_iter().enumerate() {
            let step = 1usize << i;
            let out_len = 2 * step;
            let out = add_ef_buf(g, device, &format!("eq_layer_{}", i + 1), out_len);
            let src = *layers.last().unwrap();
            eq_hypercube_nonoverlapping_stage_ext_ir(g, src, out, x_i, step);
            layers.push(out);
        }
        Self { layers }
    }
}

/// Graph-IR mirror of [`crate::poly::SqrtEqLayers`]: square-root memory
/// decomposition of an eq-hypercube evaluation table, materialized as two
/// `EqEvalLayersIR` chains sharing one seed layer.
///
/// A challenge vector `xi = [a, b, c, d]` is split as
/// `low` = layers for `[d, c]` (later half of `xi`, inserted back-first)
/// and `high` = layers for `[b, a]` (earlier half, inserted back-first),
/// exactly matching [`crate::poly::SqrtEqLayers::from_xi`]. The full
/// `eq(xi, -)` value at hypercube index `i` is
/// `low[i % 2^low_n] * high[i / 2^low_n]`.
///
/// See the module docs on challenge threading: `from_xi` currently takes
/// `xi` by value, matching the host `SqrtEqLayers::from_xi`; a
/// `BufId`-shaped variant is a follow-up.
#[derive(Debug, Clone)]
pub struct SqrtEqLayersIR {
    /// Layers for `xi[(n + 1) / 2..]`.
    pub low: EqEvalLayersIR,
    /// Layers for `xi[..(n + 1) / 2]`.
    pub high: EqEvalLayersIR,
}

impl SqrtEqLayersIR {
    /// The length-1 constant seed layer (`EF::ONE`) that both `low` and
    /// `high` chain off of. Emitted as an [`crypto_compiler::graph_ir::GraphNode::Const`]
    /// with a host byte payload; the graph runtime memcpys it to device
    /// at execution start.
    pub fn seed_layer(g: &mut GraphBuilder, device: DeviceType) -> BufId {
        use crypto_compiler::graph_ir::ConstBuf;
        use p3_field::PrimeCharacteristicRing;
        let buf = add_ef_buf(g, device, "eq_layer_0", 1);
        let one = EF::ONE;
        // Byte-copy the in-memory representation of `EF::ONE` (matches the
        // layout the CUDA `_eq_hypercube_*_stage_ext` kernels read).
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(&one as *const EF as *const u8, size_of::<EF>()).to_vec()
        };
        g.insert_const(buf, ConstBuf::HostBuf(bytes));
        buf
    }

    /// Build the two eq-layer chains from `xi`. Mirrors
    /// [`crate::poly::SqrtEqLayers::from_xi`] — `xi[a,b,c,d]` produces
    /// `low: [[d], [d, c]], high: [[b], [b, a]]` (each pair listed
    /// low-to-high layer index).
    ///
    /// Each `xi[j]` is an EF-valued graph node — a `[D_EF]`-shaped
    /// `BabyBear` [`BufId`], the same shape
    /// [`FiatShamirTranscriptGraphIR::sample_ext`] produces. The
    /// challenge stays on-device: no host `EF` values are captured at
    /// graph-build time.
    pub fn from_xi(g: &mut GraphBuilder, xi: &[BufId], device: DeviceType) -> Self {
        let n = xi.len();
        let low_n = n / 2;
        let high_n = n - low_n;

        let layer_0 = Self::seed_layer(g, device);
        let low =
            EqEvalLayersIR::new_with_one(g, xi[high_n..].iter().rev().copied(), layer_0, device);
        let high =
            EqEvalLayersIR::new_with_one(g, xi[..high_n].iter().rev().copied(), layer_0, device);
        debug_assert_eq!(low.n(), low_n);
        debug_assert_eq!(high.n(), high_n);
        Self { low, high }
    }

    pub fn low_n(&self) -> usize {
        self.low.n()
    }

    pub fn high_n(&self) -> usize {
        self.high.n()
    }

    pub fn max_n(&self) -> usize {
        self.low_n() + self.high_n()
    }

    /// Drop the highest layer. Drops from `high` first, then `low`.
    /// Mirrors [`crate::poly::SqrtEqLayers::drop_layer`] (a `pop_front`
    /// from the underlying `xi`).
    pub fn drop_layer(&mut self) {
        if self.high.layers.len() > 1 {
            self.high.layers.pop();
        } else if self.low.layers.len() > 1 {
            self.low.layers.pop();
        }
    }
}

/// Build the `2^n`-entry eq-MLE table of `points` on device — the graph-IR
/// analogue of the eager prover's host-side `eval_mle_table`:
/// `out[bits] = prod_i (bit_i ? x_i : 1 - x_i)` with
/// `bit_i = (bits >> (n - 1 - i)) & 1`, i.e. `points[0]` owns the top bit.
///
/// Feeding the points back-to-front through the "nonoverlapping" stage
/// kernel gives exactly this big-endian convention (each stage's point owns
/// the top bit of the doubled index, so the last-inserted `points[0]` ends
/// up most significant). Empty `points` yields the length-1 constant
/// `[EF::ONE]` seed itself.
///
/// Each point is a `[D_EF]`-shaped BabyBear [`BufId`]
/// ([`FiatShamirTranscriptGraphIR::sample_ext`]'s shape); the returned
/// buffer holds `2^n` `EF` values.
pub fn eq_mle_table_ir(g: &mut GraphBuilder, points: &[BufId], device: DeviceType) -> BufId {
    let seed = SqrtEqLayersIR::seed_layer(g, device);
    let layers = EqEvalLayersIR::new_with_one(g, points.iter().rev().copied(), seed, device);
    layers.get(points.len())
}

// ---------------------------------------------------------------------------
// Sumcheck host helpers, ported to graph-IR (Principle 4: no data-dependent
// host values — every EF that depends on a kernel output is a `BufId`).

/// Graph-IR port of `super::fractional::reduce_to_single_evaluation`.
///
/// Emits one structured `insert_kernel` node whose module computes the
/// interpolations `numer = interpolate_linear_at_01([p_xi_0, p_xi_1], mu)`
/// and `denom = interpolate_linear_at_01([q_xi_0, q_xi_1], mu)` — i.e.
/// `y0 + (y1 - y0) * mu` on `FpExt` values. All buffers are the 16-byte
/// EF-scalar buffers of [`add_ext_scalar_buf`].
pub fn reduce_to_single_evaluation_ir(
    g: &mut GraphBuilder,
    claim: GkrLayerClaimIR,
    mu: BufId,
    device: DeviceType,
) -> (BufId, BufId) {
    let numer = add_ext_scalar_buf(g, device, "numer");
    let denom = add_ext_scalar_buf(g, device, "denom");
    g.insert_kernel(
        build_reduce_to_single_evaluation_module(),
        [claim.p_xi_0, claim.p_xi_1, claim.q_xi_0, claim.q_xi_1, mu],
        [numer, denom],
    );
    (numer, denom)
}

/// The `ir::Module` behind [`reduce_to_single_evaluation_ir`].
///
/// Layout: the four claim inputs are `[1]`-shaped `FpExt` tensors (16
/// bytes each — byte-identical to the `[D_EF]` BabyBear view of the same
/// buffers), `mu` is a `[D_EF]`-shaped `BabyBear` input (the shape
/// [`FiatShamirTranscriptGraphIR::sample_ext`] produces) lifted to an
/// `FpExt` scalar, and the output is a tuple of two `[1]`-shaped `FpExt`
/// tensors. All arithmetic uses the DSL's native `FpExt` scalar type.
fn build_reduce_to_single_evaluation_module() -> Module {
    let mut b = IRBuilder::new();
    let p_xi_0 = b.input("p_xi_0", ScalarType::FpExt, vec![1]);
    let p_xi_1 = b.input("p_xi_1", ScalarType::FpExt, vec![1]);
    let q_xi_0 = b.input("q_xi_0", ScalarType::FpExt, vec![1]);
    let q_xi_1 = b.input("q_xi_1", ScalarType::FpExt, vec![1]);
    let mu = b.input("mu", ScalarType::BabyBear, vec![D_EF]);

    let i0 = b.const_u32(0);
    let [p0, p1, q0, q1] = [p_xi_0, p_xi_1, q_xi_0, q_xi_1].map(|t| b.index(t, &[i0]));
    let m = {
        let coeffs = load_ext_coeffs(&mut b, mu);
        fpext_from_coeffs(&mut b, coeffs)
    };

    // `interpolate_linear_at_01([y0, y1], mu) = y0 + (y1 - y0) * mu`.
    let interp = |b: &mut IRBuilder, y0: NodeId, y1: NodeId| -> NodeId {
        let d = b.sub(y1, y0);
        let dm = b.mul(d, m);
        b.add(y0, dm)
    };
    let n = interp(&mut b, p0, p1);
    let d = interp(&mut b, q0, q1);

    let numer = b.compute(1, move |_b, _i| n);
    let denom = b.compute(1, move |_b, _i| d);
    let out = b.tuple(&[numer, denom]);
    b.finish("reduce_to_single_evaluation".to_string(), out)
}

/// Emit `prev_s_eval = numer + lambda * denom` — the seed of each outer
/// round's running sumcheck claim, which the host prover computes on `EF`
/// values right after sampling `lambda`.
///
/// `numer` / `denom` are EF-scalar buffers bound as `[1] FpExt` (typically
/// the outputs of [`reduce_to_single_evaluation_ir`]); `lambda` is a
/// transcript challenge bound as `[D_EF]` BabyBear. Returns a fresh
/// EF-scalar buffer.
pub fn claim_combine_ir(
    g: &mut GraphBuilder,
    numer: BufId,
    denom: BufId,
    lambda: BufId,
    device: DeviceType,
) -> BufId {
    let out = add_ext_scalar_buf(g, device, "prev_s_eval_seed");
    g.insert_kernel(claim_combine_module(), [numer, denom, lambda], [out]);
    out
}

/// Cached `ir::Module` for [`claim_combine_ir`] (see
/// [`reconstruct_s_evals_module`] for why the cache matters).
fn claim_combine_module() -> Arc<Module> {
    thread_local! {
        static MODULE: OnceCell<Arc<Module>> = const { OnceCell::new() };
    }
    MODULE.with(|m| Arc::clone(m.get_or_init(|| Arc::new(build_claim_combine_module()))))
}

/// Structured module for [`claim_combine_ir`]: `out = numer + lambda *
/// denom`, all native `FpExt` arithmetic, with `lambda` lifted from its
/// `[D_EF]` BabyBear transcript shape.
fn build_claim_combine_module() -> Module {
    let mut b = IRBuilder::new();
    let numer = b.input("numer", ScalarType::FpExt, vec![1]);
    let denom = b.input("denom", ScalarType::FpExt, vec![1]);
    let lambda = b.input("lambda", ScalarType::BabyBear, vec![D_EF]);

    let i0 = b.const_u32(0);
    let n = b.index(numer, &[i0]);
    let d = b.index(denom, &[i0]);
    let l = {
        let coeffs = load_ext_coeffs(&mut b, lambda);
        fpext_from_coeffs(&mut b, coeffs)
    };
    let ld = b.mul(l, d);
    let v = b.add(n, ld);
    let out = b.compute(1, move |_b, _i| v);
    b.finish("claim_combine".to_string(), out)
}

/// Canonical `2⁻¹ = (p + 1) / 2` in BabyBear (`p = 2013265921`), used by the
/// quadratic-interpolation `halve` step.
const INV2_CANONICAL: u32 = 1_006_632_961;

// ---------------------------------------------------------------------------
// FpExt scalar helpers for structured modules.
//
// EF-valued *data* buffers ([`add_ef_buf`] / [`add_ext_scalar_buf`], both
// 16-byte aligned) bind directly as `ScalarType::FpExt` tensors and use the
// DSL's native FpExt arithmetic. Transcript-produced challenge buffers bind
// as `[D_EF]` BabyBear (the shape `sample_ext` guarantees, 4-byte aligned)
// and are lifted to an FpExt scalar with the two helpers below — the same
// convention as [`build_eq_hypercube_stage_module`].

/// Load the four base-field coefficients of a `[D_EF]`-shaped BabyBear
/// input.
pub(crate) fn load_ext_coeffs(b: &mut IRBuilder, x: NodeId) -> [NodeId; D_EF] {
    std::array::from_fn(|k| {
        let idx = b.const_u32(k as u32);
        b.index(x, &[idx])
    })
}

/// Recombine four BabyBear coefficient scalars into one `FpExt` scalar
/// against the extension basis `{1, t, t², t³}`.
pub(crate) fn fpext_from_coeffs(b: &mut IRBuilder, coeffs: [NodeId; D_EF]) -> NodeId {
    let [a0, a1, a2, a3] = coeffs;
    let e0 = b.lift_fpext(a0);
    let e1 = b.lift_fpext(a1);
    let e2 = b.lift_fpext(a2);
    let e3 = b.lift_fpext(a3);
    let t = b.const_fpext([0, 1, 0, 0]);
    let t2 = b.const_fpext([0, 0, 1, 0]);
    let t3 = b.const_fpext([0, 0, 0, 1]);
    let e1t = b.mul(e1, t);
    let e2t2 = b.mul(e2, t2);
    let e3t3 = b.mul(e3, t3);
    let sum01 = b.add(e0, e1t);
    let sum23 = b.add(e2t2, e3t3);
    b.add(sum01, sum23)
}

/// Graph-IR port of `super::fractional::reconstruct_s_evals`.
///
/// From `d_sum_evals: [2] EF` (the `s'(1)`, `s'(2)` block written by the
/// compute-round kernel), `prev_s_eval` (the sumcheck's running claim
/// carried between rounds), `xi_j` (the outer-round challenge), and
/// `eq_r_acc` (the running product of `eq(xi_{j+1..}, r_{j+1..})`),
/// produces
///
/// - `s_evals: [BufId; GKR_S_DEG]` with `s({1, 2, 3})`, ready to be observed into the transcript,
///   and
/// - `sp_evals: [BufId; GKR_S_DEG]` with `s'({0, 1, 2})` — kept because [`observe_and_update_ir`]
///   needs `sp_evals` after sampling `r`.
///
/// # Buffer bindings
/// `d_sum_evals` is the `[GKR_S_DEG - 1]` EF buffer written by the
/// compute-round kernel (an [`add_ef_buf`], 16-byte aligned — bound as
/// `FpExt`), `prev_s_eval` / `eq_r_acc` are EF-scalar buffers
/// ([`add_ext_scalar_buf`], bound as `[1] FpExt`), and `xi_j` is a
/// transcript challenge bound as `[D_EF]` BabyBear. All eight outputs are
/// fresh [`add_ext_scalar_buf`]s.
pub fn reconstruct_s_evals_ir(
    g: &mut GraphBuilder,
    d_sum_evals: BufId,
    prev_s_eval: BufId,
    xi_j: BufId,
    eq_r_acc: BufId,
    device: DeviceType,
) -> ([BufId; GKR_S_DEG], [BufId; GKR_S_DEG]) {
    let s_evals: [BufId; GKR_S_DEG] =
        std::array::from_fn(|i| add_ext_scalar_buf(g, device, &format!("s_eval_{}", i + 1)));
    let sp_evals: [BufId; GKR_S_DEG] =
        std::array::from_fn(|i| add_ext_scalar_buf(g, device, &format!("sp_eval_{i}")));
    g.insert_kernel(
        reconstruct_s_evals_module(),
        [d_sum_evals, prev_s_eval, xi_j, eq_r_acc],
        s_evals.into_iter().chain(sp_evals),
    );
    (s_evals, sp_evals)
}

/// Cached `ir::Module` for [`reconstruct_s_evals_ir`]. The module is
/// shape-static and `GraphCompiler` dedups JIT compilation by `Rc` pointer
/// identity, so sharing one instance means the per-round calls in the
/// sumcheck driver compile the kernel exactly once.
fn reconstruct_s_evals_module() -> Arc<Module> {
    thread_local! {
        static MODULE: OnceCell<Arc<Module>> = const { OnceCell::new() };
    }
    MODULE.with(|m| Arc::clone(m.get_or_init(|| Arc::new(build_reconstruct_s_evals_module()))))
}

/// Structured module for [`reconstruct_s_evals_ir`], following the
/// derivation in `docs/cuda-backend/gkr-prover.md` § "Sumcheck round
/// implementation":
///
/// ```text
/// sp[1] = d_sum_evals[0] * eq_r_acc
/// sp[2] = d_sum_evals[1] * eq_r_acc
/// sp[0] = (prev_s_eval - xi_j * sp[1]) * (1 - xi_j)⁻¹
/// s[i]  = eval_eq_mle(xi_j, i + 1) * sp(i + 1)      for i in 0..3
/// ```
///
/// with `eval_eq_mle(x, y) = 1 - y - x + 2xy` (so `eq(xi, 1) = xi`,
/// `eq(xi, 2) = 3·xi - 1`, `eq(xi, 3) = 5·xi - 2`) and
/// `sp(3) = interpolate_quadratic_at_012(sp, 3) = sp[0] + 3·(sp[2] - sp[1])`.
///
/// The `(1 - xi_j)⁻¹` step runs on the base-field coefficient view via
/// [`ef_inverse_coeffs`] (norm-based inversion); everything else is native
/// `FpExt` arithmetic. Outputs, in order: `s(1), s(2), s(3), sp(0), sp(1),
/// sp(2)`, each a `[1]`-shaped `FpExt` tensor.
fn build_reconstruct_s_evals_module() -> Module {
    let mut b = IRBuilder::new();
    let d_sum_evals = b.input("d_sum_evals", ScalarType::FpExt, vec![GKR_S_DEG - 1]);
    let prev_s_eval = b.input("prev_s_eval", ScalarType::FpExt, vec![1]);
    let xi_j = b.input("xi_j", ScalarType::BabyBear, vec![D_EF]);
    let eq_r_acc = b.input("eq_r_acc", ScalarType::FpExt, vec![1]);

    let i0 = b.const_u32(0);
    let i1 = b.const_u32(1);
    let d0 = b.index(d_sum_evals, &[i0]);
    let d1 = b.index(d_sum_evals, &[i1]);
    let prev = b.index(prev_s_eval, &[i0]);
    let eqr = b.index(eq_r_acc, &[i0]);
    let xi_coeffs = load_ext_coeffs(&mut b, xi_j);
    let xi = fpext_from_coeffs(&mut b, xi_coeffs);
    let xi = b.let_bound(xi);

    let sp1 = b.mul(d0, eqr);
    let sp2 = b.mul(d1, eqr);

    // (1 - xi_j)⁻¹ on the coefficient view: negate all coefficients, add 1
    // to the constant term, invert.
    let one_f = b.const_field(1);
    let zero_f = b.const_field(0);
    let one_minus_xi_coeffs: [NodeId; D_EF] = std::array::from_fn(|k| {
        let base = if k == 0 { one_f } else { zero_f };
        b.sub(base, xi_coeffs[k])
    });
    let inv_coeffs = ef_inverse_coeffs(&mut b, one_minus_xi_coeffs);
    let inv_one_minus_xi = fpext_from_coeffs(&mut b, inv_coeffs);

    let xi_sp1 = b.mul(xi, sp1);
    let claim_rem = b.sub(prev, xi_sp1);
    let sp0 = b.mul(claim_rem, inv_one_minus_xi);

    // s(1) = xi * sp(1) — the same product as `xi_sp1` (hash-consed).
    let s_at_1 = xi_sp1;
    // s(2) = (3·xi - 1) * sp(2).
    let one_e = b.const_fpext([1, 0, 0, 0]);
    let two_e = b.const_fpext([2, 0, 0, 0]);
    let three_e = b.const_fpext([3, 0, 0, 0]);
    let five_e = b.const_fpext([5, 0, 0, 0]);
    let three_xi = b.mul(three_e, xi);
    let eq_at_2 = b.sub(three_xi, one_e);
    let s_at_2 = b.mul(eq_at_2, sp2);
    // s(3) = (5·xi - 2) * (sp(0) + 3·(sp(2) - sp(1))).
    let five_xi = b.mul(five_e, xi);
    let eq_at_3 = b.sub(five_xi, two_e);
    let d21 = b.sub(sp2, sp1);
    let three_d21 = b.mul(three_e, d21);
    let sp_at_3 = b.add(sp0, three_d21);
    let s_at_3 = b.mul(eq_at_3, sp_at_3);

    let outs: Vec<NodeId> = [s_at_1, s_at_2, s_at_3, sp0, sp1, sp2]
        .into_iter()
        .map(|v| b.compute(1, move |_b, _i| v))
        .collect();
    let out = b.tuple(&outs);
    b.finish("reconstruct_s_evals".to_string(), out)
}

/// Number of s-poly evaluations returned per sumcheck round.
pub(crate) const GKR_S_DEG: usize = 3;

/// Graph-IR port of `super::fractional::observe_and_update`.
///
/// Emits the transcript / update block that runs after every compute
/// kernel: reconstruct `s_evals` from `d_sum_evals`, observe each of
/// them into the transcript, sample `r` from the transcript, then update
/// `prev_s_eval` and `eq_r_acc`.
///
/// Every scalar carried across rounds is a `BufId`; no host `EF` is ever
/// synthesized from a kernel output at graph-build time (Principle 4).
pub struct ObserveAndUpdateOut {
    /// The freshly sampled challenge `r` (`[D_EF]` BabyBear).
    pub r: BufId,
    /// The three s-evals just observed into the transcript (each
    /// `[D_EF]` BabyBear), kept so the caller can push them into its
    /// per-round polynomial list.
    pub s_evals: [BufId; GKR_S_DEG],
    /// Updated `prev_s_eval` — the sumcheck's running claim after this
    /// round's fold.
    pub prev_s_eval: BufId,
    /// Updated `eq_r_acc` — the running product used by the next round's
    /// `reconstruct_s_evals_ir`.
    pub eq_r_acc: BufId,
}

#[allow(clippy::too_many_arguments)]
pub fn observe_and_update_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    d_sum_evals: BufId,
    prev_s_eval: BufId,
    xi_j: BufId,
    eq_r_acc: BufId,
    device: DeviceType,
) -> ObserveAndUpdateOut
where
    TS: FiatShamirTranscriptGraphIR,
{
    let (s_evals, sp_evals) =
        reconstruct_s_evals_ir(g, d_sum_evals, prev_s_eval, xi_j, eq_r_acc, device);
    for s_eval in s_evals {
        transcript.observe_ext(g, s_eval);
    }
    let r = transcript.sample_ext(g);
    let (prev_s_eval_next, eq_r_acc_next) =
        update_running_scalars_ir(g, sp_evals, xi_j, eq_r_acc, r, device);
    ObserveAndUpdateOut {
        r,
        s_evals,
        prev_s_eval: prev_s_eval_next,
        eq_r_acc: eq_r_acc_next,
    }
}

/// Emit `eq_r = eval_eq_mle([xi_j], [r])`,
/// `eq_r_acc' = eq_r * eq_r_acc`,
/// `prev_s_eval' = eq_r * interpolate_quadratic_at_012(sp_evals, r)`.
///
/// Returns `(prev_s_eval', eq_r_acc')` as two fresh
/// [`add_ext_scalar_buf`]s. Splitting this out of
/// [`observe_and_update_ir`] keeps it free of DSL details, and lets a
/// caller reuse the update block if it computes `sp_evals` some other way
/// (e.g. from a fold + revert fused kernel).
///
/// # Buffer bindings
/// `sp_evals` / `eq_r_acc` are EF-scalar buffers bound as `[1] FpExt`;
/// `xi_j` and `r` are transcript challenges bound as `[D_EF]` BabyBear.
pub fn update_running_scalars_ir(
    g: &mut GraphBuilder,
    sp_evals: [BufId; GKR_S_DEG],
    xi_j: BufId,
    eq_r_acc: BufId,
    r: BufId,
    device: DeviceType,
) -> (BufId, BufId) {
    let prev_s_eval_next = add_ext_scalar_buf(g, device, "prev_s_eval");
    let eq_r_acc_next = add_ext_scalar_buf(g, device, "eq_r_acc");
    g.insert_kernel(
        update_running_scalars_module(),
        [sp_evals[0], sp_evals[1], sp_evals[2], xi_j, eq_r_acc, r],
        [prev_s_eval_next, eq_r_acc_next],
    );
    (prev_s_eval_next, eq_r_acc_next)
}

/// Cached `ir::Module` for [`update_running_scalars_ir`] (see
/// [`reconstruct_s_evals_module`] for why the cache matters).
fn update_running_scalars_module() -> Arc<Module> {
    thread_local! {
        static MODULE: OnceCell<Arc<Module>> = const { OnceCell::new() };
    }
    MODULE.with(|m| Arc::clone(m.get_or_init(|| Arc::new(build_update_running_scalars_module()))))
}

/// Structured module for [`update_running_scalars_ir`]:
///
/// ```text
/// eq_r         = eval_eq_mle([xi_j], [r]) = 1 - xi_j - r + 2·xi_j·r
/// eq_r_acc'    = eq_r * eq_r_acc
/// prev_s_eval' = eq_r * interpolate_quadratic_at_012(sp_evals, r)
/// ```
///
/// where the quadratic interpolation follows
/// `openvm_stark_backend::poly_common::interpolate_quadratic_at_012`:
/// `s1 = sp[1] - sp[0]`, `s2 = sp[2] - sp[1]`, `p = (s2 - s1) / 2`,
/// `q = s1 - p`, result `= (p·r + q)·r + sp[0]`. All arithmetic uses the
/// DSL's native `FpExt` scalar type. Outputs, in order:
/// `prev_s_eval'`, `eq_r_acc'`, each a `[1]`-shaped `FpExt` tensor.
fn build_update_running_scalars_module() -> Module {
    let mut b = IRBuilder::new();
    let sp0_in = b.input("sp_eval_0", ScalarType::FpExt, vec![1]);
    let sp1_in = b.input("sp_eval_1", ScalarType::FpExt, vec![1]);
    let sp2_in = b.input("sp_eval_2", ScalarType::FpExt, vec![1]);
    let xi_j = b.input("xi_j", ScalarType::BabyBear, vec![D_EF]);
    let eq_r_acc = b.input("eq_r_acc", ScalarType::FpExt, vec![1]);
    let r = b.input("r", ScalarType::BabyBear, vec![D_EF]);

    let i0 = b.const_u32(0);
    let [sp0, sp1, sp2, eqr] = [sp0_in, sp1_in, sp2_in, eq_r_acc].map(|t| b.index(t, &[i0]));
    let xi = {
        let coeffs = load_ext_coeffs(&mut b, xi_j);
        fpext_from_coeffs(&mut b, coeffs)
    };
    let rv = {
        let coeffs = load_ext_coeffs(&mut b, r);
        let v = fpext_from_coeffs(&mut b, coeffs);
        b.let_bound(v)
    };

    // eq_r = 1 - xi - r + 2·xi·r.
    let one_e = b.const_fpext([1, 0, 0, 0]);
    let two_e = b.const_fpext([2, 0, 0, 0]);
    let xir = b.mul(xi, rv);
    let two_xir = b.mul(two_e, xir);
    let one_minus_xi = b.sub(one_e, xi);
    let lin = b.sub(one_minus_xi, rv);
    let eq_r = b.add(lin, two_xir);
    let eq_r = b.let_bound(eq_r);

    let eq_r_acc_next = b.mul(eq_r, eqr);

    // interpolate_quadratic_at_012(sp_evals, r).
    let s1d = b.sub(sp1, sp0);
    let s2d = b.sub(sp2, sp1);
    let dd = b.sub(s2d, s1d);
    let inv2 = b.const_fpext([INV2_CANONICAL, 0, 0, 0]);
    let p = b.mul(dd, inv2);
    let q = b.sub(s1d, p);
    let pr = b.mul(p, rv);
    let prq = b.add(pr, q);
    let prqr = b.mul(prq, rv);
    let interp = b.add(prqr, sp0);
    let prev_s_eval_next = b.mul(eq_r, interp);

    let prev_out = b.compute(1, move |_b, _i| prev_s_eval_next);
    let eqacc_out = b.compute(1, move |_b, _i| eq_r_acc_next);
    let out = b.tuple(&[prev_out, eqacc_out]);
    b.finish("update_running_scalars".to_string(), out)
}

/// Graph-IR port of `super::fractional::do_sumcheck_round_and_revert`.
///
/// Composes the [`frac_compute_round_and_revert_ir_bufid`] launch with
/// `SqrtEqLayersIR::drop_layer` and [`observe_and_update_ir`]. Structured
/// so the three `do_*_ir` composites in this module all bottom out on the
/// same [`observe_and_update_ir`] tail, exactly mirroring the eager
/// composites in `fractional.rs`.
///
/// `d_sum_evals` must be a `[GKR_S_DEG - 1]` EF buffer and
/// `tmp_block_sums` an EF scratch buffer of at least
/// `_frac_compute_round_temp_buffer_size(pq_size / 2)` elements; both are
/// written by the kernel node and `d_sum_evals` is then consumed by the
/// reconstruction kernel inside [`observe_and_update_ir`].
#[allow(clippy::too_many_arguments)]
pub fn do_sumcheck_round_and_revert_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    eq_buffer: &mut SqrtEqLayersIR,
    layer: BufId,
    layer_len: usize,
    pq_size: usize,
    total_leaves: usize,
    lambda: BufId,
    alpha: EF,
    d_sum_evals: BufId,
    tmp_block_sums: BufId,
    prev_s_eval: BufId,
    xi_j: BufId,
    eq_r_acc: BufId,
    device: DeviceType,
) -> ObserveAndUpdateOut
where
    TS: FiatShamirTranscriptGraphIR,
{
    frac_compute_round_and_revert_ir_bufid(
        g,
        eq_buffer,
        layer,
        layer_len,
        pq_size / 2,
        total_leaves,
        lambda,
        alpha,
        d_sum_evals,
        tmp_block_sums,
    );
    eq_buffer.drop_layer();
    observe_and_update_ir(
        g,
        transcript,
        d_sum_evals,
        prev_s_eval,
        xi_j,
        eq_r_acc,
        device,
    )
}

/// Graph-IR port of `super::fractional::do_fused_sumcheck_round`.
///
/// Same tail (`drop_layer` + `observe_and_update_ir`) as
/// [`do_sumcheck_round_and_revert_ir`]; only the kernel launched
/// upstream differs ([`frac_compute_round_and_fold_ir_bufid`] —
/// out-of-place, reads from `src_pq_buffer` and writes `dst_pq_buffer`).
#[allow(clippy::too_many_arguments)]
pub fn do_fused_sumcheck_round_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    eq_buffer: &mut SqrtEqLayersIR,
    src_pq_buffer: BufId,
    dst_pq_buffer: BufId,
    src_pq_size: usize,
    src_real_len: usize,
    src_logical_len: usize,
    lambda: BufId,
    r_prev: BufId,
    alpha: EF,
    d_sum_evals: BufId,
    tmp_block_sums: BufId,
    prev_s_eval: BufId,
    xi_j: BufId,
    eq_r_acc: BufId,
    device: DeviceType,
) -> ObserveAndUpdateOut
where
    TS: FiatShamirTranscriptGraphIR,
{
    frac_compute_round_and_fold_ir_bufid(
        g,
        eq_buffer,
        src_pq_buffer,
        dst_pq_buffer,
        src_pq_size,
        src_real_len,
        src_logical_len,
        lambda,
        r_prev,
        alpha,
        d_sum_evals,
        tmp_block_sums,
    );
    eq_buffer.drop_layer();
    observe_and_update_ir(
        g,
        transcript,
        d_sum_evals,
        prev_s_eval,
        xi_j,
        eq_r_acc,
        device,
    )
}

/// Graph-IR port of `super::fractional::do_fused_sumcheck_round_inplace`.
///
/// Same shape as [`do_fused_sumcheck_round_ir`] but folds `pq_buffer` in
/// place instead of ping-ponging to a dedicated destination.
#[allow(clippy::too_many_arguments)]
pub fn do_fused_sumcheck_round_inplace_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    eq_buffer: &mut SqrtEqLayersIR,
    pq_buffer: BufId,
    src_pq_size: usize,
    src_real_len: usize,
    src_logical_len: usize,
    dst_real_len: usize,
    dst_logical_len: usize,
    lambda: BufId,
    r_prev: BufId,
    alpha: EF,
    d_sum_evals: BufId,
    tmp_block_sums: BufId,
    prev_s_eval: BufId,
    xi_j: BufId,
    eq_r_acc: BufId,
    device: DeviceType,
) -> ObserveAndUpdateOut
where
    TS: FiatShamirTranscriptGraphIR,
{
    frac_compute_round_and_fold_inplace_ir_bufid(
        g,
        eq_buffer,
        pq_buffer,
        src_pq_size,
        src_real_len,
        src_logical_len,
        dst_real_len,
        dst_logical_len,
        lambda,
        r_prev,
        alpha,
        d_sum_evals,
        tmp_block_sums,
    );
    eq_buffer.drop_layer();
    observe_and_update_ir(
        g,
        transcript,
        d_sum_evals,
        prev_s_eval,
        xi_j,
        eq_r_acc,
        device,
    )
}

// ---------------------------------------------------------------------------
// Principle-4-shaped kernel wrappers.
//
// The `_ir_bufid` variants below take every data-dependent `EF` challenge
// (`lambda`, `r_prev`, `r`) as a [`BufId`], not by value at graph-build
// time. `alpha` stays as `EF`: it is fixed per prover invocation and does
// not depend on any kernel output (Principle 4's compile-time constant
// carve-out).
//
// Each challenge `BufId` is the `[D_EF]`-shaped BabyBear buffer
// [`FiatShamirTranscriptGraphIR::sample_ext`] produces — byte-identical to
// an `EF`, so the `_dev_challenge` CUDA entry point loads it on-device
// with a plain `*ptr` read (see `gkr.cu`'s `DEV_CH` kernel templates). The
// wrappers bind the eq layers and challenges as extra kernel inputs and
// cast the runtime-resolved raw pointers straight into the raw-pointer
// safe wrappers in [`crate::cuda::logup_zerocheck`].
//
// The eq layers come from a [`SqrtEqLayersIR`]: the current top layers
// `low.get(low_n())` / `high.get(high_n())` are bound, mirroring how the
// eager wrappers read `SqrtEqLayers`.

/// Resolve the current top eq-layer buffers and `eq_low_cap` of a
/// [`SqrtEqLayersIR`], asserting they serve a round of `num_x` terms.
fn eq_layer_bufs(eq_xi: &SqrtEqLayersIR, num_x: usize) -> (BufId, BufId, usize) {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    (
        eq_xi.low.get(low_n),
        eq_xi.high.get(high_n),
        1usize << low_n,
    )
}

/// Graph-IR analogue of `super::fractional::eq_tail_ptrs`: resolve the
/// low/high eq layers for the tail portion of `eq_xi`, skipping
/// `drop_count` layers from the top (high first, then low).
///
/// Where the eager helper returns null pointers for an empty tail (never
/// hit with valid window parameters — defensive only), this returns two
/// length-1 constant `[EF::ONE]` seed layers instead: `sqrt_buffer_get`
/// then reads `low[0] * high[0] == 1`, the correct eq-value over zero
/// tail variables.
fn eq_tail_bufs(
    g: &mut GraphBuilder,
    eq_xi: &SqrtEqLayersIR,
    drop_count: usize,
    device: DeviceType,
) -> (BufId, BufId, usize) {
    let mut high_n = eq_xi.high_n();
    let mut low_n = eq_xi.low_n();
    let total_n = high_n + low_n;
    if drop_count >= total_n {
        let seed_low = SqrtEqLayersIR::seed_layer(g, device);
        let seed_high = SqrtEqLayersIR::seed_layer(g, device);
        return (seed_low, seed_high, 1);
    }
    if drop_count <= high_n {
        high_n -= drop_count;
    } else {
        low_n -= drop_count - high_n;
        high_n = 0;
    }
    (eq_xi.low.get(low_n), eq_xi.high.get(high_n), 1 << low_n)
}

/// Principle-4 variant of [`frac_compute_round_ir`]: `lambda` becomes a
/// `[D_EF]`-shaped BabyBear `BufId`, and the eq layers are graph buffers
/// from a [`SqrtEqLayersIR`].
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_ir_bufid(
    g: &mut GraphBuilder,
    eq_xi: &SqrtEqLayersIR,
    pq_buffer: BufId,
    num_x: usize,
    lambda: BufId,
    out_device: BufId,
    tmp_block_sums: BufId,
) {
    let (eq_low, eq_high, eq_low_cap) = eq_layer_bufs(eq_xi, num_x);
    g.insert_blackbox_kernel(
        "frac_compute_round_dev_challenge",
        [eq_low, eq_high, pq_buffer, lambda].into_iter(),
        [out_device, tmp_block_sums].into_iter(),
        [false, false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_compute_round_dev_challenge(
                inputs[0] as *const EF,
                inputs[1] as *const EF,
                inputs[2] as *const Frac<EF>,
                num_x,
                eq_low_cap,
                inputs[3] as *const EF,
                outputs[0] as *mut EF,
                outputs[1] as *mut EF,
                stream,
            )
            .expect("frac_compute_round_dev_challenge");
        },
    );
}

/// Principle-4 variant of [`frac_compute_round_and_revert_ir`]: `lambda`
/// becomes a `[D_EF]`-shaped BabyBear `BufId`. Modifies `layer` in place
/// (`layer_len` is its physical `Frac<EF>` length, passed to the kernel
/// as `real_len`).
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_and_revert_ir_bufid(
    g: &mut GraphBuilder,
    eq_xi: &SqrtEqLayersIR,
    layer: BufId,
    layer_len: usize,
    num_x: usize,
    logical_len: usize,
    lambda: BufId,
    alpha: EF,
    out_device: BufId,
    tmp_block_sums: BufId,
) {
    let (eq_low, eq_high, eq_low_cap) = eq_layer_bufs(eq_xi, num_x);
    g.insert_blackbox_kernel(
        "frac_compute_round_and_revert_dev_challenge",
        [eq_low, eq_high, layer, lambda].into_iter(),
        [out_device, tmp_block_sums].into_iter(),
        [false, false, true, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_compute_round_and_revert_dev_challenge(
                inputs[0] as *const EF,
                inputs[1] as *const EF,
                inputs[2] as *mut Frac<EF>,
                num_x,
                layer_len,
                logical_len,
                eq_low_cap,
                inputs[3] as *const EF,
                alpha,
                outputs[0] as *mut EF,
                outputs[1] as *mut EF,
                stream,
            )
            .expect("frac_compute_round_and_revert_dev_challenge");
        },
    );
}

/// Principle-4 variant of [`fold_ef_frac_columns_ir`]: `r` becomes a
/// `[D_EF]`-shaped BabyBear `BufId`.
#[allow(clippy::too_many_arguments)]
pub fn fold_ef_frac_columns_ir_bufid(
    g: &mut GraphBuilder,
    src: BufId,
    dst: BufId,
    size: usize,
    real_len: usize,
    logical_len: usize,
    r: BufId,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "fold_ef_frac_columns_dev_challenge",
        [src, r].into_iter(),
        std::iter::once(dst),
        [false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            fold_ef_frac_columns_dev_challenge(
                inputs[0] as *const Frac<EF>,
                outputs[0] as *mut Frac<EF>,
                size,
                real_len,
                logical_len,
                inputs[1] as *const EF,
                alpha,
                stream,
            )
            .expect("fold_ef_frac_columns_dev_challenge");
        },
    );
}

/// Principle-4 variant of [`fold_ef_frac_columns_inplace_ir`].
#[allow(clippy::too_many_arguments)]
pub fn fold_ef_frac_columns_inplace_ir_bufid(
    g: &mut GraphBuilder,
    buf: BufId,
    size: usize,
    real_len: usize,
    logical_len: usize,
    r: BufId,
    alpha: EF,
) {
    g.insert_blackbox_kernel(
        "fold_ef_frac_columns_inplace_dev_challenge",
        [buf, r].into_iter(),
        std::iter::empty(),
        [true, false].into_iter(),
        move |inputs, _outputs, stream| unsafe {
            fold_ef_frac_columns_inplace_dev_challenge(
                inputs[0] as *mut Frac<EF>,
                size,
                real_len,
                logical_len,
                inputs[1] as *const EF,
                alpha,
                stream,
            )
            .expect("fold_ef_frac_columns_inplace_dev_challenge");
        },
    );
}

/// Principle-4 variant of [`frac_compute_round_and_fold_ir`]: `lambda`
/// and `r_prev` become `[D_EF]`-shaped BabyBear `BufId`s. Out-of-place:
/// reads `src_pq_buffer`, writes `dst_pq_buffer` (they must not alias).
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_and_fold_ir_bufid(
    g: &mut GraphBuilder,
    eq_xi: &SqrtEqLayersIR,
    src_pq_buffer: BufId,
    dst_pq_buffer: BufId,
    src_pq_size: usize,
    real_len: usize,
    logical_len: usize,
    lambda: BufId,
    r_prev: BufId,
    alpha: EF,
    out_device: BufId,
    tmp_block_sums: BufId,
) {
    // Post-fold: num_x = src_pq_size / 4.
    let (eq_low, eq_high, eq_low_cap) = eq_layer_bufs(eq_xi, src_pq_size >> 2);
    g.insert_blackbox_kernel(
        "frac_compute_round_and_fold_dev_challenge",
        [eq_low, eq_high, src_pq_buffer, lambda, r_prev].into_iter(),
        [dst_pq_buffer, out_device, tmp_block_sums].into_iter(),
        [false, false, false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_compute_round_and_fold_dev_challenge(
                inputs[0] as *const EF,
                inputs[1] as *const EF,
                inputs[2] as *const Frac<EF>,
                outputs[0] as *mut Frac<EF>,
                src_pq_size,
                real_len,
                logical_len,
                eq_low_cap,
                inputs[3] as *const EF,
                inputs[4] as *const EF,
                alpha,
                outputs[1] as *mut EF,
                outputs[2] as *mut EF,
                stream,
            )
            .expect("frac_compute_round_and_fold_dev_challenge");
        },
    );
}

/// Principle-4 variant of [`frac_compute_round_and_fold_inplace_ir`]:
/// `lambda` and `r_prev` become `[D_EF]`-shaped BabyBear `BufId`s.
/// Modifies `pq_buffer` in place.
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_and_fold_inplace_ir_bufid(
    g: &mut GraphBuilder,
    eq_xi: &SqrtEqLayersIR,
    pq_buffer: BufId,
    src_pq_size: usize,
    real_len: usize,
    logical_len: usize,
    dst_real_len: usize,
    dst_logical_len: usize,
    lambda: BufId,
    r_prev: BufId,
    alpha: EF,
    out_device: BufId,
    tmp_block_sums: BufId,
) {
    // Post-fold: num_x = src_pq_size / 4.
    let (eq_low, eq_high, eq_low_cap) = eq_layer_bufs(eq_xi, src_pq_size >> 2);
    g.insert_blackbox_kernel(
        "frac_compute_round_and_fold_inplace_dev_challenge",
        [eq_low, eq_high, pq_buffer, lambda, r_prev].into_iter(),
        [out_device, tmp_block_sums].into_iter(),
        [false, false, true, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_compute_round_and_fold_inplace_dev_challenge(
                inputs[0] as *const EF,
                inputs[1] as *const EF,
                inputs[2] as *mut Frac<EF>,
                src_pq_size,
                real_len,
                logical_len,
                dst_real_len,
                dst_logical_len,
                eq_low_cap,
                inputs[3] as *const EF,
                inputs[4] as *const EF,
                alpha,
                outputs[0] as *mut EF,
                outputs[1] as *mut EF,
                stream,
            )
            .expect("frac_compute_round_and_fold_inplace_dev_challenge");
        },
    );
}

/// Principle-4 variant of [`frac_precompute_m_build_ir`]: `lambda` and
/// `r_prev` become `[D_EF]`-shaped `BabyBear` `BufId`s. `r_prev` is only
/// read on-device when `inline_fold` is true, but must always be a valid
/// buffer.
#[allow(clippy::too_many_arguments)]
pub fn frac_precompute_m_build_ir_bufid(
    g: &mut GraphBuilder,
    pq: BufId,
    eq_tail_low: BufId,
    eq_tail_high: BufId,
    m_partial: BufId,
    m_total: BufId,
    real_len: usize,
    logical_len: usize,
    rem_n: usize,
    w: usize,
    lambda: BufId,
    r_prev: BufId,
    alpha: EF,
    inline_fold: bool,
    eq_tail_low_cap: usize,
    tail_tile: usize,
    partial_len: usize,
) {
    g.insert_blackbox_kernel(
        "frac_precompute_m_build_dev_challenge",
        [pq, eq_tail_low, eq_tail_high, lambda, r_prev].into_iter(),
        [m_partial, m_total].into_iter(),
        [false, false, false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            frac_precompute_m_build_dev_challenge_raw(
                inputs[0] as *const Frac<EF>,
                real_len,
                logical_len,
                rem_n,
                w,
                inputs[3] as *const EF,
                inputs[4] as *const EF,
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
            .expect("frac_precompute_m_build_dev_challenge");
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
/// # Representation (`root_p` / `root_q`)
/// The bytes in the CUDA `layer` buffer are BabyBears in Montgomery form
/// (that is how the `Frac<EF>` kernels store them). Since the DSL's
/// Montgomery codegen switch, `ScalarType::BabyBear` buffers use the same
/// encoding, so the graph-IR sponge in [`crate::sponge_graph_ir`] absorbs
/// the raw extracted bytes as the same field elements the host
/// `DuplexSpongeGpu` would observe — the transcript trajectories match.
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
    extract_root_pq_ir(g, layer, real_len, root_p, root_q);

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
/// they surface here as [`BufId`]s rather than concrete `EF` values.
///
/// Mirrors [`super::fractional::fractional_sumcheck_gpu`]'s return type
/// (`FracSumcheckProof` + the final `xi_prev` randomness), with every `EF`
/// replaced by the device buffer that will hold it after `GraphExe::run`.
#[derive(Debug)]
pub struct FracSumcheckProofIR {
    /// `(root.p, root.q)`, each an EF-scalar buffer
    /// ([`add_ext_scalar_buf`]).
    pub fractional_sum: (BufId, BufId),
    /// Per-layer claims (`total_rounds` entries), each four EF-scalar
    /// buffers in transcript-observe order.
    pub claims_per_layer: Vec<GkrLayerClaimIR>,
    /// Per-layer sumcheck polynomials. Outer: one entry per GKR layer
    /// (outer rounds `1..total_rounds`). Inner: one `s({1, 2, 3})` triple
    /// of EF-scalar buffers per inner sumcheck round.
    pub sumcheck_polys: Vec<Vec<[BufId; GKR_S_DEG]>>,
    /// Final randomness `xi_prev = [mu_last] ++ r_vec` (`total_rounds`
    /// challenges, each the `[D_EF]`-shaped BabyBear buffer
    /// [`FiatShamirTranscriptGraphIR::sample_ext`] produced).
    pub final_randomness: Vec<BufId>,
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
/// # Differences from the eager prover
/// - Both round strategies (FoldEval and PrecomputeM) are implemented, chosen per round by the same
///   env-driven [`choose_round_strategy`] the eager prover uses (env read once at driver entry).
///   The strategies produce identical transcripts (exact field arithmetic), so either graph replays
///   an eager run regardless of the eager prover's env-chosen strategy — but the emitted nodes
///   differ. The eager PrecomputeM arm's host-side `eval_mle_table` tables (eq_r_prefix/suffix and
///   eq_r_window) become on-device [`eq_mle_table_ir`] stage chains fed by the sampled challenge
///   buffers.
/// - Empty input is not supported (the eager prover returns an empty proof; an empty graph segment
///   has no meaningful `BufId`s) — asserts instead.
/// - `d_sum_evals` / `tmp_block_sums` scratch and the FoldEval work buffer are fresh `BufId`s per
///   kernel launch / per outer round rather than reused allocations. This is semantically identical
///   (the eager prover resets its scheduler each round with the data in `layer`, so work contents
///   never carry across rounds) and leaves aliasing decisions to the graph memory planner.
pub fn fractional_sumcheck_gpu_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    leaves: BufId,
    sizes: FractionalInputSize,
    alpha: EF,
    assert_zero: bool,
    device: DeviceType,
) -> Result<FracSumcheckProofIR, FractionalSumcheckError>
where
    TS: FiatShamirTranscriptGraphIR,
{
    use p3_field::PrimeCharacteristicRing;

    fn tmp_len(num_x: usize) -> usize {
        (unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) }) as usize
    }

    let real_len = sizes.real_len;
    let total_leaves = sizes.logical_len;
    assert!(
        real_len > 0,
        "fractional_sumcheck_gpu_ir requires nonempty input"
    );

    let layer = leaves;
    let layer_len = real_len;

    // ---- Segment-tree build + root extraction / observes ------------------
    let tree = build_segment_tree_ir(g, transcript, layer, sizes, alpha, assert_zero, device)?;

    let total_rounds = log2_strict_usize(total_leaves);
    let virtual_input = real_len < total_leaves;

    // ---- First claims (post-revert `layer[0]` / `layer[1]`) ----------------
    //
    // Mirrors `copy_compact_node_from_device(layer, {0, 1}, 2, ..)`: with
    // `active_size == 2` the only case that reads outside the physical
    // buffer is `real_len == total_leaves / 2`, where dense index 1 maps to
    // the all-padding right subtree `(0, virtual_padding_q(alpha, ..))` —
    // a pair of host constants under Principle 4's carve-out (they depend
    // only on `alpha` and the sizes, never on kernel outputs).
    let first_claim = if virtual_input && real_len == total_leaves / 2 {
        let p_xi_0 = add_ext_scalar_buf(g, device, "claim0_p_xi_0");
        let q_xi_0 = add_ext_scalar_buf(g, device, "claim0_q_xi_0");
        extract_root_pq_ir(g, layer, layer_len, p_xi_0, q_xi_0);
        GkrLayerClaimIR {
            p_xi_0,
            q_xi_0,
            p_xi_1: ef_const_ext_scalar_buf(g, device, "claim0_p_xi_1", EF::ZERO),
            q_xi_1: ef_const_ext_scalar_buf(
                g,
                device,
                "claim0_q_xi_1",
                virtual_padding_q(alpha, total_leaves / 2),
            ),
        }
    } else {
        extract_claim_pair_ir(g, layer, layer_len, 1, "claim0", device)
    };

    let mut claims_per_layer = Vec::with_capacity(total_rounds);
    claims_per_layer.push(first_claim);
    for buf in first_claim.as_array() {
        transcript.observe_ext(g, buf);
    }
    let mu_1 = transcript.sample_ext(g);
    let mut xi_prev: Vec<BufId> = vec![mu_1];
    let mut sumcheck_polys: Vec<Vec<[BufId; GKR_S_DEG]>> = Vec::with_capacity(total_rounds);

    // Shared read-only `EF::ONE` seed for each round's `eq_r_acc`.
    let eq_r_acc_one = ef_const_ext_scalar_buf(g, device, "eq_r_acc_one", EF::ONE);

    // FoldEval work-buffer capacity, mirroring the eager prover.
    let max_work_size = if total_rounds > 2 {
        FractionalGkrMemoryModel::fold_eval_work_buffer_elements(total_leaves)
    } else {
        0
    };

    // PrecomputeM env knobs, read once (as the eager prover does). These
    // only steer strategy/window selection — size- and env-dependent, never
    // data-dependent, so they are Principle-4-safe host values.
    let pm_env = precompute_m_enabled();
    let pm_min_blocks = precompute_m_min_blocks_threshold();
    let pm_target_blocks = precompute_m_target_blocks();
    let pm_tile_override = precompute_m_tail_tile_override();
    let pm_min_n = precompute_m_min_n();

    // ---- Outer GKR loop ----------------------------------------------------
    for round in 1..total_rounds {
        debug_assert_eq!(xi_prev.len(), round);
        // eq_buffer stores eq(xi_prev[j..], x) for x in H_{xi_prev.len()-j}
        // for j = 1, ..., xi_prev.len() - 1.
        let mut eq_buffer = SqrtEqLayersIR::from_xi(g, &xi_prev[1..], device);

        let mut round_polys: Vec<[BufId; GKR_S_DEG]> = Vec::with_capacity(round);
        let mut r_vec: Vec<BufId> = Vec::with_capacity(round);
        let mut pq_size = 2usize << round;

        let lambda = transcript.sample_ext(g);
        let last_outer_round = round == total_rounds - 1;
        let strategy = choose_round_strategy(
            round,
            pm_env,
            pm_min_blocks,
            pm_target_blocks,
            pm_tile_override,
            pm_min_n,
        );

        // In round `j`, `prev_s_eval` holds `s_{j-1}(r_{j-1})`; seeded from
        // the previous layer's claims as `numer + lambda * denom`.
        let (numer, denom) = reduce_to_single_evaluation_ir(
            g,
            *claims_per_layer.last().unwrap(),
            /* mu */ xi_prev[0],
            device,
        );
        let mut prev_s_eval = claim_combine_ir(g, numer, denom, lambda, device);
        let mut eq_r_acc = eq_r_acc_one;

        // Round 0: compute + revert fused; the fold of `r0` is fused into
        // the next round's compute.
        let d_sum = add_ef_buf(g, device, "d_sum_evals", GKR_S_DEG - 1);
        let tmp = add_ef_buf(g, device, "tmp_block_sums", tmp_len(pq_size / 2));
        let out0 = do_sumcheck_round_and_revert_ir(
            g,
            transcript,
            &mut eq_buffer,
            layer,
            layer_len,
            pq_size,
            total_leaves,
            lambda,
            alpha,
            d_sum,
            tmp,
            prev_s_eval,
            xi_prev[0],
            eq_r_acc,
            device,
        );
        round_polys.push(out0.s_evals);
        r_vec.push(out0.r);
        prev_s_eval = out0.prev_s_eval;
        eq_r_acc = out0.eq_r_acc;
        let mut prev_r = out0.r;

        // Rounds 1..round, per the env-chosen strategy. Both arms yield
        // `(active, active_len)`: the folded buffer holding this layer's
        // claims and its physical `Frac<EF>` length.
        let (active, active_len) = match strategy {
            GkrRoundStrategy::FoldEval => {
                // Work buffers are fresh `BufId`s per round (the planner handles
                // aliasing); created per strategy arm so an arm that never touches
                // one doesn't leave an unwritten buffer in the graph.
                let work_buffer = (max_work_size > 0).then(|| {
                    add_frac_ef_buf(g, device, &format!("gkr_work_{round}"), max_work_size)
                });

                // Fused rounds 1..round: compute + fold using `prev_r` (FoldEval
                // scheduling, mirroring the eager prover's `BufferScheduler` walk
                // including the virtual-compact size tracking).
                let mut scheduler = BufferScheduler::new(max_work_size);
                let mut source_real_len = real_len;
                let mut source_logical_len = total_leaves;
                for &xi_j in xi_prev.iter().skip(1) {
                    let src_pq_size = pq_size;
                    let post_fold_size = pq_size >> 1;
                    let target = scheduler.next_target(post_fold_size, last_outer_round);
                    let source_support_len = if source_logical_len == total_leaves && virtual_input
                    {
                        let subtree_len = total_leaves / src_pq_size;
                        real_len.div_ceil(subtree_len)
                    } else {
                        source_real_len
                    };
                    let compact_inplace_layer = matches!(target, BufferTarget::InPlaceLayer)
                        && source_logical_len == total_leaves
                        && virtual_input;
                    let dst_real_len = if compact_inplace_layer {
                        folded_virtual_support_len(source_support_len)
                    } else {
                        post_fold_size
                    };
                    let dst_logical_len = post_fold_size;

                    let d_sum = add_ef_buf(g, device, "d_sum_evals", GKR_S_DEG - 1);
                    let tmp = add_ef_buf(g, device, "tmp_block_sums", tmp_len(src_pq_size >> 2));
                    let out = match target {
                        BufferTarget::LayerToWork => do_fused_sumcheck_round_ir(
                            g,
                            transcript,
                            &mut eq_buffer,
                            layer,
                            work_buffer.expect("work buffer"),
                            src_pq_size,
                            source_real_len,
                            source_logical_len,
                            lambda,
                            prev_r,
                            alpha,
                            d_sum,
                            tmp,
                            prev_s_eval,
                            xi_j,
                            eq_r_acc,
                            device,
                        ),
                        BufferTarget::WorkToLayer => do_fused_sumcheck_round_ir(
                            g,
                            transcript,
                            &mut eq_buffer,
                            work_buffer.expect("work buffer"),
                            layer,
                            src_pq_size,
                            source_real_len,
                            source_logical_len,
                            lambda,
                            prev_r,
                            alpha,
                            d_sum,
                            tmp,
                            prev_s_eval,
                            xi_j,
                            eq_r_acc,
                            device,
                        ),
                        BufferTarget::InPlaceLayer => do_fused_sumcheck_round_inplace_ir(
                            g,
                            transcript,
                            &mut eq_buffer,
                            layer,
                            src_pq_size,
                            source_real_len,
                            source_logical_len,
                            dst_real_len,
                            dst_logical_len,
                            lambda,
                            prev_r,
                            alpha,
                            d_sum,
                            tmp,
                            prev_s_eval,
                            xi_j,
                            eq_r_acc,
                            device,
                        ),
                        BufferTarget::InPlaceWork => do_fused_sumcheck_round_inplace_ir(
                            g,
                            transcript,
                            &mut eq_buffer,
                            work_buffer.expect("work buffer"),
                            src_pq_size,
                            source_real_len,
                            source_logical_len,
                            dst_real_len,
                            dst_logical_len,
                            lambda,
                            prev_r,
                            alpha,
                            d_sum,
                            tmp,
                            prev_s_eval,
                            xi_j,
                            eq_r_acc,
                            device,
                        ),
                    };
                    round_polys.push(out.s_evals);
                    r_vec.push(out.r);
                    prev_s_eval = out.prev_s_eval;
                    eq_r_acc = out.eq_r_acc;
                    prev_r = out.r;
                    pq_size >>= 1;
                    source_real_len = dst_real_len;
                    source_logical_len = dst_logical_len;
                }

                // Final fold after the last `r` (no next compute to fuse with).
                let compact_virtual_final_fold = source_real_len < source_logical_len;
                match scheduler.final_fold_target(last_outer_round) {
                    target @ (BufferTarget::InPlaceWork | BufferTarget::InPlaceLayer)
                        if compact_virtual_final_fold =>
                    {
                        // Virtual-compact source: fold out of place into a dense
                        // buffer (in-place would clobber compact source slots).
                        let src = if matches!(target, BufferTarget::InPlaceWork) {
                            work_buffer.expect("work buffer")
                        } else {
                            layer
                        };
                        let output_len = pq_size / 2;
                        let dst =
                            add_frac_ef_buf(g, device, &format!("final_fold_{round}"), output_len);
                        fold_ef_frac_columns_ir_bufid(
                            g,
                            src,
                            dst,
                            pq_size,
                            source_real_len,
                            source_logical_len,
                            prev_r,
                            alpha,
                        );
                        (dst, output_len)
                    }
                    BufferTarget::InPlaceWork => {
                        let work = work_buffer.expect("work buffer");
                        fold_ef_frac_columns_inplace_ir_bufid(
                            g,
                            work,
                            pq_size,
                            source_real_len,
                            source_logical_len,
                            prev_r,
                            alpha,
                        );
                        (work, max_work_size)
                    }
                    BufferTarget::InPlaceLayer => {
                        fold_ef_frac_columns_inplace_ir_bufid(
                            g,
                            layer,
                            pq_size,
                            source_real_len,
                            source_logical_len,
                            prev_r,
                            alpha,
                        );
                        (layer, layer_len)
                    }
                    BufferTarget::LayerToWork => {
                        let work = work_buffer.expect("work buffer");
                        fold_ef_frac_columns_ir_bufid(
                            g,
                            layer,
                            work,
                            pq_size,
                            source_real_len,
                            source_logical_len,
                            prev_r,
                            alpha,
                        );
                        (work, max_work_size)
                    }
                    BufferTarget::WorkToLayer => unreachable!(),
                }
            }
            GkrRoundStrategy::PrecomputeM => {
                let stop = round.div_ceil(2);

                // First window reads from `layer` with `pending_fold=true`
                // (the M-build folds round 0's `r` inline); the multifold
                // writes `active_pq` — the work buffer for non-last
                // rounds, `layer` itself on the dense last round. Virtual
                // last-round multifolds must NOT write `layer` in place:
                // compact reads recover spilled entries from slots not
                // owned by the writing thread (see the eager arm in
                // `super::fractional`).
                let mut pending_fold = true;
                debug_assert!(max_work_size > 0, "PrecomputeM implies total_rounds > 2");
                let (active_pq, active_len) = if last_outer_round && !virtual_input {
                    (layer, layer_len)
                } else {
                    (
                        add_frac_ef_buf(g, device, &format!("gkr_work_{round}"), max_work_size),
                        max_work_size,
                    )
                };
                let mut active_real_len = 0usize;
                let mut active_logical_len = 0usize;

                let mut base = 1usize;
                while base < stop {
                    let rem_n = round - base;
                    let rounds_left = stop - base;
                    let Some(w) = choose_precompute_m_window_w(
                        rem_n,
                        rounds_left,
                        pm_min_blocks,
                        pm_target_blocks,
                        pm_tile_override,
                        pm_min_n,
                    ) else {
                        break;
                    };

                    let tail_tile = precompute_m_build_tail_tile(
                        rem_n,
                        w,
                        pm_min_blocks,
                        pm_target_blocks,
                        pm_tile_override,
                    );
                    let num_blocks = precompute_m_num_tail_blocks(rem_n, w, tail_tile);
                    let m_len = 1usize << (2 * w);
                    let partial_len = num_blocks * m_len;
                    let m_partial = add_ef_buf(g, device, "m_partial", partial_len);
                    let m_total = add_ef_buf(g, device, "m_total", m_len);

                    let (eq_tail_low, eq_tail_high, eq_low_cap) =
                        eq_tail_bufs(g, &eq_buffer, w - 1, device);

                    // Save before the eval loop overwrites `prev_r`.
                    let r_fold = prev_r;
                    let (build_src, build_real_len, build_logical_len) = if pending_fold {
                        (layer, real_len, total_leaves)
                    } else {
                        (active_pq, active_real_len, active_logical_len)
                    };
                    frac_precompute_m_build_ir_bufid(
                        g,
                        build_src,
                        eq_tail_low,
                        eq_tail_high,
                        m_partial,
                        m_total,
                        build_real_len,
                        build_logical_len,
                        rem_n,
                        w,
                        lambda,
                        r_fold,
                        alpha,
                        pending_fold, // inline fold only on the first window
                        eq_low_cap,
                        tail_tile,
                        partial_len,
                    );

                    let mut window_rs: Vec<BufId> = Vec::with_capacity(w);
                    for t in 0..w {
                        let eq_r_prefix = eq_mle_table_ir(g, &window_rs, device);
                        let eq_suffix =
                            eq_mle_table_ir(g, &xi_prev[base + t + 1..base + w], device);
                        let d_sum = add_ef_buf(g, device, "d_sum_evals", GKR_S_DEG - 1);
                        // Dense-only DSL: `m_total`, `eq_r_prefix`, `eq_suffix`,
                        // and `d_sum` are all freshly-allocated exact-sized
                        // buffers, so the DSL's byte-size type check succeeds.
                        // `frac_precompute_m_eval_round` has no virtual variant.
                        frac_precompute_m_eval_round_ir_dsl(
                            g,
                            m_total,
                            eq_r_prefix,
                            eq_suffix,
                            d_sum,
                            w,
                            t,
                        );
                        eq_buffer.drop_layer();
                        let out = observe_and_update_ir(
                            g,
                            transcript,
                            d_sum,
                            prev_s_eval,
                            xi_prev[base + t],
                            eq_r_acc,
                            device,
                        );
                        round_polys.push(out.s_evals);
                        r_vec.push(out.r);
                        prev_s_eval = out.prev_s_eval;
                        eq_r_acc = out.eq_r_acc;
                        prev_r = out.r;
                        window_rs.push(out.r);
                    }

                    // eq_r_window table for the multifold; prepend the
                    // pending round-0 challenge on the first window.
                    let (buf_vars, w_fold, eq_r_window) = if pending_fold {
                        let all_rs: Vec<BufId> = std::iter::once(r_fold).chain(window_rs).collect();
                        (rem_n + 1, w + 1, eq_mle_table_ir(g, &all_rs, device))
                    } else {
                        (rem_n, w, eq_mle_table_ir(g, &window_rs, device))
                    };
                    let (mf_real_len, mf_logical_len) = if pending_fold {
                        (real_len, total_leaves)
                    } else {
                        (active_real_len, active_logical_len)
                    };
                    if pending_fold && active_pq != layer {
                        frac_multifold_ir(
                            g,
                            layer,
                            active_pq,
                            eq_r_window,
                            mf_real_len,
                            mf_logical_len,
                            buf_vars,
                            w_fold,
                            alpha,
                        );
                    } else {
                        // Dense `src == dst` aliasing is safe (see
                        // `frac_multifold_inplace_ir`).
                        frac_multifold_inplace_ir(
                            g,
                            active_pq,
                            eq_r_window,
                            mf_real_len,
                            mf_logical_len,
                            buf_vars,
                            w_fold,
                            alpha,
                        );
                    }
                    pq_size >>= w_fold;
                    active_real_len = pq_size;
                    active_logical_len = pq_size;
                    pending_fold = false;
                    base += w;
                }
                debug_assert!(!pending_fold, "first PrecomputeM window must run");
                debug_assert_eq!(active_real_len, active_logical_len);

                if base < round {
                    // First tail round is a standalone compute; subsequent
                    // tail rounds are fused fold+compute.
                    let d_sum = add_ef_buf(g, device, "d_sum_evals", GKR_S_DEG - 1);
                    let tmp = add_ef_buf(g, device, "tmp_block_sums", tmp_len(pq_size / 2));
                    frac_compute_round_ir_bufid(
                        g,
                        &eq_buffer,
                        active_pq,
                        pq_size / 2,
                        lambda,
                        d_sum,
                        tmp,
                    );
                    eq_buffer.drop_layer();
                    let out = observe_and_update_ir(
                        g,
                        transcript,
                        d_sum,
                        prev_s_eval,
                        xi_prev[base],
                        eq_r_acc,
                        device,
                    );
                    round_polys.push(out.s_evals);
                    r_vec.push(out.r);
                    prev_s_eval = out.prev_s_eval;
                    eq_r_acc = out.eq_r_acc;
                    prev_r = out.r;

                    for &xi_j in xi_prev.iter().skip(base + 1) {
                        let src_pq_size = pq_size;
                        let d_sum = add_ef_buf(g, device, "d_sum_evals", GKR_S_DEG - 1);
                        let tmp =
                            add_ef_buf(g, device, "tmp_block_sums", tmp_len(src_pq_size >> 2));
                        let out = do_fused_sumcheck_round_inplace_ir(
                            g,
                            transcript,
                            &mut eq_buffer,
                            active_pq,
                            src_pq_size,
                            pq_size,
                            pq_size,
                            pq_size >> 1,
                            pq_size >> 1,
                            lambda,
                            prev_r,
                            alpha,
                            d_sum,
                            tmp,
                            prev_s_eval,
                            xi_j,
                            eq_r_acc,
                            device,
                        );
                        round_polys.push(out.s_evals);
                        r_vec.push(out.r);
                        prev_s_eval = out.prev_s_eval;
                        eq_r_acc = out.eq_r_acc;
                        prev_r = out.r;
                        pq_size >>= 1;
                    }
                }

                fold_ef_frac_columns_inplace_ir_bufid(
                    g, active_pq, pq_size, pq_size, pq_size, prev_r, alpha,
                );
                (active_pq, active_len)
            }
        };
        pq_size >>= 1;

        // Layer claims from the folded buffer: dense reads at physical
        // indices 0 and pq_size / 2 (the final fold always materializes a
        // dense prefix).
        let claim = extract_claim_pair_ir(
            g,
            active,
            active_len,
            pq_size / 2,
            &format!("claim{round}"),
            device,
        );
        claims_per_layer.push(claim);
        for buf in claim.as_array() {
            transcript.observe_ext(g, buf);
        }

        let mu = transcript.sample_ext(g);
        xi_prev = std::iter::once(mu).chain(r_vec).collect();
        sumcheck_polys.push(round_polys);
    }

    Ok(FracSumcheckProofIR {
        fractional_sum: (tree.root_p, tree.root_q),
        claims_per_layer,
        sumcheck_polys,
        final_randomness: xi_prev,
    })
}

// ---------------------------------------------------------------------------
// Tests.

#[cfg(test)]
mod tests {
    use std::mem::transmute;

    use crypto_compiler::{
        graph_exe::GraphCompiler,
        graph_ir::{DeviceType, GraphBuilder},
        planner::SchedulerMode,
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
        prelude::{EF, SC},
        sponge::DuplexSpongeGpu,
        sponge_graph_ir::DuplexSpongeGpuIR,
    };

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        }
    }

    #[link(name = "cudart")]
    extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
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
        leaves.to_device_on(ctx).expect("H2D copy")
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
        g.register_input(leaves_in);
        g.register_output(layer_out);

        let mut exe = GraphCompiler::new()
            .device(device)
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");

        let leaves_dev_bytes: DeviceBuffer<u8> = frac_bytes(leaves)
            .to_device_on(ctx)
            .expect("H2D input leaves");
        exe.set_input(ctx, 0, &leaves_dev_bytes).expect("set_input");
        exe.run(ctx).expect("graph run");

        let out_idx = (0..exe.num_outputs())
            .find(|&i| exe.output_buf_id(i) == layer_out)
            .expect("layer_out is a graph output");
        exe.get_output(out_idx)
            .to_host_on(ctx)
            .expect("D2H layer_out")
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

    /// Build a `SqrtEqLayersIR` over `xi` and read every low/high layer
    /// buffer back to host bytes.
    fn ir_sqrt_eq_layer_bytes(xi: &[EF], ctx: &GpuDeviceCtx) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        use crypto_compiler::graph_ir::ConstBuf;
        let device = DeviceType::Cuda(0);
        let mut g = GraphBuilder::new();

        // Stage each `xi[j]` as a `[D_EF]`-shaped BabyBear const buffer
        // (byte layout of a Rust `EF`, matching what
        // `sample_ext` would produce on-device).
        let xi_bufs: Vec<BufId> = xi
            .iter()
            .enumerate()
            .map(|(j, v)| {
                let buf = add_ext_scalar_buf(&mut g, device, &format!("xi_{j}"));
                let bytes: Vec<u8> = unsafe {
                    std::slice::from_raw_parts(v as *const EF as *const u8, size_of::<EF>())
                        .to_vec()
                };
                g.insert_const(buf, ConstBuf::HostBuf(bytes));
                buf
            })
            .collect();
        let layers = SqrtEqLayersIR::from_xi(&mut g, &xi_bufs, device);

        // Chain each layer BufId to a registered graph output via
        // `insert_memcpy` so the caller can read them after `GraphExe::run`.
        let low_outs: Vec<BufId> = (0..layers.low.layers.len())
            .map(|i| {
                let n = 1usize << i;
                let out = add_ef_buf(&mut g, device, &format!("low_out_{i}"), n);
                g.insert_memcpy(layers.low.layers[i], out);
                g.register_output(out);
                out
            })
            .collect();
        let high_outs: Vec<BufId> = (0..layers.high.layers.len())
            .map(|i| {
                let n = 1usize << i;
                let out = add_ef_buf(&mut g, device, &format!("high_out_{i}"), n);
                g.insert_memcpy(layers.high.layers[i], out);
                g.register_output(out);
                out
            })
            .collect();

        let mut exe = GraphCompiler::new()
            .device(device)
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");
        exe.run(ctx).expect("graph run");

        let collect = |bids: &[BufId]| -> Vec<Vec<u8>> {
            bids.iter()
                .map(|&bid| {
                    let idx = (0..exe.num_outputs())
                        .find(|&i| exe.output_buf_id(i) == bid)
                        .expect("layer buf in outputs");
                    exe.get_output(idx).to_host_on(ctx).expect("D2H")
                })
                .collect()
        };
        (collect(&low_outs), collect(&high_outs))
    }

    /// Same, but from the eager [`SqrtEqLayers::from_xi`] on the same `xi`.
    fn host_sqrt_eq_layer_bytes(xi: &[EF], ctx: &GpuDeviceCtx) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let layers = crate::poly::SqrtEqLayers::from_xi(xi, ctx).expect("SqrtEqLayers::from_xi");
        let per_side = |side: &crate::poly::EqEvalLayers<EF>| -> Vec<Vec<u8>> {
            side.layers
                .iter()
                .map(|arc| {
                    let host: Vec<EF> = arc.to_host_on(ctx).expect("D2H");
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            host.as_ptr() as *const u8,
                            std::mem::size_of_val(host.as_slice()),
                        )
                    };
                    bytes.to_vec()
                })
                .collect()
        };
        (per_side(&layers.low), per_side(&layers.high))
    }

    #[test]
    fn sqrt_eq_layers_ir_matches_host() {
        let ctx = test_ctx();
        let xi = [
            EF::from_u32(3),
            EF::from_u32(5),
            EF::from_u32(7),
            EF::from_u32(11),
            EF::from_u32(13),
        ];
        for len in [1usize, 2, 3, 4, 5] {
            let (ir_low, ir_high) = ir_sqrt_eq_layer_bytes(&xi[..len], &ctx);
            let (host_low, host_high) = host_sqrt_eq_layer_bytes(&xi[..len], &ctx);
            assert_eq!(ir_low, host_low, "low chain mismatch for xi.len() = {len}");
            assert_eq!(
                ir_high, host_high,
                "high chain mismatch for xi.len() = {len}"
            );
        }
    }

    /// Build a graph whose only compute is `reduce_to_single_evaluation_ir`
    /// and run it. Every input is staged as a `[D_EF]`-shaped BabyBear
    /// `insert_const` (the same shape `sample_ext` produces), and the
    /// two output buffers are memcpy'd to graph outputs before the run.
    fn ir_reduce_to_single_evaluation(
        p_xi_0: EF,
        p_xi_1: EF,
        q_xi_0: EF,
        q_xi_1: EF,
        mu: EF,
        ctx: &GpuDeviceCtx,
    ) -> (EF, EF) {
        use crypto_compiler::graph_ir::ConstBuf;
        let device = DeviceType::Cuda(0);
        let mut g = GraphBuilder::new();

        let ef_const = |g: &mut GraphBuilder, name: &str, v: EF| {
            let buf = add_ext_scalar_buf(g, device, name);
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(&v as *const EF as *const u8, size_of::<EF>()).to_vec()
            };
            g.insert_const(buf, ConstBuf::HostBuf(bytes));
            buf
        };
        let claim = GkrLayerClaimIR {
            p_xi_0: ef_const(&mut g, "p_xi_0", p_xi_0),
            p_xi_1: ef_const(&mut g, "p_xi_1", p_xi_1),
            q_xi_0: ef_const(&mut g, "q_xi_0", q_xi_0),
            q_xi_1: ef_const(&mut g, "q_xi_1", q_xi_1),
        };
        let mu_buf = ef_const(&mut g, "mu", mu);
        let (numer, denom) = reduce_to_single_evaluation_ir(&mut g, claim, mu_buf, device);

        let numer_out = add_ext_scalar_buf(&mut g, device, "numer_out");
        g.insert_memcpy(numer, numer_out);
        let denom_out = add_ext_scalar_buf(&mut g, device, "denom_out");
        g.insert_memcpy(denom, denom_out);
        g.register_output(numer_out);
        g.register_output(denom_out);

        let mut exe = GraphCompiler::new()
            .device(device)
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");
        exe.run(ctx).expect("graph run");

        let ef_from = |bid: BufId| -> EF {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == bid)
                .expect("output buf");
            let bytes: Vec<u8> = exe.get_output(idx).to_host_on(ctx).expect("D2H");
            assert_eq!(bytes.len(), size_of::<EF>());
            unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const EF) }
        };
        (ef_from(numer_out), ef_from(denom_out))
    }

    #[test]
    fn reduce_to_single_evaluation_ir_matches_host() {
        use openvm_stark_backend::poly_common::interpolate_linear_at_01;
        let ctx = test_ctx();
        let cases: [(EF, EF, EF, EF, EF); 5] = [
            (
                EF::from_u32(1),
                EF::from_u32(2),
                EF::from_u32(3),
                EF::from_u32(5),
                EF::from_u32(7),
            ),
            (
                EF::from_u32(100),
                EF::from_u32(200),
                EF::from_u32(300),
                EF::from_u32(400),
                EF::from_u32(500),
            ),
            (
                EF::ZERO,
                EF::from_u32(2013265920), // p - 1
                EF::from_u32(11),
                EF::from_u32(13),
                EF::from_u32(17),
            ),
            (
                EF::from_u32(1_234_567),
                EF::from_u32(7_654_321),
                EF::from_u32(12_345),
                EF::from_u32(54_321),
                EF::from_u32(999_983),
            ),
            (EF::ONE, EF::ZERO, EF::ZERO, EF::ONE, EF::from_u32(2)),
        ];
        for (i, (p0, p1, q0, q1, mu)) in cases.into_iter().enumerate() {
            let (got_n, got_d) = ir_reduce_to_single_evaluation(p0, p1, q0, q1, mu, &ctx);
            let want_n = interpolate_linear_at_01(&[p0, p1], mu);
            let want_d = interpolate_linear_at_01(&[q0, q1], mu);
            assert_eq!(
                got_n, want_n,
                "case {i}: numer mismatch — got {got_n:?}, want {want_n:?}"
            );
            assert_eq!(
                got_d, want_d,
                "case {i}: denom mismatch — got {got_d:?}, want {want_d:?}"
            );
        }
    }

    /// Stage an `EF` value as a 16-byte const buffer (raw p3 memory layout —
    /// Montgomery coefficients, byte-identical to the DSL / CUDA view).
    fn ef_const_buf(g: &mut GraphBuilder, name: &str, v: EF) -> BufId {
        use crypto_compiler::graph_ir::ConstBuf;
        let buf = add_ext_scalar_buf(g, DeviceType::Cuda(0), name);
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(&v as *const EF as *const u8, size_of::<EF>()).to_vec()
        };
        g.insert_const(buf, ConstBuf::HostBuf(bytes));
        buf
    }

    /// Compile a graph with no runtime inputs, run it, and read back the
    /// given buffers as raw bytes. Each buffer is registered as a graph
    /// output here, so it must have a writer (e.g. be an `insert_memcpy`
    /// destination).
    ///
    /// Uses the heuristic scheduler: the full-driver graphs are too large
    /// for CP-SAT to find an incumbent within its wall-time cap.
    fn run_graph_read_bufs(
        mut g: GraphBuilder,
        bufs: &[BufId],
        ctx: &GpuDeviceCtx,
    ) -> Vec<Vec<u8>> {
        for &b in bufs {
            g.register_output(b);
        }
        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .scheduler(SchedulerMode::Heuristic)
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");
        exe.run(ctx).expect("graph run");
        bufs.iter()
            .map(|&bid| {
                let idx = (0..exe.num_outputs())
                    .find(|&i| exe.output_buf_id(i) == bid)
                    .expect("output buf");
                exe.get_output(idx).to_host_on(ctx).expect("D2H")
            })
            .collect()
    }

    fn ef_from_bytes(bytes: &[u8]) -> EF {
        assert_eq!(bytes.len(), size_of::<EF>());
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const EF) }
    }

    /// Compile a graph with no runtime inputs, run it, and read back the
    /// given EF-scalar buffers.
    fn run_graph_read_efs(g: GraphBuilder, bufs: &[BufId], ctx: &GpuDeviceCtx) -> Vec<EF> {
        run_graph_read_bufs(g, bufs, ctx)
            .iter()
            .map(|bytes| ef_from_bytes(bytes))
            .collect()
    }

    /// Host reference for `reconstruct_s_evals` (private in
    /// `super::super::fractional`), reimplemented from the same derivation.
    fn host_reconstruct_s_evals(
        sp_vec: [EF; GKR_S_DEG - 1],
        prev_s_eval: EF,
        xi_j: EF,
        eq_r_acc: EF,
    ) -> ([EF; GKR_S_DEG], [EF; GKR_S_DEG]) {
        use openvm_stark_backend::poly_common::{eval_eq_mle, interpolate_quadratic_at_012};
        use p3_field::Field;
        let mut sp_evals = [EF::ZERO; GKR_S_DEG];
        sp_evals[1] = sp_vec[0] * eq_r_acc;
        sp_evals[2] = sp_vec[1] * eq_r_acc;
        sp_evals[0] = (prev_s_eval - xi_j * sp_evals[1]) * (EF::ONE - xi_j).inverse();
        let s_evals = std::array::from_fn(|i| {
            let x = EF::from_usize(i + 1);
            let sp_eval = if i < GKR_S_DEG - 1 {
                sp_evals[i + 1]
            } else {
                interpolate_quadratic_at_012(&sp_evals, x)
            };
            eval_eq_mle(&[xi_j], &[x]) * sp_eval
        });
        (s_evals, sp_evals)
    }

    #[test]
    fn reconstruct_s_evals_ir_matches_host() {
        use crypto_compiler::graph_ir::ConstBuf;
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_0001);
        for case in 0..3 {
            let sp_vec: [EF; GKR_S_DEG - 1] = std::array::from_fn(|_| rng.random());
            let prev_s_eval: EF = rng.random();
            let xi_j: EF = rng.random();
            let eq_r_acc: EF = rng.random();

            let mut g = GraphBuilder::new();
            let d_sum = add_ef_buf(&mut g, device, "d_sum_evals", GKR_S_DEG - 1);
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    sp_vec.as_ptr() as *const u8,
                    std::mem::size_of_val(&sp_vec),
                )
                .to_vec()
            };
            g.insert_const(d_sum, ConstBuf::HostBuf(bytes));
            let prev_buf = ef_const_buf(&mut g, "prev_s_eval", prev_s_eval);
            let xi_buf = ef_const_buf(&mut g, "xi_j", xi_j);
            let eqr_buf = ef_const_buf(&mut g, "eq_r_acc", eq_r_acc);
            let (s_bufs, sp_bufs) =
                reconstruct_s_evals_ir(&mut g, d_sum, prev_buf, xi_buf, eqr_buf, device);
            let all: Vec<BufId> = s_bufs.iter().chain(sp_bufs.iter()).copied().collect();
            let vals = run_graph_read_efs(g, &all, &ctx);
            let got_s: [EF; GKR_S_DEG] = [vals[0], vals[1], vals[2]];
            let got_sp: [EF; GKR_S_DEG] = [vals[3], vals[4], vals[5]];

            let (want_s, want_sp) = host_reconstruct_s_evals(sp_vec, prev_s_eval, xi_j, eq_r_acc);
            assert_eq!(got_s, want_s, "case {case}: s_evals mismatch");
            assert_eq!(got_sp, want_sp, "case {case}: sp_evals mismatch");
        }
    }

    #[test]
    fn update_running_scalars_ir_matches_host() {
        use openvm_stark_backend::poly_common::{eval_eq_mle, interpolate_quadratic_at_012};
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_0002);
        for case in 0..3 {
            let sp_evals: [EF; GKR_S_DEG] = std::array::from_fn(|_| rng.random());
            let xi_j: EF = rng.random();
            let eq_r_acc: EF = rng.random();
            let r: EF = rng.random();

            let mut g = GraphBuilder::new();
            let sp_bufs: [BufId; GKR_S_DEG] =
                std::array::from_fn(|i| ef_const_buf(&mut g, &format!("sp_{i}"), sp_evals[i]));
            let xi_buf = ef_const_buf(&mut g, "xi_j", xi_j);
            let eqr_buf = ef_const_buf(&mut g, "eq_r_acc", eq_r_acc);
            let r_buf = ef_const_buf(&mut g, "r", r);
            let (prev_buf, eqacc_buf) =
                update_running_scalars_ir(&mut g, sp_bufs, xi_buf, eqr_buf, r_buf, device);
            let vals = run_graph_read_efs(g, &[prev_buf, eqacc_buf], &ctx);

            let eq_r = eval_eq_mle(&[xi_j], &[r]);
            let want_prev = eq_r * interpolate_quadratic_at_012(&sp_evals, r);
            let want_eqacc = eq_r * eq_r_acc;
            assert_eq!(vals[0], want_prev, "case {case}: prev_s_eval mismatch");
            assert_eq!(vals[1], want_eqacc, "case {case}: eq_r_acc mismatch");
        }
    }

    /// The `_dev_challenge` CUDA entry points must produce bit-identical
    /// results to their host-value counterparts when the device buffer holds
    /// the same challenge. Exercises all five variants on a small dense
    /// instance: compute_round, compute_round_and_revert, fold (out-of-place
    /// and in-place), compute_round_and_fold (out-of-place and in-place).
    #[test]
    fn dev_challenge_entry_points_match_host_value() {
        use crate::cuda::logup_zerocheck::{
            _frac_compute_round_temp_buffer_size, fold_ef_frac_columns_dev_challenge,
            fold_ef_frac_columns_inplace_dev_challenge, frac_compute_round_and_fold_dev_challenge,
            frac_compute_round_and_fold_inplace_dev_challenge,
            frac_compute_round_and_revert_dev_challenge, frac_compute_round_dev_challenge,
        };

        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let mut rng = StdRng::seed_from_u64(0x5EED_0003);

        // eq layers over n coordinates serve rounds with num_x = 2^(n+1).
        let n = 5usize;
        let num_x = 2usize << n;
        let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let eq_xi = SqrtEqLayers::from_xi(&xi, &ctx).expect("SqrtEqLayers::from_xi");
        let (low_n, high_n) = (eq_xi.low_n(), eq_xi.high_n());
        assert_eq!(2 << (low_n + high_n), num_x);
        let eq_low_ptr = eq_xi.low.get_ptr(low_n);
        let eq_high_ptr = eq_xi.high.get_ptr(high_n);
        let eq_low_cap = 1usize << low_n;

        let lambda: EF = rng.random();
        let r: EF = rng.random();
        let alpha: EF = rng.random();
        let lambda_dev = [lambda].as_slice().to_device_on(&ctx).expect("H2D lambda");
        let r_dev = [r].as_slice().to_device_on(&ctx).expect("H2D r");

        let tmp_len = unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) } as usize;
        let mut tmp = DeviceBuffer::<EF>::with_capacity_on(tmp_len, &ctx);
        let mut out_a = DeviceBuffer::<EF>::with_capacity_on(2, &ctx);
        let out_b = DeviceBuffer::<EF>::with_capacity_on(2, &ctx);
        let read_outs = |a: &DeviceBuffer<EF>, b: &DeviceBuffer<EF>, ctx: &GpuDeviceCtx| {
            ctx.stream.synchronize().expect("sync");
            (
                a.to_host_on(ctx).expect("D2H out_a"),
                b.to_host_on(ctx).expect("D2H out_b"),
            )
        };

        // 1. frac_compute_round.
        let pq = make_host_leaves(2 * num_x, 0x11);
        let pq_dev = leaves_to_device(&pq, &ctx);
        unsafe {
            frac_compute_round(&eq_xi, &pq_dev, num_x, lambda, &mut out_a, &mut tmp, stream)
                .expect("frac_compute_round");
            frac_compute_round_dev_challenge(
                eq_low_ptr,
                eq_high_ptr,
                pq_dev.as_ptr(),
                num_x,
                eq_low_cap,
                lambda_dev.as_ptr(),
                out_b.as_mut_ptr(),
                tmp.as_mut_ptr(),
                stream,
            )
            .expect("frac_compute_round_dev_challenge");
        }
        let (want, got) = read_outs(&out_a, &out_b, &ctx);
        assert_eq!(want, got, "compute_round out mismatch");

        // 2. frac_compute_round_and_revert (modifies layer in place).
        let layer = make_host_leaves(2 * num_x, 0x12);
        let mut layer_a = leaves_to_device(&layer, &ctx);
        let layer_b = leaves_to_device(&layer, &ctx);
        unsafe {
            frac_compute_round_and_revert(
                &eq_xi,
                &mut layer_a,
                num_x,
                2 * num_x,
                lambda,
                alpha,
                &mut out_a,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_revert");
            frac_compute_round_and_revert_dev_challenge(
                eq_low_ptr,
                eq_high_ptr,
                layer_b.as_mut_ptr(),
                num_x,
                layer_b.len(),
                2 * num_x,
                eq_low_cap,
                lambda_dev.as_ptr(),
                alpha,
                out_b.as_mut_ptr(),
                tmp.as_mut_ptr(),
                stream,
            )
            .expect("frac_compute_round_and_revert_dev_challenge");
        }
        let (want, got) = read_outs(&out_a, &out_b, &ctx);
        assert_eq!(want, got, "compute_round_and_revert out mismatch");
        let (la, lb) = (
            layer_a.to_host_on(&ctx).expect("D2H layer_a"),
            layer_b.to_host_on(&ctx).expect("D2H layer_b"),
        );
        assert_eq!(
            frac_bytes(&la),
            frac_bytes(&lb),
            "compute_round_and_revert layer mismatch"
        );

        // 3. fold_ef_frac_columns (out-of-place, dense).
        let src = make_host_leaves(2 * num_x, 0x13);
        let src_dev = leaves_to_device(&src, &ctx);
        let mut dst_a = DeviceBuffer::<Frac<EF>>::with_capacity_on(num_x, &ctx);
        let dst_b = DeviceBuffer::<Frac<EF>>::with_capacity_on(num_x, &ctx);
        unsafe {
            fold_ef_frac_columns(
                &src_dev,
                &mut dst_a,
                2 * num_x,
                2 * num_x,
                2 * num_x,
                r,
                alpha,
                stream,
            )
            .expect("fold_ef_frac_columns");
            fold_ef_frac_columns_dev_challenge(
                src_dev.as_ptr(),
                dst_b.as_mut_ptr(),
                2 * num_x,
                2 * num_x,
                2 * num_x,
                r_dev.as_ptr(),
                alpha,
                stream,
            )
            .expect("fold_ef_frac_columns_dev_challenge");
        }
        ctx.stream.synchronize().expect("sync");
        let (da, db) = (
            dst_a.to_host_on(&ctx).expect("D2H dst_a"),
            dst_b.to_host_on(&ctx).expect("D2H dst_b"),
        );
        assert_eq!(frac_bytes(&da), frac_bytes(&db), "fold dst mismatch");

        // 4. fold_ef_frac_columns_inplace (dense).
        let mut buf_a = leaves_to_device(&src, &ctx);
        let buf_b = leaves_to_device(&src, &ctx);
        unsafe {
            fold_ef_frac_columns_inplace(
                &mut buf_a,
                2 * num_x,
                2 * num_x,
                2 * num_x,
                r,
                alpha,
                stream,
            )
            .expect("fold_ef_frac_columns_inplace");
            fold_ef_frac_columns_inplace_dev_challenge(
                buf_b.as_mut_ptr(),
                2 * num_x,
                2 * num_x,
                2 * num_x,
                r_dev.as_ptr(),
                alpha,
                stream,
            )
            .expect("fold_ef_frac_columns_inplace_dev_challenge");
        }
        ctx.stream.synchronize().expect("sync");
        let (ba, bb) = (
            buf_a.to_host_on(&ctx).expect("D2H buf_a"),
            buf_b.to_host_on(&ctx).expect("D2H buf_b"),
        );
        assert_eq!(frac_bytes(&ba), frac_bytes(&bb), "inplace fold mismatch");

        // 5. frac_compute_round_and_fold (out-of-place, dense).
        // Post-fold num_x = src_pq_size / 4 must match the eq layers.
        let src_pq_size = 4 * num_x;
        let src_pq = make_host_leaves(src_pq_size, 0x14);
        let src_pq_dev = leaves_to_device(&src_pq, &ctx);
        let mut dst_pq_a = DeviceBuffer::<Frac<EF>>::with_capacity_on(src_pq_size / 2, &ctx);
        let dst_pq_b = DeviceBuffer::<Frac<EF>>::with_capacity_on(src_pq_size / 2, &ctx);
        unsafe {
            frac_compute_round_and_fold(
                &eq_xi,
                &src_pq_dev,
                &mut dst_pq_a,
                src_pq_size,
                src_pq_size,
                src_pq_size,
                lambda,
                r,
                alpha,
                &mut out_a,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_fold");
            frac_compute_round_and_fold_dev_challenge(
                eq_low_ptr,
                eq_high_ptr,
                src_pq_dev.as_ptr(),
                dst_pq_b.as_mut_ptr(),
                src_pq_size,
                src_pq_size,
                src_pq_size,
                eq_low_cap,
                lambda_dev.as_ptr(),
                r_dev.as_ptr(),
                alpha,
                out_b.as_mut_ptr(),
                tmp.as_mut_ptr(),
                stream,
            )
            .expect("frac_compute_round_and_fold_dev_challenge");
        }
        let (want, got) = read_outs(&out_a, &out_b, &ctx);
        assert_eq!(want, got, "compute_round_and_fold out mismatch");
        let (da, db) = (
            dst_pq_a.to_host_on(&ctx).expect("D2H dst_pq_a"),
            dst_pq_b.to_host_on(&ctx).expect("D2H dst_pq_b"),
        );
        assert_eq!(
            frac_bytes(&da),
            frac_bytes(&db),
            "compute_round_and_fold dst mismatch"
        );

        // 6. frac_compute_round_and_fold_inplace (dense).
        let mut pq_a = leaves_to_device(&src_pq, &ctx);
        let pq_b = leaves_to_device(&src_pq, &ctx);
        let pq_size = src_pq_size / 2;
        unsafe {
            frac_compute_round_and_fold_inplace(
                &eq_xi,
                &mut pq_a,
                src_pq_size,
                src_pq_size,
                src_pq_size,
                pq_size,
                pq_size,
                lambda,
                r,
                alpha,
                &mut out_a,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_fold_inplace");
            frac_compute_round_and_fold_inplace_dev_challenge(
                eq_low_ptr,
                eq_high_ptr,
                pq_b.as_mut_ptr(),
                src_pq_size,
                src_pq_size,
                src_pq_size,
                pq_size,
                pq_size,
                eq_low_cap,
                lambda_dev.as_ptr(),
                r_dev.as_ptr(),
                alpha,
                out_b.as_mut_ptr(),
                tmp.as_mut_ptr(),
                stream,
            )
            .expect("frac_compute_round_and_fold_inplace_dev_challenge");
        }
        let (want, got) = read_outs(&out_a, &out_b, &ctx);
        assert_eq!(want, got, "compute_round_and_fold_inplace out mismatch");
        let (pa, pb) = (
            pq_a.to_host_on(&ctx).expect("D2H pq_a"),
            pq_b.to_host_on(&ctx).expect("D2H pq_b"),
        );
        assert_eq!(
            frac_bytes(&pa[..pq_size]),
            frac_bytes(&pb[..pq_size]),
            "compute_round_and_fold_inplace pq mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // `do_*_ir` round composites vs an eager replication of the private
    // `fractional::do_*` counterparts (eager kernel + reconstruct + eager
    // `DuplexSpongeGpu` transcript + host scalar updates).

    /// Stage `leaves` as an immutable const buffer and memcpy it into a
    /// fresh working buffer (so blackbox kernels may mutate the copy).
    fn frac_const_buf(g: &mut GraphBuilder, name: &str, leaves: &[Frac<EF>]) -> BufId {
        use crypto_compiler::graph_ir::ConstBuf;
        let device = DeviceType::Cuda(0);
        let init = add_frac_ef_buf(g, device, &format!("{name}_init"), leaves.len());
        g.insert_const(init, ConstBuf::HostBuf(frac_bytes(leaves).to_vec()));
        let buf = add_frac_ef_buf(g, device, name, leaves.len());
        g.insert_memcpy(init, buf);
        buf
    }

    /// Build a `SqrtEqLayersIR` from host `xi` values staged as `[D_EF]`
    /// BabyBear challenge consts (the shape `sample_ext` produces).
    fn ir_eq_layers(g: &mut GraphBuilder, xi: &[EF]) -> SqrtEqLayersIR {
        let xi_bufs: Vec<BufId> = xi
            .iter()
            .enumerate()
            .map(|(j, v)| ef_const_buf(g, &format!("xi_{j}"), *v))
            .collect();
        SqrtEqLayersIR::from_xi(g, &xi_bufs, DeviceType::Cuda(0))
    }

    /// Eager replication of the private `fractional::observe_and_update`
    /// tail: reconstruct s_evals from `d_sum_evals`, observe them, sample
    /// `r`, and apply the running-scalar updates. Returns `(r, s_evals)`.
    fn host_observe_and_update(
        d_sum_evals: &DeviceBuffer<EF>,
        transcript: &mut DuplexSpongeGpu,
        prev_s_eval: &mut EF,
        xi_j: EF,
        eq_r_acc: &mut EF,
        ctx: &GpuDeviceCtx,
    ) -> (EF, [EF; GKR_S_DEG]) {
        use openvm_stark_backend::{
            poly_common::{eval_eq_mle, interpolate_quadratic_at_012},
            FiatShamirTranscript,
        };
        ctx.stream.synchronize().expect("sync");
        let sp_host: Vec<EF> = d_sum_evals.to_host_on(ctx).expect("D2H d_sum_evals");
        let (s_evals, sp_evals) =
            host_reconstruct_s_evals([sp_host[0], sp_host[1]], *prev_s_eval, xi_j, *eq_r_acc);
        for &eval in &s_evals {
            FiatShamirTranscript::<SC>::observe_ext(transcript, eval);
        }
        let r = FiatShamirTranscript::<SC>::sample_ext(transcript);
        let eq_r = eval_eq_mle(&[xi_j], &[r]);
        *eq_r_acc *= eq_r;
        *prev_s_eval = eq_r * interpolate_quadratic_at_012(&sp_evals, r);
        (r, s_evals)
    }

    /// Wire the `ObserveAndUpdateOut` buffers (plus any `Frac<EF>` buffers
    /// of interest) to graph outputs, run the graph, and decode. Returns
    /// `(r, s_evals, prev_s_eval', eq_r_acc', frac buffer bytes)`.
    fn finish_round_graph(
        mut g: GraphBuilder,
        out: ObserveAndUpdateOut,
        frac_bufs: &[(BufId, usize)],
        ctx: &GpuDeviceCtx,
    ) -> (EF, [EF; GKR_S_DEG], EF, EF, Vec<Vec<u8>>) {
        let device = DeviceType::Cuda(0);
        // `r` and `s_evals` have graph readers (the update kernel and the
        // transcript), so chain them to fresh output buffers.
        let r_out = add_ext_scalar_buf(&mut g, device, "r_out");
        g.insert_memcpy(out.r, r_out);
        let s_outs: Vec<BufId> = (0..GKR_S_DEG)
            .map(|i| {
                let b = add_ext_scalar_buf(&mut g, device, &format!("s_eval_out_{i}"));
                g.insert_memcpy(out.s_evals[i], b);
                b
            })
            .collect();
        let frac_outs: Vec<BufId> = frac_bufs
            .iter()
            .enumerate()
            .map(|(i, &(buf, n))| {
                let o = add_frac_ef_buf(&mut g, device, &format!("frac_out_{i}"), n);
                g.insert_memcpy(buf, o);
                o
            })
            .collect();

        let mut bufs = vec![r_out];
        bufs.extend(&s_outs);
        bufs.push(out.prev_s_eval);
        bufs.push(out.eq_r_acc);
        bufs.extend(&frac_outs);
        let bytes = run_graph_read_bufs(g, &bufs, ctx);

        let r = ef_from_bytes(&bytes[0]);
        let s_evals = std::array::from_fn(|i| ef_from_bytes(&bytes[1 + i]));
        let prev_s_eval = ef_from_bytes(&bytes[1 + GKR_S_DEG]);
        let eq_r_acc = ef_from_bytes(&bytes[2 + GKR_S_DEG]);
        (
            r,
            s_evals,
            prev_s_eval,
            eq_r_acc,
            bytes[3 + GKR_S_DEG..].to_vec(),
        )
    }

    #[test]
    fn do_sumcheck_round_and_revert_ir_matches_eager() {
        use crate::cuda::logup_zerocheck::_frac_compute_round_temp_buffer_size;
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_0004);

        let n = 4usize;
        let num_x = 2usize << n;
        let pq_size = 2 * num_x;
        let total_leaves = pq_size;
        let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let lambda: EF = rng.random();
        let alpha: EF = rng.random();
        let xi_j: EF = rng.random();
        let prev0: EF = rng.random();
        let eqacc0: EF = rng.random();
        let layer_host = make_host_leaves(pq_size, 0x21);
        let tmp_len = unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) } as usize;

        // Eager reference.
        let eq_host = SqrtEqLayers::from_xi(&xi, &ctx).expect("SqrtEqLayers::from_xi");
        let mut layer_dev = leaves_to_device(&layer_host, &ctx);
        let mut d_sum = DeviceBuffer::<EF>::with_capacity_on(GKR_S_DEG - 1, &ctx);
        let mut tmp = DeviceBuffer::<EF>::with_capacity_on(tmp_len, &ctx);
        unsafe {
            frac_compute_round_and_revert(
                &eq_host,
                &mut layer_dev,
                num_x,
                total_leaves,
                lambda,
                alpha,
                &mut d_sum,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_revert");
        }
        let mut sponge = DuplexSpongeGpu::default();
        let (mut want_prev, mut want_eqacc) = (prev0, eqacc0);
        let (want_r, want_s) = host_observe_and_update(
            &d_sum,
            &mut sponge,
            &mut want_prev,
            xi_j,
            &mut want_eqacc,
            &ctx,
        );
        let want_layer = layer_dev.to_host_on(&ctx).expect("D2H layer");

        // Graph-IR side.
        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut eq_ir = ir_eq_layers(&mut g, &xi);
        let layer = frac_const_buf(&mut g, "layer", &layer_host);
        let lambda_buf = ef_const_buf(&mut g, "lambda", lambda);
        let prev_buf = ef_const_buf(&mut g, "prev_s_eval0", prev0);
        let xi_j_buf = ef_const_buf(&mut g, "xi_j", xi_j);
        let eqacc_buf = ef_const_buf(&mut g, "eq_r_acc0", eqacc0);
        let d_sum_ir = add_ef_buf(&mut g, device, "d_sum_evals", GKR_S_DEG - 1);
        let tmp_ir = add_ef_buf(&mut g, device, "tmp_block_sums", tmp_len);
        let out = do_sumcheck_round_and_revert_ir(
            &mut g,
            &mut transcript,
            &mut eq_ir,
            layer,
            pq_size,
            pq_size,
            total_leaves,
            lambda_buf,
            alpha,
            d_sum_ir,
            tmp_ir,
            prev_buf,
            xi_j_buf,
            eqacc_buf,
            device,
        );
        let (got_r, got_s, got_prev, got_eqacc, got_fracs) =
            finish_round_graph(g, out, &[(layer, pq_size)], &ctx);

        assert_eq!(got_r, want_r, "sampled challenge mismatch");
        assert_eq!(got_s, want_s, "s_evals mismatch");
        assert_eq!(got_prev, want_prev, "prev_s_eval mismatch");
        assert_eq!(got_eqacc, want_eqacc, "eq_r_acc mismatch");
        assert_eq!(
            got_fracs[0],
            frac_bytes(&want_layer),
            "reverted layer mismatch"
        );
    }

    #[test]
    fn do_fused_sumcheck_round_ir_matches_eager() {
        use crate::cuda::logup_zerocheck::_frac_compute_round_temp_buffer_size;
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_0005);

        // Post-fold num_x = src_pq_size / 4 must match the eq layers.
        let n = 4usize;
        let num_x = 2usize << n;
        let src_pq_size = 4 * num_x;
        let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let lambda: EF = rng.random();
        let r_prev: EF = rng.random();
        let alpha: EF = rng.random();
        let xi_j: EF = rng.random();
        let prev0: EF = rng.random();
        let eqacc0: EF = rng.random();
        let src_host = make_host_leaves(src_pq_size, 0x22);
        let tmp_len = unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) } as usize;

        // Eager reference.
        let eq_host = SqrtEqLayers::from_xi(&xi, &ctx).expect("SqrtEqLayers::from_xi");
        let src_dev = leaves_to_device(&src_host, &ctx);
        let mut dst_dev = DeviceBuffer::<Frac<EF>>::with_capacity_on(src_pq_size / 2, &ctx);
        let mut d_sum = DeviceBuffer::<EF>::with_capacity_on(GKR_S_DEG - 1, &ctx);
        let mut tmp = DeviceBuffer::<EF>::with_capacity_on(tmp_len, &ctx);
        unsafe {
            frac_compute_round_and_fold(
                &eq_host,
                &src_dev,
                &mut dst_dev,
                src_pq_size,
                src_pq_size,
                src_pq_size,
                lambda,
                r_prev,
                alpha,
                &mut d_sum,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_fold");
        }
        let mut sponge = DuplexSpongeGpu::default();
        let (mut want_prev, mut want_eqacc) = (prev0, eqacc0);
        let (want_r, want_s) = host_observe_and_update(
            &d_sum,
            &mut sponge,
            &mut want_prev,
            xi_j,
            &mut want_eqacc,
            &ctx,
        );
        let want_dst = dst_dev.to_host_on(&ctx).expect("D2H dst");

        // Graph-IR side.
        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut eq_ir = ir_eq_layers(&mut g, &xi);
        let src = frac_const_buf(&mut g, "src_pq", &src_host);
        let dst = add_frac_ef_buf(&mut g, device, "dst_pq", src_pq_size / 2);
        let lambda_buf = ef_const_buf(&mut g, "lambda", lambda);
        let r_prev_buf = ef_const_buf(&mut g, "r_prev", r_prev);
        let prev_buf = ef_const_buf(&mut g, "prev_s_eval0", prev0);
        let xi_j_buf = ef_const_buf(&mut g, "xi_j", xi_j);
        let eqacc_buf = ef_const_buf(&mut g, "eq_r_acc0", eqacc0);
        let d_sum_ir = add_ef_buf(&mut g, device, "d_sum_evals", GKR_S_DEG - 1);
        let tmp_ir = add_ef_buf(&mut g, device, "tmp_block_sums", tmp_len);
        let out = do_fused_sumcheck_round_ir(
            &mut g,
            &mut transcript,
            &mut eq_ir,
            src,
            dst,
            src_pq_size,
            src_pq_size,
            src_pq_size,
            lambda_buf,
            r_prev_buf,
            alpha,
            d_sum_ir,
            tmp_ir,
            prev_buf,
            xi_j_buf,
            eqacc_buf,
            device,
        );
        let (got_r, got_s, got_prev, got_eqacc, got_fracs) =
            finish_round_graph(g, out, &[(dst, src_pq_size / 2)], &ctx);

        assert_eq!(got_r, want_r, "sampled challenge mismatch");
        assert_eq!(got_s, want_s, "s_evals mismatch");
        assert_eq!(got_prev, want_prev, "prev_s_eval mismatch");
        assert_eq!(got_eqacc, want_eqacc, "eq_r_acc mismatch");
        assert_eq!(got_fracs[0], frac_bytes(&want_dst), "folded dst mismatch");
    }

    #[test]
    fn do_fused_sumcheck_round_inplace_ir_matches_eager() {
        use crate::cuda::logup_zerocheck::_frac_compute_round_temp_buffer_size;
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_0006);

        let n = 4usize;
        let num_x = 2usize << n;
        let src_pq_size = 4 * num_x;
        let dst_pq_size = src_pq_size / 2;
        let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let lambda: EF = rng.random();
        let r_prev: EF = rng.random();
        let alpha: EF = rng.random();
        let xi_j: EF = rng.random();
        let prev0: EF = rng.random();
        let eqacc0: EF = rng.random();
        let pq_host = make_host_leaves(src_pq_size, 0x23);
        let tmp_len = unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) } as usize;

        // Eager reference.
        let eq_host = SqrtEqLayers::from_xi(&xi, &ctx).expect("SqrtEqLayers::from_xi");
        let mut pq_dev = leaves_to_device(&pq_host, &ctx);
        let mut d_sum = DeviceBuffer::<EF>::with_capacity_on(GKR_S_DEG - 1, &ctx);
        let mut tmp = DeviceBuffer::<EF>::with_capacity_on(tmp_len, &ctx);
        unsafe {
            frac_compute_round_and_fold_inplace(
                &eq_host,
                &mut pq_dev,
                src_pq_size,
                src_pq_size,
                src_pq_size,
                dst_pq_size,
                dst_pq_size,
                lambda,
                r_prev,
                alpha,
                &mut d_sum,
                &mut tmp,
                stream,
            )
            .expect("frac_compute_round_and_fold_inplace");
        }
        let mut sponge = DuplexSpongeGpu::default();
        let (mut want_prev, mut want_eqacc) = (prev0, eqacc0);
        let (want_r, want_s) = host_observe_and_update(
            &d_sum,
            &mut sponge,
            &mut want_prev,
            xi_j,
            &mut want_eqacc,
            &ctx,
        );
        let want_pq = pq_dev.to_host_on(&ctx).expect("D2H pq");

        // Graph-IR side.
        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut eq_ir = ir_eq_layers(&mut g, &xi);
        let pq = frac_const_buf(&mut g, "pq", &pq_host);
        let lambda_buf = ef_const_buf(&mut g, "lambda", lambda);
        let r_prev_buf = ef_const_buf(&mut g, "r_prev", r_prev);
        let prev_buf = ef_const_buf(&mut g, "prev_s_eval0", prev0);
        let xi_j_buf = ef_const_buf(&mut g, "xi_j", xi_j);
        let eqacc_buf = ef_const_buf(&mut g, "eq_r_acc0", eqacc0);
        let d_sum_ir = add_ef_buf(&mut g, device, "d_sum_evals", GKR_S_DEG - 1);
        let tmp_ir = add_ef_buf(&mut g, device, "tmp_block_sums", tmp_len);
        let out = do_fused_sumcheck_round_inplace_ir(
            &mut g,
            &mut transcript,
            &mut eq_ir,
            pq,
            src_pq_size,
            src_pq_size,
            src_pq_size,
            dst_pq_size,
            dst_pq_size,
            lambda_buf,
            r_prev_buf,
            alpha,
            d_sum_ir,
            tmp_ir,
            prev_buf,
            xi_j_buf,
            eqacc_buf,
            device,
        );
        let (got_r, got_s, got_prev, got_eqacc, got_fracs) =
            finish_round_graph(g, out, &[(pq, src_pq_size)], &ctx);

        assert_eq!(got_r, want_r, "sampled challenge mismatch");
        assert_eq!(got_s, want_s, "s_evals mismatch");
        assert_eq!(got_prev, want_prev, "prev_s_eval mismatch");
        assert_eq!(got_eqacc, want_eqacc, "eq_r_acc mismatch");
        // Only the folded prefix (dst_pq_size elements) is meaningful.
        let prefix = dst_pq_size * FRAC_EF_BYTES;
        assert_eq!(
            &got_fracs[0][..prefix],
            &frac_bytes(&want_pq)[..prefix],
            "in-place folded pq mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // End-to-end: `fractional_sumcheck_gpu_ir` vs the eager
    // `fractional_sumcheck_gpu`, same leaves, fresh transcripts.

    /// Run the graph-IR driver on `leaves` and read every proof artifact
    /// back as host `EF` values, reshaped to match the eager proof.
    #[allow(clippy::type_complexity)]
    fn run_ir_sumcheck(
        leaves: &[Frac<EF>],
        sizes: FractionalInputSize,
        alpha: EF,
        ctx: &GpuDeviceCtx,
    ) -> ((EF, EF), Vec<[EF; 4]>, Vec<Vec<[EF; GKR_S_DEG]>>, Vec<EF>) {
        let device = DeviceType::Cuda(0);
        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let layer = frac_const_buf(&mut g, "leaves", leaves);
        let proof =
            fractional_sumcheck_gpu_ir(&mut g, &mut transcript, layer, sizes, alpha, false, device)
                .expect("fractional_sumcheck_gpu_ir");

        // Every artifact has in-graph readers (transcript observes / later
        // kernels), so memcpy each into a fresh buffer that
        // `run_graph_read_efs` registers as a graph output.
        let mut exports: Vec<BufId> = Vec::new();
        let export = |g: &mut GraphBuilder, src: BufId, name: String| {
            let out = add_ext_scalar_buf(g, device, &name);
            g.insert_memcpy(src, out);
            out
        };
        let (root_p, root_q) = proof.fractional_sum;
        exports.push(export(&mut g, root_p, "out_root_p".to_string()));
        exports.push(export(&mut g, root_q, "out_root_q".to_string()));
        for (i, claim) in proof.claims_per_layer.iter().enumerate() {
            for (k, buf) in claim.as_array().into_iter().enumerate() {
                exports.push(export(&mut g, buf, format!("out_claim_{i}_{k}")));
            }
        }
        for (i, layer_polys) in proof.sumcheck_polys.iter().enumerate() {
            for (j, s) in layer_polys.iter().enumerate() {
                for (k, &buf) in s.iter().enumerate() {
                    exports.push(export(&mut g, buf, format!("out_s_{i}_{j}_{k}")));
                }
            }
        }
        for (i, &buf) in proof.final_randomness.iter().enumerate() {
            exports.push(export(&mut g, buf, format!("out_xi_{i}")));
        }

        let efs = run_graph_read_efs(g, &exports, ctx);
        let mut it = efs.into_iter();
        let fractional_sum = (it.next().unwrap(), it.next().unwrap());
        let claims: Vec<[EF; 4]> = (0..proof.claims_per_layer.len())
            .map(|_| std::array::from_fn(|_| it.next().unwrap()))
            .collect();
        let polys: Vec<Vec<[EF; GKR_S_DEG]>> = proof
            .sumcheck_polys
            .iter()
            .map(|layer_polys| {
                layer_polys
                    .iter()
                    .map(|_| std::array::from_fn(|_| it.next().unwrap()))
                    .collect()
            })
            .collect();
        let final_randomness: Vec<EF> = proof
            .final_randomness
            .iter()
            .map(|_| it.next().unwrap())
            .collect();
        assert!(it.next().is_none(), "leftover exported values");
        (fractional_sum, claims, polys, final_randomness)
    }

    fn assert_e2e_matches_eager(real_len: usize, logical_len: usize, seed: u64) {
        use openvm_cuda_common::memory_manager::MemTracker;

        use super::super::fractional::fractional_sumcheck_gpu;

        let ctx = test_ctx();
        let sizes = FractionalInputSize::new(real_len, logical_len);
        let mut rng = StdRng::seed_from_u64(seed);
        let alpha: EF = rng.random();
        let leaves = make_host_leaves(real_len, seed ^ 0xA5A5);

        // Eager reference. The round strategy is env-chosen (FoldEval vs
        // PrecomputeM) on both the eager and graph sides, but both
        // strategies produce identical transcripts, so the comparison holds
        // for any combination. The `_precompute_m_*` variants below force
        // the PrecomputeM arm via env and separately assert the graph
        // actually contains its build nodes.
        let mut sponge = DuplexSpongeGpu::default();
        let mut mem = MemTracker::start("test.fractional_ir_e2e");
        let (want_proof, want_xi) = fractional_sumcheck_gpu::<SC, _>(
            &mut sponge,
            leaves_to_device(&leaves, &ctx),
            sizes,
            alpha,
            false,
            &mut mem,
            &ctx,
        )
        .expect("eager fractional_sumcheck_gpu");
        ctx.stream.synchronize().expect("sync");

        // Graph-IR side.
        let (got_sum, got_claims, got_polys, got_xi) = run_ir_sumcheck(&leaves, sizes, alpha, &ctx);

        assert_eq!(
            got_sum, want_proof.fractional_sum,
            "fractional_sum mismatch"
        );
        assert_eq!(
            got_claims.len(),
            want_proof.claims_per_layer.len(),
            "claims_per_layer length mismatch"
        );
        for (i, (got, want)) in got_claims
            .iter()
            .zip(&want_proof.claims_per_layer)
            .enumerate()
        {
            assert_eq!(
                *got,
                [want.p_xi_0, want.q_xi_0, want.p_xi_1, want.q_xi_1],
                "layer {i} claims mismatch"
            );
        }
        assert_eq!(
            got_polys, want_proof.sumcheck_polys,
            "sumcheck_polys mismatch"
        );
        assert_eq!(got_xi, want_xi, "final randomness mismatch");
    }

    #[test]
    fn fractional_sumcheck_gpu_ir_matches_eager_dense() {
        assert_e2e_matches_eager(16, 16, 0x5EED_0007);
    }

    #[test]
    fn fractional_sumcheck_gpu_ir_matches_eager_virtual() {
        assert_e2e_matches_eager(12, 16, 0x5EED_0008);
    }

    #[test]
    fn fractional_sumcheck_gpu_ir_matches_eager_virtual_half_edge() {
        // `real_len == logical_len / 2`: the first-claims right node is the
        // all-padding subtree `(0, virtual_padding_q(alpha, ..))`.
        assert_e2e_matches_eager(8, 16, 0x5EED_0009);
    }

    #[test]
    fn eq_mle_table_ir_matches_host() {
        use super::super::fractional::eval_mle_table;
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_0011);
        let points: Vec<EF> = (0..4).map(|_| rng.random()).collect();
        for n in 0..=points.len() {
            let mut g = GraphBuilder::new();
            let point_bufs: Vec<BufId> = points[..n]
                .iter()
                .enumerate()
                .map(|(i, &p)| ef_const_ext_scalar_buf(&mut g, device, &format!("x{i}"), p))
                .collect();
            let table = eq_mle_table_ir(&mut g, &point_bufs, device);
            let out = add_ef_buf(&mut g, device, "out_table", 1 << n);
            g.insert_memcpy(table, out);
            let bytes = run_graph_read_bufs(g, &[out], &ctx).remove(0);
            let got: Vec<EF> = bytes
                .chunks_exact(size_of::<EF>())
                .map(ef_from_bytes)
                .collect();
            let mut want = vec![EF::ZERO; 1 << n];
            eval_mle_table(&points[..n], &mut want);
            assert_eq!(got, want, "eq_mle_table mismatch for n = {n}");
        }
    }

    /// PrecomputeM e2e: force the strategy on with lowered thresholds
    /// (mirroring the eager `run_from_host` env pattern), assert the graph
    /// actually contains `want_build_nodes` M-build kernels (so the test
    /// cannot silently pass on the FoldEval arm), then compare against the
    /// eager prover end to end.
    fn assert_e2e_matches_eager_precompute_m(
        real_len: usize,
        logical_len: usize,
        seed: u64,
        want_build_nodes: usize,
    ) {
        // SAFETY: nextest runs each test binary invocation in its own
        // process, so setting the process env only affects this test.
        unsafe {
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M", "1");
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS", "1");
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N", "4");
        }

        // Host-side graph build only: count the PrecomputeM M-build nodes.
        {
            use crypto_compiler::graph_ir::GraphNode;
            let device = DeviceType::Cuda(0);
            let mut g = GraphBuilder::new();
            let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
            let leaves = make_host_leaves(real_len, seed ^ 0xA5A5);
            let layer = frac_const_buf(&mut g, "leaves", &leaves);
            fractional_sumcheck_gpu_ir(
                &mut g,
                &mut transcript,
                layer,
                FractionalInputSize::new(real_len, logical_len),
                EF::ZERO,
                false,
                device,
            )
            .expect("fractional_sumcheck_gpu_ir");
            let n_build = g
                .nodes
                .iter()
                .filter(|n| {
                    matches!(n, GraphNode::BlackboxKernel(k)
                        if k.name == "frac_precompute_m_build_dev_challenge")
                })
                .count();
            assert_eq!(
                n_build, want_build_nodes,
                "PrecomputeM M-build node count mismatch"
            );
        }

        assert_e2e_matches_eager(real_len, logical_len, seed);

        unsafe {
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M");
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS");
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N");
        }
    }

    #[test]
    fn fractional_sumcheck_gpu_ir_matches_eager_precompute_m_dense() {
        // total_rounds = 8: round 7 (the last, dense ⇒ in-place multifold
        // on `layer`) runs one PrecomputeM window (base=1, w=3) plus a
        // standalone + 2 fused tail rounds.
        assert_e2e_matches_eager_precompute_m(256, 256, 0x5EED_000A, 1);
    }

    #[test]
    fn fractional_sumcheck_gpu_ir_matches_eager_precompute_m_virtual() {
        // Virtual input ⇒ `active_pq` is the work buffer, so the first
        // window's pending fold takes the out-of-place `frac_multifold_ir`
        // path reading the compact `layer`.
        assert_e2e_matches_eager_precompute_m(192, 256, 0x5EED_000B, 1);
    }

    #[test]
    fn fractional_sumcheck_gpu_ir_matches_eager_precompute_m_multi_window() {
        // total_rounds = 14: rounds 7..=12 each run one window; round 13
        // (stop = 7) runs two (base=1 and base=4), exercising the
        // `pending_fold = false` second-window M-build + in-place multifold
        // on the work buffer.
        assert_e2e_matches_eager_precompute_m(1 << 14, 1 << 14, 0x5EED_000C, 8);
    }

    /// Artifact generator, not a correctness test: builds the composed
    /// fractional-sumcheck graph (PrecomputeM forced on via env, so both
    /// round strategies appear), writes the `GraphBuilder` SSA dump,
    /// compiles it, and writes the planner-ordered `GraphExe` dump plus
    /// per-structured-kernel `.hir` / `.cu` dumps.
    ///
    /// Input size via `FRAC_DUMP_LOG_N` (log2 leaf count, default 8);
    /// output directory via `CRYPTO_COMPILER_DUMP_IR` (default
    /// `target/ir_dump/`).
    ///
    /// Run explicitly:
    ///     cargo nextest run -p openvm-cuda-backend --features graph-ir \
    ///         --run-ignored all -E 'test(dump_fractional_sumcheck_ir_graph)'
    #[test]
    #[ignore = "artifact generator; run explicitly to produce IR dumps"]
    fn dump_fractional_sumcheck_ir_graph() {
        let _ctx = test_ctx();
        // SAFETY: nextest runs each test binary invocation in its own
        // process, so setting the process env only affects this test.
        unsafe {
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M", "1");
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS", "1");
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N", "4");
        }

        let log_n: usize = std::env::var("FRAC_DUMP_LOG_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let real_len = 1usize << log_n;
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_00D0);
        let alpha: EF = rng.random();
        let leaves = make_host_leaves(real_len, 0x5EED_00D1);

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let layer = frac_const_buf(&mut g, "leaves", &leaves);
        let proof = fractional_sumcheck_gpu_ir(
            &mut g,
            &mut transcript,
            layer,
            FractionalInputSize::new(real_len, real_len),
            alpha,
            false,
            device,
        )
        .expect("fractional_sumcheck_gpu_ir");
        // Give the graph registered outputs; every proof artifact already
        // has in-graph readers, so export the root sum via memcpy. Without
        // registered outputs DCE would delete the whole graph.
        let (root_p, root_q) = proof.fractional_sum;
        for (src, name) in [(root_p, "out_root_p"), (root_q, "out_root_q")] {
            let out = add_ext_scalar_buf(&mut g, device, name);
            g.insert_memcpy(src, out);
            g.register_output(out);
        }

        let dir = std::env::var_os("CRYPTO_COMPILER_DUMP_IR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/ir_dump")
            });
        std::fs::create_dir_all(&dir).expect("create dump dir");
        std::fs::write(
            dir.join(format!("fractional_sumcheck_n{real_len}.graph.txt")),
            g.print(),
        )
        .expect("write graph dump");

        let exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::Heuristic)
            .compile_options(CompileOptions {
                dump_ir: Some(dir.clone()),
                ..CompileOptions::default()
            })
            .compile(g)
            .expect("graph compile");
        std::fs::write(
            dir.join(format!("fractional_sumcheck_n{real_len}.exe.txt")),
            exe.print(),
        )
        .expect("write exe dump");

        unsafe {
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M");
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS");
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N");
        }
        println!("IR dumps written to {}", dir.display());
    }

    /// Artifact generator, not a correctness test: builds the composed
    /// fractional-sumcheck graph (PrecomputeM forced on via env) and writes
    /// a Cytoscape.js elements-JSON dump for browser visualization. No
    /// compilation happens, so this runs in seconds at any size.
    ///
    /// Input size via `FRAC_DUMP_LOG_N` (log2 leaf count, default 8);
    /// output directory via `CRYPTO_COMPILER_DUMP_IR` (default
    /// `target/ir_dump/`). View with:
    ///     python3 scripts/serve_graph.py \
    ///         target/ir_dump/fractional_sumcheck_n256.cy.json
    #[test]
    #[ignore = "artifact generator; run explicitly to produce the Cytoscape dump"]
    fn dump_fractional_sumcheck_cytoscape() {
        let _ctx = test_ctx();
        // SAFETY: nextest runs each test binary invocation in its own
        // process, so setting the process env only affects this test.
        unsafe {
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M", "1");
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS", "1");
            std::env::set_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N", "4");
        }

        let log_n: usize = std::env::var("FRAC_DUMP_LOG_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let real_len = 1usize << log_n;
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_00D0);
        let alpha: EF = rng.random();
        let leaves = make_host_leaves(real_len, 0x5EED_00D1);

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let layer = frac_const_buf(&mut g, "leaves", &leaves);
        let proof = fractional_sumcheck_gpu_ir(
            &mut g,
            &mut transcript,
            layer,
            FractionalInputSize::new(real_len, real_len),
            alpha,
            false,
            device,
        )
        .expect("fractional_sumcheck_gpu_ir");
        let (root_p, root_q) = proof.fractional_sum;
        for (src, name) in [(root_p, "out_root_p"), (root_q, "out_root_q")] {
            let out = add_ext_scalar_buf(&mut g, device, name);
            g.insert_memcpy(src, out);
        }

        let dir = std::env::var_os("CRYPTO_COMPILER_DUMP_IR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/ir_dump")
            });
        std::fs::create_dir_all(&dir).expect("create dump dir");
        let path = dir.join(format!("fractional_sumcheck_n{real_len}.cy.json"));
        std::fs::write(&path, g.to_cytoscape_json()).expect("write cytoscape dump");

        unsafe {
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M");
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS");
            std::env::remove_var("SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N");
        }
        println!("Cytoscape dump written to {}", path.display());
    }

    /// Benchmark, not a correctness test: eager `fractional_sumcheck_gpu`
    /// vs the graph-IR pipeline, reporting per size the eager wall time
    /// and the graph pipeline split into build (host), compile (planner +
    /// JIT), and execution. Both sides start from device-resident leaves:
    /// the eager prover consumes its input buffer directly, while the
    /// graph copies its registered input D2D into a scratch working layer
    /// (the driver folds the layer in place, and registered graph inputs
    /// must not have writers).
    ///
    /// The round strategy follows the ambient `SWIRL_CUDA_GKR_PRECOMPUTE_M*`
    /// env on both sides. Sizes via `FRAC_BENCH_LOG_N` (comma-separated
    /// log2 leaf counts, default `16,20`).
    ///
    /// Run explicitly:
    ///     cargo nextest run -p openvm-cuda-backend --features graph-ir \
    ///         --run-ignored all --no-capture \
    ///         -E 'test(bench_fractional_sumcheck_eager_vs_ir)'
    ///
    /// nsys profile (env `NSYS_ENABLED=1`): a *single* cudaProfilerStart/Stop
    /// window wraps only the timed eager + graph iterations across every
    /// size. Warmup and graph build/compile run in a separate setup pass
    /// beforehand, so the profile contains only measured kernel work.
    /// NVTX ranges `eager n=2^{k}` and `graph n=2^{k}` label each phase:
    ///     NSYS_ENABLED=1 nsys profile --capture-range=cudaProfilerApi \
    ///         --trace=cuda,nvtx -o frac_bench \
    ///         cargo nextest run -p openvm-cuda-backend --features graph-ir \
    ///             --run-ignored all --no-capture \
    ///             -E 'test(bench_fractional_sumcheck_eager_vs_ir)'
    #[test]
    #[ignore = "benchmark; run explicitly with --run-ignored"]
    fn bench_fractional_sumcheck_eager_vs_ir() {
        use std::time::Instant;

        use crypto_compiler::graph_exe::GraphExe;
        use openvm_cuda_common::memory_manager::MemTracker;

        use super::super::fractional::fractional_sumcheck_gpu;

        const ITERS: usize = 3;

        struct PerSize {
            log_n: usize,
            n: usize,
            sizes: FractionalInputSize,
            leaves: Vec<Frac<EF>>,
            alpha: EF,
            eager_sum: (EF, EF),
            exe: GraphExe,
            exports: Vec<BufId>,
            build_ms: f64,
            compile_ms: f64,
            n_nodes: usize,
            eager_ms: Vec<f64>,
            graph_ms: Vec<f64>,
        }

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let log_ns: Vec<usize> = std::env::var("FRAC_BENCH_LOG_N")
            .unwrap_or_else(|_| "16,20".into())
            .split(',')
            .map(|s| s.trim().parse().expect("FRAC_BENCH_LOG_N entry"))
            .collect();

        let nsys_enabled = std::env::var_os("NSYS_ENABLED").is_some();

        // ---- Setup pass: build/compile the graph, prime both eager and
        // graph kernels. All runs here happen *before* cudaProfilerStart,
        // so first-launch driver init and JIT cost never enter the profile.
        let mut states: Vec<PerSize> = Vec::with_capacity(log_ns.len());
        for log_n in log_ns {
            let n = 1usize << log_n;
            let sizes = FractionalInputSize::new(n, n);
            let leaves = make_host_leaves(n, 0x5EED_BE9C ^ log_n as u64);
            let mut rng = StdRng::seed_from_u64(0xA1FA ^ log_n as u64);
            let alpha: EF = rng.random();

            println!("\n=== fractional sumcheck: n = 2^{log_n} = {n} leaves ===");

            // Eager warmup — also records `eager_sum` for the sanity check.
            let eager_sum = {
                let d_leaves = leaves_to_device(&leaves, &ctx);
                let mut sponge = DuplexSpongeGpu::default();
                let mut mem = MemTracker::start("bench.fractional_eager");
                ctx.stream.synchronize().expect("sync");
                let (proof, _xi) = fractional_sumcheck_gpu::<SC, _>(
                    &mut sponge,
                    d_leaves,
                    sizes,
                    alpha,
                    false,
                    &mut mem,
                    &ctx,
                )
                .expect("eager warmup");
                ctx.stream.synchronize().expect("sync");
                proof.fractional_sum
            };

            // Graph build + compile — never inside an NVTX range or profile.
            let t0 = Instant::now();
            let mut g = GraphBuilder::new();
            let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
            let input = add_frac_ef_buf(&mut g, device, "leaves_in", n);
            let layer = add_frac_ef_buf(&mut g, device, "leaves_work", n);
            g.insert_memcpy(input, layer);
            g.register_input(input);
            let proof_ir = fractional_sumcheck_gpu_ir(
                &mut g,
                &mut transcript,
                layer,
                sizes,
                alpha,
                false,
                device,
            )
            .expect("fractional_sumcheck_gpu_ir");
            // Export the root sum so the graph has outputs and the result
            // can be checked against the eager proof.
            let (root_p, root_q) = proof_ir.fractional_sum;
            let exports: Vec<BufId> = [(root_p, "out_root_p"), (root_q, "out_root_q")]
                .into_iter()
                .map(|(src, name)| {
                    let out = add_ext_scalar_buf(&mut g, device, name);
                    g.insert_memcpy(src, out);
                    g.register_output(out);
                    out
                })
                .collect();
            let build_ms = t0.elapsed().as_secs_f64() * 1e3;
            let n_nodes = g.nodes.len();

            let t0 = Instant::now();
            let mut exe = GraphCompiler::new()
                .device(device)
                .scheduler(SchedulerMode::Heuristic)
                .compile_options(CompileOptions::default())
                .compile(g)
                .expect("graph compile");
            let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "graph build: {build_ms:>8.2} ms ({n_nodes} nodes); compile: {compile_ms:>8.2} \
                 ms ({} unique modules, {} loaded from cache, scratch pool {} bytes)",
                exe.num_unique_modules(),
                exe.num_cached_modules(),
                exe.scratch_bytes(),
            );

            assert_eq!(exe.num_inputs(), 1, "leaves should be the only input");
            let d_input = frac_bytes(&leaves).to_device_on(&ctx).expect("H2D");
            exe.set_input(&ctx, 0, &d_input).expect("set_input");

            // Graph exec warmup.
            ctx.stream.synchronize().expect("sync");
            exe.run(&ctx).expect("graph warmup");
            ctx.stream.synchronize().expect("sync");

            states.push(PerSize {
                log_n,
                n,
                sizes,
                leaves,
                alpha,
                eager_sum,
                exe,
                exports,
                build_ms,
                compile_ms,
                n_nodes,
                eager_ms: Vec::with_capacity(ITERS),
                graph_ms: Vec::with_capacity(ITERS),
            });
        }

        // ---- Timed pass: everything below runs inside a single
        // cudaProfilerStart/Stop window so nsys emits one .nsys-rep file
        // containing only the labeled timed work.
        if nsys_enabled {
            unsafe { cudaProfilerStart() };
        }
        for st in states.iter_mut() {
            if nsys_enabled {
                nvtx::range_push!("eager n=2^{}", st.log_n);
            }
            for _ in 0..ITERS {
                let d_leaves = leaves_to_device(&st.leaves, &ctx);
                let mut sponge = DuplexSpongeGpu::default();
                let mut mem = MemTracker::start("bench.fractional_eager");
                ctx.stream.synchronize().expect("sync");
                let t0 = Instant::now();
                let (proof, _xi) = fractional_sumcheck_gpu::<SC, _>(
                    &mut sponge,
                    d_leaves,
                    st.sizes,
                    st.alpha,
                    false,
                    &mut mem,
                    &ctx,
                )
                .expect("eager fractional_sumcheck_gpu");
                ctx.stream.synchronize().expect("sync");
                st.eager_ms.push(t0.elapsed().as_secs_f64() * 1e3);
                st.eager_sum = proof.fractional_sum;
            }
            if nsys_enabled {
                nvtx::range_pop!();
                nvtx::range_push!("graph n=2^{}", st.log_n);
            }
            for _ in 0..ITERS {
                ctx.stream.synchronize().expect("sync");
                let t0 = Instant::now();
                st.exe.run(&ctx).expect("graph run");
                ctx.stream.synchronize().expect("sync");
                st.graph_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            if nsys_enabled {
                nvtx::range_pop!();
            }
        }
        if nsys_enabled {
            unsafe { cudaProfilerStop() };
        }

        // ---- Report + sanity check.
        for st in &states {
            let eager_mean = st.eager_ms.iter().sum::<f64>() / ITERS as f64;
            let graph_mean = st.graph_ms.iter().sum::<f64>() / ITERS as f64;
            println!(
                "\n--- fractional sumcheck: n = 2^{} = {} leaves ---\n\
                 eager      : {:>8.2?} ms (mean {:.2} ms)\n\
                 graph build: {:>8.2} ms ({} nodes); compile: {:>8.2} ms\n\
                 graph exec : {:>8.2?} ms (mean {:.2} ms, {:.3}x eager)",
                st.log_n,
                st.n,
                st.eager_ms,
                eager_mean,
                st.build_ms,
                st.n_nodes,
                st.compile_ms,
                st.graph_ms,
                graph_mean,
                graph_mean / eager_mean,
            );

            let read_export = |bid: BufId| -> EF {
                let idx = (0..st.exe.num_outputs())
                    .find(|&i| st.exe.output_buf_id(i) == bid)
                    .expect("export output index");
                ef_from_bytes(&st.exe.get_output(idx).to_host_on(&ctx).expect("D2H"))
            };
            let got_sum = (read_export(st.exports[0]), read_export(st.exports[1]));
            assert_eq!(
                got_sum, st.eager_sum,
                "fractional_sum mismatch at 2^{}",
                st.log_n
            );
        }
    }
}
