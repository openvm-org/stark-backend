//! Structured `ir::Module` ports of the fractional-GKR CUDA kernels.
//!
//! Each `*_ir_dsl` function inserts a [`GraphNode::Kernel`] whose body is a
//! DSL `ir::Module` (via [`GraphBuilder::insert_kernel`]) instead of a
//! blackbox launch of the underlying `_frac_*` / `fold_ef_*` CUDA kernel.
//! The functional/pure DSL forces in-place mutation to be replaced by
//! producing a fresh output buffer; the graph memory planner is free to
//! alias later.
//!
//! # Dense-only
//!
//! Every module here **assumes `real_len == logical_len`** (the dense case).
//! The eager blackbox kernels support a virtual/compact mode
//! (`virtual_node_value` addressing plus a runtime `real_len < logical_len`
//! guard reading through a bit-reversed index) that the DSL's quasi-affine
//! index-expression checker cannot express. When the CUDA body branches on
//! `virtual_mode`, only the `else` (dense) branch is ported.
//!
//! # Alpha as a compile-time host EF constant
//!
//! `alpha` remains a Rust-side `EF` value baked into each `ir::Module` as
//! `b.const_fpext([...])`. Its four coefficients are the *canonical*
//! `u32`s of the EF value's basis representation — codegen converts to
//! Montgomery internally, so no conversion is needed on the Rust side.
//! (In the dense case the modules that don't consume padding never touch
//! `alpha`, but we keep the API mirroring the blackbox wrappers.)
//!
//! # Data-dependent challenges as `[D_EF] BabyBear` `BufId`s
//!
//! Challenges like `lambda` and `r` come from `sample_ext` as
//! `[D_EF]`-shaped `BabyBear` buffers. Inside a module they are lifted to
//! an `FpExt` scalar via [`super::fractional_ir::load_ext_coeffs`] +
//! [`super::fractional_ir::fpext_from_coeffs`] and re-used via
//! [`IRBuilder::let_bound`].
//!
//! # Frac<EF> binding: [n, 2] FpExt
//!
//! A `Frac<EF>` element is `(p: EF, q: EF)` = 2 * 16 = 32 bytes. The
//! [`crate::logup_zerocheck::fractional_ir::add_frac_ef_buf`] allocates 32
//! bytes per element (`elem_size = 32`). Inside a module we bind it as
//! `[n, 2] FpExt`: rows are Fracs, column 0 is `p`, column 1 is `q`. The
//! byte size matches (`n * 2 * 16 == 32 * n`).

use crypto_compiler::{
    field_ext::ef_inverse_coeffs,
    graph_ir::{BufId, GraphBuilder},
    ir::{IRBuilder, Module, NodeId, ScalarType},
};

use super::fractional_ir::{fpext_from_coeffs, load_ext_coeffs, FRAC_EF_BYTES};
use crate::{logup_zerocheck::fractional_ir::GKR_S_DEG, types::D_EF};

// ---------------------------------------------------------------------------
// Small helpers.

/// Bind a `[D_EF]`-shaped BabyBear challenge input, lift it to an `FpExt`
/// scalar, and `let_bound` it so the recombine fires once per launch
/// rather than once per compute thread.
fn bind_challenge_as_fpext(b: &mut IRBuilder, name: &str) -> NodeId {
    let x = b.input(name, ScalarType::BabyBear, vec![D_EF]);
    let coeffs = load_ext_coeffs(b, x);
    let combined = fpext_from_coeffs(b, coeffs);
    b.let_bound(combined)
}

/// Byte-size guard: a graph buffer bound as `[n, 2] FpExt` must have been
/// allocated with byte size `n * 32` (= `n * FRAC_EF_BYTES`).
fn assert_frac_size(_n: usize) {
    debug_assert_eq!(FRAC_EF_BYTES, 32, "Frac<EF> size drift");
}

// ---------------------------------------------------------------------------
// Kernel 1: fold_ef_frac_columns (dev-challenge, dense out-of-place).
//
// Fold pattern (from `fold_ef_columns_kernel`, dense branch):
//   quarter = size / 4;
//   half    = size / 2;
//   for idx in [0, quarter):
//     dst[idx]           = fold(src[idx],       src[idx+quarter])   with r
//     dst[idx+quarter]   = fold(src[idx+half],  src[idx+half+quarter])
//   where fold(a, b) = (a.p + r*(b.p-a.p), a.q + r*(b.q-a.q))
//
// Output length is `size/2` Fracs.
//
// Note this ports ONLY the dense branch (real_len == logical_len ==
// size). The virtual/compact branch reads through `virtual_node_value` at
// a bit-reversed index and is not expressible as a quasi-affine index in
// the DSL.

/// Build the DSL module for a dense out-of-place `fold_ef_frac_columns`.
///
/// Inputs:
///   - `src : [size, 2] FpExt` (a Frac<EF> buffer, `size` elements)
///   - `r   : [D_EF] BabyBear` (folding challenge)
///
/// Output:
///   - `dst : [size/2, 2] FpExt` (a Frac<EF> buffer of half the length)
///
/// Dense-only (`real_len == logical_len == size`).
pub fn build_fold_ef_frac_columns_module(size: usize) -> Module {
    assert!(
        size >= 4 && size.is_power_of_two(),
        "fold module: size must be a power of two >= 4, got {size}"
    );
    let out_len = size / 2;
    let quarter = size / 4;
    let mut b = IRBuilder::new();
    let src = b.input("src", ScalarType::FpExt, vec![size, 2]);
    let r = bind_challenge_as_fpext(&mut b, "r");

    let body = b.compute(out_len, move |b, i| {
        // Determine the two source indices for output row `i`:
        //   a_idx = i + (i / quarter) * quarter
        //   b_idx = a_idx + quarter
        // For `i in [0, quarter)`: a_idx = i,           b_idx = i + quarter
        // For `i in [quarter, 2q)`: a_idx = i + quarter, b_idx = i + 2*quarter
        let quarter_c = b.const_u32(quarter as u32);
        let g = b.div(i, quarter_c);
        let off = b.mul(g, quarter_c);
        let a_idx = b.add(i, off);
        let b_idx = b.add(a_idx, quarter_c);

        let zero_c = b.const_u32(0);
        let one_c = b.const_u32(1);
        let ap = b.index(src, &[a_idx, zero_c]);
        let aq = b.index(src, &[a_idx, one_c]);
        let bp = b.index(src, &[b_idx, zero_c]);
        let bq = b.index(src, &[b_idx, one_c]);

        // out.p = a.p + r * (b.p - a.p),  out.q = a.q + r * (b.q - a.q)
        let dp = b.sub(bp, ap);
        let dq = b.sub(bq, aq);
        let rdp = b.mul(r, dp);
        let rdq = b.mul(r, dq);
        let op = b.add(ap, rdp);
        let oq = b.add(aq, rdq);
        b.pack(&[op, oq])
    });
    b.finish(format!("fold_ef_frac_columns_dsl_{size}"), body)
}

/// Insert an out-of-place dense fold as a structured [`GraphNode::Kernel`].
/// Mirrors [`super::fractional_ir::fold_ef_frac_columns_ir_bufid`] but
/// under [`GraphBuilder::insert_kernel`]. Dense-only.
pub fn fold_ef_frac_columns_ir_dsl(
    g: &mut GraphBuilder,
    src: BufId,
    dst: BufId,
    size: usize,
    r: BufId,
) {
    assert_frac_size(size);
    g.insert_kernel(build_fold_ef_frac_columns_module(size), [src, r], [dst]);
}

// ---------------------------------------------------------------------------
// Kernel 6: frac_multifold (dense).
//
// From `multifold_kernel<W>` (dense branch):
//   beta_size = 2^W
//   poly_stride = tail_size << W    (= tail_size * beta_size, and also =
//                                    pre-fold pq_size / 2)
//   For out_idx in [0, tail_size):
//     for beta in [0, beta_size):
//       v0 = src[beta*tail_size + out_idx]
//       v1 = src[poly_stride + beta*tail_size + out_idx]
//       acc0 += eq_r_window[beta] * v0
//       acc1 += eq_r_window[beta] * v1
//     dst[out_idx]              = acc0
//     dst[tail_size + out_idx]  = acc1
//
// Output length is `2 * tail_size` Fracs.
//
// Dense-only: virtual_node_value branch is skipped.

/// Build the DSL module for a dense out-of-place `frac_multifold` at a
/// fixed compile-time window `w`.
///
/// Inputs:
///   - `src         : [pre_size, 2] FpExt` — pre-fold Frac<EF> buffer, `pre_size = 2 * poly_stride
///     = 2 * tail_size * 2^w`
///   - `eq_r_window : [2^w] FpExt`
///
/// Output:
///   - `dst : [2*tail_size, 2] FpExt`
///
/// Dense-only.
pub fn build_frac_multifold_module(tail_size: usize, w: usize) -> Module {
    assert!(
        (1..=6).contains(&w),
        "multifold module: w must be in 1..=6, got {w}"
    );
    let beta_size = 1usize << w;
    let poly_stride = tail_size * beta_size;
    let pre_size = 2 * poly_stride;
    let out_len = 2 * tail_size;
    let mut b = IRBuilder::new();
    let src = b.input("src", ScalarType::FpExt, vec![pre_size, 2]);
    let eq_r_window = b.input("eq_r_window", ScalarType::FpExt, vec![beta_size]);

    let body = b.compute(out_len, move |b, i| {
        // Which poly (0 or 1) and which tail index `out_idx`.
        let tail_c = b.const_u32(tail_size as u32);
        let poly = b.div(i, tail_c); // 0 or 1
        let poly_off = b.mul(poly, tail_c);
        let out_idx = b.sub(i, poly_off);
        // base = poly * poly_stride + out_idx
        let poly_stride_c = b.const_u32(poly_stride as u32);
        let poly_base = b.mul(poly, poly_stride_c);
        let base = b.add(poly_base, out_idx);

        let zero_c = b.const_u32(0);
        let one_c = b.const_u32(1);
        // Reduce over beta: acc_p += eq[beta] * src[base + beta*tail_size, 0];
        //                   acc_q += eq[beta] * src[base + beta*tail_size, 1];
        // Do two reduces (once for p, once for q) — both hash-cons the
        // same base indexing so the compiler CSEs the shared work.
        let acc_p = b.reduce_add(beta_size, move |b, beta| {
            let beta_t = b.mul(beta, tail_c);
            let idx = b.add(base, beta_t);
            let v = b.index(src, &[idx, zero_c]);
            let eq_r = b.index(eq_r_window, &[beta]);
            b.mul(eq_r, v)
        });
        let acc_q = b.reduce_add(beta_size, move |b, beta| {
            let beta_t = b.mul(beta, tail_c);
            let idx = b.add(base, beta_t);
            let v = b.index(src, &[idx, one_c]);
            let eq_r = b.index(eq_r_window, &[beta]);
            b.mul(eq_r, v)
        });
        b.pack(&[acc_p, acc_q])
    });
    b.finish(format!("frac_multifold_dsl_ts{tail_size}_w{w}"), body)
}

/// Insert a dense out-of-place `frac_multifold` as a structured kernel.
/// Mirrors [`super::fractional_ir::frac_multifold_ir`] but under
/// [`GraphBuilder::insert_kernel`]. Dense-only.
pub fn frac_multifold_ir_dsl(
    g: &mut GraphBuilder,
    src: BufId,
    dst: BufId,
    eq_r_window: BufId,
    tail_size: usize,
    w: usize,
) {
    g.insert_kernel(
        build_frac_multifold_module(tail_size, w),
        [src, eq_r_window],
        [dst],
    );
}

// ---------------------------------------------------------------------------
// Kernel 5: frac_precompute_m_eval_round.
//
// From `precompute_m_eval_round_kernel`:
//   prefix_bits = t; suffix_bits = w - t - 1;
//   prefix_size = 1 << prefix_bits; suffix_size = 1 << suffix_bits;
//   total = prefix_size * prefix_size * suffix_size = 2^(t + w - 1)
//   For idx in [0, total):
//     suffix = idx % suffix_size
//     tmp    = idx / suffix_size
//     b2     = tmp % prefix_size
//     b1     = tmp / prefix_size
//     weight = eq_r_prefix[b1] * eq_r_prefix[b2] * eq_suffix[suffix]
//     beta1_0 = (b1 << (suffix_bits + 1)) | suffix
//     beta1_1 = beta1_0 | (1 << suffix_bits)
//     beta2_0 = (b2 << (suffix_bits + 1)) | suffix
//     beta2_1 = beta2_0 | (1 << suffix_bits)
//     m = 1 << w
//     m00 = m_total[beta1_0 * m + beta2_0]
//     m01 = m_total[beta1_0 * m + beta2_1]
//     m10 = m_total[beta1_1 * m + beta2_0]
//     m11 = m_total[beta1_1 * m + beta2_1]
//     local_s1 += weight * m11
//     local_s2 += weight * (m00 - 2*(m01 + m10 - m11 - m11))
//   out[0] = sum of local_s1, out[1] = sum of local_s2

/// Build the DSL module for `frac_precompute_m_eval_round`.
///
/// Inputs:
///   - `m_total     : [m*m] FpExt` where `m = 1 << w`
///   - `eq_r_prefix : [1 << t] FpExt`
///   - `eq_suffix   : [1 << (w - t - 1)] FpExt`
///
/// Output:
///   - `out : [2] FpExt` (s'(1), s'(2))
pub fn build_frac_precompute_m_eval_round_module(w: usize, t: usize) -> Module {
    assert!(w >= 1, "precompute_m_eval_round: w must be >= 1, got {w}");
    assert!(
        t < w,
        "precompute_m_eval_round: t must be < w (got t={t}, w={w})"
    );
    let m = 1usize << w;
    let prefix_bits = t;
    let suffix_bits = w - t - 1;
    let prefix_size = 1usize << prefix_bits;
    let suffix_size = 1usize << suffix_bits;
    let total = prefix_size * prefix_size * suffix_size;

    let cur_bit = 1usize << suffix_bits;
    let prefix_shift = suffix_bits + 1;

    let mut b = IRBuilder::new();
    let m_total = b.input("m_total", ScalarType::FpExt, vec![m * m]);
    let eq_r_prefix = b.input("eq_r_prefix", ScalarType::FpExt, vec![prefix_size]);
    let eq_suffix = b.input("eq_suffix", ScalarType::FpExt, vec![suffix_size]);

    // Both outputs share the same reduction domain; produce a `[2]` tensor
    // via `compute(2)` where each entry is a `reduce_add` over `total`.
    // Hash-consing lets the compiler CSE all the shared per-iteration
    // work between the two reductions.
    let body = b.compute(2, move |b, out_i| {
        b.reduce_add(total, move |b, idx| {
            let suffix_c = b.const_u32(suffix_size as u32);
            let prefix_c = b.const_u32(prefix_size as u32);
            // suffix = idx % suffix_size ; tmp = idx / suffix_size
            let tmp = b.div(idx, suffix_c);
            let tmp_off = b.mul(tmp, suffix_c);
            let suffix = b.sub(idx, tmp_off);
            // b2 = tmp % prefix_size, b1 = tmp / prefix_size
            let b1 = b.div(tmp, prefix_c);
            let b1_off = b.mul(b1, prefix_c);
            let b2 = b.sub(tmp, b1_off);

            let weight = {
                let eq_b1 = b.index(eq_r_prefix, &[b1]);
                let eq_b2 = b.index(eq_r_prefix, &[b2]);
                let eq_s = b.index(eq_suffix, &[suffix]);
                let p = b.mul(eq_b1, eq_b2);
                b.mul(p, eq_s)
            };

            // beta1_0 = (b1 << prefix_shift) | suffix
            //         = b1 * (1 << prefix_shift) + suffix          (disjoint bits: suffix <
            // 1<<suffix_bits < 1<<prefix_shift) beta1_1 = beta1_0 | cur_bit = beta1_0 +
            // cur_bit      (bit already unset in beta1_0)
            let ps = b.const_u32((1u32) << prefix_shift);
            let cur_bit_c = b.const_u32(cur_bit as u32);
            let b1_shift = b.mul(b1, ps);
            let b2_shift = b.mul(b2, ps);
            let beta1_0 = b.add(b1_shift, suffix);
            let beta1_1 = b.add(beta1_0, cur_bit_c);
            let beta2_0 = b.add(b2_shift, suffix);
            let beta2_1 = b.add(beta2_0, cur_bit_c);

            // Row-major m_total[beta1 * m + beta2] flattened.
            let m_c = b.const_u32(m as u32);
            let idx00 = {
                let b1m = b.mul(beta1_0, m_c);
                b.add(b1m, beta2_0)
            };
            let idx01 = {
                let b1m = b.mul(beta1_0, m_c);
                b.add(b1m, beta2_1)
            };
            let idx10 = {
                let b1m = b.mul(beta1_1, m_c);
                b.add(b1m, beta2_0)
            };
            let idx11 = {
                let b1m = b.mul(beta1_1, m_c);
                b.add(b1m, beta2_1)
            };
            let m00 = b.index(m_total, &[idx00]);
            let m01 = b.index(m_total, &[idx01]);
            let m10 = b.index(m_total, &[idx10]);
            let m11 = b.index(m_total, &[idx11]);

            // s1 branch: weight * m11
            //
            // s2 branch: weight * (m00 - 2*(m01 + m10 - m11 - m11))
            //          = weight * (m00 - 2*m01 - 2*m10 + 4*m11)
            let two_e = b.const_fpext([2, 0, 0, 0]);
            let sum_val_s1 = m11;
            let s2_01 = b.add(m01, m10);
            let s2_01_2 = b.mul(two_e, s2_01);
            let s2_11 = b.add(m11, m11);
            let s2_inner1 = b.sub(s2_01_2, s2_11);
            let s2_inner2 = b.sub(s2_inner1, s2_11);
            let sum_val_s2 = b.sub(m00, s2_inner2);

            // out_i == 0 -> s1, out_i == 1 -> s2.
            let zero_u = b.const_u32(0);
            let is_s1 = b.eq(out_i, zero_u);
            let val = b.select(is_s1, sum_val_s1, sum_val_s2);
            b.mul(weight, val)
        })
    });
    b.finish(format!("frac_precompute_m_eval_round_dsl_w{w}_t{t}"), body)
}

/// Insert a `frac_precompute_m_eval_round` as a structured kernel. Mirrors
/// [`super::fractional_ir::frac_precompute_m_eval_round_ir`].
pub fn frac_precompute_m_eval_round_ir_dsl(
    g: &mut GraphBuilder,
    m_total: BufId,
    eq_r_prefix: BufId,
    eq_suffix: BufId,
    out: BufId,
    w: usize,
    t: usize,
) {
    g.insert_kernel(
        build_frac_precompute_m_eval_round_module(w, t),
        [m_total, eq_r_prefix, eq_suffix],
        [out],
    );
}

// ---------------------------------------------------------------------------
// Kernel 2: frac_compute_round (dev-challenge, dense).
//
// From `compute_round_block_sum_kernel<DEV_CH=true>`, dense branch:
//   pq_size = 2 * num_x
//   For idx in [0, num_x/2):
//     eq_val = sqrt_buffer_get(eq_low, eq_high, log_eq_low_cap, idx)
//            = eq_low[idx & (eq_low_cap - 1)] * eq_high[idx >> log_eq_low_cap]
//     (p0_even, q0_even) = pq_buffer[idx]                             (bits ..00)
//     (p1_even, q1_even) = pq_buffer[with_rev_bits(idx, pq_size, 1, 0)]
//     (p0_odd , q0_odd ) = pq_buffer[with_rev_bits(idx, pq_size, 0, 1)]
//     (p1_odd , q1_odd ) = pq_buffer[with_rev_bits(idx, pq_size, 1, 1)]
//     ... accumulate contributions into local[0], local[1] ...
//   sum-reduce local[i] over the grid -> out[i]
//
// `with_rev_bits(idx, pq_size, hi, lo)` inserts two bits `(hi, lo)` at the
// top of `idx` in bit-reversed position (i.e. `hi` is the very top bit,
// `lo` is next-below). In dense terms it evaluates to:
//   `idx | (hi * pq_size/2) | (lo * pq_size/4)`
// -- see the DSL comment below.

/// with_rev_bits helper. The CUDA function `with_rev_bits(idx, size, hi,
/// lo)` sets the top bit (`log2(size) - 1`) to `hi` and the bit below it
/// (`log2(size) - 2`) to `lo`, with `idx` occupying the low bits.
/// Equivalent to `idx + hi * (size/2) + lo * (size/4)` when the two top
/// bits of `idx` are zero (which they are here: `idx < num_x/2 =
/// pq_size/4`, so idx has ceil(log2(pq_size))-2 low bits).
fn with_rev_bits_dsl(
    b: &mut IRBuilder,
    idx: NodeId,
    pq_size: usize,
    hi: usize,
    lo: usize,
) -> NodeId {
    let mut off = 0usize;
    if hi != 0 {
        off += pq_size / 2;
    }
    if lo != 0 {
        off += pq_size / 4;
    }
    if off == 0 {
        idx
    } else {
        // Disjoint bits: `idx < pq_size/4`, so the (hi, lo) bits are
        // guaranteed zero in `idx` — safe to `or` via `add`.
        b.or(idx, off)
    }
}

/// Build the DSL module for `frac_compute_round` (dev-challenge, dense).
///
/// Inputs:
///   - `eq_low     : [eq_low_cap] FpExt`
///   - `eq_high    : [num_x / 2 / eq_low_cap] FpExt`
///   - `pq_buffer  : [2*num_x, 2] FpExt`
///   - `lambda     : [D_EF] BabyBear`
///
/// Output:
///   - `out : [2] FpExt` — the block-summed `(s'(1), s'(2))` pair.
///
/// Dense-only (`real_len == logical_len == 2*num_x`).
pub fn build_frac_compute_round_module(num_x: usize, eq_low_cap: usize) -> Module {
    assert!(num_x.is_power_of_two() && num_x >= 2);
    assert!(eq_low_cap.is_power_of_two());
    let pq_size = 2 * num_x;
    let iter = num_x / 2;
    let eq_high_cap = iter / eq_low_cap;
    assert!(
        eq_high_cap.is_power_of_two() && eq_high_cap * eq_low_cap == iter,
        "eq_low_cap must partition num_x/2"
    );
    let mut b = IRBuilder::new();
    let eq_low = b.input("eq_low", ScalarType::FpExt, vec![eq_low_cap]);
    let eq_high = b.input("eq_high", ScalarType::FpExt, vec![eq_high_cap]);
    let pq = b.input("pq", ScalarType::FpExt, vec![pq_size, 2]);
    let lambda = bind_challenge_as_fpext(&mut b, "lambda");

    let body = b.compute(GKR_S_DEG - 1, move |b, out_i| {
        b.reduce_add(iter, move |b, idx| {
            // eq_val = eq_low[idx & (eq_low_cap-1)] * eq_high[idx >> log_eq_low_cap]
            let low_c = b.const_u32(eq_low_cap as u32);
            let lo_idx = b.rem(idx, low_c);
            let hi_idx = b.div(idx, low_c);
            let el = b.index(eq_low, &[lo_idx]);
            let eh = b.index(eq_high, &[hi_idx]);
            let eq_val = b.mul(el, eh);

            // Load the four pq slots for this idx.
            let zero_c = b.const_u32(0);
            let one_c = b.const_u32(1);
            let read = |b: &mut IRBuilder, at: NodeId| -> (NodeId, NodeId) {
                let p = b.index(pq, &[at, zero_c]);
                let q = b.index(pq, &[at, one_c]);
                (p, q)
            };
            let (p0_e, q0_e) = read(b, idx);
            let idx_10 = with_rev_bits_dsl(b, idx, pq_size, 1, 0);
            let (p1_e, q1_e) = read(b, idx_10);
            let idx_01 = with_rev_bits_dsl(b, idx, pq_size, 0, 1);
            let (p0_o, q0_o) = read(b, idx_01);
            let idx_11 = with_rev_bits_dsl(b, idx, pq_size, 1, 1);
            let (p1_o, q1_o) = read(b, idx_11);

            let contrib = compute_round_contrib(
                b, out_i, lambda, p0_e, q0_e, p0_o, q0_o, p1_e, q1_e, p1_o, q1_o,
            );
            b.mul(eq_val, contrib)
        })
    });
    b.finish(
        format!("frac_compute_round_dsl_n{num_x}_c{eq_low_cap}"),
        body,
    )
}

/// Emit the per-idx contribution used by the compute-round module,
/// selecting either the `s'(1)` or `s'(2)` accumulator branch by
/// `out_i`. Matches the CUDA `accumulate_compute_contributions` unrolled
/// loop:
///   p_j0 = p0_even + lambda * q0_even;
///   q_j0 = q0_even;
///   p_j1 = p1_even;
///   q_j1 = q1_even;
///   for i in 0..2:
///     p_j0 += p0_diff + lambda*q0_diff;
///     q_j0 += q0_diff;
///     p_j1 += p1_diff;
///     q_j1 += q1_diff;
///     contrib_i = p_j0 * q_j1 + p_j1 * q_j0
///
/// This unrolls both iterations and selects between them on `out_i`.
#[allow(clippy::too_many_arguments)]
fn compute_round_contrib(
    b: &mut IRBuilder,
    out_i: NodeId,
    lambda: NodeId,
    p0_e: NodeId,
    q0_e: NodeId,
    p0_o: NodeId,
    q0_o: NodeId,
    p1_e: NodeId,
    q1_e: NodeId,
    p1_o: NodeId,
    q1_o: NodeId,
) -> NodeId {
    // diffs
    let p0d = b.sub(p0_o, p0_e);
    let q0d = b.sub(q0_o, q0_e);
    let p1d = b.sub(p1_o, p1_e);
    let q1d = b.sub(q1_o, q1_e);
    let l_q0d = b.mul(lambda, q0d);
    // running p_j0/q_j0/p_j1/q_j1 initial:
    let l_q0e = b.mul(lambda, q0_e);
    let mut p_j0 = b.add(p0_e, l_q0e);
    let mut q_j0 = q0_e;
    let mut p_j1 = p1_e;
    let mut q_j1 = q1_e;

    // First iteration
    p_j0 = b.add(p_j0, p0d);
    p_j0 = b.add(p_j0, l_q0d);
    q_j0 = b.add(q_j0, q0d);
    p_j1 = b.add(p_j1, p1d);
    q_j1 = b.add(q_j1, q1d);
    let c1_a = b.mul(p_j0, q_j1);
    let c1_b = b.mul(p_j1, q_j0);
    let contrib_1 = b.add(c1_a, c1_b);

    // Second iteration
    p_j0 = b.add(p_j0, p0d);
    p_j0 = b.add(p_j0, l_q0d);
    q_j0 = b.add(q_j0, q0d);
    p_j1 = b.add(p_j1, p1d);
    q_j1 = b.add(q_j1, q1d);
    let c2_a = b.mul(p_j0, q_j1);
    let c2_b = b.mul(p_j1, q_j0);
    let contrib_2 = b.add(c2_a, c2_b);

    let zero_u = b.const_u32(0);
    let is_first = b.eq(out_i, zero_u);
    b.select(is_first, contrib_1, contrib_2)
}

/// Insert a `frac_compute_round` (dev-challenge, dense) as a structured
/// kernel. Mirrors [`super::fractional_ir::frac_compute_round_ir_bufid`]
/// but produces a `[2] FpExt` output directly (no per-block temp scratch —
/// the reduction is folded into the one-kernel module).
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_ir_dsl(
    g: &mut GraphBuilder,
    eq_low: BufId,
    eq_high: BufId,
    pq_buffer: BufId,
    lambda: BufId,
    out: BufId,
    num_x: usize,
    eq_low_cap: usize,
) {
    g.insert_kernel(
        build_frac_compute_round_module(num_x, eq_low_cap),
        [eq_low, eq_high, pq_buffer, lambda],
        [out],
    );
}

// ---------------------------------------------------------------------------
// Kernel 4: frac_compute_round_and_fold (dev-challenge, dense).
//
// Two modules composed at the graph builder level:
//   1. Fold module (kernel 1): src_pq -> folded_pq using r_prev.
//   2. Compute-round module (kernel 2): folded_pq -> out using lambda.
//
// The two-module composition is not "fused" in the CUDA sense but is
// exactly equivalent in output; the memory planner may alias buffers.
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_and_fold_ir_dsl(
    g: &mut GraphBuilder,
    eq_low: BufId,
    eq_high: BufId,
    src_pq_buffer: BufId,
    dst_pq_buffer: BufId,
    lambda: BufId,
    r_prev: BufId,
    out: BufId,
    src_pq_size: usize,
    eq_low_cap: usize,
) {
    // Fold: src (size = src_pq_size) -> dst (size = src_pq_size / 2)
    fold_ef_frac_columns_ir_dsl(g, src_pq_buffer, dst_pq_buffer, src_pq_size, r_prev);
    // Compute-round on the folded buffer. num_x = src_pq_size / 4.
    let num_x = src_pq_size >> 2;
    frac_compute_round_ir_dsl(
        g,
        eq_low,
        eq_high,
        dst_pq_buffer,
        lambda,
        out,
        num_x,
        eq_low_cap,
    );
}

// ---------------------------------------------------------------------------
// Kernel 8: frac_build_tree_two_layers (dense).
//
// From `frac_build_tree_two_layers_kernel`:
//   half_i = 2 * half_i1
//   layer_size = 4 * half_i1
//   For j in [0, half_i1):
//     A = layer[j]
//     B = layer[j + half_i1]
//     C = layer[j + half_i]
//     D = layer[j + half_i + half_i1]
//     lhs    = frac_add(A, C)
//     rhs    = frac_add(B, D)
//     result = frac_add(lhs, rhs)
//     layer[j]           = result
//     layer[j + half_i1] = rhs
//     (layer[j + half_i], layer[j + half_i + half_i1] untouched)
//
// Where frac_add((p, q), (p', q')) = (p*q' + q*p', q*q').
//
// Functional port produces a fresh output buffer of the same shape:
//   out[j]                    = result       for j in [0, half_i1)
//   out[j]                    = layer[j]     for j in [half_i1, half_i)   ← rhs of that thread
//   ... wait, out[j] for j in [half_i1, half_i) is rhs of thread (j -
//   half_i1). Let me re-read.
//
// Actually looking again:
//   layer[j]           = result       (j in [0, half_i1))
//   layer[j + half_i1] = rhs          (j + half_i1 in [half_i1, half_i))
//   layer[j + half_i]           unchanged   (j + half_i in [half_i, 3*half_i1))
//   layer[j + half_i + half_i1] unchanged   (in [3*half_i1, 4*half_i1))
//
// So the output layer is:
//   out[k] = result_{k}         for k in [0, half_i1)         (k = j)
//   out[k] = rhs_{k - half_i1}  for k in [half_i1, half_i)     (k = j + half_i1)
//   out[k] = in[k]              for k in [half_i, layer_size) (k = j + half_i or j + half_i +
// half_i1)

/// Build the DSL module for a dense `frac_build_tree_two_layers`.
///
/// Inputs:
///   - `layer_in : [layer_size, 2] FpExt`
///
/// Output:
///   - `layer_out : [layer_size, 2] FpExt`
///
/// Where `layer_size = 4 * half_i1`. Dense-only.
pub fn build_frac_build_tree_two_layers_module(half_i1: usize) -> Module {
    assert!(half_i1 >= 1, "half_i1 must be >= 1");
    let half_i = 2 * half_i1;
    let layer_size = 4 * half_i1;
    let mut b = IRBuilder::new();
    let layer_in = b.input("layer_in", ScalarType::FpExt, vec![layer_size, 2]);

    let body = b.compute(layer_size, move |b, k| {
        let zero_c = b.const_u32(0);
        let one_c = b.const_u32(1);
        let h1 = b.const_u32(half_i1 as u32);
        let hi = b.const_u32(half_i as u32);

        // Region 1: k in [0, half_i1)          -> result = frac_add(frac_add(A,C), frac_add(B,D))
        // Region 2: k in [half_i1, half_i)     -> rhs = frac_add(B, D), j = k - half_i1
        // Region 3: k in [half_i, layer_size)  -> unchanged (layer_in[k])

        // Region-boundary flags.
        let is_r1 = b.lt(k, h1);
        let is_r2 = b.lt(k, hi);
        // is_r2 alone doesn't distinguish; combine with !is_r1 outside if needed.

        // Compute all three candidates and select — the compiler should
        // DCE the unused ones per-region under the select on out.
        // A/B/C/D at region-1 offsets (j = k):
        let read = |b: &mut IRBuilder, i: NodeId| -> (NodeId, NodeId) {
            let p = b.index(layer_in, &[i, zero_c]);
            let q = b.index(layer_in, &[i, one_c]);
            (p, q)
        };
        // For region-1, j = k
        let j1 = k;
        let j1_h1 = b.add(j1, h1);
        let j1_hi = b.add(j1, hi);
        let j1_hi_h1 = b.add(j1_hi, h1);
        let (ap, aq) = read(b, j1);
        let (bp, bq) = read(b, j1_h1);
        let (cp, cq) = read(b, j1_hi);
        let (dp, dq) = read(b, j1_hi_h1);
        let (ac_p, ac_q) = frac_add(b, ap, aq, cp, cq);
        let (bd_p, bd_q) = frac_add(b, bp, bq, dp, dq);
        let (r1_p, r1_q) = frac_add(b, ac_p, ac_q, bd_p, bd_q);

        // For region-2, j = k - half_i1, we need `rhs = frac_add(B, D)`.
        // With `j2 = k - half_i1`, B = layer_in[j2 + half_i1] = layer_in[k],
        // D = layer_in[j2 + half_i + half_i1] = layer_in[k + half_i].
        // So rhs uses layer_in[k] and layer_in[k + half_i].
        let k_hi = b.add(k, hi);
        let (b2p, b2q) = read(b, k);
        let (d2p, d2q) = read(b, k_hi);
        let (r2_p, r2_q) = frac_add(b, b2p, b2q, d2p, d2q);

        // Region-3: unchanged.
        let (r3_p, r3_q) = read(b, k);

        // Assemble.
        let sel_p_23 = b.select(is_r2, r2_p, r3_p);
        let sel_q_23 = b.select(is_r2, r2_q, r3_q);
        let out_p = b.select(is_r1, r1_p, sel_p_23);
        let out_q = b.select(is_r1, r1_q, sel_q_23);
        b.pack(&[out_p, out_q])
    });
    b.finish(format!("frac_build_tree_two_layers_dsl_h{half_i1}"), body)
}

/// frac_add((p, q), (p', q')) = (p*q' + q*p', q*q').
fn frac_add(b: &mut IRBuilder, p: NodeId, q: NodeId, p2: NodeId, q2: NodeId) -> (NodeId, NodeId) {
    let pq2 = b.mul(p, q2);
    let qp2 = b.mul(q, p2);
    let out_p = b.add(pq2, qp2);
    let out_q = b.mul(q, q2);
    (out_p, out_q)
}

/// Insert an out-of-place dense `frac_build_tree_two_layers` as a
/// structured kernel. Mirrors
/// [`super::fractional_ir::frac_build_tree_two_layers_ir`] but produces a
/// fresh output buffer of the same length rather than modifying `layer`
/// in place (the DSL is pure).
pub fn frac_build_tree_two_layers_ir_dsl(
    g: &mut GraphBuilder,
    layer_in: BufId,
    layer_out: BufId,
    half_i1: usize,
) {
    g.insert_kernel(
        build_frac_build_tree_two_layers_module(half_i1),
        [layer_in],
        [layer_out],
    );
}

// ---------------------------------------------------------------------------
// Kernel 7: frac_build_tree_layer (revert, dense, apply_alpha=false).
//
// From `frac_build_tree_layer_kernel<revert=true, apply_alpha=false>`:
//   half = layer_size / 2
//   For idx in [0, half):
//     lhs = layer[idx]
//     rhs = layer[idx + half]
//     rhs_q_inv = inv(rhs.q)
//     lhs.q = lhs.q * rhs_q_inv                       (new q)
//     lhs.p = (lhs.p - lhs.q * rhs.p) * rhs_q_inv     (new p, uses new q)
//     layer[idx] = lhs
//     (layer[idx + half] untouched)
//
// Where `inv(x)` is the FpExt inverse. Functional port produces a fresh
// output buffer:
//   out[k] = frac_unadd(in[k], in[k + half])   for k in [0, half)
//   out[k] = in[k]                             for k in [half, layer_size)
//
// This uses ef_inverse_coeffs on rhs.q's four base-field coefficients.
// Since the DSL doesn't have an `FpExt scalar -> [D_EF] BabyBear
// coeffs` projection, we bind the buffer as `[layer_size, 2, D_EF]
// BabyBear` instead of `[layer_size, 2] FpExt` — same 32 bytes per row,
// same alignment (16-byte, since a Frac<EF> is 32 bytes = two 16-byte
// FpExts, and each FpExt's four u32 coefficients are 4-byte aligned).

/// Build the DSL module for a dense `frac_build_tree_layer` with
/// `revert=true` and `apply_alpha=false`.
///
/// Inputs:
///   - `layer_in : [layer_size, 2, D_EF] BabyBear` — the raw base-field view of the input Frac<EF>
///     buffer (32 bytes per row = two 16-byte EFs = 8 base-field u32s). We need the base-field
///     coefficients here to run [`ef_inverse_coeffs`].
///
/// Output:
///   - `layer_out : [layer_size, 2] FpExt` — same 32-byte row size, byte identical to a Frac<EF>
///     buffer.
///
/// # Layout equivalence
///
/// A `Frac<EF>` is `(p: EF, q: EF)` = two 16-byte FpExts. Its raw memory
/// is eight canonical BabyBear u32s: `[p0, p1, p2, p3, q0, q1, q2, q3]`.
/// Binding the input as `[layer_size, 2, D_EF] BabyBear` reads those
/// bytes as base-field values (Montgomery-decoded internally to
/// canonical form for arithmetic, then re-encoded on writes). Binding
/// the output as `[layer_size, 2] FpExt` writes the same 32 bytes per
/// row as two 16-byte Montgomery FpExt scalars.
pub fn build_frac_build_tree_layer_revert_module(layer_size: usize) -> Module {
    assert!(
        layer_size >= 2 && layer_size.is_power_of_two(),
        "revert module: layer_size must be a power of two >= 2, got {layer_size}"
    );
    let half = layer_size / 2;

    let mut b = IRBuilder::new();
    let layer = b.input("layer_in", ScalarType::BabyBear, vec![layer_size, 2, D_EF]);

    let body = b.compute(layer_size, move |b, k| {
        let half_c = b.const_u32(half as u32);
        let is_first_half = b.lt(k, half_c);

        // rhs row = (k + half) mod layer_size. For k in [0, half), this
        // is k + half (the intended rhs). For k in [half, layer_size),
        // it is k - half (a different row); the result is discarded via
        // `select(is_first_half, unadd, in[k])`. `%` by a power-of-two
        // is quasi-affine so the DSL accepts it as an index expression.
        let layer_size_c = b.const_u32(layer_size as u32);
        let k_plus_half = b.add(k, half_c);
        let rhs_row = b.rem(k_plus_half, layer_size_c);

        // Load lhs = layer[k] and rhs = layer[rhs_row], as (p_coeffs,
        // q_coeffs).
        let load_pq_coeffs = |b: &mut IRBuilder, row: NodeId| -> ([NodeId; D_EF], [NodeId; D_EF]) {
            let zero_c = b.const_u32(0);
            let one_c = b.const_u32(1);
            let p_coeffs = std::array::from_fn(|c| {
                let ci = b.const_u32(c as u32);
                b.index(layer, &[row, zero_c, ci])
            });
            let q_coeffs = std::array::from_fn(|c| {
                let ci = b.const_u32(c as u32);
                b.index(layer, &[row, one_c, ci])
            });
            (p_coeffs, q_coeffs)
        };
        let (lhs_p_coeffs, lhs_q_coeffs) = load_pq_coeffs(b, k);
        let (rhs_p_coeffs, rhs_q_coeffs) = load_pq_coeffs(b, rhs_row);

        // rhs_q_inv from norm-based inversion on the base-field
        // coefficients — this is why we bound the input as BabyBear.
        let rhs_q_inv_coeffs = ef_inverse_coeffs(b, rhs_q_coeffs);

        // Recombine to FpExt scalars for the arithmetic.
        let lhs_p = fpext_from_coeffs(b, lhs_p_coeffs);
        let lhs_q = fpext_from_coeffs(b, lhs_q_coeffs);
        let rhs_p = fpext_from_coeffs(b, rhs_p_coeffs);
        let rhs_q_inv = fpext_from_coeffs(b, rhs_q_inv_coeffs);

        // frac_unadd:
        //   new_q = lhs.q * rhs_q_inv
        //   new_p = (lhs.p - new_q * rhs.p) * rhs_q_inv
        let new_q = b.mul(lhs_q, rhs_q_inv);
        let nq_rp = b.mul(new_q, rhs_p);
        let inner = b.sub(lhs_p, nq_rp);
        let new_p = b.mul(inner, rhs_q_inv);

        // "Unchanged" path: FpExt scalars from lhs (row = k).
        let out_p = b.select(is_first_half, new_p, lhs_p);
        let out_q = b.select(is_first_half, new_q, lhs_q);
        b.pack(&[out_p, out_q])
    });
    b.finish(
        format!("frac_build_tree_layer_revert_dsl_{layer_size}"),
        body,
    )
}

/// Insert an out-of-place dense `frac_build_tree_layer` with `revert=true`
/// and `apply_alpha=false` as a structured kernel. Mirrors the
/// revert-only path of [`super::fractional_ir::frac_build_tree_layer_ir`]
/// (the eager wrapper's `revert=true, apply_alpha=false` call).
///
/// Callers must bind `layer_in` as a `Frac<EF>`-sized buffer (32 bytes
/// per row); the module internally views it as `[layer_size, 2, D_EF]
/// BabyBear`. `layer_out` is a fresh `Frac<EF>`-sized buffer.
pub fn frac_build_tree_layer_revert_ir_dsl(
    g: &mut GraphBuilder,
    layer_in: BufId,
    layer_out: BufId,
    layer_size: usize,
) {
    g.insert_kernel(
        build_frac_build_tree_layer_revert_module(layer_size),
        [layer_in],
        [layer_out],
    );
}

// ---------------------------------------------------------------------------
// Kernel 3: frac_compute_round_and_revert (dev-challenge, dense).
//
// Two modules composed at the graph builder level:
//   1. Revert module (kernel 7): layer_pre -> layer_post.
//   2. Compute-round module (kernel 2): layer_post -> out using lambda.
//
// The two-module composition is not "fused" in the CUDA sense but is
// exactly equivalent in output.

/// Insert a `frac_compute_round_and_revert` split as (revert kernel 7) +
/// (compute-round kernel 2). Dense-only.
#[allow(clippy::too_many_arguments)]
pub fn frac_compute_round_and_revert_ir_dsl(
    g: &mut GraphBuilder,
    eq_low: BufId,
    eq_high: BufId,
    layer_in: BufId,
    layer_post_revert: BufId,
    lambda: BufId,
    out: BufId,
    layer_size: usize,
    eq_low_cap: usize,
) {
    // Revert: layer_in -> layer_post_revert (both `layer_size` Fracs).
    frac_build_tree_layer_revert_ir_dsl(g, layer_in, layer_post_revert, layer_size);
    // Compute-round on the reverted layer. num_x = layer_size / 2.
    let num_x = layer_size >> 1;
    frac_compute_round_ir_dsl(
        g,
        eq_low,
        eq_high,
        layer_post_revert,
        lambda,
        out,
        num_x,
        eq_low_cap,
    );
}

// ---------------------------------------------------------------------------
// Tests.

#[cfg(test)]
mod dsl_port_tests {
    use crypto_compiler::{
        graph_exe::GraphCompiler,
        graph_ir::{ConstBuf, DeviceType, GraphBuilder},
        planner::SchedulerMode,
        runtime::CompileOptions,
    };
    use openvm_cuda_common::{
        common::get_device,
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{
        cuda::logup_zerocheck::{
            _frac_compute_round_temp_buffer_size, fold_ef_frac_columns, frac_build_tree_two_layers,
            frac_compute_round, frac_multifold_raw, frac_precompute_m_eval_round_raw,
        },
        logup_zerocheck::fractional_ir::{
            add_ef_buf, add_frac_ef_buf, ef_const_ext_scalar_buf, FRAC_EF_BYTES,
        },
        poly::SqrtEqLayers,
        prelude::EF,
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

    fn frac_bytes(leaves: &[Frac<EF>]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(leaves.as_ptr() as *const u8, std::mem::size_of_val(leaves))
        }
    }

    /// Stage `leaves` as a Frac<EF> const buffer.
    fn frac_const_buf(g: &mut GraphBuilder, name: &str, leaves: &[Frac<EF>]) -> BufId {
        let buf = add_frac_ef_buf(g, DeviceType::Cuda(0), name, leaves.len());
        g.insert_const(buf, ConstBuf::HostBuf(frac_bytes(leaves).to_vec()));
        buf
    }

    /// Stage a slice of EFs as an EF const buffer (aligned 16).
    fn ef_slice_const_buf(g: &mut GraphBuilder, name: &str, xs: &[EF]) -> BufId {
        let buf = add_ef_buf(g, DeviceType::Cuda(0), name, xs.len());
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)).to_vec()
        };
        g.insert_const(buf, ConstBuf::HostBuf(bytes));
        buf
    }

    /// Compile a graph with no runtime inputs, run it, and read back the
    /// given buffers as raw bytes.
    fn run_graph_read_bufs(g: GraphBuilder, bufs: &[BufId], ctx: &GpuDeviceCtx) -> Vec<Vec<u8>> {
        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .scheduler(SchedulerMode::Heuristic)
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");
        let inputs: Vec<DeviceBuffer<u8>> = Vec::new();
        let mut outputs: Vec<DeviceBuffer<u8>> = (0..exe.num_outputs())
            .map(|i| DeviceBuffer::<u8>::with_capacity_on(exe.output_size(i), ctx))
            .collect();
        let mut scratch = DeviceBuffer::<u8>::with_capacity_on(exe.scratch_bytes().max(1), ctx);
        exe.run(ctx, &inputs, &mut outputs, &mut scratch)
            .expect("graph run");
        bufs.iter()
            .map(|&bid| {
                let idx = (0..exe.num_outputs())
                    .find(|&i| exe.output_buf_id(i) == bid)
                    .expect("output buf");
                outputs[idx].to_host_on(ctx).expect("D2H")
            })
            .collect()
    }

    #[test]
    fn fold_ef_frac_columns_dsl_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0xF01D_5EED);

        for size in [8usize, 16, 32, 64, 128, 256] {
            let src = make_host_leaves(size, 0xF01D ^ size as u64);
            let r: EF = rng.random();
            let alpha: EF = rng.random();

            // Eager reference (dense: real_len == logical_len == size).
            let src_dev: DeviceBuffer<Frac<EF>> = src.as_slice().to_device_on(&ctx).unwrap();
            let mut dst_dev: DeviceBuffer<Frac<EF>> =
                DeviceBuffer::with_capacity_on(size / 2, &ctx);
            unsafe {
                fold_ef_frac_columns(&src_dev, &mut dst_dev, size, size, size, r, alpha, stream)
                    .expect("fold_ef_frac_columns");
            }
            ctx.stream.synchronize().unwrap();
            let want = dst_dev.to_host_on(&ctx).unwrap();

            // DSL side.
            let mut g = GraphBuilder::new();
            let src_buf = frac_const_buf(&mut g, "src", &src);
            let r_buf = ef_const_ext_scalar_buf(&mut g, device, "r", r);
            let dst_buf = add_frac_ef_buf(&mut g, device, "dst", size / 2);
            fold_ef_frac_columns_ir_dsl(&mut g, src_buf, dst_buf, size, r_buf);
            let dst_out = add_frac_ef_buf(&mut g, device, "dst_out", size / 2);
            g.insert_memcpy(dst_buf, dst_out);
            let got = run_graph_read_bufs(g, &[dst_out], &ctx).remove(0);
            assert_eq!(
                got.len(),
                (size / 2) * FRAC_EF_BYTES,
                "output byte length mismatch for size={size}"
            );
            assert_eq!(
                &got[..],
                frac_bytes(&want),
                "fold_ef_frac_columns_dsl mismatch at size={size}"
            );
        }
    }

    #[test]
    fn frac_multifold_dsl_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x2FF0_1D5D);

        // The CUDA `_frac_multifold` dispatcher only supports w in [2, 5].
        for (tail_size, w) in [(4usize, 2usize), (8, 2), (8, 3), (16, 3), (4, 4)] {
            let beta_size = 1usize << w;
            let poly_stride = tail_size * beta_size;
            let pre_size = 2 * poly_stride;
            let src = make_host_leaves(pre_size, 0x2FF0 ^ (w * 31 + tail_size) as u64);
            let eq_r_window: Vec<EF> = (0..beta_size).map(|_| rng.random()).collect();
            let alpha: EF = rng.random();

            // Eager reference.
            let src_dev: DeviceBuffer<Frac<EF>> = src.as_slice().to_device_on(&ctx).unwrap();
            let dst_len = 2 * tail_size;
            let dst_dev: DeviceBuffer<Frac<EF>> = DeviceBuffer::with_capacity_on(dst_len, &ctx);
            let eq_dev: DeviceBuffer<EF> = eq_r_window.as_slice().to_device_on(&ctx).unwrap();
            unsafe {
                frac_multifold_raw(
                    src_dev.as_ptr(),
                    dst_dev.as_mut_ptr(),
                    pre_size,
                    pre_size,
                    (pre_size / 2).trailing_zeros() as usize, // rem_n = log2(poly_stride) = log2(tail_size*2^w)
                    w,
                    alpha,
                    eq_dev.as_ptr(),
                    stream,
                )
                .expect("frac_multifold_raw");
            }
            ctx.stream.synchronize().unwrap();
            let want = dst_dev.to_host_on(&ctx).unwrap();

            // DSL side.
            let mut g = GraphBuilder::new();
            let src_buf = frac_const_buf(&mut g, "src", &src);
            let eq_buf = ef_slice_const_buf(&mut g, "eq_r_window", &eq_r_window);
            let dst_buf = add_frac_ef_buf(&mut g, device, "dst", dst_len);
            frac_multifold_ir_dsl(&mut g, src_buf, dst_buf, eq_buf, tail_size, w);
            let dst_out = add_frac_ef_buf(&mut g, device, "dst_out", dst_len);
            g.insert_memcpy(dst_buf, dst_out);
            let got = run_graph_read_bufs(g, &[dst_out], &ctx).remove(0);
            assert_eq!(
                &got[..],
                frac_bytes(&want),
                "frac_multifold_dsl mismatch at tail_size={tail_size}, w={w}"
            );
        }
    }

    #[test]
    fn frac_precompute_m_eval_round_dsl_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_ABCD);

        for (w, t) in [(2usize, 0usize), (2, 1), (3, 0), (3, 1), (3, 2), (4, 2)] {
            let m = 1usize << w;
            let prefix_size = 1usize << t;
            let suffix_size = 1usize << (w - t - 1);
            let m_total: Vec<EF> = (0..m * m).map(|_| rng.random()).collect();
            let eq_r_prefix: Vec<EF> = (0..prefix_size).map(|_| rng.random()).collect();
            let eq_suffix: Vec<EF> = (0..suffix_size).map(|_| rng.random()).collect();

            // Eager reference.
            let m_dev: DeviceBuffer<EF> = m_total.as_slice().to_device_on(&ctx).unwrap();
            let ep_dev: DeviceBuffer<EF> = eq_r_prefix.as_slice().to_device_on(&ctx).unwrap();
            let es_dev: DeviceBuffer<EF> = eq_suffix.as_slice().to_device_on(&ctx).unwrap();
            let out_dev: DeviceBuffer<EF> = DeviceBuffer::with_capacity_on(2, &ctx);
            unsafe {
                frac_precompute_m_eval_round_raw(
                    m_dev.as_ptr(),
                    w,
                    t,
                    ep_dev.as_ptr(),
                    es_dev.as_ptr(),
                    out_dev.as_mut_ptr(),
                    stream,
                )
                .expect("frac_precompute_m_eval_round_raw");
            }
            ctx.stream.synchronize().unwrap();
            let want = out_dev.to_host_on(&ctx).unwrap();

            // DSL side.
            let mut g = GraphBuilder::new();
            let m_buf = ef_slice_const_buf(&mut g, "m_total", &m_total);
            let ep_buf = ef_slice_const_buf(&mut g, "eq_r_prefix", &eq_r_prefix);
            let es_buf = ef_slice_const_buf(&mut g, "eq_suffix", &eq_suffix);
            let out_buf = add_ef_buf(&mut g, device, "out", 2);
            frac_precompute_m_eval_round_ir_dsl(&mut g, m_buf, ep_buf, es_buf, out_buf, w, t);
            let out_out = add_ef_buf(&mut g, device, "out_out", 2);
            g.insert_memcpy(out_buf, out_out);
            let bytes = run_graph_read_bufs(g, &[out_out], &ctx).remove(0);
            let got: Vec<EF> = bytes
                .chunks_exact(size_of::<EF>())
                .map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const EF) })
                .collect();
            assert_eq!(
                got, want,
                "frac_precompute_m_eval_round_dsl mismatch at w={w}, t={t}"
            );
        }
    }

    #[test]
    fn frac_compute_round_dsl_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0xCC00_11FF);

        for n in [3usize, 4, 5] {
            let num_x = 2usize << n;
            let pq_size = 2 * num_x;
            let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let lambda: EF = rng.random();
            let pq = make_host_leaves(pq_size, 0xCC00_11FF ^ n as u64);

            // Eager reference.
            let eq_host = SqrtEqLayers::from_xi(&xi, &ctx).unwrap();
            let low_n = eq_host.low_n();
            let high_n = eq_host.high_n();
            assert_eq!(2 << (low_n + high_n), num_x);
            let eq_low_cap = 1usize << low_n;
            let eq_high_cap = 1usize << high_n;
            let pq_dev: DeviceBuffer<Frac<EF>> = pq.as_slice().to_device_on(&ctx).unwrap();
            let tmp_len = unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) } as usize;
            let mut tmp = DeviceBuffer::<EF>::with_capacity_on(tmp_len, &ctx);
            let mut out_ref = DeviceBuffer::<EF>::with_capacity_on(2, &ctx);
            unsafe {
                frac_compute_round(
                    &eq_host,
                    &pq_dev,
                    num_x,
                    lambda,
                    &mut out_ref,
                    &mut tmp,
                    stream,
                )
                .expect("frac_compute_round");
            }
            ctx.stream.synchronize().unwrap();
            let want = out_ref.to_host_on(&ctx).unwrap();

            // Read eq_low / eq_high back to host so we can pass them as
            // const buffers to the DSL graph.
            let eq_low_host: Vec<EF> = eq_host.low.layers[low_n].to_host_on(&ctx).unwrap();
            let eq_high_host: Vec<EF> = eq_host.high.layers[high_n].to_host_on(&ctx).unwrap();

            // DSL side.
            let mut g = GraphBuilder::new();
            let el_buf = ef_slice_const_buf(&mut g, "eq_low", &eq_low_host);
            let eh_buf = ef_slice_const_buf(&mut g, "eq_high", &eq_high_host);
            let pq_buf = frac_const_buf(&mut g, "pq", &pq);
            let lambda_buf = ef_const_ext_scalar_buf(&mut g, device, "lambda", lambda);
            let out_buf = add_ef_buf(&mut g, device, "out", 2);
            frac_compute_round_ir_dsl(
                &mut g, el_buf, eh_buf, pq_buf, lambda_buf, out_buf, num_x, eq_low_cap,
            );
            let out_out = add_ef_buf(&mut g, device, "out_out", 2);
            g.insert_memcpy(out_buf, out_out);
            let bytes = run_graph_read_bufs(g, &[out_out], &ctx).remove(0);
            let got: Vec<EF> = bytes
                .chunks_exact(size_of::<EF>())
                .map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const EF) })
                .collect();
            assert_eq!(got, want, "frac_compute_round_dsl mismatch at n={n}");
            let _ = eq_high_cap; // silence unused
        }
    }

    #[test]
    fn frac_build_tree_layer_revert_dsl_matches_eager() {
        use crate::cuda::logup_zerocheck::frac_build_tree_layer;
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);

        for layer_size in [2usize, 4, 8, 16, 32] {
            // The revert applies to layer_size/2 rows; use dense
            // logical_len == layer_size, apply_alpha=false.
            let leaves = make_host_leaves(layer_size, 0x7777 ^ layer_size as u64);
            let alpha: EF = EF::from_u32(0); // irrelevant, apply_alpha=false

            // Eager reference: run the CUDA revert kernel in place.
            let mut layer_dev: DeviceBuffer<Frac<EF>> =
                leaves.as_slice().to_device_on(&ctx).unwrap();
            unsafe {
                frac_build_tree_layer(
                    &mut layer_dev,
                    layer_size,
                    layer_size,
                    true,
                    alpha,
                    false,
                    stream,
                )
                .expect("frac_build_tree_layer(revert)");
            }
            ctx.stream.synchronize().unwrap();
            let want = layer_dev.to_host_on(&ctx).unwrap();

            // DSL side (out-of-place).
            let mut g = GraphBuilder::new();
            let src = frac_const_buf(&mut g, "layer_in", &leaves);
            let dst = add_frac_ef_buf(&mut g, device, "layer_out", layer_size);
            frac_build_tree_layer_revert_ir_dsl(&mut g, src, dst, layer_size);
            let dst_out = add_frac_ef_buf(&mut g, device, "dst_out", layer_size);
            g.insert_memcpy(dst, dst_out);
            let got = run_graph_read_bufs(g, &[dst_out], &ctx).remove(0);
            assert_eq!(
                &got[..],
                frac_bytes(&want),
                "frac_build_tree_layer_revert_dsl mismatch at layer_size={layer_size}"
            );
        }
    }

    #[test]
    fn frac_compute_round_and_fold_dsl_matches_eager() {
        use crate::cuda::logup_zerocheck::frac_compute_round_and_fold;
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x4444_5555);

        for n in [3usize, 4] {
            let num_x = 2usize << n;
            let src_pq_size = 4 * num_x;
            let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let lambda: EF = rng.random();
            let r_prev: EF = rng.random();
            let alpha: EF = rng.random();
            let src_pq = make_host_leaves(src_pq_size, 0x4444 ^ n as u64);

            // Eager reference.
            let eq_host = SqrtEqLayers::from_xi(&xi, &ctx).unwrap();
            let low_n = eq_host.low_n();
            let high_n = eq_host.high_n();
            let eq_low_cap = 1usize << low_n;
            let src_pq_dev: DeviceBuffer<Frac<EF>> = src_pq.as_slice().to_device_on(&ctx).unwrap();
            let mut dst_pq_dev: DeviceBuffer<Frac<EF>> =
                DeviceBuffer::with_capacity_on(src_pq_size / 2, &ctx);
            let tmp_len = unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) } as usize;
            let mut tmp = DeviceBuffer::<EF>::with_capacity_on(tmp_len, &ctx);
            let mut out_ref = DeviceBuffer::<EF>::with_capacity_on(2, &ctx);
            unsafe {
                frac_compute_round_and_fold(
                    &eq_host,
                    &src_pq_dev,
                    &mut dst_pq_dev,
                    src_pq_size,
                    src_pq_size,
                    src_pq_size,
                    lambda,
                    r_prev,
                    alpha,
                    &mut out_ref,
                    &mut tmp,
                    stream,
                )
                .expect("frac_compute_round_and_fold");
            }
            ctx.stream.synchronize().unwrap();
            let want_out = out_ref.to_host_on(&ctx).unwrap();
            let want_dst = dst_pq_dev.to_host_on(&ctx).unwrap();
            let eq_low_host: Vec<EF> = eq_host.low.layers[low_n].to_host_on(&ctx).unwrap();
            let eq_high_host: Vec<EF> = eq_host.high.layers[high_n].to_host_on(&ctx).unwrap();

            // DSL side (fold module + compute-round module composition).
            let mut g = GraphBuilder::new();
            let el_buf = ef_slice_const_buf(&mut g, "eq_low", &eq_low_host);
            let eh_buf = ef_slice_const_buf(&mut g, "eq_high", &eq_high_host);
            let src_buf = frac_const_buf(&mut g, "src_pq", &src_pq);
            let dst_buf = add_frac_ef_buf(&mut g, device, "dst_pq", src_pq_size / 2);
            let lambda_buf = ef_const_ext_scalar_buf(&mut g, device, "lambda", lambda);
            let r_prev_buf = ef_const_ext_scalar_buf(&mut g, device, "r_prev", r_prev);
            let out_buf = add_ef_buf(&mut g, device, "out", 2);
            frac_compute_round_and_fold_ir_dsl(
                &mut g,
                el_buf,
                eh_buf,
                src_buf,
                dst_buf,
                lambda_buf,
                r_prev_buf,
                out_buf,
                src_pq_size,
                eq_low_cap,
            );
            let out_out = add_ef_buf(&mut g, device, "out_out", 2);
            g.insert_memcpy(out_buf, out_out);
            let dst_out = add_frac_ef_buf(&mut g, device, "dst_out", src_pq_size / 2);
            g.insert_memcpy(dst_buf, dst_out);
            let bytes = run_graph_read_bufs(g, &[out_out, dst_out], &ctx);
            let got_out: Vec<EF> = bytes[0]
                .chunks_exact(size_of::<EF>())
                .map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const EF) })
                .collect();
            let got_dst = &bytes[1];
            assert_eq!(
                got_out, want_out,
                "compute_round_and_fold out mismatch at n={n}"
            );
            assert_eq!(
                &got_dst[..],
                frac_bytes(&want_dst),
                "compute_round_and_fold dst mismatch at n={n}"
            );
        }
    }

    #[test]
    fn frac_compute_round_and_revert_dsl_matches_eager() {
        use crate::cuda::logup_zerocheck::frac_compute_round_and_revert;
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x8888_9999);

        for n in [3usize, 4] {
            let num_x = 2usize << n;
            let layer_size = 2 * num_x;
            let xi: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let lambda: EF = rng.random();
            let alpha: EF = EF::from_u32(0); // dense
            let leaves = make_host_leaves(layer_size, 0x8888 ^ n as u64);

            // Eager reference.
            let eq_host = SqrtEqLayers::from_xi(&xi, &ctx).unwrap();
            let low_n = eq_host.low_n();
            let high_n = eq_host.high_n();
            let eq_low_cap = 1usize << low_n;
            let mut layer_dev: DeviceBuffer<Frac<EF>> =
                leaves.as_slice().to_device_on(&ctx).unwrap();
            let tmp_len = unsafe { _frac_compute_round_temp_buffer_size(num_x as u32) } as usize;
            let mut tmp = DeviceBuffer::<EF>::with_capacity_on(tmp_len, &ctx);
            let mut out_ref = DeviceBuffer::<EF>::with_capacity_on(2, &ctx);
            unsafe {
                frac_compute_round_and_revert(
                    &eq_host,
                    &mut layer_dev,
                    num_x,
                    layer_size,
                    lambda,
                    alpha,
                    &mut out_ref,
                    &mut tmp,
                    stream,
                )
                .expect("frac_compute_round_and_revert");
            }
            ctx.stream.synchronize().unwrap();
            let want_out = out_ref.to_host_on(&ctx).unwrap();
            let want_layer = layer_dev.to_host_on(&ctx).unwrap();
            let eq_low_host: Vec<EF> = eq_host.low.layers[low_n].to_host_on(&ctx).unwrap();
            let eq_high_host: Vec<EF> = eq_host.high.layers[high_n].to_host_on(&ctx).unwrap();

            // DSL side (revert module + compute-round module composition).
            let mut g = GraphBuilder::new();
            let el_buf = ef_slice_const_buf(&mut g, "eq_low", &eq_low_host);
            let eh_buf = ef_slice_const_buf(&mut g, "eq_high", &eq_high_host);
            let layer_in_buf = frac_const_buf(&mut g, "layer_in", &leaves);
            let layer_post_buf = add_frac_ef_buf(&mut g, device, "layer_post", layer_size);
            let lambda_buf = ef_const_ext_scalar_buf(&mut g, device, "lambda", lambda);
            let out_buf = add_ef_buf(&mut g, device, "out", 2);
            frac_compute_round_and_revert_ir_dsl(
                &mut g,
                el_buf,
                eh_buf,
                layer_in_buf,
                layer_post_buf,
                lambda_buf,
                out_buf,
                layer_size,
                eq_low_cap,
            );
            let out_out = add_ef_buf(&mut g, device, "out_out", 2);
            g.insert_memcpy(out_buf, out_out);
            let layer_out = add_frac_ef_buf(&mut g, device, "layer_out", layer_size);
            g.insert_memcpy(layer_post_buf, layer_out);
            let bytes = run_graph_read_bufs(g, &[out_out, layer_out], &ctx);
            let got_out: Vec<EF> = bytes[0]
                .chunks_exact(size_of::<EF>())
                .map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const EF) })
                .collect();
            let got_layer = &bytes[1];
            assert_eq!(
                got_out, want_out,
                "compute_round_and_revert out mismatch at n={n}"
            );
            // Only the first-half of the layer is reverted; both should
            // match in the first half. The second half is unchanged.
            assert_eq!(
                &got_layer[..],
                frac_bytes(&want_layer),
                "compute_round_and_revert layer mismatch at n={n}"
            );
        }
    }

    #[test]
    fn frac_build_tree_two_layers_dsl_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x2222_3333);

        for half_i1 in [2usize, 4, 8, 16] {
            let layer_size = 4 * half_i1;
            let leaves = make_host_leaves(layer_size, 0x1111 ^ half_i1 as u64);
            let alpha: EF = rng.random();

            // Eager reference: run the CUDA kernel dense.
            let mut layer_dev: DeviceBuffer<Frac<EF>> =
                leaves.as_slice().to_device_on(&ctx).unwrap();
            unsafe {
                frac_build_tree_two_layers(&mut layer_dev, half_i1, layer_size, alpha, stream)
                    .expect("frac_build_tree_two_layers");
            }
            ctx.stream.synchronize().unwrap();
            let want = layer_dev.to_host_on(&ctx).unwrap();

            // DSL side (out-of-place).
            let mut g = GraphBuilder::new();
            let src = frac_const_buf(&mut g, "layer_in", &leaves);
            let dst = add_frac_ef_buf(&mut g, device, "layer_out", layer_size);
            frac_build_tree_two_layers_ir_dsl(&mut g, src, dst, half_i1);
            let dst_out = add_frac_ef_buf(&mut g, device, "dst_out", layer_size);
            g.insert_memcpy(dst, dst_out);
            let got = run_graph_read_bufs(g, &[dst_out], &ctx).remove(0);
            assert_eq!(
                &got[..],
                frac_bytes(&want),
                "frac_build_tree_two_layers_dsl mismatch at half_i1={half_i1}"
            );
        }
    }
}
