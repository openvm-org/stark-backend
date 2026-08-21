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
//! # Alpha as an input buffer
//!
//! `alpha` is bound like any other challenge: a `[D_EF] BabyBear` input
//! buffer lifted to an `FpExt` scalar inside the module (see
//! [`bind_challenge_as_fpext`]), so the module hash — and hence the JIT
//! cache — is stable across alphas. (In the dense case the modules that
//! don't consume padding never touch `alpha` at all.)
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
    ir::{IRBuilder, Module, NodeId, ScalarType, SizeExpr},
    passes::parallel_reduce_rewrite::reduce_lowers_multi_stage,
};

use super::fractional_ir::{add_ef_buf, fpext_from_coeffs, load_ext_coeffs, FRAC_EF_BYTES};
use crate::{logup_zerocheck::fractional_ir::GKR_S_DEG, types::D_EF};

// ---------------------------------------------------------------------------
// Small helpers.

/// Bind a `[D_EF]`-shaped BabyBear challenge input, lift it to an `FpExt`
/// scalar, and `let_bound` it so the recombine fires once per launch
/// rather than once per compute thread.
pub(crate) fn bind_challenge_as_fpext(b: &mut IRBuilder, name: &str) -> NodeId {
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
///   - `src : [q*4, 2] FpExt` (a Frac<EF> buffer, `size = q*4` elements)
///   - `r   : [D_EF] BabyBear` (folding challenge)
///
/// Output:
///   - `dst : [q*2, 2] FpExt` (a Frac<EF> buffer of half the length)
///
/// Fully symbolic over `q = size / 4`: `q` is inferred from the bound
/// `src` buffer at [`GraphBuilder::insert_kernel`] and survives as a
/// runtime parameter, so every fold size shares ONE compiled kernel.
///
/// Dense-only (`real_len == logical_len == size`).
pub fn build_fold_ef_frac_columns_module() -> Module {
    let mut b = IRBuilder::new();
    let q = b.symbol("q");
    let src = b.input(
        "src",
        ScalarType::FpExt,
        vec![SizeExpr::from(q * 4), 2usize.into()],
    );
    let r = bind_challenge_as_fpext(&mut b, "r");

    let body = b.compute(q * 2, move |b, i| {
        // Determine the two source indices for output row `i`:
        //   a_idx = i + (i / quarter) * quarter
        //   b_idx = a_idx + quarter
        // For `i in [0, quarter)`: a_idx = i,           b_idx = i + quarter
        // For `i in [quarter, 2q)`: a_idx = i + quarter, b_idx = i + 2*quarter
        let quarter_c = b.const_sym(q);
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
    b.finish("fold_ef_frac_columns_dsl", body)
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
    assert!(
        size >= 4 && size.is_power_of_two(),
        "fold: size must be a power of two >= 4, got {size}"
    );
    assert_frac_size(size);
    g.insert_kernel(build_fold_ef_frac_columns_module(), [src, r], [dst], &[]);
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
        &[],
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
        &[],
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
///   - `eq_low     : [c] FpExt` (symbolic `c` = eq_low_cap)
///   - `eq_high    : [h] FpExt` (symbolic `h` = num_x / 2 / eq_low_cap)
///   - `pq_buffer  : [2*num_x, 2] FpExt`
///   - `lambda     : [D_EF] BabyBear`
///
/// Output:
///   - `out : [2] FpExt` — the block-summed `(s'(1), s'(2))` pair.
///
/// Dense-only (`real_len == logical_len == 2*num_x`). The eq-buffer split
/// point is symbolic (both caps bind from the eq input shapes at
/// `insert_kernel`), so every `(eq_low_cap, eq_high_cap)` partition of a
/// given `num_x` shares one JIT'd kernel. `num_x` itself must stay
/// concrete: it is the inner `reduce_add` bound.
pub fn build_frac_compute_round_module(num_x: usize) -> Module {
    assert!(num_x.is_power_of_two() && num_x >= 2);
    let pq_size = 2 * num_x;
    let iter = num_x / 2;
    let mut b = IRBuilder::new();
    let c = b.symbol("c");
    let h = b.symbol("h");
    let eq_low = b.input("eq_low", ScalarType::FpExt, vec![SizeExpr::from(c)]);
    let eq_high = b.input("eq_high", ScalarType::FpExt, vec![SizeExpr::from(h)]);
    let pq = b.input("pq", ScalarType::FpExt, vec![pq_size, 2]);
    let lambda = bind_challenge_as_fpext(&mut b, "lambda");

    let body = b.compute(GKR_S_DEG - 1, move |b, out_i| {
        b.reduce_add(iter, move |b, idx| {
            compute_round_term(b, c, eq_low, eq_high, pq, lambda, pq_size, out_i, idx)
        })
    });
    b.finish(format!("frac_compute_round_dsl_n{num_x}"), body)
}

/// Per-`(out_i, idx)` term of the compute-round reduction: loads the eq
/// pair (`c` = symbolic eq_low_cap) and the four pq slots for `idx` and
/// returns `eq_val * contrib(out_i)`.
#[allow(clippy::too_many_arguments)]
fn compute_round_term(
    b: &mut IRBuilder,
    c: crypto_compiler::ir::Sym,
    eq_low: NodeId,
    eq_high: NodeId,
    pq: NodeId,
    lambda: NodeId,
    pq_size: usize,
    out_i: NodeId,
    idx: NodeId,
) -> NodeId {
    // eq_val = eq_low[idx % eq_low_cap] * eq_high[idx / eq_low_cap]
    let low_c = b.const_sym(c);
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
}

/// Stage-0 block-sums variant of [`build_frac_compute_round_module`] for
/// large `num_x`: each output row's `num_x / 2` reduction is split into
/// `g_blocks` contiguous chunks summed independently.
///
/// Output: `[(GKR_S_DEG - 1) * g_blocks] FpExt` partials, row-major
/// (`out_i * g_blocks + blk`). A follow-up [`build_ef_rowsum_module`]
/// collapses each row to the final `(s'(1), s'(2))` pair. Both stages stay
/// single-kernel under the JIT parallel-reduce rewrite, which the
/// single-module form would not (its full-length reduce lowers to two
/// internal kernels with module-level scratch — unsupported in graphs).
pub fn build_frac_compute_round_block_module(num_x: usize, g_blocks: usize) -> Module {
    assert!(num_x.is_power_of_two() && num_x >= 2);
    let pq_size = 2 * num_x;
    let iter = num_x / 2;
    assert!(
        g_blocks.is_power_of_two() && g_blocks >= 2 && g_blocks <= iter,
        "g_blocks must be a power of two in [2, num_x/2], got {g_blocks}"
    );
    let chunk = iter / g_blocks;
    let rows = GKR_S_DEG - 1;
    let mut b = IRBuilder::new();
    let c = b.symbol("c");
    let h = b.symbol("h");
    let eq_low = b.input("eq_low", ScalarType::FpExt, vec![SizeExpr::from(c)]);
    let eq_high = b.input("eq_high", ScalarType::FpExt, vec![SizeExpr::from(h)]);
    let pq = b.input("pq", ScalarType::FpExt, vec![pq_size, 2]);
    let lambda = bind_challenge_as_fpext(&mut b, "lambda");

    let body = b.compute(rows * g_blocks, move |b, j| {
        let g_c = b.const_u32(g_blocks as u32);
        let out_i = b.div(j, g_c);
        let blk = b.rem(j, g_c);
        let chunk_c = b.const_u32(chunk as u32);
        let base = b.mul(blk, chunk_c);
        b.reduce_add(chunk, move |b, cc| {
            let idx = b.add(base, cc);
            compute_round_term(b, c, eq_low, eq_high, pq, lambda, pq_size, out_i, idx)
        })
    });
    b.finish(
        format!("frac_compute_round_block_dsl_n{num_x}_g{g_blocks}"),
        body,
    )
}

/// Row-sum module: `out[i] = Σ_k input[i, k]` over FpExt.
pub fn build_ef_rowsum_module(rows: usize, k: usize) -> Module {
    assert!(rows >= 1 && k >= 1);
    let mut b = IRBuilder::new();
    let input = b.input("rows_in", ScalarType::FpExt, vec![rows, k]);
    let body = b.compute(rows, move |b, i| {
        b.reduce_add(k, move |b, kk| b.index(input, &[i, kk]))
    });
    b.finish(format!("ef_rowsum_dsl_r{rows}_k{k}"), body)
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
    let iter = num_x / 2;
    assert!(eq_low_cap.is_power_of_two());
    assert!(
        (iter / eq_low_cap).is_power_of_two() && (iter / eq_low_cap) * eq_low_cap == iter,
        "eq_low_cap must partition num_x/2"
    );
    let rows = GKR_S_DEG - 1;
    if !reduce_lowers_multi_stage(iter, rows) {
        g.insert_kernel(
            build_frac_compute_round_module(num_x),
            [eq_low, eq_high, pq_buffer, lambda],
            [out],
            &[],
        );
        return;
    }
    // Large reduce: the single-module form would JIT-lower to two internal
    // kernels with module-level scratch (unsupported in graphs), so
    // decompose at the graph level: block sums into a partials buffer,
    // then a row-sum. `g_blocks` mirrors the block-reduce heuristics —
    // ~256 items per block, capped at 64 so stage 0's `2 * g_blocks` outer
    // parallelism stays under the rewrite's saturation threshold (above it
    // the reduce would lower sequentially on too few threads).
    let g_blocks = (iter / 256).min(64);
    debug_assert!(!reduce_lowers_multi_stage(iter / g_blocks, rows * g_blocks));
    debug_assert!(!reduce_lowers_multi_stage(g_blocks, rows));
    let device = g.buf_info(out).device_type;
    let partials = add_ef_buf(
        g,
        device,
        &format!("frac_cr_partials_n{num_x}"),
        rows * g_blocks,
    );
    g.insert_kernel(
        build_frac_compute_round_block_module(num_x, g_blocks),
        [eq_low, eq_high, pq_buffer, lambda],
        [partials],
        &[],
    );
    g.insert_kernel(
        build_ef_rowsum_module(rows, g_blocks),
        [partials],
        [out],
        &[],
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
        &[],
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
        &[],
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
// Small-round overlap kernels (challenge-free precompute-M).
//
// Design: crates/compiler/gkr-small-round-overlap-plan.md. For small outer
// GKR rounds (pq_size = 2^(R+1) <= SWIRL_CUDA_GKR_SMALL_M_MAX_PQ) the
// overlap driver restructures precompute-M so the [revert + M-build] stage
// has no transcript inputs: full-round window (w = R, no eq tail), base = 0
// (no pending fold), and lambda split out of the build as
// `M = M_a + lambda * M_b` (pure outer products). The graph scheduler can
// then run round R+1's build chain concurrently with round R's transcript
// chain.

/// Build the DSL module for a two-input out-of-place tree-layer revert.
///
/// Inputs:
///   - `parents  : [half, 2] FpExt` — pq(R-1), the reverted previous layer (a side buffer, NOT the
///     tree).
///   - `layer_in : [layer_rows, 2, D_EF] BabyBear` — the whole segment-tree buffer (concrete row
///     count — inner passes such as layout inference need concrete global shapes); only rows
///     `[half, 2*half)` (the stored right children for this level) are read. BabyBear binding
///     because [`ef_inverse_coeffs`] needs the base-field coefficients of `rhs.q`.
///
/// Output:
///   - `pq_out : [2*half, 2] FpExt` — pq(R) = [left children | right children]:
///     - `out[k] = frac_unadd(parents[k], layer[half+k])` for `k < half`
///     - `out[k] = layer[k]` for `k >= half`
///
/// where `frac_unadd(lhs, rhs) = (p, q)` with `q = lhs.q * inv(rhs.q)` and
/// `p = (lhs.p - q * rhs.p) * inv(rhs.q)` (same math as
/// [`build_frac_build_tree_layer_revert_module`], parents externalized).
///
/// Dense-only.
pub fn build_frac_tree_revert_two_input_module(half: usize, layer_rows: usize) -> Module {
    assert!(
        half >= 2 && half.is_power_of_two(),
        "revert two-input module: half must be a power of two >= 2, got {half}"
    );
    assert!(
        layer_rows >= 2 * half,
        "revert two-input module: layer_rows {layer_rows} must cover rows [half, 2*half) for \
         half={half}"
    );
    let out_len = 2 * half;
    let mut b = IRBuilder::new();
    let parents = b.input("parents", ScalarType::FpExt, vec![half, 2]);
    let layer = b.input("layer_in", ScalarType::BabyBear, vec![layer_rows, 2, D_EF]);

    let body = b.compute(out_len, move |b, k| {
        let half_c = b.const_u32(half as u32);
        let is_first_half = b.lt(k, half_c);

        // parent_row = k mod half: the intended parent for k < half; an
        // in-bounds clamp (result discarded via select) for k >= half.
        let parent_row = b.rem(k, half_c);
        // rhs_row = half + (k mod half). For k < half this is the stored
        // right child `half + k`; for k >= half it equals `k`, which is
        // exactly the row the copy branch forwards — both branches read
        // one shared row.
        let rhs_row = b.add(half_c, parent_row);

        let zero_c = b.const_u32(0);
        let one_c = b.const_u32(1);
        let par_p = b.index(parents, &[parent_row, zero_c]);
        let par_q = b.index(parents, &[parent_row, one_c]);

        let rhs_p_coeffs: [NodeId; D_EF] = std::array::from_fn(|c| {
            let ci = b.const_u32(c as u32);
            b.index(layer, &[rhs_row, zero_c, ci])
        });
        let rhs_q_coeffs: [NodeId; D_EF] = std::array::from_fn(|c| {
            let ci = b.const_u32(c as u32);
            b.index(layer, &[rhs_row, one_c, ci])
        });
        let rhs_q_inv_coeffs = ef_inverse_coeffs(b, rhs_q_coeffs);
        let rhs_p = fpext_from_coeffs(b, rhs_p_coeffs);
        let rhs_q = fpext_from_coeffs(b, rhs_q_coeffs);
        let rhs_q_inv = fpext_from_coeffs(b, rhs_q_inv_coeffs);

        let new_q = b.mul(par_q, rhs_q_inv);
        let nq_rp = b.mul(new_q, rhs_p);
        let inner = b.sub(par_p, nq_rp);
        let new_p = b.mul(inner, rhs_q_inv);

        let out_p = b.select(is_first_half, new_p, rhs_p);
        let out_q = b.select(is_first_half, new_q, rhs_q);
        b.pack(&[out_p, out_q])
    });
    b.finish(
        format!("frac_tree_revert_two_input_dsl_h{half}_l{layer_rows}"),
        body,
    )
}

/// Insert a two-input out-of-place tree-layer revert as a structured
/// kernel: `pq_out` (2*half Fracs) from `parents` (half Fracs) and the
/// stored right children at `layer_in[half..2*half)`. `layer_rows` is
/// the full `Frac<EF>` row count of `layer_in`.
pub fn frac_tree_revert_two_input_ir_dsl(
    g: &mut GraphBuilder,
    parents: BufId,
    layer_in: BufId,
    pq_out: BufId,
    half: usize,
    layer_rows: usize,
) {
    g.insert_kernel(
        build_frac_tree_revert_two_input_module(half, layer_rows),
        [parents, layer_in],
        [pq_out],
        &[],
    );
}

/// Build the DSL module for the challenge-free M build (full-round
/// window, empty tail): pure outer products of the reverted pq layer.
///
/// Input:
///   - `pq : [2*m, 2] FpExt` — pq(R), `m = 2^R`; poly 0 = rows `[0, m)`, poly 1 = rows `[m, 2m)`.
///
/// Output:
///   - `m_ab : [m*m, 2] FpExt`, row `u*m + v`:
///     - col 0: `M_a[u,v] = p0[u]*q1[v] + p1[u]*q0[v]`
///     - col 1: `M_b[u,v] = q0[u]*q1[v]`
///
/// so that `M = M_a + lambda * M_b` equals the eager
/// `M[u,v] = (p0[u] + lambda*q0[u])*q1[v] + p1[u]*q0[v]` with the tail
/// reduction empty (`eq(b, z_b)` weight = 1). Dense-only.
pub fn build_frac_m_outer_product_module(m: usize) -> Module {
    assert!(
        m >= 2 && m.is_power_of_two(),
        "m outer-product module: m must be a power of two >= 2, got {m}"
    );
    let mut b = IRBuilder::new();
    let pq = b.input("pq", ScalarType::FpExt, vec![2 * m, 2]);
    let body = b.compute(m * m, move |b, i| {
        let m_c = b.const_u32(m as u32);
        let u = b.div(i, m_c);
        let v = b.rem(i, m_c);
        let mu = b.add(u, m_c);
        let mv = b.add(v, m_c);
        let zero_c = b.const_u32(0);
        let one_c = b.const_u32(1);
        let p0u = b.index(pq, &[u, zero_c]);
        let q0u = b.index(pq, &[u, one_c]);
        let p1u = b.index(pq, &[mu, zero_c]);
        let q0v = b.index(pq, &[v, one_c]);
        let q1v = b.index(pq, &[mv, one_c]);
        let a1 = b.mul(p0u, q1v);
        let a2 = b.mul(p1u, q0v);
        let m_a = b.add(a1, a2);
        let m_b = b.mul(q0u, q1v);
        b.pack(&[m_a, m_b])
    });
    b.finish(format!("frac_m_outer_product_dsl_{m}"), body)
}

/// Insert the challenge-free M build as a structured kernel.
pub fn frac_m_outer_product_ir_dsl(g: &mut GraphBuilder, pq: BufId, m_ab: BufId, m: usize) {
    g.insert_kernel(build_frac_m_outer_product_module(m), [pq], [m_ab], &[]);
}

/// Per-idx contribution for the lambda-combined eval round: identical
/// index math to [`build_frac_precompute_m_eval_round_module`], but each
/// M load is reconstituted as `m = m_a + lambda * m_b` from the
/// interleaved `m_ab` buffer.
#[allow(clippy::too_many_arguments)]
fn m_eval_lambda_term(
    b: &mut IRBuilder,
    m_ab: NodeId,
    lambda: NodeId,
    eq_r_prefix: NodeId,
    eq_suffix: NodeId,
    w: usize,
    t: usize,
    out_i: NodeId,
    idx: NodeId,
) -> NodeId {
    let m = 1usize << w;
    let prefix_size = 1usize << t;
    let suffix_bits = w - t - 1;
    let suffix_size = 1usize << suffix_bits;
    let cur_bit = 1usize << suffix_bits;
    let prefix_shift = suffix_bits + 1;

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

    let ps = b.const_u32(1u32 << prefix_shift);
    let cur_bit_c = b.const_u32(cur_bit as u32);
    let b1_shift = b.mul(b1, ps);
    let b2_shift = b.mul(b2, ps);
    let beta1_0 = b.add(b1_shift, suffix);
    let beta1_1 = b.add(beta1_0, cur_bit_c);
    let beta2_0 = b.add(b2_shift, suffix);
    let beta2_1 = b.add(beta2_0, cur_bit_c);

    let m_c = b.const_u32(m as u32);
    let zero_c = b.const_u32(0);
    let one_c = b.const_u32(1);
    let load = |b: &mut IRBuilder, row: NodeId, col: NodeId| -> NodeId {
        let rm = b.mul(row, m_c);
        let flat = b.add(rm, col);
        let ma = b.index(m_ab, &[flat, zero_c]);
        let mb = b.index(m_ab, &[flat, one_c]);
        let lmb = b.mul(lambda, mb);
        b.add(ma, lmb)
    };
    let m00 = load(b, beta1_0, beta2_0);
    let m01 = load(b, beta1_0, beta2_1);
    let m10 = load(b, beta1_1, beta2_0);
    let m11 = load(b, beta1_1, beta2_1);

    // s1 branch: weight * m11
    // s2 branch: weight * (m00 - 2*m01 - 2*m10 + 4*m11)
    let two_e = b.const_fpext([2, 0, 0, 0]);
    let sum_val_s1 = m11;
    let s2_01 = b.add(m01, m10);
    let s2_01_2 = b.mul(two_e, s2_01);
    let s2_11 = b.add(m11, m11);
    let s2_inner1 = b.sub(s2_01_2, s2_11);
    let s2_inner2 = b.sub(s2_inner1, s2_11);
    let sum_val_s2 = b.sub(m00, s2_inner2);

    let zero_u = b.const_u32(0);
    let is_s1 = b.eq(out_i, zero_u);
    let val = b.select(is_s1, sum_val_s1, sum_val_s2);
    b.mul(weight, val)
}

/// Build the DSL module for the lambda-combined
/// `frac_precompute_m_eval_round` (single-stage reduce form).
///
/// Inputs:
///   - `m_ab        : [m*m, 2] FpExt` where `m = 1 << w` (col 0 = M_a, col 1 = M_b)
///   - `eq_r_prefix : [1 << t] FpExt`
///   - `eq_suffix   : [1 << (w - t - 1)] FpExt`
///   - `lambda      : [D_EF] BabyBear`
///
/// Output:
///   - `out : [2] FpExt` (s'(1), s'(2))
pub fn build_frac_precompute_m_eval_lambda_module(w: usize, t: usize) -> Module {
    assert!(w >= 1, "m_eval_lambda: w must be >= 1, got {w}");
    assert!(t < w, "m_eval_lambda: t must be < w (got t={t}, w={w})");
    let m = 1usize << w;
    let total = 1usize << (t + w - 1);
    let mut b = IRBuilder::new();
    let m_ab = b.input("m_ab", ScalarType::FpExt, vec![m * m, 2]);
    let eq_r_prefix = b.input("eq_r_prefix", ScalarType::FpExt, vec![1 << t]);
    let eq_suffix = b.input("eq_suffix", ScalarType::FpExt, vec![1 << (w - t - 1)]);
    let lambda = bind_challenge_as_fpext(&mut b, "lambda");

    let body = b.compute(2, move |b, out_i| {
        b.reduce_add(total, move |b, idx| {
            m_eval_lambda_term(b, m_ab, lambda, eq_r_prefix, eq_suffix, w, t, out_i, idx)
        })
    });
    b.finish(format!("frac_precompute_m_eval_lambda_dsl_w{w}_t{t}"), body)
}

/// Block-partials variant of the lambda-combined eval round, for reduce
/// domains large enough that the single-module form would lower
/// multi-stage. Output: `[2 * g_blocks] FpExt` partials, row-major
/// (`out_i * g_blocks + blk`); collapse with [`build_ef_rowsum_module`].
pub fn build_frac_precompute_m_eval_lambda_block_module(
    w: usize,
    t: usize,
    g_blocks: usize,
) -> Module {
    assert!(w >= 1 && t < w);
    let m = 1usize << w;
    let total = 1usize << (t + w - 1);
    assert!(
        g_blocks.is_power_of_two() && g_blocks >= 2 && g_blocks <= total,
        "g_blocks must be a power of two in [2, total], got {g_blocks}"
    );
    let chunk = total / g_blocks;
    let rows = GKR_S_DEG - 1;
    let mut b = IRBuilder::new();
    let m_ab = b.input("m_ab", ScalarType::FpExt, vec![m * m, 2]);
    let eq_r_prefix = b.input("eq_r_prefix", ScalarType::FpExt, vec![1 << t]);
    let eq_suffix = b.input("eq_suffix", ScalarType::FpExt, vec![1 << (w - t - 1)]);
    let lambda = bind_challenge_as_fpext(&mut b, "lambda");

    let body = b.compute(rows * g_blocks, move |b, j| {
        let g_c = b.const_u32(g_blocks as u32);
        let out_i = b.div(j, g_c);
        let blk = b.rem(j, g_c);
        let chunk_c = b.const_u32(chunk as u32);
        let base = b.mul(blk, chunk_c);
        b.reduce_add(chunk, move |b, cc| {
            let idx = b.add(base, cc);
            m_eval_lambda_term(b, m_ab, lambda, eq_r_prefix, eq_suffix, w, t, out_i, idx)
        })
    });
    b.finish(
        format!("frac_precompute_m_eval_lambda_block_dsl_w{w}_t{t}_g{g_blocks}"),
        body,
    )
}

/// Insert the lambda-combined eval round as structured kernel(s),
/// decomposing into block partials + row-sum when the reduce domain
/// (`2^(t + w - 1)`) would lower multi-stage.
#[allow(clippy::too_many_arguments)]
pub fn frac_precompute_m_eval_lambda_ir_dsl(
    g: &mut GraphBuilder,
    m_ab: BufId,
    lambda: BufId,
    eq_r_prefix: BufId,
    eq_suffix: BufId,
    out: BufId,
    w: usize,
    t: usize,
) {
    let total = 1usize << (t + w - 1);
    let rows = GKR_S_DEG - 1;
    if !reduce_lowers_multi_stage(total, rows) {
        g.insert_kernel(
            build_frac_precompute_m_eval_lambda_module(w, t),
            [m_ab, eq_r_prefix, eq_suffix, lambda],
            [out],
            &[],
        );
        return;
    }
    // Same decomposition heuristics as `frac_compute_round_ir_dsl`.
    let g_blocks = (total / 256).min(64);
    debug_assert!(!reduce_lowers_multi_stage(
        total / g_blocks,
        rows * g_blocks
    ));
    debug_assert!(!reduce_lowers_multi_stage(g_blocks, rows));
    let device = g.buf_info(out).device_type;
    let partials = add_ef_buf(
        g,
        device,
        &format!("frac_mel_partials_w{w}_t{t}"),
        rows * g_blocks,
    );
    g.insert_kernel(
        build_frac_precompute_m_eval_lambda_block_module(w, t, g_blocks),
        [m_ab, eq_r_prefix, eq_suffix, lambda],
        [partials],
        &[],
    );
    g.insert_kernel(
        build_ef_rowsum_module(rows, g_blocks),
        [partials],
        [out],
        &[],
    );
}

/// Build the DSL module for the final claims contraction of a small
/// round: `claim_alpha = sum_u eq_r[u] * poly_alpha[u]` for the four
/// (poly, p/q) combinations. Exact fold-reordering equivalent of the
/// eager prover's R sequential folds followed by reads at 0 and
/// pq_size/2.
///
/// Inputs:
///   - `pq   : [2*m, 2] FpExt` — pq(R) (unfolded)
///   - `eq_r : [m] FpExt` — eq table over this round's window challenges (big-endian: r_0 owns the
///     top within-poly bit)
///
/// Output:
///   - `claims : [4] FpExt` = (p0, q0, p1, q1) — byte-identical to a 2-element Frac<EF> buffer, so
///     `extract_claim_pair_ir(g, claims, 2, 1, ..)` reads (claim at 0, claim at 1).
pub fn build_frac_claims_fold_module(m: usize) -> Module {
    assert!(
        m >= 2 && m.is_power_of_two(),
        "claims fold module: m must be a power of two >= 2, got {m}"
    );
    let mut b = IRBuilder::new();
    let pq = b.input("pq", ScalarType::FpExt, vec![2 * m, 2]);
    let eq_r = b.input("eq_r", ScalarType::FpExt, vec![m]);
    let body = b.compute(4, move |b, out_i| {
        let two_c = b.const_u32(2);
        let poly = b.div(out_i, two_c);
        let col = b.rem(out_i, two_c);
        let m_c = b.const_u32(m as u32);
        let poly_off = b.mul(poly, m_c);
        b.reduce_add(m, move |b, u| {
            let row = b.add(poly_off, u);
            let v = b.index(pq, &[row, col]);
            let e = b.index(eq_r, &[u]);
            b.mul(e, v)
        })
    });
    b.finish(format!("frac_claims_fold_dsl_{m}"), body)
}

/// Insert the claims contraction as a structured kernel. `claims_out`
/// must be a 2-element Frac<EF> buffer (64 bytes). With 4 output rows the
/// reduce never lowers multi-stage (outer parallelism 4 saturates the
/// single-stage heuristic), asserted below.
pub fn frac_claims_fold_ir_dsl(
    g: &mut GraphBuilder,
    pq: BufId,
    eq_r: BufId,
    claims_out: BufId,
    m: usize,
) {
    assert!(
        !reduce_lowers_multi_stage(m, 4),
        "claims fold reduce unexpectedly lowers multi-stage for m={m}"
    );
    g.insert_kernel(
        build_frac_claims_fold_module(m),
        [pq, eq_r],
        [claims_out],
        &[],
    );
}

// ---------------------------------------------------------------------------
// M-build λ-split lerp: `m_total = m_zero + λ · (m_one − m_zero)`.
//
// The M-build kernel (`frac_precompute_m_build_dev_challenge_raw`) with
// `inline_fold = true` is affine in `r_prev` (the round-0 challenge it
// folds into the layer): `M(r) = M(0) + r · (M(1) − M(0))`. Emitting
// the two challenge-free builds (`r_prev = 0`, `r_prev = 1`) lets the
// scheduler overlap them with the transcript activity that produces the
// actual `r_prev`; this lerp then combines them once `r_prev` is
// sampled. See `fractional_sumcheck_gpu_ir_pipelined`.

/// Build the DSL module for the M-build λ-split lerp at window size `w`.
pub fn build_frac_m_lerp_module(w: usize) -> Module {
    assert!(w >= 1, "m_lerp: w must be >= 1, got {w}");
    let m_len = 1usize << (2 * w);
    let mut b = IRBuilder::new();
    let m_zero = b.input("m_zero", ScalarType::FpExt, vec![m_len]);
    let m_one = b.input("m_one", ScalarType::FpExt, vec![m_len]);
    let r = bind_challenge_as_fpext(&mut b, "r");
    let body = b.compute(m_len, move |b, i| {
        let z = b.index(m_zero, &[i]);
        let o = b.index(m_one, &[i]);
        let d = b.sub(o, z);
        let ld = b.mul(r, d);
        b.add(z, ld)
    });
    b.finish(format!("frac_m_lerp_dsl_w{w}"), body)
}

/// Insert the M-build λ-split lerp: `m_total = m_zero + r · (m_one − m_zero)`.
/// All three EF-scaled inputs (`m_zero`, `m_one`, `m_total`) must be
/// `1 << (2 * w)`-element `EF` buffers; `r` is a `[D_EF]`-shaped
/// `BabyBear` challenge buffer.
pub fn frac_m_lerp_ir_dsl(
    g: &mut GraphBuilder,
    m_zero: BufId,
    m_one: BufId,
    r: BufId,
    m_total: BufId,
    w: usize,
) {
    g.insert_kernel(
        build_frac_m_lerp_module(w),
        [m_zero, m_one, r],
        [m_total],
        &[],
    );
}

// ---------------------------------------------------------------------------
// M-build r-split quadratic combine.
//
// With `inline_fold = true` the M-build kernel folds `r_prev` into every
// pq pair it reads: each folded value is affine in `r_prev`, and the
// accumulated sums multiply two folded values, so `M(r_prev)` is a
// degree-2 polynomial in `r_prev` (elementwise). Three challenge-free
// builds at `r_prev ∈ {0, 1, 2}` therefore determine it exactly; these
// modules interpolate at the sampled `r_prev` following
// `openvm_stark_backend::poly_common::interpolate_quadratic_at_012`:
//   s1 = M(1) − M(0), s2 = M(2) − M(1),
//   p = (s2 − s1) / 2, q = s1 − p,
//   M(r) = (p·r + q)·r + M(0).

/// `1 / 2` in BabyBear canonical form (`(p + 1) / 2`).
const INV2_CANONICAL: u32 = 1_006_632_961;

/// Elementwise quadratic interpolation at nodes `{0, 1, 2}`; `inv2` must
/// be a `const_fpext([INV2_CANONICAL, 0, 0, 0])` handle.
fn interp_quad_012_at(
    b: &mut IRBuilder,
    x0: NodeId,
    x1: NodeId,
    x2: NodeId,
    r: NodeId,
    inv2: NodeId,
) -> NodeId {
    let s1 = b.sub(x1, x0);
    let s2 = b.sub(x2, x1);
    let d = b.sub(s2, s1);
    let p = b.mul(d, inv2);
    let q = b.sub(s1, p);
    let pr = b.mul(p, r);
    let prq = b.add(pr, q);
    let prqr = b.mul(prq, r);
    b.add(prqr, x0)
}

/// Build the DSL module for the M-build r-split combine at window size
/// `w`: `m_total = interp_quad_012([m_r0, m_r1, m_r2], r)` elementwise.
pub fn build_frac_m_lagrange3_module(w: usize) -> Module {
    assert!(w >= 1, "m_lagrange3: w must be >= 1, got {w}");
    let m_len = 1usize << (2 * w);
    let mut b = IRBuilder::new();
    let m_r0 = b.input("m_r0", ScalarType::FpExt, vec![m_len]);
    let m_r1 = b.input("m_r1", ScalarType::FpExt, vec![m_len]);
    let m_r2 = b.input("m_r2", ScalarType::FpExt, vec![m_len]);
    let r = bind_challenge_as_fpext(&mut b, "r");
    let inv2 = b.const_fpext([INV2_CANONICAL, 0, 0, 0]);
    let body = b.compute(m_len, move |b, i| {
        let x0 = b.index(m_r0, &[i]);
        let x1 = b.index(m_r1, &[i]);
        let x2 = b.index(m_r2, &[i]);
        interp_quad_012_at(b, x0, x1, x2, r, inv2)
    });
    b.finish(format!("frac_m_lagrange3_dsl_w{w}"), body)
}

/// Insert the M-build r-split combine: `m_total[i]` is the quadratic
/// through `(0, m_r0[i])`, `(1, m_r1[i])`, `(2, m_r2[i])` evaluated at
/// `r`. All EF-valued buffers must be `1 << (2 * w)`-element `EF`
/// buffers; `r` is a `[D_EF]`-shaped BabyBear challenge buffer.
pub fn frac_m_lagrange3_ir_dsl(
    g: &mut GraphBuilder,
    m_r0: BufId,
    m_r1: BufId,
    m_r2: BufId,
    r: BufId,
    m_total: BufId,
    w: usize,
) {
    g.insert_kernel(
        build_frac_m_lagrange3_module(w),
        [m_r0, m_r1, m_r2, r],
        [m_total],
        &[],
    );
}

/// Build the DSL module for the fused λ×r combine at window size `w`.
/// The inline-fold M-build output is affine in `lambda` and quadratic in
/// `r_prev`, so six challenge-free builds at `(λ, r) ∈ {0,1} × {0,1,2}`
/// determine it exactly:
///   `M_λ(r)  = interp_quad_012([m_l{λ}_r0, m_l{λ}_r1, m_l{λ}_r2], r)`
///   `m_total = M_0(r) + λ · (M_1(r) − M_0(r))`
pub fn build_frac_m_lagrange3_lerp_module(w: usize) -> Module {
    assert!(w >= 1, "m_lagrange3_lerp: w must be >= 1, got {w}");
    let m_len = 1usize << (2 * w);
    let mut b = IRBuilder::new();
    let m_l0 = [
        b.input("m_l0_r0", ScalarType::FpExt, vec![m_len]),
        b.input("m_l0_r1", ScalarType::FpExt, vec![m_len]),
        b.input("m_l0_r2", ScalarType::FpExt, vec![m_len]),
    ];
    let m_l1 = [
        b.input("m_l1_r0", ScalarType::FpExt, vec![m_len]),
        b.input("m_l1_r1", ScalarType::FpExt, vec![m_len]),
        b.input("m_l1_r2", ScalarType::FpExt, vec![m_len]),
    ];
    let lambda = bind_challenge_as_fpext(&mut b, "lambda");
    let r = bind_challenge_as_fpext(&mut b, "r");
    let inv2 = b.const_fpext([INV2_CANONICAL, 0, 0, 0]);
    let body = b.compute(m_len, move |b, i| {
        let [x0, x1, x2] = m_l0.map(|m| b.index(m, &[i]));
        let v0 = interp_quad_012_at(b, x0, x1, x2, r, inv2);
        let [y0, y1, y2] = m_l1.map(|m| b.index(m, &[i]));
        let v1 = interp_quad_012_at(b, y0, y1, y2, r, inv2);
        let d = b.sub(v1, v0);
        let ld = b.mul(lambda, d);
        b.add(v0, ld)
    });
    b.finish(format!("frac_m_lagrange3_lerp_dsl_w{w}"), body)
}

/// Insert the fused λ×r combine of six challenge-free M builds.
/// `m[li][ri]` holds the build at `lambda = li`, `r_prev = ri`; `lambda`
/// and `r` are `[D_EF]`-shaped BabyBear challenge buffers; all M buffers
/// are `1 << (2 * w)`-element `EF` buffers.
pub fn frac_m_lagrange3_lerp_ir_dsl(
    g: &mut GraphBuilder,
    m: [[BufId; 3]; 2],
    lambda: BufId,
    r: BufId,
    m_total: BufId,
    w: usize,
) {
    g.insert_kernel(
        build_frac_m_lagrange3_lerp_module(w),
        [
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], lambda, r,
        ],
        [m_total],
        &[],
    );
}

// ---------------------------------------------------------------------------
// Tests.

#[cfg(test)]
mod dsl_port_tests {
    use crypto_compiler::{
        graph_exe::GraphCompiler,
        graph_ir::{ConstBuf, DeviceType, GraphBuilder},
        passes::fusion_v2::FusionOptionsV2,
        planner::SchedulerMode,
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
    /// given buffers as raw bytes. Registers each buffer as a graph output
    /// before compiling.
    fn run_graph_read_bufs(
        mut g: GraphBuilder,
        bufs: &[BufId],
        ctx: &GpuDeviceCtx,
    ) -> Vec<Vec<u8>> {
        for &b in bufs {
            g.register_output(b);
        }
        let mut compiler =
            GraphCompiler::new()
                .device(DeviceType::Cuda(0))
                .scheduler(SchedulerMode::ListV1 {
                    params: crypto_compiler::planner::ListSchedulerV1::default(),
                });
        // `FRAC_DSL_FUSION=v2` replays the whole suite through the
        // fusion-v2 pipeline (M12 bit-for-bit gate; enable
        // `crypto-compiler/planner-ortools` so extraction is CP-SAT-backed
        // beyond the brute-force cap); `off` compiles the graph unfused.
        // Default remains the existing fusion pass.
        match std::env::var("FRAC_DSL_FUSION").as_deref() {
            Ok("v2") => {
                compiler = compiler.fusion_v2_options(FusionOptionsV2 {
                    verbose: true,
                    ..FusionOptionsV2::default()
                })
            }
            Ok("off") => compiler = compiler.without_fusion(),
            _ => {}
        }
        let mut exe = compiler.compile(g).expect("graph compile");
        if let Some(v2) = exe.fusion_report().and_then(|r| r.v2.as_ref()) {
            eprintln!(
                "[dsl-fusion-v2] nodes {} -> {}, inserted={}, selected={}, fallback={:?}",
                v2.nodes_before,
                v2.nodes_after,
                v2.candidates_inserted,
                v2.selected_from_solver,
                v2.fallback_reason,
            );
        }
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

        // n >= 9 (num_x >= 1024) exercises the graph-level block-sums +
        // row-sum decomposition (the single-module form would need a
        // multi-stage reduce lowering, which graph modules don't support);
        // n = 14 hits the g_blocks = 64 cap.
        for n in [3usize, 4, 5, 9, 10, 14] {
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

    // ---- Small-round overlap kernels (host references) --------------------

    #[test]
    fn frac_tree_revert_two_input_dsl_matches_host() {
        use p3_field::Field;
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);

        for half in [2usize, 8, 32] {
            // Layer bigger than the read region [half, 2*half) to exercise
            // the whole-tree binding.
            let layer_len = 4 * half;
            let layer = make_host_leaves(layer_len, 0x7EE7 ^ half as u64);
            let parents = make_host_leaves(half, 0x7EE8 ^ half as u64);

            // Host reference.
            let mut want = Vec::with_capacity(2 * half);
            for k in 0..half {
                let rhs = layer[half + k];
                let inv = rhs.q.inverse();
                let new_q = parents[k].q * inv;
                let new_p = (parents[k].p - new_q * rhs.p) * inv;
                want.push(Frac { p: new_p, q: new_q });
            }
            want.extend_from_slice(&layer[half..2 * half]);

            // DSL side.
            let mut g = GraphBuilder::new();
            let parents_buf = frac_const_buf(&mut g, "parents", &parents);
            let layer_buf = frac_const_buf(&mut g, "layer_in", &layer);
            let out_buf = add_frac_ef_buf(&mut g, device, "pq_out", 2 * half);
            frac_tree_revert_two_input_ir_dsl(
                &mut g,
                parents_buf,
                layer_buf,
                out_buf,
                half,
                layer_len,
            );
            let out_out = add_frac_ef_buf(&mut g, device, "pq_out_out", 2 * half);
            g.insert_memcpy(out_buf, out_out);
            let got = run_graph_read_bufs(g, &[out_out], &ctx).remove(0);
            assert_eq!(
                &got[..],
                frac_bytes(&want),
                "frac_tree_revert_two_input_dsl mismatch at half={half}"
            );
        }
    }

    #[test]
    fn frac_m_outer_product_dsl_matches_host() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);

        for m in [2usize, 8, 16] {
            let pq = make_host_leaves(2 * m, 0x0AB0 ^ m as u64);

            // Host reference; the interleaved [m*m, 2] FpExt output is
            // byte-identical to a Frac<EF> buffer with p = M_a, q = M_b.
            let mut want = Vec::with_capacity(m * m);
            for u in 0..m {
                for v in 0..m {
                    let m_a = pq[u].p * pq[m + v].q + pq[m + u].p * pq[v].q;
                    let m_b = pq[u].q * pq[m + v].q;
                    want.push(Frac { p: m_a, q: m_b });
                }
            }

            let mut g = GraphBuilder::new();
            let pq_buf = frac_const_buf(&mut g, "pq", &pq);
            let m_ab = add_frac_ef_buf(&mut g, device, "m_ab", m * m);
            frac_m_outer_product_ir_dsl(&mut g, pq_buf, m_ab, m);
            let m_out = add_frac_ef_buf(&mut g, device, "m_ab_out", m * m);
            g.insert_memcpy(m_ab, m_out);
            let got = run_graph_read_bufs(g, &[m_out], &ctx).remove(0);
            assert_eq!(
                &got[..],
                frac_bytes(&want),
                "frac_m_outer_product_dsl mismatch at m={m}"
            );
        }
    }

    #[test]
    fn frac_precompute_m_eval_lambda_dsl_matches_host() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x1A3B_5C7D);

        // total = 2^(t + w - 1); (6,5) and (8,7) lower multi-stage
        // (total >= 512) and exercise the block + row-sum decomposition.
        for (w, t) in [(2usize, 0usize), (2, 1), (3, 1), (5, 4), (6, 5), (8, 7)] {
            let m = 1usize << w;
            let prefix_size = 1usize << t;
            let suffix_bits = w - t - 1;
            let suffix_size = 1usize << suffix_bits;
            let total = prefix_size * prefix_size * suffix_size;
            let m_ab = make_host_leaves(m * m, 0x1A3B ^ (w * 31 + t) as u64);
            let lambda: EF = rng.random();
            let eq_r_prefix: Vec<EF> = (0..prefix_size).map(|_| rng.random()).collect();
            let eq_suffix: Vec<EF> = (0..suffix_size).map(|_| rng.random()).collect();

            // Host reference on M = M_a + lambda * M_b.
            let mt = |i: usize| m_ab[i].p + lambda * m_ab[i].q;
            let two = EF::ONE + EF::ONE;
            let mut s1 = EF::ZERO;
            let mut s2 = EF::ZERO;
            for idx in 0..total {
                let suffix = idx % suffix_size;
                let tmp = idx / suffix_size;
                let b2 = tmp % prefix_size;
                let b1 = tmp / prefix_size;
                let weight = eq_r_prefix[b1] * eq_r_prefix[b2] * eq_suffix[suffix];
                let beta1_0 = (b1 << (suffix_bits + 1)) | suffix;
                let beta1_1 = beta1_0 | (1 << suffix_bits);
                let beta2_0 = (b2 << (suffix_bits + 1)) | suffix;
                let beta2_1 = beta2_0 | (1 << suffix_bits);
                let m00 = mt(beta1_0 * m + beta2_0);
                let m01 = mt(beta1_0 * m + beta2_1);
                let m10 = mt(beta1_1 * m + beta2_0);
                let m11 = mt(beta1_1 * m + beta2_1);
                s1 += weight * m11;
                s2 += weight * (m00 - two * (m01 + m10 - m11 - m11));
            }
            let want = vec![s1, s2];

            // DSL side.
            let mut g = GraphBuilder::new();
            let m_buf = frac_const_buf(&mut g, "m_ab", &m_ab);
            let lambda_buf = ef_const_ext_scalar_buf(&mut g, device, "lambda", lambda);
            let ep_buf = ef_slice_const_buf(&mut g, "eq_r_prefix", &eq_r_prefix);
            let es_buf = ef_slice_const_buf(&mut g, "eq_suffix", &eq_suffix);
            let out_buf = add_ef_buf(&mut g, device, "out", 2);
            frac_precompute_m_eval_lambda_ir_dsl(
                &mut g, m_buf, lambda_buf, ep_buf, es_buf, out_buf, w, t,
            );
            let out_out = add_ef_buf(&mut g, device, "out_out", 2);
            g.insert_memcpy(out_buf, out_out);
            let bytes = run_graph_read_bufs(g, &[out_out], &ctx).remove(0);
            let got: Vec<EF> = bytes
                .chunks_exact(size_of::<EF>())
                .map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const EF) })
                .collect();
            assert_eq!(
                got, want,
                "frac_precompute_m_eval_lambda_dsl mismatch at w={w}, t={t}"
            );
        }
    }

    #[test]
    fn frac_claims_fold_dsl_matches_host() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0xC1A1_35F0);

        // m = 2048 is the largest small-regime window (pq_size = 4096).
        for m in [4usize, 64, 2048] {
            let pq = make_host_leaves(2 * m, 0xC1A1 ^ m as u64);
            let eq_r: Vec<EF> = (0..m).map(|_| rng.random()).collect();

            // Host reference: claims = (p0, q0, p1, q1) as 2 Fracs.
            let mut acc = [EF::ZERO; 4];
            for u in 0..m {
                acc[0] += eq_r[u] * pq[u].p;
                acc[1] += eq_r[u] * pq[u].q;
                acc[2] += eq_r[u] * pq[m + u].p;
                acc[3] += eq_r[u] * pq[m + u].q;
            }
            let want = vec![
                Frac {
                    p: acc[0],
                    q: acc[1],
                },
                Frac {
                    p: acc[2],
                    q: acc[3],
                },
            ];

            let mut g = GraphBuilder::new();
            let pq_buf = frac_const_buf(&mut g, "pq", &pq);
            let eq_buf = ef_slice_const_buf(&mut g, "eq_r", &eq_r);
            let claims = add_frac_ef_buf(&mut g, device, "claims", 2);
            frac_claims_fold_ir_dsl(&mut g, pq_buf, eq_buf, claims, m);
            let claims_out = add_frac_ef_buf(&mut g, device, "claims_out", 2);
            g.insert_memcpy(claims, claims_out);
            let got = run_graph_read_bufs(g, &[claims_out], &ctx).remove(0);
            assert_eq!(
                &got[..],
                frac_bytes(&want),
                "frac_claims_fold_dsl mismatch at m={m}"
            );
        }
    }

    #[test]
    fn frac_m_lagrange3_dsl_matches_host() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x1A63 ^ 0x2222);

        // Host quadratic interpolation at nodes {0, 1, 2}, mirroring
        // `interpolate_quadratic_at_012`.
        use p3_field::Field;
        let inv2 = (EF::ONE + EF::ONE).inverse();
        let interp = |x0: EF, x1: EF, x2: EF, r: EF| {
            let s1 = x1 - x0;
            let s2 = x2 - x1;
            let p = (s2 - s1) * inv2;
            let q = s1 - p;
            (p * r + q) * r + x0
        };
        let read_efs = |bytes: Vec<u8>| -> Vec<EF> {
            bytes
                .chunks_exact(size_of::<EF>())
                .map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const EF) })
                .collect()
        };

        for w in [1usize, 2, 3] {
            let m_len = 1usize << (2 * w);
            let ms: Vec<Vec<EF>> = (0..6)
                .map(|_| (0..m_len).map(|_| rng.random()).collect())
                .collect();
            let r: EF = rng.random();
            let lambda: EF = rng.random();

            // 3-input r-only combine.
            let want_r: Vec<EF> = (0..m_len)
                .map(|i| interp(ms[0][i], ms[1][i], ms[2][i], r))
                .collect();
            // 6-input fused λ×r combine.
            let want_lr: Vec<EF> = (0..m_len)
                .map(|i| {
                    let v0 = interp(ms[0][i], ms[1][i], ms[2][i], r);
                    let v1 = interp(ms[3][i], ms[4][i], ms[5][i], r);
                    v0 + lambda * (v1 - v0)
                })
                .collect();

            let mut g = GraphBuilder::new();
            let m_bufs: Vec<BufId> = ms
                .iter()
                .enumerate()
                .map(|(k, m)| ef_slice_const_buf(&mut g, &format!("m{k}"), m))
                .collect();
            let r_buf = ef_const_ext_scalar_buf(&mut g, device, "r", r);
            let lambda_buf = ef_const_ext_scalar_buf(&mut g, device, "lambda", lambda);
            let out_r = add_ef_buf(&mut g, device, "out_r", m_len);
            frac_m_lagrange3_ir_dsl(&mut g, m_bufs[0], m_bufs[1], m_bufs[2], r_buf, out_r, w);
            let out_lr = add_ef_buf(&mut g, device, "out_lr", m_len);
            frac_m_lagrange3_lerp_ir_dsl(
                &mut g,
                [
                    [m_bufs[0], m_bufs[1], m_bufs[2]],
                    [m_bufs[3], m_bufs[4], m_bufs[5]],
                ],
                lambda_buf,
                r_buf,
                out_lr,
                w,
            );
            let out_r_copy = add_ef_buf(&mut g, device, "out_r_copy", m_len);
            g.insert_memcpy(out_r, out_r_copy);
            let out_lr_copy = add_ef_buf(&mut g, device, "out_lr_copy", m_len);
            g.insert_memcpy(out_lr, out_lr_copy);
            let mut bufs = run_graph_read_bufs(g, &[out_r_copy, out_lr_copy], &ctx);
            let got_lr = read_efs(bufs.remove(1));
            let got_r = read_efs(bufs.remove(0));
            assert_eq!(got_r, want_r, "frac_m_lagrange3_dsl mismatch at w={w}");
            assert_eq!(
                got_lr, want_lr,
                "frac_m_lagrange3_lerp_dsl mismatch at w={w}"
            );
        }
    }
}
