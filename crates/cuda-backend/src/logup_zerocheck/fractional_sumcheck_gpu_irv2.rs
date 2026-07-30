//! Alternate DSL-first port of the fractional-GKR sumcheck prover.
//!
//! Compared with [`super::fractional_ir::fractional_sumcheck_gpu_ir`] (v1),
//! this v2 driver deliberately simplifies the graph structure:
//!
//! 1. **No PrecomputeM.** Only the FoldEval strategy is emitted.
//! 2. **No in-place reverts.** The segment tree is materialized layer by
//!    layer as separate persistent buffers (`tree_layers[k]` has `2^k`
//!    `Frac<EF>` elements). Every outer round reads its input pq buffer
//!    straight from `tree_layers[round + 1]` — no `frac_build_tree_layer`
//!    revert nodes anywhere in the graph.
//! 3. **Dense-only.** `real_len == logical_len`. Virtual/compact mode is
//!    unsupported; asserted at the entry point.
//! 4. **DSL-first — every kernel node.** No blackbox kernels at all. Tree
//!    combines via a small forward-combine DSL module added here
//!    ([`build_frac_tree_layer_forward_module`]); the leaves-preparation
//!    step (bit-reversal + alpha shift) is a single DSL kernel
//!    ([`build_bit_rev_and_alpha_module`]) — bit-reverse is quasi-affine
//!    when written as `Σ_{b=0..k} ((i mod 2^{b+1}) / 2^b) · 2^{k-1-b}`
//!    (see [`bit_rev_expr`]), which the DSL's index-expression checker
//!    accepts. Compute rounds go through [`frac_compute_round_ir_dsl`],
//!    folds via [`fold_ef_frac_columns_ir_dsl`], scalar helpers via
//!    [`reduce_to_single_evaluation_ir`] / [`claim_combine_ir`] /
//!    [`observe_and_update_ir`].
//!
//! Proofs emitted by v2 are byte-identical to those of the eager
//! [`super::fractional::fractional_sumcheck_gpu`] (and hence v1) for the
//! dense case: the segment-tree layer values are mathematically the same
//! (v1's in-place forward-build + top revert produces the same layer values
//! v2 stores per layer; see the module tests below), and the FoldEval
//! sumcheck reads the same layer positions in the same order.

use crypto_compiler::{
    graph_ir::{BufId, DeviceType, GraphBuilder},
    ir::{IRBuilder, Module, NodeId, ScalarType},
};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32};
use p3_util::log2_strict_usize;
// (`PrimeCharacteristicRing` is used below for `EF::ONE` via the trait.)

use super::{
    errors::FractionalSumcheckError,
    fractional_ir::{
        add_frac_ef_buf, claim_combine_ir, ef_const_ext_scalar_buf, extract_claim_pair_ir,
        extract_root_pq_ir, reduce_to_single_evaluation_ir, FracSumcheckProofIR, GkrLayerClaimIR,
        SqrtEqLayersIR,
    },
    fractional_ir_dsl::{fold_ef_frac_columns_ir_dsl, frac_compute_round_ir_dsl},
};
use crate::{
    logup_zerocheck::fractional_ir::{add_ef_buf, add_ext_scalar_buf, observe_and_update_ir},
    prelude::EF,
    sponge_graph_ir::FiatShamirTranscriptGraphIR,
};

// ---------------------------------------------------------------------------
// DSL bit-reversal helper.

/// Convert an `EF` value into the four canonical `BabyBear` coefficients
/// [`IRBuilder::const_fpext`] expects.
fn ef_to_const_coeffs(x: EF) -> [u32; 4] {
    let coeffs: &[crate::prelude::F] = x.as_basis_coefficients_slice();
    debug_assert_eq!(coeffs.len(), 4);
    [
        coeffs[0].as_canonical_u32(),
        coeffs[1].as_canonical_u32(),
        coeffs[2].as_canonical_u32(),
        coeffs[3].as_canonical_u32(),
    ]
}

/// Build a DSL expression tree for the bit-reversal of `i` on `k` bits.
///
/// Uses the identity
///
/// ```text
/// bit_rev(i) = Σ_{b=0..k} ((i mod 2^{b+1}) / 2^b) · 2^{k-1-b}
/// ```
///
/// Every term is quasi-affine (mod and floor-div by compile-time powers of
/// two, mul by constant, add), so the DSL's index-expression checker
/// accepts a `b.index(buf, &[bit_rev_expr(b, i, k), ...])` access.
///
/// `k == 0` returns the constant 0; `k == 1` returns `i` unchanged.
fn bit_rev_expr(b: &mut IRBuilder, i: NodeId, k: usize) -> NodeId {
    if k == 0 {
        return b.const_u32(0);
    }
    if k == 1 {
        return i;
    }
    let mut acc: Option<NodeId> = None;
    for bit in 0..k {
        let mod_c = b.const_u32(1u32 << (bit + 1));
        let div_c = b.const_u32(1u32 << bit);
        let i_mod = b.rem(i, mod_c);
        let bit_val = b.div(i_mod, div_c);
        let out_bit_pos = k - 1 - bit;
        // Multiply-by-power-of-two: skip when the output bit is bit 0.
        let term = if out_bit_pos == 0 {
            bit_val
        } else {
            let coeff = b.const_u32(1u32 << out_bit_pos);
            b.mul(bit_val, coeff)
        };
        acc = Some(match acc {
            None => term,
            // The bit positions are disjoint (each `term` contributes to a
            // single, distinct output bit), so `add` is the same as OR here.
            Some(a) => b.add(a, term),
        });
    }
    acc.unwrap_or_else(|| b.const_u32(0))
}

// ---------------------------------------------------------------------------
// Bit-rev + alpha-shift DSL module.

/// DSL module for the leaves-preparation step:
///
/// ```text
/// out[i] = (leaves[bit_rev(i)].p,  leaves[bit_rev(i)].q + alpha)
/// ```
///
/// One structured [`insert_kernel`] node — replaces the two blackbox calls
/// (`bit_rev_frac_ext_ir` + `frac_add_alpha_ir`) v2 originally used to
/// populate `tree[N]`. The compiler's quasi-affine index checker accepts the
/// bit-reverse read index because it decomposes into a sum of
/// `((i mod 2^{b+1}) / 2^b) · 2^{k-1-b}` terms — mod / floor-div by
/// compile-time powers of two, then a sum with constant multipliers.
pub fn build_bit_rev_and_alpha_module(n: usize, alpha: EF) -> Module {
    assert!(
        n >= 2 && n.is_power_of_two(),
        "bit_rev_and_alpha: n must be a power of two >= 2, got {n}"
    );
    let k = log2_strict_usize(n);
    let alpha_coeffs = ef_to_const_coeffs(alpha);
    let mut b = IRBuilder::new();
    let leaves = b.input("leaves", ScalarType::FpExt, vec![n, 2]);
    let body = b.compute(n, move |b, i| {
        let br = bit_rev_expr(b, i, k);
        let idx0 = b.const_u32(0);
        let idx1 = b.const_u32(1);
        let p = b.index(leaves, &[br, idx0]);
        let q_raw = b.index(leaves, &[br, idx1]);
        let alpha_c = b.const_fpext(alpha_coeffs);
        let q = b.add(q_raw, alpha_c);
        b.pack(&[p, q])
    });
    b.finish(format!("bit_rev_and_alpha_dsl_{n}"), body)
}

/// Insert the DSL bit-rev + alpha-shift kernel: fills `layer_out` (size `n`
/// `Frac<EF>`) with `leaves` in bit-reversed order and `alpha` added to every
/// `q` slot.
pub fn bit_rev_and_alpha_ir_dsl(
    g: &mut GraphBuilder,
    leaves: BufId,
    layer_out: BufId,
    n: usize,
    alpha: EF,
) {
    g.insert_kernel(build_bit_rev_and_alpha_module(n, alpha), [leaves], [layer_out]);
}

// ---------------------------------------------------------------------------
// Forward tree-combine DSL module.

/// Build a DSL module for one forward tree-layer combine.
///
/// Reads `layer_in : [layer_in_size, 2] FpExt` (a `Frac<EF>` buffer) and
/// produces `layer_out : [layer_in_size / 2, 2] FpExt` where each output
/// row `i` is the fractional addition of `(layer_in[i], layer_in[i + half])`.
///
/// Dense-only. `layer_in_size` must be a power of two `>= 2`.
pub fn build_frac_tree_layer_forward_module(layer_in_size: usize) -> Module {
    assert!(
        layer_in_size >= 2 && layer_in_size.is_power_of_two(),
        "tree forward module: layer_in_size must be a power of two >= 2, got {layer_in_size}"
    );
    let half = layer_in_size / 2;
    let mut b = IRBuilder::new();
    let layer_in = b.input("layer_in", ScalarType::FpExt, vec![layer_in_size, 2]);

    let body = b.compute(half, move |b, i| {
        let zero_c = b.const_u32(0);
        let one_c = b.const_u32(1);
        let half_c = b.const_u32(half as u32);
        let j = b.add(i, half_c);

        let pa = b.index(layer_in, &[i, zero_c]);
        let qa = b.index(layer_in, &[i, one_c]);
        let pb = b.index(layer_in, &[j, zero_c]);
        let qb = b.index(layer_in, &[j, one_c]);

        // frac_add((pa, qa), (pb, qb)) = (pa*qb + pb*qa, qa*qb).
        let pa_qb = b.mul(pa, qb);
        let pb_qa = b.mul(pb, qa);
        let out_p = b.add(pa_qb, pb_qa);
        let out_q = b.mul(qa, qb);
        b.pack(&[out_p, out_q])
    });
    b.finish(format!("frac_tree_layer_forward_dsl_{layer_in_size}"), body)
}

/// Insert a forward tree-layer combine as a structured DSL kernel node.
///
/// `layer_in` is a `Frac<EF>` buffer of `layer_in_size` elements; `layer_out`
/// is a fresh `Frac<EF>` buffer of `layer_in_size / 2` elements. Emits one
/// [`crypto_compiler::graph_ir::GraphNode::Kernel`] (no blackbox).
pub fn frac_tree_layer_forward_ir_dsl(
    g: &mut GraphBuilder,
    layer_in: BufId,
    layer_out: BufId,
    layer_in_size: usize,
) {
    g.insert_kernel(
        build_frac_tree_layer_forward_module(layer_in_size),
        [layer_in],
        [layer_out],
    );
}

// ---------------------------------------------------------------------------
// v2 driver.

/// Simplified DSL-first port of
/// [`super::fractional::fractional_sumcheck_gpu`].
///
/// # Contract
/// - Dense: `leaves` must be exactly `logical_len` `Frac<EF>` entries.
/// - `logical_len` must be a power of two `>= 2`.
/// - `assert_zero` follows v1 semantics: skip the `p_root` observe when true.
///
/// # Structure
/// 1. Bit-reverse `leaves` and add `alpha` to every `q` slot, into
///    `layer[N]` — one DSL kernel ([`bit_rev_and_alpha_ir_dsl`]).
/// 2. Emit `N` forward tree-combine DSL kernels to materialize
///    `layer[N-1] .. layer[0]` (each half the size of its parent).
/// 4. Observe `root.p` (unless `assert_zero`) and `root.q` from `layer[0]`.
/// 5. Extract first claim from `layer[N-1]` (size 2). Observe claim, sample
///    `mu_1`. Seed `xi_prev = [mu_1]`.
/// 6. Outer GKR loop for `round in 1..N`:
///    - Read `layer[round + 1]` (size `2^{round + 1}`) as `pq_buffer`.
///    - Sample `lambda_{round + 1}`.
///    - Seed `prev_s_eval` via [`reduce_to_single_evaluation_ir`] +
///      [`claim_combine_ir`] on the previous layer's claims.
///    - Build `SqrtEqLayersIR` from `&xi_prev[1..]`.
///    - Inner sumcheck rounds `t in 0..round`:
///      * Compute `d_sum` via [`frac_compute_round_ir_dsl`] on the current
///        pq buffer.
///      * Drop one eq layer.
///      * [`observe_and_update_ir`] with `xi_j = xi_prev[t]` → sample `r_t`,
///        push `s_evals`, update `prev_s_eval` and `eq_r_acc`.
///      * Fold pq buffer by `r_t` via [`fold_ef_frac_columns_ir_dsl`] into a
///        fresh half-sized buffer (used in the next inner round OR by the
///        final claim extraction).
///    - Extract layer claim at positions `(0, 1)` of the final folded buffer
///      (size 2). Observe. Sample `mu_{round + 1}`. Update `xi_prev`.
///
/// # Byte-equality with the eager prover
///
/// Every transcript observe / sample is emitted in the same order and with
/// the same values as v1 (and hence the eager `fractional_sumcheck_gpu`):
/// - Tree building writes the same "layer j values" v1's in-place tree +
///   top-revert produces — see the module docs for the derivation.
/// - The `frac_compute_round_ir_dsl` module reads the same four positions
///   `(idx, idx + q, idx + h, idx + h + q)` in the size-`2^{round+1}` layer
///   that v1's `compute_round_and_revert` reads (v1 reverts `[0, h)` on the
///   fly; v2 reads the same values directly from `tree_layers[round + 1]`).
/// - Folds use the same DSL kernel as v1's dense fold.
///
/// The single conceptual difference is that v1 fuses (revert + compute) and
/// (fold + compute) into single CUDA kernels while v2 emits them as separate
/// DSL nodes; the arithmetic is identical.
#[allow(clippy::type_complexity)]
pub fn fractional_sumcheck_gpu_irv2<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    leaves: BufId,
    logical_len: usize,
    alpha: EF,
    assert_zero: bool,
    device: DeviceType,
) -> Result<FracSumcheckProofIR, FractionalSumcheckError>
where
    TS: FiatShamirTranscriptGraphIR,
{
    assert!(logical_len >= 2, "logical_len must be >= 2");
    assert!(
        logical_len.is_power_of_two(),
        "logical_len must be a power of two"
    );
    let total_rounds = log2_strict_usize(logical_len);

    // -----------------------------------------------------------------------
    // Segment tree: persistent per-layer buffers `tree[k]` of `2^k` Fracs.
    //
    // `tree[N]` = bit-reversed leaves with `alpha` added to `q` of every
    // entry. `tree[k-1][idx] = frac_add(tree[k][idx], tree[k][idx + 2^{k-1}])`
    // for k = N, N-1, ..., 1.
    let mut tree: Vec<BufId> = Vec::with_capacity(total_rounds + 1);
    tree.resize(total_rounds + 1, leaves); // sentinel; will overwrite

    // Layer N: bit-reversed leaves with alpha baked into every `q` slot.
    // One DSL kernel — the bit-reverse permutation is quasi-affine per
    // `build_bit_rev_and_alpha_module`.
    tree[total_rounds] = add_frac_ef_buf(g, device, "tree_layer_N", logical_len);
    bit_rev_and_alpha_ir_dsl(g, leaves, tree[total_rounds], logical_len, alpha);

    // Forward combines: layer[k] from layer[k+1], for k = N-1 down to 0.
    for k in (0..total_rounds).rev() {
        let out_size = 1usize << k;
        tree[k] = add_frac_ef_buf(g, device, &format!("tree_layer_{k}"), out_size);
        frac_tree_layer_forward_ir_dsl(g, tree[k + 1], tree[k], out_size * 2);
    }

    // -----------------------------------------------------------------------
    // Root observe.
    let root_p = add_ext_scalar_buf(g, device, "frac_root_p");
    let root_q = add_ext_scalar_buf(g, device, "frac_root_q");
    extract_root_pq_ir(g, tree[0], 1, root_p, root_q);
    if !assert_zero {
        transcript.observe_ext(g, root_p);
    }
    transcript.observe_ext(g, root_q);

    // -----------------------------------------------------------------------
    // First claim (from layer 1, size 2).
    let first_claim = extract_claim_pair_ir(g, tree[1], 2, 1, "claim0", device);
    let mut claims_per_layer: Vec<GkrLayerClaimIR> = Vec::with_capacity(total_rounds);
    claims_per_layer.push(first_claim);
    for buf in first_claim.as_array() {
        transcript.observe_ext(g, buf);
    }
    let mu_1 = transcript.sample_ext(g);
    let mut xi_prev: Vec<BufId> = vec![mu_1];
    let mut sumcheck_polys: Vec<Vec<[BufId; crate::logup_zerocheck::fractional_ir::GKR_S_DEG]>> =
        Vec::with_capacity(total_rounds);

    // Shared read-only `EF::ONE` seed for `eq_r_acc` reset each round.
    let eq_r_acc_one = ef_const_ext_scalar_buf(g, device, "eq_r_acc_one_v2", EF::ONE);

    // -----------------------------------------------------------------------
    // Outer GKR loop.
    for round in 1..total_rounds {
        debug_assert_eq!(xi_prev.len(), round);

        // Input pq buffer: the persistent tree layer of size 2^{round + 1}.
        let mut pq_buffer = tree[round + 1];
        let mut pq_size = 1usize << (round + 1);

        // Fresh challenge lambda for this outer round.
        let lambda = transcript.sample_ext(g);

        // Reduce previous layer's claims via mu = xi_prev[0], combine with
        // lambda to seed `prev_s_eval`.
        let (numer, denom) =
            reduce_to_single_evaluation_ir(g, *claims_per_layer.last().unwrap(), xi_prev[0], device);
        let mut prev_s_eval = claim_combine_ir(g, numer, denom, lambda, device);
        let mut eq_r_acc = eq_r_acc_one;

        // Eq buffer covering the tail challenges xi_prev[1..].
        let mut eq_buffer = SqrtEqLayersIR::from_xi(g, &xi_prev[1..], device);

        let mut round_polys: Vec<[BufId; crate::logup_zerocheck::fractional_ir::GKR_S_DEG]> =
            Vec::with_capacity(round);
        let mut r_vec: Vec<BufId> = Vec::with_capacity(round);

        // Inner sumcheck rounds: t = 0, 1, ..., round - 1. Each does
        //   compute → observe → sample r → fold pq_buffer by r.
        #[allow(clippy::needless_range_loop)]
        for t in 0..round {
            let d_sum = add_ef_buf(
                g,
                device,
                &format!("d_sum_v2_{round}_{t}"),
                crate::logup_zerocheck::fractional_ir::GKR_S_DEG - 1,
            );
            let (eq_low, eq_high, eq_low_cap) = eq_layer_bufs(&eq_buffer, pq_size / 2);
            frac_compute_round_ir_dsl(
                g,
                eq_low,
                eq_high,
                pq_buffer,
                lambda,
                d_sum,
                pq_size / 2,
                eq_low_cap,
            );
            eq_buffer.drop_layer();

            let out = observe_and_update_ir(
                g,
                transcript,
                d_sum,
                prev_s_eval,
                xi_prev[t],
                eq_r_acc,
                device,
            );
            round_polys.push(out.s_evals);
            r_vec.push(out.r);
            prev_s_eval = out.prev_s_eval;
            eq_r_acc = out.eq_r_acc;

            // Fold by r_t → new half-sized buffer.
            let new_size = pq_size / 2;
            let new_pq = add_frac_ef_buf(
                g,
                device,
                &format!("pq_folded_v2_{round}_{t}"),
                new_size,
            );
            fold_ef_frac_columns_ir_dsl(g, pq_buffer, new_pq, pq_size, out.r);
            pq_buffer = new_pq;
            pq_size = new_size;
        }

        // Extract next-layer claim at (0, 1) of the size-2 folded buffer.
        debug_assert_eq!(pq_size, 2);
        let claim =
            extract_claim_pair_ir(g, pq_buffer, pq_size, 1, &format!("claim_v2_{round}"), device);
        claims_per_layer.push(claim);
        for buf in claim.as_array() {
            transcript.observe_ext(g, buf);
        }

        let mu = transcript.sample_ext(g);
        xi_prev = std::iter::once(mu).chain(r_vec).collect();
        sumcheck_polys.push(round_polys);
    }

    Ok(FracSumcheckProofIR {
        fractional_sum: (root_p, root_q),
        claims_per_layer,
        sumcheck_polys,
        final_randomness: xi_prev,
    })
}

/// Local mirror of `super::fractional_ir::eq_layer_bufs` (which is
/// `pub(crate)`). Resolves the current top eq-layer buffers of a
/// [`SqrtEqLayersIR`] for a round of `num_x` terms.
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

// ---------------------------------------------------------------------------
// Tests.

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use crypto_compiler::{
        graph_exe::GraphCompiler,
        graph_ir::{ConstBuf, DeviceType, GraphBuilder},
        planner::SchedulerMode,
        runtime::CompileOptions,
    };
    use openvm_cuda_common::{
        common::get_device,
        d_buffer::DeviceBuffer,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{
        logup_zerocheck::{
            fractional::{fractional_sumcheck_gpu, FractionalInputSize},
            fractional_ir::{add_frac_ef_buf, GKR_S_DEG},
        },
        prelude::SC,
        sponge::DuplexSpongeGpu,
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

    fn frac_bytes(leaves: &[Frac<EF>]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(leaves.as_ptr() as *const u8, std::mem::size_of_val(leaves))
        }
    }

    fn frac_const_buf(g: &mut GraphBuilder, name: &str, leaves: &[Frac<EF>]) -> BufId {
        let device = DeviceType::Cuda(0);
        let init = add_frac_ef_buf(g, device, &format!("{name}_init"), leaves.len());
        g.insert_const(init, ConstBuf::HostBuf(frac_bytes(leaves).to_vec()));
        let buf = add_frac_ef_buf(g, device, name, leaves.len());
        g.insert_memcpy(init, buf);
        buf
    }

    fn ef_from_bytes(bytes: &[u8]) -> EF {
        assert_eq!(bytes.len(), size_of::<EF>());
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const EF) }
    }

    fn leaves_to_device(leaves: &[Frac<EF>], ctx: &GpuDeviceCtx) -> DeviceBuffer<Frac<EF>> {
        leaves.to_device_on(ctx).expect("H2D copy")
    }

    #[allow(clippy::type_complexity)]
    fn run_irv2_sumcheck(
        leaves: &[Frac<EF>],
        alpha: EF,
        ctx: &GpuDeviceCtx,
    ) -> ((EF, EF), Vec<[EF; 4]>, Vec<Vec<[EF; GKR_S_DEG]>>, Vec<EF>) {
        let device = DeviceType::Cuda(0);
        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let leaves_buf = frac_const_buf(&mut g, "leaves_v2", leaves);
        let proof = fractional_sumcheck_gpu_irv2(
            &mut g,
            &mut transcript,
            leaves_buf,
            leaves.len(),
            alpha,
            false,
            device,
        )
        .expect("fractional_sumcheck_gpu_irv2");

        // Every artifact has in-graph readers (transcript observes); memcpy
        // each into a fresh reader-less buffer so the graph exposes it.
        let mut exports: Vec<BufId> = Vec::new();
        let export = |g: &mut GraphBuilder, src: BufId, name: String| -> BufId {
            let out = add_ext_scalar_buf(g, device, &name);
            g.insert_memcpy(src, out);
            out
        };
        let (rp, rq) = proof.fractional_sum;
        exports.push(export(&mut g, rp, "out_root_p_v2".to_string()));
        exports.push(export(&mut g, rq, "out_root_q_v2".to_string()));
        for (i, claim) in proof.claims_per_layer.iter().enumerate() {
            for (k, buf) in claim.as_array().into_iter().enumerate() {
                exports.push(export(&mut g, buf, format!("out_claim_v2_{i}_{k}")));
            }
        }
        for (i, layer_polys) in proof.sumcheck_polys.iter().enumerate() {
            for (j, s) in layer_polys.iter().enumerate() {
                for (k, &buf) in s.iter().enumerate() {
                    exports.push(export(&mut g, buf, format!("out_s_v2_{i}_{j}_{k}")));
                }
            }
        }
        for (i, &buf) in proof.final_randomness.iter().enumerate() {
            exports.push(export(&mut g, buf, format!("out_xi_v2_{i}")));
        }

        let mut exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::Heuristic)
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");
        let inputs: Vec<DeviceBuffer<u8>> = Vec::new();
        let mut outputs: Vec<DeviceBuffer<u8>> = (0..exe.num_outputs())
            .map(|i| DeviceBuffer::<u8>::with_capacity_on(exe.output_size(i), ctx))
            .collect();
        let mut scratch =
            DeviceBuffer::<u8>::with_capacity_on(exe.scratch_bytes().max(1), ctx);
        exe.run(ctx, &inputs, &mut outputs, &mut scratch)
            .expect("graph run");
        let read = |bid: BufId| -> EF {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == bid)
                .expect("output buf");
            let bytes = outputs[idx].to_host_on(ctx).expect("D2H");
            ef_from_bytes(&bytes)
        };
        let mut it = exports.into_iter();
        let fractional_sum = (read(it.next().unwrap()), read(it.next().unwrap()));
        let claims: Vec<[EF; 4]> = (0..proof.claims_per_layer.len())
            .map(|_| std::array::from_fn(|_| read(it.next().unwrap())))
            .collect();
        let polys: Vec<Vec<[EF; GKR_S_DEG]>> = proof
            .sumcheck_polys
            .iter()
            .map(|layer_polys| {
                layer_polys
                    .iter()
                    .map(|_| std::array::from_fn(|_| read(it.next().unwrap())))
                    .collect()
            })
            .collect();
        let final_randomness: Vec<EF> = proof
            .final_randomness
            .iter()
            .map(|_| read(it.next().unwrap()))
            .collect();
        assert!(it.next().is_none(), "leftover exported values");
        (fractional_sum, claims, polys, final_randomness)
    }

    fn assert_irv2_matches_eager(logical_len: usize, seed: u64) {
        use openvm_cuda_common::memory_manager::MemTracker;

        let ctx = test_ctx();
        let sizes = FractionalInputSize::dense(logical_len);
        let mut rng = StdRng::seed_from_u64(seed);
        let alpha: EF = rng.random();
        let leaves = make_host_leaves(logical_len, seed ^ 0xA5A5);

        // Eager reference.
        let mut sponge = DuplexSpongeGpu::default();
        let mut mem = MemTracker::start("test.fractional_irv2");
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

        // v2 graph-IR side.
        let (got_sum, got_claims, got_polys, got_xi) = run_irv2_sumcheck(&leaves, alpha, &ctx);

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

    /// Byte-equality test: v2 must reproduce the eager prover's proof for
    /// dense inputs. Small sizes to keep CI fast.
    #[test]
    fn fractional_sumcheck_gpu_irv2_matches_eager_dense_16() {
        assert_irv2_matches_eager(16, 0x5EED_D002);
    }

    #[test]
    fn fractional_sumcheck_gpu_irv2_matches_eager_dense_32() {
        assert_irv2_matches_eager(32, 0x5EED_D003);
    }

    #[test]
    fn fractional_sumcheck_gpu_irv2_matches_eager_dense_64() {
        assert_irv2_matches_eager(64, 0x5EED_D004);
    }

    /// Larger correctness case: 2^10 leaves. Slower (a few seconds to build
    /// the graph + compile ~30 kernels + run) so run it single-threaded.
    #[test]
    fn fractional_sumcheck_gpu_irv2_matches_eager_dense_1024() {
        assert_irv2_matches_eager(1024, 0x5EED_D010);
    }

    /// Dump the v2 graph as text (both the pre-compile `GraphBuilder` and the
    /// planner-scheduled `GraphExe`), plus per-module HIR/KIR/CUDA for every
    /// unique DSL module. Ignored by default so it doesn't fire in CI.
    ///
    /// Input size via `FRAC_V2_DUMP_LOG_N` (log2 leaf count, default 6);
    /// output dir via `CRYPTO_COMPILER_DUMP_IR` (default `target/ir_dump_v2/`).
    ///
    /// Run:
    ///
    /// ```sh
    /// cargo test -p openvm-cuda-backend --lib --features graph-ir --release \
    ///     dump_fractional_sumcheck_gpu_irv2_graph -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "artifact generator; run explicitly to produce IR dumps"]
    fn dump_fractional_sumcheck_gpu_irv2_graph() {
        let _ctx = test_ctx();

        let log_n: usize = std::env::var("FRAC_V2_DUMP_LOG_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        let n = 1usize << log_n;
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_11D0);
        let alpha: EF = rng.random();
        let leaves = make_host_leaves(n, 0x5EED_11D1);

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let leaves_in = frac_const_buf(&mut g, "leaves_v2", &leaves);
        let proof = fractional_sumcheck_gpu_irv2(
            &mut g,
            &mut transcript,
            leaves_in,
            n,
            alpha,
            false,
            device,
        )
        .expect("fractional_sumcheck_gpu_irv2");
        let (root_p, root_q) = proof.fractional_sum;
        for (src, name) in [(root_p, "out_root_p_v2"), (root_q, "out_root_q_v2")] {
            let out = add_ext_scalar_buf(&mut g, device, name);
            g.insert_memcpy(src, out);
        }

        let dir = std::env::var_os("CRYPTO_COMPILER_DUMP_IR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/ir_dump_v2")
            });
        std::fs::create_dir_all(&dir).expect("create dump dir");

        // GraphBuilder text dump — the ordered node list before planning.
        // Cheap: no JIT, just a `String::push_str` walk over the node list.
        std::fs::write(
            dir.join(format!("fractional_sumcheck_v2_n{n}.graph.txt")),
            g.print(),
        )
        .expect("write graph dump");
        println!(
            "v2 graph dump written to {} ({} nodes)",
            dir.display(),
            g.nodes.len()
        );

        // At very large `n` the full compile (~hundreds of unique nvcc
        // kernels + planner) is prohibitively expensive; skip it when
        // `FRAC_V2_DUMP_GRAPH_ONLY=1`. Otherwise emit the planner-scheduled
        // exe dump plus per-module HIR/KIR/CU dumps.
        let graph_only = std::env::var_os("FRAC_V2_DUMP_GRAPH_ONLY").is_some();
        if graph_only {
            println!("(FRAC_V2_DUMP_GRAPH_ONLY set: skipping compile + exe dump)");
            return;
        }

        let exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::Heuristic)
            .compile_options(CompileOptions {
                dump_ir: Some(dir.clone()),
                ..CompileOptions::default()
            })
            .compile(g)
            .expect("graph compile");

        // GraphExe text dump — planner-ordered nodes with concrete scratch
        // offsets, plus per-node module hashes.
        std::fs::write(
            dir.join(format!("fractional_sumcheck_v2_n{n}.exe.txt")),
            exe.print(),
        )
        .expect("write exe dump");

        println!(
            "  nodes: {}, unique modules: {}, scratch pool: {} bytes",
            exe.num_unique_modules() + exe.num_cached_modules(),
            exe.num_unique_modules(),
            exe.scratch_bytes(),
        );
    }

    /// End-to-end wall-time benchmark of v2 (the DSL-first, no-PrecomputeM,
    /// full-tree prover) against the eager `fractional_sumcheck_gpu`.
    ///
    /// Sizes controlled by `FRAC_V2_BENCH_LOG_N` (comma-separated `log2`
    /// leaf counts; default `16,20,22,24`). Byte-equality on `fractional_sum`
    /// is asserted per size — the bench doubles as a correctness check.
    ///
    /// Run explicitly:
    ///
    /// ```sh
    /// cargo nextest run -p openvm-cuda-backend --features graph-ir \
    ///     --run-ignored all --no-capture \
    ///     -E 'test(bench_fractional_sumcheck_eager_vs_irv2)'
    /// ```
    #[test]
    #[ignore = "benchmark; run explicitly with --run-ignored"]
    fn bench_fractional_sumcheck_eager_vs_irv2() {
        use std::time::Instant;

        use crypto_compiler::graph_exe::GraphExe;
        use openvm_cuda_common::memory_manager::MemTracker;

        const ITERS: usize = 5;
        const WARMUPS: usize = 2;

        struct PerSize {
            log_n: usize,
            n: usize,
            leaves: Vec<Frac<EF>>,
            alpha: EF,
            eager_sum: (EF, EF),
            exe: GraphExe,
            inputs: Vec<DeviceBuffer<u8>>,
            outputs: Vec<DeviceBuffer<u8>>,
            scratch: DeviceBuffer<u8>,
            root_exports: [BufId; 2],
            build_ms: f64,
            compile_ms: f64,
            n_nodes: usize,
            eager_ms: Vec<f64>,
            graph_ms: Vec<f64>,
        }

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let log_ns: Vec<usize> = std::env::var("FRAC_V2_BENCH_LOG_N")
            .unwrap_or_else(|_| "16,20,22,24".into())
            .split(',')
            .map(|s| s.trim().parse().expect("FRAC_V2_BENCH_LOG_N entry"))
            .collect();

        // Setup pass: build/compile graph + warmups happen OUTSIDE the timing
        // loop so first-touch JIT / driver init costs don't leak in.
        let mut states: Vec<PerSize> = Vec::with_capacity(log_ns.len());
        for log_n in log_ns {
            let n = 1usize << log_n;
            let leaves = make_host_leaves(n, 0x5EED_BE9C ^ log_n as u64);
            let mut rng = StdRng::seed_from_u64(0xA1FA ^ log_n as u64);
            let alpha: EF = rng.random();

            println!("\n=== v2 fractional sumcheck: n = 2^{log_n} = {n} leaves ===");

            // Eager warmup — computes the reference `eager_sum`.
            let eager_sum = {
                let d_leaves = leaves_to_device(&leaves, &ctx);
                let mut sponge = DuplexSpongeGpu::default();
                let mut mem = MemTracker::start("bench.fractional_v2_eager");
                ctx.stream.synchronize().expect("sync");
                let (proof, _xi) = fractional_sumcheck_gpu::<SC, _>(
                    &mut sponge,
                    d_leaves,
                    FractionalInputSize::dense(n),
                    alpha,
                    false,
                    &mut mem,
                    &ctx,
                )
                .expect("eager warmup");
                ctx.stream.synchronize().expect("sync");
                proof.fractional_sum
            };

            // v2 graph build.
            let t0 = Instant::now();
            let mut g = GraphBuilder::new();
            let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
            let leaves_in = add_frac_ef_buf(&mut g, device, "leaves_in_v2", n);
            let proof_ir = fractional_sumcheck_gpu_irv2(
                &mut g,
                &mut transcript,
                leaves_in,
                n,
                alpha,
                false,
                device,
            )
            .expect("fractional_sumcheck_gpu_irv2");
            let (root_p, root_q) = proof_ir.fractional_sum;
            let root_exports: [BufId; 2] = [
                {
                    let out = add_ext_scalar_buf(&mut g, device, "out_root_p_v2");
                    g.insert_memcpy(root_p, out);
                    out
                },
                {
                    let out = add_ext_scalar_buf(&mut g, device, "out_root_q_v2");
                    g.insert_memcpy(root_q, out);
                    out
                },
            ];
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
                "graph build: {build_ms:>8.2} ms ({n_nodes} nodes); \
                 compile: {compile_ms:>8.2} ms ({} unique modules, {} loaded from cache, \
                 scratch pool {} bytes)",
                exe.num_unique_modules(),
                exe.num_cached_modules(),
                exe.scratch_bytes(),
            );

            assert_eq!(exe.num_inputs(), 1);
            let d_input = frac_bytes(&leaves).to_device_on(&ctx).expect("H2D");
            let inputs: Vec<DeviceBuffer<u8>> = vec![d_input];
            let mut outputs: Vec<DeviceBuffer<u8>> = (0..exe.num_outputs())
                .map(|i| DeviceBuffer::<u8>::with_capacity_on(exe.output_size(i), &ctx))
                .collect();
            let mut scratch =
                DeviceBuffer::<u8>::with_capacity_on(exe.scratch_bytes().max(1), &ctx);

            // Graph warmups.
            for _ in 0..WARMUPS {
                ctx.stream.synchronize().expect("sync");
                exe.run(&ctx, &inputs, &mut outputs, &mut scratch)
                    .expect("graph warmup");
                ctx.stream.synchronize().expect("sync");
            }

            states.push(PerSize {
                log_n,
                n,
                leaves,
                alpha,
                eager_sum,
                exe,
                inputs,
                outputs,
                scratch,
                root_exports,
                build_ms,
                compile_ms,
                n_nodes,
                eager_ms: Vec::with_capacity(ITERS),
                graph_ms: Vec::with_capacity(ITERS),
            });
        }

        // Timed pass.
        for st in states.iter_mut() {
            for _ in 0..ITERS {
                let d_leaves = leaves_to_device(&st.leaves, &ctx);
                let mut sponge = DuplexSpongeGpu::default();
                let mut mem = MemTracker::start("bench.fractional_v2_eager");
                ctx.stream.synchronize().expect("sync");
                let t0 = Instant::now();
                let (proof, _xi) = fractional_sumcheck_gpu::<SC, _>(
                    &mut sponge,
                    d_leaves,
                    FractionalInputSize::dense(st.n),
                    st.alpha,
                    false,
                    &mut mem,
                    &ctx,
                )
                .expect("eager");
                ctx.stream.synchronize().expect("sync");
                st.eager_ms.push(t0.elapsed().as_secs_f64() * 1e3);
                st.eager_sum = proof.fractional_sum;
            }
            for _ in 0..ITERS {
                ctx.stream.synchronize().expect("sync");
                let t0 = Instant::now();
                st.exe
                    .run(&ctx, &st.inputs, &mut st.outputs, &mut st.scratch)
                    .expect("graph run");
                ctx.stream.synchronize().expect("sync");
                st.graph_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }

        // Report + sanity check.
        for st in &states {
            let median = |v: &[f64]| {
                let mut w = v.to_vec();
                w.sort_by(|a, b| a.partial_cmp(b).unwrap());
                w[w.len() / 2]
            };
            let eager_median = median(&st.eager_ms);
            let graph_median = median(&st.graph_ms);
            println!(
                "\n--- fractional sumcheck v2: n = 2^{} = {} leaves ---\n\
                 eager (median): {:.2} ms   raw: {:.2?}\n\
                 v2 build: {:.2} ms ({} nodes); compile: {:.2} ms\n\
                 v2 exec (median): {:.2} ms   raw: {:.2?}   ratio: {:.3}× eager",
                st.log_n,
                st.n,
                eager_median,
                st.eager_ms,
                st.build_ms,
                st.n_nodes,
                st.compile_ms,
                graph_median,
                st.graph_ms,
                graph_median / eager_median,
            );

            let read_export = |bid: BufId| -> EF {
                let idx = (0..st.exe.num_outputs())
                    .find(|&i| st.exe.output_buf_id(i) == bid)
                    .expect("export output index");
                ef_from_bytes(&st.outputs[idx].to_host_on(&ctx).expect("D2H"))
            };
            let got_sum = (
                read_export(st.root_exports[0]),
                read_export(st.root_exports[1]),
            );
            assert_eq!(
                got_sum, st.eager_sum,
                "fractional_sum mismatch at 2^{}",
                st.log_n
            );
        }
    }
}
