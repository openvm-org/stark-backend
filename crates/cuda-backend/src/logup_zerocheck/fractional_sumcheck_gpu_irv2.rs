//! Alternate DSL-first port of the fractional-GKR sumcheck prover.
//!
//! Compared with [`super::fractional_ir::fractional_sumcheck_gpu_ir`] (v1),
//! this v2 driver deliberately simplifies the graph structure:
//!
//! 1. **No PrecomputeM.** Only the FoldEval strategy is emitted.
//! 2. **No in-place reverts.** The segment tree is materialized layer by layer as separate
//!    persistent buffers (`tree_layers[k]` has `2^k` `Frac<EF>` elements). Every outer round reads
//!    its input pq buffer straight from `tree_layers[round + 1]` — no `frac_build_tree_layer`
//!    revert nodes anywhere in the graph.
//! 3. **Dense-only.** `real_len == logical_len`. Virtual/compact mode is unsupported; asserted at
//!    the entry point.
//! 4. **DSL-first — every kernel node.** No blackbox kernels at all. Tree combines via a small
//!    forward-combine DSL module added here ([`build_frac_tree_layer_forward_module`]); the
//!    leaves-preparation step (bit-reversal + alpha shift) is a single DSL kernel
//!    ([`build_bit_rev_and_alpha_module`]) — bit-reverse is quasi-affine when written as
//!    `Σ_{b=0..k} ((i mod 2^{b+1}) / 2^b) · 2^{k-1-b}` (see [`bit_rev_expr`]), which the DSL's
//!    index-expression checker accepts. Compute rounds go through [`frac_compute_round_ir_dsl`],
//!    folds via [`fold_ef_frac_columns_ir_dsl`], scalar helpers via
//!    [`reduce_to_single_evaluation_ir`] / [`claim_combine_ir`] / [`observe_and_update_ir`].
//!
//! Proofs emitted by v2 are byte-identical to those of the eager
//! [`super::fractional::fractional_sumcheck_gpu`] (and hence v1) for the
//! dense case: the segment-tree layer values are mathematically the same
//! (v1's in-place forward-build + top revert produces the same layer values
//! v2 stores per layer; see the module tests below), and the FoldEval
//! sumcheck reads the same layer positions in the same order.

use crypto_compiler::{
    graph_ir::{BufId, DeviceType, GraphBuilder},
    ir::{IRBuilder, Module, NodeId, ScalarType, SizeExpr},
};
use p3_field::PrimeCharacteristicRing;
use p3_util::log2_strict_usize;

// (`PrimeCharacteristicRing` is used below for `EF::ONE` via the trait.)
use super::{
    errors::FractionalSumcheckError,
    fractional_ir::{
        add_frac_ef_buf, claim_combine_ir, ef_const_ext_scalar_buf, extract_claim_pair_ir,
        extract_root_pq_ir, reduce_to_single_evaluation_ir, FracSumcheckProofIR, GkrLayerClaimIR,
        SqrtEqLayersIR,
    },
    fractional_ir_dsl::{
        bind_challenge_as_fpext, fold_ef_frac_columns_ir_dsl, frac_compute_round_ir_dsl,
    },
};
use crate::{
    logup_zerocheck::fractional_ir::{add_ef_buf, add_ext_scalar_buf, observe_and_update_ir},
    prelude::EF,
    sponge_graph_ir::FiatShamirTranscriptGraphIR,
};

// ---------------------------------------------------------------------------
// DSL bit-reversal helper.

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
///
/// `alpha` is a `[D_EF] BabyBear` input buffer (not a baked constant), so
/// the module hash — and hence the JIT cache entry — is stable across
/// alphas. `n` stays concrete: the bit-reverse expression has `log2(n)`
/// terms, so the module structure itself depends on it.
pub fn build_bit_rev_and_alpha_module(n: usize) -> Module {
    assert!(
        n >= 2 && n.is_power_of_two(),
        "bit_rev_and_alpha: n must be a power of two >= 2, got {n}"
    );
    let k = log2_strict_usize(n);
    let mut b = IRBuilder::new();
    let leaves = b.input("leaves", ScalarType::FpExt, vec![n, 2]);
    let alpha = bind_challenge_as_fpext(&mut b, "alpha");
    let body = b.compute(n, move |b, i| {
        let br = bit_rev_expr(b, i, k);
        let idx0 = b.const_u32(0);
        let idx1 = b.const_u32(1);
        let p = b.index(leaves, &[br, idx0]);
        let q_raw = b.index(leaves, &[br, idx1]);
        let q = b.add(q_raw, alpha);
        b.pack(&[p, q])
    });
    b.finish(format!("bit_rev_and_alpha_dsl_{n}"), body)
}

/// Insert the DSL bit-rev + alpha-shift kernel: fills `layer_out` (size `n`
/// `Frac<EF>`) with `leaves` in bit-reversed order and `alpha` (an EF-scalar
/// `[D_EF] BabyBear` buffer) added to every `q` slot.
pub fn bit_rev_and_alpha_ir_dsl(
    g: &mut GraphBuilder,
    leaves: BufId,
    layer_out: BufId,
    n: usize,
    alpha: BufId,
) {
    g.insert_kernel(
        build_bit_rev_and_alpha_module(n),
        [leaves, alpha],
        [layer_out],
        &[],
    );
}

// ---------------------------------------------------------------------------
// Forward tree-combine DSL module.

/// Build a DSL module for one forward tree-layer combine.
///
/// Reads `layer_in : [h*2, 2] FpExt` (a `Frac<EF>` buffer) and produces
/// `layer_out : [h, 2] FpExt` where each output row `i` is the fractional
/// addition of `(layer_in[i], layer_in[i + h])`.
///
/// Fully symbolic over `h = layer_in_size / 2`: `h` is inferred from the
/// bound input buffer at [`GraphBuilder::insert_kernel`] and survives as a
/// runtime parameter, so every tree layer shares ONE compiled kernel.
///
/// Dense-only.
pub fn build_frac_tree_layer_forward_module() -> Module {
    let mut b = IRBuilder::new();
    let h = b.symbol("h");
    let layer_in = b.input(
        "layer_in",
        ScalarType::FpExt,
        vec![SizeExpr::from(h * 2), 2usize.into()],
    );

    let body = b.compute(h, move |b, i| {
        let zero_c = b.const_u32(0);
        let one_c = b.const_u32(1);
        let half_c = b.const_sym(h);
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
    b.finish("frac_tree_layer_forward_dsl", body)
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
    assert!(
        layer_in_size >= 2 && layer_in_size.is_power_of_two(),
        "tree forward: layer_in_size must be a power of two >= 2, got {layer_in_size}"
    );
    g.insert_kernel(
        build_frac_tree_layer_forward_module(),
        [layer_in],
        [layer_out],
        &[],
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
/// 1. Bit-reverse `leaves` and add `alpha` to every `q` slot, into `layer[N]` — one DSL kernel
///    ([`bit_rev_and_alpha_ir_dsl`]).
/// 2. Emit `N` forward tree-combine DSL kernels to materialize `layer[N-1] .. layer[0]` (each half
///    the size of its parent).
/// 4. Observe `root.p` (unless `assert_zero`) and `root.q` from `layer[0]`.
/// 5. Extract first claim from `layer[N-1]` (size 2). Observe claim, sample `mu_1`. Seed `xi_prev =
///    [mu_1]`.
/// 6. Outer GKR loop for `round in 1..N`:
///    - Read `layer[round + 1]` (size `2^{round + 1}`) as `pq_buffer`.
///    - Sample `lambda_{round + 1}`.
///    - Seed `prev_s_eval` via [`reduce_to_single_evaluation_ir`] + [`claim_combine_ir`] on the
///      previous layer's claims.
///    - Build `SqrtEqLayersIR` from `&xi_prev[1..]`.
///    - Inner sumcheck rounds `t in 0..round`:
///      * Compute `d_sum` via [`frac_compute_round_ir_dsl`] on the current pq buffer.
///      * Drop one eq layer.
///      * [`observe_and_update_ir`] with `xi_j = xi_prev[t]` → sample `r_t`, push `s_evals`, update
///        `prev_s_eval` and `eq_r_acc`.
///      * Fold pq buffer by `r_t` via [`fold_ef_frac_columns_ir_dsl`] into a fresh half-sized
///        buffer (used in the next inner round OR by the final claim extraction).
///    - Extract layer claim at positions `(0, 1)` of the final folded buffer (size 2). Observe.
///      Sample `mu_{round + 1}`. Update `xi_prev`.
///
/// # Byte-equality with the eager prover
///
/// Every transcript observe / sample is emitted in the same order and with
/// the same values as v1 (and hence the eager `fractional_sumcheck_gpu`):
/// - Tree building writes the same "layer j values" v1's in-place tree + top-revert produces — see
///   the module docs for the derivation.
/// - The `frac_compute_round_ir_dsl` module reads the same four positions `(idx, idx + q, idx + h,
///   idx + h + q)` in the size-`2^{round+1}` layer that v1's `compute_round_and_revert` reads (v1
///   reverts `[0, h)` on the fly; v2 reads the same values directly from `tree_layers[round + 1]`).
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

    // Layer N: bit-reversed leaves with alpha added to every `q` slot.
    // One DSL kernel — the bit-reverse permutation is quasi-affine per
    // `build_bit_rev_and_alpha_module`. Alpha rides in as a const EF-scalar
    // buffer so the kernel is reusable across alphas.
    tree[total_rounds] = add_frac_ef_buf(g, device, "tree_layer_N", logical_len);
    let alpha_buf = ef_const_ext_scalar_buf(g, device, "frac_alpha_v2", alpha);
    bit_rev_and_alpha_ir_dsl(g, leaves, tree[total_rounds], logical_len, alpha_buf);

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
        let (numer, denom) = reduce_to_single_evaluation_ir(
            g,
            *claims_per_layer.last().unwrap(),
            xi_prev[0],
            device,
        );
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
            let new_pq = add_frac_ef_buf(g, device, &format!("pq_folded_v2_{round}_{t}"), new_size);
            fold_ef_frac_columns_ir_dsl(g, pq_buffer, new_pq, pq_size, out.r);
            pq_buffer = new_pq;
            pq_size = new_size;
        }

        // Extract next-layer claim at (0, 1) of the size-2 folded buffer.
        debug_assert_eq!(pq_size, 2);
        let claim = extract_claim_pair_ir(
            g,
            pq_buffer,
            pq_size,
            1,
            &format!("claim_v2_{round}"),
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
        graph_ir::{DeviceType, GraphBuilder},
        passes::fusion::FusionOptions,
        planner::SchedulerMode,
    };
    use openvm_cuda_common::{
        common::get_device,
        copy::{cuda_memcpy_on, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{
        logup_zerocheck::{
            frac_bench_utils::{
                cc_compiler, cc_graph_dump_path, frac_log_n_single, frac_log_ns,
                load_or_compile_and_dump,
            },
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

    #[link(name = "cudart")]
    extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
        fn cudaDeviceSynchronize() -> i32;
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

    fn ef_from_bytes(bytes: &[u8]) -> EF {
        assert_eq!(bytes.len(), size_of::<EF>());
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const EF) }
    }

    fn leaves_to_device(leaves: &[Frac<EF>], ctx: &GpuDeviceCtx) -> DeviceBuffer<Frac<EF>> {
        leaves.to_device_on(ctx).expect("H2D copy")
    }

    /// Everything the v2 GKR sumcheck graph produces at build time. The
    /// leaves buffer is already registered as the graph's single input
    /// (fed via `exe.set_input(0, ...)` at run time); callers only need
    /// to decide how to expose the `proof` buffers — direct
    /// `register_output` for observability, or a memcpy-into-fresh-buffer
    /// indirection that keeps producers fusable through the transcript-
    /// observe path.
    struct V2GraphBundle {
        g: GraphBuilder,
        proof: FracSumcheckProofIR,
        /// Alpha value baked into the sumcheck at build time. Derived
        /// deterministically from `log_n` so every caller for the same
        /// `log_n` gets an identical graph (identical kernel-cache hits)
        /// — exposed so callers running an eager reference can pass the
        /// same alpha to their eager code path.
        alpha: EF,
    }

    /// Builds the v2 GKR sumcheck graph for `n = 2^log_n` leaves and
    /// registers the leaves buffer as the single graph input. Alpha is
    /// derived deterministically from `log_n`. Callers hand the leaves
    /// bytes to the compiled [`GraphExe`] at run time via `set_input`.
    fn build_v2_graph(log_n: usize) -> V2GraphBundle {
        let n = 1usize << log_n;
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5EED_11D0_u64.wrapping_add(log_n as u64));
        let alpha: EF = rng.random();
        let mut g = GraphBuilder::new();
        let leaves_in = add_frac_ef_buf(&mut g, device, "leaves_v2", n);
        g.register_input(leaves_in);
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
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
        V2GraphBundle { g, proof, alpha }
    }

    /// Direct-registration pattern: register every `FracSumcheckProofIR`
    /// buffer as a graph output. The producers of these buffers become
    /// pinned interface writers and can no longer be fused with their
    /// downstream transcript-observe consumers — use only when observing
    /// the raw producer output is more important than that specific
    /// fusion opportunity (dump / benchmark, where readback shape
    /// simplicity wins).
    fn register_all_proof_outputs(g: &mut GraphBuilder, proof: &FracSumcheckProofIR) {
        let (rp, rq) = proof.fractional_sum;
        g.register_output(rp);
        g.register_output(rq);
        for claim in &proof.claims_per_layer {
            for buf in claim.as_array() {
                g.register_output(buf);
            }
        }
        for layer_polys in &proof.sumcheck_polys {
            for s in layer_polys {
                for &buf in s {
                    g.register_output(buf);
                }
            }
        }
        for &buf in &proof.final_randomness {
            g.register_output(buf);
        }
    }

    #[allow(clippy::type_complexity)]
    fn run_irv2_sumcheck(
        log_n: usize,
        leaves: &[Frac<EF>],
        ctx: &GpuDeviceCtx,
    ) -> (
        EF, // alpha the graph was built with
        (EF, EF),
        Vec<[EF; 4]>,
        Vec<Vec<[EF; GKR_S_DEG]>>,
        Vec<EF>,
    ) {
        assert_eq!(
            leaves.len(),
            1usize << log_n,
            "leaves.len() must equal 2^log_n"
        );
        let device = DeviceType::Cuda(0);
        let V2GraphBundle {
            mut g,
            proof,
            alpha,
        } = build_v2_graph(log_n);

        // Every artifact has in-graph readers (transcript observes); memcpy
        // each into a fresh registered output buffer so the graph exposes
        // it *without* pinning the internal producer as an interface (which
        // would block its fusion with the downstream transcript-observe).
        let mut exports: Vec<BufId> = Vec::new();
        let export = |g: &mut GraphBuilder, src: BufId, name: String| -> BufId {
            let out = add_ext_scalar_buf(g, device, &name);
            g.insert_memcpy(src, out);
            g.register_output(out);
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
            .scheduler(SchedulerMode::ListV1 {
                params: crypto_compiler::planner::ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("graph compile");
        // Copy the caller's leaves into the compiled graph's registered
        // input slot. The graph itself no longer bakes leaves in via a
        // const + memcpy — leaves are supplied at run time.
        assert_eq!(exe.num_inputs(), 1);
        let d_leaves = frac_bytes(leaves).to_device_on(ctx).expect("H2D");
        exe.set_input(ctx, 0, &d_leaves).expect("set_input");
        exe.run(ctx).expect("graph run");
        let read = |bid: BufId| -> EF {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == bid)
                .expect("output buf");
            let bytes = exe.get_output(idx).to_host_on(ctx).expect("D2H");
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
        (alpha, fractional_sum, claims, polys, final_randomness)
    }

    fn assert_irv2_matches_eager(logical_len: usize, seed: u64) {
        use openvm_cuda_common::memory_manager::MemTracker;

        assert!(
            logical_len.is_power_of_two() && logical_len >= 2,
            "logical_len must be a power of two ≥ 2"
        );
        let log_n = logical_len.trailing_zeros() as usize;
        let ctx = test_ctx();
        let sizes = FractionalInputSize::dense(logical_len);
        let leaves = make_host_leaves(logical_len, seed ^ 0xA5A5);

        // v2 graph-IR side. The alpha the graph was built with is derived
        // from `log_n` inside `build_v2_graph`; we pull it back out so the
        // eager reference below runs on the same alpha.
        let (alpha, got_sum, got_claims, got_polys, got_xi) =
            run_irv2_sumcheck(log_n, &leaves, &ctx);

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
    /// Input size via `FRAC_LOG_N` (log2 leaf count, default 6); output
    /// dir via `CRYPTO_COMPILER_DUMP_IR` (default `target/ir_dump_v2/`).
    /// Fusion strategy and tunables via `CC_FUSION*` (see
    /// [`super::frac_bench_utils`]); build with
    /// `--features crypto-compiler/planner-ortools`.
    ///
    /// Setting `CC_TIMING_JSON_PATH=<file>` merges the
    /// [`crypto_compiler::graph_info::GraphInfo`] JSON at `<file>`
    /// into the fused cytoscape dump (`*.cy.fused.json`). Produce that
    /// JSON with `bench_fractional_sumcheck_eager_vs_ir` +
    /// `CC_CY_DUMP_PATH=…`; each per-node timing lines up with the
    /// fused graph's node indices when the same `FRAC_LOG_N` +
    /// `CC_FUSION*` env are supplied on both sides.
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

        let log_n: usize = frac_log_n_single(6);
        let n = 1usize << log_n;
        let device = DeviceType::Cuda(0);

        let V2GraphBundle {
            mut g,
            proof,
            alpha: _,
        } = build_v2_graph(log_n);
        // Register every user-visible proof buffer as a graph output
        // directly — no intermediate export buffer + memcpy. That pattern
        // (allocate + memcpy + register) is only necessary when the caller
        // needs to keep the producing buffer internal; for a dump we just
        // want the compiler to see the producers as outputs.
        register_all_proof_outputs(&mut g, &proof);

        let dir = std::env::var_os("CRYPTO_COMPILER_DUMP_IR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/ir_dump_v2")
            });
        std::fs::create_dir_all(&dir).expect("create dump dir");

        // Pre-fusion GraphBuilder text dump — the ordered node list before
        // any pass runs. Cheap: no JIT, just a `String::push_str` walk.
        std::fs::write(
            dir.join(format!("fractional_sumcheck_v2_n{n}.graph.txt")),
            g.print(),
        )
        .expect("write graph dump");
        // Cytoscape.js elements-JSON for `scripts/serve_graph.py`.
        std::fs::write(
            dir.join(format!("fractional_sumcheck_v2_n{n}.cy.json")),
            g.to_cytoscape_json(),
        )
        .expect("write cytoscape dump");
        let nodes_before = g.nodes.len();
        println!(
            "v2 graph dump written to {} ({nodes_before} nodes before fusion)",
            dir.display(),
        );

        // Run the normalize + fusion passes without compiling, and dump
        // the post-fusion graph alongside per-round stats. Honor
        // `CC_FUSION` for strategy selection; the dump path defaults to a
        // verbose v1 configuration so the on-disk artifacts capture the
        // per-round stats useful for offline analysis.
        let compiler = if std::env::var_os("CC_FUSION").is_some() {
            cc_compiler(device)
        } else {
            GraphCompiler::new()
                .device(device)
                .fusion_options(FusionOptions {
                    verbose: true,
                    max_iterations: 50,
                    ..FusionOptions::default()
                })
        };
        let mut g_fused = g;
        let report = compiler
            .fuse(&mut g_fused)
            .expect("fuse pass")
            .expect("fusion enabled");
        std::fs::write(
            dir.join(format!("fractional_sumcheck_v2_n{n}.graph.fused.txt")),
            g_fused.print(),
        )
        .expect("write fused graph dump");
        // Load an optional `GraphInfo` JSON — if `CC_TIMING_JSON_PATH` is
        // set, its `nodes` line up with `g_fused`'s post-fusion indices
        // (see this test's docstring) and get embedded into the cy.json.
        let timings: Option<crypto_compiler::graph_info::GraphInfo> =
            std::env::var_os("CC_TIMING_JSON_PATH").and_then(|p| {
                let path = std::path::PathBuf::from(p);
                match std::fs::read_to_string(&path) {
                    Ok(s) => match serde_json::from_str(&s) {
                        Ok(info) => {
                            println!("loaded GraphInfo from {}", path.display());
                            Some(info)
                        }
                        Err(e) => {
                            eprintln!("failed to parse GraphInfo at {}: {e}", path.display());
                            None
                        }
                    },
                    Err(e) => {
                        eprintln!("failed to read {}: {e}", path.display());
                        None
                    }
                }
            });
        std::fs::write(
            dir.join(format!("fractional_sumcheck_v2_n{n}.cy.fused.json")),
            g_fused.to_cytoscape_json_with_timings(timings.as_ref()),
        )
        .expect("write fused cytoscape dump");
        if let Some(v2) = report.v2.as_ref() {
            println!(
                "fusion v2 summary: nodes {} -> {}, generated={}, inserted={}, \
                 selected={}, rounds_run={} (inserted per round: {:?}, max_rounds_hit={}), \
                 cost cache {}h/{}m ({} sentinel failures), fallback={:?}",
                v2.nodes_before,
                v2.nodes_after,
                v2.candidates_generated,
                v2.candidates_inserted,
                v2.selected_from_solver,
                v2.rounds_run,
                v2.rounds_inserted,
                v2.max_rounds_hit,
                v2.cost_cache_hits,
                v2.cost_cache_misses,
                v2.cost_failures,
                v2.fallback_reason,
            );
        } else {
            println!(
                "fusion summary: nodes {} -> {} (deduped modules: {}), rounds: {}",
                report.nodes_before, report.nodes_after, report.deduped, report.rounds,
            );
            for stats in &report.rounds_detail {
                println!(
                    "  round {}: fused={}, dce_removed={}, nodes_after={}, est_modules={}",
                    stats.round,
                    stats.fused,
                    stats.dce_removed,
                    stats.nodes_after,
                    stats.est_modules,
                );
            }
        }

        // At very large `n` the full compile (~hundreds of unique nvcc
        // kernels + planner) is prohibitively expensive; skip it when
        // `FRAC_V2_DUMP_GRAPH_ONLY=1`. Otherwise emit the planner-scheduled
        // exe dump plus per-module HIR/KIR/CU dumps.
        let graph_only = std::env::var_os("FRAC_V2_DUMP_GRAPH_ONLY").is_some();
        if graph_only {
            println!("(FRAC_V2_DUMP_GRAPH_ONLY set: skipping compile + exe dump)");
            return;
        }

        // Compile the already-fused graph so the exe dump matches the
        // fused-graph dump we just wrote (avoids running fusion twice).
        let exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: crypto_compiler::planner::ListSchedulerV1::default(),
            })
            .without_fusion()
            .dump_dir(dir.clone())
            .compile(g_fused)
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
    /// Sizes controlled by `FRAC_LOG_N` (comma-separated `log2` leaf
    /// counts; default `16,20,22,24`). Byte-equality on `fractional_sum`
    /// is asserted per size — the bench doubles as a correctness check.
    /// Compiler tuning via `CC_*` (see [`super::frac_bench_utils`]);
    /// setting `CC_GRAPH_DUMP_PATH` caches the compiled `GraphExe` to
    /// disk and reuses it on subsequent runs.
    ///
    /// Three workloads are timed per size:
    /// - `eager`: reference `fractional_sumcheck_gpu`.
    /// - `graph`: per-node `GraphExe::run` dispatch (host-side loop over the planner-scheduled
    ///   nodes).
    /// - `graph_capture`: CUDA-graph replay of the same nodes via `GraphExe::launch_graph`. The
    ///   graph is captured + instantiated once in the setup pass, so the timed iterations only
    ///   measure the `cudaGraphLaunch` replay (host dispatch reduced to a single call).
    ///
    /// Run explicitly:
    ///
    /// ```sh
    /// cargo nextest run -p openvm-cuda-backend --features graph-ir \
    ///     --run-ignored all --no-capture \
    ///     -E 'test(bench_fractional_sumcheck_eager_vs_irv2)'
    /// ```
    ///
    /// nsys profile (env `NSYS_ENABLED=1`): a *single* cudaProfilerStart/Stop
    /// window wraps only the timed iterations across all sizes. Warmup +
    /// graph build/compile + one-time graph capture happen in the setup
    /// pass beforehand, so the profile contains only measured kernel work.
    /// One NVTX range is pushed *per iteration* (`eager n=2^{k} iter={i}`,
    /// `graph n=2^{k} iter={i}`, `graph_capture n=2^{k} iter={i}`), tightly
    /// wrapping the compute — H2D leaf uploads and eager's scalar-result
    /// readback fall outside the range. Suggested invocation for `LOG_N=20`:
    ///
    /// ```sh
    /// FRAC_LOG_N=20 NSYS_ENABLED=1 nsys profile \
    ///     --capture-range=cudaProfilerApi \
    ///     --trace=cuda,nvtx --cuda-graph-trace=node \
    ///     --gpu-metrics-devices=cuda-visible \
    ///     -o frac_v2_bench_log20 --force-overwrite=true \
    ///     cargo nextest run -p openvm-cuda-backend --features graph-ir \
    ///         --run-ignored all --no-capture \
    ///         -E 'test(bench_fractional_sumcheck_eager_vs_irv2)'
    /// ```
    #[test]
    #[ignore = "benchmark; run explicitly with --run-ignored"]
    fn bench_fractional_sumcheck_eager_vs_irv2() {
        // Reproduce the nsys profile at LOG_N=20 (matches the docstring
        // above; kept as a plain comment here so it's visible from the
        // function body without navigating up through the doc block):
        //
        //   FRAC_LOG_N=20 NSYS_ENABLED=1 nsys profile \
        //       --capture-range=cudaProfilerApi \
        //       --trace=cuda,nvtx --cuda-graph-trace=node \
        //       --gpu-metrics-devices=cuda-visible \
        //       -o frac_v2_bench_log20 --force-overwrite=true \
        //       cargo nextest run -p openvm-cuda-backend --features graph-ir \
        //           --run-ignored all --no-capture \
        //           -E 'test(bench_fractional_sumcheck_eager_vs_irv2)'
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
            root_exports: [BufId; 2],
            build_ms: f64,
            compile_ms: f64,
            capture_ms: f64,
            n_nodes: usize,
            n_nodes_post_fusion: usize,
            fusion_rounds: usize,
            fused_total: usize,
            unique_modules: usize,
            eager_ms: Vec<f64>,
            graph_ms: Vec<f64>,
            graph_capture_ms: Vec<f64>,
        }

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let log_ns: Vec<usize> = frac_log_ns("16,20,22,24");

        let nsys_enabled = std::env::var_os("NSYS_ENABLED").is_some();

        // Setup pass: build/compile graph + warmups happen OUTSIDE the timing
        // loop so first-touch JIT / driver init costs don't leak in.
        let mut states: Vec<PerSize> = Vec::with_capacity(log_ns.len());
        for log_n in log_ns {
            let n = 1usize << log_n;
            let leaves = make_host_leaves(n, 0x5EED_BE9C ^ log_n as u64);

            println!("\n=== v2 fractional sumcheck: n = 2^{log_n} = {n} leaves ===");

            // v2 graph build. The shared `build_v2_graph` also derives the
            // alpha (as a deterministic function of `log_n`); the eager
            // reference below reuses that same alpha.
            let build_graph = || {
                let t0 = Instant::now();
                let V2GraphBundle {
                    mut g,
                    proof: proof_ir,
                    alpha,
                } = build_v2_graph(log_n);
                let root_exports: [BufId; 2] =
                    [proof_ir.fractional_sum.0, proof_ir.fractional_sum.1];
                register_all_proof_outputs(&mut g, &proof_ir);
                let build_ms = t0.elapsed().as_secs_f64() * 1e3;
                let n_nodes = g.nodes.len();
                (g, (root_exports, alpha, build_ms, n_nodes))
            };
            let t_prep = Instant::now();
            let (exe, (root_exports, alpha, build_ms, n_nodes)) = load_or_compile_and_dump(
                build_graph,
                cc_compiler(device),
                &ctx,
                cc_graph_dump_path(log_n),
            );
            let compile_ms = t_prep.elapsed().as_secs_f64() * 1e3 - build_ms;
            let (n_nodes_post_fusion, fusion_rounds, fused_total) = exe
                .fusion_report()
                .map(|r| (r.nodes_after, r.rounds, r.fused.len()))
                .unwrap_or((n_nodes, 0, 0));
            let unique_modules = exe.num_unique_modules();
            println!(
                "graph build: {build_ms:>8.2} ms ({n_nodes} nodes pre-fusion, \
                 {n_nodes_post_fusion} post-fusion via {fusion_rounds} rounds, \
                 {fused_total} fusions applied); \
                 compile-or-load: {compile_ms:>8.2} ms ({unique_modules} unique modules, \
                 {} loaded from cache, scratch pool {} bytes)",
                exe.num_cached_modules(),
                exe.scratch_bytes(),
            );
            if let Some(v2) = exe.fusion_report().and_then(|r| r.v2.as_ref()) {
                println!(
                    "fusion v2: generated={}, inserted={}, selected={}, rounds={:?}, \
                     cost cache {}h/{}m, total_runtime_units={}, fallback={:?}",
                    v2.candidates_generated,
                    v2.candidates_inserted,
                    v2.selected_from_solver,
                    v2.rounds_inserted,
                    v2.cost_cache_hits,
                    v2.cost_cache_misses,
                    v2.total_runtime_units,
                    v2.fallback_reason,
                );
            }

            assert_eq!(exe.num_inputs(), 1);

            // Pool alloc, H2D of leaves, graph warmups, and graph capture
            // are deferred to the timed pass — done per-size *after* the
            // eager loop so eager's ~13 GiB of transient buffers at
            // log_n=28 don't have to coexist with the ~25 GiB graph pool
            // on a 32 GiB card. `eager_sum` and `capture_ms` start as
            // placeholders and are populated in the timed pass.
            states.push(PerSize {
                log_n,
                n,
                leaves,
                alpha,
                eager_sum: (EF::ZERO, EF::ZERO),
                exe,
                root_exports,
                build_ms,
                compile_ms,
                capture_ms: 0.0,
                n_nodes,
                n_nodes_post_fusion,
                fusion_rounds,
                fused_total,
                unique_modules,
                eager_ms: Vec::with_capacity(ITERS),
                graph_ms: Vec::with_capacity(ITERS),
                graph_capture_ms: Vec::with_capacity(ITERS),
            });
        }

        // Timed pass — wrapped in a single cudaProfilerStart/Stop window
        // when `NSYS_ENABLED=1` so nsys emits one .nsys-rep containing
        // only the labeled timed work. Setup (build, compile) already ran
        // above and is excluded; per-size eager warmup, pool alloc, graph
        // warmups, and one-shot graph capture happen inside the profile
        // window but outside NVTX ranges — the NVTX-tagged intervals are
        // exactly the timed iterations.
        if nsys_enabled {
            unsafe { cudaProfilerStart() };
        }
        // Per-size order: (1) eager warmup + eager timed iters (no graph
        // pool held), (2) bind the graph input by H2D-ing leaves directly
        // into the pool slot via `get_input_ptr`, (3) graph warmups +
        // capture + launch warmups, (4) graph timed + graph_capture
        // timed iters. Eager runs first so its ~13 GiB of transient
        // per-iter allocations (input leaves, work buffer, tmp block
        // sums, …) don't have to coexist with the ~24 GiB graph pool —
        // critical at log_n=28 on a 32 GiB card.
        for st in states.iter_mut() {
            // Eager warmup — untimed run that also sets `eager_sum` as
            // the reference the graph's fractional_sum is asserted
            // against at report time. All device buffers drop at scope
            // exit, restoring the GPU to a clean state before the pool
            // allocation below.
            {
                let d_leaves = leaves_to_device(&st.leaves, &ctx);
                let mut sponge = DuplexSpongeGpu::default();
                let mut mem = MemTracker::start("bench.fractional_v2_eager");
                ctx.stream.synchronize().expect("sync");
                let (proof, _xi) = fractional_sumcheck_gpu::<SC, _>(
                    &mut sponge,
                    d_leaves,
                    FractionalInputSize::dense(st.n),
                    st.alpha,
                    false,
                    &mut mem,
                    &ctx,
                )
                .expect("eager warmup");
                ctx.stream.synchronize().expect("sync");
                st.eager_sum = proof.fractional_sum;
            }

            for i in 0..ITERS {
                // Setup (H2D, sponge init) outside the NVTX range and outside
                // the timed window.
                let d_leaves = leaves_to_device(&st.leaves, &ctx);
                let mut sponge = DuplexSpongeGpu::default();
                let mut mem = MemTracker::start("bench.fractional_v2_eager");
                ctx.stream.synchronize().expect("sync post-H2D");
                if nsys_enabled {
                    nvtx::range_push!("eager n=2^{} iter={i}", st.log_n);
                }
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
                ctx.stream.synchronize().expect("sync post-eager");
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                st.eager_ms.push(elapsed_ms);
                st.eager_sum = proof.fractional_sum;
                let _ = unsafe { cudaDeviceSynchronize() };
            }

            // Bind the graph input by H2D-ing leaves directly into the
            // pool slot returned by `get_input_ptr`. This skips the
            // ~8 GiB device staging buffer that `set_input` would
            // require, so peak concurrent device memory during pool
            // alloc drops from `pool + staging` (~32 GiB at log_n=28)
            // to just `pool` (~24 GiB).
            ctx.stream.synchronize().expect("sync");
            let dst = st.exe.get_input_ptr(&ctx, 0).expect("get_input_ptr");
            let src = frac_bytes(&st.leaves);
            unsafe {
                cuda_memcpy_on::<false, true>(
                    dst,
                    src.as_ptr() as *const std::ffi::c_void,
                    src.len(),
                    &ctx,
                )
                .expect("H2D leaves into pool slot");
            }
            ctx.stream.synchronize().expect("sync");

            // Graph warmups.
            for _ in 0..WARMUPS {
                ctx.stream.synchronize().expect("sync");
                st.exe.run(&ctx).expect("graph warmup");
                ctx.stream.synchronize().expect("sync");
            }

            // One-time CUDA-graph capture + instantiate, then a replay
            // warmup so the timed pass measures only `cudaGraphLaunch`
            // cost. Capture is bound to the same device pool as `run`,
            // so future `launch_graph` calls resolve identical device
            // addresses (capture-stability contract enforced by
            // `GraphExe`'s planner).
            ctx.stream.synchronize().expect("sync");
            let t0 = Instant::now();
            st.exe.capture_graph(&ctx).expect("graph capture");
            ctx.stream.synchronize().expect("sync");
            st.capture_ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "graph capture: {:>8.2} ms (single cudaStreamBeginCapture + \
                 cudaGraphInstantiateWithFlags over the {}-node graph)",
                st.capture_ms, st.n_nodes_post_fusion,
            );
            for _ in 0..WARMUPS {
                ctx.stream.synchronize().expect("sync");
                st.exe.launch_graph(&ctx).expect("graph capture warmup");
                ctx.stream.synchronize().expect("sync");
            }

            // Timed graph iterations. Re-H2D the leaves into the pool
            // slot before every iteration — the graph mutates its
            // input's storage in-place (the pool doesn't preserve
            // inputs across `launch_graph` / `run` calls), so each
            // iteration needs a fresh copy. The H2D sits OUTSIDE the
            // NVTX range so nsys measures only the kernel work.
            let refresh_input = |st: &mut PerSize, ctx: &GpuDeviceCtx| {
                let dst = st.exe.get_input_ptr(ctx, 0).expect("get_input_ptr");
                let src = frac_bytes(&st.leaves);
                unsafe {
                    cuda_memcpy_on::<false, true>(
                        dst,
                        src.as_ptr() as *const std::ffi::c_void,
                        src.len(),
                        ctx,
                    )
                    .expect("H2D leaves into pool slot");
                }
                ctx.stream.synchronize().expect("sync post-H2D");
            };

            for i in 0..ITERS {
                refresh_input(st, &ctx);
                if nsys_enabled {
                    nvtx::range_push!("graph n=2^{} iter={i}", st.log_n);
                }
                let t0 = Instant::now();
                st.exe.run(&ctx).expect("graph run");
                ctx.stream.synchronize().expect("sync post-run");
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                st.graph_ms.push(elapsed_ms);
                let _ = unsafe { cudaDeviceSynchronize() };
            }
            for i in 0..ITERS {
                refresh_input(st, &ctx);
                if nsys_enabled {
                    nvtx::range_push!("graph_capture n=2^{} iter={i}", st.log_n);
                }
                let t0 = Instant::now();
                st.exe.launch_graph(&ctx).expect("graph launch");
                ctx.stream.synchronize().expect("sync post-launch");
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                st.graph_capture_ms.push(elapsed_ms);
                let _ = unsafe { cudaDeviceSynchronize() };
            }
        }
        if nsys_enabled {
            unsafe { cudaProfilerStop() };
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
            let capture_median = median(&st.graph_capture_ms);
            let peak_mib = st.exe.scratch_bytes() as f64 / (1024.0 * 1024.0);
            println!(
                "\n--- fractional sumcheck v2: n = 2^{} = {} leaves ---\n\
                 eager (median):          {:.2} ms   raw: {:.2?}\n\
                 v2 build: {:.2} ms ({} nodes pre-fusion -> {} post-fusion, \
                 {} fusions in {} rounds, {} unique modules); compile: {:.2} ms; \
                 capture: {:.2} ms\n\
                 v2 peak memory:          {} bytes ({:.2} MiB)\n\
                 v2 exec (median):        {:.2} ms   raw: {:.2?}   ratio: {:.3}× eager\n\
                 v2 capture (median):     {:.2} ms   raw: {:.2?}   \
                 ratio: {:.3}× eager, {:.3}× graph",
                st.log_n,
                st.n,
                eager_median,
                st.eager_ms,
                st.build_ms,
                st.n_nodes,
                st.n_nodes_post_fusion,
                st.fused_total,
                st.fusion_rounds,
                st.unique_modules,
                st.compile_ms,
                st.capture_ms,
                st.exe.scratch_bytes(),
                peak_mib,
                graph_median,
                st.graph_ms,
                graph_median / eager_median,
                capture_median,
                st.graph_capture_ms,
                capture_median / eager_median,
                capture_median / graph_median,
            );

            let read_export = |bid: BufId| -> EF {
                let idx = (0..st.exe.num_outputs())
                    .find(|&i| st.exe.output_buf_id(i) == bid)
                    .expect("export output index");
                ef_from_bytes(&st.exe.get_output(idx).to_host_on(&ctx).expect("D2H"))
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

    /// Single-nsys-profile bench comparing eager, heuristic, and list_v1
    /// with 1/2/3 streams — every workload wrapped in an NVTX range inside
    /// one `cudaProfilerStart/Stop` window per AGENTS.md profiling guide.
    ///
    /// Size controlled by `FRAC_LOG_N` (single value, defaults 20).
    /// Compiler tuning via `CC_*` (see [`super::frac_bench_utils`]); the
    /// stream-count sweep uses its own `CC_STREAMS_SWEEP` env var
    /// (comma-separated `max_concurrency` values, defaults `1,2,3`).
    ///
    /// Recommended invocation (matches AGENTS.md profile requirements):
    ///
    /// ```text
    /// FRAC_LOG_N=20 NSYS_ENABLED=1 nsys profile \
    ///     --capture-range=cudaProfilerApi \
    ///     --trace=cuda,nvtx --cuda-graph-trace=node \
    ///     --gpu-metrics-devices=visible \
    ///     -o frac_v2_all_schedulers --force-overwrite=true \
    ///     cargo nextest run -p openvm-cuda-backend --features graph-ir \
    ///         --run-ignored all --no-capture \
    ///         -E 'test(bench_fractional_sumcheck_all_schedulers_nsys)'
    /// ```
    #[test]
    #[ignore]
    fn bench_fractional_sumcheck_all_schedulers_nsys() {
        use std::{sync::Arc, time::Instant};

        use crypto_compiler::{
            graph_exe::GraphExe, kernel_cache::KernelCache, planner::ListSchedulerV1,
        };
        use openvm_cuda_common::memory_manager::MemTracker;

        const ITERS: usize = 3;
        const WARMUPS: usize = 2;

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let log_n: usize = frac_log_n_single(20);
        let n = 1usize << log_n;
        let nsys_enabled = std::env::var_os("NSYS_ENABLED").is_some();
        let no_fusion = matches!(std::env::var("CC_FUSION").as_deref(), Ok("off"));

        // ---------- Setup phase (excluded from cudaProfilerStart window) ----------

        println!(
            "\n=== all-schedulers nsys bench: n = 2^{log_n} = {n} leaves, \
             nsys_enabled={nsys_enabled} ==="
        );

        let leaves = make_host_leaves(n, 0x5EED_BE9C ^ log_n as u64);

        // We rebuild the graph per scheduler variant (build_v2_graph is
        // cheap, ~40 ms at log_n=20); GraphBuilder isn't Clone because it
        // holds boxed closures for blackbox kernels.
        let make_graph = |log_n: usize| {
            let V2GraphBundle {
                mut g,
                proof: proof_ir,
                alpha,
            } = build_v2_graph(log_n);
            let roots: [BufId; 2] = [proof_ir.fractional_sum.0, proof_ir.fractional_sum.1];
            register_all_proof_outputs(&mut g, &proof_ir);
            (g, roots, alpha)
        };
        // Discover the alpha + root export ids from the first build (they're
        // deterministic in log_n).
        let (_g0, root_exports, alpha) = make_graph(log_n);

        // Eager reference — computes the ground-truth sum on the same
        // (leaves, alpha) the graph was built with.
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
            .expect("eager reference");
            ctx.stream.synchronize().expect("sync");
            proof.fractional_sum
        };

        // Shared kernel cache so subsequent scheduler compiles hit warm
        // artifacts. The compiled binaries are identical across schedulers
        // (only order/stream/offset differ); after the first compile every
        // subsequent one is a cache-hit walk.
        let kernel_cache = Arc::new(
            KernelCache::new()
                .max_kernels(4096)
                .storage_size(200 * 1024 * 1024 * 1024),
        );

        // Stream counts for list_v1 sweep — env-var-driven so the same
        // test doubles as a spot-check (1,2,3) or a full sweep (1..=8).
        let stream_sweep: Vec<u32> = std::env::var("CC_STREAMS_SWEEP")
            .unwrap_or_else(|_| "1,2,3".into())
            .split(',')
            .map(|s| s.trim().parse().expect("CC_STREAMS_SWEEP entry"))
            .collect();

        let mut configs: Vec<(String, SchedulerMode)> = vec![(
            "heuristic".to_string(),
            SchedulerMode::ListV1 {
                params: crypto_compiler::planner::ListSchedulerV1::default(),
            },
        )];
        for s in &stream_sweep {
            configs.push((
                format!("listv1_s{s}"),
                SchedulerMode::ListV1 {
                    params: ListSchedulerV1 {
                        max_concurrency: *s,
                        ..ListSchedulerV1::default()
                    },
                },
            ));
        }

        struct Compiled {
            name: String,
            exe: GraphExe,
            peak_bytes: usize,
            /// Device-resident leaves, rebound via `set_input` before
            /// every timed iteration so the graph sees fresh input on
            /// each launch (the pool doesn't preserve input slots
            /// across replays).
            d_input: DeviceBuffer<u8>,
        }

        // Compile every scheduler variant, upload the input, warm each up
        // and pre-capture their CUDA graphs — all before cudaProfilerStart.
        // Fusion strategy and nvcc timeout come from `cc_compiler`; the
        // scheduler is overridden per sweep entry and the kernel cache is
        // shared across variants so only the first compile pays nvcc.
        let mut compiled: Vec<Compiled> = Vec::with_capacity(configs.len());
        for (name, mode) in configs {
            let _ = no_fusion; // recorded above; cc_compiler applied CC_FUSION already
            println!("[setup] compiling scheduler={name} ...");
            let t0 = Instant::now();
            let compiler = cc_compiler(device)
                .scheduler(mode)
                .kernel_cache(kernel_cache.clone());
            let (g, _, _) = make_graph(log_n);
            let mut exe = compiler.compile(g).expect("graph compile");
            let peak_bytes = exe.scratch_bytes();
            let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "[setup] {name}: compile={compile_ms:.1} ms, peak={peak_bytes} bytes \
                 ({:.2} MiB), unique_modules={}",
                peak_bytes as f64 / (1024.0 * 1024.0),
                exe.num_unique_modules()
            );

            // Bind input once — the exe owns a stable device slot.
            assert_eq!(exe.num_inputs(), 1);
            let d_input = frac_bytes(&leaves).to_device_on(&ctx).expect("H2D");
            exe.set_input(&ctx, 0, &d_input).expect("set_input");

            // Warmup direct exec + capture graph + warmup graph replay.
            for _ in 0..WARMUPS {
                ctx.stream.synchronize().expect("sync");
                exe.run(&ctx).expect("graph warmup");
                ctx.stream.synchronize().expect("sync");
            }
            ctx.stream.synchronize().expect("sync");
            exe.capture_graph(&ctx).expect("graph capture");
            ctx.stream.synchronize().expect("sync");
            for _ in 0..WARMUPS {
                ctx.stream.synchronize().expect("sync");
                exe.launch_graph(&ctx).expect("graph capture warmup");
                ctx.stream.synchronize().expect("sync");
            }
            compiled.push(Compiled {
                name,
                exe,
                peak_bytes,
                d_input,
            });
        }

        // Eager needs its own d_leaves per iteration (the eager solver
        // consumes the buffer), so we pre-allocate a handful up front.
        let mut eager_leaves: Vec<DeviceBuffer<Frac<EF>>> = (0..ITERS)
            .map(|_| leaves_to_device(&leaves, &ctx))
            .collect();

        // ---------- Timed phase: single profiler window over every workload ----------

        ctx.stream.synchronize().expect("sync pre-profile");
        if nsys_enabled {
            unsafe { cudaProfilerStart() };
        }

        // Eager reference workload. The per-iter H2D happens above
        // when pre-allocating `eager_leaves`, so nothing here touches
        // host memory — the NVTX range wraps kernel work only.
        for i in 0..ITERS {
            let d_leaves = eager_leaves.pop().expect("pre-alloc'd eager leaves");
            let mut sponge = DuplexSpongeGpu::default();
            let mut mem = MemTracker::start("bench.fractional_v2_eager");
            ctx.stream.synchronize().expect("sync pre-eager");
            if nsys_enabled {
                nvtx::range_push!("eager iter={i}");
            }
            let _ = fractional_sumcheck_gpu::<SC, _>(
                &mut sponge,
                d_leaves,
                FractionalInputSize::dense(n),
                alpha,
                false,
                &mut mem,
                &ctx,
            )
            .expect("eager");
            ctx.stream.synchronize().expect("sync post-eager");
            if nsys_enabled {
                nvtx::range_pop!();
            }
            let _ = unsafe { cudaDeviceSynchronize() };
        }

        // Per-scheduler: exec workload, then CUDA-graph replay workload.
        // Rebind the input via `set_input` before every iteration —
        // the pool doesn't preserve input slots across replays, so
        // each launch needs a fresh copy. The `set_input` D2D sits
        // OUTSIDE the NVTX range so nsys measures only kernel work.
        for c in compiled.iter_mut() {
            for i in 0..ITERS {
                c.exe
                    .set_input(&ctx, 0, &c.d_input)
                    .expect("set_input pre-run");
                ctx.stream.synchronize().expect("sync post-set_input");
                if nsys_enabled {
                    let label = format!("{}_exec iter={i}", c.name);
                    nvtx::range_push!("{}", label);
                }
                c.exe.run(&ctx).expect("graph run");
                ctx.stream.synchronize().expect("sync post-run");
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                let _ = unsafe { cudaDeviceSynchronize() };
            }
            for i in 0..ITERS {
                c.exe
                    .set_input(&ctx, 0, &c.d_input)
                    .expect("set_input pre-launch");
                ctx.stream.synchronize().expect("sync post-set_input");
                if nsys_enabled {
                    let label = format!("{}_graph iter={i}", c.name);
                    nvtx::range_push!("{}", label);
                }
                c.exe.launch_graph(&ctx).expect("graph launch");
                ctx.stream.synchronize().expect("sync post-launch");
                if nsys_enabled {
                    nvtx::range_pop!();
                }
                let _ = unsafe { cudaDeviceSynchronize() };
            }
        }

        if nsys_enabled {
            unsafe { cudaProfilerStop() };
        }

        // ---------- Correctness sanity check (outside profile window) ----------

        for c in &compiled {
            let read_export = |bid: BufId| -> EF {
                let idx = (0..c.exe.num_outputs())
                    .find(|&i| c.exe.output_buf_id(i) == bid)
                    .expect("export output index");
                ef_from_bytes(&c.exe.get_output(idx).to_host_on(&ctx).expect("D2H"))
            };
            let got_sum = (read_export(root_exports[0]), read_export(root_exports[1]));
            assert_eq!(
                got_sum, eager_sum,
                "fractional_sum mismatch on scheduler {}",
                c.name
            );
        }

        // Summary print.
        println!("\n=== nsys-bench summary (n = 2^{log_n}) ===");
        println!("scheduler        peak_bytes     peak_MiB");
        for c in &compiled {
            println!(
                "{:<15}  {:>10}   {:>8.2}",
                c.name,
                c.peak_bytes,
                c.peak_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }
}
