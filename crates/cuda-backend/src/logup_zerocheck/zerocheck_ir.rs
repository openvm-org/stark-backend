//! Graph-IR port of the logup-zerocheck phase
//! ([`super::prove_zerocheck_and_logup_gpu`], `mod.rs:129-445`).
//!
//! This module mirrors the eagerly-executed CUDA logup-zerocheck prover in
//! `mod.rs`, but records every kernel launch as a node on a
//! [`GraphBuilder`] instead of running it immediately — the same thing
//! [`super::fractional_ir`] does for the fractional-GKR phase. The
//! transcript is expressed through [`FiatShamirTranscriptGraphIR`], so
//! every `observe_ext` / `sample_ext` becomes a graph node as well and the
//! whole phase can be compiled and re-executed via `GraphCompiler` /
//! `GraphExe`.
//!
//! # Status — tier 1 (blackbox-first)
//!
//! Every kernel wrapper below inserts a [`GraphNode::BlackboxKernel`] node
//! that calls the same `crate::cuda::logup_zerocheck` entry point the eager
//! path calls. No `ir::Module` DSL ports live here: the fastest route to a
//! *captured* phase is one blackbox per launcher, and the structured ports
//! can replace them one at a time afterwards (that is exactly how
//! `fractional_ir.rs` → `fractional_ir_dsl.rs` → `fractional_sumcheck_gpu_irv2.rs`
//! evolved).
//!
//! The top-level driver is [`logup_zerocheck_gpu_ir`]. It mirrors stages
//! C (univariate round 0), D (the `n_max` MLE rounds) and E (column
//! openings) of the eager prover, unrolling the round loop at graph-build
//! time — the author's structure for `fractional_sumcheck_gpu_ir`.
//!
//! # Host seams
//!
//! The eager phase is not a pure kernel DAG: three host computations sit
//! *on the data path* between kernels, and no device equivalent exists yet.
//! They are the honest seams of this port, and every one of them is marked
//! with a `TODO(cc-ir)` at its site:
//!
//! 1. **Round 0's iDFT chain** (`mod.rs:270-352, 858-875, 918-946`) — there is no device `EF` iDFT,
//!    so `UnivariatePoly::from_geometric_cosets_evals_idft` and the `s_0_poly` assembly stay on the
//!    host.
//! 2. **`compute_batch_s_poly`** (`mod.rs:1430-1502`) — ~70 lines of host `EF` algebra between a
//!    round's evaluator output and its `observe_ext` / `sample_ext`. Closing this on-device is the
//!    campaign's HIGH-risk item.
//! 3. **Fiat–Shamir values.** Because of (1) and (2) the challenge *values* (`lambda`, `mu`, `r_0`,
//!    `r_round`) are known on the host at graph-build time and are captured by value into kernel
//!    closures, exactly like `fractional_ir.rs`'s plain (non-`_bufid`) wrappers. The transcript's
//!    *state* still threads through the graph as a `BufId`: [`ZerocheckPhasePlan`] supplies the
//!    host values and this module emits the matching `observe_ext` / `sample_ext` nodes in the same
//!    order, so the sponge chain in the graph is bit-identical to the eager one. Promoting the
//!    values to `BufId`s is the `_dev_challenge` follow-up (B3 §4 lists the four challenge-by-value
//!    entry points that need a `template <bool DEV_CH>` sibling).
//!
//! # `ctx`-struct buffers
//!
//! Seven of the seventeen entry points in this phase are *runtime
//! interpreters* driven by arrays of `#[repr(C)]` context structs
//! (`ZerocheckCtx`, `LogupCtx`, `MonomialAirCtx`, …) that embed raw device
//! pointers. The eager path assembles those structs on the host and uploads
//! them (`batch_mle.rs:158-201`, `batch_mle_monomial.rs:176-192`), which
//! hides every embedded pointer from the planner — a struct of host-baked
//! addresses is an opaque leaf and alias analysis through it is impossible.
//!
//! `ZerocheckCtx` and `LogupCtx` (and the `MainMatrixPtrs<EF>` descriptor
//! array they point at) are instead built **on the device**, one element per
//! node, by [`materialize_zerocheck_ctx_ir`] / [`materialize_logup_ctx_ir`] /
//! [`materialize_main_matrix_ptr_ir`]. Each pointer field is a
//! [`DevicePtrArg::Graph`] whose `BufId` is bound as a node input and
//! resolved at invocation time, so it stays a real graph edge; only
//! host-known scalars are captured by value.
//!
//! Still host-assembled, and therefore still opaque: the three *monomial* ctx
//! structs (`MonomialAirCtx`, `LogupMonomialCommonCtx`, `LogupMonomialCtx`),
//! the round-0 `*const F` main-pointer table, and the stage-D
//! `interpolate_columns` column-pointer table. Each is the same shape of hole
//! and the same shape of fix.

use std::mem::{forget, size_of};

use crypto_compiler::{
    graph_ir::{BufId, BufInfo, ConstBuf, DeviceType, GraphBuilder},
    quast::Quast,
};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
use p3_field::PrimeCharacteristicRing;

use super::fractional_ir_utils::{add_ef_buf, ef_const_ext_scalar_buf};
use crate::{
    cuda::{
        logup_zerocheck::{
            _logup_batch_mle_intermediates_buffer_size,
            _zerocheck_batch_mle_intermediates_buffer_size, fold_ple_from_evals,
            fold_selectors_round0, interpolate_columns_gpu, logup_bary_eval_interactions_round0,
            logup_batch_eval_mle, logup_monomial_batched, materialize_logup_ctx_raw,
            materialize_main_matrix_ptr_raw, materialize_zerocheck_ctx_raw,
            precompute_lambda_combinations, precompute_logup_denom_combinations,
            precompute_logup_numer_combinations, zerocheck_batch_eval_mle,
            zerocheck_monomial_batched, zerocheck_monomial_par_y_batched,
            zerocheck_ntt_eval_constraints, BlockCtx, LogupCtx, LogupMonomialCommonCtx,
            LogupMonomialCtx, MainMatrixPtrs, MonomialAirCtx, ZerocheckCtx,
        },
        poly::eq_hypercube_interleaved_stage_ext,
        sumcheck::batch_fold_mle,
    },
    monomial::{InteractionMonomialTerm, LambdaTerm, MonomialHeader},
    prelude::{EF, F},
    sponge_graph_ir::FiatShamirTranscriptGraphIR,
};

// ---------------------------------------------------------------------------
// Buffer allocation helpers.
//
// `add_ef_buf` / `add_ext_scalar_buf` / `ef_const_ext_scalar_buf` are reused
// from `super::fractional_ir_utils` rather than duplicated.

/// Byte size of a base-field element.
pub(crate) const F_BYTES: usize = size_of::<F>();
/// Byte size of an `EF` element.
#[allow(dead_code)]
pub(crate) const EF_BYTES: usize = size_of::<EF>();
/// Byte size of a `Frac<EF>` element (two `EF`s).
pub(crate) const FRAC_EF_BYTES: usize = size_of::<Frac<EF>>();

/// Allocate a device buffer of `n` base-field elements.
pub fn add_f_buf(g: &mut GraphBuilder, device: DeviceType, name: &str, n: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst((n.max(1) * F_BYTES) as i64),
        concrete_size: n.max(1) * F_BYTES,
        elem_size: F_BYTES,
    })
}

/// Allocate a device buffer of `n` `Frac<EF>` elements.
pub fn add_frac_buf(g: &mut GraphBuilder, device: DeviceType, name: &str, n: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst((n.max(1) * FRAC_EF_BYTES) as i64),
        concrete_size: n.max(1) * FRAC_EF_BYTES,
        elem_size: FRAC_EF_BYTES,
    })
}

/// Allocate a device buffer of `n` `T`s — used for the `#[repr(C)]` context
/// arrays and for the pointer tables (`*const EF`, `*mut EF`, `u32`, `u8`)
/// the batched launchers index.
pub fn add_typed_buf<T>(g: &mut GraphBuilder, device: DeviceType, name: &str, n: usize) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst((n.max(1) * size_of::<T>()) as i64),
        concrete_size: n.max(1) * size_of::<T>(),
        elem_size: size_of::<T>(),
    })
}

/// Stage a slice of `EF`s as a read-only const buffer.
pub fn ef_slice_const_buf(
    g: &mut GraphBuilder,
    device: DeviceType,
    name: &str,
    xs: &[EF],
) -> BufId {
    let buf = add_ef_buf(g, device, name, xs.len().max(1));
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)).to_vec()
    };
    g.insert_const(buf, ConstBuf::HostBuf(bytes));
    buf
}

/// Stage a slice of base-field elements as a read-only const buffer.
pub fn f_slice_const_buf(g: &mut GraphBuilder, device: DeviceType, name: &str, xs: &[F]) -> BufId {
    let buf = add_f_buf(g, device, name, xs.len().max(1));
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)).to_vec()
    };
    g.insert_const(buf, ConstBuf::HostBuf(bytes));
    buf
}

// ===========================================================================
// Blackbox kernel wrappers.
//
// Each `*_ir` function has the same rough shape as the corresponding safe
// wrapper in `crate::cuda::logup_zerocheck`: buffer arguments come in as
// `BufId`s, scalar/config arguments are captured by value in the closure,
// and the closure reconstructs `DeviceBuffer`s from the raw pointers the
// graph runtime hands it, calls the underlying kernel, and `mem::forget`s
// the wrappers so the borrowed pointers are not freed.
// ===========================================================================

// ---------------------------------------------------------------------------
// Stage C — univariate round 0.

/// Insert one `eq_hypercube_interleaved_stage_ext` stage node: doubles an
/// `eq` hypercube layer by splitting on `x_i`.
///
/// This is the interleaved index map (`poly.cu:157`), *not* the
/// nonoverlapping one [`super::fractional_ir::eq_hypercube_nonoverlapping_stage_ext_ir`]
/// wraps — `EqEvalLayers::new_rev` (`poly.rs:417-445`) is what stage C uses.
// TODO(cc-ir): `x_i` is a Fiat-Shamir challenge captured by value.
// WHY: `_eq_hypercube_interleaved_stage_ext` takes `x_i: EF` by value
//      (`cuda/poly.rs:42-48`) and has no `_dev_challenge` sibling; the
//      fractional twin was given a *structured module* instead
//      (`fractional_ir.rs:922-955`), which is the better long-term answer.
// RISK: as long as `xi` comes from the host (it does — GKR returns it to
//      the host), this is exactly what the eager path does and is bit-exact.
//      It blocks a fully device-resident Fiat-Shamir chain, nothing else.
pub fn eq_hypercube_interleaved_stage_ext_ir(
    g: &mut GraphBuilder,
    input: BufId,
    out: BufId,
    x_i: EF,
    step: u32,
) {
    g.insert_blackbox_kernel(
        "eq_hypercube_interleaved_stage_ext",
        std::iter::once(input),
        std::iter::once(out),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            eq_hypercube_interleaved_stage_ext(
                outputs[0] as *mut EF,
                inputs[0] as *const EF,
                x_i,
                step,
                stream,
            )
            .expect("eq_hypercube_interleaved_stage_ext");
        },
    );
}

/// Insert a `precompute_lambda_combinations` node
/// (`batch_mle_monomial.rs:80-100` ← `mod.rs:664`).
pub fn precompute_lambda_combinations_ir(
    g: &mut GraphBuilder,
    headers: BufId,
    lambda_terms: BufId,
    lambda_pows: BufId,
    lambda_pows_len: usize,
    out: BufId,
    num_monomials: u32,
) {
    g.insert_blackbox_kernel(
        "precompute_lambda_combinations",
        [headers, lambda_terms, lambda_pows].into_iter(),
        std::iter::once(out),
        [false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            let mut out_buf =
                DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, num_monomials as usize);
            let pows =
                DeviceBuffer::<EF>::from_raw_parts(inputs[2] as *mut EF, lambda_pows_len.max(1));
            precompute_lambda_combinations(
                &mut out_buf,
                inputs[0] as *const MonomialHeader,
                inputs[1] as *const LambdaTerm<F>,
                &pows,
                num_monomials,
                stream,
            )
            .expect("precompute_lambda_combinations");
            forget(out_buf);
            forget(pows);
        },
    );
}

/// Insert a `precompute_logup_numer_combinations` node
/// (`batch_mle_monomial.rs:529` ← `mod.rs:729`).
#[allow(clippy::too_many_arguments)]
pub fn precompute_logup_numer_combinations_ir(
    g: &mut GraphBuilder,
    headers: BufId,
    terms: BufId,
    eq_3bs: BufId,
    eq_3bs_len: usize,
    out: BufId,
    num_monomials: u32,
) {
    g.insert_blackbox_kernel(
        "precompute_logup_numer_combinations",
        [headers, terms, eq_3bs].into_iter(),
        std::iter::once(out),
        [false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            let mut out_buf =
                DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, num_monomials as usize);
            let eq = DeviceBuffer::<EF>::from_raw_parts(inputs[2] as *mut EF, eq_3bs_len.max(1));
            precompute_logup_numer_combinations(
                &mut out_buf,
                inputs[0] as *const MonomialHeader,
                inputs[1] as *const InteractionMonomialTerm<F>,
                &eq,
                num_monomials,
                stream,
            )
            .expect("precompute_logup_numer_combinations");
            forget(out_buf);
            forget(eq);
        },
    );
}

/// Insert a `precompute_logup_denom_combinations` node
/// (`batch_mle_monomial.rs:549`).
#[allow(clippy::too_many_arguments)]
pub fn precompute_logup_denom_combinations_ir(
    g: &mut GraphBuilder,
    headers: BufId,
    terms: BufId,
    beta_pows: BufId,
    beta_pows_len: usize,
    eq_3bs: BufId,
    eq_3bs_len: usize,
    out: BufId,
    num_monomials: u32,
) {
    g.insert_blackbox_kernel(
        "precompute_logup_denom_combinations",
        [headers, terms, beta_pows, eq_3bs].into_iter(),
        std::iter::once(out),
        [false, false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            let mut out_buf =
                DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, num_monomials as usize);
            let betas =
                DeviceBuffer::<EF>::from_raw_parts(inputs[2] as *mut EF, beta_pows_len.max(1));
            let eq = DeviceBuffer::<EF>::from_raw_parts(inputs[3] as *mut EF, eq_3bs_len.max(1));
            precompute_logup_denom_combinations(
                &mut out_buf,
                inputs[0] as *const MonomialHeader,
                inputs[1] as *const InteractionMonomialTerm<F>,
                &betas,
                &eq,
                num_monomials,
                stream,
            )
            .expect("precompute_logup_denom_combinations");
            forget(out_buf);
            forget(betas);
            forget(eq);
        },
    );
}

/// Shapes for one `zerocheck_ntt_eval_constraints` launch — bundled so the
/// wrapper stays under the argument-count lint without splitting the launch.
#[derive(Clone, Copy, Debug)]
pub struct Round0ZcShape {
    pub rules_len: usize,
    pub used_nodes_len: usize,
    pub lambda_pows_len: usize,
    pub buffer_size: u32,
    pub skip_domain: u32,
    pub num_x: u32,
    pub height: u32,
    pub num_cosets: u32,
    pub g_shift: F,
    pub max_temp_bytes: usize,
    pub tmp_sums_len: usize,
    pub out_len: usize,
    pub intermediates_len: usize,
    pub sels_len: usize,
    pub main_ptrs_len: usize,
    pub public_len: usize,
}

/// Buffers for one `zerocheck_ntt_eval_constraints` launch.
#[derive(Clone, Copy, Debug)]
pub struct Round0ZcBufs {
    pub tmp_sums: BufId,
    pub out: BufId,
    pub intermediates: BufId,
    pub selectors_cube: BufId,
    /// `*const F` into the preprocessed matrix; `None` for AIRs without one.
    pub preprocessed: Option<BufId>,
    /// Table of `*const F` main-matrix pointers.
    pub main_ptrs: BufId,
    pub eq_cube: BufId,
    pub lambda_pows: BufId,
    pub public_values: BufId,
    pub rules: BufId,
    pub used_nodes: BufId,
}

/// Insert the round-0 constraint evaluator (`round0.rs:127` ← `mod.rs:841`).
///
/// # Principle-1 exception: multi-launch compatibility node (T4)
///
/// `_zerocheck_ntt_eval_constraints` enqueues **two** kernels (the per-coset
/// NTT evaluator and `sumcheck::final_reduce_block_sums`,
/// `zerocheck_round0.cu:525-594, 600-669`), so one blackbox over it is not
/// the guide's literal "one CUDA launch per blackbox".
///
/// This is the same precedent-backed exception documented on
/// [`ZerocheckEvalBufs`]: the IR author ships exactly this shape in
/// `frac_compute_round_dev_challenge` (`fractional_ir.rs:1927-1946`,
/// `gkr.cu:1429-1445, 1466-1480`, commit `b566fed5`). It is safe **because
/// the node below declares the complete access set across both launches** —
/// every input the evaluator reads, `intermediates` as a written input, and
/// `tmp_sums` (written by the main kernel, read by the reducer) plus `out`
/// (written by the reducer) as outputs.
///
/// The split that removes the exception is scoped in
/// `todo-solutions/T4-principle1-splits.md`
/// (`_zerocheck_ntt_eval_constraints_main` +
/// `_logup_zerocheck_final_reduce_block_sums`, with this symbol kept as a
/// two-line compatibility composition).
// TODO(cc-ir,T5): `main_ptrs` is a `*const F` pointer table assembled on the
//   host (`mod.rs:826-832`), the round-0 analogue of the ctx hole T5 closed
//   for stage D.
// WHY: T5 scoped device materializers for `MainMatrixPtrs<EF>` /
//   `ZerocheckCtx` / `LogupCtx`; the round-0 table is a bare `*const F[]`
//   with a different ABI and was not in scope.
// RISK: the planner cannot see through the table to the matrices it points
//   at, so the individual `mats` buffers must be bound explicitly by the
//   caller (as `interpolate_columns_ir` does) or the ordering is unstated.
//   The fix is a `_materialize_main_ptr_f` twin of
//   `_materialize_main_matrix_ptr`.
pub fn zerocheck_ntt_eval_constraints_ir(
    g: &mut GraphBuilder,
    bufs: Round0ZcBufs,
    shape: Round0ZcShape,
) {
    let mut inputs = vec![
        bufs.selectors_cube,
        bufs.main_ptrs,
        bufs.eq_cube,
        bufs.lambda_pows,
        bufs.public_values,
        bufs.rules,
        bufs.used_nodes,
    ];
    let has_prep = bufs.preprocessed.is_some();
    if let Some(prep) = bufs.preprocessed {
        inputs.push(prep);
    }
    // `intermediates` is pure scratch: the kernel writes it and reads back
    // only its own writes, so no value flows *in*. It is therefore declared
    // as a node **output**, not a carried input — same treatment the
    // fractional mirror gives `tmp_block_sums` (`fractional_ir.rs:399, 436`).
    // Declaring it carried would make it a read of an unproduced buffer,
    // which the fusion pass rejects with `TakeGraphError::ReadBeforeWrite`.
    let modifies: Vec<bool> = vec![false; inputs.len()];
    g.insert_blackbox_kernel(
        "zerocheck_ntt_eval_constraints",
        inputs.into_iter(),
        [bufs.tmp_sums, bufs.out, bufs.intermediates].into_iter(),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let mut tmp =
                DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, shape.tmp_sums_len);
            let mut out = DeviceBuffer::<EF>::from_raw_parts(outputs[1] as *mut EF, shape.out_len);
            let sels = DeviceBuffer::<F>::from_raw_parts(inputs[0] as *mut F, shape.sels_len);
            let main_ptrs = DeviceBuffer::<*const F>::from_raw_parts(
                inputs[1] as *mut *const F,
                shape.main_ptrs_len,
            );
            let lambda_pows =
                DeviceBuffer::<EF>::from_raw_parts(inputs[3] as *mut EF, shape.lambda_pows_len);
            let public =
                DeviceBuffer::<F>::from_raw_parts(inputs[4] as *mut F, shape.public_len.max(1));
            let rules =
                DeviceBuffer::<u128>::from_raw_parts(inputs[5] as *mut u128, shape.rules_len);
            let used_nodes = DeviceBuffer::<usize>::from_raw_parts(
                inputs[6] as *mut usize,
                shape.used_nodes_len,
            );
            let mut intermediates =
                DeviceBuffer::<F>::from_raw_parts(outputs[2] as *mut F, shape.intermediates_len);
            let prep_ptr = if has_prep {
                inputs[7] as *const F
            } else {
                std::ptr::null()
            };
            zerocheck_ntt_eval_constraints(
                &mut tmp,
                &mut out,
                &sels,
                prep_ptr,
                &main_ptrs,
                inputs[2] as *const EF,
                &lambda_pows,
                &public,
                &rules,
                &used_nodes,
                shape.buffer_size,
                &mut intermediates,
                shape.skip_domain,
                shape.num_x,
                shape.height,
                shape.num_cosets,
                shape.g_shift,
                shape.max_temp_bytes,
                stream,
            )
            .expect("zerocheck_ntt_eval_constraints");
            forget(tmp);
            forget(out);
            forget(sels);
            forget(main_ptrs);
            forget(lambda_pows);
            forget(public);
            forget(rules);
            forget(used_nodes);
            forget(intermediates);
        },
    );
}

/// Shapes for one `logup_bary_eval_interactions_round0` launch.
#[derive(Clone, Copy, Debug)]
pub struct Round0LogupShape {
    pub rules_len: usize,
    pub buffer_size: u32,
    pub skip_domain: u32,
    pub num_x: u32,
    pub height: u32,
    pub num_cosets: u32,
    pub g_shift: F,
    pub max_temp_bytes: usize,
    pub tmp_sums_len: usize,
    pub out_len: usize,
    pub intermediates_len: usize,
    pub sels_len: usize,
    pub main_ptrs_len: usize,
    pub public_len: usize,
    pub weights_len: usize,
    /// `denom_sum_init` — an `EF` derived from the logup challenges.
    pub denom_sum_init: EF,
}

/// Buffers for one `logup_bary_eval_interactions_round0` launch.
#[derive(Clone, Copy, Debug)]
pub struct Round0LogupBufs {
    pub tmp_sums: BufId,
    pub out: BufId,
    pub intermediates: BufId,
    pub selectors_cube: BufId,
    pub preprocessed: Option<BufId>,
    pub main_ptrs: BufId,
    pub eq_cube: BufId,
    pub public_values: BufId,
    pub numer_weights: BufId,
    pub denom_weights: BufId,
    pub rules: BufId,
}

/// Insert the round-0 interaction evaluator (`round0.rs:282` ← `mod.rs:896`).
///
/// # Principle-1 exception: multi-launch compatibility node (T4)
///
/// `_logup_bary_eval_interactions_round0` enqueues two kernels (main +
/// `final_reduce_block_sums`, `logup_round0.cu:519-586, 593-660`). Same named
/// exception, same precedent, same condition as
/// [`zerocheck_ntt_eval_constraints_ir`]: the node declares the complete
/// read/write union across both launches.
// TODO(cc-ir): `denom_sum_init: EF` is a challenge-derived scalar captured by
//   value.
// WHY: no `_dev_challenge` sibling exists for `_logup_bary_eval_interactions_round0`
//   (`cuda/logup_zerocheck.rs:476`); adding one is the `template <bool DEV_CH>`
//   pattern the author used six times in `gkr.cu` at `b566fed5`.
// RISK: blocks a device-resident challenge chain, not correctness.
//   `alpha_logup`/`beta_logup` are host values in the eager path too. The
//   host-assembled `main_ptrs` table carries the same T5-shaped hole noted on
//   [`zerocheck_ntt_eval_constraints_ir`].
pub fn logup_bary_eval_interactions_round0_ir(
    g: &mut GraphBuilder,
    bufs: Round0LogupBufs,
    shape: Round0LogupShape,
) {
    let mut inputs = vec![
        bufs.selectors_cube,
        bufs.main_ptrs,
        bufs.eq_cube,
        bufs.public_values,
        bufs.numer_weights,
        bufs.denom_weights,
        bufs.rules,
    ];
    let has_prep = bufs.preprocessed.is_some();
    if let Some(prep) = bufs.preprocessed {
        inputs.push(prep);
    }
    // `intermediates` is a node output, not a carried input — see the note
    // in [`zerocheck_ntt_eval_constraints_ir`].
    let modifies: Vec<bool> = vec![false; inputs.len()];
    g.insert_blackbox_kernel(
        "logup_bary_eval_interactions_round0",
        inputs.into_iter(),
        [bufs.tmp_sums, bufs.out, bufs.intermediates].into_iter(),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let mut tmp = DeviceBuffer::<Frac<EF>>::from_raw_parts(
                outputs[0] as *mut Frac<EF>,
                shape.tmp_sums_len,
            );
            let mut out = DeviceBuffer::<Frac<EF>>::from_raw_parts(
                outputs[1] as *mut Frac<EF>,
                shape.out_len,
            );
            let sels = DeviceBuffer::<F>::from_raw_parts(inputs[0] as *mut F, shape.sels_len);
            let main_ptrs = DeviceBuffer::<*const F>::from_raw_parts(
                inputs[1] as *mut *const F,
                shape.main_ptrs_len,
            );
            let public =
                DeviceBuffer::<F>::from_raw_parts(inputs[3] as *mut F, shape.public_len.max(1));
            let numer =
                DeviceBuffer::<EF>::from_raw_parts(inputs[4] as *mut EF, shape.weights_len.max(1));
            let denom =
                DeviceBuffer::<EF>::from_raw_parts(inputs[5] as *mut EF, shape.weights_len.max(1));
            let rules =
                DeviceBuffer::<u128>::from_raw_parts(inputs[6] as *mut u128, shape.rules_len);
            let mut intermediates =
                DeviceBuffer::<F>::from_raw_parts(outputs[2] as *mut F, shape.intermediates_len);
            let prep_ptr = if has_prep {
                inputs[7] as *const F
            } else {
                std::ptr::null()
            };
            logup_bary_eval_interactions_round0(
                &mut tmp,
                &mut out,
                &sels,
                prep_ptr,
                &main_ptrs,
                inputs[2] as *const EF,
                &public,
                &numer,
                &denom,
                shape.denom_sum_init,
                &rules,
                shape.buffer_size,
                &mut intermediates,
                shape.skip_domain,
                shape.num_x,
                shape.height,
                shape.num_cosets,
                shape.g_shift,
                shape.max_temp_bytes,
                stream,
            )
            .expect("logup_bary_eval_interactions_round0");
            forget(tmp);
            forget(out);
            forget(sels);
            forget(main_ptrs);
            forget(public);
            forget(numer);
            forget(denom);
            forget(rules);
            forget(intermediates);
        },
    );
}

/// Insert a `fold_ple_from_evals` node (`fold_ple.rs:95` ← `mod.rs:977/991/1004`).
///
/// The rotated fold writes the *second half of the same buffer* the plain
/// fold wrote the first half of (`fold_ple.rs:38, 50`). The graph is SSA, so
/// that is expressed as a rename: the rotated launch declares a fresh `dst`
/// aliased (`GraphBuilder::alias_bufs`) to the plain fold's output, takes the
/// plain output as `carry_in` — a read, so the halves stay ordered — and
/// writes at `dst_offset`.
#[allow(clippy::too_many_arguments)]
pub fn fold_ple_from_evals_ir(
    g: &mut GraphBuilder,
    input_matrix: BufId,
    input_len: usize,
    carry_in: Option<BufId>,
    dst: BufId,
    dst_offset: usize,
    omega_skip_pows: BufId,
    skip_domain: usize,
    inv_lagrange_denoms: BufId,
    height: u32,
    width: u32,
    l_skip: u32,
    new_height: u32,
    rotate: bool,
) {
    let mut inputs = vec![input_matrix, omega_skip_pows, inv_lagrange_denoms];
    if let Some(prev) = carry_in {
        inputs.push(prev);
    }
    let modifies: Vec<bool> = inputs.iter().map(|_| false).collect();
    g.insert_blackbox_kernel(
        if rotate {
            "fold_ple_from_evals<rot>"
        } else {
            "fold_ple_from_evals"
        },
        inputs.into_iter(),
        std::iter::once(dst),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let out_ptr = (outputs[0] as *mut EF).add(dst_offset);
            let mat = DeviceBuffer::<F>::from_raw_parts(inputs[0] as *mut F, input_len);
            let omega = DeviceBuffer::<F>::from_raw_parts(inputs[1] as *mut F, skip_domain);
            let denoms = DeviceBuffer::<EF>::from_raw_parts(inputs[2] as *mut EF, skip_domain);
            fold_ple_from_evals(
                &mat, out_ptr, &omega, &denoms, height, width, l_skip, new_height, rotate, stream,
            )
            .expect("fold_ple_from_evals");
            forget(mat);
            forget(omega);
            forget(denoms);
        },
    );
}

/// Insert a `fold_selectors_round0` node (`mod.rs:1044`).
// TODO(cc-ir): `is_first` / `is_last` are `EF` challenge-derived scalars
//   captured by value (`cuda/logup_zerocheck.rs:532-540`).
// WHY: both are `eval_eq_uni_at_one(l, r_0 …)` — pure host algebra over `r_0`
//   (`mod.rs:1040-1041`). Closing them needs both a `_dev_challenge` variant
//   *and* a device `eval_eq_uni_at_one`.
// RISK: none for a host-driven `r_0` (today's shape); it is the last
//   round-0 hop that would have to move for a device-resident `r_0`.
#[allow(clippy::too_many_arguments)]
pub fn fold_selectors_round0_ir(
    g: &mut GraphBuilder,
    out: BufId,
    input: BufId,
    is_first: EF,
    is_last: EF,
    num_x: usize,
) {
    g.insert_blackbox_kernel(
        "fold_selectors_round0",
        std::iter::once(input),
        std::iter::once(out),
        std::iter::once(false),
        move |inputs, outputs, stream| unsafe {
            fold_selectors_round0(
                outputs[0] as *mut EF,
                inputs[0] as *const F,
                is_first,
                is_last,
                num_x,
                stream,
            )
            .expect("fold_selectors_round0");
        },
    );
}

// ---------------------------------------------------------------------------
// Stage D — MLE rounds.

/// Insert an `interpolate_columns` node (`mod.rs:1207`).
///
/// `columns` is a device table of `*const EF` column pointers assembled
/// host-side (`mod.rs:1189-1199`) — the last unclosed instance of the ctx
/// pointer hole T5 closed for `ZerocheckCtx` / `LogupCtx`. The node
/// therefore binds `srcs` (the buffers the table points at) explicitly, so
/// the ordering is declared even though the planner cannot see through the
/// table itself.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_columns_ir(
    g: &mut GraphBuilder,
    interpolated: BufId,
    interpolated_len: usize,
    columns: BufId,
    srcs: &[BufId],
    num_columns: usize,
    s_deg: usize,
    num_y: usize,
) {
    // `columns` is a table of device pointers into `srcs`; name `srcs` too so
    // the planner keeps this node after whatever produced them.
    let mut inputs = vec![columns];
    inputs.extend_from_slice(srcs);
    let modifies: Vec<bool> = inputs.iter().map(|_| false).collect();
    g.insert_blackbox_kernel(
        "interpolate_columns",
        inputs.into_iter(),
        std::iter::once(interpolated),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let out = DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, interpolated_len);
            let cols =
                DeviceBuffer::<*const EF>::from_raw_parts(inputs[0] as *mut *const EF, num_columns);
            interpolate_columns_gpu(&out, &cols, s_deg, num_y, stream)
                .expect("interpolate_columns");
            forget(out);
            forget(cols);
        },
    );
}

// ===========================================================================
// Device-side ctx materialization (T5).
// ===========================================================================
//
// The eager path assembles `MainMatrixPtrs<EF>` / `ZerocheckCtx` / `LogupCtx`
// on the host and uploads them (`batch_mle.rs:158-201`,
// `batch_mle_monomial.rs:176-192`). Every device pointer inside those structs
// is then a *host-baked address*: the planner sees an opaque leaf and cannot
// alias-analyse through it to the buffers the evaluator will dereference.
//
// The materializers below write one element of a ctx array on the device from
// explicitly-typed launcher arguments. Each pointer argument is either a
// [`DevicePtrArg::Graph`] — a `BufId` bound as a node input and resolved to
// its pool address at invocation time — or a [`DevicePtrArg::Static`] address
// the graph does not own (keygen-static proving-key tables, or null). Only
// host-known scalars are captured by value. That is what puts every device
// pointer back on a real graph edge.
//
// The CUDA side is `cuda/src/logup_zerocheck/batch_mle.cu`
// ("CTX MATERIALIZERS"); it zeroes the destination element first, so ABI
// padding is deterministic across replays, then assigns field by field.

/// One device-pointer field of a batched ctx struct.
#[derive(Clone, Copy, Debug)]
pub enum DevicePtrArg {
    /// A graph-owned buffer, plus a byte offset applied to the *resolved*
    /// pointer inside the closure (never baked at build time).
    Graph { buf: BufId, byte_offset: usize },
    /// A raw address the graph does not own.
    ///
    /// Used for the null pointer (`Static(0)`) and for keygen-static
    /// proving-key tables. Captured as `usize` because a blackbox closure
    /// must be `Send + Sync + 'static`.
    // TODO(cc-ir,T5): a `Static` proving-key address is only valid while that
    //   exact `DeviceMultiStarkProvingKey` lives (its `DeviceBuffer`s free on
    //   drop, `pkey.rs:268-334`).
    // WHY: the rule tables are not graph inputs today, and copying them into
    //   graph-owned buffers would duplicate multi-MB keygen data per graph.
    // RISK: a `GraphExe` cached beyond the borrowed pk's lifetime (e.g. a
    //   process-global graph cache) would replay dangling pointers. Keep the
    //   compiled graph below the pk's lifetime until the tables become graph
    //   inputs or the cache takes an owning pk handle.
    Static(usize),
}

impl DevicePtrArg {
    /// The null pointer — the eager path's "absent" encoding for
    /// `d_preprocessed.data` and for `d_intermediates` when `buffer_size == 0`.
    pub const NULL: Self = DevicePtrArg::Static(0);

    /// A whole graph buffer.
    pub fn buf(buf: BufId) -> Self {
        DevicePtrArg::Graph {
            buf,
            byte_offset: 0,
        }
    }

    /// A graph buffer offset by `byte_offset` bytes.
    pub fn at(buf: BufId, byte_offset: usize) -> Self {
        DevicePtrArg::Graph { buf, byte_offset }
    }
}

/// How one declared [`DevicePtrArg`] is recovered inside a blackbox closure.
#[derive(Clone, Copy, Debug)]
enum PtrSlot {
    /// `inputs[idx] + byte_offset`.
    Input { idx: usize, byte_offset: usize },
    /// A raw address captured by value.
    Static(usize),
}

/// Build-time half: collects a materializer node's graph inputs and remembers
/// where each declared pointer argument ended up.
struct PtrBinder {
    inputs: Vec<BufId>,
    slots: Vec<PtrSlot>,
}

impl PtrBinder {
    /// `out` is bound as input 0 with `modifies = true` (the node's only
    /// write), so graph pointer arguments start at input index 1.
    fn new(out: BufId) -> Self {
        Self {
            inputs: vec![out],
            slots: Vec::new(),
        }
    }

    fn push(&mut self, arg: DevicePtrArg) {
        let slot = match arg {
            DevicePtrArg::Static(addr) => PtrSlot::Static(addr),
            DevicePtrArg::Graph { buf, byte_offset } => {
                // Dedupe: one buffer may back several fields.
                let idx = self
                    .inputs
                    .iter()
                    .position(|&b| b == buf)
                    .unwrap_or_else(|| {
                        self.inputs.push(buf);
                        self.inputs.len() - 1
                    });
                PtrSlot::Input { idx, byte_offset }
            }
        };
        self.slots.push(slot);
    }

    fn push_all(&mut self, args: impl IntoIterator<Item = DevicePtrArg>) {
        for a in args {
            self.push(a);
        }
    }

    fn finish(self) -> (Vec<BufId>, Vec<bool>, PtrResolver) {
        let mut modifies = vec![false; self.inputs.len()];
        modifies[0] = true;
        (self.inputs, modifies, PtrResolver { slots: self.slots })
    }
}

/// The read set a materializer node declares: its [`PtrBinder`] inputs with
/// `out` (index 0, the node's only write) dropped.
///
/// This is the *machine-derived* half of the Principle-1 access set. Every
/// evaluator that dereferences a materialized ctx element must declare at
/// least these buffers, or the planner will give their pool slots away while
/// the evaluator still needs them (`planner/heuristic.rs:184-190`). Deriving
/// the list from the binder — instead of re-typing it by hand at the call
/// site — is what makes an under-declared access set unrepresentable: a new
/// pointer field cannot be added to a ctx struct without its buffer flowing
/// into this vector.
fn declared_reads(inputs: &[BufId]) -> Vec<BufId> {
    inputs[1..].to_vec()
}

/// Union `reads` into `set` in first-seen order, dropping duplicates.
///
/// One buffer commonly backs several ctx fields and several AIRs;
/// [`eval_node_bindings`] would fold the duplicates anyway, but it does so
/// with a linear scan per element, so deduplicating here keeps the access
/// set proportional to the distinct buffers rather than to the field count.
fn extend_reads(set: &mut Vec<BufId>, reads: impl IntoIterator<Item = BufId>) {
    for b in reads {
        if !set.contains(&b) {
            set.push(b);
        }
    }
}

/// Invocation-time half of [`PtrBinder`].
#[derive(Clone, Debug)]
struct PtrResolver {
    slots: Vec<PtrSlot>,
}

impl PtrResolver {
    /// Resolve the `k`-th declared pointer argument against the pointers the
    /// graph runner handed the closure.
    ///
    /// # Safety
    /// `inputs` must be the slice the graph runner passed to the closure.
    unsafe fn ptr(&self, k: usize, inputs: &[*mut ()]) -> *mut u8 {
        match self.slots[k] {
            PtrSlot::Static(addr) => addr as *mut u8,
            PtrSlot::Input { idx, byte_offset } => (inputs[idx] as *mut u8).add(byte_offset),
        }
    }
}

/// The five logical components of `EvalCoreCtx` (`cuda/logup_zerocheck.rs:39`).
#[derive(Clone, Copy, Debug)]
pub struct EvalCoreCtxArgs {
    pub d_selectors: DevicePtrArg,
    /// `d_preprocessed.data`; [`DevicePtrArg::NULL`] when the AIR has none.
    pub d_preprocessed_data: DevicePtrArg,
    /// `d_preprocessed.air_width`; `0` when the AIR has no preprocessed data.
    pub preprocessed_air_width: u32,
    /// The materialized `MainMatrixPtrs<EF>` descriptor array.
    pub d_main: DevicePtrArg,
    pub d_public: DevicePtrArg,
}

/// Every field of `ZerocheckCtx` (`cuda/logup_zerocheck.rs:47`).
#[derive(Clone, Copy, Debug)]
pub struct ZerocheckCtxArgs {
    pub eval_ctx: EvalCoreCtxArgs,
    /// [`DevicePtrArg::NULL`] when `buffer_size == 0`, matching the eager
    /// branch at `batch_mle.rs:164-176`.
    pub d_intermediates: DevicePtrArg,
    pub num_y: u32,
    pub d_eq_xi: DevicePtrArg,
    pub d_rules: DevicePtrArg,
    pub rules_len: usize,
    pub d_used_nodes: DevicePtrArg,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

/// Every field of `LogupCtx` (`cuda/logup_zerocheck.rs:62`).
#[derive(Clone, Copy, Debug)]
pub struct LogupCtxArgs {
    pub eval_ctx: EvalCoreCtxArgs,
    /// [`DevicePtrArg::NULL`] when `buffer_size == 0`
    /// (`batch_mle.rs:303-315`).
    pub d_intermediates: DevicePtrArg,
    pub num_y: u32,
    pub d_eq_xi: DevicePtrArg,
    pub d_challenges: DevicePtrArg,
    pub d_eq_3bs: DevicePtrArg,
    pub d_rules: DevicePtrArg,
    pub rules_len: usize,
    pub d_used_nodes: DevicePtrArg,
    pub d_pair_idxs: DevicePtrArg,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

/// Insert a `materialize_main_matrix_ptr` node: writes `out[idx]` of a
/// `MainMatrixPtrs<EF>` descriptor array.
///
/// Mirrors the host-side `MainMatrixPtrs` construction the eager path does
/// per trace (`mod.rs:1131-1149, 1224-1256`).
///
/// Returns the node's *declared pointer reads* — see
/// [`materialize_zerocheck_ctx_ir`] for why the caller must forward them.
#[must_use = "the declared reads must reach the evaluator's `ctx_reads`"]
pub fn materialize_main_matrix_ptr_ir(
    g: &mut GraphBuilder,
    out: BufId,
    idx: u32,
    data: DevicePtrArg,
    air_width: u32,
) -> Vec<BufId> {
    let mut b = PtrBinder::new(out);
    b.push(data);
    let (inputs, modifies, ptrs) = b.finish();
    let reads = declared_reads(&inputs);
    g.insert_blackbox_kernel(
        "materialize_main_matrix_ptr",
        inputs.into_iter(),
        std::iter::empty(),
        modifies.into_iter(),
        move |inputs, _outputs, stream| unsafe {
            materialize_main_matrix_ptr_raw(
                inputs[0] as *mut MainMatrixPtrs<EF>,
                idx,
                ptrs.ptr(0, inputs) as *const EF,
                air_width,
                stream,
            )
            .expect("materialize_main_matrix_ptr");
        },
    );
    reads
}

/// Insert a `materialize_zerocheck_ctx` node: writes `out[idx]` of a
/// `ZerocheckCtx` array.
///
/// Mirrors `ZerocheckMleBatchBuilder::new`'s per-trace struct
/// (`batch_mle.rs:158-201`) field for field.
///
/// Returns the node's *declared pointer reads* — see [`declared_reads`].
#[must_use = "the declared reads must reach the evaluator's `ctx_reads`"]
pub fn materialize_zerocheck_ctx_ir(
    g: &mut GraphBuilder,
    out: BufId,
    idx: u32,
    args: ZerocheckCtxArgs,
) -> Vec<BufId> {
    let mut b = PtrBinder::new(out);
    b.push_all([
        args.eval_ctx.d_selectors,
        args.eval_ctx.d_preprocessed_data,
        args.eval_ctx.d_main,
        args.eval_ctx.d_public,
        args.d_intermediates,
        args.d_eq_xi,
        args.d_rules,
        args.d_used_nodes,
    ]);
    let (inputs, modifies, ptrs) = b.finish();
    let reads = declared_reads(&inputs);
    g.insert_blackbox_kernel(
        "materialize_zerocheck_ctx",
        inputs.into_iter(),
        std::iter::empty(),
        modifies.into_iter(),
        move |inputs, _outputs, stream| unsafe {
            materialize_zerocheck_ctx_raw(
                inputs[0] as *mut ZerocheckCtx,
                idx,
                ptrs.ptr(0, inputs) as *const EF,
                ptrs.ptr(1, inputs) as *const EF,
                args.eval_ctx.preprocessed_air_width,
                ptrs.ptr(2, inputs) as *const MainMatrixPtrs<EF>,
                ptrs.ptr(3, inputs) as *const F,
                ptrs.ptr(4, inputs) as *mut EF,
                args.num_y,
                ptrs.ptr(5, inputs) as *const EF,
                ptrs.ptr(6, inputs) as *const std::ffi::c_void,
                args.rules_len,
                ptrs.ptr(7, inputs) as *const usize,
                args.used_nodes_len,
                args.buffer_size,
                stream,
            )
            .expect("materialize_zerocheck_ctx");
        },
    );
    reads
}

/// Insert a `materialize_logup_ctx` node: writes `out[idx]` of a `LogupCtx`
/// array.
///
/// Mirrors `LogupMleBatchBuilder::new`'s per-trace struct
/// (`batch_mle.rs:297-353`) field for field.
///
/// Returns the node's *declared pointer reads* — see [`declared_reads`].
#[must_use = "the declared reads must reach the evaluator's `ctx_reads`"]
pub fn materialize_logup_ctx_ir(
    g: &mut GraphBuilder,
    out: BufId,
    idx: u32,
    args: LogupCtxArgs,
) -> Vec<BufId> {
    let mut b = PtrBinder::new(out);
    b.push_all([
        args.eval_ctx.d_selectors,
        args.eval_ctx.d_preprocessed_data,
        args.eval_ctx.d_main,
        args.eval_ctx.d_public,
        args.d_intermediates,
        args.d_eq_xi,
        args.d_challenges,
        args.d_eq_3bs,
        args.d_rules,
        args.d_used_nodes,
        args.d_pair_idxs,
    ]);
    let (inputs, modifies, ptrs) = b.finish();
    let reads = declared_reads(&inputs);
    g.insert_blackbox_kernel(
        "materialize_logup_ctx",
        inputs.into_iter(),
        std::iter::empty(),
        modifies.into_iter(),
        move |inputs, _outputs, stream| unsafe {
            materialize_logup_ctx_raw(
                inputs[0] as *mut LogupCtx,
                idx,
                ptrs.ptr(0, inputs) as *const EF,
                ptrs.ptr(1, inputs) as *const EF,
                args.eval_ctx.preprocessed_air_width,
                ptrs.ptr(2, inputs) as *const MainMatrixPtrs<EF>,
                ptrs.ptr(3, inputs) as *const F,
                ptrs.ptr(4, inputs) as *mut EF,
                args.num_y,
                ptrs.ptr(5, inputs) as *const EF,
                ptrs.ptr(6, inputs) as *const EF,
                ptrs.ptr(7, inputs) as *const EF,
                ptrs.ptr(8, inputs) as *const std::ffi::c_void,
                args.rules_len,
                ptrs.ptr(9, inputs) as *const usize,
                ptrs.ptr(10, inputs) as *const u32,
                args.used_nodes_len,
                args.buffer_size,
                stream,
            )
            .expect("materialize_logup_ctx");
        },
    );
    reads
}

/// Launch geometry shared by every batched MLE evaluator in stage D.
#[derive(Clone, Copy, Debug)]
pub struct BatchEvalShape {
    pub num_blocks: u32,
    pub num_x: u32,
    pub num_airs: u32,
    pub threads_per_block: u32,
    pub tmp_sums_len: usize,
    pub out_len: usize,
    /// Only used by the zerocheck DAG evaluator.
    pub lambda_pows_len: usize,
    /// Only used by `zerocheck_monomial_par_y_batched`.
    pub chunk_size: u32,
}

/// The buffers one batched evaluator touches.
///
/// # Principle-1 exception: multi-launch compatibility node (T4)
///
/// Each of these evaluators is **one blackbox node over a `.cu` entry point
/// that enqueues two kernels** (the main evaluator plus
/// `batched_final_reduce_block_sums`), so it violates the porting guide's
/// literal "one CUDA launch per blackbox" rule
/// (`crates/compiler/gpu_ir_porting_guide.md:32-56`).
///
/// This is a *named, precedent-backed* exception, not an oversight: the IR
/// author ships exactly this shape himself — `frac_compute_round_dev_challenge`
/// is one blackbox over a launcher that enqueues its main kernel *and* its
/// final reduction (`fractional_ir.rs:1927-1946`, `gkr.cu:1429-1445,
/// 1466-1480`, both added in `b566fed5`).
///
/// What the exception costs is only planner visibility *at the hidden seam*:
/// no node can be scheduled between main and reduce, and the temporary's
/// logical lifetime cannot be shortened. Fusion is unaffected (both fusion
/// passes only admit structured `Kernel` nodes), CUDA-graph capture is
/// unaffected (capture wraps the whole `GraphExe::run`, so every launch the
/// closure issues on the supplied stream is captured), and **scheduling
/// correctness is unaffected provided the node declares the complete
/// read/write union** — an under-declared access set is a real correctness
/// bug, because the planner derives edges only from declared accesses
/// (`crates/compiler/src/planner/ctx.rs:215-294`).
///
/// So the access set here is the union across *both* launches, including
/// everything reached only by dereferencing the ctx array. The follow-up that
/// removes the exception is the launcher split in
/// `todo-solutions/T4-principle1-splits.md` (ten main-only entries plus two
/// shared reducer entries, old symbols kept as compatibility compositions).
#[derive(Clone, Debug)]
pub struct ZerocheckEvalBufs {
    /// Written by the main kernel, read by the reducer.
    pub tmp_sums: BufId,
    /// Written by the reducer.
    pub out: BufId,
    pub block_ctxs: BufId,
    /// The `ZerocheckCtx` / `LogupCtx` / `MonomialAirCtx` array, built on the
    /// device by the T5 materializers.
    pub air_ctxs: BufId,
    /// Read by the reducer only.
    pub air_block_offsets: BufId,
    /// `lambda_pows`; unused by the monomial evaluators.
    pub lambda_pows: Option<BufId>,
    /// Principle-1 access set: every buffer the kernels **read** that is
    /// reachable only by dereferencing `air_ctxs` — selectors, folded main /
    /// preprocessed matrices, the `MainMatrixPtrs` descriptor array, public
    /// values, `eq_xi`, the rule / used-node / pair-index streams, logup
    /// challenges and `eq_3bs`.
    pub ctx_reads: Vec<BufId>,
    /// Principle-1 access set: per-AIR `d_intermediates` scratch, **written**
    /// through the ctx by the DAG interpreter (`batch_mle.cu:186`). Bound
    /// with `modifies = true`.
    pub intermediates: Vec<BufId>,
}

/// Assemble a node's `(inputs, modifies)` from a fixed positional prefix (the
/// buffers the closure indexes by hand), a written set, and a read set.
///
/// Later duplicates are folded into the first binding; a buffer that is both
/// read and written ends up with `modifies = true`.
fn eval_node_bindings(
    fixed: &[BufId],
    written: &[BufId],
    read: &[BufId],
) -> (Vec<BufId>, Vec<bool>) {
    let mut inputs: Vec<BufId> = Vec::with_capacity(fixed.len() + written.len() + read.len());
    let mut modifies: Vec<bool> = Vec::with_capacity(inputs.capacity());
    for &b in fixed {
        inputs.push(b);
        modifies.push(false);
    }
    for &b in written {
        match inputs.iter().position(|&x| x == b) {
            Some(i) => modifies[i] = true,
            None => {
                inputs.push(b);
                modifies.push(true);
            }
        }
    }
    for &b in read {
        if !inputs.contains(&b) {
            inputs.push(b);
            modifies.push(false);
        }
    }
    (inputs, modifies)
}

/// Insert a `zerocheck_batch_eval_mle` node (`batch_mle.rs:696`) — the
/// multi-AIR DAG-interpreter constraint evaluator.
pub fn zerocheck_batch_eval_mle_ir(
    g: &mut GraphBuilder,
    bufs: &ZerocheckEvalBufs,
    shape: BatchEvalShape,
) {
    let lambda_pows = bufs
        .lambda_pows
        .expect("zerocheck DAG eval needs lambda_pows");
    // Principle-1 exception (see [`ZerocheckEvalBufs`]): two launches behind
    // one node, so the access set is the union across both.
    //   main   reads  block_ctxs, air_ctxs + everything reachable through it,
    //                 lambda_pows;  writes tmp_sums, d_intermediates
    //   reduce reads  tmp_sums, air_block_offsets;  writes out
    let (inputs, modifies) = eval_node_bindings(
        &[
            bufs.block_ctxs,
            bufs.air_ctxs,
            bufs.air_block_offsets,
            lambda_pows,
        ],
        &bufs.intermediates,
        &bufs.ctx_reads,
    );
    g.insert_blackbox_kernel(
        "zerocheck_batch_eval_mle",
        inputs.into_iter(),
        [bufs.tmp_sums, bufs.out].into_iter(),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let mut tmp =
                DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, shape.tmp_sums_len);
            let mut out = DeviceBuffer::<EF>::from_raw_parts(outputs[1] as *mut EF, shape.out_len);
            let block_ctxs = DeviceBuffer::<BlockCtx>::from_raw_parts(
                inputs[0] as *mut BlockCtx,
                shape.num_blocks as usize,
            );
            let zc_ctxs = DeviceBuffer::<ZerocheckCtx>::from_raw_parts(
                inputs[1] as *mut ZerocheckCtx,
                shape.num_airs as usize,
            );
            let offsets = DeviceBuffer::<u32>::from_raw_parts(
                inputs[2] as *mut u32,
                shape.num_airs as usize + 1,
            );
            let pows =
                DeviceBuffer::<EF>::from_raw_parts(inputs[3] as *mut EF, shape.lambda_pows_len);
            zerocheck_batch_eval_mle(
                &mut tmp,
                &mut out,
                &block_ctxs,
                &zc_ctxs,
                &offsets,
                &pows,
                shape.lambda_pows_len,
                shape.num_blocks,
                shape.num_x,
                shape.num_airs,
                shape.threads_per_block,
                stream,
            )
            .expect("zerocheck_batch_eval_mle");
            forget(tmp);
            forget(out);
            forget(block_ctxs);
            forget(zc_ctxs);
            forget(offsets);
            forget(pows);
        },
    );
}

/// Insert a `logup_batch_eval_mle` node (`batch_mle.rs:731`).
pub fn logup_batch_eval_mle_ir(
    g: &mut GraphBuilder,
    bufs: &ZerocheckEvalBufs,
    shape: BatchEvalShape,
) {
    // Principle-1 exception (see [`ZerocheckEvalBufs`]): main + batched
    // reducer behind one node; access set is the union across both.
    let (inputs, modifies) = eval_node_bindings(
        &[bufs.block_ctxs, bufs.air_ctxs, bufs.air_block_offsets],
        &bufs.intermediates,
        &bufs.ctx_reads,
    );
    g.insert_blackbox_kernel(
        "logup_batch_eval_mle",
        inputs.into_iter(),
        [bufs.tmp_sums, bufs.out].into_iter(),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let mut tmp = DeviceBuffer::<Frac<EF>>::from_raw_parts(
                outputs[0] as *mut Frac<EF>,
                shape.tmp_sums_len,
            );
            let mut out = DeviceBuffer::<Frac<EF>>::from_raw_parts(
                outputs[1] as *mut Frac<EF>,
                shape.out_len,
            );
            let block_ctxs = DeviceBuffer::<BlockCtx>::from_raw_parts(
                inputs[0] as *mut BlockCtx,
                shape.num_blocks as usize,
            );
            let logup_ctxs = DeviceBuffer::<LogupCtx>::from_raw_parts(
                inputs[1] as *mut LogupCtx,
                shape.num_airs as usize,
            );
            let offsets = DeviceBuffer::<u32>::from_raw_parts(
                inputs[2] as *mut u32,
                shape.num_airs as usize + 1,
            );
            logup_batch_eval_mle(
                &mut tmp,
                &mut out,
                &block_ctxs,
                &logup_ctxs,
                &offsets,
                shape.num_blocks,
                shape.num_x,
                shape.num_airs,
                shape.threads_per_block,
                stream,
            )
            .expect("logup_batch_eval_mle");
            forget(tmp);
            forget(out);
            forget(block_ctxs);
            forget(logup_ctxs);
            forget(offsets);
        },
    );
}

/// Insert a `zerocheck_monomial_batched` node (`batch_mle_monomial.rs:250`).
///
/// When `par_y` is set, emits `zerocheck_monomial_par_y_batched`
/// (`batch_mle_monomial.rs:471`) instead — the same buffers, one extra
/// `chunk_size` argument.
pub fn zerocheck_monomial_batched_ir(
    g: &mut GraphBuilder,
    bufs: &ZerocheckEvalBufs,
    shape: BatchEvalShape,
    par_y: bool,
) {
    // Principle-1 exception (see [`ZerocheckEvalBufs`]): main + batched
    // reducer behind one node; access set is the union across both.
    let (inputs, modifies) = eval_node_bindings(
        &[bufs.block_ctxs, bufs.air_ctxs, bufs.air_block_offsets],
        &bufs.intermediates,
        &bufs.ctx_reads,
    );
    g.insert_blackbox_kernel(
        if par_y {
            "zerocheck_monomial_par_y_batched"
        } else {
            "zerocheck_monomial_batched"
        },
        inputs.into_iter(),
        [bufs.tmp_sums, bufs.out].into_iter(),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let mut tmp =
                DeviceBuffer::<EF>::from_raw_parts(outputs[0] as *mut EF, shape.tmp_sums_len);
            let mut out = DeviceBuffer::<EF>::from_raw_parts(outputs[1] as *mut EF, shape.out_len);
            let block_ctxs = DeviceBuffer::<BlockCtx>::from_raw_parts(
                inputs[0] as *mut BlockCtx,
                shape.num_blocks as usize,
            );
            let air_ctxs = DeviceBuffer::<MonomialAirCtx>::from_raw_parts(
                inputs[1] as *mut MonomialAirCtx,
                shape.num_airs as usize,
            );
            let offsets = DeviceBuffer::<u32>::from_raw_parts(
                inputs[2] as *mut u32,
                shape.num_airs as usize + 1,
            );
            if par_y {
                zerocheck_monomial_par_y_batched(
                    &mut tmp,
                    &mut out,
                    &block_ctxs,
                    &air_ctxs,
                    &offsets,
                    shape.num_blocks,
                    shape.num_x,
                    shape.num_airs,
                    shape.chunk_size,
                    shape.threads_per_block,
                    stream,
                )
                .expect("zerocheck_monomial_par_y_batched");
            } else {
                zerocheck_monomial_batched(
                    &mut tmp,
                    &mut out,
                    &block_ctxs,
                    &air_ctxs,
                    &offsets,
                    shape.num_blocks,
                    shape.num_x,
                    shape.num_airs,
                    shape.threads_per_block,
                    stream,
                )
                .expect("zerocheck_monomial_batched");
            }
            forget(tmp);
            forget(out);
            forget(block_ctxs);
            forget(air_ctxs);
            forget(offsets);
        },
    );
}

/// Buffers for a `logup_monomial_batched` launch — it takes three ctx
/// arrays (common, numerator, denominator) instead of one.
///
/// The Principle-1 exception documented on [`ZerocheckEvalBufs`] applies here
/// too, and harder: this entry point enqueues **three** kernels (numerator
/// pass, denominator pass, batched reducer —
/// `batch_mle_monomial.cu:446-498`). The numerator kernel writes `tmp_sums.p`
/// and the denominator kernel writes `tmp_sums.q` *while preserving* `.p`, so
/// in a split graph the denominator pass would take `tmp_sums` as an input
/// with `modifies = true`; behind one node the union is simply "tmp_sums is
/// written".
#[derive(Clone, Debug)]
pub struct LogupMonomialBufs {
    /// Written by the numerator pass (`.p`) then the denominator pass (`.q`),
    /// read by the reducer.
    pub tmp_sums: BufId,
    /// Written by the reducer.
    pub out: BufId,
    pub block_ctxs: BufId,
    pub common_ctxs: BufId,
    pub numer_ctxs: BufId,
    pub denom_ctxs: BufId,
    pub air_block_offsets: BufId,
    /// Principle-1 access set: buffers read only through the three ctx
    /// arrays (monomial headers / variable streams / combination tables,
    /// selectors, folded matrices, the `MainMatrixPtrs` array, public
    /// values, `eq_xi`).
    pub ctx_reads: Vec<BufId>,
}

/// Insert a `logup_monomial_batched` node (`batch_mle_monomial.rs:772`).
///
/// Principle-1 exception, see [`LogupMonomialBufs`]. The full access set is
/// declared below, which is what makes the exception safe; the split that
/// removes it is scoped in `todo-solutions/T4-principle1-splits.md`.
// TODO(cc-ir,T5): `LogupMonomialCommonCtx::bus_term_sum` is an `EF` challenge
//   scalar riding *inside* the uploaded ctx (`batch_mle_monomial.rs:671`), so
//   this node is challenge-by-value in disguise.
// WHY: T5 scoped device materializers for `MainMatrixPtrs` / `ZerocheckCtx` /
//   `LogupCtx` only; the three monomial ctx structs are not covered and are
//   still staged host-side (here: zeroed placeholders).
// RISK: a `_dev_challenge` port of the monomial path has to reach into the
//   struct, i.e. it needs a `materialize_logup_monomial_common_ctx` that
//   takes `bus_term_sum` as a `BufId` rather than a value. Until then the
//   monomial evaluators cannot be driven by device-sampled challenges.
pub fn logup_monomial_batched_ir(
    g: &mut GraphBuilder,
    bufs: &LogupMonomialBufs,
    shape: BatchEvalShape,
) {
    let (inputs, modifies) = eval_node_bindings(
        &[
            bufs.block_ctxs,
            bufs.common_ctxs,
            bufs.numer_ctxs,
            bufs.denom_ctxs,
            bufs.air_block_offsets,
        ],
        &[],
        &bufs.ctx_reads,
    );
    g.insert_blackbox_kernel(
        "logup_monomial_batched",
        inputs.into_iter(),
        [bufs.tmp_sums, bufs.out].into_iter(),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let n_airs = shape.num_airs as usize;
            let mut tmp = DeviceBuffer::<Frac<EF>>::from_raw_parts(
                outputs[0] as *mut Frac<EF>,
                shape.tmp_sums_len,
            );
            let mut out = DeviceBuffer::<Frac<EF>>::from_raw_parts(
                outputs[1] as *mut Frac<EF>,
                shape.out_len,
            );
            let block_ctxs = DeviceBuffer::<BlockCtx>::from_raw_parts(
                inputs[0] as *mut BlockCtx,
                shape.num_blocks as usize,
            );
            let common = DeviceBuffer::<LogupMonomialCommonCtx>::from_raw_parts(
                inputs[1] as *mut LogupMonomialCommonCtx,
                n_airs,
            );
            let numer = DeviceBuffer::<LogupMonomialCtx>::from_raw_parts(
                inputs[2] as *mut LogupMonomialCtx,
                n_airs,
            );
            let denom = DeviceBuffer::<LogupMonomialCtx>::from_raw_parts(
                inputs[3] as *mut LogupMonomialCtx,
                n_airs,
            );
            let offsets = DeviceBuffer::<u32>::from_raw_parts(inputs[4] as *mut u32, n_airs + 1);
            logup_monomial_batched(
                &mut tmp,
                &mut out,
                &block_ctxs,
                &common,
                &numer,
                &denom,
                &offsets,
                shape.num_blocks,
                shape.num_x,
                shape.num_airs,
                shape.threads_per_block,
                stream,
            )
            .expect("logup_monomial_batched");
            forget(tmp);
            forget(out);
            forget(block_ctxs);
            forget(common);
            forget(numer);
            forget(denom);
            forget(offsets);
        },
    );
}

/// Insert a `batch_fold_mle` node (`mod.rs:1541`) — the end-of-round fold
/// `out = in[i] + r·(in[i|half] − in[i])` over ragged-height matrices.
// TODO(cc-ir): `r_val: EF` is the round challenge, captured by value.
// WHY: `_batch_fold_mle` (`cuda/mod.rs:64-72`) has no `_dev_challenge`
//   sibling. Adding one is the established `template <bool DEV_CH>` pattern
//   (six precedents at `b566fed5` in `gkr.cu`) and is the single highest-value
//   `.cu` change for this port: it is the one kernel on the *round-to-round*
//   critical path that forces the challenge back to the host.
// RISK: with `compute_batch_s_poly` still on the host, `r_round` is a host
//   value anyway, so this changes nothing today. It becomes blocking the
//   moment seam (2) closes.
#[allow(clippy::too_many_arguments)]
pub fn batch_fold_mle_ir(
    g: &mut GraphBuilder,
    input_ptrs: BufId,
    output_ptrs: BufId,
    widths: BufId,
    log_output_heights: BufId,
    srcs: &[BufId],
    dsts: &[BufId],
    num_matrices: u16,
    max_output_cells: u32,
    r_val: EF,
) {
    // The kernel reaches its operands through `input_ptrs` / `output_ptrs`,
    // which the planner cannot see through. Naming the pointed-to buffers as
    // real inputs and outputs of the node restores every dependency edge.
    let mut inputs = vec![input_ptrs, output_ptrs, widths, log_output_heights];
    inputs.extend_from_slice(srcs);
    let modifies: Vec<bool> = inputs.iter().map(|_| false).collect();
    g.insert_blackbox_kernel(
        "batch_fold_mle",
        inputs.into_iter(),
        dsts.iter().copied(),
        modifies.into_iter(),
        move |inputs, _outputs, stream| unsafe {
            let n = num_matrices as usize;
            let ins = DeviceBuffer::<*const EF>::from_raw_parts(inputs[0] as *mut *const EF, n);
            let outs = DeviceBuffer::<*mut EF>::from_raw_parts(inputs[1] as *mut *mut EF, n);
            let widths = DeviceBuffer::<u32>::from_raw_parts(inputs[2] as *mut u32, n);
            let logh = DeviceBuffer::<u8>::from_raw_parts(inputs[3] as *mut u8, n);
            batch_fold_mle(
                &ins,
                &outs,
                &widths,
                num_matrices,
                &logh,
                max_output_cells,
                r_val,
                stream,
            )
            .expect("batch_fold_mle");
            forget(ins);
            forget(outs);
            forget(widths);
            forget(logh);
        },
    );
}

// ===========================================================================
// Phase plan — the host-side shape descriptor.
//
// `fractional_sumcheck_gpu_irv2` takes `(leaves: BufId, logical_len, alpha,
// assert_zero, device)`: input buffers plus the shape and host scalars it
// needs to unroll. This is the same idea, scaled to a phase with
// `num_traces` AIRs, `n_max` rounds and a per-trace evaluator choice.
// ===========================================================================

/// One matrix of a trace (preprocessed / cached / common main).
#[derive(Clone, Copy, Debug)]
pub struct MatPlan {
    pub width: usize,
    pub height: usize,
}

/// Which of the phase's evaluator families a trace uses in stage D.
// TODO(cc-ir): the eager dispatch is seven-way (`mod.rs:1332-1424` plus the
//   FFD bin-packing inside `batch_mle.rs:430-440, 543-568`); this is a
//   three-way summary of it.
// WHY: the missing branches are the *single-AIR* fallbacks
//   (`evaluate_single_logup`, `evaluate_mle_constraints_gpu`) and the
//   oversized-trace split, all of which are choices about how many AIRs
//   share one launch, not about what is computed.
// RISK: a real port must emit the same partition the eager path picks or the
//   graph's launch geometry (`num_blocks`, `air_block_offsets`) will not
//   match the ctx arrays staged for it. Passing the eager path's chosen
//   partition in through the plan is the fix; classifying here is a
//   stand-in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundEvalKind {
    /// `_zerocheck_batch_eval_mle` / `_logup_batch_eval_mle`.
    Dag,
    /// `_zerocheck_monomial_batched` / `_logup_monomial_batched`.
    Monomial,
    /// `_zerocheck_monomial_par_y_batched`.
    MonomialParY,
}

/// Per-trace shape, all of it known at keygen / graph-build time.
#[derive(Clone, Debug)]
pub struct TracePlan {
    /// `log2(common_main.height()) - l_skip`; **may be negative** for AIRs
    /// shorter than the skip domain (`mod.rs:549-551`).
    pub n: isize,
    pub need_rot: bool,
    pub has_constraints: bool,
    pub has_interactions: bool,
    /// `local_constraint_deg` (`mod.rs:806-816`).
    pub local_constraint_deg: usize,
    pub num_interactions: usize,
    pub num_monomials: usize,
    pub num_public_values: usize,
    /// `preprocessed?`, then `cached*`, then `common_main` — the eager order.
    pub mats: Vec<MatPlan>,
    pub has_preprocessed: bool,
    /// DAG rule-stream sizes for the constraint interpreter.
    pub zc_rules_len: usize,
    pub zc_used_nodes_len: usize,
    pub zc_buffer_size: u32,
    /// DAG rule-stream sizes for the interaction interpreter.
    pub logup_rules_len: usize,
    pub logup_used_nodes_len: usize,
    pub logup_buffer_size: u32,
    pub eval_kind: RoundEvalKind,
}

impl TracePlan {
    /// `max(n, 0)` — the number of MLE rounds this trace participates in.
    pub fn n_lift(&self) -> usize {
        self.n.max(0) as usize
    }

    /// Width of the folded matrix for `mats[i]`, doubled when the AIR needs
    /// its rotation (`air_width_for_mat`, `mod.rs:118`).
    pub fn folded_width(&self, i: usize) -> usize {
        self.mats[i].width * if self.need_rot { 2 } else { 1 }
    }
}

/// Everything [`logup_zerocheck_gpu_ir`] needs to unroll the phase.
///
/// The Fiat–Shamir values are host inputs; see the module docs, seam (3).
#[derive(Clone, Debug)]
pub struct ZerocheckPhasePlan {
    pub l_skip: usize,
    pub n_max: usize,
    /// `mpk.max_constraint_degree` — `sp_deg`, and the `num_x` every stage-D
    /// evaluator is launched with (`mod.rs:1396`).
    pub constraint_degree: usize,
    pub traces: Vec<TracePlan>,
    /// `xi` from fractional GKR, padded to `l_skip + n_global` (`mod.rs:246-248`).
    pub xi: Vec<EF>,
    /// `lambda` (`mod.rs:261`) and its powers (`mod.rs:654`).
    pub lambda_pows: Vec<EF>,
    /// `mu_pows`, `3 * num_traces` of them (`mod.rs:322-324`).
    pub mu_pows: Vec<EF>,
    /// `r_0 .. r_{n_max}` — `n_max + 1` values (`mod.rs:405`).
    pub r: Vec<EF>,
    /// `omega_skip_pows`, `2^l_skip` of them (`mod.rs:529`).
    pub omega_skip_pows: Vec<F>,
    /// `compute_barycentric_inv_lagrange_denoms(l_skip, ω*, r_0)` (`mod.rs:960`).
    pub inv_lagrange_denoms_r0: Vec<EF>,
    /// Per-trace `(is_first, is_last)` for `fold_selectors_round0` (`mod.rs:1040-1041`).
    pub fold_selector_scalars: Vec<(EF, EF)>,
    /// `denom_sum_init` per trace for the round-0 interaction evaluator.
    pub round0_denom_sum_init: Vec<EF>,
    /// Per-trace `g_shift` for the round-0 evaluators (`mod.rs:818-819`).
    pub round0_g_shift: Vec<F>,
    /// The `s_0_deg + 1` coefficients observed at `mod.rs:360`.
    pub s_0_coeffs: Vec<EF>,
    /// Per-trace `(sum_claim_p, sum_claim_q)` observed at `mod.rs:315-316`.
    pub logup_sum_claims: Vec<(EF, EF)>,
    /// `batch_s_evals[round - 1][i]` for `i in 0..s_deg` — the values
    /// observed at `mod.rs:394`.
    pub round_evals: Vec<Vec<EF>>,
    /// Column-opening claims observed at `mod.rs:415-428`, already in
    /// transcript order (common main first, then preprocessed/cached).
    pub opening_claims: Vec<EF>,
    /// `threads_per_block` for the batched launchers.
    pub threads_per_block: u32,
    /// `sm_count`-derived block count for the batched launchers.
    pub num_blocks: u32,
}

impl ZerocheckPhasePlan {
    pub fn num_traces(&self) -> usize {
        self.traces.len()
    }

    /// `s_deg = constraint_degree + 1` (`mod.rs:268`).
    pub fn s_deg(&self) -> usize {
        self.constraint_degree + 1
    }

    /// Distinct `n_lift` values, ascending — one `eq_xi` tree is built per
    /// distinct value (`mod.rs:752-758`).
    pub fn distinct_n_lifts(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.traces.iter().map(|t| t.n_lift()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

// ===========================================================================
// Per-trace device inputs.
// ===========================================================================

/// The `BufId`s of one trace's device-resident inputs.
///
/// A caller that has run keygen supplies these as const buffers holding the
/// real bytes; [`TraceBufs::alloc_zeroed`] allocates and zeroes them, which
/// is enough to *build and compile* the graph (used by
/// `examples/dump_ir_zerocheck_phase.rs` and by the shape tests).
#[derive(Clone, Debug)]
pub struct TraceBufs {
    /// `3 * 2^n_lift` base-field selector cube (`mod.rs:769-782`).
    pub selectors_cube: BufId,
    /// Folded `EF` selectors, written by `fold_selectors_round0`.
    pub selectors_folded: BufId,
    /// `*const F` table over `[cached…, common_main]` (`mod.rs:826-832`).
    pub main_ptrs: BufId,
    pub preprocessed: Option<BufId>,
    pub public_values: BufId,
    pub zc_rules: BufId,
    pub zc_used_nodes: BufId,
    pub logup_rules: BufId,
    /// Interaction-interpreter used-node stream (`interaction_rules.inner`).
    pub logup_used_nodes: BufId,
    /// `pair_idx = 2 * interaction_idx + is_denom` (`interaction_rules`).
    pub logup_pair_idxs: BufId,
    /// `eq_3b_per_trace[t]`, read by the stage-D logup evaluator through
    /// `LogupCtx::d_eq_3bs` (`mod.rs:677-707`).
    pub eq_3bs: BufId,
    pub numer_weights: BufId,
    pub denom_weights: BufId,
    /// One `BufId` per matrix, same order as [`TracePlan::mats`].
    pub mats: Vec<BufId>,
    /// Folded `EF` matrices produced by round 0's `fold_ple`.
    pub folded_mats: Vec<BufId>,
}

impl TraceBufs {
    /// Allocate every buffer a trace needs and zero it.
    // TODO(cc-ir): zeroed inputs make the graph buildable and compilable
    //   without a proving key, but running it produces garbage.
    // WHY: the real bytes come from keygen + trace generation
    //   (`DeviceMultiStarkProvingKey`, `ProvingContext`), which the graph
    //   builder deliberately does not depend on — mirroring
    //   `fractional_sumcheck_gpu_irv2`, which takes `leaves: BufId` and
    //   knows nothing about where the leaves came from.
    // RISK: none for graph-build / compile / dump. A `from_proving_ctx`
    //   bridge is the next piece of work; see the report.
    pub fn alloc_zeroed(
        g: &mut GraphBuilder,
        device: DeviceType,
        plan: &ZerocheckPhasePlan,
        t: usize,
    ) -> Self {
        let tp = &plan.traces[t];
        let n_lift = tp.n_lift();
        let cube = 1usize << n_lift;
        let num_x0 = (1usize << plan.l_skip).max(1);

        let selectors_cube = add_f_buf(g, device, &format!("t{t}_sels_cube"), 3 * cube);
        g.insert_memset(selectors_cube, 0);
        // Folded selectors are `3 * num_x` EFs (is_first / is_last /
        // is_transition). No memset: `fold_selectors_round0` is their producer,
        // and the graph is SSA — one writer per buffer.
        let selectors_folded = add_ef_buf(g, device, &format!("t{t}_sels_folded"), 3 * cube);

        let n_main = tp.mats.len() - usize::from(tp.has_preprocessed);
        let main_ptrs = add_typed_buf::<*const F>(g, device, &format!("t{t}_main_ptrs"), n_main);
        g.insert_memset(main_ptrs, 0);

        let public_values = add_f_buf(
            g,
            device,
            &format!("t{t}_public"),
            tp.num_public_values.max(1),
        );
        g.insert_memset(public_values, 0);

        let zc_rules =
            add_typed_buf::<u128>(g, device, &format!("t{t}_zc_rules"), tp.zc_rules_len.max(1));
        g.insert_memset(zc_rules, 0);
        let zc_used_nodes = add_typed_buf::<usize>(
            g,
            device,
            &format!("t{t}_zc_used_nodes"),
            tp.zc_used_nodes_len.max(1),
        );
        g.insert_memset(zc_used_nodes, 0);
        let logup_rules = add_typed_buf::<u128>(
            g,
            device,
            &format!("t{t}_logup_rules"),
            tp.logup_rules_len.max(1),
        );
        g.insert_memset(logup_rules, 0);
        let logup_used_nodes = add_typed_buf::<usize>(
            g,
            device,
            &format!("t{t}_logup_used_nodes"),
            tp.logup_used_nodes_len.max(1),
        );
        g.insert_memset(logup_used_nodes, 0);
        let logup_pair_idxs = add_typed_buf::<u32>(
            g,
            device,
            &format!("t{t}_logup_pair_idxs"),
            tp.logup_used_nodes_len.max(1),
        );
        g.insert_memset(logup_pair_idxs, 0);
        let eq_3bs = add_ef_buf(
            g,
            device,
            &format!("t{t}_eq_3bs"),
            tp.num_interactions.max(1),
        );
        g.insert_memset(eq_3bs, 0);

        let numer_weights = add_ef_buf(
            g,
            device,
            &format!("t{t}_numer_w"),
            tp.num_interactions.max(1),
        );
        g.insert_memset(numer_weights, 0);
        let denom_weights = add_ef_buf(
            g,
            device,
            &format!("t{t}_denom_w"),
            tp.num_interactions.max(1),
        );
        g.insert_memset(denom_weights, 0);

        let preprocessed = tp.has_preprocessed.then(|| {
            let b = add_f_buf(
                g,
                device,
                &format!("t{t}_prep"),
                tp.mats[0].width * tp.mats[0].height,
            );
            g.insert_memset(b, 0);
            b
        });

        let mut mats = Vec::with_capacity(tp.mats.len());
        let mut folded_mats = Vec::with_capacity(tp.mats.len());
        for (i, m) in tp.mats.iter().enumerate() {
            let b = add_f_buf(g, device, &format!("t{t}_mat{i}"), m.width * m.height);
            g.insert_memset(b, 0);
            mats.push(b);
            // Round 0 folds `height` rows down to `max(height >> l_skip, 1)`
            // and doubles the width when the AIR needs its rotation
            // (`fold_ple.rs:24-27`).
            let num_x = (m.height / num_x0).max(1);
            // Producer is `fold_ple_from_evals`; no memset (SSA).
            let f = add_ef_buf(
                g,
                device,
                &format!("t{t}_mat{i}_folded"),
                num_x * tp.folded_width(i),
            );
            folded_mats.push(f);
        }

        Self {
            selectors_cube,
            selectors_folded,
            main_ptrs,
            preprocessed,
            public_values,
            zc_rules,
            zc_used_nodes,
            logup_rules,
            logup_used_nodes,
            logup_pair_idxs,
            eq_3bs,
            numer_weights,
            denom_weights,
            mats,
            folded_mats,
        }
    }
}

// ===========================================================================
// The phase proof, as BufIds.
// ===========================================================================

/// What [`logup_zerocheck_gpu_ir`] leaves on the graph.
///
/// Mirrors `FracSumcheckProofIR` (`fractional_sumcheck_gpu_irv2.rs:441-446`):
/// proof artifacts stay as `BufId`s and the caller reads them out of the
/// compiled `GraphExe`.
#[derive(Clone, Debug)]
pub struct ZerocheckPhaseProofIR {
    /// Per-trace round-0 constraint evaluations (`mod.rs:841` output).
    pub round0_zc_evals: Vec<BufId>,
    /// Per-trace round-0 interaction evaluations (`mod.rs:896` output).
    pub round0_logup_evals: Vec<BufId>,
    /// `[round][0] = zerocheck evals, [round][1] = logup evals` for each of
    /// the `n_max` MLE rounds.
    pub round_evals: Vec<[Option<BufId>; 2]>,
    /// Per-trace, per-matrix final folded buffers — the column openings
    /// before the host-side doubled-width split.
    pub column_openings: Vec<Vec<BufId>>,
    /// Final sponge state after the whole phase.
    pub transcript_state: BufId,
}

// ===========================================================================
// The phase driver.
// ===========================================================================

/// Graph-IR mirror of [`super::prove_zerocheck_and_logup_gpu`], stages C, D
/// and E (`mod.rs:253-445`).
///
/// The round loop is unrolled at graph-build time with an ordinary Rust
/// `for`, exactly as `fractional_sumcheck_gpu_ir` unrolls its rounds: the
/// transcript's *control* state (`absorb_idx` / `sample_idx`) lives on the
/// host and picks which module each `observe` / `sample` emits, while the
/// sponge *state* threads through the graph as a `BufId`.
///
/// Stages A and B (grinding, `alpha_logup`/`beta_logup`, GKR input eval and
/// the fractional sumcheck) are **not** built here: grinding has no
/// graph-IR counterpart at all (`FiatShamirTranscriptGraphIR` has no
/// `grind`, `sponge_graph_ir.rs:71-85`), and the fractional phase is
/// [`super::fractional_ir`]'s. Seed the transcript with
/// [`crate::sponge_graph_ir::DuplexSpongeGpuIR::from_live`] so this graph
/// chains onto the live Fiat–Shamir stream instead of restarting it.
///
/// # Wiring
///
/// Nothing calls this yet — it is enablement, like `fractional_ir.rs`.
/// `mod.rs:237`'s eager call is untouched.
pub fn logup_zerocheck_gpu_ir<TS>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    plan: &ZerocheckPhasePlan,
    bufs: &[TraceBufs],
    device: DeviceType,
) -> ZerocheckPhaseProofIR
where
    TS: FiatShamirTranscriptGraphIR,
{
    assert_eq!(
        bufs.len(),
        plan.num_traces(),
        "one TraceBufs per trace required"
    );
    assert_eq!(
        plan.r.len(),
        plan.n_max + 1,
        "plan.r must hold r_0 .. r_{{n_max}}"
    );

    let num_traces = plan.num_traces();
    let l_skip = plan.l_skip;
    let sp_deg = plan.constraint_degree;
    let s_deg = plan.s_deg();
    let skip_domain = 1usize << l_skip;

    // -----------------------------------------------------------------------
    // STAGE C — univariate round 0 (`mod.rs:253-374`).
    // -----------------------------------------------------------------------

    // C.0 — `lambda = sample_ext()` (`mod.rs:261`).
    //
    // The node is emitted so the sponge advances in the graph exactly as it
    // does eagerly; the *value* is taken from the plan (module docs, seam 3).
    let _lambda_buf = transcript.sample_ext(g);

    let lambda_pows = ef_slice_const_buf(g, device, "lambda_pows", &plan.lambda_pows);
    let omega_skip_pows = f_slice_const_buf(g, device, "omega_skip_pows", &plan.omega_skip_pows);

    // C.1 — per-AIR lambda combinations for the monomial evaluators
    // (`mod.rs:664`). Emitted only for traces that use a monomial path.
    let mut lambda_combinations: Vec<Option<BufId>> = Vec::with_capacity(num_traces);
    for (t, tp) in plan.traces.iter().enumerate() {
        if tp.num_monomials == 0 || tp.eval_kind == RoundEvalKind::Dag {
            lambda_combinations.push(None);
            continue;
        }
        // TODO(cc-ir): the monomial headers / variable stream / lambda-term
        //   stream are keygen-static device buffers we do not carry in
        //   `TraceBufs`, so they are allocated (zeroed) here.
        // WHY: they live on `pk.per_air[..].zerocheck_monomials`, which the
        //   builder deliberately does not depend on (see `TraceBufs`).
        // RISK: same as `TraceBufs::alloc_zeroed` — build/compile only.
        let headers =
            add_typed_buf::<MonomialHeader>(g, device, &format!("t{t}_mono_hdr"), tp.num_monomials);
        g.insert_memset(headers, 0);
        let terms = add_typed_buf::<LambdaTerm<F>>(
            g,
            device,
            &format!("t{t}_lambda_terms"),
            tp.num_monomials,
        );
        g.insert_memset(terms, 0);
        let out = add_ef_buf(g, device, &format!("t{t}_lambda_comb"), tp.num_monomials);
        precompute_lambda_combinations_ir(
            g,
            headers,
            terms,
            lambda_pows,
            plan.lambda_pows.len(),
            out,
            tp.num_monomials as u32,
        );
        lambda_combinations.push(Some(out));
    }

    // C.2 — logup numerator / denominator combinations (`mod.rs:729`).
    //
    // `eq_3b_per_trace` is host-computed (`mod.rs:677-707`, an
    // `O(num_interactions · (n_logup − n_lift))` `EF` loop) and uploaded; it
    // enters the graph as a zeroed buffer for the same reason as above.
    let beta_pows = add_ef_buf(g, device, "beta_pows", plan.lambda_pows.len().max(1));
    g.insert_memset(beta_pows, 0);
    // `LogupCtx::d_challenges` — the interaction challenge vector the DAG
    // interpreter reads through `ENTRY_CHALLENGE` (`batch_mle.cu:224`).
    // TODO(cc-ir): allocated zeroed like the other keygen/challenge inputs.
    // WHY: the eager builder receives it as a raw `*const EF` from the
    //   caller (`batch_mle.rs:271`); it is not part of `TraceBufs`.
    // RISK: build/compile only, same as `TraceBufs::alloc_zeroed`. It is a
    //   *shared* buffer across traces, so it must stay one `BufId` — every
    //   `LogupCtx` points at the same allocation.
    let logup_challenges = add_ef_buf(g, device, "logup_challenges", plan.lambda_pows.len().max(1));
    g.insert_memset(logup_challenges, 0);
    let mut logup_combinations: Vec<Option<(BufId, BufId)>> = vec![None; num_traces];
    for (t, tp) in plan.traces.iter().enumerate() {
        if !tp.has_interactions || tp.num_monomials == 0 {
            continue;
        }
        let headers =
            add_typed_buf::<MonomialHeader>(g, device, &format!("t{t}_ia_hdr"), tp.num_monomials);
        g.insert_memset(headers, 0);
        let terms = add_typed_buf::<InteractionMonomialTerm<F>>(
            g,
            device,
            &format!("t{t}_ia_terms"),
            tp.num_monomials,
        );
        g.insert_memset(terms, 0);
        // Same `eq_3bs` the stage-D `LogupCtx` points at — one buffer, one
        // `BufId`, so the planner sees the shared read.
        let eq_3bs = bufs[t].eq_3bs;
        let numer_out = add_ef_buf(g, device, &format!("t{t}_numer_comb"), tp.num_monomials);
        let denom_out = add_ef_buf(g, device, &format!("t{t}_denom_comb"), tp.num_monomials);
        precompute_logup_numer_combinations_ir(
            g,
            headers,
            terms,
            eq_3bs,
            tp.num_interactions.max(1),
            numer_out,
            tp.num_monomials as u32,
        );
        precompute_logup_denom_combinations_ir(
            g,
            headers,
            terms,
            beta_pows,
            plan.lambda_pows.len().max(1),
            eq_3bs,
            tp.num_interactions.max(1),
            denom_out,
            tp.num_monomials as u32,
        );
        logup_combinations[t] = Some((numer_out, denom_out));
    }

    // C.3 — one `eq(xi[..], ·)` hypercube tree per distinct `n_lift`
    // (`mod.rs:752-758`). `eq_layers[n_lift][j]` is the `2^j`-sized layer.
    let mut eq_layers: std::collections::BTreeMap<usize, Vec<BufId>> = Default::default();
    let one_layer = ef_slice_const_buf(g, device, "eq_one", &[EF::ONE]);
    for n_lift in plan.distinct_n_lifts() {
        let mut layers = Vec::with_capacity(n_lift + 1);
        layers.push(one_layer);
        for i in 0..n_lift {
            let step = 1usize << i;
            let out = add_ef_buf(g, device, &format!("eq_n{n_lift}_l{}", i + 1), 2 * step);
            // `EqEvalLayers::new_rev` inserts `x_i` from the front, and the
            // eager caller passes `xi[l_skip + 1 ..]` (`mod.rs:758`).
            let x_i = plan.xi.get(l_skip + 1 + i).copied().unwrap_or(EF::ONE);
            eq_hypercube_interleaved_stage_ext_ir(
                g,
                *layers.last().unwrap(),
                out,
                x_i,
                step as u32,
            );
            layers.push(out);
        }
        eq_layers.insert(n_lift, layers);
    }

    // C.4 — per-trace round-0 evaluators (`mod.rs:841`, `mod.rs:896`).
    let mut round0_zc_evals = Vec::with_capacity(num_traces);
    let mut round0_logup_evals = Vec::with_capacity(num_traces);
    for (t, tp) in plan.traces.iter().enumerate() {
        let tb = &bufs[t];
        let n_lift = tp.n_lift();
        let num_x = 1u32 << n_lift;
        let height = tp.mats.last().map(|m| m.height).unwrap_or(1) as u32;
        let eq_cube = eq_layers[&n_lift][n_lift];

        // Constraint side.
        let num_cosets_zc = tp.local_constraint_deg.saturating_sub(1).max(1) as u32;
        let zc_out = add_ef_buf(
            g,
            device,
            &format!("t{t}_r0_zc_out"),
            (num_x as usize) * (num_cosets_zc as usize) * skip_domain,
        );
        // TODO(cc-ir): `tmp_sums` / `intermediates` are sized by the CUDA
        //   helpers `_zerocheck_r0_temp_sums_buffer_size` /
        //   `_zerocheck_r0_intermediates_buffer_size` (`round0.rs:51-127`),
        //   which take the *launch* geometry, not just the shape.
        // WHY: calling them here means duplicating `round0.rs`'s block-count
        //   computation; they are `unsafe extern` and cheap to call, but the
        //   grid sizing they depend on lives inside the launcher.
        // RISK: these buffers are over-allocated with a shape-derived bound
        //   below. Under-allocation would corrupt; over-allocation only
        //   wastes memory, so the bound is deliberately generous. A real port
        //   must call the sizing helpers.
        let scratch = (num_x as usize) * (num_cosets_zc as usize) * skip_domain;
        // `zc_tmp` and `zc_inter` are both node outputs: the kernel is their
        // sole producer, so neither may also be memset.
        let zc_tmp = add_ef_buf(g, device, &format!("t{t}_r0_zc_tmp"), scratch.max(1));
        let zc_inter = add_f_buf(
            g,
            device,
            &format!("t{t}_r0_zc_inter"),
            (scratch * tp.zc_buffer_size.max(1) as usize).max(1),
        );

        if tp.has_constraints {
            zerocheck_ntt_eval_constraints_ir(
                g,
                Round0ZcBufs {
                    tmp_sums: zc_tmp,
                    out: zc_out,
                    intermediates: zc_inter,
                    selectors_cube: tb.selectors_cube,
                    preprocessed: tb.preprocessed,
                    main_ptrs: tb.main_ptrs,
                    eq_cube,
                    lambda_pows,
                    public_values: tb.public_values,
                    rules: tb.zc_rules,
                    used_nodes: tb.zc_used_nodes,
                },
                Round0ZcShape {
                    rules_len: tp.zc_rules_len.max(1),
                    used_nodes_len: tp.zc_used_nodes_len.max(1),
                    lambda_pows_len: plan.lambda_pows.len(),
                    buffer_size: tp.zc_buffer_size,
                    skip_domain: skip_domain as u32,
                    num_x,
                    height,
                    num_cosets: num_cosets_zc,
                    g_shift: plan.round0_g_shift[t],
                    max_temp_bytes: 0,
                    tmp_sums_len: scratch.max(1),
                    out_len: (num_x as usize) * (num_cosets_zc as usize) * skip_domain,
                    intermediates_len: (scratch * tp.zc_buffer_size.max(1) as usize).max(1),
                    sels_len: 3 << n_lift,
                    main_ptrs_len: tp.mats.len() - usize::from(tp.has_preprocessed),
                    public_len: tp.num_public_values,
                },
            );
        } else {
            // No producer otherwise — keep the registered output well-formed.
            g.insert_memset(zc_out, 0);
        }
        g.register_output(zc_out);
        round0_zc_evals.push(zc_out);

        // Interaction side.
        let num_cosets_logup = tp.local_constraint_deg.max(1) as u32;
        let lg_len = (num_x as usize) * (num_cosets_logup as usize) * skip_domain;
        let lg_out = add_frac_buf(g, device, &format!("t{t}_r0_lg_out"), lg_len);
        let lg_tmp = add_frac_buf(g, device, &format!("t{t}_r0_lg_tmp"), lg_len.max(1));
        let lg_inter = add_f_buf(
            g,
            device,
            &format!("t{t}_r0_lg_inter"),
            (lg_len * tp.logup_buffer_size.max(1) as usize).max(1),
        );
        if tp.has_interactions {
            logup_bary_eval_interactions_round0_ir(
                g,
                Round0LogupBufs {
                    tmp_sums: lg_tmp,
                    out: lg_out,
                    intermediates: lg_inter,
                    selectors_cube: tb.selectors_cube,
                    preprocessed: tb.preprocessed,
                    main_ptrs: tb.main_ptrs,
                    eq_cube,
                    public_values: tb.public_values,
                    numer_weights: tb.numer_weights,
                    denom_weights: tb.denom_weights,
                    rules: tb.logup_rules,
                },
                Round0LogupShape {
                    rules_len: tp.logup_rules_len.max(1),
                    buffer_size: tp.logup_buffer_size,
                    skip_domain: skip_domain as u32,
                    num_x,
                    height,
                    num_cosets: num_cosets_logup,
                    g_shift: plan.round0_g_shift[t],
                    max_temp_bytes: 0,
                    tmp_sums_len: lg_len.max(1),
                    out_len: lg_len,
                    intermediates_len: (lg_len * tp.logup_buffer_size.max(1) as usize).max(1),
                    sels_len: 3 << n_lift,
                    main_ptrs_len: tp.mats.len() - usize::from(tp.has_preprocessed),
                    public_len: tp.num_public_values,
                    weights_len: tp.num_interactions.max(1),
                    denom_sum_init: plan.round0_denom_sum_init[t],
                },
            );
        } else {
            g.insert_memset(lg_out, 0);
        }
        g.register_output(lg_out);
        round0_logup_evals.push(lg_out);
    }

    // C.5 — HOST SEAM (module docs, seam 1).
    //
    // TODO(cc-ir): everything between the round-0 evaluators and `r_0` stays
    //   on the host: the D2H of `zc_out` / `lg_out` (`mod.rs:857, 914`), the
    //   transpose + `from_geometric_cosets_evals_idft` (`mod.rs:867-872,
    //   936-946`), the `s_0_logup_polys` / `s_0_zc_poly` DFT chains
    //   (`mod.rs:271-352`) and the `s_0_poly` assembly.
    // WHY: there is no device `EF` iDFT in this crate. `Radix2BowersSerial`
    //   is a p3 host DFT and the geometric-coset unshift + Lagrange matvec
    //   that follow it (`prover/poly.rs:621-683`) have no CUDA counterpart.
    // RISK: this is the phase's largest remaining host cost and it forces a
    //   full stream sync per trace. The graph below is still correct — the
    //   observed values come from the plan and the sponge chain is emitted
    //   in the eager order — but the phase is not device-resident across
    //   this point.
    for &(sum_p, sum_q) in &plan.logup_sum_claims {
        let p = ef_const_ext_scalar_buf(g, device, "sum_claim_p", sum_p);
        let q = ef_const_ext_scalar_buf(g, device, "sum_claim_q", sum_q);
        transcript.observe_ext(g, p);
        transcript.observe_ext(g, q);
    }
    // `mu = sample_ext()` (`mod.rs:322`).
    let _mu_buf = transcript.sample_ext(g);
    for (i, &c) in plan.s_0_coeffs.iter().enumerate() {
        let b = ef_const_ext_scalar_buf(g, device, &format!("s0_coeff{i}"), c);
        transcript.observe_ext(g, b);
    }
    // `r_0 = sample_ext()` (`mod.rs:367`).
    let _r0_buf = transcript.sample_ext(g);
    let r_0 = plan.r[0];

    // C.6 — `fold_ple_evals` (`mod.rs:373` → `:954`).
    let inv_denoms = ef_slice_const_buf(g, device, "inv_lagrange_denoms_r0", &{
        let mut v = plan.inv_lagrange_denoms_r0.clone();
        v.resize(skip_domain, EF::ZERO);
        v
    });
    // Freshest SSA name of each trace's folded matrices after round 0.
    let mut folded_after_r0: Vec<Vec<BufId>> = Vec::with_capacity(num_traces);
    for (t, tp) in plan.traces.iter().enumerate() {
        let tb = &bufs[t];
        let mut folded = Vec::with_capacity(tp.mats.len());
        for (i, m) in tp.mats.iter().enumerate() {
            let num_x = (m.height / skip_domain).max(1);
            fold_ple_from_evals_ir(
                g,
                tb.mats[i],
                m.width * m.height,
                None,
                tb.folded_mats[i],
                0,
                omega_skip_pows,
                skip_domain,
                inv_denoms,
                m.height as u32,
                m.width as u32,
                l_skip as u32,
                num_x as u32,
                false,
            );
            let mut newest = tb.folded_mats[i];
            if tp.need_rot {
                // Second launch writes the upper half of the *same* buffer
                // (`fold_ple.rs:38, 50`) — an SSA rename over the same slot.
                let rot = add_ef_buf(
                    g,
                    device,
                    &format!("t{t}_mat{i}_folded_rot"),
                    num_x * tp.folded_width(i),
                );
                g.alias_bufs(rot, tb.folded_mats[i]);
                fold_ple_from_evals_ir(
                    g,
                    tb.mats[i],
                    m.width * m.height,
                    Some(tb.folded_mats[i]),
                    rot,
                    num_x * m.width,
                    omega_skip_pows,
                    skip_domain,
                    inv_denoms,
                    m.height as u32,
                    m.width as u32,
                    l_skip as u32,
                    num_x as u32,
                    true,
                );
                newest = rot;
            }
            folded.push(newest);
        }
        folded_after_r0.push(folded);
        // `fold_selectors_round0` (`mod.rs:1044`).
        let (is_first, is_last) = plan.fold_selector_scalars[t];
        let num_x = 1usize << tp.n_lift();
        fold_selectors_round0_ir(
            g,
            tb.selectors_folded,
            tb.selectors_cube,
            is_first,
            is_last,
            num_x,
        );
    }
    let _ = r_0; // `r_0`'s only kernel use is baked into `inv_lagrange_denoms_r0`.

    // -----------------------------------------------------------------------
    // STAGE D — the `n_max` MLE rounds (`mod.rs:387-404`), fully unrolled.
    // -----------------------------------------------------------------------
    let mut round_evals: Vec<[Option<BufId>; 2]> = Vec::with_capacity(plan.n_max);
    // Current per-trace folded buffers; rebound each round by the fold.
    let mut cur_mats: Vec<Vec<BufId>> = folded_after_r0;
    let mut cur_sels: Vec<BufId> = bufs.iter().map(|b| b.selectors_folded).collect();

    for round in 1..=plan.n_max {
        // D.1 — per-trace column interpolation for traces still "early"
        // (`round <= n_lift`, `mod.rs:1183-1275`).
        let mut early: Vec<usize> = Vec::new();
        let mut late: Vec<usize> = Vec::new();
        for (t, tp) in plan.traces.iter().enumerate() {
            if !tp.has_constraints && !tp.has_interactions {
                continue;
            }
            let n_lift = tp.n_lift();
            if round <= n_lift {
                early.push(t);
            } else if round == n_lift + 1 {
                late.push(t);
            }
            // `round > n_lift + 1` is the EXHAUSTED case: the eager path only
            // scales host-side `tilde_evals` by `r_prev` (`mod.rs:1166-1181`)
            // and launches no kernel, so the graph emits nothing either.
        }

        for &t in &early {
            let tp = &plan.traces[t];
            let tb = &bufs[t];
            let log_num_y = tp.n_lift() - round;
            let num_y = 1usize << log_num_y;
            // Columns: selectors (3 wide) plus every folded matrix column.
            let num_columns: usize = 3
                + (0..tp.mats.len())
                    .map(|i| tp.folded_width(i))
                    .sum::<usize>();
            let columns =
                add_typed_buf::<*const EF>(g, device, &format!("t{t}_r{round}_cols"), num_columns);
            // TODO(cc-ir,T5): the column pointer table is still assembled
            //   host-side from the *current* folded buffers
            //   (`mod.rs:1189-1199`) — the ctx-pointer hole T5 closed for
            //   `ZerocheckCtx` / `LogupCtx`, in its last remaining form here.
            // WHY: `interpolate_columns` takes a bare `*const EF[]` table,
            //   not one of the three structs T5 scoped materializers for; the
            //   fix is a `_materialize_column_ptr` twin of
            //   `_materialize_main_matrix_ptr`, one node per column.
            // RISK: the planner cannot see through this table to
            //   `cur_mats[t]` / `cur_sels[t]`, so those are listed as
            //   explicit extra inputs of the node below to keep the ordering
            //   honest. Correct, but the table's contents are unwritten (the
            //   memset below leaves it null), so this node cannot RUN until
            //   the materializer twin lands.
            g.insert_memset(columns, 0);
            let interpolated = add_ef_buf(
                g,
                device,
                &format!("t{t}_r{round}_interp"),
                sp_deg * num_y * num_columns,
            );
            let mut srcs = cur_mats[t].clone();
            srcs.push(cur_sels[t]);
            interpolate_columns_ir(
                g,
                interpolated,
                sp_deg * num_y * num_columns,
                columns,
                &srcs,
                num_columns,
                sp_deg,
                num_y,
            );
            let _ = tb;
        }

        // D.2 — the batched evaluators. One launch per family per round,
        // matching the eager path's "gather then dispatch" shape
        // (`mod.rs:1281-1424`).
        let zc_traces: Vec<usize> = early
            .iter()
            .chain(late.iter())
            .copied()
            .filter(|&t| plan.traces[t].has_constraints)
            .collect();
        let lg_traces: Vec<usize> = early
            .iter()
            .chain(late.iter())
            .copied()
            .filter(|&t| plan.traces[t].has_interactions)
            .collect();

        // Per-trace stage-D device inputs, as `BufId`s. This is what the T5
        // ctx materializers consume: every pointer that used to be baked into
        // a host-assembled ctx struct is a graph edge here.
        let round_bufs: Vec<Option<RoundTraceBufs>> = (0..num_traces)
            .map(|t| {
                let tp = &plan.traces[t];
                let tb = &bufs[t];
                if !tp.has_constraints && !tp.has_interactions {
                    return None;
                }
                let n_lift = tp.n_lift();
                if round > n_lift + 1 {
                    return None;
                }
                // `num_y` halves each round; at `round == n_lift + 1` the
                // trace is on its last (single-`y`) round.
                let log_num_y = n_lift.saturating_sub(round);
                let has_prep = usize::from(tp.has_preprocessed);
                Some(RoundTraceBufs {
                    num_y: 1u32 << log_num_y,
                    selectors: cur_sels[t],
                    preprocessed: tp
                        .has_preprocessed
                        .then(|| (cur_mats[t][0], tp.mats[0].width as u32)),
                    mains: (has_prep..tp.mats.len())
                        .map(|i| (cur_mats[t][i], tp.mats[i].width as u32))
                        .collect(),
                    public_values: tb.public_values,
                    eq_xi: eq_layers[&n_lift][log_num_y],
                    zc_rules: tb.zc_rules,
                    zc_rules_len: tp.zc_rules_len.max(1),
                    zc_used_nodes: tb.zc_used_nodes,
                    zc_used_nodes_len: tp.zc_used_nodes_len.max(1),
                    zc_buffer_size: tp.zc_buffer_size,
                    logup_rules: tb.logup_rules,
                    logup_rules_len: tp.logup_rules_len.max(1),
                    logup_used_nodes: tb.logup_used_nodes,
                    logup_used_nodes_len: tp.logup_used_nodes_len.max(1),
                    logup_pair_idxs: tb.logup_pair_idxs,
                    logup_buffer_size: tp.logup_buffer_size,
                    eq_3bs: tb.eq_3bs,
                    challenges: logup_challenges,
                    lambda_combinations: lambda_combinations[t],
                    logup_combinations: logup_combinations[t],
                })
            })
            .collect();

        // One `MainMatrixPtrs` array per (round, trace), shared across the
        // two families — see [`MainDescCache`].
        let mut main_descs = MainDescCache::default();
        let zc_eval = (!zc_traces.is_empty()).then(|| {
            emit_zerocheck_round_eval(
                g,
                device,
                plan,
                round,
                &zc_traces,
                lambda_pows,
                &round_bufs,
                &mut main_descs,
            )
        });
        let lg_eval = (!lg_traces.is_empty()).then(|| {
            emit_logup_round_eval(
                g,
                device,
                plan,
                round,
                &lg_traces,
                &round_bufs,
                &mut main_descs,
            )
        });
        round_evals.push([zc_eval, lg_eval]);

        // D.3 — HOST SEAM (module docs, seam 2).
        //
        // TODO(cc-ir): `compute_batch_s_poly` (`mod.rs:1430-1502`) runs on the
        //   host between the evaluator output and `observe_ext`.
        // WHY: it is ~70 lines of `EF` algebra over `3 * num_traces`
        //   accumulands with a head/tail split on `round <= n_lift`, two
        //   running eq chains, an `EF` inverse (`mod.rs:1477-1483`) and a
        //   `lagrange_interpolate` over `s_deg` points. None of those exist
        //   as device kernels; the fractional twin's much smaller
        //   `reconstruct_s_evals` needed four purpose-built DSL modules
        //   (`fractional_ir.rs:1334-1500`) to close.
        // RISK: this is THE ring of the port. Until it closes, every round
        //   still ends in a D2H + host algebra + H2D, so the graph is a
        //   per-round DAG rather than a whole-phase one. Everything else
        //   here is structured so that replacing this block with device
        //   nodes is a local change: the observes below already take
        //   `BufId`s.
        for (i, &eval) in plan.round_evals[round - 1].iter().enumerate().take(s_deg) {
            let b = ef_const_ext_scalar_buf(g, device, &format!("r{round}_s{i}"), eval);
            transcript.observe_ext(g, b);
        }
        let _r_buf = transcript.sample_ext(g);
        let r_round = plan.r[round];

        // D.4 — `fold_mle_evals` (`mod.rs:403` → `:1505`).
        //
        // The eager path makes two `batch_fold_mle` launches: one over every
        // trace matrix with `height > 1`, one over the selectors
        // (`mod.rs:1557-1573, 1585`).
        let mut fold_mats: Vec<(usize, usize)> = Vec::new(); // (trace, mat)
        for (t, tp) in plan.traces.iter().enumerate() {
            if round > tp.n_lift() {
                continue;
            }
            for i in 0..tp.mats.len() {
                fold_mats.push((t, i));
            }
        }
        if !fold_mats.is_empty() {
            let n = fold_mats.len();
            let in_ptrs = add_typed_buf::<*const EF>(g, device, &format!("r{round}_fold_in"), n);
            let out_ptrs = add_typed_buf::<*mut EF>(g, device, &format!("r{round}_fold_out"), n);
            let widths = add_typed_buf::<u32>(g, device, &format!("r{round}_fold_w"), n);
            let logh = add_typed_buf::<u8>(g, device, &format!("r{round}_fold_logh"), n);
            for b in [in_ptrs, out_ptrs, widths, logh] {
                g.insert_memset(b, 0);
            }
            let mut max_cells = 0u32;
            let mut new_mats: Vec<(usize, usize, BufId)> = Vec::with_capacity(n);
            let mut srcs: Vec<BufId> = Vec::with_capacity(n);
            let mut dsts: Vec<BufId> = Vec::with_capacity(n);
            for &(t, i) in &fold_mats {
                let tp = &plan.traces[t];
                let out_h = 1usize << (tp.n_lift() - round);
                let w = tp.folded_width(i);
                max_cells = max_cells.max((out_h * w) as u32);
                let nb = add_ef_buf(
                    g,
                    device,
                    &format!("t{t}_mat{i}_r{round}_folded"),
                    (out_h * w).max(1),
                );
                srcs.push(cur_mats[t][i]);
                dsts.push(nb);
                new_mats.push((t, i, nb));
            }
            batch_fold_mle_ir(
                g, in_ptrs, out_ptrs, widths, logh, &srcs, &dsts, n as u16, max_cells, r_round,
            );
            for (t, i, nb) in new_mats {
                cur_mats[t][i] = nb;
            }
        }
        // Selector fold (`mod.rs:1585`).
        {
            let n = plan.traces.len();
            let in_ptrs = add_typed_buf::<*const EF>(g, device, &format!("r{round}_sfold_in"), n);
            let out_ptrs = add_typed_buf::<*mut EF>(g, device, &format!("r{round}_sfold_out"), n);
            let widths = add_typed_buf::<u32>(g, device, &format!("r{round}_sfold_w"), n);
            let logh = add_typed_buf::<u8>(g, device, &format!("r{round}_sfold_logh"), n);
            for b in [in_ptrs, out_ptrs, widths, logh] {
                g.insert_memset(b, 0);
            }
            let mut max_cells = 0u32;
            let mut new_sels = Vec::with_capacity(n);
            let mut srcs: Vec<BufId> = Vec::with_capacity(n);
            let mut dsts: Vec<BufId> = Vec::with_capacity(n);
            for (t, tp) in plan.traces.iter().enumerate() {
                let out_h = 1usize << tp.n_lift().saturating_sub(round);
                max_cells = max_cells.max((out_h * 3) as u32);
                let nb = add_ef_buf(
                    g,
                    device,
                    &format!("t{t}_sels_r{round}"),
                    (out_h * 3).max(1),
                );
                srcs.push(cur_sels[t]);
                dsts.push(nb);
                new_sels.push((t, nb));
            }
            batch_fold_mle_ir(
                g, in_ptrs, out_ptrs, widths, logh, &srcs, &dsts, n as u16, max_cells, r_round,
            );
            for (t, nb) in new_sels {
                cur_sels[t] = nb;
            }
        }
        // The eager path pops the top layer of every `eq_xi` tree here
        // (`mod.rs:1587-1592`); in the graph that is pure host bookkeeping
        // over the `BufId` vectors, and the layer buffer simply stops being
        // referenced.
        for layers in eq_layers.values_mut() {
            if layers.len() > 1 {
                layers.pop();
            }
        }
    }

    // -----------------------------------------------------------------------
    // STAGE E — column openings (`mod.rs:405-445`).
    // -----------------------------------------------------------------------
    //
    // TODO(cc-ir): the eager exit D2Hs every folded matrix
    //   (`transport_matrix_d2h_col_major`, `mod.rs:1621`), splits the doubled
    //   width into `(orig, rot)` (`mod.rs:1622-1659`) and reorders
    //   common-main-first (`mod.rs:1661-1699`) before observing.
    // WHY: both transforms are keygen-static *permutations* of a device
    //   buffer, so they are cheap to express as graph memcpys — but the
    //   permutation itself comes from `column_openings_by_rot` and the
    //   per-AIR matrix layout, which the plan does not carry yet.
    // RISK: the openings below are the raw folded buffers, in build order,
    //   not the proof's order. A consumer must apply the same permutation
    //   the eager path applies. The observes use the plan's already-ordered
    //   claim list, so the transcript is right regardless.
    for (i, &claim) in plan.opening_claims.iter().enumerate() {
        let b = ef_const_ext_scalar_buf(g, device, &format!("opening{i}"), claim);
        transcript.observe_ext(g, b);
    }
    let mut column_openings = Vec::with_capacity(num_traces);
    for mats in &cur_mats {
        for &m in mats {
            g.register_output(m);
        }
        column_openings.push(mats.clone());
    }

    // The final sponge state is a phase output: without it the whole
    // Fiat-Shamir chain is dead code and DCE removes every transcript node.
    let transcript_state = transcript_state_of(g, transcript);
    g.register_output(transcript_state);

    ZerocheckPhaseProofIR {
        round0_zc_evals,
        round0_logup_evals,
        round_evals,
        column_openings,
        transcript_state,
    }
}

/// Emit the round's zerocheck evaluator for `traces`, returning its output
/// buffer.
/// Which ctx struct a batched evaluator's per-AIR array holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CtxKind {
    /// `ZerocheckCtx` — built on the device by [`materialize_zerocheck_ctx_ir`].
    Zerocheck,
    /// `LogupCtx` — built on the device by [`materialize_logup_ctx_ir`].
    Logup,
    /// `MonomialAirCtx` — still a zeroed placeholder, see the TODO on
    /// [`LogupMonomialBufs`].
    Monomial,
}

/// One trace's stage-D device inputs at a given round, as `BufId`s.
///
/// This is what the T5 ctx materializers turn into one device-resident ctx
/// element. Every field that becomes a pointer inside `ZerocheckCtx` /
/// `LogupCtx` is a `BufId` here, which is exactly what keeps it a graph edge.
#[derive(Clone, Debug)]
pub struct RoundTraceBufs {
    /// `2^(n_lift - round)` for this trace at this round.
    pub num_y: u32,
    /// Folded `EF` selectors, freshest SSA name.
    pub selectors: BufId,
    /// Folded preprocessed matrix and its `air_width`, when the AIR has one.
    pub preprocessed: Option<(BufId, u32)>,
    /// Folded main matrices (`cached…`, `common_main`) and their `air_width`s,
    /// in the eager order.
    pub mains: Vec<(BufId, u32)>,
    pub public_values: BufId,
    /// The `2^(n_lift - round)`-sized `eq(xi, ·)` layer.
    pub eq_xi: BufId,
    pub zc_rules: BufId,
    pub zc_rules_len: usize,
    pub zc_used_nodes: BufId,
    pub zc_used_nodes_len: usize,
    pub zc_buffer_size: u32,
    pub logup_rules: BufId,
    pub logup_rules_len: usize,
    pub logup_used_nodes: BufId,
    pub logup_used_nodes_len: usize,
    pub logup_pair_idxs: BufId,
    pub logup_buffer_size: u32,
    pub eq_3bs: BufId,
    /// `LogupCtx::d_challenges`.
    pub challenges: BufId,
    /// Precomputed per-monomial lambda combinations, when the trace uses a
    /// monomial evaluator.
    pub lambda_combinations: Option<BufId>,
    /// Precomputed `(numerator, denominator)` monomial combinations.
    pub logup_combinations: Option<(BufId, BufId)>,
}

/// Emit the per-trace `MainMatrixPtrs<EF>` descriptor array on the device.
///
/// The eager path builds this array on the host out of raw pointers
/// (`mod.rs:1131-1149, 1224-1256`); here each element is written by a
/// materializer node whose `data` pointer is a resolved `BufId`, so the
/// planner keeps an edge from the folded matrix to every evaluator that
/// dereferences it.
///
/// Returns `(array, reads)`. `reads` is the union of the element
/// materializers' declared reads — the folded main matrices the array points
/// at. The array itself is *not* in `reads`: it reaches the evaluator through
/// the ctx materializer, which binds it as `EvalCoreCtxArgs::d_main`.
fn emit_main_matrix_ptrs(
    g: &mut GraphBuilder,
    device: DeviceType,
    tag: &str,
    rb: &RoundTraceBufs,
) -> (BufId, Vec<BufId>) {
    let arr = add_typed_buf::<MainMatrixPtrs<EF>>(
        g,
        device,
        &format!("{tag}_main_desc"),
        rb.mains.len().max(1),
    );
    g.insert_memset(arr, 0);
    let mut reads = Vec::with_capacity(rb.mains.len());
    for (i, &(buf, air_width)) in rb.mains.iter().enumerate() {
        extend_reads(
            &mut reads,
            materialize_main_matrix_ptr_ir(g, arr, i as u32, DevicePtrArg::buf(buf), air_width),
        );
    }
    (arr, reads)
}

/// One `MainMatrixPtrs<EF>` descriptor array per `(round, trace)`, shared by
/// the zerocheck and logup evaluator families.
///
/// Before this cache both families emitted their own array for the same AIR
/// — `1 + |mains|` duplicate nodes and a duplicate pool allocation per AIR
/// per round, for byte-identical contents.
///
/// **The key is the trace index, never the position in the family's loop.**
/// The two loops iterate *different* trace subsets (`has_constraints` vs
/// `has_interactions`), so `air` numbers the same slot to different AIRs in
/// the two families; keying on it would silently hand one AIR's main
/// matrices to another.
#[derive(Default)]
struct MainDescCache {
    entries: std::collections::BTreeMap<usize, (BufId, Vec<BufId>)>,
}

impl MainDescCache {
    /// The descriptor array for `trace`, emitting it on first use.
    ///
    /// `reads` is returned on cache hits too: *every* evaluator that
    /// dereferences the array must declare the matrices it points at, so a
    /// shared array means a shared read set, not a read set declared once.
    fn get_or_emit(
        &mut self,
        g: &mut GraphBuilder,
        device: DeviceType,
        round: usize,
        trace: usize,
        rb: &RoundTraceBufs,
    ) -> (BufId, Vec<BufId>) {
        if let Some(hit) = self.entries.get(&trace) {
            return hit.clone();
        }
        let entry = emit_main_matrix_ptrs(g, device, &format!("r{round}_t{trace}"), rb);
        self.entries.insert(trace, entry.clone());
        entry
    }
}

/// `EvalCoreCtxArgs` + the reads it implies, shared by both DAG ctx kinds.
fn eval_core_args(main_desc: BufId, rb: &RoundTraceBufs) -> EvalCoreCtxArgs {
    EvalCoreCtxArgs {
        d_selectors: DevicePtrArg::buf(rb.selectors),
        d_preprocessed_data: match rb.preprocessed {
            Some((b, _)) => DevicePtrArg::buf(b),
            None => DevicePtrArg::NULL,
        },
        preprocessed_air_width: rb.preprocessed.map(|(_, w)| w).unwrap_or(0),
        d_main: DevicePtrArg::buf(main_desc),
        d_public: DevicePtrArg::buf(rb.public_values),
    }
}

/// Allocate the per-AIR `d_intermediates` scratch, or `None` when
/// `buffer_size == 0` (the eager path passes null there,
/// `batch_mle.rs:164-176`).
fn alloc_intermediates(
    g: &mut GraphBuilder,
    device: DeviceType,
    tag: &str,
    buffer_size: u32,
    num_x: u32,
    num_y: u32,
    logup: bool,
) -> Option<BufId> {
    if buffer_size == 0 {
        return None;
    }
    let len = unsafe {
        if logup {
            _logup_batch_mle_intermediates_buffer_size(buffer_size, num_x, num_y)
        } else {
            _zerocheck_batch_mle_intermediates_buffer_size(buffer_size, num_x, num_y)
        }
    };
    let b = add_ef_buf(g, device, &format!("{tag}_inter"), len.max(1));
    // The materializer *reads* this buffer's address before the evaluator
    // writes its contents, and the graph requires a writer to precede every
    // reader in insertion order. There is no "address-of" edge, so the
    // scratch gets an explicit zero producer.
    // TODO(cc-ir,T5): one extra memset per AIR per round purely to satisfy
    //   write-before-read on an address-only dependency.
    // WHY: `insert_blackbox_kernel` declares whole-buffer reads/writes; the
    //   materializer wants "I need your pointer, I touch no bytes".
    // RISK: performance only (a tiny memset), never correctness — the DAG
    //   interpreter writes every intermediate slot before reading it.
    g.insert_memset(b, 0);
    Some(b)
}

/// Materialize `air_ctxs[air]` as a `ZerocheckCtx` and record the accesses it
/// adds to the evaluator node.
///
/// `main_desc` is the shared descriptor array for this `(round, trace)`
/// ([`MainDescCache`]); its own reads are folded in by the caller, which is
/// the only one that knows whether the array was freshly emitted or reused.
#[allow(clippy::too_many_arguments)]
fn emit_zerocheck_ctx_element(
    g: &mut GraphBuilder,
    device: DeviceType,
    tag: &str,
    bufs: &mut ZerocheckEvalBufs,
    air: u32,
    rb: &RoundTraceBufs,
    num_x: u32,
    main_desc: BufId,
) {
    let tag = format!("{tag}_a{air}");
    let inter = alloc_intermediates(
        g,
        device,
        &tag,
        rb.zc_buffer_size,
        num_x,
        rb.num_y,
        /* logup */ false,
    );
    let reads = materialize_zerocheck_ctx_ir(
        g,
        bufs.air_ctxs,
        air,
        ZerocheckCtxArgs {
            eval_ctx: eval_core_args(main_desc, rb),
            d_intermediates: inter.map(DevicePtrArg::buf).unwrap_or(DevicePtrArg::NULL),
            num_y: rb.num_y,
            d_eq_xi: DevicePtrArg::buf(rb.eq_xi),
            d_rules: DevicePtrArg::buf(rb.zc_rules),
            rules_len: rb.zc_rules_len,
            d_used_nodes: DevicePtrArg::buf(rb.zc_used_nodes),
            used_nodes_len: rb.zc_used_nodes_len,
            buffer_size: rb.zc_buffer_size,
        },
    );
    extend_reads(&mut bufs.ctx_reads, reads);
    bufs.intermediates.extend(inter);
}

/// Materialize `air_ctxs[air]` as a `LogupCtx` and record the accesses it adds
/// to the evaluator node.
///
/// `main_desc` is the shared descriptor array for this `(round, trace)` —
/// see [`emit_zerocheck_ctx_element`].
#[allow(clippy::too_many_arguments)]
fn emit_logup_ctx_element(
    g: &mut GraphBuilder,
    device: DeviceType,
    tag: &str,
    bufs: &mut ZerocheckEvalBufs,
    air: u32,
    rb: &RoundTraceBufs,
    num_x: u32,
    main_desc: BufId,
) {
    let tag = format!("{tag}_a{air}");
    let inter = alloc_intermediates(
        g,
        device,
        &tag,
        rb.logup_buffer_size,
        num_x,
        rb.num_y,
        /* logup */ true,
    );
    let reads = materialize_logup_ctx_ir(
        g,
        bufs.air_ctxs,
        air,
        LogupCtxArgs {
            eval_ctx: eval_core_args(main_desc, rb),
            d_intermediates: inter.map(DevicePtrArg::buf).unwrap_or(DevicePtrArg::NULL),
            num_y: rb.num_y,
            d_eq_xi: DevicePtrArg::buf(rb.eq_xi),
            d_challenges: DevicePtrArg::buf(rb.challenges),
            d_eq_3bs: DevicePtrArg::buf(rb.eq_3bs),
            d_rules: DevicePtrArg::buf(rb.logup_rules),
            rules_len: rb.logup_rules_len,
            d_used_nodes: DevicePtrArg::buf(rb.logup_used_nodes),
            d_pair_idxs: DevicePtrArg::buf(rb.logup_pair_idxs),
            used_nodes_len: rb.logup_used_nodes_len,
            buffer_size: rb.logup_buffer_size,
        },
    );
    extend_reads(&mut bufs.ctx_reads, reads);
    bufs.intermediates.extend(inter);
}

/// Principle-1 access set for the monomial evaluators, whose ctx arrays are
/// still zeroed placeholders (T5 covers only the three DAG ctx structs).
///
/// The reads are declared anyway: an under-declared access set is a real
/// correctness bug the moment the ctx arrays are filled, and declaring them
/// now costs nothing but a few planner edges.
// TODO(cc-ir,R4): this is the last *hand-written* read list in the phase.
//   The three DAG ctx structs derive theirs from the materializer that binds
//   them ([`declared_reads`]), so they cannot drift; this one can.
// WHY: there is nothing to derive it from. `MonomialAirCtx`,
//   `LogupMonomialCommonCtx` and `LogupMonomialCtx` have no materializers yet
//   — their arrays are `memset` to zero (`alloc_eval_bufs`), so no
//   `PtrBinder` exists whose inputs could be returned.
// RISK: `evaluator_declares_every_materialized_buffer` cannot police this
//   list — its closure over a monomial ctx array is the bare array, so the
//   assertion is satisfied vacuously for those nodes. A monomial ctx field
//   added without a matching line here is silent, exactly as the DAG structs
//   were before R4. Closing it is DA step 5.3 (write the monomial
//   materializers); the assertion then starts policing them for free.
fn push_monomial_reads(reads: &mut Vec<BufId>, rb: &RoundTraceBufs) {
    reads.push(rb.selectors);
    reads.push(rb.public_values);
    reads.push(rb.eq_xi);
    reads.extend(rb.mains.iter().map(|&(b, _)| b));
    if let Some((b, _)) = rb.preprocessed {
        reads.push(b);
    }
    reads.extend(rb.lambda_combinations);
    if let Some((numer, denom)) = rb.logup_combinations {
        reads.extend([numer, denom]);
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_zerocheck_round_eval(
    g: &mut GraphBuilder,
    device: DeviceType,
    plan: &ZerocheckPhasePlan,
    round: usize,
    traces: &[usize],
    lambda_pows: BufId,
    round_bufs: &[Option<RoundTraceBufs>],
    main_descs: &mut MainDescCache,
) -> BufId {
    let num_airs = traces.len() as u32;
    let num_x = plan.constraint_degree as u32;
    let shape = BatchEvalShape {
        num_blocks: plan.num_blocks.max(num_airs),
        num_x,
        num_airs,
        threads_per_block: plan.threads_per_block,
        tmp_sums_len: (plan.num_blocks.max(num_airs) as usize) * (num_x as usize),
        out_len: (num_airs as usize) * (num_x as usize),
        lambda_pows_len: plan.lambda_pows.len(),
        // TODO(cc-ir): `chunk_size` is auto-tuned by a host loop
        //   (`batch_mle_monomial.rs:356-370`) that measures occupancy.
        // WHY: the loop needs the launcher's block count and shared-memory
        //   budget; reproducing it here would duplicate that logic.
        // RISK: a wrong `chunk_size` changes performance, not results — the
        //   kernel loops over `y` in chunks either way. `1` is the safe
        //   floor.
        chunk_size: 1,
    };
    let kind = plan.traces[traces[0]].eval_kind;
    let tag = format!("zc_r{round}");
    let mut bufs = alloc_eval_bufs(
        g,
        device,
        &tag,
        shape,
        Some(lambda_pows),
        /* frac_out */ false,
        if kind == RoundEvalKind::Dag {
            CtxKind::Zerocheck
        } else {
            CtxKind::Monomial
        },
    );
    for (air, &t) in traces.iter().enumerate() {
        let rb = round_bufs[t]
            .as_ref()
            .expect("stage-D trace bufs for an evaluated trace");
        if kind == RoundEvalKind::Dag {
            let (main_desc, desc_reads) = main_descs.get_or_emit(g, device, round, t, rb);
            extend_reads(&mut bufs.ctx_reads, desc_reads);
            emit_zerocheck_ctx_element(
                g, device, &tag, &mut bufs, air as u32, rb, num_x, main_desc,
            );
        } else {
            push_monomial_reads(&mut bufs.ctx_reads, rb);
        }
    }
    match kind {
        RoundEvalKind::Dag => zerocheck_batch_eval_mle_ir(g, &bufs, shape),
        RoundEvalKind::Monomial => zerocheck_monomial_batched_ir(g, &bufs, shape, false),
        RoundEvalKind::MonomialParY => zerocheck_monomial_batched_ir(g, &bufs, shape, true),
    }
    g.register_output(bufs.out);
    bufs.out
}

/// Emit the round's logup evaluator for `traces`, returning its output
/// buffer.
fn emit_logup_round_eval(
    g: &mut GraphBuilder,
    device: DeviceType,
    plan: &ZerocheckPhasePlan,
    round: usize,
    traces: &[usize],
    round_bufs: &[Option<RoundTraceBufs>],
    main_descs: &mut MainDescCache,
) -> BufId {
    let num_airs = traces.len() as u32;
    let num_x = plan.constraint_degree as u32;
    let shape = BatchEvalShape {
        num_blocks: plan.num_blocks.max(num_airs),
        num_x,
        num_airs,
        threads_per_block: plan.threads_per_block,
        tmp_sums_len: (plan.num_blocks.max(num_airs) as usize) * (num_x as usize),
        out_len: (num_airs as usize) * (num_x as usize),
        lambda_pows_len: plan.lambda_pows.len(),
        chunk_size: 1,
    };
    let use_monomial = plan.traces[traces[0]].eval_kind != RoundEvalKind::Dag;
    let tag = format!("lg_r{round}");
    if use_monomial {
        let n = num_airs as usize;
        let tmp_sums = add_frac_buf(g, device, &format!("{tag}_tmp"), shape.tmp_sums_len);
        let out = add_frac_buf(g, device, &format!("{tag}_out"), shape.out_len);
        let block_ctxs = add_typed_buf::<BlockCtx>(
            g,
            device,
            &format!("{tag}_blocks"),
            shape.num_blocks as usize,
        );
        let common_ctxs =
            add_typed_buf::<LogupMonomialCommonCtx>(g, device, &format!("{tag}_common"), n);
        let numer_ctxs = add_typed_buf::<LogupMonomialCtx>(g, device, &format!("{tag}_numer"), n);
        let denom_ctxs = add_typed_buf::<LogupMonomialCtx>(g, device, &format!("{tag}_denom"), n);
        let air_block_offsets = add_typed_buf::<u32>(g, device, &format!("{tag}_offsets"), n + 1);
        for b in [
            block_ctxs,
            common_ctxs,
            numer_ctxs,
            denom_ctxs,
            air_block_offsets,
        ] {
            g.insert_memset(b, 0);
        }
        let mut ctx_reads = Vec::new();
        for &t in traces {
            let rb = round_bufs[t]
                .as_ref()
                .expect("stage-D trace bufs for an evaluated trace");
            push_monomial_reads(&mut ctx_reads, rb);
        }
        logup_monomial_batched_ir(
            g,
            &LogupMonomialBufs {
                tmp_sums,
                out,
                block_ctxs,
                common_ctxs,
                numer_ctxs,
                denom_ctxs,
                air_block_offsets,
                ctx_reads,
            },
            shape,
        );
        g.register_output(out);
        out
    } else {
        let mut bufs = alloc_eval_bufs(
            g,
            device,
            &tag,
            shape,
            None,
            /* frac_out */ true,
            CtxKind::Logup,
        );
        for (air, &t) in traces.iter().enumerate() {
            let rb = round_bufs[t]
                .as_ref()
                .expect("stage-D trace bufs for an evaluated trace");
            let (main_desc, desc_reads) = main_descs.get_or_emit(g, device, round, t, rb);
            extend_reads(&mut bufs.ctx_reads, desc_reads);
            emit_logup_ctx_element(g, device, &tag, &mut bufs, air as u32, rb, num_x, main_desc);
        }
        logup_batch_eval_mle_ir(g, &bufs, shape);
        g.register_output(bufs.out);
        bufs.out
    }
}

/// Allocate (and zero) the ctx / scratch buffers one batched evaluator needs.
///
/// The ctx array is only *allocated* here; its elements are written by the T5
/// materializer nodes (device-side), not staged from the host.
fn alloc_eval_bufs(
    g: &mut GraphBuilder,
    device: DeviceType,
    tag: &str,
    shape: BatchEvalShape,
    lambda_pows: Option<BufId>,
    frac_out: bool,
    ctx_kind: CtxKind,
) -> ZerocheckEvalBufs {
    let n = shape.num_airs as usize;
    let (tmp_sums, out) = if frac_out {
        (
            add_frac_buf(g, device, &format!("{tag}_tmp"), shape.tmp_sums_len),
            add_frac_buf(g, device, &format!("{tag}_out"), shape.out_len),
        )
    } else {
        (
            add_ef_buf(g, device, &format!("{tag}_tmp"), shape.tmp_sums_len),
            add_ef_buf(g, device, &format!("{tag}_out"), shape.out_len),
        )
    };
    let block_ctxs = add_typed_buf::<BlockCtx>(
        g,
        device,
        &format!("{tag}_blocks"),
        shape.num_blocks as usize,
    );
    let air_ctxs = match ctx_kind {
        CtxKind::Zerocheck => add_typed_buf::<ZerocheckCtx>(g, device, &format!("{tag}_ctxs"), n),
        CtxKind::Logup => add_typed_buf::<LogupCtx>(g, device, &format!("{tag}_ctxs"), n),
        CtxKind::Monomial => add_typed_buf::<MonomialAirCtx>(g, device, &format!("{tag}_ctxs"), n),
    };
    let air_block_offsets = add_typed_buf::<u32>(g, device, &format!("{tag}_offsets"), n + 1);
    // The ctx array is memset first so the materializers' per-element writes
    // chain off a defined value and ABI padding is deterministic.
    for b in [block_ctxs, air_ctxs, air_block_offsets] {
        g.insert_memset(b, 0);
    }
    ZerocheckEvalBufs {
        tmp_sums,
        out,
        block_ctxs,
        air_ctxs,
        air_block_offsets,
        lambda_pows,
        ctx_reads: Vec::new(),
        intermediates: Vec::new(),
    }
}

/// The transcript's current state buffer.
// TODO(cc-ir): `FiatShamirTranscriptGraphIR` has no `state_buf` accessor, so
//   the phase driver samples one extra `[1, D_EF]` buffer to name the tail of
//   the sponge chain.
// WHY: `DuplexSpongeGpuIR::state_buf` is inherent, not part of the trait
//   (`sponge_graph_ir.rs:180`), and the driver is generic over `TS`.
// RISK: the extra `sample_ext` ADVANCES the transcript one squeeze past the
//   eager phase. Callers that continue the Fiat-Shamir stream after this
//   phase must account for it — or the trait should grow `state_buf`, which
//   is the right fix and is IMPL-A's file to change.
fn transcript_state_of<TS: FiatShamirTranscriptGraphIR>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
) -> BufId {
    transcript.sample_ext(g)
}

/// Build a minimal but structurally faithful plan: `num_traces` AIRs, all
/// of the same height, one common-main matrix each.
///
/// Used by `examples/dump_ir_zerocheck_phase.rs` and by the shape tests; it
/// is the analogue of the fractional port's `make_synthetic_leaves`.
pub fn synthetic_plan(num_traces: usize, l_skip: usize, n_max: usize) -> ZerocheckPhasePlan {
    let constraint_degree = 3usize;
    let s_deg = constraint_degree + 1;
    let ef = |i: usize| EF::from_usize(i + 1);
    let height = 1usize << (l_skip + n_max);
    let traces = (0..num_traces)
        .map(|t| TracePlan {
            n: n_max as isize,
            need_rot: t % 2 == 0,
            has_constraints: true,
            has_interactions: true,
            local_constraint_deg: constraint_degree,
            num_interactions: 4,
            num_monomials: 8,
            num_public_values: 2,
            mats: vec![MatPlan { width: 4, height }],
            has_preprocessed: false,
            zc_rules_len: 16,
            zc_used_nodes_len: 8,
            zc_buffer_size: 4,
            logup_rules_len: 16,
            logup_used_nodes_len: 8,
            logup_buffer_size: 4,
            eval_kind: RoundEvalKind::Dag,
        })
        .collect();
    ZerocheckPhasePlan {
        l_skip,
        n_max,
        constraint_degree,
        traces,
        xi: (0..l_skip + n_max + 1).map(ef).collect(),
        lambda_pows: (0..8).map(ef).collect(),
        mu_pows: (0..3 * num_traces).map(ef).collect(),
        r: (0..=n_max).map(ef).collect(),
        omega_skip_pows: (0..1 << l_skip).map(F::from_usize).collect(),
        inv_lagrange_denoms_r0: (0..1 << l_skip).map(ef).collect(),
        fold_selector_scalars: (0..num_traces).map(|t| (ef(t), ef(t + 1))).collect(),
        round0_denom_sum_init: (0..num_traces).map(ef).collect(),
        round0_g_shift: (0..num_traces).map(|_| F::ONE).collect(),
        s_0_coeffs: (0..=(1 << l_skip) * s_deg).map(ef).collect(),
        logup_sum_claims: (0..num_traces).map(|t| (ef(t), ef(t + 1))).collect(),
        round_evals: (0..n_max).map(|_| (0..s_deg).map(ef).collect()).collect(),
        opening_claims: (0..2 * num_traces).map(ef).collect(),
        threads_per_block: 128,
        num_blocks: 32,
    }
}

// ===========================================================================
// Tests.
// ===========================================================================

#[cfg(test)]
mod zerocheck_ir_tests {
    use crypto_compiler::{
        graph_compiler::GraphCompiler,
        graph_ir::{DeviceType, GraphBuilder, GraphNode},
        planner::{ListSchedulerV1, SchedulerMode},
    };
    use openvm_cuda_common::{
        common::get_device,
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::{cudaMemsetAsync, DeviceBuffer},
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{cuda::logup_zerocheck::fold_selectors_round0, sponge_graph_ir::DuplexSpongeGpuIR};

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        }
    }

    /// Byte view of an `[EF]` slice.
    fn ef_bytes(xs: &[EF]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)) }
    }

    /// Compile a graph with no runtime inputs, run it, and read back the
    /// given buffers as raw bytes.
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
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
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

    /// The blackbox nodes that dereference a ctx array — every batched
    /// evaluator in stage D.
    const EVAL_NODE_NAMES: [&str; 5] = [
        "zerocheck_batch_eval_mle",
        "logup_batch_eval_mle",
        "zerocheck_monomial_batched",
        "zerocheck_monomial_par_y_batched",
        "logup_monomial_batched",
    ];

    fn buf_name(g: &GraphBuilder, b: BufId) -> &str {
        g.bufs[b.0].name.as_deref().unwrap_or("<unnamed>")
    }

    /// Every buffer reachable from `root` by following materializer nodes.
    ///
    /// A materializer's input 0 is the array it writes and inputs `1..` are
    /// the pointer fields it stores into that array
    /// ([`PtrBinder::new`]), so the fixpoint over "a materializer whose
    /// target is already in the set contributes its pointer fields" is
    /// exactly the transitive dereference closure of the ctx array — two
    /// levels deep here, since a ctx element points at the
    /// `MainMatrixPtrs` array which points at the folded main matrices.
    fn materializer_closure(g: &GraphBuilder, root: BufId) -> Vec<BufId> {
        let mut seen = vec![root];
        loop {
            let before = seen.len();
            for node in &g.nodes {
                let GraphNode::BlackboxKernel(k) = node else {
                    continue;
                };
                if !k.name.starts_with("materialize_")
                    || k.inputs.is_empty()
                    || !seen.contains(&k.inputs[0])
                {
                    continue;
                }
                for &b in &k.inputs[1..] {
                    if !seen.contains(&b) {
                        seen.push(b);
                    }
                }
            }
            if seen.len() == before {
                return seen;
            }
        }
    }

    /// R4 step 1: an evaluator must declare every buffer the materializers
    /// reached through its ctx array.
    ///
    /// This is the machine check for the property this phase's correctness
    /// rests on. The compiler has **no pointer analysis**: a blackbox node's
    /// entire access model is the hand-supplied `(inputs, outputs, modifies)`
    /// triple (`graph_ir.rs:1259-1266`), folded into `NodeAccess { reads,
    /// writes }` verbatim (`planner/ctx.rs:26-29, 274-281`), and
    /// `verify_graph` skips blackbox nodes on its first line
    /// (`graph_ir.rs:2191-2192`). So an evaluator that dereferences
    /// `ctx->d_selectors` without declaring `selectors` gives the packer
    /// permission to hand that pool slot to another buffer
    /// (`planner/heuristic.rs:184-190`) and then reads garbage — silently,
    /// with no compile error and no other test failing.
    ///
    /// Before R4 the declared set was retyped by hand at the emit sites
    /// beside the materializer call. Now it is derived from the same
    /// [`PtrBinder`] the materializer binds ([`declared_reads`]), so the two
    /// cannot drift; this test is what proves the derivation actually
    /// reaches the evaluator node.
    #[test]
    fn evaluator_declares_every_materialized_buffer() {
        let device = DeviceType::Cuda(0);
        let plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 2, /* n_max */ 3,
        );

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_zeroed(&mut g, device, &plan, t))
            .collect();
        let _ = logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device);

        let mut evaluators = 0usize;
        let mut widest = 0usize;
        for node in &g.nodes {
            let GraphNode::BlackboxKernel(k) = node else {
                continue;
            };
            if !EVAL_NODE_NAMES.contains(&k.name.as_str()) {
                continue;
            }
            // `air_ctxs` is input 1 of every batched evaluator — the fixed
            // positional prefix `eval_node_bindings` is called with.
            let ctx_arr = k.inputs[1];
            let closure = materializer_closure(&g, ctx_arr);
            widest = widest.max(closure.len());
            for b in closure {
                assert!(
                    k.inputs.contains(&b),
                    "`{}` does not declare `{}` ({b:?}), which its ctx array `{}` \
                     ({ctx_arr:?}) points at (directly or through a pointer table). \
                     An undeclared read is a pool-reuse corruption hazard.",
                    k.name,
                    buf_name(&g, b),
                    buf_name(&g, ctx_arr),
                );
            }
            evaluators += 1;
        }
        assert!(evaluators > 0, "fixture emitted no batched evaluator nodes");
        assert!(
            widest > 1,
            "every closure was the bare ctx array — the assertion is vacuous"
        );
    }

    /// R4 step 3: one `MainMatrixPtrs` array per `(round, trace)`, shared by
    /// the zerocheck and logup families, and never crossed between AIRs.
    ///
    /// The fixture is deliberately *asymmetric*: trace 0 has constraints
    /// only and trace 2 interactions only, so `zc_traces == [0, 1]` while
    /// `lg_traces == [1, 2]`. That is the only shape in which keying the
    /// descriptor cache on the loop position (`air`) rather than the trace
    /// index goes wrong — and it goes wrong silently, by handing one AIR's
    /// folded main matrices to a different AIR's evaluator.
    #[test]
    fn main_matrix_ptrs_shared_per_trace_across_families() {
        let device = DeviceType::Cuda(0);
        let mut plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 1, /* n_max */ 2,
        );
        plan.traces[0].has_interactions = false;
        plan.traces[2].has_constraints = false;

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_zeroed(&mut g, device, &plan, t))
            .collect();
        let _ = logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device);

        // --- one array per (round, trace), not per (round, family, trace)
        let mut got: Vec<String> = g
            .bufs
            .iter()
            .filter_map(|b| b.name.clone())
            .filter(|n| n.ends_with("_main_desc"))
            .collect();
        let mut want: Vec<String> = (1..=plan.n_max)
            .flat_map(|r| (0..plan.num_traces()).map(move |t| format!("r{r}_t{t}_main_desc")))
            .collect();
        got.sort();
        want.sort();
        assert_eq!(
            got, want,
            "descriptor arrays must be one per (round, trace); duplicates mean the \
             zerocheck and logup families are still each emitting their own"
        );

        // --- and one element materializer per (round, trace, main matrix)
        let materializers = g
            .nodes
            .iter()
            .filter(|n| {
                matches!(n, GraphNode::BlackboxKernel(k) if k.name == "materialize_main_matrix_ptr")
            })
            .count();
        let per_round: usize = plan
            .traces
            .iter()
            .map(|tp| tp.mats.len() - usize::from(tp.has_preprocessed))
            .sum();
        assert_eq!(
            materializers,
            plan.n_max * per_round,
            "duplicated `MainMatrixPtrs` element writes"
        );

        // --- no cross-AIR: each evaluator declares exactly its own traces'
        //     descriptor arrays.
        let zc_traces: Vec<usize> = (0..plan.num_traces())
            .filter(|&t| plan.traces[t].has_constraints)
            .collect();
        let lg_traces: Vec<usize> = (0..plan.num_traces())
            .filter(|&t| plan.traces[t].has_interactions)
            .collect();
        let mut checked = 0usize;
        for node in &g.nodes {
            let GraphNode::BlackboxKernel(k) = node else {
                continue;
            };
            if !EVAL_NODE_NAMES.contains(&k.name.as_str()) {
                continue;
            }
            // `{zc,lg}_r{round}_ctxs` — the family and round the node serves.
            let ctx_name = buf_name(&g, k.inputs[1]).to_string();
            let (family, rest) = ctx_name.split_at(3);
            let round: usize = rest
                .trim_start_matches('r')
                .trim_end_matches("_ctxs")
                .parse()
                .expect("round in ctx array name");
            let traces = match family {
                "zc_" => &zc_traces,
                "lg_" => &lg_traces,
                other => panic!("unexpected ctx array name prefix {other:?}"),
            };
            let mut want: Vec<String> = traces
                .iter()
                .map(|t| format!("r{round}_t{t}_main_desc"))
                .collect();
            let mut got: Vec<String> = k
                .inputs
                .iter()
                .map(|&b| buf_name(&g, b).to_string())
                .filter(|n| n.ends_with("_main_desc"))
                .collect();
            want.sort();
            got.sort();
            got.dedup();
            assert_eq!(
                got, want,
                "`{}` (ctx array `{ctx_name}`) declares the wrong descriptor arrays",
                k.name
            );
            checked += 1;
        }
        assert!(checked > 0, "fixture emitted no batched evaluator nodes");
    }

    /// T5: a graph-materialized ctx element must equal the eager builder's
    /// struct field for field.
    ///
    /// The oracle is `ZerocheckMleBatchBuilder::new` / `LogupMleBatchBuilder::new`
    /// (`batch_mle.rs:158-201, 297-353`): the same field map, but built from
    /// the graph's own pool addresses instead of host-allocated
    /// `DeviceBuffer`s. Comparing typed fields (pointers as `usize`) rather
    /// than `memcmp` keeps ABI padding out of the assertion; a second replay
    /// then checks the *complete* bytes are stable, which is what catches
    /// unwritten padding.
    ///
    /// Two records are covered, exactly the branches the eager builders have:
    /// preprocessed present / absent, and `buffer_size` nonzero / zero (the
    /// null-`d_intermediates` branch, `batch_mle.rs:164-176`).
    #[test]
    fn ctx_materializers_match_eager_field_by_field() {
        // The materializers write through the private C++ ctx definitions;
        // fail loudly if the Rust mirrors have drifted in size.
        crate::cuda::logup_zerocheck::assert_ctx_abi_matches_cuda();

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);

        for (case, has_prep, buffer_size) in [(0usize, true, 4u32), (1, false, 0u32)] {
            let num_x = 3usize;
            let num_y = 4usize;
            let height = num_x * num_y;
            let main_widths = [2u32, 5u32];
            let prep_width: u32 = if has_prep { 3 } else { 0 };
            let rules_len = 7usize;
            let used_nodes_len = 5usize;
            let num_interactions = 6usize;

            let mut g = GraphBuilder::new();

            // --- source buffers (every one of them becomes a graph edge)
            let selectors = add_ef_buf(&mut g, device, "sel", 3 * height);
            g.insert_memset(selectors, 0);
            let public = add_f_buf(&mut g, device, "public", 8);
            g.insert_memset(public, 0);
            let eq_xi = add_ef_buf(&mut g, device, "eq_xi", num_y);
            g.insert_memset(eq_xi, 0);
            let challenges = add_ef_buf(&mut g, device, "challenges", 4);
            g.insert_memset(challenges, 0);
            let eq_3bs = add_ef_buf(&mut g, device, "eq_3bs", num_interactions);
            g.insert_memset(eq_3bs, 0);
            let rules = add_typed_buf::<u128>(&mut g, device, "rules", rules_len);
            g.insert_memset(rules, 0);
            let used_nodes = add_typed_buf::<usize>(&mut g, device, "used_nodes", used_nodes_len);
            g.insert_memset(used_nodes, 0);
            let pair_idxs = add_typed_buf::<u32>(&mut g, device, "pair_idxs", used_nodes_len);
            g.insert_memset(pair_idxs, 0);
            let prep = has_prep.then(|| {
                let b = add_ef_buf(&mut g, device, "prep", height * prep_width as usize);
                g.insert_memset(b, 0);
                b
            });
            let mains: Vec<BufId> = main_widths
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    let b = add_ef_buf(&mut g, device, &format!("main{i}"), height * w as usize);
                    g.insert_memset(b, 0);
                    b
                })
                .collect();
            let intermediates = (buffer_size > 0).then(|| {
                let b = add_ef_buf(&mut g, device, "inter", height * buffer_size as usize);
                g.insert_memset(b, 0);
                b
            });

            // --- the `MainMatrixPtrs<EF>` descriptor array, built on device
            let main_desc =
                add_typed_buf::<MainMatrixPtrs<EF>>(&mut g, device, "main_desc", mains.len());
            g.insert_memset(main_desc, 0);
            for (i, (&b, &w)) in mains.iter().zip(main_widths.iter()).enumerate() {
                let _ = materialize_main_matrix_ptr_ir(
                    &mut g,
                    main_desc,
                    i as u32,
                    DevicePtrArg::buf(b),
                    w,
                );
            }

            let eval_ctx = EvalCoreCtxArgs {
                d_selectors: DevicePtrArg::buf(selectors),
                d_preprocessed_data: match prep {
                    Some(b) => DevicePtrArg::buf(b),
                    None => DevicePtrArg::NULL,
                },
                preprocessed_air_width: prep_width,
                d_main: DevicePtrArg::buf(main_desc),
                d_public: DevicePtrArg::buf(public),
            };
            let d_intermediates = match intermediates {
                Some(b) => DevicePtrArg::buf(b),
                None => DevicePtrArg::NULL,
            };

            // --- two-element ctx arrays: element 1 exercises `idx != 0`.
            let zc_ctxs = add_typed_buf::<ZerocheckCtx>(&mut g, device, "zc_ctxs", 2);
            g.insert_memset(zc_ctxs, 0);
            let zc_args = ZerocheckCtxArgs {
                eval_ctx,
                d_intermediates,
                num_y: num_y as u32,
                d_eq_xi: DevicePtrArg::buf(eq_xi),
                d_rules: DevicePtrArg::buf(rules),
                rules_len,
                d_used_nodes: DevicePtrArg::buf(used_nodes),
                used_nodes_len,
                buffer_size,
            };
            let _ = materialize_zerocheck_ctx_ir(&mut g, zc_ctxs, 1, zc_args);

            let lg_ctxs = add_typed_buf::<LogupCtx>(&mut g, device, "lg_ctxs", 2);
            g.insert_memset(lg_ctxs, 0);
            let lg_args = LogupCtxArgs {
                eval_ctx,
                d_intermediates,
                num_y: num_y as u32,
                d_eq_xi: DevicePtrArg::buf(eq_xi),
                d_challenges: DevicePtrArg::buf(challenges),
                d_eq_3bs: DevicePtrArg::buf(eq_3bs),
                d_rules: DevicePtrArg::buf(rules),
                rules_len,
                d_used_nodes: DevicePtrArg::buf(used_nodes),
                d_pair_idxs: DevicePtrArg::buf(pair_idxs),
                used_nodes_len,
                buffer_size,
            };
            let _ = materialize_logup_ctx_ir(&mut g, lg_ctxs, 1, lg_args);

            // --- compile + run
            let mut tracked = vec![
                selectors, public, eq_xi, challenges, eq_3bs, rules, used_nodes, pair_idxs,
                main_desc, zc_ctxs, lg_ctxs,
            ];
            tracked.extend(mains.iter().copied());
            tracked.extend(prep);
            tracked.extend(intermediates);
            for &b in &tracked {
                g.register_output(b);
            }
            let mut exe = GraphCompiler::new()
                .device(device)
                .scheduler(SchedulerMode::ListV1 {
                    params: ListSchedulerV1::default(),
                })
                .compile(g)
                .expect("graph compile");
            exe.run(&ctx).expect("graph run");
            ctx.stream.synchronize().unwrap();

            let out_idx = |exe: &crypto_compiler::graph_exe::GraphExe, b: BufId| {
                (0..exe.num_outputs())
                    .find(|&i| exe.output_buf_id(i) == b)
                    .unwrap_or_else(|| panic!("buf {b:?} not registered as an output"))
            };
            let addr = |exe: &crypto_compiler::graph_exe::GraphExe, b: BufId| {
                exe.get_output(out_idx(exe, b)).as_raw_ptr() as usize
            };
            let bytes = |exe: &crypto_compiler::graph_exe::GraphExe, b: BufId| {
                exe.get_output(out_idx(exe, b))
                    .to_host_on(&ctx)
                    .expect("D2H")
            };

            // --- MainMatrixPtrs array, element by element
            let desc_bytes = bytes(&exe, main_desc);
            for (i, (&b, &w)) in mains.iter().zip(main_widths.iter()).enumerate() {
                let got: MainMatrixPtrs<EF> = unsafe {
                    (desc_bytes.as_ptr() as *const MainMatrixPtrs<EF>)
                        .add(i)
                        .read_unaligned()
                };
                assert_eq!(
                    got.data as usize,
                    addr(&exe, b),
                    "case {case}: main_desc[{i}].data"
                );
                assert_eq!(got.air_width, w, "case {case}: main_desc[{i}].air_width");
            }

            let prep_addr = prep.map(|b| addr(&exe, b)).unwrap_or(0);
            let inter_addr = intermediates.map(|b| addr(&exe, b)).unwrap_or(0);

            // --- ZerocheckCtx, field by field (the eager field map is the
            //     oracle: `batch_mle.rs:178-196`).
            let zc_bytes = bytes(&exe, zc_ctxs);
            let got: ZerocheckCtx = unsafe {
                (zc_bytes.as_ptr() as *const ZerocheckCtx)
                    .add(1)
                    .read_unaligned()
            };
            assert_eq!(
                got.eval_ctx.d_selectors as usize,
                addr(&exe, selectors),
                "case {case}: zc.eval_ctx.d_selectors"
            );
            assert_eq!(
                got.eval_ctx.d_preprocessed.data as usize, prep_addr,
                "case {case}: zc.eval_ctx.d_preprocessed.data"
            );
            assert_eq!(
                got.eval_ctx.d_preprocessed.air_width, prep_width,
                "case {case}: zc.eval_ctx.d_preprocessed.air_width"
            );
            assert_eq!(
                got.eval_ctx.d_main as usize,
                addr(&exe, main_desc),
                "case {case}: zc.eval_ctx.d_main"
            );
            assert_eq!(
                got.eval_ctx.d_public as usize,
                addr(&exe, public),
                "case {case}: zc.eval_ctx.d_public"
            );
            assert_eq!(
                got.d_intermediates as usize, inter_addr,
                "case {case}: zc.d_intermediates"
            );
            assert_eq!(got.num_y, num_y as u32, "case {case}: zc.num_y");
            assert_eq!(
                got.d_eq_xi as usize,
                addr(&exe, eq_xi),
                "case {case}: zc.d_eq_xi"
            );
            assert_eq!(
                got.d_rules as usize,
                addr(&exe, rules),
                "case {case}: zc.d_rules"
            );
            assert_eq!(got.rules_len, rules_len, "case {case}: zc.rules_len");
            assert_eq!(
                got.d_used_nodes as usize,
                addr(&exe, used_nodes),
                "case {case}: zc.d_used_nodes"
            );
            assert_eq!(
                got.used_nodes_len, used_nodes_len,
                "case {case}: zc.used_nodes_len"
            );
            assert_eq!(got.buffer_size, buffer_size, "case {case}: zc.buffer_size");

            // Element 0 was only memset: proves `idx` is honoured.
            let zc0: ZerocheckCtx =
                unsafe { (zc_bytes.as_ptr() as *const ZerocheckCtx).read_unaligned() };
            assert_eq!(
                zc0.eval_ctx.d_selectors as usize, 0,
                "case {case}: zc_ctxs[0] must be untouched"
            );

            // --- LogupCtx, field by field (`batch_mle.rs:317-347`).
            let lg_bytes = bytes(&exe, lg_ctxs);
            let got: LogupCtx = unsafe {
                (lg_bytes.as_ptr() as *const LogupCtx)
                    .add(1)
                    .read_unaligned()
            };
            assert_eq!(
                got.eval_ctx.d_selectors as usize,
                addr(&exe, selectors),
                "case {case}: lg.eval_ctx.d_selectors"
            );
            assert_eq!(
                got.eval_ctx.d_preprocessed.data as usize, prep_addr,
                "case {case}: lg.eval_ctx.d_preprocessed.data"
            );
            assert_eq!(
                got.eval_ctx.d_preprocessed.air_width, prep_width,
                "case {case}: lg.eval_ctx.d_preprocessed.air_width"
            );
            assert_eq!(
                got.eval_ctx.d_main as usize,
                addr(&exe, main_desc),
                "case {case}: lg.eval_ctx.d_main"
            );
            assert_eq!(
                got.eval_ctx.d_public as usize,
                addr(&exe, public),
                "case {case}: lg.eval_ctx.d_public"
            );
            assert_eq!(
                got.d_intermediates as usize, inter_addr,
                "case {case}: lg.d_intermediates"
            );
            assert_eq!(got.num_y, num_y as u32, "case {case}: lg.num_y");
            assert_eq!(
                got.d_eq_xi as usize,
                addr(&exe, eq_xi),
                "case {case}: lg.d_eq_xi"
            );
            assert_eq!(
                got.d_challenges as usize,
                addr(&exe, challenges),
                "case {case}: lg.d_challenges"
            );
            assert_eq!(
                got.d_eq_3bs as usize,
                addr(&exe, eq_3bs),
                "case {case}: lg.d_eq_3bs"
            );
            assert_eq!(
                got.d_rules as usize,
                addr(&exe, rules),
                "case {case}: lg.d_rules"
            );
            assert_eq!(got.rules_len, rules_len, "case {case}: lg.rules_len");
            assert_eq!(
                got.d_used_nodes as usize,
                addr(&exe, used_nodes),
                "case {case}: lg.d_used_nodes"
            );
            assert_eq!(
                got.d_pair_idxs as usize,
                addr(&exe, pair_idxs),
                "case {case}: lg.d_pair_idxs"
            );
            assert_eq!(
                got.used_nodes_len, used_nodes_len,
                "case {case}: lg.used_nodes_len"
            );
            assert_eq!(got.buffer_size, buffer_size, "case {case}: lg.buffer_size");

            // --- replay: the same compiled graph, no rebinding. Pool
            //     addresses are stable, so the *complete* bytes — padding
            //     included — must be identical.
            exe.run(&ctx).expect("graph replay");
            ctx.stream.synchronize().unwrap();
            assert_eq!(
                bytes(&exe, zc_ctxs),
                zc_bytes,
                "case {case}: ZerocheckCtx bytes not stable across replay"
            );
            assert_eq!(
                bytes(&exe, lg_ctxs),
                lg_bytes,
                "case {case}: LogupCtx bytes not stable across replay"
            );
            assert_eq!(
                bytes(&exe, main_desc),
                desc_bytes,
                "case {case}: MainMatrixPtrs bytes not stable across replay"
            );
        }
    }

    /// R4 step 2: the materialized pointers must still address live,
    /// correct bytes when the pool packer is *allowed* to reuse slots.
    ///
    /// `ctx_materializers_match_eager_field_by_field` above cannot check
    /// this. It registers every pointed-to buffer as a graph output, and
    /// `GraphCompiler` pins inputs ∪ outputs (`graph_exe.rs:634-644`), which
    /// forces `death[b] = n` for all of them (`planner/heuristic.rs:184-185`)
    /// — no slot can ever be reused, which is precisely the configuration in
    /// which an under-declared `ctx_reads` is harmless. It proves the field
    /// map; it proves nothing about liveness.
    ///
    /// Here only the three ctx arrays are pinned, every pointed-to buffer is
    /// left packable, and the packer is given a reason to reuse: a `dummy`
    /// buffer larger than the whole rest of the pool, alive across the
    /// evaluator. Because `to_place` is sorted largest-first
    /// (`heuristic.rs:190`), `dummy` is placed at offset 0, and any buffer
    /// whose lifetime does *not* overlap it lands inside its range.
    ///
    /// The node order is a deterministic sandwich, forced by data edges:
    ///
    /// ```text
    ///   ctx materializers -> dummy_gate (fills dummy 0x5c) -> eval -> dummy_sink
    /// ```
    ///
    /// `dummy_gate` reads the ctx arrays (so it cannot precede the
    /// materializers) and writes `gate`, which the evaluator declares (so it
    /// cannot follow the evaluator). It is `dummy`'s sole writer — the graph
    /// IR is SSA, and `verify_graph` rejects a second one — so it both opens
    /// `dummy`'s lifetime and issues the clobbering fill. Therefore:
    ///
    /// * **declared correctly** — `selectors` &c. are read by the evaluator, so they die after
    ///   `dummy` is born, their lifetimes overlap, and the packer must give them disjoint slots.
    ///   The bytes survive.
    /// * **under-declared** — they die at the ctx materializer, *before* `dummy` is born, so the
    ///   packer puts them inside `dummy` and the memset overwrites the bytes the ctx pointers
    ///   address.
    ///
    /// Red-green: drop one buffer from `ctx_reads` below and this test
    /// fails on that buffer's byte pattern. Demonstrated on R4: removing
    /// `selectors` puts it at pool offset 0, inside `dummy`, and the probe
    /// reads `0x5c` instead of `0x11`.
    // TODO(cc-ir,R4): the teeth depend on planner behaviour this test does
    //   not assert — largest-first placement (`heuristic.rs:190`) putting
    //   `dummy` at offset 0, and the sandwich order actually being chosen.
    // WHY: neither is observable from `GraphExe`; the pool base and the
    //   per-buffer offsets of unpinned buffers are private, and pinning them
    //   to read them back is exactly what destroys the property under test.
    // RISK: a future scheduler or packer change could make this test pass
    //   *vacuously* — green, but no longer able to fail. It would not go
    //   red, so nothing would flag it. Re-arm the commented `ctx_reads`
    //   line below after any planner change and confirm it still fails.
    #[test]
    fn ctx_pointers_survive_pool_reuse() {
        crate::cuda::logup_zerocheck::assert_ctx_abi_matches_cuda();

        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let ef = size_of::<EF>();
        let f = size_of::<F>();

        let num_y = 4usize;
        let height = 3 * num_y;
        let main_widths = [2u32, 5u32];
        let prep_width = 3u32;
        let rules_len = 7usize;
        let used_nodes_len = 5usize;
        let num_interactions = 6usize;
        let buffer_size = 4u32;

        let mut g = GraphBuilder::new();

        // Every source buffer carries a distinct byte pattern, so a byte
        // read back through a ctx pointer identifies which buffer's storage
        // the pointer actually landed on.
        let mut seeded: Vec<(&str, BufId, usize, u8)> = Vec::new();
        let seed = |g: &mut GraphBuilder,
                    seeded: &mut Vec<(&'static str, BufId, usize, u8)>,
                    name: &'static str,
                    b: BufId,
                    bytes: usize,
                    pat: u8| {
            // The graph's memset only accepts byte-uniform fills.
            g.insert_memset(b, u32::from_ne_bytes([pat; 4]));
            seeded.push((name, b, bytes, pat));
        };

        let selectors = add_ef_buf(&mut g, device, "sel", 3 * height);
        seed(
            &mut g,
            &mut seeded,
            "selectors",
            selectors,
            3 * height * ef,
            0x11,
        );
        let public = add_f_buf(&mut g, device, "public", 8);
        seed(&mut g, &mut seeded, "public", public, 8 * f, 0x22);
        let eq_xi = add_ef_buf(&mut g, device, "eq_xi", num_y);
        seed(&mut g, &mut seeded, "eq_xi", eq_xi, num_y * ef, 0x33);
        let challenges = add_ef_buf(&mut g, device, "challenges", 4);
        seed(&mut g, &mut seeded, "challenges", challenges, 4 * ef, 0x44);
        let eq_3bs = add_ef_buf(&mut g, device, "eq_3bs", num_interactions);
        seed(
            &mut g,
            &mut seeded,
            "eq_3bs",
            eq_3bs,
            num_interactions * ef,
            0x55,
        );
        let rules = add_typed_buf::<u128>(&mut g, device, "rules", rules_len);
        seed(&mut g, &mut seeded, "rules", rules, rules_len * 16, 0x66);
        let used_nodes = add_typed_buf::<usize>(&mut g, device, "used_nodes", used_nodes_len);
        seed(
            &mut g,
            &mut seeded,
            "used_nodes",
            used_nodes,
            used_nodes_len * 8,
            0x77,
        );
        let pair_idxs = add_typed_buf::<u32>(&mut g, device, "pair_idxs", used_nodes_len);
        seed(
            &mut g,
            &mut seeded,
            "pair_idxs",
            pair_idxs,
            used_nodes_len * 4,
            0x78,
        );
        let prep = add_ef_buf(&mut g, device, "prep", height * prep_width as usize);
        seed(
            &mut g,
            &mut seeded,
            "prep",
            prep,
            height * prep_width as usize * ef,
            0x99,
        );
        let mains: Vec<BufId> = main_widths
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let b = add_ef_buf(&mut g, device, &format!("main{i}"), height * w as usize);
                let name = if i == 0 { "main0" } else { "main1" };
                let pat = if i == 0 { 0xAAu8 } else { 0xBBu8 };
                seed(&mut g, &mut seeded, name, b, height * w as usize * ef, pat);
                b
            })
            .collect();
        let inter = add_ef_buf(&mut g, device, "inter", height * buffer_size as usize);
        seed(
            &mut g,
            &mut seeded,
            "inter",
            inter,
            height * buffer_size as usize * ef,
            0xCC,
        );

        // The declared read set, derived from the materializers themselves
        // (R4 step 1) — this vector is the thing under test.
        let mut ctx_reads: Vec<BufId> = Vec::new();

        let main_desc =
            add_typed_buf::<MainMatrixPtrs<EF>>(&mut g, device, "main_desc", mains.len());
        g.insert_memset(main_desc, 0);
        for (i, (&b, &w)) in mains.iter().zip(main_widths.iter()).enumerate() {
            extend_reads(
                &mut ctx_reads,
                materialize_main_matrix_ptr_ir(
                    &mut g,
                    main_desc,
                    i as u32,
                    DevicePtrArg::buf(b),
                    w,
                ),
            );
        }

        let eval_ctx = EvalCoreCtxArgs {
            d_selectors: DevicePtrArg::buf(selectors),
            d_preprocessed_data: DevicePtrArg::buf(prep),
            preprocessed_air_width: prep_width,
            d_main: DevicePtrArg::buf(main_desc),
            d_public: DevicePtrArg::buf(public),
        };

        let zc_ctxs = add_typed_buf::<ZerocheckCtx>(&mut g, device, "zc_ctxs", 1);
        g.insert_memset(zc_ctxs, 0);
        extend_reads(
            &mut ctx_reads,
            materialize_zerocheck_ctx_ir(
                &mut g,
                zc_ctxs,
                0,
                ZerocheckCtxArgs {
                    eval_ctx,
                    d_intermediates: DevicePtrArg::buf(inter),
                    num_y: num_y as u32,
                    d_eq_xi: DevicePtrArg::buf(eq_xi),
                    d_rules: DevicePtrArg::buf(rules),
                    rules_len,
                    d_used_nodes: DevicePtrArg::buf(used_nodes),
                    used_nodes_len,
                    buffer_size,
                },
            ),
        );

        let lg_ctxs = add_typed_buf::<LogupCtx>(&mut g, device, "lg_ctxs", 1);
        g.insert_memset(lg_ctxs, 0);
        extend_reads(
            &mut ctx_reads,
            materialize_logup_ctx_ir(
                &mut g,
                lg_ctxs,
                0,
                LogupCtxArgs {
                    eval_ctx,
                    d_intermediates: DevicePtrArg::buf(inter),
                    num_y: num_y as u32,
                    d_eq_xi: DevicePtrArg::buf(eq_xi),
                    d_challenges: DevicePtrArg::buf(challenges),
                    d_eq_3bs: DevicePtrArg::buf(eq_3bs),
                    d_rules: DevicePtrArg::buf(rules),
                    rules_len,
                    d_used_nodes: DevicePtrArg::buf(used_nodes),
                    d_pair_idxs: DevicePtrArg::buf(pair_idxs),
                    used_nodes_len,
                    buffer_size,
                },
            ),
        );

        // --- RED-GREEN HANDLE: uncomment to under-declare one buffer and
        //     watch the byte assertion below fail on `selectors`.
        // ctx_reads.retain(|&b| b != selectors);

        // --- the packer's reason to reuse: `dummy` dwarfs the rest of the
        //     pool and stays alive across the evaluator.
        let dummy = add_f_buf(&mut g, device, "dummy", 1 << 18);
        let gate = add_f_buf(&mut g, device, "gate", 1);
        // Ordered after the ctx materializers (it reads their outputs) and
        // before the evaluator (which reads `gate`). It is `dummy`'s only
        // writer — the graph IR is SSA, one writer per buffer — so
        // `birth[dummy]` lands in that window, and the fill it issues is the
        // clobber: whatever shares `dummy`'s slot loses its bytes here.
        let dummy_bytes = (1usize << 18) * f;
        g.insert_blackbox_kernel(
            "dummy_gate",
            [zc_ctxs, lg_ctxs].into_iter(),
            [dummy, gate].into_iter(),
            [false, false].into_iter(),
            move |_, outputs, stream| unsafe {
                cudaMemsetAsync(outputs[0] as *mut _, 0x5c, dummy_bytes, stream);
                cudaMemsetAsync(outputs[1] as *mut _, 0, f, stream);
            },
        );

        // --- the evaluator stand-in. It declares exactly what the real
        //     batched evaluators declare — `eval_node_bindings` over the
        //     derived read set — and launches nothing, because what is under
        //     test is the *declaration*, not the kernel.
        let eval_out = add_ef_buf(&mut g, device, "eval_out", 4);
        let (inputs, modifies) =
            eval_node_bindings(&[zc_ctxs, lg_ctxs, gate], &[inter], &ctx_reads);
        g.insert_blackbox_kernel(
            "eval_stand_in",
            inputs.into_iter(),
            [eval_out].into_iter(),
            modifies.into_iter(),
            |_, _, _| {},
        );
        // Keeps `dummy` alive past the evaluator (it reads `eval_out`).
        let sink = add_f_buf(&mut g, device, "sink", 1);
        g.insert_blackbox_kernel(
            "dummy_sink",
            [dummy, eval_out].into_iter(),
            [sink].into_iter(),
            [false, false].into_iter(),
            |_, _, _| {},
        );

        // Only the ctx arrays are pinned (plus `sink`, pure scaffolding —
        // it keeps `dummy_sink` out of DCE's reach, and it is not a buffer
        // any ctx pointer addresses). Everything the pointers *do* address
        // is left packable.
        for b in [zc_ctxs, lg_ctxs, main_desc, sink] {
            g.register_output(b);
        }
        let mut exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("graph compile");
        exe.run(&ctx).expect("graph run");
        ctx.stream.synchronize().unwrap();

        let out_idx = |b: BufId| {
            (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == b)
                .unwrap_or_else(|| panic!("buf {b:?} not registered as an output"))
        };
        let bytes = |b: BufId| exe.get_output(out_idx(b)).to_host_on(&ctx).expect("D2H");
        // Read `n` bytes straight from a device address the graph produced.
        let peek = |addr: usize, n: usize| -> Vec<u8> {
            assert_ne!(addr, 0, "null ctx pointer");
            let view = unsafe { DeviceBuffer::<u8>::from_raw_parts(addr as *mut u8, n) };
            let host = view.to_host_on(&ctx).expect("D2H");
            forget(view);
            host
        };

        let zc_bytes = bytes(zc_ctxs);
        let lg_bytes = bytes(lg_ctxs);
        let desc_bytes = bytes(main_desc);
        let zc: ZerocheckCtx =
            unsafe { (zc_bytes.as_ptr() as *const ZerocheckCtx).read_unaligned() };
        let lg: LogupCtx = unsafe { (lg_bytes.as_ptr() as *const LogupCtx).read_unaligned() };

        // Every pointer field, paired with the buffer it must address.
        let mut probes: Vec<(&str, usize)> = vec![
            ("selectors", zc.eval_ctx.d_selectors as usize),
            ("selectors", lg.eval_ctx.d_selectors as usize),
            ("prep", zc.eval_ctx.d_preprocessed.data as usize),
            ("prep", lg.eval_ctx.d_preprocessed.data as usize),
            ("public", zc.eval_ctx.d_public as usize),
            ("public", lg.eval_ctx.d_public as usize),
            ("inter", zc.d_intermediates as usize),
            ("inter", lg.d_intermediates as usize),
            ("eq_xi", zc.d_eq_xi as usize),
            ("eq_xi", lg.d_eq_xi as usize),
            ("rules", zc.d_rules as usize),
            ("rules", lg.d_rules as usize),
            ("used_nodes", zc.d_used_nodes as usize),
            ("used_nodes", lg.d_used_nodes as usize),
            ("challenges", lg.d_challenges as usize),
            ("eq_3bs", lg.d_eq_3bs as usize),
            ("pair_idxs", lg.d_pair_idxs as usize),
        ];
        // …reached through the second pointer level, too.
        assert_eq!(zc.eval_ctx.d_main as usize, lg.eval_ctx.d_main as usize);
        for i in 0..mains.len() {
            let got: MainMatrixPtrs<EF> = unsafe {
                (desc_bytes.as_ptr() as *const MainMatrixPtrs<EF>)
                    .add(i)
                    .read_unaligned()
            };
            probes.push((if i == 0 { "main0" } else { "main1" }, got.data as usize));
        }

        for (label, addr) in probes {
            let &(_, _, n, pat) = seeded
                .iter()
                .find(|(name, ..)| *name == label)
                .expect("seeded buffer");
            let got = peek(addr, n);
            assert!(
                got.iter().all(|&b| b == pat),
                "`{label}` at {addr:#x} does not hold its own bytes any more \
                 (expected all {pat:#04x}, first mismatch at index {}, value {:#04x}). \
                 The pool packer reused its slot, which means the evaluator's \
                 declared `ctx_reads` is missing it.",
                got.iter().position(|&b| b != pat).unwrap(),
                got.iter().find(|&&b| b != pat).unwrap(),
            );
        }
    }

    /// `fold_ple_from_evals`: graph node vs eager launcher, raw device bytes.
    ///
    /// This is the phase's most self-contained kernel (one launch, no ctx
    /// structs, no challenge scalars beyond the pre-baked
    /// `inv_lagrange_denoms`), so it is the port's first oracle.
    #[test]
    fn fold_ple_from_evals_ir_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x20C0_1DEF);

        for (l_skip, log_height, width) in [(2usize, 5usize, 3usize), (3, 7, 4), (1, 4, 2)] {
            let skip_domain = 1usize << l_skip;
            let height = 1usize << log_height;
            let num_x = height / skip_domain;

            let mat: Vec<F> = (0..height * width).map(|_| rng.random::<F>()).collect();
            let omega: Vec<F> = (0..skip_domain).map(|_| rng.random::<F>()).collect();
            let denoms: Vec<EF> = (0..skip_domain).map(|_| rng.random::<EF>()).collect();

            // --- eager reference
            let d_mat: DeviceBuffer<F> = mat.as_slice().to_device_on(&ctx).unwrap();
            let d_omega: DeviceBuffer<F> = omega.as_slice().to_device_on(&ctx).unwrap();
            let d_denoms: DeviceBuffer<EF> = denoms.as_slice().to_device_on(&ctx).unwrap();
            let d_out: DeviceBuffer<EF> = DeviceBuffer::with_capacity_on(num_x * width, &ctx);
            unsafe {
                fold_ple_from_evals(
                    &d_mat,
                    d_out.as_mut_ptr(),
                    &d_omega,
                    &d_denoms,
                    height as u32,
                    width as u32,
                    l_skip as u32,
                    num_x as u32,
                    false,
                    stream,
                )
                .expect("fold_ple_from_evals");
            }
            ctx.stream.synchronize().unwrap();
            let want: Vec<EF> = d_out.to_host_on(&ctx).unwrap();

            // --- graph side
            let mut g = GraphBuilder::new();
            let mat_buf = f_slice_const_buf(&mut g, device, "mat", &mat);
            let omega_buf = f_slice_const_buf(&mut g, device, "omega", &omega);
            let denom_buf = ef_slice_const_buf(&mut g, device, "denoms", &denoms);
            let out_buf = add_ef_buf(&mut g, device, "out", num_x * width);
            fold_ple_from_evals_ir(
                &mut g,
                mat_buf,
                height * width,
                None,
                out_buf,
                0,
                omega_buf,
                skip_domain,
                denom_buf,
                height as u32,
                width as u32,
                l_skip as u32,
                num_x as u32,
                false,
            );
            let got = run_graph_read_bufs(g, &[out_buf], &ctx).remove(0);
            assert_eq!(
                got.len(),
                num_x * width * EF_BYTES,
                "output byte length mismatch (l_skip={l_skip}, log_height={log_height})"
            );
            assert_eq!(
                &got[..],
                ef_bytes(&want),
                "fold_ple_from_evals_ir mismatch (l_skip={l_skip}, log_height={log_height}, width={width})"
            );
        }
    }

    /// `fold_selectors_round0`: graph node vs eager launcher, raw device
    /// bytes. Covers the challenge-by-value path (`is_first` / `is_last`).
    #[test]
    fn fold_selectors_round0_ir_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0x5E1E_C705);

        for log_num_x in [2usize, 4, 6] {
            let num_x = 1usize << log_num_x;
            let cube: Vec<F> = (0..3 * num_x).map(|_| rng.random::<F>()).collect();
            let is_first: EF = rng.random();
            let is_last: EF = rng.random();

            // --- eager reference
            let d_in: DeviceBuffer<F> = cube.as_slice().to_device_on(&ctx).unwrap();
            let d_out: DeviceBuffer<EF> = DeviceBuffer::with_capacity_on(3 * num_x, &ctx);
            unsafe {
                fold_selectors_round0(
                    d_out.as_mut_ptr(),
                    d_in.as_ptr(),
                    is_first,
                    is_last,
                    num_x,
                    stream,
                )
                .expect("fold_selectors_round0");
            }
            ctx.stream.synchronize().unwrap();
            let want: Vec<EF> = d_out.to_host_on(&ctx).unwrap();

            // --- graph side
            let mut g = GraphBuilder::new();
            let in_buf = f_slice_const_buf(&mut g, device, "sels_cube", &cube);
            let out_buf = add_ef_buf(&mut g, device, "sels_folded", 3 * num_x);
            fold_selectors_round0_ir(&mut g, out_buf, in_buf, is_first, is_last, num_x);
            let got = run_graph_read_bufs(g, &[out_buf], &ctx).remove(0);
            assert_eq!(
                &got[..],
                ef_bytes(&want),
                "fold_selectors_round0_ir mismatch at num_x={num_x}"
            );
        }
    }

    /// `eq_hypercube_interleaved_stage_ext`: graph node vs eager launcher.
    #[test]
    fn eq_hypercube_interleaved_stage_ir_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);
        let mut rng = StdRng::seed_from_u64(0xE9_C0BE);

        for log_step in [0usize, 3, 7] {
            let step = 1usize << log_step;
            let input: Vec<EF> = (0..step).map(|_| rng.random::<EF>()).collect();
            let x_i: EF = rng.random();

            let d_in: DeviceBuffer<EF> = input.as_slice().to_device_on(&ctx).unwrap();
            let d_out: DeviceBuffer<EF> = DeviceBuffer::with_capacity_on(2 * step, &ctx);
            unsafe {
                crate::cuda::poly::eq_hypercube_interleaved_stage_ext(
                    d_out.as_mut_ptr(),
                    d_in.as_ptr(),
                    x_i,
                    step as u32,
                    stream,
                )
                .expect("eq_hypercube_interleaved_stage_ext");
            }
            ctx.stream.synchronize().unwrap();
            let want: Vec<EF> = d_out.to_host_on(&ctx).unwrap();

            let mut g = GraphBuilder::new();
            let in_buf = ef_slice_const_buf(&mut g, device, "eq_in", &input);
            let out_buf = add_ef_buf(&mut g, device, "eq_out", 2 * step);
            eq_hypercube_interleaved_stage_ext_ir(&mut g, in_buf, out_buf, x_i, step as u32);
            let got = run_graph_read_bufs(g, &[out_buf], &ctx).remove(0);
            assert_eq!(
                &got[..],
                ef_bytes(&want),
                "eq_hypercube_interleaved_stage_ir mismatch at step={step}"
            );
        }
    }

    /// The acceptance bar: the whole phase builds as a graph and the graph
    /// compiles to a `GraphExe`.
    ///
    /// This does **not** run the graph — its inputs are zeroed
    /// (`TraceBufs::alloc_zeroed`), so the outputs would be meaningless. It
    /// asserts the thing the port is for: `logup_zerocheck_gpu_ir` emits a
    /// well-formed graph for a realistic phase shape and `GraphCompiler`
    /// accepts it.
    #[test]
    fn logup_zerocheck_phase_graph_compiles() {
        let device = DeviceType::Cuda(0);
        let plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 2, /* n_max */ 4,
        );

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_zeroed(&mut g, device, &plan, t))
            .collect();
        let proof = logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device);

        assert_eq!(proof.round0_zc_evals.len(), plan.num_traces());
        assert_eq!(proof.round0_logup_evals.len(), plan.num_traces());
        assert_eq!(proof.round_evals.len(), plan.n_max);
        assert_eq!(proof.column_openings.len(), plan.num_traces());

        let exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("phase graph compile");
        assert!(exe.num_outputs() > 0, "phase graph produced no outputs");
    }

    /// The phase graph must chain onto a live transcript, not restart it.
    #[test]
    fn logup_zerocheck_phase_graph_seeds_from_live_sponge() {
        use openvm_stark_backend::FiatShamirTranscript;

        use crate::sponge::DuplexSpongeGpu;

        let device = DeviceType::Cuda(0);
        let plan = synthetic_plan(2, 1, 2);

        // A live sponge that has already absorbed (stages A/B).
        let mut live = DuplexSpongeGpu::default();
        for i in 0..5u32 {
            live.observe(F::from_u32(i));
        }
        let snap = live.snapshot();

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::from_live(&mut g, device, &snap);
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_zeroed(&mut g, device, &plan, t))
            .collect();
        let _ = logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device);
        GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("seeded phase graph compile");
    }
}
