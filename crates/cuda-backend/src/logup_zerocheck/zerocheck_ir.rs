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
//! The eager phase is not a pure kernel DAG: host computations sit *on the
//! data path* between kernels. They are the honest seams of this port, and
//! every one that remains is marked with a `TODO(cc-ir)` at its site:
//!
//! 1. **Round 0's iDFT chain** (`mod.rs:270-352, 858-875, 918-946`) — there is no device `EF` iDFT,
//!    so `UnivariatePoly::from_geometric_cosets_evals_idft` and the `s_0_poly` assembly stay on the
//!    host.
//! 2. **`compute_batch_s_poly`** (`mod.rs:1448-1520`) — **CLOSED (P1)**. Every steady MLE round now
//!    computes its batched sumcheck polynomial on device from graph-resident evaluator outputs
//!    (`observe_and_update_zerocheck_round_ir` → `batch_s_ring_pre` / `batch_s_ring_post`),
//!    observes `s(1)..s(s_deg)` from device buffers, samples `r_round` into a device buffer, and
//!    hands that same buffer to both folds. `{tilde[3T], prev_s_eval, eq_n, eq_sharp_n}` update on
//!    device.
//! 3. **Fiat–Shamir values.** Because of (1) the round-0 challenge *values* (`lambda`, `mu`, `r_0`)
//!    are still known on the host at graph-build time and are captured by value into the round-0
//!    kernel closures, exactly like `fractional_ir.rs`'s plain (non-`_bufid`) wrappers. `r_1..r_n`
//!    are **not**: after P1 they exist only as device buffers. `xi`, `lambda_pows` and `mu_pows`
//!    remain const producers because stages A/B (grinding, GKR) are out of this module's scope, so
//!    do not read this graph as device-input-only. The remaining challenge-by-value entry points
//!    are listed in B3 §4.
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
//! `ZerocheckCtx` and `LogupCtx` (and the `MainMatrixDesc` array they point
//! at) instead hold **no pointers at all**: every device-pointer field is a
//! [`BaseOff`] byte offset into the `GraphExe`'s unified pool, and the
//! launcher takes the pool base as a kernel argument. The offsets come
//! straight from `GraphExe::plan().offsets`, so the whole array is
//! host-computable after `compile()` and is uploaded once as a registered
//! graph input by [`DescriptorPlan::bind`]. See the "base+offset descriptor
//! ABI (R6)" section below.
//!
//! The same mechanism now covers three more tables that used to be
//! `insert_memset(_, 0)` — i.e. handed to a kernel as nulls and zeros:
//! the round-0 main-matrix array (a `MainMatrixDesc` array since S1.2a),
//! `batch_fold_mle`'s `input_matrices` / `output_matrices`, and the stage-D
//! `interpolate_columns` column table. The latter two are bare `T*` tables in
//! CUDA rather than `BaseOff` structs, so they use
//! [`DescriptorPlan::set_ptr`], which encodes the absolute
//! `pool_base + offset` at bind time through the same writer that derives the
//! read set.
//!
//! Still host-assembled, and therefore still opaque: the three *monomial* ctx
//! structs (`MonomialAirCtx`, `LogupMonomialCommonCtx`, `LogupMonomialCtx`).
//!
//! # Graph inputs (S1.1)
//!
//! Every keygen- and challenge-derived buffer the phase reads is a
//! **registered graph input** recorded in a [`PhaseInputBinder`], not a
//! zeroed buffer. Only the selector cube is filled at build time (it is a
//! pure function of the plan); the rest must be supplied before `run`, and a
//! missing one is a hard error naming the buffer instead of a silent
//! all-zero prove.

use std::{
    mem::{forget, offset_of, size_of},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use crypto_compiler::{
    graph_ir::{BufId, BufInfo, ConstBuf, DeviceType, GraphBuilder},
    quast::Quast,
};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::{
    poly_common::{eval_eq_sharp_uni, eval_eq_uni, horner_eval},
    prover::fractional_sumcheck_gkr::Frac,
};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};

use super::{
    batch_mle_monomial::{DEFAULT_MAX_MONOMIALS_PER_THREAD, THREADS_PER_BLOCK_PAR_Y, WAVES_TARGET},
    fractional_ir_utils::{add_ef_buf, add_ext_scalar_buf, ef_const_ext_scalar_buf},
};
use crate::{
    cuda::{
        logup_zerocheck::{
            _logup_batch_mle_intermediates_buffer_size,
            _zerocheck_batch_mle_intermediates_buffer_size, batch_s_ring_post, batch_s_ring_pre,
            fold_ple_from_evals, fold_selectors_round0, interpolate_columns_gpu,
            logup_bary_eval_interactions_round0, logup_batch_eval_mle, logup_monomial_batched,
            precompute_lambda_combinations, precompute_logup_denom_combinations,
            precompute_logup_numer_combinations, zerocheck_batch_eval_mle,
            zerocheck_monomial_batched, zerocheck_monomial_par_y_batched,
            zerocheck_ntt_eval_constraints, BaseOff, BatchSRingTraceDesc, BlockCtx, EvalCoreCtx,
            LogupCtx, LogupMonomialCommonCtx, LogupMonomialCtx, MainMatrixDesc, MonomialAirCtx,
            ZerocheckCtx, BATCH_S_RING_HAS_CONSTRAINTS, BATCH_S_RING_HAS_INTERACTIONS,
        },
        poly::eq_hypercube_interleaved_stage_ext,
        sumcheck::batch_fold_mle_dev_challenge,
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

/// Stage a slice of POD values as a read-only const buffer.
///
/// Used for the *shape* halves of the `batch_fold_mle` control tables
/// (`widths`, `log_output_heights`): those are pure functions of the plan, so
/// unlike the pointer halves they need no bind-time encoding at all.
///
/// # Safety-relevant contract
///
/// `T` must be `Copy` and free of padding-sensitive invariants — the bytes are
/// reinterpreted verbatim and uploaded. Every call site here uses `u32` / `u8`.
pub fn typed_slice_const_buf<T: Copy>(
    g: &mut GraphBuilder,
    device: DeviceType,
    name: &str,
    xs: &[T],
) -> BufId {
    let buf = add_typed_buf::<T>(g, device, name, xs.len());
    let mut bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)).to_vec()
    };
    // `add_typed_buf` rounds an empty table up to one element; the const's
    // byte length must match the buffer's or the runtime rejects the stage.
    bytes.resize(xs.len().max(1) * size_of::<T>(), 0);
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
#[derive(Clone, Debug)]
pub struct Round0ZcBufs {
    pub tmp_sums: BufId,
    pub out: BufId,
    pub intermediates: BufId,
    pub selectors_cube: BufId,
    /// `*const F` into the preprocessed matrix; `None` for AIRs without one.
    pub preprocessed: Option<BufId>,
    /// `MainMatrixDesc` descriptor array over the main matrices, on the same
    /// base+offset ABI stage D uses (S1.2a).
    pub main_ptrs: BufId,
    /// The pool base the kernel decodes every [`BaseOff`] against.
    pub pool_base: PoolBase,
    pub eq_cube: BufId,
    pub lambda_pows: BufId,
    pub public_values: BufId,
    pub rules: BufId,
    pub used_nodes: BufId,
    /// The matrices [`Self::main_ptrs`] points at. The planner cannot see
    /// through a descriptor array, so they are declared here — the list comes
    /// from the writer itself (`DescriptorPlan::set_main_matrix_desc`), not
    /// from retyping it at the call site.
    pub main_reads: Vec<BufId>,
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
/// `main_ptrs` is a `MainMatrixDesc` descriptor array on the base+offset ABI
/// (S1.2a), filled by [`DescriptorPlan::set_main_matrix_desc`] — the same
/// mechanism stage D uses. The planner still cannot see *through* the table,
/// so the caller declares the pointed-to matrices as explicit extra inputs;
/// [`DescriptorPlan::referenced_bufs`] is what derives that list.
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
    // Appended LAST and deduplicated: the closure addresses `preprocessed` by
    // the fixed index 7, so nothing may be inserted ahead of it.
    extend_reads(&mut inputs, bufs.main_reads.iter().copied());
    // `intermediates` is pure scratch: the kernel writes it and reads back
    // only its own writes, so no value flows *in*. It is therefore declared
    // as a node **output**, not a carried input — same treatment the
    // fractional mirror gives `tmp_block_sums` (`fractional_ir.rs:399, 436`).
    // Declaring it carried would make it a read of an unproduced buffer,
    // which the fusion pass rejects with `TakeGraphError::ReadBeforeWrite`.
    let modifies: Vec<bool> = vec![false; inputs.len()];
    let pool_base = bufs.pool_base.clone();
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
                inputs[1] as *const MainMatrixDesc,
                pool_base.get(),
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
#[derive(Clone, Debug)]
pub struct Round0LogupBufs {
    pub tmp_sums: BufId,
    pub out: BufId,
    pub intermediates: BufId,
    pub selectors_cube: BufId,
    pub preprocessed: Option<BufId>,
    /// `MainMatrixDesc` descriptor array; see [`Round0ZcBufs::main_ptrs`].
    pub main_ptrs: BufId,
    /// The pool base the kernel decodes every [`BaseOff`] against.
    pub pool_base: PoolBase,
    pub eq_cube: BufId,
    pub public_values: BufId,
    pub numer_weights: BufId,
    pub denom_weights: BufId,
    pub rules: BufId,
    /// See [`Round0ZcBufs::main_reads`].
    pub main_reads: Vec<BufId>,
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
//   `alpha_logup`/`beta_logup` are host values in the eager path too.
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
    // Appended LAST and deduplicated: the closure addresses `preprocessed` by
    // the fixed index 7, so nothing may be inserted ahead of it.
    extend_reads(&mut inputs, bufs.main_reads.iter().copied());
    // `intermediates` is a node output, not a carried input — see the note
    // in [`zerocheck_ntt_eval_constraints_ir`].
    let modifies: Vec<bool> = vec![false; inputs.len()];
    let pool_base = bufs.pool_base.clone();
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
                inputs[1] as *const MainMatrixDesc,
                pool_base.get(),
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
            forget(public);
            forget(numer);
            forget(denom);
            forget(rules);
            forget(intermediates);
        },
    );
}

/// Where a [`fold_ple_from_evals_ir`] node writes.
///
/// Round 0 folds a `need_rot` matrix with **two** launches into **one**
/// doubled-width buffer: the plain fold writes `[0, num_x*width)`, the
/// rotated fold writes `[num_x*width, 2*num_x*width)` (`fold_ple.rs:38, 50`).
///
/// The graph is SSA, so the obvious spelling of the second launch is a fresh
/// `BufId` aliased to the first — and that spelling is **wrong**. The author's
/// rule is explicit: an in-place mutation must retain one `BufId`, because the
/// memory scheduler cannot infer that renamed IDs are one allocation
/// (`crates/compiler/notes.md:37-44`). `GraphBuilder::alias_bufs` only records
/// a parent id (`graph_ir.rs:1021-1048`); `plan_memory` never reads the alias
/// table (`graph_compiler.rs:594-639`), so the two ids get **different pool
/// offsets** under every shipped scheduler. The lower half then lands in one
/// allocation, the upper half in another, and every later round reads a
/// half-written buffer.
///
/// So the second launch is expressed the only way the compiler understands:
/// [`Self::InPlace`] — the destination is carried in as an input with
/// `modifies = true`, the node declares no fresh output, and the closure takes
/// its destination pointer from the input slice (the runtime passes carried
/// mutations there, `graph_exe.rs:1045-1057`). `access_from_node` then sees
/// the buffer in both the read and the write set, which is the definition of
/// a mutation, and the ATG inserts the ordering edge from the plain fold.
#[derive(Clone, Copy, Debug)]
pub enum FoldPleDst {
    /// A fresh buffer this launch is the sole producer of: declared as the
    /// node's output.
    Fresh(BufId),
    /// An existing buffer this launch overwrites part of: declared as a
    /// carried input (`modifies = true`), never as a renamed output.
    InPlace(BufId),
}

/// Insert a `fold_ple_from_evals` node (`fold_ple.rs:95` ← `mod.rs:977/991/1004`).
///
/// `dst_offset` is an `EF` element offset into `dst`'s buffer; see
/// [`FoldPleDst`] for why the rotated launch must be [`FoldPleDst::InPlace`]
/// and not a fresh aliased id.
#[allow(clippy::too_many_arguments)]
pub fn fold_ple_from_evals_ir(
    g: &mut GraphBuilder,
    input_matrix: BufId,
    input_len: usize,
    dst: FoldPleDst,
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
    let mut modifies = vec![false, false, false];
    // `None` ⇒ the destination is `outputs[0]`; `Some(i)` ⇒ it is the carried
    // input `inputs[i]`, because the runtime hands carried mutations to the
    // closure in the *input* slice (`graph_exe.rs:1045-1057`).
    let carried_dst = match dst {
        FoldPleDst::Fresh(_) => None,
        FoldPleDst::InPlace(b) => {
            assert!(
                !inputs.contains(&b),
                "fold_ple_from_evals_ir: in-place destination {b:?} is already an input"
            );
            inputs.push(b);
            modifies.push(true);
            Some(inputs.len() - 1)
        }
    };
    let outputs: Vec<BufId> = match dst {
        FoldPleDst::Fresh(b) => vec![b],
        FoldPleDst::InPlace(_) => vec![],
    };
    g.insert_blackbox_kernel(
        if rotate {
            "fold_ple_from_evals<rot>"
        } else {
            "fold_ple_from_evals"
        },
        inputs.into_iter(),
        outputs.into_iter(),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            let dst_base = match carried_dst {
                Some(i) => inputs[i],
                None => outputs[0],
            };
            let out_ptr = (dst_base as *mut EF).add(dst_offset);
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
/// pointer hole R6 closed for `ZerocheckCtx` / `LogupCtx`. The node
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
// The base+offset descriptor ABI (R6).
// ===========================================================================
//
// The eager path assembles `ZerocheckCtx` / `LogupCtx` / the `MainMatrixDesc`
// array on the host and uploads them (`batch_mle.rs:158-201`,
// `batch_mle_monomial.rs:176-192`). When those structs embedded raw device
// pointers, the planner saw an opaque leaf and could not alias-analyse through
// it to the buffers the evaluator would dereference — and the IR mirror had to
// build them with one device *materializer launch per pointer field* just to
// keep each address on a graph edge.
//
// Both problems go away once the descriptor arrays hold **integers**. Every
// device-pointer field is now a [`BaseOff`]: a byte offset from one
// `pool_base` pointer that the launcher takes as a kernel argument
// (`cuda/include/base_off.cuh`). The graph-IR encoding of that offset is
// literally `GraphExe::plan().offsets[b]` (`crates/compiler/src/planner/plan.rs:44-52`,
// exposed by `GraphExe::plan()`, `graph_exe.rs:317-324`), and the base is the
// pool we hand the exe ourselves via `GraphExe::set_scratch`
// (`graph_exe.rs:565`). Both terms are constant for the exe's lifetime — the
// same invariant that makes CUDA-graph capture legal (`graph_exe.rs:5-12`) —
// so the whole array is host-computable before any kernel runs, uploaded once
// as a graph input, and never touched again.
//
// What that buys, concretely:
//
// * no embedded device pointers anywhere in an uploaded struct;
// * no per-round, per-pointer device materializer launches in the captured graph;
// * the descriptor array is a *registered graph input*, so `GraphExe::run` refuses to run until
//   [`DescriptorPlan::bind`] has filled it (`graph_exe.rs:719`) — a forgotten bind is a hard error,
//   not silent garbage.
//
// What it does **not** buy (stated plainly, because it is easy to oversell): a
// batched multi-AIR kernel must still be handed *some* per-AIR array, and the
// kernel still indexes it by `air_idx`. Base+offset removes the embedded
// pointers and the runtime materialization; it does not collapse N descriptors
// into a scalar. Nor does it remove the evaluator's obligation to *declare*
// every buffer whose offset it embeds: an offset into a pool slot the planner
// has handed to someone else is exactly as wrong as a stale pointer. That
// declaration is still machine-derived — see [`OffSink`].

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

/// The pool base every [`BaseOff`] in this graph decodes against.
///
/// Published exactly once, by [`DescriptorPlan::bind`], after
/// `GraphExe::set_scratch` has taken ownership of the pool we allocated. The
/// evaluator closures read it when they launch; it is constant from that point
/// on, so a CUDA-graph capture bakes in the same value a plain replay uses.
///
/// `0` means "not published yet" — a real CUDA allocation is never at address
/// zero, and [`Self::get`] panics rather than launching against a null base.
#[derive(Clone, Debug, Default)]
pub struct PoolBase(Arc<AtomicU64>);

impl PoolBase {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the base. Idempotent for the same value; panics on a second,
    /// different base, because the descriptors already encoded against the
    /// first one would silently address the wrong pool.
    fn publish(&self, base: *const u8) {
        let v = base as usize as u64;
        assert_ne!(v, 0, "pool base must not be null");
        let prev = self.0.swap(v, Ordering::SeqCst);
        assert!(
            prev == 0 || prev == v,
            "pool base republished ({prev:#x} -> {v:#x}); descriptors encoded \
             against the old base would address the wrong pool"
        );
    }

    /// The published base. Panics if [`DescriptorPlan::bind`] has not run.
    pub fn get(&self) -> *const u8 {
        let v = self.0.load(Ordering::SeqCst);
        assert_ne!(
            v, 0,
            "pool base not published: call `DescriptorPlan::bind` after \
             `GraphCompiler::compile` and before `GraphExe::run`"
        );
        v as *const u8
    }
}

/// Sink for the device-pointer fields of one descriptor, in ABI order.
///
/// Every `*_desc` writer below funnels **all** of its pointer fields through
/// this trait, and the trait is instantiated twice over the same code:
///
/// * [`ReadCollector`] at graph-build time, which records the `BufId`s and returns a placeholder
///   offset — that recorded list *is* the access set the consuming evaluator declares;
/// * [`OffEncoder`] at bind time, which returns the real `plan().offsets[b] + byte_offset`.
///
/// This is what makes an under-declared access set unrepresentable: a new
/// pointer field cannot reach the device without also flowing into the read
/// set, because both come from the same call.
trait OffSink {
    fn off(&mut self, arg: DevicePtrArg) -> BaseOff;

    /// The **raw device address** of `arg`, for the pointer tables whose ABI
    /// is a bare `T*` rather than a [`BaseOff`] (`batch_fold_mle`'s
    /// `input_matrices` / `output_matrices`, `interpolate_columns`' column
    /// table). Same collect-at-build / encode-at-bind contract as
    /// [`Self::off`]: the `BufId` still flows into the read set, so a table
    /// entry cannot reach the device without its buffer being declared.
    fn addr(&mut self, arg: DevicePtrArg) -> u64;
}

/// Build-time [`OffSink`]: collects the graph buffers a descriptor references.
#[derive(Default)]
struct ReadCollector {
    reads: Vec<BufId>,
}

impl OffSink for ReadCollector {
    fn off(&mut self, arg: DevicePtrArg) -> BaseOff {
        if let DevicePtrArg::Graph { buf, .. } = arg {
            if !self.reads.contains(&buf) {
                self.reads.push(buf);
            }
        }
        // Never reaches the device: `bind` re-runs the same writer with
        // [`OffEncoder`].
        BaseOff::NULL
    }

    fn addr(&mut self, arg: DevicePtrArg) -> u64 {
        let _ = self.off(arg);
        0
    }
}

/// Bind-time [`OffSink`]: the real pool encoding.
struct OffEncoder<'a> {
    /// `GraphExe::plan().offsets`, byte offset per `BufId` in the pool.
    offsets: &'a [Option<u64>],
    /// The pool base, needed only to re-base [`DevicePtrArg::Static`]
    /// addresses the graph does not own.
    base: u64,
}

impl OffSink for OffEncoder<'_> {
    fn off(&mut self, arg: DevicePtrArg) -> BaseOff {
        match arg {
            // `Static(0)` is the eager path's "absent" encoding.
            DevicePtrArg::Static(0) => BaseOff::NULL,
            // A keygen-static proving-key table: not in the pool, so store the
            // delta that recovers it. Wrapping is deliberate and exact — the
            // kernel adds the same wrapped value back to `base`.
            DevicePtrArg::Static(addr) => {
                BaseOff::from_offset((addr as u64).wrapping_sub(self.base))
            }
            DevicePtrArg::Graph { buf, byte_offset } => {
                BaseOff::from_offset(self.pool_offset(buf) + byte_offset as u64)
            }
        }
    }

    fn addr(&mut self, arg: DevicePtrArg) -> u64 {
        match arg {
            // The eager path's "absent" encoding is a literal null pointer.
            DevicePtrArg::Static(0) => 0,
            // Not in the pool: the address is already absolute.
            DevicePtrArg::Static(addr) => addr as u64,
            DevicePtrArg::Graph { buf, byte_offset } => {
                self.base + self.pool_offset(buf) + byte_offset as u64
            }
        }
    }
}

impl OffEncoder<'_> {
    fn pool_offset(&self, buf: BufId) -> u64 {
        self.offsets[buf.0].unwrap_or_else(|| {
            panic!(
                "buffer {buf:?} has no pool slot on the plan's device; a descriptor \
                 field cannot be encoded against the pool base"
            )
        })
    }
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

/// The five logical components of `EvalCoreCtx` (`cuda/logup_zerocheck.rs:39`).
#[derive(Clone, Copy, Debug)]
pub struct EvalCoreCtxArgs {
    pub d_selectors: DevicePtrArg,
    /// `d_preprocessed.data`; [`DevicePtrArg::NULL`] when the AIR has none.
    pub d_preprocessed_data: DevicePtrArg,
    /// `d_preprocessed.air_width`; `0` when the AIR has no preprocessed data.
    pub preprocessed_air_width: u32,
    /// The `MainMatrixDesc` array for this AIR.
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

/// Every field of `BatchSRingTraceDesc` (`cuda/src/logup_zerocheck/ring.cu`).
///
/// Unlike the three `*Ctx` arrays this one is not consumed by an evaluator: it
/// tells the ring kernel where *this round's* compact evaluator outputs live
/// for each trace. Both pointer fields still flow through [`OffSink`], so the
/// buffers they address land in the read set the consuming node declares.
#[derive(Clone, Copy, Debug)]
pub struct BatchSRingTraceDescArgs {
    /// `[num_x] EF` for this trace inside its zerocheck batch, or
    /// [`DevicePtrArg::NULL`].
    pub zc_evals: DevicePtrArg,
    /// `[num_x] Frac<EF>` for this trace inside its logup batch, or
    /// [`DevicePtrArg::NULL`].
    pub logup_evals: DevicePtrArg,
    pub n_lift: u32,
    pub flags: u32,
}

// ---------------------------------------------------------------------------
// Descriptor writers.
//
// One function per `#[repr(C)]` context struct. Each routes *every* device
// pointer field through the [`OffSink`], and is run twice: once at graph-build
// time with a [`ReadCollector`] (to derive the evaluator's access set) and once
// at bind time with an [`OffEncoder`] (to produce the bytes that are uploaded).

fn write_main_matrix_desc<S: OffSink>(
    s: &mut S,
    data: DevicePtrArg,
    air_width: u32,
) -> MainMatrixDesc {
    MainMatrixDesc {
        data: s.off(data),
        air_width,
    }
}

/// Mirrors the eager `EvalCoreCtx` construction (`batch_mle.rs:178-183`).
fn write_eval_core_ctx<S: OffSink>(s: &mut S, a: &EvalCoreCtxArgs) -> EvalCoreCtx {
    EvalCoreCtx {
        d_selectors: s.off(a.d_selectors),
        d_preprocessed: write_main_matrix_desc(s, a.d_preprocessed_data, a.preprocessed_air_width),
        d_main: s.off(a.d_main),
        d_public: s.off(a.d_public),
    }
}

/// Mirrors `ZerocheckMleBatchBuilder::new`'s per-trace struct
/// (`batch_mle.rs:158-201`) field for field.
fn write_zerocheck_ctx<S: OffSink>(s: &mut S, a: &ZerocheckCtxArgs) -> ZerocheckCtx {
    ZerocheckCtx {
        eval_ctx: write_eval_core_ctx(s, &a.eval_ctx),
        d_intermediates: s.off(a.d_intermediates),
        num_y: a.num_y,
        d_eq_xi: s.off(a.d_eq_xi),
        d_rules: s.off(a.d_rules),
        rules_len: a.rules_len,
        d_used_nodes: s.off(a.d_used_nodes),
        used_nodes_len: a.used_nodes_len,
        buffer_size: a.buffer_size,
    }
}

/// Mirrors `LogupMleBatchBuilder::new`'s per-trace struct
/// (`batch_mle.rs:297-353`) field for field.
fn write_logup_ctx<S: OffSink>(s: &mut S, a: &LogupCtxArgs) -> LogupCtx {
    LogupCtx {
        eval_ctx: write_eval_core_ctx(s, &a.eval_ctx),
        d_intermediates: s.off(a.d_intermediates),
        num_y: a.num_y,
        d_eq_xi: s.off(a.d_eq_xi),
        d_challenges: s.off(a.d_challenges),
        d_eq_3bs: s.off(a.d_eq_3bs),
        d_rules: s.off(a.d_rules),
        rules_len: a.rules_len,
        d_used_nodes: s.off(a.d_used_nodes),
        d_pair_idxs: s.off(a.d_pair_idxs),
        used_nodes_len: a.used_nodes_len,
        buffer_size: a.buffer_size,
    }
}

/// Mirrors the ring's per-trace descriptor (`ring.cu`) field for field.
fn write_batch_s_ring_trace_desc<S: OffSink>(
    s: &mut S,
    a: &BatchSRingTraceDescArgs,
) -> BatchSRingTraceDesc {
    BatchSRingTraceDesc {
        zc_evals: s.off(a.zc_evals),
        logup_evals: s.off(a.logup_evals),
        n_lift: a.n_lift,
        flags: a.flags,
    }
}

/// One element of a descriptor array, recorded at graph-build time and encoded
/// at bind time.
#[derive(Clone, Debug)]
enum DescElem {
    MainMatrix {
        data: DevicePtrArg,
        air_width: u32,
    },
    Zerocheck(Box<ZerocheckCtxArgs>),
    Logup(Box<LogupCtxArgs>),
    /// One entry of a bare `T*` pointer table (no [`BaseOff`] indirection):
    /// `batch_fold_mle`'s `input_matrices` / `output_matrices`
    /// (`cuda/src/sumcheck.cu:209-210`) and `interpolate_columns`' column
    /// table (`mod.rs:1196-1199`). Encodes to the 8-byte absolute device
    /// address.
    RawPtr(DevicePtrArg),
    /// One entry of the ring's per-trace evaluator-output table
    /// (`batch_s_ring_pre`).
    BatchSRingTrace(Box<BatchSRingTraceDescArgs>),
}

// ---------------------------------------------------------------------------
// ABI serialization.
//
// Field-wise, into a zero-filled buffer of the struct's exact `size_of`.
//
// The obvious alternative — `slice::from_raw_parts(v as *const T as *const u8,
// size_of::<T>())` — is wrong twice over. Every one of these `#[repr(C)]`
// structs has padding: `MainMatrixDesc` is a `u64` followed by a `u32` at
// align 8, so four trailing bytes; `ZerocheckCtx` and `LogupCtx` add interior
// padding around their `u32`s. Rust never initializes padding, so reading it
// as `u8` is an uninitialized read (undefined behaviour), and the bytes it
// yields are not a function of the descriptor's fields. That second half is
// what makes it a *test* problem and not only a soundness one: almost every
// oracle in this module compares encoded descriptor bytes for equality
// (`descriptors_decode_like_the_eager_pointer_path`'s replay leg compares
// them literally, padding included), and a byte that can differ run to run
// silently weakens all of them.
//
// Writing field by field at `offset_of!` positions fixes both: no byte of the
// source value is ever read except through a named field, and every byte of
// the output is either a field or a deterministic zero.

/// A zero-filled ABI image of one `#[repr(C)]` struct.
struct AbiBytes(Vec<u8>);

impl AbiBytes {
    fn new<T>() -> Self {
        AbiBytes(vec![0u8; size_of::<T>()])
    }

    /// Place one scalar field at its `offset_of!` position.
    ///
    /// `V` must be a type with no padding of its own — an integer, or a
    /// `#[repr(transparent)]` newtype over one. Every call below passes
    /// `u64` / `u32` / `usize`.
    fn put<V: Copy>(&mut self, at: usize, v: V) {
        let n = size_of::<V>();
        assert!(
            at + n <= self.0.len(),
            "ABI field at {at}..{} overflows a {}-byte image",
            at + n,
            self.0.len()
        );
        // SAFETY: `v` is a fully initialized `V` with no padding, the
        // destination range was bounds-checked above, and `Vec<u8>` has
        // alignment 1 so an unaligned byte copy is well-defined.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &v as *const V as *const u8,
                self.0.as_mut_ptr().add(at),
                n,
            );
        }
    }

    /// Place a nested struct's image at its `offset_of!` position.
    fn put_nested(&mut self, at: usize, sub: AbiBytes) {
        assert!(
            at + sub.0.len() <= self.0.len(),
            "nested ABI field at {at}..{} overflows a {}-byte image",
            at + sub.0.len(),
            self.0.len()
        );
        self.0[at..at + sub.0.len()].copy_from_slice(&sub.0);
    }

    fn into_vec(self) -> Vec<u8> {
        self.0
    }
}

fn encode_main_matrix_desc(d: &MainMatrixDesc) -> AbiBytes {
    let mut b = AbiBytes::new::<MainMatrixDesc>();
    b.put(offset_of!(MainMatrixDesc, data), d.data.0);
    b.put(offset_of!(MainMatrixDesc, air_width), d.air_width);
    b
}

fn encode_eval_core_ctx(c: &EvalCoreCtx) -> AbiBytes {
    let mut b = AbiBytes::new::<EvalCoreCtx>();
    b.put(offset_of!(EvalCoreCtx, d_selectors), c.d_selectors.0);
    b.put_nested(
        offset_of!(EvalCoreCtx, d_preprocessed),
        encode_main_matrix_desc(&c.d_preprocessed),
    );
    b.put(offset_of!(EvalCoreCtx, d_main), c.d_main.0);
    b.put(offset_of!(EvalCoreCtx, d_public), c.d_public.0);
    b
}

fn encode_zerocheck_ctx(c: &ZerocheckCtx) -> AbiBytes {
    let mut b = AbiBytes::new::<ZerocheckCtx>();
    b.put_nested(
        offset_of!(ZerocheckCtx, eval_ctx),
        encode_eval_core_ctx(&c.eval_ctx),
    );
    b.put(
        offset_of!(ZerocheckCtx, d_intermediates),
        c.d_intermediates.0,
    );
    b.put(offset_of!(ZerocheckCtx, num_y), c.num_y);
    b.put(offset_of!(ZerocheckCtx, d_eq_xi), c.d_eq_xi.0);
    b.put(offset_of!(ZerocheckCtx, d_rules), c.d_rules.0);
    b.put(offset_of!(ZerocheckCtx, rules_len), c.rules_len);
    b.put(offset_of!(ZerocheckCtx, d_used_nodes), c.d_used_nodes.0);
    b.put(offset_of!(ZerocheckCtx, used_nodes_len), c.used_nodes_len);
    b.put(offset_of!(ZerocheckCtx, buffer_size), c.buffer_size);
    b
}

fn encode_logup_ctx(c: &LogupCtx) -> AbiBytes {
    let mut b = AbiBytes::new::<LogupCtx>();
    b.put_nested(
        offset_of!(LogupCtx, eval_ctx),
        encode_eval_core_ctx(&c.eval_ctx),
    );
    b.put(offset_of!(LogupCtx, d_intermediates), c.d_intermediates.0);
    b.put(offset_of!(LogupCtx, num_y), c.num_y);
    b.put(offset_of!(LogupCtx, d_eq_xi), c.d_eq_xi.0);
    b.put(offset_of!(LogupCtx, d_challenges), c.d_challenges.0);
    b.put(offset_of!(LogupCtx, d_eq_3bs), c.d_eq_3bs.0);
    b.put(offset_of!(LogupCtx, d_rules), c.d_rules.0);
    b.put(offset_of!(LogupCtx, rules_len), c.rules_len);
    b.put(offset_of!(LogupCtx, d_used_nodes), c.d_used_nodes.0);
    b.put(offset_of!(LogupCtx, d_pair_idxs), c.d_pair_idxs.0);
    b.put(offset_of!(LogupCtx, used_nodes_len), c.used_nodes_len);
    b.put(offset_of!(LogupCtx, buffer_size), c.buffer_size);
    b
}

fn encode_batch_s_ring_trace_desc(d: &BatchSRingTraceDesc) -> AbiBytes {
    let mut b = AbiBytes::new::<BatchSRingTraceDesc>();
    b.put(offset_of!(BatchSRingTraceDesc, zc_evals), d.zc_evals.0);
    b.put(
        offset_of!(BatchSRingTraceDesc, logup_evals),
        d.logup_evals.0,
    );
    b.put(offset_of!(BatchSRingTraceDesc, n_lift), d.n_lift);
    b.put(offset_of!(BatchSRingTraceDesc, flags), d.flags);
    b
}

impl DescElem {
    /// Run this element's writer against `sink`, returning its raw ABI bytes.
    ///
    /// The bytes are built field by field (see [`AbiBytes`]); no padding byte
    /// of the constructed struct is ever read, so the encoding is a pure
    /// function of the descriptor's fields.
    fn encode<S: OffSink>(&self, sink: &mut S) -> Vec<u8> {
        match self {
            DescElem::MainMatrix { data, air_width } => {
                encode_main_matrix_desc(&write_main_matrix_desc(sink, *data, *air_width)).into_vec()
            }
            DescElem::Zerocheck(a) => {
                encode_zerocheck_ctx(&write_zerocheck_ctx(sink, a)).into_vec()
            }
            DescElem::Logup(a) => encode_logup_ctx(&write_logup_ctx(sink, a)).into_vec(),
            // A bare `u64` has no padding, but it still goes through the same
            // explicit path rather than a whole-struct byte view.
            DescElem::RawPtr(data) => sink.addr(*data).to_ne_bytes().to_vec(),
            DescElem::BatchSRingTrace(a) => {
                encode_batch_s_ring_trace_desc(&write_batch_s_ring_trace_desc(sink, a)).into_vec()
            }
        }
    }
}

/// One descriptor array: a registered graph input whose bytes are computed on
/// the host after `compile()` and uploaded once.
#[derive(Clone, Debug)]
struct DescArray {
    buf: BufId,
    name: String,
    elem_bytes: usize,
    /// `None` for an element no `set_*` call ever filled — it stays zeroed,
    /// exactly as the old `insert_memset(arr, 0)` left it.
    elems: Vec<Option<DescElem>>,
}

/// Every base+offset descriptor array in one graph, plus the pool base they
/// decode against.
///
/// Built while the graph is built; filled in by [`Self::bind`] once the plan's
/// buffer offsets and our pool's address are both known.
#[derive(Clone, Debug, Default)]
pub struct DescriptorPlan {
    base: PoolBase,
    arrays: Vec<DescArray>,
}

/// Handle to one array inside a [`DescriptorPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescArrayId(usize);

impl DescriptorPlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// The base handle the evaluator closures read at launch time.
    pub fn pool_base(&self) -> &PoolBase {
        &self.base
    }

    /// Register `buf` as a `len`-element descriptor array of `T`, and as a
    /// *graph input* — which is what makes a forgotten [`Self::bind`] a hard
    /// error at `GraphExe::run` (`graph_exe.rs:719`) instead of silent garbage.
    fn add_array<T>(
        &mut self,
        g: &mut GraphBuilder,
        buf: BufId,
        name: &str,
        len: usize,
    ) -> DescArrayId {
        g.register_input(buf);
        self.arrays.push(DescArray {
            buf,
            name: name.to_string(),
            elem_bytes: size_of::<T>(),
            elems: vec![None; len],
        });
        DescArrayId(self.arrays.len() - 1)
    }

    /// Record one element and return the graph buffers it references.
    ///
    /// The returned reads are the machine-derived half of the Principle-1
    /// access set: every evaluator that dereferences this descriptor must
    /// declare at least these buffers, or the planner will hand their pool
    /// slots to someone else while the evaluator still needs them
    /// (`planner/list_v1.rs` liveness is derived only from declared accesses,
    /// `planner/abstract_timing.rs`). Deriving the list from the same writer
    /// that produces the bytes — instead of retyping it at the call site — is
    /// what makes an under-declared set unrepresentable.
    #[must_use = "the declared reads must reach the evaluator's `ctx_reads`"]
    fn set(&mut self, arr: DescArrayId, idx: usize, elem: DescElem) -> Vec<BufId> {
        let mut rc = ReadCollector::default();
        let _ = elem.encode(&mut rc);
        let a = &mut self.arrays[arr.0];
        assert!(
            idx < a.elems.len(),
            "descriptor index {idx} out of range for `{}` ({} elements)",
            a.name,
            a.elems.len()
        );
        a.elems[idx] = Some(elem);
        rc.reads
    }

    /// Record `out[idx]` of a `MainMatrixDesc` array.
    #[must_use = "the declared reads must reach the evaluator's `ctx_reads`"]
    pub fn set_main_matrix_desc(
        &mut self,
        arr: DescArrayId,
        idx: usize,
        data: DevicePtrArg,
        air_width: u32,
    ) -> Vec<BufId> {
        self.set(arr, idx, DescElem::MainMatrix { data, air_width })
    }

    /// Record `out[idx]` of a `ZerocheckCtx` array.
    #[must_use = "the declared reads must reach the evaluator's `ctx_reads`"]
    pub fn set_zerocheck_ctx(
        &mut self,
        arr: DescArrayId,
        idx: usize,
        args: ZerocheckCtxArgs,
    ) -> Vec<BufId> {
        self.set(arr, idx, DescElem::Zerocheck(Box::new(args)))
    }

    /// Register `buf` as a `len`-entry table of bare device pointers.
    ///
    /// Unlike the three `*Ctx` arrays this is not a `BaseOff` struct — the
    /// consuming kernels (`batch_fold_mle`, `interpolate_columns`) take a
    /// plain `T*const*`, so each entry encodes to the absolute address
    /// `pool_base + offset`. Everything else — registration as a graph input,
    /// the "forgotten `bind` is a hard error" guarantee, and the read-set
    /// derivation — is identical.
    pub fn add_ptr_array(
        &mut self,
        g: &mut GraphBuilder,
        buf: BufId,
        name: &str,
        len: usize,
    ) -> DescArrayId {
        self.add_array::<*const u8>(g, buf, name, len)
    }

    /// Record `out[idx]` of a bare pointer table.
    #[must_use = "the declared reads must reach the consuming node's inputs"]
    pub fn set_ptr(&mut self, arr: DescArrayId, idx: usize, data: DevicePtrArg) -> Vec<BufId> {
        self.set(arr, idx, DescElem::RawPtr(data))
    }

    /// Record `out[idx]` of a `LogupCtx` array.
    #[must_use = "the declared reads must reach the evaluator's `ctx_reads`"]
    pub fn set_logup_ctx(
        &mut self,
        arr: DescArrayId,
        idx: usize,
        args: LogupCtxArgs,
    ) -> Vec<BufId> {
        self.set(arr, idx, DescElem::Logup(Box::new(args)))
    }

    /// Register `buf` as a `len`-element [`BatchSRingTraceDesc`] array.
    pub fn add_batch_s_ring_trace_array(
        &mut self,
        g: &mut GraphBuilder,
        buf: BufId,
        name: &str,
        len: usize,
    ) -> DescArrayId {
        self.add_array::<BatchSRingTraceDesc>(g, buf, name, len)
    }

    /// Record `out[idx]` of a [`BatchSRingTraceDesc`] array.
    #[must_use = "the declared reads must reach the consuming node's inputs"]
    pub fn set_batch_s_ring_trace(
        &mut self,
        arr: DescArrayId,
        idx: usize,
        args: BatchSRingTraceDescArgs,
    ) -> Vec<BufId> {
        self.set(arr, idx, DescElem::BatchSRingTrace(Box::new(args)))
    }

    /// Every graph buffer reachable from the descriptor array `root` by
    /// following stored offsets, transitively.
    ///
    /// An array's elements contribute the `BufId`s they reference; if one of
    /// those is itself a descriptor array (a `ZerocheckCtx`'s `d_main` points
    /// at a `MainMatrixDesc` array), its references are folded in too. This is
    /// the exact dereference closure an evaluator's declared access set must
    /// cover — see [`ZerocheckEvalBufs::ctx_reads`].
    pub fn referenced_bufs(&self, root: BufId) -> Vec<BufId> {
        let mut seen = vec![root];
        loop {
            let before = seen.len();
            for a in &self.arrays {
                if !seen.contains(&a.buf) {
                    continue;
                }
                for elem in a.elems.iter().flatten() {
                    let mut rc = ReadCollector::default();
                    let _ = elem.encode(&mut rc);
                    for b in rc.reads {
                        if !seen.contains(&b) {
                            seen.push(b);
                        }
                    }
                }
            }
            if seen.len() == before {
                return seen;
            }
        }
    }

    /// `(buffer, name, filled elements, total elements)` per array, in
    /// registration order. Exposed for the R6 invariant tests.
    pub fn array_summary(&self) -> Vec<(BufId, String, usize, usize)> {
        self.arrays
            .iter()
            .map(|a| {
                (
                    a.buf,
                    a.name.clone(),
                    a.elems.iter().flatten().count(),
                    a.elems.len(),
                )
            })
            .collect()
    }

    /// The raw bytes of every array, encoded against `offsets` and `base`.
    ///
    /// Split out from [`Self::bind`] so a test can encode without a device.
    fn encode_all(&self, offsets: &[Option<u64>], base: u64) -> Vec<(BufId, Vec<u8>)> {
        self.arrays
            .iter()
            .map(|a| {
                let mut bytes = vec![0u8; a.elems.len() * a.elem_bytes];
                for (i, elem) in a.elems.iter().enumerate() {
                    let Some(elem) = elem else { continue };
                    let mut enc = OffEncoder { offsets, base };
                    let e = elem.encode(&mut enc);
                    assert_eq!(
                        e.len(),
                        a.elem_bytes,
                        "descriptor element size mismatch in `{}`",
                        a.name
                    );
                    bytes[i * a.elem_bytes..(i + 1) * a.elem_bytes].copy_from_slice(&e);
                }
                (a.buf, bytes)
            })
            .collect()
    }

    /// Hand `pool` to `exe` and publish its base. **One-shot per exe.**
    ///
    /// The ordering inside is load-bearing:
    ///
    /// 1. `set_scratch` must precede the first `set_input`/`run`, because the pool is what gives
    ///    every buffer its stable address (`graph_exe.rs:565-572` rejects a late call);
    /// 2. the base is published before any launch, so every evaluator closure sees it.
    ///
    /// `pool` must be at least `exe.scratch_bytes()` long; use
    /// [`Self::alloc_pool`]. Calling this twice on one exe is an error —
    /// `set_scratch` rejects a pool that is already installed. The repeatable
    /// half is [`Self::upload`].
    pub fn install_pool(
        &self,
        exe: &mut crypto_compiler::graph_exe::GraphExe,
        pool: DeviceBuffer<u8>,
    ) -> Result<(), crypto_compiler::CompileError> {
        let base = pool.as_mut_raw_ptr() as *const u8;
        exe.set_scratch(pool)?;
        self.base.publish(base);
        Ok(())
    }

    /// Encode every descriptor array against the installed pool and upload it
    /// into its input slot. **Call before every `run` / `launch_graph`.**
    ///
    /// # Why this is not one-shot
    ///
    /// Graph inputs are *not* preserved across an execution — the compiler
    /// reuses their slots to save memory, and the author's rule is explicit
    /// that the inputs must be set for every launch
    /// (`crates/compiler/notes.md:46-52`). ListV1 happens to pin graph inputs
    /// and outputs through the schedule (`planner/list_v1.rs:638-655,
    /// 789-795`), which is why a single upload appeared to work; ListV2 —
    /// which is what `SchedulerConfig::default()` ships
    /// (`graph_compiler_config.rs:129-147`) — gives an input birth 0 but
    /// infinite death only to *outputs* (`planner/list_v2.rs:371-401`), so a
    /// descriptor array's slot is free for reuse after its last first-run
    /// consumer.
    ///
    /// The failure that causes is not a benign stale read. The decoder adds
    /// whatever 64-bit value now occupies the slot to the pool base
    /// (`cuda/include/base_off.cuh:34-54`), so a replay that skipped the
    /// upload reads either wrong in-pool data or an invalid device address.
    ///
    /// The upstream remedy the note suggests — "declare it as a graph output"
    /// — is unavailable here: interface validation rejects an id registered in
    /// both lists and requires every output to have a writer
    /// (`graph_compiler.rs:1278-1293`), and a descriptor array has no
    /// in-graph writer.
    ///
    /// The offsets are re-read from `exe.plan()` on every call. They are
    /// constant for the exe's lifetime (`plan()` is immutable and the pool is
    /// never reallocated), so re-encoding produces identical bytes — which is
    /// exactly what makes a CUDA-graph replay legal.
    pub fn upload(
        &self,
        exe: &mut crypto_compiler::graph_exe::GraphExe,
        ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Result<(), crypto_compiler::CompileError> {
        use openvm_cuda_common::copy::MemCopyH2D;

        let base = self.base.get();
        assert!(
            !base.is_null(),
            "DescriptorPlan::upload before install_pool: the pool base is not published yet"
        );
        let encoded = self.encode_all(&exe.plan().offsets, base as usize as u64);
        for (buf, bytes) in encoded {
            let i = (0..exe.num_inputs())
                .find(|&i| exe.input_buf_id(i) == buf)
                .unwrap_or_else(|| {
                    panic!("descriptor array {buf:?} is not a registered graph input")
                });
            assert_eq!(
                exe.input_size(i),
                bytes.len(),
                "descriptor array {buf:?} is {} bytes in the plan, {} encoded",
                exe.input_size(i),
                bytes.len()
            );
            let staged = bytes
                .to_device_on(ctx)
                .expect("stage descriptor array to device");
            exe.set_input(ctx, i, &staged)?;
        }
        Ok(())
    }

    /// [`Self::install_pool`] followed by [`Self::upload`] — the first-run
    /// convenience form.
    ///
    /// This is **not** enough on its own for a second execution: call
    /// [`Self::upload`] again (together with [`PhaseInputBinder::bind`])
    /// before every subsequent `run`. See [`Self::upload`] for why.
    pub fn bind(
        &self,
        exe: &mut crypto_compiler::graph_exe::GraphExe,
        ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
        pool: DeviceBuffer<u8>,
    ) -> Result<(), crypto_compiler::CompileError> {
        self.install_pool(exe, pool)?;
        self.upload(exe, ctx)
    }

    /// Allocate a pool of exactly the size the plan needs.
    pub fn alloc_pool(
        exe: &crypto_compiler::graph_exe::GraphExe,
        ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> DeviceBuffer<u8> {
        DeviceBuffer::<u8>::with_capacity_on(exe.scratch_bytes().max(1), ctx)
    }
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
    /// The `ZerocheckCtx` / `LogupCtx` / `MonomialAirCtx` array. For the two
    /// DAG kinds it is a registered graph input filled by
    /// [`DescriptorPlan::bind`]; for `Monomial` it is still a zeroed
    /// placeholder (see the TODO on [`push_monomial_reads`]).
    pub air_ctxs: BufId,
    /// [`Self::air_ctxs`]'s slot in the [`DescriptorPlan`]. Unused for the
    /// monomial placeholder arrays.
    pub ctx_array: DescArrayId,
    /// The pool base the kernel decodes every [`BaseOff`] against.
    pub pool_base: PoolBase,
    /// Read by the reducer only.
    pub air_block_offsets: BufId,
    /// `lambda_pows`; unused by the monomial evaluators.
    pub lambda_pows: Option<BufId>,
    /// Principle-1 access set: every buffer the kernels **read** that is
    /// reachable only by dereferencing `air_ctxs` — selectors, folded main /
    /// preprocessed matrices, the `MainMatrixDesc` array, public
    /// values, `eq_xi`, the rule / used-node / pair-index streams, logup
    /// challenges and `eq_3bs`.
    ///
    /// Base+offset does **not** retire this obligation: an offset into a pool
    /// slot the planner has reassigned is exactly as wrong as a stale pointer.
    /// What it retires is the *materializer* node whose own access set could
    /// be under-declared; the list here is still machine-derived, now from the
    /// descriptor writer itself ([`OffSink`]).
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
    let pool_base = bufs.pool_base.clone();
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
                pool_base.get(),
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
    let pool_base = bufs.pool_base.clone();
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
                pool_base.get(),
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
    let pool_base = bufs.pool_base.clone();
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
                    pool_base.get(),
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
                    pool_base.get(),
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
    /// The pool base the kernel decodes every [`BaseOff`] against.
    pub pool_base: PoolBase,
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
    /// selectors, folded matrices, the `MainMatrixDesc` array, public
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
// WHY: R6 scoped the base+offset ABI for `MainMatrixDesc` / `ZerocheckCtx` /
//   `LogupCtx` only; the three monomial ctx structs are not covered and are
//   still staged host-side (here: zeroed placeholders).
// RISK: a `_dev_challenge` port of the monomial path has to reach into the
//   struct, i.e. it needs a `LogupMonomialCommonCtx` descriptor writer that
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
    let pool_base = bufs.pool_base.clone();
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
                pool_base.get(),
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
///
/// `r` is the round challenge as a **device buffer**, not a host `EF`: it is
/// sampled from the graph transcript and must never be resolved on the host,
/// which is the whole point of the ring. The launch goes to
/// `batch_fold_mle_dev_challenge` (`cuda/src/sumcheck.cu:200-238`,
/// `cuda/mod.rs:201-240`), whose `DEV_CH == true` instantiation reads it from
/// `*r_dev`; that ABI is byte-for-byte equivalent to the by-value one and is
/// already pinned by `batch_fold_mle_dev_challenge_matches_host_value`.
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
    r: BufId,
) {
    // The kernel reaches its operands through `input_ptrs` / `output_ptrs`,
    // which the planner cannot see through. Naming the pointed-to buffers as
    // real inputs and outputs of the node restores every dependency edge.
    //
    // `srcs` therefore starts at index 5 of `inputs`, but only as a dependency
    // declaration: the closure must NOT build views over those slots, it
    // reaches the matrices through the two pointer tables.
    let mut inputs = vec![input_ptrs, output_ptrs, widths, log_output_heights, r];
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
            batch_fold_mle_dev_challenge(
                ins.as_ptr(),
                outs.as_mut_ptr(),
                widths.as_ptr(),
                num_matrices,
                logh.as_ptr(),
                max_output_cells,
                inputs[4] as *const EF,
                stream,
            )
            .expect("batch_fold_mle_dev_challenge");
            forget(ins);
            forget(outs);
            forget(widths);
            forget(logh);
        },
    );
}

/// The `interpolate_columns` column table (`mod.rs:1196-1203`), as a
/// bind-time-filled descriptor array.
///
/// One pointer **per column**, in the eager order — **selectors first, then
/// every matrix in plan order** — and within a matrix `buffer + col * height`
/// with `height = 2 * num_y` (the pre-fold height, `mod.rs:1199-1200`).
///
/// The order is the whole point of factoring this out. The consuming node's
/// `srcs` list is only a dependency declaration, so it may be in any order;
/// the *table* may not. Sharing one builder between the phase driver and
/// `interpolate_columns_ir_column_table_matches_eager` is what keeps the
/// driver's order under test.
///
/// Returns `(table, the buffers the table points at)`.
fn emit_column_table(
    g: &mut GraphBuilder,
    descs: &mut DescriptorPlan,
    device: DeviceType,
    name: &str,
    sels: BufId,
    mats: &[(BufId, usize)],
    num_y: usize,
) -> (BufId, Vec<BufId>) {
    let num_columns = 3 + mats.iter().map(|&(_, w)| w).sum::<usize>();
    let table = add_typed_buf::<*const EF>(g, device, name, num_columns);
    let arr = descs.add_ptr_array(g, table, name, num_columns);
    let height = 2 * num_y;
    let mut reads: Vec<BufId> = Vec::new();
    let mut cols: Vec<(BufId, usize)> = (0..3).map(|c| (sels, c)).collect();
    for &(b, w) in mats {
        cols.extend((0..w).map(|c| (b, c)));
    }
    assert_eq!(
        cols.len(),
        num_columns,
        "column table `{name}` under-filled"
    );
    for (k, (b, col)) in cols.into_iter().enumerate() {
        extend_reads(
            &mut reads,
            descs.set_ptr(arr, k, DevicePtrArg::at(b, col * height * size_of::<EF>())),
        );
    }
    (table, reads)
}

/// The `(input_matrices, output_matrices)` pointer tables one
/// [`batch_fold_mle_ir`] launch dereferences (`cuda/src/sumcheck.cu:209-210`),
/// as bind-time-filled descriptor arrays.
///
/// These were `insert_memset(_, 0)` before, which is the worst failure shape
/// available here: `fold_mle` reads `width = widths[mat_idx]` and returns for
/// every thread once `output_height * width == 0`
/// (`cuda/include/sumcheck.cuh:306-309`), so a zeroed control table makes the
/// kernel a **silent no-op** — no null is dereferenced, nothing errors, and
/// the `dsts` keep whatever the pool slot held.
fn fold_ptr_tables(
    g: &mut GraphBuilder,
    descs: &mut DescriptorPlan,
    device: DeviceType,
    name: &str,
    srcs: &[BufId],
    dsts: &[BufId],
) -> (BufId, BufId) {
    assert_eq!(srcs.len(), dsts.len(), "one output matrix per input matrix");
    let n = srcs.len();
    let in_ptrs = add_typed_buf::<*const EF>(g, device, &format!("{name}_in"), n);
    let out_ptrs = add_typed_buf::<*mut EF>(g, device, &format!("{name}_out"), n);
    let in_arr = descs.add_ptr_array(g, in_ptrs, &format!("{name}_in"), n);
    let out_arr = descs.add_ptr_array(g, out_ptrs, &format!("{name}_out"), n);
    for (k, (&s, &d)) in srcs.iter().zip(dsts.iter()).enumerate() {
        // Every buffer these tables point at is already an explicit input
        // (`srcs`) or output (`dsts`) of the node, so the read sets the
        // writers derive are covered by construction. Asserting it here is
        // what keeps that true if either list is ever edited.
        let r_in = descs.set_ptr(in_arr, k, DevicePtrArg::buf(s));
        let r_out = descs.set_ptr(out_arr, k, DevicePtrArg::buf(d));
        assert_eq!(r_in, vec![s], "fold input table entry {k} escaped `srcs`");
        assert_eq!(r_out, vec![d], "fold output table entry {k} escaped `dsts`");
    }
    (in_ptrs, out_ptrs)
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
    /// `r_0` (`mod.rs:385`).
    ///
    /// `r_1 .. r_{n_max}` are **not** here any more: after P1 the steady
    /// rounds sample them into device buffers and never resolve them on the
    /// host (`ZerocheckPhaseProofIR::r`). `r_0` survives only because the
    /// round-0 launchers still take their challenge by value.
    pub r_0: EF,
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
    /// Column-opening claims observed at `mod.rs:415-428`, already in
    /// transcript order (common main first, then preprocessed/cached).
    pub opening_claims: Vec<EF>,
    /// `threads_per_block` for the batched launchers.
    pub threads_per_block: u32,
    /// `sm_count`-derived block count for the batched launchers.
    pub num_blocks: u32,
    /// `device.sm_count()` (`device.rs:63`) — an input to the monomial
    /// par-Y `chunk_size` auto-tune, which is a device property and therefore
    /// **not** keygen-static.
    pub sm_count: u32,
    /// `max_monomials_per_thread` as the eager caller passes it
    /// (`None` there means [`DEFAULT_MAX_MONOMIALS_PER_THREAD`]).
    pub max_monomials_per_thread: u32,
}

impl ZerocheckPhasePlan {
    pub fn num_traces(&self) -> usize {
        self.traces.len()
    }

    /// `s_deg = constraint_degree + 1` (`mod.rs:268`).
    pub fn s_deg(&self) -> usize {
        self.constraint_degree + 1
    }

    /// `num_y` for trace `t` at `round`, matching the eager `TraceCtx`.
    ///
    /// Early traces (`round <= n_lift`) get `1 << (n_lift - round)`
    /// (`mod.rs:1198-1199`); late traces (`round == n_lift + 1`) evaluate at
    /// `num_y = 1` (`mod.rs:1169`).
    pub fn round_num_y(&self, t: usize, round: usize) -> u32 {
        let n_lift = self.traces[t].n_lift();
        if round <= n_lift {
            1u32 << (n_lift - round)
        } else {
            1
        }
    }

    /// The par-Y monomial `chunk_size` the eager path auto-tunes for a batch.
    ///
    /// Line-for-line replica of `ZerocheckMonomialParYBatch::new`'s occupancy
    /// loop (`batch_mle_monomial.rs:344-372`): halve `chunk_size` from
    /// `max_monomials_per_thread` down until the batch reaches
    /// `sm_count * WAVES_TARGET` blocks, floored at 1.
    ///
    /// Reproducible at graph-build time because every input is either a plan
    /// shape or a device property — but `sm_count` *is* a device property, so
    /// the plan carrying it must be built per-prove, never cached per-key.
    pub fn monomial_chunk_size(&self, traces: &[usize], round: usize, num_x: u32) -> u32 {
        let per_air: Vec<(u32, u32)> = traces
            .iter()
            .map(|&t| {
                let y_blocks = self.round_num_y(t, round).div_ceil(THREADS_PER_BLOCK_PAR_Y);
                (y_blocks, self.traces[t].num_monomials as u32)
            })
            .collect();
        let target_blocks = self.sm_count * WAVES_TARGET;
        let mut chunk_size = self.max_monomials_per_thread.max(1);
        loop {
            let total_blocks: u32 = per_air
                .iter()
                .map(|&(y_blocks, num_mono)| y_blocks * num_mono.div_ceil(chunk_size))
                .sum();
            // `saturating_mul` where the eager path multiplies plainly: the
            // two agree for every reachable shape (an overflow needs
            // `total_blocks >= 2^32 / num_x`), and this cannot panic in a
            // debug build mid-graph-build.
            if total_blocks.saturating_mul(num_x) >= target_blocks || chunk_size <= 1 {
                break;
            }
            chunk_size = (chunk_size / 2).max(1);
        }
        chunk_size
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
// Graph inputs — the bytes the phase graph is fed (S1.1).
// ===========================================================================

/// Where one graph input's bytes come from at bind time.
#[derive(Clone, Debug)]
pub enum InputSource {
    /// Host bytes, staged to device by [`PhaseInputBinder::bind`].
    Host(Vec<u8>),
    /// Bytes that already live on the device — a keygen table on
    /// `DeviceMultiStarkProvingKey`, a trace matrix on `ProvingContext`.
    ///
    /// # Safety contract
    ///
    /// The allocation must be at least `len` bytes and must stay alive across
    /// the [`PhaseInputBinder::bind`] call. `bind` performs a D2D copy into
    /// the graph's pool slot and never retains the pointer.
    Device { ptr: *const u8, len: usize },
}

/// One registered graph input and the bytes that fill it.
#[derive(Clone, Debug)]
struct InputEntry {
    buf: BufId,
    name: String,
    /// `None` until a producer supplies the bytes. [`PhaseInputBinder::bind`]
    /// refuses to run with any entry still `None`.
    src: Option<InputSource>,
}

/// Every non-descriptor graph input of a zerocheck phase graph, paired with
/// its byte source.
///
/// # Why this exists
///
/// Before S1.1 these buffers were `insert_memset(_, 0)`: the graph built,
/// compiled and *ran*, computing on zeros. Registering them as inputs instead
/// makes a missing binding a hard error at `GraphExe::run`
/// (`graph_exe.rs:719-723`) rather than a silent all-zero prove.
///
/// # The trap this API is shaped around
///
/// `graph_ir.rs:1050-1053` claims inputs may not be written by any node and
/// that this is validated at compile time. **It is not.** The real check
/// (`graph_compiler.rs:1236-1266`) verifies existence, device, no double
/// registration and at-least-one-reader, and explicitly *permits* an input to
/// be written in place (`:1256-1260`). So a `register_input` left next to its
/// old `insert_memset` compiles clean and the memset node silently zeroes the
/// bound bytes at run time. Every registration below therefore *replaces* its
/// memset; none supplements it. The reverse mistake is caught loudly — a
/// buffer read but never written and not a registered input is rejected
/// (`graph_compiler.rs:1298-1304`) — which is why the safe edit order is
/// delete-then-register.
#[derive(Clone, Debug, Default)]
pub struct PhaseInputBinder {
    entries: Vec<InputEntry>,
}

impl PhaseInputBinder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `buf` as a graph input whose bytes are still unknown.
    pub fn register(&mut self, g: &mut GraphBuilder, buf: BufId, name: &str) -> BufId {
        g.register_input(buf);
        self.entries.push(InputEntry {
            buf,
            name: name.to_string(),
            src: None,
        });
        buf
    }

    /// Register `buf` and immediately supply host bytes for it — for the
    /// inputs that are a pure function of the plan (the selector cube).
    pub fn register_host<T: Copy>(
        &mut self,
        g: &mut GraphBuilder,
        buf: BufId,
        name: &str,
        xs: &[T],
    ) -> BufId {
        self.register(g, buf, name);
        self.set_host(buf, xs);
        buf
    }

    /// Supply host bytes for an already-registered input.
    ///
    /// # Panics
    ///
    /// If `buf` was never registered here.
    pub fn set_host<T: Copy>(&mut self, buf: BufId, xs: &[T]) {
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)).to_vec()
        };
        self.entry_mut(buf).src = Some(InputSource::Host(bytes));
    }

    /// Supply device-resident bytes for an already-registered input.
    ///
    /// # Safety
    ///
    /// See [`InputSource::Device`]: the allocation must outlive [`Self::bind`].
    pub unsafe fn set_device_raw(&mut self, buf: BufId, ptr: *const u8, len: usize) {
        self.entry_mut(buf).src = Some(InputSource::Device { ptr, len });
    }

    /// Supply the bytes of an existing [`DeviceBuffer`] for an input.
    pub fn set_device<T>(&mut self, buf: BufId, src: &DeviceBuffer<T>) {
        let len = src.len() * size_of::<T>();
        unsafe { self.set_device_raw(buf, src.as_ptr() as *const u8, len) }
    }

    /// Fill every still-unbound input with zeros.
    ///
    /// This is the **explicit** form of what `insert_memset` used to do
    /// implicitly. It exists for the shape tests and
    /// `examples/dump_ir_zerocheck_phase.rs`, which build and compile the
    /// graph without a proving key. Calling it makes the resulting run compute
    /// on zeros — deliberately, and at a named call site.
    pub fn zero_fill_unbound(&mut self, exe: &crypto_compiler::graph_exe::GraphExe) {
        for e in self.entries.iter_mut().filter(|e| e.src.is_none()) {
            let n = (0..exe.num_inputs())
                .find(|&i| exe.input_buf_id(i) == e.buf)
                .map(|i| exe.input_size(i))
                .unwrap_or(0);
            e.src = Some(InputSource::Host(vec![0u8; n]));
        }
    }

    /// Names of the inputs that still have no byte source.
    pub fn unbound(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| e.src.is_none())
            .map(|e| e.name.as_str())
            .collect()
    }

    /// `(BufId, name)` for every input registered here, in registration order.
    pub fn manifest(&self) -> Vec<(BufId, String)> {
        self.entries
            .iter()
            .map(|e| (e.buf, e.name.clone()))
            .collect()
    }

    /// Upload every input into its pool slot. Returns the number filled.
    ///
    /// Call **after** [`DescriptorPlan::install_pool`] and **before every**
    /// `GraphExe::run` — not once. Graph inputs are not preserved across an
    /// execution (`crates/compiler/notes.md:46-52`): ListV1 pins them through
    /// the schedule, but ListV2 — the shipped `SchedulerConfig` default — is
    /// free to reuse an input's slot once its last consumer has run
    /// (`planner/list_v2.rs:371-401`). This method is already idempotent and
    /// re-runnable; nothing in it is one-shot.
    ///
    /// Fails rather than silently zeroing if any input is unbound, and fails
    /// if an input's planned size disagrees with the bytes supplied.
    pub fn bind(
        &self,
        exe: &mut crypto_compiler::graph_exe::GraphExe,
        ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Result<usize, crypto_compiler::CompileError> {
        use openvm_cuda_common::copy::MemCopyH2D;

        let missing = self.unbound();
        if !missing.is_empty() {
            return Err(crypto_compiler::CompileError::Runtime(format!(
                "phase graph has {} unbound input(s): {}",
                missing.len(),
                missing.join(", ")
            )));
        }
        let mut filled = 0usize;
        for e in &self.entries {
            let i = (0..exe.num_inputs())
                .find(|&i| exe.input_buf_id(i) == e.buf)
                .ok_or_else(|| {
                    crypto_compiler::CompileError::Runtime(format!(
                        "input `{}` ({:?}) is not a registered graph input",
                        e.name, e.buf
                    ))
                })?;
            let need = exe.input_size(i);
            match e.src.as_ref().expect("checked above") {
                InputSource::Host(bytes) => {
                    if bytes.len() != need {
                        return Err(crypto_compiler::CompileError::Runtime(format!(
                            "input `{}` is {need} bytes in the plan, {} supplied",
                            e.name,
                            bytes.len()
                        )));
                    }
                    let staged = bytes.to_device_on(ctx).map_err(|err| {
                        crypto_compiler::CompileError::Runtime(format!(
                            "staging input `{}`: {err}",
                            e.name
                        ))
                    })?;
                    exe.set_input(ctx, i, &staged)?;
                }
                InputSource::Device { ptr, len } => {
                    if *len < need {
                        return Err(crypto_compiler::CompileError::Runtime(format!(
                            "input `{}` is {need} bytes in the plan, device source is {len}",
                            e.name
                        )));
                    }
                    // SAFETY: the caller's contract on `InputSource::Device`.
                    // `set_input` only reads through the view; `forget` keeps
                    // it from freeing memory it does not own.
                    let view = unsafe { DeviceBuffer::<u8>::from_raw_parts(*ptr as *mut u8, *len) };
                    let r = exe.set_input(ctx, i, &view);
                    forget(view);
                    r?;
                }
            }
            filled += 1;
        }
        Ok(filled)
    }

    fn entry_mut(&mut self, buf: BufId) -> &mut InputEntry {
        self.entries
            .iter_mut()
            .find(|e| e.buf == buf)
            .unwrap_or_else(|| panic!("{buf:?} is not a registered phase input"))
    }
}

/// The `3 * 2^n_lift` base-field selector cube for a trace, exactly as the
/// eager path builds it (`mod.rs:788-796`): `is_first` at row 0,
/// `is_transition` over `[height, 2*height - 1)`, `is_last` at the last row of
/// the third column.
///
/// This is a pure function of `n_lift`, so it is one of the few phase inputs
/// the graph builder can fill without a proving key.
pub fn selector_cube_host(n_lift: usize) -> Vec<F> {
    let height = 1usize << n_lift;
    let mut cols = vec![F::ZERO; 3 * height];
    cols[0] = F::ONE;
    for c in cols.iter_mut().take(2 * height - 1).skip(height) {
        *c = F::ONE;
    }
    cols[3 * height - 1] = F::ONE;
    cols
}

// ===========================================================================
// Per-trace device inputs.
// ===========================================================================

/// The `BufId`s of one trace's device-resident inputs.
///
/// [`TraceBufs::alloc_inputs`] allocates them and registers each as a graph
/// *input* (S1.1), recording it in a [`PhaseInputBinder`]. Building and
/// compiling the graph needs nothing more; **running** it needs every input
/// bound, which is a hard error at `GraphExe::run` if one is missed
/// (`graph_exe.rs:719-723`) rather than the silent all-zero prove the old
/// `insert_memset` produced.
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
    /// Allocate every buffer a trace needs and **register it as a graph
    /// input** (S1.1).
    ///
    /// Every buffer here used to be `insert_memset(_, 0)`, which made the
    /// graph runnable on zeros. They are now registered inputs recorded in
    /// `inputs`, so the bytes must be supplied before `run` — see
    /// [`PhaseInputBinder`] for the trap that shapes this API and for
    /// `zero_fill_unbound`, the explicit form of the old behaviour.
    ///
    /// Only [`Self::selectors_cube`] is filled here, because it is the one
    /// input that is a pure function of the plan ([`selector_cube_host`]).
    /// Everything else must be supplied by the caller with
    /// [`PhaseInputBinder::set_host`] / [`PhaseInputBinder::set_device`].
    ///
    /// A buffer is registered **only if some node will read it** — a
    /// registered input with no reader is rejected outright
    /// (`graph_compiler.rs:1236-1266`), and the eager path emits no logup
    /// launch for a trace without interactions (nor a constraint launch for
    /// one without constraints). Left unregistered such a buffer is simply
    /// unreferenced; if one turns out to be read after all, the compiler
    /// rejects it loudly (`graph_compiler.rs:1298-1304`) rather than running
    /// it as zeros.
    // TODO(cc-ir): the remaining per-trace inputs have no producer in this
    //   crate yet — the caller must `set_host` / `set_device` them.
    // WHY: the eager bytes are assembled inside `LogupZerocheckGpu`'s
    //   prove-time state (`mod.rs:462-527`), not on `pk` alone:
    //   `eq_3bs` / `numer_weights` / `denom_weights` are challenge-derived
    //   (`mod.rs:677-707`, `logup_combinations`), so a `pk`-only bridge
    //   cannot fill them.
    // RISK: a caller that forgets one now gets a hard error naming the buffer
    //   (`PhaseInputBinder::bind`), not a silent all-zero prove. That is the
    //   whole point of the change; the extraction itself is still to write.
    pub fn alloc_inputs(
        g: &mut GraphBuilder,
        device: DeviceType,
        plan: &ZerocheckPhasePlan,
        t: usize,
        inputs: &mut PhaseInputBinder,
    ) -> Self {
        let tp = &plan.traces[t];
        let n_lift = tp.n_lift();
        let cube = 1usize << n_lift;
        let num_x0 = (1usize << plan.l_skip).max(1);

        let selectors_cube = add_f_buf(g, device, &format!("t{t}_sels_cube"), 3 * cube);
        inputs.register_host(
            g,
            selectors_cube,
            &format!("t{t}_sels_cube"),
            &selector_cube_host(n_lift),
        );
        // Folded selectors are `3 * num_x` EFs (is_first / is_last /
        // is_transition). No memset: `fold_selectors_round0` is their producer,
        // and the graph is SSA — one writer per buffer.
        let selectors_folded = add_ef_buf(g, device, &format!("t{t}_sels_folded"), 3 * cube);

        // S1.2a: the round-0 evaluators now take a `MainMatrixDesc` array on
        // the base+offset ABI, not a bare `*const F` table. It is a
        // *descriptor input* — registered and filled by
        // [`DescriptorPlan::bind`] in the driver — so it must NOT be memset
        // here. (`graph_compiler.rs:1256-1260` permits an input to be written
        // in place, so a leftover memset would compile clean and silently
        // zero the descriptors.)
        let n_main = tp.mats.len() - usize::from(tp.has_preprocessed);
        let main_ptrs =
            add_typed_buf::<MainMatrixDesc>(g, device, &format!("t{t}_main_ptrs"), n_main);

        // Which evaluator families this trace reaches decides which of its
        // inputs are read at all. See the note on this function.
        let zc = tp.has_constraints;
        let lg = tp.has_interactions;

        let public_values = add_f_buf(
            g,
            device,
            &format!("t{t}_public"),
            tp.num_public_values.max(1),
        );
        if zc || lg {
            inputs.register(g, public_values, &format!("t{t}_public"));
        }

        let zc_rules =
            add_typed_buf::<u128>(g, device, &format!("t{t}_zc_rules"), tp.zc_rules_len.max(1));
        if zc {
            inputs.register(g, zc_rules, &format!("t{t}_zc_rules"));
        }
        let zc_used_nodes = add_typed_buf::<usize>(
            g,
            device,
            &format!("t{t}_zc_used_nodes"),
            tp.zc_used_nodes_len.max(1),
        );
        if zc {
            inputs.register(g, zc_used_nodes, &format!("t{t}_zc_used_nodes"));
        }
        let logup_rules = add_typed_buf::<u128>(
            g,
            device,
            &format!("t{t}_logup_rules"),
            tp.logup_rules_len.max(1),
        );
        if lg {
            inputs.register(g, logup_rules, &format!("t{t}_logup_rules"));
        }
        let logup_used_nodes = add_typed_buf::<usize>(
            g,
            device,
            &format!("t{t}_logup_used_nodes"),
            tp.logup_used_nodes_len.max(1),
        );
        if lg {
            inputs.register(g, logup_used_nodes, &format!("t{t}_logup_used_nodes"));
        }
        let logup_pair_idxs = add_typed_buf::<u32>(
            g,
            device,
            &format!("t{t}_logup_pair_idxs"),
            tp.logup_used_nodes_len.max(1),
        );
        if lg {
            inputs.register(g, logup_pair_idxs, &format!("t{t}_logup_pair_idxs"));
        }
        let eq_3bs = add_ef_buf(
            g,
            device,
            &format!("t{t}_eq_3bs"),
            tp.num_interactions.max(1),
        );
        if lg {
            inputs.register(g, eq_3bs, &format!("t{t}_eq_3bs"));
        }

        let numer_weights = add_ef_buf(
            g,
            device,
            &format!("t{t}_numer_w"),
            tp.num_interactions.max(1),
        );
        if lg {
            inputs.register(g, numer_weights, &format!("t{t}_numer_w"));
        }
        let denom_weights = add_ef_buf(
            g,
            device,
            &format!("t{t}_denom_w"),
            tp.num_interactions.max(1),
        );
        if lg {
            inputs.register(g, denom_weights, &format!("t{t}_denom_w"));
        }

        let preprocessed = tp.has_preprocessed.then(|| {
            let b = add_f_buf(
                g,
                device,
                &format!("t{t}_prep"),
                tp.mats[0].width * tp.mats[0].height,
            );
            if zc || lg {
                inputs.register(g, b, &format!("t{t}_prep"));
            }
            b
        });

        let mut mats = Vec::with_capacity(tp.mats.len());
        let mut folded_mats = Vec::with_capacity(tp.mats.len());
        for (i, m) in tp.mats.iter().enumerate() {
            let b = add_f_buf(g, device, &format!("t{t}_mat{i}"), m.width * m.height);
            inputs.register(g, b, &format!("t{t}_mat{i}"));
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
    /// The compact stage-D evaluator outputs, one entry per launch, for each
    /// of the `n_max` MLE rounds. Debug/inspection only — the ring reads them
    /// through the descriptor array, not from here.
    pub evaluator_outputs: Vec<Vec<BufId>>,
    /// `s_round(1) .. s_round(s_deg)` per steady round: `n_max` vectors of
    /// `constraint_degree + 1` device buffers. These are proof messages, and
    /// after P1 they are *computed* rather than supplied.
    pub sumcheck_round_polys: Vec<Vec<BufId>>,
    /// `r_0 .. r_{n_max}`, as device buffers.
    pub r: Vec<BufId>,
    /// Per-trace, per-matrix final folded buffers — the column openings
    /// before the host-side doubled-width split.
    pub column_openings: Vec<Vec<BufId>>,
    /// Final sponge state after the whole phase.
    pub transcript_state: BufId,
    /// Every base+offset descriptor array this phase emitted. The caller must
    /// call [`DescriptorPlan::bind`] on the compiled `GraphExe` before
    /// `run` — the arrays are registered graph inputs, so `run` refuses
    /// otherwise (`graph_exe.rs:719`).
    pub descriptors: DescriptorPlan,
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
    inputs: &mut PhaseInputBinder,
) -> ZerocheckPhaseProofIR
where
    TS: FiatShamirTranscriptGraphIR,
{
    assert_eq!(
        bufs.len(),
        plan.num_traces(),
        "one TraceBufs per trace required"
    );
    let num_traces = plan.num_traces();
    let l_skip = plan.l_skip;
    let sp_deg = plan.constraint_degree;
    let skip_domain = 1usize << l_skip;

    // Every base+offset descriptor array this phase needs. Filled in after
    // `compile()` by [`DescriptorPlan::bind`]; see the R6 section above.
    let mut descs = DescriptorPlan::new();

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
        // Keygen-static streams on `pk.per_air[..].other_data.zerocheck_monomials`;
        // registered as inputs (S1.1) so a missing bind is a hard error.
        let headers =
            add_typed_buf::<MonomialHeader>(g, device, &format!("t{t}_mono_hdr"), tp.num_monomials);
        inputs.register(g, headers, &format!("t{t}_mono_hdr"));
        let terms = add_typed_buf::<LambdaTerm<F>>(
            g,
            device,
            &format!("t{t}_lambda_terms"),
            tp.num_monomials,
        );
        inputs.register(g, terms, &format!("t{t}_lambda_terms"));
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
    // Registered only if something will read them. Unlike the old
    // `insert_memset`, a *registered input with no reader* is rejected
    // (`graph_compiler.rs:1236-1266`), so a plan whose traces have no
    // interactions must not register these. Left unregistered they are simply
    // unreferenced buffers; if one turns out to be read after all, the
    // compiler rejects it loudly (`graph_compiler.rs:1298-1304`) rather than
    // running on zeros — which is the direction we want to fail in.
    let any_logup_monomials = plan
        .traces
        .iter()
        .any(|tp| tp.has_interactions && tp.num_monomials > 0);
    let any_logup = plan.traces.iter().any(|tp| tp.has_interactions);

    let beta_pows = add_ef_buf(g, device, "beta_pows", plan.lambda_pows.len().max(1));
    if any_logup_monomials {
        inputs.register(g, beta_pows, "beta_pows");
    }
    // `LogupCtx::d_challenges` — the interaction challenge vector the DAG
    // interpreter reads through `ENTRY_CHALLENGE` (`batch_mle.cu:224`).
    // Registered as one graph input (S1.1). It is a *shared* buffer across
    // traces, so it must stay one `BufId` — every `LogupCtx` points at the
    // same allocation. The eager builder receives it as a raw `*const EF`
    // from the caller (`batch_mle.rs:271`).
    let logup_challenges = add_ef_buf(g, device, "logup_challenges", plan.lambda_pows.len().max(1));
    if any_logup {
        inputs.register(g, logup_challenges, "logup_challenges");
    }
    let mut logup_combinations: Vec<Option<(BufId, BufId)>> = vec![None; num_traces];
    for (t, tp) in plan.traces.iter().enumerate() {
        if !tp.has_interactions || tp.num_monomials == 0 {
            continue;
        }
        let headers =
            add_typed_buf::<MonomialHeader>(g, device, &format!("t{t}_ia_hdr"), tp.num_monomials);
        inputs.register(g, headers, &format!("t{t}_ia_hdr"));
        let terms = add_typed_buf::<InteractionMonomialTerm<F>>(
            g,
            device,
            &format!("t{t}_ia_terms"),
            tp.num_monomials,
        );
        inputs.register(g, terms, &format!("t{t}_ia_terms"));
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
    for t in 0..num_traces {
        let tb = &bufs[t];
        // One descriptor array per trace, shared by both round-0 evaluators
        // (they read the same matrices).
        let main_reads = emit_round0_main_descs(g, &mut descs, plan, bufs, t);
        let tp = &plan.traces[t];
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
                    pool_base: descs.pool_base().clone(),
                    eq_cube,
                    lambda_pows,
                    public_values: tb.public_values,
                    rules: tb.zc_rules,
                    used_nodes: tb.zc_used_nodes,
                    main_reads: main_reads.clone(),
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
                    pool_base: descs.pool_base().clone(),
                    eq_cube,
                    public_values: tb.public_values,
                    numer_weights: tb.numer_weights,
                    denom_weights: tb.denom_weights,
                    rules: tb.logup_rules,
                    main_reads: main_reads.clone(),
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
    //
    // The *value* still comes from the plan (the round-0 launchers below take
    // their challenge by value), but the sampled buffer is retained: it is the
    // first `r_prev` the ring consumes, and the first entry of the proof's
    // challenge list.
    let r0_buf = transcript.sample_ext(g);
    let r_0 = plan.r_0;

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
                FoldPleDst::Fresh(tb.folded_mats[i]),
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
            if tp.need_rot {
                // The second launch writes the upper half of the *same*
                // doubled-width buffer (`fold_ple.rs:38, 50`). It keeps the
                // first launch's `BufId` and declares it a carried mutation —
                // never a fresh aliased id, which the memory scheduler would
                // place at a different pool offset. See [`FoldPleDst`].
                fold_ple_from_evals_ir(
                    g,
                    tb.mats[i],
                    m.width * m.height,
                    FoldPleDst::InPlace(tb.folded_mats[i]),
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
            }
            // One allocation holds both halves, so the freshest name after
            // round 0 is the same id the plain fold produced.
            folded.push(tb.folded_mats[i]);
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
    let mut evaluator_outputs: Vec<Vec<BufId>> = Vec::with_capacity(plan.n_max);
    let mut sumcheck_round_polys: Vec<Vec<BufId>> = Vec::with_capacity(plan.n_max);
    let mut r_bufs: Vec<BufId> = Vec::with_capacity(plan.n_max + 1);
    g.register_output(r0_buf);
    r_bufs.push(r0_buf);
    // Current per-trace folded buffers; rebound each round by the fold.
    let mut cur_mats: Vec<Vec<BufId>> = folded_after_r0;
    let mut cur_sels: Vec<BufId> = bufs.iter().map(|b| b.selectors_folded).collect();

    // ---- ring state, initialized from the eager round-0 state.
    //
    // `mu_pows` and `norm_factors` are constant for the whole phase;
    // `norm_factors[t] = F::from_usize(1 << max(-n, 0)).inverse()`
    // (`mod.rs:1144-1146`), i.e. `F::ONE` for every trace at least as tall as
    // the skip domain.
    let mu_pows_buf = ef_slice_const_buf(g, device, "mu_pows", &plan.mu_pows);
    let norm_factors_buf = f_slice_const_buf(g, device, "norm_factors", &{
        plan.traces
            .iter()
            .map(|tp| F::from_usize(1usize << (-tp.n).max(0)).inverse())
            .collect::<Vec<F>>()
    });
    // `tilde[3 * T]` starts at zero exactly as `zerocheck_tilde_evals` /
    // `logup_tilde_evals` do (`mod.rs:628-629`). A memset is a genuine
    // in-place write, so it keeps its own `BufId` (`notes.md:37-44`).
    let tilde0 = add_ef_buf(g, device, "r0_tilde", 3 * num_traces);
    g.insert_memset(tilde0, 0);
    // `[prev_s_eval, eq_n, eq_sharp_n]` after round 0 (`mod.rs:388`,
    // `mod.rs:1083-1088`).
    let scalars0 = ef_slice_const_buf(
        g,
        device,
        "r0_ring_scalars",
        &[
            horner_eval::<EF, EF, EF>(&plan.s_0_coeffs, r_0),
            eval_eq_uni(l_skip, plan.xi[0], r_0),
            eval_eq_sharp_uni(&plan.omega_skip_pows, &plan.xi[..l_skip], r_0),
        ],
    );
    let mut ring_state = ZerocheckRoundStateIr {
        tilde: tilde0,
        scalars: scalars0,
    };
    let mut r_prev = r0_buf;

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
            let mats: Vec<(BufId, usize)> = (0..tp.mats.len())
                .map(|i| (cur_mats[t][i], tp.folded_width(i)))
                .collect();
            let (columns, col_reads) = emit_column_table(
                g,
                &mut descs,
                device,
                &format!("t{t}_r{round}_cols"),
                cur_sels[t],
                &mats,
                num_y,
            );
            let interpolated = add_ef_buf(
                g,
                device,
                &format!("t{t}_r{round}_interp"),
                sp_deg * num_y * num_columns,
            );
            // Selectors first, matrices after — same order as the table, so
            // the declared reads and the dereference closure agree.
            let mut srcs = vec![cur_sels[t]];
            srcs.extend_from_slice(&cur_mats[t]);
            for b in &col_reads {
                assert!(
                    srcs.contains(b),
                    "column table entry points at {b:?}, which the node does not declare"
                );
            }
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
        //
        // TODO(cc-ir,S1.4-a): late traces must be launched SEPARATELY at
        //   `num_x = 1`, not merged into the early batch at
        //   `num_x = constraint_degree`.
        // WHY: `early` and `late` are concatenated below and the whole batch
        //   goes out at `num_x = plan.constraint_degree`
        //   (`emit_zerocheck_round_eval`, `emit_logup_round_eval`). The eager
        //   path evaluates a late trace (`round == n_lift + 1`) in its own
        //   launch at `num_x = 1` — `LogupMonomialBatch..evaluate(1)`
        //   (`mod.rs:1309-1311`) and `ZerocheckMonomialBatch..evaluate(1)`
        //   (`mod.rs:1333-1335`) — because a late trace contributes ONE
        //   scalar to `tilde`, not `constraint_degree` head evaluations.
        // RISK: WRONG ANSWER, not just wrong performance, for every round in
        //   which some trace is late (i.e. every round past the shortest
        //   trace's `n_lift`). The fix is a second launch per family with
        //   `num_x = 1` and its own `BatchEvalShape`.
        // STATUS (P1): no longer silent. `build_batch_s_ring_trace_descs`
        //   asserts that a late trace's batch has `num_x == 1`, so a
        //   mixed-height plan now PANICS at graph-build time instead of
        //   emitting a graph that computes the wrong polynomial. Nothing in
        //   production calls this driver yet, and every fixture is
        //   equal-height, so the assertion is currently unreachable in tests
        //   — it exists so that S1.4-a cannot be forgotten.
        //
        // RESOLVED (P1): `norm_factor` now exists, in ONE place — the ring's
        // `pre` kernel scales the interaction numerator by
        // `norm_factors[t] = F::from_usize(1 << max(-n, 0)).inverse()`
        // (`mod.rs:1144-1146`) as it reads the raw evaluator output. Keeping
        // the graph evaluators' output raw is what makes "exactly once"
        // checkable: `batch_s_ring_pre_post_matches_eager` covers an `n < 0`
        // trace and fails if the factor is dropped or applied twice.
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

        // Per-trace stage-D device inputs, as `BufId`s. This is what the R6
        // descriptor writers consume: every pointer that used to be baked into
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

        // One `MainMatrixDesc` array per (round, trace), shared across the
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
                &mut descs,
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
                &mut descs,
            )
        });
        let batches: Vec<RoundEvalBatchIr> = zc_eval.into_iter().chain(lg_eval).collect();
        evaluator_outputs.push(batches.iter().map(|b| b.evals).collect());

        // D.3 — THE RING (was seam 2).
        //
        // `compute_batch_s_poly` (`mod.rs:1448-1520`), the `s_deg` observes,
        // `sample_ext`, and the running-scalar updates, all as graph nodes.
        // Nothing between the evaluator outputs and the fold below touches the
        // host any more: `s_round(1..=s_deg)` is observed from device buffers,
        // `r_round` is sampled into one, and the same buffer drives both folds.
        //
        // `xi_j` is still a const producer — `xi` comes back from fractional
        // GKR on the host (stage B, out of scope) — but the ring ABI takes it
        // as a pointer so closing that stage later is a producer swap, not a
        // signature change.
        let xi_j = ef_const_ext_scalar_buf(
            g,
            device,
            &format!("r{round}_xi"),
            plan.xi[l_skip + round - 1],
        );
        let ring = observe_and_update_zerocheck_round_ir(
            g,
            transcript,
            &mut descs,
            plan,
            &batches,
            mu_pows_buf,
            norm_factors_buf,
            xi_j,
            r_prev,
            ring_state,
            round,
            device,
        );
        // Proof artifacts: the caller reads them out of the compiled exe, so
        // they must survive DCE and the memory planner's liveness analysis.
        for &b in &ring.s_evals {
            g.register_output(b);
        }
        g.register_output(ring.r_round);
        sumcheck_round_polys.push(ring.s_evals.clone());
        r_bufs.push(ring.r_round);
        let r_round = ring.r_round;

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
            let mut max_cells = 0u32;
            let mut new_mats: Vec<(usize, usize, BufId)> = Vec::with_capacity(n);
            let mut srcs: Vec<BufId> = Vec::with_capacity(n);
            let mut dsts: Vec<BufId> = Vec::with_capacity(n);
            let mut widths_host: Vec<u32> = Vec::with_capacity(n);
            let mut logh_host: Vec<u8> = Vec::with_capacity(n);
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
                // `(width, log2(output_height))` exactly as the eager
                // `multiunzip` builds them (`mod.rs:1520-1533`).
                widths_host.push(w as u32);
                logh_host.push(out_h.ilog2() as u8);
            }
            let (in_ptrs, out_ptrs) = fold_ptr_tables(
                g,
                &mut descs,
                device,
                &format!("r{round}_fold"),
                &srcs,
                &dsts,
            );
            let widths =
                typed_slice_const_buf(g, device, &format!("r{round}_fold_w"), &widths_host);
            let logh = typed_slice_const_buf(g, device, &format!("r{round}_fold_logh"), &logh_host);
            batch_fold_mle_ir(
                g, in_ptrs, out_ptrs, widths, logh, &srcs, &dsts, n as u16, max_cells, r_round,
            );
            for (t, i, nb) in new_mats {
                cur_mats[t][i] = nb;
            }
        }
        // Selector fold (`mod.rs:1585`).
        //
        // Only traces whose selector cube still has `height > 1` are folded:
        // the eager `batch_fold` takes `partition_point(|m| m.height() > 1)`
        // and passes the rest through untouched (`mod.rs:1519, 1562`). Here
        // that predicate is `round <= n_lift`. Folding an exhausted trace
        // would read `input[1]` out of a height-1 buffer.
        let sfold_traces: Vec<usize> = (0..plan.traces.len())
            .filter(|&t| round <= plan.traces[t].n_lift())
            .collect();
        if !sfold_traces.is_empty() {
            let n = sfold_traces.len();
            let mut max_cells = 0u32;
            let mut new_sels = Vec::with_capacity(n);
            let mut srcs: Vec<BufId> = Vec::with_capacity(n);
            let mut dsts: Vec<BufId> = Vec::with_capacity(n);
            let mut widths_host: Vec<u32> = Vec::with_capacity(n);
            let mut logh_host: Vec<u8> = Vec::with_capacity(n);
            for &t in &sfold_traces {
                let out_h = 1usize << (plan.traces[t].n_lift() - round);
                max_cells = max_cells.max((out_h * 3) as u32);
                let nb = add_ef_buf(g, device, &format!("t{t}_sels_r{round}"), out_h * 3);
                srcs.push(cur_sels[t]);
                dsts.push(nb);
                new_sels.push((t, nb));
                widths_host.push(3);
                logh_host.push(out_h.ilog2() as u8);
            }
            let (in_ptrs, out_ptrs) = fold_ptr_tables(
                g,
                &mut descs,
                device,
                &format!("r{round}_sfold"),
                &srcs,
                &dsts,
            );
            let widths =
                typed_slice_const_buf(g, device, &format!("r{round}_sfold_w"), &widths_host);
            let logh =
                typed_slice_const_buf(g, device, &format!("r{round}_sfold_logh"), &logh_host);
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
        // The ring's state is a value, not a slot: the next round reads the
        // buffers this round produced.
        ring_state = ring.state;
        r_prev = ring.r_round;
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
        evaluator_outputs,
        sumcheck_round_polys,
        r: r_bufs,
        column_openings,
        transcript_state,
        descriptors: descs,
    }
}

/// Which ctx struct a batched evaluator's per-AIR array holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CtxKind {
    /// `ZerocheckCtx` — a base+offset descriptor array ([`DescriptorPlan`]).
    Zerocheck,
    /// `LogupCtx` — a base+offset descriptor array ([`DescriptorPlan`]).
    Logup,
    /// `MonomialAirCtx` — still a zeroed placeholder, see the TODO on
    /// [`LogupMonomialBufs`].
    Monomial,
}

/// One trace's stage-D device inputs at a given round, as `BufId`s.
///
/// This is what the R6 descriptor writers turn into one device-resident ctx
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

/// Emit the per-trace `MainMatrixDesc` array as a graph input.
///
/// The eager path builds this array on the host out of absolute addresses
/// (`mod.rs:1131-1149, 1224-1256`); here each element stores a pool byte
/// offset instead, so the array holds integers and the planner never sees an
/// embedded pointer.
///
/// Returns `(array, reads)`. `reads` is the union of the elements' declared
/// reads — the folded main matrices the array points at. The array itself is
/// *not* in `reads`: it reaches the evaluator through the ctx descriptor,
/// which stores it as `EvalCoreCtxArgs::d_main`.
fn emit_main_matrix_ptrs(
    g: &mut GraphBuilder,
    device: DeviceType,
    descs: &mut DescriptorPlan,
    tag: &str,
    rb: &RoundTraceBufs,
) -> (BufId, Vec<BufId>) {
    let name = format!("{tag}_main_desc");
    let len = rb.mains.len().max(1);
    let arr = add_typed_buf::<MainMatrixDesc>(g, device, &name, len);
    let id = descs.add_array::<MainMatrixDesc>(g, arr, &name, len);
    let mut reads = Vec::with_capacity(rb.mains.len());
    for (i, &(buf, air_width)) in rb.mains.iter().enumerate() {
        extend_reads(
            &mut reads,
            descs.set_main_matrix_desc(id, i, DevicePtrArg::buf(buf), air_width),
        );
    }
    (arr, reads)
}

/// Fill a trace's round-0 `MainMatrixDesc` array (S1.2a).
///
/// The eager table is the AIR's `[cached…, common_main]` in order, skipping
/// the preprocessed matrix (`mod.rs:826-832`); `TracePlan::mats` is
/// `preprocessed?, cached*, common_main`, so the descriptors start at
/// `has_preprocessed as usize`.
///
/// `air_width` is written as **0**, matching `MainMatrixDesc::round0`
/// (`cuda/logup_zerocheck.rs:96-111`): round 0 addresses a main matrix
/// column-major with stride `height` (`dag_entry.cuh`, `ENTRY_MAIN`) and never
/// reads the field, and the batched evaluators' `air_width` means the *padded*
/// AIR width, which is not what `TracePlan::mats[i].width` holds. Writing the
/// plan width here would agree with nothing and would only look correct.
fn emit_round0_main_descs(
    g: &mut GraphBuilder,
    descs: &mut DescriptorPlan,
    plan: &ZerocheckPhasePlan,
    bufs: &[TraceBufs],
    t: usize,
) -> Vec<BufId> {
    let tp = &plan.traces[t];
    let first_main = usize::from(tp.has_preprocessed);
    let n_main = tp.mats.len() - first_main;
    let name = format!("t{t}_main_ptrs");
    let id = descs.add_array::<MainMatrixDesc>(g, bufs[t].main_ptrs, &name, n_main.max(1));
    let mut reads = Vec::with_capacity(n_main);
    for i in 0..n_main {
        let m = first_main + i;
        extend_reads(
            &mut reads,
            descs.set_main_matrix_desc(id, i, DevicePtrArg::buf(bufs[t].mats[m]), 0),
        );
    }
    reads
}

/// One `MainMatrixDesc` array per `(round, trace)`, shared by
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
        descs: &mut DescriptorPlan,
        round: usize,
        trace: usize,
        rb: &RoundTraceBufs,
    ) -> (BufId, Vec<BufId>) {
        if let Some(hit) = self.entries.get(&trace) {
            return hit.clone();
        }
        let entry = emit_main_matrix_ptrs(g, device, descs, &format!("r{round}_t{trace}"), rb);
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
    // The evaluator declares this scratch as a *written* input, and the graph
    // requires a writer to precede every reader in insertion order, so the
    // scratch gets an explicit zero producer.
    // TODO(cc-ir,R6): one extra memset per AIR per round purely to give the
    //   scratch a producer.
    // WHY: `insert_blackbox_kernel` declares whole-buffer reads/writes; there
    //   is no "this buffer is pure output scratch" declaration.
    // RISK: performance only (a tiny memset), never correctness — the DAG
    //   interpreter writes every intermediate slot before reading it. Note
    //   R6 removed the *other* reason this memset existed (the materializer's
    //   address-only read), so it is now a pure planner artefact.
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
    descs: &mut DescriptorPlan,
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
    let reads = descs.set_zerocheck_ctx(
        bufs.ctx_array,
        air as usize,
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
    descs: &mut DescriptorPlan,
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
    let reads = descs.set_logup_ctx(
        bufs.ctx_array,
        air as usize,
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
/// still zeroed placeholders (R6 converts only the three DAG ctx structs).
///
/// The reads are declared anyway: an under-declared access set is a real
/// correctness bug the moment the ctx arrays are filled, and declaring them
/// now costs nothing but a few planner edges.
// TODO(cc-ir,R4): this is the last *hand-written* read list in the phase.
//   The three DAG ctx structs derive theirs from the descriptor writer that
//   encodes them ([`OffSink`]), so they cannot drift; this one can.
// WHY: there is nothing to derive it from. `MonomialAirCtx`,
//   `LogupMonomialCommonCtx` and `LogupMonomialCtx` have no descriptor
//   writers yet — their arrays are `memset` to zero (`alloc_eval_bufs`), so
//   no [`ReadCollector`] run exists whose reads could be returned.
// RISK: `evaluator_declares_every_referenced_buffer` cannot police this
//   list — its closure over a monomial ctx array is the bare array, so the
//   assertion is satisfied vacuously for those nodes. A monomial ctx field
//   added without a matching line here is silent, exactly as the DAG structs
//   were before R4. Closing it is DA step 5.3 (write the monomial
//   descriptor writers); the assertion then starts policing them for free.
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

/// The ring state carried from one steady MLE round to the next.
///
/// Both members are *values*, never host scalars: `tilde` is the `[3 * T]`
/// vector of per-trace exhausted/late evaluations in the `mu_pows` slot order
/// (`p_t = 2t`, `q_t = 2t + 1`, `zc_t = 2T + t`), and `scalars` is
/// `[prev_s_eval, eq_n, eq_sharp_n]`.
#[derive(Clone, Copy, Debug)]
pub struct ZerocheckRoundStateIr {
    /// `[3 * num_traces] EF`.
    pub tilde: BufId,
    /// `[3] EF` — `prev_s_eval`, `eq_ns[round - 1]`, `eq_sharp_ns[round - 1]`.
    pub scalars: BufId,
}

/// What one steady round's ring leaves on the graph.
#[derive(Clone, Debug)]
pub struct ZerocheckRoundRingOut {
    /// The challenge this round sampled — feeds both folds, and the next
    /// round's `r_prev`.
    pub r_round: BufId,
    /// `s_round(1) .. s_round(s_deg)`, one `[D_EF]` buffer each, in the order
    /// they were observed.
    pub s_evals: Vec<BufId>,
    /// `[constraint_degree + 2] EF` — the round polynomial in coefficient
    /// form. Not part of the proof (the verifier gets the evaluations); kept
    /// because it is what `post` consumes and what the byte-equality oracle
    /// compares against `compute_batch_s_poly`'s output.
    pub poly_coeffs: BufId,
    /// State for the next round.
    pub state: ZerocheckRoundStateIr,
}

/// Insert the ring's `pre` node: the whole of `compute_batch_s_poly`
/// (`mod.rs:1448-1520`) plus the Horner evaluations at `1..=s_deg`.
///
/// Returns `(tilde_out, poly_coeffs, s_evals_contiguous)`, all three fresh —
/// the round's tilde vector and scalar state are mathematically new values,
/// not an in-place edit of the previous round's (`notes.md:37-44` is about
/// *genuine* mutations, which these are not).
///
/// `trace_desc_reads` is appended verbatim to the explicit inputs: the kernel
/// reaches this round's evaluator outputs only through offsets stored in
/// `trace_descs`, which the planner cannot see through.
#[allow(clippy::too_many_arguments)]
fn batch_s_ring_pre_ir(
    g: &mut GraphBuilder,
    trace_descs: BufId,
    trace_desc_reads: &[BufId],
    pool_base: PoolBase,
    tilde_in: BufId,
    mu_pows: BufId,
    norm_factors: BufId,
    scalar_state_in: BufId,
    xi_j: BufId,
    r_prev: BufId,
    num_traces: usize,
    constraint_degree: usize,
    round: usize,
    device: DeviceType,
) -> (BufId, BufId, BufId) {
    let tilde_out = add_ef_buf(g, device, &format!("r{round}_tilde"), 3 * num_traces);
    let poly_coeffs = add_ef_buf(
        g,
        device,
        &format!("r{round}_s_coeffs"),
        constraint_degree + 2,
    );
    let s_evals = add_ef_buf(
        g,
        device,
        &format!("r{round}_s_evals"),
        constraint_degree + 1,
    );

    let mut inputs = vec![
        trace_descs,
        tilde_in,
        mu_pows,
        norm_factors,
        scalar_state_in,
        xi_j,
        r_prev,
    ];
    inputs.extend_from_slice(trace_desc_reads);
    let modifies: Vec<bool> = inputs.iter().map(|_| false).collect();
    let (t, d, r) = (num_traces as u32, constraint_degree as u32, round as u32);
    g.insert_blackbox_kernel(
        "batch_s_ring_pre",
        inputs.into_iter(),
        [tilde_out, poly_coeffs, s_evals].into_iter(),
        modifies.into_iter(),
        move |inputs, outputs, stream| unsafe {
            batch_s_ring_pre(
                inputs[0] as *const BatchSRingTraceDesc,
                pool_base.get(),
                inputs[1] as *const EF,
                inputs[2] as *const EF,
                inputs[3] as *const F,
                inputs[4] as *const EF,
                inputs[5] as *const EF,
                inputs[6] as *const EF,
                outputs[0] as *mut EF,
                outputs[1] as *mut EF,
                outputs[2] as *mut EF,
                t,
                d,
                r,
                stream,
            )
            .expect("batch_s_ring_pre");
        },
    );
    (tilde_out, poly_coeffs, s_evals)
}

/// Insert the ring's `post` node: `s(r_round)` and the two running
/// equality-product updates (`mod.rs:401`, `:1612-1614`).
fn batch_s_ring_post_ir(
    g: &mut GraphBuilder,
    poly_coeffs: BufId,
    scalar_state_in: BufId,
    xi_j: BufId,
    r_round: BufId,
    constraint_degree: usize,
    device: DeviceType,
) -> BufId {
    let scalar_state_out = add_ef_buf(g, device, "ring_scalars", 3);
    let d = constraint_degree as u32;
    g.insert_blackbox_kernel(
        "batch_s_ring_post",
        [poly_coeffs, scalar_state_in, xi_j, r_round].into_iter(),
        std::iter::once(scalar_state_out),
        [false, false, false, false].into_iter(),
        move |inputs, outputs, stream| unsafe {
            batch_s_ring_post(
                inputs[0] as *const EF,
                inputs[1] as *const EF,
                inputs[2] as *const EF,
                inputs[3] as *const EF,
                outputs[0] as *mut EF,
                d,
                stream,
            )
            .expect("batch_s_ring_post");
        },
    );
    scalar_state_out
}

/// One steady MLE round's compute -> observe -> sample -> update block, as
/// graph nodes.
///
/// The graph-IR mirror of `mod.rs:393-402`: `compute_batch_s_poly`, the
/// `s_deg` `observe_ext` calls, `sample_ext`, and the `prev_s_eval` /
/// `eq_ns` / `eq_sharp_ns` updates that `fold_mle_evals` performs
/// (`mod.rs:1610-1615`).
///
/// Shaped exactly like `fractional_ir::observe_and_update_ir`: the
/// transcript's *control* state stays on the host (it picks which sponge
/// module each observe/sample emits) while every *value* stays a `BufId`.
/// Deliberately does not fold — the driver owns the two shape-specific fold
/// plans and this helper has no business rebuilding them.
#[allow(clippy::too_many_arguments)]
fn observe_and_update_zerocheck_round_ir<TS: FiatShamirTranscriptGraphIR>(
    g: &mut GraphBuilder,
    transcript: &mut TS,
    descs: &mut DescriptorPlan,
    plan: &ZerocheckPhasePlan,
    batches: &[RoundEvalBatchIr],
    mu_pows: BufId,
    norm_factors: BufId,
    xi_j: BufId,
    r_prev: BufId,
    state: ZerocheckRoundStateIr,
    round: usize,
    device: DeviceType,
) -> ZerocheckRoundRingOut {
    let s_deg = plan.s_deg();
    let pool_base = descs.pool_base().clone();
    let (trace_descs, trace_desc_reads) = build_batch_s_ring_trace_descs(
        g,
        descs,
        device,
        &format!("r{round}_ring_descs"),
        plan,
        round,
        batches,
    );
    let (tilde_out, poly_coeffs, s_evals_contig) = batch_s_ring_pre_ir(
        g,
        trace_descs,
        &trace_desc_reads,
        pool_base,
        state.tilde,
        mu_pows,
        norm_factors,
        state.scalars,
        xi_j,
        r_prev,
        plan.num_traces(),
        plan.constraint_degree,
        round,
        device,
    );

    // The transcript absorbs one `[D_EF]` buffer per point, in increasing
    // point order, so the contiguous `[s_deg] EF` output is split by explicit
    // 16-byte range copies rather than observed as a block.
    let s_evals: Vec<BufId> = (0..s_deg)
        .map(|i| {
            let b = add_ext_scalar_buf(g, device, &format!("r{round}_s{i}"));
            g.insert_memcpy_range(
                s_evals_contig,
                Quast::cst((i * EF_BYTES) as i64),
                b,
                Quast::cst(0),
                Quast::cst(EF_BYTES as i64),
            );
            b
        })
        .collect();
    for &b in &s_evals {
        transcript.observe_ext(g, b);
    }
    let r_round = transcript.sample_ext(g);
    let scalars = batch_s_ring_post_ir(
        g,
        poly_coeffs,
        state.scalars,
        xi_j,
        r_round,
        plan.constraint_degree,
        device,
    );

    ZerocheckRoundRingOut {
        r_round,
        s_evals,
        poly_coeffs,
        state: ZerocheckRoundStateIr {
            tilde: tilde_out,
            scalars,
        },
    }
}

/// Which family produced one compact evaluator batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RoundEvalFamily {
    Zerocheck,
    Logup,
}

/// One stage-D evaluator launch, as the ring needs to see it.
///
/// The emitters used to return a bare `BufId`, which loses the two facts the
/// ring cannot recover: which traces are inside the batch and in what order
/// (the kernel writes `out[air * num_x + i]`, `air` being the position in
/// `traces`), and how many values per trace there are. Carrying them as a
/// typed value is what lets the descriptor builder compute byte offsets that
/// cannot silently drift when the dispatch grows more batches (S1.4).
#[derive(Clone, Debug)]
struct RoundEvalBatchIr {
    family: RoundEvalFamily,
    /// The compact `[air][num_x]` output buffer: `EF` for
    /// [`RoundEvalFamily::Zerocheck`], `Frac<EF>` for
    /// [`RoundEvalFamily::Logup`].
    evals: BufId,
    /// Trace indices, in the batch's `air` order.
    traces: Vec<usize>,
    num_x: usize,
}

/// Build this round's `BatchSRingTraceDesc[num_traces]` array.
///
/// Returns `(array buffer, the graph buffers its offsets point into)`. The
/// second half is not decoration: the ring kernel dereferences those buffers
/// through the descriptor, and a pointer the consuming node does not declare
/// lets the planner hand that pool slot to someone else while the kernel still
/// needs it. Deriving it from the same writer that produces the bytes is what
/// makes an under-declared set unrepresentable — see [`OffSink`].
fn build_batch_s_ring_trace_descs(
    g: &mut GraphBuilder,
    descs: &mut DescriptorPlan,
    device: DeviceType,
    name: &str,
    plan: &ZerocheckPhasePlan,
    round: usize,
    batches: &[RoundEvalBatchIr],
) -> (BufId, Vec<BufId>) {
    let num_traces = plan.num_traces();
    let buf = add_typed_buf::<BatchSRingTraceDesc>(g, device, name, num_traces);
    let arr = descs.add_batch_s_ring_trace_array(g, buf, name, num_traces);

    let mut args: Vec<BatchSRingTraceDescArgs> = plan
        .traces
        .iter()
        .map(|tp| BatchSRingTraceDescArgs {
            zc_evals: DevicePtrArg::NULL,
            logup_evals: DevicePtrArg::NULL,
            n_lift: tp.n_lift() as u32,
            flags: if tp.has_constraints {
                BATCH_S_RING_HAS_CONSTRAINTS
            } else {
                0
            } | if tp.has_interactions {
                BATCH_S_RING_HAS_INTERACTIONS
            } else {
                0
            },
        })
        .collect();

    for b in batches {
        for (air, &t) in b.traces.iter().enumerate() {
            let tp = &plan.traces[t];
            let n_lift = tp.n_lift();
            // The eager shapes: an early trace yields `constraint_degree` head
            // values, a late one yields exactly one (`mod.rs:1309-1311`,
            // `mod.rs:1333-1335`). A late trace merged into an early batch is
            // the known S1.4-a defect; refuse it here rather than silently
            // reading `num_x` values where one exists.
            if round <= n_lift {
                assert_eq!(
                    b.num_x, plan.constraint_degree,
                    "round {round}: early trace {t} evaluated at num_x = {} instead of the \
                     constraint degree {}",
                    b.num_x, plan.constraint_degree
                );
            } else if round == n_lift + 1 {
                assert_eq!(
                    b.num_x, 1,
                    "round {round}: late trace {t} (n_lift = {n_lift}) evaluated at num_x = {} \
                     instead of 1 — late traces need their own launch (S1.4-a)",
                    b.num_x
                );
            } else {
                panic!(
                    "round {round}: exhausted trace {t} (n_lift = {n_lift}) must not be in an \
                     evaluator batch at all"
                );
            }
            match b.family {
                RoundEvalFamily::Zerocheck => {
                    assert!(
                        tp.has_constraints,
                        "round {round}: trace {t} has no constraints but appears in a \
                         zerocheck batch"
                    );
                    assert!(
                        matches!(args[t].zc_evals, DevicePtrArg::Static(0)),
                        "round {round}: trace {t} appears in two zerocheck batches"
                    );
                    args[t].zc_evals = DevicePtrArg::at(b.evals, air * b.num_x * size_of::<EF>());
                }
                RoundEvalFamily::Logup => {
                    assert!(
                        tp.has_interactions,
                        "round {round}: trace {t} has no interactions but appears in a \
                         logup batch"
                    );
                    assert!(
                        matches!(args[t].logup_evals, DevicePtrArg::Static(0)),
                        "round {round}: trace {t} appears in two logup batches"
                    );
                    args[t].logup_evals =
                        DevicePtrArg::at(b.evals, air * b.num_x * size_of::<Frac<EF>>());
                }
            }
        }
    }

    // Every trace that is still evaluated this round must have a pointer for
    // each family it enables; an exhausted or disabled family keeps its null.
    for (t, tp) in plan.traces.iter().enumerate() {
        if round > tp.n_lift() + 1 {
            continue;
        }
        if tp.has_constraints {
            assert!(
                !matches!(args[t].zc_evals, DevicePtrArg::Static(0)),
                "round {round}: trace {t} has constraints and is not exhausted, but no \
                 zerocheck batch claimed it"
            );
        }
        if tp.has_interactions {
            assert!(
                !matches!(args[t].logup_evals, DevicePtrArg::Static(0)),
                "round {round}: trace {t} has interactions and is not exhausted, but no \
                 logup batch claimed it"
            );
        }
    }

    let mut reads = Vec::new();
    for (t, a) in args.into_iter().enumerate() {
        extend_reads(&mut reads, descs.set_batch_s_ring_trace(arr, t, a));
    }
    (buf, reads)
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
    descs: &mut DescriptorPlan,
) -> RoundEvalBatchIr {
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
        // Same occupancy auto-tune the eager path runs
        // (`batch_mle_monomial.rs:344-372`); see
        // [`ZerocheckPhasePlan::monomial_chunk_size`].
        chunk_size: plan.monomial_chunk_size(traces, round, num_x),
    };
    let kind = plan.traces[traces[0]].eval_kind;
    let tag = format!("zc_r{round}");
    let mut bufs = alloc_eval_bufs(
        g,
        device,
        descs,
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
            let (main_desc, desc_reads) = main_descs.get_or_emit(g, device, descs, round, t, rb);
            extend_reads(&mut bufs.ctx_reads, desc_reads);
            emit_zerocheck_ctx_element(
                g, device, descs, &tag, &mut bufs, air as u32, rb, num_x, main_desc,
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
    RoundEvalBatchIr {
        family: RoundEvalFamily::Zerocheck,
        evals: bufs.out,
        traces: traces.to_vec(),
        num_x: num_x as usize,
    }
}

/// Emit the round's logup evaluator for `traces`, returning its output
/// buffer.
#[allow(clippy::too_many_arguments)]
fn emit_logup_round_eval(
    g: &mut GraphBuilder,
    device: DeviceType,
    plan: &ZerocheckPhasePlan,
    round: usize,
    traces: &[usize],
    round_bufs: &[Option<RoundTraceBufs>],
    main_descs: &mut MainDescCache,
    descs: &mut DescriptorPlan,
) -> RoundEvalBatchIr {
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
        chunk_size: plan.monomial_chunk_size(traces, round, num_x),
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
                pool_base: descs.pool_base().clone(),
            },
            shape,
        );
        g.register_output(out);
        RoundEvalBatchIr {
            family: RoundEvalFamily::Logup,
            evals: out,
            traces: traces.to_vec(),
            num_x: num_x as usize,
        }
    } else {
        let mut bufs = alloc_eval_bufs(
            g,
            device,
            descs,
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
            let (main_desc, desc_reads) = main_descs.get_or_emit(g, device, descs, round, t, rb);
            extend_reads(&mut bufs.ctx_reads, desc_reads);
            emit_logup_ctx_element(
                g, device, descs, &tag, &mut bufs, air as u32, rb, num_x, main_desc,
            );
        }
        logup_batch_eval_mle_ir(g, &bufs, shape);
        g.register_output(bufs.out);
        RoundEvalBatchIr {
            family: RoundEvalFamily::Logup,
            evals: bufs.out,
            traces: traces.to_vec(),
            num_x: num_x as usize,
        }
    }
}

/// Allocate the ctx / scratch buffers one batched evaluator needs.
///
/// For the two DAG kinds the ctx array is registered as a *graph input* and
/// its bytes are computed on the host by [`DescriptorPlan::bind`]; the
/// monomial placeholder array is still memset to zero.
#[allow(clippy::too_many_arguments)]
fn alloc_eval_bufs(
    g: &mut GraphBuilder,
    device: DeviceType,
    descs: &mut DescriptorPlan,
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
    let ctx_name = format!("{tag}_ctxs");
    let (air_ctxs, ctx_array) = match ctx_kind {
        CtxKind::Zerocheck => {
            let b = add_typed_buf::<ZerocheckCtx>(g, device, &ctx_name, n);
            let id = descs.add_array::<ZerocheckCtx>(g, b, &ctx_name, n);
            (b, id)
        }
        CtxKind::Logup => {
            let b = add_typed_buf::<LogupCtx>(g, device, &ctx_name, n);
            let id = descs.add_array::<LogupCtx>(g, b, &ctx_name, n);
            (b, id)
        }
        // TODO(cc-ir,R6): the three monomial ctx structs still carry their own
        //   raw pointer fields (`d_headers`, `d_variables`,
        //   `d_lambda_combinations`, `d_eq_xi`); only their embedded
        //   `EvalCoreCtx` is base+offset.
        // WHY: the IR mirror never fills these arrays — they are zeroed
        //   placeholders — so converting the remaining fields buys nothing
        //   today and would widen an already large ABI change.
        // RISK: whoever fills them must convert those fields first, or they
        //   will be host-baked addresses again. `MonomialAirCtx` is *not*
        //   registered as a graph input, so nothing forces a bind.
        CtxKind::Monomial => {
            let b = add_typed_buf::<MonomialAirCtx>(g, device, &ctx_name, n);
            g.insert_memset(b, 0);
            (b, DescArrayId(usize::MAX))
        }
    };
    let air_block_offsets = add_typed_buf::<u32>(g, device, &format!("{tag}_offsets"), n + 1);
    for b in [block_ctxs, air_block_offsets] {
        g.insert_memset(b, 0);
    }
    ZerocheckEvalBufs {
        tmp_sums,
        out,
        block_ctxs,
        air_ctxs,
        ctx_array,
        pool_base: descs.pool_base().clone(),
        air_block_offsets,
        lambda_pows,
        ctx_reads: Vec::new(),
        intermediates: Vec::new(),
    }
}

/// The transcript's current state buffer.
///
/// Reads [`FiatShamirTranscriptGraphIR::state_buf`] — it does **not** squeeze.
/// The earlier stand-in sampled one extra `[1, D_EF]` value to name the tail
/// of the sponge chain, which advanced the graph transcript one squeeze past
/// the eager phase and made the two states disagree.
fn transcript_state_of<TS: FiatShamirTranscriptGraphIR>(
    _g: &mut GraphBuilder,
    transcript: &mut TS,
) -> BufId {
    transcript.state_buf()
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
        r_0: ef(0),
        // Real powers of the skip-domain generator, not `0..2^l_skip`: the
        // ring's round-0 `eq_sharp` seed goes through
        // `eval_eq_sharp_uni`, whose debug identity is only true for an
        // actual multiplicative subgroup (`poly_common.rs:143-166`).
        omega_skip_pows: F::two_adic_generator(l_skip)
            .powers()
            .take(1 << l_skip)
            .collect(),
        inv_lagrange_denoms_r0: (0..1 << l_skip).map(ef).collect(),
        fold_selector_scalars: (0..num_traces).map(|t| (ef(t), ef(t + 1))).collect(),
        round0_denom_sum_init: (0..num_traces).map(ef).collect(),
        round0_g_shift: (0..num_traces).map(|_| F::ONE).collect(),
        s_0_coeffs: (0..=(1 << l_skip) * s_deg).map(ef).collect(),
        logup_sum_claims: (0..num_traces).map(|t| (ef(t), ef(t + 1))).collect(),
        opening_claims: (0..2 * num_traces).map(ef).collect(),
        threads_per_block: 128,
        num_blocks: 32,
        // A plausible mid-range SM count; the synthetic plan is a shape
        // fixture, not a device query.
        sm_count: 128,
        max_monomials_per_thread: DEFAULT_MAX_MONOMIALS_PER_THREAD,
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
        planner::{ListSchedulerV1, ListSchedulerV2, SchedulerMode},
    };
    use itertools::Itertools;
    use openvm_cuda_common::{
        common::get_device,
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::{cudaMemsetAsync, DeviceBuffer},
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::poly_common::{eval_eq_mle, UnivariatePoly};
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{
        cuda::{logup_zerocheck::fold_selectors_round0, sumcheck::batch_fold_mle},
        logup_zerocheck::{compute_batch_s_poly_from_state, BatchSPolyState},
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

    /// Byte view of an `[EF]` slice.
    fn ef_bytes(xs: &[EF]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)) }
    }

    /// The in-tree default: ListV1 pins every graph input and output through
    /// the schedule (`planner/list_v1.rs:638-655, 789-795`).
    fn scheduler_v1() -> SchedulerMode {
        SchedulerMode::ListV1 {
            params: ListSchedulerV1::default(),
        }
    }

    /// The shipped `SchedulerConfig::default()` (`graph_compiler_config.rs:129-147`).
    ///
    /// ListV2 gives inputs birth 0 but infinite death only to *outputs*
    /// (`planner/list_v2.rs:371-401`), so an input slot may be reused after
    /// its last first-run consumer. Any test that claims a property holds
    /// "under the shipped default" has to run here too.
    fn scheduler_v2() -> SchedulerMode {
        SchedulerMode::ListV2 {
            params: ListSchedulerV2::default(),
        }
    }

    /// Compile a graph with no runtime inputs, run it, and read back the
    /// given buffers as raw bytes. ListV1 — see [`run_graph_read_bufs_with`]
    /// to pick the scheduler.
    fn run_graph_read_bufs(g: GraphBuilder, bufs: &[BufId], ctx: &GpuDeviceCtx) -> Vec<Vec<u8>> {
        run_graph_read_bufs_with(scheduler_v1(), g, bufs, ctx)
    }

    /// [`run_graph_read_bufs`] under an explicit scheduler.
    fn run_graph_read_bufs_with(
        mode: SchedulerMode,
        mut g: GraphBuilder,
        bufs: &[BufId],
        ctx: &GpuDeviceCtx,
    ) -> Vec<Vec<u8>> {
        for &b in bufs {
            g.register_output(b);
        }
        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .scheduler(mode)
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

    /// S1.1 — every registered graph input has a producer, and an unbound
    /// one is a *loud* failure.
    ///
    /// Two properties, both of which the old `insert_memset` world lacked:
    ///
    /// 1. Every `GraphExe` input is accounted for by either the descriptor plan or the
    ///    [`PhaseInputBinder`] manifest. A buffer registered without a binder entry would slip
    ///    through as zeros.
    /// 2. `PhaseInputBinder::bind` refuses, naming the buffers, rather than filling them with
    ///    zeros.
    #[test]
    fn phase_graph_every_input_is_accounted_for() {
        let device = DeviceType::Cuda(0);
        let plan = synthetic_plan(
            /* num_traces */ 2, /* l_skip */ 2, /* n_max */ 3,
        );

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut inputs = PhaseInputBinder::new();
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_inputs(&mut g, device, &plan, t, &mut inputs))
            .collect();
        let proof =
            logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device, &mut inputs);

        let exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("phase graph compile");

        let desc_bufs: Vec<BufId> = proof
            .descriptors
            .array_summary()
            .iter()
            .map(|(b, ..)| *b)
            .collect();
        let manifest: Vec<BufId> = inputs.manifest().iter().map(|(b, _)| *b).collect();
        assert!(!manifest.is_empty(), "no phase inputs were registered");

        for i in 0..exe.num_inputs() {
            let b = exe.input_buf_id(i);
            assert!(
                desc_bufs.contains(&b) || manifest.contains(&b),
                "graph input {b:?} is in neither the descriptor plan nor the input manifest — \
                 it would run as whatever the pool slot held"
            );
        }

        // The selector cube is the one input the plan alone determines, so it
        // is bound already; everything else is not.
        let unbound = inputs.unbound();
        assert!(
            !unbound.is_empty(),
            "expected the keygen/challenge inputs to be unbound in this fixture"
        );
        assert!(
            !unbound.iter().any(|n| n.ends_with("_sels_cube")),
            "the selector cube is plan-derived and must be pre-filled, got {unbound:?}"
        );
    }

    /// S1.1 guard — a trace that skips a family must not register that
    /// family's inputs.
    ///
    /// Registered inputs carry an obligation the old `insert_memset` did not:
    /// `graph_compiler.rs:1236-1266` rejects an input **no node reads**. The
    /// eager path emits no logup launch for a trace without interactions and
    /// no constraint launch for one without constraints, so blanket-registering
    /// `t{t}_logup_rules` / `t{t}_zc_rules` would make such a plan fail to
    /// compile at all. `synthetic_plan` gives every trace both families, so
    /// this fixture is the only thing that covers the guard.
    #[test]
    fn phase_graph_compiles_when_a_trace_skips_a_family() {
        let device = DeviceType::Cuda(0);
        let mut plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 1, /* n_max */ 2,
        );
        plan.traces[0].has_interactions = false;
        plan.traces[2].has_constraints = false;

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut inputs = PhaseInputBinder::new();
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_inputs(&mut g, device, &plan, t, &mut inputs))
            .collect();
        let _ = logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device, &mut inputs);

        let names: Vec<String> = inputs.manifest().into_iter().map(|(_, n)| n).collect();
        assert!(
            !names.iter().any(|n| n == "t0_logup_rules"),
            "trace 0 has no interactions, so its logup rule stream is never read"
        );
        assert!(
            !names.iter().any(|n| n == "t2_zc_rules"),
            "trace 2 has no constraints, so its constraint rule stream is never read"
        );
        assert!(names.iter().any(|n| n == "t0_zc_rules"));
        assert!(names.iter().any(|n| n == "t2_logup_rules"));

        // The real assertion: the compiler accepts it.
        GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("asymmetric phase graph must compile");
    }

    /// S1.2(b) — the `interpolate_columns` column table, graph vs eager, on
    /// raw device bytes.
    ///
    /// The table was `insert_memset(_, 0)`, i.e. a table of null pointers, so
    /// this node could not run at all. Filling it is only half the fix; the
    /// other half is **order**. The eager table is
    /// `iter::once(sels).chain(mats)` (`mod.rs:1196-1203`) while the mirror's
    /// `srcs` dependency list is matrices-then-selectors, and copying that
    /// order into the table interpolates the wrong columns while erroring
    /// nowhere.
    ///
    /// The reference below builds its pointer vector *independently*, in the
    /// eager order, and the graph side goes through [`emit_column_table`] —
    /// the same builder the phase driver uses. Ragged widths and per-buffer
    /// distinguishable data mean a permuted table cannot pass.
    ///
    /// The `sabotage` leg perturbs one element of the selector buffer, which
    /// only the first three columns read: if the comparison were vacuous, or
    /// if the table skipped the selectors, it would not show up.
    #[test]
    fn interpolate_columns_ir_column_table_matches_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);

        let s_deg = 3usize;
        let num_y = 8usize;
        let height = 2 * num_y;
        let mat_widths = [2usize, 5, 1];
        let num_columns = 3 + mat_widths.iter().sum::<usize>();

        for sabotage in [false, true] {
            let mut rng = StdRng::seed_from_u64(0xC01D_5EED);
            let mut sels: Vec<EF> = (0..3 * height).map(|_| rng.random::<EF>()).collect();
            let mats: Vec<Vec<EF>> = mat_widths
                .iter()
                .map(|&w| (0..w * height).map(|_| rng.random::<EF>()).collect())
                .collect();
            if sabotage {
                sels[height + 3] += EF::ONE;
            }

            // --- eager reference, on the UNsabotaged selectors, with its own
            // independently built column vector.
            let mut ref_sels = sels.clone();
            if sabotage {
                ref_sels[height + 3] -= EF::ONE;
            }
            let d_sels: DeviceBuffer<EF> = ref_sels.as_slice().to_device_on(&ctx).unwrap();
            let d_mats: Vec<DeviceBuffer<EF>> = mats
                .iter()
                .map(|m| m.as_slice().to_device_on(&ctx).unwrap())
                .collect();
            let mut columns_h: Vec<*const EF> = Vec::with_capacity(num_columns);
            for col in 0..3 {
                columns_h.push(unsafe { d_sels.as_ptr().add(col * height) });
            }
            for (d, &w) in d_mats.iter().zip(mat_widths.iter()) {
                for col in 0..w {
                    columns_h.push(unsafe { d.as_ptr().add(col * height) });
                }
            }
            let d_interp: DeviceBuffer<EF> =
                DeviceBuffer::with_capacity_on(s_deg * num_y * num_columns, &ctx);
            unsafe {
                interpolate_columns_gpu(
                    &d_interp,
                    &columns_h.as_slice().to_device_on(&ctx).unwrap(),
                    s_deg,
                    num_y,
                    stream,
                )
                .expect("interpolate_columns_gpu");
            }
            ctx.stream.synchronize().unwrap();
            let want: Vec<EF> = d_interp.to_host_on(&ctx).unwrap();

            // --- graph side, through the driver's own table builder.
            let mut g = GraphBuilder::new();
            let mut descs = DescriptorPlan::new();
            let sels_buf = ef_slice_const_buf(&mut g, device, "sels", &sels);
            let mat_bufs: Vec<(BufId, usize)> = mats
                .iter()
                .zip(mat_widths.iter())
                .enumerate()
                .map(|(i, (m, &w))| (ef_slice_const_buf(&mut g, device, &format!("m{i}"), m), w))
                .collect();
            let (columns, col_reads) = emit_column_table(
                &mut g, &mut descs, device, "cols", sels_buf, &mat_bufs, num_y,
            );
            let mut srcs = vec![sels_buf];
            srcs.extend(mat_bufs.iter().map(|&(b, _)| b));
            for b in &col_reads {
                assert!(srcs.contains(b), "table points at an undeclared buffer");
            }
            let interpolated = add_ef_buf(&mut g, device, "interp", s_deg * num_y * num_columns);
            interpolate_columns_ir(
                &mut g,
                interpolated,
                s_deg * num_y * num_columns,
                columns,
                &srcs,
                num_columns,
                s_deg,
                num_y,
            );
            let got = run_graph_with_descs(g, &descs, &[interpolated], &ctx).remove(0);

            if sabotage {
                assert_ne!(
                    &got[..],
                    ef_bytes(&want),
                    "SABOTAGE LEG IS BLIND: perturbing one selector element did not change \
                     the interpolation — the selector columns are not in the table"
                );
            } else {
                assert_eq!(
                    &got[..],
                    ef_bytes(&want),
                    "interpolate_columns column table mismatch (order or contents)"
                );
            }
        }
    }

    /// S0.1 — reading the transcript's tail must not *advance* it.
    ///
    /// `transcript_state_of` used to `sample_ext`, which squeezed one extra
    /// `EF` purely to have a `BufId` to register as the phase output. That
    /// desynchronized the graph transcript from the eager one by one squeeze,
    /// so every downstream Fiat-Shamir value would differ. Written against
    /// that version this test fails on both assertions.
    #[test]
    fn transcript_state_of_does_not_advance_the_sponge() {
        let device = DeviceType::Cuda(0);
        let mut g = GraphBuilder::new();
        let mut ts = DuplexSpongeGpuIR::new(&mut g, device);

        // Advance to a non-trivial position first, so a stray squeeze would
        // have somewhere to move to.
        for i in 0..5 {
            let b =
                ef_const_ext_scalar_buf(&mut g, device, &format!("v{i}"), EF::from_usize(i + 1));
            ts.observe_ext(&mut g, b);
        }
        let _ = ts.sample_ext(&mut g);

        let pos_before = ts.position();
        let nodes_before = g.nodes.len();
        let st = transcript_state_of(&mut g, &mut ts);

        assert_eq!(
            ts.position(),
            pos_before,
            "reading the transcript tail advanced the sponge position"
        );
        assert_eq!(
            g.nodes.len(),
            nodes_before,
            "reading the transcript tail emitted a graph node"
        );
        assert_eq!(
            st,
            FiatShamirTranscriptGraphIR::state_buf(&ts),
            "the phase's transcript output is not the sponge's state buffer"
        );
    }

    /// S0.2 — the plan reproduces the eager par-Y `chunk_size` auto-tune.
    ///
    /// Hand-evaluated against `batch_mle_monomial.rs:344-372` rather than
    /// against a second copy of the loop, so the test would catch the loop
    /// being transcribed wrongly (a re-implementation compared against itself
    /// cannot).
    #[test]
    fn monomial_chunk_size_matches_eager_autotune() {
        let mut plan = synthetic_plan(
            /* num_traces */ 2, /* l_skip */ 1, /* n_max */ 6,
        );
        plan.max_monomials_per_thread = DEFAULT_MAX_MONOMIALS_PER_THREAD; // 64

        // round 1, n_lift = 6 => num_y = 32 => y_blocks = ceil(32/128) = 1 per AIR.
        // num_monomials = 8 (synthetic_plan), so with chunk_size = 64 each AIR
        // contributes 1 * ceil(8/64) = 1 block => total_blocks = 2.
        //
        // sm_count = 1 => target 4; 2 * num_x(=3) = 6 >= 4 on the first pass.
        plan.sm_count = 1;
        assert_eq!(plan.monomial_chunk_size(&[0, 1], 1, 3), 64);

        // sm_count = 64 => target 256. Halving: 64,32,16,8 all give
        // ceil(8/c) = 1 => total 2 * 3 = 6 < 256. At c = 4, ceil(8/4) = 2 =>
        // total 4 * 3 = 12; c = 2 => 4 blocks/AIR => 8 * 3 = 24; c = 1 =>
        // 8 blocks/AIR => 16 * 3 = 48, still < 256, and the loop stops at the
        // `chunk_size <= 1` floor.
        plan.sm_count = 64;
        assert_eq!(plan.monomial_chunk_size(&[0, 1], 1, 3), 1);

        // A late trace evaluates at num_y = 1 (`mod.rs:1169`), so y_blocks is
        // still 1 and the tune is unchanged — but `round_num_y` must say 1,
        // not `1 << (n_lift - round)` on an underflowing subtraction.
        assert_eq!(plan.round_num_y(0, 7), 1);
        assert_eq!(plan.round_num_y(0, 1), 32);

        // The floor and the ceiling are both respected for every shape.
        for sm in [1u32, 8, 128, 1024] {
            plan.sm_count = sm;
            let c = plan.monomial_chunk_size(&[0, 1], 1, 3);
            assert!(
                (1..=DEFAULT_MAX_MONOMIALS_PER_THREAD).contains(&c),
                "chunk_size {c} out of range for sm_count {sm}"
            );
            assert!(c.is_power_of_two(), "chunk_size {c} is not a halving step");
        }
    }

    /// Like [`run_graph_read_bufs`], but installs a pool and binds `descs`
    /// first — required for any graph whose nodes dereference a descriptor
    /// array.
    fn run_graph_with_descs(
        mut g: GraphBuilder,
        descs: &DescriptorPlan,
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
        let pool = DescriptorPlan::alloc_pool(&exe, ctx);
        descs.bind(&mut exe, ctx, pool).expect("descriptor bind");
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

    /// S1.2(c) — `batch_fold_mle`'s four control tables, graph vs eager, on
    /// raw device bytes.
    ///
    /// # Why this test exists
    ///
    /// `in_ptrs` / `out_ptrs` / `widths` / `log_output_heights` used to be
    /// `insert_memset(_, 0)`, which is the *silent* failure: `fold_mle` reads
    /// `width = widths[mat_idx]` and returns for every thread once
    /// `output_height * width == 0` (`cuda/include/sumcheck.cuh:306-309`), so
    /// the kernel was a no-op, nothing errored, and the destination buffers
    /// kept whatever the pool slot held. Written against the memset version
    /// this test fails; against the descriptor version it passes.
    ///
    /// The `sabotage` leg is the oracle's teeth: perturbing **one** element of
    /// one input matrix must change the compared bytes. Without it, a
    /// zero-vs-zero comparison would pass vacuously — which is exactly the
    /// failure mode the memset produced.
    #[test]
    fn batch_fold_mle_ir_ptr_tables_match_eager() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);

        // (log_height, width) per matrix — deliberately ragged, so a table
        // filled in the wrong order or with the wrong widths cannot pass.
        let shapes = [(4usize, 3usize), (5, 2), (3, 5), (6, 1)];

        for sabotage in [false, true] {
            let mut rng = StdRng::seed_from_u64(0x3A17_F01D);
            let r_val: EF = rng.random::<EF>();
            let mut mats: Vec<Vec<EF>> = shapes
                .iter()
                .map(|&(lh, w)| {
                    (0..(1usize << lh) * w)
                        .map(|_| rng.random::<EF>())
                        .collect()
                })
                .collect();
            if sabotage {
                // One element, one matrix. Everything else is identical.
                mats[2][7] += EF::ONE;
            }

            // --- eager reference (always on the UNsabotaged data)
            let mut reference = mats.clone();
            if sabotage {
                reference[2][7] -= EF::ONE;
            }
            let d_ins: Vec<DeviceBuffer<EF>> = reference
                .iter()
                .map(|m| m.as_slice().to_device_on(&ctx).unwrap())
                .collect();
            let d_outs: Vec<DeviceBuffer<EF>> = shapes
                .iter()
                .map(|&(lh, w)| {
                    DeviceBuffer::<EF>::with_capacity_on((1usize << (lh - 1)) * w, &ctx)
                })
                .collect();
            let in_ptrs_h: Vec<*const EF> = d_ins.iter().map(|b| b.as_ptr()).collect();
            let out_ptrs_h: Vec<*mut EF> = d_outs.iter().map(|b| b.as_mut_ptr()).collect();
            let widths_h: Vec<u32> = shapes.iter().map(|&(_, w)| w as u32).collect();
            let logh_h: Vec<u8> = shapes.iter().map(|&(lh, _)| (lh - 1) as u8).collect();
            let max_cells = shapes
                .iter()
                .map(|&(lh, w)| ((1usize << (lh - 1)) * w) as u32)
                .max()
                .unwrap();
            unsafe {
                batch_fold_mle(
                    &in_ptrs_h.as_slice().to_device_on(&ctx).unwrap(),
                    &out_ptrs_h.as_slice().to_device_on(&ctx).unwrap(),
                    &widths_h.as_slice().to_device_on(&ctx).unwrap(),
                    shapes.len() as u16,
                    &logh_h.as_slice().to_device_on(&ctx).unwrap(),
                    max_cells,
                    r_val,
                    stream,
                )
                .expect("batch_fold_mle");
            }
            ctx.stream.synchronize().unwrap();
            let want: Vec<Vec<EF>> = d_outs.iter().map(|b| b.to_host_on(&ctx).unwrap()).collect();

            // --- graph side, on `mats` (sabotaged on the second pass)
            let mut g = GraphBuilder::new();
            let mut descs = DescriptorPlan::new();
            let srcs: Vec<BufId> = mats
                .iter()
                .enumerate()
                .map(|(i, m)| ef_slice_const_buf(&mut g, device, &format!("src{i}"), m))
                .collect();
            let dsts: Vec<BufId> = shapes
                .iter()
                .enumerate()
                .map(|(i, &(lh, w))| {
                    add_ef_buf(&mut g, device, &format!("dst{i}"), (1usize << (lh - 1)) * w)
                })
                .collect();
            let (in_ptrs, out_ptrs) =
                fold_ptr_tables(&mut g, &mut descs, device, "t", &srcs, &dsts);
            let widths = typed_slice_const_buf(&mut g, device, "w", &widths_h);
            let logh = typed_slice_const_buf(&mut g, device, "lh", &logh_h);
            // The challenge is a device buffer now; here it is a constant so
            // this test stays about the four pointer/shape tables. The
            // transcript-sampled path is
            // `batch_fold_mle_ir_uses_sampled_device_challenge`.
            let r_buf = ef_const_ext_scalar_buf(&mut g, device, "r", r_val);
            batch_fold_mle_ir(
                &mut g,
                in_ptrs,
                out_ptrs,
                widths,
                logh,
                &srcs,
                &dsts,
                shapes.len() as u16,
                max_cells,
                r_buf,
            );
            let got = run_graph_with_descs(g, &descs, &dsts, &ctx);

            for (i, (g_bytes, w)) in got.iter().zip(want.iter()).enumerate() {
                let want_bytes = ef_bytes(w);
                if sabotage && i == 2 {
                    assert_ne!(
                        &g_bytes[..],
                        want_bytes,
                        "SABOTAGE LEG IS BLIND: perturbing one element of matrix 2 did not \
                         change the folded bytes — the oracle has no teeth"
                    );
                } else {
                    assert_eq!(
                        &g_bytes[..],
                        want_bytes,
                        "batch_fold_mle_ir mismatch on matrix {i} (sabotage={sabotage})"
                    );
                }
            }
        }
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

    /// R6 step 1: an evaluator must declare every buffer its descriptor array
    /// references, transitively.
    ///
    /// This is the machine check for the property this phase's correctness
    /// rests on, and base+offset does **not** retire it. The compiler has no
    /// pointer analysis: a blackbox node's entire access model is the
    /// hand-supplied `(inputs, outputs, modifies)` triple
    /// (`graph_ir.rs:1259-1266`), and `verify_graph` skips blackbox nodes on
    /// its first line (`graph_ir.rs:2191-2192`). An evaluator that reaches
    /// `selectors` through a stored *offset* without declaring `selectors`
    /// gives the packer permission to hand that pool slot to another buffer,
    /// and then reads garbage — exactly as it would through a stale pointer.
    ///
    /// What changed at R6 is where the declaration comes from. Before, it was
    /// derived from the removed materializer node's own inputs; now from
    /// the [`OffSink`] run that produces the descriptor bytes
    /// ([`DescriptorPlan::set`]). Same "cannot drift" guarantee, one fewer
    /// node in the graph.
    #[test]
    fn evaluator_declares_every_referenced_buffer() {
        let device = DeviceType::Cuda(0);
        let plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 2, /* n_max */ 3,
        );

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut inputs = PhaseInputBinder::new();
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_inputs(&mut g, device, &plan, t, &mut inputs))
            .collect();
        let proof =
            logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device, &mut inputs);

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
            let closure = proof.descriptors.referenced_bufs(ctx_arr);
            widest = widest.max(closure.len());
            for b in closure {
                assert!(
                    k.inputs.contains(&b),
                    "`{}` does not declare `{}` ({b:?}), which its descriptor array `{}` \
                     ({ctx_arr:?}) references (directly or through a nested descriptor \
                     array). An undeclared read is a pool-reuse corruption hazard.",
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

    /// R6: every descriptor array is a registered graph input, and every
    /// element that an evaluator will index is filled.
    ///
    /// The teeth: an unbound input makes `GraphExe::run` fail outright
    /// (`graph_exe.rs:719`), so a descriptor array that is registered but
    /// never filled would run against zeroed offsets — i.e. against the pool
    /// base — silently. Checking "filled == total" here is what catches that
    /// at build time.
    #[test]
    fn every_descriptor_array_is_a_registered_input_and_fully_filled() {
        let device = DeviceType::Cuda(0);
        let plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 2, /* n_max */ 3,
        );

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut inputs = PhaseInputBinder::new();
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_inputs(&mut g, device, &plan, t, &mut inputs))
            .collect();
        let proof =
            logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device, &mut inputs);

        let summary = proof.descriptors.array_summary();
        assert!(!summary.is_empty(), "fixture emitted no descriptor arrays");
        for (buf, name, filled, total) in &summary {
            assert!(
                g.input_bufs().contains(buf),
                "descriptor array `{name}` ({buf:?}) is not a registered graph input, \
                 so nothing forces `DescriptorPlan::bind` to fill it"
            );
            assert_eq!(
                filled, total,
                "descriptor array `{name}` ({buf:?}) has {filled}/{total} elements filled; \
                 an unfilled element decodes to the pool base, not to null"
            );
        }
        // Every registered input must have a producer. Since S1.1 there are
        // two kinds: descriptor arrays, filled by `DescriptorPlan::bind`, and
        // the keygen/challenge buffers, filled by `PhaseInputBinder::bind`.
        // An input in neither would reach `run` unbound.
        let manifest: Vec<BufId> = inputs.manifest().iter().map(|(b, _)| *b).collect();
        for b in g.input_bufs() {
            assert!(
                summary.iter().any(|(buf, ..)| buf == b) || manifest.contains(b),
                "graph input `{}` ({b:?}) is in neither the descriptor plan nor the \
                 `PhaseInputBinder` manifest, so nothing would bind it",
                buf_name(&g, *b),
            );
        }
    }

    /// R6 step 3: one `MainMatrixDesc` array per `(round, trace)`, shared by
    /// the zerocheck and logup families, and never crossed between AIRs.
    ///
    /// The fixture is deliberately *asymmetric*: trace 0 has constraints
    /// only and trace 2 interactions only, so `zc_traces == [0, 1]` while
    /// `lg_traces == [1, 2]`. That is the only shape in which keying the
    /// descriptor cache on the loop position (`air`) rather than the trace
    /// index goes wrong — and it goes wrong silently, by handing one AIR's
    /// folded main matrices to a different AIR's evaluator.
    #[test]
    fn main_matrix_descs_shared_per_trace_across_families() {
        let device = DeviceType::Cuda(0);
        let mut plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 1, /* n_max */ 2,
        );
        plan.traces[0].has_interactions = false;
        plan.traces[2].has_constraints = false;

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut inputs = PhaseInputBinder::new();
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_inputs(&mut g, device, &plan, t, &mut inputs))
            .collect();
        let proof =
            logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device, &mut inputs);

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

        // --- and one filled element per (round, trace, main matrix)
        let elems: usize = proof
            .descriptors
            .array_summary()
            .iter()
            .filter(|(_, name, ..)| name.ends_with("_main_desc"))
            .map(|(_, _, filled, _)| *filled)
            .sum();
        let per_round: usize = plan
            .traces
            .iter()
            .map(|tp| tp.mats.len() - usize::from(tp.has_preprocessed))
            .sum();
        assert_eq!(
            elems,
            plan.n_max * per_round,
            "duplicated `MainMatrixDesc` element writes"
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

    /// Compile `g`, hand it a pool we own, bind `descs`, run.
    ///
    /// Returns the exe and the pool base. This is the R6 call order, and the
    /// order is load-bearing — see [`DescriptorPlan::bind`].
    fn compile_bind_run(
        g: GraphBuilder,
        descs: &DescriptorPlan,
        ctx: &GpuDeviceCtx,
    ) -> (crypto_compiler::graph_exe::GraphExe, *const u8) {
        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("graph compile");
        let pool = DescriptorPlan::alloc_pool(&exe, ctx);
        descs.bind(&mut exe, ctx, pool).expect("descriptor bind");
        exe.run(ctx).expect("graph run");
        ctx.stream.synchronize().unwrap();
        let base = descs.pool_base().get();
        (exe, base)
    }

    /// Read `n` bytes straight from a device address.
    fn peek(addr: *const u8, n: usize, ctx: &GpuDeviceCtx) -> Vec<u8> {
        assert!(!addr.is_null(), "null decoded descriptor pointer");
        let view = unsafe { DeviceBuffer::<u8>::from_raw_parts(addr as *mut u8, n) };
        let host = view.to_host_on(ctx).expect("D2H");
        forget(view);
        host
    }

    /// R6: the offset-decoded descriptor must address exactly what the pointer
    /// path it replaces addressed — and the offsets it stores must be the ones
    /// the runtime itself resolves against.
    ///
    /// Two assertions per pointer field, both against primary sources:
    ///
    /// 1. **Decode equivalence.** The eager encoding of the same field is `BaseOff::from_ptr(addr)`
    ///    against a null base (`batch_mle.rs:178-183`). This test builds it and asserts
    ///    `eager.resolve(null) == pool.resolve(base)` — i.e. the two encodings decode to the same
    ///    byte, which is what "identical to the pointer path it replaces" means at the ABI level.
    /// 2. **The A6 §6.5 must-verify.** `base + plan().offsets[b]` must equal the address the
    ///    runtime's own `resolve_ptr` produces (`graph_exe.rs:1674-1688`), observed through
    ///    `GraphExe::get_output`, which is the only public caller of it. This is what would catch a
    ///    recompile or replan between reading `plan()` and running.
    ///
    /// What it does *not* cover: it cannot catch a `plan()`/`resolve_ptr`
    /// divergence for a buffer that is **not** registered as an output, since
    /// `get_output` is the only window onto `resolve_ptr`. Every buffer a
    /// descriptor points at is registered as an output here, so the coverage
    /// is complete for the fields under test — but a future field pointing at
    /// an unpinnable buffer would be unchecked.
    ///
    /// Two records are covered, exactly the branches the eager builders have:
    /// preprocessed present / absent, and `buffer_size` nonzero / zero (the
    /// null-`d_intermediates` branch, `batch_mle.rs:164-176`).
    #[test]
    fn descriptors_decode_like_the_eager_pointer_path() {
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
            let mut descs = DescriptorPlan::new();

            // --- source buffers
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

            // --- the `MainMatrixDesc` array: a graph input, host-computed
            let main_desc =
                add_typed_buf::<MainMatrixDesc>(&mut g, device, "main_desc", mains.len());
            let main_id =
                descs.add_array::<MainMatrixDesc>(&mut g, main_desc, "main_desc", mains.len());
            for (i, (&b, &w)) in mains.iter().zip(main_widths.iter()).enumerate() {
                let _ = descs.set_main_matrix_desc(main_id, i, DevicePtrArg::buf(b), w);
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
            let zc_id = descs.add_array::<ZerocheckCtx>(&mut g, zc_ctxs, "zc_ctxs", 2);
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
            let _ = descs.set_zerocheck_ctx(zc_id, 1, zc_args);

            let lg_ctxs = add_typed_buf::<LogupCtx>(&mut g, device, "lg_ctxs", 2);
            let lg_id = descs.add_array::<LogupCtx>(&mut g, lg_ctxs, "lg_ctxs", 2);
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
            let _ = descs.set_logup_ctx(lg_id, 1, lg_args);

            // A graph input must be read by *some* node
            // (`GraphExe`'s interface check), which in the real phase is the
            // evaluator. Here it is a no-op stand-in that declares the same
            // three arrays.
            let consumed = add_f_buf(&mut g, device, "consumed", 1);
            g.insert_blackbox_kernel(
                "desc_consumer",
                [main_desc, zc_ctxs, lg_ctxs].into_iter(),
                [consumed].into_iter(),
                [false, false, false].into_iter(),
                |_, _, _| {},
            );
            g.register_output(consumed);

            // Everything a descriptor points at is registered as an output, so
            // `resolve_ptr` is observable for it. The descriptor arrays
            // themselves cannot be: they are graph *inputs*, and an output
            // must be written by a node (`graph_ir.rs:1062-1065`).
            let mut tracked = vec![
                selectors, public, eq_xi, challenges, eq_3bs, rules, used_nodes, pair_idxs,
            ];
            tracked.extend(mains.iter().copied());
            tracked.extend(prep);
            tracked.extend(intermediates);
            for &b in &tracked {
                g.register_output(b);
            }

            let (mut exe, base) = compile_bind_run(g, &descs, &ctx);

            let out_idx = |b: BufId| {
                (0..exe.num_outputs())
                    .find(|&i| exe.output_buf_id(i) == b)
                    .unwrap_or_else(|| panic!("buf {b:?} not registered as an output"))
            };
            // The runtime's own `resolve_ptr`, via its only public caller.
            let runtime_addr = |b: BufId| exe.get_output(out_idx(b)).as_raw_ptr() as *const u8;
            // `base + plan().offsets[b]` — what the descriptors encoded.
            let planned_addr = |b: BufId| {
                let off = exe.plan().offsets[b.0].expect("pool slot");
                (base as usize).wrapping_add(off as usize) as *const u8
            };
            // A descriptor array's own bytes: it is an input, so read the pool
            // slot directly.
            let desc_bytes = |b: BufId, n: usize| peek(planned_addr(b), n, &ctx);

            // --- A6 §6.5 must-verify, for every buffer under test.
            for &b in &tracked {
                assert_eq!(
                    planned_addr(b),
                    runtime_addr(b),
                    "case {case}: `plan().offsets[{b:?}]` disagrees with the address \
                     `resolve_ptr` produces at run time; every descriptor encoded against \
                     the plan is pointing at the wrong buffer"
                );
            }

            // --- decode equivalence, field by field.
            //
            // `pool` is what the graph uploaded; `eager` is what
            // `batch_mle.rs` would have uploaded for the same addresses.
            // Both must decode to the same device byte.
            let check = |what: &str, pool: BaseOff, want: *const u8| {
                let eager = if want.is_null() {
                    BaseOff::NULL
                } else {
                    BaseOff::from_ptr(want)
                };
                assert_eq!(
                    pool.resolve(base),
                    want,
                    "case {case}: {what}: pool-encoded {pool:?} decodes to {:?}, want {want:?}",
                    pool.resolve(base),
                );
                assert_eq!(
                    eager.resolve(std::ptr::null()),
                    pool.resolve(base),
                    "case {case}: {what}: the eager encoding {eager:?} and the pool encoding \
                     {pool:?} decode to different addresses"
                );
            };

            let main_bytes = desc_bytes(main_desc, mains.len() * size_of::<MainMatrixDesc>());
            for (i, (&b, &w)) in mains.iter().zip(main_widths.iter()).enumerate() {
                let got: MainMatrixDesc = unsafe {
                    (main_bytes.as_ptr() as *const MainMatrixDesc)
                        .add(i)
                        .read_unaligned()
                };
                check(&format!("main_desc[{i}].data"), got.data, runtime_addr(b));
                assert_eq!(got.air_width, w, "case {case}: main_desc[{i}].air_width");
            }

            let prep_addr = prep.map(runtime_addr).unwrap_or(std::ptr::null());
            let inter_addr = intermediates.map(runtime_addr).unwrap_or(std::ptr::null());

            let zc_raw = desc_bytes(zc_ctxs, 2 * size_of::<ZerocheckCtx>());
            let zc: ZerocheckCtx = unsafe {
                (zc_raw.as_ptr() as *const ZerocheckCtx)
                    .add(1)
                    .read_unaligned()
            };
            check(
                "zc.d_selectors",
                zc.eval_ctx.d_selectors,
                runtime_addr(selectors),
            );
            check(
                "zc.d_preprocessed.data",
                zc.eval_ctx.d_preprocessed.data,
                prep_addr,
            );
            assert_eq!(
                zc.eval_ctx.d_preprocessed.air_width, prep_width,
                "case {case}: zc.d_preprocessed.air_width"
            );
            // `main_desc` is a graph input, so `resolve_ptr` is not observable
            // for it (see this test's doc comment); `planned_addr` is the best
            // available reference.
            check("zc.d_main", zc.eval_ctx.d_main, planned_addr(main_desc));
            check("zc.d_public", zc.eval_ctx.d_public, runtime_addr(public));
            check("zc.d_intermediates", zc.d_intermediates, inter_addr);
            check("zc.d_eq_xi", zc.d_eq_xi, runtime_addr(eq_xi));
            check("zc.d_rules", zc.d_rules, runtime_addr(rules));
            check("zc.d_used_nodes", zc.d_used_nodes, runtime_addr(used_nodes));
            assert_eq!(zc.num_y, num_y as u32, "case {case}: zc.num_y");
            assert_eq!(zc.rules_len, rules_len, "case {case}: zc.rules_len");
            assert_eq!(
                zc.used_nodes_len, used_nodes_len,
                "case {case}: zc.used_nodes_len"
            );
            assert_eq!(zc.buffer_size, buffer_size, "case {case}: zc.buffer_size");

            // Element 0 was never `set`: it stays all-zero, proving `idx` is
            // honoured. Note this is *not* the null encoding — offset 0 is a
            // valid pool offset, which is exactly why `BaseOff::NULL` is
            // `u64::MAX`.
            let zc0: ZerocheckCtx =
                unsafe { (zc_raw.as_ptr() as *const ZerocheckCtx).read_unaligned() };
            assert_eq!(
                zc0.eval_ctx.d_selectors,
                BaseOff::from_offset(0),
                "case {case}: zc_ctxs[0] must be untouched"
            );

            let lg_raw = desc_bytes(lg_ctxs, 2 * size_of::<LogupCtx>());
            let lg: LogupCtx =
                unsafe { (lg_raw.as_ptr() as *const LogupCtx).add(1).read_unaligned() };
            check(
                "lg.d_selectors",
                lg.eval_ctx.d_selectors,
                runtime_addr(selectors),
            );
            check(
                "lg.d_preprocessed.data",
                lg.eval_ctx.d_preprocessed.data,
                prep_addr,
            );
            assert_eq!(
                lg.eval_ctx.d_preprocessed.air_width, prep_width,
                "case {case}: lg.d_preprocessed.air_width"
            );
            check("lg.d_main", lg.eval_ctx.d_main, planned_addr(main_desc));
            check("lg.d_public", lg.eval_ctx.d_public, runtime_addr(public));
            check("lg.d_intermediates", lg.d_intermediates, inter_addr);
            check("lg.d_eq_xi", lg.d_eq_xi, runtime_addr(eq_xi));
            check("lg.d_challenges", lg.d_challenges, runtime_addr(challenges));
            check("lg.d_eq_3bs", lg.d_eq_3bs, runtime_addr(eq_3bs));
            check("lg.d_rules", lg.d_rules, runtime_addr(rules));
            check("lg.d_used_nodes", lg.d_used_nodes, runtime_addr(used_nodes));
            check("lg.d_pair_idxs", lg.d_pair_idxs, runtime_addr(pair_idxs));
            assert_eq!(lg.num_y, num_y as u32, "case {case}: lg.num_y");
            assert_eq!(lg.rules_len, rules_len, "case {case}: lg.rules_len");
            assert_eq!(
                lg.used_nodes_len, used_nodes_len,
                "case {case}: lg.used_nodes_len"
            );
            assert_eq!(lg.buffer_size, buffer_size, "case {case}: lg.buffer_size");

            // --- replay: same exe, same pool, descriptors re-uploaded.
            //     Pool addresses are stable, so re-encoding produces the same
            //     bytes at the same addresses — the capture-stability
            //     contract. The re-upload is not optional bookkeeping: graph
            //     inputs are not preserved across an execution
            //     (`crates/compiler/notes.md:46-52`), so a replay that skipped
            //     it would be reading whatever the planner put in those slots.
            //     See [`DescriptorPlan::upload`].
            let zc_addr = planned_addr(zc_ctxs);
            let lg_addr = planned_addr(lg_ctxs);
            let md_addr = planned_addr(main_desc);
            descs.upload(&mut exe, &ctx).expect("descriptor re-upload");
            exe.run(&ctx).expect("graph replay");
            ctx.stream.synchronize().unwrap();
            assert_eq!(
                peek(zc_addr, zc_raw.len(), &ctx),
                zc_raw,
                "case {case}: ZerocheckCtx bytes not stable across replay"
            );
            assert_eq!(
                peek(lg_addr, lg_raw.len(), &ctx),
                lg_raw,
                "case {case}: LogupCtx bytes not stable across replay"
            );
            assert_eq!(
                peek(md_addr, main_bytes.len(), &ctx),
                main_bytes,
                "case {case}: MainMatrixDesc bytes not stable across replay"
            );
        }
    }

    /// R6 step 2: the encoded offsets must still address live, correct bytes
    /// when the pool packer is *allowed* to reuse slots.
    ///
    /// `descriptors_decode_like_the_eager_pointer_path` above cannot check
    /// this. It registers every pointed-to buffer as a graph output, and
    /// `GraphCompiler` pins inputs ∪ outputs (`graph_exe.rs:634-644`), which
    /// forces `death[b] = n` for all of them — no slot can ever be reused,
    /// which is precisely the configuration in which an under-declared
    /// `ctx_reads` is harmless. It proves the field map; it proves nothing
    /// about liveness.
    ///
    /// Here only the three descriptor arrays are pinned (automatically — they
    /// are graph inputs), every pointed-to buffer is left packable, and the
    /// packer is given a reason to reuse: a `dummy` buffer larger than the
    /// whole rest of the pool, alive across the evaluator.
    ///
    /// The node order is a deterministic sandwich, forced by data edges:
    ///
    /// ```text
    ///   seed memsets -> dummy_gate (fills dummy 0x5c) -> eval -> dummy_sink
    /// ```
    ///
    /// `dummy_gate` reads the ctx arrays and writes `gate`, which the
    /// evaluator declares (so it cannot follow the evaluator). It is `dummy`'s
    /// sole writer — the graph IR is SSA, and `verify_graph` rejects a second
    /// one — so it both opens `dummy`'s lifetime and issues the clobbering
    /// fill. Therefore:
    ///
    /// * **declared correctly** — `selectors` &c. are read by the evaluator, so they die after
    ///   `dummy` is born, their lifetimes overlap, and the packer must give them disjoint slots.
    ///   The bytes survive.
    /// * **under-declared** — nothing reads them at all: they die at their seeding memset, before
    ///   `dummy` is born, so the packer puts them inside `dummy` and the fill overwrites the bytes
    ///   the encoded offsets address.
    ///
    /// Red-green: uncomment the `ctx_reads.retain` line below and this test
    /// fails on `selectors`' byte pattern.
    // TODO(cc-ir,R6): the teeth depend on planner behaviour this test does
    //   not assert — largest-first placement putting `dummy` at offset 0, and
    //   the sandwich order actually being chosen.
    // WHY: the per-buffer offsets of *unpinned* buffers are readable from
    //   `plan()` now, but asserting a specific packing would nail the test to
    //   one scheduler.
    // RISK: a future scheduler or packer change could make this test pass
    //   *vacuously* — green, but no longer able to fail. Re-arm the commented
    //   `ctx_reads` line after any planner change and confirm it still fails.
    #[test]
    fn descriptor_offsets_survive_pool_reuse() {
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
        let mut descs = DescriptorPlan::new();

        // Every source buffer carries a distinct byte pattern, so a byte read
        // back through a decoded offset identifies which buffer's storage the
        // offset actually landed on.
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

        // The declared read set, derived from the descriptor writers
        // themselves (R6 step 1) — this vector is the thing under test.
        let mut ctx_reads: Vec<BufId> = Vec::new();

        let main_desc = add_typed_buf::<MainMatrixDesc>(&mut g, device, "main_desc", mains.len());
        let main_id =
            descs.add_array::<MainMatrixDesc>(&mut g, main_desc, "main_desc", mains.len());
        for (i, (&b, &w)) in mains.iter().zip(main_widths.iter()).enumerate() {
            extend_reads(
                &mut ctx_reads,
                descs.set_main_matrix_desc(main_id, i, DevicePtrArg::buf(b), w),
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
        let zc_id = descs.add_array::<ZerocheckCtx>(&mut g, zc_ctxs, "zc_ctxs", 1);
        extend_reads(
            &mut ctx_reads,
            descs.set_zerocheck_ctx(
                zc_id,
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
        let lg_id = descs.add_array::<LogupCtx>(&mut g, lg_ctxs, "lg_ctxs", 1);
        extend_reads(
            &mut ctx_reads,
            descs.set_logup_ctx(
                lg_id,
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

        // `sink` is pure scaffolding — it keeps `dummy_sink` out of DCE's
        // reach, and it is not a buffer any descriptor addresses. The three
        // descriptor arrays are pinned automatically, as graph inputs.
        g.register_output(sink);

        let (exe, base) = compile_bind_run(g, &descs, &ctx);
        let planned_addr = |b: BufId| {
            let off = exe.plan().offsets[b.0].expect("pool slot");
            (base as usize).wrapping_add(off as usize) as *const u8
        };

        let zc_raw = peek(planned_addr(zc_ctxs), size_of::<ZerocheckCtx>(), &ctx);
        let lg_raw = peek(planned_addr(lg_ctxs), size_of::<LogupCtx>(), &ctx);
        let desc_raw = peek(
            planned_addr(main_desc),
            mains.len() * size_of::<MainMatrixDesc>(),
            &ctx,
        );
        let zc: ZerocheckCtx = unsafe { (zc_raw.as_ptr() as *const ZerocheckCtx).read_unaligned() };
        let lg: LogupCtx = unsafe { (lg_raw.as_ptr() as *const LogupCtx).read_unaligned() };

        // Every pointer field, paired with the buffer it must address.
        let mut probes: Vec<(&str, BaseOff)> = vec![
            ("selectors", zc.eval_ctx.d_selectors),
            ("selectors", lg.eval_ctx.d_selectors),
            ("prep", zc.eval_ctx.d_preprocessed.data),
            ("prep", lg.eval_ctx.d_preprocessed.data),
            ("public", zc.eval_ctx.d_public),
            ("public", lg.eval_ctx.d_public),
            ("inter", zc.d_intermediates),
            ("inter", lg.d_intermediates),
            ("eq_xi", zc.d_eq_xi),
            ("eq_xi", lg.d_eq_xi),
            ("rules", zc.d_rules),
            ("rules", lg.d_rules),
            ("used_nodes", zc.d_used_nodes),
            ("used_nodes", lg.d_used_nodes),
            ("challenges", lg.d_challenges),
            ("eq_3bs", lg.d_eq_3bs),
            ("pair_idxs", lg.d_pair_idxs),
        ];
        // …reached through the second descriptor level, too.
        assert_eq!(zc.eval_ctx.d_main, lg.eval_ctx.d_main);
        for i in 0..mains.len() {
            let got: MainMatrixDesc = unsafe {
                (desc_raw.as_ptr() as *const MainMatrixDesc)
                    .add(i)
                    .read_unaligned()
            };
            probes.push((if i == 0 { "main0" } else { "main1" }, got.data));
        }

        for (label, off) in probes {
            let &(_, _, n, pat) = seeded
                .iter()
                .find(|(name, ..)| *name == label)
                .expect("seeded buffer");
            let addr = off.resolve(base);
            let got = peek(addr, n, &ctx);
            assert!(
                got.iter().all(|&b| b == pat),
                "`{label}` at {addr:?} ({off:?} + base) does not hold its own bytes any \
                 more (expected all {pat:#04x}, first mismatch at index {}, value {:#04x}). \
                 The pool packer reused its slot, which means the evaluator's declared \
                 `ctx_reads` is missing it.",
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
                FoldPleDst::Fresh(out_buf),
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

    /// A fixed set of descriptor elements covering all four `DescElem`
    /// variants, with every field distinct so a swapped or dropped field
    /// cannot pass unnoticed.
    fn descriptor_fixture() -> Vec<(&'static str, DescElem)> {
        let core = |k: usize| EvalCoreCtxArgs {
            d_selectors: DevicePtrArg::Static(0x1_0000 + k),
            d_preprocessed_data: DevicePtrArg::Static(0x2_0000 + k),
            preprocessed_air_width: 7 + k as u32,
            d_main: DevicePtrArg::Static(0x3_0000 + k),
            d_public: DevicePtrArg::Static(0x4_0000 + k),
        };
        vec![
            (
                "MainMatrixDesc",
                DescElem::MainMatrix {
                    data: DevicePtrArg::Static(0xDEAD_0000),
                    air_width: 0x1234_5678,
                },
            ),
            (
                "MainMatrixDesc/absent",
                DescElem::MainMatrix {
                    data: DevicePtrArg::NULL,
                    air_width: 0,
                },
            ),
            (
                "ZerocheckCtx",
                DescElem::Zerocheck(Box::new(ZerocheckCtxArgs {
                    eval_ctx: core(1),
                    d_intermediates: DevicePtrArg::Static(0x5_0000),
                    num_y: 0x0BAD_F00D,
                    d_eq_xi: DevicePtrArg::Static(0x6_0000),
                    d_rules: DevicePtrArg::Static(0x7_0000),
                    rules_len: 0x1122_3344,
                    d_used_nodes: DevicePtrArg::Static(0x8_0000),
                    used_nodes_len: 0x5566_7788,
                    buffer_size: 0x99AA_BBCC,
                })),
            ),
            (
                "LogupCtx",
                DescElem::Logup(Box::new(LogupCtxArgs {
                    eval_ctx: core(2),
                    d_intermediates: DevicePtrArg::Static(0x9_0000),
                    num_y: 0x0FEE_1DAD,
                    d_eq_xi: DevicePtrArg::Static(0xA_0000),
                    d_challenges: DevicePtrArg::Static(0xB_0000),
                    d_eq_3bs: DevicePtrArg::Static(0xC_0000),
                    d_rules: DevicePtrArg::Static(0xD_0000),
                    rules_len: 0x0102_0304,
                    d_used_nodes: DevicePtrArg::Static(0xE_0000),
                    d_pair_idxs: DevicePtrArg::Static(0xF_0000),
                    used_nodes_len: 0x0506_0708,
                    buffer_size: 0x090A_0B0C,
                })),
            ),
            (
                "RawPtr",
                DescElem::RawPtr(DevicePtrArg::Static(0x1234_5678)),
            ),
        ]
    }

    /// Dirty a few KiB of stack with a recognisable pattern, so that any
    /// *uninitialized* byte a subsequent encode reads is likely to come back
    /// as `0xA5` rather than as an incidental zero.
    #[inline(never)]
    fn poison_stack() -> u8 {
        let mut buf = [0xA5u8; 8192];
        // Defeat const-propagation; the array must really be written.
        buf[(buf.len() - 1) & 0x1FFF] = 0xA5;
        std::hint::black_box(&buf);
        buf[4096]
    }

    /// S2 — a descriptor array must be **re-uploaded before every run**.
    ///
    /// # The rule and the violation
    ///
    /// `crates/compiler/notes.md:46-52` is explicit: graph inputs are not
    /// preserved across an execution, so they must be set for every launch.
    /// The pre-fix `DescriptorPlan` uploaded its arrays exactly once, inside
    /// `bind`, and offered no way to do it again: `bind` starts with
    /// `set_scratch`, which rejects an exe that already owns a pool
    /// (`graph_exe.rs:565-572`). A second execution therefore read whatever
    /// the planner had since put in those slots — and the decoder adds that
    /// value to the pool base (`cuda/include/base_off.cuh:34-54`), so the
    /// consequence is wrong in-pool data or an invalid device address, not a
    /// benign stale read.
    ///
    /// Two legs:
    ///
    /// 1. **The hazard is real**, not theoretical. ListV1 pins graph inputs and outputs through the
    ///    schedule (`planner/list_v1.rs:638-655, 789-795`); ListV2 — what
    ///    `SchedulerConfig::default()` ships (`graph_compiler_config.rs:129-147`) — gives an input
    ///    birth 0 but infinite death only to outputs (`planner/list_v2.rs:371-401`). On the fixture
    ///    below ListV2 really does hand the descriptor input's slot to a later buffer and ListV1
    ///    really does not.
    /// 2. **The remedy works and the old API could not express it.** Clobbering the descriptor slot
    ///    between runs — exactly what leg 1 shows the planner is allowed to do — corrupts the next
    ///    run's read; `DescriptorPlan::upload` restores it byte for byte; and a second `bind`, the
    ///    only tool the pre-fix API had, fails.
    #[test]
    fn descriptor_inputs_must_be_reuploaded_before_every_run() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);

        // ---- leg 1: does the planner reuse a registered input's slot?
        //
        // One descriptor-array input, then a chain of same-sized buffers. The
        // descriptor references only a `Static` address, so no other graph
        // buffer competes for the pool.
        let reuses_input_slot = |mode: SchedulerMode| {
            let mut g = GraphBuilder::new();
            let mut descs = DescriptorPlan::new();
            let db = add_typed_buf::<MainMatrixDesc>(&mut g, device, "descs", 1);
            let id = descs.add_array::<MainMatrixDesc>(&mut g, db, "descs", 1);
            let _ = descs.set_main_matrix_desc(id, 0, DevicePtrArg::Static(0x1234_0000), 3);
            let mut prev = db;
            let mut chain = vec![];
            for k in 0..8 {
                let b = add_typed_buf::<MainMatrixDesc>(&mut g, device, &format!("s{k}"), 1);
                g.insert_memcpy(prev, b);
                chain.push(b);
                prev = b;
            }
            g.register_output(prev);
            let exe = GraphCompiler::new()
                .device(DeviceType::Cuda(0))
                .scheduler(mode)
                .compile(g)
                .expect("compile");
            let offs = &exe.plan().offsets;
            let desc_off = offs[db.0].expect("descriptor input has no pool slot");
            chain
                .iter()
                .filter(|b| **b != prev)
                .any(|b| offs[b.0] == Some(desc_off))
        };
        assert!(
            !reuses_input_slot(scheduler_v1()),
            "ListV1 is documented to pin graph inputs through the schedule; if it now reuses \
             their slots, every one-shot upload in this crate is unsound under it too"
        );
        assert!(
            reuses_input_slot(scheduler_v2()),
            "ListV2 did not reuse the descriptor input's slot on this fixture, so leg 2's \
             clobber is no longer modelling something the planner actually does. Re-derive the \
             fixture before trusting the one-shot-upload argument."
        );

        // ---- leg 2: the remedy, on a graph whose output *is* the descriptor
        //      slot's runtime contents.
        let mut g = GraphBuilder::new();
        let mut descs = DescriptorPlan::new();
        let db = add_typed_buf::<MainMatrixDesc>(&mut g, device, "descs", 1);
        let id = descs.add_array::<MainMatrixDesc>(&mut g, db, "descs", 1);
        let _ = descs.set_main_matrix_desc(id, 0, DevicePtrArg::Static(0x1234_0000), 3);
        let obs = add_typed_buf::<MainMatrixDesc>(&mut g, device, "obs", 1);
        g.insert_memcpy(db, obs);
        g.register_output(obs);
        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .scheduler(scheduler_v2())
            .compile(g)
            .expect("compile");
        let pool = DescriptorPlan::alloc_pool(&exe, &ctx);
        descs.install_pool(&mut exe, pool).expect("install pool");
        descs.upload(&mut exe, &ctx).expect("first upload");
        let oi = (0..exe.num_outputs())
            .find(|&i| exe.output_buf_id(i) == obs)
            .expect("obs output");
        let desc_bytes = size_of::<MainMatrixDesc>();
        let base = descs.pool_base().get();
        let desc_addr = (base as usize)
            .wrapping_add(exe.plan().offsets[db.0].expect("pool slot") as usize)
            as *mut u8;

        exe.run(&ctx).expect("run 1");
        ctx.stream.synchronize().unwrap();
        let out1 = exe.get_output(oi).to_host_on(&ctx).expect("D2H");
        assert!(
            out1.iter().any(|&b| b != 0xEE),
            "setup: the encoded descriptor must not already look like the clobber pattern"
        );

        // The planner is allowed to have put something else here between runs
        // (leg 1). Model that.
        unsafe {
            cudaMemsetAsync(desc_addr as *mut _, 0xEE, desc_bytes, ctx.stream.as_raw());
        }
        ctx.stream.synchronize().unwrap();

        // Replay with no re-upload: the run reads the clobbered bytes. This is
        // the failure the one-shot `bind` shipped.
        exe.run(&ctx).expect("run 2");
        ctx.stream.synchronize().unwrap();
        let out2 = exe.get_output(oi).to_host_on(&ctx).expect("D2H");
        assert_eq!(
            out2,
            vec![0xEEu8; desc_bytes],
            "a replay without re-upload must be observably reading the overwritten slot; if it \
             is not, this test no longer demonstrates anything"
        );

        // The pre-fix API's only recourse — call `bind` again — cannot work.
        let pool2 = DescriptorPlan::alloc_pool(&exe, &ctx);
        assert!(
            descs.bind(&mut exe, &ctx, pool2).is_err(),
            "`bind` must stay one-shot: `set_scratch` rejects an exe that already owns a pool. \
             That is exactly why the repeatable half had to be split out."
        );

        // The remedy.
        descs.upload(&mut exe, &ctx).expect("re-upload");
        exe.run(&ctx).expect("run 3");
        ctx.stream.synchronize().unwrap();
        let out3 = exe.get_output(oi).to_host_on(&ctx).expect("D2H");
        assert_eq!(
            out3, out1,
            "after `upload` the replay must reproduce the first run byte for byte"
        );
    }

    /// R6/S2 — the same logical descriptor must encode to identical bytes
    /// every time, and every byte of the image must be either a named field
    /// or a zero.
    ///
    /// # The defect this locks out
    ///
    /// `DescElem::encode` used to build the `#[repr(C)]` value and then take
    /// `slice::from_raw_parts(v as *const T as *const u8, size_of::<T>())`.
    /// Every one of these structs has padding — `MainMatrixDesc` is a `u64`
    /// plus a `u32` at align 8, so four trailing bytes; `ZerocheckCtx` and
    /// `LogupCtx` add interior padding around their `u32`s — and Rust does not
    /// initialize padding. That read is undefined behaviour, and, more
    /// practically for this branch, it makes the encoded bytes not a function
    /// of the descriptor's fields. Every byte-equality oracle here compares
    /// encoded descriptor bytes; a byte that can differ run to run silently
    /// weakens all of them, including the replay leg of
    /// `descriptors_decode_like_the_eager_pointer_path`, which compares the
    /// *complete* image, padding included.
    ///
    /// No device is needed: `DevicePtrArg::Static` never consults the plan.
    #[test]
    fn descriptor_encoding_is_byte_deterministic() {
        let offsets: Vec<Option<u64>> = vec![];
        let encode = |elem: &DescElem| {
            let mut enc = OffEncoder {
                offsets: &offsets,
                base: 0,
            };
            elem.encode(&mut enc)
        };

        for (name, elem) in descriptor_fixture() {
            let first = encode(&elem);
            for round in 0..8 {
                std::hint::black_box(poison_stack());
                let again = encode(&elem);
                assert_eq!(
                    first, again,
                    "`{name}` encoded to different bytes on round {round}; the encoding must be \
                     a pure function of the descriptor's fields"
                );
            }
        }

        // Every byte outside a named field must be a deterministic zero.
        let padding_free = |name: &str, bytes: &[u8], fields: &[(usize, usize)]| {
            let mut covered = vec![false; bytes.len()];
            for &(at, n) in fields {
                assert!(
                    at + n <= bytes.len(),
                    "`{name}`: field {at}..{} is out of range",
                    at + n
                );
                covered[at..at + n].fill(true);
            }
            for (i, (&b, &c)) in bytes.iter().zip(covered.iter()).enumerate() {
                if !c {
                    assert_eq!(
                        b, 0,
                        "`{name}`: padding byte {i} is {b:#04x}, not zero — the encoder is \
                         reading uninitialized memory"
                    );
                }
            }
            // Sanity: the fixture must actually exercise padding somewhere.
            covered.iter().filter(|c| !**c).count()
        };

        let mm_fields = [
            (offset_of!(MainMatrixDesc, data), size_of::<BaseOff>()),
            (offset_of!(MainMatrixDesc, air_width), size_of::<u32>()),
        ];
        let pad = padding_free(
            "MainMatrixDesc",
            &encode(&descriptor_fixture()[0].1),
            &mm_fields,
        );
        assert!(
            pad > 0,
            "MainMatrixDesc was expected to carry trailing padding; if the layout changed, this \
             test no longer proves anything"
        );

        let core_fields = |at: usize| {
            vec![
                (
                    at + offset_of!(EvalCoreCtx, d_selectors),
                    size_of::<BaseOff>(),
                ),
                (
                    at + offset_of!(EvalCoreCtx, d_preprocessed) + mm_fields[0].0,
                    mm_fields[0].1,
                ),
                (
                    at + offset_of!(EvalCoreCtx, d_preprocessed) + mm_fields[1].0,
                    mm_fields[1].1,
                ),
                (at + offset_of!(EvalCoreCtx, d_main), size_of::<BaseOff>()),
                (at + offset_of!(EvalCoreCtx, d_public), size_of::<BaseOff>()),
            ]
        };

        let mut zc_fields = core_fields(offset_of!(ZerocheckCtx, eval_ctx));
        zc_fields.extend([
            (
                offset_of!(ZerocheckCtx, d_intermediates),
                size_of::<BaseOff>(),
            ),
            (offset_of!(ZerocheckCtx, num_y), size_of::<u32>()),
            (offset_of!(ZerocheckCtx, d_eq_xi), size_of::<BaseOff>()),
            (offset_of!(ZerocheckCtx, d_rules), size_of::<BaseOff>()),
            (offset_of!(ZerocheckCtx, rules_len), size_of::<usize>()),
            (offset_of!(ZerocheckCtx, d_used_nodes), size_of::<BaseOff>()),
            (offset_of!(ZerocheckCtx, used_nodes_len), size_of::<usize>()),
            (offset_of!(ZerocheckCtx, buffer_size), size_of::<u32>()),
        ]);
        let pad = padding_free(
            "ZerocheckCtx",
            &encode(&descriptor_fixture()[2].1),
            &zc_fields,
        );
        assert!(pad > 0, "ZerocheckCtx was expected to carry padding");

        let mut lg_fields = core_fields(offset_of!(LogupCtx, eval_ctx));
        lg_fields.extend([
            (offset_of!(LogupCtx, d_intermediates), size_of::<BaseOff>()),
            (offset_of!(LogupCtx, num_y), size_of::<u32>()),
            (offset_of!(LogupCtx, d_eq_xi), size_of::<BaseOff>()),
            (offset_of!(LogupCtx, d_challenges), size_of::<BaseOff>()),
            (offset_of!(LogupCtx, d_eq_3bs), size_of::<BaseOff>()),
            (offset_of!(LogupCtx, d_rules), size_of::<BaseOff>()),
            (offset_of!(LogupCtx, rules_len), size_of::<usize>()),
            (offset_of!(LogupCtx, d_used_nodes), size_of::<BaseOff>()),
            (offset_of!(LogupCtx, d_pair_idxs), size_of::<BaseOff>()),
            (offset_of!(LogupCtx, used_nodes_len), size_of::<usize>()),
            (offset_of!(LogupCtx, buffer_size), size_of::<u32>()),
        ]);
        let pad = padding_free("LogupCtx", &encode(&descriptor_fixture()[3].1), &lg_fields);
        assert!(pad > 0, "LogupCtx was expected to carry padding");
    }

    /// Round 0's **rotated** fold: both halves must land in ONE allocation.
    ///
    /// # The defect this locks out
    ///
    /// A `need_rot` matrix is folded by two launches into one doubled-width
    /// buffer (`fold_ple.rs:24-57`). Spelling the second launch as a fresh
    /// `BufId` + [`GraphBuilder::alias_bufs`] compiles and runs, and is
    /// silently wrong: `alias_bufs` only records a parent id
    /// (`graph_ir.rs:1021-1048`) and `plan_memory` never reads the alias table
    /// (`graph_compiler.rs:594-639`), so the planner hands the two ids
    /// *different* pool offsets. The lower half is then written into one
    /// allocation and the upper half into another, and every later round reads
    /// a buffer whose lower half no launch ever wrote.
    ///
    /// Three properties, in increasing strength:
    ///
    /// 1. the builder introduces no buffer rename at all (`g.aliases` is empty of `Some`), so the
    ///    forbidden spelling cannot creep back in;
    /// 2. the rotated node declares the fold destination as a **carried mutation** and no fresh
    ///    output — the only form the scheduler treats as one allocation (`notes.md:37-44`);
    /// 3. the full doubled-width buffer is byte-identical to the eager two-launch result, under
    ///    both shipped schedulers. Property 3 is what actually fails if 1 or 2 regress: the aliased
    ///    spelling leaves `[0, num_x*width)` of the read-back buffer holding whatever the pool slot
    ///    happened to contain.
    #[test]
    fn fold_ple_rotate_ir_writes_one_allocation() {
        let ctx = test_ctx();
        let stream = ctx.stream.as_raw();
        let device = DeviceType::Cuda(0);

        for (l_skip, log_height, width) in [(2usize, 5usize, 3usize), (3, 7, 4)] {
            let mut rng = StdRng::seed_from_u64(0xF01D_0A7Au64.wrapping_add(width as u64));
            let skip_domain = 1usize << l_skip;
            let height = 1usize << log_height;
            let num_x = height / skip_domain;
            // The `need_rot` shape: one buffer, `2 * width` columns.
            let out_len = num_x * 2 * width;

            let mat: Vec<F> = (0..height * width).map(|_| rng.random::<F>()).collect();
            let omega: Vec<F> = (0..skip_domain).map(|_| rng.random::<F>()).collect();
            let denoms: Vec<EF> = (0..skip_domain).map(|_| rng.random::<EF>()).collect();

            // --- eager reference: `fold_ple_evals_rotate`'s two launches into
            //     one allocation (`fold_ple.rs:30-56`).
            let d_mat: DeviceBuffer<F> = mat.as_slice().to_device_on(&ctx).unwrap();
            let d_omega: DeviceBuffer<F> = omega.as_slice().to_device_on(&ctx).unwrap();
            let d_denoms: DeviceBuffer<EF> = denoms.as_slice().to_device_on(&ctx).unwrap();
            let d_out: DeviceBuffer<EF> = DeviceBuffer::with_capacity_on(out_len, &ctx);
            for (rotate, off) in [(false, 0usize), (true, num_x * width)] {
                unsafe {
                    fold_ple_from_evals(
                        &d_mat,
                        d_out.as_mut_ptr().add(off),
                        &d_omega,
                        &d_denoms,
                        height as u32,
                        width as u32,
                        l_skip as u32,
                        num_x as u32,
                        rotate,
                        stream,
                    )
                    .expect("fold_ple_from_evals");
                }
            }
            ctx.stream.synchronize().unwrap();
            let want: Vec<EF> = d_out.to_host_on(&ctx).unwrap();

            // --- graph side: exactly the two calls the phase makes.
            //     `GraphBuilder` is not `Clone` (it owns the blackbox
            //     closures), so the graph is rebuilt per scheduler.
            let build = || {
                let mut g = GraphBuilder::new();
                let mat_buf = f_slice_const_buf(&mut g, device, "mat", &mat);
                let omega_buf = f_slice_const_buf(&mut g, device, "omega", &omega);
                let denom_buf = ef_slice_const_buf(&mut g, device, "denoms", &denoms);
                let out_buf = add_ef_buf(&mut g, device, "folded", out_len);
                fold_ple_from_evals_ir(
                    &mut g,
                    mat_buf,
                    height * width,
                    FoldPleDst::Fresh(out_buf),
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
                fold_ple_from_evals_ir(
                    &mut g,
                    mat_buf,
                    height * width,
                    FoldPleDst::InPlace(out_buf),
                    num_x * width,
                    omega_buf,
                    skip_domain,
                    denom_buf,
                    height as u32,
                    width as u32,
                    l_skip as u32,
                    num_x as u32,
                    true,
                );
                (g, out_buf)
            };
            let (g, out_buf) = build();

            // (1) no rename anywhere in the graph.
            assert!(
                g.aliases.iter().all(|a| a.is_none()),
                "the fold builder introduced a buffer rename; `alias_bufs` does not make two \
                 ids one allocation (notes.md:37-44)"
            );
            // (2) the rotated launch is a carried mutation of the SAME id.
            let folds: Vec<_> = g
                .nodes
                .iter()
                .filter_map(|n| match n {
                    GraphNode::BlackboxKernel(k) if k.name.starts_with("fold_ple_from_evals") => {
                        Some(k)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(folds.len(), 2, "expected two fold launches");
            assert_eq!(folds[0].outputs, vec![out_buf], "plain fold output");
            assert!(
                folds[0].carried_outputs.is_empty(),
                "plain fold must not carry"
            );
            assert!(
                folds[1].outputs.is_empty(),
                "rotated fold must declare no fresh output, got {:?}",
                folds[1].outputs
            );
            assert_eq!(
                folds[1].carried_outputs,
                vec![out_buf],
                "rotated fold must carry the plain fold's buffer"
            );

            // (3) the bytes, under both shipped schedulers.
            drop(g);
            for mode in [scheduler_v1(), scheduler_v2()] {
                let (g, out_buf) = build();
                let got = run_graph_read_bufs_with(mode, g, &[out_buf], &ctx).remove(0);
                assert_eq!(
                    &got[..],
                    ef_bytes(&want),
                    "rotated fold mismatch (l_skip={l_skip}, log_height={log_height}, \
                     width={width}); the lower half is written by the plain fold and the upper \
                     half by the rotated one — a mismatch confined to one half means they \
                     landed in different allocations"
                );
            }
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

    /// R6 at phase scale: the whole logup-zerocheck graph binds, and every
    /// offset it encoded lands inside the pool we handed the exe.
    ///
    /// This is the end-to-end shape of the contract — compile, own the pool,
    /// bind, and then check that `base + off` is in `[base, base + peak)` for
    /// every graph-owned descriptor field. An offset that escaped the pool
    /// would mean the encoding and the plan disagree.
    ///
    /// It deliberately does **not** `run`: the phase graph's `block_ctxs` and
    /// rule streams are zeroed placeholders in this fixture, so the batched
    /// evaluators would interpret garbage. What is under test here is the
    /// binding, not the arithmetic — the arithmetic oracle is
    /// `tests::test_monomial_vs_dag_equivalence`'s rebased differential.
    #[test]
    fn phase_graph_binds_and_encodes_inside_the_pool() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        let plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 2, /* n_max */ 3,
        );

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut inputs = PhaseInputBinder::new();
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_inputs(&mut g, device, &plan, t, &mut inputs))
            .collect();
        let proof =
            logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device, &mut inputs);
        let descs = proof.descriptors.clone();

        let mut exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("phase graph compile");
        let pool = DescriptorPlan::alloc_pool(&exe, &ctx);
        descs.bind(&mut exe, &ctx, pool).expect("descriptor bind");

        let base = descs.pool_base().get() as usize;
        let peak = exe.scratch_bytes();
        assert!(peak > 0, "empty pool");

        let mut fields = 0usize;
        for (buf, name, filled, total) in descs.array_summary() {
            assert_eq!(filled, total, "`{name}` ({buf:?}) is not fully filled");
            let off = exe.plan().offsets[buf.0].expect("descriptor array pool slot");
            assert!(
                (off as usize) < peak,
                "`{name}` ({buf:?}) sits at pool offset {off}, past the {peak}-byte pool"
            );
            // Every graph-owned field of every element must land in the pool.
            for b in descs.referenced_bufs(buf) {
                let boff = exe.plan().offsets[b.0].expect("referenced buffer pool slot");
                let addr = base.wrapping_add(boff as usize);
                assert!(
                    (base..base + peak).contains(&addr),
                    "`{name}` references {b:?} at {addr:#x}, outside the pool \
                     [{base:#x}, {:#x})",
                    base + peak
                );
                fields += 1;
            }
        }
        assert!(fields > 0, "no descriptor fields were checked");
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
        let mut inputs = PhaseInputBinder::new();
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_inputs(&mut g, device, &plan, t, &mut inputs))
            .collect();
        let _ = logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device, &mut inputs);
        GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("seeded phase graph compile");
    }

    // =======================================================================
    // The ring: `batch_s_ring_{pre,post}`, the composer, and the folds.
    // =======================================================================

    /// One trace of a ring fixture. `n` may be negative — that is the only
    /// case in which `norm_factor != 1` (`mod.rs:1144-1146`).
    #[derive(Clone, Copy, Debug)]
    struct RingTraceSpec {
        n: isize,
        has_constraints: bool,
        has_interactions: bool,
    }

    impl RingTraceSpec {
        fn n_lift(&self) -> usize {
            self.n.max(0) as usize
        }
    }

    /// One seeded steady round, with everything both sides consume.
    ///
    /// The raw evaluator values are stored *unnormalized*: normalization of
    /// the interaction numerator is the ring kernel's job, and making it the
    /// descriptor contract is what stops it from being applied twice.
    struct RingCase {
        round: usize,
        d: usize,
        l_skip: usize,
        specs: Vec<RingTraceSpec>,
        xi: Vec<EF>,
        mu_pows: Vec<EF>,
        /// `[3 * T]` in the `mu_pows` slot order.
        tilde_in: Vec<EF>,
        prev_s_eval: EF,
        eq_n: EF,
        eq_sharp_n: EF,
        r_prev: EF,
        r_round: EF,
        /// Raw constraint evaluations per trace: `d` values when early, one
        /// when late, none when exhausted or absent.
        zc_raw: Vec<Vec<EF>>,
        /// Raw interaction evaluations per trace, same shape.
        lg_raw: Vec<Vec<Frac<EF>>>,
    }

    /// Which of the three eager cases a trace is in this round.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RingPhase {
        Early,
        Late,
        Exhausted,
    }

    impl RingCase {
        fn new(seed: u64, round: usize, d: usize, specs: Vec<RingTraceSpec>) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            let t_n = specs.len();
            let l_skip = 1usize;
            let xi: Vec<EF> = (0..l_skip + round + 2)
                .map(|_| rng.random::<EF>())
                .collect();
            let mu_pows: Vec<EF> = (0..3 * t_n).map(|_| rng.random::<EF>()).collect();
            let mut tilde_in: Vec<EF> = (0..3 * t_n).map(|_| rng.random::<EF>()).collect();
            // A disabled family's tilde slot is ZERO in production
            // (`mod.rs:628-629` initializes both vectors to zero and nothing
            // ever writes a slot whose family is absent). The eager
            // `round == n_lift + 1` equality correction is unguarded
            // (`mod.rs:1467-1472`), so a nonzero disabled slot here would make
            // the fixture, not the kernel, the thing that disagrees.
            for (t, spec) in specs.iter().enumerate() {
                if !spec.has_interactions {
                    tilde_in[2 * t] = EF::ZERO;
                    tilde_in[2 * t + 1] = EF::ZERO;
                }
                if !spec.has_constraints {
                    tilde_in[2 * t_n + t] = EF::ZERO;
                }
            }
            let mut zc_raw = Vec::with_capacity(t_n);
            let mut lg_raw = Vec::with_capacity(t_n);
            for spec in &specs {
                let num_x = match ring_phase(round, spec) {
                    RingPhase::Early => d,
                    RingPhase::Late => 1,
                    RingPhase::Exhausted => 0,
                };
                zc_raw.push(if spec.has_constraints {
                    (0..num_x).map(|_| rng.random::<EF>()).collect()
                } else {
                    Vec::new()
                });
                lg_raw.push(if spec.has_interactions {
                    (0..num_x)
                        .map(|_| Frac::new(rng.random::<EF>(), rng.random::<EF>()))
                        .collect()
                } else {
                    Vec::new()
                });
            }
            Self {
                round,
                d,
                l_skip,
                specs,
                xi,
                mu_pows,
                tilde_in,
                prev_s_eval: rng.random(),
                eq_n: rng.random(),
                eq_sharp_n: rng.random(),
                r_prev: rng.random(),
                r_round: rng.random(),
                zc_raw,
                lg_raw,
            }
        }

        fn num_traces(&self) -> usize {
            self.specs.len()
        }

        fn s_deg(&self) -> usize {
            self.d + 1
        }

        /// `F::from_usize(1 << max(-n, 0)).inverse()` — `mod.rs:1144-1146`.
        fn norm(&self, t: usize) -> F {
            F::from_usize(1usize << (-self.specs[t].n).max(0)).inverse()
        }

        fn norm_factors(&self) -> Vec<F> {
            (0..self.num_traces()).map(|t| self.norm(t)).collect()
        }

        /// A `ZerocheckPhasePlan` whose *ring-relevant* fields match this case.
        ///
        /// Only `num_traces`, `constraint_degree` and each trace's
        /// `n` / `has_constraints` / `has_interactions` are read by the
        /// composer and the descriptor builder; the rest is `synthetic_plan`'s
        /// filler.
        fn plan(&self) -> ZerocheckPhasePlan {
            let mut plan = synthetic_plan(self.num_traces(), self.l_skip, self.round.max(1));
            plan.constraint_degree = self.d;
            plan.mu_pows = self.mu_pows.clone();
            for (t, spec) in self.specs.iter().enumerate() {
                plan.traces[t].n = spec.n;
                plan.traces[t].has_constraints = spec.has_constraints;
                plan.traces[t].has_interactions = spec.has_interactions;
            }
            plan
        }

        /// The eager reference: `(poly coefficients, s(1..=s_deg), tilde_out)`.
        ///
        /// The polynomial comes from the **production**
        /// `compute_batch_s_poly_from_state` (`mod.rs`), which is
        /// `LogupZerocheckGpu::compute_batch_s_poly`'s body verbatim — not a
        /// formula retyped for the test.
        fn host_pre(&self, sabotage: Option<(usize, bool)>) -> (Vec<EF>, Vec<EF>, Vec<EF>) {
            let t_n = self.num_traces();
            let mut zc_tilde: Vec<EF> = (0..t_n).map(|t| self.tilde_in[2 * t_n + t]).collect();
            let mut lg_tilde: Vec<[EF; 2]> = (0..t_n)
                .map(|t| [self.tilde_in[2 * t], self.tilde_in[2 * t + 1]])
                .collect();
            // Exactly what `sumcheck_polys_batch_eval` leaves behind
            // (`mod.rs:1107-1203`): a `3 * T` list of `d`-long vectors, the
            // numerator already normalized, late values installed into tilde,
            // exhausted tilde pre-scaled by `r_prev`.
            let mut sp: Vec<Vec<EF>> = vec![vec![EF::ZERO; self.d]; 3 * t_n];
            for (t, spec) in self.specs.iter().enumerate() {
                let norm = self.norm(t);
                let bump = |v: EF, is_zc: bool| match sabotage {
                    Some((st, sz)) if st == t && sz == is_zc => v + EF::ONE,
                    _ => v,
                };
                match ring_phase(self.round, spec) {
                    RingPhase::Early => {
                        if spec.has_constraints {
                            for (i, out) in sp[2 * t_n + t].iter_mut().enumerate() {
                                *out = bump(self.zc_raw[t][i], true);
                            }
                        }
                        if spec.has_interactions {
                            let (numer, rest) = sp[2 * t..].split_first_mut().unwrap();
                            let denom = &mut rest[0];
                            for (i, f) in self.lg_raw[t].iter().enumerate() {
                                numer[i] = bump(f.p, false) * norm;
                                denom[i] = f.q;
                            }
                        }
                    }
                    RingPhase::Late => {
                        if spec.has_constraints {
                            zc_tilde[t] = bump(self.zc_raw[t][0], true);
                        }
                        if spec.has_interactions {
                            lg_tilde[t][0] = bump(self.lg_raw[t][0].p, false) * norm;
                            lg_tilde[t][1] = self.lg_raw[t][0].q;
                        }
                    }
                    RingPhase::Exhausted => {
                        if spec.has_constraints {
                            zc_tilde[t] *= self.r_prev;
                        }
                        if spec.has_interactions {
                            for x in lg_tilde[t].iter_mut() {
                                *x *= self.r_prev;
                            }
                        }
                    }
                }
            }
            // `eq_ns.len() == round` and `last() == [round - 1]` at this point
            // (`mod.rs:1083-1088`, `mod.rs:1606-1615`), which is exactly why
            // the ring carries ONE of each.
            let eq_ns = vec![self.eq_n; self.round];
            let eq_sharp_ns = vec![self.eq_sharp_n; self.round];
            let poly = compute_batch_s_poly_from_state(
                BatchSPolyState {
                    constraint_degree: self.d,
                    l_skip: self.l_skip,
                    xi: &self.xi,
                    prev_s_eval: self.prev_s_eval,
                    n_per_trace: &self.specs.iter().map(|s| s.n).collect_vec(),
                    eq_ns: &eq_ns,
                    eq_sharp_ns: &eq_sharp_ns,
                    zerocheck_tilde_evals: &mut zc_tilde,
                    logup_tilde_evals: &mut lg_tilde,
                },
                sp,
                t_n,
                self.round,
                &self.mu_pows,
            );
            let coeffs = poly.coeffs().to_vec();
            let s_evals = (1..=self.s_deg())
                .map(|i| poly.eval_at_point(EF::from_usize(i)))
                .collect_vec();
            let mut tilde_out = vec![EF::ZERO; 3 * t_n];
            for t in 0..t_n {
                tilde_out[2 * t] = lg_tilde[t][0];
                tilde_out[2 * t + 1] = lg_tilde[t][1];
                tilde_out[2 * t_n + t] = zc_tilde[t];
            }
            (coeffs, s_evals, tilde_out)
        }

        /// `[s(r), eq_n * eq_r, eq_sharp_n * eq_r]` — `mod.rs:401`,
        /// `mod.rs:1612-1614`.
        fn host_post(&self, coeffs: &[EF], r_round: EF) -> Vec<EF> {
            let poly = UnivariatePoly::new(coeffs.to_vec());
            let eq_r: EF = eval_eq_mle(&[self.xi[self.l_skip + self.round - 1]], &[r_round]);
            vec![
                poly.eval_at_point(r_round),
                self.eq_n * eq_r,
                self.eq_sharp_n * eq_r,
            ]
        }
    }

    fn ring_phase(round: usize, spec: &RingTraceSpec) -> RingPhase {
        let n_lift = spec.n_lift();
        if round <= n_lift {
            RingPhase::Early
        } else if round == n_lift + 1 {
            RingPhase::Late
        } else {
            RingPhase::Exhausted
        }
    }

    /// One compact evaluator batch of a fixture, in `air` order.
    struct RingBatchSpec {
        family: RoundEvalFamily,
        num_x: usize,
        /// Trace indices in batch order — deliberately not identity.
        order: Vec<usize>,
    }

    /// The four batches the *fixed* stage-D dispatch will emit: early and late
    /// are separate launches with different `num_x` (S1.4-a), one per family.
    ///
    /// The order inside each batch is reversed so that `air != trace_idx`: a
    /// descriptor builder that used the trace index as the batch position
    /// would pass with identity ordering and fail here.
    fn ring_batches(case: &RingCase) -> Vec<RingBatchSpec> {
        let mut out = Vec::new();
        for (phase, num_x) in [(RingPhase::Early, case.d), (RingPhase::Late, 1)] {
            for family in [RoundEvalFamily::Zerocheck, RoundEvalFamily::Logup] {
                let order: Vec<usize> = (0..case.num_traces())
                    .rev()
                    .filter(|&t| {
                        ring_phase(case.round, &case.specs[t]) == phase
                            && match family {
                                RoundEvalFamily::Zerocheck => case.specs[t].has_constraints,
                                RoundEvalFamily::Logup => case.specs[t].has_interactions,
                            }
                    })
                    .collect();
                if !order.is_empty() {
                    out.push(RingBatchSpec {
                        family,
                        num_x,
                        order,
                    });
                }
            }
        }
        out
    }

    /// Flatten one batch's raw values, optionally perturbing one trace's
    /// contribution by `EF::ONE` (the sabotage leg).
    fn ring_batch_zc_host(case: &RingCase, b: &RingBatchSpec, sab: Option<usize>) -> Vec<EF> {
        b.order
            .iter()
            .flat_map(|&t| {
                case.zc_raw[t]
                    .iter()
                    .map(move |&v| if sab == Some(t) { v + EF::ONE } else { v })
            })
            .collect()
    }

    fn ring_batch_lg_host(case: &RingCase, b: &RingBatchSpec, sab: Option<usize>) -> Vec<Frac<EF>> {
        b.order
            .iter()
            .flat_map(|&t| {
                case.lg_raw[t].iter().map(move |f| {
                    if sab == Some(t) {
                        Frac::new(f.p + EF::ONE, f.q)
                    } else {
                        *f
                    }
                })
            })
            .collect()
    }

    /// The fixtures: both production degrees, `T` of 1 / 2 / 5, permuted
    /// batches, both / one / neither family, a mixed early+late+exhausted
    /// round, and the `n < 0` normalization.
    fn ring_cases() -> Vec<(&'static str, RingCase)> {
        let both = |n: isize| RingTraceSpec {
            n,
            has_constraints: true,
            has_interactions: true,
        };
        let zc_only = |n: isize| RingTraceSpec {
            n,
            has_constraints: true,
            has_interactions: false,
        };
        let lg_only = |n: isize| RingTraceSpec {
            n,
            has_constraints: false,
            has_interactions: true,
        };
        let neither = |n: isize| RingTraceSpec {
            n,
            has_constraints: false,
            has_interactions: false,
        };
        vec![
            ("d3_T1_early", RingCase::new(0x11, 1, 3, vec![both(3)])),
            (
                "d4_T2_early",
                RingCase::new(0x22, 2, 4, vec![both(4), zc_only(5)]),
            ),
            (
                "d3_T5_mixed",
                RingCase::new(
                    0x33,
                    3,
                    3,
                    vec![both(5), both(2), both(1), zc_only(4), lg_only(3)],
                ),
            ),
            (
                "d4_T3_late_and_exhausted",
                RingCase::new(0x44, 4, 4, vec![both(3), lg_only(2), both(6)]),
            ),
            (
                "d3_T2_negative_n_late",
                RingCase::new(0x55, 1, 3, vec![both(-2), both(2)]),
            ),
            (
                "d3_T2_family_free_trace",
                RingCase::new(0x66, 2, 3, vec![neither(3), both(3)]),
            ),
        ]
    }

    /// Step 1 — the shared descriptor ABI, Rust mirror vs the C++ definition.
    #[test]
    fn batch_s_ring_trace_desc_layout_matches_cuda() {
        crate::cuda::logup_zerocheck::assert_batch_s_ring_abi_matches_cuda();
        // Spelled out here too, so a silent change on BOTH sides still has to
        // get past a literal.
        assert_eq!(size_of::<BatchSRingTraceDesc>(), 24);
        assert_eq!(align_of::<BatchSRingTraceDesc>(), 8);
        assert_eq!(offset_of!(BatchSRingTraceDesc, zc_evals), 0);
        assert_eq!(offset_of!(BatchSRingTraceDesc, logup_evals), 8);
        assert_eq!(offset_of!(BatchSRingTraceDesc, n_lift), 16);
        assert_eq!(offset_of!(BatchSRingTraceDesc, flags), 20);
        assert_eq!(BATCH_S_RING_HAS_CONSTRAINTS, 1);
        assert_eq!(BATCH_S_RING_HAS_INTERACTIONS, 2);
    }

    /// Step 1 — the descriptor array addresses the *permuted* compact batches,
    /// nulls the absent families, and declares every buffer it points into.
    #[test]
    fn batch_s_ring_descriptor_offsets_match_compact_batches() {
        let ctx = test_ctx();
        let device = DeviceType::Cuda(0);
        for (name, case) in ring_cases() {
            let plan = case.plan();
            let batches_spec = ring_batches(&case);
            let mut g = GraphBuilder::new();
            let mut descs = DescriptorPlan::new();

            // Stand-in producers for the evaluator outputs; only their
            // addresses matter here.
            let mut batches = Vec::new();
            for (i, b) in batches_spec.iter().enumerate() {
                let evals = match b.family {
                    RoundEvalFamily::Zerocheck => ef_slice_const_buf(
                        &mut g,
                        device,
                        &format!("zc{i}"),
                        &ring_batch_zc_host(&case, b, None),
                    ),
                    RoundEvalFamily::Logup => {
                        let host = ring_batch_lg_host(&case, b, None);
                        let buf = add_frac_buf(&mut g, device, &format!("lg{i}"), host.len());
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                host.as_ptr() as *const u8,
                                std::mem::size_of_val(&host[..]),
                            )
                            .to_vec()
                        };
                        g.insert_const(buf, ConstBuf::HostBuf(bytes));
                        buf
                    }
                };
                batches.push(RoundEvalBatchIr {
                    family: b.family,
                    evals,
                    traces: b.order.clone(),
                    num_x: b.num_x,
                });
            }
            let (desc_buf, reads) = build_batch_s_ring_trace_descs(
                &mut g,
                &mut descs,
                device,
                "ring_descs",
                &plan,
                case.round,
                &batches,
            );

            // Every buffer a descriptor points into must be in the declared
            // read set — that list is what the consuming blackbox declares.
            for b in &batches {
                let referenced = case.specs.iter().enumerate().any(|(t, spec)| {
                    b.traces.contains(&t)
                        && match b.family {
                            RoundEvalFamily::Zerocheck => spec.has_constraints,
                            RoundEvalFamily::Logup => spec.has_interactions,
                        }
                });
                assert_eq!(
                    referenced,
                    reads.contains(&b.evals),
                    "{name}: batch {:?} declared-read mismatch",
                    b.family
                );
            }
            assert_eq!(
                reads,
                descs.referenced_bufs(desc_buf)[1..].to_vec(),
                "{name}: the returned reads must be the descriptor's dereference closure"
            );

            // Now bind for real and check the decoded addresses. The ring's
            // `pre` node is inserted so the descriptor array and the compact
            // batches have a real consumer — a registered input nothing reads
            // is rejected outright (`graph_compiler.rs:1236-1266`), and an
            // unread const would have no pool slot for its offset to name.
            let tilde_in = ef_slice_const_buf(&mut g, device, "tilde_in", &case.tilde_in);
            let mu = ef_slice_const_buf(&mut g, device, "mu", &case.mu_pows);
            let norm = f_slice_const_buf(&mut g, device, "norm", &case.norm_factors());
            let scalars = ef_slice_const_buf(
                &mut g,
                device,
                "scalars",
                &[case.prev_s_eval, case.eq_n, case.eq_sharp_n],
            );
            let xi_j = ef_const_ext_scalar_buf(&mut g, device, "xi", case.xi[case.l_skip]);
            let r_prev = ef_const_ext_scalar_buf(&mut g, device, "r_prev", case.r_prev);
            let (tilde_out, _, _) = batch_s_ring_pre_ir(
                &mut g,
                desc_buf,
                &reads,
                descs.pool_base().clone(),
                tilde_in,
                mu,
                norm,
                scalars,
                xi_j,
                r_prev,
                case.num_traces(),
                case.d,
                case.round,
                device,
            );
            // Export through a fresh copy rather than pinning the descriptor
            // input itself.
            let desc_copy = add_typed_buf::<BatchSRingTraceDesc>(
                &mut g,
                device,
                "desc_copy",
                case.num_traces(),
            );
            g.insert_memcpy(desc_buf, desc_copy);
            let bytes = run_graph_with_descs(g, &descs, &[desc_copy, tilde_out], &ctx).remove(0);
            let decoded: &[BatchSRingTraceDesc] = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const BatchSRingTraceDesc,
                    case.num_traces(),
                )
            };
            for (t, spec) in case.specs.iter().enumerate() {
                assert_eq!(
                    decoded[t].n_lift as usize,
                    spec.n_lift(),
                    "{name}: n_lift[{t}]"
                );
                let want_flags = (u32::from(spec.has_constraints) * BATCH_S_RING_HAS_CONSTRAINTS)
                    | (u32::from(spec.has_interactions) * BATCH_S_RING_HAS_INTERACTIONS);
                assert_eq!(decoded[t].flags, want_flags, "{name}: flags[{t}]");
                let phase = ring_phase(case.round, spec);
                for (family, got) in [
                    (RoundEvalFamily::Zerocheck, decoded[t].zc_evals),
                    (RoundEvalFamily::Logup, decoded[t].logup_evals),
                ] {
                    let enabled = match family {
                        RoundEvalFamily::Zerocheck => spec.has_constraints,
                        RoundEvalFamily::Logup => spec.has_interactions,
                    };
                    if !enabled || phase == RingPhase::Exhausted {
                        assert_eq!(
                            got,
                            BaseOff::NULL,
                            "{name}: trace {t} family {family:?} must be the absent encoding"
                        );
                        continue;
                    }
                    assert_ne!(
                        got,
                        BaseOff::NULL,
                        "{name}: trace {t} family {family:?} must carry an address"
                    );
                }
            }
            // Distinct traces in the same batch must decode to distinct,
            // correctly strided addresses.
            for b in &batches_spec {
                let stride = match b.family {
                    RoundEvalFamily::Zerocheck => b.num_x * size_of::<EF>(),
                    RoundEvalFamily::Logup => b.num_x * size_of::<Frac<EF>>(),
                } as u64;
                let off = |t: usize| match b.family {
                    RoundEvalFamily::Zerocheck => decoded[t].zc_evals.0,
                    RoundEvalFamily::Logup => decoded[t].logup_evals.0,
                };
                for (air, &t) in b.order.iter().enumerate() {
                    assert_eq!(
                        off(t),
                        off(b.order[0]) + air as u64 * stride,
                        "{name}: trace {t} sits at batch position {air}, so its offset must be \
                         the batch base plus {air} * {stride}"
                    );
                }
            }
        }
    }

    /// Upload a case's batches eagerly and build the eager (absolute-address,
    /// null-pool-base) descriptor array the CUDA launcher decodes.
    ///
    /// Returns the descriptors plus the device buffers they point at, which
    /// the caller must keep alive across the launch.
    #[allow(clippy::type_complexity)]
    fn eager_ring_descs(
        case: &RingCase,
        sabotage: Option<(usize, bool)>,
        ctx: &GpuDeviceCtx,
    ) -> (
        Vec<BatchSRingTraceDesc>,
        Vec<DeviceBuffer<EF>>,
        Vec<DeviceBuffer<Frac<EF>>>,
    ) {
        let mut descs: Vec<BatchSRingTraceDesc> = case
            .specs
            .iter()
            .map(|spec| BatchSRingTraceDesc {
                zc_evals: BaseOff::NULL,
                logup_evals: BaseOff::NULL,
                n_lift: spec.n_lift() as u32,
                flags: (u32::from(spec.has_constraints) * BATCH_S_RING_HAS_CONSTRAINTS)
                    | (u32::from(spec.has_interactions) * BATCH_S_RING_HAS_INTERACTIONS),
            })
            .collect();
        let mut zc_bufs = Vec::new();
        let mut lg_bufs = Vec::new();
        for b in ring_batches(case) {
            match b.family {
                RoundEvalFamily::Zerocheck => {
                    let sab = sabotage.and_then(|(t, is_zc)| is_zc.then_some(t));
                    let host = ring_batch_zc_host(case, &b, sab);
                    let dev = host.as_slice().to_device_on(ctx).expect("H2D zc batch");
                    for (air, &t) in b.order.iter().enumerate() {
                        descs[t].zc_evals =
                            BaseOff::from_ptr(dev.as_ptr().wrapping_add(air * b.num_x));
                    }
                    zc_bufs.push(dev);
                }
                RoundEvalFamily::Logup => {
                    let sab = sabotage.and_then(|(t, is_zc)| (!is_zc).then_some(t));
                    let host = ring_batch_lg_host(case, &b, sab);
                    let dev = host.as_slice().to_device_on(ctx).expect("H2D lg batch");
                    for (air, &t) in b.order.iter().enumerate() {
                        descs[t].logup_evals =
                            BaseOff::from_ptr(dev.as_ptr().wrapping_add(air * b.num_x));
                    }
                    lg_bufs.push(dev);
                }
            }
        }
        (descs, zc_bufs, lg_bufs)
    }

    /// Run `batch_s_ring_pre` + `batch_s_ring_post` eagerly on one case.
    ///
    /// Returns `(tilde_out, poly_coeffs, s_evals, scalar_state_out)`.
    #[allow(clippy::type_complexity)]
    fn run_eager_ring(
        case: &RingCase,
        sabotage: Option<(usize, bool)>,
        ctx: &GpuDeviceCtx,
    ) -> (Vec<EF>, Vec<EF>, Vec<EF>, Vec<EF>) {
        let stream = ctx.stream.as_raw();
        let t_n = case.num_traces();
        let (descs, _zc_keep, _lg_keep) = eager_ring_descs(case, sabotage, ctx);
        let d_descs = descs.as_slice().to_device_on(ctx).expect("H2D descs");
        let d_tilde_in = case
            .tilde_in
            .as_slice()
            .to_device_on(ctx)
            .expect("H2D tilde");
        let d_mu = case.mu_pows.as_slice().to_device_on(ctx).expect("H2D mu");
        let d_norm = case
            .norm_factors()
            .as_slice()
            .to_device_on(ctx)
            .expect("H2D norm");
        let d_state_in = [case.prev_s_eval, case.eq_n, case.eq_sharp_n]
            .as_slice()
            .to_device_on(ctx)
            .expect("H2D state");
        let d_xi = [case.xi[case.l_skip + case.round - 1]]
            .as_slice()
            .to_device_on(ctx)
            .expect("H2D xi");
        let d_r_prev = [case.r_prev]
            .as_slice()
            .to_device_on(ctx)
            .expect("H2D r_prev");
        let d_r_round = [case.r_round]
            .as_slice()
            .to_device_on(ctx)
            .expect("H2D r_round");
        let d_tilde_out = DeviceBuffer::<EF>::with_capacity_on(3 * t_n, ctx);
        let d_coeffs = DeviceBuffer::<EF>::with_capacity_on(case.d + 2, ctx);
        let d_s_evals = DeviceBuffer::<EF>::with_capacity_on(case.d + 1, ctx);
        let d_state_out = DeviceBuffer::<EF>::with_capacity_on(3, ctx);
        unsafe {
            batch_s_ring_pre(
                d_descs.as_ptr(),
                std::ptr::null(),
                d_tilde_in.as_ptr(),
                d_mu.as_ptr(),
                d_norm.as_ptr(),
                d_state_in.as_ptr(),
                d_xi.as_ptr(),
                d_r_prev.as_ptr(),
                d_tilde_out.as_mut_ptr(),
                d_coeffs.as_mut_ptr(),
                d_s_evals.as_mut_ptr(),
                t_n as u32,
                case.d as u32,
                case.round as u32,
                stream,
            )
            .expect("batch_s_ring_pre");
            batch_s_ring_post(
                d_coeffs.as_ptr(),
                d_state_in.as_ptr(),
                d_xi.as_ptr(),
                d_r_round.as_ptr(),
                d_state_out.as_mut_ptr(),
                case.d as u32,
                stream,
            )
            .expect("batch_s_ring_post");
        }
        ctx.stream.synchronize().expect("sync");
        (
            d_tilde_out.to_host_on(ctx).expect("D2H tilde"),
            d_coeffs.to_host_on(ctx).expect("D2H coeffs"),
            d_s_evals.to_host_on(ctx).expect("D2H s_evals"),
            d_state_out.to_host_on(ctx).expect("D2H state"),
        )
    }

    /// Step 2 — the two ring kernels reproduce `compute_batch_s_poly` and its
    /// caller's post-sample updates, byte for byte, on every seeded shape.
    #[test]
    fn batch_s_ring_pre_post_matches_eager() {
        let ctx = test_ctx();
        for (name, case) in ring_cases() {
            let (want_coeffs, want_s, want_tilde) = case.host_pre(None);
            let want_state = case.host_post(&want_coeffs, case.r_round);
            let (got_tilde, got_coeffs, got_s, got_state) = run_eager_ring(&case, None, &ctx);
            assert_eq!(
                ef_bytes(&got_coeffs),
                ef_bytes(&want_coeffs),
                "{name}: coeffs"
            );
            assert_eq!(ef_bytes(&got_s), ef_bytes(&want_s), "{name}: s(1..=s_deg)");
            assert_eq!(ef_bytes(&got_tilde), ef_bytes(&want_tilde), "{name}: tilde");
            assert_eq!(
                ef_bytes(&got_state),
                ef_bytes(&want_state),
                "{name}: [s(r), eq_n', eq_sharp_n']"
            );
        }
    }

    /// Step 2 — the oracle has teeth: perturbing ONE raw evaluator value on
    /// the device side, after the eager bytes are frozen, must change the
    /// result.
    ///
    /// Both families are perturbed in turn, including a numerator (which is
    /// the one value the ring normalizes) so a dropped `norm_factor` cannot
    /// hide behind an untested path.
    #[test]
    fn batch_s_ring_oracle_detects_sabotage() {
        let ctx = test_ctx();
        for (name, case) in ring_cases() {
            let (want_coeffs, want_s, want_tilde) = case.host_pre(None);
            for t in 0..case.num_traces() {
                for is_zc in [true, false] {
                    let spec = case.specs[t];
                    let enabled = if is_zc {
                        spec.has_constraints
                    } else {
                        spec.has_interactions
                    };
                    if !enabled || ring_phase(case.round, &spec) == RingPhase::Exhausted {
                        continue;
                    }
                    let sab = Some((t, is_zc));
                    // The host reference under the SAME perturbation must
                    // differ from the frozen one; otherwise the fixture, not
                    // the kernel, is blind.
                    let (host_coeffs, host_s, host_tilde) = case.host_pre(sab);
                    assert!(
                        host_coeffs != want_coeffs || host_s != want_s || host_tilde != want_tilde,
                        "{name}: perturbing trace {t} (zc={is_zc}) does not move the EAGER \
                         result — the fixture cannot detect anything here"
                    );
                    let (got_tilde, got_coeffs, got_s, _) = run_eager_ring(&case, sab, &ctx);
                    assert!(
                        ef_bytes(&got_coeffs) != ef_bytes(&want_coeffs)
                            || ef_bytes(&got_s) != ef_bytes(&want_s)
                            || ef_bytes(&got_tilde) != ef_bytes(&want_tilde),
                        "SABOTAGE LEG IS BLIND: {name}, trace {t} (zc={is_zc}) — a corrupted \
                         evaluator value produced the frozen eager bytes"
                    );
                    // ...and the corrupted device result must still equal the
                    // corrupted host result, which is what makes the pass
                    // meaningful rather than merely different.
                    assert_eq!(
                        ef_bytes(&got_coeffs),
                        ef_bytes(&host_coeffs),
                        "{name}: sabotaged device coeffs must track the sabotaged host coeffs"
                    );
                }
            }
        }
    }

    /// Wire one case's compact batches into a graph and run the composer.
    ///
    /// Returns `(observed s-evals, sampled r, tilde_out, scalar_state_out,
    /// coeffs, final sponge state)` on the graph side and the eager side.
    #[allow(clippy::type_complexity)]
    struct RingRoundGraph {
        s_evals: Vec<EF>,
        r: EF,
        tilde: Vec<EF>,
        state: Vec<EF>,
        coeffs: Vec<EF>,
        sponge: Vec<u8>,
        /// Folded outputs, when the caller asked for folds.
        folded: Vec<Vec<EF>>,
        /// Node count of the composer plus the folds, and the blackbox names
        /// it inserted.
        nodes: usize,
        node_names: Vec<String>,
    }

    /// Ragged fold shapes, matching the low-level device-challenge ABI test
    /// (`cuda/mod.rs:358`): unequal heights, unequal widths, one height-1
    /// output.
    const RING_FOLD_SHAPES: [(usize, usize); 3] = [(8, 3), (4, 5), (2, 7)];

    /// Host `batch_fold_mle`, column-major with ADJACENT pairing:
    /// `out[col][row] = t0 + r * (t1 - t0)` for `t0 = in[col][2 * row]`,
    /// `t1 = in[col][2 * row + 1]`.
    ///
    /// Same reference as `cuda::sumcheck::dev_challenge_tests::host_fold`
    /// (`cuda/mod.rs:329-340`), which the already-green low-level ABI test
    /// pins the kernel against.
    fn host_fold_matrix(input: &[EF], height: usize, width: usize, r: EF) -> Vec<EF> {
        let out_h = height >> 1;
        let mut out = Vec::with_capacity(out_h * width);
        for col in 0..width {
            for row in 0..out_h {
                let t0 = input[col * height + 2 * row];
                let t1 = input[col * height + 2 * row + 1];
                out.push(t0 + r * (t1 - t0));
            }
        }
        out
    }

    /// Build + run one round of the ring as graph nodes, optionally followed
    /// by the two `batch_fold_mle` launches driven by the sampled challenge.
    fn run_ring_round_graph(
        case: &RingCase,
        sabotage: Option<(usize, bool)>,
        fold_inputs: Option<&[Vec<EF>]>,
        snap: &crate::sponge::SpongeSnapshot,
        ctx: &GpuDeviceCtx,
    ) -> RingRoundGraph {
        let device = DeviceType::Cuda(0);
        let plan = case.plan();
        let t_n = case.num_traces();
        let mut g = GraphBuilder::new();
        let (mut transcript, seed_buf) = DuplexSpongeGpuIR::from_live_input(&mut g, device, snap);
        let mut descs = DescriptorPlan::new();

        let mut batches = Vec::new();
        for (i, b) in ring_batches(case).iter().enumerate() {
            let evals = match b.family {
                RoundEvalFamily::Zerocheck => {
                    let sab = sabotage.and_then(|(t, is_zc)| is_zc.then_some(t));
                    ef_slice_const_buf(
                        &mut g,
                        device,
                        &format!("zc{i}"),
                        &ring_batch_zc_host(case, b, sab),
                    )
                }
                RoundEvalFamily::Logup => {
                    let sab = sabotage.and_then(|(t, is_zc)| (!is_zc).then_some(t));
                    let host = ring_batch_lg_host(case, b, sab);
                    let buf = add_frac_buf(&mut g, device, &format!("lg{i}"), host.len());
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            host.as_ptr() as *const u8,
                            std::mem::size_of_val(&host[..]),
                        )
                        .to_vec()
                    };
                    g.insert_const(buf, ConstBuf::HostBuf(bytes));
                    buf
                }
            };
            batches.push(RoundEvalBatchIr {
                family: b.family,
                evals,
                traces: b.order.clone(),
                num_x: b.num_x,
            });
        }

        let mu = ef_slice_const_buf(&mut g, device, "mu_pows", &case.mu_pows);
        let norm = f_slice_const_buf(&mut g, device, "norm", &case.norm_factors());
        let xi_j = ef_const_ext_scalar_buf(
            &mut g,
            device,
            "xi_j",
            case.xi[case.l_skip + case.round - 1],
        );
        let r_prev = ef_const_ext_scalar_buf(&mut g, device, "r_prev", case.r_prev);
        let tilde_in = ef_slice_const_buf(&mut g, device, "tilde_in", &case.tilde_in);
        let scalars_in = ef_slice_const_buf(
            &mut g,
            device,
            "scalars_in",
            &[case.prev_s_eval, case.eq_n, case.eq_sharp_n],
        );

        // Fold *setup* (const matrices, pointer tables, shape tables) is
        // hoisted out of the node-count window: the budget is about the ring
        // and the two launches, not about how a test fixture materializes its
        // matrices. Two groups, mirroring the driver's matrix + selector
        // folds.
        struct FoldGroup {
            srcs: Vec<BufId>,
            dsts: Vec<BufId>,
            in_ptrs: BufId,
            out_ptrs: BufId,
            widths: BufId,
            logh: BufId,
            max_cells: u32,
        }
        let mut fold_groups: Vec<FoldGroup> = Vec::new();
        if let Some(mats) = fold_inputs {
            for tag in ["mats", "sels"] {
                let srcs: Vec<BufId> = mats
                    .iter()
                    .enumerate()
                    .map(|(i, m)| ef_slice_const_buf(&mut g, device, &format!("{tag}_src{i}"), m))
                    .collect();
                let dsts: Vec<BufId> = RING_FOLD_SHAPES
                    .iter()
                    .enumerate()
                    .map(|(i, &(h, w))| {
                        add_ef_buf(&mut g, device, &format!("{tag}_dst{i}"), (h / 2) * w)
                    })
                    .collect();
                let widths_h: Vec<u32> = RING_FOLD_SHAPES.iter().map(|&(_, w)| w as u32).collect();
                let logh_h: Vec<u8> = RING_FOLD_SHAPES
                    .iter()
                    .map(|&(h, _)| ((h / 2) as u32).ilog2() as u8)
                    .collect();
                let max_cells = RING_FOLD_SHAPES
                    .iter()
                    .map(|&(h, w)| ((h / 2) * w) as u32)
                    .max()
                    .unwrap();
                let (in_ptrs, out_ptrs) =
                    fold_ptr_tables(&mut g, &mut descs, device, tag, &srcs, &dsts);
                let widths = typed_slice_const_buf(&mut g, device, &format!("{tag}_w"), &widths_h);
                let logh = typed_slice_const_buf(&mut g, device, &format!("{tag}_lh"), &logh_h);
                fold_groups.push(FoldGroup {
                    srcs,
                    dsts,
                    in_ptrs,
                    out_ptrs,
                    widths,
                    logh,
                    max_cells,
                });
            }
        }

        let nodes_before = g.nodes.len();
        let ring = observe_and_update_zerocheck_round_ir(
            &mut g,
            &mut transcript,
            &mut descs,
            &plan,
            &batches,
            mu,
            norm,
            xi_j,
            r_prev,
            ZerocheckRoundStateIr {
                tilde: tilde_in,
                scalars: scalars_in,
            },
            case.round,
            device,
        );

        // The two folds, driven by the SAME sampled buffer the ring produced.
        let mut fold_dsts: Vec<BufId> = Vec::new();
        for fg in &fold_groups {
            batch_fold_mle_ir(
                &mut g,
                fg.in_ptrs,
                fg.out_ptrs,
                fg.widths,
                fg.logh,
                &fg.srcs,
                &fg.dsts,
                RING_FOLD_SHAPES.len() as u16,
                fg.max_cells,
                ring.r_round,
            );
            fold_dsts.extend(fg.dsts.iter().copied());
        }
        let nodes_after = g.nodes.len();
        let node_names: Vec<String> = g.nodes[nodes_before..nodes_after]
            .iter()
            .filter_map(|n| match n {
                GraphNode::BlackboxKernel(k) => Some(k.name.clone()),
                _ => None,
            })
            .collect();

        // Every artifact is exported through a FRESH memcpy: exporting an
        // internal producer directly can change fusion and liveness, so the
        // oracle would no longer be measuring the same graph.
        let s_outs: Vec<BufId> = ring
            .s_evals
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let o = add_ext_scalar_buf(&mut g, device, &format!("s_out{i}"));
                g.insert_memcpy(b, o);
                o
            })
            .collect();
        let r_out = add_ext_scalar_buf(&mut g, device, "r_out");
        g.insert_memcpy(ring.r_round, r_out);
        let tilde_out = add_ef_buf(&mut g, device, "tilde_out", 3 * t_n);
        g.insert_memcpy(ring.state.tilde, tilde_out);
        let state_out = add_ef_buf(&mut g, device, "state_out", 3);
        g.insert_memcpy(ring.state.scalars, state_out);
        let coeffs_out = add_ef_buf(&mut g, device, "coeffs_out", case.d + 2);
        g.insert_memcpy(ring.poly_coeffs, coeffs_out);
        let sponge_out = transcript.state_buf();

        let mut wanted = s_outs.clone();
        wanted.extend([r_out, tilde_out, state_out, coeffs_out, sponge_out]);
        wanted.extend(fold_dsts.iter().copied());

        for &b in &wanted {
            g.register_output(b);
        }
        let mut exe = GraphCompiler::new()
            .device(device)
            .scheduler(scheduler_v1())
            .compile(g)
            .expect("ring graph compile");
        let pool = DescriptorPlan::alloc_pool(&exe, ctx);
        descs.bind(&mut exe, ctx, pool).expect("descriptor bind");
        crate::sponge_graph_ir::bind_sponge_seed(&mut exe, ctx, seed_buf, snap)
            .expect("bind_sponge_seed");
        exe.run(ctx).expect("ring graph run");
        let read = |b: BufId| -> Vec<u8> {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == b)
                .expect("output buf");
            exe.get_output(idx).to_host_on(ctx).expect("D2H")
        };
        let ef_of = |bytes: &[u8]| -> Vec<EF> {
            unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const EF,
                    bytes.len() / size_of::<EF>(),
                )
                .to_vec()
            }
        };
        RingRoundGraph {
            s_evals: s_outs.iter().map(|&b| ef_of(&read(b))[0]).collect(),
            r: ef_of(&read(r_out))[0],
            tilde: ef_of(&read(tilde_out)),
            state: ef_of(&read(state_out)),
            coeffs: ef_of(&read(coeffs_out)),
            sponge: read(sponge_out),
            folded: fold_dsts.iter().map(|&b| ef_of(&read(b))).collect(),
            nodes: nodes_after - nodes_before,
            node_names,
        }
    }

    /// A live sponge that has already absorbed, so the graph transcript is
    /// seeded mid-stream exactly as the phase driver seeds it.
    ///
    /// No grinding happens anywhere in these fixtures — the transcript is
    /// driven only through `observe_ext` / `sample_ext`, and both PoW paths
    /// (`transcript/traits.rs:83-86`, `sponge.cu:80-86`) are only reachable
    /// through `grind`, which is never called. There is therefore no
    /// nondeterminism for `pow_bits = 0` to remove here.
    fn seeded_sponge(seed: u64) -> (DuplexSpongeGpu, crate::sponge::SpongeSnapshot) {
        use openvm_stark_backend::FiatShamirTranscript;
        let mut live = DuplexSpongeGpu::default();
        for i in 0..(3 + seed % 5) as u32 {
            FiatShamirTranscript::<SC>::observe(&mut live, F::from_u32(i + 1));
        }
        let snap = live.snapshot();
        (live, snap)
    }

    /// The eager tail of one steady round: observe `s(1..=s_deg)`, sample `r`.
    fn host_observe_and_sample(sponge: &mut DuplexSpongeGpu, s_evals: &[EF]) -> EF {
        use openvm_stark_backend::FiatShamirTranscript;
        for &e in s_evals {
            FiatShamirTranscript::<SC>::observe_ext(sponge, e);
        }
        FiatShamirTranscript::<SC>::sample_ext(sponge)
    }

    /// Step 3 — the composer: observed values, sampled challenge, next state
    /// and the final sponge state all agree with the eager path, on bytes.
    #[test]
    fn observe_and_update_zerocheck_round_ir_matches_eager() {
        let ctx = test_ctx();
        for (i, (name, case)) in ring_cases().into_iter().enumerate() {
            let (mut sponge, snap) = seeded_sponge(i as u64);
            let (want_coeffs, want_s, want_tilde) = case.host_pre(None);
            let want_r = host_observe_and_sample(&mut sponge, &want_s);
            let want_state = case.host_post(&want_coeffs, want_r);
            let want_sponge = sponge.snapshot();

            let got = run_ring_round_graph(&case, None, None, &snap, &ctx);
            assert_eq!(
                ef_bytes(&got.coeffs),
                ef_bytes(&want_coeffs),
                "{name}: coeffs"
            );
            assert_eq!(
                ef_bytes(&got.s_evals),
                ef_bytes(&want_s),
                "{name}: observed s-evals"
            );
            assert_eq!(ef_bytes(&[got.r]), ef_bytes(&[want_r]), "{name}: sampled r");
            assert_eq!(ef_bytes(&got.tilde), ef_bytes(&want_tilde), "{name}: tilde");
            assert_eq!(
                ef_bytes(&got.state),
                ef_bytes(&want_state),
                "{name}: scalar state"
            );
            assert_eq!(
                got.sponge,
                sponge_state_bytes(&want_sponge),
                "{name}: final sponge state — the transcript diverged even though the values did not"
            );
        }
    }

    fn sponge_state_bytes(snap: &crate::sponge::SpongeSnapshot) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                snap.state().as_ptr() as *const u8,
                std::mem::size_of_val(snap.state()),
            )
            .to_vec()
        }
    }

    /// Random ragged fold inputs for [`RING_FOLD_SHAPES`].
    fn ring_fold_inputs(seed: u64) -> Vec<Vec<EF>> {
        let mut rng = StdRng::seed_from_u64(seed);
        RING_FOLD_SHAPES
            .iter()
            .map(|&(h, w)| (0..h * w).map(|_| rng.random::<EF>()).collect())
            .collect()
    }

    /// Step 4 — `batch_fold_mle_ir` folds with the challenge the graph
    /// transcript sampled, never a host `EF`.
    #[test]
    fn batch_fold_mle_ir_uses_sampled_device_challenge() {
        let ctx = test_ctx();
        let (_, case) = ring_cases().into_iter().next().unwrap();
        let (mut sponge, snap) = seeded_sponge(7);
        let mats = ring_fold_inputs(0xF01D_0001);

        let (want_coeffs, want_s, _) = case.host_pre(None);
        let want_r = host_observe_and_sample(&mut sponge, &want_s);
        let _ = want_coeffs;

        let got = run_ring_round_graph(&case, None, Some(&mats), &snap, &ctx);
        assert_eq!(ef_bytes(&[got.r]), ef_bytes(&[want_r]), "sampled r");
        assert_eq!(got.folded.len(), 2 * RING_FOLD_SHAPES.len());
        for (i, folded) in got.folded.iter().enumerate() {
            let (h, w) = RING_FOLD_SHAPES[i % RING_FOLD_SHAPES.len()];
            let want = host_fold_matrix(&mats[i % RING_FOLD_SHAPES.len()], h, w, want_r);
            assert_eq!(
                ef_bytes(folded),
                ef_bytes(&want),
                "matrix {i}: graph fold under the sampled device challenge differs from the \
                 by-value host fold"
            );
        }
    }

    /// Step 6 — the whole ring, end to end: compute -> observe -> sample ->
    /// fold, all compared on raw bytes against the eager path.
    #[test]
    fn observe_and_fold_zerocheck_round_ir_matches_eager() {
        let ctx = test_ctx();
        for (i, (name, case)) in ring_cases().into_iter().enumerate() {
            let (mut sponge, snap) = seeded_sponge(i as u64 + 11);
            let mats = ring_fold_inputs(0xF01D_1000 + i as u64);

            let (want_coeffs, want_s, want_tilde) = case.host_pre(None);
            let want_r = host_observe_and_sample(&mut sponge, &want_s);
            let want_state = case.host_post(&want_coeffs, want_r);
            let want_sponge = sponge.snapshot();

            let got = run_ring_round_graph(&case, None, Some(&mats), &snap, &ctx);
            assert_eq!(
                ef_bytes(&got.coeffs),
                ef_bytes(&want_coeffs),
                "{name}: coeffs"
            );
            assert_eq!(
                ef_bytes(&got.s_evals),
                ef_bytes(&want_s),
                "{name}: observed s-evals"
            );
            assert_eq!(ef_bytes(&[got.r]), ef_bytes(&[want_r]), "{name}: sampled r");
            assert_eq!(ef_bytes(&got.tilde), ef_bytes(&want_tilde), "{name}: tilde");
            assert_eq!(
                ef_bytes(&got.state),
                ef_bytes(&want_state),
                "{name}: scalar state"
            );
            assert_eq!(
                got.sponge,
                sponge_state_bytes(&want_sponge),
                "{name}: sponge state"
            );
            for (j, folded) in got.folded.iter().enumerate() {
                let k = j % RING_FOLD_SHAPES.len();
                let (h, w) = RING_FOLD_SHAPES[k];
                let want = host_fold_matrix(&mats[k], h, w, want_r);
                assert_eq!(
                    ef_bytes(folded),
                    ef_bytes(&want),
                    "{name}: folded matrix {j}"
                );
            }
        }
    }

    /// Step 6 — the end-to-end oracle has teeth across the whole chain, not
    /// only at the pre kernel: one corrupted evaluator value must move the
    /// polynomial AND at least one folded buffer (via the sampled challenge).
    #[test]
    fn observe_and_fold_zerocheck_round_ir_detects_sabotage() {
        let ctx = test_ctx();
        for (i, (name, case)) in ring_cases().into_iter().enumerate() {
            // Pick the first trace/family that actually contributes.
            let Some((t, is_zc)) = (0..case.num_traces())
                .flat_map(|t| [(t, true), (t, false)])
                .find(|&(t, is_zc)| {
                    let spec = case.specs[t];
                    ring_phase(case.round, &spec) != RingPhase::Exhausted
                        && if is_zc {
                            spec.has_constraints
                        } else {
                            spec.has_interactions
                        }
                })
            else {
                continue;
            };

            let (mut sponge, snap) = seeded_sponge(i as u64 + 23);
            let mats = ring_fold_inputs(0xF01D_2000 + i as u64);
            let (want_coeffs, want_s, _) = case.host_pre(None);
            let want_r = host_observe_and_sample(&mut sponge, &want_s);
            let want_folded: Vec<Vec<EF>> = (0..2 * RING_FOLD_SHAPES.len())
                .map(|j| {
                    let k = j % RING_FOLD_SHAPES.len();
                    let (h, w) = RING_FOLD_SHAPES[k];
                    host_fold_matrix(&mats[k], h, w, want_r)
                })
                .collect();

            let got = run_ring_round_graph(&case, Some((t, is_zc)), Some(&mats), &snap, &ctx);
            assert_ne!(
                ef_bytes(&got.coeffs),
                ef_bytes(&want_coeffs),
                "SABOTAGE LEG IS BLIND: {name}, trace {t} (zc={is_zc}) — the polynomial did not \
                 move"
            );
            assert!(
                (0..want_folded.len())
                    .any(|j| ef_bytes(&got.folded[j]) != ef_bytes(&want_folded[j])),
                "SABOTAGE LEG IS BLIND: {name}, trace {t} (zc={is_zc}) — the polynomial moved but \
                 no folded buffer did, so the sampled challenge is not actually driving the fold"
            );
        }
    }

    /// Step 6 — the per-round node budget.
    ///
    /// `1 pre + s_deg split copies + s_deg observes + (1..2) sample + 1 post`
    /// from the composer, plus the two fold launches: `5 + 2s` or `6 + 2s`
    /// depending on whether `sample_ext` needs a permutation at the
    /// transcript position it lands on (`sponge_graph_ir.rs:339-345`).
    #[test]
    fn zerocheck_ring_node_budget() {
        let ctx = test_ctx();
        let (_, case) = ring_cases().into_iter().next().unwrap();
        let (_, snap) = seeded_sponge(3);
        let mats = ring_fold_inputs(0xF01D_3000);
        // A second fold group, so the count includes BOTH launches the driver
        // makes (matrices and selectors).
        let got = run_ring_round_graph(&case, None, Some(&mats), &snap, &ctx);
        let s = case.s_deg();
        let lo = 5 + 2 * s;
        assert!(
            got.nodes == lo || got.nodes == lo + 1,
            "ring node count {} is outside {{{lo}, {}}} for s_deg = {s}",
            got.nodes,
            lo + 1
        );
        let count = |n: &str| got.node_names.iter().filter(|x| x.as_str() == n).count();
        assert_eq!(count("batch_s_ring_pre"), 1, "exactly one pre kernel");
        assert_eq!(count("batch_s_ring_post"), 1, "exactly one post kernel");
        assert_eq!(count("batch_fold_mle"), 2, "exactly two fold launches");
    }

    /// The acceptance bar (was `logup_zerocheck_phase_graph_compiles`): the
    /// whole phase builds as a graph and the graph compiles to a `GraphExe`
    /// — now from a plan that supplies **no** steady-round polynomial
    /// evaluations and **no** `r_1..r_n`.
    ///
    /// This does **not** run the graph — its inputs are registered but
    /// unbound (`TraceBufs::alloc_inputs`), so `run` would refuse. It asserts
    /// the thing the port is for: `logup_zerocheck_gpu_ir` emits a well-formed
    /// graph for a realistic phase shape, `GraphCompiler` accepts it, and the
    /// round polynomials and challenges are resident rather than supplied.
    #[test]
    fn logup_zerocheck_phase_graph_compiles_without_round_messages() {
        let device = DeviceType::Cuda(0);
        let plan = synthetic_plan(
            /* num_traces */ 3, /* l_skip */ 2, /* n_max */ 4,
        );

        let mut g = GraphBuilder::new();
        let mut transcript = DuplexSpongeGpuIR::new(&mut g, device);
        let mut inputs = PhaseInputBinder::new();
        let bufs: Vec<TraceBufs> = (0..plan.num_traces())
            .map(|t| TraceBufs::alloc_inputs(&mut g, device, &plan, t, &mut inputs))
            .collect();
        let proof =
            logup_zerocheck_gpu_ir(&mut g, &mut transcript, &plan, &bufs, device, &mut inputs);

        assert_eq!(proof.round0_zc_evals.len(), plan.num_traces());
        assert_eq!(proof.round0_logup_evals.len(), plan.num_traces());
        assert_eq!(proof.evaluator_outputs.len(), plan.n_max);
        assert_eq!(proof.column_openings.len(), plan.num_traces());
        // The two shapes that used to come from `ZerocheckPhasePlan`.
        assert_eq!(
            proof.sumcheck_round_polys.len(),
            plan.n_max,
            "one round polynomial per steady round"
        );
        for (i, polys) in proof.sumcheck_round_polys.iter().enumerate() {
            assert_eq!(
                polys.len(),
                plan.s_deg(),
                "round {}: expected s_deg = {} resident evaluations",
                i + 1,
                plan.s_deg()
            );
        }
        assert_eq!(
            proof.r.len(),
            plan.n_max + 1,
            "r_0 .. r_{{n_max}} are resident"
        );

        let exe = GraphCompiler::new()
            .device(device)
            .scheduler(SchedulerMode::ListV1 {
                params: ListSchedulerV1::default(),
            })
            .compile(g)
            .expect("phase graph compile");
        assert!(exe.num_outputs() > 0, "phase graph produced no outputs");
    }
}
