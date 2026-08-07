//! Register liveness estimate for a KIR kernel.
//!
//! `detailed-fusion-plan-v2.md` §12.3. Conservative backwards liveness over
//! every [`SSABlock`], stepped through every nested region. At each program
//! point, sum the scalar-word count of every [`SSARes`] that is live. The
//! maximum across all points is the estimated per-thread register footprint,
//! before the fixed overhead and scale on top (§12.3):
//!
//! ```text
//! est_registers_per_thread =
//!     register_fixed_overhead
//!     + ceil(register_liveness_scale * max_live_words)
//! ```
//!
//! Word cost per type: `BabyBear`/`U32`/`Bool` = 1 word, `FpExt` = 4 words,
//! unknown = 4 words (conservative per plan §12.3).

use crate::{
    ir::ScalarType,
    kernel_ir::{Kernel, SSABlock, SSAOp, SSAOpCode, SSARes},
};

/// Result of running the liveness pass on one kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RegisterEstimate {
    /// Peak simultaneously-live scalar words at any program point across
    /// every nested block.
    pub max_live_words: u32,
    /// Estimated registers per thread, incorporating
    /// [`EstimatorConfig::register_fixed_overhead`] and
    /// [`EstimatorConfig::register_liveness_scale`].
    pub registers_per_thread: u32,
}

/// Words that fit an [`ScalarType`], per plan §12.3.
pub fn words_of(t: ScalarType) -> u32 {
    match t {
        ScalarType::FpExt => 4,
        ScalarType::BabyBear | ScalarType::U32 | ScalarType::Bool => 1,
    }
}

/// Conservative default word count for values whose type could not be
/// determined from the local op information (§12.3 fallback).
pub const UNKNOWN_WORDS: u32 = 4;

/// Runs the register-liveness pass on `k`, returning the peak simultaneously
/// live word count plus a scaled per-thread register estimate.
pub fn estimate_registers(
    k: &Kernel,
    buffers: &[crate::kernel_ir::BufferDecl],
    fixed_overhead: u32,
    liveness_scale: f64,
) -> RegisterEstimate {
    let types = build_type_table(k, buffers);
    let mut ctx = LivenessCtx {
        kernel: k,
        types: &types,
        peak: 0,
    };
    // Kick off with the grid block; its operand (blockIdx.x) is live for
    // the whole kernel by construction — we approximate that by seeding
    // grid.block.operands into the initial live set.
    let mut live: LiveSet = LiveSet::default();
    for &op in &k.grid.block.operands {
        live.insert(op, &types);
    }
    ctx.visit_block(&k.grid.block, &mut live);
    let max_live_words = ctx.peak;
    let scaled = (max_live_words as f64 * liveness_scale).ceil() as u32;
    let registers_per_thread = fixed_overhead.saturating_add(scaled);
    RegisterEstimate {
        max_live_words,
        registers_per_thread,
    }
}

// -------------------------------------------------------------------------
// Value type table (SSARes -> word count)
// -------------------------------------------------------------------------

fn build_type_table(k: &Kernel, buffers: &[crate::kernel_ir::BufferDecl]) -> Vec<u32> {
    // next_val is the exclusive upper bound of SSARes used in the kernel;
    // we size the table to that number.
    let n = k
        .ops()
        .iter()
        .flat_map(|op| op.operands.iter().chain(op.results.iter()))
        .chain(
            k.grid
                .block
                .operands
                .iter()
                .chain(k.grid.block.yields.iter()),
        )
        .map(|r| r.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let mut words = vec![UNKNOWN_WORDS; n];
    // Grid index is a u32.
    if let Some(&SSARes(v)) = k.grid.block.operands.first() {
        if let Some(w) = words.get_mut(v as usize) {
            *w = 1;
        }
    }
    walk_block(k, buffers, &k.grid.block, &mut words);
    words
}

fn set_words(words: &mut Vec<u32>, r: SSARes, w: u32) {
    let idx = r.0 as usize;
    if idx >= words.len() {
        words.resize(idx + 1, UNKNOWN_WORDS);
    }
    words[idx] = w;
}

fn walk_block(
    k: &Kernel,
    buffers: &[crate::kernel_ir::BufferDecl],
    block: &SSABlock,
    words: &mut Vec<u32>,
) {
    for &sid in &block.body {
        let op = k.op(sid);
        classify_op(k, buffers, op, words);
    }
}

fn classify_op(
    k: &Kernel,
    buffers: &[crate::kernel_ir::BufferDecl],
    op: &SSAOp,
    words: &mut Vec<u32>,
) {
    match &op.opcode {
        SSAOpCode::ConstU32(_) | SSAOpCode::ConstSym(_) => {
            for &r in &op.results {
                set_words(words, r, 1);
            }
        }
        SSAOpCode::ConstField(_) => {
            for &r in &op.results {
                set_words(words, r, 1);
            }
        }
        SSAOpCode::ConstFpExt(_) | SSAOpCode::LiftFpExt => {
            for &r in &op.results {
                set_words(words, r, 4);
            }
        }
        SSAOpCode::Bin(_, ty) => {
            let w = words_of(*ty);
            for &r in &op.results {
                set_words(words, r, w);
            }
            walk_block(k, buffers, &op.block, words);
        }
        SSAOpCode::Select { else_block } => {
            // Recurse into both branches. Result type is inferred from the
            // then-branch's yield type after that walk.
            walk_block(k, buffers, &op.block, words);
            walk_block(k, buffers, else_block, words);
            if let (Some(&r), Some(&y)) = (op.results.first(), op.block.yields.first()) {
                let w = *words.get(y.0 as usize).unwrap_or(&UNKNOWN_WORDS);
                set_words(words, r, w);
            }
        }
        SSAOpCode::Loop { .. } => {
            // Statement-level loop: no results/operands at op level for
            // grid-nested loops; carried loops inside a par take the type
            // of the initial value at operands[i].
            //
            // Block operands: [induction var (u32), carried...]
            if let Some(&iv) = op.block.operands.first() {
                set_words(words, iv, 1);
            }
            for (carried_i, &b) in op.block.operands.iter().skip(1).enumerate() {
                if let Some(&init) = op.operands.get(carried_i) {
                    let w = *words.get(init.0 as usize).unwrap_or(&UNKNOWN_WORDS);
                    set_words(words, b, w);
                }
            }
            walk_block(k, buffers, &op.block, words);
            for (i, &r) in op.results.iter().enumerate() {
                if let Some(&y) = op.block.yields.get(i) {
                    let w = *words.get(y.0 as usize).unwrap_or(&UNKNOWN_WORDS);
                    set_words(words, r, w);
                }
            }
        }
        SSAOpCode::Par { reads, writes, .. } => {
            // Par block operands: [par idx (u32), read0, read1, ...]
            let ops = &op.block.operands;
            if let Some(&pi) = ops.first() {
                set_words(words, pi, 1);
            }
            for (i, acc) in reads.iter().enumerate() {
                if let Some(&b) = ops.get(i + 1) {
                    let ty = buffers
                        .get(acc.buf.0 as usize)
                        .map(|d| words_of(d.elem))
                        .unwrap_or(UNKNOWN_WORDS);
                    set_words(words, b, ty);
                }
            }
            walk_block(k, buffers, &op.block, words);
            // Par results correspond to writes (unused, but keep them typed).
            for (i, &r) in op.results.iter().enumerate() {
                if let Some(w) = writes
                    .get(i)
                    .and_then(|acc| buffers.get(acc.buf.0 as usize))
                    .map(|d| words_of(d.elem))
                {
                    set_words(words, r, w);
                }
            }
        }
        SSAOpCode::Alloc { .. } | SSAOpCode::Sync | SSAOpCode::ConvertLayout { .. } => {}
    }
}

// -------------------------------------------------------------------------
// Backwards liveness (max concurrent live words)
// -------------------------------------------------------------------------

/// A small live-set of SSA values with their word-count contribution so
/// the running total updates in O(1) per insert/remove.
#[derive(Default)]
struct LiveSet {
    /// SSARes -> word count for currently-live values.
    live: std::collections::HashMap<SSARes, u32>,
    /// Sum of `live.values()` maintained incrementally.
    total_words: u32,
}

impl LiveSet {
    fn insert(&mut self, r: SSARes, types: &[u32]) {
        let w = *types.get(r.0 as usize).unwrap_or(&UNKNOWN_WORDS);
        if self.live.insert(r, w).is_none() {
            self.total_words = self.total_words.saturating_add(w);
        }
    }

    fn remove(&mut self, r: SSARes) {
        if let Some(w) = self.live.remove(&r) {
            self.total_words = self.total_words.saturating_sub(w);
        }
    }
}

struct LivenessCtx<'a> {
    kernel: &'a Kernel,
    types: &'a [u32],
    peak: u32,
}

impl LivenessCtx<'_> {
    fn record(&mut self, live: &LiveSet) {
        if live.total_words > self.peak {
            self.peak = live.total_words;
        }
    }

    /// Walks `block` in reverse from `live_out` (mutated in place to become
    /// `live_in`). At each op boundary the current `total_words` is recorded
    /// against the running peak.
    fn visit_block(&mut self, block: &SSABlock, live: &mut LiveSet) {
        // Yields are live at the end of the block.
        for &y in &block.yields {
            live.insert(y, self.types);
        }
        self.record(live);
        for &sid in block.body.iter().rev() {
            self.visit_op(sid, live);
            self.record(live);
        }
        // Block operands: leave in the caller's live set; the caller
        // introduced them so they belong to its running state.
    }

    fn visit_op(&mut self, sid: crate::kernel_ir::SSANode, live: &mut LiveSet) {
        let op = self.kernel.op(sid);
        // Recurse into nested region(s) first — their body writes and
        // captures might introduce values that flow into `live` through
        // the block's operands or yields.
        //
        // Backwards direction: op.results are killed, op.operands become live.
        for &r in &op.results {
            live.remove(r);
        }
        // Recurse into the region body(s). Region walk uses a scratch
        // clone so the outer live set only accumulates values that
        // outlive the region.
        match &op.opcode {
            SSAOpCode::Loop { .. } => {
                self.visit_loop_body(op, live);
            }
            SSAOpCode::Par { .. } => {
                self.visit_par_body(op, live);
            }
            SSAOpCode::Select { else_block } => {
                self.visit_select_body(op, else_block, live);
            }
            _ => {}
        }
        // Operands become live in the enclosing block.
        for &o in &op.operands {
            live.insert(o, self.types);
        }
    }

    fn visit_loop_body(&mut self, op: &SSAOp, outer_live: &mut LiveSet) {
        // Iterate loop bodies to a fixed point per plan §12.3.5. The
        // body's initial live-out is (outer live) ∪ (block yields).
        // Its live-in becomes (outer live \ carried block operands) ∪
        // (block operand carried values). We approximate the fixed point
        // by running the body twice, which suffices for the peak we
        // record (nothing new appears after the first pass because block
        // operands are stable).
        for _ in 0..2 {
            let mut body_live = clone_live(outer_live, self.types);
            self.visit_block(&op.block, &mut body_live);
            // Block operands: carried values live in every iteration.
            for &b in &op.block.operands {
                body_live.insert(b, self.types);
            }
            self.record(&body_live);
            // Propagate: carried block operands need to be live going into
            // the next iteration — we've folded them above.
        }
    }

    fn visit_par_body(&mut self, op: &SSAOp, outer_live: &mut LiveSet) {
        let mut body_live = clone_live(outer_live, self.types);
        self.visit_block(&op.block, &mut body_live);
        for &b in &op.block.operands {
            body_live.insert(b, self.types);
        }
        self.record(&body_live);
    }

    fn visit_select_body(&mut self, op: &SSAOp, else_block: &SSABlock, outer_live: &mut LiveSet) {
        for block in [&op.block, else_block] {
            let mut body_live = clone_live(outer_live, self.types);
            self.visit_block(block, &mut body_live);
            self.record(&body_live);
        }
    }
}

fn clone_live(src: &LiveSet, types: &[u32]) -> LiveSet {
    let mut out = LiveSet::default();
    for &r in src.live.keys() {
        out.insert(r, types);
    }
    out
}
