//! Critical-path interpreter over KIR SSA dependencies.
//!
//! `detailed-fusion-plan-v2.md` §12.6. Walks the kernel body per plan:
//!
//! - constants are ready at cycle 0;
//! - a scalar op result is ready at `max(operand_ready) + op_latency`;
//! - a global read entering a `Par` block adds base global latency, attenuated by the estimated
//!   latency-hiding factor;
//! - a loop repeats its body critical path by `bound` while respecting loop-carried dependencies;
//! - a `Select` uses the maximum branch path (until branch probabilities land);
//! - `Sync` adds `sync_latency_cycles`;
//! - a write becomes ready after its yielded value's ready cycle plus a store-issue latency.
//!
//! The interpreter also totals weighted dynamic operations, synchronization
//! count, longest dependent global-load chain, and requested/transaction
//! bytes so [`estimator::aggregate_cycles`](super::estimator::aggregate_cycles)
//! can combine them.

use std::collections::HashMap;

use crate::{
    ir::{BinOp, ScalarType},
    kernel_ir::{Kernel, SSABlock, SSANode, SSAOp, SSAOpCode, SSARes},
};

/// Per-thread critical path summary for one kernel body.
///
/// All fields are per-thread and per-block-launch — the outer aggregator
/// scales them by block-wave count and warp counts.
#[derive(Clone, Debug, Default)]
pub struct CriticalPath {
    /// Longest dependency chain in cycles, at whichever program point owns
    /// the terminal write.
    pub critical_cycles_per_block: f64,
    /// Total weighted dynamic ops (scalar ops * their static count *
    /// dynamic multiplier of the enclosing pars/loops).
    pub weighted_dynamic_ops: f64,
    /// Number of `Sync` instructions on the critical path.
    pub sync_count: u32,
    /// Longest chain of dependent global loads on the critical path.
    pub longest_dep_load_chain: u32,
}

/// Latency configuration reused by the interpreter (a slice of the full
/// [`EstimatorConfig`](super::estimator::EstimatorConfig)).
#[derive(Clone, Debug)]
pub struct InterpLatencies {
    pub global_latency_cycles: f64,
    pub sync_latency_cycles: f64,
    /// Approximation of the latency-hiding factor (how many active warps
    /// on an SM before global loads are fully hidden). The critical-path
    /// contribution of a global load is `global_latency_cycles /
    /// min(active_warps, latency_saturation_warps)`.
    pub effective_hiding_factor: f64,
    /// Per-op latency table.
    pub op_latency: OpLatencyTable,
}

/// Latencies (in cycles) for each scalar op class.
#[derive(Clone, Debug)]
pub struct OpLatencyTable {
    pub add_bb: f64,
    pub sub_bb: f64,
    pub mul_bb: f64,
    pub add_fpext: f64,
    pub sub_fpext: f64,
    pub mul_fpext: f64,
    pub add_u32: f64,
    pub sub_u32: f64,
    pub mul_u32: f64,
    pub div_u32: f64,
    pub rem_u32: f64,
    pub cmp: f64,
    pub const_scalar: f64,
    pub lift_fpext: f64,
    pub select: f64,
    pub load_issue: f64,
    pub store_issue: f64,
}

impl Default for OpLatencyTable {
    /// A conservative default profile — replace with a calibrated table
    /// once profiling data is available. Numbers reflect the rough
    /// magnitudes on modern Ampere/Hopper for scalar BabyBear/FpExt work.
    fn default() -> Self {
        Self {
            add_bb: 4.0,
            sub_bb: 4.0,
            mul_bb: 6.0,
            add_fpext: 12.0,
            sub_fpext: 12.0,
            mul_fpext: 60.0,
            add_u32: 4.0,
            sub_u32: 4.0,
            mul_u32: 4.0,
            div_u32: 20.0,
            rem_u32: 20.0,
            cmp: 4.0,
            const_scalar: 1.0,
            lift_fpext: 1.0,
            select: 4.0,
            load_issue: 4.0,
            store_issue: 4.0,
        }
    }
}

impl OpLatencyTable {
    /// Cycles for `Bin(op, ty)`.
    pub fn bin_latency(&self, op: BinOp, ty: ScalarType) -> f64 {
        use BinOp::*;
        use ScalarType::*;
        match (op, ty) {
            (Add, BabyBear) => self.add_bb,
            (Sub, BabyBear) => self.sub_bb,
            (Mul, BabyBear) => self.mul_bb,
            (Add, FpExt) => self.add_fpext,
            (Sub, FpExt) => self.sub_fpext,
            (Mul, FpExt) => self.mul_fpext,
            (Add, U32) | (Add, Bool) => self.add_u32,
            (Sub, U32) | (Sub, Bool) => self.sub_u32,
            (Mul, U32) | (Mul, Bool) => self.mul_u32,
            (Div, _) => self.div_u32,
            (Rem, _) => self.rem_u32,
            (Lt, _) | (Le, _) | (Eq, _) => self.cmp,
        }
    }
}

/// Runs the interpreter on `k`'s grid block.
///
/// `dynamic_mul` is a scale applied to weighted ops: the critical path is
/// intrinsically per-thread, but `weighted_dynamic_ops` accumulates
/// per-thread op counts scaled by the enclosing loop counts and the
/// per-block thread count.
pub fn interpret(
    k: &Kernel,
    latencies: &InterpLatencies,
    active_warps_for_hiding: f64,
    per_block_threads: u32,
) -> CriticalPath {
    let mut ctx = InterpCtx {
        kernel: k,
        latencies,
        latency_hiding: latency_hiding_factor(latencies, active_warps_for_hiding),
        per_block_threads: per_block_threads.max(1) as f64,
        ready: HashMap::new(),
        dep_load_depth: HashMap::new(),
        sync_count: 0,
        longest_dep_load_chain: 0,
        weighted_dynamic_ops: 0.0,
        dynamic_scale_stack: vec![1.0],
    };
    // The grid block runs once per block wave; the outer aggregator
    // multiplies by block_waves.
    let end = ctx.visit_block(&k.grid.block, 0.0);
    CriticalPath {
        critical_cycles_per_block: end,
        weighted_dynamic_ops: ctx.weighted_dynamic_ops,
        sync_count: ctx.sync_count,
        longest_dep_load_chain: ctx.longest_dep_load_chain,
    }
}

fn latency_hiding_factor(latencies: &InterpLatencies, active_warps: f64) -> f64 {
    let saturating = latencies.effective_hiding_factor.max(1.0);
    active_warps.max(1.0).min(saturating)
}

struct InterpCtx<'a> {
    kernel: &'a Kernel,
    latencies: &'a InterpLatencies,
    latency_hiding: f64,
    per_block_threads: f64,
    /// Ready cycle for each SSARes produced in the block.
    ready: HashMap<SSARes, f64>,
    /// Longest dependent-load chain terminating at each SSARes.
    dep_load_depth: HashMap<SSARes, u32>,
    sync_count: u32,
    longest_dep_load_chain: u32,
    weighted_dynamic_ops: f64,
    /// Stack of dynamic-instance multipliers for the enclosing pars/loops
    /// (in log space if we ever need to worry about overflow, but summed
    /// as f64 here for simplicity).
    dynamic_scale_stack: Vec<f64>,
}

impl InterpCtx<'_> {
    fn current_scale(&self) -> f64 {
        *self.dynamic_scale_stack.last().unwrap_or(&1.0)
    }

    fn set_ready(&mut self, r: SSARes, t: f64) {
        self.ready.insert(r, t);
    }

    fn readiness(&self, r: SSARes) -> f64 {
        *self.ready.get(&r).unwrap_or(&0.0)
    }

    fn dep_depth(&self, r: SSARes) -> u32 {
        *self.dep_load_depth.get(&r).unwrap_or(&0)
    }

    fn max_operand_ready(&self, op: &SSAOp) -> f64 {
        op.operands
            .iter()
            .map(|&r| self.readiness(r))
            .fold(0.0, f64::max)
    }

    fn max_operand_depth(&self, op: &SSAOp) -> u32 {
        op.operands
            .iter()
            .map(|&r| self.dep_depth(r))
            .max()
            .unwrap_or(0)
    }

    /// Returns the block's exit-cycle (max ready over its yields + any
    /// side-effect ops in it).
    fn visit_block(&mut self, block: &SSABlock, entry_cycle: f64) -> f64 {
        // Block operands are "ready" at entry.
        for &b in &block.operands {
            self.ready.entry(b).or_insert(entry_cycle);
            self.dep_load_depth.entry(b).or_insert(0);
        }
        let mut running = entry_cycle;
        for &sid in &block.body {
            running = self.visit_op(sid, running);
        }
        // Yields carry the block's terminal ready cycle.
        let end = block
            .yields
            .iter()
            .map(|&r| self.readiness(r))
            .fold(running, f64::max);
        end.max(running)
    }

    fn visit_op(&mut self, sid: SSANode, running: f64) -> f64 {
        let op = self.kernel.op(sid);
        match &op.opcode {
            SSAOpCode::ConstU32(_)
            | SSAOpCode::ConstSym(_)
            | SSAOpCode::ConstField(_)
            | SSAOpCode::ConstFpExt(_) => {
                let t = self.latencies.op_latency.const_scalar;
                self.weighted_dynamic_ops += t * self.current_scale();
                for &r in &op.results {
                    self.set_ready(r, t);
                    self.dep_load_depth.insert(r, 0);
                }
                running
            }
            SSAOpCode::LiftFpExt => {
                let lat = self.latencies.op_latency.lift_fpext;
                let ready = self.max_operand_ready(op) + lat;
                let depth = self.max_operand_depth(op);
                for &r in &op.results {
                    self.set_ready(r, ready);
                    self.dep_load_depth.insert(r, depth);
                }
                self.weighted_dynamic_ops += lat * self.current_scale();
                running
            }
            SSAOpCode::Bin(bin, ty) => {
                let lat = self.latencies.op_latency.bin_latency(*bin, *ty);
                let ready = self.max_operand_ready(op) + lat;
                let depth = self.max_operand_depth(op);
                for &r in &op.results {
                    self.set_ready(r, ready);
                    self.dep_load_depth.insert(r, depth);
                }
                self.weighted_dynamic_ops += lat * self.current_scale();
                running
            }
            SSAOpCode::Select { else_block } => {
                // Recurse both branches from the current running cycle.
                let then_end = self.visit_block(&op.block, running);
                let else_end = self.visit_block(else_block, running);
                let branch_end = then_end.max(else_end);
                let ready = branch_end + self.latencies.op_latency.select;
                // Depth aggregated from the two yielded values.
                let mut depth = self.max_operand_depth(op);
                if let Some(&y) = op.block.yields.first() {
                    depth = depth.max(self.dep_depth(y));
                }
                if let Some(&y) = else_block.yields.first() {
                    depth = depth.max(self.dep_depth(y));
                }
                for &r in &op.results {
                    self.set_ready(r, ready);
                    self.dep_load_depth.insert(r, depth);
                }
                self.weighted_dynamic_ops +=
                    self.latencies.op_latency.select * self.current_scale();
                running.max(ready)
            }
            SSAOpCode::Loop { bound } => {
                // Loop-carried dependency: the body's critical path is
                // replicated `bound` times when there is a cross-iteration
                // chain. When operands and results correspond, the plan
                // asks us to respect loop-carried dependencies.
                let bound = *bound as f64;
                self.dynamic_scale_stack
                    .push(self.current_scale() * bound.max(1.0));
                // Seed carried block operands with the initial values'
                // ready cycles.
                for (i, &carried) in op.block.operands.iter().skip(1).enumerate() {
                    if let Some(&init) = op.operands.get(i) {
                        let r = self.readiness(init);
                        self.set_ready(carried, r);
                        let d = self.dep_depth(init);
                        self.dep_load_depth.insert(carried, d);
                    }
                }
                let body_end = self.visit_block(&op.block, running);
                let body_len = body_end - running;
                let total = running + body_len * bound.max(1.0);
                // Results carry the terminal ready cycle.
                for (i, &r) in op.results.iter().enumerate() {
                    let ready_i = if let Some(&y) = op.block.yields.get(i) {
                        (self.readiness(y) - running).max(0.0) * bound.max(1.0) + running
                    } else {
                        total
                    };
                    self.set_ready(r, ready_i);
                    let d = op
                        .block
                        .yields
                        .get(i)
                        .map(|&y| self.dep_depth(y))
                        .unwrap_or(0);
                    self.dep_load_depth.insert(r, d);
                }
                self.dynamic_scale_stack.pop();
                total.max(running)
            }
            SSAOpCode::Par {
                bound,
                reads,
                writes,
                spans_grid,
                ..
            } => {
                // Enter par: bump dynamic scale by lanes-per-block (block
                // dim) unless it spans the grid.
                let lanes = if *spans_grid {
                    self.per_block_threads
                } else {
                    self.per_block_threads
                        .min(bound.as_const().unwrap_or(usize::MAX) as f64)
                };
                self.dynamic_scale_stack
                    .push(self.current_scale() * lanes.max(1.0));
                // Reads: each read entering the block adds one global
                // load hop to the depth if it's a global buffer.
                let load_issue = self.latencies.op_latency.load_issue;
                let global_latency = self.latencies.op_latency.load_issue
                    + self.latencies.global_latency_cycles / self.latency_hiding.max(1.0);
                let mut load_ready = running;
                let mut load_depth = self.max_operand_depth(op);
                for (i, acc) in reads.iter().enumerate() {
                    let is_global = self.kernel_buffer_is_global(acc.buf);
                    let latency = if is_global {
                        global_latency
                    } else {
                        load_issue
                    };
                    let ready = running + latency;
                    load_ready = load_ready.max(ready);
                    let depth = load_depth + if is_global { 1 } else { 0 };
                    load_depth = load_depth.max(depth);
                    if let Some(&b) = op.block.operands.get(i + 1) {
                        self.set_ready(b, ready);
                        self.dep_load_depth.insert(b, depth);
                    }
                }
                self.longest_dep_load_chain = self.longest_dep_load_chain.max(load_depth);
                // Recurse into the par body.
                let body_end = self.visit_block(&op.block, load_ready);
                // Writes: yields' ready cycle plus a store issue latency.
                let store_issue = self.latencies.op_latency.store_issue;
                let mut par_end = body_end;
                for (i, _) in writes.iter().enumerate() {
                    if let Some(&y) = op.block.yields.get(i) {
                        let s_ready = self.readiness(y) + store_issue;
                        par_end = par_end.max(s_ready);
                    }
                }
                // Op results carry the par's exit cycle.
                for &r in &op.results {
                    self.set_ready(r, par_end);
                    self.dep_load_depth.insert(r, load_depth);
                }
                self.dynamic_scale_stack.pop();
                par_end
            }
            SSAOpCode::Sync => {
                self.sync_count += 1;
                running + self.latencies.sync_latency_cycles
            }
            SSAOpCode::Alloc { .. } | SSAOpCode::ConvertLayout { .. } => running,
        }
    }

    fn kernel_buffer_is_global(&self, buf: crate::kernel_ir::BufId) -> bool {
        // We don't have direct BufferDecl access here — the caller of
        // `interpret` should feed a kernel referencing the KirProgram's
        // buffer table. For determination, we conservatively treat every
        // access as global (worst-case latency); the transaction sampler
        // is authoritative for byte counts.
        //
        // TODO(M4-refine): plumb a `&[BufferDecl]` here so we distinguish
        // shared/register loads with lower latency.
        let _ = buf;
        true
    }
}
