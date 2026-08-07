//! Top-level KIR estimator: HIR module -> `GraphNodeCost`.
//!
//! `detailed-fusion-plan-v2.md` §12. The main entry points are:
//!
//! - [`estimate_kernel`] — lower an [`ir::Module`] through the shared `ModuleCompiler` lowering
//!   path, then run register liveness (§12.3), occupancy (§12.4), transaction sampling (§12.5),
//!   critical-path interpretation (§12.6), and aggregate cycles (§12.7).
//! - [`estimate_non_kernel`] — closed-form cost for the graph's non-kernel variants (§12.8:
//!   `Const`, `Memcpy`, `Memset`, `BlackboxKernel`).
//!
//! The estimator is deterministic: every random-looking choice (warp
//! samples) uses a seed derived from the normalized module hash and the
//! model version.

use std::collections::BTreeMap;

use super::GraphNodeCost;
use crate::{
    graph_ir::{GraphNode, MemSetNode, MemcpyNode},
    ir::{self, VarId},
    kernel_ir::{KBound, KirProgram, SSAOpCode},
    module_compiler::ModuleCompiler,
    passes::{
        fusion_v2::cost::{
            interpreter::{interpret, CriticalPath, InterpLatencies, OpLatencyTable},
            liveness::{estimate_registers, RegisterEstimate},
            transactions::{estimate_access, is_global, AccessSampleCfg},
        },
        plan_shared_mem,
    },
    quast::{Expr, Quast, SExpr, SymConst},
    CompileError,
};

/// Device-model constants that anchor the estimator's cycle math (§12.1).
#[derive(Clone, Debug)]
pub struct DeviceModel {
    pub sms: u32,
    pub warp_size: u32,
    pub max_threads_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub max_warps_per_sm: u32,
    pub registers_per_sm: u32,
    pub shared_bytes_per_sm: u32,
    pub global_sector_bytes: u32,
    pub dram_bytes_per_cycle: f64,
    pub issue_weighted_ops_per_cycle: f64,
    /// Launch overhead tiers, selected by grid size (§12.7).
    pub launch_cycles_fit_sm: f64,
    pub launch_cycles_within_wave: f64,
    pub launch_cycles_multi_wave: f64,
    pub global_latency_cycles: f64,
    pub sync_latency_cycles: f64,
    pub latency_saturation_warps: u32,
    pub op_latency: OpLatencyTable,
    /// Bytes/cycle for host-side `cudaMemcpy` / `cudaMemset`. Used by
    /// [`estimate_non_kernel`] to price the transaction phase (§12.8).
    pub memcpy_bytes_per_cycle: f64,
    /// Overhead cycles for one memcpy/memset launch.
    pub memop_launch_cycles: f64,
}

impl DeviceModel {
    /// A calibration-neutral synthetic profile used by unit tests.
    ///
    /// The numbers are round and internally consistent: they exist so
    /// tests can hand-check estimator outputs. Production callers should
    /// override with a device-specific profile from calibration.
    pub fn synthetic() -> Self {
        Self {
            sms: 100,
            warp_size: 32,
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 16,
            max_warps_per_sm: 64,
            registers_per_sm: 65_536,
            shared_bytes_per_sm: 98_304,
            global_sector_bytes: 32,
            dram_bytes_per_cycle: 32.0,
            issue_weighted_ops_per_cycle: 4.0,
            launch_cycles_fit_sm: 800.0,
            launch_cycles_within_wave: 1300.0,
            launch_cycles_multi_wave: 2000.0,
            global_latency_cycles: 400.0,
            sync_latency_cycles: 30.0,
            latency_saturation_warps: 16,
            op_latency: OpLatencyTable::default(),
            memcpy_bytes_per_cycle: 32.0,
            memop_launch_cycles: 600.0,
        }
    }
}

/// Tunable estimator constants that live alongside the device model (§12.1).
#[derive(Clone, Debug)]
pub struct EstimatorConfig {
    pub device: DeviceModel,
    pub warp_samples_per_par: u32,
    pub register_fixed_overhead: u32,
    pub register_liveness_scale: f64,
    pub unknown_global_sectors_per_warp: u32,
    /// Bumped when the estimator's numerical model changes; feeds the
    /// warp-sample seed and the cost-cache key (§12.10).
    pub model_version: u32,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            device: DeviceModel::synthetic(),
            warp_samples_per_par: 4,
            register_fixed_overhead: 8,
            register_liveness_scale: 1.0,
            unknown_global_sectors_per_warp: 32,
            model_version: 0,
        }
    }
}

/// Per-candidate context: symbolic sizes and param bindings that resolve
/// symbolic bounds (§12.1).
#[derive(Clone, Debug, Default)]
pub struct EstimateContext {
    pub graph_symbols: BTreeMap<VarId, i64>,
    pub param_bindings: BTreeMap<String, i64>,
}

/// Detailed cost breakdown returned by [`estimate_kernel`]. The extractor
/// consumes only `runtime_cycles`, but keeping the pieces on the return
/// value is useful for debugging / reporting.
#[derive(Clone, Debug)]
pub struct KernelCostBreakdown {
    pub registers: RegisterEstimate,
    pub blocks_per_sm: u32,
    pub active_warps_per_sm: u32,
    pub critical: CriticalPath,
    pub access: AccessAggregate,
    pub launch_cycles: f64,
    pub latency_cycles: f64,
    pub bandwidth_cycles: f64,
    pub issue_cycles: f64,
    pub raw_cycles: f64,
    pub total_cycles: f64,
}

/// Sum of per-access estimates over one kernel.
#[derive(Clone, Debug, Default)]
pub struct AccessAggregate {
    pub requested_bytes: u64,
    pub transaction_bytes: u64,
    pub avg_sectors_per_warp: f64,
    pub dynamic_warp_accesses: u64,
}

/// Estimates one candidate kernel node.
///
/// Returns the ILP-consumed [`GraphNodeCost`] plus the breakdown for
/// reporting. Any lowering or param-resolution failure surfaces as a
/// `CompileError`.
pub fn estimate_kernel(
    module: &ir::Module,
    module_hash: [u8; 32],
    ctx: &EstimateContext,
    cfg: &EstimatorConfig,
    cycle_quantum: i64,
) -> Result<(GraphNodeCost, KernelCostBreakdown), CompileError> {
    let mc = ModuleCompiler::new();
    // Lowering the module can panic on malformed inputs; the compiler
    // pipeline returns `CompileError` for structured failures.
    let kp = mc.lower(module.clone())?;
    let plan = plan_shared_mem(&kp);
    let breakdown = analyze_program(&kp, &plan.per_kernel, ctx, cfg, module_hash)?;
    let cost = GraphNodeCost::from_cycles(breakdown.total_cycles, cycle_quantum);
    Ok((cost, breakdown))
}

/// Closed-form cost for non-kernel graph nodes (§12.8).
///
/// - `Const`: zero (setup dominates a rare initial cost, not the run cost).
/// - `Memcpy`/`Memset`: launch overhead + transaction bytes / bandwidth.
/// - `BlackboxKernel`: caller-supplied `blackbox_hint_cycles` or 0.
pub fn estimate_non_kernel(
    node: &GraphNode,
    bufs: &[crate::graph_ir::BufInfo],
    ctx: &EstimateContext,
    cfg: &EstimatorConfig,
    cycle_quantum: i64,
    blackbox_hint_cycles: f64,
) -> GraphNodeCost {
    let cycles = match node {
        GraphNode::Const(_) => 0.0,
        GraphNode::Memcpy(m) => memcpy_cycles(m, bufs, ctx, cfg),
        GraphNode::Memset(m) => memset_cycles(m, bufs, ctx, cfg),
        GraphNode::BlackboxKernel(_) => blackbox_hint_cycles.max(0.0),
        GraphNode::Kernel(_) => {
            // Callers should route kernel nodes through `estimate_kernel`.
            // Falling back to zero here would be misleading; keep the safe
            // launch cost so a mis-routed kernel is still visible.
            cfg.device.launch_cycles_within_wave
        }
    };
    GraphNodeCost::from_cycles(cycles, cycle_quantum)
}

// -------------------------------------------------------------------------
// Non-kernel closed-form helpers
// -------------------------------------------------------------------------

fn memcpy_cycles(
    m: &MemcpyNode,
    _bufs: &[crate::graph_ir::BufInfo],
    ctx: &EstimateContext,
    cfg: &EstimatorConfig,
) -> f64 {
    let bytes = try_eval_quast(&m.num_bytes, ctx)
        .map(|c| c as f64)
        .unwrap_or(1024.0);
    cfg.device.memop_launch_cycles + bytes / cfg.device.memcpy_bytes_per_cycle.max(1.0)
}

fn memset_cycles(
    m: &MemSetNode,
    _bufs: &[crate::graph_ir::BufInfo],
    ctx: &EstimateContext,
    cfg: &EstimatorConfig,
) -> f64 {
    let bytes = try_eval_quast(&m.num_bytes, ctx)
        .map(|c| c as f64)
        .unwrap_or(1024.0);
    cfg.device.memop_launch_cycles + bytes / cfg.device.memcpy_bytes_per_cycle.max(1.0)
}

fn try_eval_quast(q: &Quast, ctx: &EstimateContext) -> Option<i64> {
    fn eval(q: &Quast, env: &BTreeMap<VarId, i64>) -> Option<i64> {
        match q {
            Expr::Sym(v) => env.get(v).copied(),
            Expr::Const(c) => Some(*c),
            Expr::Add(a, b) => Some(eval(a, env)? + eval(b, env)?),
            Expr::Mul(a, c) => Some(eval(a, env)? * *c),
            Expr::FloorDiv(a, c) => {
                if *c <= 0 {
                    return None;
                }
                Some(eval(a, env)?.div_euclid(*c))
            }
            Expr::Neg(a) => Some(-eval(a, env)?),
        }
    }
    eval(q, &ctx.graph_symbols)
}

// -------------------------------------------------------------------------
// Kernel analysis
// -------------------------------------------------------------------------

fn analyze_program(
    kp: &KirProgram,
    shared_bytes_per_kernel: &[usize],
    ctx: &EstimateContext,
    cfg: &EstimatorConfig,
    module_hash: [u8; 32],
) -> Result<KernelCostBreakdown, CompileError> {
    if kp.kernels.is_empty() {
        return Err(CompileError::Lower("empty KIR program".into()));
    }
    if kp.kernels.len() > 1 {
        return Err(CompileError::Lower(format!(
            "estimator supports one KIR kernel per module, got {}",
            kp.kernels.len()
        )));
    }
    let k = &kp.kernels[0];
    let shared_bytes = *shared_bytes_per_kernel.first().unwrap_or(&0);
    // Grid dimension (may be symbolic).
    let grid_dim = resolve_kbound(&k.grid.bound, ctx).unwrap_or(1) as u32;

    // Register estimate (§12.3).
    let registers = estimate_registers(
        k,
        &kp.buffers,
        cfg.register_fixed_overhead,
        cfg.register_liveness_scale,
    );

    // Occupancy (§12.4).
    let occ = compute_occupancy(
        k.block as u32,
        registers.registers_per_thread,
        shared_bytes as u32,
        &cfg.device,
    );

    // Interpreter (§12.6). Latency hiding is a function of active warps
    // per SM, capped by the saturation constant.
    let latencies = InterpLatencies {
        global_latency_cycles: cfg.device.global_latency_cycles,
        sync_latency_cycles: cfg.device.sync_latency_cycles,
        effective_hiding_factor: cfg.device.latency_saturation_warps as f64,
        op_latency: cfg.device.op_latency.clone(),
    };
    let critical = interpret(
        k,
        &latencies,
        occ.active_warps_per_sm as f64,
        k.block as u32,
    );

    // Transaction sampling (§12.5). Walk every Par access site.
    let access = aggregate_accesses(kp, ctx, cfg, module_hash, grid_dim, occ.blocks_per_sm);

    // Aggregate cycles (§12.7).
    let block_count = grid_dim.max(1);
    let resident_blocks = cfg.device.sms.saturating_mul(occ.blocks_per_sm).max(1);
    let block_waves = ((block_count as f64) / (resident_blocks as f64))
        .ceil()
        .max(1.0);
    let latency_cycles = block_waves * critical.critical_cycles_per_block;
    let bandwidth_cycles =
        access.transaction_bytes as f64 / cfg.device.dram_bytes_per_cycle.max(1.0);
    let issue_cycles =
        critical.weighted_dynamic_ops / cfg.device.issue_weighted_ops_per_cycle.max(1.0);
    let launch_cycles =
        launch_cycles_for(block_count, occ.blocks_per_sm, resident_blocks, &cfg.device);
    let raw_cycles = latency_cycles.max(bandwidth_cycles).max(issue_cycles);
    let total_cycles = launch_cycles + raw_cycles;

    Ok(KernelCostBreakdown {
        registers,
        blocks_per_sm: occ.blocks_per_sm,
        active_warps_per_sm: occ.active_warps_per_sm,
        critical,
        access,
        launch_cycles,
        latency_cycles,
        bandwidth_cycles,
        issue_cycles,
        raw_cycles,
        total_cycles,
    })
}

// -------------------------------------------------------------------------
// Occupancy (§12.4)
// -------------------------------------------------------------------------

struct Occupancy {
    blocks_per_sm: u32,
    active_warps_per_sm: u32,
}

fn compute_occupancy(
    block_size: u32,
    regs_per_thread: u32,
    shared_bytes: u32,
    dev: &DeviceModel,
) -> Occupancy {
    let t = block_size.max(1);
    let warps_per_block = t.div_ceil(dev.warp_size);
    let blocks_by_threads = dev.max_threads_per_sm / t;
    let blocks_by_warps = if warps_per_block > 0 {
        dev.max_warps_per_sm / warps_per_block
    } else {
        dev.max_blocks_per_sm
    };
    let regs_per_block = regs_per_thread.saturating_mul(t).max(1);
    let blocks_by_regs = dev.registers_per_sm / regs_per_block;
    let blocks_by_smem = if shared_bytes == 0 {
        dev.max_blocks_per_sm
    } else {
        dev.shared_bytes_per_sm / shared_bytes.max(1)
    };
    let blocks_per_sm = dev
        .max_blocks_per_sm
        .min(blocks_by_threads)
        .min(blocks_by_warps)
        .min(blocks_by_regs)
        .min(blocks_by_smem)
        .max(1);
    let active_warps_per_sm = blocks_per_sm * warps_per_block;
    Occupancy {
        blocks_per_sm,
        active_warps_per_sm,
    }
}

// -------------------------------------------------------------------------
// Transaction aggregation
// -------------------------------------------------------------------------

fn aggregate_accesses(
    kp: &KirProgram,
    ctx: &EstimateContext,
    cfg: &EstimatorConfig,
    module_hash: [u8; 32],
    grid_dim: u32,
    blocks_per_sm: u32,
) -> AccessAggregate {
    let k = &kp.kernels[0];
    let block_dim = k.block as u32;
    let warps_per_block = block_dim.div_ceil(cfg.device.warp_size);
    // Baseline dynamic-warp multiplier: grid * warps_per_block * (loops
    // enclosing this par). For per-block pars, every block runs the par
    // once (we sample one warp per site so the multiplier represents
    // warp instances). For grid-spanning pars, the whole logical domain
    // is covered by the launch; we ignore the block factor.
    let mut agg = AccessAggregate::default();
    walk_par_accesses(
        k,
        ctx,
        cfg,
        module_hash,
        grid_dim,
        warps_per_block,
        &kp.buffers,
        &mut agg,
    );
    let _ = blocks_per_sm;
    agg
}

#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn walk_par_accesses(
    k: &crate::kernel_ir::Kernel,
    ctx: &EstimateContext,
    cfg: &EstimatorConfig,
    module_hash: [u8; 32],
    grid_dim: u32,
    warps_per_block: u32,
    buffers: &[crate::kernel_ir::BufferDecl],
    agg: &mut AccessAggregate,
) {
    #[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
    fn walk(
        k: &crate::kernel_ir::Kernel,
        ctx: &EstimateContext,
        cfg: &EstimatorConfig,
        module_hash: [u8; 32],
        grid_dim: u32,
        warps_per_block: u32,
        buffers: &[crate::kernel_ir::BufferDecl],
        block: &crate::kernel_ir::SSABlock,
        enclosing_scale: u64,
        enclosing: &mut BTreeMap<VarId, i64>,
        par_index: &mut u32,
        agg: &mut AccessAggregate,
    ) {
        for &sid in &block.body {
            let op = k.op(sid);
            match &op.opcode {
                SSAOpCode::Loop { bound } => {
                    let scale = enclosing_scale.saturating_mul(*bound as u64);
                    walk(
                        k,
                        ctx,
                        cfg,
                        module_hash,
                        grid_dim,
                        warps_per_block,
                        buffers,
                        &op.block,
                        scale,
                        enclosing,
                        par_index,
                        agg,
                    );
                }
                SSAOpCode::Par {
                    reads,
                    writes,
                    bound,
                    spans_grid,
                    attr,
                    ..
                } => {
                    let par_dyn = if *spans_grid {
                        // The whole logical bound is covered by the grid.
                        // Each warp does one access site — multiplier is
                        // grid * warps_per_block.
                        enclosing_scale
                            .saturating_mul(grid_dim as u64)
                            .saturating_mul(warps_per_block as u64)
                    } else {
                        // Per-block par: every block runs it, and every
                        // warp in the block participates.
                        enclosing_scale
                            .saturating_mul(grid_dim as u64)
                            .saturating_mul(warps_per_block as u64)
                    };
                    // One access index within this par site.
                    for (idx, acc) in reads.iter().chain(writes.iter()).enumerate() {
                        let Some(buffer) = buffers.get(acc.buf.0 as usize) else {
                            continue;
                        };
                        if !is_global(buffer) {
                            continue;
                        }
                        let seed =
                            mix_seed(module_hash, 0, *par_index, idx as u32, cfg.model_version);
                        let sample_cfg = AccessSampleCfg {
                            warp_size: cfg.device.warp_size,
                            global_sector_bytes: cfg.device.global_sector_bytes,
                            warp_samples_per_par: cfg.warp_samples_per_par,
                            unknown_global_sectors_per_warp: cfg.unknown_global_sectors_per_warp,
                            seed,
                        };
                        let est = estimate_access(
                            acc,
                            buffer,
                            enclosing,
                            bound,
                            attr.as_ref(),
                            k.block as u32,
                            par_dyn,
                            &sample_cfg,
                        );
                        agg.requested_bytes =
                            agg.requested_bytes.saturating_add(est.requested_bytes);
                        agg.transaction_bytes =
                            agg.transaction_bytes.saturating_add(est.transaction_bytes);
                        agg.dynamic_warp_accesses = agg
                            .dynamic_warp_accesses
                            .saturating_add(est.dynamic_warp_accesses);
                        agg.avg_sectors_per_warp += est.avg_sectors_per_warp;
                    }
                    *par_index = par_index.saturating_add(1);
                    // Do not descend into the par's block: its body ops
                    // are per-lane scalar work, not memory transactions.
                }
                _ => {}
            }
        }
    }

    // Grid var: unknown at estimate time; the sampler treats it as such.
    // Populate enclosing bindings with any graph_symbols the caller already
    // resolved. `k.params` is buffer-parameter metadata; the symbolic module
    // params live on `kp.params` (VarId, name pairs), but the sampler only
    // needs VarId bindings that are already in `ctx.graph_symbols`.
    let mut enclosing = ctx.graph_symbols.clone();
    let mut par_index = 0u32;
    walk(
        k,
        ctx,
        cfg,
        module_hash,
        grid_dim,
        warps_per_block,
        buffers,
        &k.grid.block,
        1,
        &mut enclosing,
        &mut par_index,
        agg,
    );
}

fn mix_seed(
    module_hash: [u8; 32],
    kernel_index: u32,
    par_node: u32,
    access_index: u32,
    model_version: u32,
) -> u64 {
    // Fold the module hash into 64 bits, then mix in the per-site tuple.
    let mut acc = 0xcbf29ce484222325u64;
    for byte in module_hash {
        acc ^= byte as u64;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    for chunk in [kernel_index, par_node, access_index, model_version] {
        acc ^= chunk as u64;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    acc
}

fn launch_cycles_for(
    block_count: u32,
    blocks_per_sm: u32,
    resident_blocks: u32,
    dev: &DeviceModel,
) -> f64 {
    if block_count <= blocks_per_sm {
        dev.launch_cycles_fit_sm
    } else if block_count <= resident_blocks {
        dev.launch_cycles_within_wave
    } else {
        dev.launch_cycles_multi_wave
    }
}

fn resolve_kbound(b: &KBound, ctx: &EstimateContext) -> Option<i64> {
    if let Some(c) = b.as_const() {
        return Some(c as i64);
    }
    match b {
        KBound::Const(c) => Some(*c as i64),
        KBound::Expr(e) => resolve_sexpr(e, ctx),
    }
}

fn resolve_sexpr(e: &SExpr, ctx: &EstimateContext) -> Option<i64> {
    let env = &ctx.graph_symbols;
    fn eval(e: &SExpr, env: &BTreeMap<VarId, i64>) -> Option<i64> {
        match e {
            Expr::Sym(v) => env.get(v).copied(),
            Expr::Const(c) => match c {
                SymConst::Lit(l) => Some(*l),
                SymConst::Sym(v) => env.get(v).copied(),
            },
            Expr::Add(a, b) => Some(eval(a, env)? + eval(b, env)?),
            Expr::Mul(a, c) => match c {
                SymConst::Lit(l) => Some(eval(a, env)? * *l),
                SymConst::Sym(v) => Some(eval(a, env)? * env.get(v).copied()?),
            },
            Expr::FloorDiv(a, c) => {
                let d = match c {
                    SymConst::Lit(l) => *l,
                    SymConst::Sym(v) => env.get(v).copied()?,
                };
                if d == 0 {
                    return None;
                }
                Some(eval(a, env)?.div_euclid(d))
            }
            Expr::Neg(a) => Some(-eval(a, env)?),
        }
    }
    eval(e, env)
}
