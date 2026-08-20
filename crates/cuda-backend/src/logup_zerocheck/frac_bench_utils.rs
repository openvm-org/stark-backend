//! Shared test helpers for the fractional-sumcheck benches and dumps.
//!
//! Two sibling files exercise the fractional-sumcheck compiler pipeline
//! ([`super::fractional_ir`] and [`super::fractional_sumcheck_gpu_irv2`])
//! and used to read their tunables from a mix of `FRAC_BENCH_*` /
//! `FRAC_V2_BENCH_*` / `FRAC_IR_FUSION*` / `FRAC_DUMP_LOG_N` env vars.
//! This module unifies that surface behind a single `CC_*` naming for
//! compiler options and a single `FRAC_LOG_N` for input size, so a
//! caller can flip a knob once and both benches respond.
//!
//! # Env vars
//!
//! Compiler tuning (`CC_*` — same across both files):
//!
//! - `CC_FUSION` — `v1` (default), `v2`, or `off`.
//! - `CC_FUSION_DISABLE` — comma-separated v2 pass names to disable (`producer_consumer`, `fanout`,
//!   `small_kernel`, `horizontal`, `epilogue`).
//! - `CC_FUSION_SOLVER_SECS` — CP-SAT wall-time per lex stage (default 120; the crate default of 5s
//!   returns `SolverStatusUnknown` on large graphs and falls back to the original unfused
//!   extraction).
//! - `CC_FUSION_MAX_ALTS` — total cap on inserted alternatives per outer iteration (default
//!   10_000).
//! - `CC_FUSION_MAX_ROUNDS` — saturation round bound.
//! - `CC_FUSION_MAX_ENUM_MS` — per-round enumeration wall-time budget.
//! - `CC_FUSION_OUTER_ITERS` — number of outer fusion iterations.
//! - `CC_FUSION_HORIZONTAL=1` — re-enable horizontal fusion.
//! - `CC_FUSION_SOLVER_WORKERS` — CP-SAT workers (default: all cores).
//! - `CC_SCHEDULER` — `list_v1` (default) or `list_v2`. `list_v2` selects
//!   `SchedulerMode::ListV2` with default beam params (num_beams=1,
//!   beam_depth=1, frontier_cap=1 — pure-greedy) and `num_streams=CC_STREAMS`.
//!   By default `plan_v2` routes through v1 for our carry-chain graphs
//!   (v2's SSA assumption is violated by blackbox kernels with
//!   `carried_outputs`); set `CC_LIST_V2_STRICT=1` to force the raw v2
//!   beam solver instead.
//! - `CC_STREAMS` — `ListSchedulerV1::max_concurrency` (default 8).
//! - `CC_KERNEL_CACHE_MAX_ENTRIES` — max entries in the on-disk kernel cache (default 4096, larger
//!   than the crate default so the v2 graph's ~hundreds of modules survive across bench runs).
//! - `CC_KERNEL_CACHE_MAX_BYTES` — max total on-disk bytes (default 200 GiB, matching the v2
//!   bench's prior hard-coded value).
//! - `CC_NVCC_TIMEOUT_SECS` — per-invocation nvcc timeout (default 900).
//! - `CC_GRAPH_DUMP_PATH` — if set, compile-or-load the [`GraphExe`] to/from this path (see
//!   [`load_or_compile_and_dump`]). The env value is treated as a *base path*: the utility appends
//!   `.n{n}` where `n = 2^log_n` so multiple sizes coexist under one env value.
//!
//! Input size (`FRAC_LOG_N`): comma-separated log2 leaf counts, same
//! name for benches and dumps. Callers pick the first entry for
//! single-size use.

#![cfg(test)]

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use crypto_compiler::{
    graph_exe::{GraphCompiler, GraphExe},
    graph_ir::{DeviceType, GraphBuilder},
    graph_serializer::SerializableGraphBuilder,
    kernel_cache::KernelCache,
    passes::fusion_v2::FusionOptionsV2,
    planner::{ListSchedulerV1, ListSchedulerV2, SchedulerMode},
};
use openvm_cuda_common::stream::GpuDeviceCtx;

/// Parses `FRAC_LOG_N` as a comma-separated list of `log2(leaves)`. If
/// unset, `default` (also comma-separated) is used.
#[allow(dead_code)]
pub(crate) fn frac_log_ns(default: &str) -> Vec<usize> {
    std::env::var("FRAC_LOG_N")
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .expect("FRAC_LOG_N entry must be a non-negative integer")
        })
        .collect()
}

/// Reads `FRAC_LOG_N` as a single value (first comma-separated entry).
/// Returns `default` when the env var is unset or empty.
#[allow(dead_code)]
pub(crate) fn frac_log_n_single(default: usize) -> usize {
    std::env::var("FRAC_LOG_N")
        .ok()
        .and_then(|s| {
            s.trim()
                .split(',')
                .next()
                .and_then(|v| v.trim().parse().ok())
        })
        .unwrap_or(default)
}

/// Fusion-v2 options driven by the `CC_FUSION_*` environment. Used by
/// [`cc_compiler`] when `CC_FUSION=v2`; exposed so callers that need a
/// customized `GraphCompiler` (e.g. dump paths that also set a
/// dump_dir) can start from these defaults.
#[allow(dead_code)]
pub(crate) fn cc_fusion_v2_options() -> FusionOptionsV2 {
    let defaults = FusionOptionsV2::default();
    let solver_secs = std::env::var("CC_FUSION_SOLVER_SECS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(120.0);
    let max_alts = std::env::var("CC_FUSION_MAX_ALTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10_000);
    let max_rounds = std::env::var("CC_FUSION_MAX_ROUNDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(defaults.max_rounds);
    let horizontal = std::env::var_os("CC_FUSION_HORIZONTAL").is_some();
    let solver_workers = std::env::var("CC_FUSION_SOLVER_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let max_enum = std::env::var("CC_FUSION_MAX_ENUM_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(defaults.max_enumeration_time_per_round);
    let outer_iters = std::env::var("CC_FUSION_OUTER_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(defaults.max_outer_iterations);
    FusionOptionsV2 {
        verbose: true,
        solver_time_limit_secs: solver_secs,
        solver_num_workers: solver_workers,
        max_total_alternatives: max_alts,
        max_rounds,
        max_enumeration_time_per_round: max_enum,
        max_outer_iterations: outer_iters,
        enable_horizontal: horizontal,
        ..defaults
    }
}

fn apply_fusion_disable_env(opts: &mut FusionOptionsV2) {
    let Ok(disable) = std::env::var("CC_FUSION_DISABLE") else {
        return;
    };
    for pass in disable.split(',') {
        match pass.trim() {
            "producer_consumer" => opts.enable_producer_consumer = false,
            "fanout" => opts.enable_fanout = false,
            "small_kernel" => opts.enable_small_kernel = false,
            "horizontal" => opts.enable_horizontal = false,
            "epilogue" => opts.enable_epilogue = false,
            "" => {}
            other => panic!("CC_FUSION_DISABLE: unknown fusion pass `{other}`"),
        }
    }
}

fn cc_kernel_cache() -> Arc<KernelCache> {
    let max_entries = std::env::var("CC_KERNEL_CACHE_MAX_ENTRIES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4096);
    let max_bytes = std::env::var("CC_KERNEL_CACHE_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(200 * 1024 * 1024 * 1024);
    Arc::new(
        KernelCache::new()
            .max_kernels(max_entries)
            .storage_size(max_bytes),
    )
}

fn cc_nvcc_timeout() -> Duration {
    Duration::from_secs(
        std::env::var("CC_NVCC_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(900),
    )
}

fn cc_scheduler_mode() -> SchedulerMode {
    let max_conc: u32 = std::env::var("CC_STREAMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    match std::env::var("CC_SCHEDULER").as_deref() {
        Ok("list_v2") => SchedulerMode::ListV2 {
            params: ListSchedulerV2 {
                num_streams: max_conc as usize,
                ..ListSchedulerV2::default()
            },
        },
        // "list_v1" or unset — default profile-guided beam-search list scheduler.
        _ => SchedulerMode::ListV1 {
            params: ListSchedulerV1 {
                max_concurrency: max_conc,
                ..ListSchedulerV1::default()
            },
        },
    }
}

/// Builds a `GraphCompiler` from the unified `CC_*` env vars — scheduler,
/// fusion strategy, kernel cache and nvcc timeout are all honored.
#[allow(dead_code)]
pub(crate) fn cc_compiler(device: DeviceType) -> GraphCompiler {
    let mut compiler = GraphCompiler::new()
        .device(device)
        .scheduler(cc_scheduler_mode())
        .kernel_cache(cc_kernel_cache())
        .nvcc_timeout(Some(cc_nvcc_timeout()));
    match std::env::var("CC_FUSION").as_deref() {
        Ok("v2") => {
            let mut opts = cc_fusion_v2_options();
            apply_fusion_disable_env(&mut opts);
            compiler = compiler.fusion_v2_options(opts);
        }
        Ok("off") => compiler = compiler.without_fusion(),
        // "v1" or unset uses the default v1 pipeline.
        _ => {}
    }
    compiler
}

/// Resolves `CC_GRAPH_DUMP_PATH` into a size-suffixed file path.
///
/// The env value is treated as a *base path*: the size marker `.n{n}` is
/// injected before the extension so multiple `log_n` values coexist. E.g.
/// `CC_GRAPH_DUMP_PATH=/tmp/frac.bin` with `log_n=10` yields
/// `/tmp/frac.n1024.bin`.
///
/// Returns `None` when the env var is unset.
#[allow(dead_code)]
pub(crate) fn cc_graph_dump_path(log_n: usize) -> Option<PathBuf> {
    let base = PathBuf::from(std::env::var_os("CC_GRAPH_DUMP_PATH")?);
    let n = 1usize << log_n;
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("graph");
    let file_name = match base.extension().and_then(|s| s.to_str()) {
        Some(ext) => format!("{stem}.n{n}.{ext}"),
        None => format!("{stem}.n{n}"),
    };
    let dir = base.parent();
    Some(match dir {
        Some(d) if !d.as_os_str().is_empty() => d.join(file_name),
        _ => PathBuf::from(file_name),
    })
}

/// Try to load a previously-serialized `GraphBuilder` from `path`,
/// consuming a fresh `existing` builder to reattach blackbox closures.
/// On any failure (missing file, stale hash, IO, deserialization) returns
/// `Err(msg)` so the caller can fall back to a fresh construction.
fn try_load_builder(
    path: &Path,
    existing: &mut GraphBuilder,
    ctx: &GpuDeviceCtx,
) -> Result<GraphBuilder, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let ser: SerializableGraphBuilder =
        bincode::deserialize(&bytes).map_err(|e| format!("bincode: {e}"))?;
    ser.into_graph_builder(Some(existing), ctx)
        .map_err(|e| format!("into_graph_builder: {e:?}"))
}

fn dump_builder(path: &Path, builder: &mut GraphBuilder, ctx: &GpuDeviceCtx) -> Result<(), String> {
    // Prime `original_hash` cache before serializing so downstream
    // consumers can compare fingerprints without needing `&mut`.
    let _ = builder.original_hash();
    let ser = SerializableGraphBuilder::from_graph_builder(builder, ctx)
        .map_err(|e| format!("from_graph_builder: {e:?}"))?;
    let bytes = bincode::serialize(&ser).map_err(|e| format!("bincode: {e}"))?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Compile-or-load driver: honors `CC_GRAPH_DUMP_PATH`.
///
/// The snapshot format is now the pre-pass [`GraphBuilder`] (see
/// [`SerializableGraphBuilder`]), so *compilation always runs* on the
/// deserialized builder — the snapshot skips graph *construction*, not
/// fusion / lowering / codegen. The compiled kernels still reuse the
/// on-disk kernel cache, which absorbs most of the compile cost after
/// the first run.
///
/// - When `dump_path` is `Some(path)` *and* the file exists, deserializes the snapshot, re-attaches
///   blackbox closures from a freshly-built `GraphBuilder` (via `build`), then compiles that
///   builder.
/// - Otherwise, compiles the freshly-built graph directly, and (if `dump_path` is set) writes the
///   pre-pass snapshot to `path` so the next invocation picks it up.
///
/// `build` is a closure that reconstructs the source `GraphBuilder` from
/// scratch and returns it alongside caller-owned `metadata` (derived
/// `BufId`s, per-graph constants). It is called once on load-hit
/// (supplies closures) and once on load-miss (compiled directly, plus
/// a second time to snapshot the pre-pass state for the dump). The
/// closure must therefore be deterministic and side-effect-free w.r.t.
/// the graph shape.
///
/// Load failures (missing file, bincode error, hash mismatch, etc.) are
/// logged to stderr but never propagated: the driver silently falls
/// back to compiling from scratch.
#[allow(dead_code)]
pub(crate) fn load_or_compile_and_dump<F, M>(
    build: F,
    compiler: GraphCompiler,
    ctx: &GpuDeviceCtx,
    dump_path: Option<PathBuf>,
) -> (GraphExe, M)
where
    F: FnMut() -> (GraphBuilder, M),
{
    load_or_compile_and_dump_with_hook(build, compiler, ctx, dump_path, None)
}

/// Same as [`load_or_compile_and_dump`] but additionally serializes the
/// post-fuse+dce `GraphBuilder` to `fused_snapshot_path` (when `Some`)
/// via `GraphCompiler::compile_with_post_fuse_hook`. That snapshot lines
/// up index-by-index with the compiled exe's `ExeNode` list, so a
/// downstream cytoscape dump can attach `GraphInfo` timings without
/// having to re-run fusion (which the CP-SAT solver is non-
/// deterministic about across worker counts).
#[allow(dead_code)]
pub(crate) fn load_or_compile_and_dump_with_hook<F, M>(
    mut build: F,
    compiler: GraphCompiler,
    ctx: &GpuDeviceCtx,
    dump_path: Option<PathBuf>,
    fused_snapshot_path: Option<PathBuf>,
) -> (GraphExe, M)
where
    F: FnMut() -> (GraphBuilder, M),
{
    let (mut builder, metadata) = build();

    if let Some(path) = dump_path.as_ref() {
        if path.exists() {
            match try_load_builder(path, &mut builder, ctx) {
                Ok(restored) => {
                    eprintln!("[cc] loaded graph-builder snapshot from {}", path.display());
                    let exe =
                        compile_with_optional_hook(compiler, restored, ctx, fused_snapshot_path);
                    return (exe, metadata);
                }
                Err(e) => {
                    eprintln!(
                        "[cc] failed to load graph-builder snapshot from {} ({e}); \
                         building + compiling from source",
                        path.display()
                    );
                }
            }
        }
    }

    // Dump the pre-pass snapshot *before* compile consumes the builder.
    if let Some(path) = dump_path.as_ref() {
        match dump_builder(path, &mut builder, ctx) {
            Ok(()) => {
                eprintln!("[cc] dumped graph-builder snapshot to {}", path.display());
            }
            Err(e) => {
                eprintln!(
                    "[cc] failed to dump graph-builder snapshot to {} ({e}); ignoring",
                    path.display()
                );
            }
        }
    }

    let exe = compile_with_optional_hook(compiler, builder, ctx, fused_snapshot_path);
    (exe, metadata)
}

fn compile_with_optional_hook(
    compiler: GraphCompiler,
    builder: GraphBuilder,
    ctx: &GpuDeviceCtx,
    fused_snapshot_path: Option<PathBuf>,
) -> GraphExe {
    match fused_snapshot_path {
        Some(path) => compiler
            .compile_with_post_fuse_hook(builder, |g_fused| {
                let ser = SerializableGraphBuilder::from_graph_builder(g_fused, ctx)?;
                let bytes = bincode::serialize(&ser).map_err(|e| {
                    crypto_compiler::CompileError::Runtime(format!(
                        "fused-graph snapshot bincode: {e}"
                    ))
                })?;
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                }
                std::fs::write(&path, bytes).map_err(|e| {
                    crypto_compiler::CompileError::Runtime(format!(
                        "fused-graph snapshot write {}: {e}",
                        path.display()
                    ))
                })?;
                eprintln!(
                    "[cc] dumped fused post-DCE graph snapshot to {}",
                    path.display()
                );
                Ok(())
            })
            .expect("graph compile"),
        None => compiler.compile(builder).expect("graph compile"),
    }
}

// Unit tests for these helpers would need to mutate process env; that
// races with parallel test runs, so they're intentionally not included.
// The GPU-side callers of `load_or_compile_and_dump` exercise the
// end-to-end round trip.
