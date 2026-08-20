//! Shared test helpers for the fractional-sumcheck benches and dumps.
//!
//! Two sibling files exercise the fractional-sumcheck compiler pipeline
//! ([`super::fractional_ir`] and [`super::fractional_sumcheck_gpu_irv2`])
//! and share their compiler tunables through this module.
//!
//! All `GraphCompiler` knobs live in a single TOML config; see the
//! [`GraphCompilerConfig`](crypto_compiler::graph_compiler_config::GraphCompilerConfig)
//! docs plus `crates/compiler/cc_default_config.toml` for the full schema
//! and defaults.
//!
//! # Env vars
//!
//! - `CC_CONFIG` — path to the compiler-config TOML. When unset,
//!   [`cc_compiler`] falls back to
//!   [`GraphCompilerConfig::default`](crypto_compiler::graph_compiler_config::GraphCompilerConfig::default),
//!   which mirrors the compiler's programmatic defaults.
//! - `FRAC_LOG_N` — comma-separated `log2(leaves)` list. Not a compiler
//!   flag: it's the bench input size and is orthogonal to the TOML.
//! - `CC_GRAPH_DUMP_PATH` — optional base path for the pre-pass
//!   `GraphBuilder` snapshot (see [`cc_graph_dump_path`] and
//!   [`load_or_compile_and_dump`]). Not a compiler flag either: it toggles
//!   the bench's compile-or-load driver, not the compiler itself.

#![cfg(test)]

use std::path::{Path, PathBuf};

use crypto_compiler::{
    graph_exe::{GraphCompiler, GraphExe},
    graph_ir::{DeviceType, GraphBuilder},
    graph_serializer::SerializableGraphBuilder,
};
use openvm_cuda_common::stream::GpuDeviceCtx;

/// Env var pointing at the compiler-config TOML consumed by [`cc_compiler`].
pub(crate) const CC_CONFIG_ENV: &str = "CC_CONFIG";

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

/// Builds a `GraphCompiler` from `CC_CONFIG` (or defaults when unset) and
/// binds it to `device`.
///
/// The caller-supplied `device` is authoritative: if the TOML sets a
/// different device, this function overwrites it. That keeps the bench in
/// control of which GPU ordinal it runs on regardless of the shared
/// config.
///
/// Panics if `CC_CONFIG` is set but the file can't be read or parsed —
/// silently falling back to defaults would hide a misconfigured bench.
#[allow(dead_code)]
pub(crate) fn cc_compiler(device: DeviceType) -> GraphCompiler {
    let compiler = match std::env::var_os(CC_CONFIG_ENV) {
        Some(path) => GraphCompiler::from_toml(PathBuf::from(path))
            .unwrap_or_else(|e| panic!("failed to load {CC_CONFIG_ENV}: {e}")),
        None => GraphCompiler::new(),
    };
    compiler.device(device)
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
