//! Hardware-mapped CUDA profiler for OpenVM.
//!
//! This crate is a sidecar to `openvm-cuda-backend` that records, for an entire
//! proof run:
//!
//! 1. **Per-CTA hardware placement** — every CTA's start/end time and which SM it ran on, captured
//!    by lightweight device-side probes (see `cta_probe.cuh`).
//! 2. **Kernel/memcpy/NVTX activity** — streamed by the CUPTI Activity sidecar.
//! 3. **(Optional) PC-sampling stall mix** and **Range Profiler** throughput samples, both
//!    whole-proof scope without kernel replay.
//!
//! ## Activation model
//!
//! Two switches:
//!
//! | Switch                              | Effect                                     |
//! |-------------------------------------|--------------------------------------------|
//! | `--features profiler` on cuda-backend | Compiles probes + crate into the binary. |
//! | `SHADOW_PROFILER=1` env var          | Activates probes at runtime.              |
//!
//! With the feature off, this crate is not compiled, not linked, and the CUDA
//! probe macros expand to nothing.
//!
//! With the feature on but the env var off, every kernel launch is unchanged
//! except for one extra CTA-lane-0 branch (mask == 0 short-circuits the
//! probe). No allocations, no writes, no host thread.
//!
//! With both on, the profiler allocates a pinned-mapped ring buffer, publishes
//! the device-visible context to every kernel launch via `shadow_set_cta_ctx`,
//! spawns a drain thread, and writes a framed binary log to
//! `$SHADOW_PROFILER_OUT` (default `./shadow_profile.bin`). Use
//! `cuda-profiler-assemble` to convert it into a Perfetto trace.

#![allow(clippy::missing_safety_doc)]

pub mod cupti;
pub mod ffi;
pub mod kernel_id;
pub mod log_writer;
pub mod record;
pub mod ring;

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
};

pub use kernel_id::{fnv1a, register_kernel, registered_kernels};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use thiserror::Error;

/// Errors from profiler init/shutdown. Anything that fails here is non-fatal:
/// the caller logs and continues with the profiler disabled.
#[derive(Debug, Error)]
pub enum ProfilerError {
    #[error("cuda runtime call failed: {0}")]
    Cuda(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("environment misconfigured: {0}")]
    BadEnv(String),
    #[error("profiler already initialized")]
    AlreadyInit,
}

/// Default ring capacity in bytes (~2.7 M records). Override with
/// `SHADOW_PROFILER_RING_BYTES`. Must be a power of two; rounded up otherwise.
pub const DEFAULT_RING_BYTES: usize = 64 * 1024 * 1024;

/// Output path environment variable.
pub const ENV_OUT: &str = "SHADOW_PROFILER_OUT";
/// Master enable switch.
pub const ENV_ENABLE: &str = "SHADOW_PROFILER";
/// Optional ring-byte override.
pub const ENV_RING_BYTES: &str = "SHADOW_PROFILER_RING_BYTES";
/// Optional drain interval override (milliseconds).
pub const ENV_DRAIN_INTERVAL_MS: &str = "SHADOW_PROFILER_DRAIN_MS";
/// Default drain interval.
pub const DEFAULT_DRAIN_INTERVAL_MS: u64 = 50;

/// Process-global active profiler. Populated by `init` exactly once.
static PROFILER: OnceCell<Mutex<Option<ActiveProfiler>>> = OnceCell::new();

/// Internal handle held by the process while the profiler is active.
/// Dropped via `shutdown`.
struct ActiveProfiler {
    /// Pinned host ring; lives until shutdown.
    _ring: ring::HostRing,
    /// Drain thread handle; joined in shutdown.
    drain: Option<JoinHandle<()>>,
    /// Set by shutdown to signal the drain thread to exit.
    stop: Arc<AtomicBool>,
    /// Output file path, kept for diagnostics.
    out_path: PathBuf,
    /// CUPTI sidecar (Activity API). None if CUPTI failed to load.
    cupti: Option<cupti::CuptiSidecar>,
}

/// Returns whether the profiler is currently active (env var was on at init).
pub fn is_active() -> bool {
    PROFILER.get().map(|m| m.lock().is_some()).unwrap_or(false)
}

/// Returns the output path, if the profiler is active.
pub fn output_path() -> Option<PathBuf> {
    PROFILER.get()?.lock().as_ref().map(|p| p.out_path.clone())
}

/// Initialize the profiler. Returns `Ok(true)` if it activated, `Ok(false)` if
/// the env var was off. `Err` only on actually broken state (CUDA failure,
/// unwritable output).
///
/// Calling `init` while the profiler is already active returns
/// `Err(AlreadyInit)`. Calling it after a successful `shutdown` rebuilds the
/// CTA-ring half of the profiler, but the **CUPTI sidecar is one-shot per
/// process**: CUPTI's `cuptiActivityRegisterCallbacks` cannot be torn down
/// safely, so re-init produces a profiler with the CTA ring active and the
/// CUPTI sidecar disabled. This is rarely a problem in practice — the
/// expected lifecycle is exactly one init/shutdown pair per process.
pub fn init() -> Result<bool, ProfilerError> {
    let cell = PROFILER.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock();
    if guard.is_some() {
        return Err(ProfilerError::AlreadyInit);
    }

    // Env-var gate.
    let enabled = std::env::var(ENV_ENABLE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"))
        .unwrap_or(false);
    if !enabled {
        // Make sure the device-visible ctx is zeroed (mask == 0 — probes are
        // no-ops). It is already, but be explicit for callers that build with
        // the feature and toggle env-var on/off mid-process.
        unsafe { ffi::shadow_clear_cta_ctx() };
        return Ok(false);
    }

    let ring_bytes = std::env::var(ENV_RING_BYTES)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_RING_BYTES);
    let out_path: PathBuf = std::env::var(ENV_OUT)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("shadow_profile.bin"));
    // Clamp to a reasonable range: 0 would spin the drain thread; very large
    // values would make shutdown wait minutes/years on the final sleep.
    let drain_ms = std::env::var(ENV_DRAIN_INTERVAL_MS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_DRAIN_INTERVAL_MS)
        .clamp(1, 60_000);

    // Ensure CUDA is initialized on the current device. The cuda-common helper
    // does the cudaFree(NULL) primary-context bring-up for us.
    openvm_cuda_common::common::set_device()
        .map_err(|e| ProfilerError::Cuda(format!("set_device: {e:?}")))?;

    // Allocate the pinned-mapped ring + head counter.
    let host_ring = ring::HostRing::allocate(ring_bytes)
        .map_err(|e| ProfilerError::Cuda(format!("HostRing alloc: {e}")))?;

    // Open the binary log writer (process-shared via Arc<Mutex>).
    let mut writer = log_writer::LogWriter::create(&out_path)?;
    // Front-matter: process start + every kernel registered before init.
    let pid = std::process::id();
    let walltime_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let gpu_index_i = openvm_cuda_common::common::get_device().unwrap_or(0);
    let gpu_index = gpu_index_i as u32;
    let gpu_name = ffi::device_name(gpu_index_i).unwrap_or_default();
    let _ = writer.write_record(&record::Record::ProcessStart {
        pid,
        gpu_index,
        start_walltime_ns: walltime_ns,
        gpu_name,
    });
    for (id, name) in kernel_id::registered_kernels() {
        let _ = writer.write_record(&record::Record::KernelName { id, name });
    }
    let writer = Arc::new(Mutex::new(writer));

    // Spawn the drain thread.
    let stop = Arc::new(AtomicBool::new(false));
    let drain = ring::spawn_drain(host_ring.clone(), writer.clone(), stop.clone(), drain_ms);

    // Try to start the CUPTI sidecar. Failure is non-fatal: we just lose the
    // host-timeline context but keep the per-CTA data.
    let cupti = match cupti::CuptiSidecar::start(writer.clone()) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!("cuda-profiler: CUPTI sidecar disabled: {e}");
            None
        }
    };

    // Publish the ctx to every instrumented kernel launch site.
    let ctx = host_ring.ctx();
    unsafe { ffi::shadow_set_cta_ctx(ctx) };

    tracing::info!(
        out = %out_path.display(),
        ring_bytes,
        drain_ms,
        cupti = cupti.is_some(),
        "cuda-profiler: activated"
    );

    *guard = Some(ActiveProfiler {
        _ring: host_ring,
        drain: Some(drain),
        stop,
        out_path,
        cupti,
    });
    Ok(true)
}

/// Shutdown the profiler. Idempotent.
///
/// Lifecycle ordering matters here: in-flight kernel launches captured the
/// old `CtaProbeCtx` by value at launch time, so they will keep writing
/// to the ring until the GPU is *actually idle*. We:
///
///   1. Clear the device-visible ctx so any future launch sees `mask == 0` and short-circuits the
///      probe.
///   2. `cudaDeviceSynchronize()` to drain in-flight kernels that captured the previous
///      (still-valid) ctx. Without this step, the `cudaFreeHost` that fires when we drop `_ring`
///      can race against a CTA's atomicAdd and slot store, with undefined consequences (the freed
///      pinned-mapped pages can be reused by an unrelated allocation by then).
///   3. Stop CUPTI (also after sync, so its activity records cover all kernels that ran).
///   4. Drain thread does a final sweep, then exits.
///   5. `_ring` drops -> `cudaFreeHost`.
pub fn shutdown() {
    let Some(cell) = PROFILER.get() else { return };
    let Some(mut active) = cell.lock().take() else {
        return;
    };

    unsafe { ffi::shadow_clear_cta_ctx() };

    // Wait for every in-flight kernel that captured the previous ctx to
    // finish. cudaDeviceSynchronize is global to the current device; that's
    // appropriate because shadow_cta_ctx is a process-global symbol that any
    // stream's launch may have read. Best-effort: log on failure.
    let rc = unsafe { ffi::cudaDeviceSynchronize() };
    if rc != 0 {
        tracing::warn!(
            rc,
            "cuda-profiler: cudaDeviceSynchronize failed at shutdown"
        );
    }

    // Stop the CUPTI sidecar after device sync so all kernel/memcpy activity
    // up to this point is captured.
    if let Some(cupti) = active.cupti.take() {
        cupti.stop();
    }

    // Tell the drain thread to do one last sweep + exit.
    active.stop.store(true, Ordering::Release);
    if let Some(h) = active.drain.take() {
        let _ = h.join();
    }

    tracing::info!(out = %active.out_path.display(), "cuda-profiler: shut down");
    // active dropped here -> ring freed via HostRingInner::drop.
}
