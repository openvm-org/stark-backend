//! Profiler bring-up.
//!
//! When `openvm-cuda-backend` is built with `--features profiler`, this module:
//!
//!   1. Auto-initializes the `openvm-cuda-profiler` crate at first use of the backend (lazy, behind
//!      `Once`). This pulls in env-var configuration (`SHADOW_PROFILER=1` etc.) without requiring
//!      callers to do anything.
//!   2. Registers the names of all instrumented kernels so the binary log has a `KernelName` entry
//!      for every `kernel_id` the device might emit.
//!
//! The hook here is best-effort: any failure logs and continues with the
//! profiler disabled. Callers do not pay any runtime cost when the feature is
//! disabled at compile time (this module is `#[cfg(feature = "profiler")]`).
//!
//! `init_profiler_once()` is called from the public CUDA engine constructor.

use std::sync::Once;

use openvm_cuda_profiler as prof;

static INIT: Once = Once::new();

/// Idempotently initialize the profiler. Safe to call from multiple threads.
pub fn init_profiler_once() {
    INIT.call_once(|| {
        // Register every known kernel name *before* init, so the registry is
        // present when the profiler dumps it into the log header.
        register_known_kernels();
        match prof::init() {
            Ok(true) => {
                tracing::info!(
                    out = ?prof::output_path(),
                    "openvm-cuda-backend: profiler activated"
                );
            }
            Ok(false) => {
                // Compiled in, runtime-disabled. No log line — the env var is
                // intentionally unset and noise here would be wrong.
            }
            Err(e) => {
                tracing::warn!("openvm-cuda-backend: profiler init failed: {e}");
            }
        }
    });
}

/// Register all kernel names that we annotate with `SHADOW_KERNEL_BEGIN(name)`.
/// Keep in lock-step with the literals in the .cu sources.
///
/// New kernels go here when they get instrumented, *not* in the .cu file
/// alone — without a host-side registration the binary log will only contain
/// the FNV-1a hash, not the human-readable name.
fn register_known_kernels() {
    // Kernel names to match those used in SHADOW_KERNEL_BEGIN("..") on the
    // device side. The device-side constexpr FNV-1a hash and the host-side
    // `register_kernel` MUST agree on the spelling.
    //
    // Add new entries here as kernels become instrumented; an entry without a
    // matching `SHADOW_KERNEL_BEGIN("name")` is harmless (it just means there
    // will never be CTA records carrying that id) so this list can lead the
    // instrumentation work, but each `SHADOW_KERNEL_BEGIN` *must* have its
    // name listed here or its CTA records will only carry a hex id.
    const KERNEL_NAMES: &[&str] = &[
        // poly.cu — stage updates used by the sumcheck/zerocheck round drivers.
        "eq_hypercube_stage_ext",
        "mobius_eq_hypercube_stage_ext",
        "eq_hypercube_nonoverlapping_stage_ext",
        "eq_hypercube_interleaved_stage_ext",
    ];

    for name in KERNEL_NAMES {
        let _ = prof::register_kernel(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_idempotent() {
        // Calling init_profiler_once twice should not double-register.
        init_profiler_once();
        init_profiler_once();
    }
}
