//! SHDWPROF binary log format and Perfetto exporter.
//!
//! The crate now contains only the producer-agnostic pieces of the profiler
//! pipeline:
//!
//!   * [`record`] — the on-disk record format (`SHDWPROF v1.0`) and a
//!     streaming reader/writer.
//!   * [`kernel_id`] — the FNV-1a 32-bit hash that maps kernel names to the
//!     `u32` ids stored in `CtaRecord::kernel_id`.
//!
//! The actual instrumentation lives in [`nvbit-tool/`], a standalone NVBit
//! tool that produces `SHDWPROF` records by intercepting every kernel
//! launch at the SASS level. The bin target [`cuda-profiler-assemble`]
//! consumes those logs and emits a Perfetto Chrome trace.
//!
//! Earlier revisions of this crate also held an in-process source-level
//! profiler (`cta_probe.cuh` macros, a CUPTI sidecar, a host ring buffer,
//! and a drain thread); those have been removed in favor of the
//! NVBit-only path. The wire format is unchanged so logs produced by the
//! old infrastructure still parse with this assembler.

pub mod kernel_id;
pub mod record;

pub use kernel_id::fnv1a;
