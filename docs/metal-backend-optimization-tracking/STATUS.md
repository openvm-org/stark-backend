# Milestone Status Board

| Milestone | Title | Status | Owner | Branch | PR | Last Update | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [0](./milestone-0.md) | Baseline Harness and Guardrails | completed | codex | `metal-backend` |  | 2026-02-19 | Baseline trace + guardrails recorded |
| [1](./milestone-1.md) | Batch Synchronization and Dispatch Plumbing | completed | codex | `perf/m1` |  | 2026-02-20 | Staged dispatch added for round0/reduction hot paths; merged into `metal-backend` |
| [2](./milestone-2.md) | Stacked Trace Copy Path Optimization | completed | codex | `perf/m2` |  | 2026-02-20 | Batched stacked trace blit copies; merged into `metal-backend` |
| [3](./milestone-3.md) | Unified-Memory Copy Elimination | completed | codex | `perf/m3` |  | 2026-02-20 | Unified-memory row/root reads and cached Merkle layer pointers; merged into `metal-backend` |
| [4](./milestone-4.md) | Trace Transport Layout Simplification | completed | codex | `perf/m5` |  | 2026-02-20 | Removed strided transport double buffering + collapse kernel dispatch; merged into `metal-backend` |
| [5](./milestone-5.md) | LogUp Round0 Scratch Reuse and Reduction Batching | completed | codex | `perf/m5` |  | 2026-02-20 | Round0 scratch reuse + single staged sync across cosets; merged into `metal-backend` |
| [6](./milestone-6.md) | Device-Aware `batch_ntt_small` Tuning | completed | codex | `metal-backend` |  | 2026-02-20 | Device-aware launch candidate regressed release `keccakf`; reverted by `fb16bd32` |
| [7](./milestone-7.md) | End-to-End Validation and Profiling Report | completed | codex | `metal-backend` |  | 2026-02-20 | Release-mode validation finalized; post-revert total prove time improved to `2.42s` |
| [8](./milestone-8.md) | Benchmark Harness Expansion and KPI Contract | not_started |  |  |  | 2026-02-20 | Phase 2 baseline setup for `keccakf` and LogUp paths |
| [9](./milestone-9.md) | Round0/LogUp Coset Dispatch Fusion | not_started |  |  |  | 2026-02-20 | Reduce round0 dispatch/sync overhead and scratch churn |
| [10](./milestone-10.md) | LogUp MLE Batching and Fallback Elimination | not_started |  |  |  | 2026-02-20 | Remove oversized-trace fallback serialization and improve MLE throughput |
| [11](./milestone-11.md) | Main Trace Commit and RS Pipeline Optimization | not_started |  |  |  | 2026-02-20 | Reduce `main_trace_commit` and `rs_code_matrix` spans |
| [12](./milestone-12.md) | Trace Transport and Layout Path Optimization | not_started |  |  |  | 2026-02-20 | Lower transport allocations and dispatch count |
| [13](./milestone-13.md) | Openings and WHIR Pipeline Tightening | completed | codex | `perf/m13` |  | 2026-02-20 | WHIR fold scratch reuse + batched stacked-reduction MLE window dispatch/readback |
| [14](./milestone-14.md) | Cross-Phase Scheduling and Buffer Pool | not_started |  |  |  | 2026-02-20 | Introduce shared dispatch scheduling and temporary buffer pooling |
| [15](./milestone-15.md) | Final 2x Validation and Residual Backlog | not_started |  |  |  | 2026-02-20 | Final benchmark gate and gap-to-target report |
