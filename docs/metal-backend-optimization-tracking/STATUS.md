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
