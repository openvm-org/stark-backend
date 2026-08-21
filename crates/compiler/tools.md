# Compiler tools

A user guide for the artifacts and helpers around the crypto-compiler
pipeline. These are the knobs a caller of
[`GraphCompiler`](src/graph_exe.rs) reaches for when tuning a compile,
capturing a graph for offline replay, or debugging a scheduler.

## Compiler options

Every `GraphCompiler` knob is expressed as a TOML document; the schema
is [`GraphCompilerConfig`](src/graph_compiler_config.rs) and the shipped
default profile lives at [`cc_default_config.toml`](cc_default_config.toml).

Top-level sections:

| Section            | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `[device]`         | `kind = "cuda" \| "cpu_pinned" \| "cpu_paged"` (+ `ordinal` for CUDA) |
| `[module_compiler]`| `nvcc`, `arch`, `extra_nvcc_flags`, `dump_dir`, `verbosity`, `check_accesses`, `nvcc_timeout_secs` — passthrough to the module-level nvcc invocation |
| `[scheduler]`      | `mode = "list_v1" \| "list_v2"` plus per-mode knobs (see below) |
| `[kernel_cache]`   | `kind = "enabled" \| "disabled"`; when enabled, `directory` / `max_kernels` / `max_bytes` |
| `[fusion]`         | `kind = "v2" \| "off" \| "v1"` (v1 is deprecated); v2 knobs mirror [`FusionOptionsV2`](src/passes/fusion_v2/mod.rs) |

Scheduler knobs (see [`planner`](src/planner/mod.rs)):

- **`list_v1`** — profile-guided list scheduler with beam look-ahead.
  `max_concurrency`, `max_memory_bytes`, `lookahead_k`, `beam`, `w_cp`,
  `w_mem`, `mem_target_frac`.
- **`list_v2`** — persistent-beam scheduler with parallel fan-out.
  `num_streams`, `max_memory_bound`, `num_beams`, `beam_depth`,
  `frontier_cap`, `w_m`, `w_t`, `w_c`. A working example lives in
  [`cc_list_v2_config.toml`](cc_list_v2_config.toml).

Loading a config:

```rust
let compiler = GraphCompiler::from_toml("crates/compiler/cc_default_config.toml")?;
// or, from an already-parsed struct:
let compiler = GraphCompiler::from_config(cfg)?;
```

**Fields not in the TOML surface** (set programmatically on the builder):
symbol bindings (`GraphCompiler::symbol`), fusion-v2 `estimator` /
`artifact` / `graph_symbols` (hardware- and graph-specific), and per-node
timings for the list schedulers (populated from profiling, not tuning).

The `CC_CONFIG` env var (see [`frac_bench_utils`](../cuda-backend/src/logup_zerocheck/frac_bench_utils.rs))
is the standard way the fractional-sumcheck benches pick up a profile
without recompiling. The caller-supplied `device` overrides whatever the
TOML says so the bench keeps control of the GPU ordinal.

Module-compiler env fallbacks (`NVCC`, `CRYPTO_COMPILER_CUDA_ARCH`,
`CRYPTO_COMPILER_DUMP_IR`, `CRYPTO_COMPILER_VERBOSITY`,
`CRYPTO_COMPILER_CHECK_ACCESSES`, `NVCC_TIMEOUT_SECS`) are consulted only
when the TOML omits the corresponding field.

## Graph dump

Two dump surfaces exist, one for humans and one for tools:

### Human-readable text dumps

- `GraphBuilder::print()` — reflects the pre-pass IR (buffers, node kinds,
  fusion history). Written to `<dir>/<name>.graph.txt` by the
  `dump_fractional_sumcheck_ir` test.
- `GraphExe::print()` — post-fusion + post-DCE IR with launch metadata,
  scratch layout, and stream assignment. Written to `<dir>/<name>.exe.txt`
  via the `FRAC_BENCH_DUMP_EXE=<path>` env var (see
  [`fractional_ir.rs`](../cuda-backend/src/logup_zerocheck/fractional_ir.rs)).
- Per-pass module IR: set `[module_compiler].dump_dir` (or
  `CRYPTO_COMPILER_DUMP_IR`) and pick `verbosity = "basic"` / `"verbose"`
  to control how much lowered HIR gets emitted per module.

### Cytoscape JSON (browser-facing)

`GraphBuilder::to_cytoscape_json{,_with_timings}()` writes a self-contained
`elements` document plus a top-level `modules` map (one HIR body per unique
`Arc<ir::Module>`) and, optionally, an embedded [`GraphInfo`](src/graph_info.rs)
`timings` block. Kernel nodes reference their `module` by key; the viewer
resolves it to the HIR body. Fields on each node include I/O shapes,
producer/consumer lists, `param_bindings`, `fusion_history`, and (when
`GraphInfo` is attached) `timing_mean_ms` / `timing_std_ms` / `timing_kind`.

The `dump_fractional_sumcheck_cytoscape` test is the canonical entrypoint.
It writes `<dir>/fractional_sumcheck_n{N}.cy.json`:

```bash
FRAC_LOG_N=10 \
    cargo nextest run -p openvm-cuda-backend --features graph-ir \
        --run-ignored all --no-capture \
        -E 'test(dump_fractional_sumcheck_cytoscape)'
```

Relevant env vars:

- `CRYPTO_COMPILER_DUMP_IR` — output directory (default `target/ir_dump/`).
- `FRAC_LOG_N` — log₂ leaf count (default 8). Ignored when
  `CC_GRAPH_INPUT_PATH` is set.
- `CC_GRAPH_INPUT_PATH` — load a preloaded bincode `SerializableGraphBuilder`
  instead of building + fusing from source; blackbox closures become panic
  placeholders (fine for cytoscape emission).
- `CC_TIMING_JSON_PATH` — overlay a `GraphInfo` JSON so the viewer surfaces
  per-node timings and per-variant cumulative cost.

To capture a matched (`graph.bin`, `cy.json`, `timings.json`) triple in one
pass, run the fractional-sumcheck bench with `CC_CY_DUMP_PATH=<dir>` — see
`bench_fractional_sumcheck_eager_vs_ir` in [`fractional_ir.rs`](../cuda-backend/src/logup_zerocheck/fractional_ir.rs).
That path is the recommended way to produce a snapshot that lines up
index-by-index with the compiled `ExeNode` list (post-fuse+dce), which is
what makes per-node timing overlays and `sim_scheduler` replays valid.

## Graph visualizer

[`scripts/serve_graph.py`](../../scripts/serve_graph.py) is a tiny HTTP
server that renders a `.cy.json` dump with Cytoscape.js in the browser:

```bash
python3 scripts/serve_graph.py --port 8086 \
    target/ir_dump/fractional_sumcheck_n1024.cy.json
```

Then open http://localhost:8086. The JSON is re-read on every page reload,
so regenerate the dump and hit F5 to see the new graph.

Layout is computed offline. Two engines both produce top-to-bottom layered
DAG layouts:

- **`dot`** (Graphviz) — full Sugiyama with crossing minimization. Best
  quality but O(|E|²) per mincross iteration; times out past a few
  thousand edges. Requires `apt install graphviz` / `brew install graphviz`.
- **`python-layered`** (built-in) — longest-path ranking + median-heuristic
  ordering, no dummy nodes. Sub-second on graphs `dot` can't finish, at the
  cost of more edge crossings.

Positions are cached in a `<dump>.pos.json` sidecar keyed on the dump's
mtime plus the layout knobs, so switching engines / rankdir invalidates
correctly. Selected flags:

| Flag                          | Effect                                                             |
| ----------------------------- | ------------------------------------------------------------------ |
| `--engine auto\|dot\|python-layered` | Engine choice (`auto` uses `--dot-max-edges`).              |
| `--dot-max-edges N`           | `auto` uses `dot` when edges ≤ N, else `python-layered` (dflt 2000). |
| `--rankdir TB\|LR\|BT\|RL`    | Flow direction (dot only).                                         |
| `--dup-leaves-threshold N`    | Duplicate `Const`/`Input` nodes with > N consumers to collapse crossing bundles (dflt 4; 0 to disable). |
| `--max-nodes N`               | Fall back to breadthfirst above this size (dflt 25000).            |
| `--no-layout`                 | Skip offline layout entirely.                                      |

**In the UI:**

- Click any node — opens a right-side panel with I/O shapes, param
  bindings, producer/consumer chips (click to focus), the node's HIR
  and (for Kernels) the shared module IR, plus a Reingold-Tilford-lite
  SVG fusion-history tree (click nodes to inspect intermediate IR).
- **Stats** button (top bar) — kernel/blackbox frequency tables, timings
  summary and per-variant cumulative-cost table (when the dump embedded a
  `GraphInfo`). Click rows for module IR / node lists.
- Wheel = zoom, drag = pan (both in the main canvas and in the fusion
  tree). Long edges fade adaptively so the viewer isn't dominated by a
  single high-fanout leaf.

## Graph serialization format

[`SerializableGraphBuilder`](src/graph_serializer.rs) is the bincode wire
format for a `GraphBuilder`. It captures buffers, nodes, fusion history,
constants, and the pre-fuse and post-fuse hashes; blackbox closures are
elided and re-attached from a caller-supplied builder on load.

Two round-trip modes:

- **Online** (`into_graph_builder(existing, ctx)`) — validates the
  snapshot's hashes and buffer layouts against a freshly-built `existing`
  builder, then re-attaches its blackbox closures. Used by
  `load_or_compile_and_dump`.
- **Offline** (`into_graph_builder_offline()`) — returns a builder whose
  blackboxes are panic-placeholders; constants come back as `HostBuf`
  regardless of their original residency. No GPU context needed; this
  is what `sim_scheduler` and the ATG dump-loaders use.

Standard extension: `.graph.bin`. Companion files that ship alongside a
graph bin:

| Path                        | Content                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `<name>.graph.bin`          | `bincode(SerializableGraphBuilder)` — pre-pass or post-fuse+dce depending on how it was produced |
| `<name>.cy.json`            | Cytoscape doc from `to_cytoscape_json{,_with_timings}`        |
| `<name>.timings.json`       | `serde_json(GraphInfo)` — per-node mean/std ms + graph hash   |

The `sim_scheduler` binary and `sim_scheduler.rs` prints a warning when a
snapshot's `content_hash == original_hash` (pre-pass) so you don't
accidentally simulate a graph the timings weren't collected against.

### Producing snapshots

- **Pre-pass snapshot** — [`load_or_compile_and_dump`](../cuda-backend/src/logup_zerocheck/frac_bench_utils.rs)
  writes one to `CC_GRAPH_DUMP_PATH` (base path; size suffix `.n{n}` is
  auto-injected). Re-runs with the same path skip graph *construction*
  but still fuse + compile; kernels reuse the on-disk cache.
- **Post-fuse+dce snapshot** — `load_or_compile_and_dump_with_hook` +
  `fused_snapshot_path` (or `CC_CY_DUMP_PATH=<dir>` on the fractional
  bench) uses `GraphCompiler::compile_with_post_fuse_hook` to serialize
  the graph *after* fusion. Post-pass node indices align with
  `GraphExe::collect_graph_info`'s `GraphInfo::nodes`, so timings replay
  cleanly against this dump.

## Scheduler sim

[`sim_scheduler`](src/bin/sim_scheduler.rs) is an offline harness that
replays the memory + stream planner against a captured graph without
touching a GPU:

```bash
cargo run --release -p crypto-compiler --features planner --bin sim_scheduler -- \
    crates/compiler/cc_default_config.toml \
    graph_ir_dumps/frac_ir.n1048576.graph.bin \
    graph_ir_dumps/frac_ir.n1048576.timings.json
```

Inputs:

1. A compiler-config TOML (only `[scheduler]` and `[device]` actually
   affect the sim; the rest is inert).
2. A **post-fuse+dce** `SerializableGraphBuilder` bincode dump (produced
   via `fused_snapshot_path` — the tool does *not* run fusion; if the
   node counts disagree with the timings, you handed it a pre-pass
   snapshot).
3. A `GraphInfo` JSON of per-node mean/std ms. The `graph_hash` field
   must match the snapshot's `original_hash`; a mismatch triggers a
   warning.

Report sections:

- **inputs / outputs** — buffer name, element count, byte size, device.
- **schedule** — scheduler mode, stream count, cross-stream event count,
  peak pool bytes, instruction breakdown.
- **planner cost** — wall-time for `access_from_node`, ATG build, `plan`,
  `validate_plan`, `perf_est`.
- **validation** — `validate_plan` output aggregated into
  `MissingCrossStreamSync` (WAR-alias / real SSA / real multi-writer),
  `DataDepRace`, `PoolLifetimeOverlap`, with the first 20 raw errors.
  Multi-writer buffer counts are surfaced too — most sync misses on
  real graphs come from carried_outputs aliasing rather than genuine
  bugs. Sim exits non-zero if any errors remain.
- **predicted latency** — `perf_est.time` alongside a local simulator's
  makespan (should agree to fp noise), plus `sum(node_times)` and the
  speedup vs serial.
- **per-stream** — busy / wait / idle ms, utilization %, `WaitOn`
  count per stream.

Related offline-planner tools (both take the same `graph.bin` +
`timings.json` inputs; use the `planner` feature):

- `examples/bench_abstract_planners.rs` — runs the heuristic packer and
  `list_v1` at `max_concurrency ∈ {1,2,4,8}` and prints wall-clock +
  peak-pool per plan.
- `examples/bench_list_v2_ops.rs` — micro-benchmarks `list_v2`'s per-step
  primitives (`cost`, `put_on`, `state.clone()`). Env: `V2_OPS_STREAMS`,
  `V2_OPS_MAX_MEM_GIB`.

`load_abstract_timing_graph(graph_path, timing_path)` (in
[`planner::abstract_timing`](src/planner/abstract_timing.rs)) is the
shared entry point they all funnel through — a good starting place for
writing a new offline planner experiment.
