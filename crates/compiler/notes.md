# Notes and misc details

## Graph Mutation Notes

The `GraphBuilder` is lowered to an `AbstractTimingGraph` (ATG) to perform scheduling. This is necessary to handle some hairy mutation details that the scheduler shouldn't care about (in principle). The invariant we want to enforce is that any topological ordering of the `AbstractTimingGraph` respects the RAW/WAR/WAW dependencies of the original graph (some nodes mutate their inputs). 

To this end, we first enforce the condition (prior to constructing the ATG) that a `BufId` is mutated by node `n` iff it is present in both `n`'s consumer and producer list.

`access_from_node` enforces this per GraphNode variant:

| Variant           | reads                       | writes                              | Notes                                   |
| ----------------- | --------------------------- | ----------------------------------- | --------------------------------------- |
| `Const`           | `[]`                        | `[c.buf]`                           | pure initial producer — no prior value  |
| `Memset`          | `[m.node]`                  | `[m.node]`                          | self-carry — overwrites in place        |
| `Memcpy`          | `[m.src, m.dst]`            | `[m.dst]`                           | dst overwrites its prior slot           |
| `Kernel`          | `k.inputs`                  | `k.outputs`                         | in-place iff an output is also an input |
| `BlackboxKernel`  | `k.inputs`                  | `k.carried_outputs ∪ k.outputs`     | `carried_outputs ⊆ inputs` by construction |


In other words, a real buffer `b` is **mutated** iff some node `v` has `b ∈ node_produces[v] ∩ node_consumes[v]`.

After that, for any buffer that's mutated, we insert synthetic directed edges between any two users of that buffer. eg. for a mutation chain 

```
Memset(b0)
Memset(b0)
Memset(b0)
```

we insert synthetic edges so the directed graph is 
```
[b0, s0] = Memset(b0)
[b0, s1] = Memset(b0, s0)
[b0, s2] = Memset(b0, s1)
```

all synthetic edges are `BufId`s with zero size. The old `BufId` that's mutated is kept because the memory scheduler must make the memory offset stable, as compared to renaming mutated `BufId`s:

```
[b1] = Memset(b0)
[b2] = Memset(b1)
[b3] = Memset(b2)
```
because in this situation the memory scheduler doesn't know that `b0, b1, b2, b3` are actually the same buffer.

### Input preservation

Currently it's not guaranteed that the inputs are preserved across a graph execution. Meaning that after a graph execution, the `.get_input_ptr` method's buffer may not be the same. This is to save memory.

So you'd have to set the inputs everytime for a graph launch.

If you want to preserve the input just declare it as a graph output.

## Benchmarking

All fractional-sumcheck benches are `#[ignore]`d — they need a GPU and take tens of seconds
to minutes. Run them explicitly with `--run-ignored all --no-capture` (`nextest`) or
`--ignored --nocapture --exact` (`cargo test`).

### Common configuration

| env var                        | purpose                                                              |
| ------------------------------ | -------------------------------------------------------------------- |
| `CUDA_VISIBLE_DEVICES`         | GPU selection (project convention: GPU 5).                           |
| `CC_CONFIG`                    | Path to compiler-config TOML (`cc_default_config.toml` ships list_v2). |
| `NSYS_ENABLED=1`               | Enable `cudaProfilerStart/Stop` + NVTX ranges in every bench.        |
| `CC_GRAPH_DUMP_PATH=<file>`    | Cache the compiled `GraphExe` between runs (skips fusion/schedule).  |
| `CC_CY_DUMP_PATH=<dir>`        | Dump post-fuse+dce graph + cytoscape JSON + timings JSON.            |
| `FRAC_BENCH_DUMP_EXE=<file>`   | Dump `exe.print()` (used by `bench_fractional_sumcheck_eager_vs_ir`).|
| `FRAC_LOG_N=<csv>`             | Leaf-count log2 list (default varies by bench).                      |
| `FRAC_ROUND=<n>` / `FRAC_ROUNDS=<csv>` | Which outer round(s) the pipelined benches build for.        |
| `CC_STREAMS_SWEEP=<csv>`       | Override list_v2 stream sweep in `all_schedulers_nsys`.              |

Recommended nsys flags (per `AGENTS.md`):

```
nsys profile \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node \
    --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx \
    --wait=all \
    --force-overwrite=true -o <out_name>
```

### Fractional sumcheck benches

The table lists every `bench_*` test in the fractional-sumcheck source tree, its purpose,
and where it lives.

| Bench                                          | File                                | Purpose                                                                                             |
| ---------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------- |
| `bench_fractional_sumcheck_eager_vs_ir`        | `fractional_ir.rs`                  | Full fractional sumcheck — eager vs `GraphExe.launch_graph()` at every `FRAC_LOG_N` (default `16,20`). |
| `bench_fractional_sumcheck_eager_vs_irv2`      | `fractional_sumcheck_gpu_irv2.rs`   | irv2 (fusion-optimised) full sumcheck vs eager, with `exe.run()` and `launch_graph()` variants.     |
| `bench_pipelined_ir_vs_eager`                  | `fractional_ir_pipelined.rs`        | Single outer round `FRAC_ROUND` of the α-tiled pipelined driver vs eager.                           |
| `bench_pipelined_full_sumcheck_vs_eager`       | `fractional_ir_pipelined.rs`        | Full pipelined sumcheck vs eager at every `FRAC_LOG_N`.                                             |
| `bench_pipelined_ir_sweep`                     | `fractional_ir_pipelined.rs`        | Per-`j` single-round pipelined graphs across `FRAC_ROUNDS` (default `4,10,16,20,24`).               |
| `bench_pipelined_full_sumcheck_sweep`          | `fractional_ir_pipelined.rs`        | Eager + IR + pipelined drivers side-by-side at every `FRAC_LOG_N` (default `16,20,24`).             |
| `bench_fractional_sumcheck_all_schedulers_nsys`| `fractional_sumcheck_gpu_irv2.rs`   | Sweep scheduler variants (list_v1 / list_v2 knobs) on the irv2 graph, single nsys window.           |

All bench functions live in the `tests` submodule of their file — the nextest filter is
`test(=logup_zerocheck::<file>::tests::<bench>)`.

### Command templates

`$CFG=cc_default_config.toml` in every command below ships list_v2; swap
to a list_v1 profile per the example block in the file to compare.
Default profiling target is GPU 5.

**`bench_fractional_sumcheck_eager_vs_ir`** — default full-proof bench, `FRAC_LOG_N=16,20`:

```bash
CUDA_VISIBLE_DEVICES=5 NSYS_ENABLED=1 \
  CC_CONFIG=$(pwd)/crates/compiler/$CFG FRAC_LOG_N=24 \
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx --wait=all --force-overwrite=true -o frac_ir_n24 \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_ir::tests::bench_fractional_sumcheck_eager_vs_ir)'
```

Dump the fused graph + timings and serve them in the Cytoscape viewer
(no nsys; writes `frac_ir.n{N}.{graph.bin,cy.json,timings.json}` to
`$CC_CY_DUMP_PATH`):

```bash
CUDA_VISIBLE_DEVICES=5 FRAC_LOG_N=12 \
  CC_CONFIG=$(pwd)/crates/compiler/$CFG \
  CC_CY_DUMP_PATH=$(pwd)/target/frac_cy_dump \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture --test-threads=1 \
    -E 'test(=logup_zerocheck::fractional_ir::tests::bench_fractional_sumcheck_eager_vs_ir)'

python3 scripts/serve_graph.py --port 8086 \
  target/frac_cy_dump/frac_ir.n4096.cy.json
# then open http://localhost:8086
```

Non-nsys run (fast turnaround, no profiler overhead):

```bash
CUDA_VISIBLE_DEVICES=5 CC_CONFIG=$(pwd)/crates/compiler/$CFG FRAC_LOG_N=24 \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_ir::tests::bench_fractional_sumcheck_eager_vs_ir)'
```

Optional: dump the compiled exe or fused cytoscape graph:

```bash
FRAC_BENCH_DUMP_EXE=/tmp/frac_ir.exe CC_CY_DUMP_PATH=/tmp/frac_ir_dump ...
```

**`bench_fractional_sumcheck_eager_vs_irv2`** — fusion-optimised full sumcheck:

```bash
CUDA_VISIBLE_DEVICES=5 NSYS_ENABLED=1 \
  CC_CONFIG=$(pwd)/crates/compiler/$CFG FRAC_LOG_N=24 \
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx --wait=all --force-overwrite=true -o frac_irv2_n24 \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_sumcheck_gpu_irv2::tests::bench_fractional_sumcheck_eager_vs_irv2)'
```

**`bench_pipelined_ir_vs_eager`** — single-round pipelined driver (choose the round via `FRAC_ROUND`):

```bash
CUDA_VISIBLE_DEVICES=5 NSYS_ENABLED=1 FRAC_ROUND=12 \
  CC_CONFIG=$(pwd)/crates/compiler/$CFG \
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx --wait=all --force-overwrite=true -o pipelined_j12 \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_ir_pipelined::tests::bench_pipelined_ir_vs_eager)'
```

**`bench_pipelined_full_sumcheck_vs_eager`** — full pipelined sumcheck:

```bash
CUDA_VISIBLE_DEVICES=5 NSYS_ENABLED=1 FRAC_LOG_N=24 \
  CC_CONFIG=$(pwd)/crates/compiler/$CFG \
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx --wait=all --force-overwrite=true -o pipelined_full_n24 \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_ir_pipelined::tests::bench_pipelined_full_sumcheck_vs_eager)'
```

**`bench_pipelined_ir_sweep`** — per-round sweep across `FRAC_ROUNDS` (default `4,10,16,20,24`):

```bash
CUDA_VISIBLE_DEVICES=5 NSYS_ENABLED=1 FRAC_ROUNDS=4,10,16,20,24 \
  CC_CONFIG=$(pwd)/crates/compiler/$CFG \
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx --wait=all --force-overwrite=true -o pipelined_sweep \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_ir_pipelined::tests::bench_pipelined_ir_sweep)'
```

**`bench_pipelined_full_sumcheck_sweep`** — eager + IR + pipelined per `FRAC_LOG_N`:

```bash
CUDA_VISIBLE_DEVICES=5 NSYS_ENABLED=1 FRAC_LOG_N=16,20,24 \
  CC_CONFIG=$(pwd)/crates/compiler/$CFG \
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx --wait=all --force-overwrite=true -o pipelined_full_sweep \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_ir_pipelined::tests::bench_pipelined_full_sumcheck_sweep)'
```

**`bench_fractional_sumcheck_all_schedulers_nsys`** — scheduler-variant sweep on the irv2 graph.
Set `CC_STREAMS_SWEEP` to override the list_v2 stream sweep:

```bash
CUDA_VISIBLE_DEVICES=5 NSYS_ENABLED=1 FRAC_LOG_N=24 \
  CC_STREAMS_SWEEP=1,4,8 \
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx --wait=all --force-overwrite=true -o irv2_schedulers_n24 \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_sumcheck_gpu_irv2::tests::bench_fractional_sumcheck_all_schedulers_nsys)'
```

### Handy variants

- **Use `cargo test` instead of nextest** (older CI or when `--test-threads=1 --exact` is
  required to serialise a GPU-heavy sweep):

  ```bash
  cargo test -p openvm-cuda-backend --features graph-ir --release \
    --lib logup_zerocheck::fractional_ir_pipelined::tests::bench_pipelined_full_sumcheck_sweep \
    -- --ignored --nocapture --test-threads=1 --exact
  ```

- **Warm the kernel cache before profiling**: run the target bench once without
  `NSYS_ENABLED` — the JIT'd modules land in `$HOME/.openvm/kernel_cache` and later
  profile runs skip nvcc.

- **Cache the compiled graph** across bench runs with `CC_GRAPH_DUMP_PATH=<file>` — cuts
  the ~20 s fusion + schedule compile cost when iterating on runtime knobs.

- **Read back the plan** with `sim_scheduler` (see `crates/compiler/src/bin/sim_scheduler.rs`):

  ```bash
  CC_CY_DUMP_PATH=graph_ir_dumps ...  # produce ir.n{n}.graph.bin + timings.json
  ./target/release/sim_scheduler crates/compiler/$CFG graph_ir_dumps/ir.n1048576.graph.bin \
      graph_ir_dumps/ir.n1048576.timings.json
  ```

## Environment variables reference

Every env var read by the compiler crate or the fractional-sumcheck benches.
Compiler-side vars are read once at builder / runtime setup; bench-side vars
are read inside the timed tests.

### Compiler (`crates/compiler`)

**`runtime::CompileOptions::default()`** — nvcc + verbosity knobs, populated
from the environment at first use:

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `NVCC`                             | nvcc binary path (default `"nvcc"`).                                                          |
| `CRYPTO_COMPILER_CUDA_ARCH`        | `-arch=` value (default `"native"`).                                                          |
| `CRYPTO_COMPILER_DUMP_IR`          | Directory for IR dumps; unset = no dumps.                                                     |
| `CRYPTO_COMPILER_VERBOSITY`        | `none` / `basic` / `verbose` (default `basic`).                                               |
| `CRYPTO_COMPILER_CHECK_ACCESSES`   | Run per-module access checker (`0` / `false` = off, else on).                                 |
| `NVCC_TIMEOUT_SECS`                | Per-kernel nvcc wall-time limit.                                                              |
| `CUDA_LINEINFO=1`                  | Pass `-lineinfo` to nvcc for source-line PTX debug info.                                      |

**`kernel_cache`** — on-disk cache location:

| var                | purpose                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `HOME`             | Default cache dir: `$HOME/.openvm/kernel_cache`.                                              |
| `XDG_CACHE_HOME`   | Fallback cache dir: `$XDG_CACHE_HOME/openvm/kernel_cache` when `HOME` is unset.               |

**`graph_exe` — compile / run debug knobs:**

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `GRAPH_EXE_DUMP_SCHEDULE=1`        | Dump per-stream schedule after memory plan.                                                   |
| `GRAPH_EXE_DUMP_MODULES=1`         | List unique module names during compile.                                                      |
| `GRAPH_EXE_TRACE=<stride>`         | Print an instruction trace every `stride` dispatches.                                         |
| `GRAPH_EXE_SLOW_INSTR_MS=<ms>`     | Arm a per-instruction wall-time watchdog; dumps the offending module.                         |
| `GRAPH_EXE_STOP_AT_INSTR=<idx>`    | Dump instruction `idx`'s module and return before dispatch.                                   |
| `GRAPH_EXE_SYNC_EACH_INSTR=1`      | Sync after each dispatch (serializes launches).                                               |
| `GRAPH_EXE_DISPATCH_WATCHDOG_MS=<ms>` | Kill the process if a single dispatch blocks longer than the given wall time.              |

**`passes/fusion` — fusion pass debug:**

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `FUSION_DEBUG=1`                   | Print per-pass reject aggregates.                                                             |
| `FUSION_DEBUG=2`                   | As above, plus dump the cost interpreter's HIR on cost-failure.                               |
| `FUSION_CHECK_SELECTED=1`          | Post-solve, re-verify each selected module compiles.                                          |

**`planner/list_v2`:**

| var                | purpose                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `LIST_V2_TRACE`    | Any value → chatty per-step trace of the persistent-beam scheduler.                           |

**`test_utils::maybe_bench()`** — micro-bench helper for unit tests:

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `BENCH_KERNEL`                     | Enable timing of a `TestModuleRunner` (unset = no-op).                                        |
| `BENCH_KERNEL_WARMUP`              | Warmup iterations (default 5).                                                                |
| `BENCH_KERNEL_ITERS`               | Timed iterations (default 50).                                                                |

**Compiler examples:**

| var                                | example                              | purpose                                                                     |
| ---------------------------------- | ------------------------------------ | --------------------------------------------------------------------------- |
| `V2_OPS_STREAMS`                   | `bench_list_v2_ops`                  | `num_streams` for the walk (default 8).                                     |
| `V2_OPS_MAX_MEM_GIB`               | `bench_list_v2_ops`                  | Memory bound in GiB (default 256).                                          |
| `NCU_ENABLED`                      | `profile_ntt_supra`                  | Wrap a single launch each in an NVTX `NCU_PROFILE` range.                   |

### CUDA-backend fractional-sumcheck benches (`crates/cuda-backend/src/logup_zerocheck`)

**`frac_bench_utils.rs`** — shared setup:

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `CC_CONFIG`                        | Path to a `GraphCompilerConfig` TOML; unset ⇒ programmatic defaults.                          |
| `FRAC_LOG_N`                       | Comma-separated `log2(leaves)` list (bench input sizes).                                      |
| `CC_GRAPH_DUMP_PATH`               | Base path for the pre-pass `GraphBuilder` snapshot (suffixed with `.n{n}`).                   |

**`fractional_ir.rs`** — plain IR driver + full-proof bench:

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `GRAPH_PATH`                       | Per-driver + per-size pre-pass graph snapshot (label + `.n{n}` suffix).                       |
| `CC_GRAPH_INPUT_PATH`              | Load a preloaded fused graph and skip build + fusion.                                         |
| `CC_TIMING_JSON_PATH`              | Inject a `GraphInfo` timings JSON into the cytoscape dump.                                    |
| `CC_CY_DUMP_PATH`                  | Where cytoscape dumps land.                                                                   |
| `FRAC_BENCH_DUMP_EXE`              | Write `exe.print()` next to each `log_n`.                                                     |
| `FRAC_BENCH_DRIVERS`               | Comma filter over driver labels (skips eviction pressure).                                    |
| `FRAC_BENCH_LOG_N`                 | Comma list of `log_n`s (default `"16,24"`).                                                   |
| `FRAC_BENCH_SCHEDULER`             | `v1` (default) or `v2` (profile-guided replan).                                               |
| `CC_STREAMS`                       | Max concurrency for the replanning scheduler (default 8).                                     |
| `NSYS_ENABLED`                     | Wrap the timed window in `cudaProfilerStart/Stop`.                                            |

**`fractional_sumcheck_gpu_irv2.rs`** — irv2 driver bench:

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `FRAC_V2_DUMP_GRAPH_ONLY=1`        | Skip the expensive compile at large `n`.                                                      |
| `CC_STREAMS_SWEEP`                 | Comma stream counts for the list-v1 sweep (default `"1,2,3"`).                                |
| Also honours `CRYPTO_COMPILER_DUMP_IR`, `CC_TIMING_JSON_PATH`, `NSYS_ENABLED`.                                                       |

**`fractional_ir_pipelined.rs`** — pipelined driver benches:

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `FRAC_ROUND`                       | Single-round GKR bench: which sumcheck round to test (default 12).                            |
| `FRAC_ROUNDS`                      | Multi-round bench: comma list (default `"4,10,16,20,24"`).                                    |
| `NSYS_ENABLED`                     | As above.                                                                                     |

**`fractional_ir_utils.rs`** — DSL-port replay harness:

| var                                | purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `FRAC_DSL_FUSION`                  | `off` disables fusion, `verbose` enables verbose logging, unset = default.                    |

**PrecomputeM tunables (`fractional.rs`, read by `fractional_ir.rs`, `fractional_ir_pipelined.rs`):**

| var                                        | purpose                                                                                       |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M`              | Master switch for PrecomputeM strategy selection.                                             |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_BLOCKS`   | Minimum tail-block count before PrecomputeM is preferred.                                     |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M_MIN_N`        | Minimum `rem_n` gate for PrecomputeM.                                                         |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M_TARGET_BLOCKS`| Target tail-block count for the M-build kernel.                                               |
| `SWIRL_CUDA_GKR_PRECOMPUTE_M_TAIL_TILE`    | Override auto-selected tail tile size.                                                        |

### CUDA-backend examples

| var                                | example                                       | purpose                                             |
| ---------------------------------- | --------------------------------------------- | --------------------------------------------------- |
| `BENCH_WARMUP` / `BENCH_ITERS` / `BENCH_SEED` | `bench_ir_dsl_ports`                    | Warmup (5) / iters (50) / RNG seed.                 |
| `W` / `T` / `DUMP_DIR`             | `dump_ir_eval_round`                         | Window / point-count / output dir.                  |
| `NUM_X` / `DUMP_DIR`               | `dump_ir_frac_compute_round`                 | Row count / output dir.                             |
| `DUMP_DIR`                         | `dump_fold_frac_fusion_chain`                | Output dir.                                         |
| `NUM_THREADS` / `NUM_TASKS`        | `keccakf`                                    | Threads (streams) / proofs to run.                  |
