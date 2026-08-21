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
| `CC_CONFIG`                    | Path to compiler-config TOML (`cc_default_config.toml` = list_v1,   |
|                                | `cc_list_v2_config.toml` = list_v2).                                 |
| `NSYS_ENABLED=1`               | Enable `cudaProfilerStart/Stop` + NVTX ranges in every bench.        |
| `CC_GRAPH_DUMP_PATH=<file>`    | Cache the compiled `GraphExe` between runs (skips fusion/schedule).  |
| `CC_CY_DUMP_PATH=<dir>`        | Dump post-fuse+dce graph + cytoscape JSON + timings JSON.            |
| `FRAC_BENCH_DUMP_EXE=<file>`   | Dump `exe.print()` (used by `bench_fractional_sumcheck_eager_vs_ir`).|
| `FRAC_LOG_N=<csv>`             | Leaf-count log2 list (default varies by bench).                      |
| `FRAC_ROUND=<n>` / `FRAC_ROUNDS=<csv>` | Which outer round(s) the pipelined benches build for.        |
| `CC_STREAMS_SWEEP=<csv>`       | Override list_v2 stream sweep in `all_schedulers_nsys`.              |
| `SWIRL_CUDA_GKR_PIPELINE_SPLITS=<n>` | Pipeline splits knob for the pipelined driver graphs.          |

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
| `bench_fractional_sumcheck_eager_vs_ir_overlap`| `fractional_ir.rs`                  | Same graph as above but sweeps overlap-based execution drivers side-by-side.                        |
| `bench_fractional_sumcheck_eager_vs_irv2`      | `fractional_sumcheck_gpu_irv2.rs`   | irv2 (fusion-v2 optimised) full sumcheck vs eager, with `exe.run()` and `launch_graph()` variants.  |
| `bench_pipelined_ir_vs_eager`                  | `fractional_ir_pipelined.rs`        | Single outer round `FRAC_ROUND` of the α-tiled pipelined driver vs eager.                           |
| `bench_pipelined_full_sumcheck_vs_eager`       | `fractional_ir_pipelined.rs`        | Full pipelined sumcheck vs eager at every `FRAC_LOG_N`.                                             |
| `bench_pipelined_ir_sweep`                     | `fractional_ir_pipelined.rs`        | Per-`j` single-round pipelined graphs across `FRAC_ROUNDS` (default `4,10,16,20,24`).               |
| `bench_pipelined_full_sumcheck_sweep`          | `fractional_ir_pipelined.rs`        | Eager + IR + pipelined drivers side-by-side at every `FRAC_LOG_N` (default `16,20,24`).             |
| `bench_fractional_sumcheck_all_schedulers_nsys`| `fractional_sumcheck_gpu_irv2.rs`   | Sweep scheduler variants (list_v1 / list_v2 knobs) on the irv2 graph, single nsys window.           |

All bench functions live in the `tests` submodule of their file — the nextest filter is
`test(=logup_zerocheck::<file>::tests::<bench>)`.

### Command templates

Substitute `$CFG=cc_default_config.toml` (list_v1) or `$CFG=cc_list_v2_config.toml` (list_v2)
in every command below. Default profiling target is GPU 5.

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

**`bench_fractional_sumcheck_eager_vs_ir_overlap`** — same file, driver-overlap sweep:

```bash
CUDA_VISIBLE_DEVICES=5 NSYS_ENABLED=1 \
  CC_CONFIG=$(pwd)/crates/compiler/$CFG FRAC_LOG_N=24 \
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \
    --trace=cuda,nvtx --wait=all --force-overwrite=true -o frac_ir_overlap_n24 \
  cargo nextest run -p openvm-cuda-backend --features graph-ir --release \
    --run-ignored all --no-capture \
    -E 'test(=logup_zerocheck::fractional_ir::tests::bench_fractional_sumcheck_eager_vs_ir_overlap)'
```

**`bench_fractional_sumcheck_eager_vs_irv2`** — fusion-v2 optimised full sumcheck:

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


