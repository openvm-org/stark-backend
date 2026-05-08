# Hardware-mapped CUDA profiler — usage guide

This directory is a self-contained pipeline that captures per-CTA placement
and timing for **every kernel a CUDA program launches** and turns it into a
PNG overview. Built around [NVBit](https://github.com/NVlabs/NVBit) for
zero-source-change instrumentation, libzstd for log compression, and a
pure-C aggregator + Python+matplotlib renderer for offline reduction.

```
  ┌────────────────────────┐  zstd-framed     ┌─────────────────┐  binary  ┌──────────────────────┐  PNG
  │  nvbit-tool.so         │ ─────────────▶   │  aggregate      │ ───────▶ │  render_overview.py  │ ────▶
  │  (LD_PRELOAD'd into    │  shadow_profile  │  (C, libzstd)   │  agg.bin │  (matplotlib)        │  trace_overview.png
  │   any CUDA binary)     │  .bin            │                 │          │                      │
  └────────────────────────┘                  └─────────────────┘          └──────────────────────┘
```

There are no source-level probes anywhere in this repo — instrumentation
happens at SASS load time, on whatever CUDA binary you point at it. With
`SHADOW_PROFILER` unset, `LD_PRELOAD`'ing the tool is a no-op (drain
thread isn't created, no log file is opened).

## Quick start (TL;DR)

```bash
# 1. one-time setup: download NVBit, install zstd headers
curl -L -O https://github.com/NVlabs/NVBit/releases/download/v1.8/nvbit-Linux-x86_64-1.8.tar.bz2
tar -xjf nvbit-Linux-x86_64-1.8.tar.bz2
sudo apt install libzstd-dev    # or `dnf install libzstd-devel`

# 2. build the tool + aggregator (pin ARCH to your GPU)
cd crates/cuda-profiler/nvbit-tool
make NVBIT_PATH=$PWD/../../../../nvbit_release_x86_64/core ARCH=sm_90

# 3. capture (records 1/8 of CTAs from any kernel with grid >= 8, all CTAs
#    from smaller kernels; output is zstd-compressed)
SHADOW_PROFILER=1 \
SHADOW_PROFILER_OUT=$PWD/shadow_profile.bin \
SHADOW_PROFILER_RING_BYTES=$((1<<30)) \
SHADOW_PROFILER_SAMPLE_N=8 \
LD_PRELOAD=$PWD/nvbit-tool.so \
  ./your-cuda-binary --your-args

# 4. fold the log to a small aggregate
./aggregate shadow_profile.bin agg.bin

# 5. render the PNG
pip install --user matplotlib numpy   # if not already installed
python3 render_overview.py agg.bin trace_overview.png
```

That's the whole flow. The PNG is two panels: the top is a per-bucket-
normalized stacked area showing CTA-residency composition by top-12
kernel families; the bottom is a per-SM heatmap (rows = SM, cols = time
bucket, color = sqrt of CTA count landing in that cell).

## Components

| File | What it is |
|--|--|
| `tool.cu` | Host-side NVBit tool: env-var parsing, ring + drain thread, zstd-framed log writer, SASS instrumentation hook |
| `inject_funcs.cu` | Device-side BEGIN/END probes (lane-0-only, atomicExch dedup, decimation gate) |
| `Makefile` | Builds `nvbit-tool.so` and `aggregate` |
| `aggregate.c` | Pure-C two-pass aggregator: streams the (raw or zstd) log, folds into a per-(SM, time-bucket) grid + per-(kernel, bucket) cumulative residency time. Auto-detects zstd magic. |
| `render_overview.py` | Reads `agg.bin`, emits the PNG via matplotlib |

## Step-by-step

### 1. Build

```bash
make NVBIT_PATH=/path/to/nvbit_release_x86_64/core ARCH=sm_90
```

`ARCH=all` (the default) compiles for every real arch the local nvcc
knows; pinning to `sm_90` (H100) or `sm_120` (consumer Blackwell) is
faster and produces slightly smaller binaries. NVBit 1.8 covers CC 7.0
through 12.1.

Outputs in this directory:
- `nvbit-tool.so` — the `LD_PRELOAD`-able shared lib (~3 MB)
- `aggregate` — a small (~20 KB) C binary

### 2. Capture

`LD_PRELOAD` the tool into any CUDA binary. Configuration is via
environment variables:

| | default | what it does |
|--|--|--|
| `SHADOW_PROFILER` | unset | master switch — any of `1`/`true`/`on` activates. With it unset, `LD_PRELOAD` is a no-op. |
| `SHADOW_PROFILER_OUT` | `shadow_profile.bin` | output log path (zstd-framed) |
| `SHADOW_PROFILER_RING_BYTES` | 64 MB | host-pinned ring capacity (rounded up to a power of 2). Bump to 1 GB for long runs that can produce records faster than the drain can consume. |
| `SHADOW_PROFILER_SCRATCH_SLOTS` | 8 M (64 MB device memory) | size of the global hash table that hands `t_start` from BEGIN probe to END probe. Increase if you see many CTAs with `t_start ≈ t_end` (clamped to 1 ns) — that's the symptom of hash collisions across concurrent kernels. |
| `SHADOW_PROFILER_SAMPLE_N` | 1 | record 1 of every N CTAs from kernels with grid ≥ N; record every CTA from smaller kernels. Power-of-two; rounded up if not. **N=8 + zstd shrinks a 25-min prove-app log from ~58 GB to ~2-3 GB with the PNG visually unchanged.** |
| `SHADOW_PROFILER_VERBOSE` | unset | set to `1` to log every kernel as it gets instrumented |
| `NOBANNER` | unset | NVBit's own — set to `1` to suppress the NVBit banner at start |

The on-disk log is **zstd-framed by default**: file starts with the zstd
magic `28 b5 2f fd`, the SHDWPROF header lives inside the compressed
stream. The C `aggregate` auto-detects format by peeking the first 4
bytes — no flag needed. (The older Rust `cuda-profiler-assemble` binary
in the parent crate currently only reads raw logs; pipe through `zstd
-d` if you need it.)

### 3. Aggregate

```bash
./aggregate shadow_profile.bin agg.bin
```

Two passes over the log:
1. discover `(t_min, t_max)` and the kernel-id → name table
2. fold every CTA into a `(smid, time-bucket)` count grid plus a
   `(kernel, time-bucket)` cumulative-CTA-residency-time grid

Constants are baked in: `N_SMS = 132` (fits all current Hopper/Blackwell
GPUs), `N_BUCKETS = 1200` time buckets across the whole run. The output
`agg.bin` is on the order of MB regardless of how big the input log was
— even 1.9 billion CTAs across 144 distinct kernel families folded into
2 MB on this hardware.

Memory footprint is fixed: ~100 MB for the aggregator regardless of log
size, because records are streamed and never stored.

### 4. Render

```bash
python3 render_overview.py agg.bin trace_overview.png
```

Requires `matplotlib` and `numpy`. The script also calls `c++filt` if
present to demangle kernel names; if it's missing the labels just stay
mangled.

The PNG has two panels:

- **Top — CTA-residency share, normalized per bucket.** A stacked area
  where each bucket sums to 1 and the colors say which kernel families
  contributed how much CTA-residency in that bucket. Reads as
  composition over time, not amount.

- **Bottom — Per-SM CTA placement heatmap.** Rows = SM index (only the
  SMs that actually saw any CTAs are shown), cols = time bucket, color
  = `sqrt(CTAs landing per (SM, bucket))`. Every kernel contributes;
  load-balance and idle/stall periods are immediately visible.

We deliberately do **not** render a "top kernels by GPU time" bar
chart. The number we'd plot — `Σ CTAs (t_end − t_start)` — is *sum of
per-CTA wall-clocks*, which heavily over-counts kernels with many short
concurrent CTAs and is meaningless as a "GPU wall-clock time" answer.
For that use nsys or CUPTI; this profiler is for the per-SM placement
view that nsys/CUPTI cannot give.

## What the records look like

Each per-CTA record is 32 bytes: `kernel_id u32`, `smid u32`,
`block_linear u32`, `seq_tag u32`, `t_start u64`, `t_end u64`. The full
on-disk format is described in [`../src/record.rs`](../src/record.rs)
along with the assorted other tags (ProcessStart, KernelName, Drop).

Timestamps are PTX `%globaltimer` ns since boot — same source CUPTI's
Activity API uses, so wallclocks line up across the two if you ever
combine them.

## Caveats

- **No CUPTI side-data.** This tool only emits per-CTA records (plus
  `ProcessStart` and `KernelName` headers). Kernel-launch / memcpy /
  NVTX activity records — which CUPTI provides — are not captured. If
  you need those, run nsys side-by-side or extend `tool.cu` to also
  call `cuptiActivityRegisterCallbacks`.

- **Hash collisions in the scratch table.** BEGIN and END probes hand
  `t_start` between each other through a 64 MB hash table keyed by
  `(smid, block_linear)`. Two CTAs that land on the same hash slot
  cause one to lose its `t_start` reference, and the END probe clamps
  the affected record to a 1-ns span. With the default 8 M slots this
  is rare (<1 in 10⁴) for typical workloads; bump
  `SHADOW_PROFILER_SCRATCH_SLOTS` if your trace shows many degenerate
  spans.

- **No atexit / TLS race.** The drain thread `fflush`es every sweep so
  on-disk content stays current; `nvbit_at_term` does the final
  `ZSTD_e_end` + close. We deliberately don't use a libc `atexit`
  hook because Rust's TLS is being torn down by the time atexit
  fires — `parking_lot`/`tracing`/`JoinHandle::join` all touch TLS
  and panic.

- **Compute capability.** Anything NVBit's release supports. NVBit 1.8
  covers CC 7.0 through 12.1.

## Sample numbers from a real run

H100 PCIe (114 SMs), Reth `prove-app` block 23992138, 25 minutes of GPU
work. With `SHADOW_PROFILER_SAMPLE_N=8`:

| | |
|--|--|
| Distinct kernel families instrumented | 151 |
| CTAs captured (after 1/8 sampling) | ~180 M |
| `shadow_profile.bin` (zstd-framed) | ~2 GB |
| `agg.bin` | 2 MB |
| `trace_overview.png` | ~900 KB |

Without sampling and without zstd, the same run produces a 58 GB log.
The PNG looks the same.
