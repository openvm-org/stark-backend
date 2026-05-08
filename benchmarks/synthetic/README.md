# Synthetic benchmark suite

GPU/CPU benchmarks built on synthetic AIRs with all-zero traces:
kernel-cost-faithful for the prover, but trivially valid (column 0 is
a "kill column" — every constraint multiplies by it, every interaction
count uses it). Proof validity is not exercised end-to-end; rely on
the cuda-backend test suite for correctness.

Two binaries:

| Binary             | Use case                                         | Inputs                          |
|--------------------|--------------------------------------------------|---------------------------------|
| `synthetic_runner` | Replay a captured workload's segment shapes for champ-vs-candidate iteration on `openvm-cuda-backend` changes. | A profile JSONL from the `SHADOW_BENCH_PROFILE_PATH` probe. |
| `uniform_runner`   | Sweep cost dimensions with N identical AIRs.     | CLI flags (num AIRs, cols/AIR, constraints/col, etc.). |

`synthetic_runner` is GPU-only (`required-features = ["cuda"]`).
`uniform_runner` runs CPU by default; build with `--features cuda` for GPU.

## What's in here

| File                                | What it is |
|-------------------------------------|------------|
| `reth-block-23992138-profile.jsonl` | Captured baseline profile: 209 segment proofs from a real reth `prove-stark` run on block 23992138 (schema v2, 6,382 per-AIR records). |
| `src/lib.rs`                        | Re-exports the synthesizer + schema modules.                            |
| `src/synthetic_air.rs`              | `SyntheticAir` / `SyntheticShape` — kernel-cost-faithful replay AIR.     |
| `src/segment_profile.rs`            | Deserialization types (`SegmentProfile`, `AirShapeRecord`) for the JSONL. |
| `src/bin/synthetic_runner.rs`       | Profile-replay runner.                                                  |
| `src/bin/uniform_runner.rs`         | Uniform-shape runner.                                                   |

## `synthetic_runner` — profile replay

Reads a captured profile JSONL, samples a fraction of segments,
replays each as a `Coordinator::prove` call on the GPU, and writes a
scorecard JSON.

```
reth-block-23992138-profile.jsonl                 (one SegmentProfile per line)
  │
  ▼  read_profile
Vec<SegmentProfile>
  │
  ▼  seeded sample of `ceil(total * sample_frac)` segments
  │
  ▼  shape_from_record  →  SyntheticShape         (per-AIR shape)
  │
  ▼  SyntheticAir::from_shape
Vec<AirRef<SC>>                                   (one AIR per record)
  │
  ▼  generate_trace (all zeros) + transport to device
  │
  ▼  engine.keygen + engine.prove
Per-segment prove_ms / keygen_ms                  (timed)
  │
  ▼  aggregate
Scorecard JSON                                    (figure of merit: total_prove_ms)
```

### Quick start

Requires CUDA + an NVIDIA GPU.

```bash
# Build (compiles cuda kernels — first build is slow, ~5 min)
cargo build --release -p openvm-benchmark-synthetic --features cuda --bin synthetic_runner

# 16 GiB GPU memory pool.
export VPMM_PAGE_SIZE=$((4 << 20))
export VPMM_PAGES=$((16 << 8))
export RUST_LOG=warn

# The runner does not create its --out parent dir.
mkdir -p benchmarks/synthetic/scorecards

# Screening tier (~60 s wall, 3.9 s GPU prove).
target/release/synthetic_runner \
  --profile benchmarks/synthetic/reth-block-23992138-profile.jsonl \
  --sample-frac 0.1 --seed 42 --max-log-height 22 \
  --out benchmarks/synthetic/scorecards/my-run.json
```

For the same `(sample-frac, seed, max-log-height)` triple the result
is deterministic; champ-vs-candidate comparisons must use the same
triple in both runs.

CLI flags (defaults in parentheses):

| Flag               | Default | Meaning                                                       |
|--------------------|---------|---------------------------------------------------------------|
| `--profile <path>` | —       | Required. Path to the profile JSONL.                          |
| `--sample-frac`    | `0.1`   | Fraction of segments to sample. `ceil(total * frac)` segments.|
| `--seed`           | `42`    | RNG seed for the segment sampler.                             |
| `--max-log-height` | `22`    | Per-AIR `log_height` clamp. Anything above is clipped.        |
| `--out <path>`     | —       | If unset, the scorecard JSON is printed to stdout instead.    |

### Tiered modes

| `--sample-frac` | Use case            | Segments | GPU prove (reference) | Wall   |
|-----------------|---------------------|----------|-----------------------|--------|
| `0.1`           | screening           | 21/209   | ~3.9 s                | ~60 s  |
| `0.5`           | fail-fast           | 105/209  | ~21.6 s               | ~333 s |
| `1.0`           | rigorous full sweep | 209/209  | ~43.6 s               | ~666 s |

Wall ≫ prove because per-segment H2D transport of the all-zero traces
(up to 150 M cells) dominates outside the timed `prove()` call. Report
`total_prove_ms` for comparison; that's the apples-to-apples metric
across champ vs. candidate.

### Champ-vs-candidate workflow

1. Check out the champion, run the screening tier with a fixed seed:
   ```bash
   target/release/synthetic_runner \
     --profile benchmarks/synthetic/reth-block-23992138-profile.jsonl \
     --sample-frac 0.1 --seed 42 --max-log-height 22 \
     --out benchmarks/synthetic/scorecards/champ.json
   ```
2. Check out the candidate, repeat with the same seed → `cand.json`.
3. Compute `(cand.total_prove_ms - champ.total_prove_ms) /
   champ.total_prove_ms`. Reference jitter floor is **0.40 % CoV** at
   the screening tier (4 same-config runs); a real wall delta > ~0.5 %
   is detectable above noise.
4. If the screening result is promising, escalate to `--sample-frac
   0.5` and finally `1.0` before final acceptance.

### Repeatability + jitter floor

Run the same `(frac, seed)` triple multiple times to characterize
runtime noise on your machine. Reference numbers for `(0.1, 42)`:

```
mean=3895 ms  stddev=16 ms  range=33 ms  CoV=0.40 %
```

If your CoV is much higher, possible causes: GPU shared with another
workload, thermal throttling, kernel-launch contention from background
processes.

### Scorecard format

```jsonc
{
  "profile_path": "...",
  "sample_frac": 0.1,
  "seed": 42,
  "max_log_height": 22,
  "total_segments": 209,
  "sampled_segments": 21,
  "skipped_segments": 0,
  "total_keygen_ms": 510,
  "total_prove_ms": 3891,
  "results": [
    {
      "segment_idx": 5,
      "num_airs": 30,
      "total_main_cells": 38123456,
      "keygen_ms": 24,
      "prove_ms": 178,
      "clamped": false
    }
    // ...
  ]
}
```

`clamped: true` means at least one AIR in that segment had
`log_height > max_log_height` and was clipped down. `total_prove_ms`
is the figure of merit for champ-vs-candidate comparison.
`skipped_segments` should be `0`; non-zero means a prove call returned
an error and that segment's timing was excluded.

## `uniform_runner` — uniform-shape sweep

Builds N identical synthetic AIRs sized via CLI flags and runs one
proof. Useful for understanding how cost moves with a single
parameter (cols per AIR, constraints per column, etc.).

### Quick start

```bash
# CPU
cargo run -p openvm-benchmark-synthetic --release --bin uniform_runner -- \
  --num-airs 4 --cols-per-air 100 --constraints-per-col 2 --log-rows-per-air 20

# GPU
cargo run -p openvm-benchmark-synthetic --release --features cuda --bin uniform_runner -- \
  --num-airs 4 --cols-per-air 100 --constraints-per-col 2 --log-rows-per-air 20

# Optional: write a metrics.json on completion
METRICS_OUTPUT=metrics.json cargo run -p openvm-benchmark-synthetic --release \
  --bin uniform_runner -- --num-airs 4 --cols-per-air 100 --log-rows-per-air 20
```

CLI flags (defaults in parentheses):

| Flag                     | Default | Meaning                                                        |
|--------------------------|---------|----------------------------------------------------------------|
| `--num-airs`             | `1`     | Number of identical AIRs to keygen + prove.                    |
| `--cols-per-air`         | `20`    | Common-main width per AIR.                                     |
| `--constraints-per-col`  | `1.0`   | Boolean constraints per column (fractional OK).                |
| `--interactions-per-col` | `0.25`  | Send/receive bus pairs per column (fractional OK).             |
| `--log-rows-per-air`     | `18`    | log2 of rows per AIR.                                          |
| `--log-stacked-height`   | `24`    | log2 of stacked height for the PCS (matches default app config). |

Verifies the proof at the end — surfaces correctness regressions that
the synthetic_runner's profile-replay path skips.
