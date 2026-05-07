# Synthetic benchmark suite

Replays a captured segment-shape profile as parametric synthetic AIRs
— same number of constraints, interactions, buses, message lengths,
and trace heights, but with all-zero traces that are
kernel-cost-faithful. Intended for fast champ-vs-candidate iteration
on `openvm-cuda-backend` changes without paying the original
workload's build / proving / RPC cost.

The runner (`synthetic_runner.rs`) and synthesizer (`synthetic_air.rs`)
are generic to any captured profile. The bundled
`reth-block-23992138-profile.jsonl` is the canonical baseline; swap in
any other profile JSONL captured by the upstream probe to benchmark a
different workload.

This directory is the **consumer-only** bundle: a captured profile +
the source files that read it and run the synthetic prove. The capture
side (the `SHADOW_BENCH_PROFILE_PATH` probe and any
workload-specific wrapper for invoking it) is not included here.

## What's in here

| File                                | What it is |
|-------------------------------------|------------|
| `reth-block-23992138-profile.jsonl` | Captured baseline profile: 209 segment proofs from a real reth `prove-stark` run on block 23992138 (schema v2, 6,382 per-AIR records). |
| `segment_profile.rs`                | Consumer-side deser types (`SegmentProfile`, `AirShapeRecord`) for each line of the JSONL. |
| `synthetic_air.rs`                  | `SyntheticAir` / `SyntheticShape` — kernel-cost-faithful replay AIR. Generic to any captured profile. |
| `synthetic_runner.rs`               | Runner binary: samples segments, builds synthetic AIRs, times `prove`, writes a scorecard JSON. Generic to any captured profile. |

## How the consumer pipeline works

```
reth-block-23992138-profile.jsonl                 (one SegmentProfile per line)
  │
  ▼  read_profile (synthetic_runner.rs)
Vec<SegmentProfile>                               (segment_profile.rs schema)
  │
  ▼  seeded sample of `ceil(total * sample_frac)` segments
  │
  ▼  shape_from_record  →  SyntheticShape         (per-AIR shape)
  │
  ▼  SyntheticAir::from_shape                     (synthetic_air.rs)
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

Why all-zero traces work: column 0 is a "kill column", every constraint
multiplies by it, every interaction count uses it. Constraints
trivially satisfied, multiplicities trivially zero, but the prover
still touches the same number of trace cells and constraint /
interaction terms — kernel timing is preserved. Proof validity is
*not* exercised end-to-end; rely on the cuda-backend test suite for
correctness.

## Wiring into the workspace

The .rs files in this directory are sources, not a buildable crate.
To run the benchmark, place them where the workspace can compile them:

| Source file            | Suggested location                                                                 |
|------------------------|------------------------------------------------------------------------------------|
| `segment_profile.rs`   | `crates/stark-backend/src/prover/segment_profile.rs` (and add `pub mod segment_profile;` to `crates/stark-backend/src/prover/mod.rs` and re-export the types) |
| `synthetic_air.rs`     | `crates/stark-sdk/src/bench/synthetic_air.rs` (and add `pub mod synthetic_air;` to `crates/stark-sdk/src/bench/mod.rs`) |
| `synthetic_runner.rs`  | `crates/cuda-backend/examples/synthetic_runner.rs`                                 |

The runner imports:
- `openvm_stark_backend::prover::segment_profile::{SegmentProfile, AirShapeRecord}`
- `openvm_stark_sdk::bench::synthetic_air::{SyntheticAir, SyntheticShape}`
- `openvm_stark_sdk::config::app_params_with_100_bits_security`
- `openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend, prelude::SC}`

If you place the files at the suggested paths above, those imports
resolve unmodified.

## Quick start

Requires CUDA + an NVIDIA GPU.

```bash
# Build (compiles cuda kernels — first build is slow, ~5 min)
cargo build --release -p openvm-cuda-backend --example synthetic_runner

# 16 GiB GPU memory pool.
export VPMM_PAGE_SIZE=$((4 << 20))
export VPMM_PAGES=$((16 << 8))
export RUST_LOG=warn

# The runner does not create its --out parent dir.
mkdir -p benchmarks/synthetic/scorecards

# Screening tier (~60 s wall, 3.9 s GPU prove).
target/release/examples/synthetic_runner \
  --profile benchmarks/synthetic/reth-block-23992138-profile.jsonl \
  --sample-frac 0.1 --seed 42 --max-log-height 22 \
  --out benchmarks/synthetic/scorecards/my-run.json
```

The runner accepts these flags (defaults in parentheses):

| Flag               | Default | Meaning                                                       |
|--------------------|---------|---------------------------------------------------------------|
| `--profile <path>` | —       | Required. Path to the profile JSONL.                          |
| `--sample-frac`    | `0.1`   | Fraction of segments to sample. `ceil(total * frac)` segments.|
| `--seed`           | `42`    | RNG seed for the segment sampler.                             |
| `--max-log-height` | `22`    | Per-AIR `log_height` clamp. Anything above is clipped.        |
| `--out <path>`     | —       | If unset, the scorecard JSON is printed to stdout instead.    |

For the same `(sample-frac, seed, max-log-height)` triple the result
is deterministic; champ-vs-candidate comparisons must use the same
triple in both runs.

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
   target/release/examples/synthetic_runner \
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

## Output format

Scorecard JSON written by the runner:

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
