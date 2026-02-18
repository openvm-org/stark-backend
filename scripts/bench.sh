#!/usr/bin/env bash
# CUDA microbenchmark runner.
#
# Usage:
#   ./scripts/bench.sh fractional-sumcheck
#   ./scripts/bench.sh fractional-sumcheck --tag baseline
#   ./scripts/bench.sh fractional-sumcheck --n 26
#   ./scripts/bench.sh fractional-sumcheck --nsys
#   ./scripts/bench.sh fractional-sumcheck --ncu-kernel 'compute_round_and_revert_kernel'
#   ./scripts/bench.sh fractional-sumcheck --nsys --compare baseline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Benchmark name (required first positional arg) ────────────────
if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "Usage: $0 <benchmark> [OPTIONS]" >&2
  echo "" >&2
  echo "Run 'cargo run -p openvm-cuda-backend --bin bench --release' to list benchmarks." >&2
  exit 1
fi

BENCH_NAME="$1"; shift
RESULTS_BASE="$REPO_ROOT/benchmarks/results/$BENCH_NAME"

# ── Defaults ────────────────────────────────────────────────────────
TAG=""
N=24
REPEATS=3
WARMUPS=1
USE_NSYS=false
USE_NCU=false
NCU_KERNEL=""
NCU_LAUNCH_SKIP=0
NCU_LAUNCH_COUNT=4
COMPARE_TAG=""
EXTRA_ENV=()

# ── Parse arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)          TAG="$2"; shift 2 ;;
    --n)            N="$2"; shift 2 ;;
    --repeats)      REPEATS="$2"; shift 2 ;;
    --warmups)      WARMUPS="$2"; shift 2 ;;
    --nsys)         USE_NSYS=true; shift ;;
    --ncu-kernel)   USE_NCU=true; NCU_KERNEL="$2"; shift 2 ;;
    --launch-skip)  NCU_LAUNCH_SKIP="$2"; shift 2 ;;
    --launch-count) NCU_LAUNCH_COUNT="$2"; shift 2 ;;
    --compare)      COMPARE_TAG="$2"; shift 2 ;;
    --env)          EXTRA_ENV+=("$2"); shift 2 ;;
    -h|--help)
      echo "Usage: $0 <benchmark> [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --tag <name>       Label appended to result directory"
      echo "  --n <size>         Problem size (default: 24)"
      echo "  --repeats <n>      Measured runs per size (default: 3)"
      echo "  --warmups <n>      Warmup runs per size (default: 1)"
      echo "  --nsys             Wrap with nsys profiling"
      echo "  --ncu-kernel <re>  Wrap with ncu kernel profiling (uses sudo)"
      echo "  --launch-skip <n>  ncu: skip first N kernel launches (default: 0)"
      echo "  --launch-count <n> ncu: profile N kernel launches (default: 4)"
      echo "  --compare <tag>    Auto-compare after finishing"
      echo "  --env KEY=VALUE    Pass env vars (repeatable)"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Apply extra env vars ────────────────────────────────────────────
for kv in "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}"; do
  export "${kv?}"
done

# ── Build ───────────────────────────────────────────────────────────
echo "Building bench binary..."
cargo build -p openvm-cuda-backend --bin bench --release
BENCH_BIN="$REPO_ROOT/target/release/bench"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "Error: could not find bench binary at $BENCH_BIN" >&2
  exit 1
fi
echo "Binary: $BENCH_BIN"

# ── Create result directory ─────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DIR_NAME="$TIMESTAMP"
[[ -n "$TAG" ]] && DIR_NAME="${TIMESTAMP}-${TAG}"
RUN_DIR="$RESULTS_BASE/$DIR_NAME"
mkdir -p "$RUN_DIR"

# ── Write metadata ──────────────────────────────────────────────────
{
  echo "timestamp: $TIMESTAMP"
  echo "benchmark: $BENCH_NAME"
  echo "tag: ${TAG:-<none>}"
  echo "git_commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_branch: $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty: $(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "n: $N"
  echo "repeats: $REPEATS"
  echo "warmups: $WARMUPS"
  echo "nsys: $USE_NSYS"
  echo "ncu: $USE_NCU"
  echo "command: $0 $BENCH_NAME $*"
  echo "binary: $BENCH_BIN"
  if [[ ${#EXTRA_ENV[@]} -gt 0 ]]; then
    echo "env:"
    for kv in "${EXTRA_ENV[@]}"; do
      echo "  $kv"
    done
  fi
} > "$RUN_DIR/meta.txt"

# ── Construct run command ───────────────────────────────────────────
export SWIRL_BENCH_N="$N"
export SWIRL_BENCH_REPEATS="$REPEATS"
export SWIRL_BENCH_WARMUPS="$WARMUPS"

RUN_CMD=("$BENCH_BIN" "$BENCH_NAME")

if $USE_NSYS; then
  # Larger VPMM page size reduces allocation noise during profiling.
  export VPMM_PAGE_SIZE=$((4 << 20))
  NSYS_OUT="$RUN_DIR/nsys"
  RUN_CMD=("nsys" "profile"
    "--output" "$NSYS_OUT"
    "--force-overwrite" "true"
    "--trace" "cuda"
    "--stats" "false"
    "--" "${RUN_CMD[@]}")
fi

if $USE_NCU; then
  export VPMM_PAGE_SIZE=$((4 << 20))
  NCU_OUT="$RUN_DIR/ncu"
  NCU_ARGS=("ncu"
    "--set" "full"
    "-f" "-o" "$NCU_OUT"
    "--target-processes" "all"
    "--launch-skip" "$NCU_LAUNCH_SKIP"
    "--launch-count" "$NCU_LAUNCH_COUNT")
  [[ -n "$NCU_KERNEL" ]] && NCU_ARGS+=("--kernel-name" "$NCU_KERNEL")
  RUN_CMD=("sudo" "env" "PATH=$PATH"
    "SWIRL_BENCH_N=$N"
    "SWIRL_BENCH_REPEATS=$REPEATS"
    "SWIRL_BENCH_WARMUPS=$WARMUPS"
    "VPMM_PAGE_SIZE=$VPMM_PAGE_SIZE"
    "${NCU_ARGS[@]}" "${RUN_CMD[@]}")
fi

# ── Run ─────────────────────────────────────────────────────────────
echo "Running benchmark '$BENCH_NAME' (n=$N, repeats=$REPEATS, warmups=$WARMUPS)..."
echo "Results: $RUN_DIR"

RAW_STDOUT="$(mktemp)"
trap 'rm -f "$RAW_STDOUT"' EXIT

if "${RUN_CMD[@]}" > "$RAW_STDOUT" 2>"$RUN_DIR/stderr.log"; then
  echo "Benchmark completed successfully."
else
  EXIT_CODE=$?
  echo "Warning: benchmark exited with code $EXIT_CODE" >&2
  echo "stderr saved to $RUN_DIR/stderr.log" >&2
fi

# ── Extract CSV ─────────────────────────────────────────────────────
# Filter: keep the CSV header line and numeric data lines
{
  echo "run_idx,is_warmup,elapsed_ms"
  grep -E '^[0-9]' "$RAW_STDOUT" || true
} > "$RUN_DIR/bench.csv"

DATA_LINES="$(grep -cE '^[0-9]' "$RUN_DIR/bench.csv" || echo 0)"
echo "Collected $DATA_LINES data rows."

# ── Generate summary ────────────────────────────────────────────────
awk -F',' '
  NR == 1 { next }
  $2 != "0" { next }  # skip warmups
  {
    ms = $3 + 0
    count++
    sum += ms
    vals[count] = ms
  }
  function median(   m, i, j, t) {
    m = count
    for (i = 1; i <= m; i++) a[i] = vals[i]
    for (i = 1; i <= m; i++)
      for (j = i + 1; j <= m; j++)
        if (a[i] > a[j]) { t = a[i]; a[i] = a[j]; a[j] = t }
    if (m % 2 == 1) return a[(m + 1) / 2]
    return (a[m / 2] + a[m / 2 + 1]) / 2
  }
  END {
    if (count > 0) {
      printf "n=%d  median=%.2f ms  mean=%.2f ms  runs=%d\n", '"$N"', median(), sum / count, count
    }
  }
' "$RUN_DIR/bench.csv" > "$RUN_DIR/summary.txt"

echo ""
echo "=== Summary ==="
cat "$RUN_DIR/summary.txt"


# ── nsys analysis ───────────────────────────────────────────────────
if $USE_NSYS && [[ -f "$RUN_DIR/nsys.nsys-rep" ]]; then
  echo ""
  echo "Running nsys analysis..."
  bash "$SCRIPT_DIR/bench-nsys-analyze.sh" "$RUN_DIR/nsys.nsys-rep" "$RUN_DIR"
fi

# ── ncu text export ────────────────────────────────────────────────
if $USE_NCU && [[ -f "$RUN_DIR/ncu.ncu-rep" ]]; then
  echo ""
  echo "Exporting ncu text report..."
  ncu -i "$RUN_DIR/ncu.ncu-rep" > "$RUN_DIR/ncu.txt"
  echo "ncu report: $RUN_DIR/ncu.txt"
fi

# ── Update latest symlink ──────────────────────────────────────────
ln -sfn "$DIR_NAME" "$RESULTS_BASE/latest"

# ── Compare ─────────────────────────────────────────────────────────
if [[ -n "$COMPARE_TAG" ]]; then
  echo ""
  bash "$SCRIPT_DIR/bench-compare.sh" "$BENCH_NAME" "$COMPARE_TAG" "$DIR_NAME"
fi

echo ""
echo "Done. Results in: $RUN_DIR"
