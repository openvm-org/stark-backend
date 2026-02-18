#!/usr/bin/env bash
# Compare two benchmark runs.
#
# Usage:
#   ./scripts/bench-compare.sh <benchmark> <baseline> <candidate>
#
# Names are resolved by searching benchmarks/results/<benchmark>/ for:
#   1. Symlinks (e.g. "latest")
#   2. Exact directory name match
#   3. Substring/suffix match (e.g. "baseline" matches "20260212-143052-baseline")
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <benchmark> <baseline> <candidate>" >&2
  echo "" >&2
  echo "Names are resolved by searching benchmarks/results/<benchmark>/ for:" >&2
  echo "  - Symlinks (e.g. 'latest')" >&2
  echo "  - Exact directory name" >&2
  echo "  - Substring match on tag suffix" >&2
  exit 1
fi

BENCH_NAME="$1"
RESULTS_BASE="$REPO_ROOT/benchmarks/results/$BENCH_NAME"

# ── Resolve a run name to a directory ───────────────────────────────
resolve_run() {
  local name="$1"

  # 1. Symlink or exact match
  if [[ -d "$RESULTS_BASE/$name" ]]; then
    echo "$RESULTS_BASE/$name"
    return 0
  fi

  # 2. Substring match (search for directories containing the name)
  local matches=()
  for d in "$RESULTS_BASE"/*/; do
    [[ -d "$d" ]] || continue
    local base
    base="$(basename "$d")"
    if [[ "$base" == *"$name"* ]]; then
      matches+=("$d")
    fi
  done

  if [[ ${#matches[@]} -eq 1 ]]; then
    echo "${matches[0]%/}"
    return 0
  elif [[ ${#matches[@]} -gt 1 ]]; then
    echo "Error: ambiguous name '$name' matches multiple directories:" >&2
    for m in "${matches[@]}"; do
      echo "  $(basename "${m%/}")" >&2
    done
    return 1
  fi

  echo "Error: no run found matching '$name' in $RESULTS_BASE/" >&2
  return 1
}

BASELINE_DIR="$(resolve_run "$2")"
CANDIDATE_DIR="$(resolve_run "$3")"

BASELINE_CSV="$BASELINE_DIR/bench.csv"
CANDIDATE_CSV="$CANDIDATE_DIR/bench.csv"

if [[ ! -f "$BASELINE_CSV" ]]; then
  echo "Error: baseline bench.csv not found: $BASELINE_CSV" >&2
  exit 1
fi
if [[ ! -f "$CANDIDATE_CSV" ]]; then
  echo "Error: candidate bench.csv not found: $CANDIDATE_CSV" >&2
  exit 1
fi

# ── Read metadata ───────────────────────────────────────────────────
get_meta() {
  local dir="$1" key="$2"
  if [[ -f "$dir/meta.txt" ]]; then
    grep "^${key}:" "$dir/meta.txt" 2>/dev/null | sed "s/^${key}: *//" || echo "?"
  else
    echo "?"
  fi
}

BASELINE_NAME="$(basename "$BASELINE_DIR")"
CANDIDATE_NAME="$(basename "$CANDIDATE_DIR")"
BASELINE_COMMIT="$(get_meta "$BASELINE_DIR" "git_commit")"
CANDIDATE_COMMIT="$(get_meta "$CANDIDATE_DIR" "git_commit")"

# ── Wall-time comparison ────────────────────────────────────────────
BASELINE_N="$(get_meta "$BASELINE_DIR" "n")"
CANDIDATE_N="$(get_meta "$CANDIDATE_DIR" "n")"

echo "=== Benchmark Comparison ($BENCH_NAME) ==="
echo "Baseline:  $BASELINE_NAME (n=$BASELINE_N, commit $BASELINE_COMMIT)"
echo "Candidate: $CANDIDATE_NAME (n=$CANDIDATE_N, commit $CANDIDATE_COMMIT)"
echo ""

# Compute median from a bench.csv (run_idx,is_warmup,elapsed_ms)
compute_median() {
  local file="$1"
  awk -F',' '
    NR == 1 { next }
    $2 != "0" { next }  # skip warmups
    {
      ms = $3 + 0
      count++; vals[count] = ms
    }
    END {
      if (count == 0) { print 0; exit }
      for (i = 1; i <= count; i++)
        for (j = i + 1; j <= count; j++)
          if (vals[i] > vals[j]) { t = vals[i]; vals[i] = vals[j]; vals[j] = t }
      if (count % 2 == 1) printf "%.4f\n", vals[(count + 1) / 2]
      else printf "%.4f\n", (vals[count / 2] + vals[count / 2 + 1]) / 2
    }
  ' "$file"
}

BASE_MS="$(compute_median "$BASELINE_CSV")"
CAND_MS="$(compute_median "$CANDIDATE_CSV")"

printf "%14s %14s %10s %10s\n" "baseline_ms" "candidate_ms" "delta" "speedup"
printf "%14s %14s %10s %10s\n" "-----------" "------------" "-----" "-------"

awk -v b="$BASE_MS" -v c="$CAND_MS" 'BEGIN {
  if (b > 0) {
    delta_pct = ((c - b) / b) * 100.0
    speedup = b / c
    printf "%14.2f %14.2f %+9.1f%% %9.2fx\n", b, c, delta_pct, speedup
  }
}'

# ── Kernel comparison (nsys) ────────────────────────────────────────
BASELINE_KERN="$BASELINE_DIR/nsys-kernels.csv"
CANDIDATE_KERN="$CANDIDATE_DIR/nsys-kernels.csv"

if [[ -f "$BASELINE_KERN" && -f "$CANDIDATE_KERN" ]]; then
  echo ""
  echo "=== Kernel Breakdown (from nsys) ==="

  # Build combined kernel table. Strip common namespace prefixes for readability.
  awk -F',' '
    BEGIN { OFS="," }
    function short(name) {
      # Strip common namespace prefixes
      sub(/^fractional_sumcheck_gkr::/, "", name)
      sub(/^sumcheck::/, "", name)
      return name
    }
    # File 1 = baseline, File 2 = candidate
    FILENAME == ARGV[1] {
      if (NR == 1 || $1 == "name") next
      k = short($1)
      base_ns[k] += $3 + 0
      base_total += $3 + 0
      seen[k] = 1
      next
    }
    {
      if (FNR == 1 || $1 == "name") next
      k = short($1)
      cand_ns[k] += $3 + 0
      cand_total += $3 + 0
      seen[k] = 1
    }
    END {
      # Collect and sort by max(baseline, candidate) total (desc)
      nk = 0
      for (k in seen) {
        nk++
        knames[nk] = k
        b = (k in base_ns) ? base_ns[k] : 0
        c = (k in cand_ns) ? cand_ns[k] : 0
        ksort[nk] = (b > c) ? b : c
      }
      for (i = 1; i <= nk; i++)
        for (j = i + 1; j <= nk; j++)
          if (ksort[i] < ksort[j]) {
            t = knames[i]; knames[i] = knames[j]; knames[j] = t
            t = ksort[i]; ksort[i] = ksort[j]; ksort[j] = t
          }

      w = 45
      printf "%-" w "s  %12s  %12s  %10s\n", "Kernel", "baseline_ms", "candidate_ms", "delta"
      printf "%-" w "s  %12s  %12s  %10s\n", "------", "-----------", "------------", "-----"

      for (i = 1; i <= nk; i++) {
        k = knames[i]
        b_ms = (k in base_ns) ? base_ns[k] / 1e6 : 0
        c_ms = (k in cand_ns) ? cand_ns[k] / 1e6 : 0

        if (b_ms == 0 && c_ms > 0) {
          printf "%-" w "s  %12s  %12.2f  %10s\n", k, "---", c_ms, "(new)"
        } else if (b_ms > 0 && c_ms == 0) {
          printf "%-" w "s  %12.2f  %12s  %10s\n", k, b_ms, "---", "(gone)"
        } else if (b_ms > 0) {
          delta_pct = ((c_ms - b_ms) / b_ms) * 100.0
          marker = ""
          if (delta_pct < -10) marker = "  <--"
          if (delta_pct > 10) marker = "  !!!"
          printf "%-" w "s  %12.2f  %12.2f  %+9.1f%%%s\n", k, b_ms, c_ms, delta_pct, marker
        }
      }

      b_total_ms = base_total / 1e6
      c_total_ms = cand_total / 1e6
      if (b_total_ms > 0) {
        total_delta = ((c_total_ms - b_total_ms) / b_total_ms) * 100.0
        printf "\nGPU Total: %.2f -> %.2f ms (%+.1f%%)\n", b_total_ms, c_total_ms, total_delta
      }
    }
  ' "$BASELINE_KERN" "$CANDIDATE_KERN"
elif [[ -f "$BASELINE_KERN" ]]; then
  echo ""
  echo "(Candidate has no nsys data — skipping kernel comparison)"
elif [[ -f "$CANDIDATE_KERN" ]]; then
  echo ""
  echo "(Baseline has no nsys data — skipping kernel comparison)"
fi
