#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PATH="${1:-"$ROOT_DIR/crates/metal-backend/src"}"

if [[ ! -e "$TARGET_PATH" ]]; then
  echo "error: target path not found: $TARGET_PATH" >&2
  exit 2
fi

FORBIDDEN_SYMBOLS=(
  "\\bCpuBackend\\b"
  "\\bCpuDevice\\b"
  "\\bStackedReductionCpu\\b"
  "\\bprove_zerocheck_and_logup\\b"
  "\\bprove_whir_opening\\b"
  "stacked_pcs::stacked_commit\\b"
  "\\bmpk_to_cpu\\b"
  "\\bctx_to_cpu\\b"
  "\\bmake_cpu_device\\b"
)

FORBIDDEN_PATTERN="$(IFS='|'; echo "${FORBIDDEN_SYMBOLS[*]}")"
if MATCHES="$(
  rg -n \
    --glob '!**/tests.rs' \
    --glob '!**/test_*.rs' \
    --glob '!**/mod_tests.rs' \
    "${FORBIDDEN_PATTERN}" \
    "$TARGET_PATH" \
    | rg -v '^[^:]+:[0-9]+:\\s*(//|//!|/\\*|\\*)'
)"; then
  echo "error: CPU fallback references found in Metal backend sources: $TARGET_PATH" >&2
  echo "$MATCHES" >&2
  echo >&2
  echo "forbidden symbols:" >&2
  echo "  - CpuBackend" >&2
  echo "  - CpuDevice" >&2
  echo "  - StackedReductionCpu" >&2
  echo "  - prove_zerocheck_and_logup" >&2
  echo "  - prove_whir_opening" >&2
  echo "  - stacked_pcs::stacked_commit" >&2
  echo "  - mpk_to_cpu" >&2
  echo "  - ctx_to_cpu" >&2
  echo "  - make_cpu_device" >&2
  exit 1
fi

echo "ok: no CPU fallback references found in $TARGET_PATH"
