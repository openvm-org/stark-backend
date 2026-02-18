#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_FILE="${1:-"$ROOT_DIR/crates/metal-backend/src/metal_backend.rs"}"

if [[ ! -f "$TARGET_FILE" ]]; then
  echo "error: target file not found: $TARGET_FILE" >&2
  exit 2
fi

FORBIDDEN_PATTERN=$(
  cat <<'EOF'
stacked_pcs::stacked_commit|prove_zerocheck_and_logup|StackedReductionCpu|prove_whir_opening|crate::convert::mpk_to_cpu|crate::convert::ctx_to_cpu|crate::convert::make_cpu_device|ColMajorMatrix
EOF
)

if MATCHES="$(rg -n -e "$FORBIDDEN_PATTERN" "$TARGET_FILE")"; then
  echo "error: CPU fallback references found in Metal prover hot path: $TARGET_FILE" >&2
  echo "$MATCHES" >&2
  echo >&2
  echo "forbidden symbols:" >&2
  echo "  - stacked_pcs::stacked_commit" >&2
  echo "  - prove_zerocheck_and_logup" >&2
  echo "  - StackedReductionCpu" >&2
  echo "  - prove_whir_opening" >&2
  echo "  - crate::convert::mpk_to_cpu" >&2
  echo "  - crate::convert::ctx_to_cpu" >&2
  echo "  - crate::convert::make_cpu_device" >&2
  echo "  - ColMajorMatrix" >&2
  exit 1
fi

echo "ok: no CPU fallback references found in $TARGET_FILE"
