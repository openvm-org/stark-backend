#!/usr/bin/env bash
# Re-extract the SWIRL verifier ULLBC for the rust-to-lean translator.
# Run from the stark-backend repo root. See rust-to-lean memory phase-e-verifier-bringup.
set -euo pipefail
CHARON="${CHARON:-$HOME/Documents/GitHub/rust-consistency-tools/aeneas/charon/bin/charon}"
cd "$(dirname "$0")/crates/stark-sdk"
"$CHARON" cargo --ullbc --monomorphize --index-to-function-calls --treat-box-as-builtin \
  --start-from 'openvm_stark_sdk::verify_mono::verify_babybear_poseidon2' \
  --start-from 'openvm_stark_sdk::verify_mono::force_exp_powers_next' \
  --include 'openvm_stark_backend::_' \
  --opaque p3_air --opaque p3_baby_bear --opaque p3_challenger --opaque p3_dft \
  --opaque p3_field --opaque p3_interpolation --opaque p3_keccak_air --opaque p3_matrix \
  --opaque p3_maybe_rayon --opaque p3_mds --opaque p3_monty_31 --opaque p3_poseidon2 \
  --opaque p3_symmetric --opaque p3_util --dest-file /tmp/verifier2.ullbc
