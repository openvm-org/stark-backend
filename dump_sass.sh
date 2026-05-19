#!/usr/bin/env bash
set -euo pipefail

NVCC="${NVCC:-nvcc}"
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
OUT="${OUT:-./bn254_core_sass}"

extra_flags=()
if [[ -n "${CUDA_ARCH_FLAGS:-}" ]]; then
  read -r -a extra_flags <<<"${CUDA_ARCH_FLAGS}"
fi

"${NVCC}" \
  -O3 \
  -std=c++17 \
  --expt-relaxed-constexpr \
  "${extra_flags[@]}" \
  -Icrates/cuda-common/include \
  -Icrates/cuda-backend/cuda/include \
  bn254_core_sass.cu \
  -o "${OUT}"

# Functions migrated to bn254_b32 (kernels exist in both variants).
b32_functions=(
  add256_ret
  sub256_ret
  mul_small
  mul_small_and_acc
  imr
  bn254_monty_mul
  bn254_add
  bn254_sub
  bn254_neg
  bn254_double
  bn254_mul
  bn254_sbox
  bn254_from_canonical
  bn254_to_canonical
  bn254_pack_base_2_31
  bn254_mds_external
  bn254_mds_internal
)

# All b64 wrappers we expose for SASS inspection.
b64_functions=(
  add256_ret
  sub256_ret
  mul_small
  mul_small_and_acc
  imr
  bn254_monty_mul
  bn254_poseidon2_permute
  bn254_add
  bn254_sub
  bn254_neg
  bn254_double
  bn254_mul
  bn254_sbox
  bn254_from_canonical
  bn254_to_canonical
  bn254_pack_base_2_31
  bn254_mds_external
  bn254_mds_internal
)

dump_one() {
  local fn=$1
  local variant=$2
  local outdir=$3
  local sass_file="${outdir}/bn254_${fn}.sass"
  "${CUOBJDUMP}" --dump-sass "${OUT}" | awk -v target="kernel_${fn}_${variant}" '
    /^[[:space:]]*Function : / {
      if (in_fn) exit
      in_fn = ($NF == target) ? 1 : 0
      if (in_fn) print
      next
    }
    in_fn { print }
  ' >"${sass_file}"

  if [[ ! -s "${sass_file}" ]]; then
    printf "WARN: %s is empty (kernel_%s_%s not found)\n" \
      "${sass_file}" "${fn}" "${variant}" >&2
  else
    lines=$(wc -l <"${sass_file}")
    printf "Wrote %s (%d lines)\n" "${sass_file}" "${lines}"
  fi
}

mkdir -p sass_b32 sass_b64

for fn in "${b32_functions[@]}"; do
  dump_one "${fn}" b32 sass_b32
done

for fn in "${b64_functions[@]}"; do
  dump_one "${fn}" b64 sass_b64
done

printf "\nRunning %s to verify b32 vs b64 agreement\n" "${OUT}"
"${OUT}"
