#!/usr/bin/env bash
set -euo pipefail

NVCC="${NVCC:-nvcc}"
NCU="${NCU:-ncu}"
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
OUT="${OUT:-./bn254_bench}"
KERNEL="bn254_compressing_row_hashes_kernel_v2"

extra_flags=()
if [[ -n "${CUDA_ARCH_FLAGS:-}" ]]; then
  read -r -a extra_flags <<<"${CUDA_ARCH_FLAGS}"
fi

"${NVCC}" \
  -O3 \
  -std=c++17 \
  --expt-relaxed-constexpr \
  -Xptxas -v \
  "${extra_flags[@]}" \
  -Icrates/cuda-common/include \
  -Icrates/cuda-backend/cuda/include \
  -lineinfo \
  bn254_bench.cu \
  bn254_compress.cu \
  -o "${OUT}"

sass_count=$("${CUOBJDUMP}" --dump-sass "${OUT}" | awk -v kernel="${KERNEL}" '
  /^[[:space:]]*Function : / {
    in_fn = (index($0, kernel) > 0) ? 1 : 0
    next
  }
  in_fn && /^[[:space:]]*\/\*[0-9a-fA-F]+\*\// { count++ }
  END { print count + 0 }
')
printf "SASS instruction count for %s: %d\n\n" "${KERNEL}" "${sass_count}"

$OUT
# if "${OUT}" "$@"; then
#   printf "\nOutputs match; profiling %s with ncu\n\n" "${KERNEL}"
#   "${NCU}" \
#     --kernel-name "regex:${KERNEL}" \
#     --launch-count 1 \
#     --section SpeedOfLight \
#     --section LaunchStats \
#     --section WarpStateStats \
#     "${OUT}" "$@"
# else
#   status=$?
#   printf "Outputs did not match; skipping ncu\n" >&2
#   exit "${status}"
# fi
