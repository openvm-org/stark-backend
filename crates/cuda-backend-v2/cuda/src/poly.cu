#include "fp.h"
#include "launcher.cuh"
#include <cstddef>
#include <cstdint>

__global__ void ple_interpolate_stage_kernel(Fp *buffer, size_t total_pairs,
                                             uint32_t step) {
  size_t span = size_t(step) << 1;
  size_t pair_idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  size_t chunk = pair_idx / step;
  uint32_t offset = pair_idx % step;
  size_t base = chunk * span + offset;
  size_t second = base + step;
  buffer[second] -= buffer[base];
}

extern "C" int _ple_interpolate_stage(Fp *buffer, size_t buffer_len,
                                      uint32_t step) {
  if (buffer_len < 2 || step == 0) {
    return 0;
  }

  size_t total_pairs = buffer_len >> 1;
  auto [grid, block] = kernel_launch_params(total_pairs);
  ple_interpolate_stage_kernel<<<grid, block>>>(buffer, total_pairs, step);
  return CHECK_KERNEL();
}
