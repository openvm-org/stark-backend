#pragma once

#include <cstdint>

// Per-block descriptor for flat-block-list dispatch across heterogeneous AIRs.
// Used by kernels that pack varying per-AIR block counts into a single 1D grid.
//
// `air_idx` is local to the kernel's per-AIR context buffer (not a global AIR index).
struct BlockCtx {
    uint32_t local_block_idx_x;
    uint32_t air_idx;
};
