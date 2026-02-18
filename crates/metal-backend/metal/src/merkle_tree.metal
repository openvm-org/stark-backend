// Merkle tree kernels for Metal
// Translated from CUDA: cuda-backend/cuda/src/merkle_tree.cu
#include <metal_stdlib>
using namespace metal;

#include "../include/baby_bear.h"
#include "../include/baby_bear_ext.h"
#include "../include/poseidon2.h"

struct digest_t {
    Fp cells[CELLS_OUT];
};

// Row hash kernel for Fp matrices.
// Each thread handles one stride_idx. The y-dimension (leaf_idx) is flattened
// into the thread grid since Metal does not have blockDim.y in the CUDA sense.
// Instead, Rust dispatch uses threads_per_threadgroup = (stride_threads, rows_per_query).
kernel void poseidon2_compressing_row_hashes_kernel(
    device digest_t *out                    [[buffer(0)]],
    const device Fp *matrix                 [[buffer(1)]],
    constant uint32_t &width                [[buffer(2)]],
    constant uint32_t &height               [[buffer(3)]],
    constant uint32_t &query_stride         [[buffer(4)]],
    constant uint32_t &log_rows_per_query   [[buffer(5)]],
    threadgroup Fp *shared                  [[threadgroup(0)]],
    uint2 tid                               [[thread_position_in_threadgroup]],
    uint2 gid                               [[threadgroup_position_in_grid]],
    uint2 tpg                               [[threads_per_threadgroup]]
) {
    uint32_t stride_idx = tpg.x * gid.x + tid.x;
    uint32_t leaf_idx = tid.y;
    uint32_t row = leaf_idx * query_stride + stride_idx;
    uint32_t shared_stride = tpg.x * (tpg.y >> 1);

    uint32_t used = 0;
    Fp cells[CELLS];
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = Fp(0u);
    }

    if (stride_idx < query_stride) {
        // compute row hash
        for (uint32_t col = 0; col < width; col++) {
            cells[used++] = matrix[col * height + row];
            if (used == CELLS_RATE) {
                poseidon2_mix(cells);
                used = 0;
            }
        }
        if (used != 0) {
            poseidon2_mix(cells);
        }
    }

    for (uint32_t layer = 0; layer < log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        uint32_t shared_offset = ((leaf_idx >> (layer + 1)) << layer) * tpg.x + tid.x;
        if ((leaf_idx & mask) == (1u << layer)) {
            for (uint i = 0; i < CELLS_OUT; i++) {
                shared[i * shared_stride + shared_offset] = cells[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((leaf_idx & mask) == 0) {
            for (uint i = 0; i < CELLS_OUT; i++) {
                cells[CELLS_OUT + i] = shared[i * shared_stride + shared_offset];
            }
            poseidon2_mix(cells);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
        for (uint i = 0; i < CELLS_OUT; i++) {
            out[stride_idx].cells[i] = cells[i];
        }
    }
}

// Row hash kernel for FpExt matrices.
kernel void poseidon2_compressing_row_hashes_ext_kernel(
    device digest_t *out                    [[buffer(0)]],
    const device FpExt *matrix              [[buffer(1)]],
    constant uint32_t &width                [[buffer(2)]],
    constant uint32_t &height               [[buffer(3)]],
    constant uint32_t &query_stride         [[buffer(4)]],
    constant uint32_t &log_rows_per_query   [[buffer(5)]],
    threadgroup Fp *shared                  [[threadgroup(0)]],
    uint2 tid                               [[thread_position_in_threadgroup]],
    uint2 gid                               [[threadgroup_position_in_grid]],
    uint2 tpg                               [[threads_per_threadgroup]]
) {
    uint32_t stride_idx = tpg.x * gid.x + tid.x;
    uint32_t leaf_idx = tid.y;
    uint32_t row = leaf_idx * query_stride + stride_idx;
    uint32_t shared_stride = tpg.x * (tpg.y >> 1);

    uint32_t used = 0;
    Fp cells[CELLS];
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = Fp(0u);
    }

    if (stride_idx < query_stride) {
        for (uint32_t col = 0; col < width; col++) {
            // Extension field degree is 4
            for (uint i = 0; i < 4; i++) {
                cells[used++] = matrix[col * height + row].elems[i];
                if (used == CELLS_RATE) {
                    poseidon2_mix(cells);
                    used = 0;
                }
            }
        }
        if (used != 0) {
            poseidon2_mix(cells);
        }
    }

    for (uint32_t layer = 0; layer < log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        uint32_t shared_offset = ((leaf_idx >> (layer + 1)) << layer) * tpg.x + tid.x;
        if ((leaf_idx & mask) == (1u << layer)) {
            for (uint i = 0; i < CELLS_OUT; i++) {
                shared[i * shared_stride + shared_offset] = cells[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((leaf_idx & mask) == 0) {
            for (uint i = 0; i < CELLS_OUT; i++) {
                cells[CELLS_OUT + i] = shared[i * shared_stride + shared_offset];
            }
            poseidon2_mix(cells);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
        for (uint i = 0; i < CELLS_OUT; i++) {
            out[stride_idx].cells[i] = cells[i];
        }
    }
}

// Strided compress: pairs adjacent digests and hashes them
kernel void poseidon2_strided_compress_layer_kernel(
    device digest_t *output                 [[buffer(0)]],
    const device digest_t *prev_layer       [[buffer(1)]],
    constant uint32_t &output_size          [[buffer(2)]],
    constant uint32_t &stride               [[buffer(3)]],
    uint gid                                [[thread_position_in_grid]]
) {
    if (gid >= output_size) {
        return;
    }
    uint32_t x = gid / stride;
    uint32_t y = gid % stride;

    Fp cells[CELLS];
    for (uint i = 0; i < CELLS_OUT; i++) {
        cells[i] = prev_layer[2 * x * stride + y].cells[i];
        cells[i + CELLS_OUT] = prev_layer[(2 * x + 1) * stride + y].cells[i];
    }

    poseidon2_mix(cells);

    for (uint i = 0; i < CELLS_OUT; i++) {
        output[gid].cells[i] = cells[i];
    }
}

// Adjacent compress (stride=1 specialization)
kernel void poseidon2_adjacent_compress_layer_kernel(
    device digest_t *output                 [[buffer(0)]],
    const device digest_t *prev_layer       [[buffer(1)]],
    constant uint32_t &output_size          [[buffer(2)]],
    uint gid                                [[thread_position_in_grid]]
) {
    if (gid >= output_size) {
        return;
    }

    Fp cells[CELLS];
    for (uint i = 0; i < CELLS_OUT; i++) {
        cells[i] = prev_layer[2 * gid].cells[i];
        cells[i + CELLS_OUT] = prev_layer[2 * gid + 1].cells[i];
    }

    poseidon2_mix(cells);

    for (uint i = 0; i < CELLS_OUT; i++) {
        output[gid].cells[i] = cells[i];
    }
}

// Query digest layers: gather digest elements for opening proofs
kernel void query_digest_layers_kernel(
    device Fp *d_digest_matrix              [[buffer(0)]],
    const device uint64_t *d_layers_ptr     [[buffer(1)]],
    const device uint64_t *d_indices        [[buffer(2)]],
    constant uint64_t &num_query            [[buffer(3)]],
    constant uint64_t &num_layer            [[buffer(4)]],
    uint2 gid                               [[thread_position_in_grid]]
) {
    uint32_t ELEM_PER_DIGEST = CELLS_OUT; // 8
    uint64_t flat_idx = gid.x;
    uint64_t layer_idx = flat_idx / ELEM_PER_DIGEST;
    uint64_t elem_offset = flat_idx % ELEM_PER_DIGEST;
    uint64_t query_idx = gid.y;

    if (layer_idx >= num_layer) {
        return;
    }

    // d_layers_ptr[layer_idx] is a device pointer to the layer data
    const device Fp *d_layer = reinterpret_cast<const device Fp *>(d_layers_ptr[layer_idx]);
    uint64_t digest_offset = d_indices[query_idx * num_layer + layer_idx];
    Fp digest_elem = d_layer[digest_offset * ELEM_PER_DIGEST + elem_offset];

    uint64_t output_query_offset = query_idx * num_layer * ELEM_PER_DIGEST;
    uint64_t output_layer_offset = layer_idx * ELEM_PER_DIGEST + elem_offset;
    d_digest_matrix[output_query_offset + output_layer_offset] = digest_elem;
}
