// logup_zerocheck/utils - Utility kernels for logup/zerocheck protocol
// Ported from CUDA: cuda-backend/cuda/src/logup_zerocheck/utils.cu
#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"
#include "sumcheck.h"

struct ColumnPtrExt {
    const device FpExt *data;
};

// Fold PLE evaluations by barycentric interpolation on coset domain
// Input: column-major matrix [height * width] of Fp evaluations
// Output: column-major matrix [new_height * width] of FpExt folded evaluations
// For ROTATE=true, accesses (x * skip_domain + z + 1) % height
kernel void fold_ple_from_evals_kernel(
    const device Fp *input_matrix [[buffer(0)]],
    device FpExt *output_matrix [[buffer(1)]],
    const device Fp *omega_skip_pows [[buffer(2)]],
    const device FpExt *inv_lagrange_denoms [[buffer(3)]],
    constant uint32_t &height [[buffer(4)]],
    constant uint32_t &skip_domain [[buffer(5)]],
    constant uint32_t &l_skip [[buffer(6)]],
    constant uint32_t &new_height [[buffer(7)]],
    constant uint32_t &col [[buffer(8)]],
    constant uint32_t &do_rotate [[buffer(9)]],
    threadgroup FpExt *smem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    uint chunks_per_block = tg_size / skip_domain;
    uint chunk_in_block = tid / skip_domain;
    uint tid_in_chunk = tid % skip_domain;
    uint x = gid * chunks_per_block + chunk_in_block;

    bool active_chunk = (x < new_height);

    FpExt local_val = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    if (active_chunk) {
        uint z = tid_in_chunk;
        uint offset = do_rotate ? 1 : 0;
        uint row_idx = ((x << l_skip) + z + offset) % height;
        uint input_idx = col * height + row_idx;
        Fp eval = input_matrix[input_idx];
        local_val = inv_lagrange_denoms[z] * omega_skip_pows[z] * eval;
    }

    smem[tid] = local_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active_chunk && tid_in_chunk == 0) {
        FpExt result = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
        uint base = chunk_in_block * skip_domain;
        for (uint i = 0; i < skip_domain; ++i) {
            result = result + smem[base + i];
        }
        output_matrix[col * new_height + x] = result;
    }
}

// Interpolate columns: given pairs (t0, t1), compute t0 + (t1 - t0) * (x+1) for x in [0, s_deg)
kernel void interpolate_columns_kernel(
    device FpExt *interpolated [[buffer(0)]],
    const device ColumnPtrExt *columns [[buffer(1)]],
    constant uint32_t &s_deg [[buffer(2)]],
    constant uint32_t &num_y [[buffer(3)]],
    constant uint32_t &num_columns [[buffer(4)]],
    uint tidx [[thread_position_in_grid]]
) {
    uint y = tidx % num_y;
    uint col_idx = tidx / num_y;
    if (col_idx >= num_columns) return;

    const device FpExt *column = columns[col_idx].data;
    FpExt t0 = column[y << 1];
    FpExt t1 = column[(y << 1) | 1];
    device FpExt *this_interpolated = interpolated + col_idx * s_deg * num_y;

    for (uint x = 0; x < s_deg; x++) {
        this_interpolated[x * num_y + y] = t0 + (t1 - t0) * Fp(x + 1u);
    }
}

// Vertically repeat a FracExt matrix: out[col*lifted_h + row] = in[col*h + (row%h)]
kernel void frac_matrix_vertically_repeat_kernel(
    device FpExt *out_p [[buffer(0)]],
    device FpExt *out_q [[buffer(1)]],
    const device FpExt *in_p [[buffer(2)]],
    const device FpExt *in_q [[buffer(3)]],
    constant uint32_t &width [[buffer(4)]],
    constant uint32_t &lifted_height [[buffer(5)]],
    constant uint32_t &height [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint col = gid.y;
    if (col >= width || row >= lifted_height) return;

    uint src_row = row % height;
    uint src_idx = col * height + src_row;
    uint dst_idx = col * lifted_height + row;
    out_p[dst_idx] = in_p[src_idx];
    out_q[dst_idx] = in_q[src_idx];
}
