/// Matrix operation kernels for Metal.
/// Translated from cuda-backend/cuda/src/matrix.cu.

#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"

constant uint32_t TILE_SIZE = 32;

// ============================================================================
// Matrix Transpose
// ============================================================================

/// Transpose a column-major Fp matrix using threadgroup memory tiling.
/// input is col_size x row_size, output is row_size x col_size.
kernel void matrix_transpose_fp(
    device Fp *output [[buffer(0)]],
    const device Fp *input [[buffer(1)]],
    constant uint32_t &col_size [[buffer(2)]],
    constant uint32_t &row_size [[buffer(3)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup Fp s_mem[TILE_SIZE][TILE_SIZE + 1];

    uint32_t dim_x = (col_size + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t bid_y = bid / dim_x;
    uint32_t bid_x = bid % dim_x;

    uint32_t index_i = bid_y * TILE_SIZE * col_size + bid_x * TILE_SIZE + tid;
    uint32_t index_o = bid_x * TILE_SIZE * row_size + bid_y * TILE_SIZE + tid;

    // Load tile into threadgroup memory
    bool boundary_column = bid_x * TILE_SIZE + tid < col_size;
    uint32_t row_offset = bid_y * TILE_SIZE;
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        bool boundary = boundary_column && (row_offset + i < row_size);
        s_mem[i][tid] = boundary ? input[index_i + i * col_size] : Fp(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write transposed tile
    boundary_column = bid_y * TILE_SIZE + tid < row_size;
    row_offset = bid_x * TILE_SIZE;
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        bool boundary = boundary_column && (row_offset + i < col_size);
        if (boundary) {
            output[index_o + i * row_size] = s_mem[tid][i];
        }
    }
}

/// Transpose for FpExt elements.
kernel void matrix_transpose_fpext(
    device FpExt *output [[buffer(0)]],
    const device FpExt *input [[buffer(1)]],
    constant uint32_t &col_size [[buffer(2)]],
    constant uint32_t &row_size [[buffer(3)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup FpExt s_mem[TILE_SIZE][TILE_SIZE + 1];

    uint32_t dim_x = (col_size + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t bid_y = bid / dim_x;
    uint32_t bid_x = bid % dim_x;

    uint32_t index_i = bid_y * TILE_SIZE * col_size + bid_x * TILE_SIZE + tid;
    uint32_t index_o = bid_x * TILE_SIZE * row_size + bid_y * TILE_SIZE + tid;

    bool boundary_column = bid_x * TILE_SIZE + tid < col_size;
    uint32_t row_offset = bid_y * TILE_SIZE;
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        bool boundary = boundary_column && (row_offset + i < row_size);
        s_mem[i][tid] = boundary ? input[index_i + i * col_size] : FpExt(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    boundary_column = bid_y * TILE_SIZE + tid < row_size;
    row_offset = bid_x * TILE_SIZE;
    for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        bool boundary = boundary_column && (row_offset + i < col_size);
        if (boundary) {
            output[index_o + i * row_size] = s_mem[tid][i];
        }
    }
}

// ============================================================================
// Matrix Row Selection
// ============================================================================

/// Extract specific rows from a column-major Fp matrix into a row-major output.
kernel void matrix_get_rows_fp(
    device Fp *output [[buffer(0)]],
    const device Fp *input [[buffer(1)]],
    const device uint32_t *row_indices [[buffer(2)]],
    constant uint64_t &matrix_width [[buffer(3)]],
    constant uint64_t &matrix_height [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]] // (col_idx, output_row)
) {
    uint32_t col_idx = gid.x;
    if (col_idx >= matrix_width) return;

    uint64_t input_row = row_indices[gid.y];
    uint64_t output_row = gid.y;
    uint64_t input_idx = col_idx * matrix_height + input_row;   // col-major
    uint64_t output_idx = output_row * matrix_width + col_idx;  // row-major
    output[output_idx] = input[input_idx];
}

// ============================================================================
// Extension Field Split
// ============================================================================

/// Split a column-major FpExt vector into 4 column-major Fp columns.
kernel void split_ext_to_base_col_major(
    device Fp *d_matrix [[buffer(0)]],
    const device FpExt *d_poly [[buffer(1)]],
    constant uint64_t &poly_len [[buffer(2)]],
    constant uint32_t &matrix_height [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t row_idx = tid;
    if (row_idx >= matrix_height) return;

    uint32_t col_num = uint32_t(poly_len / matrix_height);
    for (uint32_t col_idx = 0; col_idx < col_num; col_idx++) {
        FpExt ext_val = d_poly[col_idx * matrix_height + row_idx];
        d_matrix[(col_idx * 4 + 0) * matrix_height + row_idx] = ext_val.elems[0];
        d_matrix[(col_idx * 4 + 1) * matrix_height + row_idx] = ext_val.elems[1];
        d_matrix[(col_idx * 4 + 2) * matrix_height + row_idx] = ext_val.elems[2];
        d_matrix[(col_idx * 4 + 3) * matrix_height + row_idx] = ext_val.elems[3];
    }
}

// ============================================================================
// Batch Rotate and Pad
// ============================================================================

/// Rotate by 1 and zero-pad from domain_size to padded_size.
/// Rotation is with respect to domain_size * num_x.
kernel void batch_rotate_pad(
    constant uint64_t &out_ptr [[buffer(0)]],
    constant uint64_t &in_ptr [[buffer(1)]],
    constant uint32_t &width [[buffer(2)]],
    constant uint32_t &num_x [[buffer(3)]],
    constant uint32_t &domain_size [[buffer(4)]],
    constant uint32_t &padded_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]] // (tidx, pidx)
) {
    device Fp *out = reinterpret_cast<device Fp *>(out_ptr);
    const device Fp *in = reinterpret_cast<const device Fp *>(in_ptr);
    uint32_t tidx = gid.x;
    uint32_t pidx = gid.y;

    if (pidx >= width * num_x) return;

    if (tidx < domain_size) {
        uint32_t tidx_rot = tidx + 1;
        uint32_t pidx_rot = pidx;
        if (tidx_rot == domain_size) {
            tidx_rot = 0;
            pidx_rot += 1;
            if (pidx_rot % num_x == 0) {
                pidx_rot -= num_x;
            }
        }
        out[padded_size * pidx + tidx] = in[domain_size * pidx_rot + tidx_rot];
    } else if (tidx < padded_size) {
        out[padded_size * pidx + tidx] = Fp(0);
    }
}

// ============================================================================
// Batch Expand and Pad
// ============================================================================

/// Expand each polynomial from inSize to outSize, zero-padding the rest.
kernel void batch_expand_pad(
    device Fp *out [[buffer(0)]],
    const device Fp *in [[buffer(1)]],
    constant uint32_t &polyCount [[buffer(2)]],
    constant uint32_t &outSize [[buffer(3)]],
    constant uint32_t &inSize [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t idx = tid;
    if (idx >= outSize) return;

    for (uint32_t i = 0; i < polyCount; i++) {
        Fp res = (idx < inSize) ? in[i * inSize + idx] : Fp(0);
        out[i * outSize + idx] = res;
    }
}

/// Wide variant: when width is large but height is small.
/// Column-major: out is padded_height x width, in is height x width.
kernel void batch_expand_pad_wide(
    device Fp *out [[buffer(0)]],
    const device Fp *in [[buffer(1)]],
    constant uint32_t &width [[buffer(2)]],
    constant uint32_t &padded_height [[buffer(3)]],
    constant uint32_t &height [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]] // (row, col)
) {
    uint32_t row = gid.x;
    uint32_t col = gid.y;
    if (col >= width) return;

    if (row < height) {
        out[col * padded_height + row] = in[col * height + row];
    } else if (row < padded_height) {
        out[col * padded_height + row] = Fp(0);
    }
}

// ============================================================================
// Collapse Strided Matrix
// ============================================================================

/// Downsample a matrix by taking every stride-th row.
kernel void collapse_strided_matrix(
    device Fp *out [[buffer(0)]],
    const device Fp *in [[buffer(1)]],
    constant uint32_t &width [[buffer(2)]],
    constant uint32_t &lifted_height [[buffer(3)]],
    constant uint32_t &height [[buffer(4)]],
    constant uint32_t &stride [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]] // (row, col)
) {
    uint32_t row = gid.x;
    uint32_t col = gid.y;
    if (row >= height || col >= width) return;

    out[col * height + row] = in[col * lifted_height + row * stride];
}

// ============================================================================
// Lift Padded Matrix Evaluations
// ============================================================================

/// Cyclically repeat the first `height` rows for `lifted_height` rows.
kernel void lift_padded_matrix_evals(
    constant uint64_t &matrix_ptr [[buffer(0)]],
    constant uint32_t &width [[buffer(1)]],
    constant uint32_t &height [[buffer(2)]],
    constant uint32_t &lifted_height [[buffer(3)]],
    constant uint32_t &padded_height [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]] // (tidx, col)
) {
    device Fp *matrix = reinterpret_cast<device Fp *>(matrix_ptr);
    uint32_t tidx = gid.x;
    uint32_t col = gid.y;
    if (tidx >= lifted_height || col >= width) return;

    matrix[col * padded_height + tidx] = matrix[col * padded_height + (tidx % height)];
}
