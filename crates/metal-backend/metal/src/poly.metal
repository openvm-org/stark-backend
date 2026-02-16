/// Polynomial operation kernels for Metal.
/// Translated from cuda-backend/cuda/src/poly.cu.

#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"

// ============================================================================
// Algebraic Batching
// ============================================================================

/// Compute a batched linear combination of multiple column-major Fp matrices
/// into a single column of FpExt values.
/// output[row] = sum over mats: sum over cols: mu_powers[mu_idx+col] * mat[col*height + row]
///
/// All matrices are packed into a single contiguous buffer `all_mats_data`.
/// `mat_offsets[i]` gives the element offset of matrix i within `all_mats_data`.
kernel void algebraic_batch_matrices(
    device FpExt *output [[buffer(0)]],
    const device Fp *all_mats_data [[buffer(1)]],
    const device uint32_t *mat_offsets [[buffer(2)]],
    const device FpExt *mu_powers [[buffer(3)]],
    const device uint32_t *mu_idxs [[buffer(4)]],
    const device uint32_t *widths [[buffer(5)]],
    constant uint32_t &height [[buffer(6)]],
    constant uint32_t &num_mats [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t row = tid;
    if (row >= height) return;

    output[row] = FpExt(0);
    for (uint32_t mat_idx = 0; mat_idx < num_mats; mat_idx++) {
        uint32_t w = widths[mat_idx];
        uint32_t mu_start = mu_idxs[mat_idx];
        uint32_t mat_base = mat_offsets[mat_idx];
        for (uint32_t col = 0; col < w; col++) {
            FpExt mu_pow = mu_powers[mu_start + col];
            output[row] += mu_pow * all_mats_data[mat_base + col * height + row];
        }
    }
}

// ============================================================================
// Equality Hypercube Kernels
// ============================================================================

/// In-place eq hypercube stage (insert x_i from the back).
/// out[y | step] = out[y] * x_i
/// out[y] = out[y] * (1 - x_i)  [saves one mul: out[y] -= hi]
kernel void eq_hypercube_stage_ext(
    device FpExt *out [[buffer(0)]],
    constant FpExt &x_i [[buffer(1)]],
    constant uint32_t &step [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= step) return;

    FpExt hi = out[tid] * x_i;
    out[tid | step] = hi;
    out[tid] -= hi;
}

/// Mobius-adjusted equality hypercube stage.
/// K_i(0) = 1 - 2*omega_i, K_i(1) = omega_i
kernel void mobius_eq_hypercube_stage_ext(
    device FpExt *out [[buffer(0)]],
    constant FpExt &omega_i [[buffer(1)]],
    constant uint32_t &step [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= step) return;

    FpExt prev = out[tid];
    FpExt hi = prev * omega_i;
    out[tid | step] = hi;
    out[tid] = prev - hi - hi; // prev * (1 - 2*omega_i)
}

/// Non-overlapping eq hypercube stage (separate input and output buffers).
kernel void eq_hypercube_nonoverlapping_stage_ext(
    device FpExt *out [[buffer(0)]],
    const device FpExt *input [[buffer(1)]],
    constant FpExt &x_i [[buffer(2)]],
    constant uint32_t &step [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= step) return;

    FpExt prev = input[tid];
    FpExt hi = prev * x_i;
    out[tid | step] = hi;
    out[tid] = prev - hi;
}

/// Interleaved eq hypercube stage (insert x_i from the front).
kernel void eq_hypercube_interleaved_stage_ext(
    device FpExt *out [[buffer(0)]],
    const device FpExt *input [[buffer(1)]],
    constant FpExt &x_i [[buffer(2)]],
    constant uint32_t &step [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= step) return;

    FpExt prev = input[tid];
    FpExt hi = prev * x_i;
    out[(tid << 1) | 1] = hi;
    out[tid << 1] = prev - hi;
}

/// Batch eq hypercube stage for Fp.
/// out is height x width column-major matrix.
kernel void batch_eq_hypercube_stage(
    device Fp *out [[buffer(0)]],
    const device Fp *x [[buffer(1)]],
    constant uint32_t &step [[buffer(2)]],
    constant uint32_t &width [[buffer(3)]],
    constant uint32_t &height [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]] // (y, x_idx)
) {
    uint32_t y = gid.x;
    uint32_t x_idx = gid.y;
    if (y >= step) return;

    Fp x_i = x[x_idx];
    uint32_t lo_idx = x_idx * height + y;
    out[lo_idx | step] = out[lo_idx] * x_i;
    out[lo_idx] *= (Fp(1) - x_i);
}

// ============================================================================
// Polynomial Evaluation
// ============================================================================

/// Evaluate a polynomial (with FpExt coefficients stored in F-column-major
/// form as 4 x len Fp layout) at a given FpExt point.
/// Uses a single threadgroup with chunk-based Horner evaluation.
kernel void eval_poly_ext_at_point(
    const device Fp *coeffs [[buffer(0)]],
    constant uint32_t &len [[buffer(1)]],
    constant FpExt &x [[buffer(2)]],
    device FpExt *out [[buffer(3)]],
    threadgroup FpExt *smem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    uint32_t chunk_size = (len + block_size - 1) / block_size;

    // Phase 1: Thread 0 computes x^chunk_size
    if (tid == 0) {
        smem[0] = pow(x, chunk_size);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    FpExt x_pow_chunk = smem[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Each thread evaluates its chunk via Horner's method
    uint32_t start = tid * chunk_size;
    uint32_t end = min(start + chunk_size, len);

    FpExt acc = FpExt(Fp(0));
    if (start < len) {
        for (uint32_t idx = end; idx > start; idx--) {
            FpExt coeff;
            for (int i = 0; i < 4; i++) {
                coeff.elems[i] = coeffs[i * len + idx - 1];
            }
            acc = acc * x + coeff;
        }
    }

    // Phase 3: Scale by x^{tid * chunk_size}
    FpExt my_power = pow(x_pow_chunk, tid);
    acc = acc * my_power;

    // Phase 4: Reduction via threadgroup memory
    // SIMD reduction first
    for (uint offset = 16; offset > 0; offset >>= 1) {
        FpExt other;
        for (int i = 0; i < 4; i++) {
            other.elems[i] = Fp::fromRaw(simd_shuffle_down(acc.elems[i].asRaw(), offset));
        }
        acc += other;
    }

    uint32_t simd_id = tid / 32;
    uint32_t lane_id = tid % 32;
    uint32_t num_simds = (block_size + 31) / 32;

    if (lane_id == 0) {
        smem[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        FpExt val = (lane_id < num_simds) ? smem[lane_id] : FpExt(0);
        for (uint offset = 16; offset > 0; offset >>= 1) {
            FpExt other;
            for (int i = 0; i < 4; i++) {
                other.elems[i] = Fp::fromRaw(simd_shuffle_down(val.elems[i].asRaw(), offset));
            }
            val += other;
        }
        if (tid == 0) {
            *out = val;
        }
    }
}

// ============================================================================
// Vector Scalar Multiply
// ============================================================================

/// Multiply each element of an FpExt vector by a scalar.
kernel void vector_scalar_multiply_ext(
    device FpExt *vec [[buffer(0)]],
    constant FpExt &scalar [[buffer(1)]],
    constant uint32_t &length [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= length) return;
    vec[tid] *= scalar;
}

// ============================================================================
// Transpose Fp to FpExt
// ============================================================================

/// Transpose 4 x height Fp data (column-major) into height FpExt elements.
kernel void transpose_fp_to_fpext_vec(
    device FpExt *output [[buffer(0)]],
    const device Fp *input [[buffer(1)]],
    constant uint32_t &height [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= height) return;

    for (int i = 0; i < 4; i++) {
        output[tid].elems[i] = input[i * height + tid];
    }
}
