#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector_types.h>

template <typename Field>
__device__ __forceinline__ Field horner_eval(const Field *coeffs, size_t len, Field x) {
    Field acc = coeffs[len - 1];
    for (size_t idx = len - 1; idx > 0; idx--) {
        acc = acc * x + coeffs[idx - 1];
    }
    return acc;
}

// Parallel polynomial evaluation using chunk-based Horner with tree reduction.
// Coefficients are FpExt stored in F-column major form (4 x len layout).
// BlockSize must be a power of 2. Uses dynamic shared memory.
// Kernel uses a single block.
template <int BlockSize>
__global__ void eval_poly_ext_at_point_kernel(const Fp *coeffs, size_t len, FpExt x, FpExt *out) {
    // Dynamic shared memory: BlockSize elements for partial sums + 1 for x^chunk_size
    extern __shared__ FpExt smem[];

    const int tid = threadIdx.x;
    const size_t chunk_size = (len + BlockSize - 1) / BlockSize;

    // Phase 1: Thread 0 computes x^chunk_size using repeated squaring
    if (tid == 0) {
        FpExt xk = pow(x, chunk_size);
        smem[BlockSize] = xk; // Store x^chunk_size in the extra slot
    }
    __syncthreads();

    FpExt x_pow_chunk = smem[BlockSize]; // All threads read x^chunk_size

    // Phase 2: Each thread evaluates its chunk via Horner's method
    // Thread tid handles coefficients [tid * chunk_size, (tid + 1) * chunk_size)
    const size_t start = tid * chunk_size;
    const size_t end = min(start + chunk_size, len);

    FpExt acc = FpExt(Fp(0));
    if (start < len) {
        // Horner evaluation from highest to lowest degree within chunk
        // p_tid(x) = c[start] + c[start+1]*x + ... + c[end-1]*x^{end-1-start}
        for (size_t idx = end; idx > start; idx--) {
            FpExt coeff;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                coeff.elems[i] = coeffs[i * len + idx - 1];
            }
            acc = acc * x + coeff;
        }
    }

    // Phase 3: Scale partial sum by x^{tid * chunk_size}
    // Compute (x^chunk_size)^tid using binary exponentiation on tid
    FpExt my_power = pow(x_pow_chunk, tid);
    acc = acc * my_power;
    smem[tid] = acc;
    __syncthreads();

// Phase 4: Tree reduction to sum all partial results
#pragma unroll
    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = smem[tid] + smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out = smem[0];
    }
}

template <typename Field, bool EvalToCoeff>
__global__ void mle_interpolate_stage_kernel(Field *buffer, size_t total_pairs, uint32_t step) {
    size_t span = size_t(step) << 1;
    size_t pair_idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    size_t chunk = pair_idx / step;
    uint32_t offset = pair_idx % step;
    size_t base = chunk * span + offset;
    size_t second = base + step;
    if (EvalToCoeff) {
        buffer[second] -= buffer[base];
    } else {
        buffer[second] += buffer[base];
    }
}

template <typename Field, bool EvalToCoeff>
__global__ void mle_interpolate_stage_2d_kernel(
    Field *buffer,
    uint32_t padded_height,
    uint32_t span,
    uint32_t step
) {
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y;

    auto chunk = tidx / step;
    auto offset = tidx % step;
    uint32_t base = col * padded_height + chunk * span + offset;
    uint32_t second = base + step;

    if (EvalToCoeff) {
        buffer[second] -= buffer[base];
    } else {
        buffer[second] += buffer[base];
    }
}

__global__ void algebraic_batch_matrices_kernel(
    FpExt *output,           // Length is height
    const Fp *const *mats,   // Array of pointers to matrices
    const FpExt *mu_powers,  // Len is sum of widths of all matrices
    const uint32_t *mu_idxs, // Starting index in `mu_powers` for each matrix. Length = num_mats
    const uint32_t *widths,  // Width of each matrix
    size_t height,
    size_t num_mats
) {

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) {
        return;
    }

    // NOTE(perf): Depending on the number of columns, we may want to use shared memory to reduce this sum.
    output[row] = FpExt(0);
    for (int mat_idx = 0; mat_idx < num_mats; mat_idx++) {
        size_t width = widths[mat_idx];
        uint32_t mu_start_idx = mu_idxs[mat_idx];
        for (int col = 0; col < width; col++) {
            uint32_t mu_idx = mu_start_idx + col;
            FpExt mu_pow = mu_powers[mu_idx];
            output[row] += mu_pow * mats[mat_idx][col * height + row];
        }
    }
}

__global__ void eq_hypercube_stage_ext_kernel(FpExt *out, FpExt x_i, uint32_t step) {
    size_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= step)
        return;
    FpExt hi = out[y] * x_i;
    out[y | step] = hi;
    out[y] -= hi; // out[y] = out[y] * (FpExt(Fp(1)) - x_i), saves a multiplication
}

// Same as eq_hypercube_stage_ext_kernel but does not modify in-place
__global__ void eq_hypercube_nonoverlapping_stage_ext_kernel(
    FpExt *out,
    const FpExt *in,
    FpExt x_i,
    uint32_t step
) {
    size_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= step)
        return;
    FpExt hi = in[y] * x_i;
    out[y | step] = hi;
    out[y] = in[y] - hi; // save a multiplication
}

// out is `height x width` column-major matrix of evaluations of eq(x[j], -) on hypercube for j in 0..width
// This kernel is for fixed step 2^i
__global__ void batch_eq_hypercube_stage_kernel(
    Fp *out,
    Fp *x,
    uint32_t step,
    uint32_t width,
    uint32_t height
) {
    size_t y = blockIdx.x * blockDim.x + threadIdx.x;
    size_t x_idx = blockIdx.y;
    if (y >= step)
        return;
    Fp x_i = x[x_idx];
    size_t lo_idx = x_idx * height + y;
    out[lo_idx | step] = out[lo_idx] * x_i;
    out[lo_idx] *= (Fp(1) - x_i);
}

template <typename Field>
__global__ void vector_scalar_multiply_kernel(Field *vec, Field scalar, uint32_t length) {
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= length)
        return;

    vec[tidx] *= scalar;
}

__global__ void transpose_fp_to_fpext_vec_kernel(
    FpExt *__restrict__ output,
    const Fp *__restrict__ input,
    uint32_t height
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height)
        return;

#pragma unroll
    for (int i = 0; i < 4; i++) {
        output[idx].elems[i] = input[i * height + idx];
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

template <typename Field, bool EvalToCoeff>
int launch_mle_interpolate_stage(Field *buffer, size_t buffer_len, uint32_t step) {
    size_t total_pairs = buffer_len >> 1;
    auto [grid, block] = kernel_launch_params(total_pairs);
    mle_interpolate_stage_kernel<Field, EvalToCoeff><<<grid, block>>>(buffer, total_pairs, step);
    return CHECK_KERNEL();
}

extern "C" int _mle_interpolate_stage(
    Fp *buffer,
    size_t buffer_len,
    uint32_t step,
    bool is_eval_to_coeff
) {
    if (buffer_len < 2 || step == 0) {
        return 0;
    }

    if (is_eval_to_coeff) {
        return launch_mle_interpolate_stage<Fp, true>(buffer, buffer_len, step);
    } else {
        return launch_mle_interpolate_stage<Fp, false>(buffer, buffer_len, step);
    }
}

extern "C" int _mle_interpolate_stage_ext(
    FpExt *buffer,
    size_t buffer_len,
    uint32_t step,
    bool is_eval_to_coeff
) {
    if (buffer_len < 2 || step == 0) {
        return 0;
    }

    if (is_eval_to_coeff) {
        return launch_mle_interpolate_stage<FpExt, true>(buffer, buffer_len, step);
    } else {
        return launch_mle_interpolate_stage<FpExt, false>(buffer, buffer_len, step);
    }
}

template <typename Field, bool EvalToCoeff>
int launch_mle_interpolate_stage_2d(
    Field *buffer,
    uint16_t width,
    uint32_t height,
    uint32_t padded_height,
    uint32_t step
) {
    auto span = step * 2;
    auto [grid, block] = kernel_launch_params(height >> 1);
    grid.y = width;
    mle_interpolate_stage_2d_kernel<Field, EvalToCoeff>
        <<<grid, block>>>(buffer, padded_height, span, step);
    return CHECK_KERNEL();
}

extern "C" int _mle_interpolate_stage_2d(
    Fp *buffer,
    uint16_t width,
    uint32_t height,
    uint32_t padded_height,
    uint32_t step,
    bool is_eval_to_coeff
) {
    if (is_eval_to_coeff) {
        return launch_mle_interpolate_stage_2d<Fp, true>(
            buffer, width, height, padded_height, step
        );
    } else {
        return launch_mle_interpolate_stage_2d<Fp, false>(
            buffer, width, height, padded_height, step
        );
    }
}

extern "C" int _algebraic_batch_matrices(
    FpExt *output,           // Length is height
    const Fp *const *mats,   // Array of pointers to matrices
    const FpExt *mu_powers,  // Len is sum of widths of all matrices
    const uint32_t *mu_idxs, // Starting index in `mu_powers` for each matrix. Length = num_mats
    const uint32_t *widths,  // Width of each matrix
    size_t height,
    size_t num_mats
) {
    auto [grid, block] = kernel_launch_params(height);
    algebraic_batch_matrices_kernel<<<grid, block>>>(
        output, mats, mu_powers, mu_idxs, widths, height, num_mats
    );
    return CHECK_KERNEL();
}

extern "C" int _eq_hypercube_stage_ext(FpExt *out, FpExt x_i, uint32_t step) {
    auto [grid, block] = kernel_launch_params(step);
    eq_hypercube_stage_ext_kernel<<<grid, block>>>(out, x_i, step);
    return CHECK_KERNEL();
}

extern "C" int _eq_hypercube_nonoverlapping_stage_ext(
    FpExt *out,
    const FpExt *in,
    FpExt x_i,
    uint32_t step
) {
    auto [grid, block] = kernel_launch_params(step);
    eq_hypercube_nonoverlapping_stage_ext_kernel<<<grid, block>>>(out, in, x_i, step);
    return CHECK_KERNEL();
}

extern "C" int _batch_eq_hypercube_stage(
    Fp *out,
    Fp *x,
    uint32_t step,
    uint32_t width,
    uint32_t height
) {

    auto [grid, block] = kernel_launch_params(step);
    grid.y = width;
    batch_eq_hypercube_stage_kernel<<<grid, block>>>(out, x, step, width, height);
    return CHECK_KERNEL();
}

// Helper to dispatch templated kernel based on runtime block size
template <int BlockSize>
int launch_eval_poly_ext_at_point(const Fp *coeffs, size_t len, FpExt x, FpExt *out) {
    // Dynamic shared memory: BlockSize + 1 elements for partial sums and x^chunk_size
    size_t smem_size = (BlockSize + 1) * sizeof(FpExt);
    eval_poly_ext_at_point_kernel<BlockSize><<<1, BlockSize, smem_size>>>(coeffs, len, x, out);
    return CHECK_KERNEL();
}

extern "C" int _eval_poly_ext_at_point(const Fp *coeffs, size_t len, FpExt x, FpExt *out) {
    // Choose block size based on polynomial length for optimal performance
    // Larger polynomials benefit from more parallelism
    if (len <= 256) {
        return launch_eval_poly_ext_at_point<64>(coeffs, len, x, out);
    } else if (len <= 4096) {
        return launch_eval_poly_ext_at_point<128>(coeffs, len, x, out);
    } else if (len <= 65536) {
        return launch_eval_poly_ext_at_point<256>(coeffs, len, x, out);
    } else if (len <= 524288) {
        return launch_eval_poly_ext_at_point<512>(coeffs, len, x, out);
    } else {
        return launch_eval_poly_ext_at_point<1024>(coeffs, len, x, out);
    }
}

extern "C" int _vector_scalar_multiply_ext(FpExt *vec, FpExt scalar, uint32_t length) {
    auto [grid, block] = kernel_launch_params(length);
    vector_scalar_multiply_kernel<FpExt><<<grid, block>>>(vec, scalar, length);
    return CHECK_KERNEL();
}

extern "C" int _transpose_fp_to_fpext_vec(FpExt *output, const Fp *input, uint32_t height) {
    auto [grid, block] = kernel_launch_params(height);
    transpose_fp_to_fpext_vec_kernel<<<grid, block>>>(output, input, height);
    return CHECK_KERNEL();
}
