#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
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

// Single-thread Horner evaluation used when only one point is required.
template <typename Field>
__global__ void eval_poly_at_point_kernel(const Field *coeffs, size_t len, Field x, Field *out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    Field acc = horner_eval(coeffs, len, x);
    *out = acc;
}

// Single-threaded Horner evaluation, but `coeffs` is the coefficients of `EF`-polynomial, but in F-column major form.
__global__ void eval_poly_ext_at_point_kernel(const Fp *coeffs, size_t len, FpExt x, FpExt *out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    FpExt acc;
    for (int i = 0; i < 4; i++) {
        acc.elems[i] = coeffs[i * len + len - 1];
    }
    for (size_t idx = len - 1; idx > 0; idx--) {
        FpExt coeff;
        for (int i = 0; i < 4; i++) {
            coeff.elems[i] = coeffs[i * len + idx - 1];
        }
        acc = acc * x + coeff;
    }
    *out = acc;
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

// Horner evaluation using just one thread.
template <typename Field>
int launch_eval_poly_at_point(const Field *coeffs, size_t len, Field x, Field *out) {
    dim3 grid(1);
    dim3 block(32);
    eval_poly_at_point_kernel<Field><<<grid, block>>>(coeffs, len, x, out);
    return CHECK_KERNEL();
}

extern "C" int _eval_poly_at_point(const Fp *coeffs, size_t len, Fp x, Fp *out) {
    return launch_eval_poly_at_point(coeffs, len, x, out);
}

extern "C" int _eval_poly_ext_at_point(const Fp *coeffs, size_t len, FpExt x, FpExt *out) {
    dim3 grid(1);
    dim3 block(32);
    eval_poly_ext_at_point_kernel<<<grid, block>>>(coeffs, len, x, out);
    return CHECK_KERNEL();
}

extern "C" int _vector_scalar_multiply_ext(FpExt *vec, FpExt scalar, uint32_t length) {
    auto [grid, block] = kernel_launch_params(length);
    vector_scalar_multiply_kernel<FpExt><<<grid, block>>>(vec, scalar, length);
    return CHECK_KERNEL();
}
