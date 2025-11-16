#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <utility>

// Includes GKR for fractional sumcheck
// The LogUp specific input to GKR is calculated in interactions.cu
namespace fractional_sumcheck_gkr {
// ============================================================================
// KERNELS
// ============================================================================
__global__ void build_level_kernel(FracExt *tree, size_t level_start, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    size_t node = level_start + idx;
    FracExt left = tree[node << 1];
    FracExt right = tree[(node << 1) + 1];
    tree[node] = frac_add(left, right);
}

__global__ void prepare_round_kernel(
    const FracExt *tree,
    size_t segment_start,
    size_t eval_size,
    FpExt *pq_out
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= eval_size) {
        return;
    }

    const FracExt left = tree[segment_start + (idx << 1)];
    const FracExt right = tree[segment_start + (idx << 1) + 1];

    pq_out[idx] = left.p;
    pq_out[eval_size + idx] = left.q;
    pq_out[(eval_size << 1) + idx] = right.p;
    pq_out[(eval_size * 3) + idx] = right.q;
}

__global__ void compute_round_kernel(
    const FpExt *eq_xi,
    const FpExt *pq,
    size_t stride,
    FpExt lambda,
    FpExt *out
) {
    extern __shared__ FpExt shared[];
    const size_t half = stride >> 1;
    const FpExt zero = {0, 0, 0, 0};

    FpExt local[3] = {zero, zero, zero};
    const FpExt xs[3] = {FpExt(Fp(1u)), FpExt(Fp(2u)), FpExt(Fp(3u))};

    for (size_t idx = threadIdx.x; idx < half; idx += blockDim.x) {
        size_t even = idx << 1;
        FpExt eq_even = eq_xi[even];
        FpExt eq_odd = eq_xi[even + 1];
        FpExt eq_diff = eq_odd - eq_even;

        FpExt p0_even = pq[even];
        FpExt p0_odd = pq[even + 1];
        FpExt p0_diff = p0_odd - p0_even;

        FpExt q0_even = pq[stride + even];
        FpExt q0_odd = pq[stride + even + 1];
        FpExt q0_diff = q0_odd - q0_even;

        FpExt p1_even = pq[(stride << 1) + even];
        FpExt p1_odd = pq[(stride << 1) + even + 1];
        FpExt p1_diff = p1_odd - p1_even;

        FpExt q1_even = pq[(stride * 3) + even];
        FpExt q1_odd = pq[(stride * 3) + even + 1];
        FpExt q1_diff = q1_odd - q1_even;

        for (int i = 0; i < 3; ++i) {
            FpExt eq_val = eq_even + xs[i] * eq_diff;
            FpExt p_j0 = p0_even + xs[i] * p0_diff;
            FpExt q_j0 = q0_even + xs[i] * q0_diff;
            FpExt p_j1 = p1_even + xs[i] * p1_diff;
            FpExt q_j1 = q1_even + xs[i] * q1_diff;

            FpExt p_prev = p_j0 * q_j1 + p_j1 * q_j0;
            FpExt q_prev = q_j0 * q_j1;
            local[i] = local[i] + eq_val * (p_prev + lambda * q_prev);
        }
    }

    for (int i = 0; i < 3; ++i) {
        FpExt reduced = sumcheck::block_reduce_sum(local[i], shared);
        if (threadIdx.x == 0) {
            out[i] = reduced;
        }
        __syncthreads();
    }
}

__global__ void fold_columns_kernel(
    const FpExt *input,
    size_t in_stride,
    size_t width,
    FpExt r,
    FpExt *output
) {
    size_t out_stride = in_stride >> 1;
    size_t total = out_stride * width;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    size_t row = idx % out_stride;
    size_t col = idx / out_stride;

    size_t in_offset = col * in_stride + (row << 1);
    size_t out_offset = col * out_stride + row;

    FpExt t0 = input[in_offset];
    FpExt t1 = input[in_offset + 1];
    output[out_offset] = t0 + r * (t1 - t0);
}

__global__ void extract_claims_kernel(const FpExt *data, size_t stride, FpExt *out) {
    if (threadIdx.x != 0) {
        return;
    }

    out[0] = data[0];
    out[1] = data[stride];
    out[2] = data[stride * 2];
    out[3] = data[stride * 3];
}

__global__ void add_alpha_kernel(FracExt *data, size_t len, FpExt alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        data[idx].q = data[idx].q + alpha;
    }
}

template <typename F, typename EF>
__global__ void frac_vector_scalar_multiply_kernel(
    std::pair<EF, EF> *frac_vec,
    F scalar,
    uint32_t length
) {
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= length)
        return;

    frac_vec[tidx].first *= scalar;
}

// ============================================================================
// LAUNCHERS
// ============================================================================
extern "C" int _frac_build_segment_tree(FracExt *tree, size_t total_leaves) {
    if (total_leaves == 0) {
        return 0;
    }

    size_t level_start = total_leaves;
    size_t nodes_in_level = total_leaves;

    while (nodes_in_level > 1) {
        level_start >>= 1;
        nodes_in_level >>= 1;

        auto [grid, block] = kernel_launch_params(nodes_in_level);
        build_level_kernel<<<grid, block>>>(tree, level_start, nodes_in_level);
        int err = CHECK_KERNEL();
        if (err != 0) {
            return err;
        }
    }

    return 0;
}

extern "C" int _frac_prepare_round(
    const FracExt *tree,
    size_t segment_start,
    size_t eval_size,
    FpExt *pq_out
) {
    if (eval_size == 0) {
        return 0;
    }

    auto [grid, block] = kernel_launch_params(eval_size);
    prepare_round_kernel<<<grid, block>>>(tree, segment_start, eval_size, pq_out);
    return CHECK_KERNEL();
}

extern "C" int _frac_compute_round(
    const FpExt *eq_xi,
    const FpExt *pq,
    size_t stride,
    FpExt lambda,
    FpExt *out_device
) {
    size_t half = stride >> 1;
    if (half == 0) {
        cudaError_t err = cudaMemsetAsync(out_device, 0, 3 * sizeof(FpExt), cudaStreamPerThread);
        return err == cudaSuccess ? 0 : err;
    }

    size_t threads = std::min<size_t>(std::max<size_t>(half, WARP_SIZE), 512);
    auto [grid, block] = kernel_launch_params(threads, threads);
    size_t num_warps = std::max<size_t>(1, (block.x + WARP_SIZE - 1) / WARP_SIZE);
    size_t shmem_bytes = num_warps * sizeof(FpExt);
    compute_round_kernel<<<grid, block, shmem_bytes>>>(eq_xi, pq, stride, lambda, out_device);
    return CHECK_KERNEL();
}

extern "C" int _frac_fold_columns(
    const FpExt *input,
    size_t in_stride,
    size_t width,
    FpExt r,
    FpExt *output
) {
    if (in_stride <= 1) {
        return 0;
    }
    size_t half = in_stride >> 1;
    auto [grid, block] = kernel_launch_params(half * width);
    fold_columns_kernel<<<grid, block>>>(input, in_stride, width, r, output);
    return CHECK_KERNEL();
}

extern "C" int _frac_extract_claims(const FpExt *data, size_t stride, FpExt *out_device) {
    extract_claims_kernel<<<1, 32>>>(data, stride, out_device);
    return CHECK_KERNEL();
}

extern "C" int _frac_add_alpha(FracExt *data, size_t len, FpExt alpha) {
    auto [grid, block] = kernel_launch_params(len);
    add_alpha_kernel<<<grid, block>>>(data, len, alpha);
    return CHECK_KERNEL();
}

extern "C" int _frac_vector_scalar_multiply_ext_fp(FracExt *frac_vec, Fp scalar, uint32_t length) {
    auto [grid, block] = kernel_launch_params(length);
    frac_vector_scalar_multiply_kernel<Fp, FpExt>
        <<<grid, block>>>(reinterpret_cast<std::pair<FpExt, FpExt> *>(frac_vec), scalar, length);
    return CHECK_KERNEL();
}
} // namespace fractional_sumcheck_gkr
