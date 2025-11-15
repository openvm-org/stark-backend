#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "sumcheck.cuh"
#include <algorithm>

namespace logup_zerocheck {

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


// Folds PLE evaluations by interpolating univariate polynomials on coset D and evaluating at r_0
// Input: column-major matrix [height * width] of evaluations
// Output: two column-major matrices [new_height * width] of folded evaluations (original and rotated)
// For each (x, col), collects 2^l_skip evaluations on coset D and interpolates for both offsets
template<bool ROTATE>
__global__ void fold_ple_from_evals_kernel(
    const Fp* input_matrix,      // [height * width] column-major
    FpExt* output_matrix,         // [new_height * output_width] column-major
                                   // If ROTATE: output_width = width * 2, layout: [orig_cols, rot_cols]
                                   // If !ROTATE: output_width = width
    const FpExt* numerators,    // [domain_size] Π_{j≠i} (r_0 - omega^j)
    const FpExt* inv_lagrange_denoms,  // [domain_size] 1/(Π_{j≠i} (omega^i - omega^j))
    uint32_t height,
    uint32_t width,
    uint32_t domain_size,        // 2^l_skip
    uint32_t l_skip,            // log2(domain_size)
    uint32_t new_height        // lifted_height >> l_skip
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = new_height * width;
    if (idx >= total_outputs) return;
    
    int x = idx % new_height;
    int col = idx / new_height;
    
    // Compute offset 0 (always needed)
    FpExt result_0(Fp::zero());
    
    for (int z = 0; z < domain_size; z++) {
        // Offset 0: ((x << l_skip) + z + 0) % height
        int row_idx_0 = ((x << l_skip) + z) % height;
        int input_idx_0 = col * height + row_idx_0;
        Fp eval_0 = input_matrix[input_idx_0];
        // Lagrange interpolation: eval * numerators[z] * inv_lagrange_denoms[z]
        result_0 = result_0 + FpExt(eval_0) * numerators[z] * inv_lagrange_denoms[z];
    }
    
    // Write original columns: output_matrix[col * new_height + x]
    int output_idx_original = col * new_height + x;
    output_matrix[output_idx_original] = result_0;
    
    // Compute offset 1 only if rotate=true
    if constexpr (ROTATE) {
        FpExt result_1(Fp::zero());
        for (int z = 0; z < domain_size; z++) {
            // Offset 1: ((x << l_skip) + z + 1) % height
            int row_idx_1 = ((x << l_skip) + z + 1) % height;
            int input_idx_1 = col * height + row_idx_1;
            Fp eval_1 = input_matrix[input_idx_1];
            result_1 = result_1 + FpExt(eval_1) * numerators[z] * inv_lagrange_denoms[z];
        }
        // Write rotated columns: output_matrix[(width + col) * new_height + x]
        // Layout: [orig_col0...orig_col{width-1}, rot_col0...rot_col{width-1}]
        int output_idx_rotated = (width + col) * new_height + x;
        output_matrix[output_idx_rotated] = result_1;
    }
}

// Combined kernel: mutates eq_xi in-place (eq_xi *= eq_r0) and computes eq_sharp (original_eq_xi * eq_sharp_r0)
// Note: eq_sharp uses the ORIGINAL eq_xi value, not the multiplied one (matches CPU behavior)
__global__ void compute_eq_sharp_kernel(
    FpExt* eq_xi,              // [count] input/output: mutated in-place to eq_xi * eq_r0
    FpExt* eq_sharp,           // [count] output: original_eq_xi * eq_sharp_r0
    FpExt eq_r0,               // scalar
    FpExt eq_sharp_r0,        // scalar
    uint32_t count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    FpExt original_eq_xi = eq_xi[idx];
    
    // Mutate in-place: eq_xi *= eq_r0
    eq_xi[idx] = original_eq_xi * eq_r0;
    
    // Compute eq_sharp using the ORIGINAL eq_xi value 
    eq_sharp[idx] = original_eq_xi * eq_sharp_r0;
}

__global__ void interpolate_columns_kernel(
    FpExt *interpolated,
    const uintptr_t *columns,
    uint32_t s_deg,
    uint32_t num_y,
    uint32_t num_columns
) {
    int y = threadIdx.x + blockIdx.x * blockDim.x;
    if (y >= num_y) return;

    int col_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (col_idx >= num_columns) return;

    auto column = reinterpret_cast<const FpExt*>(columns[col_idx]);
    auto t0 = column[y << 1];
    auto t1 = column[(y << 1) | 1];
    auto this_interpolated = interpolated + col_idx * s_deg * num_y;

    for (int x = 0; x < s_deg; x++) {
        this_interpolated[y * s_deg + x] = t0 + (t1 - t0) * FpExt(Fp(x + 1u));
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _interpolate_columns(
    FpExt *interpolated,
    const uintptr_t *columns,
    size_t s_deg,
    size_t num_y,
    size_t num_columns
) {
    auto [grid, block] = kernel_launch_2d_params(num_y, num_columns);

    interpolate_columns_kernel<<<grid, block>>>(
        interpolated,
        columns,
        s_deg,
        num_y,
        num_columns
    );
    return CHECK_KERNEL();
}

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

    size_t threads = std::min<size_t>(std::max<size_t>(half, WARP_SIZE), MAX_THREADS);
    auto [grid, block] = kernel_launch_params(threads, threads);
    size_t num_warps = std::max<size_t>(1, (block.x + WARP_SIZE - 1) / WARP_SIZE);
    size_t shmem_bytes = num_warps * sizeof(FpExt);
    compute_round_kernel<<<grid, block, shmem_bytes>>>(
        eq_xi,
        pq,
        stride,
        lambda,
        out_device
    );
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

extern "C" int _fold_ple_from_evals(
    const Fp* input_matrix,
    FpExt* output_matrix,  // Single buffer: [orig_cols, rot_cols] when rotate=true
    const FpExt* numerators,
    const FpExt* inv_lagrange_denoms,
    uint32_t height,
    uint32_t width,
    uint32_t domain_size,
    uint32_t l_skip,
    uint32_t new_height,
    bool rotate
) {
    int total_outputs = new_height * width;
    auto [grid, block] = kernel_launch_params(total_outputs);
    
    if (rotate) {
        fold_ple_from_evals_kernel<true><<<grid, block>>>(
            input_matrix,
            output_matrix,
            numerators,
            inv_lagrange_denoms,
            height,
            width,
            domain_size,
            l_skip,
            new_height
        );
    } else {
        fold_ple_from_evals_kernel<false><<<grid, block>>>(
            input_matrix,
            output_matrix,
            numerators,
            inv_lagrange_denoms,
            height,
            width,
            domain_size,
            l_skip,
            new_height
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _compute_eq_sharp(
    FpExt* eq_xi,
    FpExt* eq_sharp,
    const FpExt eq_r0,
    const FpExt eq_sharp_r0,
    uint32_t count
) {
    if (count == 0) return 0;
    auto [grid, block] = kernel_launch_params(count);
    compute_eq_sharp_kernel<<<grid, block>>>(
        eq_xi,
        eq_sharp,
        eq_r0,
        eq_sharp_r0,
        count
    );
    return CHECK_KERNEL();
}

} // namespace logup_zerocheck

