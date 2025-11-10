#include "fpext.h"
#include "launcher.cuh"
#include "sumcheck.cuh"

namespace plain_sumcheck {

// Reduces evaluations over x and column dimensions for PLE round 0
// Input:  [num_x * num_cols * large_domain_size] - evaluations on large domain
// Output: [large_domain_size] - summed evaluations
// Each thread handles one z value, summing over all (x, col) pairs
// Memory: Column-major layout where each (x,col) is a column of size large_domain_size
__global__ void reduce_over_x_and_cols_kernel(
    const Fp* input,         // [num_x * num_cols * large_domain_size]
    Fp* output,              // [large_domain_size]
    uint32_t num_x,
    uint32_t num_cols,
    uint32_t large_domain_size
) {
    // Each thread handles one z value
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= large_domain_size) return;
    
    Fp sum = Fp::zero();
    
    // Sum input[(x * num_cols + col) * large_domain_size + z] over all (x, col)
    for (uint32_t x = 0; x < num_x; x++) {
        for (uint32_t col = 0; col < num_cols; col++) {
            // Column-major layout: [x * num_cols + col][z]
            uint32_t offset = (x * num_cols + col) * large_domain_size;
            sum = sum + input[offset + z];
        }
    }
    
    output[z] = sum;
}

// Computes univariate polynomial evaluations for MLE sumcheck round
// For each X ∈ {1, ..., d}, computes: s(X) = Σ_{y ∈ H_{n-1}} f̂(X, y)
// where f̂(X, y) is obtained by linear interpolation: f(0,y) + X*(f(1,y) - f(0,y))
//
// Memory layout: Column-major matrices with height = 2^n
// - For each y: reads indices [2*y, 2*y+1] (even/odd pairs)
// - Outputs partial sums per block, final reduction done in separate kernel
//
// Template parameter WD: Number of output polynomials (typically 1)
// NOTE: This implements identity W function. For custom W, modify accumulation section.
template<int WD>
__global__ void sumcheck_mle_round_kernel(
    const uintptr_t* input_matrices,
    FpExt* block_sums,        // Output: [gridDim.x][d][WD] partial sums
    const uint32_t* widths,
    const uint32_t num_matrices,
    const uint32_t height,         // = 2^n
    const uint32_t d               // degree (runtime parameter)
) {
    extern __shared__ char smem[];
    FpExt* shared = (FpExt*)smem;
    
    int half_height = height >> 1;
    
    // Local accumulators for all (d, WD) pairs
    constexpr int MAX_D = 5;   // [TODO] Mark this in doc that it's max degree 4 + 1
    FpExt local_sums[WD * MAX_D];
    
    // Initialize accumulators
    for (int i = 0; i < d * WD; i++) {
        local_sums[i] = {0, 0, 0, 0};
    }
    
    // Map phase: each thread processes multiple y values
    for (int y = blockIdx.x * blockDim.x + threadIdx.x; 
         y < half_height; 
         y += gridDim.x * blockDim.x) {
        
        // For each evaluation point X in {1, 2, ..., d}
        for (uint32_t x_int = 1; x_int <= d; x_int++) {
            FpExt X = FpExt(Fp(x_int));
            
            // For identity W: we simply sum all matrix column evaluations
            // For each matrix
            for (int mat_idx = 0; mat_idx < num_matrices; mat_idx++) {
                const FpExt* input = reinterpret_cast<const FpExt*>(input_matrices[mat_idx]);
                int width = widths[mat_idx];
                
                // For each column
                for (int col = 0; col < width; col++) {
                    int col_offset = col * height;
                    int idx_0 = col_offset + (y << 1);
                    int idx_1 = col_offset + (y << 1) + 1;
                    
                    FpExt eval_0 = input[idx_0];
                    FpExt eval_1 = input[idx_1];
                    
                    FpExt eval_X = eval_0 + X * (eval_1 - eval_0);

                    // For identity W and WD=1: accumulate directly
                    // TODO: When implementing custom W, replace this section
                    local_sums[(x_int - 1) * WD + 0] += eval_X;
                }
            }
        }
    }
    
    // Reduce phase: for each (x_int, wd) pair
    for (int x_int = 0; x_int < d; x_int++) {
        for (int wd = 0; wd < WD; wd++) {
            int idx = x_int * WD + wd;
            FpExt reduced = sumcheck::block_reduce_sum(local_sums[idx], shared);
            
            if (threadIdx.x == 0) {
                block_sums[blockIdx.x * d * WD + idx] = reduced;
            }
            __syncthreads();  // Needed before reusing shared memory
        }
    }
}

// Final reduction: combines partial block sums into final result
// Grid dimension: d * WD blocks (one per output value)
// Each block reduces num_blocks partial sums for its assigned output
template<int WD>
__global__ void reduce_blocks_sumcheck(
    const FpExt* block_sums,  // [num_blocks][d * WD]
    FpExt* output,             // [d * WD]
    uint32_t num_blocks,
    uint32_t d
) {
    extern __shared__ char smem[];
    FpExt* shared = (FpExt*)smem;
    
    int tid = threadIdx.x;
    
    // blockIdx.x selects which of the (d * WD) outputs we're computing
    int out_idx = blockIdx.x;
    if (out_idx >= d * WD) return;
    
    FpExt sum = {0, 0, 0, 0};
    
    // Each thread accumulates subset of blocks
    for (int block_id = tid; block_id < num_blocks; block_id += blockDim.x) {
        sum += block_sums[block_id * d * WD + out_idx];
    }
    
    // Block-level reduction
    sum = sumcheck::block_reduce_sum(sum, shared);
    
    if (tid == 0) {
        output[out_idx] = sum;
    }
}

// Folds MLE evaluations using challenge r: output[y] = input[2*y] + r*(input[2*y+1] - input[2*y])
// Memory: Column-major layout, each column folded independently
// - Input:  height = 2^n per column
// - Output: height = 2^(n-1) per column
// Grid: (blocks for height/2, num_matrices) to parallelize over both dimensions
// [TODO] Check is it valid that all matrices have the same height?
__global__ void fold_mle_kernel(
    const uintptr_t* input_matrices,   // Array of input matrix pointers
    const uintptr_t* output_matrices,  // Array of output matrix pointers
    const uint32_t* widths,                 // Width of each matrix
    const uint32_t num_matrices,
    const uint32_t output_height,
    const FpExt r_val
) {
    int input_height = output_height << 1;

    // blockIdx.x and threadIdx.x parallelize over output_height
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= output_height) return;
    
    // blockIdx.y selects which matrix we're working on
    int mat_idx = blockIdx.y;
    if (mat_idx >= num_matrices) return;
    
    int width = widths[mat_idx];
    const FpExt* input = reinterpret_cast<const FpExt*>(input_matrices[mat_idx]);
    FpExt* output = reinterpret_cast<FpExt*>(output_matrices[mat_idx]);

    // Loop over all columns for this (matrix, y) pair
    for (int col = 0; col < width; col++) {
        int col_offset_in = col * input_height;
        int col_offset_out = col * output_height;
        
        int idx_0 = col_offset_in + (y << 1);
        int idx_1 = col_offset_in + (y << 1) + 1;
        int out_idx = col_offset_out + y;
        
        FpExt t0 = input[idx_0];
        FpExt t1 = input[idx_1];
        
        output[out_idx] = t0 + r_val * (t1 - t0);
    }
}

// Evaluates univariate polynomials at challenge point r using Horner's method
// Input: Polynomial coefficients (from iDFT in round 0) in natural order
// Output: Evaluations in extension field
//
// Memory layout: Column-major with domain_size coefficients per polynomial
// - input_coeffs[(x * width + col) * domain_size + i] = coefficient c_i for polynomial (x, col)
// - Evaluates: c_0 + c_1*r + c_2*r^2 + ... using Horner's method
//
// Why this works: During PLE round 0, we performed iDFT with bit_reverse=true,
// converting evaluations to coefficients in natural order. We reuse those
// coefficients here instead of re-interpolating from evaluations (more efficient).
__global__ void fold_ple_from_coeffs_kernel(
    const Fp* input_coeffs,    // [num_x * width * domain_size] after iDFT
    FpExt* output,             // [num_x * width]
    uint32_t num_x,
    uint32_t width,
    uint32_t domain_size,     // 2^l_skip
    FpExt r_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_polys = num_x * width;
    
    if (idx >= total_polys) return;
    
    // Coefficients for this (x, col) are at offset idx * domain_size
    const Fp* coeffs = input_coeffs + idx * domain_size;
    
    // Horner's method: evaluate polynomial at r
    FpExt result = FpExt(Fp(coeffs[domain_size - 1]));
    for (int i = domain_size - 2; i >= 0; i--) {
        result = result * r_val + FpExt(Fp(coeffs[i]));
    }
    
    output[idx] = result;
}

// ============================================================================
// LAUNCHERS
// ============================================================================


extern "C" int _fold_mle(
    const uintptr_t* input_matrices,
    const uintptr_t* output_matrices,
    const uint32_t* widths,
    const uint32_t num_matrices,
    const uint32_t output_height,
    const FpExt r_val
) {
    auto [rows_grid, rows_block] = kernel_launch_params(output_height);
    dim3 grid(rows_grid.x, num_matrices);
    fold_mle_kernel<<<grid, rows_block>>>(input_matrices, output_matrices, widths, num_matrices, output_height, r_val);
    return CHECK_KERNEL();
}

extern "C" int _fold_ple_from_coeffs(
    const Fp* input_coeffs,
    FpExt* output,
    const uint32_t num_x,
    const uint32_t width,
    const uint32_t domain_size,
    const FpExt r_val
) {
    int total_polys = num_x * width;
    auto [grid, block] = kernel_launch_params(total_polys);
    
    fold_ple_from_coeffs_kernel<<<grid, block>>>(
        input_coeffs,
        output,
        num_x,
        width,
        domain_size,
        r_val
    );
    
    return CHECK_KERNEL();
}

extern "C" int _reduce_over_x_and_cols(
    const Fp* input,
    Fp* output,
    uint32_t num_x,
    uint32_t num_cols,
    uint32_t large_domain_size
) {
    auto [grid, block] = kernel_launch_params(large_domain_size);
    reduce_over_x_and_cols_kernel<<<grid, block>>>(
        input,
        output,
        num_x,
        num_cols,
        large_domain_size
    );
    return CHECK_KERNEL();
}

// WD = 1
extern "C" int _sumcheck_mle_round(
    const uintptr_t* input_matrices,
    FpExt* output,           // Output: [d * WD] final results
    FpExt* tmp_block_sums,   // Temporary buffer: [num_blocks * d * WD]
    const uint32_t* widths,
    const uint32_t num_matrices,
    const uint32_t height,
    const uint32_t d
) {
    int half_height = height >> 1;
    auto [grid, block] = kernel_launch_params(half_height);
    unsigned int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t shmem_bytes = std::max(1u, num_warps) * sizeof(FpExt);
    
    // Launch main kernel - writes to tmp_block_sums
    sumcheck_mle_round_kernel<1><<<grid, block, shmem_bytes>>>(
        input_matrices,
        tmp_block_sums,
        widths,
        num_matrices,
        height,
        d
    );

    int err = CHECK_KERNEL();
    if (err != 0) return err;
    
    // Launch final reduction kernel - reads from tmp_block_sums, writes to output
    auto num_blocks = grid.x;
    auto [reduce_grid, reduce_block] = kernel_launch_params(num_blocks);
    unsigned int reduce_warps = (reduce_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t reduce_shmem = std::max(1u, reduce_warps) * sizeof(FpExt);
    reduce_blocks_sumcheck<1><<<d, reduce_block, reduce_shmem>>>(
        tmp_block_sums,
        output,
        num_blocks,
        d
    );

    return CHECK_KERNEL();
}

} // namespace plain_sumcheck