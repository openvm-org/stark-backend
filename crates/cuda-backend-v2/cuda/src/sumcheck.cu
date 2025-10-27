#include "fpext.h"
#include "launcher.cuh"
#include <vector>
#include <iostream>

// Warp-level reduction for FpExt
__device__ FpExt warp_reduce_sum(FpExt val) {
    unsigned mask = __activemask();

    for (int offset = 16; offset > 0; offset /= 2) {
        FpExt other;
        other.elems[0] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[0].asRaw(), offset));
        other.elems[1] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[1].asRaw(), offset));
        other.elems[2] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[2].asRaw(), offset));
        other.elems[3] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[3].asRaw(), offset));
        val = val + other;
    }
    return val;
}

// Block-level reduction for FpExt
__device__ FpExt block_reduce_sum(FpExt val, FpExt* shared) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (blockDim.x + 31) / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    FpExt zero = {0, 0, 0, 0};
    if (warp_id == 0) {
        // Only the first warp participates in the second reduction. Within that warp we
        // reuse the lane id to index the number of warps stored in shared memory.
        FpExt warp_val = (lane_id < num_warps) ? shared[lane_id] : zero;
        val = warp_reduce_sum(warp_val);
    }
    
    return val;
}


// NOTE: This kernel currently implements identity W function: W(evals) = evals[0][0]
// For more complex use cases with custom W functions, the main modification would be
// in the accumulation section where we compute `local_sums`. Instead of directly
// accumulating eval_X, we would:
// 1. Collect all matrix evaluations at (X, y) into a buffer
// 2. Call W function to combine them: W(X, y, all_evals) -> [FpExt; WD]
// 3. Accumulate the result into local_sums
// The map-reduce structure and reduction logic would remain unchanged.
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
            FpExt reduced = block_reduce_sum(local_sums[idx], shared);
            
            if (threadIdx.x == 0) {
                block_sums[blockIdx.x * d * WD + idx] = reduced;
            }
            __syncthreads();  // Needed before reusing shared memory
        }
    }
}

// Final reduction kernel: combines block sums into final result
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
    
    // Sum all num_blocks partial sums for this output index
    for (int block_id = tid; block_id < num_blocks; block_id += blockDim.x) {
        sum += block_sums[block_id * d * WD + out_idx];
    }
    
    // Block-level reduction
    sum = block_reduce_sum(sum, shared);
    
    if (tid == 0) {
        output[out_idx] = sum;
    }
}

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