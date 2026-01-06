/*
 * Batch MLE evaluation kernel for low num_y traces
 *
 * This kernel evaluates constraint polynomials by parallelizing over monomials
 * rather than (x_int, y_int) pairs. This is more efficient when num_y is small
 * because it better utilizes GPU parallelism.
 *
 * The polynomial is represented as:
 *   C(x) = Σ_m coeff_m(λ) * ∏_i var_i(x)
 * where coeff_m(λ) is a sparse polynomial in the batching variable λ.
 */

#include "monomial.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"

// Block size for monomial evaluation
#define BLOCK_SIZE 256

// Shared memory reduction for FpExt
__device__ __forceinline__ FpExt block_reduce_sum(FpExt val, FpExt* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = shared[tid] + shared[tid + s];
        }
        __syncthreads();
    }
    return shared[0];
}

// Main kernel: evaluate monomials and accumulate results
// Grid: (num_blocks_x, num_x, num_y) where num_blocks_x = ceil(num_monomials / BLOCK_SIZE)
// Each block handles a chunk of monomials for one (x_int, y_int) pair
__global__ void batch_mle_low_kernel(
    // Output: partial sums per block
    FpExt* __restrict__ d_tmp_sums,           // [num_blocks_x * num_x * num_y]
    // Monomial data
    const uint8_t* __restrict__ d_data,       // Serialized monomial data
    const uint32_t* __restrict__ d_offsets,   // Byte offset of each monomial
    uint32_t num_monomials,
    // Lambda powers for coefficient evaluation
    const FpExt* __restrict__ d_lambda_pows,  // [num_constraints]
    // Matrix evaluations for variable evaluation (local row)
    const FpExt* __restrict__ d_mat_evals_local,
    // Matrix evaluations for variable evaluation (next row)
    const FpExt* __restrict__ d_mat_evals_next,
    const uint32_t* __restrict__ d_mat_widths,
    const uint32_t* __restrict__ d_mat_offsets,
    // Selector evaluations
    const FpExt* __restrict__ d_sels,
    // Dimensions
    uint32_t num_x,
    uint32_t num_y,
    uint32_t num_blocks_x
) {
    extern __shared__ char smem[];
    FpExt* shared = (FpExt*)smem;

    uint32_t x_int = blockIdx.y;
    uint32_t y_int = blockIdx.z;
    uint32_t xy_idx = x_int * num_y + y_int;

    FpExt sum = FpExt(0);

    // Each thread processes multiple monomials (grid-stride loop)
    for (uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
         m < num_monomials;
         m += gridDim.x * blockDim.x) {

        // Parse monomial header
        uint8_t num_lambda_terms, num_vars;
        const LambdaTerm* lambda_terms;
        const uint64_t* variables;
        parse_monomial(d_data + d_offsets[m], num_lambda_terms, num_vars, lambda_terms, variables);

        // Evaluate F[λ] coefficient
        FpExt coeff = eval_lambda_coeff(lambda_terms, num_lambda_terms, d_lambda_pows);

        // Evaluate product of variables
        FpExt var_product = FpExt(Fp(1));
        for (uint8_t i = 0; i < num_vars; ++i) {
            SourceInfo src = decode_source(variables[i]);

            FpExt var_val;
            bool use_next = (src.offset == 1);
            const FpExt* mat_evals = use_next ? d_mat_evals_next : d_mat_evals_local;

            switch (src.type) {
            case ENTRY_PREPROCESSED: {
                uint32_t col = src.index;
                uint32_t base = d_mat_offsets[0];
                var_val = mat_evals[base + col * (num_x * num_y) + xy_idx];
                break;
            }
            case ENTRY_MAIN: {
                uint32_t part = src.part;
                uint32_t col = src.index;
                uint32_t base = d_mat_offsets[part + 1]; // +1 to skip preprocessed
                var_val = mat_evals[base + col * (num_x * num_y) + xy_idx];
                break;
            }
            case SRC_IS_FIRST:
                var_val = d_sels[0 * (num_x * num_y) + xy_idx];
                break;
            case SRC_IS_LAST:
                var_val = d_sels[1 * (num_x * num_y) + xy_idx];
                break;
            case SRC_IS_TRANSITION:
                var_val = d_sels[2 * (num_x * num_y) + xy_idx];
                break;
            default:
                var_val = FpExt(0);
                break;
            }

            var_product = var_product * var_val;
        }

        sum = sum + coeff * var_product;
    }

    // Block reduction
    FpExt reduced = block_reduce_sum(sum, shared);

    if (threadIdx.x == 0) {
        uint32_t num_xy = num_x * num_y;
        d_tmp_sums[blockIdx.x * num_xy + xy_idx] = reduced;
    }
}

// Reduction kernel: sum partial results from multiple blocks
__global__ void batch_mle_low_reduce_kernel(
    FpExt* __restrict__ d_output,             // [num_x * num_y]
    const FpExt* __restrict__ d_tmp_sums,     // [num_blocks_x * num_x * num_y]
    const FpExt* __restrict__ d_eq_xi,        // [num_x * num_y] eq polynomial evaluation
    uint32_t num_blocks_x,
    uint32_t num_x,
    uint32_t num_y
) {
    uint32_t xy_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (xy_idx >= num_x * num_y) return;

    uint32_t num_xy = num_x * num_y;
    FpExt sum = FpExt(0);
    for (uint32_t i = 0; i < num_blocks_x; ++i) {
        sum = sum + d_tmp_sums[i * num_xy + xy_idx];
    }

    // Multiply by eq_xi to get final contribution
    d_output[xy_idx] = sum * d_eq_xi[xy_idx];
}

// C interface for launching kernels
extern "C" {

cudaError_t launch_batch_mle_low(
    FpExt* d_output,
    FpExt* d_tmp_sums,
    const uint8_t* d_data,
    const uint32_t* d_offsets,
    uint32_t num_monomials,
    const FpExt* d_lambda_pows,
    const FpExt* d_mat_evals_local,
    const FpExt* d_mat_evals_next,
    const uint32_t* d_mat_widths,
    const uint32_t* d_mat_offsets,
    const FpExt* d_sels,
    const FpExt* d_eq_xi,
    uint32_t num_x,
    uint32_t num_y,
    cudaStream_t stream
) {
    uint32_t num_blocks_x = (num_monomials + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch main kernel
    dim3 grid(num_blocks_x, num_x, num_y);
    dim3 block(BLOCK_SIZE);
    size_t shared_mem = BLOCK_SIZE * sizeof(FpExt);

    batch_mle_low_kernel<<<grid, block, shared_mem, stream>>>(
        d_tmp_sums,
        d_data,
        d_offsets,
        num_monomials,
        d_lambda_pows,
        d_mat_evals_local,
        d_mat_evals_next,
        d_mat_widths,
        d_mat_offsets,
        d_sels,
        num_x,
        num_y,
        num_blocks_x
    );

    // Launch reduction kernel
    uint32_t num_xy = num_x * num_y;
    uint32_t reduce_blocks = (num_xy + 255) / 256;
    batch_mle_low_reduce_kernel<<<reduce_blocks, 256, 0, stream>>>(
        d_output,
        d_tmp_sums,
        d_eq_xi,
        num_blocks_x,
        num_x,
        num_y
    );

    return cudaGetLastError();
}

} // extern "C"
