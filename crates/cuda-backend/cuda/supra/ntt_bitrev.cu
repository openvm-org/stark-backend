/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ntt/kernels.cu
 * Imported: 2025-08-13 by @gaxiom
 * 
 * LOCAL CHANGES (high level):
 * - 2025-08-13: Support multiple rows in bit_rev_permutation & bit_rev_permutation_z
 * - 2025-09-10: Add extern "C" launcher from sppark/ntt/ntt.cuh
 */

#include "launcher.cuh"
#include "ntt/ntt.cuh"

constexpr uint32_t LOW_BITREV_BITS = 5;
constexpr uint32_t HIGH_BITREV_BITS = 5;

/// Split the binary representation of each number in the following way:
/// [A = HIGH_BITREV_BITS bits][B = the remaining middle bits][C = LOW_BITREV_BITS bits]
/// This function, given `idx` and `log_n`, returns the `idx`-th number
/// among all numbers from `0` to `2^{log_n} - 1`, sorted lexicographically by (B, A, C).
/// This should go well with the SM's L1 cache.
__device__ __forceinline__ uint32_t choose_index_to_bitrev(uint32_t idx, uint32_t log_n) {
    return 
        (idx & ((1u << LOW_BITREV_BITS) - 1))
        | ((idx >> (log_n - HIGH_BITREV_BITS)) << LOW_BITREV_BITS)
        | ((idx & ((1u << (log_n - HIGH_BITREV_BITS)) - (1u << LOW_BITREV_BITS))) << (log_n - HIGH_BITREV_BITS - LOW_BITREV_BITS));
}

// Permutes the data in an array such that data[i] = data[bit_reverse(i)]
// and data[bit_reverse(i)] = data[i]
__launch_bounds__(1024) __global__
void bit_rev_permutation(fr_t* d_out, const fr_t *d_in, uint32_t lg_domain_size, uint32_t padded_poly_size)
{
    d_out += blockIdx.y * padded_poly_size; // [DIFF]: move out ptr to another row
    d_in += blockIdx.y * padded_poly_size;  // [DIFF]: move in ptr to another row

    if (gridDim.x == 1 && blockDim.x == (1 << lg_domain_size)) {
        uint32_t idx = threadIdx.x;
        uint32_t rev = bit_rev(idx, lg_domain_size);

        fr_t t = d_in[idx];
        if (d_out == d_in)
            __syncthreads();
        d_out[rev] = t;
    } else {
        index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
        idx = (lg_domain_size > LOW_BITREV_BITS + HIGH_BITREV_BITS)
            ? choose_index_to_bitrev(idx, lg_domain_size)
            : idx;
        index_t rev = bit_rev(idx, lg_domain_size);
        bool copy = d_out != d_in && idx == rev;

        if (idx < rev || copy) {
            fr_t t0 = d_in[idx];
            if (!copy) {
                fr_t t1 = d_in[rev];
                d_out[idx] = t1;
            }
            d_out[rev] = t0;
        }
    }
}

template<unsigned int Z_COUNT>
__launch_bounds__(192, 2) __global__
void bit_rev_permutation_z(fr_t* out, const fr_t* in, uint32_t lg_domain_size, uint32_t padded_poly_size)
{
    out += blockIdx.y * padded_poly_size;   // [DIFF]: move out ptr to another row
    in += blockIdx.y * padded_poly_size;    // [DIFF]: move in ptr to another row

    const uint32_t LG_Z_COUNT = 31 - __clz(Z_COUNT); // [DIFF]: use __clz to get lg2

    extern __shared__ fr_t xchg[][Z_COUNT][Z_COUNT];

    uint32_t gid = threadIdx.x / Z_COUNT;
    uint32_t idx = threadIdx.x % Z_COUNT;
    uint32_t rev = bit_rev(idx, LG_Z_COUNT);

    index_t step = (index_t)1 << (lg_domain_size - LG_Z_COUNT);
    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    #pragma unroll 1
    do {
        index_t group_idx = tid >> LG_Z_COUNT;
        index_t group_rev = bit_rev(group_idx, lg_domain_size - 2*LG_Z_COUNT);

        if (group_idx > group_rev)
            continue;

        index_t base_idx = group_idx * Z_COUNT + idx;
        index_t base_rev = group_rev * Z_COUNT + idx;

        fr_t regs[Z_COUNT];

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++) {
            xchg[gid][i][rev] = (regs[i] = in[i * step + base_idx]);
            if (group_idx != group_rev)
                regs[i] = in[i * step + base_rev];
        }

        (Z_COUNT > WARP_SIZE) ? __syncthreads() : __syncwarp();

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            out[i * step + base_rev] = xchg[gid][rev][i];

        if (group_idx == group_rev)
            continue;

        (Z_COUNT > WARP_SIZE) ? __syncthreads() : __syncwarp();

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            xchg[gid][i][rev] = regs[i];

        (Z_COUNT > WARP_SIZE) ? __syncthreads() : __syncwarp();

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            out[i * step + base_idx] = xchg[gid][rev][i];

    } while (Z_COUNT <= WARP_SIZE && (tid += blockDim.x*gridDim.x) < step);
    // without "Z_COUNT <= WARP_SIZE" compiler spills 128 bytes to stack
}


extern "C" int _bit_rev(fr_t* d_out, const fr_t* d_inp, 
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

    size_t domain_size = (size_t)1 << lg_domain_size;
    // aim to read 4 cache lines of consecutive data per read
    const uint32_t Z_COUNT = 256 / sizeof(fr_t);
    const uint32_t bsize = Z_COUNT > WARP_SIZE ? Z_COUNT : WARP_SIZE;

    // [DIFF]: N -> dim3(N, poly_count) in grid_size; stream -> cudaStreamPerThread
    if (domain_size <= 1024)
        bit_rev_permutation<<<dim3(1, poly_count), domain_size>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size);
    else if (domain_size < bsize * Z_COUNT)
        bit_rev_permutation<<<dim3(domain_size / WARP_SIZE, poly_count), WARP_SIZE>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size);
    else if (Z_COUNT > WARP_SIZE || lg_domain_size <= 32)
        bit_rev_permutation_z<Z_COUNT><<<dim3(domain_size / Z_COUNT / bsize, poly_count), bsize,
                                            bsize * Z_COUNT * sizeof(fr_t)>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size);
    else {
        // Those GPUs that can reserve 96KB of shared memory can
        // schedule 2 blocks to each SM...
        int device;
        cudaGetDevice(&device);
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

        bit_rev_permutation_z<Z_COUNT><<<dim3(sm_count * 2, poly_count), 192,
                                            192 * Z_COUNT * sizeof(fr_t)>>>
                                (d_out, d_inp, lg_domain_size, padded_poly_size);
    }

    return CHECK_KERNEL();
}