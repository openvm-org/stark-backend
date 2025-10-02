/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ntt/parameters.cuh
 * Imported: 2025-08-13 by @gaxiom
 * 
 * LOCAL CHANGES (high level):
 * - 2025-08-13: NTTParameters constructor async on custom stream
 * - 2025-08-26: NTTParameters constructor on cudaStreamPerThread
 * - 2025-09-05: Stop using __constant__ for twiddles[0]
 * - 2025-09-10: Delete NTTParameters & add extern "C" launcher
 * - 2025-10-02: move all twiddles to __constant__
 */

#include "launcher.cuh"
#include "ntt/parameters.cuh"

__constant__ fr_t FORWARD_TWIDDLES[TWIDDLES_SIZE];
__constant__ fr_t INVERSE_TWIDDLES[TWIDDLES_SIZE];
__constant__ fr_t FORWARD_PARTIAL_TWIDDLES[WINDOW_NUM][WINDOW_SIZE];
__constant__ fr_t INVERSE_PARTIAL_TWIDDLES[WINDOW_NUM][WINDOW_SIZE];

__global__ void generate_all_twiddles(fr_t* d_radixX_twiddles, 
    const fr_t root6, const fr_t root7, const fr_t root8, const fr_t root9, const fr_t root10)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int pow = 0;
    fr_t root_of_unity;

    if (tid < 32) {
        pow = tid;
        root_of_unity = root6;
    } else if (tid < 32 + 64) {
        pow = tid - 32;
        root_of_unity = root7;
    } else if (tid < 32 + 64 + 128) {
        pow = tid - 32 - 64;
        root_of_unity = root8;
    } else if (tid < 32 + 64 + 128 + 256) {
        pow = tid - 32 - 64 - 128;
        root_of_unity = root9;
    } else if (tid < 32 + 64 + 128 + 256 + 512) {
        pow = tid - 32 - 64 - 128 - 256;
        root_of_unity = root10;
    } else {
        assert(false);
    }

    d_radixX_twiddles[tid] = root_of_unity^pow;
}

__global__ void generate_partial_twiddles(fr_t (*roots)[WINDOW_SIZE],
                               const fr_t root_of_unity)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    assert(tid < WINDOW_SIZE);
    fr_t root;

    root = root_of_unity^tid;

    roots[0][tid] = root;

    for (int off = 1; off < WINDOW_NUM; off++) {
        for (int i = 0; i < LG_WINDOW_SIZE; i++)
            root.sqr();
        roots[off][tid] = root;
    }
}

extern "C" int _generate_all_twiddles(fr_t* twiddles, bool inverse) {
    const fr_t* roots = inverse ? inverse_roots_of_unity : forward_roots_of_unity;

    generate_all_twiddles<<<TWIDDLES_SIZE/32, 32>>>(
            twiddles, roots[6], roots[7], roots[8], roots[9], roots[10]);

    if (inverse) {
        cudaMemcpyToSymbolAsync(INVERSE_TWIDDLES, twiddles, TWIDDLES_SIZE * sizeof(fr_t),
                                0, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    } else {
        cudaMemcpyToSymbolAsync(FORWARD_TWIDDLES, twiddles, TWIDDLES_SIZE * sizeof(fr_t),
                                0, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    }
    cudaStreamSynchronize(cudaStreamPerThread);
    return CHECK_KERNEL();
}

extern "C" int _generate_partial_twiddles(fr_t (*partial_twiddles)[WINDOW_SIZE], bool inverse) {
    const fr_t* roots = inverse ? inverse_roots_of_unity : forward_roots_of_unity;
    generate_partial_twiddles<<<WINDOW_SIZE/32, 32>>>(
            partial_twiddles, roots[MAX_LG_DOMAIN_SIZE]);

    if (inverse) {
        cudaMemcpyToSymbolAsync(INVERSE_PARTIAL_TWIDDLES, partial_twiddles, WINDOW_NUM * WINDOW_SIZE * sizeof(fr_t),
                                0, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    } else {
        cudaMemcpyToSymbolAsync(FORWARD_PARTIAL_TWIDDLES, partial_twiddles, WINDOW_NUM * WINDOW_SIZE * sizeof(fr_t),
                                0, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    }
    cudaStreamSynchronize(cudaStreamPerThread);
    return CHECK_KERNEL();
}