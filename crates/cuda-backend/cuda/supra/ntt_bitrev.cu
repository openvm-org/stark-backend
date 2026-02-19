/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ntt/kernels.cu
 * Imported: 2025-08-13 by @gaxiom
 *
 * LOCAL CHANGES (high level):
 * - 2025-08-13: Support multiple rows in bit_rev_permutation & bit_rev_permutation_z
 * - 2025-09-10: Add extern "C" launcher from sppark/ntt/ntt.cuh
 * - 2025-12-24: Template field type to support fr_t and bb31_4_t
 */

#include <cstdint>

#include "launcher.cuh"
#include "ntt/ntt.cuh"

// [DIFF]: Add new type for bit reversal kernel
struct frac_fpext_t {
    bb31_4_t num;
    bb31_4_t denom;
};

/*
 * Template type T requirements:
 * - Default constructible: T()
 * - Copy constructible: T t = other;
 * - Copy assignable: t1 = t2;
 * - Trivially copyable (for shared memory and global memory operations)
 */

// Permutes the data in an array such that data[i] = data[bit_reverse(i)]
// and data[bit_reverse(i)] = data[i]
template<typename T>
__launch_bounds__(1024) __global__
void bit_rev_permutation(T* d_out, const T *d_in, uint32_t lg_domain_size,
                         uint32_t padded_poly_size, uint32_t poly_count)
{
    const uint32_t poly_idx = blockIdx.y + blockIdx.z * gridDim.y; // [DIFF]: use gridDim.y to calculate poly_idx
    if (poly_idx >= poly_count)
        return;
    d_out += static_cast<size_t>(poly_idx) * padded_poly_size; // [DIFF]: move out ptr to another row
    d_in += static_cast<size_t>(poly_idx) * padded_poly_size;  // [DIFF]: move in ptr to another row

    if (gridDim.x == 1 && blockDim.x == (1 << lg_domain_size)) {
        uint32_t idx = threadIdx.x;
        uint32_t rev = bit_rev(idx, lg_domain_size);

        T t = d_in[idx];
        if (d_out == d_in)
            __syncthreads();
        d_out[rev] = t;
    } else {
        index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
        index_t rev = bit_rev(idx, lg_domain_size);
        bool copy = d_out != d_in && idx == rev;

        if (idx < rev || copy) {
            T t0 = d_in[idx];
            if (!copy) {
                T t1 = d_in[rev];
                d_out[idx] = t1;
            }
            d_out[rev] = t0;
        }
    }
}

template<typename T, unsigned int Z_COUNT>
__launch_bounds__(192, 2) __global__
void bit_rev_permutation_z(T* out, const T* in, uint32_t lg_domain_size,
                           uint32_t padded_poly_size, uint32_t poly_count)
{
    const uint32_t poly_idx = blockIdx.y + blockIdx.z * gridDim.y;
    if (poly_idx >= poly_count)
        return;
    out += static_cast<size_t>(poly_idx) * padded_poly_size;   // [DIFF]: move out ptr to another row
    in += static_cast<size_t>(poly_idx) * padded_poly_size;    // [DIFF]: move in ptr to another row

    const uint32_t LG_Z_COUNT = 31 - __clz(Z_COUNT); // [DIFF]: use __clz to get lg2

    // Use byte array for extern shared memory to avoid symbol conflicts across template instantiations
    extern __shared__ unsigned char xchg_raw[];
    T (*xchg)[Z_COUNT][Z_COUNT] = reinterpret_cast<T (*)[Z_COUNT][Z_COUNT]>(xchg_raw);

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

        T regs[Z_COUNT];

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


// Inline frac_add for frac_fpext_t. Cannot include frac_ext.cuh from this translation unit.
__device__ __forceinline__ void frac_fpext_add_inplace(frac_fpext_t& lhs, const frac_fpext_t& rhs) {
    bb31_4_t new_num = lhs.num * rhs.denom + lhs.denom * rhs.num;
    lhs.denom = lhs.denom * rhs.denom;
    lhs.num = new_num;
}

// Fused bitrev + K=2 tree build kernel for frac_fpext_t.
// After bit-reversing, applies alpha to all elements and computes 2 tree layers in shmem,
// then writes results to global memory. Eliminates 2 separate tree-layer kernel passes.
//
// Output layout for each Z-tile at base_rev (elements at i*step + base_rev, i=0..7):
//   [0,1]  : layer 1 results (tree level 2)
//   [2,3]  : layer 0 results preserved (tree level 1 right half, for revert of layer 1)
//   [4..7] : original+alpha preserved (leaves, for revert of layer 0)
__launch_bounds__(32, 2) __global__
void bit_rev_frac_build_k2_kernel(
    frac_fpext_t* inout, uint32_t lg_domain_size, bb31_4_t alpha)
{
    constexpr uint32_t Z_COUNT = 8;
    constexpr uint32_t LG_Z_COUNT = 3;

    extern __shared__ unsigned char xchg_raw[];
    frac_fpext_t (*xchg)[Z_COUNT][Z_COUNT] =
        reinterpret_cast<frac_fpext_t (*)[Z_COUNT][Z_COUNT]>(xchg_raw);

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

        frac_fpext_t regs[Z_COUNT];

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++) {
            xchg[gid][i][rev] = (regs[i] = inout[i * step + base_idx]);
            if (group_idx != group_rev)
                regs[i] = inout[i * step + base_rev];
        }

        __syncwarp();

        // Apply alpha + tree layers 0 and 1 to transposed tile, then write to base_rev
        {
            frac_fpext_t* row = xchg[gid][rev];
            #pragma unroll
            for (uint32_t i = 0; i < Z_COUNT; i++)
                row[i].denom = row[i].denom + alpha;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++)  // tree layer 0: pairs (0,4),(1,5),(2,6),(3,7)
                frac_fpext_add_inplace(row[i], row[i + 4]);
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++)  // tree layer 1: pairs (0,2),(1,3)
                frac_fpext_add_inplace(row[i], row[i + 2]);
        }
        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            inout[i * step + base_rev] = xchg[gid][rev][i];

        if (group_idx == group_rev)
            continue;

        __syncwarp();

        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            xchg[gid][i][rev] = regs[i];

        __syncwarp();

        // Apply alpha + tree layers 0 and 1 to transposed tile, then write to base_idx
        {
            frac_fpext_t* row = xchg[gid][rev];
            #pragma unroll
            for (uint32_t i = 0; i < Z_COUNT; i++)
                row[i].denom = row[i].denom + alpha;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++)
                frac_fpext_add_inplace(row[i], row[i + 4]);
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++)
                frac_fpext_add_inplace(row[i], row[i + 2]);
        }
        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            inout[i * step + base_idx] = xchg[gid][rev][i];

    } while (Z_COUNT <= WARP_SIZE && (tid += blockDim.x*gridDim.x) < step);
}

template<typename T>
static int bit_rev_impl(T* d_out, const T* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    size_t domain_size = (size_t)1 << lg_domain_size;
    // aim to read 4 cache lines of consecutive data per read
    const uint32_t Z_COUNT = 256 / sizeof(T);
    const uint32_t bsize = Z_COUNT > WARP_SIZE ? Z_COUNT : WARP_SIZE;

    if (poly_count == 0)
        return cudaSuccess;

    // [DIFF]: calculate grid_y, grid_z from poly_count
    const uint32_t MAX_Y = 65535;
    uint32_t grid_y = poly_count < MAX_Y ? poly_count : MAX_Y;
    uint32_t grid_z = (poly_count + grid_y - 1) / grid_y;

    // [DIFF]: N -> dim3(N, poly_count) in grid_size; stream -> cudaStreamPerThread
    if (domain_size <= 1024)
        bit_rev_permutation<T><<<dim3(1u, grid_y, grid_z), domain_size>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    else if (domain_size < bsize * Z_COUNT)
        bit_rev_permutation<T><<<dim3(domain_size / WARP_SIZE, grid_y, grid_z), WARP_SIZE>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    else if (Z_COUNT > WARP_SIZE || lg_domain_size <= 32)
        bit_rev_permutation_z<T, Z_COUNT><<<dim3(domain_size / Z_COUNT / bsize, grid_y, grid_z), bsize,
                                            bsize * Z_COUNT * sizeof(T)>>>
                            (d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    else {
        // Those GPUs that can reserve 96KB of shared memory can
        // schedule 2 blocks to each SM...
        int device;
        cudaGetDevice(&device);
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

        bit_rev_permutation_z<T, Z_COUNT><<<dim3(sm_count * 2, grid_y, grid_z), 192,
                                            192 * Z_COUNT * sizeof(T)>>>
                                (d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    }

    return CHECK_KERNEL();
}

extern "C" int _bit_rev(fr_t* d_out, const fr_t* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    return bit_rev_impl(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
}

extern "C" int _bit_rev_ext(bb31_4_t* d_out, const bb31_4_t* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    return bit_rev_impl(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
}

extern "C" int _bit_rev_frac_ext(frac_fpext_t* d_out, const frac_fpext_t* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    return bit_rev_impl(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
}

// Fused bitrev + K=2 tree build for a single frac_fpext_t buffer.
// Requires domain_size >= 256 (uses Z-tile shmem kernel).
// Applies alpha to denominators and fuses tree layers 0 and 1 in shmem.
extern "C" int _bit_rev_frac_ext_build_k2(
    frac_fpext_t* inout, uint32_t lg_domain_size, bb31_4_t alpha)
{
    constexpr uint32_t Z_COUNT = 8;
    constexpr uint32_t bsize = WARP_SIZE;  // 32
    size_t domain_size = (size_t)1 << lg_domain_size;
    if (domain_size < (size_t)(bsize * Z_COUNT))
        return cudaErrorInvalidValue;
    uint32_t grid_x = (uint32_t)(domain_size / Z_COUNT / bsize);
    size_t shmem_bytes = (size_t)bsize * Z_COUNT * sizeof(frac_fpext_t);
    bit_rev_frac_build_k2_kernel<<<dim3(grid_x), bsize, shmem_bytes>>>(
        inout, lg_domain_size, alpha);
    return CHECK_KERNEL();
}