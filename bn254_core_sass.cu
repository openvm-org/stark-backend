/// Standalone CUDA kernels wrapping the BN254 core field-arithmetic and
/// permutation device functions from:
///   - bn254_utils.cu      (namespace bn254_b64, 64-bit-limb reference impl)
///   - bn254_u32_utils.cu  (namespace bn254_b32, 32-bit-limb GPU-friendly impl)
///
/// Each kernel loads its inputs from global memory, invokes the (force-inlined)
/// device function once, and stores the result back so nvcc can't fold the
/// call away. Kernel names are `kernel_<fn>_b{32,64}` (extern "C", unmangled).
///
/// The host main launches each migrated b32 kernel alongside its b64 twin on
/// identical input bytes and compares the results. Both layouts share the same
/// little-endian byte order, so `uint32_t[8]` and `uint64_t[4]` overlap exactly
/// in memory — we cast pointers between the two when launching the kernels.
///
/// The unmigrated b64 kernels exist purely so dump_sass.sh can pull their SASS
/// into sass_b64/.

#include "bn254_u32_utils.cu" // bn254_b32::*
#include "bn254_utils.cu"     // bn254_b64::*
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ===========================================================================
// b64 wrappers (full set)
// ===========================================================================

extern "C" __global__ void kernel_add256_ret_b64(
    uint64_t *r, uint64_t *carry_out, const uint64_t *a, const uint64_t *b
) {
    r += threadIdx.x; carry_out += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    uint64_t la[4], lb[4], lr[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la[i] = a[i];
        lb[i] = b[i];
    }
    uint64_t carry = bn254_b64::add256_ret(lr, la, lb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr[i];
    }
    *carry_out = carry;
}

extern "C" __global__ void kernel_sub256_ret_b64(
    uint64_t *r, uint64_t *borrow_out, const uint64_t *a, const uint64_t *b
) {
    r += threadIdx.x; borrow_out += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    uint64_t la[4], lb[4], lr[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la[i] = a[i];
        lb[i] = b[i];
    }
    uint64_t borrow = bn254_b64::sub256_ret(lr, la, lb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr[i];
    }
    *borrow_out = borrow;
}

extern "C" __global__ void kernel_mul_small_b64(
    uint64_t *high4, uint64_t *low_out, const uint64_t *lhs, const uint64_t *rhs
) {
    high4 += threadIdx.x; low_out += threadIdx.x; lhs += threadIdx.x; rhs += threadIdx.x;
    uint64_t llhs[4], lhigh[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        llhs[i] = lhs[i];
    }
    uint64_t low = bn254_b64::mul_small(lhigh, llhs, *rhs);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        high4[i] = lhigh[i];
    }
    *low_out = low;
}

extern "C" __global__ void kernel_mul_small_and_acc_b64(
    uint64_t *high4,
    uint64_t *low_out,
    const uint64_t *lhs,
    const uint64_t *rhs,
    const uint64_t *add
) {
    high4 += threadIdx.x; low_out += threadIdx.x; lhs += threadIdx.x; rhs += threadIdx.x; add += threadIdx.x;
    uint64_t llhs[4], ladd[4], lhigh[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        llhs[i] = lhs[i];
        ladd[i] = add[i];
    }
    uint64_t low = bn254_b64::mul_small_and_acc(lhigh, llhs, *rhs, ladd);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        high4[i] = lhigh[i];
    }
    *low_out = low;
}

extern "C" __global__ void
kernel_imr_b64(uint64_t *r, const uint64_t *acc0, const uint64_t *acc) {
    r += threadIdx.x; acc0 += threadIdx.x; acc += threadIdx.x;
    uint64_t lacc[4], lr[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        lacc[i] = acc[i];
    }
    bn254_b64::imr(lr, *acc0, lacc);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr[i];
    }
}

extern "C" __global__ void
kernel_bn254_monty_mul_b64(uint64_t *r, const uint64_t *lhs, const uint64_t *rhs) {
    r += threadIdx.x; lhs += threadIdx.x; rhs += threadIdx.x;
    uint64_t llhs[4], lrhs[4], lr[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        llhs[i] = lhs[i];
        lrhs[i] = rhs[i];
    }
    bn254_b64::bn254_monty_mul(lr, llhs, lrhs);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr[i];
    }
}

extern "C" __global__ void kernel_bn254_poseidon2_permute_b64(Bn254Fr *state) {
    state += threadIdx.x;
    Bn254Fr ls[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
        ls[i] = state[i];
    }
    bn254_b64::bn254_poseidon2_permute(ls);
#pragma unroll
    for (int i = 0; i < 3; i++) {
        state[i] = ls[i];
    }
}

extern "C" __global__ void
kernel_bn254_add_b64(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    r += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    Bn254Fr la, lb;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la.limbs[i] = a[i];
        lb.limbs[i] = b[i];
    }
    Bn254Fr lr = bn254_b64::bn254_add(la, lb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_sub_b64(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    r += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    Bn254Fr la, lb;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la.limbs[i] = a[i];
        lb.limbs[i] = b[i];
    }
    Bn254Fr lr = bn254_b64::bn254_sub(la, lb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void kernel_bn254_neg_b64(uint64_t *r, const uint64_t *a) {
    r += threadIdx.x; a += threadIdx.x;
    Bn254Fr la;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la.limbs[i] = a[i];
    }
    Bn254Fr lr = bn254_b64::bn254_neg(la);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void kernel_bn254_double_b64(uint64_t *r, const uint64_t *a) {
    r += threadIdx.x; a += threadIdx.x;
    Bn254Fr la;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la.limbs[i] = a[i];
    }
    Bn254Fr lr = bn254_b64::bn254_double(la);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_mul_b64(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    r += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    Bn254Fr la, lb;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la.limbs[i] = a[i];
        lb.limbs[i] = b[i];
    }
    Bn254Fr lr = bn254_b64::bn254_mul(la, lb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void kernel_bn254_sbox_b64(uint64_t *r, const uint64_t *a) {
    r += threadIdx.x; a += threadIdx.x;
    Bn254Fr la;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la.limbs[i] = a[i];
    }
    Bn254Fr lr = bn254_b64::bn254_sbox(la);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_from_canonical_b64(uint64_t *r, const uint64_t *canonical) {
    r += threadIdx.x; canonical += threadIdx.x;
    uint64_t lc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        lc[i] = canonical[i];
    }
    Bn254Fr lr = bn254_b64::bn254_from_canonical(lc);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_to_canonical_b64(uint64_t *canonical_out, const uint64_t *a) {
    canonical_out += threadIdx.x; a += threadIdx.x;
    Bn254Fr la;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        la.limbs[i] = a[i];
    }
    uint64_t lc[4];
    bn254_b64::bn254_to_canonical(lc, la);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        canonical_out[i] = lc[i];
    }
}

extern "C" __global__ void
kernel_bn254_pack_base_2_31_b64(uint64_t *r, const uint32_t *bb) {
    r += threadIdx.x; bb += threadIdx.x;
    Bn254Fr lr = bn254_b64::bn254_pack_base_2_31(bb, 8);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_mds_external_b64(uint64_t *state) {
    state += threadIdx.x;
    Bn254Fr s[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            s[i].limbs[j] = state[i * 4 + j];
        }
    }
    bn254_b64::bn254_mds_external<3>(s);
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            state[i * 4 + j] = s[i].limbs[j];
        }
    }
}

extern "C" __global__ void
kernel_bn254_mds_internal_b64(uint64_t *state) {
    state += threadIdx.x;
    Bn254Fr s[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            s[i].limbs[j] = state[i * 4 + j];
        }
    }
    bn254_b64::bn254_mds_internal<3>(s);
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            state[i * 4 + j] = s[i].limbs[j];
        }
    }
}

// ===========================================================================
// b32 wrappers (migrated subset only)
// ===========================================================================

extern "C" __global__ void kernel_add256_ret_b32(
    uint32_t *r, uint32_t *carry_out, const uint32_t *a, const uint32_t *b
) {
    r += threadIdx.x; carry_out += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    uint32_t la[8], lb[8], lr[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la[i] = a[i];
        lb[i] = b[i];
    }
    uint32_t carry = bn254_b32::add256_ret(lr, la, lb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr[i];
    }
    *carry_out = carry;
}

extern "C" __global__ void kernel_sub256_ret_b32(
    uint32_t *r, uint32_t *borrow_out, const uint32_t *a, const uint32_t *b
) {
    r += threadIdx.x; borrow_out += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    uint32_t la[8], lb[8], lr[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la[i] = a[i];
        lb[i] = b[i];
    }
    uint32_t borrow = bn254_b32::sub256_ret(lr, la, lb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr[i];
    }
    *borrow_out = borrow;
}

// b32 mul_small produces 9 limbs (256x32 → 288): lowest limb is the return
// value, written to *low_out; upper 8 limbs go to high[0..7]. With rhs fitting
// in u32, the math also fits in 9 u32 limbs, so high[7] is genuinely the top.
extern "C" __global__ void kernel_mul_small_b32(
    uint32_t *high, uint32_t *low_out, const uint32_t *lhs, const uint32_t *rhs
) {
    high += threadIdx.x; low_out += threadIdx.x; lhs += threadIdx.x; rhs += threadIdx.x;
    uint32_t llhs[8], lhigh[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        llhs[i] = lhs[i];
    }
    uint32_t low = bn254_b32::mul_small(lhigh, llhs, *rhs);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        high[i] = lhigh[i];
    }
    *low_out = low;
}

extern "C" __global__ void kernel_mul_small_and_acc_b32(
    uint32_t *high,
    uint32_t *low_out,
    const uint32_t *lhs,
    const uint32_t *rhs,
    const uint32_t *add
) {
    high += threadIdx.x; low_out += threadIdx.x; lhs += threadIdx.x; rhs += threadIdx.x; add += threadIdx.x;
    uint32_t llhs[8], ladd[8], lhigh[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        llhs[i] = lhs[i];
        ladd[i] = add[i];
    }
    uint32_t low = bn254_b32::mul_small_and_acc(lhigh, llhs, *rhs, ladd);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        high[i] = lhigh[i];
    }
    *low_out = low;
}

extern "C" __global__ void
kernel_imr_b32(uint32_t *r, const uint32_t *acc0, const uint32_t *acc) {
    r += threadIdx.x; acc0 += threadIdx.x; acc += threadIdx.x;
    uint32_t lacc[8], lr[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        lacc[i] = acc[i];
    }
    bn254_b32::imr(lr, *acc0, lacc);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr[i];
    }
}

extern "C" __global__ void
kernel_bn254_monty_mul_b32(uint32_t *r, const uint32_t *lhs, const uint32_t *rhs) {
    r += threadIdx.x; lhs += threadIdx.x; rhs += threadIdx.x;
    uint32_t llhs[8], lrhs[8], lr[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        llhs[i] = lhs[i];
        lrhs[i] = rhs[i];
    }
    bn254_b32::bn254_monty_mul(lr, llhs, lrhs);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr[i];
    }
}

extern "C" __global__ void
kernel_bn254_add_b32(uint32_t *r, const uint32_t *a, const uint32_t *b) {
    r += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    Bn254Fr32 la, lb;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la.limbs[i] = a[i];
        lb.limbs[i] = b[i];
    }
    Bn254Fr32 lr = bn254_b32::bn254_add(la, lb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_sub_b32(uint32_t *r, const uint32_t *a, const uint32_t *b) {
    r += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    Bn254Fr32 la, lb;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la.limbs[i] = a[i];
        lb.limbs[i] = b[i];
    }
    Bn254Fr32 lr = bn254_b32::bn254_sub(la, lb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void kernel_bn254_neg_b32(uint32_t *r, const uint32_t *a) {
    r += threadIdx.x; a += threadIdx.x;
    Bn254Fr32 la;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la.limbs[i] = a[i];
    }
    Bn254Fr32 lr = bn254_b32::bn254_neg(la);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void kernel_bn254_double_b32(uint32_t *r, const uint32_t *a) {
    r += threadIdx.x; a += threadIdx.x;
    Bn254Fr32 la;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la.limbs[i] = a[i];
    }
    Bn254Fr32 lr = bn254_b32::bn254_double(la);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_mul_b32(uint32_t *r, const uint32_t *a, const uint32_t *b) {
    r += threadIdx.x; a += threadIdx.x; b += threadIdx.x;
    Bn254Fr32 la, lb;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la.limbs[i] = a[i];
        lb.limbs[i] = b[i];
    }
    Bn254Fr32 lr = bn254_b32::bn254_mul(la, lb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void kernel_bn254_sbox_b32(uint32_t *r, const uint32_t *a) {
    r += threadIdx.x; a += threadIdx.x;
    Bn254Fr32 la;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la.limbs[i] = a[i];
    }
    Bn254Fr32 lr = bn254_b32::bn254_sbox(la);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_from_canonical_b32(uint32_t *r, const uint32_t *canonical) {
    r += threadIdx.x; canonical += threadIdx.x;
    uint32_t lc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        lc[i] = canonical[i];
    }
    Bn254Fr32 lr = bn254_b32::bn254_from_canonical(lc);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_to_canonical_b32(uint32_t *canonical_out, const uint32_t *a) {
    canonical_out += threadIdx.x; a += threadIdx.x;
    Bn254Fr32 la;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        la.limbs[i] = a[i];
    }
    uint32_t lc[8];
    bn254_b32::bn254_to_canonical(lc, la);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        canonical_out[i] = lc[i];
    }
}

extern "C" __global__ void
kernel_bn254_pack_base_2_31_b32(uint32_t *r, const uint32_t *bb) {
    r += threadIdx.x; bb += threadIdx.x;
    Bn254Fr32 lr = bn254_b32::bn254_pack_base_2_31(bb, 8);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = lr.limbs[i];
    }
}

extern "C" __global__ void
kernel_bn254_mds_external_b32(uint32_t *state) {
    state += threadIdx.x;
    Bn254Fr32 s[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            s[i].limbs[j] = state[i * 8 + j];
        }
    }
    bn254_b32::bn254_mds_external<3>(s);
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            state[i * 8 + j] = s[i].limbs[j];
        }
    }
}

extern "C" __global__ void
kernel_bn254_mds_internal_b32(uint32_t *state) {
    state += threadIdx.x;
    Bn254Fr32 s[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            s[i].limbs[j] = state[i * 8 + j];
        }
    }
    bn254_b32::bn254_mds_internal<3>(s);
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            state[i * 8 + j] = s[i].limbs[j];
        }
    }
}

// ===========================================================================
// Host driver: launch b32 and b64 kernels on the same inputs and compare.
// ===========================================================================

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

static bool compare_256(const char *label, const uint64_t b32[4], const uint64_t b64[4]) {
    if (std::memcmp(b32, b64, 32) == 0) {
        std::printf("[OK]   %-22s result match\n", label);
        return true;
    }
    std::printf("[FAIL] %-22s result mismatch:\n", label);
    for (int i = 0; i < 4; i++) {
        std::printf(
            "         limb%d  b32=0x%016llx  b64=0x%016llx\n",
            i,
            (unsigned long long)b32[i],
            (unsigned long long)b64[i]
        );
    }
    return false;
}

static bool compare_carry(const char *label, uint32_t b32, uint64_t b64) {
    if ((uint64_t)b32 == b64) {
        std::printf(
            "[OK]   %-22s carry/borrow match (= %llu)\n", label, (unsigned long long)b64
        );
        return true;
    }
    std::printf(
        "[FAIL] %-22s carry/borrow mismatch: b32=%u b64=%llu\n",
        label,
        b32,
        (unsigned long long)b64
    );
    return false;
}

static bool compare_bytes(const char *label, const void *b32, const void *b64, size_t n) {
    if (std::memcmp(b32, b64, n) == 0) {
        std::printf("[OK]   %-22s %zu-byte result match\n", label, n);
        return true;
    }
    std::printf("[FAIL] %-22s %zu-byte result mismatch:\n", label, n);
    const uint8_t *pa = (const uint8_t *)b32;
    const uint8_t *pb = (const uint8_t *)b64;
    for (size_t i = 0; i < n; i++) {
        if (pa[i] != pb[i]) {
            std::printf("         byte %2zu: b32=0x%02x b64=0x%02x\n", i, pa[i], pb[i]);
        }
    }
    return false;
}

struct TestVec {
    const char *name;
    uint64_t a[4];
    uint64_t b[4];
    uint32_t rhs; // used as the rhs in mul_small{,_and_acc}; must fit in u32.
};

int main() {
    // Two 256-bit pairs. Identical bytes for both layouts (little-endian).
    //  - vec 0: a + b overflows; a - b doesn't borrow.
    //  - vec 1: a + b doesn't overflow; a - b borrows (a < b).
    constexpr uint64_t U64_MAX = 0xffffffffffffffffULL;
    TestVec vecs[] = {
        {"carry / no-borrow",
         {0xfedcba9876543210ULL,
          0x123456789abcdef0ULL,
          0xdeadbeefcafebabeULL,
          0xfffffffffffffff0ULL},
         {0x0fedcba987654321ULL,
          0x0123456789abcdefULL,
          0x9988776655443322ULL,
          0x000000000000000fULL},
         0xcafebabeu},
        {"no-carry / borrow",
         {0x0000000000000010ULL, 0, 0, 0},
         {0xfedcba9876543210ULL,
          0x123456789abcdef0ULL,
          0xdeadbeefcafebabeULL,
          0x000000000000000fULL},
         0x12345678u},
        {"all zeros",
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         0u},
        {"a=max b=zero",
         {U64_MAX, U64_MAX, U64_MAX, U64_MAX},
         {0, 0, 0, 0},
         0xffffffffu},
        {"a=zero b=max",
         {0, 0, 0, 0},
         {U64_MAX, U64_MAX, U64_MAX, U64_MAX},
         0u},
        {"a=max b=max rhs=max",
         {U64_MAX, U64_MAX, U64_MAX, U64_MAX},
         {U64_MAX, U64_MAX, U64_MAX, U64_MAX},
         0xffffffffu},
        {"a=1 b=2 rhs=1",
         {1, 0, 0, 0},
         {2, 0, 0, 0},
         1u},
        {"single-bit top",
         {0, 0, 0, 0x8000000000000000ULL},
         {0, 0, 0, 0x8000000000000000ULL},
         0x80000000u},
        {"alternating 0xa/0x5",
         {0xaaaaaaaaaaaaaaaaULL,
          0xaaaaaaaaaaaaaaaaULL,
          0xaaaaaaaaaaaaaaaaULL,
          0xaaaaaaaaaaaaaaaaULL},
         {0x5555555555555555ULL,
          0x5555555555555555ULL,
          0x5555555555555555ULL,
          0x5555555555555555ULL},
         0x55555555u},
        {"bn254-P-ish",
         {0x43e1f593f0000001ULL,
          0x2833e84879b97091ULL,
          0xb85045b68181585dULL,
          0x30644e72e131a029ULL},
         {0x43e1f593f0000000ULL,
          0x2833e84879b97091ULL,
          0xb85045b68181585dULL,
          0x30644e72e131a029ULL},
         0x43e1f593u},
        {"low-limb-max only",
         {U64_MAX, 0, 0, 0},
         {U64_MAX, 0, 0, 0},
         0xffffffffu},
        {"high-limb-max only",
         {0, 0, 0, U64_MAX},
         {0, 0, 0, U64_MAX},
         0xffffffffu},
        {"powers spread",
         {0x0000000080000000ULL,
          0x0000000080000000ULL,
          0x0000000080000000ULL,
          0x0000000080000000ULL},
         {0x00000000ffffffffULL,
          0x00000000ffffffffULL,
          0x00000000ffffffffULL,
          0x00000000ffffffffULL},
         0x00000001u},
        {"golden ratio mix",
         {0x9e3779b97f4a7c15ULL,
          0xf39cc0605cedc834ULL,
          0x1082276bf3a27251ULL,
          0xf86c6a11d0c18e95ULL},
         {0xa1c2d3e4f5061728ULL,
          0x394a5b6c7d8e9fa0ULL,
          0xb1c2d3e4f5061728ULL,
          0x394a5b6c7d8e9fa0ULL},
         0xa1c2d3e4u},
        {"deadbeef mix",
         {0x123456789abcdef0ULL,
          0x0fedcba987654321ULL,
          0xdeadbeefcafebabeULL,
          0x1234567890abcdefULL},
         {0xfedcba0987654321ULL,
          0x0123456789abcdefULL,
          0xabcdef0123456789ULL,
          0x9876543210fedcbaULL},
         0xdeadbeefu},
        {"cafe/feed mix",
         {0x0011223344556677ULL,
          0x8899aabbccddeeffULL,
          0xfedcba9876543210ULL,
          0x0123456789abcdefULL},
         {0xcafebabe12345678ULL,
          0xdeadbeef9abcdef0ULL,
          0xfeedface0badf00dULL,
          0x8badf00d0badf00dULL},
         0xcafe0001u},
        {"7-8-9-a vs 1-2-3-4",
         {0x7777777777777777ULL,
          0x8888888888888888ULL,
          0x9999999999999999ULL,
          0xaaaaaaaaaaaaaaaaULL},
         {0x1111111111111111ULL,
          0x2222222222222222ULL,
          0x3333333333333333ULL,
          0x4444444444444444ULL},
         0xabcdef01u},
        {"near-overflow add",
         {U64_MAX - 1, U64_MAX, U64_MAX, U64_MAX - 1},
         {1, 0, 0, 1},
         0x00000002u},
    };

    uint64_t *d_a = nullptr;
    uint64_t *d_b = nullptr;
    uint64_t *d_r_b32 = nullptr;
    uint64_t *d_r_b64 = nullptr;
    uint32_t *d_c_b32 = nullptr;
    uint64_t *d_c_b64 = nullptr;
    uint32_t *d_low_b32 = nullptr; // mul_small low return for b32 (uint32)
    // rhs storage: 8 bytes so the b64 kernel sees a uint64 with high half = 0
    // and the b32 kernel sees the low uint32 via pointer cast.
    uint64_t *d_rhs = nullptr;
    // 3-element state buffer (96 bytes) for mds_external / mds_internal.
    uint64_t *d_state_b32 = nullptr;
    uint64_t *d_state_b64 = nullptr;

    check_cuda(cudaMalloc(&d_a, 32), "malloc a");
    check_cuda(cudaMalloc(&d_b, 32), "malloc b");
    check_cuda(cudaMalloc(&d_r_b32, 32), "malloc r_b32");
    check_cuda(cudaMalloc(&d_r_b64, 32), "malloc r_b64");
    check_cuda(cudaMalloc(&d_c_b32, sizeof(uint32_t)), "malloc c_b32");
    check_cuda(cudaMalloc(&d_c_b64, sizeof(uint64_t)), "malloc c_b64");
    check_cuda(cudaMalloc(&d_low_b32, sizeof(uint32_t)), "malloc low_b32");
    check_cuda(cudaMalloc(&d_rhs, sizeof(uint64_t)), "malloc rhs");
    check_cuda(cudaMalloc(&d_state_b32, 96), "malloc state_b32");
    check_cuda(cudaMalloc(&d_state_b64, 96), "malloc state_b64");

    int failures = 0;

    for (const auto &v : vecs) {
        std::printf("--- test vector: %s ---\n", v.name);
        check_cuda(cudaMemcpy(d_a, v.a, 32, cudaMemcpyHostToDevice), "copy a");
        check_cuda(cudaMemcpy(d_b, v.b, 32, cudaMemcpyHostToDevice), "copy b");

        // add256_ret
        kernel_add256_ret_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32,
            d_c_b32,
            (const uint32_t *)d_a,
            (const uint32_t *)d_b
        );
        kernel_add256_ret_b64<<<1, 1>>>(d_r_b64, d_c_b64, d_a, d_b);
        check_cuda(cudaDeviceSynchronize(), "sync add");

        uint64_t r_b32[4]{};
        uint64_t r_b64[4]{};
        uint32_t c_b32 = 0;
        uint64_t c_b64 = 0;
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 add");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 add");
        check_cuda(
            cudaMemcpy(&c_b32, d_c_b32, sizeof(uint32_t), cudaMemcpyDeviceToHost),
            "copy c_b32 add"
        );
        check_cuda(
            cudaMemcpy(&c_b64, d_c_b64, sizeof(uint64_t), cudaMemcpyDeviceToHost),
            "copy c_b64 add"
        );
        failures += !compare_256("add256_ret", r_b32, r_b64);
        failures += !compare_carry("add256_ret", c_b32, c_b64);

        // sub256_ret
        kernel_sub256_ret_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32,
            d_c_b32,
            (const uint32_t *)d_a,
            (const uint32_t *)d_b
        );
        kernel_sub256_ret_b64<<<1, 1>>>(d_r_b64, d_c_b64, d_a, d_b);
        check_cuda(cudaDeviceSynchronize(), "sync sub");

        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 sub");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 sub");
        check_cuda(
            cudaMemcpy(&c_b32, d_c_b32, sizeof(uint32_t), cudaMemcpyDeviceToHost),
            "copy c_b32 sub"
        );
        check_cuda(
            cudaMemcpy(&c_b64, d_c_b64, sizeof(uint64_t), cudaMemcpyDeviceToHost),
            "copy c_b64 sub"
        );
        failures += !compare_256("sub256_ret", r_b32, r_b64);
        failures += !compare_carry("sub256_ret", c_b32, c_b64);

        // Upload rhs as a u64 with high half = 0 so b64's mul_small sees a
        // value that fits in u32 (top limb of its 320-bit result will be 0)
        // and b32 reads the low u32 directly via pointer cast.
        uint64_t rhs_u64 = (uint64_t)v.rhs;
        check_cuda(cudaMemcpy(d_rhs, &rhs_u64, sizeof(uint64_t), cudaMemcpyHostToDevice), "copy rhs");

        // mul_small: 256x32 → 288. The bottom 288 bits (36 bytes) of either
        // result are byte-identical; we pack low+high contiguously on the host
        // and memcmp the relevant 36-byte span.
        kernel_mul_small_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32, d_low_b32, (const uint32_t *)d_a, (const uint32_t *)d_rhs
        );
        kernel_mul_small_b64<<<1, 1>>>(d_r_b64, d_c_b64, d_a, d_rhs);
        check_cuda(cudaDeviceSynchronize(), "sync mul_small");

        uint32_t low_b32 = 0;
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 mul");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 mul");
        check_cuda(
            cudaMemcpy(&low_b32, d_low_b32, sizeof(uint32_t), cudaMemcpyDeviceToHost),
            "copy low_b32 mul"
        );
        check_cuda(
            cudaMemcpy(&c_b64, d_c_b64, sizeof(uint64_t), cudaMemcpyDeviceToHost),
            "copy low_b64 mul"
        );

        // b32 layout: 4-byte low + 32-byte high = 36 bytes.
        // b64 layout: 8-byte low + 32-byte high = 40 bytes (top 4 bytes = 0).
        uint8_t packed_b32[36];
        uint8_t packed_b64[40];
        std::memcpy(packed_b32, &low_b32, 4);
        std::memcpy(packed_b32 + 4, r_b32, 32);
        std::memcpy(packed_b64, &c_b64, 8);
        std::memcpy(packed_b64 + 8, r_b64, 32);
        failures += !compare_bytes("mul_small", packed_b32, packed_b64, 36);

        // mul_small_and_acc: a * rhs + b (use b as add[8])
        kernel_mul_small_and_acc_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32,
            d_low_b32,
            (const uint32_t *)d_a,
            (const uint32_t *)d_rhs,
            (const uint32_t *)d_b
        );
        kernel_mul_small_and_acc_b64<<<1, 1>>>(d_r_b64, d_c_b64, d_a, d_rhs, d_b);
        check_cuda(cudaDeviceSynchronize(), "sync mul_small_and_acc");

        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 mac");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 mac");
        check_cuda(
            cudaMemcpy(&low_b32, d_low_b32, sizeof(uint32_t), cudaMemcpyDeviceToHost),
            "copy low_b32 mac"
        );
        check_cuda(
            cudaMemcpy(&c_b64, d_c_b64, sizeof(uint64_t), cudaMemcpyDeviceToHost),
            "copy low_b64 mac"
        );
        std::memcpy(packed_b32, &low_b32, 4);
        std::memcpy(packed_b32 + 4, r_b32, 32);
        std::memcpy(packed_b64, &c_b64, 8);
        std::memcpy(packed_b64 + 8, r_b64, 32);
        failures += !compare_bytes("mul_small_and_acc", packed_b32, packed_b64, 36);

        // bn254_monty_mul: 256x256 → 256-bit Montgomery product. Both versions
        // implement CIOS with the same R = 2^256, so the 32-byte results must
        // match byte-for-byte. (imr alone is not directly comparable: b32
        // reduces by 32 bits per call while b64 reduces by 64, but monty_mul
        // calls imr the right number of times in each variant.)
        kernel_bn254_monty_mul_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32, (const uint32_t *)d_a, (const uint32_t *)d_b
        );
        kernel_bn254_monty_mul_b64<<<1, 1>>>(d_r_b64, d_a, d_b);
        check_cuda(cudaDeviceSynchronize(), "sync monty_mul");

        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 mm");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 mm");
        failures += !compare_256("bn254_monty_mul", r_b32, r_b64);

        // bn254_add (Bn254Fr / Bn254Fr32 wrappers around add256_ret + cond -P)
        kernel_bn254_add_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32, (const uint32_t *)d_a, (const uint32_t *)d_b
        );
        kernel_bn254_add_b64<<<1, 1>>>(d_r_b64, d_a, d_b);
        check_cuda(cudaDeviceSynchronize(), "sync bn254_add");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 add");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 add");
        failures += !compare_256("bn254_add", r_b32, r_b64);

        // bn254_sub
        kernel_bn254_sub_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32, (const uint32_t *)d_a, (const uint32_t *)d_b
        );
        kernel_bn254_sub_b64<<<1, 1>>>(d_r_b64, d_a, d_b);
        check_cuda(cudaDeviceSynchronize(), "sync bn254_sub");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 sub");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 sub");
        failures += !compare_256("bn254_sub", r_b32, r_b64);

        // bn254_neg (unary)
        kernel_bn254_neg_b32<<<1, 1>>>((uint32_t *)d_r_b32, (const uint32_t *)d_a);
        kernel_bn254_neg_b64<<<1, 1>>>(d_r_b64, d_a);
        check_cuda(cudaDeviceSynchronize(), "sync bn254_neg");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 neg");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 neg");
        failures += !compare_256("bn254_neg", r_b32, r_b64);

        // bn254_double (unary)
        kernel_bn254_double_b32<<<1, 1>>>((uint32_t *)d_r_b32, (const uint32_t *)d_a);
        kernel_bn254_double_b64<<<1, 1>>>(d_r_b64, d_a);
        check_cuda(cudaDeviceSynchronize(), "sync bn254_double");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 dbl");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 dbl");
        failures += !compare_256("bn254_double", r_b32, r_b64);

        // bn254_mul
        kernel_bn254_mul_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32, (const uint32_t *)d_a, (const uint32_t *)d_b
        );
        kernel_bn254_mul_b64<<<1, 1>>>(d_r_b64, d_a, d_b);
        check_cuda(cudaDeviceSynchronize(), "sync bn254_mul");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 mul");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 mul");
        failures += !compare_256("bn254_mul", r_b32, r_b64);

        // bn254_sbox (x^5)
        kernel_bn254_sbox_b32<<<1, 1>>>((uint32_t *)d_r_b32, (const uint32_t *)d_a);
        kernel_bn254_sbox_b64<<<1, 1>>>(d_r_b64, d_a);
        check_cuda(cudaDeviceSynchronize(), "sync bn254_sbox");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 sbox");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 sbox");
        failures += !compare_256("bn254_sbox", r_b32, r_b64);

        // bn254_from_canonical (uses a as a canonical 256-bit input)
        kernel_bn254_from_canonical_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32, (const uint32_t *)d_a
        );
        kernel_bn254_from_canonical_b64<<<1, 1>>>(d_r_b64, d_a);
        check_cuda(cudaDeviceSynchronize(), "sync from_canonical");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 fc");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 fc");
        failures += !compare_256("bn254_from_canonical", r_b32, r_b64);

        // bn254_to_canonical (uses a as a Montgomery-form input)
        kernel_bn254_to_canonical_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32, (const uint32_t *)d_a
        );
        kernel_bn254_to_canonical_b64<<<1, 1>>>(d_r_b64, d_a);
        check_cuda(cudaDeviceSynchronize(), "sync to_canonical");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 tc");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 tc");
        failures += !compare_256("bn254_to_canonical", r_b32, r_b64);

        // bn254_pack_base_2_31 with count=8. Treat d_b as an array of 8 u32
        // BabyBear values (each must be < 2^31 for the spec, but the algorithm
        // doesn't care for byte-comparison purposes — both impls do the same
        // packing arithmetic).
        kernel_bn254_pack_base_2_31_b32<<<1, 1>>>(
            (uint32_t *)d_r_b32, (const uint32_t *)d_b
        );
        kernel_bn254_pack_base_2_31_b64<<<1, 1>>>(d_r_b64, (const uint32_t *)d_b);
        check_cuda(cudaDeviceSynchronize(), "sync pack_base_2_31");
        check_cuda(cudaMemcpy(r_b32, d_r_b32, 32, cudaMemcpyDeviceToHost), "copy r_b32 pk");
        check_cuda(cudaMemcpy(r_b64, d_r_b64, 32, cudaMemcpyDeviceToHost), "copy r_b64 pk");
        failures += !compare_256("bn254_pack_base_2_31", r_b32, r_b64);

        // bn254_mds_external<3> / bn254_mds_internal<3>: in-place on a 3-Fr
        // state. Initialize state[0]=a, state[1]=b, state[2]=a (some pattern).
        uint64_t state_init[12];
        std::memcpy(&state_init[0], v.a, 32);
        std::memcpy(&state_init[4], v.b, 32);
        std::memcpy(&state_init[8], v.a, 32);

        check_cuda(cudaMemcpy(d_state_b32, state_init, 96, cudaMemcpyHostToDevice), "copy state_b32");
        check_cuda(cudaMemcpy(d_state_b64, state_init, 96, cudaMemcpyHostToDevice), "copy state_b64");
        kernel_bn254_mds_external_b32<<<1, 1>>>((uint32_t *)d_state_b32);
        kernel_bn254_mds_external_b64<<<1, 1>>>(d_state_b64);
        check_cuda(cudaDeviceSynchronize(), "sync mds_external");
        uint64_t state_b32[12]{};
        uint64_t state_b64[12]{};
        check_cuda(cudaMemcpy(state_b32, d_state_b32, 96, cudaMemcpyDeviceToHost), "copy state_b32 mds_e");
        check_cuda(cudaMemcpy(state_b64, d_state_b64, 96, cudaMemcpyDeviceToHost), "copy state_b64 mds_e");
        failures += !compare_bytes("bn254_mds_external", state_b32, state_b64, 96);

        check_cuda(cudaMemcpy(d_state_b32, state_init, 96, cudaMemcpyHostToDevice), "copy state_b32 (i)");
        check_cuda(cudaMemcpy(d_state_b64, state_init, 96, cudaMemcpyHostToDevice), "copy state_b64 (i)");
        kernel_bn254_mds_internal_b32<<<1, 1>>>((uint32_t *)d_state_b32);
        kernel_bn254_mds_internal_b64<<<1, 1>>>(d_state_b64);
        check_cuda(cudaDeviceSynchronize(), "sync mds_internal");
        check_cuda(cudaMemcpy(state_b32, d_state_b32, 96, cudaMemcpyDeviceToHost), "copy state_b32 mds_i");
        check_cuda(cudaMemcpy(state_b64, d_state_b64, 96, cudaMemcpyDeviceToHost), "copy state_b64 mds_i");
        failures += !compare_bytes("bn254_mds_internal", state_b32, state_b64, 96);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_r_b32);
    cudaFree(d_r_b64);
    cudaFree(d_c_b32);
    cudaFree(d_c_b64);
    cudaFree(d_low_b32);
    cudaFree(d_rhs);
    cudaFree(d_state_b32);
    cudaFree(d_state_b64);

    if (failures > 0) {
        std::printf("\n%d mismatch(es)\n", failures);
        return 1;
    }
    std::printf("\nAll comparisons OK\n");
    return 0;
}
