/**
 * Poseidon2 Benchmark: Our Implementation vs cuPQC (NVIDIA)
 *
 * Benchmarks both implementations on the same GPU with identical methodology:
 *   - N = 4M states
 *   - reps = 100 operations per state
 *   - Throughput in Gops/s (Giga-operations per second)
 *
 * "Our" implementation: raw poseidon2_mix() permutation called reps times.
 * "cuPQC" implementation: full sponge hash (reset+update+finalize+digest) called
 *   reps times, chaining output→input. With Width=16, Capacity=8, Rate=8, each
 *   call ≈ 1 permutation internally.
 *
 * Build: make
 * Run:   make run  (or  ./bench)
 */

// cuPQC headers first (avoids STL conflicts)
#include <hash.hpp>

// Our field/poseidon2 headers
#include "poseidon2.cuh"               // poseidon2::poseidon2_mix(Fp*)
#include "koala_bear/poseidon2_kb.cuh" // kb_poseidon2::poseidon2_mix(Kb*)
#include "fp.h"                        // BabyBear Fp (P = 2^31 - 2^27 + 1)
#include "koala_bear/kb.h"             // KoalaBear Kb (P = 2^31 - 2^24 + 1)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cupqc;

// cuPQC descriptors: Width=16, Capacity=8, Rate=8 elements per call
using P2_BB = decltype(POSEIDON2_BB_8_16() + Thread());
using P2_KB = decltype(POSEIDON2_KB_8_16() + Thread());

static constexpr size_t CUPQC_RATE        = 8;  // Width - Capacity
static constexpr size_t CUPQC_DIGEST_SIZE = 8;  // elements squeezed out
static constexpr int    BLOCK_SIZE        = 512;

// ============================================================================
// Init kernel: raw u32 → field elements
// ============================================================================

template<typename FieldT>
__global__ void field_init_kernel(FieldT* out, const uint32_t* raw, size_t n16)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) out[i] = FieldT(raw[i]);
}

// ============================================================================
// Kernel: our raw permutation  (16 elements per state, reps calls)
// ============================================================================

template<typename FieldT, void (*mix_fn)(FieldT*)>
__global__ void our_poseidon2_kernel(FieldT* states, size_t N, int reps)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    FieldT cells[16];
    size_t base = idx * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) cells[i] = states[base + i];

    #pragma unroll 1
    for (int r = 0; r < reps; r++) {
        mix_fn(cells);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) states[base + i] = cells[i];
}

// ============================================================================
// Kernel: cuPQC sponge hash  (rate == digest_size == 8, output feeds next input)
// ============================================================================

template<typename HashT>
__global__ void cupqc_poseidon2_kernel(
    const uint32_t* __restrict__ inputs,
    uint32_t*       __restrict__ outputs,
    size_t N,
    int reps)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint32_t state[CUPQC_RATE];
    for (int i = 0; i < (int)CUPQC_RATE; i++) {
        state[i] = inputs[idx * CUPQC_RATE + i];
    }

    #pragma unroll 1
    for (int r = 0; r < reps; r++) {
        HashT hash{};
        hash.reset();
        hash.update(state, CUPQC_RATE);
        hash.finalize();
        hash.digest(state, CUPQC_DIGEST_SIZE);
    }

    for (int i = 0; i < (int)CUPQC_DIGEST_SIZE; i++) {
        outputs[idx * CUPQC_DIGEST_SIZE + i] = state[i];
    }
}

// ============================================================================
// Helpers
// ============================================================================

static void cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static uint64_t& lcg_step(uint64_t& r)
{
    r = r * 6364136223846793005ULL + 1442695040888963407ULL;
    return r;
}

struct BenchResult { double avg_ms, gops; };

// ============================================================================
// Bench: "our" raw permutation
// ============================================================================

template<typename FieldT, void (*mix_fn)(FieldT*)>
BenchResult bench_our(const char* name, size_t N, int reps,
                      int warmup, int iters, uint32_t prime)
{
    std::vector<uint32_t> h(N * 16);
    uint64_t rng = 0xCAFEBABE00000001ULL;
    for (auto& v : h) { lcg_step(rng); v = (uint32_t)(rng >> 32) % prime; }

    uint32_t* tmp = nullptr;
    FieldT*   d_s = nullptr;
    cuda_check(cudaMalloc(&tmp, N * 16 * sizeof(uint32_t)), "malloc tmp");
    cuda_check(cudaMalloc(&d_s, N * 16 * sizeof(FieldT)),   "malloc states");
    cuda_check(cudaMemcpy(tmp, h.data(), N * 16 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "H2D");

    dim3 blk512(512);
    dim3 grd16((N * 16 + 511) / 512);
    field_init_kernel<FieldT><<<grd16, blk512>>>(d_s, tmp, N * 16);
    cuda_check(cudaDeviceSynchronize(), "init");
    cudaFree(tmp);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int i = 0; i < warmup; i++)
        our_poseidon2_kernel<FieldT, mix_fn><<<grid, block>>>(d_s, N, reps);
    cuda_check(cudaDeviceSynchronize(), "warmup");

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        our_poseidon2_kernel<FieldT, mix_fn><<<grid, block>>>(d_s, N, reps);
    cuda_check(cudaDeviceSynchronize(), "bench");
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    BenchResult r{ elapsed * 1000.0 / iters,
                   (double)((uint64_t)N * reps * iters) / elapsed / 1e9 };
    printf("| %-24s | %10.3f | %12.1f |\n", name, r.avg_ms, r.gops);

    cudaFree(d_s);
    return r;
}

// ============================================================================
// Bench: cuPQC sponge hash
// ============================================================================

template<typename HashT>
BenchResult bench_cupqc(const char* name, size_t N, int reps,
                         int warmup, int iters, uint32_t prime)
{
    std::vector<uint32_t> h(N * CUPQC_RATE);
    uint64_t rng = 0xDEADBEEF12345678ULL;
    for (auto& v : h) { lcg_step(rng); v = (uint32_t)(rng >> 32) % prime; }

    uint32_t* d_in  = nullptr;
    uint32_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_in,  N * CUPQC_RATE        * sizeof(uint32_t)), "malloc d_in");
    cuda_check(cudaMalloc(&d_out, N * CUPQC_DIGEST_SIZE * sizeof(uint32_t)), "malloc d_out");
    cuda_check(cudaMemcpy(d_in, h.data(), N * CUPQC_RATE * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "H2D");

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int i = 0; i < warmup; i++)
        cupqc_poseidon2_kernel<HashT><<<grid, block>>>(d_in, d_out, N, reps);
    cuda_check(cudaDeviceSynchronize(), "warmup");

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        cupqc_poseidon2_kernel<HashT><<<grid, block>>>(d_in, d_out, N, reps);
    cuda_check(cudaDeviceSynchronize(), "bench");
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    BenchResult r{ elapsed * 1000.0 / iters,
                   (double)((uint64_t)N * reps * iters) / elapsed / 1e9 };
    printf("| %-24s | %10.3f | %12.1f |\n", name, r.avg_ms, r.gops);

    cudaFree(d_in);
    cudaFree(d_out);
    return r;
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "getDeviceProperties");
    printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    const size_t N      = 1u << 22;  // 4M states (matches field_bench default)
    const int    reps   = 100;
    const int    warmup = 3;
    const int    iters  = 10;

    printf("=== Poseidon2 Benchmark: Our CUDA vs NVIDIA cuPQC 0.4.1 ===\n\n");
    printf("States/run  : %zu (2^22)\n", N);
    printf("Reps/state  : %d\n",         reps);
    printf("Warmup iters: %d\n",         warmup);
    printf("Bench iters : %d\n",         iters);
    printf("\n");
    printf("\"Our\" kernel   = raw poseidon2_mix() called %d× on 16-element state\n", reps);
    printf("\"cuPQC\" kernel = sponge (reset+update[8]+finalize+digest[8]) called %d×\n", reps);
    printf("               Width=16, Capacity=8, Rate=8 → ≈1 permutation/call\n");
    printf("\n");

    // ---- BabyBear ----
    printf("### BabyBear Poseidon2 (p = 2^31 - 2^27 + 1)\n\n");
    printf("| %-24s | %10s | %12s |\n", "Implementation", "Time (ms)", "Gops/s");
    printf("|--------------------------|------------|--------------|");
    printf("\n");

    auto our_bb = bench_our<Fp, poseidon2::poseidon2_mix>(
        "Our BB Poseidon2", N, reps, warmup, iters, Fp::P);
    auto cpq_bb = bench_cupqc<P2_BB>(
        "cuPQC BB Poseidon2", N, reps, warmup, iters, cupqc_common::BabyBearPrime);

    printf("\n  cuPQC / Our speedup: %.2fx\n\n", cpq_bb.gops / our_bb.gops);

    // ---- KoalaBear ----
    printf("### KoalaBear Poseidon2 (p = 2^31 - 2^24 + 1)\n\n");
    printf("| %-24s | %10s | %12s |\n", "Implementation", "Time (ms)", "Gops/s");
    printf("|--------------------------|------------|--------------|");
    printf("\n");

    auto our_kb = bench_our<Kb, kb_poseidon2::poseidon2_mix>(
        "Our KB Poseidon2", N, reps, warmup, iters, Kb::P);
    auto cpq_kb = bench_cupqc<P2_KB>(
        "cuPQC KB Poseidon2", N, reps, warmup, iters, cupqc_common::KoalaBearPrime);

    printf("\n  cuPQC / Our speedup: %.2fx\n\n", cpq_kb.gops / our_kb.gops);

    // ---- Summary ----
    printf("### Summary\n\n");
    printf("| %-24s | %12s | %12s |\n", "Implementation", "BB Gops/s", "KB Gops/s");
    printf("|--------------------------|--------------|--------------|");
    printf("\n");
    printf("| %-24s | %12.1f | %12.1f |\n", "Our CUDA",           our_bb.gops, our_kb.gops);
    printf("| %-24s | %12.1f | %12.1f |\n", "NVIDIA cuPQC 0.4.1", cpq_bb.gops, cpq_kb.gops);
    printf("\n");
    printf("Note: \"Our\" measures raw permutations; \"cuPQC\" measures full sponge\n");
    printf("hash calls. One cuPQC call (rate=8 inputs) ≈ one permutation.\n");

    return 0;
}
