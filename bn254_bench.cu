#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// Requested benchmark knobs. kHeight is derived as query_stride << log_rows_per_query,
// matching _bn254_poseidon2_compressing_row_hashes(_v2).
static constexpr size_t kWidth = 100;
static constexpr size_t kQueryStride = 10000;
static constexpr size_t kLogRowsPerQuery = 4;

static constexpr uint32_t kBabyBearPrime = 2013265921u;

extern "C" int _init_bn254_poseidon2_rc(
    const uint64_t *initial_rc,
    const uint64_t *partial_rc,
    const uint64_t *terminal_rc,
    cudaStream_t stream
);

extern "C" int _init_bn254_poseidon2_rc_w2(
    const uint64_t *initial_rc,
    const uint64_t *partial_rc,
    const uint64_t *terminal_rc,
    cudaStream_t stream
);

extern "C" int _bn254_poseidon2_compressing_row_hashes(
    uint64_t *out,
    const uint32_t *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
);

extern "C" int _bn254_poseidon2_compressing_row_hashes_v3(
    uint64_t *out,
    const uint32_t *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
);

// FpExt variants: matrix is a packed array of FpExt elements (= 4 u32 baby-bear
// values each). For benchmarking we pass the raw byte pointer; on the device
// side the kernel reinterprets it as `const FpExt *`.
extern "C" int _bn254_poseidon2_compressing_row_hashes_ext(
    uint64_t *out,
    const uint32_t *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
);

extern "C" int _bn254_poseidon2_compressing_row_hashes_ext_v3(
    uint64_t *out,
    const uint32_t *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
);

using HashFn = int (*)(
    uint64_t *out,
    const uint32_t *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query,
    cudaStream_t stream
);

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s (%d)\n", what, cudaGetErrorString(err), int(err));
        std::exit(1);
    }
}

static void check_cuda_code(int code, const char *what) {
    if (code != int(cudaSuccess)) {
        cudaError_t err = static_cast<cudaError_t>(code);
        std::fprintf(stderr, "%s failed: %s (%d)\n", what, cudaGetErrorString(err), code);
        std::exit(1);
    }
}

static bool checked_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static size_t checked_height() {
    if (kLogRowsPerQuery >= 8 * sizeof(size_t) ||
        kQueryStride > (std::numeric_limits<size_t>::max() >> kLogRowsPerQuery)) {
        std::fprintf(
            stderr,
            "height overflow for query_stride=%zu log_rows_per_query=%zu\n",
            kQueryStride,
            kLogRowsPerQuery
        );
        std::exit(1);
    }
    return kQueryStride << kLogRowsPerQuery;
}

static double bytes_to_gib(size_t bytes) { return double(bytes) / (1024.0 * 1024.0 * 1024.0); }

// Host-side Montgomery conversion, copied to keep this standalone nvcc benchmark
// independent of Rust constant generation.
using Fr = std::array<uint64_t, 4>;

static constexpr Fr kBn254Prime = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL,
};
static constexpr uint64_t kBn254Mu = 0x3d1e0a6c10000001ULL;
static constexpr Fr kBn254RSq = {
    0x1bb8e645ae216da7ULL,
    0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL,
    0x0216d0b17f4e44a5ULL,
};

static std::pair<uint64_t, Fr> mul_small(const Fr &lhs, uint64_t rhs) {
    __uint128_t acc = (__uint128_t)lhs[0] * rhs;
    uint64_t low = uint64_t(acc);
    acc >>= 64;

    Fr high{};
    for (size_t i = 1; i < 4; ++i) {
        acc += (__uint128_t)lhs[i] * rhs;
        high[i - 1] = uint64_t(acc);
        acc >>= 64;
    }
    high[3] = uint64_t(acc);
    return {low, high};
}

static std::pair<uint64_t, Fr> mul_small_and_acc(const Fr &lhs, uint64_t rhs, const Fr &add) {
    __uint128_t acc = (__uint128_t)lhs[0] * rhs + add[0];
    uint64_t low = uint64_t(acc);
    acc >>= 64;

    Fr high{};
    for (size_t i = 1; i < 4; ++i) {
        acc += (__uint128_t)lhs[i] * rhs + add[i];
        high[i - 1] = uint64_t(acc);
        acc >>= 64;
    }
    high[3] = uint64_t(acc);
    return {low, high};
}

static Fr imr(uint64_t acc0, const Fr &acc) {
    uint64_t t = acc0 * kBn254Mu;
    Fr u = mul_small(kBn254Prime, t).second;

    Fr sub{};
    uint64_t borrow = 0;
    for (size_t i = 0; i < 4; ++i) {
        uint64_t d = acc[i] - u[i];
        uint64_t b1 = acc[i] < u[i];
        sub[i] = d - borrow;
        uint64_t b2 = d < borrow;
        borrow = b1 + b2;
    }

    if (borrow == 0) {
        return sub;
    }

    Fr result{};
    uint64_t carry = 0;
    for (size_t i = 0; i < 4; ++i) {
        __uint128_t sum = (__uint128_t)sub[i] + kBn254Prime[i] + carry;
        result[i] = uint64_t(sum);
        carry = uint64_t(sum >> 64);
    }
    return result;
}

static Fr monty_mul(const Fr &lhs, const Fr &rhs) {
    auto [acc0, acc] = mul_small(lhs, rhs[0]);
    Fr res = imr(acc0, acc);

    auto [acc1, acc_1] = mul_small_and_acc(lhs, rhs[1], res);
    res = imr(acc1, acc_1);

    auto [acc2, acc_2] = mul_small_and_acc(lhs, rhs[2], res);
    res = imr(acc2, acc_2);

    auto [acc3, acc_3] = mul_small_and_acc(lhs, rhs[3], res);
    return imr(acc3, acc_3);
}

static Fr canonical_to_monty(uint64_t value) { return monty_mul(kBn254RSq, Fr{value, 0, 0, 0}); }

template <size_t N>
static void fill_monty_constants(std::array<uint64_t, N> &dst, uint64_t first_value) {
    static_assert(N % 4 == 0);
    for (size_t i = 0; i < N / 4; ++i) {
        Fr elem = canonical_to_monty(first_value + i);
        std::memcpy(dst.data() + i * 4, elem.data(), 4 * sizeof(uint64_t));
    }
}

static void init_round_constants(cudaStream_t stream) {
    std::array<uint64_t, 4 * 3 * 4> initial_w3{};
    std::array<uint64_t, 56 * 4> partial_w3{};
    std::array<uint64_t, 4 * 3 * 4> terminal_w3{};
    std::array<uint64_t, 3 * 2 * 4> initial_w2{};
    std::array<uint64_t, 50 * 4> partial_w2{};
    std::array<uint64_t, 3 * 2 * 4> terminal_w2{};

    fill_monty_constants(initial_w3, 1);
    fill_monty_constants(partial_w3, 1001);
    fill_monty_constants(terminal_w3, 2001);
    fill_monty_constants(initial_w2, 3001);
    fill_monty_constants(partial_w2, 4001);
    fill_monty_constants(terminal_w2, 5001);

    check_cuda_code(
        _init_bn254_poseidon2_rc(initial_w3.data(), partial_w3.data(), terminal_w3.data(), stream),
        "_init_bn254_poseidon2_rc"
    );
    check_cuda_code(
        _init_bn254_poseidon2_rc_w2(
            initial_w2.data(), partial_w2.data(), terminal_w2.data(), stream
        ),
        "_init_bn254_poseidon2_rc_w2"
    );
}

__global__ void fill_matrix_kernel(uint32_t *matrix, size_t len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < len; idx += stride) {
        uint32_t value = uint32_t((idx * 1664525ULL + 1013904223ULL) & 0x7fffffffULL);
        if (value >= kBabyBearPrime) {
            value -= kBabyBearPrime;
        }
        matrix[idx] = value;
    }
}

static void fill_matrix(uint32_t *matrix, size_t len, cudaStream_t stream) {
    constexpr int block = 256;
    size_t blocks = (len + block - 1) / block;
    int grid = int(std::min<size_t>(blocks, 65535));
    fill_matrix_kernel<<<grid, block, 0, stream>>>(matrix, len);
    check_cuda(cudaGetLastError(), "fill_matrix_kernel launch");
}

static constexpr int kWarmupIters = 3;
static constexpr int kTimedIters = 10;

template <typename F> static void benchmark(const char *label, cudaStream_t stream, F &&fn) {
    for (int i = 0; i < kWarmupIters; ++i) {
        fn();
    }
    check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    std::array<float, kTimedIters> samples{};
    for (int i = 0; i < kTimedIters; ++i) {
        check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
        fn();
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        check_cuda(cudaEventElapsedTime(&samples[i], start, stop), "cudaEventElapsedTime");
    }
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    double sum = 0.0;
    for (float s : samples) {
        sum += s;
    }
    double mean = sum / kTimedIters;
    double sq_sum = 0.0;
    for (float s : samples) {
        double d = double(s) - mean;
        sq_sum += d * d;
    }
    double stddev = std::sqrt(sq_sum / kTimedIters);

    std::printf(
        "%s: mean=%.3f ms stddev=%.3f ms (warmup=%d, n=%d)\n",
        label,
        mean,
        stddev,
        kWarmupIters,
        kTimedIters
    );
}

static void compare_outputs(
    const char *label,
    const uint64_t *d_cand,
    const uint64_t *d_ref,
    size_t out_u64
) {
    std::vector<uint64_t> cand(out_u64);
    std::vector<uint64_t> ref(out_u64);

    check_cuda(
        cudaMemcpy(cand.data(), d_cand, out_u64 * sizeof(uint64_t), cudaMemcpyDeviceToHost),
        "copy candidate output"
    );
    check_cuda(
        cudaMemcpy(ref.data(), d_ref, out_u64 * sizeof(uint64_t), cudaMemcpyDeviceToHost),
        "copy baseline output"
    );

    size_t mismatches = 0;
    for (size_t i = 0; i < out_u64; ++i) {
        if (cand[i] != ref[i]) {
            if (mismatches < 8) {
                std::fprintf(
                    stderr,
                    "%s mismatch digest=%zu limb=%zu: cand=0x%016" PRIx64 " baseline=0x%016" PRIx64
                    "\n",
                    label,
                    i / 4,
                    i % 4,
                    cand[i],
                    ref[i]
                );
            }
            ++mismatches;
        }
    }

    if (mismatches != 0) {
        std::fprintf(stderr, "%s outputs differ: %zu mismatched u64 limbs\n", label, mismatches);
        std::exit(1);
    }
    std::printf("%s outputs match (%zu digests, %zu u64 limbs)\n", label, out_u64 / 4, out_u64);
}

int main() {
    const size_t height = checked_height();
    size_t matrix_elems = 0;
    size_t matrix_bytes = 0;
    size_t out_u64 = 0;
    size_t out_bytes = 0;

    if (!checked_mul(kWidth, height, &matrix_elems) ||
        !checked_mul(matrix_elems, sizeof(uint32_t), &matrix_bytes) ||
        !checked_mul(kQueryStride, size_t{4}, &out_u64) ||
        !checked_mul(out_u64, sizeof(uint64_t), &out_bytes)) {
        std::fprintf(stderr, "allocation size overflow\n");
        return 1;
    }

    std::printf(
        "width=%zu height=%zu query_stride=%zu log_rows_per_query=%zu\n",
        kWidth,
        height,
        kQueryStride,
        kLogRowsPerQuery
    );
    std::printf(
        "matrix: %zu uint32_t values (%.3f GiB)\n", matrix_elems, bytes_to_gib(matrix_bytes)
    );
    std::printf("out: %zu uint64_t values (%.3f GiB)\n", out_u64, bytes_to_gib(out_bytes));

    if (kLogRowsPerQuery > 10) {
        std::fprintf(
            stderr,
            "note: the current launcher in bn254_poseidon2.cu rejects "
            "log_rows_per_query > 10; this benchmark keeps the requested value %zu.\n",
            kLogRowsPerQuery
        );
    }

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    init_round_constants(stream);

    uint32_t *d_matrix = nullptr;
    uint64_t *d_out_v2 = nullptr;
    uint64_t *d_out_v3 = nullptr;
    uint64_t *d_out_ref = nullptr;

    check_cuda(cudaMalloc(&d_matrix, matrix_bytes), "cudaMalloc(matrix)");
    check_cuda(cudaMalloc(&d_out_v2, out_bytes), "cudaMalloc(out_v2)");
    check_cuda(cudaMalloc(&d_out_v3, out_bytes), "cudaMalloc(out_v3)");
    check_cuda(cudaMalloc(&d_out_ref, out_bytes), "cudaMalloc(out_ref)");

    fill_matrix(d_matrix, matrix_elems, stream);
    check_cuda(cudaMemsetAsync(d_out_v2, 0, out_bytes, stream), "cudaMemsetAsync(out_v2)");
    check_cuda(cudaMemsetAsync(d_out_v3, 0, out_bytes, stream), "cudaMemsetAsync(out_v3)");
    check_cuda(cudaMemsetAsync(d_out_ref, 0, out_bytes, stream), "cudaMemsetAsync(out_ref)");
    check_cuda(cudaStreamSynchronize(stream), "initialization synchronize");

    auto run_hash = [&](const char *label, HashFn fn, uint64_t *out) {
        benchmark(label, stream, [&]() {
            check_cuda_code(
                fn(out, d_matrix, kWidth, kQueryStride, kLogRowsPerQuery, stream), label
            );
        });
    };

    run_hash(
        "_bn254_poseidon2_compressing_row_hashes_v3",
        _bn254_poseidon2_compressing_row_hashes_v3,
        d_out_v3
    );
    run_hash(
        "_bn254_poseidon2_compressing_row_hashes",
        _bn254_poseidon2_compressing_row_hashes,
        d_out_ref
    );

    compare_outputs("v3", d_out_v3, d_out_ref, out_u64);

    // ---- FpExt variants ----
    // FpExt = 4 BabyBear u32s per matrix entry → matrix is 4× the byte size.
    size_t ext_matrix_elems = 0;
    size_t ext_matrix_bytes = 0;
    if (!checked_mul(matrix_elems, size_t{4}, &ext_matrix_elems) ||
        !checked_mul(ext_matrix_elems, sizeof(uint32_t), &ext_matrix_bytes)) {
        std::fprintf(stderr, "ext allocation size overflow\n");
        return 1;
    }
    std::printf(
        "\next matrix: %zu uint32_t values (%.3f GiB)\n",
        ext_matrix_elems,
        bytes_to_gib(ext_matrix_bytes)
    );

    uint32_t *d_matrix_ext = nullptr;
    uint64_t *d_out_ext_v3 = nullptr;
    uint64_t *d_out_ext_ref = nullptr;
    check_cuda(cudaMalloc(&d_matrix_ext, ext_matrix_bytes), "cudaMalloc(matrix_ext)");
    check_cuda(cudaMalloc(&d_out_ext_v3, out_bytes), "cudaMalloc(out_ext_v3)");
    check_cuda(cudaMalloc(&d_out_ext_ref, out_bytes), "cudaMalloc(out_ext_ref)");

    fill_matrix(d_matrix_ext, ext_matrix_elems, stream);
    check_cuda(cudaMemsetAsync(d_out_ext_v3, 0, out_bytes, stream), "cudaMemsetAsync(out_ext_v3)");
    check_cuda(cudaMemsetAsync(d_out_ext_ref, 0, out_bytes, stream), "cudaMemsetAsync(out_ext_ref)");
    check_cuda(cudaStreamSynchronize(stream), "ext initialization synchronize");

    auto run_hash_ext = [&](const char *label, HashFn fn, uint64_t *out) {
        benchmark(label, stream, [&]() {
            check_cuda_code(
                fn(out, d_matrix_ext, kWidth, kQueryStride, kLogRowsPerQuery, stream), label
            );
        });
    };

    run_hash_ext(
        "_bn254_poseidon2_compressing_row_hashes_ext_v3",
        _bn254_poseidon2_compressing_row_hashes_ext_v3,
        d_out_ext_v3
    );
    run_hash_ext(
        "_bn254_poseidon2_compressing_row_hashes_ext",
        _bn254_poseidon2_compressing_row_hashes_ext,
        d_out_ext_ref
    );

    compare_outputs("ext_v3", d_out_ext_v3, d_out_ext_ref, out_u64);

    check_cuda(cudaFree(d_out_ext_ref), "cudaFree(out_ext_ref)");
    check_cuda(cudaFree(d_out_ext_v3), "cudaFree(out_ext_v3)");
    check_cuda(cudaFree(d_matrix_ext), "cudaFree(matrix_ext)");

    check_cuda(cudaFree(d_out_ref), "cudaFree(out_ref)");
    check_cuda(cudaFree(d_out_v3), "cudaFree(out_v3)");
    check_cuda(cudaFree(d_out_v2), "cudaFree(out_v2)");
    check_cuda(cudaFree(d_matrix), "cudaFree(matrix)");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}
