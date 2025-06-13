// FROM https://github.com/scroll-tech/plonky3-gpu/blob/openvm-v2/gpu-backend/src/cuda/kernels/fri.cu

#include "fpext.h"
#include "launcher.cuh"

const uint32_t TILE_WIDTH = 32;

__forceinline__ __device__ uint32_t bit_rev(uint32_t x, uint32_t n) {
    return __brev(x) >> (__clz(n) + 1);
}

// result[i] = (1/2 + beta/2 g_inv^i) * folded[2*i]
//           + (1/2 - beta/2 g_inv^i) * folded[2*i+1]
//           + fri_input[i]
__global__ void cukernel_fri_fold(
    FpExt *result,
    FpExt *folded,
    const FpExt *fri_input,
    FpExt *d_constants,
    Fp *g_inv_powers,
    uint64_t N
) {
    FpExt half_beta = d_constants[0]; // beta/2
    FpExt half_one = d_constants[1];  // 1/2
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        FpExt beta_g_inv = half_beta * g_inv_powers[idx]; // beta/2 * g_inv^i
        FpExt c1 = half_one + beta_g_inv;                 // 1/2 + beta/2 * g_inv^i
        FpExt c2 = half_one - beta_g_inv;                 // 1/2 - beta/2 * g_inv^i
        FpExt a = folded[2 * idx];
        FpExt b = folded[2 * idx + 1];
        FpExt res = c1 * a;
        res += c2 * b;
        if (fri_input != nullptr) {
            res += fri_input[idx];
        }
        result[idx] = res;
    }
}

// compute diffs = { (z - shift*g^j) } for j in 0..N
__global__ void compute_diffs(FpExt *diffs, FpExt *d_z, Fp *d_domain, uint32_t log_n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    Fp shift = d_domain[0];
    Fp g = d_domain[1];
    uint32_t N = 1 << log_n;
    Fp g_idx = pow(g, idx);
    Fp g_pow = pow(g, stride);
    FpExt z = *d_z;

    for (; idx < N; idx += stride, g_idx *= g_pow) {
        FpExt diff = z - shift * g_idx;
        diffs[idx] = diff;
    }
}

// data[i] = g^i for i in 0..N
__global__ void powers(Fp *data, Fp *d_g, uint32_t N) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    Fp g = *d_g;
    Fp g_idx = pow(g, idx);
    Fp g_pow = pow(g, stride);

    for (; idx < N; idx += stride, g_idx *= g_pow) {
        data[idx] = g_idx;
    }
}

__global__ void powers_ext(FpExt *data, FpExt *d_g, uint32_t N) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    FpExt g = *d_g;
    FpExt g_idx = pow(g, idx);
    FpExt g_pow = pow(g, stride);

    for (; idx < N; idx += stride, g_idx *= g_pow) {
        data[idx] = g_idx;
    }
}

__global__ void precompute_diff_powers(
    FpExt *d_output,
    const FpExt *diff_invs,
    const Fp *powers,
    uint32_t N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; idx < N; idx += stride) {
        if (idx < N) {
            d_output[idx] = diff_invs[idx] * powers[idx];
        }
    }
}

__global__ void fri_bit_reverse(FpExt *data, uint32_t log_n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    uint32_t N = 1 << log_n;
    for (; idx < N; idx += stride) {
        uint32_t ridx = bit_rev(idx, N);
        if (idx < ridx) {
            FpExt tmp = data[idx];
            data[idx] = data[ridx];
            data[ridx] = tmp;
        }
    }
}

// batch inversion
__global__ void batch_invert(FpExt *data, uint64_t log_n) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    const uint32_t batch_size = 16;

    uint32_t N = 1ULL << log_n;
    if (idx >= N) {
        return;
    }

    // a, ab, abc, abcd, ...
    FpExt accums[batch_size];
    accums[0] = data[idx];

    uint32_t j = 1;
    uint32_t pos = idx + stride;
    for (; (j < batch_size) && pos < N; pos += stride, j += 1) {
        accums[j] = accums[j - 1] * data[pos];
    }

    j -= 1;
    pos -= stride;
    // accum_inv = inv(prod(data[idx], data[idx+stride], ...,
    // data[idx+(j-1)*stride]))
    FpExt accum_inv = binomial_inversion(accums[j]);

    for (; j > 0; pos -= stride, j -= 1) {
        FpExt tmp = accum_inv * accums[j - 1]; // inv(data[pos])
        accum_inv *= data[pos];
        data[pos] = tmp;
    }

    data[idx] = accum_inv;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// recall that barycentric algorithm for getting the evaluation of polynomial
/// p(x) at a random point z given its evaluations over coset domain s*H where
/// H = { g^j | j in 0..(N-1) } and N is a power of 2.
///
///   1. p(z) = M(z) * \sum_j p(s*g^j) / (dj * (z - s*g^j))
///   2. dj = \prod_{k != j} (s*g^j - s*g^k)
///         = s^{N-1} * \prod_{k != j} (g^j - g^k)
///         = s^{N-1} * N/g^j
///   3. M(z) = \prod_j (z - s*g^j) = z^N - s^N
///
/// therefore, we have
///   p(z) = (M(z)/(s^{N-1}*N)) * \sum_j p(s*g^j) * (g^j / (z-s*g^j))
///        = c * dot_product([p(s*g^j)/(z-s*g^j)],  [g^j])
///   where c = M(z)/(s^{N-1}*N)
///
///
/// we can generalize this formula to get the random linear combination of
/// evaluations of m_i(x) at z. where `m` is a matrix and `m_i(x)` is the i-th
/// column of `m`.
///
/// therefore, we have
///     m(z) = \sum alpha^i * m_i(z)
///          = c * \sum_j [\sum alpha^i * m_i(s*g^j)/(z-s*g^j)] * g^j
///          = c * dot_product([m_rlc], [g^j])
///
/// the evaluation of m(z) is done in three steps:
/// 1. let m_rlc be a vector of size N such that m_rlc[j] = \sum alpha^i *
/// m_i(s*g^j) / (z-s*g^j)
/// 2. let g_powers be a vector of size N such that g_powers[j] = g^j
/// 3. m(z) = c * dot_product(m_rlc, g_powers).
///
/// the above method requires that `m`, `z_diff_invs` and `g_powers` have same
/// order.
//////////////////////////////////////////////////////////////////////////////////////////////

// 1. scale_vec(matrix.rows[i], g_powers[i] * diff_invs[i])
// 2. sum(matrix.rows[i]...matrix.rows[i+TILE_WIDTH])
// this kernel returns the evaluation of each polynomial m_i(x) at z.

static const uint32_t REDUCTION_THREADS_PER_BLOCK = 256;

/// matrix could have bigger height than domain size
/// in this case we shoud do bit_rev twice (first time for domain height, second time for matrix
/// height)
__global__ void matrix_scale_rows_then_reduce(
    FpExt *o_matrix,
    Fp *matrix,
    FpExt *diff_invs_dot_g_powers,
    uint32_t width,
    uint32_t matrix_height,
    uint32_t domain_height
) {
    uint32_t tid = threadIdx.x;
    uint64_t row_index = blockIdx.x * blockDim.x + tid;
    uint64_t col_index = blockIdx.y;

    __shared__ FpExt s_mem[REDUCTION_THREADS_PER_BLOCK];
    s_mem[threadIdx.x] = FpExt(0);

    FpExt sum = FpExt(0);
    uint64_t row_stride = gridDim.x * blockDim.x;
    for (uint32_t i = row_index; i < domain_height; i += row_stride) {
        if (i < domain_height) {
            uint32_t br_i = (domain_height == matrix_height)
                                ? i
                                : bit_rev(bit_rev(i, domain_height), matrix_height);
            sum += FpExt(matrix[col_index * matrix_height + br_i]) * diff_invs_dot_g_powers[i];
        }
    }
    s_mem[tid] = sum;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mem[tid] += s_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        uint64_t out_height_size = gridDim.x;
        uint64_t out_height_index = blockIdx.x;
        o_matrix[col_index * out_height_size + out_height_index] = s_mem[0];
    }
}

__global__ void matrix_reduce(
    FpExt *o_matrix,
    FpExt *i_matrix,
    uint32_t width,
    uint32_t current_height, // the involved height of the matrix, in current round
    uint32_t buffer_height   // the original height of the matrix
) {
    uint32_t tid = threadIdx.x;
    uint64_t row_index = blockIdx.x * blockDim.x + tid;
    uint64_t col_index = blockIdx.y;

    __shared__ FpExt s_mem[REDUCTION_THREADS_PER_BLOCK];
    s_mem[threadIdx.x] = FpExt(0);

    FpExt sum = FpExt(0);
    uint64_t row_stride = gridDim.x * blockDim.x;
    for (uint64_t i = row_index; i < current_height; i += row_stride) {
        if (i < current_height) {
            sum += i_matrix[col_index * buffer_height + i];
        }
    }
    s_mem[tid] = sum;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mem[tid] += s_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        uint64_t out_height_index = blockIdx.x;
        o_matrix[col_index * buffer_height + out_height_index] = s_mem[0];
    }
}

__global__ void matrix_get_first_column(
    FpExt *o_col,
    FpExt *i_matrix,
    uint32_t width,
    uint32_t height
) {
    uint32_t gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx < width) {
        o_col[gidx] = i_matrix[gidx * height];
    }
}

// the quotient polynomial of [m(z) - \sum alpha^i * m_i(x)] divide by (z - x)is
// given by its evaluations at coset domain s*H.
//
// qm(x) = \sum alpha^i * \sum_j (m_i(z)-m_i(x)) / (z-x)
//       = 1/(z-x) * \sum alpha^i * (m_i(z) - m_i(x))
//       = 1/(z-x) * [\sum alpha^i * m_i(z) - \sum alpha^i * m_i(z)]
//       = m(z) / (z-x) - [\sum alpha^i * m_i(x)/(z-x)]
//
// key observation: m_rlc[j] is `\sum alpha^i * m_i(s*g^j)/(z-s*g^j)`.
// therefore, we can reused the result of kernel `matrix_interpolate_coset`.
//
// qm(x) = m(z) / (z-x) - m_rlc(x)
//
// this kernel requires that `acc`, `z_diff_invs` and `m_rlc` have same order.
// for each row, this kernel computes m_rlc[j] = \sum alpha^i * m_i(s*g^j)
__global__ void reduce_matrix_quotient_acc(
    FpExt *quotient_acc,
    Fp *matrix,
    FpExt *z_diff_invs,
    const FpExt *matrix_eval,
    FpExt *d_alphas,
    FpExt *d_alphas_offset,
    uint32_t width,
    uint32_t height,
    bool is_first
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx >= height) {
        return;
    }

    FpExt accum = {0, 0, 0, 0};

    // matrix has a natural order, but all other arrays are bit_reversed
    // so we need to bit_rev when read
    uint32_t br_row_idx = bit_rev(row_idx, height);
    for (uint32_t col_idx = 0; col_idx < width; col_idx++) {
        if (col_idx < width) {
            accum += d_alphas[col_idx] * matrix[col_idx * height + br_row_idx];
        }
    }

    FpExt mz = *matrix_eval;
    FpExt alpha_offset = *d_alphas_offset; // alpha^matrix_offset
    FpExt quotient = alpha_offset * z_diff_invs[row_idx] * (mz - accum);
    if (is_first) {
        quotient_acc[row_idx] = quotient;
    } else {
        quotient_acc[row_idx] += quotient;
    }
}

__global__ void cukernel_split_ext_poly_to_base_col_major_matrix(
    Fp *d_matrix,
    FpExt *d_poly,
    uint64_t poly_len,
    uint32_t matrix_height
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= matrix_height) {
        return;
    }

    // d_poly is bit_reversed, so we need to bit_rev when write to keep the natural order
    uint32_t br_row_idx = bit_rev(row_idx, matrix_height);
    uint32_t col_num = (poly_len / matrix_height); // SPLIT_FACTOR = 2
    for (uint32_t col_idx = 0; col_idx < col_num; col_idx++) {
        FpExt ext_val = d_poly[row_idx * col_num + col_idx];
        d_matrix[(col_idx * 4 + 0) * matrix_height + br_row_idx] = ext_val.elems[0];
        d_matrix[(col_idx * 4 + 1) * matrix_height + br_row_idx] = ext_val.elems[1];
        d_matrix[(col_idx * 4 + 2) * matrix_height + br_row_idx] = ext_val.elems[2];
        d_matrix[(col_idx * 4 + 3) * matrix_height + br_row_idx] = ext_val.elems[3];
    }
}

// END OF gpu-backend/src/cuda/kernels/fri.cu

static const size_t FRI_MAX_THREADS = 256;
int get_num_sms() {
    static int multiprocessorCount = []() {
        cudaDeviceProp prop;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        return prop.multiProcessorCount;
    }();
    return multiprocessorCount;
}

extern "C" int _compute_diffs(FpExt *diffs, FpExt *d_z, Fp *d_domain, uint32_t log_max_height) {
    auto block = FRI_MAX_THREADS;
    auto grid = get_num_sms() * 2;
    compute_diffs<<<grid, block>>>(diffs, d_z, d_domain, log_max_height);
    return cudaGetLastError();
}

extern "C" int _fri_bit_reverse(FpExt *diffs, uint32_t log_max_height) {
    auto block = FRI_MAX_THREADS;
    auto grid = get_num_sms() * 2;
    fri_bit_reverse<<<grid, block>>>(diffs, log_max_height);
    return cudaGetLastError();
}

extern "C" int _batch_invert(FpExt *diffs, uint32_t log_max_height, uint32_t invert_task_num) {
    auto [grid, block] = kernel_launch_params(invert_task_num, FRI_MAX_THREADS);
    batch_invert<<<grid, block>>>(diffs, log_max_height);
    return cudaGetLastError();
}

extern "C" int _powers(Fp *data, Fp *g, uint32_t N) {
    auto block = FRI_MAX_THREADS;
    auto grid = get_num_sms() * 2;
    powers<<<grid, block>>>(data, g, N);
    return cudaGetLastError();
}

extern "C" int _powers_ext(FpExt *data, FpExt *g, uint32_t N) {
    auto block = FRI_MAX_THREADS;
    auto grid = get_num_sms() * 2;
    powers_ext<<<grid, block>>>(data, g, N);
    return cudaGetLastError();
}

extern "C" int _precompute_diff_powers(
    FpExt *d_output,
    const FpExt *diff_invs,
    const Fp *powers,
    uint32_t N
) {
    auto [grid, block] = kernel_launch_params(N, FRI_MAX_THREADS);
    precompute_diff_powers<<<grid, block>>>(d_output, diff_invs, powers, N);
    return cudaGetLastError();
}

extern "C" int _matrix_scale_rows_then_reduce(
    FpExt *o_matrix,
    Fp *matrix,
    FpExt *diff_invs_dot_g_powers,
    uint32_t width,
    uint32_t matrix_height,
    uint32_t domain_height,
    uint32_t reduce_matrix_height
) {
    dim3 grid = dim3(reduce_matrix_height, width);
    auto block = REDUCTION_THREADS_PER_BLOCK;
    matrix_scale_rows_then_reduce<<<grid, block>>>(
        o_matrix, matrix, diff_invs_dot_g_powers, width, matrix_height, domain_height
    );
    return cudaGetLastError();
}

extern "C" int _matrix_reduce(
    FpExt *o_matrix,
    FpExt *i_matrix,
    uint32_t width,
    uint32_t current_height,
    uint32_t buffer_height,
    uint32_t next_round_height
) {
    dim3 grid = dim3(next_round_height, width);
    auto block = REDUCTION_THREADS_PER_BLOCK;
    matrix_reduce<<<grid, block>>>(o_matrix, i_matrix, width, current_height, buffer_height);
    return cudaGetLastError();
}

extern "C" int _matrix_get_first_column(
    FpExt *o_col,
    FpExt *i_matrix,
    uint32_t width,
    uint32_t height
) {
    auto [grid, block] = kernel_launch_params(width, FRI_MAX_THREADS);
    matrix_get_first_column<<<grid, block>>>(o_col, i_matrix, width, height);
    return cudaGetLastError();
}

extern "C" int _reduce_matrix_quotient_acc(
    FpExt *quotient_acc,
    Fp *matrix,
    FpExt *z_diff_invs,
    const FpExt *matrix_eval,
    FpExt *d_alphas,
    FpExt *d_alphas_offset,
    uint32_t width,
    uint32_t height,
    bool is_first
) {
    auto [grid, block] = kernel_launch_params(height, TILE_WIDTH);
    reduce_matrix_quotient_acc<<<grid, block>>>(
        quotient_acc,
        matrix,
        z_diff_invs,
        matrix_eval,
        d_alphas,
        d_alphas_offset,
        width,
        height,
        is_first
    );
    return cudaGetLastError();
}

extern "C" int _cukernel_split_ext_poly_to_base_col_major_matrix(
    Fp *d_matrix,
    FpExt *d_poly,
    uint64_t poly_len,
    uint32_t matrix_height
) {
    auto [grid, block] = kernel_launch_params(matrix_height, FRI_MAX_THREADS);
    cukernel_split_ext_poly_to_base_col_major_matrix<<<grid, block>>>(
        d_matrix, d_poly, poly_len, matrix_height
    );
    return cudaGetLastError();
}

extern "C" int _cukernel_fri_fold(
    FpExt *result,
    FpExt *folded,
    const FpExt *fri_input,
    FpExt *d_constants,
    Fp *g_invs,
    uint64_t N
) {
    auto [grid, block] = kernel_launch_params(N, FRI_MAX_THREADS);
    cukernel_fri_fold<<<grid, block>>>(result, folded, fri_input, d_constants, g_invs, N);
    return cudaGetLastError();
}