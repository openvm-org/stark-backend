/*
 * Source: https://github.com/scroll-tech/plonky3-gpu (private repo)
 * Status: BASED ON plonky3-gpu/gpu-backend/src/cuda/kernels/quotient.cu
 * Imported: 2025-01-25 by @gaxiom
 */

#include "codec.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#ifdef DEBUG
#include <cstdio>

// Helper function to print decoded information
__host__ __device__ void print_decoded_rule(uint32_t rule_idx, Rule encoded, DecodedRule rule) {
    printf("Rule[%d]: {%lx, %lx}\n", rule_idx, encoded.high, encoded.low);
    printf("    Operation: %d\n", rule.op);
    printf("    Is Constraint: %s\n", rule.is_constraint ? "true" : "false");

    printf(
        "    X Entry - Type: %d, part: %d, offset: %d, index: %d\n",
        rule.x.type,
        rule.x.part,
        rule.x.offset,
        rule.x.index
    );

    printf(
        "    Y Entry - Type: %d, part: %d, offset: %d, index: %d\n",
        rule.y.type,
        rule.y.part,
        rule.y.offset,
        rule.y.index
    );

    printf(
        "    Z Entry - Type: %d, offset: %d, index: %d\n", rule.z.type, rule.z.offset, rule.z.index
    );
}

#endif

__forceinline__ __device__ uint32_t bit_rev(uint32_t x, uint32_t n) {
    return __brev(x) >> (__clz(n) + 1);
}

/// LDE could have bigger height than quotient size, in that case
/// we need to bit_rev twice (first time for quotient size, second time for LDE height)
__device__ __forceinline__ FpExt evaluate_source(
    const SourceInfo &src,
    uint32_t q_row,           // quotient value row
    const Fp *d_preprocessed, // preprocessed LDE over Fp
    const uint64_t *d_main,   // array of partitioned main LDEs over Fp
    const Fp *d_permutation,  // permutation LDE over FpExt (see comments below)
    const FpExt *d_exposed,
    const Fp *d_public,
    const FpExt *d_challenge,
    const Fp *d_first,
    const Fp *d_last,
    const Fp *d_transition,
    FpExt *d_intermediate,
    const bool should_buffer,
    const uint64_t intermediate_stride,
    const uint32_t next_step,
    const uint32_t quotient_size,
    const uint32_t prep_height,
    const uint32_t main_height,
    const uint32_t perm_height
) {
    FpExt result = FpExt(0);
    switch (src.type) {
    case ENTRY_PREPROCESSED: {
        uint32_t q_row_idx = (q_row + src.offset * next_step) & (quotient_size - 1);
        if (quotient_size != prep_height) {
            q_row_idx = bit_rev(bit_rev(q_row_idx, quotient_size), prep_height);
        }
        result = FpExt(d_preprocessed[prep_height * src.index + q_row_idx]);
        break;
    }
    case ENTRY_MAIN: {
        uint32_t q_row_idx = (q_row + src.offset * next_step) & (quotient_size - 1);
        if (quotient_size != main_height) {
            q_row_idx = bit_rev(bit_rev(q_row_idx, quotient_size), main_height);
        }
        Fp *d_main_fp = (Fp *)d_main[src.part];
        result = FpExt(d_main_fp[main_height * src.index + q_row_idx]);
        break;
    }
    case ENTRY_PERMUTATION: {
        uint32_t q_row_idx = (q_row + src.offset * next_step) & (quotient_size - 1);
        if (quotient_size != perm_height) {
            q_row_idx = bit_rev(bit_rev(q_row_idx, quotient_size), perm_height);
        }
        // LDE over FpExt is stored as 4 LDEs over Fp
        // d_permutation[i][0] is **not** adjacent to d_permutation[i][1]
        result.elems[0] = d_permutation[perm_height * (4 * src.index + 0) + q_row_idx];
        result.elems[1] = d_permutation[perm_height * (4 * src.index + 1) + q_row_idx];
        result.elems[2] = d_permutation[perm_height * (4 * src.index + 2) + q_row_idx];
        result.elems[3] = d_permutation[perm_height * (4 * src.index + 3) + q_row_idx];
        break;
    }
    case ENTRY_PUBLIC:
        result = FpExt(d_public[src.index]);
        break;
    case ENTRY_CHALLENGE:
        result = d_challenge[src.index];
        break;
    case ENTRY_EXPOSED:
        result = d_exposed[src.index];
        break;
    case SRC_INTERMEDIATE:
        result = d_intermediate[intermediate_stride * src.index];
        break;
    case SRC_CONSTANT:
        result = FpExt(Fp(src.index));
        break;
    case SRC_IS_FIRST:
        result = FpExt(d_first[q_row]);
        break;
    case SRC_IS_LAST:
        result = FpExt(d_last[q_row]);
        break;
    case SRC_IS_TRANSITION:
        result = FpExt(d_transition[q_row]);
        break;
    default:
        // Handle error
        ;
    }
    if (should_buffer) {
        d_intermediate[intermediate_stride * src.buffer_idx] = result;
    }
    return result;
}

// In this kernel we have interemediates stored in global memory.
template <bool GLOBAL>
__global__ void cukernel_quotient(
    // output
    FpExt *__restrict__ d_quotient_values,
    // LDEs
    const Fp *__restrict__ d_preprocessed, // preprocessed LDE over Fp
    const uint64_t *__restrict__ d_main,   // array of partitioned main LDEs over Fp
    const Fp *__restrict__ d_permutation,  // permutation LDE over FpExt (see comments below)
    // public values, challenges, ...
    const FpExt *__restrict__ d_exposed,
    const Fp *__restrict__ d_public,
    const Fp *__restrict__ d_first,
    const Fp *__restrict__ d_last,
    const Fp *__restrict__ d_transition,
    const Fp *__restrict__ d_inv_zeroifier,
    const FpExt *__restrict__ d_challenge,
    const FpExt *__restrict__ d_alpha,
    // intermediates
    const FpExt *__restrict__ d_intermediates,
    // symbolic constraints (rules)
    const Rule *__restrict__ d_rules,
    const uint64_t num_rules,
    const uint64_t quotient_size,
    const uint32_t prep_height,
    const uint32_t main_height,
    const uint32_t perm_height,
    const uint64_t qdb_degree,
    const uint32_t num_rows_per_tile
) {
    uint32_t next_step = 1 << qdb_degree;
    uint64_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t task_stride = gridDim.x * blockDim.x;

    FpExt alpha = *d_alpha;
    FpExt *intermediates_ptr;
    uint64_t intermediate_stride;

    if constexpr (GLOBAL) {
        intermediates_ptr = (FpExt *)d_intermediates + task_offset;
        intermediate_stride = task_stride;
    } else {
        FpExt intermediates[10];
        intermediates_ptr = intermediates;
        intermediate_stride = 1;
    }

    for (uint32_t j = 0; j < num_rows_per_tile; j++) {
        uint64_t index = task_offset + j * task_stride;
        bool valid = index < quotient_size;

        if (valid) {
            FpExt accumulator = FpExt(0);
            for (uint32_t i = 0; i < num_rules; i++) {
                __syncthreads();
                Rule rule = d_rules[i];
                DecodedRule decoded_rule = decode_rule(rule);

                FpExt x = evaluate_source(
                    decoded_rule.x,
                    index,
                    d_preprocessed,
                    d_main,
                    d_permutation,
                    d_exposed,
                    d_public,
                    d_challenge,
                    d_first,
                    d_last,
                    d_transition,
                    intermediates_ptr,
                    decoded_rule.buffer_x,
                    intermediate_stride,
                    next_step,
                    quotient_size,
                    prep_height,
                    main_height,
                    perm_height
                );
                FpExt y = evaluate_source(
                    decoded_rule.y,
                    index,
                    d_preprocessed,
                    d_main,
                    d_permutation,
                    d_exposed,
                    d_public,
                    d_challenge,
                    d_first,
                    d_last,
                    d_transition,
                    intermediates_ptr,
                    decoded_rule.buffer_y,
                    intermediate_stride,
                    next_step,
                    quotient_size,
                    prep_height,
                    main_height,
                    perm_height
                );
                FpExt result = FpExt(0);
                switch (decoded_rule.op) {
                case OP_ADD:
                    result = x + y;
                    break;
                case OP_SUB:
                    result = x - y;
                    break;
                case OP_MUL:
                    x *= y; // gpu-field/src/baby_bear/cuda/fpext.h #L132
                    result += x;
                    break;
                case OP_NEG:
                    result = -x;
                    break;
                case OP_VAR:
                    result = x;
                    break;
                default:
                    assert(false);
                }

                if (decoded_rule.buffer_result) {
                    intermediates_ptr[decoded_rule.z_index * intermediate_stride] = result;
                }

                if (decoded_rule.is_constraint) {
                    accumulator *= alpha;
                    accumulator += result;
                }
            }

            accumulator *= d_inv_zeroifier[index];
            d_quotient_values[index] = accumulator;
        }
    }
}

__global__ void cukernel_quotient_selectors(
    Fp *first_row,
    Fp *last_row,
    Fp *transition,
    Fp *inv_zeroifier,
    const uint64_t log_n,
    const uint64_t coset_log_n,
    const uint32_t shift
) {
    uint64_t rate_bits = coset_log_n - log_n;
    uint64_t evals_size = 1 << rate_bits;
    uint64_t coset_size = 1 << coset_log_n;

    Fp point_gen(0);
    Fp denom(0);
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < coset_size) {
        Fp base = Fp(shift);

        // eval
        point_gen = TWO_ADIC_GENERATORS[rate_bits];
        Fp s_pow_n = pow(base, 1 << log_n);
        Fp eval = s_pow_n * pow(point_gen, idx % evals_size) - Fp(1);

        // xs
        point_gen = TWO_ADIC_GENERATORS[coset_log_n];
        Fp xs = base * pow(point_gen, idx);

        // first_row
        point_gen = TWO_ADIC_GENERATORS[log_n];
        denom = xs - pow(point_gen, 0);
        first_row[idx] = eval * inv(denom);

        // last_row
        point_gen = TWO_ADIC_GENERATORS[log_n];
        denom = xs - pow(point_gen, (1 << log_n) - 1);
        last_row[idx] = eval * inv(denom);

        // transition
        point_gen = TWO_ADIC_GENERATORS[log_n];
        Fp subgroup_last = inv(point_gen);
        transition[idx] = xs - subgroup_last;

        // inv_zeroifier
        inv_zeroifier[idx] = inv(eval);
    }
}

static const uint64_t TASK_SIZE = 65536;

extern "C" int _cukernel_quotient_selectors(
    Fp *first_row,
    Fp *last_row,
    Fp *transition,
    Fp *inv_zeroifier,
    const uint64_t log_n,
    const uint64_t coset_log_n,
    const uint32_t shift
) {
    assert(coset_log_n >= log_n);
    auto [grid, block] = kernel_launch_params(1 << coset_log_n, 256);
    cukernel_quotient_selectors<<<grid, block>>>(
        first_row, last_row, transition, inv_zeroifier, log_n, coset_log_n, shift
    );
    return cudaGetLastError();
}

extern "C" int _cukernel_quotient(
    bool is_global,
    FpExt *d_quotient_values,
    const Fp *d_preprocessed,
    const uint64_t *d_main,
    const Fp *d_permutation,
    const FpExt *d_exposed,
    const Fp *d_public,
    const Fp *d_first,
    const Fp *d_last,
    const Fp *d_transition,
    const Fp *d_inv_zeroifier,
    const FpExt *d_challenge,
    const FpExt *d_alpha,
    const FpExt *d_intermediates,
    const Rule *d_rules,
    const uint64_t num_rules,
    const uint32_t quotient_size,
    const uint32_t prep_height,
    const uint32_t main_height,
    const uint32_t perm_height,
    const uint64_t qdb_degree,
    const uint32_t num_rows_per_tile
) {
    auto [grid, block] = kernel_launch_params(TASK_SIZE, 256);

    #define QUOTIENT_ARGUMENTS \
        d_quotient_values, \
        d_preprocessed, \
        d_main, \
        d_permutation, \
        d_exposed, \
        d_public, \
        d_first, \
        d_last, \
        d_transition, \
        d_inv_zeroifier, \
        d_challenge, \
        d_alpha, \
        d_intermediates, \
        d_rules, \
        num_rules, \
        quotient_size, \
        prep_height, \
        main_height, \
        perm_height, \
        qdb_degree, \
        num_rows_per_tile

    if (is_global) {
        cukernel_quotient<true><<<grid, block>>>(QUOTIENT_ARGUMENTS);
    } else {
        cukernel_quotient<false><<<grid, block>>>(QUOTIENT_ARGUMENTS);
    }
    return cudaGetLastError();
}