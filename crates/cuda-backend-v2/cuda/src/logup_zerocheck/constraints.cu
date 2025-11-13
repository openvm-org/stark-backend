#include <cstdio>
#include <cstdlib>

#include "codec.cuh"
#include "fp.h"
#include "fpext.h"
#include "frac_ext.cuh"
#include "launcher.cuh"
#include "dag_entry.cuh"

using namespace logup_round0;

// Device function equivalent to helper.acc_constraints without eq_* parts
// This computes the constraint sum: sum(lambda_i * constraint_i) for all constraints
template<bool GLOBAL>
__device__ __forceinline__ FpExt acc_constraints(
    uint32_t row,
    const Fp * __restrict__ d_selectors,
    const MainMatrixPtrs * __restrict__ d_main,
    uint32_t height,
    uint32_t selectors_width,
    const Fp * __restrict__ d_preprocessed,
    uint32_t preprocessed_width,
    const FpExt * __restrict__ d_eq_z,
    const FpExt * __restrict__ d_eq_x,
    const Fp * __restrict__ d_public,
    uint32_t public_len,
    const FpExt * __restrict__ d_lambda_pows,
    const uint32_t * __restrict__ d_lambda_indices,
    const Rule * __restrict__ d_rules,
    size_t rules_len,
    const size_t * __restrict__ d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    FpExt * __restrict__ inter_buffer,
    FpExt * __restrict__ local_buffer,
    uint32_t buffer_stride,
    uint32_t large_domain
) {
    size_t lambda_idx = 0;
    FpExt constraint_sum(Fp::zero());

    for (size_t node = 0; node < rules_len; ++node) {
        Rule rule = d_rules[node];
        DecodedRule decoded = decode_rule(rule);

        FpExt x_val = evaluate_dag_entry(
            decoded.x,
            row,
            d_selectors,
            d_main,
            height,
            selectors_width,
            d_preprocessed,
            preprocessed_width,
            d_eq_z,
            d_eq_x,
            d_public,
            public_len,
            inter_buffer,
            buffer_stride,
            buffer_size,
            large_domain
        );
        FpExt result;
        switch (decoded.op) {
        case OP_ADD: {
            FpExt y_val = evaluate_dag_entry(
                decoded.y,
                row,
                d_selectors,
                d_main,
                height,
                selectors_width,
                d_preprocessed,
                preprocessed_width,
                d_eq_z,
                d_eq_x,
                d_public,
                public_len,
                inter_buffer,
                buffer_stride,
                buffer_size,
                large_domain
            );
            result = x_val + y_val;
            break;
        }
        case OP_SUB: {
            FpExt y_val = evaluate_dag_entry(
                decoded.y,
                row,
                d_selectors,
                d_main,
                height,
                selectors_width,
                d_preprocessed,
                preprocessed_width,
                d_eq_z,
                d_eq_x,
                d_public,
                public_len,
                inter_buffer,
                buffer_stride,
                buffer_size,
                large_domain
            );
            result = x_val - y_val;
            break;
        }
        case OP_MUL: {
            FpExt y_val = evaluate_dag_entry(
                decoded.y,
                row,
                d_selectors,
                d_main,
                height,
                selectors_width,
                d_preprocessed,
                preprocessed_width,
                d_eq_z,
                d_eq_x,
                d_public,
                public_len,
                inter_buffer,
                buffer_stride,
                buffer_size,
                large_domain
            );
            result = x_val * y_val;
            break;
        }
        case OP_NEG:
            result = -x_val;
            break;
        case OP_VAR:
            result = x_val;
            break;
        case OP_INV:
            result = inv(x_val);
            break;
        }

        if (decoded.buffer_result && buffer_size > 0) {
            if constexpr (GLOBAL) {
                inter_buffer[decoded.z_index * buffer_stride] = result;
            } else {
                local_buffer[decoded.z_index] = result;
            }
        }

        if (decoded.is_constraint) {
            while (lambda_idx < lambda_len
                && lambda_idx < used_nodes_len
                && d_used_nodes[lambda_idx] == node)
            {
                uint32_t mapped_idx = d_lambda_indices != nullptr
                    ? d_lambda_indices[lambda_idx]
                    : static_cast<uint32_t>(lambda_idx);
                FpExt lambda = d_lambda_pows[mapped_idx];
                lambda_idx++;
                constraint_sum += lambda * result;
            }
        }
    }

    return constraint_sum;
}

template<bool GLOBAL>
__global__ void evaluate_constraints_kernel(
    FpExt * __restrict__ d_output,
    const Fp * __restrict__ d_selectors,
    uint32_t selectors_width,
    const MainMatrixPtrs * __restrict__ d_main,
    uint32_t main_count,
    const Fp * __restrict__ d_preprocessed,
    uint32_t preprocessed_width,
    const FpExt * __restrict__ d_eq_z,
    const FpExt * __restrict__ d_eq_x,
    const FpExt * __restrict__ d_lambda_pows,
    const uint32_t * __restrict__ d_lambda_indices,
    const Fp * __restrict__ d_public,
    uint32_t public_len,
    const Rule * __restrict__ d_rules,
    size_t rules_len,
    const size_t * __restrict__ d_used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    FpExt * __restrict__ d_intermediates,
    uint32_t large_domain,
    uint32_t num_x,
    uint32_t num_rows_per_tile,
    uint32_t skip_stride
) {
    uint32_t task_offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t task_stride = gridDim.x * blockDim.x;

    FpExt local_buffer[16];
    FpExt *inter_buffer;
    uint32_t buffer_stride;
    if constexpr (GLOBAL) {
        inter_buffer = d_intermediates + task_offset;
        buffer_stride = task_stride;
    } else {
        inter_buffer = local_buffer;
        buffer_stride = 1;
    }

    uint32_t height = large_domain * num_x;

    for (uint32_t tile = 0; tile < num_rows_per_tile; ++tile) {
        uint32_t row = task_offset + tile * task_stride;
        if (row >= height) {
            continue;
        }

        uint32_t z_idx = row % large_domain;
        uint32_t x_idx = row / large_domain;

        if (buffer_size > 0) {
            if constexpr (GLOBAL) {
                for (uint32_t idx = 0; idx < buffer_size; ++idx) {
                    inter_buffer[idx * buffer_stride] = FpExt(Fp::zero());
                }
            } else {
                uint32_t limit = buffer_size < 16 ? buffer_size : 16;
                for (uint32_t idx = 0; idx < limit; ++idx) {
                    local_buffer[idx] = FpExt(Fp::zero());
                }
            }
        }

        FpExt eq_val = d_eq_z[z_idx] * d_eq_x[x_idx];

        FpExt constraint_sum = acc_constraints<GLOBAL>(
            row,
            d_selectors,
            d_main,
            height,
            selectors_width,
            d_preprocessed,
            preprocessed_width,
            d_eq_z,
            d_eq_x,
            d_public,
            public_len,
            d_lambda_pows,
            d_lambda_indices,
            d_rules,
            rules_len,
            d_used_nodes,
            used_nodes_len,
            lambda_len,
            buffer_size,
            inter_buffer,
            local_buffer,
            buffer_stride,
            large_domain
        );

        FpExt final_val = constraint_sum * eq_val;
        d_output[row] = final_val;
    }
}

__global__ void aggregate_constraints_kernel(
    const FpExt * __restrict__ d_output,
    FpExt * __restrict__ d_sums,
    uint32_t large_domain,
    uint32_t num_x
) {
    uint32_t z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= large_domain) {
        return;
    }

    FpExt acc(Fp::zero());
    for (uint32_t x = 0; x < num_x; ++x) {
        acc = acc + d_output[x * large_domain + z];
    }
    d_sums[z] = acc;
}

__global__ void extract_component_kernel(
    const FpExt * __restrict__ input,
    Fp * __restrict__ output,
    uint32_t len,
    uint32_t component
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    output[idx] = input[idx].elems[component];
}

__global__ void assign_component_kernel(
    const Fp * __restrict__ input,
    FpExt * __restrict__ output,
    uint32_t len,
    uint32_t component
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    FpExt value = output[idx];
    value.elems[component] = input[idx];
    output[idx] = value;
}

extern "C" int _zerocheck_eval_constraints(
    FpExt *output,
    const Fp *selectors,
    uint32_t selectors_width,
    const MainMatrixPtrs *partitioned_main,
    uint32_t main_count,
    const Fp *preprocessed,
    uint32_t preprocessed_width,
    const FpExt *eq_z,
    const FpExt *eq_x,
    const FpExt *lambda_pows,
    const uint32_t *lambda_indices,
    const Fp *public_values,
    uint32_t public_len,
    const Rule *rules,
    size_t rules_len,
    const size_t *used_nodes,
    size_t used_nodes_len,
    size_t lambda_len,
    uint32_t buffer_size,
    FpExt *intermediates,
    uint32_t large_domain,
    uint32_t num_x,
    uint32_t num_rows_per_tile,
    uint32_t skip_stride
) {
    auto [grid, block] = kernel_launch_params(large_domain * num_x, 256);
#ifdef CUDA_DEBUG
    if (std::getenv("LOGUP_GPU_SINGLE_THREAD") != nullptr) {
        grid = dim3(1, 1, 1);
        block = dim3(1, 1, 1);
    }
#endif
    if (buffer_size > 16) {
        evaluate_constraints_kernel<true><<<grid, block>>>(
            output,
            selectors,
            selectors_width,
            partitioned_main,
            main_count,
            preprocessed,
            preprocessed_width,
            eq_z,
            eq_x,
            lambda_pows,
            lambda_indices,
            public_values,
            public_len,
            rules,
            rules_len,
            used_nodes,
            used_nodes_len,
            lambda_len,
            buffer_size,
            intermediates,
            large_domain,
            num_x,
            num_rows_per_tile,
            skip_stride
        );
    } else {
        evaluate_constraints_kernel<false><<<grid, block>>>(
            output,
            selectors,
            selectors_width,
            partitioned_main,
            main_count,
            preprocessed,
            preprocessed_width,
            eq_z,
            eq_x,
            lambda_pows,
            lambda_indices,
            public_values,
            public_len,
            rules,
            rules_len,
            used_nodes,
            used_nodes_len,
            lambda_len,
            buffer_size,
            intermediates,
            large_domain,
            num_x,
            num_rows_per_tile,
            skip_stride
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _accumulate_constraints(
    const FpExt *output,
    FpExt *sums,
    uint32_t large_domain,
    uint32_t num_x
) {
    if (large_domain == 0) {
        return 0;
    }

    auto [grid, block] = kernel_launch_params(large_domain, 256);
    aggregate_constraints_kernel<<<grid, block>>>(
        output,
        sums,
        large_domain,
        num_x
    );
    return CHECK_KERNEL();
}

extern "C" int _extract_component(
    const FpExt *input,
    Fp *output,
    uint32_t len,
    uint32_t component
) {
    auto [grid, block] = kernel_launch_params(len, 256);
    extract_component_kernel<<<grid, block>>>(
        input,
        output,
        len,
        component
    );
    return CHECK_KERNEL();
}

extern "C" int _assign_component(
    const Fp *input,
    FpExt *output,
    uint32_t len,
    uint32_t component
) {
    auto [grid, block] = kernel_launch_params(len, 256);
    assign_component_kernel<<<grid, block>>>(
        input,
        output,
        len,
        component
    );
    return CHECK_KERNEL();
}



