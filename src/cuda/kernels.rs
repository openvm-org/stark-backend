#![allow(clippy::missing_safety_doc)]
#![allow(dead_code)]

use crate::cuda::{
    d_buffer::DeviceBuffer,
    error::{CudaError, KernelError},
};

pub const FP_SIZE: usize = 4; // sizeof(uint32_t)
pub const FP_EXT_SIZE: usize = 16; // 4 * sizeof(uint32_t)

// relate to matrix.cu
pub mod matrix {
    use super::*;

    extern "C" {
        fn matrix_transpose_fp(
            output: *mut std::ffi::c_void,
            input: *const std::ffi::c_void,
            col_size: usize,
            row_size: usize,
        ) -> i32;

        fn matrix_transpose_fpext(
            output: *mut std::ffi::c_void,
            input: *const std::ffi::c_void,
            col_size: usize,
            row_size: usize,
        ) -> i32;

        fn _split_ext_poly_to_multiple_base_matrix(
            d_matrix_ptr: *const std::ffi::c_void,
            d_poly: *mut std::ffi::c_void,
            poly_len: u64,
            num_chunk: u64,
        ) -> i32;

        fn _matrix_get_rows_fp(
            output: *mut std::ffi::c_void,
            input: *const std::ffi::c_void,
            row_indices: *const u32,
            matrix_width: u64,
            matrix_height: u64,
            row_indices_len: u32,
        ) -> i32;

        fn _unpack_matrix(
            output: *mut std::ffi::c_void,
            input: *const std::ffi::c_void,
            input_height: usize,
            input_width: usize,
        ) -> i32;
    }

    pub unsafe fn matrix_transpose<T>(
        output: &DeviceBuffer<T>,
        input: &DeviceBuffer<T>,
        width: usize,
        height: usize,
    ) -> Result<(), KernelError> {
        let size = std::mem::size_of::<T>();
        let result = match size {
            FP_SIZE => {
                matrix_transpose_fp(output.as_mut_raw_ptr(), input.as_raw_ptr(), width, height)
            }
            FP_EXT_SIZE => {
                matrix_transpose_fpext(output.as_mut_raw_ptr(), input.as_raw_ptr(), width, height)
            }
            _ => return Err(KernelError::UnsupportedTypeSize { size }),
        };

        CudaError::from_result(result).map_err(KernelError::from)
    }

    pub unsafe fn split_ext_poly_to_multiple_base_matrix<EF>(
        d_matrix_buf: &DeviceBuffer<u64>,
        d_poly_buf: &DeviceBuffer<EF>,
        poly_len: u64,
        num_chunk: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_split_ext_poly_to_multiple_base_matrix(
            d_matrix_buf.as_raw_ptr(),
            d_poly_buf.as_mut_raw_ptr(),
            poly_len,
            num_chunk,
        ))
    }

    pub unsafe fn matrix_get_rows_fp_kernel<F>(
        output: &DeviceBuffer<F>,
        input: &DeviceBuffer<F>,
        row_indices: &DeviceBuffer<u32>,
        matrix_width: u64,
        matrix_height: u64,
        row_indices_len: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_matrix_get_rows_fp(
            output.as_mut_raw_ptr(),
            input.as_raw_ptr(),
            row_indices.as_ptr(),
            matrix_width,
            matrix_height,
            row_indices_len,
        ))
    }

    pub unsafe fn unpack_matrix<F, EF>(
        output: &DeviceBuffer<F>,
        input: &DeviceBuffer<EF>,
        input_height: usize,
        input_width: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_unpack_matrix(
            output.as_mut_raw_ptr(),
            input.as_raw_ptr(),
            input_height,
            input_width,
        ))
    }
}

// relate to supra_ntt_api.cu
pub mod ntt {
    use super::*;
    extern "C" {
        fn _sppark_init(device_idx: u32) -> i32;

        fn _batch_NTT(
            inout: *mut std::ffi::c_void,
            lg_domain_size: u32,
            poly_count: u32,
            device_idx: u32,
        ) -> i32;

        fn _batch_iNTT(
            inout: *mut std::ffi::c_void,
            lg_domain_size: u32,
            lg_blowup: u32,
            poly_count: u32,
            device_idx: u32,
        ) -> i32;
    }

    pub unsafe fn sppark_init(device_idx: u32) -> Result<(), CudaError> {
        CudaError::from_result(_sppark_init(device_idx))
    }

    pub unsafe fn batch_ntt<T>(
        inout: &DeviceBuffer<T>,
        lg_domain_size: u32,
        poly_count: u32,
        device_idx: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_batch_NTT(
            inout.as_mut_raw_ptr(),
            lg_domain_size,
            poly_count,
            device_idx,
        ))
    }

    /// batch inverse NTT on polynomials of degree `2^lg_domain_size` but where polynomials
    /// are placed in a buffer where each polynomial has `2^{lg_domain_size + lg_blowup}` field
    /// elements allocated for it.
    pub unsafe fn batch_interpolate_ntt<T>(
        inout: &DeviceBuffer<T>,
        lg_domain_size: u32,
        lg_blowup: u32,
        poly_count: u32,
        device_idx: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_batch_iNTT(
            inout.as_mut_raw_ptr(),
            lg_domain_size,
            lg_blowup,
            poly_count,
            device_idx,
        ))
    }
}

// relate to lde.cu
pub mod lde {
    use super::*;

    extern "C" {
        fn _multi_bit_reverse(io: *mut std::ffi::c_void, n_bits: u32, count: u32) -> i32;
        fn _rows_bit_reverse(
            output: *mut std::ffi::c_void,
            input: *const std::ffi::c_void,
            n_bits: u32,
            height: u32,
            width: u32,
        ) -> i32;
        fn _zk_shift(io: *mut std::ffi::c_void, io_size: u32, log_n: u32, shift: u32) -> i32;
        fn _batch_expand_pad(
            output: *mut std::ffi::c_void,
            input: *const std::ffi::c_void,
            poly_count: u32,
            out_size: u32,
            in_size: u32,
        ) -> i32;
        fn _batch_polynomial_eval(
            output: *mut std::ffi::c_void,
            input: *const std::ffi::c_void,
            points: *const std::ffi::c_void,
            num_points: usize,
            width: usize,
            log_trace_height: usize,
        ) -> i32;
    }

    pub unsafe fn batch_bit_reverse<T>(
        io: &DeviceBuffer<T>,
        n_bits: u32,
        count: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_multi_bit_reverse(io.as_mut_raw_ptr(), n_bits, count))
    }

    pub unsafe fn rows_bit_reverse<T>(
        output: &DeviceBuffer<T>,
        input: &DeviceBuffer<T>,
        n_bits: u32,
        height: u32,
        width: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_rows_bit_reverse(
            output.as_mut_raw_ptr(),
            input.as_raw_ptr(),
            n_bits,
            height,
            width,
        ))
    }

    pub unsafe fn zk_shift<T>(
        io: &DeviceBuffer<T>,
        io_size: u32,
        log_n: u32,
        shift: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_zk_shift(io.as_mut_raw_ptr(), io_size, log_n, shift))
    }

    pub unsafe fn batch_expand_pad<T>(
        output: &DeviceBuffer<T>,
        input: &DeviceBuffer<T>,
        poly_count: u32,
        out_size: u32,
        in_size: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_batch_expand_pad(
            output.as_mut_raw_ptr(),
            input.as_raw_ptr(),
            poly_count,
            out_size,
            in_size,
        ))
    }

    pub unsafe fn batch_polynomial_eval<T>(
        output: &DeviceBuffer<T>,
        input: &DeviceBuffer<T>,
        points: &DeviceBuffer<T>,
        num_points: usize,
        width: usize,
        log_trace_height: usize,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_batch_polynomial_eval(
            output.as_mut_raw_ptr(),
            input.as_raw_ptr(),
            points.as_raw_ptr(),
            num_points,
            width,
            log_trace_height,
        ))
    }
}

// relate to poseidon2.cu
pub mod poseidon2 {
    use super::*;

    extern "C" {
        fn _poseidon2_rows_p3_multi(
            out: *mut std::ffi::c_void,
            ptrs: *const u64,
            cols: *const u64,
            rows: *const u64,
            row_size: u64,
            matrix_num: u64,
        ) -> i32;

        fn _poseidon2_compress(
            output: *mut std::ffi::c_void,
            input: *const std::ffi::c_void,
            output_size: u32,
            is_inject: bool,
        ) -> i32;

        fn _babybear_encode_mont_form(inout: *mut std::ffi::c_void, size: u32) -> i32;

        fn _query_digest_layers(
            d_digest_matrix: *mut std::ffi::c_void,
            d_layers_ptr: *const u64,
            d_indices: *const u64,
            num_query: u64,
            num_layer: u64,
        ) -> i32;
    }

    pub unsafe fn poseidon2_rows_p3_multi<T>(
        out: &DeviceBuffer<T>,
        ptrs: &DeviceBuffer<u64>,
        cols: &DeviceBuffer<u64>,
        rows: &DeviceBuffer<u64>,
        row_size: u64,
        matrix_num: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_poseidon2_rows_p3_multi(
            out.as_mut_raw_ptr(),
            ptrs.as_ptr(),
            cols.as_ptr(),
            rows.as_ptr(),
            row_size,
            matrix_num,
        ))
    }

    pub unsafe fn poseidon2_compress<T>(
        output: &DeviceBuffer<T>,
        input: &DeviceBuffer<T>,
        output_size: u32,
        is_inject: bool,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_poseidon2_compress(
            output.as_mut_raw_ptr(),
            input.as_raw_ptr(),
            output_size,
            is_inject,
        ))
    }

    pub unsafe fn babybear_encode_mont_form<T>(
        inout: &DeviceBuffer<T>,
        size: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_babybear_encode_mont_form(inout.as_mut_raw_ptr(), size))
    }

    pub unsafe fn query_digest_layers_kernel<T>(
        d_digest_matrix: &DeviceBuffer<T>,
        d_layers_ptr: &DeviceBuffer<u64>,
        d_indices: &DeviceBuffer<u64>,
        num_query: u64,
        num_layer: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_query_digest_layers(
            d_digest_matrix.as_mut_raw_ptr(),
            d_layers_ptr.as_ptr(),
            d_indices.as_ptr(),
            num_query,
            num_layer,
        ))
    }
}

// relate to quotient.cu
pub mod quotient {
    use super::*;

    extern "C" {
        fn _cukernel_quotient_selectors(
            first_row: *mut std::ffi::c_void,
            last_row: *mut std::ffi::c_void,
            transition: *mut std::ffi::c_void,
            inv_zeroifier: *mut std::ffi::c_void,
            log_n: u64,
            coset_log_n: u64,
            shift: u32,
        ) -> i32;

        fn _cukernel_quotient_local(
            d_quotient_values: *mut std::ffi::c_void,
            d_preprocessed: *const std::ffi::c_void,
            d_main: *const u64,
            d_permutation: *const std::ffi::c_void,
            d_exposed: *const std::ffi::c_void,
            d_public: *const std::ffi::c_void,
            d_first: *const std::ffi::c_void,
            d_last: *const std::ffi::c_void,
            d_transition: *const std::ffi::c_void,
            d_inv_zeroifier: *const std::ffi::c_void,
            d_challenge: *const std::ffi::c_void,
            d_alpha: *const std::ffi::c_void,
            d_intermediates: *const std::ffi::c_void,
            d_rules: *const std::ffi::c_void,
            num_rules: u64,
            quotient_size: u32,
            prep_height: u32,
            main_height: u32,
            perm_height: u32,
            qdb_degree: u64,
            num_rows_per_tile: u32,
        ) -> i32;

        fn _cukernel_quotient_global(
            d_quotient_values: *mut std::ffi::c_void,
            d_preprocessed: *const std::ffi::c_void,
            d_main: *const u64,
            d_permutation: *const std::ffi::c_void,
            d_exposed: *const std::ffi::c_void,
            d_public: *const std::ffi::c_void,
            d_first: *const std::ffi::c_void,
            d_last: *const std::ffi::c_void,
            d_transition: *const std::ffi::c_void,
            d_inv_zeroifier: *const std::ffi::c_void,
            d_challenge: *const std::ffi::c_void,
            d_alpha: *const std::ffi::c_void,
            d_intermediates: *const std::ffi::c_void,
            d_rules: *const std::ffi::c_void,
            num_rules: u64,
            quotient_size: u32,
            prep_height: u32,
            main_height: u32,
            perm_height: u32,
            qdb_degree: u64,
            num_rows_per_tile: u32,
        ) -> i32;
    }

    pub unsafe fn quotient_selectors<F>(
        first_row: &DeviceBuffer<F>,
        last_row: &DeviceBuffer<F>,
        transition: &DeviceBuffer<F>,
        inv_zeroifier: &DeviceBuffer<F>,
        log_n: u64,
        coset_log_n: u64,
        shift: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_cukernel_quotient_selectors(
            first_row.as_mut_raw_ptr(),
            last_row.as_mut_raw_ptr(),
            transition.as_mut_raw_ptr(),
            inv_zeroifier.as_mut_raw_ptr(),
            log_n,
            coset_log_n,
            shift,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn quotient_local<F, EF, R>(
        d_quotient_values: &DeviceBuffer<EF>,
        d_preprocessed: &DeviceBuffer<F>,
        d_main: &DeviceBuffer<u64>,
        d_permutation: &DeviceBuffer<F>,
        d_exposed: &DeviceBuffer<EF>,
        d_public: &DeviceBuffer<F>,
        d_first: &DeviceBuffer<F>,
        d_last: &DeviceBuffer<F>,
        d_transition: &DeviceBuffer<F>,
        d_inv_zeroifier: &DeviceBuffer<F>,
        d_challenge: &DeviceBuffer<EF>,
        d_alpha: &DeviceBuffer<EF>,
        d_intermediates: &DeviceBuffer<EF>,
        d_rules: &DeviceBuffer<R>,
        num_rules: u64,
        quotient_size: u32,
        prep_height: u32,
        main_height: u32,
        perm_height: u32,
        qdb_degree: u64,
        num_rows_per_tile: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_cukernel_quotient_local(
            d_quotient_values.as_mut_raw_ptr(),
            d_preprocessed.as_raw_ptr(),
            d_main.as_ptr(),
            d_permutation.as_raw_ptr(),
            d_exposed.as_raw_ptr(),
            d_public.as_raw_ptr(),
            d_first.as_raw_ptr(),
            d_last.as_raw_ptr(),
            d_transition.as_raw_ptr(),
            d_inv_zeroifier.as_raw_ptr(),
            d_challenge.as_raw_ptr(),
            d_alpha.as_raw_ptr(),
            d_intermediates.as_raw_ptr(),
            d_rules.as_raw_ptr(),
            num_rules,
            quotient_size,
            prep_height,
            main_height,
            perm_height,
            qdb_degree,
            num_rows_per_tile,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn quotient_global<F, EF, R>(
        d_quotient_values: &DeviceBuffer<EF>,
        d_preprocessed: &DeviceBuffer<F>,
        d_main: &DeviceBuffer<u64>,
        d_permutation: &DeviceBuffer<F>,
        d_exposed: &DeviceBuffer<EF>,
        d_public: &DeviceBuffer<F>,
        d_first: &DeviceBuffer<F>,
        d_last: &DeviceBuffer<F>,
        d_transition: &DeviceBuffer<F>,
        d_inv_zeroifier: &DeviceBuffer<F>,
        d_challenge: &DeviceBuffer<EF>,
        d_alpha: &DeviceBuffer<EF>,
        d_intermediates: &DeviceBuffer<EF>,
        d_rules: &DeviceBuffer<R>,
        num_rules: u64,
        quotient_size: u32,
        prep_height: u32,
        main_height: u32,
        perm_height: u32,
        qdb_degree: u64,
        num_rows_per_tile: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_cukernel_quotient_global(
            d_quotient_values.as_mut_raw_ptr(),
            d_preprocessed.as_raw_ptr(),
            d_main.as_ptr(),
            d_permutation.as_raw_ptr(),
            d_exposed.as_raw_ptr(),
            d_public.as_raw_ptr(),
            d_first.as_raw_ptr(),
            d_last.as_raw_ptr(),
            d_transition.as_raw_ptr(),
            d_inv_zeroifier.as_raw_ptr(),
            d_challenge.as_raw_ptr(),
            d_alpha.as_raw_ptr(),
            d_intermediates.as_raw_ptr(),
            d_rules.as_raw_ptr(),
            num_rules,
            quotient_size,
            prep_height,
            main_height,
            perm_height,
            qdb_degree,
            num_rows_per_tile,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn quotient_global_or_local<F, EF, R>(
        is_global: bool,
        d_quotient_values: &DeviceBuffer<EF>,
        d_preprocessed: &DeviceBuffer<F>,
        d_main: &DeviceBuffer<u64>,
        d_permutation: &DeviceBuffer<F>,
        d_exposed: &DeviceBuffer<EF>,
        d_public: &DeviceBuffer<F>,
        d_first: &DeviceBuffer<F>,
        d_last: &DeviceBuffer<F>,
        d_transition: &DeviceBuffer<F>,
        d_inv_zeroifier: &DeviceBuffer<F>,
        d_challenge: &DeviceBuffer<EF>,
        d_alpha: &DeviceBuffer<EF>,
        d_intermediates: &DeviceBuffer<EF>,
        d_rules: &DeviceBuffer<R>,
        num_rules: u64,
        quotient_size: u32,
        prep_height: u32,
        main_height: u32,
        perm_height: u32,
        qdb_degree: u64,
        num_rows_per_tile: u32,
    ) -> Result<(), CudaError> {
        if is_global {
            quotient_global(
                d_quotient_values,
                d_preprocessed,
                d_main,
                d_permutation,
                d_exposed,
                d_public,
                d_first,
                d_last,
                d_transition,
                d_inv_zeroifier,
                d_challenge,
                d_alpha,
                d_intermediates,
                d_rules,
                num_rules,
                quotient_size,
                prep_height,
                main_height,
                perm_height,
                qdb_degree,
                num_rows_per_tile,
            )
        } else {
            quotient_local(
                d_quotient_values,
                d_preprocessed,
                d_main,
                d_permutation,
                d_exposed,
                d_public,
                d_first,
                d_last,
                d_transition,
                d_inv_zeroifier,
                d_challenge,
                d_alpha,
                d_intermediates,
                d_rules,
                num_rules,
                quotient_size,
                prep_height,
                main_height,
                perm_height,
                qdb_degree,
                num_rows_per_tile,
            )
        }
    }
}

// relate to prefix.cu
pub mod prefix {
    use super::*;

    extern "C" {
        fn _prefix_scan_block_ext(
            d_inout: *mut std::ffi::c_void,
            length: u64,
            round_stride: u64,
            block_num: u64,
        ) -> i32;

        fn _prefix_scan_block_downsweep_ext(
            d_inout: *mut std::ffi::c_void,
            length: u64,
            round_stride: u64,
        ) -> i32;

        fn _prefix_scan_epilogue_ext(d_inout: *mut std::ffi::c_void, length: u64) -> i32;
    }

    pub unsafe fn prefix_scan_block_ext<T>(
        d_inout: &DeviceBuffer<T>,
        length: u64,
        round_stride: u64,
        block_num: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_prefix_scan_block_ext(
            d_inout.as_mut_raw_ptr(),
            length,
            round_stride,
            block_num,
        ))
    }

    pub unsafe fn prefix_scan_block_downsweep_ext<T>(
        d_inout: &DeviceBuffer<T>,
        length: u64,
        round_stride: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_prefix_scan_block_downsweep_ext(
            d_inout.as_mut_raw_ptr(),
            length,
            round_stride,
        ))
    }

    pub unsafe fn prefix_scan_epilogue_ext<T>(
        d_inout: &DeviceBuffer<T>,
        length: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_prefix_scan_epilogue_ext(d_inout.as_mut_raw_ptr(), length))
    }
}

// relate to permute.cu
pub mod permute {
    use super::*;

    extern "C" {
        fn _calculate_cumulative_sums(
            is_global: bool,
            d_permutation: *mut std::ffi::c_void,
            d_cumulative_sums: *mut std::ffi::c_void,
            d_preprocessed: *const std::ffi::c_void,
            d_main: *const u64,
            d_challenges: *const std::ffi::c_void,
            d_intermediates: *const std::ffi::c_void,
            d_rules: *const std::ffi::c_void,
            d_used_nodes: *const usize,
            d_partition_lens: *const u32,
            num_partitions: u32,
            permutation_height: u32,
            permutation_width_ext: u32,
            num_rows_per_tile: u32,
        ) -> i32;

        fn _permute_update(
            d_sum: *mut std::ffi::c_void,
            d_permutation: *mut std::ffi::c_void,
            d_cumulative_sums: *mut std::ffi::c_void,
            permutation_height: u32,
            permutation_width_ext: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn calculate_cumulative_sums<F, EF, R>(
        is_global: bool,
        d_permutation: &DeviceBuffer<F>,
        d_cumulative_sums: &DeviceBuffer<EF>,
        d_preprocessed: &DeviceBuffer<F>,
        d_main: &DeviceBuffer<u64>,
        d_challenges: &DeviceBuffer<EF>,
        d_intermediates: &DeviceBuffer<EF>,
        d_rules: &DeviceBuffer<R>,
        d_used_nodes: &DeviceBuffer<usize>,
        d_partition_lens: &DeviceBuffer<u32>,
        num_partitions: u32,
        permutation_height: u32,
        permutation_width_ext: u32,
        num_rows_per_tile: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_calculate_cumulative_sums(
            is_global,
            d_permutation.as_mut_raw_ptr(),
            d_cumulative_sums.as_mut_raw_ptr(),
            d_preprocessed.as_raw_ptr(),
            d_main.as_ptr(),
            d_challenges.as_raw_ptr(),
            d_intermediates.as_raw_ptr(),
            d_rules.as_raw_ptr(),
            d_used_nodes.as_ptr(),
            d_partition_lens.as_ptr(),
            num_partitions,
            permutation_height,
            permutation_width_ext,
            num_rows_per_tile,
        ))
    }

    pub unsafe fn permute_update<F, EF>(
        d_sum: &DeviceBuffer<EF>,
        d_permutation: &DeviceBuffer<F>,
        d_cumulative_sums: &DeviceBuffer<EF>,
        permutation_height: u32,
        permutation_width_ext: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_permute_update(
            d_sum.as_mut_raw_ptr(),
            d_permutation.as_mut_raw_ptr(),
            d_cumulative_sums.as_mut_raw_ptr(),
            permutation_height,
            permutation_width_ext,
        ))
    }
}

// relate to fri.cu
pub mod fri {
    use super::*;

    extern "C" {
        fn _compute_diffs(
            d_diffs: *mut std::ffi::c_void,
            d_z: *mut std::ffi::c_void,
            d_domain: *mut std::ffi::c_void,
            log_max_height: u32,
        ) -> i32;

        fn _fri_bit_reverse(d_diffs: *mut std::ffi::c_void, log_max_height: u32) -> i32;

        fn _batch_invert(
            d_diffs: *mut std::ffi::c_void,
            log_max_height: u32,
            invert_task_num: u32,
        ) -> i32;

        fn _powers(d_data: *mut std::ffi::c_void, d_g: *const std::ffi::c_void, N: u32) -> i32;
        fn _powers_ext(d_data: *mut std::ffi::c_void, d_g: *const std::ffi::c_void, N: u32) -> i32;

        fn _precompute_diff_powers(
            d_output: *mut std::ffi::c_void,
            d_diff_invs: *const std::ffi::c_void,
            d_powers: *const std::ffi::c_void,
            N: u32,
        ) -> i32;

        fn _matrix_scale_rows_then_reduce(
            d_output: *mut std::ffi::c_void,
            d_matrix: *const std::ffi::c_void,
            d_diff_invs_dot_g_powers: *const std::ffi::c_void,
            width: u32,
            matrix_height: u32,
            domain_height: u32,
            reduce_matrix_height: u32,
        ) -> i32;

        fn _matrix_reduce(
            d_output: *mut std::ffi::c_void,
            d_input: *const std::ffi::c_void,
            width: u32,
            current_height: u32,
            buffer_height: u32,
            next_round_height: u32,
        ) -> i32;

        fn _matrix_get_first_column(
            d_output: *mut std::ffi::c_void,
            d_input: *const std::ffi::c_void,
            width: u32,
            height: u32,
        ) -> i32;

        fn _reduce_matrix_quotient_acc(
            d_quotient_acc: *mut std::ffi::c_void,
            d_matrix: *const std::ffi::c_void,
            d_z_diff_invs: *const std::ffi::c_void,
            d_matrix_eval: *const std::ffi::c_void,
            d_alphas: *const std::ffi::c_void,
            d_alphas_offset: *const std::ffi::c_void,
            width: u32,
            height: u32,
            is_first: bool,
        ) -> i32;

        fn _cukernel_split_ext_poly_to_base_col_major_matrix(
            d_matrix: *mut std::ffi::c_void,
            d_poly: *const std::ffi::c_void,
            poly_len: u64,
            matrix_height: u32,
        ) -> i32;

        fn _cukernel_fri_fold(
            d_result: *mut std::ffi::c_void,
            d_poly: *const std::ffi::c_void,
            fri_input: *const std::ffi::c_void,
            d_constants: *const std::ffi::c_void,
            g_invs: *const std::ffi::c_void,
            N: u64,
        ) -> i32;
    }

    pub unsafe fn diffs_kernel<F, EF>(
        d_diffs: &DeviceBuffer<EF>,
        d_z: &DeviceBuffer<EF>,
        d_domain: &DeviceBuffer<F>,
        log_max_height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_compute_diffs(
            d_diffs.as_mut_raw_ptr(),
            d_z.as_mut_raw_ptr(),
            d_domain.as_mut_raw_ptr(),
            log_max_height,
        ))
    }

    pub unsafe fn bit_rev_kernel<EF>(
        d_diffs: &DeviceBuffer<EF>,
        log_max_height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_fri_bit_reverse(d_diffs.as_mut_raw_ptr(), log_max_height))
    }

    pub unsafe fn batch_invert_kernel<EF>(
        d_diffs: &DeviceBuffer<EF>,
        log_max_height: u32,
        invert_task_num: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_batch_invert(
            d_diffs.as_mut_raw_ptr(),
            log_max_height,
            invert_task_num,
        ))
    }

    pub unsafe fn powers<F>(
        d_data: &DeviceBuffer<F>,
        d_g: &DeviceBuffer<F>,
        n: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_powers(d_data.as_mut_raw_ptr(), d_g.as_raw_ptr(), n))
    }

    pub unsafe fn powers_ext<EF>(
        d_data: &DeviceBuffer<EF>,
        d_g: &DeviceBuffer<EF>,
        n: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_powers_ext(d_data.as_mut_raw_ptr(), d_g.as_raw_ptr(), n))
    }

    pub unsafe fn precompute_diff_powers_kernel<F, EF>(
        d_output: &DeviceBuffer<EF>,
        d_diff_invs: &DeviceBuffer<EF>,
        d_powers: &DeviceBuffer<F>,
        n: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_precompute_diff_powers(
            d_output.as_mut_raw_ptr(),
            d_diff_invs.as_raw_ptr(),
            d_powers.as_raw_ptr(),
            n,
        ))
    }

    pub unsafe fn scale_and_reduce_kernel<F, EF>(
        d_output: &DeviceBuffer<EF>,
        d_matrix: &DeviceBuffer<F>,
        d_diff_invs_dot_g_powers: &DeviceBuffer<EF>,
        width: u32,
        matrix_height: u32,
        domain_height: u32,
        reduce_matrix_height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_matrix_scale_rows_then_reduce(
            d_output.as_mut_raw_ptr(),
            d_matrix.as_raw_ptr(),
            d_diff_invs_dot_g_powers.as_raw_ptr(),
            width,
            matrix_height,
            domain_height,
            reduce_matrix_height,
        ))
    }

    pub unsafe fn round_reduce_kernel<EF>(
        d_output: &DeviceBuffer<EF>,
        d_input: &DeviceBuffer<EF>,
        width: u32,
        current_height: u32,
        buffer_height: u32,
        next_round_height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_matrix_reduce(
            d_output.as_mut_raw_ptr(),
            d_input.as_raw_ptr(),
            width,
            current_height,
            buffer_height,
            next_round_height,
        ))
    }

    pub unsafe fn get_first_col_kernel<EF>(
        d_output: &DeviceBuffer<EF>,
        d_input: &DeviceBuffer<EF>,
        width: u32,
        height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_matrix_get_first_column(
            d_output.as_mut_raw_ptr(),
            d_input.as_raw_ptr(),
            width,
            height,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn reduce_matrix_quotient_kernel<F, EF>(
        d_quotient_acc: &DeviceBuffer<EF>,
        d_matrix: &DeviceBuffer<F>,
        d_z_diff_invs: &DeviceBuffer<EF>,
        d_matrix_eval: &DeviceBuffer<EF>,
        d_alphas: &DeviceBuffer<EF>,
        d_alphas_offset: &DeviceBuffer<EF>,
        width: u32,
        height: u32,
        is_first: bool,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_reduce_matrix_quotient_acc(
            d_quotient_acc.as_mut_raw_ptr(),
            d_matrix.as_raw_ptr(),
            d_z_diff_invs.as_raw_ptr(),
            d_matrix_eval.as_raw_ptr(),
            d_alphas.as_raw_ptr(),
            d_alphas_offset.as_raw_ptr(),
            width,
            height,
            is_first,
        ))
    }

    pub unsafe fn split_ext_poly_to_base_col_major_matrix<F, EF>(
        d_matrix: &DeviceBuffer<F>,
        d_poly: &DeviceBuffer<EF>,
        poly_len: u64,
        matrix_height: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_cukernel_split_ext_poly_to_base_col_major_matrix(
            d_matrix.as_mut_raw_ptr(),
            d_poly.as_raw_ptr(),
            poly_len,
            matrix_height,
        ))
    }

    pub unsafe fn fri_fold_kernel<F, EF>(
        d_result: &DeviceBuffer<EF>,
        d_poly: &DeviceBuffer<EF>,
        fri_input: &DeviceBuffer<EF>,
        d_constants: &DeviceBuffer<EF>,
        g_invs: &DeviceBuffer<F>,
        half_folded_len: u64,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_cukernel_fri_fold(
            d_result.as_mut_raw_ptr(),
            d_poly.as_raw_ptr(),
            fri_input.as_raw_ptr(),
            d_constants.as_raw_ptr(),
            g_invs.as_raw_ptr(),
            half_folded_len,
        ))
    }
}
