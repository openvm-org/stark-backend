use itertools::Itertools;
use openvm_stark_backend::prover::hal::MatrixDimensions;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_util::log2_strict_usize;

use openvm_cuda_common::copy::MemCopyH2D;
use crate::{
    base::DeviceMatrix,
    cuda::
        kernels::{lde::*, matrix::matrix_get_rows_fp_kernel, ntt::*},
    prelude::F,
};

pub(crate) fn compute_lde_matrix(
    trace_matrix: &DeviceMatrix<F>,
    domain_size: usize,
    shift: F,
) -> DeviceMatrix<F> {
    let width = trace_matrix.width();
    let trace_height = trace_matrix.height();
    let lde_height = domain_size;

    let log_trace_height = log2_strict_usize(trace_height) as u32;
    let log_lde_height = log2_strict_usize(lde_height) as u32;
    let log_blowup = log_lde_height - log_trace_height;

    let lde_matrix = DeviceMatrix::<F>::with_capacity(lde_height, width);
    let lde_size = (lde_matrix.height() * lde_matrix.width()) as u32;

    unsafe {
        batch_expand_pad(
            lde_matrix.buffer(),
            trace_matrix.buffer(),
            width as u32,
            lde_height as u32,
            trace_height as u32,
        )
        .unwrap();

        batch_interpolate_ntt(
            lde_matrix.buffer(),
            log_trace_height,
            log_blowup,
            width as u32,
        )
        .unwrap();

        if shift != F::ONE {
            zk_shift(
                lde_matrix.buffer(),
                lde_size,
                log_lde_height,
                shift.as_canonical_u32(),
            )
            .unwrap();
        }

        batch_bit_reverse(lde_matrix.buffer(), log_lde_height, lde_size).unwrap();

        batch_ntt(lde_matrix.buffer(), log_lde_height, width as u32).unwrap();
    }

    lde_matrix
}

pub(crate) fn get_rows_from_matrix(
    lde: &DeviceMatrix<F>,
    row_indices: &[usize],
) -> DeviceMatrix<F> {
    let result = DeviceMatrix::<F>::with_capacity(row_indices.len(), lde.width());
    let d_row_indices = row_indices
        .iter()
        .map(|&x| (x as u32).reverse_bits() >> (32 - log2_strict_usize(lde.height())))
        .collect_vec()
        .to_device()
        .unwrap();
    unsafe {
        matrix_get_rows_fp_kernel(
            result.buffer(),
            lde.buffer(),
            &d_row_indices,
            lde.width() as u64,
            lde.height() as u64,
            row_indices.len() as u32,
        )
        .unwrap();
    }
    result
}
