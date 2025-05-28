use cuda_kernels::{lde::*, matrix::matrix_get_rows_fp_kernel, ntt::*};
use cuda_utils::copy::MemCopyH2D;
use itertools::Itertools;
use openvm_stark_backend::prover::hal::MatrixDimensions;
use p3_field::{FieldAlgebra, PrimeField32, TwoAdicField};
use p3_util::log2_strict_usize;

use crate::{base::DeviceMatrix, prelude::F};

pub(crate) fn inplace_ifft(trace_matrix: DeviceMatrix<F>, device_id: u32) -> DeviceMatrix<F> {
    let width = trace_matrix.width();
    let log_trace_height = log2_strict_usize(trace_matrix.height());
    unsafe {
        batch_interpolate_ntt(
            trace_matrix.buffer(),
            log_trace_height as u32,
            0,
            width as u32,
            device_id,
        )
        .unwrap();
    }
    trace_matrix
}

pub(crate) fn compute_lde_matrix<const FULL: bool>(
    trace_matrix: DeviceMatrix<F>,
    device_id: u32,
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
        if FULL {
            batch_interpolate_ntt(
                lde_matrix.buffer(),
                log_trace_height,
                log_blowup,
                width as u32,
                device_id,
            )
            .unwrap();
        }
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

        batch_ntt(lde_matrix.buffer(), log_lde_height, width as u32, device_id).unwrap();
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

pub(crate) fn polynomial_evaluate(
    trace: &DeviceMatrix<F>,
    shift: F,
    lde_height: usize,
    row_indices: &[usize],
) -> DeviceMatrix<F> {
    let num_points = row_indices.len();
    let log_trace_height = log2_strict_usize(trace.height());
    let log_lde_height = log2_strict_usize(lde_height);

    let result = DeviceMatrix::<F>::with_capacity(num_points, trace.width());
    let d_points = row_indices
        .iter()
        .map(|&idx| {
            let bit_rev_idx = (idx as u64).reverse_bits() >> (64 - log_lde_height);
            shift * F::two_adic_generator(log_lde_height).exp_u64(bit_rev_idx)
        })
        .collect_vec()
        .to_device()
        .unwrap();
    unsafe {
        batch_polynomial_eval(
            result.buffer(),
            trace.buffer(),
            &d_points,
            num_points,
            trace.width(),
            log_trace_height,
        )
        .unwrap();
    }
    result
}
