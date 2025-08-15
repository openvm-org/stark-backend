use itertools::{izip, Itertools};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::prover::hal::MatrixDimensions;
use p3_field::{ExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use p3_util::{linear_map::LinearMap, log2_ceil_usize, log2_strict_usize};

use crate::{
    base::{DeviceMatrix, DevicePoly, ExtendedLagrangeCoeff},
    cuda::kernels::{fri::*, lde::batch_bit_reverse},
    prelude::*,
};

// return a map from point to its inverse denominator of largest height in bit reverse order
pub(crate) fn compute_inverse_denominators_on_gpu(
    heights_and_points: &[(Vec<usize>, &Vec<Vec<EF>>)],
    coset_shift: F,
) -> LinearMap<EF, DevicePoly<EF, ExtendedLagrangeCoeff>> {
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();

    for (heights, points) in heights_and_points {
        for (height, points_for_mat) in izip!(heights, *points) {
            let log_height = log2_strict_usize(*height);
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            let g = F::two_adic_generator(log_height);
            let diff_invs = get_diff_invs(z, coset_shift, g, log_height, true).unwrap();
            (z, diff_invs)
        })
        .collect()
}

// return a map from point to inverse denominators of different heights in non-bitrev order
// note this function at most takes twice the memory of
// `compute_bitrev_inverse_denominators_on_gpu`. if the max log height is 23, then the memory usage
// for one point is 2^23 * 2 * sizeof<EF>. for field like babybear and its deg 4 extension, the
// memory usage is 2^23 * 2 * 16 = 256MB. #[allow(clippy::type_complexity)]
pub(crate) fn compute_non_bitrev_inverse_denominators_on_gpu(
    heights_and_points: &[(Vec<usize>, &Vec<Vec<EF>>)],
    coset_shift: F,
) -> LinearMap<EF, Vec<Option<DevicePoly<EF, ExtendedLagrangeCoeff>>>> {
    let mut log_heights_for_point: LinearMap<EF, Vec<Option<usize>>> = LinearMap::new();

    for (heights, points) in heights_and_points {
        for (height, points_for_mat) in izip!(heights, *points) {
            let log_height = log2_strict_usize(*height);

            for &z in points_for_mat {
                if let Some(lh) = log_heights_for_point.get_mut(&z) {
                    lh[log_height] = Some(log_height);
                } else {
                    let mut lh = vec![None; 32];
                    lh[log_height] = Some(log_height);
                    log_heights_for_point.insert(z, lh);
                }
            }
        }
    }
    log_heights_for_point
        .into_iter()
        .map(|(z, log_heights)| {
            (
                z,
                log_heights
                    .into_iter()
                    .map(|log_height| {
                        log_height.map(|log_height| {
                            let g = F::two_adic_generator(log_height);
                            get_diff_invs(z, coset_shift, g, log_height, false).unwrap()
                        })
                    })
                    .collect_vec(),
            )
        })
        .collect()
}

// Returns evaluations of `1 / (z - x)` over coset domain `s*H` and
// store it in bit reversed order if `bitrev` is true
fn get_diff_invs(
    z: EF,
    shift: F,
    g: F,
    log_max_height: usize,
    bitrev: bool,
) -> Result<DevicePoly<EF, ExtendedLagrangeCoeff>, ()> {
    let n_elems = 1 << log_max_height;
    let d_inv_diffs: DeviceBuffer<EF> = DeviceBuffer::<EF>::with_capacity(n_elems);
    let d_z: DeviceBuffer<EF> = [z].to_device().unwrap();
    let d_domain: DeviceBuffer<F> = [shift, g].to_device().unwrap();
    let invert_task_num = (n_elems as u32).div_ceil(16); // hardcoded Scrolls parameter

    unsafe {
        diffs_kernel(&d_inv_diffs, &d_z, &d_domain, log_max_height as u32).unwrap();
        if bitrev {
            fpext_bit_rev_kernel(&d_inv_diffs, log_max_height as u32).unwrap();
        }
        batch_invert_kernel(&d_inv_diffs, log_max_height as u32, invert_task_num).unwrap();
    }
    Ok(DevicePoly::new(bitrev, d_inv_diffs))
}

pub(crate) fn reduce_matrix_quotient_acc(
    quotient_acc: &mut DevicePoly<EF, ExtendedLagrangeCoeff>,
    matrix: &DeviceMatrix<F>,
    z_diff_invs: &DevicePoly<EF, ExtendedLagrangeCoeff>,
    m_z: EF,
    alpha: EF,
    matrix_offset: usize,
    is_first: bool,
) -> Result<(), ()> {
    // TODO: assert quotient_acc and matrix have same order (bitrev or not)
    //  assert matrix is also evaluations on coset domain
    assert_eq!(matrix.height(), quotient_acc.len());

    // quotient poly q(x) += alpha^(matrix_offset) * m(z)-m_rlc(x) / (z-x)
    let d_alpha_powers = DeviceBuffer::<EF>::with_capacity(matrix.width());
    let d_alpha = [alpha].to_device().unwrap();
    let d_m_eval = [m_z].to_device().unwrap();
    let alpha_offset = alpha.exp_u64(matrix_offset as u64);
    let d_alphas_offset = [alpha_offset].to_device().unwrap();

    unsafe {
        powers_ext(&d_alpha_powers, &d_alpha, matrix.width() as u32).unwrap();
        reduce_matrix_quotient_kernel(
            &quotient_acc.coeff,
            matrix.buffer(),
            &z_diff_invs.coeff,
            &d_m_eval,
            &d_alpha_powers,
            &d_alphas_offset,
            matrix.width().try_into().unwrap(),
            matrix.height().try_into().unwrap(),
            is_first,
        )
        .unwrap();
    }

    Ok(())
}

pub(crate) fn fri_ext_poly_to_base_matrix(
    poly: &DevicePoly<EF, ExtendedLagrangeCoeff>,
) -> Result<DeviceMatrix<F>, ()> {
    const SPLIT_FACTOR: usize = 2;
    const EF_D: usize = 4; // Self::ExtElem::D;
    let matrix_width = SPLIT_FACTOR * EF_D;
    let matrix_height = poly.len() / SPLIT_FACTOR;
    let matrix = DeviceMatrix::<F>::with_capacity(matrix_height, matrix_width);
    assert_eq!(matrix_width * matrix_height, poly.len() * EF_D);

    tracing::debug!(
        "poly {} fold {}: matrix_width = {}, matrix_height = {}",
        poly.len(),
        SPLIT_FACTOR,
        matrix_width,
        matrix_height
    );

    unsafe {
        split_ext_poly_to_base_col_major_matrix(
            matrix.buffer(),
            &poly.coeff,
            poly.len() as u64,
            matrix_height as u32,
        )
        .unwrap();
    }

    Ok(matrix)
}

pub(crate) fn fri_fold(
    folded: DevicePoly<EF, ExtendedLagrangeCoeff>,
    fri_input: Option<DevicePoly<EF, ExtendedLagrangeCoeff>>,
    beta: EF,
    g_inv: EF,
) -> Result<DevicePoly<EF, ExtendedLagrangeCoeff>, ()> {
    // we don't support folded poly whose length exceed 2^27
    // as we take advantage of the fact g_inv is a base field element if folded.len() <= 2^27
    assert!(log2_strict_usize(folded.len()) <= 27);
    assert!(g_inv.as_base_slice()[1..].iter().all(F::is_zero)); // g_inv.is_in_basefield()
    assert!(folded.is_bit_reversed);

    let half_one = (F::ONE / F::from_canonical_usize(2)).into();
    let half_beta = beta * half_one;
    let beta_square = beta * beta;

    let d_constants = [half_beta, half_one, beta_square].to_device().unwrap();

    let half_folded_len = folded.len() / 2;
    let g_invs = DeviceBuffer::<F>::with_capacity(half_folded_len);
    let d_g_inv = [g_inv.as_base().unwrap()].to_device().unwrap();
    let d_result = DeviceBuffer::<EF>::with_capacity(half_folded_len);
    unsafe {
        powers(&g_invs, &d_g_inv, half_folded_len as u32).unwrap();
        batch_bit_reverse(
            &g_invs,
            log2_ceil_usize(half_folded_len) as u32,
            half_folded_len as u32,
        )
        .unwrap();
        fri_fold_kernel(
            &d_result,
            &folded.coeff,
            &fri_input.map_or(DeviceBuffer::<EF>::new(), |f| f.coeff),
            &d_constants,
            &g_invs,
            half_folded_len as u64,
        )
        .unwrap();
    }

    Ok(DevicePoly::new(folded.is_bit_reversed, d_result))
}

const TARGET_CHUNK_SIZE: usize = 16384;
const MAX_CHUNKS: usize = 512;

pub(crate) fn matrix_evaluate(
    matrix: &DeviceMatrix<F>,
    inv_denoms: &DevicePoly<EF, ExtendedLagrangeCoeff>,
    z: EF,
    shift: F,
    g: F,
    domain_height: usize,
) -> Result<Vec<EF>, ()> {
    assert_eq!(domain_height, inv_denoms.len());

    // Scale factor: M(z) / (N * s^{N-1})
    let log_height = log2_strict_usize(domain_height);
    let zerofier = z.exp_power_of_2(log_height) - shift.exp_power_of_2(log_height);
    let denominator =
        EF::from_canonical_usize(domain_height) * shift.exp_u64(domain_height as u64 - 1);
    let scale_factor = zerofier * denominator.inverse();

    let ideal_chunks = domain_height.div_ceil(TARGET_CHUNK_SIZE);
    let num_chunks = ideal_chunks.clamp(1, MAX_CHUNKS);
    let chunk_size = domain_height.div_ceil(num_chunks);

    let partial_sums = DeviceBuffer::<EF>::with_capacity(num_chunks * matrix.width());

    unsafe {
        matrix_evaluate_chunked_kernel(
            &partial_sums,
            matrix.buffer(),
            &inv_denoms.coeff,
            g,
            domain_height as u32,
            matrix.width() as u32,
            chunk_size as u32,
            num_chunks as u32,
            matrix.height() as u32,
            inv_denoms.is_bit_reversed,
        )
        .unwrap();
    }

    let output = DeviceBuffer::<EF>::with_capacity(matrix.width());

    unsafe {
        matrix_evaluate_finalize_kernel(
            &output,
            &partial_sums,
            scale_factor,
            num_chunks as u32,
            matrix.width() as u32,
        )
        .unwrap();
    }

    Ok(output.to_host().unwrap())
}
