use std::mem::transmute;

use itertools::Itertools;
use openvm_stark_backend::utils::batch_multiplicative_inverse_serial;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField64};

use crate::{D_EF, EF, F};

// https://hackmd.io/@vbuterin/barycentric_evaluation#Special-case-roots-of-unity
pub fn compute_barycentric_inv_lagrange_denoms<F: Field, EF: ExtensionField<F>>(
    l_skip: usize,
    omega_skip_pows: &[F],
    z: EF,
) -> Vec<EF> {
    debug_assert_eq!(1 << l_skip, omega_skip_pows.len());
    let denoms = omega_skip_pows
        .iter()
        .map(|&w_i| {
            let denom = z - w_i;
            if denom.is_zero() {
                EF::ONE
            } else {
                denom
            }
        })
        .collect_vec();
    let mut inv_denoms = batch_multiplicative_inverse_serial(&denoms);
    let zerofier = z.exp_power_of_2(l_skip) - F::ONE;
    let denominator = F::from_usize(1 << l_skip);
    let scale_factor = zerofier * denominator.inverse();
    for v in &mut inv_denoms {
        *v *= scale_factor;
    }
    inv_denoms
}

/// Reduce overflowing u64 to extension field elements. After reducing modulo `p`, the `u64` are in
/// Montgomery form.
#[inline]
pub fn reduce_raw_u64_to_ef(accum: &[u64]) -> Vec<EF> {
    debug_assert_eq!(accum.len() % D_EF, 0);
    debug_assert_eq!(size_of::<F>(), size_of::<u32>());
    accum
        .chunks_exact(D_EF)
        .map(|chunk| {
            EF::from_basis_coefficients_fn(|i| {
                let monty_raw = (chunk[i] % F::ORDER_U64) as u32;
                // SAFETY:
                // - BabyBear has same memory layout as u32
                // - Internally stored in Montgomery form
                unsafe { transmute::<u32, F>(monty_raw) }
            })
        })
        .collect()
}
