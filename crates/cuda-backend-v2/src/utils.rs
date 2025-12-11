use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use stark_backend_v2::utils::batch_multiplicative_inverse_serial;

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
            if denom.is_zero() { EF::ONE } else { denom }
        })
        .collect_vec();
    let mut inv_denoms = batch_multiplicative_inverse_serial(&denoms);
    let zerofier = z.exp_power_of_2(l_skip) - F::ONE;
    let denominator = F::from_canonical_usize(1 << l_skip);
    let scale_factor = zerofier * denominator.inverse();
    for v in &mut inv_denoms {
        *v *= scale_factor;
    }
    inv_denoms
}
