use core::ops::{Add, Sub};
use std::{iter::zip, ops::Mul};

use itertools::Itertools;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};

pub fn eval_eq_mle<F1, F2, F3>(x: &[F1], y: &[F2]) -> F3
where
    F1: Field,
    F2: Field,
    F3: Field,
    F1: Mul<F2, Output = F3>,
    F3: Sub<F1, Output = F3>,
    F3: Sub<F2, Output = F3>,
{
    debug_assert_eq!(x.len(), y.len());
    zip(x, y).fold(F3::ONE, |acc, (&x_i, &y_i)| {
        acc * (F3::ONE - y_i - x_i + (x_i * y_i).double())
    })
}

/// Evaluate `mobius_eq_poly(u_tilde)` at an arbitrary point `x`.
///
/// ```text
/// mobius_eq_poly(u_tilde)(x) = ∏_i ((1 - 2*u_tilde_i) * (1 - x_i) + u_tilde_i * x_i)
/// ```
pub fn eval_mobius_eq_mle<F: Field>(u: &[F], x: &[F]) -> F {
    debug_assert_eq!(u.len(), x.len());
    zip(u, x).fold(F::ONE, |acc, (&u_i, &x_i)| {
        let w0 = F::ONE - u_i.double();
        acc * (w0 * (F::ONE - x_i) + u_i * x_i)
    })
}

/// Evaluate the MLE defined by its hypercube evaluations at an arbitrary point, in place.
///
/// `evals` has length `2^n` and contains `f(b)` for each `b ∈ {0,1}^n`.
/// Returns `f(x)` where `x = x[0..n]`.
pub fn eval_mle_evals_at_point<F: Field>(evals: &mut [F], x: &[F]) -> F {
    debug_assert_eq!(evals.len(), 1 << x.len());
    let mut len = evals.len();
    for &xj in x.iter().rev() {
        len >>= 1;
        let (lo, hi) = evals.split_at_mut(len);
        for i in 0..len {
            lo[i] = lo[i] * (F::ONE - xj) + hi[i] * xj;
        }
    }
    evals[0]
}

/// Let D be univariate skip domain, the subgroup of `F^*` of order `l_skip`.
///
/// Computes the polynomial ```text
///     eq_D(X, Y) = \sum_{z_1 \in D} \prod_{z_2 \in D, z_2 != z_1} (X - z_1)(Y - z_2) / (z_1 -
/// z_2)^2 ```
pub fn eval_eq_uni<F: Field>(l_skip: usize, x: F, y: F) -> F {
    let mut res = F::ONE;
    for (x_pow, y_pow) in zip(x.exp_powers_of_2(), y.exp_powers_of_2()).take(l_skip) {
        res = (x_pow + y_pow) * res + (x_pow - F::ONE) * (y_pow - F::ONE);
    }
    res * F::ONE.halve().exp_u64(l_skip as u64)
}

/// Let D be univariate skip domain, the subgroup of `F^*` of order `l_skip`.
///
/// Computes the polynomial eq_D(X, 1); see `eval_eq_uni`.
pub fn eval_eq_uni_at_one<F: Field>(l_skip: usize, x: F) -> F {
    let mut res = F::ONE;
    for x_pow in x.exp_powers_of_2().take(l_skip) {
        res *= x_pow + F::ONE;
    }
    res * F::ONE.halve().exp_u64(l_skip as u64)
}

/// Returns `eq_D(x, Z)` as a polynomial in `Z` in coefficient form.
/// Derived from `eq_D(x, Z)` being the Lagrange basis at `x`, which is the character sum over the
/// roots of unity.
///
/// If z in D, then `eq_D(x, z) = 1/N sum_{k=1}^N (x/z)^k = 1/N sum_{k=1}^N x^k
/// z^{N-k}`.
pub fn eq_uni_poly<F, EF>(l_skip: usize, x: EF) -> UnivariatePoly<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let n_inv = F::ONE.halve().exp_u64(l_skip as u64);
    let mut coeffs = x
        .powers()
        .skip(1)
        .take(1 << l_skip)
        .map(|x_pow| x_pow * n_inv)
        .collect_vec();
    coeffs.reverse();
    coeffs[0] = n_inv.into();
    UnivariatePoly::new(coeffs)
}

pub fn eval_in_uni<F: Field>(l_skip: usize, n: isize, z: F) -> F {
    debug_assert!(n >= -(l_skip as isize));
    if n.is_negative() {
        eval_eq_uni_at_one(
            n.unsigned_abs(),
            z.exp_power_of_2(l_skip.wrapping_add_signed(n)),
        )
    } else {
        F::ONE
    }
}

pub fn eval_eq_prism<F: Field>(l_skip: usize, x: &[F], y: &[F]) -> F {
    eval_eq_uni(l_skip, x[0], y[0]) * eval_eq_mle(&x[1..], &y[1..])
}

pub fn evals_eq_hypercube_serial<F: Field>(x: &[F]) -> Vec<F> {
    let n = x.len();
    let mut out = F::zero_vec(1 << n);
    out[0] = F::ONE;
    for (i, &x_i) in x.iter().enumerate() {
        let (los, his) = out[..2 << i].split_at_mut(1 << i);
        for (lo, hi) in los.iter_mut().zip(his.iter_mut()) {
            *hi = *lo * x_i;
            *lo *= F::ONE - x_i;
        }
    }
    out
}

/// Length of `xi_1` should be `l_skip`.
pub fn eval_eq_sharp_uni<F, EF>(omega_skip_pows: &[F], xi_1: &[EF], z: EF) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    let l_skip = xi_1.len();
    debug_assert_eq!(omega_skip_pows.len(), 1 << l_skip);

    let mut res = EF::ZERO;
    let eq_xi_evals = evals_eq_hypercube_serial(xi_1);
    for (&omega_pow, eq_xi_eval) in omega_skip_pows.iter().zip(eq_xi_evals) {
        res += eval_eq_uni(l_skip, z, omega_pow.into()) * eq_xi_eval;
    }
    #[cfg(debug_assertions)]
    {
        let coeffs = (0..(1 << l_skip))
            .map(|k| {
                let mut c = EF::ONE;
                #[allow(clippy::needless_range_loop)]
                for i in 0..l_skip {
                    let idx = (k << i) % (1 << l_skip);
                    c *= EF::ONE - xi_1[i]
                        + xi_1[i] * omega_skip_pows[((1 << l_skip) - idx) % (1 << l_skip)];
                }
                c
            })
            .collect_vec();
        let mut rpow = EF::ONE;
        let mut other = EF::ZERO;
        for c in coeffs {
            other += rpow * c;
            rpow *= z;
        }
        other *= EF::TWO.inverse().exp_u64(l_skip as u64);
        debug_assert_eq!(other, res);
    }
    res
}

/// `\kappa_\rot(x, y)` should equal `\delta_{x,rot(y)}` on hyperprism.
///
/// `omega_pows` must have length `2^{l_skip}`.
pub fn eval_rot_kernel_prism<F: TwoAdicField>(l_skip: usize, x: &[F], y: &[F]) -> F {
    let omega = F::two_adic_generator(l_skip);

    let (eq_cube, rot_cube) = eval_eq_rot_cube(&x[1..], &y[1..]);
    // If not at boundary of D, just rotate in D, don't change cube coordinates. Otherwise at
    // boundary, rotate the cube
    eval_eq_uni(l_skip, x[0], y[0] * omega) * eq_cube
        + eval_eq_uni_at_one(l_skip, x[0])
            * eval_eq_uni_at_one(l_skip, y[0] * omega)
            * (rot_cube - eq_cube)
}

/// MLE of cyclic rotation kernel on hypercube
pub fn eval_eq_rot_cube<F: Field>(x: &[F], y: &[F]) -> (F, F) {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    // Recursive formula: rot(x, y) = x[0] * (1 - y[0]) * eq(x[1..], y[1..]) + (1 - x[0]) y[0] *
    // rot(x[1..], y[1..])
    let mut rot = F::ONE;
    let mut eq = F::ONE;
    for i in (0..n).rev() {
        rot = x[i] * (F::ONE - y[i]) * eq + (F::ONE - x[i]) * y[i] * rot;
        eq *= x[i] * y[i] + (F::ONE - x[i]) * (F::ONE - y[i]);
    }
    (eq, rot)
}

// Source: https://github.com/starkware-libs/stwo/blob/dev/crates/stwo/src/prover/lookups/utils.rs#L12
/// Univariate polynomial in coefficient form.
#[derive(Clone, Debug)]
pub struct UnivariatePoly<F>(pub(crate) Vec<F>);

impl<F> UnivariatePoly<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        Self(coeffs)
    }

    pub fn coeffs(&self) -> &[F] {
        &self.0
    }

    pub fn coeffs_mut(&mut self) -> &mut Vec<F> {
        &mut self.0
    }

    pub fn into_coeffs(self) -> Vec<F> {
        self.0
    }
}

impl<F: Field> UnivariatePoly<F> {
    pub fn eval_at_point<EF: ExtensionField<F>>(&self, x: EF) -> EF {
        horner_eval(&self.0, x)
    }
}

/// Evaluates univariate polynomial using [Horner's method].
///
/// [Horner's method]: https://en.wikipedia.org/wiki/Horner%27s_method
pub fn horner_eval<F1, F2, F3>(coeffs: &[F1], x: F2) -> F3
where
    F1: Field,
    F2: Field,
    F3: Field + Add<F1, Output = F3>,
    F3: Mul<F2, Output = F3>,
{
    coeffs.iter().rfold(F3::ZERO, |acc, coeff| acc * x + *coeff)
}

/// Interpolates a linear polynomial through points (0, evals[0]), (1, evals[1])
/// and evaluates it at x.
#[inline(always)]
pub fn interpolate_linear_at_01<F: Field>(evals: &[F; 2], x: F) -> F {
    let p = evals[1] - evals[0];
    p * x + evals[0]
}

/// Interpolates a quadratic polynomial through points (0, evals[0]), (1, evals[1]),
/// (2, evals[2])  and evaluates it at x.
#[inline(always)]
pub fn interpolate_quadratic_at_012<F: Field>(evals: &[F; 3], x: F) -> F {
    let s1 = evals[1] - evals[0];
    let s2 = evals[2] - evals[1];
    let p = (s2 - s1).halve();
    let q = s1 - p;
    (p * x + q) * x + evals[0]
}

/// Interpolates a cubic polynomial through points (0, evals[0]), (1, evals[1]),
/// (2, evals[2]), (3, evals[3]) and evaluates it at x.
#[inline(always)]
pub fn interpolate_cubic_at_0123<F: Field>(evals: &[F; 4], x: F) -> F {
    let inv6 = F::from_u64(6).inverse();

    let s1 = evals[1] - evals[0];
    let s2 = evals[2] - evals[0];
    let s3 = evals[3] - evals[0];

    let d3 = s3 - (s2 - s1) * F::from_u64(3);

    let p = d3 * inv6;
    let q = (s2 - d3).halve() - s1;
    let r = s1 - p - q;

    ((p * x + q) * x + r) * x + evals[0]
}

pub struct ExpPowers2<T> {
    current: Option<T>,
}

impl<T: Squarable + PrimeCharacteristicRing> Iterator for ExpPowers2<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(curr) = self.current.take() {
            let next = curr.square();
            self.current = Some(next);
            Some(curr)
        } else {
            None
        }
    }
}

pub trait Squarable: PrimeCharacteristicRing + Clone {
    #[inline]
    fn exp_powers_of_2(&self) -> ExpPowers2<Self> {
        ExpPowers2 {
            current: Some(self.clone()),
        }
    }
}

impl<T: PrimeCharacteristicRing + Clone> Squarable for T {}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use openvm_stark_sdk::config::baby_bear_poseidon2::*;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn test_interpolate_linear() {
        let evals = [
            EF::from_u64(20), // s(0)
            EF::from_u64(10), // s(1)
        ];

        // Test interpolation at known points
        assert_eq!(interpolate_linear_at_01(&evals, EF::ZERO), evals[0]);
        assert_eq!(interpolate_linear_at_01(&evals, EF::ONE), evals[1]);
    }

    #[test]
    fn test_interpolate_quadratic() {
        let evals = [
            EF::from_u64(20), // s(0)
            EF::from_u64(10), // s(1)
            EF::from_u64(18), // s(2)
        ];

        // Test interpolation at known points
        assert_eq!(interpolate_quadratic_at_012(&evals, EF::ZERO), evals[0]);
        assert_eq!(interpolate_quadratic_at_012(&evals, EF::ONE), evals[1]);
        assert_eq!(
            interpolate_quadratic_at_012(&evals, EF::from_u64(2)),
            evals[2]
        );
    }

    #[test]
    fn test_interpolate_cubic() {
        let evals = [
            EF::from_u64(20), // s(0)
            EF::from_u64(10), // s(1)
            EF::from_u64(18), // s(2)
            EF::from_u64(28), // s(3)
        ];

        // Test interpolation at known points
        assert_eq!(interpolate_cubic_at_0123(&evals, EF::ZERO), evals[0]);
        assert_eq!(interpolate_cubic_at_0123(&evals, EF::ONE), evals[1]);
        assert_eq!(interpolate_cubic_at_0123(&evals, EF::from_u64(2)), evals[2]);
        assert_eq!(interpolate_cubic_at_0123(&evals, EF::from_u64(3)), evals[3]);
    }

    #[test]
    fn test_exp_powers_of_2() {
        let x = F::from_u32(3);
        let s = x.exp_powers_of_2().take(3).collect_vec();
        assert_eq!(s, vec![x, x * x, x * x * x * x],);
    }

    #[test]
    fn test_eval_in_uni() {
        let l = 3;
        let n = -2;
        let u_0 = F::from_u32(12345);
        let ind = eval_in_uni(l, n, u_0);
        let expected = (u_0.exp_power_of_2(l) - F::ONE)
            * (u_0.exp_power_of_2(l.wrapping_add_signed(n)) - F::ONE).inverse()
            * F::from_usize(1 << n.unsigned_abs()).inverse();
        assert_eq!(ind, expected);
    }
}
