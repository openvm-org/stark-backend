use core::ops::{Add, Sub};
use std::{iter::zip, ops::Mul};

use itertools::Itertools;
use p3_dft::{Radix2Bowers, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, FieldAlgebra, TwoAdicField, batch_multiplicative_inverse};
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::prover::poly::evals_eq_hypercube;

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

pub fn eval_eq_prism<F: Field>(l_skip: usize, x: &[F], y: &[F]) -> F {
    eval_eq_uni(l_skip, x[0], y[0]) * eval_eq_mle(&x[1..], &y[1..])
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
    let eq_xi_evals = evals_eq_hypercube(xi_1);
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
pub struct UnivariatePoly<F>(pub(crate) Vec<F>);

impl<F: TwoAdicField> UnivariatePoly<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        Self(coeffs)
    }

    pub fn coeffs(&self) -> &[F] {
        &self.0
    }

    pub fn eval_at_point<EF: ExtensionField<F>>(&self, x: EF) -> EF {
        horner_eval(&self.0, x)
    }

    /// Computes P(1), P(omega), ..., P(omega^{n-1}).
    fn chirp_z(poly: &[F], omega: F, n: usize) -> Vec<F> {
        if n == 0 {
            return Vec::new();
        }
        if poly.is_empty() {
            return vec![F::ZERO; n];
        }
        let s = poly.len() + n;
        let omega_powers = (0..(s as u64))
            .map(|i| omega.exp_u64(i * (i.saturating_sub(1)) / 2))
            .collect_vec();
        let omega_powers_inv = batch_multiplicative_inverse(&omega_powers);
        let mut p = zip(poly, &omega_powers_inv)
            .map(|(&c, &inv)| c * inv)
            .collect_vec();
        let mut q = omega_powers.iter().rev().copied().collect_vec();

        let dft_deg = (p.len() + q.len() - 1).next_power_of_two();
        p.resize(dft_deg, F::ZERO);
        q.resize(dft_deg, F::ZERO);
        let dft = Radix2Bowers;
        p = dft.dft(p);
        q = dft.dft(q);
        for (x, y) in p.iter_mut().zip(q.iter()) {
            *x *= *y;
        }
        p = dft.idft(p);
        zip(p.into_iter().skip(n).take(n).rev(), omega_powers_inv)
            .map(|(x, inv)| x * inv)
            .collect()
    }

    /// Given z and n, find product (1 - x)(1 - zx)...(1 - z^{n-1}x).
    /// If n is odd, this can be trivially computed from n - 1.
    /// Otherwise, F_n(x) = F_{n/2}(x) * F_{n/2}(x * z^{n/2}).
    fn geometric_sequence_linear_product_helper(dft: &Radix2Bowers, z: F, n: usize) -> Vec<F> {
        if n == 1 {
            vec![F::ONE, F::NEG_ONE]
        } else if n % 2 == 1 {
            let mut prev = Self::geometric_sequence_linear_product_helper(dft, z, n - 1);
            let zp = z.exp_u64((n - 1) as u64);
            prev.push(F::ZERO);
            for i in (1..prev.len()).rev() {
                let value = prev[i - 1] * zp;
                prev[i] -= value;
            }
            prev
        } else {
            let mut prev = Self::geometric_sequence_linear_product_helper(dft, z, n / 2);
            let zp = z.exp_u64((n / 2) as u64);
            let mut another = prev
                .iter()
                .zip(zp.powers())
                .map(|(a, b)| *a * b)
                .collect_vec();
            let len = prev.len().next_power_of_two() * 2;
            prev.resize(len, F::ZERO);
            another.resize(len, F::ZERO);
            prev = dft.dft(prev);
            another = dft.dft(another);
            for (x, y) in prev.iter_mut().zip(another.into_iter()) {
                *x *= y;
            }
            prev = dft.idft(prev);
            prev.truncate(n + 1);
            prev
        }
    }

    /// Constructs the polynomial in coefficient form from its evaluations on
    /// `{omega^0,...,omega^d}` where `d` is the degree of the polynomial. Here `omega` is a
    /// (fixed) generator of the two-adic subgroup of order `(d+1).next_power_of_two()`.
    #[instrument(level = "debug", skip_all)]
    pub fn from_evals(evals: &[F]) -> Self {
        let n = evals.len();
        let log_n = log2_ceil_usize(n);
        let omega = F::two_adic_generator(log_n);
        let omega_pows = omega.powers().take((1 << log_n) + 1).collect_vec();
        if n == 0 {
            return Self(Vec::new());
        }
        if n == 1 {
            return Self(vec![evals[0]]);
        }

        // We know that, by Lagrange interpolation,
        // P(x) = \sum_i evals[i] * \prod_{j\neq i} (x - omega^j) / (omega^i - omega^j).
        // Let y[i] = evals[i] / (omega^{(n-1) * i} * prod_{j < n - 1 - i}(1 - omega^j) * prod_{j <
        // i}(1 - omega^{-j})). Then P(x) = \sum_i y[i] * \prod_{j\neq i} (x - omega^j).

        let mut positive_denoms = vec![F::ONE; n];
        let mut negative_denoms = vec![F::ONE; n];
        for i in 0..(n - 1) {
            positive_denoms[i + 1] = positive_denoms[i] / (F::ONE - omega_pows[i + 1]);
            negative_denoms[i + 1] =
                negative_denoms[i] / (F::ONE - omega_pows[(1 << log_n) - 1 - i]);
        }
        let omega_inv = omega_pows[(1 << log_n) - 1];
        let y = (0..n)
            .map(|i| {
                evals[i]
                    * omega_inv.exp_u64(((n - 1) * i) as u64)
                    * negative_denoms[i]
                    * positive_denoms[n - 1 - i]
            })
            .collect_vec();

        // If we reverse both P and replace all (x - a) with (1 - ax), we'll still have an equality.
        // So from now we assume that P(x) = \sum_i y[i] * \prod_{j\neq i} (1 - omega^j * x).
        // If we divide everything by Q(x) = \prod_i (1 - omega^i * x), then we'll have
        // P(x) / Q(x) = \sum_i y[i] / (1 - omega^i * x).

        // We want to find the first n coefficients of the right-hand side.
        // [x^k](\sum_i y[i] / (1 - x * omega^i)) = \sum_i y[i] / (omega^{ik}) = Y(omega^k).

        let mut rhs = Self::chirp_z(&y, omega, n);

        // Now we need the denominator in the left-hand side.
        let dft = Radix2Bowers;
        let mut denom = Self::geometric_sequence_linear_product_helper(&dft, omega, n);

        let len = (denom.len() + rhs.len() - 1).next_power_of_two();
        denom.resize(len, F::ZERO);
        rhs.resize(len, F::ZERO);
        denom = dft.dft(denom);
        rhs = dft.dft(rhs);
        let res = denom.into_iter().zip(rhs).map(|(a, b)| a * b).collect_vec();
        let mut res = dft.idft(res);
        res.truncate(n);
        // Remember that P(x) is reversed
        res.reverse();
        Self(res)
    }

    #[instrument(level = "debug", skip_all)]
    pub fn lagrange_interpolate(points: &[F], evals: &[F]) -> Self {
        assert_eq!(points.len(), evals.len());
        let len = points.len();

        // Special case: empty or single evaluation
        if len == 0 {
            return Self(vec![]);
        }
        if len == 1 {
            return Self(vec![evals[0]]);
        }

        // Lagrange interpolation algorithm
        // P(x) = sum_{i=0}^{len-1} evals[i] * L_i(x)
        // where L_i(x) = prod_{j != i} (x - points[j]) / (points[i] - points[j])

        // Step 1: Compute all denominators (points[i] - points[j]) for i != j
        let mut denominators = Vec::with_capacity(len * (len - 1));
        for i in 0..len {
            for j in 0..len {
                if i != j {
                    denominators.push(points[i] - points[j]);
                }
            }
        }

        // Step 2: Batch invert all denominators
        let inv_denominators = batch_multiplicative_inverse(&denominators);

        // Step 3: Build coefficient form by accumulating Lagrange basis polynomials
        let mut coeffs = vec![F::ZERO; len];

        // Reusable workspace for Lagrange polynomial computation
        let mut lagrange_poly = Vec::with_capacity(len);

        #[allow(clippy::needless_range_loop)]
        for i in 0..len {
            // Skip if evaluation is zero (optimization)
            if evals[i] == F::ZERO {
                continue;
            }

            // Build L_i(x) in coefficient form using polynomial multiplication
            // L_i(x) = prod_{j != i} (x - points[j]) / (points[i] - points[j])

            // Start with constant polynomial 1
            lagrange_poly.clear();
            lagrange_poly.push(F::ONE);

            // Get the precomputed inverse denominators for this i
            let inv_denom_start = i * (len - 1);
            let mut inv_idx = 0;

            // Multiply by (x - points[j]) / (points[i] - points[j]) for each j != i
            #[allow(clippy::needless_range_loop)]
            for j in 0..len {
                if i != j {
                    let scale = inv_denominators[inv_denom_start + inv_idx];
                    inv_idx += 1;

                    // Multiply lagrange_poly by (x - points[j]) * scale in place
                    // This is equivalent to: lagrange_poly * (x - points[j]) * scale
                    // = lagrange_poly * x * scale - lagrange_poly * points[j] * scale

                    lagrange_poly.push(F::ZERO); // Extend by one for the new highest degree term
                    for k in (1..lagrange_poly.len()).rev() {
                        let prev_coeff = lagrange_poly[k - 1] * scale;
                        lagrange_poly[k] += prev_coeff;
                        lagrange_poly[k - 1] = -prev_coeff * points[j];
                    }
                }
            }

            // Add evals[i] * L_i(x) to the result
            for (k, &coeff) in lagrange_poly.iter().enumerate() {
                coeffs[k] += evals[i] * coeff;
            }
        }

        Self(coeffs)
    }

    /// Constructs the polynomial in coefficient form from its evaluations on a smooth subgroup of
    /// `F^*` by performing inverse DFT.
    ///
    /// Requires that `evals.len()` is a power of 2.
    pub fn from_evals_idft(evals: &[F]) -> Self {
        // NOTE[jpw]: Use Bowers instead of Dit to avoid RefCell
        let dft = Radix2Bowers;
        let coeffs = dft.idft(evals.to_vec());
        Self(coeffs)
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
    let inv6 = F::from_canonical_u64(6).inverse();

    let s1 = evals[1] - evals[0];
    let s2 = evals[2] - evals[0];
    let s3 = evals[3] - evals[0];

    let d3 = s3 - (s2 - s1) * F::from_canonical_u64(3);

    let p = d3 * inv6;
    let q = (s2 - d3).halve() - s1;
    let r = s1 - p - q;

    ((p * x + q) * x + r) * x + evals[0]
}

pub struct ExpPowers2<T> {
    current: Option<T>,
}

impl<T: Squarable + FieldAlgebra> Iterator for ExpPowers2<T> {
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

pub trait Squarable: FieldAlgebra + Clone {
    #[inline]
    fn exp_powers_of_2(&self) -> ExpPowers2<Self> {
        ExpPowers2 {
            current: Some(self.clone()),
        }
    }
}

impl<T: FieldAlgebra + Clone> Squarable for T {}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;
    use p3_field::{FieldAlgebra, TwoAdicField};
    use p3_util::log2_ceil_usize;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::{EF, F};

    #[test]
    fn test_lagrange_interpolation_round_trip() {
        let mut rng = StdRng::seed_from_u64(0);
        // Test various polynomial degrees
        for degree in 0..8usize {
            let num_evals: usize = degree + 1;

            // Create random coefficients for a polynomial of given degree
            let mut original_coeffs = vec![];
            for _ in 0..num_evals {
                // Use deterministic values for reproducibility
                original_coeffs.push(F::from_wrapped_u32(rng.random()));
            }

            // Generate evaluation points (powers of omega)
            let log_domain_size = log2_ceil_usize(num_evals);
            let omega = F::two_adic_generator(log_domain_size);

            // Evaluate polynomial at these points
            let mut evals = vec![];
            let points = omega.powers().take(num_evals).collect_vec();
            for &x in &points {
                evals.push(horner_eval(&original_coeffs, x));
            }

            // Reconstruct polynomial from evaluations using Lagrange interpolation
            let reconstructed_poly = UnivariatePoly::lagrange_interpolate(&points, &evals);

            // Verify coefficients match (up to the original degree)
            for (i, (coeff, reconstructed_coeff)) in
                zip(original_coeffs, reconstructed_poly.0).enumerate()
            {
                assert_eq!(
                    coeff, reconstructed_coeff,
                    "Coefficient mismatch at index {} for degree {} polynomial",
                    i, degree
                );
            }
        }
    }

    #[test]
    fn test_chirp_z() {
        let mut rng = StdRng::seed_from_u64(0);
        for degree in 0..8usize {
            let num_evals: usize = degree + 1;

            // Create random coefficients for a polynomial of given degree
            let mut original_coeffs = vec![];
            for _ in 0..num_evals {
                // Use deterministic values for reproducibility
                original_coeffs.push(F::from_wrapped_u32(rng.random()));
            }

            let log_domain_size = log2_ceil_usize(num_evals);
            let omega = F::two_adic_generator(log_domain_size);

            // Evaluate polynomial at these points
            let mut evals = vec![];
            let points = omega.powers().take(num_evals).collect_vec();
            for &x in &points {
                evals.push(horner_eval(&original_coeffs, x));
            }

            assert_eq!(
                evals,
                UnivariatePoly::chirp_z(&original_coeffs, omega, num_evals)
            );

            let reconstructed_poly = UnivariatePoly::from_evals(&evals);

            // Verify coefficients match (up to the original degree)
            for (i, (coeff, reconstructed_coeff)) in
                zip(original_coeffs, reconstructed_poly.0).enumerate()
            {
                assert_eq!(
                    coeff, reconstructed_coeff,
                    "Coefficient mismatch at index {} for degree {} polynomial",
                    i, degree
                );
            }
        }
    }

    #[test]
    fn test_interpolate_linear() {
        let evals = [
            EF::from_canonical_u64(20), // s(0)
            EF::from_canonical_u64(10), // s(1)
        ];

        // Test interpolation at known points
        assert_eq!(interpolate_linear_at_01(&evals, EF::ZERO), evals[0]);
        assert_eq!(interpolate_linear_at_01(&evals, EF::ONE), evals[1]);
    }

    #[test]
    fn test_interpolate_quadratic() {
        let evals = [
            EF::from_canonical_u64(20), // s(0)
            EF::from_canonical_u64(10), // s(1)
            EF::from_canonical_u64(18), // s(2)
        ];

        // Test interpolation at known points
        assert_eq!(interpolate_quadratic_at_012(&evals, EF::ZERO), evals[0]);
        assert_eq!(interpolate_quadratic_at_012(&evals, EF::ONE), evals[1]);
        assert_eq!(
            interpolate_quadratic_at_012(&evals, EF::from_canonical_u64(2)),
            evals[2]
        );
    }

    #[test]
    fn test_interpolate_cubic() {
        let evals = [
            EF::from_canonical_u64(20), // s(0)
            EF::from_canonical_u64(10), // s(1)
            EF::from_canonical_u64(18), // s(2)
            EF::from_canonical_u64(28), // s(3)
        ];

        // Test interpolation at known points
        assert_eq!(interpolate_cubic_at_0123(&evals, EF::ZERO), evals[0]);
        assert_eq!(interpolate_cubic_at_0123(&evals, EF::ONE), evals[1]);
        assert_eq!(
            interpolate_cubic_at_0123(&evals, EF::from_canonical_u64(2)),
            evals[2]
        );
        assert_eq!(
            interpolate_cubic_at_0123(&evals, EF::from_canonical_u64(3)),
            evals[3]
        );
    }

    #[test]
    fn test_exp_powers_of_2() {
        let x = F::from_canonical_u32(3);
        let s = x.exp_powers_of_2().take(3).collect_vec();
        assert_eq!(s, vec![x, x * x, x * x * x * x],);
    }
}
