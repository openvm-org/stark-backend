//! Copied from starkware-libs/stwo under Apache-2.0 license.
use std::{
    iter::zip,
    ops::{Deref, DerefMut},
};

use p3_maybe_rayon::prelude::*;

use p3_field::{ExtensionField, Field};

use super::uni::UnivariatePolynomial;

/// Represents a multivariate polynomial `g(x_1, ..., x_n)`.
pub trait MultivariatePolyOracle<F>: Send + Sync {
    /// For an n-variate polynomial, returns n.
    fn arity(&self) -> usize;

    /// Returns the sum of `g(x_1, x_2, ..., x_n)` over all `(x_2, ..., x_n)` in `{0, 1}^(n-1)` as a polynomial in `x_1`.
    fn partial_hypercube_sum(&self, claim: F) -> UnivariatePolynomial<F>;

    /// Returns the multivariate polynomial `h(x_2, ..., x_n) = g(alpha, x_2, ..., x_n)`.
    fn fix_first_in_place(&mut self, alpha: F);
}

/// Multilinear extension of the function defined on the boolean hypercube.
///
/// The evaluations are stored in lexicographic order.
#[derive(Debug, Clone)]
pub struct Mle<F> {
    evals: Vec<F>,
}

impl<F: Field> Mle<F> {
    /// Creates a [`Mle`] from evaluations of a multilinear polynomial on the boolean hypercube.
    ///
    /// # Panics
    ///
    /// Panics if the number of evaluations is not a power of two.
    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        Self { evals }
    }

    /// Evaluates the multilinear extension at `point`.
    pub fn eval(&self, point: &[F]) -> F {
        let mut buf = self.evals.clone();
        Self::eval_slice(&mut buf, point)
    }

    /// Evaluates the multilinear extension given by hypercube evaluations `evals` at `point`.
    ///
    /// Uses `evals` as a computational buffer.
    ///
    /// # Panics
    ///
    /// Panics if `evals.len() != 2^point.len()`
    pub fn eval_slice(evals: &mut [F], point: &[F]) -> F {
        let buf = evals;
        let n = point.len();
        let mut len = buf.len();

        assert_eq!(len, 1 << n, "Point dimension mismatch");

        for &x_i in point.iter().rev() {
            len /= 2;
            for i in 0..len {
                let a = buf[2 * i];
                let b = buf[2 * i + 1];
                buf[i] = a + x_i * (b - a);
            }
        }
        buf[0]
    }
}

impl<F: Field> MultivariatePolyOracle<F> for Mle<F> {
    fn arity(&self) -> usize {
        self.evals.len().ilog2() as usize
    }

    fn partial_hypercube_sum(&self, claim: F) -> UnivariatePolynomial<F> {
        let mut y0 = F::ZERO;
        for i in 0..self.len() / 2 {
            y0 += self[i];
        }

        let y1 = claim - y0;

        // Direct degree-1 polynomial: f(x) = y0 + (y1 - y0) * x
        let slope = y1 - y0;

        UnivariatePolynomial::from_coeffs(vec![y0, slope])
    }

    fn fix_first_in_place(&mut self, alpha: F) {
        let midpoint = self.len() / 2;
        let (lhs_evals, rhs_evals) = self.split_at_mut(midpoint);
        lhs_evals
            .par_iter_mut()
            .zip(rhs_evals.par_iter())
            .for_each(|(lhs, &rhs)| {
                *lhs += alpha * (rhs - *lhs);
            });
        self.evals.truncate(midpoint);
    }
}

impl<F> Deref for Mle<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

impl<F: Field> DerefMut for Mle<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evals
    }
}

/// Evaluates the boolean Lagrange basis polynomial `eq(x, y)`.
///
/// Formally, the boolean Lagrange basis polynomial is defined as:
/// ```text
/// eq(x_1, \dots, x_n, y_1, \dots, y_n) = \prod_{i=1}^n (x_i * y_i + (1 - x_i) * (1 - y_i)).
/// ```
/// For boolean vectors `x` and `y`, the function returns `1` if `x` equals `y` and `0` otherwise.
///
/// # Panics
/// - Panics if `x` and `y` have different lengths.
pub fn hypercube_eq<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    zip(x, y)
        .map(|(&xi, &yi)| xi * yi + (xi - F::ONE) * (yi - F::ONE))
        .product()
}

/// Computes `hypercube_eq(0, assignment) * eval0 + hypercube_eq(1, assignment) * eval1`.
pub fn fold_mle_evals<F, EF>(assignment: EF, eval0: F, eval1: F) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    assignment * (eval1 - eval0) + eval0
}

/// Computes eq_n(x, y) for fixed x and each y in {0, 1}^n, where
///          eq_n(x, y) = prod_i (x_i y_i + (1 - x_i)(1 - y_i)).
pub fn hypercube_eq_over_y<F: Field>(y: &[F]) -> Vec<F> {
    let n = y.len();
    let size = 1 << n;

    let mut result = vec![F::ZERO; size];
    result[0] = F::ONE;

    let mut cur_size = 1;

    for &yi in y {
        let one_minus_yi = F::ONE - yi;

        // Fill in reverse to avoid overwriting values we still need
        for i in (0..cur_size).rev() {
            let val = result[i];
            result[2 * i] = val * one_minus_yi; // x_i = 0
            result[2 * i + 1] = val * yi; // x_i = 1
        }
        cur_size *= 2;
    }
    result
}

#[cfg(test)]
mod test {
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_baby_bear::BabyBear;
    use p3_field::FieldAlgebra;
    use rand::Rng;

    use super::*;

    #[test]
    fn test_mle_evaluation() {
        let evals = vec![
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(3),
            BabyBear::from_canonical_u32(4),
        ];
        // (1 - x_1)(1 - x_2) + 2 (1 - x_1) x_2 + 3 x_1 (1 - x_2) + 4 x_1 x_2
        let mle = Mle::new(evals);
        let point = vec![
            BabyBear::from_canonical_u32(0),
            BabyBear::from_canonical_u32(0),
        ];
        assert_eq!(mle.eval(&point), BabyBear::from_canonical_u32(1));

        let point = vec![
            BabyBear::from_canonical_u32(0),
            BabyBear::from_canonical_u32(1),
        ];
        assert_eq!(mle.eval(&point), BabyBear::from_canonical_u32(2));

        let point = vec![
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(0),
        ];
        assert_eq!(mle.eval(&point), BabyBear::from_canonical_u32(3));

        let point = vec![
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(1),
        ];
        assert_eq!(mle.eval(&point), BabyBear::from_canonical_u32(4));

        // Out of domain evaluation
        let point = vec![
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(2),
        ];
        assert_eq!(mle.eval(&point), BabyBear::from_canonical_u32(7));
    }

    #[test]
    fn test_mle_marginalize_first() {
        let evals = vec![
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(3),
            BabyBear::from_canonical_u32(4),
        ];
        let sum = BabyBear::from_canonical_u32(10);

        // (1 - x_1)(1 - x_2) + 2 (1 - x_1) x_2 + 3 x_1 (1 - x_2) + 4 x_1 x_2
        let mle = Mle::new(evals);
        // (1 - x_1) + 2 (1 - x_1) + 3 x_1 + 4 x_1
        let poly = mle.partial_hypercube_sum(sum);

        assert_eq!(
            poly.evaluate(BabyBear::ZERO),
            BabyBear::from_canonical_u32(3)
        );
        assert_eq!(
            poly.evaluate(BabyBear::ONE),
            BabyBear::from_canonical_u32(7)
        );
    }

    #[test]
    fn test_mle_partial_evaluation() {
        let evals = vec![
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(3),
            BabyBear::from_canonical_u32(4),
        ];
        // (1 - x_1)(1 - x_2) + 2 (1 - x_1) x_2 + 3 x_1 (1 - x_2) + 4 x_1 x_2
        let mut mle = Mle::new(evals);
        let alpha = BabyBear::from_canonical_u32(2);
        // -(1 - x_2) - 2 x_2 + 6 (1 - x_2) + 8 x_2 = x_2 + 5
        mle.fix_first_in_place(alpha);

        assert_eq!(mle.eval(&[BabyBear::ZERO]), BabyBear::from_canonical_u32(5));
        assert_eq!(mle.eval(&[BabyBear::ONE]), BabyBear::from_canonical_u32(6));
    }

    #[test]
    fn eq_identical_hypercube_points_returns_one() {
        let zero = BabyBear::ZERO;
        let one = BabyBear::ONE;
        let a = &[one, zero, one];

        let eq_eval = hypercube_eq(a, a);

        assert_eq!(eq_eval, one);
    }

    #[test]
    fn eq_different_hypercube_points_returns_zero() {
        let zero = BabyBear::ZERO;
        let one = BabyBear::ONE;
        let a = &[one, zero, one];
        let b = &[one, zero, zero];

        let eq_eval = hypercube_eq(a, b);

        assert_eq!(eq_eval, zero);
    }

    #[test]
    #[should_panic]
    fn eq_different_size_points() {
        let zero = BabyBear::ZERO;
        let one = BabyBear::ONE;

        hypercube_eq(&[zero, one], &[zero]);
    }

    #[test]
    fn test_eqs_at_hypercube() {
        // Try on a hypercube point.
        let r = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::ONE, BabyBear::ZERO];
        let eqs = hypercube_eq_over_y(&r);

        let mut expected = vec![BabyBear::ZERO; 16];
        expected[6] = BabyBear::ONE;
        assert_eq!(eqs, expected);
    }

    #[test]
    fn test_eqs_at_dim_3() {
        let mut rng = create_seeded_rng();

        let [a, b, c] = rng.gen::<[BabyBear; 3]>();

        let eqs = hypercube_eq_over_y(&[a, b, c]);

        let expected = vec![
            (BabyBear::ONE - a) * (BabyBear::ONE - b) * (BabyBear::ONE - c),
            (BabyBear::ONE - a) * (BabyBear::ONE - b) * c,
            (BabyBear::ONE - a) * b * (BabyBear::ONE - c),
            (BabyBear::ONE - a) * b * c,
            a * (BabyBear::ONE - b) * (BabyBear::ONE - c),
            a * (BabyBear::ONE - b) * c,
            a * b * (BabyBear::ONE - c),
            a * b * c,
        ];
        assert_eq!(eqs, expected);
    }

    #[test]
    fn test_eqs_at_vs_mle() {
        let mut rng = create_seeded_rng();

        let vals: [BabyBear; 1024] = rng.gen();
        let mle = Mle::new(vals.to_vec());

        let r: [BabyBear; 10] = rng.gen();

        let eqs_at_r = hypercube_eq_over_y(&r);
        let inner_prod: BabyBear = eqs_at_r
            .iter()
            .zip(vals.iter())
            .map(|(&eq, &val)| eq * val)
            .sum();
        assert_eq!(inner_prod, mle.eval(&r));
    }
}
