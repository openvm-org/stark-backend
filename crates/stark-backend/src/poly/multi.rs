//! Copied from starkware-libs/stwo under Apache-2.0 license.
use std::{
    iter::zip,
    ops::{Deref, DerefMut},
};

use p3_field::{ExtensionField, Field};

use super::uni::UnivariatePolynomial;

/// Represents a multivariate polynomial `g(x_1, ..., x_n)`.
pub trait MultivariatePolyOracle<F> {
    /// For an n-variate polynomial, returns n.
    fn arity(&self) -> usize;

    /// Returns the sum of `g(x_1, x_2, ..., x_n)` over all `(x_2, ..., x_n)` in `{0, 1}^(n-1)` as a polynomial in `x_1`.
    fn marginalize_first(&self, claim: F) -> UnivariatePolynomial<F>;

    /// Returns the multivariate polynomial `h(x_2, ..., x_n) = g(alpha, x_2, ..., x_n)`.
    fn partial_evaluation(self, alpha: F) -> Self;
}

/// Multilinear extension of the function defined on the boolean hypercube.
///
/// The evaluations are stored in lexicographic order.
#[derive(Debug, Clone)]
pub struct Mle<F> {
    pub evals: Vec<F>,
}

impl<F: Field> Mle<F> {
    /// Creates a [`Mle`] from evaluations of a multilinear polynomial on the boolean hypercube.
    ///
    /// # Panics
    ///
    /// Panics if the number of evaluations is not a power of two.
    pub fn from_vec(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        Self { evals }
    }

    /// Evaluates the multilinear polynomial at `point`.
    pub fn eval(&self, point: &[F]) -> F {
        pub fn eval_rec<F: Field>(mle_evals: &[F], p: &[F]) -> F {
            match p {
                [] => mle_evals[0],
                &[p_i, ref p @ ..] => {
                    let (lhs, rhs) = mle_evals.split_at(mle_evals.len() / 2);
                    let lhs_eval = eval_rec(lhs, p);
                    let rhs_eval = eval_rec(rhs, p);
                    // Equivalent to `eq(0, p_i) * lhs_eval + eq(1, p_i) * rhs_eval`.
                    p_i * (rhs_eval - lhs_eval) + lhs_eval
                }
            }
        }

        let mle_evals = self.evals.clone();
        eval_rec(&mle_evals, point)
    }
}

impl<F: Field> MultivariatePolyOracle<F> for Mle<F> {
    fn arity(&self) -> usize {
        self.evals.len().ilog2() as usize
    }

    fn marginalize_first(&self, claim: F) -> UnivariatePolynomial<F> {
        let x0 = F::ZERO;
        let x1 = F::ONE;

        let y0 = self[0..self.len() / 2]
            .iter()
            .fold(F::ZERO, |acc, x| acc + *x);
        let y1 = claim - y0;

        UnivariatePolynomial::from_points(&[(x0, y0), (x1, y1)])
    }

    fn partial_evaluation(self, alpha: F) -> Self {
        let midpoint = self.len() / 2;
        let (lhs_evals, rhs_evals) = self.split_at(midpoint);

        let res = zip(lhs_evals, rhs_evals)
            .map(|(&lhs_eval, &rhs_eval)| alpha * (rhs_eval - lhs_eval) + lhs_eval)
            .collect();

        Mle::from_vec(res)
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

/// Computes eq_n(r, x) for fixed r and each x in {0, 1}^n, where
///          eq_n(r, x) = prod_i (r_i x_i + (1 - r_i)(1 - x_i)).
pub fn hypercube_eq_partial<F: Field>(r: &[F]) -> Vec<F> {
    let height = 1 << r.len();

    // This runs in time O(n 2^n) but can be optimized to O(2^n).
    (0..height)
        .map(|k| {
            let mut eq_eval = F::ONE;
            let mut k_left = k;
            for &r_i in r.iter().rev() {
                if k_left % 2 == 0 {
                    eq_eval *= F::ONE - r_i;
                } else {
                    eq_eval *= r_i;
                }
                k_left /= 2;
            }
            eq_eval
        })
        .collect()
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
        let mle = Mle::from_vec(evals);
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
        let mle = Mle::from_vec(evals);
        // (1 - x_1) + 2 (1 - x_1) + 3 x_1 + 4 x_1
        let poly = mle.marginalize_first(sum);

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
        let mle = Mle::from_vec(evals);
        let alpha = BabyBear::from_canonical_u32(2);
        // -(1 - x_2) - 2 x_2 + 6 (1 - x_2) + 8 x_2 = x_2 + 5
        let partial_eval = mle.partial_evaluation(alpha);

        assert_eq!(
            partial_eval.eval(&[BabyBear::ZERO]),
            BabyBear::from_canonical_u32(5)
        );
        assert_eq!(
            partial_eval.eval(&[BabyBear::ONE]),
            BabyBear::from_canonical_u32(6)
        );
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
        let eqs = hypercube_eq_partial(&r);

        let mut expected = vec![BabyBear::ZERO; 16];
        expected[6] = BabyBear::ONE;
        assert_eq!(eqs, expected);
    }

    #[test]
    fn test_eqs_at_dim_3() {
        let mut rng = create_seeded_rng();

        let [a, b, c] = rng.gen::<[BabyBear; 3]>();

        let eqs = hypercube_eq_partial(&[a, b, c]);

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
        let mle = Mle::from_vec(vals.to_vec());

        let r: [BabyBear; 10] = rng.gen();

        let eqs_at_r = hypercube_eq_partial(&r);
        let inner_prod: BabyBear = eqs_at_r
            .iter()
            .zip(vals.iter())
            .map(|(&eq, &val)| eq * val)
            .sum();
        assert_eq!(inner_prod, mle.eval(&r));
    }
}
