use core::ops::Mul;
use std::iter::zip;

use getset::Getters;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{
    poly_common::{UnivariatePoly, eval_eq_uni},
    prover::{ColMajorMatrix, ColMajorMatrixView},
};

/// Multilinear extension polynomial, in coefficient form.
///
/// Length of `coeffs` is `2^n` where `n` is hypercube dimension.
/// Indexing of `coeffs` is to use the little-endian encoding of integer index as the powers of
/// variables.
#[derive(Getters)]
pub struct Mle<F> {
    #[getset(get = "pub")]
    coeffs: Vec<F>,
}

impl<F: Field> Mle<F> {
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    /// Create MLE from evaluations on the hypercube.
    ///
    /// Takes evaluations of the polynomial at all points in {0,1}^n and converts
    /// them to coefficient form using a recursive algorithm.
    ///
    /// The input `evals` should have length 2^n where n is the number of variables.
    /// The evaluation at index i corresponds to the point whose binary representation
    /// is i (with bit 0 being the least significant).
    ///
    /// Algorithm: Recursively applies the transformation matrix [[1, 0], [-1, 1]]
    /// to convert from evaluation form to coefficient form.
    //
    // TODO[jpw]: Better implementation with wavelet transform: <https://github.com/tcoratger/whir-p3/blob/main/src/poly/wavelet.rs#L72>
    pub fn from_evaluations(evals: &[F]) -> Self {
        assert!(!evals.is_empty(), "Evaluations cannot be empty");
        assert!(
            evals.len().is_power_of_two(),
            "Number of evaluations must be a power of 2"
        );

        let coeffs = Self::evals_to_coeffs_recursive(evals);
        Self { coeffs }
    }

    pub fn into_coeffs(self) -> Vec<F> {
        self.coeffs
    }

    pub fn eval_at_point<F2: Field, EF: ExtensionField<F> + Mul<F2, Output = EF>>(
        &self,
        x: &[F2],
    ) -> EF {
        debug_assert_eq!(log2_strict_usize(self.coeffs.len()), x.len());
        let mut res = EF::ZERO;
        for (i, coeff) in self.coeffs.iter().enumerate() {
            let mut term = EF::from(*coeff);
            for (j, x_j) in x.iter().enumerate() {
                if (i >> j) & 1 == 1 {
                    term = term * *x_j;
                }
            }
            res += term;
        }
        res
    }

    /// Recursive helper to convert evaluations to coefficients.
    fn evals_to_coeffs_recursive(evals: &[F]) -> Vec<F> {
        if evals.len() == 1 {
            return vec![evals[0]];
        }

        let len = evals.len();
        let half = len / 2;

        // Recursively process left and right halves
        let left = Self::evals_to_coeffs_recursive(&evals[..half]);
        let right = Self::evals_to_coeffs_recursive(&evals[half..]);

        // Combine: first half stays the same, second half is difference
        let mut result = Vec::with_capacity(len);
        result.extend_from_slice(&left);

        for i in 0..half {
            result.push(right[i] - left[i]);
        }

        result
    }

    pub fn coeffs_to_evals_inplace(a: &mut [F]) {
        let n = log2_strict_usize(a.len());

        for b in 0..n {
            let step = 1usize << b;
            let span = step << 1;
            for i in (0..a.len()).step_by(span) {
                for j in 0..step {
                    let u = i + j;
                    let v = u + step;
                    a[v] += a[u];
                }
            }
        }
    }
}

/// Given vector `x` in `F^n`, populates `out` with `eq_n(x, y)` for `y` on hypercube `H_n`.
///
/// The multilinear equality polynomial is defined as:
/// ```text
///     eq(x, z) = \prod_{i=0}^{n-1} (x_i z_i + (1 - x_i)(1 - z_i)).
/// ```
// Reference: <https://github.com/Plonky3/Plonky3/blob/main/multilinear-util/src/eq.rs>
pub fn evals_eq_hypercube<F: Field>(x: &[F]) -> Vec<F> {
    let n = x.len();
    let mut out = F::zero_vec(1 << n);
    out[0] = F::ONE;
    for (i, &x_i) in x.iter().enumerate() {
        for y in (0..(1 << i)).rev() {
            // Don't overwrite in y = 0 case
            out[y | (1 << i)] = out[y] * x_i;
            out[y] *= F::ONE - x_i;
        }
    }
    out
}

/// Given vector `(z,x)` in `F^{n+1}`, populates `out` with `eq_{l_skip,n}(x, y)` for `y` on
/// hyperprism `D_n`.
pub fn evals_eq_hyperprism<F: TwoAdicField, EF: ExtensionField<F>>(
    omega_pows: &[F],
    z: EF,
    x: &[EF],
) -> Vec<EF> {
    // Size of D
    let d_size = omega_pows.len();
    let l_skip = log2_strict_usize(d_size);
    let n = x.len();
    let mut out = EF::zero_vec(d_size << n);
    for (omega_pow, eq_uni) in zip(omega_pows, out.iter_mut()) {
        *eq_uni = eval_eq_uni(l_skip, z, EF::from(*omega_pow));
    }
    for (i, &x_i) in x.iter().enumerate() {
        for y in (0..d_size << i).rev() {
            let eq_prev = out[y];
            // Don't overwrite in y = 0 case
            out[y | (d_size << i)] = eq_prev * x_i;
            out[y] = eq_prev * (EF::ONE - x_i);
        }
    }
    out
}

/// Prismalinear extension polynomial, in coefficient form.
///
/// Depends on implicit univariate skip parameter `l_skip`.
/// Length of `coeffs` is `2^{l_skip + n}` where `n` is hypercube dimension.
/// Indexing is to decompose `i = i_0 + 2^{l_skip} * (i_1 + 2 * i_2 .. + 2^{n-1} * i_n)` and let
/// `coeffs[i]` be the coefficient of `Z^{i_0} X_1^{i_1} .. X_n^{i_n}`.
#[derive(Getters)]
pub struct Ple<F> {
    #[getset(get = "pub")]
    pub(crate) coeffs: Vec<F>,
}

impl<F: TwoAdicField> Ple<F> {
    /// Create PLE from evaluations on the hypercube with univariate skip.
    ///
    /// Takes evaluations at 2^{l_skip + n} points and converts them to coefficient form
    /// for a polynomial in n+1 variables: degree < 2^l_skip in the first variable,
    /// degree < 2 (linear) in the other n variables.
    ///
    /// The input `evals` should have length 2^{l_skip + n}.
    /// The evaluation at index i corresponds to:
    /// - bits 0 to l_skip-1: univariate point index
    /// - bits l_skip to l_skip+n-1: multilinear variable assignments
    ///
    /// Algorithm: Recursively applies multilinear transformation, but stops at
    /// size 2^l_skip to apply inverse DFT for the univariate dimension.
    pub fn from_evaluations(l_skip: usize, evals: &[F]) -> Self {
        let prism_dim = log2_strict_usize(evals.len());
        assert!(
            prism_dim >= l_skip,
            "Total variables must be at least l_skip"
        );

        let mut coeffs = vec![F::ZERO; evals.len()];
        Self::evals_to_coeffs_recursive(l_skip, evals, &mut coeffs);
        Self { coeffs }
    }

    pub fn eval_at_point<EF: ExtensionField<F>>(&self, l_skip: usize, z: EF, x: &[EF]) -> EF {
        let n = x.len();
        debug_assert_eq!(l_skip + n, log2_strict_usize(self.coeffs.len()));
        let mut res = EF::ZERO;
        let mut z_pow = EF::ONE;
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if i.trailing_zeros() >= l_skip as u32 {
                z_pow = EF::ONE;
            }
            let i_x = i >> l_skip;
            let mut term = z_pow * *coeff;
            for (j, x_j) in x.iter().enumerate() {
                if (i_x >> j) & 1 == 1 {
                    term *= *x_j;
                }
            }
            z_pow *= z;
            res += term;
        }
        res
    }

    /// Recursive helper to convert evaluations to coefficients.
    fn evals_to_coeffs_recursive(l_skip: usize, evals: &[F], buf: &mut [F]) {
        debug_assert_eq!(buf.len(), evals.len());
        let base_size = 1 << l_skip;

        // Base case: when we reach the univariate dimension, apply inverse DFT
        if evals.len() == base_size {
            let result = UnivariatePoly::from_evals_idft(evals).0;
            buf.copy_from_slice(&result);
        } else {
            // Recursive case: split and apply multilinear transformation
            let len = evals.len();
            let half = len / 2;

            // Recursively process left and right halves
            let (left_buf, right_buf) = buf.split_at_mut(half);
            let (left_evals, right_evals) = evals.split_at(half);

            join(
                || Self::evals_to_coeffs_recursive(l_skip, left_evals, left_buf),
                || Self::evals_to_coeffs_recursive(l_skip, right_evals, right_buf),
            );

            for (r, l) in right_buf.iter_mut().zip(left_buf.iter()) {
                *r -= *l;
            }
        }
    }
}

pub struct MleMatrix<F> {
    pub columns: Vec<Mle<F>>,
}

impl<F: Field> MleMatrix<F> {
    pub fn from_evaluations(evals: &ColMajorMatrix<F>) -> Self {
        let width = evals.width();
        let columns = (0..width)
            .into_par_iter()
            .map(|j| Mle::from_evaluations(evals.column(j)))
            .collect();
        Self { columns }
    }
}

pub struct PleMatrix<F> {
    pub columns: Vec<Ple<F>>,
}

impl<F: TwoAdicField> PleMatrix<F> {
    pub fn from_evaluations(l_skip: usize, evals: &ColMajorMatrixView<F>) -> Self {
        let width = evals.width();
        let columns = (0..width)
            .into_par_iter()
            .map(|j| Ple::from_evaluations(l_skip, evals.column(j)))
            .collect();
        Self { columns }
    }
}
