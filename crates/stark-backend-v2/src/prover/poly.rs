use core::ops::Mul;
use std::iter::zip;

use getset::Getters;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_dft::{Radix2Bowers, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{
    poly_common::eval_eq_uni,
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
    /// them to coefficient form.
    ///
    /// The input `evals` should have length 2^n where n is the number of variables.
    /// The evaluation at index i corresponds to the point whose binary representation
    /// is i (with bit 0 being the least significant).
    pub fn from_evaluations(evals: &[F]) -> Self {
        assert!(!evals.is_empty(), "Evaluations cannot be empty");
        let mut coeffs = evals.to_vec();
        Self::evals_to_coeffs_inplace(&mut coeffs);
        Self { coeffs }
    }

    pub fn into_coeffs(self) -> Vec<F> {
        self.coeffs
    }

    /// Evaluate with `O(1)` extra memory via naive algorithm.
    ///
    /// Performs `N*log(N)/2` multiplications when `N = x.len()` is a power of two.
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

    /// Evaluate with `O(1)` extra memory but consuming `self`.
    ///
    /// Performs `N - 1` multiplications for `N = x.len()`.
    pub fn eval_at_point_inplace<F2>(self, x: &[F2]) -> F
    where
        F2: Field,
        F: ExtensionField<F2>,
    {
        let mut buf = self.coeffs;
        debug_assert_eq!(buf.len(), 1 << x.len());
        let mut len = 1usize << x.len();
        // Assumes caller ensured buf[..len] is initialized with the current coefficients.
        for &xj in x.iter().rev() {
            len >>= 1;
            let (left, right) = buf.split_at_mut(len);
            for (li, &ri) in zip(left.iter_mut(), right.iter()) {
                *li += ri * xj;
            }
        }
        buf[0]
    }

    pub fn evals_to_coeffs_inplace(a: &mut [F]) {
        let n = log2_strict_usize(a.len());
        // Go through coordinates X_1, ..., X_n and interpolate each one from s(0), s(1) -> s(0) +
        // (s(1) - s(0)) X_i
        for log_step in 0..n {
            let step = 1usize << log_step;
            let span = step << 1;
            a.par_chunks_exact_mut(span).for_each(|chunk| {
                let (first_half, second_half) = chunk.split_at_mut(step);
                first_half
                    .par_iter()
                    .zip(second_half.par_iter_mut())
                    .for_each(|(u, v)| {
                        *v -= *u;
                    });
            });
        }
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
        let (los, his) = out[..2 << i].split_at_mut(1 << i);
        los.par_iter_mut()
            .zip(his.par_iter_mut())
            .for_each(|(lo, hi)| {
                *hi = *lo * x_i;
                *lo *= F::ONE - x_i;
            })
    }
    out
}

/// Given vector `u_tilde` in `F^n`, populates `out` with `mobius_eq_kernel(u_tilde, b)` for
/// `b` on hypercube `H_n`.
///
/// For boolean `b_i ∈ {0,1}` the per-coordinate kernel is:
/// - `K_i(0) = 1 - 2 * u_tilde_i`
/// - `K_i(1) = u_tilde_i`
///
/// The output ordering matches [`evals_eq_hypercube`]: mask bit `i` corresponds to `u_tilde[i]`.
pub fn evals_mobius_eq_hypercube<F: Field>(u_tilde: &[F]) -> Vec<F> {
    let n = u_tilde.len();
    let mut out = F::zero_vec(1 << n);
    out[0] = F::ONE;
    for (i, &u_i) in u_tilde.iter().enumerate() {
        let w0 = F::ONE - u_i.double();
        let w1 = u_i;
        let (los, his) = out[..2 << i].split_at_mut(1 << i);
        los.par_iter_mut()
            .zip(his.par_iter_mut())
            .for_each(|(lo, hi)| {
                let prev = *lo;
                *hi = prev * w1;
                *lo = prev * w0;
            })
    }
    out
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

/// Given vector `x` in `F^n`, returns a concatenation of `evals_eq_hypercube(x[..n])` for all valid
/// `n` in order. Also, the order of masks is of different endianness.
pub fn evals_eq_hypercubes<'a, F: Field>(n: usize, x: impl IntoIterator<Item = &'a F>) -> Vec<F> {
    let mut out = F::zero_vec((2 << n) - 1);
    out[0] = F::ONE;
    for (i, &x_i) in x.into_iter().enumerate() {
        for y in 0..(1 << i) {
            out[(1 << (i + 1)) - 1 + (2 * y + 1)] = out[(1 << i) - 1 + y] * x_i;
            out[(1 << (i + 1)) - 1 + (2 * y)] = out[(1 << i) - 1 + y] * (F::ONE - x_i);
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
    pub fn from_evaluations(l_skip: usize, evals: &[F]) -> Self {
        let prism_dim = log2_strict_usize(evals.len());
        assert!(
            prism_dim >= l_skip,
            "Total variables must be at least l_skip"
        );
        // Go through coordinates Z, X_1, ..., X_n and interpolate each one
        // For first Z coordinate, we do parallel iDFT on each 2^l_skip sized chunk
        let mut buf: Vec<_> = evals
            .par_chunks_exact(1 << l_skip)
            .flat_map(|chunk| {
                let dft = Radix2Bowers;
                dft.idft(chunk.to_vec())
            })
            .collect();

        let n = prism_dim - l_skip;
        // Go through coordinates X_1, ..., X_n and interpolate each one from s(0), s(1) -> s(0) +
        // (s(1) - s(0)) X_i
        for i in 0..n {
            let step = 1usize << (l_skip + i);
            let span = step << 1;
            buf.par_chunks_exact_mut(span).for_each(|chunk| {
                let (first_half, second_half) = chunk.split_at_mut(step);
                first_half
                    .par_iter()
                    .zip(second_half.par_iter_mut())
                    .for_each(|(u, v)| {
                        *v -= *u;
                    });
            });
        }
        Self { coeffs: buf }
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

    pub fn into_coeffs(self) -> Vec<F> {
        self.coeffs
    }
}

/// Convert evaluations of a prismalinear polynomial on `D × {0,1}^n` into the RS coefficient
/// vector for **eval-to-coeff** encoding.
///
/// - `|D| = 2^l_skip`, and `evals` must be ordered so the lower `l_skip` bits of the index select
///   the point in `D`, and higher bits select the boolean assignment.
/// - The output ordering matches the same convention: `idx = z_mask + (1 << l_skip) * x_mask`.
///
/// This avoids computing full prismalinear monomial coefficients in the boolean variables (which
/// would later be re-zeta-transformed), by:
/// 1) Performing an iDFT in `Z` for each boolean assignment.
/// 2) Applying the subset-zeta transform only over the `Z`-mask bits.
pub fn eval_to_coeff_rs_message<F: TwoAdicField>(l_skip: usize, evals: &[F]) -> Vec<F> {
    assert!(!evals.is_empty(), "Evaluations cannot be empty");
    let prism_dim = log2_strict_usize(evals.len());
    assert!(
        prism_dim >= l_skip,
        "Total variables must be at least l_skip"
    );

    let chunk_len = 1usize << l_skip;
    let mut buf: Vec<_> = evals
        .par_chunks_exact(chunk_len)
        .flat_map(|chunk| {
            let dft = Radix2Bowers;
            dft.idft(chunk.to_vec())
        })
        .collect();

    // For each fixed boolean assignment, convert Z-monomial coefficients into hypercube
    // evaluations over the Z-bit variables.
    buf.par_chunks_exact_mut(chunk_len)
        .for_each(Mle::coeffs_to_evals_inplace);

    buf
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

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::F;

    #[test]
    fn test_evals_mobius_eq_hypercube_matches_naive() {
        let mut rng = StdRng::seed_from_u64(0);

        for n in 0..=10usize {
            let u_tilde = (0..n)
                .map(|_| F::from_u64(rng.random()))
                .collect::<Vec<_>>();
            let evals = evals_mobius_eq_hypercube(&u_tilde);
            assert_eq!(evals.len(), 1 << n);

            for (mask, &eval) in evals.iter().enumerate() {
                let mut expected = F::ONE;
                for (i, &w) in u_tilde.iter().enumerate() {
                    expected *= if (mask >> i) & 1 == 0 {
                        F::ONE - w.double()
                    } else {
                        w
                    };
                }
                assert_eq!(eval, expected);
            }
        }
    }

    #[test]
    fn test_decoder_kernel_identity() {
        let mut rng = StdRng::seed_from_u64(1);

        for m in 0..=10usize {
            let a = (0..(1usize << m))
                .map(|_| F::from_u64(rng.random()))
                .collect::<Vec<_>>();

            // RS coefficients under eval-to-coeff encoding are the hypercube evaluations of f.
            let mut rs_coeffs = a.clone();
            Mle::coeffs_to_evals_inplace(&mut rs_coeffs);

            // WHIR works with the associated MLE HatF whose coefficients are `rs_coeffs`.
            // We need HatF evaluations on the hypercube.
            let mut hatf_evals = rs_coeffs.clone();
            Mle::coeffs_to_evals_inplace(&mut hatf_evals);

            let u_tilde = (0..m)
                .map(|_| F::from_u64(rng.random()))
                .collect::<Vec<_>>();

            let mobius_eq_evals = evals_mobius_eq_hypercube(&u_tilde);
            let lhs = zip(&hatf_evals, &mobius_eq_evals).fold(F::ZERO, |acc, (&f, &g)| acc + f * g);

            // Naive evaluation of f(u_tilde) from its coefficient table.
            let rhs = a.iter().enumerate().fold(F::ZERO, |acc, (mask, &coeff)| {
                let mut term = coeff;
                for (i, &w) in u_tilde.iter().enumerate() {
                    if (mask >> i) & 1 == 1 {
                        term *= w;
                    }
                }
                acc + term
            });

            assert_eq!(lhs, rhs);
        }
    }
}
