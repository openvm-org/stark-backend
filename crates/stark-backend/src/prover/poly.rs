use core::ops::Mul;
use std::iter::zip;

use getset::Getters;
use itertools::Itertools;
use p3_dft::{Radix2Bowers, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::instrument;

use crate::{
    dft::Radix2BowersSerial,
    poly_common::{eval_eq_uni, evals_eq_hypercube_serial, UnivariatePoly},
    prover::{ColMajorMatrix, ColMajorMatrixView, MatrixDimensions},
    utils::batch_multiplicative_inverse_serial,
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

pub fn eq_sharp_uni_poly<EF: TwoAdicField>(xi_1: &[EF]) -> UnivariatePoly<EF> {
    let evals = evals_eq_hypercube_serial(xi_1);
    UnivariatePoly::from_evals_idft(&evals)
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

impl<F: Field> UnivariatePoly<F> {
    #[instrument(level = "debug", skip_all)]
    pub fn lagrange_interpolate<BF: Field>(points: &[BF], evals: &[F]) -> Self
    where
        F: ExtensionField<BF>,
    {
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
        let inv_denominators = batch_multiplicative_inverse_serial(&denominators);

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
}

impl<F: TwoAdicField> UnivariatePoly<F> {
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
        let omega_powers_inv = batch_multiplicative_inverse_serial(&omega_powers);
        let mut p = zip(poly, &omega_powers_inv)
            .map(|(&c, &inv)| c * inv)
            .collect_vec();
        let mut q = omega_powers.iter().rev().copied().collect_vec();

        let dft_deg = (p.len() + q.len() - 1).next_power_of_two();
        p.resize(dft_deg, F::ZERO);
        q.resize(dft_deg, F::ZERO);
        let dft = Radix2BowersSerial;
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
    fn geometric_sequence_linear_product_helper(
        dft: &Radix2BowersSerial,
        z: F,
        n: usize,
    ) -> Vec<F> {
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
        let dft = Radix2BowersSerial;
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

    /// Constructs the polynomial in coefficient form from its evaluations on a smooth subgroup of
    /// `F^*` by performing inverse DFT.
    ///
    /// Requires that `evals.len()` is a power of 2.
    pub fn from_evals_idft(evals: &[F]) -> Self {
        // NOTE[jpw]: Use Bowers instead of Dit to avoid RefCell
        let dft = Radix2BowersSerial;
        let coeffs = dft.idft(evals.to_vec());
        Self(coeffs)
    }

    /// Interpolates from evaluations on cosets `init * g^i D` for `i = 0,..,width-1` where `D` is
    /// a smooth subgroup of F.
    pub fn from_geometric_cosets_evals_idft<BF: Field>(
        evals: RowMajorMatrix<F>,
        shift: BF,
        init: BF,
    ) -> Self
    where
        F: ExtensionField<BF>,
    {
        let height = evals.height();
        let width = evals.width();
        if height == 0 || width == 0 {
            return Self(Vec::new());
        }

        let log_height = log2_strict_usize(height);
        let dft = Radix2BowersSerial;

        // First interpolate within each coset (size `height`) to get the remainder
        // modulo `X^height - shift^height`, then unshift coefficients by `(init * shift^i)^{-t}`.
        let mut coeffs_mat = dft.idft_batch(evals);
        let shift_inv = shift.inverse();
        let init_inv = init.inverse();
        // shift_invs[i] = (init * shift^i)^{-1} = init^{-1} * shift^{-i}
        let shift_invs = (0..width)
            .fold(
                (Vec::with_capacity(width), init_inv),
                |(mut acc, pow), _| {
                    acc.push(pow);
                    (acc, pow * shift_inv)
                },
            )
            .0;
        let mut shift_pows = vec![F::ONE; width];
        for row in coeffs_mat.rows_mut() {
            for (col, value) in row.iter_mut().enumerate() {
                *value *= shift_pows[col];
                shift_pows[col] *= shift_invs[col];
            }
        }

        // Interpolate across cosets for each coefficient degree.
        // Points are init^height, init^height * shift^height, ..., init^height *
        // shift^{(width-1)*height}
        let coset_base = shift.exp_power_of_2(log_height);
        let init_base = init.exp_power_of_2(log_height);
        let lagrange_basis = lagrange_basis_from_geometric_points(coset_base, width, init_base);
        let mut coeffs = vec![F::ZERO; height * width];
        for (row_idx, row_vals) in coeffs_mat.row_slices().enumerate() {
            let mut poly_coeffs = vec![F::ZERO; width];
            for (i, &value) in row_vals.iter().enumerate() {
                if value == F::ZERO {
                    continue;
                }
                for (k, basis_coeff) in lagrange_basis[i].iter().enumerate() {
                    poly_coeffs[k] += value * *basis_coeff;
                }
            }
            for (coset_idx, coeff) in poly_coeffs.into_iter().enumerate() {
                coeffs[coset_idx * height + row_idx] = coeff;
            }
        }
        Self(coeffs)
    }
}

/// Precompute Lagrange basis polynomials for interpolation at `init * base^i` for i=0..width-1.
fn lagrange_basis_from_geometric_points<F: Field>(base: F, width: usize, init: F) -> Vec<Vec<F>> {
    if width == 0 {
        return Vec::new();
    }
    if width == 1 {
        return vec![vec![F::ONE]];
    }

    // Points are init, init * base, init * base^2, ..., init * base^{width-1}
    let points = (0..width)
        .fold((Vec::with_capacity(width), init), |(mut acc, pow), _| {
            acc.push(pow);
            (acc, pow * base)
        })
        .0;

    // Build the monic polynomial P(x) = ∏(x - points[i]).
    let mut root_poly = vec![F::ONE];
    for &x in &points {
        root_poly.push(F::ZERO);
        for k in (1..root_poly.len()).rev() {
            let prev = root_poly[k - 1];
            root_poly[k] = prev - x * root_poly[k];
        }
        root_poly[0] = -x * root_poly[0];
    }

    // Precompute products of (1 - base^k) for k=1..width-1.
    let mut prefix = vec![F::ONE; width];
    for (i, base_pow) in base.powers().skip(1).take(width - 1).enumerate() {
        prefix[i + 1] = prefix[i] * (F::ONE - base_pow);
    }

    // Compute P(x)/(x - points[i]) for each i and scale by the inverse denominator.
    let mut quotients = Vec::with_capacity(width);
    for (i, &x) in points.iter().enumerate() {
        let mut q = vec![F::ZERO; width];
        q[width - 1] = root_poly[width];
        for k in (0..width - 1).rev() {
            q[k] = root_poly[k + 1] + x * q[k + 1];
        }

        // Denominator is prod_{j != i} (points[i] - points[j])
        // = prod_{j != i} (init * base^i - init * base^j)
        // = init^{width-1} * base^{i*(width-1)} * prod_{j != i} (1 - base^{j-i})
        // = init^{width-1} * base^{i*(width-1)} * prod_{k=1}^{i} (1 - base^{-k}) *
        // prod_{k=1}^{width-1-i} (1 - base^k) For k=1..i: (1 - base^{-k}) = -base^{-k} * (1
        // - base^k) So prod_{k=1}^{i} (1 - base^{-k}) = (-1)^i * base^{-i*(i+1)/2} *
        // prefix[i]
        let sign = if i % 2 == 0 { F::ONE } else { F::NEG_ONE };
        let exp = i * (width - 1) - (i * (i + 1) / 2);
        let pow = base.exp_u64(exp as u64);
        let init_pow = init.exp_u64((width - 1) as u64);
        let denom = sign * init_pow * pow * prefix[i] * prefix[width - 1 - i];
        let inv_denom = denom.inverse();
        for coeff in q.iter_mut() {
            *coeff *= inv_denom;
        }
        quotients.push(q);
    }
    quotients
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;
    use openvm_stark_sdk::config::baby_bear_poseidon2::*;
    use p3_field::{PrimeCharacteristicRing, TwoAdicField};
    use p3_util::{log2_ceil_usize, log2_strict_usize};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::poly_common::horner_eval;

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
                original_coeffs.push(F::from_u32(rng.random()));
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
                original_coeffs.push(F::from_u32(rng.random()));
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
    fn test_from_geometric_cosets_evals_idft_round_trip() {
        let mut rng = StdRng::seed_from_u64(0);
        let height = 8usize;
        let log_height = log2_strict_usize(height);
        let omega = F::two_adic_generator(log_height);

        let configs = [
            (F::GENERATOR, F::ONE),
            (F::two_adic_generator(log_height + 2), F::GENERATOR),
        ];

        for (shift, init) in configs {
            for width in 2..=4usize {
                let coeffs = (0..height * width)
                    .map(|_| F::from_u32(rng.random()))
                    .collect_vec();
                let coeffs_ref = &coeffs;

                // Evaluations on cosets init * shift^i * D for i = 0, ..., width - 1
                let evals: Vec<F> = (0..height)
                    .flat_map(|row| {
                        let omega_pow = omega.exp_u64(row as u64);
                        (0..width).map(move |col| {
                            let coset_shift = init * shift.exp_u64(col as u64);
                            horner_eval(coeffs_ref, coset_shift * omega_pow)
                        })
                    })
                    .collect_vec();

                let evals_mat = RowMajorMatrix::new(evals, width);
                let poly = UnivariatePoly::from_geometric_cosets_evals_idft(evals_mat, shift, init);
                assert_eq!(
                    poly.into_coeffs(),
                    coeffs,
                    "width {width} round-trip failed"
                );
            }
        }
    }
}
