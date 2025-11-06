use std::array::from_fn;

use cfg_if::cfg_if;
use itertools::Itertools;
use openvm_stark_backend::prover::MatrixDimensions;
use p3_dft::{Radix2Bowers, TwoAdicSubgroupDft};
use p3_field::{
    ExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra, TwoAdicField,
    batch_multiplicative_inverse,
};
use p3_interpolation::interpolate_coset;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::{debug, instrument, trace};

use crate::{
    EF,
    poly_common::UnivariatePoly,
    poseidon2::sponge::FiatShamirTranscript,
    prover::{ColMajorMatrix, ColMajorMatrixView, MatrixView, StridedColMajorMatrixView},
};

/// The univariate skip round 0: we want to compute the univariate polynomial `s(Z) = sum_{x \in
/// H_n} \hat{f}(Z, x)`. For this function, assume that `\hat{f}(\vec z) = \hat\eps(\vec z)
/// W(\hat{T}_0(\vec z), .., \hat{T}_{m-1}(\vec z))` for a sequence of `\hat{T}_i` where each
/// `\hat{T}_i` consists of a collection of prismalinear polynomials in `n + 1` variables, with
/// degree `< 2^{l_skip}` in the first variable.
///
/// The `mats` consists of the evaluations of `\hat{T}_i` on the hyperprism `D_n`, where evaluations
/// of each `\hat{T}_i` are in column-major order.
/// For round 0, we also provide a boolean `is_rotation` indicating whether the matrix should be
/// accessed at a cyclic offset of 1 (aka rotation).
///
/// `eps` is a single column vector of evaluations on `D_n`, except valued in extension field.
///
/// Let `W` be degree `d` in each variable. Then `s` is degree `<= d * (2^{l_skip} - 1)`, so it can
/// be interpolated using `d * (2^{l_skip} - 1) + 1` points.
///
/// This function returns `s` in **coefficient** form.
///
/// If `n > 0`, then all `mats` should have the same height equal to `2^{l_skip + n}`.
/// If `n = 0`, then all `mats` should have height `<= 2^{l_skip}` and they will be univariate
/// lifted to height `2^l_skip`.
#[instrument(level = "trace", skip_all)]
pub fn sumcheck_uni_round0_poly<F, EF, FN, const WD: usize>(
    l_skip: usize,
    n: usize,
    d: usize,
    mats: &[(StridedColMajorMatrixView<F>, bool)],
    w: FN,
) -> [UnivariatePoly<EF>; WD]
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    FN: Fn(
            F,         /* Z */
            usize,     /* x_int */
            &[Vec<F>], /* mats eval at (Z, bin(x_int)) */
        ) -> [EF; WD]
        + Sync,
{
    #[cfg(debug_assertions)]
    if n > 0 {
        for (m, _) in mats.iter() {
            assert_eq!(m.height(), 1 << (l_skip + n));
        }
    } else {
        for (m, _) in mats.iter() {
            assert!(
                m.height() <= 1 << l_skip,
                "mat height {} > 2^{l_skip}",
                m.height()
            );
        }
    }
    let s_deg = sumcheck_round0_deg(l_skip, d);
    let log_domain_size = log2_ceil_usize(s_deg + 1);
    let omega = F::two_adic_generator(log_domain_size);
    // Let L = `{ω^0, ω^1, ..., ω^{s_deg}}` be the larger domain we want to evaluate `s` on.

    // Map-Reduce
    // Map: for each x in H_n, compute
    // ```
    // [W(\hat{T}_0(z, x), ..., \hat{T}_{m-1}(z, x)) for z in {ω^0,...,ω^{s_deg}}]
    // ```
    // We choose to iterate over x first to avoid multiple memory accesses to `\hat{T}`s
    let evals = (0..1 << n).into_par_iter().map(|x| {
        let dft = Radix2Bowers;
        // For fixed `x`, `Z -> \hat{T}_i(Z, x)` is a polynomial of degree `<2^l_skip` and we
        // have evaluations on the univariate skip domain `D = <ω_skip>` and we want to get
        // evaluations on the larger domain `L`.
        //
        // For now, we apply iDFT on D and then DFT on L.
        // PERF[jpw]: the most efficient algorithm would be to use Chirp-Z transform on L.
        let mats_at_zs = mats
            .iter()
            .map(|(mat, is_rot)| {
                let height = mat.height();
                let offset = usize::from(*is_rot);
                (0..mat.width())
                    .map(|col_idx| {
                        // SAFETY: col_idx < width
                        // Note that the % height is necessary even when `offset = 0` because we may
                        // have `height < 2^{l_skip + n}` in the case where we are taking the lifts
                        // of `mats`
                        let col_x = ((x << l_skip)..(x + 1) << l_skip)
                            .map(|i| unsafe { *mat.get_unchecked((i + offset) % height, col_idx) })
                            .collect_vec();
                        let mut coeffs = dft.idft(col_x);
                        coeffs.resize(1 << log_domain_size, F::ZERO);
                        dft.dft(coeffs)
                    })
                    .collect_vec()
            })
            .collect_vec();
        // Apply W(..) to `{\hat{T}_i(z, x)}` for each z in L
        // NOTE: `2^log_domain_size` can be larger than `s_deg + 1`, and we want to avoid
        // unnecessary sums over the hypercube.
        omega
            .powers()
            .take(s_deg + 1)
            .enumerate()
            .map(|(z_idx, z)| {
                let row_z_x = mats_at_zs
                    .iter()
                    .map(|mat_at_zs| {
                        mat_at_zs
                            .iter()
                            .map(|col_at_zs| col_at_zs[z_idx])
                            .collect_vec()
                    })
                    .collect_vec();
                w(z, x, &row_z_x)
            })
            .collect_vec()
    });
    // Reduce: sum over H_n
    let hypercube_sum = |mut acc: Vec<[EF; WD]>, x| {
        for (acc, x) in acc.iter_mut().zip(x) {
            for (acc_i, x_i) in acc.iter_mut().zip(x) {
                *acc_i += x_i;
            }
        }
        acc
    };
    cfg_if! {
        if #[cfg(feature = "parallel")] {
            let evals = evals.reduce(
                || vec![[EF::ZERO; WD]; s_deg + 1],
                hypercube_sum
            );
        } else {
            let evals = evals.collect_vec();
            let evals = evals.into_iter().fold(
                vec![[EF::ZERO; WD]; s_deg + 1],
                hypercube_sum
            );
        }
    }
    from_fn(|i| UnivariatePoly::from_evals(&evals.iter().map(|eval| eval[i]).collect_vec()))
}

pub const fn sumcheck_round0_deg(l_skip: usize, d: usize) -> usize {
    d * ((1 << l_skip) - 1)
}

/// `mat` is a matrix of the evaluations on hyperprism D_n of a prismalinear extensions of the
/// columns. We "fold" it by evaluating the prismalinear polynomials at `r` in the univariate
/// variable `Z`.
///
/// If `n < 0`, then we evaluate `mat` at `r^{-n}`, which is equivalent to folding the lift of
/// `mat`.
#[instrument(level = "trace", skip_all)]
pub fn fold_ple_evals<F, EF>(
    l_skip: usize,
    mat: StridedColMajorMatrixView<F>,
    is_rot: bool,
    r: EF,
) -> ColMajorMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    let height = mat.height();
    let lifted_height = height.max(1 << l_skip);
    let width = mat.width();

    let omega = F::two_adic_generator(l_skip);
    let denoms = omega
        .powers()
        .take(1 << l_skip)
        .map(|x_i| r - EF::from(x_i))
        .collect_vec();
    let inv_denoms = batch_multiplicative_inverse(&denoms);

    let offset = usize::from(is_rot);
    let new_height = lifted_height >> l_skip;
    let values = (0..width * new_height)
        .into_par_iter()
        .map(|idx| {
            // `values` needs to be column-major
            let x = idx % new_height;
            let j = idx / new_height;
            // SAFETY: j < width and we mod by height so row_idx < height
            // Note that the `% height` is also necessary to handle lifting of `mats`
            let uni_evals = (0..1 << l_skip)
                .map(|z| unsafe { *mat.get_unchecked(((x << l_skip) + z + offset) % height, j) })
                .collect_vec();
            interpolate_coset(
                &RowMajorMatrix::new_col(uni_evals),
                F::ONE,
                r,
                Some(&inv_denoms),
            )[0]
        })
        .collect::<Vec<_>>();
    ColMajorMatrix::new(values, width)
}

pub fn batch_fold_ple_evals<F, EF>(
    l_skip: usize,
    mats: Vec<ColMajorMatrix<F>>,
    is_rot: bool,
    r: EF,
) -> Vec<ColMajorMatrix<EF>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    mats.into_par_iter()
        .map(|mat| fold_ple_evals(l_skip, mat.as_view().into(), is_rot, r))
        .collect()
}

/// For a sumcheck round, we want to compute the univariate polynomial `s(X) = sum_{y \in H_{n-1}}
/// \hat{f}(X, y)`. For this function, assume that `\hat{f}(\vec x) = W(\hat{T}_0(\vec x), ..,
/// \hat{T}_{m-1}(\vec x))` for a sequence of `\hat{T}_i` where each `\hat{T}_i` consists of a
/// collection of MLE polynomials in `n` variables.
///
/// The `mats` consists of the evaluations of `\hat{T}_i` on the hypercube `H_n`, where evaluations
/// of each `\hat{T}_i` are in column-major order.
///
/// Let `W` be degree `d` in each variable. Then `s` is degree `d`, so it can be interpolated using
/// `d + 1` points. This function returns the evaluations of `s` at `{1, ..., d}`. The evaluation at
/// `0` is omitted because our use of sumcheck always leaves the verifier to infer the evaluation at
/// `0` from the previous round's claim.
///
/// The generic `WF` is a closure `{\hat{T}_i(X, y)}_i -> W(\hat{T}_0(X, y), .., \hat{T}_{m-1}(X,
/// y))`.
///
/// This function should **not** be used for the univariate skip round.
#[instrument(level = "trace", skip_all)]
pub fn sumcheck_round_poly_evals<F, FN, const WD: usize>(
    n: usize,
    d: usize,
    mats: &[ColMajorMatrixView<F>],
    w: FN,
) -> [Vec<F>; WD]
where
    F: Field,
    FN: Fn(
            F,         /* X */
            usize,     /* y_int */
            &[Vec<F>], /* mats eval at (X, bin(y_int)) */
        ) -> [F; WD]
        + Sync,
{
    debug_assert!(mats.iter().all(|mat| mat.height() == 1 << n));
    if n == 0 {
        // Sum is trivial, s(X) is constant
        let evals = mats.iter().map(|row| row.values.to_vec()).collect_vec();
        return w(F::ONE, 0, &evals).map(|x| vec![x; d]);
    }
    let hypercube_dim = n - 1;
    // \hat{f}(x, \vec y) where \vec y is point on hypercube H_{n-1}
    let f_hat = |x: usize, y: usize| {
        let x = F::from_canonical_usize(x);
        let row_x_y = mats
            .iter()
            .map(|mat| {
                mat.columns()
                    .map(|col| {
                        let t_0 = col[y << 1];
                        let t_1 = col[(y << 1) | 1];
                        // Evaluate \hat{t}(x, \vec y) by linear interpolation since
                        // \hat{t} is MLE
                        t_0 + (t_1 - t_0) * x
                    })
                    .collect_vec()
            })
            .collect_vec();
        w(x, y, &row_x_y)
    };
    trace!(sum_claim = ?{(0..1 << n)
        .map(|x| f_hat(x & 1, x >> 1))
        .fold([F::ZERO; WD], |mut acc, x| {
            for (acc_i, x_i) in acc.iter_mut().zip(x) {
                *acc_i += x_i;
            }
            acc
        })
    }, "sumcheck_round");
    // Map-Reduce
    // Map: for each y in H_{n-1}, compute
    // ```
    // [W(\hat{T}_0(x, y), ..., \hat{T}_{m-1}(x, y)) for x in {1,...,d}]
    // ```
    // We choose to iterate over y first to avoid multiple memory accesses to `\hat{T}`s
    let evals = (0..1 << hypercube_dim)
        .into_par_iter()
        .map(|y| (1..=d).map(|x| f_hat(x, y)).collect_vec());
    // Reduce: sum over H_{n-1}
    let hypercube_sum = |mut acc: Vec<[F; WD]>, x| {
        for (acc, x) in acc.iter_mut().zip(x) {
            for (acc_i, x_i) in acc.iter_mut().zip(x) {
                *acc_i += x_i;
            }
        }
        acc
    };
    cfg_if! {
        if #[cfg(feature = "parallel")] {
            let evals = evals.reduce(
                || vec![[F::ZERO; WD]; d],
                hypercube_sum
            );
        } else {
            let evals = evals.collect_vec();
            let evals = evals.into_iter().fold(
                vec![[F::ZERO; WD]; d],
                hypercube_sum
            );
        }
    }
    from_fn(|i| evals.iter().map(|eval| eval[i]).collect_vec())
}

#[instrument(level = "trace", skip_all)]
pub fn fold_mle_evals<EF: Field>(mat: ColMajorMatrix<EF>, r: EF) -> ColMajorMatrix<EF> {
    let height = mat.height();
    if height <= 1 {
        return mat;
    }
    let width = mat.width();
    let values = mat
        .values
        .par_chunks_exact(height)
        .flat_map(|t| {
            t.par_chunks_exact(2).map(|t_01| {
                let t_0 = t_01[0];
                let t_1 = t_01[1];
                t_0 + (t_1 - t_0) * r
            })
        })
        .collect::<Vec<_>>();
    ColMajorMatrix::new(values, width)
}

pub fn batch_fold_mle_evals<EF: Field>(
    mats: Vec<ColMajorMatrix<EF>>,
    r: EF,
) -> Vec<ColMajorMatrix<EF>> {
    mats.into_par_iter()
        .map(|mat| fold_mle_evals(mat, r))
        .collect()
}

/// `mat` is column major evaluations on H_n
pub fn fold_mle_evals_inplace<EF: Field>(mat: &mut ColMajorMatrix<EF>, r: EF) {
    let height = mat.height();
    if height <= 1 {
        return;
    }
    mat.values.par_chunks_exact_mut(height).for_each(|t| {
        for y in 0..height / 2 {
            let t_0 = t[y << 1];
            let t_1 = t[(y << 1) + 1];
            t[y] = t_0 + (t_1 - t_0) * r;
        }
    });
}

pub struct SumcheckCubeProof<EF> {
    /// Note: the sum claim is always observed as an element of the extension field.
    pub sum_claim: EF,
    /// For each `round`, we have univariate polynomial `s_round`. We store evaluations at `{1,
    /// ..., deg(s_round)}` where evaluation at `0` is left for the verifier to infer from the
    /// previous round claim.
    pub round_polys_eval: Vec<Vec<EF>>,
    /// Final evaluation claim of the polynomial at the random vector `r`
    pub eval_claim: EF,
}

pub struct SumcheckPrismProof<EF> {
    pub sum_claim: EF,
    /// The univariate polynomial `s_0` in coefficient form.
    pub s_0: UnivariatePoly<EF>,
    /// for each hypercube `round`, the evaluations of univariate polynomial `s_round` at `{1, ...,
    /// deg(s_round)}`. See [SumcheckCubeProof] for details.
    pub round_polys_eval: Vec<Vec<EF>>,
    /// Final evaluation claim of the polynomial at the random vector `r`
    pub eval_claim: EF,
}

/// "Plain" sumcheck on a multilinear polynomial
///
/// The slice `evals` contains the evaluations of a multilinear polynomial on boolean hypercube.
/// The length of `evals` should equal `2^n` where `n` is hypercube dimension.
///
/// Returns the sumcheck proof containing all prover messages and the random evaluation point.
//
// NOTE[jpw]: we currently fix EF for the transcript, but the evaluations in F can be either base
// field or extension field
pub fn sumcheck_multilinear<F: Field, TS: FiatShamirTranscript>(
    transcript: &mut TS,
    evals: &[F],
) -> (SumcheckCubeProof<EF>, Vec<EF>)
where
    EF: ExtensionField<F>,
{
    let n = log2_strict_usize(evals.len());
    let mut round_polys_eval = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);

    // Working copy of evaluations that gets folded after each round
    // PERF[jpw]: the first round should be treated specially in the case F is the base field
    let mut current_evals =
        ColMajorMatrix::new(evals.iter().map(|&x| EF::from_base(x)).collect(), 1);
    let sum_claim: EF = evals.iter().fold(F::ZERO, |acc, &x| acc + x).into();
    transcript.observe_ext(sum_claim);

    // Sumcheck rounds:
    // - each round the prover needs to compute univariate polynomial `s_round`. This poly is linear
    //   since we are taking MLE of `evals`.
    // - at end of each round, sample random `r_round` in `EF`
    for round in 0..n {
        let [s] =
            sumcheck_round_poly_evals(n - round, 1, &[current_evals.as_view()], |_x, _y, evals| {
                [evals[0][0]]
            });

        println!("CPU s: {:?}", s);
        assert_eq!(s.len(), 1);
        transcript.observe_ext(s[0]);
        round_polys_eval.push(s);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);

        current_evals = fold_mle_evals(current_evals, r_round);
    }

    // After all rounds, current_evals should have exactly one element
    assert_eq!(current_evals.values.len(), 1);
    let eval_claim = current_evals.values[0];

    // Add final evaluation to transcript
    transcript.observe_ext(eval_claim);

    (
        SumcheckCubeProof {
            sum_claim,
            round_polys_eval,
            eval_claim,
        },
        r,
    )
}

/// "Plain" sumcheck on a prismalinear polynomial with Gruen's univariate skip.
///
/// The slice `evals` contains the evaluations of a prismalinear polynomial on the hyperprism.
/// The length of `evals` should equal `2^{l_skip + n}` where `l_skip` is the univariate skip
/// parameter and `n` is hypercube dimension.
/// Indexing is such that `evals[x * 2^{l_skip} + i]` is the evaluation of `f(omega_D^i, x)` where
/// `omega_D` is a fixed generator of the univariate skip domain `D` (which is a subgroup of
/// `F^*`).
///
/// Returns the sumcheck proof containing all prover messages and the random evaluation point.
//
// NOTE[jpw]:
// - we currently fix EF for the transcript, but the evaluations in F can be either base
// field or extension field.
// - for simplicity, the transcript observes `sum_claim` and `s_0` as valued in `EF`. More
//   fine-grained approaches may observe in `F`.
pub fn sumcheck_prismalinear<F, TS: FiatShamirTranscript>(
    transcript: &mut TS,
    l_skip: usize,
    evals: &[F],
) -> (SumcheckPrismProof<EF>, Vec<EF>)
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let prism_dim = log2_strict_usize(evals.len());
    assert!(prism_dim >= l_skip);
    let n = prism_dim - l_skip;

    let mut round_polys_eval = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n + 1);

    let sum_claim: EF = evals.iter().copied().sum::<F>().into();
    transcript.observe_ext(sum_claim);
    let current_evals = ColMajorMatrix::new(evals.to_vec(), 1);
    let [s_0] = sumcheck_uni_round0_poly(
        l_skip,
        n,
        1,
        &[(current_evals.as_view().into(), false)],
        |_z, _x, evals| [evals[0][0]],
    );
    let s_0_ext = UnivariatePoly::new(
        s_0.0
            .into_iter()
            .map(|x| {
                let ext = EF::from(x);
                transcript.observe_ext(ext);
                ext
            })
            .collect(),
    );

    let r_0 = transcript.sample_ext();
    debug!(round = 0, r_round = %r_0);
    r.push(r_0);

    // After sampling r_0, we need to evaluate the prismalinear polynomial at (r_0, x) for each x in
    // hypercube. For each x in the hypercube, we have evaluations f(z, x) for z in the
    // univariate skip domain D. We interpolate these to get a univariate polynomial and evaluate
    // at r_0.
    let mut current_evals = fold_ple_evals(l_skip, current_evals.as_view().into(), false, r_0);
    debug_assert_eq!(current_evals.height(), 1 << n);

    // Sumcheck rounds:
    // - each round the prover needs to compute univariate polynomial `s_round`. This poly is linear
    //   since we are taking MLE of `evals`.
    // - at end of each round, sample random `r_round` in `EF`
    for round in 1..=n {
        debug!(
            cur_sum = %current_evals
                .values
                .iter()
                .fold(EF::ZERO, |acc, x| acc + *x)
        );
        let [s] = sumcheck_round_poly_evals(
            n + 1 - round,
            1,
            &[current_evals.as_view()],
            |_x, _y, evals| [evals[0][0]],
        );
        assert_eq!(s.len(), 1);
        transcript.observe_ext(s[0]);
        round_polys_eval.push(s);

        let r_round = transcript.sample_ext();
        debug!(%round, %r_round);
        r.push(r_round);

        current_evals = fold_mle_evals(current_evals, r_round);
    }

    assert_eq!(r.len(), n + 1);
    // After all rounds, current_evals should have exactly one element
    assert_eq!(current_evals.values.len(), 1);
    let eval_claim = current_evals.values[0];

    // Add final evaluation to transcript
    transcript.observe_ext(eval_claim);

    (
        SumcheckPrismProof {
            sum_claim,
            s_0: s_0_ext,
            round_polys_eval,
            eval_claim,
        },
        r,
    )
}
