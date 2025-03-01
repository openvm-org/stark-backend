use core::{
    array,
    fmt::{self, Debug, Display, Formatter},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use itertools::Itertools;
use num_bigint::BigUint;
use p3_field::{
    extension::{BinomiallyExtendable, HasFrobenius, HasTwoAdicBinomialExtension},
    field_to_array, ExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra, Packable,
    TwoAdicField,
};
use p3_util::convert_vec;
use rand::{distributions::Standard, prelude::Distribution, Rng};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // to make the zero_vec implementation safe
pub struct FastBinomialExtensionField<FA, const D: usize> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "FA: Serialize", deserialize = "FA: Deserialize<'de>")
    )]
    pub(crate) value: [FA; D],
}

impl<FA: FieldAlgebra, const D: usize> Default for FastBinomialExtensionField<FA, D> {
    fn default() -> Self {
        Self {
            value: array::from_fn(|_| FA::ZERO),
        }
    }
}

impl<FA: FieldAlgebra, const D: usize> From<FA> for FastBinomialExtensionField<FA, D> {
    fn from(x: FA) -> Self {
        Self {
            value: field_to_array::<FA, D>(x),
        }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Packable for FastBinomialExtensionField<F, D> {}

impl<F: BinomiallyExtendable<D>, const D: usize> ExtensionField<F>
    for FastBinomialExtensionField<F, D>
{
    type ExtensionPacking = FastBinomialExtensionField<F::Packing, D>;
}

impl<F: BinomiallyExtendable<D>, const D: usize> HasFrobenius<F>
    for FastBinomialExtensionField<F, D>
{
    /// FrobeniusField automorphisms: x -> x^n, where n is the order of BaseField.
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(n^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= D {
            // x |-> x^(n^D) is the identity, so x^(n^count) ==
            // x^(n^(count % D))
            return self.repeated_frobenius(count % D);
        }
        let arr: &[F] = self.as_base_slice();

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((n-1)/D)
        let mut z0 = F::DTH_ROOT;
        for _ in 1..count {
            z0 *= F::DTH_ROOT;
        }

        let mut res = [F::ZERO; D];
        for (i, z) in z0.powers().take(D).enumerate() {
            res[i] = arr[i] * z;
        }

        Self::from_base_slice(&res)
    }

    /// Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn frobenius_inv(&self) -> Self {
        // Writing 'a' for self, we need to compute a^(r-1):
        // r = n^D-1/n-1 = n^(D-1)+n^(D-2)+...+n
        let mut f = Self::ONE;
        for _ in 1..D {
            f = (f * *self).frobenius();
        }

        // g = a^r is in the base field, so only compute that
        // coefficient rather than the full product.
        let a = self.value;
        let b = f.value;
        let mut g = F::ZERO;
        for i in 1..D {
            g += a[i] * b[D - i];
        }
        g *= F::W;
        g += a[0] * b[0];
        debug_assert_eq!(Self::from(g), *self * f);

        f * g.inverse()
    }
}

impl<FA, const D: usize> FieldAlgebra for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    type F = FastBinomialExtensionField<FA::F, D>;

    const ZERO: Self = Self {
        value: [FA::ZERO; D],
    };

    const ONE: Self = Self {
        value: field_to_array::<FA, D>(FA::ONE),
    };

    const TWO: Self = Self {
        value: field_to_array::<FA, D>(FA::TWO),
    };

    const NEG_ONE: Self = Self {
        value: field_to_array::<FA, D>(FA::NEG_ONE),
    };

    #[inline]
    fn from_f(f: Self::F) -> Self {
        Self {
            value: f.value.map(FA::from_f),
        }
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        FA::from_bool(b).into()
    }

    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        FA::from_canonical_u8(n).into()
    }

    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        FA::from_canonical_u16(n).into()
    }

    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        FA::from_canonical_u32(n).into()
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        FA::from_canonical_u64(n).into()
    }

    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        FA::from_canonical_usize(n).into()
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        FA::from_wrapped_u32(n).into()
    }

    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        FA::from_wrapped_u64(n).into()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        match D {
            2 => {
                let a = self.value.clone();
                let mut res = Self::default();
                res.value[0] = a[0].square() + a[1].square() * FA::from_f(FA::F::W);
                res.value[1] = a[0].clone() * a[1].double();
                res
            }
            3 => {
                let mut res = Self::default();
                cubic_square(&self.value, &mut res.value, FA::F::W);
                res
            }
            _ => <Self as Mul<Self>>::mul(self.clone(), self.clone()),
        }
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { convert_vec(FA::zero_vec(len * D)) }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Field for FastBinomialExtensionField<F, D> {
    type Packing = Self;

    const GENERATOR: Self = Self {
        value: F::EXT_GENERATOR,
    };

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        match D {
            2 => Some(Self::from_base_slice(&qudratic_inv(&self.value, F::W))),
            3 => Some(Self::from_base_slice(&cubic_inv(&self.value, F::W))),
            _ => Some(self.frobenius_inv()),
        }
    }

    fn halve(&self) -> Self {
        Self {
            value: self.value.map(|x| x.halve()),
        }
    }

    fn order() -> BigUint {
        F::order().pow(D as u32)
    }
}

impl<F, const D: usize> Display for FastBinomialExtensionField<F, D>
where
    F: BinomiallyExtendable<D>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            let str = self
                .value
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, x)| match (i, x.is_one()) {
                    (0, _) => format!("{x}"),
                    (1, true) => "X".to_string(),
                    (1, false) => format!("{x} X"),
                    (_, true) => format!("X^{i}"),
                    (_, false) => format!("{x} X^{i}"),
                })
                .join(" + ");
            write!(f, "{}", str)
        }
    }
}

impl<FA, const D: usize> Neg for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.value.map(FA::neg),
        }
    }
}

impl<FA, const D: usize> Add for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r += rhs_val;
        }
        Self { value: res }
    }
}

impl<FA, const D: usize> Add<FA> for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: FA) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<FA, const D: usize> AddAssign for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.value[i] += rhs.value[i].clone();
        }
    }
}

impl<FA, const D: usize> AddAssign<FA> for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    #[inline]
    fn add_assign(&mut self, rhs: FA) {
        self.value[0] += rhs;
    }
}

impl<FA, const D: usize> Sum for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<FA, const D: usize> Sub for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r -= rhs_val;
        }
        Self { value: res }
    }
}

impl<FA, const D: usize> Sub<FA> for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: FA) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl<FA, const D: usize> SubAssign for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<FA, const D: usize> SubAssign<FA> for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: FA) {
        *self = self.clone() - rhs;
    }
}

impl<FA, const D: usize> Mul for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        let w = FA::F::W;
        let w_af = FA::from_f(w);

        match D {
            2 => {
                res.value[0] = a[0].clone() * b[0].clone() + a[1].clone() * w_af * b[1].clone();
                res.value[1] = a[0].clone() * b[1].clone() + a[1].clone() * b[0].clone();
            }
            3 => cubic_mul(&a, &b, &mut res.value, w_af),
            4 => {
                quartic_mul_karatsuba(&a, &b, &mut res.value, w_af);
                // let c0 = a[0].clone() * b[0].clone();
                // let c1 = a[1].clone() * b[0].clone() + a[0].clone() * b[1].clone();
                // let c2 = a[2].clone() * b[0].clone()
                //     + a[1].clone() * b[1].clone()
                //     + a[0].clone() * b[2].clone();
                // res.value[3] = a[3].clone() * b[0].clone()
                //     + a[2].clone() * b[1].clone()
                //     + a[1].clone() * b[2].clone()
                //     + a[0].clone() * b[3].clone();
                // let c4 = a[3].clone() * b[1].clone()
                //     + a[2].clone() * b[2].clone()
                //     + a[1].clone() * b[3].clone();
                // let c5 = a[3].clone() * b[2].clone() + a[2].clone() * b[3].clone();
                // let c6 = a[3].clone() * b[3].clone();
                // for i in 0..4 {
                //     for j in 0..4 {
                //         if i + j >= buf.len() + 4 {
                //             buf.push(a[i].clone() * b[j].clone());
                //         } else if {
                //             buf[i + j] += a[i].clone() * b[j].clone();
                //         }
                //         else {

                //         }
                //     }
                // }
                // #[allow(clippy::needless_range_loop)]
                // for i in 0..3 {
                //     res.value[i] = buf[i].clone() + w_af.clone() * buf[i + 4].clone();
                // }
                // res.value[3] = buf[3].clone();
            }
            _ =>
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..D {
                    for j in 0..D {
                        if i + j >= D {
                            res.value[i + j - D] += a[i].clone() * w_af.clone() * b[j].clone();
                        } else {
                            res.value[i + j] += a[i].clone() * b[j].clone();
                        }
                    }
                }
            }
        }
        res
    }
}

impl<FA, const D: usize> Mul<FA> for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: FA) -> Self {
        Self {
            value: self.value.map(|x| x * rhs.clone()),
        }
    }
}

impl<FA, const D: usize> Product for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F, const D: usize> Div for FastBinomialExtensionField<F, D>
where
    F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F, const D: usize> DivAssign for FastBinomialExtensionField<F, D>
where
    F: BinomiallyExtendable<D>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<FA, const D: usize> MulAssign for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<FA, const D: usize> MulAssign<FA> for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: FA) {
        *self = self.clone() * rhs;
    }
}

impl<FA, const D: usize> FieldExtensionAlgebra<FA> for FastBinomialExtensionField<FA, D>
where
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D>,
{
    const D: usize = D;

    #[inline]
    fn from_base(b: FA) -> Self {
        Self {
            value: field_to_array(b),
        }
    }

    #[inline]
    fn from_base_slice(bs: &[FA]) -> Self {
        Self::from_base_fn(|i| bs[i].clone())
    }

    #[inline]
    fn from_base_fn<F: FnMut(usize) -> FA>(f: F) -> Self {
        Self {
            value: array::from_fn(f),
        }
    }

    #[inline]
    fn from_base_iter<I: Iterator<Item = FA>>(iter: I) -> Self {
        let mut res = Self::default();
        for (i, b) in iter.enumerate() {
            res.value[i] = b;
        }
        res
    }

    #[inline(always)]
    fn as_base_slice(&self) -> &[FA] {
        &self.value
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Distribution<FastBinomialExtensionField<F, D>>
    for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> FastBinomialExtensionField<F, D> {
        let mut res = [F::ZERO; D];
        for r in res.iter_mut() {
            *r = Standard.sample(rng);
        }
        FastBinomialExtensionField::<F, D>::from_base_slice(&res)
    }
}

impl<F: Field + HasTwoAdicBinomialExtension<D>, const D: usize> TwoAdicField
    for FastBinomialExtensionField<F, D>
{
    const TWO_ADICITY: usize = F::EXT_TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        Self {
            value: F::ext_two_adic_generator(bits),
        }
    }
}

///Section 11.3.6b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn qudratic_inv<F: Field>(a: &[F], w: F) -> [F; 2] {
    let scalar = (a[0].square() - w * a[1].square()).inverse();
    [a[0] * scalar, -a[1] * scalar]
}

/// Section 11.3.6b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn cubic_inv<F: Field>(a: &[F], w: F) -> [F; 3] {
    let a0_square = a[0].square();
    let a1_square = a[1].square();
    let a2_w = w * a[2];
    let a0_a1 = a[0] * a[1];

    // scalar = (a0^3+wa1^3+w^2a2^3-3wa0a1a2)^-1
    let scalar = (a0_square * a[0] + w * a[1] * a1_square + a2_w.square() * a[2]
        - (F::ONE + F::TWO) * a2_w * a0_a1)
        .inverse();

    //scalar*[a0^2-wa1a2, wa2^2-a0a1, a1^2-a0a2]
    [
        scalar * (a0_square - a[1] * a2_w),
        scalar * (a2_w * a[2] - a0_a1),
        scalar * (a1_square - a[0] * a[2]),
    ]
}

/// karatsuba multiplication for cubic extension field
#[inline]
fn cubic_mul<FA: FieldAlgebra, const D: usize>(a: &[FA; D], b: &[FA; D], res: &mut [FA; D], w: FA) {
    assert_eq!(D, 3);

    let a0_b0 = a[0].clone() * b[0].clone();
    let a1_b1 = a[1].clone() * b[1].clone();
    let a2_b2 = a[2].clone() * b[2].clone();

    res[0] = a0_b0.clone()
        + ((a[1].clone() + a[2].clone()) * (b[1].clone() + b[2].clone())
            - a1_b1.clone()
            - a2_b2.clone())
            * w.clone();
    res[1] = (a[0].clone() + a[1].clone()) * (b[0].clone() + b[1].clone())
        - a0_b0.clone()
        - a1_b1.clone()
        + a2_b2.clone() * w;
    res[2] = (a[0].clone() + a[2].clone()) * (b[0].clone() + b[2].clone()) - a0_b0 - a2_b2 + a1_b1;
}

/// Section 11.3.6a in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn cubic_square<FA: FieldAlgebra, const D: usize>(a: &[FA; D], res: &mut [FA; D], w: FA::F) {
    assert_eq!(D, 3);

    let w_a2 = a[2].clone() * FA::from_f(w);

    res[0] = a[0].square() + (a[1].clone() * w_a2.clone()).double();
    res[1] = w_a2 * a[2].clone() + (a[0].clone() * a[1].clone()).double();
    res[2] = a[1].square() + (a[0].clone() * a[2].clone()).double();
}

/// karatsuba multiplication for quartic extension field
#[inline]
fn quartic_mul_karatsuba<FA: FieldAlgebra, const D: usize>(
    a: &[FA; D],
    b: &[FA; D],
    res: &mut [FA; D],
    w: FA,
) {
    assert_eq!(D, 4);

    let s1 = a[0].clone() + a[1].clone();
    let s2 = a[0].clone() + a[2].clone();
    let s3 = a[0].clone() + a[1].clone() + a[2].clone() + a[3].clone();

    let t1 = b[0].clone() + b[1].clone();
    let t2 = b[0].clone() + b[2].clone();
    let t3 = b[0].clone() + b[1].clone() + b[2].clone() + b[3].clone();

    let e1 = s1.clone() * t1.clone();
    let e2 = a[0].clone() * b[0].clone();
    let e3 = a[1].clone() * b[1].clone();

    let d1 = (s3.clone() - s1) * (t3.clone() - t1);
    let d2 = a[2].clone() * b[2].clone();
    let d3 = a[3].clone() * b[3].clone();

    let f1 = s3.clone() * t3.clone();
    let f2 = s2.clone() * t2.clone();
    let f3 = (s3 - s2) * (t3 - t2);

    let c0 = e2.clone();
    let c1 = e1 - e2.clone() - e3.clone();
    let c2 = f2.clone() - e2 - d2.clone() + e3.clone();

    let c4 = f3.clone() - e3 - d3.clone() + d2.clone();
    let c5 = d1 - d2 - d3.clone();
    let c6 = d3;

    res[3] = f1 - f2 - f3 - c1.clone() - c5.clone();
    res[0] = c0 + w.clone() * c4;
    res[1] = c1 + w.clone() * c5;
    res[2] = c2 + w * c6;
}

/// normal multiplication for quartic extension field
#[allow(dead_code)]
#[inline]
fn quartic_mul_normal<FA: FieldAlgebra, const D: usize>(
    a: &[FA; D],
    b: &[FA; D],
    res: &mut [FA; D],
    w: FA,
) {
    assert_eq!(D, 4);
    let mut buf = Vec::with_capacity(7);
    for i in 0..4 {
        for j in 0..4 {
            if i + j >= buf.len() + 4 {
                buf.push(a[i].clone() * b[j].clone());
            } else {
                buf[i + j] += a[i].clone() * b[j].clone();
            }
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        res[i] = buf[i].clone() + w.clone() * buf[i + 4].clone();
    }
    res[3] = buf[3].clone();
}
