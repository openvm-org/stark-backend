// Copied from uni-stark/src/symbolic_expression.rs to use Arc instead of Rc.

use core::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use std::sync::Arc;

use p3_field::{AbstractField, Field};
use serde::{Deserialize, Serialize};

use super::symbolic_variable::SymbolicVariable;

/// An expression over `SymbolicVariable`s.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub enum SymbolicExpression<F: Field> {
    Variable(SymbolicVariable<F>),
    IsFirstRow,
    IsLastRow,
    IsTransition,
    Constant(F),
    Add {
        x: Arc<Self>,
        y: Arc<Self>,
        degree_multiple: usize,
    },
    Sub {
        x: Arc<Self>,
        y: Arc<Self>,
        degree_multiple: usize,
    },
    Neg {
        x: Arc<Self>,
        degree_multiple: usize,
    },
    Mul {
        x: Arc<Self>,
        y: Arc<Self>,
        degree_multiple: usize,
    },
}

impl<F: Field> SymbolicExpression<F> {
    /// Returns the multiple of `n` (the trace length) in this expression's degree.
    pub const fn degree_multiple(&self) -> usize {
        match self {
            SymbolicExpression::Variable(v) => v.degree_multiple(),
            SymbolicExpression::IsFirstRow => 1,
            SymbolicExpression::IsLastRow => 1,
            SymbolicExpression::IsTransition => 0,
            SymbolicExpression::Constant(_) => 0,
            SymbolicExpression::Add {
                degree_multiple, ..
            } => *degree_multiple,
            SymbolicExpression::Sub {
                degree_multiple, ..
            } => *degree_multiple,
            SymbolicExpression::Neg {
                degree_multiple, ..
            } => *degree_multiple,
            SymbolicExpression::Mul {
                degree_multiple, ..
            } => *degree_multiple,
        }
    }

    pub fn rotate(&self, offset: usize) -> Self {
        match self {
            SymbolicExpression::Variable(v) => v.rotate(offset).into(),
            SymbolicExpression::IsFirstRow => unreachable!("IsFirstRow should not be rotated"),
            SymbolicExpression::IsLastRow => unreachable!("IsLastRow should not be rotated"),
            SymbolicExpression::IsTransition => unreachable!("IsTransition should not be rotated"),
            SymbolicExpression::Constant(c) => Self::Constant(*c),
            SymbolicExpression::Add {
                x,
                y,
                degree_multiple,
            } => Self::Add {
                x: Arc::new(x.rotate(offset)),
                y: Arc::new(y.rotate(offset)),
                degree_multiple: *degree_multiple,
            },
            SymbolicExpression::Sub {
                x,
                y,
                degree_multiple,
            } => Self::Sub {
                x: Arc::new(x.rotate(offset)),
                y: Arc::new(y.rotate(offset)),
                degree_multiple: *degree_multiple,
            },
            SymbolicExpression::Neg { x, degree_multiple } => Self::Neg {
                x: Arc::new(x.rotate(offset)),
                degree_multiple: *degree_multiple,
            },
            SymbolicExpression::Mul {
                x,
                y,
                degree_multiple,
            } => Self::Mul {
                x: Arc::new(x.rotate(offset)),
                y: Arc::new(y.rotate(offset)),
                degree_multiple: *degree_multiple,
            },
        }
    }

    pub fn next(&self) -> Self {
        self.rotate(1)
    }
}

impl<F: Field> Default for SymbolicExpression<F> {
    fn default() -> Self {
        Self::Constant(F::ZERO)
    }
}

impl<F: Field> From<F> for SymbolicExpression<F> {
    fn from(value: F) -> Self {
        Self::Constant(value)
    }
}

impl<F: Field> AbstractField for SymbolicExpression<F> {
    type F = F;

    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f.into()
    }

    fn from_bool(b: bool) -> Self {
        Self::Constant(F::from_bool(b))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::Constant(F::from_canonical_u8(n))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::Constant(F::from_canonical_u16(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::Constant(F::from_canonical_u32(n))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::Constant(F::from_canonical_u64(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::Constant(F::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::Constant(F::from_wrapped_u32(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::Constant(F::from_wrapped_u64(n))
    }
}

impl<F: Field> Add for SymbolicExpression<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let degree_multiple = self.degree_multiple().max(rhs.degree_multiple());
        Self::Add {
            x: Arc::new(self),
            y: Arc::new(rhs),
            degree_multiple,
        }
    }
}

impl<F: Field> Add<F> for SymbolicExpression<F> {
    type Output = Self;

    fn add(self, rhs: F) -> Self {
        self + Self::from(rhs)
    }
}

impl<F: Field> AddAssign for SymbolicExpression<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F: Field> AddAssign<F> for SymbolicExpression<F> {
    fn add_assign(&mut self, rhs: F) {
        *self += Self::from(rhs);
    }
}

impl<F: Field> Sum for SymbolicExpression<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl<F: Field> Sum<F> for SymbolicExpression<F> {
    fn sum<I: Iterator<Item = F>>(iter: I) -> Self {
        iter.map(|x| Self::from(x)).sum()
    }
}

impl<F: Field> Sub for SymbolicExpression<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let degree_multiple = self.degree_multiple().max(rhs.degree_multiple());
        Self::Sub {
            x: Arc::new(self),
            y: Arc::new(rhs),
            degree_multiple,
        }
    }
}

impl<F: Field> Sub<F> for SymbolicExpression<F> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self {
        self - Self::from(rhs)
    }
}

impl<F: Field> SubAssign for SymbolicExpression<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F: Field> SubAssign<F> for SymbolicExpression<F> {
    fn sub_assign(&mut self, rhs: F) {
        *self -= Self::from(rhs);
    }
}

impl<F: Field> Neg for SymbolicExpression<F> {
    type Output = Self;

    fn neg(self) -> Self {
        let degree_multiple = self.degree_multiple();
        Self::Neg {
            x: Arc::new(self),
            degree_multiple,
        }
    }
}

impl<F: Field> Mul for SymbolicExpression<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        #[allow(clippy::suspicious_arithmetic_impl)]
        let degree_multiple = self.degree_multiple() + rhs.degree_multiple();
        Self::Mul {
            x: Arc::new(self),
            y: Arc::new(rhs),
            degree_multiple,
        }
    }
}

impl<F: Field> Mul<F> for SymbolicExpression<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        self * Self::from(rhs)
    }
}

impl<F: Field> MulAssign for SymbolicExpression<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F: Field> MulAssign<F> for SymbolicExpression<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self *= Self::from(rhs);
    }
}

impl<F: Field> Product for SymbolicExpression<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl<F: Field> Product<F> for SymbolicExpression<F> {
    fn product<I: Iterator<Item = F>>(iter: I) -> Self {
        iter.map(|x| Self::from(x)).product()
    }
}

pub trait SymbolicEvaluator<F: Field, E: AbstractField + From<F>> {
    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> E;

    fn eval_expr(&self, symbolic_expr: &SymbolicExpression<F>) -> E {
        match symbolic_expr {
            SymbolicExpression::Variable(var) => self.eval_var(*var),
            SymbolicExpression::Constant(c) => (*c).into(),
            SymbolicExpression::Add { x, y, .. } => self.eval_expr(x) + self.eval_expr(y),
            SymbolicExpression::Sub { x, y, .. } => self.eval_expr(x) - self.eval_expr(y),
            SymbolicExpression::Neg { x, .. } => -self.eval_expr(x),
            SymbolicExpression::Mul { x, y, .. } => self.eval_expr(x) * self.eval_expr(y),
            _ => unreachable!("Expression cannot be evaluated"),
        }
    }
}
