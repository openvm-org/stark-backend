use std::ops::{Add, Mul, Neg, Sub};

use derivative::Derivative;
use p3_field::FieldAlgebra;

use crate::{
    air_builders::symbolic::{
        dag::SymbolicExpressionDag,
        symbolic_expression::SymbolicEvaluator,
        symbolic_variable::{Entry, SymbolicVariable},
    },
    config::{PackedChallenge, PackedVal, StarkGenericConfig, Val},
};

type ViewPair<'a, T> = [Vec<T>; 2];

/// A struct for quotient polynomial evaluation. This evaluates `WIDTH` rows of the quotient polynomial
/// simultaneously using SIMD (if target arch allows it) via `PackedVal` and `PackedChallenge` types.
pub struct ProverConstraintEvaluator<'a, SC: StarkGenericConfig> {
    pub preprocessed: ViewPair<'a, PackedVal<SC>>,
    pub partitioned_main: Vec<ViewPair<'a, PackedVal<SC>>>,
    pub after_challenge: Vec<ViewPair<'a, PackedChallenge<SC>>>,
    pub challenges: &'a [Vec<PackedChallenge<SC>>],
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub public_values: &'a [Val<SC>],
    pub exposed_values_after_challenge: &'a [&'a [PackedChallenge<SC>]],
}

/// In order to avoid extension field arithmetic as much as possible, we evaluate into
/// the smallest packed expression possible.
#[derive(Derivative, Copy)]
#[derivative(Clone(bound = ""))]
enum PackedExpr<SC: StarkGenericConfig> {
    Val(PackedVal<SC>),
    Challenge(PackedChallenge<SC>),
}

impl<SC: StarkGenericConfig> Add for PackedExpr<SC> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self, other) {
            (PackedExpr::Val(x), PackedExpr::Val(y)) => PackedExpr::Val(x + y),
            (PackedExpr::Val(x), PackedExpr::Challenge(y)) => PackedExpr::Challenge(y + x),
            (PackedExpr::Challenge(x), PackedExpr::Val(y)) => PackedExpr::Challenge(x + y),
            (PackedExpr::Challenge(x), PackedExpr::Challenge(y)) => PackedExpr::Challenge(x + y),
        }
    }
}

impl<SC: StarkGenericConfig> Sub for PackedExpr<SC> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (PackedExpr::Val(x), PackedExpr::Val(y)) => PackedExpr::Val(x - y),
            (PackedExpr::Val(x), PackedExpr::Challenge(y)) => {
                let x: PackedChallenge<SC> = x.into();
                // We could alternative do (-y) + x
                PackedExpr::Challenge(x - y)
            }
            (PackedExpr::Challenge(x), PackedExpr::Val(y)) => PackedExpr::Challenge(x - y),
            (PackedExpr::Challenge(x), PackedExpr::Challenge(y)) => PackedExpr::Challenge(x - y),
        }
    }
}

impl<SC: StarkGenericConfig> Mul for PackedExpr<SC> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (PackedExpr::Val(x), PackedExpr::Val(y)) => PackedExpr::Val(x * y),
            (PackedExpr::Val(x), PackedExpr::Challenge(y)) => PackedExpr::Challenge(y * x),
            (PackedExpr::Challenge(x), PackedExpr::Val(y)) => PackedExpr::Challenge(x * y),
            (PackedExpr::Challenge(x), PackedExpr::Challenge(y)) => PackedExpr::Challenge(x * y),
        }
    }
}

impl<SC: StarkGenericConfig> Neg for PackedExpr<SC> {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            PackedExpr::Val(x) => PackedExpr::Val(-x),
            PackedExpr::Challenge(x) => PackedExpr::Challenge(-x),
        }
    }
}

impl<SC> SymbolicEvaluator<Val<SC>, PackedExpr<SC>> for ProverConstraintEvaluator<'_, SC>
where
    SC: StarkGenericConfig,
{
    fn eval_const(&self, c: Val<SC>) -> PackedExpr<SC> {
        PackedExpr::Val(c.into())
    }
    fn eval_is_first_row(&self) -> PackedExpr<SC> {
        PackedExpr::Val(self.is_first_row)
    }
    fn eval_is_last_row(&self) -> PackedExpr<SC> {
        PackedExpr::Val(self.is_last_row)
    }
    fn eval_is_transition(&self) -> PackedExpr<SC> {
        PackedExpr::Val(self.is_transition)
    }

    fn eval_var(&self, symbolic_var: SymbolicVariable<Val<SC>>) -> PackedExpr<SC> {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => PackedExpr::Val(self.preprocessed[offset][index]),
            Entry::Main { part_index, offset } => {
                PackedExpr::Val(self.partitioned_main[part_index][offset][index])
            }
            Entry::Public => PackedExpr::Val(self.public_values[index].into()),
            Entry::Permutation { offset } => {
                let perm = self
                    .after_challenge
                    .first()
                    .expect("Challenge phase not supported");
                PackedExpr::Challenge(perm[offset][index])
            }
            Entry::Challenge => {
                let permutation_randomness = self
                    .challenges
                    .first()
                    .map(|c| c.as_slice())
                    .expect("Challenge phase not supported");
                PackedExpr::Challenge(permutation_randomness[index])
            }
            Entry::Exposed => {
                let permutation_exposed_values = self
                    .exposed_values_after_challenge
                    .first()
                    .expect("Challenge phase not supported");
                PackedExpr::Challenge(permutation_exposed_values[index])
            }
        }
    }
}

impl<SC: StarkGenericConfig> ProverConstraintEvaluator<'_, SC> {
    /// `alpha_powers` are in **reversed** order, with highest power coming first.
    // Note: this could be split into multiple functions if additional constraints need to be folded in
    pub fn accumulate(
        &self,
        constraints: &SymbolicExpressionDag<Val<SC>>,
        alpha_powers: &[PackedChallenge<SC>],
    ) -> PackedChallenge<SC> {
        let evaluated_nodes = self.eval_nodes(&constraints.nodes);
        let mut accumulator = PackedChallenge::<SC>::ZERO;
        for (&alpha_pow, &node_idx) in alpha_powers.iter().zip(&constraints.constraint_idx) {
            match evaluated_nodes[node_idx] {
                PackedExpr::Val(x) => accumulator += alpha_pow * x,
                PackedExpr::Challenge(x) => accumulator += alpha_pow * x,
            }
        }
        accumulator
    }
}