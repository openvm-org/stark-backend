use std::{
    iter::zip,
    ops::{Add, Mul, Neg, Sub},
};

use derivative::Derivative;
use p3_field::FieldAlgebra;

use crate::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicEvaluator,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicExpressionDag, SymbolicExpressionNode,
    },
    config::{PackedChallenge, PackedVal, StarkGenericConfig, Val},
};

pub(super) struct ViewPair<T> {
    pub(super) local: Vec<T>,
    pub(super) next: Option<Vec<T>>,
}

impl<T> ViewPair<T> {
    pub fn new(local: Vec<T>, next: Option<Vec<T>>) -> Self {
        Self { local, next }
    }

    /// SAFETY: no matrix bounds checks are done.
    pub unsafe fn get(&self, row_offset: usize, column_idx: usize) -> &T {
        match row_offset {
            0 => self.local.get_unchecked(column_idx),
            1 => self
                .next
                .as_ref()
                .unwrap_unchecked()
                .get_unchecked(column_idx),
            _ => panic!("row offset {row_offset} not supported"),
        }
    }
}

/// A struct for quotient polynomial evaluation. This evaluates `WIDTH` rows of the quotient
/// polynomial simultaneously using SIMD (if target arch allows it) via `PackedVal` and
/// `PackedChallenge` types.
pub(super) struct ProverConstraintEvaluator<'a, SC: StarkGenericConfig> {
    pub preprocessed: &'a ViewPair<PackedVal<SC>>,
    pub partitioned_main: &'a [ViewPair<PackedVal<SC>>],
    pub after_challenge: &'a [ViewPair<PackedChallenge<SC>>],
    pub challenges: &'a [Vec<PackedChallenge<SC>>],
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub public_values: &'a [Val<SC>],
    pub exposed_values_after_challenge: &'a [Vec<PackedChallenge<SC>>],
}

/// In order to avoid extension field arithmetic as much as possible, we evaluate into
/// the smallest packed expression possible.
#[derive(Derivative, Copy)]
#[derivative(Clone(bound = ""))]
pub(super) enum PackedExpr<SC: StarkGenericConfig> {
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

    /// SAFETY: we only use this trait implementation when we have already done
    /// a previous scan to ensure all matrix bounds are satisfied,
    /// so no bounds checks are done here.
    fn eval_var(&self, symbolic_var: SymbolicVariable<Val<SC>>) -> PackedExpr<SC> {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => unsafe {
                PackedExpr::Val(*self.preprocessed.get(offset, index))
            },
            Entry::Main { part_index, offset } => unsafe {
                PackedExpr::Val(*self.partitioned_main[part_index].get(offset, index))
            },
            Entry::Public => unsafe {
                PackedExpr::Val((*self.public_values.get_unchecked(index)).into())
            },
            Entry::Permutation { offset } => unsafe {
                let perm = self.after_challenge.get_unchecked(0);
                PackedExpr::Challenge(*perm.get(offset, index))
            },
            Entry::Challenge => unsafe {
                PackedExpr::Challenge(*self.challenges.get_unchecked(0).get_unchecked(index))
            },
            Entry::Exposed => unsafe {
                PackedExpr::Challenge(
                    *self
                        .exposed_values_after_challenge
                        .get_unchecked(0)
                        .get_unchecked(index),
                )
            },
        }
    }
}

impl<SC: StarkGenericConfig> ProverConstraintEvaluator<'_, SC> {
    /// # Safety
    /// - The `nodes` must already be topologically sorted, so they only reference previous nodes.
    /// - `exprs` should have capacity at least `constraints.nodes.len()`.
    unsafe fn eval_nodes_mut(
        &self,
        nodes: &[SymbolicExpressionNode<Val<SC>>],
        exprs: &mut Vec<PackedExpr<SC>>,
    ) where
        PackedExpr<SC>: Clone,
    {
        debug_assert!(exprs.capacity() >= nodes.len());
        // SAFETY: we will set all `exprs` in the loop; this is to make debug assertions happy for
        // `exprs.get_unchecked`.
        unsafe {
            exprs.set_len(nodes.len());
        }
        let mut expr_ptr = exprs.as_mut_ptr();
        for node in nodes.iter() {
            // SAFETY: dereference raw pointer `expr_ptr` because we assume `exprs` has enough
            // capacity.
            *expr_ptr = match *node {
                SymbolicExpressionNode::Variable(var) => self.eval_var(var),
                SymbolicExpressionNode::Constant(c) => self.eval_const(c),
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                } => exprs.get_unchecked(left_idx).clone() + exprs.get_unchecked(right_idx).clone(),
                SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                } => exprs.get_unchecked(left_idx).clone() - exprs.get_unchecked(right_idx).clone(),
                SymbolicExpressionNode::Neg { idx, .. } => -exprs.get_unchecked(idx).clone(),
                SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    ..
                } => exprs.get_unchecked(left_idx).clone() * exprs.get_unchecked(right_idx).clone(),
                SymbolicExpressionNode::IsFirstRow => self.eval_is_first_row(),
                SymbolicExpressionNode::IsLastRow => self.eval_is_last_row(),
                SymbolicExpressionNode::IsTransition => self.eval_is_transition(),
            };
            expr_ptr = expr_ptr.add(1);
        }
    }

    /// `alpha_powers` are in **increasing** order of powers, `alpha^0, alpha^1, ...`
    ///
    /// # Panics
    /// If `alpha_powers.len() < constraints.constraint_idx.len()`.
    ///
    /// # Safety
    /// - The `nodes` must already be topologically sorted, so they only reference previous nodes.
    /// - `exprs` should have capacity at least `constraints.nodes.len()`.
    // Note: this could be split into multiple functions if additional constraints need to be folded
    // in
    pub unsafe fn accumulate(
        &self,
        constraints: &SymbolicExpressionDag<Val<SC>>,
        alpha_powers: &[PackedChallenge<SC>],
        exprs: &mut Vec<PackedExpr<SC>>,
    ) -> PackedChallenge<SC> {
        debug_assert!(alpha_powers.len() >= constraints.constraint_idx.len());
        // We want alpha powers to have highest power first, because of how accumulator "folding"
        // works So this will be alpha^{num_constraints - 1}, ..., alpha^0
        self.eval_nodes_mut(&constraints.nodes, exprs);
        let mut accumulator = PackedChallenge::<SC>::ZERO;
        for (&alpha_pow, &node_idx) in zip(alpha_powers, constraints.constraint_idx.iter().rev()) {
            match *exprs.get_unchecked(node_idx) {
                PackedExpr::Val(x) => accumulator += alpha_pow * x,
                PackedExpr::Challenge(x) => accumulator += alpha_pow * x,
            }
        }
        accumulator
    }
}
