use std::{
    marker::PhantomData,
    ops::{AddAssign, MulAssign},
};

use p3_field::{AbstractField, ExtensionField, Field};
use p3_matrix::Matrix;

use super::{
    symbolic::{
        dag::{build_symbolic_expr_dag, SymbolicExpressionNode},
        symbolic_expression::{SymbolicEvaluator, SymbolicExpression},
        symbolic_variable::{Entry, SymbolicVariable},
    },
    ViewPair,
};
use crate::config::{StarkGenericConfig, Val};

pub type VerifierConstraintFolder<'a, SC> = GenericVerifierConstraintFolder<
    'a,
    Val<SC>,
    <SC as StarkGenericConfig>::Challenge,
    Val<SC>,
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenge,
>;
// Struct definition copied from sp1 under MIT license.
/// A folder for verifier constraints with generic types.
///
/// `Var` is still a challenge type because this is a verifier.
pub struct GenericVerifierConstraintFolder<'a, F, EF, PubVar, Var, Expr> {
    pub preprocessed: ViewPair<'a, Var>,
    pub partitioned_main: Vec<ViewPair<'a, Var>>,
    pub after_challenge: Vec<ViewPair<'a, Var>>,
    pub challenges: &'a [Vec<Var>],
    pub is_first_row: Var,
    pub is_last_row: Var,
    pub is_transition: Var,
    pub alpha: Var,
    pub accumulator: Expr,
    pub public_values: &'a [PubVar],
    pub exposed_values_after_challenge: &'a [Vec<Var>],
    pub _marker: PhantomData<(F, EF)>,
}

impl<F, EF, PubVar, Var, Expr> GenericVerifierConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField + From<F> + MulAssign<Var> + AddAssign<Var> + Send + Sync,
    Var: Into<Expr> + Copy + Send + Sync,
    PubVar: Into<Expr> + Copy + Send + Sync,
{
    pub fn eval_constraints(&mut self, constraints: &[SymbolicExpression<F>]) {
        let dag = build_symbolic_expr_dag(constraints);
        // node_idx -> evaluation
        // We do a simple serial evaluation in topological order.
        // This can be parallelized if necessary.
        let mut exprs: Vec<Expr> = Vec::with_capacity(dag.nodes.len());
        for node in &dag.nodes {
            let expr = match *node {
                SymbolicExpressionNode::Variable(var) => self.eval_var(var),
                SymbolicExpressionNode::Constant(f) => Expr::from(f),
                SymbolicExpressionNode::Add { x, y, .. } => exprs[x].clone() + exprs[y].clone(),
                SymbolicExpressionNode::Sub { x, y, .. } => exprs[x].clone() - exprs[y].clone(),
                SymbolicExpressionNode::Neg { x, .. } => -exprs[x].clone(),
                SymbolicExpressionNode::Mul { x, y, .. } => exprs[x].clone() * exprs[y].clone(),
                SymbolicExpressionNode::IsFirstRow => self.is_first_row.into(),
                SymbolicExpressionNode::IsLastRow => self.is_last_row.into(),
                SymbolicExpressionNode::IsTransition => self.is_transition.into(),
            };
            exprs.push(expr);
        }
        for idx in dag.constraint_idx {
            self.assert_zero(exprs[idx].clone());
        }
    }

    pub fn assert_zero(&mut self, x: impl Into<Expr>) {
        let x = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<F, EF, PubVar, Var, Expr> SymbolicEvaluator<F, Expr>
    for GenericVerifierConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField + From<F> + Send + Sync,
    Var: Into<Expr> + Copy + Send + Sync,
    PubVar: Into<Expr> + Copy + Send + Sync,
{
    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> Expr {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => self.preprocessed.get(offset, index).into(),
            Entry::Main { part_index, offset } => {
                self.partitioned_main[part_index].get(offset, index).into()
            }
            Entry::Public => self.public_values[index].into(),
            Entry::Permutation { offset } => self
                .after_challenge
                .first()
                .expect("Challenge phase not supported")
                .get(offset, index)
                .into(),
            Entry::Challenge => self
                .challenges
                .first()
                .expect("Challenge phase not supported")[index]
                .into(),
            Entry::Exposed => self
                .exposed_values_after_challenge
                .first()
                .expect("Challenge phase not supported")[index]
                .into(),
        }
    }
    // NOTE: do not use the eval_expr function as it can have exponential complexity!
    // Instead use the `SymbolicExpressionDag`
}
