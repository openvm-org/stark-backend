// Originally copied from uni-stark/src/symbolic_builder.rs to allow A: ?Sized

use std::iter;

use itertools::Itertools;
use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAirWithPublicValues, ExtensionBuilder,
    PairBuilder,
};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use self::{
    symbolic_expression::SymbolicExpression,
    symbolic_variable::{Entry, SymbolicVariable},
};
use super::PartitionedAirBuilder;
use crate::{
    interaction::{Interaction, InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
};

mod dag;
pub mod statistics;
pub mod symbolic_expression;
pub mod symbolic_variable;

pub use dag::*;

use crate::interaction::BusIndex;

/// Symbolic constraints for a single AIR with interactions.
#[derive(Clone, Debug)]
pub struct SymbolicConstraints<F> {
    /// All plain AIR constraints. These do **not** include interaction constraints that are proven
    /// via LogUp-GKR.
    pub constraints: Vec<SymbolicExpression<F>>,
    /// Symbolic representation of interactions. These are converted into a LogUp fractional sum
    /// which must be proven using GKR.
    pub interactions: Vec<SymbolicInteraction<F>>,
}

impl<F: Field> SymbolicConstraints<F> {
    pub fn max_constraint_degree(&self) -> usize {
        iter::empty()
            .chain(&self.constraints)
            .chain(
                self.interactions
                    .iter()
                    .flat_map(|i| iter::once(&i.count).chain(&i.message)),
            )
            .map(|expr| expr.degree_multiple())
            .max()
            .unwrap_or(0)
    }

    /// Returns the maximum field degree and count degree across all interactions
    pub fn max_interaction_degrees(&self) -> (usize, usize) {
        let max_field_degree = self
            .interactions
            .iter()
            .map(|interaction| {
                interaction
                    .message
                    .iter()
                    .map(|field| field.degree_multiple())
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);

        let max_count_degree = self
            .interactions
            .iter()
            .map(|interaction| interaction.count.degree_multiple())
            .max()
            .unwrap_or(0);

        (max_field_degree, max_count_degree)
    }
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_builder<F, R>(rap: &R, width: &TraceWidth) -> SymbolicRapBuilder<F>
where
    F: Field,
    R: Air<SymbolicRapBuilder<F>> + BaseAirWithPublicValues<F> + ?Sized,
{
    let mut builder = SymbolicRapBuilder::new(width, rap.num_public_values());
    Air::eval(rap, &mut builder);
    builder
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicRapBuilder<F> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    partitioned_main: Vec<RowMajorMatrix<SymbolicVariable<F>>>,
    public_values: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
    interactions: Vec<SymbolicInteraction<F>>,
    trace_width: TraceWidth,
}

impl<F: Field> SymbolicRapBuilder<F> {
    pub(crate) fn new(width: &TraceWidth, num_public_values: usize) -> Self {
        let preprocessed_width = width.preprocessed.unwrap_or(0);
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width.preprocessed.unwrap_or(0))
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let preprocessed = RowMajorMatrix::new(prep_values, preprocessed_width);

        let mut partitioned_main: Vec<_> = width
            .cached_mains
            .iter()
            .enumerate()
            .map(|(part_index, &width)| gen_main_trace(part_index, width))
            .collect();
        if width.common_main != 0 {
            partitioned_main.push(gen_main_trace(width.cached_mains.len(), width.common_main));
        }

        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();

        Self {
            preprocessed,
            partitioned_main,
            public_values,
            constraints: vec![],
            interactions: vec![],
            trace_width: width.clone(),
        }
    }

    pub fn constraints(self) -> SymbolicConstraints<F> {
        SymbolicConstraints {
            constraints: self.constraints,
            interactions: self.interactions,
        }
    }

    pub fn num_public_values(&self) -> usize {
        self.public_values.len()
    }

    pub fn width(&self) -> TraceWidth {
        self.trace_width.clone()
    }
}

impl<F: Field> AirBuilder for SymbolicRapBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<Self::F>;
    type Var = SymbolicVariable<Self::F>;
    type M = RowMajorMatrix<Self::Var>;

    /// It is difficult to horizontally concatenate matrices when the main trace is partitioned, so
    /// we disable this method in that case.
    fn main(&self) -> Self::M {
        if self.partitioned_main.len() == 1 {
            self.partitioned_main[0].clone()
        } else {
            panic!("Main trace is either empty or partitioned. This function should not be used.")
        }
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }
}

impl<F: Field> PairBuilder for SymbolicRapBuilder<F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field> ExtensionBuilder for SymbolicRapBuilder<F> {
    type EF = F;
    type ExprEF = SymbolicExpression<F>;
    type VarEF = SymbolicVariable<F>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.constraints.push(x.into());
    }
}

impl<F: Field> AirBuilderWithPublicValues for SymbolicRapBuilder<F> {
    type PublicVar = SymbolicVariable<F>;

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field> InteractionBuilder for SymbolicRapBuilder<F> {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: BusIndex,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        count_weight: u32,
    ) {
        let fields = fields.into_iter().map(|f| f.into()).collect();
        let count = count.into();
        self.interactions.push(Interaction {
            bus_index,
            message: fields,
            count,
            count_weight,
        });
    }

    fn num_interactions(&self) -> usize {
        self.interactions.len()
    }

    fn all_interactions(&self) -> &[Interaction<Self::Expr>] {
        &self.interactions
    }
}

impl<F: Field> PartitionedAirBuilder for SymbolicRapBuilder<F> {
    fn cached_mains(&self) -> &[Self::M] {
        &self.partitioned_main[..self.trace_width.cached_mains.len()]
    }
    fn common_main(&self) -> &Self::M {
        assert_ne!(
            self.trace_width.common_main, 0,
            "AIR doesn't have a common main trace"
        );
        &self.partitioned_main[self.trace_width.cached_mains.len()]
    }
}

#[allow(dead_code)]
struct LocalOnlyChecker;

#[allow(dead_code)]
impl LocalOnlyChecker {
    fn check_var<F: Field>(var: SymbolicVariable<F>) -> bool {
        match var.entry {
            Entry::Preprocessed { offset } => offset == 0,
            Entry::Main { offset, .. } => offset == 0,
            Entry::Public => true,
            Entry::Challenge => true,
        }
    }

    fn check_expr<F: Field>(expr: &SymbolicExpression<F>) -> bool {
        match expr {
            SymbolicExpression::Variable(var) => Self::check_var(*var),
            SymbolicExpression::IsFirstRow => false,
            SymbolicExpression::IsLastRow => false,
            SymbolicExpression::IsTransition => false,
            SymbolicExpression::Constant(_) => true,
            SymbolicExpression::Add { x, y, .. } => Self::check_expr(x) && Self::check_expr(y),
            SymbolicExpression::Sub { x, y, .. } => Self::check_expr(x) && Self::check_expr(y),
            SymbolicExpression::Neg { x, .. } => Self::check_expr(x),
            SymbolicExpression::Mul { x, y, .. } => Self::check_expr(x) && Self::check_expr(y),
        }
    }
}

fn gen_main_trace<F: Field>(
    part_index: usize,
    width: usize,
) -> RowMajorMatrix<SymbolicVariable<F>> {
    let mat_values = [0, 1]
        .into_iter()
        .flat_map(|offset| {
            (0..width)
                .map(move |index| SymbolicVariable::new(Entry::Main { part_index, offset }, index))
        })
        .collect_vec();
    RowMajorMatrix::new(mat_values, width)
}
