// Folder: Folding builder
use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::{
    symbolic::{
        symbolic_expression::SymbolicEvaluator,
        symbolic_variable::{Entry, SymbolicVariable},
    },
    PartitionedAirBuilder, ViewPair,
};
use crate::{
    config::{PackedChallenge, PackedVal, StarkGenericConfig, Val},
    interaction::{
        rap::InteractionPhaseAirBuilder, Interaction, InteractionBuilder, InteractionType,
        RapPhaseSeqKind,
    },
    rap::PermutationAirBuilderWithExposedValues,
};

/// A folder for prover constraints.
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    pub preprocessed: ViewPair<'a, PackedVal<SC>>,
    pub partitioned_main: Vec<ViewPair<'a, PackedVal<SC>>>,
    pub after_challenge: Vec<ViewPair<'a, PackedChallenge<SC>>>,
    pub challenges: &'a [Vec<PackedChallenge<SC>>],
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub alpha_powers: &'a [SC::Challenge],
    pub accumulator: PackedChallenge<SC>,
    pub constraint_index: usize,
    pub public_values: &'a [Val<SC>],
    pub exposed_values_after_challenge: &'a [&'a [PackedChallenge<SC>]],
    pub interactions: Vec<Interaction<PackedVal<SC>>>,
    /// Number of interactions to bundle in permutation trace, if applicable.
    pub interaction_chunk_size: usize,
    pub rap_phase_seq_kind: RapPhaseSeqKind,
    pub has_common_main: bool,
}

impl<'a, SC> AirBuilder for ProverConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = ViewPair<'a, PackedVal<SC>>;

    /// It is difficult to horizontally concatenate matrices when the main trace is partitioned, so we disable this method in that case.
    fn main(&self) -> Self::M {
        if self.partitioned_main.len() == 1 {
            self.partitioned_main[0]
        } else {
            panic!("Main trace is either empty or partitioned. This function should not be used.")
        }
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: PackedVal<SC> = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += PackedChallenge::<SC>::from_f(alpha_power) * x;
        self.constraint_index += 1;
    }
}

impl<SC> PairBuilder for ProverConstraintFolder<'_, SC>
where
    SC: StarkGenericConfig,
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<SC> ExtensionBuilder for ProverConstraintFolder<'_, SC>
where
    SC: StarkGenericConfig,
{
    type EF = SC::Challenge;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: PackedChallenge<SC> = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += PackedChallenge::<SC>::from_f(alpha_power) * x;
        self.constraint_index += 1;
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for ProverConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

// PermutationAirBuilder is just a special kind of RAP builder
impl<'a, SC> PermutationAirBuilder for ProverConstraintFolder<'a, SC>
where
    SC: StarkGenericConfig,
{
    type MP = ViewPair<'a, PackedChallenge<SC>>;

    type RandomVar = PackedChallenge<SC>;

    fn permutation(&self) -> Self::MP {
        *self
            .after_challenge
            .first()
            .expect("Challenge phase not supported")
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.challenges
            .first()
            .map(|c| c.as_slice())
            .expect("Challenge phase not supported")
    }
}

impl<SC> PermutationAirBuilderWithExposedValues for ProverConstraintFolder<'_, SC>
where
    SC: StarkGenericConfig,
{
    fn permutation_exposed_values(&self) -> &[Self::VarEF] {
        self.exposed_values_after_challenge
            .first()
            .expect("Challenge phase not supported")
    }
}

impl<SC> PartitionedAirBuilder for ProverConstraintFolder<'_, SC>
where
    SC: StarkGenericConfig,
{
    fn cached_mains(&self) -> &[Self::M] {
        let mut num_cached_mains = self.partitioned_main.len();
        if self.has_common_main {
            num_cached_mains -= 1;
        }
        &self.partitioned_main[..num_cached_mains]
    }
    fn common_main(&self) -> &Self::M {
        assert!(self.has_common_main, "AIR doesn't have a common main trace");
        self.partitioned_main.last().unwrap()
    }
}

impl<SC> InteractionBuilder for ProverConstraintFolder<'_, SC>
where
    SC: StarkGenericConfig,
{
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: usize,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        interaction_type: InteractionType,
    ) {
        let fields = fields.into_iter().map(|f| f.into()).collect();
        let count = count.into();
        self.interactions.push(Interaction {
            bus_index,
            fields,
            count,
            interaction_type,
        });
    }

    fn num_interactions(&self) -> usize {
        self.interactions.len()
    }

    fn all_interactions(&self) -> &[Interaction<Self::Expr>] {
        &self.interactions
    }
}

impl<SC> SymbolicEvaluator<Val<SC>, PackedVal<SC>> for ProverConstraintFolder<'_, SC>
where
    SC: StarkGenericConfig,
{
    fn eval_var(&self, symbolic_var: SymbolicVariable<Val<SC>>) -> PackedVal<SC> {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => self.preprocessed.get(offset, index),
            Entry::Main { part_index, offset } => {
                self.partitioned_main[part_index].get(offset, index)
            }
            Entry::Public => self.public_values[index].into(),
            _ => panic!("After challenge evaluation not allowed"),
        }
    }
}

impl<SC: StarkGenericConfig> InteractionPhaseAirBuilder for ProverConstraintFolder<'_, SC> {
    fn finalize_interactions(&mut self) {}

    fn interaction_chunk_size(&self) -> usize {
        self.interaction_chunk_size
    }

    fn rap_phase_seq_kind(&self) -> RapPhaseSeqKind {
        self.rap_phase_seq_kind
    }
}
