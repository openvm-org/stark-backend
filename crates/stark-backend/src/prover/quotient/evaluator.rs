use p3_matrix::Matrix;

use crate::{
    air_builders::{
        symbolic::{
            symbolic_expression::SymbolicEvaluator,
            symbolic_variable::{Entry, SymbolicVariable},
        },
        ViewPair,
    },
    config::{PackedChallenge, PackedVal, StarkGenericConfig, Val},
};

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
enum PackedExpr<SC: StarkGenericConfig> {
    Val(PackedVal<SC>),
    Challenge(PackedChallenge<SC>),
}

// TODO: impl binops

impl<SC> SymbolicEvaluator<Val<SC>, PackedExpr<SC>> for ProverConstraintEvaluator<'_, SC>
where
    SC: StarkGenericConfig,
{
    fn eval_var(&self, symbolic_var: SymbolicVariable<Val<SC>>) -> PackedExpr<SC> {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => PackedExpr::Val(self.preprocessed.get(offset, index)),
            Entry::Main { part_index, offset } => {
                PackedExpr::Val(self.partitioned_main[part_index].get(offset, index))
            }
            Entry::Public => PackedExpr::Val(self.public_values[index].into()),
            Entry::Permutation { offset } => {
                let perm = self
                    .after_challenge
                    .first()
                    .expect("Challenge phase not supported");
                PackedExpr::Challenge(perm.get(offset, index))
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

// fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
//     let x: PackedVal<SC> = x.into();
//     let alpha_power = self.alpha_powers[self.constraint_index];
//     self.accumulator += PackedChallenge::<SC>::from_f(alpha_power) * x;
//     self.constraint_index += 1;
// }

// fn assert_zero_ext<I>(&mut self, x: I)
// where
//     I: Into<Self::ExprEF>,
// {
//     let x: PackedChallenge<SC> = x.into();
//     let alpha_power = self.alpha_powers[self.constraint_index];
//     self.accumulator += PackedChallenge::<SC>::from_f(alpha_power) * x;
//     self.constraint_index += 1;
// }
