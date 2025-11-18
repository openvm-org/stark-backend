//! An AIR with specified interactions can be augmented into a RAP.
//! This module auto-converts any [Air] implemented on an [InteractionBuilder] into a [Rap].

use p3_air::Air;

use super::InteractionBuilder;
use crate::rap::{PermutationAirBuilderWithExposedValues, Rap};

impl<AB, A> Rap<AB> for A
where
    A: Air<AB>,
    AB: InteractionBuilder + PermutationAirBuilderWithExposedValues,
{
    fn eval(&self, builder: &mut AB) {
        // Constraints for the main trace:
        Air::eval(self, builder);
    }
}
