//! An AIR with specified interactions can be augmented into a RAP.
//! This module auto-converts any [Air] implemented on an [InteractionBuilder] into a [Rap].

use p3_air::{Air, ExtensionBuilder};

use super::{InteractionBuilder, RapPhaseSeqKind};
use crate::{
    interaction::{gkr_log_up::eval_gkr_log_up_phase, stark_log_up::eval_stark_log_up_phase},
    rap::{PermutationAirBuilderWithExposedValues, Rap},
};

/// Used internally to select RAP phase evaluation function.
pub(crate) trait InteractionPhaseAirBuilder: ExtensionBuilder {
    fn finalize_interactions(&mut self);
    fn interaction_chunk_size(&self) -> Option<usize>;
    fn rap_phase_seq_kind(&self) -> RapPhaseSeqKind;
}

impl<AB, A> Rap<AB> for A
where
    A: Air<AB>,
    AB: InteractionBuilder + PermutationAirBuilderWithExposedValues + InteractionPhaseAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Constraints for the main trace:
        Air::eval(self, builder);
        builder.finalize_interactions();
        if builder.num_interactions() != 0 {
            match builder.rap_phase_seq_kind() {
                RapPhaseSeqKind::StarkLogUp => {
                    let interaction_chunk_size = builder
                        .interaction_chunk_size()
                        .expect("interaction_chunk_size should be set on StarkLogUp");
                    eval_stark_log_up_phase(builder, interaction_chunk_size);
                }
                RapPhaseSeqKind::GkrLogUp => eval_gkr_log_up_phase(builder),
            }
        }
    }
}
