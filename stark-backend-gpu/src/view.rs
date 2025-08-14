use openvm_stark_backend::keygen::types::{LinearConstraint, StarkVerifyingKey};

#[derive(Clone, derive_new::new)]
#[allow(dead_code)]
pub(crate) struct MultiStarkVerifyingKeyView<'a, Val, Com> {
    pub per_air: Vec<&'a StarkVerifyingKey<Val, Com>>,
    /// Trace height constraints are *not* filtered by AIR. When computing the dot product, this
    /// will be indexed into by air_id.
    pub trace_height_constraints: &'a [LinearConstraint],
    pub pre_hash: Com,
}

impl<Val, Com: Clone> MultiStarkVerifyingKeyView<'_, Val, Com> {
    /// Returns the preprocessed commit of each AIR. If the AIR does not have a preprocessed trace,
    /// returns None.
    #[allow(dead_code)]
    pub fn preprocessed_commits(&self) -> Vec<Option<Com>> {
        self.per_air
            .iter()
            .map(|vk| {
                vk.preprocessed_data
                    .as_ref()
                    .map(|data| data.commit.clone())
            })
            .collect()
    }

    /// Returns all non-empty preprocessed commits.
    #[allow(dead_code)]
    pub fn flattened_preprocessed_commits(&self) -> Vec<Com> {
        self.preprocessed_commits().into_iter().flatten().collect()
    }

    pub fn num_phases(&self) -> usize {
        self.per_air
            .iter()
            .map(|vk| {
                // Consistency check
                let num = vk.params.width.after_challenge.len();
                assert_eq!(num, vk.params.num_challenges_to_sample.len());
                assert_eq!(num, vk.params.num_exposed_values_after_challenge.len());
                num
            })
            .max()
            .unwrap_or(0)
    }

    #[allow(dead_code)]
    pub fn num_challenges_per_phase(&self) -> Vec<usize> {
        let num_phases = self.num_phases();
        (0..num_phases)
            .map(|phase_idx| self.num_challenges_in_phase(phase_idx))
            .collect()
    }

    pub fn num_challenges_in_phase(&self, phase_idx: usize) -> usize {
        self.per_air
            .iter()
            .flat_map(|vk| vk.params.num_challenges_to_sample.get(phase_idx))
            .copied()
            .max()
            .unwrap_or_else(|| panic!("No challenges used in challenge phase {phase_idx}"))
    }
}
