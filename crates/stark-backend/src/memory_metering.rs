//! Memory estimates for segmented proving.

use std::{cmp::max, mem::size_of};

use crate::{StarkProtocolConfig, SystemParams};

/// Fixed per-segment interaction scratch not proportional to interaction cells.
const DEFAULT_INTERACTION_MEMORY_OVERHEAD: usize = 2 << 20;

/// Per-segment cell counts derived from trace heights.
///
/// `main_cells_*` are `Σ(padded_height * width)` in base-field cells, split by whether a trace
/// opens next-row rotations. `interaction_cells` is the metered row-interaction slot count after
/// power-of-two trace padding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentMemoryCounts {
    /// Main trace cells for AIRs that open next-row rotations after power-of-two padding.
    pub main_cells_with_rot: usize,
    /// Main trace cells for AIRs without next-row rotations after power-of-two padding.
    pub main_cells_without_rot: usize,
    /// Metered row-interaction slots after power-of-two padding.
    pub interaction_cells: usize,
}

impl SegmentMemoryCounts {
    pub const fn new(
        main_cells_with_rot: usize,
        main_cells_without_rot: usize,
        interaction_cells: usize,
    ) -> Self {
        Self {
            main_cells_with_rot,
            main_cells_without_rot,
            interaction_cells,
        }
    }

    #[inline]
    pub const fn main_cells(&self) -> usize {
        self.main_cells_with_rot + self.main_cells_without_rot
    }
}

/// Estimated memory components for one segment, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentMemoryEstimate {
    /// Selected peak estimate for the segment.
    pub total: usize,
    /// Cached main trace data.
    pub main: usize,
    /// Reed-Solomon code matrix for main traces.
    pub rs_code_matrix: usize,
    /// Secondary main-trace buffers after PCS opening.
    pub main_secondary: usize,
    /// Interaction GKR buffers plus fixed interaction overhead.
    pub interaction: usize,
    /// Peak among secondary phases, excluding cached main trace data.
    pub secondary_peak: usize,
}

/// Configuration for segment memory estimates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentMemoryConfig {
    /// Size of one base-field element in bytes.
    pub base_field_size: usize,
    /// Degree of the extension field over the base field.
    pub extension_degree: usize,
    /// `-log_2` of the rate for the initial Reed-Solomon code.
    pub log_blowup: usize,
    /// Whether the prover keeps the Reed-Solomon code matrix cached after `stacked_commit`.
    pub cache_rs_code_matrix: bool,
    /// Secondary memory contribution per main cell, per PCS opening, in base-field cells.
    pub main_cell_secondary_weight: f64,
    /// Weight multiplier for interaction cells in base-field cells.
    pub interaction_cell_weight: f64,
    /// Interaction memory overhead: eq buffers, M matrix, and misc small buffers.
    /// Added once per segment.
    pub interaction_memory_overhead: usize,
}

impl SegmentMemoryConfig {
    pub fn from_protocol_config<SC: StarkProtocolConfig>(config: &SC) -> Self {
        Self::from_params::<SC::F>(config.params(), SC::D_EF)
    }

    fn from_params<F>(params: &SystemParams, extension_degree: usize) -> Self {
        Self {
            base_field_size: size_of::<F>(),
            extension_degree,
            log_blowup: params.log_blowup,
            cache_rs_code_matrix: true,
            main_cell_secondary_weight: default_main_cell_secondary_weight(
                params,
                extension_degree,
            ),
            interaction_cell_weight: default_interaction_cell_weight(extension_degree),
            interaction_memory_overhead: DEFAULT_INTERACTION_MEMORY_OVERHEAD,
        }
    }

    pub fn with_cache_rs_code_matrix(mut self, cache_rs_code_matrix: bool) -> Self {
        self.cache_rs_code_matrix = cache_rs_code_matrix;
        self
    }

    #[inline]
    pub fn main_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * self.base_field_size
    }

    #[inline]
    pub fn rs_code_matrix_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * (1usize << self.log_blowup) * self.base_field_size
    }

    /// Secondary memory for `main_cells = Σ(padded_height * width)`.
    ///
    /// ```text
    /// mat_eval_bytes = (padded_height / 2^l_skip) * ((1 + need_rot) * width) * sizeof(EF)
    ///                = (1 + need_rot) * main_cells * D_EF / 2^l_skip * sizeof(F)
    ///
    /// interp_bytes      = (constraint_degree / 2) * mat_eval_bytes
    /// secondary_bytes   = (1 + constraint_degree / 2) * mat_eval_bytes
    /// secondary_weight  = (1 + constraint_degree / 2) * D_EF / 2^l_skip
    /// ```
    ///
    /// AIRs with `need_rot = true` open two PCS cells per column.
    #[inline]
    pub fn main_secondary_memory_bytes_for_rot(&self, main_cells: usize, need_rot: bool) -> usize {
        let weight = if need_rot {
            2.0 * self.main_cell_secondary_weight
        } else {
            self.main_cell_secondary_weight
        };
        ceil_weighted_bytes(main_cells, self.base_field_size, weight)
    }

    #[inline]
    pub fn main_secondary_memory_bytes(&self, counts: SegmentMemoryCounts) -> usize {
        self.main_secondary_memory_bytes_for_rot(counts.main_cells_with_rot, true)
            + self.main_secondary_memory_bytes_for_rot(counts.main_cells_without_rot, false)
    }

    #[inline]
    pub fn interaction_memory_bytes_without_overhead(&self, interaction_cells: usize) -> usize {
        ceil_weighted_bytes(
            interaction_cells,
            self.base_field_size,
            self.interaction_cell_weight,
        )
    }

    #[inline]
    pub fn interaction_memory_bytes(&self, interaction_cells: usize) -> usize {
        self.interaction_memory_bytes_without_overhead(interaction_cells)
            + self.interaction_memory_overhead
    }

    /// Convert main trace cells and interaction cells to memory bytes.
    ///
    /// ```text
    /// main_cells       = main_cells_with_rot + main_cells_without_rot
    /// main             = main_cells * sizeof(F)
    /// rs_code_matrix   = main_cells * 2^log_blowup * sizeof(F)
    /// main_secondary   = sizeof(F) *
    ///                    (2 * main_cell_secondary_weight * main_cells_with_rot
    ///                     + main_cell_secondary_weight * main_cells_without_rot)
    /// interaction      = sizeof(F) * interaction_cell_weight * interaction_cells
    ///                    + interaction_memory_overhead
    /// ```
    ///
    /// Cached RS code matrix:
    ///
    /// ```text
    /// total = main + rs_code_matrix + max(main_secondary, interaction)
    /// ```
    ///
    /// Dropped RS code matrix (`cache_rs_code_matrix = false`):
    ///
    /// ```text
    /// total = main + max(rs_code_matrix, main_secondary, interaction)
    /// ```
    #[inline]
    pub fn estimate(&self, counts: SegmentMemoryCounts) -> SegmentMemoryEstimate {
        let main_cells = counts.main_cells();
        let main = self.main_memory_bytes(main_cells);
        let rs_code_matrix = self.rs_code_matrix_memory_bytes(main_cells);
        let main_secondary = self.main_secondary_memory_bytes(counts);
        let interaction = self.interaction_memory_bytes(counts.interaction_cells);
        let secondary_peak = if self.cache_rs_code_matrix {
            rs_code_matrix + max(main_secondary, interaction)
        } else {
            max(rs_code_matrix, max(main_secondary, interaction))
        };

        SegmentMemoryEstimate {
            total: main + secondary_peak,
            main,
            rs_code_matrix,
            main_secondary,
            interaction,
            secondary_peak,
        }
    }
}

fn default_main_cell_secondary_weight(params: &SystemParams, extension_degree: usize) -> f64 {
    default_main_cell_secondary_weight_from_parts(
        extension_degree,
        params.l_skip,
        params.max_constraint_degree,
    )
}

fn default_main_cell_secondary_weight_from_parts(
    extension_degree: usize,
    l_skip: usize,
    max_constraint_degree: usize,
) -> f64 {
    (1.0 + max_constraint_degree as f64 / 2.0) * extension_degree as f64 / (1usize << l_skip) as f64
}

/// ```
/// leaf_weight = 2 * extension_degree
///
/// work_buffer    <= logical_len / 16
/// tmp_block_sums ~= logical_len / 256
/// logical_len    <= 2 * real_len
///
/// interaction_weight = leaf_weight * (1 + 2 * (1 / 16 + 1 / 256))
/// ```
fn default_interaction_cell_weight(extension_degree: usize) -> f64 {
    (2 * extension_degree) as f64 * (1.0 + 2.0 * (1.0 / 16.0 + 1.0 / 256.0))
}

/// `ceil(cell_count * base_field_size * weight)`
fn ceil_weighted_bytes(cell_count: usize, base_field_size: usize, weight: f64) -> usize {
    ((cell_count * base_field_size) as f64 * weight).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::default_test_params_small;

    fn test_memory_config() -> SegmentMemoryConfig {
        let params = default_test_params_small();
        SegmentMemoryConfig::from_params::<u32>(&params, 4)
    }

    #[test]
    fn dropped_rs_code_matrix_is_phase_disjoint() {
        let config = test_memory_config().with_cache_rs_code_matrix(false);
        let counts = SegmentMemoryCounts::new(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(estimate.main, 30 * 4);
        assert_eq!(estimate.rs_code_matrix, 30 * 2 * 4);
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
        assert_eq!(
            estimate.secondary_peak,
            max(
                estimate.rs_code_matrix,
                max(estimate.main_secondary, estimate.interaction)
            )
        );
    }

    #[test]
    fn cached_rs_code_matrix_is_additive() {
        let config = test_memory_config();
        let counts = SegmentMemoryCounts::new(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(
            estimate.secondary_peak,
            estimate.rs_code_matrix + max(estimate.main_secondary, estimate.interaction)
        );
    }
}
