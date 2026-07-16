//! Memory estimates for proving.

use std::{cmp::max, mem::size_of};

use crate::{StarkProtocolConfig, SystemParams};

/// Fixed interaction scratch not proportional to interaction cells.
pub const INTERACTION_MEMORY_OVERHEAD: usize = 2 << 20;

/// Cell counts for a proving memory estimate.
///
/// `main_cells_*` are `Σ(padded_height * width)` in base-field cells, split by whether a trace
/// opens next-row rotations. `interaction_cells` is the metered row-interaction slot count after
/// power-of-two trace padding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProvingMemoryCounts {
    /// Main trace cells for AIRs that open next-row rotations after power-of-two padding.
    pub main_cells_with_rot: usize,
    /// Main trace cells for AIRs without next-row rotations after power-of-two padding.
    pub main_cells_without_rot: usize,
    /// Metered row-interaction slots after power-of-two padding.
    pub interaction_cells: usize,
}

impl ProvingMemoryCounts {
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

/// Estimated memory components, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProvingMemoryEstimate {
    /// Selected peak estimate.
    pub total: usize,
    /// Cached main trace data.
    pub main: usize,
    /// Reed-Solomon code matrix for main traces.
    pub rs_code_matrix: usize,
    /// Batch-constraint main-trace buffers after PCS opening.
    pub batch_constraint: usize,
    /// Fractional-GKR buffers plus fixed GKR-phase overhead.
    pub gkr: usize,
    /// Peak among secondary phases, excluding cached main trace data.
    pub secondary_peak: usize,
}

/// Configuration for proving memory estimates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProvingMemoryConfig {
    /// Size of one base-field element in bytes.
    pub base_field_size: usize,
    /// Degree of the extension field over the base field.
    pub extension_degree: usize,
    /// `-log_2` of the rate for the initial Reed-Solomon code.
    pub log_blowup: usize,
    /// `log_2` of the univariate skip domain.
    pub l_skip: usize,
    /// Maximum constraint degree across AIR and interaction constraints.
    pub max_constraint_degree: usize,
    /// Whether the prover keeps the Reed-Solomon code matrix cached after `stacked_commit`.
    pub cache_rs_code_matrix: bool,
}

impl ProvingMemoryConfig {
    pub fn from_protocol_config<SC: StarkProtocolConfig>(
        config: &SC,
        cache_rs_code_matrix: bool,
    ) -> Self {
        Self::from_params::<SC::F>(config.params(), SC::D_EF, cache_rs_code_matrix)
    }

    fn from_params<F>(
        params: &SystemParams,
        extension_degree: usize,
        cache_rs_code_matrix: bool,
    ) -> Self {
        Self {
            base_field_size: size_of::<F>(),
            extension_degree,
            log_blowup: params.log_blowup,
            l_skip: params.l_skip,
            max_constraint_degree: params.max_constraint_degree,
            cache_rs_code_matrix,
        }
    }

    /// Resident main trace matrices.
    #[inline]
    pub fn main_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * self.base_field_size
    }

    /// Reed-Solomon code matrix for the committed main trace data.
    #[inline]
    pub fn rs_code_matrix_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * (1usize << self.log_blowup) * self.base_field_size
    }

    #[inline]
    pub fn batch_constraint_main_cell_weight(&self) -> f64 {
        default_batch_constraint_main_cell_weight(
            self.extension_degree,
            self.l_skip,
            self.max_constraint_degree,
        )
    }

    /// Batch-constraint main-trace buffers for `main_cells = Σ(padded_height * width)`.
    ///
    /// ```text
    /// mat_eval_bytes = (padded_height / 2^l_skip) * ((1 + need_rot) * width) * sizeof(EF)
    ///                = (1 + need_rot) * main_cells * D_EF / 2^l_skip * sizeof(F)
    ///
    /// interp_bytes = (constraint_degree / 2) * mat_eval_bytes
    /// weight       = (1 + constraint_degree / 2) * D_EF / 2^l_skip
    /// ```
    ///
    /// AIRs with `need_rot = true` open two PCS cells per column.
    #[inline]
    pub fn batch_constraint_memory_bytes_for_rot(
        &self,
        main_cells: usize,
        need_rot: bool,
    ) -> usize {
        let main_cell_weight = self.batch_constraint_main_cell_weight();
        let weight = if need_rot {
            2.0 * main_cell_weight
        } else {
            main_cell_weight
        };
        ceil_weighted_bytes(main_cells, self.base_field_size, weight)
    }

    /// Batch-constraint main-trace buffers across rotation classes.
    #[inline]
    pub fn batch_constraint_memory_bytes(&self, counts: ProvingMemoryCounts) -> usize {
        self.batch_constraint_memory_bytes_for_rot(counts.main_cells_with_rot, true)
            + self.batch_constraint_memory_bytes_for_rot(counts.main_cells_without_rot, false)
    }

    /// Fractional-GKR interaction buffers before fixed interaction-phase overhead.
    #[inline]
    pub fn gkr_memory_bytes_without_overhead(&self, interaction_cells: usize) -> usize {
        ceil_weighted_bytes(
            interaction_cells,
            self.base_field_size,
            default_gkr_cell_weight(self.extension_degree),
        )
    }

    /// Fractional-GKR interaction phase, including fixed non-cell-proportional overhead.
    #[inline]
    pub fn gkr_memory_bytes(&self, interaction_cells: usize) -> usize {
        self.gkr_memory_bytes_without_overhead(interaction_cells) + INTERACTION_MEMORY_OVERHEAD
    }

    /// Convert main trace cells and interaction cells to proving memory bytes.
    ///
    /// ```text
    /// main_cells       = main_cells_with_rot + main_cells_without_rot
    /// main             = main_memory_bytes(main_cells)
    /// rs_code_matrix   = rs_code_matrix_memory_bytes(main_cells)
    /// batch_constraint = batch_constraint_memory_bytes(counts)
    /// gkr              = gkr_memory_bytes(interaction_cells)
    /// ```
    ///
    /// Cached RS code matrix:
    ///
    /// ```text
    /// total = main + rs_code_matrix + max(batch_constraint, gkr)
    /// ```
    ///
    /// Dropped RS code matrix:
    ///
    /// ```text
    /// total = main + max(rs_code_matrix, batch_constraint, gkr)
    /// ```
    #[inline]
    pub fn estimate(&self, counts: ProvingMemoryCounts) -> ProvingMemoryEstimate {
        let main_cells = counts.main_cells();
        let main = self.main_memory_bytes(main_cells);
        let rs_code_matrix = self.rs_code_matrix_memory_bytes(main_cells);
        let batch_constraint = self.batch_constraint_memory_bytes(counts);
        let gkr = self.gkr_memory_bytes(counts.interaction_cells);
        let secondary_peak = if self.cache_rs_code_matrix {
            rs_code_matrix + max(batch_constraint, gkr)
        } else {
            max(rs_code_matrix, max(batch_constraint, gkr))
        };

        ProvingMemoryEstimate {
            total: main + secondary_peak,
            main,
            rs_code_matrix,
            batch_constraint,
            gkr,
            secondary_peak,
        }
    }
}

fn default_batch_constraint_main_cell_weight(
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
fn default_gkr_cell_weight(extension_degree: usize) -> f64 {
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

    fn test_memory_config() -> ProvingMemoryConfig {
        let params = default_test_params_small();
        ProvingMemoryConfig::from_params::<u32>(&params, 4, true)
    }

    #[test]
    fn dropped_rs_code_matrix_is_phase_disjoint() {
        let params = default_test_params_small();
        let config = ProvingMemoryConfig::from_params::<u32>(&params, 4, false);
        let counts = ProvingMemoryCounts::new(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(estimate.main, 30 * 4);
        assert_eq!(estimate.rs_code_matrix, 30 * 2 * 4);
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
        assert_eq!(
            estimate.secondary_peak,
            max(
                estimate.rs_code_matrix,
                max(estimate.batch_constraint, estimate.gkr)
            )
        );
    }

    #[test]
    fn cached_rs_code_matrix_is_additive() {
        let config = test_memory_config();
        let counts = ProvingMemoryCounts::new(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(
            estimate.secondary_peak,
            estimate.rs_code_matrix + max(estimate.batch_constraint, estimate.gkr)
        );
    }
}
