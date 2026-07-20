//! Memory estimates for proving.

use std::{cmp::max, mem::size_of};

use crate::{StarkProtocolConfig, SystemParams};

/// Fixed batch-constraint scratch on top of the modeled main-trace buffers.
pub const BATCH_CONSTRAINT_MEMORY_OVERHEAD: usize = 192 << 20;

/// Minimum batch-MLE scratch budget when `zerocheck_save_memory` is off.
pub const BATCH_MLE_MEMORY_FLOOR: usize = 6 << 30;

/// Fixed fractional-GKR scratch not proportional to interaction cells.
pub const GKR_MEMORY_OVERHEAD: usize = 64 << 20;

/// Minimum fractional-GKR work-buffer length in `Frac<EF>` entries.
pub const GKR_MIN_WORK_BUFFER_LEN: usize = 1 << 22;

/// Fixed WHIR opening scratch not proportional to the stacked height.
pub const WHIR_MEMORY_OVERHEAD: usize = 64 << 20;

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
    /// Cached stacked PCS matrix, if retained after commitment.
    pub stacked_matrix: usize,
    /// Reed-Solomon code matrix for main traces.
    pub rs_code_matrix: usize,
    /// Batch-constraint phase peak.
    pub batch_constraint: usize,
    /// Fractional-GKR buffers plus fixed GKR-phase overhead.
    pub gkr: usize,
    /// WHIR opening working set that coexists with the RS code matrix.
    pub whir: usize,
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
    /// Size of one commitment digest in bytes.
    pub digest_size: usize,
    /// `-log_2` of the rate for the initial Reed-Solomon code.
    pub log_blowup: usize,
    /// `log_2` of the univariate skip domain.
    pub l_skip: usize,
    /// `log_2` of the stacked matrix height (`l_skip + n_stack`).
    pub log_stacked_height: usize,
    /// `log_2` of the number of codeword rows per WHIR Merkle-tree leaf.
    pub k_whir: usize,
    /// Maximum constraint degree across AIR and interaction constraints.
    pub max_constraint_degree: usize,
    /// Whether the prover keeps the stacked matrix cached after `stacked_commit`.
    pub cache_stacked_matrix: bool,
    /// Whether the prover keeps the Reed-Solomon code matrix cached after `stacked_commit`.
    pub cache_rs_code_matrix: bool,
    /// Whether the batch-MLE scratch budget is reduced by the resident `mat_eval` buffers.
    pub zerocheck_save_memory: bool,
}

impl ProvingMemoryConfig {
    pub fn from_protocol_config<SC: StarkProtocolConfig>(
        config: &SC,
        cache_stacked_matrix: bool,
        cache_rs_code_matrix: bool,
        zerocheck_save_memory: bool,
    ) -> Self {
        Self::from_params::<SC::F, SC::Digest>(
            config.params(),
            SC::D_EF,
            cache_stacked_matrix,
            cache_rs_code_matrix,
            zerocheck_save_memory,
        )
    }

    fn from_params<F, Digest>(
        params: &SystemParams,
        extension_degree: usize,
        cache_stacked_matrix: bool,
        cache_rs_code_matrix: bool,
        zerocheck_save_memory: bool,
    ) -> Self {
        Self {
            base_field_size: size_of::<F>(),
            extension_degree,
            digest_size: size_of::<Digest>(),
            log_blowup: params.log_blowup,
            l_skip: params.l_skip,
            log_stacked_height: params.log_stacked_height(),
            k_whir: params.k_whir(),
            max_constraint_degree: params.max_constraint_degree,
            cache_stacked_matrix,
            cache_rs_code_matrix,
            zerocheck_save_memory,
        }
    }

    /// Resident main trace matrices.
    #[inline]
    pub fn main_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * self.base_field_size
    }

    /// Cached stacked PCS matrix for the committed main trace data.
    #[inline]
    pub fn stacked_matrix_memory_bytes(&self, main_cells: usize) -> usize {
        if !self.cache_stacked_matrix {
            return 0;
        }
        let stacked_height = 1usize << self.log_stacked_height;
        main_cells.next_multiple_of(stacked_height) * self.base_field_size
    }

    /// Reed-Solomon code matrix for the committed main trace data.
    #[inline]
    pub fn rs_code_matrix_memory_bytes(&self, main_cells: usize) -> usize {
        let stacked_height = 1usize << self.log_stacked_height;
        main_cells.next_multiple_of(stacked_height)
            * (1usize << self.log_blowup)
            * self.base_field_size
    }

    /// Batch-constraint phase peak.
    ///
    /// The main-trace buffer's per-opening weight is
    /// `(1 + constraint_degree / 2) * D_EF / 2^l_skip`; equivalently,
    /// `1 + constraint_degree / 2 = (constraint_degree + 2) / 2`. AIRs with rotations have
    /// `num_openings = 2`; AIRs without rotations have `num_openings = 1`.
    ///
    /// ```text
    /// main = ceil(
    ///   main_cells * num_openings * D_EF * sizeof(F) * (constraint_degree + 2)
    ///   / 2^(l_skip + 1)
    /// )
    /// ```
    ///
    /// Round0 uses `gkr_mem_contribution`: GKR leaves plus the largest strategy
    /// workspace (typically `/4`), without GKR-only `tmp_block_sums`. The GKR
    /// estimate instead uses default precompute-M (`/16` plus `tmp_block_sums`).
    ///
    /// ```text
    /// round0_max_temp_bytes = gkr_mem_contribution(interaction_cells)
    /// working_set (save memory) = max(main, round0_max_temp_bytes)
    /// working_set (no save memory) = main + max(round0_max_temp_bytes, 6 GiB)
    /// batch_constraint = working_set + 192 MiB
    /// ```
    #[inline]
    pub fn batch_constraint_memory_bytes(&self, counts: ProvingMemoryCounts) -> usize {
        let main_bytes = {
            let bytes_per_opening_numerator =
                self.extension_degree * self.base_field_size * (self.max_constraint_degree + 2);
            let denominator = 1usize << (self.l_skip + 1);

            let bytes_for = |main_cells: usize, num_openings: usize| {
                (main_cells * num_openings * bytes_per_opening_numerator).div_ceil(denominator)
            };

            let bytes_with_rot = bytes_for(counts.main_cells_with_rot, 2);
            let bytes_without_rot = bytes_for(counts.main_cells_without_rot, 1);

            bytes_with_rot + bytes_without_rot
        };

        let round0_max_temp_bytes = if counts.interaction_cells == 0 {
            0
        } else {
            let leaf_bytes = 2 * self.extension_degree * self.base_field_size;
            let logical_len = (counts.interaction_cells + 1).next_power_of_two();
            let leaves = counts.interaction_cells * leaf_bytes;
            let work_buffer = max(logical_len / 4, GKR_MIN_WORK_BUFFER_LEN) * leaf_bytes;
            leaves + work_buffer
        };

        let batch_constraint_working_set_bytes = if self.zerocheck_save_memory {
            max(main_bytes, round0_max_temp_bytes)
        } else {
            main_bytes + max(round0_max_temp_bytes, BATCH_MLE_MEMORY_FLOOR)
        };
        batch_constraint_working_set_bytes + BATCH_CONSTRAINT_MEMORY_OVERHEAD
    }

    /// Fractional-GKR phase peak, including fixed overhead.
    ///
    /// ```text
    /// leaf_bytes     = 2 * extension_degree * sizeof(F)
    /// real_len       = interaction_cells
    /// logical_len    = 2^ceil_log2(real_len + 1)
    /// leaves         = real_len * leaf_bytes
    /// work_buffer    = max(logical_len / 16, 2^22) * leaf_bytes
    /// tmp_block_sums = logical_len / 256 * leaf_bytes
    /// gkr             = leaves + work_buffer + tmp_block_sums + 64 MiB
    /// ```
    #[inline]
    pub fn gkr_memory_bytes(&self, interaction_cells: usize) -> usize {
        if interaction_cells == 0 {
            return 0;
        }
        let leaf_bytes = 2 * self.extension_degree * self.base_field_size;
        let logical_len = (interaction_cells + 1).next_power_of_two();
        let leaves = interaction_cells * leaf_bytes;
        let work_buffer = max(logical_len / 16, GKR_MIN_WORK_BUFFER_LEN) * leaf_bytes;
        let tmp_block_sums = logical_len / 256 * leaf_bytes;
        leaves + work_buffer + tmp_block_sums + GKR_MEMORY_OVERHEAD
    }

    /// WHIR opening working set that coexists with the RS code matrix.
    ///
    /// ```text
    /// codeword_height = 2^(log_stacked_height + log_blowup)
    /// commit_tree     = 2 * digest_size * codeword_height / 2^k_whir
    /// g_codeword      = D_EF * sizeof(F) * codeword_height / 2
    /// g_tree          = 2 * digest_size * codeword_height / 2^(k_whir + 1)
    /// ```
    #[inline]
    pub fn whir_memory_bytes(&self) -> usize {
        let codeword_height = 1usize << (self.log_stacked_height + self.log_blowup);
        let commit_tree = 2 * self.digest_size * (codeword_height >> self.k_whir);
        let g_codeword = self.extension_degree * self.base_field_size * (codeword_height >> 1);
        let g_tree = 2 * self.digest_size * (codeword_height >> (self.k_whir + 1));
        commit_tree + g_codeword + g_tree + WHIR_MEMORY_OVERHEAD
    }

    /// Convert main trace cells and interaction cells to proving memory bytes.
    ///
    /// ```text
    /// main_cells       = main_cells_with_rot + main_cells_without_rot
    /// main             = main_memory_bytes(main_cells)
    /// stacked_matrix   = stacked_matrix_memory_bytes(main_cells)
    /// rs_code_matrix   = rs_code_matrix_memory_bytes(main_cells)
    /// batch_constraint = batch-constraint phase peak
    /// gkr              = GKR phase peak
    /// whir             = WHIR working set that coexists with rs_code_matrix
    /// ```
    ///
    /// Cached RS code matrix:
    ///
    /// ```text
    /// total = main + stacked_matrix + rs_code_matrix + max(whir, batch_constraint, gkr)
    /// ```
    ///
    /// Dropped RS code matrix:
    ///
    /// ```text
    /// total = main + stacked_matrix + max(rs_code_matrix + whir, batch_constraint, gkr)
    /// ```
    #[inline]
    pub fn estimate(&self, counts: ProvingMemoryCounts) -> ProvingMemoryEstimate {
        let main_cells = counts.main_cells();
        let main = self.main_memory_bytes(main_cells);
        let stacked_matrix = self.stacked_matrix_memory_bytes(main_cells);
        let rs_code_matrix = self.rs_code_matrix_memory_bytes(main_cells);

        let batch_constraint = self.batch_constraint_memory_bytes(counts);
        let gkr = self.gkr_memory_bytes(counts.interaction_cells);
        let whir = self.whir_memory_bytes();

        let batch_or_gkr = max(batch_constraint, gkr);
        let secondary_peak = if self.cache_rs_code_matrix {
            rs_code_matrix + max(whir, batch_or_gkr)
        } else {
            max(rs_code_matrix + whir, batch_or_gkr)
        };

        ProvingMemoryEstimate {
            total: main + stacked_matrix + secondary_peak,
            main,
            stacked_matrix,
            rs_code_matrix,
            batch_constraint,
            gkr,
            whir,
            secondary_peak,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::default_test_params_small;

    fn test_memory_config() -> ProvingMemoryConfig {
        let params = default_test_params_small();
        ProvingMemoryConfig::from_params::<u32, [u32; 8]>(&params, 4, false, true, true)
    }

    #[test]
    fn dropped_rs_code_matrix_is_phase_disjoint() {
        let params = default_test_params_small();
        let config =
            ProvingMemoryConfig::from_params::<u32, [u32; 8]>(&params, 4, false, false, true);
        let counts = ProvingMemoryCounts::new(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(estimate.main, 30 * 4);
        let stacked_height = 1usize << config.log_stacked_height;
        assert_eq!(
            estimate.rs_code_matrix,
            30usize.next_multiple_of(stacked_height) * 2 * 4
        );
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
        assert_eq!(
            estimate.secondary_peak,
            max(
                estimate.rs_code_matrix + estimate.whir,
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
            estimate.rs_code_matrix
                + max(estimate.whir, max(estimate.batch_constraint, estimate.gkr))
        );
    }

    #[test]
    fn batch_constraint_memory_uses_integer_formula() {
        let config = test_memory_config();
        let counts = ProvingMemoryCounts::new(7, 11, 0);
        let weighted_bytes = |main_cells: usize, need_rot: bool| {
            let weight = (1.0 + config.max_constraint_degree as f64 / 2.0)
                * config.extension_degree as f64
                / (1usize << config.l_skip) as f64;
            let weight = if need_rot { 2.0 * weight } else { weight };
            ((main_cells * config.base_field_size) as f64 * weight).ceil() as usize
        };

        assert_eq!(
            config.batch_constraint_memory_bytes(counts),
            weighted_bytes(counts.main_cells_with_rot, true)
                + weighted_bytes(counts.main_cells_without_rot, false)
                + BATCH_CONSTRAINT_MEMORY_OVERHEAD
        );
    }

    #[test]
    fn no_save_memory_batch_scratch_is_additive() {
        let mut config = test_memory_config();
        let counts = ProvingMemoryCounts::default();

        let saved = config.estimate(counts);
        assert_eq!(saved.batch_constraint, BATCH_CONSTRAINT_MEMORY_OVERHEAD);

        config.zerocheck_save_memory = false;
        let unsaved = config.estimate(counts);
        assert_eq!(
            unsaved.batch_constraint,
            BATCH_MLE_MEMORY_FLOOR + BATCH_CONSTRAINT_MEMORY_OVERHEAD
        );
    }

    #[test]
    fn stacked_matrix_and_whir_components_are_counted_separately() {
        let mut config = test_memory_config();
        let counts = ProvingMemoryCounts::new(10, 20, 5);
        let stacked_height = 1usize << config.log_stacked_height;
        let expected_stacked =
            counts.main_cells().next_multiple_of(stacked_height) * config.base_field_size;

        let without_stacked = config.estimate(counts);
        assert_eq!(config.stacked_matrix_memory_bytes(counts.main_cells()), 0);
        config.cache_stacked_matrix = true;
        let with_stacked = config.estimate(counts);

        assert_eq!(
            config.stacked_matrix_memory_bytes(counts.main_cells()),
            expected_stacked
        );
        assert_eq!(without_stacked.stacked_matrix, 0);
        assert_eq!(with_stacked.stacked_matrix, expected_stacked);
        assert_eq!(with_stacked.total - without_stacked.total, expected_stacked);

        let codeword_height = stacked_height << config.log_blowup;
        let expected_whir = 2 * config.digest_size * (codeword_height >> config.k_whir)
            + config.extension_degree * config.base_field_size * (codeword_height >> 1)
            + 2 * config.digest_size * (codeword_height >> (config.k_whir + 1))
            + WHIR_MEMORY_OVERHEAD;
        assert_eq!(config.whir_memory_bytes(), expected_whir);
        assert_eq!(with_stacked.whir, expected_whir);
    }
}
