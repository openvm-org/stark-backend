//! Memory estimates for proving.

use std::{cmp::max, mem::size_of};

use crate::{StarkProtocolConfig, SystemParams};

/// Fixed interaction scratch not proportional to interaction cells.
pub const INTERACTION_MEMORY_OVERHEAD: usize = 2 << 20;

// Mirrors `frac_compute_round_launch_params` and `_frac_compute_round_temp_buffer_size`
// in the CUDA fractional-GKR kernels.
const FRACTIONAL_GKR_SP_DEG: usize = 2;
const FRACTIONAL_GKR_DEFAULT_BLOCK_SIZE: usize = 256;
const FRACTIONAL_GKR_MIN_BLOCK_SIZE: usize = 64;
const FRACTIONAL_GKR_WARP_SIZE: usize = 32;
const FRACTIONAL_GKR_ROUND_COMPUTE_FALLBACK_BLOCKS: usize = 228;

/// Cell counts for a proving memory estimate.
///
/// `main_cells_*` are `Σ(padded_height * width)` in base-field cells for traces opened by the
/// constraint and opening phases, split by whether a trace opens next-row rotations.
/// `interaction_cells` is the metered row-interaction slot count after power-of-two trace padding.
/// `main_stacked_cells` is the common-main stacked matrix size before Reed-Solomon blowup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProvingMemoryCounts {
    /// Opened trace cells for AIRs that open next-row rotations after power-of-two padding.
    pub main_cells_with_rot: usize,
    /// Opened trace cells for AIRs without next-row rotations after power-of-two padding.
    pub main_cells_without_rot: usize,
    /// Common-main stacked matrix cells before Reed-Solomon blowup.
    pub main_stacked_cells: usize,
    /// Metered row-interaction slots after power-of-two padding.
    pub interaction_cells: usize,
}

impl ProvingMemoryCounts {
    pub const fn new(
        main_cells_with_rot: usize,
        main_cells_without_rot: usize,
        interaction_cells: usize,
    ) -> Self {
        Self::new_with_stacked_cells(
            main_cells_with_rot,
            main_cells_without_rot,
            main_cells_with_rot + main_cells_without_rot,
            interaction_cells,
        )
    }

    pub const fn new_with_stacked_cells(
        main_cells_with_rot: usize,
        main_cells_without_rot: usize,
        main_stacked_cells: usize,
        interaction_cells: usize,
    ) -> Self {
        Self {
            main_cells_with_rot,
            main_cells_without_rot,
            main_stacked_cells,
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
    /// Resident opened trace data.
    pub main: usize,
    /// Retained common-main PCS data used by the selected secondary phase.
    pub main_persistent: usize,
    /// Reed-Solomon code matrix for main traces.
    pub rs_code_matrix: usize,
    /// Secondary main-trace buffers after PCS opening.
    pub main_secondary: usize,
    /// Interaction GKR buffers plus fractional-GKR scratch.
    pub interaction: usize,
    /// Peak among secondary phases, excluding resident opened trace data.
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
    /// Whether the prover keeps the stacked matrix cached after `stacked_commit`.
    pub cache_stacked_matrix: bool,
    /// Whether the prover keeps the Reed-Solomon code matrix cached after `stacked_commit`.
    pub cache_rs_code_matrix: bool,
    /// Whether to include CUDA retained opening state in the estimate.
    pub retained_opening_memory: bool,
    /// `log_2` of the common-main stacked matrix height.
    pub log_stacked_height: usize,
    /// WHIR folding factor.
    pub k_whir: usize,
    /// Number of WHIR rounds.
    pub num_whir_rounds: usize,
    /// Size of one PCS digest in bytes.
    pub digest_size: usize,
    /// Target minimum block count for CUDA fractional-GKR round kernels.
    pub fractional_gkr_round_compute_min_blocks: usize,
}

impl ProvingMemoryConfig {
    pub fn from_protocol_config<SC: StarkProtocolConfig>(
        config: &SC,
        cache_rs_code_matrix: bool,
    ) -> Self {
        Self::from_params::<SC::F>(
            config.params(),
            SC::D_EF,
            size_of::<SC::Digest>(),
            cache_rs_code_matrix,
        )
    }

    fn from_params<F>(
        params: &SystemParams,
        extension_degree: usize,
        digest_size: usize,
        cache_rs_code_matrix: bool,
    ) -> Self {
        Self {
            base_field_size: size_of::<F>(),
            extension_degree,
            log_blowup: params.log_blowup,
            l_skip: params.l_skip,
            max_constraint_degree: params.max_constraint_degree,
            cache_stacked_matrix: false,
            cache_rs_code_matrix,
            retained_opening_memory: false,
            log_stacked_height: params.log_stacked_height(),
            k_whir: params.k_whir(),
            num_whir_rounds: params.num_whir_rounds(),
            digest_size,
            fractional_gkr_round_compute_min_blocks: FRACTIONAL_GKR_ROUND_COMPUTE_FALLBACK_BLOCKS,
        }
    }

    #[inline]
    pub fn main_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * self.base_field_size
    }

    #[inline]
    pub fn rs_code_matrix_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * (1usize << self.log_blowup) * self.base_field_size
    }

    #[inline]
    pub fn stacked_matrix_cells(&self, stacked_slice_cells: usize) -> usize {
        if stacked_slice_cells == 0 {
            return 0;
        }

        let stacked_height = 1usize << self.log_stacked_height;
        stacked_slice_cells.div_ceil(stacked_height) * stacked_height
    }

    #[inline]
    pub fn stacked_slice_height(&self, padded_height: usize) -> usize {
        padded_height.max(1usize << self.l_skip)
    }

    #[inline]
    pub fn main_persistent_memory_bytes(
        &self,
        main_cells: usize,
        main_stacked_cells: usize,
    ) -> usize {
        if main_cells == 0 || main_stacked_cells == 0 {
            return 0;
        }

        let common_digest_layers = if self.retained_opening_memory {
            self.main_commitment_digest_layers_memory_bytes()
        } else {
            0
        };
        self.cached_stacked_matrix_memory_bytes(main_stacked_cells) + common_digest_layers
    }

    #[inline]
    fn cached_stacked_matrix_memory_bytes(&self, main_stacked_cells: usize) -> usize {
        if self.cache_stacked_matrix {
            self.main_memory_bytes(main_stacked_cells)
        } else {
            0
        }
    }

    #[inline]
    fn main_commitment_digest_layers_memory_bytes(&self) -> usize {
        self.merkle_digest_layers_memory_bytes(self.log_stacked_height + self.log_blowup)
    }

    #[inline]
    fn first_whir_tree_memory_bytes(&self) -> usize {
        (self.log_stacked_height + self.log_blowup)
            .checked_sub(1)
            .map_or(0, |log_codeword_height| {
                self.whir_tree_memory_bytes(log_codeword_height)
            })
    }

    #[inline]
    fn second_whir_tree_memory_bytes(&self) -> usize {
        (self.num_whir_rounds > 2)
            .then(|| self.log_stacked_height + self.log_blowup)
            .and_then(|log_codeword_height| log_codeword_height.checked_sub(2))
            .map_or(0, |log_codeword_height| {
                self.whir_tree_memory_bytes(log_codeword_height)
            })
    }

    #[inline]
    fn whir_tree_memory_bytes(&self, log_codeword_height: usize) -> usize {
        (1usize << log_codeword_height) * self.extension_degree * self.base_field_size
            + self.merkle_digest_layers_memory_bytes(log_codeword_height)
    }

    #[inline]
    fn merkle_digest_layers_memory_bytes(&self, log_codeword_height: usize) -> usize {
        let log_bottom_layer = log_codeword_height.saturating_sub(self.k_whir);
        let bottom_layer_len = 1usize << log_bottom_layer;
        (2 * bottom_layer_len - 1) * self.digest_size
    }

    #[inline]
    pub fn main_cell_secondary_weight(&self) -> f64 {
        default_main_cell_secondary_weight(
            self.extension_degree,
            self.l_skip,
            self.max_constraint_degree,
        )
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
        let main_cell_secondary_weight = self.main_cell_secondary_weight();
        let weight = if need_rot {
            2.0 * main_cell_secondary_weight
        } else {
            main_cell_secondary_weight
        };
        ceil_weighted_bytes(main_cells, self.base_field_size, weight)
    }

    #[inline]
    pub fn main_secondary_memory_bytes(&self, counts: ProvingMemoryCounts) -> usize {
        self.main_secondary_memory_bytes_for_rot(counts.main_cells_with_rot, true)
            + self.main_secondary_memory_bytes_for_rot(counts.main_cells_without_rot, false)
    }

    #[inline]
    pub fn interaction_memory_bytes_without_overhead(&self, interaction_cells: usize) -> usize {
        ceil_weighted_bytes(
            interaction_cells,
            self.base_field_size,
            default_interaction_cell_weight(self.extension_degree),
        )
    }

    #[inline]
    fn fractional_gkr_logical_len(&self, interaction_cells: usize) -> usize {
        if interaction_cells == 0 {
            return 0;
        }

        let log_len = usize::BITS - interaction_cells.leading_zeros();
        1usize.checked_shl(log_len).unwrap_or(usize::MAX)
    }

    #[inline]
    fn fractional_gkr_round_temp_buffer_memory_bytes(&self, interaction_cells: usize) -> usize {
        let logical_len = self.fractional_gkr_logical_len(interaction_cells);
        if logical_len <= 2 {
            return 0;
        }

        let elements = logical_len >> 2;
        let min_blocks = self.fractional_gkr_round_compute_min_blocks.max(1);
        let mut block_size = FRACTIONAL_GKR_DEFAULT_BLOCK_SIZE;
        let mut blocks_needed = elements.div_ceil(block_size);
        if blocks_needed < min_blocks && elements >= FRACTIONAL_GKR_MIN_BLOCK_SIZE {
            block_size = elements.div_ceil(min_blocks);
            block_size = block_size
                .div_ceil(FRACTIONAL_GKR_WARP_SIZE)
                .saturating_mul(FRACTIONAL_GKR_WARP_SIZE)
                .max(FRACTIONAL_GKR_MIN_BLOCK_SIZE);
            blocks_needed = elements.div_ceil(block_size);
        }

        blocks_needed
            .saturating_mul(FRACTIONAL_GKR_SP_DEG)
            .saturating_mul(self.extension_degree)
            .saturating_mul(self.base_field_size)
    }

    #[inline]
    pub fn interaction_memory_bytes(&self, interaction_cells: usize) -> usize {
        if interaction_cells == 0 {
            return 0;
        }

        self.interaction_memory_bytes_without_overhead(interaction_cells)
            + max(
                INTERACTION_MEMORY_OVERHEAD,
                self.fractional_gkr_round_temp_buffer_memory_bytes(interaction_cells),
            )
    }

    /// Convert opened trace cells, common-main stacked cells, and interaction cells to memory
    /// bytes.
    ///
    /// ```text
    /// main_cells         = main_cells_with_rot + main_cells_without_rot
    /// main_stacked_cells = stacked_width * 2^log_stacked_height
    /// main               = main_cells * sizeof(F)
    /// cached_stacked     = main_stacked_cells * sizeof(F), when cached
    /// common_digest      = Merkle digest layers for the common-main commitment
    /// rs_code_matrix     = main_stacked_cells * 2^log_blowup * sizeof(F)
    /// first_g_tree       = first WHIR folding codeword + Merkle digest layers
    /// second_g_tree      = second WHIR folding codeword + Merkle digest layers
    /// main_secondary     = sizeof(F) *
    ///                      (2 * main_cell_secondary_weight() * main_cells_with_rot
    ///                       + main_cell_secondary_weight() * main_cells_without_rot)
    /// interaction        = sizeof(F) * interaction_cell_weight * interaction_cells
    ///                      + max(INTERACTION_MEMORY_OVERHEAD, fractional_gkr_round_temp_buffer)
    ///
    /// total = main + max(
    ///     cached_stacked + common_digest + rs_code_matrix,
    ///     cached_stacked + common_digest + cached_rs_code_matrix + max(main_secondary, interaction),
    ///     cached_stacked + common_digest + rs_code_matrix + first_g_tree,
    ///     first_g_tree + second_g_tree,
    /// )
    /// ```
    #[inline]
    pub fn estimate(&self, counts: ProvingMemoryCounts) -> ProvingMemoryEstimate {
        let main_cells = counts.main_cells();
        let main = self.main_memory_bytes(main_cells);
        let has_common_main = counts.main_stacked_cells != 0;
        let cached_stacked_matrix = if has_common_main {
            self.cached_stacked_matrix_memory_bytes(counts.main_stacked_cells)
        } else {
            0
        };
        let common_digest_layers = if has_common_main && self.retained_opening_memory {
            self.main_commitment_digest_layers_memory_bytes()
        } else {
            0
        };
        let rs_code_matrix = self.rs_code_matrix_memory_bytes(counts.main_stacked_cells);
        let main_secondary = self.main_secondary_memory_bytes(counts);
        let interaction = self.interaction_memory_bytes(counts.interaction_cells);
        let first_g_tree = if has_common_main && self.retained_opening_memory {
            self.first_whir_tree_memory_bytes()
        } else {
            0
        };
        let second_g_tree = if has_common_main && self.retained_opening_memory {
            self.second_whir_tree_memory_bytes()
        } else {
            0
        };
        let cached_rs_code_matrix = if self.cache_rs_code_matrix {
            rs_code_matrix
        } else {
            0
        };
        let retained_common_main = cached_stacked_matrix + common_digest_layers;
        let commit_peak = retained_common_main + rs_code_matrix;
        let constraint_peak =
            retained_common_main + cached_rs_code_matrix + max(main_secondary, interaction);
        let whir_first_round_peak = retained_common_main + rs_code_matrix + first_g_tree;
        let whir_later_round_peak = first_g_tree + second_g_tree;
        let secondary_peak = max(
            max(commit_peak, constraint_peak),
            max(whir_first_round_peak, whir_later_round_peak),
        );
        let main_persistent = retained_common_main;

        ProvingMemoryEstimate {
            total: main + secondary_peak,
            main,
            main_persistent,
            rs_code_matrix,
            main_secondary,
            interaction,
            secondary_peak,
        }
    }
}

fn default_main_cell_secondary_weight(
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

    const TEST_DIGEST_SIZE: usize = 32;

    fn test_memory_config() -> ProvingMemoryConfig {
        let params = default_test_params_small();
        let mut config =
            ProvingMemoryConfig::from_params::<u32>(&params, 4, TEST_DIGEST_SIZE, true);
        config.retained_opening_memory = true;
        config
    }

    fn expected_retained_common_main(
        config: ProvingMemoryConfig,
        main_stacked_cells: usize,
    ) -> usize {
        let cached_stacked_matrix = if config.cache_stacked_matrix {
            main_stacked_cells * config.base_field_size
        } else {
            0
        };
        let common_digest_layers = if config.retained_opening_memory && main_stacked_cells != 0 {
            config.main_commitment_digest_layers_memory_bytes()
        } else {
            0
        };
        cached_stacked_matrix + common_digest_layers
    }

    fn expected_secondary_peak(config: ProvingMemoryConfig, counts: ProvingMemoryCounts) -> usize {
        let retained_common_main = expected_retained_common_main(config, counts.main_stacked_cells);
        let rs_code_matrix = config.rs_code_matrix_memory_bytes(counts.main_stacked_cells);
        let cached_rs_code_matrix = if config.cache_rs_code_matrix {
            rs_code_matrix
        } else {
            0
        };
        let main_secondary = config.main_secondary_memory_bytes(counts);
        let interaction = config.interaction_memory_bytes(counts.interaction_cells);
        let has_common_main = counts.main_stacked_cells != 0 && config.retained_opening_memory;
        let first_g_tree = if has_common_main {
            config.first_whir_tree_memory_bytes()
        } else {
            0
        };
        let second_g_tree = if has_common_main {
            config.second_whir_tree_memory_bytes()
        } else {
            0
        };
        let commit_peak = retained_common_main + rs_code_matrix;
        let constraint_peak =
            retained_common_main + cached_rs_code_matrix + max(main_secondary, interaction);
        let whir_first_round_peak = retained_common_main + rs_code_matrix + first_g_tree;
        let whir_later_round_peak = first_g_tree + second_g_tree;
        max(
            max(commit_peak, constraint_peak),
            max(whir_first_round_peak, whir_later_round_peak),
        )
    }

    #[test]
    fn dropped_rs_code_matrix_uses_phase_peak() {
        let mut config = test_memory_config();
        config.cache_rs_code_matrix = false;
        let counts = ProvingMemoryCounts::new(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(estimate.main, 30 * 4);
        assert_eq!(
            estimate.main_persistent,
            expected_retained_common_main(config, counts.main_stacked_cells)
        );
        assert_eq!(estimate.rs_code_matrix, 30 * 2 * 4);
        assert_eq!(
            estimate.secondary_peak,
            expected_secondary_peak(config, counts)
        );
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
    }

    #[test]
    fn cached_rs_code_matrix_is_retained_for_constraints() {
        let config = test_memory_config();
        let counts = ProvingMemoryCounts::new(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(
            estimate.main_persistent,
            expected_retained_common_main(config, counts.main_stacked_cells)
        );
        assert_eq!(
            estimate.secondary_peak,
            expected_secondary_peak(config, counts)
        );
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
    }

    #[test]
    fn rs_code_matrix_uses_stacked_main_cells() {
        let mut config = test_memory_config();
        config.cache_rs_code_matrix = false;
        let counts = ProvingMemoryCounts::new_with_stacked_cells(10, 20, 64, 5);

        let estimate = config.estimate(counts);

        assert_eq!(estimate.main, 30 * config.base_field_size);
        assert_eq!(
            estimate.rs_code_matrix,
            config.rs_code_matrix_memory_bytes(64)
        );
        assert_eq!(
            estimate.main_secondary,
            config.main_secondary_memory_bytes(ProvingMemoryCounts::new(10, 20, 5))
        );
    }

    #[test]
    fn cached_stacked_matrix_is_retained_opening_memory() {
        let mut config = test_memory_config();
        config.cache_stacked_matrix = true;
        let counts = ProvingMemoryCounts::new_with_stacked_cells(10, 20, 64, 5);

        let estimate = config.estimate(counts);

        assert_eq!(
            estimate.main_persistent,
            expected_retained_common_main(config, counts.main_stacked_cells)
        );
    }

    #[test]
    fn generic_memory_config_has_no_persistent_main_memory() {
        let params = default_test_params_small();
        let estimate = ProvingMemoryConfig::from_params::<u32>(&params, 4, TEST_DIGEST_SIZE, true)
            .estimate(ProvingMemoryCounts::new(10, 20, 5));

        assert_eq!(estimate.main_persistent, 0);
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
    }

    #[test]
    fn empty_main_trace_has_no_persistent_main_memory() {
        let estimate = test_memory_config().estimate(ProvingMemoryCounts::new(0, 0, 0));

        assert_eq!(estimate.main, 0);
        assert_eq!(estimate.main_persistent, 0);
        assert_eq!(estimate.interaction, 0);
    }

    #[test]
    fn persistent_main_memory_uses_retained_common_main() {
        let config = test_memory_config();

        assert_eq!(
            config.main_persistent_memory_bytes(1, 1),
            expected_retained_common_main(config, 1)
        );
    }
}
