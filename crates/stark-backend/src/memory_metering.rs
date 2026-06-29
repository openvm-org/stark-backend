//! Memory estimates for proving.

use std::{cmp::max, mem::size_of};

use crate::{
    prover::fractional_sumcheck_gkr::FractionalGkrMemoryModel, StarkProtocolConfig, SystemParams,
};

/// Fixed interaction scratch not proportional to interaction cells.
pub const INTERACTION_MEMORY_OVERHEAD: usize = 2 << 20;

const WHIR_SUMCHECK_DEGREE: usize = 2;
const CUDA_KERNEL_DEFAULT_BLOCK_SIZE: usize = 256;

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
    pub const fn new_with_unstacked_cells(
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
    pub base_field_bytes: usize,
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
    retained_opening_memory: bool,
    /// `log_2` of the common-main stacked matrix height.
    log_stacked_height: usize,
    /// WHIR folding factor.
    k_whir: usize,
    /// Number of WHIR rounds.
    num_whir_rounds: usize,
    /// Size of one PCS digest in bytes.
    digest_bytes: usize,
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
        digest_bytes: usize,
        cache_rs_code_matrix: bool,
    ) -> Self {
        Self {
            base_field_bytes: size_of::<F>(),
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
            digest_bytes,
        }
    }

    /// Calibrate CUDA-only retained state estimates.
    #[inline]
    pub fn with_cuda_backend(mut self, cache_stacked_matrix: bool) -> Self {
        self.cache_stacked_matrix = cache_stacked_matrix;
        self.retained_opening_memory = true;
        self
    }

    #[inline]
    pub fn main_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * self.base_field_bytes
    }

    #[inline]
    pub fn rs_code_matrix_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * (1usize << self.log_blowup) * self.base_field_bytes
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
    fn main_commitment_initial_digest_layer_memory_bytes(&self) -> usize {
        let log_codeword_height = self.log_stacked_height + self.log_blowup;
        let log_query_layer_height = log_codeword_height.saturating_sub(self.k_whir);
        (1usize << log_query_layer_height) * self.digest_bytes
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
    fn whir_initial_workspace_memory_bytes(&self) -> usize {
        let height = 1usize << self.log_stacked_height;
        let ext_poly = height
            .saturating_mul(self.extension_degree)
            .saturating_mul(self.base_field_bytes);
        let first_sumcheck_tmp = height
            .div_ceil(2)
            .div_ceil(CUDA_KERNEL_DEFAULT_BLOCK_SIZE)
            .saturating_mul(WHIR_SUMCHECK_DEGREE)
            .saturating_mul(self.extension_degree)
            .saturating_mul(self.base_field_bytes);

        // First WHIR sumcheck round keeps f_coeffs and w_moments live while allocating folded
        // outputs. This is larger than the earlier f_ple_evals + f_coeffs conversion peak.
        let first_sumcheck_peak = ext_poly
            .saturating_mul(3)
            .saturating_add(first_sumcheck_tmp);

        // After k_whir folds, f_coeffs, w_moments, and the split base-field g_coeffs are live
        // while building the first folded codeword tree.
        let folded_ext_poly = ext_poly >> self.k_whir.min(usize::BITS as usize - 1);
        let first_codeword_build_peak = folded_ext_poly.saturating_mul(3);

        max(first_sumcheck_peak, first_codeword_build_peak)
    }

    #[inline]
    fn whir_tree_memory_bytes(&self, log_codeword_height: usize) -> usize {
        (1usize << log_codeword_height) * self.extension_degree * self.base_field_bytes
            + self.merkle_digest_layers_memory_bytes(log_codeword_height)
    }

    #[inline]
    fn merkle_digest_layers_memory_bytes(&self, log_codeword_height: usize) -> usize {
        let log_bottom_layer = log_codeword_height.saturating_sub(self.k_whir);
        let bottom_layer_len = 1usize << log_bottom_layer;
        (2 * bottom_layer_len - 1) * self.digest_bytes
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
        ceil_weighted_bytes(main_cells, self.base_field_bytes, weight)
    }

    #[inline]
    pub fn main_secondary_memory_bytes(&self, counts: ProvingMemoryCounts) -> usize {
        self.main_secondary_memory_bytes_for_rot(counts.main_cells_with_rot, true)
            + self.main_secondary_memory_bytes_for_rot(counts.main_cells_without_rot, false)
    }

    /// Fractional-GKR input and peak scratch, before the fixed interaction overhead.
    ///
    /// The input layer stores one `Frac<EF>` per real interaction cell:
    ///
    /// ```text
    /// input = 2 * interaction_cells * D_EF * sizeof(F)
    /// ```
    ///
    /// Work-buffer sizing is delegated to the prover-owned fractional-GKR model so CUDA batching
    /// and metering use the same fold-eval/precompute-M peak.
    #[inline]
    pub fn interaction_memory_bytes_without_overhead(&self, interaction_cells: usize) -> usize {
        self.fractional_gkr_memory_model().peak_memory_bytes(
            interaction_cells,
            Self::fractional_gkr_logical_len(interaction_cells),
        )
    }

    #[inline]
    fn fractional_gkr_memory_model(&self) -> FractionalGkrMemoryModel {
        FractionalGkrMemoryModel::new(self.base_field_bytes, self.extension_degree)
    }

    #[inline]
    fn fractional_gkr_logical_len(interaction_cells: usize) -> usize {
        FractionalGkrMemoryModel::logical_len(interaction_cells)
    }

    #[inline]
    fn fractional_gkr_round_temp_buffer_memory_bytes(&self, interaction_cells: usize) -> usize {
        let logical_len = Self::fractional_gkr_logical_len(interaction_cells);
        if logical_len <= 2 {
            return 0;
        }

        FractionalGkrMemoryModel::round_temp_buffer_elements(
            logical_len >> 1,
            FractionalGkrMemoryModel::ROUND_COMPUTE_FALLBACK_BLOCKS,
        )
        .saturating_mul(self.extension_degree)
        .saturating_mul(self.base_field_bytes)
    }

    /// Interaction memory includes the fractional-GKR model plus the larger of fixed interaction
    /// scratch and CUDA round-reduction scratch:
    ///
    /// ```text
    /// interaction = gkr_input_and_work + max(2 MiB, round_temp_buffer)
    /// ```
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
    /// `main` is resident across the whole proving segment. `secondary_peak` is the maximum of the
    /// secondary phases that can overlap with that resident main memory.
    ///
    /// ```text
    /// main_cells         = main_cells_with_rot + main_cells_without_rot
    /// main_stacked_cells = stacked_width * 2^log_stacked_height
    /// main               = main_cells * sizeof(F)
    ///
    /// retained_common    = cached_stacked + common_digest
    ///
    /// cached_stacked     = main_stacked_cells * sizeof(F), when cached
    /// common_digest      = retained Merkle digest layers for the common-main commitment
    /// common_first_layer = first Merkle digest layer built from the RS code matrix
    /// rs_code_matrix     = main_stacked_cells * 2^log_blowup * sizeof(F)
    ///
    /// first_g_tree       = first WHIR folding codeword + Merkle digest layers
    /// second_g_tree      = second WHIR folding codeword + Merkle digest layers
    /// whir_workspace     = CUDA WHIR coefficient/moment buffers and sumcheck scratch
    ///
    /// main_secondary     = sizeof(F) *
    ///                      (2 * main_cell_secondary_weight() * main_cells_with_rot
    ///                       + main_cell_secondary_weight() * main_cells_without_rot)
    /// interaction        = fractional-GKR inputs + max supported work buffer
    ///                      + max(INTERACTION_MEMORY_OVERHEAD, fractional_gkr_round_temp_buffer)
    ///
    /// commit_peak = cached_stacked + if cache_rs_code_matrix {
    ///     rs_code_matrix + common_digest
    /// } else {
    ///     max(rs_code_matrix + common_first_layer, common_digest)
    /// }
    ///
    /// constraint_peak       = retained_common + cached_rs_code_matrix
    ///                         + max(main_secondary, interaction)
    /// whir_first_round_peak = retained_common + rs_code_matrix
    ///                         + max(whir_workspace, first_g_tree)
    /// whir_later_round_peak = first_g_tree + second_g_tree
    ///
    /// secondary_peak = max(
    ///     commit_peak,
    ///     constraint_peak,
    ///     whir_first_round_peak,
    ///     whir_later_round_peak,
    /// )
    /// total = main + secondary_peak
    /// ```
    #[inline]
    pub fn estimate(&self, counts: ProvingMemoryCounts) -> ProvingMemoryEstimate {
        let main_cells = counts.main_cells();
        let main = self.main_memory_bytes(main_cells);
        let has_common_main = counts.main_stacked_cells != 0;
        let retained_opening_memory = has_common_main && self.retained_opening_memory;
        let cached_stacked_matrix =
            self.cached_stacked_matrix_memory_bytes(counts.main_stacked_cells);
        let rs_code_matrix = self.rs_code_matrix_memory_bytes(counts.main_stacked_cells);
        let main_secondary = self.main_secondary_memory_bytes(counts);
        let interaction = self.interaction_memory_bytes(counts.interaction_cells);
        let (
            common_digest_layers,
            common_initial_digest_layer,
            first_g_tree,
            second_g_tree,
            whir_workspace,
        ) = if retained_opening_memory {
            (
                self.main_commitment_digest_layers_memory_bytes(),
                self.main_commitment_initial_digest_layer_memory_bytes(),
                self.first_whir_tree_memory_bytes(),
                self.second_whir_tree_memory_bytes(),
                self.whir_initial_workspace_memory_bytes(),
            )
        } else {
            (0, 0, 0, 0, 0)
        };
        let cached_rs_code_matrix = if self.cache_rs_code_matrix {
            rs_code_matrix
        } else {
            0
        };
        let retained_common_main = cached_stacked_matrix + common_digest_layers;
        let commit_peak = cached_stacked_matrix
            + if self.cache_rs_code_matrix {
                rs_code_matrix + common_digest_layers
            } else {
                max(
                    rs_code_matrix + common_initial_digest_layer,
                    common_digest_layers,
                )
            };
        let constraint_peak =
            retained_common_main + cached_rs_code_matrix + max(main_secondary, interaction);
        let whir_first_round_peak =
            retained_common_main + rs_code_matrix + max(whir_workspace, first_g_tree);
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

/// `ceil(cell_count * base_field_bytes * weight)`
fn ceil_weighted_bytes(cell_count: usize, base_field_bytes: usize, weight: f64) -> usize {
    ((cell_count * base_field_bytes) as f64 * weight).ceil() as usize
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
            main_stacked_cells * config.base_field_bytes
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

    #[test]
    fn dropped_rs_code_matrix_uses_phase_peak() {
        let mut config = test_memory_config();
        config.cache_rs_code_matrix = false;
        let counts = ProvingMemoryCounts::new_with_unstacked_cells(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(estimate.main, 30 * 4);
        assert_eq!(
            estimate.main_persistent,
            expected_retained_common_main(config, counts.main_stacked_cells)
        );
        assert_eq!(estimate.rs_code_matrix, 30 * 2 * 4);
        let commit_peak = max(
            estimate.rs_code_matrix + config.main_commitment_initial_digest_layer_memory_bytes(),
            estimate.main_persistent,
        );
        let constraint_peak =
            estimate.main_persistent + max(estimate.main_secondary, estimate.interaction);
        let whir_first_round_peak = estimate.main_persistent
            + estimate.rs_code_matrix
            + max(
                config.whir_initial_workspace_memory_bytes(),
                config.first_whir_tree_memory_bytes(),
            );
        let whir_later_round_peak =
            config.first_whir_tree_memory_bytes() + config.second_whir_tree_memory_bytes();
        assert_eq!(
            estimate.secondary_peak,
            max(
                max(commit_peak, constraint_peak),
                max(whir_first_round_peak, whir_later_round_peak)
            )
        );
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
    }

    #[test]
    fn cached_rs_code_matrix_is_retained_for_constraints() {
        let mut config = test_memory_config();
        config.num_whir_rounds = 1;
        let counts = ProvingMemoryCounts::new_with_unstacked_cells(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(
            estimate.main_persistent,
            expected_retained_common_main(config, counts.main_stacked_cells)
        );
        assert_eq!(
            estimate.secondary_peak,
            estimate.main_persistent
                + estimate.rs_code_matrix
                + max(estimate.main_secondary, estimate.interaction)
        );
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
    }

    #[test]
    fn rs_code_matrix_uses_stacked_main_cells() {
        let mut config = test_memory_config();
        config.cache_rs_code_matrix = false;
        let counts = ProvingMemoryCounts::new_with_stacked_cells(10, 20, 64, 5);

        let estimate = config.estimate(counts);

        assert_eq!(estimate.main, 30 * config.base_field_bytes);
        assert_eq!(
            estimate.rs_code_matrix,
            config.rs_code_matrix_memory_bytes(64)
        );
        assert_eq!(
            estimate.main_secondary,
            config.main_secondary_memory_bytes(ProvingMemoryCounts::new_with_unstacked_cells(
                10, 20, 5,
            ))
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
    fn interaction_memory_includes_fractional_gkr_work_buffers() {
        let config = test_memory_config();
        let interaction_cells = 16;
        let input = config
            .fractional_gkr_memory_model()
            .input_memory_bytes(interaction_cells);

        assert!(config.interaction_memory_bytes_without_overhead(interaction_cells) > input);
    }

    #[test]
    fn cuda_whir_peak_includes_initial_workspace() {
        let config = test_memory_config();
        let counts = ProvingMemoryCounts::new_with_stacked_cells(1, 0, 1, 0);
        let estimate = config.estimate(counts);
        let commit_peak = estimate.rs_code_matrix + estimate.main_persistent;
        let constraint_peak =
            estimate.main_persistent + estimate.rs_code_matrix + estimate.main_secondary;
        let whir_first_round_peak = estimate.main_persistent
            + estimate.rs_code_matrix
            + config.whir_initial_workspace_memory_bytes();
        let whir_later_round_peak =
            config.first_whir_tree_memory_bytes() + config.second_whir_tree_memory_bytes();

        assert_eq!(
            estimate.secondary_peak,
            max(
                max(commit_peak, constraint_peak),
                max(whir_first_round_peak, whir_later_round_peak)
            )
        );
    }

    #[test]
    fn uncached_commit_peak_keeps_initial_digest_layer_only() {
        let mut config = test_memory_config();
        config.cache_rs_code_matrix = false;
        config.num_whir_rounds = 1;
        config.max_constraint_degree = 0;
        let counts = ProvingMemoryCounts::new_with_stacked_cells(1, 0, 64, 0);

        let estimate = config.estimate(counts);
        let commit_peak = max(
            estimate.rs_code_matrix + config.main_commitment_initial_digest_layer_memory_bytes(),
            estimate.main_persistent,
        );
        let constraint_peak = estimate.main_persistent + estimate.main_secondary;
        let whir_first_round_peak = estimate.main_persistent
            + estimate.rs_code_matrix
            + max(
                config.whir_initial_workspace_memory_bytes(),
                config.first_whir_tree_memory_bytes(),
            );
        let whir_later_round_peak =
            config.first_whir_tree_memory_bytes() + config.second_whir_tree_memory_bytes();

        assert_eq!(
            estimate.secondary_peak,
            max(
                max(commit_peak, constraint_peak),
                max(whir_first_round_peak, whir_later_round_peak)
            )
        );
        assert_eq!(
            estimate.main_persistent,
            config.main_commitment_digest_layers_memory_bytes()
        );
    }

    #[test]
    fn cuda_backend_calibrates_derived_memory_state() {
        let params = default_test_params_small();
        let config = ProvingMemoryConfig::from_params::<u32>(&params, 4, TEST_DIGEST_SIZE, false)
            .with_cuda_backend(true);

        assert!(config.retained_opening_memory);
        assert!(config.cache_stacked_matrix);
    }

    #[test]
    fn generic_memory_config_has_no_persistent_main_memory() {
        let params = default_test_params_small();
        let estimate = ProvingMemoryConfig::from_params::<u32>(&params, 4, TEST_DIGEST_SIZE, true)
            .estimate(ProvingMemoryCounts::new_with_unstacked_cells(10, 20, 5));

        assert_eq!(estimate.main_persistent, 0);
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
    }

    #[test]
    fn empty_main_trace_has_no_persistent_main_memory() {
        let estimate =
            test_memory_config().estimate(ProvingMemoryCounts::new_with_unstacked_cells(0, 0, 0));

        assert_eq!(estimate.main, 0);
        assert_eq!(estimate.main_persistent, 0);
        assert_eq!(estimate.interaction, 0);
    }

    #[test]
    fn persistent_main_memory_uses_retained_common_main() {
        let config = test_memory_config();
        let counts = ProvingMemoryCounts::new_with_stacked_cells(1, 0, 1, 0);
        let estimate = config.estimate(counts);

        assert_eq!(
            estimate.main_persistent,
            expected_retained_common_main(config, 1)
        );
    }
}
