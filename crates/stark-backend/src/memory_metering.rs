//! Memory estimates for proving.

use std::{cmp::max, mem::size_of};

use crate::{StarkProtocolConfig, SystemParams};

/// Fixed interaction scratch not proportional to interaction cells: challenge/metadata
/// buffers, per-bus tables, and allocator page rounding across the live GKR buffers.
pub const INTERACTION_MEMORY_OVERHEAD: usize = 64 << 20;

/// Minimum fractional-GKR work-buffer length in `Frac<EF>` entries. Mirrors
/// `GKR_WINDOW_DEFAULT_MIN_N` in the CUDA fractional sumcheck: the work buffer never
/// shrinks below `2^22` entries, which dominates the `logical_len / 16` sizing for
/// small interaction counts.
pub const GKR_MIN_WORK_BUFFER_LEN: usize = 1 << 22;

/// Fixed WHIR opening scratch not proportional to the stacked height: folded sumcheck
/// buffers, query-opening staging, proof-of-work grinding, and allocator page rounding
/// across the many live buffers at the round-0 query step.
pub const WHIR_MEMORY_OVERHEAD: usize = 64 << 20;

/// Fixed batch-constraint scratch on top of the modeled `mat_eval`/interpolation buffers:
/// eq segment trees, folded selector columns, logup monomial combination tables, and
/// allocator page rounding. Calibrated against measured batch-phase peaks (~90 MiB at
/// 2^22 stacked height on the reth profile).
pub const BATCH_CONSTRAINT_MEMORY_OVERHEAD: usize = 192 << 20;

/// Minimum batch-MLE scratch budget when `zerocheck_save_memory` is off. Mirrors
/// `BATCH_MLE_DEFAULT_MEMORY_FLOOR` in `openvm-cuda-backend`: without the save-memory
/// subtraction, evaluation scratch is greedily packed up to
/// `max(gkr_peak, BATCH_MLE_MEMORY_FLOOR)` *in addition to* the `mat_eval` buffers.
pub const BATCH_MLE_MEMORY_FLOOR: usize = 6 << 30;

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
    /// WHIR opening working set that coexists with the RS code matrix at the round-0
    /// query step (commitment Merkle layers, round-0 codeword and its tree, scratch).
    pub whir_overhead: usize,
    /// Batch-constraint `mat_eval` and interpolation buffers derived from the main traces.
    pub main_secondary: usize,
    /// Interaction GKR buffers plus fixed interaction overhead.
    pub interaction: usize,
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
    /// Whether the prover keeps the Reed-Solomon code matrix cached after `stacked_commit`.
    pub cache_rs_code_matrix: bool,
    /// Whether the prover keeps the stacked matrix cached after `stacked_commit`
    /// (`GpuProverConfig::cache_stacked_matrix`).
    pub cache_stacked_matrix: bool,
    /// Whether the batch-MLE scratch budget is reduced by the resident `mat_eval` buffers
    /// (`GpuProverConfig::zerocheck_save_memory`). When off, evaluation scratch up to
    /// `max(gkr_peak, BATCH_MLE_MEMORY_FLOOR)` is additive on `main_secondary`.
    pub zerocheck_save_memory: bool,
}

impl ProvingMemoryConfig {
    pub fn from_protocol_config<SC: StarkProtocolConfig>(
        config: &SC,
        cache_rs_code_matrix: bool,
    ) -> Self {
        Self::from_params::<SC::F, SC::Digest>(config.params(), SC::D_EF, cache_rs_code_matrix)
    }

    fn from_params<F, Digest>(
        params: &SystemParams,
        extension_degree: usize,
        cache_rs_code_matrix: bool,
    ) -> Self {
        Self {
            base_field_size: size_of::<F>(),
            extension_degree,
            digest_size: size_of::<Digest>(),
            log_blowup: params.log_blowup,
            l_skip: params.l_skip,
            log_stacked_height: params.l_skip + params.n_stack,
            k_whir: params.k_whir(),
            max_constraint_degree: params.max_constraint_degree,
            cache_rs_code_matrix,
            cache_stacked_matrix: false,
            // Default coupling used by `GpuDevice::new`; override with the actual prover
            // config flag when available.
            zerocheck_save_memory: params.log_blowup == 1,
        }
    }

    #[inline]
    pub fn main_memory_bytes(&self, main_cells: usize) -> usize {
        main_cells * self.base_field_size
    }

    /// Bytes of the Reed-Solomon code matrix of the stacked traces.
    ///
    /// Stacking pads the used width up to whole columns of height `2^log_stacked_height`,
    /// so the codeword can cover up to one stacked column beyond the raw cell count.
    #[inline]
    pub fn rs_code_matrix_memory_bytes(&self, main_cells: usize) -> usize {
        (main_cells + self.stacked_height()) * (1usize << self.log_blowup) * self.base_field_size
    }

    #[inline]
    pub fn stacked_height(&self) -> usize {
        1usize << self.log_stacked_height
    }

    /// Peak WHIR opening working set that coexists with the resident main traces and the
    /// (cached or recomputed) RS code matrix at the round-0 query step:
    ///
    /// ```text
    /// codeword_height = 2^(log_stacked_height + log_blowup)
    /// commit_tree     = 2 * digest_size * codeword_height / 2^k_whir
    /// g_codeword      = D_EF * base_field_size * codeword_height / 2
    /// g_tree          = 2 * digest_size * codeword_height / 2^(k_whir + 1)
    /// whir_overhead   = commit_tree + g_codeword + g_tree + WHIR_MEMORY_OVERHEAD
    /// ```
    ///
    /// `commit_tree` is the digest layers of the commitment Merkle tree (still open for
    /// query proofs), `g_codeword`/`g_tree` are the round-0 folded codeword and its tree.
    #[inline]
    pub fn whir_overhead_memory_bytes(&self) -> usize {
        let codeword_height = self.stacked_height() << self.log_blowup;
        let commit_tree = 2 * self.digest_size * (codeword_height >> self.k_whir);
        let g_codeword = self.extension_degree * self.base_field_size * (codeword_height >> 1);
        let g_tree = 2 * self.digest_size * (codeword_height >> (self.k_whir + 1));
        commit_tree + g_codeword + g_tree + WHIR_MEMORY_OVERHEAD
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

    /// Peak fractional-GKR buffer bytes for `interaction_cells` metered interaction slots.
    ///
    /// Mirrors the CUDA fractional sumcheck (see `FractionalInputSize` in
    /// `openvm-cuda-backend`), with `logical_len = next_power_of_two(real_len) <= 2 * real_len`
    /// and one `Frac<EF>` (two extension-field elements) per slot:
    ///
    /// ```text
    /// leaf_bytes     = 2 * extension_degree * base_field_size
    /// leaves         = interaction_cells * leaf_bytes
    /// work_buffer    = max(logical_len / 16, GKR_MIN_WORK_BUFFER_LEN) * leaf_bytes
    /// tmp_block_sums = logical_len / 256 * leaf_bytes
    /// ```
    #[inline]
    pub fn interaction_memory_bytes_without_overhead(&self, interaction_cells: usize) -> usize {
        let leaf_bytes = 2 * self.extension_degree * self.base_field_size;
        let logical_len_bound = 2 * interaction_cells;
        let leaves = interaction_cells * leaf_bytes;
        let work_buffer = max(logical_len_bound / 16, GKR_MIN_WORK_BUFFER_LEN) * leaf_bytes;
        let tmp_block_sums = logical_len_bound / 256 * leaf_bytes;
        leaves + work_buffer + tmp_block_sums
    }

    #[inline]
    pub fn interaction_memory_bytes(&self, interaction_cells: usize) -> usize {
        self.interaction_memory_bytes_without_overhead(interaction_cells)
            + INTERACTION_MEMORY_OVERHEAD
    }

    /// Convert main trace cells and interaction cells to memory bytes.
    ///
    /// ```text
    /// main_cells       = main_cells_with_rot + main_cells_without_rot
    /// main             = main_cells * sizeof(F)
    /// rs_code_matrix   = (main_cells + 2^log_stacked_height) * 2^log_blowup * sizeof(F)
    /// whir_overhead    = whir_overhead_memory_bytes()
    /// main_secondary   = sizeof(F) *
    ///                    (2 * main_cell_secondary_weight() * main_cells_with_rot
    ///                     + main_cell_secondary_weight() * main_cells_without_rot)
    /// interaction      = interaction_memory_bytes_without_overhead(interaction_cells)
    ///                    + INTERACTION_MEMORY_OVERHEAD
    /// ```
    ///
    /// The main traces stay resident through all proving phases, so `total` is `main` plus
    /// the peak among the phases that follow (plus the resident stacked matrix when
    /// `cache_stacked_matrix` is on). The opening phase holds the RS code matrix
    /// (recomputed if not cached) together with the WHIR round-0 working set, which is why
    /// `whir_overhead` is added on top of `rs_code_matrix` rather than maxed with it.
    ///
    /// The batch-constraint phase costs `main_secondary` plus fixed scratch; with
    /// `zerocheck_save_memory` the evaluation scratch is budgeted inside the GKR peak
    /// (`batch = max(main_secondary, interaction) + overhead`), without it the scratch is
    /// additive (`batch = main_secondary + max(gkr, BATCH_MLE_MEMORY_FLOOR) + overhead`).
    ///
    /// Cached RS code matrix:
    ///
    /// ```text
    /// total = main + stacked + rs_code_matrix + max(whir_overhead, batch, interaction)
    /// ```
    ///
    /// Dropped RS code matrix:
    ///
    /// ```text
    /// total = main + stacked + max(rs_code_matrix + whir_overhead, batch, interaction)
    /// ```
    #[inline]
    pub fn estimate(&self, counts: ProvingMemoryCounts) -> ProvingMemoryEstimate {
        let main_cells = counts.main_cells();
        let main = self.main_memory_bytes(main_cells);
        let rs_code_matrix = self.rs_code_matrix_memory_bytes(main_cells);
        let whir_overhead = self.whir_overhead_memory_bytes();
        let main_secondary = self.main_secondary_memory_bytes(counts);
        let gkr = self.interaction_memory_bytes_without_overhead(counts.interaction_cells);
        let interaction = gkr + INTERACTION_MEMORY_OVERHEAD;
        let batch_secondary = if self.zerocheck_save_memory {
            max(main_secondary, gkr) + BATCH_CONSTRAINT_MEMORY_OVERHEAD
        } else {
            main_secondary + max(gkr, BATCH_MLE_MEMORY_FLOOR) + BATCH_CONSTRAINT_MEMORY_OVERHEAD
        };
        // Kept stacked matrix is resident through all phases, like `main`.
        let stacked_matrix = if self.cache_stacked_matrix {
            (main_cells + self.stacked_height()) * self.base_field_size
        } else {
            0
        };
        let secondary_phases = max(batch_secondary, interaction);
        let secondary_peak = if self.cache_rs_code_matrix {
            rs_code_matrix + max(whir_overhead, secondary_phases)
        } else {
            max(rs_code_matrix + whir_overhead, secondary_phases)
        };

        ProvingMemoryEstimate {
            total: main + stacked_matrix + secondary_peak,
            main,
            rs_code_matrix,
            whir_overhead,
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
        ProvingMemoryConfig::from_params::<u32, [u32; 8]>(&params, 4, true)
    }

    fn batch_secondary(config: &ProvingMemoryConfig, counts: ProvingMemoryCounts) -> usize {
        let gkr = config.interaction_memory_bytes_without_overhead(counts.interaction_cells);
        if config.zerocheck_save_memory {
            max(config.main_secondary_memory_bytes(counts), gkr) + BATCH_CONSTRAINT_MEMORY_OVERHEAD
        } else {
            config.main_secondary_memory_bytes(counts)
                + max(gkr, BATCH_MLE_MEMORY_FLOOR)
                + BATCH_CONSTRAINT_MEMORY_OVERHEAD
        }
    }

    #[test]
    fn dropped_rs_code_matrix_is_phase_disjoint() {
        let params = default_test_params_small();
        let config = ProvingMemoryConfig::from_params::<u32, [u32; 8]>(&params, 4, false);
        let counts = ProvingMemoryCounts::new(10, 20, 5);

        let estimate = config.estimate(counts);

        assert_eq!(estimate.main, 30 * 4);
        assert_eq!(
            estimate.rs_code_matrix,
            (30 + config.stacked_height()) * (1 << params.log_blowup) * 4
        );
        assert_eq!(estimate.total, estimate.main + estimate.secondary_peak);
        assert_eq!(
            estimate.secondary_peak,
            max(
                estimate.rs_code_matrix + estimate.whir_overhead,
                max(batch_secondary(&config, counts), estimate.interaction)
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
                + max(
                    estimate.whir_overhead,
                    max(batch_secondary(&config, counts), estimate.interaction)
                )
        );
    }

    #[test]
    fn no_save_memory_batch_scratch_is_additive() {
        let params = default_test_params_small();
        let mut config = ProvingMemoryConfig::from_params::<u32, [u32; 8]>(&params, 4, false);
        let counts = ProvingMemoryCounts::new(10, 20, 5);

        config.zerocheck_save_memory = true;
        let saved = config.estimate(counts);
        config.zerocheck_save_memory = false;
        let unsaved = config.estimate(counts);

        assert!(unsaved.total > saved.total);
        assert_eq!(
            unsaved.secondary_peak,
            max(
                unsaved.rs_code_matrix + unsaved.whir_overhead,
                max(batch_secondary(&config, counts), unsaved.interaction)
            )
        );
    }

    #[test]
    fn cached_stacked_matrix_is_resident() {
        let params = default_test_params_small();
        let mut config = ProvingMemoryConfig::from_params::<u32, [u32; 8]>(&params, 4, false);
        let counts = ProvingMemoryCounts::new(10, 20, 5);

        let without = config.estimate(counts);
        config.cache_stacked_matrix = true;
        let with = config.estimate(counts);

        assert_eq!(
            with.total - without.total,
            (counts.main_cells() + config.stacked_height()) * config.base_field_size
        );
    }

    #[test]
    fn whir_overhead_component_formulas() {
        let config = test_memory_config();
        let codeword_height = config.stacked_height() << config.log_blowup;

        let commit_tree = 2 * config.digest_size * (codeword_height >> config.k_whir);
        let g_codeword = config.extension_degree * config.base_field_size * (codeword_height >> 1);
        let g_tree = 2 * config.digest_size * (codeword_height >> (config.k_whir + 1));
        assert_eq!(
            config.whir_overhead_memory_bytes(),
            commit_tree + g_codeword + g_tree + WHIR_MEMORY_OVERHEAD
        );
    }
}
