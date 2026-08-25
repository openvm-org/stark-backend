use openvm_cuda_common::stream::cudaStream_t;

use super::*;
use crate::{
    monomial::{InteractionMonomialTerm, LambdaTerm, MonomialHeader, PackedVar},
    poly::SqrtEqLayers,
};

/// A device address encoded as a byte offset from a single base pointer —
/// the ABI mirrored by `BaseOff` in `cuda/include/base_off.cuh`.
///
/// Two producers encode into it:
///
/// * the **graph-IR** path stores `GraphExe::plan().offsets[b]` (plus an intra-buffer byte offset)
///   and hands the kernel the base of the exe's unified device pool, so the descriptor arrays hold
///   *integers* and no device pointer is ever embedded in an uploaded struct;
/// * the **eager** path stores the absolute device address ([`Self::from_ptr`]) and hands the
///   kernel a null base, so `base + off` reproduces the original pointer exactly and eager
///   behaviour is unchanged.
///
/// [`Self::NULL`] is the "absent" encoding. Offset `0` cannot serve as the
/// sentinel: it is a valid pool offset — the first packed buffer lives there.
///
/// # Exactly what is on this ABI, and what is not
///
/// The conversion is **not** universal, and the boundary is load-bearing for
/// anyone reasoning about which pointers the graph compiler can see through.
///
/// On the base+offset ABI (every device-pointer field is a [`BaseOff`]):
///
/// * [`MainMatrixDesc`] — `data`
/// * [`EvalCoreCtx`] — `d_selectors`, `d_preprocessed.data`, `d_main`, `d_public`
/// * [`ZerocheckCtx`] — `d_intermediates`, `d_eq_xi`, `d_rules`, `d_used_nodes`
/// * [`LogupCtx`] — `d_intermediates`, `d_eq_xi`, `d_challenges`, `d_eq_3bs`, `d_rules`,
///   `d_used_nodes`, `d_pair_idxs`
///
/// Still raw device pointers, on the *graph* path too:
///
/// * [`MonomialAirCtx`] — `d_headers`, `d_variables`, `d_lambda_combinations`, `d_eq_xi` (its
///   nested `eval_ctx` **is** converted)
/// * [`LogupMonomialCommonCtx`] — `d_eq_xi` (nested `eval_ctx` converted)
/// * [`LogupMonomialCtx`] — `d_headers`, `d_variables`, `d_combinations` (all three)
/// * [`GkrInputCtx`] — all nine pointer fields
/// * The bare `T *const *` tables whose kernel ABI is a pointer table rather than a descriptor
///   struct: `batch_fold_mle`'s `input_matrices` / `output_matrices` and `interpolate_columns`'
///   column table. The graph path fills these with absolute addresses computed as `pool_base +
///   offset` after compile (`DescElem::RawPtr` in `logup_zerocheck/zerocheck_ir.rs`) — host-known
///   and stable, but not a `BaseOff` the kernel decodes.
///
/// The monomial and GKR fields above are keygen-static tables, so they are not
/// graph buffers today and there is nothing for an offset to be relative to.
/// Converting them is only worthwhile once they become graph inputs.
///
/// # The eager path is not on this ABI at all
///
/// See [`MainMatrixPtrs`]: the eager prover keeps the original raw-pointer
/// context structs and its own entry points, so it stays an independent
/// reference for the encoding above.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BaseOff(pub u64);

impl BaseOff {
    /// Decodes to a null pointer for every base.
    pub const NULL: Self = BaseOff(u64::MAX);

    /// A byte offset inside the graph's unified pool.
    pub const fn from_offset(off: u64) -> Self {
        BaseOff(off)
    }

    /// The eager encoding: an absolute device address, decoded against a null
    /// base. A null pointer maps to [`Self::NULL`].
    pub fn from_ptr<T>(p: *const T) -> Self {
        if p.is_null() {
            Self::NULL
        } else {
            BaseOff(p as usize as u64)
        }
    }

    /// [`Self::from_ptr`] for a mutable pointer.
    pub fn from_mut_ptr<T>(p: *mut T) -> Self {
        Self::from_ptr(p.cast_const())
    }

    /// `base + self`, the host-side twin of `base_off_ptr` in
    /// `cuda/include/base_off.cuh`. Used by tests and by the descriptor
    /// builders' debug assertions.
    pub fn resolve(self, base: *const u8) -> *const u8 {
        if self == Self::NULL {
            std::ptr::null()
        } else {
            (base as usize).wrapping_add(self.0 as usize) as *const u8
        }
    }
}

impl std::fmt::Debug for BaseOff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Self::NULL {
            f.write_str("BaseOff::NULL")
        } else {
            write!(f, "BaseOff({:#x})", self.0)
        }
    }
}

/// One matrix's **raw device pointer** plus its padded AIR width. Mirrors
/// `MainMatrixPtrs<T>` in `cuda/include/matrix.cuh`.
///
/// This is the original, pre-base+offset ABI, and it is still the ABI of the
/// single-AIR `mle.cu` entry points (`zerocheck_eval_mle` / `logup_eval_mle`),
/// which only the eager prover reaches.
///
/// # Why it was kept
///
/// Every equality test in the graph-IR port compares a graph result against an
/// eager one. If the eager path also encoded [`BaseOff`] and decoded it with
/// `base_off_ptr`, it would stop being an independent reference for the ABI it
/// is being used to check: a wrong null sentinel, or a Rust/C++ layout drift
/// in [`MainMatrixDesc`], would make both sides identically wrong and every
/// such test would still pass. Keeping one raw-pointer path means those defect
/// classes show up as a graph-vs-eager mismatch.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MainMatrixPtrs<T> {
    pub data: *const T,
    pub air_width: u32,
}

impl<T> MainMatrixPtrs<T> {
    /// The "no preprocessed trace" encoding: a genuine null pointer, not a
    /// sentinel value that has to be recognised.
    pub const ABSENT: Self = MainMatrixPtrs {
        data: std::ptr::null(),
        air_width: 0,
    };
}

/// One matrix's device base address — as a [`BaseOff`] — plus its padded AIR
/// width. Mirrors `MainMatrixDesc` in `cuda/include/matrix.cuh`.
///
/// This is what the context arrays store, so those arrays hold integers rather
/// than embedded device pointers. `MainMatrixPtrs` (the decoded pointer form)
/// still exists on the CUDA side as the device-local register type; it has no
/// Rust mirror any more because no host code produces one.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MainMatrixDesc {
    pub data: BaseOff,
    pub air_width: u32,
}

impl MainMatrixDesc {
    /// The eager encoding of one main/preprocessed matrix.
    pub fn from_ptr(data: *const EF, air_width: u32) -> Self {
        Self {
            data: BaseOff::from_ptr(data),
            air_width,
        }
    }

    /// The round-0 encoding of one main matrix.
    ///
    /// Round 0 addresses a main matrix column-major with stride `height`
    /// (`cuda/include/dag_entry.cuh`, `ENTRY_MAIN`) and never reads
    /// `air_width`. The field is left `0` rather than filled with a width whose
    /// meaning here would differ from the batched evaluators' *padded* AIR
    /// width — see [`zerocheck_ntt_eval_constraints`].
    ///
    /// Generic over the element type because round-0 main matrices are base
    /// field ([`F`]) while the batched ones are extension field ([`EF`]).
    pub fn round0<T>(data: *const T) -> Self {
        Self {
            data: BaseOff::from_ptr(data),
            air_width: 0,
        }
    }

    /// The "no preprocessed trace" encoding (`batch_mle.rs`'s null branch).
    pub const ABSENT: Self = MainMatrixDesc {
        data: BaseOff::NULL,
        air_width: 0,
    };
}

// Types for batch MLE:
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockCtx {
    pub local_block_idx_x: u32,
    /// Caution: this refers to the index within buffer of `ZerocheckCtx` or `LogupCtx`. It is
    /// hence a "local" AIR index and not the global AIR index within the proving key.
    pub air_idx: u32,
}

/// Per-AIR context for batched monomial evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MonomialAirCtx {
    pub d_headers: *const MonomialHeader,
    pub d_variables: *const PackedVar,
    pub d_lambda_combinations: *const EF, // Precomputed per-monomial
    pub num_monomials: u32,
    pub eval_ctx: EvalCoreCtx,
    pub d_eq_xi: *const EF,
    pub num_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EvalCoreCtx {
    pub d_selectors: BaseOff,
    pub d_preprocessed: MainMatrixDesc,
    /// Offset of the `MainMatrixDesc` array — descriptors, not pointers, so no
    /// level of this structure embeds a device address.
    pub d_main: BaseOff,
    pub d_public: BaseOff,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZerocheckCtx {
    pub eval_ctx: EvalCoreCtx,
    pub d_intermediates: BaseOff,
    pub num_y: u32,
    pub d_eq_xi: BaseOff,
    pub d_rules: BaseOff,
    pub rules_len: usize,
    pub d_used_nodes: BaseOff,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupCtx {
    pub eval_ctx: EvalCoreCtx,
    pub d_intermediates: BaseOff,
    pub num_y: u32,
    pub d_eq_xi: BaseOff,
    pub d_challenges: BaseOff,
    pub d_eq_3bs: BaseOff,
    pub d_rules: BaseOff,
    pub rules_len: usize,
    pub d_used_nodes: BaseOff,
    pub d_pair_idxs: BaseOff,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

/// The raw-pointer twin of [`EvalCoreCtx`] — the ORIGINAL, pre-base+offset ABI.
/// Mirrors `EvalCoreCtxRaw` in `cuda/include/eval_ctx.cuh`.
///
/// See [`MainMatrixPtrs`] for why a raw path is kept: the eager prover has to
/// stay an independent reference for the base+offset ABI it is used to check.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EvalCoreCtxRaw {
    pub d_selectors: *const EF,
    pub d_preprocessed: MainMatrixPtrs<EF>,
    pub d_main: *const MainMatrixPtrs<EF>,
    pub d_public: *const F,
}

/// The raw-pointer twin of [`ZerocheckCtx`]. Mirrors `ZerocheckCtxRaw` in
/// `cuda/src/logup_zerocheck/batch_mle.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZerocheckCtxRaw {
    pub eval_ctx: EvalCoreCtxRaw,
    pub d_intermediates: *mut EF,
    pub num_y: u32,
    pub d_eq_xi: *const EF,
    pub d_rules: *const std::ffi::c_void,
    pub rules_len: usize,
    pub d_used_nodes: *const usize,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

/// The raw-pointer twin of [`LogupCtx`]. Mirrors `LogupCtxRaw` in
/// `cuda/src/logup_zerocheck/batch_mle.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupCtxRaw {
    pub eval_ctx: EvalCoreCtxRaw,
    pub d_intermediates: *mut EF,
    pub num_y: u32,
    pub d_eq_xi: *const EF,
    pub d_challenges: *const EF,
    pub d_eq_3bs: *const EF,
    pub d_rules: *const std::ffi::c_void,
    pub rules_len: usize,
    pub d_used_nodes: *const usize,
    pub d_pair_idxs: *const u32,
    pub used_nodes_len: usize,
    pub buffer_size: u32,
}

/// Common per-AIR context for batched logup monomial evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupMonomialCommonCtx {
    pub eval_ctx: EvalCoreCtx,
    pub d_eq_xi: *const EF,
    pub bus_term_sum: EF, // Precomputed sum_i(beta[message_len_i] * (bus_idx[i]+1) * eq_3bs[i])
    pub num_y: u32,
    pub mono_blocks: u32,
}

/// The raw-pointer twin of [`MonomialAirCtx`] — the eager ABI. Only
/// `eval_ctx` differs; the other fields were already raw pointers. Mirrors
/// `MonomialAirCtxRaw` in `cuda/src/logup_zerocheck/batch_mle_monomial.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MonomialAirCtxRaw {
    pub d_headers: *const MonomialHeader,
    pub d_variables: *const PackedVar,
    pub d_lambda_combinations: *const EF,
    pub num_monomials: u32,
    pub eval_ctx: EvalCoreCtxRaw,
    pub d_eq_xi: *const EF,
    pub num_y: u32,
}

/// The raw-pointer twin of [`LogupMonomialCommonCtx`] — see
/// [`MonomialAirCtxRaw`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupMonomialCommonCtxRaw {
    pub eval_ctx: EvalCoreCtxRaw,
    pub d_eq_xi: *const EF,
    pub bus_term_sum: EF,
    pub num_y: u32,
    pub mono_blocks: u32,
}

/// Per-AIR context for batched logup monomial evaluation (numerator or denominator).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LogupMonomialCtx {
    pub d_headers: *const MonomialHeader,
    pub d_variables: *const PackedVar,
    pub d_combinations: *const EF,
    pub num_monomials: u32,
}

/// Per-AIR context for GKR input evaluation. Dispatched via a flat `BlockCtx` block list:
/// each block reads its `air_idx` and `local_block_idx_x` from the `BlockCtx` table and
/// then loads this struct for the AIR's pointers + sizing.
///
/// `task_stride` is the per-AIR thread count = `num_blocks_x * THREADS_PER_BLOCK`. It also
/// serves as the column stride for `d_intermediates`, which must hold at least
/// `task_stride * buffer_size` `EF` elements.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GkrInputCtx {
    pub d_fracs: *mut Frac<EF>,
    pub d_preprocessed: *const F,
    pub d_main: *const u64,
    pub d_public_values: *const F,
    pub d_challenges: *const EF,
    pub d_intermediates: *mut EF,
    pub d_rules: *const std::ffi::c_void,
    pub d_used_nodes: *const usize,
    pub d_pair_idxs: *const u32,
    pub used_nodes_len: usize,
    pub height: u32,
    pub task_stride: u32,
    pub num_rows_per_tile: u32,
}

// end of types for batch MLE

extern "C" {
    // gkr.cu
    fn _frac_build_tree_layer(
        layer: *mut Frac<EF>,
        layer_size: usize,
        real_len: usize,
        logical_len: usize,
        revert: bool,
        alpha: EF,
        apply_alpha: bool,
        stream: cudaStream_t,
    ) -> i32;

    /// Fused two-layer tree build. Applies layers i and i+1 in one kernel pass,
    /// keeping intermediate right-half nodes for revert operations.
    /// `half_i1` = N >> (i+2), where i is the first of the two layers.
    fn _frac_build_tree_two_layers(
        layer: *mut Frac<EF>,
        half_i1: usize,
        real_len: usize,
        logical_len: usize,
        alpha: EF,
        stream: cudaStream_t,
    ) -> i32;

    pub fn _frac_compute_round_temp_buffer_size(stride: u32) -> u32;

    fn _frac_compute_round(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        pq_buffer: *const Frac<EF>,
        num_x: usize,
        eq_low_cap: usize,
        lambda: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Device-challenge variant of [`_frac_compute_round`]: `lambda` is read on-device
    /// from a device pointer (graph-IR path).
    fn _frac_compute_round_dev_challenge(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        pq_buffer: *const Frac<EF>,
        num_x: usize,
        eq_low_cap: usize,
        lambda_dev: *const EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Fused compute round + tree layer revert. Combines frac_build_tree_layer(revert=true)
    /// with compute_round for the first inner round. Modifies layer in-place for revert.
    fn _frac_compute_round_and_revert(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        layer: *mut Frac<EF>,
        num_x: usize,
        real_len: usize,
        logical_len: usize,
        eq_low_cap: usize,
        lambda: EF,
        alpha: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Device-challenge variant of [`_frac_compute_round_and_revert`]: `lambda` is read
    /// on-device from a device pointer (graph-IR path). `alpha` stays a host value.
    fn _frac_compute_round_and_revert_dev_challenge(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        layer: *mut Frac<EF>,
        num_x: usize,
        real_len: usize,
        logical_len: usize,
        eq_low_cap: usize,
        lambda_dev: *const EF,
        alpha: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    fn _frac_fold_fpext_columns(
        src: *const Frac<EF>,
        dst: *mut Frac<EF>,
        size: usize,
        real_len: usize,
        logical_len: usize,
        r: EF,
        alpha: EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Device-challenge variant of [`_frac_fold_fpext_columns`]: `r` is read on-device
    /// from a device pointer (graph-IR path). `alpha` stays a host value.
    fn _frac_fold_fpext_columns_dev_challenge(
        src: *const Frac<EF>,
        dst: *mut Frac<EF>,
        size: usize,
        real_len: usize,
        logical_len: usize,
        r_dev: *const EF,
        alpha: EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Fused compute round + fold (out-of-place). Reads from pre-fold src_pq_buffer (size
    /// src_pq_size), computes sumcheck sums, and writes folded output to dst_pq_buffer (size
    /// src_pq_size/2). IMPORTANT: src_pq_buffer and dst_pq_buffer must NOT alias.
    fn _frac_compute_round_and_fold(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        src_pq_buffer: *const Frac<EF>,
        dst_pq_buffer: *mut Frac<EF>,
        src_pq_size: usize,
        real_len: usize,
        logical_len: usize,
        eq_low_cap: usize,
        lambda: EF,
        r_prev: EF,
        alpha: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Device-challenge variant of [`_frac_compute_round_and_fold`]: `lambda` and `r_prev`
    /// are read on-device from device pointers (graph-IR path). `alpha` stays a host value.
    fn _frac_compute_round_and_fold_dev_challenge(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        src_pq_buffer: *const Frac<EF>,
        dst_pq_buffer: *mut Frac<EF>,
        src_pq_size: usize,
        real_len: usize,
        logical_len: usize,
        eq_low_cap: usize,
        lambda_dev: *const EF,
        r_prev_dev: *const EF,
        alpha: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Fused compute round + fold (in-place). Reads from pre-fold pq_buffer (size src_pq_size),
    /// computes sumcheck sums, and writes folded output to the same buffer (first src_pq_size/2
    /// elements).
    fn _frac_compute_round_and_fold_inplace(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        pq_buffer: *mut Frac<EF>,
        src_pq_size: usize,
        real_len: usize,
        logical_len: usize,
        dst_real_len: usize,
        dst_logical_len: usize,
        eq_low_cap: usize,
        lambda: EF,
        r_prev: EF,
        alpha: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Device-challenge variant of [`_frac_compute_round_and_fold_inplace`]: `lambda` and
    /// `r_prev` are read on-device from device pointers (graph-IR path). `alpha` stays a
    /// host value.
    fn _frac_compute_round_and_fold_inplace_dev_challenge(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        pq_buffer: *mut Frac<EF>,
        src_pq_size: usize,
        real_len: usize,
        logical_len: usize,
        dst_real_len: usize,
        dst_logical_len: usize,
        eq_low_cap: usize,
        lambda_dev: *const EF,
        r_prev_dev: *const EF,
        alpha: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    fn _frac_precompute_m_build(
        pq: *const Frac<EF>,
        real_len: usize,
        logical_len: usize,
        rem_n: usize,
        w: usize,
        lambda: EF,
        r_prev: EF,
        alpha: EF,
        inline_fold: bool,
        eq_tail_low: *const EF,
        eq_tail_high: *const EF,
        eq_tail_low_cap: usize,
        tail_tile: usize,
        partial_out: *mut EF,
        partial_len: usize,
        m_total: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    /// Device-challenge variant of [`_frac_precompute_m_build`]: `lambda` and `r_prev`
    /// are read on-device from `lambda_dev` / `r_prev_dev` (graph-IR path).
    fn _frac_precompute_m_build_dev_challenge(
        pq: *const Frac<EF>,
        real_len: usize,
        logical_len: usize,
        rem_n: usize,
        w: usize,
        lambda_dev: *const EF,
        r_prev_dev: *const EF,
        alpha: EF,
        inline_fold: bool,
        eq_tail_low: *const EF,
        eq_tail_high: *const EF,
        eq_tail_low_cap: usize,
        tail_tile: usize,
        partial_out: *mut EF,
        partial_len: usize,
        m_total: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    fn _frac_precompute_m_eval_round(
        m_total: *const EF,
        w: usize,
        t: usize,
        eq_r_prefix: *const EF,
        eq_suffix: *const EF,
        out: *mut EF,
        stream: cudaStream_t,
    ) -> i32;

    fn _frac_multifold(
        src: *const Frac<EF>,
        dst: *mut Frac<EF>,
        real_len: usize,
        logical_len: usize,
        rem_n: usize,
        w: usize,
        alpha: EF,
        eq_r_window: *const EF,
        stream: cudaStream_t,
    ) -> i32;

    fn _frac_add_alpha(
        data: *mut std::ffi::c_void,
        len: usize,
        alpha: EF,
        stream: cudaStream_t,
    ) -> i32;

    fn _frac_vector_scalar_multiply_ext_fp(
        frac_vec: *mut Frac<EF>,
        scalar: F,
        length: u32,
        stream: cudaStream_t,
    ) -> i32;

    // utils.cu
    fn _fold_ple_from_evals(
        input_matrix: *const F,
        output_matrix: *mut EF,
        omega_skip_pows: *const F,
        inv_lagrange_denoms: *const EF,
        height: u32,
        width: u32,
        l_skip: u32,
        new_height: u32,
        rotate: bool,
        stream: cudaStream_t,
    ) -> i32;

    fn _interpolate_columns(
        interpolated: *mut EF,
        columns: *const *const EF,
        s_deg: usize,
        num_y: usize,
        num_columns: usize,
        stream: cudaStream_t,
    ) -> i32;

    fn _frac_matrix_vertically_repeat(
        out: *mut Frac<EF>,
        input: *const Frac<EF>,
        width: u32,
        lifted_height: u32,
        height: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _frac_matrix_vertically_repeat_ext(
        out_numerators: *mut EF,
        out_denominators: *mut EF,
        in_numerators: *const EF,
        in_denominators: *const EF,
        width: u32,
        lifted_height: u32,
        height: u32,
        stream: cudaStream_t,
    ) -> i32;

    // gkr_input.cu
    fn _logup_gkr_input_eval(
        d_block_ctxs: *const BlockCtx,
        d_ctxs: *const GkrInputCtx,
        num_blocks: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    // logup_round0.cu
    pub fn _logup_r0_temp_sums_buffer_size(
        buffer_size: u32,
        skip_domain: u32,
        num_x: u32,
        num_cosets: u32,
        max_temp_bytes: usize,
    ) -> usize;

    pub fn _logup_r0_intermediates_buffer_size(
        buffer_size: u32,
        skip_domain: u32,
        num_x: u32,
        num_cosets: u32,
        max_temp_bytes: usize,
    ) -> usize;

    fn _logup_bary_eval_interactions_round0(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        selectors_cube: *const F,
        preprocessed: *const F,
        main_descs: *const MainMatrixDesc,
        pool_base: *const u8,
        eq_cube: *const EF,
        public_values: *const F,
        numer_weights: *const EF,
        denom_weights: *const EF,
        denom_sum_init: EF,
        d_rules: *const std::ffi::c_void,
        rules_len: usize,
        buffer_size: u32,
        d_intermediates: *mut F,
        skip_domain: u32,
        num_x: u32,
        height: u32,
        num_cosets: u32,
        g_shift: F,
        max_temp_bytes: usize,
        stream: cudaStream_t,
    ) -> i32;

    // zerocheck_round0.cu
    pub fn _zerocheck_r0_temp_sums_buffer_size(
        buffer_size: u32,
        skip_domain: u32,
        num_x: u32,
        num_cosets: u32,
        max_temp_bytes: usize,
    ) -> usize;

    pub fn _zerocheck_r0_intermediates_buffer_size(
        buffer_size: u32,
        skip_domain: u32,
        num_x: u32,
        num_cosets: u32,
        max_temp_bytes: usize,
    ) -> usize;

    fn _zerocheck_ntt_eval_constraints(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        selectors_cube: *const F,
        preprocessed: *const F,
        main_descs: *const MainMatrixDesc,
        pool_base: *const u8,
        eq_cube: *const EF,
        d_lambda_pows: *const EF,
        public_values: *const F,
        d_rules: *const std::ffi::c_void,
        rules_len: usize,
        d_used_nodes: *const usize,
        used_nodes_len: usize,
        lambda_len: usize,
        buffer_size: u32,
        d_intermediates: *mut F,
        skip_domain: u32,
        num_x: u32,
        height: u32,
        num_cosets: u32,
        g_shift: F,
        max_temp_bytes: usize,
        stream: cudaStream_t,
    ) -> i32;

    fn _fold_selectors_round0(
        out: *mut EF,
        input: *const F,
        is_first: EF,
        is_last: EF,
        num_x: u32,
        stream: cudaStream_t,
    ) -> i32;

    // mle.cu
    pub fn _zerocheck_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize;

    pub fn _zerocheck_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    fn _zerocheck_eval_mle(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        eq_xi: *const EF,
        selectors: *const EF,
        preprocessed: MainMatrixPtrs<EF>,
        main: *const MainMatrixPtrs<EF>,
        lambda_pows: *const EF,
        public_values: *const F,
        rules: *const std::ffi::c_void,
        rules_len: usize,
        used_nodes: *const usize,
        used_nodes_len: usize,
        lambda_len: usize,
        buffer_size: u32,
        intermediates: *mut EF,
        num_y: u32,
        num_x: u32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn _logup_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize;
    pub fn _logup_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize;

    fn _logup_eval_mle(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        eq_xi: *const EF,
        selectors: *const EF,
        preprocessed: MainMatrixPtrs<EF>,
        main: *const MainMatrixPtrs<EF>,
        challenges: *const EF,
        eq_3bs: *const EF,
        public_values: *const F,
        rules: *const std::ffi::c_void,
        used_nodes: *const usize,
        pair_idxs: *const u32,
        used_nodes_len: usize,
        buffer_size: u32,
        intermediates: *mut EF,
        num_y: u32,
        num_x: u32,
        stream: cudaStream_t,
    ) -> i32;

    // batch_mle.cu (batch kernels always use global intermediates when buffer_size > 0)
    pub fn _zerocheck_batch_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    pub fn _logup_batch_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    fn _zerocheck_batch_eval_mle(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        zc_ctxs: *const ZerocheckCtx,
        pool_base: *const u8,
        air_block_offsets: *const u32,
        lambda_pows: *const EF,
        lambda_len: usize,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _logup_batch_eval_mle(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        block_ctxs: *const BlockCtx,
        logup_ctxs: *const LogupCtx,
        pool_base: *const u8,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    /// The eager, raw-pointer entry points. Same evaluator, different context
    /// ABI: no [`BaseOff`], no pool base, no null sentinel. See
    /// [`MainMatrixPtrs`] for why this path is kept separate.
    fn _zerocheck_batch_eval_mle_raw(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        zc_ctxs: *const ZerocheckCtxRaw,
        air_block_offsets: *const u32,
        lambda_pows: *const EF,
        lambda_len: usize,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _logup_batch_eval_mle_raw(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        block_ctxs: *const BlockCtx,
        logup_ctxs: *const LogupCtxRaw,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    /// `sizeof` of the C++ ctx ABI, for the layout static-asserts in
    /// [`assert_ctx_abi_matches_cuda`].
    pub fn _main_matrix_desc_size() -> usize;
    pub fn _eval_core_ctx_size() -> usize;
    pub fn _zerocheck_ctx_size() -> usize;
    pub fn _logup_ctx_size() -> usize;

    /// Field offsets and the null sentinel of the base+offset ABI, which the
    /// round-0 entry points decode against.
    pub fn _base_off_size() -> usize;
    pub fn _base_off_null() -> u64;
    pub fn _main_matrix_desc_data_offset() -> usize;
    pub fn _main_matrix_desc_air_width_offset() -> usize;

    /// `sizeof` of the raw-pointer (eager) ctx ABI — a second ABI, so a
    /// second guard.
    pub fn _main_matrix_ptrs_size() -> usize;
    pub fn _eval_core_ctx_raw_size() -> usize;
    pub fn _zerocheck_ctx_raw_size() -> usize;
    pub fn _logup_ctx_raw_size() -> usize;

    fn _zerocheck_monomial_batched(
        tmp_sums: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        air_ctxs: *const MonomialAirCtx,
        pool_base: *const u8,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _zerocheck_monomial_par_y_batched(
        tmp_sums: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        air_ctxs: *const MonomialAirCtx,
        pool_base: *const u8,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        chunk_size: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _precompute_lambda_combinations(
        out: *mut EF,
        headers: *const MonomialHeader,
        lambda_terms: *const LambdaTerm<F>,
        lambda_pows: *const EF,
        num_monomials: u32,
        stream: cudaStream_t,
    ) -> i32;

    // Logup monomial kernels
    fn _precompute_logup_numer_combinations(
        out: *mut EF,
        headers: *const MonomialHeader,
        terms: *const InteractionMonomialTerm<F>,
        eq_3bs: *const EF,
        num_monomials: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _precompute_logup_denom_combinations(
        out: *mut EF,
        headers: *const MonomialHeader,
        terms: *const InteractionMonomialTerm<F>,
        beta_pows: *const EF,
        eq_3bs: *const EF,
        num_monomials: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _logup_monomial_batched(
        tmp_sums: *mut Frac<EF>,
        output: *mut Frac<EF>,
        block_ctxs: *const BlockCtx,
        common_ctxs: *const LogupMonomialCommonCtx,
        numer_ctxs: *const LogupMonomialCtx,
        denom_ctxs: *const LogupMonomialCtx,
        pool_base: *const u8,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    /// The eager, raw-pointer monomial entry points.
    fn _zerocheck_monomial_batched_raw(
        tmp_sums: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        air_ctxs: *const MonomialAirCtxRaw,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _zerocheck_monomial_par_y_batched_raw(
        tmp_sums: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        air_ctxs: *const MonomialAirCtxRaw,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        chunk_size: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn _logup_monomial_batched_raw(
        tmp_sums: *mut Frac<EF>,
        output: *mut Frac<EF>,
        block_ctxs: *const BlockCtx,
        common_ctxs: *const LogupMonomialCommonCtxRaw,
        numer_ctxs: *const LogupMonomialCtx,
        denom_ctxs: *const LogupMonomialCtx,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
        stream: cudaStream_t,
    ) -> i32;
}

pub unsafe fn interpolate_columns_gpu(
    interpolated: &DeviceBuffer<EF>,
    columns: &DeviceBuffer<*const EF>,
    s_deg: usize,
    num_y: usize,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_interpolate_columns(
        interpolated.as_mut_ptr(),
        columns.as_ptr(),
        s_deg,
        num_y,
        columns.len(),
        stream,
    ))
}

pub unsafe fn frac_build_tree_layer(
    layer: &mut DeviceBuffer<Frac<EF>>,
    layer_size: usize,
    logical_len: usize,
    revert: bool,
    alpha: EF,
    apply_alpha: bool,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(layer.len() >= layer_size || layer_size == logical_len);
    CudaError::from_result(_frac_build_tree_layer(
        layer.as_mut_ptr(),
        layer_size,
        layer.len(),
        logical_len,
        revert,
        alpha,
        apply_alpha,
        stream,
    ))
}

/// Fused two-layer tree build kernel.
/// `half_i1` = N >> (i+2), where i is the first of the two layers being fused.
pub unsafe fn frac_build_tree_two_layers(
    layer: &mut DeviceBuffer<Frac<EF>>,
    half_i1: usize,
    logical_len: usize,
    alpha: EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_build_tree_two_layers(
        layer.as_mut_ptr(),
        half_i1,
        layer.len(),
        logical_len,
        alpha,
        stream,
    ))
}

// `eq_xi` will not store evaluations for the first hypercube coordinate because the prover factors
// out the first eq term.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round(
    eq_xi: &SqrtEqLayers,
    pq_buffer: &DeviceBuffer<Frac<EF>>,
    num_x: usize,
    lambda: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    debug_assert!(pq_buffer.len() >= 2 * num_x);
    #[cfg(debug_assertions)]
    {
        let len = tmp_block_sums.len();
        let required = _frac_compute_round_temp_buffer_size(num_x.try_into().unwrap());
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    CudaError::from_result(_frac_compute_round(
        eq_xi.low.get_ptr(low_n),
        eq_xi.high.get_ptr(high_n),
        pq_buffer.as_ptr(),
        num_x,
        1 << low_n,
        lambda,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
        stream,
    ))
}

/// Device-challenge variant of [`frac_compute_round`], taking raw device pointers:
/// `lambda` is read on-device from `lambda_dev` (graph-IR path, where the challenge and
/// eq layers live in graph-managed buffers). `eq_low_cap` = 2^low_n and
/// `num_x` = 2^(low_n + high_n + 1) for the eq layer pointers passed in.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_dev_challenge(
    eq_xi_low: *const EF,
    eq_xi_high: *const EF,
    pq_buffer: *const Frac<EF>,
    num_x: usize,
    eq_low_cap: usize,
    lambda_dev: *const EF,
    out_device: *mut EF,
    tmp_block_sums: *mut EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(eq_low_cap.is_power_of_two());
    debug_assert!(num_x.is_power_of_two());
    debug_assert!(!lambda_dev.is_null());
    CudaError::from_result(_frac_compute_round_dev_challenge(
        eq_xi_low,
        eq_xi_high,
        pq_buffer,
        num_x,
        eq_low_cap,
        lambda_dev,
        out_device,
        tmp_block_sums,
        stream,
    ))
}

/// Fused compute round + tree layer revert kernel.
///
/// Combines `frac_build_tree_layer(revert=true)` with `compute_round` for the first inner round.
/// The revert operation modifies `layer` in-place: `layer[i] = layer[i] - layer[i + half]` for `i <
/// half`.
///
/// This eliminates one kernel launch per outer round.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_revert(
    eq_xi: &SqrtEqLayers,
    layer: &mut DeviceBuffer<Frac<EF>>,
    num_x: usize,
    logical_len: usize,
    lambda: EF,
    alpha: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    #[cfg(debug_assertions)]
    {
        let len = tmp_block_sums.len();
        let required = _frac_compute_round_temp_buffer_size(num_x.try_into().unwrap());
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
        assert!(
            layer.len() >= 2 * num_x || 2 * num_x == logical_len,
            "layer too small for pq_size"
        );
    }
    CudaError::from_result(_frac_compute_round_and_revert(
        eq_xi.low.get_ptr(low_n),
        eq_xi.high.get_ptr(high_n),
        layer.as_mut_ptr(),
        num_x,
        layer.len(),
        logical_len,
        1 << low_n,
        lambda,
        alpha,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
        stream,
    ))
}

/// Device-challenge variant of [`frac_compute_round_and_revert`], taking raw device
/// pointers: `lambda` is read on-device from `lambda_dev` (graph-IR path). `alpha` stays
/// a host value. `real_len` is the physical length of `layer`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_revert_dev_challenge(
    eq_xi_low: *const EF,
    eq_xi_high: *const EF,
    layer: *mut Frac<EF>,
    num_x: usize,
    real_len: usize,
    logical_len: usize,
    eq_low_cap: usize,
    lambda_dev: *const EF,
    alpha: EF,
    out_device: *mut EF,
    tmp_block_sums: *mut EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(eq_low_cap.is_power_of_two());
    debug_assert!(num_x.is_power_of_two());
    debug_assert!(!lambda_dev.is_null());
    debug_assert!(
        real_len >= 2 * num_x || 2 * num_x == logical_len,
        "layer too small for pq_size"
    );
    CudaError::from_result(_frac_compute_round_and_revert_dev_challenge(
        eq_xi_low,
        eq_xi_high,
        layer,
        num_x,
        real_len,
        logical_len,
        eq_low_cap,
        lambda_dev,
        alpha,
        out_device,
        tmp_block_sums,
        stream,
    ))
}

/// Folds `Frac<EF>` buffer. Pairs (idx, idx+quarter) and (idx+half, idx+3*quarter),
/// writes results to dst[idx] and dst[idx+quarter]. Output size is `size / 2`.
/// Dense folds are safe for src == dst because each thread handles disjoint indices.
/// Compact virtual folds must use an out-of-place destination because virtual reads can
/// recover source values from physical slots in the output range.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fold_ef_frac_columns(
    src: &DeviceBuffer<Frac<EF>>,
    dst: &mut DeviceBuffer<Frac<EF>>,
    size: usize,
    real_len: usize,
    logical_len: usize,
    r: EF,
    alpha: EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(
        src.len() >= real_len,
        "compact buffer must hold at least real_len entries"
    );
    debug_assert!(real_len <= logical_len);
    debug_assert!(dst.len() >= size / 2);
    CudaError::from_result(_frac_fold_fpext_columns(
        src.as_ptr(),
        dst.as_mut_ptr(),
        size,
        real_len,
        logical_len,
        r,
        alpha,
        stream,
    ))
}

/// Device-challenge variant of [`fold_ef_frac_columns`], taking raw device pointers:
/// `r` is read on-device from `r_dev` (graph-IR path). `alpha` stays a host value.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fold_ef_frac_columns_dev_challenge(
    src: *const Frac<EF>,
    dst: *mut Frac<EF>,
    size: usize,
    real_len: usize,
    logical_len: usize,
    r_dev: *const EF,
    alpha: EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(real_len <= logical_len);
    debug_assert!(!r_dev.is_null());
    CudaError::from_result(_frac_fold_fpext_columns_dev_challenge(
        src,
        dst,
        size,
        real_len,
        logical_len,
        r_dev,
        alpha,
        stream,
    ))
}

/// In-place fold. See [`fold_ef_frac_columns`] for details.
pub unsafe fn fold_ef_frac_columns_inplace(
    buffer: &mut DeviceBuffer<Frac<EF>>,
    size: usize,
    real_len: usize,
    logical_len: usize,
    r: EF,
    alpha: EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(
        buffer.len() >= real_len,
        "compact buffer must hold at least real_len entries"
    );
    debug_assert!(real_len <= logical_len);
    debug_assert_eq!(
        real_len, logical_len,
        "virtual compact folds must use an out-of-place destination"
    );
    let ptr = buffer.as_mut_ptr();
    CudaError::from_result(_frac_fold_fpext_columns(
        ptr,
        ptr,
        size,
        real_len,
        logical_len,
        r,
        alpha,
        stream,
    ))
}

/// Device-challenge variant of [`fold_ef_frac_columns_inplace`], taking raw device
/// pointers: `r` is read on-device from `r_dev` (graph-IR path). `alpha` stays a host
/// value.
pub unsafe fn fold_ef_frac_columns_inplace_dev_challenge(
    buffer: *mut Frac<EF>,
    size: usize,
    real_len: usize,
    logical_len: usize,
    r_dev: *const EF,
    alpha: EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert_eq!(
        real_len, logical_len,
        "virtual compact folds must use an out-of-place destination"
    );
    debug_assert!(!r_dev.is_null());
    CudaError::from_result(_frac_fold_fpext_columns_dev_challenge(
        buffer,
        buffer,
        size,
        real_len,
        logical_len,
        r_dev,
        alpha,
        stream,
    ))
}

/// Fused compute round + fold kernel.
///
/// Reads from pre-fold `src_pq_buffer` (size `src_pq_size`), performs fold-on-the-fly using
/// `r_prev`, computes s' polynomial evaluations (degree 2), and writes folded output to
/// `dst_pq_buffer` (size `src_pq_size/2`).
///
/// This fuses the fold operation into the next round's compute, eliminating one kernel launch per
/// inner round and reducing memory traffic.
///
/// The eq_xi layers should have max_n = log2(src_pq_size / 4) = log2(post-fold num_x).
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_fold(
    eq_xi: &SqrtEqLayers,
    src_pq_buffer: &DeviceBuffer<Frac<EF>>,
    dst_pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    src_pq_size: usize,
    real_len: usize,
    logical_len: usize,
    lambda: EF,
    r_prev: EF,
    alpha: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    // Post-fold: num_x = src_pq_size / 4
    let num_x = src_pq_size >> 2;
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    #[cfg(debug_assertions)]
    {
        assert!(src_pq_size > 2, "src_pq_size must be > 2");
        let pq_size = src_pq_size >> 1;
        assert!(num_x > 0, "num_x must be > 0");
        assert!(
            src_pq_buffer.len() >= src_pq_size || src_pq_size == logical_len,
            "src_pq_buffer too small: {} < {}",
            src_pq_buffer.len(),
            src_pq_size
        );
        assert!(
            dst_pq_buffer.len() >= pq_size,
            "dst_pq_buffer too small: {} < {}",
            dst_pq_buffer.len(),
            pq_size
        );
        let len = tmp_block_sums.len();
        let required = _frac_compute_round_temp_buffer_size(num_x as u32);
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    CudaError::from_result(_frac_compute_round_and_fold(
        eq_xi.low.get_ptr(low_n),
        eq_xi.high.get_ptr(high_n),
        src_pq_buffer.as_ptr(),
        dst_pq_buffer.as_mut_ptr(),
        src_pq_size,
        real_len,
        logical_len,
        1 << low_n,
        lambda,
        r_prev,
        alpha,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
        stream,
    ))
}

/// Device-challenge variant of [`frac_compute_round_and_fold`], taking raw device
/// pointers: `lambda` and `r_prev` are read on-device from `lambda_dev` / `r_prev_dev`
/// (graph-IR path). `alpha` stays a host value. `eq_low_cap` = 2^low_n for the eq layer
/// pointers passed in, which must have max_n = log2(src_pq_size / 4).
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_fold_dev_challenge(
    eq_xi_low: *const EF,
    eq_xi_high: *const EF,
    src_pq_buffer: *const Frac<EF>,
    dst_pq_buffer: *mut Frac<EF>,
    src_pq_size: usize,
    real_len: usize,
    logical_len: usize,
    eq_low_cap: usize,
    lambda_dev: *const EF,
    r_prev_dev: *const EF,
    alpha: EF,
    out_device: *mut EF,
    tmp_block_sums: *mut EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(eq_low_cap.is_power_of_two());
    debug_assert!(src_pq_size > 2, "src_pq_size must be > 2");
    debug_assert!(!lambda_dev.is_null());
    debug_assert!(!r_prev_dev.is_null());
    CudaError::from_result(_frac_compute_round_and_fold_dev_challenge(
        eq_xi_low,
        eq_xi_high,
        src_pq_buffer,
        dst_pq_buffer,
        src_pq_size,
        real_len,
        logical_len,
        eq_low_cap,
        lambda_dev,
        r_prev_dev,
        alpha,
        out_device,
        tmp_block_sums,
        stream,
    ))
}

/// In-place fused compute round + fold kernel. See [`frac_compute_round_and_fold`] for details.
///
/// Uses a dedicated in-place kernel that doesn't have `__restrict__` on the pq pointer,
/// avoiding undefined behavior from aliased restrict pointers.
///
/// **IN-PLACE SAFETY:** Each thread writes only to indices it reads from in the first half of the
/// buffer, so there are no cross-thread conflicts. The second half is read-only.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_fold_inplace(
    eq_xi: &SqrtEqLayers,
    pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    src_pq_size: usize,
    real_len: usize,
    logical_len: usize,
    dst_real_len: usize,
    dst_logical_len: usize,
    lambda: EF,
    r_prev: EF,
    alpha: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    // Post-fold: num_x = src_pq_size / 4
    let num_x = src_pq_size >> 2;
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    #[cfg(debug_assertions)]
    {
        assert!(src_pq_size > 2, "src_pq_size must be > 2");
        assert!(num_x > 0, "num_x must be > 0");
        assert!(
            pq_buffer.len() >= src_pq_size || src_pq_size == logical_len,
            "pq_buffer too small: {} < {}",
            pq_buffer.len(),
            src_pq_size
        );
        let len = tmp_block_sums.len();
        let required = _frac_compute_round_temp_buffer_size(num_x as u32);
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    CudaError::from_result(_frac_compute_round_and_fold_inplace(
        eq_xi.low.get_ptr(low_n),
        eq_xi.high.get_ptr(high_n),
        pq_buffer.as_mut_ptr(),
        src_pq_size,
        real_len,
        logical_len,
        dst_real_len,
        dst_logical_len,
        1 << low_n,
        lambda,
        r_prev,
        alpha,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
        stream,
    ))
}

/// Device-challenge variant of [`frac_compute_round_and_fold_inplace`], taking raw
/// device pointers: `lambda` and `r_prev` are read on-device from `lambda_dev` /
/// `r_prev_dev` (graph-IR path). `alpha` stays a host value. `eq_low_cap` = 2^low_n for
/// the eq layer pointers passed in, which must have max_n = log2(src_pq_size / 4).
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_fold_inplace_dev_challenge(
    eq_xi_low: *const EF,
    eq_xi_high: *const EF,
    pq_buffer: *mut Frac<EF>,
    src_pq_size: usize,
    real_len: usize,
    logical_len: usize,
    dst_real_len: usize,
    dst_logical_len: usize,
    eq_low_cap: usize,
    lambda_dev: *const EF,
    r_prev_dev: *const EF,
    alpha: EF,
    out_device: *mut EF,
    tmp_block_sums: *mut EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(eq_low_cap.is_power_of_two());
    debug_assert!(src_pq_size > 2, "src_pq_size must be > 2");
    debug_assert!(!lambda_dev.is_null());
    debug_assert!(!r_prev_dev.is_null());
    CudaError::from_result(_frac_compute_round_and_fold_inplace_dev_challenge(
        eq_xi_low,
        eq_xi_high,
        pq_buffer,
        src_pq_size,
        real_len,
        logical_len,
        dst_real_len,
        dst_logical_len,
        eq_low_cap,
        lambda_dev,
        r_prev_dev,
        alpha,
        out_device,
        tmp_block_sums,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_precompute_m_build_raw(
    pq: *const Frac<EF>,
    real_len: usize,
    logical_len: usize,
    rem_n: usize,
    w: usize,
    lambda: EF,
    r_prev: EF,
    alpha: EF,
    inline_fold: bool,
    eq_tail_low: *const EF,
    eq_tail_high: *const EF,
    eq_tail_low_cap: usize,
    tail_tile: usize,
    partial_out: *mut EF,
    partial_len: usize,
    m_total: *mut EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(rem_n > 0);
    debug_assert!(w > 0 && w <= rem_n);
    debug_assert!(eq_tail_low_cap.is_power_of_two());
    debug_assert!(tail_tile > 0);
    CudaError::from_result(_frac_precompute_m_build(
        pq,
        real_len,
        logical_len,
        rem_n,
        w,
        lambda,
        r_prev,
        alpha,
        inline_fold,
        eq_tail_low,
        eq_tail_high,
        eq_tail_low_cap,
        tail_tile,
        partial_out,
        partial_len,
        m_total,
        stream,
    ))
}

/// Device-challenge variant of [`frac_precompute_m_build_raw`]: `lambda` and
/// `r_prev` are read on-device (graph-IR path). `alpha` stays a host value.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_precompute_m_build_dev_challenge_raw(
    pq: *const Frac<EF>,
    real_len: usize,
    logical_len: usize,
    rem_n: usize,
    w: usize,
    lambda_dev: *const EF,
    r_prev_dev: *const EF,
    alpha: EF,
    inline_fold: bool,
    eq_tail_low: *const EF,
    eq_tail_high: *const EF,
    eq_tail_low_cap: usize,
    tail_tile: usize,
    partial_out: *mut EF,
    partial_len: usize,
    m_total: *mut EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(rem_n > 0);
    debug_assert!(w > 0 && w <= rem_n);
    debug_assert!(eq_tail_low_cap.is_power_of_two());
    debug_assert!(tail_tile > 0);
    debug_assert!(!lambda_dev.is_null());
    debug_assert!(!r_prev_dev.is_null());
    CudaError::from_result(_frac_precompute_m_build_dev_challenge(
        pq,
        real_len,
        logical_len,
        rem_n,
        w,
        lambda_dev,
        r_prev_dev,
        alpha,
        inline_fold,
        eq_tail_low,
        eq_tail_high,
        eq_tail_low_cap,
        tail_tile,
        partial_out,
        partial_len,
        m_total,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_precompute_m_eval_round_raw(
    m_total: *const EF,
    w: usize,
    t: usize,
    eq_r_prefix: *const EF,
    eq_suffix: *const EF,
    out: *mut EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(w > 0);
    debug_assert!(t < w);
    CudaError::from_result(_frac_precompute_m_eval_round(
        m_total,
        w,
        t,
        eq_r_prefix,
        eq_suffix,
        out,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_multifold_raw(
    src: *const Frac<EF>,
    dst: *mut Frac<EF>,
    real_len: usize,
    logical_len: usize,
    rem_n: usize,
    w: usize,
    alpha: EF,
    eq_r_window: *const EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(rem_n > 0);
    debug_assert!(w > 0 && w <= rem_n);
    CudaError::from_result(_frac_multifold(
        src,
        dst,
        real_len,
        logical_len,
        rem_n,
        w,
        alpha,
        eq_r_window,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn fold_ple_from_evals(
    input_matrix: &DeviceBuffer<F>,
    output_matrix: *mut EF,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<EF>,
    height: u32,
    width: u32,
    l_skip: u32,
    new_height: u32,
    rotate: bool,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_fold_ple_from_evals(
        input_matrix.as_ptr(),
        output_matrix,
        omega_skip_pows.as_ptr(),
        inv_lagrange_denoms.as_ptr(),
        height,
        width,
        l_skip,
        new_height,
        rotate,
        stream,
    ))
}

/// GKR input eval that processes multiple AIRs in a single kernel launch via a flat block list.
///
/// Each entry of `d_block_ctxs` describes one block in the launch (which AIR it serves and its
/// sub-block index within that AIR's allotment). Per-AIR sizing lives in `d_ctxs[air_idx]`.
///
/// # Safety
/// - `d_block_ctxs` must contain exactly `num_blocks` valid `BlockCtx` entries.
/// - `d_ctxs` must contain a valid `GkrInputCtx` for every distinct `air_idx` referenced by
///   `d_block_ctxs`. All referenced device pointers in each ctx must be valid.
pub unsafe fn logup_gkr_input_eval(
    d_block_ctxs: &DeviceBuffer<BlockCtx>,
    d_ctxs: &DeviceBuffer<GkrInputCtx>,
    num_blocks: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_gkr_input_eval(
        d_block_ctxs.as_ptr(),
        d_ctxs.as_ptr(),
        num_blocks,
        threads_per_block,
        stream,
    ))
}

pub unsafe fn frac_add_alpha(
    data: &DeviceBuffer<Frac<EF>>,
    alpha: EF,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_add_alpha(
        data.as_mut_raw_ptr(),
        data.len(),
        alpha,
        stream,
    ))
}

/// # Safety
/// - `buffer_size` does not refer to the capacity of `intermediates`. It refers to "how many DAG
///   nodes per row need to be buffered". The capacity is a multiple of `buffer_size` which is
///   runtime calculated based on `buffer_size`.
/// - `eq_cube` must be a pointer to device buffer with at least `num_x` elements representing
///   evaluations on hypercube.
/// - `main_descs` must point to a device array of at least `n_main_parts` [`MainMatrixDesc`], where
///   `n_main_parts` is one past the largest `part` index any `ENTRY_MAIN` rule reads. Each
///   descriptor's `data` is a [`BaseOff`] decoded against `pool_base` — the base+offset ABI of
///   `cuda/include/base_off.cuh`. The eager path passes absolute addresses with `pool_base =
///   std::ptr::null()`; the graph-IR path passes pool offsets with the exe's pool base.
///   [`BaseOff::NULL`] is the only "absent" encoding — offset `0` is a valid pool offset.
/// - Round 0 does **not** read `MainMatrixDesc::air_width`: it strides a main matrix column-major
///   by `height` (`cuda/include/dag_entry.cuh`, `ENTRY_MAIN`). The field is carried only so this
///   table has the same layout as the batched evaluators' descriptor arrays.
// TODO(cc-ir): the round-0 entry points are the one evaluator family where
//   eager and graph still share the base+offset decoder — there is no
//   `_raw` twin here as there now is for the single-AIR, batched-DAG and
//   monomial families (`MainMatrixPtrs`).
// WHY: round 0 dispatches through `DISPATCH_BOOL_PAIR` /
//   `launch_zerocheck_coset_parallel<bool, bool>` / `dispatch_zerocheck`
//   (`cuda/src/logup_zerocheck/zerocheck_round0.cu:680-750`), so a raw variant
//   means threading a third template parameter through that whole macro
//   dispatch — much larger and riskier than the two flat launchers the other
//   families use, and not something to land unrehearsed.
// RISK: for round 0 only, a defect in the *shared* parts of the ABI — the
//   `BASE_OFF_NULL` sentinel, or a Rust/C++ layout drift in `MainMatrixDesc` —
//   is applied identically to eager and graph and no equality test can see it.
//   Partly mitigated: `assert_ctx_abi_matches_cuda` pins `MainMatrixDesc`'s
//   size *and* both field offsets against the C++ truth, and pins the sentinel
//   against `_base_off_null()`, so both defect classes have a direct guard
//   even without an independent oracle. What stays uncovered is a decode bug
//   that those size/offset/sentinel asserts do not express.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_ntt_eval_constraints(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    selectors_cube: &DeviceBuffer<F>,
    preprocessed: *const F,
    main_descs: *const MainMatrixDesc,
    pool_base: *const u8,
    eq_cube: *const EF,
    lambda_pows: &DeviceBuffer<EF>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<F>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_ntt_eval_constraints(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        selectors_cube.as_ptr(),
        preprocessed,
        main_descs,
        pool_base,
        eq_cube,
        lambda_pows.as_ptr(),
        public_values.as_ptr(),
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        lambda_pows.len(),
        buffer_size,
        intermediates.as_mut_ptr(),
        skip_domain,
        num_x,
        height,
        num_cosets,
        g_shift,
        max_temp_bytes,
        stream,
    ))
}

/// # Safety
/// - `buffer_size` does not refer to the capacity of `intermediates`. It refers to "how many DAG
///   nodes per row need to be buffered". The capacity is a multiple of `buffer_size` which is
///   runtime calculated based on `buffer_size`.
/// - `eq_cube` must be a pointer to device buffer with at least `num_x` elements representing
///   evaluations on hypercube.
/// - `output` will not be written to by this function. Only `tmp_sums_buffer` is written.
/// - `main_descs` must point to a device array of at least `n_main_parts` [`MainMatrixDesc`], where
///   `n_main_parts` is one past the largest `part` index any `ENTRY_MAIN` rule reads. Each
///   descriptor's `data` is a [`BaseOff`] decoded against `pool_base` — the base+offset ABI of
///   `cuda/include/base_off.cuh`. The eager path passes absolute addresses with `pool_base =
///   std::ptr::null()`; the graph-IR path passes pool offsets with the exe's pool base.
///   [`BaseOff::NULL`] is the only "absent" encoding — offset `0` is a valid pool offset.
/// - Round 0 does **not** read `MainMatrixDesc::air_width`: it strides a main matrix column-major
///   by `height` (`cuda/include/dag_entry.cuh`, `ENTRY_MAIN`). The field is carried only so this
///   table has the same layout as the batched evaluators' descriptor arrays.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_bary_eval_interactions_round0(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    selectors_cube: &DeviceBuffer<F>,
    preprocessed: *const F,
    main_descs: *const MainMatrixDesc,
    pool_base: *const u8,
    eq_cube: *const EF,
    public_values: &DeviceBuffer<F>,
    numer_weights: &DeviceBuffer<EF>,
    denom_weights: &DeviceBuffer<EF>,
    denom_sum_init: EF,
    rules: &DeviceBuffer<u128>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<F>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_bary_eval_interactions_round0(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        selectors_cube.as_ptr(),
        preprocessed,
        main_descs,
        pool_base,
        eq_cube,
        public_values.as_ptr(),
        numer_weights.as_ptr(),
        denom_weights.as_ptr(),
        denom_sum_init,
        rules.as_raw_ptr(),
        rules.len(),
        buffer_size,
        intermediates.as_mut_ptr(),
        skip_domain,
        num_x,
        height,
        num_cosets,
        g_shift,
        max_temp_bytes,
        stream,
    ))
}

/// Evaluate zerocheck MLE constraints on GPU with raw device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_mle(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    eq_xi: *const EF,
    selectors: *const EF,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: *const MainMatrixPtrs<EF>,
    lambda_pows: *const EF,
    lambda_len: usize,
    public_values: *const F,
    rules: *const std::ffi::c_void,
    rules_len: usize,
    used_nodes: *const usize,
    used_nodes_len: usize,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<EF>,
    num_y: u32,
    num_x: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        eq_xi,
        selectors,
        preprocessed,
        main_ptrs,
        lambda_pows,
        public_values,
        rules,
        rules_len,
        used_nodes,
        used_nodes_len,
        lambda_len,
        buffer_size,
        intermediates.as_mut_ptr(),
        num_y,
        num_x,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_batch_eval_mle(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    zc_ctxs: &DeviceBuffer<ZerocheckCtx>,
    pool_base: *const u8,
    air_block_offsets: &DeviceBuffer<u32>,
    lambda_pows: &DeviceBuffer<EF>,
    lambda_len: usize,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_batch_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        zc_ctxs.as_ptr(),
        pool_base,
        air_block_offsets.as_ptr(),
        lambda_pows.as_ptr(),
        lambda_len,
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
        stream,
    ))
}

/// Evaluate logup MLE interactions on GPU with raw device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_eval_mle(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    eq_xi: *const EF,
    selectors: *const EF,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: *const MainMatrixPtrs<EF>,
    challenges: *const EF,
    eq_3bs: *const EF,
    public_values: *const F,
    rules: *const std::ffi::c_void,
    used_nodes: *const usize,
    pair_idxs: *const u32,
    used_nodes_len: usize,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<EF>,
    num_y: u32,
    num_x: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        eq_xi,
        selectors,
        preprocessed,
        main_ptrs,
        challenges,
        eq_3bs,
        public_values,
        rules,
        used_nodes,
        pair_idxs,
        used_nodes_len,
        buffer_size,
        intermediates.as_mut_ptr(),
        num_y,
        num_x,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_batch_eval_mle(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    logup_ctxs: &DeviceBuffer<LogupCtx>,
    pool_base: *const u8,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_batch_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        logup_ctxs.as_ptr(),
        pool_base,
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
        stream,
    ))
}

/// The eager batched zerocheck evaluator: raw-pointer contexts, no pool base.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_batch_eval_mle_raw(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    zc_ctxs: &DeviceBuffer<ZerocheckCtxRaw>,
    air_block_offsets: &DeviceBuffer<u32>,
    lambda_pows: &DeviceBuffer<EF>,
    lambda_len: usize,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_batch_eval_mle_raw(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        zc_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        lambda_pows.as_ptr(),
        lambda_len,
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
        stream,
    ))
}

/// The eager batched logup evaluator: raw-pointer contexts, no pool base.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_batch_eval_mle_raw(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    logup_ctxs: &DeviceBuffer<LogupCtxRaw>,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_batch_eval_mle_raw(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        logup_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
        stream,
    ))
}

/// The eager batched monomial evaluator: raw-pointer contexts, no pool base.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_monomial_batched_raw(
    tmp_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    air_ctxs: &DeviceBuffer<MonomialAirCtxRaw>,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_monomial_batched_raw(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        air_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
        stream,
    ))
}

/// The eager par-y batched monomial evaluator: raw-pointer contexts.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_monomial_par_y_batched_raw(
    tmp_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    air_ctxs: &DeviceBuffer<MonomialAirCtxRaw>,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    chunk_size: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_monomial_par_y_batched_raw(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        air_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        chunk_size,
        threads_per_block,
        stream,
    ))
}

/// The eager batched logup-monomial evaluator: raw-pointer contexts.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_monomial_batched_raw(
    tmp_sums: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    common_ctxs: &DeviceBuffer<LogupMonomialCommonCtxRaw>,
    numer_ctxs: &DeviceBuffer<LogupMonomialCtx>,
    denom_ctxs: &DeviceBuffer<LogupMonomialCtx>,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_monomial_batched_raw(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        common_ctxs.as_ptr(),
        numer_ctxs.as_ptr(),
        denom_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_monomial_batched(
    tmp_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    air_ctxs: &DeviceBuffer<MonomialAirCtx>,
    pool_base: *const u8,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_monomial_batched(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        air_ctxs.as_ptr(),
        pool_base,
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_monomial_par_y_batched(
    tmp_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    air_ctxs: &DeviceBuffer<MonomialAirCtx>,
    pool_base: *const u8,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    chunk_size: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_zerocheck_monomial_par_y_batched(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        air_ctxs.as_ptr(),
        pool_base,
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        chunk_size,
        threads_per_block,
        stream,
    ))
}

pub unsafe fn precompute_lambda_combinations(
    out: &mut DeviceBuffer<EF>,
    headers: *const MonomialHeader,
    lambda_terms: *const LambdaTerm<F>,
    lambda_pows: &DeviceBuffer<EF>,
    num_monomials: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_precompute_lambda_combinations(
        out.as_mut_ptr(),
        headers,
        lambda_terms,
        lambda_pows.as_ptr(),
        num_monomials,
        stream,
    ))
}

pub unsafe fn precompute_logup_numer_combinations(
    out: &mut DeviceBuffer<EF>,
    headers: *const MonomialHeader,
    terms: *const InteractionMonomialTerm<F>,
    eq_3bs: &DeviceBuffer<EF>,
    num_monomials: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_precompute_logup_numer_combinations(
        out.as_mut_ptr(),
        headers,
        terms,
        eq_3bs.as_ptr(),
        num_monomials,
        stream,
    ))
}

pub unsafe fn precompute_logup_denom_combinations(
    out: &mut DeviceBuffer<EF>,
    headers: *const MonomialHeader,
    terms: *const InteractionMonomialTerm<F>,
    beta_pows: &DeviceBuffer<EF>,
    eq_3bs: &DeviceBuffer<EF>,
    num_monomials: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_precompute_logup_denom_combinations(
        out.as_mut_ptr(),
        headers,
        terms,
        beta_pows.as_ptr(),
        eq_3bs.as_ptr(),
        num_monomials,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_monomial_batched(
    tmp_sums: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    common_ctxs: &DeviceBuffer<LogupMonomialCommonCtx>,
    numer_ctxs: &DeviceBuffer<LogupMonomialCtx>,
    denom_ctxs: &DeviceBuffer<LogupMonomialCtx>,
    pool_base: *const u8,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_logup_monomial_batched(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        common_ctxs.as_ptr(),
        numer_ctxs.as_ptr(),
        denom_ctxs.as_ptr(),
        pool_base,
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
        stream,
    ))
}

pub unsafe fn frac_vector_scalar_multiply_ext_fp(
    frac_vec: *mut Frac<EF>,
    scalar: F,
    length: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_frac_vector_scalar_multiply_ext_fp(
        frac_vec, scalar, length, stream,
    ))
}

/// Vertically repeats the rows of `input` and writes them to `out`. Both matrices are column-major.
///
/// # Safety
/// - `out` must be a pointer to `DeviceBuffer<F>` with length at least `lifted_height * width`.
/// - `input` must be a pointer to `DeviceBuffer<F>` with length at least `height * width`.
/// - `out` and `input` must not overlap.
pub unsafe fn frac_matrix_vertically_repeat(
    out: *mut Frac<EF>,
    input: *const Frac<EF>,
    width: u32,
    lifted_height: u32,
    height: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(lifted_height > height);
    CudaError::from_result(_frac_matrix_vertically_repeat(
        out,
        input,
        width,
        lifted_height,
        height,
        stream,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_matrix_vertically_repeat_ext(
    out_numerators: *mut EF,
    out_denominators: *mut EF,
    in_numerators: *const EF,
    in_denominators: *const EF,
    width: u32,
    lifted_height: u32,
    height: u32,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    debug_assert!(lifted_height > height);
    CudaError::from_result(_frac_matrix_vertically_repeat_ext(
        out_numerators,
        out_denominators,
        in_numerators,
        in_denominators,
        width,
        lifted_height,
        height,
        stream,
    ))
}

/// Create folded selectors around round 0 from hypercube evaluations and univariate factors.
///
/// Note: `is_transition` is not a product of univariate and hypercube factors.
pub unsafe fn fold_selectors_round0(
    out: *mut EF,
    input: *const F,
    is_first: EF,
    is_last: EF,
    num_x: usize,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    CudaError::from_result(_fold_selectors_round0(
        out,
        input,
        is_first,
        is_last,
        num_x as u32,
        stream,
    ))
}

// ===========================================================================
// Ctx materializers (graph-IR)
// ===========================================================================
//
// One `<<<1,1>>>` launch writes one complete element of a batched ctx array.
// Callers pass raw device pointers that were resolved at *invocation* time
// (in the graph-IR case, `BufId`s resolved by the graph runner), so the
// planner keeps a real edge for every buffer the evaluator will dereference
// through the ctx.
//
// These are `unsafe fn`s over raw pointers on purpose: the graph-facing
// wrappers must not construct owning `DeviceBuffer`s (that would free pool
// memory on drop), matching `fractional_ir.rs`'s device-challenge wrappers.

/// Panics unless the Rust ctx mirrors agree byte-for-byte in size with the
/// private C++ definitions in `batch_mle.cu`.
///
/// The two layouts are hand-duplicated and relied upon by both producers of
/// the base+offset ABI — the eager H2D upload and the graph-IR descriptor
/// builder — so drift must fail loudly. Field *order* is covered by the
/// graph-vs-eager field-by-field test in `zerocheck_ir.rs`.
pub fn assert_ctx_abi_matches_cuda() {
    unsafe {
        assert_eq!(
            std::mem::size_of::<BaseOff>(),
            8,
            "BaseOff must be exactly the `uint64_t` the CUDA ABI stores"
        );
        assert_eq!(
            std::mem::size_of::<BaseOff>(),
            _base_off_size(),
            "BaseOff layout drift vs CUDA"
        );
        // The absent encoding. `0` cannot serve as the sentinel — it is a valid
        // pool offset (the first packed buffer lives there) — so a drift here
        // would make every offset-0 descriptor decode to `nullptr`.
        assert_eq!(
            BaseOff::NULL.0,
            _base_off_null(),
            "BASE_OFF_NULL drift vs CUDA"
        );
        assert_ne!(
            BaseOff::NULL,
            BaseOff::from_offset(0),
            "offset 0 must not be the absent encoding"
        );
        assert_eq!(
            std::mem::size_of::<MainMatrixDesc>(),
            _main_matrix_desc_size(),
            "MainMatrixDesc layout drift vs CUDA"
        );
        // The round-0 entry points (`_zerocheck_ntt_eval_constraints`,
        // `_logup_bary_eval_interactions_round0`) index a `MainMatrixDesc`
        // array on device and decode `.data` against `pool_base`. Field-order
        // drift there mis-addresses every main matrix instead of failing to
        // compile, so pin the offsets, not just the size.
        assert_eq!(
            std::mem::offset_of!(MainMatrixDesc, data),
            _main_matrix_desc_data_offset(),
            "MainMatrixDesc::data offset drift vs CUDA"
        );
        assert_eq!(
            std::mem::offset_of!(MainMatrixDesc, data),
            0,
            "MainMatrixDesc::data must be the first field"
        );
        assert_eq!(
            std::mem::offset_of!(MainMatrixDesc, air_width),
            _main_matrix_desc_air_width_offset(),
            "MainMatrixDesc::air_width offset drift vs CUDA"
        );
        assert_eq!(
            std::mem::size_of::<EvalCoreCtx>(),
            _eval_core_ctx_size(),
            "EvalCoreCtx layout drift vs CUDA"
        );
        assert_eq!(
            std::mem::size_of::<ZerocheckCtx>(),
            _zerocheck_ctx_size(),
            "ZerocheckCtx layout drift vs CUDA"
        );
        assert_eq!(
            std::mem::size_of::<LogupCtx>(),
            _logup_ctx_size(),
            "LogupCtx layout drift vs CUDA"
        );

        // The raw-pointer (eager) ABI. It is a *separate* ABI from the one
        // above — that separation is what makes eager an independent oracle
        // (see [`MainMatrixPtrs`]) — so it needs its own guard rather than
        // inheriting the base+offset one.
        assert_eq!(
            std::mem::size_of::<MainMatrixPtrs<EF>>(),
            _main_matrix_ptrs_size(),
            "MainMatrixPtrs<EF> layout drift vs CUDA"
        );
        assert_eq!(
            std::mem::size_of::<EvalCoreCtxRaw>(),
            _eval_core_ctx_raw_size(),
            "EvalCoreCtxRaw layout drift vs CUDA"
        );
        assert_eq!(
            std::mem::size_of::<ZerocheckCtxRaw>(),
            _zerocheck_ctx_raw_size(),
            "ZerocheckCtxRaw layout drift vs CUDA"
        );
        assert_eq!(
            std::mem::size_of::<LogupCtxRaw>(),
            _logup_ctx_raw_size(),
            "LogupCtxRaw layout drift vs CUDA"
        );
    }
}

// ===========================================================================
// Tests.
// ===========================================================================

/// Differential tests for the round-0 evaluators' base+offset main-matrix ABI.
///
/// The two round-0 entry points used to take a bare `*const *const F` pointer
/// table. They now take a [`MainMatrixDesc`] array plus a `pool_base`, decoded
/// with `base_off_ptr` (`cuda/include/base_off.cuh`). These tests pin the two
/// properties that move depends on:
///
/// 1. The *eager* encoding — absolute addresses against a null `pool_base` — is byte-identical in
///    effect to the raw pointer table it replaced, and equals the *pool* encoding (offsets against
///    a real base) on the same bytes. That equality is the differential oracle for the graph-IR
///    path.
/// 2. Offset `0` is a live offset, not the absent encoding. Part 0 sits at pool offset 0 in every
///    pool-encoded run here, so a regression that treated `0` as `nullptr` would fault or produce
///    garbage rather than pass quietly.
#[cfg(test)]
mod round0_base_off_tests {
    use openvm_cuda_common::{
        common::get_device,
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::{
        air_builders::symbolic::{
            symbolic_variable::{Entry, SymbolicVariable},
            SymbolicExpressionDag, SymbolicExpressionNode,
        },
        prover::fractional_sumcheck_gkr::Frac,
    };
    use p3_field::{PrimeCharacteristicRing, TwoAdicField};
    use p3_util::log2_ceil_usize;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::{
        _logup_r0_intermediates_buffer_size, _logup_r0_temp_sums_buffer_size,
        _zerocheck_r0_intermediates_buffer_size, _zerocheck_r0_temp_sums_buffer_size,
        logup_bary_eval_interactions_round0, zerocheck_ntt_eval_constraints, BaseOff,
        MainMatrixDesc, EF, F,
    };
    use crate::logup_zerocheck::rules::{codec::Codec, SymbolicRulesGpu};

    /// Columns per main part. Part 0 is read at column 0 and column 2, part 1
    /// at column 1, so a wrong descriptor for *either* part changes the result.
    const PART0_WIDTH: usize = 3;
    const PART1_WIDTH: usize = 2;
    const NUM_PARTS: usize = 2;
    const MAX_TEMP_BYTES: usize = 1 << 30;

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        }
    }

    fn rand_f(rng: &mut StdRng) -> F {
        F::from_u32(rng.random_range(1..1 << 20))
    }

    /// A DAG whose only leaves are `ENTRY_MAIN` reads:
    /// `main[0][0] * main[1][1]` and `(that) + main[0][2].next()`.
    fn main_only_dag() -> SymbolicExpressionDag<F> {
        let var = |part_index: usize, offset: usize, index: usize| {
            SymbolicExpressionNode::Variable(SymbolicVariable::new(
                Entry::Main { part_index, offset },
                index,
            ))
        };
        SymbolicExpressionDag {
            nodes: vec![
                var(0, 0, 0),
                var(1, 0, 1),
                SymbolicExpressionNode::Mul {
                    left_idx: 0,
                    right_idx: 1,
                    degree_multiple: 2,
                },
                // A rotated read, so `SourceInfo::offset` is exercised too.
                var(0, 1, 2),
                SymbolicExpressionNode::Add {
                    left_idx: 2,
                    right_idx: 3,
                    degree_multiple: 2,
                },
            ],
            // Must stay sorted: `SymbolicRulesGpu::new` debug-asserts it.
            constraint_idx: vec![2, 4],
        }
    }

    /// How to encode the main-matrix table for one run.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum MainEncoding {
        /// The eager encoding, and the reference arm of every assertion here:
        /// `BaseOff::from_ptr` stores the absolute device address and the
        /// launcher gets a null `pool_base`, so `base_off_ptr` returns exactly
        /// the pointer the old `*const *const F` table held. This arm *is* the
        /// pointer path.
        Absolute,
        /// The graph encoding: byte offsets into a pool, decoded against the
        /// pool's real base. Part 0 is deliberately at offset `0`.
        PoolOffsets,
        /// [`Self::PoolOffsets`] with exactly one descriptor element moved by
        /// exactly one `F`. The read stays inside the pool (part 1's base
        /// shifts *down*), so this is a wrong answer, not a fault.
        PoolOffsetsSabotaged,
    }

    /// The descriptor array and matching `pool_base` for one encoding.
    fn main_table(
        enc: MainEncoding,
        pool: &DeviceBuffer<F>,
        height: usize,
    ) -> (Vec<MainMatrixDesc>, *const u8) {
        let part1_elems = PART0_WIDTH * height;
        let part1_bytes = (part1_elems * size_of::<F>()) as u64;
        let desc = |off: u64| MainMatrixDesc {
            data: BaseOff::from_offset(off),
            air_width: 0,
        };
        match enc {
            MainEncoding::Absolute => {
                let descs = vec![
                    MainMatrixDesc::round0(pool.as_ptr()),
                    MainMatrixDesc::round0(unsafe { pool.as_ptr().add(part1_elems) }),
                ];
                // The equivalence this arm stands for: decoding against a null
                // base reproduces the raw pointer the old table held.
                assert_eq!(
                    descs[0].data.resolve(std::ptr::null()),
                    pool.as_ptr() as *const u8,
                    "eager encoding must decode to the original pointer"
                );
                (descs, std::ptr::null())
            }
            MainEncoding::PoolOffsets => {
                (vec![desc(0), desc(part1_bytes)], pool.as_ptr() as *const u8)
            }
            MainEncoding::PoolOffsetsSabotaged => (
                vec![desc(0), desc(part1_bytes - size_of::<F>() as u64)],
                pool.as_ptr() as *const u8,
            ),
        }
    }

    /// Shapes and read-only inputs shared by every run of one configuration.
    struct Fixture {
        ctx: GpuDeviceCtx,
        num_x: u32,
        skip_domain: u32,
        height: u32,
        num_cosets: u32,
        g_shift: F,
        /// Both main parts packed back to back, part 0 first at offset 0.
        pool: DeviceBuffer<F>,
        selectors: DeviceBuffer<F>,
        eq_cube: DeviceBuffer<EF>,
        public_values: DeviceBuffer<F>,
    }

    impl Fixture {
        fn new(num_x: u32, skip_domain: u32, num_cosets: u32, seed: u64) -> Self {
            let ctx = test_ctx();
            let mut rng = StdRng::seed_from_u64(seed);
            let height = num_x * skip_domain;
            let pool_len = (PART0_WIDTH + PART1_WIDTH) * height as usize;
            let host_pool = (0..pool_len).map(|_| rand_f(&mut rng)).collect::<Vec<_>>();
            let host_sels = (0..3 * num_x as usize)
                .map(|_| rand_f(&mut rng))
                .collect::<Vec<_>>();
            let host_eq = (0..num_x as usize)
                .map(|_| rng.random::<EF>())
                .collect::<Vec<_>>();
            let host_pub = (0..4).map(|_| rand_f(&mut rng)).collect::<Vec<_>>();

            // Mirrors `logup_zerocheck/mod.rs`: the round-0 coset generator is
            // the two-adic root of the large domain `constraint_deg << l_skip`.
            let l_skip = skip_domain.ilog2() as usize;
            let constraint_deg = num_cosets as usize + 1;
            let g_shift = F::two_adic_generator(log2_ceil_usize(constraint_deg << l_skip));

            Self {
                pool: host_pool.to_device_on(&ctx).unwrap(),
                selectors: host_sels.to_device_on(&ctx).unwrap(),
                eq_cube: host_eq.to_device_on(&ctx).unwrap(),
                public_values: host_pub.to_device_on(&ctx).unwrap(),
                ctx,
                num_x,
                skip_domain,
                height,
                num_cosets,
                g_shift,
            }
        }

        fn out_len(&self) -> usize {
            (self.num_cosets * self.skip_domain) as usize
        }
    }

    /// One `zerocheck_ntt_eval_constraints` run; returns the `output` buffer.
    fn run_zerocheck(fx: &Fixture, enc: MainEncoding) -> Vec<EF> {
        let dag = main_only_dag();
        let rules = SymbolicRulesGpu::new(&dag, false);
        // Same construction as `pkey.rs`: rule index per accumulated node.
        let used_nodes = dag
            .constraint_idx
            .iter()
            .map(|i| rules.dag_idx_to_rule_idx[i])
            .collect::<Vec<_>>();
        let encoded = rules.rules.iter().map(|r| r.encode()).collect::<Vec<_>>();
        let d_rules = encoded.to_device_on(&fx.ctx).unwrap();
        let d_used_nodes = used_nodes.to_device_on(&fx.ctx).unwrap();
        let buffer_size: u32 = rules.buffer_size.try_into().unwrap();

        let mut rng = StdRng::seed_from_u64(0x1A_B0DA);
        let lambda = (0..used_nodes.len())
            .map(|_| rng.random::<EF>())
            .collect::<Vec<_>>();
        let d_lambda = lambda.to_device_on(&fx.ctx).unwrap();

        let inter_cap = unsafe {
            _zerocheck_r0_intermediates_buffer_size(
                buffer_size,
                fx.skip_domain,
                fx.num_x,
                fx.num_cosets,
                MAX_TEMP_BYTES,
            )
        };
        let mut intermediates = if inter_cap > 0 {
            DeviceBuffer::<F>::with_capacity_on(inter_cap, &fx.ctx)
        } else {
            DeviceBuffer::<F>::new()
        };
        let tmp_cap = unsafe {
            _zerocheck_r0_temp_sums_buffer_size(
                buffer_size,
                fx.skip_domain,
                fx.num_x,
                fx.num_cosets,
                MAX_TEMP_BYTES,
            )
        };
        let mut tmp = DeviceBuffer::<EF>::with_capacity_on(tmp_cap, &fx.ctx);
        let mut out = DeviceBuffer::<EF>::with_capacity_on(fx.out_len(), &fx.ctx);

        let (descs, pool_base) = main_table(enc, &fx.pool, fx.height as usize);
        assert_eq!(descs.len(), NUM_PARTS);
        let d_descs = descs.to_device_on(&fx.ctx).unwrap();

        unsafe {
            zerocheck_ntt_eval_constraints(
                &mut tmp,
                &mut out,
                &fx.selectors,
                std::ptr::null(), // no preprocessed trace
                d_descs.as_ptr(),
                pool_base,
                fx.eq_cube.as_ptr(),
                &d_lambda,
                &fx.public_values,
                &d_rules,
                &d_used_nodes,
                buffer_size,
                &mut intermediates,
                fx.skip_domain,
                fx.num_x,
                fx.height,
                fx.num_cosets,
                fx.g_shift,
                MAX_TEMP_BYTES,
                fx.ctx.stream.as_raw(),
            )
            .expect("zerocheck round-0 launch");
        }
        out.to_host_on(&fx.ctx).unwrap()
    }

    /// One `logup_bary_eval_interactions_round0` run; returns `output`.
    fn run_logup(fx: &Fixture, enc: MainEncoding) -> Vec<Frac<EF>> {
        let dag = main_only_dag();
        // `true` matches the logup round-0 path in `logup_zerocheck/round0.rs`.
        let rules = SymbolicRulesGpu::new(&dag, true);
        let encoded = rules.rules.iter().map(|r| r.encode()).collect::<Vec<_>>();
        let d_rules = encoded.to_device_on(&fx.ctx).unwrap();
        let buffer_size: u32 = rules.buffer_size.try_into().unwrap();

        // The kernel indexes both weight vectors by rule index. Their *values*
        // are irrelevant to this test — only that both arms see the same ones.
        let mut rng = StdRng::seed_from_u64(0xBEEF_0FF5);
        let numer = (0..rules.rules.len())
            .map(|_| rng.random::<EF>())
            .collect::<Vec<_>>();
        let denom = (0..rules.rules.len())
            .map(|_| rng.random::<EF>())
            .collect::<Vec<_>>();
        let denom_sum_init = rng.random::<EF>();
        let d_numer = numer.to_device_on(&fx.ctx).unwrap();
        let d_denom = denom.to_device_on(&fx.ctx).unwrap();

        let inter_cap = unsafe {
            _logup_r0_intermediates_buffer_size(
                buffer_size,
                fx.skip_domain,
                fx.num_x,
                fx.num_cosets,
                MAX_TEMP_BYTES,
            )
        };
        let mut intermediates = if inter_cap > 0 {
            DeviceBuffer::<F>::with_capacity_on(inter_cap, &fx.ctx)
        } else {
            DeviceBuffer::<F>::new()
        };
        let tmp_cap = unsafe {
            _logup_r0_temp_sums_buffer_size(
                buffer_size,
                fx.skip_domain,
                fx.num_x,
                fx.num_cosets,
                MAX_TEMP_BYTES,
            )
        };
        let mut tmp = DeviceBuffer::<Frac<EF>>::with_capacity_on(tmp_cap, &fx.ctx);
        let mut out = DeviceBuffer::<Frac<EF>>::with_capacity_on(fx.out_len(), &fx.ctx);

        let (descs, pool_base) = main_table(enc, &fx.pool, fx.height as usize);
        let d_descs = descs.to_device_on(&fx.ctx).unwrap();

        unsafe {
            logup_bary_eval_interactions_round0(
                &mut tmp,
                &mut out,
                &fx.selectors,
                std::ptr::null(), // no preprocessed trace
                d_descs.as_ptr(),
                pool_base,
                fx.eq_cube.as_ptr(),
                &fx.public_values,
                &d_numer,
                &d_denom,
                denom_sum_init,
                &d_rules,
                buffer_size,
                &mut intermediates,
                fx.skip_domain,
                fx.num_x,
                fx.height,
                fx.num_cosets,
                fx.g_shift,
                MAX_TEMP_BYTES,
                fx.ctx.stream.as_raw(),
            )
            .expect("logup round-0 launch");
        }
        out.to_host_on(&fx.ctx).unwrap()
    }

    fn bytes_of<T>(v: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    /// `(num_x, skip_domain, num_cosets)` for the two dispatch modes the
    /// round-0 launchers pick between: `num_x * skip_domain < 32768` takes the
    /// coset-parallel kernel, otherwise lockstep
    /// (`logup_zerocheck/round0.rs`, `ROUND0_COSET_PARALLEL_THRESHOLD`).
    const CONFIGS: [(u32, u32, u32); 2] = [
        (8, 4, 2),    // coset-parallel
        (8192, 4, 2), // lockstep
    ];

    /// The pool encoding must reproduce the eager encoding byte for byte, for
    /// both round-0 evaluators and both dispatch modes.
    #[test]
    fn round0_pool_offsets_match_absolute_addresses() {
        for (num_x, skip_domain, num_cosets) in CONFIGS {
            let fx = Fixture::new(num_x, skip_domain, num_cosets, 0x5EED_0A81);
            let label = format!("num_x={num_x} skip_domain={skip_domain}");

            let zc_abs = run_zerocheck(&fx, MainEncoding::Absolute);
            let zc_pool = run_zerocheck(&fx, MainEncoding::PoolOffsets);
            assert_eq!(
                bytes_of(&zc_abs),
                bytes_of(&zc_pool),
                "zerocheck round-0: pool offsets diverge from absolute addresses ({label})"
            );

            let lg_abs = run_logup(&fx, MainEncoding::Absolute);
            let lg_pool = run_logup(&fx, MainEncoding::PoolOffsets);
            assert_eq!(
                bytes_of(&lg_abs),
                bytes_of(&lg_pool),
                "logup round-0: pool offsets diverge from absolute addresses ({label})"
            );

            // The oracle only has teeth if the outputs are not trivially zero
            // (a null main table would give exactly that).
            assert!(
                bytes_of(&zc_abs).iter().any(|b| *b != 0),
                "zerocheck round-0 output is all zero — the table was not read ({label})"
            );
            assert!(
                bytes_of(&lg_abs).iter().any(|b| *b != 0),
                "logup round-0 output is all zero — the table was not read ({label})"
            );
        }
    }

    /// Sabotage: move exactly one descriptor element by exactly one `F`. Both
    /// evaluators must produce a different answer — otherwise the equality
    /// above proves nothing about the descriptors actually being dereferenced.
    #[test]
    fn round0_one_element_descriptor_sabotage_changes_output() {
        let (num_x, skip_domain, num_cosets) = CONFIGS[0];
        let fx = Fixture::new(num_x, skip_domain, num_cosets, 0x5EED_0A81);

        let zc_good = run_zerocheck(&fx, MainEncoding::PoolOffsets);
        let zc_bad = run_zerocheck(&fx, MainEncoding::PoolOffsetsSabotaged);
        assert_ne!(
            bytes_of(&zc_good),
            bytes_of(&zc_bad),
            "zerocheck round-0 ignored a one-element descriptor perturbation"
        );

        let lg_good = run_logup(&fx, MainEncoding::PoolOffsets);
        let lg_bad = run_logup(&fx, MainEncoding::PoolOffsetsSabotaged);
        assert_ne!(
            bytes_of(&lg_good),
            bytes_of(&lg_bad),
            "logup round-0 ignored a one-element descriptor perturbation"
        );
    }

    /// `BASE_OFF_NULL` is the only absent encoding; offset `0` is a live
    /// offset. Both round-0 pool runs above put part 0 at offset `0`, so this
    /// is the host-side statement of what those runs depend on.
    #[test]
    fn base_off_zero_is_a_live_offset() {
        assert_ne!(BaseOff::from_offset(0), BaseOff::NULL);
        assert_eq!(
            BaseOff::from_offset(0).resolve(0x1000 as *const u8),
            0x1000 as *const u8
        );
        assert!(BaseOff::NULL.resolve(0x1000 as *const u8).is_null());
        // A null matrix pointer is the absent encoding, not offset 0.
        assert_eq!(
            MainMatrixDesc::round0(std::ptr::null::<F>()).data,
            BaseOff::NULL
        );
        // And the CUDA side agrees on all of the above.
        super::assert_ctx_abi_matches_cuda();
    }
}
