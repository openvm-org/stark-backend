use thiserror::Error;

/// Top-level error type for the CPU prover, used as `ProverDevice::Error`.
#[derive(Error, Debug)]
pub enum CpuProverError {
    #[error("Stacked PCS: {0}")]
    StackedPcs(#[from] StackedPcsError),
    #[error("Sumcheck: {0}")]
    Sumcheck(#[from] SumcheckError),
    #[error("LogupZerocheck: {0}")]
    LogupZerocheck(#[from] LogupZerocheckError),
    #[error("Stacked reduction: {0}")]
    StackedReduction(#[from] StackedReductionError),
    #[error("WHIR: {0}")]
    Whir(#[from] WhirProverError),
}

/// Errors from `stacked_pcs.rs` — Merkle tree, stacked layout, RS code.
#[derive(Error, Debug)]
pub enum StackedPcsError {
    #[error("StackedLayout::new: column height {log_height} exceeds stacked height {log_stacked_height}")]
    LayoutHeightExceeded {
        log_height: usize,
        log_stacked_height: usize,
    },
    #[error("StackedLayout::new: row overflow at col_idx={col_idx}, expected row_idx to equal stacked height {stacked_height}")]
    LayoutRowOverflow {
        col_idx: usize,
        stacked_height: usize,
    },
    #[error("StackedLayout::from_raw_parts: mat_idx {mat_idx} does not equal mat_starts.len() {mat_starts_len}")]
    LayoutRawPartsMatIdx {
        mat_idx: usize,
        mat_starts_len: usize,
    },
    #[error("MerkleTree::new: matrix height must be > 0")]
    MerkleTreeEmptyMatrix,
    #[error("MerkleTree::new: rows_per_query ({rows_per_query}) is not a power of two")]
    MerkleTreeRowsPerQueryNotPow2 { rows_per_query: usize },
    #[error("MerkleTree::new: rows_per_query ({rows_per_query}) exceeds number of Merkle leaves ({num_leaves})")]
    MerkleTreeRowsPerQueryExceeded {
        rows_per_query: usize,
        num_leaves: usize,
    },
    #[error("MerkleTree: empty digest layers (no root)")]
    MerkleTreeNoRoot,
    #[error("MerkleTree::query_merkle_proof: query_idx {query_idx} out of bounds for query_stride {query_stride}")]
    MerkleTreeQueryOutOfBounds {
        query_idx: usize,
        query_stride: usize,
    },
    #[error(
        "MerkleTree::get_opened_rows: index {index} out of bounds for query_stride {query_stride}"
    )]
    MerkleTreeOpenedRowsOutOfBounds { index: usize, query_stride: usize },
    #[error("stacked_matrix: width * height overflow")]
    StackedMatrixOverflow,
    #[error("rs_code_matrix: checked_shl overflow for height={height}, log_blowup={log_blowup}")]
    RsCodeShiftOverflow { height: usize, log_blowup: usize },
}

/// Errors from `sumcheck.rs`.
#[derive(Error, Debug)]
pub enum SumcheckError {
    #[error("sumcheck_multilinear: round polynomial length {len} != 1")]
    MultilinearRoundPolyLen { len: usize },
    #[error("sumcheck_multilinear: final evaluation count {len} != 1")]
    MultilinearFinalEvalLen { len: usize },
    #[error("sumcheck_prismalinear: prism_dim {prism_dim} < l_skip {l_skip}")]
    PrismalinearDimTooSmall { prism_dim: usize, l_skip: usize },
    #[error("sumcheck_prismalinear: round polynomial length {len} != 1")]
    PrismalinearRoundPolyLen { len: usize },
    #[error("sumcheck_prismalinear: r.len() {r_len} != n + 1 = {expected}")]
    PrismalinearRLen { r_len: usize, expected: usize },
    #[error("sumcheck_prismalinear: final evaluation count {len} != 1")]
    PrismalinearFinalEvalLen { len: usize },
}

/// Errors from `logup_zerocheck/` module.
#[derive(Error, Debug)]
pub enum LogupZerocheckError {
    #[error("Stacked PCS: {0}")]
    StackedPcs(#[from] StackedPcsError),
    #[error("fractional_sumcheck: non-zero root sum in assert_zero mode")]
    NonZeroRootSum,
    #[error(
        "LogupZerocheckCpu::new: preprocessed trace index {index} out of bounds for width {width}"
    )]
    PreprocessedIndexOutOfBounds { index: usize, width: usize },
    #[error("LogupZerocheckCpu::new: main partition {part_index} col_index {col_index} >= width {width}")]
    MainPartitionIndexOutOfBounds {
        part_index: usize,
        col_index: usize,
        width: usize,
    },
    #[error("LogupZerocheckCpu::new: public value index {index} out of bounds for len {len}")]
    PublicValueIndexOutOfBounds { index: usize, len: usize },
    #[error("LogupZerocheckCpu::new: after_challenge not supported")]
    AfterChallengeNotSupported,
    #[error("interactions_layout.get returned None for (trace_idx={trace_idx}, interaction_idx={interaction_idx})")]
    InteractionsLayoutMissing {
        trace_idx: usize,
        interaction_idx: usize,
    },
    #[error("sumcheck_polys_eval: eq_ns is empty at round {round}")]
    EqNsEmpty { round: usize },
    #[error("sumcheck_polys_eval: eq_sharp_ns is empty at round {round}")]
    EqSharpNsEmpty { round: usize },
    #[error("into_column_openings: mat_evals.pop() returned None")]
    MatEvalsPopNone,
    #[error("into_column_openings: claim.len() {len} != 1")]
    ClaimLenNotOne { len: usize },
    #[error("prove_zerocheck_and_logup: r.len() {r_len} != n_max + 1 = {expected}")]
    RLenMismatch { r_len: usize, expected: usize },
}

/// Errors from `whir.rs`.
#[derive(Error, Debug)]
pub enum WhirProverError {
    #[error("Stacked PCS: {0}")]
    StackedPcs(#[from] StackedPcsError),
    #[error("prove_whir_opening: try_into failed for sumcheck poly evals")]
    TryIntoFailed,
    #[error("prove_whir_opening: tree height {tree_height} != expected {expected}")]
    TreeHeightMismatch { tree_height: usize, expected: usize },
    #[error("prove_whir_opening: rs_tree is None in whir_round > 0")]
    RsTreeNone,
    #[error("prove_whir_opening: tree width {width} != 1")]
    TreeWidthNotOne { width: usize },
    #[error("prove_whir_opening: z_0 is None (not last round)")]
    Z0None,
    #[error("prove_whir_opening: final_poly is None")]
    FinalPolyNone,
    #[error("prove_openings: u_prisma is empty (split_first failed)")]
    UPrismaEmpty,
}

/// Errors from `stacked_reduction.rs`.
#[derive(Error, Debug)]
pub enum StackedReductionError {
    #[error(
        "batch_sumcheck_uni_round0_poly: eq_r_per_lht missing entry for log_height={log_height}"
    )]
    EqRMissing { log_height: usize },
    #[error("batch_sumcheck_poly_eval: eq_r_per_lht missing entry for log_height={log_height}")]
    EqRMissingPolyEval { log_height: usize },
    #[error("batch_sumcheck_poly_eval: k_rot_r_per_lht missing entry for log_height={log_height}")]
    KRotRMissing { log_height: usize },
}
