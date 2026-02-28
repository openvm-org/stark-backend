use openvm_cuda_common::error::{CudaError, MemCopyError};
use thiserror::Error;

use crate::{
    logup_zerocheck::{
        FoldPleError, FractionalSumcheckError, InteractionGpuError, Round0EvalError,
    },
    sponge::GrindError,
};

#[derive(Error, Debug)]
pub enum ProverError {
    #[error("MemCopy: {0}")]
    MemCopy(#[from] MemCopyError),
    #[error("current_stream_sync: {0}")]
    CurrentStreamSync(CudaError),
    #[error("collapse_strided_matrix: {0}")]
    CollapseStrided(CudaError),
    #[error("Stack traces: {0}")]
    StackTraces(#[from] StackTracesError),
    #[error("MerkleTree: {0}")]
    MerkleTree(#[from] MerkleTreeError),
    #[error("rs_code_matrix: {0}")]
    RsCodeMatrix(#[from] RsCodeMatrixError),
    #[error("WHIR: {0}")]
    Whir(#[from] WhirProverError),
    #[error("Stacked reduction: {0}")]
    StackedReduction(#[from] StackedReductionError),
    #[error("LogupZerocheck: {0}")]
    LogupZerocheck(#[from] LogupZerocheckError),
}

#[derive(Error, Debug)]
pub enum StackedReductionError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("EqEvalSegments: {0}")]
    EqEvalSegments(KernelError),
    #[error("fill_zero: {0}")]
    FillZero(CudaError),
    #[error("stacked_reduction_sumcheck_round0: {0}")]
    SumcheckRound0(CudaError),
    #[error("stacked_reduction_fold_ple: {0}")]
    FoldPle(CudaError),
    #[error("init_k_rot_from_eq_segments: {0}")]
    InitKRot(CudaError),
    #[error("vector_scalar_multiply_ext: {0}")]
    VectorScalarMul(CudaError),
    #[error("sumcheck_mle_round_degenerate: {0}")]
    SumcheckMleRoundDegenerate(CudaError),
    #[error("sumcheck_mle_round: {0}")]
    SumcheckMleRound(CudaError),
    #[error("fold_mle: {0}")]
    FoldMle(CudaError),
    #[error("triangular_fold_mle: {0}")]
    TriangularFoldMle(CudaError),
}

#[derive(Error, Debug)]
pub enum LogupZerocheckError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("Grind: {0}")]
    Grind(GrindError),
    #[error("Round0 eval: {0}")]
    Round0Eval(#[from] Round0EvalError),
    #[error("Interaction eval: {0}")]
    InteractionEval(#[from] InteractionGpuError),
    #[error("Fractional sumcheck: {0}")]
    FractionalSumcheck(#[from] FractionalSumcheckError),
    #[error("Fold PLE: {0}")]
    FoldPle(#[from] FoldPleError),
    #[error("Lambda combinations: {0}")]
    LambdaCombinations(CudaError),
    #[error("Logup combinations: {0}")]
    LogupCombinations(CudaError),
    #[error("Sumcheck: {0}")]
    Sumcheck(#[from] SumcheckError),
    #[error("MLE constraint eval: {0}")]
    MleConstraintEval(KernelError),
    #[error("MLE interaction eval: {0}")]
    MleInteractionEval(KernelError),
    #[error("EqEvalLayers: {0}")]
    EqEvalLayers(KernelError),
    #[error("fold_selectors_round0: {0}")]
    FoldSelectorsRound0(CudaError),
    #[error("interpolate_columns: {0}")]
    InterpolateColumns(KernelError),
    #[error("batch_fold_mle: {0}")]
    BatchFoldMle(CudaError),
    #[error("fill_zero: {0}")]
    FillZero(CudaError),
}

#[derive(Error, Debug)]
pub enum SumcheckError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("sumcheck_mle_round: {0}")]
    SumcheckMleRound(KernelError),
    #[error("fold_mle: {0}")]
    FoldMle(KernelError),
    #[error("batch_ntt_small: {0}")]
    BatchNttSmall(KernelError),
    #[error("reduce_over_x_and_cols: {0}")]
    ReduceOverXAndCols(KernelError),
    #[error("batch_expand_pad_wide: {0}")]
    BatchExpandPadWide(KernelError),
    #[error("fold_ple_from_coeffs: {0}")]
    FoldPleFromCoeffs(KernelError),
}

#[derive(Error, Debug)]
pub enum MerkleTreeError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("poseidon2_compressing_row_hashes error: {0}")]
    CompressingRowHashes(CudaError),
    #[error("poseidon2_compressing_row_hashes_ext error: {0}")]
    CompressingRowHashesExt(CudaError),
    #[error("poseidon2_adjacent_compress_layer [layer={layer}] error: {error}")]
    AdjacentCompressLayer { error: CudaError, layer: usize },
    #[error("query_digest_layers_kernel error: {0}")]
    QueryDigestLayers(CudaError),
    #[error("matrix_get_rows_fp_kernel [matrix_idx={matrix_idx}] error: {error}")]
    MatrixGetRows { error: CudaError, matrix_idx: usize },
}

#[derive(Error, Debug)]
pub enum StackTracesError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("batch_expand_pad_wide error: {0}")]
    BatchExpandPadWide(CudaError),
    #[error("fill_zero error: {0}")]
    FillZero(CudaError),
}

#[derive(Error, Debug)]
pub enum RsCodeMatrixError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("stack_traces_into_expanded error: {0}")]
    StackTraces(StackTracesError),
    #[error("batch_expand_pad error: {0}")]
    BatchExpandPad(CudaError),
    #[error("custom_batch_intt error: {0}")]
    CustomBatchIntt(CudaError),
    #[error("mle_interpolate_stage_2d [step={step}] error: {error}")]
    MleInterpolateStage2d { error: CudaError, step: u32 },
    #[error("bit_rev error: {0}")]
    BitRev(CudaError),
}

#[derive(Error, Debug)]
pub enum WhirProverError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("MerkleTree: {0}")]
    MerkleTree(MerkleTreeError),
    #[error("rs_code_matrix: {0}")]
    RsCodeMatrix(RsCodeMatrixError),
    #[error("whir_algebraic_batch_traces: {0}")]
    AlgebraicBatch(CudaError),
    #[error("transpose_fp_to_fpext_vec: {0}")]
    Transpose(CudaError),
    #[error("mle_interpolate_stage_ext [step={step}]: {error}")]
    MleInterpolate { error: CudaError, step: u32 },
    #[error("custom_batch_intt error: {0}")]
    CustomBatchIntt(CudaError),
    #[error("evals_eq_hypercube: {0}")]
    EvalEq(KernelError),
    #[error("whir_sumcheck_coeff_moments_round [whir_round={whir_round}, round={round}]: {error}")]
    SumcheckMleRound {
        error: CudaError,
        whir_round: usize,
        round: usize,
    },
    #[error("fold_mle [whir_round={whir_round}, round={round}]: {error}")]
    FoldMle {
        error: CudaError,
        whir_round: usize,
        round: usize,
    },
    #[error("split_ext_poly_to_base_col_major_matrix [whir_round={whir_round}]: {error}")]
    SplitExtPoly { error: CudaError, whir_round: usize },
    #[error("batch_expand_pad [whir_round={whir_round}]: {error}")]
    BatchExpandPad { error: CudaError, whir_round: usize },
    #[error("eval_poly_ext_at_point_from_base [whir_round={whir_round}]: {error}")]
    EvalPolyAtPoint {
        error: KernelError,
        whir_round: usize,
    },
    #[error("w_moments_accumulate [whir_round={whir_round}]: {error}")]
    WMomentsAccumulate { error: CudaError, whir_round: usize },
    #[error("Mu grind error: {0}")]
    MuGrind(GrindError),
    #[error("Folding grind error: {0}")]
    FoldingGrind(GrindError),
    #[error("Query phase grind error: {0}")]
    QueryPhaseGrind(GrindError),
}

/// Error type for functions that call CUDA kernels and involve some memcpy operations.
#[derive(Error, Debug)]
pub enum KernelError {
    #[error("CUDA error: {0}")]
    Kernel(#[from] CudaError),
    #[error("Memory copy error: {0}")]
    MemCopy(#[from] MemCopyError),
}
