use openvm_cuda_common::error::{CudaError, MemCopyError};
use thiserror::Error;

use crate::sponge::GrindError;

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
    #[error("mle_interpolate_stage_2d [step={step}] error: {error}")]
    MleInterpolateStage2d { error: CudaError, step: u32 },
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
    #[error("evals_eq_hypercube: {0}")]
    EvalEq(KernelError),
    #[error("whir_sumcheck_mle_round [whir_round={whir_round}, round={round}]: {error}")]
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
    #[error("mle_evals_to_coeffs_inplace [whir_round={whir_round}]: {error}")]
    MleEvalToCoeff {
        error: KernelError,
        whir_round: usize,
    },
    #[error("batch_expand_pad [whir_round={whir_round}]: {error}")]
    BatchExpandPad { error: CudaError, whir_round: usize },
    #[error("eval_poly_ext_at_point_from_base [whir_round={whir_round}]: {error}")]
    EvalPolyAtPoint {
        error: KernelError,
        whir_round: usize,
    },
    #[error("eq_hypercube_stage_ext [whir_round={whir_round}, step={step}]: {error}")]
    EqHypercubeStageExt {
        error: CudaError,
        whir_round: usize,
        step: u32,
    },
    #[error("batch_eq_hypercube_stage [whir_round={whir_round}, step={step}]: {error}")]
    BatchEqHypercubeStage {
        error: CudaError,
        whir_round: usize,
        step: u32,
    },
    #[error("w_evals_accumulate [whir_round={whir_round}]: {error}")]
    WEvalsAccumulate { error: CudaError, whir_round: usize },
    #[error("Grind error: {0}")]
    Grind(GrindError),
}

/// Error type for functions that call CUDA kernels and involve some memcpy operations.
#[derive(Error, Debug)]
pub enum KernelError {
    #[error("CUDA error: {0}")]
    Kernel(#[from] CudaError),
    #[error("Memory copy error: {0}")]
    MemCopy(#[from] MemCopyError),
}
