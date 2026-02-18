use openvm_metal_common::error::{MemCopyError, MetalError};
use thiserror::Error;

use crate::sponge::GrindError;

#[derive(Error, Debug)]
pub enum ProverError {
    #[error("MemCopy: {0}")]
    MemCopy(#[from] MemCopyError),
    #[error("Metal execution: {0}")]
    MetalExecution(MetalError),
    #[error("collapse_strided_matrix: {0}")]
    CollapseStrided(MetalError),
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
    CompressingRowHashes(MetalError),
    #[error("poseidon2_compressing_row_hashes_ext error: {0}")]
    CompressingRowHashesExt(MetalError),
    #[error("poseidon2_adjacent_compress_layer [layer={layer}] error: {error}")]
    AdjacentCompressLayer { error: MetalError, layer: usize },
    #[error("query_digest_layers_kernel error: {0}")]
    QueryDigestLayers(MetalError),
    #[error("matrix_get_rows_fp_kernel [matrix_idx={matrix_idx}] error: {error}")]
    MatrixGetRows {
        error: MetalError,
        matrix_idx: usize,
    },
}

#[derive(Error, Debug)]
pub enum StackTracesError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("batch_expand_pad_wide error: {0}")]
    BatchExpandPadWide(MetalError),
    #[error("fill_zero error: {0}")]
    FillZero(MetalError),
}

#[derive(Error, Debug)]
pub enum RsCodeMatrixError {
    #[error(transparent)]
    MemCopy(#[from] MemCopyError),
    #[error("stack_traces_into_expanded error: {0}")]
    StackTraces(StackTracesError),
    #[error("batch_expand_pad error: {0}")]
    BatchExpandPad(MetalError),
    #[error("custom_batch_intt error: {0}")]
    CustomBatchIntt(MetalError),
    #[error("mle_interpolate_stage_2d [step={step}] error: {error}")]
    MleInterpolateStage2d { error: MetalError, step: u32 },
    #[error("bit_rev error: {0}")]
    BitRev(MetalError),
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
    AlgebraicBatch(MetalError),
    #[error("transpose_fp_to_fpext_vec: {0}")]
    Transpose(MetalError),
    #[error("mle_interpolate_stage_ext [step={step}]: {error}")]
    MleInterpolate { error: MetalError, step: u32 },
    #[error("custom_batch_intt error: {0}")]
    CustomBatchIntt(MetalError),
    #[error("evals_eq_hypercube: {0}")]
    EvalEq(KernelError),
    #[error("whir_sumcheck_coeff_moments_round [whir_round={whir_round}, round={round}]: {error}")]
    SumcheckMleRound {
        error: MetalError,
        whir_round: usize,
        round: usize,
    },
    #[error("fold_mle [whir_round={whir_round}, round={round}]: {error}")]
    FoldMle {
        error: MetalError,
        whir_round: usize,
        round: usize,
    },
    #[error("split_ext_poly_to_base_col_major_matrix [whir_round={whir_round}]: {error}")]
    SplitExtPoly {
        error: MetalError,
        whir_round: usize,
    },
    #[error("batch_expand_pad [whir_round={whir_round}]: {error}")]
    BatchExpandPad {
        error: MetalError,
        whir_round: usize,
    },
    #[error("eval_poly_ext_at_point_from_base [whir_round={whir_round}]: {error}")]
    EvalPolyAtPoint {
        error: KernelError,
        whir_round: usize,
    },
    #[error("w_moments_accumulate [whir_round={whir_round}]: {error}")]
    WMomentsAccumulate {
        error: MetalError,
        whir_round: usize,
    },
    #[error("Mu grind error: {0}")]
    MuGrind(GrindError),
    #[error("Folding grind error: {0}")]
    FoldingGrind(GrindError),
    #[error("Query phase grind error: {0}")]
    QueryPhaseGrind(GrindError),
}

/// Error type for functions that call Metal kernels and involve some memcpy operations.
#[derive(Error, Debug)]
pub enum KernelError {
    #[error("Metal error: {0}")]
    Kernel(#[from] MetalError),
    #[error("Memory copy error: {0}")]
    MemCopy(#[from] MemCopyError),
}
