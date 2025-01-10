use serde::{Deserialize, Serialize};

/// In a multi-matrix system, we record a pointer from each matrix to the commitment its stored in
/// as well as the index of the matrix within that commitment.
/// The intended use case is to track the list of pointers for all main trace matrix parts in a single STARK.
///
/// The pointers are in reference to an implicit global list of commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatrixCommitmentPointers {
    /// For each matrix, the pointer
    pub matrix_ptrs: Vec<SingleMatrixCommitPtr>,
}

impl MatrixCommitmentPointers {
    pub fn new(matrix_ptrs: Vec<SingleMatrixCommitPtr>) -> Self {
        Self { matrix_ptrs }
    }
}

/// When a single matrix belong to a multi-matrix commitment in some list of commitments,
/// this pointer identifies the index of the commitment in the list, and then the index
/// of the matrix within that commitment.
///
/// The pointer is in reference to an implicit global list of commitments
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SingleMatrixCommitPtr {
    pub commit_index: usize,
    pub matrix_index: usize,
}

impl SingleMatrixCommitPtr {
    pub fn new(commit_index: usize, matrix_index: usize) -> Self {
        Self {
            commit_index,
            matrix_index,
        }
    }
}
