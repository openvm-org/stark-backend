use std::{cmp::Reverse, collections::HashMap};

use cuda_kernels::poseidon2::*;
use cuda_utils::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::CudaError,
};
use itertools::Itertools;
use openvm_stark_backend::prover::hal::MatrixDimensions;
use p3_symmetric::Hash;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::debug_span;

use crate::{base::DeviceMatrix, gpu_device::GpuDevice, lde::GpuLde, prelude::F};

const DIGEST_WIDTH: usize = 8;
type H = [F; DIGEST_WIDTH];

pub struct GpuMerkleTree<LDE: GpuLde> {
    pub(crate) leaves: Vec<LDE>,
    pub(crate) digest_layers: Vec<DeviceBuffer<H>>,
}

impl<LDE: GpuLde> GpuMerkleTree<LDE> {
    pub fn new(leaves: Vec<LDE>, _gpu_device: &GpuDevice) -> Result<Self, CudaError> {
        assert!(!leaves.is_empty(), "No matrices given?");

        let mut matrices_largest_first = leaves
            .iter()
            .sorted_by_key(|m| Reverse(m.height()))
            .peekable();

        let max_height = matrices_largest_first.peek().unwrap().height();

        let digests = DeviceBuffer::<H>::with_capacity(max_height);
        {
            let tallest_matrices = matrices_largest_first
                .peeking_take_while(|m| m.height() == max_height)
                .map(|m| m.take_lde(m.height()))
                .collect_vec();

            Self::hash_matrices(&digests, &tallest_matrices).unwrap();
        }
        let mut digest_layers: Vec<DeviceBuffer<H>> = vec![digests];
        loop {
            let prev_layer = digest_layers.last().unwrap();
            if prev_layer.len() == 1 {
                break;
            }
            let next_layer_len = prev_layer.len() / 2;
            let next_layer = DeviceBuffer::<H>::with_capacity(next_layer_len);
            let is_inject = {
                let matrices_to_inject = matrices_largest_first
                    .peeking_take_while(|m| m.height().next_power_of_two() == next_layer_len)
                    .map(|m| m.take_lde(m.height()))
                    .collect_vec();

                let has_matrices = !matrices_to_inject.is_empty();
                if has_matrices {
                    Self::hash_matrices(&next_layer, &matrices_to_inject).unwrap();
                }
                has_matrices
            };

            Self::hash_compress(&next_layer, prev_layer, is_inject).unwrap();

            digest_layers.push(next_layer);
        }

        Ok(Self {
            leaves,
            digest_layers,
        })
    }

    fn hash_matrices(out: &DeviceBuffer<H>, matrices: &[DeviceMatrix<F>]) -> Result<(), CudaError> {
        // For poseidon2_rows_p3_multi we need:
        // matrices_ptr - array of pointers to matrices
        // matrices_col - array of column sizes
        // matrices_row - array of row sizes
        let matrices_ptr: Vec<u64> = matrices
            .iter()
            .map(|m| m.buffer().as_ptr() as u64)
            .collect();
        let matrices_col: Vec<u64> = matrices.iter().map(|m| m.width() as u64).collect();
        let matrices_row: Vec<u64> = matrices.iter().map(|m| m.height() as u64).collect();

        let d_matrices_ptr = matrices_ptr.to_device().unwrap();
        let d_matrices_col = matrices_col.to_device().unwrap();
        let d_matrices_row = matrices_row.to_device().unwrap();

        unsafe {
            poseidon2_rows_p3_multi(
                out,
                &d_matrices_ptr,
                &d_matrices_col,
                &d_matrices_row,
                matrices_row[0],
                matrices.len() as u64,
            )
        }
    }

    fn hash_compress(
        out: &DeviceBuffer<H>,
        prev_layer: &DeviceBuffer<H>,
        is_inject: bool,
    ) -> Result<(), CudaError> {
        unsafe { poseidon2_compress(out, prev_layer, out.len() as u32, is_inject) }
    }

    pub fn root(&self) -> Hash<F, F, DIGEST_WIDTH> {
        let root = self.digest_layers.last().unwrap();
        assert_eq!(root.len(), 1, "Only one root is supported");
        root.to_host().unwrap()[0].into()
    }

    #[allow(clippy::type_complexity)]
    pub fn open_batch_at_multiple_indices(
        &self,
        indices: &[usize],
    ) -> Result<Vec<(Vec<Vec<F>>, Vec<H>)>, ()> {
        let max_height = self.leaves.iter().map(|m| m.height()).max().unwrap();
        let log_max_height = log2_strict_usize(max_height);

        // open all indices of one leaf at once to reduce gpu peak memory
        // the structure of openings: [leaf_index][point_index][column_index]
        let mut openings_indexed_by_leaf_idx = self
            .leaves
            .iter()
            .map(|matrix| {
                let openings_per_matrix = debug_span!("read rows").in_scope(|| {
                    let log_matrix_height = log2_ceil_usize(matrix.height());
                    let bits_reduced = log_max_height - log_matrix_height;

                    let mut unique_map = HashMap::new();
                    let mut unique_indices = Vec::new();

                    let reduced_indices = indices
                        .iter()
                        .map(|&index| {
                            let reduced_idx = index >> bits_reduced;
                            if !unique_map.contains_key(&reduced_idx) {
                                unique_map.insert(reduced_idx, unique_map.len());
                                unique_indices.push(reduced_idx);
                            }
                            reduced_idx
                        })
                        .collect_vec();

                    let unique_openings_vec =
                        matrix.get_lde_rows(&unique_indices).to_host().unwrap();
                    let unique_openings_rows =
                        unique_openings_vec.chunks(matrix.width()).collect_vec();

                    reduced_indices
                        .iter()
                        .map(|idx| Ok(unique_openings_rows[unique_map[idx]].to_vec()))
                        .collect::<Result<Vec<_>, ()>>()
                })?;

                Ok(openings_per_matrix)
            })
            .collect::<Result<Vec<_>, ()>>()?;

        // convert openings' structure to [point_index][leaf_index][column_index]
        let openings_indexed_by_point_idx = indices
            .iter()
            .rev()
            .map(|_| {
                // For each point (in reverse)
                openings_indexed_by_leaf_idx
                    .iter_mut()
                    .map(|openings_per_matrix| openings_per_matrix.pop().unwrap())
                    .collect_vec()
            })
            .collect_vec()
            .into_iter()
            .rev()
            .collect_vec();

        let proofs: Vec<Vec<H>> = debug_span!("read digests").in_scope(|| {
            let query_indices = indices
                .iter()
                .flat_map(|index| {
                    (0..log_max_height)
                        .map(|i| {
                            ((index >> i) ^ 1) as u64 //sibling_index
                        })
                        .collect::<Vec<u64>>()
                })
                .collect::<Vec<_>>();

            let num_query = indices.len();
            let num_layer = log_max_height;

            let all_digests = Self::query_digest_layers(
                &self.digest_layers[..log_max_height],
                &query_indices,
                num_query,
                num_layer,
            );
            assert_eq!(num_layer + 1, self.digest_layers.len());
            assert_eq!(all_digests.len(), query_indices.len());

            all_digests
                .chunks(num_layer)
                .map(|layers_digest| Ok(layers_digest.to_vec()))
                .collect::<Result<Vec<_>, ()>>()
        })?;

        Ok(openings_indexed_by_point_idx
            .into_iter()
            .zip(proofs)
            .collect::<Vec<(_, _)>>())
    }

    fn query_digest_layers(
        digest_layers: &[DeviceBuffer<H>],
        indices: &[u64],
        num_query: usize,
        num_layer: usize,
    ) -> Vec<H> {
        assert_eq!(num_layer, digest_layers.len());
        assert_eq!(num_layer * num_query, indices.len());

        let digest_layers_ptr = digest_layers
            .iter()
            .map(|layer| layer.as_ptr() as u64)
            .collect_vec();
        let digest_layers_ptr_buf = digest_layers_ptr.to_device().unwrap();

        let d_indices = indices.to_device().unwrap();
        let digest_buffer = DeviceBuffer::<H>::with_capacity(indices.len());

        unsafe {
            query_digest_layers_kernel(
                &digest_buffer,
                &digest_layers_ptr_buf,
                &d_indices,
                num_query.try_into().unwrap(),
                num_layer.try_into().unwrap(),
            )
            .unwrap();
        }
        digest_buffer.to_host().unwrap()
    }

    pub fn get_max_height(&self) -> usize {
        self.leaves.iter().map(|m| m.height()).max().unwrap()
    }
}
