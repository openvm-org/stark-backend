//! Shared helper for building per-batch `BlockCtx` lists for flat-block-list dispatch.

use crate::cuda::logup_zerocheck::BlockCtx;

/// Builds a flat `Vec<BlockCtx>` from per-AIR block counts, plus the prefix-sum offsets
/// (`air_offsets`) that mark each AIR's range in the flat list.
///
/// Block ordering is the contract for kernels using this list: the kernel does
/// `block_ctxs[blockIdx.x]` directly, so block descriptors are emitted grouped by AIR in input
/// order. Within each AIR, `local_block_idx_x` runs from `0` to `n_blocks - 1`.
///
/// `air_offsets` has length `blocks_per_air.len() + 1`; `air_offsets[i]` is the index of the
/// first block for AIR `i` and `air_offsets.last()` is the total block count. Callers without
/// batched reductions can ignore the second tuple slot.
pub(super) fn build_block_ctxs(
    blocks_per_air: impl IntoIterator<Item = u32>,
) -> (Vec<BlockCtx>, Vec<u32>) {
    let mut block_ctxs: Vec<BlockCtx> = Vec::new();
    let mut air_offsets: Vec<u32> = vec![0];
    for (air_idx, n_blocks) in blocks_per_air.into_iter().enumerate() {
        for local_idx in 0..n_blocks {
            block_ctxs.push(BlockCtx {
                local_block_idx_x: local_idx,
                air_idx: air_idx as u32,
            });
        }
        air_offsets.push(block_ctxs.len() as u32);
    }
    (block_ctxs, air_offsets)
}
