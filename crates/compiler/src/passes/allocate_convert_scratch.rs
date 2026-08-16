//! Scratch sizing for [`SSAOpCode::ConvertLayout`] (Phase B.2).
//!
//! `layout_infer` emits a placeholder shared scratch [`SSAOpCode::Alloc`]
//! alongside every `ConvertLayout` with shape `[0]`. This pass walks each
//! `ConvertLayout`, computes the scratch bytes required from the
//! `(src, dst)` layout+space pair, and updates the scratch buffer's
//! shape. Everything downstream — `plan_shared_mem`'s liveness packer,
//! `insert_sync`'s dirty-shared walk, codegen's `__shared__` declarations
//! — then treats the scratch buffer as any other shared `Alloc`. No
//! private pool; no special case.
//!
//! The sizing function [`convert_scratch_bytes`] is a **pure function of
//! the layout pair** and runs the same [`best_decomposition`] that
//! `gen_convert` re-runs at codegen, so the two always agree: scratch is
//! a full tile exactly when [`Strategy::Bounce`] wins, 0 otherwise.
//! (A future hybrid Strategy C could use fewer bytes than a full tile;
//! until then the tile is exact, not just an upper bound.)
//!
//! Sizing per `(src.space, dst.space)`:
//!
//! - **`Register` → `Register`**: 0 bytes unless [`best_decomposition`] picks [`Strategy::Bounce`]
//!   (warp-crossing composite, or a shuffle whose round count prices worse than the bounce); then a
//!   full tile sized for `phys_len(dst) * elem_bytes(src)`.
//! - **`Register` → `Shared`** and **`Shared` → `Register`**: 0 bytes — one endpoint is already
//!   shared, no intermediate scratch.
//! - **`Shared` → `Shared`**: 0 bytes — codegen's arm is a single read-then-write loop through a
//!   register temporary, no shared staging.
//! - Anything involving `Global`: 0 bytes — direct load/store, no bounce through shared.

use crate::{
    ir::SizeExpr,
    kernel_ir::{AddressSpace, BufferDecl, KirProgram, LinearLayout, SSAOpCode},
    passes::{
        convert_decompose::{best_decomposition, Strategy},
        layout_cost::ConversionCostModel,
    },
};

/// Fill in every `ConvertLayout`'s scratch buffer size from its
/// `(src, dst)` layout pair. Buffers whose scratch requirement is zero
/// keep the placeholder shape `[0]` inserted by [`layout_infer`] and get
/// filtered out of [`plan_shared_mem`]'s liveness (no write, no
/// allocation).
pub fn allocate_convert_scratch(p: &mut KirProgram) {
    // Collect (scratch_bufid, bytes, layout) tuples; apply after the walk so we
    // don't hold a mutable borrow of `p.buffers` while iterating kernels.
    let mut updates: Vec<(crate::kernel_ir::BufId, usize, Option<LinearLayout>)> = Vec::new();
    for k in &p.kernels {
        walk_stmts(&p.buffers, k, &k.grid.block.body, k.block, &mut updates);
    }
    for (buf, bytes, layout) in updates {
        let decl = &mut p.buffers[buf.0 as usize];
        let elem_bytes = decl.elem.size_bytes().max(1);
        let elems = bytes.div_ceil(elem_bytes);
        decl.shape = vec![SizeExpr::from(elems)];
        if let Some(ll) = layout {
            decl.layout = Some(ll);
        }
    }
}

fn walk_stmts(
    buffers: &[BufferDecl],
    kernel: &crate::kernel_ir::Kernel,
    stmts: &[crate::kernel_ir::SSANode],
    block: usize,
    updates: &mut Vec<(crate::kernel_ir::BufId, usize, Option<LinearLayout>)>,
) {
    for &sid in stmts {
        let op = kernel.op(sid);
        match &op.opcode {
            SSAOpCode::ConvertLayout {
                dst,
                src,
                scratch,
                map,
            } => {
                let src_decl = &buffers[src.0 as usize];
                let dst_decl = &buffers[dst.0 as usize];
                let elem_bytes = src_decl.elem.size_bytes().max(1);
                // Mirror gen_convert: the destination's effective layout
                // folds in the op's map (`dst_eff = map ∘ ld`).
                let kb = map.bases.len();
                let ld = dst_decl
                    .layout
                    .clone()
                    .unwrap_or_else(|| LinearLayout::identity(kb));
                let dst_eff = map.compose(&ld);
                let bytes = convert_scratch_bytes(
                    src_decl.layout.as_ref(),
                    Some(&dst_eff),
                    src_decl.space,
                    dst_decl.space,
                    dst_decl.len(),
                    block,
                    elem_bytes,
                );
                if bytes > 0 {
                    // B.1 (b): pick a bank-conflict-minimizing swizzle for
                    // the scratch buffer via `choose_shared_layout` given
                    // the src and dst accesses. Falls back to row-major
                    // identity when the src/dst layouts aren't available.
                    let layout =
                        pick_scratch_layout(src_decl, dst_decl, block, dst_decl.len(), elem_bytes);
                    updates.push((*scratch, bytes, layout));
                }
            }
            SSAOpCode::Loop { .. } => walk_stmts(buffers, kernel, &op.block.body, block, updates),
            _ => {}
        }
    }
}

/// Pick a bank-conflict-minimizing shared-memory layout for the scratch
/// buffer given the two accesses that touch it: the writer (composed
/// from `src.layout`) and the reader (composed from `dst.layout`). Uses
/// B.6.2's [`choose_shared_layout`] to score candidate partitions and
/// projects the winner to a [`LinearLayout`] via
/// [`shared_swizzle::to_linear_layout`].
///
/// Returns `None` when either layout isn't available — the caller
/// leaves the scratch's `layout` as-is (defaults to identity at
/// codegen).
fn pick_scratch_layout(
    src_decl: &BufferDecl,
    dst_decl: &BufferDecl,
    block: usize,
    dst_len: usize,
    elem_bytes: usize,
) -> Option<LinearLayout> {
    let src_layout = src_decl.layout.as_ref()?;
    let dst_layout = dst_decl.layout.as_ref()?;
    let output_dim = crate::passes::utils::ceil_log2(dst_len.max(1));
    let src_thread_bits = src_layout.phys_partition(block).thread_bits();
    let dst_thread_bits = dst_layout.phys_partition(block).thread_bits();
    use crate::passes::shared_swizzle::{choose_shared_layout, to_linear_layout, Access};
    let accesses = vec![
        Access {
            layout: src_layout.clone(),
            loop_iters: Vec::new(),
            thread_bits: src_thread_bits,
        },
        Access {
            layout: dst_layout.clone(),
            loop_iters: Vec::new(),
            thread_bits: dst_thread_bits,
        },
    ];
    let cost = crate::passes::layout_cost::ConversionCostModel::default();
    let sh = choose_shared_layout(&cost, &accesses, output_dim, elem_bytes, block);
    Some(to_linear_layout(&sh))
}

/// Scratch bytes required for a `ConvertLayout` from a `src` in
/// `src_space` to a `dst` in `dst_space`. Pure function.
///
/// `dst_len` is the logical element count of `dst` (used only for the
/// `Register`-touching arms that stage through a tile-sized scratch).
/// `block` is the kernel's warp-aligned `blockDim.x`. `elem_bytes` is
/// the source element width.
pub fn convert_scratch_bytes(
    src: Option<&LinearLayout>,
    dst: Option<&LinearLayout>,
    src_space: AddressSpace,
    dst_space: AddressSpace,
    dst_len: usize,
    block: usize,
    elem_bytes: usize,
) -> usize {
    match (src_space, dst_space) {
        (AddressSpace::Register, AddressSpace::Register) => {
            // Run the same decomposition `gen_convert` re-runs at codegen
            // so the two agree exactly: Copy / Slot / Shuffle (including
            // the multi-round pull path) need no shared traffic; only a
            // winning Bounce stages through a full tile.
            //
            // `logical_bits = log2(dst_len)`: the codomain of both src and dst
            // layouts is the buffer's *logical* space, not the phys width of
            // its bases. Under Gap 2 (Phase A) and B.1's producer-layout
            // convention, layouts have `phys_bits` bases (some zero — replicated
            // inputs), but their image is always `logical_bits` wide.
            let src_l = src.cloned().unwrap_or_else(|| identity_for(dst_len));
            let dst_l = dst.cloned().unwrap_or_else(|| identity_for(dst_len));
            let logical_bits = crate::passes::utils::ceil_log2(dst_len.max(1));
            let cost = ConversionCostModel::default();
            let dec = best_decomposition(&cost, &src_l, &dst_l, block, logical_bits, &[]);
            match dec.strategy {
                Strategy::Bounce { .. } => tile_scratch_bytes(dst_len, block, elem_bytes),
                _ => 0,
            }
        }
        // Shared→Shared is a single read-then-write loop through a
        // register temporary in codegen — no shared staging buffer.
        (AddressSpace::Shared, AddressSpace::Shared) => 0,
        _ => 0,
    }
}

fn identity_for(len: usize) -> LinearLayout {
    LinearLayout::identity(crate::passes::utils::ceil_log2(len.max(1)))
}

fn tile_scratch_bytes(dst_len: usize, block: usize, elem_bytes: usize) -> usize {
    // Every physical thread holds up to `ceil(dst_len / block)` elements
    // across the CTA; the bounce stages the whole tile through shared
    // once (store → sync → load). Sized for `block * elems_per_thread *
    // elem_bytes`, which equals `dst_len * elem_bytes` when dst_len is a
    // multiple of block and otherwise pads up.
    let elems_per_thread = dst_len.div_ceil(block.max(1));
    let elems = elems_per_thread * block.max(1);
    elems * elem_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::ScalarType, kernel_ir::LinearLayout};

    #[test]
    fn reg_to_reg_warp_identity_needs_no_scratch() {
        // Both layouts identity(9) — composite is identity, warp column
        // is identity, no shared bounce needed.
        let l = LinearLayout::identity(9);
        let bytes = convert_scratch_bytes(
            Some(&l),
            Some(&l),
            AddressSpace::Register,
            AddressSpace::Register,
            1 << 9,
            256,
            ScalarType::BabyBear.size_bytes(),
        );
        assert_eq!(bytes, 0);
    }

    #[test]
    fn reg_to_reg_warp_crossing_needs_full_tile() {
        // dst mixes warp bit 5 into lane bit 0: composite fails
        // is_warp_column_identity, so a shared bounce is required.
        let src = LinearLayout::identity(9);
        let mut dst = LinearLayout::identity(9);
        dst.bases[0] = 1 << 5;
        dst.bases[5] = 1;
        let bytes = convert_scratch_bytes(
            Some(&src),
            Some(&dst),
            AddressSpace::Register,
            AddressSpace::Register,
            1 << 9,
            256,
            ScalarType::BabyBear.size_bytes(),
        );
        // 512 elements × 4 bytes = 2048.
        assert_eq!(bytes, 2048);
    }

    #[test]
    fn reg_to_shared_needs_no_scratch() {
        // Reg→Shared writes directly to dst; scratch stays 0.
        let l = LinearLayout::identity(8);
        let bytes = convert_scratch_bytes(
            Some(&l),
            Some(&l),
            AddressSpace::Register,
            AddressSpace::Shared,
            1 << 8,
            256,
            ScalarType::BabyBear.size_bytes(),
        );
        assert_eq!(bytes, 0);
    }

    #[test]
    fn shared_to_reg_needs_no_scratch() {
        let l = LinearLayout::identity(8);
        let bytes = convert_scratch_bytes(
            Some(&l),
            Some(&l),
            AddressSpace::Shared,
            AddressSpace::Register,
            1 << 8,
            256,
            ScalarType::BabyBear.size_bytes(),
        );
        assert_eq!(bytes, 0);
    }

    #[test]
    fn shared_to_shared_needs_no_scratch() {
        // Codegen's Shared→Shared arm is a single read-then-write loop
        // through a register temporary — no shared staging buffer.
        let l = LinearLayout::identity(8);
        let bytes = convert_scratch_bytes(
            Some(&l),
            Some(&l),
            AddressSpace::Shared,
            AddressSpace::Shared,
            1 << 8,
            256,
            ScalarType::BabyBear.size_bytes(),
        );
        assert_eq!(bytes, 0);
    }

    #[test]
    fn multi_round_shuffle_composite_needs_no_scratch() {
        // Lane bit 0 ↔ slot bit 5 mix: singular lane block, non-constant
        // sender slot — the multi-round pull path emits it without a
        // bounce, so scratch stays 0.
        let src = LinearLayout::identity(6);
        let dst = LinearLayout {
            bases: vec![32, 2, 4, 8, 16, 1],
            offset: 0,
        };
        let bytes = convert_scratch_bytes(
            Some(&src),
            Some(&dst),
            AddressSpace::Register,
            AddressSpace::Register,
            1 << 6,
            32,
            ScalarType::BabyBear.size_bytes(),
        );
        assert_eq!(bytes, 0);
    }

    #[test]
    fn global_touching_needs_no_scratch() {
        let l = LinearLayout::identity(8);
        assert_eq!(
            convert_scratch_bytes(
                Some(&l),
                Some(&l),
                AddressSpace::Global,
                AddressSpace::Register,
                1 << 8,
                256,
                ScalarType::BabyBear.size_bytes(),
            ),
            0
        );
        assert_eq!(
            convert_scratch_bytes(
                Some(&l),
                Some(&l),
                AddressSpace::Register,
                AddressSpace::Global,
                1 << 8,
                256,
                ScalarType::BabyBear.size_bytes(),
            ),
            0
        );
    }

    #[test]
    fn tile_smaller_than_lanes_never_crosses_warp() {
        // dst_len=8, block=32 — the whole logical space fits within one
        // warp's lanes. Any distributed lane-only permutation classifies
        // as pure shuffle (warp column trivially identity — there are no
        // warp columns), and scratch stays 0.
        let src = LinearLayout::identity(3);
        let mut dst = LinearLayout::identity(3);
        dst.bases[0] = 1 << 2;
        dst.bases[2] = 1;
        let bytes = convert_scratch_bytes(
            Some(&src),
            Some(&dst),
            AddressSpace::Register,
            AddressSpace::Register,
            8,
            32,
            ScalarType::BabyBear.size_bytes(),
        );
        assert_eq!(bytes, 0);
    }

    #[test]
    fn allocate_pass_updates_scratch_shape() {
        use crate::{
            ir::{IRBuilder, ScalarType},
            passes::{layout_infer, utils::test_util::lowered},
        };
        // Own-index tile → registers, no shared mirror or bounce. Every
        // ConvertLayout in this shape needs 0-byte scratch, but the
        // pass must still walk them and leave them at shape=[0].
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let tile = b.compute(t, |b, j| {
                let tc = b.const_u32(t as u32);
                let base = b.mul(i, tc);
                let ix = b.add(base, j);
                b.index(a, &[ix])
            });
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    let x = b.index(tile, &[j]);
                    b.mul(x, x)
                })
            })
        });
        let module = b.finish("scratch_test", body);
        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        allocate_convert_scratch(&mut kprog);
        // Every scratch buffer stays at shape [0] since Direct/View
        // paths pay no shared traffic.
        for buf in &kprog.buffers {
            if buf.name.contains("_cs") {
                assert_eq!(
                    buf.shape,
                    vec![SizeExpr::from(0usize)],
                    "scratch {} should be 0-byte",
                    buf.name
                );
            }
        }
    }
}
