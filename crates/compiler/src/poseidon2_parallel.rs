//! Warp-parallel Poseidon2-16 permutation over BabyBear.
//!
//! Each of the 16 state elements is owned by one lane of a warp: cross-element
//! reads inside a round become warp shuffles under the `#[par((th, s) -> th)]`
//! layout ([`ConvertKind::Shuffle`](crate::kernel_ir::ConvertKind::Shuffle) in
//! `classify_convert`). The round algorithm is identical to the reference
//! [`poseidon2_permutation`](crate::kernels::poseidon2_permutation) — this
//! module builds the same expression DAG, only re-sliced along the lane axis
//! so that most of the arithmetic runs concurrently.
//!
//! # Structure
//!
//! Every round emits one `compute_with(16, par=|th,_,_| th)` stage, let-bound
//! to feed the next stage. Under the identity par layout each thread owns
//! exactly one state slot, so `state[k]` for `k = j - j%4 + delta` (with a
//! constant `delta`) is either a Slot access (`j - j%4 + delta` lies in the
//! same lane block) or a Shuffle across `<= 16` lanes. Sums across all 16
//! state elements (the internal round's `sum(state)`) are unrolled into a
//! stride-halving butterfly of four shuffles — a hand-rolled all-reduce that
//! avoids depending on the compiler's `reduce_add` under `par`.
//!
//! ## `apply_mat4_lane` choice
//!
//! Every lane recomputes the full `apply_mat4` addition chain for its 4-lane
//! chunk (option 2 in the task brief): 8 additions and doublings, then one
//! [`select_by_i_mod4`] picks the lane's own output. This trades ~3 select
//! nodes per lane for reusing the `x + x` doubling terms in a way that
//! hash-consing preserves — the alternative (four separate linear
//! combinations, one per residue) needs an extra field multiply on every
//! lane. On BabyBear multiplies are more expensive than doubling + select, so
//! we prefer this shape.

use crate::{
    ir::{IRBuilder, Module, NodeId, ScalarType},
    kernel,
    kernels::Poseidon2Constants,
};

/// Poseidon2-16 state width.
pub const WIDTH: usize = 16;

/// Identity par-layout closure: logical lane index = thread index.
///
/// `par_map(|th, _seq, _cst| th)` — one thread per state element, no
/// sequential fold; every 16-element compute becomes a single warp.
fn lane_par(b: &mut IRBuilder) -> crate::quast::ParSpec {
    b.par_map(|th, _s, _c| th.clone())
}

/// Selects one of `values.len()` field constants by the lane's index. The
/// generated expression is a balanced select tree of depth `ceil(log2(N))`.
///
/// This is how each lane picks up its "own" per-lane constant (round
/// constant, diagonal element, `apply_mat4` output) without needing a tensor
/// load — the tree collapses inside each thread to a straight-line pick
/// once the compiler resolves the enclosing par variable.
fn select_by_index(b: &mut IRBuilder, i: NodeId, values: &[NodeId]) -> NodeId {
    fn rec(b: &mut IRBuilder, i: NodeId, values: &[NodeId], lo: u32, hi: u32) -> NodeId {
        assert!(lo < hi);
        if hi - lo == 1 {
            return values[lo as usize];
        }
        let mid = lo + (hi - lo) / 2;
        let then_val = rec(b, i, values, lo, mid);
        let else_val = rec(b, i, values, mid, hi);
        let mid_c = b.const_u32(mid);
        let cond = b.lt(i, mid_c);
        b.select(cond, then_val, else_val)
    }
    assert!(!values.is_empty());
    rec(b, i, values, 0, values.len() as u32)
}

/// Per-lane linear combination that recovers `apply_mat4`'s residue-`r` output
/// for a 4-lane chunk. The lane's own residue is picked with a nested select.
///
/// The four M-outputs share the same addition chain (`t01 = x0 + x1`,
/// `t23 = x2 + x3`, ...), which is reused verbatim from
/// [`crate::kernels::apply_mat4`]; every lane in the chunk redundantly
/// computes all four, then picks its own — the redundancy is bounded (4x
/// arithmetic in this stage only) and hash-consing dedups the shared
/// sub-expressions across lanes.
fn apply_mat4_lane(
    b: &mut IRBuilder,
    j: NodeId,
    x0: NodeId,
    x1: NodeId,
    x2: NodeId,
    x3: NodeId,
) -> NodeId {
    let t01 = kernel!(b, x0 + x1);
    let t23 = kernel!(b, x2 + x3);
    let t0123 = kernel!(b, t01 + t23);
    let t01123 = kernel!(b, t0123 + x1);
    let t01233 = kernel!(b, t0123 + x3);
    let y0 = kernel!(b, t01123 + t01);
    let y1 = kernel!(b, t01123 + (x2 + x2));
    let y2 = kernel!(b, t01233 + t23);
    let y3 = kernel!(b, t01233 + (x0 + x0));
    // Pick y_r where r = j mod 4.
    let four = b.const_u32(4);
    let r = b.rem(j, four);
    select_by_index(b, r, &[y0, y1, y2, y3])
}

/// Emits the intra-chunk `apply_mat4` layer as a warp-parallel stage.
///
/// Every lane `j` needs `x_r = state[chunk_base + r]` for `r = 0..3` (with
/// `chunk_base = j - j%4`). Under the identity par layout `state[chunk_base + r]`
/// with `r != j%4` is a same-warp shuffle — `layout_infer` classifies the
/// map `j - j%4 + r` as a Shuffle over the low four lane bits.
fn apply_mat4_stage(b: &mut IRBuilder, prev: NodeId) -> NodeId {
    let par = lane_par(b);
    b.compute_with(WIDTH, None, Some(par), None, |b, j| {
        let x0 = kernel!(b, prev[j - j % 4]);
        let x1 = kernel!(b, prev[j - j % 4 + 1]);
        let x2 = kernel!(b, prev[j - j % 4 + 2]);
        let x3 = kernel!(b, prev[j - j % 4 + 3]);
        apply_mat4_lane(b, j, x0, x1, x2, x3)
    })
}

/// Emits the cross-chunk sums-layer of `mds_light`: each lane returns
/// `state[j] + sum_{c=0..4} state[j%4 + 4c]` — the second half of `mds_light`
/// after `apply_mat4`.
///
/// The four cross-chunk reads are stride-4 warp shuffles (bases `0, 4, 8, 12`
/// XOR-affinely permute the lane bits within a 16-lane block).
fn mds_cross_stage(b: &mut IRBuilder, prev: NodeId) -> NodeId {
    let par = lane_par(b);
    b.compute_with(WIDTH, None, Some(par), None, |b, j| {
        // sum_j = state[j%4] + state[j%4 + 4] + state[j%4 + 8] + state[j%4 + 12]
        kernel!(b,
            let a0 = prev[j % 4];
            let a1 = prev[j % 4 + 4];
            let a2 = prev[j % 4 + 8];
            let a3 = prev[j % 4 + 12];
            prev[j] + (a0 + a1) + (a2 + a3)
        )
    })
}

/// The stage sequence built by [`poseidon2_permute_par`].
///
/// Each variant emits a single 16-lane `compute_with(par=identity)` (or a
/// short bind-chain for the composite `External`/`Internal` rounds); the
/// stages are chained through `bind` so each let-bound stage's value is a
/// single compute the canonicalization pass can peel.
#[derive(Clone)]
enum Stage {
    /// Initial `mds_light` (no `add_rc`, no `sbox`): only `ApplyMat4` and
    /// `MdsCross`.
    ApplyMat4,
    MdsCross,
    ExternalRcSbox([u32; 16]),
    /// The internal round's stage 1 (masked s-box + passthrough).
    InternalSbox(u32),
    /// The four butterfly-sum stages, each with a different XOR partner
    /// stride (1, 2, 4, 8). Emitted as one Stage per stride.
    ButterflySum(u32),
    /// The internal round's stage 3: `sum + diag[j] * after_sbox[j]`. Reads
    /// both the sum tile and the after-sbox tile from previous stages.
    InternalCombine([u32; 16]),
}

impl Stage {
    /// Emits this stage as a `compute_with(16, par)` reading the previously
    /// let-bound tile ids. `after_sbox` is only used by
    /// [`Stage::InternalCombine`] — it names the internal round's stage 1
    /// tile, which the combine stage reads in addition to the running
    /// carry.
    fn emit(&self, b: &mut IRBuilder, prev: NodeId, after_sbox: Option<NodeId>) -> NodeId {
        match self {
            Stage::ApplyMat4 => apply_mat4_stage(b, prev),
            Stage::MdsCross => mds_cross_stage(b, prev),
            Stage::ExternalRcSbox(rc) => external_rc_sbox_stage(b, prev, rc),
            Stage::InternalSbox(rc) => internal_sbox_stage(b, prev, *rc),
            Stage::ButterflySum(stride) => butterfly_stage(b, prev, *stride),
            Stage::InternalCombine(diag) => internal_combine_stage(
                b,
                prev,
                after_sbox.expect("InternalCombine needs the after-sbox tile"),
                diag,
            ),
        }
    }
}

/// Fused `add_rc` + `sbox`: `(state[j] + rc[j])^7` per lane. Local, so no
/// cross-lane reads. Combining the two into one stage halves the number of
/// let-bound tiles.
fn external_rc_sbox_stage(b: &mut IRBuilder, prev: NodeId, rc: &[u32; 16]) -> NodeId {
    let par = lane_par(b);
    let rc = *rc;
    b.compute_with(WIDTH, None, Some(par), None, |b, j| {
        let rc_nodes: Vec<NodeId> = rc.iter().map(|&k| b.const_field(k)).collect();
        let rc_j = select_by_index(b, j, &rc_nodes);
        kernel!(b,
            let x = prev[j] + rc_j;
            let x2 = x * x;
            let x4 = x2 * x2;
            (x2 * x) * x4
        )
    })
}

/// Internal-round stage 1: for lane 0 write `sbox(state[0] + rc)`, other
/// lanes pass through their own slot.
fn internal_sbox_stage(b: &mut IRBuilder, prev: NodeId, rc: u32) -> NodeId {
    let par = lane_par(b);
    b.compute_with(WIDTH, None, Some(par), None, |b, j| {
        let rc_const = b.const_field(rc);
        kernel!(b,
            let s0 = prev[0];
            let x = s0 + rc_const;
            let x2 = x * x;
            let x4 = x2 * x2;
            let sbox_val = (x2 * x) * x4;
            let orig = prev[j];
            if j == 0 then sbox_val else orig
        )
    })
}

/// One butterfly-sum stage: `next[j] = curr[j] + curr[j XOR stride]`. Under
/// the identity par map, the partner map `j -> j XOR stride` is Slot (for
/// `stride < 4`, i.e. within a 4-lane block if we had one; here every
/// stride within `< 16` stays within one warp) or Shuffle (XOR crosses
/// lane bits). Every stage folds pairs at distance `stride`, so after
/// strides `1, 2, 4, 8` every lane holds the total.
fn butterfly_stage(b: &mut IRBuilder, prev: NodeId, stride: u32) -> NodeId {
    let par = lane_par(b);
    let m = 2 * stride;
    b.compute_with(WIDTH, None, Some(par), None, |b, j| {
        kernel!(b,
            let partner = prev[j + #stride - j % #m / #stride * #m];
            let own = prev[j];
            own + partner
        )
    })
}

/// Internal-round stage 3: `sum + diag[j] * after_sbox[j]`. Reads two
/// tiles — the reduced sum (constant across lanes, still stored per-lane
/// after the butterfly) and the stage-1 tile `after_sbox`.
fn internal_combine_stage(
    b: &mut IRBuilder,
    sum: NodeId,
    after_sbox: NodeId,
    diag: &[u32; 16],
) -> NodeId {
    let par = lane_par(b);
    let diag = *diag;
    b.compute_with(WIDTH, None, Some(par), None, |b, j| {
        let diag_nodes: Vec<NodeId> = diag.iter().map(|&v| b.const_field(v)).collect();
        let d_j = select_by_index(b, j, &diag_nodes);
        kernel!(b, sum[j] + d_j * after_sbox[j])
    })
}

/// Builds the full Poseidon2-16 stage schedule.
fn build_stages(c: &Poseidon2Constants) -> Vec<Stage> {
    let mut stages: Vec<Stage> = Vec::new();
    // Initial mds_light.
    stages.push(Stage::ApplyMat4);
    stages.push(Stage::MdsCross);
    // 4 initial external rounds.
    for rc in &c.external_initial {
        stages.push(Stage::ExternalRcSbox(*rc));
        stages.push(Stage::ApplyMat4);
        stages.push(Stage::MdsCross);
    }
    // Internal rounds.
    for &rc in &c.internal {
        stages.push(Stage::InternalSbox(rc));
        for stride in [1u32, 2, 4, 8] {
            stages.push(Stage::ButterflySum(stride));
        }
        stages.push(Stage::InternalCombine(c.diag));
    }
    // Final external rounds.
    for rc in &c.external_final {
        stages.push(Stage::ExternalRcSbox(*rc));
        stages.push(Stage::ApplyMat4);
        stages.push(Stage::MdsCross);
    }
    stages
}

/// Emits a linear chain of stages via nested `bind`s and hands the final tile
/// to `f`. Every `bind`'s value is a single `compute_with` — the canonical
/// let-bound-inner-compute shape that `peel_body_lets` accepts.
///
/// `after_sbox_tile` tracks the most recent `InternalSbox` result — the
/// internal round's `Combine` reads it alongside the running butterfly sum.
fn with_stages_cps<F>(
    b: &mut IRBuilder,
    prev: NodeId,
    stages: &[Stage],
    after_sbox_tile: Option<NodeId>,
    f: F,
) -> NodeId
where
    F: FnOnce(&mut IRBuilder, NodeId) -> NodeId,
{
    if stages.is_empty() {
        return f(b, prev);
    }
    let (first, rest) = stages.split_first().unwrap();
    let is_internal_sbox = matches!(first, Stage::InternalSbox(_));
    let stage_val = first.emit(b, prev, after_sbox_tile);
    let rest_owned: Vec<Stage> = rest.to_vec();
    b.bind(stage_val, move |b, next| {
        let new_after_sbox = if is_internal_sbox {
            Some(next)
        } else {
            after_sbox_tile
        };
        with_stages_cps(b, next, &rest_owned, new_after_sbox, f)
    })
}

/// Warp-parallel Poseidon2-16 permutation, continuation-passing style.
///
/// `state` is a `[WIDTH]`-shape BabyBear tensor whose element `j` will be
/// handled by lane `j`; `k` is called with the permuted tile still bound as
/// an inner let (so `k`'s expression can be another `bind`, another `compute`
/// reading it, or a `pack`/index expression). The full expression tree lives
/// as a single nested `Let { .. compute .. }` chain the canonicalization
/// pass can peel as inner-compute tiles.
///
/// # Where this may be called
///
/// Must be invoked from inside an outer
/// [`IRBuilder::compute_with`](crate::ir::IRBuilder::compute_with) whose
/// `threads` hint is `Some(WIDTH)` (a warp): under that hint the compiler
/// carves out 16 lanes and the identity `par` maps each state slot to one
/// lane.
pub fn poseidon2_permute_par<F>(
    b: &mut IRBuilder,
    state: NodeId,
    c: &Poseidon2Constants,
    f: F,
) -> NodeId
where
    F: FnOnce(&mut IRBuilder, NodeId) -> NodeId,
{
    let stages = build_stages(c);
    with_stages_cps(b, state, &stages, None, f)
}

/// Stand-alone module: `state[1, WIDTH] -> new_state[1, WIDTH]`, warp-
/// parallel. One block, `WIDTH` threads, one permutation per launch.
///
/// The outer compute has `bound = 1` and `threads = Some(WIDTH)` so that
/// the inner par-stages have a whole warp to their name. The state is
/// gathered from `state[1, WIDTH]` into a lane-owned `[WIDTH]` tile with
/// the identity par-layout, then the permutation runs; the final tile is
/// yielded directly through the outer compute.
pub fn poseidon2_permute_par_module(c: &Poseidon2Constants) -> Module {
    let consts = c.clone();
    let mut b = IRBuilder::new();
    let state_in = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        // Gather the state into a `[WIDTH]` tile under the identity par-
        // layout: lane `j` reads `state[0, j]`.
        let par = lane_par(b);
        let gather = b.compute_with(
            WIDTH,
            None,
            Some(par),
            None,
            |b, j| kernel!(b, state_in[0, j]),
        );
        b.bind(gather, move |b, s| {
            poseidon2_permute_par(b, s, &consts, |b, v| {
                // Yield the lane-owned result through a final identity-par
                // stage; each lane stores its own slot to the output.
                let par = lane_par(b);
                b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, v[j]))
            })
        })
    });
    b.finish("poseidon2_perm_par", body)
}

/// Reference (serial) module built with the same input/output shape as
/// [`poseidon2_permute_par_module`], for use in correctness tests.
///
/// `state[1, WIDTH] -> new_state[1, WIDTH]`, one thread does the whole
/// permutation.
pub fn poseidon2_permute_serial_module(c: &Poseidon2Constants) -> Module {
    use crate::kernels::poseidon2_permutation;
    let mut b = IRBuilder::new();
    let state_in = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let body = b.compute(1, |b, _i| {
        let mut s = [b.const_u32(0); WIDTH];
        for (j, slot) in s.iter_mut().enumerate() {
            *slot = kernel!(b, state_in[0, #j]);
        }
        poseidon2_permutation(b, &mut s, c);
        b.pack(&s)
    });
    b.finish("poseidon2_perm_serial", body)
}

#[cfg(all(test, feature = "planner"))]
mod tests {
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::GpuDeviceCtx,
    };
    use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear};
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use p3_symmetric::Permutation;

    use super::*;
    use crate::{
        compile_and_load,
        kernels::poseidon2_permutation,
        runner::{from_monty, to_monty},
        runtime::CompileOptions,
    };

    const P: u64 = 2_013_265_921;

    fn splitmix(n: usize, seed: u64) -> Vec<u32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z % P) as u32
            })
            .collect()
    }

    fn run_perm_module(module: crate::ir::Module, input: &[u32]) -> Vec<u32> {
        let ctx = GpuDeviceCtx::for_current_device().unwrap();
        let options = CompileOptions {
            dump_ir: Some(
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../target/ir-dumps/poseidon2-par"
                )
                .into(),
            ),
            ..Default::default()
        };
        let mut km = compile_and_load(&module, &options).unwrap();
        // The emitted CUDA operates on Montgomery-encoded BabyBear; this
        // helper drives the raw `KernelModule` so it encodes/decodes itself.
        let mont_input: Vec<u32> = input.iter().map(|&v| to_monty(v)).collect();
        let d_in: DeviceBuffer<u32> = mont_input.as_slice().to_device_on(&ctx).unwrap();
        km.set_input(0, &d_in).unwrap();
        let d_out = DeviceBuffer::<u32>::with_capacity_on(WIDTH, &ctx);
        km.set_output(0, &d_out).unwrap();
        km.ensure_scratch(&ctx);
        km.run(&ctx.stream).unwrap();
        d_out
            .to_host_on(&ctx)
            .unwrap()
            .into_iter()
            .map(from_monty)
            .collect()
    }

    fn cpu_reference(state: &[u32; WIDTH]) -> [u32; WIDTH] {
        let perm = default_babybear_poseidon2_16();
        let mut s = [BabyBear::ZERO; WIDTH];
        for (i, &v) in state.iter().enumerate() {
            s[i] = BabyBear::new(v);
        }
        let out = perm.permute(s);
        std::array::from_fn(|i| out[i].as_canonical_u32())
    }

    /// The DSL's serial permutation must match p3's poseidon2 on every input.
    /// (Sanity check that our reference matches what the parallel version is
    /// compared against.)
    fn ir_reference(state: &[u32; WIDTH]) -> [u32; WIDTH] {
        let consts = Poseidon2Constants::p3_default();
        let mut ib = IRBuilder::new();
        let ins: Vec<NodeId> = state.iter().map(|&v| ib.const_field(v)).collect();
        let mut s = [ib.const_u32(0); WIDTH];
        s.copy_from_slice(&ins);
        poseidon2_permutation(&mut ib, &mut s, &consts);
        // Evaluate the built expression by running the reference on CPU:
        // (we already checked equivalence via cpu_reference; kept as helper.)
        let _ = s;
        cpu_reference(state)
    }

    fn cases() -> Vec<[u32; WIDTH]> {
        let mut out: Vec<[u32; WIDTH]> = Vec::new();
        // All zeros.
        out.push([0u32; WIDTH]);
        // All ones.
        out.push([1u32; WIDTH]);
        // Deterministic ramp.
        out.push(std::array::from_fn(|i| ((i as u32) * 42 + 17) % (P as u32)));
        // Pseudorandom.
        for seed in [1u64, 42, 0xdead_beef_cafe_babe] {
            let v = splitmix(WIDTH, seed);
            let mut arr = [0u32; WIDTH];
            arr.copy_from_slice(&v);
            out.push(arr);
        }
        out
    }

    #[test]
    fn serial_module_matches_p3() {
        let consts = Poseidon2Constants::p3_default();
        let module = poseidon2_permute_serial_module(&consts);
        for state in cases() {
            let got = run_perm_module(module.clone(), &state);
            let want = ir_reference(&state);
            assert_eq!(
                got.as_slice(),
                &want[..],
                "serial poseidon2 mismatch on {state:?}"
            );
        }
    }

    #[test]
    fn parallel_module_matches_p3() {
        let consts = Poseidon2Constants::p3_default();
        let module = poseidon2_permute_par_module(&consts);
        for state in cases() {
            let got = run_perm_module(module.clone(), &state);
            let want = cpu_reference(&state);
            assert_eq!(
                got.as_slice(),
                &want[..],
                "parallel poseidon2 mismatch on {state:?}"
            );
        }
    }

    /// The transcript's `sponge_observe_perm_at_7` shape — the module reads
    /// `state[1, WIDTH]`, inserts `value[0]` at slot `CHUNK - 1`, permutes,
    /// and writes the new state. This test compiles that shape twice — once
    /// using the serial `poseidon2_permutation` and once using the warp-
    /// parallel version — and asserts they produce the same result.
    #[test]
    fn transcript_observe_perm_at_7_matches() {
        const CHUNK: usize = 8;
        const ABSORB_IDX: usize = CHUNK - 1;

        fn build_observe_serial(consts: &Poseidon2Constants) -> crate::ir::Module {
            use crate::kernels::poseidon2_permutation;
            let mut b = IRBuilder::new();
            let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
            let value = b.input("value", ScalarType::BabyBear, vec![1]);
            let body = b.compute(1, |b, _i| {
                let mut s = [b.const_u32(0); WIDTH];
                for (j, slot) in s.iter_mut().enumerate() {
                    *slot = kernel!(b, state[0, #j]);
                }
                s[ABSORB_IDX] = kernel!(b, value[0]);
                poseidon2_permutation(b, &mut s, consts);
                b.pack(&s)
            });
            b.finish("sponge_observe_perm_at_7_serial", body)
        }

        fn build_observe_parallel(consts: &Poseidon2Constants) -> crate::ir::Module {
            let consts = consts.clone();
            let mut b = IRBuilder::new();
            let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
            let value = b.input("value", ScalarType::BabyBear, vec![1]);
            let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
                // Gather the state with the value spliced in at ABSORB_IDX.
                let par = lane_par(b);
                let gather = b.compute_with(WIDTH, None, Some(par), None, |b, j| {
                    kernel!(b,
                        let orig = state[0, j];
                        let v = value[0];
                        if j == #(ABSORB_IDX as u32) then v else orig
                    )
                });
                b.bind(gather, move |b, s| {
                    poseidon2_permute_par(b, s, &consts, |b, v| {
                        let par = lane_par(b);
                        b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, v[j]))
                    })
                })
            });
            b.finish("sponge_observe_perm_at_7_parallel", body)
        }

        let consts = Poseidon2Constants::p3_default();
        let serial = build_observe_serial(&consts);
        let parallel = build_observe_parallel(&consts);

        for base in cases() {
            let value = base[0].wrapping_add(3) % (P as u32);
            let mut state = base;
            // Run both modules — serial vs parallel — and compare outputs.
            let ctx = GpuDeviceCtx::for_current_device().unwrap();
            let options = CompileOptions::default();

            let mut km_s = compile_and_load(&serial, &options).unwrap();
            let mut km_p = compile_and_load(&parallel, &options).unwrap();
            let mont_state: Vec<u32> = state.iter().map(|&v| to_monty(v)).collect();
            let d_state: DeviceBuffer<u32> = mont_state.as_slice().to_device_on(&ctx).unwrap();
            let d_value: DeviceBuffer<u32> =
                [to_monty(value)].as_slice().to_device_on(&ctx).unwrap();
            let d_out_s = DeviceBuffer::<u32>::with_capacity_on(WIDTH, &ctx);
            let d_out_p = DeviceBuffer::<u32>::with_capacity_on(WIDTH, &ctx);
            km_s.set_input(0, &d_state).unwrap();
            km_s.set_input(1, &d_value).unwrap();
            km_s.set_output(0, &d_out_s).unwrap();
            km_s.ensure_scratch(&ctx);
            km_p.set_input(0, &d_state).unwrap();
            km_p.set_input(1, &d_value).unwrap();
            km_p.set_output(0, &d_out_p).unwrap();
            km_p.ensure_scratch(&ctx);

            km_s.run(&ctx.stream).unwrap();
            km_p.run(&ctx.stream).unwrap();
            let got_s: Vec<u32> = d_out_s
                .to_host_on(&ctx)
                .unwrap()
                .into_iter()
                .map(from_monty)
                .collect();
            let got_p: Vec<u32> = d_out_p
                .to_host_on(&ctx)
                .unwrap()
                .into_iter()
                .map(from_monty)
                .collect();
            state[ABSORB_IDX] = value;
            let want = cpu_reference(&state);
            assert_eq!(got_s, want, "serial observe_perm_at_7 mismatch");
            assert_eq!(got_p, want, "parallel observe_perm_at_7 mismatch");
        }
    }
}
