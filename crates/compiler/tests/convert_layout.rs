//! Direct-KIR tests for `ConvertLayout` swizzling and codegen: general
//! register↔register shuffles (fast / ternary / multi-round pull / bounce),
//! shared→shared, shared→register and register→shared, with deterministic
//! edge cases and property tests over randomly generated linear layouts.
//!
//! Programs are hand-built at the KIR level (bypassing `layout_infer`, which
//! is exercised separately) so each test pins the exact `(src, dst, map)`
//! triple fed to `gen_convert`. Every chain is `input → t0 → … → tN →
//! output` where each link is one `ConvertLayout`; with distinct input
//! values, any misrouted element is detected. GPU tests require a CUDA GPU.

use crypto_compiler::{
    ir::{ScalarType, SizeExpr},
    kernel_ir::{
        const_src_slot, f2, lane_block_invertible, lane_slot_mix_dirs, shuffle_rounds, Access,
        AddressSpace, BufId, BufferDecl, BufferKind, IndexMap, Kernel, KirProgram, LinearLayout,
        ParAttr, SSABlock, SSAOp, SSAOpCode,
    },
    module_compiler::ModuleCompiler,
    passes,
    passes::{
        allocate_convert_scratch::convert_scratch_bytes,
        convert_decompose::{best_decomposition, Strategy},
        layout_cost::ConversionCostModel,
    },
};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};

// ---------------------------------------------------------------------------
// Deterministic RNG (splitmix64) and random layout generation
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A uniformly random invertible XOR-affine map on `kb` bits (rejection
/// sampling; the singular fraction is bounded away from 1).
fn rand_bijective(kb: usize, rng: &mut Rng, allow_offset: bool) -> LinearLayout {
    loop {
        let l = LinearLayout {
            bases: (0..kb).map(|_| rng.below(1 << kb)).collect(),
            offset: if allow_offset && rng.below(2) == 1 {
                rng.below(1 << kb)
            } else {
                0
            },
        };
        if l.inverse().is_some() {
            return l;
        }
    }
}

/// A random (possibly singular) map whose bases and offset stay within
/// `kb` bits.
fn rand_layout(kb: usize, rng: &mut Rng) -> LinearLayout {
    LinearLayout {
        bases: (0..kb).map(|_| rng.below(1 << kb)).collect(),
        offset: rng.below(1 << kb),
    }
}

fn swap_bits(kb: usize, a: usize, b: usize) -> LinearLayout {
    let mut l = LinearLayout::identity(kb);
    l.bases[a] = 1 << b;
    l.bases[b] = 1 << a;
    l
}

fn bit_reversal(kb: usize) -> LinearLayout {
    LinearLayout {
        bases: (0..kb).map(|i| 1u64 << (kb - 1 - i)).collect(),
        offset: 0,
    }
}

/// `phys = v ^ (v >> shift)`: bit `i` also receives bit `i + shift`.
/// Unitriangular, hence invertible — the classic bank-swizzle family.
fn xor_swizzle(kb: usize, shift: usize) -> LinearLayout {
    LinearLayout {
        bases: (0..kb)
            .map(|i| {
                let mut b = 1u64 << i;
                if i >= shift {
                    b |= 1 << (i - shift);
                }
                b
            })
            .collect(),
        offset: 0,
    }
}

/// Rotate the five lane bits left by one, identity above: invertible lane
/// block, non-constant sender slot — gen_shuffle's ternary path.
fn lane_rotation(kb: usize) -> LinearLayout {
    let mut l = LinearLayout::identity(kb);
    for i in 0..5 {
        l.bases[i] = 1 << ((i + 1) % 5);
    }
    l
}

// ---------------------------------------------------------------------------
// KIR chain builder
// ---------------------------------------------------------------------------

/// One conversion-chain kernel: `input --load par--> tiles[0] --convert-->
/// … --convert--> tiles[last] --store par--> output`. `tiles[i].0` picks
/// Shared (true) or Register (false); `maps[i]` is the i-th ConvertLayout's
/// logical map (`dst[v] = src[map(v)]`).
struct Chain {
    name: String,
    n: usize,
    block: usize,
    tiles: Vec<(bool, LinearLayout)>,
    maps: Vec<LinearLayout>,
}

impl Chain {
    fn identity_maps(name: &str, n: usize, block: usize, tiles: Vec<(bool, LinearLayout)>) -> Self {
        let kb = n.trailing_zeros() as usize;
        let maps = vec![LinearLayout::identity(kb); tiles.len() - 1];
        Chain {
            name: name.to_string(),
            n,
            block,
            tiles,
            maps,
        }
    }

    /// `output[v] = input[maps[0](maps[1](…maps[last](v)))]`.
    fn expected(&self, input: &[u32]) -> Vec<u32> {
        (0..self.n)
            .map(|v| {
                let mut x = v as u64;
                for m in self.maps.iter().rev() {
                    x = m.apply(x);
                }
                input[x as usize]
            })
            .collect()
    }
}

#[derive(Default)]
struct ProgBuilder {
    buffers: Vec<BufferDecl>,
    kernels: Vec<Kernel>,
    input_bufs: Vec<BufId>,
    output_bufs: Vec<BufId>,
}

impl ProgBuilder {
    fn push_buffer(&mut self, decl: BufferDecl) -> BufId {
        let id = BufId(self.buffers.len() as u32);
        self.buffers.push(decl);
        id
    }

    fn add_chain(&mut self, c: &Chain) {
        assert!(c.n.is_power_of_two() && c.n >= c.block && c.block >= 32);
        assert_eq!(c.maps.len() + 1, c.tiles.len());
        let kb = c.n.trailing_zeros() as usize;
        let io = self.input_bufs.len();
        let global = |kind| BufferDecl {
            name: format!("{}_g{io}", c.name),
            elem: ScalarType::U32,
            shape: vec![SizeExpr::from(c.n)],
            kind,
            space: AddressSpace::Global,
            layout: None,
        };
        let a_id = self.push_buffer(global(BufferKind::Input(io)));
        let out_id = self.push_buffer(global(BufferKind::Output(io)));
        self.input_bufs.push(a_id);
        self.output_bufs.push(out_id);

        let mut k = Kernel::new(c.name.clone(), 1usize, c.block);
        k.params = vec![(a_id, false), (out_id, true)];

        let mut tile_ids = Vec::new();
        for (ti, (shared, l)) in c.tiles.iter().enumerate() {
            assert_eq!(l.bases.len(), kb);
            assert!(l.inverse().is_some(), "tile layouts must be bijective");
            let id = self.push_buffer(BufferDecl {
                name: format!("{}_t{ti}", c.name),
                elem: ScalarType::U32,
                shape: vec![SizeExpr::from(c.n)],
                kind: if *shared {
                    BufferKind::Shared
                } else {
                    BufferKind::Register
                },
                space: if *shared {
                    AddressSpace::Shared
                } else {
                    AddressSpace::Register
                },
                layout: Some(l.clone()),
            });
            push_stmt(&mut k, SSAOpCode::Alloc { buf: id });
            tile_ids.push(id);
        }

        // Load par: registers demand the par attr equal the tile layout
        // (own-index rule); shared tiles take the identity attr and route
        // through the decl layout at the access.
        let attr_for = |(shared, l): &(bool, LinearLayout)| {
            if *shared {
                LinearLayout::identity(kb)
            } else {
                l.clone()
            }
        };
        push_copy_par(&mut k, c, kb, a_id, tile_ids[0], attr_for(&c.tiles[0]));

        for (i, m) in c.maps.iter().enumerate() {
            let scratch = self.push_buffer(BufferDecl {
                name: format!("{}_cs{i}", c.name),
                elem: ScalarType::U32,
                shape: vec![SizeExpr::from(0usize)],
                kind: BufferKind::Shared,
                space: AddressSpace::Shared,
                layout: None,
            });
            push_stmt(&mut k, SSAOpCode::Alloc { buf: scratch });
            push_stmt(
                &mut k,
                SSAOpCode::ConvertLayout {
                    dst: tile_ids[i + 1],
                    src: tile_ids[i],
                    scratch,
                    map: m.clone(),
                },
            );
        }

        let last = c.tiles.last().unwrap();
        push_copy_par(
            &mut k,
            c,
            kb,
            *tile_ids.last().unwrap(),
            out_id,
            attr_for(last),
        );
        self.kernels.push(k);
    }

    fn build(self, name: &str) -> KirProgram {
        KirProgram {
            name: name.to_string(),
            buffers: self.buffers,
            kernels: self.kernels,
            input_bufs: self.input_bufs,
            output_bufs: self.output_bufs,
            params: Vec::new(),
        }
    }
}

fn push_stmt(k: &mut Kernel, opcode: SSAOpCode) {
    let id = k.push_op(SSAOp {
        operands: Default::default(),
        results: Default::default(),
        opcode,
        block: SSABlock::default(),
    });
    k.grid.block.body.push(id);
}

/// A par copying `src[v] → dst[v]` over the chain's logical domain under
/// `attr_layout` (both accesses are the identity index map; each buffer's
/// own decl layout places the element physically).
fn push_copy_par(k: &mut Kernel, c: &Chain, kb: usize, src: BufId, dst: BufId, attr: LinearLayout) {
    let v = k.fresh_val();
    let r = k.fresh_val();
    let w = k.fresh_val();
    let id = k.push_op(SSAOp {
        operands: Default::default(),
        results: vec![w].into(),
        opcode: SSAOpCode::Par {
            bound: c.n.into(),
            spans_grid: false,
            attr: Some(ParAttr {
                seq_size: c.n / c.block,
                layout: attr,
            }),
            reads: vec![Access {
                buf: src,
                index: IndexMap::Linear(LinearLayout::identity(kb)),
            }],
            writes: vec![Access {
                buf: dst,
                index: IndexMap::Linear(LinearLayout::identity(kb)),
            }],
        },
        block: SSABlock {
            operands: vec![v, r].into(),
            body: Default::default(),
            yields: vec![r].into(),
        },
    });
    k.grid.block.body.push(id);
}

/// Runs `allocate_convert_scratch` + `insert_sync`, then codegen + verify,
/// returning the finalized program and its CUDA source (no nvcc).
fn finalize(mut kp: KirProgram) -> (KirProgram, String) {
    passes::allocate_convert_scratch(&mut kp);
    passes::insert_sync(&mut kp);
    let src = passes::codegen(&kp).expect("codegen");
    passes::verify(&kp).expect("verify");
    (kp, src)
}

/// Full pipeline: passes → nvcc → run on the GPU. One `Vec<u32>` per
/// module output.
fn compile_and_run(kp: KirProgram, inputs: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut kp = kp;
    passes::allocate_convert_scratch(&mut kp);
    passes::insert_sync(&mut kp);
    let out_lens: Vec<usize> = kp.output_bufs.iter().map(|&b| kp.buffer(b).len()).collect();
    let mut prog = ModuleCompiler::new().codegen(kp).expect("compile");
    let ctx = GpuDeviceCtx::for_current_device().expect("gpu ctx");
    let mut in_bufs = Vec::new();
    for (i, data) in inputs.iter().enumerate() {
        let mut buf = DeviceBuffer::<u32>::with_capacity_on(data.len(), &ctx);
        data.copy_to_on(&mut buf, &ctx).expect("h2d");
        prog.set_input(i, &buf).expect("set_input");
        in_bufs.push(buf);
    }
    let out_bufs: Vec<DeviceBuffer<u32>> = out_lens
        .iter()
        .map(|&l| DeviceBuffer::<u32>::with_capacity_on(l, &ctx))
        .collect();
    for (i, buf) in out_bufs.iter().enumerate() {
        prog.set_output(i, buf).expect("set_output");
    }
    prog.run(&ctx.stream).expect("run");
    ctx.stream.synchronize().expect("sync");
    out_bufs
        .iter()
        .map(|b| b.to_host_on(&ctx).expect("d2h"))
        .collect()
}

/// Builds one program from `chains`, runs it with distinct inputs, and
/// checks every kernel's output against its expected permutation.
fn run_chains(name: &str, chains: &[Chain]) {
    let mut pb = ProgBuilder::default();
    for c in chains {
        pb.add_chain(c);
    }
    let kp = pb.build(name);
    let inputs: Vec<Vec<u32>> = chains
        .iter()
        .enumerate()
        .map(|(i, c)| (0..c.n as u32).map(|v| v * 7 + i as u32).collect())
        .collect();
    let outs = compile_and_run(kp, &inputs);
    for (i, c) in chains.iter().enumerate() {
        assert_eq!(
            outs[i],
            c.expected(&inputs[i]),
            "chain `{}` (n={}, block={}) mismatched",
            c.name,
            c.n,
            c.block
        );
    }
}

// ---------------------------------------------------------------------------
// CPU property tests: the multi-round pull scheme's invariants
// ---------------------------------------------------------------------------

/// Simulates the multi-round pull emitter for a composite `c`: for every
/// destination slot `sp` and thread `tid`, the enumerated sender slots
/// `σ ∈ (cs >> tb) ^ span(dirs)` must all be in-range (they index
/// `src_reg[σ]` unconditionally) and exactly one must equal the needed
/// `q >> tb` — the guard `(q >> tb) == σ` fires exactly once per pair.
fn check_pull_scheme(c: &LinearLayout, block: usize) {
    let kb = c.bases.len();
    let tb = kb.min(block.trailing_zeros() as usize);
    let slots = 1u64 << (kb - tb);
    let dirs = lane_slot_mix_dirs(c, tb);
    let c_lin = LinearLayout {
        bases: c.bases.clone(),
        offset: 0,
    };
    let warp_identity = c.is_warp_column_identity(block);
    let warp_mask = if tb > 5 { ((1u64 << tb) - 1) & !31 } else { 0 };
    for sp in 0..slots {
        let cs = c.apply(sp << tb);
        for tid in 0..(1u64 << tb) {
            let q = cs ^ c_lin.apply(tid);
            // The pull scheme reads src phys q for dst phys sp<<tb | tid.
            assert_eq!(q, c.apply(sp << tb | tid));
            let mut matches = 0;
            for j in 0..(1u64 << dirs.len()) {
                let mut sigma = cs >> tb;
                for (bit, &d) in dirs.iter().enumerate() {
                    if (j >> bit) & 1 == 1 {
                        sigma ^= d;
                    }
                }
                assert!(sigma < slots, "sender slot {sigma} out of range");
                if sigma == q >> tb {
                    matches += 1;
                }
            }
            assert_eq!(matches, 1, "sp={sp} tid={tid}: {matches} senders matched");
            if warp_identity {
                // Intra-warp shuffles stay valid: q's warp bits are tid's.
                assert_eq!(q & warp_mask, tid & warp_mask);
            }
        }
    }
}

#[test]
fn prop_pull_scheme_covers_every_pair_exactly_once() {
    let mut rng = Rng::new(0x5EED_0001);
    for &block in &[32usize, 64, 256] {
        for kb in 5..=8 {
            for _ in 0..25 {
                // Arbitrary (even singular) composites keep the invariants.
                check_pull_scheme(&rand_layout(kb, &mut rng), block);
                check_pull_scheme(&rand_bijective(kb, &mut rng, true), block);
            }
        }
    }
}

#[test]
fn prop_shuffle_rounds_matches_dispatch() {
    let mut rng = Rng::new(0x5EED_0002);
    for &block in &[32usize, 64, 256] {
        for kb in 5..=8 {
            for _ in 0..50 {
                let c = rand_bijective(kb, &mut rng, true);
                let tb = kb.min(block.trailing_zeros() as usize);
                let slots = 1u64 << (kb - tb);
                let rounds = shuffle_rounds(&c, block);
                if slots == 1 || const_src_slot(&c, tb) || lane_block_invertible(&c, tb) {
                    assert_eq!(rounds, slots, "fast/ternary paths emit one per slot");
                } else {
                    let dirs = lane_slot_mix_dirs(&c, tb);
                    assert_eq!(rounds, slots << dirs.len(), "pull path count");
                    assert!(!dirs.is_empty(), "pull path implies lane→slot mixing");
                }
                // dirs is an independent basis: canonical RREF from f2.
                let dirs = lane_slot_mix_dirs(&c, tb);
                assert_eq!(f2::reduce(dirs.clone()), dirs);
            }
        }
    }
}

#[test]
fn prop_scratch_sizing_agrees_with_decomposition() {
    let mut rng = Rng::new(0x5EED_0003);
    let cost = ConversionCostModel::default();
    for &block in &[32usize, 64, 128] {
        for kb in 5..=8 {
            for _ in 0..40 {
                let src = rand_bijective(kb, &mut rng, true);
                let dst = rand_bijective(kb, &mut rng, true);
                let dec = best_decomposition(&cost, &src, &dst, block, kb, &[]);
                let bytes = convert_scratch_bytes(
                    Some(&src),
                    Some(&dst),
                    AddressSpace::Register,
                    AddressSpace::Register,
                    1 << kb,
                    block,
                    4,
                );
                assert_eq!(
                    bytes > 0,
                    matches!(dec.strategy, Strategy::Bounce { .. }),
                    "scratch bytes ({bytes}) must be non-zero iff Bounce wins ({:?})",
                    dec.strategy
                );
                if let Strategy::Shuffle { rounds } = dec.strategy {
                    let c = src.right_inverse(kb).unwrap().compose(&dst);
                    assert!(c.is_warp_column_identity(block));
                    assert_eq!(rounds, shuffle_rounds(&c, block));
                }
            }
        }
    }
}

/// `right_inverse` contract on random dense affine maps: `T(T⁺(y)) == y`
/// for every y. Dense matrices exercise the pivot-column cleanup that
/// permutation/triangular layouts never do (caught a real bug: preimages
/// snapshotted mid-elimination inverted partially reduced columns).
#[test]
fn prop_right_inverse_is_right_inverse() {
    let mut rng = Rng::new(0x51DE_1DE1);
    for kb in 3..=8usize {
        for _ in 0..40 {
            let l = rand_bijective(kb, &mut rng, true);
            let r = l.right_inverse(kb).expect("bijective map is surjective");
            for y in 0..1u64 << kb {
                assert_eq!(l.apply(r.apply(y)), y, "layout {l:?}, rinv {r:?}");
            }
            let inv = l.inverse().expect("bijective");
            for y in 0..1u64 << kb {
                assert_eq!(l.apply(inv.apply(y)), y);
                assert_eq!(inv.apply(l.apply(y)), y);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU source-marker tests: each strategy leaves its fingerprint
// ---------------------------------------------------------------------------

fn single_chain_source(c: &Chain) -> (KirProgram, String) {
    let mut pb = ProgBuilder::default();
    pb.add_chain(c);
    let (kp, src) = finalize(pb.build(&c.name));
    // Slice off the prelude (FpExt `__shfl_sync` overload etc.) so marker
    // counts only see the kernel bodies.
    let body_start = src.find("__global__").expect("kernel entry in source");
    let body = src[body_start..].to_string();
    (kp, body)
}

#[test]
fn source_multi_round_emits_exact_shuffle_count() {
    // c = dst layout (src identity): lane bits 0,1 ↔ slot bits 5,6 —
    // singular lane block, non-constant sender slot. slots=4, |dirs|=2:
    // exactly 16 unconditional `__shfl_sync`s, no shared traffic.
    let l = LinearLayout {
        bases: vec![32, 64, 4, 8, 16, 1, 2],
        offset: 0,
    };
    assert_eq!(shuffle_rounds(&l, 32), 16);
    let chain = Chain::identity_maps(
        "src_multi_round",
        128,
        32,
        vec![(false, LinearLayout::identity(7)), (false, l)],
    );
    let (kp, src) = single_chain_source(&chain);
    assert_eq!(src.matches("__shfl_sync").count(), 16);
    assert_eq!(src.matches("__syncthreads").count(), 0);
    for b in &kp.buffers {
        if b.name.contains("_cs") {
            assert!(b.is_empty(), "pure shuffle must keep scratch 0-byte");
        }
    }
}

#[test]
fn source_slot_strategy_has_no_shuffles() {
    // Slot bits 5,6 swapped, lanes identity: in-thread slot rename.
    let chain = Chain::identity_maps(
        "src_slot",
        128,
        32,
        vec![
            (false, LinearLayout::identity(7)),
            (false, swap_bits(7, 5, 6)),
        ],
    );
    let (_, src) = single_chain_source(&chain);
    assert_eq!(src.matches("__shfl_sync").count(), 0);
    assert_eq!(src.matches("__syncthreads").count(), 0);
}

#[test]
fn source_bounce_stages_through_sized_scratch() {
    // block=64: lane bit 0 ↔ warp bit 5 crosses warps — no shuffle can
    // realize it; the bounce stages a full tile through shared scratch
    // with exactly one internal barrier.
    let chain = Chain::identity_maps(
        "src_bounce",
        128,
        64,
        vec![
            (false, LinearLayout::identity(7)),
            (false, swap_bits(7, 0, 5)),
        ],
    );
    let (kp, src) = single_chain_source(&chain);
    assert_eq!(src.matches("__shfl_sync").count(), 0);
    assert_eq!(src.matches("__syncthreads").count(), 1);
    let scratch: Vec<_> = kp
        .buffers
        .iter()
        .filter(|b| b.name.contains("_cs"))
        .collect();
    assert_eq!(scratch.len(), 1);
    assert_eq!(scratch[0].len(), 128, "bounce scratch is a full tile");
}

#[test]
fn source_shared_to_shared_is_direct_loop() {
    // Shared→Shared: one read-then-write loop, no shuffles, no scratch;
    // insert_sync fences the converted tile before its reader.
    let chain = Chain::identity_maps(
        "src_sh_sh",
        128,
        32,
        vec![(true, LinearLayout::identity(7)), (true, xor_swizzle(7, 3))],
    );
    let (kp, src) = single_chain_source(&chain);
    assert_eq!(src.matches("__shfl_sync").count(), 0);
    assert!(src.contains("_sh_pool"), "shared tiles live in the pool");
    assert!(src.matches("__syncthreads").count() >= 1);
    for b in &kp.buffers {
        if b.name.contains("_cs") {
            assert!(b.is_empty(), "shared→shared needs no staging scratch");
        }
    }
}

// ---------------------------------------------------------------------------
// GPU edge cases
// ---------------------------------------------------------------------------

/// Register→register composites, one kernel per gen_shuffle dispatch
/// path (plus Copy/Slot), all under a 32-thread block.
#[test]
fn gpu_reg_reg_edge_cases() {
    let id7 = LinearLayout::identity(7);
    let butterfly = LinearLayout {
        bases: LinearLayout::identity(7).bases,
        offset: 1,
    };
    let slot_xor = LinearLayout {
        bases: LinearLayout::identity(7).bases,
        offset: 96, // slot bits 5,6: constant sender slot, fast path
    };
    let multi_round_1dir = LinearLayout {
        bases: vec![32, 2, 4, 8, 16, 1],
        offset: 0,
    };
    let multi_round_2dir = LinearLayout {
        bases: vec![32, 64, 4, 8, 16, 1, 2],
        offset: 0,
    };
    let reg = |l: LinearLayout| (false, l);
    let chains = vec![
        Chain::identity_maps("copy", 128, 32, vec![reg(id7.clone()), reg(id7.clone())]),
        Chain::identity_maps(
            "slot_swap",
            128,
            32,
            vec![reg(id7.clone()), reg(swap_bits(7, 5, 6))],
        ),
        Chain::identity_maps(
            "lane_rotation",
            128,
            32,
            vec![reg(id7.clone()), reg(lane_rotation(7))],
        ),
        Chain::identity_maps(
            "butterfly_offset",
            128,
            32,
            vec![reg(id7.clone()), reg(butterfly)],
        ),
        Chain::identity_maps(
            "slot_offset",
            128,
            32,
            vec![reg(id7.clone()), reg(slot_xor)],
        ),
        Chain::identity_maps(
            "multi_round_one_dir",
            64,
            32,
            vec![reg(LinearLayout::identity(6)), reg(multi_round_1dir)],
        ),
        Chain::identity_maps(
            "multi_round_two_dirs",
            128,
            32,
            vec![reg(id7.clone()), reg(multi_round_2dir.clone())],
        ),
        Chain::identity_maps(
            "lane_only_single_slot",
            32,
            32,
            vec![reg(LinearLayout::identity(5)), reg(bit_reversal(5))],
        ),
        Chain::identity_maps(
            "two_hop_chain",
            128,
            32,
            vec![reg(id7), reg(multi_round_2dir), reg(bit_reversal(7))],
        ),
    ];
    run_chains("reg_reg_edges", &chains);
}

/// Warp-crossing (bounce) and warp-preserving multi-round composites
/// under 64-thread blocks.
#[test]
fn gpu_warp_crossing_edge_cases() {
    let reg = |l: LinearLayout| (false, l);
    let chains = vec![
        Chain::identity_maps(
            "bounce_lane_warp_swap",
            128,
            64,
            vec![reg(LinearLayout::identity(7)), reg(swap_bits(7, 0, 5))],
        ),
        // bit 0 ↔ bit 6 swap keeps warp bit 5 identity: multi-round pull
        // with tb=6 (dirs from lane→slot), no bounce.
        Chain::identity_maps(
            "warp_identity_multi_round",
            128,
            64,
            vec![reg(LinearLayout::identity(7)), reg(swap_bits(7, 0, 6))],
        ),
        // Bit reversal on 8 bits at block=64 crosses warps; the chain
        // bounces out and back.
        Chain::identity_maps(
            "bounce_roundtrip",
            256,
            64,
            vec![
                reg(LinearLayout::identity(8)),
                reg(bit_reversal(8)),
                reg(LinearLayout::identity(8)),
            ],
        ),
    ];
    run_chains("warp_edges", &chains);
}

/// Shared-memory endpoints: register→shared (plain and swizzled),
/// shared→shared, shared→register, and a reg→shared→reg sandwich.
#[test]
fn gpu_shared_edge_cases() {
    let id7 = LinearLayout::identity(7);
    let swz = xor_swizzle(7, 3);
    let reg = |l: &LinearLayout| (false, l.clone());
    let sh = |l: &LinearLayout| (true, l.clone());
    let multi_round = LinearLayout {
        bases: vec![32, 64, 4, 8, 16, 1, 2],
        offset: 0,
    };
    let chains = vec![
        Chain::identity_maps("reg_to_shared", 128, 32, vec![reg(&id7), sh(&id7)]),
        Chain::identity_maps("reg_to_shared_swz", 128, 32, vec![reg(&id7), sh(&swz)]),
        Chain::identity_maps("shared_to_shared_swz", 128, 32, vec![sh(&id7), sh(&swz)]),
        Chain::identity_maps(
            "shared_swz_to_reg",
            128,
            32,
            vec![sh(&swz), reg(&bit_reversal(7))],
        ),
        Chain::identity_maps(
            "reg_shared_reg_sandwich",
            128,
            32,
            vec![reg(&id7), sh(&swz), reg(&multi_round)],
        ),
    ];
    run_chains("shared_edges", &chains);
}

/// Non-identity `map`s: the ConvertLayout's logical permutation composes
/// with the layouts. Expected outputs are the host-side map compositions.
#[test]
fn gpu_non_identity_maps() {
    let id7 = LinearLayout::identity(7);
    let chains = vec![
        Chain {
            name: "map_rev_shared".into(),
            n: 128,
            block: 32,
            tiles: vec![(true, id7.clone()), (true, xor_swizzle(7, 3))],
            maps: vec![bit_reversal(7)],
        },
        Chain {
            name: "map_rev_reg".into(),
            n: 128,
            block: 32,
            tiles: vec![(false, id7.clone()), (false, lane_rotation(7))],
            maps: vec![bit_reversal(7)],
        },
        Chain {
            name: "map_chain_mixed".into(),
            n: 128,
            block: 32,
            tiles: vec![
                (false, id7.clone()),
                (true, xor_swizzle(7, 3)),
                (false, id7),
            ],
            maps: vec![bit_reversal(7), swap_bits(7, 1, 4)],
        },
    ];
    run_chains("non_identity_maps", &chains);
}

// ---------------------------------------------------------------------------
// GPU property tests: random layout chains
// ---------------------------------------------------------------------------

/// Random chains of 2–4 tiles with random spaces and random bijective
/// affine layouts. `block` picks the warp structure: 32 keeps every
/// composite warp-column identity (shuffle-only), 64 adds warp bits so
/// random composites also exercise the bounce.
fn random_chains(seed: u64, block: usize, count: usize, maps_random: bool) -> Vec<Chain> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|i| {
            let n = [64usize, 128, 256][rng.below(3) as usize].max(block);
            let kb = n.trailing_zeros() as usize;
            let tiles: Vec<(bool, LinearLayout)> = (0..2 + rng.below(3))
                .map(|_| (rng.below(2) == 1, rand_bijective(kb, &mut rng, true)))
                .collect();
            let maps: Vec<LinearLayout> = (0..tiles.len() - 1)
                .map(|_| {
                    if maps_random {
                        rand_bijective(kb, &mut rng, true)
                    } else {
                        LinearLayout::identity(kb)
                    }
                })
                .collect();
            Chain {
                name: format!("rnd_b{block}_{i}"),
                n,
                block,
                tiles,
                maps,
            }
        })
        .collect()
}

#[test]
fn gpu_prop_random_chains_block32() {
    run_chains("prop_chains_32", &random_chains(0xFACE_0032, 32, 10, false));
}

#[test]
fn gpu_prop_random_chains_block64() {
    run_chains("prop_chains_64", &random_chains(0xFACE_0064, 64, 8, false));
}

#[test]
fn gpu_prop_random_maps() {
    run_chains("prop_maps", &random_chains(0xFACE_1111, 32, 6, true));
}
