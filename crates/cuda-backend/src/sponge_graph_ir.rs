//! Graph-IR mirror of [`crate::sponge::DuplexSpongeGpu`].
//!
//! The host `DuplexSpongeGpu` runs its Poseidon2 sponge on the CPU and only
//! offloads to CUDA for expensive kernels (e.g. proof-of-work grinding). This
//! module instead expresses every `observe` / `sample` as a node on a
//! [`crypto_compiler::graph_ir::GraphBuilder`], so the whole transcript can be
//! baked into a compiled [`crypto_compiler::graph_exe::GraphExe`].
//!
//! Values that would be host-side `F` in `DuplexSpongeGpu` become [`BufId`]s
//! here: `observe(f: F)` becomes `observe(g, buf: BufId)` where `buf` is a
//! `[1]`-shaped BabyBear buffer, and `sample() -> F` becomes
//! `sample(g) -> BufId` that yields a freshly allocated `[1]`-shaped
//! BabyBear buffer. The sponge's internal state lives in a device buffer
//! that is threaded through each operation.
//!
//! The `absorb_idx` / `sample_idx` transcript positions are tracked at
//! graph-build time (the same static behavior `DuplexSpongeGpu` inherits from
//! `DuplexChallenger`), so every emitted kernel is branch-free: whether the
//! Poseidon2 permutation runs is a compile-time decision determined by the
//! host-side state of the [`DuplexSpongeGpuIR`] builder.
//!
//! # Module reuse
//!
//! Every reachable transcript state (indexed by `absorb_idx` / `sample_idx`)
//! corresponds to exactly one kernel shape. [`DuplexSpongeGpuIR::new`] builds
//! all of them up front and stores them as `Arc<ir::Module>` clones inside the
//! sponge, so subsequent `observe` / `sample` calls just hand a *clone* of
//! the relevant `Arc` to [`GraphBuilder::insert_kernel`].
//!
//! Position-dependent modules are *symbolic over the transcript position*:
//! the absorb / sample index enters the kernel body through
//! [`crypto_compiler::ir::IRBuilder::const_sym`] splices and is bound
//! per-node from the shape hint each operation passes to
//! [`GraphBuilder::insert_kernel`] (`&[pos]` — the position is not derivable
//! from the buffer shapes, which are all `[1, WIDTH]`). The spliced
//! parameter survives monomorphization as a runtime kernel argument, so
//! every position of one op kind lowers to the *same* residual and
//! [`crypto_compiler::graph_exe::GraphCompiler`]'s content-hash dedup
//! collapses them into a single JIT compilation — e.g. a transcript with
//! 100 non-permuting observes across 7 absorb positions ends up with 1
//! compiled kernel instead of 7.

use std::sync::Arc;

use crypto_compiler::{
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder},
    ir::{IRBuilder, Module, NodeId, ScalarType},
    kernel,
    kernels::Poseidon2Constants,
    poseidon2_parallel::poseidon2_permute_par,
    quast::Quast,
};

use crate::types::{CHUNK, D_EF, WIDTH};

/// Fiat-Shamir transcript expressed as nodes on a [`GraphBuilder`].
///
/// Mirrors [`openvm_stark_backend::FiatShamirTranscript`], but represents each
/// operation as a graph node: observed / sampled values live on the device
/// and are referenced through [`BufId`]s rather than passed by value.
///
/// # Buffer shapes
///
/// - `observe(g, value_buf)`: `value_buf` must be a `[1]`-shaped BabyBear buffer (4 bytes).
/// - `sample(g)`: returns a fresh `[1]`-shaped BabyBear buffer.
/// - `observe_ext(g, value_buf)`: `value_buf` must be a `[D_EF]`-shaped BabyBear buffer (16 bytes).
///   Its four basis coefficients are absorbed in order (same convention as
///   `SC::EF::as_basis_coefficients_slice`).
/// - `sample_ext(g)`: returns a fresh `[1, D_EF]`-shaped BabyBear buffer holding the sampled `EF`'s
///   four basis coefficients (16 bytes).
pub trait FiatShamirTranscriptGraphIR {
    /// Absorb one BabyBear from a `[1]`-shaped buffer into the sponge.
    fn observe(&mut self, g: &mut GraphBuilder, value_buf: BufId);

    /// Squeeze one BabyBear from the sponge into a fresh `[1]`-shaped buffer.
    fn sample(&mut self, g: &mut GraphBuilder) -> BufId;

    /// Absorb one `EF` value (four basis coefficients) from a `[D_EF]`-shaped
    /// buffer into the sponge.
    fn observe_ext(&mut self, g: &mut GraphBuilder, value_buf: BufId);

    /// Squeeze one `EF` value (four basis coefficients) into a fresh
    /// `[1, D_EF]`-shaped buffer.
    fn sample_ext(&mut self, g: &mut GraphBuilder) -> BufId;
}

/// Every kernel module a `DuplexSpongeGpuIR` may need over its lifetime,
/// stored as `Arc<Module>` clones so multiple `insert_kernel` calls of one
/// op kind all point at the same module instance.
///
/// Position-dependent modules are symbolic over the transcript position;
/// each operation supplies the concrete position as the `shape_hint`
/// argument of its `insert_kernel` call, so one module per op kind serves
/// every position — see the module-level docs.
struct SpongeModules {
    /// Insert a single value at absorb slot `hint[0]` (positions
    /// `0..CHUNK-1` — the permuting position uses [`Self::observe_perm`]).
    observe: Arc<Module>,
    /// Insert a value at slot `CHUNK - 1`, then permute the state.
    observe_perm: Arc<Module>,
    /// Read slot `hint[0]` (any index in `0..CHUNK`). State is unchanged
    /// (no state output).
    sample_no_perm: Arc<Module>,
    /// Standalone Poseidon2 permutation `state -> new_state`, shared by
    /// every permuting sample path (`sample` and `sample_ext`).
    permute: Arc<Module>,
    /// Statically unrolled `D_EF` observes starting at absorb slot
    /// `hint[0]` in `0..CHUNK-D_EF` (no permutation crossing).
    observe_ext: Arc<Module>,
    /// Statically unrolled `D_EF` observes starting at absorb slot
    /// `hint[0]` in `CHUNK-D_EF..CHUNK`, with the permutation in the
    /// middle.
    observe_ext_perm: Arc<Module>,
    /// Pack `D_EF` sample reads from a (pre, post) state pair: `hint[0]`
    /// reads walk down from `pre[hint[0] - 1]`, the rest walk down from
    /// `post[CHUNK - 1]`. The no-perm path passes the same state twice
    /// with `hint[0] = sample_idx >= D_EF`, so every read hits `pre`.
    sample_ext_pack: Arc<Module>,
}

impl SpongeModules {
    fn new() -> Self {
        let consts = Poseidon2Constants::p3_default();
        Self {
            observe: Arc::new(build_observe_module()),
            observe_perm: Arc::new(build_observe_module_par(CHUNK - 1, &consts)),
            sample_no_perm: Arc::new(build_sample_no_perm_module()),
            permute: Arc::new(build_permute_module(&consts)),
            observe_ext: Arc::new(build_observe_ext_module_serial()),
            observe_ext_perm: Arc::new(build_observe_ext_module_par(&consts)),
            sample_ext_pack: Arc::new(build_sample_ext_pack_module()),
        }
    }
}

/// Graph-IR mirror of [`crate::sponge::DuplexSpongeGpu`].
///
/// The Poseidon2-16 sponge state is laid out as a `[1, WIDTH]` BabyBear
/// tensor and lives in the device buffer pointed to by [`Self::state_buf`].
/// Every `observe` / `sample` allocates a fresh output buffer and inserts a
/// pre-built kernel module (see [`SpongeModules`]); the builder's `state_buf`
/// is then updated to the new state's [`BufId`], so subsequent operations
/// chain to the freshest state.
pub struct DuplexSpongeGpuIR {
    /// Device buffer for the current sponge state (`[1, WIDTH]` BabyBear,
    /// `WIDTH * 4` bytes).
    state_buf: BufId,
    /// Absorb position: `0 <= absorb_idx < CHUNK`.
    absorb_idx: usize,
    /// Sample position: `0 <= sample_idx <= CHUNK`.
    sample_idx: usize,
    device: DeviceType,
    /// Serial counter used to give each emitted buffer a distinct name.
    n_ops: usize,
    /// Pre-built kernel modules, cloned into the graph by each operation.
    modules: SpongeModules,
}

impl DuplexSpongeGpuIR {
    /// Creates a fresh transcript with all-zero sponge state on `device`.
    ///
    /// Builds every kernel module the transcript may emit ahead of time and
    /// stores them as `Arc<Module>` clones. Adds a `[1, WIDTH]` BabyBear state
    /// buffer to `g` and emits a `Memset(0)` node that initializes it
    /// (matching the default `DuplexSpongeGpu` starting state).
    pub fn new(g: &mut GraphBuilder, device: DeviceType) -> Self {
        let state_buf = alloc_state_buf(g, device, "sponge_state_init");
        g.insert_memset(state_buf, 0);
        Self {
            state_buf,
            absorb_idx: 0,
            sample_idx: 0,
            device,
            n_ops: 0,
            modules: SpongeModules::new(),
        }
    }

    /// Current state buffer id (mostly for tests / inspection).
    pub fn state_buf(&self) -> BufId {
        self.state_buf
    }

    fn observe_triggers_perm(&self) -> bool {
        self.absorb_idx + 1 == CHUNK
    }

    fn sample_triggers_perm(&self) -> bool {
        self.absorb_idx != 0 || self.sample_idx == 0
    }

    /// Pack hint for the permuting `sample_ext` path: how many samples
    /// happen before the (unique) permutation fires.
    fn pre_perm_reads(&self) -> usize {
        if self.absorb_idx != 0 || self.sample_idx == 0 {
            // The very first sample already permutes.
            0
        } else {
            // sample_idx samples happen from pre-perm state (reading indices
            // sample_idx - 1 down to 0), then the next sample permutes.
            self.sample_idx
        }
    }
}

impl FiatShamirTranscriptGraphIR for DuplexSpongeGpuIR {
    fn observe(&mut self, g: &mut GraphBuilder, value_buf: BufId) {
        let permute = self.observe_triggers_perm();
        let new_state_buf = alloc_state_buf(
            g,
            self.device,
            &format!("sponge_state_after_observe_{}", self.n_ops),
        );
        if permute {
            let module = self.modules.observe_perm.clone();
            g.insert_kernel(module, [self.state_buf, value_buf], [new_state_buf], &[]);
        } else {
            let module = self.modules.observe.clone();
            g.insert_kernel(
                module,
                [self.state_buf, value_buf],
                [new_state_buf],
                &[("i", self.absorb_idx as i64)],
            );
        }
        self.state_buf = new_state_buf;
        if permute {
            self.absorb_idx = 0;
            self.sample_idx = CHUNK;
        } else {
            self.absorb_idx += 1;
        }
        self.n_ops += 1;
    }

    fn sample(&mut self, g: &mut GraphBuilder) -> BufId {
        if self.sample_triggers_perm() {
            // Permute the state, then read the top slot of the new state
            // with the shared single-slot reader.
            let new_state_buf = alloc_state_buf(
                g,
                self.device,
                &format!("sponge_state_after_sample_{}", self.n_ops),
            );
            let sample_buf =
                alloc_single_f_buf(g, self.device, &format!("sponge_sample_{}", self.n_ops));
            let permute = self.modules.permute.clone();
            g.insert_kernel(permute, [self.state_buf], [new_state_buf], &[]);
            let reader = self.modules.sample_no_perm.clone();
            g.insert_kernel(
                reader,
                [new_state_buf],
                [sample_buf],
                &[("i", (CHUNK - 1) as i64)],
            );
            self.state_buf = new_state_buf;
            self.absorb_idx = 0;
            // The permutation resets `sample_idx = CHUNK`, then the read
            // decrements it once.
            self.sample_idx = CHUNK - 1;
            self.n_ops += 1;
            sample_buf
        } else {
            // No permutation, no state update — just read one slot into a
            // fresh output buffer. `self.state_buf` stays the same.
            let read_idx = self.sample_idx - 1;
            let sample_buf =
                alloc_single_f_buf(g, self.device, &format!("sponge_sample_{}", self.n_ops));
            let module = self.modules.sample_no_perm.clone();
            g.insert_kernel(
                module,
                [self.state_buf],
                [sample_buf],
                &[("i", read_idx as i64)],
            );
            self.sample_idx = read_idx;
            self.n_ops += 1;
            sample_buf
        }
    }

    fn observe_ext(&mut self, g: &mut GraphBuilder, value_buf: BufId) {
        let module = if self.absorb_idx >= CHUNK - D_EF {
            self.modules.observe_ext_perm.clone()
        } else {
            self.modules.observe_ext.clone()
        };
        let new_state_buf = alloc_state_buf(
            g,
            self.device,
            &format!("sponge_state_after_observe_ext_{}", self.n_ops),
        );
        g.insert_kernel(
            module,
            [self.state_buf, value_buf],
            [new_state_buf],
            &[("p", self.absorb_idx as i64)],
        );
        self.state_buf = new_state_buf;
        // Simulate the four host-side observes to update the transcript state.
        let (abs, sam) = simulate_ext_observes(self.absorb_idx, self.sample_idx);
        self.absorb_idx = abs;
        self.sample_idx = sam;
        self.n_ops += 1;
    }

    fn sample_ext(&mut self, g: &mut GraphBuilder) -> BufId {
        // Statically unroll four samples to decide whether any of them
        // triggers a permutation.
        let (abs, sam, any_perm) = simulate_ext_samples(self.absorb_idx, self.sample_idx);
        let ext_buf = alloc_ext_buf(g, self.device, &format!("sponge_sample_ext_{}", self.n_ops));
        if any_perm {
            let new_state_buf = alloc_state_buf(
                g,
                self.device,
                &format!("sponge_state_after_sample_ext_{}", self.n_ops),
            );
            let pre_reads = self.pre_perm_reads();
            debug_assert!(pre_reads < D_EF);
            let permute = self.modules.permute.clone();
            g.insert_kernel(permute, [self.state_buf], [new_state_buf], &[]);
            let pack = self.modules.sample_ext_pack.clone();
            g.insert_kernel(
                pack,
                [self.state_buf, new_state_buf],
                [ext_buf],
                &[("p", pre_reads as i64)],
            );
            self.state_buf = new_state_buf;
        } else {
            // No perm ⇒ absorb_idx == 0 and sample_idx ∈ D_EF..=CHUNK, so
            // `k < sample_idx` holds for every read and the pack's `post`
            // input is never touched — pass the same state twice.
            debug_assert!(self.sample_idx >= D_EF);
            let pack = self.modules.sample_ext_pack.clone();
            g.insert_kernel(
                pack,
                [self.state_buf, self.state_buf],
                [ext_buf],
                &[("p", self.sample_idx as i64)],
            );
        }
        self.absorb_idx = abs;
        self.sample_idx = sam;
        self.n_ops += 1;
        ext_buf
    }
}

// ---------------------------------------------------------------------------
// Buffer allocation helpers.

fn alloc_state_buf(g: &mut GraphBuilder, device: DeviceType, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst((WIDTH as i64) * 4),
        elem_size: 4,
    })
}

fn alloc_single_f_buf(g: &mut GraphBuilder, device: DeviceType, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst(4),
        elem_size: 4,
    })
}

fn alloc_ext_buf(g: &mut GraphBuilder, device: DeviceType, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst((D_EF as i64) * 4),
        elem_size: 4,
    })
}

// ---------------------------------------------------------------------------
// Static simulation of the host sponge, used to update the builder's absorb /
// sample indices in step with the emitted kernels.

fn step_observe(abs: usize, sam: usize) -> (usize, usize) {
    let new_abs = abs + 1;
    if new_abs == CHUNK {
        (0, CHUNK)
    } else {
        (new_abs, sam)
    }
}

fn step_sample(abs: usize, sam: usize) -> (usize, usize, bool) {
    // Returns (new_abs, new_sam, permuted).
    if abs != 0 || sam == 0 {
        // Permute, then decrement sample_idx once.
        (0, CHUNK - 1, true)
    } else {
        (abs, sam - 1, false)
    }
}

fn simulate_ext_observes(mut abs: usize, mut sam: usize) -> (usize, usize) {
    for _ in 0..D_EF {
        let (a, s) = step_observe(abs, sam);
        abs = a;
        sam = s;
    }
    (abs, sam)
}

fn simulate_ext_samples(mut abs: usize, mut sam: usize) -> (usize, usize, bool) {
    let mut any_perm = false;
    for _ in 0..D_EF {
        let (a, s, p) = step_sample(abs, sam);
        abs = a;
        sam = s;
        if p {
            any_perm = true;
        }
    }
    (abs, sam, any_perm)
}

// ---------------------------------------------------------------------------
// Kernel builders (pure functions of the transcript position — invoked once
// per unique position during `SpongeModules::new`).

/// `observe(state[1, WIDTH], value[1]) -> new_state[1, WIDTH]`
///
/// Non-permuting variant: a single-thread compute that rewrites slot
/// `hint[0]` and passes every other slot through — the work is a one-slot
/// store, so single-thread launch is the right shape. Symbolic over the
/// absorb position (`const_sym` splice bound from the insert-site shape
/// hint), so all `CHUNK - 1` non-permuting positions share one JIT'd
/// kernel. The permuting position `CHUNK - 1` uses
/// [`build_observe_module_par`] instead.
fn build_observe_module() -> Module {
    let mut b = IRBuilder::new();
    let pos = b.symbol("i");
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![1]);
    let body = b.compute(1, move |b, _i| {
        let pos_c = b.const_sym(pos);
        let val = kernel!(b, value[0]);
        let mut s = load_state(b, state);
        // Only positions 0..CHUNK-1 reach this module (position CHUNK - 1
        // permutes and uses the parallel variant), so the remaining slots
        // always pass through.
        for (j, slot) in s.iter_mut().enumerate().take(CHUNK - 1) {
            let j_c = b.const_u32(j as u32);
            let cond = b.eq(j_c, pos_c);
            *slot = b.select(cond, val, *slot);
        }
        b.pack(&s)
    });
    b.finish("sponge_observe", body)
}

/// Warp-parallel variant of [`build_observe_module`]: one warp per launch,
/// gather → replace slot `absorb_idx` → Poseidon2 permutation → store.
fn build_observe_module_par(absorb_idx: usize, consts: &Poseidon2Constants) -> Module {
    let mut b = IRBuilder::new();
    let state_in = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value_in = b.input("value", ScalarType::BabyBear, vec![1]);
    let consts = consts.clone();
    let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        let gathered = b.compute_with(WIDTH, None, Some(par), None, move |b, j| {
            let cond = kernel!(b, j == #absorb_idx);
            let val = kernel!(b, value_in[0]);
            let existing = kernel!(b, state_in[0, j]);
            b.select(cond, val, existing)
        });
        b.bind(gathered, move |b, tile| {
            poseidon2_permute_par(b, tile, &consts, |b, v| {
                let par = b.par_map(|th, _s, _c| th.clone());
                b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, v[j]))
            })
        })
    });
    b.finish(format!("sponge_observe_perm_at_{absorb_idx}"), body)
}

/// `sample_no_perm(state[1, WIDTH]) -> sample[1]`. State is unchanged, so no
/// state output. Symbolic over the read index (insert-site hint), so all
/// `CHUNK` read positions share one JIT'd kernel.
fn build_sample_no_perm_module() -> Module {
    let mut b = IRBuilder::new();
    let pos = b.symbol("i");
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let body = b.compute(1, move |b, _i| {
        let zero = b.const_u32(0);
        let pos_c = b.const_sym(pos);
        b.index(state, &[zero, pos_c])
    });
    b.finish("sponge_sample", body)
}

/// `permute(state[1, WIDTH]) -> new_state[1, WIDTH]` — standalone
/// warp-parallel Poseidon2 permutation (gather → [`poseidon2_permute_par`]
/// → store), shared by every permuting sample path. The permuting sample
/// reads happen in separate follow-up kernels
/// ([`build_sample_no_perm_module`] / [`build_sample_ext_pack_module`]),
/// so one JIT'd permutation serves them all.
fn build_permute_module(consts: &Poseidon2Constants) -> Module {
    let mut b = IRBuilder::new();
    let state_in = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let consts = consts.clone();
    let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        let gathered = b.compute_with(
            WIDTH,
            None,
            Some(par),
            None,
            |b, j| kernel!(b, state_in[0, j]),
        );
        b.bind(gathered, move |b, tile| {
            poseidon2_permute_par(b, tile, &consts, |b, v| {
                let par = b.par_map(|th, _s, _c| th.clone());
                b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, v[j]))
            })
        })
    });
    b.finish("sponge_permute", body)
}

/// `observe_ext(state[1, WIDTH], value[D_EF]) -> new_state[1, WIDTH]`.
///
/// Statically unrolls `D_EF` observes; when the sequence crosses the
/// `absorb_idx == CHUNK` boundary a Poseidon2 permutation is emitted between
/// the pre- and post-perm inserts.
///
/// Both variants are symbolic over the starting absorb position (`const_sym`
/// splices bound from the insert-site shape hint), so each variant is one
/// JIT'd kernel:
///
/// - Non-permuting variant ([`build_observe_ext_module_serial`], starts `< CHUNK - D_EF`):
///   single-thread compute that writes `D_EF` slots and passes the rest through.
/// - Permuting variant ([`build_observe_ext_module_par`], starts `>= CHUNK - D_EF`): warp-parallel
///   — one `compute_with(WIDTH, par)` fuses the gather with the pre-perm inserts,
///   [`poseidon2_permute_par`] runs the round schedule, and the final `compute_with(WIDTH, par)`
///   fuses the store with any post-perm inserts.
///
/// Serial `observe_ext` (no permutation crossing): slots
/// `p..p + D_EF` receive `value[0..D_EF]`. The insert site only uses this
/// variant for starts `<= CHUNK - D_EF`, so every write lands in slots
/// `0..CHUNK`.
fn build_observe_ext_module_serial() -> Module {
    let mut b = IRBuilder::new();
    let pos = b.symbol("p");
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![D_EF]);
    let body = b.compute(1, move |b, _i| {
        let pos_c = b.const_sym(pos);
        let d_c = b.const_u32(D_EF as u32);
        let mut s = load_state(b, state);
        for (j, slot) in s.iter_mut().enumerate().take(CHUNK) {
            // Slot `j` takes `value[j - p]` iff `p <= j < p + D_EF`. The
            // single `j - p < D_EF` guard covers both bounds via u32
            // wraparound: for `j < p` the subtraction wraps to a huge
            // value. `select` is short-circuit, so the (would-be OOB)
            // load in the untaken branch never executes.
            let j_c = b.const_u32(j as u32);
            let vidx = b.sub(j_c, pos_c);
            let cond = b.lt(vidx, d_c);
            let val = b.index(value, &[vidx]);
            *slot = b.select(cond, val, *slot);
        }
        b.pack(&s)
    });
    b.finish("sponge_observe_ext", body)
}

/// Warp-parallel `observe_ext` with a permutation in the middle.  The gather
/// stage fuses the pre-perm inserts (`value[j - p]` into slots
/// `p <= j < CHUNK`); the final store fuses the post-perm inserts
/// (`value[j + CHUNK - p]` into slots `j < p + D_EF - CHUNK`).
fn build_observe_ext_module_par(consts: &Poseidon2Constants) -> Module {
    let mut b = IRBuilder::new();
    let pos = b.symbol("p");
    let state_in = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value_in = b.input("value", ScalarType::BabyBear, vec![D_EF]);
    let consts = consts.clone();
    let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        // Gather + pre-perm inserts: lane `j` overrides its slot with
        // `value[j - p]` when `p <= j < CHUNK`. `j - p < D_EF` covers the
        // lower bound via u32 wraparound (and the upper one because
        // `CHUNK - p <= D_EF` here); the explicit `j < CHUNK` keeps the
        // capacity lanes out.
        let par = b.par_map(|th, _s, _c| th.clone());
        let gathered = b.compute_with(WIDTH, None, Some(par), None, move |b, j| {
            let existing = kernel!(b, state_in[0, j]);
            let pos_c = b.const_sym(pos);
            let d_c = b.const_u32(D_EF as u32);
            let chunk_c = b.const_u32(CHUNK as u32);
            let vidx = b.sub(j, pos_c);
            let in_value = b.lt(vidx, d_c);
            let in_chunk = b.lt(j, chunk_c);
            let val = b.index(value_in, &[vidx]);
            let with_val = b.select(in_value, val, existing);
            b.select(in_chunk, with_val, existing)
        });
        b.bind(gathered, move |b, tile| {
            poseidon2_permute_par(b, tile, &consts, move |b, v| {
                // Store + post-perm inserts: lane `j` takes
                // `value[j + CHUNK - p]` when that index is `< D_EF`
                // (`j + CHUNK - p` never wraps since `p < CHUNK`).
                let par = b.par_map(|th, _s, _c| th.clone());
                b.compute_with(WIDTH, None, Some(par), None, move |b, j| {
                    let existing = kernel!(b, v[j]);
                    let pos_c = b.const_sym(pos);
                    let d_c = b.const_u32(D_EF as u32);
                    let chunk_c = b.const_u32(CHUNK as u32);
                    let shifted = b.add(j, chunk_c);
                    let vidx = b.sub(shifted, pos_c);
                    let cond = b.lt(vidx, d_c);
                    let val = b.index(value_in, &[vidx]);
                    b.select(cond, val, existing)
                })
            })
        })
    });
    b.finish("sponge_observe_ext_perm", body)
}

/// `sample_ext_pack(pre[1, WIDTH], post[1, WIDTH]) -> samples[1, D_EF]` —
/// the unified sample-read pack behind every `sample_ext`:
///
///   samples[k]  =  pre[0, p - 1 - k]             if k < p
///   samples[k]  =  post[0, CHUNK - 1 - (k - p)]  otherwise
///
/// (The same read pattern as the host `DuplexSpongeGpu`: pre-perm reads
/// walk down from `sample_idx - 1`, then the permutation kicks
/// `sample_idx` to CHUNK and post-perm reads walk down from `CHUNK - 1`.)
///
/// The permuting path binds `pre` to the old state, `post` to the freshly
/// permuted state (see [`build_permute_module`]) and `p = pre_reads <
/// D_EF`. The no-perm path binds the *same* state to both inputs with
/// `p = sample_idx >= D_EF`, making `k < p` true for every read — one
/// JIT'd kernel serves both paths at every position.
///
/// With `p` symbolic, sample `k` branches on `k < p`. The pre-perm index
/// `p - 1 - k` wraps for `k >= p`, but that branch is untaken and
/// `select` is short-circuit, so the load never executes. The post index
/// `CHUNK - 1 - k + p` stays within `WIDTH` even when its branch is
/// untaken (`p <= CHUNK`, so it is at most `CHUNK - 1 + CHUNK = 15`).
fn build_sample_ext_pack_module() -> Module {
    let mut b = IRBuilder::new();
    let pos = b.symbol("p");
    let pre = b.input("pre", ScalarType::BabyBear, vec![1, WIDTH]);
    let post = b.input("post", ScalarType::BabyBear, vec![1, WIDTH]);
    let body = b.compute(1, move |b, _i| {
        let zero = b.const_u32(0);
        let pos_c = b.const_sym(pos);
        let mut samples = Vec::with_capacity(D_EF);
        for k in 0..D_EF {
            let k_c = b.const_u32(k as u32);
            let cond = b.lt(k_c, pos_c);
            let k1_c = b.const_u32((k + 1) as u32);
            let pre_idx = b.sub(pos_c, k1_c);
            let pre_val = b.index(pre, &[zero, pre_idx]);
            let base_c = b.const_u32((CHUNK - 1 - k) as u32);
            let post_idx = b.add(base_c, pos_c);
            let post_val = b.index(post, &[zero, post_idx]);
            samples.push(b.select(cond, pre_val, post_val));
        }
        b.pack(&samples)
    });
    b.finish("sponge_sample_ext_pack", body)
}

/// Loads the 16-element sponge state from a `state[1, WIDTH]` input into an
/// array of `NodeId`s ready for in-place updates.
fn load_state(b: &mut IRBuilder, state: NodeId) -> [NodeId; 16] {
    let mut s = [b.const_u32(0); 16];
    for (j, slot) in s.iter_mut().enumerate() {
        *slot = kernel!(b, state[0, #j]);
    }
    s
}

#[cfg(test)]
mod tests {
    use crypto_compiler::graph_exe::GraphCompiler;
    use openvm_cuda_common::{
        common::get_device,
        copy::MemCopyH2D,
        stream::{CudaStream, GpuDeviceCtx, StreamGuard},
    };
    use openvm_stark_backend::FiatShamirTranscript;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{
        prelude::SC,
        sponge::DuplexSpongeGpu,
        types::{D_EF, F},
    };

    fn test_ctx() -> GpuDeviceCtx {
        GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        }
    }

    fn f_from_u32(x: u32) -> F {
        F::from_u32(x)
    }

    fn add_input_f_buf(g: &mut GraphBuilder, name: &str) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.to_string()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(4),
            elem_size: 4,
        })
    }

    fn add_input_ext_buf(g: &mut GraphBuilder, name: &str) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.to_string()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst((D_EF as i64) * 4),
            elem_size: 4,
        })
    }

    fn f_to_bytes(f: F) -> [u8; 4] {
        // Raw p3 `BabyBear` memory layout — Montgomery-encoded u32,
        // little-endian. Matches the on-device encoding the DSL's
        // Montgomery codegen reads/writes for `ScalarType::BabyBear`.
        unsafe { std::mem::transmute::<F, [u8; 4]>(f) }
    }

    fn bytes_to_f(bytes: &[u8]) -> F {
        assert_eq!(bytes.len(), 4);
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const F) }
    }

    /// Op kinds a test can sequence against both transcripts.
    #[derive(Clone, Copy)]
    enum Op {
        Observe(F),
        ObserveExt([F; 4]),
        Sample,
        SampleExt,
    }

    /// Runs `ops` on a fresh `DuplexSpongeGpuIR` (compiling and executing the
    /// resulting graph on the GPU) and returns the ordered list of sampled F
    /// values. Sample_ext contributes four values in coefficient order.
    fn run_ir(ctx: &GpuDeviceCtx, ops: &[Op]) -> Vec<F> {
        let mut g = GraphBuilder::new();
        let mut sponge = DuplexSpongeGpuIR::new(&mut g, DeviceType::Cuda(0));

        // Register input buffers for every observe operation.
        let mut observe_bufs: Vec<(BufId, Vec<u8>)> = Vec::new();
        // Register output buffers for every sample operation, tagged with the
        // number of F values in the output.
        let mut sample_bufs: Vec<(BufId, usize)> = Vec::new();

        for (i, op) in ops.iter().enumerate() {
            match op {
                Op::Observe(f) => {
                    let buf = add_input_f_buf(&mut g, &format!("in_{i}"));
                    let bytes = f_to_bytes(*f).to_vec();
                    observe_bufs.push((buf, bytes));
                    sponge.observe(&mut g, buf);
                }
                Op::ObserveExt(vals) => {
                    let buf = add_input_ext_buf(&mut g, &format!("in_ext_{i}"));
                    let mut bytes = Vec::with_capacity(16);
                    for v in vals {
                        bytes.extend_from_slice(&f_to_bytes(*v));
                    }
                    observe_bufs.push((buf, bytes));
                    sponge.observe_ext(&mut g, buf);
                }
                Op::Sample => {
                    let out = sponge.sample(&mut g);
                    sample_bufs.push((out, 1));
                }
                Op::SampleExt => {
                    let out = sponge.sample_ext(&mut g);
                    sample_bufs.push((out, D_EF));
                }
            }
        }

        // Declare the graph interface: observe values are inputs, sampled
        // values are outputs; the sponge state chain stays internal.
        for (buf, _) in &observe_bufs {
            g.register_input(*buf);
        }
        for (buf, _) in &sample_bufs {
            g.register_output(*buf);
        }

        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile(g)
            .expect("graph compile");

        // Bind each registered input to its slot in the compiled exe.
        for i in 0..exe.num_inputs() {
            let bid = exe.input_buf_id(i);
            let (_, bytes) = observe_bufs
                .iter()
                .find(|(b, _)| *b == bid)
                .expect("input buf not found");
            let dbuf = bytes.as_slice().to_device_on(ctx).expect("H2D");
            exe.set_input(ctx, i, &dbuf).expect("set_input");
        }
        exe.run(ctx).expect("run");

        // Now collect the sampled bytes in the order the caller requested.
        let mut result = Vec::new();
        for (bid, len) in sample_bufs {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == bid)
                .expect("sample buf not found in exe outputs");
            let bytes = exe.get_output(idx).to_host_on(ctx).expect("D2H");
            for k in 0..len {
                result.push(bytes_to_f(&bytes[4 * k..4 * (k + 1)]));
            }
        }
        result
    }

    /// Runs the same op sequence on a plain `DuplexSpongeGpu` and returns the
    /// list of sampled F values, in the same order as `run_ir`.
    fn run_host(ops: &[Op]) -> Vec<F> {
        let mut sponge = DuplexSpongeGpu::default();
        let mut out = Vec::new();
        for op in ops {
            match op {
                Op::Observe(f) => FiatShamirTranscript::<SC>::observe(&mut sponge, *f),
                Op::ObserveExt(vals) => {
                    for v in vals {
                        FiatShamirTranscript::<SC>::observe(&mut sponge, *v);
                    }
                }
                Op::Sample => out.push(FiatShamirTranscript::<SC>::sample(&mut sponge)),
                Op::SampleExt => {
                    for _ in 0..D_EF {
                        out.push(FiatShamirTranscript::<SC>::sample(&mut sponge));
                    }
                }
            }
        }
        out
    }

    fn assert_match(ops: &[Op]) {
        let ctx = test_ctx();
        let got = run_ir(&ctx, ops);
        let want = run_host(ops);
        assert_eq!(got.len(), want.len(), "sample count mismatch");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "sample {i} mismatch: got {g:?}, want {w:?}");
        }
    }

    #[test]
    fn observe_then_sample_no_perm() {
        // absorb_idx = 0 -> after one observe, absorb_idx = 1 (no perm on
        // observe). First sample: absorb_idx != 0, permutes then reads.
        // Second sample: absorb_idx = 0 and sample_idx > 0, no perm.
        let ops = [Op::Observe(f_from_u32(7)), Op::Sample, Op::Sample];
        assert_match(&ops);
    }

    #[test]
    fn observe_fills_chunk_and_triggers_perm() {
        // Fill up the absorb block (CHUNK observes) so the last observe
        // itself triggers a permutation.
        let mut ops = Vec::new();
        for i in 0..CHUNK {
            ops.push(Op::Observe(f_from_u32(1 + i as u32 * 7)));
        }
        for _ in 0..3 {
            ops.push(Op::Sample);
        }
        assert_match(&ops);
    }

    #[test]
    fn first_sample_forces_perm_from_zero_state() {
        // Fresh sponge -> sample_idx = 0 so the first sample triggers a
        // permutation and reads state[CHUNK - 1].
        let ops = [Op::Sample, Op::Sample];
        assert_match(&ops);
    }

    #[test]
    fn observe_ext_and_sample_ext() {
        let ops = [
            Op::ObserveExt([f_from_u32(1), f_from_u32(2), f_from_u32(3), f_from_u32(4)]),
            Op::SampleExt,
        ];
        assert_match(&ops);
    }

    #[test]
    fn observe_ext_boundary_triggers_perm_inside_kernel() {
        // Start with 6 single observes -> absorb_idx = 6; then observe_ext
        // absorbs 4 more, triggering a permutation on the 2nd inner step
        // and continuing after the reset.
        let mut ops: Vec<Op> = (0..6)
            .map(|i| Op::Observe(f_from_u32(i as u32 + 1)))
            .collect();
        ops.push(Op::ObserveExt([
            f_from_u32(101),
            f_from_u32(102),
            f_from_u32(103),
            f_from_u32(104),
        ]));
        ops.push(Op::SampleExt);
        ops.push(Op::Sample);
        assert_match(&ops);
    }

    #[test]
    fn sample_ext_crosses_perm_boundary() {
        // After a permutation from an initial sample, sample_idx = CHUNK - 1.
        // A subsequent sample_ext exhausts the remaining CHUNK - 1 slots and
        // permutes once inside the kernel.
        let ops = [Op::Sample, Op::SampleExt, Op::SampleExt, Op::SampleExt];
        assert_match(&ops);
    }

    #[test]
    fn interleaved_observe_sample() {
        let mut ops = Vec::new();
        let mut seed: u32 = 1_234_567;
        for i in 0..20 {
            match i % 4 {
                0 => ops.push(Op::Observe(f_from_u32(seed))),
                1 => ops.push(Op::Sample),
                2 => ops.push(Op::ObserveExt([
                    f_from_u32(seed),
                    f_from_u32(seed.wrapping_add(1)),
                    f_from_u32(seed.wrapping_add(2)),
                    f_from_u32(seed.wrapping_add(3)),
                ])),
                _ => ops.push(Op::SampleExt),
            }
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        }
        assert_match(&ops);
    }

    /// Sanity check: emitting many ops through the sponge produces a graph
    /// whose content-hash dedup collapses the kernel count down to one
    /// module per op kind (positions are symbolic runtime params).  Also
    /// asserts on the exact count for the emitted sequence so a regression
    /// that broke sharing would show up here.
    #[test]
    fn modules_are_deduplicated_across_operations() {
        // 20 observes from state (0, 0) walk absorb_idx through
        //   0,1,2,3,4,5,6,7 (perm) 0,1,2,3,4,5,6,7 (perm) 0,1,2,3
        // — positions 0..=6 all bind the one symbolic non-perm `observe`
        // module (1 unique after content dedup) and position 7 uses the
        // concrete permuting variant (1 unique).
        //
        // Then 15 samples from state (4, 8) walk:
        //   perm → sample_no_perm[6..=0] → perm → sample_no_perm[6..=1]
        // — permuting samples emit the standalone `permute` module (1
        // unique) followed by a `sample_no_perm` read of the new state, and
        // every `sample_no_perm` read index binds the one symbolic module
        // (1 unique).
        //
        // Expected total unique kernel modules: 2 + 2 = 4, vs. the
        // 20 + 15 = 35 emitted sponge operations.
        let mut g = GraphBuilder::new();
        let mut sponge = DuplexSpongeGpuIR::new(&mut g, DeviceType::Cuda(0));
        for i in 0..20 {
            let buf = add_input_f_buf(&mut g, &format!("in_{i}"));
            g.register_input(buf);
            sponge.observe(&mut g, buf);
        }
        for _ in 0..15 {
            let out = sponge.sample(&mut g);
            g.register_output(out);
        }
        // Fusion would rewrite the emitted modules (the state chain is
        // internal, hence fusable) and change the count under test; this
        // test is about the Arc/hash dedup accounting of the emitted nodes.
        let exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .without_fusion()
            .compile(g)
            .expect("graph compile");
        assert_eq!(
            exe.num_unique_modules(),
            4,
            "expected exactly 4 unique kernel modules (2 observe + permute + \
             sample read) for the emitted 20 observes + 15 samples"
        );
    }
}
