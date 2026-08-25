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
//! [`crypto_compiler::graph_compiler::GraphCompiler`]'s content-hash dedup
//! collapses them into a single JIT compilation — e.g. a transcript with
//! 100 non-permuting observes across 7 absorb positions ends up with 1
//! compiled kernel instead of 7.

use std::{ffi::c_void, sync::Arc};

use crypto_compiler::{
    graph_exe::GraphExe,
    graph_ir::{BufId, BufInfo, ConstBuf, DeviceType, GraphBuilder},
    ir::{IRBuilder, Module, NodeId, ScalarType},
    kernel,
    kernels::Poseidon2Constants,
    poseidon2_parallel::poseidon2_permute_par,
    quast::Quast,
    CompileError,
};
use openvm_cuda_common::{copy::cuda_memcpy_on, stream::GpuDeviceCtx};

use crate::{
    sponge::SpongeSnapshot,
    types::{CHUNK, D_EF, WIDTH},
};

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

    /// The buffer holding the sponge state **as of now**, without advancing
    /// the transcript.
    ///
    /// A phase driver needs this to name the tail of its Fiat-Shamir chain:
    /// registering it as a graph output is what stops DCE from deleting every
    /// transcript node. Squeezing one extra value would do the same job but
    /// would also advance the sponge one step past the eager phase, so the
    /// two transcripts would no longer agree.
    fn state_buf(&self) -> BufId;
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

    /// Creates a transcript that **continues** a live [`crate::sponge::DuplexSpongeGpu`]
    /// instead of starting from the all-zero state.
    ///
    /// A phase that runs mid-transcript (the logup-zerocheck phase starts
    /// after grinding and the fractional GKR phase have already absorbed and
    /// squeezed) cannot use [`Self::new`]: it would restart the Fiat-Shamir
    /// stream. `from_live` takes the snapshot produced by
    /// [`crate::sponge::DuplexSpongeGpu::snapshot`] and seeds the graph with it,
    /// so the emitted `observe` / `sample` nodes chain onto the live stream.
    ///
    /// The seeded state is emitted as a `Const` node carrying the snapshot's
    /// `WIDTH * 4` bytes (the graph runtime stages them once, before the run).
    /// The buffer is therefore **read-only**, which is exactly how the sponge
    /// uses it: every `observe` / `sample` writes a *fresh* state buffer and
    /// never mutates its input state.
    ///
    /// The `absorb_idx` / `sample_idx` positions are carried over verbatim, so
    /// the build-time simulation that decides which kernel shape each op emits
    /// (permuting vs not) picks up exactly where the live sponge left off.
    ///
    /// [`Self::new`] is unchanged: `new` still memsets to zero.
    ///
    /// # Prefer [`Self::from_live_input`]
    ///
    /// The seed here is baked in as a `Const` node, so the resulting graph is
    /// valid for **exactly one transcript** and cannot be reused for a second
    /// proof — the state bytes are part of the graph definition, so a new
    /// transcript means a new `GraphCompiler::compile` (30s-192s in this
    /// crate). [`Self::from_live_input`] registers the state as a runtime
    /// graph input instead, which keeps the compiled graph reusable across
    /// proofs; use it for anything that is compiled once and run many times.
    ///
    /// This constructor is kept because it needs no post-compile bind step,
    /// which makes it convenient for build-only tests.
    pub fn from_live(g: &mut GraphBuilder, device: DeviceType, snap: &SpongeSnapshot) -> Self {
        assert!(
            (snap.absorb_idx as usize) < CHUNK,
            "seed absorb_idx {} out of range (must be < CHUNK = {CHUNK})",
            snap.absorb_idx,
        );
        assert!(
            (snap.sample_idx as usize) <= CHUNK,
            "seed sample_idx {} out of range (must be <= CHUNK = {CHUNK})",
            snap.sample_idx,
        );
        let state_buf = alloc_state_buf(g, device, "sponge_state_seed");
        g.insert_const(state_buf, ConstBuf::HostBuf(snap.state_bytes()));
        Self {
            state_buf,
            absorb_idx: snap.absorb_idx as usize,
            sample_idx: snap.sample_idx as usize,
            device,
            n_ops: 0,
            modules: SpongeModules::new(),
        }
    }

    /// Creates a transcript that **continues** a live [`crate::sponge::DuplexSpongeGpu`],
    /// with the seed state supplied at *run* time rather than baked into the
    /// graph — so the compiled graph is reusable across proofs.
    ///
    /// Returns `(transcript, seed_buf)`. `seed_buf` is registered as a graph
    /// input (`GraphBuilder::register_input`) holding the `[1, WIDTH]`
    /// BabyBear sponge state; the caller must fill it after compilation, most
    /// conveniently via [`bind_sponge_seed`].
    ///
    /// # Why this exists (vs [`Self::from_live`])
    ///
    /// The two halves of a seed live at different times:
    ///
    /// - `absorb_idx` / `sample_idx` are **build**-time data. They decide which kernel module each
    ///   `observe` / `sample` emits (permuting vs not), so they must be known while the graph is
    ///   being constructed. They are copied out of `snap` here.
    /// - the 16-word state is **run**-time data. No graph node inspects it at build time, so it can
    ///   be an ordinary input.
    ///
    /// Splitting them this way is what makes graph caching possible: for a
    /// fixed AIR/config the positions are deterministic across proofs, while
    /// the state bytes differ every time. [`Self::from_live`] bakes the state
    /// into a `Const`, which forces a fresh `GraphCompiler::compile` per proof
    /// — prohibitive at 30s-192s per compile.
    ///
    /// # The snapshot is borrowed, not consumed
    ///
    /// Graph construction reads only `snap.position()`. The caller keeps
    /// ownership because it still needs `snap.state()` *after* compilation, at
    /// bind time. Nothing here retains a pointer into the live sponge — the
    /// snapshot is an owned, pointer-free value, so the live sponge may be
    /// advanced or dropped in between.
    ///
    /// # No initializer node
    ///
    /// Unlike [`Self::new`], this emits no `Memset`: registered graph inputs
    /// must not be written by any graph node (validated at compile time).
    ///
    /// # Caching contract
    ///
    /// A cached graph is only valid for the `(absorb_idx, sample_idx)` it was
    /// built at. Reusing one at a different position silently selects the
    /// wrong permute/read path even when the state bytes are right, so any
    /// graph cache **must include the position pair in its key**.
    /// [`Self::position`] exposes it for that purpose, and
    /// [`bind_sponge_seed`] is deliberately position-agnostic (it only moves
    /// bytes) — checking the position is the cache's job, not the bind's.
    pub fn from_live_input(
        g: &mut GraphBuilder,
        device: DeviceType,
        snap: &SpongeSnapshot,
    ) -> (Self, BufId) {
        let (absorb_idx, sample_idx) = snap.position();
        assert!(
            absorb_idx < CHUNK,
            "seed absorb_idx {absorb_idx} out of range (must be < CHUNK = {CHUNK})",
        );
        assert!(
            sample_idx <= CHUNK,
            "seed sample_idx {sample_idx} out of range (must be <= CHUNK = {CHUNK})",
        );
        let state_buf = alloc_state_buf(g, device, "sponge_state_seed_input");
        g.register_input(state_buf);
        let me = Self {
            state_buf,
            absorb_idx,
            sample_idx,
            device,
            n_ops: 0,
            modules: SpongeModules::new(),
        };
        (me, state_buf)
    }

    /// Current state buffer id (mostly for tests / inspection).
    pub fn state_buf(&self) -> BufId {
        self.state_buf
    }

    /// Current transcript position `(absorb_idx, sample_idx)`.
    ///
    /// Mostly for tests / inspection: lets a caller assert that a seeded
    /// transcript picked up where the live one left off.
    pub fn position(&self) -> (usize, usize) {
        (self.absorb_idx, self.sample_idx)
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
    fn state_buf(&self) -> BufId {
        // Inherent method of the same name (`Self::state_buf`); the trait
        // method just re-exports it to generic drivers.
        DuplexSpongeGpuIR::state_buf(self)
    }

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
// Runtime seed binding.

/// Copies a [`SpongeSnapshot`]'s sponge state into the seed input slot of a
/// compiled graph, for a transcript built with
/// [`DuplexSpongeGpuIR::from_live_input`].
///
/// Call this after `GraphCompiler::compile` and before `run` / `launch_graph`.
/// It resolves `seed_buf` to its input index, checks the slot size, and does a
/// single `WIDTH * 4`-byte H2D straight into the executor's unified pool slot
/// (`GraphExe::get_input_ptr`) — no staging `DeviceBuffer`. This is the same
/// direct-to-pool path
/// `logup_zerocheck::fractional_sumcheck_gpu_irv2` uses for its leaf input.
///
/// # Reuse across proofs
///
/// `get_input_ptr` returns a pool address that is stable for the exe's
/// lifetime, so a cached graph is re-seeded by simply calling this again with
/// the next proof's snapshot — no recompile, no rebind of anything else.
/// Under `launch_graph`, re-copy on the same stream before each replay: capture
/// records addresses, not contents, so a replay reads whatever the slot holds.
///
/// # Asynchrony
///
/// The copy is enqueued on `ctx.stream` and is **not** complete when this
/// returns. `snap` must stay alive until the stream has drained past it —
/// satisfied automatically if the caller holds it until the graph's terminal
/// D2H (which synchronizes), otherwise synchronize the stream explicitly.
pub fn bind_sponge_seed(
    exe: &mut GraphExe,
    ctx: &GpuDeviceCtx,
    seed_buf: BufId,
    snap: &SpongeSnapshot,
) -> Result<(), CompileError> {
    let idx = (0..exe.num_inputs())
        .find(|&i| exe.input_buf_id(i) == seed_buf)
        .ok_or_else(|| {
            CompileError::Runtime(format!(
                "bind_sponge_seed: {seed_buf:?} is not a registered input of this graph                  (was the transcript built with `from_live_input`?)"
            ))
        })?;
    let want = SpongeSnapshot::state_size_bytes();
    let got = exe.input_size(idx);
    if got != want {
        return Err(CompileError::Runtime(format!(
            "bind_sponge_seed: seed input {idx} is {got} bytes, expected {want} \
             (WIDTH = {WIDTH} BabyBear elements)"
        )));
    }
    let dst = exe.get_input_ptr(ctx, idx)?;
    // SAFETY: `dst` is the graph pool slot for `seed_buf`, sized `want` bytes
    // (checked above). `snap.state()` is `[F; WIDTH]` — exactly `want` bytes of
    // initialized, contiguous, Montgomery-encoded memory. The copy is enqueued
    // on `ctx.stream`; see the asynchrony note above for the lifetime of `snap`.
    unsafe {
        cuda_memcpy_on::<false, true>(dst, snap.state().as_ptr() as *const c_void, want, ctx)
            .map_err(|e| CompileError::Runtime(format!("bind_sponge_seed: H2D failed: {e}")))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Buffer allocation helpers.

fn alloc_state_buf(g: &mut GraphBuilder, device: DeviceType, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst((WIDTH as i64) * 4),
        concrete_size: WIDTH * 4,
        elem_size: 4,
    })
}

fn alloc_single_f_buf(g: &mut GraphBuilder, device: DeviceType, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst(4),
        concrete_size: 4,
        elem_size: 4,
    })
}

fn alloc_ext_buf(g: &mut GraphBuilder, device: DeviceType, name: &str) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.to_string()),
        device_type: device,
        size: Quast::cst((D_EF as i64) * 4),
        concrete_size: D_EF * 4,
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
    use crypto_compiler::{graph_compiler::GraphCompiler, graph_exe::GraphExe};
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
            concrete_size: 4,
            elem_size: 4,
        })
    }

    fn add_input_ext_buf(g: &mut GraphBuilder, name: &str) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.to_string()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst((D_EF as i64) * 4),
            concrete_size: D_EF * 4,
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

    /// How the graph transcript under test gets its starting state.
    #[derive(Clone, Copy)]
    enum Seed<'a> {
        /// `DuplexSpongeGpuIR::new` — all-zero state.
        Zero,
        /// `DuplexSpongeGpuIR::from_live` — state baked in as a `Const` node.
        Const(&'a SpongeSnapshot),
        /// `DuplexSpongeGpuIR::from_live_input` — state registered as a graph
        /// input and H2D'd into the pool slot after compile, via
        /// [`bind_sponge_seed`]. This is the cacheable path.
        Input(&'a SpongeSnapshot),
    }

    impl Seed<'_> {
        fn label(&self) -> &'static str {
            match self {
                Seed::Zero => "zero",
                Seed::Const(_) => "const",
                Seed::Input(_) => "input",
            }
        }
    }

    /// Runs `ops` on a fresh `DuplexSpongeGpuIR` (compiling and executing the
    /// resulting graph on the GPU) and returns the ordered list of sampled F
    /// values. Sample_ext contributes four values in coefficient order.
    fn run_ir(ctx: &GpuDeviceCtx, ops: &[Op]) -> Vec<F> {
        run_ir_with(ctx, Seed::Zero, ops)
    }

    /// Same as [`run_ir`], but starts the graph transcript from `seed`, so the
    /// emitted graph can *continue* a live transcript instead of starting from
    /// zero. All three seeding paths must produce identical challenges.
    fn run_ir_with(ctx: &GpuDeviceCtx, seed: Seed<'_>, ops: &[Op]) -> Vec<F> {
        let mut g = GraphBuilder::new();
        // `seed_buf` is `Some` only on the `Input` path; it must be excluded
        // from the observe-input binding loop below and filled separately.
        let mut seed_buf: Option<BufId> = None;
        let mut sponge = match seed {
            Seed::Zero => DuplexSpongeGpuIR::new(&mut g, DeviceType::Cuda(0)),
            Seed::Const(snap) => DuplexSpongeGpuIR::from_live(&mut g, DeviceType::Cuda(0), snap),
            Seed::Input(snap) => {
                let (sp, buf) =
                    DuplexSpongeGpuIR::from_live_input(&mut g, DeviceType::Cuda(0), snap);
                seed_buf = Some(buf);
                sp
            }
        };

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

        let g_input_ids: Vec<BufId> = g.input_bufs().to_vec();
        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile(g)
            .expect("graph compile");

        // Bind each registered observe input to its slot in the compiled exe.
        // On the `Input` seeding path the seed is also a registered input, so
        // skip it here and fill it through `bind_sponge_seed` instead.
        for i in 0..exe.num_inputs() {
            let bid = exe.input_buf_id(i);
            if Some(bid) == seed_buf {
                continue;
            }
            let (_, bytes) = observe_bufs
                .iter()
                .find(|(b, _)| *b == bid)
                .expect("input buf not found");
            let dbuf = bytes.as_slice().to_device_on(ctx).expect("H2D");
            exe.set_input(ctx, i, &dbuf).expect("set_input");
        }

        // Fill the seed slot directly in the pool, immediately before run.
        if let (Some(buf), Seed::Input(snap)) = (seed_buf, seed) {
            assert!(
                g_input_ids.contains(&buf),
                "seed buf should have been registered as a graph input"
            );
            bind_sponge_seed(&mut exe, ctx, buf, snap).expect("bind_sponge_seed");
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
        run_host_on(&mut sponge, ops)
    }

    /// Same as [`run_host`], but drives an existing (possibly already
    /// advanced) sponge, so a caller can run a prefix, snapshot, and then
    /// continue the *same* live transcript.
    fn run_host_on(sponge: &mut DuplexSpongeGpu, ops: &[Op]) -> Vec<F> {
        let mut out = Vec::new();
        for op in ops {
            match op {
                Op::Observe(f) => FiatShamirTranscript::<SC>::observe(sponge, *f),
                Op::ObserveExt(vals) => {
                    for v in vals {
                        FiatShamirTranscript::<SC>::observe(sponge, *v);
                    }
                }
                Op::Sample => out.push(FiatShamirTranscript::<SC>::sample(sponge)),
                Op::SampleExt => {
                    for _ in 0..D_EF {
                        out.push(FiatShamirTranscript::<SC>::sample(sponge));
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

    /// Deterministic op sequence used to drive a transcript into an arbitrary
    /// mid-flight position. `n` ops are produced, mixing observes, ext
    /// observes and samples so the prefix lands on a variety of
    /// `(absorb_idx, sample_idx)` pairs.
    fn prefix_ops(n: usize) -> Vec<Op> {
        (0..n)
            .map(|i| match i % 4 {
                0 => Op::Observe(f_from_u32(1_000 + i as u32 * 37)),
                1 => Op::Observe(f_from_u32(2_000 + i as u32 * 91)),
                2 => Op::Sample,
                _ => Op::ObserveExt([
                    f_from_u32(3_000 + i as u32),
                    f_from_u32(4_000 + i as u32),
                    f_from_u32(5_000 + i as u32),
                    f_from_u32(6_000 + i as u32),
                ]),
            })
            .collect()
    }

    /// The zerocheck phase starts mid-transcript. Drive a real
    /// `DuplexSpongeGpu` through a prefix, snapshot it, seed a
    /// `DuplexSpongeGpuIR` from that snapshot, then run the *same*
    /// continuation on both and require every sampled challenge to match
    /// byte-for-byte.
    ///
    /// The prefix lengths are chosen to land the seed on many distinct
    /// `(absorb_idx, sample_idx)` pairs, including `absorb_idx != 0`
    /// (continuation's first sample must permute), `sample_idx` mid-block
    /// (reads walk down the seeded state without permuting) and
    /// `sample_idx < D_EF` (a `sample_ext` that straddles the permutation
    /// boundary).
    #[test]
    fn seeded_ir_continues_live_transcript() {
        let ctx = test_ctx();
        // Continuation exercises every op kind, including the ext paths that
        // depend on the seeded indices.
        let cont = [
            Op::Sample,
            Op::Observe(f_from_u32(11)),
            Op::SampleExt,
            Op::ObserveExt([
                f_from_u32(21),
                f_from_u32(22),
                f_from_u32(23),
                f_from_u32(24),
            ]),
            Op::Sample,
            Op::Sample,
            Op::SampleExt,
        ];

        // Teeth: the same continuation run from a *fresh* (zero-seeded)
        // transcript must NOT match, otherwise the test would pass even if
        // `from_live` silently ignored the snapshot.
        let unseeded = run_ir(&ctx, &cont);

        for prefix_len in [1usize, 2, 3, 5, 7, 9, 12] {
            let prefix = prefix_ops(prefix_len);

            // 1. Drive the live (eager) sponge through the prefix.
            let mut live = DuplexSpongeGpu::default();
            let _ = run_host_on(&mut live, &prefix);

            // 2. Export.
            let snap = live.snapshot();

            // 3. Continue on the live sponge — this is the oracle.
            let want = run_host_on(&mut live, &cont);

            // 4. Drop the live sponge before running any graph. The snapshot is owned and
            //    pointer-free, so this must not affect anything; if it ever did, the assertions
            //    below would catch it.
            drop(live);

            // 5. Both seeding paths must reproduce the oracle. `Input` is the cacheable one: its
            //    state arrives as a post-compile H2D into the graph's pool slot, not as part of the
            //    graph definition.
            for seed in [Seed::Const(&snap), Seed::Input(&snap)] {
                let got = run_ir_with(&ctx, seed, &cont);
                let path = seed.label();

                assert_eq!(
                    got.len(),
                    want.len(),
                    "prefix_len={prefix_len} path={path}: sample count mismatch"
                );
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        f_to_bytes(*g),
                        f_to_bytes(*w),
                        "prefix_len={prefix_len} path={path} (seed absorb_idx={}, \
                         sample_idx={}): challenge {i} mismatch: got {g:?}, want {w:?}",
                        snap.absorb_idx,
                        snap.sample_idx,
                    );
                }
                assert_ne!(
                    got, unseeded,
                    "prefix_len={prefix_len} path={path}: seeded continuation matched \
                     the zero-seeded one — the snapshot is not reaching the graph"
                );
            }
        }
    }

    /// A graph built by `from_live_input` must be **reusable**: compile once,
    /// then re-seed it with a *different* snapshot and re-run to get that
    /// snapshot's challenges. This is the property the whole variant exists
    /// for — with the `Const` path it is impossible, because the state is part
    /// of the graph definition.
    ///
    /// Both re-seeds share the same `(absorb_idx, sample_idx)`, which is the
    /// documented reuse contract: positions select kernel shapes at build
    /// time, so only the state bytes may vary across runs of one graph.
    #[test]
    fn seeded_input_graph_is_reusable_across_snapshots() {
        let ctx = test_ctx();
        let cont = [
            Op::Sample,
            Op::Observe(f_from_u32(77)),
            Op::SampleExt,
            Op::Sample,
        ];

        // Two live transcripts at the SAME position but with different state:
        // same op *kinds*, different observed values.
        let mut live_a = DuplexSpongeGpu::default();
        let mut live_b = DuplexSpongeGpu::default();
        for i in 0..5u32 {
            FiatShamirTranscript::<SC>::observe(&mut live_a, f_from_u32(100 + i));
            FiatShamirTranscript::<SC>::observe(&mut live_b, f_from_u32(900 + i));
        }
        let snap_a = live_a.snapshot();
        let snap_b = live_b.snapshot();
        assert_eq!(
            snap_a.position(),
            snap_b.position(),
            "setup: both snapshots must share a position to reuse one graph"
        );
        assert_ne!(
            snap_a.state_bytes(),
            snap_b.state_bytes(),
            "setup: the snapshots must differ in state, else reuse proves nothing"
        );

        let want_a = run_host_on(&mut live_a, &cont);
        let want_b = run_host_on(&mut live_b, &cont);
        assert_ne!(want_a, want_b, "setup: the two oracles must differ");

        // Build ONE graph, compile ONCE.
        let mut g = GraphBuilder::new();
        let (mut sponge, seed_buf) =
            DuplexSpongeGpuIR::from_live_input(&mut g, DeviceType::Cuda(0), &snap_a);
        let mut observe_bufs: Vec<(BufId, Vec<u8>)> = Vec::new();
        let mut sample_bufs: Vec<(BufId, usize)> = Vec::new();
        for (i, op) in cont.iter().enumerate() {
            match op {
                Op::Observe(f) => {
                    let buf = add_input_f_buf(&mut g, &format!("in_{i}"));
                    observe_bufs.push((buf, f_to_bytes(*f).to_vec()));
                    sponge.observe(&mut g, buf);
                }
                Op::Sample => sample_bufs.push((sponge.sample(&mut g), 1)),
                Op::SampleExt => sample_bufs.push((sponge.sample_ext(&mut g), D_EF)),
                Op::ObserveExt(_) => unreachable!("not used in this continuation"),
            }
        }
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

        // Bind the observe inputs once — identical for both runs.
        for i in 0..exe.num_inputs() {
            let bid = exe.input_buf_id(i);
            if bid == seed_buf {
                continue;
            }
            let (_, bytes) = observe_bufs
                .iter()
                .find(|(b, _)| *b == bid)
                .expect("input buf not found");
            let dbuf = bytes.as_slice().to_device_on(&ctx).expect("H2D");
            exe.set_input(&ctx, i, &dbuf).expect("set_input");
        }

        fn collect(exe: &GraphExe, ctx: &GpuDeviceCtx, sample_bufs: &[(BufId, usize)]) -> Vec<F> {
            let mut out = Vec::new();
            for (bid, len) in sample_bufs {
                let idx = (0..exe.num_outputs())
                    .find(|&i| exe.output_buf_id(i) == *bid)
                    .expect("sample buf not found");
                let bytes = exe.get_output(idx).to_host_on(ctx).expect("D2H");
                for k in 0..*len {
                    out.push(bytes_to_f(&bytes[4 * k..4 * (k + 1)]));
                }
            }
            out
        }

        // Run 1: seed A.
        bind_sponge_seed(&mut exe, &ctx, seed_buf, &snap_a).expect("bind A");
        exe.run(&ctx).expect("run A");
        let got_a = collect(&exe, &ctx, &sample_bufs);
        assert_eq!(got_a, want_a, "run A: seeded graph did not match oracle A");

        // Run 2: SAME compiled exe, re-seeded with B. No recompile.
        bind_sponge_seed(&mut exe, &ctx, seed_buf, &snap_b).expect("bind B");
        exe.run(&ctx).expect("run B");
        let got_b = collect(&exe, &ctx, &sample_bufs);
        assert_eq!(
            got_b, want_b,
            "run B: re-seeding the cached graph did not pick up snapshot B"
        );
        assert_ne!(
            got_a, got_b,
            "re-seed had no effect — the graph is not reading the seed input"
        );
    }

    /// `from_live_input` must register the state as a graph input of exactly
    /// `WIDTH * 4` bytes, adopt the live position, and emit no initializer
    /// (registered inputs may not be written by any graph node).
    #[test]
    fn from_live_input_registers_state_input() {
        for prefix_len in [0usize, 1, 2, 3, 5, 7, 9, 12, 16] {
            let mut live = DuplexSpongeGpu::default();
            let _ = run_host_on(&mut live, &prefix_ops(prefix_len));
            let snap = live.snapshot();

            let mut g = GraphBuilder::new();
            let (sponge, seed_buf) =
                DuplexSpongeGpuIR::from_live_input(&mut g, DeviceType::Cuda(0), &snap);

            assert_eq!(
                sponge.position(),
                snap.position(),
                "prefix_len={prefix_len}: seeded position mismatch"
            );
            assert_eq!(
                sponge.state_buf(),
                seed_buf,
                "prefix_len={prefix_len}: the returned seed buf must be the \
                 transcript's initial state buf"
            );
            assert!(
                g.input_bufs().contains(&seed_buf),
                "prefix_len={prefix_len}: seed buf was not registered as a graph input"
            );
            assert!(
                g.buf_is_interface(seed_buf),
                "prefix_len={prefix_len}: seed buf must be interface (DCE-protected)"
            );
            assert_eq!(
                g.buf_info(seed_buf).size.eval(&Default::default()),
                SpongeSnapshot::state_size_bytes() as i64,
                "prefix_len={prefix_len}: seed buf size mismatch"
            );
        }
    }

    /// Cheap structural half of [`seeded_ir_continues_live_transcript`]: the
    /// seeded builder must adopt the live sponge's transcript position, and
    /// the const bytes it bakes in must be the live overlayed state.
    ///
    /// Runs no graph, so it stays fast and pins the seeding contract even if
    /// graph compilation regresses.
    #[test]
    fn seeded_ir_adopts_live_position_and_state() {
        for prefix_len in [0usize, 1, 2, 3, 5, 7, 9, 12, 16] {
            let mut live = DuplexSpongeGpu::default();
            let _ = run_host_on(&mut live, &prefix_ops(prefix_len));
            let snap = live.snapshot();

            let mut g = GraphBuilder::new();
            let sponge = DuplexSpongeGpuIR::from_live(&mut g, DeviceType::Cuda(0), &snap);

            assert_eq!(
                sponge.position(),
                (snap.absorb_idx as usize, snap.sample_idx as usize),
                "prefix_len={prefix_len}: seeded position mismatch"
            );

            // The bytes handed to `insert_const` are the raw device encoding
            // of the overlayed state that `sync_h2d` would have uploaded.
            let bytes = snap.state_bytes();
            assert_eq!(bytes.len(), WIDTH * 4);
            let dev = live.device_state();
            for j in 0..WIDTH {
                assert_eq!(
                    bytes[4 * j..4 * (j + 1)],
                    f_to_bytes(dev.state[j]),
                    "prefix_len={prefix_len}: state slot {j} byte mismatch"
                );
            }
            assert_eq!(dev.absorb_idx, snap.absorb_idx);
            assert_eq!(dev.sample_idx, snap.sample_idx);

            // A fresh `new()` must still start from zero — seeding is additive.
            let mut g2 = GraphBuilder::new();
            let fresh = DuplexSpongeGpuIR::new(&mut g2, DeviceType::Cuda(0));
            assert_eq!(fresh.position(), (0, 0));
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

    // -----------------------------------------------------------------
    // ncu profiling — one ncu report per transcript-op module.

    #[link(name = "cudart")]
    extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
    }

    const PROFILE_OPS: &[&str] = &["observe", "observe_perm", "observe_ext_perm", "permute"];
    const PROFILE_ENV: &str = "SPONGE_PROFILE_CHILD";
    const PROFILE_TEST_PATH: &str = "sponge_graph_ir::tests::profile_transcript_ops";
    const REPORT_STEM: &str = "target/ncu_reports/sponge_transcript_ops";

    /// Emits a single `ncu --set full --import-source yes` report covering
    /// all four transcript-op kernels (`observe`, `observe_perm`,
    /// `observe_ext_perm`, `permute`) at `target/ncu_reports/sponge_transcript_ops.ncu-rep`.
    ///
    /// The test doubles as its own child: the orchestrator spawns `ncu`
    /// once on the same test binary with `SPONGE_PROFILE_CHILD=1`, and the
    /// child runs all four kernels back-to-back inside one
    /// `cudaProfilerStart/Stop` window with per-op NVTX ranges so ncu can
    /// attribute launches by name. Setup (JIT + H2D + warmup) happens
    /// outside the window. `-lineinfo` is forwarded to nvcc via
    /// `CUDA_LINEINFO=1`, and the kernel cache is bypassed so every child
    /// compiles fresh — a prior cache hit compiled without lineinfo would
    /// suppress the SASS↔source mapping ncu needs.
    #[test]
    #[ignore]
    fn profile_transcript_ops() {
        if std::env::var(PROFILE_ENV).is_ok() {
            run_profile_child();
            return;
        }

        let exe = std::env::current_exe().expect("current_exe");
        let cwd = std::env::current_dir().expect("cwd");
        let report_stem = cwd.join(REPORT_STEM);
        std::fs::create_dir_all(report_stem.parent().unwrap()).expect("create ncu_reports dir");
        eprintln!(
            "[ncu] profiling all sponge transcript ops -> {}.ncu-rep",
            report_stem.display()
        );
        let status = std::process::Command::new("ncu")
            .arg("--set")
            .arg("full")
            .arg("--import-source")
            .arg("yes")
            .arg("--profile-from-start")
            .arg("off")
            .arg("-f")
            .arg("-o")
            .arg(&report_stem)
            .arg(&exe)
            .arg("--exact")
            .arg(PROFILE_TEST_PATH)
            .arg("--ignored")
            .arg("--nocapture")
            .env("CUDA_LINEINFO", "1")
            .env(PROFILE_ENV, "1")
            .status()
            .expect("spawn ncu (is it on PATH?)");
        assert!(status.success(), "ncu failed: {status:?}");
    }

    fn run_profile_child() {
        let ctx = test_ctx();
        let modules = SpongeModules::new();
        let device = DeviceType::Cuda(0);

        // Build one single-kernel `GraphExe` per op ahead of time so JIT,
        // H2D uploads, and warmup all sit outside the profiler window.
        let mut runners: Vec<(&'static str, GraphExe)> = PROFILE_OPS
            .iter()
            .map(|op| (*op, build_op_exe(&ctx, &modules, device, op)))
            .collect();

        // Warm each op once so first-launch driver work (module loads,
        // cubin JITs, CUDA graph capture) doesn't pollute the profile.
        for (_op, exe) in &mut runners {
            exe.run(&ctx).expect("warmup run");
        }
        ctx.stream.synchronize().expect("warmup sync");

        unsafe { cudaProfilerStart() };
        for (op, exe) in &mut runners {
            nvtx::range_push!("sponge_{}", op);
            exe.run(&ctx).expect("profiled run");
            ctx.stream.synchronize().expect("profiled sync");
            nvtx::range_pop!();
        }
        unsafe { cudaProfilerStop() };
    }

    /// Builds a single-kernel graph for one transcript op, compiles it
    /// with the kernel cache disabled (so nvcc runs fresh with lineinfo),
    /// and binds zero-initialized inputs.
    fn build_op_exe(
        ctx: &GpuDeviceCtx,
        modules: &SpongeModules,
        device: DeviceType,
        op: &str,
    ) -> GraphExe {
        let mut g = GraphBuilder::new();
        let inputs: Vec<Vec<u8>> = match op {
            "observe" => {
                let state_in = alloc_state_buf(&mut g, device, "state_in");
                let value_in = add_input_f_buf(&mut g, "value_in");
                let state_out = alloc_state_buf(&mut g, device, "state_out");
                g.register_input(state_in);
                g.register_input(value_in);
                g.register_output(state_out);
                g.insert_kernel(
                    modules.observe.clone(),
                    [state_in, value_in],
                    [state_out],
                    &[("i", 0)],
                );
                vec![vec![0u8; WIDTH * 4], vec![0u8; 4]]
            }
            "observe_perm" => {
                let state_in = alloc_state_buf(&mut g, device, "state_in");
                let value_in = add_input_f_buf(&mut g, "value_in");
                let state_out = alloc_state_buf(&mut g, device, "state_out");
                g.register_input(state_in);
                g.register_input(value_in);
                g.register_output(state_out);
                g.insert_kernel(
                    modules.observe_perm.clone(),
                    [state_in, value_in],
                    [state_out],
                    &[],
                );
                vec![vec![0u8; WIDTH * 4], vec![0u8; 4]]
            }
            "observe_ext_perm" => {
                // Any p in `CHUNK - D_EF..CHUNK` selects the permuting
                // variant; pick the boundary so the pre-perm run fills
                // exactly `D_EF` slots.
                let state_in = alloc_state_buf(&mut g, device, "state_in");
                let value_in = add_input_ext_buf(&mut g, "value_in");
                let state_out = alloc_state_buf(&mut g, device, "state_out");
                g.register_input(state_in);
                g.register_input(value_in);
                g.register_output(state_out);
                g.insert_kernel(
                    modules.observe_ext_perm.clone(),
                    [state_in, value_in],
                    [state_out],
                    &[("p", (CHUNK - D_EF) as i64)],
                );
                vec![vec![0u8; WIDTH * 4], vec![0u8; D_EF * 4]]
            }
            "permute" => {
                let state_in = alloc_state_buf(&mut g, device, "state_in");
                let state_out = alloc_state_buf(&mut g, device, "state_out");
                g.register_input(state_in);
                g.register_output(state_out);
                g.insert_kernel(modules.permute.clone(), [state_in], [state_out], &[]);
                vec![vec![0u8; WIDTH * 4]]
            }
            other => panic!("unknown op {other:?}"),
        };

        // `without_kernel_cache` forces a fresh nvcc compile in this
        // process — needed because `CUDA_LINEINFO=1` doesn't change the
        // module hash, so an existing cache entry built without lineinfo
        // would be reused and ncu's source annotation would be blank.
        // The `.cu` file lives in the JIT tempdir owned by the loaded
        // `KernelProgram`, which stays alive for the profiler window.
        let mut exe = GraphCompiler::new()
            .device(device)
            .without_kernel_cache()
            .compile(g)
            .expect("graph compile");
        assert_eq!(
            exe.num_inputs(),
            inputs.len(),
            "input count mismatch for op {op}"
        );
        for (i, bytes) in inputs.iter().enumerate() {
            let dbuf = bytes.as_slice().to_device_on(ctx).expect("H2D");
            exe.set_input(ctx, i, &dbuf).expect("set_input");
        }
        exe
    }
}
