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
//! the relevant `Arc` to [`GraphBuilder::insert_kernel`]. Every graph node
//! carrying the same `Arc` shares one JIT compilation via
//! [`crypto_compiler::graph_exe::GraphCompiler`]'s Arc-keyed dedup — so, for
//! example, a transcript with 100 observes across 8 absorb positions ends up
//! with 8 compiled kernels instead of 100.

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

/// Number of `sample_ext` no-perm modules: one per starting `sample_idx` in
/// `D_EF..=CHUNK` (indices `0..N_SAMPLE_EXT_NO_PERM` map to
/// `sample_idx = D_EF..=CHUNK`).
const N_SAMPLE_EXT_NO_PERM: usize = CHUNK + 1 - D_EF;

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
/// stored as `Arc<Module>` clones so multiple `insert_kernel` calls at the
/// same transcript position all point at the same JIT compilation unit.
struct SpongeModules {
    /// One per absorb position: `observe[i]` inserts a single value at slot
    /// `i` and (when `i == CHUNK - 1`) permutes the state.
    observe: [Arc<Module>; CHUNK],
    /// One per read index in `0..CHUNK`. State is unchanged (no state output).
    sample_no_perm: [Arc<Module>; CHUNK],
    /// Permute the state and read `state'[CHUNK - 1]`. Two outputs
    /// (new_state, sample).
    sample_perm: Arc<Module>,
    /// One per starting absorb index; statically unrolls `D_EF` observes and
    /// weaves in permutations where they'd trigger.
    observe_ext: [Arc<Module>; CHUNK],
    /// One per starting `sample_idx` in `D_EF..=CHUNK` (indexed by
    /// `sample_idx - D_EF`). Reads `D_EF` slots; state unchanged.
    sample_ext_no_perm: [Arc<Module>; N_SAMPLE_EXT_NO_PERM],
    /// One per "pre-permutation reads" count in `0..D_EF`. `k` reads from
    /// pre-perm state, then one permutation, then `D_EF - k` reads from
    /// post-perm state. Two outputs (new_state, samples).
    sample_ext_perm: [Arc<Module>; D_EF],
}

impl SpongeModules {
    fn new() -> Self {
        let consts = Poseidon2Constants::p3_default();
        Self {
            observe: std::array::from_fn(|abs| {
                Arc::new(build_observe_module(abs, abs + 1 == CHUNK, &consts))
            }),
            sample_no_perm: std::array::from_fn(|read_idx| {
                Arc::new(build_sample_no_perm_module(read_idx))
            }),
            sample_perm: Arc::new(build_sample_perm_module(&consts)),
            observe_ext: std::array::from_fn(|start_abs| {
                Arc::new(build_observe_ext_module(start_abs, &consts))
            }),
            sample_ext_no_perm: std::array::from_fn(|i| {
                Arc::new(build_sample_ext_no_perm_module(i + D_EF))
            }),
            sample_ext_perm: std::array::from_fn(|pre_reads| {
                Arc::new(build_sample_ext_perm_module(pre_reads, &consts))
            }),
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

    /// Index into [`SpongeModules::sample_ext_perm`] for the current state:
    /// how many samples happen before the (unique) permutation fires.
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
        let module = self.modules.observe[self.absorb_idx].clone();
        let new_state_buf = alloc_state_buf(
            g,
            self.device,
            &format!("sponge_state_after_observe_{}", self.n_ops),
        );
        g.insert_kernel(module, [self.state_buf, value_buf], [new_state_buf]);
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
            // Emit a kernel that permutes the state and reads the top slot.
            let new_state_buf = alloc_state_buf(
                g,
                self.device,
                &format!("sponge_state_after_sample_{}", self.n_ops),
            );
            let sample_buf =
                alloc_single_f_buf(g, self.device, &format!("sponge_sample_{}", self.n_ops));
            let module = self.modules.sample_perm.clone();
            g.insert_kernel(module, [self.state_buf], [new_state_buf, sample_buf]);
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
            let module = self.modules.sample_no_perm[read_idx].clone();
            g.insert_kernel(module, [self.state_buf], [sample_buf]);
            self.sample_idx = read_idx;
            self.n_ops += 1;
            sample_buf
        }
    }

    fn observe_ext(&mut self, g: &mut GraphBuilder, value_buf: BufId) {
        let module = self.modules.observe_ext[self.absorb_idx].clone();
        let new_state_buf = alloc_state_buf(
            g,
            self.device,
            &format!("sponge_state_after_observe_ext_{}", self.n_ops),
        );
        g.insert_kernel(module, [self.state_buf, value_buf], [new_state_buf]);
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
            let module = self.modules.sample_ext_perm[self.pre_perm_reads()].clone();
            g.insert_kernel(module, [self.state_buf], [new_state_buf, ext_buf]);
            self.state_buf = new_state_buf;
        } else {
            // No perm ⇒ absorb_idx == 0 and sample_idx ∈ D_EF..=CHUNK.
            let module = self.modules.sample_ext_no_perm[self.sample_idx - D_EF].clone();
            g.insert_kernel(module, [self.state_buf], [ext_buf]);
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
/// - Non-permuting variant (`permute == false`): a single-thread compute that rewrites slot
///   `absorb_idx` and passes every other slot through — the work is a one-slot store, so
///   single-thread launch is the right shape.
/// - Permuting variant (`permute == true`): warp-parallel — 16 lanes gather the state, one lane
///   replaces slot `absorb_idx` with the observed value, [`poseidon2_permute_par`] runs the round
///   schedule with cross-lane shuffles, and lanes store their slots.
fn build_observe_module(absorb_idx: usize, permute: bool, consts: &Poseidon2Constants) -> Module {
    if permute {
        return build_observe_module_par(absorb_idx, consts);
    }
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![1]);
    let body = b.compute(1, |b, _i| {
        let mut s = load_state(b, state);
        s[absorb_idx] = kernel!(b, value[0]);
        b.pack(&s)
    });
    b.finish(format!("sponge_observe_at_{absorb_idx}"), body)
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
            let slot = absorb_idx as u32;
            let cond = kernel!(b, j == #slot);
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
/// state output.
fn build_sample_no_perm_module(read_idx: usize) -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let body = b.compute(1, |b, _i| kernel!(b, state[0, #read_idx]));
    b.finish(format!("sponge_sample_at_{read_idx}"), body)
}

/// `sample_perm(state[1, WIDTH]) -> (new_state[1, WIDTH], sample[1])`.
///
/// Permutes the state and reads the top slot. Emitted whenever
/// `absorb_idx != 0` or `sample_idx == 0`. Warp-parallel Poseidon2 via
/// [`poseidon2_permute_par`]; a single-thread reader picks the top slot
/// out of the resulting permuted-state tensor.
fn build_sample_perm_module(consts: &Poseidon2Constants) -> Module {
    let mut b = IRBuilder::new();
    let state_in = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let consts_perm = consts.clone();
    let permuted = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        let gathered = b.compute_with(
            WIDTH,
            None,
            Some(par),
            None,
            |b, j| kernel!(b, state_in[0, j]),
        );
        b.bind(gathered, move |b, tile| {
            poseidon2_permute_par(b, tile, &consts_perm, |b, v| {
                let par = b.par_map(|th, _s, _c| th.clone());
                b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, v[j]))
            })
        })
    });
    let permuted = b.let_bound(permuted);
    // The permutation sets sample_idx = CHUNK, then the read decrements to
    // CHUNK - 1.
    let read_idx = CHUNK - 1;
    let sample = b.compute(1, |b, _i| kernel!(b, permuted[0, #read_idx]));
    let out = b.tuple(&[permuted, sample]);
    b.finish("sponge_sample_perm".to_string(), out)
}

/// `observe_ext(state[1, WIDTH], value[D_EF]) -> new_state[1, WIDTH]`.
///
/// Statically unrolls `D_EF` observes; when the sequence crosses the
/// `absorb_idx == CHUNK` boundary a Poseidon2 permutation is emitted between
/// the pre- and post-perm inserts.
///
/// - Non-permuting variant (`start_abs < CHUNK - D_EF`): single-thread compute that writes `D_EF`
///   slots and passes the rest through.
/// - Permuting variant (`start_abs >= CHUNK - D_EF`): warp-parallel — one `compute_with(WIDTH,
///   par)` fuses the gather with the pre-perm inserts, [`poseidon2_permute_par`] runs the round
///   schedule, and the final `compute_with(WIDTH, par)` fuses the store with any post-perm inserts.
fn build_observe_ext_module(start_abs: usize, consts: &Poseidon2Constants) -> Module {
    let (pre_slots, post_slots) = ext_observe_slots(start_abs);
    let perm_happens = start_abs >= CHUNK - D_EF;
    if !perm_happens {
        return build_observe_ext_module_serial(start_abs, &pre_slots);
    }
    build_observe_ext_module_par(start_abs, consts, pre_slots, post_slots)
}

/// `(state_slot, value_idx)`: one absorb step in `observe_ext` writes
/// `value[value_idx]` to state slot `state_slot`.
type ExtInsert = (usize, usize);

/// Simulates the `D_EF` observes starting at `start_abs` and returns
/// `(pre_slots, post_slots)`. Writes before the (possible) permutation land in
/// `pre_slots`; writes after land in `post_slots`.
fn ext_observe_slots(start_abs: usize) -> (Vec<ExtInsert>, Vec<ExtInsert>) {
    let mut pre = Vec::new();
    let mut post = Vec::new();
    let mut in_post = false;
    let mut abs = start_abs;
    for k in 0..D_EF {
        if in_post {
            post.push((abs, k));
        } else {
            pre.push((abs, k));
        }
        abs += 1;
        if abs == CHUNK {
            in_post = true;
            abs = 0;
        }
    }
    (pre, post)
}

/// Serial `observe_ext` (no permutation crossing): every `slot` in
/// `pre_slots` receives the corresponding `value[value_idx]`.
fn build_observe_ext_module_serial(start_abs: usize, pre_slots: &[ExtInsert]) -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value = b.input("value", ScalarType::BabyBear, vec![D_EF]);
    let pre_slots = pre_slots.to_vec();
    let body = b.compute(1, move |b, _i| {
        let mut s = load_state(b, state);
        for (slot, val_idx) in pre_slots {
            s[slot] = kernel!(b, value[#val_idx]);
        }
        b.pack(&s)
    });
    b.finish(format!("sponge_observe_ext_from_{start_abs}"), body)
}

/// Warp-parallel `observe_ext` with a permutation in the middle.  The gather
/// stage fuses the pre-perm inserts; the final store fuses any post-perm
/// inserts.
fn build_observe_ext_module_par(
    start_abs: usize,
    consts: &Poseidon2Constants,
    pre_slots: Vec<ExtInsert>,
    post_slots: Vec<ExtInsert>,
) -> Module {
    let mut b = IRBuilder::new();
    let state_in = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let value_in = b.input("value", ScalarType::BabyBear, vec![D_EF]);
    let consts = consts.clone();
    let body = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        // Gather + pre-perm inserts: each lane picks its own slot value,
        // overriding to `value[val_idx]` for lanes named in `pre_slots`.
        let par = b.par_map(|th, _s, _c| th.clone());
        let pre_slots_gather = pre_slots.clone();
        let gathered = b.compute_with(WIDTH, None, Some(par), None, move |b, j| {
            let mut result = kernel!(b, state_in[0, j]);
            for &(slot, val_idx) in pre_slots_gather.iter().rev() {
                let slot_u32 = slot as u32;
                let cond = kernel!(b, j == #slot_u32);
                let val = kernel!(b, value_in[#val_idx]);
                result = b.select(cond, val, result);
            }
            result
        });
        b.bind(gathered, move |b, tile| {
            poseidon2_permute_par(b, tile, &consts, move |b, v| {
                // Store + post-perm inserts, same pattern in reverse.
                let par = b.par_map(|th, _s, _c| th.clone());
                let post_slots = post_slots.clone();
                b.compute_with(WIDTH, None, Some(par), None, move |b, j| {
                    let mut result = kernel!(b, v[j]);
                    for &(slot, val_idx) in post_slots.iter().rev() {
                        let slot_u32 = slot as u32;
                        let cond = kernel!(b, j == #slot_u32);
                        let val = kernel!(b, value_in[#val_idx]);
                        result = b.select(cond, val, result);
                    }
                    result
                })
            })
        })
    });
    b.finish(format!("sponge_observe_ext_from_{start_abs}"), body)
}

/// `sample_ext_no_perm(state[1, WIDTH]) -> samples[1, D_EF]`. State is
/// unchanged, so no state output. Called when all `D_EF` samples read from
/// the current state without triggering a permutation (`start_sam >= D_EF`
/// with `absorb_idx == 0`).
fn build_sample_ext_no_perm_module(start_sam: usize) -> Module {
    let mut b = IRBuilder::new();
    let state = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let body = b.compute(1, |b, _i| {
        let mut samples = Vec::with_capacity(D_EF);
        for k in 0..D_EF {
            let idx = start_sam - 1 - k;
            samples.push(kernel!(b, state[0, #idx]));
        }
        b.pack(&samples)
    });
    b.finish(format!("sponge_sample_ext_no_perm_from_{start_sam}"), body)
}

/// `sample_ext_perm(state[1, WIDTH]) -> (new_state[1, WIDTH], samples[1, D_EF])`.
///
/// Statically unrolls `D_EF` samples with exactly one permutation. `pre_reads`
/// counts how many samples happen from the pre-perm state before the
/// permutation fires (`0` when the very first sample permutes). The
/// permuted-state output is produced warp-parallel via
/// [`poseidon2_permute_par`]; the `samples[1, D_EF]` output is a single-thread
/// pack that reads `pre_reads` slots from the original state and
/// `D_EF - pre_reads` slots from the freshly permuted state.
fn build_sample_ext_perm_module(pre_reads: usize, consts: &Poseidon2Constants) -> Module {
    assert!(pre_reads < D_EF);
    let mut b = IRBuilder::new();
    let state_in = b.input("state", ScalarType::BabyBear, vec![1, WIDTH]);
    let consts_perm = consts.clone();
    // Warp-parallel permutation: gather → permute → store.
    let new_state = b.compute_with(1, None, None, Some(WIDTH), move |b, _outer| {
        let par = b.par_map(|th, _s, _c| th.clone());
        let gathered = b.compute_with(
            WIDTH,
            None,
            Some(par),
            None,
            |b, j| kernel!(b, state_in[0, j]),
        );
        b.bind(gathered, move |b, tile| {
            poseidon2_permute_par(b, tile, &consts_perm, |b, v| {
                let par = b.par_map(|th, _s, _c| th.clone());
                b.compute_with(WIDTH, None, Some(par), None, |b, j| kernel!(b, v[j]))
            })
        })
    });
    let new_state = b.let_bound(new_state);
    // Samples are cheap indexes; do them single-threaded.
    //
    //   samples[k]  =  state_in[0, pre_reads - 1 - k]        if k < pre_reads
    //   samples[k]  =  new_state[0, CHUNK - 1 - (k - pre_reads)] otherwise
    //
    // (This is the same read pattern as the host `DuplexSpongeGpu`: pre-perm
    // reads walk down from `sample_idx - 1` to 0, then the permutation kicks
    // `sample_idx` to CHUNK and post-perm reads walk down from CHUNK - 1.)
    let sample_out = b.compute(1, |b, _i| {
        let mut samples = Vec::with_capacity(D_EF);
        for k in 0..D_EF {
            if k < pre_reads {
                let pos = pre_reads - 1 - k;
                samples.push(kernel!(b, state_in[0, #pos]));
            } else {
                let pos = CHUNK - 1 - (k - pre_reads);
                samples.push(kernel!(b, new_state[0, #pos]));
            }
        }
        b.pack(&samples)
    });
    let out = b.tuple(&[new_state, sample_out]);
    b.finish(format!("sponge_sample_ext_perm_after_{pre_reads}"), out)
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
    use crypto_compiler::{graph_exe::GraphCompiler, runtime::CompileOptions};
    use openvm_cuda_common::{
        common::get_device,
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
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

        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");

        // Map each declared input/output BufId to its slot in the compiled
        // exe's input/output lists.
        let inputs: Vec<DeviceBuffer<u8>> = (0..exe.num_inputs())
            .map(|i| {
                let bid = exe.input_buf_id(i);
                let (_, bytes) = observe_bufs
                    .iter()
                    .find(|(b, _)| *b == bid)
                    .expect("input buf not found");
                bytes.as_slice().to_device_on(ctx).expect("H2D")
            })
            .collect();

        let mut outputs: Vec<DeviceBuffer<u8>> = (0..exe.num_outputs())
            .map(|i| {
                let size = exe.output_size(i);
                DeviceBuffer::<u8>::with_capacity_on(size, ctx)
            })
            .collect();

        let mut scratch = DeviceBuffer::<u8>::with_capacity_on(exe.scratch_bytes().max(1), ctx);
        exe.run(ctx, &inputs, &mut outputs, &mut scratch)
            .expect("run");

        // Now collect the sampled bytes in the order the caller requested.
        let mut result = Vec::new();
        for (bid, len) in sample_bufs {
            let idx = (0..exe.num_outputs())
                .find(|&i| exe.output_buf_id(i) == bid)
                .expect("sample buf not found in exe outputs");
            let bytes = outputs[idx].to_host_on(ctx).expect("D2H");
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
    /// whose Arc-keyed dedup collapses the kernel count down to at most one
    /// module per distinct transcript position.  Also asserts on the exact
    /// count for the emitted sequence so a regression that broke sharing
    /// would show up here.
    #[test]
    fn modules_are_deduplicated_across_operations() {
        // 20 observes from state (0, 0) walk absorb_idx through
        //   0,1,2,3,4,5,6,7 (perm) 0,1,2,3,4,5,6,7 (perm) 0,1,2,3
        // — hitting every `observe` module at least once (8 unique).
        //
        // Then 15 samples from state (4, 8) walk:
        //   perm (sample_perm) → sample_no_perm[6..=0] → perm → sample_no_perm[6..=1]
        // — one `sample_perm` module + `sample_no_perm[0..=6]` = 8 unique.
        //
        // Expected total unique kernel modules: 8 + 8 = 16, vs.
        // 20 + 15 = 35 emitted `Kernel` graph nodes.
        let mut g = GraphBuilder::new();
        let mut sponge = DuplexSpongeGpuIR::new(&mut g, DeviceType::Cuda(0));
        for i in 0..20 {
            let buf = add_input_f_buf(&mut g, &format!("in_{i}"));
            sponge.observe(&mut g, buf);
        }
        for _ in 0..15 {
            let _ = sponge.sample(&mut g);
        }
        let exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");
        assert_eq!(
            exe.num_unique_modules(),
            16,
            "expected exactly 16 unique kernel modules (8 observe + 8 sample) \
             for the emitted 20 observes + 15 samples"
        );
    }
}
