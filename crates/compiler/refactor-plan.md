# Refactor plan 

## Refactor 1: symbolic kernel ir shapes

Currently, all the kernels built by the graph and kernel irs have concrete shapes, as well as concrete launch dimensions. This is good for analysis but together with fusion introduces unacceptable compile times, as close to a thousand kernels are generated per graph after fusion.

Instead, we'll go for a hybrid design: the graph itself still is static shaped. That means inputs and all intermediate buffers see a constant for the shape. But the kernels themselves, as isolated under `ir::Module` accept:
1. symbolic shape input buffers 
2. symbolic constants as parameters 

During fusion, which operates on `ir::Module`s, a couple things happen. The job of the fusion pass is to fuse kernels that may be symbolic or use monomorphization to make fusion possible. 

After fusion the compilation for each kernel starts. To lower to a valid `kir` module, a couple things need to hold:
1. If there is a single level `compute` where the bound depends on a symbolic constant, you need to tile it to a two level compute.
2. If there are multiple levels of computes, you need to flatten them in canonicalization to a two level compute
3. For all two level computes, the outer bound can be a symbolic expression (not necessarily quasi-affine), but all inner ones must be constant (representing the block dim)

To make 3 happen, simplify each inner expression as much as possible first, and if it is still symbolic, use monomorphization (instantiations of symbolic parameters derived from the graph) as little as possible until all inner computes are constants.

In the lowering of `kir` to CUDA, the codegen and layout inference pipeline needs to be amended. First, generalize `Quast` with a type parameter specifying the type of constants, call this struct `Expr<T>`, where `T` substitutes the i64 in `Const`, `Mul` and `FloorDiv`.

Let `type Quast = Expr<i64>;`, which is a regular quasi-affine expression. And let `type SExpr = Expr<VarId>;` the generic expression that allows for arbitrary arithmetic.

To make this work with `kir` optimizations, specifically the layout inference pass, you need to amend this pass by allowing for the existence of `SExpr` and not just `Quast`. However, the layout inference rules are quite similar:
- if a write to a buffer, i.e. `a[f(i)] = ...` uses `Quast` or `SExpr`, put that buffer on shared memory.
- if a read to a buffer, i.e. `...a[f(i)]...` uses a `Quast` or `SExpr`, derive a shared memory mirror.

Note that `Quast` can be converted to `LinearLayout`s but `SExpr`s can't. In this formulation, we can even use data-dependent accesses for `SExpr`. To this end, add two separate arms for the enum `IndexMap` called `Blackbox(SExpr)` and `SExpr(SExpr)`. The former is an arbitrary, expression, and could be data-dependent, while the latter is a non-quasi-affine expression that only depends on loop iteration variables and symbolic constants.

### Resolved design decisions

1. **`SExpr` payload**: `Expr<VarId>` cannot represent literal coefficients (`2*i` would be unrepresentable), so `type SExpr = Expr<SymConst>` with `enum SymConst { Lit(i64), Sym(VarId) }`. `type Quast = Expr<i64>` unchanged in meaning.
2. **Symbolic-ness is author-driven.** Module authors declare symbolic constants on the builder: `let n = b.symbol("n")`. The `kernel!` macro accepts the returned handle anywhere a constant splice is allowed today: `#n`, `#(n + 2)`, `#(n * i + 4)`. No automatic shape-abstraction pass. Existing modules are NOT migrated in this refactor.
3. **Single `VarId` namespace.** Params share the `VarId` space with loop/let vars; the module's `params` registry is the source of truth for which VarIds are symbolic constants. `renumber_module` α-normalizes params so dedup works.
4. **Param unification during fusion is seam-based (structural), not value-based.** Value-based unification is correctness-safe but makes dedup unstable: accidental value collisions (e.g. an outer bound coinciding with an unrelated tile depth) unify unrelated params at one site and not another, fragmenting a single kernel family into multiple templates; worse, unifying a must-monomorphize param with an outer-bound param concretizes the outer bound and silently destroys cross-size dedup. Instead, unify only params forced equal by the fusion seam: match the producer's output shape expr against the consumer's input shape expr on the shared buffer.
5. **Scatter maps require an `inverse` annotation.** Every explicit scatter map carries an author-provided inverse (macro syntax extended). This makes non-trivial writes trivially invertible for fusion. Bijectivity/injectivity is exhaustively checked only when the map is Quast/LinearLayout (including checking the provided inverse composes to identity); for `SExpr`/`Blackbox` maps the author is trusted. A compile-time option "type-checks" all access relations of each fully-monomorphized instantiation (without compiling anything).
6. **Block size for symbolic tiling**: let `M` = max compute size over all concrete instantiations of a template (known from the static graph); block = `min(ceildiv(M, 32) * 32, 256)` (warp-rounded, capped).
7. **`Blackbox` accesses are not restricted to reads.** Data-dependent accesses are entirely the author's responsibility — no runtime guards (no clamp/trap), and `check_accesses` skips them silently. The check option validates only accesses that concretize to Quast/LinearLayout.
8. **Graph-level `BufInfo::size: Quast` stays** as-is for now; the graph remains static-shaped.
9. **Param bindings are inferred, plus an optional per-module shape hint.** At `insert_kernel`, per-node bindings are inferred by unifying the module's input shape exprs against the concrete graph buffer shapes (bare-param dims solve directly; compound dims are verified for consistency). Separately, `ir::Module::add_shape_hint(...)` attaches zero-or-one canonical concrete instantiation to the module itself; the hint supplies values for access checking and for monomorphization when no graph bindings exist (standalone `compile_and_load`). If monomorphization is required for codegen and no values are available (no bindings, no hint) → compile error. *(Refactor 2, decision 4 deletes the module shape hint; hints survive only as the `insert_kernel` inference fallback.)*
10. **Same-HIR variants live in one structure.** Kernels sharing an HIR template but differing in block size (or monomorphized values) are collected into a single template structure holding its variants — groundwork for future autotuning.

### Detailed implementation plan

Pipeline after the refactor:

```
author (symbolic Module, optional shape hint) → insert_kernel/split_module (bindings inferred per node)
  → fusion (symbolic modules, concrete analysis via bindings, seam-based param unification)
  → group by pre-tiling residual hash into template structures → per-group block-size selection
  → monomorphize (minimal, graph-derived values) + tile → variant hash = cache/dedup key
  → nvcc (parallel, cached) → GraphExe: set_param per node → plan scratch → run/capture
```

#### Phase 1 — Expression layer (`quast.rs`)

- Introduce `Expr<T>`: `Sym(VarId) | Const(T) | Add(Rc, Rc) | Mul(Rc, T) | FloorDiv(Rc, T) | Neg(Rc)`; `type Quast = Expr<i64>`; `enum SymConst { Lit(i64), Sym(VarId) }`; `type SExpr = Expr<SymConst>`.
- Generic over `T` (mechanical): `substitute`, `syms`, `linearize`/`delinearize`, structural hash, dump printing.
- Stays `Expr<i64>`-only: `to_linear_layout`, full `LinComb` normalization + `fold_rems`, `range`, scatter bind/validate. (Precision lesson stands: range analysis must run on the rem-folded normal form; do not attempt symbolic range analysis.)
- New on `SExpr`: best-effort `simplify` (fold literal subterms only — `SymConst` is not closed under arithmetic, so no like-term combining across symbolic coefficients), `try_concretize(env: &BTreeMap<VarId, i64>) -> Option<Quast>` (the monomorphization primitive), `eval`, `param_syms()` (syms in `SymConst` position, distinct from loop-var `Sym`s). *(DONE — landed as `fold_lits`, `concretize`, `try_to_quast`, `try_concretize`, `eval`, `param_syms`, plus `From<&Quast>/From<Quast> for SExpr`.)*
- `CStrEmitter` generalized: `SymConst::Sym` renders as the kernel-parameter identifier (`p_n`); coefficient u32-overflow checks apply to `Lit` only. *(Deferred to Phase 6/7, where SExpr emission is first consumed — the parameter identifier format is dictated by the codegen param ABI.)*
- Known representational limit: `Mul(expr, T)` cannot express a product of two compound exprs (`(n+1)*(m+2)`). Acceptable initially; flattening two compound-symbolic-bound nests errors out with a "monomorphize first" diagnostic.

#### Phase 2 — HIR: module params + symbolic shapes (`ir.rs`, `type_infer`, `canonicalize`, `split_module`, `module_hash`, `dump`)

- `ir::Module`/`IRBuilder` gain a param registry: `params: Vec<VarId>` + names. `IRBuilder::symbol(name) -> SymExpr` registers a param and returns a host-side handle.
- `IRBuilder::add_shape_hint(values: &[i64])` (parallel to `params`, zero or one per module): a canonical concrete instantiation stored on `ir::Module`. Consumed by `check_accesses` and by the monomorphizer when compiling without graph bindings. Not hashed (it's metadata, not semantics).
- `SymExpr` handle: wraps an `SExpr`, implements `Add/Sub/Mul/Div<i64>` and `Add/Mul<SymExpr>` (within the `Mul` limits above) so `#(n + 2)` etc. work as plain Rust expressions.
- `type SizeExpr = SExpr`. `Shape = Vec<SizeExpr>` (ir.rs:45), `Compute.bound`/`Reduce.bound: SizeExpr` (ir.rs:129-143). Keep an `is_const() -> Option<usize>` fast path so fully-concrete modules behave exactly as today.
- `type_infer`: `lift_compute_type` prepends symbolic bounds; shape equality is structural expr equality (post-simplify).
- `canonicalize`: `flatten_nests` (canonicalize.rs:667) multiplies bounds — symbolic outer × const inner works (`Mul(n, Lit(M))`), index recovery stays quasi-affine because divisors are const. Two-level absorption (canonicalize.rs:448) only when involved bounds are const (unchanged behavior otherwise).
- `module_hash` (module_hash.rs:27-53): hash param declarations (count + order, α-normalized ids, NOT names or bindings) and shape/bound exprs structurally. `renumber_module` (fusion.rs:1864) renumbers param VarIds too. *(Refactor 2, decision 10 reverses the "NOT names" part: param names become hashed interface.)*
- `split_module`: preserves param VarIds verbatim like all others; each split inherits the subset of params its subtree references.
- `dump.rs`: print `params: [n, m]` in module headers; exprs in shapes/bounds.

#### Phase 3 — Proc macro (`macros/src/lib.rs`)

- Splice codegen changes from `__cc_b.const_u32(#v)` (line 674) / i64 coercions to a trait call: `impl IntoDslConst` (or similar) implemented for integer types AND `SymExpr`. `#n` and `#(expr)` then work unchanged for concrete values and accept symbolic handles/expressions.
- Splice positions that accept symbols: compute/reduce bounds, scatter map operands and physical `[bounds]`, DSL scalar constants where a u32 const is built (a symbolic constant used as a *value* becomes a kernel scalar param read, not a literal).
- Splice positions that must stay concrete: `#[grid(threads = N)]` (block-size hint) and `#[par(...)]` map coefficients (must convert to `LinearLayout`) — reject symbolic handles there with a clear error.
- `#[scatter]` grammar gains the inverse clause: `#[scatter(params -> exprs, inverse = params -> exprs, [bounds])]`. Absent scatter attr = identity = no inverse needed. Existing explicit-scatter call sites (the NTT implementations in kernels.rs) are **commented out for now** rather than migrated — along with their tests/benches/examples (`ntt_scale_graph`, NTT goldens) — and revived when modules are migrated to symbolic form.

#### Phase 4 — Scatter inverse (`quast.rs` Scatter, `canonicalize`, fusion)

- `Scatter` struct gains `inverse: <same repr as forward map>`.
- Validation split: if the forward map concretizes to Quast → keep exhaustive bijectivity check AND verify `inverse ∘ map == identity` on the domain. If SExpr → trust the author (checked only under the type-check option).
- Fusion uses `inverse` to read through scattered producer outputs: consumer reads physical `g(i)` → grafted producer body indexed at `inverse(g(i))`.

#### Phase 5 — Monomorphization + tiling (new `passes/monomorphize.rs`, changes in `lower_to_kir`)

- Input: symbolic module + the set of concrete binding vectors from all graph nodes sharing the template (graph-derived; the module's shape hint substitutes when compiling standalone with no graph). Output: residual module (+ per-node residual bindings). If a must-be-concrete expr is symbolic and no values are available → `CompileError`.
- Must-be-concrete set (superset of rule 3 above): inner compute bounds / block dims, inner-let tile shapes (`__shared__` sizes), `#[par]` layout maps, anything feeding `to_linear_layout` or register promotion, scatter maps+bounds when exhaustive validation is expected (concrete-authored scatters).
- Algorithm: simplify each must-be-concrete expr; while still symbolic, substitute the param whose removal concretizes the most remaining exprs (greedy, ties by param index), re-simplify. Params only appearing in outer bounds / outer index exprs survive.
- Tiling (rule 1): in `lower_to_kir` (lower_to_kir.rs:141-155), a flat kernel with symbolic bound `n` gets constant `block = B` (per-group policy below), `grid = ceildiv(n, B)` as `SExpr`, grid-spanning par with guard `i < n` (n = runtime param). Concrete path unchanged (`block = n.min(256)`).
- Block-size policy: per template group, `M = max` over instantiations of the compute size; `B = min(ceildiv(M, 32) * 32, 256)`. Because `B` (and any monomorphized values) is baked into the tiled structure, the final cache key is the hash of the *post*-monomorphization+tiling variant; grouping happens on the *pre*-tiling hash.
- Template structure: `KernelTemplate { hir_hash, variants: Vec<Variant> }` where each `Variant` records `(block, monomorphized values) -> compiled module`, and each graph node points at (template, variant). One structure per HIR family — the future autotuning surface (multiple `B` candidates per template, pick per instantiation).

#### Phase 6 — KIR + codegen + runtime ABI (`kernel_ir.rs`, `codegen.rs`, `runtime.rs`)

- `Grid.bound: usize` → `enum KBound { Const(usize), Expr(SExpr) }` (kernel_ir.rs:414). `Par`/`Loop` bounds and `ParAttr.seq_size` remain `usize` — guaranteed by Phase 5. Grid-spanning guard bounds may be `KBound::Expr`.
- `KernelProgram` gains `params: Vec<(VarId, String)>`; device kernels take trailing `const uint32_t p_n` args; guards (codegen.rs:530-587) emit param expressions when symbolic.
- Host side (codegen.rs:1139-1216): `Prog` gains `int64_t params[P]`; new `extern "C" void set_param(Prog*, uint64_t i, int64_t v)`; `run()` computes `dim3(<expr over p->params>)` per launch; `input_size`/`output_size`/`scratch_size` become expressions over `p->params` instead of static consts. `__launch_bounds__` stays literal (block always concrete).
- `runtime.rs`: `VTable` + `KernelModule::set_param(i, v)`. Metadata queries (`input_size` etc.) are only valid after params are set — enforce with a `params_bound` flag like `scratch_bound`. *(Refactor 2 renames `KernelModule` → `KernelProgram` and `set_param` → `set_symbol`.)*
- `plan_shared_mem`, `insert_sync`, register promotion, `ConvertLayout`/shuffle emission: untouched (all inner-level, concrete by construction).
- Scratch: `parallel_reduce_rewrite` partials sized `ceildiv(n, B)` make module scratch param-dependent → `plan_global_scratch` offsets/total become `SExpr`, emitted as expressions in `buf_arg`/`scratch_size`.

#### Phase 7 — `layout_infer` + `IndexMap` arms

- `IndexMap` gains `SExpr(SExpr)` (loop vars + params only) and `Blackbox(SExpr)` (may reference loaded SSA values via the `VarId(i) <-> SSARes(i)` convention).
- Rules: an `SExpr`/`Blackbox` **write** pins the buffer to shared memory and poisons register promotion; an `SExpr`/`Blackbox` **read** derives a shared-memory mirror (reuse `Plan::Mirror`, layout_infer.rs:258). Quast paths (`linearize_accesses`, `promote_tiles`) unchanged — they only fire on concrete pow2 domains.
- `lower_to_kir` rejection sites (`passes/utils.rs:48-76`) become classification: non-quasi-affine over loop-vars+params → `SExpr`; references loaded data → `Blackbox`; no more hard errors for these.
- `Blackbox` codegen ordering: the index-producing load must be emitted before the dependent access — Par read emission becomes two-phase (independent loads, then data-dependent ones). Global data-dependent stores emit directly; races are the author's responsibility.
- No exhaustive injectivity check for `SExpr`/`Blackbox` writes (only Quast/LinearLayout writes keep it).

#### Phase 8 — Fusion + graph integration (`fusion.rs`, `graph_ir.rs`, `graph_exe.rs`)

- `KernelModuleNode` gains `param_bindings: Vec<i64>` (parallel to `module.params`). `insert_kernel` **infers** bindings by unifying the module's input shape exprs against the node's concrete graph buffer shapes: a bare-param dim solves the param directly; compound dims are checked for consistency against already-solved params; unsolvable/inconsistent → error at insertion. `split_module` threads the referenced subset through to each split node.
- Fusion analysis stays concrete: `extract_relations` (fusion.rs:68) evaluates bounds/costs through the node's bindings, so scoring/legality logic is unchanged. The fused artifact stays symbolic.
- `apply_fusion`: fused module params = union of both sides' params, then **seam-based unification**: match producer output shape expr vs consumer input shape expr on the shared buffer; when one side is a bare param, substitute it with the other side's expr; compound-vs-compound stays un-unified (bindings satisfy the relation regardless). Merged bindings recorded on the fused node. `dedup_modules` hashes the renumbered symbolic module.
- `GraphCompiler::compile` (graph_exe.rs:240-425): phase 1b groups by pre-tiling residual hash into `KernelTemplate`s → per-group block-size selection + monomorphization produce variants → variant hash is the KernelCache key (kernel_cache.rs:102) → parallel nvcc on variant misses. Per-node instantiation: `set_param` from bindings **before** querying `scratch_size` and running the planner. `GraphExe::run` sets params during node replay (before `set_input`); CUDA-graph capture is safe since the graph is static and params never change post-compile.

#### Phase 9 — Access type-check option

- `CompileOptions::check_accesses: bool` (env-driven like `dump_ir`). For each concrete instantiation (per-node inferred bindings in a graph; the module's shape hint standalone): fully monomorphize ALL params, lower to KIR, and run exhaustive checks on every access that concretizes to Quast/LinearLayout — scatter bijectivity + inverse correctness, write injectivity, bounds-in-range. No nvcc, no codegen. `Blackbox` (data-dependent) accesses are skipped silently — correctness is the author's responsibility, no runtime guards. *(DONE — `passes/check_accesses.rs`: `check_program_accesses` walks each par's reads/writes (bounds for both; write injectivity across the par + grid dims for fixed loop vars, so sequential overwrites stay legal); domains over `EXHAUSTIVE_LIMIT` fall back to the interval check. Drivers: `check_module_accesses(module, bindings)` from `GraphCompiler::compile` phase 1a per unique (module, bindings); `check_accesses_from_hint` from `compile_and_load`; env `CRYPTO_COMPILER_CHECK_ACCESSES`. Enumeration models KIR precisely: only symbols in the expr count (access `bounds` maps are the whole kernel env), the grid sym is derived (`par / blockDim`) for `spans_grid` pars, and shared/register buffers are per-block so grid is sequential for them. To keep large-domain NTT/bit-reversal reads decidable, `Quast::range` gained a mixed-radix digit decomposition: `floor(x/c)` terms whose divisors chain are exact digit-wise intervals. The checker also exposed a real codegen bug: par reads were all emitted at the top of the par body, so select-branch loads ran speculatively, violating `SSAOpCode::Select`'s no-speculation guarantee. Fixed by `codegen::compute_read_sinks`: a read whose value is used only inside one select branch is sunk to that branch's entry (unless its index depends on par-body values or feeds a gather index); the checker exempts exactly the sunk reads.)*

#### Testing

- First (per workflow): a `gpu_macro.rs` test — one symbolic module (`b.symbol("n")`) instantiated at 2+ sizes in one graph; assert a single unique module hash, one nvcc compile, correct results at all sizes.
- CPU-only unit tests per pass: `Expr` simplify/`try_concretize`; monomorphize minimality (param only in outer bound survives; param in inner bound doesn't); binding inference (bare dim, compound-dim consistency check, unsolvable error); seam unification (bare-param, expr, compound-compound cases); layout rules route `SExpr` writes/reads to shared/mirror; scatter inverse validation catches a wrong inverse on a Quast map; error path for monomorphization-required-but-no-hint (standalone).
- Macro UI tests: `#n`, `#(n+2)`, symbolic in bounds/scatter; rejection in `threads=`/`par`; scatter-without-inverse error.
- NTT implementations (kernels.rs) plus their tests/benches/examples are commented out for the duration; the remaining concrete-module suite must stay green throughout (all changes keep an `is_const` fast path).
- E2E: fractional-GKR graph as the compile-time benchmark once its modules are migrated (later); until then, a synthetic multi-size graph measuring unique-module count and cold-compile wall time.

### Resolutions log (2026-08-03)

- Block size: `min(ceildiv(M, 32) * 32, 256)` (warp-rounded threads, capped).
- Bindings: inferred at `insert_kernel` from buffer shapes; `add_shape_hint` (0-or-1 per module) covers checking + standalone monomorphization; error if monomorphization required with no values.
- Blackbox: author's responsibility — no guards, `check_accesses` skips them.
- NTT scatter sites: commented out, not migrated.
- Symbolic scalars as values: become u32 kernel params; `cf` field constants stay concrete-only.
- Same-HIR variants (block size / monomorphized values) collected under one `KernelTemplate` structure → future autotuning. *(Refactor 2, decision 15 defers this: the outer template layer collapses to a flat per-residual-hash list; autotuning reintroduces it when it lands.)*


## Refactor 2: cleanup fusion & graph ir

### Problems with the current structure

1. **Dedup happens at insertion time and is re-derived everywhere else.** `insert_kernel` (graph_ir.rs:459) eagerly splits (`split_module`, which internally type-infers + canonicalizes), hashes, and dedups onto canonical `Arc`s via `module_dedup` (graph_ir.rs:352) and `subgraph_cache` (graph_ir.rs:357). Fusion then maintains its *own* dedup: `extract_one` re-runs `type_infer` + `canonicalize` per unique `(module_hash, bindings)` every round (fusion.rs:200-210), `apply_fusion` re-canonicalizes both sides again (fusion.rs:1194-1195), and a post-fusion `dedup_modules` sweep re-hashes everything (fusion.rs:2052). `GraphCompiler::compile` runs a *third* hash-dedup backstop (graph_exe.rs:404-427).
2. **The per-module JIT pipeline re-does graph-level work.** `compile_and_load` (lib.rs:87) runs monomorphize → type_infer → `rewrite_parallel_reduce` → type_infer → canonicalize → `plan_global_scratch` → lower_to_kir per kernel, even though the graph compiler already monomorphized (graph_exe.rs:341) and insertion already canonicalized. Worse, fusion must *predict* what `rewrite_parallel_reduce` will do at JIT time (`has_tree_lowered_reduce`, fusion.rs:334) to score candidates — coupling fusion to the internals of a pass that hasn't run yet.
3. **Multi-kernel modules are half-supported.** `plan_global_scratch` + the scratch ABI (`scratch_size`/`set_scratch_buf`, codegen.rs:1520-1537; runtime.rs:550-591) exist to place intermediates between a module's internal kernels, but graphs hard-reject any module with scratch (graph_exe.rs:642-649) — so a two-stage parallel reduce cannot be used in a graph at all. Intermediates between kernels are a *graph* memory-planning concern; the module backend shouldn't own an allocator.

### Target separation of concerns

- **`ir::Module` + graph IR**: graph-level concerns — typecheck, reduce lowering, monomorphization, canonicalization, splitting, fusion, DCE, dedup, memory planning.
- **`kir` backend (`ModuleCompiler`)**: per-kernel, per-CTA concerns — lowering a *canonical, monomorphized, single-kernel* module to KIR, layout inference, sync insertion, shared-memory planning, codegen, nvcc.

```
insert_kernel (dumb: arity + binding inference, Arc<ir::Module>)
  → GraphCompiler passes (any order; each pass auto-runs its prerequisites):
      typecheck / lower_reduce / monomorphize / canonicalize (+split) / fuse / dce / plan_memory
      (each pass: kernel_dedup, compute per unique module, fan out to aliases)
  → per unique node.hash (parallel, cached):
      ModuleCompiler::lower(ir::Module)   -> KirProgram     (optimized KIR)
      ModuleCompiler::codegen(KirProgram) -> KernelProgram  (dlopen'd artifact)
  → GraphExe { kernels: Vec<KernelProgram>, plan, pool }
      run(): node.param_bindings → KernelProgram::set_symbol → launch
```

**Graph passes are self-normalizing.** A pass never assumes an ordering was respected: it *calls the passes it depends on in its own body* (e.g. `fuse` calls `canonicalize`, which calls `typecheck`), and prerequisites are cheap no-ops when already satisfied because derived state is memoized on the node / graph and invalidated exactly when the underlying module or node set changes:

| pass          | auto-runs first                            | derives                                          | invalidates                                                 |
| ------------- | ------------------------------------------ | ------------------------------------------------ | ----------------------------------------------------------- |
| `typecheck`   | `kernel_dedup`                             | `node.types` (once per unique hash, fanned out)  | —                                                           |
| `lower_reduce`| `typecheck`                                | rewritten modules (+ their fresh `types`)        | `hash`/`canonical` of rewritten nodes                       |
| `monomorphize`| `kernel_dedup`                             | residual `module`, block hints                   | `types`/`hash`/`canonical` of changed nodes                 |
| `canonicalize`| `typecheck`                                | `node.canonical`, refreshed `types`; node splits | `hash` of rewritten nodes; `g.plan` when a split adds nodes |
| `fuse`        | `lower_reduce`, `monomorphize`, `canonicalize` | fused nodes (canonical + typed, `hash = None`) | `g.plan`                                                    |
| `dce`         | —                                          | —                                                | `g.plan` (when anything was removed)                        |
| `plan_memory` | —                                          | `g.plan: Option<MemoryPlan>`                     | —                                                           |

`kernel_dedup` is internal plumbing, not a user-visible pass: invoked at pass entry, it fills missing `node.hash`es and collapses structurally identical modules onto canonical `Arc`s so per-module work runs once per hash. Any structural graph mutation (`insert_*`, splits, fusion, dce) clears `g.plan`.

### Types and lifecycle

| Type                                                    | Role                                                                                                                                                                                                                                                        | Lifecycle                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`ir::Module`** (HIR)                                  | Author-written kernel; may be symbolic (`IRBuilder::symbol`) and single- *or* multi-kernel until canonicalize.                                                                                                                                              | Built via `kernel!` macro, wrapped in `Arc`, pushed by `insert_kernel`. Graph passes swap it in place via `KernelModuleNode::replace_module` (never mutate in place). Consumed by `ModuleCompiler::lower`.                                                                                                 |
| **`KirProgram`** (KIR container; ex-`KernelProgram`)    | Optimized single-kernel KIR — post `lower_to_kir` + `layout_infer` + `insert_sync`.                                                                                                                                                                         | Output of `ModuleCompiler::lower`; consumed immediately by `ModuleCompiler::codegen`. Transient, not persisted.                                                                                                                                                                                            |
| **`KernelProgram`** (dlopen'd artifact; ex-`KernelModule`) | CUDA `.so` loaded via dlopen; exposes `set_symbol(i, v)` / `set_input` / `set_output` / `input_size(i)` / `output_size(i)` / `run`.                                                                                                                        | Output of `ModuleCompiler::codegen`; disk-cached by `KernelCache` across compiles; owned at runtime by `GraphExe.kernels` — one entry per unique residual hash. Serves any binding whose residual hash matches (grid guard covers all `n`; block is a perf hint).                                          |
| **`KernelModuleNode`** (graph node)                     | Wraps `Arc<ir::Module>` with per-node `param_bindings: BTreeMap<String, i64>`, `inputs`/`outputs`, and derived state (`types`, `hash`, `canonical`, `fusion_history`). No `baked` (decision 14), no `template_hash` (decision 15).                          | Created by `insert_kernel`; mutated by graph passes exclusively through `replace_module` (which nulls derived state); consumed at compile-time to build each `ExeNode` (kernel index + bindings + buffer wiring).                                                                                          |
| **`GraphBuilder`**                                      | Mutable device-level graph: `bufs`, `nodes` (Kernel / BlackboxKernel / Const / Memcpy / Memset), registered inputs/outputs, and `plan: Option<MemoryPlan>`.                                                                                                 | Built by author code; passed by value to `GraphCompiler::compile`, which mutates it via passes and consumes it to build `GraphExe`. Every structural mutation (`insert_*`, splits, fusion, dce-removal) resets `plan` to `None`.                                                                           |
| **`GraphCompiler`**                                     | Pass driver + backend + planner config: `module_compiler: ModuleCompiler`, `fusion: Option<FusionOptions>`, `cache`, `device`, `plan_env`.                                                                                                                  | Constructed by caller. Exposes public passes as methods over `&mut GraphBuilder` (`typecheck` / `canonicalize` / `lower_reduce` / `monomorphize` / `fuse` / `dce` / `plan_memory`) plus `compile(g) -> GraphExe`.                                                                                          |
| **`ModuleCompiler`**                                    | Pure per-kernel backend: nvcc / arch / dump / verbosity / `check_accesses` config + `lower(ir::Module) -> KirProgram` + `codegen(KirProgram) -> KernelProgram`. No monomorphizer, no hints (decisions 1, 4).                                                | Held by `GraphCompiler.module_compiler`; also usable standalone but only via a one-node graph — no direct API for that (`compile_and_load` is deleted).                                                                                                                                                     |
| **`GraphExe`**                                          | Runtime executable: `kernels: Vec<KernelProgram>` + `nodes: Vec<ExeNode>` + `plan: MemoryPlan` + `pool: DevicePool` + input/output slots.                                                                                                                   | Produced by `GraphCompiler::compile`. Stateful: `set_input(ctx, i, &DeviceBuffer<u8>)` D2D-copies into a pool slot, `run(&ctx)` replays the node list, `get_output(i)` views the output slot. `kernel_program(node) -> &mut KernelProgram` returns a direct artifact handle (Phase 7).                     |
| **`MemoryPlan`**                                        | Planner output: `{ order, offsets, peak_bytes }`.                                                                                                                                                                                                           | Filled on `GraphBuilder.plan` by `plan_memory`; reused across passes; moved into `GraphExe.plan` at compile assembly.                                                                                                                                                                                      |
| **`GraphMono`** *(pub(crate))*                          | Internal data structure of the graph monomorphize pass: `{ residual, residual_bindings, baked (diagnostic), max_outer }`.                                                                                                                                    | Produced by `monomorphize_for_graph`; consumed by `GraphCompiler::monomorphize` and `apply_fusion`'s re-monomorphize tail. `baked` is diagnostic only — never stored on any node (decision 14).                                                                                                             |

**Deleted in this refactor:**

- **`ModuleRunner`** (`runner.rs`) — a small `test-utils` helper over `GraphBuilder`/`GraphCompiler`/`GraphExe` replaces it; multi-size test runs drive `GraphExe::kernel_program(node)` directly (decision 3, Phase 7).
- **`CompileOptions`**, **`compile_and_load`**, **`compile_and_load_with_hint`** — subsumed by `ModuleCompiler`; standalone runs use a one-node graph (decisions 2, 4).
- **`KernelTemplate` / `KernelVariant`** (graph_exe.rs) — collapsed to a flat `Vec<KernelProgram>` keyed by residual hash; autotuning reintroduces the outer layer when it lands (decision 15).
- **`GlobalScratchPlan`** (`plan_global_scratch.rs`) — module-scratch model eliminated; the graph planner places what it used to (Phase 7).
- **`SubgraphCacheKey`** + `module_dedup` + `subgraph_cache` (graph_ir.rs) — replaced by internal `kernel_dedup` at pass entry (Phase 2).
- **`monomorphize_from_hint`** (monomorphize.rs) — no standalone-hint path; the graph monomorphize pass is the sole monomorphization entry point (decision 4).

### Resolved design decisions

1. **`ModuleCompiler` is a pure per-kernel backend: config + `lower` + `codegen`.** Naming first: the KIR container `KernelProgram` (kernel_ir.rs:574) is renamed **`KirProgram`**, and the dlopen'd artifact `KernelModule` (runtime.rs) is renamed **`KernelProgram`** — all Refactor 2 text uses the new names. `lower(&self, ir::Module) -> KirProgram` = precondition checks (monomorphized inner bounds, canonical, single-kernel) + `lower_to_kir` + `layout_infer` + `insert_sync`; the returned `KirProgram` is fully optimized. `codegen(&self, KirProgram) -> KernelProgram` = source emission (`plan_shared_mem` is already internal to codegen, codegen.rs:327) + `verify` + nvcc + dlopen. No monomorphizer and no hints on the compiler — monomorphization is exclusively a graph pass (decision 4).
2. **`CompileOptions` and `compile_and_load`/`compile_and_load_with_hint` are replaced outright** by `ModuleCompiler`. All callers (graph_exe, kernel_cache, tests, benches; `runner.rs` is deleted outright — decision 3) migrate in this refactor. Env-var defaults (`NVCC`, `CRYPTO_COMPILER_CUDA_ARCH`, `CRYPTO_COMPILER_DUMP_IR`, `CRYPTO_COMPILER_VERBOSITY`, `CRYPTO_COMPILER_CHECK_ACCESSES`) move to `ModuleCompiler::default`.
3. **`ModuleRunner` is deleted.** It predates the graph API — a convenience wrapper for driving one module in tests/benches — and everything it did is subsumed: compilation is a one-node `GraphBuilder` + `GraphCompiler::compile`, execution is `GraphExe`, and symbol rebinding is `KernelProgram::set_symbol` on the already-open artifact (the landed `set_param` slot, renamed — Phase 7), so a rebind never re-dlopens. Staging/bench conveniences (input upload, output readback, bench loops) become a small `test-utils` helper over `GraphExe` + direct artifact handles (Phase 7). To avoid feature-gating gpu.rs and the benches, promote `planner` (heuristic backend — no external link deps) to a default feature of the crate; `planner-ortools` stays optional.
4. **Shape hints exist only as the `insert_kernel` inference fallback.** `IRBuilder::add_shape_hint`/`shape_hint`/`extend_shape_hint` (ir.rs:525-548) are deleted, and `ModuleCompiler` carries no hints. The one surviving hint surface is the `insert_kernel` argument (`&[("n", 4096)]`, by name): a pure binding-inference fallback, consumed at insertion, never attached to the module. All concrete values live in per-node `param_bindings`; `monomorphize_from_hint` (monomorphize.rs:332) is deleted along with the standalone `compile_and_load` path — a standalone module run is a one-node graph.
5. **`lower_reduce` is binding-aware.** `should_tree_lower` needs concrete `M`/`K`; before monomorphization those may be symbolic, so the graph pass evaluates bounds under each node's binding env (like fusion's `binding_env`, fusion.rs:163) and memoizes per `(module hash, relevant binding values)`. Same-HIR nodes with different `K` bindings intentionally diverge (the rewrite concretizes the bounds it restructures). This is why lower_reduce precedes monomorphize: the chains it emits introduce new must-be-concrete inner bounds that monomorphize then handles uniformly.
6. **"Is canonicalized" = the canonicalize walk fires no rewrite.** Thread a `changed` flag through `CanonCx`; `extract_program(&Module, TypeMap) -> Result<Program>` errors if `changed` (used by `lower` as the strictness check), while the rewriting `canonicalize` returns the canonical `Module` for the graph pass. `is_canonicalized` (canonicalize.rs:779) folds into this.
7. **Fusion re-normalizes its own output.** Grafting denormalizes, and monomorphize-before-fusion means a fused kernel can expose *new* must-be-concrete positions (a producer's outer compute becoming an inner tile). `apply_fusion` therefore ends with: type_infer → canonicalize → single-kernel check (exists, fusion.rs:1420-1425) → re-monomorphize against the merged bindings → block-hint fixup. Node hash is left `None`; the next `kernel_dedup` recomputes it.
8. **Monomorphize-before-fusion is safe for dedup** (inversion of Refactor 1's diagram): residuals keep symbolic outer bounds, so cross-size template sharing survives; only must-concretize params fragment, exactly as they would post-fusion. Seam-based param unification (Refactor 1, decision 4) now operates on residuals.
9. **Bindings are keyed by param name: `BTreeMap<String, i64>`.** Replaces the positional `Vec<i64>` in `node.param_bindings`, in `GraphMono::{residual_bindings, baked}`, and in the `insert_kernel` hint argument. Prerequisite: `IRBuilder::symbol` rejects duplicate param names within one module (they'd collide as map keys — and already collide as `p_<name>` kernel arguments). The alignment invariant becomes: binding keys = the module's param-name set; `prune_unused_params` (invoked from `kernel_dedup`) removes a dead param and its binding entry together, so α-normalized hashes don't fragment on params stranded by lower_reduce/monomorphize.
10. **Param names are part of the hash; fusion names inherit from the producer.** `module_hash` gains param *names* (count + order + name per declaration; VarIds stay α-normalized). Rationale: the hash already covers the module name (module_hash.rs:29) and input declaration names (:50), and name-keyed bindings promote param names to interface — so hash-equal ⇒ identical param registries, and `kernel_dedup` needs **no binding-key remapping**. Name determinism holds by construction: same builder fn ⇒ same names, and fusion's seam unification is directional — a bare-vs-bare seam keeps the *producer's* param (deterministic tie-break); bare-vs-compound is forced by structure (the bare side is substituted by the compound expr, whichever side it's on). Non-unified consumer params whose names collide with a surviving producer param are renamed deterministically (`{name}__c`) before the merge, keeping the name-keyed binding union well-defined. So structurally equal fusion products carry equal names and dedup as before.
11. **Split kernels project bindings by name.** `split_module` preserves param VarIds *and names* (Refactor 1, Phase 2), so a child's bindings are the parent's map filtered to the child's inherited param names. `infer_param_bindings` (graph_ir.rs:660) remains only at the user-facing `insert_kernel` boundary.
12. **`dce` and `plan_memory` are public graph passes; the plan lives on the graph.** `dce` (today invoked inside fusion rounds, fusion.rs:1879) is exposed as a pass in its own right. `plan_memory` runs the planner and fills `GraphBuilder::plan: Option<MemoryPlan>`; every structural mutation (insert, split, fuse, dce-removal) resets it to `None`, and `compile` reuses `g.plan` when already present. With `planner` promoted to a default feature (decision 3), `MemoryPlan` is unconditionally available.
13. **`fuse` is a public pass; `fuse_only` is deleted.** Because passes self-normalize, calling `fuse` on a freshly built graph runs the whole prerequisite chain itself — `fuse_only` (graph_exe.rs:206) has no remaining purpose. Its callers (visualizer, fusion-inspection tests) call `fuse` and read the graph.
14. **No stored `baked` metadata.** The monomorphize pass's entire effect is `node.module` ← residual: baked values are embodied in the module structure, and `node.hash` (residual hash) already distinguishes instantiations. `KernelModuleNode` carries no `baked` field; `GraphMono::baked` survives only as a transient diagnostic return (dumps, pass unit tests). Combined with decision 15 (no `KernelVariant` at all — graph_exe.rs:1044-1066 goes away), `variant.baked` disappears with its surrounding struct. Tests asserting on `variant.baked` (gpu_graph.rs:619, 724, 742) re-target `node.module`'s param registry and `node.param_bindings` keys.

15. **No stored `template_hash` either.** The "template group" concept is transient — the monomorphize pass snapshots each node's pre-mono `hash` into a local map at entry, picks block size per group, then rewrites; after the pass the grouping is no longer needed. Idempotence follows from `monomorphize_for_graph` being a no-op when `required_params(module)` is empty (an already-baked residual just re-hashes to the same value). The autotuning surface (Refactor 1 decision 10 — one `KernelTemplate` structure holding multiple block-choice variants per HIR family) collapses to a flat `Vec<KernelProgram>` keyed by residual hash for this refactor; autotuning reintroduces the outer layer when it lands, either by re-grouping at compile time or by having the block-hint policy expose its memo.

### Detailed implementation plan

#### Phase 1 — `ModuleCompiler` (new `module_compiler.rs`; `lib.rs`, `runtime.rs`, `kernel_cache.rs`)

The per-kernel backend, replacing `CompileOptions` + `compile_and_load`:

```rust
// module_compiler.rs
pub struct ModuleCompiler {
    verbosity: Verbosity,
    check_accesses: bool,        // config only; the check runs graph-side (needs bindings)
    dump_dir: Option<PathBuf>,
    nvcc: String,
    arch: Option<String>,
    flags: Vec<String>,          // extra nvcc flags
    nvcc_timeout: Duration,
}

impl ModuleCompiler {
    /// Env-var defaults: NVCC, CRYPTO_COMPILER_CUDA_ARCH, CRYPTO_COMPILER_DUMP_IR,
    /// CRYPTO_COMPILER_VERBOSITY, CRYPTO_COMPILER_CHECK_ACCESSES.
    pub fn new() -> Self { ... }

    pub fn set_verbosity(&mut self, v: Verbosity) -> &mut Self { ... }
    pub fn check_accesses(&mut self, on: bool) -> &mut Self { ... }
    pub fn dump_dir(&mut self, dir: impl Into<PathBuf>) -> &mut Self { ... }
    pub fn arch(&mut self, arch: &str) -> &mut Self { ... }
    pub fn flag(&mut self, flag: &str) -> &mut Self { ... }

    /// HIR → optimized KIR. Strict: no rewriting, only checking + lowering.
    /// Monomorphization is the graph compiler's job — a module with params in
    /// must-be-concrete positions is rejected, not fixed up.
    pub fn lower(&self, m: ir::Module) -> Result<KirProgram, CompileError> {
        // (1) internal block shapes must be monomorphized
        if let Some(&v) = required_params(&m).first() {
            return Err(CompileError::Monomorphize(format!(
                "param `{}` appears in a must-be-concrete position; \
                 monomorphize in the graph before lower", m.builder.param_name(v))));
        }
        let types = type_infer(&m)?;
        // (2) module must already be canonical (decision 6: errors if any
        //     canonicalize rewrite would fire)
        let program = extract_program(&m, types)?;
        // (3) exactly one kernel — splitting is the graph compiler's job
        if program.kernels.len() != 1 {
            return Err(CompileError::Lower(format!(
                "module `{}` contains {} kernels; split it in the graph first",
                m.name, program.kernels.len())));
        }
        // check_accesses needs concrete bindings, so the check itself runs
        // graph-side (GraphCompiler::compile, per unique (module, bindings));
        // the flag here only configures it.
        let mut kp = lower_to_kir(&program)?; // GlobalScratchPlan param dropped
        layout_infer(&mut kp)?;
        insert_sync(&mut kp);
        Ok(kp) // step dumps per verbosity/dump_dir throughout
    }

    /// KIR → dlopen'd CUDA artifact (source emission + verify + nvcc + dlopen).
    pub fn codegen(&self, kp: KirProgram) -> Result<KernelProgram, CompileError> {
        let src = passes::codegen(&kp)?; // plan_shared_mem stays internal (codegen.rs:327)
        verify(&kp)?;
        KernelProgram::load(&src, &self.nvcc_config()) // no more CompileOptions
    }
}
```

- Mechanical rename across the crate (decision 1): KIR container `KernelProgram` → `KirProgram` (kernel_ir.rs:574 + the codegen/layout_infer/insert_sync signatures); dlopen'd artifact `KernelModule` → `KernelProgram` (runtime.rs, kernel_cache.rs, graph_exe.rs). `KernelModuleNode` (the graph node) keeps its name.
- No `monomorphize` on `ModuleCompiler` (decision 4): `monomorphize_from_hint` (monomorphize.rs:332) is deleted, not relocated — the graph monomorphize pass (Phase 5) is the only monomorphization entry point.

- In `lower_to_kir` (lower_to_kir.rs:70-73) the scratch parameter is gone; with one kernel there are no `TensorRef::Let` intermediates, so any non-output let read is an internal error.
- `kernel_cache::compile_and_cache` (kernel_cache.rs:231) reworked around `lower`/`codegen`; the cache key (module_hash, kernel_cache.rs:101) is unchanged.
- Landing note: graph_exe keeps compiling via a temporary private `ModuleCompiler::compile_legacy` (the old full pipeline) until Phase 7 deletes it; the public surface is strict from day one.

#### Phase 2 — node metadata + `kernel_dedup` (`graph_ir.rs`)

The node carries its derived state; passes own it:

```rust
// graph_ir.rs
pub struct KernelModuleNode {
    pub module: Arc<ir::Module>,
    /// param name → concrete value for *this* node (decision 9).
    pub param_bindings: BTreeMap<String, i64>,
    pub inputs: Vec<BufId>,
    pub outputs: Vec<BufId>,
    // ---- derived state, owned by the graph passes ----
    /// Filled by `typecheck`; cleared when `module` is replaced.
    pub types: Option<Arc<TypeMap>>,
    /// α-normalized `module_hash`; filled by `kernel_dedup`; `None` = dirty.
    pub hash: Option<[u8; 32]>,
    /// `extract_program` fires no rewrite on `module`.
    pub canonical: bool,
    // No `baked` field (decision 14: residual module embodies baked values)
    // and no `template_hash` (decision 15: template grouping is a transient
    // view inside the monomorphize pass, not persistent state).
    pub fusion_history: Option<Arc<FusionHistory>>,
}

impl KernelModuleNode {
    /// The single mutation point: swap the module, invalidate derived state.
    pub fn replace_module(&mut self, m: impl Into<Arc<ir::Module>>) {
        self.module = m.into();
        self.types = None;
        self.hash = None;
        self.canonical = false;
    }
}

pub struct GraphBuilder {
    pub bufs: Vec<BufInfo>,
    pub nodes: Vec<GraphNode>,
    ..., // registered inputs/outputs etc. — unchanged
    /// Result of `plan_memory` (decision 12); any structural mutation
    /// (insert_*, split, fuse, dce-removal) resets it to `None`.
    pub plan: Option<MemoryPlan>,
    // DELETED: module_dedup, subgraph_cache, SubgraphCacheKey
}
```

`kernel_dedup` — internal, invoked at every pass entry:

```rust
fn kernel_dedup(g: &mut GraphBuilder) {
    // hash → canonical (module, types). First occurrence wins.
    let mut canon: HashMap<[u8; 32], (Arc<ir::Module>, Option<Arc<TypeMap>>)> = HashMap::new();
    for node in g.kernel_nodes_mut() {
        if node.hash.is_none() {
            prune_unused_params(node); // decision 9: drops param + its binding entry
            node.hash = Some(module_hash(&renumber_module(&node.module)));
        }
        match canon.entry(node.hash.unwrap()) {
            Entry::Vacant(e) => { e.insert((node.module.clone(), node.types.clone())); }
            Entry::Occupied(mut e) => {
                let (cm, ct) = e.get_mut();
                // hash covers param names (decision 10): same names in the same
                // order (VarIds may differ — α-normalized away), so the pointer
                // swap needs no binding-key remapping
                if !Arc::ptr_eq(&node.module, cm) {
                    debug_assert!(node.module.builder.params().iter().map(|(_, n)| n)
                        .eq(cm.builder.params().iter().map(|(_, n)| n)));
                    node.module = cm.clone();
                }
                // share types in whichever direction has them
                match (&node.types, &ct) {
                    (None, Some(t)) => node.types = Some(t.clone()),
                    (Some(t), None) => *ct = Some(t.clone()),
                    _ => {}
                }
                // canonical-ness is structural, so aliases agree by construction
            }
        }
    }
}
```

- Replaces insertion-time `module_dedup`, fusion's `dedup_modules` (fusion.rs:2052) and `g.dedup_module` (graph_ir.rs:783, fusion.rs:1446), and compile's phase-1b hash dedup (graph_exe.rs:404-427).
- Pass driver helper `for_each_unique_kernel(g, |module, node_indices| ...)` so per-module work (typecheck, canonicalize, …) runs once per hash and fans out.
- `IRBuilder::symbol` panics on a duplicate param name (decision 9).
- `module_hash` extended to include param *names* (decision 10) — the Refactor 1 "NOT names" choice (module_hash.rs:30-34) is reversed. This also changes on-disk kernel-cache keys once; acceptable (cache is a cache).
- `insert_kernel` slims to arity asserts + binding inference + push:

```rust
pub fn insert_kernel(
    &mut self,
    module: impl Into<Arc<ir::Module>>,
    inputs: impl IntoIterator<Item = BufId>,
    outputs: impl IntoIterator<Item = BufId>,
    shape_hint: &[(&str, i64)], // inference fallback only; never attached to the module
) {
    // arity asserts (as today), then:
    let param_bindings = self.infer_param_bindings(&module, &inputs, shape_hint);
    self.nodes.push(GraphNode::Kernel(KernelModuleNode {
        module, param_bindings, inputs, outputs,
        types: None, hash: None, canonical: false,
        fusion_history: None,
    }));
    self.plan = None;
}
```

- Delete `split_module` at insertion, `module_dedup`, `subgraph_cache`, `SubgraphCacheKey` (graph_ir.rs:330-357, 483-509). Multi-kernel modules now legally sit in the graph as one node until the canonicalize pass. `insert_subgraph` is deleted (verified: no callers outside graph_ir.rs).

#### Phase 3 — `typecheck` + `canonicalize` graph passes (methods on `GraphCompiler`)

All passes are `GraphCompiler` methods over `&mut GraphBuilder`, self-normalizing per the table above:

```rust
impl GraphCompiler {
    pub fn typecheck(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
        kernel_dedup(g);
        for_each_unique_kernel(g, |module, nodes| {
            if nodes.iter().any(|n| n.types.is_none()) {
                let t = Arc::new(type_infer(module)?);
                for n in nodes { n.types.get_or_insert_with(|| t.clone()); }
            }
            Ok(())
        })
    }

    pub fn canonicalize(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
        self.typecheck(g)?;
        for_each_unique_kernel(g, |module, nodes| {
            if nodes[0].canonical { return Ok(()); }
            let program = canonicalize(module, nodes[0].types_unwrapped())?; // rewriting form
            match program.kernels.len() {
                1 => for n in nodes {
                    n.replace_module(program.module.clone());
                    n.types = Some(program.types.clone()); // canonicalize refreshes types
                    n.canonical = true;                    // hash stays None → next dedup
                },
                _ => self.split_node(g, nodes, &program)?, // below; sets g.plan = None
            }
            Ok(())
        })?;
        kernel_dedup(g); // splits/rewrites minted new modules
        Ok(())
    }
}
```

- `split_node`: `split_module` on the canonical program, replace each alias node with one node per kernel in dependency order — today's `insert_subgraph_impl` wiring (graph_ir.rs:533-642): bound outputs land in the original node's output BufIds; intermediates get fresh `BufInfo`s sized from `OutputSpec::num_elems` *evaluated under that node's binding env* (error if symbolic). Child bindings by name projection (decision 11): `parent.param_bindings` filtered to the child's param names. Children start `canonical = true` (splitting a canonical program yields canonical kernels), `hash = None`.
- Finish with graph validation: every Kernel node is canonical single-kernel; `validate_interface` invariants hold.

#### Phase 4 — `lower_reduce` graph pass (`parallel_reduce_rewrite.rs`, fusion.rs)

```rust
pub fn lower_reduce(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
    self.typecheck(g)?;
    // decision 5: the gate needs concrete M/K, so memoize per (hash, the
    // binding values the module's reduce bounds actually read).
    let mut memo: HashMap<([u8; 32], BTreeMap<String, i64>), Option<Arc<ir::Module>>> = HashMap::new();
    for node in g.kernel_nodes_mut() {
        let key = (node.hash.unwrap(), relevant_bindings(node));
        let rewritten = memo.entry(key).or_insert_with(|| {
            // reuses node.types instead of re-running type_infer
            // (parallel_reduce_rewrite.rs:96); evaluates bounds under
            // binding_env(node) (as fusion.rs:163 does today);
            // None = gate says leave untouched.
            rewrite_parallel_reduce(&node.module, node.types_unwrapped(), &binding_env(node))
                .map(Arc::new)
        });
        if let Some(m) = rewritten {
            node.replace_module(m.clone()); // multi-kernel chain: stays ONE node
        }                                   // until canonicalize splits it
    }
    Ok(())
}
```

- Same-HIR nodes with different `K` bindings intentionally diverge (the rewrite bakes the bounds it restructures).
- Runs *before* monomorphize and canonicalize; a multi-stage chain stays one (multi-kernel) node until Phase 3's split turns the partials into graph buffers — the graph planner now places what `plan_global_scratch` used to.
- Delete `has_tree_lowered_reduce` / `top_value_tree_lowers` (fusion.rs:334-372): fusion scores the kernels that will actually run.

#### Phase 5 — `monomorphize` graph pass (moves graph_exe.rs:290-460 logic)

`GraphMono` goes name-keyed and `pub(crate)` — it is an internal data structure of the graph monomorphize pass, not crate API (its only consumers are `GraphCompiler::monomorphize` and `apply_fusion`'s re-monomorphize tail):

```rust
// monomorphize.rs
pub(crate) struct GraphMono {
    pub residual: Module,
    pub residual_bindings: BTreeMap<String, i64>, // surviving params
    /// Concretized params — transient diagnostic only (decision 14):
    /// never stored on the node, the residual embodies these values.
    pub baked: BTreeMap<String, i64>,
    pub max_outer: Option<i64>, // Some iff residual keeps a symbolic outer bound
}
pub(crate) fn monomorphize_for_graph(m: &Module, bindings: &BTreeMap<String, i64>)
    -> Result<GraphMono, CompileError>;
```

```rust
pub fn monomorphize(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
    kernel_dedup(g);
    // Template group = pre-mono hash (each node's current node.hash before
    // this pass rewrites it); transient, local to this call (decision 15).
    let mut groups: HashMap<[u8; 32], Vec<NodeIdx>> = HashMap::new();
    for (i, node) in g.kernel_nodes().enumerate() {
        groups.entry(node.hash.unwrap()).or_default().push(i);
    }
    let mut max_outer: HashMap<[u8; 32], i64> = HashMap::new(); // by pre-mono hash
    let mut gms: HashMap<NodeIdx, GraphMono> = HashMap::new();  // memo per (hash, bindings) elided
    for (pre_hash, nodes) in &groups {
        for &i in nodes {
            let node = &g.kernel_nodes()[i];
            let gm = monomorphize_for_graph(&node.module, &node.param_bindings)?;
            // Already-baked residuals produce required_params.is_empty() and
            // gm.residual is structurally equal to node.module — no-op path
            // gives idempotence without a stored marker.
            if let Some(m) = gm.max_outer {
                max_outer.entry(*pre_hash).and_modify(|v| *v = (*v).max(m)).or_insert(m);
            }
            gms.insert(i, gm);
        }
    }
    // Block hints per template group — verbatim move of phase 1a′
    // (graph_exe.rs:363-391): residuals without an author hint get
    // set_block_hint(block_size_policy(max_outer[pre_hash])).
    for (pre_hash, nodes) in &groups {
        let block = block_size_policy(max_outer.get(pre_hash).copied());
        for &i in nodes {
            let gm = gms.remove(&i).unwrap();
            let mut residual = gm.residual;
            residual.set_block_hint_if_absent(block);
            let node = &mut g.kernel_nodes_mut()[i];
            node.replace_module(residual);           // residual IS the record of
            node.param_bindings = gm.residual_bindings; // what got baked (decision 14)
        }
    }
    kernel_dedup(g); // residuals agreeing across sizes collapse (old phase-1b)
    Ok(())
}
```

#### Phase 6 — `fuse` + `dce` + `plan_memory` passes (fusion.rs, planner.rs)

```rust
pub fn fuse(&self, g: &mut GraphBuilder) -> Result<FusionReport, CompileError> {
    self.lower_reduce(g)?;   // each self-normalizes; cheap no-ops when clean
    self.monomorphize(g)?;
    self.canonicalize(g)?;
    let report = fuse_graph(g, &self.fusion_options); // rounds end with kernel_dedup
    g.plan = None;
    Ok(report)
}

pub fn dce(&self, g: &mut GraphBuilder) -> usize {
    let removed = fusion::dce(g); // fusion.rs:1879, logic unchanged
    if removed > 0 { g.plan = None; }
    removed
}

pub fn plan_memory(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
    if g.plan.is_none() {
        g.plan = Some(planner::plan(g, &self.plan_env, self.device)?);
    }
    Ok(())
}
```

- `extract_relations`/`extract_one` consume `node.hash` + `node.types` + the canonical single-kernel invariant: extraction becomes `extract_program` (strict, no rewrites) — delete the internal `type_infer` + `canonicalize` (fusion.rs:209-210) and the hand-rolled `(module_hash, bindings)` memo keying on recomputed hashes (fusion.rs:156, 186).
- `apply_fusion`: drop the re-canonicalization of inputs (fusion.rs:1190-1195, inputs are canonical now); keep grafting; finish with the normalize sequence of decision 7:

```rust
// apply_fusion tail — fusion re-normalizes its own output
let types = type_infer(&fused)?;
let program = canonicalize(&fused, types)?;
assert_single_kernel(&program)?;                 // exists today (fusion.rs:1420-1425)
// grafting can expose NEW must-be-concrete positions (producer's outer
// compute became an inner tile) → re-monomorphize under merged bindings
let gm = monomorphize_for_graph(&program.module, &merged_bindings)?;
fixup_block_hint(&mut gm.residual, ...);         // per template-group policy
node.replace_module(gm.residual);                // hash = None → next kernel_dedup
node.param_bindings = gm.residual_bindings;
node.canonical = true;
```

- Seam unification is directional (decision 10): a bare-vs-bare seam substitutes the *consumer's* param with the producer's — producer names survive; bare-vs-compound is forced by structure. Non-unified consumer params colliding with a surviving producer name are renamed `{name}__c` first, then merged bindings are a plain name-keyed map union (seam-unified params already agree on values by construction; the rename applies to the consumer's binding keys too). `fusion_history` mechanics unchanged (the visualizer depends on them).
- `fuse_graph` (fusion.rs:1966): rounds end with `kernel_dedup` instead of `dedup_modules`; `distinct_kernel_modules` reads node hashes.
- Delete `fuse_only` (graph_exe.rs:206) per decision 13; the visualizer calls `fuse`.

#### Phase 7 — `GraphCompiler::compile` + scratch removal + runner deletion (`graph_exe.rs`, `kernel_ir.rs`, `codegen.rs`, `runtime.rs`, `runner.rs`)

```rust
pub struct GraphCompiler {
    module_compiler: ModuleCompiler, // replaces compile_options: CompileOptions
    fusion: Option<FusionOptions>,
    ..., // cache, device, planner options — unchanged
}

pub fn compile(&self, mut g: GraphBuilder, ...) -> Result<GraphExe, CompileError> {
    validate_interface(&g)?;
    match &self.fusion {
        Some(_) => { self.fuse(&mut g)?; }   // pulls in lower_reduce/monomorphize/canonicalize
        None => {
            self.lower_reduce(&mut g)?;
            self.monomorphize(&mut g)?;
            self.canonicalize(&mut g)?;
        }
    }
    self.dce(&mut g);
    // check_accesses (flag on module_compiler) per unique (module, bindings) —
    // as today (phase 1a); needs bindings, hence graph-side, not in lower()
    // per unique node.hash (flat — no KernelTemplate outer layer per
    // decision 15; autotuning will reintroduce it): disk-cache probe →
    // on miss, parallel
    //   module_compiler.lower(module)? -> KirProgram
    //   module_compiler.codegen(kir)?  -> KernelProgram   (dlopen'd artifact)
    self.plan_memory(&mut g)?;
    // assemble GraphExe from g.plan (order + offsets) — no scratch step
}

pub struct GraphExe {
    kernels: Vec<KernelProgram>, // one dlopen'd artifact per unique residual hash
    nodes: Vec<ExeNode>,         // kernel index + param_bindings + buffer wiring
    plan: MemoryPlan,
    pool: DevicePool,
    ...,
}
```

- `GraphExe::run` (and CUDA-graph capture): for each node, push `param_bindings` through `KernelProgram::set_symbol`, then launch — today's set_param plumbing unchanged in shape. Params never change post-compile, so capture is unaffected.
- `GraphExe::kernel_program(&mut self, node) -> &mut KernelProgram`: direct handle to a node's artifact for standalone driving. The artifact serves *any* binding whose residual hash matches (the grid guard makes every `n` correct; the block choice is perf-only): `set_symbol` a residual param, re-query `input_size(i)`/`output_size(i)`, realloc, `run`. Baked params have no symbol slot — rebinding one errors by absence.
- Delete all module-scratch plumbing: `PreExe::Kernel.scratch` + the scratch-rejection error (graph_exe.rs:626-655), scratch access injection (graph_exe.rs:684-703), run-time `set_scratch` (graph_exe.rs:1286); `plan_global_scratch.rs` entirely; `KirProgram::scratch_bytes` (kernel_ir.rs:578); host-ABI `scratch_size`/`set_scratch_buf`/`Prog::scratch` (codegen.rs:1465, 1509-1542); runtime `scratch`/`scratch_bound`/`ensure_scratch`/`set_scratch`/`scratch_size` (runtime.rs:137-157, 550-591).
- **`runner.rs` is deleted** (decision 3). Its conveniences (input upload, `run`/`bench`, output readback) become a small `test-utils` helper over `GraphBuilder`/`GraphCompiler`/`GraphExe`. Multi-size re-runs (gpu_macro.rs:2367-2445) compile once and then drive `kernel_program(node)` directly: `set_symbol("n", size)` → realloc from the artifact's `input_size` queries (today's runner `bind_buffers` pattern moves into the helper) → `run`. One nvcc + one dlopen for N sizes, trivially — no re-JIT, no loaded-artifact bookkeeping.
- Rename the `set_param` surface to `set_symbol` end to end (vtable slot + emitted host fn, codegen.rs:1139-1216; `KernelProgram::set_param` runtime.rs:511; `params_bound` → `symbols_bound`) — mechanical, aligns the ABI with `IRBuilder::symbol`/`register_symbol`.
- Make `planner` a default feature; drop the now-redundant `#![cfg(feature = "planner")]`-driven gaps in coverage.
- Delete `compile_and_load`, `compile_and_load_with_hint`, `CompileOptions`, and `ModuleCompiler::compile_legacy`.

#### Testing

- First (per workflow), a new gpu_graph test for the headline capability: a graph containing a large top-level reduce that tree-lowers to **two stages** — today this dies with the module-scratch error (graph_exe.rs:642); after the refactor it must compile (reduce split into multiple graph nodes, partials planner-placed) and match the CPU reference. Assert the extra kernel node exists.
- CPU-only unit tests per pass:
  - `kernel_dedup`: two Arc-distinct but structurally identical modules → one hash, shared Arc; dead-param pruning keeps binding keys = param names;
  - `module_hash`: param names are hashed — same structure, different param name (`n` vs `len`) → **different** hashes (decision 10);
  - fusion naming: bare-vs-bare seam keeps the producer's param name; a non-unified consumer param colliding with a producer name is renamed `{name}__c` with its binding key; two structurally equal fusion products still dedup to one hash;
  - `typecheck`: aliases share one `Arc<TypeMap>`;
  - `canonicalize`: multi-kernel module → N nodes, intermediate buffer sizes evaluated under bindings, name-projected child bindings; an already-canonical module round-trips with a stable hash;
  - `lower_reduce`: one module, two binding sets (K above/below the gate) → one node rewritten, one untouched;
  - `monomorphize`: block hint chosen from the transient template group's max size (decision 15); two sizes of one template → one unique residual after the trailing `kernel_dedup` when the residuals structurally agree;
  - **any-order/self-normalizing protocol**: `fuse` called directly on a freshly built graph (no prior passes) produces the same result as the explicit chain; calling any pass twice is a no-op (node hashes and module `Arc` pointers stable across the second call);
  - `plan_memory`/`dce`: `plan_memory` fills `g.plan`; a subsequent `dce` that removes a node clears it; a `dce` that removes nothing keeps it;
  - `IRBuilder::symbol`: duplicate param name in one module → error;
  - `ModuleCompiler`: `lower` rejects (a) non-canonical, (b) multi-kernel, (c) unmonomorphized-inner-bound modules with the right error variants;
  - fusion: `apply_fusion` output is canonical + single-kernel + hash-`None`; extraction does not mutate node modules (hash stable across a `fuse` round);
  - direct artifact driving: the existing gpu_macro multi-size tests compile once, then re-run via `kernel_program(node)` — `set_symbol` + realloc from `input_size` queries at N sizes — asserting one nvcc (one disk-cache insert), one dlopen, correct results at every size, and that rebinding a baked param errors.
- Existing suites stay green throughout: gpu.rs, gpu_macro.rs, gpu_graph.rs, custom_kernels.rs, benches (now under default `planner`).
- Compile-time regression: the synthetic multi-size graph from Refactor 1's E2E — unique-module count and cold-compile wall time must not regress (the pipeline does strictly less repeated work, so expect improvement).

### Open questions

- `GraphExe::run` binds params per node before launch; with CUDA-graph capture this is unchanged, but if `lower_reduce` baking makes some previously-shared modules diverge, launch-count-sensitive tests (gpu_graph.rs fused-vs-unfused) may need count updates rather than logic changes.
- The old `ModuleRunner::bench` timed a single artifact `run`; the test-utils bench helper can time either one artifact's `run` (per-kernel) or a whole (possibly multi-launch) `GraphExe` replay. Default to per-kernel for the NTT benches so figures stay comparable to historical numbers.

