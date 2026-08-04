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
9. **Param bindings are inferred, plus an optional per-module shape hint.** At `insert_kernel`, per-node bindings are inferred by unifying the module's input shape exprs against the concrete graph buffer shapes (bare-param dims solve directly; compound dims are verified for consistency). Separately, `ir::Module::add_shape_hint(...)` attaches zero-or-one canonical concrete instantiation to the module itself; the hint supplies values for access checking and for monomorphization when no graph bindings exist (standalone `compile_and_load`). If monomorphization is required for codegen and no values are available (no bindings, no hint) → compile error.
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
- `module_hash` (module_hash.rs:27-53): hash param declarations (count + order, α-normalized ids, NOT names or bindings) and shape/bound exprs structurally. `renumber_module` (fusion.rs:1864) renumbers param VarIds too.
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
- `runtime.rs`: `VTable` + `KernelModule::set_param(i, v)`. Metadata queries (`input_size` etc.) are only valid after params are set — enforce with a `params_bound` flag like `scratch_bound`.
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
- Same-HIR variants (block size / monomorphized values) collected under one `KernelTemplate` structure → future autotuning.



