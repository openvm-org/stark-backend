//! Splits a multi-kernel [`Module`] into a [`ModuleSubgraph`]: one
//! single-kernel `Module` per top-level compute/reduce, wired together by
//! [`SubgraphValue`] edges.
//!
//! A user-facing `Module` may chain several kernels through top-level `let`
//! bindings (`let t = compute ...; compute ... t[i] ...`). Graph-level
//! optimizations (fusion, in-place rewrites, scheduling) want one kernel per
//! graph node, so [`crate::graph_ir::GraphBuilder::insert_kernel`] runs
//! [`split_module`] on insertion and inserts one
//! [`crate::graph_ir::GraphNode::Kernel`] per [`SubgraphKernel`].
//!
//! Each split module's body is the *original* HIR subtree of its kernel
//! (extracted via [`CanonKernel::source`]), deep-copied into a fresh builder:
//!
//! - `VarId`s are preserved verbatim (module hashes stay deterministic; the destination builder's
//!   fresh-variable watermark is raised past the source's so later passes can't collide);
//! - references to module inputs and to other kernels' outputs become new input declarations,
//!   recorded in [`SubgraphKernel::inputs`];
//! - scalar top-level `let`s are inlined at their use sites;
//! - a scalar-typed reference to another kernel's output (e.g. a top-level `reduce` result used
//!   inside a later compute) becomes a shape-`[]` scalar input — a pattern the single-module
//!   pipeline rejects, but which works across split kernels.
//!
//! Invariant: every split module contains exactly one top-level kernel (its
//! body is a single `compute` or a bare `reduce` — never a `let` chain).
//! This is an HIR-level statement; JIT-time rewrites such as
//! [`rewrite_parallel_reduce`](crate::passes::parallel_reduce_rewrite) may
//! still expand one HIR kernel into several CUDA kernels inside its compiled
//! [`crate::runtime::KernelModule`].

use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

#[cfg(doc)]
use crate::passes::canonicalize::CanonKernel;
use crate::{
    ir::{IRBuilder, Module, Node, NodeId, ScalarType, SizeExpr, Type, VarId},
    passes::{
        canonicalize::{canonicalize, CanonValue, Program, TensorRef},
        type_infer::type_infer,
    },
    CompileError,
};

/// Where a split kernel's input (or a subgraph output) comes from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SubgraphValue {
    /// Input `k` of the original module.
    Input(usize),
    /// Output `out_idx` of [`ModuleSubgraph::kernels`]`[kernel]`.
    KernelOutput { kernel: usize, out_idx: usize },
}

/// Element type and count of one split-kernel output, matching the buffer
/// the compiled kernel will write (`num_elems * elem.size_bytes()` bytes).
/// The count may reference module parameters; graph insertion evaluates it
/// through the node's inferred bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSpec {
    pub elem: ScalarType,
    pub num_elems: SizeExpr,
}

impl OutputSpec {
    /// Concrete byte size; `None` while the count references parameters.
    pub fn size_bytes(&self) -> Option<usize> {
        Some(self.num_elems.as_const()? as usize * self.elem.size_bytes())
    }
}

/// One single-kernel module of a split, with its dataflow edges.
pub struct SubgraphKernel {
    /// Single-kernel module (body is one `compute` or a bare `reduce`).
    pub module: Arc<Module>,
    /// One entry per `module.builder.inputs()` declaration, in order: where
    /// each input reads from.
    pub inputs: Vec<SubgraphValue>,
    /// One entry per module output.
    pub outputs: Vec<OutputSpec>,
}

/// A [`Module`] split into single-kernel modules. Kernels are listed in
/// dependency order: a kernel only reads [`SubgraphValue::KernelOutput`]s of
/// kernels at earlier indices.
pub struct ModuleSubgraph {
    /// Name of the original module.
    pub name: String,
    /// Number of inputs of the original module.
    pub num_inputs: usize,
    pub kernels: Vec<SubgraphKernel>,
    /// The original module's outputs, one per top-level output tensor.
    pub outputs: Vec<SubgraphValue>,
}

/// Type-checks and canonicalizes `module`, then splits it into one
/// single-kernel [`Module`] per top-level compute/reduce.
///
/// Splitting is deterministic: structurally identical modules produce
/// structurally identical (equal [`crate::module_hash::module_hash`]) split
/// modules.
pub fn split_module(module: &Module) -> Result<ModuleSubgraph, CompileError> {
    let types = type_infer(module)?;
    let program = canonicalize(module.clone(), types)?;

    let name = program.module.name.clone();
    let num_inputs = program.module.builder.inputs().len();
    let single = program.kernels.len() == 1;

    let mut kernels = Vec::with_capacity(program.kernels.len());
    for (i, ck) in program.kernels.iter().enumerate() {
        // A single-kernel module keeps its original name so dumps, profiles
        // and cache keys stay recognizable; multi-kernel splits get a
        // deterministic per-kernel suffix.
        let split_name = if single {
            name.clone()
        } else {
            format!("{name}__k{i}")
        };
        let (module, inputs) = extract_kernel(&program, ck.source, split_name)?;
        let outputs = ck
            .member_types
            .iter()
            .map(|t| {
                let elem = t.scalar_type().ok_or_else(|| {
                    CompileError::Canonicalize(format!(
                        "kernel `{}` has a non-tensor member type {t:?}",
                        ck.name
                    ))
                })?;
                let num_elems = SizeExpr::product(t.shape()).ok_or_else(|| {
                    CompileError::Canonicalize(format!(
                        "kernel `{}` output shape ({t:?}) multiplies two compound symbolic \
                         dimensions; the element count is not representable",
                        ck.name
                    ))
                })?;
                Ok(OutputSpec { elem, num_elems })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;
        kernels.push(SubgraphKernel {
            module: Arc::new(module),
            inputs,
            outputs,
        });
    }

    let outputs = program.outputs.iter().map(|&r| value_of(r)).collect();
    Ok(ModuleSubgraph {
        name,
        num_inputs,
        kernels,
        outputs,
    })
}

fn value_of(r: TensorRef) -> SubgraphValue {
    match r {
        TensorRef::Input(k) => SubgraphValue::Input(k),
        TensorRef::Let { let_id, out_idx } => SubgraphValue::KernelOutput {
            kernel: let_id,
            out_idx,
        },
    }
}

/// Deep-copies the kernel subtree rooted at `source` out of
/// `program.module.builder` into a fresh single-kernel [`Module`].
fn extract_kernel(
    program: &Program,
    source: NodeId,
    name: String,
) -> Result<(Module, Vec<SubgraphValue>), CompileError> {
    let mut ex = Extract {
        program,
        dst: IRBuilder::new(),
        inputs: Vec::new(),
        input_nodes: HashMap::new(),
        memo: HashMap::new(),
        used_params: BTreeSet::new(),
    };
    let body = ex.copy(source)?;
    let Extract {
        mut dst,
        inputs,
        used_params,
        ..
    } = ex;

    // Inherit the parent's parameter declarations referenced by the copied
    // subtree (bounds and input shapes), in parent declaration order so
    // splitting stays deterministic; slice the shape hint to match.
    let src = &program.module.builder;
    let mut hint = Vec::new();
    for (idx, (v, pname)) in src.params().iter().enumerate() {
        if used_params.contains(v) {
            dst.inherit_param(*v, pname.clone());
            if let Some(h) = src.shape_hint() {
                hint.push(h[idx]);
            }
        }
    }
    if src.shape_hint().is_some() && !dst.params().is_empty() {
        dst.add_shape_hint(&hint);
    }

    dst.raise_var_watermark(src.var_watermark());
    Ok((dst.finish(name, body), inputs))
}

struct Extract<'a> {
    program: &'a Program,
    dst: IRBuilder,
    /// Source of each input declared in `dst` so far, in declaration order.
    inputs: Vec<SubgraphValue>,
    /// Dedup: one `dst` input per distinct [`SubgraphValue`].
    input_nodes: HashMap<SubgraphValue, NodeId>,
    /// Source `NodeId` -> copied `dst` `NodeId`.
    memo: HashMap<NodeId, NodeId>,
    /// Parent-module parameter `VarId`s referenced by copied bounds/shapes.
    used_params: BTreeSet<VarId>,
}

impl Extract<'_> {
    fn src(&self) -> &IRBuilder {
        &self.program.module.builder
    }

    /// Declares (or reuses) a `dst` input reading from `val`. `src_node` is
    /// the node being replaced; its type determines the declaration's
    /// element type and shape (a scalar type declares a shape-`[]` input).
    fn value_input(
        &mut self,
        val: SubgraphValue,
        src_node: NodeId,
    ) -> Result<NodeId, CompileError> {
        if let Some(&n) = self.input_nodes.get(&val) {
            return Ok(n);
        }
        let (name, elem, shape) = match val {
            SubgraphValue::Input(k) => {
                let d = &self.src().inputs()[k];
                (d.name.clone(), d.elem, d.shape.clone())
            }
            SubgraphValue::KernelOutput { kernel, out_idx } => {
                let ty = self.program.types.try_get(src_node).ok_or_else(|| {
                    CompileError::Canonicalize(format!("missing type for node {src_node:?}"))
                })?;
                let (elem, shape) = match ty {
                    Type::Scalar(s) => (*s, Vec::new()),
                    Type::Tensor(s, sh) => (*s, sh.clone()),
                    Type::Tuple(_) => {
                        return Err(CompileError::Canonicalize(
                            "tuple-valued reference cannot become a kernel input; \
                             project it first"
                                .into(),
                        ))
                    }
                };
                (format!("t{kernel}_{out_idx}"), elem, shape)
            }
        };
        for dim in &shape {
            dim.param_syms(&mut self.used_params);
        }
        let n = self.dst.input(name, elem, shape);
        self.inputs.push(val);
        self.input_nodes.insert(val, n);
        Ok(n)
    }

    fn copy(&mut self, id: NodeId) -> Result<NodeId, CompileError> {
        if let Some(&r) = self.memo.get(&id) {
            return Ok(r);
        }
        let node = self.src().node(id).clone();
        let result = match node {
            Node::Input(k) => self.value_input(SubgraphValue::Input(k), id)?,
            Node::ConstSym(ref e) => {
                e.param_syms(&mut self.used_params);
                self.dst.intern(node)
            }
            Node::Var(v) => match self.program.env.get(&v).cloned() {
                // Kernel-local binder (loop index, inner let): preserved.
                None => self.dst.intern(Node::Var(v)),
                // Scalar top-level let: inline a copy of its value.
                Some(CanonValue::Scalar(e)) => self.copy(e)?,
                Some(CanonValue::Tensors(refs)) => {
                    if refs.len() != 1 {
                        return Err(CompileError::Canonicalize(format!(
                            "tuple-valued variable {v:?} used without projection"
                        )));
                    }
                    self.value_input(value_of(refs[0]), id)?
                }
            },
            Node::Proj(t, k) => {
                // `Proj(Var(v), k)` where `v` is a top-level tensor tuple
                // resolves straight to that member.
                if let Node::Var(v) = self.src().node(t) {
                    if let Some(CanonValue::Tensors(refs)) = self.program.env.get(v) {
                        let r = *refs.get(k).ok_or_else(|| {
                            CompileError::Canonicalize(format!(
                                "projection index {k} out of bounds"
                            ))
                        })?;
                        let n = self.value_input(value_of(r), id)?;
                        self.memo.insert(id, n);
                        return Ok(n);
                    }
                }
                let t2 = self.copy(t)?;
                self.dst.intern(Node::Proj(t2, k))
            }
            Node::ConstU32(_) | Node::ConstField(_) | Node::ConstFpExt(_) => self.dst.intern(node),
            Node::LiftFpExt(x) => {
                let x2 = self.copy(x)?;
                self.dst.intern(Node::LiftFpExt(x2))
            }
            Node::Bin(op, a, b) => {
                let a2 = self.copy(a)?;
                let b2 = self.copy(b)?;
                self.dst.intern(Node::Bin(op, a2, b2))
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                let c2 = self.copy(cond)?;
                let t2 = self.copy(then_val)?;
                let e2 = self.copy(else_val)?;
                self.dst.intern(Node::Select {
                    cond: c2,
                    then_val: t2,
                    else_val: e2,
                })
            }
            Node::Index { tensor, indices } => {
                let t2 = self.copy(tensor)?;
                let ix2 = indices
                    .iter()
                    .map(|&ix| self.copy(ix))
                    .collect::<Result<Vec<_>, _>>()?;
                self.dst.intern(Node::Index {
                    tensor: t2,
                    indices: ix2,
                })
            }
            Node::Compute {
                bound,
                var,
                body,
                scatter,
                par,
                threads,
            } => {
                bound.param_syms(&mut self.used_params);
                let body2 = self.copy(body)?;
                self.dst.intern(Node::Compute {
                    bound,
                    var,
                    body: body2,
                    scatter,
                    par,
                    threads,
                })
            }
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => {
                bound.param_syms(&mut self.used_params);
                let body2 = self.copy(body)?;
                self.dst.intern(Node::Reduce {
                    op,
                    bound,
                    var,
                    body: body2,
                })
            }
            Node::Let { var, value, body } => {
                let v2 = self.copy(value)?;
                let b2 = self.copy(body)?;
                self.dst.intern(Node::Let {
                    var,
                    value: v2,
                    body: b2,
                })
            }
            Node::Tuple(elems) => {
                let e2 = elems
                    .iter()
                    .map(|&e| self.copy(e))
                    .collect::<Result<Vec<_>, _>>()?;
                self.dst.intern(Node::Tuple(e2))
            }
            Node::Pack(elems) => {
                let e2 = elems
                    .iter()
                    .map(|&e| self.copy(e))
                    .collect::<Result<Vec<_>, _>>()?;
                self.dst.intern(Node::Pack(e2))
            }
        };
        self.memo.insert(id, result);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::ScalarType, module_hash::module_hash};

    /// `let t = a * 2; out = t + a` as a two-kernel chain over one input.
    fn two_kernel_chain(n: usize) -> Module {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let t = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let two = b.const_field(2);
            b.mul(ai, two)
        });
        let t = b.let_bound(t);
        let out = b.compute(n, |b, i| {
            let ti = b.index(t, &[i]);
            let ai = b.index(a, &[i]);
            b.add(ti, ai)
        });
        b.finish("chain", out)
    }

    #[test]
    fn splits_chain_into_two_kernels() {
        let sg = split_module(&two_kernel_chain(8)).expect("split");
        assert_eq!(sg.name, "chain");
        assert_eq!(sg.num_inputs, 1);
        assert_eq!(sg.kernels.len(), 2);

        let k0 = &sg.kernels[0];
        assert_eq!(k0.module.name, "chain__k0");
        assert_eq!(k0.inputs, vec![SubgraphValue::Input(0)]);
        assert_eq!(
            k0.outputs,
            vec![OutputSpec {
                elem: ScalarType::BabyBear,
                num_elems: 8usize.into()
            }]
        );

        let k1 = &sg.kernels[1];
        assert_eq!(k1.module.name, "chain__k1");
        // Declaration order follows first use in the body: `t` then `a`.
        assert_eq!(
            k1.inputs,
            vec![
                SubgraphValue::KernelOutput {
                    kernel: 0,
                    out_idx: 0
                },
                SubgraphValue::Input(0),
            ]
        );
        assert_eq!(k1.module.builder.inputs().len(), 2);

        assert_eq!(
            sg.outputs,
            vec![SubgraphValue::KernelOutput {
                kernel: 1,
                out_idx: 0
            }]
        );
    }

    #[test]
    fn split_kernels_are_single_kernel_modules() {
        let sg = split_module(&two_kernel_chain(8)).expect("split");
        for k in &sg.kernels {
            let types = type_infer(&k.module).expect("type_infer");
            let program = canonicalize((*k.module).clone(), types).expect("canonicalize");
            assert_eq!(program.kernels.len(), 1, "module `{}`", k.module.name);
        }
    }

    #[test]
    fn split_is_deterministic_across_identical_modules() {
        let a = split_module(&two_kernel_chain(8)).expect("split");
        let b = split_module(&two_kernel_chain(8)).expect("split");
        assert_eq!(a.kernels.len(), b.kernels.len());
        for (ka, kb) in a.kernels.iter().zip(&b.kernels) {
            assert_eq!(module_hash(&ka.module), module_hash(&kb.module));
        }
    }

    #[test]
    fn single_kernel_module_keeps_name() {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![4]);
        let body = b.compute(4, |b, i| {
            let ai = b.index(a, &[i]);
            let two = b.const_field(2);
            b.mul(ai, two)
        });
        let m = b.finish("scale_by_two", body);

        let sg = split_module(&m).expect("split");
        assert_eq!(sg.kernels.len(), 1);
        assert_eq!(sg.kernels[0].module.name, "scale_by_two");
    }

    #[test]
    fn tuple_output_kernel_and_projection_consumer() {
        // k0 emits a tuple (x*2, x*3); k1 consumes only member 1.
        let n = 4;
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let pair = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let two = b.const_field(2);
            let three = b.const_field(3);
            let x = b.mul(ai, two);
            let y = b.mul(ai, three);
            b.tuple(&[x, y])
        });
        let pair = b.let_bound(pair);
        let second = b.proj(pair, 1);
        let out = b.compute(n, |b, i| {
            let si = b.index(second, &[i]);
            let one = b.const_field(1);
            b.add(si, one)
        });
        let m = b.finish("pair_use", out);

        let sg = split_module(&m).expect("split");
        assert_eq!(sg.kernels.len(), 2);
        assert_eq!(sg.kernels[0].outputs.len(), 2);
        assert_eq!(
            sg.kernels[1].inputs,
            vec![SubgraphValue::KernelOutput {
                kernel: 0,
                out_idx: 1
            }]
        );
    }

    #[test]
    fn reduce_result_becomes_scalar_input() {
        // k0 is a bare top-level reduce; k1 uses its scalar result.
        let n = 8;
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let s = b.reduce_add(n, |b, i| b.index(a, &[i]));
        let s = b.let_bound(s);
        let out = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            b.mul(ai, s)
        });
        let m = b.finish("scale_by_sum", out);

        let sg = split_module(&m).expect("split");
        assert_eq!(sg.kernels.len(), 2);
        // k0: bare reduce body, one 1-element output.
        assert!(matches!(
            sg.kernels[0].module.builder.node(sg.kernels[0].module.body),
            Node::Reduce { .. }
        ));
        assert_eq!(
            sg.kernels[0].outputs,
            vec![OutputSpec {
                elem: ScalarType::BabyBear,
                num_elems: 1usize.into()
            }]
        );
        // k1 declares the sum as a shape-[] scalar input.
        let k1 = &sg.kernels[1];
        assert_eq!(
            k1.inputs,
            vec![
                SubgraphValue::Input(0),
                SubgraphValue::KernelOutput {
                    kernel: 0,
                    out_idx: 0
                },
            ]
        );
        let decls = k1.module.builder.inputs();
        assert!(decls[1].shape.is_empty());
        assert_eq!(decls[1].elem, ScalarType::BabyBear);
    }

    #[test]
    fn split_inherits_referenced_params() {
        // k0's reduce bound references `n`, k1's references `m`; each split
        // module inherits only its own parameter (VarId preserved) plus the
        // matching slice of the shape hint. Outputs stay concrete so the
        // split itself succeeds.
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let m = b.symbol("m");
        b.add_shape_hint(&[8, 16]);
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let t = b.compute(4, |b, _i| b.reduce_add(n, |b, j| b.index(a, &[j])));
        let t = b.let_bound(t);
        let out = b.compute(4, |b, i| {
            let s = b.reduce_add(m, |b, _j| b.index(t, &[i]));
            let ti = b.index(t, &[i]);
            b.add(ti, s)
        });
        let module = b.finish("sym_chain", out);

        let sg = split_module(&module).expect("split");
        assert_eq!(sg.kernels.len(), 2);

        let src_params = module.builder.params();
        let b0 = &sg.kernels[0].module.builder;
        assert_eq!(b0.params(), &src_params[..1]);
        assert_eq!(b0.shape_hint(), Some(&[8i64][..]));
        // k0's input `a` keeps its symbolic shape.
        assert!(b0.inputs()[0].shape[0].as_const().is_none());

        let b1 = &sg.kernels[1].module.builder;
        assert_eq!(b1.params(), &src_params[1..]);
        assert_eq!(b1.shape_hint(), Some(&[16i64][..]));
    }

    #[test]
    fn scalar_top_level_let_is_inlined() {
        // `let c = 2 + 3; out = compute |i| a[i] * c` stays one kernel with
        // the scalar expression inlined (no extra input).
        let n = 4;
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let two = b.const_field(2);
        let three = b.const_field(3);
        let c = b.add(two, three);
        let c = b.let_bound(c);
        let out = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            b.mul(ai, c)
        });
        let m = b.finish("scale_const", out);

        let sg = split_module(&m).expect("split");
        assert_eq!(sg.kernels.len(), 1);
        assert_eq!(sg.kernels[0].inputs, vec![SubgraphValue::Input(0)]);
        assert_eq!(sg.kernels[0].module.builder.inputs().len(), 1);
    }
}
