//! Helpers shared between passes.

use std::collections::HashMap;

use crate::{
    ir::{IRBuilder, Node, NodeId, VarId},
    passes::canonicalize::{CanonValue, TensorRef},
    CompileError,
};

pub(crate) fn ceil_log2(n: usize) -> usize {
    n.max(1).next_power_of_two().trailing_zeros() as usize
}

/// Resolves a tensor-typed HIR expression to a top-level tensor reference.
pub(crate) fn resolve_tensor_ref(
    b: &IRBuilder,
    env: &HashMap<VarId, CanonValue>,
    id: NodeId,
) -> Result<TensorRef, CompileError> {
    match b.node(id) {
        Node::Input(k) => Ok(TensorRef::Input(*k)),
        Node::Var(v) => match env.get(v) {
            Some(CanonValue::Tensors(refs)) if refs.len() == 1 => Ok(refs[0]),
            Some(CanonValue::Tensors(_)) => Err(CompileError::Lower(
                "tuple-valued variable used where a single tensor is expected".into(),
            )),
            Some(CanonValue::Scalar(_)) => Err(CompileError::Lower(
                "scalar variable indexed as a tensor".into(),
            )),
            None => Err(CompileError::Lower(format!(
                "indexed variable {v:?} is not bound to a top-level tensor; \
                 inner tensors are not supported yet"
            ))),
        },
        Node::Proj(t, k) => match b.node(*t) {
            Node::Var(v) => match env.get(v) {
                Some(CanonValue::Tensors(refs)) => refs.get(*k).copied().ok_or_else(|| {
                    CompileError::Lower(format!("projection index {k} out of bounds"))
                }),
                _ => Err(CompileError::Lower(
                    "projection from a non-tuple variable".into(),
                )),
            },
            Node::Tuple(elems) => {
                let e = *elems
                    .get(*k)
                    .ok_or_else(|| CompileError::Lower(format!("projection index {k} OOB")))?;
                resolve_tensor_ref(b, env, e)
            }
            _ => Err(CompileError::Lower(
                "projection from an unsupported expression".into(),
            )),
        },
        other => Err(CompileError::Lower(format!(
            "cannot index expression {other:?}; only module inputs and \
             top-level let results can be indexed"
        ))),
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use crate::{
        ir::Module,
        kernel_ir::{Kernel, KernelProgram, SSAOpCode},
        passes::{canonicalize, lower_to_kir, plan_global_scratch, type_infer},
    };

    /// Runs the HIR passes: type inference, canonicalization, scratch
    /// planning and lowering.
    pub(crate) fn lowered(module: Module) -> KernelProgram {
        let types = type_infer(&module).unwrap();
        let program = canonicalize(module, types).unwrap();
        let scratch = plan_global_scratch(&program).unwrap();
        lower_to_kir(&program, &scratch).unwrap()
    }

    pub(crate) fn stmt_kinds(kernel: &Kernel) -> Vec<&'static str> {
        kernel
            .grid
            .block
            .body
            .iter()
            .map(|&id| match kernel.op(id).opcode {
                SSAOpCode::Alloc { .. } => "alloc",
                SSAOpCode::Par { .. } => "par",
                SSAOpCode::Loop { .. } => "loop",
                SSAOpCode::Sync => "sync",
                SSAOpCode::ConvertLayout { .. } => "convert",
                _ => "scalar",
            })
            .collect()
    }
}
