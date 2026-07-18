#[derive(Copy, Clone, Debug)]
pub struct ValId(usize);

#[derive(Copy, Clone, Debug)]
pub struct NodeId(usize);

pub enum ScalarType {
    BabyBear,
    U32,
    I32,
    U64,
    I64,
    Bool,
}

pub struct Array {
    tt: ScalarType,
    shape: Vec<usize>,
}

pub enum TType {
    Scalar(ScalarType),
    Array(Array),
}

enum KernelExpr {
    Let {
        bind: Vec<ValId>,
        expr: NodeId,
        inexpr: NodeId,
    },
    Compute {
        n: usize,
        i: ValId,
        expr: NodeId,
    },
    Reduce {
        n: usize,
        i: ValId,
        expr: NodeId,
    },
    Tuple {
        nodes: NodeId,
    },

    Index(ValId, ValId),
    // TODO: other ops
    Add(ValId, ValId),
    Mul(ValId, ValId),
    Ident(ValId),
}

struct IRBuilder {}

impl IRBuilder {
    fn type_of(&self, id: ValId) -> TType {
        todo!()
    }
    fn compute(&self, n: usize, f: impl FnMut(ValId) -> NodeId) -> NodeId {
        todo!()
    }
    fn reduce(&self, n: usize, f: impl FnMut(ValId) -> NodeId) -> NodeId {
        todo!()
    }

    fn tuple(&self, nodes: impl Iterator<Item = NodeId>) -> NodeId {
        todo!()
    }

    fn let_expr(&self, node: NodeId, expr: impl FnMut(Vec<ValId>) -> NodeId) -> NodeId {
        todo!()
    }

    fn add(&self, a: ValId, b: ValId) -> NodeId {
        todo!()
    }
    // TODO: other ops
}
