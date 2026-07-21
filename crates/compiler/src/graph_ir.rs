//! Graph IR: a raw computation graph over a GPU device.
//!
//! Nodes are kernel launches, memcpys and memsets operating on buffers
//! identified by [`BufId`]; buffers carry their device placement and sizes
//! in [`BufInfo`]. See `graph-ir.md`.

use std::fmt;

/// Index of a buffer in the graph's buffer table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceType {
    /// CUDA device with the given ordinal.
    Cuda(usize),
    CpuPinned,
    CpuPaged,
}

#[derive(Clone, Debug)]
pub struct BufInfo {
    pub name: Option<String>,
    pub device_type: DeviceType,
    /// Size in bytes.
    pub size: usize,
    pub elem_size: usize,
}

/// Type-erased kernel launch: receives raw pointers to the input buffers
/// and to the output buffers, in that order.
pub type KernelFn = Box<dyn Fn(&[*mut ()], &[*mut ()])>;

pub struct KernelNode {
    pub inputs: Vec<BufId>,
    pub outputs: Vec<BufId>,
    /// Parallel to `inputs`: whether the kernel also writes the buffer
    /// in place (read-write dependency rather than read-only).
    pub modifies: Vec<bool>,
    pub func: KernelFn,
    pub name: String,
}

impl fmt::Debug for KernelNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelNode")
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("modifies", &self.modifies)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MemcpyNode {
    pub src: BufId,
    pub dst: BufId,
}

#[derive(Copy, Clone, Debug)]
pub struct MemSetNode {
    pub node: BufId,
    pub val: u32,
}

#[derive(Debug)]
pub enum GraphNode {
    Kernel(KernelNode),
    Memcpy(MemcpyNode),
    Memset(MemSetNode),
}

#[derive(Default)]
pub struct GraphBuilder {
    pub bufs: Vec<BufInfo>,
    pub nodes: Vec<GraphNode>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_buf(&mut self, info: BufInfo) -> BufId {
        let id = BufId(self.bufs.len());
        self.bufs.push(info);
        id
    }

    pub fn buf_info(&self, id: BufId) -> &BufInfo {
        &self.bufs[id.0]
    }

    pub fn insert_kernel(
        &mut self,
        name: impl Into<String>,
        inputs: impl Iterator<Item = BufId>,
        outputs: impl Iterator<Item = BufId>,
        modifies: impl Iterator<Item = bool>,
        f: impl Fn(&[*mut ()], &[*mut ()]) + 'static,
    ) {
        let inputs: Vec<_> = inputs.collect();
        let modifies: Vec<_> = modifies.collect();
        assert_eq!(
            inputs.len(),
            modifies.len(),
            "modifies must have one flag per input"
        );
        self.nodes.push(GraphNode::Kernel(KernelNode {
            inputs,
            outputs: outputs.collect(),
            modifies,
            func: Box::new(f),
            name: name.into(),
        }));
    }

    pub fn insert_memcpy(&mut self, src: BufId, dst: BufId) {
        self.nodes.push(GraphNode::Memcpy(MemcpyNode { src, dst }));
    }

    pub fn insert_memset(&mut self, node: BufId, val: u32) {
        self.nodes.push(GraphNode::Memset(MemSetNode { node, val }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(builder: &mut GraphBuilder, name: &str, device_type: DeviceType) -> BufId {
        builder.add_buf(BufInfo {
            name: Some(name.to_string()),
            device_type,
            size: 1024,
            elem_size: 4,
        })
    }

    #[test]
    fn builds_graph() {
        let mut b = GraphBuilder::new();
        let host = buf(&mut b, "host", DeviceType::CpuPinned);
        let x = buf(&mut b, "x", DeviceType::Cuda(0));
        let y = buf(&mut b, "y", DeviceType::Cuda(0));

        b.insert_memcpy(host, x);
        b.insert_memset(y, 0);
        b.insert_kernel(
            "add",
            [x].into_iter(),
            [y].into_iter(),
            [false].into_iter(),
            |_inputs, _outputs| {},
        );

        assert_eq!(b.bufs.len(), 3);
        assert_eq!(b.buf_info(x).name.as_deref(), Some("x"));
        assert_eq!(b.nodes.len(), 3);
        assert!(matches!(
            &b.nodes[0],
            GraphNode::Memcpy(MemcpyNode { src, dst }) if *src == host && *dst == x
        ));
        assert!(matches!(
            &b.nodes[1],
            GraphNode::Memset(MemSetNode { node, val: 0 }) if *node == y
        ));
        match &b.nodes[2] {
            GraphNode::Kernel(k) => {
                assert_eq!(k.name, "add");
                assert_eq!(k.inputs, vec![x]);
                assert_eq!(k.outputs, vec![y]);
                assert_eq!(k.modifies, vec![false]);
                (k.func)(&[], &[]);
            }
            other => panic!("expected kernel node, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "modifies must have one flag per input")]
    fn kernel_modifies_length_mismatch_panics() {
        let mut b = GraphBuilder::new();
        let x = buf(&mut b, "x", DeviceType::Cuda(0));
        b.insert_kernel(
            "bad",
            [x].into_iter(),
            [x].into_iter(),
            [].into_iter(),
            |_, _| {},
        );
    }
}
