# Cryptography GPU compiler

The goal of this compiler is to lower a light functional DSL that can express the vast majority of compute kernels in zkVMs with high performance on GPUs and low verbosity, with special optimizations for large fields like Bn254, and generic optimizations like fusion, tiling and layout management.

To this end, the compiler supports two key primitives:

```
compute [N] |i| {
  e(i)
}
```
and 
```
reduce [N] |i| {
  e(i)
}
```

that is the functional parallel for and associative reduction (more ops can be added in the future like scan). The language is pure and functional, everything is an expression operating on either scalar values or tensor values. 

`comppute` is semantically like a parallel map, if `e(i)` has type `T`, then `compute [N] |i| { e(i) }` has type `T[N]`, if `e(i)` has type `T[M]`, then `compute [N] |i| { e(i) }` has type `T[N, M]`.

`reduce` is semantically a parallel associative reduction, it preserves the type of `e(i)`


Aside from these primitives, there's the `if cond then e else e'` if guard, the indexing expression `A[i, j, k, ...]`, and standard elementwise operations like `add/sub/div/mul/type_cast`, etc., and the tuple expression `(a, ...)`. A compute with iteration bound `N` whose inner expression has tuple type `(T1, T2[N], T3[K, M], ...)` will have type `(T1[N], T2[N, N], T3[N, K, M], ...)`

For example, the standard matrix multiplication can be expressed as 
```
compute [N] |i| {
  compute [M] |j| {
    reduce [K] |k| {
      A[i, k] * B[k, j]
    }
  }
}
```

or equivalently:
```
compute [N * M] |i| {
  reduce [K] |k| {
    A[i / M, k] * B[k, i % M]
  }
}
```

except the first version has type `T[N, M]` and the second has type `T[N * M]`. All the shapes and iteration bounds here are constants.

To support multiple kernels or more complex programs, there's the let primitive that lets you bind expressions to variables.

```
let v = compute [N] |i| { ... } in 
let b = reduce [N] |i| { v[i] }
```

Finally, a complete program is a kernel module that declares the list of inputs, their types, and the expression that represents the entire sequence of computations.

```
kernel_mod(A: BB[512], B: BB[512]) {
  let a = ... in 
  ...
}
```

the job of the compiler is to lower this kernel module to a C++ file that exports the following interface:

```
extern "C" Prog* make_module();
extern "C" uint64_t scratch_size(Prog*);

extern "C" uint64_t num_outputs(Prog*);
extern "C" uint64_t output_size(Prog*, uint64_t);

extern "C" uint64_t num_inputs(Prog*);
extern "C" uint64_t input_size(Prog*, uint64_t);

extern "C" void set_input(Prog*, uint64_t, void*);
extern "C" void set_output(Prog*, uint64_t, void*);

extern "C" void set_scratch_buf(Prog*, void*);

extern "C" cudaError_t run(Prog*, cudaStream_t);
```

The job of the compiler is as follows:
1. canonicalize the expressions to the following canonical form: on the top-level is a sequence of let expressions, each let binding on the rhs is either a scalar expression or a compute. Each compute has a maximum of 1 compute nested within, or a reduce. For example:

the following top-level expressions are legal (as the rhs of a let bind):
```
compute [N] |i| { A[i] }

compute [N] |i| { 
  let a = compute [M] |j| { A[i, j] * 2 } in 
  let b = reduce [M] |j| { a[j] } in 
  b
}


```

the following are not legal:
```
compute [N] |i| {
  compute [N] |j| {
    compute [M] |k| {
      ...
    }
  }
}


compute [N] |i| {
  reduce [N] |j| {
    compute [M] |k| {
      ...
    }
  }
}
```

but it is always possible to canonicalize any arbitrary nesting into canonical form. 

Then after canonicalization in the outer scope we have an ordered sequence of let binds:
```
let (x1, x2, ...) = compute [N1] { ... } in 
let (y1, ...) = compute [N2] { ... } in 
...
```
and each `compute` is interpreted as a kernel, the outputs are arrays in global memory. Meanwhile the inner computes in the outer compute is interpreted as a thread block. Ie. the outer compute is parallel over the grid and the inner is parallel over the threads. 

At this point the compiler is free perform tiling and fusion and then to lower to a lower level representation, the KernelIR: each tensor gets annotated with a data layout, each compute gets annotated with a compute layout and each reduce is lowered to a sequence of computes and `for` loops. 

There are 3 kinds of data layouts: the register layout, the shared memory layout and the global memory layout. Shared and register layouts map logical indices into physical resource ids, for the register layout it maps logical indices into the local id, the lane id and the warp id. Global layouts are affine expressions that map logical indices to physical indices. 
Shared and register buffers are always powers of 2 and use triton's linear layout system: https://arxiv.org/html/2505.23819v1.

Compute layouts map physical compute ids into logical indices. This describes a factorization of the logical iteration domain into parts iterated over sequentially exploiting ILP and parts exploiting thread level parallelism. For example, `compute [N] |i| { e(i) }` can be factored into 
```
for T |i| {
  e(perm(i, threadIdx.x)) 
}
```
where `blockIdx.x == N / T` and `perm` is a bijection between `(Z_T, Z_{N/T})` and `Z_N`. For all inner computes over threads we can assume the bounds are powers of 2 and the bijection is anything describable via a linear layout, ie. a linear map `T: Z_2^k -> Z_2^k` between the indices represented as vectors in `Z_2`. 
For outer computes we cannot assume the buffers are powers of 2 sized, so we must allow a more general layout where expressions are polyhedral quasi-affine expressions mapping logical to physical indices.

The lower level IR must support a layout conversion op, and optimization passes to eliminate layout conversions. It must also deal with memory more explicitly: buffers are annotated with what address space they belong to. Buffers nested within a compute can belong to register, shared or global address space, while outer buffers can only belong to global. The lower level IR must also expression mutation semantics: in contrast to the higher level purely functional semantics, the lower level IR allows mutations to happen. 

After the compiler has layout information on all data buffers, all computes are annotated with compute layout information, and all `reduce`s are lowered to a combination of `for` and `compute`, the compiler can begin planning shared memory usage within each kernel, and global memory usage across kernels, insertion of synchronization primitives, and finally code generation. 

The compiler generates code in CUDA C++, which exports a C interface that can be DL-opened in Rust. In Rust for now the compiler exports the public IRBuilder through which users can build the higher level functional IR, as well as APIs for controlling codegen and exporting a `KernelModule` that wraps around the DL-opened dynamic library.



