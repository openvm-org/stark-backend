We need to compute the evaluations of the quotient polynomial on the quotient domain.
This seems to (unavoidably?) require evaluating the constraints on each point of the domain.

Evaluation happens by taking the evaluations of the trace polynomials on the quotient domain
(this is nearly free if you have the LDE polynomials from the trace commitment by doing some bit reversal permutation).
For later purposes, the evaluations need to be grouped into "chunks", where each chunk is vertically strided by `quotient_degree`
(which is approximately constraint degree minus 1).
For SIMD, we want to simultaneously evaluate a SIMD `WIDTH` number of rows of the final chunk at once.
This is done by first collecting the required row values into SIMD `PackedValue`s, forming a new "fat" row
(or two fat rows when adjacent next row is necessary).

Once these fat rows are formed, we essentially just use Rayon parallel iterators to evaluation constraints on each fat row.
We (`stark-backend`) do this by storing the constraints as a direct acyclic graph of symbolic expression nodes.
Then evaluation is an **interpreter** of the DAG which does traversal of the DAG, which is already in topologically sorted order.

Optimizations to try:
1. ✅ Does not using `par_iter` and using more explicit multi-threading have better cache/memory properties? Done: see `parallelize_chunks`
2. avoid per-iter allocations of the fat row, also the interpreter also allocates right now
3. Getting rid of interpreter overhead
  3a. Use function pointers to avoid match statements
  3b. Pre-alloc memory for each node value: the "type" of each node is determined by the DAG so we can unsafely transmute
  3c. Optimize the graph so that nodes that are only used by one "child" don't need intermediate storage in memory (can go directly in registers)

On complicated constraints, the interpreter has major overhead. To compare,
`plonky3`'s approach is that the constraint evaluation is stored directly in the `Air` trait. This way the compiler knows the exact graph shape and can optimize evaluation for it. The downside is it makes serialization of the vkey hard and also there is some simplification that happens in the conversion from `Air` to DAG already.
The best of both worlds would be to AOT compile the graph evaluator for a given vkey into a special evaluation function for that AIR, and dynamically (or statically)
link the function into the quotient value computation.
We will benchmark via the keccak-f AIR to compare these approaches.
