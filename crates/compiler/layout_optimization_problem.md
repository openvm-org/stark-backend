# The layout inference problem

We have an abstract representation of a compute DAG as a bipartite graph where nodes in set A are resource buffers, nodes in set B are compute nodes, and edges between them are access functions. As well as layout equality constraints between resource buffers.

For example:

```
T = N1 // number of threads (constant)
G = M1 // grid size, constant
// local index: i 
// global index: gid

in: [A, B]
out: [F]

res A: [M1]; // shapes are constant
res B: [M2];
res C: [N2];
res D: [N3];
res E: [N3];
res F: [N4];

compute(N2, in={A[f_1(i)], B[f_2(i)]}, out={C[g_1(i)]})
compute(N3, in={C[f_3(gid, i)], B[f_4(i)]}, out={D[g_2(i)], E[g_3(i)]})
compute(N4, in={E[f_5(i)], D[f_6(i)]}, out={F[g_4(i)]})

assert_equal(B, C)
assert_equal(D, E)

```

Each res has exactly one compute node that writes to it, and the access function that writes to it is bijective, and does not depend on the global index. The read access function may depend on the global index, and may not be injective. 


