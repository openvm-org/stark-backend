# STARK Backend

The backend is a low-level API built on top of Plonky3. Its goal is to prove multiple STARKs presented in the form of AIRs, interactions, and their trace matrices.
Trace matrices are supplied by the caller; the backend commits to them and proves their AIR and interaction constraints.

The backend supports interactions for synchronizing elements between different AIRs (see below). [This](https://hackmd.io/@aztec-network/plonk-arithmetiization-air) can be used as a reference for the definition of AIRs and related concepts.

## Table of Contents

1. [Definitions](#definitions)
2. [Traces as polynomials](#traces-as-polynomials)
3. [The Protocol](#the-protocol)
4. [Proving Key](#proving-key)
5. [Verifying Key](#verifying-key)

## Definitions

Everywhere below we assume that all elements we are dealing with are in a field $\mathbb{F}$ or its extension fields.

### AIRs

The following is a vague definition of an AIR; there is a [more general definition](https://eprint.iacr.org/2021/582.pdf), which is not fully supported by this backend and has some parts which overload the definition and will be assumed implicitly whenever needed.

An **AIR** is a set of pairs $(C_i, H_i)$, where $C_i\colon \mathbb{F}^{2w} \to \mathbb{F}$ is a _constraint polynomial_. Here $w$ is the width associated with the AIR; the trace matrix of the AIR must be of width $w$. $H_i$ can be one of `All`, `First`, `Last`, `Transition`. If $(x_1, \ldots, x_w, y_1, \ldots, y_w)$ are two cyclically consecutive rows of the trace matrix, then each such pair constrains $C_i(x_1, \ldots, x_w, y_1, \ldots, y_w)$ to be zero; with the following domains:
- If $H_i$ is `All`, then this applies to all pairs of cyclically consecutive rows.
- If $H_i$ is `First`, then this applies to the first pair of cyclically consecutive rows.
- If $H_i$ is `Last`, then this applies to the pair (last row, first row).
- If $H_i$ is `Transition`, then this applies to all pairs of non-cyclically consecutive rows (that is, like `All` but except (last, first)).

In general it is possible to support more domains, but our backend supports only the above.

An AIR only supports trace matrices whose height is a power of two (or zero-height matrices, which get skipped).

### Trace Parts

The trace is supplied to the backend in the following parts:
- `preprocessed` is a fixed part of the trace that is preprocessed and fixed at keygen time. It can be omitted. These columns do not depend on the inputs. If an AIR has preprocessed trace, its height is fixed at keygen time. If an AIR does not have preprocessed trace, the trace height can vary at proof time but must be a power of two.
- `main` is the main part of the trace that is processed at proof time. We partition it into _two_ parts: "common main", which all have the common commitment, and "cached main", which are committed separately.

### Interactions

Interactions are a set of pairs $(\mathrm{bus}, I_j, M_j)$, where $\mathrm{bus}\in\mathbb{F}\setminus\{0\}$ is a bus index, $I_j\in\mathbb{F}[x_1, \ldots, x_w, y_1, \ldots, y_w]^{\ell}$ is a sequence of polynomials, and $M_j\in\mathbb{F}[x_1, \ldots, x_w, y_1, \ldots, y_w]$ is a multiplicity polynomial. The constraints are that for every different pair $(\mathrm{bus}, I_j(x_1, \ldots, x_w, y_1, \ldots, y_w))$ the sum of all multiplicities $M_j(x_1, \ldots, x_w, y_1, \ldots, y_w)$ over all AIRs must be zero. See [Interactions API](./interactions.md) for more details.

Interaction expressions are evaluated as LogUp input fractions during proving
and are checked by the GKR fractional sumcheck plus the batched constraint
sumcheck. This interaction handling does not materialize additional trace
columns.

## Traces as polynomials

Let $\ell$ be the univariate skip parameter, and let $D$ be the univariate skip
domain of size $2^\ell$, with generator $\omega_D$. For $n \ge 0$, write

$$
\mathbb{D}_n = D \times H_n,
$$

where $H_n = \{0, 1\}^n$; these domains are the hyperprisms. Extend this
notation to $n = -i < 0$ by setting

$$
\mathbb{D}_{-i} = D^{(2^i)} = \{x^{2^i} \mid x \in D\},
$$

using generator $\omega_D^{2^i}$. Thus $|\mathbb{D}_n| = 2^{\ell+n}$ for every
$n \ge -\ell$.

For a trace of height $2^h$, let $n = h - \ell$ be the relative dimension with
respect to $\ell$. A trace column is identified with a function

$$
f\colon \mathbb{D}_n \to \mathbb{F}.
$$

The trace column polynomial $\hat f$ is the prismalinear extension of $f$. When
$n \ge 0$, it has degree less than $2^\ell$ in the $D$ coordinate and is
multilinear in the $H_n$ coordinates. When $n = -i < 0$, $\hat f$ is the
univariate polynomial of degree less than $2^{\ell-i}$ that interpolates $f$ on
$D^{(2^i)}$. In this negative case the protocol also uses the lift
$\tilde f(Z) = \hat f(Z^{2^i})$, a univariate polynomial of degree less than
$2^\ell$ on the full domain $D$.

In the stacked matrix, a trace with $n < 0$ is embedded into the $D$ coordinate
with stride $2^i = 2^{-n}$, and reductions use an indicator polynomial to
ignore the other points of $D$ when needed. The relative dimension is stored in
preprocessed verifier data as `hypercube_dim`.

AIR constraints and interaction expressions are stored symbolically and evaluated over this
representation, including next-row rotations when needed. The batched constraint sumcheck reduces
the AIR zerocheck and LogUp input-consistency claims to opening claims about the trace polynomials.

## The Protocol

### In Summary

The prover entry point is `Coordinator::prove`, which implements the `Prover` trait. The flow is:

1. **Main trace commitment:** commit all present `common_main` traces together with the stacked PCS.
   Cached main commitments are supplied in the proving context.
2. **Transcript observations:** observe the verifying-key prehash, common main commitment, AIR
   presence flags for optional AIRs, preprocessed commitments or trace heights, cached commitments,
   and public values.
3. **LogUp GKR:** prove that the global LogUp fractional sum is zero using GKR fractional sumcheck.
4. **Batch constraint sumcheck:** batch AIR zerocheck constraints with LogUp input consistency and
   reduce them to trace opening claims.
5. **Stacked opening reduction:** reduce trace opening claims, including rotations when needed, to
   opening claims about the stacked matrix polynomials.
6. **WHIR opening proof:** prove the stacked PCS openings with WHIR.

The PCS is the stacked PCS opened with WHIR.

The proof structure is:

```rust
pub struct Proof<SC: StarkProtocolConfig> {
    pub common_main_commit: SC::Digest,
    pub trace_vdata: Vec<Option<TraceVData<SC>>>,
    pub public_values: Vec<Vec<SC::F>>,
    pub gkr_proof: GkrProof<SC>,
    pub batch_constraint_proof: BatchConstraintProof<SC>,
    pub stacking_proof: StackingProof<SC>,
    pub whir_proof: WhirProof<SC>,
}
```

The verifier replays the same transcript order, checks trace-height constraints, verifies the
LogUp/batch-constraint proof, verifies the stacked opening reduction, and then verifies the WHIR
opening proof against the common, preprocessed, and cached commitments.


Below we show, in structure, the contents of the proving key and verifying key.

## Proving Key

The proving key (often represented by the type `StarkProvingKey` and collected into a `MultiStarkProvingKey`) contains, for each AIR:

- **Air Identifier**  
  A human‐readable name (e.g. `air_name`) for display and debugging.

- **Prover-Only Data**  
  - **Preprocessed Trace Data:**  
    Located in the field `preprocessed_data` of type `Option<Arc<StackedPcsData<SC::F, SC::Digest>>>`. This is the stacked PCS data for the fixed preprocessed trace, including the stacked matrix, layout, and Merkle tree.

- **Verifying Key Portion (`vk`)**  
  See below.

## Verifying Key

The verifying key (often represented by the type `StarkVerifyingKey`, and then collected into a `MultiStarkVerifyingKey`) is derived solely from the public portion of the proving key. For each AIR it contains:

- **Preprocessed Commitment (Public View):**  
  If the AIR has a preprocessed trace, the verifier stores a `VerifierSinglePreprocessedData` value containing the commitment, `hypercube_dim`, and `stacking_width`.

- **Public AIR Parameters:**  
  This includes all the static configuration details such as the trace widths and the number of public values. These allow the verifier to know the structure of the proof system and the expected sizes of various objects.

- **Symbolic Constraints:**  
  These constraints define the AIR. They are stored in the form of DAG.
  Symbolic interactions are also stored separately in the verifying key.

- **Constraint Degree, Optionality, and Stacking Information:**  
  The key stores `max_constraint_degree`, `is_required`, and `unused_variables`.
