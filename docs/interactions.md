# AIR Interactions

We describe a framework for imposing constraints that span multiple AIRs or span non-adjacent rows in a single AIR.
This framework allows an AIR designer to define "messages" and "multiplicities" determined by the trace data over a
given "bus". The core protocol enforces that every bus carrying these messages is "balanced": the total multiplicity of
every distinct message sent on this bus is zero.

Below is the formal description, followed by the three main use cases and the assumptions needed for soundness.
Proofs with the precise soundness guarantees justifying the below can be found [here](Soundness_of_Interactions_via_LogUp.pdf).

## Formal Definition of an Interaction

An interaction with **width** $w$ and **message length** $\ell$ on bus $b$ is a triple $(\sigma, m, b)$, where:

- $\sigma \in \mathbb{F}[x_1, \dots, x_w, y_1, \dots, y_w]^\ell$ is a sequence of $\ell$ polynomials defining the **message**.
- $m \in \mathbb{F}[x_1, \dots, x_w, y_1, \dots, y_w]$ is a polynomial that determines the **multiplicity** of the corresponding message.
- $b \in \mathbb{F} \setminus \{0\}$ is the **bus index** specifying the bus. It must be nonzero.

If the trace domain is $\langle \omega \rangle$ and the entry in row $i$ column $j$ of the trace matrix $\mathbf{T}$ is
given by $T_j(\omega^i)$, then an interaction $(\sigma, m, b)$ defined on the AIR **sends** over bus $b$, for each
row $i$, the message

$$
\sigma(T_1(\omega^i), \dots, T_w(\omega^i), T_1(\omega^{i+1}), \dots, T_w(\omega^{i+1})) \in \mathbb{F}^\ell
$$

with multiplicity

$$
m(T_1(\omega^i), \dots, T_w(\omega^i), T_1(\omega^{i+1}), \dots, T_w(\omega^{i+1})) \in \mathbb{F}.
$$

Each AIR of width $w$ can define multiple interactions of the same width $w$ (and each interaction can use a different
message length and/or bus index). The combination of an AIR's definition and the prover-provided trace for that AIR
determines a multiset of messages sent, one for each distinct bus index.

## $\mathbb{F}$-Multiset Balancing

Consider a circuit with $t$ AIRs of widths $w_1, \dots, w_t$, where the $i$-th AIR has $k_i$ interactions
$(\sigma^{(i)}_k, m^{(i)}_k, b^{(i)}_k)$ for $k \in \{ 1, \dots, k_i \}$.

The set of possible messages is denoted $\mathbb{F}^+ = \bigcup_{i\ge 1} \mathbb{F}^i$. An **$\mathbb{F}$-multiset**
is a function $M : \mathbb{F}^+ \to \mathbb{F}$ that assigns an $\mathbb{F}$-valued "multiplicity" to each message
$\mathbb{F}^+$.

Given a set of traces ${\mathbf T}^{(1)}, \dots, {\mathbf T}^{(t)}$ with respective
domains $\langle \omega_1 \rangle, \dots, \langle \omega_t \rangle$, these traces together define a multiset $M_b$ for
each bus index $b$. To simplify notation, for $i \in \{1, \dots, t\}$ and $k \in \{1, \dots, k_t\}$,
define $\hat{m}^{(i)}_k : \langle \omega_i \rangle \to \mathbb{F}$ by:

$$
\hat{m}^{(i)}_k(x) = m^{(i)}_k(T^{(i)}_1(x), \dots, T^{(i)}_w(x), T^{(i)}_1(\omega x), \dots, T^{(i)}_w(\omega x))
$$

and define $\hat{\sigma}^{(i)}_k$ analogously. The multiset defined by the traces is then given by:

$$
M_b(\sigma) = \sum_{i=1}^{t} \sum_{k=1}^{k_i} \sum_{x \in \langle \omega_i \rangle}
\hat{m}^{(i)}_k(x) \mathbf{1}(b = b^{(i)}_k \wedge \sigma = \sigma^{(i)}_k)
$$

We say that a bus $b$ is **balanced** if the $\mathbb{F}$-multiset $M_b$ satisfies $M_b(\tau) = 0$ for all messages
$\tau \in \mathbb{F}^+$.

### Soundness Statement

If there exists a bus $b$ whose net message multiplicities fail to sum to zero (i.e., fail the balancing condition),
then the verifier will reject with high probability (in the random oracle model).

## Main Use Cases

Many constraints in the context of AIRs can be phrased in terms of balancing a multiset of messages:

1. $\mathbb{F}$-multiset: We want to add elements to a multiset with multiplicities. The guarantee is that the
   multiplicities all sum to zero for each message.
2. Lookup tables: we want to "look up" certain values from a table. The guarantee is that all "requested" lookups indeed
   appear in the table.
3. Permutation checks: We want to prove that two multisets of messages _with integer multiplicities_ (e.g., the "send"
   multiset and the "receive" multiset) form the same multiset.

Note that (1) is explained above, while (2) and (3) can be implemented using (1), provided some additional assumptions
hold. We describe the reductions below and what the additional assumptions are.

### Lookup Tables

A **lookup table** is an AIR that (conceptually) enumerates all valid elements of some set, such as the integers in a
range $[a,b]$. Another AIR wants to check that some columns’ (or expressions derived from the columns) values always
appear in that table. We use the interaction framework to do this:

1. **Lookup requests**: The AIR (or multiple AIRs) that want to ensure a value $\sigma$ is in the table sends an
   interaction with multiplicity $1$ for that value.
2. **Lookup table**: The table itself sends an interaction with multiplicity equal to the negative sum of all requests
   for that value. (Note that this multiplicity is unconstrained and prover-provided.)

#### Soundness assumptions

- **Characteristic bounds**: The number of times a value is looked up must be strictly less than the field characteristic.
- **Single lookup table**: Only one AIR is allowed to play the role of providing the negative “all table entries” side for a given bus index.

#### Verifier's conclusion

The **single lookup table** assumption must be checked by inspection of the verifying key; failure to do so may result
in an unsound circuit. The characteristic bounds cannot be known statically, since the trace heights are prover-provided
values. However, the verifier knows upper bounds on the number of lookups performed per bus per AIR statically (i.e.,
put in some form into the verifying key). When verifying a proof, the verifier takes the dot product of these values
with the heights of each matrix and checks if they overflow the field characteristic.

Provided the soundness assumptions above hold, if the bus is balanced, the verifier concludes:

    Every "lookup request" value σ was matched by the (single) table's negative multiplicities, so σ is indeed in the table.

### Permutation Checks

In a permutation check, we want to prove that two global multisets of messages _with integer multiplicities_
—collectively defined by all participating AIRs and a given bus index—are identical. We call the two multisets the
"send" multiset and the "receive" multiset.

An AIR specifies which messages to add the "send" multiset or to the "receive" multiset along with some small
multiplicity (typically one) as functions of its trace matrix, and the goal is to constrain that the number of times any
messages appears in the "send" multiset is equal to the number of times it appears in the "receive" multiset. We again
emphasize that the goal is to enforce a constraint over _integer_ multiplicities.

We can use the $\mathbb{F}$-multiset guarantee of interactions to implement integer multiset requirement, provided we
do not overflow the field characteristic for any message. Adding to the "send" multiset means sending an interaction
with multiplicity 1, while adding to the "receive" multiset means sending an interaction with multiplicity -1. This can
be generalized to small multiplicities (i.e., it's okay to send a message with multiplicity 2, which would be slightly
more performant than sending the message twice), but one cannot, e.g., add a message to the "send" or "receive"
multisets with multiplicity $(p + 1)/2$.

The above constraint comes from how sends and receives are implemented using the interaction framework. Adding to the
"send" multiset corresponds to sending a message with the given multiplicity. Adding to the "receive" multiset
corresponds to sending a message with the _negative_ multiplicity. In other words, interaction
multiplicities $\{1, \dots, (p - 1)/2\}$ correspond to the "send" multiset while interaction
multiplicities $\{(p - 1)/2 + 1, \dots, p - 1 \}$ correspond to "receive" multiplicities.

#### Soundness assumptions

- **Characteristic bounds**: The number of times a message is added to the "send" multiset and the number of times a
  message is added to the "receive" multiset is (strictly) less than the field characteristic.

#### Verifier's conclusion

Similar to the lookup table case, the verifier can check a sufficient condition for the **characteristic bound**
assumption by taking the dot products of some statically computed interaction counts with the prover-provided trace
heights.

Provided the soundness assumptions above hold, if the bus is balanced, the verifier concludes:

    The “send” multiset equals the “receive” multiset in an integer sense.

## Interaction API

The lowest-level interface is controlled by the trait [`InteractionBuilder`](./mod.rs)

```rust
pub trait InteractionBuilder: AirBuilder {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_index: BusIndex,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        count_weight: u32,
   );
}
```

The `InteractionBuilder` trait is an extension of `AirBuilder`. You should use
`impl<AB: InteractionBuilder> Air<AB> for MyAir` to enable usage of the above API within the `Air::eval` function. For a
given AIR, the interface allows to specify adding messages with a given multiplicity to the multiset defined on the `bus`,
defined by its `identifier`. The interaction specifies `message` $(\sigma_i)$ and `count` $m$ where each $\sigma_i$
and $m$ is a polynomial expression on the main and preprocessed trace polynomials with rotations. This means that we
want to send the tuple $(\sigma_1(\mathbf T),\dotsc,\sigma_{\ell}(\mathbf T))$ to the $i$-th bus with
multiplicity $m(\mathbf T)$, where $\mathbf T$ refers to the trace (including preprocessed columns) as polynomials (as
well as rotations).

The `InteractionBuilder` keeps track of all interactions pushed and their corresponding buses. The buses used must be
consistent, or keygen will fail. More specifically, if `push_interaction` is called twice with two buses that have the
same bus index, the constraint type must also match.

The quantity `count_weight` must be set correctly for the interactions to be sound. See
the [Ensuring Interaction Soundness](#ensuring-interaction-soundness) section below. If using the standard `LookupBus`
and `PermutationCheckBus`, this quantity is set to `1`.

### LookupBus

For the common case of lookup tables, we provided a more tailored API. To define a lookup bus, one instantiates the
`LookupBus` struct. This struct provides methods `perform_fields_lookup`, for AIRs that wish to perform a lookup, and
`add_fields` for the _single_ AIR that serves as the lookup table. These methods are light wrappers around
`push_interaction`.

```rust
pub struct LookupBus {
    pub index: BusIndex,
}

impl LookupBus {
    /// Performs a lookup on the given bus.
    ///
    /// This method asserts that `key` is present in the lookup table. The parameter `enabled`
    /// must be constrained to be boolean, and the lookup constraint is imposed provided `enabled`
    /// is one.
    ///
    /// Caller must constrain that `enabled` is boolean.
    pub fn lookup_key<AB, E>(
        &self,
        builder: &mut AB,
        query: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    )
    where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // We embed the query multiplicity as {0, 1} in the integers and the lookup table key
        // multiplicity to be {0, -1, ..., -p + 1}. Setting `count_weight = 1` will ensure that the
        // total number of lookups is at most p, which is sufficient to establish lookup multiset is
        // a subset of the key multiset. See Corollary 3.6 in [docs/Soundess_of_Interactions_via_LogUp.pdf].
        builder.push_interaction(self.index, query, enabled, 1);
    }

    /// Adds a key to the lookup table.
    ///
    /// The `num_lookups` parameter should equal the number of enabled lookups performed.
    pub fn add_key_with_lookups<AB, E>(
        &self,
        builder: &mut AB,
        key: impl IntoIterator<Item = E>,
        num_lookups: impl Into<AB::Expr>,
    )
    where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // Since we only want a subset constraint, `count_weight` can be zero here. See the comment
        // in `LookupBus::lookup_key`.
        builder.push_interaction(self.index, key, -num_lookups.into(), 0);
    }
}
```

### Permutation Checks

Similarly to lookup tables, we also provide a `PermutationCheckBus` struct with a more direct interface for building the
send and receive multisets mentioned to above.

```rust
pub struct PermutationCheckBus {
    pub index: BusIndex,
}

impl PermutationCheckBus {
    /// Send a message.
    ///
    /// Caller must constrain `enabled` to be boolean.
    pub fn send<AB, E>(
        &self,
        builder: &mut AB,
        message: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // We embed the multiplicity `enabled` as an integer {0, 1}.
        builder.push_interaction(self.index, message, enabled, 1);
    }

    /// Receive a message.
    ///
    /// Caller must constrain `enabled` to be boolean.
    pub fn receive<AB, E>(
        &self,
        builder: &mut AB,
        message: impl IntoIterator<Item = E>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        E: Into<AB::Expr>,
    {
        // We embed the multiplicity `enabled` as an integer {0, -1}.
        builder.push_interaction(self.index, message, -enabled.into(), 1);
    }
}
```

## Trace Height Constraints for Interaction Soundness

The soundness of LogUp depends on that the total number of interactions is not too large. Also, as mentioned earlier,
for lookup tables and permutation checks to be sound, we must also ensure that no specific message is sent too many
times (otherwise we would not be able to distinguish multiplicity 0 from multiplicity p, for example). Since these
quantities are determined by the trace heights, and the trace heights are determined by the prover, we must provide a
mechanism to convince the verifier that the necessary quantities are in bound.

To achieve this, we hard-code linear constraints on the trace heights into the verifier.
Let $F = \{0, \dots, p - 1\} \subseteq \mathbb{Z}$. Let $x \in F^n$ be the vector of trace heights. For a fixed matrix
$A \in F^{m \times n}$ and vector $b \in F^n$ determined by the AIRs and their constraints, the verifier checks
that $Ax \le b$ over the integers, where "$\le$" denotes component-wise less-than-or-equal.

### Per-Bus Trace Height Constraints

The matrix $A$ and threshold vector $b$ are determined as follows. For each bus, we add a constraint $i$ (i.e., a row to
matrix $A$ and a value to vector $b$) where $a_{ij}$ is the sum of the `count_weight` of all interactions on this bus
on AIR $i$. The threshold $b_i$ is set to the field characteristic $p$. For the lookup bus and permutation bus,
the value we set for `count_weight` ensures that this linear constraint in the heights is sufficient to guarantee the
soundness condition.

### Total Interactions Trace Height Constraints

We also add another constraint related to the bits of soundness for the LogUp procedure. For this, we set the trace
height coefficients to be the number of interactions on the corresponding AIR and the threshold to be $p$. This
allows us to claim a certain number of bits of security. See [here](Soundness_of_Interactions_via_LogUp.pdf) for more
details.

## Backend implementation via logUp

The backend implementation of the prover will constrain the computation of a cumulative sum _for just a single AIR_ with
trace domain $\langle \omega \rangle$ and interactions $J$:

$$
\sum_{x \in \langle \omega \rangle} \sum_{(\sigma, m, b)} \frac{\hat{m}(x)}{\alpha + h_{\beta}(\hat{\sigma}(x) \circ b)}
$$

where:

- $\alpha$, $\beta$ are two challenge extension field elements,
- $h_{\beta}(\tau) = \sum_{j \ge 1} \beta^{j-1} \tau_j$ is a random linear combination (RLC) of any message $\tau = (\tau_1,\dotsc,\tau_\ell) \in \mathbb{F}^+$,
- $\circ$ denotes concatenation in $\mathbb{F}^+$, with $b \in \mathbb{F}^1$ a singleton.

Note that the bus index being nonzero and concatenated to the end of the message ensures that messages on different
buses are mapped to distinct preimages of $h_\beta$. It also distinguishes between two different length messages with
trailing zeroes.

Globally, the verifier will sum this per-AIR cumulative sum over all AIRs and lastly constrain that the sum is $0$. This
will enforce that the sends and receives are balanced globally across all AIRs.

### Virtual columns and constraints

The $\sigma_j, m$ can be any multivariate polynomial expression, which is expressed via the `AB::Expr` type within the
`Air::eval` function.

For each interaction $\mu = (\sigma, m, b)$, we add one virtual column $q_\mu$ with row $x$ equal to

$$
q_\mu(x) = \frac{m}{\alpha + h_{\beta}(\hat{\sigma}(x) \circ b)}
$$

The constraint is

$$
q_\mu(x) \cdot \left(\alpha + h_{\beta}(\hat{\sigma}(x) \circ b) \right) = \hat{m}(x),
$$

which has degree $\max(1 + \max_j \deg(\sigma_j), \deg(m))$.

Note: To optimize column usage, we sometimes combine several $q$ columns, though this increases the constraint degree.

We need one more virtual column $\phi$ for the cumulative sum of all multiplicities. The row $x$ of $\phi$ contains the
partial sum of all reciprocals up to row $x$.

$$
\phi(x) = \sum_{y \leq x} \left(\sum_\mu q_\mu(y)\right)
$$

The constraints are:

- $sel_{first} \cdot \phi = sel_{first} \cdot \sum_\mu q_\mu$
- $sel_{transition} \cdot (\phi' - \phi) = sel_{transition} \cdot \sum_\mu sign(\mu) q_\mu'$ where $\phi'$ and $q'$ mean the next row (rotation by $1$).
- $sel_{last} \cdot \phi = sum$

where $sum$ is exposed to the verifier.

In summary, we need 1 additional virtual column for each multiplicities, and 1 additional virtual column to track the
partial sum. These columns are all virtual in the sense that they are only materialized by the prover after the main
trace has been committed, since a random challenge is needed.
