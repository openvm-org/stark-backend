//! Quasi-affine expressions (`Quast`) over integer symbols, and the
//! `#[scatter(...)]` maps built from them.
//!
//! A `Quast` is an integer expression built from symbols, constants,
//! addition, negation, multiplication by a constant and floor division by a
//! positive constant. `x % c` is not a node: it is represented as
//! `x - c * floor(x / c)` (see [`Quast::rem_c`]) and recovered as a `%`
//! during emission. Expressions are analyzed together with a map from
//! [`VarId`] to the symbol's bound (`0 <= sym < bound`) when one is known.
//!
//! Simplification normalizes an expression into a linear combination of
//! atoms (symbols and floor divisions), which folds constants, collects like
//! terms and cancels `floor` chains — e.g. the composition
//! `linearize(delinearize(f))` collapses back to `f`.

use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use crate::{
    ir::{NodeId, VarId},
    kernel_ir::LinearLayout,
    passes::type_infer::TypeMap,
    CompileError,
};

fn err(msg: impl Into<String>) -> CompileError {
    CompileError::Quast(msg.into())
}

/// A quasi-affine expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Quast {
    Sym(VarId),
    Const(i64),
    Add(Rc<Quast>, Rc<Quast>),
    /// Multiplication by an integer constant.
    Mul(Rc<Quast>, i64),
    /// Floor division by a positive integer constant.
    FloorDiv(Rc<Quast>, i64),
    Neg(Rc<Quast>),
}

impl Quast {
    pub fn sym(v: VarId) -> Quast {
        Quast::Sym(v)
    }

    pub fn cst(c: i64) -> Quast {
        Quast::Const(c)
    }

    pub fn add(&self, other: &Quast) -> Quast {
        Quast::Add(Rc::new(self.clone()), Rc::new(other.clone()))
    }

    pub fn sub(&self, other: &Quast) -> Quast {
        Quast::Add(Rc::new(self.clone()), Rc::new(other.neg()))
    }

    pub fn neg(&self) -> Quast {
        Quast::Neg(Rc::new(self.clone()))
    }

    pub fn mul_c(&self, c: i64) -> Quast {
        Quast::Mul(Rc::new(self.clone()), c)
    }

    pub fn floordiv(&self, c: i64) -> Quast {
        Quast::FloorDiv(Rc::new(self.clone()), c)
    }

    /// `self % c` as `self - c * floor(self / c)`.
    pub fn rem_c(&self, c: i64) -> Quast {
        self.sub(&self.floordiv(c).mul_c(c))
    }

    /// Evaluates with every symbol bound in `env`. Panics on unbound symbols
    /// or non-positive divisors (programmer error).
    pub fn eval(&self, env: &BTreeMap<VarId, i64>) -> i64 {
        match self {
            Quast::Sym(v) => *env
                .get(v)
                .unwrap_or_else(|| panic!("unbound symbol {v:?} in Quast::eval")),
            Quast::Const(c) => *c,
            Quast::Add(a, b) => a.eval(env) + b.eval(env),
            Quast::Mul(a, c) => a.eval(env) * c,
            Quast::FloorDiv(a, c) => {
                assert!(*c > 0, "FloorDiv divisor must be positive");
                a.eval(env).div_euclid(*c)
            }
            Quast::Neg(a) => -a.eval(env),
        }
    }

    /// Inserts every symbol appearing in the expression into `out`.
    pub fn syms(&self, out: &mut BTreeSet<VarId>) {
        match self {
            Quast::Sym(v) => {
                out.insert(*v);
            }
            Quast::Const(_) => {}
            Quast::Add(a, b) => {
                a.syms(out);
                b.syms(out);
            }
            Quast::Mul(a, _) | Quast::FloorDiv(a, _) | Quast::Neg(a) => a.syms(out),
        }
    }

    /// Replaces symbols per `map` (symbols absent from the map are kept).
    pub fn substitute(&self, map: &BTreeMap<VarId, Quast>) -> Quast {
        match self {
            Quast::Sym(v) => map.get(v).cloned().unwrap_or_else(|| self.clone()),
            Quast::Const(_) => self.clone(),
            Quast::Add(a, b) => Quast::Add(Rc::new(a.substitute(map)), Rc::new(b.substitute(map))),
            Quast::Mul(a, c) => Quast::Mul(Rc::new(a.substitute(map)), *c),
            Quast::FloorDiv(a, c) => Quast::FloorDiv(Rc::new(a.substitute(map)), *c),
            Quast::Neg(a) => Quast::Neg(Rc::new(a.substitute(map))),
        }
    }

    /// Inclusive value range, when derivable from the symbol bounds.
    /// Computed on the rem-folded normal form, so `x - c*floor(x/c)`
    /// patterns get the tight `[0, c)` interval instead of the (correlated)
    /// naive interval difference.
    pub fn range(&self, bounds: &BTreeMap<VarId, u64>) -> Option<(i64, i64)> {
        let mut lc = LinComb::normalize(self, bounds).ok()?;
        let rems = lc.fold_rems();
        let (mut lo, mut hi) = (lc.cst, lc.cst);
        let mut add = |l: i64, h: i64, k: i64| {
            if k >= 0 {
                lo += l * k;
                hi += h * k;
            } else {
                lo += h * k;
                hi += l * k;
            }
        };
        for (atom, &k) in &lc.terms {
            let (l, h) = match atom {
                Quast::Sym(v) => (0, *bounds.get(v)? as i64 - 1),
                Quast::FloorDiv(inner, c) => {
                    let (l, h) = inner.range(bounds)?;
                    (l.div_euclid(*c), h.div_euclid(*c))
                }
                other => unreachable!("non-atom in linear normal form: {other:?}"),
            };
            add(l, h, k);
        }
        for (inner, c, k) in &rems {
            // Euclidean `x % c` is always in `[0, c)`; tighter when the
            // operand's own range already fits.
            let (l, h) = match inner.range(bounds) {
                Some((il, ih)) if il >= 0 && ih < *c => (il, ih),
                _ => (0, c - 1),
            };
            add(l, h, *k);
        }
        Some((lo, hi))
    }

    /// Normal form: constants folded, like terms collected, divisible parts
    /// hoisted out of floor divisions.
    pub fn simplify(&self, bounds: &BTreeMap<VarId, u64>) -> Result<Quast, CompileError> {
        Ok(LinComb::normalize(self, bounds)?.to_quast())
    }

    /// Emits the simplified expression through `em` as unsigned integer
    /// arithmetic; `x - c*floor(x/c)` patterns are emitted as `%`. Sound as
    /// long as the expression (and every floor-division operand) is
    /// non-negative on the symbol bounds, which is checked when the range is
    /// derivable.
    pub fn emit<E: QuastEmitter>(
        &self,
        bounds: &BTreeMap<VarId, u64>,
        em: &mut E,
    ) -> Result<E::Val, CompileError> {
        let mut lc = LinComb::normalize(self, bounds)?;
        let rems = lc.fold_rems();

        let coeff_val = |em: &mut E, coeff: i64| -> Result<E::Val, CompileError> {
            let c = u32::try_from(coeff.unsigned_abs())
                .map_err(|_| err(format!("coefficient {coeff} overflows u32")))?;
            Ok(em.cst(c))
        };
        let mut pos: Option<E::Val> = None;
        let mut neg: Option<E::Val> = None;
        let mut push = |em: &mut E, side: bool, v: E::Val| {
            let slot = if side { &mut pos } else { &mut neg };
            *slot = Some(match slot.take() {
                None => v,
                Some(acc) => em.add(acc, v),
            });
        };

        for (atom, coeff) in &lc.terms {
            if *coeff == 0 {
                continue;
            }
            let mut v = emit_atom(atom, bounds, em)?;
            if coeff.unsigned_abs() != 1 {
                let c = coeff_val(em, *coeff)?;
                v = em.mul(v, c);
            }
            push(em, *coeff > 0, v);
        }
        for (inner, c, coeff) in &rems {
            let iv = emit_atom(inner, bounds, em)?;
            let cv = coeff_val(em, *c)?;
            let mut v = em.rem(iv, cv);
            if coeff.unsigned_abs() != 1 {
                let k = coeff_val(em, *coeff)?;
                v = em.mul(v, k);
            }
            push(em, *coeff > 0, v);
        }
        if lc.cst != 0 {
            let v = coeff_val(em, lc.cst)?;
            push(em, lc.cst > 0, v);
        }

        Ok(match (pos, neg) {
            (Some(p), Some(n)) => em.sub(p, n),
            (Some(p), None) => p,
            (None, Some(n)) => {
                let z = em.cst(0);
                em.sub(z, n)
            }
            (None, None) => em.cst(0),
        })
    }
}

/// Target of [`Quast::emit`]: unsigned integer arithmetic value builder.
pub trait QuastEmitter {
    type Val: Clone;
    fn sym(&mut self, v: VarId) -> Self::Val;
    fn cst(&mut self, c: u32) -> Self::Val;
    fn add(&mut self, a: Self::Val, b: Self::Val) -> Self::Val;
    fn sub(&mut self, a: Self::Val, b: Self::Val) -> Self::Val;
    fn mul(&mut self, a: Self::Val, b: Self::Val) -> Self::Val;
    fn div(&mut self, a: Self::Val, b: Self::Val) -> Self::Val;
    fn rem(&mut self, a: Self::Val, b: Self::Val) -> Self::Val;
}

/// [`QuastEmitter`] producing C-style expression strings; symbols resolve
/// to SSA value names by the `VarId(i) <-> SSARes(i)` convention.
pub struct CStrEmitter;

impl QuastEmitter for CStrEmitter {
    type Val = String;

    fn sym(&mut self, v: VarId) -> String {
        format!("v{}", v.0)
    }

    fn cst(&mut self, c: u32) -> String {
        format!("{c}u")
    }

    fn add(&mut self, a: String, b: String) -> String {
        format!("({a} + {b})")
    }

    fn sub(&mut self, a: String, b: String) -> String {
        format!("({a} - {b})")
    }

    fn mul(&mut self, a: String, b: String) -> String {
        format!("({a} * {b})")
    }

    fn div(&mut self, a: String, b: String) -> String {
        format!("({a} / {b})")
    }

    fn rem(&mut self, a: String, b: String) -> String {
        format!("({a} % {b})")
    }
}

/// Emits a normal-form atom: a symbol or a floor division.
fn emit_atom<E: QuastEmitter>(
    atom: &Quast,
    bounds: &BTreeMap<VarId, u64>,
    em: &mut E,
) -> Result<E::Val, CompileError> {
    match atom {
        Quast::Sym(v) => Ok(em.sym(*v)),
        Quast::FloorDiv(inner, c) => {
            if *c <= 0 {
                return Err(err(format!("floor division by non-positive constant {c}")));
            }
            if let Some((lo, _)) = inner.range(bounds) {
                if lo < 0 {
                    return Err(err(format!(
                        "floor division operand may be negative: {inner:?}"
                    )));
                }
            }
            let iv = inner.emit(bounds, em)?;
            let cv = em.cst(*c as u32);
            Ok(em.div(iv, cv))
        }
        other => unreachable!("non-atom in linear normal form: {other:?}"),
    }
}

/// Linear combination of atoms (normal form): `cst + sum coeff * atom`,
/// where each atom is a `Sym` or a `FloorDiv` of a normalized expression.
struct LinComb {
    terms: BTreeMap<Quast, i64>,
    cst: i64,
}

impl LinComb {
    fn new() -> Self {
        LinComb {
            terms: BTreeMap::new(),
            cst: 0,
        }
    }

    fn add_term(&mut self, atom: Quast, coeff: i64) {
        if coeff == 0 {
            return;
        }
        let e = self.terms.entry(atom.clone()).or_insert(0);
        *e += coeff;
        if *e == 0 {
            self.terms.remove(&atom);
        }
    }

    fn scaled(mut self, c: i64) -> Self {
        if c == 0 {
            return LinComb::new();
        }
        for v in self.terms.values_mut() {
            *v *= c;
        }
        self.cst *= c;
        self
    }

    fn merge(&mut self, other: LinComb) {
        for (atom, coeff) in other.terms {
            self.add_term(atom, coeff);
        }
        self.cst += other.cst;
    }

    fn normalize(q: &Quast, bounds: &BTreeMap<VarId, u64>) -> Result<LinComb, CompileError> {
        Ok(match q {
            Quast::Sym(v) => {
                let mut lc = LinComb::new();
                lc.add_term(Quast::Sym(*v), 1);
                lc
            }
            Quast::Const(c) => {
                let mut lc = LinComb::new();
                lc.cst = *c;
                lc
            }
            Quast::Add(a, b) => {
                let mut lc = LinComb::normalize(a, bounds)?;
                lc.merge(LinComb::normalize(b, bounds)?);
                lc
            }
            Quast::Neg(a) => LinComb::normalize(a, bounds)?.scaled(-1),
            Quast::Mul(a, c) => LinComb::normalize(a, bounds)?.scaled(*c),
            Quast::FloorDiv(a, c) => {
                if *c <= 0 {
                    return Err(err(format!("floor division by non-positive constant {c}")));
                }
                let inner = LinComb::normalize(a, bounds)?;
                if *c == 1 {
                    return Ok(inner);
                }
                floordiv_lc(inner, *c, bounds)
            }
        })
    }

    fn to_quast(&self) -> Quast {
        let mut acc: Option<Quast> = None;
        for (atom, &coeff) in &self.terms {
            if coeff == 0 {
                continue;
            }
            let term = if coeff == 1 {
                atom.clone()
            } else {
                Quast::Mul(Rc::new(atom.clone()), coeff)
            };
            acc = Some(match acc {
                None => term,
                Some(a) => Quast::Add(Rc::new(a), Rc::new(term)),
            });
        }
        match (acc, self.cst) {
            (None, c) => Quast::Const(c),
            (Some(a), 0) => a,
            (Some(a), c) => Quast::Add(Rc::new(a), Rc::new(Quast::Const(c))),
        }
    }

    /// Extracts `k * (x % c)` patterns: a pair of terms `k*x` and
    /// `-k*c*floor(x/c)` (possibly partial) becomes a remainder term.
    /// Returns `(x, c, k)` triples; `self` keeps the leftovers.
    fn fold_rems(&mut self) -> Vec<(Quast, i64, i64)> {
        let fd_atoms: Vec<Quast> = self
            .terms
            .keys()
            .filter(|a| matches!(a, Quast::FloorDiv(_, c) if *c > 1))
            .cloned()
            .collect();
        let mut rems = Vec::new();
        for fd in fd_atoms {
            let Quast::FloorDiv(inner, c) = &fd else {
                unreachable!()
            };
            let (inner, c) = ((**inner).clone(), *c);
            let Some(&qa) = self.terms.get(&inner) else {
                continue;
            };
            let qf = *self.terms.get(&fd).unwrap_or(&0);
            // k * (x % c) consumes (k, -k*c) from the (x, floor(x/c))
            // coefficients; pick k to minimize the floor-div leftover.
            let k = if qa > 0 && qf < 0 {
                ((-qf + c / 2) / c).min(qa)
            } else if qa < 0 && qf > 0 {
                -(((qf + c / 2) / c).min(-qa))
            } else {
                continue;
            };
            if k == 0 {
                continue;
            }
            rems.push((inner.clone(), c, k));
            self.add_term(inner, -k);
            self.add_term(fd, k * c);
        }
        rems
    }
}

/// `floor((sum coeff*atom + cst) / c)`: hoists the part with coefficients
/// divisible by `c`; the rest stays under the division (dropped entirely if
/// its range provably lies in `[0, c)`).
fn floordiv_lc(inner: LinComb, c: i64, bounds: &BTreeMap<VarId, u64>) -> LinComb {
    let mut outer = LinComb::new();
    let mut rest = LinComb::new();
    for (atom, coeff) in inner.terms {
        if coeff % c == 0 {
            outer.add_term(atom, coeff / c);
        } else {
            rest.add_term(atom, coeff);
        }
    }
    outer.cst = inner.cst.div_euclid(c);
    rest.cst = inner.cst.rem_euclid(c);

    if rest.terms.is_empty() {
        // rest.cst is in [0, c), so floor(rest / c) == 0.
        return outer;
    }
    let rest_q = rest.to_quast();
    if let Some((lo, hi)) = rest_q.range(bounds) {
        if lo >= 0 && hi < c {
            return outer;
        }
    }
    outer.add_term(Quast::FloorDiv(Rc::new(rest_q), c), 1);
    outer
}

/// Row-major strides for `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Recovers the row-major coordinates of `flat` over `shape`.
pub fn delinearize(flat: &Quast, shape: &[usize]) -> Vec<Quast> {
    let strides = strides(shape);
    shape
        .iter()
        .enumerate()
        .map(|(d, &dim)| {
            let q = if strides[d] == 1 {
                flat.clone()
            } else {
                flat.floordiv(strides[d] as i64)
            };
            // The leading coordinate needs no wrap: flat < product(shape).
            if d == 0 {
                q
            } else {
                q.rem_c(dim as i64)
            }
        })
        .collect()
}

/// Row-major linearization of per-dimension index expressions.
pub fn linearize(exprs: &[Quast], shape: &[usize]) -> Quast {
    let strides = strides(shape);
    let mut acc: Option<Quast> = None;
    for (e, &s) in exprs.iter().zip(&strides) {
        let term = if s == 1 { e.clone() } else { e.mul_c(s as i64) };
        acc = Some(match acc {
            None => term,
            Some(a) => a.add(&term),
        });
    }
    acc.unwrap_or(Quast::Const(0))
}

// ---------------------------------------------------------------------------
// LinearLayout recovery
// ---------------------------------------------------------------------------

impl Quast {
    /// Tries to express the map as a [`LinearLayout`] over the concatenated
    /// bits of its symbols.
    ///
    /// Every symbol must have a power-of-two bound; the flat `k`-bit input
    /// packs each symbol's bits in ascending [`VarId`] order (first symbol
    /// in the low bits). The offset is read off the zero input, bases off
    /// the one-hot inputs, and the candidate layout is verified against the
    /// expression on the whole domain, so `None` is returned for any map
    /// that is not XOR-affine (e.g. addition with carries), has a missing
    /// or non-power-of-two bound, or whose domain exceeds
    /// [`EXHAUSTIVE_LIMIT`].
    pub fn to_linear_layout(&self, bounds: &BTreeMap<VarId, u64>) -> Option<LinearLayout> {
        let mut syms = BTreeSet::new();
        self.syms(&mut syms);
        // (symbol, bit offset in the flat input, bit width)
        let mut vars = Vec::new();
        let mut k = 0usize;
        for v in syms {
            let bound = *bounds.get(&v)?;
            if !bound.is_power_of_two() {
                return None;
            }
            let width = bound.trailing_zeros() as usize;
            vars.push((v, k, width));
            k += width;
        }
        if k > EXHAUSTIVE_LIMIT.trailing_zeros() as usize {
            return None;
        }
        let eval_at = |x: u64| -> Option<u64> {
            let env = vars
                .iter()
                .map(|&(v, off, width)| (v, ((x >> off) & ((1u64 << width) - 1)) as i64))
                .collect();
            u64::try_from(self.eval(&env)).ok()
        };
        let offset = eval_at(0)?;
        let bases = (0..k)
            .map(|i| eval_at(1 << i).map(|b| b ^ offset))
            .collect::<Option<Vec<_>>>()?;
        let layout = LinearLayout { bases, offset };
        for x in 0..(1u64 << k) {
            if eval_at(x)? != layout.apply(x) {
                return None;
            }
        }
        Some(layout)
    }
}

// ---------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------

/// Exhaustive map checking (scatter bijectivity, [`Quast::to_linear_layout`]
/// verification) is limited to this many points.
const EXHAUSTIVE_LIMIT: usize = 1 << 16;

/// The `#[scatter(...)]` attribute of a compute: a bijective quasi-affine map
/// from the logical output coordinates to physical coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Scatter {
    /// One symbol per logical dimension, outermost first.
    pub params: Vec<VarId>,
    /// Physical coordinate expressions, one per physical dimension.
    pub exprs: Vec<Quast>,
    /// Physical shape; `None` means identical to the logical shape (only
    /// allowed when the map preserves the number of dimensions).
    pub out_shape: Option<Vec<usize>>,
    /// Bounds of `params` (the logical shape); filled by canonicalization.
    pub bounds: BTreeMap<VarId, u64>,
}

impl Scatter {
    /// The physical shape given the logical one.
    pub fn out_shape_for(&self, logical: &[usize]) -> Result<Vec<usize>, CompileError> {
        match &self.out_shape {
            Some(s) => Ok(s.clone()),
            None if self.exprs.len() == logical.len() => Ok(logical.to_vec()),
            None => Err(err(format!(
                "scatter changes rank ({} -> {}) so it must specify output bounds",
                logical.len(),
                self.exprs.len()
            ))),
        }
    }

    /// Fills the symbol bounds from the logical shape and checks the map is
    /// a bijection onto the physical shape. Returns the physical shape.
    pub fn bind_and_validate(&mut self, logical: &[usize]) -> Result<Vec<usize>, CompileError> {
        if self.params.len() != logical.len() {
            return Err(err(format!(
                "scatter has {} parameters but the compute output has rank {}",
                self.params.len(),
                logical.len()
            )));
        }
        self.bounds = self
            .params
            .iter()
            .zip(logical)
            .map(|(&p, &b)| (p, b as u64))
            .collect();
        let out = self.out_shape_for(logical)?;
        if self.exprs.len() != out.len() {
            return Err(err(format!(
                "scatter has {} expressions but {} output bounds",
                self.exprs.len(),
                out.len()
            )));
        }
        let total: usize = logical.iter().product();
        if out.iter().product::<usize>() != total {
            return Err(err(format!(
                "scatter output shape {out:?} does not have the same number of \
                 elements as the logical shape {logical:?}"
            )));
        }
        for (e, &dim) in self.exprs.iter().zip(&out) {
            let (lo, hi) = e.range(&self.bounds).ok_or_else(|| {
                err(format!(
                    "scatter expression uses an unbounded symbol: {e:?}"
                ))
            })?;
            if lo < 0 || hi >= dim as i64 {
                return Err(err(format!(
                    "scatter expression range [{lo}, {hi}] exceeds output bound {dim}"
                )));
            }
        }
        if total <= EXHAUSTIVE_LIMIT {
            self.check_bijective(logical, &out)?;
        }
        Ok(out)
    }

    fn check_bijective(&self, logical: &[usize], out: &[usize]) -> Result<(), CompileError> {
        let total: usize = logical.iter().product();
        let out_strides = strides(out);
        let mut seen = vec![false; total];
        let mut coords = vec![0i64; logical.len()];
        for flat in 0..total {
            let mut r = flat;
            for (d, &dim) in logical.iter().enumerate().rev() {
                coords[d] = (r % dim) as i64;
                r /= dim;
            }
            let env: BTreeMap<VarId, i64> = self
                .params
                .iter()
                .copied()
                .zip(coords.iter().copied())
                .collect();
            let phys: usize = self
                .exprs
                .iter()
                .zip(&out_strides)
                .map(|(e, &s)| e.eval(&env) as usize * s)
                .sum();
            if seen[phys] {
                return Err(err(format!(
                    "scatter is not bijective: physical index {phys} is hit twice"
                )));
            }
            seen[phys] = true;
        }
        Ok(())
    }

    /// The composed store map `logical flat index -> physical flat index`,
    /// as a simplified expression in the single symbol `flat`. Requires
    /// [`Scatter::bind_and_validate`] to have run.
    pub fn store_map(&self, flat: VarId) -> Result<ScatterStore, CompileError> {
        let logical: Vec<usize> = self
            .params
            .iter()
            .map(|p| self.bounds[p] as usize)
            .collect();
        let coords = delinearize(&Quast::sym(flat), &logical);
        let map: BTreeMap<VarId, Quast> = self.params.iter().copied().zip(coords).collect();
        let out = self.out_shape_for(&logical)?;
        let exprs: Vec<Quast> = self.exprs.iter().map(|e| e.substitute(&map)).collect();
        let mut bounds = BTreeMap::new();
        bounds.insert(flat, logical.iter().product::<usize>() as u64);
        let expr = linearize(&exprs, &out).simplify(&bounds)?;
        Ok(ScatterStore { flat, expr, bounds })
    }
}

/// A precomposed scatter store map: `expr` gives the physical flat index as
/// a function of the logical flat index (the symbol `flat`).
#[derive(Clone, Debug)]
pub struct ScatterStore {
    pub flat: VarId,
    pub expr: Quast,
    pub bounds: BTreeMap<VarId, u64>,
}

/// The `#[par((t, s) -> f(t, s))]` attribute of a compute: its compute
/// layout, mapping the physical coordinates — thread index `t` and per-thread
/// sequential (repeat) index `s` — to the logical compute index. Must be
/// convertible to a [`LinearLayout`](crate::kernel_ir::LinearLayout) once the
/// bounds are known.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParSpec {
    /// Thread-index symbol; allocated before `seq` so that it occupies the
    /// low bits of the physical index in [`Quast::to_linear_layout`].
    pub thread: VarId,
    /// Per-thread sequential index symbol.
    pub seq: VarId,
    /// Logical index as a quasi-affine expression of `thread` and `seq`.
    pub expr: Quast,
}

// ---------------------------------------------------------------------------
// HIR emission
// ---------------------------------------------------------------------------

/// [`QuastEmitter`] that builds HIR nodes; every created node is typed U32.
pub(crate) struct NodeEmitter<'a> {
    pub b: &'a mut crate::ir::IRBuilder,
    pub types: &'a mut TypeMap,
    pub env: &'a BTreeMap<VarId, NodeId>,
}

impl NodeEmitter<'_> {
    fn typed(&mut self, id: NodeId) -> NodeId {
        self.types
            .insert(id, crate::ir::Type::Scalar(crate::ir::ScalarType::U32));
        id
    }
}

impl QuastEmitter for NodeEmitter<'_> {
    type Val = NodeId;

    fn sym(&mut self, v: VarId) -> NodeId {
        *self
            .env
            .get(&v)
            .unwrap_or_else(|| panic!("unbound symbol {v:?} in NodeEmitter"))
    }

    fn cst(&mut self, c: u32) -> NodeId {
        let id = self.b.const_u32(c);
        self.typed(id)
    }

    fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let id = self.b.add(a, b);
        self.typed(id)
    }

    fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let id = self.b.sub(a, b);
        self.typed(id)
    }

    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let id = self.b.mul(a, b);
        self.typed(id)
    }

    fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let id = self.b.div(a, b);
        self.typed(id)
    }

    fn rem(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let id = self.b.rem(a, b);
        self.typed(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: u32) -> VarId {
        VarId(i)
    }

    /// Emitter producing fully parenthesized strings, for shape assertions.
    struct StrEmitter;

    impl QuastEmitter for StrEmitter {
        type Val = &'static str;

        fn sym(&mut self, _v: VarId) -> &'static str {
            "f"
        }
        fn cst(&mut self, c: u32) -> &'static str {
            Box::leak(format!("{c}").into_boxed_str())
        }
        fn add(&mut self, a: &'static str, b: &'static str) -> &'static str {
            Box::leak(format!("({a} + {b})").into_boxed_str())
        }
        fn sub(&mut self, a: &'static str, b: &'static str) -> &'static str {
            Box::leak(format!("({a} - {b})").into_boxed_str())
        }
        fn mul(&mut self, a: &'static str, b: &'static str) -> &'static str {
            Box::leak(format!("({a} * {b})").into_boxed_str())
        }
        fn div(&mut self, a: &'static str, b: &'static str) -> &'static str {
            Box::leak(format!("({a} / {b})").into_boxed_str())
        }
        fn rem(&mut self, a: &'static str, b: &'static str) -> &'static str {
            Box::leak(format!("({a} % {b})").into_boxed_str())
        }
    }

    #[test]
    fn identity_reshape_simplifies_to_sym() {
        // linearize(delinearize(f, [3, 4]), [3, 4]) == f
        let f = Quast::sym(v(0));
        let coords = delinearize(&f, &[3, 4]);
        let q = linearize(&coords, &[3, 4]);
        let bounds = BTreeMap::from([(v(0), 12u64)]);
        assert_eq!(q.simplify(&bounds).unwrap(), f);
    }

    #[test]
    fn transpose_emits_rem_and_div() {
        // i -> (i % 4, i / 4) into [4, 3]: physical = (i % 4) * 3 + i / 4.
        let f = Quast::sym(v(0));
        let q = linearize(&[f.rem_c(4), f.floordiv(4)], &[4, 3]);
        let bounds = BTreeMap::from([(v(0), 12u64)]);
        let s = q.emit(&bounds, &mut StrEmitter).unwrap();
        assert_eq!(s, "((f / 4) + ((f % 4) * 3))");
        // Semantics check over the whole domain.
        for i in 0..12i64 {
            let env = BTreeMap::from([(v(0), i)]);
            assert_eq!(q.eval(&env), (i % 4) * 3 + i / 4);
        }
    }

    #[test]
    fn bounded_floordiv_is_zero() {
        let f = Quast::sym(v(0));
        let bounds = BTreeMap::from([(v(0), 8u64)]);
        assert_eq!(f.floordiv(8).simplify(&bounds).unwrap(), Quast::Const(0));
        assert_eq!(f.rem_c(8).simplify(&bounds).unwrap(), f);
    }

    #[test]
    fn rem_range_is_tight() {
        let f = Quast::sym(v(0));
        let bounds = BTreeMap::from([(v(0), 512u64)]);
        assert_eq!(f.rem_c(2).range(&bounds), Some((0, 1)));
        // Deinterleave: (f % 2) * 256 + f / 2 covers exactly [0, 512).
        let q = f.rem_c(2).mul_c(256).add(&f.floordiv(2));
        assert_eq!(q.range(&bounds), Some((0, 511)));
    }

    #[test]
    fn scatter_validation() {
        // Transpose [3, 4] -> [4, 3] is bijective.
        let (i, j) = (v(0), v(1));
        let mut sc = Scatter {
            params: vec![i, j],
            exprs: vec![Quast::sym(j), Quast::sym(i)],
            out_shape: Some(vec![4, 3]),
            bounds: BTreeMap::new(),
        };
        assert_eq!(sc.bind_and_validate(&[3, 4]).unwrap(), vec![4, 3]);

        // i -> i / 2 is not injective.
        let mut sc = Scatter {
            params: vec![i],
            exprs: vec![Quast::sym(i).floordiv(2)],
            out_shape: None,
            bounds: BTreeMap::new(),
        };
        assert!(sc.bind_and_validate(&[4]).is_err());

        // Size mismatch is rejected.
        let mut sc = Scatter {
            params: vec![i],
            exprs: vec![Quast::sym(i).rem_c(4), Quast::sym(i).floordiv(4)],
            out_shape: Some(vec![4, 4]),
            bounds: BTreeMap::new(),
        };
        assert!(sc.bind_and_validate(&[12]).is_err());
    }

    #[test]
    fn store_map_composition() {
        // Nested transpose: logical [10, 7], scatter (i, j) -> (j, i) into
        // [7, 10]; store(f) = (f % 7) * 10 + f / 7.
        let (i, j, f) = (v(0), v(1), v(2));
        let mut sc = Scatter {
            params: vec![i, j],
            exprs: vec![Quast::sym(j), Quast::sym(i)],
            out_shape: Some(vec![7, 10]),
            bounds: BTreeMap::new(),
        };
        sc.bind_and_validate(&[10, 7]).unwrap();
        let store = sc.store_map(f).unwrap();
        let s = store.expr.emit(&store.bounds, &mut StrEmitter).unwrap();
        assert_eq!(s, "((f / 7) + ((f % 7) * 10))");
        for flat in 0..70i64 {
            let env = BTreeMap::from([(f, flat)]);
            assert_eq!(store.expr.eval(&env), (flat % 7) * 10 + flat / 7);
        }
    }

    #[test]
    fn identity_quast_recovers_identity_layout() {
        let f = Quast::sym(v(0));
        let bounds = BTreeMap::from([(v(0), 16u64)]);
        assert_eq!(f.to_linear_layout(&bounds), Some(LinearLayout::identity(4)));
    }

    #[test]
    fn transpose_recovers_bit_rotation() {
        // Transpose [8, 4] -> [4, 8]: f -> (f % 4) * 8 + f / 4 rotates the
        // five index bits.
        let f = Quast::sym(v(0));
        let q = f.rem_c(4).mul_c(8).add(&f.floordiv(4));
        let bounds = BTreeMap::from([(v(0), 32u64)]);
        let layout = q.to_linear_layout(&bounds).unwrap();
        assert_eq!(layout.bases, vec![8, 16, 1, 2, 4]);
        for x in 0..32u64 {
            assert_eq!(layout.apply(x), (x % 4) * 8 + x / 4);
        }
    }

    #[test]
    fn multi_symbol_linearization_concatenates_bits() {
        // (i, j) -> i * 8 + j over i < 4, j < 8: i occupies the low input
        // bits (ascending VarId order) but the high output bits.
        let (i, j) = (v(0), v(1));
        let q = Quast::sym(i).mul_c(8).add(&Quast::sym(j));
        let bounds = BTreeMap::from([(i, 4u64), (j, 8u64)]);
        let layout = q.to_linear_layout(&bounds).unwrap();
        assert_eq!(layout.bases, vec![8, 16, 1, 2, 4]);
    }

    #[test]
    fn non_power_of_two_bound_is_rejected() {
        let f = Quast::sym(v(0));
        assert_eq!(f.to_linear_layout(&BTreeMap::from([(v(0), 12u64)])), None);
        // A missing bound is also rejected.
        assert_eq!(f.to_linear_layout(&BTreeMap::new()), None);
    }

    #[test]
    fn non_linear_maps_are_rejected() {
        let bounds = BTreeMap::from([(v(0), 4u64), (v(1), 4u64)]);
        // Constant offset: T(0) != 0.
        let f = Quast::sym(v(0));
        assert_eq!(f.add(&Quast::cst(1)).to_linear_layout(&bounds), None);
        // i + j carries between bits, so it is not XOR-linear even though
        // the one-hot evaluations look like the identity.
        let q = Quast::sym(v(0)).add(&Quast::sym(v(1)));
        assert_eq!(q.to_linear_layout(&bounds), None);
        // f + f/2 carries within a single symbol's bits.
        assert_eq!(f.add(&f.floordiv(2)).to_linear_layout(&bounds), None);
    }

    #[test]
    fn oversized_domain_is_rejected() {
        let f = Quast::sym(v(0));
        let bounds = BTreeMap::from([(v(0), 1u64 << 20)]);
        assert_eq!(f.to_linear_layout(&bounds), None);
    }
}
