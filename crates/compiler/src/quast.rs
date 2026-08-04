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
    sync::Arc,
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

/// A quasi-affine expression over constants of type `T`.
///
/// `T = i64` is the classic quasi-affine [`Quast`]; `T = SymConst` is
/// [`SExpr`], whose "constants" may be symbolic module parameters that only
/// resolve to numbers at kernel-instantiation time.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Expr<T> {
    Sym(VarId),
    Const(T),
    Add(Arc<Expr<T>>, Arc<Expr<T>>),
    /// Multiplication by a constant.
    Mul(Arc<Expr<T>>, T),
    /// Floor division by a positive constant.
    FloorDiv(Arc<Expr<T>>, T),
    Neg(Arc<Expr<T>>),
}

/// A quasi-affine expression with literal integer constants.
pub type Quast = Expr<i64>;

/// A constant term of an [`SExpr`]: a literal or a symbolic module
/// parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymConst {
    Lit(i64),
    Sym(VarId),
}

impl SymConst {
    pub fn as_lit(&self) -> Option<i64> {
        match self {
            SymConst::Lit(c) => Some(*c),
            SymConst::Sym(_) => None,
        }
    }
}

impl From<i64> for SymConst {
    fn from(c: i64) -> Self {
        SymConst::Lit(c)
    }
}

/// A quasi-affine-shaped expression whose constants may be symbolic module
/// parameters. Unlike [`Quast`] it cannot be normalized, range-analyzed or
/// converted to a `LinearLayout`; it supports only literal folding,
/// substitution and concretization back to `Quast`.
pub type SExpr = Expr<SymConst>;

// The only integer `From` for `SExpr`: a unique impl lets bare integer
// literals at `impl Into<SizeExpr>` call sites infer `usize`.
impl From<usize> for SExpr {
    fn from(c: usize) -> Self {
        SExpr::Const(SymConst::Lit(c as i64))
    }
}

impl std::fmt::Display for SymConst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymConst::Lit(c) => write!(f, "{c}"),
            SymConst::Sym(v) => write!(f, "s{}", v.0),
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Expr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Sym(v) => write!(f, "v{}", v.0),
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Add(a, b) => write!(f, "({a} + {b})"),
            Expr::Mul(a, c) => write!(f, "({a} * {c})"),
            Expr::FloorDiv(a, c) => write!(f, "({a} / {c})"),
            Expr::Neg(a) => write!(f, "(-{a})"),
        }
    }
}

impl<T: Clone> Expr<T> {
    pub fn sym(v: VarId) -> Self {
        Expr::Sym(v)
    }

    pub fn cst(c: T) -> Self {
        Expr::Const(c)
    }

    pub fn add(&self, other: &Self) -> Self {
        Expr::Add(Arc::new(self.clone()), Arc::new(other.clone()))
    }

    pub fn sub(&self, other: &Self) -> Self {
        Expr::Add(Arc::new(self.clone()), Arc::new(other.neg()))
    }

    pub fn neg(&self) -> Self {
        Expr::Neg(Arc::new(self.clone()))
    }

    pub fn mul_c(&self, c: T) -> Self {
        Expr::Mul(Arc::new(self.clone()), c)
    }

    pub fn floordiv(&self, c: T) -> Self {
        Expr::FloorDiv(Arc::new(self.clone()), c)
    }

    /// `self % c` as `self - c * floor(self / c)`.
    pub fn rem_c(&self, c: T) -> Self {
        self.sub(&self.floordiv(c.clone()).mul_c(c))
    }

    /// Inserts every symbol appearing in a `Sym` position into `out`.
    /// (For [`SExpr`], parameters in constant positions are collected by
    /// [`SExpr::param_syms`] instead.)
    pub fn syms(&self, out: &mut BTreeSet<VarId>) {
        match self {
            Expr::Sym(v) => {
                out.insert(*v);
            }
            Expr::Const(_) => {}
            Expr::Add(a, b) => {
                a.syms(out);
                b.syms(out);
            }
            Expr::Mul(a, _) | Expr::FloorDiv(a, _) | Expr::Neg(a) => a.syms(out),
        }
    }

    /// Replaces symbols per `map` (symbols absent from the map are kept).
    pub fn substitute(&self, map: &BTreeMap<VarId, Expr<T>>) -> Expr<T> {
        match self {
            Expr::Sym(v) => map.get(v).cloned().unwrap_or_else(|| self.clone()),
            Expr::Const(_) => self.clone(),
            Expr::Add(a, b) => Expr::Add(Arc::new(a.substitute(map)), Arc::new(b.substitute(map))),
            Expr::Mul(a, c) => Expr::Mul(Arc::new(a.substitute(map)), c.clone()),
            Expr::FloorDiv(a, c) => Expr::FloorDiv(Arc::new(a.substitute(map)), c.clone()),
            Expr::Neg(a) => Expr::Neg(Arc::new(a.substitute(map))),
        }
    }
}

impl From<&Quast> for SExpr {
    fn from(q: &Quast) -> SExpr {
        match q {
            Quast::Sym(v) => SExpr::Sym(*v),
            Quast::Const(c) => SExpr::Const(SymConst::Lit(*c)),
            Quast::Add(a, b) => SExpr::Add(Arc::new((&**a).into()), Arc::new((&**b).into())),
            Quast::Mul(a, c) => SExpr::Mul(Arc::new((&**a).into()), SymConst::Lit(*c)),
            Quast::FloorDiv(a, c) => SExpr::FloorDiv(Arc::new((&**a).into()), SymConst::Lit(*c)),
            Quast::Neg(a) => SExpr::Neg(Arc::new((&**a).into())),
        }
    }
}

impl From<Quast> for SExpr {
    fn from(q: Quast) -> SExpr {
        SExpr::from(&q)
    }
}

impl SExpr {
    /// Evaluates with every loop symbol *and* parameter bound in `env`.
    /// Panics on unbound symbols or non-positive divisors (programmer
    /// error). Loop vars and params share the `VarId` namespace, so a single
    /// environment serves both.
    pub fn eval(&self, env: &BTreeMap<VarId, i64>) -> i64 {
        let c_of = |c: &SymConst| match c {
            SymConst::Lit(l) => *l,
            SymConst::Sym(v) => *env
                .get(v)
                .unwrap_or_else(|| panic!("unbound parameter {v:?} in SExpr::eval")),
        };
        match self {
            SExpr::Sym(v) => *env
                .get(v)
                .unwrap_or_else(|| panic!("unbound symbol {v:?} in SExpr::eval")),
            SExpr::Const(c) => c_of(c),
            SExpr::Add(a, b) => a.eval(env) + b.eval(env),
            SExpr::Mul(a, c) => a.eval(env) * c_of(c),
            SExpr::FloorDiv(a, c) => {
                let c = c_of(c);
                assert!(c > 0, "FloorDiv divisor must be positive");
                a.eval(env).div_euclid(c)
            }
            SExpr::Neg(a) => -a.eval(env),
        }
    }

    /// Inserts every parameter appearing in a constant position into `out`
    /// (the counterpart of [`Expr::syms`], which collects loop-var `Sym`s).
    pub fn param_syms(&self, out: &mut BTreeSet<VarId>) {
        match self {
            SExpr::Sym(_) => {}
            SExpr::Const(c) => {
                if let SymConst::Sym(v) = c {
                    out.insert(*v);
                }
            }
            SExpr::Add(a, b) => {
                a.param_syms(out);
                b.param_syms(out);
            }
            SExpr::Mul(a, c) | SExpr::FloorDiv(a, c) => {
                if let SymConst::Sym(v) = c {
                    out.insert(*v);
                }
                a.param_syms(out);
            }
            SExpr::Neg(a) => a.param_syms(out),
        }
    }

    /// Substitutes parameters bound in `env` with literals; unbound
    /// parameters and loop-var `Sym`s are untouched (loop vars substitute
    /// via [`Expr::substitute`]).
    pub fn concretize(&self, env: &BTreeMap<VarId, i64>) -> SExpr {
        let c_of = |c: &SymConst| match c {
            SymConst::Sym(v) => env.get(v).copied().map(SymConst::Lit).unwrap_or(*c),
            lit => *lit,
        };
        match self {
            SExpr::Sym(_) => self.clone(),
            SExpr::Const(c) => SExpr::Const(c_of(c)),
            SExpr::Add(a, b) => {
                SExpr::Add(Arc::new(a.concretize(env)), Arc::new(b.concretize(env)))
            }
            SExpr::Mul(a, c) => SExpr::Mul(Arc::new(a.concretize(env)), c_of(c)),
            SExpr::FloorDiv(a, c) => SExpr::FloorDiv(Arc::new(a.concretize(env)), c_of(c)),
            SExpr::Neg(a) => SExpr::Neg(Arc::new(a.concretize(env))),
        }
    }

    /// Substitutes parameters bound in `map` with expressions: the
    /// param-side counterpart of [`Expr::substitute`], which replaces
    /// loop-var `Sym` positions. A param in a `Mul`/`FloorDiv` coefficient
    /// position can only take an image that folds to a single [`SymConst`];
    /// `None` when a mapped image is not representable there.
    pub fn subst_params(&self, map: &BTreeMap<VarId, SExpr>) -> Option<SExpr> {
        let c_of = |c: &SymConst| match c {
            SymConst::Sym(v) => match map.get(v) {
                None => Some(*c),
                Some(e) => match e.fold_lits() {
                    SExpr::Const(c) => Some(c),
                    _ => None,
                },
            },
            lit => Some(*lit),
        };
        Some(match self {
            SExpr::Sym(_) => self.clone(),
            SExpr::Const(SymConst::Sym(v)) if map.contains_key(v) => map[v].clone(),
            SExpr::Const(_) => self.clone(),
            SExpr::Add(a, b) => SExpr::Add(
                Arc::new(a.subst_params(map)?),
                Arc::new(b.subst_params(map)?),
            ),
            SExpr::Mul(a, c) => SExpr::Mul(Arc::new(a.subst_params(map)?), c_of(c)?),
            SExpr::FloorDiv(a, c) => SExpr::FloorDiv(Arc::new(a.subst_params(map)?), c_of(c)?),
            SExpr::Neg(a) => SExpr::Neg(Arc::new(a.subst_params(map)?)),
        })
    }

    /// Converts to a [`Quast`] if no symbolic parameters remain.
    pub fn try_to_quast(&self) -> Option<Quast> {
        Some(match self {
            SExpr::Sym(v) => Quast::Sym(*v),
            SExpr::Const(c) => Quast::Const(c.as_lit()?),
            SExpr::Add(a, b) => {
                Quast::Add(Arc::new(a.try_to_quast()?), Arc::new(b.try_to_quast()?))
            }
            SExpr::Mul(a, c) => Quast::Mul(Arc::new(a.try_to_quast()?), c.as_lit()?),
            SExpr::FloorDiv(a, c) => Quast::FloorDiv(Arc::new(a.try_to_quast()?), c.as_lit()?),
            SExpr::Neg(a) => Quast::Neg(Arc::new(a.try_to_quast()?)),
        })
    }

    /// The monomorphization primitive: [`SExpr::concretize`] with `env`,
    /// then [`SExpr::try_to_quast`]. `None` iff a parameter is unbound.
    pub fn try_concretize(&self, env: &BTreeMap<VarId, i64>) -> Option<Quast> {
        self.concretize(env).try_to_quast()
    }

    /// Literal value if the expression contains no loop vars or parameters.
    pub fn as_const(&self) -> Option<i64> {
        match self {
            SExpr::Sym(_) => None,
            SExpr::Const(c) => c.as_lit(),
            SExpr::Add(a, b) => Some(a.as_const()? + b.as_const()?),
            SExpr::Mul(a, c) => Some(a.as_const()? * c.as_lit()?),
            SExpr::FloorDiv(a, c) => {
                let c = c.as_lit()?;
                if c <= 0 {
                    return None;
                }
                Some(a.as_const()?.div_euclid(c))
            }
            SExpr::Neg(a) => Some(-a.as_const()?),
        }
    }

    /// Best-effort literal folding. `SymConst` is not closed under
    /// arithmetic, so like terms with symbolic coefficients are not
    /// combined; only all-literal subterms fold and arithmetic identities
    /// (`+0`, `*1`, `*0`, `/1`, double negation) are erased.
    pub fn fold_lits(&self) -> SExpr {
        use SymConst::Lit;
        match self {
            SExpr::Sym(_) | SExpr::Const(_) => self.clone(),
            SExpr::Add(a, b) => {
                let (a, b) = (a.fold_lits(), b.fold_lits());
                match (&a, &b) {
                    (SExpr::Const(Lit(x)), SExpr::Const(Lit(y))) => SExpr::Const(Lit(x + y)),
                    (SExpr::Const(Lit(0)), _) => b,
                    (_, SExpr::Const(Lit(0))) => a,
                    _ => SExpr::Add(Arc::new(a), Arc::new(b)),
                }
            }
            SExpr::Mul(a, c) => {
                let a = a.fold_lits();
                match (&a, c) {
                    (_, Lit(0)) => SExpr::Const(Lit(0)),
                    (_, Lit(1)) => a,
                    (SExpr::Const(Lit(x)), Lit(y)) => SExpr::Const(Lit(x * y)),
                    _ => SExpr::Mul(Arc::new(a), *c),
                }
            }
            SExpr::FloorDiv(a, c) => {
                let a = a.fold_lits();
                match (&a, c) {
                    (_, Lit(1)) => a,
                    (SExpr::Const(Lit(x)), Lit(y)) if *y > 0 => SExpr::Const(Lit(x.div_euclid(*y))),
                    _ => SExpr::FloorDiv(Arc::new(a), *c),
                }
            }
            SExpr::Neg(a) => {
                let a = a.fold_lits();
                match &a {
                    SExpr::Const(Lit(x)) => SExpr::Const(Lit(-x)),
                    SExpr::Neg(inner) => (**inner).clone(),
                    _ => SExpr::Neg(Arc::new(a)),
                }
            }
        }
    }

    /// Product of `dims` as a single expression. `Mul` only carries a
    /// [`SymConst`] coefficient, so the product is representable as long as
    /// at most one factor is a *compound* expression (literals and bare
    /// parameters always fold in); otherwise `None`.
    pub fn product(dims: &[SExpr]) -> Option<SExpr> {
        let mut acc = SExpr::Const(SymConst::Lit(1));
        for d in dims {
            let d = d.fold_lits();
            acc = if let SExpr::Const(c) = d {
                acc.mul_c(c)
            } else if let SExpr::Const(c) = acc {
                d.mul_c(c)
            } else {
                return None;
            };
        }
        Some(acc.fold_lits())
    }
}

impl Quast {
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

    /// Inclusive value range, when derivable from the symbol bounds.
    ///
    /// `floor(x/c)` terms sharing a root symbol whose divisors form a
    /// divisibility chain are a mixed-radix digit decomposition of `x`; the
    /// digits vary independently, so those groups get an exact digit-wise
    /// interval (bit-reversal and row-major (de)linearization expressions
    /// all have this shape). Everything else is computed on the rem-folded
    /// normal form, so `x - c*floor(x/c)` patterns still get the tight
    /// `[0, c)` interval instead of the (correlated) naive difference.
    pub fn range(&self, bounds: &BTreeMap<VarId, u64>) -> Option<(i64, i64)> {
        let lc = LinComb::normalize(self, bounds).ok()?;
        let mut groups: BTreeMap<VarId, Vec<(i64, i64)>> = BTreeMap::new();
        let mut rest = LinComb::new();
        rest.cst = lc.cst;
        for (atom, &k) in &lc.terms {
            match simple_root(atom) {
                Some((v, c)) => groups.entry(v).or_default().push((c, k)),
                None => rest.add_term(atom.clone(), k),
            }
        }
        let (mut lo, mut hi) = (rest.cst, rest.cst);
        for (v, terms) in groups {
            let n = *bounds.get(&v)? as i64;
            if let Some((l, h)) = digit_range(&terms, n) {
                lo += l;
                hi += h;
            } else {
                for (c, k) in terms {
                    let atom = if c == 1 {
                        Quast::Sym(v)
                    } else {
                        Quast::FloorDiv(Arc::new(Quast::Sym(v)), c)
                    };
                    rest.add_term(atom, k);
                }
            }
        }
        let rems = rest.fold_rems();
        let mut add = |l: i64, h: i64, k: i64| {
            if k >= 0 {
                lo += l * k;
                hi += h * k;
            } else {
                lo += h * k;
                hi += l * k;
            }
        };
        for (atom, &k) in &rest.terms {
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
                Quast::Mul(Arc::new(atom.clone()), coeff)
            };
            acc = Some(match acc {
                None => term,
                Some(a) => Quast::Add(Arc::new(a), Arc::new(term)),
            });
        }
        match (acc, self.cst) {
            (None, c) => Quast::Const(c),
            (Some(a), 0) => a,
            (Some(a), c) => Quast::Add(Arc::new(a), Arc::new(Quast::Const(c))),
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

/// `x` or an arbitrarily nested `floor(x / c)`: the root symbol and the
/// combined divisor (`floor(floor(x/a)/b) = floor(x/(a·b))` for positive
/// divisors).
fn simple_root(atom: &Quast) -> Option<(VarId, i64)> {
    match atom {
        Quast::Sym(v) => Some((*v, 1)),
        Quast::FloorDiv(inner, c) => {
            let (v, c2) = simple_root(inner)?;
            Some((v, c2.checked_mul(*c)?))
        }
        _ => None,
    }
}

/// Interval of `Σⱼ kⱼ · floor(x / dⱼ)` over `x ∈ [0, n)` when the divisors
/// form a divisibility chain: `x` then decomposes into independent
/// mixed-radix digits (`digitᵢ = floor(x/dᵢ) % (dᵢ₊₁/dᵢ)`, the top digit is
/// `floor(x/d_top)` itself) and each `floor(x/dⱼ)` is the digit combination
/// `Σ_{i≥j} digitᵢ · dᵢ/dⱼ`, so the extremes are the digit-wise extremes.
/// `None` when the divisors do not chain — the caller falls back to
/// per-term interval arithmetic.
fn digit_range(terms: &[(i64, i64)], n: i64) -> Option<(i64, i64)> {
    if n <= 0 {
        return None;
    }
    let mut merged: BTreeMap<i64, i128> = BTreeMap::new();
    for &(d, k) in terms {
        debug_assert!(d > 0, "normal form has positive divisors");
        *merged.entry(d).or_insert(0) += k as i128;
    }
    let ds: Vec<(i64, i128)> = merged.into_iter().collect();
    if ds.windows(2).any(|w| w[1].0 % w[0].0 != 0) {
        return None;
    }
    let (mut lo, mut hi) = (0i128, 0i128);
    for (i, &(d_i, _)) in ds.iter().enumerate() {
        let max_digit = match ds.get(i + 1) {
            Some(&(d_next, _)) => (d_next / d_i - 1).min((n - 1) / d_i),
            None => (n - 1) / d_i,
        } as i128;
        let w: i128 = ds[..=i]
            .iter()
            .map(|&(d_j, k_j)| k_j * (d_i / d_j) as i128)
            .sum();
        let ext = w * max_digit;
        if ext >= 0 {
            hi += ext;
        } else {
            lo += ext;
        }
    }
    Some((i64::try_from(lo).ok()?, i64::try_from(hi).ok()?))
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
    outer.add_term(Quast::FloorDiv(Arc::new(rest_q), c), 1);
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
pub(crate) const EXHAUSTIVE_LIMIT: usize = 1 << 16;

/// The `#[scatter(...)]` attribute of a compute: a bijective quasi-affine map
/// from the logical output coordinates to physical coordinates, together
/// with its author-supplied inverse (physical back to logical).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Scatter {
    /// One symbol per logical dimension, outermost first.
    pub params: Vec<VarId>,
    /// Physical coordinate expressions, one per physical dimension.
    pub exprs: Vec<Quast>,
    /// Inverse map: one symbol per physical dimension, outermost first.
    pub inv_params: Vec<VarId>,
    /// Logical coordinate expressions, one per logical dimension.
    pub inv_exprs: Vec<Quast>,
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
    /// a bijection onto the physical shape whose declared inverse actually
    /// inverts it. Returns the physical shape.
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
        if self.inv_params.len() != out.len() {
            return Err(err(format!(
                "scatter inverse has {} parameters but the physical shape has rank {}",
                self.inv_params.len(),
                out.len()
            )));
        }
        if self.inv_exprs.len() != logical.len() {
            return Err(err(format!(
                "scatter inverse has {} expressions but the logical shape has rank {}",
                self.inv_exprs.len(),
                logical.len()
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
        let inv_bounds: BTreeMap<VarId, u64> = self
            .inv_params
            .iter()
            .zip(&out)
            .map(|(&p, &b)| (p, b as u64))
            .collect();
        for (e, &dim) in self.inv_exprs.iter().zip(logical) {
            let (lo, hi) = e.range(&inv_bounds).ok_or_else(|| {
                err(format!(
                    "scatter inverse expression uses an unbounded symbol: {e:?}"
                ))
            })?;
            if lo < 0 || hi >= dim as i64 {
                return Err(err(format!(
                    "scatter inverse expression range [{lo}, {hi}] exceeds logical bound {dim}"
                )));
            }
        }
        if total <= EXHAUSTIVE_LIMIT {
            self.check_bijective_with_inverse(logical, &out)?;
        }
        Ok(out)
    }

    /// Exhaustive check: the forward map hits every physical index exactly
    /// once, and the declared inverse maps each image back to its preimage.
    fn check_bijective_with_inverse(
        &self,
        logical: &[usize],
        out: &[usize],
    ) -> Result<(), CompileError> {
        let total: usize = logical.iter().product();
        let out_strides = strides(out);
        let mut seen = vec![false; total];
        let mut coords = vec![0i64; logical.len()];
        let mut phys_coords = vec![0i64; out.len()];
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
            for (d, e) in self.exprs.iter().enumerate() {
                phys_coords[d] = e.eval(&env);
            }
            let phys: usize = phys_coords
                .iter()
                .zip(&out_strides)
                .map(|(&c, &s)| c as usize * s)
                .sum();
            if seen[phys] {
                return Err(err(format!(
                    "scatter is not bijective: physical index {phys} is hit twice"
                )));
            }
            seen[phys] = true;
            let inv_env: BTreeMap<VarId, i64> = self
                .inv_params
                .iter()
                .copied()
                .zip(phys_coords.iter().copied())
                .collect();
            for (d, e) in self.inv_exprs.iter().enumerate() {
                let back = e.eval(&inv_env);
                if back != coords[d] {
                    return Err(err(format!(
                        "scatter inverse does not invert the map: logical {coords:?} \
                         maps to physical {phys_coords:?}, whose inverse coordinate \
                         {d} is {back}"
                    )));
                }
            }
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
    fn subst_params_replaces_and_vetoes_coefficients() {
        let (p, q, i) = (v(0), v(1), v(2));
        let psym = SExpr::cst(SymConst::Sym(p));
        let compound = SExpr::sym(q).add(&SExpr::cst(SymConst::Lit(1)));

        // Bare-constant position: any image is representable.
        let m = BTreeMap::from([(p, compound.clone())]);
        assert_eq!(psym.subst_params(&m), Some(compound.clone()));
        // Unmapped params and loop syms are untouched.
        assert_eq!(psym.subst_params(&BTreeMap::new()), Some(psym.clone()));
        assert_eq!(SExpr::sym(i).subst_params(&m), Some(SExpr::sym(i)));

        // Coefficient position: a constant image folds in, a compound one
        // is unrepresentable.
        let mul = SExpr::sym(i).mul_c(SymConst::Sym(p));
        let lit = BTreeMap::from([(p, SExpr::cst(SymConst::Lit(4)))]);
        assert_eq!(
            mul.subst_params(&lit),
            Some(SExpr::sym(i).mul_c(SymConst::Lit(4)))
        );
        assert_eq!(mul.subst_params(&m), None);
        let div = SExpr::sym(i).floordiv(SymConst::Sym(p));
        assert_eq!(div.subst_params(&m), None);

        // Param-to-param renaming works in every position.
        let rename = BTreeMap::from([(p, SExpr::cst(SymConst::Sym(q)))]);
        assert_eq!(
            mul.subst_params(&rename),
            Some(SExpr::sym(i).mul_c(SymConst::Sym(q)))
        );
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

    /// Bit-reversal `Σⱼ 2^(3-j) · bitⱼ(x)` permutes `[0, 16)`; the digit
    /// decomposition sees each bit independently even through the nested
    /// `(x/2)/2` atoms the `%`-expansions produce.
    #[test]
    fn range_bit_reversal_tight() {
        let x = Quast::sym(v(1));
        let mut e = Quast::cst(0);
        for j in 0..4 {
            let bit = x.floordiv(1 << j).rem_c(2);
            e = e.add(&bit.mul_c(1 << (3 - j)));
        }
        let bounds = BTreeMap::from([(v(1), 16u64)]);
        assert_eq!(e.range(&bounds), Some((0, 15)));
    }

    /// A mixed-radix transpose-style permutation of `[0, 540)` gets the
    /// exact interval (naive per-term arithmetic would overshoot both ends).
    #[test]
    fn range_mixed_radix_tight() {
        let x = Quast::sym(v(1));
        let e = x
            .rem_c(9)
            .mul_c(12)
            .add(&x.floordiv(9).rem_c(12))
            .add(&x.floordiv(108).mul_c(108));
        let bounds = BTreeMap::from([(v(1), 540u64)]);
        assert_eq!(e.range(&bounds), Some((0, 539)));
    }

    /// Divisors that do not chain (2 and 3) fall back to per-term interval
    /// arithmetic.
    #[test]
    fn range_non_chain_falls_back() {
        let x = Quast::sym(v(1));
        let e = x.floordiv(2).add(&x.floordiv(3));
        let bounds = BTreeMap::from([(v(1), 12u64)]);
        assert_eq!(e.range(&bounds), Some((0, 8)));
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
        let (i, j, a, b) = (v(0), v(1), v(2), v(3));
        let mut sc = Scatter {
            params: vec![i, j],
            exprs: vec![Quast::sym(j), Quast::sym(i)],
            inv_params: vec![a, b],
            inv_exprs: vec![Quast::sym(b), Quast::sym(a)],
            out_shape: Some(vec![4, 3]),
            bounds: BTreeMap::new(),
        };
        assert_eq!(sc.bind_and_validate(&[3, 4]).unwrap(), vec![4, 3]);

        // i -> i / 2 is not injective.
        let mut sc = Scatter {
            params: vec![i],
            exprs: vec![Quast::sym(i).floordiv(2)],
            inv_params: vec![a],
            inv_exprs: vec![Quast::sym(a)],
            out_shape: None,
            bounds: BTreeMap::new(),
        };
        assert!(sc.bind_and_validate(&[4]).is_err());

        // A wrong inverse is rejected even when the map is bijective.
        let mut sc = Scatter {
            params: vec![i, j],
            exprs: vec![Quast::sym(j), Quast::sym(i)],
            inv_params: vec![a, b],
            inv_exprs: vec![Quast::sym(a), Quast::sym(b)],
            out_shape: Some(vec![4, 3]),
            bounds: BTreeMap::new(),
        };
        assert!(sc.bind_and_validate(&[3, 4]).is_err());

        // Size mismatch is rejected.
        let mut sc = Scatter {
            params: vec![i],
            exprs: vec![Quast::sym(i).rem_c(4), Quast::sym(i).floordiv(4)],
            inv_params: vec![a, b],
            inv_exprs: vec![Quast::sym(b).mul_c(4).add(&Quast::sym(a))],
            out_shape: Some(vec![4, 4]),
            bounds: BTreeMap::new(),
        };
        assert!(sc.bind_and_validate(&[12]).is_err());
    }

    #[test]
    fn store_map_composition() {
        // Nested transpose: logical [10, 7], scatter (i, j) -> (j, i) into
        // [7, 10]; store(f) = (f % 7) * 10 + f / 7.
        let (i, j, f, a, b) = (v(0), v(1), v(2), v(3), v(4));
        let mut sc = Scatter {
            params: vec![i, j],
            exprs: vec![Quast::sym(j), Quast::sym(i)],
            inv_params: vec![a, b],
            inv_exprs: vec![Quast::sym(b), Quast::sym(a)],
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

    #[test]
    fn sexpr_from_quast_roundtrips() {
        let q = Quast::sym(v(0)).mul_c(3).add(&Quast::cst(7)).floordiv(2);
        let s: SExpr = (&q).into();
        assert_eq!(s.try_to_quast(), Some(q));
    }

    #[test]
    fn sexpr_eval_with_params() {
        // (n - 1) - i with n = 8, i = 2.
        let n = SymConst::Sym(v(9));
        let e = SExpr::cst(n)
            .sub(&SExpr::cst(1.into()))
            .sub(&SExpr::sym(v(0)));
        let env = BTreeMap::from([(v(9), 8), (v(0), 2)]);
        assert_eq!(e.eval(&env), 5);
    }

    #[test]
    fn sexpr_concretize_binds_params_only() {
        // n*i + (n - 1) with param n = v(9) and loop var i = v(0).
        let n = SymConst::Sym(v(9));
        let e = SExpr::sym(v(0))
            .mul_c(n)
            .add(&SExpr::cst(n).sub(&SExpr::cst(1.into())));

        let mut params = BTreeSet::new();
        e.param_syms(&mut params);
        assert_eq!(params, BTreeSet::from([v(9)]));
        let mut loops = BTreeSet::new();
        e.syms(&mut loops);
        assert_eq!(loops, BTreeSet::from([v(0)]));

        // Unbound param: no Quast yet.
        assert_eq!(e.try_to_quast(), None);
        assert_eq!(e.try_concretize(&BTreeMap::new()), None);

        // A binding for the loop var must not leak into param positions.
        let env = BTreeMap::from([(v(0), 100), (v(9), 4)]);
        let q = e.try_concretize(&env).unwrap();
        let mut syms = BTreeSet::new();
        q.syms(&mut syms);
        assert_eq!(syms, BTreeSet::from([v(0)]));
        // i = 5: 4*5 + 3 = 23.
        assert_eq!(q.eval(&BTreeMap::from([(v(0), 5)])), 23);
    }

    #[test]
    fn sexpr_fold_lits() {
        use SymConst::Lit;
        let n = SymConst::Sym(v(9));
        // (2 + 3) folds; n-scaled terms survive untouched.
        let e = SExpr::cst(Lit(2)).add(&SExpr::cst(Lit(3)));
        assert_eq!(e.fold_lits(), SExpr::Const(Lit(5)));
        // x * 1 and x + 0 erase; x * 0 collapses.
        let x = SExpr::sym(v(0));
        assert_eq!(x.mul_c(Lit(1)).fold_lits(), x);
        assert_eq!(x.add(&SExpr::cst(Lit(0))).fold_lits(), x);
        assert_eq!(x.mul_c(Lit(0)).fold_lits(), SExpr::Const(Lit(0)));
        assert_eq!(x.floordiv(Lit(1)).fold_lits(), x);
        assert_eq!(x.neg().neg().fold_lits(), x);
        // Symbolic coefficients block folding but literal subterms inside
        // still fold: (4 + 4) * n -> 8 * n.
        let e = SExpr::cst(Lit(4)).add(&SExpr::cst(Lit(4))).mul_c(n);
        assert_eq!(e.fold_lits(), SExpr::Const(Lit(8)).mul_c(n));
        // Literal floordiv folds euclidean.
        let e = SExpr::cst(Lit(-7)).floordiv(Lit(2));
        assert_eq!(e.fold_lits(), SExpr::Const(Lit(-4)));
    }
}
