//! The `kernel!` proc macro: a concrete syntax for the crypto-compiler DSL.
//!
//! ```ignore
//! let y = kernel!(ib,
//!     compute [n] |i| {
//!         let v = a[i];
//!         if v < 10 then v * 2bb else v + 1bb
//!     }
//! );
//! ```
//!
//! Grammar (expressions produce `NodeId`s via `IRBuilder` method calls):
//!
//! - `let v = e; e'` (or `let v = e in e'`) — let binding (`IRBuilder::bind`);
//! - `if c then e else e'` — select (both branches are evaluated);
//! - `compute [bound] |i| { e }` / `reduce [bound] |i| { e }` — the parallel primitives; `bound` is
//!   a host Rust expression;
//! - `t[i, j, ...]` — tensor indexing;
//! - `+ - * / % < <= ==` with the usual precedence;
//! - `(a, b, ...)` — tuple; `[a, b, ...]` — pack (array literal);
//! - `17` / `17u32` — u32 constant; `17bb` — BabyBear constant;
//! - `#x` / `#(expr)` — u32 constant from a host Rust expression;
//! - `foo(a, b)` — calls the Rust function `foo(builder, a, b)`: a function used with n arguments
//!   must take n+1, the builder first. Identifier arguments are passed through verbatim (so host
//!   values of any type can be forwarded); all other arguments are built as DSL expressions.
//! - any other identifier — a Rust variable of type `NodeId` in scope (a module input or a
//!   previously built expression).
//!
//! The first macro argument is the `IRBuilder` place (e.g. `ib`); the macro
//! expands to an expression of type `NodeId` that borrows it mutably.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    braced, bracketed, parenthesized,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Expr, Ident, LitInt, Path, Token,
};

mod kw {
    syn::custom_keyword!(compute);
    syn::custom_keyword!(reduce);
    syn::custom_keyword!(then);
}

enum DslExpr {
    Var(Path),
    LitU32(u32),
    LitField(u32),
    /// `#x` / `#(expr)`: u32 constant from a host expression.
    Splice(Expr),
    Bin {
        method: &'static str,
        lhs: Box<DslExpr>,
        rhs: Box<DslExpr>,
    },
    If {
        cond: Box<DslExpr>,
        then_val: Box<DslExpr>,
        else_val: Box<DslExpr>,
    },
    Index {
        tensor: Box<DslExpr>,
        indices: Vec<DslExpr>,
    },
    Tuple(Vec<DslExpr>),
    Pack(Vec<DslExpr>),
    Call {
        path: Path,
        args: Vec<DslExpr>,
    },
    Let {
        var: Ident,
        value: Box<DslExpr>,
        body: Box<DslExpr>,
    },
    Compute {
        bound: Expr,
        var: Ident,
        body: Box<DslExpr>,
    },
    Reduce {
        bound: Expr,
        var: Ident,
        body: Box<DslExpr>,
    },
}

impl Parse for DslExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        parse_expr(input)
    }
}

fn parse_expr(input: ParseStream) -> syn::Result<DslExpr> {
    if input.peek(Token![let]) {
        input.parse::<Token![let]>()?;
        let var: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let value = parse_expr(input)?;
        if input.peek(Token![in]) {
            input.parse::<Token![in]>()?;
        } else {
            input.parse::<Token![;]>()?;
        }
        let body = parse_expr(input)?;
        return Ok(DslExpr::Let {
            var,
            value: Box::new(value),
            body: Box::new(body),
        });
    }
    if input.peek(Token![if]) {
        input.parse::<Token![if]>()?;
        let cond = parse_cmp(input)?;
        input.parse::<kw::then>()?;
        let then_val = parse_expr(input)?;
        input.parse::<Token![else]>()?;
        let else_val = parse_expr(input)?;
        return Ok(DslExpr::If {
            cond: Box::new(cond),
            then_val: Box::new(then_val),
            else_val: Box::new(else_val),
        });
    }
    if input.peek(kw::compute) || input.peek(kw::reduce) {
        return parse_binder(input);
    }
    parse_cmp(input)
}

/// `compute [bound] |i| { body }` or `reduce [bound] |i| { body }`.
fn parse_binder(input: ParseStream) -> syn::Result<DslExpr> {
    let is_compute = input.peek(kw::compute);
    if is_compute {
        input.parse::<kw::compute>()?;
    } else {
        input.parse::<kw::reduce>()?;
    }
    let bracket;
    bracketed!(bracket in input);
    let bound: Expr = bracket.parse()?;
    input.parse::<Token![|]>()?;
    let var: Ident = input.parse()?;
    input.parse::<Token![|]>()?;
    let brace;
    braced!(brace in input);
    let body = Box::new(parse_expr(&brace)?);
    if !brace.is_empty() {
        return Err(brace.error("unexpected tokens after body expression"));
    }
    Ok(if is_compute {
        DslExpr::Compute { bound, var, body }
    } else {
        DslExpr::Reduce { bound, var, body }
    })
}

fn parse_cmp(input: ParseStream) -> syn::Result<DslExpr> {
    let lhs = parse_add(input)?;
    let method = if input.peek(Token![<=]) {
        input.parse::<Token![<=]>()?;
        "le"
    } else if input.peek(Token![<]) {
        input.parse::<Token![<]>()?;
        "lt"
    } else if input.peek(Token![==]) {
        input.parse::<Token![==]>()?;
        "eq"
    } else {
        return Ok(lhs);
    };
    let rhs = parse_add(input)?;
    Ok(DslExpr::Bin {
        method,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    })
}

fn parse_add(input: ParseStream) -> syn::Result<DslExpr> {
    let mut lhs = parse_mul(input)?;
    loop {
        let method = if input.peek(Token![+]) {
            input.parse::<Token![+]>()?;
            "add"
        } else if input.peek(Token![-]) {
            input.parse::<Token![-]>()?;
            "sub"
        } else {
            return Ok(lhs);
        };
        let rhs = parse_mul(input)?;
        lhs = DslExpr::Bin {
            method,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }
}

fn parse_mul(input: ParseStream) -> syn::Result<DslExpr> {
    let mut lhs = parse_postfix(input)?;
    loop {
        let method = if input.peek(Token![*]) {
            input.parse::<Token![*]>()?;
            "mul"
        } else if input.peek(Token![/]) {
            input.parse::<Token![/]>()?;
            "div"
        } else if input.peek(Token![%]) {
            input.parse::<Token![%]>()?;
            "rem"
        } else {
            return Ok(lhs);
        };
        let rhs = parse_postfix(input)?;
        lhs = DslExpr::Bin {
            method,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }
}

/// Postfix indexing: `atom[i, j] [k] ...`.
fn parse_postfix(input: ParseStream) -> syn::Result<DslExpr> {
    let mut e = parse_atom(input)?;
    while input.peek(syn::token::Bracket) {
        let bracket;
        bracketed!(bracket in input);
        let indices: Punctuated<DslExpr, Token![,]> =
            bracket.parse_terminated(DslExpr::parse, Token![,])?;
        e = DslExpr::Index {
            tensor: Box::new(e),
            indices: indices.into_iter().collect(),
        };
    }
    Ok(e)
}

fn parse_atom(input: ParseStream) -> syn::Result<DslExpr> {
    if input.peek(LitInt) {
        let lit: LitInt = input.parse()?;
        let value: u32 = lit.base10_parse()?;
        return match lit.suffix() {
            "" | "u32" => Ok(DslExpr::LitU32(value)),
            "bb" => Ok(DslExpr::LitField(value)),
            other => Err(syn::Error::new(
                lit.span(),
                format!("unknown literal suffix `{other}`; use `u32` (or none) or `bb`"),
            )),
        };
    }
    if input.peek(Token![#]) {
        input.parse::<Token![#]>()?;
        let expr: Expr = if input.peek(syn::token::Paren) {
            let paren;
            parenthesized!(paren in input);
            paren.parse()?
        } else {
            let ident: Ident = input.parse()?;
            syn::parse_quote!(#ident)
        };
        return Ok(DslExpr::Splice(expr));
    }
    if input.peek(syn::token::Paren) {
        let paren;
        parenthesized!(paren in input);
        let elems: Punctuated<DslExpr, Token![,]> =
            paren.parse_terminated(DslExpr::parse, Token![,])?;
        let trailing = elems.trailing_punct();
        let mut elems: Vec<DslExpr> = elems.into_iter().collect();
        return match (elems.len(), trailing) {
            (0, _) => Err(paren.error("expected an expression")),
            (1, false) => Ok(elems.pop().unwrap()),
            _ => Ok(DslExpr::Tuple(elems)),
        };
    }
    if input.peek(syn::token::Bracket) {
        let bracket;
        bracketed!(bracket in input);
        let elems: Punctuated<DslExpr, Token![,]> =
            bracket.parse_terminated(DslExpr::parse, Token![,])?;
        return Ok(DslExpr::Pack(elems.into_iter().collect()));
    }
    // `parse_mod_style` rejects generic arguments, so a following `<` is
    // parsed as the comparison operator rather than as `path<...>`.
    let path = Path::parse_mod_style(input)?;
    if input.peek(syn::token::Paren) {
        let paren;
        parenthesized!(paren in input);
        let args: Punctuated<DslExpr, Token![,]> =
            paren.parse_terminated(DslExpr::parse, Token![,])?;
        return Ok(DslExpr::Call {
            path,
            args: args.into_iter().collect(),
        });
    }
    Ok(DslExpr::Var(path))
}

/// Generates a Rust expression of type `NodeId`. The builder is always in
/// scope as `__cc_b: &mut IRBuilder`; closures introduced for binders rebind
/// the same name, and operands are hoisted into temporaries so the builder
/// is only borrowed for one call at a time.
fn gen_expr(e: &DslExpr) -> TokenStream2 {
    match e {
        DslExpr::Var(path) => quote!(#path),
        DslExpr::LitU32(v) => quote!(__cc_b.const_u32(#v)),
        DslExpr::LitField(v) => quote!(__cc_b.const_field(#v)),
        DslExpr::Splice(expr) => quote!(__cc_b.const_u32((#expr) as u32)),
        DslExpr::Bin { method, lhs, rhs } => {
            let m = format_ident!("{method}");
            let l = gen_expr(lhs);
            let r = gen_expr(rhs);
            quote!({
                let __cc_l = #l;
                let __cc_r = #r;
                __cc_b.#m(__cc_l, __cc_r)
            })
        }
        DslExpr::If {
            cond,
            then_val,
            else_val,
        } => {
            let c = gen_expr(cond);
            let t = gen_expr(then_val);
            let e = gen_expr(else_val);
            quote!({
                let __cc_c = #c;
                let __cc_t = #t;
                let __cc_e = #e;
                __cc_b.select(__cc_c, __cc_t, __cc_e)
            })
        }
        DslExpr::Index { tensor, indices } => {
            let t = gen_expr(tensor);
            let (names, values) = temps("__cc_i", indices);
            quote!({
                let __cc_t = #t;
                #(let #names = #values;)*
                __cc_b.index(__cc_t, &[#(#names),*])
            })
        }
        DslExpr::Tuple(elems) => {
            let (names, values) = temps("__cc_e", elems);
            quote!({
                #(let #names = #values;)*
                __cc_b.tuple(&[#(#names),*])
            })
        }
        DslExpr::Pack(elems) => {
            let (names, values) = temps("__cc_e", elems);
            quote!({
                #(let #names = #values;)*
                __cc_b.pack(&[#(#names),*])
            })
        }
        DslExpr::Call { path, args } => {
            // Bare identifiers are forwarded verbatim so host values of any
            // type can be passed; other arguments are DSL expressions.
            let mut lets = Vec::new();
            let mut arg_toks = Vec::new();
            for (k, arg) in args.iter().enumerate() {
                if let DslExpr::Var(p) = arg {
                    arg_toks.push(quote!(#p));
                } else {
                    let name = format_ident!("__cc_a{k}");
                    let value = gen_expr(arg);
                    lets.push(quote!(let #name = #value;));
                    arg_toks.push(quote!(#name));
                }
            }
            quote!({
                #(#lets)*
                #path(&mut *__cc_b, #(#arg_toks),*)
            })
        }
        DslExpr::Let { var, value, body } => {
            let v = gen_expr(value);
            let b = gen_expr(body);
            quote!({
                let __cc_v = #v;
                __cc_b.bind(__cc_v, |__cc_b, #var| #b)
            })
        }
        DslExpr::Compute { bound, var, body } => {
            let b = gen_expr(body);
            quote!(__cc_b.compute((#bound) as usize, |__cc_b, #var| #b))
        }
        DslExpr::Reduce { bound, var, body } => {
            let b = gen_expr(body);
            quote!(__cc_b.reduce_add((#bound) as usize, |__cc_b, #var| #b))
        }
    }
}

fn temps(prefix: &str, elems: &[DslExpr]) -> (Vec<Ident>, Vec<TokenStream2>) {
    let names = (0..elems.len())
        .map(|k| format_ident!("{prefix}{k}"))
        .collect();
    let values = elems.iter().map(gen_expr).collect();
    (names, values)
}

struct KernelInput {
    ctx: Expr,
    expr: DslExpr,
}

impl Parse for KernelInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ctx: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let expr = parse_expr(input)?;
        if !input.is_empty() {
            input.parse::<Token![,]>()?;
        }
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after kernel expression"));
        }
        Ok(Self { ctx, expr })
    }
}

/// Builds a DSL expression on an `IRBuilder`; see the crate docs for the
/// grammar. `kernel!(ib, <expr>)` evaluates to the expression's `NodeId`.
#[proc_macro]
pub fn kernel(input: TokenStream) -> TokenStream {
    let KernelInput { ctx, expr } = parse_macro_input!(input as KernelInput);
    let body = gen_expr(&expr);
    quote!({
        let __cc_b = #ctx.as_builder_mut();
        #body
    })
    .into()
}
