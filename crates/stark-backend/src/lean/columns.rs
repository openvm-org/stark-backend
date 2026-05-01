//! Per-AIR column descriptions for Lean extraction.
//!
//! A `Cols` struct deriving `LeanColumns` exposes a flat list of columns
//! plus opaque sub-air slots. The new-dialect emitter wants a flat
//! `Vec<String>` of length `width`, so [`flatten_lean_columns`] rolls the
//! sub-air entries up into prefixed columns when the caller can resolve
//! the inner type's column list.

/// An entry produced by `#[derive(LeanColumns)]`.
#[derive(Debug, Clone)]
pub enum LeanEntry {
    /// A single named column.
    Column(String),
    /// A nested sub-air. The inner type's columns are not known to the
    /// derive macro; callers either resolve `type_name` via a registry or
    /// fall back to numeric placeholders (`<field_name>_0` ..
    /// `<field_name>_{width-1}`).
    SubAir {
        field_name: String,
        type_name: String,
        width: usize,
    },
}

/// Trait implemented by `#[derive(LeanColumns)]` on `Cols` structs.
pub trait LeanColumns {
    fn lean_columns() -> Vec<LeanEntry>;
}

/// Flatten `entries` into a `Vec<String>` of column names. `resolver`
/// maps a sub-air type name to its own `Vec<LeanEntry>` (typically
/// `<InnerType as LeanColumns>::lean_columns()`); when it returns
/// `None`, the sub-air slot is filled with numeric placeholders
/// `<field_name>_0` .. `<field_name>_{width-1}`.
pub fn flatten_lean_columns<R>(entries: Vec<LeanEntry>, resolver: &R) -> Vec<String>
where
    R: Fn(&str) -> Option<Vec<LeanEntry>>,
{
    let mut out = Vec::new();
    flatten_into(entries, "", resolver, &mut out);
    out
}

fn flatten_into<R>(entries: Vec<LeanEntry>, prefix: &str, resolver: &R, out: &mut Vec<String>)
where
    R: Fn(&str) -> Option<Vec<LeanEntry>>,
{
    for entry in entries {
        match entry {
            LeanEntry::Column(name) => out.push(prefixed(prefix, &name)),
            LeanEntry::SubAir {
                field_name,
                type_name,
                width,
            } => {
                let combined = prefixed(prefix, &field_name);
                if let Some(inner) = resolver(&type_name) {
                    flatten_into(inner, &combined, resolver, out);
                } else {
                    for i in 0..width {
                        out.push(format!("{combined}_{i}"));
                    }
                }
            }
        }
    }
}

fn prefixed(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}_{name}")
    }
}

/// Convenience wrapper: flatten the column list of a `LeanColumns`
/// implementor with a no-op resolver. Sub-air entries become numeric
/// placeholders.
pub fn flat_columns_of<C: LeanColumns>() -> Vec<String> {
    flatten_lean_columns(C::lean_columns(), &|_| None)
}
