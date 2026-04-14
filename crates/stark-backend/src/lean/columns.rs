/// An entry produced by `#[derive(LeanColumns)]`.
#[derive(Debug, Clone)]
pub enum LeanEntry {
    /// A single column: `Column["name"]`
    Column(String),
    /// A nested sub-air: `MainSubAir["field": "TypeName" width := N]`
    SubAir {
        field_name: String,
        type_name: String,
        width: usize,
    },
}

/// Trait implemented by `#[derive(LeanColumns)]` on Cols structs.
pub trait LeanColumns {
    fn lean_columns() -> Vec<LeanEntry>;
}

/// Generates a Lean4 `#define_air` block from any type implementing `LeanColumns`.
pub fn generate_lean_air_definition<C: LeanColumns>(air_name: &str) -> String {
    let entries = C::lean_columns();
    let mut lines = vec![format!(
        "#define_air \"{air_name}\" using \"openvm_encapsulation\" where"
    )];
    for entry in &entries {
        match entry {
            LeanEntry::Column(name) => {
                lines.push(format!("  Column[\"{name}\"]"));
            }
            LeanEntry::SubAir {
                field_name,
                type_name,
                width,
            } => {
                lines.push(format!(
                    "  MainSubAir[\"{field_name}\": \"{type_name}\" width := {width}]"
                ));
            }
        }
    }
    lines.join("\n")
}
