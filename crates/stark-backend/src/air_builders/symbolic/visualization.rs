use std::collections::{BTreeMap, HashMap, HashSet};

use serde::Serialize;

use super::{SymbolicConstraintsDag, SymbolicExpressionDag, SymbolicExpressionNode};
use crate::air_builders::symbolic::symbolic_variable::SymbolicVariable;

/// Column name mapping: (part_index, offset, column_index) -> name
pub type ColumnNameMap = HashMap<(usize, usize, usize), String>;

#[derive(Serialize)]
pub struct DagJson {
    pub nodes: Vec<NodeJson>,
    pub constraints: Vec<usize>,
    pub stats: DagStats,
    /// Number of monomials per constraint (after combining like terms).
    pub monomial_counts: Vec<usize>,
    /// Formatted monomial strings per constraint (precomputed in Rust).
    pub monomial_strings: Vec<Vec<String>>,
}

#[derive(Serialize)]
pub struct NodeJson {
    pub id: usize,
    #[serde(rename = "type")]
    pub node_type: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<usize>>,
    pub depth: usize,
    pub ref_count: usize,
}

#[derive(Serialize)]
pub struct DagStats {
    pub total_nodes: usize,
    pub num_constraints: usize,
    pub num_variables: usize,
    pub num_constants: usize,
    pub num_intermediates: usize,
    pub max_depth: usize,
    /// Total monomials over F (sum across all constraints, treating λ-batching as disjoint)
    pub monomials_over_f: usize,
    /// Unique monomials over F[λ] (deduplicated variable combinations across constraints)
    pub monomials_over_f_lambda: usize,
}

use crate::p3_field::Field;

/// A polynomial represented as a map from monomial (variable structure) to coefficient.
/// The key is a canonical string representation of the variables and their exponents.
/// This representation automatically combines like terms.
type Polynomial<F> = HashMap<BTreeMap<String, usize>, F>;

fn poly_constant<F: Field>(c: F) -> Polynomial<F> {
    let mut p = HashMap::new();
    if c != F::ZERO {
        p.insert(BTreeMap::new(), c);
    }
    p
}

fn poly_variable<F: Field>(name: String) -> Polynomial<F> {
    let mut vars = BTreeMap::new();
    vars.insert(name, 1);
    let mut p = HashMap::new();
    p.insert(vars, F::ONE);
    p
}

fn poly_add<F: Field>(mut a: Polynomial<F>, b: Polynomial<F>) -> Polynomial<F> {
    for (vars, coeff) in b {
        *a.entry(vars).or_insert(F::ZERO) += coeff;
    }
    // Remove zero coefficients
    a.retain(|_, c| *c != F::ZERO);
    a
}

fn poly_sub<F: Field>(mut a: Polynomial<F>, b: Polynomial<F>) -> Polynomial<F> {
    for (vars, coeff) in b {
        *a.entry(vars).or_insert(F::ZERO) -= coeff;
    }
    a.retain(|_, c| *c != F::ZERO);
    a
}

fn poly_neg<F: Field>(mut p: Polynomial<F>) -> Polynomial<F> {
    for coeff in p.values_mut() {
        *coeff = -*coeff;
    }
    p
}

fn poly_mul<F: Field>(a: &Polynomial<F>, b: &Polynomial<F>) -> Polynomial<F> {
    let mut result: Polynomial<F> = HashMap::new();
    for (vars_a, coeff_a) in a {
        for (vars_b, coeff_b) in b {
            // Combine variable maps
            let mut combined_vars = vars_a.clone();
            for (var, exp) in vars_b {
                *combined_vars.entry(var.clone()).or_insert(0) += exp;
            }
            *result.entry(combined_vars).or_insert(F::ZERO) += *coeff_a * *coeff_b;
        }
    }
    result.retain(|_, c| *c != F::ZERO);
    result
}

fn expand_to_polynomial<F: Field>(
    node_idx: usize,
    nodes: &[SymbolicExpressionNode<F>],
    col_names: Option<&ColumnNameMap>,
    cache: &mut HashMap<usize, Polynomial<F>>,
) -> Polynomial<F> {
    if let Some(cached) = cache.get(&node_idx) {
        return cached.clone();
    }

    let result = match &nodes[node_idx] {
        SymbolicExpressionNode::Constant(c) => poly_constant(*c),
        SymbolicExpressionNode::Variable(v) => {
            let name = format_variable(v, col_names);
            poly_variable(name)
        }
        SymbolicExpressionNode::IsFirstRow => poly_variable("is_first".to_string()),
        SymbolicExpressionNode::IsLastRow => poly_variable("is_last".to_string()),
        SymbolicExpressionNode::IsTransition => poly_variable("is_trans".to_string()),
        SymbolicExpressionNode::Add {
            left_idx,
            right_idx,
            ..
        } => {
            let left = expand_to_polynomial(*left_idx, nodes, col_names, cache);
            let right = expand_to_polynomial(*right_idx, nodes, col_names, cache);
            poly_add(left, right)
        }
        SymbolicExpressionNode::Sub {
            left_idx,
            right_idx,
            ..
        } => {
            let left = expand_to_polynomial(*left_idx, nodes, col_names, cache);
            let right = expand_to_polynomial(*right_idx, nodes, col_names, cache);
            poly_sub(left, right)
        }
        SymbolicExpressionNode::Neg { idx, .. } => {
            let p = expand_to_polynomial(*idx, nodes, col_names, cache);
            poly_neg(p)
        }
        SymbolicExpressionNode::Mul {
            left_idx,
            right_idx,
            ..
        } => {
            let left = expand_to_polynomial(*left_idx, nodes, col_names, cache);
            let right = expand_to_polynomial(*right_idx, nodes, col_names, cache);
            poly_mul(&left, &right)
        }
    };

    cache.insert(node_idx, result.clone());
    result
}

fn format_monomial<F: Field>(vars: &BTreeMap<String, usize>, coeff: F) -> String {
    let coeff_str = format!("{:?}", coeff);

    if vars.is_empty() {
        return coeff_str;
    }

    let mut result = String::new();
    if coeff == F::ONE {
        // Don't show coefficient of 1
    } else if coeff == -F::ONE {
        result.push('-');
    } else {
        result.push_str(&coeff_str);
        result.push_str(" · ");
    }

    for (i, (var, exp)) in vars.iter().enumerate() {
        if i > 0 {
            result.push_str(" · ");
        }
        if *exp == 1 {
            result.push_str(var);
        } else {
            result.push_str(&format!("{}^{}", var, exp));
        }
    }
    result
}

/// Convert variable map to a string key for deduplication
fn var_key(vars: &BTreeMap<String, usize>) -> String {
    vars.iter()
        .map(|(v, e)| format!("{}^{}", v, e))
        .collect::<Vec<_>>()
        .join("|")
}

/// Compute global monomial statistics for a DAG
/// Returns (total_over_f, unique_over_f_lambda, per_constraint_counts, per_constraint_strings)
fn compute_monomial_stats<F: Field>(
    dag: &SymbolicExpressionDag<F>,
    col_names: Option<&ColumnNameMap>,
) -> (usize, usize, Vec<usize>, Vec<Vec<String>>) {
    let mut cache = HashMap::new();
    let mut total_over_f = 0usize;
    let mut unique_var_keys: HashSet<String> = HashSet::new();
    let mut per_constraint_counts = Vec::with_capacity(dag.constraint_idx.len());
    let mut per_constraint_strings = Vec::with_capacity(dag.constraint_idx.len());

    for &constraint_idx in &dag.constraint_idx {
        let poly = expand_to_polynomial(constraint_idx, &dag.nodes, col_names, &mut cache);
        let count = poly.len();
        per_constraint_counts.push(count);
        total_over_f += count;

        // Track unique variable combinations
        for vars in poly.keys() {
            unique_var_keys.insert(var_key(vars));
        }

        // Format monomials as strings
        let strings: Vec<String> = poly
            .iter()
            .map(|(vars, coeff)| format_monomial(vars, *coeff))
            .collect();
        per_constraint_strings.push(strings);
    }

    (
        total_over_f,
        unique_var_keys.len(),
        per_constraint_counts,
        per_constraint_strings,
    )
}

/// Export a symbolic constraints DAG to JSON format for visualization.
pub fn export_dag_to_json<F: Field>(dag: &SymbolicConstraintsDag<F>) -> DagJson {
    export_dag_to_json_with_names(dag, None)
}

/// Export a symbolic constraints DAG to JSON format with optional column names.
pub fn export_dag_to_json_with_names<F: Field>(
    dag: &SymbolicConstraintsDag<F>,
    col_names: Option<&ColumnNameMap>,
) -> DagJson {
    export_expression_dag_to_json_with_names(&dag.constraints, col_names)
}

/// Export a symbolic expression DAG to JSON format for visualization.
pub fn export_expression_dag_to_json<F: Field>(dag: &SymbolicExpressionDag<F>) -> DagJson {
    export_expression_dag_to_json_with_names(dag, None)
}

/// Export a symbolic expression DAG to JSON format with optional column names.
pub fn export_expression_dag_to_json_with_names<F: Field>(
    dag: &SymbolicExpressionDag<F>,
    col_names: Option<&ColumnNameMap>,
) -> DagJson {
    let nodes = &dag.nodes;

    // Compute depths for all nodes
    let depths = compute_depths(nodes);

    // Compute reference counts
    let ref_counts = compute_ref_counts(nodes, &dag.constraint_idx);

    // Count node types
    let mut num_variables = 0;
    let mut num_constants = 0;
    let mut num_intermediates = 0;

    let json_nodes: Vec<NodeJson> = nodes
        .iter()
        .enumerate()
        .map(|(id, node)| {
            let (node_type, label, children) = match node {
                SymbolicExpressionNode::Variable(v) => {
                    num_variables += 1;
                    ("Variable".to_string(), format_variable(v, col_names), None)
                }
                SymbolicExpressionNode::Constant(c) => {
                    num_constants += 1;
                    ("Constant".to_string(), format!("{:?}", c), None)
                }
                SymbolicExpressionNode::IsFirstRow => {
                    num_constants += 1;
                    ("IsFirstRow".to_string(), "is_first".to_string(), None)
                }
                SymbolicExpressionNode::IsLastRow => {
                    num_constants += 1;
                    ("IsLastRow".to_string(), "is_last".to_string(), None)
                }
                SymbolicExpressionNode::IsTransition => {
                    num_constants += 1;
                    ("IsTransition".to_string(), "is_trans".to_string(), None)
                }
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    num_intermediates += 1;
                    (
                        "Add".to_string(),
                        "+".to_string(),
                        Some(vec![*left_idx, *right_idx]),
                    )
                }
                SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    num_intermediates += 1;
                    (
                        "Sub".to_string(),
                        "-".to_string(),
                        Some(vec![*left_idx, *right_idx]),
                    )
                }
                SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    num_intermediates += 1;
                    (
                        "Mul".to_string(),
                        "*".to_string(),
                        Some(vec![*left_idx, *right_idx]),
                    )
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    num_intermediates += 1;
                    ("Neg".to_string(), "-".to_string(), Some(vec![*idx]))
                }
            };

            NodeJson {
                id,
                node_type,
                label,
                children,
                depth: depths[id],
                ref_count: ref_counts[id],
            }
        })
        .collect();

    let max_depth = depths.iter().copied().max().unwrap_or(0);

    // Compute monomial statistics
    let (monomials_over_f, monomials_over_f_lambda, monomial_counts, monomial_strings) =
        compute_monomial_stats(dag, col_names);

    DagJson {
        nodes: json_nodes,
        constraints: dag.constraint_idx.clone(),
        stats: DagStats {
            total_nodes: nodes.len(),
            num_constraints: dag.constraint_idx.len(),
            num_variables,
            num_constants,
            num_intermediates,
            max_depth,
            monomials_over_f,
            monomials_over_f_lambda,
        },
        monomial_counts,
        monomial_strings,
    }
}

fn format_variable<F>(v: &SymbolicVariable<F>, col_names: Option<&ColumnNameMap>) -> String {
    use crate::air_builders::symbolic::symbolic_variable::Entry;

    // Try to look up a friendly name for main trace variables
    if let Some(names) = col_names {
        if let Entry::Main { part_index, offset } = v.entry {
            if let Some(name) = names.get(&(part_index, offset, v.index)) {
                let prefix = if offset == 0 { "local" } else { "next" };
                return format!("{}.{}", prefix, name);
            }
        }
    }

    // Fall back to generic format
    match v.entry {
        Entry::Preprocessed { offset } => format!("prep[{}][{}]", offset, v.index),
        Entry::Main { part_index, offset } => {
            format!("main[{}][{}][{}]", part_index, offset, v.index)
        }
        Entry::Permutation { offset } => format!("perm[{}][{}]", offset, v.index),
        Entry::Public => format!("pub[{}]", v.index),
        Entry::Challenge => format!("chal[{}]", v.index),
        Entry::Exposed => format!("exp[{}]", v.index),
    }
}

fn compute_depths<F>(nodes: &[SymbolicExpressionNode<F>]) -> Vec<usize> {
    let mut depths = vec![0; nodes.len()];

    for (i, node) in nodes.iter().enumerate() {
        depths[i] = match node {
            // Leaf nodes have depth 0 (no operations)
            SymbolicExpressionNode::Variable(_)
            | SymbolicExpressionNode::Constant(_)
            | SymbolicExpressionNode::IsFirstRow
            | SymbolicExpressionNode::IsLastRow
            | SymbolicExpressionNode::IsTransition => 0,
            // Operations add 1 to the max depth of their children
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            }
            | SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            }
            | SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => 1 + depths[*left_idx].max(depths[*right_idx]),
            SymbolicExpressionNode::Neg { idx, .. } => 1 + depths[*idx],
        };
    }

    depths
}

fn compute_ref_counts<F>(
    nodes: &[SymbolicExpressionNode<F>],
    constraint_idx: &[usize],
) -> Vec<usize> {
    let mut ref_counts = vec![0; nodes.len()];

    // Count constraint roots
    for &idx in constraint_idx {
        ref_counts[idx] += 1;
    }

    // Count internal references
    for node in nodes {
        match node {
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            }
            | SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            }
            | SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => {
                ref_counts[*left_idx] += 1;
                ref_counts[*right_idx] += 1;
            }
            SymbolicExpressionNode::Neg { idx, .. } => {
                ref_counts[*idx] += 1;
            }
            _ => {}
        }
    }

    ref_counts
}

/// Compute the rule length (number of nodes in subdag) for each constraint.
fn compute_rule_lengths<F>(dag: &SymbolicExpressionDag<F>) -> Vec<usize> {
    use std::collections::HashSet;

    fn count_nodes<F>(
        idx: usize,
        nodes: &[SymbolicExpressionNode<F>],
        visited: &mut HashSet<usize>,
    ) {
        if visited.contains(&idx) {
            return;
        }
        visited.insert(idx);

        match &nodes[idx] {
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            }
            | SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            }
            | SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => {
                count_nodes(*left_idx, nodes, visited);
                count_nodes(*right_idx, nodes, visited);
            }
            SymbolicExpressionNode::Neg { idx: child_idx, .. } => {
                count_nodes(*child_idx, nodes, visited);
            }
            _ => {}
        }
    }

    dag.constraint_idx
        .iter()
        .map(|&idx| {
            let mut visited = HashSet::new();
            count_nodes(idx, &dag.nodes, &mut visited);
            visited.len()
        })
        .collect()
}

/// Generate a self-contained HTML file with embedded JSON data.
pub fn generate_html_explorer<F: Field>(dag: &SymbolicConstraintsDag<F>, title: &str) -> String {
    generate_html_explorer_with_names(dag, title, None)
}

/// Generate a self-contained HTML file with embedded JSON data and optional column names.
pub fn generate_html_explorer_with_names<F: Field>(
    dag: &SymbolicConstraintsDag<F>,
    title: &str,
    col_names: Option<&ColumnNameMap>,
) -> String {
    let json = export_dag_to_json_with_names(dag, col_names);
    let json_str = serde_json::to_string(&json).expect("Failed to serialize DAG to JSON");

    // Compute rule lengths (number of nodes in each constraint's subdag)
    let rule_lengths = compute_rule_lengths(&dag.constraints);
    let rule_lengths_json =
        serde_json::to_string(&rule_lengths).expect("Failed to serialize rule lengths");

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - DAG Explorer</title>
    <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            font-size: 14px;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        .panel {{ overflow: auto; padding: 10px; }}
        #left-panel {{ width: 400px; border-right: 1px solid #ccc; background: #f5f5f5; }}
        #right-panel {{ flex: 1; background: #fff; }}
        h2 {{ font-size: 16px; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #ccc; }}
        .stats {{ background: #e8e8e8; padding: 8px; margin-bottom: 10px; border-radius: 4px; font-size: 12px; }}
        .stats div {{ margin: 2px 0; }}
        .filter-section {{ margin-bottom: 10px; }}
        .filter-section label {{ font-weight: bold; font-size: 12px; display: block; margin-bottom: 4px; }}
        .filter-row {{ display: flex; gap: 5px; margin-bottom: 5px; flex-wrap: wrap; }}
        .filter-input {{ flex: 1; min-width: 80px; padding: 4px 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }}
        .filter-btn {{
            padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; background: #fff;
            cursor: pointer; font-size: 11px; white-space: nowrap;
        }}
        .filter-btn:hover {{ background: #e8e8e8; }}
        .filter-btn.active {{ background: #007bff; color: white; border-color: #007bff; }}
        .sort-section {{ margin-bottom: 10px; font-size: 12px; }}
        .sort-section select {{ padding: 4px; border-radius: 4px; }}
        .constraint-list {{ list-style: none; }}
        .constraint-item {{
            padding: 6px 8px; cursor: pointer; border-radius: 4px; margin-bottom: 2px;
            display: grid; grid-template-columns: 45px 35px 35px 40px 45px 1fr; gap: 4px; font-size: 11px; align-items: center;
        }}
        .constraint-item:hover {{ background: #ddd; }}
        .constraint-item.selected {{ background: #007bff; color: white; }}
        .constraint-item .idx {{ font-weight: bold; }}
        .meta-item {{ color: #666; white-space: nowrap; }}
        .constraint-item.selected .meta-item {{ color: #cce5ff; }}
        .var-name {{ color: #28a745; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .constraint-item.selected .var-name {{ color: #90EE90; }}
        .histograms {{ margin-bottom: 10px; }}
        .histogram {{ margin-bottom: 8px; }}
        .histogram-title {{ font-size: 11px; font-weight: bold; margin-bottom: 3px; color: #333; }}
        .histogram-bars {{ display: flex; align-items: flex-end; height: 40px; gap: 1px; }}
        .histogram-bar {{ background: #007bff; flex: 1 1 0; min-width: 0; cursor: pointer; transition: background 0.1s; position: relative; }}
        .histogram-bar:hover {{ background: #0056b3; }}
        .histogram-bar:hover::after {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 3px 6px;
            border-radius: 3px;
            font-size: 10px;
            white-space: nowrap;
            z-index: 100;
            pointer-events: none;
        }}
        .histogram-bar:first-child:hover::after {{ left: 0; transform: none; }}
        .histogram-bar:last-child:hover::after {{ left: auto; right: 0; transform: none; }}
        .histogram-labels {{ display: flex; justify-content: space-between; font-size: 9px; color: #666; margin-top: 2px; }}
        #constraint-title {{ margin-bottom: 10px; padding: 8px; background: #e8e8e8; border-radius: 4px; }}
        #graph-container {{ width: 100%; height: calc(100vh - 150px); border: 1px solid #ccc; border-radius: 4px; }}
        .hidden {{ display: none; }}
        .count-badge {{ background: #007bff; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; margin-left: 8px; font-weight: bold; }}
    </style>
</head>
<body>
    <div id="left-panel" class="panel">
        <h1 style="font-size: 18px; margin-bottom: 8px;">{title}</h1>
        <h2>Constraints <span class="count-badge" id="shown-count"></span></h2>
        <div class="stats" id="stats"></div>

        <div class="filter-section">
            <label>Quick Filters</label>
            <div class="filter-row">
                <button class="filter-btn" data-filter="depth" data-min="1" data-max="2">bool (d:1-2)</button>
                <button class="filter-btn" data-filter="depth" data-min="3" data-max="10">medium (d:3-10)</button>
                <button class="filter-btn" data-filter="depth" data-min="25" data-max="100">deep (d:25+)</button>
            </div>
            <div class="filter-row">
                <button class="filter-btn" data-filter="vars" data-min="1" data-max="1">1 var</button>
                <button class="filter-btn" data-filter="vars" data-min="2" data-max="10">2-10 vars</button>
                <button class="filter-btn" data-filter="vars" data-min="40" data-max="100">40+ vars</button>
            </div>
            <div class="filter-row">
                <button class="filter-btn" data-filter="len" data-min="1" data-max="10">tiny (len:1-10)</button>
                <button class="filter-btn" data-filter="len" data-min="100" data-max="1000000">large (len:100+)</button>
            </div>
            <div class="filter-row">
                <button class="filter-btn" id="clear-filter">Clear All</button>
            </div>
        </div>

        <div class="filter-section">
            <label>Custom Range Filters</label>
            <div class="filter-row">
                <input type="number" class="filter-input" id="depth-min" placeholder="depth min">
                <input type="number" class="filter-input" id="depth-max" placeholder="depth max">
            </div>
            <div class="filter-row">
                <input type="number" class="filter-input" id="vars-min" placeholder="vars min">
                <input type="number" class="filter-input" id="vars-max" placeholder="vars max">
            </div>
            <div class="filter-row">
                <input type="number" class="filter-input" id="len-min" placeholder="len min">
                <input type="number" class="filter-input" id="len-max" placeholder="len max">
            </div>
            <div class="filter-row">
                <input type="number" class="filter-input" id="mono-min" placeholder="mono min">
                <input type="number" class="filter-input" id="mono-max" placeholder="mono max">
            </div>
        </div>

        <div class="histograms">
            <div class="histogram">
                <div class="histogram-title">Depth Distribution</div>
                <div class="histogram-bars" id="hist-depth"></div>
                <div class="histogram-labels" id="hist-depth-labels"></div>
            </div>
            <div class="histogram">
                <div class="histogram-title">Variable Count Distribution</div>
                <div class="histogram-bars" id="hist-vars"></div>
                <div class="histogram-labels" id="hist-vars-labels"></div>
            </div>
            <div class="histogram">
                <div class="histogram-title">Rule Length Distribution</div>
                <div class="histogram-bars" id="hist-len"></div>
                <div class="histogram-labels" id="hist-len-labels"></div>
            </div>
            <div class="histogram">
                <div class="histogram-title">Monomial Count Distribution</div>
                <div class="histogram-bars" id="hist-mono"></div>
                <div class="histogram-labels" id="hist-mono-labels"></div>
            </div>
        </div>

        <div class="sort-section">
            <label>Sort by: </label>
            <select id="sort-select">
                <option value="index">Index</option>
                <option value="depth">Depth</option>
                <option value="vars">Variables</option>
                <option value="len">Rule Length</option>
                <option value="mono">Monomials</option>
            </select>
            <select id="sort-dir">
                <option value="asc">Ascending</option>
                <option value="desc">Descending</option>
            </select>
        </div>

        <ul class="constraint-list" id="constraint-list"></ul>
    </div>
    <div id="right-panel" class="panel">
        <h2>Constraint View</h2>
        <div id="constraint-title">Select a constraint from the left panel</div>
        <div id="monomials-section" style="margin-bottom: 10px;">
            <h3 id="monomials-header" style="font-size: 14px; margin-bottom: 5px; cursor: pointer; user-select: none;">
                ▶ Monomial Expansion <span id="monomial-count"></span>
            </h3>
            <div id="monomials-container" style="display: none; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; background: #f8f8f8; padding: 8px; border-radius: 4px; border: 1px solid #ddd;"></div>
        </div>
        <h3 style="font-size: 14px; margin-bottom: 5px;">DAG View</h3>
        <div id="graph-wrapper" style="position: relative;">
            <div id="graph-container"></div>
            <div id="loading-overlay" style="display: none; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255,255,255,0.8); z-index: 10; justify-content: center; align-items: center; font-size: 16px; color: #666;">
                Computing layout...
            </div>
        </div>
    </div>

    <script>
    const DAG_DATA = {json_str};
    const RULE_LENGTHS = {rule_lengths_json};
    const MONOMIAL_COUNTS = DAG_DATA.monomial_counts;
    const MONOMIAL_STRINGS = DAG_DATA.monomial_strings;

    const nodes = DAG_DATA.nodes;
    const constraints = DAG_DATA.constraints;
    const stats = DAG_DATA.stats;

    // Compute variables per constraint and collect variable names
    function getVarInfo(nodeId, visited = new Set()) {{
        if (visited.has(nodeId)) return [];
        visited.add(nodeId);
        const node = nodes[nodeId];
        if (node.type === 'Variable') return [node.label];
        if (!node.children) return [];
        return node.children.flatMap(c => getVarInfo(c, visited));
    }}

    const constraintInfo = constraints.map((nodeId, i) => {{
        const varNames = getVarInfo(nodeId);
        return {{
            index: i,
            nodeId: nodeId,
            depth: nodes[nodeId].depth,
            vars: varNames.length,
            len: RULE_LENGTHS[i],
            mono: MONOMIAL_COUNTS[i],
            varNames: varNames
        }};
    }});

    // Render stats
    document.getElementById('stats').innerHTML = `
        <div><b>Nodes:</b> ${{stats.total_nodes.toLocaleString()}}</div>
        <div><b>Constraints:</b> ${{stats.num_constraints.toLocaleString()}}</div>
        <div><b>Variables:</b> ${{stats.num_variables.toLocaleString()}}</div>
        <div><b>Max depth:</b> ${{stats.max_depth}}</div>
        <div><b>Monomials/F:</b> ${{stats.monomials_over_f.toLocaleString()}}</div>
        <div><b>Monomials/F[λ]:</b> ${{stats.monomials_over_f_lambda.toLocaleString()}}</div>
    `;

    // Filter state
    let filters = {{ depth: {{min: null, max: null}}, vars: {{min: null, max: null}}, len: {{min: null, max: null}}, mono: {{min: null, max: null}} }};
    let sortBy = 'index';
    let sortDir = 'asc';

    function applyFilters(c) {{
        if (filters.depth.min !== null && c.depth < filters.depth.min) return false;
        if (filters.depth.max !== null && c.depth > filters.depth.max) return false;
        if (filters.vars.min !== null && c.vars < filters.vars.min) return false;
        if (filters.vars.max !== null && c.vars > filters.vars.max) return false;
        if (filters.len.min !== null && c.len < filters.len.min) return false;
        if (filters.len.max !== null && c.len > filters.len.max) return false;
        if (filters.mono.min !== null && c.mono < filters.mono.min) return false;
        if (filters.mono.max !== null && c.mono > filters.mono.max) return false;
        return true;
    }}

    function sortConstraints(a, b) {{
        let cmp = 0;
        if (sortBy === 'index') cmp = a.index - b.index;
        else if (sortBy === 'depth') cmp = a.depth - b.depth;
        else if (sortBy === 'vars') cmp = a.vars - b.vars;
        else if (sortBy === 'len') cmp = a.len - b.len;
        else if (sortBy === 'mono') cmp = a.mono - b.mono;
        return sortDir === 'desc' ? -cmp : cmp;
    }}

    // Render constraint list
    const listEl = document.getElementById('constraint-list');
    function renderConstraintList() {{
        listEl.innerHTML = '';
        const filtered = constraintInfo.filter(applyFilters).sort(sortConstraints);
        document.getElementById('shown-count').textContent = filtered.length + ' / ' + constraintInfo.length;
        renderHistograms(filtered);

        filtered.forEach(c => {{
            const li = document.createElement('li');
            li.className = 'constraint-item';
            li.dataset.index = c.index;
            let varDisplay = '';
            if (c.varNames.length === 1) {{
                varDisplay = `<span class="var-name">${{c.varNames[0]}}</span>`;
            }} else if (c.varNames.length === 2) {{
                varDisplay = `<span class="var-name">${{c.varNames[0]}}, ${{c.varNames[1]}}</span>`;
            }} else if (c.varNames.length > 2) {{
                varDisplay = `<span class="var-name">${{c.varNames[0]}} +${{c.varNames.length - 1}}</span>`;
            }}
            li.innerHTML = `
                <span class="idx">#${{c.index}}</span>
                <span class="meta-item">d:${{c.depth}}</span>
                <span class="meta-item">v:${{c.vars}}</span>
                <span class="meta-item">l:${{c.len}}</span>
                <span class="meta-item">m:${{c.mono}}</span>
                ${{varDisplay}}
            `;
            li.onclick = () => selectConstraint(c.index);
            listEl.appendChild(li);
        }});
    }}
    // Histogram rendering
    function renderHistograms(filtered) {{
        const numBuckets = 20;

        function computeHistogram(data, maxBuckets) {{
            if (data.length === 0) return {{ buckets: [], min: 0, max: 0, bucketSize: 1 }};
            const min = Math.min(...data);
            const max = Math.max(...data);
            const range = max - min + 1;
            // Use fewer buckets if range is small
            const numBuckets = Math.min(maxBuckets, range);
            if (numBuckets <= 1) return {{ buckets: [data.length], min, max, bucketSize: range }};
            const bucketSize = range / numBuckets;
            const buckets = new Array(numBuckets).fill(0);
            data.forEach(v => {{
                const idx = Math.min(Math.floor((v - min) / bucketSize), numBuckets - 1);
                buckets[idx]++;
            }});
            return {{ buckets, min, max, bucketSize }};
        }}

        function renderHist(containerId, labelsId, data, filterType) {{
            const hist = computeHistogram(data, numBuckets);
            const container = document.getElementById(containerId);
            const labels = document.getElementById(labelsId);
            const maxCount = Math.max(...hist.buckets, 1);

            container.innerHTML = '';
            hist.buckets.forEach((count, i) => {{
                const height = (count / maxCount) * 100;
                const bucketMin = Math.round(hist.min + i * hist.bucketSize);
                const bucketMax = Math.max(bucketMin, Math.round(hist.min + (i + 1) * hist.bucketSize - 1));
                const bar = document.createElement('div');
                bar.className = 'histogram-bar';
                bar.style.height = height + '%';
                bar.dataset.tooltip = `${{bucketMin}}-${{bucketMax}}: ${{count}}`;
                bar.onclick = () => {{
                    filters[filterType] = {{ min: bucketMin, max: bucketMax }};
                    document.querySelectorAll(`.filter-btn[data-filter="${{filterType}}"]`).forEach(b => b.classList.remove('active'));
                    updateInputsFromFilters();
                    renderConstraintList();
                }};
                container.appendChild(bar);
            }});

            labels.innerHTML = `<span>${{hist.min}}</span><span>${{hist.max}}</span>`;
        }}

        renderHist('hist-depth', 'hist-depth-labels', filtered.map(c => c.depth), 'depth');
        renderHist('hist-vars', 'hist-vars-labels', filtered.map(c => c.vars), 'vars');
        renderHist('hist-len', 'hist-len-labels', filtered.map(c => c.len), 'len');
        renderHist('hist-mono', 'hist-mono-labels', filtered.map(c => c.mono), 'mono');
    }}

    renderConstraintList();

    // Add counts to quick filter buttons
    function countMatching(filterType, min, max) {{
        return constraintInfo.filter(c => {{
            const val = filterType === 'depth' ? c.depth : filterType === 'vars' ? c.vars : c.len;
            return val >= min && val <= max;
        }}).length;
    }}
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {{
        const filterType = btn.dataset.filter;
        const min = parseInt(btn.dataset.min);
        const max = parseInt(btn.dataset.max);
        const count = countMatching(filterType, min, max);
        btn.textContent = btn.textContent + ` (${{count}})`;
    }});

    // Quick filter buttons
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {{
        btn.onclick = () => {{
            const filterType = btn.dataset.filter;
            const min = parseInt(btn.dataset.min);
            const max = parseInt(btn.dataset.max);

            // Toggle if same filter clicked again
            if (filters[filterType].min === min && filters[filterType].max === max) {{
                filters[filterType] = {{min: null, max: null}};
                btn.classList.remove('active');
            }} else {{
                // Clear other buttons of same type
                document.querySelectorAll(`.filter-btn[data-filter="${{filterType}}"]`).forEach(b => b.classList.remove('active'));
                filters[filterType] = {{min, max}};
                btn.classList.add('active');
            }}
            updateInputsFromFilters();
            renderConstraintList();
        }};
    }});

    document.getElementById('clear-filter').onclick = () => {{
        filters = {{ depth: {{min: null, max: null}}, vars: {{min: null, max: null}}, len: {{min: null, max: null}}, mono: {{min: null, max: null}} }};
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        updateInputsFromFilters();
        renderConstraintList();
    }};

    function updateInputsFromFilters() {{
        document.getElementById('depth-min').value = filters.depth.min ?? '';
        document.getElementById('depth-max').value = filters.depth.max ?? '';
        document.getElementById('vars-min').value = filters.vars.min ?? '';
        document.getElementById('vars-max').value = filters.vars.max ?? '';
        document.getElementById('len-min').value = filters.len.min ?? '';
        document.getElementById('len-max').value = filters.len.max ?? '';
        document.getElementById('mono-min').value = filters.mono.min ?? '';
        document.getElementById('mono-max').value = filters.mono.max ?? '';
    }}

    // Custom filter inputs
    ['depth', 'vars', 'len', 'mono'].forEach(filterType => {{
        const minInput = document.getElementById(filterType + '-min');
        const maxInput = document.getElementById(filterType + '-max');
        const update = () => {{
            filters[filterType].min = minInput.value ? parseInt(minInput.value) : null;
            filters[filterType].max = maxInput.value ? parseInt(maxInput.value) : null;
            document.querySelectorAll(`.filter-btn[data-filter="${{filterType}}"]`).forEach(b => b.classList.remove('active'));
            renderConstraintList();
        }};
        minInput.oninput = update;
        maxInput.oninput = update;
    }});

    // Sort controls
    document.getElementById('sort-select').onchange = (e) => {{ sortBy = e.target.value; renderConstraintList(); }};
    document.getElementById('sort-dir').onchange = (e) => {{ sortDir = e.target.value; renderConstraintList(); }};

    // Graph rendering
    let selectedConstraint = null;
    let cy = null;

    function selectConstraint(index) {{
        selectedConstraint = index;

        document.querySelectorAll('.constraint-item').forEach(el => {{
            el.classList.toggle('selected', parseInt(el.dataset.index) === index);
        }});

        const c = constraintInfo[index];
        document.getElementById('constraint-title').innerHTML =
            `<b>Constraint #${{index}}</b> | Depth: ${{c.depth}} | Vars: ${{c.vars}} | Length: ${{c.len}}`;

        updateMonomials(index);

        // Show loading and defer layout computation
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = 'flex';

        setTimeout(() => {{
            renderGraph(c.nodeId);
            overlay.style.display = 'none';
        }}, 10);
    }}

    // Collect all node IDs in subtree
    function collectAllNodes(nodeId, visited = new Set()) {{
        if (visited.has(nodeId)) return visited;
        visited.add(nodeId);
        const node = nodes[nodeId];
        if (node.children) {{
            node.children.forEach(childId => collectAllNodes(childId, visited));
        }}
        return visited;
    }}

    function renderGraph(rootId) {{
        const subNodes = collectAllNodes(rootId);
        const elements = [];

        // Add nodes
        subNodes.forEach(nodeId => {{
            const node = nodes[nodeId];
            let label = node.type;
            if (node.type === 'Variable' || node.type === 'Constant') {{
                label = node.label.length > 35 ? node.label.substring(0, 32) + '...' : node.label;
            }} else if (node.type === 'Add') label = '+';
            else if (node.type === 'Sub') label = '-';
            else if (node.type === 'Mul') label = '*';
            else if (node.type === 'Neg') label = 'neg';

            const colors = {{
                'Variable': '#fff3cd', 'Constant': '#e2e3e5',
                'Add': '#cce5ff', 'Sub': '#d4edda', 'Mul': '#d1ecf1', 'Neg': '#f8d7da',
                'IsFirstRow': '#ffeeba', 'IsLastRow': '#ffeeba', 'IsTransition': '#ffeeba'
            }};

            elements.push({{
                data: {{
                    id: 'n' + nodeId,
                    label: label,
                    fullLabel: node.label,
                    nodeType: node.type
                }},
                style: {{ 'background-color': colors[node.type] || '#ccc' }}
            }});
        }});

        // Add edges
        subNodes.forEach(nodeId => {{
            const node = nodes[nodeId];
            if (node.children) {{
                node.children.forEach((childId, i) => {{
                    elements.push({{
                        data: {{
                            id: 'e' + nodeId + '_' + i + '_' + childId,
                            source: 'n' + nodeId,
                            target: 'n' + childId
                        }}
                    }});
                }});
            }}
        }});

        cy = cytoscape({{
            container: document.getElementById('graph-container'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-family': 'monospace',
                        'font-size': '10px',
                        'width': 'label',
                        'height': 'label',
                        'padding': '5px',
                        'shape': 'round-rectangle',
                        'border-width': 1,
                        'border-color': '#999'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 1,
                        'line-color': '#666',
                        'target-arrow-color': '#666',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }}
            ],
            layout: {{
                name: 'dagre',
                rankDir: 'TB',
                nodeSep: 50,
                rankSep: 70,
                edgeSep: 10
            }}
        }});

        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            alert(node.data('nodeType') + ': ' + node.data('fullLabel'));
        }});
    }}

    // Monomial expansion (precomputed in Rust)
    let monomialsExpanded = false;

    document.getElementById('monomials-header').onclick = function() {{
        monomialsExpanded = !monomialsExpanded;
        const container = document.getElementById('monomials-container');
        container.style.display = monomialsExpanded ? 'block' : 'none';
        this.firstChild.textContent = monomialsExpanded ? '▼ ' : '▶ ';
    }};

    function updateMonomials(constraintIndex) {{
        const container = document.getElementById('monomials-container');
        const countSpan = document.getElementById('monomial-count');
        const monomials = MONOMIAL_STRINGS[constraintIndex];
        const count = monomials.length;

        countSpan.textContent = '(' + count + ' terms)';

        if (count === 0) {{
            container.innerHTML = '<i>0 (empty polynomial)</i>';
        }} else if (count > 500) {{
            container.innerHTML = '<i>' + count + ' monomials (showing first 500)</i><br><br>' +
                monomials.slice(0, 500).join('<br>') + '<br>...';
        }} else {{
            container.innerHTML = monomials.join('<br>');
        }}
    }}

    // Initialize
    if (constraintInfo.length > 0) selectConstraint(0);
    </script>
</body>
</html>
"##,
        title = title,
        json_str = json_str,
        rule_lengths_json = rule_lengths_json
    )
}
