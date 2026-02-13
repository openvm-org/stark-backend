use std::{fs::File, io::Write, path::Path};

use serde::{Deserialize, Serialize};

use crate::air_builders::symbolic::{
    SymbolicConstraintsDag, SymbolicExpressionDag, SymbolicExpressionNode,
};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AirStatistics {
    pub air_name: String,
    pub num_nodes: usize,
    pub num_interactions: usize,
    pub num_constraints: usize,
    pub max_constraint_depth: usize,
    pub average_constraint_depth: f64,
    pub num_constants: usize,
    pub num_variables: usize,
    pub num_intermediates: usize,
    pub max_intermediate_use: usize,
    pub average_intermediate_use: f64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub depth: usize,
    pub uses: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AirStatisticsGenerator {
    pub stats: Vec<AirStatistics>,
}

impl AirStatisticsGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn generate<F>(&mut self, name: String, dag: &SymbolicConstraintsDag<F>) {
        let mut stats = AirStatistics {
            air_name: name,
            num_nodes: dag.constraints.nodes.len(),
            num_interactions: dag.interactions.len(),
            num_constraints: dag.constraints.constraint_idx.len(),
            ..Default::default()
        };
        let mut node_info = vec![NodeInfo::default(); stats.num_nodes];
        for (i, node) in dag.constraints.nodes.iter().enumerate() {
            match node {
                SymbolicExpressionNode::Variable(_) => {
                    node_info[i].uses += 1;
                    node_info[i].depth = 1;
                    stats.num_variables += 1;
                }
                SymbolicExpressionNode::Constant(_) => {
                    node_info[i].uses += 1;
                    node_info[i].depth = 1;
                    stats.num_constants += 1;
                }
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    node_info[i].uses += 1;
                    node_info[*left_idx].uses += 1;
                    node_info[*right_idx].uses += 1;
                    node_info[i].depth =
                        node_info[*left_idx].depth.max(node_info[*right_idx].depth) + 1;
                    stats.num_intermediates += 1;
                }
                SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    node_info[i].uses += 1;
                    node_info[*left_idx].uses += 1;
                    node_info[*right_idx].uses += 1;
                    node_info[i].depth =
                        node_info[*left_idx].depth.max(node_info[*right_idx].depth) + 1;
                    stats.num_intermediates += 1;
                }
                SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    node_info[i].uses += 1;
                    node_info[*left_idx].uses += 1;
                    node_info[*right_idx].uses += 1;
                    node_info[i].depth =
                        node_info[*left_idx].depth.max(node_info[*right_idx].depth) + 1;
                    stats.num_intermediates += 1;
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    node_info[i].uses += 1;
                    node_info[*idx].uses += 1;
                    node_info[i].depth = node_info[*idx].depth + 1;
                    stats.num_intermediates += 1;
                }
                _ => {
                    node_info[i].uses += 1;
                    node_info[i].depth = 1;
                }
            }
        }

        for node in node_info.iter().filter(|node| node.depth > 1) {
            stats.max_intermediate_use = stats.max_intermediate_use.max(node.uses);
            stats.average_intermediate_use += node.uses as f64;
        }
        stats.average_intermediate_use /= stats.num_intermediates as f64;

        for constraint_idx in &dag.constraints.constraint_idx {
            stats.max_constraint_depth = stats
                .max_constraint_depth
                .max(node_info[*constraint_idx].depth);
            stats.average_constraint_depth += node_info[*constraint_idx].depth as f64;
        }
        stats.average_constraint_depth /= stats.num_constraints as f64;

        self.stats.push(stats);
    }

    pub fn write_json<P: AsRef<Path>>(&self, file_path: P) -> eyre::Result<()> {
        serde_json::to_writer_pretty(File::create(file_path)?, &self.stats)?;
        Ok(())
    }

    pub fn write_csv<P: AsRef<Path>>(&self, file_path: P) -> eyre::Result<()> {
        let mut file = File::create(file_path)?;
        writeln!(
            file,
            "air_name,num_nodes,num_interactions,num_constraints,max_constraint_depth,average_constraint_depth,num_constants,num_variables,num_intermediates,max_intermediate_use,average_intermediate_use"
        )?;
        for stat in &self.stats {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{}",
                stat.air_name,
                stat.num_nodes,
                stat.num_interactions,
                stat.num_constraints,
                stat.max_constraint_depth,
                stat.average_constraint_depth,
                stat.num_constants,
                stat.num_variables,
                stat.num_intermediates,
                stat.max_intermediate_use,
                stat.average_intermediate_use
            )?;
        }
        Ok(())
    }

    pub fn print_dag<F: std::fmt::Debug>(dag: &SymbolicExpressionDag<F>) {
        for (idx, node) in dag.nodes.iter().enumerate() {
            println!("  Node {}: {:?}", idx, node);
        }
    }
}

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use crate::{
        air_builders::symbolic::{
            build_symbolic_constraints_dag,
            statistics::AirStatisticsGenerator,
            symbolic_expression::SymbolicExpression,
            symbolic_variable::{Entry, SymbolicVariable},
        },
        interaction::Interaction,
    };

    type F = BabyBear;

    #[test]
    fn test_dag_statistics() {
        let expr1 = SymbolicExpression::Variable(SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            1,
        ));
        let expr2 = SymbolicExpression::Variable(SymbolicVariable::new(
            Entry::Main {
                part_index: 1,
                offset: 1,
            },
            2,
        ));
        let expr3 = expr1.clone() + expr2.clone();
        let expr4 = expr3.clone() * SymbolicExpression::Constant(F::TWO);
        let expr5 = SymbolicExpression::IsFirstRow + expr4.clone();

        let constraints = vec![expr5, expr4.clone()];
        let interactions = vec![Interaction {
            bus_index: 0,
            message: vec![expr1, expr2],
            count: SymbolicExpression::Constant(F::ONE),
            count_weight: 1,
        }];

        let dag = build_symbolic_constraints_dag(&constraints, &interactions);
        AirStatisticsGenerator::print_dag(&dag.constraints);

        let mut generator = AirStatisticsGenerator::new();
        generator.generate("test".to_string(), &dag);
        println!("{:?}", generator.stats);

        assert_eq!(generator.stats[0].num_nodes, 8);
        assert_eq!(generator.stats[0].num_interactions, 1);
        assert_eq!(generator.stats[0].num_constraints, 2);
        assert_eq!(generator.stats[0].max_constraint_depth, 4);
        assert_eq!(generator.stats[0].average_constraint_depth, 3.5);
        assert_eq!(generator.stats[0].num_constants, 2);
        assert_eq!(generator.stats[0].num_variables, 2);
        assert_eq!(generator.stats[0].num_intermediates, 3);
        assert_eq!(generator.stats[0].max_intermediate_use, 2);
        assert_eq!(generator.stats[0].average_intermediate_use, 5f64 / 3f64);
    }
}
