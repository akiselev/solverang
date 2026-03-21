//! Component decomposition for parallel solving.
//!
//! This module provides functionality to decompose a constraint problem into
//! independent components that can be solved in parallel. Two constraints are
//! in the same component if they share variables (directly or transitively).
//!
//! # Algorithm
//!
//! The decomposition uses a union-find (disjoint set) data structure to efficiently
//! find connected components in the constraint-variable bipartite graph.

use crate::problem::Problem;

/// Unique identifier for a constraint component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComponentId(pub usize);

/// A connected component of constraints that share variables.
///
/// Variables and constraints within a component are interdependent and must be
/// solved together. Different components can be solved independently (in parallel).
#[derive(Clone, Debug)]
pub struct Component {
    /// Unique identifier for this component.
    pub id: ComponentId,
    /// Indices of variables in this component (into the original problem's variable array).
    pub variable_indices: Vec<usize>,
    /// Indices of constraints in this component (into the original problem's constraint array).
    pub constraint_indices: Vec<usize>,
}

impl Component {
    /// Returns true if this component has no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraint_indices.is_empty()
    }

    /// Returns true if this component has no variables.
    pub fn has_no_variables(&self) -> bool {
        self.variable_indices.is_empty()
    }

    /// Number of variables in this component.
    pub fn variable_count(&self) -> usize {
        self.variable_indices.len()
    }

    /// Number of constraints in this component.
    pub fn constraint_count(&self) -> usize {
        self.constraint_indices.len()
    }
}

/// Trait for problems that can provide their constraint-variable graph.
///
/// Implementing this trait allows a problem to be decomposed into independent
/// components for parallel solving.
pub trait DecomposableProblem: Problem {
    /// Returns the constraint-variable connectivity graph.
    ///
    /// Each element `(constraint_index, variable_index)` indicates that
    /// constraint `constraint_index` depends on variable `variable_index`.
    ///
    /// Default implementation computes this from the Jacobian sparsity pattern
    /// at the initial point, which works for most problems but may miss
    /// dependencies that only appear in specific regions of the search space.
    fn constraint_graph(&self) -> Vec<(usize, usize)> {
        // Default: compute from Jacobian at initial point
        let x0 = self.initial_point(1.0);
        let jacobian = self.jacobian(&x0);

        jacobian
            .into_iter()
            .map(|(row, col, _)| (row, col))
            .collect()
    }

    /// Returns which variables are affected by each constraint.
    ///
    /// This is an alternative way to specify the constraint graph that may be
    /// more natural for some problem formulations.
    fn variables_in_constraint(&self, constraint_index: usize) -> Vec<usize> {
        self.constraint_graph()
            .into_iter()
            .filter(|(c, _)| *c == constraint_index)
            .map(|(_, v)| v)
            .collect()
    }
}

/// Union-Find data structure for efficiently computing connected components.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Create a new union-find structure with `n` elements.
    fn new(elements: usize) -> Self {
        Self {
            parent: (0..elements).collect(),
            rank: vec![0; elements],
        }
    }

    /// Find the root of the set containing `element` with path compression.
    fn find(&mut self, element: usize) -> usize {
        if self.parent[element] != element {
            self.parent[element] = self.find(self.parent[element]);
        }
        self.parent[element]
    }

    /// Union the sets containing `a` and `b` by rank.
    fn union(&mut self, first: usize, second: usize) {
        let root_a = self.find(first);
        let root_b = self.find(second);

        if root_a != root_b {
            // Union by rank to keep trees balanced
            if self.rank[root_a] < self.rank[root_b] {
                self.parent[root_a] = root_b;
            } else if self.rank[root_a] > self.rank[root_b] {
                self.parent[root_b] = root_a;
            } else {
                self.parent[root_b] = root_a;
                self.rank[root_a] += 1;
            }
        }
    }
}

/// Decompose a problem into independent constraint components.
///
/// Two constraints are in the same component if they share any variables
/// (directly or transitively through other constraints).
///
/// # Arguments
///
/// * `problem` - The problem to decompose
///
/// # Returns
///
/// A vector of `Component` structs, each containing the indices of variables
/// and constraints that belong to that component. Components are sorted by
/// their component ID.
///
/// # Edge Cases
///
/// - If the problem has no constraints, returns an empty vector.
/// - If all constraints share variables, returns a single component.
/// - Variables that appear in no constraints will not be in any component.
pub fn decompose<P: DecomposableProblem>(problem: &P) -> Vec<Component> {
    let constraint_count = problem.residual_count();
    let variable_count = problem.variable_count();

    if constraint_count == 0 {
        return Vec::new();
    }

    let graph = problem.constraint_graph();

    // Handle empty graph (no dependencies)
    if graph.is_empty() {
        // Each constraint is its own component with no variables
        return (0..constraint_count)
            .map(|index| Component {
                id: ComponentId(index),
                variable_indices: Vec::new(),
                constraint_indices: vec![index],
            })
            .collect();
    }

    // Use union-find on a combined space: first `constraint_count` elements are constraints,
    // next `variable_count` elements are variables
    let total_elements = constraint_count + variable_count;
    let mut union_find = UnionFind::new(total_elements);

    // Union constraints with the variables they depend on
    for (constraint_idx, variable_idx) in &graph {
        if *constraint_idx < constraint_count && *variable_idx < variable_count {
            // Map variable index to the combined space
            let variable_element = constraint_count + variable_idx;
            union_find.union(*constraint_idx, variable_element);
        }
    }

    // Group elements by their root
    let mut component_map = std::collections::HashMap::new();

    // Group constraints
    for constraint_idx in 0..constraint_count {
        let root = union_find.find(constraint_idx);
        component_map
            .entry(root)
            .or_insert_with(|| (Vec::new(), Vec::new()))
            .1
            .push(constraint_idx);
    }

    // Group variables (only those that appear in some constraint)
    let mut variables_in_graph = std::collections::HashSet::new();
    for (_, variable_idx) in &graph {
        if *variable_idx < variable_count {
            variables_in_graph.insert(*variable_idx);
        }
    }

    for variable_idx in variables_in_graph {
        let variable_element = constraint_count + variable_idx;
        let root = union_find.find(variable_element);
        if let Some(entry) = component_map.get_mut(&root) {
            entry.0.push(variable_idx);
        }
    }

    // Convert to Component structs
    let mut components: Vec<Component> = component_map
        .into_iter()
        .enumerate()
        .map(|(idx, (_, (mut variables, mut constraints)))| {
            variables.sort_unstable();
            constraints.sort_unstable();
            Component {
                id: ComponentId(idx),
                variable_indices: variables,
                constraint_indices: constraints,
            }
        })
        .collect();

    // Sort components by their first constraint index for deterministic ordering
    components.sort_by_key(|component| {
        component
            .constraint_indices
            .first()
            .copied()
            .unwrap_or(usize::MAX)
    });

    // Reassign component IDs after sorting
    for (idx, component) in components.iter_mut().enumerate() {
        component.id = ComponentId(idx);
    }

    components
}

/// Decompose using an explicit edge list.
///
/// This is useful when you have the constraint graph available directly
/// without needing to compute it from a Problem trait.
///
/// # Arguments
///
/// * `constraint_count` - Number of constraints
/// * `variable_count` - Number of variables
/// * `edges` - List of (constraint_index, variable_index) pairs
///
/// # Returns
///
/// A vector of `Component` structs.
pub fn decompose_from_edges(
    constraint_count: usize,
    variable_count: usize,
    edges: &[(usize, usize)],
) -> Vec<Component> {
    if constraint_count == 0 {
        return Vec::new();
    }

    // Handle empty graph
    if edges.is_empty() {
        return (0..constraint_count)
            .map(|index| Component {
                id: ComponentId(index),
                variable_indices: Vec::new(),
                constraint_indices: vec![index],
            })
            .collect();
    }

    let total_elements = constraint_count + variable_count;
    let mut union_find = UnionFind::new(total_elements);

    for (constraint_idx, variable_idx) in edges {
        if *constraint_idx < constraint_count && *variable_idx < variable_count {
            let variable_element = constraint_count + variable_idx;
            union_find.union(*constraint_idx, variable_element);
        }
    }

    let mut component_map = std::collections::HashMap::new();

    for constraint_idx in 0..constraint_count {
        let root = union_find.find(constraint_idx);
        component_map
            .entry(root)
            .or_insert_with(|| (Vec::new(), Vec::new()))
            .1
            .push(constraint_idx);
    }

    let mut variables_in_graph = std::collections::HashSet::new();
    for (_, variable_idx) in edges {
        if *variable_idx < variable_count {
            variables_in_graph.insert(*variable_idx);
        }
    }

    for variable_idx in variables_in_graph {
        let variable_element = constraint_count + variable_idx;
        let root = union_find.find(variable_element);
        if let Some(entry) = component_map.get_mut(&root) {
            entry.0.push(variable_idx);
        }
    }

    let mut components: Vec<Component> = component_map
        .into_iter()
        .enumerate()
        .map(|(idx, (_, (mut variables, mut constraints)))| {
            variables.sort_unstable();
            constraints.sort_unstable();
            Component {
                id: ComponentId(idx),
                variable_indices: variables,
                constraint_indices: constraints,
            }
        })
        .collect();

    components.sort_by_key(|component| {
        component
            .constraint_indices
            .first()
            .copied()
            .unwrap_or(usize::MAX)
    });

    for (idx, component) in components.iter_mut().enumerate() {
        component.id = ComponentId(idx);
    }

    components
}

/// Result of extracting a sub-problem from a component.
#[derive(Clone, Debug)]
pub struct SubProblem {
    /// The component this sub-problem was extracted from.
    pub component: Component,
    /// Mapping from sub-problem variable index to original variable index.
    pub variable_mapping: Vec<usize>,
    /// Mapping from sub-problem constraint index to original constraint index.
    pub constraint_mapping: Vec<usize>,
}

impl SubProblem {
    /// Create a sub-problem from a component.
    pub fn from_component(component: Component) -> Self {
        let variable_mapping = component.variable_indices.clone();
        let constraint_mapping = component.constraint_indices.clone();
        Self {
            component,
            variable_mapping,
            constraint_mapping,
        }
    }

    /// Map a local variable index to the original problem's variable index.
    pub fn map_variable(&self, local_index: usize) -> Option<usize> {
        self.variable_mapping.get(local_index).copied()
    }

    /// Map a local constraint index to the original problem's constraint index.
    pub fn map_constraint(&self, local_index: usize) -> Option<usize> {
        self.constraint_mapping.get(local_index).copied()
    }

    /// Extract sub-problem variables from a full solution vector.
    pub fn extract_variables(&self, full_solution: &[f64]) -> Vec<f64> {
        self.variable_mapping
            .iter()
            .filter_map(|&idx| full_solution.get(idx).copied())
            .collect()
    }

    /// Inject sub-problem solution into a full solution vector.
    ///
    /// This modifies `full_solution` in place, updating only the variables
    /// that belong to this sub-problem.
    pub fn inject_solution(&self, sub_solution: &[f64], full_solution: &mut [f64]) {
        for (local_idx, &global_idx) in self.variable_mapping.iter().enumerate() {
            if let Some(value) = sub_solution.get(local_idx) {
                if global_idx < full_solution.len() {
                    full_solution[global_idx] = *value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a mock problem for testing
    struct MockDecomposableProblem {
        constraints: usize,
        variables: usize,
        graph: Vec<(usize, usize)>,
    }

    impl Problem for MockDecomposableProblem {
        fn name(&self) -> &str {
            "mock"
        }
        fn residual_count(&self) -> usize {
            self.constraints
        }
        fn variable_count(&self) -> usize {
            self.variables
        }
        fn residuals(&self, _x: &[f64]) -> Vec<f64> {
            vec![0.0; self.constraints]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            self.graph.iter().map(|&(r, c)| (r, c, 1.0)).collect()
        }
        fn initial_point(&self, _factor: f64) -> Vec<f64> {
            vec![0.0; self.variables]
        }
    }

    impl DecomposableProblem for MockDecomposableProblem {
        fn constraint_graph(&self) -> Vec<(usize, usize)> {
            self.graph.clone()
        }
    }

    #[test]
    fn test_single_component_chain() {
        // Chain: c0 -> v0 -> c1 -> v1 -> c2
        let problem = MockDecomposableProblem {
            constraints: 3,
            variables: 2,
            graph: vec![(0, 0), (1, 0), (1, 1), (2, 1)],
        };

        let components = decompose(&problem);
        assert_eq!(components.len(), 1, "Chain should form single component");
        assert_eq!(components[0].variable_count(), 2);
        assert_eq!(components[0].constraint_count(), 3);
    }

    #[test]
    fn test_two_independent_components() {
        // Two separate components: {c0, c1, v0} and {c2, c3, v1}
        let problem = MockDecomposableProblem {
            constraints: 4,
            variables: 2,
            graph: vec![(0, 0), (1, 0), (2, 1), (3, 1)],
        };

        let components = decompose(&problem);
        assert_eq!(
            components.len(),
            2,
            "Should have two independent components"
        );

        // Check that each component has the right variables
        let component_vars: Vec<Vec<usize>> = components
            .iter()
            .map(|c| c.variable_indices.clone())
            .collect();

        assert!(
            component_vars.iter().any(|v| v == &[0]),
            "One component should have variable 0"
        );
        assert!(
            component_vars.iter().any(|v| v == &[1]),
            "One component should have variable 1"
        );
    }

    #[test]
    fn test_maximum_parallelism() {
        // Each constraint uses a different variable
        let problem = MockDecomposableProblem {
            constraints: 4,
            variables: 4,
            graph: vec![(0, 0), (1, 1), (2, 2), (3, 3)],
        };

        let components = decompose(&problem);
        assert_eq!(
            components.len(),
            4,
            "Each constraint should be its own component"
        );

        for component in &components {
            assert_eq!(component.variable_count(), 1);
            assert_eq!(component.constraint_count(), 1);
        }
    }

    #[test]
    fn test_empty_graph() {
        // Constraints with no variable dependencies
        let problem = MockDecomposableProblem {
            constraints: 3,
            variables: 2,
            graph: vec![],
        };

        let components = decompose(&problem);
        assert_eq!(
            components.len(),
            3,
            "Each constraint should be its own component"
        );

        for component in &components {
            assert_eq!(component.variable_count(), 0);
            assert_eq!(component.constraint_count(), 1);
        }
    }

    #[test]
    fn test_no_constraints() {
        let problem = MockDecomposableProblem {
            constraints: 0,
            variables: 5,
            graph: vec![],
        };

        let components = decompose(&problem);
        assert!(components.is_empty());
    }

    #[test]
    fn test_cyclic_dependencies() {
        // All constraints share all variables (fully connected)
        let problem = MockDecomposableProblem {
            constraints: 3,
            variables: 3,
            graph: vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)],
        };

        let components = decompose(&problem);
        assert_eq!(components.len(), 1, "Cycle should form single component");
        assert_eq!(components[0].variable_count(), 3);
        assert_eq!(components[0].constraint_count(), 3);
    }

    #[test]
    fn test_tree_structure() {
        // Tree: v0 connects c0, c1; v1 connects c1, c2; v2 connects c2, c3
        let problem = MockDecomposableProblem {
            constraints: 4,
            variables: 3,
            graph: vec![(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
        };

        let components = decompose(&problem);
        assert_eq!(components.len(), 1, "Tree should form single component");
        assert_eq!(components[0].variable_count(), 3);
        assert_eq!(components[0].constraint_count(), 4);
    }

    #[test]
    fn test_sub_problem_extraction() {
        let component = Component {
            id: ComponentId(0),
            variable_indices: vec![2, 5, 8],
            constraint_indices: vec![1, 3],
        };

        let sub_problem = SubProblem::from_component(component);

        assert_eq!(sub_problem.map_variable(0), Some(2));
        assert_eq!(sub_problem.map_variable(1), Some(5));
        assert_eq!(sub_problem.map_variable(2), Some(8));
        assert_eq!(sub_problem.map_variable(3), None);

        assert_eq!(sub_problem.map_constraint(0), Some(1));
        assert_eq!(sub_problem.map_constraint(1), Some(3));
    }

    #[test]
    fn test_variable_extraction_and_injection() {
        let component = Component {
            id: ComponentId(0),
            variable_indices: vec![1, 3],
            constraint_indices: vec![0],
        };

        let sub_problem = SubProblem::from_component(component);

        let full_solution = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let extracted = sub_problem.extract_variables(&full_solution);
        assert_eq!(extracted, vec![1.0, 3.0]);

        let sub_solution = vec![10.0, 30.0];
        let mut new_full = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        sub_problem.inject_solution(&sub_solution, &mut new_full);
        assert_eq!(new_full, vec![0.0, 10.0, 0.0, 30.0, 0.0]);
    }

    #[test]
    fn test_decompose_from_edges() {
        let edges = vec![(0, 0), (1, 1), (2, 0), (3, 1)];
        let components = decompose_from_edges(4, 2, &edges);

        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_component_is_empty() {
        let empty_component = Component {
            id: ComponentId(0),
            variable_indices: vec![1],
            constraint_indices: vec![],
        };
        assert!(empty_component.is_empty());

        let non_empty = Component {
            id: ComponentId(1),
            variable_indices: vec![1],
            constraint_indices: vec![0],
        };
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_out_of_bounds_edges_ignored() {
        // Edges with out-of-bounds indices should be safely ignored
        let edges = vec![(0, 0), (100, 0), (0, 100)];
        let components = decompose_from_edges(2, 2, &edges);

        // Should have 2 components: one with constraint 0 and variable 0,
        // one with just constraint 1
        assert_eq!(components.len(), 2);
    }
}
