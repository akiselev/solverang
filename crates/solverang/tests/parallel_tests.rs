//! Tests for the parallel solver.

use solverang::{
    decompose, decompose_from_edges, Component, ComponentId, DecomposableProblem,
    ParallelSolver, ParallelSolverConfig, Problem, SolveError,
};

/// A multi-component test problem with three independent sub-systems.
///
/// Component 1: x0^2 - 2 = 0 (solution: sqrt(2))
/// Component 2: x1 - 1 = 0, x2 - 2 = 0 (solution: (1, 2))
/// Component 3: x3^2 + x4^2 - 1 = 0, x3 - x4 = 0 (solution: (1/sqrt(2), 1/sqrt(2)))
struct ThreeComponentProblem;

impl Problem for ThreeComponentProblem {
    fn name(&self) -> &str {
        "three-component"
    }

    fn residual_count(&self) -> usize {
        5
    }

    fn variable_count(&self) -> usize {
        5
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![
            x[0] * x[0] - 2.0,               // Component 1
            x[1] - 1.0,                      // Component 2
            x[2] - 2.0,                      // Component 2
            x[3] * x[3] + x[4] * x[4] - 1.0, // Component 3
            x[3] - x[4],                     // Component 3
        ]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, 0, 2.0 * x[0]), // d(x0^2-2)/dx0
            (1, 1, 1.0),       // d(x1-1)/dx1
            (2, 2, 1.0),       // d(x2-2)/dx2
            (3, 3, 2.0 * x[3]), // d(x3^2+x4^2-1)/dx3
            (3, 4, 2.0 * x[4]), // d(x3^2+x4^2-1)/dx4
            (4, 3, 1.0),       // d(x3-x4)/dx3
            (4, 4, -1.0),      // d(x3-x4)/dx4
        ]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![1.0 * factor, 0.0, 0.0, 0.5 * factor, 0.5 * factor]
    }
}

impl DecomposableProblem for ThreeComponentProblem {
    fn constraint_graph(&self) -> Vec<(usize, usize)> {
        vec![
            (0, 0), // Constraint 0 uses x0
            (1, 1), // Constraint 1 uses x1
            (2, 2), // Constraint 2 uses x2
            (3, 3), // Constraint 3 uses x3
            (3, 4), // Constraint 3 uses x4
            (4, 3), // Constraint 4 uses x3
            (4, 4), // Constraint 4 uses x4
        ]
    }
}

#[test]
fn test_decomposition_three_components() {
    let problem = ThreeComponentProblem;
    let components = decompose(&problem);

    // Should have 3 or 4 components depending on whether c1,c2 are grouped
    // Since c1 and c2 don't share variables, they're separate
    assert!(
        components.len() >= 3,
        "Expected at least 3 components, got {}",
        components.len()
    );

    // Verify total constraints covered
    let total_constraints: usize = components.iter().map(|c| c.constraint_count()).sum();
    assert_eq!(total_constraints, 5);

    // Verify total variables covered (some variables may be in multiple constraint's Jacobians)
    let all_vars: std::collections::HashSet<usize> = components
        .iter()
        .flat_map(|c| c.variable_indices.iter().copied())
        .collect();
    assert_eq!(all_vars.len(), 5);
}

#[test]
fn test_parallel_solver_three_components() {
    let problem = ThreeComponentProblem;
    let solver = ParallelSolver::new(ParallelSolverConfig {
        min_parallel_size: 1, // Lower threshold for testing
        min_parallel_components: 2,
        ..Default::default()
    });

    let x0 = vec![1.5, 0.5, 0.5, 0.7, 0.7];
    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged(),
        "Should converge, got {:?}",
        result
    );

    let solution = result.solution().expect("should have solution");

    // Verify component 1: x0 = sqrt(2)
    assert!(
        (solution[0] - std::f64::consts::SQRT_2).abs() < 1e-6,
        "x0 should be sqrt(2), got {}",
        solution[0]
    );

    // Verify component 2: x1 = 1, x2 = 2
    assert!(
        (solution[1] - 1.0).abs() < 1e-6,
        "x1 should be 1, got {}",
        solution[1]
    );
    assert!(
        (solution[2] - 2.0).abs() < 1e-6,
        "x2 should be 2, got {}",
        solution[2]
    );

    // Verify component 3: x3 = x4 = 1/sqrt(2)
    let expected = 1.0 / std::f64::consts::SQRT_2;
    assert!(
        (solution[3] - expected).abs() < 1e-5,
        "x3 should be {}, got {}",
        expected,
        solution[3]
    );
    assert!(
        (solution[4] - expected).abs() < 1e-5,
        "x4 should be {}, got {}",
        expected,
        solution[4]
    );
}

/// Test with a chain graph where all constraints are connected.
struct ChainProblem {
    size: usize,
}

impl Problem for ChainProblem {
    fn name(&self) -> &str {
        "chain"
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn variable_count(&self) -> usize {
        self.size
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // Each constraint involves x[i] and x[i+1] (except last)
        let mut r = Vec::with_capacity(self.size);
        for i in 0..self.size {
            if i < self.size - 1 {
                r.push(x[i] + x[i + 1] - (i as f64 + 1.0));
            } else {
                r.push(x[i] - (i as f64));
            }
        }
        r
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut jac = Vec::new();
        for i in 0..self.size {
            jac.push((i, i, 1.0));
            if i < self.size - 1 {
                jac.push((i, i + 1, 1.0));
            }
        }
        jac
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        vec![0.0; self.size]
    }
}

impl DecomposableProblem for ChainProblem {}

#[test]
fn test_chain_single_component() {
    let problem = ChainProblem { size: 5 };
    let components = decompose(&problem);

    // Chain should form a single component
    assert_eq!(components.len(), 1, "Chain should be single component");
    assert_eq!(components[0].variable_count(), 5);
    assert_eq!(components[0].constraint_count(), 5);
}

#[test]
fn test_parallel_solver_chain() {
    let problem = ChainProblem { size: 5 };
    let solver = ParallelSolver::default_solver();
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged() || result.is_completed(),
        "Result: {:?}",
        result
    );
}

/// Test decomposition from explicit edges.
#[test]
fn test_decompose_from_edges() {
    // Two independent 2-constraint components
    let edges = vec![
        (0, 0), (0, 1), // c0 uses v0, v1
        (1, 0), (1, 1), // c1 uses v0, v1 (same component as c0)
        (2, 2), (2, 3), // c2 uses v2, v3
        (3, 2), (3, 3), // c3 uses v2, v3 (same component as c2)
    ];

    let components = decompose_from_edges(4, 4, &edges);

    assert_eq!(components.len(), 2);

    // Each component should have 2 constraints and 2 variables
    for component in &components {
        assert_eq!(component.constraint_count(), 2);
        assert_eq!(component.variable_count(), 2);
    }
}

/// Test with maximum parallelism (each constraint independent).
#[test]
fn test_maximum_parallelism() {
    // 4 independent constraints, each using a different variable
    let edges = vec![(0, 0), (1, 1), (2, 2), (3, 3)];

    let components = decompose_from_edges(4, 4, &edges);

    assert_eq!(components.len(), 4, "Should have 4 independent components");

    for component in &components {
        assert_eq!(component.constraint_count(), 1);
        assert_eq!(component.variable_count(), 1);
    }
}

/// Test incremental solving.
struct IncrementalProblem;

impl Problem for IncrementalProblem {
    fn name(&self) -> &str {
        "incremental"
    }

    fn residual_count(&self) -> usize {
        4
    }

    fn variable_count(&self) -> usize {
        4
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![
            x[0] - 1.0, // Independent
            x[1] - 2.0, // Independent
            x[2] - 3.0, // Independent
            x[3] - 4.0, // Independent
        ]
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)]
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        vec![0.0, 0.0, 0.0, 0.0]
    }
}

impl DecomposableProblem for IncrementalProblem {}

#[test]
fn test_incremental_solve() {
    let problem = IncrementalProblem;
    let solver = ParallelSolver::default_solver();

    // Start with x already at solution for components 0 and 1
    let x0 = vec![1.0, 2.0, 0.0, 0.0];

    // Only solve components 2 and 3
    let result = solver.solve_incremental(&problem, &x0, &[ComponentId(2), ComponentId(3)]);

    assert!(result.is_converged() || result.is_completed());
}

/// Test error handling.
#[test]
fn test_dimension_mismatch() {
    let problem = ThreeComponentProblem;
    let solver = ParallelSolver::default_solver();

    let result = solver.solve(&problem, &[1.0, 2.0]); // Wrong size

    assert!(!result.is_completed());
    assert_eq!(
        result.error(),
        Some(&SolveError::DimensionMismatch {
            expected: 5,
            got: 2
        })
    );
}

/// Test with empty constraints.
struct EmptyConstraintsProblem;

impl Problem for EmptyConstraintsProblem {
    fn name(&self) -> &str {
        "empty"
    }
    fn residual_count(&self) -> usize {
        0
    }
    fn variable_count(&self) -> usize {
        2
    }
    fn residuals(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![]
    }
    fn initial_point(&self, _f: f64) -> Vec<f64> {
        vec![0.0, 0.0]
    }
}

impl DecomposableProblem for EmptyConstraintsProblem {}

#[test]
fn test_empty_constraints() {
    let problem = EmptyConstraintsProblem;
    let solver = ParallelSolver::default_solver();

    let result = solver.solve(&problem, &[1.0, 2.0]);

    // Should fail - no equations
    assert!(!result.is_completed());
    assert_eq!(result.error(), Some(&SolveError::NoEquations));
}

/// Test component properties.
#[test]
fn test_component_properties() {
    let component = Component {
        id: ComponentId(5),
        variable_indices: vec![1, 3, 5],
        constraint_indices: vec![0, 2],
    };

    assert_eq!(component.id, ComponentId(5));
    assert_eq!(component.variable_count(), 3);
    assert_eq!(component.constraint_count(), 2);
    assert!(!component.is_empty());
    assert!(!component.has_no_variables());

    let empty = Component {
        id: ComponentId(0),
        variable_indices: vec![1],
        constraint_indices: vec![],
    };
    assert!(empty.is_empty());

    let no_vars = Component {
        id: ComponentId(0),
        variable_indices: vec![],
        constraint_indices: vec![1],
    };
    assert!(no_vars.has_no_variables());
}

/// Test config presets.
#[test]
fn test_config_presets() {
    let many_small = ParallelSolverConfig::many_small();
    assert_eq!(many_small.min_parallel_size, 2);
    assert_eq!(many_small.min_parallel_components, 4);

    let few_large = ParallelSolverConfig::few_large();
    assert_eq!(few_large.min_parallel_size, 50);
    assert_eq!(few_large.min_parallel_components, 2);
}

/// Test out-of-bounds edges are handled gracefully.
#[test]
fn test_out_of_bounds_edges() {
    let edges = vec![
        (0, 0),
        (100, 0), // Out of bounds constraint
        (0, 100), // Out of bounds variable
    ];

    let components = decompose_from_edges(2, 2, &edges);

    // Should still create valid components, ignoring invalid edges
    assert!(!components.is_empty());
}
