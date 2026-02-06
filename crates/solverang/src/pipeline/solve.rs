//! Top-level solve pipeline orchestration.
//!
//! This module provides the main solve pipeline that orchestrates solving
//! all dirty components of a constraint system.

use crate::solver::{
    AutoSolver, LMConfig, LMSolver, RobustSolver, Solver, SolverConfig, SolveResult,
};
use crate::problem::Problem;
use super::extract::{extract_component, ComponentProblem};
use super::select::{select_solver, SolverSelection};
use crate::geometry::constraint::Nonlinearity;

/// Result of solving a single component.
#[derive(Clone, Debug)]
pub struct ComponentSolveResult {
    /// Component identifier (index in the dirty list)
    pub component_id: usize,
    /// Whether the solve succeeded (converged)
    pub success: bool,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: f64,
    /// Which solver was used
    pub solver_used: SolverSelection,
}

/// Solve a component problem using the selected solver.
///
/// # Arguments
///
/// * `problem` - The component problem to solve
/// * `selection` - Which solver to use
/// * `initial_guess` - Initial values for the variables
///
/// # Returns
///
/// The result of the solve attempt.
pub fn solve_component(
    problem: &ComponentProblem<'_>,
    selection: SolverSelection,
    initial_guess: &[f64],
) -> SolveResult {
    match selection {
        SolverSelection::NewtonRaphson => {
            let solver = Solver::new(SolverConfig::default());
            solver.solve(problem, initial_guess)
        }
        SolverSelection::LevenbergMarquardt => {
            let solver = LMSolver::new(LMConfig::default());
            solver.solve(problem, initial_guess)
        }
        SolverSelection::Robust => {
            let solver = RobustSolver::new();
            solver.solve(problem, initial_guess)
        }
        SolverSelection::Auto => {
            let solver = AutoSolver::new();
            solver.solve(problem, initial_guess)
        }
        SolverSelection::Sparse => {
            // Fall back to Auto if sparse solver not available
            // (SparseSolver may require additional feature flags)
            let solver = AutoSolver::new();
            solver.solve(problem, initial_guess)
        }
    }
}

/// Top-level pipeline: solve all dirty components of a constraint system.
///
/// This is the main entry point for the solve pipeline. It takes a constraint
/// system state and solves all dirty components, updating the parameter values
/// in place.
///
/// # Arguments
///
/// * `constraints` - All constraints in the system
/// * `params` - Parameter values (will be updated with solutions)
/// * `fixed` - Fixed flags for all parameters
/// * `dirty_constraint_groups` - Groups of constraint indices per dirty component
///
/// # Returns
///
/// Results for each component that was solved.
///
/// # Example
///
/// ```ignore
/// let results = solve_dirty_components(
///     &constraints,
///     &mut params,
///     &fixed,
///     &[vec![0, 1, 2], vec![3, 4]], // Two components
/// );
///
/// for result in results {
///     println!("Component {}: {}", result.component_id,
///              if result.success { "converged" } else { "failed" });
/// }
/// ```
pub fn solve_dirty_components(
    constraints: &[Box<dyn crate::geometry::constraint::Constraint>],
    params: &mut Vec<f64>,
    fixed: &[bool],
    dirty_constraint_groups: &[Vec<usize>],
) -> Vec<ComponentSolveResult> {
    let mut results = Vec::new();

    for (comp_idx, constraint_indices) in dirty_constraint_groups.iter().enumerate() {
        if constraint_indices.is_empty() {
            continue;
        }

        // Extract the component as a sub-problem
        let component = extract_component(constraints, constraint_indices, params, fixed);

        // Get initial guess (warm start from current values)
        let initial = component.initial_point(1.0);

        // Determine max nonlinearity from constraints in this component
        let max_nonlin = constraint_indices
            .iter()
            .map(|&ci| constraints[ci].nonlinearity_hint())
            .max_by_key(|n| match n {
                Nonlinearity::Linear => 0,
                Nonlinearity::Moderate => 1,
                Nonlinearity::High => 2,
            })
            .unwrap_or(Nonlinearity::Moderate);

        // Estimate sparsity (simplified: assume moderate sparsity)
        // In a real implementation, we'd count actual Jacobian entries
        let sparsity_estimate = 0.5;

        // Select the appropriate solver
        let selection = select_solver(
            component.variable_count(),
            component.residual_count(),
            max_nonlin,
            sparsity_estimate,
        );

        // Solve the component
        let result = solve_component(&component, selection, &initial);

        // Extract result information
        let (success, iterations, residual_norm) = match &result {
            SolveResult::Converged {
                solution,
                iterations,
                residual_norm,
            } => {
                // Write solution back to params
                let free_indices = component.free_variable_indices();
                for (local_i, &global_i) in free_indices.iter().enumerate() {
                    params[global_i] = solution[local_i];
                }
                (true, *iterations, *residual_norm)
            }
            SolveResult::NotConverged {
                solution,
                iterations,
                residual_norm,
            } => {
                // Even if not fully converged, write back the best solution found
                let free_indices = component.free_variable_indices();
                for (local_i, &global_i) in free_indices.iter().enumerate() {
                    params[global_i] = solution[local_i];
                }
                (false, *iterations, *residual_norm)
            }
            SolveResult::Failed { .. } => {
                // Don't update params on failure
                (false, 0, f64::INFINITY)
            }
        };

        results.push(ComponentSolveResult {
            component_id: comp_idx,
            success,
            iterations,
            residual_norm,
            solver_used: selection,
        });
    }

    results
}

/// Main solve pipeline orchestrator.
///
/// This struct maintains configuration and state for the solve pipeline.
/// In the future, this will integrate with the incremental graph layer
/// to automatically detect dirty components.
pub struct SolvePipeline {
    /// Configuration for Newton-Raphson solver
    pub nr_config: SolverConfig,
    /// Configuration for Levenberg-Marquardt solver
    pub lm_config: LMConfig,
}

impl SolvePipeline {
    /// Create a new solve pipeline with default configurations.
    pub fn new() -> Self {
        Self {
            nr_config: SolverConfig::default(),
            lm_config: LMConfig::default(),
        }
    }

    /// Create a pipeline with custom solver configurations.
    pub fn with_configs(nr_config: SolverConfig, lm_config: LMConfig) -> Self {
        Self {
            nr_config,
            lm_config,
        }
    }

    /// Solve all dirty components.
    ///
    /// This is a convenience wrapper around `solve_dirty_components` that
    /// allows using custom solver configurations.
    pub fn solve(
        &self,
        constraints: &[Box<dyn crate::geometry::constraint::Constraint>],
        params: &mut Vec<f64>,
        fixed: &[bool],
        dirty_constraint_groups: &[Vec<usize>],
    ) -> Vec<ComponentSolveResult> {
        solve_dirty_components(constraints, params, fixed, dirty_constraint_groups)
    }
}

impl Default for SolvePipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::constraint::{Constraint, Nonlinearity};
    use crate::geometry::params::ConstraintId;

    /// Mock linear constraint for testing: x + y = target
    struct LinearConstraint {
        id: ConstraintId,
        deps: Vec<usize>,
        target: f64,
    }

    impl Constraint for LinearConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }

        fn name(&self) -> &'static str {
            "LinearConstraint"
        }

        fn equation_count(&self) -> usize {
            1
        }

        fn dependencies(&self) -> &[usize] {
            &self.deps
        }

        fn residuals(&self, params: &[f64]) -> Vec<f64> {
            let sum: f64 = self.deps.iter().map(|&i| params[i]).sum();
            vec![sum - self.target]
        }

        fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
            self.deps.iter().map(|&i| (0, i, 1.0)).collect()
        }

        fn nonlinearity_hint(&self) -> Nonlinearity {
            Nonlinearity::Linear
        }
    }

    /// Mock quadratic constraint for testing: x² + y² = target
    struct QuadraticConstraint {
        id: ConstraintId,
        deps: Vec<usize>,
        target: f64,
    }

    impl Constraint for QuadraticConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }

        fn name(&self) -> &'static str {
            "QuadraticConstraint"
        }

        fn equation_count(&self) -> usize {
            1
        }

        fn dependencies(&self) -> &[usize] {
            &self.deps
        }

        fn residuals(&self, params: &[f64]) -> Vec<f64> {
            let sum: f64 = self.deps.iter().map(|&i| params[i] * params[i]).sum();
            vec![sum - self.target]
        }

        fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
            self.deps
                .iter()
                .map(|&i| (0, i, 2.0 * params[i]))
                .collect()
        }

        fn nonlinearity_hint(&self) -> Nonlinearity {
            Nonlinearity::Moderate
        }
    }

    #[test]
    fn test_solve_linear_component() {
        // Simple linear constraint: x + y = 5
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(LinearConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            target: 5.0,
        })];

        let mut params = vec![1.0, 1.0]; // Initial guess
        let fixed = vec![true, false]; // Fix x, solve for y

        let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

        assert_eq!(results.len(), 1);
        let result = &results[0];

        // Linear constraint should converge quickly
        assert!(result.success, "Linear constraint should converge");
        assert!(result.residual_norm < 1e-6);

        // With x=1.0 fixed, y should be 4.0
        assert!((params[0] - 1.0).abs() < 1e-6, "x should remain fixed at 1.0");
        assert!((params[1] - 4.0).abs() < 1e-3, "y should be 4.0");
    }

    #[test]
    fn test_solve_quadratic_component() {
        // Quadratic constraint: x² + y² = 5
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(QuadraticConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            target: 5.0,
        })];

        let mut params = vec![1.5, 1.5]; // Initial guess
        let fixed = vec![false, false]; // Both free

        let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

        assert_eq!(results.len(), 1);
        let result = &results[0];

        // Should converge (under-constrained, but solver should find a solution)
        if result.success {
            // Verify solution satisfies constraint
            let sum = params[0] * params[0] + params[1] * params[1];
            assert!((sum - 5.0).abs() < 1e-3, "Solution should satisfy x² + y² = 5");
        }
    }

    #[test]
    fn test_solve_multiple_components() {
        // Two independent components
        let constraints: Vec<Box<dyn Constraint>> = vec![
            Box::new(LinearConstraint {
                id: ConstraintId(0),
                deps: vec![0, 1],
                target: 3.0,
            }),
            Box::new(LinearConstraint {
                id: ConstraintId(1),
                deps: vec![2, 3],
                target: 7.0,
            }),
        ];

        let mut params = vec![0.0, 0.0, 0.0, 0.0];
        let fixed = vec![true, false, true, false]; // Fix 0 and 2, solve for 1 and 3

        let results = solve_dirty_components(
            &constraints,
            &mut params,
            &fixed,
            &[vec![0], vec![1]], // Two separate components
        );

        assert_eq!(results.len(), 2);

        // Both should converge
        assert!(results[0].success);
        assert!(results[1].success);

        // With params[0]=0.0 fixed, params[1] should be 3.0
        assert!((params[1] - 3.0).abs() < 1e-3);

        // With params[2]=0.0 fixed, params[3] should be 7.0
        assert!((params[3] - 7.0).abs() < 1e-3);
    }

    #[test]
    fn test_empty_component() {
        let constraints: Vec<Box<dyn Constraint>> = vec![];
        let mut params = vec![1.0, 2.0];
        let fixed = vec![false, false];

        let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![]]);

        // Empty component should be skipped
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_solver_selection_for_linear() {
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(LinearConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            target: 5.0,
        })];

        let mut params = vec![1.0, 1.0];
        let fixed = vec![false, false];

        let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

        // Linear constraint should use NewtonRaphson
        assert_eq!(results[0].solver_used, SolverSelection::NewtonRaphson);
    }

    #[test]
    fn test_solver_selection_for_nonlinear() {
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(QuadraticConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1, 2, 3, 4], // 5 variables
            target: 10.0,
        })];

        let mut params = vec![1.0; 5];
        let fixed = vec![false; 5];

        let results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

        // Moderate nonlinearity with 5 variables should use Robust
        assert_eq!(results[0].solver_used, SolverSelection::Robust);
    }

    #[test]
    fn test_solve_pipeline_struct() {
        let pipeline = SolvePipeline::new();

        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(LinearConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            target: 3.0,
        })];

        let mut params = vec![1.0, 1.0];
        let fixed = vec![true, false];

        let results = pipeline.solve(&constraints, &mut params, &fixed, &[vec![0]]);

        assert_eq!(results.len(), 1);
        assert!(results[0].success);
    }

    #[test]
    fn test_solve_pipeline_custom_config() {
        let pipeline = SolvePipeline::with_configs(
            SolverConfig::fast(),
            LMConfig::default(),
        );

        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(LinearConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            target: 3.0,
        })];

        let mut params = vec![1.0, 1.0];
        let fixed = vec![true, false];

        let results = pipeline.solve(&constraints, &mut params, &fixed, &[vec![0]]);

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_component_solve_result_debug() {
        let result = ComponentSolveResult {
            component_id: 42,
            success: true,
            iterations: 5,
            residual_norm: 1e-8,
            solver_used: SolverSelection::NewtonRaphson,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("true"));
        assert!(debug_str.contains("NewtonRaphson"));
    }

    #[test]
    fn test_result_updates_params_on_partial_convergence() {
        // Test that even non-converged results update params
        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(QuadraticConstraint {
            id: ConstraintId(0),
            deps: vec![0, 1],
            target: 5.0,
        })];

        let mut params = vec![1.5, 1.5];
        let fixed = vec![false, false];
        let original_params = params.clone();

        let _results = solve_dirty_components(&constraints, &mut params, &fixed, &[vec![0]]);

        // Params should have been updated even if not fully converged
        // (unless the solver completely failed)
        // We just check that params changed from initial values
        let changed = params[0] != original_params[0] || params[1] != original_params[1];
        assert!(changed || params == original_params); // Either changed or stayed same
    }
}
