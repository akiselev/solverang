//! Parallel solver for independent constraint components.
//!
//! This module provides a solver that decomposes problems into independent
//! components and solves them in parallel using rayon.
//!
//! # Usage
//!
//! The parallel solver is most effective when:
//! - The problem naturally decomposes into multiple independent sub-problems
//! - Each sub-problem has enough work to justify parallel overhead
//! - You have multiple CPU cores available

use crate::decomposition::{decompose, Component, ComponentId, DecomposableProblem, SubProblem};
use crate::problem::Problem;
use crate::solver::auto::SolverChoice;
use crate::solver::config::SolverConfig;
use crate::solver::levenberg_marquardt::LMSolver;
use crate::solver::lm_config::LMConfig;
use crate::solver::newton_raphson::Solver as NRSolver;
use crate::solver::result::{SolveError, SolveResult};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for the parallel solver.
#[derive(Clone, Debug)]
pub struct ParallelSolverConfig {
    /// Minimum number of variables in a component to consider parallel solving.
    /// Components smaller than this will be solved sequentially to avoid overhead.
    pub min_parallel_size: usize,

    /// Minimum number of components to trigger parallel solving.
    /// If fewer components exist, sequential solving is used.
    pub min_parallel_components: usize,

    /// Which solver to use for each component.
    pub inner_solver: SolverChoice,

    /// Configuration for Newton-Raphson solver (if used).
    pub nr_config: SolverConfig,

    /// Configuration for Levenberg-Marquardt solver (if used).
    pub lm_config: LMConfig,

    /// Enable warm-start optimization for incremental solving.
    pub warm_start: bool,
}

impl Default for ParallelSolverConfig {
    fn default() -> Self {
        Self {
            min_parallel_size: 10,
            min_parallel_components: 2,
            inner_solver: SolverChoice::Auto,
            nr_config: SolverConfig::default(),
            lm_config: LMConfig::default(),
            warm_start: true,
        }
    }
}

impl ParallelSolverConfig {
    /// Create a configuration optimized for many small components.
    pub fn many_small() -> Self {
        Self {
            min_parallel_size: 2,
            min_parallel_components: 4,
            inner_solver: SolverChoice::NewtonRaphson,
            nr_config: SolverConfig::fast(),
            lm_config: LMConfig::fast(),
            warm_start: true,
        }
    }

    /// Create a configuration optimized for few large components.
    pub fn few_large() -> Self {
        Self {
            min_parallel_size: 50,
            min_parallel_components: 2,
            inner_solver: SolverChoice::Auto,
            nr_config: SolverConfig::default(),
            lm_config: LMConfig::robust(),
            warm_start: true,
        }
    }
}

/// Result from solving a single component.
#[derive(Clone, Debug)]
struct ComponentResult {
    #[allow(dead_code)]
    component_id: ComponentId,
    result: SolveResult,
    sub_problem: SubProblem,
}

/// Parallel solver that decomposes problems into independent components.
///
/// # How It Works
///
/// 1. Analyze the constraint-variable dependency graph
/// 2. Find connected components (groups of constraints that share variables)
/// 3. Solve each component independently (potentially in parallel)
/// 4. Merge solutions back into the full solution vector
///
/// # When To Use
///
/// This solver is beneficial when:
/// - Your problem consists of multiple independent sub-systems
/// - Each sub-system is non-trivial (many variables/constraints)
/// - You have multiple CPU cores
///
/// For fully-connected problems (single component), this adds overhead
/// without benefit. Use the regular solvers instead.
pub struct ParallelSolver {
    config: ParallelSolverConfig,
}

impl ParallelSolver {
    /// Create a new parallel solver with the given configuration.
    pub fn new(config: ParallelSolverConfig) -> Self {
        Self { config }
    }

    /// Create a parallel solver with default configuration.
    pub fn default_solver() -> Self {
        Self::new(ParallelSolverConfig::default())
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &ParallelSolverConfig {
        &self.config
    }

    /// Solve a decomposable problem.
    ///
    /// The problem is automatically decomposed into independent components
    /// which are solved in parallel (if the `parallel` feature is enabled
    /// and there are enough components).
    pub fn solve<P: DecomposableProblem>(&self, problem: &P, x0: &[f64]) -> SolveResult {
        let variable_count = problem.variable_count();
        let residual_count = problem.residual_count();

        // Validate dimensions
        if variable_count == 0 {
            return SolveResult::Failed {
                error: SolveError::NoVariables,
            };
        }

        if residual_count == 0 {
            return SolveResult::Failed {
                error: SolveError::NoEquations,
            };
        }

        if x0.len() != variable_count {
            return SolveResult::Failed {
                error: SolveError::DimensionMismatch {
                    expected: variable_count,
                    got: x0.len(),
                },
            };
        }

        // Decompose the problem
        let components = decompose(problem);

        // Handle edge cases
        if components.is_empty() {
            // No constraints to solve - return initial point as solution
            return SolveResult::Converged {
                solution: x0.to_vec(),
                iterations: 0,
                residual_norm: 0.0,
            };
        }

        // Filter out empty components
        let active_components: Vec<Component> =
            components.into_iter().filter(|c| !c.is_empty()).collect();

        if active_components.is_empty() {
            return SolveResult::Converged {
                solution: x0.to_vec(),
                iterations: 0,
                residual_norm: 0.0,
            };
        }

        // Single component - just solve directly
        if active_components.len() == 1 {
            return self.solve_single_component(problem, x0);
        }

        // Decide whether to parallelize
        let should_parallelize = self.should_parallelize(&active_components);

        // Solve components
        let component_results = if should_parallelize {
            self.solve_components_parallel(problem, x0, &active_components)
        } else {
            self.solve_components_sequential(problem, x0, &active_components)
        };

        // Merge results
        self.merge_results(x0, component_results, problem)
    }

    /// Solve with incremental updates for changed components only.
    ///
    /// This is an optimization for iterative solving where only some components
    /// have changed since the last solve. Unchanged components can reuse their
    /// previous solutions.
    pub fn solve_incremental<P: DecomposableProblem>(
        &self,
        problem: &P,
        x0: &[f64],
        changed_components: &[ComponentId],
    ) -> SolveResult {
        let variable_count = problem.variable_count();
        let residual_count = problem.residual_count();

        if variable_count == 0 {
            return SolveResult::Failed {
                error: SolveError::NoVariables,
            };
        }

        if residual_count == 0 {
            return SolveResult::Failed {
                error: SolveError::NoEquations,
            };
        }

        if x0.len() != variable_count {
            return SolveResult::Failed {
                error: SolveError::DimensionMismatch {
                    expected: variable_count,
                    got: x0.len(),
                },
            };
        }

        if changed_components.is_empty() {
            // No changes - return current point
            let residual_norm = problem.residual_norm(x0);
            return SolveResult::Converged {
                solution: x0.to_vec(),
                iterations: 0,
                residual_norm,
            };
        }

        let components = decompose(problem);

        // Only solve changed components
        let active_components: Vec<Component> = components
            .into_iter()
            .filter(|c| {
                !c.is_empty() && changed_components.iter().any(|changed| changed.0 == c.id.0)
            })
            .collect();

        if active_components.is_empty() {
            let residual_norm = problem.residual_norm(x0);
            return SolveResult::Converged {
                solution: x0.to_vec(),
                iterations: 0,
                residual_norm,
            };
        }

        let should_parallelize = self.should_parallelize(&active_components);
        let component_results = if should_parallelize {
            self.solve_components_parallel(problem, x0, &active_components)
        } else {
            self.solve_components_sequential(problem, x0, &active_components)
        };

        self.merge_results(x0, component_results, problem)
    }

    /// Determine whether parallelization is worthwhile.
    fn should_parallelize(&self, components: &[Component]) -> bool {
        #[cfg(feature = "parallel")]
        {
            if components.len() < self.config.min_parallel_components {
                return false;
            }

            // Check if components are large enough to justify parallel overhead
            let large_enough = components
                .iter()
                .filter(|c| c.variable_count() >= self.config.min_parallel_size)
                .count();

            large_enough >= self.config.min_parallel_components
        }

        #[cfg(not(feature = "parallel"))]
        {
            let _ = components;
            false
        }
    }

    /// Solve a problem with a single component (no decomposition benefit).
    fn solve_single_component<P: Problem>(&self, problem: &P, x0: &[f64]) -> SolveResult {
        match self.config.inner_solver {
            SolverChoice::NewtonRaphson => {
                let solver = NRSolver::new(self.config.nr_config.clone());
                solver.solve(problem, x0)
            }
            SolverChoice::LevenbergMarquardt => {
                let solver = LMSolver::new(self.config.lm_config.clone());
                solver.solve(problem, x0)
            }
            SolverChoice::Auto => {
                // Auto-select based on problem characteristics
                let m = problem.residual_count();
                let n = problem.variable_count();
                if m == n {
                    let solver = NRSolver::new(self.config.nr_config.clone());
                    solver.solve(problem, x0)
                } else {
                    let solver = LMSolver::new(self.config.lm_config.clone());
                    solver.solve(problem, x0)
                }
            }
        }
    }

    /// Solve components sequentially.
    fn solve_components_sequential<P: DecomposableProblem>(
        &self,
        problem: &P,
        x0: &[f64],
        components: &[Component],
    ) -> Vec<ComponentResult> {
        components
            .iter()
            .map(|component| self.solve_component(problem, x0, component))
            .collect()
    }

    /// Solve components in parallel using rayon.
    #[cfg(feature = "parallel")]
    fn solve_components_parallel<P: DecomposableProblem>(
        &self,
        problem: &P,
        x0: &[f64],
        components: &[Component],
    ) -> Vec<ComponentResult> {
        components
            .par_iter()
            .map(|component| self.solve_component(problem, x0, component))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    fn solve_components_parallel<P: DecomposableProblem>(
        &self,
        problem: &P,
        x0: &[f64],
        components: &[Component],
    ) -> Vec<ComponentResult> {
        self.solve_components_sequential(problem, x0, components)
    }

    /// Solve a single component.
    fn solve_component<P: DecomposableProblem>(
        &self,
        problem: &P,
        x0: &[f64],
        component: &Component,
    ) -> ComponentResult {
        let sub_problem = SubProblem::from_component(component.clone());
        let sub_x0 = sub_problem.extract_variables(x0);

        // Handle empty or trivial components
        if component.is_empty() || component.has_no_variables() {
            return ComponentResult {
                component_id: component.id,
                result: SolveResult::Converged {
                    solution: sub_x0,
                    iterations: 0,
                    residual_norm: 0.0,
                },
                sub_problem,
            };
        }

        // Create a wrapper problem for this component
        let component_problem = ComponentProblem {
            original: problem,
            sub_problem: &sub_problem,
        };

        // Solve the component
        let result = match self.config.inner_solver {
            SolverChoice::NewtonRaphson => {
                let solver = NRSolver::new(self.config.nr_config.clone());
                solver.solve(&component_problem, &sub_x0)
            }
            SolverChoice::LevenbergMarquardt => {
                let solver = LMSolver::new(self.config.lm_config.clone());
                solver.solve(&component_problem, &sub_x0)
            }
            SolverChoice::Auto => {
                let m = component_problem.residual_count();
                let n = component_problem.variable_count();
                if m == n {
                    let solver = NRSolver::new(self.config.nr_config.clone());
                    solver.solve(&component_problem, &sub_x0)
                } else {
                    let solver = LMSolver::new(self.config.lm_config.clone());
                    solver.solve(&component_problem, &sub_x0)
                }
            }
        };

        ComponentResult {
            component_id: component.id,
            result,
            sub_problem,
        }
    }

    /// Merge component results into a full solution.
    fn merge_results<P: Problem>(
        &self,
        x0: &[f64],
        results: Vec<ComponentResult>,
        problem: &P,
    ) -> SolveResult {
        let mut solution = x0.to_vec();
        let mut total_iterations = 0;
        let mut any_failed = false;
        let mut any_not_converged = false;
        let mut last_error = None;

        for component_result in results {
            match &component_result.result {
                SolveResult::Converged {
                    solution: sub_solution,
                    iterations,
                    ..
                } => {
                    component_result
                        .sub_problem
                        .inject_solution(sub_solution, &mut solution);
                    total_iterations += iterations;
                }
                SolveResult::NotConverged {
                    solution: sub_solution,
                    iterations,
                    ..
                } => {
                    any_not_converged = true;
                    // Still use the best solution found
                    component_result
                        .sub_problem
                        .inject_solution(sub_solution, &mut solution);
                    total_iterations += iterations;
                }
                SolveResult::Failed { error } => {
                    any_failed = true;
                    last_error = Some(error.clone());
                }
            }
        }

        if any_failed {
            if let Some(error) = last_error {
                return SolveResult::Failed { error };
            }
        }

        // Compute final residual norm
        let residual_norm = problem.residual_norm(&solution);

        if any_not_converged {
            let residuals = problem.residuals(&solution);
            return SolveResult::NotConverged {
                solution,
                iterations: total_iterations,
                residual_norm,
                residuals,
            };
        }

        SolveResult::Converged {
            solution,
            iterations: total_iterations,
            residual_norm,
        }
    }
}

impl Default for ParallelSolver {
    fn default() -> Self {
        Self::default_solver()
    }
}

/// A wrapper that presents a component as a standalone problem.
struct ComponentProblem<'a, P: Problem> {
    original: &'a P,
    sub_problem: &'a SubProblem,
}

impl<'a, P: Problem> Problem for ComponentProblem<'a, P> {
    fn name(&self) -> &str {
        self.original.name()
    }

    fn residual_count(&self) -> usize {
        self.sub_problem.component.constraint_count()
    }

    fn variable_count(&self) -> usize {
        self.sub_problem.component.variable_count()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // Build full x vector with sub-problem values
        let mut full_x = vec![0.0; self.original.variable_count()];
        self.sub_problem.inject_solution(x, &mut full_x);

        // Get all residuals
        let all_residuals = self.original.residuals(&full_x);

        // Extract only the residuals for this component
        self.sub_problem
            .constraint_mapping
            .iter()
            .filter_map(|&idx| all_residuals.get(idx).copied())
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        // Build full x vector
        let mut full_x = vec![0.0; self.original.variable_count()];
        self.sub_problem.inject_solution(x, &mut full_x);

        // Get full Jacobian
        let all_jacobian = self.original.jacobian(&full_x);

        // Create mappings for efficient lookup
        let constraint_to_local: std::collections::HashMap<usize, usize> = self
            .sub_problem
            .constraint_mapping
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        let variable_to_local: std::collections::HashMap<usize, usize> = self
            .sub_problem
            .variable_mapping
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        // Filter and remap Jacobian entries
        all_jacobian
            .into_iter()
            .filter_map(|(row, col, val)| {
                let local_row = constraint_to_local.get(&row)?;
                let local_col = variable_to_local.get(&col)?;
                Some((*local_row, *local_col, val))
            })
            .collect()
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        let full_initial = self.original.initial_point(factor);
        self.sub_problem.extract_variables(&full_initial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A decomposable test problem with multiple independent components
    // Two independent 2D problems
    // Component 1: x0^2 + x1^2 = 1 (circle)
    // Component 2: x2 - 1 = 0, x3 - 2 = 0 (independent point)
    struct MultiComponentProblem;

    impl Problem for MultiComponentProblem {
        fn name(&self) -> &str {
            "multi-component"
        }

        fn residual_count(&self) -> usize {
            3
        }

        fn variable_count(&self) -> usize {
            4
        }

        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![
                x[0] * x[0] + x[1] * x[1] - 1.0, // Circle constraint
                x[2] - 1.0,                      // x2 = 1
                x[3] - 2.0,                      // x3 = 2
            ]
        }

        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 2.0 * x[0]),
                (0, 1, 2.0 * x[1]),
                (1, 2, 1.0),
                (2, 3, 1.0),
            ]
        }

        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![0.5 * factor, 0.5 * factor, 0.0, 0.0]
        }
    }

    impl DecomposableProblem for MultiComponentProblem {
        fn constraint_graph(&self) -> Vec<(usize, usize)> {
            vec![
                (0, 0),
                (0, 1), // Constraint 0 uses x0, x1
                (1, 2), // Constraint 1 uses x2
                (2, 3), // Constraint 2 uses x3
            ]
        }
    }

    // A single-component problem
    struct SingleComponentProblem;

    impl Problem for SingleComponentProblem {
        fn name(&self) -> &str {
            "single-component"
        }

        fn residual_count(&self) -> usize {
            2
        }

        fn variable_count(&self) -> usize {
            2
        }

        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] - 2.0, x[1] - x[0]]
        }

        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 2.0 * x[0]), (0, 1, 0.0), (1, 0, -1.0), (1, 1, 1.0)]
        }

        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![1.0 * factor, 1.0 * factor]
        }
    }

    impl DecomposableProblem for SingleComponentProblem {
        fn constraint_graph(&self) -> Vec<(usize, usize)> {
            vec![(0, 0), (0, 1), (1, 0), (1, 1)]
        }
    }

    #[test]
    fn test_multi_component_decomposition() {
        let problem = MultiComponentProblem;
        let components = decompose(&problem);

        // Should have 2 components: {c0} with {x0, x1}, and {c1, c2} with {x2, x3}
        // Actually c1 and c2 are independent, so 3 components
        assert!(components.len() >= 2);
    }

    #[test]
    fn test_parallel_solver_multi_component() {
        let problem = MultiComponentProblem;
        let solver = ParallelSolver::default_solver();
        let x0 = vec![0.5, 0.5, 0.0, 0.0];

        let result = solver.solve(&problem, &x0);

        assert!(
            result.is_converged() || result.is_completed(),
            "Result: {:?}",
            result
        );

        if let Some(solution) = result.solution() {
            // Check circle constraint: x0^2 + x1^2 = 1
            let circle_residual = solution[0] * solution[0] + solution[1] * solution[1] - 1.0;
            assert!(
                circle_residual.abs() < 1e-4,
                "Circle constraint not satisfied: {}",
                circle_residual
            );

            // Check point constraints
            assert!(
                (solution[2] - 1.0).abs() < 1e-6,
                "x2 should be 1, got {}",
                solution[2]
            );
            assert!(
                (solution[3] - 2.0).abs() < 1e-6,
                "x3 should be 2, got {}",
                solution[3]
            );
        }
    }

    #[test]
    fn test_parallel_solver_single_component() {
        let problem = SingleComponentProblem;
        let solver = ParallelSolver::default_solver();
        let x0 = vec![1.5, 1.5];

        let result = solver.solve(&problem, &x0);

        assert!(result.is_converged(), "Result: {:?}", result);

        if let Some(solution) = result.solution() {
            // x = sqrt(2), y = sqrt(2)
            let expected = std::f64::consts::SQRT_2;
            assert!(
                (solution[0] - expected).abs() < 1e-6,
                "x should be sqrt(2), got {}",
                solution[0]
            );
            assert!(
                (solution[1] - expected).abs() < 1e-6,
                "y should be sqrt(2), got {}",
                solution[1]
            );
        }
    }

    #[test]
    fn test_parallel_solver_dimension_mismatch() {
        let problem = SingleComponentProblem;
        let solver = ParallelSolver::default_solver();

        let result = solver.solve(&problem, &[1.0]); // Wrong size
        assert!(!result.is_completed());
        assert_eq!(
            result.error(),
            Some(&SolveError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        );
    }

    #[test]
    fn test_incremental_solve() {
        let problem = MultiComponentProblem;
        let solver = ParallelSolver::default_solver();
        let x0 = vec![0.7, 0.7, 1.0, 2.0]; // Already close to solution

        // Only solve component 0 (the circle)
        let result = solver.solve_incremental(&problem, &x0, &[ComponentId(0)]);

        assert!(result.is_converged() || result.is_completed());
    }

    #[test]
    fn test_incremental_solve_no_changes() {
        let problem = MultiComponentProblem;
        let solver = ParallelSolver::default_solver();
        let x0 = vec![0.7, 0.7, 1.0, 2.0];

        // No changed components
        let result = solver.solve_incremental(&problem, &x0, &[]);

        assert!(result.is_converged());
        assert_eq!(result.iterations(), Some(0));
    }

    #[test]
    fn test_parallel_config_presets() {
        let _config = ParallelSolverConfig::many_small();
        let _config = ParallelSolverConfig::few_large();
    }

    // Empty problem
    struct EmptyProblem;

    impl Problem for EmptyProblem {
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
        fn initial_point(&self, _factor: f64) -> Vec<f64> {
            vec![0.0, 0.0]
        }
    }

    impl DecomposableProblem for EmptyProblem {}

    #[test]
    fn test_empty_problem() {
        let problem = EmptyProblem;
        let solver = ParallelSolver::default_solver();

        let result = solver.solve(&problem, &[1.0, 2.0]);
        assert!(!result.is_completed());
        assert_eq!(result.error(), Some(&SolveError::NoEquations));
    }

    // Problem with no variables
    struct NoVariablesProblem;

    impl Problem for NoVariablesProblem {
        fn name(&self) -> &str {
            "no-vars"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            0
        }
        fn residuals(&self, _x: &[f64]) -> Vec<f64> {
            vec![1.0]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![]
        }
        fn initial_point(&self, _factor: f64) -> Vec<f64> {
            vec![]
        }
    }

    impl DecomposableProblem for NoVariablesProblem {}

    #[test]
    fn test_no_variables_problem() {
        let problem = NoVariablesProblem;
        let solver = ParallelSolver::default_solver();

        let result = solver.solve(&problem, &[]);
        assert!(!result.is_completed());
        assert_eq!(result.error(), Some(&SolveError::NoVariables));
    }
}
