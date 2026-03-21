//! Property-based tests for solverang.
//!
//! These tests use proptest to verify invariants that should hold for all valid inputs,
//! providing stronger guarantees than example-based testing alone.
//!
//! Run with: cargo test -p solverang --features parallel,sparse prop_

use proptest::prelude::*;
use solverang::{
    verify_jacobian, AutoSolver, LMConfig, LMSolver, Problem, RobustSolver, SolveResult,
};

// =============================================================================
// Test Problem Implementations
// =============================================================================

/// A simple configurable test problem for property testing.
/// F(x) = x - target, so the solution is x = target.
struct LinearProblem {
    target: Vec<f64>,
}

impl Problem for LinearProblem {
    fn name(&self) -> &str {
        "linear-test"
    }

    fn residual_count(&self) -> usize {
        self.target.len()
    }

    fn variable_count(&self) -> usize {
        self.target.len()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(&self.target).map(|(xi, ti)| xi - ti).collect()
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        // Identity matrix: dF_i/dx_j = 1 if i==j, 0 otherwise
        (0..self.target.len()).map(|i| (i, i, 1.0)).collect()
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor; self.target.len()]
    }
}

/// A quadratic problem for testing: x_i^2 - target_i = 0
struct QuadraticProblem {
    target: Vec<f64>,
}

impl Problem for QuadraticProblem {
    fn name(&self) -> &str {
        "quadratic-test"
    }

    fn residual_count(&self) -> usize {
        self.target.len()
    }

    fn variable_count(&self) -> usize {
        self.target.len()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.target)
            .map(|(xi, ti)| xi * xi - ti)
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        x.iter()
            .enumerate()
            .map(|(i, xi)| (i, i, 2.0 * xi))
            .collect()
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        self.target
            .iter()
            .map(|t| t.abs().sqrt() * factor)
            .collect()
    }
}

/// A coupled nonlinear problem: x_i + x_{i+1}^2 - target_i = 0
struct CoupledProblem {
    target: Vec<f64>,
}

impl Problem for CoupledProblem {
    fn name(&self) -> &str {
        "coupled-test"
    }

    fn residual_count(&self) -> usize {
        self.target.len()
    }

    fn variable_count(&self) -> usize {
        self.target.len()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        x.iter()
            .enumerate()
            .zip(&self.target)
            .map(|((i, xi), ti)| {
                let next = if i + 1 < n { x[i + 1] } else { 0.0 };
                xi + next * next - ti
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = x.len();
        let mut entries = Vec::new();

        for i in 0..n {
            // dF_i/dx_i = 1
            entries.push((i, i, 1.0));

            // dF_i/dx_{i+1} = 2 * x_{i+1}
            if i + 1 < n {
                entries.push((i, i + 1, 2.0 * x[i + 1]));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor; self.target.len()]
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a SolveResult is a valid variant (converged, not converged, or failed).
fn is_valid_result(result: &SolveResult) -> bool {
    matches!(
        result,
        SolveResult::Converged { .. }
            | SolveResult::NotConverged { .. }
            | SolveResult::Failed { .. }
    )
}

// =============================================================================
// Proptest Strategies
// =============================================================================

/// Strategy for generating reasonable coordinate values.
fn coord_strategy() -> impl Strategy<Value = f64> {
    prop::num::f64::NORMAL.prop_map(|x| {
        // Clamp to reasonable range to avoid numerical issues
        x.clamp(-1000.0, 1000.0)
    })
}

/// Strategy for generating a positive distance.
#[allow(dead_code)]
fn positive_distance_strategy() -> impl Strategy<Value = f64> {
    0.001f64..1000.0
}

/// Strategy for generating a small vector of coordinates.
fn small_vec_strategy(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(coord_strategy(), min_len..=max_len)
}

/// Strategy for generating strictly positive values.
fn positive_vec_strategy(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(1.0f64..100.0, min_len..=max_len)
}

// =============================================================================
// Property Tests: Jacobian Verification
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// The analytical Jacobian should match finite differences for linear problems.
    #[test]
    fn prop_linear_jacobian_matches_finite_diff(
        target in small_vec_strategy(1, 10),
        x in small_vec_strategy(1, 10),
    ) {
        // Skip if dimensions mismatch
        if target.len() != x.len() || target.is_empty() {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let verification = verify_jacobian(&problem, &x, 1e-7, 1e-5);

        prop_assert!(
            verification.passed,
            "Linear Jacobian mismatch: max_error={}, location={:?}",
            verification.max_absolute_error,
            verification.max_error_location
        );
    }

    /// The analytical Jacobian should match finite differences for quadratic problems.
    #[test]
    fn prop_quadratic_jacobian_matches_finite_diff(
        target in positive_vec_strategy(1, 10),
        x in small_vec_strategy(1, 10),
    ) {
        if target.len() != x.len() || target.is_empty() {
            return Ok(());
        }

        // Avoid points near zero where the Jacobian has singularities
        let x_safe: Vec<f64> = x.iter().map(|v| if v.abs() < 0.1 { 0.1 } else { *v }).collect();

        let problem = QuadraticProblem { target };
        let verification = verify_jacobian(&problem, &x_safe, 1e-7, 1e-4);

        prop_assert!(
            verification.passed,
            "Quadratic Jacobian mismatch: max_error={}, location={:?}",
            verification.max_absolute_error,
            verification.max_error_location
        );
    }

    /// The analytical Jacobian should match finite differences for coupled problems.
    #[test]
    fn prop_coupled_jacobian_matches_finite_diff(
        target in small_vec_strategy(2, 8),
        x in small_vec_strategy(2, 8),
    ) {
        if target.len() != x.len() || target.is_empty() {
            return Ok(());
        }

        // Filter out extreme values that cause numerical issues
        // Values too close to zero or too large can cause finite difference errors
        let x_safe: Vec<f64> = x.iter().map(|v| {
            if v.abs() < 1e-6 {
                1.0 // Replace tiny values with reasonable ones
            } else if v.abs() > 1e6 {
                v.signum() * 1e6
            } else {
                *v
            }
        }).collect();

        let problem = CoupledProblem { target };
        let verification = verify_jacobian(&problem, &x_safe, 1e-7, 1e-3);

        prop_assert!(
            verification.passed,
            "Coupled Jacobian mismatch: max_error={}, location={:?}",
            verification.max_absolute_error,
            verification.max_error_location
        );
    }
}

// =============================================================================
// Property Tests: Solver Behavior
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Solver should not crash on valid inputs (may not converge, but should not panic).
    #[test]
    fn prop_solver_no_crash_lm(
        target in small_vec_strategy(1, 5),
        x0 in small_vec_strategy(1, 5),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }

        // Filter out non-finite values
        if target.iter().any(|v| !v.is_finite()) || x0.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let solver = LMSolver::new(LMConfig::default());

        // Should not panic
        let result = solver.solve(&problem, &x0);

        // Result should be one of the valid variants
        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }

    /// Solver should not crash on valid inputs using AutoSolver.
    #[test]
    fn prop_solver_no_crash_auto(
        target in small_vec_strategy(1, 5),
        x0 in small_vec_strategy(1, 5),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }

        if target.iter().any(|v| !v.is_finite()) || x0.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let solver = AutoSolver::new();

        let result = solver.solve(&problem, &x0);

        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }

    /// Solver should not crash on valid inputs using RobustSolver.
    #[test]
    fn prop_solver_no_crash_robust(
        target in small_vec_strategy(1, 5),
        x0 in small_vec_strategy(1, 5),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }

        if target.iter().any(|v| !v.is_finite()) || x0.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let solver = RobustSolver::new();

        let result = solver.solve(&problem, &x0);

        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }

    /// Solution residual should be less than or equal to initial residual for converged problems.
    #[test]
    fn prop_solver_improves_residual(
        target in small_vec_strategy(1, 4),
        x0 in small_vec_strategy(1, 4),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }

        if target.iter().any(|v| !v.is_finite()) || x0.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let initial_residual = problem.residual_norm(&x0);

        // Skip if initial residual is already very small or non-finite
        if !initial_residual.is_finite() || initial_residual < 1e-10 {
            return Ok(());
        }

        let solver = LMSolver::new(LMConfig::default());
        let result = solver.solve(&problem, &x0);

        match result {
            SolveResult::Converged { residual_norm, .. } |
            SolveResult::NotConverged { residual_norm, .. } => {
                // Final residual should not be worse than initial (within tolerance)
                // Allow small increase due to numerical noise
                prop_assert!(
                    residual_norm <= initial_residual * 1.001 + 1e-10,
                    "Residual increased: initial={}, final={}",
                    initial_residual,
                    residual_norm
                );
            }
            SolveResult::Failed { .. } => {
                // Failure is acceptable for some inputs
            }
        }
    }

    /// Linear problems should always converge (they have exact solutions).
    #[test]
    fn prop_linear_always_converges(
        target in small_vec_strategy(1, 5),
    ) {
        if target.is_empty() || target.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target: target.clone() };
        let x0 = vec![0.0; target.len()];

        let solver = LMSolver::new(LMConfig::default());
        let result = solver.solve(&problem, &x0);

        prop_assert!(
            result.is_converged(),
            "Linear problem should converge, got {:?}",
            result
        );

        if let SolveResult::Converged { solution, .. } = result {
            // Solution should be close to target
            for (sol, tgt) in solution.iter().zip(&target) {
                prop_assert!(
                    (sol - tgt).abs() < 1e-5,
                    "Solution mismatch: sol={}, target={}",
                    sol,
                    tgt
                );
            }
        }
    }
}

// =============================================================================
// Property Tests: Numerical Stability
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Solver should handle large initial values gracefully.
    #[test]
    fn prop_solver_handles_large_values(
        target in prop::collection::vec(0.0f64..10.0, 2..4),
        scale in 1.0f64..1e6,
    ) {
        if target.is_empty() {
            return Ok(());
        }

        let problem = LinearProblem { target: target.clone() };
        let x0: Vec<f64> = target.iter().map(|_| scale).collect();

        let solver = LMSolver::new(LMConfig::robust());
        let result = solver.solve(&problem, &x0);

        // Should not panic and should return a valid result
        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }

    /// Solver should handle small initial values gracefully.
    #[test]
    fn prop_solver_handles_small_values(
        target in prop::collection::vec(0.0f64..10.0, 2..4),
        scale in 1e-10f64..1e-6,
    ) {
        if target.is_empty() {
            return Ok(());
        }

        let problem = LinearProblem { target: target.clone() };
        let x0: Vec<f64> = target.iter().map(|_| scale).collect();

        let solver = LMSolver::new(LMConfig::robust());
        let result = solver.solve(&problem, &x0);

        // Should not panic
        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }
}
