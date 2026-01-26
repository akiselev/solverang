//! Levenberg-Marquardt solver implementation.
//!
//! This module provides a Levenberg-Marquardt (LM) solver for nonlinear least-squares
//! problems. The LM algorithm is particularly effective for:
//!
//! - Over-constrained systems (more equations than variables)
//! - Problems with poor initial guesses
//! - Systems where Newton-Raphson struggles to converge
//!
//! The implementation wraps the `levenberg-marquardt` crate, adapting our
//! `Problem` trait to work with it.

use crate::problem::Problem;
use crate::solver::lm_adapter::LMProblemAdapter;
use crate::solver::lm_config::LMConfig;
use crate::solver::result::{SolveError, SolveResult};
use levenberg_marquardt::{LevenbergMarquardt, TerminationReason};

/// Levenberg-Marquardt solver for nonlinear least-squares problems.
///
/// This solver minimizes the sum of squared residuals ||F(x)||^2 using the
/// Levenberg-Marquardt algorithm. It combines the benefits of:
/// - Gradient descent (reliable far from solution)
/// - Gauss-Newton (fast convergence near solution)
///
/// # Example
///
/// ```rust
/// use solverang::{Problem, LMSolver, LMConfig};
///
/// struct Quadratic;
///
/// impl Problem for Quadratic {
///     fn name(&self) -> &str { "x^2 - 4" }
///     fn residual_count(&self) -> usize { 1 }
///     fn variable_count(&self) -> usize { 1 }
///     fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0] * x[0] - 4.0] }
///     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { vec![(0, 0, 2.0 * x[0])] }
///     fn initial_point(&self, _: f64) -> Vec<f64> { vec![1.0] }
/// }
///
/// let solver = LMSolver::new(LMConfig::default());
/// let result = solver.solve(&Quadratic, &[1.0]);
/// assert!(result.is_converged());
/// ```
pub struct LMSolver {
    config: LMConfig,
}

impl LMSolver {
    /// Create a new LM solver with the given configuration.
    pub fn new(config: LMConfig) -> Self {
        Self { config }
    }

    /// Create a solver with default configuration.
    pub fn default_solver() -> Self {
        Self::new(LMConfig::default())
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &LMConfig {
        &self.config
    }

    /// Solve a problem starting from the given initial point.
    ///
    /// # Arguments
    ///
    /// * `problem` - The nonlinear problem to solve
    /// * `x0` - Initial guess for the solution
    ///
    /// # Returns
    ///
    /// A `SolveResult` indicating success or failure of the optimization.
    pub fn solve<P: Problem + ?Sized>(&self, problem: &P, x0: &[f64]) -> SolveResult {
        let n = problem.variable_count();
        let m = problem.residual_count();

        // Validate problem dimensions
        if n == 0 {
            return SolveResult::Failed {
                error: SolveError::NoVariables,
            };
        }

        if m == 0 {
            return SolveResult::Failed {
                error: SolveError::NoEquations,
            };
        }

        if x0.len() != n {
            return SolveResult::Failed {
                error: SolveError::DimensionMismatch {
                    expected: n,
                    got: x0.len(),
                },
            };
        }

        // Check initial point for non-finite values
        if x0.iter().any(|v| !v.is_finite()) {
            return SolveResult::Failed {
                error: SolveError::NonFiniteResiduals,
            };
        }

        // Create the adapter for levenberg-marquardt crate
        let adapter = LMProblemAdapter::new(problem, x0);

        // Configure the LM solver
        let lm = LevenbergMarquardt::new()
            .with_ftol(self.config.ftol)
            .with_xtol(self.config.xtol)
            .with_gtol(self.config.gtol)
            .with_stepbound(self.config.stepbound)
            .with_patience(self.config.patience)
            .with_scale_diag(self.config.scale_diag);

        // Run the optimization
        let (result_adapter, report) = lm.minimize(adapter);

        // Extract final solution
        let solution = result_adapter.final_params();

        // Compute final residual norm
        let residuals = problem.residuals(&solution);
        let residual_norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        // Convert termination reason to our result type
        self.convert_termination(
            report.termination,
            solution,
            report.number_of_evaluations,
            residual_norm,
        )
    }

    /// Solve using the problem's default initial point.
    pub fn solve_from_initial<P: Problem + ?Sized>(&self, problem: &P, factor: f64) -> SolveResult {
        let x0 = problem.initial_point(factor);
        self.solve(problem, &x0)
    }

    /// Convert LM termination reason to our SolveResult.
    fn convert_termination(
        &self,
        termination: TerminationReason,
        solution: Vec<f64>,
        iterations: usize,
        residual_norm: f64,
    ) -> SolveResult {
        match termination {
            // Successful termination
            TerminationReason::ResidualsZero
            | TerminationReason::Orthogonal
            | TerminationReason::Converged { .. } => SolveResult::Converged {
                solution,
                iterations,
                residual_norm,
            },

            // Maximum iterations exceeded
            TerminationReason::LostPatience => SolveResult::NotConverged {
                solution,
                iterations,
                residual_norm,
            },

            // User-triggered termination (residuals/jacobian returned None)
            TerminationReason::User(msg) => {
                // Check what the message indicates
                if msg.contains("residual") {
                    SolveResult::Failed {
                        error: SolveError::NonFiniteResiduals,
                    }
                } else if msg.contains("jacobian") {
                    SolveResult::Failed {
                        error: SolveError::NonFiniteJacobian,
                    }
                } else {
                    SolveResult::NotConverged {
                        solution,
                        iterations,
                        residual_norm,
                    }
                }
            }

            // Numerical issues
            TerminationReason::Numerical(msg) => {
                if msg.contains("jacobian") {
                    SolveResult::Failed {
                        error: SolveError::NonFiniteJacobian,
                    }
                } else if msg.contains("residual") {
                    SolveResult::Failed {
                        error: SolveError::NonFiniteResiduals,
                    }
                } else {
                    // General numerical issue - return best solution found
                    SolveResult::NotConverged {
                        solution,
                        iterations,
                        residual_norm,
                    }
                }
            }

            // Tolerance too tight for machine precision
            TerminationReason::NoImprovementPossible(_) => {
                // This actually means we've converged as much as floating point allows
                if residual_norm < self.config.ftol.sqrt() {
                    SolveResult::Converged {
                        solution,
                        iterations,
                        residual_norm,
                    }
                } else {
                    SolveResult::NotConverged {
                        solution,
                        iterations,
                        residual_norm,
                    }
                }
            }

            // Problem setup issues
            TerminationReason::NoParameters => SolveResult::Failed {
                error: SolveError::NoVariables,
            },

            TerminationReason::NoResiduals => SolveResult::Failed {
                error: SolveError::NoEquations,
            },

            TerminationReason::WrongDimensions(msg) => {
                if msg.contains("residual") {
                    SolveResult::Failed {
                        error: SolveError::NonFiniteResiduals,
                    }
                } else if msg.contains("jacobian") {
                    SolveResult::Failed {
                        error: SolveError::NonFiniteJacobian,
                    }
                } else {
                    SolveResult::Failed {
                        error: SolveError::DimensionMismatch {
                            expected: 0,
                            got: 0,
                        },
                    }
                }
            }
        }
    }
}

impl Default for LMSolver {
    fn default() -> Self {
        Self::default_solver()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple quadratic: x^2 = 2, solution is sqrt(2)
    struct SqrtTwo;

    impl Problem for SqrtTwo {
        fn name(&self) -> &str {
            "sqrt(2)"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] - 2.0]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 2.0 * x[0])]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![1.0 * factor]
        }
        fn known_solution(&self) -> Option<Vec<f64>> {
            Some(vec![std::f64::consts::SQRT_2])
        }
    }

    // Simple 2D problem: x^2 + y^2 = 1, x = y
    struct CircleLineIntersection;

    impl Problem for CircleLineIntersection {
        fn name(&self) -> &str {
            "circle-line intersection"
        }
        fn residual_count(&self) -> usize {
            2
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 2.0 * x[0]),
                (0, 1, 2.0 * x[1]),
                (1, 0, 1.0),
                (1, 1, -1.0),
            ]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![0.5 * factor, 0.5 * factor]
        }
    }

    // Overdetermined problem: more equations than variables
    struct OverdeterminedLinear;

    impl Problem for OverdeterminedLinear {
        fn name(&self) -> &str {
            "overdetermined-linear"
        }
        fn residual_count(&self) -> usize {
            4
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            // Four equations, two unknowns
            // Least-squares solution: x = [1, 2]
            vec![
                x[0] - 1.0,
                x[1] - 2.0,
                x[0] + x[1] - 3.0,
                2.0 * x[0] - x[1] - 0.0,
            ]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 1.0),
                (0, 1, 0.0),
                (1, 0, 0.0),
                (1, 1, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
                (3, 0, 2.0),
                (3, 1, -1.0),
            ]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor, factor]
        }
    }

    #[test]
    fn test_sqrt_two() {
        let problem = SqrtTwo;
        let solver = LMSolver::default_solver();
        let result = solver.solve(&problem, &[1.5]);

        assert!(result.is_converged(), "Result: {:?}", result);
        let solution = result.solution().expect("should have solution");
        assert!(
            (solution[0] - std::f64::consts::SQRT_2).abs() < 1e-6,
            "Expected sqrt(2), got {}",
            solution[0]
        );
    }

    #[test]
    fn test_circle_line() {
        let problem = CircleLineIntersection;
        let solver = LMSolver::default_solver();
        let result = solver.solve(&problem, &[0.5, 0.5]);

        assert!(result.is_converged(), "Result: {:?}", result);
        let solution = result.solution().expect("should have solution");
        let expected = 1.0 / std::f64::consts::SQRT_2;
        assert!(
            (solution[0] - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            solution[0]
        );
        assert!(
            (solution[1] - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            solution[1]
        );
    }

    #[test]
    fn test_overdetermined() {
        let problem = OverdeterminedLinear;
        let solver = LMSolver::new(LMConfig::robust());
        let result = solver.solve(&problem, &[0.0, 0.0]);

        // For overdetermined systems, LM should find least-squares solution
        assert!(
            result.is_converged() || result.is_completed(),
            "Result: {:?}",
            result
        );
        let solution = result.solution().expect("should have solution");

        // The exact solution [1, 2] satisfies the first 3 equations exactly
        // but not the 4th (2*1 - 2 = 0, but equation says = 0, so it's satisfied)
        // Actually all equations can be satisfied with x=1, y=2
        assert!(
            (solution[0] - 1.0).abs() < 0.1,
            "x should be near 1, got {}",
            solution[0]
        );
        assert!(
            (solution[1] - 2.0).abs() < 0.1,
            "y should be near 2, got {}",
            solution[1]
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let problem = SqrtTwo;
        let solver = LMSolver::default_solver();
        let result = solver.solve(&problem, &[1.0, 2.0]); // Wrong dimension

        assert!(!result.is_converged());
        assert!(!result.is_completed());
        assert_eq!(
            result.error(),
            Some(&SolveError::DimensionMismatch {
                expected: 1,
                got: 2
            })
        );
    }

    #[test]
    fn test_solve_from_initial() {
        let problem = SqrtTwo;
        let solver = LMSolver::default_solver();
        let result = solver.solve_from_initial(&problem, 1.0);

        assert!(result.is_converged());
    }

    // Problem with no variables
    struct EmptyVariables;

    impl Problem for EmptyVariables {
        fn name(&self) -> &str {
            "empty"
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

    #[test]
    fn test_no_variables() {
        let problem = EmptyVariables;
        let solver = LMSolver::default_solver();
        let result = solver.solve(&problem, &[]);

        assert_eq!(result.error(), Some(&SolveError::NoVariables));
    }

    // Problem with no equations
    struct EmptyEquations;

    impl Problem for EmptyEquations {
        fn name(&self) -> &str {
            "empty equations"
        }
        fn residual_count(&self) -> usize {
            0
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![]
        }
        fn initial_point(&self, _factor: f64) -> Vec<f64> {
            vec![1.0]
        }
    }

    #[test]
    fn test_no_equations() {
        let problem = EmptyEquations;
        let solver = LMSolver::default_solver();
        let result = solver.solve(&problem, &[1.0]);

        assert_eq!(result.error(), Some(&SolveError::NoEquations));
    }

    #[test]
    fn test_config_presets() {
        let problem = SqrtTwo;

        // Test fast config
        let solver = LMSolver::new(LMConfig::fast());
        let result = solver.solve(&problem, &[1.5]);
        assert!(result.is_converged());

        // Test robust config
        let solver = LMSolver::new(LMConfig::robust());
        let result = solver.solve(&problem, &[1.5]);
        assert!(result.is_converged());

        // Test precise config
        let solver = LMSolver::new(LMConfig::precise());
        let result = solver.solve(&problem, &[1.5]);
        assert!(result.is_converged());
    }

    #[test]
    fn test_different_starting_points() {
        let problem = SqrtTwo;
        let solver = LMSolver::new(LMConfig::robust());

        // Test with various starting points
        for factor in &[0.1, 1.0, 10.0, 100.0] {
            let result = solver.solve_from_initial(&problem, *factor);
            assert!(
                result.is_converged() || result.is_completed(),
                "Should handle factor {}: {:?}",
                factor,
                result
            );
        }
    }

    // Test for handling non-finite initial values
    #[test]
    fn test_non_finite_initial() {
        let problem = SqrtTwo;
        let solver = LMSolver::default_solver();

        // Test with NaN
        let result = solver.solve(&problem, &[f64::NAN]);
        assert!(!result.is_converged());

        // Test with infinity
        let result = solver.solve(&problem, &[f64::INFINITY]);
        assert!(!result.is_converged());
    }
}
