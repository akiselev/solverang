//! Newton-Raphson solver implementation.

use crate::problem::Problem;
use crate::solver::config::SolverConfig;
use crate::solver::result::{SolveError, SolveResult};
use nalgebra::{DMatrix, DVector};

/// Newton-Raphson solver for nonlinear systems.
///
/// This solver finds roots of F(x) = 0 using the Newton-Raphson iteration:
///   x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)
///
/// For overdetermined systems (m > n), it uses the pseudoinverse via SVD.
/// The solver includes optional backtracking line search for robustness.
///
/// # Example
///
/// ```rust
/// use solverang::{Problem, Solver, SolverConfig, SolveResult};
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
/// let solver = Solver::new(SolverConfig::default());
/// let result = solver.solve(&Quadratic, &[1.0]);
/// assert!(result.is_converged());
/// ```
pub struct Solver {
    config: SolverConfig,
}

impl Solver {
    /// Create a new solver with the given configuration.
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Create a solver with default configuration.
    pub fn default_solver() -> Self {
        Self::new(SolverConfig::default())
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Solve a problem starting from the given initial point.
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

        let mut x = DVector::from_column_slice(x0);

        for iteration in 0..self.config.max_iterations {
            // Compute residuals
            let residuals = problem.residuals(x.as_slice());

            // Check for non-finite residuals
            if residuals.iter().any(|r| !r.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteResiduals,
                };
            }

            let r = DVector::from_column_slice(&residuals);
            let norm = r.norm();

            // Check convergence
            if norm < self.config.tolerance {
                return SolveResult::Converged {
                    solution: x.as_slice().to_vec(),
                    iterations: iteration,
                    residual_norm: norm,
                };
            }

            // Compute Jacobian
            let jac_entries = problem.jacobian(x.as_slice());

            // Check for non-finite Jacobian entries
            if jac_entries.iter().any(|(_, _, v)| !v.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteJacobian,
                };
            }

            let mut j = DMatrix::zeros(m, n);
            for (row, col, val) in jac_entries {
                if row < m && col < n {
                    j[(row, col)] = val;
                }
            }

            // Solve J * delta = -r for the Newton step
            let delta = match self.solve_linear(&j, &(-&r)) {
                Some(d) => d,
                None => {
                    return SolveResult::Failed {
                        error: SolveError::SingularJacobian,
                    };
                }
            };

            // Line search (optional)
            let alpha = if self.config.line_search {
                match self.line_search(problem, &x, &delta, norm) {
                    Some(a) => a,
                    None => {
                        // Line search failed, but we can still return the current best
                        return SolveResult::NotConverged {
                            solution: x.as_slice().to_vec(),
                            iterations: iteration,
                            residual_norm: norm,
                        };
                    }
                }
            } else {
                1.0
            };

            // Update solution
            x += alpha * delta;
        }

        // Did not converge within max iterations
        let residuals = problem.residuals(x.as_slice());
        let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        SolveResult::NotConverged {
            solution: x.as_slice().to_vec(),
            iterations: self.config.max_iterations,
            residual_norm: norm,
        }
    }

    /// Solve using the problem's default initial point.
    pub fn solve_from_initial<P: Problem + ?Sized>(&self, problem: &P, factor: f64) -> SolveResult {
        let x0 = problem.initial_point(factor);
        self.solve(problem, &x0)
    }

    /// Solve the linear system J * delta = rhs.
    ///
    /// For square systems, tries LU decomposition first.
    /// For rectangular or singular systems, falls back to SVD-based pseudoinverse.
    fn solve_linear(&self, j: &DMatrix<f64>, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        let n_rows = j.nrows();
        let n_cols = j.ncols();

        if n_rows == n_cols {
            // Square system: try LU decomposition first
            if let Some(solution) = j.clone().lu().solve(rhs) {
                return Some(solution);
            }
        }

        // Rectangular or singular: use SVD-based pseudoinverse
        let svd = j.clone().svd(true, true);
        svd.solve(rhs, self.config.svd_tolerance).ok()
    }

    /// Backtracking line search with Armijo condition.
    ///
    /// Finds the largest alpha in {1, rho, rho^2, ...} such that:
    ///   ||F(x + alpha*d)|| <= ||F(x)|| * (1 - c*alpha)
    fn line_search<P: Problem + ?Sized>(
        &self,
        problem: &P,
        x: &DVector<f64>,
        delta: &DVector<f64>,
        f0: f64,
    ) -> Option<f64> {
        let mut alpha = 1.0;
        let rho = self.config.backtrack_factor;
        let c = self.config.armijo_c;

        for _ in 0..self.config.max_line_search_iterations {
            let x_new = x + alpha * delta;
            let residuals = problem.residuals(x_new.as_slice());

            // Check for non-finite values
            if residuals.iter().any(|r| !r.is_finite()) {
                alpha *= rho;
                if alpha < self.config.min_step_size {
                    return None;
                }
                continue;
            }

            let f_new: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

            // Armijo condition
            if f_new <= f0 * (1.0 - c * alpha) {
                return Some(alpha);
            }

            alpha *= rho;

            if alpha < self.config.min_step_size {
                // Accept the small step anyway - better than failing
                return Some(alpha);
            }
        }

        // Return the smallest step size tried
        Some(alpha)
    }
}

impl Default for Solver {
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

    // Simple 2D problem: x^2 + y^2 = 1, x = y (solution: (1/sqrt(2), 1/sqrt(2)))
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

    #[test]
    fn test_sqrt_two() {
        let problem = SqrtTwo;
        let solver = Solver::default_solver();
        let result = solver.solve(&problem, &[1.5]);

        assert!(result.is_converged());
        let solution = result.solution().expect("should have solution");
        assert!((solution[0] - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_circle_line() {
        let problem = CircleLineIntersection;
        let solver = Solver::default_solver();
        let result = solver.solve(&problem, &[0.5, 0.5]);

        assert!(result.is_converged());
        let solution = result.solution().expect("should have solution");
        let expected = 1.0 / std::f64::consts::SQRT_2;
        assert!((solution[0] - expected).abs() < 1e-6);
        assert!((solution[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let problem = SqrtTwo;
        let solver = Solver::default_solver();
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
        let solver = Solver::default_solver();
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
        let solver = Solver::default_solver();
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
        let solver = Solver::default_solver();
        let result = solver.solve(&problem, &[1.0]);

        assert_eq!(result.error(), Some(&SolveError::NoEquations));
    }

    #[test]
    fn test_fast_config() {
        let problem = SqrtTwo;
        let solver = Solver::new(SolverConfig::fast());
        let result = solver.solve(&problem, &[1.5]);

        assert!(result.is_converged());
    }

    #[test]
    fn test_robust_config() {
        let problem = SqrtTwo;
        let solver = Solver::new(SolverConfig::robust());
        let result = solver.solve(&problem, &[1.5]);

        assert!(result.is_converged());
    }
}
