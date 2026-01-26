//! Problem trait definitions for nonlinear systems.
//!
//! This module defines the core abstraction for problems that can be solved by
//! the numerical solver. A problem represents finding x such that F(x) = 0,
//! where F is the residual vector function.

/// A nonlinear problem for numerical solvers.
///
/// This trait abstracts the common interface between:
/// - Least-squares problems (m equations, n variables, m >= n)
/// - Nonlinear equation systems (n equations, n variables)
/// - Constraint systems from various domains
///
/// All problems express finding x such that F(x) approaches 0, where F is the residual
/// vector. For least-squares problems, we minimize ||F(x)||^2.
///
/// # Implementation Notes
///
/// Implementors should ensure:
/// - `residuals()` returns a vector of length `residual_count()`
/// - `jacobian()` returns entries only within bounds (row < residual_count, col < variable_count)
/// - `initial_point()` returns a vector of length `variable_count()`
pub trait Problem: Send + Sync {
    /// Problem name for reporting and debugging.
    fn name(&self) -> &str;

    /// Number of residual equations (m).
    ///
    /// For least-squares problems, this may be greater than `variable_count()`.
    /// For nonlinear equation problems, this equals `variable_count()`.
    fn residual_count(&self) -> usize;

    /// Number of variables (n).
    fn variable_count(&self) -> usize;

    /// Evaluate residuals at x: F(x) where we seek F(x) = 0.
    ///
    /// Returns a vector of length `residual_count()`.
    fn residuals(&self, x: &[f64]) -> Vec<f64>;

    /// Compute the Jacobian matrix as sparse triplets: (row, col, value).
    ///
    /// The Jacobian J\[i,j\] = dF\[i\]/dx\[j\].
    /// Row indices are 0..residual_count(), column indices are 0..variable_count().
    ///
    /// This returns only non-zero entries for efficiency with sparse matrices.
    /// For dense problems, all m*n entries may be returned.
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)>;

    /// Compute the dense Jacobian matrix.
    ///
    /// Returns a row-major m x n matrix where m = residual_count(), n = variable_count().
    /// Default implementation builds from sparse triplets.
    fn jacobian_dense(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let m = self.residual_count();
        let n = self.variable_count();
        let mut jac = vec![vec![0.0; n]; m];

        for (row, col, val) in self.jacobian(x) {
            if row < m && col < n {
                jac[row][col] = val;
            }
        }

        jac
    }

    /// Standard starting point, optionally scaled by a factor.
    ///
    /// MINPACK tests use factor values of 1.0, 10.0, and 100.0 to test
    /// solver robustness with different initial conditions.
    fn initial_point(&self, factor: f64) -> Vec<f64>;

    /// Known solution (if available).
    ///
    /// For many test problems, the optimal solution is known analytically.
    /// This is used for verification and testing.
    fn known_solution(&self) -> Option<Vec<f64>> {
        None
    }

    /// Expected final residual norm ||F(x*)||.
    ///
    /// For nonlinear equation problems, this should be essentially zero.
    /// For least-squares problems with non-zero residuals (e.g., data fitting),
    /// this is the expected optimal residual norm.
    fn expected_residual_norm(&self) -> Option<f64> {
        None
    }

    /// Whether this is a least-squares problem (m > n possible) or
    /// a square system (m = n).
    fn is_least_squares(&self) -> bool {
        self.residual_count() >= self.variable_count()
    }

    /// Compute residual norm ||F(x)||_2 = sqrt(sum(F\[i\]^2)).
    fn residual_norm(&self, x: &[f64]) -> f64 {
        let r = self.residuals(x);
        r.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Compute sum of squared residuals ||F(x)||^2 = sum(F\[i\]^2).
    fn sum_of_squares(&self, x: &[f64]) -> f64 {
        let r = self.residuals(x);
        r.iter().map(|v| v * v).sum()
    }
}

/// Extension trait for problems with configurable dimensions.
///
/// Some MINPACK problems can be instantiated with different numbers of
/// variables and/or equations.
pub trait ConfigurableProblem: Problem {
    /// Create a new instance with the specified dimensions.
    ///
    /// Returns None if the dimensions are invalid for this problem type.
    fn with_dimensions(n: usize, m: usize) -> Option<Self>
    where
        Self: Sized;

    /// Minimum allowed number of variables.
    fn min_variables() -> usize
    where
        Self: Sized;

    /// Maximum allowed number of variables (None if unlimited).
    fn max_variables() -> Option<usize>
    where
        Self: Sized,
    {
        None
    }

    /// Whether the number of equations is determined by n.
    fn equations_fixed_by_variables() -> bool
    where
        Self: Sized,
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test problem: f(x) = x - a where we want x = a
    struct LinearProblem {
        target: Vec<f64>,
    }

    impl Problem for LinearProblem {
        fn name(&self) -> &str {
            "Linear"
        }

        fn residual_count(&self) -> usize {
            self.target.len()
        }

        fn variable_count(&self) -> usize {
            self.target.len()
        }

        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            x.iter().zip(&self.target).map(|(xi, ai)| xi - ai).collect()
        }

        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            // Identity matrix
            (0..self.target.len()).map(|i| (i, i, 1.0)).collect()
        }

        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor; self.target.len()]
        }

        fn known_solution(&self) -> Option<Vec<f64>> {
            Some(self.target.clone())
        }

        fn expected_residual_norm(&self) -> Option<f64> {
            Some(0.0)
        }
    }

    #[test]
    fn test_linear_problem() {
        let problem = LinearProblem {
            target: vec![1.0, 2.0, 3.0],
        };

        assert_eq!(problem.name(), "Linear");
        assert_eq!(problem.residual_count(), 3);
        assert_eq!(problem.variable_count(), 3);

        // At solution, residuals should be zero
        let solution = problem.known_solution().expect("should have solution");
        let residuals = problem.residuals(&solution);
        assert!(residuals.iter().all(|r| r.abs() < 1e-10));

        // Jacobian should be identity
        let jac = problem.jacobian(&solution);
        assert_eq!(jac.len(), 3);
        for (i, (row, col, val)) in jac.iter().enumerate() {
            assert_eq!(*row, i);
            assert_eq!(*col, i);
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_residual_norm() {
        let problem = LinearProblem {
            target: vec![0.0, 0.0],
        };

        let x = vec![3.0, 4.0];
        let norm = problem.residual_norm(&x);
        assert!((norm - 5.0).abs() < 1e-10); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_jacobian_dense() {
        let problem = LinearProblem {
            target: vec![1.0, 2.0],
        };

        let x = vec![0.0, 0.0];
        let dense = problem.jacobian_dense(&x);

        assert_eq!(dense.len(), 2);
        assert_eq!(dense[0].len(), 2);

        // Should be identity matrix
        assert!((dense[0][0] - 1.0).abs() < 1e-10);
        assert!((dense[0][1] - 0.0).abs() < 1e-10);
        assert!((dense[1][0] - 0.0).abs() < 1e-10);
        assert!((dense[1][1] - 1.0).abs() < 1e-10);
    }
}
