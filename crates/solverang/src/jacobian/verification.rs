//! Jacobian verification utilities.
//!
//! Provides tools for verifying analytical Jacobians against finite differences.

use crate::problem::Problem;

/// Result of verifying a Jacobian against finite differences.
#[derive(Clone, Debug)]
pub struct JacobianVerification {
    /// Maximum absolute error across all entries.
    pub max_absolute_error: f64,
    /// Mean absolute error across all entries.
    pub mean_absolute_error: f64,
    /// Maximum relative error across all entries (where applicable).
    pub max_relative_error: f64,
    /// Location (row, col) of the maximum error.
    pub max_error_location: (usize, usize),
    /// Whether verification passed (max error < tolerance).
    pub passed: bool,
}

impl JacobianVerification {
    /// Create a verification result indicating success.
    pub fn success() -> Self {
        Self {
            max_absolute_error: 0.0,
            mean_absolute_error: 0.0,
            max_relative_error: 0.0,
            max_error_location: (0, 0),
            passed: true,
        }
    }
}

/// Verify that an analytical Jacobian matches finite differences.
///
/// Uses central differences: J\[i,j\] ~ (F_i(x + h*e_j) - F_i(x - h*e_j)) / (2h)
///
/// # Arguments
///
/// * `problem` - The problem to verify
/// * `x` - Point at which to verify the Jacobian
/// * `epsilon` - Step size for finite differences (typically 1e-7 to 1e-8)
/// * `tolerance` - Maximum allowed absolute error
///
/// # Example
///
/// ```rust
/// use solverang::{Problem, verify_jacobian};
///
/// struct Quadratic;
///
/// impl Problem for Quadratic {
///     fn name(&self) -> &str { "x^2" }
///     fn residual_count(&self) -> usize { 1 }
///     fn variable_count(&self) -> usize { 1 }
///     fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0] * x[0]] }
///     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { vec![(0, 0, 2.0 * x[0])] }
///     fn initial_point(&self, _: f64) -> Vec<f64> { vec![1.0] }
/// }
///
/// let problem = Quadratic;
/// let result = verify_jacobian(&problem, &[2.0], 1e-7, 1e-5);
/// assert!(result.passed);
/// ```
pub fn verify_jacobian<P: Problem + ?Sized>(
    problem: &P,
    x: &[f64],
    epsilon: f64,
    tolerance: f64,
) -> JacobianVerification {
    let n = problem.variable_count();
    let m = problem.residual_count();

    if n == 0 || m == 0 {
        return JacobianVerification::success();
    }

    // Get analytical Jacobian
    let analytical = problem.jacobian_dense(x);

    // Compute finite difference Jacobian
    let numerical = finite_difference_jacobian(problem, x, epsilon);

    // Compare
    let mut max_abs_error = 0.0;
    let mut max_rel_error = 0.0;
    let mut sum_error = 0.0;
    let mut max_location = (0, 0);
    let mut count = 0;

    for i in 0..m {
        for j in 0..n {
            let analytical_val = analytical
                .get(i)
                .and_then(|row| row.get(j))
                .copied()
                .unwrap_or(0.0);
            let numerical_val = numerical
                .get(i)
                .and_then(|row| row.get(j))
                .copied()
                .unwrap_or(0.0);

            let abs_error = (analytical_val - numerical_val).abs();
            sum_error += abs_error;
            count += 1;

            // Relative error (avoid division by zero)
            let rel_error = if analytical_val.abs() > 1e-10 {
                abs_error / analytical_val.abs()
            } else if numerical_val.abs() > 1e-10 {
                abs_error / numerical_val.abs()
            } else {
                0.0
            };

            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_location = (i, j);
            }

            if rel_error > max_rel_error {
                max_rel_error = rel_error;
            }
        }
    }

    let mean_error = if count > 0 {
        sum_error / (count as f64)
    } else {
        0.0
    };

    JacobianVerification {
        max_absolute_error: max_abs_error,
        mean_absolute_error: mean_error,
        max_relative_error: max_rel_error,
        max_error_location: max_location,
        passed: max_abs_error < tolerance,
    }
}

/// Compute Jacobian using central finite differences.
///
/// J\[i,j\] = (F_i(x + h*e_j) - F_i(x - h*e_j)) / (2h)
///
/// # Arguments
///
/// * `problem` - The problem whose Jacobian to compute
/// * `x` - Point at which to compute the Jacobian
/// * `epsilon` - Step size for finite differences
///
/// Returns a row-major m x n matrix.
pub fn finite_difference_jacobian<P: Problem + ?Sized>(
    problem: &P,
    x: &[f64],
    epsilon: f64,
) -> Vec<Vec<f64>> {
    let n = problem.variable_count();
    let m = problem.residual_count();

    if n == 0 || m == 0 {
        return vec![vec![]; m];
    }

    let mut jacobian = vec![vec![0.0; n]; m];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for j in 0..n {
        // Adaptive step size based on the magnitude of x[j]
        let h = epsilon * (1.0 + x[j].abs());

        // Perturb x_j
        x_plus[j] = x[j] + h;
        x_minus[j] = x[j] - h;

        let f_plus = problem.residuals(&x_plus);
        let f_minus = problem.residuals(&x_minus);

        // Central difference
        for (row, jac_row) in jacobian.iter_mut().enumerate().take(m) {
            let f_plus_i = f_plus.get(row).copied().unwrap_or(0.0);
            let f_minus_i = f_minus.get(row).copied().unwrap_or(0.0);
            jac_row[j] = (f_plus_i - f_minus_i) / (2.0 * h);
        }

        // Restore x
        x_plus[j] = x[j];
        x_minus[j] = x[j];
    }

    jacobian
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LinearProblem;

    impl Problem for LinearProblem {
        fn name(&self) -> &str {
            "Linear"
        }
        fn residual_count(&self) -> usize {
            2
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![2.0 * x[0] + 3.0 * x[1], 4.0 * x[0] + 5.0 * x[1]]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 2.0), (0, 1, 3.0), (1, 0, 4.0), (1, 1, 5.0)]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor, factor]
        }
    }

    struct QuadraticProblem;

    impl Problem for QuadraticProblem {
        fn name(&self) -> &str {
            "Quadratic"
        }
        fn residual_count(&self) -> usize {
            2
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1], x[0] * x[1]]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 2.0 * x[0]), (0, 1, 1.0), (1, 0, x[1]), (1, 1, x[0])]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor, factor]
        }
    }

    #[test]
    fn test_linear_jacobian_verification() {
        let problem = LinearProblem;
        let x = vec![1.0, 1.0];

        let result = verify_jacobian(&problem, &x, 1e-7, 1e-6);
        assert!(result.passed, "Linear Jacobian should be exact");
        assert!(result.max_absolute_error < 1e-8);
    }

    #[test]
    fn test_quadratic_jacobian_verification() {
        let problem = QuadraticProblem;
        let x = vec![2.0, 3.0];

        let result = verify_jacobian(&problem, &x, 1e-7, 1e-5);
        assert!(
            result.passed,
            "Quadratic Jacobian verification failed: max error = {}",
            result.max_absolute_error
        );
    }

    #[test]
    fn test_finite_difference_accuracy() {
        let problem = LinearProblem;
        let x = vec![1.0, 1.0];

        let fd = finite_difference_jacobian(&problem, &x, 1e-7);

        // Should be close to [[2, 3], [4, 5]]
        assert!((fd[0][0] - 2.0).abs() < 1e-6);
        assert!((fd[0][1] - 3.0).abs() < 1e-6);
        assert!((fd[1][0] - 4.0).abs() < 1e-6);
        assert!((fd[1][1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_problem() {
        struct Empty;
        impl Problem for Empty {
            fn name(&self) -> &str {
                "Empty"
            }
            fn residual_count(&self) -> usize {
                0
            }
            fn variable_count(&self) -> usize {
                0
            }
            fn residuals(&self, _x: &[f64]) -> Vec<f64> {
                vec![]
            }
            fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
                vec![]
            }
            fn initial_point(&self, _factor: f64) -> Vec<f64> {
                vec![]
            }
        }

        let problem = Empty;
        let result = verify_jacobian(&problem, &[], 1e-7, 1e-6);
        assert!(result.passed);
    }

    #[test]
    fn test_verification_location() {
        // Problem with intentionally wrong Jacobian for entry (1, 0)
        struct WrongJacobian;
        impl Problem for WrongJacobian {
            fn name(&self) -> &str {
                "WrongJacobian"
            }
            fn residual_count(&self) -> usize {
                2
            }
            fn variable_count(&self) -> usize {
                2
            }
            fn residuals(&self, x: &[f64]) -> Vec<f64> {
                vec![x[0], x[0] + x[1]]
            }
            fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
                vec![
                    (0, 0, 1.0), // Correct
                    (0, 1, 0.0), // Correct
                    (1, 0, 0.0), // Wrong! Should be 1.0
                    (1, 1, 1.0), // Correct
                ]
            }
            fn initial_point(&self, _factor: f64) -> Vec<f64> {
                vec![1.0, 1.0]
            }
        }

        let problem = WrongJacobian;
        let result = verify_jacobian(&problem, &[1.0, 1.0], 1e-7, 0.5);

        assert!(!result.passed);
        assert_eq!(result.max_error_location, (1, 0));
        assert!((result.max_absolute_error - 1.0).abs() < 0.01);
    }
}
