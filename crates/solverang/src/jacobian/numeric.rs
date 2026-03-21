//! Numeric Jacobian computation fallback.
//!
//! Provides a wrapper that computes Jacobians numerically for problems
//! that do not provide analytical Jacobians.

use crate::problem::Problem;

/// A wrapper that provides numeric Jacobian computation for any residual function.
///
/// This is useful when you have a function that computes residuals but no analytical
/// Jacobian is available. The numeric Jacobian is computed using central finite differences.
///
/// # Example
///
/// ```rust
/// use solverang::jacobian::NumericJacobian;
/// use solverang::{Problem, Solver, SolverConfig};
///
/// struct MyProblem;
///
/// impl MyProblem {
///     fn compute_residuals(&self, x: &[f64]) -> Vec<f64> {
///         vec![x[0] * x[0] - 2.0]  // x^2 = 2
///     }
/// }
///
/// let wrapper = NumericJacobian::new(
///     "sqrt(2)",
///     1,  // m = number of residuals
///     1,  // n = number of variables
///     |x| vec![x[0] * x[0] - 2.0],
///     |factor| vec![1.0 * factor],
/// );
///
/// let solver = Solver::new(SolverConfig::default());
/// let result = solver.solve(&wrapper, &[1.5]);
/// assert!(result.is_converged());
/// ```
pub struct NumericJacobian<F, G>
where
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
    G: Fn(f64) -> Vec<f64> + Send + Sync,
{
    name: String,
    residual_count: usize,
    variable_count: usize,
    residual_fn: F,
    initial_point_fn: G,
    epsilon: f64,
}

impl<F, G> NumericJacobian<F, G>
where
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
    G: Fn(f64) -> Vec<f64> + Send + Sync,
{
    /// Create a new numeric Jacobian wrapper.
    ///
    /// # Arguments
    ///
    /// * `name` - Problem name for debugging
    /// * `residual_count` - Number of residual equations (m)
    /// * `variable_count` - Number of variables (n)
    /// * `residual_fn` - Function that computes residuals
    /// * `initial_point_fn` - Function that returns initial point given a scale factor
    pub fn new(
        name: impl Into<String>,
        residual_count: usize,
        variable_count: usize,
        residual_fn: F,
        initial_point_fn: G,
    ) -> Self {
        Self {
            name: name.into(),
            residual_count,
            variable_count,
            residual_fn,
            initial_point_fn,
            epsilon: 1e-7,
        }
    }

    /// Set the finite difference step size.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Get the epsilon value.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl<F, G> Problem for NumericJacobian<F, G>
where
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
    G: Fn(f64) -> Vec<f64> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn residual_count(&self) -> usize {
        self.residual_count
    }

    fn variable_count(&self) -> usize {
        self.variable_count
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        (self.residual_fn)(x)
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.variable_count;
        let m = self.residual_count;

        if n == 0 || m == 0 {
            return vec![];
        }

        let mut entries = Vec::with_capacity(m * n);
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();

        for j in 0..n {
            // Adaptive step size based on magnitude of x[j]
            let h = self.epsilon * (1.0 + x[j].abs());

            x_plus[j] = x[j] + h;
            x_minus[j] = x[j] - h;

            let f_plus = (self.residual_fn)(&x_plus);
            let f_minus = (self.residual_fn)(&x_minus);

            for i in 0..m {
                let f_plus_i = f_plus.get(i).copied().unwrap_or(0.0);
                let f_minus_i = f_minus.get(i).copied().unwrap_or(0.0);
                let derivative = (f_plus_i - f_minus_i) / (2.0 * h);
                entries.push((i, j, derivative));
            }

            x_plus[j] = x[j];
            x_minus[j] = x[j];
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        (self.initial_point_fn)(factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{Solver, SolverConfig};

    #[test]
    fn test_numeric_jacobian_sqrt_two() {
        let problem = NumericJacobian::new(
            "sqrt(2)",
            1,
            1,
            |x| vec![x[0] * x[0] - 2.0],
            |factor| vec![1.0 * factor],
        );

        let solver = Solver::new(SolverConfig::default());
        let result = solver.solve(&problem, &[1.5]);

        assert!(result.is_converged());
        let solution = result.solution().expect("should have solution");
        assert!((solution[0] - std::f64::consts::SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn test_numeric_jacobian_2d() {
        // Solve x^2 + y^2 = 1, x = y
        let problem = NumericJacobian::new(
            "circle-line",
            2,
            2,
            |x| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]],
            |factor| vec![0.5 * factor, 0.5 * factor],
        );

        let solver = Solver::new(SolverConfig::default());
        let result = solver.solve(&problem, &[0.5, 0.5]);

        assert!(result.is_converged());
        let solution = result.solution().expect("should have solution");
        let expected = 1.0 / std::f64::consts::SQRT_2;
        assert!((solution[0] - expected).abs() < 1e-5);
        assert!((solution[1] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_with_epsilon() {
        let problem = NumericJacobian::new("test", 1, 1, |x| vec![x[0]], |factor| vec![factor])
            .with_epsilon(1e-8);

        assert!((problem.epsilon() - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_problem_name() {
        let problem = NumericJacobian::new("my_problem", 1, 1, |x| vec![x[0]], |_| vec![1.0]);

        assert_eq!(problem.name(), "my_problem");
    }
}
