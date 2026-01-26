//! Adapter for bridging our Problem trait to the levenberg-marquardt crate.
//!
//! The `levenberg-marquardt` crate requires a `LeastSquaresProblem` trait implementation
//! with nalgebra's type-level dimensions. This module provides an adapter that wraps
//! our dynamic `Problem` trait to work with the crate's API.

use crate::problem::Problem;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn};

/// Adapter that wraps our `Problem` trait for use with the `levenberg-marquardt` crate.
///
/// This adapter translates between our dynamic Problem interface and the
/// levenberg-marquardt crate's LeastSquaresProblem trait which uses nalgebra's
/// type-level dimensions.
///
/// The adapter stores current parameters internally and delegates residual/Jacobian
/// computation to the wrapped Problem.
pub struct LMProblemAdapter<'a, P: Problem + ?Sized> {
    /// Reference to the underlying problem
    problem: &'a P,
    /// Current parameter vector
    params: DVector<f64>,
    /// Number of residuals (m)
    residual_count: usize,
    /// Number of variables (n)
    variable_count: usize,
}

impl<'a, P: Problem + ?Sized> LMProblemAdapter<'a, P> {
    /// Create a new adapter wrapping the given problem.
    ///
    /// The initial parameters are set from the provided slice.
    pub fn new(problem: &'a P, initial_params: &[f64]) -> Self {
        let residual_count = problem.residual_count();
        let variable_count = problem.variable_count();

        Self {
            problem,
            params: DVector::from_column_slice(initial_params),
            residual_count,
            variable_count,
        }
    }

    /// Get the final parameters after optimization.
    pub fn final_params(&self) -> Vec<f64> {
        self.params.as_slice().to_vec()
    }
}

impl<P: Problem + ?Sized> LeastSquaresProblem<f64, Dyn, Dyn> for LMProblemAdapter<'_, P> {
    type ParameterStorage = nalgebra::VecStorage<f64, Dyn, nalgebra::U1>;
    type ResidualStorage = nalgebra::VecStorage<f64, Dyn, nalgebra::U1>;
    type JacobianStorage = nalgebra::VecStorage<f64, Dyn, Dyn>;

    fn set_params(&mut self, params: &DVector<f64>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let residuals = self.problem.residuals(self.params.as_slice());

        // Validate dimensions
        if residuals.len() != self.residual_count {
            return None;
        }

        // Check for non-finite values
        if residuals.iter().any(|r| !r.is_finite()) {
            return None;
        }

        Some(DVector::from_vec(residuals))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let entries = self.problem.jacobian(self.params.as_slice());

        // Check for non-finite values
        if entries.iter().any(|(_, _, v)| !v.is_finite()) {
            return None;
        }

        // Build dense Jacobian matrix
        let mut jacobian = DMatrix::zeros(self.residual_count, self.variable_count);

        for (row, col, val) in entries {
            if row < self.residual_count && col < self.variable_count {
                jacobian[(row, col)] = val;
            }
        }

        Some(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test problem: find x such that x^2 - 2 = 0
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
    }

    #[test]
    fn test_adapter_creation() {
        let problem = SqrtTwo;
        let adapter = LMProblemAdapter::new(&problem, &[1.5]);

        let params = adapter.params();
        assert_eq!(params.len(), 1);
        assert!((params[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_adapter_residuals() {
        let problem = SqrtTwo;
        let adapter = LMProblemAdapter::new(&problem, &[1.5]);

        let residuals = adapter.residuals().expect("should compute residuals");
        assert_eq!(residuals.len(), 1);
        // 1.5^2 - 2 = 2.25 - 2 = 0.25
        assert!((residuals[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_adapter_jacobian() {
        let problem = SqrtTwo;
        let adapter = LMProblemAdapter::new(&problem, &[1.5]);

        let jacobian = adapter.jacobian().expect("should compute jacobian");
        assert_eq!(jacobian.nrows(), 1);
        assert_eq!(jacobian.ncols(), 1);
        // d(x^2 - 2)/dx = 2x = 2 * 1.5 = 3.0
        assert!((jacobian[(0, 0)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_adapter_set_params() {
        let problem = SqrtTwo;
        let mut adapter = LMProblemAdapter::new(&problem, &[1.5]);

        let new_params = DVector::from_vec(vec![2.0]);
        adapter.set_params(&new_params);

        let params = adapter.params();
        assert!((params[0] - 2.0).abs() < 1e-10);

        let residuals = adapter.residuals().expect("should compute residuals");
        // 2^2 - 2 = 2
        assert!((residuals[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_adapter_final_params() {
        let problem = SqrtTwo;
        let mut adapter = LMProblemAdapter::new(&problem, &[1.5]);

        let new_params = DVector::from_vec(vec![std::f64::consts::SQRT_2]);
        adapter.set_params(&new_params);

        let final_params = adapter.final_params();
        assert_eq!(final_params.len(), 1);
        assert!((final_params[0] - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    // Test with 2D problem
    struct CircleLine;

    impl Problem for CircleLine {
        fn name(&self) -> &str {
            "circle-line"
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
    fn test_adapter_2d_problem() {
        let problem = CircleLine;
        let adapter = LMProblemAdapter::new(&problem, &[0.5, 0.5]);

        let residuals = adapter.residuals().expect("should compute residuals");
        assert_eq!(residuals.len(), 2);
        // 0.5^2 + 0.5^2 - 1 = 0.5 - 1 = -0.5
        assert!((residuals[0] - (-0.5)).abs() < 1e-10);
        // 0.5 - 0.5 = 0
        assert!(residuals[1].abs() < 1e-10);

        let jacobian = adapter.jacobian().expect("should compute jacobian");
        assert_eq!(jacobian.nrows(), 2);
        assert_eq!(jacobian.ncols(), 2);
        assert!((jacobian[(0, 0)] - 1.0).abs() < 1e-10); // 2 * 0.5
        assert!((jacobian[(0, 1)] - 1.0).abs() < 1e-10); // 2 * 0.5
        assert!((jacobian[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((jacobian[(1, 1)] - (-1.0)).abs() < 1e-10);
    }

    // Test with overdetermined problem (more residuals than variables)
    struct OverdeterminedProblem;

    impl Problem for OverdeterminedProblem {
        fn name(&self) -> &str {
            "overdetermined"
        }
        fn residual_count(&self) -> usize {
            3
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] - 1.0, x[1] - 2.0, x[0] + x[1] - 3.0]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 1.0),
                (0, 1, 0.0),
                (1, 0, 0.0),
                (1, 1, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
            ]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor, factor]
        }
    }

    #[test]
    fn test_adapter_overdetermined() {
        let problem = OverdeterminedProblem;
        let adapter = LMProblemAdapter::new(&problem, &[0.0, 0.0]);

        let residuals = adapter.residuals().expect("should compute residuals");
        assert_eq!(residuals.len(), 3);

        let jacobian = adapter.jacobian().expect("should compute jacobian");
        assert_eq!(jacobian.nrows(), 3);
        assert_eq!(jacobian.ncols(), 2);
    }
}
