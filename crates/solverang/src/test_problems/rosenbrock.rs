//! Rosenbrock function (MGH Problem 4, HYBRJ Problem 1).
//!
//! A classic 2-variable test problem that is notoriously difficult for
//! gradient-based methods due to its curved valley.
//!
//! # Mathematical Definition
//!
//! Residuals (m=2, n=2):
//! - F_1(x) = 10(x_2 - x_1^2)
//! - F_2(x) = 1 - x_1
//!
//! Minimum: x* = (1, 1) with F(x*) = 0
//!
//! Starting point: x_0 = (-1.2, 1)

use crate::Problem;

/// Rosenbrock function problem.
#[derive(Clone, Debug, Default)]
pub struct Rosenbrock;

impl Problem for Rosenbrock {
    fn name(&self) -> &str {
        "Rosenbrock"
    }

    fn residual_count(&self) -> usize {
        2
    }

    fn variable_count(&self) -> usize {
        2
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 2);
        vec![
            10.0 * (x[1] - x[0].powi(2)), // ten*(x(2) - x(1)**2)
            1.0 - x[0],                   // one - x(1)
        ]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 2);
        vec![
            (0, 0, -20.0 * x[0]), // dF1/dx1 = -20*x1
            (0, 1, 10.0),         // dF1/dx2 = 10
            (1, 0, -1.0),         // dF2/dx1 = -1
            (1, 1, 0.0),          // dF2/dx2 = 0
        ]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![-1.2 * factor, factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0, 1.0])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rosenbrock_at_solution() {
        let problem = Rosenbrock;
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Residuals at solution should be zero: {:?}",
            residuals
        );
    }

    #[test]
    fn test_rosenbrock_initial_point() {
        let problem = Rosenbrock;
        let x0 = problem.initial_point(1.0);

        assert_eq!(x0.len(), 2);
        assert!((x0[0] - (-1.2)).abs() < 1e-10);
        assert!((x0[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rosenbrock_residuals() {
        let problem = Rosenbrock;
        let x = vec![0.0, 0.0];

        let residuals = problem.residuals(&x);
        assert!((residuals[0] - 0.0).abs() < 1e-10); // 10*(0 - 0) = 0
        assert!((residuals[1] - 1.0).abs() < 1e-10); // 1 - 0 = 1
    }

    #[test]
    fn test_rosenbrock_jacobian() {
        let problem = Rosenbrock;
        let x = vec![1.0, 1.0];

        let jac = problem.jacobian(&x);

        // Convert to dense for easier checking
        let mut dense = vec![vec![0.0; 2]; 2];
        for (row, col, val) in jac {
            dense[row][col] = val;
        }

        assert!((dense[0][0] - (-20.0)).abs() < 1e-10); // -20*1 = -20
        assert!((dense[0][1] - 10.0).abs() < 1e-10);
        assert!((dense[1][0] - (-1.0)).abs() < 1e-10);
        assert!((dense[1][1] - 0.0).abs() < 1e-10);
    }
}
