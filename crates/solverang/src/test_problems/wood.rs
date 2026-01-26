//! Wood function (HYBRJ Problem 4).
//!
//! A 4-variable nonlinear equation problem.
//!
//! # Mathematical Definition
//!
//! Residuals (n=4):
//! - F_1(x) = -200 * x_1 * (x_2 - x_1^2) - (1 - x_1)
//! - F_2(x) = 200 * (x_2 - x_1^2) + 20.2 * (x_2 - 1) + 19.8 * (x_4 - 1)
//! - F_3(x) = -180 * x_3 * (x_4 - x_3^2) - (1 - x_3)
//! - F_4(x) = 180 * (x_4 - x_3^2) + 20.2 * (x_4 - 1) + 19.8 * (x_2 - 1)
//!
//! Minimum: x* = (1, 1, 1, 1) with F(x*) = 0
//!
//! Starting point: x_0 = (-3, -1, -3, -1)

use crate::Problem;

/// Wood function problem.
#[derive(Clone, Debug, Default)]
pub struct Wood;

impl Problem for Wood {
    fn name(&self) -> &str {
        "Wood"
    }

    fn residual_count(&self) -> usize {
        4
    }

    fn variable_count(&self) -> usize {
        4
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 4);

        let temp1 = x[1] - x[0].powi(2);
        let temp2 = x[3] - x[2].powi(2);

        vec![
            -200.0 * x[0] * temp1 - (1.0 - x[0]),
            200.0 * temp1 + 20.2 * (x[1] - 1.0) + 19.8 * (x[3] - 1.0),
            -180.0 * x[2] * temp2 - (1.0 - x[2]),
            180.0 * temp2 + 20.2 * (x[3] - 1.0) + 19.8 * (x[1] - 1.0),
        ]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 4);

        let temp1 = x[1] - x[0].powi(2);
        let temp2 = x[3] - x[2].powi(2);

        vec![
            // Row 0: dF_1/dx
            (0, 0, -200.0 * (temp1 - 2.0 * x[0].powi(2)) + 1.0),
            (0, 1, -200.0 * x[0]),
            (0, 2, 0.0),
            (0, 3, 0.0),
            // Row 1: dF_2/dx
            (1, 0, -400.0 * x[0]),
            (1, 1, 200.0 + 20.2),
            (1, 2, 0.0),
            (1, 3, 19.8),
            // Row 2: dF_3/dx
            (2, 0, 0.0),
            (2, 1, 0.0),
            (2, 2, -180.0 * (temp2 - 2.0 * x[2].powi(2)) + 1.0),
            (2, 3, -180.0 * x[2]),
            // Row 3: dF_4/dx
            (3, 0, 0.0),
            (3, 1, 19.8),
            (3, 2, -360.0 * x[2]),
            (3, 3, 180.0 + 20.2),
        ]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![-3.0 * factor, -factor, -3.0 * factor, -factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0, 1.0, 1.0, 1.0])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wood_at_solution() {
        let problem = Wood;
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Residuals at solution should be zero: {:?}",
            residuals
        );
    }
}
