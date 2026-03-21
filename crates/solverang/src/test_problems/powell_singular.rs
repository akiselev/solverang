//! Powell Singular function (MGH Problem 6, HYBRJ Problem 2).
//!
//! A 4-variable test problem with a singular Jacobian at the solution,
//! making it particularly challenging for Newton-type methods.
//!
//! # Mathematical Definition
//!
//! Residuals (m=4, n=4):
//! - F_1(x) = x_1 + 10*x_2
//! - F_2(x) = sqrt(5) * (x_3 - x_4)
//! - F_3(x) = (x_2 - 2*x_3)^2
//! - F_4(x) = sqrt(10) * (x_1 - x_4)^2
//!
//! Minimum: x* = (0, 0, 0, 0) with F(x*) = 0
//!
//! Starting point: x_0 = (3, -1, 0, 1)

use crate::Problem;

/// Powell Singular function problem.
#[derive(Clone, Debug, Default)]
pub struct PowellSingular;

impl Problem for PowellSingular {
    fn name(&self) -> &str {
        "Powell Singular"
    }

    fn residual_count(&self) -> usize {
        4
    }

    fn variable_count(&self) -> usize {
        4
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 4);

        let sqrt5 = 5.0_f64.sqrt();
        let sqrt10 = 10.0_f64.sqrt();

        vec![
            x[0] + 10.0 * x[1],             // x(1) + ten*x(2)
            sqrt5 * (x[2] - x[3]),          // sqrt(five)*(x(3) - x(4))
            (x[1] - 2.0 * x[2]).powi(2),    // (x(2) - two*x(3))**2
            sqrt10 * (x[0] - x[3]).powi(2), // sqrt(ten)*(x(1) - x(4))**2
        ]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 4);

        let sqrt5 = 5.0_f64.sqrt();
        let sqrt10 = 10.0_f64.sqrt();

        let d3 = x[1] - 2.0 * x[2]; // For F_3
        let d4 = x[0] - x[3]; // For F_4

        vec![
            // Row 0: dF_1/dx
            (0, 0, 1.0),
            (0, 1, 10.0),
            (0, 2, 0.0),
            (0, 3, 0.0),
            // Row 1: dF_2/dx
            (1, 0, 0.0),
            (1, 1, 0.0),
            (1, 2, sqrt5),
            (1, 3, -sqrt5),
            // Row 2: dF_3/dx = 2(x_2 - 2x_3) * [0, 1, -2, 0]
            (2, 0, 0.0),
            (2, 1, 2.0 * d3),
            (2, 2, -4.0 * d3),
            (2, 3, 0.0),
            // Row 3: dF_4/dx = 2*sqrt(10)*(x_1 - x_4) * [1, 0, 0, -1]
            (3, 0, 2.0 * sqrt10 * d4),
            (3, 1, 0.0),
            (3, 2, 0.0),
            (3, 3, -2.0 * sqrt10 * d4),
        ]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![3.0 * factor, -factor, 0.0, factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0, 0.0, 0.0, 0.0])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powell_singular_at_solution() {
        let problem = PowellSingular;
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Residuals at solution should be zero: {:?}",
            residuals
        );
    }

    #[test]
    fn test_powell_singular_jacobian_at_solution() {
        let problem = PowellSingular;
        let solution = problem.known_solution().expect("should have solution");

        // At x = 0, the Jacobian should be singular
        let jac = problem.jacobian(&solution);
        let mut dense = vec![vec![0.0; 4]; 4];
        for (row, col, val) in jac {
            dense[row][col] = val;
        }

        // Row 2 and Row 3 should be zero vectors (singular)
        assert!(dense[2].iter().all(|v| v.abs() < 1e-10));
        assert!(dense[3].iter().all(|v| v.abs() < 1e-10));
    }
}
