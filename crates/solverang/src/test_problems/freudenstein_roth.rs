//! Freudenstein and Roth function (MGH Problem 7).
//!
//! A 2-variable least-squares problem with two residual equations.
//!
//! # Mathematical Definition
//!
//! Residuals (m=2, n=2):
//! - F_1(x) = -13 + x_1 + ((5 - x_2)*x_2 - 2)*x_2
//! - F_2(x) = -29 + x_1 + ((x_2 + 1)*x_2 - 14)*x_2
//!
//! This problem has multiple local minima.
//!
//! Global minimum: x* = (5, 4) with F(x*) = 0
//! Local minimum: x* ~ (11.41..., -0.8968...) with F(x*) ~ 48.98
//!
//! Starting point: x_0 = (0.5, -2)

use crate::Problem;

/// Freudenstein and Roth function problem.
#[derive(Clone, Debug, Default)]
pub struct FreudensteinRoth;

impl Problem for FreudensteinRoth {
    fn name(&self) -> &str {
        "Freudenstein-Roth"
    }

    fn residual_count(&self) -> usize {
        2
    }

    fn variable_count(&self) -> usize {
        2
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 2);

        let x2 = x[1];
        vec![
            -13.0 + x[0] + ((5.0 - x2) * x2 - 2.0) * x2,
            -29.0 + x[0] + ((x2 + 1.0) * x2 - 14.0) * x2,
        ]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 2);

        let x2 = x[1];

        // dF_1/dx_2 = d/dx2[(5-x2)*x2 - 2]*x2 = (5-x2)*x2 - 2 + x2*(-1)*x2 + (5-x2)*x2
        //           = x2*(10 - 3*x2) - 2
        let df1_dx2 = x2 * (10.0 - 3.0 * x2) - 2.0;

        // dF_2/dx_2 = d/dx2[(x2+1)*x2 - 14]*x2 = (x2+1)*x2 - 14 + x2*(2*x2+1)
        //           = 3*x2^2 + 2*x2 - 14
        let df2_dx2 = x2 * (3.0 * x2 + 2.0) - 14.0;

        vec![(0, 0, 1.0), (0, 1, df1_dx2), (1, 0, 1.0), (1, 1, df2_dx2)]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![0.5 * factor, -2.0 * factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // Global minimum
        Some(vec![5.0, 4.0])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freudenstein_roth_at_solution() {
        let problem = FreudensteinRoth;
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Residuals at solution should be zero: {:?}",
            residuals
        );
    }
}
