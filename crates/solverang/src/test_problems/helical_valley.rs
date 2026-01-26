//! Helical Valley function (MGH Problem 5, HYBRJ Problem 5).
//!
//! A 3-variable test problem featuring arctangent and square root functions,
//! which tests solver behavior with transcendental nonlinearities.
//!
//! # Mathematical Definition
//!
//! Let theta(x_1, x_2) = (1/2pi) * atan(x_2/x_1) when x_1 > 0
//!                     = (1/2pi) * atan(x_2/x_1) + 0.5 when x_1 < 0
//!                     = 0.25 when x_1 = 0 and x_2 >= 0
//!                     = -0.25 when x_1 = 0 and x_2 < 0
//!
//! Residuals (m=3, n=3):
//! - F_1(x) = 10(x_3 - 10*theta)
//! - F_2(x) = 10(sqrt(x_1^2 + x_2^2) - 1)
//! - F_3(x) = x_3
//!
//! Minimum: x* = (1, 0, 0) with F(x*) = 0
//!
//! Starting point: x_0 = (-1, 0, 0)

use crate::Problem;
use std::f64::consts::PI;

/// Helical Valley function problem.
#[derive(Clone, Debug, Default)]
pub struct HelicalValley;

impl HelicalValley {
    /// Compute theta(x_1, x_2) as defined in the problem.
    fn theta(x1: f64, x2: f64) -> f64 {
        let tpi = 2.0 * PI;
        if x1 > 0.0 {
            (x2 / x1).atan() / tpi
        } else if x1 < 0.0 {
            (x2 / x1).atan() / tpi + 0.5
        } else if x2 >= 0.0 {
            // Matches Fortran SIGN(1.0, x2) which returns +1 for x2 >= 0
            0.25
        } else {
            -0.25
        }
    }
}

impl Problem for HelicalValley {
    fn name(&self) -> &str {
        "Helical Valley"
    }

    fn residual_count(&self) -> usize {
        3
    }

    fn variable_count(&self) -> usize {
        3
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 3);

        let theta = Self::theta(x[0], x[1]);
        let r = (x[0].powi(2) + x[1].powi(2)).sqrt();

        vec![
            10.0 * (x[2] - 10.0 * theta), // ten*(x(3) - ten*tmp1)
            10.0 * (r - 1.0),             // ten*(tmp2 - one)
            x[2],                         // x(3)
        ]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 3);

        let tpi = 2.0 * PI;
        let temp = x[0].powi(2) + x[1].powi(2);
        let tmp1 = tpi * temp;
        let tmp2 = temp.sqrt();

        // Handle division by zero
        let (j00, j01, j10, j11) = if tmp1.abs() > 1e-30 && tmp2.abs() > 1e-30 {
            (
                100.0 * x[1] / tmp1,  // dF1/dx1 = 100*x2/tmp1
                -100.0 * x[0] / tmp1, // dF1/dx2 = -100*x1/tmp1
                10.0 * x[0] / tmp2,   // dF2/dx1 = 10*x1/tmp2
                10.0 * x[1] / tmp2,   // dF2/dx2 = 10*x2/tmp2
            )
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        vec![
            (0, 0, j00),
            (0, 1, j01),
            (0, 2, 10.0), // dF1/dx3 = 10
            (1, 0, j10),
            (1, 1, j11),
            (1, 2, 0.0), // dF2/dx3 = 0
            (2, 0, 0.0), // dF3/dx1 = 0
            (2, 1, 0.0), // dF3/dx2 = 0
            (2, 2, 1.0), // dF3/dx3 = 1
        ]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![-factor, 0.0, 0.0]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0, 0.0, 0.0])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helical_valley_at_solution() {
        let problem = HelicalValley;
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Residuals at solution should be zero: {:?}",
            residuals
        );
    }

    #[test]
    fn test_helical_valley_theta() {
        // theta(1, 0) = atan(0/1)/(2pi) = 0
        assert!((HelicalValley::theta(1.0, 0.0) - 0.0).abs() < 1e-10);

        // theta(-1, 0) = atan(0/-1)/(2pi) + 0.5 = 0.5
        assert!((HelicalValley::theta(-1.0, 0.0) - 0.5).abs() < 1e-10);

        // theta(0, 1) = 0.25
        assert!((HelicalValley::theta(0.0, 1.0) - 0.25).abs() < 1e-10);

        // theta(0, -1) = -0.25
        assert!((HelicalValley::theta(0.0, -1.0) - (-0.25)).abs() < 1e-10);

        // theta(0, 0) = 0.25 (matches Fortran SIGN behavior)
        assert!((HelicalValley::theta(0.0, 0.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_helical_valley_residuals_at_initial() {
        let problem = HelicalValley;
        let x0 = problem.initial_point(1.0);

        let residuals = problem.residuals(&x0);
        // At x = (-1, 0, 0):
        // theta = 0.5, r = 1
        // F1 = 10*(0 - 10*0.5) = -50
        // F2 = 10*(1 - 1) = 0
        // F3 = 0
        assert!((residuals[0] - (-50.0)).abs() < 1e-10);
        assert!((residuals[1] - 0.0).abs() < 1e-10);
        assert!((residuals[2] - 0.0).abs() < 1e-10);
    }
}
