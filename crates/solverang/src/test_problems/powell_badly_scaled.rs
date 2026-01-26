//! Powell badly scaled function (HYBRJ Problem 3).
//!
//! A 2-variable nonlinear equation problem that is badly scaled.
//!
//! # Mathematical Definition
//!
//! Residuals (n=2):
//! - F_1(x) = 10^4 * x_1 * x_2 - 1
//! - F_2(x) = exp(-x_1) + exp(-x_2) - 1.0001
//!
//! Minimum: x* ~ (1.098e-5, 9.106) with F(x*) = 0
//!
//! Starting point: x_0 = (0, 1)

use crate::Problem;

/// Powell badly scaled function problem.
#[derive(Clone, Debug, Default)]
pub struct PowellBadlyScaled;

impl Problem for PowellBadlyScaled {
    fn name(&self) -> &str {
        "Powell Badly Scaled"
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
            1.0e4 * x[0] * x[1] - 1.0,
            (-x[0]).exp() + (-x[1]).exp() - 1.0001,
        ]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 2);

        vec![
            (0, 0, 1.0e4 * x[1]),
            (0, 1, 1.0e4 * x[0]),
            (1, 0, -(-x[0]).exp()),
            (1, 1, -(-x[1]).exp()),
        ]
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![0.0, factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![
            0.1098159327798296E-04,
            0.9106146740038449E+01,
        ])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powell_badly_scaled_at_solution() {
        let problem = PowellBadlyScaled;
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        assert!(
            norm < 1e-8,
            "Residual norm at solution should be near zero: {}",
            norm
        );
    }
}
