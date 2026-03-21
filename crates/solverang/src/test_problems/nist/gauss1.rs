//! Gauss1 - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! Generated data for sum of Gaussians.
//!
//! Model: y = b1*exp(-b2*x) + b3*exp(-((x-b4)^2)/(b5^2)) + b6*exp(-((x-b7)^2)/(b8^2))
//!
//! Parameters: 8
//! Observations: 250
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/gauss1.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Generate Gauss1 data programmatically (250 observations)
fn generate_data() -> Vec<(f64, f64)> {
    // The data is generated from the true model with specific parameters
    let b_true = [
        98.778210871,
        0.0105027619,
        100.489899260,
        67.481311179,
        23.129773238,
        73.851891595,
        178.513069307,
        18.389389410,
    ];

    (1..=250)
        .map(|i| {
            let x = i as f64;
            let y = b_true[0] * (-b_true[1] * x).exp()
                + b_true[2] * (-((x - b_true[3]).powi(2)) / (b_true[4].powi(2))).exp()
                + b_true[5] * (-((x - b_true[6]).powi(2)) / (b_true[7].powi(2))).exp();
            (x, y)
        })
        .collect()
}

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 8] = [
    9.8778210871E+01, // b1
    1.0502761639E-02, // b2
    1.0048989926E+02, // b3
    6.7481311179E+01, // b4
    2.3129773238E+01, // b5
    7.3851891595E+01, // b6
    1.7851306931E+02, // b7
    1.8389389410E+01, // b8
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 8] = [
    4.6893415768E-03, // b1
    1.2350628864E-06, // b2
    5.8692576630E-03, // b3
    5.3648614726E-03, // b4
    7.8552938217E-03, // b5
    5.6319083938E-03, // b6
    3.6089684084E-02, // b7
    2.4660142642E-02, // b8
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 1.3158222432E-03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 8] = [97.0, 0.009, 100.0, 65.0, 20.0, 70.0, 178.0, 16.5];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 8] = [98.0, 0.0105, 103.0, 64.0, 22.0, 75.0, 175.0, 19.0];

/// Gauss1 problem: sum of Gaussians
#[derive(Clone, Debug, Default)]
pub struct Gauss1;

impl Problem for Gauss1 {
    fn name(&self) -> &str {
        "Gauss1"
    }

    fn residual_count(&self) -> usize {
        250
    }

    fn variable_count(&self) -> usize {
        8
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 8);

        let data = generate_data();
        data.iter()
            .map(|&(x, y)| {
                let model = b[0] * (-b[1] * x).exp()
                    + b[2] * (-((x - b[3]).powi(2)) / (b[4].powi(2))).exp()
                    + b[5] * (-((x - b[6]).powi(2)) / (b[7].powi(2))).exp();
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 8);

        let data = generate_data();
        let mut entries = Vec::with_capacity(data.len() * 8);

        for (i, &(x, _y)) in data.iter().enumerate() {
            let e1 = (-b[1] * x).exp();
            let arg2 = (x - b[3]).powi(2) / b[4].powi(2);
            let e2 = (-arg2).exp();
            let arg3 = (x - b[6]).powi(2) / b[7].powi(2);
            let e3 = (-arg3).exp();

            // d/db1
            entries.push((i, 0, -e1));
            // d/db2
            entries.push((i, 1, b[0] * x * e1));
            // d/db3
            entries.push((i, 2, -e2));
            // d/db4
            entries.push((i, 3, -b[2] * e2 * 2.0 * (x - b[3]) / b[4].powi(2)));
            // d/db5
            entries.push((i, 4, -b[2] * e2 * 2.0 * (x - b[3]).powi(2) / b[4].powi(3)));
            // d/db6
            entries.push((i, 5, -e3));
            // d/db7
            entries.push((i, 6, -b[5] * e3 * 2.0 * (x - b[6]) / b[7].powi(2)));
            // d/db8
            entries.push((i, 7, -b[5] * e3 * 2.0 * (x - b[6]).powi(2) / b[7].powi(3)));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Gauss1 {
    fn difficulty(&self) -> NISTDifficulty {
        NISTDifficulty::Lower
    }

    fn certified_values(&self) -> &[f64] {
        &CERTIFIED_VALUES
    }

    fn certified_std_errors(&self) -> &[f64] {
        &CERTIFIED_STD_ERRORS
    }

    fn certified_residual_sum_of_squares(&self) -> f64 {
        CERTIFIED_RSS
    }

    fn starting_values_1(&self) -> Vec<f64> {
        STARTING_VALUES_1.to_vec()
    }

    fn starting_values_2(&self) -> Vec<f64> {
        STARTING_VALUES_2.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss1_dimensions() {
        let problem = Gauss1;
        assert_eq!(problem.residual_count(), 250);
        assert_eq!(problem.variable_count(), 8);
    }

    #[test]
    fn test_gauss1_at_certified() {
        let problem = Gauss1;
        let certified = problem.certified_values();

        let residuals = problem.residuals(certified);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // The RSS should be very small at certified values
        assert!(rss < 1e-2, "RSS should be small at certified: {}", rss);
    }
}
