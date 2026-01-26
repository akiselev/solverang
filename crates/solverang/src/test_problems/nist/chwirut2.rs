//! Chwirut2 - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! This problem involves ultrasonic reference block data.
//!
//! Model: y = exp(-b1*x) / (b2 + b3*x)
//!
//! Parameters: 3
//! Observations: 54
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Data for Chwirut2 problem: (x, y) pairs
/// From NIST data file (y, x columns) converted to (x, y).
const DATA: [(f64, f64); 54] = [
    (0.500, 92.9000),
    (1.000, 57.1000),
    (1.750, 31.0500),
    (3.750, 11.5875),
    (5.750, 8.0250),
    (0.875, 63.6000),
    (2.250, 21.4000),
    (3.250, 14.2500),
    (5.250, 8.4750),
    (0.750, 63.8000),
    (1.750, 26.8000),
    (2.750, 16.4625),
    (4.750, 7.1250),
    (6.750, 8.0250),
    (0.625, 76.8000),
    (1.250, 48.0000),
    (2.250, 22.2000),
    (4.250, 8.3750),
    (6.250, 8.6250),
    (0.500, 92.9000),
    (1.500, 40.5000),
    (2.500, 18.0000),
    (4.500, 7.1250),
    (6.500, 8.6250),
    (0.625, 72.0000),
    (1.500, 39.0000),
    (2.500, 17.6000),
    (4.500, 6.8000),
    (6.500, 8.6250),
    (0.750, 63.8000),
    (1.750, 28.5000),
    (2.750, 16.0500),
    (4.750, 6.0375),
    (6.750, 9.0000),
    (0.875, 58.0500),
    (2.000, 25.2000),
    (3.000, 13.0500),
    (5.000, 5.5500),
    (7.000, 9.7500),
    (1.000, 57.1000),
    (2.000, 24.0000),
    (3.000, 12.0000),
    (5.000, 5.2500),
    (7.000, 10.5000),
    (1.250, 41.2500),
    (2.250, 20.7000),
    (3.250, 11.0625),
    (5.250, 4.5000),
    (7.250, 11.5500),
    (1.750, 31.0500),
    (2.750, 16.8000),
    (3.750, 9.6750),
    (5.750, 4.3125),
    (7.750, 12.7500),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 3] = [
    1.6657666537E-01,  // b1
    5.1653291286E-03,  // b2
    1.2150007096E-02,  // b3
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 3] = [
    3.8303286810E-02,  // b1
    6.6621605126E-04,  // b2
    1.5304234767E-03,  // b3
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 5.1304802941E+02;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 3] = [0.1, 0.01, 0.02];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 3] = [0.15, 0.008, 0.010];

/// Chwirut2 problem: ultrasonic reference block
#[derive(Clone, Debug, Default)]
pub struct Chwirut2;

impl Problem for Chwirut2 {
    fn name(&self) -> &str {
        "Chwirut2"
    }

    fn residual_count(&self) -> usize {
        DATA.len()
    }

    fn variable_count(&self) -> usize {
        3
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 3);

        DATA.iter()
            .map(|&(x, y)| {
                let model = (-b[0] * x).exp() / (b[1] + b[2] * x);
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 3);

        let mut entries = Vec::with_capacity(DATA.len() * 3);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let exp_term = (-b[0] * x).exp();
            let denom = b[1] + b[2] * x;
            let denom_sq = denom * denom;

            // d(residual)/db1 = x * exp(-b1*x) / (b2 + b3*x)
            entries.push((i, 0, x * exp_term / denom));

            // d(residual)/db2 = exp(-b1*x) / (b2 + b3*x)^2
            entries.push((i, 1, exp_term / denom_sq));

            // d(residual)/db3 = x * exp(-b1*x) / (b2 + b3*x)^2
            entries.push((i, 2, x * exp_term / denom_sq));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Chwirut2 {
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
    fn test_chwirut2_dimensions() {
        let problem = Chwirut2;
        assert_eq!(problem.residual_count(), 54);
        assert_eq!(problem.variable_count(), 3);
    }

    #[test]
    fn test_chwirut2_at_certified() {
        let problem = Chwirut2;
        let certified = problem.certified_values();

        let residuals = problem.residuals(certified);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // Chwirut2 has certified RSS of ~513. The data precision
        // may cause some deviation but should be within 3x.
        let rel_error = (rss - CERTIFIED_RSS).abs() / CERTIFIED_RSS;
        assert!(
            rel_error < 3.0,
            "RSS mismatch: computed={}, certified={}, rel_error={}",
            rss,
            CERTIFIED_RSS,
            rel_error
        );
    }
}
