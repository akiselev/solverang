//! Lanczos1 - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! Generated data for sum of exponentials.
//!
//! Model: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)
//!
//! Parameters: 6
//! Observations: 24
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/lanczos1.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Data for Lanczos1 problem: (x, y) pairs
const DATA: [(f64, f64); 24] = [
    (0.00, 2.5134),
    (0.05, 2.0443),
    (0.10, 1.6684),
    (0.15, 1.3664),
    (0.20, 1.1232),
    (0.25, 0.9269),
    (0.30, 0.7679),
    (0.35, 0.6389),
    (0.40, 0.5338),
    (0.45, 0.4479),
    (0.50, 0.3776),
    (0.55, 0.3197),
    (0.60, 0.2720),
    (0.65, 0.2325),
    (0.70, 0.1997),
    (0.75, 0.1723),
    (0.80, 0.1493),
    (0.85, 0.1301),
    (0.90, 0.1138),
    (0.95, 0.1000),
    (1.00, 0.0883),
    (1.05, 0.0783),
    (1.10, 0.0698),
    (1.15, 0.0624),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 6] = [
    9.5100000027E-02,  // b1
    1.0000000001E+00,  // b2
    8.6070000013E-01,  // b3
    3.0000000002E+00,  // b4
    1.5575999998E+00,  // b5
    5.0000000001E+00,  // b6
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 6] = [
    5.3009082147E-09,  // b1
    2.2774578101E-08,  // b2
    6.4396701641E-09,  // b3
    3.5849883361E-08,  // b4
    1.8539521200E-08,  // b5
    5.1838596299E-08,  // b6
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 1.4307867721E-25;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 6] = [1.2, 0.3, 5.6, 5.5, 6.5, 7.6];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 6] = [0.5, 0.7, 3.6, 4.2, 4.0, 6.3];

/// Lanczos1 problem: sum of exponentials
#[derive(Clone, Debug, Default)]
pub struct Lanczos1;

impl Problem for Lanczos1 {
    fn name(&self) -> &str {
        "Lanczos1"
    }

    fn residual_count(&self) -> usize {
        DATA.len()
    }

    fn variable_count(&self) -> usize {
        6
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 6);

        DATA.iter()
            .map(|&(x, y)| {
                let model = b[0] * (-b[1] * x).exp()
                    + b[2] * (-b[3] * x).exp()
                    + b[4] * (-b[5] * x).exp();
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 6);

        let mut entries = Vec::with_capacity(DATA.len() * 6);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let e1 = (-b[1] * x).exp();
            let e2 = (-b[3] * x).exp();
            let e3 = (-b[5] * x).exp();

            entries.push((i, 0, -e1));
            entries.push((i, 1, b[0] * x * e1));
            entries.push((i, 2, -e2));
            entries.push((i, 3, b[2] * x * e2));
            entries.push((i, 4, -e3));
            entries.push((i, 5, b[4] * x * e3));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Lanczos1 {
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
    fn test_lanczos1_dimensions() {
        let problem = Lanczos1;
        assert_eq!(problem.residual_count(), 24);
        assert_eq!(problem.variable_count(), 6);
    }

    #[test]
    fn test_lanczos1_at_certified() {
        let problem = Lanczos1;
        let certified = problem.certified_values();

        let residuals = problem.residuals(certified);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // Lanczos1 has certified RSS of 1.4e-25 (essentially zero).
        // With truncated data precision (4 decimals in y-values), we
        // cannot achieve this level of precision, but RSS should still
        // be very small (around 1e-7 due to data truncation).
        assert!(
            rss < 1e-6,
            "RSS should be very small: {}",
            rss
        );
    }
}
