//! MGH10 - NIST StRD Nonlinear Regression Problem (Higher Difficulty)
//!
//! More, Garbow, Hillstrom problem 10 - Meyer function.
//!
//! Model: y = b1 * exp(b2 / (x + b3))
//!
//! Parameters: 3
//! Observations: 16
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/mgh10.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Data for MGH10 problem: (x, y) pairs
const DATA: [(f64, f64); 16] = [
    (50.0, 34780.0),
    (55.0, 28610.0),
    (60.0, 23650.0),
    (65.0, 19630.0),
    (70.0, 16370.0),
    (75.0, 13720.0),
    (80.0, 11540.0),
    (85.0, 9744.0),
    (90.0, 8261.0),
    (95.0, 7030.0),
    (100.0, 6005.0),
    (105.0, 5147.0),
    (110.0, 4427.0),
    (115.0, 3820.0),
    (120.0, 3307.0),
    (125.0, 2872.0),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 3] = [
    5.6096364710E-03,   // b1
    6.1813463463E+03,   // b2
    3.4522363462E+02,   // b3
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 3] = [
    1.5687892471E-04,   // b1
    2.3309021107E+02,   // b2
    7.8486103508E+00,   // b3
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 8.7945855171E+01;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 3] = [2.0, 400000.0, 25000.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 3] = [0.02, 4000.0, 250.0];

/// MGH10 problem (Meyer function)
#[derive(Clone, Debug, Default)]
pub struct MGH10;

impl Problem for MGH10 {
    fn name(&self) -> &str {
        "MGH10"
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
                let model = b[0] * (b[1] / (x + b[2])).exp();
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 3);

        let mut entries = Vec::with_capacity(DATA.len() * 3);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let arg = b[1] / (x + b[2]);
            let exp_term = arg.exp();

            // d/db1 = -exp(b2/(x+b3))
            entries.push((i, 0, -exp_term));

            // d/db2 = -b1 * exp(b2/(x+b3)) / (x+b3)
            entries.push((i, 1, -b[0] * exp_term / (x + b[2])));

            // d/db3 = b1 * b2 * exp(b2/(x+b3)) / (x+b3)^2
            entries.push((i, 2, b[0] * b[1] * exp_term / ((x + b[2]) * (x + b[2]))));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for MGH10 {
    fn difficulty(&self) -> NISTDifficulty {
        NISTDifficulty::Higher
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
    fn test_mgh10_dimensions() {
        let problem = MGH10;
        assert_eq!(problem.residual_count(), 16);
        assert_eq!(problem.variable_count(), 3);
    }

    #[test]
    fn test_mgh10_at_certified() {
        let problem = MGH10;
        let certified = problem.certified_values();

        let residuals = problem.residuals(certified);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        let rel_error = (rss - CERTIFIED_RSS).abs() / CERTIFIED_RSS;
        assert!(
            rel_error < 1e-4,
            "RSS mismatch: computed={}, certified={}, rel_error={}",
            rss,
            CERTIFIED_RSS,
            rel_error
        );
    }
}
