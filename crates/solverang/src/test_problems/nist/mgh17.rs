//! MGH17 - NIST StRD Nonlinear Regression Problem (Average Difficulty)
//!
//! More, Garbow, Hillstrom problem 17.
//!
//! Model: y = b1 + b2*exp(-x*b4) + b3*exp(-x*b5)
//!
//! Parameters: 5
//! Observations: 33
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/mgh17.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Data for MGH17 problem: (x, y) pairs
const DATA: [(f64, f64); 33] = [
    (0.0, 8.44E-1),
    (10.0, 9.08E-1),
    (20.0, 9.32E-1),
    (30.0, 9.36E-1),
    (40.0, 9.25E-1),
    (50.0, 9.08E-1),
    (60.0, 8.81E-1),
    (70.0, 8.50E-1),
    (80.0, 8.18E-1),
    (90.0, 7.84E-1),
    (100.0, 7.51E-1),
    (110.0, 7.18E-1),
    (120.0, 6.85E-1),
    (130.0, 6.58E-1),
    (140.0, 6.28E-1),
    (150.0, 6.03E-1),
    (160.0, 5.80E-1),
    (170.0, 5.58E-1),
    (180.0, 5.38E-1),
    (190.0, 5.22E-1),
    (200.0, 5.06E-1),
    (210.0, 4.90E-1),
    (220.0, 4.78E-1),
    (230.0, 4.67E-1),
    (240.0, 4.57E-1),
    (250.0, 4.48E-1),
    (260.0, 4.38E-1),
    (270.0, 4.31E-1),
    (280.0, 4.24E-1),
    (290.0, 4.20E-1),
    (300.0, 4.14E-1),
    (310.0, 4.11E-1),
    (320.0, 4.06E-1),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 5] = [
    3.7541005211E-01,   // b1
    1.9358469127E+00,   // b2
    -1.4646871366E+00,  // b3
    1.2867534640E-02,   // b4
    2.2122699662E-02,   // b5
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 5] = [
    2.0723153551E-03,   // b1
    2.2031669222E-01,   // b2
    2.2175707739E-01,   // b3
    4.4861358114E-04,   // b4
    8.9471996575E-04,   // b5
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 5.4648946975E-05;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 5] = [50.0, 150.0, -100.0, 1.0, 2.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 5] = [0.5, 1.5, -1.0, 0.01, 0.02];

/// MGH17 problem
#[derive(Clone, Debug, Default)]
pub struct MGH17;

impl Problem for MGH17 {
    fn name(&self) -> &str {
        "MGH17"
    }

    fn residual_count(&self) -> usize {
        DATA.len()
    }

    fn variable_count(&self) -> usize {
        5
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 5);

        DATA.iter()
            .map(|&(x, y)| {
                let model = b[0] + b[1] * (-x * b[3]).exp() + b[2] * (-x * b[4]).exp();
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 5);

        let mut entries = Vec::with_capacity(DATA.len() * 5);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let e1 = (-x * b[3]).exp();
            let e2 = (-x * b[4]).exp();

            // d/db1 = -1
            entries.push((i, 0, -1.0));

            // d/db2 = -exp(-x*b4)
            entries.push((i, 1, -e1));

            // d/db3 = -exp(-x*b5)
            entries.push((i, 2, -e2));

            // d/db4 = b2*x*exp(-x*b4)
            entries.push((i, 3, b[1] * x * e1));

            // d/db5 = b3*x*exp(-x*b5)
            entries.push((i, 4, b[2] * x * e2));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for MGH17 {
    fn difficulty(&self) -> NISTDifficulty {
        NISTDifficulty::Average
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
    fn test_mgh17_dimensions() {
        let problem = MGH17;
        assert_eq!(problem.residual_count(), 33);
        assert_eq!(problem.variable_count(), 5);
    }

    #[test]
    fn test_mgh17_at_certified() {
        let problem = MGH17;
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
