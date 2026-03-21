//! MGH09 - NIST StRD Nonlinear Regression Problem (Higher Difficulty)
//!
//! More, Garbow, Hillstrom problem 9.
//!
//! Model: y = b1 * (x^2 + x*b2) / (x^2 + x*b3 + b4)
//!
//! Parameters: 4
//! Observations: 11
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/mgh09.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Data for MGH09 problem: (x, y) pairs
const DATA: [(f64, f64); 11] = [
    (4.0, 0.1957),
    (2.0, 0.1947),
    (1.0, 0.1735),
    (0.5, 0.1600),
    (0.25, 0.0844),
    (0.167, 0.0627),
    (0.125, 0.0456),
    (0.1, 0.0342),
    (0.0833, 0.0323),
    (0.0714, 0.0235),
    (0.0625, 0.0246),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 4] = [
    1.9280693458E-01, // b1
    1.9128232873E-01, // b2
    1.2305650693E-01, // b3
    1.3606233068E-01, // b4
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 4] = [
    1.1435312227E-02, // b1
    1.9633220911E-01, // b2
    8.0842031232E-02, // b3
    9.0025542308E-02, // b4
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 3.0750560385E-04;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 4] = [25.0, 39.0, 41.5, 39.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 4] = [0.25, 0.39, 0.415, 0.39];

/// MGH09 problem
#[derive(Clone, Debug, Default)]
pub struct MGH09;

impl Problem for MGH09 {
    fn name(&self) -> &str {
        "MGH09"
    }

    fn residual_count(&self) -> usize {
        DATA.len()
    }

    fn variable_count(&self) -> usize {
        4
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 4);

        DATA.iter()
            .map(|&(x, y)| {
                let x2 = x * x;
                let num = x2 + x * b[1];
                let denom = x2 + x * b[2] + b[3];
                let model = b[0] * num / denom;
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 4);

        let mut entries = Vec::with_capacity(DATA.len() * 4);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let x2 = x * x;
            let num = x2 + x * b[1];
            let denom = x2 + x * b[2] + b[3];
            let denom_sq = denom * denom;

            // d/db1 = -num / denom
            entries.push((i, 0, -num / denom));

            // d/db2 = -b1 * x / denom
            entries.push((i, 1, -b[0] * x / denom));

            // d/db3 = b1 * num * x / denom^2
            entries.push((i, 2, b[0] * num * x / denom_sq));

            // d/db4 = b1 * num / denom^2
            entries.push((i, 3, b[0] * num / denom_sq));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for MGH09 {
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
    fn test_mgh09_dimensions() {
        let problem = MGH09;
        assert_eq!(problem.residual_count(), 11);
        assert_eq!(problem.variable_count(), 4);
    }

    #[test]
    fn test_mgh09_at_certified() {
        let problem = MGH09;
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
