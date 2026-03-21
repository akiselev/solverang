//! Roszman1 - NIST StRD Nonlinear Regression Problem (Average Difficulty)
//!
//! Quantum defect data.
//!
//! Model: y = b1 - b2*x - arctan(b3/(x-b4)) / pi
//!
//! Parameters: 4
//! Observations: 25
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/roszman1.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;
use std::f64::consts::PI;

/// Data for Roszman1 problem: (x, y) pairs
const DATA: [(f64, f64); 25] = [
    (-4868.68, 0.252429),
    (-4868.09, 0.252141),
    (-4867.41, 0.251809),
    (-3375.19, 0.297989),
    (-3373.14, 0.296257),
    (-3372.03, 0.295319),
    (-2473.74, 0.339603),
    (-2472.35, 0.337731),
    (-2469.45, 0.333820),
    (-1894.65, 0.389510),
    (-1893.40, 0.386998),
    (-1497.24, 0.438864),
    (-1495.85, 0.434887),
    (-1493.41, 0.427893),
    (-1208.68, 0.471568),
    (-1206.18, 0.461699),
    (-1206.04, 0.461144),
    (-997.92, 0.513532),
    (-996.61, 0.506641),
    (-996.31, 0.505062),
    (-834.94, 0.535648),
    (-834.66, 0.533726),
    (-710.03, 0.568064),
    (-530.16, 0.612886),
    (-464.17, 0.624169),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 4] = [
    2.0196866396E-01,  // b1
    -6.1953516256E-06, // b2
    1.2044556708E+03,  // b3
    -1.8134269537E+02, // b4
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 4] = [
    1.9172666023E-02, // b1
    3.2160514428E-06, // b2
    7.4050293788E+01, // b3
    2.9974905535E+01, // b4
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 4.9484847331E-04;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 4] = [0.1, -0.00001, 1000.0, -100.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 4] = [0.2, -0.000005, 1200.0, -150.0];

/// Roszman1 problem: quantum defect
#[derive(Clone, Debug, Default)]
pub struct Roszman1;

impl Problem for Roszman1 {
    fn name(&self) -> &str {
        "Roszman1"
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
                let model = b[0] - b[1] * x - (b[2] / (x - b[3])).atan() / PI;
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 4);

        let mut entries = Vec::with_capacity(DATA.len() * 4);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let diff = x - b[3];
            let ratio = b[2] / diff;
            let denom = 1.0 + ratio * ratio;

            // d(residual)/db1 = -1
            entries.push((i, 0, -1.0));

            // d(residual)/db2 = x
            entries.push((i, 1, x));

            // d(residual)/db3 = 1 / (pi * diff * (1 + (b3/diff)^2))
            entries.push((i, 2, 1.0 / (PI * diff * denom)));

            // d(residual)/db4 = -b3 / (pi * diff^2 * (1 + (b3/diff)^2))
            entries.push((i, 3, -b[2] / (PI * diff * diff * denom)));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Roszman1 {
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
    fn test_roszman1_dimensions() {
        let problem = Roszman1;
        assert_eq!(problem.residual_count(), 25);
        assert_eq!(problem.variable_count(), 4);
    }

    #[test]
    fn test_roszman1_at_certified() {
        let problem = Roszman1;
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
