//! BoxBOD - NIST StRD Nonlinear Regression Problem (Higher Difficulty)
//!
//! Biochemical Oxygen Demand (BOD) data.
//!
//! Model: y = b1 * (1 - exp(-b2*x))
//!
//! Parameters: 2
//! Observations: 6
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/boxbod.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Data for BoxBOD problem: (x, y) pairs
const DATA: [(f64, f64); 6] = [
    (1.0, 109.0),
    (2.0, 149.0),
    (3.0, 149.0),
    (5.0, 191.0),
    (7.0, 213.0),
    (10.0, 224.0),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 2] = [
    2.1380940889E+02,   // b1
    5.4723748542E-01,   // b2
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 2] = [
    1.2354515176E+01,   // b1
    1.0455993237E-01,   // b2
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 1.1680088766E+03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 2] = [1.0, 1.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 2] = [100.0, 0.75];

/// BoxBOD problem: biochemical oxygen demand
#[derive(Clone, Debug, Default)]
pub struct BoxBOD;

impl Problem for BoxBOD {
    fn name(&self) -> &str {
        "BoxBOD"
    }

    fn residual_count(&self) -> usize {
        DATA.len()
    }

    fn variable_count(&self) -> usize {
        2
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 2);

        DATA.iter()
            .map(|&(x, y)| {
                let model = b[0] * (1.0 - (-b[1] * x).exp());
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 2);

        let mut entries = Vec::with_capacity(DATA.len() * 2);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let exp_term = (-b[1] * x).exp();

            // d/db1 = -(1 - exp(-b2*x))
            entries.push((i, 0, -(1.0 - exp_term)));

            // d/db2 = -b1 * x * exp(-b2*x)
            entries.push((i, 1, -b[0] * x * exp_term));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for BoxBOD {
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
    fn test_boxbod_dimensions() {
        let problem = BoxBOD;
        assert_eq!(problem.residual_count(), 6);
        assert_eq!(problem.variable_count(), 2);
    }

    #[test]
    fn test_boxbod_at_certified() {
        let problem = BoxBOD;
        let certified = problem.certified_values();

        let residuals = problem.residuals(certified);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        let rel_error = (rss - CERTIFIED_RSS).abs() / CERTIFIED_RSS;
        assert!(
            rel_error < 1e-5,
            "RSS mismatch: computed={}, certified={}, rel_error={}",
            rss,
            CERTIFIED_RSS,
            rel_error
        );
    }
}
