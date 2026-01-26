//! Misra1b - NIST StRD Nonlinear Regression Problem (Average Difficulty)
//!
//! Organic chemical reaction data.
//!
//! Model: y = b1 * (1 - (1 + b2*x/2)^(-2))
//!
//! Parameters: 2
//! Observations: 14
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/misra1b.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Data for Misra1b problem: (x, y) pairs
const DATA: [(f64, f64); 14] = [
    (77.6, 10.07),
    (114.9, 14.73),
    (141.1, 17.94),
    (190.8, 23.93),
    (239.9, 29.61),
    (289.0, 35.18),
    (332.8, 40.02),
    (378.4, 44.82),
    (434.8, 50.76),
    (477.3, 55.05),
    (536.8, 61.01),
    (593.1, 66.40),
    (689.1, 75.47),
    (760.0, 81.78),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 2] = [
    3.3799746163E+02,  // b1
    3.9039091287E-04,  // b2
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 2] = [
    3.1643950207E+00,  // b1
    4.2547321711E-06,  // b2
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 7.5464681533E-02;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 2] = [500.0, 0.0001];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 2] = [300.0, 0.0002];

/// Misra1b problem
#[derive(Clone, Debug, Default)]
pub struct Misra1b;

impl Problem for Misra1b {
    fn name(&self) -> &str {
        "Misra1b"
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
                let model = b[0] * (1.0 - (1.0 + b[1] * x / 2.0).powf(-2.0));
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 2);

        let mut entries = Vec::with_capacity(DATA.len() * 2);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let inner = 1.0 + b[1] * x / 2.0;

            // d(residual)/db1 = -(1 - inner^(-2))
            entries.push((i, 0, -(1.0 - inner.powf(-2.0))));

            // d(residual)/db2 = -b1 * inner^(-3) * x
            entries.push((i, 1, -b[0] * inner.powf(-3.0) * x));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Misra1b {
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
    fn test_misra1b_dimensions() {
        let problem = Misra1b;
        assert_eq!(problem.residual_count(), 14);
        assert_eq!(problem.variable_count(), 2);
    }

    #[test]
    fn test_misra1b_at_certified() {
        let problem = Misra1b;
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
