//! Misra1c - NIST StRD Nonlinear Regression Problem (Average Difficulty)
//!
//! Organic chemical reaction data.
//!
//! Model: y = b1 * (1 - (1 + 2*b2*x)^(-0.5))
//!
//! Parameters: 2
//! Observations: 14
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/misra1c.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Data for Misra1c problem: (x, y) pairs
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
    6.3642725809E+02,  // b1
    2.0813627256E-04,  // b2
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 2] = [
    4.6638326572E+00,  // b1
    1.7734423042E-06,  // b2
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 4.0966836971E-02;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 2] = [500.0, 0.0001];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 2] = [600.0, 0.0002];

/// Misra1c problem
#[derive(Clone, Debug, Default)]
pub struct Misra1c;

impl Problem for Misra1c {
    fn name(&self) -> &str {
        "Misra1c"
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
                let model = b[0] * (1.0 - (1.0 + 2.0 * b[1] * x).powf(-0.5));
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 2);

        let mut entries = Vec::with_capacity(DATA.len() * 2);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let inner = 1.0 + 2.0 * b[1] * x;

            // d(residual)/db1 = -(1 - inner^(-0.5))
            entries.push((i, 0, -(1.0 - inner.powf(-0.5))));

            // d(residual)/db2 = -b1 * inner^(-1.5) * x
            entries.push((i, 1, -b[0] * inner.powf(-1.5) * x));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Misra1c {
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
    fn test_misra1c_dimensions() {
        let problem = Misra1c;
        assert_eq!(problem.residual_count(), 14);
        assert_eq!(problem.variable_count(), 2);
    }

    #[test]
    fn test_misra1c_at_certified() {
        let problem = Misra1c;
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
