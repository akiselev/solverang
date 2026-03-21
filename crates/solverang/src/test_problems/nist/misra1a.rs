//! Misra1a - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! This problem involves organic chemical reaction data.
//!
//! Model: y = b1 * (1 - exp(-b2 * x))
//!
//! Parameters: 2
//! Observations: 14
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Data for Misra1a problem: (x, y) pairs
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
    2.3894212918E+02, // b1
    5.5015643181E-04, // b2
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 2] = [
    2.7070075241E+00, // b1
    7.2668688436E-06, // b2
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 1.2455138894E-01;

/// Starting values set 1 (farther from solution)
const STARTING_VALUES_1: [f64; 2] = [500.0, 0.0001];

/// Starting values set 2 (closer to solution)
const STARTING_VALUES_2: [f64; 2] = [250.0, 0.0005];

/// Misra1a problem: organic chemical reaction
#[derive(Clone, Debug, Default)]
pub struct Misra1a;

impl Problem for Misra1a {
    fn name(&self) -> &str {
        "Misra1a"
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

            // d(residual)/db1 = -(1 - exp(-b2*x))
            entries.push((i, 0, -(1.0 - exp_term)));

            // d(residual)/db2 = -b1 * x * exp(-b2*x)
            entries.push((i, 1, -b[0] * x * exp_term));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Misra1a {
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
    fn test_misra1a_dimensions() {
        let problem = Misra1a;
        assert_eq!(problem.residual_count(), 14);
        assert_eq!(problem.variable_count(), 2);
    }

    #[test]
    fn test_misra1a_at_certified() {
        let problem = Misra1a;
        let certified = problem.certified_values();

        let residuals = problem.residuals(certified);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        let rel_error = (rss - CERTIFIED_RSS).abs() / CERTIFIED_RSS;
        assert!(
            rel_error < 1e-6,
            "RSS mismatch: computed={}, certified={}, rel_error={}",
            rss,
            CERTIFIED_RSS,
            rel_error
        );
    }

    #[test]
    fn test_misra1a_starting_values() {
        let problem = Misra1a;

        let sv1 = problem.starting_values_1();
        assert_eq!(sv1.len(), 2);

        let sv2 = problem.starting_values_2();
        assert_eq!(sv2.len(), 2);
    }
}
