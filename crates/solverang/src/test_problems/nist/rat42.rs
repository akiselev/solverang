//! Rat42 - NIST StRD Nonlinear Regression Problem (Higher Difficulty)
//!
//! Pasture yield data.
//!
//! Model: y = b1 / (1 + exp(b2 - b3*x))
//!
//! Parameters: 3
//! Observations: 9
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/rat42.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Data for Rat42 problem: (x, y) pairs
const DATA: [(f64, f64); 9] = [
    (9.0, 8.930),
    (14.0, 10.800),
    (21.0, 18.590),
    (28.0, 22.330),
    (42.0, 39.350),
    (57.0, 56.110),
    (63.0, 61.730),
    (70.0, 64.620),
    (79.0, 67.080),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 3] = [
    7.2462237576E+01, // b1
    2.6180768402E+00, // b2
    6.7359200066E-02, // b3
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 3] = [
    1.7340283401E+00, // b1
    8.8295217536E-02, // b2
    3.4465663377E-03, // b3
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 8.0565229338E+00;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 3] = [100.0, 1.0, 0.1];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 3] = [75.0, 2.5, 0.07];

/// Rat42 problem: pasture yield
#[derive(Clone, Debug, Default)]
pub struct Rat42;

impl Problem for Rat42 {
    fn name(&self) -> &str {
        "Rat42"
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
                let model = b[0] / (1.0 + (b[1] - b[2] * x).exp());
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 3);

        let mut entries = Vec::with_capacity(DATA.len() * 3);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let exp_term = (b[1] - b[2] * x).exp();
            let denom = 1.0 + exp_term;
            let denom_sq = denom * denom;

            // d/db1 = -1 / (1 + exp(b2 - b3*x))
            entries.push((i, 0, -1.0 / denom));

            // d/db2 = b1 * exp(b2 - b3*x) / (1 + exp(b2 - b3*x))^2
            entries.push((i, 1, b[0] * exp_term / denom_sq));

            // d/db3 = -b1 * x * exp(b2 - b3*x) / (1 + exp(b2 - b3*x))^2
            entries.push((i, 2, -b[0] * x * exp_term / denom_sq));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Rat42 {
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
    fn test_rat42_dimensions() {
        let problem = Rat42;
        assert_eq!(problem.residual_count(), 9);
        assert_eq!(problem.variable_count(), 3);
    }

    #[test]
    fn test_rat42_at_certified() {
        let problem = Rat42;
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
