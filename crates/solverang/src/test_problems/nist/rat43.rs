//! Rat43 - NIST StRD Nonlinear Regression Problem (Higher Difficulty)
//!
//! Pasture yield data (4 parameter model).
//!
//! Model: y = b1 / ((1 + exp(b2 - b3*x))^(1/b4))
//!
//! Parameters: 4
//! Observations: 15
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/rat43.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Data for Rat43 problem: (x, y) pairs
const DATA: [(f64, f64); 15] = [
    (1.0, 16.08),
    (2.0, 33.83),
    (3.0, 65.80),
    (4.0, 97.20),
    (5.0, 191.55),
    (6.0, 326.20),
    (7.0, 386.87),
    (8.0, 520.53),
    (9.0, 590.03),
    (10.0, 651.92),
    (11.0, 724.93),
    (12.0, 699.56),
    (13.0, 689.96),
    (14.0, 637.56),
    (15.0, 717.41),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 4] = [
    6.9964151270E+02, // b1
    5.2771253025E+00, // b2
    7.5962938329E-01, // b3
    1.2792483859E+00, // b4
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 4] = [
    1.6302297817E+01, // b1
    2.0828735829E-01, // b2
    1.9566123451E-02, // b3
    6.8761936385E-02, // b4
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 8.7864049080E+03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 4] = [100.0, 10.0, 1.0, 1.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 4] = [700.0, 5.0, 0.75, 1.3];

/// Rat43 problem: pasture yield (4 parameter)
#[derive(Clone, Debug, Default)]
pub struct Rat43;

impl Problem for Rat43 {
    fn name(&self) -> &str {
        "Rat43"
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
                let inner = 1.0 + (b[1] - b[2] * x).exp();
                let model = b[0] / inner.powf(1.0 / b[3]);
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 4);

        let mut entries = Vec::with_capacity(DATA.len() * 4);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let exp_term = (b[1] - b[2] * x).exp();
            let inner = 1.0 + exp_term;
            let inv_b4 = 1.0 / b[3];
            let power = inner.powf(inv_b4);

            // d/db1 = -1 / inner^(1/b4)
            entries.push((i, 0, -1.0 / power));

            // d/db2 = b1 * exp_term * inner^(-1/b4 - 1) / b4
            entries.push((i, 1, b[0] * exp_term / (b[3] * inner.powf(inv_b4 + 1.0))));

            // d/db3 = -b1 * x * exp_term * inner^(-1/b4 - 1) / b4
            entries.push((
                i,
                2,
                -b[0] * x * exp_term / (b[3] * inner.powf(inv_b4 + 1.0)),
            ));

            // d/db4 = b1 * ln(inner) / (b4^2 * inner^(1/b4))
            let ln_inner = inner.ln();
            entries.push((i, 3, b[0] * ln_inner / (b[3] * b[3] * power)));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Rat43 {
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
    fn test_rat43_dimensions() {
        let problem = Rat43;
        assert_eq!(problem.residual_count(), 15);
        assert_eq!(problem.variable_count(), 4);
    }

    #[test]
    fn test_rat43_at_certified() {
        let problem = Rat43;
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
