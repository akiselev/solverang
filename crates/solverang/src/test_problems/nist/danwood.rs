//! DanWood - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! Radial velocities of the NGC 6341 globular cluster.
//!
//! Model: y = b1 * x^b2
//!
//! Parameters: 2
//! Observations: 6
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/danwood.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Data for DanWood problem: (x, y) pairs
const DATA: [(f64, f64); 6] = [
    (1.309, 2.138),
    (1.471, 3.421),
    (1.490, 3.597),
    (1.565, 4.340),
    (1.611, 4.882),
    (1.680, 5.660),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 2] = [
    7.6886226176E-01, // b1
    3.8604055871E+00, // b2
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 2] = [
    1.8281973860E-01, // b1
    5.1726610913E-01, // b2
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 4.3173084083E-03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 2] = [1.0, 5.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 2] = [0.7, 4.0];

/// DanWood problem: radial velocities
#[derive(Clone, Debug, Default)]
pub struct DanWood;

impl Problem for DanWood {
    fn name(&self) -> &str {
        "DanWood"
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
                let model = b[0] * x.powf(b[1]);
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 2);

        let mut entries = Vec::with_capacity(DATA.len() * 2);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let x_pow = x.powf(b[1]);

            // d/db1 = -x^b2
            entries.push((i, 0, -x_pow));

            // d/db2 = -b1 * x^b2 * ln(x)
            entries.push((i, 1, -b[0] * x_pow * x.ln()));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for DanWood {
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
    fn test_danwood_dimensions() {
        let problem = DanWood;
        assert_eq!(problem.residual_count(), 6);
        assert_eq!(problem.variable_count(), 2);
    }

    #[test]
    fn test_danwood_at_certified() {
        let problem = DanWood;
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
