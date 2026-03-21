//! Thurber - NIST StRD Nonlinear Regression Problem (Higher Difficulty)
//!
//! Semiconductor data fitting.
//!
//! Model: y = (b1 + b2*x + b3*x^2 + b4*x^3) / (1 + b5*x + b6*x^2 + b7*x^3)
//!
//! Parameters: 7
//! Observations: 37
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Data for Thurber problem: (x, y) pairs
const DATA: [(f64, f64); 37] = [
    (-3.067, 80.574),
    (-2.981, 84.248),
    (-2.921, 87.264),
    (-2.912, 87.195),
    (-2.840, 89.076),
    (-2.797, 89.608),
    (-2.702, 89.868),
    (-2.699, 90.101),
    (-2.633, 92.405),
    (-2.481, 95.854),
    (-2.363, 100.696),
    (-2.322, 101.060),
    (-1.501, 401.672),
    (-1.460, 390.724),
    (-1.274, 567.534),
    (-1.212, 635.316),
    (-1.100, 733.054),
    (-1.046, 759.087),
    (-0.915, 894.206),
    (-0.714, 990.785),
    (-0.566, 1090.109),
    (-0.545, 1080.914),
    (-0.400, 1122.643),
    (-0.309, 1178.351),
    (-0.109, 1260.531),
    (-0.103, 1273.514),
    (0.010, 1288.339),
    (0.119, 1327.543),
    (0.377, 1353.863),
    (0.790, 1414.509),
    (0.963, 1425.208),
    (1.006, 1421.384),
    (1.115, 1442.962),
    (1.572, 1464.350),
    (1.841, 1468.705),
    (2.047, 1447.894),
    (2.200, 1457.628),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 7] = [
    1.2881396800E+03, // b1
    1.4910792535E+03, // b2
    5.8323836877E+02, // b3
    7.5416644291E+01, // b4
    9.6629502864E-01, // b5
    3.9797285797E-01, // b6
    4.9727297349E-02, // b7
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 7] = [
    4.6647963344E+01, // b1
    3.9571156086E+01, // b2
    2.8698696102E+01, // b3
    5.5049381320E+00, // b4
    3.1801927260E-02, // b5
    1.4560302752E-02, // b6
    6.5842344623E-03, // b7
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 5.6427082397E+03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 7] = [1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 7] = [1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05];

/// Thurber problem: semiconductor data
#[derive(Clone, Debug, Default)]
pub struct Thurber;

impl Problem for Thurber {
    fn name(&self) -> &str {
        "Thurber"
    }

    fn residual_count(&self) -> usize {
        DATA.len()
    }

    fn variable_count(&self) -> usize {
        7
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 7);

        DATA.iter()
            .map(|&(x, y)| {
                let num = b[0] + b[1] * x + b[2] * x * x + b[3] * x * x * x;
                let denom = 1.0 + b[4] * x + b[5] * x * x + b[6] * x * x * x;
                let model = num / denom;
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 7);

        let mut entries = Vec::with_capacity(DATA.len() * 7);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let x2 = x * x;
            let x3 = x2 * x;
            let num = b[0] + b[1] * x + b[2] * x2 + b[3] * x3;
            let denom = 1.0 + b[4] * x + b[5] * x2 + b[6] * x3;
            let denom_sq = denom * denom;

            // d/db1 = -1/denom
            entries.push((i, 0, -1.0 / denom));
            // d/db2 = -x/denom
            entries.push((i, 1, -x / denom));
            // d/db3 = -x^2/denom
            entries.push((i, 2, -x2 / denom));
            // d/db4 = -x^3/denom
            entries.push((i, 3, -x3 / denom));
            // d/db5 = num*x/denom^2
            entries.push((i, 4, num * x / denom_sq));
            // d/db6 = num*x^2/denom^2
            entries.push((i, 5, num * x2 / denom_sq));
            // d/db7 = num*x^3/denom^2
            entries.push((i, 6, num * x3 / denom_sq));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Thurber {
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
    fn test_thurber_dimensions() {
        let problem = Thurber;
        assert_eq!(problem.residual_count(), 37);
        assert_eq!(problem.variable_count(), 7);
    }

    #[test]
    fn test_thurber_at_certified() {
        let problem = Thurber;
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
