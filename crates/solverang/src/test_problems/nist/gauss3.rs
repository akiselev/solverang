//! Gauss3 - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! Generated data for sum of Gaussians.
//!
//! Model: y = b1*exp(-b2*x) + b3*exp(-((x-b4)^2)/(b5^2)) + b6*exp(-((x-b7)^2)/(b8^2))
//!
//! Parameters: 8
//! Observations: 250
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/gauss3.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Generate Gauss3 data programmatically
fn generate_data() -> Vec<(f64, f64)> {
    let b_true = [
        98.940147369,
        0.0105025411,
        100.696320100,
        111.600000000,
        23.300000000,
        73.705031418,
        147.760000000,
        19.668000000,
    ];

    (1..=250)
        .map(|i| {
            let x = i as f64;
            let y = b_true[0] * (-b_true[1] * x).exp()
                + b_true[2] * (-((x - b_true[3]).powi(2)) / (b_true[4].powi(2))).exp()
                + b_true[5] * (-((x - b_true[6]).powi(2)) / (b_true[7].powi(2))).exp();
            (x, y)
        })
        .collect()
}

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 8] = [
    9.8940147369E+01, // b1
    1.0502541170E-02, // b2
    1.0069632010E+02, // b3
    1.1160000000E+02, // b4
    2.3300000000E+01, // b5
    7.3705031418E+01, // b6
    1.4776000000E+02, // b7
    1.9668000000E+01, // b8
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 8] = [
    4.7542522121E-03, // b1
    1.2517584449E-06, // b2
    6.7073630878E-03, // b3
    1.1220055389E-02, // b4
    1.4369156553E-02, // b5
    6.5721767082E-03, // b6
    4.6336049653E-02, // b7
    4.4621949604E-02, // b8
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 1.2476209121E-03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 8] = [94.9, 0.009, 99.0, 109.0, 25.0, 78.0, 150.0, 20.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 8] = [97.0, 0.0105, 100.0, 110.0, 23.0, 73.0, 148.0, 19.0];

/// Gauss3 problem: sum of Gaussians
#[derive(Clone, Debug, Default)]
pub struct Gauss3;

impl Problem for Gauss3 {
    fn name(&self) -> &str {
        "Gauss3"
    }

    fn residual_count(&self) -> usize {
        250
    }

    fn variable_count(&self) -> usize {
        8
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 8);

        let data = generate_data();
        data.iter()
            .map(|&(x, y)| {
                let model = b[0] * (-b[1] * x).exp()
                    + b[2] * (-((x - b[3]).powi(2)) / (b[4].powi(2))).exp()
                    + b[5] * (-((x - b[6]).powi(2)) / (b[7].powi(2))).exp();
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 8);

        let data = generate_data();
        let mut entries = Vec::with_capacity(data.len() * 8);

        for (i, &(x, _y)) in data.iter().enumerate() {
            let e1 = (-b[1] * x).exp();
            let arg2 = (x - b[3]).powi(2) / b[4].powi(2);
            let e2 = (-arg2).exp();
            let arg3 = (x - b[6]).powi(2) / b[7].powi(2);
            let e3 = (-arg3).exp();

            entries.push((i, 0, -e1));
            entries.push((i, 1, b[0] * x * e1));
            entries.push((i, 2, -e2));
            entries.push((i, 3, -b[2] * e2 * 2.0 * (x - b[3]) / b[4].powi(2)));
            entries.push((i, 4, -b[2] * e2 * 2.0 * (x - b[3]).powi(2) / b[4].powi(3)));
            entries.push((i, 5, -e3));
            entries.push((i, 6, -b[5] * e3 * 2.0 * (x - b[6]) / b[7].powi(2)));
            entries.push((i, 7, -b[5] * e3 * 2.0 * (x - b[6]).powi(2) / b[7].powi(3)));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Gauss3 {
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
    fn test_gauss3_dimensions() {
        let problem = Gauss3;
        assert_eq!(problem.residual_count(), 250);
        assert_eq!(problem.variable_count(), 8);
    }
}
