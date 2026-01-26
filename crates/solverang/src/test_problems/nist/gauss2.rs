//! Gauss2 - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! Generated data for sum of Gaussians (more observations).
//!
//! Model: y = b1*exp(-b2*x) + b3*exp(-((x-b4)^2)/(b5^2)) + b6*exp(-((x-b7)^2)/(b8^2))
//!
//! Parameters: 8
//! Observations: 250
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/gauss2.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Generate Gauss2 data programmatically
fn generate_data() -> Vec<(f64, f64)> {
    let b_true = [
        99.018328406,
        0.0109516650,
        101.880109106,
        107.030130822,
        23.578291561,
        72.045687633,
        153.270112668,
        19.525593953,
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
    9.9018328406E+01,  // b1
    1.0951665003E-02,  // b2
    1.0188010911E+02,  // b3
    1.0703013082E+02,  // b4
    2.3578291561E+01,  // b5
    7.2045687633E+01,  // b6
    1.5327011267E+02,  // b7
    1.9525593953E+01,  // b8
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 8] = [
    5.3087264925E-03,  // b1
    1.1308391063E-06,  // b2
    6.8541987839E-03,  // b3
    7.3018088455E-03,  // b4
    9.2067883201E-03,  // b5
    6.2446224702E-03,  // b6
    3.0399417762E-02,  // b7
    2.0811696245E-02,  // b8
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 1.2475013950E-03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 8] = [96.0, 0.009, 103.0, 106.0, 18.0, 72.0, 151.0, 18.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 8] = [98.0, 0.0105, 104.0, 105.0, 23.0, 73.0, 150.0, 20.0];

/// Gauss2 problem: sum of Gaussians
#[derive(Clone, Debug, Default)]
pub struct Gauss2;

impl Problem for Gauss2 {
    fn name(&self) -> &str {
        "Gauss2"
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

impl NISTProblem for Gauss2 {
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
    fn test_gauss2_dimensions() {
        let problem = Gauss2;
        assert_eq!(problem.residual_count(), 250);
        assert_eq!(problem.variable_count(), 8);
    }
}
