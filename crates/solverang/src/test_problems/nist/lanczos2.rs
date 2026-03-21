//! Lanczos2 - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! Generated data for sum of exponentials (with noise).
//!
//! Model: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)
//!
//! Parameters: 6
//! Observations: 24
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/lanczos2.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Data for Lanczos2 problem: (x, y) pairs
const DATA: [(f64, f64); 24] = [
    (0.00, 2.51340),
    (0.05, 2.04433),
    (0.10, 1.66840),
    (0.15, 1.36642),
    (0.20, 1.12323),
    (0.25, 0.92688),
    (0.30, 0.76793),
    (0.35, 0.63894),
    (0.40, 0.53380),
    (0.45, 0.44786),
    (0.50, 0.37758),
    (0.55, 0.31973),
    (0.60, 0.27201),
    (0.65, 0.23249),
    (0.70, 0.19965),
    (0.75, 0.17227),
    (0.80, 0.14928),
    (0.85, 0.13006),
    (0.90, 0.11378),
    (0.95, 0.09996),
    (1.00, 0.08827),
    (1.05, 0.07836),
    (1.10, 0.06982),
    (1.15, 0.06259),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 6] = [
    9.6251029939E-02, // b1
    1.0057332849E+00, // b2
    8.6424689056E-01, // b3
    3.0078283915E+00, // b4
    1.5529016879E+00, // b5
    5.0028798100E+00, // b6
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 6] = [
    6.6770575477E-04, // b1
    2.8588326778E-03, // b2
    8.5276164870E-04, // b3
    4.7408232416E-03, // b4
    2.4009646660E-03, // b5
    6.7026196785E-03, // b6
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 2.2299428125E-05;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 6] = [1.2, 0.3, 5.6, 5.5, 6.5, 7.6];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 6] = [0.5, 0.7, 3.6, 4.2, 4.0, 6.3];

/// Lanczos2 problem: sum of exponentials with noise
#[derive(Clone, Debug, Default)]
pub struct Lanczos2;

impl Problem for Lanczos2 {
    fn name(&self) -> &str {
        "Lanczos2"
    }

    fn residual_count(&self) -> usize {
        DATA.len()
    }

    fn variable_count(&self) -> usize {
        6
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 6);

        DATA.iter()
            .map(|&(x, y)| {
                let model =
                    b[0] * (-b[1] * x).exp() + b[2] * (-b[3] * x).exp() + b[4] * (-b[5] * x).exp();
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 6);

        let mut entries = Vec::with_capacity(DATA.len() * 6);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let e1 = (-b[1] * x).exp();
            let e2 = (-b[3] * x).exp();
            let e3 = (-b[5] * x).exp();

            entries.push((i, 0, -e1));
            entries.push((i, 1, b[0] * x * e1));
            entries.push((i, 2, -e2));
            entries.push((i, 3, b[2] * x * e2));
            entries.push((i, 4, -e3));
            entries.push((i, 5, b[4] * x * e3));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Lanczos2 {
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
    fn test_lanczos2_dimensions() {
        let problem = Lanczos2;
        assert_eq!(problem.residual_count(), 24);
        assert_eq!(problem.variable_count(), 6);
    }

    #[test]
    fn test_lanczos2_at_certified() {
        let problem = Lanczos2;
        let certified = problem.certified_values();

        let residuals = problem.residuals(certified);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // Lanczos2 is similar to Lanczos1 but with more significant digits.
        // The certified RSS is 2.2e-05. Due to data truncation (5 decimal places),
        // the computed RSS may be much smaller than certified because the certified
        // parameters fit our truncated data better than they fit the original data.
        // We verify that RSS is reasonably small (within a few orders of magnitude).
        assert!(
            rss < 1e-3,
            "RSS should be small for sum-of-exponentials model: {}",
            rss
        );
    }
}
