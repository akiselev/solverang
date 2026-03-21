//! Eckerle4 - NIST StRD Nonlinear Regression Problem (Higher Difficulty)
//!
//! Circular interference transmittance data.
//!
//! Model: y = (b1/b2) * exp(-0.5*((x-b3)/b2)^2)
//!
//! Parameters: 3
//! Observations: 35
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml>

use super::{NISTDifficulty, NISTProblem};
use crate::Problem;

/// Data for Eckerle4 problem: (x, y) pairs
const DATA: [(f64, f64); 35] = [
    (400.0, 0.0001575),
    (405.0, 0.0001699),
    (410.0, 0.0002350),
    (415.0, 0.0003102),
    (420.0, 0.0004917),
    (425.0, 0.0008710),
    (430.0, 0.0017418),
    (435.0, 0.0046400),
    (436.5, 0.0065895),
    (438.0, 0.0097302),
    (439.5, 0.0149002),
    (441.0, 0.0237310),
    (442.5, 0.0401683),
    (444.0, 0.0712559),
    (445.5, 0.1264458),
    (447.0, 0.2073413),
    (448.5, 0.2902366),
    (450.0, 0.3445623),
    (451.5, 0.3698049),
    (453.0, 0.3668534),
    (454.5, 0.3106727),
    (456.0, 0.2078154),
    (457.5, 0.1164354),
    (459.0, 0.0616764),
    (460.5, 0.0337200),
    (462.0, 0.0194023),
    (463.5, 0.0117831),
    (465.0, 0.0074357),
    (470.0, 0.0022732),
    (475.0, 0.0008800),
    (480.0, 0.0004579),
    (485.0, 0.0002345),
    (490.0, 0.0001586),
    (495.0, 0.0001143),
    (500.0, 0.0000710),
];

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 3] = [
    1.5543827178E+00, // b1
    4.0888321754E+00, // b2
    4.5154121844E+02, // b3
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 3] = [
    1.5408051163E-02, // b1
    4.6803020753E-02, // b2
    4.6800518816E-02, // b3
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 1.4635887487E-03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 3] = [1.0, 10.0, 500.0];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 3] = [1.5, 5.0, 450.0];

/// Eckerle4 problem: circular interference transmittance
#[derive(Clone, Debug, Default)]
pub struct Eckerle4;

impl Problem for Eckerle4 {
    fn name(&self) -> &str {
        "Eckerle4"
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
                let arg = (x - b[2]) / b[1];
                let model = (b[0] / b[1]) * (-0.5 * arg * arg).exp();
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 3);

        let mut entries = Vec::with_capacity(DATA.len() * 3);

        for (i, &(x, _y)) in DATA.iter().enumerate() {
            let arg = (x - b[2]) / b[1];
            let exp_term = (-0.5 * arg * arg).exp();

            // d/db1 = -exp_term / b2
            entries.push((i, 0, -exp_term / b[1]));

            // d/db2 = -(b1/b2^2) * exp_term * (1 - arg^2)
            entries.push((i, 1, -(b[0] / (b[1] * b[1])) * exp_term * (1.0 - arg * arg)));

            // d/db3 = -(b1/b2) * exp_term * arg / b2
            entries.push((i, 2, -(b[0] / b[1]) * exp_term * arg / b[1]));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Eckerle4 {
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
    fn test_eckerle4_dimensions() {
        let problem = Eckerle4;
        assert_eq!(problem.residual_count(), 35);
        assert_eq!(problem.variable_count(), 3);
    }

    #[test]
    fn test_eckerle4_at_certified() {
        let problem = Eckerle4;
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
