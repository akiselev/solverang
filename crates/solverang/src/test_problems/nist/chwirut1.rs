//! Chwirut1 - NIST StRD Nonlinear Regression Problem (Lower Difficulty)
//!
//! This problem involves ultrasonic reference block data.
//!
//! Model: y = exp(-b1*x) / (b2 + b3*x)
//!
//! Parameters: 3
//! Observations: 214
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/chwirut1.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Data for Chwirut1 problem: (y, x) pairs - note NIST format is y, x
/// Data from NIST website
fn get_data() -> Vec<(f64, f64)> {
    // Format: (y, x) - we return (x, y) for our convention
    let raw_data = [
        (92.9000E0, 0.5000E0),
        (78.7000E0, 0.6250E0),
        (64.2000E0, 0.7500E0),
        (64.9000E0, 0.8750E0),
        (57.1000E0, 1.0000E0),
        (43.3000E0, 1.2500E0),
        (31.1000E0, 1.7500E0),
        (23.6000E0, 2.2500E0),
        (31.0500E0, 1.7500E0),
        (23.7750E0, 2.2500E0),
        (17.7375E0, 2.7500E0),
        (13.8000E0, 3.2500E0),
        (11.5875E0, 3.7500E0),
        (9.4125E0, 4.2500E0),
        (7.7250E0, 4.7500E0),
        (7.3500E0, 5.2500E0),
        (8.0250E0, 5.7500E0),
        (90.6000E0, 0.5000E0),
        (76.9000E0, 0.6250E0),
        (71.6000E0, 0.7500E0),
        (63.6000E0, 0.8750E0),
        (54.0000E0, 1.0000E0),
        (39.2000E0, 1.2500E0),
        (29.3000E0, 1.7500E0),
        (21.4000E0, 2.2500E0),
        (29.1750E0, 1.7500E0),
        (22.1250E0, 2.2500E0),
        (17.5125E0, 2.7500E0),
        (14.2500E0, 3.2500E0),
        (9.4500E0, 3.7500E0),
        (9.1500E0, 4.2500E0),
        (7.9125E0, 4.7500E0),
        (8.4750E0, 5.2500E0),
        (6.1125E0, 5.7500E0),
        (80.0000E0, 0.5000E0),
        (79.0000E0, 0.6250E0),
        (63.8000E0, 0.7500E0),
        (57.2000E0, 0.8750E0),
        (53.2000E0, 1.0000E0),
        (42.5000E0, 1.2500E0),
        (26.8000E0, 1.7500E0),
        (20.4000E0, 2.2500E0),
        (26.8500E0, 1.7500E0),
        (21.0000E0, 2.2500E0),
        (16.4625E0, 2.7500E0),
        (12.5250E0, 3.2500E0),
        (10.5375E0, 3.7500E0),
        (8.5875E0, 4.2500E0),
        (7.1250E0, 4.7500E0),
        (6.1125E0, 5.2500E0),
        (5.9625E0, 5.7500E0),
        (74.1000E0, 0.5000E0),
        (67.3000E0, 0.6250E0),
        (60.8000E0, 0.7500E0),
        (55.5000E0, 0.8750E0),
        (50.3000E0, 1.0000E0),
        (41.0000E0, 1.2500E0),
        (29.4000E0, 1.7500E0),
        (20.4000E0, 2.2500E0),
        (29.3625E0, 1.7500E0),
        (21.1500E0, 2.2500E0),
        (16.7625E0, 2.7500E0),
        (13.2000E0, 3.2500E0),
        (10.8750E0, 3.7500E0),
        (8.1750E0, 4.2500E0),
        (7.3500E0, 4.7500E0),
        (5.9625E0, 5.2500E0),
        (5.6250E0, 5.7500E0),
        (81.5000E0, 0.5000E0),
        (62.4000E0, 0.7500E0),
        (32.5000E0, 1.5000E0),
        (12.4100E0, 3.0000E0),
        (13.1200E0, 3.0000E0),
        (15.5600E0, 3.0000E0),
        (5.6300E0, 6.0000E0),
        (78.0000E0, 0.5000E0),
        (59.9000E0, 0.7500E0),
        (33.2000E0, 1.5000E0),
        (13.8400E0, 3.0000E0),
        (12.7500E0, 3.0000E0),
        (14.6200E0, 3.0000E0),
        (3.9400E0, 6.0000E0),
        (76.8000E0, 0.5000E0),
        (61.0000E0, 0.7500E0),
        (32.9000E0, 1.5000E0),
        (13.8700E0, 3.0000E0),
        (11.8100E0, 3.0000E0),
        (13.3100E0, 3.0000E0),
        (5.4400E0, 6.0000E0),
        (78.0000E0, 0.5000E0),
        (63.5000E0, 0.7500E0),
        (33.8000E0, 1.5000E0),
        (12.5600E0, 3.0000E0),
        (5.6300E0, 6.0000E0),
        (12.7500E0, 3.0000E0),
        (13.1200E0, 3.0000E0),
        (5.4400E0, 6.0000E0),
        (76.8000E0, 0.5000E0),
        (60.0000E0, 0.7500E0),
        (47.8000E0, 1.0000E0),
        (32.0000E0, 1.5000E0),
        (22.2000E0, 2.0000E0),
        (22.5700E0, 2.0000E0),
        (18.8200E0, 2.5000E0),
        (13.9500E0, 3.0000E0),
        (11.2500E0, 4.0000E0),
        (9.0000E0, 5.0000E0),
        (6.6700E0, 6.0000E0),
        (75.8000E0, 0.5000E0),
        (62.0000E0, 0.7500E0),
        (48.8000E0, 1.0000E0),
        (35.2000E0, 1.5000E0),
        (20.0000E0, 2.0000E0),
        (20.3200E0, 2.0000E0),
        (19.3100E0, 2.5000E0),
        (12.7500E0, 3.0000E0),
        (10.4200E0, 4.0000E0),
        (7.3100E0, 5.0000E0),
        (7.4200E0, 6.0000E0),
        (70.5000E0, 0.5000E0),
        (59.5000E0, 0.7500E0),
        (48.5000E0, 1.0000E0),
        (35.8000E0, 1.5000E0),
        (21.0000E0, 2.0000E0),
        (21.6700E0, 2.0000E0),
        (21.0000E0, 2.5000E0),
        (15.6400E0, 3.0000E0),
        (8.1700E0, 4.0000E0),
        (8.5500E0, 5.0000E0),
        (10.1200E0, 6.0000E0),
        (78.0000E0, 0.5000E0),
        (66.0000E0, 0.6250E0),
        (62.0000E0, 0.7500E0),
        (58.0000E0, 0.8750E0),
        (47.7000E0, 1.0000E0),
        (37.8000E0, 1.2500E0),
        (20.2000E0, 2.2500E0),
        (21.0700E0, 2.2500E0),
        (13.8700E0, 2.7500E0),
        (9.6700E0, 3.2500E0),
        (7.7600E0, 3.7500E0),
        (5.4400E0, 4.2500E0),
        (4.8700E0, 4.7500E0),
        (4.0100E0, 5.2500E0),
        (3.7500E0, 5.7500E0),
        (24.1900E0, 3.0000E0),
        (25.7600E0, 3.0000E0),
        (18.0700E0, 3.0000E0),
        (11.8100E0, 3.0000E0),
        (12.0700E0, 3.0000E0),
        (16.1200E0, 3.0000E0),
        (70.8000E0, 0.5000E0),
        (54.7000E0, 0.7500E0),
        (48.0000E0, 1.0000E0),
        (39.8000E0, 1.5000E0),
        (29.8000E0, 2.0000E0),
        (23.7000E0, 2.5000E0),
        (29.6200E0, 2.0000E0),
        (23.8100E0, 2.5000E0),
        (17.7000E0, 3.0000E0),
        (11.5500E0, 4.0000E0),
        (12.0700E0, 5.0000E0),
        (8.7400E0, 6.0000E0),
        (80.7000E0, 0.5000E0),
        (61.3000E0, 0.7500E0),
        (47.5000E0, 1.0000E0),
        (29.0000E0, 1.5000E0),
        (24.0000E0, 2.0000E0),
        (17.7000E0, 2.5000E0),
        (24.5600E0, 2.0000E0),
        (18.6700E0, 2.5000E0),
        (16.2400E0, 3.0000E0),
        (8.7400E0, 4.0000E0),
        (7.8700E0, 5.0000E0),
        (8.5100E0, 6.0000E0),
        (66.7000E0, 0.5000E0),
        (59.2000E0, 0.7500E0),
        (40.8000E0, 1.0000E0),
        (30.7000E0, 1.5000E0),
        (25.7000E0, 2.0000E0),
        (16.3000E0, 2.5000E0),
        (25.9900E0, 2.0000E0),
        (16.9500E0, 2.5000E0),
        (13.3500E0, 3.0000E0),
        (8.6200E0, 4.0000E0),
        (7.2000E0, 5.0000E0),
        (6.6400E0, 6.0000E0),
        (13.6900E0, 3.0000E0),
        (81.0000E0, 0.5000E0),
        (64.5000E0, 0.7500E0),
        (35.5000E0, 1.5000E0),
        (13.3100E0, 3.0000E0),
        (4.8700E0, 6.0000E0),
        (12.9400E0, 3.0000E0),
        (5.0600E0, 6.0000E0),
        (15.1900E0, 3.0000E0),
        (14.6200E0, 3.0000E0),
        (15.6400E0, 3.0000E0),
        (25.5000E0, 1.7500E0),
        (25.9500E0, 1.7500E0),
        (81.7000E0, 0.5000E0),
        (61.6000E0, 0.7500E0),
        (29.8000E0, 1.7500E0),
        (29.8100E0, 1.7500E0),
        (17.1700E0, 2.7500E0),
        (10.3900E0, 3.7500E0),
        (28.4000E0, 1.7500E0),
        (28.6900E0, 1.7500E0),
        (81.3000E0, 0.5000E0),
        (60.9000E0, 0.7500E0),
        (16.6500E0, 2.7500E0),
        (10.0500E0, 3.7500E0),
        (28.9000E0, 1.7500E0),
        (28.9500E0, 1.7500E0),
    ];

    raw_data.iter().map(|&(y, x)| (x, y)).collect()
}

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 3] = [
    1.9027818370E-01,  // b1
    6.1314004477E-03,  // b2
    1.0530908399E-02,  // b3
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 3] = [
    2.1938557035E-02,  // b1
    3.4500025051E-04,  // b2
    7.9281847748E-04,  // b3
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 2.3844771393E+03;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 3] = [0.1, 0.01, 0.02];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 3] = [0.15, 0.008, 0.010];

/// Chwirut1 problem: ultrasonic reference block
#[derive(Clone, Debug, Default)]
pub struct Chwirut1;

impl Problem for Chwirut1 {
    fn name(&self) -> &str {
        "Chwirut1"
    }

    fn residual_count(&self) -> usize {
        214
    }

    fn variable_count(&self) -> usize {
        3
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 3);

        let data = get_data();
        data.iter()
            .map(|&(x, y)| {
                let model = (-b[0] * x).exp() / (b[1] + b[2] * x);
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 3);

        let data = get_data();
        let mut entries = Vec::with_capacity(data.len() * 3);

        for (i, &(x, _y)) in data.iter().enumerate() {
            let exp_term = (-b[0] * x).exp();
            let denom = b[1] + b[2] * x;
            let denom_sq = denom * denom;

            // d(residual)/db1 = x * exp(-b1*x) / (b2 + b3*x)
            entries.push((i, 0, x * exp_term / denom));

            // d(residual)/db2 = exp(-b1*x) / (b2 + b3*x)^2
            entries.push((i, 1, exp_term / denom_sq));

            // d(residual)/db3 = x * exp(-b1*x) / (b2 + b3*x)^2
            entries.push((i, 2, x * exp_term / denom_sq));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Chwirut1 {
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
    fn test_chwirut1_dimensions() {
        let problem = Chwirut1;
        assert_eq!(problem.residual_count(), 214);
        assert_eq!(problem.variable_count(), 3);
    }

    #[test]
    fn test_chwirut1_at_certified() {
        let problem = Chwirut1;
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
