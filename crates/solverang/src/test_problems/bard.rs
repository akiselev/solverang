//! Bard function (MGH Problem 8).
//!
//! A 3-variable data fitting problem with 15 residual equations.
//!
//! # Mathematical Definition
//!
//! Given data points y_i for i = 1, ..., 15:
//!   y = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39,
//!        0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]
//!
//! And with u_i = i, v_i = 16 - i, w_i = min(u_i, v_i):
//!
//! Residuals (m=15, n=3):
//! - F_i(x) = y_i - (x_1 + u_i/(v_i*x_2 + w_i*x_3))
//!
//! Starting point: x_0 = (1, 1, 1)

use crate::Problem;

/// Bard function problem.
#[derive(Clone, Debug, Default)]
pub struct Bard;

/// Observed data values for the Bard problem.
const Y_DATA: [f64; 15] = [
    0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39,
];

impl Problem for Bard {
    fn name(&self) -> &str {
        "Bard"
    }

    fn residual_count(&self) -> usize {
        15
    }

    fn variable_count(&self) -> usize {
        3
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 3);

        (1..=15)
            .map(|i| {
                let u = i as f64;
                let v = (16 - i) as f64;
                let w = u.min(v);

                let denom = v * x[1] + w * x[2];
                Y_DATA[i - 1] - (x[0] + u / denom)
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 3);

        let mut entries = Vec::with_capacity(45);

        for i in 1..=15 {
            let row = i - 1;
            let u = i as f64;
            let v = (16 - i) as f64;
            let w = u.min(v);

            let denom = v * x[1] + w * x[2];
            let denom_sq = denom * denom;

            entries.push((row, 0, -1.0));
            entries.push((row, 1, u * v / denom_sq));
            entries.push((row, 2, u * w / denom_sq));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![1.0 * factor, 1.0 * factor, 1.0 * factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // Approximate solution from MINPACK
        #[allow(clippy::excessive_precision)]
        Some(vec![
            0.08241056031474880E+00,
            0.1133033982064098E+01,
            0.2343694638782446E+01,
        ])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.8214877306E-02_f64.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bard_dimensions() {
        let problem = Bard;
        assert_eq!(problem.residual_count(), 15);
        assert_eq!(problem.variable_count(), 3);
    }

    #[test]
    fn test_bard_at_approximate_solution() {
        let problem = Bard;
        let solution = problem.known_solution().expect("should have solution");

        let norm = problem.residual_norm(&solution);
        let expected = problem
            .expected_residual_norm()
            .expect("should have expected norm");

        assert!(
            (norm - expected).abs() < 1e-6,
            "Residual norm at solution: {}, expected: {}",
            norm,
            expected
        );
    }
}
