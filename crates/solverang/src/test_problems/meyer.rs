//! Meyer function (MGH Problem 10).
//!
//! A 3-variable data fitting problem with 16 residual equations.
//! This is one of the more difficult problems due to the exponential.
//!
//! # Mathematical Definition
//!
//! Given data points y_i for i = 1, ..., 16:
//!
//! Residuals (m=16, n=3):
//! - F_i(x) = x_1 * exp(x_2/(45 + 5i + x_3)) - y_i
//!
//! Starting point: x_0 = (0.02, 4000, 250)

use crate::Problem;

/// Meyer function problem.
#[derive(Clone, Debug, Default)]
pub struct Meyer;

/// Observed data values
const Y_DATA: [f64; 16] = [
    34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0, 11540.0, 9744.0,
    8261.0, 7030.0, 6005.0, 5147.0, 4427.0, 3820.0, 3307.0, 2872.0,
];

impl Problem for Meyer {
    fn name(&self) -> &str {
        "Meyer"
    }

    fn residual_count(&self) -> usize {
        16
    }

    fn variable_count(&self) -> usize {
        3
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 3);

        (1..=16)
            .map(|i| {
                let t = 45.0 + 5.0 * (i as f64);
                let exponent = x[1] / (t + x[2]);
                x[0] * exponent.exp() - Y_DATA[i - 1]
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 3);

        let mut entries = Vec::with_capacity(48);

        for i in 1..=16 {
            let row = i - 1;
            let t = 45.0 + 5.0 * (i as f64);
            let denom = t + x[2];
            let exponent = x[1] / denom;
            let exp_val = exponent.exp();

            // dF/dx_1 = exp(x_2/(t + x_3))
            entries.push((row, 0, exp_val));

            // dF/dx_2 = x_1 * exp(...) / (t + x_3)
            entries.push((row, 1, x[0] * exp_val / denom));

            // dF/dx_3 = -x_1 * x_2 * exp(...) / (t + x_3)^2
            entries.push((row, 2, -x[0] * x[1] * exp_val / (denom * denom)));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![0.02 * factor, 4000.0 * factor, 250.0 * factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // Approximate solution
        #[allow(clippy::excessive_precision)]
        Some(vec![
            0.5609636471026059E-02,
            0.6181346346286660E+04,
            0.3452236346241440E+03,
        ])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.8794585517E+02_f64.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meyer_dimensions() {
        let problem = Meyer;
        assert_eq!(problem.residual_count(), 16);
        assert_eq!(problem.variable_count(), 3);
    }
}
