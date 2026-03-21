//! Osborne 1 function (MGH Problem 17).
//!
//! A 5-variable data fitting problem with 33 residual equations.
//!
//! # Mathematical Definition
//!
//! Given 33 data points y_i:
//!
//! Residuals (m=33, n=5):
//! - F_i(x) = y_i - (x_1 + x_2*exp(-t_i*x_4) + x_3*exp(-t_i*x_5))
//!
//! where t_i = 10(i-1) for i = 1, ..., 33
//!
//! Starting point: x_0 = (0.5, 1.5, -1, 0.01, 0.02)

use crate::Problem;

/// Osborne 1 function problem.
#[derive(Clone, Debug, Default)]
pub struct Osborne1;

/// Observed y data values
const Y_DATA: [f64; 33] = [
    0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818, 0.784, 0.751, 0.718, 0.685,
    0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522, 0.506, 0.490, 0.478, 0.467, 0.457, 0.448,
    0.438, 0.431, 0.424, 0.420, 0.414, 0.411, 0.406,
];

impl Problem for Osborne1 {
    fn name(&self) -> &str {
        "Osborne 1"
    }

    fn residual_count(&self) -> usize {
        33
    }

    fn variable_count(&self) -> usize {
        5
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 5);

        (0..33)
            .map(|i| {
                let t = 10.0 * (i as f64);
                Y_DATA[i] - (x[0] + x[1] * (-t * x[3]).exp() + x[2] * (-t * x[4]).exp())
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 5);

        let mut entries = Vec::with_capacity(5 * 33);

        for i in 0..33 {
            let t = 10.0 * (i as f64);
            let e1 = (-t * x[3]).exp();
            let e2 = (-t * x[4]).exp();

            entries.push((i, 0, -1.0));
            entries.push((i, 1, -e1));
            entries.push((i, 2, -e2));
            entries.push((i, 3, t * x[1] * e1));
            entries.push((i, 4, t * x[2] * e2));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![
            0.5 * factor,
            1.5 * factor,
            -factor,
            0.01 * factor,
            0.02 * factor,
        ]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![
            0.3754100559069308E+00,
            0.1935846300456624E+01,
            -0.1464686549065656E+01,
            0.1286753391113679E-01,
            0.2212269810983129E-01,
        ])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.5464894697174226E-04_f64.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osborne1_dimensions() {
        let problem = Osborne1;
        assert_eq!(problem.residual_count(), 33);
        assert_eq!(problem.variable_count(), 5);
    }
}
