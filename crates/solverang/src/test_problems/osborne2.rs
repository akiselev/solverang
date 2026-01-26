//! Osborne 2 function (MGH Problem 18).
//!
//! An 11-variable data fitting problem with 65 residual equations.
//!
//! # Mathematical Definition
//!
//! Given 65 data points y_i:
//!
//! Residuals (m=65, n=11):
//! - F_i(x) = y_i - (x_1*exp(-t_i*x_5) + x_2*exp(-(t_i-x_9)^2*x_6)
//!   + x_3*exp(-(t_i-x_10)^2*x_7) + x_4*exp(-(t_i-x_11)^2*x_8))
//!
//! where t_i = (i-1)/10 for i = 1, ..., 65
//!
//! Starting point: x_0 = (1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5)

use crate::Problem;

/// Osborne 2 function problem.
#[derive(Clone, Debug, Default)]
pub struct Osborne2;

/// Observed y data values
const Y_DATA: [f64; 65] = [
    1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746,
    0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649,
    0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.500, 0.423, 0.395,
    0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653,
    0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739,
    0.710, 0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054,
];

impl Problem for Osborne2 {
    fn name(&self) -> &str {
        "Osborne 2"
    }

    fn residual_count(&self) -> usize {
        65
    }

    fn variable_count(&self) -> usize {
        11
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 11);

        (0..65)
            .map(|i| {
                let t = (i as f64) / 10.0;

                let term1 = x[0] * (-t * x[4]).exp();
                let term2 = x[1] * (-(t - x[8]).powi(2) * x[5]).exp();
                let term3 = x[2] * (-(t - x[9]).powi(2) * x[6]).exp();
                let term4 = x[3] * (-(t - x[10]).powi(2) * x[7]).exp();

                Y_DATA[i] - (term1 + term2 + term3 + term4)
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 11);

        let mut entries = Vec::with_capacity(11 * 65);

        for i in 0..65 {
            let t = (i as f64) / 10.0;

            let e1 = (-t * x[4]).exp();
            let d2 = t - x[8];
            let e2 = (-d2 * d2 * x[5]).exp();
            let d3 = t - x[9];
            let e3 = (-d3 * d3 * x[6]).exp();
            let d4 = t - x[10];
            let e4 = (-d4 * d4 * x[7]).exp();

            // dF/dx_1 = -e1
            entries.push((i, 0, -e1));
            // dF/dx_2 = -e2
            entries.push((i, 1, -e2));
            // dF/dx_3 = -e3
            entries.push((i, 2, -e3));
            // dF/dx_4 = -e4
            entries.push((i, 3, -e4));

            // dF/dx_5 = t*x_1*e1
            entries.push((i, 4, t * x[0] * e1));
            // dF/dx_6 = x_2*(t-x_9)^2*e2
            entries.push((i, 5, x[1] * d2 * d2 * e2));
            // dF/dx_7 = x_3*(t-x_10)^2*e3
            entries.push((i, 6, x[2] * d3 * d3 * e3));
            // dF/dx_8 = x_4*(t-x_11)^2*e4
            entries.push((i, 7, x[3] * d4 * d4 * e4));

            // dF/dx_9 = -2*x_2*x_6*(t-x_9)*e2
            entries.push((i, 8, -2.0 * x[1] * x[5] * d2 * e2));
            // dF/dx_10 = -2*x_3*x_7*(t-x_10)*e3
            entries.push((i, 9, -2.0 * x[2] * x[6] * d3 * e3));
            // dF/dx_11 = -2*x_4*x_8*(t-x_11)*e4
            entries.push((i, 10, -2.0 * x[3] * x[7] * d4 * e4));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![
            1.3 * factor,
            0.65 * factor,
            0.65 * factor,
            0.7 * factor,
            0.6 * factor,
            3.0 * factor,
            5.0 * factor,
            7.0 * factor,
            2.0 * factor,
            4.5 * factor,
            5.5 * factor,
        ]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // Approximate solution
        #[allow(clippy::excessive_precision)]
        Some(vec![
            0.1309976653844442E+01,
            0.4315524807960760E+00,
            0.6336612597880877E+00,
            0.5994297653155811E+00,
            0.7541877909490765E+00,
            0.9042887614028033E+00,
            0.1365811062301616E+01,
            0.4823697391336689E+01,
            0.2398684748716189E+01,
            0.4568874480509978E+01,
            0.5675342062618567E+01,
        ])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.4013774100046607E-01_f64.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osborne2_dimensions() {
        let problem = Osborne2;
        assert_eq!(problem.residual_count(), 65);
        assert_eq!(problem.variable_count(), 11);
    }
}
