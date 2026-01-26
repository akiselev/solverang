//! Kowalik and Osborne function (MGH Problem 9).
//!
//! A 4-variable data fitting problem with 11 residual equations.
//!
//! # Mathematical Definition
//!
//! Given data points (v_i, y_i) for i = 1, ..., 11
//!
//! Residuals (m=11, n=4):
//! - F_i(x) = y_i - x_1*(v_i^2 + v_i*x_2) / (v_i^2 + v_i*x_3 + x_4)
//!
//! Starting point: x_0 = (0.25, 0.39, 0.415, 0.39)

use crate::Problem;

/// Kowalik and Osborne function problem.
#[derive(Clone, Debug, Default)]
pub struct KowalikOsborne;

/// v data values
const V_DATA: [f64; 11] = [
    4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625,
];

/// y data values
const Y_DATA: [f64; 11] = [
    0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
];

impl Problem for KowalikOsborne {
    fn name(&self) -> &str {
        "Kowalik-Osborne"
    }

    fn residual_count(&self) -> usize {
        11
    }

    fn variable_count(&self) -> usize {
        4
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 4);

        (0..11)
            .map(|i| {
                let v = V_DATA[i];
                let v2 = v * v;
                let numer = x[0] * (v2 + v * x[1]);
                let denom = v2 + v * x[2] + x[3];
                Y_DATA[i] - numer / denom
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 4);

        let mut entries = Vec::with_capacity(44);

        for (i, &v) in V_DATA.iter().enumerate() {
            let v2 = v * v;

            let numer = v2 + v * x[1];
            let denom = v2 + v * x[2] + x[3];
            let denom_sq = denom * denom;

            // dF/dx_1 = -numer/denom
            entries.push((i, 0, -numer / denom));

            // dF/dx_2 = -x_1*v/denom
            entries.push((i, 1, -x[0] * v / denom));

            // dF/dx_3 = x_1*numer*v/denom^2
            entries.push((i, 2, x[0] * numer * v / denom_sq));

            // dF/dx_4 = x_1*numer/denom^2
            entries.push((i, 3, x[0] * numer / denom_sq));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![
            0.25 * factor,
            0.39 * factor,
            0.415 * factor,
            0.39 * factor,
        ]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // Approximate solution
        Some(vec![
            0.1928069345603173E+00,
            0.1912823287772926E+00,
            0.1230565070045498E+00,
            0.1360623308065158E+00,
        ])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.3075056038E-03_f64.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kowalik_osborne_dimensions() {
        let problem = KowalikOsborne;
        assert_eq!(problem.residual_count(), 11);
        assert_eq!(problem.variable_count(), 4);
    }
}
