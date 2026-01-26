//! Box three-dimensional function (MGH Problem 12).
//!
//! A 3-variable data fitting problem with m residual equations (m >= 3).
//!
//! # Mathematical Definition
//!
//! Residuals (m >= 3, n=3):
//! - F_i(x) = exp(-t_i*x_1) - exp(-t_i*x_2) - x_3*(exp(-t_i) - exp(-10*t_i))
//!
//! where t_i = 0.1*i for i = 1, ..., m
//!
//! Starting point: x_0 = (0, 10, 20)

use crate::{ConfigurableProblem, Problem};

/// Box three-dimensional function problem.
#[derive(Clone, Debug)]
pub struct Box3D {
    m: usize,
}

impl Box3D {
    /// Create a new Box3D problem with the specified number of equations.
    pub fn new(m: usize) -> Self {
        assert!(m >= 3, "Box3D requires m >= 3");
        Self { m }
    }
}

impl Default for Box3D {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Problem for Box3D {
    fn name(&self) -> &str {
        "Box 3D"
    }

    fn residual_count(&self) -> usize {
        self.m
    }

    fn variable_count(&self) -> usize {
        3
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 3);

        (1..=self.m)
            .map(|i| {
                let t = 0.1 * (i as f64);
                (-t * x[0]).exp() - (-t * x[1]).exp()
                    - x[2] * ((-t).exp() - (-10.0 * t).exp())
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 3);

        let mut entries = Vec::with_capacity(3 * self.m);

        for i in 1..=self.m {
            let row = i - 1;
            let t = 0.1 * (i as f64);

            // dF/dx_1 = -t * exp(-t*x_1)
            entries.push((row, 0, -t * (-t * x[0]).exp()));

            // dF/dx_2 = t * exp(-t*x_2)
            entries.push((row, 1, t * (-t * x[1]).exp()));

            // dF/dx_3 = -(exp(-t) - exp(-10t))
            entries.push((row, 2, -((-t).exp() - (-10.0 * t).exp())));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![0.0, 10.0 * factor, 20.0 * factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0, 10.0, 1.0])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl ConfigurableProblem for Box3D {
    fn with_dimensions(_n: usize, m: usize) -> Option<Self> {
        if m >= 3 {
            Some(Self::new(m))
        } else {
            None
        }
    }

    fn min_variables() -> usize {
        3
    }

    fn max_variables() -> Option<usize> {
        Some(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box3d_at_solution() {
        let problem = Box3D::default();
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Residuals at solution should be zero: {:?}",
            residuals
        );
    }
}
