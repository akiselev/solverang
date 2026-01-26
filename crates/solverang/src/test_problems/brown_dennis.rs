//! Brown and Dennis function (MGH Problem 14).
//!
//! A 4-variable least-squares problem with m residual equations (m >= 4).
//!
//! # Mathematical Definition
//!
//! For t_i = i/5:
//!
//! Residuals (m >= 4, n=4):
//! - F_i(x) = (x_1 + t_i*x_2 - exp(t_i))^2 + (x_3 + x_4*sin(t_i) - cos(t_i))^2
//!
//! Starting point: x_0 = (25, 5, -5, -1)

use crate::{ConfigurableProblem, Problem};

/// Brown and Dennis function problem.
#[derive(Clone, Debug)]
pub struct BrownDennis {
    m: usize,
}

impl BrownDennis {
    /// Create a new BrownDennis problem with the specified number of equations.
    pub fn new(m: usize) -> Self {
        assert!(m >= 4, "BrownDennis requires m >= 4");
        Self { m }
    }
}

impl Default for BrownDennis {
    fn default() -> Self {
        Self::new(20)
    }
}

impl Problem for BrownDennis {
    fn name(&self) -> &str {
        "Brown-Dennis"
    }

    fn residual_count(&self) -> usize {
        self.m
    }

    fn variable_count(&self) -> usize {
        4
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 4);

        (1..=self.m)
            .map(|i| {
                let t = (i as f64) / 5.0;
                let a = x[0] + t * x[1] - t.exp();
                let b = x[2] + x[3] * t.sin() - t.cos();
                a * a + b * b
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 4);

        let mut entries = Vec::with_capacity(4 * self.m);

        for i in 1..=self.m {
            let row = i - 1;
            let t = (i as f64) / 5.0;
            let a = x[0] + t * x[1] - t.exp();
            let b = x[2] + x[3] * t.sin() - t.cos();

            // dF/dx_1 = 2a
            entries.push((row, 0, 2.0 * a));

            // dF/dx_2 = 2a*t
            entries.push((row, 1, 2.0 * a * t));

            // dF/dx_3 = 2b
            entries.push((row, 2, 2.0 * b));

            // dF/dx_4 = 2b*sin(t)
            entries.push((row, 3, 2.0 * b * t.sin()));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![25.0 * factor, 5.0 * factor, -5.0 * factor, -factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // The solution is highly dependent on m
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        if self.m == 20 {
            Some(0.8582220162635628E+05_f64.sqrt())
        } else {
            None
        }
    }
}

impl ConfigurableProblem for BrownDennis {
    fn with_dimensions(_n: usize, m: usize) -> Option<Self> {
        if m >= 4 {
            Some(Self::new(m))
        } else {
            None
        }
    }

    fn min_variables() -> usize {
        4
    }

    fn max_variables() -> Option<usize> {
        Some(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brown_dennis_dimensions() {
        let problem = BrownDennis::default();
        assert_eq!(problem.residual_count(), 20);
        assert_eq!(problem.variable_count(), 4);
    }
}
