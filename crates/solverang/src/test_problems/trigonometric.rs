//! Trigonometric function (HYBRJ Problem 11).
//!
//! A variable-dimension nonlinear equation problem with trigonometric functions.
//!
//! # Mathematical Definition
//!
//! Residuals (n variables):
//! - F_i(x) = n - sum_{j=1}^n cos(x_j) + i*(1 - cos(x_i)) - sin(x_i)
//!
//! Starting point: x_0 = (1/n, 1/n, ..., 1/n)

use crate::{ConfigurableProblem, Problem};

/// Trigonometric function problem.
#[derive(Clone, Debug)]
pub struct Trigonometric {
    n: usize,
}

impl Trigonometric {
    /// Create a new Trigonometric problem with the specified dimension.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "Trigonometric requires n >= 1");
        Self { n }
    }
}

impl Default for Trigonometric {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Problem for Trigonometric {
    fn name(&self) -> &str {
        "Trigonometric"
    }

    fn residual_count(&self) -> usize {
        self.n
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        let n_f = self.n as f64;
        let cos_sum: f64 = x.iter().map(|&xi| xi.cos()).sum();

        (0..self.n)
            .map(|i| {
                let i_f = (i + 1) as f64;
                n_f - cos_sum + i_f * (1.0 - x[i].cos()) - x[i].sin()
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        let mut entries = Vec::with_capacity(self.n * self.n);

        for i in 0..self.n {
            let i_f = (i + 1) as f64;

            for (j, &xj) in x.iter().enumerate().take(self.n) {
                let val = if i == j {
                    // dF_i/dx_i = sin(x_i) + i*sin(x_i) - cos(x_i)
                    xj.sin() + i_f * xj.sin() - xj.cos()
                } else {
                    // dF_i/dx_j = sin(x_j)
                    xj.sin()
                };
                entries.push((i, j, val));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor / (self.n as f64); self.n]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl ConfigurableProblem for Trigonometric {
    fn with_dimensions(n: usize, _m: usize) -> Option<Self> {
        if n >= 1 {
            Some(Self::new(n))
        } else {
            None
        }
    }

    fn min_variables() -> usize {
        1
    }

    fn max_variables() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigonometric_dimensions() {
        let problem = Trigonometric::new(10);
        assert_eq!(problem.residual_count(), 10);
        assert_eq!(problem.variable_count(), 10);
    }
}
