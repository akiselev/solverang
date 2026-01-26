//! Variably dimensioned function (HYBRJ Problem 12).
//!
//! A variable-dimension nonlinear equation problem.
//!
//! # Mathematical Definition
//!
//! Let s = sum_{j=1}^n j*(x_j - 1)
//!
//! Residuals (n variables):
//! - F_i(x) = x_i - 1 + i * s * (1 + 2*s^2),  for i = 1, ..., n
//!
//! Minimum: x* = (1, 1, ..., 1) with F(x*) = 0
//!
//! Starting point: x_0_i = 1 - i/n

use crate::{ConfigurableProblem, Problem};

/// Variably dimensioned function problem.
#[derive(Clone, Debug)]
pub struct VariablyDimensioned {
    n: usize,
}

impl VariablyDimensioned {
    /// Create a new VariablyDimensioned problem with the specified dimension.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "VariablyDimensioned requires n >= 1");
        Self { n }
    }
}

impl Default for VariablyDimensioned {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Problem for VariablyDimensioned {
    fn name(&self) -> &str {
        "Variably Dimensioned"
    }

    fn residual_count(&self) -> usize {
        self.n
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        // s = sum_j j*(x_j - 1)
        let s: f64 = x.iter()
            .enumerate()
            .map(|(j, &xj)| ((j + 1) as f64) * (xj - 1.0))
            .sum();

        let temp = s * (1.0 + 2.0 * s * s);

        (0..self.n)
            .map(|i| {
                let i_f = (i + 1) as f64;
                x[i] - 1.0 + i_f * temp
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        // s = sum_j j*(x_j - 1)
        let s: f64 = x.iter()
            .enumerate()
            .map(|(j, &xj)| ((j + 1) as f64) * (xj - 1.0))
            .sum();

        // ds/dx_k = k
        // d(s*(1 + 2s^2))/ds = 1 + 6s^2
        // dF_i/dx_k = delta_ik + i * k * (1 + 6s^2)

        let ds_multiplier = 1.0 + 6.0 * s * s;

        let mut entries = Vec::with_capacity(self.n * self.n);

        for i in 0..self.n {
            let i_f = (i + 1) as f64;

            for k in 0..self.n {
                let k_f = (k + 1) as f64;
                let diag = if i == k { 1.0 } else { 0.0 };
                entries.push((i, k, diag + i_f * k_f * ds_multiplier));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        (1..=self.n)
            .map(|i| (1.0 - (i as f64) / (self.n as f64)) * factor)
            .collect()
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0; self.n])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl ConfigurableProblem for VariablyDimensioned {
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
    fn test_variably_dimensioned_at_solution() {
        let problem = VariablyDimensioned::new(10);
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Residuals at solution should be zero: {:?}",
            residuals
        );
    }
}
