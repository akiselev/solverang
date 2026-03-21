//! Brown almost-linear function (MGH Problem 16, HYBRJ Problem 8).
//!
//! A variable-dimension problem with n variables and n equations.
//!
//! # Mathematical Definition
//!
//! Residuals (m = n, n >= 1):
//! - F_i(x) = x_i + sum_{j=1}^n x_j - (n + 1),  for i = 1, ..., n-1
//! - F_n(x) = (prod_{j=1}^n x_j) - 1
//!
//! Starting point: x_0 = (0.5, 0.5, ..., 0.5)
//!
//! Solution: x* = (1, 1, ..., 1) or x* = (alpha, alpha, ..., alpha, alpha^{1-n})
//! where alpha satisfies n*alpha^n + (n-1)*alpha^{n-1} - (n+1) = 0

use crate::{ConfigurableProblem, Problem};

/// Brown almost-linear function problem.
#[derive(Clone, Debug)]
pub struct BrownAlmostLinear {
    n: usize,
}

impl BrownAlmostLinear {
    /// Create a new BrownAlmostLinear problem with the specified dimension.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "BrownAlmostLinear requires n >= 1");
        Self { n }
    }
}

impl Default for BrownAlmostLinear {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Problem for BrownAlmostLinear {
    fn name(&self) -> &str {
        "Brown Almost-Linear"
    }

    fn residual_count(&self) -> usize {
        self.n
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        let sum: f64 = x.iter().sum();
        let n_plus_1 = (self.n + 1) as f64;

        let mut fvec: Vec<f64> = x
            .iter()
            .take(self.n - 1)
            .map(|&xi| xi + sum - n_plus_1)
            .collect();

        // Last equation: product - 1
        let prod: f64 = x.iter().product();
        fvec.push(prod - 1.0);

        fvec
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        let mut entries = Vec::with_capacity(self.n * self.n);

        // First n-1 equations: F_i = x_i + sum - (n+1)
        // dF_i/dx_j = 1 + delta_ij (Kronecker delta)
        for i in 0..self.n - 1 {
            for j in 0..self.n {
                let val = if i == j { 2.0 } else { 1.0 };
                entries.push((i, j, val));
            }
        }

        // Last equation: F_n = prod(x_j) - 1
        // dF_n/dx_k = prod_{j != k} x_j
        let prod: f64 = x.iter().product();
        for k in 0..self.n {
            let val = if x[k].abs() > 1e-30 {
                prod / x[k]
            } else {
                // Compute product excluding x[k]
                x.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != k)
                    .map(|(_, &xj)| xj)
                    .product()
            };
            entries.push((self.n - 1, k, val));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![0.5 * factor; self.n]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // The obvious solution is all ones
        Some(vec![1.0; self.n])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl ConfigurableProblem for BrownAlmostLinear {
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
    fn test_brown_almost_linear_at_solution() {
        let problem = BrownAlmostLinear::new(5);
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);
        assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Residuals at solution should be zero: {:?}",
            residuals
        );
    }
}
