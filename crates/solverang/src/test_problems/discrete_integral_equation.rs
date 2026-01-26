//! Discrete integral equation function (HYBRJ Problem 10).
//!
//! A variable-dimension discretized integral equation problem.
//!
//! # Mathematical Definition
//!
//! Discretization of: u(t) + integral_0^t K1(t,s)(u(s)+s+1)^3 ds + integral_t^1 K2(t,s)(u(s)+s+1)^3 ds = 0
//!
//! where K1(t,s) = (1-t)*s and K2(t,s) = t*(1-s).
//!
//! With h = 1/(n+1) and t_k = k*h (for k = 1,...,n):
//!
//! Residuals (n variables):
//! - F_k(x) = x_k + (h/2) * [(1-t_k) * sum_{j=1}^k t_j*(x_j+t_j+1)^3 + t_k * sum_{j=k+1}^n (1-t_j)*(x_j+t_j+1)^3]
//!
//! Starting point: x_0_i = t_i*(t_i - 1)

use crate::{ConfigurableProblem, Problem};

/// Discrete integral equation function problem.
#[derive(Clone, Debug)]
pub struct DiscreteIntegralEquation {
    n: usize,
}

impl DiscreteIntegralEquation {
    /// Create a new DiscreteIntegralEquation problem with the specified dimension.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "DiscreteIntegralEquation requires n >= 1");
        Self { n }
    }
}

impl Default for DiscreteIntegralEquation {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Problem for DiscreteIntegralEquation {
    fn name(&self) -> &str {
        "Discrete Integral Equation"
    }

    fn residual_count(&self) -> usize {
        self.n
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        let h = 1.0 / ((self.n + 1) as f64);

        // Pre-compute t values and cubic terms
        let mut t = Vec::with_capacity(self.n);
        let mut cubic = Vec::with_capacity(self.n);
        for (j, &xj) in x.iter().enumerate().take(self.n) {
            let t_j = ((j + 1) as f64) * h;
            t.push(t_j);
            cubic.push((xj + t_j + 1.0).powi(3));
        }

        // Build prefix sum for sum1: prefix_sum1[k] = sum_{j=0}^k t_j * cubic_j
        let mut prefix_sum1 = Vec::with_capacity(self.n);
        let mut acc = 0.0;
        for j in 0..self.n {
            acc += t[j] * cubic[j];
            prefix_sum1.push(acc);
        }

        // Build suffix sum for sum2: suffix_sum2[k] = sum_{j=k}^{n-1} (1-t_j) * cubic_j
        let mut suffix_sum2 = vec![0.0; self.n + 1];
        for j in (0..self.n).rev() {
            suffix_sum2[j] = suffix_sum2[j + 1] + (1.0 - t[j]) * cubic[j];
        }

        // Compute residuals
        (0..self.n)
            .map(|k| {
                let sum1 = prefix_sum1[k];
                let sum2 = suffix_sum2[k + 1];
                x[k] + h * ((1.0 - t[k]) * sum1 + t[k] * sum2) / 2.0
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        let h = 1.0 / ((self.n + 1) as f64);

        let mut entries = Vec::new();

        for k in 0..self.n {
            let t_k = ((k + 1) as f64) * h;

            for (j, &xj) in x.iter().enumerate().take(self.n) {
                let t_j = ((j + 1) as f64) * h;
                let cubic_deriv = 3.0 * (xj + t_j + 1.0).powi(2);

                let val = if j <= k {
                    // Contribution from first sum
                    h * (1.0 - t_k) * t_j * cubic_deriv / 2.0
                } else {
                    // Contribution from second sum
                    h * t_k * (1.0 - t_j) * cubic_deriv / 2.0
                };

                let diag_add = if k == j { 1.0 } else { 0.0 };
                entries.push((k, j, val + diag_add));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        let h = 1.0 / ((self.n + 1) as f64);

        (1..=self.n)
            .map(|i| {
                let t = (i as f64) * h;
                t * (t - 1.0) * factor
            })
            .collect()
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl ConfigurableProblem for DiscreteIntegralEquation {
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
    fn test_discrete_integral_equation_dimensions() {
        let problem = DiscreteIntegralEquation::new(10);
        assert_eq!(problem.residual_count(), 10);
        assert_eq!(problem.variable_count(), 10);
    }
}
