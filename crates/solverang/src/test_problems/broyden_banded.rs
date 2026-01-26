//! Broyden banded function (HYBRJ Problem 14).
//!
//! A variable-dimension nonlinear equation problem with banded Jacobian.
//!
//! # Mathematical Definition
//!
//! Residuals (n variables):
//! - F_k(x) = x_k*(2 + 5*x_k^2) + 1 - sum_{j in J(k)} x_j*(1 + x_j)
//!
//! where J(k) = {j : j != k, max(1,k-5) <= j <= min(n,k+1)}
//!
//! This creates a Jacobian with lower bandwidth 5 and upper bandwidth 1.
//!
//! Starting point: x_0 = (-1, -1, ..., -1)

use crate::{ConfigurableProblem, Problem};

/// Broyden banded function problem.
#[derive(Clone, Debug)]
pub struct BroydenBanded {
    n: usize,
    ml: usize, // lower bandwidth
    mu: usize, // upper bandwidth
}

impl BroydenBanded {
    /// Create a new BroydenBanded problem with the specified dimension.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "BroydenBanded requires n >= 1");
        Self { n, ml: 5, mu: 1 }
    }

    /// Create with custom bandwidths.
    pub fn with_bandwidth(n: usize, ml: usize, mu: usize) -> Self {
        assert!(n >= 1, "BroydenBanded requires n >= 1");
        Self { n, ml, mu }
    }
}

impl Default for BroydenBanded {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Problem for BroydenBanded {
    fn name(&self) -> &str {
        "Broyden Banded"
    }

    fn residual_count(&self) -> usize {
        self.n
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        (0..self.n)
            .map(|k| {
                let k1 = k.saturating_sub(self.ml);
                let k2 = (k + self.mu + 1).min(self.n);

                let sum: f64 = (k1..k2)
                    .filter(|&j| j != k)
                    .map(|j| x[j] * (1.0 + x[j]))
                    .sum();

                x[k] * (2.0 + 5.0 * x[k].powi(2)) + 1.0 - sum
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        let mut entries = Vec::new();

        for k in 0..self.n {
            let k1 = k.saturating_sub(self.ml);
            let k2 = (k + self.mu + 1).min(self.n);

            // Diagonal: dF_k/dx_k = 2 + 15*x_k^2
            entries.push((k, k, 2.0 + 15.0 * x[k].powi(2)));

            // Off-diagonal: dF_k/dx_j = -(1 + 2*x_j) for j in band, j != k
            for (j, &xj) in x.iter().enumerate().take(k2).skip(k1) {
                if j != k {
                    entries.push((k, j, -(1.0 + 2.0 * xj)));
                }
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![-factor; self.n]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl ConfigurableProblem for BroydenBanded {
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
    fn test_broyden_banded_jacobian_structure() {
        let problem = BroydenBanded::new(10);
        let x = problem.initial_point(1.0);
        let jac = problem.jacobian(&x);

        // Jacobian should be banded with lower bandwidth 5 and upper bandwidth 1
        for (row, col, _val) in &jac {
            let row = *row as isize;
            let col = *col as isize;
            assert!(
                col >= row - 5 && col <= row + 1,
                "Out of band entry at ({}, {})",
                row,
                col
            );
        }
    }
}
