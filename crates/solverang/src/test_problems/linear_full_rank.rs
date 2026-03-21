//! Linear function - full rank (MGH Problem 1).
//!
//! A linear least-squares problem with full column rank.
//!
//! # Mathematical Definition
//!
//! Residuals (m >= n):
//! - F_i(x) = x_i - 2*sum_{j=1}^n x_j / m - 1,  for i = 1, ..., n
//! - F_i(x) = -2*sum_{j=1}^n x_j / m - 1,       for i = n+1, ..., m
//!
//! Starting point: x_0 = (1, 1, ..., 1)

use crate::{ConfigurableProblem, Problem};

/// Linear function - full rank problem.
#[derive(Clone, Debug)]
pub struct LinearFullRank {
    n: usize,
    m: usize,
}

impl LinearFullRank {
    /// Create a new LinearFullRank problem with specified dimensions.
    pub fn new(n: usize, m: usize) -> Self {
        assert!(m >= n, "LinearFullRank requires m >= n");
        Self { n, m }
    }
}

impl Default for LinearFullRank {
    fn default() -> Self {
        Self::new(5, 10)
    }
}

impl Problem for LinearFullRank {
    fn name(&self) -> &str {
        "Linear Full Rank"
    }

    fn residual_count(&self) -> usize {
        self.m
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        let sum: f64 = x.iter().sum();
        let coeff = 2.0 * sum / (self.m as f64);

        let mut fvec = Vec::with_capacity(self.m);

        // First n equations
        for &xi in x.iter().take(self.n) {
            fvec.push(xi - coeff - 1.0);
        }

        // Remaining m-n equations
        for _ in self.n..self.m {
            fvec.push(-coeff - 1.0);
        }

        fvec
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        let m_f = self.m as f64;
        let mut entries = Vec::with_capacity(self.m * self.n);

        // First n equations
        for i in 0..self.n {
            for j in 0..self.n {
                let val = if i == j { 1.0 - 2.0 / m_f } else { -2.0 / m_f };
                entries.push((i, j, val));
            }
        }

        // Remaining m-n equations
        for i in self.n..self.m {
            for j in 0..self.n {
                entries.push((i, j, -2.0 / m_f));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor; self.n]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // Solution: x_i = -1 for all i
        Some(vec![-1.0; self.n])
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        // At solution x = [-1, ..., -1]:
        // First n residuals are -1, remaining m-n are 0
        // Norm = sqrt(n * 1^2) = sqrt(n)
        Some((self.n as f64).sqrt())
    }
}

impl ConfigurableProblem for LinearFullRank {
    fn with_dimensions(n: usize, m: usize) -> Option<Self> {
        if m >= n && n >= 1 {
            Some(Self::new(n, m))
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
    fn test_linear_full_rank_at_solution() {
        let problem = LinearFullRank::new(5, 10);
        let solution = problem.known_solution().expect("should have solution");

        let residuals = problem.residuals(&solution);

        // First n residuals should be -1 (NOT zero!)
        for (i, residual) in residuals.iter().enumerate().take(5) {
            assert!(
                (residual - (-1.0)).abs() < 1e-10,
                "residual[{}] = {} (expected -1.0)",
                i,
                residual
            );
        }

        // Remaining m-n residuals should be 0
        for (i, residual) in residuals.iter().enumerate().skip(5) {
            assert!(
                residual.abs() < 1e-10,
                "residual[{}] = {} (expected 0.0)",
                i,
                residual
            );
        }

        // Verify the residual norm matches expected
        let norm = problem.residual_norm(&solution);
        let expected_norm = problem
            .expected_residual_norm()
            .expect("should have expected norm");
        assert!(
            (norm - expected_norm).abs() < 1e-10,
            "norm = {} (expected {})",
            norm,
            expected_norm
        );
    }

    #[test]
    fn test_linear_full_rank_dimensions() {
        let problem = LinearFullRank::new(5, 10);
        assert_eq!(problem.variable_count(), 5);
        assert_eq!(problem.residual_count(), 10);
    }
}
