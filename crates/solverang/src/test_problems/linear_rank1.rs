//! Linear function - rank 1 (MGH Problem 2).
//!
//! A linear least-squares problem with rank deficiency (rank 1).
//!
//! # Mathematical Definition
//!
//! Residuals (m >= n):
//! - F_i(x) = i * sum_{j=1}^n j*x_j - 1,  for i = 1, ..., m
//!
//! The Jacobian has rank 1 (all rows are multiples of [1, 2, 3, ..., n]).
//!
//! Starting point: x_0 = (1, 1, ..., 1)

use crate::{ConfigurableProblem, Problem};

/// Linear function - rank 1 problem.
#[derive(Clone, Debug)]
pub struct LinearRank1 {
    n: usize,
    m: usize,
}

impl LinearRank1 {
    /// Create a new LinearRank1 problem with specified dimensions.
    pub fn new(n: usize, m: usize) -> Self {
        assert!(m >= n, "LinearRank1 requires m >= n");
        Self { n, m }
    }
}

impl Default for LinearRank1 {
    fn default() -> Self {
        Self::new(5, 10)
    }
}

impl Problem for LinearRank1 {
    fn name(&self) -> &str {
        "Linear Rank 1"
    }

    fn residual_count(&self) -> usize {
        self.m
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        // Compute sum_{j=1}^n j*x_j
        let sum: f64 = x
            .iter()
            .enumerate()
            .map(|(j, &xj)| ((j + 1) as f64) * xj)
            .sum();

        (1..=self.m).map(|i| (i as f64) * sum - 1.0).collect()
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::with_capacity(self.m * self.n);

        for i in 1..=self.m {
            for j in 1..=self.n {
                entries.push((i - 1, j - 1, (i * j) as f64));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor; self.n]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // The problem is rank-deficient, so there are infinitely many solutions
        // Any x satisfying sum_j j*x_j = 3/(m*(m+1)) works
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        // Optimal residual norm depends on m
        let m_f = self.m as f64;
        Some(((m_f * (m_f - 1.0)) / (2.0 * (2.0 * m_f + 1.0))).sqrt())
    }
}

impl ConfigurableProblem for LinearRank1 {
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
    fn test_linear_rank1_dimensions() {
        let problem = LinearRank1::new(5, 10);
        assert_eq!(problem.residual_count(), 10);
        assert_eq!(problem.variable_count(), 5);
    }

    #[test]
    fn test_linear_rank1_jacobian_rank() {
        let problem = LinearRank1::new(3, 5);
        let x = vec![1.0, 1.0, 1.0];
        let jac = problem.jacobian(&x);

        // All rows should be proportional to [1, 2, 3]
        let mut dense = vec![vec![0.0; 3]; 5];
        for (row, col, val) in jac {
            dense[row][col] = val;
        }

        // Check that row i is i * [1, 2, 3]
        for (i, row) in dense.iter().enumerate() {
            let expected_ratio = (i + 1) as f64;
            assert!((row[0] / expected_ratio - 1.0).abs() < 1e-10);
            assert!((row[1] / expected_ratio - 2.0).abs() < 1e-10);
            assert!((row[2] / expected_ratio - 3.0).abs() < 1e-10);
        }
    }
}
