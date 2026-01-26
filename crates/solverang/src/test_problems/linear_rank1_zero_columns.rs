//! Linear function - rank 1 with zero columns and rows (MGH Problem 3).
//!
//! A linear least-squares problem with rank deficiency and zero columns/rows.
//!
//! # Mathematical Definition
//!
//! Similar to Linear Rank 1, but with modifications:
//! - F_1(x) = -1  (first row is zero in Jacobian)
//! - F_i(x) = (i-1) * sum_{j=2}^{n-1} j*x_j - 1,  for i = 2, ..., m-1
//! - F_m(x) = -1  (last row is zero in Jacobian)
//!
//! The first and last columns of the Jacobian are also zero.
//!
//! Starting point: x_0 = (1, 1, ..., 1)

use crate::{ConfigurableProblem, Problem};

/// Linear function - rank 1 with zero columns and rows problem.
#[derive(Clone, Debug)]
pub struct LinearRank1ZeroColumns {
    n: usize,
    m: usize,
}

impl LinearRank1ZeroColumns {
    /// Create a new LinearRank1ZeroColumns problem with specified dimensions.
    pub fn new(n: usize, m: usize) -> Self {
        assert!(m >= n && n >= 2, "LinearRank1ZeroColumns requires m >= n >= 2");
        Self { n, m }
    }
}

impl Default for LinearRank1ZeroColumns {
    fn default() -> Self {
        Self::new(5, 10)
    }
}

impl Problem for LinearRank1ZeroColumns {
    fn name(&self) -> &str {
        "Linear Rank 1 Zero Columns"
    }

    fn residual_count(&self) -> usize {
        self.m
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        // Compute sum_{j=2}^{n-1} j*x_j  (excluding first and last variables)
        let sum: f64 = x.iter()
            .enumerate()
            .skip(1)
            .take(self.n - 2)
            .map(|(j, &xj)| ((j + 1) as f64) * xj)
            .sum();

        let mut fvec = Vec::with_capacity(self.m);

        // First equation: F_1 = -1
        fvec.push(-1.0);

        // Middle equations: F_i = (i-1) * sum - 1
        for i in 2..self.m {
            fvec.push(((i - 1) as f64) * sum - 1.0);
        }

        // Last equation: F_m = -1
        fvec.push(-1.0);

        fvec
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::with_capacity(self.m * self.n);

        // First row: all zeros
        for j in 0..self.n {
            entries.push((0, j, 0.0));
        }

        // Middle rows
        for i in 2..self.m {
            // First column: zero
            entries.push((i - 1, 0, 0.0));

            // Middle columns: (i-1) * j
            for j in 2..self.n {
                entries.push((i - 1, j - 1, ((i - 1) * j) as f64));
            }

            // Last column: zero
            entries.push((i - 1, self.n - 1, 0.0));
        }

        // Last row: all zeros
        for j in 0..self.n {
            entries.push((self.m - 1, j, 0.0));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor; self.n]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // The problem is rank-deficient
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        // Depends on m
        None
    }
}

impl ConfigurableProblem for LinearRank1ZeroColumns {
    fn with_dimensions(n: usize, m: usize) -> Option<Self> {
        if m >= n && n >= 2 {
            Some(Self::new(n, m))
        } else {
            None
        }
    }

    fn min_variables() -> usize {
        2
    }

    fn max_variables() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_rank1_zero_columns_dimensions() {
        let problem = LinearRank1ZeroColumns::new(5, 10);
        assert_eq!(problem.residual_count(), 10);
        assert_eq!(problem.variable_count(), 5);
    }

    #[test]
    fn test_linear_rank1_zero_columns_jacobian() {
        let problem = LinearRank1ZeroColumns::new(4, 6);
        let x = vec![1.0; 4];
        let jac = problem.jacobian(&x);

        let mut dense = vec![vec![0.0; 4]; 6];
        for (row, col, val) in jac {
            dense[row][col] = val;
        }

        // First row should be all zeros
        assert!(dense[0].iter().all(|&v| v == 0.0));

        // Last row should be all zeros
        assert!(dense[5].iter().all(|&v| v == 0.0));

        // First and last columns should be zero
        for row in &dense {
            assert_eq!(row[0], 0.0);
            assert_eq!(row[3], 0.0);
        }
    }
}
