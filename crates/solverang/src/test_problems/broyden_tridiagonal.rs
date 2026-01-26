//! Broyden tridiagonal function (HYBRJ Problem 13).
//!
//! A variable-dimension nonlinear equation problem with tridiagonal Jacobian.
//!
//! # Mathematical Definition
//!
//! Residuals (n variables):
//! - F_i(x) = (3 - 2*x_i)*x_i - x_{i-1} - 2*x_{i+1} + 1
//!
//! where x_0 = x_{n+1} = 0
//!
//! Starting point: x_0 = (-1, -1, ..., -1)

use crate::{ConfigurableProblem, Problem};

/// Broyden tridiagonal function problem.
#[derive(Clone, Debug)]
pub struct BroydenTridiagonal {
    n: usize,
}

impl BroydenTridiagonal {
    /// Create a new BroydenTridiagonal problem with the specified dimension.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "BroydenTridiagonal requires n >= 1");
        Self { n }
    }
}

impl Default for BroydenTridiagonal {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Problem for BroydenTridiagonal {
    fn name(&self) -> &str {
        "Broyden Tridiagonal"
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
                let x_prev = if k == 0 { 0.0 } else { x[k - 1] };
                let x_next = if k == self.n - 1 { 0.0 } else { x[k + 1] };

                (3.0 - 2.0 * x[k]) * x[k] - x_prev - 2.0 * x_next + 1.0
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        let mut entries = Vec::new();

        for (k, &xk) in x.iter().enumerate().take(self.n) {
            // Diagonal: dF_k/dx_k = 3 - 4*x_k
            entries.push((k, k, 3.0 - 4.0 * xk));

            // Sub-diagonal: dF_k/dx_{k-1} = -1
            if k > 0 {
                entries.push((k, k - 1, -1.0));
            }

            // Super-diagonal: dF_k/dx_{k+1} = -2
            if k < self.n - 1 {
                entries.push((k, k + 1, -2.0));
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

impl ConfigurableProblem for BroydenTridiagonal {
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
    fn test_broyden_tridiagonal_jacobian_structure() {
        let problem = BroydenTridiagonal::new(5);
        let x = problem.initial_point(1.0);
        let jac = problem.jacobian(&x);

        // Jacobian should be tridiagonal
        for (row, col, _val) in &jac {
            assert!(
                (*col as isize - *row as isize).abs() <= 1,
                "Non-tridiagonal entry at ({}, {})",
                row,
                col
            );
        }
    }
}
