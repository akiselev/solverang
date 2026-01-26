//! Discrete boundary value function (HYBRJ Problem 9).
//!
//! A variable-dimension discretized boundary value problem.
//!
//! # Mathematical Definition
//!
//! Discretization of: -u'' + (u + t + 1)^3 = 0 on [0, 1] with u(0) = u(1) = 0
//!
//! With h = 1/(n+1) and t_i = i*h:
//!
//! Residuals (n variables):
//! - F_i(x) = 2*x_i - x_{i-1} - x_{i+1} + h^2*(x_i + t_i + 1)^3/2
//!
//! where x_0 = x_{n+1} = 0 (boundary conditions)
//!
//! Starting point: x_0_i = t_i*(t_i - 1)

use crate::{ConfigurableProblem, Problem};

/// Discrete boundary value function problem.
#[derive(Clone, Debug)]
pub struct DiscreteBoundaryValue {
    n: usize,
}

impl DiscreteBoundaryValue {
    /// Create a new DiscreteBoundaryValue problem with the specified dimension.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "DiscreteBoundaryValue requires n >= 1");
        Self { n }
    }
}

impl Default for DiscreteBoundaryValue {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Problem for DiscreteBoundaryValue {
    fn name(&self) -> &str {
        "Discrete Boundary Value"
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
        let h_sq_half = h * h / 2.0;

        (0..self.n)
            .map(|k| {
                let t_k = ((k + 1) as f64) * h;

                let x_prev = if k == 0 { 0.0 } else { x[k - 1] };
                let x_next = if k == self.n - 1 { 0.0 } else { x[k + 1] };

                let cubic = (x[k] + t_k + 1.0).powi(3);

                2.0 * x[k] - x_prev - x_next + h_sq_half * cubic
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        let h = 1.0 / ((self.n + 1) as f64);
        let h_sq_half = h * h / 2.0;

        let mut entries = Vec::new();

        for (k, &xk) in x.iter().enumerate().take(self.n) {
            let t_k = ((k + 1) as f64) * h;

            // Diagonal: dF_k/dx_k = 2 + 3h^2/2 * (x_k + t_k + 1)^2
            let diag = 2.0 + 3.0 * h_sq_half * (xk + t_k + 1.0).powi(2);
            entries.push((k, k, diag));

            // Sub-diagonal: dF_k/dx_{k-1} = -1
            if k > 0 {
                entries.push((k, k - 1, -1.0));
            }

            // Super-diagonal: dF_k/dx_{k+1} = -1
            if k < self.n - 1 {
                entries.push((k, k + 1, -1.0));
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
        // Solution is problem-specific and not easily expressible
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl ConfigurableProblem for DiscreteBoundaryValue {
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
    fn test_discrete_boundary_value_dimensions() {
        let problem = DiscreteBoundaryValue::new(10);
        assert_eq!(problem.residual_count(), 10);
        assert_eq!(problem.variable_count(), 10);
    }

    #[test]
    fn test_discrete_boundary_value_jacobian_tridiagonal() {
        let problem = DiscreteBoundaryValue::new(5);
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
