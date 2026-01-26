//! Chebyquad function (MGH Problem 15, HYBRJ Problem 7).
//!
//! A variable-dimension problem with n variables and m equations (m >= n).
//!
//! # Mathematical Definition
//!
//! Let T_k(x) be the k-th Chebyshev polynomial:
//!   T_0(x) = 1
//!   T_1(x) = x
//!   T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
//!
//! Residuals (m >= n, n=1 to any):
//! - F_i(x) = (1/n)*sum_{j=1}^n T_i(2*x_j - 1) - integral_0^1 T_i(2t-1)dt
//!
//! The integral term is 0 for odd i, and -1/(i^2-1) for even i.
//!
//! Starting point: x_0 = (1/(n+1), 2/(n+1), ..., n/(n+1))

use crate::{ConfigurableProblem, Problem};

/// Chebyquad function problem.
///
/// Supports both square (m = n) and overdetermined (m > n) configurations.
#[derive(Clone, Debug)]
pub struct Chebyquad {
    n: usize,
    m: usize,
}

impl Chebyquad {
    /// Create a new Chebyquad problem with n variables and n equations (square).
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "Chebyquad requires n >= 1");
        Self { n, m: n }
    }

    /// Create a new Chebyquad problem with n variables and m equations.
    ///
    /// Supports overdetermined systems where m > n.
    pub fn with_m(n: usize, m: usize) -> Self {
        assert!(n >= 1, "Chebyquad requires n >= 1");
        assert!(m >= n, "Chebyquad requires m >= n");
        Self { n, m }
    }

    /// Evaluate Chebyshev polynomial T_k at shifted point y = 2x - 1.
    fn chebyshev_at_point(y: f64, k: usize) -> f64 {
        if k == 0 {
            1.0
        } else if k == 1 {
            y
        } else {
            let mut t_prev2 = 1.0;
            let mut t_prev1 = y;
            for _ in 2..=k {
                let t_curr = 2.0 * y * t_prev1 - t_prev2;
                t_prev2 = t_prev1;
                t_prev1 = t_curr;
            }
            t_prev1
        }
    }

    /// Compute the integral of T_i(2t-1) from 0 to 1
    fn integral_term(i: usize) -> f64 {
        if i % 2 == 1 {
            0.0
        } else {
            -1.0 / ((i as f64).powi(2) - 1.0)
        }
    }
}

impl Default for Chebyquad {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Problem for Chebyquad {
    fn name(&self) -> &str {
        "Chebyquad"
    }

    fn residual_count(&self) -> usize {
        self.m
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        let n_f = self.n as f64;

        // Compute m residuals (may be more than n for overdetermined case)
        (1..=self.m)
            .map(|i| {
                let sum: f64 = x
                    .iter()
                    .map(|&xj| Self::chebyshev_at_point(2.0 * xj - 1.0, i))
                    .sum();
                sum / n_f - Self::integral_term(i)
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        let n_f = self.n as f64;
        let mut entries = Vec::with_capacity(self.m * self.n);

        // Compute m rows of Jacobian (may be more than n for overdetermined case)
        for i in 1..=self.m {
            for (j, &xj) in x.iter().enumerate().take(self.n) {
                let y = 2.0 * xj - 1.0;

                // Derivative of T_i(2xj - 1) with respect to xj
                // d/dxj T_i(2xj - 1) = 2 * T'_i(y)
                // T'_i(y) = i * U_{i-1}(y) where U is Chebyshev of second kind
                let deriv = if i == 1 {
                    // T'_1(y) = 1, so d/dx T_1(2x-1) = 2
                    2.0
                } else {
                    // Use Chebyshev derivative identity: T'_n(y) = n * U_{n-1}(y)
                    // where U_k(y) is the Chebyshev polynomial of second kind:
                    //   U_0(y) = 1
                    //   U_1(y) = 2y
                    //   U_k(y) = 2y * U_{k-1}(y) - U_{k-2}(y)
                    //
                    // d/dx T_n(2x-1) = 2 * T'_n(y) = 2 * n * U_{n-1}(y)
                    let i_f = i as f64;
                    let mut u_prev2 = 1.0; // U_0(y)
                    let mut u_prev1 = 2.0 * y; // U_1(y)

                    // For i == 2: need U_1(y) = 2y (which is u_prev1)
                    // For i > 2: iterate to get U_{i-1}(y)
                    if i > 2 {
                        for _ in 2..i {
                            let u_curr = 2.0 * y * u_prev1 - u_prev2;
                            u_prev2 = u_prev1;
                            u_prev1 = u_curr;
                        }
                    }
                    // u_prev1 now contains U_{i-1}(y)
                    2.0 * i_f * u_prev1
                };

                entries.push((i - 1, j, deriv / n_f));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        (1..=self.n)
            .map(|j| (j as f64) / ((self.n + 1) as f64) * factor)
            .collect()
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // Solutions are symmetric and depend on n
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        None
    }
}

impl ConfigurableProblem for Chebyquad {
    fn with_dimensions(n: usize, m: usize) -> Option<Self> {
        if n >= 1 && m >= n {
            Some(Self::with_m(n, m))
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
    fn test_chebyquad_dimensions() {
        let problem = Chebyquad::new(5);
        assert_eq!(problem.residual_count(), 5);
        assert_eq!(problem.variable_count(), 5);
    }

    #[test]
    fn test_chebyquad_overdetermined() {
        // Test overdetermined case (m > n)
        let problem = Chebyquad::with_m(1, 8);
        assert_eq!(problem.variable_count(), 1);
        assert_eq!(problem.residual_count(), 8);

        let x = problem.initial_point(1.0);
        assert_eq!(x.len(), 1);

        let residuals = problem.residuals(&x);
        assert_eq!(residuals.len(), 8);

        let jacobian = problem.jacobian(&x);
        // Should have 8 rows x 1 column = 8 entries
        assert_eq!(jacobian.len(), 8);
    }

    #[test]
    fn test_chebyshev_polynomials() {
        // T_0(x) = 1
        assert!((Chebyquad::chebyshev_at_point(0.5, 0) - 1.0).abs() < 1e-10);

        // T_1(x) = x
        assert!((Chebyquad::chebyshev_at_point(0.5, 1) - 0.5).abs() < 1e-10);

        // T_2(x) = 2x^2 - 1
        let y = 0.5;
        assert!((Chebyquad::chebyshev_at_point(y, 2) - (2.0 * y * y - 1.0)).abs() < 1e-10);
    }
}
