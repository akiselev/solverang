//! Watson function (MGH Problem 11, HYBRJ Problem 6).
//!
//! A variable-dimension problem with n variables and 31 residual equations.
//! The number of variables n can range from 2 to 31.
//!
//! # Mathematical Definition
//!
//! For i = 1, ..., 29, let t_i = i/29
//!
//! Residuals (m=31, n=2 to 31):
//! - F_i(x) = sum_{j=2}^n (j-1)*x_j*t_i^{j-2} - (sum_{j=1}^n x_j*t_i^{j-1})^2 - 1, i=1..29
//! - F_30(x) = x_1
//! - F_31(x) = x_2 - x_1^2 - 1
//!
//! Starting point: x_0 = (0, 0, ..., 0)

use crate::{ConfigurableProblem, Problem};

/// Watson function problem.
#[derive(Clone, Debug)]
pub struct Watson {
    n: usize,
}

impl Watson {
    /// Create a new Watson problem with the specified number of variables.
    ///
    /// n must be between 2 and 31 (inclusive).
    pub fn new(n: usize) -> Self {
        assert!((2..=31).contains(&n), "Watson problem requires 2 <= n <= 31");
        Self { n }
    }
}

impl Default for Watson {
    fn default() -> Self {
        Self::new(6)
    }
}

impl Problem for Watson {
    fn name(&self) -> &str {
        "Watson"
    }

    fn residual_count(&self) -> usize {
        31
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);

        let mut fvec = vec![0.0; 31];

        // First 29 equations
        for i in 1..=29 {
            let ti = (i as f64) / 29.0;

            // sum1 = sum_{j=2}^n (j-1)*x_j*t_i^{j-2}
            let mut sum1 = 0.0;
            let mut temp = 1.0;
            for j in 2..=self.n {
                sum1 += ((j - 1) as f64) * temp * x[j - 1];
                temp *= ti;
            }

            // sum2 = sum_{j=1}^n x_j*t_i^{j-1}
            let mut sum2 = 0.0;
            temp = 1.0;
            for j in 1..=self.n {
                sum2 += temp * x[j - 1];
                temp *= ti;
            }

            fvec[i - 1] = sum1 - sum2 * sum2 - 1.0;
        }

        // Last two equations
        fvec[29] = x[0];
        fvec[30] = x[1] - x[0] * x[0] - 1.0;

        fvec
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), self.n);

        let mut entries = Vec::new();

        // First 29 equations
        for i in 1..=29 {
            let ti = (i as f64) / 29.0;

            // Compute sum2 for this i
            let mut sum2 = 0.0;
            let mut temp = 1.0;
            for j in 1..=self.n {
                sum2 += temp * x[j - 1];
                temp *= ti;
            }

            // Jacobian entries for equation i-1
            let mut temp = 1.0;
            for k in 1..=self.n {
                let k_f = k as f64;
                // dF_i/dx_k = (k-1)*t_i^{k-2} - 2*sum2*t_i^{k-1}
                let dk = if k >= 2 {
                    (k_f - 1.0) * temp / ti
                } else {
                    0.0
                };
                let jac_val = dk - 2.0 * sum2 * temp;
                entries.push((i - 1, k - 1, jac_val));
                temp *= ti;
            }
        }

        // Equation 30: F_30 = x_1
        entries.push((29, 0, 1.0));
        for k in 1..self.n {
            entries.push((29, k, 0.0));
        }

        // Equation 31: F_31 = x_2 - x_1^2 - 1
        entries.push((30, 0, -2.0 * x[0]));
        entries.push((30, 1, 1.0));
        for k in 2..self.n {
            entries.push((30, k, 0.0));
        }

        entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        // Watson starting point is always zero regardless of factor
        vec![0.0; self.n]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // Solution depends on n, and for most n values the solution is not simple
        None
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        // Depends on n
        None
    }
}

impl ConfigurableProblem for Watson {
    fn with_dimensions(n: usize, _m: usize) -> Option<Self> {
        if (2..=31).contains(&n) {
            Some(Self::new(n))
        } else {
            None
        }
    }

    fn min_variables() -> usize {
        2
    }

    fn max_variables() -> Option<usize> {
        Some(31)
    }

    fn equations_fixed_by_variables() -> bool {
        false // m is always 31 regardless of n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watson_dimensions() {
        let problem = Watson::new(6);
        assert_eq!(problem.residual_count(), 31);
        assert_eq!(problem.variable_count(), 6);
    }

    #[test]
    fn test_watson_initial_point() {
        let problem = Watson::new(6);
        let x0 = problem.initial_point(1.0);
        assert_eq!(x0.len(), 6);
        assert!(x0.iter().all(|&v| v == 0.0));
    }
}
