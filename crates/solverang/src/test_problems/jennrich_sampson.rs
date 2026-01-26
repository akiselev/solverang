//! Jennrich and Sampson function (MGH Problem 13).
//!
//! A 2-variable data fitting problem with m residual equations (m >= 2).
//!
//! # Mathematical Definition
//!
//! Residuals (m >= 2, n=2):
//! - F_i(x) = 2 + 2i - (exp(i*x_1) + exp(i*x_2))
//!
//! Starting point: x_0 = (0.3, 0.4)

use crate::{ConfigurableProblem, Problem};

/// Jennrich and Sampson function problem.
#[derive(Clone, Debug)]
pub struct JennrichSampson {
    m: usize,
}

impl JennrichSampson {
    /// Create a new JennrichSampson problem with the specified number of equations.
    pub fn new(m: usize) -> Self {
        assert!(m >= 2, "JennrichSampson requires m >= 2");
        Self { m }
    }
}

impl Default for JennrichSampson {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Problem for JennrichSampson {
    fn name(&self) -> &str {
        "Jennrich-Sampson"
    }

    fn residual_count(&self) -> usize {
        self.m
    }

    fn variable_count(&self) -> usize {
        2
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), 2);

        (1..=self.m)
            .map(|i| {
                let i_f = i as f64;
                2.0 + 2.0 * i_f - ((i_f * x[0]).exp() + (i_f * x[1]).exp())
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(x.len(), 2);

        let mut entries = Vec::with_capacity(2 * self.m);

        for i in 1..=self.m {
            let row = i - 1;
            let i_f = i as f64;

            entries.push((row, 0, -i_f * (i_f * x[0]).exp()));
            entries.push((row, 1, -i_f * (i_f * x[1]).exp()));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![0.3 * factor, 0.4 * factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        // For m = 10, the solution is approximately
        if self.m == 10 {
            Some(vec![0.2578252135686162, 0.2578252135686162])
        } else {
            None
        }
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        if self.m == 10 {
            Some(0.1243621823556148E+02_f64.sqrt())
        } else {
            None
        }
    }
}

impl ConfigurableProblem for JennrichSampson {
    fn with_dimensions(_n: usize, m: usize) -> Option<Self> {
        if m >= 2 {
            Some(Self::new(m))
        } else {
            None
        }
    }

    fn min_variables() -> usize {
        2
    }

    fn max_variables() -> Option<usize> {
        Some(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jennrich_sampson_dimensions() {
        let problem = JennrichSampson::default();
        assert_eq!(problem.residual_count(), 10);
        assert_eq!(problem.variable_count(), 2);
    }
}
