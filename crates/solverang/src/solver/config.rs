//! Solver configuration.

/// Configuration for the nonlinear solver.
#[derive(Clone, Debug)]
pub struct SolverConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,

    /// Convergence tolerance for residual norm.
    ///
    /// The solver considers the problem converged when ||F(x)|| < tolerance.
    pub tolerance: f64,

    /// Enable backtracking line search for stability.
    ///
    /// When enabled, the solver performs Armijo backtracking line search
    /// to ensure sufficient decrease in the residual norm.
    pub line_search: bool,

    /// Armijo parameter for line search (c in Armijo condition).
    ///
    /// The line search accepts a step if f(x + alpha*d) <= f(x) - c*alpha*||d||.
    /// Typical values are 1e-4 to 1e-1.
    pub armijo_c: f64,

    /// Backtracking factor for line search.
    ///
    /// When the Armijo condition is not satisfied, alpha is multiplied by this factor.
    /// Typical values are 0.5 to 0.8.
    pub backtrack_factor: f64,

    /// Maximum number of line search iterations.
    pub max_line_search_iterations: usize,

    /// Minimum step size for line search.
    ///
    /// If alpha falls below this threshold, line search terminates.
    pub min_step_size: f64,

    /// Tolerance for singular value decomposition.
    ///
    /// Singular values below this threshold (relative to the largest) are treated as zero.
    pub svd_tolerance: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-8,
            line_search: true,
            armijo_c: 1e-4,
            backtrack_factor: 0.5,
            max_line_search_iterations: 20,
            min_step_size: 1e-12,
            svd_tolerance: 1e-10,
        }
    }
}

impl SolverConfig {
    /// Create a configuration optimized for fast convergence on well-behaved problems.
    pub fn fast() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            line_search: false,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for robustness on difficult problems.
    pub fn robust() -> Self {
        Self {
            max_iterations: 500,
            tolerance: 1e-10,
            line_search: true,
            armijo_c: 1e-4,
            backtrack_factor: 0.5,
            max_line_search_iterations: 30,
            min_step_size: 1e-14,
            svd_tolerance: 1e-12,
        }
    }

    /// Create a configuration for loose convergence (faster but less accurate).
    pub fn loose() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-4,
            line_search: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SolverConfig::default();
        assert_eq!(config.max_iterations, 200);
        assert!((config.tolerance - 1e-8).abs() < 1e-15);
        assert!(config.line_search);
    }

    #[test]
    fn test_fast_config() {
        let config = SolverConfig::fast();
        assert_eq!(config.max_iterations, 100);
        assert!(!config.line_search);
    }

    #[test]
    fn test_robust_config() {
        let config = SolverConfig::robust();
        assert_eq!(config.max_iterations, 500);
        assert!(config.tolerance < 1e-9);
    }
}
