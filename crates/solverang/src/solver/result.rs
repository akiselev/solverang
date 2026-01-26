//! Solver result types.

use thiserror::Error;

/// Result of solving a nonlinear problem.
#[derive(Clone, Debug)]
pub enum SolveResult {
    /// Solver converged to a solution.
    Converged {
        /// Final solution vector.
        solution: Vec<f64>,
        /// Number of iterations performed.
        iterations: usize,
        /// Final residual norm ||F(x)||.
        residual_norm: f64,
    },

    /// Solver did not converge within max iterations.
    NotConverged {
        /// Best solution found.
        solution: Vec<f64>,
        /// Number of iterations performed.
        iterations: usize,
        /// Final residual norm ||F(x)||.
        residual_norm: f64,
    },

    /// Solver failed due to a fatal error.
    Failed {
        /// Error describing the failure.
        error: SolveError,
    },
}

impl SolveResult {
    /// Check if the solve converged.
    pub fn is_converged(&self) -> bool {
        matches!(self, SolveResult::Converged { .. })
    }

    /// Check if the solve completed (converged or not converged, but not failed).
    pub fn is_completed(&self) -> bool {
        !matches!(self, SolveResult::Failed { .. })
    }

    /// Get the solution if available.
    pub fn solution(&self) -> Option<&[f64]> {
        match self {
            SolveResult::Converged { solution, .. } => Some(solution),
            SolveResult::NotConverged { solution, .. } => Some(solution),
            SolveResult::Failed { .. } => None,
        }
    }

    /// Get the residual norm if available.
    pub fn residual_norm(&self) -> Option<f64> {
        match self {
            SolveResult::Converged { residual_norm, .. } => Some(*residual_norm),
            SolveResult::NotConverged { residual_norm, .. } => Some(*residual_norm),
            SolveResult::Failed { .. } => None,
        }
    }

    /// Get the iteration count if available.
    pub fn iterations(&self) -> Option<usize> {
        match self {
            SolveResult::Converged { iterations, .. } => Some(*iterations),
            SolveResult::NotConverged { iterations, .. } => Some(*iterations),
            SolveResult::Failed { .. } => None,
        }
    }

    /// Get the error if the solve failed.
    pub fn error(&self) -> Option<&SolveError> {
        match self {
            SolveResult::Failed { error } => Some(error),
            _ => None,
        }
    }
}

/// Errors that can occur during solving.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SolveError {
    /// The Jacobian matrix is singular and cannot be inverted.
    #[error("Jacobian matrix is singular")]
    SingularJacobian,

    /// The problem has no equations to solve.
    #[error("problem has no equations (m = 0)")]
    NoEquations,

    /// The problem has no variables.
    #[error("problem has no variables (n = 0)")]
    NoVariables,

    /// Dimension mismatch between initial point and problem.
    #[error("dimension mismatch: initial point has {got} elements, expected {expected}")]
    DimensionMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        got: usize,
    },

    /// Line search failed to find a suitable step size.
    #[error("line search failed to find acceptable step")]
    LineSearchFailed,

    /// Maximum iterations exceeded without convergence.
    #[error("maximum iterations ({0}) exceeded")]
    MaxIterationsExceeded(usize),

    /// Residuals contain NaN or infinity.
    #[error("residuals contain non-finite values (NaN or infinity)")]
    NonFiniteResiduals,

    /// Jacobian contains NaN or infinity.
    #[error("Jacobian contains non-finite values (NaN or infinity)")]
    NonFiniteJacobian,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converged_result() {
        let result = SolveResult::Converged {
            solution: vec![1.0, 2.0],
            iterations: 5,
            residual_norm: 1e-10,
        };

        assert!(result.is_converged());
        assert!(result.is_completed());
        assert_eq!(result.solution(), Some(&[1.0, 2.0][..]));
        assert!((result.residual_norm().unwrap() - 1e-10).abs() < 1e-15);
        assert_eq!(result.iterations(), Some(5));
        assert!(result.error().is_none());
    }

    #[test]
    fn test_not_converged_result() {
        let result = SolveResult::NotConverged {
            solution: vec![1.0, 2.0],
            iterations: 100,
            residual_norm: 0.1,
        };

        assert!(!result.is_converged());
        assert!(result.is_completed());
        assert_eq!(result.solution(), Some(&[1.0, 2.0][..]));
        assert_eq!(result.iterations(), Some(100));
    }

    #[test]
    fn test_failed_result() {
        let result = SolveResult::Failed {
            error: SolveError::SingularJacobian,
        };

        assert!(!result.is_converged());
        assert!(!result.is_completed());
        assert!(result.solution().is_none());
        assert!(result.residual_norm().is_none());
        assert!(result.iterations().is_none());
        assert_eq!(result.error(), Some(&SolveError::SingularJacobian));
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            SolveError::SingularJacobian.to_string(),
            "Jacobian matrix is singular"
        );
        assert_eq!(
            SolveError::NoEquations.to_string(),
            "problem has no equations (m = 0)"
        );
        assert_eq!(
            SolveError::DimensionMismatch {
                expected: 5,
                got: 3
            }
            .to_string(),
            "dimension mismatch: initial point has 3 elements, expected 5"
        );
    }
}
