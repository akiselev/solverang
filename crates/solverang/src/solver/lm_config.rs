//! Levenberg-Marquardt solver configuration.

/// Configuration for the Levenberg-Marquardt solver.
///
/// The LM algorithm minimizes sum of squared residuals: min ||F(x)||^2
/// by iteratively solving a regularized linear least-squares subproblem.
///
/// # Configuration Parameters
///
/// - `ftol`: Relative error desired in the sum of squares (function tolerance)
/// - `xtol`: Relative error between consecutive parameter estimates
/// - `gtol`: Orthogonality tolerance between residuals and Jacobian columns
/// - `stepbound`: Initial step bound factor
/// - `patience`: Factor for maximum function evaluations (max_fev = patience * (n + 1))
/// - `scale_diag`: Whether to rescale variables internally
#[derive(Clone, Debug)]
pub struct LMConfig {
    /// Relative error desired in the sum of squares.
    ///
    /// Termination occurs when both actual and predicted relative reductions
    /// in the sum of squares are at most `ftol`.
    ///
    /// Default: 1e-12
    pub ftol: f64,

    /// Relative error between last two approximations.
    ///
    /// Termination occurs when the relative error between two consecutive
    /// parameter estimates is at most `xtol`.
    ///
    /// Default: 1e-12
    pub xtol: f64,

    /// Orthogonality desired between residual vector and Jacobian columns.
    ///
    /// Termination occurs when the cosine of the angle between the residual
    /// vector and any column of the Jacobian is at most `gtol` in absolute value.
    ///
    /// Default: 1e-12
    pub gtol: f64,

    /// Factor for initial step bound.
    ///
    /// The initial step bound is set to `stepbound * ||D*x||` if nonzero,
    /// or to `stepbound` itself. Should lie in [0.1, 100].
    ///
    /// Default: 100.0
    pub stepbound: f64,

    /// Factor for maximum function evaluations.
    ///
    /// The maximum number of function evaluations is `patience * (n + 1)`.
    ///
    /// Default: 200
    pub patience: usize,

    /// Enable variable rescaling for better conditioning.
    ///
    /// When enabled, the solver internally rescales variables based on
    /// the Jacobian column norms, which can improve convergence.
    ///
    /// Default: true
    pub scale_diag: bool,
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            ftol: 1e-12,
            xtol: 1e-12,
            gtol: 1e-12,
            stepbound: 100.0,
            patience: 200,
            scale_diag: true,
        }
    }
}

impl LMConfig {
    /// Create configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a robust configuration for difficult problems.
    ///
    /// Uses looser tolerances and more iterations to handle problems
    /// with poor conditioning or bad starting points.
    pub fn robust() -> Self {
        Self {
            ftol: 1e-10,
            xtol: 1e-10,
            gtol: 1e-10,
            stepbound: 100.0,
            patience: 500,
            scale_diag: true,
        }
    }

    /// Create a fast configuration for well-behaved problems.
    ///
    /// Uses moderate tolerances and fewer iterations for quick convergence
    /// on problems known to be well-conditioned.
    pub fn fast() -> Self {
        Self {
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-8,
            stepbound: 100.0,
            patience: 100,
            scale_diag: true,
        }
    }

    /// Create a high-precision configuration.
    ///
    /// Uses very tight tolerances for applications requiring maximum accuracy.
    pub fn precise() -> Self {
        Self {
            ftol: 1e-14,
            xtol: 1e-14,
            gtol: 1e-14,
            stepbound: 100.0,
            patience: 300,
            scale_diag: true,
        }
    }

    /// Set the function tolerance (ftol).
    #[must_use]
    pub fn with_ftol(mut self, ftol: f64) -> Self {
        self.ftol = ftol;
        self
    }

    /// Set the parameter tolerance (xtol).
    #[must_use]
    pub fn with_xtol(mut self, xtol: f64) -> Self {
        self.xtol = xtol;
        self
    }

    /// Set the gradient tolerance (gtol).
    #[must_use]
    pub fn with_gtol(mut self, gtol: f64) -> Self {
        self.gtol = gtol;
        self
    }

    /// Set all tolerances to the same value.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.ftol = tol;
        self.xtol = tol;
        self.gtol = tol;
        self
    }

    /// Set the initial step bound factor.
    #[must_use]
    pub fn with_stepbound(mut self, stepbound: f64) -> Self {
        self.stepbound = stepbound;
        self
    }

    /// Set the patience factor for maximum evaluations.
    #[must_use]
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Enable or disable variable rescaling.
    #[must_use]
    pub fn with_scale_diag(mut self, scale_diag: bool) -> Self {
        self.scale_diag = scale_diag;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LMConfig::default();
        assert!((config.ftol - 1e-12).abs() < 1e-15);
        assert!((config.xtol - 1e-12).abs() < 1e-15);
        assert!((config.gtol - 1e-12).abs() < 1e-15);
        assert!((config.stepbound - 100.0).abs() < 1e-10);
        assert_eq!(config.patience, 200);
        assert!(config.scale_diag);
    }

    #[test]
    fn test_robust_config() {
        let config = LMConfig::robust();
        assert!(config.ftol > 1e-12);
        assert_eq!(config.patience, 500);
    }

    #[test]
    fn test_fast_config() {
        let config = LMConfig::fast();
        assert!(config.ftol > 1e-10);
        assert_eq!(config.patience, 100);
    }

    #[test]
    fn test_precise_config() {
        let config = LMConfig::precise();
        assert!(config.ftol < 1e-13);
        assert_eq!(config.patience, 300);
    }

    #[test]
    fn test_builder_methods() {
        let config = LMConfig::new()
            .with_ftol(1e-6)
            .with_xtol(1e-7)
            .with_gtol(1e-8)
            .with_stepbound(50.0)
            .with_patience(150)
            .with_scale_diag(false);

        assert!((config.ftol - 1e-6).abs() < 1e-15);
        assert!((config.xtol - 1e-7).abs() < 1e-15);
        assert!((config.gtol - 1e-8).abs() < 1e-15);
        assert!((config.stepbound - 50.0).abs() < 1e-10);
        assert_eq!(config.patience, 150);
        assert!(!config.scale_diag);
    }

    #[test]
    fn test_with_tol() {
        let config = LMConfig::new().with_tol(1e-5);
        assert!((config.ftol - 1e-5).abs() < 1e-15);
        assert!((config.xtol - 1e-5).abs() < 1e-15);
        assert!((config.gtol - 1e-5).abs() < 1e-15);
    }
}
