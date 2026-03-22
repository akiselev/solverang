//! Result types for optimization solvers.

use super::multiplier_store::MultiplierStore;

/// KKT residual components for convergence assessment.
#[derive(Debug, Clone)]
pub struct KktResidual {
    /// Primal feasibility: `max(||g(x)||, max(h_j(x), 0))`.
    pub primal: f64,
    /// Dual feasibility: `||∇_x L||`.
    pub dual: f64,
    /// Complementarity: `max |μ_j * h_j(x)|`.
    pub complementarity: f64,
}

impl KktResidual {
    /// Check if all KKT components are within their tolerances.
    pub fn is_within_tolerance(&self, primal_tol: f64, dual_tol: f64, comp_tol: f64) -> bool {
        self.primal < primal_tol && self.dual < dual_tol && self.complementarity < comp_tol
    }
}

/// Status of an optimization solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationStatus {
    /// Solver converged: KKT conditions satisfied within tolerance.
    Converged,
    /// Maximum iterations reached without convergence.
    MaxIterationsReached,
    /// Problem is infeasible (constraints cannot be simultaneously satisfied).
    Infeasible,
    /// Solver diverged (multipliers or objective exploded).
    Diverged,
    /// Optimization not yet implemented (stub).
    NotImplemented,
}

impl OptimizationStatus {
    /// Whether the solver converged successfully.
    pub fn is_converged(&self) -> bool {
        matches!(self, Self::Converged)
    }
}

/// Result of an optimization solve.
#[derive(Debug)]
pub struct OptimizationResult {
    /// Final objective value f(x*).
    pub objective_value: f64,
    /// Solve status.
    pub status: OptimizationStatus,
    /// Total outer iterations (ALM outer loop, or BFGS iterations).
    pub outer_iterations: usize,
    /// Total inner iterations (ALM inner NR/LM solves, summed).
    pub inner_iterations: usize,
    /// Final KKT residual (primal, dual, complementarity).
    pub kkt_residual: KktResidual,
    /// Lagrange multipliers for sensitivity analysis.
    pub multipliers: MultiplierStore,
    /// Per-constraint violation values (positive = violated).
    pub constraint_violations: Vec<f64>,
    /// Wall-clock duration of the solve.
    pub duration: std::time::Duration,
}

impl OptimizationResult {
    /// Create a stub result for the not-yet-implemented case.
    pub(crate) fn not_implemented() -> Self {
        Self {
            objective_value: f64::NAN,
            status: OptimizationStatus::NotImplemented,
            outer_iterations: 0,
            inner_iterations: 0,
            kkt_residual: KktResidual {
                primal: f64::INFINITY,
                dual: f64::INFINITY,
                complementarity: f64::INFINITY,
            },
            multipliers: MultiplierStore::new(),
            constraint_violations: Vec::new(),
            duration: std::time::Duration::ZERO,
        }
    }
}
