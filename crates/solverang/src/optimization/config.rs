//! Configuration types for optimization solvers.

/// Which optimization algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationAlgorithm {
    /// Automatically select based on problem structure.
    /// Unconstrained with bounds → BfgsB, unconstrained → BFGS, equality-constrained → ALM.
    Auto,
    /// L-BFGS for unconstrained optimization (gradient-only).
    Bfgs,
    /// L-BFGS-B for box-constrained optimization (projected gradient).
    BfgsB,
    /// Augmented Lagrangian Method for constrained optimization.
    /// Uses existing NR/LM as inner solver.
    Alm,
    /// Trust-region method with dogleg (n < threshold) or Steihaug-CG (n >= threshold).
    TrustRegion,
}

/// Strategy for initializing Lagrange multipliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiplierInitStrategy {
    /// Initialize all multipliers to zero (simplest, always works).
    Zero,
    /// Warm-start from previous solve's multipliers.
    WarmStart,
}

/// Configuration for optimization solvers.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Which algorithm to use.
    pub algorithm: OptimizationAlgorithm,
    /// Maximum outer iterations (ALM outer loop, or BFGS total iterations).
    pub max_outer_iterations: usize,
    /// Maximum inner iterations (ALM inner NR/LM solve).
    pub max_inner_iterations: usize,
    /// Outer tolerance: primal feasibility `||g(x)|| < tol`.
    pub outer_tolerance: f64,
    /// Inner tolerance: inner solver convergence criterion.
    pub inner_tolerance: f64,
    /// Dual feasibility tolerance: `||∇_x L|| < tol`.
    pub dual_tolerance: f64,
    /// Initial penalty parameter ρ for ALM.
    pub rho_init: f64,
    /// Penalty growth factor: `ρ_{k+1} = min(ρ_k * growth, ρ_max)`.
    pub rho_growth: f64,
    /// Maximum penalty parameter.
    pub rho_max: f64,
    /// Maximum absolute value for multipliers (divergence guard).
    pub max_multiplier: f64,
    /// Strategy for initializing multipliers.
    pub multiplier_init: MultiplierInitStrategy,
    /// L-BFGS memory size (number of past gradient pairs to store).
    pub lbfgs_memory: usize,
    /// Armijo line search sufficient decrease parameter c₁.
    pub armijo_c1: f64,
    /// Wolfe curvature condition parameter c₂ (strong curvature): `|∇f(x+αd)·d| ≤ c₂|∇f(x)·d|`.
    pub wolfe_c2: f64,
    /// Line search backtracking factor.
    pub line_search_backtrack: f64,
    /// Minimum line search step size before declaring failure.
    pub line_search_min_step: f64,
    /// Use relative tolerances for convergence checks.
    ///
    /// When `true`, BFGS scales the gradient norm by `max(1.0, |f|)` and ALM
    /// scales norms by the square root of the problem dimension, making
    /// convergence criteria independent of problem size and objective magnitude.
    /// When `false`, absolute tolerances are used (backward-compatible behavior).
    pub relative_tolerance: bool,
    /// Initial trust-region radius.
    pub trust_region_init: f64,
    /// Maximum trust-region radius.
    pub trust_region_max: f64,
    /// Dimension threshold: dogleg for n < threshold, Steihaug-CG for n >= threshold.
    pub tr_subproblem_threshold: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            algorithm: OptimizationAlgorithm::Auto,
            max_outer_iterations: 100,
            max_inner_iterations: 200,
            outer_tolerance: 1e-6,
            inner_tolerance: 1e-8,
            dual_tolerance: 1e-6,
            rho_init: 1.0,
            rho_growth: 10.0,
            rho_max: 1e6,
            max_multiplier: 1e8,
            multiplier_init: MultiplierInitStrategy::Zero,
            lbfgs_memory: 10,
            armijo_c1: 1e-4,
            wolfe_c2: 0.9,
            line_search_backtrack: 0.5,
            line_search_min_step: 1e-12,
            relative_tolerance: true,
            trust_region_init: 1.0,
            trust_region_max: 100.0,
            tr_subproblem_threshold: 100,
        }
    }
}
