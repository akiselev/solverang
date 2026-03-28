//! Augmented Lagrangian Method (ALM) for equality-constrained optimization.
//!
//! Solves `min f(x) s.t. g(x) = 0` by converting to a sequence of
//! unconstrained-like subproblems that the existing LM solver handles.
//!
//! The augmented Lagrangian is:
//!   L_A(x, λ, ρ) = f(x) + λ^T g(x) + (ρ/2) ||g(x)||^2
//!
//! The ALM outer loop updates multipliers (λ) and penalty (ρ), while the
//! inner loop minimizes L_A using LMSolver on a least-squares formulation.

use std::time::Instant;

use crate::constraint::Constraint;
use crate::optimization::{
    InequalityFn, KktResidual, MultiplierId, MultiplierInitStrategy, MultiplierStore, Objective,
    OptimizationConfig, OptimizationResult, OptimizationStatus,
};
use crate::param::ParamStore;

fn compute_complementarity(
    inequalities: &[&dyn InequalityFn],
    mu: &[f64],
    store: &ParamStore,
) -> f64 {
    let mut comp = 0.0_f64;
    let mut idx = 0;
    for h in inequalities {
        for v in h.values(store) {
            comp = comp.max((mu[idx] * v).abs());
            idx += 1;
        }
    }
    comp
}

/// Augmented Lagrangian Method solver.
pub struct AlmSolver;

impl AlmSolver {
    /// Solve a constrained optimization problem.
    ///
    /// Minimizes `objective.value(store)` subject to equality constraints
    /// (existing `Constraint` objects where residuals should be zero) and
    /// inequality constraints `h(x) ≤ 0`.
    ///
    /// # Algorithm
    ///
    /// 1. Formulate augmented Lagrangian L_A as a least-squares problem
    /// 2. Solve inner subproblem with LMSolver
    /// 3. Update multipliers: λ_{k+1} = λ_k + ρ * g(x_k)
    /// 4. Optionally increase ρ if constraint violation doesn't decrease
    /// 5. Check KKT convergence (primal + dual feasibility)
    pub fn solve(
        objective: &dyn Objective,
        constraints: &[&dyn Constraint],
        inequalities: &[&dyn InequalityFn],
        store: &mut ParamStore,
        config: &OptimizationConfig,
        warm_start: Option<&MultiplierStore>,
    ) -> OptimizationResult {
        let start = Instant::now();
        let mapping = store.build_solver_mapping();
        let n = mapping.len();

        if n == 0 {
            let f = objective.value(store);
            return OptimizationResult {
                objective_value: f,
                status: OptimizationStatus::Converged,
                outer_iterations: 0,
                inner_iterations: 0,
                kkt_residual: KktResidual {
                    primal: 0.0,
                    dual: 0.0,
                    complementarity: 0.0,
                },
                multipliers: MultiplierStore::new(),
                constraint_violations: Vec::new(),
                duration: start.elapsed(),
            };
        }

        let param_ids = mapping.col_to_param.clone();

        // Count total equality equations
        let total_eq: usize = constraints.iter().map(|c| c.equation_count()).sum();

        // Count inequality equations
        let total_ineq: usize = inequalities.iter().map(|h| h.inequality_count()).sum();

        // Dispatch inner loop to BFGS-B if any free parameter has finite bounds.
        let use_bfgs_b = store.any_free_finite_bounds();

        // Initialize multipliers (warm-start if requested and available).
        let mut lambda = match (&config.multiplier_init, &warm_start) {
            (MultiplierInitStrategy::WarmStart, Some(ms)) => ms.extract_equality_vec(constraints),
            _ => vec![0.0; total_eq],
        };
        // Inequality multipliers must be >= 0
        let mut mu = match (&config.multiplier_init, &warm_start) {
            (MultiplierInitStrategy::WarmStart, Some(ms)) => {
                ms.extract_inequality_vec(inequalities)
            }
            _ => vec![0.0; total_ineq],
        };
        let mut rho = config.rho_init;
        let mut prev_violation = f64::INFINITY;
        let mut total_inner_iters = 0;

        for outer_iter in 0..config.max_outer_iterations {
            // Build the augmented Lagrangian as an Objective for BfgsSolver
            // L_A(x) = f(x) + λ^T g(x) + (ρ/2) ||g(x)||^2
            //        + Σ_ineq (ρ/2) [max(0, h_j + μ_j/ρ)² - (μ_j/ρ)²]
            let alm_objective = AugmentedLagrangianObjective {
                objective,
                constraints,
                inequalities,
                param_ids: &param_ids,
                lambda: &lambda,
                mu: &mu,
                rho,
            };

            // Solve inner subproblem with BFGS or BFGS-B depending on bounds.
            let inner_config = OptimizationConfig {
                max_outer_iterations: config.max_inner_iterations,
                dual_tolerance: config.inner_tolerance,
                ..config.clone()
            };
            let inner_result = if use_bfgs_b {
                super::bfgs_b::BfgsBSolver::solve(&alm_objective, store, &inner_config)
            } else {
                super::bfgs::BfgsSolver::solve(&alm_objective, store, &inner_config)
            };

            total_inner_iters += inner_result.outer_iterations;

            // Evaluate constraint violations
            let mut violations = Vec::with_capacity(total_eq);
            for c in constraints {
                violations.extend(c.residuals(store));
            }
            let violation_norm: f64 = violations.iter().map(|v| v * v).sum::<f64>().sqrt();

            // Dual feasibility: ∇_x L = ∇f + Σ λ_i ∇g_i + ρ Σ g_i ∇g_i
            // After the inner solve converges, ∇_x L_A ≈ 0, so we estimate
            // dual feasibility from the inner solver's residual norm.
            // A more accurate check: ∇f + J^T (λ + ρ g)
            let obj_grad = objective.gradient(store);
            let mut grad_l = vec![0.0; n];
            for (pid, val) in &obj_grad {
                if let Some(col) = param_ids.iter().position(|p| p == pid) {
                    grad_l[col] = *val;
                }
            }
            // Add J^T (λ + ρ g) contribution from equality constraints
            let mut eq_offset = 0;
            for (_ci, c) in constraints.iter().enumerate() {
                let jac = c.jacobian(store);
                let resid = c.residuals(store);
                let eq_count = c.equation_count();
                for (row, pid, val) in &jac {
                    if let Some(col) = param_ids.iter().position(|p| p == pid) {
                        let dual_coeff = lambda[eq_offset + row] + rho * resid[*row];
                        grad_l[col] += dual_coeff * val;
                    }
                }
                eq_offset += eq_count;
            }
            // Add J_h^T * max(0, μ + ρ·h) contribution from inequality constraints
            let mut ineq_offset = 0;
            for h in inequalities {
                let vals = h.values(store);
                let jac = h.jacobian(store);
                for (row, pid, val) in jac {
                    if let Some(col) = param_ids.iter().position(|p| *p == pid) {
                        let shifted = mu[ineq_offset + row] + rho * vals[row];
                        if shifted > 0.0 {
                            grad_l[col] += shifted * val;
                        }
                    }
                }
                ineq_offset += vals.len();
            }
            let dual_norm: f64 = grad_l.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Primal feasibility: max(eq_violation, max_ineq_violation)
            let ineq_violation: f64 = inequalities
                .iter()
                .flat_map(|h| h.values(store))
                .map(|v| v.max(0.0))
                .fold(0.0_f64, f64::max);
            let primal_violation = violation_norm.max(ineq_violation);

            // Complementarity: max |μ_j * h_j(x)|
            let complementarity = compute_complementarity(inequalities, &mu, store);

            // Check convergence
            let (primal_check, dual_check) = if config.relative_tolerance {
                let total_constraints = total_eq + total_ineq;
                let primal_scale = (1.0_f64).max((total_constraints as f64).sqrt());
                let dual_scale = (1.0_f64).max((n as f64).sqrt());
                (primal_violation / primal_scale, dual_norm / dual_scale)
            } else {
                (primal_violation, dual_norm)
            };
            if primal_check < config.outer_tolerance && dual_check < config.dual_tolerance {
                let mut multiplier_store = MultiplierStore::new();
                let mut eq_idx = 0;
                for c in constraints {
                    for row in 0..c.equation_count() {
                        multiplier_store.set(MultiplierId::new(c.id(), row), lambda[eq_idx]);
                        eq_idx += 1;
                    }
                }
                let mut ineq_idx = 0;
                for h in inequalities {
                    for row in 0..h.inequality_count() {
                        multiplier_store.set(MultiplierId::new(h.id(), row), mu[ineq_idx]);
                        ineq_idx += 1;
                    }
                }

                return OptimizationResult {
                    objective_value: objective.value(store),
                    status: OptimizationStatus::Converged,
                    outer_iterations: outer_iter + 1,
                    inner_iterations: total_inner_iters,
                    kkt_residual: KktResidual {
                        primal: primal_violation,
                        dual: dual_norm,
                        complementarity,
                    },
                    multipliers: multiplier_store,
                    constraint_violations: violations,
                    duration: start.elapsed(),
                };
            }

            // Update equality multipliers: λ_{k+1} = λ_k + ρ * g(x_k)
            let mut eq_idx = 0;
            for c in constraints {
                let resid = c.residuals(store);
                for (row, r) in resid.iter().enumerate() {
                    lambda[eq_idx] += rho * r;
                    // Divergence guard: clamp prevents multiplier explosion when
                    // the inner solve does not reduce constraint violation sufficiently.
                    lambda[eq_idx] =
                        lambda[eq_idx].clamp(-config.max_multiplier, config.max_multiplier);
                    eq_idx += 1;
                    let _ = row;
                }
            }

            // Update inequality multipliers: μ_{k+1} = max(0, μ_k + ρ * h(x_k))
            let mut ineq_idx = 0;
            for h in inequalities {
                let vals = h.values(store);
                for v in &vals {
                    mu[ineq_idx] = (mu[ineq_idx] + rho * v).max(0.0);
                    // Divergence guard: clamp prevents multiplier explosion when
                    // the inner solve does not reduce constraint violation sufficiently.
                    mu[ineq_idx] = mu[ineq_idx].min(config.max_multiplier);
                    ineq_idx += 1;
                }
            }

            // Increase penalty if violation didn't decrease enough
            if violation_norm > 0.25 * prev_violation {
                rho = (rho * config.rho_growth).min(config.rho_max);
            }
            prev_violation = violation_norm;
        }

        // Max outer iterations
        let mut multiplier_store = MultiplierStore::new();
        let mut eq_idx = 0;
        for c in constraints {
            for row in 0..c.equation_count() {
                multiplier_store.set(MultiplierId::new(c.id(), row), lambda[eq_idx]);
                eq_idx += 1;
            }
        }
        let mut ineq_idx = 0;
        for h in inequalities {
            for row in 0..h.inequality_count() {
                multiplier_store.set(MultiplierId::new(h.id(), row), mu[ineq_idx]);
                ineq_idx += 1;
            }
        }

        let violations: Vec<f64> = constraints
            .iter()
            .flat_map(|c| c.residuals(store))
            .collect();
        let violation_norm: f64 = violations.iter().map(|v| v * v).sum::<f64>().sqrt();
        let ineq_violation: f64 = inequalities
            .iter()
            .flat_map(|h| h.values(store))
            .map(|v| v.max(0.0))
            .fold(0.0_f64, f64::max);
        let primal_violation = violation_norm.max(ineq_violation);

        let complementarity = compute_complementarity(inequalities, &mu, store);

        OptimizationResult {
            objective_value: objective.value(store),
            status: OptimizationStatus::MaxIterationsReached,
            outer_iterations: config.max_outer_iterations,
            inner_iterations: total_inner_iters,
            kkt_residual: KktResidual {
                primal: primal_violation,
                dual: f64::INFINITY,
                complementarity,
            },
            multipliers: multiplier_store,
            constraint_violations: violations,
            duration: start.elapsed(),
        }
    }
}

// ---------------------------------------------------------------------------
// Augmented Lagrangian as an Objective (for BFGS inner loop)
// ---------------------------------------------------------------------------

/// Wraps an objective + equality + inequality constraints as a scalar Objective.
///
/// L_A(x) = f(x) + λ^T g(x) + (ρ/2) ||g(x)||^2
///         + Σ_ineq (ρ/2) [max(0, h_j + μ_j/ρ)² - (μ_j/ρ)²]
///
/// The gradient is:
/// ∇L_A = ∇f + J_g^T (λ + ρ g(x)) + J_h^T max(0, μ + ρ h(x))
///
/// BFGS minimizes this using only value + gradient — no Hessian needed.
struct AugmentedLagrangianObjective<'a> {
    objective: &'a dyn Objective,
    constraints: &'a [&'a dyn Constraint],
    inequalities: &'a [&'a dyn InequalityFn],
    param_ids: &'a [crate::id::ParamId],
    lambda: &'a [f64],
    mu: &'a [f64],
    rho: f64,
}

impl Objective for AugmentedLagrangianObjective<'_> {
    fn id(&self) -> crate::optimization::ObjectiveId {
        crate::optimization::ObjectiveId::new(u32::MAX, 0)
    }

    fn name(&self) -> &str {
        "augmented_lagrangian"
    }

    fn param_ids(&self) -> &[crate::id::ParamId] {
        self.param_ids
    }

    fn value(&self, store: &ParamStore) -> f64 {
        let mut val = self.objective.value(store);

        let mut eq_offset = 0;
        for c in self.constraints {
            let resid = c.residuals(store);
            for (row, ri) in resid.iter().enumerate() {
                // λ^T g + (ρ/2) ||g||^2
                val += self.lambda[eq_offset + row] * ri + 0.5 * self.rho * ri * ri;
            }
            eq_offset += resid.len();
        }

        // Inequality terms: (ρ/2) [max(0, h_j + μ_j/ρ)² - (μ_j/ρ)²]
        let mut ineq_offset = 0;
        for h in self.inequalities {
            let vals = h.values(store);
            for (row, hi) in vals.iter().enumerate() {
                let mu_over_rho = self.mu[ineq_offset + row] / self.rho;
                let shifted = hi + mu_over_rho;
                if shifted > 0.0 {
                    val += 0.5 * self.rho * shifted * shifted;
                }
                val -= 0.5 * self.rho * mu_over_rho * mu_over_rho;
            }
            ineq_offset += vals.len();
        }

        val
    }

    fn gradient(&self, store: &ParamStore) -> Vec<(crate::id::ParamId, f64)> {
        let n = self.param_ids.len();

        // Start with objective gradient
        let obj_grad = self.objective.gradient(store);
        let mut grad = vec![0.0; n];
        for (pid, val) in obj_grad {
            if let Some(col) = self.param_ids.iter().position(|&p| p == pid) {
                grad[col] = val;
            }
        }

        // Add J_g^T (λ + ρ g(x))
        let mut eq_offset = 0;
        for c in self.constraints {
            let resid = c.residuals(store);
            let jac = c.jacobian(store);
            for (row, pid, val) in jac {
                if let Some(col) = self.param_ids.iter().position(|&p| p == pid) {
                    let dual_coeff = self.lambda[eq_offset + row] + self.rho * resid[row];
                    grad[col] += dual_coeff * val;
                }
            }
            eq_offset += resid.len();
        }

        // Add J_h^T * max(0, μ + ρ·h(x))
        let mut ineq_offset = 0;
        for h in self.inequalities {
            let vals = h.values(store);
            let jac = h.jacobian(store);
            for (row, pid, val) in jac {
                if let Some(col) = self.param_ids.iter().position(|&p| p == pid) {
                    let shifted = self.mu[ineq_offset + row] + self.rho * vals[row];
                    if shifted > 0.0 {
                        grad[col] += shifted * val;
                    }
                }
            }
            ineq_offset += vals.len();
        }

        // Return sparse (only non-zero entries)
        grad.into_iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > 1e-30)
            .map(|(i, v)| (self.param_ids[i], v))
            .collect()
    }
}
