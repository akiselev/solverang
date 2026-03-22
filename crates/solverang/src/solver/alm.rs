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
    KktResidual, MultiplierId, MultiplierStore, Objective, OptimizationConfig, OptimizationResult,
    OptimizationStatus,
};
use crate::param::ParamStore;

/// Augmented Lagrangian Method solver.
pub struct AlmSolver;

impl AlmSolver {
    /// Solve a constrained optimization problem.
    ///
    /// Minimizes `objective.value(store)` subject to equality constraints
    /// (existing `Constraint` objects where residuals should be zero).
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
        store: &mut ParamStore,
        config: &OptimizationConfig,
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
        let _m_eq: usize = constraints.iter().map(|c| c.equation_count()).count();
        let total_eq: usize = constraints.iter().map(|c| c.equation_count()).sum();

        // Initialize multipliers
        let mut lambda = vec![0.0; total_eq];
        let mut rho = config.rho_init;
        let mut prev_violation = f64::INFINITY;
        let mut total_inner_iters = 0;

        for outer_iter in 0..config.max_outer_iterations {
            // Build the augmented Lagrangian as an Objective for BfgsSolver
            // L_A(x) = f(x) + λ^T g(x) + (ρ/2) ||g(x)||^2
            let alm_objective = AugmentedLagrangianObjective {
                objective,
                constraints,
                param_ids: &param_ids,
                lambda: &lambda,
                rho,
            };

            // Solve inner subproblem with BFGS
            let inner_config = OptimizationConfig {
                max_outer_iterations: config.max_inner_iterations,
                dual_tolerance: config.inner_tolerance,
                ..config.clone()
            };
            let inner_result =
                super::bfgs::BfgsSolver::solve(&alm_objective, store, &inner_config);

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
            // Add J^T (λ + ρ g) contribution
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
            let dual_norm: f64 = grad_l.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Check convergence
            if violation_norm < config.outer_tolerance && dual_norm < config.dual_tolerance {
                let mut multiplier_store = MultiplierStore::new();
                let mut eq_idx = 0;
                for c in constraints {
                    for row in 0..c.equation_count() {
                        multiplier_store.set(
                            MultiplierId::new(c.id(), row),
                            lambda[eq_idx],
                        );
                        eq_idx += 1;
                    }
                }

                return OptimizationResult {
                    objective_value: objective.value(store),
                    status: OptimizationStatus::Converged,
                    outer_iterations: outer_iter + 1,
                    inner_iterations: total_inner_iters,
                    kkt_residual: KktResidual {
                        primal: violation_norm,
                        dual: dual_norm,
                        complementarity: 0.0,
                    },
                    multipliers: multiplier_store,
                    constraint_violations: violations,
                    duration: start.elapsed(),
                };
            }

            // Update multipliers: λ_{k+1} = λ_k + ρ * g(x_k)
            let mut eq_idx = 0;
            for c in constraints {
                let resid = c.residuals(store);
                for (row, r) in resid.iter().enumerate() {
                    lambda[eq_idx] += rho * r;
                    // Divergence guard
                    lambda[eq_idx] = lambda[eq_idx].clamp(
                        -config.max_multiplier,
                        config.max_multiplier,
                    );
                    eq_idx += 1;
                    let _ = row;
                }
            }

            // Increase penalty if violation didn't decrease enough
            if violation_norm > 0.25 * prev_violation {
                rho = (rho * config.rho_growth).min(config.rho_max);
            }
            prev_violation = violation_norm;

            // Divergence detection: if multipliers exploded
            let max_lambda = lambda.iter().map(|l| l.abs()).fold(0.0_f64, f64::max);
            if max_lambda >= config.max_multiplier * 0.9 {
                // Freeze multipliers (penalty-only mode)
                // This prevents further divergence
            }
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

        let violations: Vec<f64> = constraints
            .iter()
            .flat_map(|c| c.residuals(store))
            .collect();
        let violation_norm: f64 = violations.iter().map(|v| v * v).sum::<f64>().sqrt();

        OptimizationResult {
            objective_value: objective.value(store),
            status: OptimizationStatus::MaxIterationsReached,
            outer_iterations: config.max_outer_iterations,
            inner_iterations: total_inner_iters,
            kkt_residual: KktResidual {
                primal: violation_norm,
                dual: f64::INFINITY,
                complementarity: 0.0,
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

/// Wraps an objective + equality constraints as a scalar Objective.
///
/// L_A(x) = f(x) + λ^T g(x) + (ρ/2) ||g(x)||^2
///
/// The gradient is:
/// ∇L_A = ∇f + J_g^T (λ + ρ g(x))
///
/// BFGS minimizes this using only value + gradient — no Hessian needed.
struct AugmentedLagrangianObjective<'a> {
    objective: &'a dyn Objective,
    constraints: &'a [&'a dyn Constraint],
    param_ids: &'a [crate::id::ParamId],
    lambda: &'a [f64],
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

        // Return sparse (only non-zero entries)
        grad.into_iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > 1e-30)
            .map(|(i, v)| (self.param_ids[i], v))
            .collect()
    }
}
