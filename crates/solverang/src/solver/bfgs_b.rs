//! L-BFGS-B solver for box-constrained optimization.
//!
//! Minimizes a scalar objective `f(x)` subject to box constraints
//! `lower_i ≤ x_i ≤ upper_i`, using the projected L-BFGS method.
//!
//! The algorithm:
//! 1. Project the initial point onto the feasible box.
//! 2. Compute gradient and check convergence via projected gradient norm.
//! 3. Compute an L-BFGS search direction (reusing `bfgs::lbfgs_direction`).
//! 4. Project the candidate step to stay feasible.
//! 5. Run line search on the projected step.
//! 6. Update L-BFGS history with curvature pair (s, y).

use std::collections::VecDeque;
use std::time::Instant;

use crate::optimization::{
    KktResidual, MultiplierStore, Objective, OptimizationConfig, OptimizationResult,
    OptimizationStatus,
};
use crate::param::ParamStore;
use crate::solver::bfgs::{
    dense_gradient, dot, lbfgs_direction, update_lbfgs_history, vec_norm, write_x_to_store,
};
use crate::solver::line_search;

/// L-BFGS-B solver for box-constrained optimization.
pub struct BfgsBSolver;

impl BfgsBSolver {
    /// Solve a box-constrained optimization problem.
    ///
    /// Minimizes `objective.value(store)` subject to the bounds stored in
    /// `store` for each free parameter. Returns when the projected gradient
    /// norm is below tolerance or max iterations are reached.
    pub fn solve(
        objective: &dyn Objective,
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

        let param_ids = &mapping.col_to_param;

        // Extract bounds for each free parameter in solver column order.
        let lower: Vec<f64> = param_ids.iter().map(|&pid| store.bounds(pid).0).collect();
        let upper: Vec<f64> = param_ids.iter().map(|&pid| store.bounds(pid).1).collect();

        // Extract initial point and project it onto the feasible box.
        let mut x: Vec<f64> = param_ids.iter().map(|&pid| store.get(pid)).collect();
        project(&mut x, &lower, &upper);
        write_x_to_store(store, param_ids, &x);

        // L-BFGS memory.
        let m = config.lbfgs_memory;
        let mut s_history: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
        let mut y_history: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);

        let mut f = objective.value(store);
        let mut grad = dense_gradient(objective, store, param_ids, n);

        for iter in 0..config.max_outer_iterations {
            // Convergence check via projected gradient norm.
            let pg_norm = projected_gradient_norm(&x, &grad, &lower, &upper);
            let dual_check = if config.relative_tolerance {
                pg_norm / (1.0_f64).max(f.abs())
            } else {
                pg_norm
            };
            if dual_check < config.dual_tolerance {
                write_x_to_store(store, param_ids, &x);
                return OptimizationResult {
                    objective_value: f,
                    status: OptimizationStatus::Converged,
                    outer_iterations: iter,
                    inner_iterations: 0,
                    kkt_residual: KktResidual {
                        primal: 0.0,
                        dual: pg_norm,
                        complementarity: 0.0,
                    },
                    multipliers: MultiplierStore::new(),
                    constraint_violations: Vec::new(),
                    duration: start.elapsed(),
                };
            }

            // Compute L-BFGS search direction.
            let mut direction = lbfgs_direction(&grad, &s_history, &y_history);

            // Project the direction: for each coordinate, if the step would
            // violate a bound that is already active, zero it out.
            for i in 0..n {
                if x[i] <= lower[i] && direction[i] < 0.0 {
                    direction[i] = 0.0;
                }
                if x[i] >= upper[i] && direction[i] > 0.0 {
                    direction[i] = 0.0;
                }
            }

            // Check that we still have a descent direction after projection.
            // If not (all components zeroed or positive dot product), fall back
            // to projected steepest descent.
            let dg = dot(&grad, &direction);
            if dg >= 0.0 {
                s_history.clear();
                y_history.clear();
                direction = grad.iter().map(|g| -g).collect();
                // Re-project the steepest descent direction.
                for i in 0..n {
                    if x[i] <= lower[i] && direction[i] < 0.0 {
                        direction[i] = 0.0;
                    }
                    if x[i] >= upper[i] && direction[i] > 0.0 {
                        direction[i] = 0.0;
                    }
                }
                // If projected steepest descent is also zero, we are at a
                // corner of the feasible box — converged.
                if vec_norm(&direction) < 1e-15 {
                    write_x_to_store(store, param_ids, &x);
                    return OptimizationResult {
                        objective_value: f,
                        status: OptimizationStatus::Converged,
                        outer_iterations: iter,
                        inner_iterations: 0,
                        kkt_residual: KktResidual {
                            primal: 0.0,
                            dual: pg_norm,
                            complementarity: 0.0,
                        },
                        multipliers: MultiplierStore::new(),
                        constraint_violations: Vec::new(),
                        duration: start.elapsed(),
                    };
                }
            }

            // Line search along the projected direction.
            let (alpha, f_new) = line_search::line_search(
                objective,
                store,
                param_ids,
                &x,
                &direction,
                f,
                &grad,
                config,
            );

            // Compute new iterate and project onto box.
            let mut x_new: Vec<f64> =
                x.iter().zip(&direction).map(|(xi, di)| xi + alpha * di).collect();
            project(&mut x_new, &lower, &upper);
            debug_assert!(
                x_new.iter().zip(lower.iter()).zip(upper.iter())
                    .all(|((xi, lo), hi)| *xi >= *lo && *xi <= *hi),
                "BfgsB bounds invariant violated after projection"
            );

            write_x_to_store(store, param_ids, &x_new);
            let grad_new = dense_gradient(objective, store, param_ids, n);

            // L-BFGS update: s = x_new - x (unprojected), y = P(g_new) - P(g_old).
            let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();

            // Project gradients: zero out components where the iterate is at a bound
            // and the gradient would push further into the bound
            let pg_new: Vec<f64> = grad_new.iter().enumerate()
                .map(|(i, &g)| {
                    if x_new[i] <= lower[i] && g > 0.0 { 0.0 }
                    else if x_new[i] >= upper[i] && g < 0.0 { 0.0 }
                    else { g }
                }).collect();
            let pg_old: Vec<f64> = grad.iter().enumerate()
                .map(|(i, &g)| {
                    if x[i] <= lower[i] && g > 0.0 { 0.0 }
                    else if x[i] >= upper[i] && g < 0.0 { 0.0 }
                    else { g }
                }).collect();
            let y: Vec<f64> = pg_new.iter().zip(&pg_old).map(|(a, b)| a - b).collect();

            update_lbfgs_history(&mut s_history, &mut y_history, s, y, m);

            x = x_new;
            f = f_new;
            grad = grad_new;
        }

        // Max iterations reached.
        write_x_to_store(store, param_ids, &x);
        let pg_norm = projected_gradient_norm(&x, &grad, &lower, &upper);
        OptimizationResult {
            objective_value: f,
            status: OptimizationStatus::MaxIterationsReached,
            outer_iterations: config.max_outer_iterations,
            inner_iterations: 0,
            kkt_residual: KktResidual {
                primal: 0.0,
                dual: pg_norm,
                complementarity: 0.0,
            },
            multipliers: MultiplierStore::new(),
            constraint_violations: Vec::new(),
            duration: start.elapsed(),
        }
    }
}

/// Project each component of `x` onto the interval `[lower[i], upper[i]]`.
fn project(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for i in 0..x.len() {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }
}

/// Projected gradient norm: `||P(x - g) - x||` where P is the box projection.
///
/// This is the standard convergence metric for box-constrained problems.
/// It equals zero if and only if x is a KKT point.
fn projected_gradient_norm(x: &[f64], grad: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut norm_sq = 0.0;
    for i in 0..x.len() {
        let pg = (x[i] - grad[i]).clamp(lower[i], upper[i]) - x[i];
        norm_sq += pg * pg;
    }
    norm_sq.sqrt()
}
