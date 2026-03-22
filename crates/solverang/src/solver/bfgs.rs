//! L-BFGS solver for unconstrained optimization.
//!
//! Minimizes a scalar objective `f(x)` using the Limited-memory BFGS
//! quasi-Newton method with Armijo backtracking line search.
//!
//! This solver requires only gradient information (no Hessian), making it
//! suitable as the default optimization algorithm.

use std::collections::VecDeque;
use std::time::Instant;

use crate::optimization::{
    KktResidual, MultiplierStore, Objective, OptimizationConfig, OptimizationResult,
    OptimizationStatus,
};
use crate::param::ParamStore;

/// L-BFGS solver for unconstrained optimization.
///
/// Uses the two-loop recursion algorithm to approximate the inverse Hessian
/// from the last `m` gradient pairs, combined with Armijo backtracking line search.
pub struct BfgsSolver;

impl BfgsSolver {
    /// Solve an unconstrained optimization problem.
    ///
    /// Minimizes `objective.value(store)` by adjusting the free parameters
    /// in `store`. Returns when `||gradient|| < tolerance` or max iterations reached.
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

        // Extract current parameter values into solver vector
        let param_ids = &mapping.col_to_param;
        let mut x: Vec<f64> = param_ids.iter().map(|&pid| store.get(pid)).collect();

        // L-BFGS memory: pairs of (s_k, y_k)
        let m = config.lbfgs_memory;
        let mut s_history: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
        let mut y_history: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);

        // Write x into store, compute initial gradient
        write_x_to_store(store, &param_ids, &x);
        let mut f = objective.value(store);
        let mut grad = dense_gradient(objective, store, &param_ids, n);
        let mut grad_norm = vec_norm(&grad);

        for iter in 0..config.max_outer_iterations {
            // Check convergence
            if grad_norm < config.dual_tolerance {
                return OptimizationResult {
                    objective_value: f,
                    status: OptimizationStatus::Converged,
                    outer_iterations: iter,
                    inner_iterations: 0,
                    kkt_residual: KktResidual {
                        primal: 0.0,
                        dual: grad_norm,
                        complementarity: 0.0,
                    },
                    multipliers: MultiplierStore::new(),
                    constraint_violations: Vec::new(),
                    duration: start.elapsed(),
                };
            }

            // Compute search direction via L-BFGS two-loop recursion
            let direction = lbfgs_direction(&grad, &s_history, &y_history);

            // Armijo backtracking line search
            let dg = dot(&grad, &direction);
            if dg >= 0.0 {
                // Not a descent direction — reset L-BFGS memory and use steepest descent
                s_history.clear();
                y_history.clear();
                let direction: Vec<f64> = grad.iter().map(|g| -g).collect();
                let dg = dot(&grad, &direction);
                let alpha = armijo_line_search(
                    objective,
                    store,
                    &param_ids,
                    &x,
                    &direction,
                    f,
                    dg,
                    config,
                );
                let x_new: Vec<f64> = x.iter().zip(&direction).map(|(xi, di)| xi + alpha * di).collect();
                write_x_to_store(store, &param_ids, &x_new);
                let f_new = objective.value(store);
                let grad_new = dense_gradient(objective, store, &param_ids, n);

                let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
                let y: Vec<f64> = grad_new.iter().zip(&grad).map(|(a, b)| a - b).collect();

                if dot(&s, &y) > 1e-10 * vec_norm(&s) * vec_norm(&y) {
                    if s_history.len() == m {
                        s_history.pop_front();
                        y_history.pop_front();
                    }
                    s_history.push_back(s);
                    y_history.push_back(y);
                }

                x = x_new;
                f = f_new;
                grad = grad_new;
                grad_norm = vec_norm(&grad);
                continue;
            }

            let alpha = armijo_line_search(
                objective,
                store,
                &param_ids,
                &x,
                &direction,
                f,
                dg,
                config,
            );

            // Update x
            let x_new: Vec<f64> = x.iter().zip(&direction).map(|(xi, di)| xi + alpha * di).collect();

            // Compute new gradient
            write_x_to_store(store, &param_ids, &x_new);
            let f_new = objective.value(store);
            let grad_new = dense_gradient(objective, store, &param_ids, n);

            // L-BFGS update: s = x_new - x, y = grad_new - grad
            let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = grad_new.iter().zip(&grad).map(|(a, b)| a - b).collect();

            // Curvature condition: only update if s^T y > 0
            if dot(&s, &y) > 1e-10 * vec_norm(&s) * vec_norm(&y) {
                if s_history.len() == m {
                    s_history.pop_front();
                    y_history.pop_front();
                }
                s_history.push_back(s);
                y_history.push_back(y);
            }

            x = x_new;
            f = f_new;
            grad = grad_new;
            grad_norm = vec_norm(&grad);
        }

        // Max iterations reached
        write_x_to_store(store, &param_ids, &x);
        OptimizationResult {
            objective_value: f,
            status: OptimizationStatus::MaxIterationsReached,
            outer_iterations: config.max_outer_iterations,
            inner_iterations: 0,
            kkt_residual: KktResidual {
                primal: 0.0,
                dual: grad_norm,
                complementarity: 0.0,
            },
            multipliers: MultiplierStore::new(),
            constraint_violations: Vec::new(),
            duration: start.elapsed(),
        }
    }
}

/// L-BFGS two-loop recursion: compute H_k * grad using stored (s, y) pairs.
///
/// Returns the search direction d = -H_k * grad.
fn lbfgs_direction(
    grad: &[f64],
    s_history: &VecDeque<Vec<f64>>,
    y_history: &VecDeque<Vec<f64>>,
) -> Vec<f64> {
    let k = s_history.len();
    if k == 0 {
        // No history: steepest descent
        return grad.iter().map(|g| -g).collect();
    }

    let mut q = grad.to_vec();
    let mut alpha_vec = vec![0.0; k];
    let mut rho_vec = vec![0.0; k];

    // Forward loop (most recent first)
    for i in (0..k).rev() {
        let sy = dot(&s_history[i], &y_history[i]);
        rho_vec[i] = if sy.abs() > 1e-30 { 1.0 / sy } else { 0.0 };
        alpha_vec[i] = rho_vec[i] * dot(&s_history[i], &q);
        for j in 0..q.len() {
            q[j] -= alpha_vec[i] * y_history[i][j];
        }
    }

    // Initial Hessian approximation: H_0 = (s^T y / y^T y) * I
    let last = k - 1;
    let yy = dot(&y_history[last], &y_history[last]);
    let sy = dot(&s_history[last], &y_history[last]);
    let gamma = if yy.abs() > 1e-30 { sy / yy } else { 1.0 };

    let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

    // Backward loop (oldest first)
    for i in 0..k {
        let beta = rho_vec[i] * dot(&y_history[i], &r);
        for j in 0..r.len() {
            r[j] += (alpha_vec[i] - beta) * s_history[i][j];
        }
    }

    // Negate for descent direction
    for ri in &mut r {
        *ri = -*ri;
    }
    r
}

/// Armijo backtracking line search.
///
/// Finds α such that f(x + α*d) ≤ f(x) + c₁*α*(∇f·d).
fn armijo_line_search(
    objective: &dyn Objective,
    store: &mut ParamStore,
    param_ids: &[crate::id::ParamId],
    x: &[f64],
    direction: &[f64],
    f_x: f64,
    directional_derivative: f64,
    config: &OptimizationConfig,
) -> f64 {
    let c1 = config.armijo_c1;
    let backtrack = config.line_search_backtrack;
    let min_step = config.line_search_min_step;
    let mut alpha = 1.0;

    loop {
        // Trial point
        let x_trial: Vec<f64> = x
            .iter()
            .zip(direction)
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        write_x_to_store(store, param_ids, &x_trial);
        let f_trial = objective.value(store);

        // Armijo condition
        if f_trial <= f_x + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= backtrack;
        if alpha < min_step {
            return alpha;
        }
    }
}

// --- Utility functions ---

fn dense_gradient(
    objective: &dyn Objective,
    store: &ParamStore,
    param_ids: &[crate::id::ParamId],
    n: usize,
) -> Vec<f64> {
    let mut grad = vec![0.0; n];
    let sparse = objective.gradient(store);
    for (pid, val) in sparse {
        // Find the solver column for this ParamId
        if let Some(col) = param_ids.iter().position(|&p| p == pid) {
            grad[col] = val;
        }
    }
    grad
}

fn write_x_to_store(
    store: &mut ParamStore,
    param_ids: &[crate::id::ParamId],
    x: &[f64],
) {
    for (i, &pid) in param_ids.iter().enumerate() {
        store.set(pid, x[i]);
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(ai, bi)| ai * bi).sum()
}

fn vec_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}
