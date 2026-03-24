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

            // Compute search direction via Generalized Cauchy Point + subspace minimization.
            let gamma = compute_gamma(&s_history, &y_history);
            let (x_cauchy, active_set) =
                generalized_cauchy_point(&x, &grad, &lower, &upper, gamma);
            let correction =
                subspace_minimization(&x_cauchy, &grad, &active_set, &lower, &upper,
                                       &s_history, &y_history);
            let mut direction: Vec<f64> = x_cauchy
                .iter()
                .zip(&correction)
                .zip(&x)
                .map(|((c, d), xi)| c + d - xi)
                .collect();

            // Fallback: if GCP produces a degenerate zero direction, use
            // projected steepest descent.
            if vec_norm(&direction) < 1e-15 {
                direction = project_direction(&grad.iter().map(|g| -g).collect::<Vec<_>>(),
                                              &x, &lower, &upper);
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
                s_history.clear();
                y_history.clear();
            }

            // Ensure descent direction.
            let dg = dot(&grad, &direction);
            if dg >= 0.0 {
                s_history.clear();
                y_history.clear();
                direction = project_direction(&grad.iter().map(|g| -g).collect::<Vec<_>>(),
                                              &x, &lower, &upper);
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

/// Project a direction vector: zero out components that would push further into
/// an already-active bound.
fn project_direction(dir: &[f64], x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    let mut d = dir.to_vec();
    for i in 0..d.len() {
        if x[i] <= lower[i] && d[i] < 0.0 {
            d[i] = 0.0;
        }
        if x[i] >= upper[i] && d[i] > 0.0 {
            d[i] = 0.0;
        }
    }
    d
}

/// Compute the scaled-identity Hessian parameter γ = yᵀy / sᵀy from L-BFGS history.
fn compute_gamma(s_history: &VecDeque<Vec<f64>>, y_history: &VecDeque<Vec<f64>>) -> f64 {
    let k = s_history.len();
    if k == 0 {
        return 1.0;
    }
    let last = k - 1;
    let sy = dot(&s_history[last], &y_history[last]);
    let yy = dot(&y_history[last], &y_history[last]);
    if sy.abs() > 1e-30 {
        yy / sy
    } else {
        1.0
    }
}

/// Generalized Cauchy Point (Byrd-Lu-Nocedal-Zhu 1995).
///
/// Finds the minimizer of the quadratic model `q(t) = f + gᵀ(x(t)-x) + γ/2 ||x(t)-x||²`
/// along the piecewise-linear path `x(t) = P[x - t·g]`.
///
/// Returns `(cauchy_point, active_set)` where `active_set[i] = true` means
/// variable `i` is at a bound at the Cauchy point.
fn generalized_cauchy_point(
    x: &[f64],
    grad: &[f64],
    lower: &[f64],
    upper: &[f64],
    gamma: f64,
) -> (Vec<f64>, Vec<bool>) {
    let n = x.len();
    let inf = f64::INFINITY;

    // Compute breakpoints: t_i where x_i(t) hits a bound along x(t) = P[x - t*g].
    // For variable i with g[i] < 0 (moves toward upper bound):  t_i = (x[i] - upper[i]) / g[i]
    // For variable i with g[i] > 0 (moves toward lower bound):  t_i = (x[i] - lower[i]) / g[i]
    let mut breakpoints: Vec<(f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let t_i = if grad[i] < 0.0 && upper[i] < inf {
            (x[i] - upper[i]) / grad[i]
        } else if grad[i] > 0.0 && lower[i] > f64::NEG_INFINITY {
            (x[i] - lower[i]) / grad[i]
        } else {
            inf
        };
        if t_i > 1e-30 {
            breakpoints.push((t_i, i));
        }
    }
    breakpoints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Track which variables are active (at a bound and gradient pointing out of feasible set).
    let mut active = vec![false; n];
    for i in 0..n {
        if grad[i] > 0.0 && x[i] <= lower[i] {
            active[i] = true;
        } else if grad[i] < 0.0 && x[i] >= upper[i] {
            active[i] = true;
        }
    }

    // Path derivative at t=0: fp = gᵀ d where d[i] = -g[i] for free vars.
    // fp = -sum_{free} g[i]^2
    // fpp = gamma * sum_{free} g[i]^2 = -gamma * fp
    let mut fp: f64 = -(0..n).filter(|&i| !active[i]).map(|i| grad[i] * grad[i]).sum::<f64>();
    let mut fpp: f64 = -gamma * fp; // = gamma * sum g_i^2

    if fpp < 1e-30 || fp >= 0.0 {
        // Degenerate or already optimal: return x clamped.
        let mut x_c = x.to_vec();
        project(&mut x_c, lower, upper);
        return (x_c, active);
    }

    let mut t_prev = 0.0_f64;

    for &(t_i, coord) in &breakpoints {
        // Already active?  Skip.
        if active[coord] {
            continue;
        }

        // Minimum of 1D quadratic in current segment [t_prev, t_i]:
        // t* relative to segment start = -fp / fpp  (absolute: t_prev + dt_opt)
        let dt_opt = -fp / fpp;
        if dt_opt <= 1e-30 {
            break;
        }
        let t_star = t_prev + dt_opt;

        if t_star <= t_i {
            // Minimum is inside this segment.
            let mut x_c = x.to_vec();
            for i in 0..n {
                if !active[i] {
                    x_c[i] = (x[i] - t_star * grad[i]).clamp(lower[i], upper[i]);
                }
            }
            return (x_c, active);
        }

        // Advance to breakpoint: update fp and fpp for the now-active variable.
        let dt = t_i - t_prev;
        // fp advances along the segment: fp += fpp * dt (path derivative at t_i)
        fp += fpp * dt;
        // Remove coord's contribution from future derivative (it's now fixed at bound).
        let g_j = grad[coord];
        fp -= gamma * g_j * g_j; // d[coord] = -g_j, so gamma * d_j^2 = gamma * g_j^2
        fpp -= gamma * g_j * g_j;
        if fpp < 1e-30 {
            fpp = 1e-30;
        }

        active[coord] = true;
        t_prev = t_i;
    }

    // Minimum is beyond all breakpoints (or no breakpoints).
    // Use t* from remaining free variables.
    let mut x_c = x.to_vec();
    if fpp > 1e-30 && fp < 0.0 {
        let dt_opt = -fp / fpp;
        let t_abs = t_prev + dt_opt;
        for i in 0..n {
            if !active[i] {
                x_c[i] = (x[i] - t_abs * grad[i]).clamp(lower[i], upper[i]);
            }
        }
    } else {
        // No further improvement; use last breakpoint position.
        for i in 0..n {
            if !active[i] {
                x_c[i] = (x[i] - t_prev * grad[i]).clamp(lower[i], upper[i]);
            }
        }
    }

    project(&mut x_c, lower, upper);
    (x_c, active)
}

/// Subspace minimization: refine the Cauchy point using L-BFGS restricted to
/// the free variables (those not at bounds at the Cauchy point).
///
/// Returns a correction vector (same size as x); active-set components are zero.
fn subspace_minimization(
    x_cauchy: &[f64],
    grad: &[f64],
    active_set: &[bool],
    lower: &[f64],
    upper: &[f64],
    s_history: &VecDeque<Vec<f64>>,
    y_history: &VecDeque<Vec<f64>>,
) -> Vec<f64> {
    let n = x_cauchy.len();
    let free_indices: Vec<usize> = (0..n).filter(|&i| !active_set[i]).collect();
    let nf = free_indices.len();

    if nf == 0 || s_history.is_empty() {
        return vec![0.0; n];
    }

    // Gradient at x_cauchy restricted to free variables.
    let grad_free: Vec<f64> = free_indices.iter().map(|&i| grad[i]).collect();

    // Run L-BFGS two-loop recursion on the free subspace.
    // Build reduced s/y histories.
    let s_reduced: VecDeque<Vec<f64>> = s_history
        .iter()
        .map(|s| free_indices.iter().map(|&i| s[i]).collect())
        .collect();
    let y_reduced: VecDeque<Vec<f64>> = y_history
        .iter()
        .map(|y| free_indices.iter().map(|&i| y[i]).collect())
        .collect();

    let dir_free = lbfgs_direction(&grad_free, &s_reduced, &y_reduced);

    // Embed free-variable result back to full space.
    let mut correction = vec![0.0; n];
    for (k, &i) in free_indices.iter().enumerate() {
        // correction[i] = x_cauchy[i] + dir_free[k] - x_cauchy[i] = dir_free[k]
        // Clamp to stay inside bounds.
        let x_new = (x_cauchy[i] + dir_free[k]).clamp(lower[i], upper[i]);
        correction[i] = x_new - x_cauchy[i];
    }

    correction
}
