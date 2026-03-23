//! Line search algorithms for gradient-based optimization.
//!
//! Provides strong Wolfe line search (Nocedal & Wright Algorithm 3.5/3.6) with
//! Armijo backtracking as a fallback when the Wolfe zoom fails to bracket.

use crate::optimization::{Objective, OptimizationConfig};
use crate::param::ParamStore;
use crate::solver::bfgs::{dense_gradient, dot, vec_norm, write_x_to_store};

/// Strong Wolfe line search (Nocedal & Wright Algorithm 3.5/3.6).
///
/// Finds a step length α satisfying both conditions:
/// - Sufficient decrease (Armijo): `f(x + α·d) ≤ f(x) + c₁·α·(∇f·d)`
/// - Curvature: `|∇f(x + α·d)·d| ≤ c₂·|∇f(x)·d|`
///
/// Returns `(alpha, f_alpha)`.
pub fn strong_wolfe_search(
    objective: &dyn Objective,
    store: &mut ParamStore,
    param_ids: &[crate::id::ParamId],
    x: &[f64],
    direction: &[f64],
    f_x: f64,
    grad: &[f64],
    config: &OptimizationConfig,
) -> (f64, f64) {
    let n = x.len();
    let c1 = config.armijo_c1;
    let c2 = config.wolfe_c2;
    let dg0 = dot(grad, direction);

    // dg0 must be negative (descent direction) for Wolfe to make sense.
    if dg0 >= 0.0 {
        return (config.line_search_min_step, f_x);
    }

    const MAX_BRACKET_ITERS: usize = 10;

    let mut alpha_prev = 0.0;
    let mut f_prev = f_x;
    let mut alpha = 1.0;

    for i in 0..MAX_BRACKET_ITERS {
        let x_trial: Vec<f64> = x.iter().zip(direction).map(|(xi, di)| xi + alpha * di).collect();
        write_x_to_store(store, param_ids, &x_trial);
        let f_alpha = objective.value(store);

        // Armijo fails or function increased relative to previous bracket point.
        if f_alpha > f_x + c1 * alpha * dg0 || (i > 0 && f_alpha >= f_prev) {
            return zoom(
                objective, store, param_ids, x, direction, f_x, grad, dg0, c1, c2,
                alpha_prev, alpha, f_prev, f_alpha, n, config,
            );
        }

        let grad_alpha = dense_gradient(objective, store, param_ids, n);
        let dg_alpha = dot(&grad_alpha, direction);

        // Strong curvature condition satisfied.
        if dg_alpha.abs() <= c2 * dg0.abs() {
            debug_assert!(
                f_alpha <= f_x + c1 * alpha * dg0,
                "Armijo condition violated at accepted Wolfe step"
            );
            return (alpha, f_alpha);
        }

        // Slope at alpha is positive: minimum lies between alpha_prev and alpha.
        if dg_alpha >= 0.0 {
            return zoom(
                objective, store, param_ids, x, direction, f_x, grad, dg0, c1, c2,
                alpha, alpha_prev, f_alpha, f_prev, n, config,
            );
        }

        alpha_prev = alpha;
        f_prev = f_alpha;
        alpha *= 2.0;
    }

    // Bracketing exhausted — fall back to Armijo.
    let dg = dot(grad, direction);
    let alpha_armijo = armijo_search(objective, store, param_ids, x, direction, f_x, dg, config);
    write_x_to_store(store, param_ids, &x.iter().zip(direction).map(|(xi, di)| xi + alpha_armijo * di).collect::<Vec<_>>());
    let f_armijo = objective.value(store);
    (alpha_armijo, f_armijo)
}

/// Zoom phase of the strong Wolfe line search (Nocedal & Wright Algorithm 3.6).
///
/// Refines the bracket [alpha_lo, alpha_hi] by bisection until strong Wolfe
/// conditions are satisfied or the maximum iteration count is reached.
#[allow(clippy::too_many_arguments)]
fn zoom(
    objective: &dyn Objective,
    store: &mut ParamStore,
    param_ids: &[crate::id::ParamId],
    x: &[f64],
    direction: &[f64],
    f_x: f64,
    grad: &[f64],
    dg0: f64,
    c1: f64,
    c2: f64,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut f_lo: f64,
    mut _f_hi: f64,
    n: usize,
    config: &OptimizationConfig,
) -> (f64, f64) {
    const MAX_ZOOM_ITERS: usize = 20;

    let mut best_alpha = alpha_lo;
    let mut best_f = f_lo;

    for _ in 0..MAX_ZOOM_ITERS {
        // Bisect the bracket.
        let alpha_j = 0.5 * (alpha_lo + alpha_hi);

        let x_trial: Vec<f64> = x.iter().zip(direction).map(|(xi, di)| xi + alpha_j * di).collect();
        write_x_to_store(store, param_ids, &x_trial);
        let f_j = objective.value(store);

        // Track best Armijo-satisfying step for fallback.
        if f_j < best_f {
            best_alpha = alpha_j;
            best_f = f_j;
        }

        if f_j > f_x + c1 * alpha_j * dg0 || f_j >= f_lo {
            // Armijo violated or no improvement: shrink from above.
            _f_hi = f_j;
            alpha_hi = alpha_j;
        } else {
            let grad_j = dense_gradient(objective, store, param_ids, n);
            let dg_j = dot(&grad_j, direction);

            // Strong Wolfe curvature condition satisfied.
            if dg_j.abs() <= c2 * dg0.abs() {
                debug_assert!(
                    f_j <= f_x + c1 * alpha_j * dg0,
                    "Armijo condition violated at accepted zoom step"
                );
                return (alpha_j, f_j);
            }

            if dg_j * (alpha_hi - alpha_lo) >= 0.0 {
                _f_hi = f_lo;
                alpha_hi = alpha_lo;
            }
            f_lo = f_j;
            alpha_lo = alpha_j;
        }
    }

    // Zoom failed to find Wolfe step: return best Armijo step found.
    (best_alpha, best_f)
}

/// Armijo backtracking line search.
///
/// Finds α such that `f(x + α·d) ≤ f(x) + c₁·α·(∇f·d)`.
/// Returns α; does not update the store (caller is responsible).
pub fn armijo_search(
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
        let x_trial: Vec<f64> = x.iter().zip(direction).map(|(xi, di)| xi + alpha * di).collect();
        write_x_to_store(store, param_ids, &x_trial);
        let f_trial = objective.value(store);

        if f_trial <= f_x + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= backtrack;
        if alpha < min_step {
            return alpha;
        }
    }
}

/// Unified line search: tries strong Wolfe first, falls back to Armijo on failure.
///
/// Returns `(alpha, f_alpha)`.
pub fn line_search(
    objective: &dyn Objective,
    store: &mut ParamStore,
    param_ids: &[crate::id::ParamId],
    x: &[f64],
    direction: &[f64],
    f_x: f64,
    grad: &[f64],
    config: &OptimizationConfig,
) -> (f64, f64) {
    let dg = dot(grad, direction);

    // Only attempt Wolfe when direction is a descent direction.
    if dg < 0.0 {
        let (alpha, f_alpha) = strong_wolfe_search(
            objective, store, param_ids, x, direction, f_x, grad, config,
        );
        // Verify the result is an improving step; if not, fall back to Armijo.
        if f_alpha < f_x {
            return (alpha, f_alpha);
        }
    }

    // Armijo fallback.
    let alpha = armijo_search(objective, store, param_ids, x, direction, f_x, dg, config);
    let x_new: Vec<f64> = x.iter().zip(direction).map(|(xi, di)| xi + alpha * di).collect();
    write_x_to_store(store, param_ids, &x_new);
    let f_alpha = objective.value(store);
    (alpha, f_alpha)
}

/// Compute the Euclidean norm of a slice (re-exported for convenience in tests).
#[allow(dead_code)]
pub(crate) fn norm(v: &[f64]) -> f64 {
    vec_norm(v)
}
