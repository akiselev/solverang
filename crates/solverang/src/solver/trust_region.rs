//! Trust-region solver with dogleg and Steihaug-CG subproblem solvers.
//!
//! Minimizes a scalar objective using a trust-region framework:
//! - **Dogleg** subproblem for n < `tr_subproblem_threshold` (default 100).
//! - **Steihaug-CG** subproblem for n >= threshold (matrix-free, handles large problems).
//!
//! The Hessian approximation uses scaled identity `B ≈ γI` where
//! `γ = yᵀy / sᵀy` from the most recent L-BFGS pair.  The Newton point in
//! dogleg is obtained via the full L-BFGS two-loop recursion.

use std::collections::VecDeque;
use std::time::Instant;

use crate::optimization::{
    KktResidual, MultiplierStore, Objective, ObjectiveHessian, OptimizationConfig,
    OptimizationResult, OptimizationStatus,
};
use crate::param::ParamStore;
use crate::solver::bfgs::{
    dense_gradient, dot, lbfgs_direction, update_lbfgs_history, vec_norm, write_x_to_store,
};

/// Trust-region solver for unconstrained optimization.
pub struct TrustRegionSolver;

impl TrustRegionSolver {
    /// Solve an unconstrained optimization problem using an exact Hessian.
    ///
    /// Uses the exact Hessian from `objective.hessian_entries()` instead of the
    /// scaled-identity L-BFGS approximation, giving quadratic convergence near
    /// the solution for problems where the Hessian is available.
    pub fn solve_with_hessian(
        objective: &dyn ObjectiveHessian,
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
        let mut x: Vec<f64> = param_ids.iter().map(|&pid| store.get(pid)).collect();

        let mut delta = config.trust_region_init;
        let delta_max = config.trust_region_max;
        let eta = 0.1;

        write_x_to_store(store, param_ids, &x);
        let mut f = objective.value(store);
        let mut grad = dense_gradient(objective, store, param_ids, n);

        for iter in 0..config.max_outer_iterations {
            let grad_norm = vec_norm(&grad);

            let dual_check = if config.relative_tolerance {
                grad_norm / (1.0_f64).max(f.abs())
            } else {
                grad_norm
            };
            if dual_check < config.dual_tolerance {
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

            // Build dense Hessian matrix from sparse entries.
            let hessian = build_dense_hessian(objective, store, param_ids, n);

            let step = if n < config.tr_subproblem_threshold {
                dogleg_step_exact(&grad, &hessian, delta, n)
            } else {
                steihaug_cg_exact(&grad, &hessian, delta, n)
            };

            let x_trial: Vec<f64> = x.iter().zip(&step).map(|(xi, si)| xi + si).collect();
            write_x_to_store(store, param_ids, &x_trial);
            let f_trial = objective.value(store);

            let actual_reduction = f - f_trial;
            let predicted_reduction = predicted_reduction_quadratic(&grad, &step, &hessian);

            let rho = if predicted_reduction <= 1e-30 {
                if actual_reduction > 0.0 { 1.0 } else { 0.0 }
            } else {
                actual_reduction / predicted_reduction
            };

            let step_norm = vec_norm(&step);
            if rho < 0.25 {
                delta = 0.25 * step_norm;
            } else if rho > 0.75 && (step_norm - delta).abs() < 1e-10 * delta {
                delta = (2.0 * delta).min(delta_max);
            }

            if rho > eta {
                let grad_new = dense_gradient(objective, store, param_ids, n);
                x = x_trial;
                f = f_trial;
                grad = grad_new;
            } else {
                write_x_to_store(store, param_ids, &x);
            }
        }

        let grad_norm = vec_norm(&grad);
        write_x_to_store(store, param_ids, &x);
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

    /// Solve an unconstrained optimization problem using a trust-region method.
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
        let mut x: Vec<f64> = param_ids.iter().map(|&pid| store.get(pid)).collect();

        let mut delta = config.trust_region_init;
        let delta_max = config.trust_region_max;
        let eta = 0.1;

        let m = config.lbfgs_memory;
        let mut s_history: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
        let mut y_history: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);

        write_x_to_store(store, param_ids, &x);
        let mut f = objective.value(store);
        let mut grad = dense_gradient(objective, store, param_ids, n);

        for iter in 0..config.max_outer_iterations {
            let grad_norm = vec_norm(&grad);

            let dual_check = if config.relative_tolerance {
                grad_norm / (1.0_f64).max(f.abs())
            } else {
                grad_norm
            };
            if dual_check < config.dual_tolerance {
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

            let step = if n < config.tr_subproblem_threshold {
                dogleg_step(&grad, &s_history, &y_history, delta, n)
            } else {
                steihaug_cg(&grad, &s_history, &y_history, delta, n)
            };

            let x_trial: Vec<f64> = x.iter().zip(&step).map(|(xi, si)| xi + si).collect();
            write_x_to_store(store, param_ids, &x_trial);
            let f_trial = objective.value(store);

            let actual_reduction = f - f_trial;
            // Use the linear predicted reduction -(gᵀp), which is always
            // positive for descent steps and avoids sign inconsistencies
            // that arise from the scaled-identity Hessian approximation.
            let predicted_reduction = predicted_reduction_linear(&grad, &step);

            let rho = if predicted_reduction <= 1e-30 {
                if actual_reduction > 0.0 { 1.0 } else { 0.0 }
            } else {
                actual_reduction / predicted_reduction
            };

            let step_norm = vec_norm(&step);
            if rho < 0.25 {
                delta = 0.25 * step_norm;
            } else if rho > 0.75 && (step_norm - delta).abs() < 1e-10 * delta {
                delta = (2.0 * delta).min(delta_max);
            }

            if rho > eta {
                let grad_new = dense_gradient(objective, store, param_ids, n);

                let s: Vec<f64> = step.clone();
                let y: Vec<f64> =
                    grad_new.iter().zip(&grad).map(|(a, b)| a - b).collect();
                update_lbfgs_history(&mut s_history, &mut y_history, s, y, m);

                x = x_trial;
                f = f_trial;
                grad = grad_new;
            } else {
                write_x_to_store(store, param_ids, &x);
            }
        }

        let grad_norm = vec_norm(&grad);
        write_x_to_store(store, param_ids, &x);
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

// ---------------------------------------------------------------------------
// Subproblem solvers
// ---------------------------------------------------------------------------

/// Dogleg step: combines Cauchy point and L-BFGS Newton point.
fn dogleg_step(
    grad: &[f64],
    s_history: &VecDeque<Vec<f64>>,
    y_history: &VecDeque<Vec<f64>>,
    delta: f64,
    _n: usize,
) -> Vec<f64> {
    let bg = lbfgs_hessian_vec_product(grad, s_history, y_history);
    let gtg = dot(grad, grad);
    let gtbg = dot(grad, &bg);

    let tau_c = if gtbg > 1e-30 {
        gtg / gtbg
    } else {
        let gn = vec_norm(grad);
        if gn > 1e-30 { delta / gn } else { 1.0 }
    };
    let p_cauchy: Vec<f64> = grad.iter().map(|g| -tau_c * g).collect();
    let cauchy_norm = vec_norm(&p_cauchy);

    if cauchy_norm >= delta {
        let scale = delta / cauchy_norm;
        return p_cauchy.iter().map(|p| scale * p).collect();
    }

    let p_newton = lbfgs_direction(grad, s_history, y_history);
    let newton_norm = vec_norm(&p_newton);

    if newton_norm <= delta {
        return p_newton;
    }

    let diff: Vec<f64> =
        p_newton.iter().zip(&p_cauchy).map(|(n, c)| n - c).collect();
    let a = dot(&diff, &diff);
    let b = 2.0 * dot(&p_cauchy, &diff);
    let c = dot(&p_cauchy, &p_cauchy) - delta * delta;
    let discriminant = b * b - 4.0 * a * c;
    let tau = if discriminant > 0.0 && a.abs() > 1e-30 {
        (-b + discriminant.sqrt()) / (2.0 * a)
    } else {
        1.0
    };
    let tau = tau.clamp(0.0, 1.0);

    p_cauchy.iter().zip(&diff).map(|(c, d)| c + tau * d).collect()
}

/// Steihaug-CG: truncated conjugate gradient for the trust-region subproblem.
fn steihaug_cg(
    grad: &[f64],
    s_history: &VecDeque<Vec<f64>>,
    y_history: &VecDeque<Vec<f64>>,
    delta: f64,
    n: usize,
) -> Vec<f64> {
    let max_cg_iters = n.min(200);
    let grad_norm = vec_norm(grad);
    let tol = grad_norm.min(0.5) * grad_norm;

    let mut z = vec![0.0; n];
    let mut r = grad.to_vec();
    let mut d: Vec<f64> = r.iter().map(|ri| -ri).collect();

    if grad_norm < tol {
        return z;
    }

    for _ in 0..max_cg_iters {
        let bd = lbfgs_hessian_vec_product(&d, s_history, y_history);
        let dtbd = dot(&d, &bd);

        if dtbd <= 0.0 {
            return to_trust_boundary(&z, &d, delta);
        }

        let rtr = dot(&r, &r);
        let alpha = rtr / dtbd;

        let z_new: Vec<f64> =
            z.iter().zip(&d).map(|(zi, di)| zi + alpha * di).collect();

        if vec_norm(&z_new) >= delta {
            return to_trust_boundary(&z, &d, delta);
        }

        let r_new: Vec<f64> =
            r.iter().zip(&bd).map(|(ri, bi)| ri + alpha * bi).collect();

        if vec_norm(&r_new) < tol {
            return z_new;
        }

        let rtr_new = dot(&r_new, &r_new);
        let beta = rtr_new / rtr;
        let d_new: Vec<f64> =
            r_new.iter().zip(&d).map(|(ri, di)| -ri + beta * di).collect();

        z = z_new;
        r = r_new;
        d = d_new;
    }

    z
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scaled-identity L-BFGS Hessian-vector product: B*v ≈ γ·v.
///
/// Uses `γ = yᵀy / sᵀy` from the most recent pair.  This is the standard
/// initial Hessian scaling and is sufficient for the trust-region subproblem
/// when the Newton point is computed separately via the full two-loop recursion.
fn lbfgs_hessian_vec_product(
    v: &[f64],
    s_history: &VecDeque<Vec<f64>>,
    y_history: &VecDeque<Vec<f64>>,
) -> Vec<f64> {
    let k = s_history.len();
    if k == 0 {
        return v.to_vec();
    }

    let last = k - 1;
    let yy = dot(&y_history[last], &y_history[last]);
    let sy = dot(&s_history[last], &y_history[last]);
    let gamma = if sy.abs() > 1e-30 { yy / sy } else { 1.0 };

    v.iter().map(|vi| gamma * vi).collect()
}

/// Find τ ≥ 0 such that ‖z + τ·d‖ = delta, then return z + τ·d.
fn to_trust_boundary(z: &[f64], d: &[f64], delta: f64) -> Vec<f64> {
    let a = dot(d, d);
    let b = 2.0 * dot(z, d);
    let c = dot(z, z) - delta * delta;
    let disc = b * b - 4.0 * a * c;
    let tau = if disc > 0.0 && a > 1e-30 {
        (-b + disc.sqrt()) / (2.0 * a)
    } else {
        0.0
    };
    z.iter().zip(d).map(|(zi, di)| zi + tau.max(0.0) * di).collect()
}

/// Predicted reduction of the linear model: m(0) - m(p) = -(gᵀp).
///
/// The linear term is always positive for descent steps and avoids sign
/// inconsistencies from the scaled-identity Hessian approximation `B ≈ γI`.
/// The quadratic correction `½ pᵀBp` is omitted because the `γI` scaling
/// is inconsistent with the L-BFGS inverse used for the Newton point,
/// which can produce negative predicted reductions even for good steps.
fn predicted_reduction_linear(grad: &[f64], step: &[f64]) -> f64 {
    -dot(grad, step)
}

// ---------------------------------------------------------------------------
// Exact-Hessian helpers (used by solve_with_hessian)
// ---------------------------------------------------------------------------

/// Build an n×n dense Hessian matrix from the objective's sparse lower-triangle entries.
fn build_dense_hessian(
    objective: &dyn ObjectiveHessian,
    store: &crate::param::ParamStore,
    param_ids: &[crate::id::ParamId],
    n: usize,
) -> Vec<Vec<f64>> {
    let mut h = vec![vec![0.0; n]; n];
    for (pi, pj, val) in objective.hessian_entries(store) {
        let Some(row) = param_ids.iter().position(|&p| p == pi) else { continue };
        let Some(col) = param_ids.iter().position(|&p| p == pj) else { continue };
        h[row][col] += val;
        if row != col {
            h[col][row] += val; // symmetrize
        }
    }
    h
}

/// Dense Hessian-vector product: H·v.
fn dense_hessian_vec(h: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    h.iter()
        .map(|row| row.iter().zip(v).map(|(hij, vi)| hij * vi).sum())
        .collect()
}

/// Dogleg step using exact dense Hessian.
fn dogleg_step_exact(grad: &[f64], hessian: &[Vec<f64>], delta: f64, _n: usize) -> Vec<f64> {
    let bg = dense_hessian_vec(hessian, grad);
    let gtg = dot(grad, grad);
    let gtbg = dot(grad, &bg);

    let tau_c = if gtbg > 1e-30 {
        gtg / gtbg
    } else {
        let gn = vec_norm(grad);
        if gn > 1e-30 { delta / gn } else { 1.0 }
    };
    let p_cauchy: Vec<f64> = grad.iter().map(|g| -tau_c * g).collect();
    let cauchy_norm = vec_norm(&p_cauchy);

    if cauchy_norm >= delta {
        let scale = delta / cauchy_norm;
        return p_cauchy.iter().map(|p| scale * p).collect();
    }

    // Newton step: solve H·p = -g.
    let p_newton = solve_linear_system(hessian, &grad.iter().map(|g| -g).collect::<Vec<_>>());
    let newton_norm = vec_norm(&p_newton);

    if newton_norm <= delta {
        return p_newton;
    }

    // Interpolate between Cauchy and Newton along the dogleg path.
    let diff: Vec<f64> =
        p_newton.iter().zip(&p_cauchy).map(|(n, c)| n - c).collect();
    let a = dot(&diff, &diff);
    let b = 2.0 * dot(&p_cauchy, &diff);
    let c = dot(&p_cauchy, &p_cauchy) - delta * delta;
    let discriminant = b * b - 4.0 * a * c;
    let tau = if discriminant > 0.0 && a.abs() > 1e-30 {
        (-b + discriminant.sqrt()) / (2.0 * a)
    } else {
        1.0
    };
    let tau = tau.clamp(0.0, 1.0);

    p_cauchy.iter().zip(&diff).map(|(c, d)| c + tau * d).collect()
}

/// Solve H·x = b using Cholesky-like approach or Gauss elimination.
///
/// Falls back to gradient direction if the system is singular or indefinite.
fn solve_linear_system(h: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return Vec::new();
    }

    // Gaussian elimination with partial pivoting.
    let mut a: Vec<Vec<f64>> = h.to_vec();
    let mut rhs = b.to_vec();

    for col in 0..n {
        // Find pivot.
        let mut max_val = a[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return b.to_vec(); // singular; fall back to steepest descent
        }

        a.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                let v = a[col][k];
                a[row][k] -= factor * v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-15 {
            x[i] = 0.0;
        } else {
            let sum: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
            x[i] = (rhs[i] - sum) / a[i][i];
        }
    }
    x
}

/// Steihaug-CG using exact dense Hessian.
fn steihaug_cg_exact(grad: &[f64], hessian: &[Vec<f64>], delta: f64, n: usize) -> Vec<f64> {
    let max_cg_iters = n.min(200);
    let grad_norm = vec_norm(grad);
    let tol = grad_norm.min(0.5) * grad_norm;

    let mut z = vec![0.0; n];
    let mut r = grad.to_vec();
    let mut d: Vec<f64> = r.iter().map(|ri| -ri).collect();

    if grad_norm < tol {
        return z;
    }

    for _ in 0..max_cg_iters {
        let bd = dense_hessian_vec(hessian, &d);
        let dtbd = dot(&d, &bd);

        if dtbd <= 0.0 {
            return to_trust_boundary(&z, &d, delta);
        }

        let rtr = dot(&r, &r);
        let alpha = rtr / dtbd;

        let z_new: Vec<f64> =
            z.iter().zip(&d).map(|(zi, di)| zi + alpha * di).collect();

        if vec_norm(&z_new) >= delta {
            return to_trust_boundary(&z, &d, delta);
        }

        let r_new: Vec<f64> =
            r.iter().zip(&bd).map(|(ri, bi)| ri + alpha * bi).collect();

        if vec_norm(&r_new) < tol {
            return z_new;
        }

        let rtr_new = dot(&r_new, &r_new);
        let beta = rtr_new / rtr;
        let d_new: Vec<f64> =
            r_new.iter().zip(&d).map(|(ri, di)| -ri + beta * di).collect();

        z = z_new;
        r = r_new;
        d = d_new;
    }

    z
}

/// Full quadratic predicted reduction: m(0) - m(p) = -(gᵀp + ½ pᵀHp).
fn predicted_reduction_quadratic(grad: &[f64], step: &[f64], hessian: &[Vec<f64>]) -> f64 {
    let hp = dense_hessian_vec(hessian, step);
    let gtP = dot(grad, step);
    let ptHp = dot(step, &hp);
    -(gtP + 0.5 * ptHp)
}
