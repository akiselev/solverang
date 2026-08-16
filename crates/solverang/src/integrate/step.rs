//! The per-step nonlinear solve: assemble the BDF / generalized-α stage residual from
//! the [`DaeResidual`](numeric_contracts::DaeResidual) seam and solve it by **reusing
//! solverang's globalized Newton** ([`Solver`](crate::solver::Solver)).
//!
//! The bridge is [`DaeStepProblem`], an adapter that implements solverang's
//! [`Problem`](crate::problem::Problem) trait over one integrator stage:
//!
//! - `residuals(x)` maps to the stage residual `G(x)` built from
//!   [`residual_at`](numeric_contracts::DaeResidual::residual_at) +
//!   [`charge`](numeric_contracts::DaeResidual::charge) /
//!   [`mass_apply`](numeric_contracts::DaeResidual::mass_apply);
//! - `jacobian(x)` maps to the COO triplets of
//!   [`iteration_matrix`](numeric_contracts::DaeResidual::iteration_matrix) for the
//!   stage [`IntegratorCoeffs`].
//!
//! Because `Problem::residuals` / `jacobian` have no error channel, a seam
//! [`NumericError`] is captured in a mutex and surfaced as non-finite values (which
//! solverang's finiteness guards reject); the step function reads the captured error
//! afterward and returns it typed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nalgebra::{DMatrix, DVector};
use numeric_contracts::{
    Ctx, DaeResidual, IntegratorCoeffs, NumericError, SparseIndex, SparseMatrix,
};

use crate::problem::Problem;
use crate::solver::{SolveResult, Solver, SolverConfig};

use super::result::IntegrateError;

/// The outcome of a single step attempt (before the local-error accept/reject test).
pub(crate) struct StepAttempt {
    /// The corrected state `x^{n+1}`.
    pub x_new: Vec<f64>,
    /// The state derivative `ẋ^{n+1}` (maintained for generalized-α and the predictor).
    pub xdot_new: Vec<f64>,
    /// The local-error estimate `x^{n+1} − x_pred` (predictor/corrector difference).
    pub local_error: Vec<f64>,
    /// Newton iterations the step consumed.
    pub newton_iters: usize,
    /// Residual evaluations the step consumed.
    pub residual_evals: usize,
    /// Iteration-matrix assemblies the step consumed.
    pub jacobian_evals: usize,
}

/// Whether the Newton solve converged; a non-convergence is *recoverable* (the driver
/// may shrink the step and retry) whereas a seam error is fatal.
pub(crate) enum StepOutcome {
    /// The step converged.
    Converged(StepAttempt),
    /// Newton did not reach tolerance; carries the last residual norm.
    NewtonFailed {
        /// Last residual norm reached.
        residual_norm: f64,
        /// Newton iterations consumed.
        newton_iters: usize,
        /// Residual evaluations consumed.
        residual_evals: usize,
        /// Iteration-matrix assemblies consumed.
        jacobian_evals: usize,
    },
}

/// The stage discretization the [`DaeStepProblem`] adapter represents.
enum Stage {
    /// A BDF stage: `G(x) = a0·q(t_np1,x) + hist + g(t_np1,x)`, iteration matrix
    /// `bdf(a0)`. `hist = Σ_{j≥1} a_j q^{n+1−j}` is precomputed (constant in `x`).
    Bdf { t_np1: f64, a0: f64, hist: Vec<f64> },
    /// A generalized-α stage evaluated at the intermediate `(t_αf, x_αf)`, applying
    /// the mass action to `ẋ_αm`. Iteration matrix `generalized_alpha(c_a, 0, c_d)`.
    GenAlpha {
        t_af: f64,
        h: f64,
        alpha_m: f64,
        alpha_f: f64,
        gamma: f64,
        c_a: f64,
        c_d: f64,
        x_n: Vec<f64>,
        xdot_n: Vec<f64>,
    },
}

/// Adapter implementing solverang's [`Problem`] over one integrator stage.
struct DaeStepProblem<'a, D: DaeResidual<f64> + Sync> {
    dae: &'a D,
    ctx: &'a Ctx,
    n: usize,
    stage: Stage,
    /// A dense `n×n` value-template handed to `iteration_matrix`; the seam impl may
    /// fill it or overwrite it with its own (sparse) matrix — either is read back via
    /// [`to_triplets`]. Value reuse / factor reuse is a documented follow-up.
    template: SparseMatrix<f64>,
    /// First seam error seen during a Newton pass (surfaced typed afterward).
    seam_error: Mutex<Option<NumericError>>,
    residual_evals: AtomicUsize,
    jacobian_evals: AtomicUsize,
}

impl<'a, D: DaeResidual<f64> + Sync> DaeStepProblem<'a, D> {
    fn record_error(&self, e: NumericError) {
        let mut slot = self.seam_error.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some(e);
        }
    }

    fn take_error(&self) -> Option<NumericError> {
        self.seam_error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
    }

    /// The magnitude of the stage residual's **constituent terms** at `x` (before their
    /// cancellation into `G(x)`) — the natural scale for a *relative* Newton tolerance.
    ///
    /// `G(x)` is a sum of large, near-cancelling terms (`a₀·q`, the BDF history, `g`),
    /// so its floating-point floor is `‖constituents‖·ε`, not `ε`. Scaling the Newton
    /// tolerance by this quantity (`newton_tol + newton_rtol·scale`) keeps the test
    /// above that floor for any state magnitude — tracking `a₀`, `‖J‖` and `‖x‖` — while
    /// vanishing toward the absolute floor for well-scaled problems. A seam failure
    /// yields `0.0` (falling back to the pure absolute tolerance); the real solve
    /// re-hits the seam and surfaces the error typed.
    fn residual_scale(&self, x: &[f64]) -> f64 {
        let n = self.n;
        let l2 = |acc: f64| acc.sqrt();
        match &self.stage {
            Stage::Bdf { t_np1, a0, hist } => {
                let mut q = vec![0.0; n];
                if self.dae.charge(self.ctx, *t_np1, x, &mut q).is_err() {
                    return 0.0;
                }
                let mut g = vec![0.0; n];
                if self.dae.residual_at(self.ctx, *t_np1, x, &mut g).is_err() {
                    return 0.0;
                }
                let acc = (0..n)
                    .map(|i| {
                        let term = (a0 * q[i]).abs() + hist[i].abs() + g[i].abs();
                        term * term
                    })
                    .sum();
                l2(acc)
            }
            Stage::GenAlpha {
                t_af,
                h,
                alpha_m,
                alpha_f,
                gamma,
                x_n,
                xdot_n,
                ..
            } => {
                let inv = 1.0 / (gamma * h);
                let c1 = (1.0 - gamma) / gamma;
                let xdot_np1: Vec<f64> = (0..n)
                    .map(|i| inv * (x[i] - x_n[i]) - c1 * xdot_n[i])
                    .collect();
                let x_af: Vec<f64> = (0..n).map(|i| x_n[i] + alpha_f * (x[i] - x_n[i])).collect();
                let xdot_am: Vec<f64> = (0..n)
                    .map(|i| xdot_n[i] + alpha_m * (xdot_np1[i] - xdot_n[i]))
                    .collect();
                let mut m = vec![0.0; n];
                if self
                    .dae
                    .mass_apply(self.ctx, *t_af, &x_af, &xdot_am, &mut m)
                    .is_err()
                {
                    return 0.0;
                }
                let mut g = vec![0.0; n];
                if self
                    .dae
                    .residual_at(self.ctx, *t_af, &x_af, &mut g)
                    .is_err()
                {
                    return 0.0;
                }
                let acc = (0..n)
                    .map(|i| {
                        let term = m[i].abs() + g[i].abs();
                        term * term
                    })
                    .sum();
                l2(acc)
            }
        }
    }
}

impl<D: DaeResidual<f64> + Sync> Problem for DaeStepProblem<'_, D> {
    fn name(&self) -> &str {
        "dae-step"
    }

    fn residual_count(&self) -> usize {
        self.n
    }

    fn variable_count(&self) -> usize {
        self.n
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        self.residual_evals.fetch_add(1, Ordering::Relaxed);
        let n = self.n;
        match &self.stage {
            Stage::Bdf { t_np1, a0, hist } => {
                let mut q = vec![0.0; n];
                if let Err(e) = self.dae.charge(self.ctx, *t_np1, x, &mut q) {
                    self.record_error(e);
                    return vec![f64::NAN; n];
                }
                let mut g = vec![0.0; n];
                if let Err(e) = self.dae.residual_at(self.ctx, *t_np1, x, &mut g) {
                    self.record_error(e);
                    return vec![f64::NAN; n];
                }
                (0..n).map(|i| a0 * q[i] + hist[i] + g[i]).collect()
            }
            Stage::GenAlpha {
                t_af,
                h,
                alpha_m,
                alpha_f,
                gamma,
                x_n,
                xdot_n,
                ..
            } => {
                // ẋ^{n+1} = (x − x_n)/(γ h) − ((1−γ)/γ) ẋ_n
                let inv = 1.0 / (gamma * h);
                let c1 = (1.0 - gamma) / gamma;
                let xdot_np1: Vec<f64> = (0..n)
                    .map(|i| inv * (x[i] - x_n[i]) - c1 * xdot_n[i])
                    .collect();
                let x_af: Vec<f64> = (0..n).map(|i| x_n[i] + alpha_f * (x[i] - x_n[i])).collect();
                let xdot_am: Vec<f64> = (0..n)
                    .map(|i| xdot_n[i] + alpha_m * (xdot_np1[i] - xdot_n[i]))
                    .collect();
                let mut m = vec![0.0; n];
                if let Err(e) = self
                    .dae
                    .mass_apply(self.ctx, *t_af, &x_af, &xdot_am, &mut m)
                {
                    self.record_error(e);
                    return vec![f64::NAN; n];
                }
                let mut g = vec![0.0; n];
                if let Err(e) = self.dae.residual_at(self.ctx, *t_af, &x_af, &mut g) {
                    self.record_error(e);
                    return vec![f64::NAN; n];
                }
                (0..n).map(|i| m[i] + g[i]).collect()
            }
        }
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_evals.fetch_add(1, Ordering::Relaxed);
        let n = self.n;
        let (t_eval, x_eval, coeffs): (f64, Vec<f64>, IntegratorCoeffs<f64>) = match &self.stage {
            Stage::Bdf { t_np1, a0, .. } => (*t_np1, x.to_vec(), IntegratorCoeffs::bdf(*a0)),
            Stage::GenAlpha {
                t_af,
                alpha_f,
                x_n,
                c_a,
                c_d,
                ..
            } => {
                let x_af: Vec<f64> = (0..n).map(|i| x_n[i] + alpha_f * (x[i] - x_n[i])).collect();
                (
                    *t_af,
                    x_af,
                    IntegratorCoeffs::generalized_alpha(*c_a, 0.0, *c_d),
                )
            }
        };
        let mut mat = self.template.clone();
        if let Err(e) = self
            .dae
            .iteration_matrix(self.ctx, t_eval, &x_eval, &coeffs, &mut mat)
        {
            self.record_error(e);
            // A single non-finite entry trips solverang's Jacobian finiteness guard.
            return vec![(0, 0, f64::NAN)];
        }
        to_triplets(&mat)
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        vec![0.0; self.n]
    }
}

/// Convert an assembled [`SparseMatrix`] (CSR or CSC) to COO `(row, col, value)`
/// triplets for solverang's dense Newton assembly. Explicit stored zeros are harmless
/// (solverang scatters into a dense `DMatrix`).
fn to_triplets<I: SparseIndex>(m: &SparseMatrix<f64, I>) -> Vec<(usize, usize, f64)> {
    let p = &*m.pattern;
    let mut out = Vec::with_capacity(m.nnz());
    match p.orientation {
        numeric_contracts::Orientation::Csr => {
            for i in 0..p.nrows {
                let start = p.offsets[i].to_usize();
                let end = p.offsets[i + 1].to_usize();
                for k in start..end {
                    out.push((i, p.indices[k].to_usize(), m.values[k]));
                }
            }
        }
        numeric_contracts::Orientation::Csc => {
            for j in 0..p.ncols {
                let start = p.offsets[j].to_usize();
                let end = p.offsets[j + 1].to_usize();
                for k in start..end {
                    out.push((p.indices[k].to_usize(), j, m.values[k]));
                }
            }
        }
    }
    out
}

/// A dense `n×n` CSR value-template (all entries present, zero-valued) handed to the
/// seam's `iteration_matrix`.
pub(crate) fn dense_template(n: usize) -> SparseMatrix<f64> {
    let mut coo = numeric_contracts::CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            coo.push(i, j, 0.0);
        }
    }
    coo.finish_csr()
}

/// Assemble the dense iteration matrix `mass·M + damp·C + stiff·K` for `coeffs`.
pub(crate) fn assemble_dense<D: DaeResidual<f64> + Sync>(
    dae: &D,
    ctx: &Ctx,
    t: f64,
    x: &[f64],
    coeffs: &IntegratorCoeffs<f64>,
) -> Result<DMatrix<f64>, IntegrateError> {
    let n = dae.n();
    let mut mat = dense_template(n);
    dae.iteration_matrix(ctx, t, x, coeffs, &mut mat)?;
    let mut dm = DMatrix::zeros(n, n);
    for (r, c, v) in to_triplets(&mat) {
        if r < n && c < n {
            dm[(r, c)] = v;
        }
    }
    Ok(dm)
}

/// Compute a consistent initial state derivative `ẋ_0` by solving `M(x_0)·ẋ_0 = −g(x_0)`
/// (the ODE `M ẋ = −g` at `t_0`). Used by generalized-α and by the first-step
/// predictor. For a singular mass (index-1 DAE) the least-squares solution is used;
/// full consistent-initialization is a documented follow-up.
pub(crate) fn initial_rate<D: DaeResidual<f64> + Sync>(
    dae: &D,
    ctx: &Ctx,
    t0: f64,
    x0: &[f64],
) -> Vec<f64> {
    let n = dae.n();
    let mass = IntegratorCoeffs {
        mass: 1.0,
        damp: 0.0,
        stiff: 0.0,
    };
    let m = match assemble_dense(dae, ctx, t0, x0, &mass) {
        Ok(m) => m,
        Err(_) => return vec![0.0; n], // matrix-free / unsupported: fall back to zero rate.
    };
    let mut g = vec![0.0; n];
    if dae.residual_at(ctx, t0, x0, &mut g).is_err() {
        return vec![0.0; n];
    }
    let rhs = DVector::from_iterator(n, g.iter().map(|v| -v));
    if let Some(sol) = m.clone().lu().solve(&rhs) {
        if sol.iter().all(|v| v.is_finite()) {
            return sol.as_slice().to_vec();
        }
    }
    // Singular / rank-deficient mass: least-squares consistent rate.
    let svd = m.svd(true, true);
    match svd.solve(&rhs, 1e-12) {
        Ok(sol) if sol.iter().all(|v| v.is_finite()) => sol.as_slice().to_vec(),
        _ => vec![0.0; n],
    }
}

/// Probe the (possibly singular) mass matrix `M = ∂q/∂x` for its **structurally
/// algebraic rows** — rows that are entirely zero, i.e. components with no reactive
/// charge (the algebraic constraints of an index-1 DAE, the `voltaic` non-reactive
/// nodes). Returns `is_algebraic[i] == true` when row `i` of `M` is all-zero.
///
/// These components carry no time-derivative, so the multistep predictor/corrector
/// difference on them is only O(h) (their rate cannot be recovered without index
/// reduction). Including them in the local-error test collapses the step size — the
/// standard remedy (SUNDIALS IDA's *suppress-alg*) is to exclude them, which the
/// driver does. A matrix-free block that cannot be probed yields "all differential"
/// (no suppression).
pub(crate) fn algebraic_mask<D: DaeResidual<f64> + Sync>(
    dae: &D,
    ctx: &Ctx,
    t0: f64,
    x0: &[f64],
) -> Vec<bool> {
    let n = dae.n();
    let mut is_alg = vec![true; n];
    let mut v = vec![0.0; n];
    let mut out = vec![0.0; n];
    for j in 0..n {
        v[j] = 1.0;
        if dae.mass_apply(ctx, t0, x0, &v, &mut out).is_err() {
            return vec![false; n]; // cannot probe: treat everything as differential
        }
        for i in 0..n {
            if out[i] != 0.0 {
                is_alg[i] = false;
            }
        }
        v[j] = 0.0;
    }
    is_alg
}

/// Variable-step BDF coefficients `(a0, a1, a2)` for the derivative stencil
/// `q̇^{n+1} ≈ a0·q^{n+1} + a1·q^n + a2·q^{n−1}`.
///
/// `h` is the current step `t^{n+1}−t^n`; `h_prev` is the previous accepted step
/// `t^n−t^{n−1}` (`None` on the bootstrap step → implicit Euler).
pub(crate) fn bdf_coeffs(h: f64, h_prev: Option<f64>) -> (f64, f64, f64) {
    match h_prev {
        None => (1.0 / h, -1.0 / h, 0.0), // BDF-1 (implicit Euler)
        Some(hp) => {
            // Second-order backward differentiation on a non-uniform grid.
            let a0 = (2.0 * h + hp) / (h * (h + hp));
            let a1 = -(h + hp) / (h * hp);
            let a2 = h / (hp * (h + hp));
            (a0, a1, a2)
        }
    }
}

/// The generalized-α (first-order form) stage parameters for a given `ρ∞`.
pub(crate) fn gen_alpha_params(rho_inf: f64) -> (f64, f64, f64) {
    let rho = rho_inf.clamp(0.0, 1.0);
    let alpha_m = (3.0 - rho) / (2.0 * (1.0 + rho));
    let alpha_f = 1.0 / (1.0 + rho);
    let gamma = 0.5 + alpha_m - alpha_f;
    (alpha_m, alpha_f, gamma)
}

/// Run one BDF step (Euler if `h_prev` is `None`, else variable-step BDF-2).
///
/// `q_n` / `q_nm1` are the reactive charges at the two most recent accepted states,
/// `x_n` is the previous accepted state (used to recover the next-step slope).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bdf_step<D: DaeResidual<f64> + Sync>(
    dae: &D,
    ctx: &Ctx,
    t_n: f64,
    h: f64,
    h_prev: Option<f64>,
    x_n: &[f64],
    q_n: &[f64],
    q_nm1: Option<&[f64]>,
    x_pred: &[f64],
    newton_cfg: &SolverConfig,
    newton_rtol: f64,
) -> Result<StepOutcome, IntegrateError> {
    let n = dae.n();
    let t_np1 = t_n + h;
    let (a0, a1, a2) = bdf_coeffs(h, h_prev);
    let mut hist = vec![0.0; n];
    for i in 0..n {
        hist[i] = a1 * q_n[i];
    }
    if let Some(q2) = q_nm1 {
        for i in 0..n {
            hist[i] += a2 * q2[i];
        }
    }
    let problem = DaeStepProblem {
        dae,
        ctx,
        n,
        stage: Stage::Bdf { t_np1, a0, hist },
        template: dense_template(n),
        seam_error: Mutex::new(None),
        residual_evals: AtomicUsize::new(0),
        jacobian_evals: AtomicUsize::new(0),
    };
    let (x_new, res_norm, iters) = solve_stage(&problem, x_pred, newton_cfg, newton_rtol)?;
    let x_n_owned = x_n.to_vec();
    finish(&problem, x_new, res_norm, iters, x_pred, move |x_new| {
        // ẋ^{n+1} ≈ (x^{n+1} − x^n) / h  (a first-order slope for the next predictor).
        (0..n).map(|i| (x_new[i] - x_n_owned[i]) / h).collect()
    })
}

/// Run one generalized-α step from `(x_n, ẋ_n)` over step `h`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gen_alpha_step<D: DaeResidual<f64> + Sync>(
    dae: &D,
    ctx: &Ctx,
    t_n: f64,
    h: f64,
    rho_inf: f64,
    x_n: &[f64],
    xdot_n: &[f64],
    x_pred: &[f64],
    newton_cfg: &SolverConfig,
    newton_rtol: f64,
) -> Result<StepOutcome, IntegrateError> {
    let n = dae.n();
    let (alpha_m, alpha_f, gamma) = gen_alpha_params(rho_inf);
    let t_af = t_n + alpha_f * h;
    let c_a = alpha_m / (gamma * h);
    let c_d = alpha_f;
    let problem = DaeStepProblem {
        dae,
        ctx,
        n,
        stage: Stage::GenAlpha {
            t_af,
            h,
            alpha_m,
            alpha_f,
            gamma,
            c_a,
            c_d,
            x_n: x_n.to_vec(),
            xdot_n: xdot_n.to_vec(),
        },
        template: dense_template(n),
        seam_error: Mutex::new(None),
        residual_evals: AtomicUsize::new(0),
        jacobian_evals: AtomicUsize::new(0),
    };
    let (x_new, res_norm, iters) = solve_stage(&problem, x_pred, newton_cfg, newton_rtol)?;
    let inv = 1.0 / (gamma * h);
    let c1 = (1.0 - gamma) / gamma;
    let x_n_owned = x_n.to_vec();
    let xdot_n_owned = xdot_n.to_vec();
    finish(&problem, x_new, res_norm, iters, x_pred, move |x_new| {
        // ẋ^{n+1} = (x^{n+1} − x^n)/(γ h) − ((1−γ)/γ) ẋ^n
        (0..n)
            .map(|i| inv * (x_new[i] - x_n_owned[i]) - c1 * xdot_n_owned[i])
            .collect()
    })
}

/// Solve the assembled stage with solverang's globalized Newton, mapping the result
/// onto `(x_new, residual_norm, iterations)` or a typed seam error.
///
/// The Newton convergence tolerance is the **mixed absolute+relative** form
/// `newton_cfg.tolerance + newton_rtol · scale`, where `scale` is the stage residual's
/// constituent magnitude at the predictor (see [`DaeStepProblem::residual_scale`]). This
/// keeps the test above the per-step cancellation floor for high-magnitude states
/// without loosening well-scaled problems.
fn solve_stage<D: DaeResidual<f64> + Sync>(
    problem: &DaeStepProblem<'_, D>,
    x_pred: &[f64],
    newton_cfg: &SolverConfig,
    newton_rtol: f64,
) -> Result<(Option<Vec<f64>>, f64, usize), IntegrateError> {
    let scale = problem.residual_scale(x_pred);
    let mut cfg = newton_cfg.clone();
    cfg.tolerance = newton_cfg.tolerance + newton_rtol * scale;
    let solver = Solver::new(cfg);
    let result = solver.solve(problem, x_pred);
    // A seam error captured mid-Newton is fatal and takes precedence.
    if let Some(e) = problem.take_error() {
        return Err(IntegrateError::Seam(e));
    }
    let iters = result.iterations().unwrap_or(newton_cfg.max_iterations);
    match result {
        SolveResult::Converged {
            solution,
            residual_norm,
            ..
        } => Ok((Some(solution), residual_norm, iters)),
        SolveResult::NotConverged { residual_norm, .. } => Ok((None, residual_norm, iters)),
        SolveResult::Failed { .. } => Ok((None, f64::INFINITY, iters)),
    }
}

/// Package a solved stage into a [`StepOutcome`], computing the local-error estimate
/// and the next-step derivative via `rate`.
fn finish<D, F>(
    problem: &DaeStepProblem<'_, D>,
    x_new: Option<Vec<f64>>,
    res_norm: f64,
    iters: usize,
    x_pred: &[f64],
    rate: F,
) -> Result<StepOutcome, IntegrateError>
where
    D: DaeResidual<f64> + Sync,
    F: FnOnce(&[f64]) -> Vec<f64>,
{
    let residual_evals = problem.residual_evals.load(Ordering::Relaxed);
    let jacobian_evals = problem.jacobian_evals.load(Ordering::Relaxed);
    match x_new {
        Some(x_new) if x_new.iter().all(|v| v.is_finite()) => {
            let local_error: Vec<f64> = (0..x_new.len()).map(|i| x_new[i] - x_pred[i]).collect();
            let xdot_new = rate(&x_new);
            Ok(StepOutcome::Converged(StepAttempt {
                x_new,
                xdot_new,
                local_error,
                newton_iters: iters,
                residual_evals,
                jacobian_evals,
            }))
        }
        _ => Ok(StepOutcome::NewtonFailed {
            residual_norm: res_norm,
            newton_iters: iters,
            residual_evals,
            jacobian_evals,
        }),
    }
}
