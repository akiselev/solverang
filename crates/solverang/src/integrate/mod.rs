//! DAE / ODE time-integration stack — the runtime half of the SINBAD M3
//! first-transient-result path.
//!
//! This module drives a first-order differential-algebraic system
//!
//! ```text
//! d/dt q(x, t) + g(x, t) = 0
//! ```
//!
//! expressed through the federation seam
//! [`numeric_contracts::DaeResidual`] (the reactive charge `q` via
//! [`charge`](numeric_contracts::DaeResidual::charge) /
//! [`mass_apply`](numeric_contracts::DaeResidual::mass_apply), the static/algebraic
//! term `g` via [`residual_at`](numeric_contracts::DaeResidual::residual_at), and the
//! Newton iteration matrix via
//! [`iteration_matrix`](numeric_contracts::DaeResidual::iteration_matrix)). It never
//! imports the physics-assembly crate: the integrator sees only `f64` slices and the
//! seam.
//!
//! # Methods
//!
//! | Method | Order | Role |
//! |--------|-------|------|
//! | [`Method::ImplicitEuler`] | 1 | L-stable BDF-1 workhorse / bootstrap |
//! | [`Method::Bdf2`] | 2 | variable-step BDF-2, the index-1 first-order-DAE workhorse |
//! | [`Method::GeneralizedAlpha`] | 2 | `ρ∞`-damped 2nd-order (the `cardan` shape) |
//!
//! Each forms its per-step nonlinear system from the seam and solves it by **reusing
//! solverang's globalized Newton + line search** ([`Solver`](crate::solver::Solver))
//! through the [`step`] module's [`Problem`](crate::problem::Problem) adapter — no
//! bespoke Newton loop. A [PI (predictive) step-size controller](controller) provides
//! local-error accept/reject/adapt (see [`IntegratorOptions::adaptive`]).
//!
//! # The per-step system (getting the form right from the seam docs)
//!
//! For BDF the reactive charge is differenced with the multistep stencil:
//!
//! ```text
//! G(x^{n+1}) = a0·q(t^{n+1}, x^{n+1}) + (a1·q^n + a2·q^{n−1}) + g(t^{n+1}, x^{n+1}) = 0
//! ```
//!
//! where `(a0, a1, a2)` are the variable-step BDF coefficients (`a0 = 1/h`,
//! `a1 = −1/h`, `a2 = 0` for implicit Euler). The Newton iteration matrix is
//! `a0·M + K = iteration_matrix(bdf(a0))` — exactly [`IntegratorCoeffs::bdf`]. For a
//! *linear* charge `q = M·x` this reduces to the `mass_apply((a0)(x^{n+1} − x^n))`
//! form; the charge-difference form above is the general (nonlinear-charge, singular-
//! mass index-1) version.
//!
//! For generalized-α the residual is enforced at the intermediate point
//! `(t + α_f h, x_{n+α_f})` with the mass action applied to `ẋ_{n+α_m}`, and the
//! iteration matrix is `generalized_alpha(α_m/(γh), 0, α_f)`.
//!
//! # Errors
//!
//! [`integrate_dae`] is panic-free. It always returns a [`Trajectory`] whose
//! [`status`](Trajectory::status) is [`IntegrateStatus::Failed`] on any typed failure
//! (Newton non-convergence, step underflow, a seam [`NumericError`], bad input); the
//! partial history up to the failure is preserved.
//!
//! # Deferred seams (documented follow-ups, not built here)
//!
//! - **Dense output** (a continuous interpolant per step) — only step endpoints are
//!   recorded today.
//! - **The event loop** ([`EventBearing`](numeric_contracts::EventBearing): crossing
//!   eval → root isolation → consistent reinit → restart). [`IntegrateStatus::Terminated`]
//!   is reserved for it.
//! - **Higher-order / order control**, Radau IIA / ESDIRK stage-Newton, the full DymNL
//!   robust globalization ladder (homotopy/continuation/scaling beyond line search),
//!   and **unified factor reuse** (one factored Jacobian amortized across
//!   Newton/adjoint/ROM via the numeric-contracts factor-reuse handle). Today each
//!   Newton iteration reassembles and refactorizes.
//!
//! # Example
//!
//! ```ignore
//! use solverang::integrate::{integrate_dae, IntegratorOptions, Method};
//! use numeric_contracts::Ctx;
//!
//! // `decay` implements numeric_contracts::DaeResidual<f64> for u' = -λu.
//! let ctx = Ctx::real_os_default();
//! let opts = IntegratorOptions::adaptive(Method::Bdf2, 1e-3);
//! let traj = integrate_dae(&decay, &ctx, (0.0, 1.0), &[1.0], &opts);
//! assert!(traj.is_completed());
//! ```

mod controller;
mod options;
mod result;
mod step;

pub use options::{IntegratorOptions, Method, PiControllerConfig};
pub use result::{IntegrateError, IntegrateStats, IntegrateStatus, Trajectory};

use numeric_contracts::{Ctx, DaeResidual};

use crate::solver::SolverConfig;
use controller::{wrms_norm, PiController};
use step::{algebraic_mask, bdf_step, gen_alpha_step, initial_rate, StepOutcome};

/// The order of the universal linear predictor used for the local-error estimate.
/// The estimate `x^{n+1} − x_pred` is dominated by this predictor's error, so the PI
/// controller is sized to it (not the method order) — sharper, method-matched error
/// estimators are a documented follow-up (see the [module docs](self)).
const PREDICTOR_ORDER: usize = 1;

/// Integrate a [`DaeResidual`] over `t_span = (t0, t1)` from initial state `x0`.
///
/// Returns a [`Trajectory`] of `(time, state)` samples (one per accepted step, plus
/// the initial point) and a [`IntegrateStatus`]. Never panics: a failed run returns
/// the partial history with `status = Failed(..)`.
///
/// See the [module docs](self) for the per-step formulation, the reuse of solverang's
/// Newton, and the deferred seams.
pub fn integrate_dae<D: DaeResidual<f64> + Sync>(
    dae: &D,
    ctx: &Ctx,
    t_span: (f64, f64),
    x0: &[f64],
    opts: &IntegratorOptions,
) -> Trajectory {
    let n = dae.n();
    let (t0, t_end) = t_span;

    // ---- input validation (typed, never panic) ----
    if x0.len() != n {
        return failed(
            vec![],
            vec![],
            IntegrateError::InvalidInput {
                reason: format!("x0.len()={} but problem dimension is {n}", x0.len()),
            },
        );
    }
    if opts.h0 <= 0.0 || !opts.h0.is_finite() {
        return failed(
            vec![t0],
            vec![x0.to_vec()],
            IntegrateError::InvalidInput {
                reason: format!("initial step h0={} must be finite and positive", opts.h0),
            },
        );
    }
    if !t0.is_finite() || !t_end.is_finite() || t_end < t0 {
        return failed(
            vec![t0],
            vec![x0.to_vec()],
            IntegrateError::InvalidInput {
                reason: format!("t_span=({t0}, {t_end}) must be finite with t1 >= t0"),
            },
        );
    }

    let mut times = vec![t0];
    let mut states = vec![x0.to_vec()];
    let mut stats = IntegrateStats::default();

    let eps = (t_end - t0).abs().max(1.0) * 1e-12;
    // Degenerate span: a single point.
    if t_end - t0 <= eps {
        return Trajectory {
            times,
            states,
            status: IntegrateStatus::Completed,
            stats,
        };
    }

    let newton_cfg = SolverConfig {
        max_iterations: opts.max_newton_iters,
        tolerance: opts.newton_tol,
        line_search: true,
        ..SolverConfig::default()
    };

    // ---- history state ----
    let mut t_prev = t0;
    let mut x_prev = x0.to_vec();
    let mut xdot_prev = initial_rate(dae, ctx, t0, x0);
    let mut prev2: Option<(f64, Vec<f64>)> = None; // (t^{n-1}, x^{n-1})
    let mut h_accepted: Option<f64> = None; // step t^{n-1} -> t^n
    let mut h = opts.h0;

    // Structurally algebraic components (zero mass rows) are excluded from the
    // local-error test when `suppress_algebraic` is set (default). A pure ODE has no
    // such rows, so the mask is all-`false` and nothing is suppressed.
    let suppress: Vec<bool> = if opts.suppress_algebraic {
        algebraic_mask(dae, ctx, t0, x0)
    } else {
        vec![false; n]
    };

    let mut controller = PiController::new(opts.controller, PREDICTOR_ORDER);

    // ---- main stepping loop ----
    let mut fatal: Option<IntegrateError> = None;
    'stepping: loop {
        let remaining = t_end - t_prev;
        if remaining <= eps {
            break;
        }
        if stats.accepted_steps >= opts.max_steps {
            fatal = Some(IntegrateError::StepLimitReached {
                t: t_prev,
                steps: opts.max_steps,
            });
            break;
        }

        // charges at the two most recent accepted states (BDF only).
        let is_bdf = matches!(opts.method, Method::ImplicitEuler | Method::Bdf2);
        let q_n = if is_bdf {
            match charge_at(dae, ctx, t_prev, &x_prev) {
                Ok(q) => Some(q),
                Err(e) => {
                    fatal = Some(e);
                    break;
                }
            }
        } else {
            None
        };
        let q_nm1 = if is_bdf {
            match &prev2 {
                Some((t2, x2)) => match charge_at(dae, ctx, *t2, x2) {
                    Ok(q) => Some(q),
                    Err(e) => {
                        fatal = Some(e);
                        break;
                    }
                },
                None => None,
            }
        } else {
            None
        };

        // whether this step uses the two-point BDF-2 stencil
        let use_bdf2 = matches!(opts.method, Method::Bdf2) && prev2.is_some();
        let h_prev_for_step = if use_bdf2 { h_accepted } else { None };

        let mut h_try = h.min(opts.max_step);
        let mut final_step = false;
        if h_try >= remaining - eps {
            h_try = remaining;
            final_step = true;
        }

        // ---- attempt loop (adaptive reject/retry) ----
        let committed: step::StepAttempt = 'attempt: loop {
            if h_try < opts.min_step && remaining > opts.min_step {
                fatal = Some(IntegrateError::StepSizeUnderflow {
                    t: t_prev,
                    h: h_try,
                });
                break 'stepping;
            }

            let x_pred = predict(&x_prev, prev2.as_ref(), &xdot_prev, h_try, h_accepted);

            let outcome = match opts.method {
                Method::ImplicitEuler => bdf_step(
                    dae,
                    ctx,
                    t_prev,
                    h_try,
                    None,
                    &x_prev,
                    q_n.as_deref().unwrap_or(&x_prev),
                    None,
                    &x_pred,
                    &newton_cfg,
                    opts.newton_rtol,
                ),
                Method::Bdf2 => bdf_step(
                    dae,
                    ctx,
                    t_prev,
                    h_try,
                    h_prev_for_step,
                    &x_prev,
                    q_n.as_deref().unwrap_or(&x_prev),
                    if use_bdf2 { q_nm1.as_deref() } else { None },
                    &x_pred,
                    &newton_cfg,
                    opts.newton_rtol,
                ),
                Method::GeneralizedAlpha { rho_inf } => gen_alpha_step(
                    dae,
                    ctx,
                    t_prev,
                    h_try,
                    rho_inf,
                    &x_prev,
                    &xdot_prev,
                    &x_pred,
                    &newton_cfg,
                    opts.newton_rtol,
                ),
            };

            let outcome = match outcome {
                Ok(o) => o,
                Err(e) => {
                    fatal = Some(e);
                    break 'stepping;
                }
            };

            match outcome {
                StepOutcome::Converged(att) => {
                    stats.newton_iterations += att.newton_iters;
                    stats.residual_evals += att.residual_evals;
                    stats.jacobian_evals += att.jacobian_evals;

                    if opts.adaptive {
                        let err = wrms_norm(
                            &att.local_error,
                            &x_prev,
                            &att.x_new,
                            opts.atol,
                            opts.rtol,
                            &suppress,
                        );
                        if err > 1.0 && h_try > opts.min_step && !final_step {
                            // reject, shrink, retry
                            stats.rejected_steps += 1;
                            let f = controller.reject_factor(err);
                            h_try = (h_try * f).max(opts.min_step);
                            if h_try >= remaining - eps {
                                h_try = remaining;
                                final_step = true;
                            }
                            continue 'attempt;
                        }
                        // accept
                        let f = controller.accept_factor(err);
                        h = (h_try * f).clamp(opts.min_step, opts.max_step);
                        break 'attempt att;
                    }
                    // fixed-step: accept unconditionally
                    break 'attempt att;
                }
                StepOutcome::NewtonFailed {
                    residual_norm,
                    newton_iters,
                    residual_evals,
                    jacobian_evals,
                } => {
                    stats.newton_iterations += newton_iters;
                    stats.residual_evals += residual_evals;
                    stats.jacobian_evals += jacobian_evals;
                    if opts.adaptive && h_try > opts.min_step {
                        // shrink hard and retry
                        stats.rejected_steps += 1;
                        h_try = (h_try * 0.25).max(opts.min_step);
                        final_step = false;
                        continue 'attempt;
                    }
                    fatal = Some(IntegrateError::NonConvergence {
                        t: t_prev,
                        h: h_try,
                        residual_norm,
                    });
                    break 'stepping;
                }
            }
        };

        // ---- commit the accepted step ----
        let t_new = t_prev + h_try;
        if !committed.x_new.iter().all(|v| v.is_finite()) {
            fatal = Some(IntegrateError::NonFiniteState { t: t_new });
            break;
        }
        prev2 = Some((t_prev, x_prev.clone()));
        h_accepted = Some(h_try);
        x_prev = committed.x_new.clone();
        xdot_prev = committed.xdot_new;
        t_prev = t_new;

        times.push(t_new);
        states.push(committed.x_new);
        stats.accepted_steps += 1;

        if final_step {
            break;
        }
    }

    let status = match fatal {
        Some(e) => IntegrateStatus::Failed(e),
        None => IntegrateStatus::Completed,
    };
    Trajectory {
        times,
        states,
        status,
        stats,
    }
}

/// Linear predictor / extrapolation for the Newton initial guess: quadratic-free
/// linear extrapolation through the last two accepted states when available, else a
/// forward step along the state derivative.
fn predict(
    x_prev: &[f64],
    prev2: Option<&(f64, Vec<f64>)>,
    xdot_prev: &[f64],
    h: f64,
    h_accepted: Option<f64>,
) -> Vec<f64> {
    match (prev2, h_accepted) {
        (Some((_, x2)), Some(hp)) if hp > 0.0 => {
            let r = h / hp;
            (0..x_prev.len())
                .map(|i| x_prev[i] + r * (x_prev[i] - x2[i]))
                .collect()
        }
        _ => (0..x_prev.len())
            .map(|i| x_prev[i] + h * xdot_prev[i])
            .collect(),
    }
}

fn charge_at<D: DaeResidual<f64> + Sync>(
    dae: &D,
    ctx: &Ctx,
    t: f64,
    x: &[f64],
) -> Result<Vec<f64>, IntegrateError> {
    let mut q = vec![0.0; dae.n()];
    dae.charge(ctx, t, x, &mut q)?;
    Ok(q)
}

fn failed(times: Vec<f64>, states: Vec<Vec<f64>>, error: IntegrateError) -> Trajectory {
    Trajectory {
        times,
        states,
        status: IntegrateStatus::Failed(error),
        stats: IntegrateStats::default(),
    }
}
