//! Verification suite for the DAE/ODE time-integration stack (`solverang::integrate`).
//!
//! The `DaeResidual<f64>` impls here are hand-built manufactured problems — the
//! integrator sees only the `numeric_contracts` seam, never a physics crate.
//!
//! Covered:
//! - linear scalar decay `u' = -λu` accuracy at fixed and adaptive steps;
//! - stiff-system A-stability (implicit methods stay bounded at large `hλ`);
//! - order of convergence (BDF-2 ≈ 4×, implicit Euler ≈ 2×, generalized-α ≈ 4×);
//! - a linear index-1 DAE with a singular mass row;
//! - the PI controller rejecting an overshooting step and retrying smaller;
//! - typed, panic-free error reporting.

use numeric_contracts::{
    CooMatrix, Ctx, DaeIndex, DaeResidual, IntegratorCoeffs, Jacobian, NumericError, SparseMatrix,
};
use solverang::integrate::{
    integrate_dae, IntegrateError, IntegrateStatus, IntegratorOptions, Method,
};

fn ctx() -> Ctx {
    Ctx::real_os_default()
}

// ===========================================================================
// Manufactured DAE residuals
// ===========================================================================

/// Scalar linear decay: `q = u`, `g = λu` ⇒ `u' + λu = 0`, exact `u = u0 e^{-λt}`.
struct ScalarDecay {
    lambda: f64,
}

impl Jacobian<f64> for ScalarDecay {
    fn n(&self) -> usize {
        1
    }
    fn residual(&self, ctx: &Ctx, x: &[f64], r: &mut [f64]) -> Result<(), NumericError> {
        self.residual_at(ctx, 0.0, x, r)
    }
    fn assemble_into(
        &self,
        _ctx: &Ctx,
        _x: &[f64],
        j: &mut SparseMatrix<f64>,
    ) -> Result<(), NumericError> {
        let mut coo = CooMatrix::<f64>::new(1, 1);
        coo.push(0, 0, self.lambda);
        *j = coo.finish_csr();
        Ok(())
    }
}

impl DaeResidual<f64> for ScalarDecay {
    fn residual_at(
        &self,
        _c: &Ctx,
        _t: f64,
        x: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError> {
        out[0] = self.lambda * x[0];
        Ok(())
    }
    fn charge(&self, _c: &Ctx, _t: f64, x: &[f64], out: &mut [f64]) -> Result<(), NumericError> {
        out[0] = x[0];
        Ok(())
    }
    fn mass_apply(
        &self,
        _c: &Ctx,
        _t: f64,
        _x: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError> {
        out[0] = v[0]; // M = 1
        Ok(())
    }
    fn iteration_matrix(
        &self,
        _c: &Ctx,
        _t: f64,
        _x: &[f64],
        coeffs: &IntegratorCoeffs<f64>,
        out: &mut SparseMatrix<f64>,
    ) -> Result<(), NumericError> {
        // J = mass·M + stiff·K = mass·1 + stiff·λ
        let mut coo = CooMatrix::<f64>::new(1, 1);
        coo.push(0, 0, coeffs.mass + coeffs.stiff * self.lambda);
        *out = coo.finish_csr();
        Ok(())
    }
    fn dae_index_hint(&self) -> DaeIndex {
        DaeIndex::Ode
    }
}

/// Diagonal stiff decay: `u_i' = -λ_i u_i`.
struct StiffDiagonal {
    lambdas: Vec<f64>,
}

impl Jacobian<f64> for StiffDiagonal {
    fn n(&self) -> usize {
        self.lambdas.len()
    }
    fn residual(&self, ctx: &Ctx, x: &[f64], r: &mut [f64]) -> Result<(), NumericError> {
        self.residual_at(ctx, 0.0, x, r)
    }
    fn assemble_into(
        &self,
        _ctx: &Ctx,
        _x: &[f64],
        j: &mut SparseMatrix<f64>,
    ) -> Result<(), NumericError> {
        let n = self.lambdas.len();
        let mut coo = CooMatrix::<f64>::new(n, n);
        for (i, &l) in self.lambdas.iter().enumerate() {
            coo.push(i, i, l);
        }
        *j = coo.finish_csr();
        Ok(())
    }
}

impl DaeResidual<f64> for StiffDiagonal {
    fn residual_at(
        &self,
        _c: &Ctx,
        _t: f64,
        x: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError> {
        for (i, &l) in self.lambdas.iter().enumerate() {
            out[i] = l * x[i];
        }
        Ok(())
    }
    fn charge(&self, _c: &Ctx, _t: f64, x: &[f64], out: &mut [f64]) -> Result<(), NumericError> {
        out.copy_from_slice(x);
        Ok(())
    }
    fn mass_apply(
        &self,
        _c: &Ctx,
        _t: f64,
        _x: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError> {
        out.copy_from_slice(v); // M = I
        Ok(())
    }
    fn iteration_matrix(
        &self,
        _c: &Ctx,
        _t: f64,
        _x: &[f64],
        coeffs: &IntegratorCoeffs<f64>,
        out: &mut SparseMatrix<f64>,
    ) -> Result<(), NumericError> {
        let n = self.lambdas.len();
        let mut coo = CooMatrix::<f64>::new(n, n);
        for (i, &l) in self.lambdas.iter().enumerate() {
            coo.push(i, i, coeffs.mass + coeffs.stiff * l);
        }
        *out = coo.finish_csr();
        Ok(())
    }
    fn dae_index_hint(&self) -> DaeIndex {
        DaeIndex::Ode
    }
}

/// Linear index-1 DAE with a singular mass row:
/// - differential row: `x0' + x0 = 0`  (charge `q0 = x0`, mass row `[1, 0]`),
/// - algebraic row:    `x1 - x0 = 0`   (charge `q1 = 0`,  mass row `[0, 0]`).
///
/// Exact (consistent init `x = [1, 1]`): `x0 = x1 = e^{-t}`.
struct Index1Dae;

impl Jacobian<f64> for Index1Dae {
    fn n(&self) -> usize {
        2
    }
    fn residual(&self, ctx: &Ctx, x: &[f64], r: &mut [f64]) -> Result<(), NumericError> {
        self.residual_at(ctx, 0.0, x, r)
    }
    fn assemble_into(
        &self,
        _ctx: &Ctx,
        _x: &[f64],
        j: &mut SparseMatrix<f64>,
    ) -> Result<(), NumericError> {
        // K = dg/dx = [[1, 0], [-1, 1]]
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(1, 0, -1.0);
        coo.push(1, 1, 1.0);
        *j = coo.finish_csr();
        Ok(())
    }
}

impl DaeResidual<f64> for Index1Dae {
    fn residual_at(
        &self,
        _c: &Ctx,
        _t: f64,
        x: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError> {
        out[0] = x[0];
        out[1] = x[1] - x[0];
        Ok(())
    }
    fn charge(&self, _c: &Ctx, _t: f64, x: &[f64], out: &mut [f64]) -> Result<(), NumericError> {
        out[0] = x[0];
        out[1] = 0.0; // singular: no reactive charge on the algebraic row
        Ok(())
    }
    fn mass_apply(
        &self,
        _c: &Ctx,
        _t: f64,
        _x: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError> {
        out[0] = v[0]; // M = diag(1, 0)
        out[1] = 0.0;
        Ok(())
    }
    fn iteration_matrix(
        &self,
        _c: &Ctx,
        _t: f64,
        _x: &[f64],
        coeffs: &IntegratorCoeffs<f64>,
        out: &mut SparseMatrix<f64>,
    ) -> Result<(), NumericError> {
        // J = mass·M + stiff·K
        //   = mass·[[1,0],[0,0]] + stiff·[[1,0],[-1,1]]
        //   = [[mass+stiff, 0], [-stiff, stiff]]
        let mut coo = CooMatrix::<f64>::new(2, 2);
        coo.push(0, 0, coeffs.mass + coeffs.stiff);
        coo.push(1, 0, -coeffs.stiff);
        coo.push(1, 1, coeffs.stiff);
        *out = coo.finish_csr();
        Ok(())
    }
    fn dae_index_hint(&self) -> DaeIndex {
        DaeIndex::Index1
    }
}

// ===========================================================================
// Tests
// ===========================================================================

/// Global error of a scalar-decay run at the span end.
fn decay_end_error(method: Method, lambda: f64, h: f64, t_end: f64, adaptive: bool) -> f64 {
    let prob = ScalarDecay { lambda };
    let opts = if adaptive {
        IntegratorOptions::adaptive(method, h).with_tolerances(1e-8, 1e-10)
    } else {
        IntegratorOptions::fixed(method, h)
    };
    let traj = integrate_dae(&prob, &ctx(), (0.0, t_end), &[1.0], &opts);
    assert!(
        traj.is_completed(),
        "integration did not complete: {:?}",
        traj.status
    );
    let u = traj.last_state().unwrap()[0];
    let exact = (-lambda * t_end).exp();
    (u - exact).abs()
}

#[test]
fn scalar_decay_fixed_step_is_accurate() {
    // BDF-2 fixed step, moderate resolution: small global error.
    let err = decay_end_error(Method::Bdf2, 1.0, 1.0 / 64.0, 2.0, false);
    assert!(err < 1e-4, "BDF2 fixed-step decay error too large: {err:e}");
}

#[test]
fn scalar_decay_adaptive_is_accurate() {
    let err = decay_end_error(Method::Bdf2, 2.0, 1e-3, 2.0, true);
    assert!(err < 1e-5, "BDF2 adaptive decay error too large: {err:e}");
}

#[test]
fn implicit_euler_adaptive_is_accurate() {
    let err = decay_end_error(Method::ImplicitEuler, 1.5, 1e-3, 2.0, true);
    assert!(
        err < 1e-4,
        "implicit-Euler adaptive decay error too large: {err:e}"
    );
}

#[test]
fn stiff_system_is_a_stable() {
    // λ = [1, 1000]. With h = 0.1, h·λ2 = 100 — an explicit method would blow up
    // (|1 - 100| = 99 per step). Implicit Euler must stay bounded and decay.
    let prob = StiffDiagonal {
        lambdas: vec![1.0, 1000.0],
    };
    let opts = IntegratorOptions::fixed(Method::ImplicitEuler, 0.1);
    let traj = integrate_dae(&prob, &ctx(), (0.0, 1.0), &[1.0, 1.0], &opts);
    assert!(traj.is_completed(), "stiff run failed: {:?}", traj.status);

    // No blow-up: every recorded state stays within the initial magnitude.
    for st in &traj.states {
        assert!(st.iter().all(|v| v.is_finite()), "non-finite state");
        assert!(
            st[0].abs() <= 1.0 + 1e-9 && st[1].abs() <= 1.0 + 1e-9,
            "state grew: {st:?}"
        );
    }
    let end = traj.last_state().unwrap();
    // Slow mode ~ e^{-1}; implicit Euler order 1 at h=0.1 -> a few % off, but bounded.
    assert!(
        (end[0] - (-1.0_f64).exp()).abs() < 0.05,
        "slow mode off: {}",
        end[0]
    );
    // Fast mode is annihilated (1/101^10 ≈ 1e-20).
    assert!(end[1].abs() < 1e-6, "fast mode not damped: {}", end[1]);
}

#[test]
fn stiff_explicit_step_would_blow_up_sanity() {
    // Documents WHY the A-stability test matters: the explicit-Euler amplification
    // factor for the fast mode at this step is enormous, so a non-A-stable method
    // would diverge where implicit Euler stays bounded.
    let h = 0.1_f64;
    let lambda = 1000.0_f64;
    let explicit_factor = (1.0 - h * lambda).abs(); // |1 - 100| = 99
    assert!(explicit_factor > 10.0);
    let implicit_factor = 1.0 / (1.0 + h * lambda); // 1/101 < 1
    assert!(implicit_factor < 1.0);
}

/// Empirical convergence order from global errors at `h` and `h/2`.
fn convergence_ratio(method: Method, h: f64) -> f64 {
    let lambda = 1.0;
    let t_end = 1.0;
    let e_coarse = decay_end_error(method, lambda, h, t_end, false);
    let e_fine = decay_end_error(method, lambda, h / 2.0, t_end, false);
    e_coarse / e_fine
}

#[test]
fn implicit_euler_is_first_order() {
    // First order -> halving h halves the error (ratio ≈ 2).
    let r = convergence_ratio(Method::ImplicitEuler, 1.0 / 50.0);
    assert!(
        (1.7..2.4).contains(&r),
        "implicit-Euler order ratio {r} not ≈ 2"
    );
}

#[test]
fn bdf2_is_second_order() {
    // Second order -> ratio ≈ 4.
    let r = convergence_ratio(Method::Bdf2, 1.0 / 40.0);
    assert!((3.2..4.8).contains(&r), "BDF2 order ratio {r} not ≈ 4");
}

#[test]
fn generalized_alpha_is_second_order() {
    let r = convergence_ratio(Method::GeneralizedAlpha { rho_inf: 0.5 }, 1.0 / 40.0);
    assert!((3.2..4.8).contains(&r), "gen-α order ratio {r} not ≈ 4");
}

#[test]
fn generalized_alpha_various_rho_are_second_order() {
    for rho in [0.0, 0.5, 1.0] {
        let r = convergence_ratio(Method::GeneralizedAlpha { rho_inf: rho }, 1.0 / 40.0);
        assert!(
            (3.0..5.0).contains(&r),
            "gen-α(ρ∞={rho}) order ratio {r} not ≈ 4"
        );
    }
}

#[test]
fn index1_dae_integrates_correctly() {
    // x = [e^{-t}, e^{-t}]; the algebraic row enforces x1 == x0 at every step.
    let prob = Index1Dae;
    let opts = IntegratorOptions::fixed(Method::ImplicitEuler, 0.01);
    let traj = integrate_dae(&prob, &ctx(), (0.0, 1.0), &[1.0, 1.0], &opts);
    assert!(traj.is_completed(), "index-1 run failed: {:?}", traj.status);

    // Algebraic constraint x1 - x0 = 0 holds tightly at every accepted state.
    for st in &traj.states {
        assert!(
            (st[1] - st[0]).abs() < 1e-8,
            "algebraic row violated: {st:?}"
        );
    }
    let end = traj.last_state().unwrap();
    let exact = (-1.0_f64).exp();
    assert!(
        (end[0] - exact).abs() < 0.02,
        "x0 off: {} vs {exact}",
        end[0]
    );
    assert!(
        (end[1] - exact).abs() < 0.02,
        "x1 off: {} vs {exact}",
        end[1]
    );
}

#[test]
fn index1_dae_bdf2_is_accurate() {
    let prob = Index1Dae;
    let opts = IntegratorOptions::adaptive(Method::Bdf2, 1e-3).with_tolerances(1e-8, 1e-10);
    let traj = integrate_dae(&prob, &ctx(), (0.0, 1.0), &[1.0, 1.0], &opts);
    assert!(
        traj.is_completed(),
        "index-1 BDF2 failed: {:?}",
        traj.status
    );
    let end = traj.last_state().unwrap();
    let exact = (-1.0_f64).exp();
    assert!((end[0] - exact).abs() < 1e-5, "x0 off: {}", end[0]);
    assert!((end[1] - exact).abs() < 1e-5, "x1 off: {}", end[1]);
}

#[test]
fn pi_controller_rejects_and_retries_overshoot() {
    // A deliberately huge first step (h0 = 0.8, λ = 5) overshoots the tolerance; the
    // predictor/corrector error estimate is large, so the controller must reject and
    // retry with a smaller step — then still complete accurately.
    let prob = ScalarDecay { lambda: 5.0 };
    let opts = IntegratorOptions::adaptive(Method::ImplicitEuler, 0.8).with_tolerances(1e-6, 1e-9);
    let traj = integrate_dae(&prob, &ctx(), (0.0, 2.0), &[1.0], &opts);

    assert!(
        traj.is_completed(),
        "adaptive run failed: {:?}",
        traj.status
    );
    assert!(
        traj.stats.rejected_steps >= 1,
        "expected at least one rejected step, got {}",
        traj.stats.rejected_steps
    );
    assert!(traj.stats.accepted_steps >= 1);
    // Still lands near the exact decayed value e^{-10}.
    let u = traj.last_state().unwrap()[0];
    let exact = (-10.0_f64).exp();
    assert!((u - exact).abs() < 1e-3, "final value off: {u} vs {exact}");
}

#[test]
fn adaptive_takes_fewer_steps_than_naive_fine_fixed() {
    // The controller grows the step where the solution is smooth, so an adaptive run
    // uses far fewer steps than a fixed run at the small initial step size.
    //
    // NOTE: the current local-error estimate uses an order-1 predictor, so the
    // controller effectively enforces order-1 *accuracy control* on BDF-2 (correct,
    // and more accurate than requested, but conservative on step growth). A
    // method-matched (order-2) error estimator is a documented follow-up; even so the
    // adaptive run is several times cheaper than the fixed-fine run.
    let prob = ScalarDecay { lambda: 1.0 };
    let opts = IntegratorOptions::adaptive(Method::Bdf2, 1e-4).with_tolerances(1e-6, 1e-9);
    let traj = integrate_dae(&prob, &ctx(), (0.0, 5.0), &[1.0], &opts);
    assert!(traj.is_completed());
    // A fixed run at h0 would take 5.0 / 1e-4 = 50_000 steps; adaptive is far fewer.
    assert!(
        traj.stats.accepted_steps < 20_000,
        "adaptive used too many steps: {}",
        traj.stats.accepted_steps
    );
    assert!(
        traj.stats.accepted_steps * 4 < 50_000,
        "not meaningfully adaptive"
    );
}

#[test]
fn dimension_mismatch_is_typed_error_not_panic() {
    let prob = ScalarDecay { lambda: 1.0 };
    let opts = IntegratorOptions::fixed(Method::ImplicitEuler, 0.1);
    let traj = integrate_dae(&prob, &ctx(), (0.0, 1.0), &[1.0, 2.0], &opts);
    assert!(!traj.is_completed());
    assert!(matches!(
        traj.status,
        IntegrateStatus::Failed(IntegrateError::InvalidInput { .. })
    ));
}

#[test]
fn reversed_span_is_typed_error() {
    let prob = ScalarDecay { lambda: 1.0 };
    let opts = IntegratorOptions::fixed(Method::ImplicitEuler, 0.1);
    let traj = integrate_dae(&prob, &ctx(), (1.0, 0.0), &[1.0], &opts);
    assert!(matches!(
        traj.status,
        IntegrateStatus::Failed(IntegrateError::InvalidInput { .. })
    ));
}

#[test]
fn degenerate_span_returns_initial_point() {
    let prob = ScalarDecay { lambda: 1.0 };
    let opts = IntegratorOptions::fixed(Method::ImplicitEuler, 0.1);
    let traj = integrate_dae(&prob, &ctx(), (3.0, 3.0), &[7.0], &opts);
    assert!(traj.is_completed());
    assert_eq!(traj.len(), 1);
    assert_eq!(traj.last_state().unwrap(), &[7.0]);
}

#[test]
fn trajectory_records_monotone_times_and_matching_states() {
    let prob = ScalarDecay { lambda: 1.0 };
    let opts = IntegratorOptions::fixed(Method::Bdf2, 0.05);
    let traj = integrate_dae(&prob, &ctx(), (0.0, 1.0), &[1.0], &opts);
    assert!(traj.is_completed());
    assert_eq!(traj.times.len(), traj.states.len());
    for w in traj.times.windows(2) {
        assert!(w[1] > w[0], "times not strictly increasing");
    }
    assert_eq!(*traj.times.first().unwrap(), 0.0);
    assert!((traj.times.last().unwrap() - 1.0).abs() < 1e-9);
    assert!(traj.stats.newton_iterations >= traj.stats.accepted_steps);
}
