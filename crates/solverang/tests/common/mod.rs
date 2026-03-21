//! Shared test utilities for property-based tests across constraint modules.

use solverang::constraint::Constraint;
use solverang::id::ParamId;
use solverang::param::ParamStore;
use std::collections::HashMap;

/// Central finite-difference check for a V3 `Constraint` operating on `ParamStore`.
///
/// Compares the analytical Jacobian from `constraint.jacobian(store)` against
/// `(f(x+h) - f(x-h)) / (2h)` for every (equation, param) pair.
///
/// Uses a scaled step `h = eps * max(1, |x|)` to avoid catastrophic cancellation
/// when coordinates are large (residuals >> FD difference). Tolerance is relative:
/// `|fd - ana| < tol * (1 + max(|fd|, |ana|))`.
///
/// The analytical Jacobian is indexed into a `HashMap<(usize, ParamId), f64>` for
/// O(1) lookup instead of linear scan.
pub fn check_jacobian_fd(
    constraint: &dyn Constraint,
    store: &ParamStore,
    eps: f64,
    tol: f64,
) -> bool {
    let params = constraint.param_ids().to_vec();
    let analytical = constraint.jacobian(store);

    // Build HashMap for O(1) lookup: (equation_index, param_id) -> value
    let mut ana_map: HashMap<(usize, ParamId), f64> = HashMap::new();
    for &(eq, pid, val) in &analytical {
        *ana_map.entry((eq, pid)).or_insert(0.0) += val;
    }

    // Evaluate residuals at the unperturbed point (for cancellation error estimate).
    let r_unperturbed = constraint.residuals(store);

    for eq in 0..constraint.equation_count() {
        for &pid in &params {
            let orig = store.get(pid);
            // Scale step with parameter magnitude to keep FD numerically stable.
            let h = eps * (1.0 + orig.abs());

            // Central finite difference
            let mut plus = store.clone();
            plus.set(pid, orig + h);
            let r_plus = constraint.residuals(&plus);

            let mut minus = store.clone();
            minus.set(pid, orig - h);
            let r_minus = constraint.residuals(&minus);

            let fd = (r_plus[eq] - r_minus[eq]) / (2.0 * h);

            let ana = ana_map.get(&(eq, pid)).copied().unwrap_or(0.0);

            // Relative + absolute tolerance: handles both large and small Jacobian entries.
            // Also account for cancellation error when |residual| >> |perturbation|:
            // subtracting two large nearly-equal f64 values loses significant digits.
            // The FD error floor is approximately |f| * eps_machine / h.
            let cancellation_error = r_unperturbed[eq].abs() * f64::EPSILON / h;
            let error = (fd - ana).abs();
            let scale = 1.0 + fd.abs().max(ana.abs()) + cancellation_error / tol;
            if error >= tol * scale {
                return false;
            }
        }
    }
    true
}
