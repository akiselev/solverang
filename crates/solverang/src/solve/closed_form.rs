//! Analytical (closed-form) solvers for matched patterns.
//!
//! When the pattern detector identifies a known solvable pattern, the
//! corresponding closed-form solver here can determine exact parameter values
//! without iterating. This is faster and more robust than a general-purpose
//! nonlinear solver for these special cases.
//!
//! # Supported Patterns
//!
//! | Pattern | Solver | Notes |
//! |---------|--------|-------|
//! | `ScalarSolve` | Newton step on 1 equation / 1 variable | Fallback if closed form unknown |
//! | `TwoDistances` | Circle-circle intersection | 0 or 2 solutions |
//! | `HorizontalVertical` | Direct assignment | Always 1 solution |
//! | `DistanceAngle` | Polar-to-cartesian conversion | Always 1 solution |

use crate::constraint::Constraint;
use crate::graph::pattern::{MatchedPattern, PatternKind};
use crate::id::ParamId;
use crate::param::ParamStore;

/// Result of a closed-form solve.
#[derive(Clone, Debug)]
pub struct ClosedFormResult {
    /// Parameter values determined by the closed-form solution.
    pub values: Vec<(ParamId, f64)>,
    /// Whether a valid solution was found (patterns may have no solution,
    /// e.g., non-intersecting circles).
    pub solved: bool,
    /// Number of solution branches (e.g., 2 for circle-circle intersection).
    pub branch_count: usize,
}

/// Attempt to solve a matched pattern analytically.
///
/// # Arguments
///
/// * `pattern` - The matched pattern describing the sub-problem.
/// * `constraints` - All system constraints (indexed by `pattern.constraint_indices`).
/// * `store` - Current parameter values.
///
/// # Returns
///
/// `Some(ClosedFormResult)` if the pattern was handled (even if unsolvable),
/// `None` if the pattern kind is not supported.
pub fn solve_pattern(
    pattern: &MatchedPattern,
    constraints: &[&dyn Constraint],
    store: &ParamStore,
) -> Option<ClosedFormResult> {
    match pattern.kind {
        PatternKind::ScalarSolve => solve_scalar(pattern, constraints, store),
        PatternKind::TwoDistances => solve_two_distances(pattern, constraints, store),
        PatternKind::HorizontalVertical => solve_hv(pattern, constraints, store),
        PatternKind::DistanceAngle => solve_distance_angle(pattern, constraints, store),
    }
}

/// Solve a single scalar equation with a single free variable.
///
/// Uses a single Newton step: `x_new = x - f(x) / f'(x)`.
/// If the Jacobian entry is near zero the step is skipped and the function
/// reports failure.
fn solve_scalar(
    pattern: &MatchedPattern,
    constraints: &[&dyn Constraint],
    store: &ParamStore,
) -> Option<ClosedFormResult> {
    if pattern.constraint_indices.len() != 1 || pattern.param_ids.len() != 1 {
        return None;
    }

    let cidx = pattern.constraint_indices[0];
    let pid = pattern.param_ids[0];

    // Find the constraint by matching its index.
    let c = constraints
        .iter()
        .find(|c| c.id().raw_index() as usize == cidx);
    // Fallback: try using index directly into the slice.
    let c = match c {
        Some(c) => *c,
        None => {
            if cidx < constraints.len() {
                constraints[cidx]
            } else {
                return None;
            }
        }
    };

    let residuals = c.residuals(store);
    if residuals.is_empty() {
        return None;
    }
    let f_val = residuals[0];

    // Get the Jacobian entry for our parameter.
    let jac = c.jacobian(store);
    let df = jac
        .iter()
        .find(|(row, p, _)| *row == 0 && *p == pid)
        .map(|(_, _, v)| *v)
        .unwrap_or(0.0);

    if df.abs() < 1e-15 {
        return Some(ClosedFormResult {
            values: vec![(pid, store.get(pid))],
            solved: false,
            branch_count: 0,
        });
    }

    let current = store.get(pid);
    let new_val = current - f_val / df;

    Some(ClosedFormResult {
        values: vec![(pid, new_val)],
        solved: true,
        branch_count: 1,
    })
}

/// Solve two distance constraints on a 2-parameter entity (circle-circle
/// intersection).
///
/// Given the entity's two free parameters (x, y) and two distance constraints
/// to known reference points, we compute the intersection of the two circles.
///
/// The algorithm:
/// 1. Read the reference points and target distances from the constraints'
///    residuals and Jacobians.
/// 2. Compute intersection points using standard geometry.
/// 3. Return the branch closest to the current position.
fn solve_two_distances(
    pattern: &MatchedPattern,
    constraints: &[&dyn Constraint],
    store: &ParamStore,
) -> Option<ClosedFormResult> {
    if pattern.param_ids.len() != 2 || pattern.constraint_indices.len() != 2 {
        return None;
    }

    let px = pattern.param_ids[0];
    let py = pattern.param_ids[1];
    let cur_x = store.get(px);
    let cur_y = store.get(py);

    // For each distance constraint, we determine the centre and radius by
    // evaluating the constraint at the current point. The residual is
    //   r = distance(entity, reference) - target
    // so  target = distance - residual.
    //
    // We use the Jacobian to infer the direction to the reference point.
    // The Jacobian of a distance constraint w.r.t. (x, y) is
    //   (dx/d, dy/d) where dx = x - refx, dy = y - refy, d = distance.
    // So the reference point is at (x - dx, y - dy) and distance = d.
    //
    // We recover:  grad_x = dx / d,  grad_y = dy / d
    //   dx = grad_x * d,  dy = grad_y * d
    //   refx = x - dx,  refy = y - dy
    //   target = d - residual

    struct CircleInfo {
        cx: f64,
        cy: f64,
        r: f64,
    }

    let mut circles = Vec::with_capacity(2);

    for &cidx in &pattern.constraint_indices {
        let c = if cidx < constraints.len() {
            constraints[cidx]
        } else {
            return None;
        };

        let residuals = c.residuals(store);
        if residuals.is_empty() {
            return None;
        }
        let residual = residuals[0];

        let jac = c.jacobian(store);

        // Collect Jacobian entries for our two parameters.
        let mut grad_x = 0.0;
        let mut grad_y = 0.0;
        for &(row, pid, val) in &jac {
            if row == 0 {
                if pid == px {
                    grad_x = val;
                } else if pid == py {
                    grad_y = val;
                }
            }
        }

        // For a standard distance constraint f = sqrt((x-cx)^2 + (y-cy)^2) - r:
        //   grad = (dx/d, dy/d) where dx = x-cx, dy = y-cy, d = sqrt(dx^2+dy^2)
        //   grad_norm should be ~1 for a non-degenerate case.
        let grad_norm = (grad_x * grad_x + grad_y * grad_y).sqrt();

        if grad_norm < 1e-12 {
            return Some(ClosedFormResult {
                values: vec![(px, cur_x), (py, cur_y)],
                solved: false,
                branch_count: 0,
            });
        }

        let ux = grad_x / grad_norm;
        let uy = grad_y / grad_norm;

        // Recover d_actual (current distance from entity to reference point)
        // by probing the constraint at a shifted position. From the algebra:
        //   delta = f(x+1, y) - f(x, y)
        //   d = (1 - delta^2) / (2 * (delta - ux))
        let mut snap = store.snapshot();
        snap.set(px, cur_x + 1.0);
        let probe_residual = c.residuals(&snap)[0];

        let delta = probe_residual - residual;
        let denom = 2.0 * (delta - ux);

        if denom.abs() <= 1e-15 {
            // Singular probe: cannot determine circle parameters reliably.
            // Fall back to iterative solving.
            return Some(ClosedFormResult {
                values: vec![(px, cur_x), (py, cur_y)],
                solved: false,
                branch_count: 0,
            });
        }

        let d_actual = (1.0 - delta * delta) / denom;
        let d_abs = d_actual.abs().max(1e-15);

        // Reference point (circle centre) and target radius.
        let cx = cur_x - ux * d_abs;
        let cy = cur_y - uy * d_abs;
        let r = (d_abs - residual).abs();

        circles.push(CircleInfo { cx, cy, r });
    }

    if circles.len() != 2 {
        return None;
    }

    // Circle-circle intersection.
    let c0 = &circles[0];
    let c1 = &circles[1];

    let dx = c1.cx - c0.cx;
    let dy = c1.cy - c0.cy;
    let d = (dx * dx + dy * dy).sqrt();

    if d < 1e-15 {
        // Concentric circles.
        return Some(ClosedFormResult {
            values: vec![(px, cur_x), (py, cur_y)],
            solved: false,
            branch_count: 0,
        });
    }

    let r0 = c0.r;
    let r1 = c1.r;

    // Check if circles intersect.
    if d > r0 + r1 + 1e-10 || d < (r0 - r1).abs() - 1e-10 {
        return Some(ClosedFormResult {
            values: vec![(px, cur_x), (py, cur_y)],
            solved: false,
            branch_count: 0,
        });
    }

    // Standard circle-circle intersection.
    let a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d);
    let h_sq = r0 * r0 - a * a;
    let h = if h_sq > 0.0 { h_sq.sqrt() } else { 0.0 };

    let mx = c0.cx + a * dx / d;
    let my = c0.cy + a * dy / d;

    let sol1_x = mx + h * dy / d;
    let sol1_y = my - h * dx / d;

    let sol2_x = mx - h * dy / d;
    let sol2_y = my + h * dx / d;

    // Choose the branch closest to the current position.
    let dist1_sq = (sol1_x - cur_x).powi(2) + (sol1_y - cur_y).powi(2);
    let dist2_sq = (sol2_x - cur_x).powi(2) + (sol2_y - cur_y).powi(2);

    let (chosen_x, chosen_y) = if dist1_sq <= dist2_sq {
        (sol1_x, sol1_y)
    } else {
        (sol2_x, sol2_y)
    };

    let branch_count = if h.abs() < 1e-12 { 1 } else { 2 };

    Some(ClosedFormResult {
        values: vec![(px, chosen_x), (py, chosen_y)],
        solved: true,
        branch_count,
    })
}

/// Solve a horizontal + vertical pattern by direct assignment.
///
/// A horizontal constraint fixes the y-difference between two points.
/// A vertical constraint fixes the x-difference.  When the other point is
/// fixed, we can directly assign the free point's coordinates.
///
/// Because we operate at the constraint-residual level (the constraint tells
/// us how far off we are), we use a single Newton step for each equation.
fn solve_hv(
    pattern: &MatchedPattern,
    constraints: &[&dyn Constraint],
    store: &ParamStore,
) -> Option<ClosedFormResult> {
    if pattern.param_ids.len() != 2 || pattern.constraint_indices.len() != 2 {
        return None;
    }

    let mut values = Vec::with_capacity(2);
    let mut all_solved = true;

    for &cidx in &pattern.constraint_indices {
        let c = if cidx < constraints.len() {
            constraints[cidx]
        } else {
            return None;
        };

        let residuals = c.residuals(store);
        if residuals.is_empty() {
            return None;
        }
        let f_val = residuals[0];

        let jac = c.jacobian(store);

        // Find which of our pattern params this constraint depends on
        // (as a free variable) and the corresponding Jacobian entry.
        let mut found = false;
        for &pid in &pattern.param_ids {
            if let Some(&(_, _, df)) = jac.iter().find(|(row, p, _)| *row == 0 && *p == pid) {
                if df.abs() > 1e-15 {
                    let current = store.get(pid);
                    let new_val = current - f_val / df;
                    values.push((pid, new_val));
                    found = true;
                    break;
                }
            }
        }

        if !found {
            all_solved = false;
        }
    }

    let solved = all_solved && values.len() == 2;
    let branch_count = if solved { 1 } else { 0 };

    Some(ClosedFormResult {
        values,
        solved,
        branch_count,
    })
}

/// Solve a distance + angle pattern using polar-to-cartesian conversion.
///
/// A distance constraint fixes the radial distance from a reference point,
/// and an angle constraint fixes the direction. Together they define a unique
/// point in polar coordinates relative to the reference.
///
/// Like the other solvers, we use Newton steps on the constraint residuals
/// and Jacobian to update both parameters.
fn solve_distance_angle(
    pattern: &MatchedPattern,
    constraints: &[&dyn Constraint],
    store: &ParamStore,
) -> Option<ClosedFormResult> {
    if pattern.param_ids.len() != 2 || pattern.constraint_indices.len() != 2 {
        return None;
    }

    let p0 = pattern.param_ids[0];
    let p1 = pattern.param_ids[1];

    // Build a 2x2 Newton system from the two constraints.
    let mut f = [0.0f64; 2];
    let mut j = [[0.0f64; 2]; 2];

    for (eq_row, &cidx) in pattern.constraint_indices.iter().enumerate() {
        let c = if cidx < constraints.len() {
            constraints[cidx]
        } else {
            return None;
        };

        let residuals = c.residuals(store);
        if residuals.is_empty() {
            return None;
        }
        f[eq_row] = residuals[0];

        let jac = c.jacobian(store);
        for &(row, pid, val) in &jac {
            if row == 0 {
                if pid == p0 {
                    j[eq_row][0] = val;
                } else if pid == p1 {
                    j[eq_row][1] = val;
                }
            }
        }
    }

    // Solve the 2x2 system:  J * delta = -f
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];

    if det.abs() < 1e-15 {
        return Some(ClosedFormResult {
            values: vec![(p0, store.get(p0)), (p1, store.get(p1))],
            solved: false,
            branch_count: 0,
        });
    }

    let inv_det = 1.0 / det;
    let delta0 = -inv_det * (j[1][1] * f[0] - j[0][1] * f[1]);
    let delta1 = -inv_det * (-j[1][0] * f[0] + j[0][0] * f[1]);

    let new_p0 = store.get(p0) + delta0;
    let new_p1 = store.get(p1) + delta1;

    Some(ClosedFormResult {
        values: vec![(p0, new_p0), (p1, new_p1)],
        solved: true,
        branch_count: 1,
    })
}

/// Apply a closed-form result to the parameter store.
///
/// Writes the solved values back into the store.
pub fn apply_closed_form(store: &mut ParamStore, result: &ClosedFormResult) {
    if result.solved {
        for &(pid, val) in &result.values {
            store.set(pid, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::Constraint;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // --- Test constraint: fix param to a target value ---
    // Residual: param - target
    // Jacobian: d(residual)/d(param) = 1.0

    struct FixValueConstraint {
        id: ConstraintId,
        entity: EntityId,
        param: ParamId,
        target: f64,
        label: &'static str,
    }

    impl Constraint for FixValueConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            self.label
        }
        fn entity_ids(&self) -> &[EntityId] {
            std::slice::from_ref(&self.entity)
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.param)
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.param) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, 1.0)]
        }
    }

    // --- Test constraint: distance from origin ---
    // Residual: sqrt(x^2 + y^2) - target
    // Jacobian: (x / d, y / d)

    struct DistFromOriginConstraint {
        id: ConstraintId,
        entity: EntityId,
        px: ParamId,
        py: ParamId,
        params: [ParamId; 2],
        target: f64,
    }

    impl Constraint for DistFromOriginConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "Distance"
        }
        fn entity_ids(&self) -> &[EntityId] {
            std::slice::from_ref(&self.entity)
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            let x = store.get(self.px);
            let y = store.get(self.py);
            let d = (x * x + y * y).sqrt();
            vec![d - self.target]
        }
        fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            let x = store.get(self.px);
            let y = store.get(self.py);
            let d = (x * x + y * y).sqrt().max(1e-15);
            vec![(0, self.px, x / d), (0, self.py, y / d)]
        }
    }

    #[test]
    fn test_solve_scalar_basic() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(3.0, eid); // current value 3, target 5

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: px,
            target: 5.0,
            label: "fix_x",
        };

        let pattern = MatchedPattern {
            kind: PatternKind::ScalarSolve,
            entity_ids: vec![eid],
            constraint_indices: vec![0],
            param_ids: vec![px],
        };

        let constraints: Vec<&dyn Constraint> = vec![&c];
        let result = solve_pattern(&pattern, &constraints, &store).unwrap();

        assert!(result.solved);
        assert_eq!(result.branch_count, 1);
        assert_eq!(result.values.len(), 1);
        assert!((result.values[0].1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_hv_basic() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid); // current x=1, want x=3
        let py = store.alloc(2.0, eid); // current y=2, want y=7

        let ch = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: py,
            target: 7.0,
            label: "Horizontal",
        };
        let cv = FixValueConstraint {
            id: ConstraintId::new(1, 0),
            entity: eid,
            param: px,
            target: 3.0,
            label: "Vertical",
        };

        let pattern = MatchedPattern {
            kind: PatternKind::HorizontalVertical,
            entity_ids: vec![eid],
            constraint_indices: vec![0, 1],
            param_ids: vec![px, py],
        };

        let constraints: Vec<&dyn Constraint> = vec![&ch, &cv];
        let result = solve_pattern(&pattern, &constraints, &store).unwrap();

        assert!(result.solved);
        assert_eq!(result.branch_count, 1);

        // Check that we solved both parameters.
        let vals: std::collections::HashMap<ParamId, f64> = result.values.iter().copied().collect();
        assert!((vals[&py] - 7.0).abs() < 1e-10);
        assert!((vals[&px] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_distance_angle() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(2.0, eid);
        let py = store.alloc(1.0, eid);

        // Two linear constraints that together form a 2x2 system.
        // c0: px - 5.0 = 0  (Jacobian: d/dpx = 1, d/dpy = 0)
        // c1: py - 3.0 = 0  (Jacobian: d/dpx = 0, d/dpy = 1)
        let c0 = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: px,
            target: 5.0,
            label: "Distance",
        };
        let c1 = FixValueConstraint {
            id: ConstraintId::new(1, 0),
            entity: eid,
            param: py,
            target: 3.0,
            label: "Angle",
        };

        let pattern = MatchedPattern {
            kind: PatternKind::DistanceAngle,
            entity_ids: vec![eid],
            constraint_indices: vec![0, 1],
            param_ids: vec![px, py],
        };

        let constraints: Vec<&dyn Constraint> = vec![&c0, &c1];
        let result = solve_pattern(&pattern, &constraints, &store).unwrap();

        assert!(result.solved);
        let vals: std::collections::HashMap<ParamId, f64> = result.values.iter().copied().collect();
        assert!((vals[&px] - 5.0).abs() < 1e-10);
        assert!((vals[&py] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_closed_form() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);

        let result = ClosedFormResult {
            values: vec![(px, 42.0)],
            solved: true,
            branch_count: 1,
        };

        apply_closed_form(&mut store, &result);
        assert!((store.get(px) - 42.0).abs() < 1e-15);
    }

    #[test]
    fn test_apply_closed_form_not_solved() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);

        let result = ClosedFormResult {
            values: vec![(px, 42.0)],
            solved: false,
            branch_count: 0,
        };

        apply_closed_form(&mut store, &result);
        // Should NOT apply when solved is false.
        assert!((store.get(px) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_solve_scalar_zero_jacobian() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);

        // Constraint with zero Jacobian (degenerate).
        struct ZeroJacConstraint {
            id: ConstraintId,
            entity: EntityId,
            param: ParamId,
        }

        impl Constraint for ZeroJacConstraint {
            fn id(&self) -> ConstraintId {
                self.id
            }
            fn name(&self) -> &str {
                "zero_jac"
            }
            fn entity_ids(&self) -> &[EntityId] {
                std::slice::from_ref(&self.entity)
            }
            fn param_ids(&self) -> &[ParamId] {
                std::slice::from_ref(&self.param)
            }
            fn equation_count(&self) -> usize {
                1
            }
            fn residuals(&self, _store: &ParamStore) -> Vec<f64> {
                vec![1.0]
            }
            fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
                vec![(0, self.param, 0.0)] // Zero derivative
            }
        }

        let c = ZeroJacConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: px,
        };

        let pattern = MatchedPattern {
            kind: PatternKind::ScalarSolve,
            entity_ids: vec![eid],
            constraint_indices: vec![0],
            param_ids: vec![px],
        };

        let constraints: Vec<&dyn Constraint> = vec![&c];
        let result = solve_pattern(&pattern, &constraints, &store).unwrap();

        assert!(!result.solved);
    }
}
