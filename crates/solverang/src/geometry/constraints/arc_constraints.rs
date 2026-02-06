//! Arc-specific constraints: endpoint matching, sweep angle, and point-on-arc.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Arc endpoint constraint: a point equals the arc evaluated at its start or end angle.
///
/// Enforces that a point lies at the start or end of an arc by requiring the point
/// to equal the arc's position at the specified angle.
///
/// # Entity Layout
/// - `point`: Point2D [x, y] — 2 params
/// - `arc`: Arc2D [cx, cy, r, start_angle, end_angle] — 5 params
///
/// # Equations (2D)
///
/// If `at_start` is true:
/// - `point.x = arc.cx + arc.r * cos(arc.start_angle)`
/// - `point.y = arc.cy + arc.r * sin(arc.start_angle)`
///
/// Otherwise:
/// - `point.x = arc.cx + arc.r * cos(arc.end_angle)`
/// - `point.y = arc.cy + arc.r * sin(arc.end_angle)`
///
/// This produces 2 equations.
///
/// # Jacobian
///
/// Let `θ = start_angle` or `end_angle`, `c = cos(θ)`, `s = sin(θ)`.
/// - `d/dpx = 1`, `d/dpy = 1`
/// - `d/dcx = -1`, `d/dcy = -1`
/// - `d/dr = -c` (for x eq), `-s` (for y eq)
/// - `d/dθ = r*s` (for x eq), `-r*c` (for y eq)
#[derive(Clone, Debug)]
pub struct ArcEndpointConstraint {
    id: ConstraintId,
    /// Point parameter range [x, y].
    point: ParamRange,
    /// Arc parameter range [cx, cy, r, start_angle, end_angle].
    arc: ParamRange,
    /// True if constraining to start, false if constraining to end.
    at_start: bool,
    /// Cached dependency list.
    dependencies: Vec<usize>,
}

impl ArcEndpointConstraint {
    /// Create a new arc endpoint constraint.
    ///
    /// # Arguments
    /// * `id` - Unique constraint identifier
    /// * `point` - Parameter range for the point (2 params)
    /// * `arc` - Parameter range for the arc (5 params: [cx, cy, r, start_angle, end_angle])
    /// * `at_start` - True to constrain to start, false to constrain to end
    ///
    /// # Panics
    /// Panics if point.count != 2 or arc.count != 5.
    pub fn new(
        id: ConstraintId,
        point: ParamRange,
        arc: ParamRange,
        at_start: bool,
    ) -> Self {
        assert_eq!(
            point.count, 2,
            "Point must be 2D (2 params), got {}",
            point.count
        );
        assert_eq!(arc.count, 5, "Arc must have 5 params, got {}", arc.count);

        let mut dependencies = Vec::with_capacity(7);
        dependencies.extend(point.iter());
        dependencies.extend(arc.iter());

        Self {
            id,
            point,
            arc,
            at_start,
            dependencies,
        }
    }
}

impl Constraint for ArcEndpointConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        if self.at_start {
            "ArcStartpoint"
        } else {
            "ArcEndpoint"
        }
    }

    fn equation_count(&self) -> usize {
        2
    }

    fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let px = params[self.point.start];
        let py = params[self.point.start + 1];

        let cx = params[self.arc.start];
        let cy = params[self.arc.start + 1];
        let r = params[self.arc.start + 2];
        let start_angle = params[self.arc.start + 3];
        let end_angle = params[self.arc.start + 4];

        let angle = if self.at_start { start_angle } else { end_angle };

        let arc_x = cx + r * angle.cos();
        let arc_y = cy + r * angle.sin();

        vec![px - arc_x, py - arc_y]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let _cx = params[self.arc.start];
        let _cy = params[self.arc.start + 1];
        let r = params[self.arc.start + 2];
        let start_angle = params[self.arc.start + 3];
        let end_angle = params[self.arc.start + 4];

        let angle = if self.at_start { start_angle } else { end_angle };
        let angle_idx = if self.at_start {
            self.arc.start + 3
        } else {
            self.arc.start + 4
        };

        let c = angle.cos();
        let s = angle.sin();

        let mut entries = Vec::with_capacity(10);

        // Equation 0: px - (cx + r*cos(θ)) = 0
        // d/dpx = 1
        entries.push((0, self.point.start, 1.0));
        // d/dcx = -1
        entries.push((0, self.arc.start, -1.0));
        // d/dr = -cos(θ)
        entries.push((0, self.arc.start + 2, -c));
        // d/dθ = -r*(-sin(θ)) = r*sin(θ)
        entries.push((0, angle_idx, r * s));

        // Equation 1: py - (cy + r*sin(θ)) = 0
        // d/dpy = 1
        entries.push((1, self.point.start + 1, 1.0));
        // d/dcy = -1
        entries.push((1, self.arc.start + 1, -1.0));
        // d/dr = -sin(θ)
        entries.push((1, self.arc.start + 2, -s));
        // d/dθ = -r*cos(θ)
        entries.push((1, angle_idx, -r * c));

        entries
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

/// Arc sweep constraint: end_angle - start_angle = target_sweep.
///
/// This is a simple linear constraint to fix the angular extent of an arc.
///
/// # Entity Layout
/// - `arc`: Arc2D [cx, cy, r, start_angle, end_angle] — 5 params
///
/// # Equation
///
/// `arc.end_angle - arc.start_angle - target_sweep = 0`
///
/// # Jacobian
///
/// - `d/d(end_angle) = 1`
/// - `d/d(start_angle) = -1`
#[derive(Clone, Debug)]
pub struct ArcSweepConstraint {
    id: ConstraintId,
    /// Arc parameter range.
    arc: ParamRange,
    /// Target sweep angle (in radians).
    target_sweep: f64,
    /// Cached dependency list.
    dependencies: Vec<usize>,
}

impl ArcSweepConstraint {
    /// Create a new arc sweep constraint.
    ///
    /// # Arguments
    /// * `id` - Unique constraint identifier
    /// * `arc` - Parameter range for the arc (5 params)
    /// * `target_sweep` - Target sweep angle in radians
    ///
    /// # Panics
    /// Panics if arc.count != 5.
    pub fn new(id: ConstraintId, arc: ParamRange, target_sweep: f64) -> Self {
        assert_eq!(arc.count, 5, "Arc must have 5 params, got {}", arc.count);

        let dependencies = vec![arc.start + 3, arc.start + 4];

        Self {
            id,
            arc,
            target_sweep,
            dependencies,
        }
    }
}

impl Constraint for ArcSweepConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "ArcSweep"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let start_angle = params[self.arc.start + 3];
        let end_angle = params[self.arc.start + 4];

        vec![end_angle - start_angle - self.target_sweep]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, self.arc.start + 4, 1.0),  // d/d(end_angle)
            (0, self.arc.start + 3, -1.0), // d/d(start_angle)
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

/// Point on arc constraint.
///
/// Enforces that a point lies on the arc, meaning:
/// 1. The point is on the circle (at distance r from center)
/// 2. The point's angle is within the arc's angular range
///
/// For now, we implement only equation 1 (point on circle).
/// Equation 2 (angle within range) would be an inequality constraint,
/// which requires special handling. We leave it as a TODO.
///
/// # Entity Layout
/// - `point`: Point2D [x, y] — 2 params
/// - `arc`: Arc2D [cx, cy, r, start_angle, end_angle] — 5 params
///
/// # Equation
///
/// `(px - cx)^2 + (py - cy)^2 - r^2 = 0`
///
/// TODO: Add angle range constraint as a second equation or separate soft constraint.
///
/// # Jacobian
///
/// Same as PointOnCircleConstraint.
#[derive(Clone, Debug)]
pub struct PointOnArcConstraint {
    id: ConstraintId,
    /// Point parameter range [x, y].
    point: ParamRange,
    /// Arc parameter range [cx, cy, r, start_angle, end_angle].
    arc: ParamRange,
    /// Cached dependency list.
    dependencies: Vec<usize>,
}

impl PointOnArcConstraint {
    /// Create a new point-on-arc constraint.
    ///
    /// # Arguments
    /// * `id` - Unique constraint identifier
    /// * `point` - Parameter range for the point (2 params)
    /// * `arc` - Parameter range for the arc (5 params)
    ///
    /// # Panics
    /// Panics if point.count != 2 or arc.count != 5.
    pub fn new(id: ConstraintId, point: ParamRange, arc: ParamRange) -> Self {
        assert_eq!(
            point.count, 2,
            "Point must be 2D (2 params), got {}",
            point.count
        );
        assert_eq!(arc.count, 5, "Arc must have 5 params, got {}", arc.count);

        // Only depend on point coords, center coords, and radius (not angles)
        let mut dependencies = Vec::with_capacity(5);
        dependencies.extend(point.iter());
        dependencies.push(arc.start); // cx
        dependencies.push(arc.start + 1); // cy
        dependencies.push(arc.start + 2); // r

        Self {
            id,
            point,
            arc,
            dependencies,
        }
    }
}

impl Constraint for PointOnArcConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "PointOnArc"
    }

    fn equation_count(&self) -> usize {
        1 // Only the circle equation for now
    }

    fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let px = params[self.point.start];
        let py = params[self.point.start + 1];

        let cx = params[self.arc.start];
        let cy = params[self.arc.start + 1];
        let r = params[self.arc.start + 2];

        let dx = px - cx;
        let dy = py - cy;
        let dist_sq = dx * dx + dy * dy;

        // TODO: Add angle range constraint
        vec![dist_sq - r * r]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let px = params[self.point.start];
        let py = params[self.point.start + 1];

        let cx = params[self.arc.start];
        let cy = params[self.arc.start + 1];
        let r = params[self.arc.start + 2];

        let dx = px - cx;
        let dy = py - cy;

        let mut entries = Vec::with_capacity(5);

        // d(dist_sq - r^2)/dpx = 2*dx
        entries.push((0, self.point.start, 2.0 * dx));

        // d(dist_sq - r^2)/dpy = 2*dy
        entries.push((0, self.point.start + 1, 2.0 * dy));

        // d(dist_sq - r^2)/dcx = -2*dx
        entries.push((0, self.arc.start, -2.0 * dx));

        // d(dist_sq - r^2)/dcy = -2*dy
        entries.push((0, self.arc.start + 1, -2.0 * dy));

        // d(dist_sq - r^2)/dr = -2*r
        entries.push((0, self.arc.start + 2, -2.0 * r));

        entries
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ===== ArcEndpointConstraint Tests =====

    #[test]
    fn test_arc_endpoint_start_satisfied() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = ArcEndpointConstraint::new(id, point, arc, true);

        // Arc: center (0, 0), radius 5, start_angle = 0, end_angle = π/2
        // Point at (5, 0) = start of arc
        let params = vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, PI / 2.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10, "x residual = {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "y residual = {}", residuals[1]);
    }

    #[test]
    fn test_arc_endpoint_end_satisfied() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = ArcEndpointConstraint::new(id, point, arc, false);

        // Arc: center (0, 0), radius 5, start_angle = 0, end_angle = π/2
        // Point at (0, 5) = end of arc
        let params = vec![0.0, 5.0, 0.0, 0.0, 5.0, 0.0, PI / 2.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10, "x residual = {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "y residual = {}", residuals[1]);
    }

    #[test]
    fn test_arc_endpoint_not_satisfied() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = ArcEndpointConstraint::new(id, point, arc, true);

        // Point at (10, 10), not at arc start
        let params = vec![10.0, 10.0, 0.0, 0.0, 5.0, 0.0, PI / 2.0];

        let residuals = constraint.residuals(&params);
        // Should be non-zero
        assert!(residuals[0].abs() > 1e-5);
        assert!(residuals[1].abs() > 1e-5);
    }

    #[test]
    fn test_arc_endpoint_jacobian() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = ArcEndpointConstraint::new(id, point, arc, true);

        let params = vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, PI / 2.0];
        let jac = constraint.jacobian(&params);

        // Should have entries for px, py, cx, cy, r, and start_angle
        assert!(!jac.is_empty());

        for (row, _col, val) in &jac {
            assert!(*row < 2);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_arc_endpoint_name() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };

        let start = ArcEndpointConstraint::new(id, point, arc, true);
        assert_eq!(start.name(), "ArcStartpoint");

        let end = ArcEndpointConstraint::new(id, point, arc, false);
        assert_eq!(end.name(), "ArcEndpoint");
    }

    // ===== ArcSweepConstraint Tests =====

    #[test]
    fn test_arc_sweep_satisfied() {
        let id = ConstraintId(0);
        let arc = ParamRange { start: 0, count: 5 };
        let constraint = ArcSweepConstraint::new(id, arc, PI / 2.0);

        // Arc with sweep = π/2
        let params = vec![0.0, 0.0, 5.0, 0.0, PI / 2.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_arc_sweep_not_satisfied() {
        let id = ConstraintId(0);
        let arc = ParamRange { start: 0, count: 5 };
        let constraint = ArcSweepConstraint::new(id, arc, PI / 2.0);

        // Arc with sweep = π (not π/2)
        let params = vec![0.0, 0.0, 5.0, 0.0, PI];

        let residuals = constraint.residuals(&params);
        // residual = π - 0 - π/2 = π/2
        assert!((residuals[0] - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_arc_sweep_jacobian() {
        let id = ConstraintId(0);
        let arc = ParamRange { start: 0, count: 5 };
        let constraint = ArcSweepConstraint::new(id, arc, PI / 2.0);

        let params = vec![0.0, 0.0, 5.0, 0.0, PI / 2.0];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 2);

        let mut found = std::collections::HashMap::new();
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            found.insert(*col, *val);
        }

        // d/d(end_angle) = 1
        assert!((found[&4] - 1.0).abs() < 1e-10);
        // d/d(start_angle) = -1
        assert!((found[&3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_arc_sweep_dependencies() {
        let id = ConstraintId(0);
        let arc = ParamRange { start: 10, count: 5 };
        let constraint = ArcSweepConstraint::new(id, arc, PI);

        let deps = constraint.dependencies();
        assert_eq!(deps, &[13, 14]); // start_angle and end_angle indices
    }

    #[test]
    fn test_arc_sweep_name() {
        let id = ConstraintId(0);
        let arc = ParamRange { start: 0, count: 5 };
        let constraint = ArcSweepConstraint::new(id, arc, PI);

        assert_eq!(constraint.name(), "ArcSweep");
    }

    #[test]
    fn test_arc_sweep_nonlinearity() {
        let id = ConstraintId(0);
        let arc = ParamRange { start: 0, count: 5 };
        let constraint = ArcSweepConstraint::new(id, arc, PI);

        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Linear);
    }

    // ===== PointOnArcConstraint Tests =====

    #[test]
    fn test_point_on_arc_satisfied() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = PointOnArcConstraint::new(id, point, arc);

        // Point at (3, 4), arc center at (0, 0), radius 5
        let params = vec![3.0, 4.0, 0.0, 0.0, 5.0, 0.0, PI / 2.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        // dist_sq = 25, r^2 = 25, residual = 0
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_point_on_arc_not_satisfied() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = PointOnArcConstraint::new(id, point, arc);

        // Point at (10, 0), not on circle
        let params = vec![10.0, 0.0, 0.0, 0.0, 5.0, 0.0, PI / 2.0];

        let residuals = constraint.residuals(&params);
        // dist_sq = 100, r^2 = 25, residual = 75
        assert!((residuals[0] - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_on_arc_jacobian() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = PointOnArcConstraint::new(id, point, arc);

        let params = vec![3.0, 4.0, 0.0, 0.0, 5.0, 0.0, PI / 2.0];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 5); // px, py, cx, cy, r

        let mut found = std::collections::HashMap::new();
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            found.insert(*col, *val);
        }

        // diff = [3, 4]
        // d/dpx = 2*3 = 6
        assert!((found[&0] - 6.0).abs() < 1e-10);
        // d/dpy = 2*4 = 8
        assert!((found[&1] - 8.0).abs() < 1e-10);
        // d/dcx = -6
        assert!((found[&2] - (-6.0)).abs() < 1e-10);
        // d/dcy = -8
        assert!((found[&3] - (-8.0)).abs() < 1e-10);
        // d/dr = -2*5 = -10
        assert!((found[&4] - (-10.0)).abs() < 1e-10);
    }

    #[test]
    fn test_point_on_arc_dependencies() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 10, count: 2 };
        let arc = ParamRange { start: 20, count: 5 };
        let constraint = PointOnArcConstraint::new(id, point, arc);

        let deps = constraint.dependencies();
        // Point coords + arc center + radius (not angles)
        assert_eq!(deps, &[10, 11, 20, 21, 22]);
    }

    #[test]
    fn test_point_on_arc_name() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = PointOnArcConstraint::new(id, point, arc);

        assert_eq!(constraint.name(), "PointOnArc");
    }

    #[test]
    fn test_point_on_arc_nonlinearity() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let arc = ParamRange { start: 2, count: 5 };
        let constraint = PointOnArcConstraint::new(id, point, arc);

        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Moderate);
    }

    #[test]
    #[should_panic(expected = "Point must be 2D")]
    fn test_arc_endpoint_invalid_point() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 3 };
        let arc = ParamRange { start: 3, count: 5 };
        ArcEndpointConstraint::new(id, point, arc, true);
    }

    #[test]
    #[should_panic(expected = "Arc must have 5 params")]
    fn test_arc_sweep_invalid_arc() {
        let id = ConstraintId(0);
        let arc = ParamRange { start: 0, count: 3 };
        ArcSweepConstraint::new(id, arc, PI);
    }
}
