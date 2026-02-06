//! Tangent constraints: line-circle and circle-circle tangency.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Line tangent to circle constraint.
///
/// Enforces that a line is tangent to a circle, i.e., the perpendicular distance
/// from the circle center to the line equals the circle radius.
///
/// # Entity Layout
/// - `line`: Line2D [x1, y1, x2, y2] — 4 params
/// - `circle`: Circle2D [cx, cy, r] — 3 params
///
/// # Equation (2D only)
///
/// We use a squared formulation to avoid sqrt and absolute value:
/// `cross^2 - r^2 * len_sq = 0`
///
/// where:
/// - `dx = x2 - x1`, `dy = y2 - y1`
/// - `cross = dx * (cy - y1) - dy * (cx - x1)` (signed perpendicular distance * line length)
/// - `len_sq = dx^2 + dy^2`
/// - `r` = circle radius
///
/// The perpendicular distance is `|cross| / sqrt(len_sq)`, so the tangency condition
/// `|cross| / sqrt(len_sq) = r` becomes `cross^2 = r^2 * len_sq`.
///
/// # Jacobian
///
/// Let `f = cross^2 - r^2 * len_sq`. We compute `df/dvar` for each variable.
/// - `df/dx1 = 2*cross*dcross_dx1 - r^2*dlen_sq_dx1`
/// - Similar for y1, x2, y2, cx, cy, r
#[derive(Clone, Debug)]
pub struct LineTangentCircleConstraint {
    id: ConstraintId,
    /// Line parameter range [x1, y1, x2, y2].
    line: ParamRange,
    /// Circle parameter range [cx, cy, r].
    circle: ParamRange,
    /// Cached dependency list.
    dependencies: Vec<usize>,
}

impl LineTangentCircleConstraint {
    /// Create a new line-tangent-circle constraint.
    ///
    /// # Arguments
    /// * `id` - Unique constraint identifier
    /// * `line` - Parameter range for the line (4 params: [x1, y1, x2, y2])
    /// * `circle` - Parameter range for the circle (3 params: [cx, cy, r])
    ///
    /// # Panics
    /// Panics if line.count != 4 or circle.count != 3.
    pub fn new(id: ConstraintId, line: ParamRange, circle: ParamRange) -> Self {
        assert_eq!(line.count, 4, "Line must have 4 params, got {}", line.count);
        assert_eq!(
            circle.count, 3,
            "Circle must have 3 params, got {}",
            circle.count
        );

        let mut dependencies = Vec::with_capacity(7);
        dependencies.extend(line.iter());
        dependencies.extend(circle.iter());

        Self {
            id,
            line,
            circle,
            dependencies,
        }
    }
}

impl Constraint for LineTangentCircleConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "LineTangentCircle"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let x1 = params[self.line.start];
        let y1 = params[self.line.start + 1];
        let x2 = params[self.line.start + 2];
        let y2 = params[self.line.start + 3];

        let cx = params[self.circle.start];
        let cy = params[self.circle.start + 1];
        let r = params[self.circle.start + 2];

        let dx = x2 - x1;
        let dy = y2 - y1;

        // cross = dx * (cy - y1) - dy * (cx - x1)
        let cross = dx * (cy - y1) - dy * (cx - x1);

        // len_sq = dx^2 + dy^2
        let len_sq = dx * dx + dy * dy;

        // Residual: cross^2 - r^2 * len_sq
        vec![cross * cross - r * r * len_sq]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let x1 = params[self.line.start];
        let y1 = params[self.line.start + 1];
        let x2 = params[self.line.start + 2];
        let y2 = params[self.line.start + 3];

        let cx = params[self.circle.start];
        let cy = params[self.circle.start + 1];
        let r = params[self.circle.start + 2];

        let dx = x2 - x1;
        let dy = y2 - y1;

        let cross = dx * (cy - y1) - dy * (cx - x1);
        let len_sq = dx * dx + dy * dy;

        // f = cross^2 - r^2 * len_sq
        // df/dvar = 2*cross*dcross_dvar - r^2*dlen_sq_dvar (for line points)
        // df/dr = -2*r*len_sq (for radius)

        let mut entries = Vec::with_capacity(7);

        // Derivatives of cross:
        // cross = dx*(cy - y1) - dy*(cx - x1)
        // cross = (x2-x1)*(cy - y1) - (y2-y1)*(cx - x1)
        //
        // dcross/dx1 = -(cy - y1) + (cx - x1)*0 + (y2-y1) = y2 - cy
        // dcross/dy1 = (x2-x1)*(-1) + (cx - x1) = cx - x2
        // dcross/dx2 = (cy - y1) - (cx - x1)*0 = cy - y1
        // dcross/dy2 = (x2-x1)*0 - (cx - x1) = -(cx - x1) = x1 - cx
        // dcross/dcx = -(y2 - y1) = y1 - y2
        // dcross/dcy = x2 - x1 = dx

        let dcross_dx1 = y2 - cy;
        let dcross_dy1 = cx - x2;
        let dcross_dx2 = cy - y1;
        let dcross_dy2 = x1 - cx;
        let dcross_dcx = y1 - y2;
        let dcross_dcy = dx;

        // Derivatives of len_sq:
        // len_sq = dx^2 + dy^2 = (x2-x1)^2 + (y2-y1)^2
        // dlen_sq/dx1 = -2*dx
        // dlen_sq/dy1 = -2*dy
        // dlen_sq/dx2 = 2*dx
        // dlen_sq/dy2 = 2*dy
        // dlen_sq/dcx = 0, dlen_sq/dcy = 0

        let dlen_sq_dx1 = -2.0 * dx;
        let dlen_sq_dy1 = -2.0 * dy;
        let dlen_sq_dx2 = 2.0 * dx;
        let dlen_sq_dy2 = 2.0 * dy;

        // df/dx1 = 2*cross*dcross_dx1 - r^2*dlen_sq_dx1
        entries.push((
            0,
            self.line.start,
            2.0 * cross * dcross_dx1 - r * r * dlen_sq_dx1,
        ));

        // df/dy1
        entries.push((
            0,
            self.line.start + 1,
            2.0 * cross * dcross_dy1 - r * r * dlen_sq_dy1,
        ));

        // df/dx2
        entries.push((
            0,
            self.line.start + 2,
            2.0 * cross * dcross_dx2 - r * r * dlen_sq_dx2,
        ));

        // df/dy2
        entries.push((
            0,
            self.line.start + 3,
            2.0 * cross * dcross_dy2 - r * r * dlen_sq_dy2,
        ));

        // df/dcx = 2*cross*dcross_dcx
        entries.push((0, self.circle.start, 2.0 * cross * dcross_dcx));

        // df/dcy = 2*cross*dcross_dcy
        entries.push((0, self.circle.start + 1, 2.0 * cross * dcross_dcy));

        // df/dr = d(cross^2 - r^2*len_sq)/dr = -2*r*len_sq
        entries.push((0, self.circle.start + 2, -2.0 * r * len_sq));

        entries
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::High
    }
}

/// Circle-circle tangent constraint.
///
/// Enforces that two circles are tangent to each other.
/// - External tangent: `||c1 - c2|| = r1 + r2`
/// - Internal tangent: `||c1 - c2|| = |r1 - r2|`
///
/// # Entity Layout
/// - `circle1`: Circle2D [cx1, cy1, r1] — 3 params
/// - `circle2`: Circle2D [cx2, cy2, r2] — 3 params
///
/// # Equation
///
/// We use squared distance to avoid sqrt:
/// - External: `dist_sq - (r1 + r2)^2 = 0`
/// - Internal: `dist_sq - (r1 - r2)^2 = 0`
///
/// where `dist_sq = (cx1 - cx2)^2 + (cy1 - cy2)^2` (2D) or similar for 3D.
///
/// # Jacobian
///
/// - `d(dist_sq)/dc1[i] = 2*(c1[i] - c2[i])`
/// - `d(dist_sq)/dc2[i] = -2*(c1[i] - c2[i])`
/// - For external: `d((r1+r2)^2)/dr1 = 2*(r1+r2)`, `d/dr2 = 2*(r1+r2)`
/// - For internal: `d((r1-r2)^2)/dr1 = 2*(r1-r2)`, `d/dr2 = -2*(r1-r2)`
#[derive(Clone, Debug)]
pub struct CircleTangentConstraint {
    id: ConstraintId,
    /// First circle parameter range.
    circle1: ParamRange,
    /// Second circle parameter range.
    circle2: ParamRange,
    /// External (true) or internal (false) tangency.
    external: bool,
    /// Cached dependency list.
    dependencies: Vec<usize>,
}

impl CircleTangentConstraint {
    /// Create a new circle-tangent-circle constraint.
    ///
    /// # Arguments
    /// * `id` - Unique constraint identifier
    /// * `circle1` - Parameter range for the first circle
    /// * `circle2` - Parameter range for the second circle
    /// * `external` - True for external tangent, false for internal
    ///
    /// # Panics
    /// Panics if circles have different dimensions or invalid param counts.
    pub fn new(
        id: ConstraintId,
        circle1: ParamRange,
        circle2: ParamRange,
        external: bool,
    ) -> Self {
        assert_eq!(
            circle1.count, circle2.count,
            "Circles must have the same dimension, got {} and {}",
            circle1.count, circle2.count
        );

        let dim = circle1.count - 1; // Dimension (2 or 3)
        assert!(
            dim == 2 || dim == 3,
            "Circles must be 2D or 3D, got {} params",
            circle1.count
        );

        let mut dependencies = Vec::with_capacity(circle1.count + circle2.count);
        dependencies.extend(circle1.iter());
        dependencies.extend(circle2.iter());

        Self {
            id,
            circle1,
            circle2,
            external,
            dependencies,
        }
    }
}

impl Constraint for CircleTangentConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        if self.external {
            "CircleTangentExternal"
        } else {
            "CircleTangentInternal"
        }
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.circle1.count - 1;

        // Extract radii
        let r1 = params[self.circle1.start + dim];
        let r2 = params[self.circle2.start + dim];

        // Compute dist_sq = sum((c1[i] - c2[i])^2)
        let mut dist_sq = 0.0;
        for i in 0..dim {
            let c1_i = params[self.circle1.start + i];
            let c2_i = params[self.circle2.start + i];
            let diff = c1_i - c2_i;
            dist_sq += diff * diff;
        }

        // Compute target^2
        let target_sq = if self.external {
            let sum = r1 + r2;
            sum * sum
        } else {
            let diff = r1 - r2;
            diff * diff
        };

        vec![dist_sq - target_sq]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.circle1.count - 1;

        let r1 = params[self.circle1.start + dim];
        let r2 = params[self.circle2.start + dim];

        let mut entries = Vec::with_capacity(dim * 2 + 2);

        // d(dist_sq)/dc1[i] and d(dist_sq)/dc2[i]
        for i in 0..dim {
            let c1_i = params[self.circle1.start + i];
            let c2_i = params[self.circle2.start + i];
            let diff = c1_i - c2_i;

            // d(dist_sq)/dc1[i] = 2*diff
            entries.push((0, self.circle1.start + i, 2.0 * diff));

            // d(dist_sq)/dc2[i] = -2*diff
            entries.push((0, self.circle2.start + i, -2.0 * diff));
        }

        // Derivatives w.r.t. radii
        if self.external {
            // target^2 = (r1 + r2)^2
            // d(target^2)/dr1 = 2*(r1 + r2)
            // d(target^2)/dr2 = 2*(r1 + r2)
            let sum = r1 + r2;
            entries.push((0, self.circle1.start + dim, -2.0 * sum));
            entries.push((0, self.circle2.start + dim, -2.0 * sum));
        } else {
            // target^2 = (r1 - r2)^2
            // d(target^2)/dr1 = 2*(r1 - r2)
            // d(target^2)/dr2 = -2*(r1 - r2)
            let diff = r1 - r2;
            entries.push((0, self.circle1.start + dim, -2.0 * diff));
            entries.push((0, self.circle2.start + dim, 2.0 * diff));
        }

        entries
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::High
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== LineTangentCircleConstraint Tests =====

    #[test]
    fn test_line_tangent_circle_satisfied() {
        let id = ConstraintId(0);
        let line = ParamRange { start: 0, count: 4 };
        let circle = ParamRange { start: 4, count: 3 };
        let constraint = LineTangentCircleConstraint::new(id, line, circle);

        // Horizontal line from (0, 5) to (10, 5)
        // Circle at (5, 0) with radius 5
        // Distance from center to line = 5, so tangent
        let params = vec![0.0, 5.0, 10.0, 5.0, 5.0, 0.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        // dx=10, dy=0, cross = 10*(-5) - 0*(-5) = -50, len_sq=100
        // cross^2 = 2500, r^2*len_sq = 25*100 = 2500, residual = 0
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_line_tangent_circle_not_satisfied() {
        let id = ConstraintId(0);
        let line = ParamRange { start: 0, count: 4 };
        let circle = ParamRange { start: 4, count: 3 };
        let constraint = LineTangentCircleConstraint::new(id, line, circle);

        // Horizontal line from (0, 10) to (10, 10)
        // Circle at (5, 0) with radius 5
        // Distance from center to line = 10, not tangent
        let params = vec![0.0, 10.0, 10.0, 10.0, 5.0, 0.0, 5.0];

        let residuals = constraint.residuals(&params);
        // dx=10, dy=0, cross = 10*(-10) = -100, len_sq=100
        // cross^2 = 10000, r^2*len_sq = 25*100 = 2500, residual = 7500
        assert!((residuals[0] - 7500.0).abs() < 1e-10);
    }

    #[test]
    fn test_line_tangent_circle_jacobian() {
        let id = ConstraintId(0);
        let line = ParamRange { start: 0, count: 4 };
        let circle = ParamRange { start: 4, count: 3 };
        let constraint = LineTangentCircleConstraint::new(id, line, circle);

        // Simple case: horizontal line, circle below
        let params = vec![0.0, 5.0, 10.0, 5.0, 5.0, 0.0, 5.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 7);

        // Just verify all are finite (complex derivatives)
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(
                val.is_finite(),
                "Non-finite Jacobian entry at col {}: {}",
                col,
                val
            );
        }
    }

    #[test]
    fn test_line_tangent_dependencies() {
        let id = ConstraintId(0);
        let line = ParamRange { start: 10, count: 4 };
        let circle = ParamRange { start: 20, count: 3 };
        let constraint = LineTangentCircleConstraint::new(id, line, circle);

        let deps = constraint.dependencies();
        assert_eq!(deps, &[10, 11, 12, 13, 20, 21, 22]);
    }

    #[test]
    fn test_line_tangent_name() {
        let id = ConstraintId(0);
        let line = ParamRange { start: 0, count: 4 };
        let circle = ParamRange { start: 4, count: 3 };
        let constraint = LineTangentCircleConstraint::new(id, line, circle);

        assert_eq!(constraint.name(), "LineTangentCircle");
    }

    #[test]
    fn test_line_tangent_nonlinearity() {
        let id = ConstraintId(0);
        let line = ParamRange { start: 0, count: 4 };
        let circle = ParamRange { start: 4, count: 3 };
        let constraint = LineTangentCircleConstraint::new(id, line, circle);

        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::High);
    }

    // ===== CircleTangentConstraint Tests =====

    #[test]
    fn test_circle_tangent_external_satisfied_2d() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 0, count: 3 };
        let circle2 = ParamRange { start: 3, count: 3 };
        let constraint = CircleTangentConstraint::new(id, circle1, circle2, true);

        // Circle1 at (0, 0, r=3), Circle2 at (8, 0, r=5)
        // Distance = 8, r1+r2 = 8, tangent
        let params = vec![0.0, 0.0, 3.0, 8.0, 0.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        // dist_sq = 64, (r1+r2)^2 = 64, residual = 0
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_circle_tangent_internal_satisfied_2d() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 0, count: 3 };
        let circle2 = ParamRange { start: 3, count: 3 };
        let constraint = CircleTangentConstraint::new(id, circle1, circle2, false);

        // Circle1 at (0, 0, r=10), Circle2 at (3, 0, r=7)
        // Distance = 3, |r1-r2| = 3, internal tangent
        let params = vec![0.0, 0.0, 10.0, 3.0, 0.0, 7.0];

        let residuals = constraint.residuals(&params);
        // dist_sq = 9, (r1-r2)^2 = 9, residual = 0
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_circle_tangent_external_3d() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 0, count: 4 };
        let circle2 = ParamRange { start: 4, count: 4 };
        let constraint = CircleTangentConstraint::new(id, circle1, circle2, true);

        // Sphere1 at (0, 0, 0, r=3), Sphere2 at (5, 0, 0, r=2)
        // Distance = 5, r1+r2 = 5, tangent
        let params = vec![0.0, 0.0, 0.0, 3.0, 5.0, 0.0, 0.0, 2.0];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_circle_tangent_jacobian_external() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 0, count: 3 };
        let circle2 = ParamRange { start: 3, count: 3 };
        let constraint = CircleTangentConstraint::new(id, circle1, circle2, true);

        let params = vec![0.0, 0.0, 3.0, 8.0, 0.0, 5.0];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 6); // 2 centers * 2 coords + 2 radii

        // Verify structure
        let mut found = std::collections::HashMap::new();
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(val.is_finite());
            found.insert(*col, *val);
        }

        // Check center derivatives: diff = [0-8, 0-0] = [-8, 0]
        // d/dc1x = 2*(-8) = -16
        assert!((found[&0] - (-16.0)).abs() < 1e-10, "d/dc1x");
        // d/dc1y = 0
        assert!((found[&1] - 0.0).abs() < 1e-10, "d/dc1y");
        // d/dc2x = -(-16) = 16
        assert!((found[&3] - 16.0).abs() < 1e-10, "d/dc2x");
        // d/dc2y = 0
        assert!((found[&4] - 0.0).abs() < 1e-10, "d/dc2y");

        // Check radius derivatives: sum = 3+5 = 8
        // d/dr1 = -2*8 = -16
        assert!((found[&2] - (-16.0)).abs() < 1e-10, "d/dr1");
        // d/dr2 = -2*8 = -16
        assert!((found[&5] - (-16.0)).abs() < 1e-10, "d/dr2");
    }

    #[test]
    fn test_circle_tangent_jacobian_internal() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 0, count: 3 };
        let circle2 = ParamRange { start: 3, count: 3 };
        let constraint = CircleTangentConstraint::new(id, circle1, circle2, false);

        let params = vec![0.0, 0.0, 10.0, 3.0, 0.0, 7.0];
        let jac = constraint.jacobian(&params);

        let mut found = std::collections::HashMap::new();
        for (_row, col, val) in &jac {
            found.insert(*col, *val);
        }

        // diff = r1 - r2 = 3
        // d/dr1 = -2*3 = -6
        assert!((found[&2] - (-6.0)).abs() < 1e-10, "d/dr1");
        // d/dr2 = +2*3 = 6
        assert!((found[&5] - 6.0).abs() < 1e-10, "d/dr2");
    }

    #[test]
    fn test_circle_tangent_dependencies() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 10, count: 3 };
        let circle2 = ParamRange { start: 20, count: 3 };
        let constraint = CircleTangentConstraint::new(id, circle1, circle2, true);

        let deps = constraint.dependencies();
        assert_eq!(deps, &[10, 11, 12, 20, 21, 22]);
    }

    #[test]
    fn test_circle_tangent_name() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 0, count: 3 };
        let circle2 = ParamRange { start: 3, count: 3 };

        let external = CircleTangentConstraint::new(id, circle1, circle2, true);
        assert_eq!(external.name(), "CircleTangentExternal");

        let internal = CircleTangentConstraint::new(id, circle1, circle2, false);
        assert_eq!(internal.name(), "CircleTangentInternal");
    }

    #[test]
    fn test_circle_tangent_nonlinearity() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 0, count: 3 };
        let circle2 = ParamRange { start: 3, count: 3 };
        let constraint = CircleTangentConstraint::new(id, circle1, circle2, true);

        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::High);
    }

    #[test]
    #[should_panic(expected = "Circles must have the same dimension")]
    fn test_circle_tangent_dimension_mismatch() {
        let id = ConstraintId(0);
        let circle1 = ParamRange { start: 0, count: 3 };
        let circle2 = ParamRange { start: 3, count: 4 };
        CircleTangentConstraint::new(id, circle1, circle2, true);
    }
}
