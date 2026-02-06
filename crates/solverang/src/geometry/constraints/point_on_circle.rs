//! Point on circle constraint: a point must lie on a circle (with variable radius).

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Point on circle constraint: point P lies on circle with center C and radius r.
///
/// Unlike the old implementation, the radius is now a VARIABLE parameter, not a constant.
/// This allows the solver to optimize circle radii.
///
/// # Entity Layout
/// - `point`: Point2D [x, y] or Point3D [x, y, z] — D params
/// - `circle`: Circle2D [cx, cy, r] (3 params) — center=[0..2], radius=[2]
///
/// # Equation
///
/// We use squared distance to avoid sqrt in the residual:
/// `sum((p[i] - c[i])^2) - r^2 = 0`
///
/// This is smoother for the Jacobian than `sqrt(sum((p[i] - c[i])^2)) - r = 0`.
///
/// # Jacobian
///
/// Let `diff[i] = p[i] - c[i]`, then:
/// - `d/dp[i] = 2 * diff[i]`
/// - `d/dc[i] = -2 * diff[i]`
/// - `d/dr = -2 * r`
#[derive(Clone, Debug)]
pub struct PointOnCircleConstraint {
    id: ConstraintId,
    /// The point that must lie on the circle (2 or 3 params).
    point: ParamRange,
    /// The circle (3 params for 2D: [cx, cy, r]).
    circle: ParamRange,
    /// Cached dependency list for graph operations.
    dependencies: Vec<usize>,
}

impl PointOnCircleConstraint {
    /// Create a new point-on-circle constraint.
    ///
    /// # Arguments
    /// * `id` - Unique constraint identifier
    /// * `point` - Parameter range for the point (2 or 3 params)
    /// * `circle` - Parameter range for the circle (3 params: [cx, cy, r])
    ///
    /// # Panics
    /// Panics if circle.count != point.count + 1 (invalid entity layout).
    pub fn new(id: ConstraintId, point: ParamRange, circle: ParamRange) -> Self {
        let dim = point.count;
        assert!(
            dim == 2 || dim == 3,
            "Point must be 2D or 3D, got {} params",
            dim
        );
        assert_eq!(
            circle.count,
            dim + 1,
            "Circle must have {} params ([center coords] + radius), got {}",
            dim + 1,
            circle.count
        );

        // Build dependency list: all point coords + all circle coords + radius
        let mut dependencies = Vec::with_capacity(point.count + circle.count);
        dependencies.extend(point.iter());
        dependencies.extend(circle.iter());

        Self {
            id,
            point,
            circle,
            dependencies,
        }
    }
}

impl Constraint for PointOnCircleConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "PointOnCircle"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.point.count;

        // Extract circle center and radius
        // Circle layout: [cx, cy, r] (2D) or [cx, cy, cz, r] (3D)
        let r = params[self.circle.start + dim];

        // Compute sum((p[i] - c[i])^2)
        let mut dist_sq = 0.0;
        for i in 0..dim {
            let p_i = params[self.point.start + i];
            let c_i = params[self.circle.start + i];
            let diff = p_i - c_i;
            dist_sq += diff * diff;
        }

        // Residual: dist_sq - r^2
        vec![dist_sq - r * r]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.point.count;
        let r = params[self.circle.start + dim];

        let mut entries = Vec::with_capacity(dim * 2 + 1);

        for i in 0..dim {
            let p_i = params[self.point.start + i];
            let c_i = params[self.circle.start + i];
            let diff = p_i - c_i;

            // d(dist_sq - r^2)/dp[i] = 2 * diff
            entries.push((0, self.point.start + i, 2.0 * diff));

            // d(dist_sq - r^2)/dc[i] = -2 * diff
            entries.push((0, self.circle.start + i, -2.0 * diff));
        }

        // d(dist_sq - r^2)/dr = -2 * r
        let r_idx = self.circle.start + dim;
        entries.push((0, r_idx, -2.0 * r));

        entries
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_on_circle_2d_satisfied() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let circle = ParamRange { start: 2, count: 3 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        // Point at (3, 4), circle center at (0, 0), radius 5
        // Distance = sqrt(9 + 16) = 5, so it's on the circle
        let params = vec![3.0, 4.0, 0.0, 0.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        // dist_sq = 25, r^2 = 25, residual = 0
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_point_on_circle_2d_not_satisfied() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let circle = ParamRange { start: 2, count: 3 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        // Point at (10, 0), circle center at (0, 0), radius 5
        let params = vec![10.0, 0.0, 0.0, 0.0, 5.0];

        let residuals = constraint.residuals(&params);
        // dist_sq = 100, r^2 = 25, residual = 75
        assert!((residuals[0] - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_on_circle_3d_satisfied() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 3 };
        let circle = ParamRange { start: 3, count: 4 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        // Point at (3, 4, 0), sphere center at (0, 0, 0), radius 5
        let params = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_2d() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let circle = ParamRange { start: 2, count: 3 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        // Point at (3, 4), circle at (0, 0, 5)
        let params = vec![3.0, 4.0, 0.0, 0.0, 5.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 5); // 2 point coords + 2 center coords + 1 radius

        // Expected derivatives:
        // diff_x = 3 - 0 = 3, diff_y = 4 - 0 = 4
        // d/dpx = 2*3 = 6, d/dpy = 2*4 = 8
        // d/dcx = -6, d/dcy = -8
        // d/dr = -2*5 = -10

        let mut found = std::collections::HashMap::new();
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            found.insert(*col, *val);
        }

        assert_eq!(found.len(), 5);
        assert!((found[&0] - 6.0).abs() < 1e-10, "d/dpx");
        assert!((found[&1] - 8.0).abs() < 1e-10, "d/dpy");
        assert!((found[&2] - (-6.0)).abs() < 1e-10, "d/dcx");
        assert!((found[&3] - (-8.0)).abs() < 1e-10, "d/dcy");
        assert!((found[&4] - (-10.0)).abs() < 1e-10, "d/dr");
    }

    #[test]
    fn test_jacobian_3d() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 3 };
        let circle = ParamRange { start: 3, count: 4 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        // Point at (1, 0, 0), sphere at (0, 0, 0, 1)
        let params = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 7); // 3 point + 3 center + 1 radius

        let mut found = std::collections::HashMap::new();
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            found.insert(*col, *val);
        }

        // diff = [1, 0, 0], r = 1
        // d/dpx = 2, d/dpy = 0, d/dpz = 0
        // d/dcx = -2, d/dcy = 0, d/dcz = 0
        // d/dr = -2
        assert!((found[&0] - 2.0).abs() < 1e-10);
        assert!((found[&1] - 0.0).abs() < 1e-10);
        assert!((found[&2] - 0.0).abs() < 1e-10);
        assert!((found[&3] - (-2.0)).abs() < 1e-10);
        assert!((found[&4] - 0.0).abs() < 1e-10);
        assert!((found[&5] - 0.0).abs() < 1e-10);
        assert!((found[&6] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dependencies() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 10, count: 2 };
        let circle = ParamRange { start: 20, count: 3 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        let deps = constraint.dependencies();
        assert_eq!(deps, &[10, 11, 20, 21, 22]);
    }

    #[test]
    fn test_equation_count() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let circle = ParamRange { start: 2, count: 3 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        assert_eq!(constraint.equation_count(), 1);
    }

    #[test]
    fn test_name() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let circle = ParamRange { start: 2, count: 3 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        assert_eq!(constraint.name(), "PointOnCircle");
    }

    #[test]
    fn test_nonlinearity() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let circle = ParamRange { start: 2, count: 3 };
        let constraint = PointOnCircleConstraint::new(id, point, circle);

        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Moderate);
    }

    #[test]
    #[should_panic(expected = "Circle must have 3 params")]
    fn test_invalid_circle_param_count() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let circle = ParamRange { start: 2, count: 2 }; // Wrong count
        PointOnCircleConstraint::new(id, point, circle);
    }

    #[test]
    #[should_panic(expected = "Point must be 2D or 3D")]
    fn test_invalid_point_dimension() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 4 }; // Invalid
        let circle = ParamRange { start: 4, count: 5 };
        PointOnCircleConstraint::new(id, point, circle);
    }
}
