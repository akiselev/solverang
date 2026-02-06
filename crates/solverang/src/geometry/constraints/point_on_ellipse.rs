//! Point on ellipse constraint: a point must lie on an ellipse.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity, MIN_EPSILON};

/// Point lies on ellipse constraint.
///
/// Enforces that a 2D point lies on the perimeter of an ellipse.
///
/// # Entity Parameters
/// - point: Point2D [px, py] — 2 params
/// - ellipse: Ellipse2D [cx, cy, rx, ry, rotation] — 5 params
///
/// # Equation
/// Transform point to ellipse-local coordinates (axis-aligned frame):
/// ```text
/// dx = px - cx
/// dy = py - cy
/// local_x = dx*cos(rot) + dy*sin(rot)
/// local_y = -dx*sin(rot) + dy*cos(rot)
/// ```
/// Then enforce the ellipse equation:
/// ```text
/// (local_x/rx)² + (local_y/ry)² - 1 = 0
/// ```
///
/// # Jacobian
/// Chain rule through rotation transform. 7 partial derivatives:
/// px, py, cx, cy, rx, ry, rot
#[derive(Clone, Debug)]
pub struct PointOnEllipseConstraint {
    id: ConstraintId,
    point: ParamRange,
    ellipse: ParamRange,
    deps: Vec<usize>,
}

impl PointOnEllipseConstraint {
    /// Create a new point-on-ellipse constraint.
    ///
    /// # Arguments
    /// - `id`: Unique constraint identifier
    /// - `point`: ParamRange for Point2D [x, y]
    /// - `ellipse`: ParamRange for Ellipse2D [cx, cy, rx, ry, rotation]
    pub fn new(id: ConstraintId, point: ParamRange, ellipse: ParamRange) -> Self {
        assert_eq!(point.count, 2, "Point must have 2 parameters");
        assert_eq!(ellipse.count, 5, "Ellipse must have 5 parameters");

        let mut deps = Vec::with_capacity(7);
        deps.extend(point.iter());
        deps.extend(ellipse.iter());

        Self {
            id,
            point,
            ellipse,
            deps,
        }
    }
}

impl Constraint for PointOnEllipseConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "PointOnEllipse"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let px = params[self.point.start];
        let py = params[self.point.start + 1];

        let cx = params[self.ellipse.start];
        let cy = params[self.ellipse.start + 1];
        let rx = params[self.ellipse.start + 2];
        let ry = params[self.ellipse.start + 3];
        let rot = params[self.ellipse.start + 4];

        // Safe radii (prevent division by zero)
        let rx_safe = if rx.abs() < MIN_EPSILON { MIN_EPSILON } else { rx.abs() };
        let ry_safe = if ry.abs() < MIN_EPSILON { MIN_EPSILON } else { ry.abs() };

        // Transform to ellipse-local coordinates
        let dx = px - cx;
        let dy = py - cy;

        let cos_r = rot.cos();
        let sin_r = rot.sin();

        let local_x = dx * cos_r + dy * sin_r;
        let local_y = -dx * sin_r + dy * cos_r;

        // Ellipse equation: (local_x/rx)² + (local_y/ry)² - 1 = 0
        let term_x = local_x / rx_safe;
        let term_y = local_y / ry_safe;

        let residual = term_x * term_x + term_y * term_y - 1.0;

        vec![residual]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let px = params[self.point.start];
        let py = params[self.point.start + 1];

        let cx = params[self.ellipse.start];
        let cy = params[self.ellipse.start + 1];
        let rx = params[self.ellipse.start + 2];
        let ry = params[self.ellipse.start + 3];
        let rot = params[self.ellipse.start + 4];

        // Safe radii
        let rx_safe = if rx.abs() < MIN_EPSILON { MIN_EPSILON } else { rx.abs() };
        let ry_safe = if ry.abs() < MIN_EPSILON { MIN_EPSILON } else { ry.abs() };

        // Intermediate values
        let dx = px - cx;
        let dy = py - cy;

        let cos_r = rot.cos();
        let sin_r = rot.sin();

        let local_x = dx * cos_r + dy * sin_r;
        let local_y = -dx * sin_r + dy * cos_r;

        let term_x = local_x / rx_safe;
        let term_y = local_y / ry_safe;

        // f = (local_x/rx)² + (local_y/ry)² - 1
        // f = term_x² + term_y² - 1

        // Chain rule:
        // df/dpx = df/dlocal_x * dlocal_x/dpx + df/dlocal_y * dlocal_y/dpx
        //        = (2*term_x/rx) * cos_r + (2*term_y/ry) * (-sin_r)

        let df_dlocal_x = 2.0 * term_x / rx_safe;
        let df_dlocal_y = 2.0 * term_y / ry_safe;

        // Derivatives w.r.t. px, py
        let df_dpx = df_dlocal_x * cos_r + df_dlocal_y * (-sin_r);
        let df_dpy = df_dlocal_x * sin_r + df_dlocal_y * cos_r;

        // Derivatives w.r.t. cx, cy (opposite sign)
        let df_dcx = -df_dpx;
        let df_dcy = -df_dpy;

        // Derivatives w.r.t. rx, ry
        // f = (local_x)²/rx² + (local_y)²/ry² - 1
        // df/drx = -2*(local_x)²/rx³
        // df/dry = -2*(local_y)²/ry³
        let df_drx = -2.0 * local_x * local_x / (rx_safe * rx_safe * rx_safe);
        let df_dry = -2.0 * local_y * local_y / (ry_safe * ry_safe * ry_safe);

        // Derivative w.r.t. rotation
        // dlocal_x/drot = -dx*sin_r + dy*cos_r
        // dlocal_y/drot = -dx*cos_r - dy*sin_r
        let dlocal_x_drot = -dx * sin_r + dy * cos_r;
        let dlocal_y_drot = -dx * cos_r - dy * sin_r;

        let df_drot = df_dlocal_x * dlocal_x_drot + df_dlocal_y * dlocal_y_drot;

        vec![
            (0, self.point.start, df_dpx),
            (0, self.point.start + 1, df_dpy),
            (0, self.ellipse.start, df_dcx),
            (0, self.ellipse.start + 1, df_dcy),
            (0, self.ellipse.start + 2, df_drx),
            (0, self.ellipse.start + 3, df_dry),
            (0, self.ellipse.start + 4, df_drot),
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::High
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_on_circle_as_ellipse() {
        // Circle is an ellipse with rx = ry and rotation = 0
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let ellipse = ParamRange { start: 2, count: 5 };

        let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

        // Point at (5, 0), circle at origin with radius 5
        let params = vec![
            5.0, 0.0,        // point
            0.0, 0.0, 5.0, 5.0, 0.0  // ellipse (cx, cy, rx, ry, rot)
        ];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10, "Residual: {}", residuals[0]);
    }

    #[test]
    fn test_point_on_ellipse_axis_aligned() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let ellipse = ParamRange { start: 2, count: 5 };

        let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

        // Ellipse at origin with rx=4, ry=3, no rotation
        // Point at (4, 0) should be on ellipse
        let params = vec![
            4.0, 0.0,        // point at major axis endpoint
            0.0, 0.0, 4.0, 3.0, 0.0  // ellipse
        ];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10, "Residual: {}", residuals[0]);

        // Point at (0, 3) should be on ellipse
        let params2 = vec![
            0.0, 3.0,        // point at minor axis endpoint
            0.0, 0.0, 4.0, 3.0, 0.0
        ];

        let residuals2 = constraint.residuals(&params2);
        assert!(residuals2[0].abs() < 1e-10, "Residual: {}", residuals2[0]);
    }

    #[test]
    fn test_point_on_rotated_ellipse() {
        use std::f64::consts::PI;

        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let ellipse = ParamRange { start: 2, count: 5 };

        let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

        // Ellipse rotated by π/2 (90 degrees)
        // rx=4, ry=3, rotation=π/2
        // After rotation, major axis is along y, minor along x
        // Point (0, 4) should be on ellipse (was (4,0) before rotation)
        let params = vec![
            0.0, 4.0,        // point
            0.0, 0.0, 4.0, 3.0, PI / 2.0  // ellipse
        ];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10, "Residual: {}", residuals[0]);

        // Point (-3, 0) should be on ellipse (was (0,-3) before rotation)
        let params2 = vec![
            -3.0, 0.0,
            0.0, 0.0, 4.0, 3.0, PI / 2.0
        ];

        let residuals2 = constraint.residuals(&params2);
        assert!(residuals2[0].abs() < 1e-10, "Residual: {}", residuals2[0]);
    }

    #[test]
    fn test_point_not_on_ellipse() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let ellipse = ParamRange { start: 2, count: 5 };

        let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

        // Point far from ellipse
        let params = vec![
            10.0, 10.0,      // point
            0.0, 0.0, 4.0, 3.0, 0.0
        ];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() > 1.0, "Should have large residual");
    }

    #[test]
    fn test_jacobian_finite() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let ellipse = ParamRange { start: 2, count: 5 };

        let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

        let params = vec![
            3.0, 2.0,        // arbitrary point
            0.0, 0.0, 4.0, 3.0, 0.5
        ];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 7);

        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(val.is_finite(), "Non-finite Jacobian value at col {}", col);
        }
    }

    #[test]
    fn test_jacobian_numerical_verification() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let ellipse = ParamRange { start: 2, count: 5 };

        let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

        let params = vec![
            2.5, 1.8,        // arbitrary point
            0.0, 0.0, 4.0, 3.0, 0.3
        ];

        let jac = constraint.jacobian(&params);
        let h = 1e-7;

        for &(row, col, analytical) in &jac {
            assert_eq!(row, 0);

            let mut params_plus = params.clone();
            params_plus[col] += h;
            let res_plus = constraint.residuals(&params_plus)[0];

            let mut params_minus = params.clone();
            params_minus[col] -= h;
            let res_minus = constraint.residuals(&params_minus)[0];

            let numerical = (res_plus - res_minus) / (2.0 * h);
            let error = (analytical - numerical).abs();
            let rel_error = if numerical.abs() > 1e-6 {
                error / numerical.abs()
            } else {
                error
            };

            assert!(
                rel_error < 1e-4,
                "Jacobian mismatch for col {}: analytical={}, numerical={}, error={}",
                col, analytical, numerical, error
            );
        }
    }

    #[test]
    fn test_dependencies() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 10, count: 2 };
        let ellipse = ParamRange { start: 20, count: 5 };

        let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

        let deps = constraint.dependencies();
        assert_eq!(deps.len(), 7);
        assert_eq!(deps, &[10, 11, 20, 21, 22, 23, 24]);
    }

    #[test]
    fn test_constraint_metadata() {
        let id = ConstraintId(42);
        let point = ParamRange { start: 0, count: 2 };
        let ellipse = ParamRange { start: 2, count: 5 };

        let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

        assert_eq!(constraint.id(), ConstraintId(42));
        assert_eq!(constraint.name(), "PointOnEllipse");
        assert_eq!(constraint.equation_count(), 1);
        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::High);
    }
}
