//! Point on curve constraints for Bezier curves.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Point lies on cubic Bezier curve at parameter t.
///
/// # Entity Parameters
/// - point: Point2D [x, y] — 2 params
/// - bezier: CubicBezier2D [x0,y0, x1,y1, x2,y2, x3,y3] — 8 params
/// - t_param: index of a Scalar entity's parameter (curve parameter t ∈ [0,1])
///
/// # Equations (2 for 2D)
/// Bezier evaluation at t:
/// ```text
/// B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
/// ```
/// Residuals:
/// ```text
/// px - Bx(t) = 0
/// py - By(t) = 0
/// ```
///
/// # Jacobian
/// Derivatives with respect to:
/// - point params (px, py): -1 for each equation
/// - bezier control points: Bernstein basis coefficients
/// - t_param: derivative of Bezier curve B'(t)
#[derive(Clone, Debug)]
pub struct PointOnBezierConstraint {
    id: ConstraintId,
    point: ParamRange,
    bezier: ParamRange,
    t_param: usize,
    deps: Vec<usize>,
}

impl PointOnBezierConstraint {
    /// Create a new point-on-bezier constraint.
    ///
    /// # Arguments
    /// - `id`: Unique constraint identifier
    /// - `point`: ParamRange for Point2D [x, y]
    /// - `bezier`: ParamRange for CubicBezier2D [x0,y0,...,x3,y3]
    /// - `t_param`: parameter index for the auxiliary t variable
    pub fn new(id: ConstraintId, point: ParamRange, bezier: ParamRange, t_param: usize) -> Self {
        assert_eq!(point.count, 2, "Point must have 2 parameters");
        assert_eq!(bezier.count, 8, "CubicBezier2D must have 8 parameters");

        let mut deps = Vec::with_capacity(11);
        deps.extend(point.iter());
        deps.extend(bezier.iter());
        deps.push(t_param);

        Self {
            id,
            point,
            bezier,
            t_param,
            deps,
        }
    }

    /// Evaluate cubic Bezier curve at parameter t.
    ///
    /// Returns (x, y) coordinates.
    fn eval_bezier(&self, params: &[f64], t: f64) -> (f64, f64) {
        let p0_x = params[self.bezier.start];
        let p0_y = params[self.bezier.start + 1];
        let p1_x = params[self.bezier.start + 2];
        let p1_y = params[self.bezier.start + 3];
        let p2_x = params[self.bezier.start + 4];
        let p2_y = params[self.bezier.start + 5];
        let p3_x = params[self.bezier.start + 6];
        let p3_y = params[self.bezier.start + 7];

        // Bernstein basis functions
        let one_minus_t = 1.0 - t;
        let b0 = one_minus_t * one_minus_t * one_minus_t;
        let b1 = 3.0 * one_minus_t * one_minus_t * t;
        let b2 = 3.0 * one_minus_t * t * t;
        let b3 = t * t * t;

        let x = b0 * p0_x + b1 * p1_x + b2 * p2_x + b3 * p3_x;
        let y = b0 * p0_y + b1 * p1_y + b2 * p2_y + b3 * p3_y;

        (x, y)
    }

    /// Evaluate derivative of cubic Bezier curve at parameter t.
    ///
    /// Returns (dx/dt, dy/dt).
    fn eval_bezier_derivative(&self, params: &[f64], t: f64) -> (f64, f64) {
        let p0_x = params[self.bezier.start];
        let p0_y = params[self.bezier.start + 1];
        let p1_x = params[self.bezier.start + 2];
        let p1_y = params[self.bezier.start + 3];
        let p2_x = params[self.bezier.start + 4];
        let p2_y = params[self.bezier.start + 5];
        let p3_x = params[self.bezier.start + 6];
        let p3_y = params[self.bezier.start + 7];

        // B'(t) = 3[(1-t)²(P1-P0) + 2(1-t)t(P2-P1) + t²(P3-P2)]
        let one_minus_t = 1.0 - t;
        let c0 = 3.0 * one_minus_t * one_minus_t;
        let c1 = 6.0 * one_minus_t * t;
        let c2 = 3.0 * t * t;

        let dx = c0 * (p1_x - p0_x) + c1 * (p2_x - p1_x) + c2 * (p3_x - p2_x);
        let dy = c0 * (p1_y - p0_y) + c1 * (p2_y - p1_y) + c2 * (p3_y - p2_y);

        (dx, dy)
    }
}

impl Constraint for PointOnBezierConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "PointOnBezier"
    }

    fn equation_count(&self) -> usize {
        2  // 2D
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let px = params[self.point.start];
        let py = params[self.point.start + 1];
        let t = params[self.t_param];

        let (bx, by) = self.eval_bezier(params, t);

        vec![
            px - bx,
            py - by,
        ]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let t = params[self.t_param];

        // Bernstein basis functions for position
        let one_minus_t = 1.0 - t;
        let b0 = one_minus_t * one_minus_t * one_minus_t;
        let b1 = 3.0 * one_minus_t * one_minus_t * t;
        let b2 = 3.0 * one_minus_t * t * t;
        let b3 = t * t * t;

        // Derivative of Bezier w.r.t. t
        let (dbx_dt, dby_dt) = self.eval_bezier_derivative(params, t);

        let mut jac = Vec::with_capacity(20);

        // === Equation 0: px - Bx(t) = 0 ===

        // d/dpx = 1
        jac.push((0, self.point.start, 1.0));

        // d/dP0.x = -b0
        jac.push((0, self.bezier.start, -b0));

        // d/dP1.x = -b1
        jac.push((0, self.bezier.start + 2, -b1));

        // d/dP2.x = -b2
        jac.push((0, self.bezier.start + 4, -b2));

        // d/dP3.x = -b3
        jac.push((0, self.bezier.start + 6, -b3));

        // d/dt = -dBx/dt
        jac.push((0, self.t_param, -dbx_dt));

        // === Equation 1: py - By(t) = 0 ===

        // d/dpy = 1
        jac.push((1, self.point.start + 1, 1.0));

        // d/dP0.y = -b0
        jac.push((1, self.bezier.start + 1, -b0));

        // d/dP1.y = -b1
        jac.push((1, self.bezier.start + 3, -b1));

        // d/dP2.y = -b2
        jac.push((1, self.bezier.start + 5, -b2));

        // d/dP3.y = -b3
        jac.push((1, self.bezier.start + 7, -b3));

        // d/dt = -dBy/dt
        jac.push((1, self.t_param, -dby_dt));

        jac
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::High  // Cubic polynomials
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_on_bezier_at_t0() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        // Bezier curve with control points
        let mut params = vec![0.0; 11];
        params[0] = 1.0; params[1] = 2.0;  // point (should match P0)
        params[2] = 1.0; params[3] = 2.0;  // P0
        params[4] = 3.0; params[5] = 4.0;  // P1
        params[6] = 5.0; params[7] = 6.0;  // P2
        params[8] = 7.0; params[9] = 8.0;  // P3
        params[10] = 0.0;  // t = 0

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10, "x residual: {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "y residual: {}", residuals[1]);
    }

    #[test]
    fn test_point_on_bezier_at_t1() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        // Point should match P3 when t=1
        let mut params = vec![0.0; 11];
        params[0] = 7.0; params[1] = 8.0;  // point (should match P3)
        params[2] = 1.0; params[3] = 2.0;  // P0
        params[4] = 3.0; params[5] = 4.0;  // P1
        params[6] = 5.0; params[7] = 6.0;  // P2
        params[8] = 7.0; params[9] = 8.0;  // P3
        params[10] = 1.0;  // t = 1

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10, "x residual: {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "y residual: {}", residuals[1]);
    }

    #[test]
    fn test_point_on_bezier_at_midpoint() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        // Linear bezier (straight line): P0=(0,0), P1=(1,0), P2=(2,0), P3=(3,0)
        // At t=0.5, B(0.5) = 1.5
        let mut params = vec![0.0; 11];
        params[0] = 1.5; params[1] = 0.0;  // point at midpoint
        params[2] = 0.0; params[3] = 0.0;  // P0
        params[4] = 1.0; params[5] = 0.0;  // P1
        params[6] = 2.0; params[7] = 0.0;  // P2
        params[8] = 3.0; params[9] = 0.0;  // P3
        params[10] = 0.5;  // t = 0.5

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10, "x residual: {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "y residual: {}", residuals[1]);
    }

    #[test]
    fn test_point_not_on_bezier() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        // Point far from curve
        let mut params = vec![0.0; 11];
        params[0] = 100.0; params[1] = 100.0;  // point far away
        params[2] = 0.0; params[3] = 0.0;  // P0
        params[4] = 1.0; params[5] = 0.0;  // P1
        params[6] = 2.0; params[7] = 0.0;  // P2
        params[8] = 3.0; params[9] = 0.0;  // P3
        params[10] = 0.5;  // t = 0.5

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() > 50.0, "Should have large x residual");
        assert!(residuals[1].abs() > 50.0, "Should have large y residual");
    }

    #[test]
    fn test_eval_bezier() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        let mut params = vec![0.0; 11];
        params[2] = 0.0; params[3] = 0.0;  // P0
        params[4] = 0.0; params[5] = 1.0;  // P1
        params[6] = 1.0; params[7] = 1.0;  // P2
        params[8] = 1.0; params[9] = 0.0;  // P3

        // At t=0, should be P0
        let (x0, y0) = constraint.eval_bezier(&params, 0.0);
        assert!((x0 - 0.0).abs() < 1e-10);
        assert!((y0 - 0.0).abs() < 1e-10);

        // At t=1, should be P3
        let (x1, y1) = constraint.eval_bezier(&params, 1.0);
        assert!((x1 - 1.0).abs() < 1e-10);
        assert!((y1 - 0.0).abs() < 1e-10);

        // At t=0.5, should be somewhere in middle
        let (x_mid, y_mid) = constraint.eval_bezier(&params, 0.5);
        assert!(x_mid > 0.0 && x_mid < 1.0);
        assert!(y_mid >= 0.0);
    }

    #[test]
    fn test_jacobian_numerical_verification() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        let mut params = vec![0.0; 11];
        params[0] = 0.5; params[1] = 0.6;  // point
        params[2] = 0.0; params[3] = 0.0;  // P0
        params[4] = 0.2; params[5] = 0.8;  // P1
        params[6] = 0.8; params[7] = 0.9;  // P2
        params[8] = 1.0; params[9] = 0.1;  // P3
        params[10] = 0.4;  // t

        let jac = constraint.jacobian(&params);
        let h = 1e-7;

        for &(row, col, analytical) in &jac {
            let mut params_plus = params.clone();
            params_plus[col] += h;
            let res_plus = constraint.residuals(&params_plus);

            let mut params_minus = params.clone();
            params_minus[col] -= h;
            let res_minus = constraint.residuals(&params_minus);

            let numerical = (res_plus[row] - res_minus[row]) / (2.0 * h);
            let error = (analytical - numerical).abs();
            let rel_error = if numerical.abs() > 1e-6 {
                error / numerical.abs()
            } else {
                error
            };

            assert!(
                rel_error < 1e-4,
                "Jacobian mismatch at ({},{}): analytical={}, numerical={}, error={}",
                row, col, analytical, numerical, error
            );
        }
    }

    #[test]
    fn test_jacobian_structure() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        let params = vec![0.0; 11];
        let jac = constraint.jacobian(&params);

        // Should have 12 entries:
        // - 2 for point coords (px, py)
        // - 8 for bezier control points (only affecting their respective equations)
        // - 2 for t_param (affects both equations)
        assert_eq!(jac.len(), 12);

        // Check that all values are finite
        for (row, col, val) in &jac {
            assert!(val.is_finite(), "Non-finite Jacobian at ({},{})", row, col);
        }
    }

    #[test]
    fn test_dependencies() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 10, count: 2 };
        let bezier = ParamRange { start: 20, count: 8 };
        let t_param = 50;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        let deps = constraint.dependencies();
        assert_eq!(deps.len(), 11);

        // Should include: point (2), bezier (8), t_param (1)
        assert_eq!(deps[0..2], [10, 11]);  // point
        assert_eq!(deps[2..10], [20, 21, 22, 23, 24, 25, 26, 27]);  // bezier
        assert_eq!(deps[10], 50);  // t_param
    }

    #[test]
    fn test_constraint_metadata() {
        let id = ConstraintId(42);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        assert_eq!(constraint.id(), ConstraintId(42));
        assert_eq!(constraint.name(), "PointOnBezier");
        assert_eq!(constraint.equation_count(), 2);
        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::High);
    }

    #[test]
    fn test_derivative_at_endpoints() {
        let id = ConstraintId(0);
        let point = ParamRange { start: 0, count: 2 };
        let bezier = ParamRange { start: 2, count: 8 };
        let t_param = 10;

        let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

        let mut params = vec![0.0; 11];
        params[2] = 0.0; params[3] = 0.0;  // P0
        params[4] = 1.0; params[5] = 0.0;  // P1
        params[6] = 2.0; params[7] = 0.0;  // P2
        params[8] = 3.0; params[9] = 0.0;  // P3

        // At t=0, derivative should be 3*(P1-P0)
        let (dx0, dy0) = constraint.eval_bezier_derivative(&params, 0.0);
        assert!((dx0 - 3.0).abs() < 1e-10);
        assert!((dy0 - 0.0).abs() < 1e-10);

        // At t=1, derivative should be 3*(P3-P2)
        let (dx1, dy1) = constraint.eval_bezier_derivative(&params, 1.0);
        assert!((dx1 - 3.0).abs() < 1e-10);
        assert!((dy1 - 0.0).abs() < 1e-10);
    }
}
