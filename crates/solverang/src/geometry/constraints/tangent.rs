//! Tangent constraints for line-circle and circle-circle tangency.

use crate::geometry::point::{Point, MIN_EPSILON};
use crate::geometry::circle::TangentType;
use super::{get_point, var_col, GeometricConstraint};

/// Line tangent to circle constraint.
///
/// The perpendicular distance from circle center to line equals radius.
/// Perpendicular distance = `|(C - A) x (B - A)| / |B - A|`
///
/// # Equation
///
/// `|(C - A) x (B - A)| / |B - A| - radius = 0`
#[derive(Clone, Debug)]
pub struct LineTangentConstraint {
    /// Index of line start point.
    pub line_start: usize,
    /// Index of line end point.
    pub line_end: usize,
    /// Index of circle center.
    pub center: usize,
    /// Circle radius.
    pub radius: f64,
}

impl LineTangentConstraint {
    /// Create a new line-tangent-circle constraint.
    pub fn new(line_start: usize, line_end: usize, center: usize, radius: f64) -> Self {
        Self {
            line_start,
            line_end,
            center,
            radius,
        }
    }
}

impl GeometricConstraint<2> for LineTangentConstraint {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let a = get_point(points, self.line_start);
        let b = get_point(points, self.line_end);
        let c = get_point(points, self.center);

        // Line direction vector
        let dx = b.get(0) - a.get(0);
        let dy = b.get(1) - a.get(1);
        let line_len = (dx * dx + dy * dy).sqrt().max(MIN_EPSILON);

        // Vector from A to C
        let acx = c.get(0) - a.get(0);
        let acy = c.get(1) - a.get(1);

        // Cross product (2D): |AC x AB| = |acx * dy - acy * dx|
        let cross = (acx * dy - acy * dx).abs();

        // Perpendicular distance = |cross| / line_len
        let perp_dist = cross / line_len;

        vec![perp_dist - self.radius]
    }

    fn jacobian(&self, points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        let a = get_point(points, self.line_start);
        let b = get_point(points, self.line_end);
        let c = get_point(points, self.center);

        let dx = b.get(0) - a.get(0);
        let dy = b.get(1) - a.get(1);
        let line_len = (dx * dx + dy * dy).sqrt().max(MIN_EPSILON);
        let line_len_sq = line_len * line_len;

        let acx = c.get(0) - a.get(0);
        let acy = c.get(1) - a.get(1);

        // Raw cross product (can be negative)
        let cross_raw = acx * dy - acy * dx;
        let cross_sign = if cross_raw >= 0.0 { 1.0 } else { -1.0 };
        let cross = cross_raw.abs().max(MIN_EPSILON);

        // perp_dist = cross / line_len
        // f = perp_dist - radius
        let ll3 = line_len * line_len_sq;

        // Jacobian via quotient rule on f = |cross| / line_len - radius.
        //
        // d(cross_raw)/d(ax) = acy - dy,   d(line_len)/d(ax) = -dx/line_len
        // d(cross_raw)/d(ay) = dx - acx,   d(line_len)/d(ay) = -dy/line_len
        // d(cross_raw)/d(bx) = -acy,        d(line_len)/d(bx) = dx/line_len
        // d(cross_raw)/d(by) = acx,          d(line_len)/d(by) = dy/line_len
        // d(cross_raw)/d(cx) = dy,           d(line_len)/d(cx) = 0
        // d(cross_raw)/d(cy) = -dx,          d(line_len)/d(cy) = 0
        //
        // df/dv = cross_sign * d(cross_raw)/dv / line_len
        //       - cross * d(line_len)/dv / line_len^2
        vec![
            (0, var_col::<2>(self.line_start, 0), cross_sign * (acy - dy) / line_len + cross * dx / ll3),
            (0, var_col::<2>(self.line_start, 1), cross_sign * (dx - acx) / line_len + cross * dy / ll3),
            (0, var_col::<2>(self.line_end, 0), -cross_sign * acy / line_len - cross * dx / ll3),
            (0, var_col::<2>(self.line_end, 1), cross_sign * acx / line_len - cross * dy / ll3),
            (0, var_col::<2>(self.center, 0), cross_sign * dy / line_len),
            (0, var_col::<2>(self.center, 1), -cross_sign * dx / line_len),
        ]
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.line_start, self.line_end, self.center]
    }

    fn name(&self) -> &'static str {
        "LineTangent"
    }
}

/// Circle-circle tangent constraint.
///
/// Two circles are tangent when:
/// - External: center distance = r1 + r2
/// - Internal: center distance = |r1 - r2|
///
/// # Equation
///
/// `|C2 - C1| - target = 0`
///
/// Where target = r1 + r2 (external) or |r1 - r2| (internal)
#[derive(Clone, Debug)]
pub struct CircleTangentConstraint {
    /// Index of first circle center.
    pub center1: usize,
    /// Radius of first circle.
    pub radius1: f64,
    /// Index of second circle center.
    pub center2: usize,
    /// Radius of second circle.
    pub radius2: f64,
    /// Type of tangency (external or internal).
    pub tangent_type: TangentType,
}

impl CircleTangentConstraint {
    /// Create a new circle-tangent constraint.
    pub fn new(
        center1: usize,
        radius1: f64,
        center2: usize,
        radius2: f64,
        tangent_type: TangentType,
    ) -> Self {
        Self {
            center1,
            radius1,
            center2,
            radius2,
            tangent_type,
        }
    }

    /// Create an external tangent constraint.
    pub fn external(center1: usize, radius1: f64, center2: usize, radius2: f64) -> Self {
        Self::new(center1, radius1, center2, radius2, TangentType::External)
    }

    /// Create an internal tangent constraint.
    pub fn internal(center1: usize, radius1: f64, center2: usize, radius2: f64) -> Self {
        Self::new(center1, radius1, center2, radius2, TangentType::Internal)
    }

    fn target_distance(&self) -> f64 {
        self.tangent_type.target_distance(self.radius1, self.radius2)
    }
}

impl GeometricConstraint<2> for CircleTangentConstraint {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let c1 = get_point(points, self.center1);
        let c2 = get_point(points, self.center2);

        let dist = c1.distance_to(&c2);
        vec![dist - self.target_distance()]
    }

    fn jacobian(&self, points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        let c1 = get_point(points, self.center1);
        let c2 = get_point(points, self.center2);

        let dist = c1.safe_distance_to(&c2);

        let dx = c2.get(0) - c1.get(0);
        let dy = c2.get(1) - c1.get(1);

        // f = dist - target
        // d/dC1x = -dx/dist, d/dC1y = -dy/dist
        // d/dC2x = dx/dist,  d/dC2y = dy/dist
        vec![
            (0, var_col::<2>(self.center1, 0), -dx / dist),
            (0, var_col::<2>(self.center1, 1), -dy / dist),
            (0, var_col::<2>(self.center2, 0), dx / dist),
            (0, var_col::<2>(self.center2, 1), dy / dist),
        ]
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.center1, self.center2]
    }

    fn name(&self) -> &'static str {
        "CircleTangent"
    }
}

impl GeometricConstraint<3> for CircleTangentConstraint {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<3>]) -> Vec<f64> {
        let c1 = get_point(points, self.center1);
        let c2 = get_point(points, self.center2);

        let dist = c1.distance_to(&c2);
        vec![dist - self.target_distance()]
    }

    fn jacobian(&self, points: &[Point<3>]) -> Vec<(usize, usize, f64)> {
        let c1 = get_point(points, self.center1);
        let c2 = get_point(points, self.center2);

        let dist = c1.safe_distance_to(&c2);

        let mut entries = Vec::with_capacity(6);

        for k in 0..3 {
            let diff = c2.get(k) - c1.get(k);
            let grad = diff / dist;

            entries.push((0, var_col::<3>(self.center1, k), -grad));
            entries.push((0, var_col::<3>(self.center2, k), grad));
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.center1, self.center2]
    }

    fn name(&self) -> &'static str {
        "CircleTangent3D"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_line_tangent_satisfied() {
        let constraint = LineTangentConstraint::new(0, 1, 2, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),   // Line start
            Point2D::new(10.0, 0.0),  // Line end (horizontal line y=0)
            Point2D::new(5.0, 5.0),   // Center at distance 5 from line
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_line_tangent_not_satisfied() {
        let constraint = LineTangentConstraint::new(0, 1, 2, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),   // Line start
            Point2D::new(10.0, 0.0),  // Line end (horizontal line y=0)
            Point2D::new(5.0, 10.0),  // Center at distance 10 from line
        ];

        let residuals = constraint.residuals(&points);
        // Distance is 10, radius is 5, residual = 10 - 5 = 5
        assert!((residuals[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_tangent_external_satisfied() {
        let constraint = CircleTangentConstraint::external(0, 3.0, 1, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),  // Center 1
            Point2D::new(8.0, 0.0),  // Center 2 at distance 8 = 3 + 5
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_circle_tangent_internal_satisfied() {
        let constraint = CircleTangentConstraint::internal(0, 5.0, 1, 3.0);
        let points = vec![
            Point2D::new(0.0, 0.0),  // Center 1 (larger circle)
            Point2D::new(2.0, 0.0),  // Center 2 at distance 2 = |5 - 3|
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_circle_tangent_external_not_satisfied() {
        let constraint = CircleTangentConstraint::external(0, 3.0, 1, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),   // Center 1
            Point2D::new(10.0, 0.0),  // Center 2 at distance 10 (not 8)
        ];

        let residuals = constraint.residuals(&points);
        // residual = 10 - 8 = 2
        assert!((residuals[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_tangent_3d() {
        let constraint = CircleTangentConstraint::external(0, 3.0, 1, 4.0);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),  // Center 1
            Point3D::new(7.0, 0.0, 0.0),  // Center 2 at distance 7 = 3 + 4
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_line_tangent_jacobian_finite() {
        let constraint = LineTangentConstraint::new(0, 1, 2, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(5.0, 5.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 6);

        for (row, _col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_circle_tangent_jacobian_finite() {
        let constraint = CircleTangentConstraint::external(0, 3.0, 1, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(8.0, 0.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 4);

        for (row, _col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_variable_indices_line() {
        let constraint = LineTangentConstraint::new(5, 10, 15, 5.0);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![5, 10, 15]);
    }

    #[test]
    fn test_variable_indices_circle() {
        let constraint = CircleTangentConstraint::external(5, 3.0, 10, 5.0);
        let indices = <CircleTangentConstraint as GeometricConstraint<2>>::variable_indices(&constraint);
        assert_eq!(indices, vec![5, 10]);
    }
}
