//! Symmetric constraint: two points are symmetric about a center point.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Symmetric constraint: two points are symmetric about a center point.
///
/// This constraint enforces that P1 and P2 are mirror images with respect
/// to the center point C, meaning C is the midpoint of P1 and P2.
///
/// # Equations
///
/// For each dimension k: `P1[k] + P2[k] - 2*C[k] = 0`
///
/// This is equivalent to: `(P1 + P2) / 2 = C`
///
/// # Jacobian
///
/// - `d(residual_k)/d(P1[k]) = 1`
/// - `d(residual_k)/d(P2[k]) = 1`
/// - `d(residual_k)/d(C[k]) = -2`
#[derive(Clone, Debug)]
pub struct SymmetricConstraint<const D: usize> {
    /// Index of first point.
    pub point1: usize,
    /// Index of second point.
    pub point2: usize,
    /// Index of center point (axis of symmetry).
    pub center: usize,
}

impl<const D: usize> SymmetricConstraint<D> {
    /// Create a new symmetric constraint.
    pub fn new(point1: usize, point2: usize, center: usize) -> Self {
        Self {
            point1,
            point2,
            center,
        }
    }
}

impl<const D: usize> GeometricConstraint<D> for SymmetricConstraint<D> {
    fn equation_count(&self) -> usize {
        D
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let p1 = get_point(points, self.point1);
        let p2 = get_point(points, self.point2);
        let c = get_point(points, self.center);

        let mut residuals = Vec::with_capacity(D);
        for k in 0..D {
            residuals.push(p1.get(k) + p2.get(k) - 2.0 * c.get(k));
        }
        residuals
    }

    fn jacobian(&self, _points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::with_capacity(D * 3);

        for k in 0..D {
            entries.push((k, var_col::<D>(self.point1, k), 1.0));   // d/dP1[k]
            entries.push((k, var_col::<D>(self.point2, k), 1.0));   // d/dP2[k]
            entries.push((k, var_col::<D>(self.center, k), -2.0));  // d/dC[k]
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2, self.center]
    }

    fn name(&self) -> &'static str {
        "Symmetric"
    }
}

/// Symmetric about line constraint: two points are mirror images across a line axis.
///
/// Two conditions must be satisfied:
/// 1. Midpoint of P1-P2 lies on the axis
/// 2. Line P1-P2 is perpendicular to axis
///
/// This is a 2D-only constraint.
#[derive(Clone, Debug)]
pub struct SymmetricAboutLineConstraint {
    /// Index of first point.
    pub point1: usize,
    /// Index of second point.
    pub point2: usize,
    /// Index of axis start point.
    pub axis_start: usize,
    /// Index of axis end point.
    pub axis_end: usize,
}

impl SymmetricAboutLineConstraint {
    /// Create a new symmetric-about-line constraint.
    pub fn new(point1: usize, point2: usize, axis_start: usize, axis_end: usize) -> Self {
        Self {
            point1,
            point2,
            axis_start,
            axis_end,
        }
    }
}

impl GeometricConstraint<2> for SymmetricAboutLineConstraint {
    fn equation_count(&self) -> usize {
        2
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let p1 = get_point(points, self.point1);
        let p2 = get_point(points, self.point2);
        let a = get_point(points, self.axis_start);
        let b = get_point(points, self.axis_end);

        // Midpoint of P1-P2
        let mx = (p1.get(0) + p2.get(0)) / 2.0;
        let my = (p1.get(1) + p2.get(1)) / 2.0;

        // Axis direction
        let ax_dx = b.get(0) - a.get(0);
        let ax_dy = b.get(1) - a.get(1);

        // P1-P2 direction
        let p_dx = p2.get(0) - p1.get(0);
        let p_dy = p2.get(1) - p1.get(1);

        // Condition 1: Midpoint on axis (cross product)
        // (M - A) x (B - A) = 0
        let mid_cross = (mx - a.get(0)) * ax_dy - (my - a.get(1)) * ax_dx;

        // Condition 2: P1-P2 perpendicular to axis (dot product)
        // (P2 - P1) . (B - A) = 0
        let perp_dot = p_dx * ax_dx + p_dy * ax_dy;

        vec![mid_cross, perp_dot]
    }

    fn jacobian(&self, points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        let p1 = get_point(points, self.point1);
        let p2 = get_point(points, self.point2);
        let a = get_point(points, self.axis_start);
        let b = get_point(points, self.axis_end);

        let mx = (p1.get(0) + p2.get(0)) / 2.0;
        let my = (p1.get(1) + p2.get(1)) / 2.0;

        let ax_dx = b.get(0) - a.get(0);
        let ax_dy = b.get(1) - a.get(1);

        let p_dx = p2.get(0) - p1.get(0);
        let p_dy = p2.get(1) - p1.get(1);

        // Equation 1: mid_cross = (Mx - Ax)(By - Ay) - (My - Ay)(Bx - Ax)
        // where Mx = (P1x + P2x)/2, My = (P1y + P2y)/2
        //
        // d/dP1x = 0.5 * ax_dy, d/dP1y = -0.5 * ax_dx
        // d/dP2x = 0.5 * ax_dy, d/dP2y = -0.5 * ax_dx
        // d/dAx = -ax_dy + (My - Ay) = My - By (simplified)
        // d/dAy = ax_dx - (Mx - Ax) = Bx - Mx
        // d/dBx = -(My - Ay)
        // d/dBy = (Mx - Ax)

        // Equation 2: perp_dot = (P2x - P1x)(Bx - Ax) + (P2y - P1y)(By - Ay)
        // d/dP1x = -ax_dx, d/dP1y = -ax_dy
        // d/dP2x = ax_dx,  d/dP2y = ax_dy
        // d/dAx = -p_dx,   d/dAy = -p_dy
        // d/dBx = p_dx,    d/dBy = p_dy

        vec![
            // Equation 1: midpoint on axis
            (0, var_col::<2>(self.point1, 0), 0.5 * ax_dy),
            (0, var_col::<2>(self.point1, 1), -0.5 * ax_dx),
            (0, var_col::<2>(self.point2, 0), 0.5 * ax_dy),
            (0, var_col::<2>(self.point2, 1), -0.5 * ax_dx),
            (0, var_col::<2>(self.axis_start, 0), my - b.get(1)),
            (0, var_col::<2>(self.axis_start, 1), b.get(0) - mx),
            (0, var_col::<2>(self.axis_end, 0), -(my - a.get(1))),
            (0, var_col::<2>(self.axis_end, 1), mx - a.get(0)),
            // Equation 2: P1-P2 perpendicular to axis
            (1, var_col::<2>(self.point1, 0), -ax_dx),
            (1, var_col::<2>(self.point1, 1), -ax_dy),
            (1, var_col::<2>(self.point2, 0), ax_dx),
            (1, var_col::<2>(self.point2, 1), ax_dy),
            (1, var_col::<2>(self.axis_start, 0), -p_dx),
            (1, var_col::<2>(self.axis_start, 1), -p_dy),
            (1, var_col::<2>(self.axis_end, 0), p_dx),
            (1, var_col::<2>(self.axis_end, 1), p_dy),
        ]
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2, self.axis_start, self.axis_end]
    }

    fn name(&self) -> &'static str {
        "SymmetricAboutLine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_symmetric_2d_satisfied() {
        let constraint = SymmetricConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(0.0, 0.0),  // P1
            Point2D::new(10.0, 10.0), // P2
            Point2D::new(5.0, 5.0),  // Center (midpoint)
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_2d_not_satisfied() {
        let constraint = SymmetricConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(0.0, 0.0),  // P1
            Point2D::new(10.0, 10.0), // P2
            Point2D::new(0.0, 0.0),  // Not at midpoint
        ];

        let residuals = constraint.residuals(&points);
        // P1 + P2 - 2*C = (0+10) - 0 = 10
        assert!((residuals[0] - 10.0).abs() < 1e-10);
        assert!((residuals[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_3d_satisfied() {
        let constraint = SymmetricConstraint::<3>::new(0, 1, 2);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),   // P1
            Point3D::new(10.0, 10.0, 10.0), // P2
            Point3D::new(5.0, 5.0, 5.0),   // Center
        ];

        let residuals = constraint.residuals(&points);
        for r in &residuals {
            assert!(r.abs() < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_jacobian() {
        let constraint = SymmetricConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 10.0),
            Point2D::new(5.0, 5.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 6); // 2 equations * 3 terms each

        // Check entries
        let expected = vec![
            (0, 0, 1.0),   // d(eq0)/d(P1x)
            (0, 2, 1.0),   // d(eq0)/d(P2x)
            (0, 4, -2.0),  // d(eq0)/d(Cx)
            (1, 1, 1.0),   // d(eq1)/d(P1y)
            (1, 3, 1.0),   // d(eq1)/d(P2y)
            (1, 5, -2.0),  // d(eq1)/d(Cy)
        ];

        for exp in &expected {
            assert!(jac.contains(exp), "Missing entry {:?}", exp);
        }
    }

    #[test]
    fn test_symmetric_about_line_satisfied() {
        let constraint = SymmetricAboutLineConstraint::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(-1.0, 0.0),  // P1
            Point2D::new(1.0, 0.0),   // P2 (symmetric about y-axis)
            Point2D::new(0.0, -5.0),  // Axis start (y-axis)
            Point2D::new(0.0, 5.0),   // Axis end (y-axis)
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10, "midpoint residual: {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "perpendicular residual: {}", residuals[1]);
    }

    #[test]
    fn test_symmetric_about_line_diagonal_axis() {
        let constraint = SymmetricAboutLineConstraint::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 1.0),  // P1
            Point2D::new(1.0, 0.0),  // P2 (symmetric about y=x)
            Point2D::new(0.0, 0.0),  // Axis start
            Point2D::new(1.0, 1.0),  // Axis end (y=x line)
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10, "midpoint residual: {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "perpendicular residual: {}", residuals[1]);
    }

    #[test]
    fn test_variable_indices_point() {
        let constraint = SymmetricConstraint::<2>::new(5, 10, 15);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![5, 10, 15]);
    }

    #[test]
    fn test_variable_indices_line() {
        let constraint = SymmetricAboutLineConstraint::new(5, 10, 15, 20);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![5, 10, 15, 20]);
    }
}
