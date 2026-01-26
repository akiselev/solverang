//! Horizontal constraint: two points must have the same y-coordinate (2D only).

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Horizontal constraint: p1.y = p2.y.
///
/// This constraint enforces that two points have the same y-coordinate,
/// making the line between them horizontal.
///
/// This is a 2D-only constraint.
///
/// # Equation
///
/// `p2.y - p1.y = 0`
///
/// # Jacobian
///
/// - `d(residual)/d(p1.y) = -1`
/// - `d(residual)/d(p2.y) = 1`
#[derive(Clone, Debug)]
pub struct HorizontalConstraint {
    /// Index of first point.
    pub point1: usize,
    /// Index of second point.
    pub point2: usize,
}

impl HorizontalConstraint {
    /// Create a new horizontal constraint.
    pub fn new(point1: usize, point2: usize) -> Self {
        Self { point1, point2 }
    }
}

impl GeometricConstraint<2> for HorizontalConstraint {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let p1 = get_point(points, self.point1);
        let p2 = get_point(points, self.point2);

        vec![p2.get(1) - p1.get(1)] // y2 - y1
    }

    fn jacobian(&self, _points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, var_col::<2>(self.point1, 1), -1.0), // d/dy1
            (0, var_col::<2>(self.point2, 1), 1.0),  // d/dy2
        ]
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }

    fn name(&self) -> &'static str {
        "Horizontal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point2D;

    #[test]
    fn test_horizontal_satisfied() {
        let constraint = HorizontalConstraint::new(0, 1);
        let points = vec![
            Point2D::new(0.0, 5.0),
            Point2D::new(10.0, 5.0),
        ];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_horizontal_not_satisfied() {
        let constraint = HorizontalConstraint::new(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 5.0),
        ];

        let residuals = constraint.residuals(&points);
        assert!((residuals[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_horizontal_jacobian() {
        let constraint = HorizontalConstraint::new(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 5.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 2);

        // d/dy1 = -1, d/dy2 = 1
        assert!(jac.contains(&(0, 1, -1.0))); // p1.y at col 1
        assert!(jac.contains(&(0, 3, 1.0)));  // p2.y at col 3
    }

    #[test]
    fn test_variable_indices() {
        let constraint = HorizontalConstraint::new(3, 7);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![3, 7]);
    }
}
