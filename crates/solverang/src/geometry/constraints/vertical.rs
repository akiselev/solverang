//! Vertical constraint: two points must have the same x-coordinate (2D only).

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Vertical constraint: p1.x = p2.x.
///
/// This constraint enforces that two points have the same x-coordinate,
/// making the line between them vertical.
///
/// This is a 2D-only constraint.
///
/// # Equation
///
/// `p2.x - p1.x = 0`
///
/// # Jacobian
///
/// - `d(residual)/d(p1.x) = -1`
/// - `d(residual)/d(p2.x) = 1`
#[derive(Clone, Debug)]
pub struct VerticalConstraint {
    /// Index of first point.
    pub point1: usize,
    /// Index of second point.
    pub point2: usize,
}

impl VerticalConstraint {
    /// Create a new vertical constraint.
    pub fn new(point1: usize, point2: usize) -> Self {
        Self { point1, point2 }
    }
}

impl GeometricConstraint<2> for VerticalConstraint {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let p1 = get_point(points, self.point1);
        let p2 = get_point(points, self.point2);

        vec![p2.get(0) - p1.get(0)] // x2 - x1
    }

    fn jacobian(&self, _points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, var_col::<2>(self.point1, 0), -1.0), // d/dx1
            (0, var_col::<2>(self.point2, 0), 1.0),  // d/dx2
        ]
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }

    fn name(&self) -> &'static str {
        "Vertical"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point2D;

    #[test]
    fn test_vertical_satisfied() {
        let constraint = VerticalConstraint::new(0, 1);
        let points = vec![
            Point2D::new(5.0, 0.0),
            Point2D::new(5.0, 10.0),
        ];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_vertical_not_satisfied() {
        let constraint = VerticalConstraint::new(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(5.0, 10.0),
        ];

        let residuals = constraint.residuals(&points);
        assert!((residuals[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vertical_jacobian() {
        let constraint = VerticalConstraint::new(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(5.0, 10.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 2);

        // d/dx1 = -1, d/dx2 = 1
        assert!(jac.contains(&(0, 0, -1.0))); // p1.x at col 0
        assert!(jac.contains(&(0, 2, 1.0)));  // p2.x at col 2
    }

    #[test]
    fn test_variable_indices() {
        let constraint = VerticalConstraint::new(3, 7);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![3, 7]);
    }
}
