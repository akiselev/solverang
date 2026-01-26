//! Coincident constraint: two points must be at the same location.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Coincident constraint: p1 = p2.
///
/// This constraint enforces that two points occupy the same location.
/// It generates D equations (one per dimension).
///
/// # Equations
///
/// For each dimension k: `p2[k] - p1[k] = 0`
///
/// # Jacobian
///
/// - `d(residual_k)/d(p1[k]) = -1`
/// - `d(residual_k)/d(p2[k]) = 1`
#[derive(Clone, Debug)]
pub struct CoincidentConstraint<const D: usize> {
    /// Index of first point.
    pub point1: usize,
    /// Index of second point.
    pub point2: usize,
}

impl<const D: usize> CoincidentConstraint<D> {
    /// Create a new coincident constraint.
    pub fn new(point1: usize, point2: usize) -> Self {
        Self { point1, point2 }
    }
}

impl<const D: usize> GeometricConstraint<D> for CoincidentConstraint<D> {
    fn equation_count(&self) -> usize {
        D
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let p1 = get_point(points, self.point1);
        let p2 = get_point(points, self.point2);

        let mut residuals = Vec::with_capacity(D);
        for k in 0..D {
            residuals.push(p2.get(k) - p1.get(k));
        }
        residuals
    }

    fn jacobian(&self, _points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::with_capacity(D * 2);

        for k in 0..D {
            entries.push((k, var_col::<D>(self.point1, k), -1.0));
            entries.push((k, var_col::<D>(self.point2, k), 1.0));
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }

    fn name(&self) -> &'static str {
        "Coincident"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_coincident_2d_satisfied() {
        let constraint = CoincidentConstraint::<2>::new(0, 1);
        let points = vec![
            Point2D::new(5.0, 3.0),
            Point2D::new(5.0, 3.0),
        ];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_coincident_2d_not_satisfied() {
        let constraint = CoincidentConstraint::<2>::new(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 4.0),
        ];

        let residuals = constraint.residuals(&points);
        assert!((residuals[0] - 3.0).abs() < 1e-10);
        assert!((residuals[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_coincident_3d() {
        let constraint = CoincidentConstraint::<3>::new(0, 1);
        let points = vec![
            Point3D::new(1.0, 2.0, 3.0),
            Point3D::new(4.0, 5.0, 6.0),
        ];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 3);
        assert!((residuals[0] - 3.0).abs() < 1e-10);
        assert!((residuals[1] - 3.0).abs() < 1e-10);
        assert!((residuals[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_coincident_jacobian() {
        let constraint = CoincidentConstraint::<2>::new(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 4); // 2 equations * 2 entries each

        // Check entries
        // Equation 0 (x): d/dp1.x = -1, d/dp2.x = 1
        // Equation 1 (y): d/dp1.y = -1, d/dp2.y = 1
        let expected = vec![
            (0, 0, -1.0), // d(eq0)/d(p1.x)
            (0, 2, 1.0),  // d(eq0)/d(p2.x)
            (1, 1, -1.0), // d(eq1)/d(p1.y)
            (1, 3, 1.0),  // d(eq1)/d(p2.y)
        ];

        for exp in &expected {
            assert!(jac.contains(exp), "Missing entry {:?}", exp);
        }
    }

    #[test]
    fn test_variable_indices() {
        let constraint = CoincidentConstraint::<2>::new(3, 7);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![3, 7]);
    }
}
