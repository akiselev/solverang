//! Fixed position constraint: a point must be at a specific location.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Fixed position constraint: p = target.
///
/// This constraint enforces that a point is at a specific fixed location.
/// It generates D equations (one per dimension).
///
/// # Equations
///
/// For each dimension k: `p[k] - target[k] = 0`
///
/// # Jacobian
///
/// - `d(residual_k)/d(p[k]) = 1`
#[derive(Clone, Debug)]
pub struct FixedConstraint<const D: usize> {
    /// Index of the point to fix.
    pub point: usize,
    /// Target position.
    pub target: Point<D>,
}

impl<const D: usize> FixedConstraint<D> {
    /// Create a new fixed position constraint.
    pub fn new(point: usize, target: Point<D>) -> Self {
        Self { point, target }
    }
}

impl FixedConstraint<2> {
    /// Create a 2D fixed constraint from coordinates.
    pub fn from_coords(point: usize, x: f64, y: f64) -> Self {
        Self::new(point, Point::from_coords([x, y]))
    }
}

impl FixedConstraint<3> {
    /// Create a 3D fixed constraint from coordinates.
    pub fn from_coords(point: usize, x: f64, y: f64, z: f64) -> Self {
        Self::new(point, Point::from_coords([x, y, z]))
    }
}

impl<const D: usize> GeometricConstraint<D> for FixedConstraint<D> {
    fn equation_count(&self) -> usize {
        D
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let p = get_point(points, self.point);

        let mut residuals = Vec::with_capacity(D);
        for k in 0..D {
            residuals.push(p.get(k) - self.target.get(k));
        }
        residuals
    }

    fn jacobian(&self, _points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::with_capacity(D);

        for k in 0..D {
            entries.push((k, var_col::<D>(self.point, k), 1.0));
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point]
    }

    fn name(&self) -> &'static str {
        "Fixed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_fixed_2d_satisfied() {
        let constraint = FixedConstraint::<2>::from_coords(0, 5.0, 3.0);
        let points = vec![Point2D::new(5.0, 3.0)];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_fixed_2d_not_satisfied() {
        let constraint = FixedConstraint::<2>::from_coords(0, 10.0, 20.0);
        let points = vec![Point2D::new(0.0, 0.0)];

        let residuals = constraint.residuals(&points);
        assert!((residuals[0] - (-10.0)).abs() < 1e-10);
        assert!((residuals[1] - (-20.0)).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_3d() {
        let constraint = FixedConstraint::<3>::from_coords(0, 1.0, 2.0, 3.0);
        let points = vec![Point3D::new(4.0, 5.0, 6.0)];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 3);
        assert!((residuals[0] - 3.0).abs() < 1e-10);
        assert!((residuals[1] - 3.0).abs() < 1e-10);
        assert!((residuals[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_jacobian() {
        let constraint = FixedConstraint::<2>::from_coords(0, 5.0, 3.0);
        let points = vec![Point2D::new(0.0, 0.0)];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 2);

        // d(eq0)/d(p.x) = 1, d(eq1)/d(p.y) = 1
        assert!(jac.contains(&(0, 0, 1.0)));
        assert!(jac.contains(&(1, 1, 1.0)));
    }

    #[test]
    fn test_fixed_with_multiple_points() {
        // Fix point 2 out of 3 points
        let constraint = FixedConstraint::<2>::from_coords(2, 10.0, 20.0);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(10.0, 20.0),
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);

        let jac = constraint.jacobian(&points);
        // Column indices should be 4 and 5 (point 2's x and y)
        assert!(jac.contains(&(0, 4, 1.0)));
        assert!(jac.contains(&(1, 5, 1.0)));
    }

    #[test]
    fn test_variable_indices() {
        let constraint = FixedConstraint::<2>::from_coords(5, 0.0, 0.0);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![5]);
    }
}
