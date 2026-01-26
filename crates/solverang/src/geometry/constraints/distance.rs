//! Distance constraint: enforce a specific distance between two points.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Distance constraint: |p2 - p1| = target.
///
/// This constraint enforces that the Euclidean distance between two points
/// equals a specified target value.
///
/// # Equation
///
/// `sqrt((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + ...) - target = 0`
///
/// # Jacobian
///
/// For coordinate `k` of point `i`:
/// - `d(residual)/d(p1[k]) = -(p2[k] - p1[k]) / distance`
/// - `d(residual)/d(p2[k]) = (p2[k] - p1[k]) / distance`
#[derive(Clone, Debug)]
pub struct DistanceConstraint<const D: usize> {
    /// Index of first point.
    pub point1: usize,
    /// Index of second point.
    pub point2: usize,
    /// Target distance.
    pub target: f64,
    /// Optional weight (default 1.0).
    weight: f64,
}

impl<const D: usize> DistanceConstraint<D> {
    /// Create a new distance constraint.
    pub fn new(point1: usize, point2: usize, target: f64) -> Self {
        Self {
            point1,
            point2,
            target,
            weight: 1.0,
        }
    }

    /// Create a distance constraint with custom weight.
    pub fn with_weight(point1: usize, point2: usize, target: f64, weight: f64) -> Self {
        Self {
            point1,
            point2,
            target,
            weight,
        }
    }
}

impl<const D: usize> GeometricConstraint<D> for DistanceConstraint<D> {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let p1 = get_point(points, self.point1);
        let p2 = get_point(points, self.point2);

        let actual = p1.distance_to(&p2);
        vec![actual - self.target]
    }

    fn jacobian(&self, points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let p1 = get_point(points, self.point1);
        let p2 = get_point(points, self.point2);

        let dist = p1.safe_distance_to(&p2);

        let mut entries = Vec::with_capacity(D * 2);

        for k in 0..D {
            let diff = p2.get(k) - p1.get(k);
            let grad = diff / dist;

            entries.push((0, var_col::<D>(self.point1, k), -grad));
            entries.push((0, var_col::<D>(self.point2, k), grad));
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point1, self.point2]
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn name(&self) -> &'static str {
        "Distance"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_distance_2d_residual() {
        let constraint = DistanceConstraint::<2>::new(0, 1, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 4.0),
        ];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 1);
        // Distance is 5, target is 5, residual should be 0
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_distance_2d_residual_nonzero() {
        let constraint = DistanceConstraint::<2>::new(0, 1, 10.0);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 4.0),
        ];

        let residuals = constraint.residuals(&points);
        // Distance is 5, target is 10, residual should be -5
        assert!((residuals[0] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_distance_3d_residual() {
        let constraint = DistanceConstraint::<3>::new(0, 1, 7.0);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(2.0, 3.0, 6.0), // distance = sqrt(4+9+36) = 7
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_distance_jacobian() {
        let constraint = DistanceConstraint::<2>::new(0, 1, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 4.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 4); // 2 points * 2 coordinates

        // Verify derivatives: grad = (p2 - p1) / dist
        // dx/dist = 3/5 = 0.6, dy/dist = 4/5 = 0.8
        // d/dp1.x = -0.6, d/dp1.y = -0.8
        // d/dp2.x = 0.6, d/dp2.y = 0.8

        let mut found = [false; 4];
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            match *col {
                0 => { assert!((*val - (-0.6)).abs() < 1e-10); found[0] = true; }
                1 => { assert!((*val - (-0.8)).abs() < 1e-10); found[1] = true; }
                2 => { assert!((*val - 0.6).abs() < 1e-10); found[2] = true; }
                3 => { assert!((*val - 0.8).abs() < 1e-10); found[3] = true; }
                _ => panic!("unexpected column"),
            }
        }
        assert!(found.iter().all(|&f| f));
    }

    #[test]
    fn test_distance_coincident_points() {
        // When points are coincident, the distance is zero
        // Jacobian should use MIN_EPSILON to avoid division by zero
        let constraint = DistanceConstraint::<2>::new(0, 1, 1.0);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(0.0, 0.0),
        ];

        let jac = constraint.jacobian(&points);
        // Should not panic and should produce finite values
        for (_, _, val) in jac {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_variable_indices() {
        let constraint = DistanceConstraint::<2>::new(5, 10, 1.0);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![5, 10]);
    }
}
