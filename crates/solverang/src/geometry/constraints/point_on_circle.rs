//! Point on circle constraint: a point must lie on a circle.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Point on circle constraint: point P lies on circle with center C and radius r.
///
/// This constraint enforces that a point lies on the perimeter of a circle.
/// Works in any dimension (circle in 2D, sphere surface in 3D).
///
/// # Equation
///
/// `|P - C| - radius = 0`
///
/// Expanded: `sqrt(sum_k (P[k] - C[k])^2) - radius = 0`
///
/// # Jacobian
///
/// For coordinate k:
/// - `d/d(P[k]) = (P[k] - C[k]) / distance`
/// - `d/d(C[k]) = -(P[k] - C[k]) / distance`
#[derive(Clone, Debug)]
pub struct PointOnCircleConstraint<const D: usize> {
    /// Index of the point that must lie on the circle.
    pub point: usize,
    /// Index of the circle center.
    pub center: usize,
    /// Radius of the circle.
    pub radius: f64,
}

impl<const D: usize> PointOnCircleConstraint<D> {
    /// Create a new point-on-circle constraint.
    pub fn new(point: usize, center: usize, radius: f64) -> Self {
        Self {
            point,
            center,
            radius,
        }
    }
}

impl<const D: usize> GeometricConstraint<D> for PointOnCircleConstraint<D> {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let p = get_point(points, self.point);
        let c = get_point(points, self.center);

        let dist = p.distance_to(&c);
        vec![dist - self.radius]
    }

    fn jacobian(&self, points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let p = get_point(points, self.point);
        let c = get_point(points, self.center);

        let dist = p.safe_distance_to(&c);

        let mut entries = Vec::with_capacity(D * 2);

        for k in 0..D {
            let diff = p.get(k) - c.get(k);
            let grad = diff / dist;

            entries.push((0, var_col::<D>(self.point, k), grad));    // d/dP[k]
            entries.push((0, var_col::<D>(self.center, k), -grad));  // d/dC[k]
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point, self.center]
    }

    fn name(&self) -> &'static str {
        "PointOnCircle"
    }
}

/// Point on circle constraint with variable radius (radius as a point index).
///
/// This is useful when the radius itself is a variable, e.g., when
/// the radius is defined by another point.
#[derive(Clone, Debug)]
pub struct PointOnCircleVariableRadiusConstraint<const D: usize> {
    /// Index of the point that must lie on the circle.
    pub point: usize,
    /// Index of the circle center.
    pub center: usize,
    /// Index of a point that defines the radius (distance from center to this point).
    pub radius_point: usize,
}

impl<const D: usize> PointOnCircleVariableRadiusConstraint<D> {
    /// Create a new point-on-circle constraint with variable radius.
    pub fn new(point: usize, center: usize, radius_point: usize) -> Self {
        Self {
            point,
            center,
            radius_point,
        }
    }
}

impl<const D: usize> GeometricConstraint<D> for PointOnCircleVariableRadiusConstraint<D> {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let p = get_point(points, self.point);
        let c = get_point(points, self.center);
        let r = get_point(points, self.radius_point);

        let dist_p = p.distance_to(&c);
        let dist_r = r.distance_to(&c);

        vec![dist_p - dist_r]
    }

    fn jacobian(&self, points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let p = get_point(points, self.point);
        let c = get_point(points, self.center);
        let r = get_point(points, self.radius_point);

        let dist_p = p.safe_distance_to(&c);
        let dist_r = r.safe_distance_to(&c);

        let mut entries = Vec::with_capacity(D * 3);

        for k in 0..D {
            let diff_p = p.get(k) - c.get(k);
            let diff_r = r.get(k) - c.get(k);

            let grad_p = diff_p / dist_p;
            let grad_r = diff_r / dist_r;

            // f = dist_p - dist_r
            // d/dP[k] = diff_p / dist_p
            // d/dR[k] = -diff_r / dist_r
            // d/dC[k] = -diff_p / dist_p + diff_r / dist_r
            entries.push((0, var_col::<D>(self.point, k), grad_p));
            entries.push((0, var_col::<D>(self.radius_point, k), -grad_r));
            entries.push((0, var_col::<D>(self.center, k), -grad_p + grad_r));
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point, self.center, self.radius_point]
    }

    fn name(&self) -> &'static str {
        "PointOnCircleVariableRadius"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_point_on_circle_2d_satisfied() {
        let constraint = PointOnCircleConstraint::<2>::new(0, 1, 5.0);
        let points = vec![
            Point2D::new(5.0, 0.0),  // Point on circle
            Point2D::new(0.0, 0.0),  // Center
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_point_on_circle_2d_satisfied_diagonal() {
        let constraint = PointOnCircleConstraint::<2>::new(0, 1, 5.0);
        let points = vec![
            Point2D::new(3.0, 4.0),  // Point on circle (3^2 + 4^2 = 25 = 5^2)
            Point2D::new(0.0, 0.0),  // Center
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_point_on_circle_2d_not_satisfied() {
        let constraint = PointOnCircleConstraint::<2>::new(0, 1, 5.0);
        let points = vec![
            Point2D::new(10.0, 0.0),  // Point outside circle
            Point2D::new(0.0, 0.0),   // Center
        ];

        let residuals = constraint.residuals(&points);
        // Distance is 10, radius is 5, residual = 10 - 5 = 5
        assert!((residuals[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_on_circle_3d_satisfied() {
        let constraint = PointOnCircleConstraint::<3>::new(0, 1, 5.0);
        let points = vec![
            Point3D::new(3.0, 4.0, 0.0),  // Point on sphere
            Point3D::new(0.0, 0.0, 0.0),  // Center
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_point_on_circle_jacobian() {
        let constraint = PointOnCircleConstraint::<2>::new(0, 1, 5.0);
        let points = vec![
            Point2D::new(3.0, 4.0),  // dist = 5
            Point2D::new(0.0, 0.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 4); // 2 points * 2 coords

        // d/dPx = 3/5 = 0.6, d/dPy = 4/5 = 0.8
        // d/dCx = -0.6, d/dCy = -0.8
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            let expected = match *col {
                0 => 0.6,   // d/dPx
                1 => 0.8,   // d/dPy
                2 => -0.6,  // d/dCx
                3 => -0.8,  // d/dCy
                _ => panic!("unexpected column"),
            };
            assert!((*val - expected).abs() < 1e-10, "col {}: expected {}, got {}", col, expected, val);
        }
    }

    #[test]
    fn test_point_on_circle_coincident_points() {
        let constraint = PointOnCircleConstraint::<2>::new(0, 1, 5.0);
        let points = vec![
            Point2D::new(0.0, 0.0),  // Point at center
            Point2D::new(0.0, 0.0),  // Center
        ];

        let jac = constraint.jacobian(&points);
        // Should not panic, values should be finite
        for (_, _, val) in &jac {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_variable_radius_constraint() {
        let constraint = PointOnCircleVariableRadiusConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(3.0, 4.0),  // Point on circle (dist = 5)
            Point2D::new(0.0, 0.0),  // Center
            Point2D::new(5.0, 0.0),  // Radius point (dist = 5)
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_variable_indices() {
        let constraint = PointOnCircleConstraint::<2>::new(5, 10, 5.0);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![5, 10]);
    }
}
