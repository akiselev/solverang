//! Midpoint constraint: a point must be at the midpoint of a line segment.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Midpoint constraint: point M is the midpoint of line from A to B.
///
/// This constraint enforces that a point is exactly at the midpoint
/// of a line segment defined by two other points.
///
/// # Equations
///
/// For each dimension k: `2*M[k] - A[k] - B[k] = 0`
///
/// # Jacobian
///
/// - `d(residual_k)/d(M[k]) = 2`
/// - `d(residual_k)/d(A[k]) = -1`
/// - `d(residual_k)/d(B[k]) = -1`
#[derive(Clone, Debug)]
pub struct MidpointConstraint<const D: usize> {
    /// Index of the midpoint.
    pub midpoint: usize,
    /// Index of line start point.
    pub line_start: usize,
    /// Index of line end point.
    pub line_end: usize,
}

impl<const D: usize> MidpointConstraint<D> {
    /// Create a new midpoint constraint.
    pub fn new(midpoint: usize, line_start: usize, line_end: usize) -> Self {
        Self {
            midpoint,
            line_start,
            line_end,
        }
    }
}

impl<const D: usize> GeometricConstraint<D> for MidpointConstraint<D> {
    fn equation_count(&self) -> usize {
        D
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let m = get_point(points, self.midpoint);
        let a = get_point(points, self.line_start);
        let b = get_point(points, self.line_end);

        let mut residuals = Vec::with_capacity(D);
        for k in 0..D {
            residuals.push(2.0 * m.get(k) - a.get(k) - b.get(k));
        }
        residuals
    }

    fn jacobian(&self, _points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::with_capacity(D * 3);

        for k in 0..D {
            entries.push((k, var_col::<D>(self.midpoint, k), 2.0));    // d/dM[k]
            entries.push((k, var_col::<D>(self.line_start, k), -1.0)); // d/dA[k]
            entries.push((k, var_col::<D>(self.line_end, k), -1.0));   // d/dB[k]
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.midpoint, self.line_start, self.line_end]
    }

    fn name(&self) -> &'static str {
        "Midpoint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_midpoint_2d_satisfied() {
        let constraint = MidpointConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(5.0, 5.0),  // Midpoint
            Point2D::new(0.0, 0.0),  // Start
            Point2D::new(10.0, 10.0), // End
        ];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_2d_not_satisfied() {
        let constraint = MidpointConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(0.0, 0.0),   // Not at midpoint
            Point2D::new(0.0, 0.0),   // Start
            Point2D::new(10.0, 10.0), // End
        ];

        let residuals = constraint.residuals(&points);
        // Expected midpoint is (5, 5)
        // 2*0 - 0 - 10 = -10
        assert!((residuals[0] - (-10.0)).abs() < 1e-10);
        assert!((residuals[1] - (-10.0)).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_3d_satisfied() {
        let constraint = MidpointConstraint::<3>::new(0, 1, 2);
        let points = vec![
            Point3D::new(5.0, 5.0, 5.0),   // Midpoint
            Point3D::new(0.0, 0.0, 0.0),   // Start
            Point3D::new(10.0, 10.0, 10.0), // End
        ];

        let residuals = constraint.residuals(&points);
        assert_eq!(residuals.len(), 3);
        for r in &residuals {
            assert!(r.abs() < 1e-10);
        }
    }

    #[test]
    fn test_midpoint_jacobian() {
        let constraint = MidpointConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(5.0, 5.0),
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 10.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 6); // 2D * 3 points

        // Check expected entries
        // Equation 0 (x): d/dMx = 2, d/dAx = -1, d/dBx = -1
        // Equation 1 (y): d/dMy = 2, d/dAy = -1, d/dBy = -1
        let expected = vec![
            (0, 0, 2.0),   // d(eq0)/d(Mx)
            (0, 2, -1.0),  // d(eq0)/d(Ax)
            (0, 4, -1.0),  // d(eq0)/d(Bx)
            (1, 1, 2.0),   // d(eq1)/d(My)
            (1, 3, -1.0),  // d(eq1)/d(Ay)
            (1, 5, -1.0),  // d(eq1)/d(By)
        ];

        for exp in &expected {
            assert!(jac.contains(exp), "Missing entry {:?}", exp);
        }
    }

    #[test]
    fn test_variable_indices() {
        let constraint = MidpointConstraint::<2>::new(5, 10, 15);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![5, 10, 15]);
    }
}
