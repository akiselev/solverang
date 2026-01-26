//! Perpendicular constraint: two lines must be perpendicular.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Perpendicular constraint: lines (p1->p2) and (p3->p4) are perpendicular.
///
/// Two lines are perpendicular when their direction vectors have zero dot product.
///
/// # Equation (2D and 3D)
///
/// `(p2 - p1) . (p4 - p3) = 0`
/// Expanded: `(p2.x - p1.x)(p4.x - p3.x) + (p2.y - p1.y)(p4.y - p3.y) + ... = 0`
#[derive(Clone, Debug)]
pub struct PerpendicularConstraint<const D: usize> {
    /// Start of first line.
    pub line1_start: usize,
    /// End of first line.
    pub line1_end: usize,
    /// Start of second line.
    pub line2_start: usize,
    /// End of second line.
    pub line2_end: usize,
}

impl<const D: usize> PerpendicularConstraint<D> {
    /// Create a new perpendicular constraint.
    pub fn new(line1_start: usize, line1_end: usize, line2_start: usize, line2_end: usize) -> Self {
        Self {
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        }
    }
}

impl<const D: usize> GeometricConstraint<D> for PerpendicularConstraint<D> {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        // Dot product of direction vectors
        let mut dot = 0.0;
        for k in 0..D {
            let v1k = b.get(k) - a.get(k);
            let v2k = d.get(k) - c.get(k);
            dot += v1k * v2k;
        }

        vec![dot]
    }

    fn jacobian(&self, points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        // f = sum_k (v1k * v2k) where v1k = Bk - Ak, v2k = Dk - Ck
        //
        // d/dAk = -v2k
        // d/dBk = v2k
        // d/dCk = -v1k
        // d/dDk = v1k

        let mut entries = Vec::with_capacity(D * 4);

        for k in 0..D {
            let v1k = b.get(k) - a.get(k);
            let v2k = d.get(k) - c.get(k);

            entries.push((0, var_col::<D>(self.line1_start, k), -v2k));  // d/dAk
            entries.push((0, var_col::<D>(self.line1_end, k), v2k));     // d/dBk
            entries.push((0, var_col::<D>(self.line2_start, k), -v1k));  // d/dCk
            entries.push((0, var_col::<D>(self.line2_end, k), v1k));     // d/dDk
        }

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![
            self.line1_start,
            self.line1_end,
            self.line2_start,
            self.line2_end,
        ]
    }

    fn name(&self) -> &'static str {
        "Perpendicular"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_perpendicular_2d_satisfied() {
        let constraint = PerpendicularConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0), // Horizontal
            Point2D::new(5.0, 0.0),
            Point2D::new(5.0, 1.0), // Vertical
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_perpendicular_2d_not_satisfied() {
        let constraint = PerpendicularConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0), // Horizontal
            Point2D::new(5.0, 0.0),
            Point2D::new(6.0, 0.0), // Also horizontal - not perpendicular!
        ];

        let residuals = constraint.residuals(&points);
        // dot = 1*1 + 0*0 = 1
        assert!((residuals[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perpendicular_2d_diagonal() {
        let constraint = PerpendicularConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0), // 45 degrees
            Point2D::new(5.0, 5.0),
            Point2D::new(6.0, 4.0), // -45 degrees (perpendicular)
        ];

        let residuals = constraint.residuals(&points);
        // dot = 1*1 + 1*(-1) = 0
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_perpendicular_3d_satisfied() {
        let constraint = PerpendicularConstraint::<3>::new(0, 1, 2, 3);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0), // Along x
            Point3D::new(5.0, 5.0, 5.0),
            Point3D::new(5.0, 6.0, 5.0), // Along y - perpendicular to x
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_perpendicular_3d_not_satisfied() {
        let constraint = PerpendicularConstraint::<3>::new(0, 1, 2, 3);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0), // Along x
            Point3D::new(5.0, 5.0, 5.0),
            Point3D::new(6.0, 5.0, 5.0), // Also along x - not perpendicular!
        ];

        let residuals = constraint.residuals(&points);
        // dot = 1*1 + 0*0 + 0*0 = 1
        assert!((residuals[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perpendicular_jacobian() {
        let constraint = PerpendicularConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 3.0),
            Point2D::new(5.0, 5.0),
            Point2D::new(8.0, 9.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 8); // 2D * 4 points

        // All entries should be finite
        for (row, _col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_variable_indices() {
        let constraint = PerpendicularConstraint::<2>::new(0, 1, 2, 3);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }
}
