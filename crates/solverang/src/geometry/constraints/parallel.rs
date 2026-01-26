//! Parallel constraint: two lines must be parallel.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Parallel constraint: lines (p1->p2) and (p3->p4) are parallel.
///
/// Two lines are parallel when their direction vectors have zero cross product.
///
/// # 2D Equation
///
/// `(p2 - p1) x (p4 - p3) = 0`
/// Expanded: `(p2.x - p1.x)(p4.y - p3.y) - (p2.y - p1.y)(p4.x - p3.x) = 0`
///
/// # 3D
///
/// In 3D, the cross product yields a 3-component vector, so we would need 2 independent
/// equations. For simplicity, this implementation uses the 2D cross product formula
/// which works well when lines are in the same plane. For full 3D support, consider
/// using the magnitude of the cross product equals zero: `|v1 x v2| = 0`
#[derive(Clone, Debug)]
pub struct ParallelConstraint<const D: usize> {
    /// Start of first line.
    pub line1_start: usize,
    /// End of first line.
    pub line1_end: usize,
    /// Start of second line.
    pub line2_start: usize,
    /// End of second line.
    pub line2_end: usize,
}

impl<const D: usize> ParallelConstraint<D> {
    /// Create a new parallel constraint.
    pub fn new(line1_start: usize, line1_end: usize, line2_start: usize, line2_end: usize) -> Self {
        Self {
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        }
    }
}

impl GeometricConstraint<2> for ParallelConstraint<2> {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        // Direction vectors
        let dx1 = b.get(0) - a.get(0);
        let dy1 = b.get(1) - a.get(1);
        let dx2 = d.get(0) - c.get(0);
        let dy2 = d.get(1) - c.get(1);

        // 2D cross product
        let cross = dx1 * dy2 - dy1 * dx2;
        vec![cross]
    }

    fn jacobian(&self, points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        let dx1 = b.get(0) - a.get(0);
        let dy1 = b.get(1) - a.get(1);
        let dx2 = d.get(0) - c.get(0);
        let dy2 = d.get(1) - c.get(1);

        // f = dx1 * dy2 - dy1 * dx2
        // where dx1 = Bx - Ax, dy1 = By - Ay, dx2 = Dx - Cx, dy2 = Dy - Cy
        //
        // d/dAx = -dy2
        // d/dAy = dx2
        // d/dBx = dy2
        // d/dBy = -dx2
        // d/dCx = dy1
        // d/dCy = -dx1
        // d/dDx = -dy1
        // d/dDy = dx1

        vec![
            (0, var_col::<2>(self.line1_start, 0), -dy2),  // d/dAx
            (0, var_col::<2>(self.line1_start, 1), dx2),   // d/dAy
            (0, var_col::<2>(self.line1_end, 0), dy2),     // d/dBx
            (0, var_col::<2>(self.line1_end, 1), -dx2),    // d/dBy
            (0, var_col::<2>(self.line2_start, 0), dy1),   // d/dCx
            (0, var_col::<2>(self.line2_start, 1), -dx1),  // d/dCy
            (0, var_col::<2>(self.line2_end, 0), -dy1),    // d/dDx
            (0, var_col::<2>(self.line2_end, 1), dx1),     // d/dDy
        ]
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
        "Parallel"
    }
}

impl GeometricConstraint<3> for ParallelConstraint<3> {
    fn equation_count(&self) -> usize {
        // In 3D, we need the cross product to be zero, which gives 3 equations
        // but only 2 are independent. We use 2 equations: x and z components of cross product.
        // Using x and z (rather than x and y) ensures we catch perpendicular lines in the XY plane.
        2
    }

    fn residuals(&self, points: &[Point<3>]) -> Vec<f64> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        // Direction vectors
        let v1 = [
            b.get(0) - a.get(0),
            b.get(1) - a.get(1),
            b.get(2) - a.get(2),
        ];
        let v2 = [
            d.get(0) - c.get(0),
            d.get(1) - c.get(1),
            d.get(2) - c.get(2),
        ];

        // Cross product components: v1 x v2 = 0
        // (v1y * v2z - v1z * v2y, v1z * v2x - v1x * v2z, v1x * v2y - v1y * v2x)
        let cross_x = v1[1] * v2[2] - v1[2] * v2[1];
        let cross_z = v1[0] * v2[1] - v1[1] * v2[0];

        vec![cross_x, cross_z]
    }

    fn jacobian(&self, points: &[Point<3>]) -> Vec<(usize, usize, f64)> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        let v1 = [
            b.get(0) - a.get(0),
            b.get(1) - a.get(1),
            b.get(2) - a.get(2),
        ];
        let v2 = [
            d.get(0) - c.get(0),
            d.get(1) - c.get(1),
            d.get(2) - c.get(2),
        ];

        // f0 = v1y * v2z - v1z * v2y (cross_x)
        // f1 = v1x * v2y - v1y * v2x (cross_z)
        //
        // v1x = Bx - Ax, v1y = By - Ay, v1z = Bz - Az
        // v2x = Dx - Cx, v2y = Dy - Cy, v2z = Dz - Cz

        let mut entries = Vec::with_capacity(24);

        // Equation 0: f0 = v1y * v2z - v1z * v2y
        // d/dAy = -v2z, d/dAz = v2y
        // d/dBy = v2z,  d/dBz = -v2y
        // d/dCy = v1z,  d/dCz = -v1y
        // d/dDy = -v1z, d/dDz = v1y
        entries.push((0, var_col::<3>(self.line1_start, 1), -v2[2]));
        entries.push((0, var_col::<3>(self.line1_start, 2), v2[1]));
        entries.push((0, var_col::<3>(self.line1_end, 1), v2[2]));
        entries.push((0, var_col::<3>(self.line1_end, 2), -v2[1]));
        entries.push((0, var_col::<3>(self.line2_start, 1), v1[2]));
        entries.push((0, var_col::<3>(self.line2_start, 2), -v1[1]));
        entries.push((0, var_col::<3>(self.line2_end, 1), -v1[2]));
        entries.push((0, var_col::<3>(self.line2_end, 2), v1[1]));

        // Equation 1: f1 = v1x * v2y - v1y * v2x
        // d/dAx = -v2y, d/dAy = v2x
        // d/dBx = v2y,  d/dBy = -v2x
        // d/dCx = v1y,  d/dCy = -v1x
        // d/dDx = -v1y, d/dDy = v1x
        entries.push((1, var_col::<3>(self.line1_start, 0), -v2[1]));
        entries.push((1, var_col::<3>(self.line1_start, 1), v2[0]));
        entries.push((1, var_col::<3>(self.line1_end, 0), v2[1]));
        entries.push((1, var_col::<3>(self.line1_end, 1), -v2[0]));
        entries.push((1, var_col::<3>(self.line2_start, 0), v1[1]));
        entries.push((1, var_col::<3>(self.line2_start, 1), -v1[0]));
        entries.push((1, var_col::<3>(self.line2_end, 0), -v1[1]));
        entries.push((1, var_col::<3>(self.line2_end, 1), v1[0]));

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
        "Parallel3D"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_parallel_2d_satisfied() {
        let constraint = ParallelConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(5.0, 0.0),
            Point2D::new(6.0, 1.0), // Same direction as first line
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_parallel_2d_not_satisfied() {
        let constraint = ParallelConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0), // Horizontal
            Point2D::new(5.0, 0.0),
            Point2D::new(5.0, 1.0), // Vertical - not parallel!
        ];

        let residuals = constraint.residuals(&points);
        // cross = 1*1 - 0*0 = 1
        assert!((residuals[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_2d_jacobian() {
        let constraint = ParallelConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(5.0, 0.0),
            Point2D::new(7.0, 1.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 8);

        // All entries should be present and finite
        for (row, _col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_parallel_3d_satisfied() {
        let constraint = ParallelConstraint::<3>::new(0, 1, 2, 3);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 2.0, 3.0),
            Point3D::new(5.0, 5.0, 5.0),
            Point3D::new(6.0, 7.0, 8.0), // Same direction
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_parallel_3d_not_satisfied() {
        let constraint = ParallelConstraint::<3>::new(0, 1, 2, 3);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0), // Along x
            Point3D::new(5.0, 5.0, 5.0),
            Point3D::new(5.0, 6.0, 5.0), // Along y - not parallel!
        ];

        let residuals = constraint.residuals(&points);
        // At least one residual should be non-zero
        let total = residuals[0].abs() + residuals[1].abs();
        assert!(total > 0.1);
    }

    #[test]
    fn test_variable_indices() {
        let constraint = ParallelConstraint::<2>::new(0, 1, 2, 3);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }
}
