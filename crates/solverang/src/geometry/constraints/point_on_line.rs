//! Point on line constraint: a point must lie on a line.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Point on line constraint: point P lies on line from A to B.
///
/// Uses the cross product formulation: `(P - A) x (B - A) = 0`
///
/// # 2D Equation
///
/// `(P.x - A.x)(B.y - A.y) - (P.y - A.y)(B.x - A.x) = 0`
///
/// # 3D
///
/// In 3D, the cross product yields a 3D vector, requiring 2 independent equations.
#[derive(Clone, Debug)]
pub struct PointOnLineConstraint<const D: usize> {
    /// Index of the point that must lie on the line.
    pub point: usize,
    /// Index of line start point.
    pub line_start: usize,
    /// Index of line end point.
    pub line_end: usize,
}

impl<const D: usize> PointOnLineConstraint<D> {
    /// Create a new point-on-line constraint.
    pub fn new(point: usize, line_start: usize, line_end: usize) -> Self {
        Self {
            point,
            line_start,
            line_end,
        }
    }
}

impl GeometricConstraint<2> for PointOnLineConstraint<2> {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let p = get_point(points, self.point);
        let a = get_point(points, self.line_start);
        let b = get_point(points, self.line_end);

        // Cross product in 2D: (P - A) x (B - A)
        let pax = p.get(0) - a.get(0);
        let pay = p.get(1) - a.get(1);
        let bax = b.get(0) - a.get(0);
        let bay = b.get(1) - a.get(1);

        let cross = pax * bay - pay * bax;
        vec![cross]
    }

    fn jacobian(&self, points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        let p = get_point(points, self.point);
        let a = get_point(points, self.line_start);
        let b = get_point(points, self.line_end);

        let bax = b.get(0) - a.get(0);
        let bay = b.get(1) - a.get(1);
        let pax = p.get(0) - a.get(0);
        let pay = p.get(1) - a.get(1);

        // f = (Px - Ax)(By - Ay) - (Py - Ay)(Bx - Ax)
        //
        // d/dPx = (By - Ay) = bay
        // d/dPy = -(Bx - Ax) = -bax
        // d/dAx = -(By - Ay) + (Py - Ay) = pay - bay
        // d/dAy = (Bx - Ax) - (Px - Ax) = bax - pax
        // d/dBx = -(Py - Ay) = -pay
        // d/dBy = (Px - Ax) = pax

        vec![
            (0, var_col::<2>(self.point, 0), bay),           // d/dPx
            (0, var_col::<2>(self.point, 1), -bax),          // d/dPy
            (0, var_col::<2>(self.line_start, 0), pay - bay), // d/dAx
            (0, var_col::<2>(self.line_start, 1), bax - pax), // d/dAy
            (0, var_col::<2>(self.line_end, 0), -pay),       // d/dBx
            (0, var_col::<2>(self.line_end, 1), pax),        // d/dBy
        ]
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point, self.line_start, self.line_end]
    }

    fn name(&self) -> &'static str {
        "PointOnLine"
    }
}

impl GeometricConstraint<3> for PointOnLineConstraint<3> {
    fn equation_count(&self) -> usize {
        // In 3D, cross product gives 3 components, but only 2 are independent
        2
    }

    fn residuals(&self, points: &[Point<3>]) -> Vec<f64> {
        let p = get_point(points, self.point);
        let a = get_point(points, self.line_start);
        let b = get_point(points, self.line_end);

        // Vectors
        let pa = [
            p.get(0) - a.get(0),
            p.get(1) - a.get(1),
            p.get(2) - a.get(2),
        ];
        let ba = [
            b.get(0) - a.get(0),
            b.get(1) - a.get(1),
            b.get(2) - a.get(2),
        ];

        // Cross product: pa x ba
        let cross_x = pa[1] * ba[2] - pa[2] * ba[1];
        let cross_y = pa[2] * ba[0] - pa[0] * ba[2];

        vec![cross_x, cross_y]
    }

    fn jacobian(&self, points: &[Point<3>]) -> Vec<(usize, usize, f64)> {
        let p = get_point(points, self.point);
        let a = get_point(points, self.line_start);
        let b = get_point(points, self.line_end);

        let pa = [
            p.get(0) - a.get(0),
            p.get(1) - a.get(1),
            p.get(2) - a.get(2),
        ];
        let ba = [
            b.get(0) - a.get(0),
            b.get(1) - a.get(1),
            b.get(2) - a.get(2),
        ];

        // f0 = pa[1] * ba[2] - pa[2] * ba[1] (cross_x)
        // f1 = pa[2] * ba[0] - pa[0] * ba[2] (cross_y)
        //
        // pa[i] = P[i] - A[i]
        // ba[i] = B[i] - A[i]

        let mut entries = Vec::with_capacity(18);

        // Equation 0: f0 = pa[1]*ba[2] - pa[2]*ba[1]
        // d/dPy = ba[2], d/dPz = -ba[1]
        // d/dAy = -ba[2] + pa[2] = pa[2] - ba[2], d/dAz = ba[1] - pa[1]
        // d/dBy = -pa[2], d/dBz = pa[1]
        entries.push((0, var_col::<3>(self.point, 1), ba[2]));
        entries.push((0, var_col::<3>(self.point, 2), -ba[1]));
        entries.push((0, var_col::<3>(self.line_start, 1), pa[2] - ba[2]));
        entries.push((0, var_col::<3>(self.line_start, 2), ba[1] - pa[1]));
        entries.push((0, var_col::<3>(self.line_end, 1), -pa[2]));
        entries.push((0, var_col::<3>(self.line_end, 2), pa[1]));

        // Equation 1: f1 = pa[2]*ba[0] - pa[0]*ba[2]
        // d/dPz = ba[0], d/dPx = -ba[2]
        // d/dAz = -ba[0] + pa[0] = pa[0] - ba[0], d/dAx = ba[2] - pa[2]
        // d/dBz = -pa[0], d/dBx = pa[2]
        entries.push((1, var_col::<3>(self.point, 2), ba[0]));
        entries.push((1, var_col::<3>(self.point, 0), -ba[2]));
        entries.push((1, var_col::<3>(self.line_start, 2), pa[0] - ba[0]));
        entries.push((1, var_col::<3>(self.line_start, 0), ba[2] - pa[2]));
        entries.push((1, var_col::<3>(self.line_end, 2), -pa[0]));
        entries.push((1, var_col::<3>(self.line_end, 0), pa[2]));

        entries
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.point, self.line_start, self.line_end]
    }

    fn name(&self) -> &'static str {
        "PointOnLine3D"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_point_on_line_2d_satisfied() {
        let constraint = PointOnLineConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(5.0, 5.0),   // Point on line
            Point2D::new(0.0, 0.0),   // Line start
            Point2D::new(10.0, 10.0), // Line end
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_point_on_line_2d_not_satisfied() {
        let constraint = PointOnLineConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(5.0, 0.0),   // Point NOT on line
            Point2D::new(0.0, 0.0),   // Line start (0,0)
            Point2D::new(10.0, 10.0), // Line end (10,10)
        ];

        let residuals = constraint.residuals(&points);
        // cross = (5-0)*(10-0) - (0-0)*(10-0) = 50 - 0 = 50
        assert!((residuals[0] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_on_line_2d_horizontal() {
        let constraint = PointOnLineConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(5.0, 3.0),   // Point on horizontal line y=3
            Point2D::new(0.0, 3.0),   // Line start
            Point2D::new(10.0, 3.0),  // Line end
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_point_on_line_3d_satisfied() {
        let constraint = PointOnLineConstraint::<3>::new(0, 1, 2);
        let points = vec![
            Point3D::new(5.0, 5.0, 5.0),   // Point on line
            Point3D::new(0.0, 0.0, 0.0),   // Line start
            Point3D::new(10.0, 10.0, 10.0), // Line end
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_point_on_line_3d_not_satisfied() {
        let constraint = PointOnLineConstraint::<3>::new(0, 1, 2);
        let points = vec![
            Point3D::new(5.0, 0.0, 0.0),   // Point NOT on line
            Point3D::new(0.0, 0.0, 0.0),   // Line start
            Point3D::new(10.0, 10.0, 10.0), // Line end
        ];

        let residuals = constraint.residuals(&points);
        // At least one residual should be non-zero
        let total = residuals[0].abs() + residuals[1].abs();
        assert!(total > 0.1);
    }

    #[test]
    fn test_point_on_line_jacobian_2d() {
        let constraint = PointOnLineConstraint::<2>::new(0, 1, 2);
        let points = vec![
            Point2D::new(5.0, 5.0),
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 10.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 6); // 1 equation, 3 points * 2 coords

        // All entries should be finite
        for (row, _col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_variable_indices() {
        let constraint = PointOnLineConstraint::<2>::new(5, 10, 15);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![5, 10, 15]);
    }
}
