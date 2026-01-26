//! Collinear constraint: two line segments lie on the same infinite line.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Collinear constraint: two line segments lie on the same infinite line.
///
/// Both endpoints of line 2 must lie on the line defined by line 1.
/// Uses cross product formulation for each endpoint.
///
/// # 2D Equations
///
/// 1. `(B - A) x (C - A) = 0` (C on line AB)
/// 2. `(B - A) x (D - A) = 0` (D on line AB)
///
/// # 3D
///
/// In 3D, each cross product constraint yields 2 independent equations,
/// giving 4 equations total.
#[derive(Clone, Debug)]
pub struct CollinearConstraint<const D: usize> {
    /// Start of first line.
    pub line1_start: usize,
    /// End of first line.
    pub line1_end: usize,
    /// Start of second line.
    pub line2_start: usize,
    /// End of second line.
    pub line2_end: usize,
}

impl<const D: usize> CollinearConstraint<D> {
    /// Create a new collinear constraint.
    pub fn new(line1_start: usize, line1_end: usize, line2_start: usize, line2_end: usize) -> Self {
        Self {
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        }
    }
}

impl GeometricConstraint<2> for CollinearConstraint<2> {
    fn equation_count(&self) -> usize {
        2 // One for each endpoint of line2
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        // Cross products: (B - A) x (C - A) and (B - A) x (D - A)
        let bax = b.get(0) - a.get(0);
        let bay = b.get(1) - a.get(1);

        let cax = c.get(0) - a.get(0);
        let cay = c.get(1) - a.get(1);

        let dax = d.get(0) - a.get(0);
        let day = d.get(1) - a.get(1);

        let cross1 = bax * cay - bay * cax;
        let cross2 = bax * day - bay * dax;

        vec![cross1, cross2]
    }

    fn jacobian(&self, points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        let bax = b.get(0) - a.get(0);
        let bay = b.get(1) - a.get(1);

        let cax = c.get(0) - a.get(0);
        let cay = c.get(1) - a.get(1);

        let dax = d.get(0) - a.get(0);
        let day = d.get(1) - a.get(1);

        // Equation 1: f1 = (Bx - Ax)(Cy - Ay) - (By - Ay)(Cx - Ax)
        // d/dAx = -(Cy - Ay) + (By - Ay) = bay - cay
        // d/dAy = (Cx - Ax) - (Bx - Ax) = cax - bax
        // d/dBx = (Cy - Ay) = cay
        // d/dBy = -(Cx - Ax) = -cax
        // d/dCx = -(By - Ay) = -bay
        // d/dCy = (Bx - Ax) = bax

        // Equation 2: f2 = (Bx - Ax)(Dy - Ay) - (By - Ay)(Dx - Ax)
        // Same pattern with D instead of C

        vec![
            // Equation 1: C on line AB
            (0, var_col::<2>(self.line1_start, 0), bay - cay),
            (0, var_col::<2>(self.line1_start, 1), cax - bax),
            (0, var_col::<2>(self.line1_end, 0), cay),
            (0, var_col::<2>(self.line1_end, 1), -cax),
            (0, var_col::<2>(self.line2_start, 0), -bay),
            (0, var_col::<2>(self.line2_start, 1), bax),
            // Equation 2: D on line AB
            (1, var_col::<2>(self.line1_start, 0), bay - day),
            (1, var_col::<2>(self.line1_start, 1), dax - bax),
            (1, var_col::<2>(self.line1_end, 0), day),
            (1, var_col::<2>(self.line1_end, 1), -dax),
            (1, var_col::<2>(self.line2_end, 0), -bay),
            (1, var_col::<2>(self.line2_end, 1), bax),
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
        "Collinear"
    }
}

impl GeometricConstraint<3> for CollinearConstraint<3> {
    fn equation_count(&self) -> usize {
        // Each point-on-line constraint in 3D needs 2 equations,
        // and we have 2 points (C and D), so 4 total
        4
    }

    fn residuals(&self, points: &[Point<3>]) -> Vec<f64> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        // Direction of line 1
        let ba = [
            b.get(0) - a.get(0),
            b.get(1) - a.get(1),
            b.get(2) - a.get(2),
        ];

        // Vector from A to C
        let ca = [
            c.get(0) - a.get(0),
            c.get(1) - a.get(1),
            c.get(2) - a.get(2),
        ];

        // Vector from A to D
        let da = [
            d.get(0) - a.get(0),
            d.get(1) - a.get(1),
            d.get(2) - a.get(2),
        ];

        // Cross products: ca x ba and da x ba
        // We use x and y components (2 independent equations each)
        let cross_c_x = ca[1] * ba[2] - ca[2] * ba[1];
        let cross_c_y = ca[2] * ba[0] - ca[0] * ba[2];

        let cross_d_x = da[1] * ba[2] - da[2] * ba[1];
        let cross_d_y = da[2] * ba[0] - da[0] * ba[2];

        vec![cross_c_x, cross_c_y, cross_d_x, cross_d_y]
    }

    fn jacobian(&self, points: &[Point<3>]) -> Vec<(usize, usize, f64)> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        let ba = [
            b.get(0) - a.get(0),
            b.get(1) - a.get(1),
            b.get(2) - a.get(2),
        ];

        let ca = [
            c.get(0) - a.get(0),
            c.get(1) - a.get(1),
            c.get(2) - a.get(2),
        ];

        let da = [
            d.get(0) - a.get(0),
            d.get(1) - a.get(1),
            d.get(2) - a.get(2),
        ];

        let mut entries = Vec::with_capacity(48);

        // f0 = ca[1]*ba[2] - ca[2]*ba[1] (cross_c_x)
        // ca[i] = C[i] - A[i], ba[i] = B[i] - A[i]
        //
        // d/dCy = ba[2], d/dCz = -ba[1]
        // d/dAy = -ba[2] + ca[2], d/dAz = ba[1] - ca[1]
        // d/dBy = -ca[2], d/dBz = ca[1]
        entries.push((0, var_col::<3>(self.line2_start, 1), ba[2]));
        entries.push((0, var_col::<3>(self.line2_start, 2), -ba[1]));
        entries.push((0, var_col::<3>(self.line1_start, 1), ca[2] - ba[2]));
        entries.push((0, var_col::<3>(self.line1_start, 2), ba[1] - ca[1]));
        entries.push((0, var_col::<3>(self.line1_end, 1), -ca[2]));
        entries.push((0, var_col::<3>(self.line1_end, 2), ca[1]));

        // f1 = ca[2]*ba[0] - ca[0]*ba[2] (cross_c_y)
        entries.push((1, var_col::<3>(self.line2_start, 2), ba[0]));
        entries.push((1, var_col::<3>(self.line2_start, 0), -ba[2]));
        entries.push((1, var_col::<3>(self.line1_start, 2), ca[0] - ba[0]));
        entries.push((1, var_col::<3>(self.line1_start, 0), ba[2] - ca[2]));
        entries.push((1, var_col::<3>(self.line1_end, 2), -ca[0]));
        entries.push((1, var_col::<3>(self.line1_end, 0), ca[2]));

        // f2 = da[1]*ba[2] - da[2]*ba[1] (cross_d_x)
        entries.push((2, var_col::<3>(self.line2_end, 1), ba[2]));
        entries.push((2, var_col::<3>(self.line2_end, 2), -ba[1]));
        entries.push((2, var_col::<3>(self.line1_start, 1), da[2] - ba[2]));
        entries.push((2, var_col::<3>(self.line1_start, 2), ba[1] - da[1]));
        entries.push((2, var_col::<3>(self.line1_end, 1), -da[2]));
        entries.push((2, var_col::<3>(self.line1_end, 2), da[1]));

        // f3 = da[2]*ba[0] - da[0]*ba[2] (cross_d_y)
        entries.push((3, var_col::<3>(self.line2_end, 2), ba[0]));
        entries.push((3, var_col::<3>(self.line2_end, 0), -ba[2]));
        entries.push((3, var_col::<3>(self.line1_start, 2), da[0] - ba[0]));
        entries.push((3, var_col::<3>(self.line1_start, 0), ba[2] - da[2]));
        entries.push((3, var_col::<3>(self.line1_end, 2), -da[0]));
        entries.push((3, var_col::<3>(self.line1_end, 0), da[2]));

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
        "Collinear3D"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_collinear_2d_satisfied() {
        let constraint = CollinearConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),   // A
            Point2D::new(10.0, 10.0), // B
            Point2D::new(3.0, 3.0),   // C (on line)
            Point2D::new(7.0, 7.0),   // D (on line)
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_collinear_2d_not_satisfied() {
        let constraint = CollinearConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),    // A
            Point2D::new(10.0, 0.0),   // B (horizontal line)
            Point2D::new(5.0, 5.0),    // C (not on line)
            Point2D::new(7.0, 7.0),    // D (not on line)
        ];

        let residuals = constraint.residuals(&points);
        // At least one residual should be non-zero
        let total = residuals[0].abs() + residuals[1].abs();
        assert!(total > 0.1);
    }

    #[test]
    fn test_collinear_2d_horizontal() {
        let constraint = CollinearConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 5.0),   // A
            Point2D::new(10.0, 5.0),  // B
            Point2D::new(3.0, 5.0),   // C (on horizontal line y=5)
            Point2D::new(8.0, 5.0),   // D (on horizontal line y=5)
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_collinear_3d_satisfied() {
        let constraint = CollinearConstraint::<3>::new(0, 1, 2, 3);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),    // A
            Point3D::new(10.0, 10.0, 10.0), // B
            Point3D::new(3.0, 3.0, 3.0),    // C (on line)
            Point3D::new(7.0, 7.0, 7.0),    // D (on line)
        ];

        let residuals = constraint.residuals(&points);
        for r in &residuals {
            assert!(r.abs() < 1e-10, "residual: {}", r);
        }
    }

    #[test]
    fn test_collinear_3d_not_satisfied() {
        let constraint = CollinearConstraint::<3>::new(0, 1, 2, 3);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),    // A
            Point3D::new(10.0, 0.0, 0.0),   // B (along x-axis)
            Point3D::new(5.0, 5.0, 0.0),    // C (not on line)
            Point3D::new(7.0, 0.0, 5.0),    // D (not on line)
        ];

        let residuals = constraint.residuals(&points);
        // At least one residual should be non-zero
        let total: f64 = residuals.iter().map(|r| r.abs()).sum();
        assert!(total > 0.1);
    }

    #[test]
    fn test_collinear_jacobian_2d() {
        let constraint = CollinearConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 10.0),
            Point2D::new(3.0, 3.0),
            Point2D::new(7.0, 7.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 12); // 2 equations * 6 terms each

        // All entries should be finite
        for (_, _, val) in &jac {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_variable_indices() {
        let constraint = CollinearConstraint::<2>::new(0, 1, 2, 3);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }
}
