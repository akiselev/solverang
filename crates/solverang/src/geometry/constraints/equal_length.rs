//! Equal length constraint: two line segments have the same length.

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Equal length constraint: |L1| = |L2|.
///
/// This constraint enforces that two line segments have the same length.
///
/// # Equation
///
/// `|B - A| - |D - C| = 0`
///
/// Expanded: `sqrt(sum_k (B[k]-A[k])^2) - sqrt(sum_k (D[k]-C[k])^2) = 0`
///
/// # Jacobian
///
/// For line 1 (A->B):
/// - `d/dA[k] = -(B[k]-A[k]) / len1`
/// - `d/dB[k] = (B[k]-A[k]) / len1`
///
/// For line 2 (C->D):
/// - `d/dC[k] = (D[k]-C[k]) / len2`
/// - `d/dD[k] = -(D[k]-C[k]) / len2`
#[derive(Clone, Debug)]
pub struct EqualLengthConstraint<const D: usize> {
    /// Start of first line.
    pub line1_start: usize,
    /// End of first line.
    pub line1_end: usize,
    /// Start of second line.
    pub line2_start: usize,
    /// End of second line.
    pub line2_end: usize,
}

impl<const D: usize> EqualLengthConstraint<D> {
    /// Create a new equal length constraint.
    pub fn new(line1_start: usize, line1_end: usize, line2_start: usize, line2_end: usize) -> Self {
        Self {
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        }
    }
}

impl<const D: usize> GeometricConstraint<D> for EqualLengthConstraint<D> {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<D>]) -> Vec<f64> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        let len1 = a.distance_to(&b);
        let len2 = c.distance_to(&d);

        vec![len1 - len2]
    }

    fn jacobian(&self, points: &[Point<D>]) -> Vec<(usize, usize, f64)> {
        let a = get_point(points, self.line1_start);
        let b = get_point(points, self.line1_end);
        let c = get_point(points, self.line2_start);
        let d = get_point(points, self.line2_end);

        let len1 = a.safe_distance_to(&b);
        let len2 = c.safe_distance_to(&d);

        let mut entries = Vec::with_capacity(D * 4);

        for k in 0..D {
            let diff1 = b.get(k) - a.get(k);
            let diff2 = d.get(k) - c.get(k);

            // f = len1 - len2
            // d(len1)/dA[k] = -diff1/len1
            // d(len1)/dB[k] = diff1/len1
            // d(len2)/dC[k] = -diff2/len2
            // d(len2)/dD[k] = diff2/len2
            //
            // d(f)/dA[k] = -diff1/len1
            // d(f)/dB[k] = diff1/len1
            // d(f)/dC[k] = diff2/len2 (negative because -len2)
            // d(f)/dD[k] = -diff2/len2

            entries.push((0, var_col::<D>(self.line1_start, k), -diff1 / len1));
            entries.push((0, var_col::<D>(self.line1_end, k), diff1 / len1));
            entries.push((0, var_col::<D>(self.line2_start, k), diff2 / len2));
            entries.push((0, var_col::<D>(self.line2_end, k), -diff2 / len2));
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
        "EqualLength"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_equal_length_2d_satisfied() {
        let constraint = EqualLengthConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),  // A
            Point2D::new(3.0, 4.0),  // B (length = 5)
            Point2D::new(10.0, 0.0), // C
            Point2D::new(10.0, 5.0), // D (length = 5)
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_equal_length_2d_not_satisfied() {
        let constraint = EqualLengthConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),  // A
            Point2D::new(3.0, 4.0),  // B (length = 5)
            Point2D::new(10.0, 0.0), // C
            Point2D::new(10.0, 10.0), // D (length = 10)
        ];

        let residuals = constraint.residuals(&points);
        // 5 - 10 = -5
        assert!((residuals[0] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_equal_length_3d_satisfied() {
        let constraint = EqualLengthConstraint::<3>::new(0, 1, 2, 3);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0), // A
            Point3D::new(1.0, 2.0, 2.0), // B (length = 3)
            Point3D::new(10.0, 0.0, 0.0), // C
            Point3D::new(10.0, 3.0, 0.0), // D (length = 3)
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_equal_length_3d_not_satisfied() {
        let constraint = EqualLengthConstraint::<3>::new(0, 1, 2, 3);
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0), // A
            Point3D::new(1.0, 2.0, 2.0), // B (length = 3)
            Point3D::new(10.0, 0.0, 0.0), // C
            Point3D::new(10.0, 4.0, 0.0), // D (length = 4)
        ];

        let residuals = constraint.residuals(&points);
        // 3 - 4 = -1
        assert!((residuals[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_equal_length_jacobian() {
        let constraint = EqualLengthConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),  // A
            Point2D::new(3.0, 4.0),  // B (length = 5)
            Point2D::new(10.0, 0.0), // C
            Point2D::new(10.0, 5.0), // D (length = 5)
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 8); // 4 points * 2 coords

        // Verify specific values for line 1: diff = (3, 4), len = 5
        // d/dAx = -3/5 = -0.6
        // d/dAy = -4/5 = -0.8
        // d/dBx = 3/5 = 0.6
        // d/dBy = 4/5 = 0.8

        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            assert!(val.is_finite());

            // Check a few specific values
            if *col == 0 {
                assert!((*val - (-0.6)).abs() < 1e-10, "d/dAx: {}", val);
            }
            if *col == 1 {
                assert!((*val - (-0.8)).abs() < 1e-10, "d/dAy: {}", val);
            }
        }
    }

    #[test]
    fn test_equal_length_zero_length() {
        // Both lines have zero length
        let constraint = EqualLengthConstraint::<2>::new(0, 1, 2, 3);
        let points = vec![
            Point2D::new(0.0, 0.0),  // A
            Point2D::new(0.0, 0.0),  // B (same as A)
            Point2D::new(10.0, 5.0), // C
            Point2D::new(10.0, 5.0), // D (same as C)
        ];

        let jac = constraint.jacobian(&points);
        // Should not panic, all values should be finite
        for (_, _, val) in &jac {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_variable_indices() {
        let constraint = EqualLengthConstraint::<2>::new(0, 1, 2, 3);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }
}
