//! Angle constraint: line makes a specific angle with horizontal (2D).

use crate::geometry::point::Point;
use super::{get_point, var_col, GeometricConstraint};

/// Angle constraint: line from p1 to p2 makes angle theta with horizontal.
///
/// This constraint enforces that the line from point1 to point2 makes
/// a specific angle (in radians) with the positive x-axis.
///
/// This is a 2D-only constraint.
///
/// # Equation
///
/// Using the formula: `(p2.y - p1.y) * cos(theta) - (p2.x - p1.x) * sin(theta) = 0`
///
/// This ensures the direction vector is aligned with the target angle.
///
/// # Jacobian
///
/// - `d/d(p1.x) = sin(theta)`
/// - `d/d(p1.y) = -cos(theta)`
/// - `d/d(p2.x) = -sin(theta)`
/// - `d/d(p2.y) = cos(theta)`
#[derive(Clone, Debug)]
pub struct AngleConstraint {
    /// Index of line start point.
    pub line_start: usize,
    /// Index of line end point.
    pub line_end: usize,
    /// Target angle in radians from horizontal (counter-clockwise positive).
    pub angle: f64,
}

impl AngleConstraint {
    /// Create a new angle constraint.
    ///
    /// # Arguments
    ///
    /// * `line_start` - Index of the start point
    /// * `line_end` - Index of the end point
    /// * `angle` - Target angle in radians
    pub fn new(line_start: usize, line_end: usize, angle: f64) -> Self {
        Self {
            line_start,
            line_end,
            angle,
        }
    }

    /// Create an angle constraint with angle in degrees.
    pub fn from_degrees(line_start: usize, line_end: usize, degrees: f64) -> Self {
        Self::new(line_start, line_end, degrees.to_radians())
    }

    /// Create a horizontal angle constraint (0 degrees).
    pub fn horizontal(line_start: usize, line_end: usize) -> Self {
        Self::new(line_start, line_end, 0.0)
    }

    /// Create a vertical angle constraint (90 degrees).
    pub fn vertical(line_start: usize, line_end: usize) -> Self {
        Self::new(line_start, line_end, std::f64::consts::FRAC_PI_2)
    }
}

impl GeometricConstraint<2> for AngleConstraint {
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, points: &[Point<2>]) -> Vec<f64> {
        let p1 = get_point(points, self.line_start);
        let p2 = get_point(points, self.line_end);

        let dx = p2.get(0) - p1.get(0);
        let dy = p2.get(1) - p1.get(1);

        let cos_t = self.angle.cos();
        let sin_t = self.angle.sin();

        // dy * cos(theta) - dx * sin(theta) = 0
        vec![dy * cos_t - dx * sin_t]
    }

    fn jacobian(&self, _points: &[Point<2>]) -> Vec<(usize, usize, f64)> {
        let cos_t = self.angle.cos();
        let sin_t = self.angle.sin();

        // f = dy * cos(theta) - dx * sin(theta)
        // where dx = p2.x - p1.x, dy = p2.y - p1.y
        //
        // d/d(p1.x) = -d(dx)/d(p1.x) * (-sin) = sin(theta)
        // d/d(p1.y) = -d(dy)/d(p1.y) * cos = -cos(theta)
        // d/d(p2.x) = d(dx)/d(p2.x) * (-sin) = -sin(theta)
        // d/d(p2.y) = d(dy)/d(p2.y) * cos = cos(theta)

        vec![
            (0, var_col::<2>(self.line_start, 0), sin_t),  // d/d(p1.x)
            (0, var_col::<2>(self.line_start, 1), -cos_t), // d/d(p1.y)
            (0, var_col::<2>(self.line_end, 0), -sin_t),   // d/d(p2.x)
            (0, var_col::<2>(self.line_end, 1), cos_t),    // d/d(p2.y)
        ]
    }

    fn variable_indices(&self) -> Vec<usize> {
        vec![self.line_start, self.line_end]
    }

    fn name(&self) -> &'static str {
        "Angle"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point2D;

    #[test]
    fn test_angle_horizontal() {
        let constraint = AngleConstraint::horizontal(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0), // Horizontal line
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_angle_vertical() {
        let constraint = AngleConstraint::vertical(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(0.0, 10.0), // Vertical line
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_angle_45_degrees() {
        let constraint = AngleConstraint::from_degrees(0, 1, 45.0);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0), // 45 degree line
        ];

        let residuals = constraint.residuals(&points);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_angle_not_satisfied() {
        let constraint = AngleConstraint::horizontal(0, 1);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0), // 45 degrees, not horizontal
        ];

        let residuals = constraint.residuals(&points);
        // For horizontal (angle=0), residual = dy * cos(0) - dx * sin(0) = dy * 1 - dx * 0 = dy = 1
        assert!((residuals[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_angle_jacobian() {
        let angle = std::f64::consts::FRAC_PI_4; // 45 degrees
        let constraint = AngleConstraint::new(0, 1, angle);
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
        ];

        let jac = constraint.jacobian(&points);
        assert_eq!(jac.len(), 4);

        let cos_t = angle.cos();
        let sin_t = angle.sin();

        // Verify entries
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            let expected = match *col {
                0 => sin_t,       // d/d(p1.x)
                1 => -cos_t,      // d/d(p1.y)
                2 => -sin_t,      // d/d(p2.x)
                3 => cos_t,       // d/d(p2.y)
                _ => panic!("unexpected column"),
            };
            assert!((*val - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_variable_indices() {
        let constraint = AngleConstraint::new(3, 7, 0.0);
        let indices = constraint.variable_indices();
        assert_eq!(indices, vec![3, 7]);
    }
}
