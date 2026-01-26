//! Line representation for geometric constraints.

use super::point::{Point, MIN_EPSILON};
use super::vector::Vector;

/// A line segment defined by two endpoints.
///
/// This representation is used for line-related constraints like
/// parallel, perpendicular, angle, and tangent constraints.
#[derive(Clone, Copy, Debug)]
pub struct Line<const D: usize> {
    /// Start point of the line segment.
    pub start: Point<D>,
    /// End point of the line segment.
    pub end: Point<D>,
}

/// Type alias for 2D line segments.
pub type Line2D = Line<2>;

/// Type alias for 3D line segments.
pub type Line3D = Line<3>;

impl<const D: usize> Line<D> {
    /// Create a new line segment from two points.
    pub fn new(start: Point<D>, end: Point<D>) -> Self {
        Self { start, end }
    }

    /// Get the direction vector (end - start).
    pub fn direction(&self) -> Vector<D> {
        self.end - self.start
    }

    /// Get the unit direction vector.
    ///
    /// Returns zero vector if line has zero length.
    pub fn unit_direction(&self) -> Vector<D> {
        self.direction().normalized()
    }

    /// Get the length of the line segment.
    pub fn length(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    /// Get the length with a minimum epsilon guard.
    pub fn safe_length(&self) -> f64 {
        self.length().max(MIN_EPSILON)
    }

    /// Get the midpoint of the line segment.
    pub fn midpoint(&self) -> Point<D> {
        self.start.midpoint(&self.end)
    }

    /// Get a point at parameter t along the line segment.
    ///
    /// `t = 0` returns start, `t = 1` returns end.
    pub fn point_at(&self, t: f64) -> Point<D> {
        self.start.lerp(&self.end, t)
    }

    /// Check if this line segment has essentially zero length.
    pub fn is_degenerate(&self, tolerance: f64) -> bool {
        self.length() < tolerance
    }
}

impl Line2D {
    /// Create a 2D line from individual coordinates.
    pub fn from_coords(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self::new(
            Point::from_coords([x1, y1]),
            Point::from_coords([x2, y2]),
        )
    }

    /// Compute the signed distance from a point to this line (extended infinitely).
    ///
    /// Positive values indicate the point is to the left of the line direction,
    /// negative values indicate the point is to the right.
    pub fn signed_distance_to_point(&self, point: &super::point::Point2D) -> f64 {
        let dir = self.direction();
        let line_len = dir.norm();
        if line_len < MIN_EPSILON {
            return self.start.distance_to(point);
        }

        let to_point = *point - self.start;
        // Cross product gives signed area of parallelogram, divide by base for height
        dir.cross(&to_point) / line_len
    }

    /// Compute the perpendicular distance from a point to this line (extended infinitely).
    pub fn distance_to_point(&self, point: &super::point::Point2D) -> f64 {
        self.signed_distance_to_point(point).abs()
    }

    /// Compute the perpendicular distance with a minimum epsilon guard.
    pub fn safe_distance_to_point(&self, point: &super::point::Point2D) -> f64 {
        self.distance_to_point(point).max(MIN_EPSILON)
    }

    /// Project a point onto this line (extended infinitely).
    ///
    /// Returns the closest point on the infinite line.
    pub fn project_point(&self, point: &super::point::Point2D) -> super::point::Point2D {
        let dir = self.direction();
        let len_sq = dir.norm_squared();
        if len_sq < MIN_EPSILON * MIN_EPSILON {
            return self.start;
        }

        let to_point = *point - self.start;
        let t = dir.dot(&to_point) / len_sq;
        self.point_at(t)
    }

    /// Check if a point lies on this line (extended infinitely).
    pub fn contains_point(&self, point: &super::point::Point2D, tolerance: f64) -> bool {
        self.distance_to_point(point) < tolerance
    }

    /// Check if this line is parallel to another.
    pub fn is_parallel_to(&self, other: &Self, tolerance: f64) -> bool {
        let d1 = self.direction();
        let d2 = other.direction();
        // Cross product is zero for parallel vectors
        d1.cross(&d2).abs() < tolerance
    }

    /// Check if this line is perpendicular to another.
    pub fn is_perpendicular_to(&self, other: &Self, tolerance: f64) -> bool {
        let d1 = self.direction();
        let d2 = other.direction();
        // Dot product is zero for perpendicular vectors
        d1.dot(&d2).abs() < tolerance
    }

    /// Compute the angle between this line and the positive x-axis.
    pub fn angle(&self) -> f64 {
        self.direction().angle()
    }

    /// Get the normal vector (perpendicular to direction, unit length).
    pub fn normal(&self) -> super::vector::Vector2D {
        self.unit_direction().perpendicular()
    }
}

impl Line3D {
    /// Create a 3D line from individual coordinates.
    pub fn from_coords(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> Self {
        Self::new(
            Point::from_coords([x1, y1, z1]),
            Point::from_coords([x2, y2, z2]),
        )
    }

    /// Compute the distance from a point to this line (extended infinitely).
    pub fn distance_to_point(&self, point: &super::point::Point3D) -> f64 {
        let dir = self.direction();
        let line_len = dir.norm();
        if line_len < MIN_EPSILON {
            return self.start.distance_to(point);
        }

        let to_point = *point - self.start;
        // Cross product magnitude gives area of parallelogram, divide by base for height
        dir.cross(&to_point).norm() / line_len
    }

    /// Compute the perpendicular distance with a minimum epsilon guard.
    pub fn safe_distance_to_point(&self, point: &super::point::Point3D) -> f64 {
        self.distance_to_point(point).max(MIN_EPSILON)
    }

    /// Project a point onto this line (extended infinitely).
    pub fn project_point(&self, point: &super::point::Point3D) -> super::point::Point3D {
        let dir = self.direction();
        let len_sq = dir.norm_squared();
        if len_sq < MIN_EPSILON * MIN_EPSILON {
            return self.start;
        }

        let to_point = *point - self.start;
        let t = dir.dot(&to_point) / len_sq;
        self.point_at(t)
    }

    /// Check if a point lies on this line (extended infinitely).
    pub fn contains_point(&self, point: &super::point::Point3D, tolerance: f64) -> bool {
        self.distance_to_point(point) < tolerance
    }

    /// Check if this line is parallel to another.
    pub fn is_parallel_to(&self, other: &Self, tolerance: f64) -> bool {
        let d1 = self.direction();
        let d2 = other.direction();
        // Cross product is zero for parallel vectors
        d1.cross(&d2).norm() < tolerance
    }

    /// Check if this line is perpendicular to another.
    pub fn is_perpendicular_to(&self, other: &Self, tolerance: f64) -> bool {
        let d1 = self.direction();
        let d2 = other.direction();
        // Dot product is zero for perpendicular vectors
        d1.dot(&d2).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_line2d_creation() {
        let line = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(3.0, 4.0));
        assert!((line.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_line2d_direction() {
        let line = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(3.0, 4.0));
        let dir = line.direction();
        assert_eq!(dir.x(), 3.0);
        assert_eq!(dir.y(), 4.0);
    }

    #[test]
    fn test_line2d_midpoint() {
        let line = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(4.0, 6.0));
        let mid = line.midpoint();
        assert_eq!(mid.x(), 2.0);
        assert_eq!(mid.y(), 3.0);
    }

    #[test]
    fn test_line2d_distance_to_point() {
        let line = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(10.0, 0.0));
        let point = Point2D::new(5.0, 3.0);
        assert!((line.distance_to_point(&point) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_line2d_project_point() {
        let line = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(10.0, 0.0));
        let point = Point2D::new(5.0, 3.0);
        let projected = line.project_point(&point);
        assert!((projected.x() - 5.0).abs() < 1e-10);
        assert!((projected.y() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_line2d_parallel() {
        let line1 = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0));
        let line2 = Line2D::new(Point2D::new(0.0, 1.0), Point2D::new(1.0, 2.0));
        assert!(line1.is_parallel_to(&line2, 1e-10));

        let line3 = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0));
        assert!(!line1.is_parallel_to(&line3, 1e-10));
    }

    #[test]
    fn test_line2d_perpendicular() {
        let line1 = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0));
        let line2 = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(0.0, 1.0));
        assert!(line1.is_perpendicular_to(&line2, 1e-10));

        let line3 = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0));
        assert!(!line1.is_perpendicular_to(&line3, 1e-10));
    }

    #[test]
    fn test_line3d_distance_to_point() {
        let line = Line3D::new(
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(10.0, 0.0, 0.0),
        );
        let point = Point3D::new(5.0, 3.0, 4.0);
        assert!((line.distance_to_point(&point) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_line3d_project_point() {
        let line = Line3D::new(
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(10.0, 0.0, 0.0),
        );
        let point = Point3D::new(5.0, 3.0, 4.0);
        let projected = line.project_point(&point);
        assert!((projected.x() - 5.0).abs() < 1e-10);
        assert!((projected.y() - 0.0).abs() < 1e-10);
        assert!((projected.z() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_degenerate_line() {
        let line = Line2D::new(Point2D::new(1.0, 2.0), Point2D::new(1.0, 2.0));
        assert!(line.is_degenerate(1e-10));

        let line2 = Line2D::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0));
        assert!(!line2.is_degenerate(1e-10));
    }
}
