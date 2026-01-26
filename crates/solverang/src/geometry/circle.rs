//! Circle and sphere representation for geometric constraints.

use super::point::{Point2D, Point3D, MIN_EPSILON};

/// A circle in 2D space defined by center and radius.
#[derive(Clone, Copy, Debug)]
pub struct Circle {
    /// Center point of the circle.
    pub center: Point2D,
    /// Radius of the circle.
    pub radius: f64,
}

impl Circle {
    /// Create a new circle with the given center and radius.
    pub fn new(center: Point2D, radius: f64) -> Self {
        Self { center, radius }
    }

    /// Create a circle from center coordinates and radius.
    pub fn from_coords(x: f64, y: f64, radius: f64) -> Self {
        Self::new(Point2D::new(x, y), radius)
    }

    /// Check if a point lies on the circle.
    pub fn contains_point(&self, point: &Point2D, tolerance: f64) -> bool {
        (self.center.distance_to(point) - self.radius).abs() < tolerance
    }

    /// Check if a point is inside the circle.
    pub fn is_inside(&self, point: &Point2D) -> bool {
        self.center.distance_to(point) < self.radius
    }

    /// Check if a point is outside the circle.
    pub fn is_outside(&self, point: &Point2D) -> bool {
        self.center.distance_to(point) > self.radius
    }

    /// Get the signed distance from a point to the circle.
    ///
    /// Negative if inside, positive if outside.
    pub fn signed_distance_to(&self, point: &Point2D) -> f64 {
        self.center.distance_to(point) - self.radius
    }

    /// Get the closest point on the circle to a given point.
    ///
    /// Returns the center if the point coincides with the center.
    pub fn closest_point_to(&self, point: &Point2D) -> Point2D {
        let dist = self.center.distance_to(point);
        if dist < MIN_EPSILON {
            // Point is at center - return any point on circle
            return Point2D::new(self.center.x() + self.radius, self.center.y());
        }

        let direction = (*point - self.center).normalized();
        self.center + direction * self.radius
    }

    /// Get the area of the circle.
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    /// Get the circumference of the circle.
    pub fn circumference(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }

    /// Check if this circle is tangent to another circle (externally).
    ///
    /// Two circles are externally tangent when the distance between
    /// their centers equals the sum of their radii.
    pub fn is_externally_tangent_to(&self, other: &Self, tolerance: f64) -> bool {
        let center_dist = self.center.distance_to(&other.center);
        let expected = self.radius + other.radius;
        (center_dist - expected).abs() < tolerance
    }

    /// Check if this circle is tangent to another circle (internally).
    ///
    /// Two circles are internally tangent when the distance between
    /// their centers equals the absolute difference of their radii.
    pub fn is_internally_tangent_to(&self, other: &Self, tolerance: f64) -> bool {
        let center_dist = self.center.distance_to(&other.center);
        let expected = (self.radius - other.radius).abs();
        (center_dist - expected).abs() < tolerance
    }

    /// Check if this circle is tangent to a line.
    pub fn is_tangent_to_line(&self, line: &super::line::Line2D, tolerance: f64) -> bool {
        let dist = line.distance_to_point(&self.center);
        (dist - self.radius).abs() < tolerance
    }

    /// Check if this circle intersects another circle.
    pub fn intersects(&self, other: &Self) -> bool {
        let center_dist = self.center.distance_to(&other.center);
        let sum_radii = self.radius + other.radius;
        let diff_radii = (self.radius - other.radius).abs();

        center_dist < sum_radii && center_dist > diff_radii
    }

    /// Check if this is a degenerate circle (zero or negative radius).
    pub fn is_degenerate(&self) -> bool {
        self.radius < MIN_EPSILON
    }
}

/// A sphere in 3D space defined by center and radius.
#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    /// Center point of the sphere.
    pub center: Point3D,
    /// Radius of the sphere.
    pub radius: f64,
}

impl Sphere {
    /// Create a new sphere with the given center and radius.
    pub fn new(center: Point3D, radius: f64) -> Self {
        Self { center, radius }
    }

    /// Create a sphere from center coordinates and radius.
    pub fn from_coords(x: f64, y: f64, z: f64, radius: f64) -> Self {
        Self::new(Point3D::new(x, y, z), radius)
    }

    /// Check if a point lies on the sphere surface.
    pub fn contains_point(&self, point: &Point3D, tolerance: f64) -> bool {
        (self.center.distance_to(point) - self.radius).abs() < tolerance
    }

    /// Check if a point is inside the sphere.
    pub fn is_inside(&self, point: &Point3D) -> bool {
        self.center.distance_to(point) < self.radius
    }

    /// Check if a point is outside the sphere.
    pub fn is_outside(&self, point: &Point3D) -> bool {
        self.center.distance_to(point) > self.radius
    }

    /// Get the signed distance from a point to the sphere surface.
    ///
    /// Negative if inside, positive if outside.
    pub fn signed_distance_to(&self, point: &Point3D) -> f64 {
        self.center.distance_to(point) - self.radius
    }

    /// Get the closest point on the sphere to a given point.
    ///
    /// Returns a point on the sphere if the query point coincides with the center.
    pub fn closest_point_to(&self, point: &Point3D) -> Point3D {
        let dist = self.center.distance_to(point);
        if dist < MIN_EPSILON {
            // Point is at center - return any point on sphere
            return Point3D::new(self.center.x() + self.radius, self.center.y(), self.center.z());
        }

        let direction = (*point - self.center).normalized();
        self.center + direction * self.radius
    }

    /// Get the volume of the sphere.
    pub fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.radius * self.radius * self.radius
    }

    /// Get the surface area of the sphere.
    pub fn surface_area(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius * self.radius
    }

    /// Check if this sphere is externally tangent to another sphere.
    pub fn is_externally_tangent_to(&self, other: &Self, tolerance: f64) -> bool {
        let center_dist = self.center.distance_to(&other.center);
        let expected = self.radius + other.radius;
        (center_dist - expected).abs() < tolerance
    }

    /// Check if this sphere is internally tangent to another sphere.
    pub fn is_internally_tangent_to(&self, other: &Self, tolerance: f64) -> bool {
        let center_dist = self.center.distance_to(&other.center);
        let expected = (self.radius - other.radius).abs();
        (center_dist - expected).abs() < tolerance
    }

    /// Check if this sphere intersects another sphere.
    pub fn intersects(&self, other: &Self) -> bool {
        let center_dist = self.center.distance_to(&other.center);
        let sum_radii = self.radius + other.radius;
        let diff_radii = (self.radius - other.radius).abs();

        center_dist < sum_radii && center_dist > diff_radii
    }

    /// Check if this is a degenerate sphere (zero or negative radius).
    pub fn is_degenerate(&self) -> bool {
        self.radius < MIN_EPSILON
    }
}

/// Type of tangency between two circles/spheres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TangentType {
    /// Circles/spheres touch from outside (center distance = r1 + r2).
    External,
    /// One circle/sphere inside the other (center distance = |r1 - r2|).
    Internal,
}

impl TangentType {
    /// Get the target distance for this tangent type.
    pub fn target_distance(&self, radius1: f64, radius2: f64) -> f64 {
        match self {
            TangentType::External => radius1 + radius2,
            TangentType::Internal => (radius1 - radius2).abs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_creation() {
        let c = Circle::new(Point2D::new(1.0, 2.0), 5.0);
        assert_eq!(c.center.x(), 1.0);
        assert_eq!(c.center.y(), 2.0);
        assert_eq!(c.radius, 5.0);
    }

    #[test]
    fn test_circle_contains_point() {
        let c = Circle::new(Point2D::origin(), 5.0);
        assert!(c.contains_point(&Point2D::new(5.0, 0.0), 1e-10));
        assert!(c.contains_point(&Point2D::new(3.0, 4.0), 1e-10));
        assert!(!c.contains_point(&Point2D::new(0.0, 0.0), 1e-10));
    }

    #[test]
    fn test_circle_inside_outside() {
        let c = Circle::new(Point2D::origin(), 5.0);
        assert!(c.is_inside(&Point2D::new(1.0, 1.0)));
        assert!(c.is_outside(&Point2D::new(6.0, 0.0)));
    }

    #[test]
    fn test_circle_closest_point() {
        let c = Circle::new(Point2D::origin(), 5.0);
        let closest = c.closest_point_to(&Point2D::new(10.0, 0.0));
        assert!((closest.x() - 5.0).abs() < 1e-10);
        assert!((closest.y() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_tangent_external() {
        let c1 = Circle::new(Point2D::new(0.0, 0.0), 3.0);
        let c2 = Circle::new(Point2D::new(8.0, 0.0), 5.0);
        assert!(c1.is_externally_tangent_to(&c2, 1e-10));
    }

    #[test]
    fn test_circle_tangent_internal() {
        let c1 = Circle::new(Point2D::new(0.0, 0.0), 5.0);
        let c2 = Circle::new(Point2D::new(2.0, 0.0), 3.0);
        assert!(c1.is_internally_tangent_to(&c2, 1e-10));
    }

    #[test]
    fn test_circle_intersects() {
        let c1 = Circle::new(Point2D::new(0.0, 0.0), 5.0);
        let c2 = Circle::new(Point2D::new(6.0, 0.0), 3.0);
        assert!(c1.intersects(&c2));

        let c3 = Circle::new(Point2D::new(10.0, 0.0), 1.0);
        assert!(!c1.intersects(&c3));
    }

    #[test]
    fn test_sphere_creation() {
        let s = Sphere::new(Point3D::new(1.0, 2.0, 3.0), 5.0);
        assert_eq!(s.center.x(), 1.0);
        assert_eq!(s.center.y(), 2.0);
        assert_eq!(s.center.z(), 3.0);
        assert_eq!(s.radius, 5.0);
    }

    #[test]
    fn test_sphere_contains_point() {
        let s = Sphere::new(Point3D::origin(), 5.0);
        assert!(s.contains_point(&Point3D::new(5.0, 0.0, 0.0), 1e-10));
        assert!(s.contains_point(&Point3D::new(3.0, 4.0, 0.0), 1e-10));
        assert!(!s.contains_point(&Point3D::new(0.0, 0.0, 0.0), 1e-10));
    }

    #[test]
    fn test_tangent_type() {
        assert_eq!(TangentType::External.target_distance(3.0, 5.0), 8.0);
        assert_eq!(TangentType::Internal.target_distance(5.0, 3.0), 2.0);
        assert_eq!(TangentType::Internal.target_distance(3.0, 5.0), 2.0);
    }
}
