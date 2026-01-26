//! Point type with const generic dimension for 2D/3D geometry.

use std::ops::{Add, Index, IndexMut, Mul, Sub};

/// Minimum epsilon for distance calculations to prevent division by zero.
pub const MIN_EPSILON: f64 = 1e-10;

/// A point in D-dimensional space.
///
/// This is the fundamental primitive for the geometric constraint solver.
/// The const generic parameter `D` specifies the dimension (2 for 2D, 3 for 3D).
///
/// # Example
///
/// ```rust
/// use solverang::geometry::{Point, Point2D, Point3D};
///
/// let p2d = Point2D::new(1.0, 2.0);
/// let p3d = Point3D::new(1.0, 2.0, 3.0);
///
/// assert_eq!(p2d.x(), 1.0);
/// assert_eq!(p3d.z(), 3.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point<const D: usize> {
    /// The coordinates of the point.
    pub coords: [f64; D],
}

/// Type alias for 2D points.
pub type Point2D = Point<2>;

/// Type alias for 3D points.
pub type Point3D = Point<3>;

impl<const D: usize> Default for Point<D> {
    fn default() -> Self {
        Self { coords: [0.0; D] }
    }
}

impl<const D: usize> Point<D> {
    /// Create a point from an array of coordinates.
    pub fn from_coords(coords: [f64; D]) -> Self {
        Self { coords }
    }

    /// Create a point with all coordinates set to zero (origin).
    pub fn origin() -> Self {
        Self::default()
    }

    /// Get coordinate at index, returning 0.0 if out of bounds.
    pub fn get(&self, index: usize) -> f64 {
        self.coords.get(index).copied().unwrap_or(0.0)
    }

    /// Set coordinate at index if within bounds.
    pub fn set(&mut self, index: usize, value: f64) {
        if let Some(coord) = self.coords.get_mut(index) {
            *coord = value;
        }
    }

    /// Compute the Euclidean distance to another point.
    pub fn distance_to(&self, other: &Self) -> f64 {
        self.distance_squared_to(other).sqrt()
    }

    /// Compute the squared Euclidean distance to another point.
    ///
    /// More efficient than `distance_to` when only comparison is needed.
    pub fn distance_squared_to(&self, other: &Self) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| {
                let d = b - a;
                d * d
            })
            .sum()
    }

    /// Compute the distance to another point, with a minimum epsilon guard.
    ///
    /// Use this when the distance will be used as a denominator to prevent division by zero.
    pub fn safe_distance_to(&self, other: &Self) -> f64 {
        self.distance_to(other).max(MIN_EPSILON)
    }

    /// Convert to a vector (displacement from origin).
    pub fn to_vec(&self) -> super::vector::Vector<D> {
        super::vector::Vector::from_coords(self.coords)
    }

    /// Create a point from a vector (position vector).
    pub fn from_vec(v: &super::vector::Vector<D>) -> Self {
        Self::from_coords(v.coords)
    }

    /// Midpoint between this point and another.
    pub fn midpoint(&self, other: &Self) -> Self {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            *coord = (self.get(i) + other.get(i)) / 2.0;
        }
        Self { coords }
    }

    /// Linear interpolation between this point and another.
    ///
    /// `t = 0` returns `self`, `t = 1` returns `other`.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            let a = self.get(i);
            let b = other.get(i);
            *coord = a + t * (b - a);
        }
        Self { coords }
    }

    /// Dimension of this point type.
    pub const fn dimension() -> usize {
        D
    }
}

impl Point2D {
    /// Create a new 2D point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { coords: [x, y] }
    }

    /// Get the x coordinate.
    pub fn x(&self) -> f64 {
        self.coords[0]
    }

    /// Get the y coordinate.
    pub fn y(&self) -> f64 {
        self.coords[1]
    }

    /// Set the x coordinate.
    pub fn set_x(&mut self, x: f64) {
        self.coords[0] = x;
    }

    /// Set the y coordinate.
    pub fn set_y(&mut self, y: f64) {
        self.coords[1] = y;
    }
}

impl Point3D {
    /// Create a new 3D point.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { coords: [x, y, z] }
    }

    /// Get the x coordinate.
    pub fn x(&self) -> f64 {
        self.coords[0]
    }

    /// Get the y coordinate.
    pub fn y(&self) -> f64 {
        self.coords[1]
    }

    /// Get the z coordinate.
    pub fn z(&self) -> f64 {
        self.coords[2]
    }

    /// Set the x coordinate.
    pub fn set_x(&mut self, x: f64) {
        self.coords[0] = x;
    }

    /// Set the y coordinate.
    pub fn set_y(&mut self, y: f64) {
        self.coords[1] = y;
    }

    /// Set the z coordinate.
    pub fn set_z(&mut self, z: f64) {
        self.coords[2] = z;
    }

    /// Create a 2D projection by dropping the z coordinate.
    pub fn to_2d(&self) -> Point2D {
        Point2D::new(self.x(), self.y())
    }
}

impl<const D: usize> Index<usize> for Point<D> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl<const D: usize> IndexMut<usize> for Point<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl<const D: usize> Add<super::vector::Vector<D>> for Point<D> {
    type Output = Point<D>;

    fn add(self, rhs: super::vector::Vector<D>) -> Self::Output {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            *coord = self.get(i) + rhs.get(i);
        }
        Point { coords }
    }
}

impl<const D: usize> Sub for Point<D> {
    type Output = super::vector::Vector<D>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            *coord = self.get(i) - rhs.get(i);
        }
        super::vector::Vector { coords }
    }
}

impl<const D: usize> Mul<f64> for Point<D> {
    type Output = Point<D>;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            *coord = self.get(i) * rhs;
        }
        Point { coords }
    }
}

impl<const D: usize> From<[f64; D]> for Point<D> {
    fn from(coords: [f64; D]) -> Self {
        Self { coords }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_creation() {
        let p = Point2D::new(3.0, 4.0);
        assert_eq!(p.x(), 3.0);
        assert_eq!(p.y(), 4.0);
    }

    #[test]
    fn test_point3d_creation() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(p.x(), 1.0);
        assert_eq!(p.y(), 2.0);
        assert_eq!(p.z(), 3.0);
    }

    #[test]
    fn test_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_3d() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(1.0, 2.0, 2.0);
        assert!((p1.distance_to(&p2) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(4.0, 6.0);
        let mid = p1.midpoint(&p2);
        assert_eq!(mid.x(), 2.0);
        assert_eq!(mid.y(), 3.0);
    }

    #[test]
    fn test_lerp() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(10.0, 20.0);

        let at_0 = p1.lerp(&p2, 0.0);
        assert_eq!(at_0.x(), 0.0);
        assert_eq!(at_0.y(), 0.0);

        let at_1 = p1.lerp(&p2, 1.0);
        assert_eq!(at_1.x(), 10.0);
        assert_eq!(at_1.y(), 20.0);

        let at_half = p1.lerp(&p2, 0.5);
        assert_eq!(at_half.x(), 5.0);
        assert_eq!(at_half.y(), 10.0);
    }

    #[test]
    fn test_indexing() {
        let mut p = Point2D::new(1.0, 2.0);
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 2.0);

        p[0] = 5.0;
        assert_eq!(p[0], 5.0);
    }

    #[test]
    fn test_origin() {
        let o = Point2D::origin();
        assert_eq!(o.x(), 0.0);
        assert_eq!(o.y(), 0.0);
    }
}
