//! Vector type with const generic dimension for 2D/3D geometry.

use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use super::point::{Point, MIN_EPSILON};

/// A vector in D-dimensional space.
///
/// Vectors represent displacements, directions, and velocities.
/// They support dot product, cross product (2D and 3D), and normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector<const D: usize> {
    /// The components of the vector.
    pub coords: [f64; D],
}

/// Type alias for 2D vectors.
pub type Vector2D = Vector<2>;

/// Type alias for 3D vectors.
pub type Vector3D = Vector<3>;

impl<const D: usize> Default for Vector<D> {
    fn default() -> Self {
        Self { coords: [0.0; D] }
    }
}

impl<const D: usize> Vector<D> {
    /// Create a vector from an array of components.
    pub fn from_coords(coords: [f64; D]) -> Self {
        Self { coords }
    }

    /// Create a zero vector.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Get component at index, returning 0.0 if out of bounds.
    pub fn get(&self, index: usize) -> f64 {
        self.coords.get(index).copied().unwrap_or(0.0)
    }

    /// Set component at index if within bounds.
    pub fn set(&mut self, index: usize, value: f64) {
        if let Some(coord) = self.coords.get_mut(index) {
            *coord = value;
        }
    }

    /// Compute the dot product with another vector.
    pub fn dot(&self, other: &Self) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Compute the squared magnitude (norm squared) of this vector.
    pub fn norm_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Compute the magnitude (norm) of this vector.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Compute the magnitude with a minimum epsilon guard.
    ///
    /// Use this when the norm will be used as a denominator to prevent division by zero.
    pub fn safe_norm(&self) -> f64 {
        self.norm().max(MIN_EPSILON)
    }

    /// Normalize this vector to unit length.
    ///
    /// Returns a zero vector if the original vector is too small to normalize safely.
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < MIN_EPSILON {
            return Self::zero();
        }
        self.scale(1.0 / n)
    }

    /// Scale the vector by a scalar.
    pub fn scale(&self, factor: f64) -> Self {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            *coord = self.get(i) * factor;
        }
        Self { coords }
    }

    /// Compute the projection of this vector onto another vector.
    ///
    /// Returns the component of `self` in the direction of `onto`.
    pub fn project_onto(&self, onto: &Self) -> Self {
        let denom = onto.norm_squared();
        if denom < MIN_EPSILON * MIN_EPSILON {
            return Self::zero();
        }
        let scale = self.dot(onto) / denom;
        onto.scale(scale)
    }

    /// Compute the perpendicular component of this vector relative to another.
    ///
    /// Returns `self - self.project_onto(direction)`.
    pub fn perpendicular_to(&self, direction: &Self) -> Self {
        *self - self.project_onto(direction)
    }

    /// Check if this vector is approximately zero.
    pub fn is_zero(&self, tolerance: f64) -> bool {
        self.norm() < tolerance
    }

    /// Convert to a point (position vector from origin).
    pub fn to_point(&self) -> Point<D> {
        Point::from_coords(self.coords)
    }

    /// Dimension of this vector type.
    pub const fn dimension() -> usize {
        D
    }
}

impl Vector2D {
    /// Create a new 2D vector.
    pub fn new(x: f64, y: f64) -> Self {
        Self { coords: [x, y] }
    }

    /// Get the x component.
    pub fn x(&self) -> f64 {
        self.coords[0]
    }

    /// Get the y component.
    pub fn y(&self) -> f64 {
        self.coords[1]
    }

    /// Set the x component.
    pub fn set_x(&mut self, x: f64) {
        self.coords[0] = x;
    }

    /// Set the y component.
    pub fn set_y(&mut self, y: f64) {
        self.coords[1] = y;
    }

    /// Compute the 2D cross product (returns scalar).
    ///
    /// This is the z-component of the 3D cross product when both vectors
    /// are in the xy-plane. It represents the signed area of the parallelogram
    /// formed by the two vectors.
    ///
    /// Positive result means `other` is counter-clockwise from `self`.
    pub fn cross(&self, other: &Self) -> f64 {
        self.x() * other.y() - self.y() * other.x()
    }

    /// Create a vector perpendicular to this one (90 degrees counter-clockwise).
    pub fn perpendicular(&self) -> Self {
        Self::new(-self.y(), self.x())
    }

    /// Compute the angle of this vector from the positive x-axis (in radians).
    pub fn angle(&self) -> f64 {
        self.y().atan2(self.x())
    }

    /// Create a unit vector at the given angle from the positive x-axis.
    pub fn from_angle(angle: f64) -> Self {
        Self::new(angle.cos(), angle.sin())
    }

    /// Compute the signed angle from this vector to another (in radians).
    ///
    /// Positive angle means counter-clockwise rotation from self to other.
    pub fn angle_to(&self, other: &Self) -> f64 {
        let cross = self.cross(other);
        let dot = self.dot(other);
        cross.atan2(dot)
    }
}

impl Vector3D {
    /// Create a new 3D vector.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { coords: [x, y, z] }
    }

    /// Get the x component.
    pub fn x(&self) -> f64 {
        self.coords[0]
    }

    /// Get the y component.
    pub fn y(&self) -> f64 {
        self.coords[1]
    }

    /// Get the z component.
    pub fn z(&self) -> f64 {
        self.coords[2]
    }

    /// Set the x component.
    pub fn set_x(&mut self, x: f64) {
        self.coords[0] = x;
    }

    /// Set the y component.
    pub fn set_y(&mut self, y: f64) {
        self.coords[1] = y;
    }

    /// Set the z component.
    pub fn set_z(&mut self, z: f64) {
        self.coords[2] = z;
    }

    /// Compute the 3D cross product.
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        )
    }

    /// Unit vector along the x-axis.
    pub fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    /// Unit vector along the y-axis.
    pub fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    /// Unit vector along the z-axis.
    pub fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    /// Create a 2D projection by dropping the z component.
    pub fn to_2d(&self) -> Vector2D {
        Vector2D::new(self.x(), self.y())
    }
}

impl<const D: usize> Index<usize> for Vector<D> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl<const D: usize> IndexMut<usize> for Vector<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl<const D: usize> Add for Vector<D> {
    type Output = Vector<D>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            *coord = self.get(i) + rhs.get(i);
        }
        Vector { coords }
    }
}

impl<const D: usize> Sub for Vector<D> {
    type Output = Vector<D>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            *coord = self.get(i) - rhs.get(i);
        }
        Vector { coords }
    }
}

impl<const D: usize> Neg for Vector<D> {
    type Output = Vector<D>;

    fn neg(self) -> Self::Output {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate() {
            *coord = -self.get(i);
        }
        Vector { coords }
    }
}

impl<const D: usize> Mul<f64> for Vector<D> {
    type Output = Vector<D>;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

impl<const D: usize> From<[f64; D]> for Vector<D> {
    fn from(coords: [f64; D]) -> Self {
        Self { coords }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector2d_creation() {
        let v = Vector2D::new(3.0, 4.0);
        assert_eq!(v.x(), 3.0);
        assert_eq!(v.y(), 4.0);
    }

    #[test]
    fn test_vector3d_creation() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(v.x(), 1.0);
        assert_eq!(v.y(), 2.0);
        assert_eq!(v.z(), 3.0);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector2D::new(1.0, 2.0);
        let v2 = Vector2D::new(3.0, 4.0);
        assert!((v1.dot(&v2) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm() {
        let v = Vector2D::new(3.0, 4.0);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_2d() {
        let v1 = Vector2D::new(1.0, 0.0);
        let v2 = Vector2D::new(0.0, 1.0);
        assert!((v1.cross(&v2) - 1.0).abs() < 1e-10);
        assert!((v2.cross(&v1) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cross_3d() {
        let v1 = Vector3D::new(1.0, 0.0, 0.0);
        let v2 = Vector3D::new(0.0, 1.0, 0.0);
        let cross = v1.cross(&v2);
        assert!((cross.x() - 0.0).abs() < 1e-10);
        assert!((cross.y() - 0.0).abs() < 1e-10);
        assert!((cross.z() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized() {
        let v = Vector2D::new(3.0, 4.0);
        let n = v.normalized();
        assert!((n.norm() - 1.0).abs() < 1e-10);
        assert!((n.x() - 0.6).abs() < 1e-10);
        assert!((n.y() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_zero() {
        let v = Vector2D::new(0.0, 0.0);
        let n = v.normalized();
        assert_eq!(n.x(), 0.0);
        assert_eq!(n.y(), 0.0);
    }

    #[test]
    fn test_perpendicular() {
        let v = Vector2D::new(1.0, 0.0);
        let p = v.perpendicular();
        assert!((p.x() - 0.0).abs() < 1e-10);
        assert!((p.y() - 1.0).abs() < 1e-10);
        assert!((v.dot(&p)).abs() < 1e-10);
    }

    #[test]
    fn test_angle() {
        let v = Vector2D::new(1.0, 0.0);
        assert!(v.angle().abs() < 1e-10);

        let v2 = Vector2D::new(0.0, 1.0);
        assert!((v2.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_projection() {
        let v = Vector2D::new(3.0, 4.0);
        let onto = Vector2D::new(1.0, 0.0);
        let proj = v.project_onto(&onto);
        assert!((proj.x() - 3.0).abs() < 1e-10);
        assert!((proj.y() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_sub() {
        let v1 = Vector2D::new(1.0, 2.0);
        let v2 = Vector2D::new(3.0, 4.0);

        let sum = v1 + v2;
        assert_eq!(sum.x(), 4.0);
        assert_eq!(sum.y(), 6.0);

        let diff = v1 - v2;
        assert_eq!(diff.x(), -2.0);
        assert_eq!(diff.y(), -2.0);
    }

    #[test]
    fn test_neg() {
        let v = Vector2D::new(1.0, -2.0);
        let neg = -v;
        assert_eq!(neg.x(), -1.0);
        assert_eq!(neg.y(), 2.0);
    }
}
