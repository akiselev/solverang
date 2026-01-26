//! Fluent builder API for constraint systems.

use super::point::Point;
use super::system::ConstraintSystem;
use super::constraints::*;

/// Builder for constructing constraint systems with a fluent API.
///
/// # Example
///
/// ```rust
/// use solverang::geometry::{ConstraintSystemBuilder, Point2D};
///
/// let system = ConstraintSystemBuilder::<2>::new()
///     .name("Triangle")
///     .point(Point2D::new(0.0, 0.0))       // p0
///     .point(Point2D::new(10.0, 0.0))      // p1
///     .point(Point2D::new(5.0, 8.0))       // p2
///     .fix(0)                              // Fix first point
///     .distance(0, 1, 10.0)                // p0-p1 = 10
///     .distance(1, 2, 10.0)                // p1-p2 = 10
///     .distance(2, 0, 10.0)                // p2-p0 = 10
///     .build();
///
/// assert_eq!(system.point_count(), 3);
/// assert_eq!(system.constraint_count(), 3);
/// ```
pub struct ConstraintSystemBuilder<const D: usize> {
    system: ConstraintSystem<D>,
}

impl<const D: usize> Default for ConstraintSystemBuilder<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> ConstraintSystemBuilder<D> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            system: ConstraintSystem::new(),
        }
    }

    /// Set the system name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.system = ConstraintSystem::with_name(name);
        self
    }

    /// Add a free point.
    pub fn point(mut self, point: Point<D>) -> Self {
        self.system.add_point(point);
        self
    }

    /// Add a fixed point.
    pub fn point_fixed(mut self, point: Point<D>) -> Self {
        self.system.add_point_fixed(point);
        self
    }

    /// Add multiple free points.
    pub fn points(mut self, points: impl IntoIterator<Item = Point<D>>) -> Self {
        for p in points {
            self.system.add_point(p);
        }
        self
    }

    /// Fix a point by index.
    pub fn fix(mut self, index: usize) -> Self {
        self.system.fix_point(index);
        self
    }

    /// Free a point by index.
    pub fn free(mut self, index: usize) -> Self {
        self.system.free_point(index);
        self
    }

    /// Add a distance constraint.
    pub fn distance(mut self, p1: usize, p2: usize, target: f64) -> Self {
        self.system
            .add_constraint(Box::new(DistanceConstraint::<D>::new(p1, p2, target)));
        self
    }

    /// Add a coincident constraint.
    pub fn coincident(mut self, p1: usize, p2: usize) -> Self {
        self.system
            .add_constraint(Box::new(CoincidentConstraint::<D>::new(p1, p2)));
        self
    }

    /// Add a fixed position constraint.
    pub fn fixed_at(mut self, point: usize, target: Point<D>) -> Self {
        self.system
            .add_constraint(Box::new(FixedConstraint::<D>::new(point, target)));
        self
    }

    /// Add a midpoint constraint.
    pub fn midpoint(mut self, mid: usize, start: usize, end: usize) -> Self {
        self.system
            .add_constraint(Box::new(MidpointConstraint::<D>::new(mid, start, end)));
        self
    }

    /// Add a perpendicular constraint.
    pub fn perpendicular(
        mut self,
        line1_start: usize,
        line1_end: usize,
        line2_start: usize,
        line2_end: usize,
    ) -> Self {
        self.system.add_constraint(Box::new(
            PerpendicularConstraint::<D>::new(line1_start, line1_end, line2_start, line2_end),
        ));
        self
    }

    /// Add a symmetric (about point) constraint.
    pub fn symmetric(mut self, p1: usize, p2: usize, center: usize) -> Self {
        self.system
            .add_constraint(Box::new(SymmetricConstraint::<D>::new(p1, p2, center)));
        self
    }

    /// Add an equal length constraint.
    pub fn equal_length(
        mut self,
        line1_start: usize,
        line1_end: usize,
        line2_start: usize,
        line2_end: usize,
    ) -> Self {
        self.system.add_constraint(Box::new(EqualLengthConstraint::<D>::new(
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        )));
        self
    }

    /// Add a custom constraint.
    pub fn constraint(mut self, constraint: Box<dyn GeometricConstraint<D>>) -> Self {
        self.system.add_constraint(constraint);
        self
    }

    /// Build the constraint system.
    pub fn build(self) -> ConstraintSystem<D> {
        self.system
    }
}

// 2D-specific builder methods
impl ConstraintSystemBuilder<2> {
    /// Add a horizontal constraint (same y).
    pub fn horizontal(mut self, p1: usize, p2: usize) -> Self {
        self.system
            .add_constraint(Box::new(HorizontalConstraint::new(p1, p2)));
        self
    }

    /// Add a vertical constraint (same x).
    pub fn vertical(mut self, p1: usize, p2: usize) -> Self {
        self.system
            .add_constraint(Box::new(VerticalConstraint::new(p1, p2)));
        self
    }

    /// Add an angle constraint (line angle from horizontal).
    pub fn angle(mut self, line_start: usize, line_end: usize, angle_radians: f64) -> Self {
        self.system
            .add_constraint(Box::new(AngleConstraint::new(line_start, line_end, angle_radians)));
        self
    }

    /// Add an angle constraint in degrees.
    pub fn angle_degrees(mut self, line_start: usize, line_end: usize, degrees: f64) -> Self {
        self.system.add_constraint(Box::new(AngleConstraint::from_degrees(
            line_start, line_end, degrees,
        )));
        self
    }

    /// Add a parallel constraint.
    pub fn parallel(
        mut self,
        line1_start: usize,
        line1_end: usize,
        line2_start: usize,
        line2_end: usize,
    ) -> Self {
        self.system.add_constraint(Box::new(ParallelConstraint::<2>::new(
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        )));
        self
    }

    /// Add a point-on-line constraint.
    pub fn point_on_line(mut self, point: usize, line_start: usize, line_end: usize) -> Self {
        self.system.add_constraint(Box::new(PointOnLineConstraint::<2>::new(
            point, line_start, line_end,
        )));
        self
    }

    /// Add a point-on-circle constraint.
    pub fn point_on_circle(mut self, point: usize, center: usize, radius: f64) -> Self {
        self.system.add_constraint(Box::new(PointOnCircleConstraint::<2>::new(
            point, center, radius,
        )));
        self
    }

    /// Add a collinear constraint.
    pub fn collinear(
        mut self,
        line1_start: usize,
        line1_end: usize,
        line2_start: usize,
        line2_end: usize,
    ) -> Self {
        self.system.add_constraint(Box::new(CollinearConstraint::<2>::new(
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        )));
        self
    }

    /// Add a line-tangent-circle constraint.
    pub fn line_tangent_circle(
        mut self,
        line_start: usize,
        line_end: usize,
        center: usize,
        radius: f64,
    ) -> Self {
        self.system.add_constraint(Box::new(LineTangentConstraint::new(
            line_start, line_end, center, radius,
        )));
        self
    }

    /// Add a circle-tangent-circle constraint (external).
    pub fn circles_tangent_external(
        mut self,
        center1: usize,
        radius1: f64,
        center2: usize,
        radius2: f64,
    ) -> Self {
        self.system.add_constraint(Box::new(CircleTangentConstraint::external(
            center1, radius1, center2, radius2,
        )));
        self
    }

    /// Add a circle-tangent-circle constraint (internal).
    pub fn circles_tangent_internal(
        mut self,
        center1: usize,
        radius1: f64,
        center2: usize,
        radius2: f64,
    ) -> Self {
        self.system.add_constraint(Box::new(CircleTangentConstraint::internal(
            center1, radius1, center2, radius2,
        )));
        self
    }

    /// Add a symmetric-about-line constraint.
    pub fn symmetric_about_line(
        mut self,
        p1: usize,
        p2: usize,
        axis_start: usize,
        axis_end: usize,
    ) -> Self {
        self.system.add_constraint(Box::new(SymmetricAboutLineConstraint::new(
            p1, p2, axis_start, axis_end,
        )));
        self
    }
}

// 3D-specific builder methods
impl ConstraintSystemBuilder<3> {
    /// Add a parallel constraint (3D).
    pub fn parallel(
        mut self,
        line1_start: usize,
        line1_end: usize,
        line2_start: usize,
        line2_end: usize,
    ) -> Self {
        self.system.add_constraint(Box::new(ParallelConstraint::<3>::new(
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        )));
        self
    }

    /// Add a point-on-line constraint (3D).
    pub fn point_on_line(mut self, point: usize, line_start: usize, line_end: usize) -> Self {
        self.system.add_constraint(Box::new(PointOnLineConstraint::<3>::new(
            point, line_start, line_end,
        )));
        self
    }

    /// Add a point-on-sphere constraint (3D).
    pub fn point_on_sphere(mut self, point: usize, center: usize, radius: f64) -> Self {
        self.system.add_constraint(Box::new(PointOnCircleConstraint::<3>::new(
            point, center, radius,
        )));
        self
    }

    /// Add a collinear constraint (3D).
    pub fn collinear(
        mut self,
        line1_start: usize,
        line1_end: usize,
        line2_start: usize,
        line2_end: usize,
    ) -> Self {
        self.system.add_constraint(Box::new(CollinearConstraint::<3>::new(
            line1_start,
            line1_end,
            line2_start,
            line2_end,
        )));
        self
    }

    /// Add a sphere-tangent-sphere constraint (external).
    pub fn spheres_tangent_external(
        mut self,
        center1: usize,
        radius1: f64,
        center2: usize,
        radius2: f64,
    ) -> Self {
        self.system.add_constraint(Box::new(CircleTangentConstraint::external(
            center1, radius1, center2, radius2,
        )));
        self
    }

    /// Add a sphere-tangent-sphere constraint (internal).
    pub fn spheres_tangent_internal(
        mut self,
        center1: usize,
        radius1: f64,
        center2: usize,
        radius2: f64,
    ) -> Self {
        self.system.add_constraint(Box::new(CircleTangentConstraint::internal(
            center1, radius1, center2, radius2,
        )));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::{Point2D, Point3D};

    #[test]
    fn test_builder_basic() {
        let system = ConstraintSystemBuilder::<2>::new()
            .point(Point2D::new(0.0, 0.0))
            .point(Point2D::new(5.0, 0.0))
            .distance(0, 1, 10.0)
            .build();

        assert_eq!(system.point_count(), 2);
        assert_eq!(system.constraint_count(), 1);
    }

    #[test]
    fn test_builder_triangle() {
        let system = ConstraintSystemBuilder::<2>::new()
            .name("Triangle")
            .point(Point2D::new(0.0, 0.0))
            .point(Point2D::new(10.0, 0.0))
            .point(Point2D::new(5.0, 8.0))
            .fix(0)
            .distance(0, 1, 10.0)
            .distance(1, 2, 10.0)
            .distance(2, 0, 10.0)
            .build();

        assert_eq!(system.point_count(), 3);
        assert_eq!(system.constraint_count(), 3);
        assert!(system.is_fixed(0));
        assert!(!system.is_fixed(1));
    }

    #[test]
    fn test_builder_rectangle() {
        let system = ConstraintSystemBuilder::<2>::new()
            .point(Point2D::new(0.0, 0.0))   // p0
            .point(Point2D::new(10.0, 0.0))  // p1
            .point(Point2D::new(10.0, 5.0))  // p2
            .point(Point2D::new(0.0, 5.0))   // p3
            .fix(0)
            .horizontal(0, 1)
            .vertical(1, 2)
            .horizontal(2, 3)
            .vertical(3, 0)
            .equal_length(0, 1, 2, 3) // top = bottom
            .equal_length(1, 2, 3, 0) // right = left
            .build();

        assert_eq!(system.constraint_count(), 6);
    }

    #[test]
    fn test_builder_3d() {
        let system = ConstraintSystemBuilder::<3>::new()
            .point(Point3D::new(0.0, 0.0, 0.0))
            .point(Point3D::new(1.0, 0.0, 0.0))
            .point(Point3D::new(0.0, 1.0, 0.0))
            .point(Point3D::new(0.0, 0.0, 1.0))
            .fix(0)
            .distance(0, 1, 1.0)
            .distance(0, 2, 1.0)
            .distance(0, 3, 1.0)
            .distance(1, 2, 1.0)
            .distance(1, 3, 1.0)
            .distance(2, 3, 1.0)
            .build();

        assert_eq!(system.point_count(), 4);
        assert_eq!(system.constraint_count(), 6);
    }

    #[test]
    fn test_builder_with_custom_constraint() {
        let system = ConstraintSystemBuilder::<2>::new()
            .point(Point2D::new(0.0, 0.0))
            .point(Point2D::new(5.0, 0.0))
            .constraint(Box::new(HorizontalConstraint::new(0, 1)))
            .build();

        assert_eq!(system.constraint_count(), 1);
    }
}
