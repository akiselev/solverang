//! Fluent builder API for constructing constraint systems (v2).
//!
//! This module provides a convenient builder pattern for creating geometric constraint
//! systems using the new parameter-centric v2 architecture. Entities (points, lines,
//! circles, arcs, bezier curves, etc.) and constraints can be added through method chaining.
//!
//! # Example
//!
//! ```rust,ignore
//! use solverang::geometry::ConstraintSystem;
//!
//! let system = ConstraintSystem::builder()
//!     .name("triangle")
//!     .point_2d_fixed(0.0, 0.0)          // p0, fixed origin
//!     .point_2d(10.0, 0.0)               // p1
//!     .point_2d(5.0, 8.0)                // p2
//!     .distance(0, 1, 10.0)              // p0-p1 distance = 10
//!     .distance(1, 2, 8.0)               // p1-p2 distance = 8
//!     .distance(2, 0, 6.0)               // p2-p0 distance = 6
//!     .build();
//! ```

use super::params::{EntityHandle, ParamRange};
use super::entity::EntityKind;
use super::system::ConstraintSystem;
use super::constraint::Constraint;
use super::constraints::*;

/// Builder for fluent construction of constraint systems.
///
/// The builder tracks created entities in order, allowing constraints to reference
/// entities by their creation index (starting from 0).
pub struct ConstraintSystemBuilder {
    /// The constraint system being built.
    system: ConstraintSystem,
    /// Handles of entities in creation order for constraint reference.
    handles: Vec<EntityHandle>,
}

impl ConstraintSystemBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            system: ConstraintSystem::new(),
            handles: Vec::new(),
        }
    }

    /// Set a name for this constraint system (for debugging/diagnostics).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.system.set_name(name);
        self
    }

    // ============================================================================
    // Entity Creation Methods
    // ============================================================================

    /// Add a 2D point entity.
    pub fn point_2d(mut self, x: f64, y: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Point2D, &[x, y]);
        self.handles.push(handle);
        self
    }

    /// Add a 2D point entity and immediately fix it.
    pub fn point_2d_fixed(mut self, x: f64, y: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Point2D, &[x, y]);
        self.system.fix_entity(&handle);
        self.handles.push(handle);
        self
    }

    /// Add a 3D point entity.
    pub fn point_3d(mut self, x: f64, y: f64, z: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Point3D, &[x, y, z]);
        self.handles.push(handle);
        self
    }

    /// Add a 3D point entity and immediately fix it.
    pub fn point_3d_fixed(mut self, x: f64, y: f64, z: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Point3D, &[x, y, z]);
        self.system.fix_entity(&handle);
        self.handles.push(handle);
        self
    }

    /// Add a 2D circle entity: center (cx, cy) and radius r.
    pub fn circle_2d(mut self, cx: f64, cy: f64, r: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Circle2D, &[cx, cy, r]);
        self.handles.push(handle);
        self
    }

    /// Add a 2D line entity: endpoints (x1, y1) to (x2, y2).
    pub fn line_2d(mut self, x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Line2D, &[x1, y1, x2, y2]);
        self.handles.push(handle);
        self
    }

    /// Add a 2D arc entity: center (cx, cy), radius r, start and end angles (radians).
    pub fn arc_2d(mut self, cx: f64, cy: f64, r: f64, start: f64, end: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Arc2D, &[cx, cy, r, start, end]);
        self.handles.push(handle);
        self
    }

    /// Add a 2D cubic Bezier curve entity: 4 control points [[x0,y0], [x1,y1], [x2,y2], [x3,y3]].
    pub fn cubic_bezier_2d(mut self, points: [[f64; 2]; 4]) -> Self {
        let flat: Vec<f64> = points.iter().flatten().copied().collect();
        let handle = self.system.add_entity(EntityKind::CubicBezier2D, &flat);
        self.handles.push(handle);
        self
    }

    /// Add a 2D ellipse entity: center (cx, cy), radii (rx, ry), rotation angle (radians).
    pub fn ellipse_2d(mut self, cx: f64, cy: f64, rx: f64, ry: f64, rot: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Ellipse2D, &[cx, cy, rx, ry, rot]);
        self.handles.push(handle);
        self
    }

    /// Add a scalar parameter entity (single value).
    pub fn scalar(mut self, value: f64) -> Self {
        let handle = self.system.add_entity(EntityKind::Scalar, &[value]);
        self.handles.push(handle);
        self
    }

    /// Add a generic entity with specified kind and values.
    pub fn entity(mut self, kind: EntityKind, values: &[f64]) -> Self {
        let handle = self.system.add_entity(kind, values);
        self.handles.push(handle);
        self
    }

    /// Fix an entity by its builder index (all parameters become fixed/driven).
    pub fn fix(mut self, entity_index: usize) -> Self {
        let handle = self.handles[entity_index];
        self.system.fix_entity(&handle);
        self
    }

    /// Fix a specific parameter of an entity by builder index and parameter offset.
    ///
    /// For example, to fix just the radius of a circle (param offset 2):
    /// ```ignore
    /// builder.circle_2d(5.0, 5.0, 3.0).fix_param_at(0, 2)
    /// ```
    pub fn fix_param_at(mut self, entity_index: usize, param_offset: usize) -> Self {
        let handle = self.handles[entity_index];
        let param_idx = handle.params.start + param_offset;
        self.system.fix_param(param_idx);
        self
    }

    // ============================================================================
    // Constraint Creation Methods
    // ============================================================================

    /// Distance between two point entities must equal target.
    ///
    /// # Arguments
    /// * `e1` - Index of first point entity
    /// * `e2` - Index of second point entity
    /// * `target` - Target distance
    pub fn distance(mut self, e1: usize, e2: usize, target: f64) -> Self {
        let id = self.system.next_constraint_id();
        let p1 = self.param_range(e1);
        let p2 = self.param_range(e2);
        self.system.add_constraint(Box::new(
            DistanceConstraint::new(id, p1, p2, target)
        ));
        self
    }

    /// Two point entities must coincide (all coordinates equal).
    pub fn coincident(mut self, e1: usize, e2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let p1 = self.param_range(e1);
        let p2 = self.param_range(e2);
        self.system.add_constraint(Box::new(
            CoincidentConstraint::new(id, p1, p2)
        ));
        self
    }

    /// Fix an entity to specific values.
    ///
    /// # Arguments
    /// * `entity` - Index of entity to fix
    /// * `values` - Target values for all parameters of this entity
    pub fn fixed_at(mut self, entity: usize, values: &[f64]) -> Self {
        let id = self.system.next_constraint_id();
        let params = self.param_range(entity);
        self.system.add_constraint(Box::new(
            FixedConstraint::new(id, params, values.to_vec())
        ));
        self
    }

    /// Two 2D points must have the same y-coordinate (horizontal alignment).
    pub fn horizontal(mut self, p1: usize, p2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let r1 = self.param_range(p1);
        let r2 = self.param_range(p2);
        self.system.add_constraint(Box::new(
            HorizontalConstraint::new(id, r1, r2)
        ));
        self
    }

    /// Two 2D points must have the same x-coordinate (vertical alignment).
    pub fn vertical(mut self, p1: usize, p2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let r1 = self.param_range(p1);
        let r2 = self.param_range(p2);
        self.system.add_constraint(Box::new(
            VerticalConstraint::new(id, r1, r2)
        ));
        self
    }

    /// A point must be at the midpoint between two other points.
    ///
    /// # Arguments
    /// * `mid` - Index of midpoint entity
    /// * `start` - Index of start point entity
    /// * `end` - Index of end point entity
    pub fn midpoint(mut self, mid: usize, start: usize, end: usize) -> Self {
        let id = self.system.next_constraint_id();
        let m = self.param_range(mid);
        let s = self.param_range(start);
        let e = self.param_range(end);
        self.system.add_constraint(Box::new(
            MidpointConstraint::new(id, m, s, e)
        ));
        self
    }

    /// Two points are symmetric about a center point.
    ///
    /// # Arguments
    /// * `p1` - Index of first point
    /// * `p2` - Index of second point
    /// * `center` - Index of center point
    pub fn symmetric(mut self, p1: usize, p2: usize, center: usize) -> Self {
        let id = self.system.next_constraint_id();
        let r1 = self.param_range(p1);
        let r2 = self.param_range(p2);
        let rc = self.param_range(center);
        self.system.add_constraint(Box::new(
            SymmetricConstraint::new(id, r1, r2, rc)
        ));
        self
    }

    /// Two line entities must be parallel.
    pub fn parallel(mut self, line1: usize, line2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let l1 = self.param_range(line1);
        let l2 = self.param_range(line2);
        self.system.add_constraint(Box::new(
            ParallelConstraint::new(id, l1, l2)
        ));
        self
    }

    /// Two line entities must be perpendicular.
    pub fn perpendicular(mut self, line1: usize, line2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let l1 = self.param_range(line1);
        let l2 = self.param_range(line2);
        self.system.add_constraint(Box::new(
            PerpendicularConstraint::new(id, l1, l2)
        ));
        self
    }

    /// Two line entities must have equal length.
    ///
    /// Both entities must be Line2D (4 params) or Line3D (6 params).
    /// For point-defined segments, use [`equal_length_segments`](Self::equal_length_segments).
    pub fn equal_length(mut self, line1: usize, line2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let l1 = self.param_range(line1);
        let l2 = self.param_range(line2);
        self.system.add_constraint(Box::new(
            EqualLengthConstraint::new(id, l1, l2)
        ));
        self
    }

    /// Two segments defined by point pairs must have equal length.
    ///
    /// Segment 1 goes from entity `p1` to entity `p2`, and segment 2 goes from
    /// entity `p3` to entity `p4`. All four entities must be points of the same
    /// dimension (2D or 3D).
    ///
    /// # Arguments
    /// * `p1` - Start point of segment 1
    /// * `p2` - End point of segment 1
    /// * `p3` - Start point of segment 2
    /// * `p4` - End point of segment 2
    pub fn equal_length_segments(mut self, p1: usize, p2: usize, p3: usize, p4: usize) -> Self {
        let id = self.system.next_constraint_id();
        let r1 = self.param_range(p1);
        let r2 = self.param_range(p2);
        let r3 = self.param_range(p3);
        let r4 = self.param_range(p4);
        self.system.add_constraint(Box::new(
            EqualLengthConstraint::from_points(id, r1, r2, r3, r4)
        ));
        self
    }

    /// A line entity must be at a specific angle (radians) from horizontal.
    pub fn angle(mut self, line: usize, angle_radians: f64) -> Self {
        let id = self.system.next_constraint_id();
        let l = self.param_range(line);
        // Line has 4 params: [x1, y1, x2, y2] — split into two 2D point ranges
        let p1 = ParamRange { start: l.start, count: 2 };
        let p2 = ParamRange { start: l.start + 2, count: 2 };
        self.system.add_constraint(Box::new(
            AngleConstraint::new(id, p1, p2, angle_radians)
        ));
        self
    }

    /// A point entity must lie on a line entity.
    pub fn point_on_line(mut self, point: usize, line: usize) -> Self {
        let id = self.system.next_constraint_id();
        let p = self.param_range(point);
        let l = self.param_range(line);
        // Line has params: [x1, y1, ...] — split into start/end point ranges
        let dim = p.count; // 2 for 2D, 3 for 3D
        let line_start = ParamRange { start: l.start, count: dim };
        let line_end = ParamRange { start: l.start + dim, count: dim };
        self.system.add_constraint(Box::new(
            PointOnLineConstraint::new(id, p, line_start, line_end)
        ));
        self
    }

    /// A point entity must lie on a circle entity.
    pub fn point_on_circle(mut self, point: usize, circle: usize) -> Self {
        let id = self.system.next_constraint_id();
        let p = self.param_range(point);
        let c = self.param_range(circle);
        self.system.add_constraint(Box::new(
            PointOnCircleConstraint::new(id, p, c)
        ));
        self
    }

    /// A line entity must be tangent to a circle entity.
    pub fn tangent_line_circle(mut self, line: usize, circle: usize) -> Self {
        let id = self.system.next_constraint_id();
        let l = self.param_range(line);
        let c = self.param_range(circle);
        self.system.add_constraint(Box::new(
            LineTangentCircleConstraint::new(id, l, c)
        ));
        self
    }

    /// Two circle entities must be tangent to each other.
    ///
    /// # Arguments
    /// * `c1` - Index of first circle
    /// * `c2` - Index of second circle
    /// * `external` - If true, circles are externally tangent; if false, internally tangent
    pub fn tangent_circles(mut self, c1: usize, c2: usize, external: bool) -> Self {
        let id = self.system.next_constraint_id();
        let r1 = self.param_range(c1);
        let r2 = self.param_range(c2);
        self.system.add_constraint(Box::new(
            CircleTangentConstraint::new(id, r1, r2, external)
        ));
        self
    }

    /// Two circle entities must have equal radii.
    pub fn equal_radius(mut self, c1: usize, c2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let r1 = self.param_range(c1);
        let r2 = self.param_range(c2);
        // Circle2D is [cx, cy, r] — radius is at offset 2
        let r1_idx = r1.start + 2;
        let r2_idx = r2.start + 2;
        self.system.add_constraint(Box::new(
            EqualRadiusConstraint::new(id, r1_idx, r2_idx)
        ));
        self
    }

    /// Two circle entities must have the same center (concentric).
    pub fn concentric(mut self, c1: usize, c2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let r1 = self.param_range(c1);
        let r2 = self.param_range(c2);
        // Circle2D is [cx, cy, r] — center is first 2 params
        let center1 = ParamRange { start: r1.start, count: 2 };
        let center2 = ParamRange { start: r2.start, count: 2 };
        self.system.add_constraint(Box::new(
            ConcentricConstraint::new(id, center1, center2)
        ));
        self
    }

    /// G0 continuity (position continuity) between two curve entities.
    ///
    /// The end of curve1 must coincide with the start of curve2.
    pub fn g0_continuity(mut self, curve1: usize, curve2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let c1 = self.param_range(curve1);
        let c2 = self.param_range(curve2);
        // End of curve1: last control point (2 params), start of curve2: first control point
        let end1 = ParamRange { start: c1.start + c1.count - 2, count: 2 };
        let start2 = ParamRange { start: c2.start, count: 2 };
        self.system.add_constraint(Box::new(
            G0ContinuityConstraint::new(id, end1, start2)
        ));
        self
    }

    /// G1 continuity (tangent continuity) between two curve entities.
    ///
    /// The curves must be G0 continuous and have collinear tangent vectors at the junction.
    pub fn g1_continuity(mut self, curve1: usize, curve2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let c1 = self.param_range(curve1);
        let c2 = self.param_range(curve2);
        self.system.add_constraint(Box::new(
            G1ContinuityConstraint::new(id, c1, c2)
        ));
        self
    }

    /// G2 continuity (curvature continuity) between two curve entities.
    ///
    /// The curves must be G1 continuous and have equal curvature at the junction.
    pub fn g2_continuity(mut self, curve1: usize, curve2: usize) -> Self {
        let id = self.system.next_constraint_id();
        let c1 = self.param_range(curve1);
        let c2 = self.param_range(curve2);
        self.system.add_constraint(Box::new(
            G2ContinuityConstraint::new(id, c1, c2)
        ));
        self
    }

    /// Add a custom constraint to the system.
    pub fn constraint(mut self, constraint: Box<dyn Constraint>) -> Self {
        self.system.add_constraint(constraint);
        self
    }

    // ============================================================================
    // Helper Methods
    // ============================================================================

    /// Get the entity handle for a given builder index.
    #[inline]
    fn handle(&self, idx: usize) -> EntityHandle {
        self.handles[idx]
    }

    /// Get the parameter range for a given builder index.
    #[inline]
    fn param_range(&self, idx: usize) -> ParamRange {
        self.handles[idx].params
    }

    // ============================================================================
    // Build & Inspection
    // ============================================================================

    /// Build and return the constraint system.
    pub fn build(self) -> ConstraintSystem {
        self.system
    }

    /// Get the list of entity handles created so far (for inspection).
    pub fn handles(&self) -> &[EntityHandle] {
        &self.handles
    }
}

impl Default for ConstraintSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_triangle() {
        // Build a triangle with fixed origin and two distance constraints
        let system = ConstraintSystemBuilder::new()
            .name("triangle")
            .point_2d_fixed(0.0, 0.0)    // p0
            .point_2d(10.0, 0.0)         // p1
            .point_2d(5.0, 8.0)          // p2
            .distance(0, 1, 10.0)
            .distance(1, 2, 8.0)
            .distance(2, 0, 6.0)
            .build();

        // Verify entities
        assert_eq!(system.entity_count(), 3);
        // Verify constraints (3 distance constraints)
        assert_eq!(system.constraint_count(), 3);
    }

    #[test]
    fn test_builder_rectangle() {
        // Build a rectangle with horizontal/vertical constraints
        let system = ConstraintSystemBuilder::new()
            .name("rectangle")
            .point_2d_fixed(0.0, 0.0)    // p0: bottom-left, fixed
            .point_2d(10.0, 0.0)         // p1: bottom-right
            .point_2d(10.0, 5.0)         // p2: top-right
            .point_2d(0.0, 5.0)          // p3: top-left
            .horizontal(0, 1)            // bottom edge
            .vertical(1, 2)              // right edge
            .horizontal(2, 3)            // top edge
            .vertical(3, 0)              // left edge
            .equal_length_segments(0, 1, 2, 3) // opposite sides equal: |p0→p1| = |p2→p3|
            .build();

        assert_eq!(system.entity_count(), 4);
        assert_eq!(system.constraint_count(), 5);
    }

    #[test]
    fn test_builder_circle_tangent() {
        // Circle with a tangent line
        let system = ConstraintSystemBuilder::new()
            .name("tangent_test")
            .circle_2d(5.0, 5.0, 3.0)     // circle at (5,5) radius 3
            .line_2d(0.0, 0.0, 10.0, 0.0) // horizontal line
            .tangent_line_circle(1, 0)
            .build();

        assert_eq!(system.entity_count(), 2);
        assert_eq!(system.constraint_count(), 1);
    }

    #[test]
    fn test_builder_fix_specific_param() {
        // Create a circle and fix only its radius
        let system = ConstraintSystemBuilder::new()
            .circle_2d(5.0, 5.0, 3.0)
            .fix_param_at(0, 2)  // Fix radius (param offset 2 in [cx, cy, r])
            .build();

        assert_eq!(system.entity_count(), 1);
        // Check that the radius is fixed
        let handle = &system.handles()[0];
        let radius_idx = handle.params.start + 2;
        assert!(system.is_param_fixed(radius_idx));
    }

    #[test]
    fn test_builder_mixed_entities() {
        // Mix points, lines, and circles
        let system = ConstraintSystemBuilder::new()
            .name("mixed")
            .point_2d(0.0, 0.0)          // 0
            .point_2d(10.0, 0.0)         // 1
            .line_2d(0.0, 0.0, 5.0, 5.0) // 2
            .circle_2d(5.0, 5.0, 2.0)    // 3
            .point_on_line(0, 2)
            .point_on_circle(1, 3)
            .build();

        assert_eq!(system.entity_count(), 4);
        assert_eq!(system.constraint_count(), 2);
    }

    #[test]
    fn test_builder_bezier_curve() {
        // Create a cubic Bezier curve
        let system = ConstraintSystemBuilder::new()
            .cubic_bezier_2d([
                [0.0, 0.0],
                [2.0, 4.0],
                [8.0, 4.0],
                [10.0, 0.0],
            ])
            .build();

        assert_eq!(system.entity_count(), 1);
        let handle = &system.handles()[0];
        assert_eq!(handle.kind, EntityKind::CubicBezier2D);
        assert_eq!(handle.params.count, 8); // 4 points × 2 coords
    }

    #[test]
    fn test_builder_continuity() {
        // Two bezier curves with G1 continuity
        let system = ConstraintSystemBuilder::new()
            .cubic_bezier_2d([
                [0.0, 0.0],
                [2.0, 4.0],
                [8.0, 4.0],
                [10.0, 0.0],
            ])
            .cubic_bezier_2d([
                [10.0, 0.0],
                [12.0, -4.0],
                [18.0, -4.0],
                [20.0, 0.0],
            ])
            .g1_continuity(0, 1)
            .build();

        assert_eq!(system.entity_count(), 2);
        assert_eq!(system.constraint_count(), 1);
    }

    #[test]
    fn test_builder_handles_access() {
        let builder = ConstraintSystemBuilder::new()
            .point_2d(1.0, 2.0)
            .circle_2d(3.0, 4.0, 5.0)
            .line_2d(0.0, 0.0, 1.0, 1.0);

        let handles = builder.handles();
        assert_eq!(handles.len(), 3);
        assert_eq!(handles[0].kind, EntityKind::Point2D);
        assert_eq!(handles[1].kind, EntityKind::Circle2D);
        assert_eq!(handles[2].kind, EntityKind::Line2D);
    }

    #[test]
    fn test_builder_default() {
        let builder = ConstraintSystemBuilder::default();
        let system = builder.build();
        assert_eq!(system.entity_count(), 0);
        assert_eq!(system.constraint_count(), 0);
    }

    #[test]
    fn test_builder_ellipse_and_arc() {
        let system = ConstraintSystemBuilder::new()
            .ellipse_2d(5.0, 5.0, 3.0, 2.0, 0.0)  // Ellipse at (5,5), rx=3, ry=2
            .arc_2d(10.0, 10.0, 4.0, 0.0, 1.57)   // Arc: quarter circle
            .build();

        assert_eq!(system.entity_count(), 2);
        let handles = system.handles();
        assert_eq!(handles[0].kind, EntityKind::Ellipse2D);
        assert_eq!(handles[1].kind, EntityKind::Arc2D);
    }

    #[test]
    fn test_builder_scalar() {
        let system = ConstraintSystemBuilder::new()
            .scalar(42.0)
            .scalar(3.14)
            .build();

        assert_eq!(system.entity_count(), 2);
        let handles = system.handles();
        assert_eq!(handles[0].kind, EntityKind::Scalar);
        assert_eq!(handles[1].kind, EntityKind::Scalar);
    }

    #[test]
    fn test_builder_3d_points() {
        let system = ConstraintSystemBuilder::new()
            .point_3d_fixed(0.0, 0.0, 0.0)
            .point_3d(1.0, 2.0, 3.0)
            .distance(0, 1, 5.0)
            .build();

        assert_eq!(system.entity_count(), 2);
        assert_eq!(system.constraint_count(), 1);
        let handles = system.handles();
        assert_eq!(handles[0].kind, EntityKind::Point3D);
        assert_eq!(handles[1].kind, EntityKind::Point3D);
    }

    #[test]
    fn test_builder_fix_entity() {
        let system = ConstraintSystemBuilder::new()
            .point_2d(5.0, 10.0)
            .circle_2d(0.0, 0.0, 3.0)
            .fix(0)  // Fix the point
            .fix(1)  // Fix the circle
            .build();

        assert_eq!(system.entity_count(), 2);

        // Check that point params are fixed
        let point_handle = &system.handles()[0];
        for i in 0..point_handle.params.count {
            let param_idx = point_handle.params.start + i;
            assert!(system.is_param_fixed(param_idx));
        }

        // Check that circle params are fixed
        let circle_handle = &system.handles()[1];
        for i in 0..circle_handle.params.count {
            let param_idx = circle_handle.params.start + i;
            assert!(system.is_param_fixed(param_idx));
        }
    }

    #[test]
    fn test_builder_equal_radius() {
        let system = ConstraintSystemBuilder::new()
            .circle_2d(0.0, 0.0, 5.0)
            .circle_2d(10.0, 10.0, 3.0)
            .equal_radius(0, 1)
            .build();

        assert_eq!(system.entity_count(), 2);
        assert_eq!(system.constraint_count(), 1);
    }

    #[test]
    fn test_builder_concentric() {
        let system = ConstraintSystemBuilder::new()
            .circle_2d(5.0, 5.0, 3.0)
            .circle_2d(5.1, 4.9, 5.0)
            .concentric(0, 1)
            .build();

        assert_eq!(system.entity_count(), 2);
        assert_eq!(system.constraint_count(), 1);
    }

    #[test]
    fn test_builder_tangent_circles() {
        let system = ConstraintSystemBuilder::new()
            .circle_2d(0.0, 0.0, 3.0)
            .circle_2d(10.0, 0.0, 4.0)
            .tangent_circles(0, 1, true)  // external tangency
            .build();

        assert_eq!(system.entity_count(), 2);
        assert_eq!(system.constraint_count(), 1);
    }
}
