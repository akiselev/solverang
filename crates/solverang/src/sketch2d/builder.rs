//! Ergonomic builder API for constructing 2D sketch constraint systems.
//!
//! The [`Sketch2DBuilder`] handles parameter allocation, entity creation, and
//! constraint wiring so callers work in terms of geometric objects rather than
//! raw parameter IDs.

use std::collections::HashMap;

use crate::id::{ConstraintId, EntityId, ParamId};
use crate::sketch2d::constraints::*;
use crate::sketch2d::entities::*;
use crate::system::ConstraintSystem;

// ---------------------------------------------------------------------------
// Entity metadata kept by the builder for constraint wiring.
// ---------------------------------------------------------------------------

/// Describes the kind and parameters of an entity added through the builder.
#[derive(Debug, Clone)]
enum EntityKind {
    Point {
        x: ParamId,
        y: ParamId,
    },
    LineSegment {
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
    },
    Circle {
        cx: ParamId,
        cy: ParamId,
        r: ParamId,
    },
}

#[derive(Debug, Clone)]
struct EntityInfo {
    kind: EntityKind,
    params: Vec<ParamId>,
}

// ---------------------------------------------------------------------------
// Sketch2DBuilder
// ---------------------------------------------------------------------------

/// Builder for creating 2D sketch constraint systems.
///
/// # Example
///
/// ```ignore
/// let mut b = Sketch2DBuilder::new();
/// let p0 = b.add_fixed_point(0.0, 0.0);
/// let p1 = b.add_point(10.0, 0.0);
/// let p2 = b.add_point(5.0, 8.0);
/// b.constrain_distance(p0, p1, 10.0);
/// b.constrain_distance(p1, p2, 8.0);
/// b.constrain_distance(p2, p0, 6.0);
/// let system = b.build();
/// ```
pub struct Sketch2DBuilder {
    system: ConstraintSystem,
    entity_info: HashMap<EntityId, EntityInfo>,
}

impl Default for Sketch2DBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Sketch2DBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self {
            system: ConstraintSystem::new(),
            entity_info: HashMap::new(),
        }
    }

    // -- ID allocation helpers (delegated to ConstraintSystem) --

    fn alloc_entity_id(&mut self) -> EntityId {
        self.system.alloc_entity_id()
    }

    fn alloc_constraint_id(&mut self) -> ConstraintId {
        self.system.alloc_constraint_id()
    }

    // -- Entity info lookup helpers --

    fn point_params(&self, entity: EntityId) -> (ParamId, ParamId) {
        match &self.entity_info[&entity].kind {
            EntityKind::Point { x, y } => (*x, *y),
            _ => panic!("Entity {:?} is not a Point2D", entity),
        }
    }

    fn line_params(&self, entity: EntityId) -> (ParamId, ParamId, ParamId, ParamId) {
        match &self.entity_info[&entity].kind {
            EntityKind::LineSegment { x1, y1, x2, y2 } => (*x1, *y1, *x2, *y2),
            _ => panic!("Entity {:?} is not a LineSegment2D", entity),
        }
    }

    fn circle_params(&self, entity: EntityId) -> (ParamId, ParamId, ParamId) {
        match &self.entity_info[&entity].kind {
            EntityKind::Circle { cx, cy, r } => (*cx, *cy, *r),
            _ => panic!("Entity {:?} is not a Circle2D", entity),
        }
    }

    // ======================================================================
    // Entity creation
    // ======================================================================

    /// Add a 2D point with the given initial position.
    pub fn add_point(&mut self, x: f64, y: f64) -> EntityId {
        let eid = self.alloc_entity_id();
        let px = self.system.params_mut().alloc(x, eid);
        let py = self.system.params_mut().alloc(y, eid);
        let entity = Point2D::new(eid, px, py);
        self.system.add_entity(Box::new(entity));
        self.entity_info.insert(
            eid,
            EntityInfo {
                kind: EntityKind::Point { x: px, y: py },
                params: vec![px, py],
            },
        );
        eid
    }

    /// Add a 2D point and immediately fix it (exclude from solving).
    pub fn add_fixed_point(&mut self, x: f64, y: f64) -> EntityId {
        let eid = self.add_point(x, y);
        self.fix_entity(eid);
        eid
    }

    /// Add a circle with the given center and radius.
    pub fn add_circle(&mut self, cx: f64, cy: f64, r: f64) -> EntityId {
        let eid = self.alloc_entity_id();
        let pcx = self.system.params_mut().alloc(cx, eid);
        let pcy = self.system.params_mut().alloc(cy, eid);
        let pr = self.system.params_mut().alloc(r, eid);
        let entity = Circle2D::new(eid, pcx, pcy, pr);
        self.system.add_entity(Box::new(entity));
        self.entity_info.insert(
            eid,
            EntityInfo {
                kind: EntityKind::Circle {
                    cx: pcx,
                    cy: pcy,
                    r: pr,
                },
                params: vec![pcx, pcy, pr],
            },
        );
        eid
    }

    /// Add a line segment between two existing points.
    ///
    /// The line segment shares parameter IDs with the two endpoint points,
    /// so moving a point automatically moves the line endpoint.
    pub fn add_line_segment(&mut self, p1: EntityId, p2: EntityId) -> EntityId {
        let (x1, y1) = self.point_params(p1);
        let (x2, y2) = self.point_params(p2);
        let eid = self.alloc_entity_id();
        let entity = LineSegment2D::new(eid, x1, y1, x2, y2);
        self.system.add_entity(Box::new(entity));
        self.entity_info.insert(
            eid,
            EntityInfo {
                kind: EntityKind::LineSegment { x1, y1, x2, y2 },
                params: vec![x1, y1, x2, y2],
            },
        );
        eid
    }

    // ======================================================================
    // Constraint creation
    // ======================================================================

    /// Constrain the distance between two point entities.
    pub fn constrain_distance(
        &mut self,
        e1: EntityId,
        e2: EntityId,
        distance: f64,
    ) -> ConstraintId {
        let (x1, y1) = self.point_params(e1);
        let (x2, y2) = self.point_params(e2);
        let cid = self.alloc_constraint_id();
        let c = DistancePtPt::new(cid, e1, e2, x1, y1, x2, y2, distance);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain two point entities to be coincident.
    pub fn constrain_coincident(
        &mut self,
        e1: EntityId,
        e2: EntityId,
    ) -> ConstraintId {
        let (x1, y1) = self.point_params(e1);
        let (x2, y2) = self.point_params(e2);
        let cid = self.alloc_constraint_id();
        let c = Coincident::new(cid, e1, e2, x1, y1, x2, y2);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain two point entities to share the same y-coordinate (horizontal).
    pub fn constrain_horizontal(
        &mut self,
        e1: EntityId,
        e2: EntityId,
    ) -> ConstraintId {
        let (_, y1) = self.point_params(e1);
        let (_, y2) = self.point_params(e2);
        let cid = self.alloc_constraint_id();
        let c = Horizontal::new(cid, e1, e2, y1, y2);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain two point entities to share the same x-coordinate (vertical).
    pub fn constrain_vertical(
        &mut self,
        e1: EntityId,
        e2: EntityId,
    ) -> ConstraintId {
        let (x1, _) = self.point_params(e1);
        let (x2, _) = self.point_params(e2);
        let cid = self.alloc_constraint_id();
        let c = Vertical::new(cid, e1, e2, x1, x2);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Fix a point entity at specific coordinates.
    pub fn constrain_fixed(
        &mut self,
        entity: EntityId,
        x: f64,
        y: f64,
    ) -> ConstraintId {
        let (px, py) = self.point_params(entity);
        let cid = self.alloc_constraint_id();
        let c = Fixed::new(cid, entity, px, py, x, y);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain two line segments to be parallel.
    pub fn constrain_parallel(
        &mut self,
        l1: EntityId,
        l2: EntityId,
    ) -> ConstraintId {
        let (x1, y1, x2, y2) = self.line_params(l1);
        let (x3, y3, x4, y4) = self.line_params(l2);
        let cid = self.alloc_constraint_id();
        let c = Parallel::new(cid, l1, l2, x1, y1, x2, y2, x3, y3, x4, y4);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain two line segments to be perpendicular.
    pub fn constrain_perpendicular(
        &mut self,
        l1: EntityId,
        l2: EntityId,
    ) -> ConstraintId {
        let (x1, y1, x2, y2) = self.line_params(l1);
        let (x3, y3, x4, y4) = self.line_params(l2);
        let cid = self.alloc_constraint_id();
        let c = Perpendicular::new(cid, l1, l2, x1, y1, x2, y2, x3, y3, x4, y4);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain a line segment to be tangent to a circle.
    pub fn constrain_tangent_line_circle(
        &mut self,
        line: EntityId,
        circle: EntityId,
    ) -> ConstraintId {
        let (x1, y1, x2, y2) = self.line_params(line);
        let (cx, cy, r) = self.circle_params(circle);
        let cid = self.alloc_constraint_id();
        let c = TangentLineCircle::new(cid, line, circle, x1, y1, x2, y2, cx, cy, r);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain a point entity to lie on a circle entity.
    pub fn constrain_point_on_circle(
        &mut self,
        point: EntityId,
        circle: EntityId,
    ) -> ConstraintId {
        let (px, py) = self.point_params(point);
        let (cx, cy, r) = self.circle_params(circle);
        let cid = self.alloc_constraint_id();
        let c = PointOnCircle::new(cid, point, circle, px, py, cx, cy, r);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain two line segments to have equal length.
    pub fn constrain_equal_length(
        &mut self,
        l1: EntityId,
        l2: EntityId,
    ) -> ConstraintId {
        let (x1, y1, x2, y2) = self.line_params(l1);
        let (x3, y3, x4, y4) = self.line_params(l2);
        let cid = self.alloc_constraint_id();
        let c = EqualLength::new(cid, l1, l2, x1, y1, x2, y2, x3, y3, x4, y4);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain a point entity to be at the midpoint of a line segment entity.
    pub fn constrain_midpoint(
        &mut self,
        point: EntityId,
        line: EntityId,
    ) -> ConstraintId {
        let (mx, my) = self.point_params(point);
        let (x1, y1, x2, y2) = self.line_params(line);
        let cid = self.alloc_constraint_id();
        let c = Midpoint::new(cid, point, line, mx, my, x1, y1, x2, y2);
        self.system.add_constraint(Box::new(c));
        cid
    }

    /// Constrain two points to be symmetric about a center point.
    pub fn constrain_symmetric(
        &mut self,
        p1: EntityId,
        p2: EntityId,
        center: EntityId,
    ) -> ConstraintId {
        let (x1, y1) = self.point_params(p1);
        let (x2, y2) = self.point_params(p2);
        let (cx, cy) = self.point_params(center);
        let cid = self.alloc_constraint_id();
        let c = Symmetric::new(cid, p1, p2, center, x1, y1, x2, y2, cx, cy);
        self.system.add_constraint(Box::new(c));
        cid
    }

    // ======================================================================
    // Fixing parameters / entities
    // ======================================================================

    /// Fix a specific parameter (exclude from solving).
    pub fn fix_param(&mut self, param: ParamId) {
        self.system.params_mut().fix(param);
    }

    /// Fix all parameters of an entity (exclude from solving).
    pub fn fix_entity(&mut self, entity: EntityId) {
        let info = self
            .entity_info
            .get(&entity)
            .expect("fix_entity: unknown EntityId");
        for &pid in &info.params {
            self.system.params_mut().fix(pid);
        }
    }

    // ======================================================================
    // System access
    // ======================================================================

    /// Get the built system (consumes builder).
    pub fn build(self) -> ConstraintSystem {
        self.system
    }

    /// Get a reference to the underlying system.
    pub fn system(&self) -> &ConstraintSystem {
        &self.system
    }

    /// Get a mutable reference to the underlying system.
    pub fn system_mut(&mut self) -> &mut ConstraintSystem {
        &mut self.system
    }

    /// Look up the entity info for an entity added through this builder.
    pub fn entity_param_ids(&self, entity: EntityId) -> &[ParamId] {
        &self.entity_info[&entity].params
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_add_point() {
        let mut b = Sketch2DBuilder::new();
        let p = b.add_point(3.0, 4.0);

        let sys = b.build();
        assert_eq!(sys.entity_count(), 1);
        assert_eq!(sys.params().alive_count(), 2);
        assert_eq!(sys.params().free_param_count(), 2);

        // Verify param values
        let ids: Vec<ParamId> = sys.params().alive_param_ids().collect();
        assert_eq!(ids.len(), 2);
        // The entity id should be valid
        assert_eq!(p.raw_index(), 0);
    }

    #[test]
    fn test_builder_add_fixed_point() {
        let mut b = Sketch2DBuilder::new();
        let _p = b.add_fixed_point(1.0, 2.0);
        let sys = b.build();
        assert_eq!(sys.params().alive_count(), 2);
        assert_eq!(sys.params().free_param_count(), 0);
    }

    #[test]
    fn test_builder_add_circle() {
        let mut b = Sketch2DBuilder::new();
        let _c = b.add_circle(0.0, 0.0, 5.0);
        let sys = b.build();
        assert_eq!(sys.entity_count(), 1);
        assert_eq!(sys.params().alive_count(), 3);
    }

    #[test]
    fn test_builder_add_line_segment() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(0.0, 0.0);
        let p1 = b.add_point(10.0, 0.0);
        let _l = b.add_line_segment(p0, p1);

        let sys = b.build();
        assert_eq!(sys.entity_count(), 3); // 2 points + 1 line
        // Line shares params with points, so still 4 params total
        assert_eq!(sys.params().alive_count(), 4);
    }

    #[test]
    fn test_builder_constrain_distance() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(0.0, 0.0);
        let p1 = b.add_point(3.0, 4.0);
        let _c = b.constrain_distance(p0, p1, 5.0);

        let sys = b.build();
        assert_eq!(sys.constraint_count(), 1);

        // Residual should be 0 since dist(0,0 to 3,4) = 5
        let residuals = sys.compute_residuals();
        assert!(residuals[0].abs() < 1e-12);
    }

    #[test]
    fn test_builder_constrain_coincident() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(3.0, 4.0);
        let p1 = b.add_point(3.0, 4.0);
        let _c = b.constrain_coincident(p0, p1);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-15);
        assert!(r[1].abs() < 1e-15);
    }

    #[test]
    fn test_builder_constrain_horizontal() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(0.0, 5.0);
        let p1 = b.add_point(10.0, 5.0);
        let _c = b.constrain_horizontal(p0, p1);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-15);
    }

    #[test]
    fn test_builder_constrain_vertical() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(5.0, 0.0);
        let p1 = b.add_point(5.0, 10.0);
        let _c = b.constrain_vertical(p0, p1);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-15);
    }

    #[test]
    fn test_builder_constrain_fixed() {
        let mut b = Sketch2DBuilder::new();
        let p = b.add_point(3.0, 4.0);
        let _c = b.constrain_fixed(p, 3.0, 4.0);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-15);
        assert!(r[1].abs() < 1e-15);
    }

    #[test]
    fn test_builder_constrain_parallel() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(0.0, 0.0);
        let p1 = b.add_point(1.0, 2.0);
        let p2 = b.add_point(3.0, 1.0);
        let p3 = b.add_point(5.0, 5.0); // dir = (2,4), parallel to (1,2)
        let l1 = b.add_line_segment(p0, p1);
        let l2 = b.add_line_segment(p2, p3);
        let _c = b.constrain_parallel(l1, l2);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-12);
    }

    #[test]
    fn test_builder_constrain_perpendicular() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(0.0, 0.0);
        let p1 = b.add_point(1.0, 0.0);
        let p2 = b.add_point(0.0, 0.0);
        let p3 = b.add_point(0.0, 1.0);
        let l1 = b.add_line_segment(p0, p1);
        let l2 = b.add_line_segment(p2, p3);
        let _c = b.constrain_perpendicular(l1, l2);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-12);
    }

    #[test]
    fn test_builder_constrain_point_on_circle() {
        let mut b = Sketch2DBuilder::new();
        let p = b.add_point(3.0, 4.0);
        let c = b.add_circle(0.0, 0.0, 5.0);
        let _cid = b.constrain_point_on_circle(p, c);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-12);
    }

    #[test]
    fn test_builder_constrain_equal_length() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(0.0, 0.0);
        let p1 = b.add_point(3.0, 4.0); // length 5
        let p2 = b.add_point(1.0, 1.0);
        let p3 = b.add_point(4.0, 5.0); // length 5
        let l1 = b.add_line_segment(p0, p1);
        let l2 = b.add_line_segment(p2, p3);
        let _c = b.constrain_equal_length(l1, l2);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-12);
    }

    #[test]
    fn test_builder_constrain_midpoint() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(0.0, 0.0);
        let p1 = b.add_point(10.0, 6.0);
        let mid = b.add_point(5.0, 3.0);
        let l = b.add_line_segment(p0, p1);
        let _c = b.constrain_midpoint(mid, l);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-12);
        assert!(r[1].abs() < 1e-12);
    }

    #[test]
    fn test_builder_constrain_symmetric() {
        let mut b = Sketch2DBuilder::new();
        let p1 = b.add_point(1.0, 2.0);
        let p2 = b.add_point(5.0, 8.0);
        let center = b.add_point(3.0, 5.0);
        let _c = b.constrain_symmetric(p1, p2, center);

        let sys = b.build();
        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-12);
        assert!(r[1].abs() < 1e-12);
    }

    #[test]
    fn test_builder_fix_param() {
        let mut b = Sketch2DBuilder::new();
        let p = b.add_point(1.0, 2.0);
        let params = b.entity_param_ids(p).to_vec();
        b.fix_param(params[0]);

        let sys = b.build();
        assert!(sys.params().is_fixed(params[0]));
        assert!(!sys.params().is_fixed(params[1]));
    }

    #[test]
    fn test_builder_fix_entity() {
        let mut b = Sketch2DBuilder::new();
        let p = b.add_point(1.0, 2.0);
        b.fix_entity(p);

        let sys = b.build();
        assert_eq!(sys.params().free_param_count(), 0);
    }

    #[test]
    fn test_builder_triangle() {
        // Build a triangle with 3 points, 3 distance constraints, fix one point.
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_fixed_point(0.0, 0.0);
        let p1 = b.add_point(10.0, 0.0);
        let p2 = b.add_point(5.0, 1.0);

        b.constrain_distance(p0, p1, 10.0);
        b.constrain_distance(p1, p2, 8.0);
        b.constrain_distance(p2, p0, 6.0);

        let sys = b.build();
        assert_eq!(sys.entity_count(), 3);
        assert_eq!(sys.constraint_count(), 3);
        assert_eq!(sys.params().alive_count(), 6);
        assert_eq!(sys.params().free_param_count(), 4);
        // 3 equations, 4 free params => 1 DOF (rotation)
        assert_eq!(sys.equation_count(), 3);
    }

    #[test]
    fn test_builder_equation_and_param_count() {
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_fixed_point(0.0, 0.0);
        let p1 = b.add_point(3.0, 4.0);
        b.constrain_distance(p0, p1, 5.0);

        let sys = b.build();
        assert_eq!(sys.equation_count(), 1);
        assert_eq!(sys.params().free_param_count(), 2);

        let r = sys.compute_residuals();
        assert!(r[0].abs() < 1e-12); // Already satisfied
    }
}
