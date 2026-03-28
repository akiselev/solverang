//! 2D entity types for the sketch constraint system.
//!
//! Each entity represents a geometric object in 2D space, defined by parameters
//! stored in the [`ParamStore`]. Entities implement the [`Entity`] trait so the
//! solver treats them uniformly as groups of parameters.

use crate::entity::Entity;
use crate::id::{EntityId, ParamId};
use crate::param::ParamStore;

// ---------------------------------------------------------------------------
// Point2D
// ---------------------------------------------------------------------------

/// A 2D point entity with parameters `[x, y]`.
#[derive(Debug, Clone)]
pub struct Point2D {
    id: EntityId,
    x: ParamId,
    y: ParamId,
    params: [ParamId; 2],
}

impl Point2D {
    /// Create a new 2D point entity.
    pub fn new(id: EntityId, x: ParamId, y: ParamId) -> Self {
        Self {
            id,
            x,
            y,
            params: [x, y],
        }
    }

    /// Parameter ID for the x-coordinate.
    pub fn x(&self) -> ParamId {
        self.x
    }

    /// Parameter ID for the y-coordinate.
    pub fn y(&self) -> ParamId {
        self.y
    }

    /// Read the current x-coordinate value from the store.
    pub fn get_x(&self, store: &ParamStore) -> f64 {
        store.get(self.x)
    }

    /// Read the current y-coordinate value from the store.
    pub fn get_y(&self, store: &ParamStore) -> f64 {
        store.get(self.y)
    }
}

impl Entity for Point2D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "Point2D"
    }
}

// ---------------------------------------------------------------------------
// LineSegment2D
// ---------------------------------------------------------------------------

/// A 2D line segment entity defined by two endpoints.
///
/// Parameters: `[x1, y1, x2, y2]` where `(x1, y1)` is the start point and
/// `(x2, y2)` is the end point. These parameters are typically shared with
/// the corresponding [`Point2D`] entities.
#[derive(Debug, Clone)]
pub struct LineSegment2D {
    id: EntityId,
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    params: [ParamId; 4],
}

impl LineSegment2D {
    /// Create a new 2D line segment entity.
    pub fn new(id: EntityId, x1: ParamId, y1: ParamId, x2: ParamId, y2: ParamId) -> Self {
        Self {
            id,
            x1,
            y1,
            x2,
            y2,
            params: [x1, y1, x2, y2],
        }
    }

    /// Parameter ID for the start point x-coordinate.
    pub fn start_x(&self) -> ParamId {
        self.x1
    }

    /// Parameter ID for the start point y-coordinate.
    pub fn start_y(&self) -> ParamId {
        self.y1
    }

    /// Parameter ID for the end point x-coordinate.
    pub fn end_x(&self) -> ParamId {
        self.x2
    }

    /// Parameter ID for the end point y-coordinate.
    pub fn end_y(&self) -> ParamId {
        self.y2
    }

    /// Read start point coordinates from the store.
    pub fn get_start(&self, store: &ParamStore) -> (f64, f64) {
        (store.get(self.x1), store.get(self.y1))
    }

    /// Read end point coordinates from the store.
    pub fn get_end(&self, store: &ParamStore) -> (f64, f64) {
        (store.get(self.x2), store.get(self.y2))
    }

    /// Compute the squared length of this segment.
    pub fn length_sq(&self, store: &ParamStore) -> f64 {
        let dx = store.get(self.x2) - store.get(self.x1);
        let dy = store.get(self.y2) - store.get(self.y1);
        dx * dx + dy * dy
    }
}

impl Entity for LineSegment2D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "LineSegment2D"
    }
}

// ---------------------------------------------------------------------------
// Circle2D
// ---------------------------------------------------------------------------

/// A 2D circle entity with parameters `[cx, cy, r]`.
///
/// `(cx, cy)` is the center and `r` is the radius.
#[derive(Debug, Clone)]
pub struct Circle2D {
    id: EntityId,
    cx: ParamId,
    cy: ParamId,
    r: ParamId,
    params: [ParamId; 3],
}

impl Circle2D {
    /// Create a new 2D circle entity.
    pub fn new(id: EntityId, cx: ParamId, cy: ParamId, r: ParamId) -> Self {
        Self {
            id,
            cx,
            cy,
            r,
            params: [cx, cy, r],
        }
    }

    /// Parameter ID for the center x-coordinate.
    pub fn center_x(&self) -> ParamId {
        self.cx
    }

    /// Parameter ID for the center y-coordinate.
    pub fn center_y(&self) -> ParamId {
        self.cy
    }

    /// Parameter ID for the radius.
    pub fn radius(&self) -> ParamId {
        self.r
    }

    /// Read the center coordinates from the store.
    pub fn get_center(&self, store: &ParamStore) -> (f64, f64) {
        (store.get(self.cx), store.get(self.cy))
    }

    /// Read the radius from the store.
    pub fn get_radius(&self, store: &ParamStore) -> f64 {
        store.get(self.r)
    }
}

impl Entity for Circle2D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "Circle2D"
    }
}

// ---------------------------------------------------------------------------
// Arc2D
// ---------------------------------------------------------------------------

/// A 2D arc entity with parameters `[cx, cy, r, start_angle, end_angle]`.
///
/// Defined by a center `(cx, cy)`, radius `r`, and angular range from
/// `start_angle` to `end_angle` (in radians, counter-clockwise).
#[derive(Debug, Clone)]
pub struct Arc2D {
    id: EntityId,
    cx: ParamId,
    cy: ParamId,
    r: ParamId,
    start_angle: ParamId,
    end_angle: ParamId,
    params: [ParamId; 5],
}

impl Arc2D {
    /// Create a new 2D arc entity.
    pub fn new(
        id: EntityId,
        cx: ParamId,
        cy: ParamId,
        r: ParamId,
        start_angle: ParamId,
        end_angle: ParamId,
    ) -> Self {
        Self {
            id,
            cx,
            cy,
            r,
            start_angle,
            end_angle,
            params: [cx, cy, r, start_angle, end_angle],
        }
    }

    /// Parameter ID for the center x-coordinate.
    pub fn center_x(&self) -> ParamId {
        self.cx
    }

    /// Parameter ID for the center y-coordinate.
    pub fn center_y(&self) -> ParamId {
        self.cy
    }

    /// Parameter ID for the radius.
    pub fn radius(&self) -> ParamId {
        self.r
    }

    /// Parameter ID for the start angle.
    pub fn start_angle(&self) -> ParamId {
        self.start_angle
    }

    /// Parameter ID for the end angle.
    pub fn end_angle(&self) -> ParamId {
        self.end_angle
    }

    /// Read the center coordinates from the store.
    pub fn get_center(&self, store: &ParamStore) -> (f64, f64) {
        (store.get(self.cx), store.get(self.cy))
    }

    /// Read the radius from the store.
    pub fn get_radius(&self, store: &ParamStore) -> f64 {
        store.get(self.r)
    }

    /// Read the start angle (radians) from the store.
    pub fn get_start_angle(&self, store: &ParamStore) -> f64 {
        store.get(self.start_angle)
    }

    /// Read the end angle (radians) from the store.
    pub fn get_end_angle(&self, store: &ParamStore) -> f64 {
        store.get(self.end_angle)
    }

    /// Compute a point on the arc at a given parameter `t` in `[0, 1]`.
    ///
    /// `t = 0` gives the start point, `t = 1` gives the end point.
    pub fn point_at(&self, store: &ParamStore, t: f64) -> (f64, f64) {
        let (cx, cy) = self.get_center(store);
        let r = self.get_radius(store);
        let a0 = self.get_start_angle(store);
        let a1 = self.get_end_angle(store);
        let angle = a0 + t * (a1 - a0);
        (cx + r * angle.cos(), cy + r * angle.sin())
    }
}

impl Entity for Arc2D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "Arc2D"
    }
}

// ---------------------------------------------------------------------------
// InfiniteLine2D
// ---------------------------------------------------------------------------

/// An infinite line in 2D defined by a point and direction.
///
/// Parameters: `[px, py, dx, dy]` where `(px, py)` is a point on the line
/// and `(dx, dy)` is the direction vector. The direction does not need to be
/// normalized; the solver may adjust it freely.
#[derive(Debug, Clone)]
pub struct InfiniteLine2D {
    id: EntityId,
    px: ParamId,
    py: ParamId,
    dx: ParamId,
    dy: ParamId,
    params: [ParamId; 4],
}

impl InfiniteLine2D {
    /// Create a new infinite line entity.
    pub fn new(id: EntityId, px: ParamId, py: ParamId, dx: ParamId, dy: ParamId) -> Self {
        Self {
            id,
            px,
            py,
            dx,
            dy,
            params: [px, py, dx, dy],
        }
    }

    /// Parameter ID for the reference point x-coordinate.
    pub fn point_x(&self) -> ParamId {
        self.px
    }

    /// Parameter ID for the reference point y-coordinate.
    pub fn point_y(&self) -> ParamId {
        self.py
    }

    /// Parameter ID for the direction x-component.
    pub fn dir_x(&self) -> ParamId {
        self.dx
    }

    /// Parameter ID for the direction y-component.
    pub fn dir_y(&self) -> ParamId {
        self.dy
    }

    /// Read the reference point from the store.
    pub fn get_point(&self, store: &ParamStore) -> (f64, f64) {
        (store.get(self.px), store.get(self.py))
    }

    /// Read the direction vector from the store.
    pub fn get_direction(&self, store: &ParamStore) -> (f64, f64) {
        (store.get(self.dx), store.get(self.dy))
    }
}

impl Entity for InfiniteLine2D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "InfiniteLine2D"
    }
}

// ---------------------------------------------------------------------------
// Ellipse2D
// ---------------------------------------------------------------------------

/// A 2D ellipse with center (cx, cy), semi-major axis a, semi-minor axis b,
/// and rotation angle theta (radians).
/// Parametric: x(t) = cx + a*cos(t)*cos(θ) - b*sin(t)*sin(θ)
///             y(t) = cy + a*cos(t)*sin(θ) + b*sin(t)*cos(θ)
/// 5 DOF: [cx, cy, a, b, theta]
#[derive(Debug)]
pub struct Ellipse2D {
    id: EntityId,
    cx: ParamId,
    cy: ParamId,
    a: ParamId,
    b: ParamId,
    theta: ParamId,
    params: [ParamId; 5],
}

impl Ellipse2D {
    /// Create a new 2D ellipse entity.
    pub fn new(
        id: EntityId,
        cx: ParamId,
        cy: ParamId,
        a: ParamId,
        b: ParamId,
        theta: ParamId,
    ) -> Self {
        Self {
            id,
            cx,
            cy,
            a,
            b,
            theta,
            params: [cx, cy, a, b, theta],
        }
    }

    /// Parameter ID for the center x-coordinate.
    pub fn center_x(&self) -> ParamId {
        self.cx
    }

    /// Parameter ID for the center y-coordinate.
    pub fn center_y(&self) -> ParamId {
        self.cy
    }

    /// Parameter ID for the semi-major axis.
    pub fn semi_major(&self) -> ParamId {
        self.a
    }

    /// Parameter ID for the semi-minor axis.
    pub fn semi_minor(&self) -> ParamId {
        self.b
    }

    /// Parameter ID for the rotation angle.
    pub fn rotation(&self) -> ParamId {
        self.theta
    }

    /// Compute a point on the ellipse at parameter `t` (radians).
    pub fn point_at(&self, store: &ParamStore, t: f64) -> (f64, f64) {
        let cx = store.get(self.cx);
        let cy = store.get(self.cy);
        let a = store.get(self.a);
        let b = store.get(self.b);
        let th = store.get(self.theta);
        let x = cx + a * t.cos() * th.cos() - b * t.sin() * th.sin();
        let y = cy + a * t.cos() * th.sin() + b * t.sin() * th.cos();
        (x, y)
    }
}

impl Entity for Ellipse2D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "Ellipse2D"
    }
}

// ---------------------------------------------------------------------------
// Spline2D
// ---------------------------------------------------------------------------

/// A 2D spline defined by n control points. DOF = 2n.
#[derive(Debug)]
pub struct Spline2D {
    id: EntityId,
    params: Vec<ParamId>,
    n_points: usize,
}

impl Spline2D {
    /// Create a new 2D spline entity from a list of (x, y) parameter ID pairs.
    pub fn new(id: EntityId, control_points: Vec<(ParamId, ParamId)>) -> Self {
        let n_points = control_points.len();
        let params: Vec<ParamId> = control_points
            .into_iter()
            .flat_map(|(x, y)| [x, y])
            .collect();
        Self {
            id,
            params,
            n_points,
        }
    }

    /// Number of control points.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Parameter ID for the x-coordinate of control point `i`.
    pub fn control_point_x(&self, i: usize) -> ParamId {
        self.params[2 * i]
    }

    /// Parameter ID for the y-coordinate of control point `i`.
    pub fn control_point_y(&self, i: usize) -> ParamId {
        self.params[2 * i + 1]
    }

    /// Read the coordinates of control point `i` from the store.
    pub fn get_control_point(&self, store: &ParamStore, i: usize) -> (f64, f64) {
        (
            store.get(self.params[2 * i]),
            store.get(self.params[2 * i + 1]),
        )
    }
}

impl Entity for Spline2D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "Spline2D"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::EntityId;
    use crate::param::ParamStore;

    fn dummy_entity_id(idx: u32) -> EntityId {
        EntityId::new(idx, 0)
    }

    #[test]
    fn test_point2d_creation_and_accessors() {
        let eid = dummy_entity_id(0);
        let mut store = ParamStore::new();
        let x = store.alloc(3.0, eid);
        let y = store.alloc(4.0, eid);

        let point = Point2D::new(eid, x, y);

        assert_eq!(point.id(), eid);
        assert_eq!(point.name(), "Point2D");
        assert_eq!(point.params().len(), 2);
        assert_eq!(point.x(), x);
        assert_eq!(point.y(), y);
        assert!((point.get_x(&store) - 3.0).abs() < 1e-15);
        assert!((point.get_y(&store) - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_point2d_params_identity() {
        let eid = dummy_entity_id(0);
        let mut store = ParamStore::new();
        let x = store.alloc(1.0, eid);
        let y = store.alloc(2.0, eid);

        let point = Point2D::new(eid, x, y);
        assert_eq!(point.params()[0], x);
        assert_eq!(point.params()[1], y);
    }

    #[test]
    fn test_line_segment2d() {
        let eid = dummy_entity_id(0);
        let mut store = ParamStore::new();
        let x1 = store.alloc(0.0, eid);
        let y1 = store.alloc(0.0, eid);
        let x2 = store.alloc(3.0, eid);
        let y2 = store.alloc(4.0, eid);

        let line = LineSegment2D::new(eid, x1, y1, x2, y2);

        assert_eq!(line.id(), eid);
        assert_eq!(line.name(), "LineSegment2D");
        assert_eq!(line.params().len(), 4);
        assert_eq!(line.start_x(), x1);
        assert_eq!(line.start_y(), y1);
        assert_eq!(line.end_x(), x2);
        assert_eq!(line.end_y(), y2);

        let (sx, sy) = line.get_start(&store);
        assert!((sx - 0.0).abs() < 1e-15);
        assert!((sy - 0.0).abs() < 1e-15);

        let (ex, ey) = line.get_end(&store);
        assert!((ex - 3.0).abs() < 1e-15);
        assert!((ey - 4.0).abs() < 1e-15);

        assert!((line.length_sq(&store) - 25.0).abs() < 1e-15);
    }

    #[test]
    fn test_circle2d() {
        let eid = dummy_entity_id(0);
        let mut store = ParamStore::new();
        let cx = store.alloc(1.0, eid);
        let cy = store.alloc(2.0, eid);
        let r = store.alloc(5.0, eid);

        let circle = Circle2D::new(eid, cx, cy, r);

        assert_eq!(circle.id(), eid);
        assert_eq!(circle.name(), "Circle2D");
        assert_eq!(circle.params().len(), 3);
        assert_eq!(circle.center_x(), cx);
        assert_eq!(circle.center_y(), cy);
        assert_eq!(circle.radius(), r);
        assert!((circle.get_radius(&store) - 5.0).abs() < 1e-15);

        let (ccx, ccy) = circle.get_center(&store);
        assert!((ccx - 1.0).abs() < 1e-15);
        assert!((ccy - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_arc2d() {
        let eid = dummy_entity_id(0);
        let mut store = ParamStore::new();
        let cx = store.alloc(0.0, eid);
        let cy = store.alloc(0.0, eid);
        let r = store.alloc(1.0, eid);
        let a0 = store.alloc(0.0, eid);
        let a1 = store.alloc(std::f64::consts::FRAC_PI_2, eid);

        let arc = Arc2D::new(eid, cx, cy, r, a0, a1);

        assert_eq!(arc.id(), eid);
        assert_eq!(arc.name(), "Arc2D");
        assert_eq!(arc.params().len(), 5);

        // t=0 -> start point at angle 0 -> (1, 0)
        let (px, py) = arc.point_at(&store, 0.0);
        assert!((px - 1.0).abs() < 1e-12);
        assert!(py.abs() < 1e-12);

        // t=1 -> end point at angle pi/2 -> (0, 1)
        let (px, py) = arc.point_at(&store, 1.0);
        assert!(px.abs() < 1e-12);
        assert!((py - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_infinite_line2d() {
        let eid = dummy_entity_id(0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);
        let py = store.alloc(2.0, eid);
        let dx = store.alloc(1.0, eid);
        let dy = store.alloc(0.0, eid);

        let line = InfiniteLine2D::new(eid, px, py, dx, dy);

        assert_eq!(line.id(), eid);
        assert_eq!(line.name(), "InfiniteLine2D");
        assert_eq!(line.params().len(), 4);
        assert_eq!(line.point_x(), px);
        assert_eq!(line.point_y(), py);
        assert_eq!(line.dir_x(), dx);
        assert_eq!(line.dir_y(), dy);

        let (gx, gy) = line.get_point(&store);
        assert!((gx - 1.0).abs() < 1e-15);
        assert!((gy - 2.0).abs() < 1e-15);

        let (gdx, gdy) = line.get_direction(&store);
        assert!((gdx - 1.0).abs() < 1e-15);
        assert!(gdy.abs() < 1e-15);
    }

    #[test]
    fn test_entity_trait_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Point2D>();
        assert_send_sync::<LineSegment2D>();
        assert_send_sync::<Circle2D>();
        assert_send_sync::<Arc2D>();
        assert_send_sync::<InfiniteLine2D>();
    }

    #[test]
    fn test_line_segment_shared_params() {
        // Verify that a line segment can share params with point entities.
        let p1_eid = dummy_entity_id(0);
        let p2_eid = dummy_entity_id(1);
        let line_eid = dummy_entity_id(2);

        let mut store = ParamStore::new();
        let x1 = store.alloc(0.0, p1_eid);
        let y1 = store.alloc(0.0, p1_eid);
        let x2 = store.alloc(10.0, p2_eid);
        let y2 = store.alloc(0.0, p2_eid);

        let _p1 = Point2D::new(p1_eid, x1, y1);
        let _p2 = Point2D::new(p2_eid, x2, y2);
        let line = LineSegment2D::new(line_eid, x1, y1, x2, y2);

        // Line shares the same param IDs as the points.
        assert_eq!(line.start_x(), x1);
        assert_eq!(line.end_x(), x2);

        // Modifying the param affects both point and line readings.
        store.set(x2, 20.0);
        let (ex, _) = line.get_end(&store);
        assert!((ex - 20.0).abs() < 1e-15);
    }
}
