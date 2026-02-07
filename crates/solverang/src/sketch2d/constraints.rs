//! 2D constraint types for the sketch constraint system.
//!
//! All constraints implement the [`Constraint`] trait. Where applicable, **squared
//! formulations** are used (e.g. `dx^2+dy^2 - d^2` instead of `sqrt(dx^2+dy^2) - d`)
//! to eliminate the singularity at zero distance and produce smooth Jacobians
//! everywhere.

use crate::constraint::Constraint;
use crate::id::{ConstraintId, EntityId, ParamId};
use crate::param::ParamStore;

// ===========================================================================
// DistancePtPt
// ===========================================================================

/// Distance between two points (squared formulation).
///
/// Residual: `(x2-x1)^2 + (y2-y1)^2 - d^2`
///
/// This eliminates the `sqrt` singularity at zero distance.
#[derive(Debug, Clone)]
pub struct DistancePtPt {
    id: ConstraintId,
    entities: [EntityId; 2],
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    target_sq: f64,
    params: [ParamId; 4],
}

impl DistancePtPt {
    /// Create a distance constraint between two points.
    ///
    /// `distance` is the desired distance (not squared).
    pub fn new(
        id: ConstraintId,
        e1: EntityId,
        e2: EntityId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
        distance: f64,
    ) -> Self {
        Self {
            id,
            entities: [e1, e2],
            x1,
            y1,
            x2,
            y2,
            target_sq: distance * distance,
            params: [x1, y1, x2, y2],
        }
    }
}

impl Constraint for DistancePtPt {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &str {
        "DistancePtPt"
    }

    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }

    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let dx = store.get(self.x2) - store.get(self.x1);
        let dy = store.get(self.y2) - store.get(self.y1);
        vec![dx * dx + dy * dy - self.target_sq]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let dx = store.get(self.x2) - store.get(self.x1);
        let dy = store.get(self.y2) - store.get(self.y1);
        vec![
            (0, self.x1, -2.0 * dx),
            (0, self.y1, -2.0 * dy),
            (0, self.x2, 2.0 * dx),
            (0, self.y2, 2.0 * dy),
        ]
    }
}

// ===========================================================================
// DistancePtLine
// ===========================================================================

/// Distance from a point to a line segment (squared formulation).
///
/// Residual: `cross^2 / len_sq - d^2`
///
/// where `cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)` and
/// `len_sq = (x2-x1)^2 + (y2-y1)^2`.
#[derive(Debug, Clone)]
pub struct DistancePtLine {
    id: ConstraintId,
    entities: [EntityId; 2],
    px: ParamId,
    py: ParamId,
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    target_sq: f64,
    params: [ParamId; 6],
}

impl DistancePtLine {
    /// Create a point-to-line distance constraint.
    ///
    /// `point_entity` is the point, `line_entity` is the line segment.
    /// `distance` is the desired distance (not squared).
    pub fn new(
        id: ConstraintId,
        point_entity: EntityId,
        line_entity: EntityId,
        px: ParamId,
        py: ParamId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
        distance: f64,
    ) -> Self {
        Self {
            id,
            entities: [point_entity, line_entity],
            px,
            py,
            x1,
            y1,
            x2,
            y2,
            target_sq: distance * distance,
            params: [px, py, x1, y1, x2, y2],
        }
    }
}

impl Constraint for DistancePtLine {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &str {
        "DistancePtLine"
    }

    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }

    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let dx = store.get(self.x2) - store.get(self.x1);
        let dy = store.get(self.y2) - store.get(self.y1);
        let vx = store.get(self.px) - store.get(self.x1);
        let vy = store.get(self.py) - store.get(self.y1);
        let cross = dx * vy - dy * vx;
        let len_sq = dx * dx + dy * dy;
        vec![cross * cross / len_sq - self.target_sq]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let x1v = store.get(self.x1);
        let y1v = store.get(self.y1);
        let x2v = store.get(self.x2);
        let y2v = store.get(self.y2);
        let pxv = store.get(self.px);
        let pyv = store.get(self.py);

        let dx = x2v - x1v;
        let dy = y2v - y1v;
        let vx = pxv - x1v;
        let vy = pyv - y1v;
        let cross = dx * vy - dy * vx;
        let len_sq = dx * dx + dy * dy;

        // R = cross^2 / len_sq - target_sq
        // dR/dp = (2*cross*dcross/dp * len_sq - cross^2 * dlen_sq/dp) / len_sq^2

        let c2 = cross * cross;
        let l2 = len_sq * len_sq;

        // dcross/d(px) = -dy,  dlen_sq/d(px) = 0
        let dr_dpx = 2.0 * cross * (-dy) / len_sq;
        // dcross/d(py) = dx,   dlen_sq/d(py) = 0
        let dr_dpy = 2.0 * cross * dx / len_sq;
        // dcross/d(x1) = y2-py,  dlen_sq/d(x1) = -2*dx
        let dr_dx1 = (2.0 * cross * (y2v - pyv) * len_sq - c2 * (-2.0 * dx)) / l2;
        // dcross/d(y1) = px-x2,  dlen_sq/d(y1) = -2*dy
        let dr_dy1 = (2.0 * cross * (pxv - x2v) * len_sq - c2 * (-2.0 * dy)) / l2;
        // dcross/d(x2) = py-y1,  dlen_sq/d(x2) = 2*dx
        let dr_dx2 = (2.0 * cross * (pyv - y1v) * len_sq - c2 * (2.0 * dx)) / l2;
        // dcross/d(y2) = x1-px,  dlen_sq/d(y2) = 2*dy
        let dr_dy2 = (2.0 * cross * (x1v - pxv) * len_sq - c2 * (2.0 * dy)) / l2;

        vec![
            (0, self.px, dr_dpx),
            (0, self.py, dr_dpy),
            (0, self.x1, dr_dx1),
            (0, self.y1, dr_dy1),
            (0, self.x2, dr_dx2),
            (0, self.y2, dr_dy2),
        ]
    }
}

// ===========================================================================
// Coincident
// ===========================================================================

/// Coincident constraint: two points at the same location.
///
/// Residuals: `[x2-x1, y2-y1]`
#[derive(Debug, Clone)]
pub struct Coincident {
    id: ConstraintId,
    entities: [EntityId; 2],
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    params: [ParamId; 4],
}

impl Coincident {
    pub fn new(
        id: ConstraintId,
        e1: EntityId,
        e2: EntityId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [e1, e2],
            x1,
            y1,
            x2,
            y2,
            params: [x1, y1, x2, y2],
        }
    }
}

impl Constraint for Coincident {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Coincident"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        2
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        vec![
            store.get(self.x2) - store.get(self.x1),
            store.get(self.y2) - store.get(self.y1),
        ]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![
            (0, self.x1, -1.0),
            (0, self.x2, 1.0),
            (1, self.y1, -1.0),
            (1, self.y2, 1.0),
        ]
    }
}

// ===========================================================================
// TangentLineCircle
// ===========================================================================

/// Tangent: line tangent to circle.
///
/// Residual: `(signed_dist_from_center_to_line)^2 - r^2`
///
/// Specifically: `cross^2 / len_sq - r^2` where
/// `cross = dx*(cy-y1) - dy*(cx-x1)`, `dx = x2-x1`, `dy = y2-y1`.
#[derive(Debug, Clone)]
pub struct TangentLineCircle {
    id: ConstraintId,
    entities: [EntityId; 2],
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    cx: ParamId,
    cy: ParamId,
    r: ParamId,
    params: [ParamId; 7],
}

impl TangentLineCircle {
    /// Create a tangent constraint between a line segment and a circle.
    pub fn new(
        id: ConstraintId,
        line_entity: EntityId,
        circle_entity: EntityId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
        cx: ParamId,
        cy: ParamId,
        r: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [line_entity, circle_entity],
            x1,
            y1,
            x2,
            y2,
            cx,
            cy,
            r,
            params: [x1, y1, x2, y2, cx, cy, r],
        }
    }
}

impl Constraint for TangentLineCircle {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "TangentLineCircle"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let x1v = store.get(self.x1);
        let y1v = store.get(self.y1);
        let x2v = store.get(self.x2);
        let y2v = store.get(self.y2);
        let cxv = store.get(self.cx);
        let cyv = store.get(self.cy);
        let rv = store.get(self.r);

        let dx = x2v - x1v;
        let dy = y2v - y1v;
        let cross = dx * (cyv - y1v) - dy * (cxv - x1v);
        let len_sq = dx * dx + dy * dy;
        vec![cross * cross / len_sq - rv * rv]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let x1v = store.get(self.x1);
        let y1v = store.get(self.y1);
        let x2v = store.get(self.x2);
        let y2v = store.get(self.y2);
        let cxv = store.get(self.cx);
        let cyv = store.get(self.cy);
        let rv = store.get(self.r);

        let dx = x2v - x1v;
        let dy = y2v - y1v;
        let cross = dx * (cyv - y1v) - dy * (cxv - x1v);
        let len_sq = dx * dx + dy * dy;
        let c2 = cross * cross;
        let l2 = len_sq * len_sq;

        // Same structure as DistancePtLine with px=cx, py=cy, plus dR/dr = -2*r.
        let dr_dcx = 2.0 * cross * (-dy) / len_sq;
        let dr_dcy = 2.0 * cross * dx / len_sq;

        let dr_dx1 = (2.0 * cross * (y2v - cyv) * len_sq - c2 * (-2.0 * dx)) / l2;
        let dr_dy1 = (2.0 * cross * (cxv - x2v) * len_sq - c2 * (-2.0 * dy)) / l2;
        let dr_dx2 = (2.0 * cross * (cyv - y1v) * len_sq - c2 * (2.0 * dx)) / l2;
        let dr_dy2 = (2.0 * cross * (x1v - cxv) * len_sq - c2 * (2.0 * dy)) / l2;

        let dr_dr = -2.0 * rv;

        vec![
            (0, self.x1, dr_dx1),
            (0, self.y1, dr_dy1),
            (0, self.x2, dr_dx2),
            (0, self.y2, dr_dy2),
            (0, self.cx, dr_dcx),
            (0, self.cy, dr_dcy),
            (0, self.r, dr_dr),
        ]
    }
}

// ===========================================================================
// TangentCircleCircle
// ===========================================================================

/// Tangent: circle to circle.
///
/// For external tangency: `(dist_between_centers)^2 - (r1+r2)^2 = 0`
/// For internal tangency: `(dist_between_centers)^2 - (r1-r2)^2 = 0`
#[derive(Debug, Clone)]
pub struct TangentCircleCircle {
    id: ConstraintId,
    entities: [EntityId; 2],
    cx1: ParamId,
    cy1: ParamId,
    r1: ParamId,
    cx2: ParamId,
    cy2: ParamId,
    r2: ParamId,
    external: bool,
    params: [ParamId; 6],
}

impl TangentCircleCircle {
    /// Create an external tangency constraint between two circles.
    pub fn external(
        id: ConstraintId,
        e1: EntityId,
        e2: EntityId,
        cx1: ParamId,
        cy1: ParamId,
        r1: ParamId,
        cx2: ParamId,
        cy2: ParamId,
        r2: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [e1, e2],
            cx1,
            cy1,
            r1,
            cx2,
            cy2,
            r2,
            external: true,
            params: [cx1, cy1, r1, cx2, cy2, r2],
        }
    }

    /// Create an internal tangency constraint between two circles.
    pub fn internal(
        id: ConstraintId,
        e1: EntityId,
        e2: EntityId,
        cx1: ParamId,
        cy1: ParamId,
        r1: ParamId,
        cx2: ParamId,
        cy2: ParamId,
        r2: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [e1, e2],
            cx1,
            cy1,
            r1,
            cx2,
            cy2,
            r2,
            external: false,
            params: [cx1, cy1, r1, cx2, cy2, r2],
        }
    }
}

impl Constraint for TangentCircleCircle {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "TangentCircleCircle"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let dcx = store.get(self.cx2) - store.get(self.cx1);
        let dcy = store.get(self.cy2) - store.get(self.cy1);
        let r1v = store.get(self.r1);
        let r2v = store.get(self.r2);
        let dist_sq = dcx * dcx + dcy * dcy;
        let rsum = if self.external {
            r1v + r2v
        } else {
            r1v - r2v
        };
        vec![dist_sq - rsum * rsum]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let dcx = store.get(self.cx2) - store.get(self.cx1);
        let dcy = store.get(self.cy2) - store.get(self.cy1);
        let r1v = store.get(self.r1);
        let r2v = store.get(self.r2);
        let rsum = if self.external {
            r1v + r2v
        } else {
            r1v - r2v
        };

        let dr_dr2 = if self.external {
            -2.0 * rsum
        } else {
            2.0 * rsum // d/dr2 of -(r1-r2)^2 = 2*(r1-r2)
        };

        vec![
            (0, self.cx1, -2.0 * dcx),
            (0, self.cy1, -2.0 * dcy),
            (0, self.r1, -2.0 * rsum),
            (0, self.cx2, 2.0 * dcx),
            (0, self.cy2, 2.0 * dcy),
            (0, self.r2, dr_dr2),
        ]
    }
}

// ===========================================================================
// Parallel
// ===========================================================================

/// Parallel: two line segments are parallel.
///
/// Residual: `(x2-x1)*(y4-y3) - (y2-y1)*(x4-x3)` (cross product of
/// direction vectors = 0).
#[derive(Debug, Clone)]
pub struct Parallel {
    id: ConstraintId,
    entities: [EntityId; 2],
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    x3: ParamId,
    y3: ParamId,
    x4: ParamId,
    y4: ParamId,
    params: [ParamId; 8],
}

impl Parallel {
    /// Create a parallel constraint between two line segments.
    ///
    /// Line 1: `(x1,y1)` to `(x2,y2)`, Line 2: `(x3,y3)` to `(x4,y4)`.
    pub fn new(
        id: ConstraintId,
        line1: EntityId,
        line2: EntityId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
        x3: ParamId,
        y3: ParamId,
        x4: ParamId,
        y4: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [line1, line2],
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            x4,
            y4,
            params: [x1, y1, x2, y2, x3, y3, x4, y4],
        }
    }
}

impl Constraint for Parallel {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Parallel"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let dx1 = store.get(self.x2) - store.get(self.x1);
        let dy1 = store.get(self.y2) - store.get(self.y1);
        let dx2 = store.get(self.x4) - store.get(self.x3);
        let dy2 = store.get(self.y4) - store.get(self.y3);
        vec![dx1 * dy2 - dy1 * dx2]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let dx1 = store.get(self.x2) - store.get(self.x1);
        let dy1 = store.get(self.y2) - store.get(self.y1);
        let dx2 = store.get(self.x4) - store.get(self.x3);
        let dy2 = store.get(self.y4) - store.get(self.y3);

        // R = dx1*dy2 - dy1*dx2
        vec![
            (0, self.x1, -dy2),
            (0, self.y1, dx2),
            (0, self.x2, dy2),
            (0, self.y2, -dx2),
            (0, self.x3, dy1),
            (0, self.y3, -dx1),
            (0, self.x4, -dy1),
            (0, self.y4, dx1),
        ]
    }
}

// ===========================================================================
// Perpendicular
// ===========================================================================

/// Perpendicular: two line segments are perpendicular.
///
/// Residual: `(x2-x1)*(x4-x3) + (y2-y1)*(y4-y3)` (dot product = 0).
#[derive(Debug, Clone)]
pub struct Perpendicular {
    id: ConstraintId,
    entities: [EntityId; 2],
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    x3: ParamId,
    y3: ParamId,
    x4: ParamId,
    y4: ParamId,
    params: [ParamId; 8],
}

impl Perpendicular {
    /// Create a perpendicular constraint between two line segments.
    pub fn new(
        id: ConstraintId,
        line1: EntityId,
        line2: EntityId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
        x3: ParamId,
        y3: ParamId,
        x4: ParamId,
        y4: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [line1, line2],
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            x4,
            y4,
            params: [x1, y1, x2, y2, x3, y3, x4, y4],
        }
    }
}

impl Constraint for Perpendicular {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Perpendicular"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let dx1 = store.get(self.x2) - store.get(self.x1);
        let dy1 = store.get(self.y2) - store.get(self.y1);
        let dx2 = store.get(self.x4) - store.get(self.x3);
        let dy2 = store.get(self.y4) - store.get(self.y3);
        vec![dx1 * dx2 + dy1 * dy2]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let dx1 = store.get(self.x2) - store.get(self.x1);
        let dy1 = store.get(self.y2) - store.get(self.y1);
        let dx2 = store.get(self.x4) - store.get(self.x3);
        let dy2 = store.get(self.y4) - store.get(self.y3);

        // R = dx1*dx2 + dy1*dy2
        vec![
            (0, self.x1, -dx2),
            (0, self.y1, -dy2),
            (0, self.x2, dx2),
            (0, self.y2, dy2),
            (0, self.x3, -dx1),
            (0, self.y3, -dy1),
            (0, self.x4, dx1),
            (0, self.y4, dy1),
        ]
    }
}

// ===========================================================================
// Angle
// ===========================================================================

/// Angle constraint: angle of a line segment from horizontal.
///
/// Residual: `(y2-y1)*cos(a) - (x2-x1)*sin(a)`
///
/// This equals zero when the direction `(x2-x1, y2-y1)` makes angle `a` with
/// the positive x-axis.
#[derive(Debug, Clone)]
pub struct Angle {
    id: ConstraintId,
    entities: [EntityId; 1],
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    sin_a: f64,
    cos_a: f64,
    params: [ParamId; 4],
}

impl Angle {
    /// Create an angle constraint for a line segment.
    ///
    /// `angle` is in radians, measured counter-clockwise from the positive x-axis.
    pub fn new(
        id: ConstraintId,
        line_entity: EntityId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
        angle: f64,
    ) -> Self {
        Self {
            id,
            entities: [line_entity],
            x1,
            y1,
            x2,
            y2,
            sin_a: angle.sin(),
            cos_a: angle.cos(),
            params: [x1, y1, x2, y2],
        }
    }
}

impl Constraint for Angle {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Angle"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let dx = store.get(self.x2) - store.get(self.x1);
        let dy = store.get(self.y2) - store.get(self.y1);
        vec![dy * self.cos_a - dx * self.sin_a]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![
            (0, self.x1, self.sin_a),
            (0, self.y1, -self.cos_a),
            (0, self.x2, -self.sin_a),
            (0, self.y2, self.cos_a),
        ]
    }
}

// ===========================================================================
// Horizontal
// ===========================================================================

/// Horizontal: two points at the same y-coordinate.
///
/// Residual: `y2 - y1`
#[derive(Debug, Clone)]
pub struct Horizontal {
    id: ConstraintId,
    entities: [EntityId; 2],
    y1: ParamId,
    y2: ParamId,
    params: [ParamId; 2],
}

impl Horizontal {
    pub fn new(
        id: ConstraintId,
        e1: EntityId,
        e2: EntityId,
        y1: ParamId,
        y2: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [e1, e2],
            y1,
            y2,
            params: [y1, y2],
        }
    }
}

impl Constraint for Horizontal {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Horizontal"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        vec![store.get(self.y2) - store.get(self.y1)]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![(0, self.y1, -1.0), (0, self.y2, 1.0)]
    }
}

// ===========================================================================
// Vertical
// ===========================================================================

/// Vertical: two points at the same x-coordinate.
///
/// Residual: `x2 - x1`
#[derive(Debug, Clone)]
pub struct Vertical {
    id: ConstraintId,
    entities: [EntityId; 2],
    x1: ParamId,
    x2: ParamId,
    params: [ParamId; 2],
}

impl Vertical {
    pub fn new(
        id: ConstraintId,
        e1: EntityId,
        e2: EntityId,
        x1: ParamId,
        x2: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [e1, e2],
            x1,
            x2,
            params: [x1, x2],
        }
    }
}

impl Constraint for Vertical {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Vertical"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        vec![store.get(self.x2) - store.get(self.x1)]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![(0, self.x1, -1.0), (0, self.x2, 1.0)]
    }
}

// ===========================================================================
// Fixed
// ===========================================================================

/// Fixed position: a point pinned to specific coordinates.
///
/// Residuals: `[x - tx, y - ty]`
#[derive(Debug, Clone)]
pub struct Fixed {
    id: ConstraintId,
    entities: [EntityId; 1],
    x: ParamId,
    y: ParamId,
    tx: f64,
    ty: f64,
    params: [ParamId; 2],
}

impl Fixed {
    pub fn new(
        id: ConstraintId,
        entity: EntityId,
        x: ParamId,
        y: ParamId,
        tx: f64,
        ty: f64,
    ) -> Self {
        Self {
            id,
            entities: [entity],
            x,
            y,
            tx,
            ty,
            params: [x, y],
        }
    }
}

impl Constraint for Fixed {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Fixed"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        2
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        vec![
            store.get(self.x) - self.tx,
            store.get(self.y) - self.ty,
        ]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![(0, self.x, 1.0), (1, self.y, 1.0)]
    }
}

// ===========================================================================
// Midpoint
// ===========================================================================

/// Midpoint: a point at the midpoint of a line segment.
///
/// Residuals: `[mx - (x1+x2)/2, my - (y1+y2)/2]`
#[derive(Debug, Clone)]
pub struct Midpoint {
    id: ConstraintId,
    entities: [EntityId; 2],
    mx: ParamId,
    my: ParamId,
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    params: [ParamId; 6],
}

impl Midpoint {
    /// `point_entity` is the midpoint, `line_entity` is the line segment.
    pub fn new(
        id: ConstraintId,
        point_entity: EntityId,
        line_entity: EntityId,
        mx: ParamId,
        my: ParamId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [point_entity, line_entity],
            mx,
            my,
            x1,
            y1,
            x2,
            y2,
            params: [mx, my, x1, y1, x2, y2],
        }
    }
}

impl Constraint for Midpoint {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Midpoint"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        2
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let mid_x = (store.get(self.x1) + store.get(self.x2)) * 0.5;
        let mid_y = (store.get(self.y1) + store.get(self.y2)) * 0.5;
        vec![
            store.get(self.mx) - mid_x,
            store.get(self.my) - mid_y,
        ]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![
            (0, self.mx, 1.0),
            (0, self.x1, -0.5),
            (0, self.x2, -0.5),
            (1, self.my, 1.0),
            (1, self.y1, -0.5),
            (1, self.y2, -0.5),
        ]
    }
}

// ===========================================================================
// Symmetric
// ===========================================================================

/// Symmetric: two points are symmetric about a center point.
///
/// Residuals: `[x1 + x2 - 2*cx, y1 + y2 - 2*cy]`
#[derive(Debug, Clone)]
pub struct Symmetric {
    id: ConstraintId,
    entities: [EntityId; 3],
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    cx: ParamId,
    cy: ParamId,
    params: [ParamId; 6],
}

impl Symmetric {
    /// `p1` and `p2` are the symmetric pair, `center` is the center of symmetry.
    pub fn new(
        id: ConstraintId,
        p1: EntityId,
        p2: EntityId,
        center: EntityId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
        cx: ParamId,
        cy: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [p1, p2, center],
            x1,
            y1,
            x2,
            y2,
            cx,
            cy,
            params: [x1, y1, x2, y2, cx, cy],
        }
    }
}

impl Constraint for Symmetric {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "Symmetric"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        2
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        vec![
            store.get(self.x1) + store.get(self.x2) - 2.0 * store.get(self.cx),
            store.get(self.y1) + store.get(self.y2) - 2.0 * store.get(self.cy),
        ]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![
            (0, self.x1, 1.0),
            (0, self.x2, 1.0),
            (0, self.cx, -2.0),
            (1, self.y1, 1.0),
            (1, self.y2, 1.0),
            (1, self.cy, -2.0),
        ]
    }
}

// ===========================================================================
// EqualLength
// ===========================================================================

/// Equal length: two line segments have equal length (squared formulation).
///
/// Residual: `(x2-x1)^2+(y2-y1)^2 - (x4-x3)^2-(y4-y3)^2`
#[derive(Debug, Clone)]
pub struct EqualLength {
    id: ConstraintId,
    entities: [EntityId; 2],
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
    x3: ParamId,
    y3: ParamId,
    x4: ParamId,
    y4: ParamId,
    params: [ParamId; 8],
}

impl EqualLength {
    /// Create an equal-length constraint between two line segments.
    pub fn new(
        id: ConstraintId,
        line1: EntityId,
        line2: EntityId,
        x1: ParamId,
        y1: ParamId,
        x2: ParamId,
        y2: ParamId,
        x3: ParamId,
        y3: ParamId,
        x4: ParamId,
        y4: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [line1, line2],
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            x4,
            y4,
            params: [x1, y1, x2, y2, x3, y3, x4, y4],
        }
    }
}

impl Constraint for EqualLength {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "EqualLength"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let dx1 = store.get(self.x2) - store.get(self.x1);
        let dy1 = store.get(self.y2) - store.get(self.y1);
        let dx2 = store.get(self.x4) - store.get(self.x3);
        let dy2 = store.get(self.y4) - store.get(self.y3);
        vec![dx1 * dx1 + dy1 * dy1 - dx2 * dx2 - dy2 * dy2]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let dx1 = store.get(self.x2) - store.get(self.x1);
        let dy1 = store.get(self.y2) - store.get(self.y1);
        let dx2 = store.get(self.x4) - store.get(self.x3);
        let dy2 = store.get(self.y4) - store.get(self.y3);

        vec![
            (0, self.x1, -2.0 * dx1),
            (0, self.y1, -2.0 * dy1),
            (0, self.x2, 2.0 * dx1),
            (0, self.y2, 2.0 * dy1),
            (0, self.x3, 2.0 * dx2),
            (0, self.y3, 2.0 * dy2),
            (0, self.x4, -2.0 * dx2),
            (0, self.y4, -2.0 * dy2),
        ]
    }
}

// ===========================================================================
// PointOnCircle
// ===========================================================================

/// Point on circle: point lies on a circle.
///
/// Residual: `(px-cx)^2 + (py-cy)^2 - r^2`
#[derive(Debug, Clone)]
pub struct PointOnCircle {
    id: ConstraintId,
    entities: [EntityId; 2],
    px: ParamId,
    py: ParamId,
    cx: ParamId,
    cy: ParamId,
    r: ParamId,
    params: [ParamId; 5],
}

impl PointOnCircle {
    pub fn new(
        id: ConstraintId,
        point_entity: EntityId,
        circle_entity: EntityId,
        px: ParamId,
        py: ParamId,
        cx: ParamId,
        cy: ParamId,
        r: ParamId,
    ) -> Self {
        Self {
            id,
            entities: [point_entity, circle_entity],
            px,
            py,
            cx,
            cy,
            r,
            params: [px, py, cx, cy, r],
        }
    }
}

impl Constraint for PointOnCircle {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "PointOnCircle"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &self.entities
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let dpx = store.get(self.px) - store.get(self.cx);
        let dpy = store.get(self.py) - store.get(self.cy);
        let rv = store.get(self.r);
        vec![dpx * dpx + dpy * dpy - rv * rv]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let dpx = store.get(self.px) - store.get(self.cx);
        let dpy = store.get(self.py) - store.get(self.cy);
        let rv = store.get(self.r);

        vec![
            (0, self.px, 2.0 * dpx),
            (0, self.py, 2.0 * dpy),
            (0, self.cx, -2.0 * dpx),
            (0, self.cy, -2.0 * dpy),
            (0, self.r, -2.0 * rv),
        ]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId};
    use crate::param::ParamStore;

    fn eid(i: u32) -> EntityId {
        EntityId::new(i, 0)
    }

    fn cid(i: u32) -> ConstraintId {
        ConstraintId::new(i, 0)
    }

    /// Verify analytical Jacobian against central finite differences.
    fn check_jacobian(constraint: &dyn Constraint, store: &ParamStore, eps: f64, tol: f64) {
        let params = constraint.param_ids().to_vec();
        let analytical = constraint.jacobian(store);

        for eq in 0..constraint.equation_count() {
            for &pid in &params {
                // Central finite difference
                let mut plus = store.snapshot();
                let orig = plus.get(pid);
                plus.set(pid, orig + eps);
                let r_plus = constraint.residuals(&plus);

                let mut minus = store.snapshot();
                minus.set(pid, orig - eps);
                let r_minus = constraint.residuals(&minus);

                let fd = (r_plus[eq] - r_minus[eq]) / (2.0 * eps);

                // Sum analytical entries for this (eq, pid).
                let ana: f64 = analytical
                    .iter()
                    .filter(|&&(r, p, _)| r == eq && p == pid)
                    .map(|&(_, _, v)| v)
                    .sum();

                let error = (fd - ana).abs();
                assert!(
                    error < tol,
                    "Jacobian mismatch for {:?} at eq={}, param={:?}: \
                     analytical={:.12}, fd={:.12}, error={:.2e}",
                    constraint.name(),
                    eq,
                    pid,
                    ana,
                    fd,
                    error,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // DistancePtPt
    // -----------------------------------------------------------------------

    #[test]
    fn test_distance_pt_pt_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12, "residual should be ~0 for d=5, got {}", r[0]);
    }

    #[test]
    fn test_distance_pt_pt_unsatisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 10.0);
        let r = c.residuals(&store);
        // actual dist^2 = 25, target_sq = 100 -> residual = -75
        assert!((r[0] - (-75.0)).abs() < 1e-12);
    }

    #[test]
    fn test_distance_pt_pt_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e1);
        let y2 = store.alloc(6.0, e1);

        let c = DistancePtPt::new(cid(0), e0, e1, x1, y1, x2, y2, 5.0);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // DistancePtLine
    // -----------------------------------------------------------------------

    #[test]
    fn test_distance_pt_line_satisfied() {
        // Point (0,1), line from (0,0) to (10,0). Distance should be 1.
        let ep = eid(0);
        let el = eid(1);
        let mut store = ParamStore::new();
        let px = store.alloc(5.0, ep);
        let py = store.alloc(1.0, ep);
        let x1 = store.alloc(0.0, el);
        let y1 = store.alloc(0.0, el);
        let x2 = store.alloc(10.0, el);
        let y2 = store.alloc(0.0, el);

        let c = DistancePtLine::new(cid(0), ep, el, px, py, x1, y1, x2, y2, 1.0);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12, "residual = {}", r[0]);
    }

    #[test]
    fn test_distance_pt_line_jacobian() {
        let ep = eid(0);
        let el = eid(1);
        let mut store = ParamStore::new();
        let px = store.alloc(3.0, ep);
        let py = store.alloc(2.0, ep);
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(0.5, el);
        let x2 = store.alloc(7.0, el);
        let y2 = store.alloc(3.0, el);

        let c = DistancePtLine::new(cid(0), ep, el, px, py, x1, y1, x2, y2, 1.0);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Coincident
    // -----------------------------------------------------------------------

    #[test]
    fn test_coincident_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(3.0, e0);
        let y1 = store.alloc(4.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(4.0, e1);

        let c = Coincident::new(cid(0), e0, e1, x1, y1, x2, y2);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-15);
        assert!(r[1].abs() < 1e-15);
    }

    #[test]
    fn test_coincident_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(3.0, e1);
        let y2 = store.alloc(5.0, e1);

        let c = Coincident::new(cid(0), e0, e1, x1, y1, x2, y2);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // TangentLineCircle
    // -----------------------------------------------------------------------

    #[test]
    fn test_tangent_line_circle_satisfied() {
        // Horizontal line y=5, circle at origin radius 5.
        let el = eid(0);
        let ec = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(-10.0, el);
        let y1 = store.alloc(5.0, el);
        let x2 = store.alloc(10.0, el);
        let y2 = store.alloc(5.0, el);
        let cx = store.alloc(0.0, ec);
        let cy = store.alloc(0.0, ec);
        let r = store.alloc(5.0, ec);

        let c = TangentLineCircle::new(cid(0), el, ec, x1, y1, x2, y2, cx, cy, r);
        let res = c.residuals(&store);
        assert!(res[0].abs() < 1e-10, "residual = {}", res[0]);
    }

    #[test]
    fn test_tangent_line_circle_jacobian() {
        let el = eid(0);
        let ec = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(2.0, el);
        let x2 = store.alloc(5.0, el);
        let y2 = store.alloc(4.0, el);
        let cx = store.alloc(3.0, ec);
        let cy = store.alloc(7.0, ec);
        let r = store.alloc(2.0, ec);

        let c = TangentLineCircle::new(cid(0), el, ec, x1, y1, x2, y2, cx, cy, r);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // TangentCircleCircle
    // -----------------------------------------------------------------------

    #[test]
    fn test_tangent_circle_circle_external() {
        // Two circles: center (0,0) r=3, center (5,0) r=2. External tangent: dist=5=3+2.
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let cx1 = store.alloc(0.0, e0);
        let cy1 = store.alloc(0.0, e0);
        let r1 = store.alloc(3.0, e0);
        let cx2 = store.alloc(5.0, e1);
        let cy2 = store.alloc(0.0, e1);
        let r2 = store.alloc(2.0, e1);

        let c = TangentCircleCircle::external(cid(0), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        let res = c.residuals(&store);
        assert!(res[0].abs() < 1e-12, "residual = {}", res[0]);
    }

    #[test]
    fn test_tangent_circle_circle_internal() {
        // Two circles: center (0,0) r=5, center (2,0) r=3. Internal tangent: dist=2=5-3.
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let cx1 = store.alloc(0.0, e0);
        let cy1 = store.alloc(0.0, e0);
        let r1 = store.alloc(5.0, e0);
        let cx2 = store.alloc(2.0, e1);
        let cy2 = store.alloc(0.0, e1);
        let r2 = store.alloc(3.0, e1);

        let c = TangentCircleCircle::internal(cid(0), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        let res = c.residuals(&store);
        assert!(res[0].abs() < 1e-12, "residual = {}", res[0]);
    }

    #[test]
    fn test_tangent_circle_circle_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let cx1 = store.alloc(1.0, e0);
        let cy1 = store.alloc(2.0, e0);
        let r1 = store.alloc(3.0, e0);
        let cx2 = store.alloc(6.0, e1);
        let cy2 = store.alloc(4.0, e1);
        let r2 = store.alloc(2.0, e1);

        let ext = TangentCircleCircle::external(cid(0), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        check_jacobian(&ext, &store, 1e-7, 1e-5);

        let int = TangentCircleCircle::internal(cid(1), e0, e1, cx1, cy1, r1, cx2, cy2, r2);
        check_jacobian(&int, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Parallel
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        // Line 1: (0,0)-(1,2), Line 2: (3,1)-(5,5) => dir (2,4) = 2*(1,2)
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(1.0, e0);
        let y2 = store.alloc(2.0, e0);
        let x3 = store.alloc(3.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(5.0, e1);
        let y4 = store.alloc(5.0, e1);

        let c = Parallel::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12);
    }

    #[test]
    fn test_parallel_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(6.0, e0);
        let x3 = store.alloc(0.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(3.0, e1);
        let y4 = store.alloc(5.0, e1);

        let c = Parallel::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Perpendicular
    // -----------------------------------------------------------------------

    #[test]
    fn test_perpendicular_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        // Line 1: dir (1,0), Line 2: dir (0,1) => dot=0
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(1.0, e0);
        let y2 = store.alloc(0.0, e0);
        let x3 = store.alloc(0.0, e1);
        let y3 = store.alloc(0.0, e1);
        let x4 = store.alloc(0.0, e1);
        let y4 = store.alloc(1.0, e1);

        let c = Perpendicular::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12);
    }

    #[test]
    fn test_perpendicular_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(3.0, e0);
        let x3 = store.alloc(2.0, e1);
        let y3 = store.alloc(0.0, e1);
        let x4 = store.alloc(5.0, e1);
        let y4 = store.alloc(7.0, e1);

        let c = Perpendicular::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Angle
    // -----------------------------------------------------------------------

    #[test]
    fn test_angle_satisfied() {
        let e = eid(0);
        let mut store = ParamStore::new();
        // Line at 45 degrees: (0,0) to (1,1)
        let x1 = store.alloc(0.0, e);
        let y1 = store.alloc(0.0, e);
        let x2 = store.alloc(1.0, e);
        let y2 = store.alloc(1.0, e);

        let c = Angle::new(cid(0), e, x1, y1, x2, y2, std::f64::consts::FRAC_PI_4);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12, "residual = {}", r[0]);
    }

    #[test]
    fn test_angle_jacobian() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e);
        let y1 = store.alloc(2.0, e);
        let x2 = store.alloc(4.0, e);
        let y2 = store.alloc(6.0, e);

        let c = Angle::new(cid(0), e, x1, y1, x2, y2, 0.7);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Horizontal
    // -----------------------------------------------------------------------

    #[test]
    fn test_horizontal_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let y1 = store.alloc(3.0, e0);
        let y2 = store.alloc(3.0, e1);

        let c = Horizontal::new(cid(0), e0, e1, y1, y2);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-15);
    }

    #[test]
    fn test_horizontal_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let y1 = store.alloc(1.0, e0);
        let y2 = store.alloc(5.0, e1);

        let c = Horizontal::new(cid(0), e0, e1, y1, y2);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Vertical
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertical_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(7.0, e0);
        let x2 = store.alloc(7.0, e1);

        let c = Vertical::new(cid(0), e0, e1, x1, x2);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-15);
    }

    #[test]
    fn test_vertical_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(2.0, e0);
        let x2 = store.alloc(8.0, e1);

        let c = Vertical::new(cid(0), e0, e1, x1, x2);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Fixed
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixed_satisfied() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x = store.alloc(3.0, e);
        let y = store.alloc(4.0, e);

        let c = Fixed::new(cid(0), e, x, y, 3.0, 4.0);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-15);
        assert!(r[1].abs() < 1e-15);
    }

    #[test]
    fn test_fixed_jacobian() {
        let e = eid(0);
        let mut store = ParamStore::new();
        let x = store.alloc(1.0, e);
        let y = store.alloc(2.0, e);

        let c = Fixed::new(cid(0), e, x, y, 5.0, 7.0);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Midpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_midpoint_satisfied() {
        let ep = eid(0);
        let el = eid(1);
        let mut store = ParamStore::new();
        let mx = store.alloc(5.0, ep);
        let my = store.alloc(3.0, ep);
        let x1 = store.alloc(2.0, el);
        let y1 = store.alloc(1.0, el);
        let x2 = store.alloc(8.0, el);
        let y2 = store.alloc(5.0, el);

        let c = Midpoint::new(cid(0), ep, el, mx, my, x1, y1, x2, y2);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12, "rx = {}", r[0]);
        assert!(r[1].abs() < 1e-12, "ry = {}", r[1]);
    }

    #[test]
    fn test_midpoint_jacobian() {
        let ep = eid(0);
        let el = eid(1);
        let mut store = ParamStore::new();
        let mx = store.alloc(3.0, ep);
        let my = store.alloc(4.0, ep);
        let x1 = store.alloc(1.0, el);
        let y1 = store.alloc(2.0, el);
        let x2 = store.alloc(7.0, el);
        let y2 = store.alloc(9.0, el);

        let c = Midpoint::new(cid(0), ep, el, mx, my, x1, y1, x2, y2);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Symmetric
    // -----------------------------------------------------------------------

    #[test]
    fn test_symmetric_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let ec = eid(2);
        let mut store = ParamStore::new();
        // Points (1,2) and (5,8), center (3,5)
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(5.0, e1);
        let y2 = store.alloc(8.0, e1);
        let cx = store.alloc(3.0, ec);
        let cy = store.alloc(5.0, ec);

        let c = Symmetric::new(cid(0), e0, e1, ec, x1, y1, x2, y2, cx, cy);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12);
        assert!(r[1].abs() < 1e-12);
    }

    #[test]
    fn test_symmetric_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let ec = eid(2);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(6.0, e1);
        let y2 = store.alloc(9.0, e1);
        let cx = store.alloc(3.0, ec);
        let cy = store.alloc(5.0, ec);

        let c = Symmetric::new(cid(0), e0, e1, ec, x1, y1, x2, y2, cx, cy);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // EqualLength
    // -----------------------------------------------------------------------

    #[test]
    fn test_equal_length_satisfied() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        // Line 1: (0,0)-(3,4) length=5, Line 2: (1,1)-(4,5) length=5
        let x1 = store.alloc(0.0, e0);
        let y1 = store.alloc(0.0, e0);
        let x2 = store.alloc(3.0, e0);
        let y2 = store.alloc(4.0, e0);
        let x3 = store.alloc(1.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(4.0, e1);
        let y4 = store.alloc(5.0, e1);

        let c = EqualLength::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12, "residual = {}", r[0]);
    }

    #[test]
    fn test_equal_length_jacobian() {
        let e0 = eid(0);
        let e1 = eid(1);
        let mut store = ParamStore::new();
        let x1 = store.alloc(1.0, e0);
        let y1 = store.alloc(2.0, e0);
        let x2 = store.alloc(4.0, e0);
        let y2 = store.alloc(6.0, e0);
        let x3 = store.alloc(0.0, e1);
        let y3 = store.alloc(1.0, e1);
        let x4 = store.alloc(3.0, e1);
        let y4 = store.alloc(3.0, e1);

        let c = EqualLength::new(cid(0), e0, e1, x1, y1, x2, y2, x3, y3, x4, y4);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // PointOnCircle
    // -----------------------------------------------------------------------

    #[test]
    fn test_point_on_circle_satisfied() {
        let ep = eid(0);
        let ec = eid(1);
        let mut store = ParamStore::new();
        // Point (3,4) on circle center (0,0) radius 5: 3^2+4^2=25=5^2
        let px = store.alloc(3.0, ep);
        let py = store.alloc(4.0, ep);
        let cx = store.alloc(0.0, ec);
        let cy = store.alloc(0.0, ec);
        let r = store.alloc(5.0, ec);

        let c = PointOnCircle::new(cid(0), ep, ec, px, py, cx, cy, r);
        let res = c.residuals(&store);
        assert!(res[0].abs() < 1e-12);
    }

    #[test]
    fn test_point_on_circle_jacobian() {
        let ep = eid(0);
        let ec = eid(1);
        let mut store = ParamStore::new();
        let px = store.alloc(2.0, ep);
        let py = store.alloc(3.0, ep);
        let cx = store.alloc(1.0, ec);
        let cy = store.alloc(1.0, ec);
        let r = store.alloc(4.0, ec);

        let c = PointOnCircle::new(cid(0), ep, ec, px, py, cx, cy, r);
        check_jacobian(&c, &store, 1e-7, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Constraint trait metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_equation_counts() {
        let e = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();
        let mut p = |v: f64| store.alloc(v, e);

        let a = p(0.0);
        let b = p(0.0);
        let c = p(0.0);
        let d = p(0.0);

        assert_eq!(
            DistancePtPt::new(cid(0), e, e2, a, b, c, d, 1.0).equation_count(),
            1
        );
        assert_eq!(
            Coincident::new(cid(0), e, e2, a, b, c, d).equation_count(),
            2
        );
        assert_eq!(
            Horizontal::new(cid(0), e, e2, a, b).equation_count(),
            1
        );
        assert_eq!(
            Vertical::new(cid(0), e, e2, a, b).equation_count(),
            1
        );
        assert_eq!(
            Fixed::new(cid(0), e, a, b, 0.0, 0.0).equation_count(),
            2
        );
    }

    #[test]
    fn test_constraint_names() {
        let e = eid(0);
        let e2 = eid(1);
        let mut store = ParamStore::new();
        let mut p = |v: f64| store.alloc(v, e);
        let a = p(0.0);
        let b = p(0.0);
        let c_p = p(0.0);
        let d = p(0.0);

        assert_eq!(
            DistancePtPt::new(cid(0), e, e2, a, b, c_p, d, 1.0).name(),
            "DistancePtPt"
        );
        assert_eq!(
            Coincident::new(cid(0), e, e2, a, b, c_p, d).name(),
            "Coincident"
        );
        assert_eq!(Horizontal::new(cid(0), e, e2, a, b).name(), "Horizontal");
        assert_eq!(Vertical::new(cid(0), e, e2, a, b).name(), "Vertical");
        assert_eq!(Fixed::new(cid(0), e, a, b, 0.0, 0.0).name(), "Fixed");
    }

    #[test]
    fn test_constraint_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DistancePtPt>();
        assert_send_sync::<DistancePtLine>();
        assert_send_sync::<Coincident>();
        assert_send_sync::<TangentLineCircle>();
        assert_send_sync::<TangentCircleCircle>();
        assert_send_sync::<Parallel>();
        assert_send_sync::<Perpendicular>();
        assert_send_sync::<Angle>();
        assert_send_sync::<Horizontal>();
        assert_send_sync::<Vertical>();
        assert_send_sync::<Fixed>();
        assert_send_sync::<Midpoint>();
        assert_send_sync::<Symmetric>();
        assert_send_sync::<EqualLength>();
        assert_send_sync::<PointOnCircle>();
    }
}
