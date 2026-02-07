//! 3D entity types implementing the [`Entity`](crate::entity::Entity) trait.
//!
//! Provides geometric primitives for 3D sketching:
//! - [`Point3D`] -- a point in 3D space (3 parameters)
//! - [`LineSegment3D`] -- a line segment between two 3D points (6 parameters)
//! - [`Plane`] -- a plane defined by a point and normal vector (6 parameters)
//! - [`Axis3D`] -- an axis defined by a point and direction vector (6 parameters)

use crate::entity::Entity;
use crate::id::{EntityId, ParamId};
use crate::param::ParamStore;

// ---------------------------------------------------------------------------
// Point3D
// ---------------------------------------------------------------------------

/// A point in 3D space.
///
/// Parameters: `[x, y, z]`.
#[derive(Debug, Clone)]
pub struct Point3D {
    id: EntityId,
    x: ParamId,
    y: ParamId,
    z: ParamId,
    params: [ParamId; 3],
}

impl Point3D {
    /// Create a new 3D point entity.
    pub fn new(id: EntityId, x: ParamId, y: ParamId, z: ParamId) -> Self {
        Self {
            id,
            x,
            y,
            z,
            params: [x, y, z],
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

    /// Parameter ID for the z-coordinate.
    pub fn z(&self) -> ParamId {
        self.z
    }

    /// Read the current (x, y, z) values from the parameter store.
    pub fn get_xyz(&self, store: &ParamStore) -> (f64, f64, f64) {
        (store.get(self.x), store.get(self.y), store.get(self.z))
    }
}

impl Entity for Point3D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "Point3D"
    }
}

// ---------------------------------------------------------------------------
// LineSegment3D
// ---------------------------------------------------------------------------

/// A line segment in 3D space defined by two endpoints.
///
/// Parameters: `[x1, y1, z1, x2, y2, z2]`.
#[derive(Debug, Clone)]
pub struct LineSegment3D {
    id: EntityId,
    x1: ParamId,
    y1: ParamId,
    z1: ParamId,
    x2: ParamId,
    y2: ParamId,
    z2: ParamId,
    params: [ParamId; 6],
}

impl LineSegment3D {
    /// Create a new 3D line segment entity.
    pub fn new(
        id: EntityId,
        x1: ParamId, y1: ParamId, z1: ParamId,
        x2: ParamId, y2: ParamId, z2: ParamId,
    ) -> Self {
        Self {
            id,
            x1, y1, z1,
            x2, y2, z2,
            params: [x1, y1, z1, x2, y2, z2],
        }
    }

    /// Parameter IDs for the first endpoint.
    pub fn start(&self) -> (ParamId, ParamId, ParamId) {
        (self.x1, self.y1, self.z1)
    }

    /// Parameter IDs for the second endpoint.
    pub fn end(&self) -> (ParamId, ParamId, ParamId) {
        (self.x2, self.y2, self.z2)
    }

    /// Read the start point coordinates from the parameter store.
    pub fn get_start(&self, store: &ParamStore) -> (f64, f64, f64) {
        (store.get(self.x1), store.get(self.y1), store.get(self.z1))
    }

    /// Read the end point coordinates from the parameter store.
    pub fn get_end(&self, store: &ParamStore) -> (f64, f64, f64) {
        (store.get(self.x2), store.get(self.y2), store.get(self.z2))
    }

    /// Compute the direction vector (unnormalized) from start to end.
    pub fn direction(&self, store: &ParamStore) -> (f64, f64, f64) {
        let (sx, sy, sz) = self.get_start(store);
        let (ex, ey, ez) = self.get_end(store);
        (ex - sx, ey - sy, ez - sz)
    }
}

impl Entity for LineSegment3D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "LineSegment3D"
    }
}

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------

/// A plane in 3D space defined by a point on the plane and a normal vector.
///
/// Parameters: `[px, py, pz, nx, ny, nz]` where `(px, py, pz)` is a point
/// on the plane and `(nx, ny, nz)` is the plane normal.
#[derive(Debug, Clone)]
pub struct Plane {
    id: EntityId,
    px: ParamId,
    py: ParamId,
    pz: ParamId,
    nx: ParamId,
    ny: ParamId,
    nz: ParamId,
    params: [ParamId; 6],
}

impl Plane {
    /// Create a new plane entity.
    pub fn new(
        id: EntityId,
        px: ParamId, py: ParamId, pz: ParamId,
        nx: ParamId, ny: ParamId, nz: ParamId,
    ) -> Self {
        Self {
            id,
            px, py, pz,
            nx, ny, nz,
            params: [px, py, pz, nx, ny, nz],
        }
    }

    /// Parameter IDs for the point on the plane.
    pub fn point(&self) -> (ParamId, ParamId, ParamId) {
        (self.px, self.py, self.pz)
    }

    /// Parameter IDs for the plane normal.
    pub fn normal(&self) -> (ParamId, ParamId, ParamId) {
        (self.nx, self.ny, self.nz)
    }

    /// Read the point-on-plane coordinates from the parameter store.
    pub fn get_point(&self, store: &ParamStore) -> (f64, f64, f64) {
        (store.get(self.px), store.get(self.py), store.get(self.pz))
    }

    /// Read the normal vector from the parameter store.
    pub fn get_normal(&self, store: &ParamStore) -> (f64, f64, f64) {
        (store.get(self.nx), store.get(self.ny), store.get(self.nz))
    }
}

impl Entity for Plane {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "Plane"
    }
}

// ---------------------------------------------------------------------------
// Axis3D
// ---------------------------------------------------------------------------

/// An axis in 3D space defined by a point on the axis and a direction vector.
///
/// Parameters: `[px, py, pz, dx, dy, dz]` where `(px, py, pz)` is a point
/// on the axis and `(dx, dy, dz)` is the axis direction.
#[derive(Debug, Clone)]
pub struct Axis3D {
    id: EntityId,
    px: ParamId,
    py: ParamId,
    pz: ParamId,
    dx: ParamId,
    dy: ParamId,
    dz: ParamId,
    params: [ParamId; 6],
}

impl Axis3D {
    /// Create a new 3D axis entity.
    pub fn new(
        id: EntityId,
        px: ParamId, py: ParamId, pz: ParamId,
        dx: ParamId, dy: ParamId, dz: ParamId,
    ) -> Self {
        Self {
            id,
            px, py, pz,
            dx, dy, dz,
            params: [px, py, pz, dx, dy, dz],
        }
    }

    /// Parameter IDs for the point on the axis.
    pub fn point(&self) -> (ParamId, ParamId, ParamId) {
        (self.px, self.py, self.pz)
    }

    /// Parameter IDs for the axis direction.
    pub fn direction(&self) -> (ParamId, ParamId, ParamId) {
        (self.dx, self.dy, self.dz)
    }

    /// Read the point-on-axis coordinates from the parameter store.
    pub fn get_point(&self, store: &ParamStore) -> (f64, f64, f64) {
        (store.get(self.px), store.get(self.py), store.get(self.pz))
    }

    /// Read the direction vector from the parameter store.
    pub fn get_direction(&self, store: &ParamStore) -> (f64, f64, f64) {
        (store.get(self.dx), store.get(self.dy), store.get(self.dz))
    }
}

impl Entity for Axis3D {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "Axis3D"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_entity() -> EntityId {
        EntityId::new(0, 0)
    }

    fn make_store_and_point() -> (ParamStore, Point3D) {
        let mut store = ParamStore::new();
        let eid = dummy_entity();
        let x = store.alloc(1.0, eid);
        let y = store.alloc(2.0, eid);
        let z = store.alloc(3.0, eid);
        let pt = Point3D::new(eid, x, y, z);
        (store, pt)
    }

    #[test]
    fn point3d_params() {
        let (_, pt) = make_store_and_point();
        assert_eq!(pt.params().len(), 3);
        assert_eq!(pt.name(), "Point3D");
    }

    #[test]
    fn point3d_get_xyz() {
        let (store, pt) = make_store_and_point();
        let (x, y, z) = pt.get_xyz(&store);
        assert!((x - 1.0).abs() < 1e-15);
        assert!((y - 2.0).abs() < 1e-15);
        assert!((z - 3.0).abs() < 1e-15);
    }

    #[test]
    fn line_segment_3d() {
        let mut store = ParamStore::new();
        let eid = EntityId::new(1, 0);
        let x1 = store.alloc(0.0, eid);
        let y1 = store.alloc(0.0, eid);
        let z1 = store.alloc(0.0, eid);
        let x2 = store.alloc(1.0, eid);
        let y2 = store.alloc(2.0, eid);
        let z2 = store.alloc(3.0, eid);

        let seg = LineSegment3D::new(eid, x1, y1, z1, x2, y2, z2);
        assert_eq!(seg.params().len(), 6);
        assert_eq!(seg.name(), "LineSegment3D");

        let (dx, dy, dz) = seg.direction(&store);
        assert!((dx - 1.0).abs() < 1e-15);
        assert!((dy - 2.0).abs() < 1e-15);
        assert!((dz - 3.0).abs() < 1e-15);
    }

    #[test]
    fn plane_entity() {
        let mut store = ParamStore::new();
        let eid = EntityId::new(2, 0);
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);
        let pz = store.alloc(0.0, eid);
        let nx = store.alloc(0.0, eid);
        let ny = store.alloc(0.0, eid);
        let nz = store.alloc(1.0, eid);

        let plane = Plane::new(eid, px, py, pz, nx, ny, nz);
        assert_eq!(plane.params().len(), 6);
        assert_eq!(plane.name(), "Plane");

        let (nvx, nvy, nvz) = plane.get_normal(&store);
        assert!((nvx).abs() < 1e-15);
        assert!((nvy).abs() < 1e-15);
        assert!((nvz - 1.0).abs() < 1e-15);
    }

    #[test]
    fn axis3d_entity() {
        let mut store = ParamStore::new();
        let eid = EntityId::new(3, 0);
        let px = store.alloc(1.0, eid);
        let py = store.alloc(2.0, eid);
        let pz = store.alloc(3.0, eid);
        let dx = store.alloc(0.0, eid);
        let dy = store.alloc(0.0, eid);
        let dz = store.alloc(1.0, eid);

        let axis = Axis3D::new(eid, px, py, pz, dx, dy, dz);
        assert_eq!(axis.params().len(), 6);
        assert_eq!(axis.name(), "Axis3D");

        let (dvx, dvy, dvz) = axis.get_direction(&store);
        assert!((dvx).abs() < 1e-15);
        assert!((dvy).abs() < 1e-15);
        assert!((dvz - 1.0).abs() < 1e-15);
    }
}
