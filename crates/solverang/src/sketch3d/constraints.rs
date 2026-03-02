//! 3D constraint types implementing the [`Constraint`] trait.
//!
//! Provides geometric constraints for 3D sketch solving:
//! - [`Distance3D`] -- distance between two 3D points
//! - [`Coincident3D`] -- two 3D points at the same location
//! - [`Fixed3D`] -- fix a 3D point at a target position
//! - [`PointOnPlane`] -- constrain a point to lie on a plane
//! - [`Coplanar`] -- multiple points on the same plane
//! - [`Parallel3D`] -- two line segments with parallel directions
//! - [`Perpendicular3D`] -- two line segments with perpendicular directions
//! - [`Coaxial`] -- two axes share the same line

use crate::constraint::Constraint;
use crate::id::{ConstraintId, EntityId, ParamId};
use crate::param::ParamStore;

// ---------------------------------------------------------------------------
// Distance3D
// ---------------------------------------------------------------------------

/// Distance between two 3D points (squared formulation).
///
/// Residual: `(x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2 - d^2`
///
/// Using the squared formulation avoids the square-root singularity at zero
/// distance and simplifies the Jacobian.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Distance3D {
    id: ConstraintId,
    p1: EntityId,
    p2: EntityId,
    distance: f64,
    x1: ParamId, y1: ParamId, z1: ParamId,
    x2: ParamId, y2: ParamId, z2: ParamId,
    params: [ParamId; 6],
    entities: [EntityId; 2],
}

impl Distance3D {
    /// Create a distance constraint between two 3D points.
    ///
    /// `distance` is the target distance (not squared).
    pub fn new(
        id: ConstraintId,
        p1: EntityId, x1: ParamId, y1: ParamId, z1: ParamId,
        p2: EntityId, x2: ParamId, y2: ParamId, z2: ParamId,
        distance: f64,
    ) -> Self {
        Self {
            id,
            p1, p2,
            distance,
            x1, y1, z1,
            x2, y2, z2,
            params: [x1, y1, z1, x2, y2, z2],
            entities: [p1, p2],
        }
    }
}

impl Constraint for Distance3D {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Distance3D" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let (vx1, vy1, vz1) = (store.get(self.x1), store.get(self.y1), store.get(self.z1));
        let (vx2, vy2, vz2) = (store.get(self.x2), store.get(self.y2), store.get(self.z2));
        let dx = vx2 - vx1;
        let dy = vy2 - vy1;
        let dz = vz2 - vz1;
        vec![dx * dx + dy * dy + dz * dz - self.distance * self.distance]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let (vx1, vy1, vz1) = (store.get(self.x1), store.get(self.y1), store.get(self.z1));
        let (vx2, vy2, vz2) = (store.get(self.x2), store.get(self.y2), store.get(self.z2));
        let dx = vx2 - vx1;
        let dy = vy2 - vy1;
        let dz = vz2 - vz1;
        vec![
            (0, self.x1, -2.0 * dx),
            (0, self.y1, -2.0 * dy),
            (0, self.z1, -2.0 * dz),
            (0, self.x2, 2.0 * dx),
            (0, self.y2, 2.0 * dy),
            (0, self.z2, 2.0 * dz),
        ]
    }
}

// ---------------------------------------------------------------------------
// Coincident3D
// ---------------------------------------------------------------------------

/// Two 3D points at the same location.
///
/// Residuals: `[x2-x1, y2-y1, z2-z1]`
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Coincident3D {
    id: ConstraintId,
    p1: EntityId,
    p2: EntityId,
    x1: ParamId, y1: ParamId, z1: ParamId,
    x2: ParamId, y2: ParamId, z2: ParamId,
    params: [ParamId; 6],
    entities: [EntityId; 2],
}

impl Coincident3D {
    /// Create a coincident constraint between two 3D points.
    pub fn new(
        id: ConstraintId,
        p1: EntityId, x1: ParamId, y1: ParamId, z1: ParamId,
        p2: EntityId, x2: ParamId, y2: ParamId, z2: ParamId,
    ) -> Self {
        Self {
            id,
            p1, p2,
            x1, y1, z1,
            x2, y2, z2,
            params: [x1, y1, z1, x2, y2, z2],
            entities: [p1, p2],
        }
    }
}

impl Constraint for Coincident3D {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Coincident3D" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 3 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        vec![
            store.get(self.x2) - store.get(self.x1),
            store.get(self.y2) - store.get(self.y1),
            store.get(self.z2) - store.get(self.z1),
        ]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![
            (0, self.x1, -1.0), (0, self.x2, 1.0),
            (1, self.y1, -1.0), (1, self.y2, 1.0),
            (2, self.z1, -1.0), (2, self.z2, 1.0),
        ]
    }
}

// ---------------------------------------------------------------------------
// Fixed3D
// ---------------------------------------------------------------------------

/// Fix a 3D point at a target position.
///
/// Residuals: `[x - tx, y - ty, z - tz]`
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Fixed3D {
    id: ConstraintId,
    entity: EntityId,
    target: [f64; 3],
    x: ParamId, y: ParamId, z: ParamId,
    params: [ParamId; 3],
    entities: [EntityId; 1],
}

impl Fixed3D {
    /// Create a fixed-position constraint.
    ///
    /// `target` is `[tx, ty, tz]`, the desired world position.
    pub fn new(
        id: ConstraintId,
        entity: EntityId,
        x: ParamId, y: ParamId, z: ParamId,
        target: [f64; 3],
    ) -> Self {
        Self {
            id,
            entity,
            target,
            x, y, z,
            params: [x, y, z],
            entities: [entity],
        }
    }
}

impl Constraint for Fixed3D {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Fixed3D" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 3 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        vec![
            store.get(self.x) - self.target[0],
            store.get(self.y) - self.target[1],
            store.get(self.z) - self.target[2],
        ]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![
            (0, self.x, 1.0),
            (1, self.y, 1.0),
            (2, self.z, 1.0),
        ]
    }
}

// ---------------------------------------------------------------------------
// PointOnPlane
// ---------------------------------------------------------------------------

/// Constrain a point to lie on a plane.
///
/// Residual: `n . (p - p0) = nx*(px-p0x) + ny*(py-p0y) + nz*(pz-p0z)`
///
/// where `(p0x, p0y, p0z)` is a point on the plane and `(nx, ny, nz)` is
/// the plane normal.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PointOnPlane {
    id: ConstraintId,
    point_entity: EntityId,
    plane_entity: EntityId,
    // Point params
    px: ParamId, py: ParamId, pz: ParamId,
    // Plane point params
    p0x: ParamId, p0y: ParamId, p0z: ParamId,
    // Plane normal params
    nx: ParamId, ny: ParamId, nz: ParamId,
    params: Vec<ParamId>,
    entities: [EntityId; 2],
}

impl PointOnPlane {
    /// Create a point-on-plane constraint.
    pub fn new(
        id: ConstraintId,
        point_entity: EntityId,
        px: ParamId, py: ParamId, pz: ParamId,
        plane_entity: EntityId,
        p0x: ParamId, p0y: ParamId, p0z: ParamId,
        nx: ParamId, ny: ParamId, nz: ParamId,
    ) -> Self {
        Self {
            id,
            point_entity,
            plane_entity,
            px, py, pz,
            p0x, p0y, p0z,
            nx, ny, nz,
            params: vec![px, py, pz, p0x, p0y, p0z, nx, ny, nz],
            entities: [point_entity, plane_entity],
        }
    }
}

impl Constraint for PointOnPlane {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "PointOnPlane" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let (vpx, vpy, vpz) = (store.get(self.px), store.get(self.py), store.get(self.pz));
        let (vp0x, vp0y, vp0z) = (store.get(self.p0x), store.get(self.p0y), store.get(self.p0z));
        let (vnx, vny, vnz) = (store.get(self.nx), store.get(self.ny), store.get(self.nz));

        let dx = vpx - vp0x;
        let dy = vpy - vp0y;
        let dz = vpz - vp0z;

        vec![vnx * dx + vny * dy + vnz * dz]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let (vpx, vpy, vpz) = (store.get(self.px), store.get(self.py), store.get(self.pz));
        let (vp0x, vp0y, vp0z) = (store.get(self.p0x), store.get(self.p0y), store.get(self.p0z));
        let (vnx, vny, vnz) = (store.get(self.nx), store.get(self.ny), store.get(self.nz));

        let dx = vpx - vp0x;
        let dy = vpy - vp0y;
        let dz = vpz - vp0z;

        vec![
            // d/d(px) = nx, d/d(py) = ny, d/d(pz) = nz
            (0, self.px, vnx),
            (0, self.py, vny),
            (0, self.pz, vnz),
            // d/d(p0x) = -nx, d/d(p0y) = -ny, d/d(p0z) = -nz
            (0, self.p0x, -vnx),
            (0, self.p0y, -vny),
            (0, self.p0z, -vnz),
            // d/d(nx) = dx, d/d(ny) = dy, d/d(nz) = dz
            (0, self.nx, dx),
            (0, self.ny, dy),
            (0, self.nz, dz),
        ]
    }
}

// ---------------------------------------------------------------------------
// Coplanar
// ---------------------------------------------------------------------------

/// Multiple points constrained to lie on the same plane.
///
/// For each point `pi`, the residual is `n . (pi - p0) = 0` where `p0` is
/// the plane reference point and `n` is the plane normal.
///
/// This produces one equation per point.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Coplanar {
    id: ConstraintId,
    plane_entity: EntityId,
    // Plane point and normal
    p0x: ParamId, p0y: ParamId, p0z: ParamId,
    nx: ParamId, ny: ParamId, nz: ParamId,
    // Point entities and their coordinates
    point_entities: Vec<EntityId>,
    point_params: Vec<(ParamId, ParamId, ParamId)>,
    all_params: Vec<ParamId>,
    all_entities: Vec<EntityId>,
}

impl Coplanar {
    /// Create a coplanar constraint.
    ///
    /// `points` is a slice of `(entity_id, px, py, pz)` tuples.
    pub fn new(
        id: ConstraintId,
        plane_entity: EntityId,
        p0x: ParamId, p0y: ParamId, p0z: ParamId,
        nx: ParamId, ny: ParamId, nz: ParamId,
        points: &[(EntityId, ParamId, ParamId, ParamId)],
    ) -> Self {
        let mut all_params = vec![p0x, p0y, p0z, nx, ny, nz];
        let mut all_entities = vec![plane_entity];
        let mut point_entities = Vec::new();
        let mut point_params = Vec::new();

        for &(eid, px, py, pz) in points {
            point_entities.push(eid);
            point_params.push((px, py, pz));
            all_params.extend_from_slice(&[px, py, pz]);
            all_entities.push(eid);
        }

        Self {
            id,
            plane_entity,
            p0x, p0y, p0z,
            nx, ny, nz,
            point_entities,
            point_params,
            all_params,
            all_entities,
        }
    }
}

impl Constraint for Coplanar {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Coplanar" }
    fn entity_ids(&self) -> &[EntityId] { &self.all_entities }
    fn param_ids(&self) -> &[ParamId] { &self.all_params }
    fn equation_count(&self) -> usize { self.point_params.len() }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let (vp0x, vp0y, vp0z) = (store.get(self.p0x), store.get(self.p0y), store.get(self.p0z));
        let (vnx, vny, vnz) = (store.get(self.nx), store.get(self.ny), store.get(self.nz));

        self.point_params.iter().map(|&(px, py, pz)| {
            let dx = store.get(px) - vp0x;
            let dy = store.get(py) - vp0y;
            let dz = store.get(pz) - vp0z;
            vnx * dx + vny * dy + vnz * dz
        }).collect()
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let (vp0x, vp0y, vp0z) = (store.get(self.p0x), store.get(self.p0y), store.get(self.p0z));
        let (vnx, vny, vnz) = (store.get(self.nx), store.get(self.ny), store.get(self.nz));

        let mut jac = Vec::new();

        for (row, &(px, py, pz)) in self.point_params.iter().enumerate() {
            let dx = store.get(px) - vp0x;
            let dy = store.get(py) - vp0y;
            let dz = store.get(pz) - vp0z;

            // d/d(pi) = n
            jac.push((row, px, vnx));
            jac.push((row, py, vny));
            jac.push((row, pz, vnz));

            // d/d(p0) = -n
            jac.push((row, self.p0x, -vnx));
            jac.push((row, self.p0y, -vny));
            jac.push((row, self.p0z, -vnz));

            // d/d(n) = (pi - p0)
            jac.push((row, self.nx, dx));
            jac.push((row, self.ny, dy));
            jac.push((row, self.nz, dz));
        }

        jac
    }
}

// ---------------------------------------------------------------------------
// Parallel3D
// ---------------------------------------------------------------------------

/// Two line segments with parallel directions in 3D.
///
/// Uses the cross product formulation. For directions `d1` and `d2`, parallel
/// means `d1 x d2 = 0`. The cross product has 3 components but only rank 2,
/// so we use 2 independent equations by selecting two components of the cross
/// product.
///
/// Residuals (2 equations):
/// - `d1y*d2z - d1z*d2y`
/// - `d1z*d2x - d1x*d2z`
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Parallel3D {
    id: ConstraintId,
    line1: EntityId,
    line2: EntityId,
    // Direction of line 1: (x2-x1, y2-y1, z2-z1) via endpoint params
    l1_x1: ParamId, l1_y1: ParamId, l1_z1: ParamId,
    l1_x2: ParamId, l1_y2: ParamId, l1_z2: ParamId,
    // Direction of line 2
    l2_x1: ParamId, l2_y1: ParamId, l2_z1: ParamId,
    l2_x2: ParamId, l2_y2: ParamId, l2_z2: ParamId,
    params: [ParamId; 12],
    entities: [EntityId; 2],
}

impl Parallel3D {
    /// Create a parallel constraint between two 3D line segments.
    pub fn new(
        id: ConstraintId,
        line1: EntityId,
        l1_x1: ParamId, l1_y1: ParamId, l1_z1: ParamId,
        l1_x2: ParamId, l1_y2: ParamId, l1_z2: ParamId,
        line2: EntityId,
        l2_x1: ParamId, l2_y1: ParamId, l2_z1: ParamId,
        l2_x2: ParamId, l2_y2: ParamId, l2_z2: ParamId,
    ) -> Self {
        Self {
            id,
            line1, line2,
            l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
            params: [
                l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
                l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
            ],
            entities: [line1, line2],
        }
    }

    /// Compute the direction vectors from the parameter store.
    fn directions(&self, store: &ParamStore) -> ([f64; 3], [f64; 3]) {
        let d1 = [
            store.get(self.l1_x2) - store.get(self.l1_x1),
            store.get(self.l1_y2) - store.get(self.l1_y1),
            store.get(self.l1_z2) - store.get(self.l1_z1),
        ];
        let d2 = [
            store.get(self.l2_x2) - store.get(self.l2_x1),
            store.get(self.l2_y2) - store.get(self.l2_y1),
            store.get(self.l2_z2) - store.get(self.l2_z1),
        ];
        (d1, d2)
    }
}

impl Constraint for Parallel3D {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Parallel3D" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 2 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let (d1, d2) = self.directions(store);
        // Cross product components:
        // cx = d1y*d2z - d1z*d2y
        // cy = d1z*d2x - d1x*d2z
        // cz = d1x*d2y - d1y*d2x  (dependent, not used)
        vec![
            d1[1] * d2[2] - d1[2] * d2[1],  // cx
            d1[2] * d2[0] - d1[0] * d2[2],  // cy
        ]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let (d1, d2) = self.directions(store);

        // Residual 0: r0 = d1y*d2z - d1z*d2y
        // d1 = (l1_x2 - l1_x1, l1_y2 - l1_y1, l1_z2 - l1_z1)
        // d2 = (l2_x2 - l2_x1, l2_y2 - l2_y1, l2_z2 - l2_z1)
        //
        // dr0/d(d1y) = d2z  =>  dr0/d(l1_y2) = d2z,  dr0/d(l1_y1) = -d2z
        // dr0/d(d1z) = -d2y =>  dr0/d(l1_z2) = -d2y, dr0/d(l1_z1) = d2y
        // dr0/d(d2z) = d1y  =>  dr0/d(l2_z2) = d1y,  dr0/d(l2_z1) = -d1y
        // dr0/d(d2y) = -d1z =>  dr0/d(l2_y2) = -d1z, dr0/d(l2_y1) = d1z

        // Residual 1: r1 = d1z*d2x - d1x*d2z
        // dr1/d(d1z) = d2x  =>  dr1/d(l1_z2) = d2x,  dr1/d(l1_z1) = -d2x
        // dr1/d(d1x) = -d2z =>  dr1/d(l1_x2) = -d2z, dr1/d(l1_x1) = d2z
        // dr1/d(d2x) = d1z  =>  dr1/d(l2_x2) = d1z,  dr1/d(l2_x1) = -d1z
        // dr1/d(d2z) = -d1x =>  dr1/d(l2_z2) = -d1x, dr1/d(l2_z1) = d1x

        vec![
            // Row 0: d1y*d2z - d1z*d2y
            (0, self.l1_y1, -d2[2]), (0, self.l1_y2, d2[2]),
            (0, self.l1_z1, d2[1]),  (0, self.l1_z2, -d2[1]),
            (0, self.l2_y1, d1[2]),  (0, self.l2_y2, -d1[2]),
            (0, self.l2_z1, -d1[1]), (0, self.l2_z2, d1[1]),

            // Row 1: d1z*d2x - d1x*d2z
            (1, self.l1_z1, -d2[0]), (1, self.l1_z2, d2[0]),
            (1, self.l1_x1, d2[2]),  (1, self.l1_x2, -d2[2]),
            (1, self.l2_x1, -d1[2]), (1, self.l2_x2, d1[2]),
            (1, self.l2_z1, d1[0]),  (1, self.l2_z2, -d1[0]),
        ]
    }
}

// ---------------------------------------------------------------------------
// Perpendicular3D
// ---------------------------------------------------------------------------

/// Two line segments with perpendicular directions in 3D.
///
/// Residual: `d1 . d2 = d1x*d2x + d1y*d2y + d1z*d2z = 0`
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Perpendicular3D {
    id: ConstraintId,
    line1: EntityId,
    line2: EntityId,
    l1_x1: ParamId, l1_y1: ParamId, l1_z1: ParamId,
    l1_x2: ParamId, l1_y2: ParamId, l1_z2: ParamId,
    l2_x1: ParamId, l2_y1: ParamId, l2_z1: ParamId,
    l2_x2: ParamId, l2_y2: ParamId, l2_z2: ParamId,
    params: [ParamId; 12],
    entities: [EntityId; 2],
}

impl Perpendicular3D {
    /// Create a perpendicular constraint between two 3D line segments.
    pub fn new(
        id: ConstraintId,
        line1: EntityId,
        l1_x1: ParamId, l1_y1: ParamId, l1_z1: ParamId,
        l1_x2: ParamId, l1_y2: ParamId, l1_z2: ParamId,
        line2: EntityId,
        l2_x1: ParamId, l2_y1: ParamId, l2_z1: ParamId,
        l2_x2: ParamId, l2_y2: ParamId, l2_z2: ParamId,
    ) -> Self {
        Self {
            id,
            line1, line2,
            l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
            params: [
                l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
                l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
            ],
            entities: [line1, line2],
        }
    }

    /// Compute the direction vectors from the parameter store.
    fn directions(&self, store: &ParamStore) -> ([f64; 3], [f64; 3]) {
        let d1 = [
            store.get(self.l1_x2) - store.get(self.l1_x1),
            store.get(self.l1_y2) - store.get(self.l1_y1),
            store.get(self.l1_z2) - store.get(self.l1_z1),
        ];
        let d2 = [
            store.get(self.l2_x2) - store.get(self.l2_x1),
            store.get(self.l2_y2) - store.get(self.l2_y1),
            store.get(self.l2_z2) - store.get(self.l2_z1),
        ];
        (d1, d2)
    }
}

impl Constraint for Perpendicular3D {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Perpendicular3D" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let (d1, d2) = self.directions(store);
        vec![d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let (d1, d2) = self.directions(store);

        // r = d1x*d2x + d1y*d2y + d1z*d2z
        // dr/d(d1x) = d2x => dr/d(l1_x2) = d2x, dr/d(l1_x1) = -d2x
        // dr/d(d1y) = d2y => dr/d(l1_y2) = d2y, dr/d(l1_y1) = -d2y
        // dr/d(d1z) = d2z => dr/d(l1_z2) = d2z, dr/d(l1_z1) = -d2z
        // dr/d(d2x) = d1x => dr/d(l2_x2) = d1x, dr/d(l2_x1) = -d1x
        // etc.
        vec![
            (0, self.l1_x1, -d2[0]), (0, self.l1_x2, d2[0]),
            (0, self.l1_y1, -d2[1]), (0, self.l1_y2, d2[1]),
            (0, self.l1_z1, -d2[2]), (0, self.l1_z2, d2[2]),
            (0, self.l2_x1, -d1[0]), (0, self.l2_x2, d1[0]),
            (0, self.l2_y1, -d1[1]), (0, self.l2_y2, d1[1]),
            (0, self.l2_z1, -d1[2]), (0, self.l2_z2, d1[2]),
        ]
    }
}

// ---------------------------------------------------------------------------
// Coaxial
// ---------------------------------------------------------------------------

/// Two axes share the same line in 3D.
///
/// This is enforced by:
/// 1. Direction cross product = 0 (parallel directions, 2 equations)
/// 2. The vector between the two axis points is parallel to the axis direction
///    (2 equations)
///
/// Total: 4 equations (but the system has rank at most 4).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Coaxial {
    id: ConstraintId,
    axis1: EntityId,
    axis2: EntityId,
    // Axis 1: point (p1x, p1y, p1z), direction (d1x, d1y, d1z)
    p1x: ParamId, p1y: ParamId, p1z: ParamId,
    d1x: ParamId, d1y: ParamId, d1z: ParamId,
    // Axis 2: point (p2x, p2y, p2z), direction (d2x, d2y, d2z)
    p2x: ParamId, p2y: ParamId, p2z: ParamId,
    d2x: ParamId, d2y: ParamId, d2z: ParamId,
    params: [ParamId; 12],
    entities: [EntityId; 2],
}

impl Coaxial {
    /// Create a coaxial constraint between two 3D axes.
    pub fn new(
        id: ConstraintId,
        axis1: EntityId,
        p1x: ParamId, p1y: ParamId, p1z: ParamId,
        d1x: ParamId, d1y: ParamId, d1z: ParamId,
        axis2: EntityId,
        p2x: ParamId, p2y: ParamId, p2z: ParamId,
        d2x: ParamId, d2y: ParamId, d2z: ParamId,
    ) -> Self {
        Self {
            id,
            axis1, axis2,
            p1x, p1y, p1z, d1x, d1y, d1z,
            p2x, p2y, p2z, d2x, d2y, d2z,
            params: [
                p1x, p1y, p1z, d1x, d1y, d1z,
                p2x, p2y, p2z, d2x, d2y, d2z,
            ],
            entities: [axis1, axis2],
        }
    }
}

impl Constraint for Coaxial {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Coaxial" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 4 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let d1 = [store.get(self.d1x), store.get(self.d1y), store.get(self.d1z)];
        let d2 = [store.get(self.d2x), store.get(self.d2y), store.get(self.d2z)];

        // Direction parallelism: d1 x d2 (take 2 components)
        let cross_x = d1[1] * d2[2] - d1[2] * d2[1];
        let cross_y = d1[2] * d2[0] - d1[0] * d2[2];

        // Point-on-axis: (p2 - p1) x d1 = 0 (take 2 components)
        let dp = [
            store.get(self.p2x) - store.get(self.p1x),
            store.get(self.p2y) - store.get(self.p1y),
            store.get(self.p2z) - store.get(self.p1z),
        ];
        let pcross_x = dp[1] * d1[2] - dp[2] * d1[1];
        let pcross_y = dp[2] * d1[0] - dp[0] * d1[2];

        vec![cross_x, cross_y, pcross_x, pcross_y]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let d1 = [store.get(self.d1x), store.get(self.d1y), store.get(self.d1z)];
        let d2 = [store.get(self.d2x), store.get(self.d2y), store.get(self.d2z)];
        let dp = [
            store.get(self.p2x) - store.get(self.p1x),
            store.get(self.p2y) - store.get(self.p1y),
            store.get(self.p2z) - store.get(self.p1z),
        ];

        let mut jac = Vec::new();

        // Row 0: cross_x = d1y*d2z - d1z*d2y
        jac.push((0, self.d1y, d2[2]));
        jac.push((0, self.d1z, -d2[1]));
        jac.push((0, self.d2z, d1[1]));
        jac.push((0, self.d2y, -d1[2]));

        // Row 1: cross_y = d1z*d2x - d1x*d2z
        jac.push((1, self.d1z, d2[0]));
        jac.push((1, self.d1x, -d2[2]));
        jac.push((1, self.d2x, d1[2]));
        jac.push((1, self.d2z, -d1[0]));

        // Row 2: pcross_x = dp_y*d1z - dp_z*d1y
        // dp_y = p2y - p1y, dp_z = p2z - p1z
        jac.push((2, self.p2y, d1[2]));
        jac.push((2, self.p1y, -d1[2]));
        jac.push((2, self.p2z, -d1[1]));
        jac.push((2, self.p1z, d1[1]));
        jac.push((2, self.d1z, dp[1]));
        jac.push((2, self.d1y, -dp[2]));

        // Row 3: pcross_y = dp_z*d1x - dp_x*d1z
        jac.push((3, self.p2z, d1[0]));
        jac.push((3, self.p1z, -d1[0]));
        jac.push((3, self.p2x, -d1[2]));
        jac.push((3, self.p1x, d1[2]));
        jac.push((3, self.d1x, dp[2]));
        jac.push((3, self.d1z, -dp[0]));

        jac
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::EntityId;

    fn eid(i: u32) -> EntityId {
        EntityId::new(i, 0)
    }

    fn cid(i: u32) -> ConstraintId {
        ConstraintId::new(i, 0)
    }

    /// Helper to verify Jacobian via finite differences.
    fn verify_jacobian_fd(
        constraint: &dyn Constraint,
        store: &mut ParamStore,
        eps: f64,
        tol: f64,
    ) {
        let _params: Vec<ParamId> = constraint.param_ids().to_vec();
        let analytic = constraint.jacobian(store);
        let n_eq = constraint.equation_count();

        for &(row, pid, analytic_val) in &analytic {
            assert!(row < n_eq, "Jacobian row {} out of range (count={})", row, n_eq);

            let orig = store.get(pid);

            store.set(pid, orig + eps);
            let r_plus = constraint.residuals(store);

            store.set(pid, orig - eps);
            let r_minus = constraint.residuals(store);

            store.set(pid, orig);

            let fd = (r_plus[row] - r_minus[row]) / (2.0 * eps);
            let err = (analytic_val - fd).abs();
            assert!(
                err < tol,
                "Jacobian mismatch for {:?} row {}: analytic={}, fd={}, err={}",
                pid, row, analytic_val, fd, err,
            );
        }
    }

    // -- Distance3D tests --

    #[test]
    fn distance3d_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(0.0, e1);
        let y1 = store.alloc(0.0, e1);
        let z1 = store.alloc(0.0, e1);
        let x2 = store.alloc(3.0, e2);
        let y2 = store.alloc(4.0, e2);
        let z2 = store.alloc(0.0, e2);

        let c = Distance3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2, 5.0);
        let r = c.residuals(&store);
        assert!((r[0]).abs() < 1e-12, "Expected zero residual, got {}", r[0]);
    }

    #[test]
    fn distance3d_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(2.0, e1);
        let z1 = store.alloc(3.0, e1);
        let x2 = store.alloc(4.0, e2);
        let y2 = store.alloc(6.0, e2);
        let z2 = store.alloc(3.0, e2);

        let c = Distance3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2, 5.0);
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }

    // -- Coincident3D tests --

    #[test]
    fn coincident3d_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(2.0, e1);
        let z1 = store.alloc(3.0, e1);
        let x2 = store.alloc(1.0, e2);
        let y2 = store.alloc(2.0, e2);
        let z2 = store.alloc(3.0, e2);

        let c = Coincident3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2);
        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn coincident3d_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);
        let x1 = store.alloc(1.0, e1);
        let y1 = store.alloc(2.0, e1);
        let z1 = store.alloc(3.5, e1);
        let x2 = store.alloc(4.0, e2);
        let y2 = store.alloc(5.0, e2);
        let z2 = store.alloc(6.0, e2);

        let c = Coincident3D::new(cid(0), e1, x1, y1, z1, e2, x2, y2, z2);
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }

    // -- Fixed3D tests --

    #[test]
    fn fixed3d_satisfied() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(1.0, e);
        let y = store.alloc(2.0, e);
        let z = store.alloc(3.0, e);

        let c = Fixed3D::new(cid(0), e, x, y, z, [1.0, 2.0, 3.0]);
        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn fixed3d_jacobian_fd() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let x = store.alloc(1.5, e);
        let y = store.alloc(2.5, e);
        let z = store.alloc(3.5, e);

        let c = Fixed3D::new(cid(0), e, x, y, z, [1.0, 2.0, 3.0]);
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }

    // -- PointOnPlane tests --

    #[test]
    fn point_on_plane_satisfied() {
        let mut store = ParamStore::new();
        let pe = eid(0);
        let ple = eid(1);

        // Point at (1, 2, 0), plane z=0 (normal (0,0,1), point (0,0,0))
        let px = store.alloc(1.0, pe);
        let py = store.alloc(2.0, pe);
        let pz = store.alloc(0.0, pe);
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.0, ple);
        let ny = store.alloc(0.0, ple);
        let nz = store.alloc(1.0, ple);

        let c = PointOnPlane::new(cid(0), pe, px, py, pz, ple, p0x, p0y, p0z, nx, ny, nz);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-15);
    }

    #[test]
    fn point_on_plane_jacobian_fd() {
        let mut store = ParamStore::new();
        let pe = eid(0);
        let ple = eid(1);

        let px = store.alloc(1.0, pe);
        let py = store.alloc(2.0, pe);
        let pz = store.alloc(0.5, pe);
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.0, ple);
        let ny = store.alloc(0.0, ple);
        let nz = store.alloc(1.0, ple);

        let c = PointOnPlane::new(cid(0), pe, px, py, pz, ple, p0x, p0y, p0z, nx, ny, nz);
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }

    // -- Parallel3D tests --

    #[test]
    fn parallel3d_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        // Line 1: (0,0,0) -> (1,0,0), direction (1,0,0)
        let l1_x1 = store.alloc(0.0, e1);
        let l1_y1 = store.alloc(0.0, e1);
        let l1_z1 = store.alloc(0.0, e1);
        let l1_x2 = store.alloc(1.0, e1);
        let l1_y2 = store.alloc(0.0, e1);
        let l1_z2 = store.alloc(0.0, e1);
        // Line 2: (0,1,0) -> (2,1,0), direction (2,0,0) -- parallel to line 1
        let l2_x1 = store.alloc(0.0, e2);
        let l2_y1 = store.alloc(1.0, e2);
        let l2_z1 = store.alloc(0.0, e2);
        let l2_x2 = store.alloc(2.0, e2);
        let l2_y2 = store.alloc(1.0, e2);
        let l2_z2 = store.alloc(0.0, e2);

        let c = Parallel3D::new(
            cid(0), e1, l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            e2, l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
        );
        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn parallel3d_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let l1_x1 = store.alloc(0.0, e1);
        let l1_y1 = store.alloc(0.0, e1);
        let l1_z1 = store.alloc(0.0, e1);
        let l1_x2 = store.alloc(1.0, e1);
        let l1_y2 = store.alloc(0.5, e1);
        let l1_z2 = store.alloc(0.3, e1);
        let l2_x1 = store.alloc(0.0, e2);
        let l2_y1 = store.alloc(1.0, e2);
        let l2_z1 = store.alloc(0.0, e2);
        let l2_x2 = store.alloc(2.0, e2);
        let l2_y2 = store.alloc(1.5, e2);
        let l2_z2 = store.alloc(0.7, e2);

        let c = Parallel3D::new(
            cid(0), e1, l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            e2, l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
        );
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }

    // -- Perpendicular3D tests --

    #[test]
    fn perpendicular3d_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        // Line 1: direction (1,0,0), Line 2: direction (0,1,0)
        let l1_x1 = store.alloc(0.0, e1);
        let l1_y1 = store.alloc(0.0, e1);
        let l1_z1 = store.alloc(0.0, e1);
        let l1_x2 = store.alloc(1.0, e1);
        let l1_y2 = store.alloc(0.0, e1);
        let l1_z2 = store.alloc(0.0, e1);
        let l2_x1 = store.alloc(0.0, e2);
        let l2_y1 = store.alloc(0.0, e2);
        let l2_z1 = store.alloc(0.0, e2);
        let l2_x2 = store.alloc(0.0, e2);
        let l2_y2 = store.alloc(1.0, e2);
        let l2_z2 = store.alloc(0.0, e2);

        let c = Perpendicular3D::new(
            cid(0), e1, l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            e2, l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
        );
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-15);
    }

    #[test]
    fn perpendicular3d_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let l1_x1 = store.alloc(0.0, e1);
        let l1_y1 = store.alloc(0.0, e1);
        let l1_z1 = store.alloc(0.0, e1);
        let l1_x2 = store.alloc(1.0, e1);
        let l1_y2 = store.alloc(0.3, e1);
        let l1_z2 = store.alloc(0.0, e1);
        let l2_x1 = store.alloc(0.0, e2);
        let l2_y1 = store.alloc(0.0, e2);
        let l2_z1 = store.alloc(0.0, e2);
        let l2_x2 = store.alloc(-0.3, e2);
        let l2_y2 = store.alloc(1.0, e2);
        let l2_z2 = store.alloc(0.5, e2);

        let c = Perpendicular3D::new(
            cid(0), e1, l1_x1, l1_y1, l1_z1, l1_x2, l1_y2, l1_z2,
            e2, l2_x1, l2_y1, l2_z1, l2_x2, l2_y2, l2_z2,
        );
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }

    // -- Coaxial tests --

    #[test]
    fn coaxial_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        // Axis 1: point (0,0,0), direction (1,0,0)
        let p1x = store.alloc(0.0, e1);
        let p1y = store.alloc(0.0, e1);
        let p1z = store.alloc(0.0, e1);
        let d1x = store.alloc(1.0, e1);
        let d1y = store.alloc(0.0, e1);
        let d1z = store.alloc(0.0, e1);
        // Axis 2: point (5,0,0) on same line, direction (2,0,0) -- parallel
        let p2x = store.alloc(5.0, e2);
        let p2y = store.alloc(0.0, e2);
        let p2z = store.alloc(0.0, e2);
        let d2x = store.alloc(2.0, e2);
        let d2y = store.alloc(0.0, e2);
        let d2z = store.alloc(0.0, e2);

        let c = Coaxial::new(
            cid(0), e1, p1x, p1y, p1z, d1x, d1y, d1z,
            e2, p2x, p2y, p2z, d2x, d2y, d2z,
        );
        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn coaxial_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let p1x = store.alloc(0.0, e1);
        let p1y = store.alloc(0.0, e1);
        let p1z = store.alloc(0.0, e1);
        let d1x = store.alloc(1.0, e1);
        let d1y = store.alloc(0.2, e1);
        let d1z = store.alloc(0.3, e1);
        let p2x = store.alloc(5.0, e2);
        let p2y = store.alloc(1.0, e2);
        let p2z = store.alloc(1.5, e2);
        let d2x = store.alloc(2.0, e2);
        let d2y = store.alloc(0.5, e2);
        let d2z = store.alloc(0.7, e2);

        let c = Coaxial::new(
            cid(0), e1, p1x, p1y, p1z, d1x, d1y, d1z,
            e2, p2x, p2y, p2z, d2x, d2y, d2z,
        );
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }

    // -- Coplanar tests --

    #[test]
    fn coplanar_satisfied() {
        let mut store = ParamStore::new();
        let ple = eid(0);
        let pe1 = eid(1);
        let pe2 = eid(2);

        // Plane z=0
        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.0, ple);
        let ny = store.alloc(0.0, ple);
        let nz = store.alloc(1.0, ple);

        // Two points on z=0
        let px1 = store.alloc(1.0, pe1);
        let py1 = store.alloc(2.0, pe1);
        let pz1 = store.alloc(0.0, pe1);
        let px2 = store.alloc(3.0, pe2);
        let py2 = store.alloc(4.0, pe2);
        let pz2 = store.alloc(0.0, pe2);

        let c = Coplanar::new(
            cid(0), ple, p0x, p0y, p0z, nx, ny, nz,
            &[(pe1, px1, py1, pz1), (pe2, px2, py2, pz2)],
        );
        let r = c.residuals(&store);
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn coplanar_jacobian_fd() {
        let mut store = ParamStore::new();
        let ple = eid(0);
        let pe1 = eid(1);

        let p0x = store.alloc(0.0, ple);
        let p0y = store.alloc(0.0, ple);
        let p0z = store.alloc(0.0, ple);
        let nx = store.alloc(0.3, ple);
        let ny = store.alloc(0.5, ple);
        let nz = store.alloc(1.0, ple);

        let px1 = store.alloc(1.0, pe1);
        let py1 = store.alloc(2.0, pe1);
        let pz1 = store.alloc(0.5, pe1);

        let c = Coplanar::new(
            cid(0), ple, p0x, p0y, p0z, nx, ny, nz,
            &[(pe1, px1, py1, pz1)],
        );
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }
}
