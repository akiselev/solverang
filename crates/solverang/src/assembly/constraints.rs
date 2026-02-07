//! Assembly constraint types implementing the [`Constraint`](crate::constraint::Constraint) trait.
//!
//! - [`Mate`] -- point-on-body1 coincides with point-on-body2
//! - [`CoaxialAssembly`] -- two body-local axes are collinear in world space
//! - [`Insert`] -- coaxial + axial mate (pin-in-hole)
//! - [`Gear`] -- rotation ratio between two bodies about their respective axes

use crate::constraint::Constraint;
use crate::id::{ConstraintId, EntityId, ParamId};
use crate::param::ParamStore;

use super::entities::{quat_rotate_derivatives, quat_to_rotation_matrix};

// ---------------------------------------------------------------------------
// Helper: transform a local point to world and produce Jacobian entries
// ---------------------------------------------------------------------------

/// Compute `world = R(q)*local + t` and return the world point.
fn transform_local(store: &ParamStore, t: [ParamId; 3], q: [ParamId; 4], local: [f64; 3]) -> [f64; 3] {
    let w = store.get(q[0]);
    let x = store.get(q[1]);
    let y = store.get(q[2]);
    let z = store.get(q[3]);
    let r = quat_to_rotation_matrix(w, x, y, z);
    let tv = [store.get(t[0]), store.get(t[1]), store.get(t[2])];
    [
        r[0][0] * local[0] + r[0][1] * local[1] + r[0][2] * local[2] + tv[0],
        r[1][0] * local[0] + r[1][1] * local[1] + r[1][2] * local[2] + tv[1],
        r[2][0] * local[0] + r[2][1] * local[1] + r[2][2] * local[2] + tv[2],
    ]
}

/// Append Jacobian entries for `world_i = R(q)*local + t` for a single body.
///
/// Pushes entries `(residual_row_offset + i, param, value)` for i in 0..3
/// into the provided `entries` vec.
/// `sign` is +1 or -1 depending on whether this body contributes positively
/// or negatively to the residual.
fn transform_jacobian_entries(
    store: &ParamStore,
    row_offset: usize,
    sign: f64,
    t: [ParamId; 3],
    q: [ParamId; 4],
    local: [f64; 3],
    entries: &mut Vec<(usize, ParamId, f64)>,
) {
    let w = store.get(q[0]);
    let x = store.get(q[1]);
    let y = store.get(q[2]);
    let z = store.get(q[3]);

    let drvdq = quat_rotate_derivatives(w, x, y, z, local);

    for i in 0..3 {
        // d(world_i)/d(t_i) = 1
        entries.push((row_offset + i, t[i], sign));

        // d(world_i)/d(qw, qx, qy, qz) = dRv/dq[j][i]
        for j in 0..4 {
            entries.push((row_offset + i, q[j], sign * drvdq[j][i]));
        }
    }
}

/// Rotate a local direction vector by quaternion (no translation).
fn rotate_direction(store: &ParamStore, q: [ParamId; 4], dir: [f64; 3]) -> [f64; 3] {
    let w = store.get(q[0]);
    let x = store.get(q[1]);
    let y = store.get(q[2]);
    let z = store.get(q[3]);
    let r = quat_to_rotation_matrix(w, x, y, z);
    [
        r[0][0] * dir[0] + r[0][1] * dir[1] + r[0][2] * dir[2],
        r[1][0] * dir[0] + r[1][1] * dir[1] + r[1][2] * dir[2],
        r[2][0] * dir[0] + r[2][1] * dir[1] + r[2][2] * dir[2],
    ]
}

// ---------------------------------------------------------------------------
// Mate
// ---------------------------------------------------------------------------

/// Mate constraint: a point on body1 coincides with a point on body2.
///
/// The points are specified in body-local coordinates (constants, not parameters).
///
/// Residuals (3 equations):
/// ```text
/// R1*local1 + t1 - R2*local2 - t2 = 0
/// ```
#[derive(Debug, Clone)]
pub struct Mate {
    id: ConstraintId,
    body1: EntityId,
    body2: EntityId,
    local1: [f64; 3],
    local2: [f64; 3],
    params: Vec<ParamId>,
    entities: [EntityId; 2],
    // Body 1 params
    b1_tx: ParamId, b1_ty: ParamId, b1_tz: ParamId,
    b1_qw: ParamId, b1_qx: ParamId, b1_qy: ParamId, b1_qz: ParamId,
    // Body 2 params
    b2_tx: ParamId, b2_ty: ParamId, b2_tz: ParamId,
    b2_qw: ParamId, b2_qx: ParamId, b2_qy: ParamId, b2_qz: ParamId,
}

impl Mate {
    /// Create a mate constraint.
    ///
    /// `local1` and `local2` are points in the respective body-local frames.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ConstraintId,
        body1: EntityId,
        b1_tx: ParamId, b1_ty: ParamId, b1_tz: ParamId,
        b1_qw: ParamId, b1_qx: ParamId, b1_qy: ParamId, b1_qz: ParamId,
        local1: [f64; 3],
        body2: EntityId,
        b2_tx: ParamId, b2_ty: ParamId, b2_tz: ParamId,
        b2_qw: ParamId, b2_qx: ParamId, b2_qy: ParamId, b2_qz: ParamId,
        local2: [f64; 3],
    ) -> Self {
        let params = vec![
            b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
        ];
        Self {
            id,
            body1, body2,
            local1, local2,
            params,
            entities: [body1, body2],
            b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
        }
    }

    fn t1(&self) -> [ParamId; 3] { [self.b1_tx, self.b1_ty, self.b1_tz] }
    fn q1(&self) -> [ParamId; 4] { [self.b1_qw, self.b1_qx, self.b1_qy, self.b1_qz] }
    fn t2(&self) -> [ParamId; 3] { [self.b2_tx, self.b2_ty, self.b2_tz] }
    fn q2(&self) -> [ParamId; 4] { [self.b2_qw, self.b2_qx, self.b2_qy, self.b2_qz] }
}

impl Constraint for Mate {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Mate" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 3 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let w1 = transform_local(store, self.t1(), self.q1(), self.local1);
        let w2 = transform_local(store, self.t2(), self.q2(), self.local2);
        vec![w1[0] - w2[0], w1[1] - w2[1], w1[2] - w2[2]]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let mut jac = Vec::with_capacity(30);
        transform_jacobian_entries(store, 0, 1.0, self.t1(), self.q1(), self.local1, &mut jac);
        transform_jacobian_entries(store, 0, -1.0, self.t2(), self.q2(), self.local2, &mut jac);
        jac
    }
}

// ---------------------------------------------------------------------------
// CoaxialAssembly
// ---------------------------------------------------------------------------

/// Coaxial assembly constraint: two body-local axes must be collinear in world space.
///
/// Each axis is defined by a point and direction in body-local coordinates.
///
/// Equations (4):
/// - Direction parallelism: `(R1*d1) x (R2*d2) = 0` (2 independent eqs)
/// - Point-on-axis: `(w_p2 - w_p1) x (R1*d1) = 0` (2 independent eqs)
///
/// where `w_p = R*local_point + t` is the world-space axis point.
#[derive(Debug, Clone)]
pub struct CoaxialAssembly {
    id: ConstraintId,
    body1: EntityId,
    body2: EntityId,
    // Local axis definitions (constants)
    local_point1: [f64; 3],
    local_dir1: [f64; 3],
    local_point2: [f64; 3],
    local_dir2: [f64; 3],
    params: Vec<ParamId>,
    entities: [EntityId; 2],
    // Body params
    b1_tx: ParamId, b1_ty: ParamId, b1_tz: ParamId,
    b1_qw: ParamId, b1_qx: ParamId, b1_qy: ParamId, b1_qz: ParamId,
    b2_tx: ParamId, b2_ty: ParamId, b2_tz: ParamId,
    b2_qw: ParamId, b2_qx: ParamId, b2_qy: ParamId, b2_qz: ParamId,
}

impl CoaxialAssembly {
    /// Create a coaxial assembly constraint.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ConstraintId,
        body1: EntityId,
        b1_tx: ParamId, b1_ty: ParamId, b1_tz: ParamId,
        b1_qw: ParamId, b1_qx: ParamId, b1_qy: ParamId, b1_qz: ParamId,
        local_point1: [f64; 3],
        local_dir1: [f64; 3],
        body2: EntityId,
        b2_tx: ParamId, b2_ty: ParamId, b2_tz: ParamId,
        b2_qw: ParamId, b2_qx: ParamId, b2_qy: ParamId, b2_qz: ParamId,
        local_point2: [f64; 3],
        local_dir2: [f64; 3],
    ) -> Self {
        let params = vec![
            b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
        ];
        Self {
            id,
            body1, body2,
            local_point1, local_dir1,
            local_point2, local_dir2,
            params,
            entities: [body1, body2],
            b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
        }
    }

    fn t1(&self) -> [ParamId; 3] { [self.b1_tx, self.b1_ty, self.b1_tz] }
    fn q1(&self) -> [ParamId; 4] { [self.b1_qw, self.b1_qx, self.b1_qy, self.b1_qz] }
    fn t2(&self) -> [ParamId; 3] { [self.b2_tx, self.b2_ty, self.b2_tz] }
    fn q2(&self) -> [ParamId; 4] { [self.b2_qw, self.b2_qx, self.b2_qy, self.b2_qz] }
}

impl Constraint for CoaxialAssembly {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "CoaxialAssembly" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 4 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let wd1 = rotate_direction(store, self.q1(), self.local_dir1);
        let wd2 = rotate_direction(store, self.q2(), self.local_dir2);

        // Direction parallelism: wd1 x wd2 (2 components)
        let cross_dir_x = wd1[1] * wd2[2] - wd1[2] * wd2[1];
        let cross_dir_y = wd1[2] * wd2[0] - wd1[0] * wd2[2];

        // World-space axis points
        let wp1 = transform_local(store, self.t1(), self.q1(), self.local_point1);
        let wp2 = transform_local(store, self.t2(), self.q2(), self.local_point2);

        // Point-on-axis: (wp2 - wp1) x wd1 (2 components)
        let dp = [wp2[0] - wp1[0], wp2[1] - wp1[1], wp2[2] - wp1[2]];
        let cross_pt_x = dp[1] * wd1[2] - dp[2] * wd1[1];
        let cross_pt_y = dp[2] * wd1[0] - dp[0] * wd1[2];

        vec![cross_dir_x, cross_dir_y, cross_pt_x, cross_pt_y]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        // Compute via finite differences for correctness and maintainability.
        // The analytic form is complex with cross-product chain rules through
        // quaternion rotations.
        finite_diff_jacobian(self, store)
    }
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

/// Insert constraint: coaxial + mate along the axis direction.
///
/// Combines:
/// 1. Coaxial constraint (4 equations) -- axes are collinear
/// 2. Axial distance constraint (1 equation) -- the projection of the
///    point-to-point vector onto the axis direction equals a target offset
///
/// Total: 5 equations.
#[derive(Debug, Clone)]
pub struct Insert {
    id: ConstraintId,
    body1: EntityId,
    body2: EntityId,
    local_point1: [f64; 3],
    local_dir1: [f64; 3],
    local_point2: [f64; 3],
    local_dir2: [f64; 3],
    offset: f64,
    params: Vec<ParamId>,
    entities: [EntityId; 2],
    b1_tx: ParamId, b1_ty: ParamId, b1_tz: ParamId,
    b1_qw: ParamId, b1_qx: ParamId, b1_qy: ParamId, b1_qz: ParamId,
    b2_tx: ParamId, b2_ty: ParamId, b2_tz: ParamId,
    b2_qw: ParamId, b2_qx: ParamId, b2_qy: ParamId, b2_qz: ParamId,
}

impl Insert {
    /// Create an insert constraint.
    ///
    /// `offset` is the signed distance along the axis between the two points
    /// (0.0 for flush insertion).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ConstraintId,
        body1: EntityId,
        b1_tx: ParamId, b1_ty: ParamId, b1_tz: ParamId,
        b1_qw: ParamId, b1_qx: ParamId, b1_qy: ParamId, b1_qz: ParamId,
        local_point1: [f64; 3],
        local_dir1: [f64; 3],
        body2: EntityId,
        b2_tx: ParamId, b2_ty: ParamId, b2_tz: ParamId,
        b2_qw: ParamId, b2_qx: ParamId, b2_qy: ParamId, b2_qz: ParamId,
        local_point2: [f64; 3],
        local_dir2: [f64; 3],
        offset: f64,
    ) -> Self {
        let params = vec![
            b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
        ];
        Self {
            id,
            body1, body2,
            local_point1, local_dir1,
            local_point2, local_dir2,
            offset,
            params,
            entities: [body1, body2],
            b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
        }
    }

    fn t1(&self) -> [ParamId; 3] { [self.b1_tx, self.b1_ty, self.b1_tz] }
    fn q1(&self) -> [ParamId; 4] { [self.b1_qw, self.b1_qx, self.b1_qy, self.b1_qz] }
    fn t2(&self) -> [ParamId; 3] { [self.b2_tx, self.b2_ty, self.b2_tz] }
    fn q2(&self) -> [ParamId; 4] { [self.b2_qw, self.b2_qx, self.b2_qy, self.b2_qz] }
}

impl Constraint for Insert {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Insert" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 5 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let wd1 = rotate_direction(store, self.q1(), self.local_dir1);
        let wd2 = rotate_direction(store, self.q2(), self.local_dir2);

        // Coaxial part (4 equations)
        let cross_dir_x = wd1[1] * wd2[2] - wd1[2] * wd2[1];
        let cross_dir_y = wd1[2] * wd2[0] - wd1[0] * wd2[2];

        let wp1 = transform_local(store, self.t1(), self.q1(), self.local_point1);
        let wp2 = transform_local(store, self.t2(), self.q2(), self.local_point2);
        let dp = [wp2[0] - wp1[0], wp2[1] - wp1[1], wp2[2] - wp1[2]];
        let cross_pt_x = dp[1] * wd1[2] - dp[2] * wd1[1];
        let cross_pt_y = dp[2] * wd1[0] - dp[0] * wd1[2];

        // Axial distance (1 equation): dp . wd1_hat - offset
        // Use unnormalized dot since the solver will find the right quaternion norm
        let dot = dp[0] * wd1[0] + dp[1] * wd1[1] + dp[2] * wd1[2];
        let len_sq = wd1[0] * wd1[0] + wd1[1] * wd1[1] + wd1[2] * wd1[2];
        let len = len_sq.sqrt().max(1e-15);
        let axial_residual = dot / len - self.offset;

        vec![cross_dir_x, cross_dir_y, cross_pt_x, cross_pt_y, axial_residual]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        finite_diff_jacobian(self, store)
    }
}

// ---------------------------------------------------------------------------
// Gear
// ---------------------------------------------------------------------------

/// Gear constraint: rotation of body1 about axis1 is linked to body2's rotation
/// about axis2 by a gear ratio.
///
/// The constraint tracks the relative rotation angle using the quaternion
/// components. For small rotations about axis `a` (unit vector in local frame),
/// the rotation angle `theta` can be extracted from `q` via:
///
/// ```text
/// sin(theta/2) = (qx*ax + qy*ay + qz*az)
/// cos(theta/2) = qw
/// theta = 2 * atan2(sin, cos)
/// ```
///
/// Residual (1 equation):
/// ```text
/// theta1 * ratio - theta2 = 0
/// ```
///
/// where `theta_i` is the rotation angle of body `i` about its local axis.
#[derive(Debug, Clone)]
pub struct Gear {
    id: ConstraintId,
    body1: EntityId,
    body2: EntityId,
    local_axis1: [f64; 3],
    local_axis2: [f64; 3],
    ratio: f64,
    params: Vec<ParamId>,
    entities: [EntityId; 2],
    b1_qw: ParamId, b1_qx: ParamId, b1_qy: ParamId, b1_qz: ParamId,
    b2_qw: ParamId, b2_qx: ParamId, b2_qy: ParamId, b2_qz: ParamId,
}

impl Gear {
    /// Create a gear constraint.
    ///
    /// `ratio` is `theta1 / theta2`: if body1 rotates by `theta1`, body2
    /// must rotate by `theta1 / ratio`. The local axes should be unit vectors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ConstraintId,
        body1: EntityId,
        b1_qw: ParamId, b1_qx: ParamId, b1_qy: ParamId, b1_qz: ParamId,
        local_axis1: [f64; 3],
        body2: EntityId,
        b2_qw: ParamId, b2_qx: ParamId, b2_qy: ParamId, b2_qz: ParamId,
        local_axis2: [f64; 3],
        ratio: f64,
    ) -> Self {
        let params = vec![b1_qw, b1_qx, b1_qy, b1_qz, b2_qw, b2_qx, b2_qy, b2_qz];
        Self {
            id,
            body1, body2,
            local_axis1, local_axis2,
            ratio,
            params,
            entities: [body1, body2],
            b1_qw, b1_qx, b1_qy, b1_qz,
            b2_qw, b2_qx, b2_qy, b2_qz,
        }
    }

    /// Extract the rotation angle about the local axis from quaternion components.
    fn axis_angle(store: &ParamStore, qw: ParamId, qx: ParamId, qy: ParamId, qz: ParamId, axis: [f64; 3]) -> f64 {
        let w = store.get(qw);
        let x = store.get(qx);
        let y = store.get(qy);
        let z = store.get(qz);
        let sin_half = x * axis[0] + y * axis[1] + z * axis[2];
        let cos_half = w;
        2.0 * sin_half.atan2(cos_half)
    }
}

impl Constraint for Gear {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "Gear" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let theta1 = Self::axis_angle(store, self.b1_qw, self.b1_qx, self.b1_qy, self.b1_qz, self.local_axis1);
        let theta2 = Self::axis_angle(store, self.b2_qw, self.b2_qx, self.b2_qy, self.b2_qz, self.local_axis2);
        vec![theta1 * self.ratio - theta2]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        // theta = 2 * atan2(sin_half, cos_half)
        // sin_half = qx*ax + qy*ay + qz*az
        // cos_half = qw
        //
        // d(theta)/d(qw) = 2 * (-sin_half) / (sin_half^2 + cos_half^2)
        // d(theta)/d(qx) = 2 * (cos_half * ax) / (sin_half^2 + cos_half^2)
        // etc.
        //
        // For atan2(y,x): d/dy = x/(x^2+y^2), d/dx = -y/(x^2+y^2)

        let compute_derivs = |qw: ParamId, qx: ParamId, qy: ParamId, qz: ParamId, axis: [f64; 3], scale: f64| -> Vec<(usize, ParamId, f64)> {
            let w = store.get(qw);
            let xv = store.get(qx);
            let yv = store.get(qy);
            let zv = store.get(qz);
            let sin_half = xv * axis[0] + yv * axis[1] + zv * axis[2];
            let cos_half = w;
            let denom = sin_half * sin_half + cos_half * cos_half;
            // d(atan2(s,c))/ds = c/denom, d(atan2(s,c))/dc = -s/denom
            // theta = 2*atan2(s,c)
            // dtheta/dqw = 2 * (-sin_half) / denom
            // dtheta/dqx = 2 * cos_half * ax / denom
            // dtheta/dqy = 2 * cos_half * ay / denom
            // dtheta/dqz = 2 * cos_half * az / denom
            let dtdqw = 2.0 * (-sin_half) / denom;
            let dtdqx = 2.0 * cos_half * axis[0] / denom;
            let dtdqy = 2.0 * cos_half * axis[1] / denom;
            let dtdqz = 2.0 * cos_half * axis[2] / denom;
            vec![
                (0, qw, scale * dtdqw),
                (0, qx, scale * dtdqx),
                (0, qy, scale * dtdqy),
                (0, qz, scale * dtdqz),
            ]
        };

        let mut jac = compute_derivs(self.b1_qw, self.b1_qx, self.b1_qy, self.b1_qz, self.local_axis1, self.ratio);
        jac.extend(compute_derivs(self.b2_qw, self.b2_qx, self.b2_qy, self.b2_qz, self.local_axis2, -1.0));
        jac
    }
}

// ---------------------------------------------------------------------------
// Finite-difference Jacobian helper
// ---------------------------------------------------------------------------

/// Compute Jacobian via central finite differences.
///
/// Used for complex constraints where the analytic Jacobian would be
/// error-prone and hard to maintain (e.g., CoaxialAssembly, Insert).
fn finite_diff_jacobian(
    constraint: &dyn Constraint,
    store: &ParamStore,
) -> Vec<(usize, ParamId, f64)> {
    let eps = 1e-8;
    let params = constraint.param_ids().to_vec();
    let n_eq = constraint.equation_count();
    let mut snapshot = store.snapshot();
    let mut jac = Vec::new();

    for &pid in &params {
        let orig = snapshot.get(pid);

        snapshot.set(pid, orig + eps);
        let r_plus = constraint.residuals(&snapshot);

        snapshot.set(pid, orig - eps);
        let r_minus = constraint.residuals(&snapshot);

        snapshot.set(pid, orig);

        for row in 0..n_eq {
            let deriv = (r_plus[row] - r_minus[row]) / (2.0 * eps);
            if deriv.abs() > 1e-14 {
                jac.push((row, pid, deriv));
            }
        }
    }

    jac
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::EntityId;

    fn eid(i: u32) -> EntityId { EntityId::new(i, 0) }
    fn cid(i: u32) -> ConstraintId { ConstraintId::new(i, 0) }

    /// Create a rigid body with identity orientation and given translation.
    fn make_body(store: &mut ParamStore, entity: EntityId, pos: [f64; 3]) -> (
        ParamId, ParamId, ParamId,
        ParamId, ParamId, ParamId, ParamId,
    ) {
        let tx = store.alloc(pos[0], entity);
        let ty = store.alloc(pos[1], entity);
        let tz = store.alloc(pos[2], entity);
        let qw = store.alloc(1.0, entity);
        let qx = store.alloc(0.0, entity);
        let qy = store.alloc(0.0, entity);
        let qz = store.alloc(0.0, entity);
        (tx, ty, tz, qw, qx, qy, qz)
    }

    /// Verify Jacobian via finite differences.
    fn verify_jacobian_fd(
        constraint: &dyn Constraint,
        store: &mut ParamStore,
        eps: f64,
        tol: f64,
    ) {
        let analytic = constraint.jacobian(store);
        let n_eq = constraint.equation_count();

        for &(row, pid, analytic_val) in &analytic {
            assert!(row < n_eq);

            let orig = store.get(pid);
            store.set(pid, orig + eps);
            let rp = constraint.residuals(store);
            store.set(pid, orig - eps);
            let rm = constraint.residuals(store);
            store.set(pid, orig);

            let fd = (rp[row] - rm[row]) / (2.0 * eps);
            let err = (analytic_val - fd).abs();
            assert!(
                err < tol,
                "Jacobian mismatch for {:?} row {}: analytic={}, fd={}, err={}",
                pid, row, analytic_val, fd, err,
            );
        }
    }

    // -- Mate tests --

    #[test]
    fn mate_identity_bodies_same_point() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [0.0, 0.0, 0.0]);

        let c = Mate::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [1.0, 0.0, 0.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [1.0, 0.0, 0.0],
        );

        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-12), "residuals: {:?}", r);
    }

    #[test]
    fn mate_translated_body() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        // Body1 at origin, Body2 at (10, 0, 0)
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [10.0, 0.0, 0.0]);

        // Local point (5,0,0) on body1 meets local point (-5,0,0) on body2
        // World: 0+(5,0,0) = (5,0,0), 10+(-5,0,0) = (5,0,0)
        let c = Mate::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [5.0, 0.0, 0.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [-5.0, 0.0, 0.0],
        );

        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-12), "residuals: {:?}", r);
    }

    #[test]
    fn mate_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [1.0, 2.0, 3.0]);
        // Slight rotation for body2
        let b2_tx = store.alloc(4.0, e2);
        let b2_ty = store.alloc(5.0, e2);
        let b2_tz = store.alloc(6.0, e2);
        let norm = (1.0_f64 + 0.01 + 0.04 + 0.0).sqrt();
        let b2_qw = store.alloc(1.0 / norm, e2);
        let b2_qx = store.alloc(0.1 / norm, e2);
        let b2_qy = store.alloc(0.2 / norm, e2);
        let b2_qz = store.alloc(0.0 / norm, e2);

        let c = Mate::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [1.0, 0.5, -0.3],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [-0.5, 1.0, 0.2],
        );

        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-4);
    }

    // -- CoaxialAssembly tests --

    #[test]
    fn coaxial_assembly_aligned() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        // Both bodies at origin, identity rotation, z-axis
        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [0.0, 0.0, 5.0]);

        let c = CoaxialAssembly::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        );

        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-12), "residuals: {:?}", r);
    }

    #[test]
    fn coaxial_assembly_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [1.0, 0.0, 0.0]);
        let norm = (1.0_f64 + 0.01 + 0.04 + 0.09).sqrt();
        let b2_tx = store.alloc(2.0, e2);
        let b2_ty = store.alloc(1.0, e2);
        let b2_tz = store.alloc(0.5, e2);
        let b2_qw = store.alloc(1.0 / norm, e2);
        let b2_qx = store.alloc(0.1 / norm, e2);
        let b2_qy = store.alloc(0.2 / norm, e2);
        let b2_qz = store.alloc(0.3 / norm, e2);

        let c = CoaxialAssembly::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        );

        // Since CoaxialAssembly uses FD internally, verify against independent FD
        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-4);
    }

    // -- Insert tests --

    #[test]
    fn insert_flush_aligned() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [0.0, 0.0, 0.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [0.0, 0.0, 0.0]);

        let c = Insert::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            0.0,
        );

        let r = c.residuals(&store);
        assert!(r.iter().all(|v| v.abs() < 1e-12), "residuals: {:?}", r);
    }

    #[test]
    fn insert_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let (b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz) =
            make_body(&mut store, e1, [1.0, 2.0, 3.0]);
        let (b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz) =
            make_body(&mut store, e2, [4.0, 5.0, 6.0]);

        let c = Insert::new(
            cid(0),
            e1, b1_tx, b1_ty, b1_tz, b1_qw, b1_qx, b1_qy, b1_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2_tx, b2_ty, b2_tz, b2_qw, b2_qx, b2_qy, b2_qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            2.0,
        );

        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-4);
    }

    // -- Gear tests --

    #[test]
    fn gear_no_rotation() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        // Both bodies at identity rotation -> theta1 = theta2 = 0
        let b1_qw = store.alloc(1.0, e1);
        let b1_qx = store.alloc(0.0, e1);
        let b1_qy = store.alloc(0.0, e1);
        let b1_qz = store.alloc(0.0, e1);
        let b2_qw = store.alloc(1.0, e2);
        let b2_qx = store.alloc(0.0, e2);
        let b2_qy = store.alloc(0.0, e2);
        let b2_qz = store.alloc(0.0, e2);

        let c = Gear::new(
            cid(0),
            e1, b1_qw, b1_qx, b1_qy, b1_qz, [0.0, 0.0, 1.0],
            e2, b2_qw, b2_qx, b2_qy, b2_qz, [0.0, 0.0, 1.0],
            2.0,
        );

        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12, "residual: {}", r[0]);
    }

    #[test]
    fn gear_ratio_satisfied() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        // Body1: 30 deg about z
        let theta1: f64 = std::f64::consts::PI / 6.0;
        let b1_qw = store.alloc((theta1 / 2.0).cos(), e1);
        let b1_qx = store.alloc(0.0, e1);
        let b1_qy = store.alloc(0.0, e1);
        let b1_qz = store.alloc((theta1 / 2.0).sin(), e1);

        // Body2: 60 deg about z (ratio = 2)
        let theta2: f64 = std::f64::consts::PI / 3.0;
        let b2_qw = store.alloc((theta2 / 2.0).cos(), e2);
        let b2_qx = store.alloc(0.0, e2);
        let b2_qy = store.alloc(0.0, e2);
        let b2_qz = store.alloc((theta2 / 2.0).sin(), e2);

        // ratio * theta1 - theta2 = 2 * 30 - 60 = 0
        let c = Gear::new(
            cid(0),
            e1, b1_qw, b1_qx, b1_qy, b1_qz, [0.0, 0.0, 1.0],
            e2, b2_qw, b2_qx, b2_qy, b2_qz, [0.0, 0.0, 1.0],
            2.0,
        );

        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-12, "residual: {}", r[0]);
    }

    #[test]
    fn gear_jacobian_fd() {
        let mut store = ParamStore::new();
        let e1 = eid(0);
        let e2 = eid(1);

        let theta1: f64 = 0.3;
        let b1_qw = store.alloc((theta1 / 2.0).cos(), e1);
        let b1_qx = store.alloc(0.0, e1);
        let b1_qy = store.alloc(0.0, e1);
        let b1_qz = store.alloc((theta1 / 2.0).sin(), e1);

        let theta2: f64 = 0.7;
        let b2_qw = store.alloc((theta2 / 2.0).cos(), e2);
        let b2_qx = store.alloc(0.0, e2);
        let b2_qy = store.alloc(0.0, e2);
        let b2_qz = store.alloc((theta2 / 2.0).sin(), e2);

        let c = Gear::new(
            cid(0),
            e1, b1_qw, b1_qx, b1_qy, b1_qz, [0.0, 0.0, 1.0],
            e2, b2_qw, b2_qx, b2_qy, b2_qz, [0.0, 0.0, 1.0],
            2.0,
        );

        verify_jacobian_fd(&c, &mut store, 1e-7, 1e-5);
    }
}
