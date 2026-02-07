//! Assembly entity types.
//!
//! - [`RigidBody`] -- a rigid body with position (3 params) and quaternion
//!   orientation (4 params).
//! - [`UnitQuaternion`] -- internal constraint enforcing unit-length quaternion.

use crate::constraint::Constraint;
use crate::entity::Entity;
use crate::id::{ConstraintId, EntityId, ParamId};
use crate::param::ParamStore;

// ---------------------------------------------------------------------------
// RigidBody
// ---------------------------------------------------------------------------

/// A rigid body in 3D space.
///
/// Parameterized by 7 values:
/// - Translation: `(tx, ty, tz)` -- position of the body origin in world space
/// - Orientation: `(qw, qx, qy, qz)` -- unit quaternion (scalar-first convention)
///
/// The quaternion must satisfy `qw^2 + qx^2 + qy^2 + qz^2 = 1`. This is
/// enforced by the companion [`UnitQuaternion`] constraint, which should be
/// added to the system alongside the rigid body.
#[derive(Debug, Clone)]
pub struct RigidBody {
    id: EntityId,
    // Translation
    tx: ParamId,
    ty: ParamId,
    tz: ParamId,
    // Quaternion (scalar-first: w, x, y, z)
    qw: ParamId,
    qx: ParamId,
    qy: ParamId,
    qz: ParamId,
    params: [ParamId; 7],
}

impl RigidBody {
    /// Create a new rigid body entity.
    pub fn new(
        id: EntityId,
        tx: ParamId, ty: ParamId, tz: ParamId,
        qw: ParamId, qx: ParamId, qy: ParamId, qz: ParamId,
    ) -> Self {
        Self {
            id,
            tx, ty, tz,
            qw, qx, qy, qz,
            params: [tx, ty, tz, qw, qx, qy, qz],
        }
    }

    /// Transform a point from body-local coordinates to world coordinates.
    ///
    /// `world = R(q) * local + t`
    ///
    /// where `R(q)` is the rotation matrix derived from the quaternion and
    /// `t = (tx, ty, tz)` is the translation.
    pub fn transform_point(&self, store: &ParamStore, local: [f64; 3]) -> [f64; 3] {
        let r = self.rotation_matrix(store);
        let t = [store.get(self.tx), store.get(self.ty), store.get(self.tz)];
        [
            r[0][0] * local[0] + r[0][1] * local[1] + r[0][2] * local[2] + t[0],
            r[1][0] * local[0] + r[1][1] * local[1] + r[1][2] * local[2] + t[1],
            r[2][0] * local[0] + r[2][1] * local[1] + r[2][2] * local[2] + t[2],
        ]
    }

    /// Compute the 3x3 rotation matrix from the quaternion parameters.
    ///
    /// Uses the standard quaternion-to-rotation-matrix formula (scalar-first):
    /// ```text
    /// R = | 1-2(qy^2+qz^2)   2(qx*qy-qz*qw)   2(qx*qz+qy*qw) |
    ///     | 2(qx*qy+qz*qw)   1-2(qx^2+qz^2)   2(qy*qz-qx*qw) |
    ///     | 2(qx*qz-qy*qw)   2(qy*qz+qx*qw)   1-2(qx^2+qy^2) |
    /// ```
    pub fn rotation_matrix(&self, store: &ParamStore) -> [[f64; 3]; 3] {
        let w = store.get(self.qw);
        let x = store.get(self.qx);
        let y = store.get(self.qy);
        let z = store.get(self.qz);
        quat_to_rotation_matrix(w, x, y, z)
    }

    /// Parameter IDs for the translation components.
    pub fn position(&self) -> (ParamId, ParamId, ParamId) {
        (self.tx, self.ty, self.tz)
    }

    /// Parameter IDs for the quaternion components `(w, x, y, z)`.
    pub fn quaternion(&self) -> (ParamId, ParamId, ParamId, ParamId) {
        (self.qw, self.qx, self.qy, self.qz)
    }
}

impl Entity for RigidBody {
    fn id(&self) -> EntityId {
        self.id
    }

    fn params(&self) -> &[ParamId] {
        &self.params
    }

    fn name(&self) -> &str {
        "RigidBody"
    }
}

// ---------------------------------------------------------------------------
// UnitQuaternion constraint
// ---------------------------------------------------------------------------

/// Constraint enforcing a unit-length quaternion.
///
/// Residual: `qw^2 + qx^2 + qy^2 + qz^2 - 1`
///
/// This constraint should always accompany a [`RigidBody`] to keep the
/// quaternion on the unit sphere during solving.
#[derive(Debug, Clone)]
pub struct UnitQuaternion {
    id: ConstraintId,
    body_entity: EntityId,
    qw: ParamId,
    qx: ParamId,
    qy: ParamId,
    qz: ParamId,
    params: [ParamId; 4],
    entities: [EntityId; 1],
}

impl UnitQuaternion {
    /// Create a unit quaternion constraint for a rigid body.
    pub fn new(
        id: ConstraintId,
        body_entity: EntityId,
        qw: ParamId, qx: ParamId, qy: ParamId, qz: ParamId,
    ) -> Self {
        Self {
            id,
            body_entity,
            qw, qx, qy, qz,
            params: [qw, qx, qy, qz],
            entities: [body_entity],
        }
    }
}

impl Constraint for UnitQuaternion {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "UnitQuaternion" }
    fn entity_ids(&self) -> &[EntityId] { &self.entities }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let w = store.get(self.qw);
        let x = store.get(self.qx);
        let y = store.get(self.qy);
        let z = store.get(self.qz);
        vec![w * w + x * x + y * y + z * z - 1.0]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let w = store.get(self.qw);
        let x = store.get(self.qx);
        let y = store.get(self.qy);
        let z = store.get(self.qz);
        vec![
            (0, self.qw, 2.0 * w),
            (0, self.qx, 2.0 * x),
            (0, self.qy, 2.0 * y),
            (0, self.qz, 2.0 * z),
        ]
    }
}

// ---------------------------------------------------------------------------
// Quaternion helper functions
// ---------------------------------------------------------------------------

/// Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.
pub(crate) fn quat_to_rotation_matrix(w: f64, x: f64, y: f64, z: f64) -> [[f64; 3]; 3] {
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    [
        [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (x2 + y2)],
    ]
}

/// Compute the derivative of R(q)*v with respect to each quaternion component.
///
/// Returns `[dRv/dw, dRv/dx, dRv/dy, dRv/dz]`, each a 3-element array.
///
/// For `R(q)*v`:
/// ```text
/// Rv_i = sum_j R_ij * v_j
/// dRv_i/dw = sum_j (dR_ij/dw) * v_j
/// ```
pub(crate) fn quat_rotate_derivatives(
    w: f64, x: f64, y: f64, z: f64,
    v: [f64; 3],
) -> [[f64; 3]; 4] {
    // dR/dw:
    // R00 = 1-2(y^2+z^2)  => dR00/dw = 0
    // R01 = 2(xy - wz)    => dR01/dw = -2z
    // R02 = 2(xz + wy)    => dR02/dw = 2y
    // R10 = 2(xy + wz)    => dR10/dw = 2z
    // R11 = 1-2(x^2+z^2)  => dR11/dw = 0
    // R12 = 2(yz - wx)    => dR12/dw = -2x
    // R20 = 2(xz - wy)    => dR20/dw = -2y
    // R21 = 2(yz + wx)    => dR21/dw = 2x
    // R22 = 1-2(x^2+y^2)  => dR22/dw = 0
    let dr_dw = [
        [0.0,      -2.0 * z,  2.0 * y],
        [2.0 * z,   0.0,     -2.0 * x],
        [-2.0 * y,  2.0 * x,  0.0],
    ];

    // dR/dx:
    // R00 = 1-2(y^2+z^2)  => 0
    // R01 = 2(xy - wz)    => 2y
    // R02 = 2(xz + wy)    => 2z
    // R10 = 2(xy + wz)    => 2y
    // R11 = 1-2(x^2+z^2)  => -4x
    // R12 = 2(yz - wx)    => -2w
    // R20 = 2(xz - wy)    => 2z
    // R21 = 2(yz + wx)    => 2w
    // R22 = 1-2(x^2+y^2)  => -4x
    let dr_dx = [
        [0.0,       2.0 * y,  2.0 * z],
        [2.0 * y,  -4.0 * x, -2.0 * w],
        [2.0 * z,   2.0 * w, -4.0 * x],
    ];

    // dR/dy:
    // R00 = 1-2(y^2+z^2)  => -4y
    // R01 = 2(xy - wz)    => 2x
    // R02 = 2(xz + wy)    => 2w
    // R10 = 2(xy + wz)    => 2x
    // R11 = 1-2(x^2+z^2)  => 0
    // R12 = 2(yz - wx)    => 2z
    // R20 = 2(xz - wy)    => -2w
    // R21 = 2(yz + wx)    => 2z
    // R22 = 1-2(x^2+y^2)  => -4y
    let dr_dy = [
        [-4.0 * y,  2.0 * x,  2.0 * w],
        [2.0 * x,   0.0,      2.0 * z],
        [-2.0 * w,  2.0 * z, -4.0 * y],
    ];

    // dR/dz:
    // R00 = 1-2(y^2+z^2)  => -4z
    // R01 = 2(xy - wz)    => -2w
    // R02 = 2(xz + wy)    => 2x
    // R10 = 2(xy + wz)    => 2w
    // R11 = 1-2(x^2+z^2)  => -4z
    // R12 = 2(yz - wx)    => 2y
    // R20 = 2(xz - wy)    => 2x
    // R21 = 2(yz + wx)    => 2y
    // R22 = 1-2(x^2+y^2)  => 0
    let dr_dz = [
        [-4.0 * z, -2.0 * w,  2.0 * x],
        [2.0 * w,  -4.0 * z,  2.0 * y],
        [2.0 * x,   2.0 * y,  0.0],
    ];

    let mat_vec = |m: [[f64; 3]; 3], v: [f64; 3]| -> [f64; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    };

    [
        mat_vec(dr_dw, v),
        mat_vec(dr_dx, v),
        mat_vec(dr_dy, v),
        mat_vec(dr_dz, v),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn eid(i: u32) -> EntityId { EntityId::new(i, 0) }
    fn cid(i: u32) -> ConstraintId { ConstraintId::new(i, 0) }

    #[test]
    fn rigid_body_identity() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let tx = store.alloc(0.0, e);
        let ty = store.alloc(0.0, e);
        let tz = store.alloc(0.0, e);
        let qw = store.alloc(1.0, e);
        let qx = store.alloc(0.0, e);
        let qy = store.alloc(0.0, e);
        let qz = store.alloc(0.0, e);

        let body = RigidBody::new(e, tx, ty, tz, qw, qx, qy, qz);
        assert_eq!(body.params().len(), 7);
        assert_eq!(body.name(), "RigidBody");

        // Identity transform should leave point unchanged
        let world = body.transform_point(&store, [1.0, 2.0, 3.0]);
        assert!((world[0] - 1.0).abs() < 1e-12);
        assert!((world[1] - 2.0).abs() < 1e-12);
        assert!((world[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rigid_body_translation() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let tx = store.alloc(10.0, e);
        let ty = store.alloc(20.0, e);
        let tz = store.alloc(30.0, e);
        let qw = store.alloc(1.0, e);
        let qx = store.alloc(0.0, e);
        let qy = store.alloc(0.0, e);
        let qz = store.alloc(0.0, e);

        let body = RigidBody::new(e, tx, ty, tz, qw, qx, qy, qz);
        let world = body.transform_point(&store, [1.0, 2.0, 3.0]);
        assert!((world[0] - 11.0).abs() < 1e-12);
        assert!((world[1] - 22.0).abs() < 1e-12);
        assert!((world[2] - 33.0).abs() < 1e-12);
    }

    #[test]
    fn rigid_body_90deg_z_rotation() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let tx = store.alloc(0.0, e);
        let ty = store.alloc(0.0, e);
        let tz = store.alloc(0.0, e);
        // 90 degrees about z: q = (cos(45), 0, 0, sin(45))
        let c = std::f64::consts::FRAC_PI_4.cos();
        let s = std::f64::consts::FRAC_PI_4.sin();
        let qw = store.alloc(c, e);
        let qx = store.alloc(0.0, e);
        let qy = store.alloc(0.0, e);
        let qz = store.alloc(s, e);

        let body = RigidBody::new(e, tx, ty, tz, qw, qx, qy, qz);
        // Rotating (1,0,0) by 90 about z -> (0,1,0)
        let world = body.transform_point(&store, [1.0, 0.0, 0.0]);
        assert!((world[0]).abs() < 1e-12, "x: {}", world[0]);
        assert!((world[1] - 1.0).abs() < 1e-12, "y: {}", world[1]);
        assert!((world[2]).abs() < 1e-12, "z: {}", world[2]);
    }

    #[test]
    fn rotation_matrix_orthogonal() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let tx = store.alloc(0.0, e);
        let ty = store.alloc(0.0, e);
        let tz = store.alloc(0.0, e);
        // Arbitrary unit quaternion
        let norm = (1.0_f64 + 4.0 + 9.0 + 16.0).sqrt();
        let qw = store.alloc(1.0 / norm, e);
        let qx = store.alloc(2.0 / norm, e);
        let qy = store.alloc(3.0 / norm, e);
        let qz = store.alloc(4.0 / norm, e);

        let body = RigidBody::new(e, tx, ty, tz, qw, qx, qy, qz);
        let r = body.rotation_matrix(&store);

        // R * R^T should be identity
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "R*R^T[{}][{}] = {}, expected {}",
                    i, j, dot, expected,
                );
            }
        }
    }

    #[test]
    fn unit_quaternion_satisfied() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let qw = store.alloc(1.0, e);
        let qx = store.alloc(0.0, e);
        let qy = store.alloc(0.0, e);
        let qz = store.alloc(0.0, e);

        let c = UnitQuaternion::new(cid(0), e, qw, qx, qy, qz);
        let r = c.residuals(&store);
        assert!(r[0].abs() < 1e-15);
    }

    #[test]
    fn unit_quaternion_jacobian_fd() {
        let mut store = ParamStore::new();
        let e = eid(0);
        let norm = (1.0_f64 + 0.04 + 0.09 + 0.01).sqrt();
        let qw = store.alloc(1.0 / norm, e);
        let qx = store.alloc(0.2 / norm, e);
        let qy = store.alloc(0.3 / norm, e);
        let qz = store.alloc(0.1 / norm, e);

        let c = UnitQuaternion::new(cid(0), e, qw, qx, qy, qz);
        let analytic = c.jacobian(&store);
        let eps = 1e-7;

        for &(row, pid, av) in &analytic {
            let orig = store.get(pid);
            store.set(pid, orig + eps);
            let rp = c.residuals(&store);
            store.set(pid, orig - eps);
            let rm = c.residuals(&store);
            store.set(pid, orig);

            let fd = (rp[row] - rm[row]) / (2.0 * eps);
            assert!(
                (av - fd).abs() < 1e-5,
                "UnitQuaternion jac mismatch: analytic={}, fd={}",
                av, fd,
            );
        }
    }

    #[test]
    fn quat_rotate_derivatives_fd() {
        let norm = (1.0_f64 + 0.04 + 0.09 + 0.16).sqrt();
        let w = 1.0 / norm;
        let x = 0.2 / norm;
        let y = 0.3 / norm;
        let z = 0.4 / norm;
        let v = [1.0, 2.0, 3.0];

        let derivs = quat_rotate_derivatives(w, x, y, z, v);
        let eps = 1e-7;

        // Test dRv/dw
        let rv_w = |w_: f64| {
            let r = quat_to_rotation_matrix(w_, x, y, z);
            [
                r[0][0]*v[0] + r[0][1]*v[1] + r[0][2]*v[2],
                r[1][0]*v[0] + r[1][1]*v[1] + r[1][2]*v[2],
                r[2][0]*v[0] + r[2][1]*v[1] + r[2][2]*v[2],
            ]
        };
        let rp = rv_w(w + eps);
        let rm = rv_w(w - eps);
        for i in 0..3 {
            let fd = (rp[i] - rm[i]) / (2.0 * eps);
            assert!((derivs[0][i] - fd).abs() < 1e-5, "dRv/dw[{}]: a={}, fd={}", i, derivs[0][i], fd);
        }

        // Test dRv/dx
        let rv_x = |x_: f64| {
            let r = quat_to_rotation_matrix(w, x_, y, z);
            [
                r[0][0]*v[0] + r[0][1]*v[1] + r[0][2]*v[2],
                r[1][0]*v[0] + r[1][1]*v[1] + r[1][2]*v[2],
                r[2][0]*v[0] + r[2][1]*v[1] + r[2][2]*v[2],
            ]
        };
        let rp = rv_x(x + eps);
        let rm = rv_x(x - eps);
        for i in 0..3 {
            let fd = (rp[i] - rm[i]) / (2.0 * eps);
            assert!((derivs[1][i] - fd).abs() < 1e-5, "dRv/dx[{}]: a={}, fd={}", i, derivs[1][i], fd);
        }

        // Test dRv/dy
        let rv_y = |y_: f64| {
            let r = quat_to_rotation_matrix(w, x, y_, z);
            [
                r[0][0]*v[0] + r[0][1]*v[1] + r[0][2]*v[2],
                r[1][0]*v[0] + r[1][1]*v[1] + r[1][2]*v[2],
                r[2][0]*v[0] + r[2][1]*v[1] + r[2][2]*v[2],
            ]
        };
        let rp = rv_y(y + eps);
        let rm = rv_y(y - eps);
        for i in 0..3 {
            let fd = (rp[i] - rm[i]) / (2.0 * eps);
            assert!((derivs[2][i] - fd).abs() < 1e-5, "dRv/dy[{}]: a={}, fd={}", i, derivs[2][i], fd);
        }

        // Test dRv/dz
        let rv_z = |z_: f64| {
            let r = quat_to_rotation_matrix(w, x, y, z_);
            [
                r[0][0]*v[0] + r[0][1]*v[1] + r[0][2]*v[2],
                r[1][0]*v[0] + r[1][1]*v[1] + r[1][2]*v[2],
                r[2][0]*v[0] + r[2][1]*v[1] + r[2][2]*v[2],
            ]
        };
        let rp = rv_z(z + eps);
        let rm = rv_z(z - eps);
        for i in 0..3 {
            let fd = (rp[i] - rm[i]) / (2.0 * eps);
            assert!((derivs[3][i] - fd).abs() < 1e-5, "dRv/dz[{}]: a={}, fd={}", i, derivs[3][i], fd);
        }
    }
}
