//! Property-based tests for the V3 assembly constraint system.
//!
//! Tests quaternion-based rigid body constraints: Mate, CoaxialAssembly,
//! Insert, Gear, and UnitQuaternion.

use proptest::prelude::*;
use solverang::assembly::{CoaxialAssembly, Gear, Insert, Mate, UnitQuaternion};
use solverang::constraint::Constraint;
use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::param::ParamStore;
use solverang::ConstraintSystem;

// =============================================================================
// Helpers
// =============================================================================

struct TestCtx {
    sys: ConstraintSystem,
}

impl TestCtx {
    fn new() -> Self {
        Self { sys: ConstraintSystem::new() }
    }
    fn entity(&mut self) -> EntityId { self.sys.alloc_entity_id() }
    fn cid(&mut self) -> ConstraintId { self.sys.alloc_constraint_id() }
    fn alloc(&mut self, value: f64, owner: EntityId) -> ParamId {
        self.sys.alloc_param(value, owner)
    }
    fn store(&self) -> ParamStore { self.sys.params().clone() }
}

/// Allocate a rigid body (translation + unit quaternion) and return all 7 param IDs.
struct BodyParams {
    tx: ParamId, ty: ParamId, tz: ParamId,
    qw: ParamId, qx: ParamId, qy: ParamId, qz: ParamId,
}

fn alloc_body(ctx: &mut TestCtx, entity: EntityId, pos: [f64; 3], quat: [f64; 4]) -> BodyParams {
    let tx = ctx.alloc(pos[0], entity);
    let ty = ctx.alloc(pos[1], entity);
    let tz = ctx.alloc(pos[2], entity);
    let qw = ctx.alloc(quat[0], entity);
    let qx = ctx.alloc(quat[1], entity);
    let qy = ctx.alloc(quat[2], entity);
    let qz = ctx.alloc(quat[3], entity);
    BodyParams { tx, ty, tz, qw, qx, qy, qz }
}

/// Central finite-difference Jacobian check with scaled step and relative tolerance.
fn check_jacobian_fd(constraint: &dyn Constraint, store: &ParamStore, eps: f64, tol: f64) -> bool {
    let params = constraint.param_ids().to_vec();
    let analytical = constraint.jacobian(store);

    for eq in 0..constraint.equation_count() {
        for &pid in &params {
            let orig = store.get(pid);
            let h = eps * (1.0 + orig.abs());

            let mut plus = store.clone();
            plus.set(pid, orig + h);
            let r_plus = constraint.residuals(&plus);

            let mut minus = store.clone();
            minus.set(pid, orig - h);
            let r_minus = constraint.residuals(&minus);

            let fd = (r_plus[eq] - r_minus[eq]) / (2.0 * h);

            let ana: f64 = analytical
                .iter()
                .filter(|&&(r, p, _)| r == eq && p == pid)
                .map(|&(_, _, v)| v)
                .sum();

            let error = (fd - ana).abs();
            let scale = 1.0 + fd.abs().max(ana.abs());
            if error >= tol * scale {
                return false;
            }
        }
    }
    true
}

// =============================================================================
// Proptest strategies
// =============================================================================

fn coord() -> impl Strategy<Value = f64> { -50.0f64..50.0 }

/// Generate a unit quaternion from axis-angle representation.
fn unit_quat() -> impl Strategy<Value = [f64; 4]> {
    (
        0.01f64..std::f64::consts::PI - 0.01,  // theta (polar)
        0.0f64..std::f64::consts::TAU,           // phi (azimuthal)
        0.0f64..std::f64::consts::TAU,           // rotation angle
    ).prop_map(|(theta, phi, angle)| {
        // axis
        let ax = theta.sin() * phi.cos();
        let ay = theta.sin() * phi.sin();
        let az = theta.cos();
        // quaternion from axis-angle
        let half = angle / 2.0;
        let s = half.sin();
        [half.cos(), ax * s, ay * s, az * s]
    })
}

/// Generate an identity quaternion (useful for known-satisfied configurations).
fn identity_quat() -> [f64; 4] { [1.0, 0.0, 0.0, 0.0] }

/// Small local coordinate offset.
fn local_coord() -> impl Strategy<Value = f64> { -5.0f64..5.0 }

/// Small local point.
fn local_point() -> impl Strategy<Value = [f64; 3]> {
    (local_coord(), local_coord(), local_coord()).prop_map(|(x, y, z)| [x, y, z])
}

// =============================================================================
// Section 1: Constraint satisfaction ⟹ residuals ≈ 0
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // ---- UnitQuaternion -----------------------------------------------------

    #[test]
    fn prop_unit_quaternion_satisfied(q in unit_quat()) {
        let mut ctx = TestCtx::new();
        let e = ctx.entity();
        let bp = alloc_body(&mut ctx, e, [0.0, 0.0, 0.0], q);
        let store = ctx.store();

        let c = UnitQuaternion::new(ctx.cid(), e, bp.qw, bp.qx, bp.qy, bp.qz);
        let r = c.residuals(&store);
        prop_assert!(r[0].abs() < 1e-10, "UnitQuaternion residual: {}", r[0]);
    }

    // ---- Mate ---------------------------------------------------------------

    /// When both bodies are at the origin with identity rotation, mating the
    /// same local point should give zero residual.
    #[test]
    fn prop_mate_same_point_satisfied(local in local_point()) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [0.0, 0.0, 0.0], identity_quat());
        let b2 = alloc_body(&mut ctx, e2, [0.0, 0.0, 0.0], identity_quat());
        let store = ctx.store();

        let c = Mate::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz, local,
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz, local,
        );
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-10), "Mate residuals: {:?}", r);
    }

    /// Mate with translated bodies: local1 on body1 meets local2 on body2.
    #[test]
    fn prop_mate_translated_satisfied(
        tx1 in coord(), ty1 in coord(), tz1 in coord(),
        local in local_point(),
    ) {
        // Body1 at (tx1,ty1,tz1), body2 at origin, both identity rotation
        // World point from body1: local + (tx1,ty1,tz1)
        // local2 should be that world point for body2 (identity): local + (tx1,ty1,tz1)
        let local2 = [local[0] + tx1, local[1] + ty1, local[2] + tz1];

        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [tx1, ty1, tz1], identity_quat());
        let b2 = alloc_body(&mut ctx, e2, [0.0, 0.0, 0.0], identity_quat());
        let store = ctx.store();

        let c = Mate::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz, local,
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz, local2,
        );
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-8), "Mate residuals: {:?}", r);
    }

    // ---- Gear ---------------------------------------------------------------

    /// When both bodies have identity rotation, theta1 = theta2 = 0, residual = 0.
    #[test]
    fn prop_gear_no_rotation_satisfied(ratio in 0.1f64..10.0) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [0.0, 0.0, 0.0], identity_quat());
        let b2 = alloc_body(&mut ctx, e2, [0.0, 0.0, 0.0], identity_quat());
        let store = ctx.store();

        let c = Gear::new(
            ctx.cid(),
            e1, b1.qw, b1.qx, b1.qy, b1.qz, [0.0, 0.0, 1.0],
            e2, b2.qw, b2.qx, b2.qy, b2.qz, [0.0, 0.0, 1.0],
            ratio,
        );
        let r = c.residuals(&store);
        prop_assert!(r[0].abs() < 1e-10, "Gear residual: {}", r[0]);
    }

    /// Gear constraint satisfied when theta1 * ratio == theta2.
    #[test]
    fn prop_gear_ratio_satisfied(
        theta1 in -1.5f64..1.5,
        ratio in 0.5f64..4.0,
    ) {
        let theta2 = theta1 * ratio;
        // Build quaternions for rotation about z-axis
        let q1 = [(theta1 / 2.0).cos(), 0.0, 0.0, (theta1 / 2.0).sin()];
        let q2 = [(theta2 / 2.0).cos(), 0.0, 0.0, (theta2 / 2.0).sin()];

        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [0.0, 0.0, 0.0], q1);
        let b2 = alloc_body(&mut ctx, e2, [0.0, 0.0, 0.0], q2);
        let store = ctx.store();

        let c = Gear::new(
            ctx.cid(),
            e1, b1.qw, b1.qx, b1.qy, b1.qz, [0.0, 0.0, 1.0],
            e2, b2.qw, b2.qx, b2.qy, b2.qz, [0.0, 0.0, 1.0],
            ratio,
        );
        let r = c.residuals(&store);
        prop_assert!(r[0].abs() < 1e-8, "Gear residual: {} (theta1={}, theta2={}, ratio={})", r[0], theta1, theta2, ratio);
    }

    // ---- CoaxialAssembly ----------------------------------------------------

    /// Both bodies at identity, same z-axis: should be coaxial.
    #[test]
    fn prop_coaxial_assembly_aligned_satisfied(
        tz1 in coord(), tz2 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [0.0, 0.0, tz1], identity_quat());
        let b2 = alloc_body(&mut ctx, e2, [0.0, 0.0, tz2], identity_quat());
        let store = ctx.store();

        let c = CoaxialAssembly::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        );
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-10), "CoaxialAssembly residuals: {:?}", r);
    }

    // ---- Insert -------------------------------------------------------------

    /// Both bodies at identity, same z-axis, flush insertion.
    #[test]
    fn prop_insert_flush_satisfied(tz in coord()) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [0.0, 0.0, 0.0], identity_quat());
        let b2 = alloc_body(&mut ctx, e2, [0.0, 0.0, tz], identity_quat());
        let store = ctx.store();

        let c = Insert::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            tz,  // offset equals the translation
        );
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-8), "Insert residuals: {:?}", r);
    }
}

// =============================================================================
// Section 2: Jacobian correctness (analytical vs. finite differences)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_unit_quaternion_jacobian(q in unit_quat()) {
        let mut ctx = TestCtx::new();
        let e = ctx.entity();
        let bp = alloc_body(&mut ctx, e, [0.0, 0.0, 0.0], q);
        let store = ctx.store();

        let c = UnitQuaternion::new(ctx.cid(), e, bp.qw, bp.qx, bp.qy, bp.qz);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "UnitQuaternion Jacobian mismatch");
    }

    #[test]
    fn prop_mate_jacobian(
        tx1 in coord(), ty1 in coord(), tz1 in coord(),
        tx2 in coord(), ty2 in coord(), tz2 in coord(),
        q1 in unit_quat(), q2 in unit_quat(),
        local1 in local_point(), local2 in local_point(),
    ) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [tx1, ty1, tz1], q1);
        let b2 = alloc_body(&mut ctx, e2, [tx2, ty2, tz2], q2);
        let store = ctx.store();

        let c = Mate::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz, local1,
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz, local2,
        );
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "Mate Jacobian mismatch");
    }

    #[test]
    fn prop_gear_jacobian(
        theta1 in -1.5f64..1.5,
        theta2 in -1.5f64..1.5,
        ratio in 0.5f64..4.0,
    ) {
        let q1 = [(theta1 / 2.0).cos(), 0.0, 0.0, (theta1 / 2.0).sin()];
        let q2 = [(theta2 / 2.0).cos(), 0.0, 0.0, (theta2 / 2.0).sin()];

        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [0.0, 0.0, 0.0], q1);
        let b2 = alloc_body(&mut ctx, e2, [0.0, 0.0, 0.0], q2);
        let store = ctx.store();

        let c = Gear::new(
            ctx.cid(),
            e1, b1.qw, b1.qx, b1.qy, b1.qz, [0.0, 0.0, 1.0],
            e2, b2.qw, b2.qx, b2.qy, b2.qz, [0.0, 0.0, 1.0],
            ratio,
        );
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "Gear Jacobian mismatch");
    }

    // CoaxialAssembly and Insert use FD internally, so this is a consistency
    // check: their internal FD should match our external FD.
    #[test]
    fn prop_coaxial_assembly_jacobian(
        tx1 in coord(), ty1 in coord(), tz1 in coord(),
        tx2 in coord(), ty2 in coord(), tz2 in coord(),
        q1 in unit_quat(), q2 in unit_quat(),
    ) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [tx1, ty1, tz1], q1);
        let b2 = alloc_body(&mut ctx, e2, [tx2, ty2, tz2], q2);
        let store = ctx.store();

        let c = CoaxialAssembly::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        );
        prop_assert!(check_jacobian_fd(&c, &store, 1e-6, 1e-3), "CoaxialAssembly Jacobian mismatch");
    }

    #[test]
    fn prop_insert_jacobian(
        tx1 in coord(), ty1 in coord(), tz1 in coord(),
        tx2 in coord(), ty2 in coord(), tz2 in coord(),
        q1 in unit_quat(), q2 in unit_quat(),
        offset in -10.0f64..10.0,
    ) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [tx1, ty1, tz1], q1);
        let b2 = alloc_body(&mut ctx, e2, [tx2, ty2, tz2], q2);
        let store = ctx.store();

        let c = Insert::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz,
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            offset,
        );
        prop_assert!(check_jacobian_fd(&c, &store, 1e-6, 1e-3), "Insert Jacobian mismatch");
    }
}

// =============================================================================
// Section 3: Equation count and finiteness invariants
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_equation_counts(
        q1 in unit_quat(), q2 in unit_quat(),
    ) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [0.0, 0.0, 0.0], q1);
        let b2 = alloc_body(&mut ctx, e2, [0.0, 0.0, 0.0], q2);
        let store = ctx.store();

        // UnitQuaternion: 1 equation
        let c = UnitQuaternion::new(ctx.cid(), e1, b1.qw, b1.qx, b1.qy, b1.qz);
        prop_assert_eq!(c.equation_count(), 1);
        prop_assert_eq!(c.residuals(&store).len(), 1);

        // Mate: 3 equations
        let c = Mate::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz, [0.0; 3],
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz, [0.0; 3],
        );
        prop_assert_eq!(c.equation_count(), 3);
        prop_assert_eq!(c.residuals(&store).len(), 3);

        // CoaxialAssembly: 4 equations
        let c = CoaxialAssembly::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz,
            [0.0; 3], [0.0, 0.0, 1.0],
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz,
            [0.0; 3], [0.0, 0.0, 1.0],
        );
        prop_assert_eq!(c.equation_count(), 4);
        prop_assert_eq!(c.residuals(&store).len(), 4);

        // Insert: 5 equations
        let c = Insert::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz,
            [0.0; 3], [0.0, 0.0, 1.0],
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz,
            [0.0; 3], [0.0, 0.0, 1.0],
            0.0,
        );
        prop_assert_eq!(c.equation_count(), 5);
        prop_assert_eq!(c.residuals(&store).len(), 5);

        // Gear: 1 equation
        let c = Gear::new(
            ctx.cid(),
            e1, b1.qw, b1.qx, b1.qy, b1.qz, [0.0, 0.0, 1.0],
            e2, b2.qw, b2.qx, b2.qy, b2.qz, [0.0, 0.0, 1.0],
            2.0,
        );
        prop_assert_eq!(c.equation_count(), 1);
        prop_assert_eq!(c.residuals(&store).len(), 1);
    }

    /// All residuals and Jacobians should produce finite values for unit quaternions.
    #[test]
    fn prop_mate_residuals_finite(
        q1 in unit_quat(), q2 in unit_quat(),
        tx1 in coord(), ty1 in coord(), tz1 in coord(),
        tx2 in coord(), ty2 in coord(), tz2 in coord(),
        local1 in local_point(), local2 in local_point(),
    ) {
        let mut ctx = TestCtx::new();
        let e1 = ctx.entity(); let e2 = ctx.entity();
        let b1 = alloc_body(&mut ctx, e1, [tx1, ty1, tz1], q1);
        let b2 = alloc_body(&mut ctx, e2, [tx2, ty2, tz2], q2);
        let store = ctx.store();

        let c = Mate::new(
            ctx.cid(),
            e1, b1.tx, b1.ty, b1.tz, b1.qw, b1.qx, b1.qy, b1.qz, local1,
            e2, b2.tx, b2.ty, b2.tz, b2.qw, b2.qx, b2.qy, b2.qz, local2,
        );

        let r = c.residuals(&store);
        let j = c.jacobian(&store);
        prop_assert!(r.iter().all(|v| v.is_finite()), "Non-finite residual");
        prop_assert!(j.iter().all(|&(_, _, v)| v.is_finite()), "Non-finite Jacobian");
    }
}
