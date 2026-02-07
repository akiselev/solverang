//! Property-based tests for the V3 sketch3d geometric constraint system.
//!
//! These tests use proptest to verify mathematical invariants of the 3D sketch
//! constraint types, including constraint satisfaction, Jacobian correctness,
//! and coordinate transformation invariance.

use proptest::prelude::*;
use solverang::constraint::Constraint;
use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::param::ParamStore;
use solverang::sketch3d::{
    Coaxial, Coincident3D, Coplanar, Distance3D, Fixed3D, Parallel3D,
    Perpendicular3D, PointOnPlane,
};
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

fn coord() -> impl Strategy<Value = f64> { -500.0f64..500.0 }
fn positive_dist() -> impl Strategy<Value = f64> { 0.1f64..200.0 }

/// Non-degenerate 3D unit normal vector via spherical coordinates.
fn unit_normal() -> impl Strategy<Value = (f64, f64, f64)> {
    (0.01f64..std::f64::consts::PI - 0.01, 0.0f64..std::f64::consts::TAU)
        .prop_map(|(theta, phi)| {
            (theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos())
        })
}

/// Non-degenerate 3D direction vector (not too short).
fn direction3d() -> impl Strategy<Value = (f64, f64, f64)> {
    (coord(), coord(), coord())
        .prop_filter("direction too short", |&(dx, dy, dz)| {
            dx * dx + dy * dy + dz * dz > 1.0
        })
}

// =============================================================================
// Section 1: Constraint satisfaction ⟹ residuals ≈ 0
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // ---- Distance3D ---------------------------------------------------------

    #[test]
    fn prop_distance3d_satisfied(
        x1 in coord(), y1 in coord(), z1 in coord(),
        theta in 0.01f64..std::f64::consts::PI - 0.01,
        phi in 0.0f64..std::f64::consts::TAU,
        dist in positive_dist(),
    ) {
        let x2 = x1 + dist * theta.sin() * phi.cos();
        let y2 = y1 + dist * theta.sin() * phi.sin();
        let z2 = z1 + dist * theta.cos();

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0); let py1 = ctx.alloc(y1, e0); let pz1 = ctx.alloc(z1, e0);
        let px2 = ctx.alloc(x2, e1); let py2 = ctx.alloc(y2, e1); let pz2 = ctx.alloc(z2, e1);
        let store = ctx.store();

        let c = Distance3D::new(ctx.cid(), e0, px1, py1, pz1, e1, px2, py2, pz2, dist);
        let r = c.residuals(&store);
        prop_assert!(r[0].abs() < 1e-6, "Distance3D residual should be ~0, got {}", r[0]);
    }

    // ---- Coincident3D -------------------------------------------------------

    #[test]
    fn prop_coincident3d_satisfied(x in coord(), y in coord(), z in coord()) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let px1 = ctx.alloc(x, e0); let py1 = ctx.alloc(y, e0); let pz1 = ctx.alloc(z, e0);
        let px2 = ctx.alloc(x, e1); let py2 = ctx.alloc(y, e1); let pz2 = ctx.alloc(z, e1);
        let store = ctx.store();

        let c = Coincident3D::new(ctx.cid(), e0, px1, py1, pz1, e1, px2, py2, pz2);
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-12), "Coincident3D residuals: {:?}", r);
    }

    // ---- Fixed3D ------------------------------------------------------------

    #[test]
    fn prop_fixed3d_satisfied(x in coord(), y in coord(), z in coord()) {
        let mut ctx = TestCtx::new();
        let e = ctx.entity();
        let px = ctx.alloc(x, e); let py = ctx.alloc(y, e); let pz = ctx.alloc(z, e);
        let store = ctx.store();

        let c = Fixed3D::new(ctx.cid(), e, px, py, pz, [x, y, z]);
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-12), "Fixed3D residuals: {:?}", r);
    }

    // ---- PointOnPlane -------------------------------------------------------

    #[test]
    fn prop_point_on_plane_satisfied(
        x in coord(), y in coord(),
        (nx, ny, nz) in unit_normal(),
        p0x in coord(), p0y in coord(), p0z in coord(),
    ) {
        // Point on the plane: project (x, y, arbitrary_z) onto the plane through p0 with normal n
        // For a plane n.(p-p0)=0, given px,py we solve for pz:
        // nx*(px-p0x) + ny*(py-p0y) + nz*(pz-p0z) = 0
        // pz = p0z - (nx*(x-p0x) + ny*(y-p0y)) / nz
        prop_assume!(nz.abs() > 0.01);
        let pz_val = p0z - (nx * (x - p0x) + ny * (y - p0y)) / nz;

        let mut ctx = TestCtx::new();
        let ep = ctx.entity(); let epl = ctx.entity();
        let ppx = ctx.alloc(x, ep); let ppy = ctx.alloc(y, ep); let ppz = ctx.alloc(pz_val, ep);
        let pp0x = ctx.alloc(p0x, epl); let pp0y = ctx.alloc(p0y, epl); let pp0z = ctx.alloc(p0z, epl);
        let pnx = ctx.alloc(nx, epl); let pny = ctx.alloc(ny, epl); let pnz = ctx.alloc(nz, epl);
        let store = ctx.store();

        let c = PointOnPlane::new(ctx.cid(), ep, ppx, ppy, ppz, epl, pp0x, pp0y, pp0z, pnx, pny, pnz);
        let r = c.residuals(&store);
        prop_assert!(r[0].abs() < 1e-6, "PointOnPlane residual: {}", r[0]);
    }

    // ---- Coplanar -----------------------------------------------------------

    #[test]
    fn prop_coplanar_satisfied(
        (nx, ny, nz) in unit_normal(),
        p0x in coord(), p0y in coord(), p0z in coord(),
        u1 in coord(), v1 in coord(),
        u2 in coord(), v2 in coord(),
    ) {
        // Two points on the plane through p0 with normal n.
        // Build orthogonal basis in the plane:
        let (bx, by, bz) = if nx.abs() > ny.abs() {
            let inv_len = 1.0 / (nx * nx + nz * nz).sqrt();
            (-nz * inv_len, 0.0, nx * inv_len)
        } else {
            let inv_len = 1.0 / (ny * ny + nz * nz).sqrt();
            (0.0, nz * inv_len, -ny * inv_len)
        };
        // Second basis vector: n x b
        let (cx, cy, cz) = (
            ny * bz - nz * by,
            nz * bx - nx * bz,
            nx * by - ny * bx,
        );

        let x1 = p0x + u1 * bx + v1 * cx;
        let y1 = p0y + u1 * by + v1 * cy;
        let z1 = p0z + u1 * bz + v1 * cz;
        let x2 = p0x + u2 * bx + v2 * cx;
        let y2 = p0y + u2 * by + v2 * cy;
        let z2 = p0z + u2 * bz + v2 * cz;

        let mut ctx = TestCtx::new();
        let epl = ctx.entity(); let ep1 = ctx.entity(); let ep2 = ctx.entity();
        let pp0x = ctx.alloc(p0x, epl); let pp0y = ctx.alloc(p0y, epl); let pp0z = ctx.alloc(p0z, epl);
        let pnx = ctx.alloc(nx, epl); let pny = ctx.alloc(ny, epl); let pnz = ctx.alloc(nz, epl);
        let px1 = ctx.alloc(x1, ep1); let py1 = ctx.alloc(y1, ep1); let pz1 = ctx.alloc(z1, ep1);
        let px2 = ctx.alloc(x2, ep2); let py2 = ctx.alloc(y2, ep2); let pz2 = ctx.alloc(z2, ep2);
        let store = ctx.store();

        let c = Coplanar::new(
            ctx.cid(), epl, pp0x, pp0y, pp0z, pnx, pny, pnz,
            &[(ep1, px1, py1, pz1), (ep2, px2, py2, pz2)],
        );
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-4), "Coplanar residuals: {:?}", r);
    }

    // ---- Parallel3D ---------------------------------------------------------

    #[test]
    fn prop_parallel3d_satisfied(
        x1 in coord(), y1 in coord(), z1 in coord(),
        (dx, dy, dz) in direction3d(),
        x3 in coord(), y3 in coord(), z3 in coord(),
        scale in 0.1f64..10.0,
    ) {
        // Line 1: (x1,y1,z1) -> (x1+dx,y1+dy,z1+dz)
        // Line 2: (x3,y3,z3) -> (x3+scale*dx,y3+scale*dy,z3+scale*dz) -- parallel
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let l1x1 = ctx.alloc(x1, e0); let l1y1 = ctx.alloc(y1, e0); let l1z1 = ctx.alloc(z1, e0);
        let l1x2 = ctx.alloc(x1 + dx, e0); let l1y2 = ctx.alloc(y1 + dy, e0); let l1z2 = ctx.alloc(z1 + dz, e0);
        let l2x1 = ctx.alloc(x3, e1); let l2y1 = ctx.alloc(y3, e1); let l2z1 = ctx.alloc(z3, e1);
        let l2x2 = ctx.alloc(x3 + scale * dx, e1); let l2y2 = ctx.alloc(y3 + scale * dy, e1); let l2z2 = ctx.alloc(z3 + scale * dz, e1);
        let store = ctx.store();

        let c = Parallel3D::new(ctx.cid(), e0, l1x1, l1y1, l1z1, l1x2, l1y2, l1z2, e1, l2x1, l2y1, l2z1, l2x2, l2y2, l2z2);
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-4), "Parallel3D residuals: {:?}", r);
    }

    // ---- Perpendicular3D ----------------------------------------------------

    #[test]
    fn prop_perpendicular3d_satisfied(
        x1 in coord(), y1 in coord(), z1 in coord(),
        (dx, dy, dz) in direction3d(),
        x3 in coord(), y3 in coord(), z3 in coord(),
    ) {
        // Generate a direction perpendicular to (dx, dy, dz)
        let (px, py, pz) = if dx.abs() > dz.abs() {
            (-dy, dx, 0.0)
        } else {
            (0.0, -dz, dy)
        };
        let perp_len = (px * px + py * py + pz * pz).sqrt();
        prop_assume!(perp_len > 0.01);

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let l1x1 = ctx.alloc(x1, e0); let l1y1 = ctx.alloc(y1, e0); let l1z1 = ctx.alloc(z1, e0);
        let l1x2 = ctx.alloc(x1 + dx, e0); let l1y2 = ctx.alloc(y1 + dy, e0); let l1z2 = ctx.alloc(z1 + dz, e0);
        let l2x1 = ctx.alloc(x3, e1); let l2y1 = ctx.alloc(y3, e1); let l2z1 = ctx.alloc(z3, e1);
        let l2x2 = ctx.alloc(x3 + px, e1); let l2y2 = ctx.alloc(y3 + py, e1); let l2z2 = ctx.alloc(z3 + pz, e1);
        let store = ctx.store();

        let c = Perpendicular3D::new(ctx.cid(), e0, l1x1, l1y1, l1z1, l1x2, l1y2, l1z2, e1, l2x1, l2y1, l2z1, l2x2, l2y2, l2z2);
        let r = c.residuals(&store);
        prop_assert!(r[0].abs() < 1e-4, "Perpendicular3D residual: {}", r[0]);
    }

    // ---- Coaxial ------------------------------------------------------------

    #[test]
    fn prop_coaxial_satisfied(
        p1x in coord(), p1y in coord(), p1z in coord(),
        (dx, dy, dz) in direction3d(),
        t in coord(),
        scale in 0.1f64..10.0,
    ) {
        // Two axes on the same line:
        // Axis 1: point (p1x, p1y, p1z), direction (dx, dy, dz)
        // Axis 2: point (p1x + t*dx, p1y + t*dy, p1z + t*dz), direction (scale*dx, scale*dy, scale*dz)
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let pp1x = ctx.alloc(p1x, e0); let pp1y = ctx.alloc(p1y, e0); let pp1z = ctx.alloc(p1z, e0);
        let pd1x = ctx.alloc(dx, e0); let pd1y = ctx.alloc(dy, e0); let pd1z = ctx.alloc(dz, e0);
        let pp2x = ctx.alloc(p1x + t * dx, e1); let pp2y = ctx.alloc(p1y + t * dy, e1); let pp2z = ctx.alloc(p1z + t * dz, e1);
        let pd2x = ctx.alloc(scale * dx, e1); let pd2y = ctx.alloc(scale * dy, e1); let pd2z = ctx.alloc(scale * dz, e1);
        let store = ctx.store();

        let c = Coaxial::new(ctx.cid(), e0, pp1x, pp1y, pp1z, pd1x, pd1y, pd1z, e1, pp2x, pp2y, pp2z, pd2x, pd2y, pd2z);
        let r = c.residuals(&store);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-3), "Coaxial residuals: {:?}", r);
    }
}

// =============================================================================
// Section 2: Jacobian correctness (analytical vs. finite differences)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_distance3d_jacobian(
        x1 in coord(), y1 in coord(), z1 in coord(),
        x2 in coord(), y2 in coord(), z2 in coord(),
        dist in positive_dist(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0); let py1 = ctx.alloc(y1, e0); let pz1 = ctx.alloc(z1, e0);
        let px2 = ctx.alloc(x2, e1); let py2 = ctx.alloc(y2, e1); let pz2 = ctx.alloc(z2, e1);
        let store = ctx.store();

        let c = Distance3D::new(ctx.cid(), e0, px1, py1, pz1, e1, px2, py2, pz2, dist);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "Distance3D Jacobian mismatch");
    }

    #[test]
    fn prop_coincident3d_jacobian(
        x1 in coord(), y1 in coord(), z1 in coord(),
        x2 in coord(), y2 in coord(), z2 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0); let py1 = ctx.alloc(y1, e0); let pz1 = ctx.alloc(z1, e0);
        let px2 = ctx.alloc(x2, e1); let py2 = ctx.alloc(y2, e1); let pz2 = ctx.alloc(z2, e1);
        let store = ctx.store();

        let c = Coincident3D::new(ctx.cid(), e0, px1, py1, pz1, e1, px2, py2, pz2);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-5), "Coincident3D Jacobian mismatch");
    }

    #[test]
    fn prop_fixed3d_jacobian(
        x in coord(), y in coord(), z in coord(),
        tx in coord(), ty in coord(), tz in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e = ctx.entity();
        let px = ctx.alloc(x, e); let py = ctx.alloc(y, e); let pz = ctx.alloc(z, e);
        let store = ctx.store();

        let c = Fixed3D::new(ctx.cid(), e, px, py, pz, [tx, ty, tz]);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-5), "Fixed3D Jacobian mismatch");
    }

    #[test]
    fn prop_point_on_plane_jacobian(
        px in coord(), py in coord(), pz in coord(),
        p0x in coord(), p0y in coord(), p0z in coord(),
        (nx, ny, nz) in unit_normal(),
    ) {
        let mut ctx = TestCtx::new();
        let ep = ctx.entity(); let epl = ctx.entity();
        let ppx = ctx.alloc(px, ep); let ppy = ctx.alloc(py, ep); let ppz = ctx.alloc(pz, ep);
        let pp0x = ctx.alloc(p0x, epl); let pp0y = ctx.alloc(p0y, epl); let pp0z = ctx.alloc(p0z, epl);
        let pnx = ctx.alloc(nx, epl); let pny = ctx.alloc(ny, epl); let pnz = ctx.alloc(nz, epl);
        let store = ctx.store();

        let c = PointOnPlane::new(ctx.cid(), ep, ppx, ppy, ppz, epl, pp0x, pp0y, pp0z, pnx, pny, pnz);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "PointOnPlane Jacobian mismatch");
    }

    #[test]
    fn prop_coplanar_jacobian(
        p0x in coord(), p0y in coord(), p0z in coord(),
        nx in -1.0f64..1.0, ny in -1.0f64..1.0, nz in -1.0f64..1.0,
        px in coord(), py in coord(), pz in coord(),
    ) {
        let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
        prop_assume!(nlen > 0.1);

        let mut ctx = TestCtx::new();
        let epl = ctx.entity(); let ep = ctx.entity();
        let pp0x = ctx.alloc(p0x, epl); let pp0y = ctx.alloc(p0y, epl); let pp0z = ctx.alloc(p0z, epl);
        let pnx = ctx.alloc(nx, epl); let pny = ctx.alloc(ny, epl); let pnz = ctx.alloc(nz, epl);
        let ppx = ctx.alloc(px, ep); let ppy = ctx.alloc(py, ep); let ppz = ctx.alloc(pz, ep);
        let store = ctx.store();

        let c = Coplanar::new(ctx.cid(), epl, pp0x, pp0y, pp0z, pnx, pny, pnz, &[(ep, ppx, ppy, ppz)]);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "Coplanar Jacobian mismatch");
    }

    #[test]
    fn prop_parallel3d_jacobian(
        x1 in coord(), y1 in coord(), z1 in coord(),
        x2 in coord(), y2 in coord(), z2 in coord(),
        x3 in coord(), y3 in coord(), z3 in coord(),
        x4 in coord(), y4 in coord(), z4 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let l1x1 = ctx.alloc(x1, e0); let l1y1 = ctx.alloc(y1, e0); let l1z1 = ctx.alloc(z1, e0);
        let l1x2 = ctx.alloc(x2, e0); let l1y2 = ctx.alloc(y2, e0); let l1z2 = ctx.alloc(z2, e0);
        let l2x1 = ctx.alloc(x3, e1); let l2y1 = ctx.alloc(y3, e1); let l2z1 = ctx.alloc(z3, e1);
        let l2x2 = ctx.alloc(x4, e1); let l2y2 = ctx.alloc(y4, e1); let l2z2 = ctx.alloc(z4, e1);
        let store = ctx.store();

        let c = Parallel3D::new(ctx.cid(), e0, l1x1, l1y1, l1z1, l1x2, l1y2, l1z2, e1, l2x1, l2y1, l2z1, l2x2, l2y2, l2z2);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "Parallel3D Jacobian mismatch");
    }

    #[test]
    fn prop_perpendicular3d_jacobian(
        x1 in coord(), y1 in coord(), z1 in coord(),
        x2 in coord(), y2 in coord(), z2 in coord(),
        x3 in coord(), y3 in coord(), z3 in coord(),
        x4 in coord(), y4 in coord(), z4 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let l1x1 = ctx.alloc(x1, e0); let l1y1 = ctx.alloc(y1, e0); let l1z1 = ctx.alloc(z1, e0);
        let l1x2 = ctx.alloc(x2, e0); let l1y2 = ctx.alloc(y2, e0); let l1z2 = ctx.alloc(z2, e0);
        let l2x1 = ctx.alloc(x3, e1); let l2y1 = ctx.alloc(y3, e1); let l2z1 = ctx.alloc(z3, e1);
        let l2x2 = ctx.alloc(x4, e1); let l2y2 = ctx.alloc(y4, e1); let l2z2 = ctx.alloc(z4, e1);
        let store = ctx.store();

        let c = Perpendicular3D::new(ctx.cid(), e0, l1x1, l1y1, l1z1, l1x2, l1y2, l1z2, e1, l2x1, l2y1, l2z1, l2x2, l2y2, l2z2);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "Perpendicular3D Jacobian mismatch");
    }

    #[test]
    fn prop_coaxial_jacobian(
        p1x in coord(), p1y in coord(), p1z in coord(),
        d1x in -10.0f64..10.0, d1y in -10.0f64..10.0, d1z in -10.0f64..10.0,
        p2x in coord(), p2y in coord(), p2z in coord(),
        d2x in -10.0f64..10.0, d2y in -10.0f64..10.0, d2z in -10.0f64..10.0,
    ) {
        let d1_len = (d1x * d1x + d1y * d1y + d1z * d1z).sqrt();
        let d2_len = (d2x * d2x + d2y * d2y + d2z * d2z).sqrt();
        prop_assume!(d1_len > 0.1 && d2_len > 0.1);

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let pp1x = ctx.alloc(p1x, e0); let pp1y = ctx.alloc(p1y, e0); let pp1z = ctx.alloc(p1z, e0);
        let pd1x = ctx.alloc(d1x, e0); let pd1y = ctx.alloc(d1y, e0); let pd1z = ctx.alloc(d1z, e0);
        let pp2x = ctx.alloc(p2x, e1); let pp2y = ctx.alloc(p2y, e1); let pp2z = ctx.alloc(p2z, e1);
        let pd2x = ctx.alloc(d2x, e1); let pd2y = ctx.alloc(d2y, e1); let pd2z = ctx.alloc(d2z, e1);
        let store = ctx.store();

        let c = Coaxial::new(ctx.cid(), e0, pp1x, pp1y, pp1z, pd1x, pd1y, pd1z, e1, pp2x, pp2y, pp2z, pd2x, pd2y, pd2z);
        prop_assert!(check_jacobian_fd(&c, &store, 1e-7, 1e-4), "Coaxial Jacobian mismatch");
    }
}

// =============================================================================
// Section 3: Residual/Jacobian finiteness
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_distance3d_residuals_finite(
        x1 in coord(), y1 in coord(), z1 in coord(),
        x2 in coord(), y2 in coord(), z2 in coord(),
        dist in positive_dist(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0); let py1 = ctx.alloc(y1, e0); let pz1 = ctx.alloc(z1, e0);
        let px2 = ctx.alloc(x2, e1); let py2 = ctx.alloc(y2, e1); let pz2 = ctx.alloc(z2, e1);
        let store = ctx.store();

        let c = Distance3D::new(ctx.cid(), e0, px1, py1, pz1, e1, px2, py2, pz2, dist);
        let r = c.residuals(&store);
        let j = c.jacobian(&store);

        prop_assert!(r.iter().all(|v| v.is_finite()), "Non-finite residual");
        prop_assert!(j.iter().all(|&(_, _, v)| v.is_finite()), "Non-finite Jacobian");
    }

    #[test]
    fn prop_all_3d_equation_counts(
        x1 in coord(), y1 in coord(), z1 in coord(),
        x2 in coord(), y2 in coord(), z2 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0); let py1 = ctx.alloc(y1, e0); let pz1 = ctx.alloc(z1, e0);
        let px2 = ctx.alloc(x2, e1); let py2 = ctx.alloc(y2, e1); let pz2 = ctx.alloc(z2, e1);
        let store = ctx.store();

        // Distance3D: 1 equation
        let c = Distance3D::new(ctx.cid(), e0, px1, py1, pz1, e1, px2, py2, pz2, 1.0);
        prop_assert_eq!(c.equation_count(), 1);
        prop_assert_eq!(c.residuals(&store).len(), 1);

        // Coincident3D: 3 equations
        let c = Coincident3D::new(ctx.cid(), e0, px1, py1, pz1, e1, px2, py2, pz2);
        prop_assert_eq!(c.equation_count(), 3);
        prop_assert_eq!(c.residuals(&store).len(), 3);
    }

    /// Distance3D is symmetric: Distance(p1→p2) == Distance(p2→p1).
    #[test]
    fn prop_distance3d_symmetric(
        x1 in coord(), y1 in coord(), z1 in coord(),
        x2 in coord(), y2 in coord(), z2 in coord(),
        dist in positive_dist(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity(); let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0); let py1 = ctx.alloc(y1, e0); let pz1 = ctx.alloc(z1, e0);
        let px2 = ctx.alloc(x2, e1); let py2 = ctx.alloc(y2, e1); let pz2 = ctx.alloc(z2, e1);
        let store = ctx.store();

        let c12 = Distance3D::new(ctx.cid(), e0, px1, py1, pz1, e1, px2, py2, pz2, dist);
        let c21 = Distance3D::new(ctx.cid(), e1, px2, py2, pz2, e0, px1, py1, pz1, dist);

        let r12 = c12.residuals(&store);
        let r21 = c21.residuals(&store);
        prop_assert!((r12[0] - r21[0]).abs() < 1e-10, "Distance3D not symmetric");
    }

    /// 3D translation invariance: translating all points doesn't change residuals.
    #[test]
    fn prop_parallel3d_translation_invariant(
        x1 in coord(), y1 in coord(), z1 in coord(),
        x2 in coord(), y2 in coord(), z2 in coord(),
        x3 in coord(), y3 in coord(), z3 in coord(),
        x4 in coord(), y4 in coord(), z4 in coord(),
        tx in -100.0f64..100.0, ty in -100.0f64..100.0, tz in -100.0f64..100.0,
    ) {
        // Original
        let mut ctx1 = TestCtx::new();
        let e0 = ctx1.entity(); let e1 = ctx1.entity();
        let l1x1 = ctx1.alloc(x1, e0); let l1y1 = ctx1.alloc(y1, e0); let l1z1 = ctx1.alloc(z1, e0);
        let l1x2 = ctx1.alloc(x2, e0); let l1y2 = ctx1.alloc(y2, e0); let l1z2 = ctx1.alloc(z2, e0);
        let l2x1 = ctx1.alloc(x3, e1); let l2y1 = ctx1.alloc(y3, e1); let l2z1 = ctx1.alloc(z3, e1);
        let l2x2 = ctx1.alloc(x4, e1); let l2y2 = ctx1.alloc(y4, e1); let l2z2 = ctx1.alloc(z4, e1);
        let s1 = ctx1.store();
        let c1 = Parallel3D::new(ctx1.cid(), e0, l1x1, l1y1, l1z1, l1x2, l1y2, l1z2, e1, l2x1, l2y1, l2z1, l2x2, l2y2, l2z2);

        // Translated
        let mut ctx2 = TestCtx::new();
        let e0t = ctx2.entity(); let e1t = ctx2.entity();
        let l1x1t = ctx2.alloc(x1 + tx, e0t); let l1y1t = ctx2.alloc(y1 + ty, e0t); let l1z1t = ctx2.alloc(z1 + tz, e0t);
        let l1x2t = ctx2.alloc(x2 + tx, e0t); let l1y2t = ctx2.alloc(y2 + ty, e0t); let l1z2t = ctx2.alloc(z2 + tz, e0t);
        let l2x1t = ctx2.alloc(x3 + tx, e1t); let l2y1t = ctx2.alloc(y3 + ty, e1t); let l2z1t = ctx2.alloc(z3 + tz, e1t);
        let l2x2t = ctx2.alloc(x4 + tx, e1t); let l2y2t = ctx2.alloc(y4 + ty, e1t); let l2z2t = ctx2.alloc(z4 + tz, e1t);
        let s2 = ctx2.store();
        let c2 = Parallel3D::new(ctx2.cid(), e0t, l1x1t, l1y1t, l1z1t, l1x2t, l1y2t, l1z2t, e1t, l2x1t, l2y1t, l2z1t, l2x2t, l2y2t, l2z2t);

        let r1 = c1.residuals(&s1);
        let r2 = c2.residuals(&s2);

        for (a, b) in r1.iter().zip(r2.iter()) {
            prop_assert!((a - b).abs() < 1e-6, "Parallel3D not translation-invariant: {} vs {}", a, b);
        }
    }
}
