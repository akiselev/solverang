//! Tests for Milestone 13: Ellipse2D, Spline2D, EqualRadius, Collinear.

mod common;

use proptest::prelude::*;
use solverang::constraint::Constraint;
use solverang::entity::Entity;
use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::param::ParamStore;
use solverang::sketch2d::{Collinear, Ellipse2D, EqualRadius, Spline2D};
use solverang::ConstraintSystem;

use common::check_jacobian_fd;

// =============================================================================
// ID allocation helpers (mirrors sketch2d_property_tests.rs)
// =============================================================================

struct TestCtx {
    sys: ConstraintSystem,
}

impl TestCtx {
    fn new() -> Self {
        Self {
            sys: ConstraintSystem::new(),
        }
    }

    fn entity(&mut self) -> EntityId {
        self.sys.alloc_entity_id()
    }

    fn cid(&mut self) -> ConstraintId {
        self.sys.alloc_constraint_id()
    }

    fn alloc(&mut self, value: f64, owner: EntityId) -> ParamId {
        self.sys.alloc_param(value, owner)
    }

    fn store(&self) -> ParamStore {
        self.sys.params().clone()
    }
}

// =============================================================================
// 1. Ellipse2D entity test
// =============================================================================

#[test]
fn test_ellipse2d_params_count() {
    let mut ctx = TestCtx::new();
    let eid = ctx.entity();
    let cx = ctx.alloc(1.0, eid);
    let cy = ctx.alloc(2.0, eid);
    let a = ctx.alloc(5.0, eid);
    let b = ctx.alloc(3.0, eid);
    let theta = ctx.alloc(0.0, eid);

    let ellipse = Ellipse2D::new(eid, cx, cy, a, b, theta);

    assert_eq!(ellipse.params().len(), 5, "Ellipse2D must have 5 DOF");
    assert_eq!(ellipse.name(), "Ellipse2D");
    assert_eq!(ellipse.id(), eid);
    assert_eq!(ellipse.center_x(), cx);
    assert_eq!(ellipse.center_y(), cy);
    assert_eq!(ellipse.semi_major(), a);
    assert_eq!(ellipse.semi_minor(), b);
    assert_eq!(ellipse.rotation(), theta);
}

/// At t=0: x(0) = cx + a*cos(theta), y(0) = cy + a*sin(theta)
/// because cos(0)=1, sin(0)=0 so the b*sin(t) terms vanish.
#[test]
fn test_ellipse2d_point_at_t0() {
    let mut ctx = TestCtx::new();
    let eid = ctx.entity();
    let cx = ctx.alloc(3.0, eid);
    let cy = ctx.alloc(-1.0, eid);
    let a = ctx.alloc(4.0, eid);
    let b = ctx.alloc(2.0, eid);
    let theta = ctx.alloc(std::f64::consts::FRAC_PI_4, eid);
    let store = ctx.store();

    let ellipse = Ellipse2D::new(eid, cx, cy, a, b, theta);
    let (px, py) = ellipse.point_at(&store, 0.0);

    // t=0 => cos(t)=1, sin(t)=0
    // x = cx + a*cos(theta), y = cy + a*sin(theta)
    let th = std::f64::consts::FRAC_PI_4;
    let expected_x = 3.0 + 4.0 * th.cos();
    let expected_y = -1.0 + 4.0 * th.sin();
    assert!(
        (px - expected_x).abs() < 1e-12,
        "px={} expected={}",
        px,
        expected_x
    );
    assert!(
        (py - expected_y).abs() < 1e-12,
        "py={} expected={}",
        py,
        expected_y
    );
}

/// Degenerate ellipse (a == b) is a circle: distance from center must be constant = a.
#[test]
fn test_ellipse2d_degenerate_is_circle() {
    let mut ctx = TestCtx::new();
    let eid = ctx.entity();
    let cx = ctx.alloc(0.0, eid);
    let cy = ctx.alloc(0.0, eid);
    let a = ctx.alloc(3.0, eid);
    let b = ctx.alloc(3.0, eid); // a == b => circle
    let theta = ctx.alloc(0.7, eid);
    let store = ctx.store();

    let ellipse = Ellipse2D::new(eid, cx, cy, a, b, theta);

    for i in 0..8 {
        let t = i as f64 * std::f64::consts::FRAC_PI_4;
        let (px, py) = ellipse.point_at(&store, t);
        // Distance from center must equal radius a=3.
        let dist = (px * px + py * py).sqrt();
        assert!(
            (dist - 3.0).abs() < 1e-12,
            "degenerate ellipse at t={}: dist={}, expected 3.0",
            t,
            dist
        );
    }
}

// =============================================================================
// 2. Spline2D entity test
// =============================================================================

#[test]
fn test_spline2d_four_control_points() {
    let mut ctx = TestCtx::new();
    let eid = ctx.entity();

    // Allocate 4 control points: (0,0), (1,2), (3,1), (4,3)
    let pts: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 2.0), (3.0, 1.0), (4.0, 3.0)];
    let param_pairs: Vec<(ParamId, ParamId)> = pts
        .iter()
        .map(|&(x, y)| (ctx.alloc(x, eid), ctx.alloc(y, eid)))
        .collect();

    // Snapshot ParamIds before move
    let px0 = param_pairs[0].0;
    let py0 = param_pairs[0].1;
    let px2 = param_pairs[2].0;
    let py2 = param_pairs[2].1;
    let store = ctx.store();

    let spline = Spline2D::new(eid, param_pairs);

    assert_eq!(spline.params().len(), 8, "4 points × 2 = 8 params");
    assert_eq!(spline.n_points(), 4);
    assert_eq!(spline.name(), "Spline2D");
    assert_eq!(spline.id(), eid);

    // Accessor correctness
    assert_eq!(spline.control_point_x(0), px0);
    assert_eq!(spline.control_point_y(0), py0);
    assert_eq!(spline.control_point_x(2), px2);
    assert_eq!(spline.control_point_y(2), py2);

    // Value access
    let (gx, gy) = spline.get_control_point(&store, 1);
    assert!((gx - 1.0).abs() < 1e-15);
    assert!((gy - 2.0).abs() < 1e-15);
}

// =============================================================================
// 3. EqualRadius satisfaction test
// =============================================================================

#[test]
fn test_equal_radius_satisfied() {
    let mut ctx = TestCtx::new();
    let e1 = ctx.entity();
    let e2 = ctx.entity();
    let r1 = ctx.alloc(5.0, e1);
    let r2 = ctx.alloc(5.0, e2); // same radius
    let store = ctx.store();

    let c = EqualRadius::new(ctx.cid(), e1, r1, e2, r2);
    let res = c.residuals(&store);
    assert!(
        res[0].abs() < 1e-15,
        "equal radii must give residual=0, got {}",
        res[0]
    );
}

#[test]
fn test_equal_radius_unsatisfied() {
    let mut ctx = TestCtx::new();
    let e1 = ctx.entity();
    let e2 = ctx.entity();
    let r1 = ctx.alloc(7.0, e1);
    let r2 = ctx.alloc(3.0, e2); // different radii
    let store = ctx.store();

    let c = EqualRadius::new(ctx.cid(), e1, r1, e2, r2);
    let res = c.residuals(&store);
    // R = r1 - r2 = 4.0
    assert!(
        (res[0] - 4.0).abs() < 1e-15,
        "residual should be 4.0, got {}",
        res[0]
    );
}

// =============================================================================
// 4. EqualRadius Jacobian FD test
// =============================================================================

#[test]
fn test_equal_radius_jacobian() {
    let mut ctx = TestCtx::new();
    let e1 = ctx.entity();
    let e2 = ctx.entity();
    let r1 = ctx.alloc(4.0, e1);
    let r2 = ctx.alloc(7.0, e2);
    let store = ctx.store();

    let c = EqualRadius::new(ctx.cid(), e1, r1, e2, r2);
    assert!(
        check_jacobian_fd(&c, &store, 1e-7, 1e-6),
        "EqualRadius analytical Jacobian must match finite differences"
    );
}

// =============================================================================
// 5. Collinear satisfaction test
// =============================================================================

#[test]
fn test_collinear_satisfied() {
    // Three collinear points on y = x line: (0,0), (1,1), (2,2)
    let mut ctx = TestCtx::new();
    let e1 = ctx.entity();
    let e2 = ctx.entity();
    let e3 = ctx.entity();
    let x1 = ctx.alloc(0.0, e1);
    let y1 = ctx.alloc(0.0, e1);
    let x2 = ctx.alloc(1.0, e2);
    let y2 = ctx.alloc(1.0, e2);
    let x3 = ctx.alloc(2.0, e3);
    let y3 = ctx.alloc(2.0, e3);
    let store = ctx.store();

    let c = Collinear::new(ctx.cid(), e1, x1, y1, e2, x2, y2, e3, x3, y3);
    let res = c.residuals(&store);
    assert!(
        res[0].abs() < 1e-15,
        "collinear points must give residual=0, got {}",
        res[0]
    );
}

#[test]
fn test_collinear_unsatisfied() {
    // Three non-collinear points: (0,0), (1,0), (0,1) form a right triangle.
    let mut ctx = TestCtx::new();
    let e1 = ctx.entity();
    let e2 = ctx.entity();
    let e3 = ctx.entity();
    let x1 = ctx.alloc(0.0, e1);
    let y1 = ctx.alloc(0.0, e1);
    let x2 = ctx.alloc(1.0, e2);
    let y2 = ctx.alloc(0.0, e2);
    let x3 = ctx.alloc(0.0, e3);
    let y3 = ctx.alloc(1.0, e3);
    let store = ctx.store();

    let c = Collinear::new(ctx.cid(), e1, x1, y1, e2, x2, y2, e3, x3, y3);
    let res = c.residuals(&store);
    // R = (1-0)*(1-0) - (0-0)*(0-0) = 1
    assert!(
        res[0].abs() > 0.5,
        "non-collinear points must give nonzero residual, got {}",
        res[0]
    );
}

// =============================================================================
// 6. Collinear Jacobian FD test
// =============================================================================

#[test]
fn test_collinear_jacobian() {
    // Perturbed-away-from-collinear to get non-degenerate Jacobian entries.
    let mut ctx = TestCtx::new();
    let e1 = ctx.entity();
    let e2 = ctx.entity();
    let e3 = ctx.entity();
    let x1 = ctx.alloc(0.0, e1);
    let y1 = ctx.alloc(0.0, e1);
    let x2 = ctx.alloc(2.0, e2);
    let y2 = ctx.alloc(1.0, e2);
    let x3 = ctx.alloc(5.0, e3);
    let y3 = ctx.alloc(-2.0, e3);
    let store = ctx.store();

    let c = Collinear::new(ctx.cid(), e1, x1, y1, e2, x2, y2, e3, x3, y3);
    assert!(
        check_jacobian_fd(&c, &store, 1e-7, 1e-6),
        "Collinear analytical Jacobian must match finite differences"
    );
}

// =============================================================================
// 7. Proptest: random collinear configuration gives residual ≈ 0
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Build three collinear points as p1, p2 = p1 + t*dir, p3 = p1 + s*dir.
    /// The cross-product residual must vanish exactly (up to float rounding).
    #[test]
    fn prop_collinear_satisfied(
        px in -200.0f64..200.0,
        py in -200.0f64..200.0,
        dx in -10.0f64..10.0,
        dy in -10.0f64..10.0,
        t in -5.0f64..5.0,
        s in -5.0f64..5.0,
    ) {
        // Degenerate direction vector: skip (both components near zero).
        prop_assume!(dx.abs() + dy.abs() > 0.01);

        let mut ctx = TestCtx::new();
        let e1 = ctx.entity();
        let e2 = ctx.entity();
        let e3 = ctx.entity();

        // p1 = (px, py)
        let x1 = ctx.alloc(px, e1);
        let y1 = ctx.alloc(py, e1);
        // p2 = p1 + t * dir
        let x2 = ctx.alloc(px + t * dx, e2);
        let y2 = ctx.alloc(py + t * dy, e2);
        // p3 = p1 + s * dir
        let x3 = ctx.alloc(px + s * dx, e3);
        let y3 = ctx.alloc(py + s * dy, e3);
        let store = ctx.store();

        let c = Collinear::new(ctx.cid(), e1, x1, y1, e2, x2, y2, e3, x3, y3);
        let res = c.residuals(&store);

        // Magnitude of the residual scales with |dir|^2 * |t-s|.
        // Use a generous tolerance proportional to coordinates.
        let scale = (dx * dx + dy * dy) * (t.abs() + s.abs() + 1.0) * (px.abs() + py.abs() + 1.0);
        let tol = 1e-10 * scale.max(1.0);

        prop_assert!(
            res[0].abs() < tol,
            "collinear residual should be ~0, got {} (tol={})",
            res[0], tol
        );
    }
}
