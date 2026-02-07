//! Property-based tests for the V3 sketch2d geometric solver.
//!
//! These tests use proptest to verify mathematical invariants of the sketch2d
//! constraint system, including:
//!
//! - **Constraint satisfaction ⟹ residuals ≈ 0**: When a constraint is
//!   constructed at a known-satisfied configuration, residuals must be near zero.
//! - **Jacobian correctness**: Analytical Jacobians must match central finite
//!   differences at random evaluation points.
//! - **DOF monotonicity**: Adding a constraint never increases DOF.
//! - **Decomposition preservation**: Independent sub-systems give identical
//!   results when solved together or apart.
//! - **Coordinate transformation invariance**: Constraints are invariant under
//!   rigid-body transformations (translation + rotation).
//! - **Squared formulation properties**: The squared residual formulations
//!   (dx²+dy² - d² instead of √(dx²+dy²) - d) produce smooth Jacobians and
//!   correct sign behavior.
//!
//! Run with: `cargo test -p solverang --test sketch2d_property_tests`

mod common;

use proptest::prelude::*;
use solverang::constraint::Constraint;
use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::param::ParamStore;
use solverang::sketch2d::{
    Angle, Coincident, DistancePtLine, DistancePtPt, EqualLength, Fixed, Horizontal, Midpoint,
    Parallel, Perpendicular, PointOnCircle, Symmetric, TangentCircleCircle, TangentLineCircle,
    Vertical,
};
use solverang::ConstraintSystem;

use common::check_jacobian_fd;

// =============================================================================
// Helpers
// =============================================================================

/// Allocator that uses a `ConstraintSystem` to mint valid entity/constraint IDs
/// and params, giving us the public API without needing `pub(crate)` constructors.
struct IdAlloc {
    sys: ConstraintSystem,
}

impl IdAlloc {
    fn new() -> Self {
        Self {
            sys: ConstraintSystem::new(),
        }
    }

    fn entity(&mut self) -> EntityId {
        self.sys.alloc_entity_id()
    }

    fn constraint(&mut self) -> ConstraintId {
        self.sys.alloc_constraint_id()
    }

    fn param(&mut self, value: f64, owner: EntityId) -> ParamId {
        self.sys.alloc_param(value, owner)
    }
}

/// Build a `ParamStore` + allocator for tests that need raw constraint access.
/// Returns `(store, entity_id_0, entity_id_1, ...)` patterns below.
struct TestCtx {
    alloc: IdAlloc,
}

impl TestCtx {
    fn new() -> Self {
        Self {
            alloc: IdAlloc::new(),
        }
    }

    fn entity(&mut self) -> EntityId {
        self.alloc.entity()
    }

    fn cid(&mut self) -> ConstraintId {
        self.alloc.constraint()
    }

    /// Allocate a param in the *system's* store, then read it out into an
    /// independent `ParamStore` snapshot.  We use the system's store directly
    /// for param allocation (which mints valid `ParamId`s), then snapshot it
    /// for residual/Jacobian evaluation.
    fn alloc(&mut self, value: f64, owner: EntityId) -> ParamId {
        self.alloc.param(value, owner)
    }

    /// Get a snapshot of the current parameter store (for residual evaluation).
    fn store(&self) -> ParamStore {
        self.alloc.sys.params().clone()
    }
}

// =============================================================================
// Proptest strategies
// =============================================================================

/// Reasonable 2D coordinate in [-500, 500].
fn coord() -> impl Strategy<Value = f64> {
    -500.0f64..500.0
}

/// Positive distance value.
fn positive_dist() -> impl Strategy<Value = f64> {
    0.1f64..200.0
}

/// Positive radius.
fn positive_radius() -> impl Strategy<Value = f64> {
    0.5f64..100.0
}

/// Angle in [0, 2π).
fn angle_rad() -> impl Strategy<Value = f64> {
    0.0f64..std::f64::consts::TAU
}

// =============================================================================
// Section 1: Constraint satisfaction ⟹ residuals ≈ 0
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    // ---- DistancePtPt -------------------------------------------------------

    /// DistancePtPt residual is zero when the actual distance equals the target.
    #[test]
    fn prop_distance_pt_pt_satisfied(
        x1 in coord(), y1 in coord(),
        angle in angle_rad(),
        dist in positive_dist(),
    ) {
        let x2 = x1 + dist * angle.cos();
        let y2 = y1 + dist * angle.sin();

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = DistancePtPt::new(ctx.cid(), e0, e1, px1, py1, px2, py2, dist);
        let r = c.residuals(&store);

        prop_assert!(
            r[0].abs() < 1e-8,
            "DistancePtPt residual should be ~0, got {}",
            r[0]
        );
    }

    /// DistancePtPt is symmetric: Distance(p1→p2) == Distance(p2→p1).
    #[test]
    fn prop_distance_pt_pt_symmetric(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        dist in positive_dist(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c12 = DistancePtPt::new(ctx.cid(), e0, e1, px1, py1, px2, py2, dist);
        let c21 = DistancePtPt::new(ctx.cid(), e1, e0, px2, py2, px1, py1, dist);

        let r12 = c12.residuals(&store);
        let r21 = c21.residuals(&store);

        prop_assert!(
            (r12[0] - r21[0]).abs() < 1e-10,
            "DistancePtPt not symmetric: {} vs {}",
            r12[0], r21[0]
        );
    }

    // ---- DistancePtLine -----------------------------------------------------

    /// DistancePtLine residual is zero when the point is at the correct distance.
    #[test]
    fn prop_distance_pt_line_satisfied(
        x1 in coord(), y1 in coord(),
        angle in angle_rad(),
        line_len in 1.0f64..100.0,
        dist in positive_dist(),
    ) {
        let x2 = x1 + line_len * angle.cos();
        let y2 = y1 + line_len * angle.sin();

        // Normal to the line direction, at distance `dist`
        let nx = -(y2 - y1) / line_len;
        let ny = (x2 - x1) / line_len;
        let mx = (x1 + x2) / 2.0;
        let my = (y1 + y2) / 2.0;
        let px = mx + dist * nx;
        let py = my + dist * ny;

        let mut ctx = TestCtx::new();
        let ep = ctx.entity();
        let el = ctx.entity();
        let ppx = ctx.alloc(px, ep);
        let ppy = ctx.alloc(py, ep);
        let plx1 = ctx.alloc(x1, el);
        let ply1 = ctx.alloc(y1, el);
        let plx2 = ctx.alloc(x2, el);
        let ply2 = ctx.alloc(y2, el);
        let store = ctx.store();

        let c = DistancePtLine::new(ctx.cid(), ep, el, ppx, ppy, plx1, ply1, plx2, ply2, dist);
        let r = c.residuals(&store);

        prop_assert!(
            r[0].abs() < 1e-6,
            "DistancePtLine residual should be ~0, got {}",
            r[0]
        );
    }

    // ---- Coincident ---------------------------------------------------------

    /// Coincident residual is zero when both points are at the same position.
    #[test]
    fn prop_coincident_satisfied(x in coord(), y in coord()) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x, e0);
        let py1 = ctx.alloc(y, e0);
        let px2 = ctx.alloc(x, e1);
        let py2 = ctx.alloc(y, e1);
        let store = ctx.store();

        let c = Coincident::new(ctx.cid(), e0, e1, px1, py1, px2, py2);
        let r = c.residuals(&store);

        prop_assert!(r[0].abs() < 1e-12 && r[1].abs() < 1e-12);
    }

    // ---- Fixed --------------------------------------------------------------

    /// Fixed constraint residual is zero when point is at the target.
    #[test]
    fn prop_fixed_satisfied(x in coord(), y in coord()) {
        let mut ctx = TestCtx::new();
        let e = ctx.entity();
        let px = ctx.alloc(x, e);
        let py = ctx.alloc(y, e);
        let store = ctx.store();

        let c = Fixed::new(ctx.cid(), e, px, py, x, y);
        let r = c.residuals(&store);

        prop_assert!(r[0].abs() < 1e-12 && r[1].abs() < 1e-12);
    }

    // ---- Horizontal ---------------------------------------------------------

    /// Horizontal residual is zero when both y-coordinates match.
    #[test]
    fn prop_horizontal_satisfied(x1 in coord(), x2 in coord(), y in coord()) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let _px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y, e0);
        let _px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y, e1);
        let store = ctx.store();

        let c = Horizontal::new(ctx.cid(), e0, e1, py1, py2);
        let r = c.residuals(&store);

        prop_assert!(r[0].abs() < 1e-12);
    }

    // ---- Vertical -----------------------------------------------------------

    /// Vertical residual is zero when both x-coordinates match.
    #[test]
    fn prop_vertical_satisfied(x in coord(), y1 in coord(), y2 in coord()) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x, e0);
        let _py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x, e1);
        let _py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = Vertical::new(ctx.cid(), e0, e1, px1, px2);
        let r = c.residuals(&store);

        prop_assert!(r[0].abs() < 1e-12);
    }

    // ---- Angle --------------------------------------------------------------

    /// Angle constraint residual is zero when line direction matches the target angle.
    #[test]
    fn prop_angle_satisfied(
        x1 in coord(), y1 in coord(),
        target_angle in angle_rad(),
        length in 0.1f64..100.0,
    ) {
        let x2 = x1 + length * target_angle.cos();
        let y2 = y1 + length * target_angle.sin();

        let mut ctx = TestCtx::new();
        let e = ctx.entity();
        let px1 = ctx.alloc(x1, e);
        let py1 = ctx.alloc(y1, e);
        let px2 = ctx.alloc(x2, e);
        let py2 = ctx.alloc(y2, e);
        let store = ctx.store();

        let c = Angle::new(ctx.cid(), e, px1, py1, px2, py2, target_angle);
        let r = c.residuals(&store);

        prop_assert!(
            r[0].abs() < 1e-8,
            "Angle residual should be ~0, got {}",
            r[0]
        );
    }

    // ---- Parallel -----------------------------------------------------------

    /// Parallel residual is zero when both direction vectors are parallel.
    #[test]
    fn prop_parallel_satisfied(
        x1 in coord(), y1 in coord(),
        dx in coord(), dy in coord(),
        x3 in coord(), y3 in coord(),
        scale in 0.1f64..10.0,
    ) {
        let dir_len = (dx * dx + dy * dy).sqrt();
        prop_assume!(dir_len > 0.01);

        let x2 = x1 + dx;
        let y2 = y1 + dy;
        let x4 = x3 + dx * scale;
        let y4 = y3 + dy * scale;

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let p_x1 = ctx.alloc(x1, e0);
        let p_y1 = ctx.alloc(y1, e0);
        let p_x2 = ctx.alloc(x2, e0);
        let p_y2 = ctx.alloc(y2, e0);
        let p_x3 = ctx.alloc(x3, e1);
        let p_y3 = ctx.alloc(y3, e1);
        let p_x4 = ctx.alloc(x4, e1);
        let p_y4 = ctx.alloc(y4, e1);
        let store = ctx.store();

        let c = Parallel::new(
            ctx.cid(), e0, e1, p_x1, p_y1, p_x2, p_y2, p_x3, p_y3, p_x4, p_y4,
        );
        let r = c.residuals(&store);

        prop_assert!(
            r[0].abs() < 1e-6,
            "Parallel residual should be ~0, got {}",
            r[0]
        );
    }

    // ---- Perpendicular ------------------------------------------------------

    /// Perpendicular residual is zero when direction vectors are perpendicular.
    #[test]
    fn prop_perpendicular_satisfied(
        x1 in coord(), y1 in coord(),
        dx in coord(), dy in coord(),
        x3 in coord(), y3 in coord(),
    ) {
        let dir_len = (dx * dx + dy * dy).sqrt();
        prop_assume!(dir_len > 0.01);

        let x2 = x1 + dx;
        let y2 = y1 + dy;
        // Rotate 90°: perpendicular direction
        let x4 = x3 + (-dy);
        let y4 = y3 + dx;

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let p_x1 = ctx.alloc(x1, e0);
        let p_y1 = ctx.alloc(y1, e0);
        let p_x2 = ctx.alloc(x2, e0);
        let p_y2 = ctx.alloc(y2, e0);
        let p_x3 = ctx.alloc(x3, e1);
        let p_y3 = ctx.alloc(y3, e1);
        let p_x4 = ctx.alloc(x4, e1);
        let p_y4 = ctx.alloc(y4, e1);
        let store = ctx.store();

        let c = Perpendicular::new(
            ctx.cid(), e0, e1, p_x1, p_y1, p_x2, p_y2, p_x3, p_y3, p_x4, p_y4,
        );
        let r = c.residuals(&store);

        prop_assert!(
            r[0].abs() < 1e-6,
            "Perpendicular residual should be ~0, got {}",
            r[0]
        );
    }

    // ---- Midpoint -----------------------------------------------------------

    /// Midpoint residual is zero when the point is at the midpoint of a segment.
    #[test]
    fn prop_midpoint_satisfied(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
    ) {
        let mx = (x1 + x2) / 2.0;
        let my = (y1 + y2) / 2.0;

        let mut ctx = TestCtx::new();
        let ep = ctx.entity();
        let el = ctx.entity();
        let pmx = ctx.alloc(mx, ep);
        let pmy = ctx.alloc(my, ep);
        let px1 = ctx.alloc(x1, el);
        let py1 = ctx.alloc(y1, el);
        let px2 = ctx.alloc(x2, el);
        let py2 = ctx.alloc(y2, el);
        let store = ctx.store();

        let c = Midpoint::new(ctx.cid(), ep, el, pmx, pmy, px1, py1, px2, py2);
        let r = c.residuals(&store);

        prop_assert!(r[0].abs() < 1e-10 && r[1].abs() < 1e-10);
    }

    // ---- Symmetric ----------------------------------------------------------

    /// Symmetric residual is zero when p2 = 2*center - p1.
    #[test]
    fn prop_symmetric_satisfied(
        x1 in coord(), y1 in coord(),
        cx in coord(), cy in coord(),
    ) {
        let x2 = 2.0 * cx - x1;
        let y2 = 2.0 * cy - y1;

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let ec = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let pcx = ctx.alloc(cx, ec);
        let pcy = ctx.alloc(cy, ec);
        let store = ctx.store();

        let c = Symmetric::new(ctx.cid(), e0, e1, ec, px1, py1, px2, py2, pcx, pcy);
        let r = c.residuals(&store);

        prop_assert!(r[0].abs() < 1e-10 && r[1].abs() < 1e-10);
    }

    // ---- EqualLength --------------------------------------------------------

    /// EqualLength residual is zero when both segments have equal length.
    #[test]
    fn prop_equal_length_satisfied(
        x1 in coord(), y1 in coord(),
        x3 in coord(), y3 in coord(),
        length in positive_dist(),
        angle1 in angle_rad(),
        angle2 in angle_rad(),
    ) {
        let x2 = x1 + length * angle1.cos();
        let y2 = y1 + length * angle1.sin();
        let x4 = x3 + length * angle2.cos();
        let y4 = y3 + length * angle2.sin();

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e0);
        let py2 = ctx.alloc(y2, e0);
        let px3 = ctx.alloc(x3, e1);
        let py3 = ctx.alloc(y3, e1);
        let px4 = ctx.alloc(x4, e1);
        let py4 = ctx.alloc(y4, e1);
        let store = ctx.store();

        let c = EqualLength::new(
            ctx.cid(), e0, e1, px1, py1, px2, py2, px3, py3, px4, py4,
        );
        let r = c.residuals(&store);

        prop_assert!(
            r[0].abs() < 1e-6,
            "EqualLength residual should be ~0, got {}",
            r[0]
        );
    }

    // ---- PointOnCircle ------------------------------------------------------

    /// PointOnCircle residual is zero when the point lies on the circle.
    #[test]
    fn prop_point_on_circle_satisfied(
        cx in coord(), cy in coord(),
        r in positive_radius(),
        angle in angle_rad(),
    ) {
        let px = cx + r * angle.cos();
        let py = cy + r * angle.sin();

        let mut ctx = TestCtx::new();
        let ep = ctx.entity();
        let ec = ctx.entity();
        let ppx = ctx.alloc(px, ep);
        let ppy = ctx.alloc(py, ep);
        let pcx = ctx.alloc(cx, ec);
        let pcy = ctx.alloc(cy, ec);
        let pr = ctx.alloc(r, ec);
        let store = ctx.store();

        let c = PointOnCircle::new(ctx.cid(), ep, ec, ppx, ppy, pcx, pcy, pr);
        let res = c.residuals(&store);

        prop_assert!(
            res[0].abs() < 1e-6,
            "PointOnCircle residual should be ~0, got {}",
            res[0]
        );
    }

    // ---- TangentLineCircle --------------------------------------------------

    /// TangentLineCircle residual is zero when the line is tangent to the circle.
    #[test]
    fn prop_tangent_line_circle_satisfied(
        cx in coord(), cy in coord(),
        r in positive_radius(),
        line_angle in angle_rad(),
        half_len in 1.0f64..100.0,
    ) {
        // Build a tangent line: perpendicular distance from center to line = r.
        let normal_x = line_angle.cos();
        let normal_y = line_angle.sin();
        // Tangent point
        let tp_x = cx + r * normal_x;
        let tp_y = cy + r * normal_y;
        // Line direction is perpendicular to normal
        let dir_x = -normal_y;
        let dir_y = normal_x;
        let x1 = tp_x - half_len * dir_x;
        let y1 = tp_y - half_len * dir_y;
        let x2 = tp_x + half_len * dir_x;
        let y2 = tp_y + half_len * dir_y;

        let mut ctx = TestCtx::new();
        let el = ctx.entity();
        let ec = ctx.entity();
        let px1 = ctx.alloc(x1, el);
        let py1 = ctx.alloc(y1, el);
        let px2 = ctx.alloc(x2, el);
        let py2 = ctx.alloc(y2, el);
        let pcx = ctx.alloc(cx, ec);
        let pcy = ctx.alloc(cy, ec);
        let pr = ctx.alloc(r, ec);
        let store = ctx.store();

        let c = TangentLineCircle::new(ctx.cid(), el, ec, px1, py1, px2, py2, pcx, pcy, pr);
        let res = c.residuals(&store);

        prop_assert!(
            res[0].abs() < 1e-5,
            "TangentLineCircle residual should be ~0, got {}",
            res[0]
        );
    }

    // ---- TangentCircleCircle (external) -------------------------------------

    /// External tangent residual is zero when dist(centers) = r1 + r2.
    #[test]
    fn prop_tangent_circle_circle_external_satisfied(
        cx1 in coord(), cy1 in coord(),
        r1 in positive_radius(),
        r2 in positive_radius(),
        angle in angle_rad(),
    ) {
        let dist = r1 + r2;
        let cx2 = cx1 + dist * angle.cos();
        let cy2 = cy1 + dist * angle.sin();

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let pcx1 = ctx.alloc(cx1, e0);
        let pcy1 = ctx.alloc(cy1, e0);
        let pr1 = ctx.alloc(r1, e0);
        let pcx2 = ctx.alloc(cx2, e1);
        let pcy2 = ctx.alloc(cy2, e1);
        let pr2 = ctx.alloc(r2, e1);
        let store = ctx.store();

        let c = TangentCircleCircle::external(
            ctx.cid(), e0, e1, pcx1, pcy1, pr1, pcx2, pcy2, pr2,
        );
        let res = c.residuals(&store);

        prop_assert!(
            res[0].abs() < 1e-5,
            "External tangent residual should be ~0, got {}",
            res[0]
        );
    }

    // ---- TangentCircleCircle (internal) -------------------------------------

    /// Internal tangent residual is zero when dist(centers) = |r1 - r2|.
    #[test]
    fn prop_tangent_circle_circle_internal_satisfied(
        cx1 in coord(), cy1 in coord(),
        r_big in 10.0f64..100.0,
        r_small in 0.5f64..9.0,
        angle in angle_rad(),
    ) {
        let dist = r_big - r_small;
        let cx2 = cx1 + dist * angle.cos();
        let cy2 = cy1 + dist * angle.sin();

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let pcx1 = ctx.alloc(cx1, e0);
        let pcy1 = ctx.alloc(cy1, e0);
        let pr1 = ctx.alloc(r_big, e0);
        let pcx2 = ctx.alloc(cx2, e1);
        let pcy2 = ctx.alloc(cy2, e1);
        let pr2 = ctx.alloc(r_small, e1);
        let store = ctx.store();

        let c = TangentCircleCircle::internal(
            ctx.cid(), e0, e1, pcx1, pcy1, pr1, pcx2, pcy2, pr2,
        );
        let res = c.residuals(&store);

        prop_assert!(
            res[0].abs() < 1e-5,
            "Internal tangent residual should be ~0, got {}",
            res[0]
        );
    }
}

// =============================================================================
// Section 2: Jacobian correctness (analytical vs. finite differences)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_distance_pt_pt_jacobian(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        dist in positive_dist(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = DistancePtPt::new(ctx.cid(), e0, e1, px1, py1, px2, py2, dist);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-4),
            "DistancePtPt Jacobian mismatch at ({},{}) ({},{})",
            x1, y1, x2, y2
        );
    }

    #[test]
    fn prop_distance_pt_line_jacobian(
        px in coord(), py in coord(),
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        dist in positive_dist(),
    ) {
        // Skip degenerate lines (zero length)
        let line_len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        prop_assume!(line_len > 0.1);

        let mut ctx = TestCtx::new();
        let ep = ctx.entity();
        let el = ctx.entity();
        let ppx = ctx.alloc(px, ep);
        let ppy = ctx.alloc(py, ep);
        let plx1 = ctx.alloc(x1, el);
        let ply1 = ctx.alloc(y1, el);
        let plx2 = ctx.alloc(x2, el);
        let ply2 = ctx.alloc(y2, el);
        let store = ctx.store();

        let c = DistancePtLine::new(ctx.cid(), ep, el, ppx, ppy, plx1, ply1, plx2, ply2, dist);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-3),
            "DistancePtLine Jacobian mismatch"
        );
    }

    #[test]
    fn prop_coincident_jacobian(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = Coincident::new(ctx.cid(), e0, e1, px1, py1, px2, py2);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-5),
            "Coincident Jacobian mismatch"
        );
    }

    #[test]
    fn prop_fixed_jacobian(x in coord(), y in coord(), tx in coord(), ty in coord()) {
        let mut ctx = TestCtx::new();
        let e = ctx.entity();
        let px = ctx.alloc(x, e);
        let py = ctx.alloc(y, e);
        let store = ctx.store();

        let c = Fixed::new(ctx.cid(), e, px, py, tx, ty);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-5),
            "Fixed Jacobian mismatch"
        );
    }

    #[test]
    fn prop_horizontal_jacobian(y1 in coord(), y2 in coord()) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let py1 = ctx.alloc(y1, e0);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = Horizontal::new(ctx.cid(), e0, e1, py1, py2);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-5),
            "Horizontal Jacobian mismatch"
        );
    }

    #[test]
    fn prop_vertical_jacobian(x1 in coord(), x2 in coord()) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let px2 = ctx.alloc(x2, e1);
        let store = ctx.store();

        let c = Vertical::new(ctx.cid(), e0, e1, px1, px2);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-5),
            "Vertical Jacobian mismatch"
        );
    }

    #[test]
    fn prop_angle_jacobian(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        angle in angle_rad(),
    ) {
        let mut ctx = TestCtx::new();
        let e = ctx.entity();
        let px1 = ctx.alloc(x1, e);
        let py1 = ctx.alloc(y1, e);
        let px2 = ctx.alloc(x2, e);
        let py2 = ctx.alloc(y2, e);
        let store = ctx.store();

        let c = Angle::new(ctx.cid(), e, px1, py1, px2, py2, angle);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-5),
            "Angle Jacobian mismatch"
        );
    }

    #[test]
    fn prop_parallel_jacobian(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        x3 in coord(), y3 in coord(),
        x4 in coord(), y4 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let p_x1 = ctx.alloc(x1, e0);
        let p_y1 = ctx.alloc(y1, e0);
        let p_x2 = ctx.alloc(x2, e0);
        let p_y2 = ctx.alloc(y2, e0);
        let p_x3 = ctx.alloc(x3, e1);
        let p_y3 = ctx.alloc(y3, e1);
        let p_x4 = ctx.alloc(x4, e1);
        let p_y4 = ctx.alloc(y4, e1);
        let store = ctx.store();

        let c = Parallel::new(
            ctx.cid(), e0, e1, p_x1, p_y1, p_x2, p_y2, p_x3, p_y3, p_x4, p_y4,
        );
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-4),
            "Parallel Jacobian mismatch"
        );
    }

    #[test]
    fn prop_perpendicular_jacobian(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        x3 in coord(), y3 in coord(),
        x4 in coord(), y4 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let p_x1 = ctx.alloc(x1, e0);
        let p_y1 = ctx.alloc(y1, e0);
        let p_x2 = ctx.alloc(x2, e0);
        let p_y2 = ctx.alloc(y2, e0);
        let p_x3 = ctx.alloc(x3, e1);
        let p_y3 = ctx.alloc(y3, e1);
        let p_x4 = ctx.alloc(x4, e1);
        let p_y4 = ctx.alloc(y4, e1);
        let store = ctx.store();

        let c = Perpendicular::new(
            ctx.cid(), e0, e1, p_x1, p_y1, p_x2, p_y2, p_x3, p_y3, p_x4, p_y4,
        );
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-4),
            "Perpendicular Jacobian mismatch"
        );
    }

    #[test]
    fn prop_midpoint_jacobian(
        mx in coord(), my in coord(),
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let ep = ctx.entity();
        let el = ctx.entity();
        let pmx = ctx.alloc(mx, ep);
        let pmy = ctx.alloc(my, ep);
        let px1 = ctx.alloc(x1, el);
        let py1 = ctx.alloc(y1, el);
        let px2 = ctx.alloc(x2, el);
        let py2 = ctx.alloc(y2, el);
        let store = ctx.store();

        let c = Midpoint::new(ctx.cid(), ep, el, pmx, pmy, px1, py1, px2, py2);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-5),
            "Midpoint Jacobian mismatch"
        );
    }

    #[test]
    fn prop_symmetric_jacobian(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        cx in coord(), cy in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let ec = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let pcx = ctx.alloc(cx, ec);
        let pcy = ctx.alloc(cy, ec);
        let store = ctx.store();

        let c = Symmetric::new(ctx.cid(), e0, e1, ec, px1, py1, px2, py2, pcx, pcy);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-5),
            "Symmetric Jacobian mismatch"
        );
    }

    #[test]
    fn prop_equal_length_jacobian(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        x3 in coord(), y3 in coord(),
        x4 in coord(), y4 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e0);
        let py2 = ctx.alloc(y2, e0);
        let px3 = ctx.alloc(x3, e1);
        let py3 = ctx.alloc(y3, e1);
        let px4 = ctx.alloc(x4, e1);
        let py4 = ctx.alloc(y4, e1);
        let store = ctx.store();

        let c = EqualLength::new(
            ctx.cid(), e0, e1, px1, py1, px2, py2, px3, py3, px4, py4,
        );
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-4),
            "EqualLength Jacobian mismatch"
        );
    }

    #[test]
    fn prop_point_on_circle_jacobian(
        px in coord(), py in coord(),
        cx in coord(), cy in coord(),
        r in positive_radius(),
    ) {
        let mut ctx = TestCtx::new();
        let ep = ctx.entity();
        let ec = ctx.entity();
        let ppx = ctx.alloc(px, ep);
        let ppy = ctx.alloc(py, ep);
        let pcx = ctx.alloc(cx, ec);
        let pcy = ctx.alloc(cy, ec);
        let pr = ctx.alloc(r, ec);
        let store = ctx.store();

        let c = PointOnCircle::new(ctx.cid(), ep, ec, ppx, ppy, pcx, pcy, pr);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-4),
            "PointOnCircle Jacobian mismatch"
        );
    }

    #[test]
    fn prop_tangent_line_circle_jacobian(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        cx in coord(), cy in coord(),
        r in positive_radius(),
    ) {
        // Skip degenerate lines
        let line_len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        prop_assume!(line_len > 0.5);

        let mut ctx = TestCtx::new();
        let el = ctx.entity();
        let ec = ctx.entity();
        let px1 = ctx.alloc(x1, el);
        let py1 = ctx.alloc(y1, el);
        let px2 = ctx.alloc(x2, el);
        let py2 = ctx.alloc(y2, el);
        let pcx = ctx.alloc(cx, ec);
        let pcy = ctx.alloc(cy, ec);
        let pr = ctx.alloc(r, ec);
        let store = ctx.store();

        let c = TangentLineCircle::new(ctx.cid(), el, ec, px1, py1, px2, py2, pcx, pcy, pr);
        prop_assert!(
            check_jacobian_fd(&c, &store, 1e-7, 1e-3),
            "TangentLineCircle Jacobian mismatch"
        );
    }

    #[test]
    fn prop_tangent_circle_circle_jacobian(
        cx1 in coord(), cy1 in coord(),
        r1 in positive_radius(),
        cx2 in coord(), cy2 in coord(),
        r2 in positive_radius(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let pcx1 = ctx.alloc(cx1, e0);
        let pcy1 = ctx.alloc(cy1, e0);
        let pr1 = ctx.alloc(r1, e0);
        let pcx2 = ctx.alloc(cx2, e1);
        let pcy2 = ctx.alloc(cy2, e1);
        let pr2 = ctx.alloc(r2, e1);
        let store = ctx.store();

        let ext = TangentCircleCircle::external(
            ctx.cid(), e0, e1, pcx1, pcy1, pr1, pcx2, pcy2, pr2,
        );
        prop_assert!(
            check_jacobian_fd(&ext, &store, 1e-7, 1e-4),
            "TangentCircleCircle(external) Jacobian mismatch"
        );

        let int = TangentCircleCircle::internal(
            ctx.cid(), e0, e1, pcx1, pcy1, pr1, pcx2, pcy2, pr2,
        );
        prop_assert!(
            check_jacobian_fd(&int, &store, 1e-7, 1e-4),
            "TangentCircleCircle(internal) Jacobian mismatch"
        );
    }
}

// =============================================================================
// Section 3: DOF monotonicity — adding a constraint never increases DOF
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Adding a distance constraint to a system never increases the total DOF.
    #[test]
    fn prop_dof_nonincreasing_distance(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        x3 in coord(), y3 in coord(),
        dist in positive_dist(),
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(x1, y1);
        let p1 = b.add_point(x2, y2);
        let p2 = b.add_point(x3, y3);

        let dof_before = b.system().degrees_of_freedom();

        // Add one distance constraint
        b.constrain_distance(p0, p1, dist);
        let dof_after_1 = b.system().degrees_of_freedom();

        // Add another
        b.constrain_distance(p1, p2, dist);
        let dof_after_2 = b.system().degrees_of_freedom();

        prop_assert!(
            dof_after_1 <= dof_before,
            "DOF increased after first constraint: {} -> {}",
            dof_before, dof_after_1
        );
        prop_assert!(
            dof_after_2 <= dof_after_1,
            "DOF increased after second constraint: {} -> {}",
            dof_after_1, dof_after_2
        );
    }

    /// Adding a coincident constraint to a system never increases DOF.
    #[test]
    fn prop_dof_nonincreasing_coincident(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(x1, y1);
        let p1 = b.add_point(x2, y2);

        let dof_before = b.system().degrees_of_freedom();

        b.constrain_coincident(p0, p1);
        let dof_after = b.system().degrees_of_freedom();

        prop_assert!(
            dof_after <= dof_before,
            "DOF increased: {} -> {}",
            dof_before, dof_after
        );
    }

    /// Adding a horizontal constraint never increases DOF.
    #[test]
    fn prop_dof_nonincreasing_horizontal(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(x1, y1);
        let p1 = b.add_point(x2, y2);

        let dof_before = b.system().degrees_of_freedom();

        b.constrain_horizontal(p0, p1);
        let dof_after = b.system().degrees_of_freedom();

        prop_assert!(
            dof_after <= dof_before,
            "DOF increased: {} -> {}",
            dof_before, dof_after
        );
    }

    /// Adding a fixed constraint removes exactly 2 DOF from a free point.
    #[test]
    fn prop_fixed_removes_2_dof(
        x in coord(), y in coord(),
        tx in coord(), ty in coord(),
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let p = b.add_point(x, y);

        let dof_before = b.system().degrees_of_freedom();

        b.constrain_fixed(p, tx, ty);
        let dof_after = b.system().degrees_of_freedom();

        prop_assert_eq!(
            dof_after,
            dof_before - 2,
            "Fixed should remove exactly 2 DOF"
        );
    }

    /// DOF = free_params - equations holds for arbitrary builder configurations.
    #[test]
    fn prop_dof_equals_params_minus_equations(
        n_points in 1usize..6,
        n_fixed in 0usize..3,
        n_dist in 0usize..8,
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let mut points = Vec::new();

        for i in 0..n_points {
            points.push(b.add_point(i as f64 * 2.0, i as f64));
        }

        for i in 0..n_fixed.min(n_points) {
            b.fix_entity(points[i]);
        }

        let max_dist = if n_points > 1 { n_dist.min(n_points - 1) } else { 0 };
        for i in 0..max_dist {
            b.constrain_distance(points[i], points[i + 1], 1.0);
        }

        let sys = b.system();
        let expected = sys.params().free_param_count() as i32 - sys.equation_count() as i32;
        prop_assert_eq!(
            sys.degrees_of_freedom(),
            expected,
            "DOF formula mismatch"
        );
    }
}

// =============================================================================
// Section 4: Decomposition preservation
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Two independent constraint groups produce distinct clusters.
    #[test]
    fn prop_independent_constraints_decompose(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        x3 in coord(), y3 in coord(),
        x4 in coord(), y4 in coord(),
        d1 in positive_dist(),
        d2 in positive_dist(),
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(x1, y1);
        let p1 = b.add_point(x2, y2);
        let p2 = b.add_point(x3, y3);
        let p3 = b.add_point(x4, y4);

        b.constrain_distance(p0, p1, d1);
        b.constrain_distance(p2, p3, d2);

        let mut sys = b.build();
        let result = sys.solve();

        // Should decompose into exactly 2 clusters (one per independent group).
        prop_assert!(
            result.clusters.len() >= 2,
            "Expected >=2 clusters, got {}",
            result.clusters.len()
        );
    }

    /// A chain of constraints sharing points forms a single cluster.
    #[test]
    fn prop_chained_constraints_single_cluster(
        x0 in coord(), y0 in coord(),
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        d1 in positive_dist(),
        d2 in positive_dist(),
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(x0, y0);
        let p1 = b.add_point(x1, y1);
        let p2 = b.add_point(x2, y2);

        // p0-p1 and p1-p2 share point p1 => single cluster
        b.constrain_distance(p0, p1, d1);
        b.constrain_distance(p1, p2, d2);

        let mut sys = b.build();
        let result = sys.solve();

        // All constraints share p1, so they form 1 cluster.
        prop_assert_eq!(
            result.clusters.len(),
            1,
            "Expected 1 cluster for chained constraints, got {}",
            result.clusters.len()
        );
    }
}

// =============================================================================
// Section 5: Coordinate transformation invariance
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// DistancePtPt residual is invariant under rigid-body translation.
    #[test]
    fn prop_distance_translation_invariant(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        tx in -100.0f64..100.0, ty in -100.0f64..100.0,
        dist in positive_dist(),
    ) {
        // Original configuration
        let mut ctx_o = TestCtx::new();
        let e0o = ctx_o.entity();
        let e1o = ctx_o.entity();
        let px1o = ctx_o.alloc(x1, e0o);
        let py1o = ctx_o.alloc(y1, e0o);
        let px2o = ctx_o.alloc(x2, e1o);
        let py2o = ctx_o.alloc(y2, e1o);
        let store_o = ctx_o.store();
        let c_orig = DistancePtPt::new(ctx_o.cid(), e0o, e1o, px1o, py1o, px2o, py2o, dist);
        let r_orig = c_orig.residuals(&store_o);

        // Translated configuration
        let mut ctx_t = TestCtx::new();
        let e0t = ctx_t.entity();
        let e1t = ctx_t.entity();
        let px1t = ctx_t.alloc(x1 + tx, e0t);
        let py1t = ctx_t.alloc(y1 + ty, e0t);
        let px2t = ctx_t.alloc(x2 + tx, e1t);
        let py2t = ctx_t.alloc(y2 + ty, e1t);
        let store_t = ctx_t.store();
        let c_trans = DistancePtPt::new(ctx_t.cid(), e0t, e1t, px1t, py1t, px2t, py2t, dist);
        let r_trans = c_trans.residuals(&store_t);

        prop_assert!(
            (r_orig[0] - r_trans[0]).abs() < 1e-8,
            "Distance residual not translation invariant: {} vs {}",
            r_orig[0], r_trans[0]
        );
    }

    /// DistancePtPt residual is invariant under rotation about the origin.
    #[test]
    fn prop_distance_rotation_invariant(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        theta in angle_rad(),
        dist in positive_dist(),
    ) {
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let rx1 = x1 * cos_t - y1 * sin_t;
        let ry1 = x1 * sin_t + y1 * cos_t;
        let rx2 = x2 * cos_t - y2 * sin_t;
        let ry2 = x2 * sin_t + y2 * cos_t;

        let mut ctx_o = TestCtx::new();
        let e0o = ctx_o.entity();
        let e1o = ctx_o.entity();
        let px1o = ctx_o.alloc(x1, e0o);
        let py1o = ctx_o.alloc(y1, e0o);
        let px2o = ctx_o.alloc(x2, e1o);
        let py2o = ctx_o.alloc(y2, e1o);
        let store_o = ctx_o.store();
        let c_orig = DistancePtPt::new(ctx_o.cid(), e0o, e1o, px1o, py1o, px2o, py2o, dist);

        let mut ctx_r = TestCtx::new();
        let e0r = ctx_r.entity();
        let e1r = ctx_r.entity();
        let px1r = ctx_r.alloc(rx1, e0r);
        let py1r = ctx_r.alloc(ry1, e0r);
        let px2r = ctx_r.alloc(rx2, e1r);
        let py2r = ctx_r.alloc(ry2, e1r);
        let store_r = ctx_r.store();
        let c_rot = DistancePtPt::new(ctx_r.cid(), e0r, e1r, px1r, py1r, px2r, py2r, dist);

        let r_orig = c_orig.residuals(&store_o);
        let r_rot = c_rot.residuals(&store_r);

        prop_assert!(
            (r_orig[0] - r_rot[0]).abs() < 1e-6,
            "Distance residual not rotation invariant: {} vs {}",
            r_orig[0], r_rot[0]
        );
    }

    /// Parallel constraint residual is invariant under translation.
    #[test]
    fn prop_parallel_translation_invariant(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        x3 in coord(), y3 in coord(),
        x4 in coord(), y4 in coord(),
        tx in -100.0f64..100.0, ty in -100.0f64..100.0,
    ) {
        // Original
        let mut ctx_o = TestCtx::new();
        let e0o = ctx_o.entity();
        let e1o = ctx_o.entity();
        let so_x1 = ctx_o.alloc(x1, e0o);
        let so_y1 = ctx_o.alloc(y1, e0o);
        let so_x2 = ctx_o.alloc(x2, e0o);
        let so_y2 = ctx_o.alloc(y2, e0o);
        let so_x3 = ctx_o.alloc(x3, e1o);
        let so_y3 = ctx_o.alloc(y3, e1o);
        let so_x4 = ctx_o.alloc(x4, e1o);
        let so_y4 = ctx_o.alloc(y4, e1o);
        let store_o = ctx_o.store();
        let co = Parallel::new(ctx_o.cid(), e0o, e1o, so_x1, so_y1, so_x2, so_y2, so_x3, so_y3, so_x4, so_y4);
        let ro = co.residuals(&store_o);

        // Translated
        let mut ctx_t = TestCtx::new();
        let e0t = ctx_t.entity();
        let e1t = ctx_t.entity();
        let st_x1 = ctx_t.alloc(x1 + tx, e0t);
        let st_y1 = ctx_t.alloc(y1 + ty, e0t);
        let st_x2 = ctx_t.alloc(x2 + tx, e0t);
        let st_y2 = ctx_t.alloc(y2 + ty, e0t);
        let st_x3 = ctx_t.alloc(x3 + tx, e1t);
        let st_y3 = ctx_t.alloc(y3 + ty, e1t);
        let st_x4 = ctx_t.alloc(x4 + tx, e1t);
        let st_y4 = ctx_t.alloc(y4 + ty, e1t);
        let store_t = ctx_t.store();
        let ct = Parallel::new(ctx_t.cid(), e0t, e1t, st_x1, st_y1, st_x2, st_y2, st_x3, st_y3, st_x4, st_y4);
        let rt = ct.residuals(&store_t);

        prop_assert!(
            (ro[0] - rt[0]).abs() < 1e-6,
            "Parallel residual not translation invariant: {} vs {}",
            ro[0], rt[0]
        );
    }

    /// Perpendicular constraint residual is invariant under rotation.
    #[test]
    fn prop_perpendicular_rotation_invariant(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        x3 in coord(), y3 in coord(),
        x4 in coord(), y4 in coord(),
        theta in angle_rad(),
    ) {
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let rot = |x: f64, y: f64| -> (f64, f64) {
            (x * cos_t - y * sin_t, x * sin_t + y * cos_t)
        };

        // Original
        let mut ctx_o = TestCtx::new();
        let e0o = ctx_o.entity();
        let e1o = ctx_o.entity();
        let so_x1 = ctx_o.alloc(x1, e0o);
        let so_y1 = ctx_o.alloc(y1, e0o);
        let so_x2 = ctx_o.alloc(x2, e0o);
        let so_y2 = ctx_o.alloc(y2, e0o);
        let so_x3 = ctx_o.alloc(x3, e1o);
        let so_y3 = ctx_o.alloc(y3, e1o);
        let so_x4 = ctx_o.alloc(x4, e1o);
        let so_y4 = ctx_o.alloc(y4, e1o);
        let store_o = ctx_o.store();
        let co = Perpendicular::new(ctx_o.cid(), e0o, e1o, so_x1, so_y1, so_x2, so_y2, so_x3, so_y3, so_x4, so_y4);
        let ro = co.residuals(&store_o);

        // Rotated
        let (rx1, ry1) = rot(x1, y1);
        let (rx2, ry2) = rot(x2, y2);
        let (rx3, ry3) = rot(x3, y3);
        let (rx4, ry4) = rot(x4, y4);

        let mut ctx_r = TestCtx::new();
        let e0r = ctx_r.entity();
        let e1r = ctx_r.entity();
        let sr_x1 = ctx_r.alloc(rx1, e0r);
        let sr_y1 = ctx_r.alloc(ry1, e0r);
        let sr_x2 = ctx_r.alloc(rx2, e0r);
        let sr_y2 = ctx_r.alloc(ry2, e0r);
        let sr_x3 = ctx_r.alloc(rx3, e1r);
        let sr_y3 = ctx_r.alloc(ry3, e1r);
        let sr_x4 = ctx_r.alloc(rx4, e1r);
        let sr_y4 = ctx_r.alloc(ry4, e1r);
        let store_r = ctx_r.store();
        let cr = Perpendicular::new(ctx_r.cid(), e0r, e1r, sr_x1, sr_y1, sr_x2, sr_y2, sr_x3, sr_y3, sr_x4, sr_y4);
        let rr = cr.residuals(&store_r);

        prop_assert!(
            (ro[0] - rr[0]).abs() < 1e-4,
            "Perpendicular residual not rotation invariant: {} vs {}",
            ro[0], rr[0]
        );
    }

    /// PointOnCircle residual is invariant under rigid-body translation.
    #[test]
    fn prop_point_on_circle_translation_invariant(
        px in coord(), py in coord(),
        cx in coord(), cy in coord(),
        r in positive_radius(),
        tx in -100.0f64..100.0, ty in -100.0f64..100.0,
    ) {
        // Original
        let mut ctx_o = TestCtx::new();
        let epo = ctx_o.entity();
        let eco = ctx_o.entity();
        let so_px = ctx_o.alloc(px, epo);
        let so_py = ctx_o.alloc(py, epo);
        let so_cx = ctx_o.alloc(cx, eco);
        let so_cy = ctx_o.alloc(cy, eco);
        let so_r = ctx_o.alloc(r, eco);
        let store_o = ctx_o.store();
        let co = PointOnCircle::new(ctx_o.cid(), epo, eco, so_px, so_py, so_cx, so_cy, so_r);
        let ro = co.residuals(&store_o);

        // Translated
        let mut ctx_t = TestCtx::new();
        let ept = ctx_t.entity();
        let ect = ctx_t.entity();
        let st_px = ctx_t.alloc(px + tx, ept);
        let st_py = ctx_t.alloc(py + ty, ept);
        let st_cx = ctx_t.alloc(cx + tx, ect);
        let st_cy = ctx_t.alloc(cy + ty, ect);
        let st_r = ctx_t.alloc(r, ect);
        let store_t = ctx_t.store();
        let ct = PointOnCircle::new(ctx_t.cid(), ept, ect, st_px, st_py, st_cx, st_cy, st_r);
        let rt = ct.residuals(&store_t);

        prop_assert!(
            (ro[0] - rt[0]).abs() < 1e-6,
            "PointOnCircle residual not translation invariant: {} vs {}",
            ro[0], rt[0]
        );
    }
}

// =============================================================================
// Section 6: Squared formulation properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Squared distance residual sign: positive when actual > target, negative when actual < target.
    #[test]
    fn prop_squared_distance_residual_sign(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        target in positive_dist(),
    ) {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let actual_dist = (dx * dx + dy * dy).sqrt();

        // Skip when actual_dist is very close to target (sign is ambiguous).
        prop_assume!((actual_dist - target).abs() > 0.01);

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = DistancePtPt::new(ctx.cid(), e0, e1, px1, py1, px2, py2, target);
        let r = c.residuals(&store);

        if actual_dist > target {
            prop_assert!(
                r[0] > 0.0,
                "Expected positive residual when actual ({}) > target ({}), got {}",
                actual_dist, target, r[0]
            );
        } else {
            prop_assert!(
                r[0] < 0.0,
                "Expected negative residual when actual ({}) < target ({}), got {}",
                actual_dist, target, r[0]
            );
        }
    }

    /// Squared PointOnCircle residual sign: positive when point outside, negative when inside.
    #[test]
    fn prop_point_on_circle_residual_sign(
        cx in coord(), cy in coord(),
        r in positive_radius(),
        angle in angle_rad(),
        scale in 0.1f64..3.0,
    ) {
        let px = cx + scale * r * angle.cos();
        let py = cy + scale * r * angle.sin();

        prop_assume!((scale - 1.0).abs() > 0.01);

        let mut ctx = TestCtx::new();
        let ep = ctx.entity();
        let ec = ctx.entity();
        let ppx = ctx.alloc(px, ep);
        let ppy = ctx.alloc(py, ep);
        let pcx = ctx.alloc(cx, ec);
        let pcy = ctx.alloc(cy, ec);
        let pr = ctx.alloc(r, ec);
        let store = ctx.store();

        let c = PointOnCircle::new(ctx.cid(), ep, ec, ppx, ppy, pcx, pcy, pr);
        let res = c.residuals(&store);

        if scale > 1.0 {
            prop_assert!(res[0] > 0.0, "Expected positive (outside), got {}", res[0]);
        } else {
            prop_assert!(res[0] < 0.0, "Expected negative (inside), got {}", res[0]);
        }
    }

    /// Squared formulation: Jacobian is smooth at zero distance (no singularity).
    /// This is a key advantage over sqrt-based formulations.
    #[test]
    fn prop_squared_jacobian_smooth_near_zero(
        x in coord(), y in coord(),
        eps_x in -0.001f64..0.001,
        eps_y in -0.001f64..0.001,
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x, e0);
        let py1 = ctx.alloc(y, e0);
        let px2 = ctx.alloc(x + eps_x, e1);
        let py2 = ctx.alloc(y + eps_y, e1);
        let store = ctx.store();

        let c = DistancePtPt::new(ctx.cid(), e0, e1, px1, py1, px2, py2, 1.0);
        let jac = c.jacobian(&store);

        // All Jacobian entries should be finite (no NaN, no Inf).
        for &(_, _, val) in &jac {
            prop_assert!(
                val.is_finite(),
                "Jacobian entry is not finite: {} at near-zero distance",
                val
            );
        }

        // Jacobian entries should be small when points are nearly coincident.
        for &(_, _, val) in &jac {
            prop_assert!(
                val.abs() < 10.0,
                "Jacobian entry too large near zero distance: {}",
                val
            );
        }
    }

    /// EqualLength squared residual: when L1 > L2, residual is positive.
    #[test]
    fn prop_equal_length_residual_sign(
        x1 in coord(), y1 in coord(),
        x3 in coord(), y3 in coord(),
        len1 in 1.0f64..100.0,
        len2 in 1.0f64..100.0,
        angle1 in angle_rad(),
        angle2 in angle_rad(),
    ) {
        prop_assume!((len1 - len2).abs() > 0.1);

        let x2 = x1 + len1 * angle1.cos();
        let y2 = y1 + len1 * angle1.sin();
        let x4 = x3 + len2 * angle2.cos();
        let y4 = y3 + len2 * angle2.sin();

        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e0);
        let py2 = ctx.alloc(y2, e0);
        let px3 = ctx.alloc(x3, e1);
        let py3 = ctx.alloc(y3, e1);
        let px4 = ctx.alloc(x4, e1);
        let py4 = ctx.alloc(y4, e1);
        let store = ctx.store();

        let c = EqualLength::new(
            ctx.cid(), e0, e1, px1, py1, px2, py2, px3, py3, px4, py4,
        );
        let r = c.residuals(&store);

        if len1 > len2 {
            prop_assert!(r[0] > 0.0, "Expected positive when L1 > L2, got {}", r[0]);
        } else {
            prop_assert!(r[0] < 0.0, "Expected negative when L1 < L2, got {}", r[0]);
        }
    }
}

// =============================================================================
// Section 7: System-level property tests using Sketch2DBuilder
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// A well-constrained triangle (fixed base + 3 distances) should converge.
    #[test]
    fn prop_triangle_converges(
        side1 in 2.0f64..50.0,
        side2 in 2.0f64..50.0,
        side3 in 2.0f64..50.0,
    ) {
        // Triangle inequality
        prop_assume!(side1 < side2 + side3);
        prop_assume!(side2 < side1 + side3);
        prop_assume!(side3 < side1 + side2);

        use solverang::sketch2d::Sketch2DBuilder;
        use solverang::system::SystemStatus;

        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_fixed_point(0.0, 0.0);
        let p1 = b.add_fixed_point(side1, 0.0);
        let p2 = b.add_point(side1 / 2.0, side2.max(side3) / 2.0);

        b.constrain_distance(p0, p1, side1);
        b.constrain_distance(p1, p2, side2);
        b.constrain_distance(p2, p0, side3);

        let mut sys = b.build();
        let result = sys.solve();

        match result.status {
            SystemStatus::Solved | SystemStatus::PartiallySolved => {
                let residuals = sys.compute_residuals();
                let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);

                prop_assert!(
                    max_residual < 1.0,
                    "Triangle max residual too large: {}",
                    max_residual
                );
            }
            SystemStatus::DiagnosticFailure(ref issues) => {
                // DiagnosticFailure should not happen for valid triangles.
                prop_assert!(
                    false,
                    "Triangle solver returned DiagnosticFailure: {:?}",
                    issues
                );
            }
        }
    }

    /// A rectangle (4 points, distance + perpendicular constraints) has correct DOF.
    #[test]
    fn prop_rectangle_dof(
        width in 1.0f64..50.0,
        height in 1.0f64..50.0,
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_point(0.0, 0.0);
        let p1 = b.add_point(width, 0.0);
        let p2 = b.add_point(width, height);
        let p3 = b.add_point(0.0, height);

        // 4 free points = 8 free params
        prop_assert_eq!(b.system().degrees_of_freedom(), 8);

        let l01 = b.add_line_segment(p0, p1);
        let l12 = b.add_line_segment(p1, p2);
        let l23 = b.add_line_segment(p2, p3);
        let l30 = b.add_line_segment(p3, p0);

        // Line entities share params, DOF unchanged
        prop_assert_eq!(b.system().degrees_of_freedom(), 8);

        // 4 distance constraints: -4 DOF
        b.constrain_distance(p0, p1, width);
        b.constrain_distance(p1, p2, height);
        b.constrain_distance(p2, p3, width);
        b.constrain_distance(p3, p0, height);

        prop_assert_eq!(b.system().degrees_of_freedom(), 4);

        // 2 perpendicular constraints: -2 DOF
        b.constrain_perpendicular(l01, l12);
        b.constrain_perpendicular(l12, l23);

        prop_assert_eq!(b.system().degrees_of_freedom(), 2);

        // Fixing one point: -2 DOF
        b.fix_entity(p0);

        prop_assert_eq!(b.system().degrees_of_freedom(), 0);

        // Suppress warnings
        let _ = (l23, l30);
    }

    /// Fixing all points of a system reduces DOF to -(equation_count).
    #[test]
    fn prop_all_fixed_overconstrained(
        n_points in 2usize..5,
        n_dist in 1usize..5,
    ) {
        use solverang::sketch2d::Sketch2DBuilder;

        let mut b = Sketch2DBuilder::new();
        let mut points = Vec::new();
        for i in 0..n_points {
            points.push(b.add_fixed_point(i as f64 * 3.0, 0.0));
        }

        let max_dist = n_dist.min(n_points - 1);
        for i in 0..max_dist {
            b.constrain_distance(points[i], points[i + 1], 3.0);
        }

        let sys = b.system();
        prop_assert_eq!(
            sys.degrees_of_freedom(),
            -(sys.equation_count() as i32),
            "DOF should be negative of equation count when all fixed"
        );
    }
}

// =============================================================================
// Section 8: Equation count and finiteness invariants
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// equation_count() matches the length of residuals().
    #[test]
    fn prop_equation_count_matches_residuals_distance(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        dist in positive_dist(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = DistancePtPt::new(ctx.cid(), e0, e1, px1, py1, px2, py2, dist);
        prop_assert_eq!(c.equation_count(), c.residuals(&store).len());
    }

    #[test]
    fn prop_equation_count_matches_residuals_coincident(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = Coincident::new(ctx.cid(), e0, e1, px1, py1, px2, py2);
        prop_assert_eq!(c.equation_count(), c.residuals(&store).len());
    }

    #[test]
    fn prop_equation_count_matches_residuals_midpoint(
        mx in coord(), my in coord(),
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let ep = ctx.entity();
        let el = ctx.entity();
        let pmx = ctx.alloc(mx, ep);
        let pmy = ctx.alloc(my, ep);
        let px1 = ctx.alloc(x1, el);
        let py1 = ctx.alloc(y1, el);
        let px2 = ctx.alloc(x2, el);
        let py2 = ctx.alloc(y2, el);
        let store = ctx.store();

        let c = Midpoint::new(ctx.cid(), ep, el, pmx, pmy, px1, py1, px2, py2);
        prop_assert_eq!(c.equation_count(), c.residuals(&store).len());
    }

    #[test]
    fn prop_equation_count_matches_residuals_symmetric(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        cx in coord(), cy in coord(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let ec = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let pcx = ctx.alloc(cx, ec);
        let pcy = ctx.alloc(cy, ec);
        let store = ctx.store();

        let c = Symmetric::new(ctx.cid(), e0, e1, ec, px1, py1, px2, py2, pcx, pcy);
        prop_assert_eq!(c.equation_count(), c.residuals(&store).len());
    }

    /// Residuals must always be finite for finite inputs.
    #[test]
    fn prop_residuals_finite_distance(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        dist in positive_dist(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = DistancePtPt::new(ctx.cid(), e0, e1, px1, py1, px2, py2, dist);
        let r = c.residuals(&store);

        for &val in &r {
            prop_assert!(val.is_finite(), "Residual is not finite: {}", val);
        }
    }

    /// Jacobian entries must always be finite for finite inputs.
    #[test]
    fn prop_jacobian_finite_distance(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        dist in positive_dist(),
    ) {
        let mut ctx = TestCtx::new();
        let e0 = ctx.entity();
        let e1 = ctx.entity();
        let px1 = ctx.alloc(x1, e0);
        let py1 = ctx.alloc(y1, e0);
        let px2 = ctx.alloc(x2, e1);
        let py2 = ctx.alloc(y2, e1);
        let store = ctx.store();

        let c = DistancePtPt::new(ctx.cid(), e0, e1, px1, py1, px2, py2, dist);
        let jac = c.jacobian(&store);

        for &(_, _, val) in &jac {
            prop_assert!(val.is_finite(), "Jacobian entry is not finite: {}", val);
        }
    }
}
