//! Integration tests exercising the full solve pipeline with real sketch2d
//! entities and constraints.
//!
//! Each test builds a geometry problem using [`ConstraintSystem`] directly,
//! calls `solve()`, and verifies actual geometric properties of the result.

use crate::id::{EntityId, ParamId};
use crate::sketch2d::constraints::*;
use crate::sketch2d::entities::*;
use crate::system::{ConstraintSystem, SystemResult, SystemStatus};

// =========================================================================
// Tolerance
// =========================================================================

const TOL: f64 = 1e-6;

// =========================================================================
// Helpers
// =========================================================================

/// Add a [`Point2D`] entity to the system, returning `(entity_id, x, y)`.
fn add_point(sys: &mut ConstraintSystem, x: f64, y: f64) -> (EntityId, ParamId, ParamId) {
    let eid = sys.alloc_entity_id();
    let px = sys.alloc_param(x, eid);
    let py = sys.alloc_param(y, eid);
    sys.add_entity(Box::new(Point2D::new(eid, px, py)));
    (eid, px, py)
}

/// Add a [`Circle2D`] entity, returning `(entity_id, cx, cy, r)`.
fn add_circle_entity(
    sys: &mut ConstraintSystem,
    cx: f64,
    cy: f64,
    r: f64,
) -> (EntityId, ParamId, ParamId, ParamId) {
    let eid = sys.alloc_entity_id();
    let pcx = sys.alloc_param(cx, eid);
    let pcy = sys.alloc_param(cy, eid);
    let pr = sys.alloc_param(r, eid);
    sys.add_entity(Box::new(Circle2D::new(eid, pcx, pcy, pr)));
    (eid, pcx, pcy, pr)
}

/// Add a [`LineSegment2D`] entity that shares parameter IDs with two
/// existing [`Point2D`] entities.
fn add_line_segment(
    sys: &mut ConstraintSystem,
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
) -> EntityId {
    let eid = sys.alloc_entity_id();
    sys.add_entity(Box::new(LineSegment2D::new(eid, x1, y1, x2, y2)));
    eid
}

/// Euclidean distance between two points given their parameter IDs.
fn pt_dist(
    sys: &ConstraintSystem,
    x1: ParamId,
    y1: ParamId,
    x2: ParamId,
    y2: ParamId,
) -> f64 {
    let dx = sys.get_param(x2) - sys.get_param(x1);
    let dy = sys.get_param(y2) - sys.get_param(y1);
    (dx * dx + dy * dy).sqrt()
}

/// Assert that the system solve succeeded.
///
/// Accepts `Solved` unconditionally and `PartiallySolved` only when every
/// cluster has a small residual. Panics on `DiagnosticFailure`.
fn assert_solved(result: &SystemResult) {
    match &result.status {
        SystemStatus::Solved => {}
        SystemStatus::PartiallySolved => {
            for cr in &result.clusters {
                assert!(
                    cr.residual_norm < 1e-4,
                    "Cluster {:?} residual too large: {:.6e}",
                    cr.cluster_id,
                    cr.residual_norm,
                );
            }
        }
        SystemStatus::DiagnosticFailure(issues) => {
            panic!(
                "Expected Solved, got DiagnosticFailure ({} issues): {:?}",
                issues.len(),
                issues,
            );
        }
    }
}

// =========================================================================
// Test 1 -- Constrained triangle (3-4-5)
// =========================================================================

/// Build 3 points near a valid 3-4-5 right triangle, add 3 distance
/// constraints plus a fixed anchor, solve, and verify all three distances
/// are satisfied to within `TOL`.
#[test]
fn test_constrained_triangle() {
    let mut sys = ConstraintSystem::new();

    // Points near a 3-4-5 right triangle, slightly perturbed.
    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 3.1, -0.2);
    let (e2, x2, y2) = add_point(&mut sys, -0.1, 4.1);

    // Anchor P0 at the origin.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 0.0, 0.0)));

    // Three side-length constraints forming a 3-4-5 triangle.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e1, x0, y0, x1, y1, 3.0,
    )));

    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e1, e2, x1, y1, x2, y2, 5.0,
    )));

    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e2, e0, x2, y2, x0, y0, 4.0,
    )));

    let result = sys.solve();
    assert_solved(&result);

    // -- Verify geometric properties --

    let d01 = pt_dist(&sys, x0, y0, x1, y1);
    let d12 = pt_dist(&sys, x1, y1, x2, y2);
    let d20 = pt_dist(&sys, x2, y2, x0, y0);

    assert!(
        (d01 - 3.0).abs() < TOL,
        "d(P0,P1) = {d01}, expected 3.0"
    );
    assert!(
        (d12 - 5.0).abs() < TOL,
        "d(P1,P2) = {d12}, expected 5.0"
    );
    assert!(
        (d20 - 4.0).abs() < TOL,
        "d(P2,P0) = {d20}, expected 4.0"
    );

    // Anchor should be at the origin.
    assert!(
        sys.get_param(x0).abs() < TOL,
        "P0.x = {}, expected 0.0",
        sys.get_param(x0)
    );
    assert!(
        sys.get_param(y0).abs() < TOL,
        "P0.y = {}, expected 0.0",
        sys.get_param(y0)
    );
}

// =========================================================================
// Test 2 -- Constrained rectangle (4 x 3)
// =========================================================================

/// Build a 4x3 rectangle from 4 points, add distance, perpendicular, and
/// horizontal constraints, solve, and verify side lengths, right angles,
/// and parallel sides.
///
/// We anchor P0 at the origin via `fix_param` and enforce orientation
/// with a `Horizontal` constraint on P0-P1 (one param fixed, so the
/// reducer handles it analytically). Right angles at vertices are
/// enforced with `Perpendicular` constraints on adjacent line segments.
#[test]
fn test_constrained_rectangle() {
    let mut sys = ConstraintSystem::new();

    // Four points near a 4x3 rectangle, slightly perturbed.
    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 3.9, 0.2);
    let (e2, x2, y2) = add_point(&mut sys, 4.1, 2.8);
    let (e3, x3, y3) = add_point(&mut sys, -0.1, 3.1);

    // (a) Anchor P0 at the origin via fix_param.
    sys.fix_param(x0);
    sys.fix_param(y0);

    // (b) Four distance constraints (side lengths).
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e1, x0, y0, x1, y1, 4.0,
    )));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e1, e2, x1, y1, x2, y2, 3.0,
    )));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e2, e3, x2, y2, x3, y3, 4.0,
    )));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e3, e0, x3, y3, x0, y0, 3.0,
    )));

    // (c) Line segments for perpendicular constraints.
    let el01 = add_line_segment(&mut sys, x0, y0, x1, y1);
    let el12 = add_line_segment(&mut sys, x1, y1, x2, y2);
    let el23 = add_line_segment(&mut sys, x2, y2, x3, y3);
    let el30 = add_line_segment(&mut sys, x3, y3, x0, y0);

    // (d) Horizontal constraint on P0-P1 to fix orientation.
    //     P0's y is fixed via fix_param so this has 1 free param (y1)
    //     and is handled analytically by the reducer.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Horizontal::new(cid, e0, e1, y0, y1)));

    // (e) Three perpendicular constraints for right angles at P1, P2, P3.
    //     (The 4th angle at P0 is implied by the other three.)
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Perpendicular::new(
        cid, el01, el12, x0, y0, x1, y1, x1, y1, x2, y2,
    )));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Perpendicular::new(
        cid, el12, el23, x1, y1, x2, y2, x2, y2, x3, y3,
    )));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Perpendicular::new(
        cid, el23, el30, x2, y2, x3, y3, x3, y3, x0, y0,
    )));

    let result = sys.solve();
    assert_solved(&result);

    // -- Verify side lengths --

    let d01 = pt_dist(&sys, x0, y0, x1, y1);
    let d12 = pt_dist(&sys, x1, y1, x2, y2);
    let d23 = pt_dist(&sys, x2, y2, x3, y3);
    let d30 = pt_dist(&sys, x3, y3, x0, y0);

    assert!((d01 - 4.0).abs() < TOL, "d(P0,P1) = {d01}, expected 4.0");
    assert!((d12 - 3.0).abs() < TOL, "d(P1,P2) = {d12}, expected 3.0");
    assert!((d23 - 4.0).abs() < TOL, "d(P2,P3) = {d23}, expected 4.0");
    assert!((d30 - 3.0).abs() < TOL, "d(P3,P0) = {d30}, expected 3.0");

    // -- Verify right angle at P1 via dot product of adjacent sides --

    let dx01 = sys.get_param(x1) - sys.get_param(x0);
    let dy01 = sys.get_param(y1) - sys.get_param(y0);
    let dx12 = sys.get_param(x2) - sys.get_param(x1);
    let dy12 = sys.get_param(y2) - sys.get_param(y1);
    let dot_p1 = dx01 * dx12 + dy01 * dy12;
    assert!(
        dot_p1.abs() < TOL,
        "Angle at P1 is not 90 degrees, dot product = {dot_p1}"
    );

    // -- Verify right angle at P2 --

    let dx23 = sys.get_param(x3) - sys.get_param(x2);
    let dy23 = sys.get_param(y3) - sys.get_param(y2);
    let dot_p2 = dx12 * dx23 + dy12 * dy23;
    assert!(
        dot_p2.abs() < TOL,
        "Angle at P2 is not 90 degrees, dot product = {dot_p2}"
    );

    // -- Verify parallel opposite sides (bottom vs top) --

    let dx32 = sys.get_param(x2) - sys.get_param(x3);
    let dy32 = sys.get_param(y2) - sys.get_param(y3);
    let cross_bottom_top = dx01 * dy32 - dy01 * dx32;
    assert!(
        cross_bottom_top.abs() < TOL,
        "Bottom and top sides not parallel, cross = {cross_bottom_top}"
    );

    // -- Verify anchor at origin --

    assert!(sys.get_param(x0).abs() < TOL, "P0.x not at 0");
    assert!(sys.get_param(y0).abs() < TOL, "P0.y not at 0");
}

// =========================================================================
// Test 3 -- Circle tangent to line
// =========================================================================

/// Build a circle and a horizontal line, add a tangent constraint, solve,
/// and verify that the distance from the circle center to the line equals
/// the radius.
///
/// We leave both `cx` and `cy` free (2 free params in the tangent
/// constraint) so the reducer does not attempt a single-step analytical
/// elimination on the nonlinear tangent equation.
#[test]
fn test_circle_tangent_to_line() {
    let mut sys = ConstraintSystem::new();

    // Circle: center at (5, 5), radius 3.
    let (ec, cx, cy, cr) = add_circle_entity(&mut sys, 5.0, 5.0, 3.0);

    // Horizontal line from (0, 0) to (10, 0) -- all four endpoint params
    // are fixed so only the circle center moves.
    let (_ep1, lx1, ly1) = add_point(&mut sys, 0.0, 0.0);
    let (_ep2, lx2, ly2) = add_point(&mut sys, 10.0, 0.0);
    let el = add_line_segment(&mut sys, lx1, ly1, lx2, ly2);

    sys.fix_param(lx1);
    sys.fix_param(ly1);
    sys.fix_param(lx2);
    sys.fix_param(ly2);

    // Fix only the radius. Leave cx and cy free so the constraint has
    // 2 free parameters and the numerical solver iterates to convergence.
    sys.fix_param(cr);

    // Tangent constraint: distance(center, line) == radius.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(TangentLineCircle::new(
        cid, el, ec, lx1, ly1, lx2, ly2, cx, cy, cr,
    )));

    let result = sys.solve();
    assert_solved(&result);

    // -- Verify tangency --
    //
    // For a horizontal line y=0, the signed distance from (cx, cy) to the
    // line is simply |cy|.  Tangency requires |cy| == r.
    let center_y = sys.get_param(cy);
    let radius = sys.get_param(cr);
    let dist_to_line = center_y.abs();

    assert!(
        (dist_to_line - radius).abs() < TOL,
        "dist(center, line) = {dist_to_line}, radius = {radius}, expected equal"
    );
}

// =========================================================================
// Test 4 -- Coincident points
// =========================================================================

/// Two points at different starting positions with a coincident constraint.
/// After solving they must share the same coordinates.
#[test]
fn test_coincident_points() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 1.0, 2.0);
    let (e1, x1, y1) = add_point(&mut sys, 4.0, 7.0);

    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Coincident::new(
        cid, e0, e1, x0, y0, x1, y1,
    )));

    let result = sys.solve();
    assert_solved(&result);

    let p0x = sys.get_param(x0);
    let p0y = sys.get_param(y0);
    let p1x = sys.get_param(x1);
    let p1y = sys.get_param(y1);

    assert!(
        (p0x - p1x).abs() < TOL,
        "x-coordinates differ: P0.x = {p0x}, P1.x = {p1x}"
    );
    assert!(
        (p0y - p1y).abs() < TOL,
        "y-coordinates differ: P0.y = {p0y}, P1.y = {p1y}"
    );
}

// =========================================================================
// Test 5 -- Horizontal and vertical constraints
// =========================================================================

/// A reference point is fixed at (3, 5) via `fix_param`.  A target point
/// starts at (0, 0) and is constrained to share the same y (Horizontal)
/// and same x (Vertical) as the reference.  Each constraint has exactly
/// 1 free parameter, so the reducer solves them analytically.
#[test]
fn test_horizontal_vertical_constraints() {
    let mut sys = ConstraintSystem::new();

    // Reference point, fixed at (3, 5) via fix_param.
    let (e_ref, x_ref, y_ref) = add_point(&mut sys, 3.0, 5.0);
    sys.fix_param(x_ref);
    sys.fix_param(y_ref);

    // Target point, starting far away.
    let (e_tgt, x_tgt, y_tgt) = add_point(&mut sys, 0.0, 0.0);

    // Horizontal: same y-coordinate.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Horizontal::new(
        cid, e_ref, e_tgt, y_ref, y_tgt,
    )));

    // Vertical: same x-coordinate.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Vertical::new(
        cid, e_ref, e_tgt, x_ref, x_tgt,
    )));

    let result = sys.solve();
    assert_solved(&result);

    assert!(
        (sys.get_param(x_tgt) - 3.0).abs() < TOL,
        "x_tgt = {}, expected 3.0",
        sys.get_param(x_tgt),
    );
    assert!(
        (sys.get_param(y_tgt) - 5.0).abs() < TOL,
        "y_tgt = {}, expected 5.0",
        sys.get_param(y_tgt),
    );
}

// =========================================================================
// Test 6 -- Parallel lines
// =========================================================================

/// Two line segments, the first fixed horizontal, the second initially
/// off-horizontal.  A Parallel constraint is applied; after solving,
/// their direction vectors must be parallel.
#[test]
fn test_parallel_lines() {
    let mut sys = ConstraintSystem::new();

    // Line 1: fixed horizontal from (0, 0) to (4, 0).
    let (_ea, xa, ya) = add_point(&mut sys, 0.0, 0.0);
    let (_eb, xb, yb) = add_point(&mut sys, 4.0, 0.0);
    let el1 = add_line_segment(&mut sys, xa, ya, xb, yb);

    sys.fix_param(xa);
    sys.fix_param(ya);
    sys.fix_param(xb);
    sys.fix_param(yb);

    // Line 2: start at (1, 3) (fixed), end at (5, 4) (free).
    let (_ec, xc, yc) = add_point(&mut sys, 1.0, 3.0);
    let (_ed, xd, yd) = add_point(&mut sys, 5.0, 4.0);
    let el2 = add_line_segment(&mut sys, xc, yc, xd, yd);

    sys.fix_param(xc);
    sys.fix_param(yc);

    // Parallel constraint.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Parallel::new(
        cid, el1, el2, xa, ya, xb, yb, xc, yc, xd, yd,
    )));

    let result = sys.solve();
    assert_solved(&result);

    // -- Verify parallel: cross product of direction vectors ~= 0 --

    let dx1 = sys.get_param(xb) - sys.get_param(xa);
    let dy1 = sys.get_param(yb) - sys.get_param(ya);
    let dx2 = sys.get_param(xd) - sys.get_param(xc);
    let dy2 = sys.get_param(yd) - sys.get_param(yc);

    let cross = dx1 * dy2 - dy1 * dx2;
    assert!(
        cross.abs() < TOL,
        "Lines not parallel: cross product = {cross}"
    );

    // Verify neither direction vector has degenerate length.
    let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();
    assert!(len1 > 0.1, "Line 1 degenerate, length = {len1}");
    assert!(len2 > 0.1, "Line 2 degenerate, length = {len2}");

    // Since line 1 is horizontal (dy1 == 0), line 2 should also be
    // horizontal after solving (dy2 == 0).
    assert!(
        dy2.abs() < TOL,
        "Line 2 not horizontal after parallel constraint: dy2 = {dy2}"
    );
}

// =========================================================================
// Test 7 -- Mixed system with multiple clusters
// =========================================================================

/// Build two independent geometric subsystems (a triangle and a separate
/// fixed point), verify they decompose into at least 2 clusters and both
/// solve correctly.
#[test]
fn test_mixed_system_multiple_clusters() {
    let mut sys = ConstraintSystem::new();

    // ---- Cluster 1: triangle P0-P1-P2 ----

    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 3.1, 0.1);
    let (e2, x2, y2) = add_point(&mut sys, 0.1, 4.1);

    // Fix P0 at origin.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 0.0, 0.0)));

    // Two distance constraints connecting the triangle vertices.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e1, x0, y0, x1, y1, 3.0,
    )));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e1, e2, x1, y1, x2, y2, 5.0,
    )));

    // ---- Cluster 2: independent fixed point ----

    let (e3, x3, y3) = add_point(&mut sys, 100.0, 100.0);
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e3, x3, y3, 99.0, 99.0)));

    // ---- Solve ----

    let result = sys.solve();
    assert_solved(&result);

    // At least 2 independent clusters.
    assert!(
        result.clusters.len() >= 2,
        "Expected >= 2 clusters, got {}",
        result.clusters.len(),
    );

    // ---- Verify cluster 1 (triangle) ----

    let d01 = pt_dist(&sys, x0, y0, x1, y1);
    let d12 = pt_dist(&sys, x1, y1, x2, y2);
    assert!(
        (d01 - 3.0).abs() < TOL,
        "Triangle d(P0,P1) = {d01}, expected 3.0"
    );
    assert!(
        (d12 - 5.0).abs() < TOL,
        "Triangle d(P1,P2) = {d12}, expected 5.0"
    );
    assert!(
        sys.get_param(x0).abs() < TOL,
        "P0.x = {}, expected 0.0",
        sys.get_param(x0),
    );
    assert!(
        sys.get_param(y0).abs() < TOL,
        "P0.y = {}, expected 0.0",
        sys.get_param(y0),
    );

    // ---- Verify cluster 2 (fixed point) ----

    assert!(
        (sys.get_param(x3) - 99.0).abs() < TOL,
        "P3.x = {}, expected 99.0",
        sys.get_param(x3),
    );
    assert!(
        (sys.get_param(y3) - 99.0).abs() < TOL,
        "P3.y = {}, expected 99.0",
        sys.get_param(y3),
    );
}

// =========================================================================
// Test 8 -- Overconstrained system
// =========================================================================

/// Fix both endpoints of a segment to exact positions AND add a consistent
/// distance constraint.  The system is overconstrained (DOF < 0) but all
/// constraints are mutually consistent, so the solver should either
/// converge or report diagnostics while still satisfying the geometry.
#[test]
fn test_overconstrained_system() {
    let mut sys = ConstraintSystem::new();

    // Two points whose fixed positions are exactly 5 apart (3-4-5).
    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 3.0, 4.0);

    // Fix both points.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 0.0, 0.0)));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e1, x1, y1, 3.0, 4.0)));

    // Redundant but consistent distance = 5.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e1, x0, y0, x1, y1, 5.0,
    )));

    // System must be overconstrained.
    assert!(
        sys.degrees_of_freedom() < 0,
        "Expected negative DOF, got {}",
        sys.degrees_of_freedom(),
    );

    let result = sys.solve();

    // We accept Solved, PartiallySolved, or even DiagnosticFailure (the
    // solver might flag redundancy) as long as the actual positions are
    // correct.
    match &result.status {
        SystemStatus::Solved | SystemStatus::PartiallySolved => {}
        SystemStatus::DiagnosticFailure(_) => {
            // Diagnostics are expected for overconstrained systems.
        }
    }

    // Regardless of reported status, verify the geometry is satisfied.
    let p0x = sys.get_param(x0);
    let p0y = sys.get_param(y0);
    let p1x = sys.get_param(x1);
    let p1y = sys.get_param(y1);

    assert!(
        (p0x - 0.0).abs() < 1e-4,
        "P0.x = {p0x}, expected 0.0"
    );
    assert!(
        (p0y - 0.0).abs() < 1e-4,
        "P0.y = {p0y}, expected 0.0"
    );
    assert!(
        (p1x - 3.0).abs() < 1e-4,
        "P1.x = {p1x}, expected 3.0"
    );
    assert!(
        (p1y - 4.0).abs() < 1e-4,
        "P1.y = {p1y}, expected 4.0"
    );

    let d = pt_dist(&sys, x0, y0, x1, y1);
    assert!(
        (d - 5.0).abs() < 1e-4,
        "Distance = {d}, expected 5.0"
    );
}

// =========================================================================
// Test 9 -- Sketch2DBuilder triangle through pipeline
// =========================================================================

/// Build an equilateral triangle through the `Sketch2DBuilder` API,
/// solve it via the pipeline, and verify distances.
#[test]
fn test_builder_triangle_through_pipeline() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(5.0, 0.5); // perturbed
    let p2 = b.add_point(2.5, 4.0); // perturbed

    b.constrain_horizontal(p0, p1);
    b.constrain_distance(p0, p1, 5.0);
    b.constrain_distance(p1, p2, 5.0);
    b.constrain_distance(p2, p0, 5.0);

    let mut sys = b.build();
    let result = sys.solve();
    assert_solved(&result);

    // Residuals near zero.
    let residuals = sys.compute_residuals();
    let max_r = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
    assert!(max_r < TOL, "Max residual = {max_r}");
}

// =========================================================================
// Test 10 -- Builder: rectangle with perpendicular + equal length
// =========================================================================

/// Builder-based rectangle using perpendicular and equal-length constraints
/// instead of explicit horizontal/vertical.
#[test]
fn test_builder_rectangle_perpendicular_equal() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(6.0, 0.5);  // perturbed
    let p2 = b.add_point(6.5, 4.5);  // perturbed
    let p3 = b.add_point(0.5, 4.0);  // perturbed

    let l01 = b.add_line_segment(p0, p1);
    let l12 = b.add_line_segment(p1, p2);
    let l23 = b.add_line_segment(p2, p3);
    let l30 = b.add_line_segment(p3, p0);

    // Perpendicular at each vertex.
    b.constrain_perpendicular(l01, l12);
    b.constrain_perpendicular(l12, l23);
    b.constrain_perpendicular(l23, l30);

    // Side lengths.
    b.constrain_distance(p0, p1, 6.0);
    b.constrain_distance(p1, p2, 4.0);

    // Equal opposite sides.
    b.constrain_equal_length(l01, l23);
    b.constrain_equal_length(l12, l30);

    // Fix orientation.
    b.constrain_horizontal(p0, p1);

    let mut sys = b.build();
    let result = sys.solve();
    assert_solved(&result);

    let residuals = sys.compute_residuals();
    let max_r = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
    assert!(max_r < 1e-4, "Max residual = {max_r}");
}

// =========================================================================
// Test 11 -- Builder: point on circle + distance constraint
// =========================================================================

/// Point constrained on a fixed circle with a second point at a fixed
/// distance from the first. Verifies circle and distance constraints
/// compose correctly through the pipeline.
#[test]
fn test_builder_point_on_circle_with_distance() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();
    let _center = b.add_fixed_point(0.0, 0.0);
    let circle = b.add_circle(0.0, 0.0, 5.0);

    // Fix circle params.
    let cp = b.entity_param_ids(circle).to_vec();
    for &pid in &cp {
        b.fix_param(pid);
    }

    let p1 = b.add_point(4.5, 2.0);  // near the circle
    let p2 = b.add_point(-3.0, 4.0); // near the circle

    b.constrain_point_on_circle(p1, circle);
    b.constrain_point_on_circle(p2, circle);
    b.constrain_distance(p1, p2, 6.0);

    let mut sys = b.build();
    let result = sys.solve();
    assert_solved(&result);

    // Check residuals.
    let residuals = sys.compute_residuals();
    let max_r = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
    assert!(max_r < TOL, "Max residual = {max_r}");
}

// =========================================================================
// Test 12 -- Builder: incremental solve after moving a point
// =========================================================================

/// Build a triangle, solve, perturb one point, solve incrementally,
/// verify the solution is still correct.
#[test]
fn test_builder_incremental_triangle() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(3.0, 0.5);
    let p2 = b.add_point(0.5, 4.0);

    b.constrain_horizontal(p0, p1);
    b.constrain_distance(p0, p1, 3.0);
    b.constrain_distance(p1, p2, 5.0);
    b.constrain_distance(p2, p0, 4.0);

    let p2_params = b.entity_param_ids(p2).to_vec();

    let mut sys = b.build();

    // First solve.
    let r1 = sys.solve();
    assert_solved(&r1);

    // Perturb p2.
    sys.set_param(p2_params[0], 1.0);
    sys.set_param(p2_params[1], 1.0);

    // Incremental solve.
    let r2 = sys.solve_incremental();
    assert_solved(&r2);

    let residuals = sys.compute_residuals();
    let max_r = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
    assert!(
        max_r < TOL,
        "Max residual after incremental solve = {max_r}",
    );
}

// =========================================================================
// Test 13 -- Builder: two independent clusters
// =========================================================================

/// Two completely independent geometric shapes built through the builder.
/// Verifies decomposition into 2+ clusters.
#[test]
fn test_builder_two_independent_shapes() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();

    // Shape 1: two coincident points.
    let s1p0 = b.add_fixed_point(1.0, 2.0);
    let s1p1 = b.add_point(1.5, 2.5);
    b.constrain_coincident(s1p0, s1p1);

    // Shape 2: a fixed point far away (completely independent).
    let s2p0 = b.add_point(50.0, 50.5);
    b.constrain_fixed(s2p0, 50.0, 50.0);

    let mut sys = b.build();
    let result = sys.solve();
    assert_solved(&result);

    assert!(
        result.clusters.len() >= 2,
        "Expected >= 2 clusters, got {}",
        result.clusters.len(),
    );
}

// =========================================================================
// Test 14 -- Midpoint with symmetric constraint
// =========================================================================

/// Combine midpoint and symmetric constraints: midpoint of line is
/// constrained, and two points are symmetric about that midpoint.
#[test]
fn test_midpoint_and_symmetric() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();

    // Line endpoints.
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_fixed_point(10.0, 0.0);
    let line = b.add_line_segment(p0, p1);

    // Midpoint.
    let mid = b.add_point(4.0, 1.0); // off-center guess
    b.constrain_midpoint(mid, line);

    // Two points symmetric about the midpoint.
    let sym1 = b.add_point(3.0, 2.0);
    let sym2 = b.add_point(8.0, -1.0);
    b.constrain_symmetric(sym1, sym2, mid);

    // Fix sym1 position.
    b.constrain_fixed(sym1, 3.0, 2.0);

    // Extract param IDs before build() consumes the builder.
    let mid_params = b.entity_param_ids(mid).to_vec();
    let sym2_params = b.entity_param_ids(sym2).to_vec();

    let mut sys = b.build();
    let result = sys.solve();
    assert_solved(&result);

    // Midpoint should be at (5, 0).
    assert!(
        (sys.get_param(mid_params[0]) - 5.0).abs() < TOL,
        "mid.x = {}, expected 5.0",
        sys.get_param(mid_params[0]),
    );
    assert!(
        (sys.get_param(mid_params[1]) - 0.0).abs() < TOL,
        "mid.y = {}, expected 0.0",
        sys.get_param(mid_params[1]),
    );

    // sym2 should be 2*mid - sym1 = (10, 0) - (3, 2) = (7, -2).
    assert!(
        (sys.get_param(sym2_params[0]) - 7.0).abs() < TOL,
        "sym2.x = {}, expected 7.0",
        sys.get_param(sym2_params[0]),
    );
    assert!(
        (sys.get_param(sym2_params[1]) - (-2.0)).abs() < TOL,
        "sym2.y = {}, expected -2.0",
        sys.get_param(sym2_params[1]),
    );
}
