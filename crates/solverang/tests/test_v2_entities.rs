//! Comprehensive tests for all entity evaluators in `solverang::geometry::entities`.

use std::f64::consts::PI;

use solverang::geometry::entities;
use solverang::geometry::EntityKind;

const TOL: f64 = 1e-10;

fn assert_near(a: f64, b: f64, label: &str) {
    assert!(
        (a - b).abs() < TOL,
        "{label}: expected {b}, got {a} (diff = {})",
        (a - b).abs()
    );
}

// =====================================================================
// Point Evaluators
// =====================================================================

#[test]
fn test_point_2d_evaluate() {
    let params = [3.0, 4.0];
    let pos = entities::point::position_2d(&params);
    assert_near(pos[0], 3.0, "x");
    assert_near(pos[1], 4.0, "y");
}

#[test]
fn test_point_3d_evaluate() {
    let params = [1.0, 2.0, 3.0];
    let pos = entities::point::position_3d(&params);
    assert_near(pos[0], 1.0, "x");
    assert_near(pos[1], 2.0, "y");
    assert_near(pos[2], 3.0, "z");
}

#[test]
fn test_point_2d_tangent() {
    // Points have no meaningful tangent; the dispatch returns None.
    let params = [3.0, 4.0];
    let result = entities::tangent_at(EntityKind::Point2D, &params, 0.0);
    assert!(result.is_none(), "Point2D tangent should be None");
}

#[test]
fn test_point_zero() {
    let params = [0.0, 0.0];
    let pos = entities::point::position_2d(&params);
    assert_near(pos[0], 0.0, "x");
    assert_near(pos[1], 0.0, "y");
}

// =====================================================================
// Line Evaluators
// =====================================================================

#[test]
fn test_line_2d_evaluate_at_0() {
    // Line from (0,0) to (10,0), t=0 -> (0,0)
    let params = [0.0, 0.0, 10.0, 0.0];
    let p = entities::line::evaluate(&params, 0.0);
    assert_near(p[0], 0.0, "x at t=0");
    assert_near(p[1], 0.0, "y at t=0");
}

#[test]
fn test_line_2d_evaluate_at_1() {
    // Line from (0,0) to (10,0), t=1 -> (10,0)
    let params = [0.0, 0.0, 10.0, 0.0];
    let p = entities::line::evaluate(&params, 1.0);
    assert_near(p[0], 10.0, "x at t=1");
    assert_near(p[1], 0.0, "y at t=1");
}

#[test]
fn test_line_2d_evaluate_at_half() {
    // Line from (0,0) to (10,0), t=0.5 -> (5,0)
    let params = [0.0, 0.0, 10.0, 0.0];
    let p = entities::line::evaluate(&params, 0.5);
    assert_near(p[0], 5.0, "x at t=0.5");
    assert_near(p[1], 0.0, "y at t=0.5");
}

#[test]
fn test_line_2d_tangent() {
    // Line from (0,0) to (10,0), tangent is constant (10, 0).
    let params = [0.0, 0.0, 10.0, 0.0];
    let tan = entities::line::tangent(&params);
    assert_near(tan[0], 10.0, "tangent x");
    assert_near(tan[1], 0.0, "tangent y");

    // Tangent should be the same regardless of evaluation point.
    let tan2 = entities::line::tangent(&params);
    assert_near(tan[0], tan2[0], "tangent consistency x");
    assert_near(tan[1], tan2[1], "tangent consistency y");
}

#[test]
fn test_line_3d_evaluate() {
    // 3D line from (0,0,0) to (10,5,3)
    let params = [0.0, 0.0, 0.0, 10.0, 5.0, 3.0];

    let p0 = entities::line::evaluate(&params, 0.0);
    assert_near(p0[0], 0.0, "x at t=0");
    assert_near(p0[1], 0.0, "y at t=0");
    assert_near(p0[2], 0.0, "z at t=0");

    let p1 = entities::line::evaluate(&params, 1.0);
    assert_near(p1[0], 10.0, "x at t=1");
    assert_near(p1[1], 5.0, "y at t=1");
    assert_near(p1[2], 3.0, "z at t=1");

    let pm = entities::line::evaluate(&params, 0.5);
    assert_near(pm[0], 5.0, "x at t=0.5");
    assert_near(pm[1], 2.5, "y at t=0.5");
    assert_near(pm[2], 1.5, "z at t=0.5");
}

#[test]
fn test_line_diagonal() {
    // Line from (0,0) to (3,4), t=0.5 -> (1.5, 2.0)
    let params = [0.0, 0.0, 3.0, 4.0];
    let p = entities::line::evaluate(&params, 0.5);
    assert_near(p[0], 1.5, "x at t=0.5");
    assert_near(p[1], 2.0, "y at t=0.5");
}

// =====================================================================
// Circle Evaluators
// =====================================================================

#[test]
fn test_circle_evaluate_at_0() {
    // Circle center (0,0), r=5. t=0 -> angle=0 -> (5, 0).
    let params = [0.0, 0.0, 5.0];
    let p = entities::circle::evaluate_2d(&params, 0.0);
    assert_near(p[0], 5.0, "x at t=0");
    assert_near(p[1], 0.0, "y at t=0");
}

#[test]
fn test_circle_evaluate_at_quarter() {
    // Circle center (0,0), r=5. t=0.25 -> angle=pi/2 -> (0, 5).
    let params = [0.0, 0.0, 5.0];
    let p = entities::circle::evaluate_2d(&params, 0.25);
    assert_near(p[0], 0.0, "x at t=0.25");
    assert_near(p[1], 5.0, "y at t=0.25");
}

#[test]
fn test_circle_evaluate_at_half() {
    // Circle center (0,0), r=5. t=0.5 -> angle=pi -> (-5, 0).
    let params = [0.0, 0.0, 5.0];
    let p = entities::circle::evaluate_2d(&params, 0.5);
    assert_near(p[0], -5.0, "x at t=0.5");
    assert_near(p[1], 0.0, "y at t=0.5");
}

#[test]
fn test_circle_tangent_at_0() {
    // Circle center (0,0), r=5. At t=0 (point (5,0)), tangent is (0, r*2pi).
    // Formula: [-r*sin(0)*2pi, r*cos(0)*2pi] = [0, 5*2pi].
    let params = [0.0, 0.0, 5.0];
    let tan = entities::circle::tangent_2d(&params, 0.0);
    assert_near(tan[0], 0.0, "tangent x at t=0");
    assert_near(tan[1], 5.0 * 2.0 * PI, "tangent y at t=0");
}

#[test]
fn test_circle_evaluate_offset_center() {
    // Circle at (3,4) r=2. t=0 -> (3+2, 4) = (5, 4).
    let params = [3.0, 4.0, 2.0];
    let p = entities::circle::evaluate_2d(&params, 0.0);
    assert_near(p[0], 5.0, "x at t=0");
    assert_near(p[1], 4.0, "y at t=0");

    // t=0.25 -> (3, 4+2) = (3, 6)
    let p2 = entities::circle::evaluate_2d(&params, 0.25);
    assert_near(p2[0], 3.0, "x at t=0.25");
    assert_near(p2[1], 6.0, "y at t=0.25");
}

#[test]
fn test_circle_full_revolution() {
    // t=0 and t=1 (2pi full revolution) should give same point.
    let params = [0.0, 0.0, 5.0];
    let p0 = entities::circle::evaluate_2d(&params, 0.0);
    let p1 = entities::circle::evaluate_2d(&params, 1.0);
    assert_near(p0[0], p1[0], "full revolution x");
    assert_near(p0[1], p1[1], "full revolution y");
}

// =====================================================================
// Arc Evaluators
// =====================================================================

#[test]
fn test_arc_evaluate_start() {
    // Arc: center (0,0), r=1, from 0 to pi/2. t=0 -> angle=0 -> (1, 0).
    let params = [0.0, 0.0, 1.0, 0.0, PI / 2.0];
    let p = entities::arc::evaluate_2d(&params, 0.0);
    assert_near(p[0], 1.0, "x at start");
    assert_near(p[1], 0.0, "y at start");
}

#[test]
fn test_arc_evaluate_end() {
    // Arc: center (0,0), r=1, from 0 to pi/2. t=1 -> angle=pi/2 -> (0, 1).
    let params = [0.0, 0.0, 1.0, 0.0, PI / 2.0];
    let p = entities::arc::evaluate_2d(&params, 1.0);
    assert_near(p[0], 0.0, "x at end");
    assert_near(p[1], 1.0, "y at end");
}

#[test]
fn test_arc_evaluate_mid() {
    // Arc: center (0,0), r=1, from 0 to pi/2. t=0.5 -> angle=pi/4 -> (sqrt2/2, sqrt2/2).
    let params = [0.0, 0.0, 1.0, 0.0, PI / 2.0];
    let p = entities::arc::evaluate_2d(&params, 0.5);
    let sqrt2_2 = 2.0_f64.sqrt() / 2.0;
    assert_near(p[0], sqrt2_2, "x at mid");
    assert_near(p[1], sqrt2_2, "y at mid");
}

#[test]
fn test_arc_tangent() {
    // Arc: center (0,0), r=1, from 0 to pi/2.
    // At t=0 (angle=0, point (1,0)), tangent should be perpendicular to radius (0, +).
    let params = [0.0, 0.0, 1.0, 0.0, PI / 2.0];
    let tan = entities::arc::tangent_2d(&params, 0.0);
    assert_near(tan[0], 0.0, "tangent x at start");
    assert!(tan[1] > 0.0, "tangent y should be positive at start");

    // At t=1 (angle=pi/2, point (0,1)), tangent should be in (-x) direction.
    let tan1 = entities::arc::tangent_2d(&params, 1.0);
    assert!(tan1[0] < 0.0, "tangent x should be negative at end");
    assert_near(tan1[1], 0.0, "tangent y at end");
}

#[test]
fn test_arc_quarter_circle() {
    // Quarter circle from 0 to pi/2, center (0,0), r=3.
    let params = [0.0, 0.0, 3.0, 0.0, PI / 2.0];

    // Start point: (3, 0)
    let p0 = entities::arc::evaluate_2d(&params, 0.0);
    assert_near(p0[0], 3.0, "start x");
    assert_near(p0[1], 0.0, "start y");

    // End point: (0, 3)
    let p1 = entities::arc::evaluate_2d(&params, 1.0);
    assert_near(p1[0], 0.0, "end x");
    assert_near(p1[1], 3.0, "end y");

    // All points should be at distance r=3 from center.
    for i in 0..=10 {
        let t = i as f64 / 10.0;
        let p = entities::arc::evaluate_2d(&params, t);
        let dist = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert_near(dist, 3.0, &format!("distance at t={t}"));
    }
}

// =====================================================================
// Ellipse Evaluators
// =====================================================================

#[test]
fn test_ellipse_at_0() {
    // Ellipse at origin, rx=3, ry=2, no rotation. t=0 -> (3, 0).
    let params = [0.0, 0.0, 3.0, 2.0, 0.0];
    let p = entities::ellipse::evaluate_2d(&params, 0.0);
    assert_near(p[0], 3.0, "x at t=0");
    assert_near(p[1], 0.0, "y at t=0");
}

#[test]
fn test_ellipse_at_quarter() {
    // Ellipse at origin, rx=3, ry=2, no rotation. t=0.25 -> angle=pi/2 -> (0, 2).
    let params = [0.0, 0.0, 3.0, 2.0, 0.0];
    let p = entities::ellipse::evaluate_2d(&params, 0.25);
    assert_near(p[0], 0.0, "x at t=0.25");
    assert_near(p[1], 2.0, "y at t=0.25");
}

#[test]
fn test_ellipse_tangent() {
    // Ellipse at origin, rx=3, ry=2, no rotation.
    // At t=0 (point (3,0)), tangent should point in +y direction (perpendicular to major axis).
    // Formula: [-rx*sin(0)*2pi, ry*cos(0)*2pi] = [0, 2*2pi]
    let params = [0.0, 0.0, 3.0, 2.0, 0.0];
    let tan = entities::ellipse::tangent_2d(&params, 0.0);
    assert_near(tan[0], 0.0, "tangent x at t=0");
    assert!(tan[1] > 0.0, "tangent y should be positive");
    assert_near(tan[1], 2.0 * 2.0 * PI, "tangent y magnitude");
}

#[test]
fn test_ellipse_circle_case() {
    // When rx == ry, the ellipse should behave like a circle.
    let r = 5.0;
    let ellipse_params = [0.0, 0.0, r, r, 0.0];
    let circle_params = [0.0, 0.0, r];

    for i in 0..20 {
        let t = i as f64 / 20.0;
        let pe = entities::ellipse::evaluate_2d(&ellipse_params, t);
        let pc = entities::circle::evaluate_2d(&circle_params, t);
        assert_near(pe[0], pc[0], &format!("circle case x at t={t}"));
        assert_near(pe[1], pc[1], &format!("circle case y at t={t}"));
    }
}

#[test]
fn test_ellipse_rotated() {
    // Ellipse at origin, rx=2, ry=1, rotated by pi/2.
    // After 90 degree rotation, major axis is along Y, minor along X.
    // t=0: local (2,0) rotated 90 -> (0, 2)
    let params = [0.0, 0.0, 2.0, 1.0, PI / 2.0];
    let p0 = entities::ellipse::evaluate_2d(&params, 0.0);
    assert_near(p0[0], 0.0, "rotated x at t=0");
    assert_near(p0[1], 2.0, "rotated y at t=0");

    // t=0.25: local (0,1) rotated 90 -> (-1, 0)
    let p1 = entities::ellipse::evaluate_2d(&params, 0.25);
    assert_near(p1[0], -1.0, "rotated x at t=0.25");
    assert_near(p1[1], 0.0, "rotated y at t=0.25");
}

#[test]
fn test_ellipse_center_offset() {
    // Ellipse at (5,3), rx=2, ry=1, no rotation. t=0 -> (5+2, 3) = (7, 3).
    let params = [5.0, 3.0, 2.0, 1.0, 0.0];
    let p = entities::ellipse::evaluate_2d(&params, 0.0);
    assert_near(p[0], 7.0, "offset x at t=0");
    assert_near(p[1], 3.0, "offset y at t=0");

    // t=0.25 -> (5, 3+1) = (5, 4)
    let p2 = entities::ellipse::evaluate_2d(&params, 0.25);
    assert_near(p2[0], 5.0, "offset x at t=0.25");
    assert_near(p2[1], 4.0, "offset y at t=0.25");
}

// =====================================================================
// Bezier Evaluators
// =====================================================================

#[test]
fn test_bezier_at_0() {
    // Cubic Bezier: P0=(0,0), P1=(1,2), P2=(3,3), P3=(4,0). t=0 -> P0.
    let params = [0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 0.0];
    let p = entities::bezier::evaluate_cubic_2d(&params, 0.0);
    assert_near(p[0], 0.0, "x at t=0");
    assert_near(p[1], 0.0, "y at t=0");
}

#[test]
fn test_bezier_at_1() {
    // Cubic Bezier: P0=(0,0), P1=(1,2), P2=(3,3), P3=(4,0). t=1 -> P3.
    let params = [0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 0.0];
    let p = entities::bezier::evaluate_cubic_2d(&params, 1.0);
    assert_near(p[0], 4.0, "x at t=1");
    assert_near(p[1], 0.0, "y at t=1");
}

#[test]
fn test_bezier_at_half() {
    // Cubic Bezier with collinear control points: (0,0), (1,1), (2,2), (3,3).
    // This is a straight line, so t=0.5 -> (1.5, 1.5).
    let params = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let p = entities::bezier::evaluate_cubic_2d(&params, 0.5);
    assert_near(p[0], 1.5, "x at t=0.5 (de Casteljau)");
    assert_near(p[1], 1.5, "y at t=0.5 (de Casteljau)");

    // For a non-trivial curve, manually compute via de Casteljau.
    // P0=(0,0), P1=(0,4), P2=(4,4), P3=(4,0), t=0.5
    let params2 = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 0.0];
    let p2 = entities::bezier::evaluate_cubic_2d(&params2, 0.5);
    // De Casteljau at t=0.5:
    //   Level 1: Q0=(0,2), Q1=(2,4), Q2=(4,2)
    //   Level 2: R0=(1,3), R1=(3,3)
    //   Level 3: S0=(2,3)
    assert_near(p2[0], 2.0, "de Casteljau x");
    assert_near(p2[1], 3.0, "de Casteljau y");
}

#[test]
fn test_bezier_tangent_at_0() {
    // Cubic Bezier: P0=(0,0), P1=(1,2), P2=(3,3), P3=(4,0).
    // Tangent at t=0: B'(0) = 3*(P1-P0) = 3*(1,2) = (3, 6).
    let params = [0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 0.0];
    let tan = entities::bezier::tangent_cubic_2d(&params, 0.0);
    assert_near(tan[0], 3.0, "tangent x at t=0");
    assert_near(tan[1], 6.0, "tangent y at t=0");
}

#[test]
fn test_bezier_tangent_at_1() {
    // Cubic Bezier: P0=(0,0), P1=(1,2), P2=(3,3), P3=(4,0).
    // Tangent at t=1: B'(1) = 3*(P3-P2) = 3*(1,-3) = (3, -9).
    let params = [0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 0.0];
    let tan = entities::bezier::tangent_cubic_2d(&params, 1.0);
    assert_near(tan[0], 3.0, "tangent x at t=1");
    assert_near(tan[1], -9.0, "tangent y at t=1");
}

#[test]
fn test_bezier_straight_line() {
    // Control points on a line from (0,0) to (6,0): P0=(0,0), P1=(2,0), P2=(4,0), P3=(6,0).
    // Evaluate should match line evaluator.
    let bezier_params = [0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0];
    let line_params = [0.0, 0.0, 6.0, 0.0];

    for i in 0..=10 {
        let t = i as f64 / 10.0;
        let pb = entities::bezier::evaluate_cubic_2d(&bezier_params, t);
        let pl = entities::line::evaluate(&line_params, t);
        assert_near(pb[0], pl[0], &format!("straight line x at t={t}"));
        assert_near(pb[1], pl[1], &format!("straight line y at t={t}"));
    }
}

// =====================================================================
// Plane Evaluators
// =====================================================================

#[test]
fn test_plane_evaluate() {
    // XY plane: point on plane (0,0,0), normal (0,0,1).
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let pop = entities::plane::point_on_plane(&params);
    assert_near(pop[0], 0.0, "plane point x");
    assert_near(pop[1], 0.0, "plane point y");
    assert_near(pop[2], 0.0, "plane point z");

    // A point on the XY plane should have zero distance.
    let test_point = [5.0, 3.0, 0.0];
    let dist = entities::plane::distance(&params, &test_point);
    assert_near(dist, 0.0, "on-plane distance");

    // A point above it should have positive signed distance.
    let above = [1.0, 2.0, 7.0];
    let sd = entities::plane::signed_distance(&params, &above);
    assert_near(sd, 7.0, "above-plane signed distance");
}

#[test]
fn test_plane_normal() {
    // XY plane at origin with normal (0,0,1).
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let n = entities::plane::normal(&params);
    assert_near(n[0], 0.0, "normal x");
    assert_near(n[1], 0.0, "normal y");
    assert_near(n[2], 1.0, "normal z");
}

#[test]
fn test_plane_tilted() {
    // YZ plane: point (0,0,0), normal (1,0,0).
    let params = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let n = entities::plane::normal(&params);
    assert_near(n[0], 1.0, "tilted normal x");
    assert_near(n[1], 0.0, "tilted normal y");
    assert_near(n[2], 0.0, "tilted normal z");

    // Project point (5, 3, 4) onto YZ plane -> (0, 3, 4).
    let proj = entities::plane::project_point(&params, &[5.0, 3.0, 4.0]);
    assert_near(proj[0], 0.0, "projected x");
    assert_near(proj[1], 3.0, "projected y");
    assert_near(proj[2], 4.0, "projected z");
}

#[test]
fn test_plane_offset() {
    // Plane at z=10 with normal pointing up.
    let params = [0.0, 0.0, 10.0, 0.0, 0.0, 1.0];

    // Point above at z=15 -> distance 5.
    let sd = entities::plane::signed_distance(&params, &[0.0, 0.0, 15.0]);
    assert_near(sd, 5.0, "offset plane signed distance above");

    // Point on plane at z=10 -> distance 0.
    let sd2 = entities::plane::signed_distance(&params, &[5.0, 3.0, 10.0]);
    assert_near(sd2, 0.0, "offset plane on-plane distance");

    // Point below at z=7 -> distance -3.
    let sd3 = entities::plane::signed_distance(&params, &[0.0, 0.0, 7.0]);
    assert_near(sd3, -3.0, "offset plane signed distance below");
}

// =====================================================================
// Sphere Evaluators
// =====================================================================

#[test]
fn test_sphere_evaluate_north_pole() {
    // Unit sphere at origin. v=pi/2 (latitude = 90) -> north pole (0, 0, 1).
    let params = [0.0, 0.0, 0.0, 1.0];
    let p = entities::sphere::evaluate(&params, 0.0, PI / 2.0);
    assert_near(p[0], 0.0, "north pole x");
    assert_near(p[1], 0.0, "north pole y");
    assert_near(p[2], 1.0, "north pole z");
}

#[test]
fn test_sphere_evaluate_equator() {
    // Unit sphere at origin. u=0, v=0 -> equator point (1, 0, 0).
    let params = [0.0, 0.0, 0.0, 1.0];
    let p = entities::sphere::evaluate(&params, 0.0, 0.0);
    assert_near(p[0], 1.0, "equator x");
    assert_near(p[1], 0.0, "equator y");
    assert_near(p[2], 0.0, "equator z");
}

#[test]
fn test_sphere_all_points_on_surface() {
    // All evaluated points should be at distance r from center.
    let params = [0.0, 0.0, 0.0, 3.0];
    let r = 3.0;

    for i in 0..10 {
        for j in 0..10 {
            let u = (i as f64 / 10.0) * 2.0 * PI;
            let v = (j as f64 / 10.0) * PI - PI / 2.0; // latitude [-pi/2, pi/2]
            let p = entities::sphere::evaluate(&params, u, v);
            let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert_near(dist, r, &format!("surface distance at u={u:.2}, v={v:.2}"));
        }
    }
}

#[test]
fn test_sphere_normal() {
    // Unit sphere at origin. Normal should point radially outward.
    let params = [0.0, 0.0, 0.0, 1.0];

    // At north pole (u=0, v=pi/2): normal is (0, 0, 1).
    let n_top = entities::sphere::normal_at(&params, 0.0, PI / 2.0);
    assert_near(n_top[0], 0.0, "normal x at north pole");
    assert_near(n_top[1], 0.0, "normal y at north pole");
    assert_near(n_top[2], 1.0, "normal z at north pole");

    // At equator (u=0, v=0): normal is (1, 0, 0).
    let n_eq = entities::sphere::normal_at(&params, 0.0, 0.0);
    assert_near(n_eq[0], 1.0, "normal x at equator");
    assert_near(n_eq[1], 0.0, "normal y at equator");
    assert_near(n_eq[2], 0.0, "normal z at equator");
}

#[test]
fn test_sphere_offset_center() {
    // Sphere at (5, 3, 2) with radius 2.
    let params = [5.0, 3.0, 2.0, 2.0];

    // North pole: (5, 3, 2+2) = (5, 3, 4).
    let p = entities::sphere::evaluate(&params, 0.0, PI / 2.0);
    assert_near(p[0], 5.0, "offset north pole x");
    assert_near(p[1], 3.0, "offset north pole y");
    assert_near(p[2], 4.0, "offset north pole z");

    // All points should be at distance 2 from center (5,3,2).
    for i in 0..8 {
        for j in 0..8 {
            let u = (i as f64 / 8.0) * 2.0 * PI;
            let v = (j as f64 / 8.0) * PI - PI / 2.0;
            let pt = entities::sphere::evaluate(&params, u, v);
            let dx = pt[0] - 5.0;
            let dy = pt[1] - 3.0;
            let dz = pt[2] - 2.0;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert_near(dist, 2.0, &format!("offset sphere radius at u={u:.2}, v={v:.2}"));
        }
    }
}

// =====================================================================
// Cylinder Evaluators
// =====================================================================

#[test]
fn test_cylinder_evaluate() {
    // Z-axis cylinder: point (0,0,0), axis (0,0,1), r=1.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

    // At u=0, v=0: should be at radius 1 from Z axis in XY plane at z=0.
    let p = entities::cylinder::evaluate(&params, 0.0, 0.0);
    let dist_xy = (p[0] * p[0] + p[1] * p[1]).sqrt();
    assert_near(dist_xy, 1.0, "cylinder radial distance");
    assert_near(p[2], 0.0, "cylinder z at v=0");

    // At u=0, v=5: same radius, at z=5.
    let p2 = entities::cylinder::evaluate(&params, 0.0, 5.0);
    let dist_xy2 = (p2[0] * p2[0] + p2[1] * p2[1]).sqrt();
    assert_near(dist_xy2, 1.0, "cylinder radial distance at v=5");
    assert_near(p2[2], 5.0, "cylinder z at v=5");
}

#[test]
fn test_cylinder_normal() {
    // Z-axis cylinder.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

    // Normal should be radial (perpendicular to Z axis) and unit length.
    let n = entities::cylinder::normal_at(&params, 0.0, 0.0);
    let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    assert_near(n_len, 1.0, "cylinder normal length");
    assert_near(n[2], 0.0, "cylinder normal z component");

    // Normal at u=pi/2 should also be unit length and perpendicular to Z.
    let n2 = entities::cylinder::normal_at(&params, PI / 2.0, 3.0);
    let n2_len = (n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]).sqrt();
    assert_near(n2_len, 1.0, "cylinder normal length at u=pi/2");
    assert_near(n2[2], 0.0, "cylinder normal z component at u=pi/2");
}

#[test]
fn test_cylinder_along_axis() {
    // Points at the same angle u but different heights should have the same XY position.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0];
    let u = 0.7;

    let p1 = entities::cylinder::evaluate(&params, u, 0.0);
    let p2 = entities::cylinder::evaluate(&params, u, 3.0);
    let p3 = entities::cylinder::evaluate(&params, u, -2.0);

    assert_near(p1[0], p2[0], "same angle x at v=0 vs v=3");
    assert_near(p1[1], p2[1], "same angle y at v=0 vs v=3");
    assert_near(p1[0], p3[0], "same angle x at v=0 vs v=-2");
    assert_near(p1[1], p3[1], "same angle y at v=0 vs v=-2");

    // Z coordinates should differ by v offset.
    assert_near(p2[2] - p1[2], 3.0, "z difference v=3 vs v=0");
    assert_near(p3[2] - p1[2], -2.0, "z difference v=-2 vs v=0");
}

#[test]
fn test_cylinder_circumference() {
    // Full revolution (u=0 and u=2pi) at same height should give same point.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let v = 2.5;

    let p0 = entities::cylinder::evaluate(&params, 0.0, v);
    let p2pi = entities::cylinder::evaluate(&params, 2.0 * PI, v);

    assert_near(p0[0], p2pi[0], "circumference x");
    assert_near(p0[1], p2pi[1], "circumference y");
    assert_near(p0[2], p2pi[2], "circumference z");
}

// =====================================================================
// Cone Evaluators
// =====================================================================

#[test]
fn test_cone_evaluate_apex() {
    // Z-axis cone: apex (0,0,0), axis (0,0,1), half_angle=pi/4.
    // At v=0 (apex), radius = 0 * tan(pi/4) = 0, so point = apex.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];
    let p = entities::cone::evaluate(&params, 0.0, 0.0);
    assert_near(p[0], 0.0, "apex x");
    assert_near(p[1], 0.0, "apex y");
    assert_near(p[2], 0.0, "apex z");

    // At any angle u, v=0 should still be at apex.
    let p2 = entities::cone::evaluate(&params, PI / 3.0, 0.0);
    assert_near(p2[0], 0.0, "apex x at u=pi/3");
    assert_near(p2[1], 0.0, "apex y at u=pi/3");
    assert_near(p2[2], 0.0, "apex z at u=pi/3");
}

#[test]
fn test_cone_evaluate_base() {
    // Z-axis cone: apex (0,0,0), axis (0,0,1), half_angle=pi/4.
    // At v=1: z=1, radius = 1 * tan(pi/4) = 1.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];
    let p = entities::cone::evaluate(&params, 0.0, 1.0);
    let dist_xy = (p[0] * p[0] + p[1] * p[1]).sqrt();
    assert_near(dist_xy, 1.0, "base circle radius");
    assert_near(p[2], 1.0, "base z");

    // At v=2: z=2, radius = 2 * tan(pi/4) = 2.
    let p2 = entities::cone::evaluate(&params, 0.0, 2.0);
    let dist_xy2 = (p2[0] * p2[0] + p2[1] * p2[1]).sqrt();
    assert_near(dist_xy2, 2.0, "base circle radius at v=2");
    assert_near(p2[2], 2.0, "base z at v=2");
}

#[test]
fn test_cone_normal() {
    // Z-axis cone with half_angle=pi/4.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];
    let n = entities::cone::normal_at(&params, 0.0, 1.0);
    let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    assert_near(n_len, 1.0, "cone normal unit length");

    // For a 45 degree cone at u=0:
    // Normal = cos(pi/4)*radial - sin(pi/4)*axis.
    // With radial pointing in +X, axis in +Z:
    // Normal = (sqrt2/2, 0, -sqrt2/2)
    let sqrt2_2 = 2.0_f64.sqrt() / 2.0;
    assert!((n[0] - sqrt2_2).abs() < 1e-9, "cone normal x component: {}", n[0]);
    assert_near(n[1], 0.0, "cone normal y component");
    assert!((n[2] + sqrt2_2).abs() < 1e-9, "cone normal z component: {}", n[2]);
}

#[test]
fn test_cone_cross_section() {
    // At any fixed v, varying u should trace a circle of radius v*tan(half_angle).
    let half_angle = PI / 6.0; // 30 degrees
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, half_angle];
    let v = 3.0;
    let expected_radius = v * half_angle.tan();

    for i in 0..20 {
        let u = (i as f64 / 20.0) * 2.0 * PI;
        let p = entities::cone::evaluate(&params, u, v);
        let dist_xy = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert_near(dist_xy, expected_radius, &format!("cross section radius at u={u:.2}"));
        assert_near(p[2], v, &format!("cross section z at u={u:.2}"));
    }
}

// =====================================================================
// Torus Evaluators
// =====================================================================

#[test]
fn test_torus_evaluate_outer() {
    // Torus: center (0,0,0), axis Z, R=2, r=1.
    // Outermost point: u=0, v=0 -> (R+r, 0, 0) = (3, 0, 0).
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let p = entities::torus::evaluate(&params, 0.0, 0.0);
    assert_near(p[0], 3.0, "outer x");
    assert_near(p[1], 0.0, "outer y");
    assert_near(p[2], 0.0, "outer z");
}

#[test]
fn test_torus_evaluate_inner() {
    // Innermost point: u=0, v=pi -> (R-r, 0, 0) = (1, 0, 0).
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let p = entities::torus::evaluate(&params, 0.0, PI);
    assert_near(p[0], 1.0, "inner x");
    assert_near(p[1], 0.0, "inner y");
    assert_near(p[2], 0.0, "inner z");
}

#[test]
fn test_torus_surface_point() {
    // For any surface point, the distance from the point to the closest point
    // on the major circle should equal the minor radius.
    // Torus: center (0,0,0), axis Z, R=3, r=1.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0];
    let big_r = 3.0;
    let small_r = 1.0;

    for i in 0..8 {
        for j in 0..8 {
            let u = (i as f64 / 8.0) * 2.0 * PI;
            let v = (j as f64 / 8.0) * 2.0 * PI;
            let p = entities::torus::evaluate(&params, u, v);

            // The tube center at angle u is at (R*cos(u)*e1 + R*sin(u)*e2) from center.
            // For Z-axis torus with our orthonormal frame, the tube center is in the XY plane.
            // Distance from axis in XY plane:
            let dist_xy = (p[0] * p[0] + p[1] * p[1]).sqrt();

            // The tube center is at distance R from Z axis.
            // Distance from point to tube center ring:
            let dist_to_tube_center = ((dist_xy - big_r).powi(2) + p[2] * p[2]).sqrt();
            assert_near(
                dist_to_tube_center,
                small_r,
                &format!("tube radius check at u={u:.2}, v={v:.2}"),
            );
        }
    }
}

#[test]
fn test_torus_normal() {
    // Torus: center (0,0,0), axis Z, R=2, r=1.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    // At u=0, v=0 (outermost, pointing in +X): normal should be (1, 0, 0).
    let n = entities::torus::normal_at(&params, 0.0, 0.0);
    let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    assert_near(n_len, 1.0, "torus normal length");
    assert_near(n[0], 1.0, "torus normal x at outer");
    assert_near(n[1], 0.0, "torus normal y at outer");
    assert_near(n[2], 0.0, "torus normal z at outer");

    // At u=0, v=pi (innermost, pointing in -X): normal should be (-1, 0, 0).
    let n2 = entities::torus::normal_at(&params, 0.0, PI);
    assert_near(n2[0], -1.0, "torus normal x at inner");
    assert_near(n2[1], 0.0, "torus normal y at inner");
    assert_near(n2[2], 0.0, "torus normal z at inner");

    // At u=0, v=pi/2 (top of tube): normal should be (0, 0, 1).
    let n3 = entities::torus::normal_at(&params, 0.0, PI / 2.0);
    assert_near(n3[0], 0.0, "torus normal x at top");
    assert_near(n3[1], 0.0, "torus normal y at top");
    assert_near(n3[2], 1.0, "torus normal z at top");
}

// =====================================================================
// Dispatch Function Tests
// =====================================================================

#[test]
fn test_dispatch_evaluate_point() {
    // Dispatch evaluate_at with EntityKind::Point2D.
    let params = [3.0, 4.0];
    let result = entities::evaluate_at(EntityKind::Point2D, &params, 0.0);
    assert!(result.is_some(), "dispatch should return Some for Point2D");
    let p = result.unwrap();
    assert_near(p[0], 3.0, "dispatch point x");
    assert_near(p[1], 4.0, "dispatch point y");

    // t value should be ignored for points.
    let result2 = entities::evaluate_at(EntityKind::Point2D, &params, 0.75);
    let p2 = result2.unwrap();
    assert_near(p2[0], 3.0, "dispatch point x (t ignored)");
    assert_near(p2[1], 4.0, "dispatch point y (t ignored)");
}

#[test]
fn test_dispatch_evaluate_circle() {
    // Dispatch evaluate_at with EntityKind::Circle2D.
    let params = [0.0, 0.0, 5.0];
    let result = entities::evaluate_at(EntityKind::Circle2D, &params, 0.0);
    assert!(result.is_some(), "dispatch should return Some for Circle2D");
    let p = result.unwrap();
    assert_near(p[0], 5.0, "dispatch circle x at t=0");
    assert_near(p[1], 0.0, "dispatch circle y at t=0");

    // Compare dispatch with direct call.
    let direct = entities::circle::evaluate_2d(&params, 0.25);
    let dispatched = entities::evaluate_at(EntityKind::Circle2D, &params, 0.25).unwrap();
    assert_near(dispatched[0], direct[0], "dispatch vs direct circle x");
    assert_near(dispatched[1], direct[1], "dispatch vs direct circle y");
}

#[test]
fn test_dispatch_tangent_line() {
    // Dispatch tangent_at with EntityKind::Line2D.
    let params = [0.0, 0.0, 10.0, 5.0];
    let result = entities::tangent_at(EntityKind::Line2D, &params, 0.5);
    assert!(result.is_some(), "dispatch should return Some for Line2D tangent");
    let tan = result.unwrap();
    assert_near(tan[0], 10.0, "dispatch line tangent x");
    assert_near(tan[1], 5.0, "dispatch line tangent y");

    // Compare dispatch with direct call.
    let direct = entities::line::tangent(&params);
    assert_near(tan[0], direct[0], "dispatch vs direct tangent x");
    assert_near(tan[1], direct[1], "dispatch vs direct tangent y");
}

#[test]
fn test_dispatch_evaluate_bezier() {
    // Dispatch evaluate_at with EntityKind::CubicBezier2D.
    let params = [0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 0.0];

    // t=0 -> first control point.
    let result0 = entities::evaluate_at(EntityKind::CubicBezier2D, &params, 0.0);
    assert!(result0.is_some(), "dispatch should return Some for CubicBezier2D");
    let p0 = result0.unwrap();
    assert_near(p0[0], 0.0, "dispatch bezier x at t=0");
    assert_near(p0[1], 0.0, "dispatch bezier y at t=0");

    // t=1 -> last control point.
    let result1 = entities::evaluate_at(EntityKind::CubicBezier2D, &params, 1.0);
    let p1 = result1.unwrap();
    assert_near(p1[0], 4.0, "dispatch bezier x at t=1");
    assert_near(p1[1], 0.0, "dispatch bezier y at t=1");

    // Compare dispatch with direct call at t=0.5.
    let direct = entities::bezier::evaluate_cubic_2d(&params, 0.5);
    let dispatched = entities::evaluate_at(EntityKind::CubicBezier2D, &params, 0.5).unwrap();
    assert_near(dispatched[0], direct[0], "dispatch vs direct bezier x");
    assert_near(dispatched[1], direct[1], "dispatch vs direct bezier y");
}

// =====================================================================
// Additional edge-case / integration tests
// =====================================================================

#[test]
fn test_dispatch_returns_none_for_surfaces() {
    // Surface entity kinds should return None from the curve dispatch.
    let sphere_params = [0.0, 0.0, 0.0, 1.0];
    assert!(entities::evaluate_at(EntityKind::Sphere, &sphere_params, 0.0).is_none());
    assert!(entities::tangent_at(EntityKind::Sphere, &sphere_params, 0.0).is_none());

    let cyl_params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    assert!(entities::evaluate_at(EntityKind::Cylinder, &cyl_params, 0.0).is_none());

    let cone_params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];
    assert!(entities::evaluate_at(EntityKind::Cone, &cone_params, 0.0).is_none());

    let torus_params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    assert!(entities::evaluate_at(EntityKind::Torus, &torus_params, 0.0).is_none());
}

#[test]
fn test_dispatch_point_3d() {
    let params = [1.0, 2.0, 3.0];
    let result = entities::evaluate_at(EntityKind::Point3D, &params, 0.0);
    assert!(result.is_some());
    let p = result.unwrap();
    assert_eq!(p.len(), 3);
    assert_near(p[0], 1.0, "3D dispatch x");
    assert_near(p[1], 2.0, "3D dispatch y");
    assert_near(p[2], 3.0, "3D dispatch z");
}

#[test]
fn test_dispatch_line_3d() {
    let params = [0.0, 0.0, 0.0, 6.0, 8.0, 10.0];
    let result = entities::evaluate_at(EntityKind::Line3D, &params, 0.5);
    assert!(result.is_some());
    let p = result.unwrap();
    assert_near(p[0], 3.0, "3D line dispatch x");
    assert_near(p[1], 4.0, "3D line dispatch y");
    assert_near(p[2], 5.0, "3D line dispatch z");
}

#[test]
fn test_line_length() {
    let params_345 = [0.0, 0.0, 3.0, 4.0];
    assert_near(entities::line::length(&params_345), 5.0, "3-4-5 triangle hypotenuse");

    let params_3d = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    assert_near(
        entities::line::length(&params_3d),
        3.0_f64.sqrt(),
        "3D unit diagonal",
    );
}

#[test]
fn test_arc_sweep_and_length() {
    // Quarter circle with radius 2.
    let params = [0.0, 0.0, 2.0, 0.0, PI / 2.0];
    assert_near(entities::arc::sweep_angle(&params), PI / 2.0, "sweep angle");
    assert_near(entities::arc::arc_length(&params), PI, "arc length (r*sweep)");
}

#[test]
fn test_ellipse_eccentricity() {
    // Circle: e = 0.
    let circle = [0.0, 0.0, 1.0, 1.0, 0.0];
    assert_near(entities::ellipse::eccentricity(&circle), 0.0, "circle eccentricity");

    // Ellipse rx=2, ry=1: e = sqrt(1 - 1/4) = sqrt(3)/2.
    let ell = [0.0, 0.0, 2.0, 1.0, 0.0];
    let expected = (3.0_f64 / 4.0).sqrt();
    assert_near(entities::ellipse::eccentricity(&ell), expected, "ellipse eccentricity");
}

#[test]
fn test_bezier_curvature_straight_line() {
    // Straight line has zero curvature.
    let params = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
    let curv = entities::bezier::curvature_cubic_2d(&params, 0.5);
    assert!(curv.abs() < TOL, "straight line curvature should be ~0");
}

#[test]
fn test_circle_3d_xy_plane() {
    // Circle in XY plane (normal = +Z), center at origin, radius 2.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0];

    // All points should be in the XY plane at distance 2 from origin.
    for i in 0..10 {
        let t = i as f64 / 10.0;
        let p = entities::circle::evaluate_3d(&params, t);
        assert_near(p[2], 0.0, &format!("3D circle z at t={t}"));
        let dist = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert_near(dist, 2.0, &format!("3D circle radius at t={t}"));
    }
}

#[test]
fn test_bezier_3d_endpoints() {
    // 3D cubic Bezier: P0=(0,0,0), P1=(1,0,1), P2=(2,3,2), P3=(3,3,3).
    let params = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0, 3.0];
    let p0 = entities::bezier::evaluate_cubic_3d(&params, 0.0);
    assert_near(p0[0], 0.0, "3D bezier x at t=0");
    assert_near(p0[1], 0.0, "3D bezier y at t=0");
    assert_near(p0[2], 0.0, "3D bezier z at t=0");

    let p1 = entities::bezier::evaluate_cubic_3d(&params, 1.0);
    assert_near(p1[0], 3.0, "3D bezier x at t=1");
    assert_near(p1[1], 3.0, "3D bezier y at t=1");
    assert_near(p1[2], 3.0, "3D bezier z at t=1");
}

#[test]
fn test_quad_bezier_2d() {
    // Quadratic Bezier: P0=(0,0), P1=(1,2), P2=(2,0).
    let params = [0.0, 0.0, 1.0, 2.0, 2.0, 0.0];

    // t=0 -> P0
    let p0 = entities::bezier::evaluate_quad_2d(&params, 0.0);
    assert_near(p0[0], 0.0, "quad bezier x at t=0");
    assert_near(p0[1], 0.0, "quad bezier y at t=0");

    // t=1 -> P2
    let p1 = entities::bezier::evaluate_quad_2d(&params, 1.0);
    assert_near(p1[0], 2.0, "quad bezier x at t=1");
    assert_near(p1[1], 0.0, "quad bezier y at t=1");

    // t=0.5: Q(0.5) = 0.25*(0,0) + 0.5*(1,2) + 0.25*(2,0) = (0 + 0.5 + 0.5, 0 + 1 + 0) = (1, 1)
    let pm = entities::bezier::evaluate_quad_2d(&params, 0.5);
    assert_near(pm[0], 1.0, "quad bezier x at t=0.5");
    assert_near(pm[1], 1.0, "quad bezier y at t=0.5");
}

#[test]
fn test_sphere_signed_distance() {
    let params = [0.0, 0.0, 0.0, 5.0]; // r=5

    // On surface
    let on = [5.0, 0.0, 0.0];
    assert_near(entities::sphere::signed_distance_to_surface(&params, &on), 0.0, "on surface");

    // Outside at distance 2
    let out = [7.0, 0.0, 0.0];
    assert_near(entities::sphere::signed_distance_to_surface(&params, &out), 2.0, "outside");

    // Inside
    let inside = [3.0, 0.0, 0.0];
    assert_near(
        entities::sphere::signed_distance_to_surface(&params, &inside),
        -2.0,
        "inside",
    );
}

#[test]
fn test_cylinder_signed_distance() {
    // Z-axis cylinder, r=1.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

    // On surface
    let on = [1.0, 0.0, 3.0];
    assert_near(
        entities::cylinder::signed_distance_to_surface(&params, &on),
        0.0,
        "cylinder on surface",
    );

    // Outside
    let out = [2.0, 0.0, 5.0];
    assert_near(
        entities::cylinder::signed_distance_to_surface(&params, &out),
        1.0,
        "cylinder outside",
    );

    // Inside (on axis)
    let on_axis = [0.0, 0.0, 10.0];
    assert_near(
        entities::cylinder::signed_distance_to_surface(&params, &on_axis),
        -1.0,
        "cylinder on axis",
    );
}

#[test]
fn test_cone_signed_distance() {
    // Z-axis cone, half_angle=pi/4.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];

    // On surface at z=1, x=1 (since tan(pi/4) = 1).
    let on = [1.0, 0.0, 1.0];
    assert_near(
        entities::cone::signed_distance_to_surface(&params, &on),
        0.0,
        "cone on surface",
    );

    // At apex.
    let apex = [0.0, 0.0, 0.0];
    assert_near(
        entities::cone::signed_distance_to_surface(&params, &apex),
        0.0,
        "cone at apex",
    );
}

#[test]
fn test_torus_signed_distance() {
    // Torus: center (0,0,0), axis Z, R=2, r=1.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    // On surface (outermost).
    let outer = [3.0, 0.0, 0.0];
    assert_near(
        entities::torus::signed_distance_to_surface(&params, &outer),
        0.0,
        "torus outer surface",
    );

    // On surface (innermost).
    let inner = [1.0, 0.0, 0.0];
    assert_near(
        entities::torus::signed_distance_to_surface(&params, &inner),
        0.0,
        "torus inner surface",
    );

    // On surface (top of tube).
    let top = [2.0, 0.0, 1.0];
    assert_near(
        entities::torus::signed_distance_to_surface(&params, &top),
        0.0,
        "torus top surface",
    );
}

#[test]
fn test_sphere_project_point() {
    let params = [0.0, 0.0, 0.0, 1.0];

    // Project from outside.
    let proj = entities::sphere::project_point(&params, &[5.0, 0.0, 0.0]);
    assert_near(proj[0], 1.0, "projected x");
    assert_near(proj[1], 0.0, "projected y");
    assert_near(proj[2], 0.0, "projected z");

    // Project from inside.
    let proj2 = entities::sphere::project_point(&params, &[0.3, 0.0, 0.0]);
    assert_near(proj2[0], 1.0, "projected from inside x");
    assert_near(proj2[1], 0.0, "projected from inside y");
    assert_near(proj2[2], 0.0, "projected from inside z");

    // Project diagonal point onto unit sphere.
    let proj3 = entities::sphere::project_point(&params, &[2.0, 2.0, 2.0]);
    let expected = 1.0 / 3.0_f64.sqrt();
    assert_near(proj3[0], expected, "projected diagonal x");
    assert_near(proj3[1], expected, "projected diagonal y");
    assert_near(proj3[2], expected, "projected diagonal z");
}

#[test]
fn test_plane_project_point() {
    // XY plane.
    let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let proj = entities::plane::project_point(&params, &[3.0, 4.0, 5.0]);
    assert_near(proj[0], 3.0, "projected x");
    assert_near(proj[1], 4.0, "projected y");
    assert_near(proj[2], 0.0, "projected z");
}
