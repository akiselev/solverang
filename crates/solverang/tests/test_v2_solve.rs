//! Comprehensive end-to-end integration tests for the v2 builder -> solve -> verify pipeline.
//!
//! These tests exercise ConstraintSystemBuilder construction, LM/Auto/Robust solving,
//! and post-solve geometric property verification for 2D, 3D, circle, bezier, and
//! mixed constraint scenarios.

#![cfg(feature = "geometry")]

use solverang::geometry::constraints::*;
use solverang::geometry::{ConstraintSystem, ConstraintSystemBuilder};
use solverang::{AutoSolver, LMConfig, LMSolver, Problem, RobustSolver, SolveResult};

// =============================================================================
// Helper Functions
// =============================================================================

/// Solve a constraint system using the LM solver.
/// Panics with a descriptive message if the solver fails outright.
/// Accepts a "nearly converged" result (residual_norm < 1e-4) as success.
fn solve_system(system: &dyn Problem) -> Vec<f64> {
    let initial = system.initial_point(1.0);
    let solver = LMSolver::new(LMConfig::default());
    match solver.solve(system, &initial) {
        SolveResult::Converged { solution, .. } => solution,
        SolveResult::NotConverged {
            solution,
            residual_norm,
            ..
        } => {
            if residual_norm < 1e-4 {
                solution
            } else {
                panic!(
                    "Solver did not converge, residual_norm={}",
                    residual_norm
                );
            }
        }
        SolveResult::Failed { error } => panic!("Solver failed: {}", error),
    }
}

/// Extract 2D point coordinates from a ConstraintSystem by entity creation index.
fn point_2d_coords(system: &ConstraintSystem, entity_index: usize) -> (f64, f64) {
    let handles = system.handles();
    let h = &handles[entity_index];
    let vals = system.params().values();
    (vals[h.params.start], vals[h.params.start + 1])
}

/// Extract 3D point coordinates from a ConstraintSystem by entity creation index.
fn point_3d_coords(system: &ConstraintSystem, entity_index: usize) -> (f64, f64, f64) {
    let handles = system.handles();
    let h = &handles[entity_index];
    let vals = system.params().values();
    (
        vals[h.params.start],
        vals[h.params.start + 1],
        vals[h.params.start + 2],
    )
}

/// Compute Euclidean distance between two 2D entities in the system.
fn dist_2d_entities(system: &ConstraintSystem, e1: usize, e2: usize) -> f64 {
    let (x1, y1) = point_2d_coords(system, e1);
    let (x2, y2) = point_2d_coords(system, e2);
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

/// Compute Euclidean distance between two 3D entities in the system.
fn dist_3d_entities(system: &ConstraintSystem, e1: usize, e2: usize) -> f64 {
    let (x1, y1, z1) = point_3d_coords(system, e1);
    let (x2, y2, z2) = point_3d_coords(system, e2);
    ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt()
}

/// Get circle parameters (cx, cy, r) by entity index.
fn circle_2d_params(system: &ConstraintSystem, entity_index: usize) -> (f64, f64, f64) {
    let handles = system.handles();
    let h = &handles[entity_index];
    let vals = system.params().values();
    (
        vals[h.params.start],
        vals[h.params.start + 1],
        vals[h.params.start + 2],
    )
}

/// Convergence tolerance for geometric property verification.
const TOL: f64 = 1e-4;

// =============================================================================
// 1. Basic 2D Geometry Solve Tests
// =============================================================================

#[test]
fn test_solve_triangle() {
    // Build a triangle: p0 fixed at origin, 3 distance constraints.
    let mut system = ConstraintSystemBuilder::new()
        .name("triangle")
        .point_2d_fixed(0.0, 0.0) // 0: fixed origin
        .point_2d(10.0, 0.0)      // 1: initial guess
        .point_2d(5.0, 1.0)       // 2: initial guess
        .horizontal(0, 1)         // pin p1 horizontally relative to p0
        .distance(0, 1, 10.0)     // |p0-p1| = 10
        .distance(1, 2, 8.0)      // |p1-p2| = 8
        .distance(2, 0, 6.0)      // |p2-p0| = 6
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let d01 = dist_2d_entities(&system, 0, 1);
    let d12 = dist_2d_entities(&system, 1, 2);
    let d20 = dist_2d_entities(&system, 2, 0);

    assert!((d01 - 10.0).abs() < TOL, "d01={}, expected 10.0", d01);
    assert!((d12 - 8.0).abs() < TOL, "d12={}, expected 8.0", d12);
    assert!((d20 - 6.0).abs() < TOL, "d20={}, expected 6.0", d20);
}

#[test]
fn test_solve_right_triangle() {
    // Fixed origin, horizontal constraint on p0-p1, distance p0-p1=3, p0-p2=4, p1-p2=5.
    // This is a 3-4-5 right triangle. The right angle is at p0.
    let mut system = ConstraintSystemBuilder::new()
        .name("right_triangle")
        .point_2d_fixed(0.0, 0.0) // 0: origin
        .point_2d(3.0, 0.5)       // 1
        .point_2d(0.5, 4.0)       // 2
        .horizontal(0, 1)         // p0-p1 horizontal
        .vertical(0, 2)           // p0-p2 vertical
        .distance(0, 1, 3.0)
        .distance(0, 2, 4.0)
        .distance(1, 2, 5.0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (p0x, p0y) = point_2d_coords(&system, 0);
    let (p1x, p1y) = point_2d_coords(&system, 1);
    let (p2x, p2y) = point_2d_coords(&system, 2);

    // Verify horizontal: same y for p0 and p1
    assert!((p0y - p1y).abs() < TOL, "p0-p1 not horizontal");
    // Verify vertical: same x for p0 and p2
    assert!((p0x - p2x).abs() < TOL, "p0-p2 not vertical");

    // Verify it's a right triangle: dot product of legs = 0
    let v1x = p1x - p0x;
    let v1y = p1y - p0y;
    let v2x = p2x - p0x;
    let v2y = p2y - p0y;
    let dot = v1x * v2x + v1y * v2y;
    assert!(dot.abs() < TOL, "Not a right triangle, dot={}", dot);

    // Verify distances
    assert!((dist_2d_entities(&system, 0, 1) - 3.0).abs() < TOL);
    assert!((dist_2d_entities(&system, 0, 2) - 4.0).abs() < TOL);
    assert!((dist_2d_entities(&system, 1, 2) - 5.0).abs() < TOL);
}

#[test]
fn test_solve_equilateral_triangle() {
    // Three equal distances, fixed first two points.
    let side = 10.0;
    let height = side * (3.0_f64).sqrt() / 2.0;

    let mut system = ConstraintSystemBuilder::new()
        .name("equilateral")
        .point_2d_fixed(0.0, 0.0)         // 0
        .point_2d_fixed(side, 0.0)         // 1
        .point_2d(side / 2.0, height + 1.0) // 2: slightly perturbed
        .distance(0, 1, side)
        .distance(1, 2, side)
        .distance(2, 0, side)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let d01 = dist_2d_entities(&system, 0, 1);
    let d12 = dist_2d_entities(&system, 1, 2);
    let d20 = dist_2d_entities(&system, 2, 0);

    assert!((d01 - side).abs() < TOL, "d01={}", d01);
    assert!((d12 - side).abs() < TOL, "d12={}", d12);
    assert!((d20 - side).abs() < TOL, "d20={}", d20);

    // All solved distances should be equal
    assert!((d01 - d12).abs() < TOL, "d01 != d12");
    assert!((d12 - d20).abs() < TOL, "d12 != d20");
}

#[test]
fn test_solve_square() {
    // 4 points, first fixed. h(0,1), v(1,2), h(2,3), v(3,0), dist(0,1)=5, dist(1,2)=5.
    let mut system = ConstraintSystemBuilder::new()
        .name("square")
        .point_2d_fixed(0.0, 0.0) // 0
        .point_2d(5.0, 0.5)       // 1
        .point_2d(4.5, 5.0)       // 2
        .point_2d(0.5, 4.5)       // 3
        .horizontal(0, 1)
        .vertical(1, 2)
        .horizontal(2, 3)
        .vertical(3, 0)
        .distance(0, 1, 5.0)
        .distance(1, 2, 5.0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (p0x, p0y) = point_2d_coords(&system, 0);
    let (p1x, p1y) = point_2d_coords(&system, 1);
    let (p2x, p2y) = point_2d_coords(&system, 2);
    let (p3x, p3y) = point_2d_coords(&system, 3);

    // Horizontal/vertical checks
    assert!((p0y - p1y).abs() < TOL, "bottom not horizontal");
    assert!((p1x - p2x).abs() < TOL, "right not vertical");
    assert!((p2y - p3y).abs() < TOL, "top not horizontal");
    assert!((p3x - p0x).abs() < TOL, "left not vertical");

    // Distance checks
    let d01 = dist_2d_entities(&system, 0, 1);
    let d12 = dist_2d_entities(&system, 1, 2);
    assert!((d01 - 5.0).abs() < TOL, "d01={}", d01);
    assert!((d12 - 5.0).abs() < TOL, "d12={}", d12);

    // All sides equal (rectangle with equal adjacent sides = square)
    let d23 = dist_2d_entities(&system, 2, 3);
    let d30 = dist_2d_entities(&system, 3, 0);
    assert!((d23 - 5.0).abs() < TOL, "d23={}", d23);
    assert!((d30 - 5.0).abs() < TOL, "d30={}", d30);
}

#[test]
fn test_solve_rectangle() {
    // 4 points with fixed origin, horizontal + vertical constraints plus two different distances.
    let mut system = ConstraintSystemBuilder::new()
        .name("rectangle")
        .point_2d_fixed(0.0, 0.0) // 0: bottom-left
        .point_2d(8.0, 0.5)       // 1: bottom-right (perturbed)
        .point_2d(7.5, 5.0)       // 2: top-right (perturbed)
        .point_2d(0.5, 4.5)       // 3: top-left (perturbed)
        .horizontal(0, 1)         // bottom edge
        .horizontal(3, 2)         // top edge
        .vertical(0, 3)           // left edge
        .vertical(1, 2)           // right edge
        .distance(0, 1, 10.0)     // width = 10
        .distance(0, 3, 5.0)      // height = 5
        .build();

    assert_eq!(system.degrees_of_freedom(), 0);

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (p0x, p0y) = point_2d_coords(&system, 0);
    let (p1x, p1y) = point_2d_coords(&system, 1);
    let (p2x, p2y) = point_2d_coords(&system, 2);
    let (p3x, p3y) = point_2d_coords(&system, 3);

    // Verify corner positions (origin at (0,0))
    assert!((p0x - 0.0).abs() < TOL && (p0y - 0.0).abs() < TOL);
    assert!((p1x - 10.0).abs() < TOL && (p1y - 0.0).abs() < TOL);
    assert!((p2x - 10.0).abs() < TOL && (p2y - 5.0).abs() < TOL);
    assert!((p3x - 0.0).abs() < TOL && (p3y - 5.0).abs() < TOL);
}

// =============================================================================
// 2. Circle Constraint Tests
// =============================================================================

#[test]
fn test_solve_point_on_circle() {
    // Fixed circle at (0,0) r=5, one free point starting at (3,4). Point on circle.
    let mut system = ConstraintSystemBuilder::new()
        .name("point_on_circle")
        .circle_2d(0.0, 0.0, 5.0)  // 0: circle
        .point_2d(3.0, 4.0)        // 1: point
        .fix(0)                     // fix the circle
        .point_on_circle(1, 0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (px, py) = point_2d_coords(&system, 1);
    let (cx, cy, r) = circle_2d_params(&system, 0);

    let dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
    assert!(
        (dist - r).abs() < TOL,
        "Point distance from center = {}, expected {}",
        dist,
        r
    );
}

#[test]
fn test_solve_tangent_line_circle() {
    // Circle at (5,5) r=3 and a line. Tangent constraint.
    // Start line near tangent position.
    let mut system = ConstraintSystemBuilder::new()
        .name("tangent_line_circle")
        .circle_2d(5.0, 5.0, 3.0)          // 0: circle
        .line_2d(0.0, 2.0, 10.0, 2.0)      // 1: line (initially horizontal at y=2)
        .fix(0)                              // fix circle
        .tangent_line_circle(1, 0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (cx, cy, r) = circle_2d_params(&system, 0);
    let handles = system.handles();
    let vals = system.params().values();
    let lh = &handles[1];
    let x1 = vals[lh.params.start];
    let y1 = vals[lh.params.start + 1];
    let x2 = vals[lh.params.start + 2];
    let y2 = vals[lh.params.start + 3];

    // Perpendicular distance from center to line = |cross| / |line_length|
    let dx = x2 - x1;
    let dy = y2 - y1;
    let cross = dx * (cy - y1) - dy * (cx - x1);
    let line_len = (dx * dx + dy * dy).sqrt();
    let perp_dist = cross.abs() / line_len;

    assert!(
        (perp_dist - r).abs() < TOL,
        "Perpendicular distance = {}, expected radius {}",
        perp_dist,
        r
    );
}

#[test]
fn test_solve_concentric_circles() {
    // Two circles, concentric constraint. Fix first.
    let mut system = ConstraintSystemBuilder::new()
        .name("concentric")
        .circle_2d(5.0, 5.0, 3.0)      // 0
        .circle_2d(5.5, 4.5, 7.0)      // 1: center slightly off
        .fix(0)
        .concentric(0, 1)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (cx1, cy1, _r1) = circle_2d_params(&system, 0);
    let (cx2, cy2, _r2) = circle_2d_params(&system, 1);

    assert!((cx1 - cx2).abs() < TOL, "Centers not concentric in x");
    assert!((cy1 - cy2).abs() < TOL, "Centers not concentric in y");
}

#[test]
fn test_solve_equal_radius() {
    // Two circles with equal_radius. Fix first circle entirely, fix center of second.
    let mut system = ConstraintSystemBuilder::new()
        .name("equal_radius")
        .circle_2d(0.0, 0.0, 5.0)    // 0
        .circle_2d(10.0, 10.0, 3.0)  // 1: different radius initially
        .fix(0)                        // fix first circle entirely
        .fix_param_at(1, 0)           // fix second circle center x
        .fix_param_at(1, 1)           // fix second circle center y
        .equal_radius(0, 1)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (_cx1, _cy1, r1) = circle_2d_params(&system, 0);
    let (_cx2, _cy2, r2) = circle_2d_params(&system, 1);

    assert!(
        (r1 - r2).abs() < TOL,
        "Radii not equal: r1={}, r2={}",
        r1,
        r2
    );
}

#[test]
fn test_solve_tangent_circles() {
    // Two circles with tangent_circles(external). Fix both centers, let radii adjust.
    // Actually: fix first circle, fix center of second, free radius of second.
    let mut system = ConstraintSystemBuilder::new()
        .name("tangent_circles")
        .circle_2d(0.0, 0.0, 3.0)     // 0: at origin, r=3
        .circle_2d(10.0, 0.0, 4.0)    // 1: at (10,0), r=4, adjust to r=7
        .fix(0)                         // fix first circle
        .fix_param_at(1, 0)            // fix cx of second
        .fix_param_at(1, 1)            // fix cy of second
        .tangent_circles(0, 1, true)   // external tangency
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (cx1, cy1, r1) = circle_2d_params(&system, 0);
    let (cx2, cy2, r2) = circle_2d_params(&system, 1);

    let center_dist = ((cx2 - cx1).powi(2) + (cy2 - cy1).powi(2)).sqrt();
    assert!(
        (center_dist - (r1 + r2)).abs() < TOL,
        "External tangency: center_dist={}, r1+r2={}",
        center_dist,
        r1 + r2
    );
}

// =============================================================================
// 3. Mixed and Complex Geometry
// =============================================================================

#[test]
fn test_solve_collinear_points() {
    // Three 2D points with collinear constraint. p0 and p1 fixed, p2 free.
    // Also add a distance constraint from p0 to p2 to pin it.
    let mut system = ConstraintSystemBuilder::new()
        .name("collinear")
        .point_2d_fixed(0.0, 0.0)  // 0
        .point_2d_fixed(10.0, 10.0) // 1
        .point_2d(3.0, 4.0)         // 2: not yet on line
        .point_2d(7.0, 8.0)         // 3: not yet on line (needed for collinear constraint)
        .distance(0, 2, 5.0 * (2.0_f64).sqrt()) // place p2 at dist 5*sqrt(2) from origin
        .build();

    // Add collinear constraint: all 4 points must be collinear
    let handles = system.handles();
    let pr = |i: usize| handles[i].params;
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(CollinearConstraint::from_points(
        id,
        pr(0),
        pr(1),
        pr(2),
        pr(3),
    )));

    // Add another distance constraint to pin p3
    let id2 = system.next_constraint_id();
    system.add_constraint(Box::new(DistanceConstraint::new(
        id2,
        pr(0),
        pr(3),
        7.0 * (2.0_f64).sqrt(),
    )));

    let solution = solve_system(&system);
    system.set_values(&solution);

    // Verify collinearity: cross product of (p1-p0) and (p2-p0) should be ~0
    let (p0x, p0y) = point_2d_coords(&system, 0);
    let (p1x, p1y) = point_2d_coords(&system, 1);
    let (p2x, p2y) = point_2d_coords(&system, 2);
    let (p3x, p3y) = point_2d_coords(&system, 3);

    let cross_02 = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x);
    let cross_03 = (p1x - p0x) * (p3y - p0y) - (p1y - p0y) * (p3x - p0x);

    assert!(cross_02.abs() < TOL, "p2 not collinear, cross={}", cross_02);
    assert!(cross_03.abs() < TOL, "p3 not collinear, cross={}", cross_03);
}

#[test]
fn test_solve_symmetric_points() {
    // Two points symmetric about a fixed center.
    let mut system = ConstraintSystemBuilder::new()
        .name("symmetric")
        .point_2d(1.0, 2.0)           // 0: p1
        .point_2d(9.0, 8.0)           // 1: p2
        .point_2d_fixed(5.0, 5.0)     // 2: center (fixed)
        .symmetric(0, 1, 2)
        .distance(0, 2, 5.0)          // pin distance from p1 to center
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (p1x, p1y) = point_2d_coords(&system, 0);
    let (p2x, p2y) = point_2d_coords(&system, 1);
    let (cx, cy) = point_2d_coords(&system, 2);

    // Midpoint of p1 and p2 should be center
    let mid_x = (p1x + p2x) / 2.0;
    let mid_y = (p1y + p2y) / 2.0;

    assert!(
        (mid_x - cx).abs() < TOL,
        "Midpoint x={}, center x={}",
        mid_x,
        cx
    );
    assert!(
        (mid_y - cy).abs() < TOL,
        "Midpoint y={}, center y={}",
        mid_y,
        cy
    );
}

#[test]
fn test_solve_parallel_lines() {
    // Two lines with parallel constraint. Fix first line.
    let mut system = ConstraintSystemBuilder::new()
        .name("parallel_lines")
        .line_2d(0.0, 0.0, 5.0, 3.0)    // 0: first line
        .line_2d(2.0, 5.0, 8.0, 9.0)    // 1: second line (not yet parallel)
        .fix(0)                            // fix first line
        .parallel(0, 1)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let handles = system.handles();
    let vals = system.params().values();
    let l1 = &handles[0];
    let l2 = &handles[1];

    let d1x = vals[l1.params.start + 2] - vals[l1.params.start];
    let d1y = vals[l1.params.start + 3] - vals[l1.params.start + 1];
    let d2x = vals[l2.params.start + 2] - vals[l2.params.start];
    let d2y = vals[l2.params.start + 3] - vals[l2.params.start + 1];

    // Cross product of direction vectors should be 0 for parallel lines
    let cross = d1x * d2y - d1y * d2x;
    assert!(cross.abs() < TOL, "Lines not parallel, cross={}", cross);
}

#[test]
fn test_solve_perpendicular_lines() {
    // Two lines with perpendicular constraint. Fix first line.
    let mut system = ConstraintSystemBuilder::new()
        .name("perpendicular_lines")
        .line_2d(0.0, 0.0, 5.0, 0.0)    // 0: horizontal line
        .line_2d(3.0, -2.0, 4.0, 3.0)   // 1: second line (not yet perpendicular)
        .fix(0)                            // fix first line
        .perpendicular(0, 1)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let handles = system.handles();
    let vals = system.params().values();
    let l1 = &handles[0];
    let l2 = &handles[1];

    let d1x = vals[l1.params.start + 2] - vals[l1.params.start];
    let d1y = vals[l1.params.start + 3] - vals[l1.params.start + 1];
    let d2x = vals[l2.params.start + 2] - vals[l2.params.start];
    let d2y = vals[l2.params.start + 3] - vals[l2.params.start + 1];

    // Dot product of direction vectors should be 0 for perpendicular lines
    let dot = d1x * d2x + d1y * d2y;
    assert!(
        dot.abs() < TOL,
        "Lines not perpendicular, dot={}",
        dot
    );
}

#[test]
fn test_solve_midpoint() {
    // Three points where one is the midpoint of the other two.
    // Fix two endpoints, let the midpoint solve.
    let mut system = ConstraintSystemBuilder::new()
        .name("midpoint")
        .point_2d(5.5, 4.5)           // 0: midpoint (perturbed initial guess)
        .point_2d_fixed(0.0, 0.0)     // 1: start (fixed)
        .point_2d_fixed(10.0, 8.0)    // 2: end (fixed)
        .midpoint(0, 1, 2)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (mx, my) = point_2d_coords(&system, 0);
    let (sx, sy) = point_2d_coords(&system, 1);
    let (ex, ey) = point_2d_coords(&system, 2);

    let expected_x = (sx + ex) / 2.0;
    let expected_y = (sy + ey) / 2.0;

    assert!(
        (mx - expected_x).abs() < TOL,
        "Midpoint x={}, expected {}",
        mx,
        expected_x
    );
    assert!(
        (my - expected_y).abs() < TOL,
        "Midpoint y={}, expected {}",
        my,
        expected_y
    );
}

// =============================================================================
// 4. 3D Geometry Tests
// =============================================================================

#[test]
fn test_solve_3d_distance() {
    // Two 3D points with a distance constraint. Fixed first point.
    let mut system = ConstraintSystemBuilder::new()
        .name("3d_distance")
        .point_3d_fixed(0.0, 0.0, 0.0) // 0
        .point_3d(3.0, 4.0, 5.0)       // 1
        .distance(0, 1, 10.0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let d = dist_3d_entities(&system, 0, 1);
    assert!((d - 10.0).abs() < TOL, "3D distance = {}, expected 10.0", d);
}

#[test]
fn test_solve_3d_triangle() {
    // Triangle in 3D space. 3 points, 3 distance constraints.
    // Fix first point and add constraints to fully pin p1 and p2 (with enough DOF removal).
    let mut system = ConstraintSystemBuilder::new()
        .name("3d_triangle")
        .point_3d_fixed(0.0, 0.0, 0.0)   // 0: fixed
        .point_3d_fixed(10.0, 0.0, 0.0)  // 1: fixed
        .point_3d(5.0, 8.0, 1.0)         // 2: free (3 DOF)
        .distance(0, 1, 10.0)
        .distance(1, 2, 8.0)
        .distance(2, 0, 6.0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let d01 = dist_3d_entities(&system, 0, 1);
    let d12 = dist_3d_entities(&system, 1, 2);
    let d20 = dist_3d_entities(&system, 2, 0);

    assert!((d01 - 10.0).abs() < TOL, "d01={}", d01);
    assert!((d12 - 8.0).abs() < TOL, "d12={}", d12);
    assert!((d20 - 6.0).abs() < TOL, "d20={}", d20);
}

// =============================================================================
// 5. Bezier and Continuity Tests
// =============================================================================

#[test]
fn test_solve_bezier_g0() {
    // Two cubic beziers with g0_continuity. End of first = start of second.
    // Fix first bezier, let second bezier's P0 adjust.
    let mut system = ConstraintSystemBuilder::new()
        .name("bezier_g0")
        .cubic_bezier_2d([
            [0.0, 0.0],
            [2.0, 4.0],
            [8.0, 4.0],
            [10.0, 0.0],
        ]) // 0: first bezier
        .cubic_bezier_2d([
            [10.5, 0.5], // slightly off from (10,0)
            [12.0, -4.0],
            [18.0, -4.0],
            [20.0, 0.0],
        ]) // 1: second bezier
        .fix(0) // fix first bezier entirely
        .g0_continuity(0, 1)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let handles = system.handles();
    let vals = system.params().values();

    // End of bezier 0: P3 at offsets 6,7
    let b0 = &handles[0];
    let end_x = vals[b0.params.start + 6];
    let end_y = vals[b0.params.start + 7];

    // Start of bezier 1: P0 at offsets 0,1
    let b1 = &handles[1];
    let start_x = vals[b1.params.start];
    let start_y = vals[b1.params.start + 1];

    assert!(
        (end_x - start_x).abs() < TOL,
        "G0 x: end={}, start={}",
        end_x,
        start_x
    );
    assert!(
        (end_y - start_y).abs() < TOL,
        "G0 y: end={}, start={}",
        end_y,
        start_y
    );
}

#[test]
fn test_solve_bezier_g1() {
    // Two cubic beziers with g1_continuity (includes G0 + tangent alignment).
    // Fix first bezier.
    let mut system = ConstraintSystemBuilder::new()
        .name("bezier_g1")
        .cubic_bezier_2d([
            [0.0, 0.0],
            [2.0, 4.0],
            [8.0, 4.0],
            [10.0, 0.0],
        ]) // 0
        .cubic_bezier_2d([
            [10.5, 0.5], // P0 off
            [12.0, -3.0], // P1 roughly aligned
            [18.0, -4.0],
            [20.0, 0.0],
        ]) // 1
        .fix(0) // fix first bezier
        .g1_continuity(0, 1)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let handles = system.handles();
    let vals = system.params().values();

    // Verify G0: P3 of bezier0 = P0 of bezier1
    let b0 = &handles[0];
    let b1 = &handles[1];
    let p3a_x = vals[b0.params.start + 6];
    let p3a_y = vals[b0.params.start + 7];
    let p0b_x = vals[b1.params.start];
    let p0b_y = vals[b1.params.start + 1];

    assert!(
        (p3a_x - p0b_x).abs() < TOL,
        "G0 x: {} != {}",
        p3a_x,
        p0b_x
    );
    assert!(
        (p3a_y - p0b_y).abs() < TOL,
        "G0 y: {} != {}",
        p3a_y,
        p0b_y
    );

    // Verify G1: tangent vectors at junction are collinear (cross product ~ 0)
    let p2a_x = vals[b0.params.start + 4];
    let p2a_y = vals[b0.params.start + 5];
    let p1b_x = vals[b1.params.start + 2];
    let p1b_y = vals[b1.params.start + 3];

    let ta_x = p3a_x - p2a_x;
    let ta_y = p3a_y - p2a_y;
    let tb_x = p1b_x - p0b_x;
    let tb_y = p1b_y - p0b_y;
    let cross = ta_x * tb_y - ta_y * tb_x;

    assert!(
        cross.abs() < TOL,
        "G1 tangent not aligned, cross={}",
        cross
    );
}

// =============================================================================
// 6. Constraint Interaction Tests (DOF Analysis)
// =============================================================================

#[test]
fn test_solve_overconstrained_warns() {
    // Build an overconstrained system (DOF < 0).
    // 1 free point (2 DOF) with 3 distance constraints from fixed points.
    let system = ConstraintSystemBuilder::new()
        .name("overconstrained")
        .point_2d_fixed(0.0, 0.0)   // 0
        .point_2d_fixed(10.0, 0.0)  // 1
        .point_2d_fixed(5.0, 10.0)  // 2
        .point_2d(5.0, 3.0)         // 3: free (2 DOF)
        .distance(0, 3, 5.0)
        .distance(1, 3, 5.0)
        .distance(2, 3, 7.0)
        .build();

    // DOF = 2 (variables) - 3 (equations) = -1
    assert!(
        system.degrees_of_freedom() < 0,
        "Expected negative DOF, got {}",
        system.degrees_of_freedom()
    );
    assert!(system.is_overconstrained());
}

#[test]
fn test_solve_underconstrained() {
    // Build an underconstrained system: DOF > 0.
    // One free point with only 1 distance constraint (2 DOF - 1 eq = 1 DOF).
    let mut system = ConstraintSystemBuilder::new()
        .name("underconstrained")
        .point_2d_fixed(0.0, 0.0) // 0
        .point_2d(5.0, 5.0)       // 1: free
        .distance(0, 1, 5.0)
        .build();

    assert!(
        system.degrees_of_freedom() > 0,
        "Expected positive DOF, got {}",
        system.degrees_of_freedom()
    );
    assert!(system.is_underconstrained());

    // LM should still find a solution (one of many) satisfying the constraint.
    let solution = solve_system(&system);
    system.set_values(&solution);

    let d = dist_2d_entities(&system, 0, 1);
    assert!((d - 5.0).abs() < TOL, "Distance = {}, expected 5.0", d);
}

#[test]
fn test_solve_well_constrained() {
    // Verify DOF = 0 for a properly constrained triangle.
    let system = ConstraintSystemBuilder::new()
        .name("well_constrained")
        .point_2d_fixed(0.0, 0.0) // 0
        .point_2d(10.0, 0.0)      // 1: 2 DOF
        .point_2d(5.0, 8.0)       // 2: 2 DOF
        .horizontal(0, 1)         // 1 eq
        .distance(0, 1, 10.0)     // 1 eq
        .distance(1, 2, 8.0)      // 1 eq
        .distance(2, 0, 6.0)      // 1 eq
        .build();

    // DOF = 4 - 4 = 0
    assert_eq!(
        system.degrees_of_freedom(),
        0,
        "Expected DOF=0, got {}",
        system.degrees_of_freedom()
    );
    assert!(system.is_well_constrained());
    assert!(!system.is_underconstrained());
    assert!(!system.is_overconstrained());
}

// =============================================================================
// 7. Stress/Scale Tests
// =============================================================================

#[test]
fn test_solve_chain_10_points() {
    // Chain of 10 points with distance constraints, first fixed.
    let target_dist = 3.0;
    let n = 10;

    let mut builder = ConstraintSystemBuilder::new()
        .name("chain_10")
        .point_2d_fixed(0.0, 0.0); // 0: fixed

    // Add 9 more points along a rough line
    for i in 1..n {
        builder = builder.point_2d((i as f64) * target_dist + 0.5, 0.3 * (i as f64));
    }

    // Add horizontal constraint to pin y-freedom of second point
    builder = builder.horizontal(0, 1);

    // Add distance constraints between consecutive points
    for i in 0..(n - 1) {
        builder = builder.distance(i, i + 1, target_dist);
    }

    let mut system = builder.build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    // Verify all consecutive distances
    for i in 0..(n - 1) {
        let d = dist_2d_entities(&system, i, i + 1);
        assert!(
            (d - target_dist).abs() < TOL,
            "Chain link {} -> {}: dist = {}, expected {}",
            i,
            i + 1,
            d,
            target_dist
        );
    }
}

#[test]
fn test_solve_mixed_constraint_types() {
    // System with distance, horizontal, vertical, and angle constraints on 4+ points.
    let mut system = ConstraintSystemBuilder::new()
        .name("mixed_constraints")
        .point_2d_fixed(0.0, 0.0)  // 0: origin
        .point_2d(5.0, 0.5)        // 1
        .point_2d(5.5, 5.0)        // 2
        .point_2d(0.5, 5.5)        // 3
        .line_2d(0.0, 0.0, 5.0, 0.0) // 4: reference line for angle
        .fix(4)                      // fix reference line
        .horizontal(0, 1)           // p0-p1 horizontal
        .vertical(1, 2)             // p1-p2 vertical
        .distance(0, 1, 5.0)        // |p0-p1| = 5
        .distance(1, 2, 5.0)        // |p1-p2| = 5
        .distance(2, 3, 5.0)        // |p2-p3| = 5
        .vertical(0, 3)             // p0-p3 vertical
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (p0x, p0y) = point_2d_coords(&system, 0);
    let (p1x, p1y) = point_2d_coords(&system, 1);
    let (p2x, _) = point_2d_coords(&system, 2);
    let (p3x, _) = point_2d_coords(&system, 3);

    // Horizontal check: p0.y == p1.y
    assert!((p0y - p1y).abs() < TOL, "p0-p1 not horizontal");
    // Vertical check: p1.x == p2.x
    assert!((p1x - p2x).abs() < TOL, "p1-p2 not vertical");
    // Vertical check: p0.x == p3.x
    assert!((p0x - p3x).abs() < TOL, "p0-p3 not vertical");

    // Distance checks
    assert!((dist_2d_entities(&system, 0, 1) - 5.0).abs() < TOL);
    assert!((dist_2d_entities(&system, 1, 2) - 5.0).abs() < TOL);
    assert!((dist_2d_entities(&system, 2, 3) - 5.0).abs() < TOL);
}

// =============================================================================
// 8. Alternative Solver Tests
// =============================================================================

#[test]
fn test_solve_with_auto_solver() {
    // Verify AutoSolver works for a simple triangle.
    let mut system = ConstraintSystemBuilder::new()
        .name("auto_triangle")
        .point_2d_fixed(0.0, 0.0)
        .point_2d(10.0, 0.5)
        .point_2d(5.0, 8.0)
        .horizontal(0, 1)
        .distance(0, 1, 10.0)
        .distance(1, 2, 8.0)
        .distance(2, 0, 6.0)
        .build();

    let solver = AutoSolver::new();
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(
        result.is_converged() || result.is_completed(),
        "AutoSolver failed: {:?}",
        result
    );

    if let Some(solution) = result.solution() {
        system.set_values(solution);
        let d01 = dist_2d_entities(&system, 0, 1);
        assert!((d01 - 10.0).abs() < TOL, "d01={}", d01);
    }
}

#[test]
fn test_solve_with_robust_solver() {
    // Verify RobustSolver works for a simple triangle.
    let mut system = ConstraintSystemBuilder::new()
        .name("robust_triangle")
        .point_2d_fixed(0.0, 0.0)
        .point_2d(10.0, 0.5)
        .point_2d(5.0, 8.0)
        .horizontal(0, 1)
        .distance(0, 1, 10.0)
        .distance(1, 2, 8.0)
        .distance(2, 0, 6.0)
        .build();

    let solver = RobustSolver::new();
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(
        result.is_converged() || result.is_completed(),
        "RobustSolver failed: {:?}",
        result
    );

    if let Some(solution) = result.solution() {
        system.set_values(solution);
        let d12 = dist_2d_entities(&system, 1, 2);
        assert!((d12 - 8.0).abs() < TOL, "d12={}", d12);
    }
}

// =============================================================================
// 9. Solver with set_values round-trip
// =============================================================================

#[test]
fn test_solve_and_verify_residuals() {
    // After solving, verify all residuals are near zero.
    let mut system = ConstraintSystemBuilder::new()
        .name("residual_check")
        .point_2d_fixed(0.0, 0.0)
        .point_2d(10.0, 0.0)
        .point_2d(5.0, 8.0)
        .horizontal(0, 1)
        .distance(0, 1, 10.0)
        .distance(1, 2, 8.0)
        .distance(2, 0, 6.0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let free_vals = system.current_values();
    let residuals = system.residuals(&free_vals);
    let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);

    assert!(
        max_residual < 1e-6,
        "Residuals not near zero after solve, max={}",
        max_residual
    );
}

#[test]
fn test_solve_point_on_circle_verify_residuals() {
    // After solving a point-on-circle, verify residuals are zero.
    let mut system = ConstraintSystemBuilder::new()
        .name("poc_residual")
        .circle_2d(0.0, 0.0, 5.0) // 0
        .point_2d(3.0, 4.5)       // 1: slightly off circle
        .fix(0)
        .point_on_circle(1, 0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let residuals = system.residuals(&system.current_values());
    let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);

    assert!(
        max_residual < 1e-6,
        "Point-on-circle residuals not near zero, max={}",
        max_residual
    );
}

// =============================================================================
// 10. Entity inspection after solve
// =============================================================================

#[test]
fn test_solve_preserves_fixed_entities() {
    // After solving, verify that fixed entity values did not change.
    let mut system = ConstraintSystemBuilder::new()
        .name("fixed_preserved")
        .point_2d_fixed(1.0, 2.0)  // 0: fixed
        .point_2d(10.0, 10.0)      // 1: free
        .distance(0, 1, 5.0)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (px, py) = point_2d_coords(&system, 0);
    assert!(
        (px - 1.0).abs() < 1e-10,
        "Fixed point x changed to {}",
        px
    );
    assert!(
        (py - 2.0).abs() < 1e-10,
        "Fixed point y changed to {}",
        py
    );
}

#[test]
fn test_solve_circle_center_fixed_radius_free() {
    // Fix only the center of a circle, let radius be free.
    // Add equal_radius with a second circle to set the radius.
    let mut system = ConstraintSystemBuilder::new()
        .name("center_fixed_radius_free")
        .circle_2d(3.0, 4.0, 1.0)   // 0: first circle, radius will adjust
        .circle_2d(10.0, 10.0, 7.0) // 1: second circle (fixed entirely)
        .fix_param_at(0, 0)          // fix cx of circle 0
        .fix_param_at(0, 1)          // fix cy of circle 0
        .fix(1)                       // fix circle 1 entirely
        .equal_radius(0, 1)
        .build();

    let solution = solve_system(&system);
    system.set_values(&solution);

    let (cx, cy, r0) = circle_2d_params(&system, 0);
    let (_, _, r1) = circle_2d_params(&system, 1);

    // Center should not have moved
    assert!((cx - 3.0).abs() < 1e-10, "cx changed");
    assert!((cy - 4.0).abs() < 1e-10, "cy changed");

    // Radii should be equal
    assert!((r0 - r1).abs() < TOL, "r0={}, r1={}", r0, r1);
}
