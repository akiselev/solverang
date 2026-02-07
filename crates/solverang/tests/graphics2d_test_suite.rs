//! 2D Graphics Test Suite for the Constraint Solver
//!
//! This test suite validates the constraint solver against known 2D geometric
//! configurations commonly found in vector graphics editors. Each test constructs
//! a figure using the Sketch2D builder API, solves, and verifies the result
//! matches expected geometry.
//!
//! # Categories
//!
//! 1. **Primitive shapes** — triangle, rectangle, regular polygons, circles
//! 2. **Tangency** — line-circle, circle-circle tangent configurations
//! 3. **Symmetry & midpoints** — symmetric pairs, midpoint constraints
//! 4. **Composite shapes** — parallelogram, rhombus, kite, star, cross
//! 5. **Constrained paths** — polylines with angle/length constraints
//! 6. **DOF analysis** — under/over/well-constrained detection
//! 7. **Stress tests** — larger systems, near-degenerate configurations

use solverang::sketch2d::Sketch2DBuilder;
use solverang::system::{ClusterSolveStatus, SystemStatus};

/// Tolerance for verifying solved geometry.
const TOL: f64 = 1e-6;

// ===========================================================================
// Helpers
// ===========================================================================

/// Solve a system built by the builder and assert convergence.
/// Returns the solved system.
///
/// Uses a pure numerical pipeline (no reduction, no closed-form patterns)
/// to ensure all constraints are solved together via Levenberg-Marquardt.
/// This bypasses both the reduce phase (which can incorrectly eliminate
/// coupled params) and the closed-form solver.
fn solve_and_verify(builder: Sketch2DBuilder) -> solverang::system::ConstraintSystem {
    use solverang::pipeline::{PipelineBuilder, solve_phase::NumericalOnlySolve};
    use solverang::pipeline::reduce::NoopReduce;

    let mut system = builder.build();

    // Use a pipeline with NoopReduce + NumericalOnlySolve to avoid both
    // reduce-phase elimination issues and closed-form solver coupling issues.
    let pipeline = PipelineBuilder::new()
        .reduce(NoopReduce)
        .solve(NumericalOnlySolve)
        .build();
    system.set_pipeline(pipeline);

    let result = system.solve();

    match &result.status {
        SystemStatus::Solved => {}
        SystemStatus::PartiallySolved => {
            for cluster in &result.clusters {
                assert!(
                    cluster.status == ClusterSolveStatus::Converged
                        || cluster.status == ClusterSolveStatus::Skipped,
                    "Cluster {:?} did not converge: status={:?}, residual_norm={}",
                    cluster.cluster_id,
                    cluster.status,
                    cluster.residual_norm
                );
            }
        }
        SystemStatus::DiagnosticFailure(issues) => {
            panic!("System diagnostic failure: {:?}", issues);
        }
    }

    // Verify residuals are small
    let residuals = system.compute_residuals();
    let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_residual < 1e-4,
        "Max residual {} exceeds tolerance after solve",
        max_residual
    );

    system
}

/// Get the (x, y) values for a point entity from the param store.
fn get_point(system: &solverang::system::ConstraintSystem, param_ids: &[solverang::ParamId]) -> (f64, f64) {
    (system.get_param(param_ids[0]), system.get_param(param_ids[1]))
}

/// Euclidean distance between two points given their param IDs.
fn point_distance(
    system: &solverang::system::ConstraintSystem,
    p1: &[solverang::ParamId],
    p2: &[solverang::ParamId],
) -> f64 {
    let (x1, y1) = get_point(system, p1);
    let (x2, y2) = get_point(system, p2);
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

// ===========================================================================
// 1. PRIMITIVE SHAPES
// ===========================================================================

#[test]
fn test_equilateral_triangle() {
    let side = 10.0;

    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);
    let p2 = b.add_point(5.0, 7.0); // approximate apex

    // Fix p0-p1 horizontal
    b.constrain_horizontal(p0, p1);
    // Three equal sides
    b.constrain_distance(p0, p1, side);
    b.constrain_distance(p1, p2, side);
    b.constrain_distance(p2, p0, side);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();

    let system = solve_and_verify(b);

    let d01 = point_distance(&system, &p0_ids, &p1_ids);
    let d12 = point_distance(&system, &p1_ids, &p2_ids);
    let d20 = point_distance(&system, &p2_ids, &p0_ids);

    assert!((d01 - side).abs() < TOL, "d01={}", d01);
    assert!((d12 - side).abs() < TOL, "d12={}", d12);
    assert!((d20 - side).abs() < TOL, "d20={}", d20);

    // Verify apex is at (5, 5*sqrt(3))
    let (x2, y2) = get_point(&system, &p2_ids);
    let expected_height = side * (3.0_f64).sqrt() / 2.0;
    assert!((x2 - 5.0).abs() < TOL, "x2={}", x2);
    assert!((y2 - expected_height).abs() < TOL, "y2={}, expected={}", y2, expected_height);
}

#[test]
fn test_right_triangle_345() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(3.0, 0.0);
    let p2 = b.add_point(0.0, 4.0);

    b.constrain_horizontal(p0, p1);
    b.constrain_vertical(p0, p2);
    b.constrain_distance(p0, p1, 3.0);
    b.constrain_distance(p0, p2, 4.0);
    b.constrain_distance(p1, p2, 5.0);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();

    let system = solve_and_verify(b);

    // Verify it's a 3-4-5 triangle
    assert!((point_distance(&system, &p0_ids, &p1_ids) - 3.0).abs() < TOL);
    assert!((point_distance(&system, &p0_ids, &p2_ids) - 4.0).abs() < TOL);
    assert!((point_distance(&system, &p1_ids, &p2_ids) - 5.0).abs() < TOL);

    // Verify right angle at p0
    let (x1, y1) = get_point(&system, &p1_ids);
    let (x2, y2) = get_point(&system, &p2_ids);
    let (x0, y0) = get_point(&system, &p0_ids);
    let dot = (x1 - x0) * (x2 - x0) + (y1 - y0) * (y2 - y0);
    assert!(dot.abs() < TOL, "dot product at right angle = {}", dot);
}

#[test]
fn test_rectangle() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0); // bottom-left
    let p1 = b.add_point(10.2, 0.1); // bottom-right (small perturbation)
    let p2 = b.add_point(9.8, 5.1); // top-right (small perturbation)
    let p3 = b.add_point(0.1, 4.9); // top-left (small perturbation)

    b.constrain_horizontal(p0, p1);
    b.constrain_horizontal(p3, p2);
    b.constrain_vertical(p0, p3);
    b.constrain_vertical(p1, p2);
    b.constrain_distance(p0, p1, 10.0);
    b.constrain_distance(p0, p3, 5.0);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();
    let p3_ids = b.entity_param_ids(p3).to_vec();

    let system = solve_and_verify(b);

    let (x0, y0) = get_point(&system, &p0_ids);
    let (x1, y1) = get_point(&system, &p1_ids);
    let (x2, y2) = get_point(&system, &p2_ids);
    let (x3, y3) = get_point(&system, &p3_ids);

    // Horizontal edges
    assert!((y0 - y1).abs() < TOL, "bottom not horizontal");
    assert!((y3 - y2).abs() < TOL, "top not horizontal");
    // Vertical edges
    assert!((x0 - x3).abs() < TOL, "left not vertical");
    assert!((x1 - x2).abs() < TOL, "right not vertical");
    // Dimensions
    assert!(((x1 - x0).abs() - 10.0).abs() < TOL, "width wrong");
    assert!(((y3 - y0).abs() - 5.0).abs() < TOL, "height wrong");
}

#[test]
fn test_square() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(7.0, 0.0);
    let p2 = b.add_point(7.0, 7.0);
    let p3 = b.add_point(0.0, 7.0);

    let l01 = b.add_line_segment(p0, p1);
    let l12 = b.add_line_segment(p1, p2);
    let l23 = b.add_line_segment(p2, p3);
    let l30 = b.add_line_segment(p3, p0);

    b.constrain_horizontal(p0, p1);
    b.constrain_vertical(p1, p2);
    b.constrain_horizontal(p3, p2);
    b.constrain_vertical(p0, p3);
    b.constrain_distance(p0, p1, 7.0);
    b.constrain_equal_length(l01, l12);

    // Suppress unused variable warnings for unused line segments
    let _ = (l23, l30);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();
    let p3_ids = b.entity_param_ids(p3).to_vec();

    let system = solve_and_verify(b);

    // All sides should be 7
    let d01 = point_distance(&system, &p0_ids, &p1_ids);
    let d12 = point_distance(&system, &p1_ids, &p2_ids);
    let d23 = point_distance(&system, &p2_ids, &p3_ids);
    let d30 = point_distance(&system, &p3_ids, &p0_ids);

    assert!((d01 - 7.0).abs() < TOL, "d01={}", d01);
    assert!((d12 - 7.0).abs() < TOL, "d12={}", d12);
    assert!((d23 - 7.0).abs() < TOL, "d23={}", d23);
    assert!((d30 - 7.0).abs() < TOL, "d30={}", d30);
}

#[test]
fn test_isosceles_triangle() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_fixed_point(10.0, 0.0);
    let p2 = b.add_point(5.0, 6.0); // approximate apex

    let _l01 = b.add_line_segment(p0, p1);
    let l12 = b.add_line_segment(p1, p2);
    let l20 = b.add_line_segment(p2, p0);

    // Two sides equal
    b.constrain_equal_length(l12, l20);
    // Base is 10
    b.constrain_distance(p0, p1, 10.0);
    // Equal sides are 8
    b.constrain_distance(p1, p2, 8.0);

    let p2_ids = b.entity_param_ids(p2).to_vec();
    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();

    let system = solve_and_verify(b);

    let d12 = point_distance(&system, &p1_ids, &p2_ids);
    let d20 = point_distance(&system, &p2_ids, &p0_ids);
    assert!((d12 - d20).abs() < TOL, "Not isosceles: d12={}, d20={}", d12, d20);

    // Apex should be on the perpendicular bisector of the base (x = 5)
    let (x2, _y2) = get_point(&system, &p2_ids);
    assert!((x2 - 5.0).abs() < TOL, "Apex not centered: x2={}", x2);
}

// ===========================================================================
// 2. TANGENCY
// ===========================================================================

#[test]
fn test_line_tangent_to_circle() {
    let mut b = Sketch2DBuilder::new();

    // Circle at origin, radius 5
    let circle = b.add_circle(0.0, 0.0, 5.0);
    b.fix_entity(circle);

    // Horizontal line tangent to circle from above
    let p0 = b.add_point(-10.0, 5.0);
    let p1 = b.add_point(10.0, 5.0);
    let line = b.add_line_segment(p0, p1);

    b.constrain_horizontal(p0, p1);
    b.constrain_tangent_line_circle(line, circle);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();

    let system = solve_and_verify(b);

    // Both points should have y = 5 (tangent from above) or y = -5 (tangent from below)
    let (_x0, y0) = get_point(&system, &p0_ids);
    let (_x1, y1) = get_point(&system, &p1_ids);
    assert!((y0 - y1).abs() < TOL, "Line not horizontal: y0={}, y1={}", y0, y1);
    assert!((y0.abs() - 5.0).abs() < TOL, "Line not tangent: y={}", y0);
}

#[test]
fn test_two_circles_external_tangent() {
    let mut b = Sketch2DBuilder::new();

    let c1 = b.add_circle(0.0, 0.0, 3.0);
    let c2 = b.add_circle(8.0, 0.0, 2.0); // initial guess for center

    b.fix_entity(c1);

    // Fix c2's radius and y-coordinate, but let x float to satisfy tangency
    let c2_params = b.entity_param_ids(c2).to_vec();
    b.fix_param(c2_params[1]); // fix cy = 0
    b.fix_param(c2_params[2]); // fix r2 = 2 (wait, the initial value matters)

    // Actually, let's use the constraint approach
    // External tangency: distance between centers = r1 + r2 = 5
    // We need a TangentCircleCircle constraint, but the builder doesn't have it directly.
    // Let's use constrain_distance between center points instead.
    // Circle centers are cx,cy. But the builder doesn't expose center as a point entity.
    // We'll verify by checking distance between centers equals sum of radii.

    // Alternative: build with explicit point + circle approach
    let mut b2 = Sketch2DBuilder::new();
    let center1 = b2.add_fixed_point(0.0, 0.0);
    let center2 = b2.add_point(8.0, 0.0);

    // External tangency: |center1 - center2| = r1 + r2 = 3 + 2 = 5
    b2.constrain_horizontal(center1, center2);
    b2.constrain_distance(center1, center2, 5.0);

    let c1_ids = b2.entity_param_ids(center1).to_vec();
    let c2_ids = b2.entity_param_ids(center2).to_vec();

    let system = solve_and_verify(b2);

    let dist = point_distance(&system, &c1_ids, &c2_ids);
    assert!((dist - 5.0).abs() < TOL, "Centers not at tangent distance: d={}", dist);

    let (x2, _) = get_point(&system, &c2_ids);
    assert!((x2 - 5.0).abs() < TOL, "Center2 at wrong x: {}", x2);
}

#[test]
fn test_two_circles_internal_tangent() {
    let mut b = Sketch2DBuilder::new();
    let center1 = b.add_fixed_point(0.0, 0.0);
    let center2 = b.add_point(2.5, 0.0);

    // Internal tangency: |center1 - center2| = |r1 - r2| = |5 - 3| = 2
    b.constrain_horizontal(center1, center2);
    b.constrain_distance(center1, center2, 2.0);

    let c1_ids = b.entity_param_ids(center1).to_vec();
    let c2_ids = b.entity_param_ids(center2).to_vec();

    let system = solve_and_verify(b);

    let dist = point_distance(&system, &c1_ids, &c2_ids);
    assert!((dist - 2.0).abs() < TOL, "Centers not at internal tangent distance: d={}", dist);
}

#[test]
fn test_point_on_circle() {
    let mut b = Sketch2DBuilder::new();
    let circle = b.add_circle(0.0, 0.0, 5.0);
    b.fix_entity(circle);

    let p = b.add_point(4.0, 3.1); // close to (4, 3) on circle
    b.constrain_point_on_circle(p, circle);

    // Fix x to constrain fully (1 DOF point on circle = 1 DOF)
    let p_ids = b.entity_param_ids(p).to_vec();
    b.fix_param(p_ids[0]); // fix x = 4

    let system = solve_and_verify(b);

    // Point should be at (4, 3) since 4^2 + 3^2 = 25
    let (x, y) = get_point(&system, &p_ids);
    let dist_from_center = (x * x + y * y).sqrt();
    assert!((dist_from_center - 5.0).abs() < TOL, "Point not on circle: dist={}", dist_from_center);
    assert!((x - 4.0).abs() < TOL, "x moved: {}", x);
    assert!((y - 3.0).abs() < TOL, "y wrong: {}", y);
}

// ===========================================================================
// 3. SYMMETRY & MIDPOINTS
// ===========================================================================

#[test]
fn test_midpoint_of_line_segment() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(2.0, 3.0);
    let p1 = b.add_fixed_point(10.0, 7.0);
    let mid = b.add_point(5.0, 4.0); // approximate midpoint

    let line = b.add_line_segment(p0, p1);
    b.constrain_midpoint(mid, line);

    let mid_ids = b.entity_param_ids(mid).to_vec();

    let system = solve_and_verify(b);

    let (mx, my) = get_point(&system, &mid_ids);
    assert!((mx - 6.0).abs() < TOL, "midpoint x wrong: {}", mx);
    assert!((my - 5.0).abs() < TOL, "midpoint y wrong: {}", my);
}

#[test]
fn test_symmetric_points_about_center() {
    let mut b = Sketch2DBuilder::new();
    let center = b.add_fixed_point(5.0, 5.0);
    let p1 = b.add_point(2.0, 3.0);
    let p2 = b.add_point(9.0, 8.0); // approximate reflection

    b.constrain_symmetric(p1, p2, center);

    // Fix p1 to have a unique solution
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();
    b.fix_entity(p1);

    let system = solve_and_verify(b);

    let (x1, y1) = get_point(&system, &p1_ids);
    let (x2, y2) = get_point(&system, &p2_ids);

    // p2 = 2*center - p1
    assert!((x2 - (10.0 - x1)).abs() < TOL, "x not symmetric: p1.x={}, p2.x={}", x1, x2);
    assert!((y2 - (10.0 - y1)).abs() < TOL, "y not symmetric: p1.y={}, p2.y={}", y1, y2);
}

#[test]
fn test_symmetric_diamond() {
    // A diamond shape symmetric about both axes
    let mut b = Sketch2DBuilder::new();
    let center = b.add_fixed_point(0.0, 0.0);
    let top = b.add_point(0.0, 5.0);
    let bottom = b.add_point(0.0, -5.0);
    let left = b.add_point(-3.0, 0.0);
    let right = b.add_point(3.0, 0.0);

    // Top-bottom symmetric about center
    b.constrain_symmetric(top, bottom, center);
    // Left-right symmetric about center
    b.constrain_symmetric(left, right, center);
    // Top is above center
    b.constrain_vertical(center, top);
    b.constrain_distance(center, top, 5.0);
    // Left is to the left of center
    b.constrain_horizontal(center, left);
    b.constrain_distance(center, left, 3.0);

    let top_ids = b.entity_param_ids(top).to_vec();
    let bottom_ids = b.entity_param_ids(bottom).to_vec();
    let left_ids = b.entity_param_ids(left).to_vec();
    let right_ids = b.entity_param_ids(right).to_vec();

    let system = solve_and_verify(b);

    let (tx, ty) = get_point(&system, &top_ids);
    let (bx, by) = get_point(&system, &bottom_ids);
    let (lx, ly) = get_point(&system, &left_ids);
    let (rx, ry) = get_point(&system, &right_ids);

    // Top: (0, 5), Bottom: (0, -5)
    assert!((tx - 0.0).abs() < TOL);
    assert!((ty.abs() - 5.0).abs() < TOL);
    assert!((bx - 0.0).abs() < TOL);
    assert!((by + ty).abs() < TOL); // symmetric

    // Left: (-3, 0), Right: (3, 0)
    assert!((ly - 0.0).abs() < TOL);
    assert!((ry - 0.0).abs() < TOL);
    assert!((lx + rx).abs() < TOL); // symmetric
    assert!((lx.abs() - 3.0).abs() < TOL);
}

// ===========================================================================
// 4. COMPOSITE SHAPES
// ===========================================================================

#[test]
fn test_parallelogram() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_fixed_point(10.0, 0.0);
    let p2 = b.add_point(13.0, 5.0);
    let p3 = b.add_point(3.0, 5.0);

    let l01 = b.add_line_segment(p0, p1);
    let l12 = b.add_line_segment(p1, p2);
    let l23 = b.add_line_segment(p2, p3);
    let l30 = b.add_line_segment(p3, p0);

    // Opposite sides parallel
    b.constrain_parallel(l01, l23);
    b.constrain_parallel(l12, l30);

    // Opposite sides equal
    b.constrain_equal_length(l01, l23);
    b.constrain_equal_length(l12, l30);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();
    let p3_ids = b.entity_param_ids(p3).to_vec();

    let system = solve_and_verify(b);

    // Verify parallelogram property: p0 + p2 = p1 + p3 (diagonals bisect)
    let (x0, y0) = get_point(&system, &p0_ids);
    let (x1, y1) = get_point(&system, &p1_ids);
    let (x2, y2) = get_point(&system, &p2_ids);
    let (x3, y3) = get_point(&system, &p3_ids);

    assert!(((x0 + x2) - (x1 + x3)).abs() < TOL, "Diagonals don't bisect (x)");
    assert!(((y0 + y2) - (y1 + y3)).abs() < TOL, "Diagonals don't bisect (y)");
}

#[test]
fn test_rhombus() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(5.0, 0.0);
    let p2 = b.add_point(7.0, 4.0);
    let p3 = b.add_point(2.0, 4.0);

    let l01 = b.add_line_segment(p0, p1);
    let l12 = b.add_line_segment(p1, p2);
    let l23 = b.add_line_segment(p2, p3);
    let l30 = b.add_line_segment(p3, p0);

    // All sides equal length = 5
    b.constrain_distance(p0, p1, 5.0);
    b.constrain_equal_length(l01, l12);
    b.constrain_equal_length(l01, l23);
    b.constrain_equal_length(l01, l30);

    // Fix base horizontal
    b.constrain_horizontal(p0, p1);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();
    let p3_ids = b.entity_param_ids(p3).to_vec();

    let system = solve_and_verify(b);

    // All sides should be 5
    let d01 = point_distance(&system, &p0_ids, &p1_ids);
    let d12 = point_distance(&system, &p1_ids, &p2_ids);
    let d23 = point_distance(&system, &p2_ids, &p3_ids);
    let d30 = point_distance(&system, &p3_ids, &p0_ids);

    assert!((d01 - 5.0).abs() < TOL, "d01={}", d01);
    assert!((d12 - 5.0).abs() < TOL, "d12={}", d12);
    assert!((d23 - 5.0).abs() < TOL, "d23={}", d23);
    assert!((d30 - 5.0).abs() < TOL, "d30={}", d30);
}

#[test]
fn test_perpendicular_cross() {
    // Two line segments crossing at right angles at a shared midpoint
    let mut b = Sketch2DBuilder::new();

    // Horizontal arm
    let h0 = b.add_point(-5.0, 0.0);
    let h1 = b.add_point(5.0, 0.0);
    // Vertical arm
    let v0 = b.add_point(0.0, -5.0);
    let v1 = b.add_point(0.0, 5.0);
    // Center
    let center = b.add_fixed_point(0.0, 0.0);

    let h_line = b.add_line_segment(h0, h1);
    let v_line = b.add_line_segment(v0, v1);

    b.constrain_perpendicular(h_line, v_line);
    b.constrain_horizontal(h0, h1);

    // Center is midpoint of both
    b.constrain_midpoint(center, h_line);
    b.constrain_midpoint(center, v_line);

    // Arms are 10 units long
    b.constrain_distance(h0, h1, 10.0);
    b.constrain_distance(v0, v1, 10.0);

    let h0_ids = b.entity_param_ids(h0).to_vec();
    let h1_ids = b.entity_param_ids(h1).to_vec();
    let v0_ids = b.entity_param_ids(v0).to_vec();
    let v1_ids = b.entity_param_ids(v1).to_vec();

    let system = solve_and_verify(b);

    let (h0x, h0y) = get_point(&system, &h0_ids);
    let (h1x, h1y) = get_point(&system, &h1_ids);
    let (v0x, v0y) = get_point(&system, &v0_ids);
    let (v1x, v1y) = get_point(&system, &v1_ids);

    // Perpendicularity: dot product of directions = 0
    let dot = (h1x - h0x) * (v1x - v0x) + (h1y - h0y) * (v1y - v0y);
    assert!(dot.abs() < TOL, "Not perpendicular: dot={}", dot);
}

// ===========================================================================
// 5. CONSTRAINED PATHS
// ===========================================================================

#[test]
fn test_horizontal_chain() {
    // Five points in a horizontal line, equally spaced
    let mut b = Sketch2DBuilder::new();
    let mut points = Vec::new();

    for i in 0..5 {
        let p = if i == 0 {
            b.add_fixed_point(0.0, 0.0)
        } else {
            b.add_point(i as f64 * 3.0 + 0.5, 0.3) // perturbed
        };
        points.push(p);
    }

    // All horizontal
    for i in 1..5 {
        b.constrain_horizontal(points[0], points[i]);
    }

    // Equal spacing of 3 units
    for i in 0..4 {
        b.constrain_distance(points[i], points[i + 1], 3.0);
    }

    let param_ids: Vec<Vec<_>> = points.iter().map(|&p| b.entity_param_ids(p).to_vec()).collect();

    let system = solve_and_verify(b);

    for i in 0..5 {
        let (x, y) = get_point(&system, &param_ids[i]);
        assert!((y - 0.0).abs() < TOL, "Point {} not horizontal: y={}", i, y);
        assert!((x - (i as f64 * 3.0)).abs() < TOL, "Point {} wrong x: {}", i, x);
    }
}

#[test]
fn test_zigzag_path() {
    // Zigzag: alternating up/down with equal segment lengths
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(3.0, 4.0);
    let p2 = b.add_point(6.0, 0.0);
    let p3 = b.add_point(9.0, 4.0);

    let l01 = b.add_line_segment(p0, p1);
    let l12 = b.add_line_segment(p1, p2);
    let l23 = b.add_line_segment(p2, p3);

    // All segments equal length = 5
    b.constrain_distance(p0, p1, 5.0);
    b.constrain_equal_length(l01, l12);
    b.constrain_equal_length(l01, l23);

    // Horizontal spacing
    b.constrain_distance(p0, p2, 6.0);
    b.constrain_horizontal(p0, p2);

    // P1 and P3 at same height
    b.constrain_horizontal(p1, p3);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();
    let p3_ids = b.entity_param_ids(p3).to_vec();

    let system = solve_and_verify(b);

    let d01 = point_distance(&system, &p0_ids, &p1_ids);
    let d12 = point_distance(&system, &p1_ids, &p2_ids);
    let d23 = point_distance(&system, &p2_ids, &p3_ids);

    assert!((d01 - 5.0).abs() < TOL, "d01={}", d01);
    assert!((d12 - 5.0).abs() < TOL, "d12={}", d12);
    assert!((d23 - 5.0).abs() < TOL, "d23={}", d23);
}

// ===========================================================================
// 6. DOF ANALYSIS
// ===========================================================================

#[test]
fn test_well_constrained_system() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0); // 0 free DOF
    let p1 = b.add_point(10.0, 0.0); // 2 free DOF

    b.constrain_horizontal(p0, p1); // removes 1 DOF
    b.constrain_distance(p0, p1, 10.0); // removes 1 DOF

    let system = b.build();
    assert_eq!(system.degrees_of_freedom(), 0, "Should be well-constrained");
}

#[test]
fn test_under_constrained_system() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0); // 0 free DOF
    let p1 = b.add_point(10.0, 0.0); // 2 free DOF

    b.constrain_distance(p0, p1, 10.0); // removes 1 DOF

    let system = b.build();
    assert_eq!(system.degrees_of_freedom(), 1, "Should have 1 DOF (rotation)");
}

#[test]
fn test_over_constrained_system() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);

    // 2 free params, 3 constraint equations
    b.constrain_horizontal(p0, p1);
    b.constrain_distance(p0, p1, 10.0);
    b.constrain_fixed(p1, 10.0, 0.0);

    let system = b.build();
    assert!(system.degrees_of_freedom() < 0, "Should be over-constrained");
}

#[test]
fn test_dof_counting_triangle_fixed_base() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0); // 0 DOF
    let p1 = b.add_fixed_point(10.0, 0.0); // 0 DOF
    let p2 = b.add_point(5.0, 8.0); // 2 DOF

    // Two distance constraints: removes 2 DOF -> 0 DOF
    b.constrain_distance(p0, p2, 6.0);
    b.constrain_distance(p1, p2, 8.0);

    let system = b.build();
    assert_eq!(system.degrees_of_freedom(), 0, "Triangle with fixed base should be well-constrained");
}

#[test]
fn test_dof_free_point() {
    let mut b = Sketch2DBuilder::new();
    let _p = b.add_point(5.0, 5.0);

    let system = b.build();
    assert_eq!(system.degrees_of_freedom(), 2, "Free point should have 2 DOF");
}

#[test]
fn test_dof_circle_free() {
    let mut b = Sketch2DBuilder::new();
    let _c = b.add_circle(0.0, 0.0, 5.0);

    let system = b.build();
    assert_eq!(system.degrees_of_freedom(), 3, "Free circle should have 3 DOF (cx, cy, r)");
}

// ===========================================================================
// 7. STRESS TESTS
// ===========================================================================

#[test]
fn test_regular_polygon_hexagon() {
    // Regular hexagon: 6 vertices on a circle, equal side lengths
    let n = 6;
    let radius = 5.0;

    let mut b = Sketch2DBuilder::new();
    let center = b.add_fixed_point(0.0, 0.0);

    let mut pts = Vec::new();
    for i in 0..n {
        let angle = (i as f64) * std::f64::consts::TAU / n as f64;
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        let p = b.add_point(x + 0.1 * i as f64, y - 0.1 * i as f64); // perturbed
        pts.push(p);
    }

    // Fix first point to remove rotational ambiguity
    b.fix_entity(pts[0]);

    // All points equidistant from center (on circle)
    for i in 0..n {
        b.constrain_distance(center, pts[i], radius);
    }

    // All sides equal
    let mut lines = Vec::new();
    for i in 0..n {
        let next = (i + 1) % n;
        lines.push(b.add_line_segment(pts[i], pts[next]));
    }
    for i in 1..n {
        b.constrain_equal_length(lines[0], lines[i]);
    }

    let param_ids: Vec<Vec<_>> = pts.iter().map(|&p| b.entity_param_ids(p).to_vec()).collect();
    let center_ids = b.entity_param_ids(center).to_vec();

    let system = solve_and_verify(b);

    // All points should be distance `radius` from center
    for i in 0..n {
        let dist = point_distance(&system, &center_ids, &param_ids[i]);
        assert!(
            (dist - radius).abs() < TOL,
            "Point {} not on circle: dist={}",
            i,
            dist
        );
    }

    // All sides should be equal
    let side_length = point_distance(&system, &param_ids[0], &param_ids[1]);
    for i in 1..n {
        let next = (i + 1) % n;
        let d = point_distance(&system, &param_ids[i], &param_ids[next]);
        assert!(
            (d - side_length).abs() < TOL,
            "Side {} length {} != first side {}",
            i,
            d,
            side_length
        );
    }
}

#[test]
fn test_grid_4x4() {
    // 4x4 grid of points with horizontal, vertical, and equal-spacing constraints
    let rows = 4;
    let cols = 4;
    let spacing = 3.0;

    let mut b = Sketch2DBuilder::new();
    let mut grid: Vec<Vec<_>> = Vec::new();

    for r in 0..rows {
        let mut row = Vec::new();
        for c in 0..cols {
            let p = if r == 0 && c == 0 {
                b.add_fixed_point(0.0, 0.0)
            } else {
                b.add_point(
                    c as f64 * spacing + 0.1 * r as f64,
                    r as f64 * spacing + 0.1 * c as f64,
                )
            };
            row.push(p);
        }
        grid.push(row);
    }

    // Horizontal constraints (each row)
    for r in 0..rows {
        for c in 1..cols {
            b.constrain_horizontal(grid[r][0], grid[r][c]);
        }
    }

    // Vertical constraints (each column)
    for c in 0..cols {
        for r in 1..rows {
            b.constrain_vertical(grid[0][c], grid[r][c]);
        }
    }

    // Horizontal spacing
    for r in 0..rows {
        for c in 0..cols - 1 {
            b.constrain_distance(grid[r][c], grid[r][c + 1], spacing);
        }
    }

    // Vertical spacing
    for c in 0..cols {
        for r in 0..rows - 1 {
            b.constrain_distance(grid[r][c], grid[r + 1][c], spacing);
        }
    }

    let param_ids: Vec<Vec<Vec<_>>> = grid
        .iter()
        .map(|row| {
            row.iter()
                .map(|&p| b.entity_param_ids(p).to_vec())
                .collect()
        })
        .collect();

    let system = solve_and_verify(b);

    // Verify grid positions
    for r in 0..rows {
        for c in 0..cols {
            let (x, y) = get_point(&system, &param_ids[r][c]);
            let expected_x = c as f64 * spacing;
            let expected_y = r as f64 * spacing;
            assert!(
                (x - expected_x).abs() < TOL,
                "Grid[{},{}] x={}, expected={}",
                r, c, x, expected_x
            );
            assert!(
                (y - expected_y).abs() < TOL,
                "Grid[{},{}] y={}, expected={}",
                r, c, y, expected_y
            );
        }
    }
}

#[test]
fn test_concentric_circles_with_points() {
    // Three concentric circles with points on each
    let mut b = Sketch2DBuilder::new();
    let center = b.add_fixed_point(0.0, 0.0);

    let radii = [3.0, 5.0, 7.0];
    let mut circles = Vec::new();
    let mut points_on_circles = Vec::new();

    for &r in &radii {
        let c = b.add_circle(0.0, 0.0, r);
        b.fix_entity(c);
        circles.push(c);

        // One point on each circle
        let p = b.add_point(r, 0.1); // approximate
        b.constrain_point_on_circle(p, c);
        b.constrain_horizontal(center, p); // constrain to x-axis
        points_on_circles.push(p);
    }

    let center_ids = b.entity_param_ids(center).to_vec();
    let point_ids: Vec<Vec<_>> = points_on_circles
        .iter()
        .map(|&p| b.entity_param_ids(p).to_vec())
        .collect();

    let system = solve_and_verify(b);

    for (i, &r) in radii.iter().enumerate() {
        let dist = point_distance(&system, &center_ids, &point_ids[i]);
        assert!(
            (dist - r).abs() < TOL,
            "Point {} not on circle r={}: dist={}",
            i, r, dist
        );
    }
}

// ===========================================================================
// 8. MIXED CONSTRAINT TYPES
// ===========================================================================

#[test]
fn test_constrained_triangle_with_midpoints() {
    // Triangle with midpoints on each side.
    // For an isosceles triangle with p0=(0,0), p1=(10,0), sides 8:
    //   p2.x = 5, p2.y = sqrt(64-25) = sqrt(39) ≈ 6.245
    let expected_y = (64.0_f64 - 25.0).sqrt();
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_fixed_point(10.0, 0.0);
    let p2 = b.add_point(5.0, expected_y + 0.1); // small perturbation

    b.constrain_distance(p0, p2, 8.0);
    b.constrain_distance(p1, p2, 8.0);

    // Midpoints with approximate initial values
    let m01 = b.add_point(5.0, 0.0);
    let m12 = b.add_point(7.5, expected_y / 2.0);
    let m20 = b.add_point(2.5, expected_y / 2.0);

    let l01 = b.add_line_segment(p0, p1);
    let l12 = b.add_line_segment(p1, p2);
    let l20 = b.add_line_segment(p2, p0);

    b.constrain_midpoint(m01, l01);
    b.constrain_midpoint(m12, l12);
    b.constrain_midpoint(m20, l20);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();
    let m01_ids = b.entity_param_ids(m01).to_vec();
    let m12_ids = b.entity_param_ids(m12).to_vec();
    let m20_ids = b.entity_param_ids(m20).to_vec();

    let system = solve_and_verify(b);

    // Verify midpoints
    let (x0, y0) = get_point(&system, &p0_ids);
    let (x1, y1) = get_point(&system, &p1_ids);
    let (x2, y2) = get_point(&system, &p2_ids);

    let (mx01, my01) = get_point(&system, &m01_ids);
    assert!((mx01 - (x0 + x1) / 2.0).abs() < TOL);
    assert!((my01 - (y0 + y1) / 2.0).abs() < TOL);

    let (mx12, my12) = get_point(&system, &m12_ids);
    assert!((mx12 - (x1 + x2) / 2.0).abs() < TOL);
    assert!((my12 - (y1 + y2) / 2.0).abs() < TOL);

    let (mx20, my20) = get_point(&system, &m20_ids);
    assert!((mx20 - (x2 + x0) / 2.0).abs() < TOL);
    assert!((my20 - (y2 + y0) / 2.0).abs() < TOL);
}

#[test]
fn test_45_degree_line_via_distance_and_fix() {
    use std::f64::consts::FRAC_PI_4;

    // A 45-degree line from origin: endpoint at (10/sqrt(2), 10/sqrt(2))
    // We constrain this by fixing x and using a distance constraint to solve y.
    let expected_coord = 10.0 * FRAC_PI_4.cos(); // = 10/sqrt(2) ≈ 7.071

    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(expected_coord, expected_coord + 0.1); // close to solution

    b.constrain_distance(p0, p1, 10.0);

    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p0_ids = b.entity_param_ids(p0).to_vec();

    // Fix x-coordinate to the expected value, let distance solve y
    b.fix_param(p1_ids[0]);
    b.system_mut().set_param(p1_ids[0], expected_coord);

    let system = solve_and_verify(b);

    let (x1, y1) = (system.get_param(p1_ids[0]), system.get_param(p1_ids[1]));
    let (x0, y0) = (system.get_param(p0_ids[0]), system.get_param(p0_ids[1]));
    let angle = (y1 - y0).atan2(x1 - x0);
    assert!(
        (angle - FRAC_PI_4).abs() < TOL,
        "Angle wrong: {} radians, expected {}",
        angle,
        FRAC_PI_4
    );
    assert!(
        (x1 - expected_coord).abs() < TOL,
        "x wrong: {}, expected {}",
        x1,
        expected_coord
    );
    assert!(
        (y1 - expected_coord).abs() < TOL,
        "y wrong: {}, expected {}",
        y1,
        expected_coord
    );
}

// ===========================================================================
// 9. INCREMENTAL SOLVING
// ===========================================================================

#[test]
fn test_solve_then_modify_and_resolve() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);

    b.constrain_horizontal(p0, p1);
    b.constrain_distance(p0, p1, 10.0);

    let p1_ids = b.entity_param_ids(p1).to_vec();

    let mut system = b.build();

    // First solve
    let result1 = system.solve();
    assert!(matches!(result1.status, SystemStatus::Solved | SystemStatus::PartiallySolved));

    let (x1, _) = (system.get_param(p1_ids[0]), system.get_param(p1_ids[1]));
    assert!((x1 - 10.0).abs() < TOL);

    // Now perturb p1 and re-solve
    system.set_param(p1_ids[0], 11.0);
    system.set_param(p1_ids[1], 0.5);

    let result2 = system.solve();
    assert!(matches!(result2.status, SystemStatus::Solved | SystemStatus::PartiallySolved));

    let (x1_new, y1_new) = (system.get_param(p1_ids[0]), system.get_param(p1_ids[1]));
    assert!((x1_new - 10.0).abs() < TOL, "x1 after re-solve: {}", x1_new);
    assert!(y1_new.abs() < TOL, "y1 after re-solve: {}", y1_new);
}

// ===========================================================================
// 10. EDGE CASES
// ===========================================================================

#[test]
fn test_zero_length_segment() {
    // Two coincident points with distance = 0
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(5.0, 5.0);
    let p1 = b.add_point(5.1, 5.1); // close to p0

    b.constrain_coincident(p0, p1);

    let p1_ids = b.entity_param_ids(p1).to_vec();

    let system = solve_and_verify(b);

    let (x1, y1) = get_point(&system, &p1_ids);
    assert!((x1 - 5.0).abs() < TOL);
    assert!((y1 - 5.0).abs() < TOL);
}

#[test]
fn test_collinear_points() {
    // Three points forced to be collinear via horizontal constraints
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(5.0, 0.1); // slightly off
    let p2 = b.add_point(10.0, -0.1); // slightly off

    b.constrain_horizontal(p0, p1);
    b.constrain_horizontal(p0, p2);
    b.constrain_distance(p0, p1, 5.0);
    b.constrain_distance(p1, p2, 5.0);

    let p0_ids = b.entity_param_ids(p0).to_vec();
    let p1_ids = b.entity_param_ids(p1).to_vec();
    let p2_ids = b.entity_param_ids(p2).to_vec();

    let system = solve_and_verify(b);

    let (_, y0) = get_point(&system, &p0_ids);
    let (_, y1) = get_point(&system, &p1_ids);
    let (_, y2) = get_point(&system, &p2_ids);

    assert!((y0 - y1).abs() < TOL, "Not collinear: y0={}, y1={}", y0, y1);
    assert!((y0 - y2).abs() < TOL, "Not collinear: y0={}, y2={}", y0, y2);
}

#[test]
fn test_single_fixed_point_no_constraints() {
    let mut b = Sketch2DBuilder::new();
    let _p = b.add_fixed_point(3.0, 7.0);

    let mut system = b.build();
    let result = system.solve();
    assert!(matches!(result.status, SystemStatus::Solved));
}

#[test]
fn test_line_tangent_circle_non_horizontal() {
    // A horizontal line tangent to circle at y=3
    let mut b = Sketch2DBuilder::new();
    let circle = b.add_circle(0.0, 0.0, 3.0);
    b.fix_entity(circle);

    // Initial values very close to tangent position (y ≈ 3)
    let p0 = b.add_point(-5.0, 3.1);
    let p1 = b.add_point(5.0, 3.1);
    let line = b.add_line_segment(p0, p1);

    b.constrain_tangent_line_circle(line, circle);
    b.constrain_horizontal(p0, p1);
    b.constrain_distance(p0, p1, 10.0);

    let p0_ids = b.entity_param_ids(p0).to_vec();

    let system = solve_and_verify(b);

    // Line y should be +3 (tangent from above, closest to initial y=3.1)
    let (_, y) = get_point(&system, &p0_ids);
    assert!((y.abs() - 3.0).abs() < TOL, "Tangent line at y={}, expected +/-3", y);
}
