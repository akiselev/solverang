//! Geometric constraint tests.
//!
//! These tests verify that the geometric constraints work correctly with the solver,
//! including Jacobian verification against finite differences.

#![cfg(feature = "geometry")]

use solverang::geometry::constraints::*;
use solverang::geometry::params::ParamRange;
use solverang::geometry::{ConstraintSystem, ConstraintSystemBuilder};
use solverang::{verify_jacobian, LMConfig, LMSolver, Problem, SolveResult};

/// Tolerance for convergence tests
const CONVERGENCE_TOL: f64 = 1e-6;

/// Tolerance for Jacobian verification
const JACOBIAN_TOL: f64 = 1e-5;

/// Helper: extract 2D point coordinates from the flat parameter store.
/// Given a system and the entity index (0-based creation order), returns (x, y).
fn point_2d_coords(system: &ConstraintSystem, entity_index: usize) -> (f64, f64) {
    let handles = system.handles();
    let h = &handles[entity_index];
    let vals = system.params().values();
    (vals[h.params.start], vals[h.params.start + 1])
}

/// Helper: compute Euclidean distance between two 2D points extracted from the system.
fn dist_2d(system: &ConstraintSystem, e1: usize, e2: usize) -> f64 {
    let (x1, y1) = point_2d_coords(system, e1);
    let (x2, y2) = point_2d_coords(system, e2);
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

// =============================================================================
// Triangle Tests
// =============================================================================

#[test]
fn test_triangle_solve() {
    // Create a triangle with 3 points and 3 distance constraints
    let mut system = ConstraintSystemBuilder::new()
        .name("Triangle")
        .point_2d_fixed(0.0, 0.0) // p0 - fixed at origin
        .point_2d(10.0, 0.0)      // p1 - initial guess
        .point_2d(5.0, 1.0)       // p2 - initial guess
        .horizontal(0, 1)         // p0-p1 is horizontal
        .distance(0, 1, 10.0)     // |p0-p1| = 10
        .distance(1, 2, 8.0)      // |p1-p2| = 8
        .distance(2, 0, 6.0)      // |p2-p0| = 6
        .build();

    // Should have 4 DOF (p1, p2) - 4 constraints = 0 DOF
    // Actually: p1 has 2 DOF, p2 has 2 DOF = 4 DOF
    // Constraints: horizontal(1) + distance(3) = 4 equations
    // DOF = 4 - 4 = 0
    assert_eq!(system.degrees_of_freedom(), 0);

    // Solve
    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(
        result.is_converged(),
        "Triangle should converge, got {:?}",
        result
    );

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);

        // Verify constraints are satisfied
        let residuals = system.residuals(&system.current_values());
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(
            max_residual < CONVERGENCE_TOL,
            "Max residual {} > tolerance",
            max_residual
        );

        // Verify distances
        let d01 = dist_2d(&system, 0, 1);
        let d12 = dist_2d(&system, 1, 2);
        let d20 = dist_2d(&system, 2, 0);

        assert!((d01 - 10.0).abs() < CONVERGENCE_TOL, "d01 = {}", d01);
        assert!((d12 - 8.0).abs() < CONVERGENCE_TOL, "d12 = {}", d12);
        assert!((d20 - 6.0).abs() < CONVERGENCE_TOL, "d20 = {}", d20);
    }
}

#[test]
fn test_equilateral_triangle() {
    let side = 10.0;
    let height = side * (3.0_f64).sqrt() / 2.0;

    let mut system = ConstraintSystemBuilder::new()
        .point_2d_fixed(0.0, 0.0)                    // p0
        .point_2d_fixed(side, 0.0)                    // p1
        .point_2d(side / 2.0, height + 0.5)          // p2 - perturbed
        .distance(0, 1, side)
        .distance(1, 2, side)
        .distance(2, 0, side)
        .build();

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);
        let (p2x, p2y) = point_2d_coords(&system, 2);

        // p2 should be at apex of equilateral triangle
        assert!((p2x - side / 2.0).abs() < CONVERGENCE_TOL);
        assert!((p2y - height).abs() < CONVERGENCE_TOL);
    }
}

// =============================================================================
// Rectangle Tests
// =============================================================================

#[test]
fn test_rectangle_solve() {
    // Rectangle with horizontal/vertical constraints and equal sides
    let mut system = ConstraintSystemBuilder::new()
        .name("Rectangle")
        .point_2d_fixed(0.0, 0.0) // p0 - bottom-left, fixed
        .point_2d(8.0, 0.5)       // p1 - bottom-right, perturbed
        .point_2d(7.5, 5.0)       // p2 - top-right, perturbed
        .point_2d(0.5, 4.5)       // p3 - top-left, perturbed
        .horizontal(0, 1)         // bottom edge
        .horizontal(3, 2)         // top edge
        .vertical(0, 3)           // left edge
        .vertical(1, 2)           // right edge
        .distance(0, 1, 10.0)     // width = 10
        .distance(0, 3, 5.0)      // height = 5
        .build();

    // DOF: 3 free points * 2 = 6
    // Constraints: 4 h/v + 2 distance = 6
    assert_eq!(system.degrees_of_freedom(), 0);

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);

        let (p0x, p0y) = point_2d_coords(&system, 0);
        let (p1x, p1y) = point_2d_coords(&system, 1);
        let (p2x, p2y) = point_2d_coords(&system, 2);
        let (p3x, p3y) = point_2d_coords(&system, 3);

        // Check rectangle properties
        assert!((p0y - p1y).abs() < CONVERGENCE_TOL); // horizontal bottom
        assert!((p3y - p2y).abs() < CONVERGENCE_TOL); // horizontal top
        assert!((p0x - p3x).abs() < CONVERGENCE_TOL); // vertical left
        assert!((p1x - p2x).abs() < CONVERGENCE_TOL); // vertical right

        // Check dimensions
        assert!((dist_2d(&system, 0, 1) - 10.0).abs() < CONVERGENCE_TOL);
        assert!((dist_2d(&system, 0, 3) - 5.0).abs() < CONVERGENCE_TOL);
    }
}

// =============================================================================
// Jacobian Verification Tests
// =============================================================================

#[test]
fn test_distance_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(0.0, 0.0);
    system.add_point_2d(3.0, 4.0);
    let id = system.next_constraint_id();
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    system.add_constraint(Box::new(DistanceConstraint::new(id, p1, p2, 5.0)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Distance Jacobian failed: max error = {} at {:?}",
        verification.max_absolute_error,
        verification.max_error_location
    );
}

#[test]
fn test_coincident_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(1.0, 2.0);
    system.add_point_2d(3.0, 4.0);
    let id = system.next_constraint_id();
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    system.add_constraint(Box::new(CoincidentConstraint::new(id, p1, p2)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Coincident Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_parallel_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(0.0, 0.0);
    system.add_point_2d(2.0, 1.0);
    system.add_point_2d(5.0, 3.0);
    system.add_point_2d(7.0, 4.0);
    let id = system.next_constraint_id();
    // 4 points: params [0,1], [2,3], [4,5], [6,7]
    // Treat as two segments: p0->p1 and p2->p3
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let p3 = ParamRange { start: 4, count: 2 };
    let p4 = ParamRange { start: 6, count: 2 };
    system.add_constraint(Box::new(ParallelConstraint::from_points(id, p1, p2, p3, p4)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Parallel Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_perpendicular_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(0.0, 0.0);
    system.add_point_2d(1.0, 0.0);
    system.add_point_2d(5.0, 5.0);
    system.add_point_2d(5.0, 6.0);
    let id = system.next_constraint_id();
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let p3 = ParamRange { start: 4, count: 2 };
    let p4 = ParamRange { start: 6, count: 2 };
    system.add_constraint(Box::new(PerpendicularConstraint::from_points(id, p1, p2, p3, p4)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Perpendicular Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_midpoint_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(5.0, 5.0);   // midpoint
    system.add_point_2d(0.0, 0.0);   // start
    system.add_point_2d(10.0, 10.0); // end
    let id = system.next_constraint_id();
    let mid = ParamRange { start: 0, count: 2 };
    let start = ParamRange { start: 2, count: 2 };
    let end = ParamRange { start: 4, count: 2 };
    system.add_constraint(Box::new(MidpointConstraint::new(id, mid, start, end)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Midpoint Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_point_on_line_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(5.0, 5.0);   // point
    system.add_point_2d(0.0, 0.0);   // line start
    system.add_point_2d(10.0, 10.0); // line end
    let id = system.next_constraint_id();
    let point = ParamRange { start: 0, count: 2 };
    let line_start = ParamRange { start: 2, count: 2 };
    let line_end = ParamRange { start: 4, count: 2 };
    system.add_constraint(Box::new(PointOnLineConstraint::new(id, point, line_start, line_end)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "PointOnLine Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_point_on_circle_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(3.0, 4.0);       // point on circle
    system.add_circle_2d(0.0, 0.0, 5.0); // circle: center (0,0), radius 5
    let id = system.next_constraint_id();
    let point = ParamRange { start: 0, count: 2 };
    let circle = ParamRange { start: 2, count: 3 };
    system.add_constraint(Box::new(PointOnCircleConstraint::new(id, point, circle)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "PointOnCircle Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_collinear_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(0.0, 0.0);
    system.add_point_2d(10.0, 10.0);
    system.add_point_2d(3.0, 3.0);
    system.add_point_2d(7.0, 7.0);
    let id = system.next_constraint_id();
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let p3 = ParamRange { start: 4, count: 2 };
    let p4 = ParamRange { start: 6, count: 2 };
    system.add_constraint(Box::new(CollinearConstraint::from_points(id, p1, p2, p3, p4)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Collinear Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_equal_length_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(0.0, 0.0);
    system.add_point_2d(3.0, 4.0);  // len = 5
    system.add_point_2d(10.0, 0.0);
    system.add_point_2d(10.0, 5.0); // len = 5
    let id = system.next_constraint_id();
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let p3 = ParamRange { start: 4, count: 2 };
    let p4 = ParamRange { start: 6, count: 2 };
    system.add_constraint(Box::new(EqualLengthConstraint::from_points(id, p1, p2, p3, p4)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "EqualLength Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_symmetric_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(0.0, 0.0);   // p1
    system.add_point_2d(10.0, 10.0); // p2
    system.add_point_2d(5.0, 5.0);   // center
    let id = system.next_constraint_id();
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let center = ParamRange { start: 4, count: 2 };
    system.add_constraint(Box::new(SymmetricConstraint::new(id, p1, p2, center)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Symmetric Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_angle_constraint_jacobian() {
    let mut system = ConstraintSystem::new();
    system.add_point_2d(0.0, 0.0);
    system.add_point_2d(1.0, 1.0);
    let id = system.next_constraint_id();
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    system.add_constraint(Box::new(AngleConstraint::from_degrees(id, p1, p2, 45.0)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Angle Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

// =============================================================================
// Complex Scenario Tests
// =============================================================================

#[test]
fn test_parallelogram() {
    // Parallelogram: opposite sides parallel and equal
    // Use line_2d entities so parallel/equal_length work on line ParamRanges.
    // Alternatively, use point entities and the from_points constructors.
    // The builder's parallel() and equal_length() expect line entities (4-param),
    // so we use equal_length_segments() for point-based segments and add raw
    // constraints for parallel via from_points.

    let mut system = ConstraintSystemBuilder::new()
        .point_2d_fixed(0.0, 0.0)  // 0: p0
        .point_2d_fixed(10.0, 0.0) // 1: p1
        .point_2d(13.0, 5.0)       // 2: p2
        .point_2d(3.0, 5.0)        // 3: p3
        .build();

    // Add parallel and equal_length constraints using from_points
    let handles = system.handles();
    let pr = |i: usize| handles[i].params;

    let id0 = system.next_constraint_id();
    system.add_constraint(Box::new(
        ParallelConstraint::from_points(id0, pr(0), pr(1), pr(3), pr(2)),
    )); // p0->p1 parallel to p3->p2

    let id1 = system.next_constraint_id();
    system.add_constraint(Box::new(
        ParallelConstraint::from_points(id1, pr(0), pr(3), pr(1), pr(2)),
    )); // p0->p3 parallel to p1->p2

    let id2 = system.next_constraint_id();
    system.add_constraint(Box::new(
        EqualLengthConstraint::from_points(id2, pr(0), pr(1), pr(3), pr(2)),
    )); // |p0-p1| = |p3-p2|

    let id3 = system.next_constraint_id();
    system.add_constraint(Box::new(
        EqualLengthConstraint::from_points(id3, pr(0), pr(3), pr(1), pr(2)),
    )); // |p0-p3| = |p1-p2|

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);
        let residuals = system.residuals(&system.current_values());
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(max_residual < CONVERGENCE_TOL);
    }
}

#[test]
fn test_pentagon_circle_inscribed() {
    // 5 points on a circle
    let radius = 5.0;
    let mut system = ConstraintSystem::new();

    // Center of circle (fixed)
    let center_handle = system.add_circle_2d(0.0, 0.0, radius);
    system.fix_entity(&center_handle);

    // 5 points with initial positions around the circle
    let mut point_handles = Vec::new();
    for i in 0..5 {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / 5.0;
        let x = radius * angle.cos() + 0.1 * (i as f64); // perturbed
        let y = radius * angle.sin() + 0.1 * (i as f64);
        point_handles.push(system.add_point_2d(x, y));
    }

    // Fix the first point to remove rotational freedom
    system.fix_entity(&point_handles[0]);

    // All points on circle
    for ph in &point_handles {
        let id = system.next_constraint_id();
        system.add_constraint(Box::new(PointOnCircleConstraint::new(
            id,
            ph.params,
            center_handle.params,
        )));
    }

    // Equal side lengths
    for i in 0..5 {
        let next = (i + 1) % 5;
        if i > 0 {
            // Equal to first side
            let id = system.next_constraint_id();
            system.add_constraint(Box::new(EqualLengthConstraint::from_points(
                id,
                point_handles[0].params,
                point_handles[1].params,
                point_handles[i].params,
                point_handles[next].params,
            )));
        }
    }

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);

        // Verify all points are on circle
        let vals = system.params().values();
        let cx = vals[center_handle.params.start];
        let cy = vals[center_handle.params.start + 1];
        let r = vals[center_handle.params.start + 2];
        for (idx, ph) in point_handles.iter().enumerate() {
            let px = vals[ph.params.start];
            let py = vals[ph.params.start + 1];
            let dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
            assert!(
                (dist - r).abs() < CONVERGENCE_TOL,
                "Point {} not on circle: dist = {}",
                idx,
                dist
            );
        }
    }
}
