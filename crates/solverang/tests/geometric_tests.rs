//! Geometric constraint tests.
//!
//! These tests verify that the geometric constraints work correctly with the solver,
//! including Jacobian verification against finite differences.

#![cfg(feature = "geometry")]

use solverang::geometry::constraints::*;
use solverang::geometry::{ConstraintSystem, ConstraintSystemBuilder, Point2D};
use solverang::{verify_jacobian, LMConfig, LMSolver, SolveResult};

/// Tolerance for convergence tests
const CONVERGENCE_TOL: f64 = 1e-6;

/// Tolerance for Jacobian verification
const JACOBIAN_TOL: f64 = 1e-5;

// =============================================================================
// Triangle Tests
// =============================================================================

#[test]
fn test_triangle_solve() {
    // Create a triangle with 3 points and 3 distance constraints
    let mut system = ConstraintSystemBuilder::<2>::new()
        .name("Triangle")
        .point(Point2D::new(0.0, 0.0)) // p0 - will be fixed
        .point(Point2D::new(10.0, 0.0)) // p1 - initial guess
        .point(Point2D::new(5.0, 1.0)) // p2 - initial guess
        .fix(0) // Fix p0 at origin
        .horizontal(0, 1) // p0-p1 is horizontal
        .distance(0, 1, 10.0) // |p0-p1| = 10
        .distance(1, 2, 8.0) // |p1-p2| = 8
        .distance(2, 0, 6.0) // |p2-p0| = 6
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
        let residuals = system.evaluate_residuals();
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(
            max_residual < CONVERGENCE_TOL,
            "Max residual {} > tolerance",
            max_residual
        );

        // Verify distances
        let p0 = system.get_point(0).copied().unwrap_or_default();
        let p1 = system.get_point(1).copied().unwrap_or_default();
        let p2 = system.get_point(2).copied().unwrap_or_default();

        let d01 = p0.distance_to(&p1);
        let d12 = p1.distance_to(&p2);
        let d20 = p2.distance_to(&p0);

        assert!((d01 - 10.0).abs() < CONVERGENCE_TOL, "d01 = {}", d01);
        assert!((d12 - 8.0).abs() < CONVERGENCE_TOL, "d12 = {}", d12);
        assert!((d20 - 6.0).abs() < CONVERGENCE_TOL, "d20 = {}", d20);
    }
}

#[test]
fn test_equilateral_triangle() {
    let side = 10.0;
    let height = side * (3.0_f64).sqrt() / 2.0;

    let mut system = ConstraintSystemBuilder::<2>::new()
        .point(Point2D::new(0.0, 0.0)) // p0
        .point(Point2D::new(side, 0.0)) // p1
        .point(Point2D::new(side / 2.0, height + 0.5)) // p2 - perturbed
        .fix(0)
        .fix(1)
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
        let p2 = system.get_point(2).copied().unwrap_or_default();

        // p2 should be at apex of equilateral triangle
        assert!((p2.x() - side / 2.0).abs() < CONVERGENCE_TOL);
        assert!((p2.y() - height).abs() < CONVERGENCE_TOL);
    }
}

// =============================================================================
// Rectangle Tests
// =============================================================================

#[test]
fn test_rectangle_solve() {
    // Rectangle with horizontal/vertical constraints and equal sides
    let mut system = ConstraintSystemBuilder::<2>::new()
        .name("Rectangle")
        .point(Point2D::new(0.0, 0.0)) // p0 - bottom-left, fixed
        .point(Point2D::new(8.0, 0.5)) // p1 - bottom-right, perturbed
        .point(Point2D::new(7.5, 5.0)) // p2 - top-right, perturbed
        .point(Point2D::new(0.5, 4.5)) // p3 - top-left, perturbed
        .fix(0)
        .horizontal(0, 1) // bottom edge
        .horizontal(3, 2) // top edge
        .vertical(0, 3) // left edge
        .vertical(1, 2) // right edge
        .distance(0, 1, 10.0) // width = 10
        .distance(0, 3, 5.0) // height = 5
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

        let p0 = system.get_point(0).copied().unwrap_or_default();
        let p1 = system.get_point(1).copied().unwrap_or_default();
        let p2 = system.get_point(2).copied().unwrap_or_default();
        let p3 = system.get_point(3).copied().unwrap_or_default();

        // Check rectangle properties
        assert!((p0.y() - p1.y()).abs() < CONVERGENCE_TOL); // horizontal bottom
        assert!((p3.y() - p2.y()).abs() < CONVERGENCE_TOL); // horizontal top
        assert!((p0.x() - p3.x()).abs() < CONVERGENCE_TOL); // vertical left
        assert!((p1.x() - p2.x()).abs() < CONVERGENCE_TOL); // vertical right

        // Check dimensions
        assert!((p0.distance_to(&p1) - 10.0).abs() < CONVERGENCE_TOL);
        assert!((p0.distance_to(&p3) - 5.0).abs() < CONVERGENCE_TOL);
    }
}

// =============================================================================
// Jacobian Verification Tests
// =============================================================================

#[test]
fn test_distance_constraint_jacobian() {
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(0.0, 0.0));
    system.add_point(Point2D::new(3.0, 4.0));
    system.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 1, 5.0)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(1.0, 2.0));
    system.add_point(Point2D::new(3.0, 4.0));
    system.add_constraint(Box::new(CoincidentConstraint::<2>::new(0, 1)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(0.0, 0.0));
    system.add_point(Point2D::new(2.0, 1.0));
    system.add_point(Point2D::new(5.0, 3.0));
    system.add_point(Point2D::new(7.0, 4.0));
    system.add_constraint(Box::new(ParallelConstraint::<2>::new(0, 1, 2, 3)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(0.0, 0.0));
    system.add_point(Point2D::new(1.0, 0.0));
    system.add_point(Point2D::new(5.0, 5.0));
    system.add_point(Point2D::new(5.0, 6.0));
    system.add_constraint(Box::new(PerpendicularConstraint::<2>::new(0, 1, 2, 3)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(5.0, 5.0)); // midpoint
    system.add_point(Point2D::new(0.0, 0.0)); // start
    system.add_point(Point2D::new(10.0, 10.0)); // end
    system.add_constraint(Box::new(MidpointConstraint::<2>::new(0, 1, 2)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(5.0, 5.0)); // point
    system.add_point(Point2D::new(0.0, 0.0)); // line start
    system.add_point(Point2D::new(10.0, 10.0)); // line end
    system.add_constraint(Box::new(PointOnLineConstraint::<2>::new(0, 1, 2)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(3.0, 4.0)); // point on circle
    system.add_point(Point2D::new(0.0, 0.0)); // center
    system.add_constraint(Box::new(PointOnCircleConstraint::<2>::new(0, 1, 5.0)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(0.0, 0.0));
    system.add_point(Point2D::new(10.0, 10.0));
    system.add_point(Point2D::new(3.0, 3.0));
    system.add_point(Point2D::new(7.0, 7.0));
    system.add_constraint(Box::new(CollinearConstraint::<2>::new(0, 1, 2, 3)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(0.0, 0.0));
    system.add_point(Point2D::new(3.0, 4.0)); // len = 5
    system.add_point(Point2D::new(10.0, 0.0));
    system.add_point(Point2D::new(10.0, 5.0)); // len = 5
    system.add_constraint(Box::new(EqualLengthConstraint::<2>::new(0, 1, 2, 3)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(0.0, 0.0)); // p1
    system.add_point(Point2D::new(10.0, 10.0)); // p2
    system.add_point(Point2D::new(5.0, 5.0)); // center
    system.add_constraint(Box::new(SymmetricConstraint::<2>::new(0, 1, 2)));

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
    let mut system = ConstraintSystem::<2>::new();
    system.add_point(Point2D::new(0.0, 0.0));
    system.add_point(Point2D::new(1.0, 1.0));
    system.add_constraint(Box::new(AngleConstraint::from_degrees(0, 1, 45.0)));

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
    let mut system = ConstraintSystemBuilder::<2>::new()
        .point(Point2D::new(0.0, 0.0)) // p0
        .point(Point2D::new(10.0, 0.0)) // p1
        .point(Point2D::new(13.0, 5.0)) // p2
        .point(Point2D::new(3.0, 5.0)) // p3
        .fix(0)
        .fix(1)
        .parallel(0, 1, 3, 2) // p0-p1 parallel to p3-p2
        .parallel(0, 3, 1, 2) // p0-p3 parallel to p1-p2
        .equal_length(0, 1, 3, 2) // |p0-p1| = |p3-p2|
        .equal_length(0, 3, 1, 2) // |p0-p3| = |p1-p2|
        .build();

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);
        let residuals = system.evaluate_residuals();
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(max_residual < CONVERGENCE_TOL);
    }
}

#[test]
fn test_pentagon_circle_inscribed() {
    // 5 points on a circle
    let radius = 5.0;
    let mut system = ConstraintSystem::<2>::new();

    // Center of circle (fixed)
    let center = system.add_point_fixed(Point2D::new(0.0, 0.0));

    // 5 points with initial positions around the circle
    let mut point_indices = Vec::new();
    for i in 0..5 {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / 5.0;
        let x = radius * angle.cos() + 0.1 * (i as f64); // perturbed
        let y = radius * angle.sin() + 0.1 * (i as f64);
        point_indices.push(system.add_point(Point2D::new(x, y)));
    }

    // Fix the first point to remove rotational freedom
    system.fix_point(point_indices[0]);

    // All points on circle
    for &idx in &point_indices {
        system.add_constraint(Box::new(PointOnCircleConstraint::<2>::new(idx, center, radius)));
    }

    // Equal side lengths
    for i in 0..5 {
        let next = (i + 1) % 5;
        if i > 0 {
            // Equal to first side
            system.add_constraint(Box::new(EqualLengthConstraint::<2>::new(
                point_indices[0],
                point_indices[1],
                point_indices[i],
                point_indices[next],
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
        let center_pt = system.get_point(center).copied().unwrap_or_default();
        for &idx in &point_indices {
            let pt = system.get_point(idx).copied().unwrap_or_default();
            let dist = center_pt.distance_to(&pt);
            assert!(
                (dist - radius).abs() < CONVERGENCE_TOL,
                "Point {} not on circle: dist = {}",
                idx,
                dist
            );
        }
    }
}
