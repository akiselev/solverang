//! 3D-specific geometric constraint tests.

#![cfg(feature = "geometry")]

use solverang::geometry::constraints::*;
use solverang::geometry::{ConstraintSystem, ConstraintSystemBuilder, Point3D};
use solverang::{verify_jacobian, LMConfig, LMSolver, Problem, SolveResult};

/// Tolerance for convergence tests
const CONVERGENCE_TOL: f64 = 1e-6;

/// Tolerance for Jacobian verification
const JACOBIAN_TOL: f64 = 1e-5;

// =============================================================================
// Tetrahedron Tests
// =============================================================================

#[test]
fn test_tetrahedron_solve() {
    // Regular tetrahedron: 4 points, 6 distance constraints
    let edge = 10.0;

    // Initial positions (perturbed from regular tetrahedron)
    let mut system = ConstraintSystemBuilder::<3>::new()
        .name("Tetrahedron")
        .point(Point3D::new(0.0, 0.0, 0.0)) // p0 - will be fixed
        .point(Point3D::new(edge + 0.5, 0.0, 0.0)) // p1 - perturbed
        .point(Point3D::new(edge / 2.0, edge * 0.8, 0.5)) // p2 - perturbed
        .point(Point3D::new(edge / 2.0, edge * 0.3, edge * 0.7)) // p3 - perturbed
        .fix(0) // Fix p0
        // All 6 edges have the same length
        .distance(0, 1, edge)
        .distance(0, 2, edge)
        .distance(0, 3, edge)
        .distance(1, 2, edge)
        .distance(1, 3, edge)
        .distance(2, 3, edge)
        .build();

    // 3 free points * 3 coords = 9 DOF
    // 6 constraints
    // DOF = 9 - 6 = 3 (rotational freedom around fixed point)

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(
        result.is_converged(),
        "Tetrahedron should converge, got {:?}",
        result
    );

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);

        // Verify all edge lengths
        let p0 = system.get_point(0).copied().unwrap_or_default();
        let p1 = system.get_point(1).copied().unwrap_or_default();
        let p2 = system.get_point(2).copied().unwrap_or_default();
        let p3 = system.get_point(3).copied().unwrap_or_default();

        let edges = [(p0, p1), (p0, p2), (p0, p3), (p1, p2), (p1, p3), (p2, p3)];

        for (i, (a, b)) in edges.iter().enumerate() {
            let dist = a.distance_to(b);
            assert!(
                (dist - edge).abs() < CONVERGENCE_TOL,
                "Edge {} has length {}, expected {}",
                i,
                dist,
                edge
            );
        }
    }
}

#[test]
fn test_cube_vertices() {
    // Cube with 8 vertices, constraining the edges
    let side = 5.0;

    let mut system = ConstraintSystem::<3>::new();

    // Add 8 vertices with perturbed positions
    let vertices = [
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(side + 0.2, 0.0, 0.0),
        Point3D::new(side, side + 0.1, 0.0),
        Point3D::new(0.0, side, 0.0),
        Point3D::new(0.0, 0.0, side + 0.1),
        Point3D::new(side, 0.0, side),
        Point3D::new(side + 0.2, side, side + 0.1),
        Point3D::new(0.0, side + 0.1, side),
    ];

    for v in &vertices {
        system.add_point(*v);
    }

    // Fix the first vertex to remove translation freedom
    system.fix_point(0);

    // Add distance constraints for all 12 edges
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0), // bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // verticals
    ];

    for (a, b) in edges {
        system.add_constraint(Box::new(DistanceConstraint::<3>::new(a, b, side)));
    }

    // Add perpendicular constraints for adjacent edges
    // Bottom face edges
    system.add_constraint(Box::new(PerpendicularConstraint::<3>::new(0, 1, 1, 2)));
    system.add_constraint(Box::new(PerpendicularConstraint::<3>::new(0, 1, 0, 4)));

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);

        // Verify edge lengths
        for (a, b) in edges {
            let pa = system.get_point(a).copied().unwrap_or_default();
            let pb = system.get_point(b).copied().unwrap_or_default();
            let dist = pa.distance_to(&pb);
            assert!(
                (dist - side).abs() < 0.01, // Slightly relaxed tolerance for complex system
                "Edge ({}, {}) has length {}, expected {}",
                a,
                b,
                dist,
                side
            );
        }
    }
}

// =============================================================================
// 3D Jacobian Verification
// =============================================================================

#[test]
fn test_distance_3d_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(0.0, 0.0, 0.0));
    system.add_point(Point3D::new(1.0, 2.0, 2.0));
    system.add_constraint(Box::new(DistanceConstraint::<3>::new(0, 1, 3.0)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Distance3D Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_coincident_3d_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(1.0, 2.0, 3.0));
    system.add_point(Point3D::new(4.0, 5.0, 6.0));
    system.add_constraint(Box::new(CoincidentConstraint::<3>::new(0, 1)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Coincident3D Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_parallel_3d_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(0.0, 0.0, 0.0));
    system.add_point(Point3D::new(1.0, 2.0, 3.0));
    system.add_point(Point3D::new(5.0, 5.0, 5.0));
    system.add_point(Point3D::new(6.0, 7.0, 8.0));
    system.add_constraint(Box::new(ParallelConstraint::<3>::new(0, 1, 2, 3)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Parallel3D Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_perpendicular_3d_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(0.0, 0.0, 0.0));
    system.add_point(Point3D::new(1.0, 0.0, 0.0));
    system.add_point(Point3D::new(5.0, 5.0, 5.0));
    system.add_point(Point3D::new(5.0, 6.0, 5.0));
    system.add_constraint(Box::new(PerpendicularConstraint::<3>::new(0, 1, 2, 3)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Perpendicular3D Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_point_on_line_3d_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(5.0, 5.0, 5.0)); // point
    system.add_point(Point3D::new(0.0, 0.0, 0.0)); // line start
    system.add_point(Point3D::new(10.0, 10.0, 10.0)); // line end
    system.add_constraint(Box::new(PointOnLineConstraint::<3>::new(0, 1, 2)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "PointOnLine3D Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_point_on_sphere_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(3.0, 4.0, 0.0)); // point on sphere
    system.add_point(Point3D::new(0.0, 0.0, 0.0)); // center
    system.add_constraint(Box::new(PointOnCircleConstraint::<3>::new(0, 1, 5.0)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "PointOnSphere Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_collinear_3d_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(0.0, 0.0, 0.0));
    system.add_point(Point3D::new(10.0, 10.0, 10.0));
    system.add_point(Point3D::new(3.0, 3.0, 3.0));
    system.add_point(Point3D::new(7.0, 7.0, 7.0));
    system.add_constraint(Box::new(CollinearConstraint::<3>::new(0, 1, 2, 3)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Collinear3D Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_equal_length_3d_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(0.0, 0.0, 0.0));
    system.add_point(Point3D::new(1.0, 2.0, 2.0)); // len = 3
    system.add_point(Point3D::new(10.0, 0.0, 0.0));
    system.add_point(Point3D::new(10.0, 3.0, 0.0)); // len = 3
    system.add_constraint(Box::new(EqualLengthConstraint::<3>::new(0, 1, 2, 3)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "EqualLength3D Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_symmetric_3d_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(0.0, 0.0, 0.0)); // p1
    system.add_point(Point3D::new(10.0, 10.0, 10.0)); // p2
    system.add_point(Point3D::new(5.0, 5.0, 5.0)); // center
    system.add_constraint(Box::new(SymmetricConstraint::<3>::new(0, 1, 2)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "Symmetric3D Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

#[test]
fn test_sphere_tangent_jacobian() {
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(0.0, 0.0, 0.0)); // center 1
    system.add_point(Point3D::new(8.0, 0.0, 0.0)); // center 2

    // External tangent: centers at distance r1 + r2 = 3 + 5 = 8
    system.add_constraint(Box::new(CircleTangentConstraint::external(0, 3.0, 1, 5.0)));

    let x = system.current_values();
    let verification = verify_jacobian(&system, &x, 1e-7, JACOBIAN_TOL);

    assert!(
        verification.passed,
        "SphereTangent Jacobian failed: max error = {}",
        verification.max_absolute_error
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_near_zero_length_3d() {
    // Test that constraints handle near-zero distances properly
    let mut system = ConstraintSystem::<3>::new();
    system.add_point(Point3D::new(0.0, 0.0, 0.0));
    system.add_point(Point3D::new(1e-11, 1e-11, 1e-11)); // Very close to first point
    system.add_constraint(Box::new(DistanceConstraint::<3>::new(0, 1, 1.0)));

    let x = system.current_values();

    // Should compute without NaN or inf
    let residuals = system.residuals(&x);
    assert!(residuals[0].is_finite());

    let jac = system.jacobian(&x);
    for (_, _, val) in &jac {
        assert!(val.is_finite(), "Jacobian contains non-finite value");
    }
}

#[test]
fn test_mixed_fixed_free_3d() {
    // Mix of fixed and free points
    let mut system = ConstraintSystem::<3>::new();
    system.add_point_fixed(Point3D::new(0.0, 0.0, 0.0)); // fixed
    system.add_point(Point3D::new(5.0, 0.0, 0.0)); // free
    system.add_point_fixed(Point3D::new(0.0, 5.0, 0.0)); // fixed
    system.add_point(Point3D::new(3.0, 3.0, 3.0)); // free

    // Constraints between mixed points
    system.add_constraint(Box::new(DistanceConstraint::<3>::new(0, 1, 5.0)));
    system.add_constraint(Box::new(DistanceConstraint::<3>::new(2, 3, 5.0)));
    system.add_constraint(Box::new(DistanceConstraint::<3>::new(1, 3, 5.0)));

    // Should have 6 variables (2 free points * 3 coords) and 3 equations
    assert_eq!(system.total_variable_count(), 6);
    assert_eq!(system.equation_count(), 3);

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());
}
