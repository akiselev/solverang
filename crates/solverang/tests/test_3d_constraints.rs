//! 3D-specific geometric constraint tests.

#![cfg(feature = "geometry")]

use solverang::geometry::constraints::*;
use solverang::geometry::entity::EntityKind;
use solverang::geometry::params::EntityHandle;
use solverang::geometry::{ConstraintSystem, ConstraintSystemBuilder};
use solverang::{verify_jacobian, LMConfig, LMSolver, Problem, SolveResult};

/// Tolerance for convergence tests
const CONVERGENCE_TOL: f64 = 1e-6;

/// Tolerance for Jacobian verification
const JACOBIAN_TOL: f64 = 1e-5;

/// Helper: compute Euclidean distance between two 3D points stored as slices.
fn distance_3d(a: &[f64], b: &[f64]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// =============================================================================
// Tetrahedron Tests
// =============================================================================

#[test]
fn test_tetrahedron_solve() {
    // Regular tetrahedron: 4 points, 6 distance constraints
    let edge = 10.0;

    // Initial positions (perturbed from regular tetrahedron)
    let mut system = ConstraintSystemBuilder::new()
        .name("Tetrahedron")
        .point_3d(0.0, 0.0, 0.0)                           // p0 - will be fixed
        .point_3d(edge + 0.5, 0.0, 0.0)                    // p1 - perturbed
        .point_3d(edge / 2.0, edge * 0.8, 0.5)             // p2 - perturbed
        .point_3d(edge / 2.0, edge * 0.3, edge * 0.7)      // p3 - perturbed
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
        let handles = system.handles();
        let p0 = system.params().get_entity_values(&handles[0]).to_vec();
        let p1 = system.params().get_entity_values(&handles[1]).to_vec();
        let p2 = system.params().get_entity_values(&handles[2]).to_vec();
        let p3 = system.params().get_entity_values(&handles[3]).to_vec();

        let edges = [
            (&p0, &p1),
            (&p0, &p2),
            (&p0, &p3),
            (&p1, &p2),
            (&p1, &p3),
            (&p2, &p3),
        ];

        for (i, (a, b)) in edges.iter().enumerate() {
            let dist = distance_3d(a, b);
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

    let mut system = ConstraintSystem::new();

    // Add 8 vertices with perturbed positions
    let h: Vec<EntityHandle> = vec![
        system.add_point_3d(0.0, 0.0, 0.0),
        system.add_point_3d(side + 0.2, 0.0, 0.0),
        system.add_point_3d(side, side + 0.1, 0.0),
        system.add_point_3d(0.0, side, 0.0),
        system.add_point_3d(0.0, 0.0, side + 0.1),
        system.add_point_3d(side, 0.0, side),
        system.add_point_3d(side + 0.2, side, side + 0.1),
        system.add_point_3d(0.0, side + 0.1, side),
    ];

    // Fix the first vertex to remove translation freedom
    system.fix_entity(&h[0]);

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
        let id = system.next_constraint_id();
        system.add_constraint(Box::new(DistanceConstraint::new(
            id,
            h[a].params,
            h[b].params,
            side,
        )));
    }

    // Add perpendicular constraints for adjacent edges
    // Bottom face edges: (0->1) perpendicular to (1->2)
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(PerpendicularConstraint::from_points(
        id,
        h[0].params,
        h[1].params,
        h[1].params,
        h[2].params,
    )));
    // (0->1) perpendicular to (0->4)
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(PerpendicularConstraint::from_points(
        id,
        h[0].params,
        h[1].params,
        h[0].params,
        h[4].params,
    )));

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());

    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);

        // Verify edge lengths
        for (a, b) in edges {
            let pa = system.params().get_entity_values(&h[a]).to_vec();
            let pb = system.params().get_entity_values(&h[b]).to_vec();
            let dist = distance_3d(&pa, &pb);
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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(0.0, 0.0, 0.0);
    let h1 = system.add_point_3d(1.0, 2.0, 2.0);
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(DistanceConstraint::new(
        id,
        h0.params,
        h1.params,
        3.0,
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(1.0, 2.0, 3.0);
    let h1 = system.add_point_3d(4.0, 5.0, 6.0);
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(CoincidentConstraint::new(
        id,
        h0.params,
        h1.params,
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(0.0, 0.0, 0.0);
    let h1 = system.add_point_3d(1.0, 2.0, 3.0);
    let h2 = system.add_point_3d(5.0, 5.0, 5.0);
    let h3 = system.add_point_3d(6.0, 7.0, 8.0);
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(ParallelConstraint::from_points(
        id,
        h0.params,
        h1.params,
        h2.params,
        h3.params,
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(0.0, 0.0, 0.0);
    let h1 = system.add_point_3d(1.0, 0.0, 0.0);
    let h2 = system.add_point_3d(5.0, 5.0, 5.0);
    let h3 = system.add_point_3d(5.0, 6.0, 5.0);
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(PerpendicularConstraint::from_points(
        id,
        h0.params,
        h1.params,
        h2.params,
        h3.params,
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(5.0, 5.0, 5.0);     // point
    let h1 = system.add_point_3d(0.0, 0.0, 0.0);     // line start
    let h2 = system.add_point_3d(10.0, 10.0, 10.0);   // line end
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(PointOnLineConstraint::new(
        id,
        h0.params,
        h1.params,
        h2.params,
    )));

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
    let mut system = ConstraintSystem::new();
    let h_point = system.add_point_3d(3.0, 4.0, 0.0); // point on sphere
    // Model as a Sphere entity [cx, cy, cz, r] with 4 params
    let h_sphere = system.add_entity(EntityKind::Sphere, &[0.0, 0.0, 0.0, 5.0]);
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(PointOnCircleConstraint::new(
        id,
        h_point.params,
        h_sphere.params,
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(0.0, 0.0, 0.0);
    let h1 = system.add_point_3d(10.0, 10.0, 10.0);
    let h2 = system.add_point_3d(3.0, 3.0, 3.0);
    let h3 = system.add_point_3d(7.0, 7.0, 7.0);
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(CollinearConstraint::from_points(
        id,
        h0.params,
        h1.params,
        h2.params,
        h3.params,
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(0.0, 0.0, 0.0);
    let h1 = system.add_point_3d(1.0, 2.0, 2.0);     // len = 3
    let h2 = system.add_point_3d(10.0, 0.0, 0.0);
    let h3 = system.add_point_3d(10.0, 3.0, 0.0);    // len = 3
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(EqualLengthConstraint::from_points(
        id,
        h0.params,
        h1.params,
        h2.params,
        h3.params,
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(0.0, 0.0, 0.0);         // p1
    let h1 = system.add_point_3d(10.0, 10.0, 10.0);       // p2
    let h2 = system.add_point_3d(5.0, 5.0, 5.0);          // center
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(SymmetricConstraint::new(
        id,
        h0.params,
        h1.params,
        h2.params,
    )));

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
    let mut system = ConstraintSystem::new();
    // Model as Sphere entities [cx, cy, cz, r]
    let h_sphere1 = system.add_entity(EntityKind::Sphere, &[0.0, 0.0, 0.0, 3.0]);
    let h_sphere2 = system.add_entity(EntityKind::Sphere, &[8.0, 0.0, 0.0, 5.0]);

    // External tangent: centers at distance r1 + r2 = 3 + 5 = 8
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(CircleTangentConstraint::new(
        id,
        h_sphere1.params,
        h_sphere2.params,
        true, // external
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(0.0, 0.0, 0.0);
    let h1 = system.add_point_3d(1e-11, 1e-11, 1e-11); // Very close to first point
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(DistanceConstraint::new(
        id,
        h0.params,
        h1.params,
        1.0,
    )));

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
    let mut system = ConstraintSystem::new();
    let h0 = system.add_point_3d(0.0, 0.0, 0.0); // will be fixed
    let h1 = system.add_point_3d(5.0, 0.0, 0.0); // free
    let h2 = system.add_point_3d(0.0, 5.0, 0.0); // will be fixed
    let h3 = system.add_point_3d(3.0, 3.0, 3.0); // free

    system.fix_entity(&h0);
    system.fix_entity(&h2);

    // Constraints between mixed points
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(DistanceConstraint::new(
        id,
        h0.params,
        h1.params,
        5.0,
    )));
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(DistanceConstraint::new(
        id,
        h2.params,
        h3.params,
        5.0,
    )));
    let id = system.next_constraint_id();
    system.add_constraint(Box::new(DistanceConstraint::new(
        id,
        h1.params,
        h3.params,
        5.0,
    )));

    // Should have 6 variables (2 free points * 3 coords) and 3 equations
    assert_eq!(system.variable_count(), 6);
    assert_eq!(system.equation_count(), 3);

    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(&system, &initial);

    assert!(result.is_converged());
}
