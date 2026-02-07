//! Edge-case and regression tests for the v2 geometry system.
//!
//! Covers degenerate geometry, near-singular configurations, large-scale values,
//! builder API edge cases, ParameterStore edge cases, ConstraintSystem edge cases,
//! and constraint residual/Jacobian sanity checks.

#![cfg(feature = "geometry")]

use solverang::geometry::constraint::Constraint;
use solverang::geometry::constraints::*;
use solverang::geometry::{
    ConstraintId, ConstraintSystem, ConstraintSystemBuilder, EntityKind,
    ParameterStore, ParamRange,
};
use solverang::{LMConfig, LMSolver, Problem, SolveResult};

use std::f64::consts::PI;

/// Tolerance for residual checks.
const TOL: f64 = 1e-8;

/// Solve a system using LM and return the converged system.
/// Panics if the solver does not converge.
fn solve_system(system: &mut ConstraintSystem) {
    let solver = LMSolver::new(LMConfig::default());
    let initial = system.current_values();
    let result = solver.solve(system, &initial);
    if let SolveResult::Converged { solution, .. } = result {
        system.set_values(&solution);
    } else {
        // Not all edge cases converge; callers that expect convergence should
        // assert themselves.
    }
}

/// Assert that all residuals of a system are near zero at the current state.
#[allow(dead_code)]
fn assert_residuals_near_zero(system: &ConstraintSystem) {
    let vals = system.current_values();
    let residuals = system.residuals(&vals);
    for (i, r) in residuals.iter().enumerate() {
        assert!(
            r.abs() < TOL,
            "Residual {} = {} exceeds tolerance {}",
            i,
            r,
            TOL
        );
    }
}

// =============================================================================
// 1. Degenerate Geometry (15 tests)
// =============================================================================

#[test]
fn test_distance_zero_target() {
    // Distance constraint with target=0 should force coincidence.
    let mut system = ConstraintSystemBuilder::new()
        .name("dist_zero")
        .point_2d_fixed(0.0, 0.0)
        .point_2d(0.1, 0.1)
        .distance(0, 1, 0.0)
        .build();

    solve_system(&mut system);

    let vals = system.params().values();
    let handles = system.handles();
    let h1 = &handles[1];
    let px = vals[h1.params.start];
    let py = vals[h1.params.start + 1];
    assert!(
        px.abs() < 1e-6 && py.abs() < 1e-6,
        "Point should be at origin, got ({}, {})",
        px,
        py
    );
}

#[test]
fn test_distance_same_point() {
    // Distance from a point to itself is zero.
    let p = ParamRange { start: 0, count: 2 };
    let constraint = DistanceConstraint::new(ConstraintId(0), p, p, 0.0);

    let params = vec![3.0, 4.0];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Self-distance residual = {}", residuals[0]);
}

#[test]
fn test_angle_zero() {
    // Angle=0 means horizontal line.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, 0.0);

    // Horizontal line: (0,0) to (5,0)
    let params = vec![0.0, 0.0, 5.0, 0.0];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Angle=0 on horizontal line: residual = {}", residuals[0]);
}

#[test]
fn test_angle_pi() {
    // Angle=pi means reversed horizontal line.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, PI);

    // Line from (0,0) to (-5,0) is at angle pi
    let params = vec![0.0, 0.0, -5.0, 0.0];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Angle=pi residual = {}", residuals[0]);
}

#[test]
fn test_angle_half_pi() {
    // Angle=pi/2 means vertical line.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, PI / 2.0);

    // Vertical line: (0,0) to (0,5)
    let params = vec![0.0, 0.0, 0.0, 5.0];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Angle=pi/2 residual = {}", residuals[0]);
}

#[test]
fn test_parallel_same_direction() {
    // Two lines already parallel (same direction).
    let l1 = ParamRange { start: 0, count: 4 };
    let l2 = ParamRange { start: 4, count: 4 };
    let constraint = ParallelConstraint::new(ConstraintId(0), l1, l2);

    let params = vec![
        0.0, 0.0, 3.0, 3.0,  // line1: direction (3,3)
        1.0, 2.0, 4.0, 5.0,  // line2: direction (3,3) same
    ];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Parallel same-dir residual = {}", residuals[0]);
}

#[test]
fn test_parallel_opposite_direction() {
    // Antiparallel lines should still satisfy parallel constraint.
    let l1 = ParamRange { start: 0, count: 4 };
    let l2 = ParamRange { start: 4, count: 4 };
    let constraint = ParallelConstraint::new(ConstraintId(0), l1, l2);

    let params = vec![
        0.0, 0.0, 1.0, 1.0,   // line1: direction (1,1)
        5.0, 5.0, 4.0, 4.0,   // line2: direction (-1,-1) (antiparallel)
    ];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Antiparallel residual = {}", residuals[0]);
}

#[test]
fn test_perpendicular_already_perpendicular() {
    // Lines already at 90 degrees.
    let l1 = ParamRange { start: 0, count: 4 };
    let l2 = ParamRange { start: 4, count: 4 };
    let constraint = PerpendicularConstraint::new(ConstraintId(0), l1, l2);

    let params = vec![
        0.0, 0.0, 1.0, 0.0,  // horizontal
        3.0, 0.0, 3.0, 1.0,  // vertical
    ];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Perp already-perp residual = {}", residuals[0]);
}

#[test]
fn test_collinear_already_collinear() {
    // Four points already on y=x line.
    let l1 = ParamRange { start: 0, count: 4 };
    let l2 = ParamRange { start: 4, count: 4 };
    let constraint = CollinearConstraint::new(ConstraintId(0), l1, l2);

    let params = vec![
        0.0, 0.0, 2.0, 2.0,  // line1 on y=x
        4.0, 4.0, 6.0, 6.0,  // line2 on y=x
    ];
    let residuals = constraint.residuals(&params);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "Collinear residual[{}] = {}", i, r);
    }
}

#[test]
fn test_coincident_already_coincident() {
    // Two points already at the same location.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

    let params = vec![7.0, 11.0, 7.0, 11.0];
    let residuals = constraint.residuals(&params);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "Coincident residual[{}] = {}", i, r);
    }
}

#[test]
fn test_symmetric_about_origin() {
    // Points at (-3, -4) and (3, 4) symmetric about origin.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let center = ParamRange { start: 4, count: 2 };
    let constraint = SymmetricConstraint::new(ConstraintId(0), p1, p2, center);

    let params = vec![-3.0, -4.0, 3.0, 4.0, 0.0, 0.0];
    let residuals = constraint.residuals(&params);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "Symmetric about origin residual[{}] = {}", i, r);
    }
}

#[test]
fn test_midpoint_already_midpoint() {
    // Point already at the midpoint.
    let mid = ParamRange { start: 0, count: 2 };
    let start = ParamRange { start: 2, count: 2 };
    let end = ParamRange { start: 4, count: 2 };
    let constraint = MidpointConstraint::new(ConstraintId(0), mid, start, end);

    let params = vec![5.0, 5.0, 0.0, 0.0, 10.0, 10.0];
    let residuals = constraint.residuals(&params);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "Midpoint residual[{}] = {}", i, r);
    }
}

#[test]
fn test_point_on_circle_already_on() {
    // Point (3,4) on circle centered at origin with radius 5.
    let point = ParamRange { start: 0, count: 2 };
    let circle = ParamRange { start: 2, count: 3 };
    let constraint = PointOnCircleConstraint::new(ConstraintId(0), point, circle);

    let params = vec![3.0, 4.0, 0.0, 0.0, 5.0];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "PointOnCircle residual = {}", residuals[0]);
}

#[test]
fn test_tangent_already_tangent() {
    // Line y=5 tangent to circle at (0,0) radius 5.
    let line = ParamRange { start: 0, count: 4 };
    let circle = ParamRange { start: 4, count: 3 };
    let constraint = LineTangentCircleConstraint::new(ConstraintId(0), line, circle);

    // Line from (0,5) to (10,5), circle center (0,0) radius 5
    // Perpendicular distance from center to line = 5 = radius
    let params = vec![0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 5.0];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Tangent residual = {}", residuals[0]);
}

#[test]
fn test_concentric_already_concentric() {
    // Two circles with same center.
    let c1 = ParamRange { start: 0, count: 2 };
    let c2 = ParamRange { start: 3, count: 2 };
    let constraint = ConcentricConstraint::new(ConstraintId(0), c1, c2);

    // Circle1 [5, 10, r=3], Circle2 [5, 10, r=7]
    let params = vec![5.0, 10.0, 3.0, 5.0, 10.0, 7.0];
    let residuals = constraint.residuals(&params);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "Concentric residual[{}] = {}", i, r);
    }
}

// =============================================================================
// 2. Near-Singular Configurations (10 tests)
// =============================================================================

#[test]
fn test_distance_nearly_zero() {
    // Distance constraint with a very small target.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let target = 1e-8;
    let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, target);

    // Points at (0,0) and (1e-8, 0)
    let params = vec![0.0, 0.0, target, 0.0];
    let residuals = constraint.residuals(&params);
    assert!(
        residuals[0].abs() < TOL,
        "Near-zero distance residual = {}",
        residuals[0]
    );

    // Jacobian should be finite.
    let jac = constraint.jacobian(&params);
    for (_, _, val) in &jac {
        assert!(val.is_finite(), "Non-finite Jacobian entry: {}", val);
    }
}

#[test]
fn test_angle_near_zero() {
    // AngleConstraint with target very close to zero.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, 1e-10);

    // Nearly horizontal line: (0,0) to (1, 1e-10)
    let params = vec![0.0, 0.0, 1.0, 1e-10];
    let residuals = constraint.residuals(&params);
    assert!(
        residuals[0].abs() < 1e-6,
        "Near-zero angle residual = {}",
        residuals[0]
    );
}

#[test]
fn test_parallel_nearly_parallel() {
    // Lines at a tiny angle from each other (0.001 radian).
    let l1 = ParamRange { start: 0, count: 4 };
    let l2 = ParamRange { start: 4, count: 4 };
    let constraint = ParallelConstraint::new(ConstraintId(0), l1, l2);

    let angle = 0.001_f64;
    let params = vec![
        0.0, 0.0, 10.0, 0.0,                       // horizontal
        0.0, 0.0, 10.0 * angle.cos(), 10.0 * angle.sin(), // nearly horizontal
    ];
    let residuals = constraint.residuals(&params);
    // Residual should be small but non-zero.
    assert!(
        residuals[0].abs() < 0.1,
        "Nearly parallel residual = {}",
        residuals[0]
    );
    assert!(
        residuals[0].abs() > 1e-15,
        "Nearly parallel residual too small: {}",
        residuals[0]
    );
}

#[test]
fn test_point_on_line_nearly_on() {
    // Point very close to the line (1e-10 away).
    let p = ParamRange { start: 0, count: 2 };
    let ls = ParamRange { start: 2, count: 2 };
    let le = ParamRange { start: 4, count: 2 };
    let constraint = PointOnLineConstraint::new(ConstraintId(0), p, ls, le);

    // Horizontal line from (0,0) to (10,0), point at (5, 1e-10)
    let params = vec![5.0, 1e-10, 0.0, 0.0, 10.0, 0.0];
    let residuals = constraint.residuals(&params);
    // Cross product: (5 - 0)*(0 - 0) - (1e-10 - 0)*(10 - 0) = -1e-9
    assert!(
        residuals[0].abs() < 1e-5,
        "Nearly-on-line residual = {}",
        residuals[0]
    );
}

#[test]
fn test_circle_zero_radius() {
    // Degenerate circle with r=0 (a point).
    let mut system = ConstraintSystem::new();
    let circle = system.add_circle_2d(5.0, 5.0, 0.0);

    // Verify the radius parameter is stored as 0.
    let r_val = system.params().get_value(circle.param(2));
    assert!((r_val - 0.0).abs() < TOL, "Zero radius: {}", r_val);

    // PointOnCircle with zero-radius circle: point must be at center.
    let point = system.add_point_2d(5.0, 5.0);
    let id = system.next_constraint_id();
    let constraint = PointOnCircleConstraint::new(id, point.params, circle.params);
    let residuals = constraint.residuals(system.params().values());
    assert!(
        residuals[0].abs() < TOL,
        "Zero-radius point-on-circle residual = {}",
        residuals[0]
    );
}

#[test]
fn test_equal_length_zero_length_lines() {
    // Both line segments have zero length (endpoints coincide).
    let l1 = ParamRange { start: 0, count: 4 };
    let l2 = ParamRange { start: 4, count: 4 };
    let constraint = EqualLengthConstraint::new(ConstraintId(0), l1, l2);

    let params = vec![
        1.0, 2.0, 1.0, 2.0,  // line1: zero length
        5.0, 6.0, 5.0, 6.0,  // line2: zero length
    ];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Zero-length equal-length residual = {}", residuals[0]);

    // Jacobian should be all zeros for zero-length lines.
    let jac = constraint.jacobian(&params);
    for (_, _, val) in &jac {
        assert!(val.is_finite(), "Non-finite Jacobian for zero-length line");
    }
}

#[test]
fn test_bezier_degenerate() {
    // All control points are the same (degenerate point Bezier).
    let b1 = ParamRange { start: 0, count: 8 };
    let b2 = ParamRange { start: 8, count: 8 };
    let constraint = G0ContinuityConstraint::from_beziers(ConstraintId(0), b1, b2);

    // Both beziers are degenerate points at (1,1)
    let params = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // bezier1: all at (1,1)
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // bezier2: all at (1,1)
    ];
    let residuals = constraint.residuals(&params);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "Degenerate bezier G0 residual[{}] = {}", i, r);
    }
}

#[test]
fn test_collinear_very_close_points() {
    // Three nearly coincident points should trivially satisfy collinearity.
    let eps = 1e-12;
    let l1 = ParamRange { start: 0, count: 4 };
    let l2 = ParamRange { start: 4, count: 4 };
    let constraint = CollinearConstraint::new(ConstraintId(0), l1, l2);

    let params = vec![
        0.0, 0.0, eps, eps,     // line1: near-zero length
        2.0 * eps, 2.0 * eps, 3.0 * eps, 3.0 * eps,  // line2: near-zero length on same line
    ];
    let residuals = constraint.residuals(&params);
    for (i, r) in residuals.iter().enumerate() {
        assert!(
            r.abs() < 1e-6,
            "Very close collinear residual[{}] = {}",
            i,
            r
        );
    }
}

#[test]
fn test_distance_coincident_jacobian_safe() {
    // Coincident points with distance constraint: Jacobian should be finite (MIN_EPSILON guard).
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);

    let params = vec![1.0, 1.0, 1.0, 1.0]; // coincident
    let jac = constraint.jacobian(&params);
    for (_, _, val) in &jac {
        assert!(val.is_finite(), "Jacobian should be finite at coincident points, got {}", val);
    }
}

#[test]
fn test_perpendicular_zero_length_lines() {
    // Zero-length lines: dot product is trivially zero.
    let l1 = ParamRange { start: 0, count: 4 };
    let l2 = ParamRange { start: 4, count: 4 };
    let constraint = PerpendicularConstraint::new(ConstraintId(0), l1, l2);

    let params = vec![
        2.0, 3.0, 2.0, 3.0,  // zero-length line1
        7.0, 8.0, 7.0, 8.0,  // zero-length line2
    ];
    let residuals = constraint.residuals(&params);
    assert!(residuals[0].abs() < TOL, "Zero-length perp residual = {}", residuals[0]);
}

// =============================================================================
// 3. Large Scale Values (5 tests)
// =============================================================================

#[test]
fn test_distance_large_coordinates() {
    // Points at very large coordinates.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);

    // Two points at (1e6, 1e6) and (1e6+3, 1e6+4): distance = 5
    let params = vec![1e6, 1e6, 1e6 + 3.0, 1e6 + 4.0];
    let residuals = constraint.residuals(&params);
    assert!(
        residuals[0].abs() < 1e-6,
        "Large coords distance residual = {}",
        residuals[0]
    );
}

#[test]
fn test_distance_mixed_scales() {
    // One point at origin, another far away.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let target = 1e5;
    let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, target);

    let params = vec![0.0, 0.0, 1e5, 0.0];
    let residuals = constraint.residuals(&params);
    assert!(
        residuals[0].abs() < 1e-3,
        "Mixed scale distance residual = {}",
        residuals[0]
    );

    // Jacobian should be well-defined.
    let jac = constraint.jacobian(&params);
    for (_, _, val) in &jac {
        assert!(val.is_finite(), "Mixed-scale Jacobian: {}", val);
    }
}

#[test]
fn test_circle_large_radius() {
    // Circle with a large radius.
    let mut system = ConstraintSystem::new();
    let circle = system.add_circle_2d(0.0, 0.0, 1000.0);
    let point = system.add_point_2d(1000.0, 0.0);

    let id = system.next_constraint_id();
    let constraint = PointOnCircleConstraint::new(id, point.params, circle.params);
    let residuals = constraint.residuals(system.params().values());
    assert!(
        residuals[0].abs() < 1e-3,
        "Large radius point-on-circle residual = {}",
        residuals[0]
    );
}

#[test]
fn test_angle_with_large_line() {
    // Line from (0,0) to (1e4, 1e4) at 45 degrees.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, PI / 4.0);

    let params = vec![0.0, 0.0, 1e4, 1e4];
    let residuals = constraint.residuals(&params);
    assert!(
        residuals[0].abs() < 1e-3,
        "Large line angle residual = {}",
        residuals[0]
    );
}

#[test]
fn test_symmetric_large_separation() {
    // Points separated by 1e6.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let center = ParamRange { start: 4, count: 2 };
    let constraint = SymmetricConstraint::new(ConstraintId(0), p1, p2, center);

    let params = vec![-5e5, -5e5, 5e5, 5e5, 0.0, 0.0];
    let residuals = constraint.residuals(&params);
    for (i, r) in residuals.iter().enumerate() {
        assert!(
            r.abs() < 1e-3,
            "Large symmetric residual[{}] = {}",
            i,
            r
        );
    }
}

// =============================================================================
// 4. Builder API Edge Cases (10 tests)
// =============================================================================

#[test]
fn test_builder_single_point() {
    // Just one free point, no constraints. DOF = 2.
    let system = ConstraintSystemBuilder::new()
        .point_2d(1.0, 2.0)
        .build();

    assert_eq!(system.entity_count(), 1);
    assert_eq!(system.constraint_count(), 0);
    assert_eq!(system.degrees_of_freedom(), 2);
}

#[test]
fn test_builder_single_fixed_point() {
    // One fixed point. DOF = 0.
    let system = ConstraintSystemBuilder::new()
        .point_2d_fixed(1.0, 2.0)
        .build();

    assert_eq!(system.entity_count(), 1);
    assert_eq!(system.constraint_count(), 0);
    assert_eq!(system.degrees_of_freedom(), 0);
}

#[test]
fn test_builder_no_constraints() {
    // Multiple entities, no constraints.
    let system = ConstraintSystemBuilder::new()
        .point_2d(0.0, 0.0)      // 2 params
        .circle_2d(0.0, 0.0, 1.0) // 3 params
        .line_2d(0.0, 0.0, 1.0, 1.0) // 4 params
        .build();

    assert_eq!(system.entity_count(), 3);
    assert_eq!(system.constraint_count(), 0);
    // All 9 params are free, 0 equations.
    assert_eq!(system.degrees_of_freedom(), 9);
}

#[test]
fn test_builder_all_fixed() {
    // All entities fixed. DOF should work correctly.
    let system = ConstraintSystemBuilder::new()
        .point_2d(1.0, 2.0)
        .circle_2d(3.0, 4.0, 5.0)
        .fix(0)
        .fix(1)
        .build();

    assert_eq!(system.entity_count(), 2);
    assert_eq!(system.variable_count(), 0);
    assert_eq!(system.degrees_of_freedom(), 0);
}

#[test]
fn test_builder_mixed_2d_3d() {
    // 2D point and 3D point in the same system.
    let system = ConstraintSystemBuilder::new()
        .point_2d(1.0, 2.0)
        .point_3d(3.0, 4.0, 5.0)
        .build();

    assert_eq!(system.entity_count(), 2);
    let handles = system.handles();
    assert_eq!(handles[0].kind, EntityKind::Point2D);
    assert_eq!(handles[0].params.count, 2);
    assert_eq!(handles[1].kind, EntityKind::Point3D);
    assert_eq!(handles[1].params.count, 3);
    assert_eq!(system.degrees_of_freedom(), 5); // 2 + 3
}

#[test]
fn test_builder_circle_then_point_on() {
    // Circle entity then point_on_circle.
    let system = ConstraintSystemBuilder::new()
        .circle_2d(0.0, 0.0, 5.0)
        .point_2d(3.0, 4.0)
        .point_on_circle(1, 0)
        .build();

    assert_eq!(system.entity_count(), 2);
    assert_eq!(system.constraint_count(), 1);
    // 3 (circle) + 2 (point) = 5 free vars, 1 equation
    assert_eq!(system.degrees_of_freedom(), 4);
}

#[test]
fn test_builder_line_then_angle() {
    // Line entity then angle constraint.
    let system = ConstraintSystemBuilder::new()
        .line_2d(0.0, 0.0, 1.0, 0.0)
        .angle(0, PI / 4.0)
        .build();

    assert_eq!(system.entity_count(), 1);
    assert_eq!(system.constraint_count(), 1);
    // 4 free params, 1 equation => DOF 3
    assert_eq!(system.degrees_of_freedom(), 3);
}

#[test]
fn test_builder_arc_entity() {
    // Arc entity creation and param count.
    let system = ConstraintSystemBuilder::new()
        .arc_2d(5.0, 5.0, 3.0, 0.0, PI / 2.0)
        .build();

    assert_eq!(system.entity_count(), 1);
    let handles = system.handles();
    assert_eq!(handles[0].kind, EntityKind::Arc2D);
    assert_eq!(handles[0].params.count, 5); // cx, cy, r, start_angle, end_angle
}

#[test]
fn test_builder_ellipse_entity() {
    // Ellipse entity creation and param count.
    let system = ConstraintSystemBuilder::new()
        .ellipse_2d(5.0, 5.0, 3.0, 2.0, 0.0)
        .build();

    assert_eq!(system.entity_count(), 1);
    let handles = system.handles();
    assert_eq!(handles[0].kind, EntityKind::Ellipse2D);
    assert_eq!(handles[0].params.count, 5); // cx, cy, rx, ry, rotation
}

#[test]
fn test_builder_cubic_bezier_entity() {
    // Bezier entity creation and param count.
    let system = ConstraintSystemBuilder::new()
        .cubic_bezier_2d([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 2.0],
            [4.0, 0.0],
        ])
        .build();

    assert_eq!(system.entity_count(), 1);
    let handles = system.handles();
    assert_eq!(handles[0].kind, EntityKind::CubicBezier2D);
    assert_eq!(handles[0].params.count, 8); // 4 control points * 2 coords
}

// =============================================================================
// 5. ParameterStore Edge Cases (8 tests)
// =============================================================================

#[test]
fn test_param_store_empty() {
    let store = ParameterStore::new();
    assert_eq!(store.entity_count(), 0);
    assert_eq!(store.param_count(), 0);
    assert_eq!(store.free_param_count(), 0);
    assert!(store.free_indices().is_empty());
    assert!(store.entity_handles().is_empty());
}

#[test]
fn test_param_store_fix_free_cycle() {
    let mut store = ParameterStore::new();
    let p = store.add_point_2d(1.0, 2.0);

    assert_eq!(store.free_param_count(), 2);
    assert!(!store.is_fixed(0));

    // Fix
    store.fix_param(0);
    assert!(store.is_fixed(0));
    assert_eq!(store.free_param_count(), 1);

    // Free again
    store.free_param(0);
    assert!(!store.is_fixed(0));
    assert_eq!(store.free_param_count(), 2);

    // Fix entire entity
    store.fix_entity(&p);
    assert_eq!(store.free_param_count(), 0);

    // Free entire entity
    store.free_entity(&p);
    assert_eq!(store.free_param_count(), 2);
}

#[test]
fn test_param_store_fix_all() {
    let mut store = ParameterStore::new();
    let p1 = store.add_point_2d(1.0, 2.0);
    let p2 = store.add_point_2d(3.0, 4.0);

    store.fix_entity(&p1);
    store.fix_entity(&p2);

    assert_eq!(store.free_param_count(), 0);
    assert!(store.free_indices().is_empty());
    assert!(store.current_free_values().is_empty());
}

#[test]
fn test_param_store_entity_handles_order() {
    let mut store = ParameterStore::new();
    let _p1 = store.add_point_2d(1.0, 2.0);
    let _c = store.add_circle_2d(0.0, 0.0, 1.0);
    let _p2 = store.add_point_3d(3.0, 4.0, 5.0);

    let handles = store.entity_handles();
    assert_eq!(handles.len(), 3);
    // Should be sorted by EntityId (creation order).
    assert!(handles[0].id.0 < handles[1].id.0);
    assert!(handles[1].id.0 < handles[2].id.0);

    assert_eq!(handles[0].kind, EntityKind::Point2D);
    assert_eq!(handles[1].kind, EntityKind::Circle2D);
    assert_eq!(handles[2].kind, EntityKind::Point3D);
}

#[test]
fn test_param_store_scalar() {
    let mut store = ParameterStore::new();
    let s = store.add_scalar(42.0);

    assert_eq!(s.kind, EntityKind::Scalar);
    assert_eq!(s.params.count, 1);
    assert_eq!(store.get_value(s.param(0)), 42.0);
}

#[test]
fn test_param_store_multiple_entities() {
    let mut store = ParameterStore::new();
    let mut handles = Vec::new();

    for i in 0..10 {
        let h = store.add_point_2d(i as f64, i as f64 * 2.0);
        handles.push(h);
    }

    assert_eq!(store.entity_count(), 10);
    assert_eq!(store.param_count(), 20); // 10 * 2

    // Verify each handle indexes correctly.
    for (i, h) in handles.iter().enumerate() {
        assert_eq!(h.params.start, i * 2);
        assert_eq!(store.get_value(h.param(0)), i as f64);
        assert_eq!(store.get_value(h.param(1)), i as f64 * 2.0);
    }
}

#[test]
fn test_param_store_param_values() {
    let mut store = ParameterStore::new();
    let p = store.add_point_2d(1.0, 2.0);

    // Read
    assert_eq!(store.get_value(p.param(0)), 1.0);
    assert_eq!(store.get_value(p.param(1)), 2.0);

    // Write
    store.set_value(p.param(0), 100.0);
    assert_eq!(store.get_value(p.param(0)), 100.0);
    assert_eq!(store.get_value(p.param(1)), 2.0);

    // Write via slice
    store.values_mut()[p.param(1)] = 200.0;
    assert_eq!(store.get_value(p.param(1)), 200.0);
}

#[test]
fn test_param_store_free_indices() {
    let mut store = ParameterStore::new();
    let _p1 = store.add_point_2d(0.0, 0.0); // params 0, 1
    let _p2 = store.add_point_2d(1.0, 1.0); // params 2, 3
    let _p3 = store.add_point_2d(2.0, 2.0); // params 4, 5

    // Fix param 1 and 4
    store.fix_param(1);
    store.fix_param(4);

    let free = store.free_indices();
    assert_eq!(free, vec![0, 2, 3, 5]);

    let free_vals = store.current_free_values();
    assert_eq!(free_vals, vec![0.0, 1.0, 1.0, 2.0]);
}

// =============================================================================
// 6. Constraint System Edge Cases (5 tests)
// =============================================================================

#[test]
fn test_system_dof_empty() {
    let system = ConstraintSystem::new();
    assert_eq!(system.degrees_of_freedom(), 0);
    assert!(system.is_well_constrained());
}

#[test]
fn test_system_dof_points_only() {
    let mut system = ConstraintSystem::new();
    for i in 0..5 {
        system.add_point_2d(i as f64, 0.0);
    }
    // 5 free 2D points = 10 variables, 0 equations => DOF = 10
    assert_eq!(system.degrees_of_freedom(), 10);
    assert!(system.is_underconstrained());
}

#[test]
fn test_system_dof_fully_constrained() {
    // Fixed origin + one point with 2 constraints => DOF 0.
    let system = ConstraintSystemBuilder::new()
        .point_2d_fixed(0.0, 0.0)
        .point_2d(3.0, 4.0)
        .distance(0, 1, 5.0)       // 1 equation
        .horizontal(0, 1)           // 1 equation (same y)
        .build();

    // 2 free vars (p1.x, p1.y), 2 equations => DOF 0
    assert_eq!(system.degrees_of_freedom(), 0);
    assert!(system.is_well_constrained());
}

#[test]
fn test_system_dof_overconstrained() {
    // Fixed origin + one point with 3 constraints => DOF < 0.
    let system = ConstraintSystemBuilder::new()
        .point_2d_fixed(0.0, 0.0)
        .point_2d(3.0, 4.0)
        .distance(0, 1, 5.0)       // 1 equation
        .horizontal(0, 1)           // 1 equation
        .vertical(0, 1)             // 1 equation
        .build();

    // 2 free vars, 3 equations => DOF = -1
    assert_eq!(system.degrees_of_freedom(), -1);
    assert!(system.is_overconstrained());
}

#[test]
fn test_system_problem_trait() {
    // Verify residual_count and variable_count match the Problem trait.
    let system = ConstraintSystemBuilder::new()
        .point_2d_fixed(0.0, 0.0)
        .point_2d(3.0, 4.0)
        .point_2d(6.0, 8.0)
        .distance(0, 1, 5.0)
        .distance(1, 2, 5.0)
        .build();

    // Problem trait
    assert_eq!(system.residual_count(), 2); // 2 distance constraints
    assert_eq!(system.variable_count(), 4); // p1(2) + p2(2)

    // Residuals should be evaluable.
    let x = system.current_values();
    assert_eq!(x.len(), 4);

    let residuals = system.residuals(&x);
    assert_eq!(residuals.len(), 2);

    // Jacobian should have correct dimensions.
    let jac = system.jacobian(&x);
    for (row, col, _) in &jac {
        assert!(*row < system.residual_count());
        assert!(*col < system.variable_count());
    }
}

// =============================================================================
// 7. Constraint Residual Checks (5 tests)
// =============================================================================

#[test]
fn test_residual_satisfied_zero() {
    // When constraint is satisfied, residual should be approximately zero.
    // Use a midpoint constraint as a representative linear constraint.
    let mid = ParamRange { start: 0, count: 2 };
    let start = ParamRange { start: 2, count: 2 };
    let end = ParamRange { start: 4, count: 2 };
    let constraint = MidpointConstraint::new(ConstraintId(0), mid, start, end);

    // Exact midpoint: (5,5) is midpoint of (0,0) and (10,10).
    let params = vec![5.0, 5.0, 0.0, 0.0, 10.0, 10.0];
    let residuals = constraint.residuals(&params);
    for r in &residuals {
        assert!(r.abs() < TOL, "Satisfied midpoint residual = {}", r);
    }
}

#[test]
fn test_residual_unsatisfied_nonzero() {
    // When violated, residual should be proportional to the violation.
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);

    // Actual distance = 10, target = 5, so residual = 10 - 5 = 5
    let params = vec![0.0, 0.0, 10.0, 0.0];
    let residuals = constraint.residuals(&params);
    assert!(
        (residuals[0] - 5.0).abs() < TOL,
        "Distance violation residual = {} (expected 5.0)",
        residuals[0]
    );
}

#[test]
fn test_residual_names() {
    // Every constraint should return a non-empty name.
    let p = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let p3 = ParamRange { start: 4, count: 2 };
    let line = ParamRange { start: 0, count: 4 };
    let line2 = ParamRange { start: 4, count: 4 };
    let circle = ParamRange { start: 0, count: 3 };
    let circle2 = ParamRange { start: 3, count: 3 };
    let bezier = ParamRange { start: 0, count: 8 };
    let bezier2 = ParamRange { start: 8, count: 8 };

    let constraints: Vec<Box<dyn Constraint>> = vec![
        Box::new(DistanceConstraint::new(ConstraintId(0), p, p2, 1.0)),
        Box::new(CoincidentConstraint::new(ConstraintId(1), p, p2)),
        Box::new(FixedConstraint::new(ConstraintId(2), p, vec![0.0, 0.0])),
        Box::new(HorizontalConstraint::new(ConstraintId(3), p, p2)),
        Box::new(VerticalConstraint::new(ConstraintId(4), p, p2)),
        Box::new(MidpointConstraint::new(ConstraintId(5), p, p2, p3)),
        Box::new(SymmetricConstraint::new(ConstraintId(6), p, p2, p3)),
        Box::new(ParallelConstraint::new(ConstraintId(7), line, line2)),
        Box::new(PerpendicularConstraint::new(ConstraintId(8), line, line2)),
        Box::new(CollinearConstraint::new(ConstraintId(9), line, line2)),
        Box::new(EqualLengthConstraint::new(ConstraintId(10), line, line2)),
        Box::new(AngleConstraint::new(ConstraintId(11), p, p2, 0.0)),
        Box::new(PointOnLineConstraint::new(ConstraintId(12), p, p2, p3)),
        Box::new(PointOnCircleConstraint::new(ConstraintId(13), p, circle)),
        Box::new(LineTangentCircleConstraint::new(ConstraintId(14), line, circle)),
        Box::new(CircleTangentConstraint::new(ConstraintId(15), circle, circle2, true)),
        Box::new(EqualRadiusConstraint::new(ConstraintId(16), 2, 5)),
        Box::new(ConcentricConstraint::new(ConstraintId(17), p, p2)),
        Box::new(G0ContinuityConstraint::from_beziers(ConstraintId(18), bezier, bezier2)),
        Box::new(G1ContinuityConstraint::new(ConstraintId(19), bezier, bezier2)),
        Box::new(G2ContinuityConstraint::new(ConstraintId(20), bezier, bezier2)),
    ];

    for c in &constraints {
        let name = c.name();
        assert!(
            !name.is_empty(),
            "Constraint with id {:?} has empty name",
            c.id()
        );
    }
}

#[test]
fn test_equation_counts() {
    // Each constraint should return the correct equation_count.
    let p = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let p3 = ParamRange { start: 4, count: 2 };
    let line = ParamRange { start: 0, count: 4 };
    let line2 = ParamRange { start: 4, count: 4 };
    let circle = ParamRange { start: 0, count: 3 };
    let circle2 = ParamRange { start: 3, count: 3 };
    let bezier = ParamRange { start: 0, count: 8 };
    let bezier2 = ParamRange { start: 8, count: 8 };

    // Constraint -> expected equation count
    let cases: Vec<(Box<dyn Constraint>, usize)> = vec![
        (Box::new(DistanceConstraint::new(ConstraintId(0), p, p2, 1.0)), 1),
        (Box::new(CoincidentConstraint::new(ConstraintId(1), p, p2)), 2), // 2D = 2 eqs
        (Box::new(FixedConstraint::new(ConstraintId(2), p, vec![0.0, 0.0])), 2),
        (Box::new(HorizontalConstraint::new(ConstraintId(3), p, p2)), 1),
        (Box::new(VerticalConstraint::new(ConstraintId(4), p, p2)), 1),
        (Box::new(MidpointConstraint::new(ConstraintId(5), p, p2, p3)), 2),
        (Box::new(SymmetricConstraint::new(ConstraintId(6), p, p2, p3)), 2),
        (Box::new(ParallelConstraint::new(ConstraintId(7), line, line2)), 1), // 2D = 1 eq
        (Box::new(PerpendicularConstraint::new(ConstraintId(8), line, line2)), 1),
        (Box::new(CollinearConstraint::new(ConstraintId(9), line, line2)), 2), // 2D = 2 eqs
        (Box::new(EqualLengthConstraint::new(ConstraintId(10), line, line2)), 1),
        (Box::new(AngleConstraint::new(ConstraintId(11), p, p2, 0.0)), 1),
        (Box::new(PointOnLineConstraint::new(ConstraintId(12), p, p2, p3)), 1), // 2D = 1 eq
        (Box::new(PointOnCircleConstraint::new(ConstraintId(13), p, circle)), 1),
        (Box::new(LineTangentCircleConstraint::new(ConstraintId(14), line, circle)), 1),
        (Box::new(CircleTangentConstraint::new(ConstraintId(15), circle, circle2, true)), 1),
        (Box::new(EqualRadiusConstraint::new(ConstraintId(16), 2, 5)), 1),
        (Box::new(ConcentricConstraint::new(ConstraintId(17), p, p2)), 2), // 2D = 2 eqs
        (Box::new(G0ContinuityConstraint::from_beziers(ConstraintId(18), bezier, bezier2)), 2),
        (Box::new(G1ContinuityConstraint::new(ConstraintId(19), bezier, bezier2)), 3),
        (Box::new(G2ContinuityConstraint::new(ConstraintId(20), bezier, bezier2)), 4),
    ];

    for (constraint, expected) in &cases {
        assert_eq!(
            constraint.equation_count(),
            *expected,
            "Constraint '{}' (id {:?}): expected {} equations, got {}",
            constraint.name(),
            constraint.id(),
            expected,
            constraint.equation_count()
        );
    }
}

#[test]
fn test_dependency_ranges() {
    // Dependencies should be within valid param indices for a realistic system.
    let mut system = ConstraintSystem::new();

    let p1 = system.add_point_2d(0.0, 0.0);
    let p2 = system.add_point_2d(5.0, 5.0);
    let p3 = system.add_point_2d(10.0, 0.0);
    let circle = system.add_circle_2d(5.0, 5.0, 3.0);
    let line = system.add_line_2d(0.0, 0.0, 10.0, 0.0);

    let total_params = system.params().param_count();

    // Build several constraints and check their dependencies.
    let c1 = DistanceConstraint::new(ConstraintId(0), p1.params, p2.params, 5.0);
    let c2 = PointOnCircleConstraint::new(ConstraintId(1), p3.params, circle.params);
    let c3 = PointOnLineConstraint::new(
        ConstraintId(2),
        p1.params,
        ParamRange { start: line.params.start, count: 2 },
        ParamRange { start: line.params.start + 2, count: 2 },
    );

    let constraints: Vec<&dyn Constraint> = vec![&c1, &c2, &c3];

    for c in constraints {
        for &dep in c.dependencies() {
            assert!(
                dep < total_params,
                "Constraint '{}' dependency {} out of range (total params = {})",
                c.name(),
                dep,
                total_params
            );
        }
    }
}

// =============================================================================
// Additional edge-case tests to strengthen coverage
// =============================================================================

#[test]
fn test_equal_radius_with_system() {
    // Test equal radius using the builder API.
    let system = ConstraintSystemBuilder::new()
        .circle_2d(0.0, 0.0, 5.0)
        .circle_2d(10.0, 0.0, 5.0) // same radius
        .equal_radius(0, 1)
        .build();

    let vals = system.current_values();
    let residuals = system.residuals(&vals);
    assert_eq!(residuals.len(), 1);
    assert!(residuals[0].abs() < TOL, "Equal radius residual = {}", residuals[0]);
}

#[test]
fn test_concentric_with_builder() {
    // Concentric circles through builder.
    let system = ConstraintSystemBuilder::new()
        .circle_2d(5.0, 5.0, 3.0)
        .circle_2d(5.0, 5.0, 7.0) // same center, different radius
        .concentric(0, 1)
        .build();

    let vals = system.current_values();
    let residuals = system.residuals(&vals);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "Concentric builder residual[{}] = {}", i, r);
    }
}

#[test]
fn test_fixed_constraint_exact() {
    // FixedConstraint should force exact values.
    let system = ConstraintSystemBuilder::new()
        .point_2d(1.0, 2.0)
        .fixed_at(0, &[1.0, 2.0])
        .build();

    let vals = system.current_values();
    let residuals = system.residuals(&vals);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "Fixed residual[{}] = {}", i, r);
    }
}

#[test]
fn test_horizontal_constraint_satisfied() {
    // Two points with same y should satisfy horizontal.
    let system = ConstraintSystemBuilder::new()
        .point_2d(0.0, 5.0)
        .point_2d(10.0, 5.0)
        .horizontal(0, 1)
        .build();

    let vals = system.current_values();
    let residuals = system.residuals(&vals);
    assert_eq!(residuals.len(), 1);
    assert!(residuals[0].abs() < TOL, "Horizontal residual = {}", residuals[0]);
}

#[test]
fn test_vertical_constraint_satisfied() {
    // Two points with same x should satisfy vertical.
    let system = ConstraintSystemBuilder::new()
        .point_2d(5.0, 0.0)
        .point_2d(5.0, 10.0)
        .vertical(0, 1)
        .build();

    let vals = system.current_values();
    let residuals = system.residuals(&vals);
    assert_eq!(residuals.len(), 1);
    assert!(residuals[0].abs() < TOL, "Vertical residual = {}", residuals[0]);
}

#[test]
fn test_circle_tangent_external_via_builder() {
    // Two externally tangent circles.
    let system = ConstraintSystemBuilder::new()
        .circle_2d(0.0, 0.0, 3.0)
        .circle_2d(7.0, 0.0, 4.0) // dist=7 = 3+4
        .tangent_circles(0, 1, true)
        .build();

    let vals = system.current_values();
    let residuals = system.residuals(&vals);
    assert_eq!(residuals.len(), 1);
    assert!(residuals[0].abs() < TOL, "Tangent circles residual = {}", residuals[0]);
}

#[test]
fn test_g1_continuity_straight_line() {
    // Two beziers forming a straight line should have G1 continuity.
    let system = ConstraintSystemBuilder::new()
        .cubic_bezier_2d([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        .cubic_bezier_2d([
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ])
        .g1_continuity(0, 1)
        .build();

    let vals = system.current_values();
    let residuals = system.residuals(&vals);
    assert_eq!(residuals.len(), 3);
    for (i, r) in residuals.iter().enumerate() {
        assert!(r.abs() < TOL, "G1 straight line residual[{}] = {}", i, r);
    }
}

#[test]
fn test_system_set_and_get_values() {
    // Verify round-trip of set/get values.
    let mut system = ConstraintSystem::new();
    let _p1 = system.add_point_2d(0.0, 0.0);
    let _p2 = system.add_point_2d(1.0, 2.0);

    system.fix_entity(&system.handles()[0]);

    // Free values should be p2's coords.
    let vals = system.current_values();
    assert_eq!(vals, vec![1.0, 2.0]);

    // Set new values.
    system.set_values(&[99.0, 88.0]);
    let new_vals = system.current_values();
    assert_eq!(new_vals, vec![99.0, 88.0]);

    // Fixed point should not change.
    let all_vals = system.params().values();
    assert_eq!(all_vals[0], 0.0);
    assert_eq!(all_vals[1], 0.0);
}

#[test]
fn test_initial_point_matches_current() {
    // Problem::initial_point should return current free values.
    let mut system = ConstraintSystem::new();
    let _p1 = system.add_point_2d(1.0, 2.0);
    system.fix_param(0); // fix x

    let initial = system.initial_point(1.0);
    assert_eq!(initial, vec![2.0]); // only y is free
}

#[test]
fn test_jacobian_column_remapping_with_mixed_fixed() {
    // Verify Jacobian correctly remaps columns when some params are fixed.
    let mut system = ConstraintSystem::new();
    let p1 = system.add_point_2d(0.0, 0.0); // params 0, 1
    let p2 = system.add_point_2d(3.0, 4.0); // params 2, 3

    // Fix p1 entirely.
    system.fix_entity(&p1);

    let id = system.next_constraint_id();
    system.add_constraint(Box::new(
        DistanceConstraint::new(id, p1.params, p2.params, 5.0),
    ));

    let x = system.current_values(); // [3.0, 4.0] (only p2)
    let jac = system.jacobian(&x);

    // Should only have entries for free variables (cols 0 and 1 mapping to p2.x and p2.y).
    assert_eq!(jac.len(), 2);
    for (row, col, val) in &jac {
        assert_eq!(*row, 0);
        assert!(*col < 2, "Column {} out of range for 2 free vars", col);
        assert!(val.is_finite());
    }
}
