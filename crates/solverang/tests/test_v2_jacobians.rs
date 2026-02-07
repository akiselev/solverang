//! Comprehensive Jacobian finite-difference verification for all v2 constraints.
//!
//! For every constraint type, constructs it with representative parameters, then
//! verifies every Jacobian entry against central finite differences:
//!     (residuals(params+h) - residuals(params-h)) / (2h)

use solverang::geometry::constraint::Constraint;
use solverang::geometry::params::{ConstraintId, ParamRange};
use solverang::geometry::constraints::*;

const EPSILON: f64 = 1e-7;
const TOLERANCE: f64 = 1e-4;

/// Verify a constraint's Jacobian against finite differences.
/// Checks every (row, col) entry from the analytical Jacobian.
fn verify_constraint_jacobian(
    constraint: &dyn Constraint,
    params: &[f64],
    epsilon: f64,
    tolerance: f64,
) {
    let jac = constraint.jacobian(params);
    let _residuals_base = constraint.residuals(params);

    for &(row, col, analytical) in &jac {
        let mut params_plus = params.to_vec();
        params_plus[col] += epsilon;
        let res_plus = constraint.residuals(&params_plus);

        let mut params_minus = params.to_vec();
        params_minus[col] -= epsilon;
        let res_minus = constraint.residuals(&params_minus);

        let numerical = (res_plus[row] - res_minus[row]) / (2.0 * epsilon);
        let error = (analytical - numerical).abs();

        assert!(
            error < tolerance,
            "Jacobian mismatch for {} at (row={}, col={}): analytical={:.10}, numerical={:.10}, error={:.10}",
            constraint.name(), row, col, analytical, numerical, error
        );
    }

    // Also verify that non-listed entries are approximately zero
    let deps = constraint.dependencies();
    let n_eqs = constraint.equation_count();
    let jac_set: std::collections::HashSet<(usize, usize)> =
        jac.iter().map(|&(r, c, _)| (r, c)).collect();

    for row in 0..n_eqs {
        for &col in deps {
            if !jac_set.contains(&(row, col)) {
                // This entry should be zero -- verify with FD
                let mut params_plus = params.to_vec();
                params_plus[col] += epsilon;
                let res_plus = constraint.residuals(&params_plus);

                let mut params_minus = params.to_vec();
                params_minus[col] -= epsilon;
                let res_minus = constraint.residuals(&params_minus);

                let numerical = (res_plus[row] - res_minus[row]) / (2.0 * epsilon);
                assert!(
                    numerical.abs() < tolerance,
                    "Non-zero unlisted Jacobian for {} at (row={}, col={}): numerical={:.10}",
                    constraint.name(), row, col, numerical
                );
            }
        }
    }
}

// =============================================================================
// 1. DistanceConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_distance_2d() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = DistanceConstraint::new(id, p1, p2, 5.0);

    // Points at (1, 2) and (4, 6), distance = 5
    let params = vec![1.0, 2.0, 4.0, 6.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 2. DistanceConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_distance_3d() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 3 };
    let p2 = ParamRange { start: 3, count: 3 };
    let constraint = DistanceConstraint::new(id, p1, p2, 5.0);

    // Points at (1, 2, 3) and (4, 5, 6)
    let params = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 3. CoincidentConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_coincident_2d() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = CoincidentConstraint::new(id, p1, p2);

    let params = vec![1.5, 2.7, 3.1, -0.3];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 4. CoincidentConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_coincident_3d() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 3 };
    let p2 = ParamRange { start: 3, count: 3 };
    let constraint = CoincidentConstraint::new(id, p1, p2);

    let params = vec![1.5, 2.7, -0.3, 4.1, 0.8, -2.5];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 5. FixedConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_fixed_2d() {
    let id = ConstraintId(0);
    let p = ParamRange { start: 0, count: 2 };
    let constraint = FixedConstraint::new(id, p, vec![3.0, 4.0]);

    let params = vec![1.5, 2.7];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 6. FixedConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_fixed_3d() {
    let id = ConstraintId(0);
    let p = ParamRange { start: 0, count: 3 };
    let constraint = FixedConstraint::new(id, p, vec![1.0, 2.0, 3.0]);

    let params = vec![1.5, 2.7, -0.3];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 7. HorizontalConstraint
// =============================================================================

#[test]
fn test_jacobian_horizontal() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = HorizontalConstraint::new(id, p1, p2);

    let params = vec![1.5, 2.7, 4.1, -0.3];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 8. VerticalConstraint
// =============================================================================

#[test]
fn test_jacobian_vertical() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = VerticalConstraint::new(id, p1, p2);

    let params = vec![1.5, 2.7, 4.1, -0.3];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 9. MidpointConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_midpoint_2d() {
    let id = ConstraintId(0);
    let mid = ParamRange { start: 0, count: 2 };
    let start = ParamRange { start: 2, count: 2 };
    let end = ParamRange { start: 4, count: 2 };
    let constraint = MidpointConstraint::new(id, mid, start, end);

    let params = vec![2.5, 3.1, 1.0, 2.0, 4.0, 4.2];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 10. MidpointConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_midpoint_3d() {
    let id = ConstraintId(0);
    let mid = ParamRange { start: 0, count: 3 };
    let start = ParamRange { start: 3, count: 3 };
    let end = ParamRange { start: 6, count: 3 };
    let constraint = MidpointConstraint::new(id, mid, start, end);

    let params = vec![2.5, 3.1, -0.5, 1.0, 2.0, 0.5, 4.0, 4.2, -1.5];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 11. SymmetricConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_symmetric_2d() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let center = ParamRange { start: 4, count: 2 };
    let constraint = SymmetricConstraint::new(id, p1, p2, center);

    let params = vec![1.5, 2.7, 8.5, 7.3, 5.0, 5.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 12. SymmetricConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_symmetric_3d() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 3 };
    let p2 = ParamRange { start: 3, count: 3 };
    let center = ParamRange { start: 6, count: 3 };
    let constraint = SymmetricConstraint::new(id, p1, p2, center);

    let params = vec![1.5, 2.7, -0.3, 8.5, 7.3, 0.9, 5.0, 5.0, 0.3];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 13. ParallelConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_parallel_2d() {
    let id = ConstraintId(0);
    let line1 = ParamRange { start: 0, count: 4 };
    let line2 = ParamRange { start: 4, count: 4 };
    let constraint = ParallelConstraint::new(id, line1, line2);

    // Two 2D lines: (1,2)-(4,6) and (0,1)-(3,5)
    let params = vec![1.0, 2.0, 4.0, 6.0, 0.0, 1.0, 3.0, 5.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 14. ParallelConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_parallel_3d() {
    let id = ConstraintId(0);
    let line1 = ParamRange { start: 0, count: 6 };
    let line2 = ParamRange { start: 6, count: 6 };
    let constraint = ParallelConstraint::new(id, line1, line2);

    // Two 3D lines
    let params = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,    // line1: (1,2,3)-(4,5,6)
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5,     // line2: (0.5,1.5,2.5)-(3.5,4.5,5.5)
    ];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 15. PerpendicularConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_perpendicular_2d() {
    let id = ConstraintId(0);
    let line1 = ParamRange { start: 0, count: 4 };
    let line2 = ParamRange { start: 4, count: 4 };
    let constraint = PerpendicularConstraint::new(id, line1, line2);

    // Two 2D lines not exactly perpendicular
    let params = vec![1.0, 2.0, 4.0, 6.0, 0.0, 3.0, 5.0, 1.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 16. PerpendicularConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_perpendicular_3d() {
    let id = ConstraintId(0);
    let line1 = ParamRange { start: 0, count: 6 };
    let line2 = ParamRange { start: 6, count: 6 };
    let constraint = PerpendicularConstraint::new(id, line1, line2);

    let params = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,    // line1
        0.5, 1.5, 2.5, 2.5, 0.5, 3.5,     // line2
    ];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 17. CollinearConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_collinear_2d() {
    let id = ConstraintId(0);
    let line1 = ParamRange { start: 0, count: 4 };
    let line2 = ParamRange { start: 4, count: 4 };
    let constraint = CollinearConstraint::new(id, line1, line2);

    // Two 2D line segments (not actually collinear)
    let params = vec![1.0, 2.0, 4.0, 6.0, 2.0, 3.5, 5.0, 7.5];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 18. CollinearConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_collinear_3d() {
    let id = ConstraintId(0);
    let line1 = ParamRange { start: 0, count: 6 };
    let line2 = ParamRange { start: 6, count: 6 };
    let constraint = CollinearConstraint::new(id, line1, line2);

    let params = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 19. EqualLengthConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_equal_length_2d() {
    let id = ConstraintId(0);
    let line1 = ParamRange { start: 0, count: 4 };
    let line2 = ParamRange { start: 4, count: 4 };
    let constraint = EqualLengthConstraint::new(id, line1, line2);

    // Two 2D lines with different lengths
    let params = vec![1.0, 2.0, 4.0, 6.0, 0.0, 1.0, 5.0, 3.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 20. EqualLengthConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_equal_length_3d() {
    let id = ConstraintId(0);
    let line1 = ParamRange { start: 0, count: 6 };
    let line2 = ParamRange { start: 6, count: 6 };
    let constraint = EqualLengthConstraint::new(id, line1, line2);

    let params = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
    ];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 21. EqualLengthConstraint::from_points
// =============================================================================

#[test]
fn test_jacobian_equal_length_from_points() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let p3 = ParamRange { start: 4, count: 2 };
    let p4 = ParamRange { start: 6, count: 2 };
    let constraint = EqualLengthConstraint::from_points(id, p1, p2, p3, p4);

    let params = vec![1.0, 2.0, 4.0, 6.0, 0.0, 1.0, 5.0, 3.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 22. AngleConstraint
// =============================================================================

#[test]
fn test_jacobian_angle() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let constraint = AngleConstraint::new(id, p1, p2, std::f64::consts::FRAC_PI_4);

    // Two 2D points forming a line not at pi/4
    let params = vec![1.5, 2.7, 4.1, 5.3];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 23. PointOnLineConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_point_on_line_2d() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 2 };
    let line_start = ParamRange { start: 2, count: 2 };
    let line_end = ParamRange { start: 4, count: 2 };
    let constraint = PointOnLineConstraint::new(id, point, line_start, line_end);

    // Point (2.5, 3.1), line from (1, 2) to (5, 8)
    let params = vec![2.5, 3.1, 1.0, 2.0, 5.0, 8.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 24. PointOnLineConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_point_on_line_3d() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 3 };
    let line_start = ParamRange { start: 3, count: 3 };
    let line_end = ParamRange { start: 6, count: 3 };
    let constraint = PointOnLineConstraint::new(id, point, line_start, line_end);

    // Point (2.5, 3.1, 1.7), line from (1, 2, 0) to (5, 8, 4)
    let params = vec![2.5, 3.1, 1.7, 1.0, 2.0, 0.0, 5.0, 8.0, 4.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 25. PointOnCircleConstraint
// =============================================================================

#[test]
fn test_jacobian_point_on_circle() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 2 };
    let circle = ParamRange { start: 2, count: 3 };
    let constraint = PointOnCircleConstraint::new(id, point, circle);

    // Point (4.5, 3.2), circle center (1.0, 2.0) radius 4.0
    let params = vec![4.5, 3.2, 1.0, 2.0, 4.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 26. LineTangentCircleConstraint
// =============================================================================

#[test]
fn test_jacobian_tangent_line_circle() {
    let id = ConstraintId(0);
    let line = ParamRange { start: 0, count: 4 };
    let circle = ParamRange { start: 4, count: 3 };
    let constraint = LineTangentCircleConstraint::new(id, line, circle);

    // Line from (0, 0) to (6, 0), circle center (3, 4) radius 3.5
    // (not exactly tangent -- that is fine for Jacobian verification)
    let params = vec![0.0, 0.0, 6.0, 0.0, 3.0, 4.0, 3.5];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 27. CircleTangentConstraint -- external
// =============================================================================

#[test]
fn test_jacobian_tangent_circles_external() {
    let id = ConstraintId(0);
    let circle1 = ParamRange { start: 0, count: 3 };
    let circle2 = ParamRange { start: 3, count: 3 };
    let constraint = CircleTangentConstraint::new(id, circle1, circle2, true);

    // Two circles: (1, 2, r=3) and (8, 5, r=2)
    let params = vec![1.0, 2.0, 3.0, 8.0, 5.0, 2.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 28. CircleTangentConstraint -- internal
// =============================================================================

#[test]
fn test_jacobian_tangent_circles_internal() {
    let id = ConstraintId(0);
    let circle1 = ParamRange { start: 0, count: 3 };
    let circle2 = ParamRange { start: 3, count: 3 };
    let constraint = CircleTangentConstraint::new(id, circle1, circle2, false);

    // Two circles: (1, 2, r=5) and (3, 4, r=2)
    let params = vec![1.0, 2.0, 5.0, 3.0, 4.0, 2.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 29. EqualRadiusConstraint
// =============================================================================

#[test]
fn test_jacobian_equal_radius() {
    let id = ConstraintId(0);
    // Radius at index 2 and index 5 in a param vector
    let constraint = EqualRadiusConstraint::new(id, 2, 5);

    // params: [cx1, cy1, r1, cx2, cy2, r2]
    let params = vec![1.0, 2.0, 3.5, 7.0, 8.0, 4.2];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 30. ConcentricConstraint -- 2D
// =============================================================================

#[test]
fn test_jacobian_concentric_2d() {
    let id = ConstraintId(0);
    let center1 = ParamRange { start: 0, count: 2 };
    let center2 = ParamRange { start: 3, count: 2 };
    let constraint = ConcentricConstraint::new(id, center1, center2);

    // Circle1: [cx=1.5, cy=2.7, r=3], Circle2: [cx=4.1, cy=0.8, r=5]
    let params = vec![1.5, 2.7, 3.0, 4.1, 0.8, 5.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 31. ConcentricConstraint -- 3D
// =============================================================================

#[test]
fn test_jacobian_concentric_3d() {
    let id = ConstraintId(0);
    let center1 = ParamRange { start: 0, count: 3 };
    let center2 = ParamRange { start: 4, count: 3 };
    let constraint = ConcentricConstraint::new(id, center1, center2);

    // Sphere1: [cx=1.5, cy=2.7, cz=-0.3, r=3], Sphere2: [cx=4.1, cy=0.8, cz=2.5, r=5]
    let params = vec![1.5, 2.7, -0.3, 3.0, 4.1, 0.8, 2.5, 5.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 32. PointOnEllipseConstraint
// =============================================================================

#[test]
fn test_jacobian_point_on_ellipse() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 2 };
    let ellipse = ParamRange { start: 2, count: 5 };
    let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

    // Point (2.5, 1.8), ellipse center (0, 0), rx=4, ry=3, rotation=0.3
    let params = vec![2.5, 1.8, 0.0, 0.0, 4.0, 3.0, 0.3];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 33. PointOnBezierConstraint
// =============================================================================

#[test]
fn test_jacobian_point_on_bezier() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 2 };
    let bezier = ParamRange { start: 2, count: 8 };
    let t_param = 10;
    let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

    // Point (0.5, 0.6), Bezier P0..P3, t=0.4
    let mut params = vec![0.0; 11];
    params[0] = 0.5;
    params[1] = 0.6;
    params[2] = 0.0;  // P0.x
    params[3] = 0.0;  // P0.y
    params[4] = 0.2;  // P1.x
    params[5] = 0.8;  // P1.y
    params[6] = 0.8;  // P2.x
    params[7] = 0.9;  // P2.y
    params[8] = 1.0;  // P3.x
    params[9] = 0.1;  // P3.y
    params[10] = 0.4; // t

    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 34. G0ContinuityConstraint
// =============================================================================

#[test]
fn test_jacobian_g0_continuity() {
    let id = ConstraintId(0);
    let curve1_end = ParamRange { start: 0, count: 2 };
    let curve2_start = ParamRange { start: 2, count: 2 };
    let constraint = G0ContinuityConstraint::new(id, curve1_end, curve2_start);

    let params = vec![3.5, 2.1, 4.2, 1.8];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 35. G1ContinuityConstraint
// =============================================================================

#[test]
fn test_jacobian_g1_continuity() {
    let id = ConstraintId(0);
    let bezier1 = ParamRange { start: 0, count: 8 };
    let bezier2 = ParamRange { start: 8, count: 8 };
    let constraint = G1ContinuityConstraint::new(id, bezier1, bezier2);

    // Two Bezier curves with non-trivial control points
    let params = vec![
        0.0, 0.0,   // P0_a
        1.0, 1.5,   // P1_a
        2.5, 0.5,   // P2_a
        4.0, 1.0,   // P3_a
        4.0, 1.0,   // P0_b (= P3_a for G0 satisfaction)
        5.5, 2.0,   // P1_b
        7.0, 0.3,   // P2_b
        9.0, 1.5,   // P3_b
    ];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 36. G2ContinuityConstraint
// =============================================================================

#[test]
fn test_jacobian_g2_continuity() {
    let id = ConstraintId(0);
    let bezier1 = ParamRange { start: 0, count: 8 };
    let bezier2 = ParamRange { start: 8, count: 8 };
    let constraint = G2ContinuityConstraint::new(id, bezier1, bezier2);

    // Two Bezier curves with non-trivial, well-separated control points
    // to avoid singularities in curvature computation
    let params = vec![
        0.0, 0.0,    // P0_a
        1.0, 2.0,    // P1_a
        3.0, 1.5,    // P2_a
        5.0, 2.5,    // P3_a
        5.0, 2.5,    // P0_b (= P3_a)
        7.0, 3.5,    // P1_b
        9.0, 2.0,    // P2_b
        11.0, 3.0,   // P3_b
    ];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 37. ArcEndpointConstraint
// =============================================================================

#[test]
fn test_jacobian_arc_endpoint() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 2 };
    let arc = ParamRange { start: 2, count: 5 };

    // Test at_start = true
    let constraint_start = ArcEndpointConstraint::new(id, point, arc, true);
    // Arc: center (1, 2), radius 3, start_angle = 0.5, end_angle = 2.0
    let params = vec![3.5, 4.1, 1.0, 2.0, 3.0, 0.5, 2.0];
    verify_constraint_jacobian(&constraint_start, &params, EPSILON, TOLERANCE);

    // Test at_start = false
    let constraint_end = ArcEndpointConstraint::new(id, point, arc, false);
    verify_constraint_jacobian(&constraint_end, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 38. ArcSweepConstraint
// =============================================================================

#[test]
fn test_jacobian_arc_sweep() {
    let id = ConstraintId(0);
    let arc = ParamRange { start: 0, count: 5 };
    let constraint = ArcSweepConstraint::new(id, arc, std::f64::consts::FRAC_PI_2);

    // Arc: center (1, 2), radius 3, start_angle = 0.5, end_angle = 2.0
    let params = vec![1.0, 2.0, 3.0, 0.5, 2.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 39. PointOnArcConstraint
// =============================================================================

#[test]
fn test_jacobian_point_on_arc() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 2 };
    let arc = ParamRange { start: 2, count: 5 };
    let constraint = PointOnArcConstraint::new(id, point, arc);

    // Point (4.5, 3.2), arc center (1, 2), radius 4, angles 0.3..1.8
    let params = vec![4.5, 3.2, 1.0, 2.0, 4.0, 0.3, 1.8];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 40. FixedParamConstraint
// =============================================================================

#[test]
fn test_jacobian_fixed_param() {
    let id = ConstraintId(0);
    let constraint = FixedParamConstraint::new(id, 3, 7.5);

    let params = vec![0.0, 1.0, 2.0, 5.5, 4.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// 41. EqualParamConstraint
// =============================================================================

#[test]
fn test_jacobian_equal_param() {
    let id = ConstraintId(0);
    let constraint = EqualParamConstraint::new(id, 1, 4);

    let params = vec![0.0, 3.5, 2.0, 1.0, 4.2, 5.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

// =============================================================================
// Additional tests for edge cases and extra coverage
// =============================================================================

/// Test DistanceConstraint with non-trivial parameter offsets.
#[test]
fn test_jacobian_distance_2d_offset() {
    let id = ConstraintId(0);
    // Points stored at non-zero offsets in the parameter vector
    let p1 = ParamRange { start: 3, count: 2 };
    let p2 = ParamRange { start: 7, count: 2 };
    let constraint = DistanceConstraint::new(id, p1, p2, 5.0);

    let params = vec![
        0.0, 0.0, 0.0,      // padding
        1.5, 2.7,            // p1 at indices 3,4
        0.0, 0.0,            // padding
        4.1, -0.3,           // p2 at indices 7,8
    ];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

/// Test PointOnEllipseConstraint with rotated ellipse at non-origin center.
#[test]
fn test_jacobian_point_on_ellipse_rotated() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 2 };
    let ellipse = ParamRange { start: 2, count: 5 };
    let constraint = PointOnEllipseConstraint::new(id, point, ellipse);

    // Point (3.0, 5.0), ellipse center (1.5, 2.5), rx=4, ry=2, rotation=1.2
    let params = vec![3.0, 5.0, 1.5, 2.5, 4.0, 2.0, 1.2];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

/// Test PointOnBezierConstraint at t near 0 and near 1.
#[test]
fn test_jacobian_point_on_bezier_edge_t() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 0, count: 2 };
    let bezier = ParamRange { start: 2, count: 8 };
    let t_param = 10;
    let constraint = PointOnBezierConstraint::new(id, point, bezier, t_param);

    // t near 0.1
    let mut params = vec![0.0; 11];
    params[0] = 0.3;
    params[1] = 0.2;
    params[2] = 0.0;  params[3] = 0.0;
    params[4] = 1.0;  params[5] = 2.0;
    params[6] = 3.0;  params[7] = 1.0;
    params[8] = 4.0;  params[9] = 0.0;
    params[10] = 0.1;
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);

    // t near 0.9
    params[10] = 0.9;
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

/// Test G0ContinuityConstraint from_beziers constructor.
#[test]
fn test_jacobian_g0_from_beziers() {
    let id = ConstraintId(0);
    let bezier1 = ParamRange { start: 0, count: 8 };
    let bezier2 = ParamRange { start: 8, count: 8 };
    let constraint = G0ContinuityConstraint::from_beziers(id, bezier1, bezier2);

    let params = vec![
        0.0, 0.0,   // P0_a
        1.0, 1.5,   // P1_a
        2.5, 0.5,   // P2_a
        4.0, 1.0,   // P3_a (indices 6,7)
        4.5, 1.2,   // P0_b (indices 8,9)
        5.5, 2.0,   // P1_b
        7.0, 0.3,   // P2_b
        9.0, 1.5,   // P3_b
    ];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

/// Test ParamRangeConstraint Jacobian.
#[test]
fn test_jacobian_param_range() {
    let id = ConstraintId(0);
    let constraint = ParamRangeConstraint::new(id, 2, 1.0, 10.0, 3, 4);

    let params = vec![0.0, 0.0, 5.5, 1.8, 1.3];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

/// Test RatioParamConstraint Jacobian.
#[test]
fn test_jacobian_ratio_param() {
    let id = ConstraintId(0);
    let constraint = RatioParamConstraint::new(id, 1, 3, 2.5);

    let params = vec![0.0, 7.0, 0.0, 3.0, 0.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

/// Test SymmetricAboutLineConstraint Jacobian.
#[test]
fn test_jacobian_symmetric_about_line() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };
    let axis_start = ParamRange { start: 4, count: 2 };
    let axis_end = ParamRange { start: 6, count: 2 };
    let constraint = SymmetricAboutLineConstraint::new(id, p1, p2, axis_start, axis_end);

    // p1 = (1.5, 3.0), p2 = (4.5, 1.0), axis from (0, 0) to (3, 2)
    let params = vec![1.5, 3.0, 4.5, 1.0, 0.0, 0.0, 3.0, 2.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

/// Test AngleConstraint with different target angles.
#[test]
fn test_jacobian_angle_various() {
    let id = ConstraintId(0);
    let p1 = ParamRange { start: 0, count: 2 };
    let p2 = ParamRange { start: 2, count: 2 };

    // Test with angle = 0
    let constraint_zero = AngleConstraint::new(id, p1, p2, 0.0);
    let params = vec![1.0, 2.0, 5.0, 3.5];
    verify_constraint_jacobian(&constraint_zero, &params, EPSILON, TOLERANCE);

    // Test with angle = pi/2
    let constraint_90 = AngleConstraint::new(id, p1, p2, std::f64::consts::FRAC_PI_2);
    verify_constraint_jacobian(&constraint_90, &params, EPSILON, TOLERANCE);

    // Test with angle = pi/3
    let constraint_60 = AngleConstraint::new(id, p1, p2, std::f64::consts::FRAC_PI_3);
    verify_constraint_jacobian(&constraint_60, &params, EPSILON, TOLERANCE);
}

/// Test LineTangentCircleConstraint with non-horizontal line.
#[test]
fn test_jacobian_tangent_line_circle_angled() {
    let id = ConstraintId(0);
    let line = ParamRange { start: 0, count: 4 };
    let circle = ParamRange { start: 4, count: 3 };
    let constraint = LineTangentCircleConstraint::new(id, line, circle);

    // Angled line from (1, 1) to (5, 7), circle center (3, 2) radius 2
    let params = vec![1.0, 1.0, 5.0, 7.0, 3.0, 2.0, 2.0];
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}

/// Test PointOnCircleConstraint with offset and larger parameter vector.
#[test]
fn test_jacobian_point_on_circle_offset() {
    let id = ConstraintId(0);
    let point = ParamRange { start: 5, count: 2 };
    let circle = ParamRange { start: 10, count: 3 };
    let constraint = PointOnCircleConstraint::new(id, point, circle);

    let mut params = vec![0.0; 13];
    params[5] = 4.5;   // px
    params[6] = 3.2;   // py
    params[10] = 1.0;  // cx
    params[11] = 2.0;  // cy
    params[12] = 4.0;  // r
    verify_constraint_jacobian(&constraint, &params, EPSILON, TOLERANCE);
}
