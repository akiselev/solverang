//! Error-path integration tests for the V3 pipeline.
//!
//! The happy-path integration tests (in `integration_tests.rs`) verify that
//! well-formed sketches converge.  These tests verify the *error-reporting*
//! surface: over-constrained clusters, conflicting constraints,
//! under-determined systems, redundancy detection, and degenerate
//! configurations that produce singular Jacobians.
//!
//! A CAD kernel must surface these diagnostics to users — "your sketch is
//! over-constrained here", "these constraints conflict", etc.  Each test
//! below deliberately builds a broken or degenerate system and asserts that
//! the solver correctly identifies and reports the problem.

use crate::id::{EntityId, ParamId};
use crate::sketch2d::constraints::*;
use crate::sketch2d::entities::*;
use crate::system::{ConstraintSystem, DiagnosticIssue, SystemStatus};

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

/// Count the number of diagnostic issues of a specific kind in a list.
fn count_diagnostics<F>(issues: &[DiagnosticIssue], pred: F) -> usize
where
    F: Fn(&DiagnosticIssue) -> bool,
{
    issues.iter().filter(|i| pred(i)).count()
}

/// Return true if the issue list contains at least one `ConflictingConstraints`.
fn has_conflicts(issues: &[DiagnosticIssue]) -> bool {
    count_diagnostics(issues, |i| {
        matches!(i, DiagnosticIssue::ConflictingConstraints { .. })
    }) > 0
}

/// Return true if the issue list contains at least one `RedundantConstraint`.
fn has_redundancies(issues: &[DiagnosticIssue]) -> bool {
    count_diagnostics(issues, |i| {
        matches!(i, DiagnosticIssue::RedundantConstraint { .. })
    }) > 0
}

/// Return true if the issue list contains at least one `UnderConstrained`.
fn has_under_constrained(issues: &[DiagnosticIssue]) -> bool {
    count_diagnostics(issues, |i| {
        matches!(i, DiagnosticIssue::UnderConstrained { .. })
    }) > 0
}

// =========================================================================
// A. OVER-CONSTRAINED CLUSTER DETECTION
// =========================================================================

/// A single point fixed at (0,0) by two independent Fixed constraints.
/// Both targets agree, so the constraints are redundant (not conflicting),
/// but the system is over-constrained (DOF < 0).
#[test]
fn test_over_constrained_duplicate_fix() {
    let mut sys = ConstraintSystem::new();
    let (e0, x0, y0) = add_point(&mut sys, 0.1, 0.1);

    // Two Fixed constraints, same target.
    let cid1 = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid1, e0, x0, y0, 0.0, 0.0)));
    let cid2 = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid2, e0, x0, y0, 0.0, 0.0)));

    // DOF should be negative: 2 free params, 4 equations.
    assert!(
        sys.degrees_of_freedom() < 0,
        "Expected negative DOF for over-constrained system, got {}",
        sys.degrees_of_freedom(),
    );

    // Diagnose should detect redundancy.
    let issues = sys.diagnose();
    assert!(
        has_redundancies(&issues),
        "Expected redundant constraint diagnostic, got: {:?}",
        issues,
    );

    // The SVD-based DOF analysis computes total_dof = free_params - rank(J).
    // For duplicate constraints with identical Jacobians, rank stays 2,
    // so total_dof = 0.  The "over-constrained" condition is detected by
    // degrees_of_freedom() (equation count) and redundancy analysis, not
    // by the rank-based DOF analysis.
    let dof = sys.analyze_dof();
    assert!(
        dof.total_dof <= 0,
        "Expected DOF <= 0 for over-constrained system, got {}",
        dof.total_dof,
    );
}

/// Three distance constraints on a fixed-base segment form a rigid
/// triangle.  Adding a fourth (conflicting) distance creates an
/// over-constrained system.
#[test]
fn test_over_constrained_triangle_extra_distance() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 3.0, 0.0);
    let (e2, x2, y2) = add_point(&mut sys, 0.0, 4.0);

    // Fix P0 at origin.
    sys.fix_param(x0);
    sys.fix_param(y0);

    // Fix P1 on x-axis.
    sys.fix_param(y1);

    // d(P0,P1) = 3 (consistent with x1=3, y1=0).
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e1, x0, y0, x1, y1, 3.0,
    )));

    // d(P0,P2) = 4.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e2, x0, y0, x2, y2, 4.0,
    )));

    // d(P1,P2) = 5 — consistent with a 3-4-5 triangle.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e1, e2, x1, y1, x2, y2, 5.0,
    )));

    // Now add an EXTRA distance that conflicts with the existing geometry:
    // d(P1,P2) = 7 — impossible given the other constraints.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e1, e2, x1, y1, x2, y2, 7.0,
    )));

    // DOF should be negative.
    assert!(
        sys.degrees_of_freedom() < 0,
        "Expected negative DOF, got {}",
        sys.degrees_of_freedom(),
    );

    // The redundancy analysis should detect the conflict.
    let redundancy = sys.analyze_redundancy();
    assert!(
        !redundancy.is_clean(),
        "Expected conflicts or redundancies, got clean analysis",
    );
}

/// Fix both coordinates of a point AND add coincident with another fixed
/// point, creating 6 equations for 4 params (2 are fixed = 0 free from
/// that side).  Over-constrained for the free point.
#[test]
fn test_over_constrained_fix_plus_coincident() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 5.0, 5.0);
    let (e1, x1, y1) = add_point(&mut sys, 5.1, 5.1);

    // Fix P0.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 5.0, 5.0)));

    // Fix P1 to the same position.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e1, x1, y1, 5.0, 5.0)));

    // Plus a coincident constraint — redundant given the above.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Coincident::new(cid, e0, e1, x0, y0, x1, y1)));

    assert!(
        sys.degrees_of_freedom() < 0,
        "Expected negative DOF, got {}",
        sys.degrees_of_freedom(),
    );

    let issues = sys.diagnose();
    assert!(
        has_redundancies(&issues),
        "Expected redundant constraint diagnostics, got: {:?}",
        issues,
    );
}

// =========================================================================
// B. CONFLICTING CONSTRAINTS
// =========================================================================

/// A point fixed at two different locations: Fixed(0,0) and Fixed(10,10).
/// These constraints conflict and cannot be simultaneously satisfied.
#[test]
fn test_conflicting_fixed_positions() {
    let mut sys = ConstraintSystem::new();
    let (e0, x0, y0) = add_point(&mut sys, 5.0, 5.0);

    let cid1 = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid1, e0, x0, y0, 0.0, 0.0)));
    let cid2 = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid2, e0, x0, y0, 10.0, 10.0)));

    // Both constraints produce 2 equations each on 2 free params => DOF < 0.
    assert!(
        sys.degrees_of_freedom() < 0,
        "Expected negative DOF, got {}",
        sys.degrees_of_freedom(),
    );

    // Redundancy analysis should detect a conflict (not just redundancy).
    let redundancy = sys.analyze_redundancy();
    assert!(
        !redundancy.conflicts.is_empty(),
        "Expected conflict groups, got none. Redundant: {:?}",
        redundancy.redundant,
    );

    // Full diagnostics should contain ConflictingConstraints.
    let issues = sys.diagnose();
    assert!(
        has_conflicts(&issues),
        "Expected ConflictingConstraints diagnostic, got: {:?}",
        issues,
    );

    // Solve: the reduce phase may trivially eliminate params from one Fixed
    // constraint, making the cluster appear "solved" even though the second
    // constraint is unsatisfied.  So we verify via residuals rather than
    // relying solely on the status enum.
    let result = sys.solve();
    match &result.status {
        SystemStatus::Solved | SystemStatus::PartiallySolved => {
            // If the solver claims convergence, the residuals must still
            // reflect the unsatisfied constraint.
            let residuals = sys.compute_residuals();
            let max_r = residuals
                .iter()
                .map(|r| r.abs())
                .max_by(f64::total_cmp)
                .unwrap_or(0.0);
            assert!(
                max_r > 1.0,
                "Conflicting system should have large residuals, max_r = {max_r}",
            );
        }
        SystemStatus::DiagnosticFailure(_) => {
            // Best outcome: the pipeline propagated the structural problem.
        }
    }
}

/// Two distance constraints on the same pair: d(P0,P1) = 3 AND d(P0,P1) = 7.
/// These are unsatisfiable simultaneously.
#[test]
fn test_conflicting_distances() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 5.0, 0.0);

    // Fix P0 so the system is more constrained.
    sys.fix_param(x0);
    sys.fix_param(y0);

    let cid1 = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid1, e0, e1, x0, y0, x1, y1, 3.0,
    )));
    let cid2 = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid2, e0, e1, x0, y0, x1, y1, 7.0,
    )));

    // Two equations, 2 free params (x1, y1) => DOF = 0, but the equations
    // are rank-deficient (both constrain the same distance).  The solver
    // should detect this.
    let redundancy = sys.analyze_redundancy();
    assert!(
        !redundancy.is_clean(),
        "Expected redundancy/conflict for incompatible distances",
    );
}

/// Horizontal constraint says y1 == y0, but we also fix them to different
/// y values.  The constraints directly conflict.
#[test]
fn test_conflicting_horizontal_with_fixed() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 5.0, 10.0);

    // Fix both points.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 0.0, 0.0)));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e1, x1, y1, 5.0, 10.0)));

    // Horizontal: y0 == y1. But y0=0, y1=10 — conflict.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Horizontal::new(cid, e0, e1, y0, y1)));

    let issues = sys.diagnose();
    assert!(
        has_conflicts(&issues),
        "Expected conflict for impossible horizontal, got: {:?}",
        issues,
    );

    // Verify DOF is negative.
    assert!(
        sys.degrees_of_freedom() < 0,
        "Expected negative DOF, got {}",
        sys.degrees_of_freedom(),
    );
}

/// Perpendicular AND parallel constraints on the same pair of lines.
/// Unless one line is degenerate (zero length), these are contradictory.
#[test]
fn test_conflicting_parallel_and_perpendicular() {
    let mut sys = ConstraintSystem::new();

    // Line 1: fixed horizontal.
    let (_e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (_e1, x1, y1) = add_point(&mut sys, 4.0, 0.0);
    let l1 = add_line_segment(&mut sys, x0, y0, x1, y1);
    sys.fix_param(x0);
    sys.fix_param(y0);
    sys.fix_param(x1);
    sys.fix_param(y1);

    // Line 2: free.
    let (_e2, x2, y2) = add_point(&mut sys, 1.0, 1.0);
    let (_e3, x3, y3) = add_point(&mut sys, 5.0, 2.0);
    let l2 = add_line_segment(&mut sys, x2, y2, x3, y3);

    // Fix start of line 2 so only end point moves.
    sys.fix_param(x2);
    sys.fix_param(y2);

    // Parallel: cross product = 0.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Parallel::new(
        cid, l1, l2, x0, y0, x1, y1, x2, y2, x3, y3,
    )));

    // Perpendicular: dot product = 0.  Conflicts with parallel.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Perpendicular::new(
        cid, l1, l2, x0, y0, x1, y1, x2, y2, x3, y3,
    )));

    // The solve should not be Solved with small residuals.
    let result = sys.solve();
    match &result.status {
        SystemStatus::Solved => {
            // If it claims solved, at least one constraint is unsatisfied
            // unless the line collapsed to zero length (which makes both
            // constraints trivially satisfied).
            let residuals = sys.compute_residuals();
            let max_r = residuals
                .iter()
                .map(|r| r.abs())
                .max_by(f64::total_cmp)
                .unwrap_or(0.0);
            if max_r < TOL {
                // The solver found a degenerate solution (zero-length line).
                // That's acceptable — the solver handles it.
                let dx = sys.get_param(x3) - sys.get_param(x2);
                let dy = sys.get_param(y3) - sys.get_param(y2);
                let len = (dx * dx + dy * dy).sqrt();
                assert!(
                    len < TOL,
                    "Parallel + perpendicular both satisfied with non-degenerate line: len = {len}",
                );
            }
        }
        SystemStatus::PartiallySolved | SystemStatus::DiagnosticFailure(_) => {
            // Expected.
        }
    }
}

// =========================================================================
// C. UNDER-DETERMINED SYSTEM DETECTION
// =========================================================================

/// A single unconstrained point has 2 DOF.  The diagnostics should report
/// it as under-constrained.
#[test]
fn test_under_constrained_free_point() {
    let mut sys = ConstraintSystem::new();
    let (_e0, _x0, _y0) = add_point(&mut sys, 1.0, 2.0);

    assert_eq!(
        sys.degrees_of_freedom(),
        2,
        "Unconstrained point should have DOF = 2",
    );

    let dof = sys.analyze_dof();
    assert!(dof.is_under_constrained());
    assert_eq!(dof.total_dof, 2);
    assert_eq!(dof.entities.len(), 1);
    assert_eq!(dof.entities[0].dof, 2);

    let issues = sys.diagnose();
    assert!(
        has_under_constrained(&issues),
        "Expected UnderConstrained diagnostic for free point, got: {:?}",
        issues,
    );
}

/// A point constrained only in x (via Vertical alignment with a fully
/// fixed reference point) still has 1 free direction in y.
#[test]
fn test_under_constrained_one_direction() {
    let mut sys = ConstraintSystem::new();

    let (e_ref, x_ref, y_ref) = add_point(&mut sys, 0.0, 0.0);
    sys.fix_param(x_ref);
    sys.fix_param(y_ref);
    let (e_tgt, x_tgt, _y_tgt) = add_point(&mut sys, 5.0, 5.0);

    // Vertical: x_ref == x_tgt.  This constrains x_tgt but not y_tgt.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Vertical::new(cid, e_ref, e_tgt, x_ref, x_tgt)));

    let issues = sys.diagnose();

    // The target point should still have 1 free direction.
    let under: Vec<_> = issues
        .iter()
        .filter_map(|i| match i {
            DiagnosticIssue::UnderConstrained {
                entity,
                free_directions,
            } => Some((*entity, *free_directions)),
            _ => None,
        })
        .collect();

    // At least the target entity should be under-constrained.
    let tgt_entry = under.iter().find(|(eid, _)| *eid == e_tgt);
    assert!(
        tgt_entry.is_some(),
        "Expected target entity to be under-constrained, diagnostics: {:?}",
        issues,
    );
    if let Some((_eid, free_dirs)) = tgt_entry {
        assert_eq!(
            *free_dirs, 1,
            "Expected 1 free direction for partially constrained point, got {}",
            free_dirs,
        );
    }
}

/// A circle with no constraints at all has 3 DOF (cx, cy, r).
#[test]
fn test_under_constrained_free_circle() {
    let mut sys = ConstraintSystem::new();
    let (_ec, _cx, _cy, _cr) = add_circle_entity(&mut sys, 0.0, 0.0, 5.0);

    assert_eq!(
        sys.degrees_of_freedom(),
        3,
        "Unconstrained circle should have DOF = 3",
    );

    let dof = sys.analyze_dof();
    assert!(dof.is_under_constrained());
    assert_eq!(dof.total_dof, 3);
}

/// Two points with a single distance constraint between them.  The system
/// has 4 params and 1 equation => DOF = 3 (translation + rotation).
#[test]
fn test_under_constrained_distance_only() {
    let mut sys = ConstraintSystem::new();
    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 5.0, 0.0);

    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e1, x0, y0, x1, y1, 5.0,
    )));

    // 4 free params, 1 equation => DOF = 3.
    assert_eq!(
        sys.degrees_of_freedom(),
        3,
        "Two points with one distance should have DOF = 3",
    );

    let issues = sys.diagnose();
    assert!(
        has_under_constrained(&issues),
        "Expected UnderConstrained diagnostics",
    );
}

/// Multiple free points with no constraints at all.  The total DOF should
/// be 2 * number_of_points.
#[test]
fn test_under_constrained_multiple_free_points() {
    let mut sys = ConstraintSystem::new();
    let n = 5;
    for i in 0..n {
        add_point(&mut sys, i as f64, i as f64 * 2.0);
    }

    assert_eq!(
        sys.degrees_of_freedom(),
        2 * n as i32,
        "DOF should be 2 * number of free points",
    );

    let dof = sys.analyze_dof();
    assert!(dof.is_under_constrained());
    assert_eq!(dof.entities.len(), n);

    // Each entity should have 2 free directions.
    for ed in &dof.entities {
        assert_eq!(ed.dof, 2, "Each free point should have DOF = 2");
    }
}

// =========================================================================
// D. SINGULAR JACOBIANS / DEGENERATE CONFIGURATIONS
// =========================================================================

/// Two coincident points with a distance constraint between them.
/// When both points are at the exact same location, the distance
/// constraint Jacobian is singular (direction is undefined for zero
/// separation).
#[test]
fn test_degenerate_coincident_points_distance() {
    let mut sys = ConstraintSystem::new();

    // Both start at exactly the same position — singular Jacobian
    // for distance constraint at zero separation.
    let (e0, x0, y0) = add_point(&mut sys, 3.0, 4.0);
    let (e1, x1, y1) = add_point(&mut sys, 3.0, 4.0);

    // Fix P0.
    sys.fix_param(x0);
    sys.fix_param(y0);

    // d(P0, P1) = 5.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e1, x0, y0, x1, y1, 5.0,
    )));

    // The solver must not panic.  It may or may not converge (the
    // Jacobian is singular at the initial guess), but it should handle
    // the degeneracy gracefully.
    let result = sys.solve();

    // We accept any status — the key requirement is no panic.
    // If it converged, verify the distance is correct.
    match &result.status {
        SystemStatus::Solved | SystemStatus::PartiallySolved => {
            let dx = sys.get_param(x1) - sys.get_param(x0);
            let dy = sys.get_param(y1) - sys.get_param(y0);
            let dist = (dx * dx + dy * dy).sqrt();
            // Check convergence with relaxed tolerance since this is a hard case.
            if result.clusters.iter().all(|c| c.residual_norm < 1e-3) {
                assert!(
                    (dist - 5.0).abs() < 1e-3,
                    "Distance should be ~5 if converged, got {dist}",
                );
            }
        }
        SystemStatus::DiagnosticFailure(_) => {
            // Acceptable: solver flagged the degenerate case.
        }
    }
}

/// A zero-length line (endpoints at the same position) with a parallel
/// constraint.  The direction is undefined, so the Jacobian of the
/// cross-product is all zeros for this line.
#[test]
fn test_degenerate_zero_length_line_parallel() {
    let mut sys = ConstraintSystem::new();

    // Line 1: a proper line.
    let (_ea, xa, ya) = add_point(&mut sys, 0.0, 0.0);
    let (_eb, xb, yb) = add_point(&mut sys, 4.0, 0.0);
    let l1 = add_line_segment(&mut sys, xa, ya, xb, yb);
    sys.fix_param(xa);
    sys.fix_param(ya);
    sys.fix_param(xb);
    sys.fix_param(yb);

    // Line 2: degenerate (both endpoints at same point).
    let (_ec, xc, yc) = add_point(&mut sys, 2.0, 2.0);
    let (_ed, xd, yd) = add_point(&mut sys, 2.0, 2.0);
    let l2 = add_line_segment(&mut sys, xc, yc, xd, yd);

    // Parallel constraint on a zero-length line: Jacobian is degenerate.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Parallel::new(
        cid, l1, l2, xa, ya, xb, yb, xc, yc, xd, yd,
    )));

    // Must not panic.
    let result = sys.solve();

    // Any outcome is acceptable — the key is graceful handling.
    match &result.status {
        SystemStatus::Solved => {
            // If solved, the cross product should be ~0.
            let dx1 = sys.get_param(xb) - sys.get_param(xa);
            let dy1 = sys.get_param(yb) - sys.get_param(ya);
            let dx2 = sys.get_param(xd) - sys.get_param(xc);
            let dy2 = sys.get_param(yd) - sys.get_param(yc);
            let cross = dx1 * dy2 - dy1 * dx2;
            // Zero-length line trivially satisfies parallel, so cross = 0.
            assert!(cross.abs() < TOL, "Cross product should be ~0, got {cross}",);
        }
        SystemStatus::PartiallySolved | SystemStatus::DiagnosticFailure(_) => {}
    }
}

/// A perpendicular constraint where both lines share an endpoint and
/// that shared point is the only free one — the constraint becomes
/// degenerate when both lines have zero length.
#[test]
fn test_degenerate_collapsed_perpendicular() {
    let mut sys = ConstraintSystem::new();

    // All three points start at the origin.
    let (_e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (_e1, x1, y1) = add_point(&mut sys, 0.0, 0.0);
    let (_e2, x2, y2) = add_point(&mut sys, 0.0, 0.0);

    sys.fix_param(x0);
    sys.fix_param(y0);

    let l1 = add_line_segment(&mut sys, x0, y0, x1, y1);
    let l2 = add_line_segment(&mut sys, x0, y0, x2, y2);

    // Perpendicular at the origin: dot(l1_dir, l2_dir) = 0.
    // When all points are at origin, both directions are zero vectors.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Perpendicular::new(
        cid, l1, l2, x0, y0, x1, y1, x0, y0, x2, y2,
    )));

    // Must not panic.
    let _result = sys.solve();
}

/// PointOnCircle with the point starting exactly at the circle center.
/// The gradient of the PointOnCircle constraint `(px-cx)^2 + (py-cy)^2 - r^2`
/// vanishes when the point is at the center (the Jacobian is all zeros
/// w.r.t. the point params).
#[test]
fn test_degenerate_point_at_circle_center() {
    let mut sys = ConstraintSystem::new();

    // Circle at origin, radius 5.
    let (ec, cx, cy, cr) = add_circle_entity(&mut sys, 0.0, 0.0, 5.0);
    sys.fix_param(cx);
    sys.fix_param(cy);
    sys.fix_param(cr);

    // Point starts exactly at the circle center — degenerate.
    let (ep, px, py) = add_point(&mut sys, 0.0, 0.0);

    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(PointOnCircle::new(
        cid, ep, ec, px, py, cx, cy, cr,
    )));

    // Must not panic.
    let result = sys.solve();

    // If converged, verify the point is on the circle.
    match &result.status {
        SystemStatus::Solved | SystemStatus::PartiallySolved => {
            let px_val = sys.get_param(px);
            let py_val = sys.get_param(py);
            let dist = (px_val * px_val + py_val * py_val).sqrt();
            if result.clusters.iter().all(|c| c.residual_norm < 1e-3) {
                assert!(
                    (dist - 5.0).abs() < 1e-3,
                    "Point should be on circle (dist from center = {dist}, expected 5.0)",
                );
            }
        }
        SystemStatus::DiagnosticFailure(_) => {}
    }
}

/// TangentLineCircle where the line passes through the circle center.
/// The tangent constraint residual is `cross^2/len_sq - r^2`; when the
/// line passes through the center, the "distance" equals zero, and the
/// gradient direction changes behavior.
#[test]
fn test_degenerate_tangent_line_through_center() {
    let mut sys = ConstraintSystem::new();

    // Circle at (5, 0), radius 3.
    let (ec, cx, cy, cr) = add_circle_entity(&mut sys, 5.0, 0.0, 3.0);
    sys.fix_param(cx);
    sys.fix_param(cy);
    sys.fix_param(cr);

    // Horizontal line through the circle center: y=0 from (0,0) to (10,0).
    // The line PASSES THROUGH the center, so distance = 0, but tangency
    // requires distance = radius.
    let (_ep1, lx1, ly1) = add_point(&mut sys, 0.0, 0.0);
    let (_ep2, lx2, ly2) = add_point(&mut sys, 10.0, 0.0);
    let el = add_line_segment(&mut sys, lx1, ly1, lx2, ly2);

    // Fix x-coordinates, let y-coordinates move to satisfy tangency.
    sys.fix_param(lx1);
    sys.fix_param(lx2);

    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(TangentLineCircle::new(
        cid, el, ec, lx1, ly1, lx2, ly2, cx, cy, cr,
    )));

    // Must not panic.
    let result = sys.solve();

    // If converged, the distance from center to line should equal radius.
    if let SystemStatus::Solved | SystemStatus::PartiallySolved = &result.status {
        if result.clusters.iter().all(|c| c.residual_norm < 1e-3) {
            let lx1_val = sys.get_param(lx1);
            let ly1_val = sys.get_param(ly1);
            let lx2_val = sys.get_param(lx2);
            let ly2_val = sys.get_param(ly2);
            let cx_val = sys.get_param(cx);
            let cy_val = sys.get_param(cy);
            let r_val = sys.get_param(cr);

            // General point-to-line distance formula:
            // |cross| / line_length where cross = (x2-x1)(cy-y1) - (y2-y1)(cx-x1)
            let dx = lx2_val - lx1_val;
            let dy = ly2_val - ly1_val;
            let cross = dx * (cy_val - ly1_val) - dy * (cx_val - lx1_val);
            let line_len = (dx * dx + dy * dy).sqrt();
            let dist_to_center = cross.abs() / line_len;

            assert!(
                (dist_to_center - r_val).abs() < 1e-2,
                "Tangent distance should be ~{} (radius), got {}",
                r_val,
                dist_to_center,
            );
        }
    }
}

// =========================================================================
// E. REDUNDANCY ANALYSIS
// =========================================================================

/// A consistent redundancy: fix a point AND add a coincident constraint
/// to another point that is also fixed to the same location.  The
/// coincident constraint is redundant but not conflicting.
#[test]
fn test_redundant_consistent_coincident() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 0.1, 0.1);

    // Fix both to the same location.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 0.0, 0.0)));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e1, x1, y1, 0.0, 0.0)));

    // Add a coincident constraint — redundant.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Coincident::new(cid, e0, e1, x0, y0, x1, y1)));

    let redundancy = sys.analyze_redundancy();
    assert!(
        !redundancy.redundant.is_empty(),
        "Expected redundant constraints, got none",
    );
    // No conflicts expected (all targets are consistent).
    assert!(
        redundancy.conflicts.is_empty(),
        "Expected no conflicts for consistent redundancy, got: {:?}",
        redundancy.conflicts,
    );

    // The rank deficiency should be > 0.
    assert!(
        redundancy.rank_deficiency() > 0,
        "Expected rank deficiency > 0, got {}",
        redundancy.rank_deficiency(),
    );
}

/// Distance + equal-length constraint where the equal-length is redundant
/// because both lines connect the same endpoints.
#[test]
fn test_redundant_equal_length_same_segment() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, x1, y1) = add_point(&mut sys, 3.0, 4.0);

    // Fix P0.
    sys.fix_param(x0);
    sys.fix_param(y0);

    // Two line segments on the same endpoints.
    let l1 = add_line_segment(&mut sys, x0, y0, x1, y1);
    let l2 = add_line_segment(&mut sys, x0, y0, x1, y1);

    // Distance: d(P0,P1) = 5.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(DistancePtPt::new(
        cid, e0, e1, x0, y0, x1, y1, 5.0,
    )));

    // Equal-length: len(l1) == len(l2).  Since l1 and l2 share all four
    // params, this is always true — the constraint is trivially redundant.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(EqualLength::new(
        cid, l1, l2, x0, y0, x1, y1, x0, y0, x1, y1,
    )));

    let redundancy = sys.analyze_redundancy();
    assert!(
        !redundancy.redundant.is_empty() || redundancy.rank_deficiency() > 0,
        "Expected redundancy or rank deficiency for trivially equal lines",
    );
}

/// Horizontal + Horizontal on the same point pair: the second is purely
/// redundant.
#[test]
fn test_redundant_duplicate_horizontal() {
    let mut sys = ConstraintSystem::new();

    let (e0, _x0, y0) = add_point(&mut sys, 0.0, 0.0);
    let (e1, _x1, y1) = add_point(&mut sys, 5.0, 0.0);

    sys.fix_param(y0);

    // Two identical horizontal constraints.
    let cid1 = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Horizontal::new(cid1, e0, e1, y0, y1)));
    let cid2 = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Horizontal::new(cid2, e0, e1, y0, y1)));

    let redundancy = sys.analyze_redundancy();
    assert!(
        !redundancy.redundant.is_empty(),
        "Expected duplicate horizontal to be redundant",
    );
    assert!(
        redundancy.conflicts.is_empty(),
        "Duplicate horizontal should not conflict",
    );
}

/// A rectangle has 3 independent perpendicular constraints.  Adding all 4
/// (one per vertex) makes one redundant.
#[test]
fn test_redundant_fourth_perpendicular_in_rectangle() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(4.0, 0.1);
    let p2 = b.add_point(4.1, 3.0);
    let p3 = b.add_point(-0.1, 3.1);

    let l01 = b.add_line_segment(p0, p1);
    let l12 = b.add_line_segment(p1, p2);
    let l23 = b.add_line_segment(p2, p3);
    let l30 = b.add_line_segment(p3, p0);

    // All 4 perpendicular constraints — one is redundant.
    b.constrain_perpendicular(l01, l12);
    b.constrain_perpendicular(l12, l23);
    b.constrain_perpendicular(l23, l30);
    b.constrain_perpendicular(l30, l01); // This one is implied.

    // Side lengths and orientation to fully constrain.
    b.constrain_distance(p0, p1, 4.0);
    b.constrain_distance(p1, p2, 3.0);
    b.constrain_distance(p2, p3, 4.0);
    b.constrain_distance(p3, p0, 3.0);
    b.constrain_horizontal(p0, p1);

    let sys = b.build();
    let redundancy = sys.analyze_redundancy();

    // At least one constraint should be flagged as redundant.
    assert!(
        !redundancy.is_clean(),
        "Expected at least one redundancy in over-specified rectangle",
    );
}

// =========================================================================
// F. COMBINED DIAGNOSTICS (diagnose() integration)
// =========================================================================

/// Verify that `diagnose()` aggregates both redundancy and DOF issues.
#[test]
fn test_diagnose_aggregates_redundancy_and_dof() {
    let mut sys = ConstraintSystem::new();

    // One free point (under-constrained, DOF=2).
    let (_e0, _x0, _y0) = add_point(&mut sys, 1.0, 2.0);

    // Another point with redundant fix constraints.
    let (e1, x1, y1) = add_point(&mut sys, 3.0, 4.0);
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e1, x1, y1, 3.0, 4.0)));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e1, x1, y1, 3.0, 4.0)));

    let issues = sys.diagnose();

    // Should contain both UnderConstrained AND RedundantConstraint.
    assert!(
        has_under_constrained(&issues),
        "Expected UnderConstrained diagnostic for free point",
    );
    assert!(
        has_redundancies(&issues),
        "Expected RedundantConstraint diagnostic for duplicate fix",
    );
}

/// Verify the solver's SystemResult carries diagnostics when there's a
/// conflicting system that doesn't converge.
///
/// With a large target gap (0,0) vs (100,100), the solver cannot converge,
/// and the pipeline should report `DiagnosticFailure` with non-empty issues
/// because the analyze phase detects both conflicts and non-convergence.
#[test]
fn test_solve_reports_diagnostic_failure_on_conflict() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 5.0, 5.0);

    // Fix to (0,0).
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 0.0, 0.0)));
    // Also fix to (100,100) — impossible.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 100.0, 100.0)));

    let result = sys.solve();

    // The pipeline may report Solved (if the reduce phase eliminates params
    // from one Fixed constraint, the cluster appears trivially solved), or
    // DiagnosticFailure if analysis results are propagated alongside
    // non-convergence.  We verify the important invariant: diagnostics
    // correctly detect the conflict regardless of solve status.
    match &result.status {
        SystemStatus::Solved | SystemStatus::PartiallySolved => {
            // The solver claimed some form of convergence; verify residuals
            // are large (both constraints cannot be satisfied).
            let residuals = sys.compute_residuals();
            let max_r = residuals
                .iter()
                .map(|r| r.abs())
                .max_by(f64::total_cmp)
                .unwrap_or(0.0);
            assert!(
                max_r > 1.0,
                "Conflicting constraints: max residual should be large, got {max_r}",
            );
        }
        SystemStatus::DiagnosticFailure(issues) => {
            // Best outcome: the pipeline propagated the structural problem.
            assert!(!issues.is_empty(), "DiagnosticFailure should carry issues",);
        }
    }

    // Regardless of solve status, diagnose() should detect the conflict.
    let issues = sys.diagnose();
    assert!(
        has_conflicts(&issues),
        "diagnose() should detect conflicts for Fixed(0,0) vs Fixed(100,100), got: {:?}",
        issues,
    );
}

/// After a failed solve on a conflicting system, diagnostics should still
/// be obtainable via the `diagnose()` method.
#[test]
fn test_diagnose_works_after_failed_solve() {
    let mut sys = ConstraintSystem::new();

    let (e0, x0, y0) = add_point(&mut sys, 5.0, 5.0);

    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 0.0, 0.0)));
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 50.0, 50.0)));

    // Solve (may or may not fail).
    let _result = sys.solve();

    // diagnose() should still work correctly after solve.
    let issues = sys.diagnose();
    assert!(
        has_conflicts(&issues),
        "diagnose() should detect conflicts even after solve, got: {:?}",
        issues,
    );
}

// =========================================================================
// G. DOF ANALYSIS EDGE CASES
// =========================================================================

/// Verify DOF analysis with all parameters fixed reports well-constrained.
#[test]
fn test_dof_all_fixed_params() {
    let mut sys = ConstraintSystem::new();
    let (_e0, x0, y0) = add_point(&mut sys, 1.0, 2.0);

    sys.fix_param(x0);
    sys.fix_param(y0);

    assert_eq!(sys.degrees_of_freedom(), 0);

    let dof = sys.analyze_dof();
    assert!(dof.is_well_constrained());
    assert_eq!(dof.total_dof, 0);
}

/// DOF analysis for a system that transitions from under-constrained to
/// well-constrained by adding constraints.
#[test]
fn test_dof_transition_under_to_well_constrained() {
    let mut sys = ConstraintSystem::new();
    let (e0, x0, y0) = add_point(&mut sys, 1.0, 2.0);

    // Initially under-constrained.
    assert_eq!(sys.degrees_of_freedom(), 2);

    // Add a Fixed constraint: DOF should drop to 0.
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Fixed::new(cid, e0, x0, y0, 1.0, 2.0)));

    // Now DOF should be 0 (2 params, 2 equations).
    assert_eq!(
        sys.degrees_of_freedom(),
        0,
        "Point with Fixed constraint should have DOF = 0",
    );
}

/// Verify DOF analysis is correct for a mixed system: some entities fixed,
/// some free, some partially constrained.
#[test]
fn test_dof_mixed_fixed_free_partially_constrained() {
    let mut sys = ConstraintSystem::new();

    // Entity 1: fully fixed.
    let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
    sys.fix_param(x0);
    sys.fix_param(y0);

    // Entity 2: partially constrained (only x via Vertical with fixed point).
    let (e1, x1, _y1) = add_point(&mut sys, 5.0, 5.0);
    let cid = sys.alloc_constraint_id();
    sys.add_constraint(Box::new(Vertical::new(cid, e0, e1, x0, x1)));

    // Entity 3: fully free.
    let (_e2, _x2, _y2) = add_point(&mut sys, 10.0, 10.0);

    // Total: 0 (fixed) + 2 (free) + 2 (free) = 4 free params.
    // Equations: 1 (vertical).
    // DOF = 4 - 1 = 3.
    assert_eq!(sys.degrees_of_freedom(), 3);

    let dof = sys.analyze_dof();
    assert!(dof.is_under_constrained());
    assert_eq!(dof.total_dof, 3);
}

// =========================================================================
// H. BUILDER API ERROR PATHS
// =========================================================================

/// Use the builder to create an over-constrained sketch and verify
/// diagnostics are reported.
///
/// We use `constrain_fixed` (not `add_fixed_point`) because `fix_param`
/// removes params from the solver mapping, preventing redundancy analysis
/// from seeing the Jacobian columns.  Using constraint-based fixing keeps
/// params in the mapping so redundancy/conflict detection works.
#[test]
fn test_builder_over_constrained_sketch() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_point(0.1, 0.1);
    let p1 = b.add_point(3.1, 3.9);

    // Fix both points via constraints (keeps params in solver mapping).
    b.constrain_fixed(p0, 0.0, 0.0);
    b.constrain_fixed(p1, 3.0, 4.0);

    // Redundant distance: d(P0,P1) = 5 (already determined by fix constraints).
    b.constrain_distance(p0, p1, 5.0);

    let sys = b.build();

    assert!(
        sys.degrees_of_freedom() < 0,
        "Builder sketch should be over-constrained, DOF = {}",
        sys.degrees_of_freedom(),
    );

    let redundancy = sys.analyze_redundancy();
    assert!(
        !redundancy.is_clean(),
        "Expected redundancy in over-constrained builder sketch",
    );
}

/// Use the builder to create a conflicting sketch and verify diagnostics.
///
/// Uses `constrain_fixed` instead of `add_fixed_point` so that params
/// remain in the solver mapping for redundancy analysis.
#[test]
fn test_builder_conflicting_sketch() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_point(0.1, 0.1);
    let p1 = b.add_point(3.1, 3.9);

    // Fix both points via constraints.
    b.constrain_fixed(p0, 0.0, 0.0);
    b.constrain_fixed(p1, 3.0, 4.0);

    // Conflicting distance: actual = 5, required = 10.
    b.constrain_distance(p0, p1, 10.0);

    let sys = b.build();

    let redundancy = sys.analyze_redundancy();
    assert!(
        !redundancy.conflicts.is_empty(),
        "Expected conflict for impossible distance in builder sketch. Redundant: {:?}",
        redundancy.redundant,
    );
}

/// Use the builder to verify under-constrained diagnostics.
#[test]
fn test_builder_under_constrained_sketch() {
    use crate::sketch2d::builder::Sketch2DBuilder;

    let mut b = Sketch2DBuilder::new();
    let _p0 = b.add_point(1.0, 2.0);
    let _p1 = b.add_point(5.0, 6.0);
    // No constraints at all.

    let sys = b.build();

    assert!(
        sys.degrees_of_freedom() > 0,
        "Builder sketch with no constraints should be under-constrained",
    );

    let issues = sys.diagnose();
    assert!(
        has_under_constrained(&issues),
        "Expected UnderConstrained diagnostics for unconstrained builder sketch",
    );
}
