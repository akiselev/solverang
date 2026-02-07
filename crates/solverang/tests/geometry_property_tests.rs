#![cfg(feature = "geometry")]
//! Property-based tests for the geometry module constraints.
//!
//! The geometry module uses `GeometricConstraint<D>` trait which operates on
//! `&[Point<D>]` instead of `ParamStore`. This test file covers both 2D and 3D
//! variants with satisfaction, Jacobian correctness, and invariance properties.

use std::collections::HashMap;
use proptest::prelude::*;
use solverang::geometry::constraints::{
    CoincidentConstraint, CollinearConstraint, DistanceConstraint, EqualLengthConstraint,
    FixedConstraint, GeometricConstraint, HorizontalConstraint, MidpointConstraint,
    ParallelConstraint, PerpendicularConstraint, PointOnCircleConstraint,
    PointOnLineConstraint, SymmetricConstraint, VerticalConstraint,
};
use solverang::geometry::{Point, Point2D, Point3D};

// =============================================================================
// Helpers
// =============================================================================

/// Central finite-difference Jacobian check for `GeometricConstraint<D>`.
///
/// The Jacobian entries are `(row, col, value)` where `col = point_idx * D + coord`.
/// Uses a `HashMap<(usize, usize), f64>` for O(1) analytical Jacobian lookup.
fn check_jacobian_fd<const D: usize>(
    constraint: &dyn GeometricConstraint<D>,
    points: &[Point<D>],
    eps: f64,
    tol: f64,
) -> bool {
    let analytical = constraint.jacobian(points);
    let n_eq = constraint.equation_count();
    let n_cols = points.len() * D;

    // Build HashMap for O(1) lookup: (equation_index, column) -> value
    let mut ana_map: HashMap<(usize, usize), f64> = HashMap::new();
    for &(eq, col, val) in &analytical {
        *ana_map.entry((eq, col)).or_insert(0.0) += val;
    }

    for eq in 0..n_eq {
        for col in 0..n_cols {
            let point_idx = col / D;
            let coord_idx = col % D;

            let orig = points[point_idx].get(coord_idx);
            let h = eps * (1.0 + orig.abs());

            let mut plus = points.to_vec();
            plus[point_idx].set(coord_idx, orig + h);
            let r_plus = constraint.residuals(&plus);

            let mut minus = points.to_vec();
            minus[point_idx].set(coord_idx, orig - h);
            let r_minus = constraint.residuals(&minus);

            let fd = (r_plus[eq] - r_minus[eq]) / (2.0 * h);

            let ana = ana_map.get(&(eq, col)).copied().unwrap_or(0.0);

            let error = (fd - ana).abs();
            let scale = 1.0 + fd.abs().max(ana.abs());
            if error >= tol * scale {
                return false;
            }
        }
    }
    true
}

// =============================================================================
// Proptest strategies
// =============================================================================

fn coord() -> impl Strategy<Value = f64> { -500.0f64..500.0 }
fn positive_dist() -> impl Strategy<Value = f64> { 0.1f64..200.0 }
fn positive_radius() -> impl Strategy<Value = f64> { 0.5f64..100.0 }
fn angle_rad() -> impl Strategy<Value = f64> { 0.0f64..std::f64::consts::TAU }

fn point2d() -> impl Strategy<Value = Point2D> {
    (coord(), coord()).prop_map(|(x, y)| Point2D::new(x, y))
}

fn point3d() -> impl Strategy<Value = Point3D> {
    (coord(), coord(), coord()).prop_map(|(x, y, z)| Point3D::new(x, y, z))
}

// =============================================================================
// Section 1: Constraint satisfaction ⟹ residuals ≈ 0 (2D)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // ---- DistanceConstraint -------------------------------------------------

    #[test]
    fn prop_distance_2d_satisfied(
        x1 in coord(), y1 in coord(),
        angle in angle_rad(),
        dist in positive_dist(),
    ) {
        let x2 = x1 + dist * angle.cos();
        let y2 = y1 + dist * angle.sin();
        let points = vec![Point2D::new(x1, y1), Point2D::new(x2, y2)];

        let c = DistanceConstraint::<2>::new(0, 1, dist);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-8, "Distance2D residual: {}", r[0]);
    }

    #[test]
    fn prop_distance_3d_satisfied(
        x1 in coord(), y1 in coord(), z1 in coord(),
        theta in 0.01f64..std::f64::consts::PI - 0.01,
        phi in 0.0f64..std::f64::consts::TAU,
        dist in positive_dist(),
    ) {
        let x2 = x1 + dist * theta.sin() * phi.cos();
        let y2 = y1 + dist * theta.sin() * phi.sin();
        let z2 = z1 + dist * theta.cos();
        let points = vec![Point3D::new(x1, y1, z1), Point3D::new(x2, y2, z2)];

        let c = DistanceConstraint::<3>::new(0, 1, dist);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-8, "Distance3D residual: {}", r[0]);
    }

    // ---- CoincidentConstraint -----------------------------------------------

    #[test]
    fn prop_coincident_2d_satisfied(x in coord(), y in coord()) {
        let points = vec![Point2D::new(x, y), Point2D::new(x, y)];
        let c = CoincidentConstraint::<2>::new(0, 1);
        let r = c.residuals(&points);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-12));
    }

    // ---- FixedConstraint ----------------------------------------------------

    #[test]
    fn prop_fixed_2d_satisfied(x in coord(), y in coord()) {
        let points = vec![Point2D::new(x, y)];
        let c = FixedConstraint::<2>::new(0, Point2D::new(x, y));
        let r = c.residuals(&points);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-12));
    }

    // ---- HorizontalConstraint -----------------------------------------------

    #[test]
    fn prop_horizontal_satisfied(x1 in coord(), x2 in coord(), y in coord()) {
        let points = vec![Point2D::new(x1, y), Point2D::new(x2, y)];
        let c = HorizontalConstraint::new(0, 1);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-12, "Horizontal residual: {}", r[0]);
    }

    // ---- VerticalConstraint -------------------------------------------------

    #[test]
    fn prop_vertical_satisfied(x in coord(), y1 in coord(), y2 in coord()) {
        let points = vec![Point2D::new(x, y1), Point2D::new(x, y2)];
        let c = VerticalConstraint::new(0, 1);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-12, "Vertical residual: {}", r[0]);
    }

    // ---- ParallelConstraint -------------------------------------------------

    #[test]
    fn prop_parallel_2d_satisfied(
        x1 in coord(), y1 in coord(),
        dx in -100.0f64..100.0, dy in -100.0f64..100.0,
        x3 in coord(), y3 in coord(),
        scale in 0.1f64..10.0,
    ) {
        let dir_len = (dx * dx + dy * dy).sqrt();
        prop_assume!(dir_len > 0.1);

        let points = vec![
            Point2D::new(x1, y1), Point2D::new(x1 + dx, y1 + dy),
            Point2D::new(x3, y3), Point2D::new(x3 + scale * dx, y3 + scale * dy),
        ];
        let c = ParallelConstraint::<2>::new(0, 1, 2, 3);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-4, "Parallel2D residual: {}", r[0]);
    }

    // ---- PerpendicularConstraint --------------------------------------------

    #[test]
    fn prop_perpendicular_2d_satisfied(
        x1 in coord(), y1 in coord(),
        dx in -100.0f64..100.0, dy in -100.0f64..100.0,
        x3 in coord(), y3 in coord(),
    ) {
        let dir_len = (dx * dx + dy * dy).sqrt();
        prop_assume!(dir_len > 0.1);
        // Perpendicular direction: (-dy, dx)
        let points = vec![
            Point2D::new(x1, y1), Point2D::new(x1 + dx, y1 + dy),
            Point2D::new(x3, y3), Point2D::new(x3 - dy, y3 + dx),
        ];
        let c = PerpendicularConstraint::<2>::new(0, 1, 2, 3);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-4, "Perpendicular2D residual: {}", r[0]);
    }

    // ---- MidpointConstraint -------------------------------------------------

    #[test]
    fn prop_midpoint_2d_satisfied(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
    ) {
        let mx = (x1 + x2) / 2.0;
        let my = (y1 + y2) / 2.0;
        let points = vec![Point2D::new(mx, my), Point2D::new(x1, y1), Point2D::new(x2, y2)];
        let c = MidpointConstraint::<2>::new(0, 1, 2);
        let r = c.residuals(&points);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-8), "Midpoint residuals: {:?}", r);
    }

    // ---- PointOnCircleConstraint --------------------------------------------

    #[test]
    fn prop_point_on_circle_2d_satisfied(
        cx in coord(), cy in coord(),
        angle in angle_rad(),
        radius in positive_radius(),
    ) {
        let px = cx + radius * angle.cos();
        let py = cy + radius * angle.sin();
        let points = vec![Point2D::new(px, py), Point2D::new(cx, cy)];
        let c = PointOnCircleConstraint::<2>::new(0, 1, radius);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-8, "PointOnCircle residual: {}", r[0]);
    }

    // ---- PointOnLineConstraint ----------------------------------------------

    #[test]
    fn prop_point_on_line_2d_satisfied(
        x1 in coord(), y1 in coord(),
        x2 in coord(), y2 in coord(),
        t in 0.0f64..1.0,
    ) {
        let line_len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        prop_assume!(line_len > 0.1);
        let px = x1 + t * (x2 - x1);
        let py = y1 + t * (y2 - y1);
        let points = vec![Point2D::new(px, py), Point2D::new(x1, y1), Point2D::new(x2, y2)];
        let c = PointOnLineConstraint::<2>::new(0, 1, 2);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-6, "PointOnLine residual: {}", r[0]);
    }

    // ---- EqualLengthConstraint ----------------------------------------------

    #[test]
    fn prop_equal_length_2d_satisfied(
        x1 in coord(), y1 in coord(),
        angle1 in angle_rad(),
        x3 in coord(), y3 in coord(),
        angle2 in angle_rad(),
        dist in positive_dist(),
    ) {
        let points = vec![
            Point2D::new(x1, y1),
            Point2D::new(x1 + dist * angle1.cos(), y1 + dist * angle1.sin()),
            Point2D::new(x3, y3),
            Point2D::new(x3 + dist * angle2.cos(), y3 + dist * angle2.sin()),
        ];
        let c = EqualLengthConstraint::<2>::new(0, 1, 2, 3);
        let r = c.residuals(&points);
        prop_assert!(r[0].abs() < 1e-8, "EqualLength residual: {}", r[0]);
    }

    // ---- SymmetricConstraint ------------------------------------------------

    #[test]
    fn prop_symmetric_2d_satisfied(
        cx in coord(), cy in coord(),
        dx in -100.0f64..100.0, dy in -100.0f64..100.0,
    ) {
        // p1 = center + offset, p2 = center - offset
        let points = vec![
            Point2D::new(cx + dx, cy + dy),
            Point2D::new(cx - dx, cy - dy),
            Point2D::new(cx, cy),
        ];
        let c = SymmetricConstraint::<2>::new(0, 1, 2);
        let r = c.residuals(&points);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-8), "Symmetric residuals: {:?}", r);
    }

    // ---- CollinearConstraint ------------------------------------------------

    #[test]
    fn prop_collinear_2d_satisfied(
        x1 in coord(), y1 in coord(),
        dx in -100.0f64..100.0, dy in -100.0f64..100.0,
        t1 in 0.1f64..2.0, t2 in 0.1f64..2.0,
    ) {
        let dir_len = (dx * dx + dy * dy).sqrt();
        prop_assume!(dir_len > 0.1);

        // Two line segments on the same line
        let points = vec![
            Point2D::new(x1, y1),
            Point2D::new(x1 + dx, y1 + dy),
            Point2D::new(x1 + t1 * dx, y1 + t1 * dy),
            Point2D::new(x1 + t2 * dx, y1 + t2 * dy),
        ];
        let c = CollinearConstraint::<2>::new(0, 1, 2, 3);
        let r = c.residuals(&points);
        prop_assert!(r.iter().all(|v| v.abs() < 1e-4), "Collinear residuals: {:?}", r);
    }
}

// =============================================================================
// Section 2: Jacobian correctness (2D)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_distance_2d_jacobian(p1 in point2d(), p2 in point2d(), dist in positive_dist()) {
        let actual_dist = p1.distance_to(&p2);
        prop_assume!(actual_dist > 0.01);  // avoid singularity at zero distance
        let points = vec![p1, p2];
        let c = DistanceConstraint::<2>::new(0, 1, dist);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "Distance2D Jacobian mismatch");
    }

    #[test]
    fn prop_coincident_2d_jacobian(p1 in point2d(), p2 in point2d()) {
        let points = vec![p1, p2];
        let c = CoincidentConstraint::<2>::new(0, 1);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Coincident2D Jacobian mismatch");
    }

    #[test]
    fn prop_fixed_2d_jacobian(p in point2d(), target in point2d()) {
        let points = vec![p];
        let c = FixedConstraint::<2>::new(0, target);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Fixed2D Jacobian mismatch");
    }

    #[test]
    fn prop_horizontal_jacobian(p1 in point2d(), p2 in point2d()) {
        let points = vec![p1, p2];
        let c = HorizontalConstraint::new(0, 1);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Horizontal Jacobian mismatch");
    }

    #[test]
    fn prop_vertical_jacobian(p1 in point2d(), p2 in point2d()) {
        let points = vec![p1, p2];
        let c = VerticalConstraint::new(0, 1);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Vertical Jacobian mismatch");
    }

    #[test]
    fn prop_parallel_2d_jacobian(p1 in point2d(), p2 in point2d(), p3 in point2d(), p4 in point2d()) {
        let points = vec![p1, p2, p3, p4];
        let c = ParallelConstraint::<2>::new(0, 1, 2, 3);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "Parallel2D Jacobian mismatch");
    }

    #[test]
    fn prop_perpendicular_2d_jacobian(p1 in point2d(), p2 in point2d(), p3 in point2d(), p4 in point2d()) {
        let points = vec![p1, p2, p3, p4];
        let c = PerpendicularConstraint::<2>::new(0, 1, 2, 3);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "Perpendicular2D Jacobian mismatch");
    }

    #[test]
    fn prop_midpoint_2d_jacobian(mid in point2d(), p1 in point2d(), p2 in point2d()) {
        let points = vec![mid, p1, p2];
        let c = MidpointConstraint::<2>::new(0, 1, 2);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Midpoint2D Jacobian mismatch");
    }

    #[test]
    fn prop_point_on_circle_2d_jacobian(p in point2d(), center in point2d(), radius in positive_radius()) {
        let actual_dist = p.distance_to(&center);
        prop_assume!(actual_dist > 0.01);  // avoid singularity
        let points = vec![p, center];
        let c = PointOnCircleConstraint::<2>::new(0, 1, radius);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "PointOnCircle2D Jacobian mismatch");
    }

    #[test]
    fn prop_point_on_line_2d_jacobian(p in point2d(), p1 in point2d(), p2 in point2d()) {
        let line_len = p1.distance_to(&p2);
        prop_assume!(line_len > 0.1);
        let points = vec![p, p1, p2];
        let c = PointOnLineConstraint::<2>::new(0, 1, 2);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "PointOnLine2D Jacobian mismatch");
    }

    #[test]
    fn prop_equal_length_2d_jacobian(p1 in point2d(), p2 in point2d(), p3 in point2d(), p4 in point2d()) {
        let l1 = p1.distance_to(&p2);
        let l2 = p3.distance_to(&p4);
        prop_assume!(l1 > 0.01 && l2 > 0.01);
        let points = vec![p1, p2, p3, p4];
        let c = EqualLengthConstraint::<2>::new(0, 1, 2, 3);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "EqualLength2D Jacobian mismatch");
    }

    #[test]
    fn prop_symmetric_2d_jacobian(p1 in point2d(), p2 in point2d(), center in point2d()) {
        let points = vec![p1, p2, center];
        let c = SymmetricConstraint::<2>::new(0, 1, 2);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Symmetric2D Jacobian mismatch");
    }

    #[test]
    fn prop_collinear_2d_jacobian(p1 in point2d(), p2 in point2d(), p3 in point2d(), p4 in point2d()) {
        let points = vec![p1, p2, p3, p4];
        let c = CollinearConstraint::<2>::new(0, 1, 2, 3);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "Collinear2D Jacobian mismatch");
    }
}

// =============================================================================
// Section 3: 3D Jacobian correctness
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_distance_3d_jacobian(p1 in point3d(), p2 in point3d(), dist in positive_dist()) {
        let actual_dist = p1.distance_to(&p2);
        prop_assume!(actual_dist > 0.01);
        let points = vec![p1, p2];
        let c = DistanceConstraint::<3>::new(0, 1, dist);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "Distance3D Jacobian mismatch");
    }

    #[test]
    fn prop_coincident_3d_jacobian(p1 in point3d(), p2 in point3d()) {
        let points = vec![p1, p2];
        let c = CoincidentConstraint::<3>::new(0, 1);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Coincident3D Jacobian mismatch");
    }

    #[test]
    fn prop_parallel_3d_jacobian(p1 in point3d(), p2 in point3d(), p3 in point3d(), p4 in point3d()) {
        let points = vec![p1, p2, p3, p4];
        let c = ParallelConstraint::<3>::new(0, 1, 2, 3);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "Parallel3D Jacobian mismatch");
    }

    #[test]
    fn prop_perpendicular_3d_jacobian(p1 in point3d(), p2 in point3d(), p3 in point3d(), p4 in point3d()) {
        let points = vec![p1, p2, p3, p4];
        let c = PerpendicularConstraint::<3>::new(0, 1, 2, 3);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "Perpendicular3D Jacobian mismatch");
    }

    #[test]
    fn prop_midpoint_3d_jacobian(mid in point3d(), p1 in point3d(), p2 in point3d()) {
        let points = vec![mid, p1, p2];
        let c = MidpointConstraint::<3>::new(0, 1, 2);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Midpoint3D Jacobian mismatch");
    }

    #[test]
    fn prop_symmetric_3d_jacobian(p1 in point3d(), p2 in point3d(), center in point3d()) {
        let points = vec![p1, p2, center];
        let c = SymmetricConstraint::<3>::new(0, 1, 2);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-5), "Symmetric3D Jacobian mismatch");
    }

    #[test]
    fn prop_equal_length_3d_jacobian(p1 in point3d(), p2 in point3d(), p3 in point3d(), p4 in point3d()) {
        let l1 = p1.distance_to(&p2);
        let l2 = p3.distance_to(&p4);
        prop_assume!(l1 > 0.01 && l2 > 0.01);
        let points = vec![p1, p2, p3, p4];
        let c = EqualLengthConstraint::<3>::new(0, 1, 2, 3);
        prop_assert!(check_jacobian_fd(&c, &points, 1e-7, 1e-4), "EqualLength3D Jacobian mismatch");
    }
}

// =============================================================================
// Section 4: Invariance and symmetry properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Distance constraint is symmetric: dist(p1,p2) == dist(p2,p1).
    #[test]
    fn prop_distance_2d_symmetric(p1 in point2d(), p2 in point2d(), dist in positive_dist()) {
        let points = vec![p1, p2];
        let c12 = DistanceConstraint::<2>::new(0, 1, dist);
        let c21 = DistanceConstraint::<2>::new(1, 0, dist);
        let r12 = c12.residuals(&points);
        let r21 = c21.residuals(&points);
        prop_assert!((r12[0] - r21[0]).abs() < 1e-10, "Distance not symmetric");
    }

    /// Parallel constraint is invariant under translation.
    #[test]
    fn prop_parallel_2d_translation_invariant(
        p1 in point2d(), p2 in point2d(), p3 in point2d(), p4 in point2d(),
        tx in -100.0f64..100.0, ty in -100.0f64..100.0,
    ) {
        let original = vec![p1, p2, p3, p4];
        let translated = vec![
            Point2D::new(p1.x() + tx, p1.y() + ty),
            Point2D::new(p2.x() + tx, p2.y() + ty),
            Point2D::new(p3.x() + tx, p3.y() + ty),
            Point2D::new(p4.x() + tx, p4.y() + ty),
        ];

        let c = ParallelConstraint::<2>::new(0, 1, 2, 3);
        let r_orig = c.residuals(&original);
        let r_trans = c.residuals(&translated);

        for (a, b) in r_orig.iter().zip(r_trans.iter()) {
            prop_assert!((a - b).abs() < 1e-6, "Parallel not translation-invariant: {} vs {}", a, b);
        }
    }

    /// Equation counts are correct for all 2D constraint types.
    #[test]
    fn prop_equation_counts_2d(p1 in point2d(), p2 in point2d(), p3 in point2d(), p4 in point2d()) {
        let pts2 = vec![p1, p2];
        let pts3 = vec![p1, p2, p3];
        let pts4 = vec![p1, p2, p3, p4];

        prop_assert_eq!(DistanceConstraint::<2>::new(0, 1, 1.0).equation_count(), 1);
        prop_assert_eq!(CoincidentConstraint::<2>::new(0, 1).equation_count(), 2);
        prop_assert_eq!(FixedConstraint::<2>::new(0, p1).equation_count(), 2);
        prop_assert_eq!(HorizontalConstraint::new(0, 1).equation_count(), 1);
        prop_assert_eq!(VerticalConstraint::new(0, 1).equation_count(), 1);
        prop_assert_eq!(ParallelConstraint::<2>::new(0, 1, 2, 3).equation_count(), 1);
        prop_assert_eq!(PerpendicularConstraint::<2>::new(0, 1, 2, 3).equation_count(), 1);
        prop_assert_eq!(MidpointConstraint::<2>::new(0, 1, 2).equation_count(), 2);
        prop_assert_eq!(PointOnCircleConstraint::<2>::new(0, 1, 1.0).equation_count(), 1);
        prop_assert_eq!(EqualLengthConstraint::<2>::new(0, 1, 2, 3).equation_count(), 1);
        prop_assert_eq!(SymmetricConstraint::<2>::new(0, 1, 2).equation_count(), 2);

        // Verify residual length matches equation count
        prop_assert_eq!(DistanceConstraint::<2>::new(0, 1, 1.0).residuals(&pts2).len(), 1);
        prop_assert_eq!(CoincidentConstraint::<2>::new(0, 1).residuals(&pts2).len(), 2);
        prop_assert_eq!(MidpointConstraint::<2>::new(0, 1, 2).residuals(&pts3).len(), 2);
        prop_assert_eq!(ParallelConstraint::<2>::new(0, 1, 2, 3).residuals(&pts4).len(), 1);
    }

    /// All residuals and Jacobians produce finite values.
    #[test]
    fn prop_residuals_finite_2d(p1 in point2d(), p2 in point2d()) {
        let dist = p1.distance_to(&p2);
        prop_assume!(dist > 0.01);  // avoid singularity

        let points = vec![p1, p2];
        let c = DistanceConstraint::<2>::new(0, 1, 5.0);
        let r = c.residuals(&points);
        let j = c.jacobian(&points);

        prop_assert!(r.iter().all(|v| v.is_finite()), "Non-finite residual");
        prop_assert!(j.iter().all(|&(_, _, v)| v.is_finite()), "Non-finite Jacobian");
    }
}
