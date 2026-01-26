//! Property-based tests for solverang.
//!
//! These tests use proptest to verify invariants that should hold for all valid inputs,
//! providing stronger guarantees than example-based testing alone.
//!
//! Run with: cargo test -p solverang --features geometry,parallel,sparse prop_

use proptest::prelude::*;
use solverang::{
    verify_jacobian, AutoSolver, LMConfig, LMSolver, Problem, RobustSolver, SolveResult,
};

#[cfg(feature = "geometry")]
use solverang::geometry::{
    constraints::{
        CoincidentConstraint, DistanceConstraint, EqualLengthConstraint, FixedConstraint,
        GeometricConstraint, HorizontalConstraint, MidpointConstraint, ParallelConstraint,
        PerpendicularConstraint, PointOnLineConstraint, SymmetricConstraint, VerticalConstraint,
    },
    ConstraintSystem, Point2D,
};

// =============================================================================
// Test Problem Implementations
// =============================================================================

/// A simple configurable test problem for property testing.
/// F(x) = x - target, so the solution is x = target.
struct LinearProblem {
    target: Vec<f64>,
}

impl Problem for LinearProblem {
    fn name(&self) -> &str {
        "linear-test"
    }

    fn residual_count(&self) -> usize {
        self.target.len()
    }

    fn variable_count(&self) -> usize {
        self.target.len()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(&self.target).map(|(xi, ti)| xi - ti).collect()
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        // Identity matrix: dF_i/dx_j = 1 if i==j, 0 otherwise
        (0..self.target.len()).map(|i| (i, i, 1.0)).collect()
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor; self.target.len()]
    }
}

/// A quadratic problem for testing: x_i^2 - target_i = 0
struct QuadraticProblem {
    target: Vec<f64>,
}

impl Problem for QuadraticProblem {
    fn name(&self) -> &str {
        "quadratic-test"
    }

    fn residual_count(&self) -> usize {
        self.target.len()
    }

    fn variable_count(&self) -> usize {
        self.target.len()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.target)
            .map(|(xi, ti)| xi * xi - ti)
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        x.iter()
            .enumerate()
            .map(|(i, xi)| (i, i, 2.0 * xi))
            .collect()
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        self.target
            .iter()
            .map(|t| t.abs().sqrt() * factor)
            .collect()
    }
}

/// A coupled nonlinear problem: x_i + x_{i+1}^2 - target_i = 0
struct CoupledProblem {
    target: Vec<f64>,
}

impl Problem for CoupledProblem {
    fn name(&self) -> &str {
        "coupled-test"
    }

    fn residual_count(&self) -> usize {
        self.target.len()
    }

    fn variable_count(&self) -> usize {
        self.target.len()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        x.iter()
            .enumerate()
            .zip(&self.target)
            .map(|((i, xi), ti)| {
                let next = if i + 1 < n { x[i + 1] } else { 0.0 };
                xi + next * next - ti
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = x.len();
        let mut entries = Vec::new();

        for i in 0..n {
            // dF_i/dx_i = 1
            entries.push((i, i, 1.0));

            // dF_i/dx_{i+1} = 2 * x_{i+1}
            if i + 1 < n {
                entries.push((i, i + 1, 2.0 * x[i + 1]));
            }
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor; self.target.len()]
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a SolveResult is a valid variant (converged, not converged, or failed).
fn is_valid_result(result: &SolveResult) -> bool {
    matches!(
        result,
        SolveResult::Converged { .. }
            | SolveResult::NotConverged { .. }
            | SolveResult::Failed { .. }
    )
}

// =============================================================================
// Proptest Strategies
// =============================================================================

/// Strategy for generating reasonable coordinate values.
fn coord_strategy() -> impl Strategy<Value = f64> {
    prop::num::f64::NORMAL.prop_map(|x| {
        // Clamp to reasonable range to avoid numerical issues
        x.clamp(-1000.0, 1000.0)
    })
}

/// Strategy for generating a 2D point.
#[cfg(feature = "geometry")]
fn point2d_strategy() -> impl Strategy<Value = Point2D> {
    (coord_strategy(), coord_strategy()).prop_map(|(x, y)| Point2D::new(x, y))
}

/// Strategy for generating a positive distance.
#[allow(dead_code)]
fn positive_distance_strategy() -> impl Strategy<Value = f64> {
    0.001f64..1000.0
}

/// Strategy for generating a small vector of coordinates.
fn small_vec_strategy(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(coord_strategy(), min_len..=max_len)
}

/// Strategy for generating strictly positive values.
fn positive_vec_strategy(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(1.0f64..100.0, min_len..=max_len)
}

// =============================================================================
// Property Tests: Jacobian Verification
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// The analytical Jacobian should match finite differences for linear problems.
    #[test]
    fn prop_linear_jacobian_matches_finite_diff(
        target in small_vec_strategy(1, 10),
        x in small_vec_strategy(1, 10),
    ) {
        // Skip if dimensions mismatch
        if target.len() != x.len() || target.is_empty() {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let verification = verify_jacobian(&problem, &x, 1e-7, 1e-5);

        prop_assert!(
            verification.passed,
            "Linear Jacobian mismatch: max_error={}, location={:?}",
            verification.max_absolute_error,
            verification.max_error_location
        );
    }

    /// The analytical Jacobian should match finite differences for quadratic problems.
    #[test]
    fn prop_quadratic_jacobian_matches_finite_diff(
        target in positive_vec_strategy(1, 10),
        x in small_vec_strategy(1, 10),
    ) {
        if target.len() != x.len() || target.is_empty() {
            return Ok(());
        }

        // Avoid points near zero where the Jacobian has singularities
        let x_safe: Vec<f64> = x.iter().map(|v| if v.abs() < 0.1 { 0.1 } else { *v }).collect();

        let problem = QuadraticProblem { target };
        let verification = verify_jacobian(&problem, &x_safe, 1e-7, 1e-4);

        prop_assert!(
            verification.passed,
            "Quadratic Jacobian mismatch: max_error={}, location={:?}",
            verification.max_absolute_error,
            verification.max_error_location
        );
    }

    /// The analytical Jacobian should match finite differences for coupled problems.
    #[test]
    fn prop_coupled_jacobian_matches_finite_diff(
        target in small_vec_strategy(2, 8),
        x in small_vec_strategy(2, 8),
    ) {
        if target.len() != x.len() || target.is_empty() {
            return Ok(());
        }

        // Filter out extreme values that cause numerical issues
        // Values too close to zero or too large can cause finite difference errors
        let x_safe: Vec<f64> = x.iter().map(|v| {
            if v.abs() < 1e-6 {
                1.0 // Replace tiny values with reasonable ones
            } else if v.abs() > 1e6 {
                v.signum() * 1e6
            } else {
                *v
            }
        }).collect();

        let problem = CoupledProblem { target };
        let verification = verify_jacobian(&problem, &x_safe, 1e-7, 1e-3);

        prop_assert!(
            verification.passed,
            "Coupled Jacobian mismatch: max_error={}, location={:?}",
            verification.max_absolute_error,
            verification.max_error_location
        );
    }
}

// =============================================================================
// Property Tests: Solver Behavior
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Solver should not crash on valid inputs (may not converge, but should not panic).
    #[test]
    fn prop_solver_no_crash_lm(
        target in small_vec_strategy(1, 5),
        x0 in small_vec_strategy(1, 5),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }

        // Filter out non-finite values
        if target.iter().any(|v| !v.is_finite()) || x0.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let solver = LMSolver::new(LMConfig::default());

        // Should not panic
        let result = solver.solve(&problem, &x0);

        // Result should be one of the valid variants
        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }

    /// Solver should not crash on valid inputs using AutoSolver.
    #[test]
    fn prop_solver_no_crash_auto(
        target in small_vec_strategy(1, 5),
        x0 in small_vec_strategy(1, 5),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }

        if target.iter().any(|v| !v.is_finite()) || x0.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let solver = AutoSolver::new();

        let result = solver.solve(&problem, &x0);

        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }

    /// Solver should not crash on valid inputs using RobustSolver.
    #[test]
    fn prop_solver_no_crash_robust(
        target in small_vec_strategy(1, 5),
        x0 in small_vec_strategy(1, 5),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }

        if target.iter().any(|v| !v.is_finite()) || x0.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let solver = RobustSolver::new();

        let result = solver.solve(&problem, &x0);

        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }

    /// Solution residual should be less than or equal to initial residual for converged problems.
    #[test]
    fn prop_solver_improves_residual(
        target in small_vec_strategy(1, 4),
        x0 in small_vec_strategy(1, 4),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }

        if target.iter().any(|v| !v.is_finite()) || x0.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let initial_residual = problem.residual_norm(&x0);

        // Skip if initial residual is already very small or non-finite
        if !initial_residual.is_finite() || initial_residual < 1e-10 {
            return Ok(());
        }

        let solver = LMSolver::new(LMConfig::default());
        let result = solver.solve(&problem, &x0);

        match result {
            SolveResult::Converged { residual_norm, .. } |
            SolveResult::NotConverged { residual_norm, .. } => {
                // Final residual should not be worse than initial (within tolerance)
                // Allow small increase due to numerical noise
                prop_assert!(
                    residual_norm <= initial_residual * 1.001 + 1e-10,
                    "Residual increased: initial={}, final={}",
                    initial_residual,
                    residual_norm
                );
            }
            SolveResult::Failed { .. } => {
                // Failure is acceptable for some inputs
            }
        }
    }

    /// Linear problems should always converge (they have exact solutions).
    #[test]
    fn prop_linear_always_converges(
        target in small_vec_strategy(1, 5),
    ) {
        if target.is_empty() || target.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target: target.clone() };
        let x0 = vec![0.0; target.len()];

        let solver = LMSolver::new(LMConfig::default());
        let result = solver.solve(&problem, &x0);

        prop_assert!(
            result.is_converged(),
            "Linear problem should converge, got {:?}",
            result
        );

        if let SolveResult::Converged { solution, .. } = result {
            // Solution should be close to target
            for (sol, tgt) in solution.iter().zip(&target) {
                prop_assert!(
                    (sol - tgt).abs() < 1e-5,
                    "Solution mismatch: sol={}, target={}",
                    sol,
                    tgt
                );
            }
        }
    }
}

// =============================================================================
// Property Tests: Geometric Constraints
// =============================================================================

#[cfg(feature = "geometry")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Distance constraint should be symmetric: Distance(p1, p2) == Distance(p2, p1).
    #[test]
    fn prop_distance_symmetric(
        p1 in point2d_strategy(),
        p2 in point2d_strategy(),
        target in positive_distance_strategy(),
    ) {
        let points = vec![p1, p2];

        // Create constraint in both orderings
        let constraint_12 = DistanceConstraint::<2>::new(0, 1, target);
        let constraint_21 = DistanceConstraint::<2>::new(1, 0, target);

        let residuals_12 = constraint_12.residuals(&points);
        let residuals_21 = constraint_21.residuals(&points);

        // Residuals should be identical (distance is symmetric)
        prop_assert!(
            (residuals_12[0] - residuals_21[0]).abs() < 1e-10,
            "Distance not symmetric: {} vs {}",
            residuals_12[0],
            residuals_21[0]
        );
    }

    /// Distance constraint Jacobian should match finite differences.
    #[test]
    fn prop_distance_jacobian(
        p1 in point2d_strategy(),
        p2 in point2d_strategy(),
        target in positive_distance_strategy(),
    ) {
        // Avoid nearly coincident points
        let dist = ((p2.x() - p1.x()).powi(2) + (p2.y() - p1.y()).powi(2)).sqrt();
        if dist < 0.01 {
            return Ok(());
        }

        let mut system = ConstraintSystem::<2>::new();
        system.add_point(p1);
        system.add_point(p2);
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 1, target)));

        let x = system.current_values();
        let verification = verify_jacobian(&system, &x, 1e-7, 1e-4);

        prop_assert!(
            verification.passed,
            "Distance Jacobian failed: max_error={}",
            verification.max_absolute_error
        );
    }

    /// Coincident constraint should produce zero residual when points coincide.
    #[test]
    fn prop_coincident_zero_at_same_point(
        p in point2d_strategy(),
    ) {
        let points = vec![p, p];
        let constraint = CoincidentConstraint::<2>::new(0, 1);

        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Coincident residuals not zero: {:?}",
            residuals
        );
    }

    /// Coincident constraint Jacobian should match finite differences.
    #[test]
    fn prop_coincident_jacobian(
        p1 in point2d_strategy(),
        p2 in point2d_strategy(),
    ) {
        let mut system = ConstraintSystem::<2>::new();
        system.add_point(p1);
        system.add_point(p2);
        system.add_constraint(Box::new(CoincidentConstraint::<2>::new(0, 1)));

        let x = system.current_values();
        let verification = verify_jacobian(&system, &x, 1e-7, 1e-5);

        prop_assert!(
            verification.passed,
            "Coincident Jacobian failed: max_error={}",
            verification.max_absolute_error
        );
    }

    /// Horizontal constraint should produce zero residual when points have same y.
    #[test]
    fn prop_horizontal_zero_when_satisfied(
        x1 in coord_strategy(),
        x2 in coord_strategy(),
        y in coord_strategy(),
    ) {
        let points = vec![Point2D::new(x1, y), Point2D::new(x2, y)];
        let constraint = HorizontalConstraint::new(0, 1);

        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals[0].abs() < 1e-10,
            "Horizontal residual not zero: {}",
            residuals[0]
        );
    }

    /// Vertical constraint should produce zero residual when points have same x.
    #[test]
    fn prop_vertical_zero_when_satisfied(
        x in coord_strategy(),
        y1 in coord_strategy(),
        y2 in coord_strategy(),
    ) {
        let points = vec![Point2D::new(x, y1), Point2D::new(x, y2)];
        let constraint = VerticalConstraint::new(0, 1);

        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals[0].abs() < 1e-10,
            "Vertical residual not zero: {}",
            residuals[0]
        );
    }

    /// Midpoint constraint should be satisfied when point is at midpoint.
    #[test]
    fn prop_midpoint_satisfied_at_center(
        p1 in point2d_strategy(),
        p2 in point2d_strategy(),
    ) {
        let mid = Point2D::new((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0);
        let points = vec![mid, p1, p2];

        let constraint = MidpointConstraint::<2>::new(0, 1, 2);
        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Midpoint residuals not zero: {:?}",
            residuals
        );
    }

    /// Fixed constraint should produce zero residual at the target position.
    #[test]
    fn prop_fixed_zero_at_target(
        target in point2d_strategy(),
    ) {
        let points = vec![target];
        let constraint = FixedConstraint::<2>::new(0, target);

        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Fixed residuals not zero: {:?}",
            residuals
        );
    }

    /// Symmetric constraint: if p2 = 2*center - p1, residuals should be zero.
    #[test]
    fn prop_symmetric_satisfied_when_reflected(
        p1 in point2d_strategy(),
        center in point2d_strategy(),
    ) {
        let p2 = Point2D::new(2.0 * center.x() - p1.x(), 2.0 * center.y() - p1.y());
        let points = vec![p1, p2, center];

        let constraint = SymmetricConstraint::<2>::new(0, 1, 2);
        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals.iter().all(|r| r.abs() < 1e-10),
            "Symmetric residuals not zero: {:?}",
            residuals
        );
    }

    /// Equal length constraint should be zero when segments have same length.
    #[test]
    fn prop_equal_length_satisfied(
        p1 in point2d_strategy(),
        length in positive_distance_strategy(),
        angle1 in 0.0f64..std::f64::consts::TAU,
        angle2 in 0.0f64..std::f64::consts::TAU,
    ) {
        let p2 = Point2D::new(p1.x() + length * angle1.cos(), p1.y() + length * angle1.sin());
        let p3 = Point2D::new(0.0, 0.0);
        let p4 = Point2D::new(length * angle2.cos(), length * angle2.sin());

        let points = vec![p1, p2, p3, p4];
        let constraint = EqualLengthConstraint::<2>::new(0, 1, 2, 3);

        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals[0].abs() < 1e-8,
            "EqualLength residual not zero: {}",
            residuals[0]
        );
    }

    /// Parallel constraint should be zero for parallel lines.
    #[test]
    fn prop_parallel_satisfied(
        p1 in point2d_strategy(),
        direction_x in coord_strategy(),
        direction_y in coord_strategy(),
        offset in coord_strategy(),
    ) {
        // Skip zero-length direction
        let dir_len = (direction_x * direction_x + direction_y * direction_y).sqrt();
        if dir_len < 0.01 {
            return Ok(());
        }

        let p2 = Point2D::new(p1.x() + direction_x, p1.y() + direction_y);

        // Second line is offset but parallel
        let p3 = Point2D::new(p1.x() + offset, p1.y() + offset);
        let p4 = Point2D::new(p3.x() + direction_x, p3.y() + direction_y);

        let points = vec![p1, p2, p3, p4];
        let constraint = ParallelConstraint::<2>::new(0, 1, 2, 3);

        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals[0].abs() < 1e-8,
            "Parallel residual not zero: {}",
            residuals[0]
        );
    }

    /// Perpendicular constraint should be zero for perpendicular lines.
    #[test]
    fn prop_perpendicular_satisfied(
        p1 in point2d_strategy(),
        direction_x in coord_strategy(),
        direction_y in coord_strategy(),
    ) {
        let dir_len = (direction_x * direction_x + direction_y * direction_y).sqrt();
        if dir_len < 0.01 {
            return Ok(());
        }

        let p2 = Point2D::new(p1.x() + direction_x, p1.y() + direction_y);

        // Perpendicular direction: rotate 90 degrees
        let perp_x = -direction_y;
        let perp_y = direction_x;

        let p3 = Point2D::new(0.0, 0.0);
        let p4 = Point2D::new(perp_x, perp_y);

        let points = vec![p1, p2, p3, p4];
        let constraint = PerpendicularConstraint::<2>::new(0, 1, 2, 3);

        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals[0].abs() < 1e-8,
            "Perpendicular residual not zero: {}",
            residuals[0]
        );
    }

    /// Point-on-line constraint should be zero when point is on line.
    #[test]
    fn prop_point_on_line_satisfied(
        p1 in point2d_strategy(),
        p2 in point2d_strategy(),
        t in 0.0f64..1.0,
    ) {
        // Skip zero-length line
        let dist = ((p2.x() - p1.x()).powi(2) + (p2.y() - p1.y()).powi(2)).sqrt();
        if dist < 0.01 {
            return Ok(());
        }

        // Point on line: p = p1 + t*(p2 - p1)
        let point_on_line = Point2D::new(
            p1.x() + t * (p2.x() - p1.x()),
            p1.y() + t * (p2.y() - p1.y()),
        );

        let points = vec![point_on_line, p1, p2];
        let constraint = PointOnLineConstraint::<2>::new(0, 1, 2);

        let residuals = constraint.residuals(&points);

        prop_assert!(
            residuals[0].abs() < 1e-8,
            "PointOnLine residual not zero: {}",
            residuals[0]
        );
    }
}

// =============================================================================
// Property Tests: Constraint System Solving
// =============================================================================

#[cfg(feature = "geometry")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// A well-constrained triangle should converge.
    #[test]
    fn prop_triangle_converges(
        side1 in 1.0f64..100.0,
        side2 in 1.0f64..100.0,
        side3 in 1.0f64..100.0,
    ) {
        // Triangle inequality: each side must be less than sum of other two
        if side1 >= side2 + side3 || side2 >= side1 + side3 || side3 >= side1 + side2 {
            return Ok(());
        }

        use solverang::geometry::ConstraintSystemBuilder;

        let system = ConstraintSystemBuilder::<2>::new()
            .point(Point2D::new(0.0, 0.0))
            .point(Point2D::new(side1, 0.0))
            .point(Point2D::new(side1 / 2.0, side2 / 2.0)) // Initial guess
            .fix(0)
            .fix(1)
            .distance(0, 1, side1)
            .distance(1, 2, side2)
            .distance(2, 0, side3)
            .build();

        let solver = LMSolver::new(LMConfig::default());
        let initial = system.current_values();
        let result = solver.solve(&system, &initial);

        prop_assert!(
            result.is_converged() || result.is_completed(),
            "Triangle should converge, got {:?}",
            result
        );
    }

    /// Degrees of freedom calculation should be correct.
    #[test]
    fn prop_dof_calculation(
        num_free_points in 1usize..5,
        num_fixed_points in 0usize..3,
        num_distance_constraints in 0usize..10,
    ) {
        let mut system = ConstraintSystem::<2>::new();

        // Add free points
        for i in 0..num_free_points {
            system.add_point(Point2D::new(i as f64, 0.0));
        }

        // Add fixed points
        for i in 0..num_fixed_points {
            system.add_point_fixed(Point2D::new(100.0 + i as f64, 0.0));
        }

        let total_points = num_free_points + num_fixed_points;

        // Add distance constraints between adjacent points (if we have at least 2)
        let max_constraints = if total_points > 1 {
            total_points - 1
        } else {
            0
        };
        let actual_constraints = num_distance_constraints.min(max_constraints);

        for i in 0..actual_constraints {
            system.add_constraint(Box::new(DistanceConstraint::<2>::new(i, i + 1, 1.0)));
        }

        // DOF = (free points * 2) - equations
        let expected_dof = (num_free_points * 2) as i32 - actual_constraints as i32;

        prop_assert_eq!(
            system.degrees_of_freedom(),
            expected_dof,
            "DOF mismatch: free_points={}, fixed_points={}, constraints={}",
            num_free_points,
            num_fixed_points,
            actual_constraints
        );
    }
}

// =============================================================================
// Property Tests: Numerical Stability
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Solver should handle large initial values gracefully.
    #[test]
    fn prop_solver_handles_large_values(
        target in prop::collection::vec(0.0f64..10.0, 2..4),
        scale in 1.0f64..1e6,
    ) {
        if target.is_empty() {
            return Ok(());
        }

        let problem = LinearProblem { target: target.clone() };
        let x0: Vec<f64> = target.iter().map(|_| scale).collect();

        let solver = LMSolver::new(LMConfig::robust());
        let result = solver.solve(&problem, &x0);

        // Should not panic and should return a valid result
        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }

    /// Solver should handle small initial values gracefully.
    #[test]
    fn prop_solver_handles_small_values(
        target in prop::collection::vec(0.0f64..10.0, 2..4),
        scale in 1e-10f64..1e-6,
    ) {
        if target.is_empty() {
            return Ok(());
        }

        let problem = LinearProblem { target: target.clone() };
        let x0: Vec<f64> = target.iter().map(|_| scale).collect();

        let solver = LMSolver::new(LMConfig::robust());
        let result = solver.solve(&problem, &x0);

        // Should not panic
        let valid = is_valid_result(&result);
        prop_assert!(valid, "Result should be valid");
    }
}
