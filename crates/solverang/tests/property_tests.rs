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
        HorizontalConstraint, MidpointConstraint, ParallelConstraint,
        PerpendicularConstraint, PointOnLineConstraint, SymmetricConstraint, VerticalConstraint,
    },
    Constraint, ConstraintId, ConstraintSystem, ParamRange,
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

/// Helper to create a ParamRange for point at a given index in a flat 2D param vector.
#[cfg(feature = "geometry")]
fn pr2d(point_index: usize) -> ParamRange {
    ParamRange {
        start: point_index * 2,
        count: 2,
    }
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

/// Strategy for generating a 2D point as (x, y) tuple.
#[cfg(feature = "geometry")]
fn point2d_strategy() -> impl Strategy<Value = (f64, f64)> {
    (coord_strategy(), coord_strategy())
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
        // Flat params: [p1.x, p1.y, p2.x, p2.y]
        let params = vec![p1.0, p1.1, p2.0, p2.1];

        // Create constraint in both orderings
        let constraint_12 = DistanceConstraint::new(ConstraintId(0), pr2d(0), pr2d(1), target);
        let constraint_21 = DistanceConstraint::new(ConstraintId(1), pr2d(1), pr2d(0), target);

        let residuals_12 = constraint_12.residuals(&params);
        let residuals_21 = constraint_21.residuals(&params);

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
        let dist = ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
        if dist < 0.01 {
            return Ok(());
        }

        let mut system = ConstraintSystem::new();
        let h1 = system.add_point_2d(p1.0, p1.1);
        let h2 = system.add_point_2d(p2.0, p2.1);
        let id = system.next_constraint_id();
        system.add_constraint(Box::new(DistanceConstraint::new(id, h1.params, h2.params, target)));

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
        // Flat params: [p.x, p.y, p.x, p.y]
        let params = vec![p.0, p.1, p.0, p.1];
        let constraint = CoincidentConstraint::new(ConstraintId(0), pr2d(0), pr2d(1));

        let residuals = constraint.residuals(&params);

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
        let mut system = ConstraintSystem::new();
        let h1 = system.add_point_2d(p1.0, p1.1);
        let h2 = system.add_point_2d(p2.0, p2.1);
        let id = system.next_constraint_id();
        system.add_constraint(Box::new(CoincidentConstraint::new(id, h1.params, h2.params)));

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
        // Flat params: [x1, y, x2, y]
        let params = vec![x1, y, x2, y];
        let constraint = HorizontalConstraint::new(ConstraintId(0), pr2d(0), pr2d(1));

        let residuals = constraint.residuals(&params);

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
        // Flat params: [x, y1, x, y2]
        let params = vec![x, y1, x, y2];
        let constraint = VerticalConstraint::new(ConstraintId(0), pr2d(0), pr2d(1));

        let residuals = constraint.residuals(&params);

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
        let mid_x = (p1.0 + p2.0) / 2.0;
        let mid_y = (p1.1 + p2.1) / 2.0;
        // Flat params: [mid.x, mid.y, p1.x, p1.y, p2.x, p2.y]
        let params = vec![mid_x, mid_y, p1.0, p1.1, p2.0, p2.1];

        let constraint = MidpointConstraint::new(ConstraintId(0), pr2d(0), pr2d(1), pr2d(2));
        let residuals = constraint.residuals(&params);

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
        // Flat params: [target.x, target.y]
        let params = vec![target.0, target.1];
        let constraint = FixedConstraint::new(ConstraintId(0), pr2d(0), vec![target.0, target.1]);

        let residuals = constraint.residuals(&params);

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
        let p2_x = 2.0 * center.0 - p1.0;
        let p2_y = 2.0 * center.1 - p1.1;
        // Flat params: [p1.x, p1.y, p2.x, p2.y, center.x, center.y]
        let params = vec![p1.0, p1.1, p2_x, p2_y, center.0, center.1];

        let constraint = SymmetricConstraint::new(ConstraintId(0), pr2d(0), pr2d(1), pr2d(2));
        let residuals = constraint.residuals(&params);

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
        let p2_x = p1.0 + length * angle1.cos();
        let p2_y = p1.1 + length * angle1.sin();
        let p3_x = 0.0;
        let p3_y = 0.0;
        let p4_x = length * angle2.cos();
        let p4_y = length * angle2.sin();

        // Flat params: [p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y]
        let params = vec![p1.0, p1.1, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y];
        let constraint = EqualLengthConstraint::from_points(
            ConstraintId(0), pr2d(0), pr2d(1), pr2d(2), pr2d(3),
        );

        let residuals = constraint.residuals(&params);

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

        let p2_x = p1.0 + direction_x;
        let p2_y = p1.1 + direction_y;

        // Second line is offset but parallel
        let p3_x = p1.0 + offset;
        let p3_y = p1.1 + offset;
        let p4_x = p3_x + direction_x;
        let p4_y = p3_y + direction_y;

        // Parallel constraint takes Line2D entities (4 params each: [x1, y1, x2, y2])
        let params = vec![p1.0, p1.1, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y];
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint = ParallelConstraint::new(ConstraintId(0), line1, line2);

        let residuals = constraint.residuals(&params);

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

        let p2_x = p1.0 + direction_x;
        let p2_y = p1.1 + direction_y;

        // Perpendicular direction: rotate 90 degrees
        let perp_x = -direction_y;
        let perp_y = direction_x;

        let p3_x = 0.0;
        let p3_y = 0.0;
        let p4_x = perp_x;
        let p4_y = perp_y;

        // Perpendicular constraint takes Line2D entities (4 params each: [x1, y1, x2, y2])
        let params = vec![p1.0, p1.1, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y];
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint = PerpendicularConstraint::new(ConstraintId(0), line1, line2);

        let residuals = constraint.residuals(&params);

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
        let dist = ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
        if dist < 0.01 {
            return Ok(());
        }

        // Point on line: p = p1 + t*(p2 - p1)
        let point_x = p1.0 + t * (p2.0 - p1.0);
        let point_y = p1.1 + t * (p2.1 - p1.1);

        // Flat params: [point.x, point.y, line_start.x, line_start.y, line_end.x, line_end.y]
        let params = vec![point_x, point_y, p1.0, p1.1, p2.0, p2.1];
        let constraint = PointOnLineConstraint::new(
            ConstraintId(0), pr2d(0), pr2d(1), pr2d(2),
        );

        let residuals = constraint.residuals(&params);

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

        let system = ConstraintSystemBuilder::new()
            .point_2d(0.0, 0.0)
            .point_2d(side1, 0.0)
            .point_2d(side1 / 2.0, side2 / 2.0) // Initial guess
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
        let mut system = ConstraintSystem::new();

        // Add free points and collect handles
        let mut handles = Vec::new();
        for i in 0..num_free_points {
            let h = system.add_point_2d(i as f64, 0.0);
            handles.push(h);
        }

        // Add fixed points
        for i in 0..num_fixed_points {
            let h = system.add_point_2d(100.0 + i as f64, 0.0);
            system.fix_entity(&h);
            handles.push(h);
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
            let id = system.next_constraint_id();
            system.add_constraint(Box::new(DistanceConstraint::new(
                id, handles[i].params, handles[i + 1].params, 1.0,
            )));
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
