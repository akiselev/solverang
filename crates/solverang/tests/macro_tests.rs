//! Tests for the automatic Jacobian generation macros.
//!
//! These tests verify that the `#[auto_jacobian]` macro correctly generates
//! Jacobian implementations from residual expressions.

#![cfg(feature = "macros")]

use solverang::{auto_jacobian, verify_jacobian, Problem};

/// Simple quadratic constraint: x^2 - target = 0
struct QuadraticConstraint {
    target: f64,
}

#[auto_jacobian(array_param = "x")]
impl QuadraticConstraint {
    #[residual]
    fn residual(&self, x: &[f64]) -> f64 {
        x[0] * x[0] - self.target
    }
}

impl Problem for QuadraticConstraint {
    fn name(&self) -> &str {
        "Quadratic"
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_count(&self) -> usize {
        1
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![self.residual(x)]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_entries(x)
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor]
    }
}

#[test]
fn test_quadratic_jacobian() {
    let constraint = QuadraticConstraint { target: 4.0 };

    // At x = 3, residual = 9 - 4 = 5
    let x = &[3.0];
    let residual = constraint.residual(x);
    assert!((residual - 5.0).abs() < 1e-10);

    // d/dx(x^2 - target) = 2x = 6 at x = 3
    let jacobian = constraint.jacobian(x);
    assert_eq!(jacobian.len(), 1);
    assert_eq!(jacobian[0].0, 0); // row 0
    assert_eq!(jacobian[0].1, 0); // column 0
    assert!((jacobian[0].2 - 6.0).abs() < 1e-10);
}

#[test]
fn test_quadratic_jacobian_verification() {
    let constraint = QuadraticConstraint { target: 4.0 };
    let x = vec![3.0];

    let result = verify_jacobian(&constraint, &x, 1e-7, 1e-5);
    assert!(
        result.passed,
        "Jacobian verification failed: max error = {}",
        result.max_absolute_error
    );
}

/// Distance constraint: sqrt((x1-x0)^2 + (y1-y0)^2) - target = 0
struct DistanceConstraint2D {
    /// Target distance
    target: f64,
}

#[auto_jacobian(array_param = "x")]
impl DistanceConstraint2D {
    #[residual]
    fn residual(&self, x: &[f64]) -> f64 {
        let dx = x[2] - x[0];
        let dy = x[3] - x[1];
        (dx * dx + dy * dy).sqrt() - self.target
    }
}

impl Problem for DistanceConstraint2D {
    fn name(&self) -> &str {
        "Distance2D"
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_count(&self) -> usize {
        4
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![self.residual(x)]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_entries(x)
    }

    fn initial_point(&self, _: f64) -> Vec<f64> {
        vec![0.0, 0.0, 1.0, 0.0]
    }
}

#[test]
fn test_distance_jacobian() {
    let constraint = DistanceConstraint2D { target: 5.0 };

    // Points: (0, 0) and (3, 4) -> distance = 5
    let x = &[0.0, 0.0, 3.0, 4.0];
    let residual = constraint.residual(x);
    assert!(residual.abs() < 1e-10);

    // Jacobian: d/dx0 = -dx/dist = -3/5 = -0.6
    //           d/dy0 = -dy/dist = -4/5 = -0.8
    //           d/dx1 = dx/dist = 3/5 = 0.6
    //           d/dy1 = dy/dist = 4/5 = 0.8
    let jacobian = constraint.jacobian(x);
    assert_eq!(jacobian.len(), 4);

    // Sort by column index for deterministic comparison
    let mut jacobian = jacobian;
    jacobian.sort_by_key(|(_, col, _)| *col);

    assert_eq!(jacobian[0].1, 0); // x0
    assert!((jacobian[0].2 - (-0.6)).abs() < 1e-10);

    assert_eq!(jacobian[1].1, 1); // y0
    assert!((jacobian[1].2 - (-0.8)).abs() < 1e-10);

    assert_eq!(jacobian[2].1, 2); // x1
    assert!((jacobian[2].2 - 0.6).abs() < 1e-10);

    assert_eq!(jacobian[3].1, 3); // y1
    assert!((jacobian[3].2 - 0.8).abs() < 1e-10);
}

#[test]
fn test_distance_jacobian_verification() {
    let constraint = DistanceConstraint2D { target: 5.0 };
    let x = vec![0.0, 0.0, 3.0, 4.0];

    let result = verify_jacobian(&constraint, &x, 1e-7, 1e-5);
    assert!(
        result.passed,
        "Jacobian verification failed: max error = {}, location = {:?}",
        result.max_absolute_error, result.max_error_location
    );
}

/// Trigonometric constraint: sin(x) + cos(y) - target = 0
struct TrigConstraint {
    target: f64,
}

#[auto_jacobian(array_param = "x")]
impl TrigConstraint {
    #[residual]
    fn residual(&self, x: &[f64]) -> f64 {
        x[0].sin() + x[1].cos() - self.target
    }
}

impl Problem for TrigConstraint {
    fn name(&self) -> &str {
        "Trig"
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_count(&self) -> usize {
        2
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![self.residual(x)]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_entries(x)
    }

    fn initial_point(&self, _: f64) -> Vec<f64> {
        vec![0.5, 0.5]
    }
}

#[test]
fn test_trig_jacobian() {
    let pi = std::f64::consts::PI;
    let constraint = TrigConstraint { target: 0.5 };

    // At x = pi/6, sin(x) = 0.5; at y = pi/3, cos(y) = 0.5
    let x = &[pi / 6.0, pi / 3.0];
    let residual = constraint.residual(x);
    assert!((residual - 0.5).abs() < 1e-10);

    // d/dx = cos(x) = cos(pi/6) = sqrt(3)/2 ~ 0.866
    // d/dy = -sin(y) = -sin(pi/3) = -sqrt(3)/2 ~ -0.866
    let jacobian = constraint.jacobian(x);
    assert_eq!(jacobian.len(), 2);

    let expected_cos = (3.0_f64).sqrt() / 2.0;

    let mut jacobian = jacobian;
    jacobian.sort_by_key(|(_, col, _)| *col);

    assert_eq!(jacobian[0].1, 0);
    assert!(
        (jacobian[0].2 - expected_cos).abs() < 1e-10,
        "d/dx expected {} got {}",
        expected_cos,
        jacobian[0].2
    );

    assert_eq!(jacobian[1].1, 1);
    assert!(
        (jacobian[1].2 - (-expected_cos)).abs() < 1e-10,
        "d/dy expected {} got {}",
        -expected_cos,
        jacobian[1].2
    );
}

#[test]
fn test_trig_jacobian_verification() {
    let constraint = TrigConstraint { target: 0.5 };
    let x = vec![0.5, 0.5];

    let result = verify_jacobian(&constraint, &x, 1e-7, 1e-5);
    assert!(
        result.passed,
        "Jacobian verification failed: max error = {}",
        result.max_absolute_error
    );
}

/// Division constraint: x / y - target = 0
struct DivisionConstraint {
    target: f64,
}

#[auto_jacobian(array_param = "x")]
impl DivisionConstraint {
    #[residual]
    fn residual(&self, x: &[f64]) -> f64 {
        x[0] / x[1] - self.target
    }
}

impl Problem for DivisionConstraint {
    fn name(&self) -> &str {
        "Division"
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_count(&self) -> usize {
        2
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![self.residual(x)]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_entries(x)
    }

    fn initial_point(&self, _: f64) -> Vec<f64> {
        vec![1.0, 1.0]
    }
}

#[test]
fn test_division_jacobian() {
    let constraint = DivisionConstraint { target: 2.0 };

    // At x = 6, y = 3: x/y = 2
    let x = &[6.0, 3.0];
    let residual = constraint.residual(x);
    assert!(residual.abs() < 1e-10);

    // d/dx = 1/y = 1/3
    // d/dy = -x/y^2 = -6/9 = -2/3
    let jacobian = constraint.jacobian(x);
    assert_eq!(jacobian.len(), 2);

    let mut jacobian = jacobian;
    jacobian.sort_by_key(|(_, col, _)| *col);

    assert_eq!(jacobian[0].1, 0);
    assert!(
        (jacobian[0].2 - (1.0 / 3.0)).abs() < 1e-10,
        "d/dx expected {} got {}",
        1.0 / 3.0,
        jacobian[0].2
    );

    assert_eq!(jacobian[1].1, 1);
    assert!(
        (jacobian[1].2 - (-2.0 / 3.0)).abs() < 1e-10,
        "d/dy expected {} got {}",
        -2.0 / 3.0,
        jacobian[1].2
    );
}

#[test]
fn test_division_jacobian_verification() {
    let constraint = DivisionConstraint { target: 2.0 };
    let x = vec![6.0, 3.0];

    let result = verify_jacobian(&constraint, &x, 1e-7, 1e-5);
    assert!(
        result.passed,
        "Jacobian verification failed: max error = {}",
        result.max_absolute_error
    );
}

/// Power constraint: x^3 - target = 0
struct PowerConstraint {
    target: f64,
}

#[auto_jacobian(array_param = "x")]
impl PowerConstraint {
    #[residual]
    fn residual(&self, x: &[f64]) -> f64 {
        x[0].powf(3.0) - self.target
    }
}

impl Problem for PowerConstraint {
    fn name(&self) -> &str {
        "Power"
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn variable_count(&self) -> usize {
        1
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![self.residual(x)]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_entries(x)
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![factor]
    }
}

#[test]
fn test_power_jacobian() {
    let constraint = PowerConstraint { target: 8.0 };

    // At x = 2, x^3 = 8
    let x = &[2.0];
    let residual = constraint.residual(x);
    assert!(residual.abs() < 1e-10);

    // d/dx(x^3) = 3x^2 = 12 at x = 2
    let jacobian = constraint.jacobian(x);
    assert_eq!(jacobian.len(), 1);
    assert_eq!(jacobian[0].1, 0);
    assert!(
        (jacobian[0].2 - 12.0).abs() < 1e-10,
        "d/dx expected 12 got {}",
        jacobian[0].2
    );
}

#[test]
fn test_power_jacobian_verification() {
    let constraint = PowerConstraint { target: 8.0 };
    let x = vec![2.0];

    let result = verify_jacobian(&constraint, &x, 1e-7, 1e-5);
    assert!(
        result.passed,
        "Jacobian verification failed: max error = {}",
        result.max_absolute_error
    );
}

// ============================================================================
// Multi-residual tests
// ============================================================================

/// Rosenbrock function via macro: 2 residuals, 2 variables.
///
/// F_0(x) = 10(x[1] - x[0]^2)
/// F_1(x) = 1 - x[0]
struct RosenbrockMacro;

#[auto_jacobian(array_param = "x")]
impl RosenbrockMacro {
    #[residual]
    fn residual_0(&self, x: &[f64]) -> f64 {
        10.0 * (x[1] - x[0] * x[0])
    }

    #[residual]
    fn residual_1(&self, x: &[f64]) -> f64 {
        1.0 - x[0]
    }
}

impl Problem for RosenbrockMacro {
    fn name(&self) -> &str {
        "RosenbrockMacro"
    }

    fn residual_count(&self) -> usize {
        2
    }

    fn variable_count(&self) -> usize {
        2
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![self.residual_0(x), self.residual_1(x)]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_entries(x)
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![-1.2 * factor, factor]
    }
}

#[test]
fn test_rosenbrock_macro_residuals() {
    let problem = RosenbrockMacro;

    // At solution (1, 1): both residuals should be 0
    let r = problem.residuals(&[1.0, 1.0]);
    assert!((r[0]).abs() < 1e-10, "F0 at solution should be 0, got {}", r[0]);
    assert!((r[1]).abs() < 1e-10, "F1 at solution should be 0, got {}", r[1]);

    // At (0, 0): F0 = 0, F1 = 1
    let r = problem.residuals(&[0.0, 0.0]);
    assert!((r[0]).abs() < 1e-10);
    assert!((r[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_rosenbrock_macro_jacobian_entries() {
    let problem = RosenbrockMacro;
    let x = &[1.0, 1.0];
    let jac = problem.jacobian(x);

    // Expected Jacobian at (1,1):
    //   dF0/dx0 = -20*x0 = -20    (row 0, col 0)
    //   dF0/dx1 = 10               (row 0, col 1)
    //   dF1/dx0 = -1               (row 1, col 0)
    //   dF1/dx1 = 0                (row 1, col 1) — should be absent (zero)
    //
    // So we expect 3 entries total.
    assert_eq!(
        jac.len(),
        3,
        "Expected 3 non-zero Jacobian entries, got {}: {:?}",
        jac.len(),
        jac
    );

    // Build dense for easier checking
    let mut dense = vec![vec![0.0; 2]; 2];
    for (row, col, val) in &jac {
        dense[*row][*col] = *val;
    }

    assert!((dense[0][0] - (-20.0)).abs() < 1e-10, "dF0/dx0 = -20");
    assert!((dense[0][1] - 10.0).abs() < 1e-10, "dF0/dx1 = 10");
    assert!((dense[1][0] - (-1.0)).abs() < 1e-10, "dF1/dx0 = -1");
}

#[test]
fn test_rosenbrock_macro_jacobian_verification() {
    let problem = RosenbrockMacro;
    let x = vec![0.5, 0.5];

    let result = verify_jacobian(&problem, &x, 1e-7, 1e-5);
    assert!(
        result.passed,
        "Jacobian verification failed: max error = {}, location = {:?}",
        result.max_absolute_error,
        result.max_error_location
    );
}

/// Multi-residual with disjoint variable sets.
///
/// F_0(x) = x[0] + x[1]     (uses x[0], x[1])
/// F_1(x) = x[2] * x[2]     (uses x[2] only)
struct DisjointVarsConstraint;

#[auto_jacobian(array_param = "x")]
impl DisjointVarsConstraint {
    #[residual]
    fn residual_sum(&self, x: &[f64]) -> f64 {
        x[0] + x[1]
    }

    #[residual]
    fn residual_square(&self, x: &[f64]) -> f64 {
        x[2] * x[2]
    }
}

impl Problem for DisjointVarsConstraint {
    fn name(&self) -> &str {
        "DisjointVars"
    }

    fn residual_count(&self) -> usize {
        2
    }

    fn variable_count(&self) -> usize {
        3
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![self.residual_sum(x), self.residual_square(x)]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.jacobian_entries(x)
    }

    fn initial_point(&self, _: f64) -> Vec<f64> {
        vec![1.0, 1.0, 1.0]
    }
}

#[test]
fn test_disjoint_vars_jacobian() {
    let problem = DisjointVarsConstraint;
    let x = &[2.0, 3.0, 4.0];
    let jac = problem.jacobian(x);

    // Expected Jacobian:
    //   dF0/dx0 = 1   (row 0, col 0)
    //   dF0/dx1 = 1   (row 0, col 1)
    //   dF0/dx2 = 0   — absent
    //   dF1/dx0 = 0   — absent
    //   dF1/dx1 = 0   — absent
    //   dF1/dx2 = 2*x[2] = 8  (row 1, col 2)
    //
    // 3 non-zero entries, block-diagonal structure.
    assert_eq!(jac.len(), 3, "Expected 3 entries, got {:?}", jac);

    let mut dense = vec![vec![0.0; 3]; 2];
    for (row, col, val) in &jac {
        dense[*row][*col] = *val;
    }

    assert!((dense[0][0] - 1.0).abs() < 1e-10);
    assert!((dense[0][1] - 1.0).abs() < 1e-10);
    assert!((dense[0][2]).abs() < 1e-10); // should be 0 (absent)
    assert!((dense[1][0]).abs() < 1e-10); // should be 0 (absent)
    assert!((dense[1][1]).abs() < 1e-10); // should be 0 (absent)
    assert!((dense[1][2] - 8.0).abs() < 1e-10);
}

#[test]
fn test_disjoint_vars_jacobian_verification() {
    let problem = DisjointVarsConstraint;
    let x = vec![2.0, 3.0, 4.0];

    let result = verify_jacobian(&problem, &x, 1e-7, 1e-5);
    assert!(
        result.passed,
        "Jacobian verification failed: max error = {}",
        result.max_absolute_error
    );
}

// =========================================================================
// #[auto_diff] + #[objective] tests (M6)
// =========================================================================

use solverang::auto_diff;

/// Simple quadratic objective: f(x) = (x[0] - 1)^2 + (x[1] - 2)^2
struct SimpleQuadObjective;

#[auto_diff(array_param = "x")]
impl SimpleQuadObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 2.0) * (x[1] - 2.0)
    }
}

#[test]
fn test_objective_gradient_simple_quadratic() {
    let obj = SimpleQuadObjective;
    let x = vec![3.0, 5.0];
    let grad = obj.gradient_entries(&x);

    // ∂f/∂x[0] = 2*(x[0]-1) = 2*(3-1) = 4.0
    // ∂f/∂x[1] = 2*(x[1]-2) = 2*(5-2) = 6.0
    assert_eq!(grad.len(), 2);

    let g0 = grad.iter().find(|(i, _)| *i == 0).map(|(_, v)| *v).unwrap();
    let g1 = grad.iter().find(|(i, _)| *i == 1).map(|(_, v)| *v).unwrap();
    assert!((g0 - 4.0).abs() < 1e-10, "∂f/∂x[0] = {} (expected 4.0)", g0);
    assert!((g1 - 6.0).abs() < 1e-10, "∂f/∂x[1] = {} (expected 6.0)", g1);
}

#[test]
fn test_objective_gradient_at_minimum() {
    let obj = SimpleQuadObjective;
    let x = vec![1.0, 2.0]; // at minimum
    let grad = obj.gradient_entries(&x);

    // At minimum, gradient should be (approximately) zero
    // All entries should be filtered out by the 1e-30 threshold
    for (i, v) in &grad {
        assert!(
            v.abs() < 1e-10,
            "∂f/∂x[{}] = {} (expected ~0 at minimum)",
            i,
            v
        );
    }
}

/// Rosenbrock objective for gradient verification
struct RosenbrockObjective;

#[auto_diff(array_param = "x")]
impl RosenbrockObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        let a = x[1] - x[0] * x[0];
        let b = 1.0 - x[0];
        100.0 * a * a + b * b
    }
}

#[test]
fn test_objective_gradient_rosenbrock_fd() {
    let obj = RosenbrockObjective;

    // Verify gradient via finite differences at several points
    let test_points = vec![
        vec![-1.2, 1.0],
        vec![0.0, 0.0],
        vec![0.5, 0.5],
        vec![1.0, 1.0], // minimum
        vec![2.0, 3.0],
    ];

    let eps = 1e-7;
    for x in &test_points {
        let grad = obj.gradient_entries(x);

        for i in 0..2 {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            let fd = (obj.value(&x_plus) - obj.value(&x_minus)) / (2.0 * eps);

            let analytical = grad.iter().find(|(j, _)| *j == i).map(|(_, v)| *v).unwrap_or(0.0);

            assert!(
                (analytical - fd).abs() < 1e-4,
                "Gradient mismatch at x={:?}, var {}: analytical={}, fd={}",
                x, i, analytical, fd
            );
        }
    }
}

/// Objective with runtime constant (self.target)
struct TargetedObjective {
    target_x: f64,
    target_y: f64,
}

#[auto_diff(array_param = "x")]
impl TargetedObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        (x[0] - self.target_x) * (x[0] - self.target_x)
            + (x[1] - self.target_y) * (x[1] - self.target_y)
    }
}

#[test]
fn test_objective_gradient_with_runtime_constants() {
    let obj = TargetedObjective {
        target_x: 3.0,
        target_y: 7.0,
    };
    let x = vec![5.0, 10.0];
    let grad = obj.gradient_entries(&x);

    let g0 = grad.iter().find(|(i, _)| *i == 0).map(|(_, v)| *v).unwrap();
    let g1 = grad.iter().find(|(i, _)| *i == 1).map(|(_, v)| *v).unwrap();
    // ∂f/∂x[0] = 2*(5-3) = 4.0
    // ∂f/∂x[1] = 2*(10-7) = 6.0
    assert!((g0 - 4.0).abs() < 1e-10);
    assert!((g1 - 6.0).abs() < 1e-10);
}

#[test]
fn test_auto_jacobian_still_works() {
    // Regression: existing #[auto_jacobian] + #[residual] must still work
    let problem = QuadraticConstraint { target: 4.0 };
    let x = vec![2.0];
    let result = verify_jacobian(&problem, &x, 1e-7, 1e-5);
    assert!(result.passed, "Regression: auto_jacobian Jacobian verification failed");
}
