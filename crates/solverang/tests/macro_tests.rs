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
