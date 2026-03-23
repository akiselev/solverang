//! Integration tests for the expanded `#[auto_diff]` macro functions.
//!
//! Verifies that asin, acos, sinh, cosh, tanh generate correct gradients
//! using finite-difference verification.

#![cfg(feature = "macros")]

use solverang::{auto_diff, objective};

const FD_H: f64 = 1e-6;
const FD_TOL: f64 = 1e-5;

/// Compute finite-difference gradient for a scalar function at point x.
fn fd_gradient(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let fx = f(x);
    let mut grad = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        xp[i] += FD_H;
        grad[i] = (f(&xp) - fx) / FD_H;
    }
    grad
}

// =========================================================================
// sinh objective: f(x) = sinh(x[0])
// =========================================================================

struct SinhObjective;

#[auto_diff(array_param = "x")]
impl SinhObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0].sinh()
    }
}

#[test]
fn test_sinh_gradient_matches_fd() {
    let obj = SinhObjective;
    // Skip x=0 since d(sinh(0))/dx = cosh(0) = 1.0 is not zero, but verify it anyway
    for &xval in &[0.0_f64, 0.5, -0.5, 1.0, -1.0] {
        let x = vec![xval];
        let entries = obj.gradient_entries(&x);
        // d(sinh(x))/dx = cosh(x), which is always >= 1, so never filtered
        assert_eq!(entries.len(), 1, "expected 1 entry at x={}", xval);
        let (idx, grad) = entries[0];
        assert_eq!(idx, 0);

        let fd = fd_gradient(|x| x[0].sinh(), &x);
        assert!(
            (grad - fd[0]).abs() < FD_TOL,
            "sinh gradient mismatch at x={}: got {}, fd={}",
            xval, grad, fd[0]
        );
    }
}

// =========================================================================
// cosh objective: f(x) = cosh(x[0])
// =========================================================================

struct CoshObjective;

#[auto_diff(array_param = "x")]
impl CoshObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0].cosh()
    }
}

#[test]
fn test_cosh_gradient_matches_fd() {
    let obj = CoshObjective;
    // Skip x=0 since d(cosh(0))/dx = sinh(0) = 0 is filtered by the sparsity threshold
    for &xval in &[0.5_f64, -0.5, 1.0, -1.0] {
        let x = vec![xval];
        let entries = obj.gradient_entries(&x);
        assert_eq!(entries.len(), 1, "expected 1 entry at x={}", xval);
        let (idx, grad) = entries[0];
        assert_eq!(idx, 0);

        let fd = fd_gradient(|x| x[0].cosh(), &x);
        assert!(
            (grad - fd[0]).abs() < FD_TOL,
            "cosh gradient mismatch at x={}: got {}, fd={}",
            xval, grad, fd[0]
        );
    }
}

// =========================================================================
// tanh objective: f(x) = tanh(x[0])
// =========================================================================

struct TanhObjective;

#[auto_diff(array_param = "x")]
impl TanhObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0].tanh()
    }
}

#[test]
fn test_tanh_gradient_matches_fd() {
    let obj = TanhObjective;
    // d(tanh(x))/dx = 1/cosh^2(x), which at x=0 equals 1.0 — never filtered
    for &xval in &[0.0_f64, 0.5, -0.5, 1.0, -1.0] {
        let x = vec![xval];
        let entries = obj.gradient_entries(&x);
        assert_eq!(entries.len(), 1, "expected 1 entry at x={}", xval);
        let (idx, grad) = entries[0];
        assert_eq!(idx, 0);

        let fd = fd_gradient(|x| x[0].tanh(), &x);
        assert!(
            (grad - fd[0]).abs() < FD_TOL,
            "tanh gradient mismatch at x={}: got {}, fd={}",
            xval, grad, fd[0]
        );
    }
}

// =========================================================================
// asin objective: f(x) = asin(x[0])
// =========================================================================

struct AsinObjective;

#[auto_diff(array_param = "x")]
impl AsinObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0].asin()
    }
}

#[test]
fn test_asin_gradient_matches_fd() {
    let obj = AsinObjective;
    for &xval in &[0.0_f64, 0.3, -0.3, 0.7, -0.7] {
        let x = vec![xval];
        let entries = obj.gradient_entries(&x);
        assert_eq!(entries.len(), 1);
        let (idx, grad) = entries[0];
        assert_eq!(idx, 0);

        let fd = fd_gradient(|x| x[0].asin(), &x);
        assert!(
            (grad - fd[0]).abs() < FD_TOL,
            "asin gradient mismatch at x={}: got {}, fd={}",
            xval, grad, fd[0]
        );
    }
}

// =========================================================================
// acos objective: f(x) = acos(x[0])
// =========================================================================

struct AcosObjective;

#[auto_diff(array_param = "x")]
impl AcosObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0].acos()
    }
}

#[test]
fn test_acos_gradient_matches_fd() {
    let obj = AcosObjective;
    for &xval in &[0.0_f64, 0.3, -0.3, 0.7, -0.7] {
        let x = vec![xval];
        let entries = obj.gradient_entries(&x);
        assert_eq!(entries.len(), 1);
        let (idx, grad) = entries[0];
        assert_eq!(idx, 0);

        let fd = fd_gradient(|x| x[0].acos(), &x);
        assert!(
            (grad - fd[0]).abs() < FD_TOL,
            "acos gradient mismatch at x={}: got {}, fd={}",
            xval, grad, fd[0]
        );
    }
}

// =========================================================================
// Composed chain rule: f(x) = sinh(x[0]).acos() (if in range)
// Actually: f(x) = sinh(x[0]).tanh() to avoid domain issues
// =========================================================================

struct SinhTanhObjective;

#[auto_diff(array_param = "x")]
impl SinhTanhObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0].sinh().tanh()
    }
}

#[test]
fn test_sinh_tanh_composed_gradient_matches_fd() {
    let obj = SinhTanhObjective;
    for &xval in &[0.0_f64, 0.3, -0.3, 0.8, -0.8] {
        let x = vec![xval];
        let entries = obj.gradient_entries(&x);
        assert_eq!(entries.len(), 1);
        let (idx, grad) = entries[0];
        assert_eq!(idx, 0);

        let fd = fd_gradient(|x| x[0].sinh().tanh(), &x);
        assert!(
            (grad - fd[0]).abs() < FD_TOL,
            "sinh.tanh() gradient mismatch at x={}: got {}, fd={}",
            xval, grad, fd[0]
        );
    }
}

// =========================================================================
// Constant folding: asin(0.0) should simplify to a constant
// =========================================================================

struct AsinConstObjective;

#[auto_diff(array_param = "x")]
impl AsinConstObjective {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0] + 0.0_f64.asin()
    }
}

#[test]
fn test_asin_const_simplification() {
    let obj = AsinConstObjective;
    let x = vec![1.0];
    let entries = obj.gradient_entries(&x);
    // d/dx[0] (x[0] + asin(0.0)) = 1.0
    assert_eq!(entries.len(), 1);
    let (idx, grad) = entries[0];
    assert_eq!(idx, 0);
    assert!((grad - 1.0).abs() < 1e-10, "gradient should be 1.0, got {}", grad);
}
