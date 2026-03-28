//! Tests for Hessian generation via #[auto_diff] + #[hessian]

#![cfg(feature = "macros")]

use solverang::{auto_diff, hessian, objective};

// =========================================================================
// Test 1: Quadratic with known exact Hessian
// f(x,y) = x^2 + 2*x*y + 3*y^2
// H = [[2, 2], [2, 6]]
// =========================================================================

struct QuadraticHessian;

#[auto_diff(array_param = "x")]
impl QuadraticHessian {
    #[objective]
    #[hessian]
    fn value(&self, x: &[f64]) -> f64 {
        x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1] * x[1]
    }
}

#[test]
fn quadratic_hessian_exact() {
    let q = QuadraticHessian;
    let x = [1.0, 1.0];
    let h = q.hessian_entries(&x);
    assert!(!h.is_empty(), "Quadratic should have non-empty Hessian");
    for &(i, j, v) in &h {
        match (i, j) {
            (0, 0) => assert!((v - 2.0).abs() < 1e-10, "H[0,0] = {}, expected 2", v),
            (1, 0) => assert!((v - 2.0).abs() < 1e-10, "H[1,0] = {}, expected 2", v),
            (1, 1) => assert!((v - 6.0).abs() < 1e-10, "H[1,1] = {}, expected 6", v),
            _ => panic!("Unexpected entry ({}, {}, {})", i, j, v),
        }
    }
}

// =========================================================================
// Test 2: Rosenbrock — FD verification
// f(x,y) = 100*(y - x^2)^2 + (1 - x)^2
// =========================================================================

struct RosenbrockHessian;

#[auto_diff(array_param = "x")]
impl RosenbrockHessian {
    #[objective]
    #[hessian]
    fn value(&self, x: &[f64]) -> f64 {
        100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1.0 - x[0]) * (1.0 - x[0])
    }
}

#[test]
fn rosenbrock_hessian_fd() {
    let r = RosenbrockHessian;
    let x = [1.5, 2.0];
    let h_analytic = r.hessian_entries(&x);

    let eps = 1e-5;
    let f0 = r.value(&x);

    for &(i, j, v) in &h_analytic {
        let mut x_i = x;
        let mut x_j = x;
        let mut x_ij = x;
        x_i[i] += eps;
        x_j[j] += eps;
        x_ij[i] += eps;
        x_ij[j] += eps;

        let fd = (r.value(&x_ij) - r.value(&x_i) - r.value(&x_j) + f0) / (eps * eps);
        let rel_err = if v.abs() > 1e-10 {
            (v - fd).abs() / v.abs()
        } else {
            (v - fd).abs()
        };
        assert!(
            rel_err < 1e-4,
            "H[{},{}]: analytic={}, fd={}, rel_err={}",
            i,
            j,
            v,
            fd,
            rel_err
        );
    }
}

// =========================================================================
// Test 3: Linear objective — zero Hessian (empty entries)
// f(x,y) = 3*x[0] + 2*x[1]
// =========================================================================

struct LinearHessian;

#[auto_diff(array_param = "x")]
impl LinearHessian {
    #[objective]
    #[hessian]
    fn value(&self, x: &[f64]) -> f64 {
        3.0 * x[0] + 2.0 * x[1]
    }
}

#[test]
fn linear_hessian_zero() {
    let l = LinearHessian;
    let h = l.hessian_entries(&[1.0, 1.0]);
    assert!(
        h.is_empty(),
        "Linear function should have empty Hessian, got {:?}",
        h
    );
}

// =========================================================================
// Test 4: Gradient still works without #[hessian]
// =========================================================================

struct GradientOnly;

#[auto_diff(array_param = "x")]
impl GradientOnly {
    #[objective]
    fn value(&self, x: &[f64]) -> f64 {
        x[0] * x[0]
    }
}

#[test]
fn gradient_only_no_hessian() {
    let g = GradientOnly;
    let grad = g.gradient_entries(&[3.0]);
    assert_eq!(grad.len(), 1);
    let (idx, val) = grad[0];
    assert_eq!(idx, 0);
    assert!(
        (val - 6.0).abs() < 1e-10,
        "gradient at x=3 should be 6.0, got {}",
        val
    );
}

// =========================================================================
// Test 5: Single-variable cubic — Hessian is 6*x[0]
// f(x) = x[0]^3  =>  H[0,0] = 6*x[0]
// =========================================================================

struct CubicHessian;

#[auto_diff(array_param = "x")]
impl CubicHessian {
    #[objective]
    #[hessian]
    fn value(&self, x: &[f64]) -> f64 {
        x[0] * x[0] * x[0]
    }
}

#[test]
fn cubic_hessian_single_var() {
    let c = CubicHessian;
    let xval = 2.0_f64;
    let h = c.hessian_entries(&[xval]);
    assert_eq!(h.len(), 1, "cubic should have exactly 1 Hessian entry");
    let (i, j, v) = h[0];
    assert_eq!(i, 0);
    assert_eq!(j, 0);
    let expected = 6.0 * xval;
    assert!(
        (v - expected).abs() < 1e-10,
        "H[0,0] at x={}: got {}, expected {}",
        xval,
        v,
        expected
    );
}
