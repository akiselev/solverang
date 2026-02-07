# Plan 03: Error Path & Negative Testing

## Goal

Several error variants in solverang are either dead code or untested. Silent failure
modes (out-of-bounds Jacobian entries dropped, duplicate entries overwritten) hide bugs
in Problem implementations. Error path testing ensures every failure mode is reachable,
tested, and behaves as documented.

## Audit of Error Paths

### `SolveResult` Variants

| Variant | Where Created | Test Coverage |
|---------|--------------|---------------|
| `Converged { solution, residual_norm, iterations }` | All solvers | Well tested |
| `NotConverged { solution, residual_norm, iterations }` | All solvers | Tested |
| `Failed { error }` | All solvers | Partially tested |

### `SolveError` Variants

| Variant | Where Created | Test Coverage | Status |
|---------|--------------|---------------|--------|
| `MaxIterationsExceeded` | **Nowhere** | None | **DEAD CODE** |
| `LineSearchFailed` | `newton_raphson.rs` line search | None | **Untested** |
| `SingularJacobian` | `newton_raphson.rs` SVD step | Inline unit test only | Weak |
| `NonFiniteResiduals` | Solver pre-check | Tested | OK |
| `NonFiniteJacobian` | Solver pre-check | Tested | OK |
| `DimensionMismatch` | Various | Tested | OK |

### Silent Failure Modes

| Behavior | Location | Current Handling |
|----------|----------|-----------------|
| Jacobian entry with `row >= residual_count` | `jacobian_dense()` | Silently dropped |
| Jacobian entry with `col >= variable_count` | `jacobian_dense()` | Silently dropped |
| Duplicate `(row, col)` entries | `jacobian_dense()` | Last value wins |
| `residual_count()` changes between calls | Solver loop | Undefined behavior |
| `residuals()` returns wrong length | Solver loop | Potential index OOB panic |
| Empty Jacobian | Solver | May produce NaN |

## Dead Code Analysis: `MaxIterationsExceeded`

**Finding:** `MaxIterationsExceeded` is defined in the `SolveError` enum but is never
constructed anywhere in the codebase.

**Investigation needed:**
1. Was it intended for the Newton-Raphson solver? (NR returns `NotConverged` instead)
2. Should `NotConverged` be replaced with `Failed { MaxIterationsExceeded }` when
   max iterations is reached?
3. Or should the variant be removed?

**Recommended action:** Decide whether `NotConverged` vs `Failed(MaxIterationsExceeded)`
is the right semantic. If `NotConverged` always means "hit max iterations", the error
variant is redundant and should be removed. If there's a meaningful difference (e.g.,
`NotConverged` = "making progress but slow" vs `MaxIterationsExceeded` = "no progress"),
then wire it up.

## Test Plan for Each Error Variant

### 1. `LineSearchFailed`

**Scenario:** Create a problem where the Newton direction always increases the residual.

```rust
/// Problem where x=0 is a saddle point: Jacobian is singular there,
/// and the Newton direction from nearby points leads away from the root.
struct LineSearchTrap;

impl Problem for LineSearchTrap {
    fn name(&self) -> &str { "line-search-trap" }
    fn residual_count(&self) -> usize { 1 }
    fn variable_count(&self) -> usize { 1 }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // f(x) = x^3 — root at x=0, but Newton from x=ε diverges
        vec![x[0].powi(3)]
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 3.0 * x[0].powi(2))]
    }

    fn initial_point(&self, _: f64) -> Vec<f64> {
        vec![1e-10]  // Near the singular point
    }
}

#[test]
fn test_line_search_failure() {
    let problem = LineSearchTrap;
    let config = SolverConfig {
        max_line_search_iterations: 5,
        ..Default::default()
    };
    let solver = Solver::new(config);
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    match result {
        SolveResult::Failed { error } => {
            assert!(matches!(error, SolveError::LineSearchFailed { .. }));
        }
        _ => panic!("Expected line search failure, got {:?}", result),
    }
}
```

### 2. `SingularJacobian`

```rust
/// Problem with a zero Jacobian at the initial point.
struct SingularJacobianProblem;

impl Problem for SingularJacobianProblem {
    fn name(&self) -> &str { "singular" }
    fn residual_count(&self) -> usize { 2 }
    fn variable_count(&self) -> usize { 2 }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0], x[1] * x[1]]
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        // Always return zero Jacobian — always singular
        vec![(0, 0, 0.0), (0, 1, 0.0), (1, 0, 0.0), (1, 1, 0.0)]
    }

    fn initial_point(&self, _: f64) -> Vec<f64> { vec![1.0, 1.0] }
}

#[test]
fn test_singular_jacobian_handling() {
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&SingularJacobianProblem, &[1.0, 1.0]);
    // Should fail gracefully, not panic
    assert!(matches!(result, SolveResult::Failed { .. }));
}
```

### 3. `NonFiniteResiduals`

```rust
/// Problem that returns NaN after a certain number of evaluations.
struct NaNAfterN { threshold: usize, count: std::cell::Cell<usize> }

impl Problem for NaNAfterN {
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let n = self.count.get();
        self.count.set(n + 1);
        if n >= self.threshold {
            vec![f64::NAN; x.len()]
        } else {
            vec![x[0] - 1.0]
        }
    }
    // ...
}

#[test]
fn test_nan_residual_mid_solve() {
    let problem = NaNAfterN { threshold: 3, count: Cell::new(0) };
    let result = solver.solve(&problem, &[0.0]);
    assert!(matches!(result, SolveResult::Failed { error: SolveError::NonFiniteResiduals }));
}
```

### 4. `DimensionMismatch`

```rust
/// Problem where residuals() returns wrong length.
struct WrongLengthResiduals;

impl Problem for WrongLengthResiduals {
    fn residual_count(&self) -> usize { 2 }
    fn variable_count(&self) -> usize { 2 }
    fn residuals(&self, _x: &[f64]) -> Vec<f64> {
        vec![1.0]  // Says 2 residuals but returns 1
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 1.0)]
    }
    fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0, 0.0] }
}

#[test]
fn test_dimension_mismatch_residuals() {
    let result = solver.solve(&WrongLengthResiduals, &[0.0, 0.0]);
    assert!(matches!(result, SolveResult::Failed { error: SolveError::DimensionMismatch { .. } }));
}
```

## Silent Failure Testing

### Out-of-bounds Jacobian entries

```rust
#[test]
fn test_jacobian_oob_row_dropped() {
    struct OOBJacobian;
    impl Problem for OOBJacobian {
        fn residual_count(&self) -> usize { 2 }
        fn variable_count(&self) -> usize { 2 }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 1.0),
                (1, 1, 1.0),
                (99, 0, 999.0),  // row 99 is out of bounds
            ]
        }
        // ...
    }

    let jac = OOBJacobian.jacobian_dense(&[0.0, 0.0]);
    // Document current behavior: OOB entry is silently dropped
    assert_eq!(jac[(0, 0)], 1.0);
    assert_eq!(jac[(1, 1)], 1.0);
    // If we want to change this to an error, update this test
}
```

### Duplicate entries

```rust
#[test]
fn test_jacobian_duplicate_entries() {
    struct DuplicateJacobian;
    impl Problem for DuplicateJacobian {
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 1.0),
                (0, 0, 5.0),  // Duplicate! Which wins?
            ]
        }
        // ...
    }

    let jac = DuplicateJacobian.jacobian_dense(&[0.0]);
    // Document: last value wins (overwrite semantics)
    assert_eq!(jac[(0, 0)], 5.0);
}
```

### Inconsistent dimensions

```rust
#[test]
fn test_residual_count_changes_between_calls() {
    struct FlippingDimensions { call: Cell<usize> }
    impl Problem for FlippingDimensions {
        fn residual_count(&self) -> usize {
            if self.call.get() % 2 == 0 { 1 } else { 2 }
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            let n = self.call.get();
            self.call.set(n + 1);
            if n % 2 == 0 { vec![x[0]] } else { vec![x[0], x[0]] }
        }
        // ...
    }

    // Should either handle gracefully or panic deterministically
    let result = std::panic::catch_unwind(|| {
        solver.solve(&FlippingDimensions { call: Cell::new(0) }, &[1.0])
    });
    // Document the behavior — either Ok(Failed{..}) or Err(panic)
    assert!(result.is_ok() || result.is_err());
}
```

## Property-Based Error Testing

```rust
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Solver should never panic, even with adversarial problem implementations.
    #[test]
    fn prop_solver_handles_bad_jacobian(
        row in 0usize..100,
        col in 0usize..100,
        val in prop::num::f64::ANY,
    ) {
        struct BadJacobian { entries: Vec<(usize, usize, f64)> }
        impl Problem for BadJacobian {
            fn residual_count(&self) -> usize { 2 }
            fn variable_count(&self) -> usize { 2 }
            fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0], x[1]] }
            fn jacobian(&self, _: &[f64]) -> Vec<(usize, usize, f64)> {
                self.entries.clone()
            }
            fn initial_point(&self, _: f64) -> Vec<f64> { vec![1.0, 1.0] }
        }

        let problem = BadJacobian { entries: vec![(row, col, val)] };
        // Must not panic — any result is fine
        let _ = std::panic::catch_unwind(|| {
            let solver = LMSolver::new(LMConfig::default());
            solver.solve(&problem, &[1.0, 1.0])
        });
    }
}
```

## Suggested Code Changes

| Silent Failure | Recommended Change |
|---------------|-------------------|
| OOB Jacobian entries dropped | Add `debug_assert!` in `jacobian_dense()`, or return `Result` |
| Duplicate entries overwrite | Document as intentional (additive semantics may be better for FEM) |
| Wrong residual length | Add dimension check in solver loop, return `DimensionMismatch` |
| `MaxIterationsExceeded` dead code | Either wire up or remove the variant |

## File Organization

```
crates/solverang/tests/
├── error_path_tests.rs           # All error variant and negative tests
├── silent_failure_tests.rs       # OOB, duplicates, dimension changes
└── property_tests.rs             # Add adversarial property tests here
```

## Estimated Effort

| Task | Time |
|------|------|
| Audit all error paths (code reading) | 2 hours |
| Write error variant trigger tests | 3-4 hours |
| Write silent failure tests | 2-3 hours |
| Add property-based error tests | 2 hours |
| Decide on `MaxIterationsExceeded` fate | 30 min discussion |
| Implement code changes (if any) | 2-4 hours |
| **Total** | **~12-16 hours** |
