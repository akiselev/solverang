# Plan 08: Contract / Design-by-Contract Testing

## Goal

The `Problem` trait is the core abstraction in solverang. Every solver trusts that
`Problem` implementations satisfy implicit contracts: dimensions are consistent,
Jacobian indices are in-bounds, values are finite, etc. If a new `Problem`
implementation violates these contracts, the solver may panic, produce wrong results,
or silently corrupt data.

Contract testing makes these implicit assumptions explicit and verifiable. It catches
bugs in new Problem implementations immediately, before they reach the solver.

## Contract Specification

### `Problem` Trait Contracts

| # | Contract | Currently Verified? |
|---|----------|-------------------|
| P1 | `residual_count() > 0` | No |
| P2 | `variable_count() > 0` | No |
| P3 | `residuals(x).len() == residual_count()` for all valid `x` | No |
| P4 | `jacobian(x)` entries have `row < residual_count()` | No (silently dropped) |
| P5 | `jacobian(x)` entries have `col < variable_count()` | No (silently dropped) |
| P6 | `jacobian(x)` entries have finite values | Partially (NonFiniteJacobian check) |
| P7 | `initial_point(factor).len() == variable_count()` | No |
| P8 | `residual_count()` is consistent across calls | No |
| P9 | `variable_count()` is consistent across calls | No |
| P10 | `residuals(x)` is deterministic (same x → same result) | No |
| P11 | `jacobian(x)` is deterministic | No |
| P12 | `residuals(x)` values are finite for finite `x` (soft) | No |
| P13 | `residual_norm(x) >= 0.0` for all `x` | No (default impl) |

### `GeometricConstraint` Trait Contracts

| # | Contract | Currently Verified? |
|---|----------|-------------------|
| G1 | `residual_count() > 0` | No |
| G2 | `point_indices()` are valid (within system's point count) | No |
| G3 | `residuals(points).len() == residual_count()` | No |
| G4 | `jacobian_entries()` indices are within bounds | No |
| G5 | `jacobian_entries()` values are finite | No |

## Contract Validator: `Problem`

```rust
use std::collections::HashSet;

/// Violations found during contract validation.
#[derive(Debug)]
pub struct ContractViolation {
    pub contract: String,
    pub message: String,
}

/// Validate all implicit contracts of a Problem implementation.
///
/// Returns a list of violations. Empty list means all contracts satisfied.
pub fn validate_problem<P: Problem>(
    problem: &P,
    test_points: &[Vec<f64>],
) -> Vec<ContractViolation> {
    let mut violations = Vec::new();
    let m = problem.residual_count();
    let n = problem.variable_count();

    // P1: residual_count > 0
    if m == 0 {
        violations.push(ContractViolation {
            contract: "P1".into(),
            message: "residual_count() returned 0".into(),
        });
    }

    // P2: variable_count > 0
    if n == 0 {
        violations.push(ContractViolation {
            contract: "P2".into(),
            message: "variable_count() returned 0".into(),
        });
    }

    // P7: initial_point length
    for factor in &[0.0, 1.0, -1.0, 10.0] {
        let x0 = problem.initial_point(*factor);
        if x0.len() != n {
            violations.push(ContractViolation {
                contract: "P7".into(),
                message: format!(
                    "initial_point({}) returned {} elements, expected {}",
                    factor, x0.len(), n
                ),
            });
        }
    }

    // P8, P9: Consistency of counts across calls
    for _ in 0..5 {
        if problem.residual_count() != m {
            violations.push(ContractViolation {
                contract: "P8".into(),
                message: "residual_count() returned different values across calls".into(),
            });
            break;
        }
        if problem.variable_count() != n {
            violations.push(ContractViolation {
                contract: "P9".into(),
                message: "variable_count() returned different values across calls".into(),
            });
            break;
        }
    }

    // Test at multiple points
    let default_points = vec![
        vec![0.0; n],
        vec![1.0; n],
        vec![-1.0; n],
        problem.initial_point(1.0),
    ];
    let points = if test_points.is_empty() { &default_points } else { test_points };

    for (pi, x) in points.iter().enumerate() {
        if x.len() != n { continue; }

        // P3: residuals length
        let r = problem.residuals(x);
        if r.len() != m {
            violations.push(ContractViolation {
                contract: "P3".into(),
                message: format!(
                    "residuals() at point {} returned {} values, expected {}",
                    pi, r.len(), m
                ),
            });
        }

        // P12: finite residuals (soft contract)
        if r.iter().any(|v| !v.is_finite()) {
            violations.push(ContractViolation {
                contract: "P12".into(),
                message: format!(
                    "residuals() at point {} contains non-finite values: {:?}",
                    pi,
                    r.iter().enumerate().filter(|(_, v)| !v.is_finite()).collect::<Vec<_>>()
                ),
            });
        }

        // P4, P5, P6: Jacobian entries
        let j = problem.jacobian(x);
        for (idx, (row, col, val)) in j.iter().enumerate() {
            if *row >= m {
                violations.push(ContractViolation {
                    contract: "P4".into(),
                    message: format!(
                        "jacobian entry {} has row={} >= residual_count={}",
                        idx, row, m
                    ),
                });
            }
            if *col >= n {
                violations.push(ContractViolation {
                    contract: "P5".into(),
                    message: format!(
                        "jacobian entry {} has col={} >= variable_count={}",
                        idx, col, n
                    ),
                });
            }
            if !val.is_finite() {
                violations.push(ContractViolation {
                    contract: "P6".into(),
                    message: format!(
                        "jacobian entry {} at ({},{}) has non-finite value {}",
                        idx, row, col, val
                    ),
                });
            }
        }

        // P10: determinism — call residuals twice with same input
        let r2 = problem.residuals(x);
        if r.len() == r2.len() {
            for (i, (a, b)) in r.iter().zip(&r2).enumerate() {
                if a.to_bits() != b.to_bits() {
                    violations.push(ContractViolation {
                        contract: "P10".into(),
                        message: format!(
                            "residuals() non-deterministic at index {}: {} vs {}",
                            i, a, b
                        ),
                    });
                    break;
                }
            }
        }

        // P11: Jacobian determinism
        let j2 = problem.jacobian(x);
        if j.len() != j2.len() {
            violations.push(ContractViolation {
                contract: "P11".into(),
                message: "jacobian() returned different number of entries".into(),
            });
        }

        // P13: residual_norm non-negative
        let norm = problem.residual_norm(x);
        if norm < 0.0 || norm.is_nan() {
            violations.push(ContractViolation {
                contract: "P13".into(),
                message: format!("residual_norm() returned {}", norm),
            });
        }
    }

    violations
}
```

## Contract Validator: `GeometricConstraint`

```rust
pub fn validate_geometric_constraint<C: GeometricConstraint<D>, const D: usize>(
    constraint: &C,
    test_points: &[Vec<Point<D>>],
) -> Vec<ContractViolation> {
    let mut violations = Vec::new();

    // G1: residual_count > 0
    if constraint.residual_count() == 0 {
        violations.push(ContractViolation {
            contract: "G1".into(),
            message: "residual_count() returned 0".into(),
        });
    }

    for points in test_points {
        // G3: residuals length
        let r = constraint.residuals(points);
        if r.len() != constraint.residual_count() {
            violations.push(ContractViolation {
                contract: "G3".into(),
                message: format!(
                    "residuals() returned {} values, expected {}",
                    r.len(), constraint.residual_count()
                ),
            });
        }

        // G4, G5: Jacobian entries
        let j_entries = constraint.jacobian_entries(points);
        for (idx, (row, col, val)) in j_entries.iter().enumerate() {
            if !val.is_finite() {
                violations.push(ContractViolation {
                    contract: "G5".into(),
                    message: format!("jacobian entry {} has non-finite value {}", idx, val),
                });
            }
        }
    }

    violations
}
```

## Applying to All Test Problems

```rust
use solverang::test_problems::*;

#[test]
fn all_test_problems_satisfy_contracts() {
    let problems: Vec<Box<dyn Problem>> = vec![
        Box::new(Rosenbrock::new()),
        Box::new(Powell::new()),
        Box::new(BroydenTridiagonal::new(10)),
        Box::new(Watson::new()),
        Box::new(HelicalValley::new()),
        Box::new(Box3D::new()),
        // ... all 50+ test problems
    ];

    let mut total_violations = 0;

    for problem in &problems {
        let violations = validate_problem(problem.as_ref(), &[]);
        if !violations.is_empty() {
            eprintln!("Contract violations in {}:", problem.name());
            for v in &violations {
                eprintln!("  [{}] {}", v.contract, v.message);
            }
            total_violations += violations.len();
        }
    }

    assert_eq!(
        total_violations, 0,
        "Found {} contract violations across test problems",
        total_violations
    );
}

#[cfg(feature = "nist")]
#[test]
fn all_nist_problems_satisfy_contracts() {
    let problems = all_nist_problems();  // Iterator over all 30+ NIST problems
    for problem in problems {
        let violations = validate_problem(&problem, &[]);
        assert!(
            violations.is_empty(),
            "NIST problem {} has {} violations: {:?}",
            problem.name(),
            violations.len(),
            violations
        );
    }
}
```

## Debug Assertions in Solver Hot Paths

Add `debug_assert!` calls in solver code that verify contracts at runtime (only in
debug builds, zero cost in release):

```rust
// In Solver::solve()
pub fn solve(&self, problem: &dyn Problem, x0: &[f64]) -> SolveResult {
    debug_assert!(
        problem.residual_count() > 0,
        "Problem has 0 residuals"
    );
    debug_assert!(
        problem.variable_count() > 0,
        "Problem has 0 variables"
    );
    debug_assert_eq!(
        x0.len(),
        problem.variable_count(),
        "Initial point length {} != variable_count {}",
        x0.len(),
        problem.variable_count()
    );

    // ... solver logic ...

    // Inside iteration loop:
    let r = problem.residuals(&x);
    debug_assert_eq!(
        r.len(),
        problem.residual_count(),
        "residuals() returned {} values, expected {}",
        r.len(),
        problem.residual_count()
    );
}
```

## Integration with Proptest

Generate random Problem implementations and validate contracts:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_random_linear_problems_satisfy_contracts(
        n in 1usize..20,
        target in prop::collection::vec(-1000.0f64..1000.0, 1..20),
    ) {
        if target.len() != n { return Ok(()); }

        let problem = LinearProblem { target };
        let violations = validate_problem(&problem, &[]);
        prop_assert!(
            violations.is_empty(),
            "Violations: {:?}",
            violations
        );
    }
}
```

## Error Reporting

The `ContractViolation` struct provides clear, actionable messages:

```
Contract violations in custom-problem:
  [P3] residuals() at point 0 returned 3 values, expected 2
  [P4] jacobian entry 5 has row=2 >= residual_count=2
  [P6] jacobian entry 7 at (1,0) has non-finite value NaN
```

## Public API Consideration

The contract validator could be exposed as a public API for library users:

```rust
// In lib.rs
pub mod validation {
    pub use crate::contract::{validate_problem, ContractViolation};
}
```

This lets users validate their own Problem implementations during development.

## File Organization

```
crates/solverang/src/
├── contract.rs                  # Contract validator implementations
└── lib.rs                       # pub mod contract; (or pub mod validation;)

crates/solverang/tests/
└── contract_tests.rs            # Apply validators to all test problems
```

## Estimated Effort

| Task | Time |
|------|------|
| Write `validate_problem()` | 2-3 hours |
| Write `validate_geometric_constraint()` | 1-2 hours |
| Apply to all test problems | 1-2 hours |
| Add debug assertions to solvers | 1-2 hours |
| Property-based contract tests | 1 hour |
| Public API design | 1 hour |
| **Total** | **~8-12 hours** |
