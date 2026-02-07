# Plan 07: Property-Based Test Expansion

## Goal

The existing `property_tests.rs` (896 lines) covers foundational invariants — Jacobian
verification, no-crash guarantees, geometric constraint satisfaction. This plan expands
property coverage to catch additional bug classes: non-determinism, idempotency
violations, monotonicity failures, composition bugs, and 3D geometry regressions.

## Gap Analysis

### What IS tested (existing)

- Jacobian matches finite differences (linear, quadratic, coupled)
- Solver doesn't crash (LM, Auto, Robust)
- Solver improves residual
- Linear problems always converge
- Geometric constraints satisfied at expected points (10 constraint types)
- Triangle convergence
- DOF calculation
- Numerical stability with large/small values

### What is NOT tested

| Property | Impact | Difficulty |
|----------|--------|------------|
| Solver idempotency | High — reveals state leaks | Low |
| Solver determinism | High — reveals race conditions | Low |
| J^T·J symmetry and PSD | Medium — core LM assumption | Medium |
| LM cost monotonicity | High — algorithm correctness | Medium |
| Constraint composition | Medium — system-level bugs | Medium |
| Residual scaling linearity | Medium — numerical stability | Low |
| Central difference Jacobian | Medium — tighter validation | Low |
| Decomposition correctness | High — parallel solver | Medium |
| 3D constraint properties | Medium — feature parity | Medium |
| Sparse pattern consistency | Medium — sparse solver | Medium |

## New Properties

### 1. Solver Idempotency

**Property:** Solving from a converged solution should return the same solution in
0-1 iterations.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_solver_idempotent(
        target in small_vec_strategy(1, 5),
    ) {
        if target.is_empty() || target.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target: target.clone() };
        let solver = LMSolver::new(LMConfig::default());
        let x0 = vec![0.0; target.len()];

        let result1 = solver.solve(&problem, &x0);
        if let SolveResult::Converged { solution, .. } = &result1 {
            let result2 = solver.solve(&problem, solution);

            match &result2 {
                SolveResult::Converged { solution: s2, iterations, .. } => {
                    // Should converge immediately (0-1 iterations)
                    prop_assert!(
                        *iterations <= 1,
                        "Idempotent solve took {} iterations",
                        iterations
                    );
                    // Same solution
                    for (a, b) in solution.iter().zip(s2) {
                        prop_assert!(
                            (a - b).abs() < 1e-10,
                            "Solution changed: {} -> {}",
                            a, b
                        );
                    }
                }
                _ => prop_assert!(false, "Second solve should converge"),
            }
        }
    }
}
```

### 2. Solver Determinism

**Property:** Same inputs should always produce bitwise identical outputs.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_solver_deterministic(
        target in small_vec_strategy(1, 4),
        x0 in small_vec_strategy(1, 4),
    ) {
        if target.len() != x0.len() || target.is_empty() {
            return Ok(());
        }
        if target.iter().chain(x0.iter()).any(|v| !v.is_finite()) {
            return Ok(());
        }

        let problem = LinearProblem { target };
        let solver = LMSolver::new(LMConfig::default());

        let result1 = solver.solve(&problem, &x0);
        let result2 = solver.solve(&problem, &x0);

        // Bitwise identical
        prop_assert_eq!(
            format!("{:?}", result1),
            format!("{:?}", result2),
            "Non-deterministic solver output"
        );
    }
}
```

### 3. J^T·J Symmetry

**Property:** For any problem, J^T·J should be symmetric positive semi-definite.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_jtj_symmetric(
        target in small_vec_strategy(2, 6),
        x in small_vec_strategy(2, 6),
    ) {
        if target.len() != x.len() || target.is_empty() {
            return Ok(());
        }

        let problem = CoupledProblem { target };
        let jac_entries = problem.jacobian(&x);

        // Build dense Jacobian
        let m = problem.residual_count();
        let n = problem.variable_count();
        let mut j = nalgebra::DMatrix::zeros(m, n);
        for (row, col, val) in &jac_entries {
            if *row < m && *col < n {
                j[(*row, *col)] = *val;
            }
        }

        // J^T * J should be symmetric
        let jtj = j.transpose() * &j;
        for i in 0..n {
            for k in i+1..n {
                prop_assert!(
                    (jtj[(i, k)] - jtj[(k, i)]).abs() < 1e-10,
                    "J^T*J not symmetric at ({},{}): {} vs {}",
                    i, k, jtj[(i, k)], jtj[(k, i)]
                );
            }
        }

        // Should be positive semi-definite (all eigenvalues ≥ 0)
        let eigenvalues = jtj.symmetric_eigenvalues();
        for (i, ev) in eigenvalues.iter().enumerate() {
            prop_assert!(
                *ev >= -1e-10,
                "J^T*J has negative eigenvalue {} at index {}",
                ev, i
            );
        }
    }
}
```

### 4. LM Cost Monotonicity

**Property:** The Levenberg-Marquardt algorithm should monotonically decrease the
cost function (||F(x)||²) at each accepted step.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_lm_monotonic_cost(
        target in positive_vec_strategy(2, 5),
    ) {
        // This requires access to per-iteration cost, which may need
        // a callback mechanism or a special diagnostic mode.
        // Alternative: verify that final residual <= initial residual
        let problem = QuadraticProblem { target: target.clone() };
        let x0: Vec<f64> = target.iter().map(|t| t.sqrt() + 1.0).collect();
        let initial_norm = problem.residual_norm(&x0);

        let solver = LMSolver::new(LMConfig::default());
        let result = solver.solve(&problem, &x0);

        if let SolveResult::Converged { residual_norm, .. }
            | SolveResult::NotConverged { residual_norm, .. } = result
        {
            prop_assert!(
                residual_norm <= initial_norm + 1e-10,
                "Cost increased: {} -> {}",
                initial_norm, residual_norm
            );
        }
    }
}
```

### 5. Constraint Composition

**Property:** Adding a constraint that is already satisfied should not change the
solution.

```rust
#[cfg(feature = "geometry")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_redundant_constraint_no_change(
        side1 in 1.0f64..50.0,
        side2 in 1.0f64..50.0,
        side3 in 1.0f64..50.0,
    ) {
        if side1 >= side2 + side3 || side2 >= side1 + side3 || side3 >= side1 + side2 {
            return Ok(());
        }

        use solverang::geometry::ConstraintSystemBuilder;

        // Solve without redundant constraint
        let system1 = ConstraintSystemBuilder::<2>::new()
            .point(Point2D::new(0.0, 0.0))
            .point(Point2D::new(side1, 0.0))
            .point(Point2D::new(side1 / 2.0, side2 / 2.0))
            .fix(0).fix(1)
            .distance(0, 1, side1)
            .distance(1, 2, side2)
            .distance(2, 0, side3)
            .build();

        let solver = LMSolver::new(LMConfig::default());
        let result1 = solver.solve(&system1, &system1.current_values());

        // Solve with a redundant horizontal constraint on fixed points
        let system2 = ConstraintSystemBuilder::<2>::new()
            .point(Point2D::new(0.0, 0.0))
            .point(Point2D::new(side1, 0.0))
            .point(Point2D::new(side1 / 2.0, side2 / 2.0))
            .fix(0).fix(1)
            .distance(0, 1, side1)
            .distance(1, 2, side2)
            .distance(2, 0, side3)
            .horizontal(0, 1)  // Redundant — already implied by fixed points
            .build();

        let result2 = solver.solve(&system2, &system2.current_values());

        // Solutions should be the same
        if let (
            SolveResult::Converged { solution: s1, .. },
            SolveResult::Converged { solution: s2, .. },
        ) = (&result1, &result2) {
            for (a, b) in s1.iter().zip(s2) {
                prop_assert!(
                    (a - b).abs() < 1e-6,
                    "Redundant constraint changed solution: {} vs {}",
                    a, b
                );
            }
        }
    }
}
```

### 6. Residual Scaling Linearity

**Property:** Moving a point twice as far from the constraint should approximately
double the residual (for distance constraints, which are linear in deviation).

```rust
#[cfg(feature = "geometry")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_distance_residual_scales_linearly(
        target_dist in 1.0f64..100.0,
        deviation in 0.1f64..10.0,
    ) {
        let p1 = Point2D::new(0.0, 0.0);
        let actual_dist = target_dist + deviation;
        let p2_single = Point2D::new(actual_dist, 0.0);
        let p2_double = Point2D::new(target_dist + 2.0 * deviation, 0.0);

        let constraint = DistanceConstraint::<2>::new(0, 1, target_dist);

        let r1 = constraint.residuals(&vec![p1, p2_single]);
        let r2 = constraint.residuals(&vec![p1, p2_double]);

        // For distance constraints, residual = actual - target, so
        // r2 should be approximately 2 * r1
        let ratio = r2[0] / r1[0];
        prop_assert!(
            (ratio - 2.0).abs() < 0.1,
            "Non-linear scaling: ratio={}, expected ~2.0",
            ratio
        );
    }
}
```

### 7. Central Difference Jacobian Verification

**Property:** Central differences (second-order accurate) should match analytical
Jacobians more tightly than forward differences (first-order).

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_central_diff_tighter_than_forward(
        target in positive_vec_strategy(2, 6),
        x in small_vec_strategy(2, 6),
    ) {
        if target.len() != x.len() { return Ok(()); }

        let problem = QuadraticProblem { target };
        let h = 1e-7;

        let analytical = problem.jacobian(&x);

        // Forward difference: (f(x+h) - f(x)) / h
        let forward_error = compute_forward_diff_error(&problem, &x, &analytical, h);

        // Central difference: (f(x+h) - f(x-h)) / (2h)
        let central_error = compute_central_diff_error(&problem, &x, &analytical, h);

        // Central should be more accurate (O(h²) vs O(h))
        prop_assert!(
            central_error <= forward_error * 1.1 + 1e-12,
            "Central diff ({}) not better than forward diff ({})",
            central_error, forward_error
        );
    }
}
```

### 8. Decomposition Correctness

```rust
#[cfg(feature = "parallel")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_decomposed_solution_matches_monolithic(
        n_groups in 2usize..4,
        group_size in 2usize..4,
    ) {
        // Build a problem with n_groups independent sub-problems
        let problem = build_block_diagonal_problem(n_groups, group_size);
        let x0 = problem.initial_point(1.0);

        // Monolithic solve
        let mono = LMSolver::new(LMConfig::default()).solve(&problem, &x0);

        // Decomposed solve
        let decomp = decompose(&problem);
        let decomp_result = solve_decomposed(&decomp, &x0);

        // Solutions should match
        if let (SolveResult::Converged { solution: s1, .. },
                SolveResult::Converged { solution: s2, .. }) = (&mono, &decomp_result) {
            for (a, b) in s1.iter().zip(s2) {
                prop_assert!((a - b).abs() < 1e-8);
            }
        }
    }
}
```

### 9. 3D Constraint Properties

Extend existing 2D properties to 3D:

```rust
#[cfg(feature = "geometry")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_3d_distance_symmetric(
        x1 in coord_strategy(), y1 in coord_strategy(), z1 in coord_strategy(),
        x2 in coord_strategy(), y2 in coord_strategy(), z2 in coord_strategy(),
        target in positive_distance_strategy(),
    ) {
        let p1 = Point3D::new(x1, y1, z1);
        let p2 = Point3D::new(x2, y2, z2);
        let points = vec![p1, p2];

        let c12 = DistanceConstraint::<3>::new(0, 1, target);
        let c21 = DistanceConstraint::<3>::new(1, 0, target);

        let r12 = c12.residuals(&points);
        let r21 = c21.residuals(&points);

        prop_assert!((r12[0] - r21[0]).abs() < 1e-10);
    }

    #[test]
    fn prop_3d_distance_jacobian(
        x1 in coord_strategy(), y1 in coord_strategy(), z1 in coord_strategy(),
        x2 in coord_strategy(), y2 in coord_strategy(), z2 in coord_strategy(),
        target in positive_distance_strategy(),
    ) {
        let p1 = Point3D::new(x1, y1, z1);
        let p2 = Point3D::new(x2, y2, z2);
        let dist = ((x2-x1).powi(2) + (y2-y1).powi(2) + (z2-z1).powi(2)).sqrt();
        if dist < 0.01 { return Ok(()); }

        let mut system = ConstraintSystem::<3>::new();
        system.add_point(p1);
        system.add_point(p2);
        system.add_constraint(Box::new(DistanceConstraint::<3>::new(0, 1, target)));

        let x = system.current_values();
        let v = verify_jacobian(&system, &x, 1e-7, 1e-4);
        prop_assert!(v.passed, "3D distance Jacobian failed: {}", v.max_absolute_error);
    }
}
```

### 10. Sparse Pattern Consistency

```rust
#[cfg(feature = "sparse")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_sparse_pattern_covers_nonzeros(
        n in 5usize..30,
    ) {
        let problem = BroydenTridiagonal::new(n);
        let x = problem.initial_point(1.0);

        let entries = problem.jacobian(&x);
        let pattern = detect_sparsity_pattern(&problem, &x);

        for (row, col, val) in &entries {
            if val.abs() > 1e-15 {
                prop_assert!(
                    pattern.contains(*row, *col),
                    "Pattern missing non-zero at ({}, {})",
                    row, col
                );
            }
        }
    }
}
```

## Improved Strategies

### Near-singular matrix strategy

```rust
fn near_singular_target_strategy(n: usize) -> impl Strategy<Value = Vec<f64>> {
    // Generate targets where some are very close together
    prop::collection::vec(
        prop_oneof![
            Just(0.0),
            Just(1e-15),
            Just(1e-300),
            -100.0f64..100.0,
        ],
        n,
    )
}
```

### Ill-conditioned problem strategy

```rust
fn ill_conditioned_strategy() -> impl Strategy<Value = Vec<f64>> {
    // Mix very large and very small values
    prop::collection::vec(
        prop_oneof![
            1e-10f64..1e-8,    // Very small
            1e8f64..1e10,      // Very large
        ],
        2..6,
    )
}
```

## ProptestConfig Tuning

```rust
// Fast tests (CI-friendly): fewer cases, tighter ranges
#[cfg(not(feature = "slow_tests"))]
const FAST_CASES: u32 = 100;

// Thorough tests (nightly): more cases, wider ranges
#[cfg(feature = "slow_tests")]
const THOROUGH_CASES: u32 = 2000;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(feature = "slow_tests") { 2000 } else { 100 },
        max_shrink_iters: 10000,  // Thorough shrinking for better minimal examples
        ..ProptestConfig::default()
    })]
    // ...
}
```

## File Organization

Split the expanded property tests across multiple files:

```
crates/solverang/tests/
├── property_tests.rs                    # Existing (keep as-is)
├── property_tests_solver.rs             # New: idempotency, determinism, monotonicity
├── property_tests_geometry_3d.rs        # New: 3D constraint properties
├── property_tests_composition.rs        # New: constraint composition, decomposition
└── property_tests_numerical.rs          # New: J^T*J, central diff, sparse pattern
```

## Estimated Effort

| Task | Time |
|------|------|
| Idempotency + determinism properties | 2 hours |
| J^T·J symmetry + PSD | 2 hours |
| LM monotonicity | 1-2 hours |
| Constraint composition | 2-3 hours |
| Central difference Jacobian | 2 hours |
| 3D properties | 2-3 hours |
| Decomposition properties | 2-3 hours |
| Sparse pattern properties | 1-2 hours |
| Improved strategies | 1 hour |
| **Total** | **~16-22 hours** |
