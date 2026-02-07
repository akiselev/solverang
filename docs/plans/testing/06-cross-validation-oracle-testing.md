# Plan 06: Cross-Validation / Oracle Testing

## Goal

Solverang has multiple independent implementations of the same computation:
- **JIT-compiled** vs **interpreted** constraint evaluation
- **Sparse** vs **dense** solvers
- **Parallel** vs **sequential** solvers
- **Macro-generated** vs **hand-written** vs **finite-difference** Jacobians
- **Newton-Raphson** vs **Levenberg-Marquardt** on square systems

These pairs are natural **mutual oracles**: if they disagree, at least one has a bug.
Cross-validation testing systematically compares all oracle pairs across the full test
problem library.

## Oracle Pairs

### Pair 1: JIT-Compiled vs Interpreted (Critical)

**What:** The JIT module compiles constraint systems to native code via Cranelift.
The interpreted path evaluates constraints directly in Rust. Both should produce
identical residual and Jacobian values for the same inputs.

**Comparison strategy:** Bitwise equality (same IEEE 754 operations should produce
same results). If Cranelift generates different instruction sequences (e.g., FMA
where interpreted doesn't), fall back to ULP comparison with max 1-2 ULPs.

```rust
use solverang::jit::JITCompiler;

fn assert_float_eq(a: f64, b: f64, max_ulps: u64) {
    if a == b { return; }
    let a_bits = a.to_bits();
    let b_bits = b.to_bits();
    let ulp_diff = if a_bits > b_bits { a_bits - b_bits } else { b_bits - a_bits };
    assert!(
        ulp_diff <= max_ulps,
        "Float mismatch: {} vs {} ({} ULPs apart, max {})",
        a, b, ulp_diff, max_ulps
    );
}

fn assert_vec_eq(a: &[f64], b: &[f64], max_ulps: u64) {
    assert_eq!(a.len(), b.len(), "Length mismatch: {} vs {}", a.len(), b.len());
    for (i, (ai, bi)) in a.iter().zip(b).enumerate() {
        assert_float_eq(*ai, *bi, max_ulps);
    }
}

#[cfg(feature = "jit")]
#[test]
fn oracle_jit_vs_interpreted_residuals() {
    for system in all_geometric_test_systems() {
        let x = system.current_values();

        // Interpreted
        let interp_residuals = system.residuals(&x);

        // JIT
        let jit = JITCompiler::compile(&system).expect("JIT compilation failed");
        let jit_residuals = jit.residuals(&x);

        assert_vec_eq(&interp_residuals, &jit_residuals, 2);
    }
}

#[cfg(feature = "jit")]
#[test]
fn oracle_jit_vs_interpreted_jacobian() {
    for system in all_geometric_test_systems() {
        let x = system.current_values();

        let interp_jac = system.jacobian_dense(&x);
        let jit = JITCompiler::compile(&system).unwrap();
        let jit_jac = jit.jacobian_dense(&x);

        // Compare dense matrices element-by-element
        for i in 0..interp_jac.nrows() {
            for j in 0..interp_jac.ncols() {
                assert_float_eq(interp_jac[(i, j)], jit_jac[(i, j)], 2);
            }
        }
    }
}
```

### Pair 2: Sparse Solver vs Dense Solver

**What:** For problems small enough to solve both ways, the sparse and dense solvers
should find the same solution (within tolerance).

**Comparison strategy:** Tolerance-based. Solutions may differ slightly due to different
linear algebra paths (faer vs nalgebra), but should agree to ~1e-8.

```rust
#[cfg(feature = "sparse")]
#[test]
fn oracle_sparse_vs_dense() {
    use solverang::{Solver, SolverConfig, SparseSolver, SparseSolverConfig};

    let problems = vec![
        Rosenbrock::new(),
        Powell::new(),
        BroydenTridiagonal::new(20),
    ];

    for problem in &problems {
        let x0 = problem.initial_point(1.0);

        let dense_result = Solver::new(SolverConfig::default()).solve(problem, &x0);
        let sparse_result = SparseSolver::new(SparseSolverConfig::default()).solve(problem, &x0);

        match (&dense_result, &sparse_result) {
            (SolveResult::Converged { solution: s1, .. },
             SolveResult::Converged { solution: s2, .. }) => {
                for (a, b) in s1.iter().zip(s2) {
                    assert!(
                        (a - b).abs() < 1e-6,
                        "Solution mismatch for {}: dense={}, sparse={}",
                        problem.name(), a, b
                    );
                }
            }
            _ => {
                // Both should converge on standard test problems
                panic!(
                    "Solver disagreement on {}: dense={:?}, sparse={:?}",
                    problem.name(), dense_result, sparse_result
                );
            }
        }
    }
}
```

### Pair 3: Parallel Solver vs Sequential Solver

**What:** For decomposable problems, the parallel solver should find the same solution
as solving sub-problems sequentially.

**Comparison strategy:** Tolerance-based. Floating-point addition order may differ
due to parallelism, so allow up to 1e-10 difference.

```rust
#[cfg(feature = "parallel")]
#[test]
fn oracle_parallel_vs_sequential() {
    let problem = build_decomposable_problem();
    let x0 = problem.initial_point(1.0);

    // Sequential
    let seq_result = LMSolver::new(LMConfig::default()).solve(&problem, &x0);

    // Parallel
    let par_result = ParallelSolver::new().solve(&problem, &x0);

    match (&seq_result, &par_result) {
        (SolveResult::Converged { solution: s1, residual_norm: r1, .. },
         SolveResult::Converged { solution: s2, residual_norm: r2, .. }) => {
            assert!(
                (r1 - r2).abs() < 1e-8,
                "Residual norm mismatch: sequential={}, parallel={}",
                r1, r2
            );
            for (a, b) in s1.iter().zip(s2) {
                assert!((a - b).abs() < 1e-8);
            }
        }
        _ => panic!("Both should converge"),
    }
}
```

### Pair 4: Macro-Generated vs Finite-Difference Jacobian

**What:** The `#[auto_jacobian]` macro generates analytical Jacobians via symbolic
differentiation. These should match numerical finite-difference Jacobians.

**Comparison strategy:** The existing `verify_jacobian()` function already does this.
The oracle test applies it systematically across all macro-generated problems.

```rust
#[cfg(feature = "macros")]
#[test]
fn oracle_macro_jacobian_vs_finite_difference() {
    let problems = all_macro_generated_problems();

    for problem in &problems {
        let x = problem.initial_point(1.0);
        let verification = verify_jacobian(problem, &x, 1e-7, 1e-4);
        assert!(
            verification.passed,
            "Macro Jacobian mismatch for {}: max_error={}, at {:?}",
            problem.name(),
            verification.max_absolute_error,
            verification.max_error_location
        );
    }
}
```

### Pair 5: Newton-Raphson vs LM on Square Systems

**What:** For square systems (residual_count == variable_count), both NR and LM
should find the same root.

**Comparison strategy:** Compare solutions within tolerance. NR and LM may converge
to different roots for problems with multiple roots — only compare residual norms
in that case.

```rust
#[test]
fn oracle_nr_vs_lm_square_systems() {
    let square_problems = all_test_problems()
        .filter(|p| p.residual_count() == p.variable_count());

    for problem in square_problems {
        let x0 = problem.initial_point(1.0);

        let nr = Solver::new(SolverConfig::default()).solve(&problem, &x0);
        let lm = LMSolver::new(LMConfig::default()).solve(&problem, &x0);

        match (&nr, &lm) {
            (SolveResult::Converged { residual_norm: r1, solution: s1, .. },
             SolveResult::Converged { residual_norm: r2, solution: s2, .. }) => {
                // Both converged — residual norms should be similar
                assert!(
                    (r1 - r2).abs() < 1e-4 || (*r1 < 1e-6 && *r2 < 1e-6),
                    "{}: NR residual={}, LM residual={}",
                    problem.name(), r1, r2
                );
                // If residuals are both tiny, solutions should match
                if *r1 < 1e-8 && *r2 < 1e-8 {
                    for (a, b) in s1.iter().zip(s2) {
                        assert!(
                            (a - b).abs() < 1e-4,
                            "{}: solution mismatch at NR={}, LM={}",
                            problem.name(), a, b
                        );
                    }
                }
            }
            // OK if one converges and the other doesn't — different algorithms
            _ => {}
        }
    }
}
```

### Pair 6: Auto-Detected Jacobian Pattern vs Explicit Pattern

```rust
#[cfg(feature = "sparse")]
#[test]
fn oracle_auto_pattern_vs_explicit() {
    let problem = BroydenTridiagonal::new(50);
    let x = problem.initial_point(1.0);

    // Auto-detected pattern
    let auto_pattern = detect_sparsity_pattern(&problem, &x);

    // Pattern from Jacobian entries
    let entries = problem.jacobian(&x);
    let explicit_pattern: HashSet<(usize, usize)> = entries
        .iter()
        .filter(|(_, _, v)| *v != 0.0)
        .map(|(r, c, _)| (*r, *c))
        .collect();

    // Auto pattern should be a superset of explicit pattern
    for &(r, c) in &explicit_pattern {
        assert!(
            auto_pattern.contains(r, c),
            "Auto pattern missing entry ({}, {})",
            r, c
        );
    }
}
```

## Comparison Utilities

### ULP-based comparison

```rust
/// Compare two f64 values with ULP (Units in Last Place) tolerance.
fn ulp_diff(a: f64, b: f64) -> u64 {
    if a == b { return 0; }
    if a.is_nan() || b.is_nan() { return u64::MAX; }
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;
    (a_bits - b_bits).unsigned_abs()
}
```

### Statistical comparison for non-deterministic paths

```rust
/// Run a solver N times and check that results are consistent.
fn check_determinism<P: Problem>(solver: &impl Solve<P>, problem: &P, x0: &[f64], runs: usize) {
    let results: Vec<_> = (0..runs)
        .map(|_| solver.solve(problem, x0))
        .collect();

    // All results should be identical
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            format!("{:?}", results[0]),
            format!("{:?}", result),
            "Non-deterministic result on run {}",
            i
        );
    }
}
```

## Test Problem Selection

| Oracle Pair | Good Test Problems | Why |
|------------|-------------------|-----|
| JIT vs Interpreted | All geometric constraint systems | Direct compilation target |
| Sparse vs Dense | Broyden tridiagonal, banded systems | Clearly sparse structure |
| Parallel vs Sequential | Independent triangles, block-diagonal | Decomposable |
| Macro vs FD Jacobian | All `#[auto_jacobian]` problems | Macro coverage |
| NR vs LM | Rosenbrock, Powell, all square NIST | Standard benchmarks |

## Automated Oracle Framework

```rust
/// Run all oracle pairs across all applicable test problems.
#[test]
fn comprehensive_oracle_validation() {
    let results = OracleTestRunner::new()
        .add_pair(JitVsInterpreted::new())
        .add_pair(SparseVsDense::new())
        .add_pair(ParallelVsSequential::new())
        .add_pair(NRvsLM::new())
        .run_all(all_test_problems());

    results.assert_all_passed();
    results.print_summary();
}
```

## Handling Expected Differences

| Scenario | Expected Behavior | Handling |
|----------|------------------|----------|
| Different convergence paths (NR vs LM) | Same root, different iteration count | Compare solutions, not paths |
| Multiple roots | Different valid solutions | Compare residual norms only |
| FP ordering in parallel | Slightly different last bits | ULP tolerance ≤ 4 |
| Sparse numerical pivoting | Different rounding | Solution tolerance 1e-8 |

## File Organization

```
crates/solverang/tests/
├── oracle_tests.rs              # Main oracle test file
├── oracle_jit.rs                # JIT vs interpreted (if large)
└── oracle_helpers.rs            # Shared comparison utilities
```

## Estimated Effort

| Task | Time |
|------|------|
| Write comparison utilities (ULP, tolerance, statistical) | 2 hours |
| JIT vs interpreted oracle | 3-4 hours |
| Sparse vs dense oracle | 2 hours |
| Parallel vs sequential oracle | 2 hours |
| NR vs LM oracle | 2 hours |
| Macro Jacobian oracle | 1 hour |
| Automated framework | 2-3 hours |
| **Total** | **~14-18 hours** |
