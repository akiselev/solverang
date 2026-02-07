# Plan 06: Cross-Validation / Oracle Testing

## Goal

Solverang has multiple independent implementations that compute the same results
through different paths. These pairs form natural **mutual oracles**: if they
disagree, at least one has a bug. This plan defines 12 oracle pairs spanning the
legacy `Problem` trait, the V3 entity/constraint/pipeline architecture, geometry
plugins, and analysis tools.

---

## Comparison Utilities

All oracle tests share a common set of comparison helpers.

### ULP-based comparison (for numerically identical paths)

```rust
fn ulp_diff(a: f64, b: f64) -> u64 {
    if a == b { return 0; }
    if a.is_nan() || b.is_nan() { return u64::MAX; }
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;
    (a_bits - b_bits).unsigned_abs()
}

fn assert_float_eq(a: f64, b: f64, max_ulps: u64) {
    let diff = ulp_diff(a, b);
    assert!(diff <= max_ulps,
        "Float mismatch: {} vs {} ({} ULPs, max {})", a, b, diff, max_ulps);
}

fn assert_vec_eq(a: &[f64], b: &[f64], max_ulps: u64) {
    assert_eq!(a.len(), b.len());
    for (i, (ai, bi)) in a.iter().zip(b).enumerate() {
        assert_float_eq(*ai, *bi, max_ulps);
    }
}
```

### Tolerance-based comparison (for numerically different but mathematically equivalent paths)

```rust
fn solutions_match(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
}

fn residual_norms_match(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol || (a < tol && b < tol)
}
```

### Determinism checker

```rust
fn check_determinism<F: Fn() -> R, R: std::fmt::Debug + PartialEq>(f: F, runs: usize) {
    let reference = f();
    for i in 1..runs {
        let result = f();
        assert_eq!(reference, result, "Non-deterministic on run {}", i);
    }
}
```

---

## Part A: Legacy Oracle Pairs

### Pair 1: JIT-Compiled vs Interpreted Residuals/Jacobians

**What:** The JIT module compiles constraint systems to native code via
Cranelift. The interpreted path evaluates constraints directly. Both should
produce identical IEEE 754 results.

**Comparison:** Bitwise (ULP <= 2) for residuals and Jacobians, tested on all
geometry constraint systems.

```rust
#[cfg(feature = "jit")]
#[test]
fn oracle_jit_vs_interpreted_residuals() {
    for system in all_geometric_test_systems() {
        let x = system.current_values();
        let interp = system.residuals(&x);
        let jit = JITCompiler::compile(&system).unwrap();
        let jit_r = jit.residuals(&x);
        assert_vec_eq(&interp, &jit_r, 2);
    }
}

#[cfg(feature = "jit")]
#[test]
fn oracle_jit_vs_interpreted_jacobian() {
    for system in all_geometric_test_systems() {
        let x = system.current_values();
        let interp = system.jacobian_dense(&x);
        let jit = JITCompiler::compile(&system).unwrap();
        let jit_j = jit.jacobian_dense(&x);
        for i in 0..interp.len() {
            for j in 0..interp[0].len() {
                assert_float_eq(interp[i][j], jit_j[i][j], 2);
            }
        }
    }
}
```

### Pair 2: Sparse Solver vs Dense Solver

**What:** For problems small enough to solve both ways, sparse and dense
solvers should find the same solution.

**Comparison:** Solution tolerance 1e-6, residual norm tolerance 1e-8.

```rust
#[cfg(feature = "sparse")]
#[test]
fn oracle_sparse_vs_dense() {
    let problems = [
        Rosenbrock::new(),
        Powell::new(),
        BroydenTridiagonal::new(20),
    ];
    for problem in &problems {
        let x0 = problem.initial_point(1.0);
        let dense = Solver::new(SolverConfig::default()).solve(problem, &x0);
        let sparse = SparseSolver::new(SparseSolverConfig::default()).solve(problem, &x0);
        match (&dense, &sparse) {
            (SolveResult::Converged { solution: s1, .. },
             SolveResult::Converged { solution: s2, .. }) => {
                assert!(solutions_match(s1, s2, 1e-6),
                    "{}: solutions diverge", problem.name());
            }
            _ => panic!("{}: one or both failed to converge", problem.name()),
        }
    }
}
```

### Pair 3: Parallel Solver vs Sequential Solver

**What:** For decomposable systems, parallel and sequential solves should
produce the same solution.

**Comparison:** Tolerance 1e-8 (FP addition order may differ).

```rust
#[cfg(feature = "parallel")]
#[test]
fn oracle_parallel_vs_sequential() {
    let problem = build_decomposable_problem();
    let x0 = problem.initial_point(1.0);
    let seq = LMSolver::new(LMConfig::default()).solve(&problem, &x0);
    let par = ParallelSolver::new().solve(&problem, &x0);
    match (&seq, &par) {
        (SolveResult::Converged { solution: s1, residual_norm: r1, .. },
         SolveResult::Converged { solution: s2, residual_norm: r2, .. }) => {
            assert!(residual_norms_match(*r1, *r2, 1e-8));
            assert!(solutions_match(s1, s2, 1e-8));
        }
        _ => panic!("Both should converge"),
    }
}
```

### Pair 4: Macro-Generated vs Finite-Difference Jacobian

**What:** `#[auto_jacobian]` generates analytical Jacobians. These should
match finite-difference Jacobians.

```rust
#[cfg(feature = "macros")]
#[test]
fn oracle_macro_jacobian_vs_finite_difference() {
    for problem in all_macro_generated_problems() {
        let x = problem.initial_point(1.0);
        let v = verify_jacobian(&problem, &x, 1e-7, 1e-4);
        assert!(v.passed, "{}: max_error={}", problem.name(), v.max_absolute_error);
    }
}
```

### Pair 5: Newton-Raphson vs LM on Square Systems

**What:** For square systems (residual_count == variable_count), both should
find the same root when both converge.

```rust
#[test]
fn oracle_nr_vs_lm_square_systems() {
    let problems = all_test_problems()
        .filter(|p| p.residual_count() == p.variable_count());
    for problem in problems {
        let x0 = problem.initial_point(1.0);
        let nr = Solver::new(SolverConfig::default()).solve(&problem, &x0);
        let lm = LMSolver::new(LMConfig::default()).solve(&problem, &x0);
        match (&nr, &lm) {
            (SolveResult::Converged { residual_norm: r1, solution: s1, .. },
             SolveResult::Converged { residual_norm: r2, solution: s2, .. }) => {
                assert!(residual_norms_match(*r1, *r2, 1e-4),
                    "{}: NR={}, LM={}", problem.name(), r1, r2);
                if *r1 < 1e-8 && *r2 < 1e-8 {
                    assert!(solutions_match(s1, s2, 1e-4),
                        "{}: solutions diverge", problem.name());
                }
            }
            _ => {} // Acceptable: different algorithms may not both converge
        }
    }
}
```

### Pair 6: Auto-Detected Sparsity Pattern vs Explicit Pattern

```rust
#[cfg(feature = "sparse")]
#[test]
fn oracle_auto_pattern_vs_explicit() {
    let problem = BroydenTridiagonal::new(50);
    let x = problem.initial_point(1.0);
    let auto_pattern = detect_sparsity_pattern(&problem, &x);
    let entries = problem.jacobian(&x);
    let explicit: HashSet<(usize, usize)> = entries.iter()
        .filter(|(_, _, v)| *v != 0.0)
        .map(|(r, c, _)| (*r, *c))
        .collect();
    for &(r, c) in &explicit {
        assert!(auto_pattern.contains(r, c), "Missing ({}, {})", r, c);
    }
}
```

---

## Part B: V3 Architecture Oracle Pairs

### Pair 7: Legacy Problem Trait vs V3 ReducedSubProblem

**What:** Build the same constraint system via the legacy `ConstraintSystem<2>`
API and via the V3 `ConstraintSystem` + `Sketch2DBuilder` API. Solve both.
Solutions should match within tolerance.

This validates that the V3 pipeline (decompose, analyze, reduce, solve,
post-process) produces the same results as the legacy direct-solve path.

**Comparison:** Solution tolerance 1e-6.

```rust
#[test]
fn oracle_legacy_vs_v3_triangle() {
    // --- Legacy path ---
    let mut legacy = geometry::ConstraintSystem::<2>::new();
    legacy.add_point(Point2D::new(0.0, 0.0));
    legacy.add_point(Point2D::new(10.0, 0.0));
    legacy.add_point(Point2D::new(5.0, 5.0));
    legacy.fix(0);
    legacy.fix(1);
    legacy.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 1, 10.0)));
    legacy.add_constraint(Box::new(DistanceConstraint::<2>::new(1, 2, 8.0)));
    legacy.add_constraint(Box::new(DistanceConstraint::<2>::new(2, 0, 6.0)));
    let legacy_result = LMSolver::new(LMConfig::default())
        .solve(&legacy, &legacy.current_values());

    // --- V3 path ---
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_fixed_point(10.0, 0.0);
    let p2 = b.add_point(5.0, 5.0);
    b.constrain_distance(p0, p1, 10.0);
    b.constrain_distance(p1, p2, 8.0);
    b.constrain_distance(p2, p0, 6.0);
    let mut system = b.build();
    let v3_result = system.solve();

    // Compare converged solutions
    // Extract p2 coordinates from both systems and compare
    // The free point (p2) should be at the same location in both
    if let SolveResult::Converged { solution, .. } = legacy_result {
        // Legacy solution is the full parameter vector; extract p2 coords
        // V3 solution is in the ParamStore; extract p2 coords
        // Compare within tolerance
    }
}

#[test]
fn oracle_legacy_vs_v3_all_constraint_types() {
    // For each of the 15 constraint types shared between legacy and V3:
    // 1. Build equivalent systems via both APIs
    // 2. Solve both
    // 3. Compare solutions
    let constraint_types = [
        "distance", "coincident", "fixed", "horizontal", "vertical",
        "parallel", "perpendicular", "angle", "midpoint", "symmetric",
        "equal_length", "point_on_circle", "tangent_line_circle",
        "tangent_circle_circle", "point_on_line",
    ];
    for ct in constraint_types {
        let (legacy_solution, v3_solution) = build_and_solve_both_apis(ct);
        assert!(solutions_match(&legacy_solution, &v3_solution, 1e-6),
            "Oracle mismatch for constraint type: {}", ct);
    }
}
```

### Pair 8: Closed-Form vs Iterative Solver

**What:** For patterns detected by `graph/pattern.rs` (`ScalarSolve`,
`TwoDistances`, `HorizontalVertical`, `DistanceAngle`), the closed-form
solution should match the iterative LM solution on the same sub-problem.

**Comparison:** Solution tolerance 1e-8.

```rust
#[test]
fn oracle_closed_form_vs_iterative_scalar_solve() {
    // Build a system where one parameter is constrained by a single equation.
    // Solve via closed-form (ScalarSolve pattern) and via full LM.
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(5.0, 0.0);
    b.constrain_horizontal(p0, p1);
    b.constrain_distance(p0, p1, 10.0);
    let mut system_cf = b.build();
    let cf_result = system_cf.solve();

    // Now solve the same system with closed-form detection disabled.
    let mut b2 = Sketch2DBuilder::new();
    // ... same setup ...
    let mut system_iter = b2.build();
    system_iter.set_pipeline(
        PipelineBuilder::new()
            .solve(NumericalOnlySolve) // No closed-form
            .build()
    );
    let iter_result = system_iter.solve();

    // Compare parameter values
    // Both should yield the same point positions
}

#[test]
fn oracle_closed_form_vs_iterative_two_distances() {
    // Circle-circle intersection: closed-form gives 2 branches.
    // LM starting from the same initial guess should converge to one of them.
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_fixed_point(6.0, 0.0);
    let p2 = b.add_point(3.0, 4.0);
    b.constrain_distance(p0, p2, 5.0);
    b.constrain_distance(p1, p2, 5.0);

    // Solve with default pipeline (tries closed-form first)
    let mut system = b.build();
    let result = system.solve();
    let cf_x = system.get_param(/* p2.x */);
    let cf_y = system.get_param(/* p2.y */);

    // The closed-form circle-circle intersection at (0,0) r=5 and (6,0) r=5
    // yields points (3, 4) and (3, -4).
    let expected_a = (3.0, 4.0);
    let expected_b = (3.0, -4.0);
    let matches_a = (cf_x - expected_a.0).abs() < 1e-6 && (cf_y - expected_a.1).abs() < 1e-6;
    let matches_b = (cf_x - expected_b.0).abs() < 1e-6 && (cf_y - expected_b.1).abs() < 1e-6;
    assert!(matches_a || matches_b,
        "Closed-form solution ({}, {}) doesn't match either expected branch", cf_x, cf_y);
}

#[test]
fn oracle_closed_form_vs_iterative_horizontal_vertical() {
    // HorizontalVertical pattern: direct assignment should match LM.
    // Build a point constrained to be horizontal with one point and
    // vertical with another. The solution is unique.
}

#[test]
fn oracle_closed_form_vs_iterative_distance_angle() {
    // DistanceAngle pattern: polar coordinates to cartesian should match LM.
}
```

### Pair 9: Full Solve vs Incremental Solve

**What:** Solve a complete system. Modify one parameter. Full re-solve vs
incremental re-solve (only dirty clusters). Both should produce the same
result.

This validates the `ChangeTracker` + `SolutionCache` incremental logic.

**Comparison:** Solution tolerance 1e-10 (should be bitwise identical in most
cases since the same solver runs on the same cluster).

```rust
#[test]
fn oracle_full_vs_incremental_solve() {
    // Build a system with 2 independent clusters.
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(5.0, 0.0);
    b.constrain_distance(p0, p1, 10.0);

    let p2 = b.add_fixed_point(20.0, 20.0);
    let p3 = b.add_point(25.0, 20.0);
    b.constrain_distance(p2, p3, 8.0);

    let mut system = b.build();
    system.solve(); // Initial solve

    // Modify one param in cluster 1 only.
    system.set_param(/* p1.x */, 3.0);

    // Full re-solve: reset pipeline cache and solve everything.
    let mut system_full = system.clone(); // if ConstraintSystem is Clone
    system_full.pipeline.invalidate();
    let full_result = system_full.solve();

    // Incremental re-solve: only dirty cluster re-solved.
    let incr_result = system.solve_incremental();

    // Compare all parameter values.
    // Both should yield identical final state.
}

#[test]
fn oracle_full_vs_incremental_structural_change() {
    // After adding a constraint (structural change), incremental should
    // fall back to full re-decompose. Verify results match.
    let mut b = Sketch2DBuilder::new();
    // ... build system ...
    let mut system = b.build();
    system.solve();

    // Add a new constraint (structural change)
    let cid = system.alloc_constraint_id();
    system.add_constraint(Box::new(/* new constraint */));

    // Incremental solve should detect structural change and re-decompose.
    let incr_result = system.solve();
    assert!(matches!(incr_result.status,
        SystemStatus::Solved | SystemStatus::PartiallySolved));
}
```

### Pair 10: With-Reduction vs Without-Reduction

**What:** Solve the same system with reduce passes enabled and disabled.
The final parameter values should be the same.

This validates the reduce module: substitution, merge, and eliminate passes.

**Comparison:** Solution tolerance 1e-8.

```rust
#[test]
fn oracle_with_reduction_vs_without() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(5.0, 0.0);
    let p2 = b.add_point(5.0, 5.0);
    b.constrain_distance(p0, p1, 10.0);
    b.constrain_distance(p1, p2, 8.0);
    b.constrain_horizontal(p0, p1);  // p1.y = p0.y; can be eliminated

    // Solve with default pipeline (reduction enabled)
    let mut system_reduced = b.build();
    let result_reduced = system_reduced.solve();

    // Solve with no-op reduce phase
    let mut system_unreduced = b.build();
    system_unreduced.set_pipeline(
        PipelineBuilder::new()
            .reduce(NoopReduce)
            .build()
    );
    let result_unreduced = system_unreduced.solve();

    // Compare final parameter values
    assert!(matches!(result_reduced.status, SystemStatus::Solved | SystemStatus::PartiallySolved));
    assert!(matches!(result_unreduced.status, SystemStatus::Solved | SystemStatus::PartiallySolved));
    // Extract and compare point coordinates
}

#[test]
fn oracle_reduction_preserves_solution_multi_pass() {
    // System with fixed params, coincident params, and trivially
    // solvable single-variable constraints -- exercises all 3 reduce passes.
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(0.0, 0.0);
    let p2 = b.add_point(5.0, 0.0);
    b.constrain_coincident(p0, p1);   // merge pass
    b.constrain_horizontal(p0, p2);   // eliminate pass (p2.y = 0)
    b.constrain_distance(p0, p2, 7.0); // remaining constraint
    // ... compare reduced vs unreduced solutions ...
}
```

### Pair 11: Sketch2D Squared vs Legacy Unsquared Formulations

**What:** The V3 sketch2d constraints use squared formulations (e.g.,
`dx^2+dy^2 - d^2`) while legacy geometry constraints use unsquared forms
(e.g., `sqrt(dx^2+dy^2) - d`). Both should converge to the same geometry.

**Comparison:** Converged point positions tolerance 1e-6.

```rust
#[test]
fn oracle_squared_vs_unsquared_distance() {
    // Legacy unsquared: residual = sqrt(dx^2+dy^2) - d
    let mut legacy = geometry::ConstraintSystem::<2>::new();
    legacy.add_point(Point2D::new(0.0, 0.0));
    legacy.add_point(Point2D::new(5.0, 3.0));
    legacy.fix(0);
    legacy.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 1, 10.0)));
    let legacy_result = LMSolver::new(LMConfig::default())
        .solve(&legacy, &legacy.current_values());

    // V3 squared: residual = dx^2+dy^2 - d^2
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(5.0, 3.0);
    b.constrain_distance(p0, p1, 10.0);
    let mut system = b.build();
    let v3_result = system.solve();

    // Both should place p1 at distance 10.0 from origin.
    // The direction may differ (under-constrained), so compare distance only.
    // legacy_x, legacy_y from legacy solution
    // v3_x, v3_y from V3 solution
    // assert!((legacy_dist - 10.0).abs() < 1e-6);
    // assert!((v3_dist - 10.0).abs() < 1e-6);
}

#[test]
fn oracle_squared_vs_unsquared_all_shared_constraints() {
    // For each constraint type that exists in both legacy and V3:
    // Build equivalent fully-constrained systems, solve, compare point positions.
    let test_cases = [
        build_distance_oracle_pair(),
        build_coincident_oracle_pair(),
        build_horizontal_oracle_pair(),
        build_vertical_oracle_pair(),
        build_parallel_oracle_pair(),
        build_perpendicular_oracle_pair(),
        build_angle_oracle_pair(),
        build_midpoint_oracle_pair(),
        build_symmetric_oracle_pair(),
        build_equal_length_oracle_pair(),
        build_point_on_circle_oracle_pair(),
        build_tangent_oracle_pair(),
    ];
    for (name, legacy_pts, v3_pts) in test_cases {
        for (lp, vp) in legacy_pts.iter().zip(&v3_pts) {
            assert!((lp.0 - vp.0).abs() < 1e-5 && (lp.1 - vp.1).abs() < 1e-5,
                "{}: legacy ({},{}) vs v3 ({},{})", name, lp.0, lp.1, vp.0, vp.1);
        }
    }
}
```

### Pair 12: DOF Analysis -- `analyze_dof()` vs `quick_dof()`

**What:** The SVD-based `analyze_dof()` computes true DOF from the Jacobian
rank. The counting-based `quick_dof()` computes `free_params - equations`.
On well-conditioned systems without redundancy, they should agree.

**Comparison:** Exact integer equality.

```rust
#[test]
fn oracle_analyze_dof_vs_quick_dof_well_conditioned() {
    let test_systems = [
        build_well_constrained_triangle(),    // DOF = 0
        build_under_constrained_triangle(),   // DOF = 1
        build_fully_free_point(),             // DOF = 2
        build_rigid_square(),                 // DOF = 0
    ];
    for (name, system) in test_systems {
        let svd_dof = system.analyze_dof().total_dof;
        let quick = system.degrees_of_freedom();
        assert_eq!(svd_dof, quick,
            "{}: SVD DOF={} vs counting DOF={}", name, svd_dof, quick);
    }
}

#[test]
fn oracle_dof_diverges_on_redundant_system() {
    // With redundancy, counting over-counts constraints while SVD detects
    // the rank deficiency. quick_dof() may report DOF < 0, while
    // analyze_dof() correctly identifies DOF >= 0.
    let system = build_redundant_triangle();
    let svd_dof = system.analyze_dof().total_dof;
    let quick = system.degrees_of_freedom();
    // quick_dof counts more equations than rank, so quick < svd_dof
    assert!(svd_dof >= 0, "SVD DOF should be non-negative");
    // Document that they diverge
    assert!(quick <= svd_dof,
        "Counting DOF ({}) should be <= SVD DOF ({}) on redundant systems",
        quick, svd_dof);
}
```

---

## Test Problem Selection

| Oracle Pair | Test Systems | Why |
|------------|-------------|-----|
| 1. JIT vs Interpreted | All legacy geometric systems | Direct JIT target |
| 2. Sparse vs Dense | Broyden, banded systems | Clearly sparse structure |
| 3. Parallel vs Sequential | Block-diagonal, independent triangles | Decomposable |
| 4. Macro vs FD Jacobian | All `#[auto_jacobian]` problems | Macro coverage |
| 5. NR vs LM | Rosenbrock, Powell, NIST square | Standard benchmarks |
| 6. Auto vs Explicit Pattern | Broyden, banded | Known sparsity |
| 7. Legacy vs V3 | Triangle, rectangle, all 15 constraint types | Full API coverage |
| 8. Closed-form vs Iterative | ScalarSolve, TwoDistances, HV, DistAngle | All 4 patterns |
| 9. Full vs Incremental | Multi-cluster, single-edit scenarios | Incremental logic |
| 10. With vs Without Reduction | Fixed points, coincident, trivial constraints | All 3 reduce passes |
| 11. Squared vs Unsquared | All shared constraint types | Formulation equivalence |
| 12. SVD DOF vs Counting DOF | Well-constrained, redundant, under-constrained | DOF accuracy |

---

## Automated Oracle Framework

```rust
/// Trait for oracle test pairs.
trait OraclePair {
    fn name(&self) -> &str;
    fn run(&self) -> OracleResult;
}

struct OracleResult {
    passed: bool,
    description: String,
    max_difference: f64,
}

struct OracleTestRunner {
    pairs: Vec<Box<dyn OraclePair>>,
}

impl OracleTestRunner {
    fn new() -> Self { Self { pairs: Vec::new() } }

    fn add_pair(mut self, pair: impl OraclePair + 'static) -> Self {
        self.pairs.push(Box::new(pair));
        self
    }

    fn run_all(&self) -> Vec<OracleResult> {
        self.pairs.iter().map(|p| {
            let result = p.run();
            if !result.passed {
                eprintln!("ORACLE FAIL [{}]: {} (max_diff={})",
                    p.name(), result.description, result.max_difference);
            }
            result
        }).collect()
    }
}

#[test]
fn comprehensive_oracle_validation() {
    let results = OracleTestRunner::new()
        .add_pair(LegacyVsV3Oracle::new())
        .add_pair(ClosedFormVsIterativeOracle::new())
        .add_pair(FullVsIncrementalOracle::new())
        .add_pair(WithVsWithoutReductionOracle::new())
        .add_pair(SquaredVsUnsquaredOracle::new())
        .add_pair(DofSvdVsCountingOracle::new())
        // Legacy pairs:
        .add_pair(JitVsInterpretedOracle::new())
        .add_pair(SparseVsDenseOracle::new())
        .add_pair(NRvsLMOracle::new())
        .run_all();

    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    assert!(failures.is_empty(),
        "{} oracle failures out of {} total", failures.len(), results.len());
}
```

---

## Handling Expected Differences

| Scenario | Expected Behavior | Handling |
|----------|------------------|----------|
| Different convergence paths (NR vs LM) | Same root, different iterations | Compare solutions, not paths |
| Multiple roots (circle-circle) | Different valid branches | Check against all known branches |
| FP ordering in parallel | Slightly different last bits | ULP tolerance <= 4 |
| Squared vs unsquared formulations | Same geometry, different residuals | Compare point positions, not residuals |
| Incremental vs full on clean clusters | Identical (cached solution reused) | Bitwise comparison |
| Reduce eliminates variables | Fewer solver variables, same final values | Compare full param store |
| SVD DOF vs counting on redundant systems | Counting under-counts DOF | Document divergence, don't assert equality |

---

## File Organization

```
crates/solverang/tests/
  oracle_tests/
    mod.rs
    comparison_utils.rs          -- ULP, tolerance, determinism helpers
    legacy_oracles.rs            -- Pairs 1-6 (JIT, sparse, parallel, macro, NR/LM, pattern)
    v3_oracles.rs                -- Pairs 7-12 (legacy-vs-V3, closed-form, incremental,
                                    reduction, squared, DOF)
    oracle_framework.rs          -- OracleTestRunner and comprehensive test
    test_system_builders.rs      -- Helper functions to build test systems for both APIs
```

## Estimated Effort

| Task | Time |
|------|------|
| Comparison utilities (ULP, tolerance, determinism) | 2 hours |
| Legacy oracle pairs (1-6) | 8-10 hours |
| Legacy vs V3 oracle (pair 7) -- all constraint types | 4-5 hours |
| Closed-form vs iterative oracle (pair 8) -- 4 patterns | 3-4 hours |
| Full vs incremental oracle (pair 9) | 2-3 hours |
| With vs without reduction oracle (pair 10) | 2-3 hours |
| Squared vs unsquared oracle (pair 11) | 3-4 hours |
| DOF SVD vs counting oracle (pair 12) | 1-2 hours |
| Test system builder helpers | 3-4 hours |
| Oracle framework and comprehensive test | 2-3 hours |
| **Total** | **~30-40 hours** |
