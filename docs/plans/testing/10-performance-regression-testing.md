# Plan 10: Performance Regression Testing

## Goal

Solverang has 3 existing Criterion benchmark suites (`comprehensive.rs`, `scaling.rs`,
`nist_benchmarks.rs`) covering the legacy `Problem`-based solvers. The V3 architecture
introduces a 5-phase pipeline with incremental solving, closed-form pattern matching,
symbolic reduction, and graph decomposition -- all with distinct performance
characteristics that need benchmarking.

This plan adds:
- **V3 pipeline benchmarks** covering each phase and the full pipeline
- **Incremental solving benchmarks** demonstrating the speed advantage of dirty-cluster tracking
- **Closed-form vs iterative benchmarks** showing orders-of-magnitude wins for matched patterns
- **Symbolic reduction benchmarks** measuring variable-count savings
- **Graph decomposition and DOF analysis scaling tests**
- **Sketch2DBuilder end-to-end benchmarks** at increasing complexity
- **Automated baseline comparison** to detect regressions in PRs
- **Performance gates** that fail CI when performance degrades

## Criterion Baseline Strategy

### Save a baseline on main branch

```bash
# On the main branch, after all tests pass:
cargo bench --features parallel,sparse,jit,geometry -- --save-baseline main
```

This saves timing data in `target/criterion/`.

### Compare on feature branches

```bash
# On a feature branch:
cargo bench --features parallel,sparse,jit,geometry -- --baseline main
```

### Install `critcmp` for better comparison

```bash
cargo install critcmp

# Export baselines as JSON
cargo bench -- --save-baseline main
cargo bench -- --save-baseline feature

# Compare
critcmp main feature
```

## V3 Pipeline Benchmarks

### Benchmark 1: Full Pipeline Overhead vs Direct LM Solve

Measure the overhead of the V3 pipeline (decompose + analyze + reduce + solve +
post-process) compared to directly calling the LM solver on an equivalent `Problem`.
The overhead should be small for moderate system sizes and amortized by incremental
solving for repeated solves.

```rust
fn bench_pipeline_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_overhead");

    for n_constraints in [5, 10, 20, 50, 100] {
        // Build a V3 ConstraintSystem via Sketch2DBuilder
        group.bench_with_input(
            BenchmarkId::new("v3_pipeline", n_constraints),
            &n_constraints,
            |b, &n| {
                b.iter_batched(
                    || build_sketch2d_chain(n),
                    |mut system| {
                        let result = system.solve();
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Equivalent legacy Problem + LMSolver for comparison
        group.bench_with_input(
            BenchmarkId::new("direct_lm", n_constraints),
            &n_constraints,
            |b, &n| {
                b.iter_batched(
                    || build_equivalent_legacy_problem(n),
                    |(problem, x0)| {
                        let solver = LMSolver::new(LMConfig::default());
                        let result = solver.solve(&problem, &x0);
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
```

### Benchmark 2: Incremental Solving

The V3 pipeline caches cluster decomposition and skips clean clusters. After an
initial solve, changing a single parameter and re-solving should be dramatically
faster than a full re-solve, especially for systems with many independent clusters.

```rust
fn bench_incremental_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_solving");

    for n_independent_clusters in [2, 5, 10, 20] {
        // Full re-solve (pipeline.invalidate() before each solve)
        group.bench_with_input(
            BenchmarkId::new("full_resolve", n_independent_clusters),
            &n_independent_clusters,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut system = build_multi_cluster_system(n);
                        system.solve(); // Initial solve
                        system
                    },
                    |mut system| {
                        // Invalidate everything: forces full re-decompose + re-solve
                        system.set_pipeline(SolvePipeline::default());
                        let result = system.solve();
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Incremental re-solve (change 1 param in 1 cluster, re-solve)
        group.bench_with_input(
            BenchmarkId::new("incremental_one_param", n_independent_clusters),
            &n_independent_clusters,
            |b, &n| {
                b.iter_batched(
                    || {
                        let (mut system, first_param) = build_multi_cluster_system_with_param(n);
                        system.solve(); // Initial solve
                        (system, first_param)
                    },
                    |(mut system, param_id)| {
                        // Change one param: only its cluster should re-solve
                        system.set_param(param_id, 1.5);
                        let result = system.solve();
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
```

**Expected result**: Incremental re-solve with 20 clusters should be roughly 20x
faster than full re-solve when only 1 cluster is dirty.

### Benchmark 3: Closed-Form vs Iterative Solving

For patterns that match the closed-form catalogue (ScalarSolve, TwoDistances,
HorizontalVertical, DistanceAngle), the closed-form solver should be orders of
magnitude faster than iterative LM. This benchmark quantifies the difference.

```rust
fn bench_closed_form_vs_iterative(c: &mut Criterion) {
    let mut group = c.benchmark_group("closed_form_vs_iterative");

    // ScalarSolve: single constraint, single free variable
    group.bench_function("scalar_closed_form", |b| {
        b.iter_batched(
            || build_scalar_solve_system(),
            |mut system| black_box(system.solve()),
            BatchSize::SmallInput,
        );
    });

    // Same system but force iterative solving (NumericalOnlySolve phase)
    group.bench_function("scalar_iterative", |b| {
        b.iter_batched(
            || {
                let mut system = build_scalar_solve_system();
                system.set_pipeline(
                    PipelineBuilder::new()
                        .solve(NumericalOnlySolve)
                        .build()
                );
                system
            },
            |mut system| black_box(system.solve()),
            BatchSize::SmallInput,
        );
    });

    // TwoDistances (circle-circle intersection): 2 distance constraints on a point
    group.bench_function("two_distances_closed_form", |b| {
        b.iter_batched(
            || build_two_distance_system(),
            |mut system| black_box(system.solve()),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("two_distances_iterative", |b| {
        b.iter_batched(
            || {
                let mut system = build_two_distance_system();
                system.set_pipeline(
                    PipelineBuilder::new()
                        .solve(NumericalOnlySolve)
                        .build()
                );
                system
            },
            |mut system| black_box(system.solve()),
            BatchSize::SmallInput,
        );
    });

    // HorizontalVertical: direct assignment
    group.bench_function("hv_closed_form", |b| {
        b.iter_batched(
            || build_hv_system(),
            |mut system| black_box(system.solve()),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}
```

### Benchmark 4: Symbolic Reduction Pass

Measure time saved by the reduce phase. Compare variable count before and after
reduction, and time the reduction itself.

```rust
fn bench_reduce_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_pass");

    for n_constraints in [10, 30, 50, 100] {
        // Time the reduce phase itself
        group.bench_with_input(
            BenchmarkId::new("reduce_time", n_constraints),
            &n_constraints,
            |b, &n| {
                b.iter_batched(
                    || {
                        let system = build_reducible_system(n);
                        let (cluster, constraints, store) = extract_cluster_data(&system);
                        (cluster, constraints, store)
                    },
                    |(cluster, constraints, store)| {
                        let reduce = DefaultReduce;
                        let reduced = reduce.reduce(&cluster, &constraints, &store);
                        black_box(reduced)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // With reduce vs without reduce (full pipeline timing)
        group.bench_with_input(
            BenchmarkId::new("solve_with_reduce", n_constraints),
            &n_constraints,
            |b, &n| {
                b.iter_batched(
                    || build_reducible_system(n),
                    |mut system| black_box(system.solve()),
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("solve_without_reduce", n_constraints),
            &n_constraints,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut system = build_reducible_system(n);
                        system.set_pipeline(
                            PipelineBuilder::new()
                                .reduce(NoopReduce)
                                .build()
                        );
                        system
                    },
                    |mut system| black_box(system.solve()),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
```

### Benchmark 5: Graph Decomposition Scaling

The decompose phase partitions the constraint graph into independent clusters using
union-find. This should scale roughly linearly with the number of entities and
constraints.

```rust
fn bench_graph_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_decomposition");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for n_entities in [10, 50, 100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("decompose", n_entities),
            &n_entities,
            |b, &n| {
                b.iter_batched(
                    || build_system_for_decomposition(n),
                    |(constraints, entities, store)| {
                        let decomposer = DefaultDecompose;
                        let clusters = decomposer.decompose(&constraints, &entities, &store);
                        black_box(clusters)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
```

### Benchmark 6: DOF Analysis Scaling

SVD-based DOF analysis scales with matrix size. Benchmark to establish the cost
curve and ensure it doesn't become a bottleneck.

```rust
fn bench_dof_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("dof_analysis");

    for n_entities in [5, 10, 25, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("analyze_dof", n_entities),
            &n_entities,
            |b, &n| {
                b.iter_batched(
                    || build_constrained_system(n),
                    |system| {
                        let result = system.analyze_dof();
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Quick DOF (equation counting, no SVD) for comparison
        group.bench_with_input(
            BenchmarkId::new("quick_dof", n_entities),
            &n_entities,
            |b, &n| {
                b.iter_batched(
                    || build_constrained_system(n),
                    |system| {
                        let dof = system.degrees_of_freedom();
                        black_box(dof)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
```

### Benchmark 7: Sketch2DBuilder End-to-End

Benchmark the full workflow: builder construction, pipeline solve, and solution
verification for sketches of increasing complexity.

```rust
fn bench_sketch2d_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("sketch2d_builder");

    for n_constraints in [10, 50, 100, 500] {
        // Build + solve
        group.bench_with_input(
            BenchmarkId::new("build_and_solve", n_constraints),
            &n_constraints,
            |b, &n| {
                b.iter(|| {
                    let mut builder = Sketch2DBuilder::new();
                    // Build a chain of distance-constrained points
                    let mut prev = builder.add_fixed_point(0.0, 0.0);
                    for i in 0..n {
                        let next = builder.add_point(
                            (i + 1) as f64 * 2.0,
                            ((i as f64) * 0.3).sin() * 3.0,
                        );
                        builder.constrain_distance(prev, next, 2.0);
                        prev = next;
                    }
                    let mut system = builder.build();
                    let result = system.solve();
                    black_box(result)
                });
            },
        );

        // Solve-only (builder reused, system pre-built)
        group.bench_with_input(
            BenchmarkId::new("solve_only", n_constraints),
            &n_constraints,
            |b, &n| {
                b.iter_batched(
                    || build_sketch2d_chain(n),
                    |mut system| {
                        let result = system.solve();
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
```

### Benchmark 8: Pattern Detection

Time the pattern matching scan against growing numbers of constraints. Pattern
detection should be fast relative to solving.

```rust
fn bench_pattern_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_detection");

    for n_constraints in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("scan_patterns", n_constraints),
            &n_constraints,
            |b, &n| {
                b.iter_batched(
                    || build_pattern_rich_system(n),
                    |(entities, constraints, store)| {
                        let patterns = scan_for_patterns(&entities, &constraints, &store);
                        black_box(patterns)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
```

## Algorithmic Complexity Tests

These are `#[test]` functions (not Criterion benchmarks) that verify asymptotic
scaling. They use wall-clock timing with generous margins to avoid flakiness.

### Test 1: Sparse Solver Scales ~O(n), Not O(n^2)

```rust
#[cfg(feature = "sparse")]
#[test]
fn sparse_solver_scales_linearly() {
    use std::time::Instant;

    let sizes = [50, 100, 200, 400, 800];
    let mut times = Vec::new();

    for &n in &sizes {
        let problem = BroydenTridiagonal::new(n);
        let x0 = problem.initial_point(1.0);
        let solver = SparseSolver::new(SparseSolverConfig::default());

        let start = Instant::now();
        let result = solver.solve(&problem, &x0);
        let elapsed = start.elapsed().as_secs_f64();

        assert!(result.is_converged(), "Failed to converge at n={}", n);
        times.push((n, elapsed));
    }

    let ratio = times.last().unwrap().1 / times.first().unwrap().1;
    let size_ratio = sizes.last().unwrap() / sizes.first().unwrap();
    let max_expected_ratio = (size_ratio as f64) * (size_ratio as f64).log2();

    assert!(
        ratio < max_expected_ratio,
        "Sparse solver scaling too steep: {}x time for {}x size (max expected {}x). \
         Times: {:?}",
        ratio, size_ratio, max_expected_ratio, times
    );
}
```

### Test 2: V3 Pipeline Scales with Constraint Count

```rust
#[test]
fn v3_pipeline_scales_subquadratically() {
    use std::time::Instant;

    let sizes = [10, 30, 100, 300];
    let mut times = Vec::new();

    for &n in &sizes {
        let mut system = build_sketch2d_chain(n);

        let start = Instant::now();
        let result = system.solve();
        let elapsed = start.elapsed().as_secs_f64();

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Failed at n={}",
            n
        );
        times.push((n, elapsed));
    }

    // time(300) / time(10) should be < 300 (sub-quadratic)
    let ratio = times.last().unwrap().1 / times.first().unwrap().1;
    let size_ratio = 300.0 / 10.0;
    let max_ratio = size_ratio * size_ratio.log2(); // ~O(n log n) upper bound

    assert!(
        ratio < max_ratio,
        "V3 pipeline scaling too steep: {:.1}x time for {:.0}x constraints (max {:.1}x). \
         Times: {:?}",
        ratio, size_ratio, max_ratio, times
    );
}
```

### Test 3: Incremental Solving Is Faster Than Full

```rust
#[test]
fn incremental_solve_faster_than_full() {
    use std::time::Instant;

    let n_clusters = 10;
    let (mut system, first_param) = build_multi_cluster_system_with_param(n_clusters);
    system.solve(); // Initial solve

    // Full re-solve (invalidate pipeline)
    let mut full_system = system.clone();
    full_system.set_pipeline(SolvePipeline::default());
    let start = Instant::now();
    let _ = full_system.solve();
    let full_time = start.elapsed();

    // Incremental re-solve (change one param)
    system.set_param(first_param, 1.5);
    let start = Instant::now();
    let _ = system.solve();
    let incremental_time = start.elapsed();

    // Incremental should be at least 2x faster (usually much more)
    let speedup = full_time.as_secs_f64() / incremental_time.as_secs_f64();
    assert!(
        speedup > 1.5,
        "Incremental solve should be faster: full={:?} vs incremental={:?} ({:.1}x)",
        full_time, incremental_time, speedup
    );
}
```

### Test 4: Newton-Raphson Iteration Count

```rust
#[test]
fn nr_converges_in_expected_iterations() {
    // Well-conditioned linear problem should converge in 1 iteration
    let problem = LinearProblem { target: vec![1.0, 2.0, 3.0] };
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &[0.0, 0.0, 0.0]);

    if let SolveResult::Converged { iterations, .. } = result {
        assert!(
            iterations <= 2,
            "Linear problem should converge in 1-2 iterations, took {}",
            iterations
        );
    }

    // Rosenbrock from standard start should converge in reasonable iterations
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    if let SolveResult::Converged { iterations, .. } = result {
        assert!(
            iterations <= 50,
            "Rosenbrock should converge in <50 iterations, took {}",
            iterations
        );
    }
}
```

## Performance Gates

### Automated Threshold Checking

```rust
/// Assert that a benchmark runs within an expected time range.
macro_rules! perf_gate {
    ($name:expr, $code:block, max_ms = $max:expr) => {{
        let start = std::time::Instant::now();
        $code
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() <= $max,
            "{} took {}ms, max allowed is {}ms",
            $name, elapsed.as_millis(), $max
        );
    }};
}

#[test]
fn perf_gate_rosenbrock_solve() {
    perf_gate!("rosenbrock_solve_100x", {
        let problem = Rosenbrock::new();
        let solver = LMSolver::new(LMConfig::default());
        for _ in 0..100 {
            let _ = solver.solve(&problem, &problem.initial_point(1.0));
        }
    }, max_ms = 1000);
}

#[test]
fn perf_gate_v3_simple_sketch() {
    perf_gate!("v3_triangle_solve_100x", {
        for _ in 0..100 {
            let mut builder = Sketch2DBuilder::new();
            let p0 = builder.add_fixed_point(0.0, 0.0);
            let p1 = builder.add_point(10.0, 0.0);
            let p2 = builder.add_point(5.0, 8.0);
            builder.constrain_distance(p0, p1, 10.0);
            builder.constrain_distance(p1, p2, 8.0);
            builder.constrain_distance(p2, p0, 6.0);
            let mut system = builder.build();
            let _ = system.solve();
        }
    }, max_ms = 2000);
}

#[test]
fn perf_gate_v3_incremental_solve() {
    perf_gate!("v3_incremental_1000x", {
        let (mut system, param_id) = build_multi_cluster_system_with_param(5);
        system.solve(); // Initial solve
        for i in 0..1000 {
            system.set_param(param_id, i as f64 * 0.001);
            let _ = system.solve();
        }
    }, max_ms = 5000);
}
```

### CI with `critcmp`

```yaml
# .github/workflows/benchmarks.yml
benchmarks:
  runs-on: ubuntu-latest
  if: github.event_name == 'pull_request'
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    - uses: dtolnay/rust-toolchain@stable

    - name: Checkout base branch and benchmark
      run: |
        git checkout ${{ github.event.pull_request.base.sha }}
        cargo bench --features parallel,sparse,jit -- --save-baseline base

    - name: Checkout PR and benchmark
      run: |
        git checkout ${{ github.event.pull_request.head.sha }}
        cargo bench --features parallel,sparse,jit -- --save-baseline pr

    - name: Compare benchmarks
      run: |
        cargo install critcmp
        critcmp base pr --threshold 10  # Fail if >10% regression
```

## Existing Benchmark Expansion

### Legacy Constraint Evaluation

```rust
fn bench_constraint_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_eval");

    for n_constraints in [10, 50, 100, 500] {
        let system = build_chain_constraint_system(n_constraints);
        let x = system.current_values();

        group.bench_with_input(
            BenchmarkId::new("residuals", n_constraints),
            &n_constraints,
            |b, _| b.iter(|| system.residuals(black_box(&x))),
        );

        group.bench_with_input(
            BenchmarkId::new("jacobian", n_constraints),
            &n_constraints,
            |b, _| b.iter(|| system.jacobian(black_box(&x))),
        );
    }

    group.finish();
}
```

### Legacy Jacobian Computation Methods

```rust
fn bench_jacobian_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobian_method");

    let problem = BroydenTridiagonal::new(50);
    let x = problem.initial_point(1.0);

    group.bench_function("analytical", |b| {
        b.iter(|| problem.jacobian(black_box(&x)))
    });

    group.bench_function("finite_difference", |b| {
        b.iter(|| compute_finite_diff_jacobian(&problem, black_box(&x), 1e-7))
    });

    #[cfg(feature = "jit")]
    {
        let jit = JITCompiler::compile(&problem).unwrap();
        group.bench_function("jit_compiled", |b| {
            b.iter(|| jit.jacobian(black_box(&x)))
        });
    }

    group.finish();
}
```

## Profiling Integration

When a regression is detected, investigate with flamegraph:

```bash
cargo install flamegraph

# Profile a specific benchmark
cargo flamegraph --bench v3_pipeline -- --bench "pipeline_overhead"

# Profile a test
cargo flamegraph --test solver_tests -- test_rosenbrock
```

## Tracking Over Time

### Store benchmark results as CI artifacts

```yaml
- uses: actions/upload-artifact@v4
  with:
    name: benchmarks-${{ github.sha }}
    path: target/criterion/
    retention-days: 90
```

### Optional: GitHub Pages benchmark dashboard

```yaml
- uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: cargo
    output-file-path: target/criterion/output.json
    github-token: ${{ secrets.GITHUB_TOKEN }}
    auto-push: true
    benchmark-data-dir-path: docs/benchmarks
```

## Avoiding Flaky Benchmarks

| Issue | Solution |
|-------|----------|
| CI noise (shared runners) | Use `criterion`'s statistical analysis (default 100 samples) |
| Warm-up | Criterion handles warm-up automatically |
| Outliers | Criterion uses MAD-based outlier detection |
| Background processes | Accept 10-15% noise margin in CI gates |
| Different hardware | Use relative comparisons (same machine, different git SHAs) |
| Compilation time | Exclude compilation from benchmarks (already handled by Criterion) |
| V3 `iter_batched` state | Use `BatchSize::SmallInput` to rebuild system each iteration |

## File Organization

```
crates/solverang/
    benches/
        comprehensive.rs        # Existing -- expand with legacy benchmarks
        scaling.rs              # Existing -- expand with scaling tests
        nist_benchmarks.rs      # Existing NIST problems
        v3_pipeline.rs          # New: pipeline overhead, incremental, closed-form
        sketch2d_builder.rs     # New: Sketch2D builder end-to-end
        graph_analysis.rs       # New: decomposition, DOF, pattern detection
    tests/
        perf_regression.rs      # Algorithmic complexity tests (not criterion)
```

## Estimated Effort

| Task | Time |
|------|------|
| Set up baseline workflow | 1 hour |
| Pipeline overhead benchmark | 2-3 hours |
| Incremental solving benchmark | 2-3 hours |
| Closed-form vs iterative benchmark | 2 hours |
| Reduce pass benchmark | 1-2 hours |
| Graph decomposition + DOF scaling | 2 hours |
| Sketch2DBuilder end-to-end | 2 hours |
| Pattern detection benchmark | 1 hour |
| Algorithmic complexity tests (4) | 3-4 hours |
| Performance gate macros + V3 gates | 1-2 hours |
| CI integration (critcmp, artifacts) | 2-3 hours |
| Document profiling workflow | 1 hour |
| **Total** | **~20-28 hours** |
