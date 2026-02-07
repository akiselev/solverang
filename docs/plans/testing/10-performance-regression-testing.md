# Plan 10: Performance Regression Testing

## Goal

Solverang already has 3 Criterion benchmark suites. This plan adds:
- **Automated baseline comparison** to detect regressions in PRs
- **Algorithmic complexity tests** that verify O(n) scaling assertions
- **Performance gates** that fail CI when performance degrades beyond a threshold
- **Expanded benchmarks** for geometry, Jacobian computation, and decomposition

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

Criterion prints statistical comparisons:
```
scaling/sparse_50x50    time:   [1.2345 ms 1.2500 ms 1.2678 ms]
                        change: [+2.1% +3.4% +4.8%] (p = 0.01 < 0.05)
                        Performance has regressed.
```

### Install `critcmp` for better comparison

```bash
cargo install critcmp

# Export baselines as JSON
cargo bench --features all -- --save-baseline main
cargo bench --features all -- --save-baseline feature

# Compare
critcmp main feature
```

## Algorithmic Complexity Tests

### Test 1: Sparse solver scales ~O(n), not O(n²)

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

    // Check scaling: time(800) / time(50) should be ~16x for O(n), not ~256x for O(n²)
    let ratio = times.last().unwrap().1 / times.first().unwrap().1;
    let size_ratio = sizes.last().unwrap() / sizes.first().unwrap();

    // Allow up to O(n * log(n)) — ratio up to size_ratio * log2(size_ratio)
    let max_expected_ratio = (size_ratio as f64) * (size_ratio as f64).log2();

    assert!(
        ratio < max_expected_ratio,
        "Sparse solver scaling too steep: {}x time for {}x size (max expected {}x). \
         Times: {:?}",
        ratio, size_ratio, max_expected_ratio, times
    );
}
```

### Test 2: JIT amortization — compilation cost recovered after N evaluations

```rust
#[cfg(feature = "jit")]
#[test]
fn jit_amortizes_over_evaluations() {
    use std::time::Instant;

    let system = build_medium_constraint_system(20); // 20 constraints
    let x = system.current_values();

    // Time interpreted evaluations
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = system.residuals(&x);
        let _ = system.jacobian(&x);
    }
    let interp_time = start.elapsed();

    // Time JIT compilation + evaluations
    let start = Instant::now();
    let jit = JITCompiler::compile(&system).unwrap();
    for _ in 0..1000 {
        let _ = jit.residuals(&x);
        let _ = jit.jacobian(&x);
    }
    let jit_time = start.elapsed();

    // JIT should be faster after 1000 evaluations
    // (or at least not significantly slower)
    let speedup = interp_time.as_secs_f64() / jit_time.as_secs_f64();
    assert!(
        speedup > 0.5,
        "JIT is too slow even after 1000 evals: {:.2}x (expected >= 0.5x). \
         Interpreted: {:?}, JIT: {:?}",
        speedup, interp_time, jit_time
    );
}
```

### Test 3: Parallel solver speedup

```rust
#[cfg(feature = "parallel")]
#[test]
fn parallel_solver_scales_with_subproblems() {
    use std::time::Instant;

    let configs = [(1, "1 sub"), (2, "2 subs"), (4, "4 subs"), (8, "8 subs")];
    let mut times = Vec::new();

    for &(n_groups, label) in &configs {
        let problem = build_block_diagonal_problem(n_groups, 20);
        let x0 = problem.initial_point(1.0);

        let start = Instant::now();
        let result = ParallelSolver::new().solve(&problem, &x0);
        let elapsed = start.elapsed();

        assert!(result.is_converged(), "Failed at {}", label);
        times.push((n_groups, elapsed));
    }

    // Time for 8 sub-problems should be much less than 8x time for 1 sub-problem
    let ratio_8_to_1 = times[3].1.as_secs_f64() / times[0].1.as_secs_f64();

    // On a multi-core machine, expect significant speedup
    // Don't require perfect scaling (overhead exists), but should be < 4x for 8x work
    assert!(
        ratio_8_to_1 < 6.0,
        "No parallel speedup: 8 subs took {:.2}x vs 1 sub. Times: {:?}",
        ratio_8_to_1, times
    );
}
```

### Test 4: Newton-Raphson iteration count

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

### Automated threshold checking

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
    perf_gate!("rosenbrock_solve", {
        let problem = Rosenbrock::new();
        let solver = LMSolver::new(LMConfig::default());
        for _ in 0..100 {
            let _ = solver.solve(&problem, &problem.initial_point(1.0));
        }
    }, max_ms = 1000);
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

## Benchmark Expansion

### Geometry constraint evaluation

```rust
// In benches/comprehensive.rs or new benches/geometry.rs
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

### Jacobian computation methods

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

### Problem decomposition

```rust
fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition");

    for n_groups in [2, 4, 8, 16] {
        let problem = build_block_diagonal_problem(n_groups, 10);

        group.bench_with_input(
            BenchmarkId::new("decompose", n_groups),
            &n_groups,
            |b, _| b.iter(|| decompose(black_box(&problem))),
        );
    }

    group.finish();
}
```

## Profiling Integration

When a regression is detected, investigate with flamegraph:

```bash
cargo install flamegraph

# Profile a specific benchmark
cargo flamegraph --bench comprehensive -- --bench "scaling/sparse_200x200"

# Or profile a test
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

Use `github-action-benchmark` to publish trend charts:

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

## File Organization

```
crates/solverang/
├── benches/
│   ├── comprehensive.rs      # Existing — expand
│   ├── scaling.rs            # Existing — expand
│   ├── nist_benchmarks.rs    # Existing
│   ├── geometry.rs           # New: constraint evaluation benchmarks
│   └── jacobian.rs           # New: Jacobian method comparison
├── tests/
│   └── perf_regression.rs    # Algorithmic complexity tests (not criterion)
```

## Estimated Effort

| Task | Time |
|------|------|
| Set up baseline workflow | 1 hour |
| Write algorithmic complexity tests (4) | 3-4 hours |
| Add new benchmarks (geometry, jacobian, decomposition) | 3-4 hours |
| CI integration (critcmp, artifacts) | 2-3 hours |
| Performance gate macros | 1 hour |
| Document profiling workflow | 1 hour |
| **Total** | **~12-15 hours** |
