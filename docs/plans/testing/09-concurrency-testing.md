# Plan 09: Concurrency & Thread-Safety Testing

## Goal

The `parallel` feature uses Rayon for parallel decomposition solving. The JIT compiler
and sparse solver have caching mechanisms that could have concurrent access issues.
This plan verifies thread safety, detects data races, and ensures parallel results
are deterministic and correct.

## Thread Safety Audit

### Types and Send/Sync Status

| Type | Send | Sync | Notes |
|------|------|------|-------|
| `Solver` / `SolverConfig` | Yes | Yes | Stateless |
| `LMSolver` / `LMConfig` | Yes | Yes | Stateless |
| `AutoSolver` | Yes | Yes | Delegates to stateless solvers |
| `RobustSolver` | Yes | Yes | Delegates |
| `ParallelSolver` | Yes | Yes | Uses Rayon internally |
| `SparseSolver` | Needs audit | Needs audit | Pattern caching? |
| `JITCompiler` | Needs audit | Needs audit | Compiled code cache? |
| `ConstraintSystem` | Yes | Needs audit | Mutable during construction |
| `Problem` (trait) | Not required | Not required | Up to implementor |

### Shared State Locations

| Location | What's Shared | Risk |
|----------|--------------|------|
| JIT compiled code | Function pointers in memory | Use-after-free if JIT context freed |
| Sparse pattern cache | Sparsity pattern struct | Stale pattern if problem changes |
| Rayon thread pool | Global thread pool | Contention, but Rayon handles this |
| Problem reference | `&dyn Problem` passed to parallel sub-solves | Read-only, but interior mutability? |

### Interior Mutability Check

Search for `Cell`, `RefCell`, `Mutex`, `RwLock`, `AtomicXxx`, `UnsafeCell` in the
codebase. Any `Cell`/`RefCell` in a type shared across threads would be unsound.

## Testing Tools

### 1. ThreadSanitizer (Primary)

Detects data races at runtime. Requires nightly Rust.

```bash
RUSTFLAGS="-Z sanitizer=thread" \
  cargo +nightly test -p solverang \
  --features parallel,sparse,jit \
  --target x86_64-unknown-linux-gnu \
  -- --test-threads=1
```

**Note:** `--test-threads=1` for the test harness, but the tests themselves use
Rayon's thread pool internally. TSan instruments all threads.

### 2. Miri (Secondary)

Detects undefined behavior including some concurrency issues. Very slow.

```bash
cargo +nightly miri test -p solverang \
  --features geometry \
  -- --test-threads=1 \
  test_basic_solver  # Run specific tests, not the whole suite
```

**Limitation:** Miri doesn't support Cranelift JIT (can't execute JIT-compiled code)
or Rayon (limited thread support). Use it for single-threaded UB detection only.

### 3. Loom (Optional, Advanced)

Systematic concurrency testing that explores all thread interleavings. Only useful
if solverang has custom synchronization primitives (unlikely with Rayon).

```toml
[dev-dependencies]
loom = "0.7"
```

Only worth adding if the audit reveals custom lock-free data structures.

## Stress Tests

### Test 1: Concurrent Solver Invocations

```rust
use std::thread;

#[test]
fn stress_concurrent_solves() {
    let handles: Vec<_> = (0..100).map(|i| {
        thread::spawn(move || {
            let problem = Rosenbrock::new();
            let solver = LMSolver::new(LMConfig::default());
            let x0 = problem.initial_point(1.0 + i as f64 * 0.01);
            let result = solver.solve(&problem, &x0);
            assert!(
                result.is_converged(),
                "Thread {} failed to converge",
                i
            );
            result
        })
    }).collect();

    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();

    // All should converge
    assert!(results.iter().all(|r| r.is_converged()));
}
```

### Test 2: Parallel Solver with Varying Thread Counts

```rust
#[cfg(feature = "parallel")]
#[test]
fn stress_parallel_solver_thread_counts() {
    let problem = build_decomposable_problem();
    let x0 = problem.initial_point(1.0);

    for num_threads in [1, 2, 4, 8, 0 /* = num_cpus */] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(if num_threads == 0 { num_cpus::get() } else { num_threads })
            .build()
            .unwrap();

        let result = pool.install(|| {
            ParallelSolver::new().solve(&problem, &x0)
        });

        assert!(
            result.is_converged(),
            "Failed with {} threads: {:?}",
            num_threads, result
        );
    }
}
```

### Test 3: Concurrent JIT Compilation

```rust
#[cfg(feature = "jit")]
#[test]
fn stress_concurrent_jit_compilation() {
    let systems: Vec<_> = (0..20).map(|i| {
        let mut system = ConstraintSystem::<2>::new();
        system.add_point(Point2D::new(0.0, 0.0));
        system.add_point(Point2D::new(i as f64 + 1.0, 0.0));
        system.add_constraint(Box::new(
            DistanceConstraint::<2>::new(0, 1, i as f64 + 1.0)
        ));
        system
    }).collect();

    let handles: Vec<_> = systems.into_iter().map(|system| {
        thread::spawn(move || {
            let jit = JITCompiler::compile(&system);
            assert!(jit.is_ok(), "JIT compilation failed: {:?}", jit.err());

            let x = system.current_values();
            let jit = jit.unwrap();
            let residuals = jit.residuals(&x);
            assert_eq!(residuals.len(), system.residual_count());
        })
    }).collect();

    for h in handles {
        h.join().expect("Thread panicked during JIT compilation");
    }
}
```

### Test 4: Mixed Workload

```rust
#[test]
fn stress_mixed_workload() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let completed = Arc::new(AtomicUsize::new(0));
    let total_tasks = 50;

    let handles: Vec<_> = (0..total_tasks).map(|i| {
        let completed = completed.clone();
        thread::spawn(move || {
            match i % 3 {
                0 => {
                    // Solver task
                    let problem = Rosenbrock::new();
                    let solver = LMSolver::new(LMConfig::default());
                    let _ = solver.solve(&problem, &problem.initial_point(1.0));
                }
                1 => {
                    // Evaluation task
                    let problem = Powell::new();
                    for j in 0..100 {
                        let x = vec![j as f64 * 0.1; problem.variable_count()];
                        let _ = problem.residuals(&x);
                        let _ = problem.jacobian(&x);
                    }
                }
                _ => {
                    // Geometry task
                    #[cfg(feature = "geometry")]
                    {
                        let mut system = ConstraintSystem::<2>::new();
                        system.add_point(Point2D::new(0.0, 0.0));
                        system.add_point(Point2D::new(3.0, 0.0));
                        system.add_point(Point2D::new(1.5, 2.0));
                        system.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 1, 3.0)));
                        system.add_constraint(Box::new(DistanceConstraint::<2>::new(1, 2, 2.5)));
                        let x = system.current_values();
                        let solver = LMSolver::new(LMConfig::default());
                        let _ = solver.solve(&system, &x);
                    }
                }
            }
            completed.fetch_add(1, Ordering::SeqCst);
        })
    }).collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::SeqCst), total_tasks);
}
```

## Determinism Testing

### Verify parallel results are reproducible

```rust
#[cfg(feature = "parallel")]
#[test]
fn parallel_solver_deterministic() {
    let problem = build_decomposable_problem();
    let x0 = problem.initial_point(1.0);
    let solver = ParallelSolver::new();

    let results: Vec<_> = (0..10)
        .map(|_| solver.solve(&problem, &x0))
        .collect();

    // All runs should produce identical results
    let first = format!("{:?}", results[0]);
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first,
            format!("{:?}", result),
            "Non-deterministic on run {}",
            i
        );
    }
}
```

### Document acceptable non-determinism

If Rayon's work-stealing causes different floating-point addition order, the results
may differ in the last few bits. Document this:

```rust
#[cfg(feature = "parallel")]
#[test]
fn parallel_solver_approximate_determinism() {
    // ... (same setup) ...

    // Allow ULP-level differences due to FP addition order
    if let (
        SolveResult::Converged { solution: s1, residual_norm: r1, .. },
        SolveResult::Converged { solution: s2, residual_norm: r2, .. },
    ) = (&results[0], &results[1]) {
        assert!((r1 - r2).abs() < 1e-12, "Residual norms differ: {} vs {}", r1, r2);
        for (a, b) in s1.iter().zip(s2) {
            assert!((a - b).abs() < 1e-10, "Solutions differ: {} vs {}", a, b);
        }
    }
}
```

## Deadlock Detection

```rust
#[test]
fn parallel_solver_no_deadlock() {
    use std::time::Duration;

    let problem = build_decomposable_problem();
    let x0 = problem.initial_point(1.0);

    let handle = thread::spawn(move || {
        ParallelSolver::new().solve(&problem, &x0)
    });

    // If solver hangs for more than 30 seconds, it's likely deadlocked
    match handle.join() {
        Ok(result) => assert!(result.is_converged()),
        Err(e) => panic!("Solver panicked: {:?}", e),
    }
    // Note: std::thread::JoinHandle doesn't support timeout.
    // For true deadlock detection, use a channel with timeout:
}

#[test]
fn parallel_solver_timeout_detection() {
    use std::sync::mpsc;
    use std::time::Duration;

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let problem = build_decomposable_problem();
        let x0 = problem.initial_point(1.0);
        let result = ParallelSolver::new().solve(&problem, &x0);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(result) => assert!(result.is_converged()),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("Solver deadlocked (30s timeout exceeded)");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("Solver thread panicked");
        }
    }
}
```

## Performance Under Contention

```rust
#[cfg(feature = "parallel")]
#[test]
fn parallel_speedup_is_real() {
    use std::time::Instant;

    let problem = build_large_decomposable_problem(8); // 8 sub-problems
    let x0 = problem.initial_point(1.0);

    // Sequential baseline
    let start = Instant::now();
    let seq_result = LMSolver::new(LMConfig::default()).solve(&problem, &x0);
    let seq_time = start.elapsed();

    // Parallel
    let start = Instant::now();
    let par_result = ParallelSolver::new().solve(&problem, &x0);
    let par_time = start.elapsed();

    assert!(seq_result.is_converged());
    assert!(par_result.is_converged());

    // Parallel should not be significantly SLOWER than sequential
    // (We don't require speedup, just no regression)
    let slowdown = par_time.as_secs_f64() / seq_time.as_secs_f64();
    assert!(
        slowdown < 3.0,
        "Parallel solver is {}x slower than sequential ({:?} vs {:?})",
        slowdown, par_time, seq_time
    );
}
```

## CI Integration

### ThreadSanitizer in nightly CI

```yaml
# .github/workflows/nightly.yml
thread-safety:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@nightly
      with:
        components: rust-src
    - name: Run tests with ThreadSanitizer
      env:
        RUSTFLAGS: "-Z sanitizer=thread"
      run: |
        cargo +nightly test -p solverang \
          --features parallel,sparse \
          --target x86_64-unknown-linux-gnu \
          -- --test-threads=1 \
          stress_ concurrent_ parallel_
```

### Miri for unsafe code

```yaml
miri:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@nightly
      with:
        components: miri
    - name: Run Miri on unit tests (no JIT, no parallel)
      run: |
        cargo +nightly miri test -p solverang \
          --no-default-features --features std,geometry \
          -- test_basic test_convergence test_residual
```

## File Organization

```
crates/solverang/tests/
├── concurrency_tests.rs         # Stress tests, determinism, deadlock detection
└── ...
```

## Estimated Effort

| Task | Time |
|------|------|
| Thread safety audit (code review) | 2-3 hours |
| Write stress tests (4 tests) | 3-4 hours |
| Determinism tests | 1-2 hours |
| Deadlock detection tests | 1 hour |
| Performance under contention | 1 hour |
| TSan CI integration | 1 hour |
| Miri CI integration | 1 hour |
| Fix any issues found | 2-4 hours |
| **Total** | **~12-18 hours** |
