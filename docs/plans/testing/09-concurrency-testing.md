# Plan 09: Concurrency & Thread-Safety Testing

## Goal

Solverang has two architectures with different concurrency profiles:

1. **Legacy**: The `parallel` feature uses Rayon for parallel decomposition solving.
   JIT and sparse solvers have caching mechanisms that could have concurrent access issues.
2. **V3**: The entity/constraint/param system uses `Send + Sync` trait bounds on
   `Entity` and `Constraint`, pluggable pipeline phases as trait objects (`Box<dyn Decompose>`
   etc.), and incremental solving with mutable `ChangeTracker` and `SolutionCache`.

This plan verifies thread safety across both architectures, detects data races,
ensures parallel results are deterministic, and stress-tests the V3 pipeline under
concurrent workloads.

## V3 Send + Sync Audit

### Trait Bounds

The V3 architecture mandates `Send + Sync` on core traits:

| Trait | Bound | Location |
|-------|-------|----------|
| `Entity` | `Send + Sync` | `entity/mod.rs` |
| `Constraint` | `Send + Sync` | `constraint/mod.rs` |
| `Decompose` | `Send + Sync` | `pipeline/traits.rs` |
| `Analyze` | `Send + Sync` | `pipeline/traits.rs` |
| `Reduce` | `Send + Sync` | `pipeline/traits.rs` |
| `SolveCluster` | `Send + Sync` | `pipeline/traits.rs` |
| `PostProcess` | `Send + Sync` | `pipeline/traits.rs` |

### Compile-Time Verification Tests

The compiler already enforces these bounds, but explicit tests document the intent
and catch regressions if bounds are accidentally relaxed.

```rust
/// Verify that all sketch2d entity types implement Send + Sync.
#[test]
fn sketch2d_entities_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<sketch2d::Point2D>();
    assert_send_sync::<sketch2d::LineSegment2D>();
    assert_send_sync::<sketch2d::Circle2D>();
    assert_send_sync::<sketch2d::Arc2D>();
    assert_send_sync::<sketch2d::InfiniteLine2D>();
}

/// Verify that all sketch2d constraint types implement Send + Sync.
#[test]
fn sketch2d_constraints_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<sketch2d::DistancePtPt>();
    assert_send_sync::<sketch2d::Coincident>();
    assert_send_sync::<sketch2d::Fixed>();
    assert_send_sync::<sketch2d::Horizontal>();
    assert_send_sync::<sketch2d::Vertical>();
    assert_send_sync::<sketch2d::Parallel>();
    assert_send_sync::<sketch2d::Perpendicular>();
    assert_send_sync::<sketch2d::Angle>();
    assert_send_sync::<sketch2d::Midpoint>();
    assert_send_sync::<sketch2d::Symmetric>();
    assert_send_sync::<sketch2d::EqualLength>();
    assert_send_sync::<sketch2d::PointOnCircle>();
    assert_send_sync::<sketch2d::TangentLineCircle>();
    assert_send_sync::<sketch2d::TangentCircleCircle>();
    assert_send_sync::<sketch2d::DistancePtLine>();
}

/// Verify that all sketch3d types implement Send + Sync.
#[test]
fn sketch3d_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    // Entities
    assert_send_sync::<sketch3d::Point3D>();
    assert_send_sync::<sketch3d::LineSegment3D>();
    assert_send_sync::<sketch3d::Plane>();
    assert_send_sync::<sketch3d::Axis3D>();
    // Constraints
    assert_send_sync::<sketch3d::Distance3D>();
    assert_send_sync::<sketch3d::Coincident3D>();
    assert_send_sync::<sketch3d::Fixed3D>();
    assert_send_sync::<sketch3d::PointOnPlane>();
    assert_send_sync::<sketch3d::Coplanar>();
    assert_send_sync::<sketch3d::Parallel3D>();
    assert_send_sync::<sketch3d::Perpendicular3D>();
    assert_send_sync::<sketch3d::Coaxial>();
}

/// Verify that all assembly types implement Send + Sync.
#[test]
fn assembly_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    // Entities
    assert_send_sync::<assembly::RigidBody>();
    // Constraints
    assert_send_sync::<assembly::UnitQuaternion>();
    assert_send_sync::<assembly::Mate>();
    assert_send_sync::<assembly::CoaxialAssembly>();
    assert_send_sync::<assembly::Insert>();
    assert_send_sync::<assembly::Gear>();
}
```

### Pipeline Phase Trait Objects

The pipeline stores phases as `Box<dyn Decompose>`, `Box<dyn Analyze>`, etc. Since
every phase trait requires `Send + Sync`, the boxed trait objects can be moved across
threads. However, `SolvePipeline` itself stores mutable cached state
(`cached_clusters`, `clusters_valid`) and takes `&mut self` in `run()`, so the
pipeline cannot be shared across threads without external synchronization.

**Verification**: Add a negative-reasoning test that proves `SolvePipeline` is `Send`
(so it can be moved to another thread) but note its `&mut self` requirement prevents
data races at the API level:

```rust
#[test]
fn solve_pipeline_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<pipeline::SolvePipeline>();
}
```

## ConstraintSystem Thread Safety Analysis

### Ownership Model

`ConstraintSystem` requires `&mut self` for all mutating operations:

| Method | Borrows | Notes |
|--------|---------|-------|
| `solve()` | `&mut self` | Takes mutable ref to system and all sub-components |
| `solve_incremental()` | `&mut self` | Delegates to `solve()` |
| `drag()` | `&mut self` | Builds constraint refs, projects displacement |
| `add_entity()` | `&mut self` | Structural mutation |
| `remove_entity()` | `&mut self` | Structural mutation |
| `add_constraint()` | `&mut self` | Structural mutation |
| `remove_constraint()` | `&mut self` | Structural mutation |
| `set_param()` | `&mut self` | Value mutation + dirty tracking |
| `diagnose()` | `&self` | Read-only analysis |
| `analyze_dof()` | `&self` | Read-only analysis |
| `analyze_redundancy()` | `&self` | Read-only analysis |
| `compute_residuals()` | `&self` | Read-only evaluation |

**Key safety property**: Because `solve()` takes `&mut self`, the Rust borrow checker
prevents concurrent reads of the `ConstraintSystem` during solving. This means
`ParamStore` cannot be aliased during a solve -- the pipeline has exclusive access
through the mutable borrow chain.

### Reentrancy Audit

`ConstraintSystem::solve()` calls `self.pipeline.run(...)`, which receives:
- `&self.constraints` (shared ref)
- `&self.entities` (shared ref)
- `&mut self.params` (exclusive ref)
- `&self.config` (shared ref)
- `&mut self.change_tracker` (exclusive ref)
- `&mut self.solution_cache` (exclusive ref)

During the pipeline run, constraint residuals and Jacobians are evaluated by calling
`Constraint::residuals(&self, store: &ParamStore)` and `Constraint::jacobian(&self, store: &ParamStore)`.
The constraints borrow `store` immutably, while the pipeline writes to `store` between
phases (after reduce and after solve). This is safe because borrows are sequential, not
overlapping.

**Test**: Verify that calling `solve()` while holding a shared reference to params
is a compile error (borrow checker enforcement):

```rust
#[test]
fn solve_requires_exclusive_access() {
    // This test verifies the API design prevents concurrent access.
    // It should NOT compile if uncommented:
    //
    // let mut system = ConstraintSystem::new();
    // let params = system.params();  // shared borrow
    // system.solve();                // mutable borrow -- conflict
    //
    // The fact that this doesn't compile is the safety guarantee.
    // This is a documentation test, not a runnable test.
}
```

## ParamStore Concurrent Access Analysis

`ParamStore` has no interior mutability (no `Cell`, `RefCell`, `Mutex`, `RwLock`,
`AtomicXxx`, or `UnsafeCell`). It is a plain `Vec<ParamEntry>` with a `Vec<u32>`
free list.

During a pipeline run:
1. **Decompose phase**: reads `store` (`&ParamStore`).
2. **Analyze phase**: reads `store` (`&ParamStore`).
3. **Reduce phase**: reads `store` (`&ParamStore`), then pipeline writes eliminated params.
4. **Solve phase**: reads `store` (`&ParamStore` for residual/Jacobian evaluation),
   then pipeline writes solution back.
5. **Post-process phase**: no store access.

The pipeline sequences reads and writes. Constraint evaluation (step 4) borrows `store`
immutably while the solver iterates. The solver does not write to `store` during
iteration -- it works on a separate `Vec<f64>` extracted via `extract_free_values()`.
Solutions are written back after the solver returns.

**Verification**: Search for `UnsafeCell`, `Cell`, `RefCell`, `unsafe` in all V3
modules. The V3 architecture has no unsafe code.

```bash
# Interior mutability check (should find zero hits in V3 modules):
rg 'UnsafeCell|Cell<|RefCell|unsafe ' \
  crates/solverang/src/{system,id,param,entity,constraint,dataflow,graph,pipeline,reduce,solve,sketch2d,sketch3d,assembly}.rs \
  crates/solverang/src/{param,dataflow,graph,pipeline,reduce,solve,sketch2d,sketch3d,assembly}/ \
  --type rust
```

## Incremental Solving Race Analysis

### ChangeTracker

`ChangeTracker` stores `HashSet<ParamId>` for dirty params, `HashSet<ClusterId>` for
dirty clusters, and `Vec<EntityId>` / `Vec<ConstraintId>` for structural changes. It
requires `&mut self` for all mutation methods (`mark_param_dirty`, `mark_entity_added`,
etc.) and `&self` for queries.

**Potential race scenario**: If a user modifies the system (calling `set_param()` which
calls `change_tracker.mark_param_dirty()`) while a solve is in progress, both
operations would require `&mut self` on `ConstraintSystem`, which the borrow checker
prevents.

**Verdict**: Safe by construction. No runtime race possible.

### SolutionCache

`SolutionCache` is a `HashMap<ClusterId, ClusterCache>` that requires `&mut self` for
`store()` and `invalidate_all()`, and `&self` for `get()`. The pipeline receives
`&mut SolutionCache` exclusively during `run()`.

**Verdict**: Safe by construction. The pipeline has exclusive access during the entire
run.

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
  --no-default-features --features std \
  -- --test-threads=1 \
  test_empty_system test_add_entity test_solve_single_fix_constraint
```

**Limitation:** Miri doesn't support Cranelift JIT or Rayon (limited thread support).
Use it for single-threaded UB detection on V3 modules only. V3 modules have no unsafe
code, so Miri is primarily useful for verifying that no UB leaks in through
dependencies (nalgebra, etc.).

### 3. Loom (Optional, Advanced)

Systematic concurrency testing that explores all thread interleavings. Only useful
if custom synchronization primitives are added (unlikely with the current `&mut self`
ownership model).

```toml
[dev-dependencies]
loom = "0.7"
```

Only worth adding if the audit reveals lock-free data structures or interior
mutability in future V3 development.

## Stress Tests

### Test 1: Concurrent ConstraintSystem Instances

Separate `ConstraintSystem` instances can be solved on different threads
simultaneously. Each system owns its own `ParamStore`, `SolvePipeline`,
`ChangeTracker`, and `SolutionCache`.

```rust
use std::thread;

#[test]
fn stress_concurrent_constraint_systems() {
    let handles: Vec<_> = (0..50).map(|i| {
        thread::spawn(move || {
            let mut builder = Sketch2DBuilder::new();
            let p0 = builder.add_fixed_point(0.0, 0.0);
            let p1 = builder.add_point(10.0, 0.0);
            let p2 = builder.add_point(5.0, 8.0);
            builder.constrain_distance(p0, p1, 10.0);
            builder.constrain_distance(p1, p2, 8.0 + i as f64 * 0.1);
            builder.constrain_distance(p2, p0, 6.0);
            let mut system = builder.build();
            let result = system.solve();
            assert!(
                matches!(
                    result.status,
                    SystemStatus::Solved | SystemStatus::PartiallySolved
                ),
                "Thread {} failed: {:?}",
                i,
                result.status
            );
        })
    }).collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }
}
```

### Test 2: Concurrent Legacy Solver Invocations

The legacy `Problem`-based solvers are stateless and should work across threads.

```rust
#[test]
fn stress_concurrent_legacy_solves() {
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
        })
    }).collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }
}
```

### Test 3: Rapid Add/Remove/Solve Cycles

Stress-test the generational ID system and free-list reuse under rapid mutation.

```rust
#[test]
fn stress_rapid_add_remove_solve_cycles() {
    let mut system = ConstraintSystem::new();

    for cycle in 0..200 {
        // Add entities and constraints
        let eid = system.alloc_entity_id();
        let px = system.alloc_param(0.0, eid);
        let py = system.alloc_param(0.0, eid);
        system.add_entity(Box::new(TestPoint {
            id: eid,
            params: vec![px, py],
        }));

        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![eid],
            param: px,
            target: cycle as f64,
        }));

        // Solve
        let result = system.solve();
        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Cycle {} failed: {:?}",
            cycle,
            result.status
        );

        // Verify
        assert!(
            (system.get_param(px) - cycle as f64).abs() < 1e-6,
            "Cycle {}: px = {}, expected {}",
            cycle,
            system.get_param(px),
            cycle
        );

        // Remove (frees slots for reuse via generation bump)
        system.remove_constraint(cid);
        system.remove_entity(eid);
    }
}
```

### Test 4: Concurrent Systems with Mixed Plugins (sketch2d, sketch3d, assembly)

```rust
#[test]
fn stress_concurrent_mixed_plugins() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let completed = Arc::new(AtomicUsize::new(0));
    let total_tasks = 60;

    let handles: Vec<_> = (0..total_tasks).map(|i| {
        let completed = completed.clone();
        thread::spawn(move || {
            match i % 3 {
                0 => {
                    // Sketch2D task
                    let mut builder = Sketch2DBuilder::new();
                    let p0 = builder.add_fixed_point(0.0, 0.0);
                    let p1 = builder.add_point(3.0, 0.0);
                    builder.constrain_distance(p0, p1, 5.0);
                    let mut system = builder.build();
                    let _ = system.solve();
                }
                1 => {
                    // Legacy solver task
                    let problem = Rosenbrock::new();
                    let solver = LMSolver::new(LMConfig::default());
                    let _ = solver.solve(&problem, &problem.initial_point(1.0));
                }
                _ => {
                    // Pure V3 system task (manual construction)
                    let mut system = ConstraintSystem::new();
                    let eid = system.alloc_entity_id();
                    let px = system.alloc_param(0.0, eid);
                    system.add_entity(Box::new(TestPoint {
                        id: eid,
                        params: vec![px],
                    }));
                    let cid = system.alloc_constraint_id();
                    system.add_constraint(Box::new(FixValueConstraint {
                        id: cid,
                        entity_ids: vec![eid],
                        param: px,
                        target: i as f64,
                    }));
                    let _ = system.solve();
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

### Test 5: Parallel Solver with Varying Thread Counts (Legacy)

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

## Determinism Testing

### V3 Pipeline Determinism

The V3 pipeline processes clusters sequentially (no internal parallelism). Results
should be perfectly deterministic across runs.

```rust
#[test]
fn v3_pipeline_is_deterministic() {
    for _ in 0..20 {
        let mut builder = Sketch2DBuilder::new();
        let p0 = builder.add_fixed_point(0.0, 0.0);
        let p1 = builder.add_point(10.0, 0.0);
        let p2 = builder.add_point(5.0, 8.0);
        builder.constrain_distance(p0, p1, 10.0);
        builder.constrain_distance(p1, p2, 8.0);
        builder.constrain_distance(p2, p0, 6.0);
        let mut system = builder.build();
        let result = system.solve();

        // Extract solved param values
        // Compare to reference values (first run)
        // All runs must produce bit-identical results
    }
}
```

### Legacy Parallel Solver Determinism

If Rayon's work-stealing causes different floating-point addition order, the results
may differ in the last few bits. Document this.

```rust
#[cfg(feature = "parallel")]
#[test]
fn parallel_solver_approximate_determinism() {
    let problem = build_decomposable_problem();
    let x0 = problem.initial_point(1.0);
    let solver = ParallelSolver::new();

    let results: Vec<_> = (0..10)
        .map(|_| solver.solve(&problem, &x0))
        .collect();

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

The V3 pipeline does not use any locks (no `Mutex`, `RwLock`, channels, etc.).
Deadlocks are only possible in the legacy `parallel` feature via Rayon. Test with
a timeout:

```rust
#[test]
fn v3_solve_completes_promptly() {
    use std::sync::mpsc;
    use std::time::Duration;

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let mut builder = Sketch2DBuilder::new();
        let p0 = builder.add_fixed_point(0.0, 0.0);
        let p1 = builder.add_point(10.0, 0.0);
        builder.constrain_distance(p0, p1, 10.0);
        let mut system = builder.build();
        let result = system.solve();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(result) => {
            assert!(matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ));
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("V3 solve deadlocked (10s timeout exceeded)");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("V3 solve thread panicked");
        }
    }
}
```

## Performance Under Contention

When multiple threads each own their own `ConstraintSystem`, there should be no
shared-state contention (no global locks, no shared allocator bottlenecks beyond
the system allocator).

```rust
#[test]
fn concurrent_systems_scale_linearly() {
    use std::time::Instant;

    // Baseline: solve 10 systems sequentially
    let start = Instant::now();
    for i in 0..10 {
        let mut system = build_test_system(i);
        let _ = system.solve();
    }
    let sequential_time = start.elapsed();

    // Parallel: solve 10 systems on 10 threads
    let start = Instant::now();
    let handles: Vec<_> = (0..10).map(|i| {
        thread::spawn(move || {
            let mut system = build_test_system(i);
            let _ = system.solve();
        })
    }).collect();
    for h in handles {
        h.join().unwrap();
    }
    let parallel_time = start.elapsed();

    // Parallel should not be significantly slower (no contention)
    let ratio = parallel_time.as_secs_f64() / sequential_time.as_secs_f64();
    assert!(
        ratio < 2.0,
        "Parallel should not be 2x slower: {:.2}x ({:?} vs {:?})",
        ratio, parallel_time, sequential_time
    );
}
```

## CI Integration

### ThreadSanitizer in Nightly CI

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

### Miri for V3 Modules (No Unsafe Code Verification)

```yaml
miri:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@nightly
      with:
        components: miri
    - name: Run Miri on V3 unit tests (no JIT, no parallel)
      run: |
        cargo +nightly miri test -p solverang \
          --no-default-features --features std \
          -- --test-threads=1 \
          test_empty_system test_add_entity test_solve_single \
          test_alloc_and_get test_param_id_equality \
          new_tracker_has_no_changes new_cache_is_empty
```

## Summary of Thread Safety Properties

| Component | Thread Safe? | Mechanism |
|-----------|-------------|-----------|
| `ConstraintSystem` | Yes (single-owner) | `&mut self` on all mutations and solves |
| `ParamStore` | Yes (no interior mutability) | Plain `Vec`, no shared state |
| `ChangeTracker` | Yes (single-owner) | `&mut self` on mutations |
| `SolutionCache` | Yes (single-owner) | `&mut self` on mutations |
| `SolvePipeline` | Yes (single-owner) | `&mut self` on `run()` |
| `Entity` impls (sketch2d/3d/assembly) | `Send + Sync` | Trait bound, immutable data |
| `Constraint` impls (sketch2d/3d/assembly) | `Send + Sync` | Trait bound, immutable data |
| Pipeline phase trait objects | `Send + Sync` | Trait bounds on `Decompose`, `Analyze`, etc. |
| Legacy `LMSolver`, `Solver` | `Send + Sync` | Stateless |
| Legacy `ParallelSolver` | `Send + Sync` | Uses Rayon internally |
| Legacy `JITCompiler` | Needs audit | Compiled code cache |
| Legacy `SparseSolver` | Needs audit | Pattern caching |

## File Organization

```
crates/solverang/tests/
    concurrency_tests.rs         # Stress tests, determinism, deadlock detection
    ...
```

## Estimated Effort

| Task | Time |
|------|------|
| V3 Send+Sync compile-time verification tests | 1-2 hours |
| Interior mutability audit (rg search) | 1 hour |
| ConstraintSystem reentrancy analysis (code review) | 2 hours |
| Write stress tests (5 tests) | 4-5 hours |
| Determinism tests (V3 + legacy parallel) | 2 hours |
| Deadlock detection tests | 1 hour |
| Performance under contention test | 1 hour |
| TSan CI integration | 1 hour |
| Miri CI integration for V3 | 1 hour |
| Fix any issues found | 2-4 hours |
| **Total** | **~16-22 hours** |
